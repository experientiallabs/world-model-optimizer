"""Distilled reply verifier vs the free selector: can a cheap scorer harvest the best-of-2 gap?

The pre-registered question (master, 2026-07-24). Free post-hoc features pick the better of two
rollouts ~66-68% of the time on decisive cells and harvest only part of the oracle-of-2 gap, and
`bo2-free` consequently declines under a 2x-cost guard. A scorer distilled from the pinned judge
(Zooter, arXiv 2311.08692) reads the reply TEXT, so it could do better. Two pre-registered bars:

  1. within-cell correct-fraction on decisive TEST cells must beat the free selector, and
  2. selected-of-2 harvest must exceed 60% of the oracle-of-2 gap.

Failing both is a real finding: the remaining gap is not harvestable without a stronger judge.

Design notes that matter for reading the output:
- Training is POOLED over all wm corpora (one verifier), because a per-corpus fit split is ~17
  scenarios against a 3072-dim embedding. Splits are `split_scenario_ids(0.7, seed)` on the same
  pooled wm-all matrix the ablation grid uses, so rows are comparable to the grid's wm-all rows.
- Two heads: `absolute` (regress reply -> reward, the literal distillation) and `pairwise`
  (regress embedding difference -> reward difference within a cell). See
  wmo/research/reply_verifier.py for why the second is expected to win at selection.
- `shuffled` is the control: same pipeline, permuted rewards. It must collapse to chance.
- Embeddings are text-embedding-3-large via the Azure google-sheets deployment, cached to disk by
  reply-text sha256, so the first run costs cents and reruns cost nothing.

Usage:
    uv run .agents/scripts/fit_reply_verifier.py [--seeds 0,1,2,3,4] [--embed-only]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from wmo.config import load_env_file
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.policy import EmbedderSpec
from wmo.research.posthoc_bounds import (
    DEFAULT_SELECTOR,
    SELECTOR_KEYS,
    pooled_correct_z,
    scored_cells,
    selector_bound,
)
from wmo.research.reply_verifier import (
    EpisodeKey,
    Projection,
    ReplyVerifier,
    episode_key,
    fit_absolute,
    fit_pairwise,
    fit_projection,
    pairwise_design,
    scenario_folds,
    shuffled_rewards,
    verifier_selector,
)
from wmo.research.routing_corpus import routing_data

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("reply-verifier")

WM = Path(".wmo/evals/wm")
CACHE = routing_data() / "cache/wm-oai3l-replies.npz"
SPEC = EmbedderSpec(
    kind="azure",
    dim=3072,
    deployment="text-embedding-3-large",
    endpoint="https://google-sheets.openai.azure.com",
    api_key_env="AZURE_GOOGLE_SHEETS_API_KEY",
    batch=256,
)
# 3-large accepts 8191 tokens; the longest wm rollout is ~5.9k, so this only guards outliers.
MAX_CHARS = 28_000
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10_000.0]
EMBED_CHUNK = 64  # small enough that one 429 costs little; the cache is flushed after each chunk
EMBED_RETRIES = 8
PCA_COMPONENTS = 64  # pairwise design is ~600 rows; 3072 raw dims leaves ridge nothing to work with


def reply_text(outcome: ScenarioOutcome) -> str:
    """The whole rollout transcript, not just the final message.

    For agentic corpora the informative signal is the trajectory (how many steps it took, whether
    it recovered from errors), so all calls are concatenated rather than keeping the last reply.
    """
    return "\n\n".join(outcome.replies)[:MAX_CHARS]


def pooled_wm_matrix() -> OutcomeMatrix:
    """All wm corpora as one matrix with corpus-prefixed scenario ids (the grid's wm-all)."""
    parts = [
        (p.stem.removesuffix("_matrix"), OutcomeMatrix.load(p))
        for p in sorted(WM.glob("*_matrix.json"))
    ]
    if not parts:
        raise SystemExit(f"no *_matrix.json under {WM}")
    combined = [
        outcome.model_copy(update={"scenario_id": f"{corpus}:{outcome.scenario_id}"})
        for corpus, matrix in parts
        for outcome in matrix.outcomes
    ]
    logger.info("pooled %d corpora -> %d outcomes", len(parts), len(combined))
    return OutcomeMatrix(pool=parts[0][1].pool, outcomes=combined)


def load_cache() -> dict[str, np.ndarray]:
    if not CACHE.exists():
        return {}
    blob = np.load(CACHE, allow_pickle=False)
    return dict(zip(blob["hashes"].tolist(), blob["vectors"], strict=True))


def save_cache(cache: dict[str, np.ndarray]) -> None:
    """Write the cache ATOMICALLY: temp file in the same dir, then replace.

    The flush happens after every chunk, so a plain in-place `savez` leaves a truncated zip behind
    if anything interrupts it - and a reader (or a second copy of this script) then sees
    BadZipFile. Learned the hard way: running the analysis pass while an --embed-only pass was
    still going corrupted this file, because both wrote the same path.
    """
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(cache)
    temporary = CACHE.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        hashes=np.asarray(ordered),
        vectors=np.asarray([cache[h] for h in ordered], dtype=np.float32),
    )
    temporary.replace(CACHE)


def embed_replies(matrix: OutcomeMatrix) -> dict[EpisodeKey, np.ndarray]:
    """Embedding per episode, reusing the disk cache and embedding only unseen reply texts."""
    texts = {
        episode_key(o): reply_text(o) for o in matrix.outcomes if o.reward is not None and o.replies
    }
    digests = {key: hashlib.sha256(text.encode()).hexdigest() for key, text in texts.items()}
    cache = load_cache()
    missing = sorted({digests[key] for key in texts} - set(cache))
    logger.info(
        "%d episodes with replies, %d unique texts, %d already cached, %d to embed",
        len(texts),
        len(set(digests.values())),
        len(set(digests.values())) - len(missing),
        len(missing),
    )
    if missing:
        by_digest = {digests[key]: text for key, text in texts.items()}
        embedder = SPEC.build()
        # The google-sheets deployment is rate limited (S0 tier), so embed in small chunks with
        # backoff and FLUSH THE CACHE AFTER EACH ONE: a 429 mid-run then costs only the current
        # chunk, and rerunning resumes instead of re-paying for everything already embedded.
        for start in range(0, len(missing), EMBED_CHUNK):
            chunk = missing[start : start + EMBED_CHUNK]
            for attempt in range(EMBED_RETRIES):
                try:
                    vectors = embedder.embed([by_digest[d] for d in chunk])
                    break
                except Exception as error:  # noqa: BLE001 - any provider error is worth retrying
                    if attempt == EMBED_RETRIES - 1:
                        save_cache(cache)
                        raise
                    wait = 65.0 if "429" in str(error) or "RateLimit" in str(error) else 5.0
                    logger.info(
                        "  chunk %d-%d failed (%s), retry %d/%d in %.0fs",
                        start,
                        start + len(chunk),
                        type(error).__name__,
                        attempt + 1,
                        EMBED_RETRIES - 1,
                        wait,
                    )
                    time.sleep(wait)
            for digest, vector in zip(chunk, vectors, strict=True):
                cache[digest] = np.asarray(vector, dtype=np.float32)
            save_cache(cache)
            logger.info(
                "  embedded %d/%d unique texts",
                min(start + EMBED_CHUNK, len(missing)),
                len(missing),
            )
        logger.info("cached %d new embeddings -> %s", len(missing), CACHE)
    return {key: cache[digests[key]] for key in texts if digests[key] in cache}


def shared_denominator_credit(
    cells: list[list[ScenarioOutcome]],
    key: Callable[[ScenarioOutcome], tuple[float, ...]],
) -> tuple[float, int]:
    """Selection credit over cells whose REWARDS differ, with a shared denominator.

    `selector_bound.correct_fraction` calls a cell decisive only when THAT selector ranks the two
    episodes, so a coarse feature (finished-then-more-steps ties whenever step counts match) is
    scored over far fewer cells than a continuous score. Measured that way the free selector got
    133 cells and the verifier heads 211 - different denominators, so the two percentages were not
    comparable.

    Here a cell counts whenever its rewards differ, which is a property of the DATA, not of the
    selector. A selector that cannot distinguish the episodes scores 0.5, the expected value of the
    coin flip it is reduced to. Returns (credit fraction, cells).
    """
    credit = 0.0
    total = 0
    for episodes in cells:
        rewards = [o.reward for o in episodes if o.reward is not None]
        if len(rewards) < 2 or max(rewards) == min(rewards):
            continue
        total += 1
        keys = [key(o) for o in episodes]
        best = min(keys)
        picked = [r for r, k in zip(rewards, keys, strict=True) if k == best]
        if len(picked) == len(rewards):
            credit += 0.5  # cannot distinguish: a tie is worth a coin flip, not a win
        elif max(picked) == max(rewards):
            credit += 1.0
    return (credit / total if total else 0.0), total


def within_cell_accuracy(
    cells: list[list[ScenarioOutcome]],
    vectors: dict[EpisodeKey, np.ndarray],
    verifier: ReplyVerifier,
) -> float:
    """Fraction of decisive cells where the verifier's top-scored episode is a best one."""
    key = verifier_selector(verifier, vectors)
    decisive = correct = 0
    for episodes in cells:
        rewards = [o.reward for o in episodes if o.reward is not None]
        if len(rewards) < 2 or max(rewards) == min(rewards):
            continue
        keys = [key(o) for o in episodes]
        if len(set(keys)) < 2:
            continue
        decisive += 1
        best = min(keys)
        picked = [o.reward for o, k in zip(episodes, keys, strict=True) if k == best]
        if max(picked) == max(rewards):  # type: ignore[type-var]
            correct += 1
    return correct / decisive if decisive else 0.0


def choose_alpha(
    mode: str,
    fit_cells: list[list[ScenarioOutcome]],
    vectors: dict[EpisodeKey, np.ndarray],
    seed: int,
) -> float:
    """Inner CV on the fit split, folding by SCENARIO, scored on within-cell selection.

    Selecting alpha on the metric we report, using only fit-split data - the test split is never
    consulted for any choice.
    """
    scenarios = [episodes[0].scenario_id for episodes in fit_cells]
    folds = scenario_folds(scenarios, folds=5, seed=seed)
    # Split each fold and fit its PCA basis ONCE. The basis depends only on the training fold, not
    # on alpha, and an SVD of ~2800x3072 per (alpha, fold) dominated the runtime when this lived
    # inside the alpha loop.
    prepared = []
    for held in folds:
        held_set = set(held)
        train = [c for c in fit_cells if c[0].scenario_id not in held_set]
        valid = [c for c in fit_cells if c[0].scenario_id in held_set]
        if train and valid:
            prepared.append((train, valid, _projection_for(mode, train, vectors)))

    best_alpha, best_score = ALPHAS[0], -1.0
    for alpha in ALPHAS:
        scores = []
        for train, valid, projection in prepared:
            verifier = _fit(mode, train, vectors, alpha, projection=projection)
            if verifier is not None:
                scores.append(within_cell_accuracy(valid, vectors, verifier))
        mean = float(np.mean(scores)) if scores else 0.0
        logger.info("    alpha=%-8g cv within-cell accuracy %.4f", alpha, mean)
        if mean > best_score:
            best_alpha, best_score = alpha, mean
    return best_alpha


def _projection_for(
    mode: str,
    cells: list[list[ScenarioOutcome]],
    vectors: dict[EpisodeKey, np.ndarray],
) -> Projection | None:
    """The PCA basis for a `-pca` head, from the TRAINING cells' embeddings only.

    Deriving it from all embeddings would leak test-split representation structure even though no
    test label is touched.
    """
    if not mode.endswith("-pca"):
        return None
    fit_vectors = [
        vectors[episode_key(o)] for episodes in cells for o in episodes if episode_key(o) in vectors
    ]
    if not fit_vectors:
        return None
    return fit_projection(np.asarray(fit_vectors, dtype=float), PCA_COMPONENTS)


def _fit(
    mode: str,
    cells: list[list[ScenarioOutcome]],
    vectors: dict[EpisodeKey, np.ndarray],
    alpha: float,
    *,
    shuffle_seed: int | None = None,
    projection: Projection | None = None,
) -> ReplyVerifier | None:
    """Fit one head. `mode` may carry a `-pca` suffix, meaning it lives in the projected basis.

    `projection` is passed in so callers can reuse one basis across an alpha sweep; when omitted it
    is derived from `cells`.
    """
    base = mode[: -len("-pca")] if mode.endswith("-pca") else mode
    if projection is None:
        projection = _projection_for(mode, cells, vectors)
    if base == "pairwise":
        differences, gaps = pairwise_design(cells, vectors)
        if not len(differences):
            return None
        if shuffle_seed is not None:
            gaps = shuffled_rewards(gaps, shuffle_seed)
        return fit_pairwise(differences, gaps, alpha=alpha, projection=projection)
    rows, rewards = [], []
    for episodes in cells:
        for outcome in episodes:
            if outcome.reward is not None and episode_key(outcome) in vectors:
                rows.append(vectors[episode_key(outcome)])
                rewards.append(outcome.reward)
    if not rows:
        return None
    targets = np.asarray(rewards, dtype=float)
    if shuffle_seed is not None:
        targets = shuffled_rewards(targets, shuffle_seed)
    return fit_absolute(np.asarray(rows, dtype=float), targets, alpha=alpha, projection=projection)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--embed-only", action="store_true", help="populate the cache and stop")
    parser.add_argument(
        "--embedder",
        choices=("azure", "hashing"),
        default="azure",
        help="hashing is offline/free (trigram lexical) - a pipeline check and a lexical baseline",
    )
    args = parser.parse_args()
    load_env_file()

    matrix = pooled_wm_matrix()
    if args.embedder == "hashing":
        # Offline lexical baseline: needs no credentials and no rate limit, so it validates every
        # code path AND answers whether a purely lexical view of the reply already beats the free
        # features. Semantic embeddings only earn their cost if they beat this.
        from wmo.retrieval.embedders import HashingEmbedder

        embedder = HashingEmbedder(dim=1024)
        keys = [episode_key(o) for o in matrix.outcomes if o.reward is not None and o.replies]
        raw = embedder.embed(
            [reply_text(o) for o in matrix.outcomes if o.reward is not None and o.replies]
        )
        vectors = {
            key: np.asarray(vec, dtype=np.float32) for key, vec in zip(keys, raw, strict=True)
        }
        logger.info("hashing-1024 embeddings for %d episodes (offline)", len(vectors))
    else:
        vectors = embed_replies(matrix)
    if args.embed_only:
        return

    from wmo.research.routerbench import split_scenario_ids

    cells_by_scenario: dict[str, list[list[ScenarioOutcome]]] = {}
    for (sid, _model), episodes in scored_cells(matrix).items():
        if len(episodes) >= 2:
            cells_by_scenario.setdefault(sid, []).append(episodes)

    heads = ("absolute", "absolute-pca", "pairwise", "pairwise-pca")
    collected: dict[str, list] = {"free": [], "shuffled": [], **{h: [] for h in heads}}
    per_seed: list[dict[str, float]] = []
    for seed in (int(s) for s in args.seeds.split(",")):
        fit_ids, test_ids = split_scenario_ids(matrix, train_fraction=0.7, seed=seed)
        fit_cells = [c for sid in fit_ids for c in cells_by_scenario.get(sid, [])]
        test_set = set(test_ids)
        logger.info(
            "\nseed %d: %d fit scenarios (%d cells), %d test scenarios",
            seed,
            len(fit_ids),
            len(fit_cells),
            len(test_ids),
        )

        def in_test(episodes, _test_set: set[str] = test_set) -> bool:  # noqa: ANN001
            # Every selector is scored on the SAME cells: test-split, and fully embedded. Without
            # the coverage clause an episode with no embedding scores 0.0, ties with its sibling,
            # and drops out of the verifier's decisive set but not the free selector's - so the
            # two correct-fractions would be computed over different denominators.
            return episodes[0].scenario_id in _test_set and all(
                episode_key(o) in vectors for o in episodes
            )

        test_cells = [c for sid in test_ids for c in cells_by_scenario.get(sid, [])]
        test_cells = [c for c in test_cells if all(episode_key(o) in vectors for o in c)]
        row: dict[str, float] = {"seed": seed}
        row["free_credit"], row["cells"] = shared_denominator_credit(
            test_cells, SELECTOR_KEYS[DEFAULT_SELECTOR]
        )
        bounds = selector_bound(
            matrix,
            "wm-all",
            DEFAULT_SELECTOR,
            SELECTOR_KEYS[DEFAULT_SELECTOR],
            cell_filter=in_test,
        )
        collected["free"].append(bounds)
        row["free_correct"], row["free_harvest"] = (
            bounds.correct_fraction,
            bounds.harvested_fraction,
        )

        for mode in heads:
            logger.info("  %s head: alpha sweep", mode)
            alpha = choose_alpha(mode, fit_cells, vectors, seed)
            projection = _projection_for(mode, fit_cells, vectors)
            verifier = _fit(mode, fit_cells, vectors, alpha, projection=projection)
            if verifier is None:
                continue
            logger.info("  %s head: alpha=%g on %d train rows", mode, alpha, verifier.train_rows)
            bound = selector_bound(
                matrix, "wm-all", mode, verifier_selector(verifier, vectors), cell_filter=in_test
            )
            collected[mode].append(bound)
            row[f"{mode}_credit"], _ = shared_denominator_credit(
                test_cells, verifier_selector(verifier, vectors)
            )
            row[f"{mode}_correct"] = bound.correct_fraction
            row[f"{mode}_harvest"] = bound.harvested_fraction
            if mode == "pairwise-pca":
                control = _fit(
                    mode,
                    fit_cells,
                    vectors,
                    alpha,
                    shuffle_seed=1000 + seed,
                    projection=projection,
                )
                if control is not None:
                    shuffled = selector_bound(
                        matrix,
                        "wm-all",
                        "shuffled",
                        verifier_selector(control, vectors),
                        cell_filter=in_test,
                    )
                    collected["shuffled"].append(shuffled)
                    row["shuffled_credit"], _ = shared_denominator_credit(
                        test_cells, verifier_selector(control, vectors)
                    )
                    row["shuffled_correct"] = shuffled.correct_fraction
        per_seed.append(row)

    logger.info("\n%s", "=" * 100)
    logger.info("SHARED-DENOMINATOR SELECTION CREDIT on identical reward-decisive test cells")
    logger.info("(a selector that cannot rank the two episodes scores 0.5, not 0)")
    logger.info("%s", "=" * 100)
    names = ("free", *heads, "shuffled")
    logger.info("%-6s %6s " + " ".join(f"{n:>14s}" for n in names), "seed", "cells")
    for row in per_seed:
        logger.info(
            "%-6d %6d " + " ".join("%14.4f" for _ in names),
            int(row["seed"]),
            int(row.get("cells", 0)),
            *[row.get(f"{n}_credit", float("nan")) for n in names],
        )
    logger.info("%s", "-" * 100)
    for name in names:
        values = [row[f"{name}_credit"] for row in per_seed if f"{name}_credit" in row]
        if values:
            logger.info(
                "MEAN   %-14s %.4f +- %.4f",
                name,
                float(np.mean(values)),
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            )
    for mode in heads:
        deltas = [
            row[f"{mode}_credit"] - row["free_credit"]
            for row in per_seed
            if f"{mode}_credit" in row
        ]
        if deltas:
            logger.info(
                "PAIRED %-14s vs free: %+.4f mean (%s), %d/%d seeds better",
                mode,
                float(np.mean(deltas)),
                ", ".join(f"{d:+.3f}" for d in deltas),
                sum(1 for d in deltas if d > 0),
                len(deltas),
            )
    logger.info("\n%s", "=" * 100)
    logger.info("HARVEST of the oracle-of-2 gap (selector_bound, per-selector decisive sets)")
    logger.info("%s", "=" * 100)
    for name in names:
        rows = collected[name]
        if rows:
            logger.info(
                "  %-14s mean harvest %+.0f%% of oracle | correct-fraction %.3f on its own %d cells",
                name,
                float(np.mean([r.harvested_fraction for r in rows])) * 100,
                float(np.mean([r.correct_fraction for r in rows])),
                int(np.mean([r.decisive_cells for r in rows])),
            )
    logger.info(
        "\nPRE-REGISTERED BARS: beat free on selection, AND exceed 60%% harvest.\n"
        "The SHUFFLED control is the one to read first: it is trained on permuted rewards, so\n"
        "anything it scores above 0.5 is signal the embedding carries WITHOUT labels - reply length\n"
        "is in the embedding, and length is a real within-cell quality cue (steps +0.31, replies\n"
        "+0.36 within-cell). A head that only matches the shuffled control has learned nothing\n"
        "from the judge."
    )


if __name__ == "__main__":
    main()
