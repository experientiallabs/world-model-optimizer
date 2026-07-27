"""The cascade: cheap model answers, the distilled verifier decides, escalate when in doubt.

Pre-registered in findings/r2.md (2026-07-25) BEFORE this ran; bars, selection rules, and
permutation-averaging are copied from there, not invented here. Economics: cost = 1x cheap +
escalation_rate x strong, against best-of-2's fatal flat 2x.

Arms per (matrix, seed):
- r2-oracle-cascade: the ceiling. Reads the cheap episode's TRUE reward at decision time
  (labeled oracle-* per the evaluate_call_sequences information boundary; never deployable).
- r2-cascade: deployable. The absolute-head reply verifier (master's validated recipe:
  full-dim ridge alpha=1 over 3-large reply embeddings, trained on the FIT split only)
  scores the cheap reply; escalate below a threshold chosen fit-side out-of-fold.
- r2-cascade-shuffled (seed 0): control; the verifier trained on permuted rewards must not
  produce a real gain.

All cascade rows are the mean over the 2 episode-order permutations (the order-dependence
trap); the best-single baseline is an evaluate_choices row (master a51f917f). Fit outputs
(chosen pair, threshold, escalation rate) go in NOTES, never in params (cohort ruling #1).

Usage: uv run python .agents/scripts/r2_cascade.py [wm-all ...] [--seeds=0,1,2,3,4]
       [--oracle-only]
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.research.reply_verifier import (
    EpisodeKey,
    ReplyVerifier,
    episode_key,
    fit_absolute,
    scenario_folds,
    shuffled_rewards,
)
from wmo.research.routing_runs import (
    Finish,
    RunRecord,
    append_run,
    evaluate_call_sequences,
    evaluate_choices,
)
from wmo.research.routing_corpus import routing_data

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("r2cascade")

DATA = routing_data()
RUNS = DATA / "runs" / "r2.jsonl"
CACHE = DATA / "cache" / "wm-oai3l-replies.npz"
ALPHA = 1.0  # master's validated verifier recipe: full-dim ridge, alpha=1
# Round 8b selection constraints (pre-registered in findings/r2.md before running):
CAP_FACTOR = 0.85  # fit cost cap safety margin against fit->test cost drift (+8-15%)
CHEAP_RATIO = 0.6  # a real cascade: cheap must cost <= this fraction of strong
FOLDS = 5
SEEDS = [0, 1, 2, 3, 4]
MAX_CHARS = 28_000  # MUST match fit_reply_verifier.reply_text or cache hashes miss


def _driver():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location(
        "r2drv", Path(__file__).parent / "run_routing_r2.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reply_text(outcome: ScenarioOutcome) -> str:
    return "\n\n".join(outcome.replies)[:MAX_CHARS]


def load_embeddings(matrix: OutcomeMatrix) -> dict[EpisodeKey, np.ndarray]:
    """Episode -> cached 3-large reply embedding (sha256-of-text keyed, master's cache)."""
    blob = np.load(CACHE, allow_pickle=False)
    cache = dict(zip(blob["hashes"].tolist(), blob["vectors"], strict=True))
    out: dict[EpisodeKey, np.ndarray] = {}
    missing = 0
    for outcome in matrix.outcomes:
        if outcome.reward is None or not outcome.replies:
            continue
        digest = hashlib.sha256(reply_text(outcome).encode()).hexdigest()
        vector = cache.get(digest)
        if vector is None:
            missing += 1
            continue
        out[episode_key(outcome)] = np.asarray(vector, dtype=np.float64)
    logger.info(
        "embeddings: %d episodes covered, %d scored-with-reply missing from cache",
        len(out),
        missing,
    )
    return out


def _cells(matrix: OutcomeMatrix, ids: set[str]) -> dict[tuple[str, str], list[ScenarioOutcome]]:
    cells: dict[tuple[str, str], list[ScenarioOutcome]] = {}
    for outcome in matrix.outcomes:
        if outcome.scenario_id in ids and outcome.reward is not None:
            cells.setdefault((outcome.scenario_id, outcome.model), []).append(outcome)
    return cells


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _fit_stats(cells: dict) -> tuple[dict, dict]:
    rewards = {key: _mean([o.reward for o in episodes]) for key, episodes in cells.items()}
    costs = {key: _mean([o.cost_usd for o in episodes]) for key, episodes in cells.items()}
    return rewards, costs


def _gain_targets(
    matrix: OutcomeMatrix, fit_ids: list[str], cheap: str, strong: str
) -> dict[EpisodeKey, float]:
    """8d target per cheap fit episode: strong cell-mean reward minus this episode's reward."""
    cells = _cells(matrix, set(fit_ids))
    strong_mean = {
        sid: _mean([o.reward for o in cells[(sid, strong)]])
        for sid in fit_ids
        if (sid, strong) in cells
    }
    out: dict[EpisodeKey, float] = {}
    for sid in fit_ids:
        for outcome in cells.get((sid, cheap), []):
            if sid in strong_mean:
                out[episode_key(outcome)] = strong_mean[sid] - outcome.reward
    return out


def _oof_gain_scores(
    matrix: OutcomeMatrix,
    fit_ids: list[str],
    embeddings: dict[EpisodeKey, np.ndarray],
    seed: int,
    cheap: str,
    strong: str,
) -> tuple[dict[EpisodeKey, float], float]:
    """OOF NEGATED predicted gains for cheap fit episodes + spearman-ish sanity corr."""
    targets = _gain_targets(matrix, fit_ids, cheap, strong)
    keyed = [(key, value) for key, value in targets.items() if key in embeddings]
    scores: dict[EpisodeKey, float] = {}
    for fold in scenario_folds(fit_ids, FOLDS, seed):
        fold_set = set(fold)
        train = [(k, v) for k, v in keyed if k[0] not in fold_set]
        held = [(k, v) for k, v in keyed if k[0] in fold_set]
        if not train or not held:
            continue
        features = np.asarray([embeddings[k] for k, _v in train])
        verifier = fit_absolute(
            features, np.asarray([v for _k, v in train], dtype=float), alpha=ALPHA
        )
        held_features = np.asarray([embeddings[k] for k, _v in held])
        for (key, _v), score in zip(held, verifier.score(held_features), strict=True):
            scores[key] = -float(score)  # NEGATED: low score = high gain = escalate
    predicted = np.asarray([-scores[k] for k, _v in keyed if k in scores])
    actual = np.asarray([v for k, v in keyed if k in scores])
    corr = 0.0
    if len(predicted) > 2 and predicted.std() > 0 and actual.std() > 0:
        corr = float(np.corrcoef(predicted, actual)[0, 1])
    return scores, corr


def _oof_scores(
    matrix: OutcomeMatrix,
    fit_ids: list[str],
    embeddings: dict[EpisodeKey, np.ndarray],
    seed: int,
    *,
    shuffle: bool = False,
) -> dict[EpisodeKey, float]:
    """Out-of-fold verifier scores for every embedded FIT episode (no self-scoring)."""
    episodes = [
        o
        for o in matrix.outcomes
        if o.scenario_id in set(fit_ids) and o.reward is not None and episode_key(o) in embeddings
    ]
    scores: dict[EpisodeKey, float] = {}
    folds = scenario_folds(fit_ids, FOLDS, seed)
    for fold in folds:
        fold_set = set(fold)
        train = [o for o in episodes if o.scenario_id not in fold_set]
        held = [o for o in episodes if o.scenario_id in fold_set]
        if not train or not held:
            continue
        features = np.asarray([embeddings[episode_key(o)] for o in train])
        rewards = np.asarray([o.reward for o in train], dtype=float)
        if shuffle:
            rewards = shuffled_rewards(rewards, seed)
        verifier = fit_absolute(features, rewards, alpha=ALPHA)
        held_features = np.asarray([embeddings[episode_key(o)] for o in held])
        for outcome, score in zip(held, verifier.score(held_features), strict=True):
            scores[episode_key(outcome)] = float(score)
    return scores


def _select(
    matrix: OutcomeMatrix,
    fit_ids: list[str],
    best_name: str,
    *,
    scorer: dict[EpisodeKey, float] | None,
    fixed_pair: tuple[str, str] | None = None,
) -> tuple[str, str, float, float, float] | None:
    """(cheap, strong, threshold, fit_acc, fit_cost) or None when the cascade declines.

    `scorer=None` is the ORACLE arm: the per-scenario decision statistic is the cheap cell's
    true mean reward, thresholds over a reward grid. Otherwise the statistic is the mean
    out-of-fold verifier score of the cheap cell's replies, thresholds over its deciles.
    All quantities are fit-side cell MEANS (selection never sees episode order or test data).
    """
    cells = _cells(matrix, set(fit_ids))
    rewards, costs = _fit_stats(cells)
    models = [entry.name for entry in matrix.pool]
    model_cost = {
        m: _mean([costs[(sid, m)] for sid in fit_ids if (sid, m) in costs]) for m in models
    }
    cap = CAP_FACTOR * _mean(
        [costs[(sid, best_name)] for sid in fit_ids if (sid, best_name) in costs]
    )

    def statistic(sid: str, cheap: str) -> float | None:
        if scorer is None:
            return rewards.get((sid, cheap))
        keys = [episode_key(o) for o in cells.get((sid, cheap), [])]
        values = [scorer[k] for k in keys if k in scorer]
        return _mean(values) if values else None

    rng = np.random.default_rng(0)
    resamples = [
        [fit_ids[i] for i in rng.integers(0, len(fit_ids), size=len(fit_ids))] for _ in range(200)
    ]

    best = None
    for cheap in models:
        for strong in models:
            if fixed_pair is not None and (cheap, strong) != fixed_pair:
                continue
            if cheap == strong:
                continue
            if model_cost.get(cheap, 0) > CHEAP_RATIO * model_cost.get(strong, 0):
                continue  # lateral hop, not a cascade (round 8a lesson)
            stats = {sid: statistic(sid, cheap) for sid in fit_ids}
            known = [v for v in stats.values() if v is not None]
            if len(known) < len(fit_ids) * 0.5:
                continue
            if scorer is None:
                grid = [round(t, 2) for t in np.arange(0.05, 1.0, 0.05)]
            else:
                grid = sorted({float(q) for q in np.quantile(known, np.arange(0.1, 1.0, 0.1))})
            for threshold in grid:
                accs, costs_out = [], []
                for sid in fit_ids:
                    value = stats[sid]
                    escalate = value is None or value < threshold
                    reward_cell = (sid, strong) if escalate else (sid, cheap)
                    if reward_cell not in rewards:
                        continue
                    accs.append(rewards[reward_cell])
                    cost = costs.get((sid, cheap), model_cost.get(cheap, 0.0))
                    if escalate:
                        cost += costs.get((sid, strong), model_cost.get(strong, 0.0))
                    costs_out.append(cost)
                if not accs:
                    continue
                acc, cost = _mean(accs), _mean(costs_out)
                if scorer is not None:
                    # 8c robust cost check: bootstrap the fit scenarios; the 80th-percentile
                    # cascade cost must clear the cap (mean caps do not survive the pooled
                    # corpora's heavy-tailed per-scenario costs - the 8a/8b lesson).
                    per_sid_cost = {}
                    for sid in fit_ids:
                        value = stats[sid]
                        escalate = value is None or value < threshold
                        cost_sid = costs.get((sid, cheap), model_cost.get(cheap, 0.0))
                        if escalate:
                            cost_sid += costs.get((sid, strong), model_cost.get(strong, 0.0))
                        per_sid_cost[sid] = cost_sid
                    boot = sorted(
                        _mean([per_sid_cost[sid] for sid in sample]) for sample in resamples
                    )
                    cost_check = boot[int(0.8 * len(boot))]
                else:
                    cost_check = cost
                if cost_check <= cap and (best is None or (acc, -cost) > (best[3], -best[4])):
                    best = (cheap, strong, threshold, acc, cost)
    return best


def _swap_episodes(matrix: OutcomeMatrix) -> OutcomeMatrix:
    """Permute per-cell episode order (0<->1) for the order-dependence average."""
    outcomes = []
    for outcome in matrix.outcomes:
        clone = outcome.model_copy()
        clone.episode = -outcome.episode  # 2-episode cells reverse; sort is stable for 1
        outcomes.append(clone)
    return OutcomeMatrix(pool=matrix.pool, outcomes=outcomes)


def run_matrix(name: str, matrix: OutcomeMatrix, seeds: list[int], *, oracle_only: bool) -> None:
    drv = _driver()
    embeddings = load_embeddings(matrix)
    ts = datetime.now(tz=UTC).isoformat()

    for seed in seeds:
        fit_ids, test_ids = drv.split_scenario_ids(matrix, train_fraction=0.7, seed=seed)
        best_name, _a, _c = drv.best_single_model(matrix, fit_ids=fit_ids, eval_ids=test_ids)
        best_eval = evaluate_choices(matrix, test_ids, lambda _sid, b=best_name: b)

        def record(
            variant: str,
            params: dict,
            result,  # noqa: ANN001
            notes: str,
            seed: int = seed,
            best_eval=best_eval,  # noqa: ANN001
            fit_ids: list[str] = fit_ids,
            test_ids: list[str] = test_ids,
        ) -> None:
            append_run(
                RunRecord(
                    run_id=f"r2-{name}-iid-s{seed}-{variant}-{uuid.uuid4().hex[:8]}",
                    ts=ts,
                    matrix=name,
                    variant=variant,
                    params={**params, "split": "iid"},
                    split_seed=seed,
                    fit_scenarios=len(fit_ids),
                    test_scenarios=len(test_ids),
                    result=result,
                    baselines={"best_single": best_eval},
                    notes=notes,
                ),
                RUNS,
            )
            logger.info(
                "%s/s%d %s: acc=%.4f cost=$%.5f (best-single %.4f/$%.5f) | %s",
                name,
                seed,
                variant,
                result.accuracy,
                result.cost_per_call,
                best_eval.accuracy,
                best_eval.cost_per_call,
                notes,
            )

        record(
            "r2-cascade-best-single",
            {"model": best_name},
            best_eval,
            "1x arm scored via evaluate_choices (order-independence)",
        )

        arms: list[tuple[str, dict[EpisodeKey, float] | None, str]] = [
            ("r2-oracle-cascade", None, "oracle")
        ]
        if not oracle_only:
            arms.append(("r2-cascade", _oof_scores(matrix, fit_ids, embeddings, seed), "absolute"))
            arms.append(("r2-cascade-gain", None, "gain"))  # scorer built after pair known
            if seed == 0:
                arms.append(
                    (
                        "r2-cascade-shuffled",
                        _oof_scores(matrix, fit_ids, embeddings, seed, shuffle=True),
                        "shuffled",
                    )
                )

        oracle_pair: tuple[str, str] | None = None
        gain_corr = 0.0
        for variant, scorer, mode in arms:
            if mode == "gain":
                if oracle_pair is None:
                    continue
                scorer, gain_corr = _oof_gain_scores(
                    matrix, fit_ids, embeddings, seed, *oracle_pair
                )
            chosen = _select(
                matrix,
                fit_ids,
                best_name,
                scorer=scorer,
                fixed_pair=oracle_pair if mode != "oracle" else None,
            )
            if chosen is None:
                record(
                    variant,
                    {"family": "cascade", "declined": True},
                    best_eval,
                    "declined: no (pair, threshold) met the fit cost cap; = best-single",
                )
                continue
            cheap, strong, threshold, fit_acc, fit_cost = chosen
            if mode == "oracle":
                oracle_pair = (cheap, strong)  # fit-label pair choice, reused by real arms

            if mode == "oracle":

                def decide(
                    sid: str,
                    transcript: list,
                    cheap: str = cheap,
                    strong: str = strong,
                    threshold: float = threshold,
                ) -> str | Finish:
                    if not transcript:
                        return cheap
                    if len(transcript) == 1:
                        reward = transcript[0].reward or 0.0
                        return Finish(pick=0) if reward >= threshold else strong
                    return Finish(pick=1)
            else:
                # Deployable: retrain on the FULL fit split, score the consumed episode.
                fit_set = set(fit_ids)
                if mode == "gain":
                    targets = _gain_targets(matrix, fit_ids, cheap, strong)
                    pairs = [(k, -v) for k, v in targets.items() if k in embeddings]
                    features = np.asarray([embeddings[k] for k, _v in pairs])
                    labels = np.asarray([v for _k, v in pairs], dtype=float)
                else:
                    train = [
                        o
                        for o in matrix.outcomes
                        if o.scenario_id in fit_set
                        and o.reward is not None
                        and episode_key(o) in embeddings
                    ]
                    features = np.asarray([embeddings[episode_key(o)] for o in train])
                    labels = np.asarray([o.reward for o in train], dtype=float)
                if mode == "shuffled":
                    labels = shuffled_rewards(labels, seed)
                final: ReplyVerifier = fit_absolute(features, labels, alpha=ALPHA)

                def decide(
                    sid: str,
                    transcript: list,
                    cheap: str = cheap,
                    strong: str = strong,
                    threshold: float = threshold,
                    final: ReplyVerifier = final,
                ) -> str | Finish:
                    if not transcript:
                        return cheap
                    if len(transcript) == 1:
                        vector = embeddings.get(episode_key(transcript[0]))
                        if vector is None:
                            return strong  # unverifiable reply: escalate
                        score = float(final.score(vector)[0])
                        return Finish(pick=0) if score >= threshold else strong
                    return Finish(pick=1)

            results = []
            for permuted in (matrix, _swap_episodes(matrix)):
                results.append(evaluate_call_sequences(permuted, test_ids, decide, max_calls=2))
            mean_result = results[0].model_copy(
                update={
                    "accuracy": _mean([r.accuracy for r in results]),
                    "cost_per_call": _mean([r.cost_per_call for r in results]),
                }
            )
            escalations = mean_result.model_mix.get(strong, 0.0)
            record(
                variant,
                {"family": "cascade", "verifier": mode, "folds": FOLDS},
                mean_result,
                f"pair={cheap}->{strong} thr={threshold:.4f} esc_rate={escalations:.2f} "
                f"fit_acc={fit_acc:.4f} fit_cost={fit_cost:.5f} "
                + (f"gain_corr={gain_corr:.3f} " if mode == "gain" else "")
                + f"perm_accs={[round(r.accuracy, 4) for r in results]}",
            )


# ---------------------------------------------------------------------------
# Round 9: two-signal escalation (cheap-reply signal AND task-side strong-success).
# Pre-registered in findings/r2.md 2026-07-25 before running.
# ---------------------------------------------------------------------------

KNN_NEIGHBORS = 50


def _task_vectors(name: str, matrix: OutcomeMatrix) -> dict[str, np.ndarray]:
    """3-large task vectors via the driver's per-corpus caches (L2-normalized)."""
    drv = _driver()
    raw = drv._oai_vectors(name, matrix)
    if raw is None:
        raise ValueError(f"no cached task vectors for {name}")
    out = {}
    for sid, vector in raw.items():
        v = np.asarray(vector, dtype=np.float64)
        norm = np.linalg.norm(v)
        out[sid] = v / norm if norm > 0 else v
    return out


def _strong_estimates(
    fit_ids: list[str],
    target_ids: list[str],
    task_vecs: dict[str, np.ndarray],
    strong_means: dict[str, float],
    *,
    loo: bool,
) -> dict[str, float]:
    """Similarity-weighted mean of the strong model's fit cell-mean rewards over kNN(50).

    `loo=True` excludes the target scenario from its own neighborhood (used whenever the
    estimate feeds fit-side selection); test-side estimates never contain test scenarios
    anyway because the bank is the fit split.
    """
    bank = [sid for sid in fit_ids if sid in strong_means and sid in task_vecs]
    bank_matrix = np.asarray([task_vecs[sid] for sid in bank])
    bank_rewards = np.asarray([strong_means[sid] for sid in bank])
    out: dict[str, float] = {}
    for sid in target_ids:
        vector = task_vecs.get(sid)
        if vector is None:
            continue
        sims = bank_matrix @ vector
        if loo and sid in bank:
            sims = sims.copy()
            sims[bank.index(sid)] = -np.inf
        k = min(KNN_NEIGHBORS, len(bank))
        top = np.argpartition(-sims, k - 1)[:k]
        weights = np.clip(sims[top], 0.0, None)
        if weights.sum() <= 0:
            continue
        out[sid] = float((weights * bank_rewards[top]).sum() / weights.sum())
    return out


def _select2(
    matrix: OutcomeMatrix,
    fit_ids: list[str],
    best_name: str,
    pair: tuple[str, str],
    cheap_stats: dict[str, float | None],
    strong_est: dict[str, float],
) -> tuple[float, float, float, float] | None:
    """(cheap threshold, strong bar, fit acc, fit cost) for the AND rule, or None.

    Escalate iff cheap_stat < t AND strong_est >= b. Grid = deciles of both signals;
    max fit accuracy under the bootstrap-p80 cost rule (8c).
    """
    cheap, strong = pair
    cells = _cells(matrix, set(fit_ids))
    rewards, costs = _fit_stats(cells)
    model_cost = {
        m: _mean([costs[(sid, m)] for sid in fit_ids if (sid, m) in costs])
        for m in (cheap, strong, best_name)
    }
    cap = CAP_FACTOR * _mean(
        [costs[(sid, best_name)] for sid in fit_ids if (sid, best_name) in costs]
    )
    rng = np.random.default_rng(0)
    resamples = [
        [fit_ids[i] for i in rng.integers(0, len(fit_ids), size=len(fit_ids))] for _ in range(200)
    ]
    cheap_known = [v for v in cheap_stats.values() if v is not None]
    strong_known = [strong_est[sid] for sid in fit_ids if sid in strong_est]
    if not cheap_known or not strong_known:
        return None
    t_grid = sorted({float(q) for q in np.quantile(cheap_known, np.arange(0.1, 1.0, 0.1))})
    b_grid = sorted({float(q) for q in np.quantile(strong_known, np.arange(0.1, 1.0, 0.1))})

    best = None
    for t in t_grid:
        for b in b_grid:
            per_acc, per_cost = {}, {}
            for sid in fit_ids:
                value = cheap_stats.get(sid)
                weak = value is None or value < t
                confident = strong_est.get(sid, -np.inf) >= b
                escalate = weak and confident
                reward_cell = (sid, strong) if escalate else (sid, cheap)
                if reward_cell not in rewards:
                    continue
                per_acc[sid] = rewards[reward_cell]
                cost_sid = costs.get((sid, cheap), model_cost[cheap])
                if escalate:
                    cost_sid += costs.get((sid, strong), model_cost[strong])
                per_cost[sid] = cost_sid
            if not per_acc:
                continue
            acc = _mean(list(per_acc.values()))
            boot = sorted(
                _mean([per_cost[sid] for sid in sample if sid in per_cost]) for sample in resamples
            )
            if boot[int(0.8 * len(boot))] <= cap and (
                best is None
                or acc > best[2]
                or (acc == best[2] and _mean(list(per_cost.values())) < best[3])
            ):
                best = (t, b, acc, _mean(list(per_cost.values())))
    return best


def run_round9(
    name: str, matrix: OutcomeMatrix, seeds: list[int], *, oracle2_only: bool = False
) -> None:
    drv = _driver()
    embeddings = load_embeddings(matrix)
    task_vecs = _task_vectors(name, matrix)
    ts = datetime.now(tz=UTC).isoformat()

    for seed in seeds:
        fit_ids, test_ids = drv.split_scenario_ids(matrix, train_fraction=0.7, seed=seed)
        best_name, _a, _c = drv.best_single_model(matrix, fit_ids=fit_ids, eval_ids=test_ids)
        best_eval = evaluate_choices(matrix, test_ids, lambda _sid, b=best_name: b)
        pair_choice = _select(matrix, fit_ids, best_name, scorer=None)
        if pair_choice is None:
            logger.info("%s/s%d: no viable pair under the cap; skipping seed", name, seed)
            continue
        cheap, strong = pair_choice[0], pair_choice[1]

        cells = _cells(matrix, set(fit_ids))
        strong_means = {
            sid: _mean([o.reward for o in cells[(sid, strong)]])
            for sid in fit_ids
            if (sid, strong) in cells
        }
        est_fit = _strong_estimates(fit_ids, fit_ids, task_vecs, strong_means, loo=True)
        est_test = _strong_estimates(fit_ids, test_ids, task_vecs, strong_means, loo=False)

        gain_scores, gain_corr = _oof_gain_scores(matrix, fit_ids, embeddings, seed, cheap, strong)

        def cheap_stat_from(
            scores: dict[EpisodeKey, float],
            fit_ids: list[str] = fit_ids,
            cells: dict = cells,
            cheap: str = cheap,
        ) -> dict[str, float | None]:
            out: dict[str, float | None] = {}
            for sid in fit_ids:
                keys = [episode_key(o) for o in cells.get((sid, cheap), [])]
                values = [scores[k] for k in keys if k in scores]
                out[sid] = _mean(values) if values else None
            return out

        oracle_stats = {
            sid: _mean([o.reward for o in cells[(sid, cheap)]]) if (sid, cheap) in cells else None
            for sid in fit_ids
        }

        arms = [("r2-oracle2-cascade", oracle_stats, None, "oracle2")]
        if not oracle2_only:
            arms.append(("r2-cascade2", cheap_stat_from(gain_scores), gain_scores, "two-signal"))
        if seed == 0 and not oracle2_only:
            sh_scores, _sc = _oof_gain_scores(matrix, fit_ids, embeddings, seed, cheap, strong)
            keys = list(sh_scores)
            values = shuffled_rewards(np.asarray([sh_scores[k] for k in keys]), seed)
            sh_scores = dict(zip(keys, [float(v) for v in values], strict=True))
            arms.append(("r2-cascade2-shuffled", cheap_stat_from(sh_scores), sh_scores, "shuffled"))

        for variant, stats, _episode_scores, mode in arms:
            chosen = _select2(matrix, fit_ids, best_name, (cheap, strong), stats, est_fit)
            if chosen is None:
                logger.info("%s/s%d %s: declined (no grid point under cap)", name, seed, variant)
                continue
            t, b, fit_acc, fit_cost = chosen

            if mode == "oracle2":

                def decide(
                    sid: str,
                    transcript: list,
                    cheap: str = cheap,
                    strong: str = strong,
                    t: float = t,
                    b: float = b,
                    est_test: dict = est_test,
                ) -> str | Finish:
                    if not transcript:
                        return cheap
                    if len(transcript) == 1:
                        weak = (transcript[0].reward or 0.0) < t
                        confident = est_test.get(sid, -np.inf) >= b
                        return strong if (weak and confident) else Finish(pick=0)
                    return Finish(pick=1)
            else:
                targets = _gain_targets(matrix, fit_ids, cheap, strong)
                pairs = [(k, -v) for k, v in targets.items() if k in embeddings]
                labels = np.asarray([v for _k, v in pairs], dtype=float)
                if mode == "shuffled":
                    labels = shuffled_rewards(labels, seed)
                final = fit_absolute(
                    np.asarray([embeddings[k] for k, _v in pairs]), labels, alpha=ALPHA
                )

                def decide(
                    sid: str,
                    transcript: list,
                    cheap: str = cheap,
                    strong: str = strong,
                    t: float = t,
                    b: float = b,
                    est_test: dict = est_test,
                    final=final,  # noqa: ANN001
                ) -> str | Finish:
                    if not transcript:
                        return cheap
                    if len(transcript) == 1:
                        vector = embeddings.get(episode_key(transcript[0]))
                        weak = True if vector is None else float(final.score(vector)[0]) < t
                        confident = est_test.get(sid, -np.inf) >= b
                        return strong if (weak and confident) else Finish(pick=0)
                    return Finish(pick=1)

            results = [
                evaluate_call_sequences(m2, test_ids, decide, max_calls=2)
                for m2 in (matrix, _swap_episodes(matrix))
            ]
            mean_result = results[0].model_copy(
                update={
                    "accuracy": _mean([r.accuracy for r in results]),
                    "cost_per_call": _mean([r.cost_per_call for r in results]),
                }
            )
            append_run(
                RunRecord(
                    run_id=f"r2-{name}-iid-s{seed}-{variant}-{uuid.uuid4().hex[:8]}",
                    ts=ts,
                    matrix=name,
                    variant=variant,
                    params={
                        "family": "cascade2",
                        "knn": KNN_NEIGHBORS,
                        "folds": FOLDS,
                        "split": "iid",
                    },
                    split_seed=seed,
                    fit_scenarios=len(fit_ids),
                    test_scenarios=len(test_ids),
                    result=mean_result,
                    baselines={"best_single": best_eval},
                    notes=(
                        f"pair={cheap}->{strong} t={t:.4f} b={b:.4f} "
                        f"esc_rate={mean_result.model_mix.get(strong, 0.0):.2f} "
                        f"fit_acc={fit_acc:.4f} gain_corr={gain_corr:.3f} "
                        f"perm_accs={[round(r.accuracy, 4) for r in results]}"
                    ),
                ),
                RUNS,
            )
            logger.info(
                "%s/s%d %s: acc=%.4f cost=$%.5f (best %.4f/$%.5f) t=%.3f b=%.3f",
                name,
                seed,
                variant,
                mean_result.accuracy,
                mean_result.cost_per_call,
                best_eval.accuracy,
                best_eval.cost_per_call,
                t,
                b,
            )


# ---------------------------------------------------------------------------
# Round 10: self-consistency cascade (gate PASSED; pre-registered 2026-07-25).
# Two cheap calls; keep the within-cell verifier pick; escalate iff the two cheap
# replies DISAGREE (low cosine) AND the kNN strong-success filter is confident.
# ---------------------------------------------------------------------------


def _consistency(
    eps: list[ScenarioOutcome], embeddings: dict[EpisodeKey, np.ndarray]
) -> float | None:
    if len(eps) < 2:
        return None
    v0, v1 = embeddings.get(episode_key(eps[0])), embeddings.get(episode_key(eps[1]))
    if v0 is None or v1 is None:
        return None
    return float(v0 @ v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))


def _select10(
    matrix: OutcomeMatrix,
    fit_ids: list[str],
    best_name: str,
    pair: tuple[str, str],
    embeddings: dict[EpisodeKey, np.ndarray],
    oof: dict[EpisodeKey, float],
    strong_est: dict[str, float],
    *,
    shuffle_seed: int | None = None,
) -> tuple[float, float, float, float] | None:
    """(consistency threshold c, strong bar b, fit acc, fit cost) under the cost rule.

    Fit value per scenario: two cheap episodes; kept reward = the OOF-verifier-picked
    episode's reward; escalate (consistency < c AND strong_est >= b) -> strong cell mean;
    cost = both cheap episodes + escalation. `shuffle_seed` permutes the episode->embedding
    map for the consistency signal (the control: must kill the gate signal).
    """
    cheap, strong = pair
    cells = _cells(matrix, set(fit_ids))
    rewards, costs = _fit_stats(cells)
    cap = CAP_FACTOR * _mean(
        [costs[(sid, best_name)] for sid in fit_ids if (sid, best_name) in costs]
    )
    emb = embeddings
    if shuffle_seed is not None:
        keys = [k for k in embeddings if k[1] == cheap]
        perm = list(keys)
        np.random.default_rng(shuffle_seed).shuffle(perm)
        emb = dict(embeddings)
        for old, new in zip(keys, perm, strict=True):
            emb[old] = embeddings[new]

    per: dict[str, tuple[float | None, float, float, float, float]] = {}
    for sid in fit_ids:
        eps = cells.get((sid, cheap), [])
        if not eps or (sid, strong) not in rewards:
            continue
        cons = _consistency(eps, emb)
        scored = [(oof.get(episode_key(o), 0.0), o.reward) for o in eps]
        kept = max(scored, key=lambda sr: sr[0])[1]
        cheap_cost = sum(o.cost_usd for o in eps[:2])
        per[sid] = (cons, kept, rewards[(sid, strong)], cheap_cost, costs[(sid, strong)])
    if not per:
        return None
    known = sorted(v[0] for v in per.values() if v[0] is not None)
    if not known:
        return None
    c_grid = sorted({float(q) for q in np.quantile(known, np.arange(0.1, 1.0, 0.1))})
    s_known = [strong_est[sid] for sid in per if sid in strong_est]
    b_grid = sorted({float(q) for q in np.quantile(s_known, np.arange(0.1, 1.0, 0.1))})
    rng = np.random.default_rng(0)
    sids = list(per)
    resamples = [[sids[i] for i in rng.integers(0, len(sids), size=len(sids))] for _ in range(200)]
    best = None
    for c_thr in c_grid:
        for b in b_grid:
            acc_map, cost_map = {}, {}
            for sid, (cons, kept, strong_r, cheap_cost, strong_cost) in per.items():
                weak = cons is None or cons < c_thr
                escalate = weak and strong_est.get(sid, -np.inf) >= b
                acc_map[sid] = strong_r if escalate else kept
                cost_map[sid] = cheap_cost + (strong_cost if escalate else 0.0)
            acc = _mean(list(acc_map.values()))
            boot = sorted(_mean([cost_map[s] for s in sample]) for sample in resamples)
            if boot[int(0.8 * len(boot))] <= cap and (best is None or acc > best[2]):
                best = (c_thr, b, acc, _mean(list(cost_map.values())))
    return best


def run_round10(name: str, matrix: OutcomeMatrix, seeds: list[int]) -> None:
    drv = _driver()
    embeddings = load_embeddings(matrix)
    task_vecs = _task_vectors(name, matrix)
    ts = datetime.now(tz=UTC).isoformat()
    for seed in seeds:
        fit_ids, test_ids = drv.split_scenario_ids(matrix, train_fraction=0.7, seed=seed)
        best_name, _a, _c = drv.best_single_model(matrix, fit_ids=fit_ids, eval_ids=test_ids)
        best_eval = evaluate_choices(matrix, test_ids, lambda _sid, b=best_name: b)
        cells = _cells(matrix, set(fit_ids))
        rewards, costs = _fit_stats(cells)
        model_cost = {
            m.name: _mean([costs[(sid, m.name)] for sid in fit_ids if (sid, m.name) in costs])
            for m in matrix.pool
        }
        oof = _oof_scores(matrix, fit_ids, embeddings, seed)

        # Pair: fit-label rule with the TWO-call cheap cost; enumerate, keep best fit value.
        chosen_pair, chosen_cfg = None, None
        for cheap in model_cost:
            for strong in model_cost:
                if cheap == strong:
                    continue
                if 2 * model_cost[cheap] > CHEAP_RATIO * model_cost[strong]:
                    continue
                strong_means = {
                    sid: rewards[(sid, strong)] for sid in fit_ids if (sid, strong) in rewards
                }
                if len(strong_means) < len(fit_ids) * 0.5:
                    continue
                est_fit = _strong_estimates(fit_ids, fit_ids, task_vecs, strong_means, loo=True)
                cfg = _select10(
                    matrix, fit_ids, best_name, (cheap, strong), embeddings, oof, est_fit
                )
                if cfg and (chosen_cfg is None or cfg[2] > chosen_cfg[2]):
                    chosen_pair, chosen_cfg = (cheap, strong), cfg
        if chosen_pair is None:
            logger.info("%s/s%d round10: declined (no pair under cap)", name, seed)
            continue
        cheap, strong = chosen_pair
        c_thr, b, fit_acc, fit_cost = chosen_cfg
        strong_means = {sid: rewards[(sid, strong)] for sid in fit_ids if (sid, strong) in rewards}
        est_test = _strong_estimates(fit_ids, test_ids, task_vecs, strong_means, loo=False)

        train = [
            o
            for o in matrix.outcomes
            if o.scenario_id in set(fit_ids)
            and o.reward is not None
            and episode_key(o) in embeddings
        ]
        final = fit_absolute(
            np.asarray([embeddings[episode_key(o)] for o in train]),
            np.asarray([o.reward for o in train], dtype=float),
            alpha=ALPHA,
        )

        def decide(
            sid: str,
            transcript: list,
            cheap: str = cheap,
            strong: str = strong,
            c_thr: float = c_thr,
            b: float = b,
            final: ReplyVerifier = final,
            est_test: dict = est_test,
        ) -> str | Finish:
            if len(transcript) < 2:
                return cheap
            if len(transcript) == 2:
                cons = _consistency(list(transcript), embeddings)
                weak = cons is None or cons < c_thr
                if weak and est_test.get(sid, -np.inf) >= b:
                    return strong
                picks = []
                for index, outcome in enumerate(transcript):
                    vector = embeddings.get(episode_key(outcome))
                    picks.append(
                        (float(final.score(vector)[0]) if vector is not None else -1e9, index)
                    )
                return Finish(pick=max(picks)[1])
            return Finish(pick=2)

        results = [
            evaluate_call_sequences(m2, test_ids, decide, max_calls=3)
            for m2 in (matrix, _swap_episodes(matrix))
        ]
        mean_result = results[0].model_copy(
            update={
                "accuracy": _mean([r.accuracy for r in results]),
                "cost_per_call": _mean([r.cost_per_call for r in results]),
            }
        )
        append_run(
            RunRecord(
                run_id=f"r2-{name}-iid-s{seed}-r2-cascade3-{uuid.uuid4().hex[:8]}",
                ts=ts,
                matrix=name,
                variant="r2-cascade3",
                params={"family": "cascade3", "knn": KNN_NEIGHBORS, "split": "iid"},
                split_seed=seed,
                fit_scenarios=len(fit_ids),
                test_scenarios=len(test_ids),
                result=mean_result,
                baselines={"best_single": best_eval},
                notes=(
                    f"pair={cheap}->{strong} c={c_thr:.4f} b={b:.4f} "
                    f"esc_rate={mean_result.model_mix.get(strong, 0.0):.2f} "
                    f"fit_acc={fit_acc:.4f} fit_cost={fit_cost:.5f} "
                    f"perm_accs={[round(r.accuracy, 4) for r in results]}"
                ),
            ),
            RUNS,
        )
        logger.info(
            "%s/s%d r2-cascade3: acc=%.4f cost=$%.5f (best %.4f/$%.5f) pair=%s->%s c=%.3f",
            name,
            seed,
            mean_result.accuracy,
            mean_result.cost_per_call,
            best_eval.accuracy,
            best_eval.cost_per_call,
            cheap,
            strong,
            c_thr,
        )


def main() -> None:
    args = sys.argv[1:]
    wanted = [a for a in args if not a.startswith("--")] or ["wm-all"]
    seeds = SEEDS
    for arg in args:
        if arg.startswith("--seeds="):
            seeds = [int(s) for s in arg.split("=", 1)[1].split(",")]
    oracle_only = "--oracle-only" in args
    round9 = "--round9" in args
    round10 = "--round10" in args
    oracle2_only = "--oracle2-only" in args
    drv = _driver()
    matrices = drv._matrices()
    for name in wanted:
        if round10:
            run_round10(name, matrices[name], seeds)
        elif round9:
            run_round9(name, matrices[name], seeds, oracle2_only=oracle2_only)
        else:
            run_matrix(name, matrices[name], seeds, oracle_only=oracle_only)


if __name__ == "__main__":
    main()
