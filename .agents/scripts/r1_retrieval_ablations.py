"""R1 retrieval-routing ablations: the jisi-proxy family, parameterized and held to proof.

Chat R1 of the routing hill-climb. One parameterized retrieval router (kNN reward profile over
task-embedding neighbors, optional proxy_s2s second route, optional guard) whose default
configuration reproduces the master chat's `jisi` variant bit for bit; every knob then ablates
independently. All variants evaluate through `wmo.research.routing_runs.evaluate_choices` on
the shared matrices and append RunRecords (variant names `r1-*`) to the shared runs file.

Usage:
    uv run python .agents/scripts/r1_retrieval_ablations.py [matrix ...] [--seeds] [--debug]

Controls included: shuffled-label (permute each model's rewards across scenarios; a real
router must collapse to ~best-single) and a fit/test near-duplicate audit.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from wmo.optimize.outcomes import OutcomeMatrix
from wmo.research.routerbench import best_single_model, oracle, split_scenario_ids
from wmo.research.routing_runs import RunRecord, append_run, evaluate_choices
from wmo.retrieval.embedders import HashingEmbedder
from wmo.research.routing_corpus import routing_data

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("r1")

DATA = routing_data()
RUNS = DATA / "runs" / "r1.jsonl"
DEBUG_DIR = DATA / "findings" / "r1_debug"
DIM = 1024
SPLIT_SEEDS = [0, 1, 2, 3, 4]


class RetrievalParams(BaseModel):
    """One retrieval-router configuration; defaults replicate the master jisi variant."""

    neighbor_rule: str = "relative"  # relative (jisi: sim > thres * kth) | fixed_k
    rag_num: int = 50
    rag_thres: float = 0.95
    second_route: bool = True  # proxy_s2s neighbor refinement
    subset_p: float = 0.5
    needle_n: int = 3
    weighted: bool = True  # sim-weighted profile; False = uniform mean
    # Guard modes. margin: fixed profile-gap margin (the master jisi guard). stat: paired
    # per-neighbor reward differences must clear z standard errors (support-aware: the margin
    # shrinks as neighbor evidence grows, unlike the fixed margin). stat_asym: pricier picks
    # must be significantly better (> z*se), cheaper picks merely not significantly worse
    # (> -z*se): the economic asymmetry that lets a cost knob act. none: unguarded.
    guard: str = "margin"  # none | margin | stat | stat_asym
    guard_margin: float = 0.03  # margin mode: doubled when the pick is pricier than baseline
    z: float = 0.5  # stat mode: required standard errors (doubled when pricier)
    min_pairs: int = 8  # stat mode: revert picks with fewer paired neighbors than this
    # Variance floor for the stat guard: se_eff = max(se, sqrt(0.25/n_pairs)), the maximal
    # binomial SE. Kills the small-bank failure where a lucky zero-variance neighborhood
    # makes any mean_d > 0 look significant (pricier-and-worse picks on 18-item banks).
    # Applied only below se_floor_max_pairs: a SMALL-SAMPLE correction (large-n empirical
    # SE is reliable; flooring it there just taxes real wins).
    se_floor: bool = False
    se_floor_max_pairs: int = 30
    distance_floor: float | None = None  # abstain to baseline when max sim < floor
    sim_gamma: float = 1.0  # weight = sim^gamma (weighted mode); >1 sharpens toward near items
    smooth_alpha: float = 0.0  # Laplace pseudo-count toward the fit-global mean per model
    pick_lam: float = 0.0  # cost-aware pick: argmax(profile - lam * cost / cost_scale)


def _openai_embed(texts: list[str], cache_path: Path) -> np.ndarray:
    """Embed with text-embedding-3-large via the OpenAI API, cached to disk (order-stable)."""
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(texts):
            return cached
        logger.warning(
            "cache %s has %d rows, need %d; re-embedding", cache_path, cached.shape[0], len(texts)
        )
    import time
    import urllib.request

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("--embed openai needs OPENAI_API_KEY in the environment")
    vectors: list[list[float]] = []
    total_tokens = 0
    for start in range(0, len(texts), 128):
        chunk = [t[:18000] if t.strip() else " " for t in texts[start : start + 128]]
        payload = json.dumps({"model": "text-embedding-3-large", "input": chunk}).encode()
        request = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    data = json.loads(response.read())
                break
            except Exception as error:  # noqa: BLE001 (retry rate limits/transients, then raise)
                if attempt == 3:
                    raise
                logger.warning("embed batch %d retry %d: %s", start, attempt + 1, error)
                time.sleep(5 * (attempt + 1))
        vectors.extend(d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"]))
        total_tokens += data.get("usage", {}).get("total_tokens", 0)
        if start % 1280 == 0:
            logger.info(
                "embedded %d/%d (tokens so far %d)", start + len(chunk), len(texts), total_tokens
            )
    array = np.asarray(vectors)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, array)
    logger.info(
        "openai embed: %d texts, %d tokens (~$%.3f) -> %s",
        len(texts),
        total_tokens,
        total_tokens / 1e6 * 0.13,
        cache_path,
    )
    return array


class MatrixContext:
    """Split-independent precomputation for one matrix: embeddings, reply vectors, cells."""

    def __init__(
        self,
        matrix: OutcomeMatrix,
        name: str,
        embed: str = "hashing",
        embed_replies: bool = True,
    ) -> None:
        self.matrix = matrix
        self.embed_kind = embed
        self.model_names = [entry.name for entry in matrix.pool]
        self.tasks: dict[str, str] = {}
        for outcome in matrix.outcomes:
            self.tasks.setdefault(outcome.scenario_id, outcome.task)
        self.scenario_ids = list(self.tasks)
        self.rewards_cell: dict[tuple[str, str], list[float]] = {}
        reply_texts: dict[tuple[str, str], str] = {}
        for outcome in matrix.outcomes:
            if outcome.reward is None:
                continue
            key = (outcome.scenario_id, outcome.model)
            self.rewards_cell.setdefault(key, []).append(outcome.reward)
            if embed_replies and outcome.replies and key not in reply_texts:
                reply_texts[key] = outcome.replies[0]
        reply_keys = list(reply_texts)

        if embed == "hashing":
            embedder = HashingEmbedder(dim=DIM)
            vecs = np.asarray(embedder.embed([self.tasks[sid] for sid in self.scenario_ids]))
            rvecs = (
                np.asarray(embedder.embed([reply_texts[k] for k in reply_keys]))
                if reply_keys
                else np.zeros((0, DIM))
            )
        elif embed == "openai":
            cache = DATA / "cache"
            vecs = _openai_embed(
                [self.tasks[sid] for sid in self.scenario_ids],
                cache / f"{name}-oai3l-tasks.npy",
            )
            # Skip the reply call entirely when replies are unused: calling it with an
            # empty list would overwrite the shared reply cache with a 0-row array.
            rvecs = (
                _openai_embed(
                    [reply_texts[k] for k in reply_keys],
                    cache / f"{name}-oai3l-replies.npy",
                )
                if reply_keys
                else np.zeros((0, 1))
            )
        else:
            raise ValueError(f"unknown embed kind: {embed}")

        def _norm(matrix_: np.ndarray) -> np.ndarray:
            norms = np.linalg.norm(matrix_, axis=1, keepdims=True)
            return matrix_ / np.where(norms > 0, norms, 1.0)

        vecs = _norm(vecs)
        self.task_vecs = {sid: vecs[i] for i, sid in enumerate(self.scenario_ids)}
        rvecs = _norm(rvecs) if len(rvecs) else rvecs
        self.reply_vecs = {k: rvecs[i] for i, k in enumerate(reply_keys)}

    def mean_cost(self, fit_ids: list[str]) -> dict[str, float]:
        fit_set = set(fit_ids)
        sums: dict[str, tuple[float, int]] = {}
        for outcome in self.matrix.outcomes:
            if outcome.scenario_id in fit_set and outcome.reward is not None:
                total, count = sums.get(outcome.model, (0.0, 0))
                sums[outcome.model] = (total + outcome.cost_usd, count + 1)
        return {m: total / count for m, (total, count) in sums.items() if count}


def route(
    ctx: MatrixContext,
    params: RetrievalParams,
    fit_ids: list[str],
    test_ids: list[str],
    best_name: str,
    rewards_cell: dict[tuple[str, str], list[float]] | None = None,
    debug: list[dict] | None = None,
    query_vecs: dict[str, np.ndarray] | None = None,
    evidence: dict[str, dict] | None = None,
) -> dict[str, str]:
    """Route every test scenario; returns sid -> model. Faithful jisi-proxy at defaults.

    `query_vecs` overrides the query embedding per sid (the dup-traffic experiment routes a
    perturbed text while scoring against the original scenario's cells). `evidence`, when
    given, receives per-sid PRE-GUARD routing evidence (raw pick, paired mean delta vs the
    baseline over the used neighbors, its standard error, pair count, pricier flag): the
    conformal guard calibrates its accept threshold on exactly this.
    """
    cells = rewards_cell if rewards_cell is not None else ctx.rewards_cell
    fit_matrix = np.stack([ctx.task_vecs[sid] for sid in fit_ids])
    mean_cost = ctx.mean_cost(fit_ids)
    base_cost = mean_cost.get(best_name, 0.0)
    cost_scale = (sum(mean_cost.values()) / len(mean_cost)) if mean_cost else 1.0
    picks: dict[str, str] = {}

    # Fit-global per-model means: the smoothing prior (and nothing else) uses them.
    global_mean: dict[str, float] = {}
    if params.smooth_alpha > 0:
        for model in ctx.model_names:
            values = [sum(cell) / len(cell) for sid in fit_ids if (cell := cells.get((sid, model)))]
            if values:
                global_mean[model] = sum(values) / len(values)

    def profile(rows_idx: np.ndarray, weights: np.ndarray) -> dict[str, float]:
        scores: dict[str, float] = {}
        for model in ctx.model_names:
            num = den = 0.0
            for j, weight in zip(rows_idx, weights, strict=True):
                cell = cells.get((fit_ids[int(j)], model))
                if cell:
                    w = max(float(weight), 0.0) ** params.sim_gamma if params.weighted else 1.0
                    num += w * (sum(cell) / len(cell))
                    den += w
            if params.smooth_alpha > 0 and model in global_mean:
                num += params.smooth_alpha * global_mean[model]
                den += params.smooth_alpha
            if den:
                scores[model] = num / den
        return scores

    for sid in test_ids:
        query = query_vecs[sid] if query_vecs is not None else ctx.task_vecs[sid]
        sims = fit_matrix @ query
        info: dict = {"sid": sid, "task": ctx.tasks[sid][:200]}
        if params.distance_floor is not None and float(np.max(sims)) < params.distance_floor:
            picks[sid] = best_name
            if debug is not None:
                info.update({"pick": best_name, "why": "distance floor abstain"})
                debug.append(info)
            continue
        k = min(params.rag_num, len(fit_ids))
        if params.neighbor_rule == "fixed_k":
            neighbor_rows = np.argsort(sims)[-k:]
        else:
            kth = np.sort(sims)[-k]
            neighbor_rows = np.where(sims > params.rag_thres * kth)[0]
        if not len(neighbor_rows):
            neighbor_rows = np.asarray([int(np.argmax(sims))])
        first = profile(neighbor_rows, sims[neighbor_rows])
        if not first:
            picks[sid] = best_name
            continue
        chosen_profile = first
        used_rows = neighbor_rows
        if params.second_route:
            needles = sorted(first, key=lambda m: -first[m])[: params.needle_n]
            refine = []
            for j in neighbor_rows:
                sims_resp = []
                for model in needles:
                    mine = ctx.reply_vecs.get((fit_ids[int(j)], model))
                    if mine is None:
                        continue
                    others = [
                        ctx.reply_vecs[(fit_ids[int(o)], model)]
                        for o in neighbor_rows
                        if o != j and (fit_ids[int(o)], model) in ctx.reply_vecs
                    ]
                    if not others:
                        continue
                    sims_resp.append(float(mine @ np.mean(others[:5], axis=0)))
                resp = float(np.mean(sims_resp)) if sims_resp else 0.0
                refine.append(0.5 * float(sims[int(j)]) + 0.5 * resp)
            refine_arr = np.asarray(refine)
            keep = max(1, int(len(neighbor_rows) * params.subset_p))
            top_idx = np.argsort(refine_arr)[-keep:]
            second = profile(neighbor_rows[top_idx], refine_arr[top_idx])
            if second:
                chosen_profile = second
                used_rows = neighbor_rows[top_idx]
        pick = max(
            chosen_profile,
            key=lambda m: (
                chosen_profile[m] - params.pick_lam * mean_cost.get(m, cost_scale) / cost_scale,
                -ctx.model_names.index(m),
            ),
        )

        def paired_stats(candidate: str, rows: np.ndarray = used_rows) -> tuple[float, float, int]:
            """Paired per-neighbor reward differences of `candidate` vs the baseline."""
            diffs = []
            for j in rows:
                cell_pick = cells.get((fit_ids[int(j)], candidate))
                cell_base = cells.get((fit_ids[int(j)], best_name))
                if cell_pick and cell_base:
                    diffs.append(sum(cell_pick) / len(cell_pick) - sum(cell_base) / len(cell_base))
            if not diffs:
                return 0.0, 0.0, 0
            mean_d = float(np.mean(diffs))
            se = float(np.std(diffs, ddof=1)) / len(diffs) ** 0.5 if len(diffs) > 1 else 0.0
            return mean_d, se, len(diffs)

        if evidence is not None:
            mean_d, se, n_pairs = paired_stats(pick)
            evidence[sid] = {
                "pick": pick,
                "mean_d": mean_d,
                "se": se,
                "n_pairs": n_pairs,
                "pricier": mean_cost.get(pick, 0.0) > base_cost,
            }
        guarded = False
        if params.guard == "margin":
            margin = params.guard_margin
            if mean_cost.get(pick, 0.0) > base_cost:
                margin = 2 * params.guard_margin
            if chosen_profile.get(pick, 0.0) <= chosen_profile.get(best_name, 0.0) + margin:
                pick = best_name
                guarded = True
        elif params.guard in ("stat", "stat_asym") and pick != best_name:
            mean_d, se, n_pairs = paired_stats(pick)
            if params.se_floor and 0 < n_pairs < params.se_floor_max_pairs:
                se = max(se, (0.25 / n_pairs) ** 0.5)
            pricier = mean_cost.get(pick, 0.0) > base_cost
            if params.guard == "stat_asym":
                z_eff = params.z if pricier else -params.z
            else:
                z_eff = 2 * params.z if pricier else params.z
            if n_pairs < params.min_pairs or not mean_d > z_eff * se:
                pick = best_name
                guarded = True
        picks[sid] = pick
        if debug is not None:
            neighbor_ids = [fit_ids[int(j)] for j in neighbor_rows]
            info.update(
                {
                    "n_neighbors": len(neighbor_rows),
                    "max_sim": round(float(np.max(sims)), 4),
                    "neighbors": [
                        {
                            "sid": nid,
                            "sim": round(float(sims[int(j)]), 4),
                            "task": ctx.tasks[nid][:120],
                        }
                        for j, nid in list(zip(neighbor_rows, neighbor_ids, strict=True))[:8]
                    ],
                    "profile": {m: round(v, 4) for m, v in sorted(chosen_profile.items())},
                    "pick": pick,
                    "guard_reverted": guarded,
                }
            )
            debug.append(info)
    return picks


def shuffled_cells(
    ctx: MatrixContext, fit_ids: list[str], seed: int
) -> dict[tuple[str, str], list[float]]:
    """Permute each model's fit rewards across scenarios: the leak/collapse control."""
    rng = random.Random(seed)
    out = dict(ctx.rewards_cell)
    for model in ctx.model_names:
        keyed = [sid for sid in fit_ids if (sid, model) in ctx.rewards_cell]
        values = [ctx.rewards_cell[(sid, model)] for sid in keyed]
        rng.shuffle(values)
        for sid, value in zip(keyed, values, strict=True):
            out[(sid, model)] = value
    return out


def audit_split(ctx: MatrixContext, fit_ids: list[str], test_ids: list[str]) -> str:
    """Leak audit: id overlap must be zero; report near-dup task-text cosine stats."""
    overlap = set(fit_ids) & set(test_ids)
    if overlap:
        raise AssertionError(f"fit/test id overlap: {sorted(overlap)[:5]}")
    fit_matrix = np.stack([ctx.task_vecs[sid] for sid in fit_ids])
    maxima = [float(np.max(fit_matrix @ ctx.task_vecs[sid])) for sid in test_ids]
    arr = np.asarray(maxima)
    exact = sum(1 for sid in test_ids if any(ctx.tasks[sid] == ctx.tasks[f] for f in fit_ids))
    return (
        f"overlap=0 exact_text_dups={exact} nearest-fit-sim p50={np.median(arr):.3f} "
        f"p95={np.percentile(arr, 95):.3f} n(>0.95)={int((arr > 0.95).sum())} "
        f"n(>0.99)={int((arr > 0.99).sum())}"
    )


def _matrices() -> dict[str, OutcomeMatrix]:
    out: dict[str, OutcomeMatrix] = {}
    wm_matrices: list[tuple[str, OutcomeMatrix]] = []
    for path in sorted((DATA / "matrices").glob("*_matrix.json")):
        corpus = path.stem.removesuffix("_matrix")
        matrix = OutcomeMatrix.load(path)
        if corpus == "routerbench-ours9":
            out[corpus] = matrix
        else:
            out[f"wm-{corpus}"] = matrix
            wm_matrices.append((corpus, matrix))
    if len(wm_matrices) >= 2:
        combined = []
        for corpus, matrix in wm_matrices:
            for outcome in matrix.outcomes:
                combined.append(
                    outcome.model_copy(update={"scenario_id": f"{corpus}:{outcome.scenario_id}"})
                )
        out["wm-all"] = OutcomeMatrix(pool=wm_matrices[0][1].pool, outcomes=combined)
    return out


ROUND1: list[tuple[str, RetrievalParams]] = [
    ("r1-jisi-proxy", RetrievalParams()),
    ("r1-knn", RetrievalParams(second_route=False)),
    ("r1-jisi-noguard", RetrievalParams(guard="none")),
    ("r1-knn-noguard", RetrievalParams(second_route=False, guard="none")),
    ("r1-knn-rag10", RetrievalParams(second_route=False, rag_num=10)),
    ("r1-knn-rag25", RetrievalParams(second_route=False, rag_num=25)),
    ("r1-knn-rag100", RetrievalParams(second_route=False, rag_num=100)),
    ("r1-knn-thres90", RetrievalParams(second_route=False, rag_thres=0.90)),
    ("r1-knn-thres98", RetrievalParams(second_route=False, rag_thres=0.98)),
    ("r1-knn-uniform", RetrievalParams(second_route=False, weighted=False)),
]

ROUND2: list[tuple[str, RetrievalParams]] = [
    ("r1-knn-statz0", RetrievalParams(second_route=False, guard="stat", z=0.0)),
    ("r1-knn-statz05", RetrievalParams(second_route=False, guard="stat", z=0.5)),
    ("r1-knn-statz1", RetrievalParams(second_route=False, guard="stat", z=1.0)),
    (
        "r1-knn-fixk25",
        RetrievalParams(
            second_route=False, guard="stat", z=0.5, neighbor_rule="fixed_k", rag_num=25
        ),
    ),
    (
        "r1-knn-fixk50",
        RetrievalParams(
            second_route=False, guard="stat", z=0.5, neighbor_rule="fixed_k", rag_num=50
        ),
    ),
    (
        "r1-knn-fixk100",
        RetrievalParams(
            second_route=False, guard="stat", z=0.5, neighbor_rule="fixed_k", rag_num=100
        ),
    ),
    (
        "r1-knn-fixk200",
        RetrievalParams(
            second_route=False, guard="stat", z=0.5, neighbor_rule="fixed_k", rag_num=200
        ),
    ),
    (
        "r1-knn-statz05-floor40",
        RetrievalParams(second_route=False, guard="stat", z=0.5, distance_floor=0.40),
    ),
    (
        "r1-knn-statz05-floor50",
        RetrievalParams(second_route=False, guard="stat", z=0.5, distance_floor=0.50),
    ),
]

# The embedder ablation reruns the family's core variants under text-embedding-3-large.
OAI_CORE: list[tuple[str, RetrievalParams]] = [
    ("r1-jisi-proxy-oai", RetrievalParams()),
    ("r1-knn-oai", RetrievalParams(second_route=False)),
    ("r1-knn-noguard-oai", RetrievalParams(second_route=False, guard="none")),
    ("r1-knn-statz05-oai", RetrievalParams(second_route=False, guard="stat", z=0.5)),
]


def _p(**kwargs) -> RetrievalParams:  # noqa: ANN003 (sweep shorthand)
    return RetrievalParams(second_route=False, guard="stat", z=0.5, **kwargs)


# Round 3: exhaust the cheap knobs on ours9 under the semantic embedder (--embed openai).
ROUND3: list[tuple[str, RetrievalParams]] = [
    # Neighbor rule re-tune (0.95 was chosen for compressed hashing sims).
    ("r1-knn3-thres90", _p(rag_thres=0.90)),
    ("r1-knn3-thres93", _p(rag_thres=0.93)),
    ("r1-knn3-thres97", _p(rag_thres=0.97)),
    ("r1-knn3-thres99", _p(rag_thres=0.99)),
    ("r1-knn3-rag25", _p(rag_num=25)),
    ("r1-knn3-rag100", _p(rag_num=100)),
    ("r1-knn3-rag200", _p(rag_num=200)),
    # Guard strictness under the semantic embedder.
    ("r1-knn3-z025", RetrievalParams(second_route=False, guard="stat", z=0.25)),
    ("r1-knn3-z075", RetrievalParams(second_route=False, guard="stat", z=0.75)),
    ("r1-knn3-z1", RetrievalParams(second_route=False, guard="stat", z=1.0)),
    # Weight sharpening (gamma=0 is uniform) and profile smoothing toward the global mean.
    ("r1-knn3-g0", _p(weighted=False)),
    ("r1-knn3-g2", _p(sim_gamma=2.0)),
    ("r1-knn3-g4", _p(sim_gamma=4.0)),
    ("r1-knn3-a2", _p(smooth_alpha=2.0)),
    ("r1-knn3-a5", _p(smooth_alpha=5.0)),
    ("r1-knn3-a10", _p(smooth_alpha=10.0)),
    # Asymmetric guard, alone and as the enabler of the cost knob.
    ("r1-knn3-asym", RetrievalParams(second_route=False, guard="stat_asym", z=0.5)),
    (
        "r1-knn3-asym-lam002",
        RetrievalParams(second_route=False, guard="stat_asym", z=0.5, pick_lam=0.02),
    ),
    (
        "r1-knn3-asym-lam005",
        RetrievalParams(second_route=False, guard="stat_asym", z=0.5, pick_lam=0.05),
    ),
    (
        "r1-knn3-asym-lam01",
        RetrievalParams(second_route=False, guard="stat_asym", z=0.5, pick_lam=0.1),
    ),
]

# Round 3b: post-hoc composition of the individually helpful knobs (labeled as such) plus
# the missing control for the Pareto-extension row.
ROUND3B: list[tuple[str, RetrievalParams]] = [
    (
        "r1-knn3-combo",
        RetrievalParams(second_route=False, guard="stat", z=0.25, smooth_alpha=5.0, sim_gamma=2.0),
    ),
    (
        "r1-knn3-asym-lam002",
        RetrievalParams(second_route=False, guard="stat_asym", z=0.5, pick_lam=0.02),
    ),
]


def run_matrix(
    name: str,
    ctx: MatrixContext,
    split_seed: int,
    *,
    variants: list[tuple[str, RetrievalParams]],
    debug: bool,
    controls: bool,
) -> None:
    matrix = ctx.matrix
    fit_ids, test_ids = split_scenario_ids(matrix, train_fraction=0.7, seed=split_seed)
    audit = audit_split(ctx, fit_ids, test_ids)
    logger.info("%s seed%d audit: %s", name, split_seed, audit)
    best_name, _acc, _cost = best_single_model(matrix, fit_ids=fit_ids, eval_ids=test_ids)
    best_eval = evaluate_choices(matrix, test_ids, lambda _sid: best_name)
    oracle_acc, oracle_cost = oracle(matrix, test_ids)
    ts = datetime.now(tz=UTC).isoformat()

    def record(variant: str, params: dict, result, notes_extra: str = "") -> None:  # noqa: ANN001
        rec = RunRecord(
            run_id=f"{name}-{variant}-{uuid.uuid4().hex[:8]}",
            ts=ts,
            matrix=name,
            variant=variant,
            params=params,
            split_seed=split_seed,
            fit_scenarios=len(fit_ids),
            test_scenarios=len(test_ids),
            result=result,
            baselines={"best_single": best_eval},
            notes=(
                f"best_single={best_name}; oracle acc={oracle_acc:.4f} cost=${oracle_cost:.5f}; "
                f"embedder={ctx.embed_kind}; audit[{audit}]{notes_extra}"
            ),
        )
        append_run(rec, RUNS)
        logger.info(
            "%s/%s seed%d: acc=%.4f (best %.4f) cost=$%.5f (best $%.5f, %+0.1f%%)",
            name,
            variant,
            split_seed,
            result.accuracy,
            best_eval.accuracy,
            result.cost_per_call,
            best_eval.cost_per_call,
            (result.cost_per_call / best_eval.cost_per_call - 1) * 100,
        )

    for variant, params in variants:
        debug_rows: list[dict] | None = [] if debug else None
        picks = route(ctx, params, fit_ids, test_ids, best_name, debug=debug_rows)
        record(
            variant,
            params.model_dump(),
            evaluate_choices(matrix, test_ids, lambda s, p=picks: p[s]),
        )
        if debug_rows is not None:
            DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            out = DEBUG_DIR / f"{name}-seed{split_seed}-{variant}.jsonl"
            with out.open("w", encoding="utf-8") as handle:
                for row in debug_rows:
                    handle.write(json.dumps(row) + "\n")

    if controls:
        cells = shuffled_cells(ctx, fit_ids, seed=split_seed)
        control_names = {
            "r1-jisi-proxy",
            "r1-knn",
            "r1-knn-noguard",
            "r1-knn-statz05",
            "r1-knn3-asym",
            "r1-knn3-asym-lam005",
            "r1-knn3-asym-lam002",
            "r1-knn3-combo",
        }
        for variant, params in [v for v in variants if v[0] in control_names]:
            picks = route(ctx, params, fit_ids, test_ids, best_name, rewards_cell=cells)
            record(
                f"{variant}-shuffled",
                params.model_dump(),
                evaluate_choices(matrix, test_ids, lambda s, p=picks: p[s]),
                notes_extra="; CONTROL shuffled labels (must collapse to ~best-single)",
            )


def main() -> None:
    wanted = [a for a in sys.argv[1:] if not a.startswith("--")]
    seeds = SPLIT_SEEDS if "--seeds" in sys.argv else [0]
    debug = "--debug" in sys.argv
    embed = "openai" if ("--embed" in sys.argv and "openai" in sys.argv) else "hashing"
    if "--round3b" in sys.argv:
        variants = ROUND3B
    elif "--round3" in sys.argv:
        variants = ROUND3
    elif "--round2" in sys.argv:
        variants = ROUND2
    elif embed == "openai":
        variants = OAI_CORE
    else:
        variants = ROUND1
    for name, matrix in _matrices().items():
        if wanted and name not in wanted:
            continue
        ctx = MatrixContext(matrix, name, embed=embed)
        for seed in seeds:
            run_matrix(
                name,
                ctx,
                seed,
                variants=variants,
                debug=debug and seed == 0,
                controls=name == "routerbench-ours9",
            )
    logger.info("runs -> %s", RUNS)


if __name__ == "__main__":
    main()
