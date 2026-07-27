"""Lambda-knob audit: is the fit-once-slide frontier Pareto-clean vs refitting per lambda?

`rerank_policy` (wmo/optimize/routing.py) implements the Hybrid-LLM fit-once-slide-the-knob
property: fit at lambda=0 (guard applied to PURE-reward rankings), then re-sort each cluster's
stored evidence by reward - lambda * cost / cost_scale. A true per-lambda refit instead applies
the ranking AND the only-replace-if-better guard to the penalized objective itself. The
clustering and per-cluster evidence are lambda-free either way, so any frontier gap comes from
guard-vs-knob ordering. This script measures that gap.

Per (matrix, seed): embed fit side, k-means once (champion config), per-cluster per-model
(reward, cost, support). Then per lambda in the grid, build two policies:

- slide: guard on pure reward (exactly `fit_rank_policy` + `rerank_policy(lam)` semantics).
- refit: rank by penalized key, guard winner must beat the baseline's penalized mean by the
  margin (doubled when pricier), same min_support.

Both evaluate through `rank_decision` + `evaluate_choices` on the same held-out side. Reports
per-lambda (acc, cost) for both arms, flags lambda points where either arm strictly dominates
the other, and compares AIQ over a shared max-cost.

Usage: uv run python .agents/scripts/r2_lambda_audit.py [matrix ...] [--seeds=0,1,2,3,4]
Writes findings/r2_lambda_audit.json.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer

from wmo.optimize.outcomes import OutcomeMatrix
from wmo.optimize.policy import ClusterRanking, EmbedderSpec, RoutingPolicy, rank_decision
from wmo.research.routerbench import aiq, best_single_model, split_scenario_ids
from wmo.research.routing_runs import evaluate_choices
from wmo.retrieval.embedders import HashingEmbedder
from wmo.research.routing_corpus import routing_data

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("r2lam")

DATA = routing_data()
DIM = 1024
K = 64
MIN_SUPPORT = 4
MARGIN = 0.03
# Wide grid: the lambda -> operating-point mapping differs between arms (same lambda is NOT
# the same cost), so both arms need enough range to trace their full frontier; comparison is
# hull-vs-hull over the shared cost range, never lambda-matched points.
LAMBDAS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]


def _cluster_stats(
    matrix: OutcomeMatrix, fit_ids: list[str], seed: int
) -> tuple[np.ndarray, dict[int, dict[str, tuple[float, float, int]]], dict[int, str], int]:
    """K-means the fit side (champion config) and gather per-cluster per-model evidence."""
    tasks = {o.scenario_id: o.task for o in matrix.outcomes}
    embedder = HashingEmbedder(dim=DIM)
    vecs = Normalizer(norm="l2").fit_transform(
        np.asarray(embedder.embed([tasks[sid] for sid in fit_ids]))
    )
    k = min(K, len(fit_ids))
    kmeans = KMeans(
        n_clusters=k, random_state=42, init="k-means++", n_init="auto",
        max_iter=1000, algorithm="elkan",
    )
    labels = kmeans.fit_predict(vecs)
    cluster_of = dict(zip(fit_ids, (int(v) for v in labels), strict=True))
    sums: dict[int, dict[str, tuple[float, float, int]]] = {c: {} for c in range(k)}
    prefixes: dict[int, Counter[str]] = {c: Counter() for c in range(k)}
    for o in matrix.outcomes:
        cluster = cluster_of.get(o.scenario_id)
        if cluster is None or o.reward is None:
            continue
        rs, cs, n = sums[cluster].get(o.model, (0.0, 0.0, 0))
        sums[cluster][o.model] = (rs + o.reward, cs + o.cost_usd, n + 1)
        if ":" in o.scenario_id:
            prefixes[cluster][o.scenario_id.split(":", 1)[0]] += 1
    label_of = {
        c: (prefixes[c].most_common(1)[0][0] if prefixes[c] else "") for c in range(k)
    }
    return kmeans.cluster_centers_, sums, label_of, k


def _policy(
    matrix: OutcomeMatrix,
    centres: np.ndarray,
    sums: dict[int, dict[str, tuple[float, float, int]]],
    labels: dict[int, str],
    k: int,
    *,
    lam: float,
    cost_scale: float,
    guard_model: str,
    guard_on_penalized: bool,
    default_model: str,
) -> RoutingPolicy:
    """Build a rank policy with the guard applied before (slide) or on (refit) the penalty."""
    pool_order = {entry.name: index for index, entry in enumerate(matrix.pool)}
    clusters = []
    for cluster in range(k):
        by_model = sums[cluster]
        if not by_model:
            clusters.append(
                ClusterRanking(
                    cluster_id=cluster, label=labels[cluster],
                    centroid=[float(v) for v in centres[cluster]],
                    ranking=[default_model], scores={}, costs={}, total=0,
                )
            )
            continue
        means = {m: rs / n for m, (rs, _cs, n) in by_model.items()}
        mean_costs = {m: cs / n for m, (_rs, cs, n) in by_model.items()}
        supports = {m: n for m, (_rs, _cs, n) in by_model.items()}

        def penalized(model: str, means=means, mean_costs=mean_costs) -> float:  # noqa: ANN001
            return means[model] - lam * mean_costs[model] / cost_scale

        if guard_on_penalized:
            # refit arm: rank AND guard on the penalized objective.
            ranking = sorted(means, key=lambda m: (-penalized(m), pool_order[m]))
            top = ranking[0]
            margin = MARGIN * (
                2 if mean_costs.get(top, 0.0) > mean_costs.get(guard_model, float("inf")) else 1
            )
            base = penalized(guard_model) if guard_model in means else 0.0
            if top != guard_model and (
                supports.get(top, 0) < MIN_SUPPORT or penalized(top) <= base + margin
            ):
                ranking = [guard_model, *[m for m in ranking if m != guard_model]]
        else:
            # slide arm: fit_rank_policy's guard on PURE reward, then rerank_policy's re-sort.
            ranking = sorted(means, key=lambda m: (-means[m], pool_order[m]))
            top = ranking[0]
            margin = MARGIN * (
                2 if mean_costs.get(top, 0.0) > mean_costs.get(guard_model, float("inf")) else 1
            )
            base = means.get(guard_model, 0.0)
            if top != guard_model and (
                supports.get(top, 0) < MIN_SUPPORT or means[top] <= base + margin
            ):
                ranking = [guard_model, *[m for m in ranking if m != guard_model]]
            if lam:
                keyed = {m: penalized(m) for m in ranking if m in means}
                keyed.setdefault(guard_model, 0.0)
                ranking = sorted(ranking, key=lambda m: (-keyed.get(m, 0.0), pool_order[m]))
        clusters.append(
            ClusterRanking(
                cluster_id=cluster, label=labels[cluster],
                centroid=[float(v) for v in centres[cluster]],
                ranking=ranking,
                scores={m: round(v, 6) for m, v in means.items()},
                costs={m: round(v, 8) for m, v in mean_costs.items()},
                total=sum(supports.values()),
            )
        )
    return RoutingPolicy(
        kind="rank", default_model=default_model, pool=matrix.pool,
        embedder=EmbedderSpec(dim=DIM), clusters=clusters, cost_scale=cost_scale,
    )


def audit(name: str, matrix: OutcomeMatrix, seeds: list[int]) -> dict:
    out: dict = {"lambdas": LAMBDAS, "seeds": {}}
    tasks = {o.scenario_id: o.task for o in matrix.outcomes}
    embedder = HashingEmbedder(dim=DIM)
    for seed in seeds:
        fit_ids, test_ids = split_scenario_ids(matrix, train_fraction=0.7, seed=seed)
        best_name, _a, _c = best_single_model(matrix, fit_ids=fit_ids, eval_ids=test_ids)
        centres, sums, labels, k = _cluster_stats(matrix, fit_ids, seed)
        total_cost = sum(
            cs for by_model in sums.values() for (_rs, cs, _n) in by_model.values()
        )
        total_n = sum(n for by_model in sums.values() for (_rs, _cs, n) in by_model.values())
        cost_scale = total_cost / total_n if total_n else 1.0
        test_vecs = Normalizer(norm="l2").transform(
            np.asarray(embedder.embed([tasks[sid] for sid in test_ids]))
        )
        rows = {}
        for arm, guard_on_penalized in (("slide", False), ("refit", True)):
            points = []
            for lam in LAMBDAS:
                policy = _policy(
                    matrix, centres, sums, labels, k,
                    lam=lam, cost_scale=cost_scale, guard_model=best_name,
                    guard_on_penalized=guard_on_penalized, default_model=best_name,
                )
                picks = {
                    sid: rank_decision(policy, test_vecs[row]).model
                    for row, sid in enumerate(test_ids)
                }
                result = evaluate_choices(matrix, test_ids, lambda sid, p=picks: p[sid])
                points.append((result.cost_per_call, result.accuracy))
            rows[arm] = points
        max_cost = max(c for pts in rows.values() for c, _q in pts)
        gap = _hull_gap(rows["slide"], rows["refit"])
        out["seeds"][seed] = {
            "slide": rows["slide"],
            "refit": rows["refit"],
            "aiq_slide": aiq(rows["slide"], max_cost=max_cost),
            "aiq_refit": aiq(rows["refit"], max_cost=max_cost),
            "hull_gap_mean": gap[0],
            "hull_gap_max": gap[1],
            "hull_gap_min": gap[2],
            "best_single": best_name,
        }
        logger.info(
            "%s s%d: AIQ slide %.4f refit %.4f | hull gap (refit-slide) over shared cost "
            "range: mean %+.4f max %+.4f min %+.4f",
            name, seed,
            out["seeds"][seed]["aiq_slide"], out["seeds"][seed]["aiq_refit"],
            gap[0], gap[1], gap[2],
        )
    return out


def _hull_gap(
    slide: list[tuple[float, float]], refit: list[tuple[float, float]]
) -> tuple[float, float, float]:
    """(mean, max, min) of refit_hull(c) - slide_hull(c) over their SHARED cost range."""
    from wmo.research.routerbench import upper_hull

    hull_s, hull_r = upper_hull(slide), upper_hull(refit)

    def interp(hull: list[tuple[float, float]], cost: float) -> float:
        xs = [x for x, _y in hull]
        ys = [y for _x, y in hull]
        return float(np.interp(cost, xs, ys))

    low = max(hull_s[0][0], hull_r[0][0])
    high = min(hull_s[-1][0], hull_r[-1][0])
    if high <= low:
        return 0.0, 0.0, 0.0
    grid = np.linspace(low, high, 200)
    gaps = [interp(hull_r, c) - interp(hull_s, c) for c in grid]
    return float(np.mean(gaps)), float(np.max(gaps)), float(np.min(gaps))


def main() -> None:
    args = sys.argv[1:]
    wanted = [a for a in args if not a.startswith("--")] or ["routerbench-ours9"]
    seeds = [0, 1, 2, 3, 4]
    for arg in args:
        if arg.startswith("--seeds="):
            seeds = [int(s) for s in arg.split("=", 1)[1].split(",")]
    results = {}
    for path in sorted((DATA / "matrices").glob("*_matrix.json")):
        name = path.stem.removesuffix("_matrix")
        if name not in wanted:
            continue
        results[name] = audit(name, OutcomeMatrix.load(path), seeds)
    out = DATA / "findings" / "r2_lambda_audit.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("results -> %s", out)


if __name__ == "__main__":
    main()
