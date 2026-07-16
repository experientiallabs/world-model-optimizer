"""Post-hoc correction analysis: selection vs random, on the saved e2e score matrix.

The 3-random-seed comparison in the e2e run was underpowered and the ours-k8 selection had a
script bug (re-clustered the pool at default k instead of reusing the pool's 8 build clusters).
This script recomputes, entirely offline from the saved score matrix:

  1. the exact MAE distribution over 2000 uniform random K=8 subsets;
  2. the CORRECTED ours-k8 (hybrid_select over the pool's true 8 clusters) and its percentile;
  3. a stratified-random control (our allocation + weights, uniform picks within cluster) that
     isolates coverage-allocation effects from medoid-picking bias;
  4. cluster-coverage distributions for every method.

Writes selection_vs_random_correction.json next to the other results.

Usage (from the repo root):
    uv run python .agents/scripts/analyze_selection_vs_random.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / ".agents" / "scripts"))

import numpy as np  # noqa: E402

from run_scenario_e2e import NOVA_LITE, OUT_DIR, REGION  # noqa: E402

from wmh.providers import get_provider  # noqa: E402
from wmh.providers.base import ProviderConfig, ProviderKind  # noqa: E402
from wmh.scenarios.mining.facets import Outcome, TraceFacet  # noqa: E402
from wmh.scenarios.mining.selection import _allocate_slots, hybrid_select  # noqa: E402

DRAWS = 2000
K = 8


def main() -> None:
    res = json.loads((OUT_DIR / "scenario_e2e_results_tau_bench.json").read_text())
    pool = json.loads((OUT_DIR / "scenario_pool_tau_bench.json").read_text())
    matrix = res["score_matrix"]["scores"]
    pool_ids = [s["scenario_id"] for s in pool["scenarios"]]
    cluster_of = {s["scenario_id"]: s["cluster_name"] for s in pool["scenarios"]}
    n_clusters = len(set(cluster_of.values()))
    agents = sorted(matrix)
    actual = {a: float(np.mean([matrix[a][i] for i in pool_ids])) for a in agents}

    def mae(weights: dict[str, float]) -> float:
        total = sum(weights.values())
        return float(
            np.mean(
                [
                    abs(sum(matrix[a][i] * w for i, w in weights.items()) / total - actual[a])
                    for a in agents
                ]
            )
        )

    def coverage(ids: set[str]) -> int:
        return len({cluster_of[i] for i in ids})

    # 1. Uniform random distribution.
    rng = np.random.default_rng(42)
    rand_maes, rand_cov = [], []
    for _ in range(DRAWS):
        ids = rng.choice(pool_ids, size=K, replace=False)
        rand_maes.append(mae({i: 1.0 for i in ids}))
        rand_cov.append(coverage(set(ids)))
    rand_maes_arr = np.asarray(rand_maes)

    # 2. Corrected ours: hybrid_select over the pool's true clusters (embed synthesized tasks).
    emb = get_provider(
        ProviderConfig(kind=ProviderKind.BEDROCK, model=NOVA_LITE, region=REGION, embed_dim=512)
    )
    embeddings = np.asarray([emb.embed([s["task"]])[0] for s in pool["scenarios"]])
    names = sorted(set(cluster_of.values()))
    label_of = {n: i for i, n in enumerate(names)}
    labels = np.asarray([label_of[cluster_of[i]] for i in pool_ids])
    facets = [
        TraceFacet(trace_id=i, task_summary=s["task"], tool_signature="", outcome=Outcome.UNKNOWN)
        for i, s in zip(pool_ids, pool["scenarios"])
    ]
    corrected = hybrid_select(facets, embeddings, labels, K)
    corrected_weights = {s.trace_id: s.weight for s in corrected}
    corrected_mae = mae(corrected_weights)

    # 3. Stratified-random control: same allocation + weights, uniform picks within cluster.
    members: dict[str, list[str]] = {}
    for i in pool_ids:
        members.setdefault(cluster_of[i], []).append(i)
    by_cluster = {j: list(range(len(members[c]))) for j, c in enumerate(sorted(members))}
    slots = _allocate_slots(by_cluster, K, 0.7)
    alloc = {sorted(members)[j]: s for j, s in slots.items()}
    strat_maes = []
    rng2 = np.random.default_rng(7)
    for _ in range(DRAWS):
        weights: dict[str, float] = {}
        for c, k in alloc.items():
            picks = rng2.choice(members[c], size=min(k, len(members[c])), replace=False)
            for p in picks:
                weights[p] = (len(members[c]) / len(pool_ids)) / len(picks)
        strat_maes.append(mae(weights))
    strat_arr = np.asarray(strat_maes)

    buggy_mae = 0.0350  # ours-k8 from the original run (selection over re-clustered pool)
    out = {
        "note": "offline correction on the saved 4-agent x 30-scenario x 3-pass score matrix",
        "uniform_random": {
            "draws": DRAWS,
            "mae_mean": float(rand_maes_arr.mean()),
            "mae_median": float(np.median(rand_maes_arr)),
            "coverage_mean": float(np.mean(rand_cov)),
        },
        "ours_original_run_buggy_reclustering": {
            "mae": buggy_mae,
            "beats_pct_of_random": float((rand_maes_arr > buggy_mae).mean()),
            "clusters_covered": 4,
        },
        "ours_corrected": {
            "mae": corrected_mae,
            "beats_pct_of_random": float((rand_maes_arr > corrected_mae).mean()),
            "clusters_covered": coverage(set(corrected_weights)),
            "selection": [s.model_dump() for s in corrected],
        },
        "stratified_random_control": {
            "draws": DRAWS,
            "mae_mean": float(strat_arr.mean()),
            "mae_median": float(np.median(strat_arr)),
            "allocation": alloc,
        },
        "n_clusters": n_clusters,
        "reading": (
            "For estimating the POOL MEAN, selection is at best random-equivalent; the "
            "coverage-corrected selection is WORSE than random because intent-space medoids "
            "are biased in score space (stratified-random with the same allocation is fine). "
            "The method's value is coverage (7/8 clusters vs ~5/8 random) and auditability, "
            "not mean-score calibration; for calibration use random-within-cluster picks or "
            "the Design-C IRT pass."
        ),
    }
    path = OUT_DIR / "selection_vs_random_correction.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "ours_corrected"}, indent=2)[:800])
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
