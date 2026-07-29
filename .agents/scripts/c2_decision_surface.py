"""C2 overnight menu A: the corrected-gate decision surface on the financebench arms.

Sweeps z x margin x min_pairs over the five financebench-s80 arm matrices under the RULED
non-inferiority gate (mean - z*se >= -margin AND effective-cost win AND no control quality
dominance), reporting per setting: which (cluster, arm) cells pass, what the fitted map
would stamp, whether the A/A bar is clean at that (z, margin), and which control flags
fire. EVERY quality-side number here is PENDING-RESCORE (the matrices' accuracy axis is
ruled unmeasurable until the master's rescore); the object of study is the MACHINERY's
behavior across its knob space, not the corpus verdict.

Run from the c2-r2 worktree:

    uv run python .agents/scripts/c2_decision_surface.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from wmo.optimize.compaction_fit import (
    ArmMatrices,
    EvidenceProvenance,
    aa_report,
    assign_to_clusters,
    fit_compaction,
)
from wmo.optimize.outcomes import OutcomeMatrix
from wmo.optimize.policy import EmbedderSpec
from wmo.optimize.routing import fit_rank_policy

logging.basicConfig(level=logging.WARNING)

MATRICES = Path.home() / "Desktop/Projects/wmh-compression-data/matrices"
OUT = Path.home() / "Desktop/Projects/wmh-compression-data/fits/c2-r2/decision-surface.json"
ARMS = ["llmlingua2-endpoint", "truncate-protect-task"]
CONTROLS = ["control-random", "control-truncate"]
Z_GRID = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5]
MARGIN_GRID = [0.0, 0.01, 0.02, 0.03]
MIN_PAIRS_GRID = [8, 16, 30]


def main() -> None:
    off = OutcomeMatrix.load(MATRICES / "financebench-s80-off_matrix.json")
    arms = []
    for name in ARMS + CONTROLS:
        matrix = OutcomeMatrix.load(MATRICES / f"financebench-s80-{name}_matrix.json")
        arms.append(
            ArmMatrices(
                matrix=matrix,
                config=matrix.measured_compression(),
                control=name in CONTROLS,
            )
        )
    spec = EmbedderSpec()
    policy = fit_rank_policy(off, embedder=spec, n_clusters=8, seed=42)
    assignment = assign_to_clusters(policy.clusters, off, embed_with=spec.build())
    provenance = EvidenceProvenance.build(
        off_source="financebench-s80-off_matrix.json",
        arm_sources={},
        scorer_era="pre-rescore-broken-wm-scorer",
    )

    rows = []
    for z in Z_GRID:
        for margin in MARGIN_GRID:
            aa = aa_report(assignment, off, z=z, margin=margin)
            for min_pairs in MIN_PAIRS_GRID:
                fit = fit_compaction(
                    assignment,
                    off,
                    arms,
                    z=z,
                    margin=margin,
                    min_pairs=min_pairs,
                    allow_uneven=True,
                    provenance=provenance,
                )
                passing = [
                    {
                        "cluster": r.cluster_id,
                        "arm": r.signature,
                        "control": r.control,
                        "mean_diff": round(r.mean_diff, 4),
                        "se": round(r.se, 4),
                        "n": r.n_pairs,
                    }
                    for r in fit.evidence
                    if r.quality_pass and r.cost_pass
                ]
                rows.append(
                    {
                        "z": z,
                        "margin": margin,
                        "min_pairs": min_pairs,
                        "aa_clean": not aa,
                        "aa_deviations": [
                            {"cluster": d.cluster_id, "mean_diff": round(d.mean_diff, 4)}
                            for d in aa
                        ],
                        "clusters_compressed": fit.compressed_clusters(),
                        "stamped": {
                            str(cid): (cfg.compressor_id if cfg else None)
                            for cid, cfg in fit.per_cluster.items()
                        },
                        "gate_passing_cells": passing,
                        "control_flags": fit.control_flags,
                    }
                )
                print(
                    f"z={z:<4} margin={margin:<5} min_pairs={min_pairs:<3} "
                    f"aa={'CLEAN' if not aa else 'FIRES'} "
                    f"compressed={fit.compressed_clusters()} "
                    f"passing_cells={len(passing)} flags={len(fit.control_flags)}"
                )

    OUT.write_text(json.dumps({"label": "PENDING-RESCORE", "rows": rows}, indent=1))
    print(f"\nwrote {len(rows)} settings -> {OUT}")


if __name__ == "__main__":
    main()
