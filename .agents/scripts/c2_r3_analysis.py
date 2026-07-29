"""C2 round 3 analysis: the deliverable table + corrected-gate fits per corpus.

Per arm x corpus, corpus-level PAIRED metrics vs the off arm on shared (scenario, model)
cells: quality delta (wm_simulated, labeled), cache-adjusted effective cost per completed
task (compressor bill folded, #294 conventions), p50 episode wall latency (call_seconds
plus compressor latency), steps per episode, achieved keep
(tokens_in_compressed / tokens_in_raw). Then the corrected non-inferiority gate at k=4
clusters (scoped-truncate as the matched-dial control for scoped-llmlingua2-endpoint),
the calibrator-v1 selection over both arms, and the A/A bar on the off arm.

Tau: the scoped arm matrices are UNIONED from chunk files here (the runner's own merge
refuses because chunk 0 ran the 3-model pool and later chunks the 2-model restriction;
rows are rows, and this analysis pairs per cell). Kimi-k3's chunk-0 rows are reported
separately as a labeled partial. The off arm is the jt grid-c2 identity matrix filtered
to the same models (comparability ruling in findings/c2.md). Cohort caveat carried on
every output: cells bought under working tree at tip 4ffc23f0 plus the then-uncommitted
scoped module, which equals aedade4d minus a non-behavioral lint fix.

Usage: uv run python .agents/scripts/c2_r3_analysis.py [tau|terminal]
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from wmo.optimize.compaction_fit import (
    ArmMatrices,
    EvidenceProvenance,
    aa_report,
    assign_to_clusters,
    fit_compaction,
)
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.policy import EmbedderSpec
from wmo.optimize.routing import fit_rank_policy
from wmo.optimize.scorecard import RowOverhead, effective_cost_per_completed_task

R3 = Path.home() / "Desktop/Projects/wmh-compression-data/fits/c2-r3"
JT = Path.home() / "Desktop/Projects/world-model-harness/.wmo/jt/grid-c2"
MODELS = ("gpt-5.4-mini", "sonnet-5")


def union_chunks(arm_dir: Path, models: tuple[str, ...]) -> OutcomeMatrix:
    outcomes: list[ScenarioOutcome] = []
    pool = None
    for chunk in sorted(arm_dir.glob("chunk-*.json")):
        matrix = OutcomeMatrix.load(chunk)
        keep = [entry for entry in matrix.pool if entry.name in models]
        if pool is None or len(keep) < len(pool):
            pool = keep
        outcomes.extend(o for o in matrix.outcomes if o.model in models)
    # Retry results live under retry/ as single-cell OutcomeMatrix files; retried-*.json
    # is just the list of retried cell keys. Fresh scored rows replace their stale cells.
    for retried in sorted((arm_dir / "retry").glob("*.json")) if (arm_dir / "retry").exists() else []:
        matrix = OutcomeMatrix.load(retried)
        fresh = {(o.scenario_id, o.model, o.episode) for o in matrix.outcomes if o.scored}
        outcomes = [
            o for o in outcomes if (o.scenario_id, o.model, o.episode) not in fresh
        ] + [o for o in matrix.outcomes if o.model in models and o.scored]
    return OutcomeMatrix(pool=pool, outcomes=outcomes)


def cells(matrix: OutcomeMatrix) -> dict[tuple[str, str], list[ScenarioOutcome]]:
    by: dict[tuple[str, str], list[ScenarioOutcome]] = {}
    for row in matrix.outcomes:
        if row.scored:
            by.setdefault((row.scenario_id, row.model), []).append(row)
    return by


def paired_metrics(off: OutcomeMatrix, arm: OutcomeMatrix) -> dict[str, object]:
    off_cells, arm_cells = cells(off), cells(arm)
    shared = sorted(set(off_cells) & set(arm_cells))
    off_rows = [r for k in shared for r in off_cells[k]]
    arm_rows = [r for k in shared for r in arm_cells[k]]

    def eff(rows):  # noqa: ANN001, ANN202
        overheads = [
            RowOverhead(
                scenario_id=r.scenario_id, model=r.model, episode=r.episode,
                component="compressor", cost_usd=r.compressor_cost_usd,
                latency_s=r.compressor_latency_s,
            )
            for r in rows if r.compressor_cost_usd > 0 or r.compressor_latency_s > 0
        ]
        return effective_cost_per_completed_task(rows, overheads=overheads)

    def p50_wall(rows):  # noqa: ANN001, ANN202
        walls = sorted(sum(r.call_seconds) + r.compressor_latency_s for r in rows)
        return walls[len(walls) // 2] if walls else 0.0

    diffs = [
        statistics.mean(x.reward for x in arm_cells[k])
        - statistics.mean(x.reward for x in off_cells[k])
        for k in shared
    ]
    n = len(diffs)
    mean = statistics.mean(diffs) if diffs else 0.0
    se = statistics.stdev(diffs) / n**0.5 if n > 1 else 0.0
    off_eff, arm_eff = eff(off_rows), eff(arm_rows)
    keeps = [
        r.tokens_in_compressed / r.tokens_in_raw for r in arm_rows if r.tokens_in_raw > 0
    ]
    return {
        "n_paired_cells": n,
        "quality_delta": round(mean, 4),
        "quality_se": round(se, 4),
        "off_cost_per_completed": off_eff.cost_per_completed_task_usd,
        "arm_cost_per_completed": arm_eff.cost_per_completed_task_usd,
        "cost_delta_pct": round(
            (arm_eff.cost_per_completed_task_usd / off_eff.cost_per_completed_task_usd - 1) * 100, 1
        )
        if off_eff.cost_per_completed_task_usd and arm_eff.cost_per_completed_task_usd
        else None,
        "p50_wall_s_off": round(p50_wall(off_rows), 2),
        "p50_wall_s_arm": round(p50_wall(arm_rows), 2),
        "latency_delta_pct": round((p50_wall(arm_rows) / p50_wall(off_rows) - 1) * 100, 1)
        if p50_wall(off_rows)
        else None,
        "steps_off": round(statistics.mean(r.steps for r in off_rows), 2),
        "steps_arm": round(statistics.mean(r.steps for r in arm_rows), 2),
        "achieved_keep": round(statistics.mean(keeps), 4) if keeps else None,
    }


def main() -> None:
    corpus = sys.argv[1] if len(sys.argv) > 1 else "tau"
    if corpus == "tau":
        off = union_chunks(JT / "identity", MODELS)
        grid = R3 / "grid-tau"
        contrast = {
            "whole-truncate": union_chunks(JT / "truncate", MODELS),
            "whole-llmlingua2": union_chunks(JT / "llmlingua2-endpoint", MODELS),
        }
        era = "tau-wm-simulated-judge-uncondemned-2026-07-28"
    else:
        off = union_chunks(R3 / "grid-terminal/identity", MODELS)
        grid = R3 / "grid-terminal"
        contrast = {}
        era = "terminal-wm-simulated-unaudited-2026-07-28"

    arms = {
        "scoped-llmlingua2-endpoint": union_chunks(grid / "scoped-llmlingua2-endpoint", MODELS),
        "scoped-truncate": union_chunks(grid / "scoped-truncate", MODELS),
    }

    table = {}
    for name, matrix in {**arms, **contrast}.items():
        table[name] = paired_metrics(off, matrix)
        print(f"{corpus} {name}: {json.dumps(table[name])}")

    spec = EmbedderSpec()
    policy = fit_rank_policy(off, embedder=spec, n_clusters=4, seed=42)
    assignment = assign_to_clusters(policy.clusters, off, embed_with=spec.build())
    aa = aa_report(assignment, off)
    provenance = EvidenceProvenance.build(
        off_source=f"{corpus} off arm", arm_sources={}, scorer_era=era
    )
    gated = fit_compaction(
        assignment,
        off,
        [
            ArmMatrices(
                matrix=arms["scoped-llmlingua2-endpoint"],
                config=arms["scoped-llmlingua2-endpoint"].measured_compression(),
            ),
            ArmMatrices(
                matrix=arms["scoped-truncate"],
                config=arms["scoped-truncate"].measured_compression(),
                control=True,
            ),
        ],
        allow_uneven=True,
        provenance=provenance,
    )
    calibrated = fit_compaction(
        assignment,
        off,
        [
            ArmMatrices(matrix=m, config=m.measured_compression())
            for m in arms.values()
        ],
        allow_uneven=True,
        provenance=provenance,
        selection="aggressiveness",
    )
    result = {
        "corpus": corpus,
        "scorer_era": era,
        "cohort_caveat": "cells bought at working-tree 4ffc23f0 + scoped module (= aedade4d minus a non-behavioral lint fix)",
        "table": table,
        "aa_clean_at_ruled_gate": not aa,
        "gated_fit": {
            "clusters_compressed": gated.compressed_clusters(),
            "per_cluster": {str(k): (v.compressor_id if v else None) for k, v in gated.per_cluster.items()},
            "control_flags": gated.control_flags,
        },
        "calibrator_v1_fit": {
            "clusters_compressed": calibrated.compressed_clusters(),
            "per_cluster": {
                str(k): (f"{v.compressor_id}@{v.aggressiveness:g}" if v else None)
                for k, v in calibrated.per_cluster.items()
            },
            "control_flags": calibrated.control_flags,
        },
        "gate_evidence": [
            {k: getattr(r, k) for k in ("cluster_id", "signature", "n_pairs", "mean_diff", "se", "quality_pass", "cost_pass", "chosen", "control_dominated")}
            for r in gated.evidence
        ],
    }
    out = R3 / f"analysis-{corpus}.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"\nAA {'CLEAN' if not aa else 'FIRES'}; gated {gated.compressed_clusters()} compressed, "
          f"flags {gated.control_flags}; calibrator {calibrated.compressed_clusters()} compressed")
    print("wrote", out)


if __name__ == "__main__":
    main()
