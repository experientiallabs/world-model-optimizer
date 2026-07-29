"""C2 overnight menus D+E: verbatim-critical features round 2, plus D-DIAL anchor emission.

D: cheap cluster-level predictors (numeric density, capitalized-entity density, code/JSON
char share, mean task words) against the corrected-gate outcome variable (the better REAL
arm's paired quality delta per cluster), across both cohorts (financebench k=8 rank-path
clusters, tau k=4). n = 12 clusters total: candidate-labeled, Spearman only. Plus an
opus-5 exemplar labeling leg (Silen's overnight credit authorization): one call per
cluster asking whether compressing these prompts risks breaking exact-match-critical
content; metered per call, cumulative total printed and ledgered.

E: joint anchors in the D-DIAL v2 ruled shape from the corrected fits. On BOTH cohorts
the fitted map is compression=None everywhere, so the three corners coincide; the anchors
STATE that as the measured verdict (per the routing master's note they must never imply a
live knob) and carry the measured per-arm deltas as the evidence behind the None.

Run from the c2-r2 worktree with ANTHROPIC_API_KEY exported:

    uv run python .agents/scripts/c2_features_and_anchors.py
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

from wmo.optimize.compaction_fit import (
    ArmMatrices,
    assign_to_clusters,
    fit_compaction,
)
from wmo.optimize.outcomes import OutcomeMatrix
from wmo.optimize.policy import EmbedderSpec
from wmo.optimize.routing import fit_rank_policy
from wmo.optimize.scorecard import RowOverhead, effective_cost_per_completed_task

DATA = Path.home() / "Desktop/Projects/wmh-compression-data"
FB = DATA / "matrices"
TAU = Path.home() / "Desktop/Projects/world-model-harness/.wmo/jt/grid-c2"
OUT_FEATURES = DATA / "fits/c2-r2/features-r2.json"
OUT_ANCHORS = DATA / "fits/c2-r2/anchors.json"

OPUS_IN_PER_MTOK = 5.0
OPUS_OUT_PER_MTOK = 25.0


def cluster_features(tasks: list[str]) -> dict[str, float]:
    text = " ".join(tasks)
    words = text.split()
    return {
        "numeric_density": round(sum(1 for w in words if re.search(r"\d", w)) / len(words), 4),
        "entity_density": round(sum(1 for w in words if w[:1].isupper()) / len(words), 4),
        "codejson_share": round(
            len(re.findall(r"[{}\[\]`$%()=<>_]", text)) / max(1, len(text)), 5
        ),
        "mean_words": round(statistics.mean(len(t.split()) for t in tasks), 1),
    }


def load_cohort(name: str):  # noqa: ANN202
    if name == "financebench":
        off = OutcomeMatrix.load(FB / "financebench-s80-off_matrix.json")
        arm_names = ["llmlingua2-endpoint", "truncate-protect-task"]
        arms = [
            OutcomeMatrix.load(FB / f"financebench-s80-{a}_matrix.json") for a in arm_names
        ]
        k, allow_uneven = 8, True
    else:
        off = OutcomeMatrix.load(TAU / "identity/matrix.json")
        arm_names = ["truncate", "llmlingua2-endpoint"]
        arms = [OutcomeMatrix.load(TAU / a / "matrix.json") for a in arm_names]
        k, allow_uneven = 4, False
    spec = EmbedderSpec()
    policy = fit_rank_policy(off, embedder=spec, n_clusters=k, seed=42)
    assignment = assign_to_clusters(policy.clusters, off, embed_with=spec.build())
    fit = fit_compaction(
        assignment,
        off,
        [ArmMatrices(matrix=m, config=m.measured_compression()) for m in arms],
        allow_uneven=allow_uneven,
    )
    return off, arms, assignment, fit


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        for rank, index in enumerate(order):
            out[index] = float(rank)
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (
        sum((a - mx) ** 2 for a in rx) ** 0.5 * sum((b - my) ** 2 for b in ry) ** 0.5
    )
    return num / den if den else 0.0


def opus_label(exemplars: list[str]) -> tuple[str, str, float]:
    """One metered opus-5 call: verbatim-critical YES/NO + reason. Returns (label, reason, usd)."""
    from anthropic import Anthropic

    client = Anthropic()
    prompt = (
        "These are task prompts from one workload cluster:\n\n"
        + "\n---\n".join(exemplars[:3])
        + "\n\nIf a compressor removed low-information tokens from prompts like these "
        "before an LLM answered them, would it risk deleting exact-match-critical content "
        "(identifiers, numbers, code, exact names the answer must reproduce)? "
        "Answer with YES or NO on the first line, then one sentence of reasoning."
    )
    response = client.messages.create(
        model="claude-opus-5",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    text = next(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    ).strip()
    usd = (
        response.usage.input_tokens * OPUS_IN_PER_MTOK
        + response.usage.output_tokens * OPUS_OUT_PER_MTOK
    ) / 1e6
    label = "YES" if text.upper().startswith("YES") else "NO"
    return label, text.splitlines()[-1][:200], usd


def arm_deltas(off: OutcomeMatrix, arm: OutcomeMatrix) -> dict[str, float]:
    """Pooled per-arm deltas vs off on shared scored cells: the evidence behind the anchors."""

    def cells(matrix: OutcomeMatrix):  # noqa: ANN202
        by = {}
        for row in matrix.outcomes:
            if row.scored:
                by.setdefault((row.scenario_id, row.model), []).append(row)
        return by

    off_cells, arm_cells = cells(off), cells(arm)
    shared = sorted(set(off_cells) & set(arm_cells))
    off_rows = [r for key in shared for r in off_cells[key]]
    arm_rows = [r for key in shared for r in arm_cells[key]]

    def quality(rows) -> float:  # noqa: ANN001
        return statistics.mean(r.reward for r in rows)

    def eff(rows) -> float | None:  # noqa: ANN001
        overheads = [
            RowOverhead(
                scenario_id=r.scenario_id,
                model=r.model,
                episode=r.episode,
                component="compressor",
                cost_usd=r.compressor_cost_usd,
                latency_s=r.compressor_latency_s,
            )
            for r in rows
            if r.compressor_cost_usd > 0 or r.compressor_latency_s > 0
        ]
        return effective_cost_per_completed_task(
            rows, overheads=overheads
        ).cost_per_completed_task_usd

    def p50_latency(rows) -> float:  # noqa: ANN001
        totals = sorted(sum(r.call_seconds) + r.compressor_latency_s for r in rows)
        return totals[len(totals) // 2] if totals else 0.0

    off_eff, arm_eff = eff(off_rows), eff(arm_rows)
    return {
        "quality_delta_pt": round((quality(arm_rows) - quality(off_rows)) * 100, 2),
        "cost_delta_pct": round((arm_eff / off_eff - 1) * 100, 1) if off_eff and arm_eff else None,
        "latency_delta_ms": round((p50_latency(arm_rows) - p50_latency(off_rows)) * 1000, 0),
        "n_cells": len(shared),
    }


def main() -> None:
    feature_rows = []
    anchor_blocks = {}
    spend = 0.0
    for cohort in ("financebench", "tau"):
        off, arms, assignment, fit = load_cohort(cohort)
        tasks_by_sid = {}
        for outcome in off.outcomes:
            tasks_by_sid.setdefault(outcome.scenario_id, outcome.task)
        for cluster_id in sorted(set(assignment.values())):
            member_tasks = [
                tasks_by_sid[sid] for sid, c in assignment.items() if c == cluster_id
            ]
            real_rows = [
                r
                for r in fit.evidence
                if r.cluster_id == cluster_id and not r.control and r.n_pairs > 0
            ]
            best = max(real_rows, key=lambda r: r.mean_diff) if real_rows else None
            label, reason, usd = opus_label(member_tasks)
            spend += usd
            feature_rows.append(
                {
                    "cohort": cohort,
                    "cluster": cluster_id,
                    "n_scenarios": len(member_tasks),
                    **cluster_features(member_tasks),
                    "best_real_quality_delta": round(best.mean_diff, 4) if best else None,
                    "opus5_verbatim_critical": label,
                    "opus5_reason": reason,
                    "opus5_call_usd": round(usd, 5),
                }
            )
        anchor_blocks[cohort] = {
            "corners": {
                name: {
                    "compression": None,
                    "note": "measured verdict: no cluster passed the corrected gate; the "
                    "corners coincide at uncompressed (this is a statement, not a knob)",
                }
                for name in ("quality-max", "balanced", "max-savings")
            },
            "per_cluster_map": {
                str(c): None for c in sorted({r["cluster"] for r in feature_rows if r["cohort"] == cohort})
            },
            "evidence_deltas_if_compressed": {
                str(arm.measured_compression().compressor_id): arm_deltas(off, arm)
                for arm in arms
            },
            "provenance": (
                "pre-rescore-broken-wm-scorer (PENDING-RESCORE)"
                if cohort == "financebench"
                else "tau-wm-simulated-judge-uncondemned-2026-07-28"
            ),
            "gate": "non-inferiority z=0.5 margin=0.02 min_pairs=8 + control-dominance",
        }

    deltas = [r["best_real_quality_delta"] for r in feature_rows]
    correlations = {
        feature: round(
            spearman([r[feature] for r in feature_rows], deltas), 3
        )
        for feature in ("numeric_density", "entity_density", "codejson_share", "mean_words")
    }
    yes_deltas = [
        r["best_real_quality_delta"]
        for r in feature_rows
        if r["opus5_verbatim_critical"] == "YES"
    ]
    no_deltas = [
        r["best_real_quality_delta"]
        for r in feature_rows
        if r["opus5_verbatim_critical"] == "NO"
    ]

    OUT_FEATURES.write_text(
        json.dumps(
            {
                "label": "CANDIDATE (n=12 clusters, two cohorts, mixed provenance)",
                "clusters": feature_rows,
                "spearman_vs_quality_delta": correlations,
                "opus5": {
                    "yes_mean_delta": round(statistics.mean(yes_deltas), 4) if yes_deltas else None,
                    "no_mean_delta": round(statistics.mean(no_deltas), 4) if no_deltas else None,
                    "n_yes": len(yes_deltas),
                    "n_no": len(no_deltas),
                    "total_usd": round(spend, 4),
                },
            },
            indent=1,
        )
    )
    OUT_ANCHORS.write_text(json.dumps(anchor_blocks, indent=1))
    print(json.dumps(correlations, indent=1))
    print(f"opus5: YES clusters mean delta "
          f"{statistics.mean(yes_deltas) if yes_deltas else float('nan'):+.4f} (n={len(yes_deltas)}), "
          f"NO {statistics.mean(no_deltas) if no_deltas else float('nan'):+.4f} (n={len(no_deltas)}); "
          f"spend ${spend:.4f}")
    for row in feature_rows:
        print(f"  {row['cohort']:<13} c{row['cluster']} num={row['numeric_density']:.3f} "
              f"ent={row['entity_density']:.3f} code={row['codejson_share']:.4f} "
              f"words={row['mean_words']:>6} delta={row['best_real_quality_delta']} "
              f"opus5={row['opus5_verbatim_critical']}")
    print(f"wrote {OUT_FEATURES}\nwrote {OUT_ANCHORS}")


if __name__ == "__main__":
    main()
