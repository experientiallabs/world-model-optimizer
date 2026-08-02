"""Run and select aggregate-only repository-tree development metrics on remote compute."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from coding_model_router_graded_repository_tree_fit import (
    PROTOCOL,
    align_features,
    evaluate_structure_seed,
    passes_primary_gates,
    structure_grid,
)
from coding_model_router_graded_swerebench_fit import (
    SEEDS,
    _sha256,
    load_data,
)

logger = logging.getLogger("coding-router-graded-repository-tree-run")


def _rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("repository feature input must be a list")
    rows = []
    for row in value:
        if not isinstance(row, dict):
            raise ValueError("repository feature input contains a non-object row")
        rows.append({str(key): item for key, item in row.items()})
    return rows


def run_seed(
    corpus: Path,
    outcomes: Path,
    audit: Path,
    features_path: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    """Run one complete frozen outer seed and retain aggregate metrics only."""
    if seed not in SEEDS:
        raise ValueError("seed is outside the frozen grouped split set")
    data = load_data(corpus, outcomes, audit)
    features = align_features(data, _rows(features_path))
    metrics: list[dict[str, Any]] = []
    grid = structure_grid()
    for index, candidate in enumerate(grid, start=1):
        metrics.extend(evaluate_structure_seed(features, candidate, seed=seed))
        logger.info(
            "seed=%d structures_completed=%d structures_total=%d",
            seed,
            index,
            len(grid),
        )
    if len(metrics) != 3_510 or len({str(row["key"]) for row in metrics}) != 3_510:
        raise RuntimeError("one seed did not emit the complete frozen operating grid")
    return {
        "protocol": PROTOCOL,
        "valid": True,
        "seed": seed,
        "tasks": len(features.data.task_ids),
        "structures": len(grid),
        "operating_points": len(metrics),
        "metrics": metrics,
        "inputs": {
            str(corpus): _sha256(corpus),
            str(outcomes): _sha256(outcomes),
            str(audit): _sha256(audit),
            str(features_path): _sha256(features_path),
        },
        "provider_calls": 0,
        "confirmation_outcomes_accessed": False,
        "deep_swe_outcomes_accessed": False,
        "fitted_numeric_state_persisted": False,
        "feature_rows_persisted_outside_worker": False,
        "outcome_matrix_persisted_outside_worker": False,
    }


def select_primary(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate five complete seed reports and apply the pointwise primary gates."""
    if len(seed_reports) != len(SEEDS):
        raise ValueError("primary selection requires exactly five seed reports")
    by_seed = {int(report.get("seed", -1)): report for report in seed_reports}
    if set(by_seed) != set(SEEDS):
        raise ValueError("primary seed reports are incomplete or duplicated")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    input_hashes: dict[str, str] | None = None
    tasks: int | None = None
    for seed in SEEDS:
        report = by_seed[seed]
        metrics = report.get("metrics")
        if (
            report.get("protocol") != PROTOCOL
            or report.get("valid") is not True
            or report.get("operating_points") != 3_510
            or report.get("structures") != 270
            or report.get("provider_calls") != 0
            or not isinstance(metrics, list)
            or len(metrics) != 3_510
        ):
            raise ValueError("one primary seed report is incomplete or unsafe")
        report_inputs = report.get("inputs")
        report_tasks = report.get("tasks")
        if not isinstance(report_inputs, dict) or not isinstance(report_tasks, int):
            raise ValueError("one primary seed report lacks input identity")
        normalized_inputs = {str(key): str(value) for key, value in report_inputs.items()}
        if input_hashes is None:
            input_hashes = normalized_inputs
            tasks = report_tasks
        elif input_hashes != normalized_inputs or tasks != report_tasks:
            raise ValueError("primary seed reports used different inputs")
        for row in metrics:
            if not isinstance(row, dict) or row.get("seed") != seed:
                raise ValueError("one primary metric row is malformed")
            grouped[str(row["key"])].append(row)
    if len(grouped) != 3_510 or any(len(rows) != len(SEEDS) for rows in grouped.values()):
        raise ValueError("primary reports do not cover the same operating grid")
    eligible = [key for key, rows in grouped.items() if passes_primary_gates(rows)]
    summaries = []
    for key, rows in grouped.items():
        summaries.append(
            {
                "key": key,
                "worst_quality_retention": min(
                    float(row["quality_retention"]) for row in rows
                ),
                "worst_cost_savings": min(float(row["cost_savings"]) for row in rows),
                "worst_matched_blind_advantage": min(
                    float(row["matched_blind_advantage"]) for row in rows
                ),
                "primary_eligible": key in eligible,
            }
        )
    closest_quality = max(
        summaries,
        key=lambda row: (
            float(row["worst_quality_retention"]),
            float(row["worst_cost_savings"]),
            str(row["key"]),
        ),
    )
    closest_savings_at_quality = max(
        (
            row
            for row in summaries
            if float(row["worst_quality_retention"]) >= 0.95
        ),
        key=lambda row: (float(row["worst_cost_savings"]), str(row["key"])),
        default=None,
    )
    return {
        "protocol": f"{PROTOCOL}-primary-selection-v1",
        "valid": True,
        "tasks": tasks,
        "candidate_count": len(grouped),
        "primary_eligible_count": len(eligible),
        "primary_eligible_keys": sorted(eligible),
        "closest_quality": closest_quality,
        "closest_savings_at_quality": closest_savings_at_quality,
        "requires_family_null": bool(eligible),
        "development_passed": False,
        "confirmation_routes_frozen": False,
        "provider_calls": 0,
        "confirmation_outcomes_accessed": False,
        "deep_swe_outcomes_accessed": False,
        "fitted_numeric_state_persisted": False,
        "feature_rows_persisted": False,
        "outcome_matrix_persisted": False,
        "input_sha256": input_hashes,
    }


def main() -> None:
    """Run one seed or select five complete aggregate reports."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    seed_parser = subparsers.add_parser("seed")
    seed_parser.add_argument("--corpus", type=Path, required=True)
    seed_parser.add_argument("--outcomes", type=Path, required=True)
    seed_parser.add_argument("--audit", type=Path, required=True)
    seed_parser.add_argument("--features", type=Path, required=True)
    seed_parser.add_argument("--seed", type=int, required=True)
    seed_parser.add_argument("--output", type=Path, required=True)
    select_parser = subparsers.add_parser("select-primary")
    select_parser.add_argument("--seed-report", type=Path, action="append", required=True)
    select_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "seed":
        report = run_seed(
            args.corpus,
            args.outcomes,
            args.audit,
            args.features,
            seed=args.seed,
        )
    else:
        reports = [
            json.loads(path.read_text(encoding="utf-8")) for path in args.seed_report
        ]
        if any(not isinstance(report, dict) for report in reports):
            raise ValueError("seed report input contains a non-object")
        report = select_primary(reports)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
