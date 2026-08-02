"""Evaluate the frozen pooled-uplift router on fresh external confirmation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from coding_model_router_swerebench_confirm import (
    ARMS,
    BOOTSTRAP_RESAMPLES,
    MIN_RETAINED_TASKS,
    SOURCE_TASKS,
    ConfirmationData,
    _evaluate_route,
    _load_data,
    _read_object,
    _read_rows,
    _repository_bootstrap_lower,
    _sha256,
)

logger = logging.getLogger("coding-router-pooled-confirmation-analyze")

PROTOCOL = "coding-router-pooled-uplift-confirmation-analysis-v1"
COLLECTION_PROTOCOL = "coding-router-pooled-uplift-confirmation-collection-v1"
SELECTED_CANDIDATE = "direct_ridge-hash8192-a10"
NULL_COUNT = 128
BOOTSTRAP_SEED = 20_260_801
ROUTES_SHA256 = "aac7523746ee9aac0f9789ba9ee4d4e260fad8d2447730102d7aaa44816224c8"
NULL_ROUTES_SHA256 = "4e1570b285eac8da96c13069479f3f7ea9e49b7bedaae8c90d636777a3212a59"
ROUTE_AUDIT_SHA256 = "4f31fe2245cbad1123beada405d18d12b6e323963bde14c3ac69a90993c4db6b"
FREEZE_LOCK_SHA256 = "c8deac37d91912e268108a94a227abae34ab858e3bbbd2c637623697da092751"


def _arm_index(row: dict[str, Any], label: str) -> int:
    arm = row.get("arm")
    if (
        arm not in {"luna-high", "luna-max"}
        or row.get("target_outcomes_used") is not False
    ):
        raise ValueError(f"{label} contains an unsafe route")
    return ARMS.index(str(arm))


def _real_choices(path: Path, data: ConfirmationData) -> np.ndarray:
    rows = _read_rows(path)
    by_task = {row.get("task_id"): row for row in rows}
    if len(rows) != SOURCE_TASKS or len(by_task) != SOURCE_TASKS:
        raise ValueError("real route does not contain 200 unique task decisions")
    choices = np.empty(len(data.task_ids), dtype=np.int64)
    for index, task_id in enumerate(data.task_ids):
        row = by_task.get(task_id)
        if not isinstance(row, dict):
            raise ValueError(f"real route lacks retained task {task_id}")
        choices[index] = _arm_index(row, "real route")
    return choices


def _null_choices(path: Path, data: ConfirmationData) -> np.ndarray:
    rows = _read_rows(path)
    if len(rows) != NULL_COUNT * SOURCE_TASKS:
        raise ValueError("null routes do not contain 128 complete task routes")
    choices = np.empty((NULL_COUNT, len(data.task_ids)), dtype=np.int64)
    for null_index in range(NULL_COUNT):
        block = rows[null_index * SOURCE_TASKS : (null_index + 1) * SOURCE_TASKS]
        by_task = {row.get("task_id"): row for row in block}
        if len(by_task) != SOURCE_TASKS:
            raise ValueError(f"null route {null_index} lacks unique task decisions")
        for task_index, task_id in enumerate(data.task_ids):
            row = by_task.get(task_id)
            if not isinstance(row, dict) or row.get("null_index") != null_index:
                raise ValueError(f"null route {null_index} lacks retained task {task_id}")
            choices[null_index, task_index] = _arm_index(
                row,
                f"null route {null_index}",
            )
    return choices


def _best_null_comparison(
    data: ConfirmationData,
    real_choices: np.ndarray,
    null_choices: np.ndarray,
    *,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, object]:
    """Compare real routing with the outcome-best frozen null route."""
    task_rows = np.arange(len(data.task_ids))
    real_rewards = data.rewards[task_rows, real_choices]
    null_rewards = np.empty(NULL_COUNT, dtype=np.float64)
    null_advantages = np.empty(NULL_COUNT, dtype=np.float64)
    for null_index, choices in enumerate(null_choices):
        traffic = np.bincount(choices, minlength=len(ARMS)) / len(choices)
        routed = data.rewards[task_rows, choices]
        null_rewards[null_index] = float(np.mean(routed))
        null_advantages[null_index] = float(np.mean(routed - data.rewards @ traffic))
    best_index = int(np.argmax(null_rewards))
    best_rewards = data.rewards[task_rows, null_choices[best_index]]
    differences = real_rewards - best_rewards
    difference = float(np.mean(differences))
    lower = _repository_bootstrap_lower(
        differences,
        data.repositories,
        resamples=bootstrap_resamples,
        seed=BOOTSTRAP_SEED,
    )
    return {
        "best_null_index": best_index,
        "best_null_reward": float(null_rewards[best_index]),
        "best_null_matched_blind_advantage": float(null_advantages[best_index]),
        "best_null_arm_counts": {
            ARMS[index]: int(value)
            for index, value in enumerate(
                np.bincount(null_choices[best_index], minlength=len(ARMS))
            )
        },
        "real_minus_best_null_reward": difference,
        "real_minus_best_null_ci95_lower": lower,
        "repository_bootstrap_resamples": bootstrap_resamples,
        "repository_bootstrap_seed": BOOTSTRAP_SEED,
        "passed": difference > 0.0 and lower > 0.0,
    }


def analyze(
    outcomes_path: Path,
    audit_path: Path,
    routes_output: Path,
    output: Path,
) -> None:
    """Apply the preregistered pooled confirmation gates exactly once."""
    if output.exists():
        raise FileExistsError(f"pooled confirmation output already exists: {output}")
    audit = _read_object(audit_path)
    if (
        audit.get("protocol") != COLLECTION_PROTOCOL
        or audit.get("valid") is not True
        or audit.get("deep_swe_outcomes_accessed") is not False
        or audit.get("target_outcomes_used") is not False
        or audit.get("confirmation_authorization_preserved") is not True
        or audit.get("outcomes_sha256") != _sha256(outcomes_path)
    ):
        raise ValueError("pooled confirmation collection audit is invalid")
    data = _load_data(outcomes_path, audit)

    paths = {
        "routes": routes_output / "confirmation-routes.jsonl",
        "null_routes": routes_output / "confirmation-null-routes.jsonl",
        "route_audit": routes_output / "route-audit.json",
        "freeze_lock": routes_output / "freeze-lock.json",
    }
    expected_hashes = {
        "routes": ROUTES_SHA256,
        "null_routes": NULL_ROUTES_SHA256,
        "route_audit": ROUTE_AUDIT_SHA256,
        "freeze_lock": FREEZE_LOCK_SHA256,
    }
    if any(_sha256(paths[label]) != value for label, value in expected_hashes.items()):
        raise ValueError("frozen pooled route artifacts changed")
    launch = audit.get("launch")
    authorization = launch.get("authorization") if isinstance(launch, dict) else None
    if not isinstance(authorization, dict):
        raise ValueError("pooled collection lacks launch authorization")
    if any(
        authorization.get(f"{label}_sha256") != expected
        for label, expected in expected_hashes.items()
    ) or authorization.get("null_route_count") != NULL_COUNT:
        raise ValueError("launch authorization does not match frozen pooled routes")

    route_audit = _read_object(paths["route_audit"])
    freeze = _read_object(paths["freeze_lock"])
    latency_p95 = route_audit.get("route_latency_p95_ms")
    if (
        route_audit.get("valid") is not True
        or route_audit.get("selected_candidate") != SELECTED_CANDIDATE
        or route_audit.get("null_count") != NULL_COUNT
        or route_audit.get("null_unique_route_hashes") != NULL_COUNT
        or not isinstance(latency_p95, (int, float))
        or isinstance(latency_p95, bool)
        or float(latency_p95) >= 5.0
        or route_audit.get("target_outcomes_used") is not False
        or route_audit.get("deep_swe_outcomes_accessed") is not False
        or route_audit.get("internet_access") is not False
        or route_audit.get("fitted_numeric_router_state_persisted") is not False
        or freeze.get("valid") is not True
        or freeze.get("provider_calls_before_freeze") != 0
        or freeze.get("target_outcomes_used") is not False
        or freeze.get("fitted_numeric_router_state_persisted") is not False
    ):
        raise ValueError("pooled route isolation or latency audit is invalid")

    real_choices = _real_choices(paths["routes"], data)
    null_choices = _null_choices(paths["null_routes"], data)
    primary = _evaluate_route(
        data,
        real_choices,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    null_comparison = _best_null_comparison(data, real_choices, null_choices)
    excluded = audit.get("excluded_tasks")
    coverage = (
        isinstance(excluded, list)
        and len(data.task_ids) >= MIN_RETAINED_TASKS
        and len(data.task_ids) + len(excluded) == SOURCE_TASKS
    )
    gates = {
        "coverage_and_dense_gradeability": coverage,
        "router_minus_matched_blind": primary["primary_matched_blind_passed"],
        "router_minus_best_of_128_nulls": null_comparison["passed"],
        "quality_retention": primary["quality_retention_passed"],
        "static_dominance": primary["static_dominance_passed"],
        "latency": float(latency_p95) < 5.0,
        "isolation": True,
    }
    report = {
        "protocol": PROTOCOL,
        "confirmation_passed": all(value is True for value in gates.values()),
        "tasks": len(data.task_ids),
        "repositories": len(set(data.repositories)),
        "gates": gates,
        "primary": primary,
        "best_frozen_null_comparison": null_comparison,
        "selected_candidate": SELECTED_CANDIDATE,
        "route_latency_p95_ms": float(latency_p95),
        "fitted_numeric_router_state_persisted": False,
        "target_outcomes_used": False,
        "deep_swe_outcomes_accessed": False,
        "inputs": {
            "outcomes_sha256": _sha256(outcomes_path),
            "collection_audit_sha256": _sha256(audit_path),
            **{f"{label}_sha256": value for label, value in expected_hashes.items()},
        },
    }
    output.mkdir(parents=True)
    report_path = output / "confirmation-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "pooled confirmation passed=%s tasks=%d advantage=%.6f null_lower=%.6f",
        report["confirmation_passed"],
        len(data.task_ids),
        float(primary["matched_blind_advantage"]),
        float(null_comparison["real_minus_best_null_ci95_lower"]),
    )


def main() -> None:
    """Parse the one-shot pooled confirmation inputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--routes-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.outcomes, args.audit, args.routes_output, args.output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
