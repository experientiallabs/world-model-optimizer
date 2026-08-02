"""Freeze the real and null pooled-uplift confirmation routes on E2B."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from coding_model_router_pooled_uplift_fit import (
    ATTEMPTS,
    AUDIT_HASHES,
    HIGH_INDEX,
    MAX_INDEX,
    NULL_COUNT,
    OUTCOME_HASHES,
    SEED,
    TASK_HASHES,
    Candidate,
    PooledData,
    _block_permutations,
    _candidate_grid,
    _features,
    _fit_model,
    _fit_prior,
    _latency_p95_ms,
    _load_cohort,
    _load_external,
    _matrix,
    _pool,
    _predict,
    _route,
)
from coding_model_router_swerebench_fit import ARMS
from coding_model_router_swesmith_null_fit import ScorerCandidate

logger = logging.getLogger("coding-router-pooled-confirmation-freeze")

PROTOCOL = "coding-router-pooled-uplift-confirmation-route-freeze-v1"
DEVELOPMENT_REPORT_SHA256 = "d168e721a97782915991f7aa92971bc97cf7feb7705242361143d4ce3358bef9"
SELECTION_LOCK_SHA256 = "43e87a80e286ff98f2e1afda4e5db8dbdf231f55c7b458fa27e824f5bbbf8e4e"
CONFIRMATION_TASKS_SHA256 = "6edd8ed4777d6bc48cf29f76a9fb4b9d60e3324908aa79d4d03df8617f6be825"
CONFIRMATION_MANIFEST_SHA256 = "7bd743a794c5054e053a9d163c088d0f9f72fbd911043c44f90b792801eade60"
SELECTED_KEY = "direct_ridge-hash8192-a10"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return {str(key): item for key, item in value.items()}


def _selected_candidate() -> Candidate:
    return next(candidate for candidate in _candidate_grid() if candidate.key == SELECTED_KEY)


def _write_routes(
    path: Path,
    tasks: list[dict[str, object]],
    choices: np.ndarray,
    *,
    null_index: int | None,
) -> None:
    rows = []
    for index, task in enumerate(tasks):
        row: dict[str, object] = {
            "task_id": str(task["task_id"]),
            "repository": str(task["repository"]),
            "language": str(task["language"]),
            "arm": ARMS[int(choices[index])],
            "target_outcomes_used": False,
        }
        if null_index is not None:
            row["null_index"] = null_index
        rows.append(json.dumps(row, sort_keys=True) + "\n")
    with path.open("a", encoding="utf-8") as handle:
        handle.writelines(rows)


def freeze(
    task_paths: tuple[Path, Path],
    outcome_paths: tuple[Path, Path],
    audit_paths: tuple[Path, Path],
    external_tasks_path: Path,
    external_manifest_path: Path,
    development_report_path: Path,
    selection_lock_path: Path,
    confirmation_tasks_path: Path,
    confirmation_manifest_path: Path,
    output: Path,
) -> None:
    """Refit ephemerally and freeze confirmation decisions without outcomes."""
    if output.exists():
        raise FileExistsError(f"route freeze output already exists: {output}")
    if _sha256(development_report_path) != DEVELOPMENT_REPORT_SHA256:
        raise ValueError("development report hash changed")
    if _sha256(selection_lock_path) != SELECTION_LOCK_SHA256:
        raise ValueError("selection lock hash changed")
    if _sha256(confirmation_tasks_path) != CONFIRMATION_TASKS_SHA256:
        raise ValueError("confirmation task hash changed")
    if _sha256(confirmation_manifest_path) != CONFIRMATION_MANIFEST_SHA256:
        raise ValueError("confirmation manifest hash changed")
    report = _read_object(development_report_path)
    lock = _read_object(selection_lock_path)
    manifest = _read_object(confirmation_manifest_path)
    if (
        report.get("development_passed") is not True
        or (report.get("selected") or {}).get("key") != SELECTED_KEY
        or lock.get("eligible") is not True
        or lock.get("candidate") != SELECTED_KEY
        or manifest.get("valid") is not True
        or manifest.get("confirmation_tasks_sha256") != CONFIRMATION_TASKS_SHA256
        or manifest.get("target_reward_fields_accessed") is not False
        or manifest.get("target_cost_fields_accessed") is not False
    ):
        raise ValueError("route freeze inputs are incomplete or unsafe")
    threshold = lock.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("selection lock has no numeric threshold")
    cohorts = [
        _load_cohort(
            task_paths[index],
            outcome_paths[index],
            audit_paths[index],
            expected_tasks_hash=TASK_HASHES[index],
            expected_outcomes_hash=OUTCOME_HASHES[index],
            expected_audit_hash=AUDIT_HASHES[index],
        )
        for index in range(2)
    ]
    development = _pool(cohorts)
    external = _load_external(external_tasks_path, external_manifest_path)
    prior = _fit_prior(external, ScorerCandidate(order=8, dim=8_192, alpha=10.0))
    candidate = _selected_candidate()
    development_bank = _features(development, prior)
    development_matrix = _matrix(development_bank, candidate)
    indices = np.arange(len(development.task_ids), dtype=np.int64)
    fitted = _fit_model(
        candidate,
        development_matrix,
        indices,
        development.rewards[:, HIGH_INDEX, :].mean(axis=1),
        development.rewards[:, MAX_INDEX, :].mean(axis=1),
    )
    raw_tasks = _read_object(confirmation_tasks_path).get("tasks")
    if not isinstance(raw_tasks, list) or len(raw_tasks) != 200:
        raise ValueError("confirmation route corpus must contain 200 tasks")
    tasks = [
        {str(key): item for key, item in task.items()}
        for task in raw_tasks
        if isinstance(task, dict)
    ]
    if len(tasks) != 200:
        raise ValueError("confirmation route corpus contains a non-object task")
    confirmation = PooledData(
        task_ids=[str(task["task_id"]) for task in tasks],
        repositories=[str(task["repository"]) for task in tasks],
        languages=[str(task["language"]) for task in tasks],
        texts=[
            f"repository={task['repository']}\n{task['prompt']}" for task in tasks
        ],
        rewards=np.zeros((len(tasks), len(ARMS), ATTEMPTS), dtype=np.float64),
        costs=np.zeros((len(tasks), len(ARMS), ATTEMPTS), dtype=np.float64),
    )
    confirmation_bank = _features(confirmation, prior)
    confirmation_matrix = _matrix(confirmation_bank, candidate)
    confirmation_indices = np.arange(len(tasks), dtype=np.int64)
    scores = _predict(fitted, confirmation_matrix, confirmation_indices)
    real_choices = _route(scores, float(threshold))
    null_scores = _block_permutations(
        confirmation,
        confirmation_indices,
        scores,
        seed=SEED,
    )
    null_choices = np.asarray(
        [_route(row, float(threshold)) for row in null_scores],
        dtype=np.int64,
    )
    latency_p95, latency_samples = _latency_p95_ms(
        confirmation.texts,
        prior,
        fitted,
    )
    if latency_p95 >= 5.0:
        raise ValueError(f"confirmation route latency exceeded 5 ms: {latency_p95}")
    output.mkdir(parents=True)
    real_path = output / "confirmation-routes.jsonl"
    null_path = output / "confirmation-null-routes.jsonl"
    _write_routes(real_path, tasks, real_choices, null_index=None)
    for null_index in range(NULL_COUNT):
        _write_routes(
            null_path,
            tasks,
            null_choices[null_index],
            null_index=null_index,
        )
    null_route_hashes = [
        hashlib.sha256(row.tobytes()).hexdigest() for row in null_choices
    ]
    audit = {
        "protocol": PROTOCOL,
        "valid": True,
        "selected_candidate": SELECTED_KEY,
        "threshold": float(threshold),
        "confirmation_tasks": len(tasks),
        "confirmation_repositories": len(set(confirmation.repositories)),
        "real_arm_counts": dict(
            sorted(Counter(ARMS[int(value)] for value in real_choices).items())
        ),
        "null_count": NULL_COUNT,
        "null_unique_route_hashes": len(set(null_route_hashes)),
        "null_route_decision_sha256": null_route_hashes,
        "route_latency_p95_ms": latency_p95,
        "route_latency_samples": latency_samples,
        "development_report_sha256": _sha256(development_report_path),
        "selection_lock_sha256": _sha256(selection_lock_path),
        "confirmation_tasks_sha256": _sha256(confirmation_tasks_path),
        "confirmation_manifest_sha256": _sha256(confirmation_manifest_path),
        "external_tasks_sha256": _sha256(external_tasks_path),
        "external_manifest_sha256": _sha256(external_manifest_path),
        "confirmation_routes_sha256": _sha256(real_path),
        "confirmation_null_routes_sha256": _sha256(null_path),
        "target_outcomes_used": False,
        "deep_swe_outcomes_accessed": False,
        "internet_access": False,
        "fitted_numeric_router_state_persisted": False,
    }
    audit_path = output / "route-audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    freeze_lock = {
        "protocol": PROTOCOL,
        "valid": True,
        "selected_candidate": SELECTED_KEY,
        "threshold": float(threshold),
        "confirmation_tasks_sha256": audit["confirmation_tasks_sha256"],
        "confirmation_routes_sha256": audit["confirmation_routes_sha256"],
        "confirmation_null_routes_sha256": audit["confirmation_null_routes_sha256"],
        "route_audit_sha256": _sha256(audit_path),
        "provider_calls_before_freeze": 0,
        "target_outcomes_used": False,
        "fitted_numeric_router_state_persisted": False,
    }
    (output / "freeze-lock.json").write_text(
        json.dumps(freeze_lock, indent=2, sort_keys=True) + "\n"
    )
    logger.info(
        "routes frozen real=%s null=%d unique_null=%d latency_p95_ms=%.3f",
        audit["real_arm_counts"],
        NULL_COUNT,
        audit["null_unique_route_hashes"],
        latency_p95,
    )


def main() -> None:
    """Run the confirmation route-freeze CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-tasks", type=Path, required=True)
    parser.add_argument("--development-outcomes", type=Path, required=True)
    parser.add_argument("--development-audit", type=Path, required=True)
    parser.add_argument("--old-confirmation-tasks", type=Path, required=True)
    parser.add_argument("--old-confirmation-outcomes", type=Path, required=True)
    parser.add_argument("--old-confirmation-audit", type=Path, required=True)
    parser.add_argument("--external-tasks", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--confirmation-tasks", type=Path, required=True)
    parser.add_argument("--confirmation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(
        (args.development_tasks, args.old_confirmation_tasks),
        (args.development_outcomes, args.old_confirmation_outcomes),
        (args.development_audit, args.old_confirmation_audit),
        args.external_tasks,
        args.external_manifest,
        args.development_report,
        args.selection_lock,
        args.confirmation_tasks,
        args.confirmation_manifest,
        args.output,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
