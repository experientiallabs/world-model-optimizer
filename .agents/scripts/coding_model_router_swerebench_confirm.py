"""Evaluate one frozen SWE-rebench route on untouched external confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("coding-router-swerebench-confirm")

PROTOCOL = "coding-router-swerebench-confirmation-analysis-v1"
COLLECTION_PROTOCOL = "coding-router-swerebench-confirmation-collection-v1"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
ARMS = tuple(f"luna-{effort}" for effort in EFFORTS)
ATTEMPTS = 2
SOURCE_TASKS = 200
MIN_RETAINED_TASKS = 190
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_731
QUALITY_RETENTION = 0.95


@dataclass(frozen=True)
class ConfirmationData:
    """Dense task-level confirmation rewards and costs."""

    task_ids: list[str]
    repositories: list[str]
    rewards: np.ndarray
    costs: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def _load_data(outcomes_path: Path, audit: dict[str, Any]) -> ConfirmationData:
    rows = _read_rows(outcomes_path)
    tasks = audit.get("tasks")
    if (
        not isinstance(tasks, int)
        or tasks < MIN_RETAINED_TASKS
        or tasks > SOURCE_TASKS
        or len(rows) != tasks * len(EFFORTS) * ATTEMPTS
    ):
        raise ValueError("confirmation outcomes do not meet the retained coverage gate")
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        task_id = row.get("task_id")
        effort = row.get("reasoning_effort")
        attempt = row.get("attempt_number")
        if (
            not isinstance(task_id, str)
            or effort not in EFFORTS
            or isinstance(attempt, bool)
            or not isinstance(attempt, int)
            or attempt not in range(ATTEMPTS)
            or row.get("target_outcomes_used") is not False
        ):
            raise ValueError("confirmation outcome identity or isolation flag is invalid")
        by_task.setdefault(task_id, []).append(row)
    if len(by_task) != tasks:
        raise ValueError("confirmation outcome task identities are not unique")
    task_ids = list(by_task)
    repositories: list[str] = []
    rewards = np.empty((tasks, len(EFFORTS)), dtype=np.float64)
    costs = np.empty_like(rewards)
    for task_index, task_id in enumerate(task_ids):
        task_rows = by_task[task_id]
        repository_values = {row.get("repository") for row in task_rows}
        if len(repository_values) != 1 or not isinstance(next(iter(repository_values)), str):
            raise ValueError(f"task {task_id} has inconsistent repository identity")
        repositories.append(str(next(iter(repository_values))))
        identities = {
            (str(row["reasoning_effort"]), int(row["attempt_number"]))
            for row in task_rows
        }
        expected = {(effort, attempt) for effort in EFFORTS for attempt in range(ATTEMPTS)}
        if identities != expected:
            raise ValueError(f"task {task_id} does not have a dense effort matrix")
        for effort_index, effort in enumerate(EFFORTS):
            effort_rows = [row for row in task_rows if row["reasoning_effort"] == effort]
            effort_rewards = [float(row["reward"]) for row in effort_rows]
            effort_costs = [float(row["cost_usd"]) for row in effort_rows]
            if (
                any(value not in {0.0, 1.0} for value in effort_rewards)
                or any(not np.isfinite(value) or value < 0.0 for value in effort_costs)
            ):
                raise ValueError(f"task {task_id} has invalid reward or cost")
            rewards[task_index, effort_index] = float(np.mean(effort_rewards))
            costs[task_index, effort_index] = float(np.mean(effort_costs))
    return ConfirmationData(task_ids, repositories, rewards, costs)


def _choices(path: Path, data: ConfirmationData) -> np.ndarray:
    rows = _read_rows(path)
    by_task = {row.get("task_id"): row for row in rows}
    if len(rows) != SOURCE_TASKS or len(by_task) != SOURCE_TASKS:
        raise ValueError(f"{path} does not contain 200 unique frozen routes")
    choices = np.empty(len(data.task_ids), dtype=np.int64)
    for index, task_id in enumerate(data.task_ids):
        row = by_task.get(task_id)
        if not isinstance(row, dict):
            raise ValueError(f"{path} lacks retained task {task_id}")
        effort = row.get("reasoning_effort")
        if effort not in EFFORTS:
            raise ValueError(f"{path} has an invalid effort for {task_id}")
        if (
            row.get("target_outcomes_used") is not False
            or row.get("deep_swe_outcomes_accessed") is not False
        ):
            raise ValueError(f"{path} has unsafe target access flags")
        choices[index] = EFFORTS.index(str(effort))
    return choices


def _repository_bootstrap_lower(
    values: np.ndarray,
    repositories: list[str],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> float:
    """Return a deterministic repository-cluster bootstrap 95 percent lower bound."""
    unique = sorted(set(repositories))
    if len(unique) < 2 or len(values) != len(repositories):
        raise ValueError("repository bootstrap requires aligned multi-repository data")
    groups = [np.flatnonzero(np.asarray(repositories, dtype=object) == repo) for repo in unique]
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = rng.integers(0, len(groups), size=len(groups))
        rows = np.concatenate([groups[group_index] for group_index in sampled])
        estimates[index] = float(np.mean(values[rows]))
    return float(np.quantile(estimates, 0.025))


def _evaluate_route(
    data: ConfirmationData,
    choices: np.ndarray,
    *,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    if choices.shape != (len(data.task_ids),) or np.any(
        (choices < 0) | (choices >= len(EFFORTS))
    ):
        raise ValueError("route choices are not aligned to confirmation tasks")
    task_rows = np.arange(len(data.task_ids))
    routed_reward = data.rewards[task_rows, choices]
    routed_cost = data.costs[task_rows, choices]
    counts = np.bincount(choices, minlength=len(EFFORTS))
    traffic = counts / len(choices)
    blind_reward = data.rewards @ traffic
    blind_cost = data.costs @ traffic
    advantages = routed_reward - blind_reward
    reward = float(np.mean(routed_reward))
    cost = float(np.mean(routed_cost))
    static = [
        {
            "arm": ARMS[index],
            "reward": float(np.mean(data.rewards[:, index])),
            "cost_usd_per_task": float(np.mean(data.costs[:, index])),
        }
        for index in range(len(EFFORTS))
    ]
    dominated = [
        row["arm"]
        for row in static
        if float(row["reward"]) >= reward
        and float(row["cost_usd_per_task"]) <= cost
    ]
    strongest_static = max(float(row["reward"]) for row in static)
    advantage = float(np.mean(advantages))
    lower = _repository_bootstrap_lower(
        advantages,
        data.repositories,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    return {
        "reward": reward,
        "cost_usd_per_task": cost,
        "arm_counts": {ARMS[index]: int(counts[index]) for index in range(len(ARMS))},
        "matched_blind_reward": float(np.mean(blind_reward)),
        "matched_blind_cost_usd_per_task": float(np.mean(blind_cost)),
        "matched_blind_advantage": advantage,
        "repository_bootstrap_resamples": bootstrap_resamples,
        "repository_bootstrap_seed": bootstrap_seed,
        "matched_blind_advantage_ci95_lower": lower,
        "primary_matched_blind_passed": advantage > 0.0 and lower > 0.0,
        "static_efforts": static,
        "dominated_by_static": dominated,
        "static_dominance_passed": not dominated,
        "strongest_static_reward": strongest_static,
        "quality_retention": reward / strongest_static if strongest_static > 0.0 else 1.0,
        "quality_retention_passed": (
            reward / strongest_static >= QUALITY_RETENTION
            if strongest_static > 0.0
            else True
        ),
    }


def confirm(
    outcomes_path: Path,
    audit_path: Path,
    fit_output: Path,
    output: Path,
) -> None:
    """Apply every frozen external confirmation gate once."""
    if output.exists():
        raise FileExistsError(f"confirmation output already exists: {output}")
    audit = _read_object(audit_path)
    if (
        audit.get("protocol") != COLLECTION_PROTOCOL
        or audit.get("valid") is not True
        or audit.get("deep_swe_outcomes_accessed") is not False
        or audit.get("target_outcomes_used") is not False
        or audit.get("confirmation_authorization_preserved") is not True
        or audit.get("outcomes_sha256") != _sha256(outcomes_path)
    ):
        raise ValueError("confirmation collection audit is invalid")
    data = _load_data(outcomes_path, audit)
    routes_path = fit_output / "confirmation-routes.jsonl"
    shuffled_path = fit_output / "confirmation-shuffled-routes.jsonl"
    route_audit_path = fit_output / "route-audit.json"
    development_report_path = fit_output / "development-report.json"
    selection_lock_path = fit_output / "selection-lock.json"
    route_audit = _read_object(route_audit_path)
    development_report = _read_object(development_report_path)
    selection_lock = _read_object(selection_lock_path)
    launch = audit.get("launch")
    authorization = launch.get("authorization") if isinstance(launch, dict) else None
    if not isinstance(authorization, dict):
        raise ValueError("confirmation collection lacks frozen launch authorization")
    expected_authorization = {
        "development_report_sha256": _sha256(development_report_path),
        "selection_lock_sha256": _sha256(selection_lock_path),
        "route_audit_sha256": _sha256(route_audit_path),
        "routes_sha256": _sha256(routes_path),
        "shuffled_routes_sha256": _sha256(shuffled_path),
    }
    if any(
        authorization.get(key) != value
        for key, value in expected_authorization.items()
    ):
        raise ValueError("confirmation launch authorization does not match fit outputs")
    if (
        route_audit.get("confirmation_routes_sha256") != _sha256(routes_path)
        or route_audit.get("shuffled_routes_sha256") != _sha256(shuffled_path)
        or route_audit.get("selection_lock_sha256") != _sha256(selection_lock_path)
        or route_audit.get("deep_swe_outcomes_accessed") is not False
        or route_audit.get("target_outcomes_used") is not False
        or selection_lock.get("deep_swe_outcomes_accessed") is not False
        or selection_lock.get("target_outcomes_used") is not False
    ):
        raise ValueError("frozen route identity or isolation audit drifted")
    latency = route_audit.get("latency")
    if not isinstance(latency, dict) or latency.get("passed") is not True:
        raise ValueError("frozen route latency gate is invalid")
    selected = development_report.get("selected")
    if not isinstance(selected, dict) or development_report.get("development_passed") is not True:
        raise ValueError("external development selection is invalid")
    folds = selected.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError("development selection lacks five grouped folds")
    development_retention_passed = all(
        isinstance(fold, dict) and float(fold.get("retention", 0.0)) >= QUALITY_RETENTION
        for fold in folds
    )
    primary = _evaluate_route(data, _choices(routes_path, data))
    shuffled = _evaluate_route(data, _choices(shuffled_path, data))
    gates = {
        "coverage_and_gradeability": (
            len(data.task_ids) >= MIN_RETAINED_TASKS
            and len(data.task_ids) + len(audit.get("excluded_tasks", [])) == SOURCE_TASKS
        ),
        "primary_matched_blind": primary["primary_matched_blind_passed"],
        "static_dominance": primary["static_dominance_passed"],
        "shuffled_control_failed_primary": not shuffled["primary_matched_blind_passed"],
        "development_fold_retention": development_retention_passed,
        "confirmation_quality_retention": primary["quality_retention_passed"],
        "latency": latency["passed"],
        "isolation": True,
    }
    report = {
        "protocol": PROTOCOL,
        "confirmation_passed": all(value is True for value in gates.values()),
        "tasks": len(data.task_ids),
        "repositories": len(set(data.repositories)),
        "gates": gates,
        "primary": primary,
        "shuffled_control": shuffled,
        "latency": latency,
        "development_selected_key": selection_lock.get("selected_key"),
        "fitted_numeric_state_persisted": False,
        "target_outcomes_used": False,
        "deep_swe_outcomes_accessed": False,
        "inputs": {
            "outcomes_sha256": _sha256(outcomes_path),
            "collection_audit_sha256": _sha256(audit_path),
            "development_report_sha256": _sha256(development_report_path),
            "selection_lock_sha256": _sha256(selection_lock_path),
            "route_audit_sha256": _sha256(route_audit_path),
            "routes_sha256": _sha256(routes_path),
            "shuffled_routes_sha256": _sha256(shuffled_path),
        },
    }
    output.mkdir(parents=True)
    report_path = output / "confirmation-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "confirmation passed=%s tasks=%d advantage=%.6f lower=%.6f",
        report["confirmation_passed"],
        len(data.task_ids),
        float(primary["matched_blind_advantage"]),
        float(primary["matched_blind_advantage_ci95_lower"]),
    )


def main() -> None:
    """Parse the one-shot confirmation analysis inputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--fit-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    confirm(args.outcomes, args.audit, args.fit_output, args.output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
