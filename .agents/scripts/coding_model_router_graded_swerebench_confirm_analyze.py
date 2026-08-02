"""Evaluate the frozen WMO route once on graded SWE-rebench confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL = "coding-router-graded-swerebench-confirmation-analysis-v1"
COLLECTION_PROTOCOL = "coding-router-graded-swerebench-confirmation-collection-v1"
FIT_PROTOCOL = "coding-router-graded-swerebench-wmo-knn-v1"
CORPUS_SHA256 = "c9443c9956e496123f396ee793efbb3368312092c4dcbd4e5e10bb77bd814f0a"
ARMS = (
    "luna-low",
    "luna-medium",
    "luna-high",
    "luna-xhigh",
    "luna-max",
    "sol-max",
)
QUALITY_RETENTION = 0.95
MIN_SAVINGS = 0.40
BOOTSTRAP_SEED = 20260801
MATCHED_BLIND_BOOTSTRAP_SEED = 20260802
BOOTSTRAP_DRAWS = 10_000
MAX_ROUTE_P95_MS = 5.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return {str(key): item for key, item in value.items()}


def _rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"row {line_number} is not an object")
            rows.append({str(key): item for key, item in value.items()})
    return rows


def _metrics(
    rewards: np.ndarray,
    costs: np.ndarray,
    choices: np.ndarray,
    baseline: int,
) -> dict[str, Any]:
    rows = np.arange(len(choices))
    reward = float(np.mean(rewards[rows, choices]))
    cost = float(np.mean(costs[rows, choices]))
    static_rewards = rewards.mean(axis=0)
    static_costs = costs.mean(axis=0)
    traffic = np.bincount(choices, minlength=len(ARMS)).astype(float) / len(choices)
    blind_reward = float(traffic @ static_rewards)
    dominated = [
        ARMS[index]
        for index in range(len(ARMS))
        if static_rewards[index] >= reward
        and static_costs[index] <= cost
        and (static_rewards[index] > reward or static_costs[index] < cost)
    ]
    return {
        "reward": reward,
        "cost_usd_per_task": cost,
        "quality_retention": reward / float(static_rewards[baseline]),
        "absolute_quality_delta": reward - float(static_rewards[baseline]),
        "cost_savings": 1.0 - cost / float(static_costs[baseline]),
        "matched_blind_reward": blind_reward,
        "matched_blind_advantage": reward - blind_reward,
        "model_mix": {
            ARMS[index]: float(share)
            for index, share in enumerate(traffic)
            if share > 0.0
        },
        "dominated_by_static": dominated,
    }


def _cluster_bootstrap(
    repositories: list[str],
    differences: np.ndarray,
    *,
    seed: int,
    estimand: str,
) -> dict[str, Any]:
    """Return a repository-cluster bootstrap interval for paired differences."""
    if differences.shape != (len(repositories),) or not np.isfinite(differences).all():
        raise ValueError("bootstrap differences must align with repositories and be finite")
    unique = sorted(set(repositories))
    if not unique:
        raise ValueError("repository bootstrap requires at least one repository")
    repository_array = np.asarray(repositories)
    cluster_values = []
    cluster_counts = []
    for repository in unique:
        indices = np.flatnonzero(repository_array == repository)
        cluster_values.append(float(np.sum(differences[indices])))
        cluster_counts.append(len(indices))
    values = np.asarray(cluster_values)
    counts = np.asarray(cluster_counts)
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_DRAWS)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = rng.integers(0, len(unique), size=len(unique))
        draws[draw] = float(values[sampled].sum() / counts[sampled].sum())
    interval = np.quantile(draws, [0.025, 0.5, 0.975])
    return {
        "seed": seed,
        "draws": BOOTSTRAP_DRAWS,
        "repositories": len(unique),
        "estimand": estimand,
        "lower_95": float(interval[0]),
        "median": float(interval[1]),
        "upper_95": float(interval[2]),
        "passed": bool(interval[0] >= 0.0),
    }


def _bootstrap_margin(
    repositories: list[str],
    router_rewards: np.ndarray,
    baseline_rewards: np.ndarray,
) -> dict[str, Any]:
    """Bootstrap router reward minus 95 percent of fit-selected static reward."""
    return _cluster_bootstrap(
        repositories,
        router_rewards - QUALITY_RETENTION * baseline_rewards,
        seed=BOOTSTRAP_SEED,
        estimand="mean router reward minus 0.95 times fit-selected static reward",
    )


def _bootstrap_matched_blind(
    repositories: list[str],
    router_rewards: np.ndarray,
    blind_rewards: np.ndarray,
) -> dict[str, Any]:
    """Bootstrap router reward minus identical-traffic task-blind mixing."""
    result = _cluster_bootstrap(
        repositories,
        router_rewards - blind_rewards,
        seed=MATCHED_BLIND_BOOTSTRAP_SEED,
        estimand="mean router reward minus identical-traffic task-blind reward",
    )
    result["passed"] = float(result["lower_95"]) > 0.0
    return result


def analyze(args: argparse.Namespace) -> None:
    """Open confirmation outcomes once and mechanically apply frozen gates."""
    if args.output.exists():
        raise FileExistsError(args.output)
    if _sha256(args.corpus) != CORPUS_SHA256:
        raise ValueError("confirmation corpus changed")
    audit = _object(args.audit)
    fit = _object(args.fit_report)
    routes = _object(args.routes)
    if (
        audit.get("protocol") != COLLECTION_PROTOCOL
        or audit.get("valid") is not True
        or audit.get("outcomes_sha256") != _sha256(args.outcomes)
        or audit.get("target_outcomes_used") is not False
        or audit.get("deep_swe_outcomes_accessed") is not False
        or audit.get("confirmation_outcomes_accessed") is not True
        or fit.get("protocol") != FIT_PROTOCOL
        or fit.get("valid") is not True
        or fit.get("development_passed") is not True
        or fit.get("confirmation_outcomes_accessed") is not False
        or routes.get("protocol") != FIT_PROTOCOL
        or routes.get("deep_swe_outcomes_accessed") is not False
        or routes.get("confirmation_outcomes_accessed") is not False
        or routes.get("fitted_numeric_state_persisted") is not False
    ):
        raise ValueError("confirmation inputs violate the seal")
    selected = fit.get("development", {}).get("selected")
    selected_candidate = selected.get("candidate") if isinstance(selected, dict) else None
    if (
        not isinstance(selected_candidate, str)
        or routes.get("selected_candidate") != selected_candidate
    ):
        raise ValueError("frozen route identity changed")
    baseline_name = fit.get("frontiers", {}).get("fit_selected_static")
    if baseline_name not in ARMS:
        raise ValueError("fit-selected static baseline is invalid")
    baseline = ARMS.index(str(baseline_name))
    raw_tasks = _object(args.corpus).get("tasks")
    exclusions = audit.get("excluded_tasks")
    if not isinstance(raw_tasks, list) or not isinstance(exclusions, list):
        raise ValueError("confirmation tasks or exclusions are invalid")
    excluded = {
        str(row["task_id"])
        for row in exclusions
        if isinstance(row, dict) and row.get("scope") == "whole-task"
    }
    tasks = [
        {str(key): value for key, value in task.items()}
        for task in raw_tasks
        if isinstance(task, dict) and str(task.get("task_id")) not in excluded
    ]
    if len(tasks) != audit.get("tasks") or len(tasks) < 304:
        raise ValueError("confirmation retained task coverage is invalid")
    task_ids = [str(task["task_id"]) for task in tasks]
    task_index = {task_id: index for index, task_id in enumerate(task_ids)}
    arm_index = {arm: index for index, arm in enumerate(ARMS)}
    rewards = np.full((len(tasks), len(ARMS)), np.nan)
    costs = np.full_like(rewards, np.nan)
    observed: set[tuple[str, str]] = set()
    for row in _rows(args.outcomes):
        task_id = row.get("task_id")
        arm = row.get("arm")
        reward = row.get("reward")
        cost = row.get("cost_usd")
        identity = (str(task_id), str(arm))
        if (
            not isinstance(task_id, str)
            or task_id not in task_index
            or not isinstance(arm, str)
            or arm not in arm_index
            or identity in observed
            or row.get("split") != "confirmation"
            or row.get("target_outcomes_used") is not False
            or isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(float(reward))
            or isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(float(cost))
        ):
            raise ValueError(f"invalid confirmation outcome: {identity}")
        rewards[task_index[task_id], arm_index[arm]] = float(reward)
        costs[task_index[task_id], arm_index[arm]] = float(cost)
        observed.add(identity)
    if len(observed) != rewards.size or np.isnan(rewards).any() or np.isnan(costs).any():
        raise ValueError("confirmation matrix is not dense")
    route_rows = routes.get("routes")
    if not isinstance(route_rows, list) or len(route_rows) != 320:
        raise ValueError("frozen routes do not cover the source confirmation cohort")
    route_by_id = {
        str(row["task_id"]): str(row["arm"])
        for row in route_rows
        if isinstance(row, dict) and row.get("arm") in ARMS
    }
    if len(route_by_id) != 320 or any(task_id not in route_by_id for task_id in task_ids):
        raise ValueError("frozen confirmation routes are incomplete")
    choices = np.asarray(
        [arm_index[route_by_id[task_id]] for task_id in task_ids],
        dtype=np.int64,
    )
    metrics = _metrics(rewards, costs, choices, baseline)
    row_indices = np.arange(len(tasks))
    repositories = [str(task["repository"]) for task in tasks]
    router_rewards = rewards[row_indices, choices]
    bootstrap = _bootstrap_margin(
        repositories,
        router_rewards,
        rewards[:, baseline],
    )
    traffic = np.bincount(choices, minlength=len(ARMS)).astype(float) / len(choices)
    matched_blind_bootstrap = _bootstrap_matched_blind(
        repositories,
        router_rewards,
        rewards @ traffic,
    )
    static = [
        {
            "arm": arm,
            "reward": float(rewards[:, index].mean()),
            "cost_usd_per_task": float(costs[:, index].mean()),
        }
        for index, arm in enumerate(ARMS)
    ]
    best_reward = rewards.max(axis=1)
    oracle_choices = np.asarray(
        [
            min(
                np.flatnonzero(rewards[index] == best_reward[index]),
                key=lambda arm: (costs[index, arm], int(arm)),
            )
            for index in row_indices
        ]
    )
    pair_oracles = []
    for left, right in combinations(range(len(ARMS)), 2):
        pair_choices = np.where(
            rewards[:, right] > rewards[:, left],
            right,
            np.where(
                rewards[:, left] > rewards[:, right],
                left,
                np.where(costs[:, right] < costs[:, left], right, left),
            ),
        )
        pair_oracles.append(
            {
                "pair": [ARMS[left], ARMS[right]],
                **_metrics(rewards, costs, pair_choices, baseline),
            }
        )
    route_latency = routes.get("route_decision_latency_ms", {})
    latency_p95 = route_latency.get("p95") if isinstance(route_latency, dict) else None
    gates = {
        "quality_retention": metrics["quality_retention"] >= QUALITY_RETENTION,
        "cost_savings": metrics["cost_savings"] >= MIN_SAVINGS,
        "paired_repository_bootstrap": bootstrap["passed"],
        "positive_matched_blind_point": metrics["matched_blind_advantage"] > 0.0,
        "positive_matched_blind_bootstrap": matched_blind_bootstrap["passed"],
        "not_dominated_by_static": not metrics["dominated_by_static"],
        "route_latency": isinstance(latency_p95, (int, float))
        and float(latency_p95) < MAX_ROUTE_P95_MS,
    }
    report = {
        "protocol": PROTOCOL,
        "valid": True,
        "confirmation_passed": all(gates.values()),
        "selected_candidate": selected_candidate,
        "fit_selected_static": baseline_name,
        "tasks": len(tasks),
        "excluded_tasks": sorted(excluded),
        "router": metrics,
        "repository_bootstrap": bootstrap,
        "matched_blind_repository_bootstrap": matched_blind_bootstrap,
        "route_decision_latency_ms": route_latency,
        "gates": gates,
        "static": static,
        "oracle": _metrics(rewards, costs, oracle_choices, baseline),
        "pair_oracles": pair_oracles,
        "rough_cumulative_spend_usd": audit["rough_cumulative_experiment_spend_usd"],
        "target_outcomes_used": False,
        "deep_swe_outcomes_accessed": False,
        "confirmation_outcomes_accessed_once": True,
        "fitted_numeric_router_state_persisted": False,
        "input_sha256": {
            "corpus": _sha256(args.corpus),
            "outcomes": _sha256(args.outcomes),
            "audit": _sha256(args.audit),
            "fit_report": _sha256(args.fit_report),
            "routes": _sha256(args.routes),
        },
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
