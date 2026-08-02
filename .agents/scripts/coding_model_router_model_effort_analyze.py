"""Analyze the untouched external confirmation for one frozen pair route."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("coding-router-model-effort-analyze")

PROTOCOL = "coding-router-model-effort-confirmation-analysis-v1"
CORPUS_SHA256 = "9798dd1e58be0d13331d097307670dc3fc3760ad211da20e6367666523f080a7"
BOOTSTRAPS = 10_000
SEED = 20_260_801
QUALITY_RETENTION = 0.95
MIN_SAVINGS = 0.40
MIN_RETAINED_TASKS = 190
NULL_COUNT = 128


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return {str(key): item for key, item in value.items()}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} contains a non-object row")
        rows.append({str(key): item for key, item in value.items()})
    return rows


def _collection(
    outcomes_path: Path,
    audit_path: Path,
) -> tuple[str, dict[str, tuple[float, float]], set[str], float, float]:
    audit = _read_object(audit_path)
    outcomes_sha = _sha256(outcomes_path)
    efforts = audit.get("efforts")
    prefix = audit.get("arm_prefix")
    retained = audit.get("tasks")
    exclusions = audit.get("excluded_tasks")
    if (
        audit.get("valid") is not True
        or not isinstance(prefix, str)
        or not isinstance(efforts, list)
        or len(efforts) != 1
        or not isinstance(retained, int)
        or not MIN_RETAINED_TASKS <= retained <= 200
        or not isinstance(exclusions, list)
        or len(exclusions) != 200 - retained
        or audit.get("cells") != retained * 2
        or audit.get("outcomes_sha256") != outcomes_sha
        or audit.get("target_outcomes_used") is not False
        or audit.get("deep_swe_outcomes_accessed") is not False
        or audit.get("confirmation_authorization_preserved") is not True
    ):
        raise ValueError("selected-arm confirmation collection is incomplete or unsafe")
    arm = f"{prefix}-{efforts[0]}"
    excluded = {
        str(row["task_id"])
        for row in exclusions
        if isinstance(row, dict) and row.get("scope") == "whole-task"
    }
    if len(excluded) != len(exclusions):
        raise ValueError("selected-arm exclusion is invalid")
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row in _read_rows(outcomes_path):
        task_id = row.get("task_id")
        attempt = row.get("attempt_number")
        reward = row.get("reward")
        cost = row.get("cost_usd")
        if (
            not isinstance(task_id, str)
            or task_id in excluded
            or row.get("arm") != arm
            or not isinstance(attempt, int)
            or isinstance(attempt, bool)
            or not 0 <= attempt < 2
            or isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or float(reward) not in {0.0, 1.0}
            or isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(float(cost))
            or float(cost) < 0.0
            or row.get("target_outcomes_used") is not False
        ):
            raise ValueError(f"invalid selected-arm outcome {(task_id, attempt)!r}")
        grouped.setdefault(task_id, []).append((float(reward), float(cost)))
    if len(grouped) != retained or any(len(values) != 2 for values in grouped.values()):
        raise ValueError("selected-arm outcomes are not dense across two attempts")
    means = {
        task_id: (
            float(np.mean([value[0] for value in values])),
            float(np.mean([value[1] for value in values])),
        )
        for task_id, values in grouped.items()
    }
    spent = audit.get("spent_matrix_cost_usd")
    cumulative = audit.get("rough_cumulative_experiment_spend_usd")
    if (
        isinstance(spent, bool)
        or not isinstance(spent, (int, float))
        or isinstance(cumulative, bool)
        or not isinstance(cumulative, (int, float))
    ):
        raise ValueError("selected-arm audit lacks spend provenance")
    return arm, means, excluded, float(spent), float(cumulative) - float(spent)


def _route_map(rows: list[dict[str, Any]], task_ids: list[str]) -> dict[str, str]:
    if len(rows) != len(task_ids) or [str(row.get("task_id")) for row in rows] != task_ids:
        raise ValueError("route does not exactly cover frozen confirmation tasks")
    result = {str(row["task_id"]): str(row["arm"]) for row in rows}
    if len(result) != len(task_ids):
        raise ValueError("route has duplicate task identities")
    return result


def _bootstrap_weights(repositories: list[str]) -> tuple[np.ndarray, np.ndarray]:
    unique = sorted(set(repositories))
    index = {repo: position for position, repo in enumerate(unique)}
    group_index = np.asarray([index[repo] for repo in repositories], dtype=np.int64)
    weights = np.random.default_rng(SEED).multinomial(
        len(unique),
        np.full(len(unique), 1.0 / len(unique)),
        size=BOOTSTRAPS,
    )
    return group_index, weights.astype(np.float64)


def _group_sums(values: np.ndarray, group_index: np.ndarray, groups: int) -> np.ndarray:
    if values.ndim == 1:
        result = np.zeros(groups, dtype=np.float64)
    else:
        result = np.zeros((groups, values.shape[1]), dtype=np.float64)
    np.add.at(result, group_index, values)
    return result


def _bootstrap_means(
    values: np.ndarray,
    group_index: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    groups = weights.shape[1]
    counts = np.bincount(group_index, minlength=groups).astype(np.float64)
    denominator = weights @ counts
    sums = _group_sums(values, group_index, groups)
    numerator = weights @ sums
    if values.ndim == 1:
        return numerator / denominator
    return numerator / denominator[:, None]


def _interval(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def analyze(
    corpus_path: Path,
    fit_output: Path,
    first_outcomes: Path,
    first_audit: Path,
    second_outcomes: Path,
    second_audit: Path,
    output: Path,
) -> None:
    """Evaluate the frozen real route against static, blind, and family-null controls."""
    if _sha256(corpus_path) != CORPUS_SHA256:
        raise ValueError("confirmation corpus changed")
    tasks = _read_object(corpus_path).get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 200:
        raise ValueError("confirmation corpus must contain 200 tasks")
    task_ids = [str(task.get("task_id")) for task in tasks if isinstance(task, dict)]
    repositories_by_task = {
        str(task["task_id"]): str(task["repository"])
        for task in tasks
        if isinstance(task, dict)
    }
    if len(task_ids) != 200 or len(set(task_ids)) != 200:
        raise ValueError("confirmation task identities are invalid")

    report_path = fit_output / "selection-report.json"
    lock_path = fit_output / "selection-lock.json"
    routes_path = fit_output / "confirmation-routes.jsonl"
    blind_path = fit_output / "confirmation-blind-routes.jsonl"
    null_path = fit_output / "confirmation-null-routes.jsonl"
    fit_report = _read_object(report_path)
    lock = _read_object(lock_path)
    pair = fit_report.get("selected_pair")
    baseline = fit_report.get("development_static_baseline")
    if (
        fit_report.get("valid") is not True
        or fit_report.get("confirmation_authorized") is not True
        or not isinstance(pair, list)
        or len(pair) != 2
        or baseline not in pair
        or fit_report.get("selection_lock_sha256") != _sha256(lock_path)
        or lock.get("confirmation_routes_sha256") != _sha256(routes_path)
        or lock.get("confirmation_blind_routes_sha256") != _sha256(blind_path)
        or lock.get("confirmation_null_routes_sha256") != _sha256(null_path)
        or fit_report.get("target_outcomes_used") is not False
        or fit_report.get("deep_swe_outcomes_accessed") is not False
    ):
        raise ValueError("fit selection or frozen route hashes changed")

    first = _collection(first_outcomes, first_audit)
    second = _collection(second_outcomes, second_audit)
    collections = {first[0]: first, second[0]: second}
    if set(collections) != set(pair):
        raise ValueError("confirmation collections do not match the selected pair")
    excluded = first[2] | second[2]
    retained_ids = [task_id for task_id in task_ids if task_id not in excluded]
    if len(retained_ids) < MIN_RETAINED_TASKS:
        raise ValueError("confirmation whole-task coverage is below 95 percent")
    repositories = [repositories_by_task[task_id] for task_id in retained_ids]
    arm_rewards = {
        arm: np.asarray([collections[arm][1][task_id][0] for task_id in retained_ids])
        for arm in pair
    }
    arm_costs = {
        arm: np.asarray([collections[arm][1][task_id][1] for task_id in retained_ids])
        for arm in pair
    }
    real_map = _route_map(_read_rows(routes_path), task_ids)
    blind_map = _route_map(_read_rows(blind_path), task_ids)
    null_rows = _read_rows(null_path)
    if len(null_rows) != NULL_COUNT * 200:
        raise ValueError("family-null routes are incomplete")
    null_maps = [
        _route_map(null_rows[index * 200 : (index + 1) * 200], task_ids)
        for index in range(NULL_COUNT)
    ]

    def route_values(route: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
        arms = [route[task_id] for task_id in retained_ids]
        if any(arm not in collections for arm in arms):
            raise ValueError("route selected an arm outside the frozen pair")
        rewards = np.asarray(
            [
                collections[arm][1][task_id][0]
                for arm, task_id in zip(arms, retained_ids, strict=True)
            ]
        )
        costs = np.asarray(
            [
                collections[arm][1][task_id][1]
                for arm, task_id in zip(arms, retained_ids, strict=True)
            ]
        )
        return rewards, costs

    real_rewards, real_costs = route_values(real_map)
    blind_rewards, blind_costs = route_values(blind_map)
    null_rewards = np.column_stack([route_values(route)[0] for route in null_maps])
    baseline_rewards = arm_rewards[str(baseline)]
    baseline_costs = arm_costs[str(baseline)]
    real_reward = float(real_rewards.mean())
    real_cost = float(real_costs.mean())
    baseline_reward = float(baseline_rewards.mean())
    baseline_cost = float(baseline_costs.mean())
    retention = real_reward / baseline_reward if baseline_reward else 0.0
    savings = 1.0 - real_cost / baseline_cost if baseline_cost else 0.0
    static_dominators = [
        arm
        for arm in pair
        if float(arm_rewards[arm].mean()) >= real_reward
        and float(arm_costs[arm].mean()) <= real_cost
        and (
            float(arm_rewards[arm].mean()) > real_reward
            or float(arm_costs[arm].mean()) < real_cost
        )
    ]

    group_index, weights = _bootstrap_weights(repositories)
    real_boot = _bootstrap_means(real_rewards, group_index, weights)
    blind_boot = _bootstrap_means(blind_rewards, group_index, weights)
    baseline_reward_boot = _bootstrap_means(baseline_rewards, group_index, weights)
    real_cost_boot = _bootstrap_means(real_costs, group_index, weights)
    baseline_cost_boot = _bootstrap_means(baseline_costs, group_index, weights)
    null_boot = _bootstrap_means(null_rewards, group_index, weights)
    quality_margin_boot = real_boot - QUALITY_RETENTION * baseline_reward_boot
    savings_boot = 1.0 - real_cost_boot / baseline_cost_boot
    blind_advantage_boot = real_boot - blind_boot
    best_null_advantage_boot = real_boot - np.max(null_boot, axis=1)
    quality_lower = float(np.percentile(quality_margin_boot, 2.5))
    savings_lower = float(np.percentile(savings_boot, 2.5))
    blind_lower = float(np.percentile(blind_advantage_boot, 2.5))
    null_lower = float(np.percentile(best_null_advantage_boot, 2.5))
    latency_p95 = float(fit_report.get("route_latency_p95_ms", math.inf))
    gates = {
        "coverage": len(retained_ids) >= MIN_RETAINED_TASKS,
        "quality_retention_point": retention >= QUALITY_RETENTION,
        "quality_retention_interval": quality_lower >= 0.0,
        "cost_savings_point": savings >= MIN_SAVINGS,
        "cost_savings_interval": savings_lower >= MIN_SAVINGS,
        "matched_blind": blind_lower > 0.0,
        "best_of_128_family_nulls": null_lower > 0.0,
        "no_static_dominance": not static_dominators,
        "route_latency": latency_p95 < 5.0,
    }
    prior_values = [first[4], second[4]]
    if not math.isclose(prior_values[0], prior_values[1], rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("confirmation arms disagree on prior cumulative spend")
    cumulative_spend = prior_values[0] + first[3] + second[3]
    report = {
        "protocol": PROTOCOL,
        "valid": True,
        "passed": all(gates.values()),
        "gates": gates,
        "selected_candidate": fit_report["selected_candidate"],
        "selected_pair": pair,
        "development_static_baseline": baseline,
        "source_tasks": 200,
        "retained_tasks": len(retained_ids),
        "retained_task_coverage": len(retained_ids) / 200,
        "excluded_task_ids": sorted(excluded),
        "reward": real_reward,
        "cost_usd_per_task": real_cost,
        "baseline_reward": baseline_reward,
        "baseline_cost_usd_per_task": baseline_cost,
        "quality_retention": retention,
        "cost_savings": savings,
        "quality_margin_interval": _interval(quality_margin_boot),
        "cost_savings_interval": _interval(savings_boot),
        "matched_blind_reward": float(blind_rewards.mean()),
        "matched_blind_cost_usd_per_task": float(blind_costs.mean()),
        "matched_blind_advantage": real_reward - float(blind_rewards.mean()),
        "matched_blind_advantage_interval": _interval(blind_advantage_boot),
        "best_null_reward": float(np.max(null_rewards.mean(axis=0))),
        "real_minus_best_null": real_reward - float(np.max(null_rewards.mean(axis=0))),
        "best_null_advantage_interval": _interval(best_null_advantage_boot),
        "static_arms": {
            arm: {
                "reward": float(arm_rewards[arm].mean()),
                "cost_usd_per_task": float(arm_costs[arm].mean()),
            }
            for arm in pair
        },
        "static_dominators": static_dominators,
        "route_latency_p95_ms": latency_p95,
        "bootstrap_repositories": len(set(repositories)),
        "bootstrap_resamples": BOOTSTRAPS,
        "bootstrap_seed": SEED,
        "rough_cumulative_experiment_spend_usd": cumulative_spend,
        "target_outcomes_used": False,
        "deep_swe_outcomes_accessed": False,
        "fitted_numeric_router_state_persisted": False,
        "inputs": {
            "corpus_sha256": _sha256(corpus_path),
            "selection_report_sha256": _sha256(report_path),
            "selection_lock_sha256": _sha256(lock_path),
            "first_outcomes_sha256": _sha256(first_outcomes),
            "first_audit_sha256": _sha256(first_audit),
            "second_outcomes_sha256": _sha256(second_outcomes),
            "second_audit_sha256": _sha256(second_audit),
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "confirmation-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "passed=%s reward=%.6f retention=%.6f savings=%.6f blind_lower=%.6f null_lower=%.6f",
        report["passed"],
        real_reward,
        retention,
        savings,
        blind_lower,
        null_lower,
    )


def main() -> None:
    """Parse frozen confirmation artifacts and analyze on the current compute host."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--fit-output", type=Path, required=True)
    parser.add_argument("--first-outcomes", type=Path, required=True)
    parser.add_argument("--first-audit", type=Path, required=True)
    parser.add_argument("--second-outcomes", type=Path, required=True)
    parser.add_argument("--second-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(
        args.corpus,
        args.fit_output,
        args.first_outcomes,
        args.first_audit,
        args.second_outcomes,
        args.second_audit,
        args.output,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
