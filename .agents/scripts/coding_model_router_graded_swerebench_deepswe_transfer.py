"""Freeze the confirmed graded SWE-rebench router, then open DeepSWE once."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from coding_model_router_graded_swerebench_fit import (
    ARMS,
    Candidate,
    _embed,
    _freeze_routes,
    candidate_grid,
    load_data,
)
from coding_model_router_graded_swerebench_fit import (
    PROTOCOL as FIT_PROTOCOL,
)

CONFIRMATION_PROTOCOL = "coding-router-graded-swerebench-confirmation-analysis-v1"
FREEZE_PROTOCOL = "coding-router-graded-swerebench-to-deepswe-route-freeze-v1"
EVALUATION_PROTOCOL = "coding-router-graded-swerebench-to-deepswe-single-transfer-v1"
TARGET_MATRIX_SHA256 = "2988742e48b1c9bfec8dc45d88af112c46c45367529d1936b709e4b4e549835f"
TARGET_FEATURE_VIEW_SHA256 = "35ad33855f63f147b1861b58b59ad635f8860677b5d0d5e902c421029d78637b"
TARGET_TASKS = 113
QUALITY_RETENTION = 0.95
MIN_SAVINGS = 0.40
BOOTSTRAP_SEED = 20260801
BOOTSTRAP_DRAWS = 10_000
TARGET_ARM = {
    "luna-low": "mini_swe_agent_gpt_5_6_luna_low",
    "luna-medium": "mini_swe_agent_gpt_5_6_luna_medium",
    "luna-high": "mini_swe_agent_gpt_5_6_luna_high",
    "luna-xhigh": "mini_swe_agent_gpt_5_6_luna_xhigh",
    "luna-max": "mini_swe_agent_gpt_5_6_luna_max",
    "sol-max": "mini_swe_agent_gpt_5_6_sol_max",
}


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
    result: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            result.append({str(key): item for key, item in value.items()})
    return result


def _selected_candidate(fit: dict[str, Any], confirmation: dict[str, Any]) -> Candidate:
    development = fit.get("development")
    selected = development.get("selected") if isinstance(development, dict) else None
    key = selected.get("candidate") if isinstance(selected, dict) else None
    configuration = selected.get("configuration") if isinstance(selected, dict) else None
    candidates = {candidate.key: candidate for candidate in candidate_grid()}
    candidate = candidates.get(str(key))
    if (
        candidate is None
        or not isinstance(configuration, dict)
        or configuration
        != {
            "order": candidate.order,
            "guard": candidate.guard,
            "k": candidate.k,
            "z": candidate.z,
            "pick_lam": candidate.pick_lam,
        }
        or confirmation.get("selected_candidate") != candidate.key
    ):
        raise ValueError("confirmed candidate identity changed")
    return candidate


def _target_tasks(path: Path) -> list[dict[str, Any]]:
    if _sha256(path) != TARGET_FEATURE_VIEW_SHA256:
        raise ValueError("DeepSWE label-free feature view changed")
    raw = _object(path)
    values = raw.get("rows")
    if (
        raw.get("protocol") != "deepswe-label-free-task-feature-view-v2"
        or raw.get("target_reward_fields_accessed") is not False
        or raw.get("target_cost_fields_accessed") is not False
        or not isinstance(values, list)
        or len(values) != TARGET_TASKS
    ):
        raise ValueError("DeepSWE feature view violated the label-free boundary")
    result: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise ValueError(f"DeepSWE feature row {index} is invalid")
        task_id = value.get("id")
        repository = value.get("repository")
        prompt = value.get("text")
        if not all(isinstance(item, str) and item for item in (task_id, repository, prompt)):
            raise ValueError(f"DeepSWE feature row {index} is incomplete")
        result.append(
            {
                "task_id": task_id,
                "repository": repository,
                "language": value.get("language")
                if isinstance(value.get("language"), str) and value.get("language")
                else "unknown",
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
        )
    if len({str(task["task_id"]) for task in result}) != TARGET_TASKS:
        raise ValueError("DeepSWE feature view contains duplicate task IDs")
    return result


def freeze_routes(args: argparse.Namespace) -> dict[str, Any]:
    """Refit development only on remote compute and persist decisions, not state."""
    if args.output.exists():
        raise FileExistsError(args.output)
    fit = _object(args.fit_report)
    confirmation = _object(args.confirmation_report)
    confirmation_inputs = confirmation.get("input_sha256")
    if (
        fit.get("protocol") != FIT_PROTOCOL
        or fit.get("valid") is not True
        or fit.get("development_passed") is not True
        or fit.get("deep_swe_outcomes_accessed") is not False
        or fit.get("target_outcomes_used") is not False
        or fit.get("fitted_numeric_state_persisted") is not False
        or confirmation.get("protocol") != CONFIRMATION_PROTOCOL
        or confirmation.get("valid") is not True
        or confirmation.get("confirmation_passed") is not True
        or confirmation.get("target_outcomes_used") is not False
        or confirmation.get("deep_swe_outcomes_accessed") is not False
        or confirmation.get("confirmation_outcomes_accessed_once") is not True
        or not isinstance(confirmation_inputs, dict)
        or confirmation_inputs.get("fit_report") != _sha256(args.fit_report)
    ):
        raise ValueError("external confirmation does not authorize DeepSWE transfer")
    candidate = _selected_candidate(fit, confirmation)
    data = load_data(args.development_corpus, args.development_outcomes, args.development_audit)
    target = _target_tasks(args.target_feature_view)
    target_texts = [
        f"repository={task['repository']}\nlanguage={task['language']}\n{task['prompt']}"
        for task in target
    ]
    vectors, embedding = _embed(
        data.texts + target_texts,
        args.embedding_model,
        args.tokenizer,
    )
    development_vectors = vectors[: len(data.texts)]
    target_vectors = vectors[len(data.texts) :]
    routes = _freeze_routes(candidate, data, target, development_vectors, target_vectors)
    route_rows = routes.get("routes")
    latency = routes.get("route_decision_latency_ms")
    if (
        not isinstance(route_rows, list)
        or len(route_rows) != TARGET_TASKS
        or not isinstance(latency, dict)
        or not isinstance(latency.get("p95"), (int, float))
        or float(latency["p95"]) >= 5.0
    ):
        raise ValueError("frozen DeepSWE routes are incomplete or too slow")
    by_id = {
        str(row.get("task_id")): str(row.get("arm"))
        for row in route_rows
        if isinstance(row, dict) and row.get("arm") in ARMS
    }
    if len(by_id) != TARGET_TASKS or any(str(task["task_id"]) not in by_id for task in target):
        raise ValueError("frozen DeepSWE route identities are incomplete")
    args.output.mkdir(parents=True)
    decisions_path = args.output / "target-decisions.jsonl"
    decisions = [
        {
            "task_id": str(task["task_id"]),
            "repository": str(task["repository"]),
            "prompt_sha256": str(task["prompt_sha256"]),
            "source_arm": by_id[str(task["task_id"])],
            "target_arm": TARGET_ARM[by_id[str(task["task_id"])]],
        }
        for task in target
    ]
    decisions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions),
        encoding="utf-8",
    )
    counts = Counter(str(row["source_arm"]) for row in decisions)
    baseline = fit.get("frontiers", {}).get("fit_selected_static")
    if baseline not in ARMS:
        raise ValueError("fit-selected static baseline is invalid")
    report = {
        "protocol": FREEZE_PROTOCOL,
        "selected_candidate": candidate.key,
        "fit_selected_static": baseline,
        "tasks": TARGET_TASKS,
        "arm_counts": {arm: counts.get(arm, 0) for arm in ARMS},
        "route_decision_latency_ms": latency,
        "embedding": embedding,
        "input_sha256": {
            "development_corpus": _sha256(args.development_corpus),
            "development_outcomes": _sha256(args.development_outcomes),
            "development_audit": _sha256(args.development_audit),
            "fit_report": _sha256(args.fit_report),
            "confirmation_report": _sha256(args.confirmation_report),
            "target_feature_view": _sha256(args.target_feature_view),
        },
        "decisions_sha256": _sha256(decisions_path),
        "target_reward_fields_accessed": False,
        "target_cost_fields_accessed": False,
        "target_outcomes_used": False,
        "deep_swe_outcomes_accessed": False,
        "fitted_numeric_state_persisted": False,
        "embedding_model_persisted": False,
        "task_embeddings_persisted": False,
        "knn_bank_persisted": False,
    }
    (args.output / "target-route-freeze.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _load_target_matrix(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, set[str]]:
    raw = _object(path)
    values = raw.get("outcomes")
    if not isinstance(values, list):
        raise ValueError("DeepSWE matrix has no outcomes list")
    target_arms = tuple(TARGET_ARM[arm] for arm in ARMS)
    cells: dict[tuple[str, str], tuple[float, float]] = {}
    all_ids: set[str] = set()
    for value in values:
        if not isinstance(value, dict):
            continue
        task_id = value.get("scenario_id")
        arm = value.get("model")
        if not isinstance(task_id, str) or not isinstance(arm, str) or arm not in target_arms:
            continue
        all_ids.add(task_id)
        reward = _finite(value.get("reward"))
        cost = _finite(value.get("cost_usd"))
        if reward is None or cost is None:
            continue
        if not 0.0 <= reward <= 1.0 or cost < 0.0:
            raise ValueError(f"invalid DeepSWE cell for {task_id}")
        identity = (task_id, arm)
        if identity in cells:
            raise ValueError(f"duplicate DeepSWE cell: {identity}")
        cells[identity] = (reward, cost)
    complete = sorted(
        task_id for task_id in all_ids if all((task_id, arm) in cells for arm in target_arms)
    )
    if not complete:
        raise ValueError("DeepSWE has no complete six-arm tasks")
    rewards = np.asarray(
        [[cells[(task_id, TARGET_ARM[arm])][0] for arm in ARMS] for task_id in complete],
        dtype=np.float64,
    )
    costs = np.asarray(
        [[cells[(task_id, TARGET_ARM[arm])][1] for arm in ARMS] for task_id in complete],
        dtype=np.float64,
    )
    return complete, rewards, costs, all_ids


def _cluster_interval(groups: list[str], values: np.ndarray) -> dict[str, Any]:
    unique = sorted(set(groups))
    if len(unique) < 2 or values.shape != (len(groups),):
        raise ValueError("repository bootstrap inputs are invalid")
    group_indices = {
        group: np.flatnonzero(np.asarray(groups, dtype=object) == group) for group in unique
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([group_indices[str(group)] for group in sampled])
        draws[draw] = float(np.mean(values[indices]))
    lower, median, upper = np.quantile(draws, [0.025, 0.5, 0.975])
    return {
        "mean": float(np.mean(values)),
        "lower_95": float(lower),
        "median": float(median),
        "upper_95": float(upper),
        "repositories": len(unique),
        "seed": BOOTSTRAP_SEED,
        "draws": BOOTSTRAP_DRAWS,
    }


def _route_metrics(
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
    if static_rewards[baseline] <= 0.0 or static_costs[baseline] <= 0.0:
        raise ValueError("DeepSWE fit-selected static baseline is degenerate")
    routed_rewards = rewards[rows, choices]
    cost_efficiency = reward / cost
    baseline_efficiency = float(static_rewards[baseline] / static_costs[baseline])
    traffic = np.bincount(choices, minlength=len(ARMS)).astype(float) / len(choices)
    blind_reward = rewards @ traffic
    blind_cost = costs @ traffic
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
        "cost_usd_total": float(np.sum(costs[rows, choices])),
        "cost_usd_per_reward": cost / reward if reward > 0.0 else None,
        "reward_per_cost_usd": cost_efficiency,
        "cost_efficiency_gain_vs_fit_static": cost_efficiency / baseline_efficiency,
        "best_reward_hit_rate": float(np.mean(routed_rewards >= rewards.max(axis=1) - 1e-12)),
        "quality_retention": reward / float(static_rewards[baseline]),
        "cost_savings": 1.0 - cost / float(static_costs[baseline]),
        "matched_blind_reward": float(np.mean(blind_reward)),
        "matched_blind_cost_usd_per_task": float(np.mean(blind_cost)),
        "matched_blind_advantage": reward - float(np.mean(blind_reward)),
        "model_mix": {
            ARMS[index]: float(share) for index, share in enumerate(traffic) if share > 0.0
        },
        "dominated_by_static": dominated,
    }


def evaluate_routes(args: argparse.Namespace) -> dict[str, Any]:
    """Open the hash-pinned DeepSWE outcome matrix after decisions are immutable."""
    if args.output.exists():
        raise FileExistsError(args.output)
    if _sha256(args.target_matrix) != TARGET_MATRIX_SHA256:
        raise ValueError("DeepSWE matrix hash changed")
    freeze = _object(args.freeze)
    decisions = _rows(args.decisions)
    if (
        freeze.get("protocol") != FREEZE_PROTOCOL
        or freeze.get("decisions_sha256") != _sha256(args.decisions)
        or freeze.get("target_outcomes_used") is not False
        or freeze.get("deep_swe_outcomes_accessed") is not False
        or freeze.get("fitted_numeric_state_persisted") is not False
        or len(decisions) != TARGET_TASKS
    ):
        raise ValueError("DeepSWE route freeze is invalid")
    by_id = {str(row.get("task_id")): row for row in decisions}
    if len(by_id) != TARGET_TASKS:
        raise ValueError("DeepSWE decisions are not unique")
    for row in decisions:
        source_arm = row.get("source_arm")
        if source_arm not in ARMS or row.get("target_arm") != TARGET_ARM[str(source_arm)]:
            raise ValueError("DeepSWE decision arm mapping changed")
    task_ids, rewards, costs, all_ids = _load_target_matrix(args.target_matrix)
    if any(task_id not in by_id for task_id in task_ids):
        raise ValueError("complete DeepSWE tasks lack frozen routes")
    choices = np.asarray(
        [ARMS.index(str(by_id[task_id]["source_arm"])) for task_id in task_ids],
        dtype=np.int64,
    )
    repositories = [str(by_id[task_id]["repository"]) for task_id in task_ids]
    baseline_name = freeze.get("fit_selected_static")
    if baseline_name not in ARMS:
        raise ValueError("frozen static baseline is invalid")
    baseline = ARMS.index(str(baseline_name))
    metrics = _route_metrics(rewards, costs, choices, baseline)
    rows = np.arange(len(task_ids))
    traffic = np.bincount(choices, minlength=len(ARMS)).astype(float) / len(choices)
    blind_rewards = rewards @ traffic
    blind_interval = _cluster_interval(
        repositories,
        rewards[rows, choices] - blind_rewards,
    )
    retention_interval = _cluster_interval(
        repositories,
        rewards[rows, choices] - QUALITY_RETENTION * rewards[:, baseline],
    )
    static = [
        {
            "arm": arm,
            "reward": float(rewards[:, index].mean()),
            "cost_usd_per_task": float(costs[:, index].mean()),
            "cost_usd_total": float(costs[:, index].sum()),
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
            for index in rows
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
                **_route_metrics(rewards, costs, pair_choices, baseline),
            }
        )
    gates = {
        "quality_retention": metrics["quality_retention"] >= QUALITY_RETENTION,
        "cost_savings": metrics["cost_savings"] >= MIN_SAVINGS,
        "positive_matched_blind_lower_95": blind_interval["lower_95"] > 0.0,
        "retention_margin_lower_95": retention_interval["lower_95"] >= 0.0,
        "not_dominated_by_static": not metrics["dominated_by_static"],
    }
    gates["passed"] = all(gates.values())
    args.output.mkdir(parents=True)
    evaluated_rows = args.output / "evaluated-rows.jsonl"
    evaluated_rows.write_text(
        "".join(
            json.dumps(
                {
                    "task_id": task_id,
                    "repository": repositories[index],
                    "source_arm": ARMS[int(choices[index])],
                    "target_arm": TARGET_ARM[ARMS[int(choices[index])]],
                    "reward": float(rewards[index, choices[index]]),
                    "cost_usd": float(costs[index, choices[index]]),
                    "matched_blind_reward": float(blind_rewards[index]),
                },
                sort_keys=True,
            )
            + "\n"
            for index, task_id in enumerate(task_ids)
        ),
        encoding="utf-8",
    )
    report = {
        "protocol": EVALUATION_PROTOCOL,
        "target_tasks": len(task_ids),
        "target_repositories": len(set(repositories)),
        "target_tasks_dropped_for_missing_cells": len(all_ids) - len(task_ids),
        "fit_selected_static": baseline_name,
        "router": metrics,
        "matched_blind_reward_interval": blind_interval,
        "quality_retention_margin_interval": retention_interval,
        "static": static,
        "full_oracle": _route_metrics(rewards, costs, oracle_choices, baseline),
        "pair_oracles": pair_oracles,
        "gates": gates,
        "input_sha256": {
            "freeze": _sha256(args.freeze),
            "decisions": _sha256(args.decisions),
            "target_matrix": _sha256(args.target_matrix),
        },
        "evaluated_rows_sha256": _sha256(evaluated_rows),
        "reward": "DeepSWE graded fail-to-pass reward",
        "cost": "DeepSWE measured trial cost",
        "target_routes_frozen_before_outcomes": True,
        "target_outcomes_used_for_fit": False,
        "target_outcomes_used_for_threshold": False,
        "target_outcomes_used_for_evaluation": True,
        "target_evaluation_count": 1,
        "fitted_numeric_state_persisted": False,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--development-corpus", type=Path, required=True)
    freeze.add_argument("--development-outcomes", type=Path, required=True)
    freeze.add_argument("--development-audit", type=Path, required=True)
    freeze.add_argument("--fit-report", type=Path, required=True)
    freeze.add_argument("--confirmation-report", type=Path, required=True)
    freeze.add_argument("--target-feature-view", type=Path, required=True)
    freeze.add_argument("--embedding-model", type=Path, required=True)
    freeze.add_argument("--tokenizer", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--freeze", type=Path, required=True)
    evaluate.add_argument("--decisions", type=Path, required=True)
    evaluate.add_argument("--target-matrix", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "freeze":
        freeze_routes(args)
    else:
        evaluate_routes(args)


if __name__ == "__main__":
    main()
