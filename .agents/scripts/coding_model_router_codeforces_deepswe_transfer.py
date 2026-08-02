"""Freeze a Codeforces effort route, then evaluate it once on sealed DeepSWE.

The freeze phase accepts only source outcomes, a passed external confirmation
report, and label-free target task text. The evaluation phase is separate and
opens the published target matrix only after the route decisions are written.
No fitted estimator is serialized in either phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np
from coding_model_router_codeforces_confirmation import (
    ALPHA,
    FROZEN_CANDIDATE,
    THRESHOLD,
    _scale,
    _score_features,
)
from coding_model_router_codeforces_fit import (
    ARMS,
    HIGH_INDEX,
    Data,
    _choose,
    _fit_delta_models,
    _score_delta_models,
    load_data,
)

logger = logging.getLogger("coding-router-codeforces-deepswe-transfer")
SEED = 20_260_731
BOOTSTRAPS = 10_000
TARGET_ARM = {
    "luna-low": "mini_swe_agent_gpt_5_6_luna_low",
    "luna-medium": "mini_swe_agent_gpt_5_6_luna_medium",
    "luna-high": "mini_swe_agent_gpt_5_6_luna_high",
    "luna-xhigh": "mini_swe_agent_gpt_5_6_luna_xhigh",
    "luna-max": "mini_swe_agent_gpt_5_6_luna_max",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return {str(key): item for key, item in value.items()}


def _target_structural(text: str) -> list[float]:
    """Map DeepSWE text to the frozen source feature schema without labels."""
    lower = text.casefold()
    return [
        math.log1p(len(text)),
        math.log1p(len(text.split())),
        math.log1p(len(text.splitlines())),
        0.0,
        0.0,
        0.0,
        float(text.count("`")),
        float(text.count("\n")),
        float(lower.count("input")),
        float(lower.count("output")),
        float(lower.count("example")),
        float(lower.count("constraint")),
        float("graph" in lower),
        float("tree" in lower),
        float("string" in lower),
        float("array" in lower),
        float("dynamic programming" in lower),
        0.0,
        0.0,
        0.0,
        0.0,
    ]


def _target_view(path: Path) -> tuple[list[str], list[str], list[str]]:
    raw = _read_object(path)
    if (
        raw.get("protocol") != "deepswe-label-free-task-feature-view-v2"
        or raw.get("target_reward_fields_accessed") is not False
        or raw.get("target_cost_fields_accessed") is not False
    ):
        raise ValueError("target feature view violated the label-free boundary")
    values = raw.get("rows")
    if not isinstance(values, list) or len(values) != 113:
        raise ValueError("target feature view must contain exactly 113 tasks")
    rows = [cast(dict[str, object], row) for row in values if isinstance(row, dict)]
    if len(rows) != len(values):
        raise ValueError("target feature view contains a non-object row")
    ids = [str(row.get("id") or "") for row in rows]
    texts = [str(row.get("text") or "") for row in rows]
    repositories = [str(row.get("repository") or "") for row in rows]
    if any(not value for value in (*ids, *texts, *repositories)):
        raise ValueError("target feature view contains an empty required field")
    if len(set(ids)) != len(ids):
        raise ValueError("target feature view contains duplicate task IDs")
    return ids, texts, repositories


def freeze_routes(
    *,
    development_corpus: Path,
    development_outcomes: Path,
    confirmation_report: Path,
    target_feature_view: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Fit on source only and freeze all label-free DeepSWE effort decisions."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    confirmation = _read_object(confirmation_report)
    inputs = confirmation.get("inputs")
    gate = confirmation.get("confirmation_gate")
    if (
        confirmation.get("frozen_candidate") != FROZEN_CANDIDATE
        or confirmation.get("deep_swe_evaluation_authorized") is not True
        or confirmation.get("target_outcomes_used") is not False
        or confirmation.get("target_embeddings_used") is not False
        or confirmation.get("no_persisted_fitted_model") is not True
        or not isinstance(gate, dict)
        or gate.get("passed") is not True
        or not isinstance(inputs, dict)
        or inputs.get("development_corpus_sha256") != _sha256(development_corpus)
        or inputs.get("development_outcomes_sha256") != _sha256(development_outcomes)
    ):
        raise ValueError("source confirmation does not authorize target transfer")
    ids, texts, repositories = _target_view(target_feature_view)
    development = load_data(development_corpus, development_outcomes, expected_tasks=160)
    target = Data(
        task_ids=ids,
        groups=repositories,
        texts=texts,
        structural=np.asarray([_target_structural(text) for text in texts], dtype=np.float64),
        rewards=np.empty((len(ids), len(ARMS)), dtype=np.float64),
        costs=np.empty((len(ids), len(ARMS)), dtype=np.float64),
    )
    scale = _scale(development)
    development_features = _score_features(development, scale)
    target_features = _score_features(target, scale)
    deltas = development.rewards - development.rewards[:, [HIGH_INDEX]]
    models = _fit_delta_models(
        development_features,
        deltas,
        np.arange(len(development.task_ids)),
        alpha=ALPHA,
    )
    predictions = _score_delta_models(
        target_features,
        np.arange(len(ids)),
        models,
    )
    choices = _choose(predictions, np.mean(development.costs, axis=0), threshold=THRESHOLD)
    output_dir.mkdir(parents=True)
    decisions_path = output_dir / "target-decisions.jsonl"
    decisions = [
        {
            "task_id": task_id,
            "repository": repository,
            "source_arm": ARMS[int(choice)],
            "target_arm": TARGET_ARM[ARMS[int(choice)]],
        }
        for task_id, repository, choice in zip(ids, repositories, choices, strict=True)
    ]
    decisions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions),
        encoding="utf-8",
    )
    counts = Counter(str(row["source_arm"]) for row in decisions)
    report: dict[str, object] = {
        "protocol": "codeforces-to-deepswe-direct-effort-route-freeze-v1",
        "frozen_candidate": FROZEN_CANDIDATE,
        "tasks": len(decisions),
        "arm_counts": {arm: counts.get(arm, 0) for arm in ARMS},
        "confirmation_report_sha256": _sha256(confirmation_report),
        "development_corpus_sha256": _sha256(development_corpus),
        "development_outcomes_sha256": _sha256(development_outcomes),
        "target_feature_view_sha256": _sha256(target_feature_view),
        "decisions_sha256": _sha256(decisions_path),
        "target_structural_adapter": (
            "text-derived shared fields; unavailable tests, limits, and Codeforces buckets zeroed"
        ),
        "target_reward_fields_accessed": False,
        "target_cost_fields_accessed": False,
        "target_outcomes_used": False,
        "no_persisted_fitted_model": True,
    }
    (output_dir / "target-route-freeze.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "froze source-only DeepSWE routes tasks=%d decisions_sha256=%s",
        len(decisions),
        report["decisions_sha256"],
    )
    return report


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _load_target_matrix(
    path: Path,
) -> tuple[list[str], np.ndarray, np.ndarray, set[str]]:
    raw = _read_object(path)
    values = raw.get("outcomes")
    if not isinstance(values, list):
        raise ValueError("DeepSWE matrix has no outcomes list")
    target_arms = [TARGET_ARM[arm] for arm in ARMS]
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
            raise ValueError(f"invalid target reward or cost for {task_id}")
        key = (task_id, arm)
        if key in cells:
            raise ValueError(f"duplicate target cell {key}")
        cells[key] = (reward, cost)
    complete = sorted(
        task_id for task_id in all_ids if all((task_id, arm) in cells for arm in target_arms)
    )
    if not complete:
        raise ValueError("DeepSWE has no complete five-effort tasks")
    rewards = np.asarray(
        [[cells[(task_id, TARGET_ARM[arm])][0] for arm in ARMS] for task_id in complete],
        dtype=np.float64,
    )
    costs = np.asarray(
        [[cells[(task_id, TARGET_ARM[arm])][1] for arm in ARMS] for task_id in complete],
        dtype=np.float64,
    )
    return complete, rewards, costs, all_ids


def _cluster_interval(groups: list[str], values: np.ndarray) -> tuple[float, float, float]:
    unique = sorted(set(groups))
    if len(unique) < 2 or values.shape != (len(groups),):
        raise ValueError("repository bootstrap inputs are invalid")
    by_group = {
        group: np.asarray([index for index, value in enumerate(groups) if value == group])
        for group in unique
    }
    rng = np.random.default_rng(SEED)
    samples = np.empty(BOOTSTRAPS, dtype=np.float64)
    for sample in range(BOOTSTRAPS):
        selected = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_group[str(group)] for group in selected])
        samples[sample] = float(np.mean(values[indices]))
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return float(np.mean(values)), float(lower), float(upper)


def evaluate_frozen_routes(
    *,
    freeze_path: Path,
    decisions_path: Path,
    target_matrix: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Open target outcomes once and score the already frozen decisions."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    freeze = _read_object(freeze_path)
    if (
        freeze.get("protocol") != "codeforces-to-deepswe-direct-effort-route-freeze-v1"
        or freeze.get("decisions_sha256") != _sha256(decisions_path)
        or freeze.get("target_outcomes_used") is not False
        or freeze.get("target_reward_fields_accessed") is not False
        or freeze.get("target_cost_fields_accessed") is not False
    ):
        raise ValueError("target route freeze is invalid")
    decisions = [
        cast(dict[str, object], json.loads(line))
        for line in decisions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(decisions) != 113:
        raise ValueError("target decision coverage is incomplete")
    by_id = {str(row["task_id"]): row for row in decisions}
    if len(by_id) != len(decisions):
        raise ValueError("target decisions contain duplicate task IDs")
    task_ids, rewards, costs, all_ids = _load_target_matrix(target_matrix)
    missing = sorted(set(task_ids) - set(by_id))
    if missing:
        raise ValueError(f"complete target tasks lack frozen routes: {missing[:5]}")
    rows = np.arange(len(task_ids))
    choices = np.asarray(
        [ARMS.index(str(by_id[task_id]["source_arm"])) for task_id in task_ids],
        dtype=np.int64,
    )
    groups = [str(by_id[task_id]["repository"]) for task_id in task_ids]
    routed_rewards = rewards[rows, choices]
    routed_costs = costs[rows, choices]
    mix = np.bincount(choices, minlength=len(ARMS)) / len(choices)
    blind_rewards = rewards @ mix
    blind_costs = costs @ mix
    reward_advantage = _cluster_interval(groups, routed_rewards - blind_rewards)
    cost_advantage = _cluster_interval(groups, routed_costs - blind_costs)
    router_reward = float(np.mean(routed_rewards))
    router_cost = float(np.sum(routed_costs))
    static = {
        arm: {
            "reward": float(np.mean(rewards[:, index])),
            "cost_usd": float(np.sum(costs[:, index])),
        }
        for index, arm in enumerate(ARMS)
    }
    dominated_by = [
        arm
        for arm, value in static.items()
        if float(value["reward"]) >= router_reward
        and float(value["cost_usd"]) <= router_cost
        and (float(value["reward"]) > router_reward or float(value["cost_usd"]) < router_cost)
    ]
    output_dir.mkdir(parents=True)
    evaluated_rows = output_dir / "evaluated-rows.jsonl"
    evaluated_rows.write_text(
        "".join(
            json.dumps(
                {
                    "task_id": task_id,
                    "repository": groups[index],
                    "source_arm": ARMS[int(choices[index])],
                    "target_arm": TARGET_ARM[ARMS[int(choices[index])]],
                    "reward": float(routed_rewards[index]),
                    "cost_usd": float(routed_costs[index]),
                    "matched_blind_reward": float(blind_rewards[index]),
                    "matched_blind_cost_usd": float(blind_costs[index]),
                },
                sort_keys=True,
            )
            + "\n"
            for index, task_id in enumerate(task_ids)
        ),
        encoding="utf-8",
    )
    transfer_gate = {
        "positive_matched_blind_reward_lower_bound": reward_advantage[1] > 0.0,
        "not_static_dominated": not dominated_by,
    }
    transfer_gate["passed"] = all(transfer_gate.values())
    report: dict[str, object] = {
        "protocol": "codeforces-to-deepswe-direct-effort-single-transfer-v1",
        "frozen_candidate": FROZEN_CANDIDATE,
        "target_tasks": len(task_ids),
        "target_repositories": len(set(groups)),
        "target_tasks_dropped_for_missing_cells": len(all_ids) - len(task_ids),
        "router": {
            "reward": router_reward,
            "cost_usd": router_cost,
            "arm_counts": {arm: int(np.sum(choices == index)) for index, arm in enumerate(ARMS)},
            "matched_blind_reward": float(np.mean(blind_rewards)),
            "matched_blind_cost_usd": float(np.sum(blind_costs)),
            "reward_advantage_95ci": reward_advantage,
            "cost_delta_usd_per_task_95ci": cost_advantage,
            "dominated_by_static_arms": dominated_by,
        },
        "static_efforts": static,
        "transfer_gate": transfer_gate,
        "freeze_sha256": _sha256(freeze_path),
        "decisions_sha256": _sha256(decisions_path),
        "target_matrix_sha256": _sha256(target_matrix),
        "evaluated_rows_sha256": _sha256(evaluated_rows),
        "target_routes_frozen_before_outcomes": True,
        "target_outcomes_used_for_fit": False,
        "target_outcomes_used_for_threshold": False,
        "target_outcomes_used_for_evaluation": True,
        "target_evaluation_count": 1,
        "no_persisted_fitted_model": True,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "single target transfer tasks=%d reward=%.6f cost_usd=%.2f passed=%s",
        len(task_ids),
        router_reward,
        router_cost,
        transfer_gate["passed"],
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--development-corpus", type=Path, required=True)
    freeze.add_argument("--development-outcomes", type=Path, required=True)
    freeze.add_argument("--confirmation-report", type=Path, required=True)
    freeze.add_argument("--target-feature-view", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--freeze", type=Path, required=True)
    evaluate.add_argument("--decisions", type=Path, required=True)
    evaluate.add_argument("--target-matrix", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    """Run the selected sealed-transfer phase."""
    args = _parser().parse_args()
    if args.command == "freeze":
        freeze_routes(
            development_corpus=args.development_corpus,
            development_outcomes=args.development_outcomes,
            confirmation_report=args.confirmation_report,
            target_feature_view=args.target_feature_view,
            output_dir=args.output_dir,
        )
    else:
        evaluate_frozen_routes(
            freeze_path=args.freeze,
            decisions_path=args.decisions,
            target_matrix=args.target_matrix,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
