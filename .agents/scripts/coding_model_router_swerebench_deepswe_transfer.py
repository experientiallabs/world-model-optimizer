"""Freeze an externally confirmed SWE-rebench route, then open DeepSWE once."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from coding_model_router_codeforces_deepswe_transfer import (
    TARGET_ARM,
    _cluster_interval,
    _load_target_matrix,
)
from coding_model_router_swerebench_fit import (
    ARMS,
    Candidate,
    _fit_router,
    _latency,
    candidate_grid,
    load_source,
)

logger = logging.getLogger("coding-router-swerebench-deepswe-transfer")

TARGET_MATRIX_SHA256 = "2988742e48b1c9bfec8dc45d88af112c46c45367529d1936b709e4b4e549835f"
CONFIRMATION_PROTOCOL = "coding-router-swerebench-confirmation-analysis-v1"
TARGET_TASKS = 113


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _target_tasks(path: Path) -> list[dict[str, Any]]:
    raw = _read_object(path)
    if (
        raw.get("protocol") != "deepswe-label-free-task-feature-view-v2"
        or raw.get("target_reward_fields_accessed") is not False
        or raw.get("target_cost_fields_accessed") is not False
    ):
        raise ValueError("DeepSWE feature view violated the label-free boundary")
    values = raw.get("rows")
    if not isinstance(values, list) or len(values) != TARGET_TASKS:
        raise ValueError("DeepSWE feature view must contain exactly 113 tasks")
    tasks: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise ValueError(f"DeepSWE feature row {index} is invalid")
        task_id = value.get("id")
        prompt = value.get("text")
        repository = value.get("repository")
        if not all(isinstance(item, str) and item for item in (task_id, prompt, repository)):
            raise ValueError(f"DeepSWE feature row {index} is incomplete")
        tasks.append(
            {
                "task_id": task_id,
                "repository": repository,
                "language": value.get("language")
                if isinstance(value.get("language"), str) and value.get("language")
                else "unknown",
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(str(prompt).encode()).hexdigest(),
            }
        )
    if len({str(task["task_id"]) for task in tasks}) != TARGET_TASKS:
        raise ValueError("DeepSWE feature view contains duplicate task IDs")
    return tasks


def _selected_candidate(lock: dict[str, Any]) -> Candidate:
    selected_key = lock.get("selected_key")
    selected_config = lock.get("selected_config")
    if not isinstance(selected_key, str) or not isinstance(selected_config, dict):
        raise ValueError("selection lock lacks the frozen candidate")
    canonical = json.dumps(selected_config, sort_keys=True, separators=(",", ":"))
    if lock.get("selected_config_sha256") != hashlib.sha256(canonical.encode()).hexdigest():
        raise ValueError("selection lock candidate config hash changed")
    candidates = {candidate.key: candidate for candidate in candidate_grid()}
    candidate = candidates.get(selected_key)
    if candidate is None or candidate.config() != selected_config:
        raise ValueError("selection lock candidate is outside the preregistered grid")
    return candidate


def freeze_routes(
    *,
    development_corpus: Path,
    development_outcomes: Path,
    development_audit: Path,
    selection_lock: Path,
    confirmation_report: Path,
    target_feature_view: Path,
    output: Path,
) -> dict[str, object]:
    """Refit on external development only and freeze label-free DeepSWE decisions."""
    if output.exists():
        raise FileExistsError(f"DeepSWE route freeze already exists: {output}")
    confirmation = _read_object(confirmation_report)
    confirmation_inputs = confirmation.get("inputs")
    if (
        confirmation.get("protocol") != CONFIRMATION_PROTOCOL
        or confirmation.get("confirmation_passed") is not True
        or confirmation.get("target_outcomes_used") is not False
        or confirmation.get("deep_swe_outcomes_accessed") is not False
        or not isinstance(confirmation_inputs, dict)
        or confirmation_inputs.get("selection_lock_sha256") != _sha256(selection_lock)
    ):
        raise ValueError("external confirmation does not authorize DeepSWE transfer")
    lock = _read_object(selection_lock)
    if (
        lock.get("development_corpus_sha256") != _sha256(development_corpus)
        or lock.get("development_outcomes_sha256") != _sha256(development_outcomes)
        or lock.get("collection_audit_sha256") != _sha256(development_audit)
        or lock.get("target_outcomes_used") is not False
        or lock.get("deep_swe_outcomes_accessed") is not False
        or lock.get("confirmation_outcomes_accessed") is not False
    ):
        raise ValueError("selection lock no longer matches external development inputs")
    candidate = _selected_candidate(lock)
    source = load_source(development_corpus, development_outcomes, development_audit)
    target_tasks = _target_tasks(target_feature_view)
    with tempfile.TemporaryDirectory(prefix="swerebench-deepswe-freeze-") as temporary:
        route = _fit_router(
            source,
            candidate,
            Path(temporary) / "ephemeral.bank.npz",
            label_rewards=source.data.rewards,
        )
        latency = _latency(route, target_tasks)
        choices = [route(task) for task in target_tasks]
    if latency.get("passed") is not True:
        raise ValueError("frozen DeepSWE route exceeded the pre-inference latency gate")
    if any(not isinstance(choice, int) or not 0 <= choice < len(ARMS) for choice in choices):
        raise ValueError("frozen router selected an invalid target effort")
    output.mkdir(parents=True)
    decisions_path = output / "target-decisions.jsonl"
    decisions = [
        {
            "task_id": str(task["task_id"]),
            "repository": str(task["repository"]),
            "prompt_sha256": str(task["prompt_sha256"]),
            "source_arm": ARMS[choice],
            "target_arm": TARGET_ARM[ARMS[choice]],
        }
        for task, choice in zip(target_tasks, choices, strict=True)
    ]
    decisions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions),
        encoding="utf-8",
    )
    counts = Counter(str(row["source_arm"]) for row in decisions)
    report: dict[str, object] = {
        "protocol": "swerebench-to-deepswe-effort-route-freeze-v1",
        "selected_key": candidate.key,
        "selected_config_sha256": lock["selected_config_sha256"],
        "tasks": len(decisions),
        "arm_counts": {arm: counts.get(arm, 0) for arm in ARMS},
        "latency": latency,
        "decisions_sha256": _sha256(decisions_path),
        "confirmation_report_sha256": _sha256(confirmation_report),
        "development_corpus_sha256": _sha256(development_corpus),
        "development_outcomes_sha256": _sha256(development_outcomes),
        "development_audit_sha256": _sha256(development_audit),
        "selection_lock_sha256": _sha256(selection_lock),
        "target_feature_view_sha256": _sha256(target_feature_view),
        "target_language_adapter": "label-free language when present, otherwise unknown",
        "target_reward_fields_accessed": False,
        "target_cost_fields_accessed": False,
        "target_outcomes_used": False,
        "deep_swe_outcomes_accessed": False,
        "fitted_numeric_state_persisted": False,
    }
    (output / "target-route-freeze.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("froze DeepSWE decisions tasks=%d key=%s", len(decisions), candidate.key)
    return report


def evaluate_routes(
    *,
    freeze: Path,
    decisions: Path,
    target_matrix: Path,
    output: Path,
) -> dict[str, object]:
    """Open the hash-pinned DeepSWE matrix once after decisions are immutable."""
    if output.exists():
        raise FileExistsError(f"DeepSWE evaluation already exists: {output}")
    if _sha256(target_matrix) != TARGET_MATRIX_SHA256:
        raise ValueError("DeepSWE matrix hash changed")
    freeze_report = _read_object(freeze)
    if (
        freeze_report.get("protocol") != "swerebench-to-deepswe-effort-route-freeze-v1"
        or freeze_report.get("decisions_sha256") != _sha256(decisions)
        or freeze_report.get("target_outcomes_used") is not False
        or freeze_report.get("deep_swe_outcomes_accessed") is not False
        or freeze_report.get("fitted_numeric_state_persisted") is not False
    ):
        raise ValueError("DeepSWE route freeze is invalid")
    decision_rows = _read_rows(decisions)
    by_task = {str(row.get("task_id")): row for row in decision_rows}
    if len(decision_rows) != TARGET_TASKS or len(by_task) != TARGET_TASKS:
        raise ValueError("DeepSWE decisions do not cover 113 unique tasks")
    task_ids, rewards, costs, all_ids = _load_target_matrix(target_matrix)
    if any(task_id not in by_task for task_id in task_ids):
        raise ValueError("complete DeepSWE tasks lack frozen decisions")
    rows = np.arange(len(task_ids))
    choices = np.asarray(
        [ARMS.index(str(by_task[task_id]["source_arm"])) for task_id in task_ids],
        dtype=np.int64,
    )
    repositories = [str(by_task[task_id]["repository"]) for task_id in task_ids]
    routed_reward = rewards[rows, choices]
    routed_cost = costs[rows, choices]
    traffic = np.bincount(choices, minlength=len(ARMS)) / len(choices)
    blind_reward = rewards @ traffic
    blind_cost = costs @ traffic
    reward_interval = _cluster_interval(repositories, routed_reward - blind_reward)
    cost_interval = _cluster_interval(repositories, routed_cost - blind_cost)
    route_reward = float(np.mean(routed_reward))
    route_cost = float(np.sum(routed_cost))
    static = {
        arm: {
            "reward": float(np.mean(rewards[:, index])),
            "cost_usd": float(np.sum(costs[:, index])),
        }
        for index, arm in enumerate(ARMS)
    }
    dominated = [
        arm
        for arm, value in static.items()
        if float(value["reward"]) >= route_reward and float(value["cost_usd"]) <= route_cost
    ]
    transfer_gate = {
        "positive_matched_blind_reward_lower_bound": reward_interval[1] > 0.0,
        "not_static_dominated": not dominated,
    }
    transfer_gate["passed"] = all(transfer_gate.values())
    output.mkdir(parents=True)
    evaluated_rows = output / "evaluated-rows.jsonl"
    evaluated_rows.write_text(
        "".join(
            json.dumps(
                {
                    "task_id": task_id,
                    "repository": repositories[index],
                    "source_arm": ARMS[int(choices[index])],
                    "target_arm": TARGET_ARM[ARMS[int(choices[index])]],
                    "reward": float(routed_reward[index]),
                    "cost_usd": float(routed_cost[index]),
                    "matched_blind_reward": float(blind_reward[index]),
                    "matched_blind_cost_usd": float(blind_cost[index]),
                },
                sort_keys=True,
            )
            + "\n"
            for index, task_id in enumerate(task_ids)
        ),
        encoding="utf-8",
    )
    report: dict[str, object] = {
        "protocol": "swerebench-to-deepswe-effort-single-transfer-v1",
        "target_matrix_sha256": TARGET_MATRIX_SHA256,
        "target_tasks": len(task_ids),
        "target_repositories": len(set(repositories)),
        "target_tasks_dropped_for_missing_cells": len(all_ids) - len(task_ids),
        "router": {
            "reward": route_reward,
            "cost_usd": route_cost,
            "arm_counts": {
                arm: int(np.sum(choices == index)) for index, arm in enumerate(ARMS)
            },
            "matched_blind_reward": float(np.mean(blind_reward)),
            "matched_blind_cost_usd": float(np.sum(blind_cost)),
            "reward_advantage_95ci": reward_interval,
            "cost_delta_usd_per_task_95ci": cost_interval,
            "dominated_by_static_arms": dominated,
        },
        "static_efforts": static,
        "transfer_gate": transfer_gate,
        "freeze_sha256": _sha256(freeze),
        "decisions_sha256": _sha256(decisions),
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
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "single DeepSWE transfer tasks=%d reward=%.6f cost=%.2f passed=%s",
        len(task_ids),
        route_reward,
        route_cost,
        transfer_gate["passed"],
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--development-corpus", type=Path, required=True)
    freeze.add_argument("--development-outcomes", type=Path, required=True)
    freeze.add_argument("--development-audit", type=Path, required=True)
    freeze.add_argument("--selection-lock", type=Path, required=True)
    freeze.add_argument("--confirmation-report", type=Path, required=True)
    freeze.add_argument("--target-feature-view", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--freeze", type=Path, required=True)
    evaluate.add_argument("--decisions", type=Path, required=True)
    evaluate.add_argument("--target-matrix", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    """Run exactly one sealed transfer phase."""
    args = _parser().parse_args()
    if args.command == "freeze":
        freeze_routes(
            development_corpus=args.development_corpus,
            development_outcomes=args.development_outcomes,
            development_audit=args.development_audit,
            selection_lock=args.selection_lock,
            confirmation_report=args.confirmation_report,
            target_feature_view=args.target_feature_view,
            output=args.output,
        )
    else:
        evaluate_routes(
            freeze=args.freeze,
            decisions=args.decisions,
            target_matrix=args.target_matrix,
            output=args.output,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
