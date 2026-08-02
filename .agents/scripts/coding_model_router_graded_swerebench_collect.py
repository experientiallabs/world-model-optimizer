"""Collect a complete graded SWE-rebench development matrix for remote fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger("coding-router-graded-swerebench-collect")

PROTOCOL = "coding-router-graded-swerebench-development-collection-v1"
EXECUTION_PROTOCOL = "coding-router-graded-swerebench-development-execution-v1"
PHASE_NAME = "development"
CORPUS_SHA256 = "48d88436a083b66972c25cd7d9439fd149c95bcf9caded2bab7f3b6453aea3d5"
SOURCE_TASKS = 673
MIN_RETAINED_TASKS = 640
ARMS = (
    "luna-low",
    "luna-medium",
    "luna-high",
    "luna-xhigh",
    "luna-max",
    "sol-max",
)
ARM_MODELS = {arm: "gpt-5.6-sol" if arm == "sol-max" else "gpt-5.6-luna" for arm in ARMS}
PRICES = {
    "gpt-5.6-luna": (1.0, 0.1, 6.0),
    "gpt-5.6-sol": (5.0, 0.5, 30.0),
}
USAGE_FIELDS = (
    "prompt_tokens",
    "cached_input_tokens",
    "completion_tokens",
    "reasoning_tokens",
)
EMPTY_PATCH_SHA256 = hashlib.sha256(b"").hexdigest()
PATCH_PROVENANCE_VALUES = {
    "post-execution agent failure",
    "official trace reported no source changes",
    "official captured patch",
}
USAGE_PROVENANCE_VALUES = {
    "exact token counts from pinned verifier Responses trace",
    "mixed exact and conservative trace-derived token estimate",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    return {str(key): item for key, item in value.items()}


def _read_object(path: Path) -> dict[str, Any]:
    return _object(json.loads(path.read_text(encoding="utf-8")), str(path))


def _usage(value: object, label: str) -> dict[str, int]:
    raw = _object(value, label)
    usage: dict[str, int] = {}
    for field in USAGE_FIELDS:
        amount = raw.get(field)
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ValueError(f"{label} has invalid {field}")
        usage[field] = amount
    if usage["reasoning_tokens"] > usage["completion_tokens"]:
        raise ValueError(f"{label} reasoning exceeds completion tokens")
    return usage


def _cost(model: str, usage: dict[str, int]) -> float:
    input_rate, cached_rate, output_rate = PRICES[model]
    return (
        usage["prompt_tokens"] * input_rate
        + usage["cached_input_tokens"] * cached_rate
        + usage["completion_tokens"] * output_rate
    ) / 1_000_000


def _patch_provenance(
    report: dict[str, Any], label: str
) -> tuple[str, str]:
    """Validate patch evidence and support the frozen legacy report schema."""
    patch_bytes = report.get("patch_bytes")
    patch_sha256 = report.get("patch_sha256")
    official = report.get("official_verifier_reached")
    if isinstance(patch_bytes, bool) or not isinstance(patch_bytes, int) or patch_bytes < 0:
        raise ValueError(f"{label} has invalid patch bytes")

    declared = report.get("patch_provenance")
    valid_sha = (
        isinstance(patch_sha256, str)
        and len(patch_sha256) == 64
        and all(character in "0123456789abcdef" for character in patch_sha256)
    )
    if declared == "post-execution agent failure":
        consistent = patch_bytes == 0 and patch_sha256 is None and official is False
    elif declared == "official trace reported no source changes":
        consistent = (
            patch_bytes == 0
            and patch_sha256 == EMPTY_PATCH_SHA256
            and official is True
        )
    elif declared == "official captured patch":
        consistent = valid_sha and official is True
    elif declared is None:
        if patch_bytes > 0 and valid_sha and official is True:
            return (
                "official captured patch",
                "inferred from validated legacy report fields",
            )
        raise ValueError(f"{label} has ambiguous legacy patch evidence")
    else:
        raise ValueError(f"{label} has inconsistent patch provenance")
    if not consistent:
        raise ValueError(f"{label} has inconsistent patch evidence")
    return str(declared), "declared by arm validator report"


def _excluded(state: dict[str, Any]) -> bool:
    exclusion = state.get("exclusion")
    return (
        isinstance(exclusion, dict)
        and (state.get("stage"), exclusion.get("reason"))
        in {
            (
                "excluded-audit-artifact-loss",
                "validator rejected official no-change trace",
            ),
            (
                "excluded-ungradeable-scientific-cell",
                "official trace lacked a graded reward after one frozen attempt",
            ),
            (
                "excluded-ungradeable-scientific-cell",
                "scientific artifact became irrecoverable after E2B transport loss",
            ),
            (
                "excluded-ungradeable-scientific-cell",
                "official graded trace became irrecoverable after missing usage audit failure",
            ),
        }
        and exclusion.get("scope") == "whole-task"
        and exclusion.get("observed_scientific_cells") == 1
        and exclusion.get("scientific_cells_rerun") == 0
        and exclusion.get("provider_usage_recoverable") is False
    )


def collect(root: Path, corpus_path: Path, output: Path) -> None:
    """Validate every retained arm and emit a dense one-attempt matrix."""
    if output.exists():
        raise FileExistsError(output)
    if _sha256(corpus_path) != CORPUS_SHA256:
        raise ValueError(f"{PHASE_NAME} corpus changed")
    launch_path = root / "launch.json"
    launch = _read_object(launch_path)
    if (
        launch.get("protocol") != EXECUTION_PROTOCOL
        or launch.get("corpus_sha256") != CORPUS_SHA256
        or launch.get("tasks") != SOURCE_TASKS
        or launch.get("attempts_per_arm") != 1
        or launch.get("deep_swe_outcomes_accessed") is not False
        or launch.get("confirmation_outcomes_accessed") is not False
        or launch.get("model_persisted") is not False
    ):
        raise ValueError(f"{PHASE_NAME} launch manifest is invalid")
    progress_path = root / "progress.json"
    progress = _read_object(progress_path)
    if (
        progress.get("protocol") != EXECUTION_PROTOCOL
        or progress.get("total_tasks") != SOURCE_TASKS
        or progress.get("complete_tasks") != 649
        or progress.get("excluded_tasks") != 24
        or progress.get("failed_tasks") != 0
        or progress.get("completed_scientific_cells") != 3941
        or not isinstance(progress.get("matrix_cost_usd"), (int, float))
        or not isinstance(
            progress.get("rough_cumulative_experiment_spend_usd"), (int, float)
        )
    ):
        raise ValueError(f"{PHASE_NAME} progress summary is invalid")
    corpus = _read_object(corpus_path)
    raw_tasks = corpus.get("tasks")
    if not isinstance(raw_tasks, list) or len(raw_tasks) != SOURCE_TASKS:
        raise ValueError(f"{PHASE_NAME} corpus must contain {SOURCE_TASKS} tasks")
    tasks = [_object(task, f"task {index}") for index, task in enumerate(raw_tasks)]

    outcomes: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    artifact_hashes: dict[str, dict[str, str]] = {}
    usage_provenance_counts = {value: 0 for value in USAGE_PROVENANCE_VALUES}
    total_cost = 0.0
    for index, task in enumerate(tasks):
        task_id = str(task["task_id"])
        task_dir = root / "tasks" / f"{index:04d}"
        state_path = task_dir / "state.json"
        state = _read_object(state_path)
        if state.get("protocol") != EXECUTION_PROTOCOL or state.get("task_id") != task_id:
            raise ValueError(f"task state identity drift: {task_id}")
        if _excluded(state):
            exclusion = _object(state["exclusion"], f"task {index} exclusion")
            exclusions.append(
                {
                    "task_id": task_id,
                    "task_index": index,
                    "scope": "whole-task",
                    "arm": exclusion["arm"],
                    "reason": exclusion["reason"],
                    "observed_scientific_cells": 1,
                    "scientific_cells_rerun": 0,
                    "provider_usage_recoverable": False,
                    "state_sha256": _sha256(state_path),
                }
            )
            artifact_hashes[task_id] = {"state": _sha256(state_path)}
            continue
        if state.get("stage") != "complete":
            raise ValueError(f"task is not complete: {task_id}")
        arm_states = _object(state.get("arms"), f"task {index} arms")
        artifact_hashes[task_id] = {"state": _sha256(state_path)}
        for arm in ARMS:
            arm_state = _object(arm_states.get(arm), f"task {index} {arm} state")
            model = ARM_MODELS[arm]
            report_path = task_dir / f"{arm}.report.json"
            archive_path = task_dir / f"{arm}.tar.gz"
            if (
                _sha256(report_path) != arm_state.get("report_sha256")
                or _sha256(archive_path) != arm_state.get("archive_sha256")
            ):
                raise ValueError(f"artifact hash mismatch: {task_id}/{arm}")
            report = _read_object(report_path)
            reward = report.get("reward")
            f2p_passed = report.get("f2p_passed")
            f2p_total = report.get("f2p_total")
            if (
                report.get("valid") is not True
                or report.get("task_id") != task_id
                or report.get("arm") != arm
                or report.get("model") != model
                or report.get("effort") != arm.split("-", 1)[1]
                or isinstance(reward, bool)
                or not isinstance(reward, (int, float))
                or not math.isfinite(float(reward))
                or not 0.0 <= float(reward) <= 1.0
                or isinstance(f2p_passed, bool)
                or not isinstance(f2p_passed, int)
                or f2p_total != task["f2p_total"]
                or not 0 <= f2p_passed <= int(f2p_total)
                or abs(float(reward) - f2p_passed / int(f2p_total)) > 1e-9
            ):
                raise ValueError(f"invalid graded report: {task_id}/{arm}")
            usage = _usage(report.get("usage"), f"task {index} {arm} usage")
            usage_provenance = report.get("usage_provenance")
            if usage_provenance not in USAGE_PROVENANCE_VALUES:
                raise ValueError(f"task {index} {arm} has invalid usage provenance")
            usage_provenance_counts[str(usage_provenance)] += 1
            patch_provenance, patch_provenance_source = _patch_provenance(
                report, f"task {index} {arm}"
            )
            cost = _cost(model, usage)
            total_cost += cost
            outcomes.append(
                {
                    "task_id": task_id,
                    "repository": task["repository"],
                    "language": task["language"],
                    "prompt": task["prompt"],
                    "prompt_sha256": task["prompt_sha256"],
                    "arm": arm,
                    "model": model,
                    "reasoning_effort": report["effort"],
                    "attempt_number": 0,
                    "reward": float(reward),
                    "f2p_passed": f2p_passed,
                    "f2p_total": f2p_total,
                    "reward_provenance": report["reward_provenance"],
                    "official_verifier_reached": report["official_verifier_reached"],
                    "cost_usd": cost,
                    "cost_provenance": "trace-derived frozen list-price estimate",
                    "usage": usage,
                    "usage_provenance": usage_provenance,
                    "provider_calls": report["provider_calls"],
                    "stop_condition": report["stop_condition"],
                    "patch_sha256": report["patch_sha256"],
                    "patch_provenance": patch_provenance,
                    "patch_provenance_source": patch_provenance_source,
                    "target_outcomes_used": False,
                    "split": PHASE_NAME,
                }
            )
            artifact_hashes[task_id][f"{arm}_report"] = _sha256(report_path)
            artifact_hashes[task_id][f"{arm}_archive"] = _sha256(archive_path)

    retained = SOURCE_TASKS - len(exclusions)
    expected_cells = retained * len(ARMS)
    identities = {(row["task_id"], row["arm"]) for row in outcomes}
    if (
        retained < MIN_RETAINED_TASKS
        or len(outcomes) != expected_cells
        or len(identities) != expected_cells
    ):
        raise ValueError("whole-task-intersected matrix is incomplete")
    outcomes.sort(key=lambda row: (str(row["task_id"]), str(row["arm"])))
    output.mkdir(parents=True)
    outcomes_path = output / "outcomes.jsonl"
    outcomes_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in outcomes),
        encoding="utf-8",
    )
    audit = {
        "protocol": PROTOCOL,
        "valid": True,
        "source_tasks": SOURCE_TASKS,
        "tasks": retained,
        "retained_task_coverage": retained / SOURCE_TASKS,
        "excluded_tasks": exclusions,
        "arms": list(ARMS),
        "attempts_per_arm": 1,
        "cells": expected_cells,
        "unique_cell_identities": len(identities),
        "retained_matrix_cost_usd": total_cost,
        "spent_matrix_cost_usd": float(progress["matrix_cost_usd"]),
        "rough_cumulative_experiment_spend_usd": float(
            progress["rough_cumulative_experiment_spend_usd"]
        ),
        "unmetered_excluded_scientific_cells": len(exclusions),
        "unmetered_excluded_cost_provenance": "user monitors provider usage externally",
        "usage_provenance_counts": usage_provenance_counts,
        "target_outcomes_used": False,
        "deep_swe_outcomes_accessed": False,
        "confirmation_outcomes_accessed": PHASE_NAME == "confirmation",
        "fitted_numeric_router_state_persisted": False,
        "corpus_sha256": CORPUS_SHA256,
        "launch_sha256": _sha256(launch_path),
        "progress_sha256": _sha256(progress_path),
        "outcomes_sha256": _sha256(outcomes_path),
        "artifact_hashes": artifact_hashes,
    }
    (output / "completion-audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "collected tasks=%d cells=%d excluded=%d cost_usd=%.6f",
        retained,
        expected_cells,
        len(exclusions),
        total_cost,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(args.root, args.corpus, args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
