"""Seal one zero-trace multi-model worker crash as a whole-task exclusion."""

from __future__ import annotations

import argparse
import json
import logging
import tarfile
from pathlib import Path
from typing import Any

import coding_model_router_swerebench_execute as runner

CORPUS = Path("/private/tmp/coding-router-swerebench-v40-corpus/development-tasks.json")
MODELS = {
    "gpt-5.6-sol": (
        "sol",
        Path("/private/tmp/coding-router-model-effort-v43-sol-development"),
    ),
    "gpt-5.6-terra": (
        "terra",
        Path("/private/tmp/coding-router-model-effort-v43-terra-development"),
    ),
}
USAGE_FIELDS = (
    "prompt_tokens",
    "cached_input_tokens",
    "completion_tokens",
    "reasoning_tokens",
)
logger = logging.getLogger("coding-router-swerebench-multimodel-exclude")


def _report(
    state: dict[str, Any],
    task_id: str,
    effort: str,
    model: str,
) -> dict[str, object]:
    if state.get("stage") != "failed" or state.get("task_id") != task_id:
        raise ValueError("task is not at the exact failed boundary")
    efforts = state.get("efforts")
    attempts = state.get("sandbox_attempts")
    if not isinstance(efforts, dict) or effort in efforts:
        raise ValueError("failed effort was already accepted")
    if not isinstance(attempts, list) or len(attempts) != 1:
        raise ValueError("task does not have exactly one preserved worker attempt")
    attempt = attempts[0]
    processes = attempt.get("effort_processes") if isinstance(attempt, dict) else None
    process = processes.get(effort) if isinstance(processes, dict) else None
    if (
        not isinstance(attempt, dict)
        or attempt.get("terminated") is True
        or not isinstance(attempt.get("sandbox_id"), str)
        or not isinstance(process, dict)
        or process.get("scientific_command_starts") != 1
        or process.get("completed") is not True
        or process.get("exit_code") != 137
        or "expected 2 outer rows, found 0" not in str(state.get("error"))
    ):
        raise ValueError("failure is not the frozen zero-trace exit-137 boundary")
    return {
        "protocol": "coding-router-model-effort-worker-exclusion-v1",
        "valid": True,
        "task_id": task_id,
        "model": model,
        "effort": effort,
        "scope": "whole-task",
        "reason": "E2B worker exited 137 before durable trace persistence",
        "sandbox_id": attempt["sandbox_id"],
        "scientific_command_starts": 1,
        "scientific_exit_code": 137,
        "durable_trace_rows": 0,
        "observed_scientific_cells": 0,
        "scientific_cells_rerun": 0,
        "provider_calls": 0,
        "provider_calls_provenance": (
            "zero durable trace rows; actual provider usage unavailable after worker crash"
        ),
        "usage": {field: 0 for field in USAGE_FIELDS},
        "usage_provenance": "zero trace-derived lower bound; actual usage unavailable",
        "target_outcomes_used": False,
    }


def _active_sandbox(
    sandbox_id: str,
    task_id: str,
    task_index: int,
    model_slug: str,
) -> None:
    paginator = runner.Sandbox.list(limit=100)
    while True:
        for item in paginator.next_items():
            if item.sandbox_id != sandbox_id:
                continue
            expected = {
                "owner": "coding-router-v43",
                "phase": f"model-effort-v43-{model_slug}-development",
                "task_id": task_id,
                "task_index": str(task_index),
            }
            if item.metadata != expected:
                raise ValueError("preserved sandbox metadata changed")
            return
        if not paginator.has_next:
            break
    raise ValueError("exact preserved failure sandbox is no longer active")


def main() -> None:
    """Write evidence, terminate the exact worker, and exclude the whole task."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--effort", choices=runner.EFFORTS, required=True)
    args = parser.parse_args()
    slug, root = MODELS[args.model]
    corpus = runner._read_object(CORPUS)
    tasks = corpus.get("tasks")
    if not isinstance(tasks, list) or args.task_index not in range(len(tasks)):
        raise ValueError("task index is outside the frozen corpus")
    row = tasks[args.task_index]
    if not isinstance(row, dict) or not isinstance(row.get("task_id"), str):
        raise ValueError("frozen corpus task identity is invalid")
    task_id = row["task_id"]
    task_dir = root / "tasks" / f"{args.task_index:04d}"
    state_path = task_dir / "state.json"
    state = runner._read_object(state_path)
    report = _report(state, task_id, args.effort, args.model)
    sandbox_id = str(report["sandbox_id"])
    _active_sandbox(sandbox_id, task_id, args.task_index, slug)

    report_path = task_dir / f"{args.effort}.infrastructure-missing.json"
    archive_path = task_dir / f"{args.effort}.infrastructure-missing.tar.gz"
    snapshot_path = task_dir / f"{args.effort}.pre-exclusion-state.json"
    if report_path.exists() or archive_path.exists() or snapshot_path.exists():
        raise FileExistsError("failure evidence already exists")
    runner._write_json(snapshot_path, state)
    runner._write_json(report_path, report)
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(report_path, arcname=report_path.name)
        archive.add(snapshot_path, arcname=snapshot_path.name)

    runner.Sandbox.connect(sandbox_id).kill()
    attempt = state["sandbox_attempts"][0]
    attempt["terminated"] = True
    attempt["excluded_effort"] = args.effort
    state.setdefault("excluded_errors", []).append(
        {"error": state.pop("error", None), "scientific_cells_rerun": 0}
    )
    state["sandbox_terminated"] = True
    state["stage"] = "excluded-infrastructure"
    state["exclusion"] = {
        "scope": "whole-task",
        "effort": args.effort,
        "reason": report["reason"],
        "evidence_sha256": runner._sha256(archive_path),
        "report_sha256": runner._sha256(report_path),
        "observed_scientific_cells": 0,
        "scientific_cells_rerun": 0,
        "provider_calls": 0,
        "usage": report["usage"],
        "provider_usage_available": False,
    }
    runner._write_json(state_path, state)
    launch = runner._read_object(root / "launch.json")
    prior_spend = launch.get("prior_spend_usd")
    if isinstance(prior_spend, bool) or not isinstance(prior_spend, (int, float)):
        raise ValueError("launch has invalid prior spend")
    runner.MODEL = args.model
    runner._update_summary(
        root,
        200,
        protocol=f"coding-router-swerebench-{slug}-development-v1",
        prior_spend_usd=float(prior_spend),
    )
    logger.info(
        "%s",
        json.dumps(
            {
                "task_id": task_id,
                "model": args.model,
                "effort": args.effort,
                "evidence_sha256": state["exclusion"]["evidence_sha256"],
                "scientific_cells_rerun": 0,
                "provider_usage_available": False,
                "sandbox_terminated": True,
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
