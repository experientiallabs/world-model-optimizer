"""Audit and relabel pre-existing Moonshiner cells as attempt-zero inputs.

Task selection is already frozen before this script reads outcomes. Only tasks
with all five efforts, matching seed fingerprints, exact model attestation, and
hash-verified raw traces are reused. Incomplete tasks remain scheduled for fresh
execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("coding-model-router-moonshiner-reuse")

ARMS = {
    "luna-low",
    "luna-medium",
    "luna-high",
    "luna-xhigh",
    "luna-max",
}


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assemble(
    corpus_path: Path,
    source_corpus_path: Path,
    sources: list[Path],
    output: Path,
) -> None:
    corpus = _read_object(corpus_path)
    raw_tasks = corpus.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError(f"{corpus_path} has no tasks")
    tasks = {
        str(task["task_id"]): {str(key): item for key, item in task.items()}
        for task in raw_tasks
        if isinstance(task, dict) and isinstance(task.get("task_id"), str)
    }
    if len(tasks) != len(raw_tasks):
        raise ValueError(f"{corpus_path} contains an invalid task")
    source_corpus = _read_object(source_corpus_path)
    raw_source_tasks = source_corpus.get("tasks")
    if not isinstance(raw_source_tasks, list):
        raise ValueError(f"{source_corpus_path} has no tasks")
    source_tasks = {
        str(task["task_id"]): {str(key): item for key, item in task.items()}
        for task in raw_source_tasks
        if isinstance(task, dict) and isinstance(task.get("task_id"), str)
    }
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    runtime_fingerprints: dict[str, str] = {}
    source_inventory: list[dict[str, Any]] = []
    for source in sources:
        rows = _read_rows(source)
        source_inventory.append(
            {
                "path": str(source),
                "sha256": _sha256(source),
                "rows": len(rows),
            }
        )
        for row in rows:
            task_id = row.get("task_id")
            arm = row.get("arm")
            if not isinstance(task_id, str) or task_id not in tasks or arm not in ARMS:
                continue
            if source_tasks.get(task_id) != tasks[task_id]:
                raise ValueError(f"reusable task {task_id} differs from frozen source")
            key = (task_id, str(arm))
            if key in cells:
                raise ValueError(f"duplicate reusable cell: {key}")
            required = {
                "model": "gpt-5.6-luna",
                "observed_model": "gpt-5.6-luna",
                "model_attested": True,
                "protected_intact": True,
                "workspace_removed": True,
                "target_outcomes_used": False,
            }
            for field, expected in required.items():
                if row.get(field) != expected:
                    raise ValueError(f"reusable cell {key} has invalid {field}")
            runtime_fingerprint = row.get("seed_fingerprint")
            if not isinstance(runtime_fingerprint, str) or not runtime_fingerprint:
                raise ValueError(f"reusable cell {key} has no runtime fingerprint")
            previous_fingerprint = runtime_fingerprints.setdefault(
                task_id, runtime_fingerprint
            )
            if previous_fingerprint != runtime_fingerprint:
                raise ValueError(f"reusable task {task_id} changed runtime fingerprint")
            trace_path = row.get("trace_path")
            if not isinstance(trace_path, str):
                raise ValueError(f"reusable cell {key} has no raw trace path")
            trace = source.parent / "traces" / trace_path
            if not trace.is_file() or _sha256(trace) != row.get("raw_sha256"):
                raise ValueError(f"reusable cell {key} has invalid raw trace")
            migrated = dict(row)
            migrated.update(
                {
                    "protocol": "moonshiner-effort-reused-outcome-v1",
                    "original_cell_id": row.get("cell_id"),
                    "cell_id": f"{task_id}:{arm}:attempt-0",
                    "attempt": 0,
                    "reused_without_provider_call": True,
                    "reuse_source": str(source),
                    "reuse_source_sha256": _sha256(source),
                }
            )
            cells[key] = migrated
    reusable_tasks = sorted(
        task_id
        for task_id in tasks
        if {arm for candidate, arm in cells if candidate == task_id} == ARMS
    )
    rows = [
        cells[(task_id, arm)]
        for task_id in reusable_tasks
        for arm in sorted(ARMS)
    ]
    missing_tasks = [task for task_id, task in tasks.items() if task_id not in reusable_tasks]
    output.mkdir(parents=True, exist_ok=False)
    outcomes_path = output / "outcomes.jsonl"
    outcomes_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    missing_corpus = dict(corpus)
    missing_corpus["tasks"] = missing_tasks
    missing_corpus["attempt_zero_missing_only"] = True
    missing_path = output / "missing-tasks.json"
    missing_path.write_text(
        json.dumps(missing_corpus, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "protocol": "moonshiner-effort-reuse-manifest-v1",
        "cohort_tasks": len(tasks),
        "reusable_tasks": len(reusable_tasks),
        "reused_cells": len(rows),
        "missing_tasks": len(missing_tasks),
        "missing_task_ids": [str(task["task_id"]) for task in missing_tasks],
        "source_outcomes": source_inventory,
        "source_corpus_sha256": _sha256(source_corpus_path),
        "selection_used_outcomes": False,
        "provider_calls_made": 0,
        "outcomes_sha256": _sha256(outcomes_path),
        "missing_tasks_sha256": _sha256(missing_path),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "reuse audit complete tasks=%d reused_cells=%d missing_tasks=%d",
        len(reusable_tasks),
        len(rows),
        len(missing_tasks),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--source-corpus", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assemble(args.corpus, args.source_corpus, args.source, args.output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
