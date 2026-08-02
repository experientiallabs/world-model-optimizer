"""Prepare, shard, and merge official BigCodeBench execution scoring."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger("coding-model-router-bigcodebench-score")

ARMS = ("luna-low", "luna-medium", "luna-high", "luna-xhigh", "luna-max")
ATTEMPTS = 5
CELLS_PER_TASK = len(ARMS) * ATTEMPTS
OFFICIAL_VERSION = "0.2.4"
PASS_STATUS = "pass"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return {str(key): item for key, item in value.items()}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} contains a non-object row")
        rows.append({str(key): item for key, item in value.items()})
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_object(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _task_id(row: dict[str, Any]) -> str:
    value = row.get("task_id")
    if not isinstance(value, str) or not value:
        raise ValueError("row has no task_id")
    return value


def _response_text(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str):
        return output_text
    fragments: list[str] = []
    output = response.get("output")
    if not isinstance(output, list):
        return ""
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and part.get("type") in {"output_text", "text"}:
                fragments.append(text)
    return "".join(fragments)


def _ordered_cells(tasks: list[dict[str, Any]]) -> list[tuple[str, str, int]]:
    return [
        (_task_id(task), arm, attempt)
        for task in tasks
        for arm in ARMS
        for attempt in range(ATTEMPTS)
    ]


def _sanitize_one(payload: tuple[str, str, str]) -> dict[str, str]:
    task_id, raw_solution, entry_point = payload
    module = cast(Any, importlib.import_module("bigcodebench.sanitize"))
    sanitizer = cast(Callable[..., str], module.sanitize)
    return {"task_id": task_id, "solution": sanitizer(raw_solution, entry_point)}


def export_raw(output: Path) -> None:
    matrix = _read_object(output / "matrix-manifest.json")
    tasks = _read_jsonl(output / "tasks.jsonl")
    outcomes = {
        str(row["cell_id"]): row for row in _read_jsonl(output / "outcomes.jsonl")
    }
    ordered = _ordered_cells(tasks)
    if (
        matrix.get("target_outcomes_used") is not False
        or matrix.get("cells") != len(ordered)
        or len(outcomes) != len(ordered)
    ):
        raise ValueError("generation matrix is incomplete or crossed the target boundary")
    raw_samples: list[dict[str, Any]] = []
    index: list[dict[str, Any]] = []
    for line_number, (task_id, arm, attempt) in enumerate(ordered):
        cell_id = f"{task_id}:{arm}:attempt-{attempt}"
        row = outcomes.get(cell_id)
        if row is None:
            raise ValueError(f"missing generation cell: {cell_id}")
        raw_path = Path(str(row.get("raw_path")))
        raw_bytes = raw_path.read_bytes()
        if _sha256_bytes(raw_bytes.rstrip(b"\n")) != row.get("raw_sha256"):
            raise ValueError(f"raw response hash mismatch: {cell_id}")
        response = _read_object(raw_path)
        solution = _response_text(response)
        raw_samples.append({"task_id": task_id, "solution": solution})
        index.append(
            {
                "line_number": line_number,
                "cell_id": cell_id,
                "task_id": task_id,
                "arm": arm,
                "attempt": attempt,
                "raw_solution_sha256": _sha256_bytes(solution.encode()),
            }
        )
    raw_samples_path = output / "raw-samples.jsonl"
    index_path = output / "sample-index.jsonl"
    _write_jsonl(raw_samples_path, raw_samples)
    _write_jsonl(index_path, index)
    _write_object(
        output / "raw-samples-manifest.json",
        {
            "protocol": "bigcodebench-effort-raw-samples-v1",
            "cells": len(raw_samples),
            "matrix_manifest_sha256": _sha256_file(output / "matrix-manifest.json"),
            "outcomes_sha256": _sha256_file(output / "outcomes.jsonl"),
            "raw_samples_sha256": _sha256_file(raw_samples_path),
            "sample_index_sha256": _sha256_file(index_path),
            "target_outcomes_used": False,
        },
    )
    logger.info("raw samples exported cells=%d", len(raw_samples))


def sanitize_samples(
    raw_samples: Path,
    samples: Path,
    manifest: Path,
    *,
    parallel: int = 8,
) -> None:
    if parallel < 1:
        raise ValueError("parallel must be positive")
    package = importlib.import_module("bigcodebench")
    observed_version = str(getattr(package, "__version__", ""))
    if observed_version != OFFICIAL_VERSION:
        raise ValueError(
            f"official evaluator version is {observed_version!r}, expected {OFFICIAL_VERSION}"
        )
    data_module = cast(Any, importlib.import_module("bigcodebench.data"))
    get_bigcodebench = cast(
        Callable[..., dict[str, dict[str, Any]]],
        data_module.get_bigcodebench,
    )
    problems = get_bigcodebench(subset="full")
    raw_rows = _read_jsonl(raw_samples)
    payloads: list[tuple[str, str, str]] = []
    for row in raw_rows:
        task_id = _task_id(row)
        problem = problems.get(task_id)
        if not isinstance(problem, dict):
            raise ValueError(f"official dataset has no task: {task_id}")
        entry_point = problem.get("entry_point")
        if not isinstance(entry_point, str):
            raise ValueError(f"official dataset has no entry point: {task_id}")
        raw_solution = row.get("solution")
        if not isinstance(raw_solution, str):
            raise ValueError(f"raw sample has no solution: {task_id}")
        payloads.append((task_id, raw_solution, entry_point))
    with ProcessPoolExecutor(max_workers=parallel) as executor:
        sanitized_rows = list(executor.map(_sanitize_one, payloads, chunksize=16))
    _write_jsonl(samples, sanitized_rows)
    _write_object(
        manifest,
        {
            "protocol": "bigcodebench-effort-samples-v1",
            "official_version": observed_version,
            "parallel": parallel,
            "samples": len(sanitized_rows),
            "raw_samples_sha256": _sha256_file(raw_samples),
            "samples_sha256": _sha256_file(samples),
            "target_outcomes_used": False,
        },
    )
    logger.info(
        "official sanitization complete samples=%d version=%s",
        len(sanitized_rows),
        observed_version,
    )


def make_chunks(
    output: Path,
    *,
    tasks_per_chunk: int,
    cells_per_task: int = CELLS_PER_TASK,
) -> None:
    if tasks_per_chunk < 1:
        raise ValueError("tasks_per_chunk must be positive")
    tasks = _read_jsonl(output / "tasks.jsonl")
    samples = _read_jsonl(output / "samples.jsonl")
    index = _read_jsonl(output / "sample-index.jsonl")
    if len(samples) != len(index) or len(samples) != len(tasks) * cells_per_task:
        raise ValueError("sanitized sample matrix is incomplete")
    sample_by_task: dict[str, list[dict[str, Any]]] = {}
    index_by_task: dict[str, list[dict[str, Any]]] = {}
    for sample, index_row in zip(samples, index, strict=True):
        task_id = _task_id(sample)
        if task_id != _task_id(index_row):
            raise ValueError("sample and index task order differ")
        sample_by_task.setdefault(task_id, []).append(sample)
        index_by_task.setdefault(task_id, []).append(index_row)
    chunk_root = output / "score-chunks"
    if chunk_root.exists():
        raise FileExistsError(f"score chunk root already exists: {chunk_root}")
    chunk_count = 0
    for start in range(0, len(tasks), tasks_per_chunk):
        task_ids = [_task_id(task) for task in tasks[start : start + tasks_per_chunk]]
        chunk_samples = [sample for task_id in task_ids for sample in sample_by_task[task_id]]
        chunk_index = [row for task_id in task_ids for row in index_by_task[task_id]]
        if any(len(sample_by_task[task_id]) != cells_per_task for task_id in task_ids):
            raise ValueError("a score chunk task is missing samples")
        chunk = chunk_root / f"chunk-{chunk_count:02d}"
        samples_path = chunk / "samples.jsonl"
        index_path = chunk / "sample-index.jsonl"
        _write_jsonl(samples_path, chunk_samples)
        _write_jsonl(index_path, chunk_index)
        _write_object(
            chunk / "chunk.json",
            {
                "protocol": "bigcodebench-effort-score-chunk-v1",
                "chunk": chunk_count,
                "official_version": OFFICIAL_VERSION,
                "task_ids": task_ids,
                "tasks": len(task_ids),
                "samples": len(chunk_samples),
                "samples_sha256": _sha256_file(samples_path),
                "sample_index_sha256": _sha256_file(index_path),
                "target_outcomes_used": False,
            },
        )
        chunk_count += 1
    logger.info("official score chunks built chunks=%d tasks=%d", chunk_count, len(tasks))


def _validate_official_result(
    result: dict[str, Any],
    task_ids: list[str],
    *,
    cells_per_task: int = CELLS_PER_TASK,
) -> None:
    raw_eval = result.get("eval")
    if not isinstance(raw_eval, dict) or set(raw_eval) != set(task_ids):
        raise ValueError("official result task set differs from its frozen chunk")
    for task_id in task_ids:
        rows = raw_eval.get(task_id)
        if not isinstance(rows, list) or len(rows) != cells_per_task:
            count = len(rows) if isinstance(rows, list) else 0
            raise ValueError(f"official task has {count} results: {task_id}")
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"official task has a non-object result: {task_id}")


def official_score(
    samples: Path,
    chunk_manifest: Path,
    *,
    parallel: int = 8,
) -> None:
    if parallel < 1:
        raise ValueError("parallel must be positive")
    package = importlib.import_module("bigcodebench")
    observed_version = str(getattr(package, "__version__", ""))
    if observed_version != OFFICIAL_VERSION:
        raise ValueError(
            f"official evaluator version is {observed_version!r}, expected {OFFICIAL_VERSION}"
        )
    chunk = _read_object(chunk_manifest)
    task_ids_untyped = chunk.get("task_ids")
    if not isinstance(task_ids_untyped, list):
        raise ValueError("score chunk has no task id list")
    task_ids = [str(task_id) for task_id in task_ids_untyped]
    if (
        chunk.get("official_version") != OFFICIAL_VERSION
        or chunk.get("samples_sha256") != _sha256_file(samples)
    ):
        raise ValueError("score chunk provenance is invalid")
    result_path = samples.with_name("samples_eval_results.json")
    if result_path.exists():
        raise FileExistsError(f"official result already exists: {result_path}")
    module = cast(Any, importlib.import_module("bigcodebench.evaluate"))
    evaluator = cast(Callable[..., Any], module.evaluate)
    evaluator(
        split="instruct",
        subset="full",
        samples=str(samples),
        execution="local",
        selective_evaluate=",".join(task_ids),
        pass_k="1",
        save_pass_rate=False,
        calibrated=True,
        parallel=parallel,
    )
    result = _read_object(result_path)
    _validate_official_result(result, task_ids)
    _write_object(
        samples.with_name("official-score-manifest.json"),
        {
            "protocol": "bigcodebench-effort-official-score-v1",
            "chunk": chunk.get("chunk"),
            "official_version": observed_version,
            "parallel": parallel,
            "tasks": len(task_ids),
            "samples": len(task_ids) * CELLS_PER_TASK,
            "samples_sha256": _sha256_file(samples),
            "official_result_sha256": _sha256_file(result_path),
            "target_outcomes_used": False,
        },
    )
    logger.info(
        "official execution scoring complete tasks=%d samples=%d version=%s",
        len(task_ids),
        len(task_ids) * CELLS_PER_TASK,
        observed_version,
    )


def merge_chunks(
    output: Path,
    *,
    cells_per_task: int = CELLS_PER_TASK,
) -> None:
    tasks = _read_jsonl(output / "tasks.jsonl")
    global_index = _read_jsonl(output / "sample-index.jsonl")
    chunk_paths = sorted((output / "score-chunks").glob("chunk-*/chunk.json"))
    if not chunk_paths:
        raise ValueError("no official score chunks exist")
    eval_by_task: dict[str, list[dict[str, Any]]] = {}
    chunk_provenance: list[dict[str, Any]] = []
    for chunk_path in chunk_paths:
        chunk = _read_object(chunk_path)
        result_path = chunk_path.with_name("samples_eval_results.json")
        result = _read_object(result_path)
        raw_eval = result.get("eval")
        if not isinstance(raw_eval, dict):
            raise ValueError(f"official result has no eval map: {result_path}")
        task_ids = chunk.get("task_ids")
        if not isinstance(task_ids, list) or set(map(str, task_ids)) != set(raw_eval):
            raise ValueError(f"official result task set differs: {result_path}")
        for task_id_untyped, rows_untyped in raw_eval.items():
            task_id = str(task_id_untyped)
            if task_id in eval_by_task or not isinstance(rows_untyped, list):
                raise ValueError(f"duplicate or invalid official task: {task_id}")
            rows = [
                {str(key): item for key, item in row.items()}
                for row in rows_untyped
                if isinstance(row, dict)
            ]
            if len(rows) != cells_per_task:
                raise ValueError(f"official task has {len(rows)} results: {task_id}")
            eval_by_task[task_id] = rows
        chunk_provenance.append(
            {
                "chunk": chunk.get("chunk"),
                "official_result_sha256": _sha256_file(result_path),
                "samples_sha256": chunk.get("samples_sha256"),
                "tasks": chunk.get("tasks"),
            }
        )
    task_ids = [_task_id(task) for task in tasks]
    if set(eval_by_task) != set(task_ids):
        raise ValueError("merged official result does not cover the frozen cohort")
    merged_result = {
        "protocol": "bigcodebench-effort-official-merged-v1",
        "official_version": OFFICIAL_VERSION,
        "eval": {task_id: eval_by_task[task_id] for task_id in task_ids},
        "chunk_provenance": chunk_provenance,
        "target_outcomes_used": False,
    }
    merged_path = output / "samples_eval_results.json"
    _write_object(merged_path, merged_result)
    index_by_task: dict[str, list[dict[str, Any]]] = {}
    for row in global_index:
        index_by_task.setdefault(_task_id(row), []).append(row)
    scores: list[dict[str, Any]] = []
    for task_id in task_ids:
        index_rows = sorted(
            index_by_task[task_id], key=lambda row: int(row["line_number"])
        )
        result_rows = eval_by_task[task_id]
        if len(index_rows) != cells_per_task:
            raise ValueError(f"global sample index is incomplete: {task_id}")
        for index_row, result in zip(index_rows, result_rows, strict=True):
            details = result.get("details")
            scores.append(
                {
                    "protocol": "bigcodebench-effort-score-v1",
                    "cell_id": index_row["cell_id"],
                    "task_id": task_id,
                    "arm": index_row["arm"],
                    "attempt": index_row["attempt"],
                    "reward": float(result.get("status") == PASS_STATUS),
                    "status": result.get("status"),
                    "details_sha256": _sha256_bytes(
                        json.dumps(details, sort_keys=True).encode()
                    ),
                    "target_outcomes_used": False,
                }
            )
    scores_path = output / "scores.jsonl"
    _write_jsonl(scores_path, scores)
    _write_object(
        output / "score-manifest.json",
        {
            "protocol": "bigcodebench-effort-score-matrix-v1",
            "official_version": OFFICIAL_VERSION,
            "tasks": len(tasks),
            "arms": len(ARMS),
            "attempts": ATTEMPTS,
            "cells": len(scores),
            "chunks": len(chunk_paths),
            "scores_sha256": _sha256_file(scores_path),
            "official_result_sha256": _sha256_file(merged_path),
            "target_outcomes_used": False,
        },
    )
    logger.info("official score chunks merged cells=%d", len(scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export-raw")
    export_parser.add_argument("--output", type=Path, required=True)
    sanitize_parser = subparsers.add_parser("sanitize")
    sanitize_parser.add_argument("--raw-samples", type=Path, required=True)
    sanitize_parser.add_argument("--samples", type=Path, required=True)
    sanitize_parser.add_argument("--manifest", type=Path, required=True)
    sanitize_parser.add_argument("--parallel", type=int, default=8)
    chunks_parser = subparsers.add_parser("make-chunks")
    chunks_parser.add_argument("--output", type=Path, required=True)
    chunks_parser.add_argument("--tasks-per-chunk", type=int, default=50)
    score_parser = subparsers.add_parser("official-score")
    score_parser.add_argument("--samples", type=Path, required=True)
    score_parser.add_argument("--chunk-manifest", type=Path, required=True)
    score_parser.add_argument("--parallel", type=int, default=8)
    merge_parser = subparsers.add_parser("merge-chunks")
    merge_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export-raw":
        export_raw(args.output)
    elif args.command == "sanitize":
        sanitize_samples(
            args.raw_samples,
            args.samples,
            args.manifest,
            parallel=args.parallel,
        )
    elif args.command == "make-chunks":
        make_chunks(args.output, tasks_per_chunk=args.tasks_per_chunk)
    elif args.command == "official-score":
        official_score(
            args.samples,
            args.chunk_manifest,
            parallel=args.parallel,
        )
    elif args.command == "merge-chunks":
        merge_chunks(args.output)
    else:
        raise AssertionError(f"unexpected command: {args.command}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
