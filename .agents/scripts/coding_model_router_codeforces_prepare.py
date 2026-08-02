"""Freeze a hard, executable Codeforces cohort from a pinned HF trace corpus.

The published generation is never loaded. Selection uses only problem metadata,
tests, and deterministic contest-index strata. A published accepted Python
solution must pass the exact frozen tests before a task enters the corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

logger = logging.getLogger("coding-model-router-codeforces-prepare")

SOURCE_DATASET = "open-r1/codeforces-cots"
SOURCE_CONFIG = "solutions_py_decontaminated"
SOURCE_REVISION = "39ac85c150806230473c70ad72c31f6232fe3f41"
BUCKETS = ("C", "D", "E", "F+")
TEST_LIMIT = 20


@dataclass(frozen=True)
class Candidate:
    task_id: str
    contest_id: str
    index: str
    bucket: str
    title: str
    prompt: str
    time_limit_s: float
    memory_limit_mb: int
    tests: tuple[tuple[str, str], ...]
    reference_code: str


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bucket(index: str) -> str | None:
    match = re.match(r"([A-Z])", index.upper())
    if not match:
        return None
    letter = match.group(1)
    if letter in {"C", "D", "E"}:
        return letter
    if letter >= "F":
        return "F+"
    return None


def _test_pairs(row: dict[str, Any], *, seed: int) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for field in ("public_tests", "private_tests", "generated_tests"):
        value = row.get(field)
        if not isinstance(value, dict):
            continue
        inputs = value.get("input")
        outputs = value.get("output")
        if not isinstance(inputs, list) or not isinstance(outputs, list):
            continue
        for test_input, expected in zip(inputs, outputs, strict=True):
            if isinstance(test_input, str) and isinstance(expected, str):
                pairs.append((test_input, expected))
    unique = list(dict.fromkeys(pairs))
    unique.sort(key=lambda pair: _stable_key(seed, f"{pair[0]}\0{pair[1]}"))
    return tuple(unique[:TEST_LIMIT])


def _reference_code(row: dict[str, Any]) -> str | None:
    solutions = row.get("accepted_solutions")
    if not isinstance(solutions, list):
        return None
    candidates = []
    for solution in solutions:
        if not isinstance(solution, dict):
            continue
        language = str(
            solution.get("programmingLanguage") or solution.get("programming_language") or ""
        ).casefold()
        code = solution.get("code")
        if isinstance(code, str) and code.strip() and ("python" in language or "pypy" in language):
            candidates.append(code)
    return min(candidates, key=len) if candidates else None


def _candidate(row: dict[str, Any], *, seed: int) -> Candidate | None:
    task_id = row.get("id")
    contest_id = row.get("contest_id")
    index = row.get("index")
    prompt = row.get("prompt")
    title = row.get("title")
    problem_type = row.get("problem_type")
    if not all(isinstance(value, str) and value for value in (task_id, contest_id, index, prompt)):
        return None
    if problem_type != "diff" or str(row.get("interaction_format") or "").strip():
        return None
    bucket = _bucket(str(index))
    tests = _test_pairs(row, seed=seed)
    reference = _reference_code(row)
    if bucket is None or len(tests) < 8 or reference is None or len(str(prompt)) > 30_000:
        return None
    return Candidate(
        task_id=str(task_id),
        contest_id=str(contest_id),
        index=str(index),
        bucket=bucket,
        title=str(title or task_id),
        prompt=str(prompt),
        time_limit_s=min(8.0, max(1.0, float(row.get("time_limit") or 2.0) * 2.0)),
        memory_limit_mb=min(1024, max(128, int(row.get("memory_limit") or 256))),
        tests=tests,
        reference_code=reference,
    )


def _sandbox_command(script: Path, memory_mb: int) -> list[str]:
    return [
        "prlimit",
        f"--as={memory_mb * 1024 * 1024}:{memory_mb * 1024 * 1024}",
        "--fsize=16777216:16777216",
        "--nproc=64:64",
        "--",
        "bwrap",
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        "/bin",
        "/bin",
        "--ro-bind",
        "/lib",
        "/lib",
        "--ro-bind",
        "/lib64",
        "/lib64",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--ro-bind",
        str(script.parent),
        "/work",
        "--chdir",
        "/work",
        "--setenv",
        "HOME",
        "/tmp",
        "--setenv",
        "PATH",
        "/usr/bin:/bin",
        "/usr/bin/python3",
        "-I",
        f"/work/{script.name}",
    ]


def _output_matches(observed: str, expected: str) -> bool:
    return observed.split() == expected.split()


def _validate(candidate: Candidate) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="codeforces-reference-") as directory:
        script = Path(directory) / "solution.py"
        script.write_text(candidate.reference_code, encoding="utf-8")
        command = _sandbox_command(script, candidate.memory_limit_mb)
        for test_index, (test_input, expected) in enumerate(candidate.tests):
            try:
                result = subprocess.run(
                    command,
                    input=test_input,
                    capture_output=True,
                    text=True,
                    timeout=candidate.time_limit_s,
                    env={},
                )
            except subprocess.TimeoutExpired:
                return False, f"reference timeout test={test_index}"
            if result.returncode != 0:
                return False, f"reference exit={result.returncode} test={test_index}"
            if not _output_matches(result.stdout, expected):
                return False, f"reference mismatch test={test_index}"
    return True, ""


def _target_prompts(path: Path | None) -> set[str]:
    if path is None:
        return set()
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("rows") if isinstance(value, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"{path} has no rows")
    prompts = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = row.get("text", row.get("problem_statement", row.get("prompt")))
        if isinstance(text, str) and text.strip():
            prompts.add(_normalize(text))
    return prompts


def _excluded_task_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("tasks") if isinstance(value, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"{path} has no tasks")
    task_ids = {
        str(row["task_id"])
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("task_id"), str)
    }
    if len(task_ids) != len(rows):
        raise ValueError(f"{path} contains invalid or duplicate task IDs")
    return task_ids


def prepare(
    shards: list[Path],
    output: Path,
    *,
    target_tasks: Path | None,
    exclude_tasks: Path | None,
    tasks_per_bucket: int,
    seed: int,
    workers: int,
) -> None:
    target_prompts = _target_prompts(target_tasks)
    excluded_task_ids = _excluded_task_ids(exclude_tasks)
    columns = [
        "id",
        "contest_id",
        "index",
        "title",
        "time_limit",
        "memory_limit",
        "prompt",
        "interaction_format",
        "problem_type",
        "public_tests",
        "private_tests",
        "generated_tests",
        "accepted_solutions",
    ]
    by_id: dict[str, Candidate] = {}
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=128, columns=columns):
            for raw in batch.to_pylist():
                row = {str(key): item for key, item in raw.items()}
                candidate = _candidate(row, seed=seed)
                if (
                    candidate is None
                    or candidate.task_id in excluded_task_ids
                    or _normalize(candidate.prompt) in target_prompts
                ):
                    continue
                by_id.setdefault(candidate.task_id, candidate)
    buckets: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in by_id.values():
        buckets[candidate.bucket].append(candidate)
    candidate_pool: list[Candidate] = []
    for bucket in BUCKETS:
        ordered = sorted(buckets[bucket], key=lambda row: _stable_key(seed, row.task_id))
        candidate_pool.extend(ordered[: tasks_per_bucket * 2])
    validation: dict[str, tuple[bool, str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_validate, row): row for row in candidate_pool}
        for completed, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            validation[row.task_id] = future.result()
            logger.info(
                "validated %d/%d task=%s valid=%s",
                completed,
                len(candidate_pool),
                row.task_id,
                validation[row.task_id][0],
            )
    selected: list[Candidate] = []
    for bucket in BUCKETS:
        valid = [
            row for row in candidate_pool if row.bucket == bucket and validation[row.task_id][0]
        ][:tasks_per_bucket]
        if len(valid) != tasks_per_bucket:
            raise ValueError(
                f"bucket {bucket} has {len(valid)} valid tasks, expected {tasks_per_bucket}"
            )
        selected.extend(valid)
    output.mkdir(parents=True, exist_ok=False)
    corpus_path = output / "tasks.json"
    corpus = {
        "protocol": "codeforces-cots-effort-corpus-v1",
        "source_dataset": SOURCE_DATASET,
        "source_config": SOURCE_CONFIG,
        "source_revision": SOURCE_REVISION,
        "seed": seed,
        "published_generations_loaded": False,
        "target_outcomes_used": False,
        "tasks": [
            {
                "task_id": row.task_id,
                "contest_id": row.contest_id,
                "index": row.index,
                "bucket": row.bucket,
                "title": row.title,
                "prompt": row.prompt,
                "time_limit_s": row.time_limit_s,
                "memory_limit_mb": row.memory_limit_mb,
                "tests": [
                    {"input": test_input, "output": expected} for test_input, expected in row.tests
                ],
            }
            for row in selected
        ],
    }
    corpus_path.write_text(json.dumps(corpus, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validation_path = output / "validation.jsonl"
    validation_path.write_text(
        "".join(
            json.dumps(
                {
                    "task_id": task_id,
                    "passed": passed,
                    "detail": detail,
                },
                sort_keys=True,
            )
            + "\n"
            for task_id, (passed, detail) in sorted(validation.items())
        ),
        encoding="utf-8",
    )
    manifest = {
        "protocol": "codeforces-cots-effort-corpus-manifest-v1",
        "source_dataset": SOURCE_DATASET,
        "source_config": SOURCE_CONFIG,
        "source_revision": SOURCE_REVISION,
        "source_shards": [
            {"name": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in shards
        ],
        "deduplicated_eligible_tasks": len(by_id),
        "candidate_tasks": len(candidate_pool),
        "selected_tasks": len(selected),
        "bucket_mix": dict(Counter(row.bucket for row in selected)),
        "contest_groups": len({row.contest_id for row in selected}),
        "target_normalized_prompt_overlap": 0,
        "excluded_task_ids": len(excluded_task_ids),
        "excluded_task_overlap": len({row.task_id for row in selected} & excluded_task_ids),
        "target_outcomes_used": False,
        "published_generations_loaded": False,
        "tasks_sha256": _sha256(corpus_path),
        "validation_sha256": _sha256(validation_path),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-tasks", type=Path)
    parser.add_argument("--exclude-tasks", type=Path)
    parser.add_argument("--tasks-per-bucket", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    prepare(
        args.shard,
        args.output,
        target_tasks=args.target_tasks,
        exclude_tasks=args.exclude_tasks,
        tasks_per_bucket=args.tasks_per_bucket,
        seed=args.seed,
        workers=args.workers,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
