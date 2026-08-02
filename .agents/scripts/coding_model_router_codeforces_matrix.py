"""Run a resumable one-shot Codeforces reasoning-effort matrix remotely."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("coding-model-router-codeforces-matrix")

MODEL = "gpt-5.6-luna"
ARMS = {
    "luna-low": "low",
    "luna-medium": "medium",
    "luna-high": "high",
    "luna-xhigh": "xhigh",
    "luna-max": "max",
}
ATTEMPTS = 2
SEED = 20260731
MAX_OUTPUT_TOKENS = 32_768
MAX_RAW_RESPONSE_BYTES = 16 * 1024 * 1024
INPUT_PER_MTOK = 1.0
CACHED_INPUT_PER_MTOK = 0.1
OUTPUT_PER_MTOK = 6.0
CELL_RESERVATION_USD = 1.0
SYSTEM_INSTRUCTION = (
    "Solve the programming problem in Python 3. Return only the complete program. "
    "Read input from stdin and write the answer to stdout. Do not use Markdown."
)


@dataclass(frozen=True)
class Cell:
    """One task, reasoning-effort arm, and independent attempt."""

    task_id: str
    arm: str
    effort: str
    attempt: int

    @property
    def cell_id(self) -> str:
        """Return the stable identity used for resume and deduplication."""
        return f"{self.task_id}:{self.arm}:attempt-{self.attempt}"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return {str(key): item for key, item in value.items()}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} contains a non-object row")
            rows.append({str(key): item for key, item in value.items()})
    return rows


def _tasks(corpus: dict[str, Any]) -> list[dict[str, Any]]:
    value = corpus.get("tasks")
    if not isinstance(value, list):
        raise ValueError("frozen corpus has no task list")
    rows = [row for row in value if isinstance(row, dict)]
    if len(rows) != len(value):
        raise ValueError("frozen corpus contains a non-object task")
    return [{str(key): item for key, item in row.items()} for row in rows]


def _schedule(
    tasks: list[dict[str, Any]],
    *,
    efforts: tuple[str, ...] = tuple(ARMS),
    attempts: int = ATTEMPTS,
    seed: int = SEED,
) -> list[Cell]:
    unknown = set(efforts) - set(ARMS)
    if unknown or attempts < 1:
        raise ValueError(f"invalid schedule efforts={sorted(unknown)} attempts={attempts}")
    cells = [
        Cell(str(task["task_id"]), arm, ARMS[arm], attempt)
        for task in tasks
        for arm in efforts
        for attempt in range(attempts)
    ]
    return sorted(cells, key=lambda cell: _stable_key(seed, cell.cell_id))


def _response_text(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str):
        return output_text
    fragments: list[str] = []
    output = response.get("output")
    if not isinstance(output, list):
        return ""
    for item in output:
        if not isinstance(item, dict) or not isinstance(item.get("content"), list):
            continue
        for part in item["content"]:
            if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                text = part.get("text")
                if isinstance(text, str):
                    fragments.append(text)
    return "".join(fragments)


def _extract_code(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:python|py)?\s*\n(.*?)```", stripped, re.DOTALL | re.I)
    if fenced:
        stripped = fenced.group(1).strip()
    return stripped + ("\n" if stripped else "")


def _number(value: object) -> int:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0, int(value))
    return 0


def _usage(response: dict[str, Any]) -> dict[str, int]:
    raw = response.get("usage")
    if not isinstance(raw, dict):
        return {}
    input_details = raw.get("input_tokens_details")
    output_details = raw.get("output_tokens_details")
    return {
        "input_tokens": _number(raw.get("input_tokens")),
        "cached_input_tokens": _number(
            input_details.get("cached_tokens") if isinstance(input_details, dict) else 0
        ),
        "output_tokens": _number(raw.get("output_tokens")),
        "reasoning_tokens": _number(
            output_details.get("reasoning_tokens") if isinstance(output_details, dict) else 0
        ),
    }


def _cost(usage: dict[str, int]) -> float:
    cached = usage.get("cached_input_tokens", 0)
    uncached = max(0, usage.get("input_tokens", 0) - cached)
    return (
        uncached * INPUT_PER_MTOK
        + cached * CACHED_INPUT_PER_MTOK
        + usage.get("output_tokens", 0) * OUTPUT_PER_MTOK
    ) / 1_000_000


def _provider_call(
    prompt: str,
    effort: str,
    *,
    api_key: str,
    max_output_tokens: int,
    max_retries: int = 5,
) -> tuple[dict[str, Any], float]:
    payload = {
        "model": MODEL,
        "input": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
        "reasoning": {"effort": effort},
        "max_output_tokens": max_output_tokens,
    }
    body = json.dumps(payload).encode()
    started = time.monotonic()
    for retry in range(max_retries + 1):
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "wmo-codeforces-effort/1",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                raw = response.read(MAX_RAW_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RAW_RESPONSE_BYTES:
                raise RuntimeError("provider response exceeds the frozen size limit")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("Responses API returned a non-object payload")
            return {str(key): item for key, item in value.items()}, time.monotonic() - started
        except urllib.error.HTTPError as error:
            detail = error.read(16_384).decode("utf-8", "replace")
            if error.code != 429 and error.code < 500:
                raise RuntimeError(
                    f"provider rejected the frozen request with HTTP {error.code}: {detail}"
                ) from error
            if retry == max_retries:
                raise RuntimeError("provider transport retries exhausted") from error
        except (TimeoutError, urllib.error.URLError) as error:
            if retry == max_retries:
                raise RuntimeError("provider transport retries exhausted") from error
        time.sleep(min(30.0, 2.0**retry))
    raise AssertionError("provider retry loop exhausted unexpectedly")


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


def _grade(code: str, task: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    tests = task.get("tests")
    if not isinstance(tests, list) or not tests:
        raise ValueError(f"task {task.get('task_id')} has no tests")
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="codeforces-generated-") as directory:
        script = Path(directory) / "solution.py"
        script.write_text(code, encoding="utf-8")
        command = _sandbox_command(script, int(task["memory_limit_mb"]))
        for index, test in enumerate(tests):
            if not isinstance(test, dict):
                raise ValueError("frozen test is not an object")
            expected = str(test["output"])
            started = time.monotonic()
            try:
                completed = subprocess.run(
                    command,
                    input=str(test["input"]),
                    capture_output=True,
                    text=True,
                    timeout=float(task["time_limit_s"]),
                    env={},
                )
                passed = completed.returncode == 0 and completed.stdout.split() == expected.split()
                result = {
                    "index": index,
                    "passed": passed,
                    "returncode": completed.returncode,
                    "stdout_sha256": _sha256_bytes(completed.stdout.encode()),
                    "stderr_sha256": _sha256_bytes(completed.stderr.encode()),
                    "duration_s": time.monotonic() - started,
                }
            except subprocess.TimeoutExpired as error:
                stdout = (
                    error.stdout or b""
                    if isinstance(error.stdout, bytes)
                    else str(error.stdout or "").encode()
                )
                stderr = (
                    error.stderr or b""
                    if isinstance(error.stderr, bytes)
                    else str(error.stderr or "").encode()
                )
                result = {
                    "index": index,
                    "passed": False,
                    "timeout": True,
                    "stdout_sha256": _sha256_bytes(stdout),
                    "stderr_sha256": _sha256_bytes(stderr),
                    "duration_s": time.monotonic() - started,
                }
            results.append(result)
    return sum(bool(row["passed"]) for row in results) / len(results), results


class MatrixState:
    """Concurrency-safe append-only matrix state."""

    def __init__(
        self,
        output: Path,
        *,
        ceiling_usd: float,
        prior_spend_usd: float,
    ) -> None:
        self.path = output / "outcomes.jsonl"
        self.ceiling_usd = ceiling_usd
        self.prior_spend_usd = prior_spend_usd
        self.lock = threading.Lock()
        self.rows = {str(row["cell_id"]): row for row in _read_jsonl(self.path)}
        self.spent = sum(float(row.get("cost_usd") or 0.0) for row in self.rows.values())
        self.reserved = 0.0

    def completed(self, cell_id: str) -> bool:
        """Return whether a cell was durably persisted."""
        with self.lock:
            return cell_id in self.rows

    def reserve(self) -> None:
        """Reserve conservative spend under the user-authorized ceiling."""
        with self.lock:
            projected = self.prior_spend_usd + self.spent + self.reserved + CELL_RESERVATION_USD
            if projected > self.ceiling_usd:
                raise RuntimeError("next cell could exceed the authorized ceiling")
            self.reserved += CELL_RESERVATION_USD

    def persist(self, row: dict[str, Any]) -> None:
        """Append one unique cell and fsync it before returning."""
        with self.lock:
            self.reserved = max(0.0, self.reserved - CELL_RESERVATION_USD)
            cell_id = str(row["cell_id"])
            if cell_id in self.rows:
                raise ValueError(f"cell is already persisted: {cell_id}")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.rows[cell_id] = row
            self.spent += float(row.get("cost_usd") or 0.0)

    def release(self) -> None:
        """Release a reservation after a cell fails before persistence."""
        with self.lock:
            self.reserved = max(0.0, self.reserved - CELL_RESERVATION_USD)


def _run_cell(
    cell: Cell,
    task: dict[str, Any],
    output: Path,
    state: MatrixState,
    *,
    api_key: str,
    max_output_tokens: int,
) -> None:
    state.reserve()
    persisted = False
    try:
        response, provider_duration = _provider_call(
            str(task["prompt"]),
            cell.effort,
            api_key=api_key,
            max_output_tokens=max_output_tokens,
        )
        if response.get("model") != MODEL:
            raise RuntimeError(f"unexpected provider model {response.get('model')!r}")
        raw = json.dumps(response, sort_keys=True).encode()
        raw_name = _sha256_bytes(cell.cell_id.encode())
        raw_path = output / "raw" / f"{raw_name}.json"
        code_path = output / "code" / f"{raw_name}.py"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        code_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(raw + b"\n")
        text = _response_text(response)
        code = _extract_code(text)
        code_path.write_text(code, encoding="utf-8")
        reward, test_results = _grade(code, task)
        usage = _usage(response)
        state.persist(
            {
                "protocol": "codeforces-cots-effort-cell-v1",
                "cell_id": cell.cell_id,
                "task_id": cell.task_id,
                "contest_id": task["contest_id"],
                "bucket": task["bucket"],
                "arm": cell.arm,
                "model": MODEL,
                "reasoning_effort": cell.effort,
                "attempt": cell.attempt,
                "provider_response_id": response.get("id"),
                "provider_status": response.get("status"),
                "provider_incomplete_details": response.get("incomplete_details"),
                "observed_model": response.get("model"),
                "provider_duration_s": provider_duration,
                "usage": usage,
                "cost_usd": _cost(usage),
                "cost_accounting": "trace_token_estimate_v1",
                "reward": reward,
                "tests_passed": sum(bool(row["passed"]) for row in test_results),
                "tests_total": len(test_results),
                "test_results": test_results,
                "response_text_sha256": _sha256_bytes(text.encode()),
                "code_sha256": _sha256_file(code_path),
                "code_path": str(code_path),
                "raw_sha256": _sha256_file(raw_path),
                "raw_path": str(raw_path),
                "target_outcomes_used": False,
            }
        )
        persisted = True
        logger.info(
            "cell complete task=%s arm=%s attempt=%d reward=%.3f",
            cell.task_id,
            cell.arm,
            cell.attempt,
            reward,
        )
    finally:
        if not persisted:
            state.release()


def run(
    corpus_path: Path,
    output: Path,
    *,
    concurrency: int,
    ceiling_usd: float,
    prior_spend_usd: float,
    efforts: tuple[str, ...],
    attempts: int,
    task_ids: set[str] | None,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
) -> dict[str, Any]:
    """Generate, grade, persist, and resume a frozen matrix."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is unavailable")
    if max_output_tokens < 1:
        raise ValueError("max_output_tokens must be positive")
    corpus = _read_object(corpus_path)
    if (
        corpus.get("target_outcomes_used") is not False
        or corpus.get("published_generations_loaded") is not False
    ):
        raise ValueError("corpus violated a frozen information boundary")
    tasks = _tasks(corpus)
    if task_ids is not None:
        tasks = [task for task in tasks if str(task["task_id"]) in task_ids]
        if {str(task["task_id"]) for task in tasks} != task_ids:
            raise ValueError("one or more requested task IDs are absent")
    scheduled = _schedule(tasks, efforts=efforts, attempts=attempts)
    state = MatrixState(
        output,
        ceiling_usd=ceiling_usd,
        prior_spend_usd=prior_spend_usd,
    )
    pending = [cell for cell in scheduled if not state.completed(cell.cell_id)]
    task_by_id = {str(task["task_id"]): task for task in tasks}
    logger.info(
        "matrix start tasks=%d cells=%d pending=%d",
        len(tasks),
        len(scheduled),
        len(pending),
    )
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                _run_cell,
                cell,
                task_by_id[cell.task_id],
                output,
                state,
                api_key=api_key,
                max_output_tokens=max_output_tokens,
            ): cell
            for cell in pending
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                for other in futures:
                    other.cancel()
                raise
    selected_ids = {cell.cell_id for cell in scheduled}
    selected_rows = [row for cell_id, row in state.rows.items() if cell_id in selected_ids]
    if len(selected_rows) != len(scheduled):
        raise RuntimeError(f"matrix has {len(selected_rows)} of {len(scheduled)} selected cells")
    report = {
        "protocol": "codeforces-cots-effort-matrix-v1",
        "corpus_sha256": _sha256_file(corpus_path),
        "model": MODEL,
        "efforts": list(efforts),
        "attempts": attempts,
        "max_output_tokens": max_output_tokens,
        "tasks": len(tasks),
        "cells": len(selected_rows),
        "pending_before_run": len(pending),
        "estimated_spend_usd": sum(float(row["cost_usd"]) for row in selected_rows),
        "outcomes_sha256": _sha256_file(state.path),
        "target_outcomes_used": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "matrix-manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    """Parse the remote worker command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--ceiling-usd", type=float, default=20_000.0)
    parser.add_argument("--prior-spend-usd", type=float, default=0.0)
    parser.add_argument(
        "--effort",
        action="append",
        choices=tuple(ARMS),
        dest="efforts",
    )
    parser.add_argument("--attempts", type=int, default=ATTEMPTS)
    parser.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    parser.add_argument("--task-id", action="append", dest="task_ids")
    args = parser.parse_args()
    run(
        args.corpus,
        args.output,
        concurrency=args.concurrency,
        ceiling_usd=args.ceiling_usd,
        prior_spend_usd=args.prior_spend_usd,
        efforts=tuple(args.efforts or ARMS),
        attempts=args.attempts,
        task_ids=set(args.task_ids) if args.task_ids else None,
        max_output_tokens=args.max_output_tokens,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
