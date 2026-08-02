"""Run a paired reasoning-effort matrix on frozen Moonshiner coding tasks.

This worker is designed for remote experiment compute. It uses Moonshiner's
workspace materialization, Pi coding-agent runtime, deterministic verifier, and
protected-file checks. It persists traces and compact outcome rows, never model
weights.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import shutil
import stat
import subprocess
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger("coding-model-router-moonshiner-matrix")

MODEL = "gpt-5.6-luna"
EFFORT_SETS = {
    "smoke": {
        "luna-high": "high",
        "luna-max": "max",
    },
    "target": {
        "luna-xhigh": "xhigh",
        "luna-max": "max",
    },
    "lower": {
        "luna-low": "low",
        "luna-medium": "medium",
        "luna-high": "high",
    },
    "full": {
        "luna-low": "low",
        "luna-medium": "medium",
        "luna-high": "high",
        "luna-xhigh": "xhigh",
        "luna-max": "max",
    },
}
INPUT_PER_MTOK = 1.0
CACHED_INPUT_PER_MTOK = 0.1
OUTPUT_PER_MTOK = 6.0
DEFAULT_RESERVATION_USD = 100.0
ATTESTATION_ADAPTER = "openai-responses-nested-model-v1"
USAGE_ADAPTER = "pi-message-end-usage-v1"
MAX_TRACE_BYTES = 64 * 1_024 * 1_024


class _WorkspaceManager(Protocol):
    def remove_workspace(self, workspace: Path) -> None: ...


@dataclass(frozen=True)
class Cell:
    task_id: str
    arm: str
    effort: str
    attempt: int = 0

    @property
    def cell_id(self) -> str:
        return f"{self.task_id}:{self.arm}:attempt-{self.attempt}"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return {str(key): item for key, item in value.items()}


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _schedule(
    task_ids: list[str],
    *,
    seed: int,
    attempt: int = 0,
    arm_efforts: dict[str, str] | None = None,
) -> list[Cell]:
    selected_arms = arm_efforts or EFFORT_SETS["target"]
    ordered_tasks = sorted(task_ids, key=lambda task_id: _stable_key(seed, task_id))
    cells: list[Cell] = []
    for task_id in ordered_tasks:
        arms = sorted(
            selected_arms,
            key=lambda arm: _stable_key(seed, f"{task_id}:{arm}"),
        )
        cells.extend(Cell(task_id, arm, selected_arms[arm], attempt) for arm in arms)
    return cells


def _usage_number(value: object, keys: tuple[str, ...]) -> int:
    if isinstance(value, dict):
        for key in keys:
            candidate = value.get(key)
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                return max(0, int(candidate))
        for candidate in value.values():
            found = _usage_number(candidate, keys)
            if found:
                return found
    return 0


def _reported_cost(value: object) -> float | None:
    if isinstance(value, dict):
        for key in ("cost_usd", "total_cost_usd", "cost"):
            candidate = value.get(key)
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                return max(0.0, float(candidate))
        for candidate in value.values():
            found = _reported_cost(candidate)
            if found is not None:
                return found
    return None


def _extract_upstream_model(body: bytes) -> str | None:
    """Extract the model from JSON or OpenAI Responses SSE payloads."""

    def payload_model(value: object) -> str | None:
        if not isinstance(value, dict):
            return None
        model = value.get("model")
        if isinstance(model, str) and model:
            return model
        response = value.get("response")
        if isinstance(response, dict):
            nested = response.get("model")
            if isinstance(nested, str) and nested:
                return nested
        return None

    text = body.decode("utf-8", "replace")
    try:
        model = payload_model(json.loads(text))
    except json.JSONDecodeError:
        model = None
    if model:
        return model
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("data:"):
            continue
        payload = stripped[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            model = payload_model(json.loads(payload))
        except json.JSONDecodeError:
            continue
        if model:
            return model
    return None


def _pi_usage_from_stream(text: str) -> dict[str, int]:
    """Aggregate final assistant-message usage across Pi tool turns."""
    input_tokens = 0
    cached_input_tokens = 0
    output_tokens = 0
    reasoning_tokens = 0
    turns = 0
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "message_end":
            continue
        message = event.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        raw_usage = message.get("usage")
        if not isinstance(raw_usage, dict):
            continue
        usage = {str(key): value for key, value in raw_usage.items()}
        direct_input = _nonnegative_int(usage.get("input"))
        cache_read = _nonnegative_int(usage.get("cacheRead"))
        cache_write = _nonnegative_int(usage.get("cacheWrite"))
        input_tokens += direct_input + cache_read + cache_write
        cached_input_tokens += cache_read
        output_tokens += _nonnegative_int(usage.get("output"))
        reasoning_tokens += _nonnegative_int(usage.get("reasoning"))
        turns += 1
    if turns == 0:
        return {}
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "turns": turns,
    }


def _nonnegative_int(value: object) -> int:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0, int(value))
    return 0


def _bounded_pi_command(command: list[str]) -> list[str]:
    limit = f"--fsize={MAX_TRACE_BYTES}:{MAX_TRACE_BYTES}"
    return ["prlimit", limit, "--", *command]


def _patch_moonshiner_runtime() -> None:
    """Adapt pinned Moonshiner to Responses SSE and Pi 0.80.7 usage events."""
    credential_proxy = importlib.import_module("runtimes.credential_proxy")
    pi_module = importlib.import_module("runtimes.pi")
    if getattr(pi_module, "_wmo_responses_adapter", False):
        return
    original_parser = pi_module.__dict__["_parse_stream_meta"]
    original_runner = pi_module.__dict__["run_streamed"]

    def parse_stream_meta(text: str) -> dict[str, Any]:
        meta = original_parser(text)
        usage = _pi_usage_from_stream(text)
        if usage:
            meta["usage"] = usage
        return meta

    def bounded_run_streamed(
        command: list[str],
        *,
        workspace: Path,
        turn: str,
        stdout_path: Path,
        stderr_path: Path,
        timeout: int,
        environment: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        return original_runner(
            _bounded_pi_command(command),
            workspace=workspace,
            turn=turn,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout=timeout,
            environment=environment,
        )

    credential_proxy.__dict__["_extract_model"] = _extract_upstream_model
    pi_module.__dict__["_parse_stream_meta"] = parse_stream_meta
    pi_module.__dict__["run_streamed"] = bounded_run_streamed
    pi_module.__dict__["_wmo_responses_adapter"] = True


def _cost(usage: dict[str, Any]) -> tuple[float, str]:
    reported = _reported_cost(usage)
    if reported is not None:
        return reported, "provider_reported"
    input_tokens = _usage_number(
        usage,
        ("input_tokens", "inputTokens", "prompt_tokens", "promptTokens", "input"),
    )
    cached_tokens = _usage_number(
        usage,
        (
            "cached_input_tokens",
            "cachedInputTokens",
            "cache_read_input_tokens",
            "cacheRead",
        ),
    )
    output_tokens = _usage_number(
        usage,
        ("output_tokens", "outputTokens", "completion_tokens", "completionTokens", "output"),
    )
    uncached = max(0, input_tokens - cached_tokens)
    estimate = (
        uncached * INPUT_PER_MTOK
        + cached_tokens * CACHED_INPUT_PER_MTOK
        + output_tokens * OUTPUT_PER_MTOK
    ) / 1_000_000
    return estimate, "trace_token_estimate"


def _load_outcomes(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not isinstance(value.get("cell_id"), str):
            raise ValueError(f"{path} contains an invalid row")
        rows[str(value["cell_id"])] = {str(key): item for key, item in value.items()}
    return rows


def _write_outcomes(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(rows[key], sort_keys=True) + "\n" for key in sorted(rows)),
        encoding="utf-8",
    )
    temporary.replace(path)


def _moonshiner_config(root: Path, effort: str) -> dict[str, Any]:
    config = _read_object(root / "config.json")
    config["workspace"] = {"confirmed_root": str(root)}
    config["teacher"] = {
        "runtime": "pi",
        "model": MODEL,
        "reasoning": effort,
        "timeout_s": 1_200,
    }
    config["runtimes"] = {
        "pi": {
            "cli": "/usr/local/bin/pi",
            "runtime": "pi-coding-agent",
            "runtime_version": "0.80.7",
            "provider": "openai",
            "display_provider": "openai",
            "base_url": "https://api.openai.com/v1",
            "api": "openai-responses",
            "key_env": "OPENAI_API_KEY",
            "require_observed_model": True,
            "max_output_tokens": 131_072,
        }
    }
    config["pipeline"] = {
        "trace": {
            "coding_system_prompt_append": True,
            "coding_system_prompt_programs": [
                "Building",
                "Debugging",
                "Project & integration",
                "Feature development",
                "Refactoring & performance",
                "Security",
            ],
        }
    }
    return config


def _configure_imports(root: Path, state_root: Path) -> None:
    os.environ["MOONSHINER_BUNDLE_ROOT"] = str(root)
    os.environ["MOONSHINER_HOME"] = str(state_root)
    source = str(root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)


def _remove_workspace(common: _WorkspaceManager, workspace: Path) -> None:
    """Remove a Moonshiner workspace on Python versions without rmtree onexc."""
    try:
        common.remove_workspace(workspace)
        return
    except TypeError as error:
        if "onexc" not in str(error):
            raise

    def force_writable(
        function: Callable[[str], object], path: str, _error: object
    ) -> None:
        Path(path).chmod(Path(path).stat().st_mode | stat.S_IWUSR)
        function(path)

    shutil.rmtree(workspace, onerror=force_writable)


def _seed(corpus_by_id: dict[str, dict[str, Any]], root: Path, task_id: str) -> dict[str, Any]:
    task = corpus_by_id[task_id]
    relative = task.get("seed_relpath")
    if not isinstance(relative, str):
        raise ValueError(f"corpus task {task_id} has no seed_relpath")
    seed_dir = root / relative
    seed = _read_object(seed_dir / "task.json")
    seed["_dir"] = seed_dir
    return seed


class MatrixState:
    def __init__(self, output: Path, *, ceiling_usd: float) -> None:
        self.output = output
        self.outcomes_path = output / "outcomes.jsonl"
        self.ceiling_usd = ceiling_usd
        self.lock = threading.Lock()
        self.rows = _load_outcomes(self.outcomes_path)
        self.reserved = 0.0

    def completed(self, cell_id: str) -> bool:
        with self.lock:
            return cell_id in self.rows

    def reserve(self) -> None:
        with self.lock:
            spent = sum(float(row.get("cost_usd") or 0.0) for row in self.rows.values())
            if spent + self.reserved + DEFAULT_RESERVATION_USD > self.ceiling_usd:
                raise RuntimeError("next Moonshiner cell would exceed the authorized ceiling")
            self.reserved += DEFAULT_RESERVATION_USD

    def persist(self, row: dict[str, Any]) -> None:
        with self.lock:
            self.reserved = max(0.0, self.reserved - DEFAULT_RESERVATION_USD)
            self.rows[str(row["cell_id"])] = row
            _write_outcomes(self.outcomes_path, self.rows)

    def release(self) -> None:
        with self.lock:
            self.reserved = max(0.0, self.reserved - DEFAULT_RESERVATION_USD)


def _run_cell(
    cell: Cell,
    *,
    root: Path,
    corpus_by_id: dict[str, dict[str, Any]],
    output: Path,
    state: MatrixState,
) -> dict[str, Any]:
    common = importlib.import_module("common")
    generate = importlib.import_module("generate_traces")
    runtime_module = importlib.import_module("runtimes.pi")
    state.reserve()
    persisted = False
    try:
        arm_root = output / "traces" / f"attempt-{cell.attempt}" / cell.arm
        config = _moonshiner_config(root, cell.effort)
        teacher = runtime_module.PiRuntime(config, config["teacher"])
        teacher.preflight(require_auth=True)
        seed = _seed(corpus_by_id, root, cell.task_id)
        record = generate.trace_task(
            seed,
            teacher,
            force=True,
            attempts=1,
            traces_root=arm_root,
        )
        teacher_record = record.get("teacher")
        if not isinstance(teacher_record, dict):
            raise RuntimeError(f"cell {cell.cell_id} has no teacher provenance")
        provenance = teacher_record.get("provenance")
        provenance = provenance if isinstance(provenance, dict) else {}
        usage = teacher_record.get("usage")
        usage_object = (
            {str(key): item for key, item in usage.items()}
            if isinstance(usage, dict)
            else {}
        )
        cost_usd, cost_accounting = _cost(usage_object)
        row: dict[str, Any] = {
            "protocol": "moonshiner-effort-outcome-v1",
            "cell_id": cell.cell_id,
            "task_id": cell.task_id,
            "arm": cell.arm,
            "model": MODEL,
            "reasoning_effort": cell.effort,
            "attempt": cell.attempt,
            "reward": 1.0 if record.get("passed") is True else 0.0,
            "passed": record.get("passed"),
            "verify_passed": record.get("verify_passed"),
            "protected_intact": record.get("protected_intact"),
            "model_attested": teacher_record.get("model_attested"),
            "observed_model": teacher_record.get("observed_model"),
            "observed_models": teacher_record.get("observed_models"),
            "upstream_response_models": provenance.get("upstream_response_models"),
            "runtime_version": provenance.get("runtime_version"),
            "provider": provenance.get("provider"),
            "return_code": record.get("return_code"),
            "timed_out": record.get("timed_out"),
            "stream_success": record.get("stream_success"),
            "duration_s": record.get("duration_s"),
            "usage": usage_object,
            "cost_usd": cost_usd,
            "cost_accounting": cost_accounting,
            "raw_sha256": record.get("raw_sha256"),
            "diff_sha256": record.get("diff_sha256"),
            "seed_fingerprint": record.get("seed_fingerprint"),
            "trace_path": record.get("raw_path"),
            "target_outcomes_used": False,
            "trace_size_cap_bytes": MAX_TRACE_BYTES,
        }
        if row["model_attested"] is not True:
            raise RuntimeError(f"cell {cell.cell_id} did not attest the requested model")
        workspace_path = record.get("_workspace_path")
        if not isinstance(workspace_path, str):
            raise RuntimeError(f"cell {cell.cell_id} has no managed workspace path")
        _remove_workspace(common, Path(workspace_path))
        row["workspace_removed"] = True
        state.persist(row)
        persisted = True
        logger.info(
            "cell complete task=%s arm=%s reward=%.1f cost_usd=%.4f",
            cell.task_id,
            cell.arm,
            row["reward"],
            cost_usd,
        )
        return row
    finally:
        if not persisted:
            state.release()


def run_matrix(
    root: Path,
    corpus: Path,
    output: Path,
    *,
    task_limit: int,
    concurrency: int,
    ceiling_usd: float,
    seed: int,
    arm_set: str,
    attempt: int,
) -> None:
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    raw_corpus = _read_object(corpus)
    raw_tasks = raw_corpus.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError(f"{corpus} has no tasks")
    tasks = [
        {str(key): item for key, item in row.items()}
        for row in raw_tasks
        if isinstance(row, dict) and isinstance(row.get("task_id"), str)
    ]
    if len(tasks) != len(raw_tasks):
        raise ValueError(f"{corpus} contains an invalid task row")
    if task_limit > 0:
        tasks = tasks[:task_limit]
    corpus_by_id = {str(row["task_id"]): row for row in tasks}
    output.mkdir(parents=True, exist_ok=True)
    _configure_imports(root, output / "moonshiner-state")
    _patch_moonshiner_runtime()
    state = MatrixState(output, ceiling_usd=ceiling_usd)
    if arm_set not in EFFORT_SETS:
        raise ValueError(f"unsupported arm set: {arm_set}")
    arm_efforts = EFFORT_SETS[arm_set]
    scheduled = _schedule(
        list(corpus_by_id),
        seed=seed,
        attempt=attempt,
        arm_efforts=arm_efforts,
    )
    pending = [cell for cell in scheduled if not state.completed(cell.cell_id)]
    logger.info(
        "matrix start tasks=%d cells=%d pending=%d concurrency=%d",
        len(tasks),
        len(scheduled),
        len(pending),
        concurrency,
    )
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                _run_cell,
                cell,
                root=root,
                corpus_by_id=corpus_by_id,
                output=output,
                state=state,
            ): cell
            for cell in pending
        }
        for future in as_completed(futures):
            future.result()
    rows = _load_outcomes(output / "outcomes.jsonl")
    if len(rows) != len(scheduled):
        raise RuntimeError(f"matrix ended with {len(rows)} of {len(scheduled)} cells")
    manifest = {
        "protocol": "moonshiner-effort-matrix-v1",
        "source_repo": "https://github.com/greghavens/moonshiner",
        "source_commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "model": MODEL,
        "arm_set": arm_set,
        "arms": arm_efforts,
        "attestation_adapter": ATTESTATION_ADAPTER,
        "usage_adapter": USAGE_ADAPTER,
        "trace_size_cap_bytes": MAX_TRACE_BYTES,
        "tasks": len(tasks),
        "cells": len(rows),
        "concurrency": concurrency,
        "seed": seed,
        "attempt": attempt,
        "target_outcomes_used": False,
        "estimated_spend_usd": sum(float(row.get("cost_usd") or 0.0) for row in rows.values()),
        "outcomes_sha256": hashlib.sha256(
            (output / "outcomes.jsonl").read_bytes()
        ).hexdigest(),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "matrix complete tasks=%d cells=%d estimated_spend_usd=%.4f",
        len(tasks),
        len(rows),
        manifest["estimated_spend_usd"],
    )
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moonshiner-root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--ceiling-usd", type=float, default=20_000.0)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--arm-set", choices=tuple(EFFORT_SETS), default="target")
    parser.add_argument("--attempt", type=int, default=0)
    args = parser.parse_args()
    run_matrix(
        args.moonshiner_root,
        args.corpus,
        args.output,
        task_limit=args.task_limit,
        concurrency=args.concurrency,
        ceiling_usd=args.ceiling_usd,
        seed=args.seed,
        arm_set=args.arm_set,
        attempt=args.attempt,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
