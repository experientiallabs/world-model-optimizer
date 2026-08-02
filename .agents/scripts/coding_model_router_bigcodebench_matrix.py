"""Build and screen a BigCodeBench reasoning-effort outcome matrix.

The script runs only on remote experiment compute. It freezes a contamination-
checked task cohort, calls one model at five reasoning efforts, scores generated
code with the pinned official evaluator, and computes a held-out-attempt oracle
before any router fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np

logger = logging.getLogger("coding-model-router-bigcodebench-matrix")

MODEL = "gpt-5.6-luna"
ARMS = {
    "luna-low": "low",
    "luna-medium": "medium",
    "luna-high": "high",
    "luna-xhigh": "xhigh",
    "luna-max": "max",
}
ARM_ORDER = tuple(ARMS)
ATTEMPTS = 5
TASK_LIMIT = 300
SEED = 20260731
MAX_OUTPUT_TOKENS = 32_768
INPUT_PER_MTOK = 1.0
CACHED_INPUT_PER_MTOK = 0.1
OUTPUT_PER_MTOK = 6.0
CELL_RESERVATION_USD = 0.5
DATASET = "bigcode/bigcodebench"
DATASET_REVISION = "b74c0d0bf70d2c0bc459be537895cca163007f1a"
DATASET_PATH = "data/v0.1.4-00000-of-00001.parquet"
HARD_DATASET = "bigcode/bigcodebench-hard"
HARD_REVISION = "298d2cc7b96612e15e47313c3603ee124cee0c1f"
HARD_PATH = "data/v0.1.4-00000-of-00001.parquet"
EVALUATOR_COMMIT = "9059fb84d1188c02edeac4995361656a2fdecbef"
SYSTEM_INSTRUCTION = (
    "Return only the complete Python implementation requested by the user. "
    "Do not use Markdown fences or explanatory prose."
)


@dataclass(frozen=True)
class Cell:
    """One task, effort, and independent attempt."""

    task_id: str
    arm: str
    effort: str
    attempt: int

    @property
    def cell_id(self) -> str:
        return f"{self.task_id}:{self.arm}:attempt-{self.attempt}"


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _target_feature_rows(path: Path) -> list[dict[str, Any]]:
    payload = _read_object(path)
    raw_rows = payload.get("rows")
    if (
        payload.get("target_reward_fields_accessed") is not False
        or payload.get("target_cost_fields_accessed") is not False
        or not isinstance(raw_rows, list)
    ):
        raise ValueError("target feature view is not label-free")
    return [
        {str(key): item for key, item in row.items()}
        for row in raw_rows
        if isinstance(row, dict)
    ]


def _task_id(row: dict[str, Any]) -> str:
    value = row.get("task_id")
    if not isinstance(value, str) or not value:
        raise ValueError("BigCodeBench row has no task_id")
    return value


def _prompt(row: dict[str, Any]) -> str:
    value = row.get("instruct_prompt")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{_task_id(row)} has no instruct_prompt")
    return value


def _library_group(value: object) -> str:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = [value]
        value = decoded
    if not isinstance(value, list):
        return "__missing_libraries__"
    libraries = sorted({str(item).strip().casefold() for item in value if str(item).strip()})
    return "|".join(libraries) if libraries else "__no_libraries__"


def _select_tasks(
    full_rows: list[dict[str, Any]],
    hard_ids: set[str],
    target_rows: list[dict[str, Any]],
    *,
    limit: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_ids = {
        str(row.get("id", row.get("instance_id", "")))
        for row in target_rows
        if str(row.get("id", row.get("instance_id", "")))
    }
    target_texts = {
        _normalize(str(row.get("text", row.get("problem_statement", ""))))
        for row in target_rows
        if _normalize(str(row.get("text", row.get("problem_statement", ""))))
    }
    by_id: dict[str, dict[str, Any]] = {}
    overlap_ids: list[str] = []
    for row in full_rows:
        task_id = _task_id(row)
        prompt = _normalize(_prompt(row))
        if task_id in target_ids or prompt in target_texts:
            overlap_ids.append(task_id)
            continue
        by_id[task_id] = row
    retained_hard = sorted(hard_ids & set(by_id))
    if len(retained_hard) > limit:
        raise ValueError("hard subset exceeds the frozen cohort size")
    fill = sorted(
        set(by_id) - set(retained_hard),
        key=lambda task_id: _stable_key(seed, task_id),
    )[: limit - len(retained_hard)]
    selected_ids = retained_hard + fill
    if len(selected_ids) != limit:
        raise ValueError(f"only {len(selected_ids)} tasks remain for a {limit}-task cohort")
    rows = []
    for task_id in selected_ids:
        source = by_id[task_id]
        rows.append(
            {
                "task_id": task_id,
                "instruct_prompt": _prompt(source),
                "entry_point": source.get("entry_point"),
                "library_group": _library_group(source.get("libs")),
                "is_hard": task_id in hard_ids,
            }
        )
    audit = {
        "source_tasks": len(full_rows),
        "source_hard_tasks": len(hard_ids),
        "target_task_ids": len(target_ids),
        "target_normalized_texts": len(target_texts),
        "overlap_task_ids": sorted(overlap_ids),
        "overlap_tasks_removed": len(overlap_ids),
        "retained_hard_tasks": len(retained_hard),
        "selected_tasks": len(rows),
        "task_family_groups": len({str(row["library_group"]) for row in rows}),
    }
    return rows, audit


def _resolve_url(dataset: str, revision: str, path: str) -> str:
    encoded = urllib.parse.quote(path, safe="/")
    return f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{encoded}"


def _download(url: str, path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "wmo-experiment/1"})
    with urllib.request.urlopen(request, timeout=300) as response:
        with temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
    temporary.replace(path)


def _parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return [
        {str(key): item for key, item in cast(dict[str, Any], row).items()}
        for row in table.to_pylist()
    ]


def prepare(
    output: Path,
    target_feature_view: Path,
    *,
    limit: int = TASK_LIMIT,
    seed: int = SEED,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    cohort_path = output / "cohort.json"
    tasks_path = output / "tasks.jsonl"
    if cohort_path.exists() or tasks_path.exists():
        raise FileExistsError("BigCodeBench cohort is already frozen")
    cache = output / "cache"
    full_path = cache / "bigcodebench-v0.1.4.parquet"
    hard_path = cache / "bigcodebench-hard-v0.1.4.parquet"
    _download(_resolve_url(DATASET, DATASET_REVISION, DATASET_PATH), full_path)
    _download(_resolve_url(HARD_DATASET, HARD_REVISION, HARD_PATH), hard_path)
    full_rows = _parquet_rows(full_path)
    hard_ids = {_task_id(row) for row in _parquet_rows(hard_path)}
    target_rows = _target_feature_rows(target_feature_view)
    selected, audit = _select_tasks(
        full_rows,
        hard_ids,
        target_rows,
        limit=limit,
        seed=seed,
    )
    _write_jsonl(tasks_path, selected)
    cohort = {
        "protocol": "bigcodebench-effort-cohort-v1",
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "dataset_path": DATASET_PATH,
        "dataset_sha256": _sha256_file(full_path),
        "hard_dataset": HARD_DATASET,
        "hard_revision": HARD_REVISION,
        "hard_path": HARD_PATH,
        "hard_sha256": _sha256_file(hard_path),
        "evaluator_commit": EVALUATOR_COMMIT,
        "seed": seed,
        "selection": (
            "all retained hard tasks then sha256(seed:task_id) fill from full"
        ),
        "target_feature_view_sha256": _sha256_file(target_feature_view),
        "target_outcomes_used": False,
        "target_costs_used": False,
        "audit": audit,
        "tasks_sha256": _sha256_file(tasks_path),
    }
    cohort_path.write_text(
        json.dumps(cohort, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "cohort frozen tasks=%d hard=%d overlap=%d groups=%d",
        audit["selected_tasks"],
        audit["retained_hard_tasks"],
        audit["overlap_tasks_removed"],
        audit["task_family_groups"],
    )


def _schedule(tasks: list[dict[str, Any]], *, seed: int = SEED) -> list[Cell]:
    cells = [
        Cell(_task_id(task), arm, effort, attempt)
        for task in tasks
        for arm, effort in ARMS.items()
        for attempt in range(ATTEMPTS)
    ]
    return sorted(cells, key=lambda cell: _stable_key(seed, cell.cell_id))


def _response_text(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str):
        return output_text
    fragments: list[str] = []
    raw_output = response.get("output")
    if not isinstance(raw_output, list):
        return ""
    for item in raw_output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and part.get("type") in {
                "output_text",
                "text",
            }:
                fragments.append(text)
    return "".join(fragments)


def _number(value: object) -> int:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0, int(value))
    return 0


def _usage(response: dict[str, Any]) -> dict[str, int]:
    raw = response.get("usage")
    if not isinstance(raw, dict):
        return {}
    details = raw.get("input_tokens_details")
    details = details if isinstance(details, dict) else {}
    output_details = raw.get("output_tokens_details")
    output_details = output_details if isinstance(output_details, dict) else {}
    return {
        "input_tokens": _number(raw.get("input_tokens")),
        "cached_input_tokens": _number(details.get("cached_tokens")),
        "output_tokens": _number(raw.get("output_tokens")),
        "reasoning_tokens": _number(output_details.get("reasoning_tokens")),
    }


def _cost(usage: dict[str, int]) -> float:
    input_tokens = usage.get("input_tokens", 0)
    cached = usage.get("cached_input_tokens", 0)
    uncached = max(0, input_tokens - cached)
    output = usage.get("output_tokens", 0)
    return (
        uncached * INPUT_PER_MTOK
        + cached * CACHED_INPUT_PER_MTOK
        + output * OUTPUT_PER_MTOK
    ) / 1_000_000


def _provider_call(
    prompt: str,
    effort: str,
    *,
    api_key: str,
    max_retries: int = 5,
) -> tuple[dict[str, Any], float]:
    payload = {
        "model": MODEL,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": SYSTEM_INSTRUCTION}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            },
        ],
        "reasoning": {"effort": effort},
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }
    encoded = json.dumps(payload).encode()
    started = time.monotonic()
    for retry in range(max_retries + 1):
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=encoded,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "wmo-bigcodebench-effort/1",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                body = response.read()
            value = json.loads(body)
            if not isinstance(value, dict):
                raise ValueError("Responses API returned a non-object payload")
            return {str(key): item for key, item in value.items()}, time.monotonic() - started
        except urllib.error.HTTPError as error:
            body = error.read(16_384).decode("utf-8", "replace")
            if error.code != 429 and error.code < 500:
                raise RuntimeError(
                    f"provider rejected the frozen request with HTTP {error.code}: {body}"
                ) from error
            if retry == max_retries:
                raise RuntimeError(
                    f"provider transport failed after retries with HTTP {error.code}: {body}"
                ) from error
        except (TimeoutError, urllib.error.URLError) as error:
            if retry == max_retries:
                raise RuntimeError("provider transport failed after retries") from error
        time.sleep(min(30.0, 2.0**retry))
    raise AssertionError("provider retry loop exhausted unexpectedly")


class MatrixState:
    """Concurrency-safe append-only state for resumable cells."""

    def __init__(
        self,
        output: Path,
        *,
        ceiling_usd: float,
        prior_spend_usd: float,
    ) -> None:
        self.output = output
        self.path = output / "outcomes.jsonl"
        self.ceiling_usd = ceiling_usd
        self.prior_spend_usd = prior_spend_usd
        self.lock = threading.Lock()
        self.rows = {
            str(row["cell_id"]): row
            for row in (_read_jsonl(self.path) if self.path.is_file() else [])
        }
        self.spent = sum(
            float(row.get("cost_usd") or 0.0) for row in self.rows.values()
        )
        self.reserved = 0.0

    def completed(self, cell_id: str) -> bool:
        with self.lock:
            return cell_id in self.rows

    def reserve(self) -> None:
        with self.lock:
            total = self.prior_spend_usd + self.spent
            if total + self.reserved + CELL_RESERVATION_USD > self.ceiling_usd:
                raise RuntimeError("next BigCodeBench cell could exceed the authorized ceiling")
            self.reserved += CELL_RESERVATION_USD

    def persist(self, row: dict[str, Any]) -> None:
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
        with self.lock:
            self.reserved = max(0.0, self.reserved - CELL_RESERVATION_USD)


def _run_cell(
    cell: Cell,
    prompt: str,
    output: Path,
    state: MatrixState,
    *,
    api_key: str,
) -> None:
    state.reserve()
    persisted = False
    try:
        response, duration_s = _provider_call(prompt, cell.effort, api_key=api_key)
        observed_model = response.get("model")
        if observed_model != MODEL:
            raise RuntimeError(
                f"cell {cell.cell_id} observed unexpected model {observed_model!r}"
            )
        raw = json.dumps(response, sort_keys=True).encode()
        raw_sha256 = _sha256_bytes(raw)
        raw_path = output / "raw" / f"{_sha256_bytes(cell.cell_id.encode())}.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(raw + b"\n")
        text = _response_text(response)
        usage = _usage(response)
        row = {
            "protocol": "bigcodebench-effort-cell-v1",
            "cell_id": cell.cell_id,
            "task_id": cell.task_id,
            "arm": cell.arm,
            "model": MODEL,
            "reasoning_effort": cell.effort,
            "attempt": cell.attempt,
            "provider_response_id": response.get("id"),
            "provider_status": response.get("status"),
            "provider_incomplete_details": response.get("incomplete_details"),
            "observed_model": observed_model,
            "duration_s": duration_s,
            "usage": usage,
            "cost_usd": _cost(usage),
            "cost_accounting": "trace_token_estimate",
            "response_text_sha256": _sha256_bytes(text.encode()),
            "response_text_bytes": len(text.encode()),
            "raw_path": str(raw_path),
            "raw_sha256": raw_sha256,
            "target_outcomes_used": False,
        }
        state.persist(row)
        persisted = True
        logger.info(
            "cell complete task=%s arm=%s attempt=%d status=%s cost_usd=%.4f",
            cell.task_id,
            cell.arm,
            cell.attempt,
            row["provider_status"],
            row["cost_usd"],
        )
    finally:
        if not persisted:
            state.release()


def generate(
    output: Path,
    *,
    concurrency: int,
    ceiling_usd: float,
    prior_spend_usd: float,
    seed: int = SEED,
) -> None:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is unavailable")
    cohort = _read_object(output / "cohort.json")
    if cohort.get("target_outcomes_used") is not False:
        raise ValueError("cohort violated the target outcome boundary")
    tasks = _read_jsonl(output / "tasks.jsonl")
    if _sha256_file(output / "tasks.jsonl") != cohort.get("tasks_sha256"):
        raise ValueError("frozen task manifest hash changed")
    prompts = {_task_id(task): _prompt(task) for task in tasks}
    scheduled = _schedule(tasks, seed=seed)
    state = MatrixState(
        output,
        ceiling_usd=ceiling_usd,
        prior_spend_usd=prior_spend_usd,
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
                prompts[cell.task_id],
                output,
                state,
                api_key=api_key,
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
    if len(state.rows) != len(scheduled):
        raise RuntimeError(f"matrix has {len(state.rows)} of {len(scheduled)} cells")
    manifest = {
        "protocol": "bigcodebench-effort-matrix-v1",
        "cohort_sha256": _sha256_file(output / "cohort.json"),
        "tasks_sha256": _sha256_file(output / "tasks.jsonl"),
        "model": MODEL,
        "arms": ARMS,
        "attempts": ATTEMPTS,
        "tasks": len(tasks),
        "cells": len(state.rows),
        "seed": seed,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "target_outcomes_used": False,
        "estimated_spend_usd": sum(
            float(row.get("cost_usd") or 0.0) for row in state.rows.values()
        ),
        "outcomes_sha256": _sha256_file(output / "outcomes.jsonl"),
    }
    (output / "matrix-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "matrix complete cells=%d estimated_spend_usd=%.4f",
        len(state.rows),
        manifest["estimated_spend_usd"],
    )


def build_samples(output: Path) -> None:
    sanitize = cast(
        Callable[..., str],
        getattr(importlib.import_module("bigcodebench.sanitize"), "sanitize"),
    )

    tasks = _read_jsonl(output / "tasks.jsonl")
    task_by_id = {_task_id(task): task for task in tasks}
    task_order = {
        _task_id(task): index for index, task in enumerate(tasks)
    }
    rows = {
        str(row["cell_id"]): row for row in _read_jsonl(output / "outcomes.jsonl")
    }
    scheduled = sorted(
        _schedule(tasks),
        key=lambda cell: (
            task_order[cell.task_id],
            ARM_ORDER.index(cell.arm),
            cell.attempt,
        ),
    )
    if len(rows) != len(scheduled):
        raise RuntimeError("generation matrix is incomplete")
    samples: list[dict[str, Any]] = []
    index: list[dict[str, Any]] = []
    for line_number, cell in enumerate(scheduled):
        row = rows[cell.cell_id]
        raw_path = Path(str(row["raw_path"]))
        response = _read_object(raw_path)
        raw_text = _response_text(response)
        task = task_by_id[cell.task_id]
        entry_point = task.get("entry_point")
        sanitized = sanitize(
            raw_text,
            entrypoint=entry_point if isinstance(entry_point, str) else None,
        )
        samples.append({"task_id": cell.task_id, "solution": sanitized})
        index.append(
            {
                "line_number": line_number,
                "cell_id": cell.cell_id,
                "task_id": cell.task_id,
                "arm": cell.arm,
                "attempt": cell.attempt,
                "solution_sha256": _sha256_bytes(sanitized.encode()),
            }
        )
    _write_jsonl(output / "samples.jsonl", samples)
    _write_jsonl(output / "sample-index.jsonl", index)
    manifest = {
        "protocol": "bigcodebench-effort-samples-v1",
        "evaluator_commit": EVALUATOR_COMMIT,
        "samples": len(samples),
        "samples_sha256": _sha256_file(output / "samples.jsonl"),
        "sample_index_sha256": _sha256_file(output / "sample-index.jsonl"),
        "target_outcomes_used": False,
    }
    (output / "samples-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("samples built rows=%d", len(samples))


def score(output: Path, *, parallel: int) -> None:
    pass_status = getattr(importlib.import_module("bigcodebench.eval"), "PASS")
    evaluate = cast(
        Callable[..., Any],
        getattr(importlib.import_module("bigcodebench.evaluate"), "evaluate"),
    )

    samples_path = output / "samples.jsonl"
    result_path = output / "samples_eval_results.json"
    if result_path.exists():
        raise FileExistsError(f"official result already exists: {result_path}")
    task_ids = [_task_id(row) for row in _read_jsonl(output / "tasks.jsonl")]
    evaluate(
        split="instruct",
        subset="full",
        samples=str(samples_path),
        execution="local",
        selective_evaluate=",".join(task_ids),
        pass_k="1",
        save_pass_rate=False,
        calibrated=True,
        parallel=parallel,
    )
    raw_result = _read_object(result_path)
    raw_eval = raw_result.get("eval")
    if not isinstance(raw_eval, dict):
        raise ValueError("official evaluator result has no eval map")
    by_task_index: dict[str, list[dict[str, Any]]] = {}
    for row in _read_jsonl(output / "sample-index.jsonl"):
        by_task_index.setdefault(str(row["task_id"]), []).append(row)
    score_rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        index_rows = sorted(
            by_task_index[task_id],
            key=lambda row: int(row["line_number"]),
        )
        result_rows = raw_eval.get(task_id)
        if not isinstance(result_rows, list) or len(result_rows) != len(index_rows):
            raise ValueError(f"official evaluator returned incomplete task {task_id}")
        for index_row, result_untyped in zip(index_rows, result_rows, strict=True):
            if not isinstance(result_untyped, dict):
                raise ValueError(f"official evaluator returned invalid task {task_id}")
            result = {str(key): item for key, item in result_untyped.items()}
            details = result.get("details")
            score_rows.append(
                {
                    "protocol": "bigcodebench-effort-score-v1",
                    "cell_id": index_row["cell_id"],
                    "task_id": task_id,
                    "arm": index_row["arm"],
                    "attempt": index_row["attempt"],
                    "reward": float(result.get("status") == pass_status),
                    "status": result.get("status"),
                    "details_sha256": _sha256_bytes(
                        json.dumps(details, sort_keys=True).encode()
                    ),
                    "target_outcomes_used": False,
                }
            )
    _write_jsonl(output / "scores.jsonl", score_rows)
    manifest = {
        "protocol": "bigcodebench-effort-score-matrix-v1",
        "evaluator_commit": EVALUATOR_COMMIT,
        "tasks": len(task_ids),
        "arms": len(ARMS),
        "attempts": ATTEMPTS,
        "cells": len(score_rows),
        "scores_sha256": _sha256_file(output / "scores.jsonl"),
        "official_result_sha256": _sha256_file(result_path),
        "target_outcomes_used": False,
    }
    (output / "score-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("official scoring complete cells=%d", len(score_rows))


def _pick_task_arms(fit: np.ndarray, global_fit: np.ndarray) -> np.ndarray:
    choices = np.empty(fit.shape[0], dtype=np.int64)
    for task_index, task_scores in enumerate(fit):
        best = float(np.max(task_scores))
        candidates = np.flatnonzero(task_scores == best)
        candidate_global = global_fit[candidates]
        choices[task_index] = int(candidates[int(np.argmax(candidate_global))])
    return choices


def _heldout_oracle(
    rewards: np.ndarray,
    groups: list[str],
    *,
    seed: int,
    bootstraps_per_split: int,
) -> dict[str, Any]:
    if rewards.ndim != 3 or rewards.shape[1:] != (len(ARMS), ATTEMPTS):
        raise ValueError("reward tensor has the wrong shape")
    if len(groups) != rewards.shape[0]:
        raise ValueError("task-family group count does not match rewards")
    rng = np.random.default_rng(seed)
    group_array = np.asarray(groups, dtype=object)
    unique_groups = sorted(set(groups))
    group_indices = {
        group: np.flatnonzero(group_array == group)
        for group in unique_groups
    }
    split_summaries: list[dict[str, Any]] = []
    split_headroom_by_task: list[np.ndarray] = []
    for split_index, fit_attempts_tuple in enumerate(
        itertools.combinations(range(ATTEMPTS), 2)
    ):
        fit_attempts = list(fit_attempts_tuple)
        heldout_attempts = [
            attempt for attempt in range(ATTEMPTS) if attempt not in fit_attempts
        ]
        fit = rewards[:, :, fit_attempts].mean(axis=2)
        heldout = rewards[:, :, heldout_attempts].mean(axis=2)
        global_fit = fit.mean(axis=0)
        static_arm = int(np.argmax(global_fit))
        choices = _pick_task_arms(fit, global_fit)
        oracle_by_task = heldout[np.arange(len(choices)), choices]
        static_by_task = heldout[:, static_arm]
        headroom_by_task = oracle_by_task - static_by_task
        headroom = float(np.mean(headroom_by_task))
        split_headroom_by_task.append(headroom_by_task)
        split_summaries.append(
            {
                "split": split_index,
                "fit_attempts": fit_attempts,
                "heldout_attempts": heldout_attempts,
                "fit_selected_static": ARM_ORDER[static_arm],
                "oracle_reward": float(np.mean(oracle_by_task)),
                "static_reward": float(np.mean(static_by_task)),
                "headroom": headroom,
            }
        )
    bootstrap_headroom: list[float] = []
    for _ in range(bootstraps_per_split):
        sampled_groups = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )
        sampled = np.concatenate(
            [group_indices[str(group)] for group in sampled_groups]
        )
        bootstrap_headroom.append(
            float(
                np.mean(
                    [
                        float(np.mean(headroom_by_task[sampled]))
                        for headroom_by_task in split_headroom_by_task
                    ]
                )
            )
        )
    interval = np.quantile(bootstrap_headroom, [0.025, 0.5, 0.975])
    mean_headroom = float(
        np.mean([float(row["headroom"]) for row in split_summaries])
    )
    return {
        "attempt_splits": len(split_summaries),
        "fit_attempts": 2,
        "heldout_attempts": 3,
        "family_bootstraps": bootstraps_per_split,
        "mean_heldout_oracle_headroom": mean_headroom,
        "heldout_oracle_headroom_95ci": [float(value) for value in interval],
        "split_summaries": split_summaries,
    }


def oracle(
    output: Path,
    *,
    seed: int = SEED,
    bootstraps_per_split: int = 2_000,
) -> None:
    tasks = _read_jsonl(output / "tasks.jsonl")
    scores = _read_jsonl(output / "scores.jsonl")
    task_index = {_task_id(task): index for index, task in enumerate(tasks)}
    arm_index = {arm: index for index, arm in enumerate(ARM_ORDER)}
    rewards = np.full((len(tasks), len(ARMS), ATTEMPTS), np.nan)
    for row in scores:
        rewards[
            task_index[str(row["task_id"])],
            arm_index[str(row["arm"])],
            int(row["attempt"]),
        ] = float(row["reward"])
    if not np.isfinite(rewards).all():
        raise ValueError("score matrix is not dense")
    groups = [str(task["library_group"]) for task in tasks]
    result = _heldout_oracle(
        rewards,
        groups,
        seed=seed,
        bootstraps_per_split=bootstraps_per_split,
    )
    interval = cast(list[float], result["heldout_oracle_headroom_95ci"])
    gates = {
        "minimum_tasks": len(tasks) >= 250,
        "complete_five_by_five_matrix": len(scores)
        == len(tasks) * len(ARMS) * ATTEMPTS,
        "minimum_mean_headroom": float(
            result["mean_heldout_oracle_headroom"]
        )
        >= 0.10,
        "minimum_lower_bound": interval[0] > 0.05,
    }
    report = {
        "protocol": {
            "dataset": DATASET,
            "dataset_revision": DATASET_REVISION,
            "hard_dataset": HARD_DATASET,
            "hard_revision": HARD_REVISION,
            "evaluator_commit": EVALUATOR_COMMIT,
            "model": MODEL,
            "arms": ARMS,
            "attempts": ATTEMPTS,
            "target_outcomes_used": False,
        },
        "matrix": {
            "tasks": len(tasks),
            "task_family_groups": len(set(groups)),
            "arms": len(ARMS),
            "attempts_per_arm": ATTEMPTS,
            "cells": len(scores),
        },
        "oracle": result,
        "gates": gates,
        "passed": all(gates.values()),
    }
    report_path = output / "oracle-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "oracle complete headroom=%.4f lower=%.4f passed=%s report_sha256=%s",
        result["mean_heldout_oracle_headroom"],
        interval[0],
        report["passed"],
        _sha256_file(report_path),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--target-feature-view", type=Path, required=True)
    prepare_parser.add_argument("--task-limit", type=int, default=TASK_LIMIT)
    prepare_parser.add_argument("--seed", type=int, default=SEED)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.add_argument("--concurrency", type=int, default=16)
    generate_parser.add_argument("--ceiling-usd", type=float, default=20_000.0)
    generate_parser.add_argument("--prior-spend-usd", type=float, default=14.5822)
    generate_parser.add_argument("--seed", type=int, default=SEED)

    samples_parser = subparsers.add_parser("build-samples")
    samples_parser.add_argument("--output", type=Path, required=True)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.add_argument("--parallel", type=int, default=8)

    oracle_parser = subparsers.add_parser("oracle")
    oracle_parser.add_argument("--output", type=Path, required=True)
    oracle_parser.add_argument("--seed", type=int, default=SEED)
    oracle_parser.add_argument("--bootstraps-per-split", type=int, default=2_000)

    args = parser.parse_args()
    if args.command == "prepare":
        prepare(
            args.output,
            args.target_feature_view,
            limit=args.task_limit,
            seed=args.seed,
        )
    elif args.command == "generate":
        generate(
            args.output,
            concurrency=args.concurrency,
            ceiling_usd=args.ceiling_usd,
            prior_spend_usd=args.prior_spend_usd,
            seed=args.seed,
        )
    elif args.command == "build-samples":
        build_samples(args.output)
    elif args.command == "score":
        score(args.output, parallel=args.parallel)
    elif args.command == "oracle":
        oracle(
            args.output,
            seed=args.seed,
            bootstraps_per_split=args.bootstraps_per_split,
        )
    else:
        raise AssertionError(f"unexpected command: {args.command}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
