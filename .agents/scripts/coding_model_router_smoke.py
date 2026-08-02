"""Run and resume the one four-cell Harbor plus E2B coding-router smoke."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml
from coding_model_router_usage import (
    ESTIMATE_METHOD,
    estimate_usage_from_trace,
    exact_cost_usd,
    usage_from_trace,
    usage_metering_error,
)
from e2b import Sandbox
from harbor.models.job.config import JobConfig
from harbor.models.trial.result import TrialResult

from wmo.agents.default import default_agent
from wmo.core.files import write_text_atomic
from wmo.evals.harbor.scorer import HarborScorer
from wmo.harness.scoring import ScoreCell
from wmo.optimize.knn import fit_knn_artifact
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.policy import EmbedderSpec
from wmo.optimize.routing import evaluate_policy
from wmo.providers.pool import ModelPool, PoolEntry, load_pool

logger = logging.getLogger("coding-model-router-smoke")

EXPERIMENT_ID = "coding-router-20260728"
BENCHMARK = "terminal-bench-2"
FIT_TASK = "break-filter-js-from-html"
HELDOUT_TASK = "log-summary-date-ranges"
TASKS = (FIT_TASK, HELDOUT_TASK)
ARMS = ("oai-luna-high", "ant-haiku45")
MAX_LOGICAL_ATTEMPTS = 3
RETRY_DELAYS_S = (15, 60)
SMOKE_MODEL_SPEND_CAP_USD = 10.0
E2B_ACCOUNT_CAP = 1000
E2B_LIST_PAGE_SIZE = 100
INFRASTRUCTURE_STOPS = frozenset(
    {
        "error",
        "provider_error",
        "unknown_done_reason",
    }
)


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return {str(key): item for key, item in value.items()}


def _write_json(path: Path, value: object) -> None:
    write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _scenario_id(task_id: str) -> str:
    return f"{BENCHMARK}:{task_id}"


def _require_e2b_capacity() -> None:
    """Fail before paid work when the configured account cap has no slot."""
    paginator = Sandbox.list(limit=E2B_LIST_PAGE_SIZE)
    active_count = 0
    while True:
        active_count += len(paginator.next_items())
        if active_count >= E2B_ACCOUNT_CAP or not paginator.has_next:
            break
    if active_count >= E2B_ACCOUNT_CAP:
        raise RuntimeError(
            f"E2B has at least {active_count} active sandboxes against the frozen "
            f"{E2B_ACCOUNT_CAP}-sandbox account cap"
        )


def _trace_path(artifact_dir: Path) -> Path:
    for candidate in (
        artifact_dir / "agent" / "wmo-run.json",
        artifact_dir / "wmo-run.json",
    ):
        if candidate.is_file():
            return candidate
    raise ValueError(f"no wmo-run.json under Harbor artifact {artifact_dir}")


def _tool_calls(trace: dict[str, object]) -> int:
    steps = trace.get("steps")
    if not isinstance(steps, list):
        return 0
    count = 0
    for step in steps:
        if not isinstance(step, dict):
            continue
        action = step.get("action")
        if isinstance(action, dict) and action.get("kind") == "tool_call":
            count += 1
    return count


def _wall_seconds(artifact_dir: Path) -> float:
    result = TrialResult.model_validate_json(
        (artifact_dir / "result.json").read_text(encoding="utf-8")
    )
    if result.started_at is None or result.finished_at is None:
        return 0.0
    return max(0.0, (result.finished_at - result.started_at).total_seconds())


def _provider_execution_started(trace: dict[str, object]) -> bool:
    steps = trace.get("steps")
    if isinstance(steps, list) and bool(steps):
        return True
    worker_usage = trace.get("worker_usage")
    if not isinstance(worker_usage, dict):
        return False
    calls = worker_usage.get("calls")
    if isinstance(calls, int) and calls > 0:
        return True
    call_seconds = worker_usage.get("call_seconds")
    if isinstance(call_seconds, list) and bool(call_seconds):
        return True
    return any(
        isinstance(worker_usage.get(key), int) and worker_usage[key] > 0
        for key in (
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
            "reasoning_tokens",
        )
    )


def _failure_class(
    cell: ScoreCell,
    stop_reason: str,
    *,
    provider_execution_started: bool,
) -> str:
    if cell.infra_failed:
        if stop_reason == "provider_error" and provider_execution_started:
            return "agent_failure"
        return "infrastructure"
    if stop_reason == "provider_error" and provider_execution_started:
        return "agent_failure"
    if stop_reason in INFRASTRUCTURE_STOPS or stop_reason.startswith("agent-exception:"):
        return "infrastructure"
    if cell.reward == 1.0:
        return ""
    return "task_failure" if stop_reason == "submitted" else "agent_failure"


def _trace_error(trace: dict[str, object]) -> str:
    steps = trace.get("steps")
    if not isinstance(steps, list):
        return ""
    for step in steps:
        if not isinstance(step, dict):
            continue
        observation = step.get("observation")
        if not isinstance(observation, dict) or observation.get("is_error") is not True:
            continue
        content = observation.get("content")
        if isinstance(content, str):
            return content[-2_000:]
    return ""


def _known_pre_worker_failure(trace: dict[str, object]) -> bool:
    """Whether trace evidence proves no provider request could have started."""
    return _trace_error(trace).startswith("remote materialize failed")


def _outcome(
    cell: ScoreCell,
    *,
    entry: PoolEntry,
    logical_attempt: int,
    artifact_dir: Path,
) -> ScenarioOutcome:
    trace = _read_object(_trace_path(artifact_dir))
    usage = usage_from_trace(trace)
    steps = trace.get("steps")
    instruction = trace.get("instruction")
    stop = trace.get("stop_reason")
    stop_reason = stop if isinstance(stop, str) else ""
    failure_class = _failure_class(
        cell,
        stop_reason,
        provider_execution_started=_provider_execution_started(trace),
    )
    metering_error = "" if _known_pre_worker_failure(trace) else usage_metering_error(usage)
    usage_estimated = bool(metering_error) and failure_class != "infrastructure"
    if usage_estimated:
        usage = estimate_usage_from_trace(trace)
    ungradeable = failure_class == "infrastructure"
    return ScenarioOutcome(
        scenario_id=_scenario_id(cell.task_id),
        task=instruction if isinstance(instruction, str) else cell.task_id,
        model=entry.name,
        benchmark=BENCHMARK,
        episode=logical_attempt - 1,
        attempt_number=logical_attempt,
        reward=None if ungradeable else cell.reward,
        success=cell.passed and not ungradeable,
        critique=cell.note,
        steps=len(steps) if isinstance(steps, list) else 0,
        tool_calls=_tool_calls(trace),
        stop_reason=stop_reason,
        usage=usage.total,
        cost_usd=exact_cost_usd(entry, usage),
        call_seconds=usage.call_seconds,
        call_input_tokens=usage.call_input_tokens,
        call_output_tokens=usage.call_output_tokens,
        call_cached_input_tokens=usage.call_cached_input_tokens,
        call_cache_write_input_tokens=usage.call_cache_write_input_tokens,
        usage_accounting="estimated" if usage_estimated else "exact",
        usage_estimate_method=ESTIMATE_METHOD if usage_estimated else "",
        wall_seconds=_wall_seconds(artifact_dir),
        completion_status=(
            "infrastructure_failure"
            if failure_class == "infrastructure"
            else "scored_pass"
            if cell.passed
            else "scored_agent_failure"
            if failure_class == "agent_failure"
            else "scored_failure"
        ),
        failure_class=failure_class,
        artifact_dir=str(artifact_dir.resolve()),
        error=(_trace_error(trace) or cell.note or None) if ungradeable else None,
        remeasured=logical_attempt > 1,
    )


def _load_matrix(path: Path, pool: ModelPool) -> OutcomeMatrix:
    measured = [pool.entry(name) for name in ARMS]
    if not path.exists():
        return OutcomeMatrix(pool=measured, outcomes=[])
    matrix = OutcomeMatrix.load(path)
    if matrix.pool != measured:
        raise ValueError(f"{path} carries a different smoke pool")
    return matrix


def _upsert_outcome(path: Path, matrix: OutcomeMatrix, outcome: ScenarioOutcome) -> None:
    key = (outcome.scenario_id, outcome.model, outcome.attempt_number)
    matrix.outcomes = [
        existing
        for existing in matrix.outcomes
        if (existing.scenario_id, existing.model, existing.attempt_number) != key
    ] + [outcome]
    matrix.save(path)


def _ledger_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} contains a non-object JSONL row")
        rows.append({str(key): item for key, item in value.items()})
    return rows


def _upsert_ledger(path: Path, outcome: ScenarioOutcome) -> None:
    rows = _ledger_rows(path)
    event_id = (
        f"smoke:{outcome.benchmark}:{outcome.scenario_id}:{outcome.model}:{outcome.attempt_number}"
    )
    row: dict[str, object] = {
        "event_id": event_id,
        "recorded_at": _utc_now(),
        "phase": "smoke",
        "benchmark": outcome.benchmark,
        "scenario_id": outcome.scenario_id,
        "model": outcome.model,
        "attempt_number": outcome.attempt_number,
        "usage": outcome.usage.model_dump(mode="json"),
        "model_call_seconds": outcome.call_seconds,
        "task_environment_wall_seconds": outcome.wall_seconds,
        "model_cost_usd": None if outcome.failure_class == "metering" else outcome.cost_usd,
        "model_cost_accounting_status": (
            "missing_provider_usage"
            if outcome.failure_class == "metering"
            else "estimated_from_trace"
            if outcome.usage_accounting == "estimated"
            else "exact_from_provider_usage"
        ),
        "usage_estimate_method": outcome.usage_estimate_method,
        "task_environment_cost_usd": None,
        "task_environment_cost_note": "E2B invoice rate is absent from Harbor artifacts",
        "completion_status": outcome.completion_status,
        "failure_class": outcome.failure_class,
        "artifact_dir": outcome.artifact_dir,
    }
    rows = [existing for existing in rows if existing.get("event_id") != event_id]
    rows.append(row)
    write_text_atomic(
        path,
        "".join(json.dumps(existing, sort_keys=True) + "\n" for existing in rows),
    )


def _model_spend(matrix: OutcomeMatrix) -> float:
    return sum(outcome.cost_usd for outcome in matrix.outcomes)


def _completed(matrix: OutcomeMatrix, task_id: str, arm: str) -> bool:
    scenario = _scenario_id(task_id)
    return any(
        outcome.scenario_id == scenario and outcome.model == arm and outcome.reward is not None
        for outcome in matrix.outcomes
    )


def _attempt_count(matrix: OutcomeMatrix, task_id: str, arm: str) -> int:
    scenario = _scenario_id(task_id)
    return sum(
        outcome.scenario_id == scenario and outcome.model == arm for outcome in matrix.outcomes
    )


def _template(path: Path, jobs_dir: Path) -> JobConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping")
    return JobConfig.model_validate({**raw, "jobs_dir": str(jobs_dir)})


async def _scorer(
    template_path: Path,
    jobs_dir: Path,
    task_id: str,
    entry: PoolEntry,
) -> HarborScorer:
    return await HarborScorer.create(
        _template(template_path, jobs_dir),
        [task_id],
        provider_config=entry.provider_config(),
        attempts=1,
        task_environment="e2b",
        harness_backend="local",
        episode_timeout_s=300,
        agent_concurrency=1,
        harbor_retries=0,
        missing_reward="zero",
    )


def _archive_infra(
    root: Path,
    artifact_dir: Path,
    *,
    task_id: str,
    arm: str,
    attempt: int,
) -> Path:
    target = root / "infra-attempts" / arm / task_id / f"attempt-{attempt}"
    if target.exists():
        source_digest = _artifact_digest(str(artifact_dir))
        if _artifact_digest(str(target)) == source_digest:
            return target
        target = target.with_name(f"{target.name}-{source_digest[:12]}")
        if target.exists():
            if _artifact_digest(str(target)) == source_digest:
                return target
            raise RuntimeError(f"archive digest collision at {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(artifact_dir, target)
    return target


def _run_cell(
    root: Path,
    *,
    template_path: Path,
    task_id: str,
    entry: PoolEntry,
    matrix_path: Path,
    ledger_path: Path,
    pool: ModelPool,
) -> None:
    matrix = _load_matrix(matrix_path, pool)
    if _completed(matrix, task_id, entry.name):
        logger.info("resume: keeping completed %s x %s", task_id, entry.name)
        return
    logical_attempt = _attempt_count(matrix, task_id, entry.name) + 1
    while logical_attempt <= MAX_LOGICAL_ATTEMPTS:
        if _model_spend(matrix) >= SMOKE_MODEL_SPEND_CAP_USD:
            raise RuntimeError(f"smoke model spend cap ${SMOKE_MODEL_SPEND_CAP_USD:.2f} reached")
        scorer = asyncio.run(
            _scorer(
                template_path,
                root / "harbor" / entry.name / task_id / f"attempt-{logical_attempt}",
                task_id,
                entry,
            )
        )
        cell = scorer.score(default_agent("coding-router-smoke")).cells[0]
        artifact_dir = Path(cell.artifact_dir)
        if cell.infra_failed:
            artifact_dir = _archive_infra(
                root,
                artifact_dir,
                task_id=task_id,
                arm=entry.name,
                attempt=logical_attempt,
            )
        outcome = _outcome(
            cell,
            entry=entry,
            logical_attempt=logical_attempt,
            artifact_dir=artifact_dir,
        )
        if outcome.reward is None and not cell.infra_failed:
            artifact_dir = _archive_infra(
                root,
                artifact_dir,
                task_id=task_id,
                arm=entry.name,
                attempt=logical_attempt,
            )
            outcome.artifact_dir = str(artifact_dir.resolve())
        _upsert_outcome(matrix_path, matrix, outcome)
        _upsert_ledger(ledger_path, outcome)
        logger.info(
            "persisted %s x %s attempt %d reward=%s cost=$%.6f",
            task_id,
            entry.name,
            logical_attempt,
            outcome.reward,
            outcome.cost_usd,
        )
        if outcome.reward is not None:
            return
        if logical_attempt >= MAX_LOGICAL_ATTEMPTS:
            raise RuntimeError(f"{task_id} x {entry.name} remained ungradeable after 3 attempts")
        delay = RETRY_DELAYS_S[logical_attempt - 1]
        logger.warning("ungradeable cell; retrying a fresh sandbox in %ds", delay)
        time.sleep(delay)
        matrix = _load_matrix(matrix_path, pool)
        logical_attempt += 1


def _normalize_existing_ungradeable_attempts(
    root: Path,
    matrix_path: Path,
    ledger_path: Path,
    matrix: OutcomeMatrix,
) -> OutcomeMatrix:
    """Correct pre-fix scaffold and unmetered scored rows before resume."""
    changed = False
    for outcome in matrix.outcomes:
        task_id = outcome.scenario_id.split(":", 1)[1]
        attempt_root = (
            root / "harbor" / outcome.model / task_id / f"attempt-{outcome.attempt_number}"
        )
        source_traces = list(attempt_root.rglob("agent/wmo-run.json"))
        artifact_dir = (
            source_traces[0].parent.parent
            if len(source_traces) == 1
            else Path(outcome.artifact_dir)
        )
        trace = _read_object(_trace_path(artifact_dir))
        stop = trace.get("stop_reason")
        stop_reason = stop if isinstance(stop, str) else ""
        usage = usage_from_trace(trace)
        metering_error = "" if _known_pre_worker_failure(trace) else usage_metering_error(usage)
        post_execution_provider_error = (
            stop_reason == "provider_error" and _provider_execution_started(trace)
        )
        if metering_error and outcome.reward is not None:
            estimated = estimate_usage_from_trace(trace)
            entry = next(
                pool_entry for pool_entry in matrix.pool if pool_entry.name == outcome.model
            )
            outcome.usage = estimated.total
            outcome.cost_usd = exact_cost_usd(entry, estimated)
            outcome.call_seconds = estimated.call_seconds
            outcome.call_input_tokens = estimated.call_input_tokens
            outcome.call_output_tokens = estimated.call_output_tokens
            outcome.call_cached_input_tokens = estimated.call_cached_input_tokens
            outcome.call_cache_write_input_tokens = estimated.call_cache_write_input_tokens
            outcome.usage_accounting = "estimated"
            outcome.usage_estimate_method = ESTIMATE_METHOD
            outcome.failure_class = (
                ""
                if outcome.reward == 1.0
                else "task_failure"
                if stop_reason == "submitted"
                else "agent_failure"
            )
            outcome.completion_status = (
                "scored_pass"
                if outcome.reward == 1.0
                else "scored_failure"
                if stop_reason == "submitted"
                else "scored_agent_failure"
            )
            outcome.error = None
            _upsert_ledger(ledger_path, outcome)
            changed = True
            continue
        if metering_error and outcome.usage_accounting != "estimated":
            failure_class = "metering"
            completion_status = "metering_failure"
            error = metering_error
        elif (
            (
                outcome.failure_class in {"scaffold", "infrastructure"}
                and not post_execution_provider_error
            )
            or (stop_reason in INFRASTRUCTURE_STOPS and not post_execution_provider_error)
            or stop_reason.startswith("agent-exception:")
        ):
            failure_class = "infrastructure"
            completion_status = "infrastructure_failure"
            error = _trace_error(trace) or "infrastructure stopped before grading"
        else:
            continue
        if (
            outcome.reward is None
            and outcome.failure_class == failure_class
            and outcome.completion_status == completion_status
            and outcome.error == error
        ):
            continue
        archived = _archive_infra(
            root,
            artifact_dir,
            task_id=task_id,
            arm=outcome.model,
            attempt=outcome.attempt_number,
        )
        outcome.reward = None
        outcome.success = False
        outcome.completion_status = completion_status
        outcome.failure_class = failure_class
        outcome.error = error
        outcome.artifact_dir = str(archived.resolve())
        _upsert_ledger(ledger_path, outcome)
        changed = True
    if changed:
        matrix.save(matrix_path)
    return matrix


def _artifact_digest(path: str) -> str:
    root = Path(path)
    digest = hashlib.sha256()
    if root.is_file():
        digest.update(root.read_bytes())
        return digest.hexdigest()
    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(root)).encode())
        digest.update(item.read_bytes())
    return digest.hexdigest()


def _quarantine_derived_path(root: Path, path: Path) -> dict[str, str]:
    """Move one stale derived artifact into a digest-addressed evidence directory."""
    digest = _artifact_digest(str(path))
    target = root / "invalid-derived" / f"{path.name}-{digest[:12]}"
    if target.exists():
        if _artifact_digest(str(target)) != digest:
            raise RuntimeError(f"quarantine digest collision at {target}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
    return {
        "original_path": str(path.resolve()),
        "quarantined_path": str(target.resolve()),
        "sha256": digest,
    }


def _invalidate_smoke_derivatives(root: Path, matrix: OutcomeMatrix) -> None:
    """Quarantine fit/replay outputs when paid rows lack exact usage evidence."""
    unknown = sorted(
        (outcome for outcome in matrix.outcomes if outcome.failure_class == "metering"),
        key=lambda row: (row.scenario_id, row.model, row.attempt_number),
    )
    if not unknown:
        return
    quarantine: list[dict[str, str]] = []
    for name in ("policy", "smoke-report.json", "resume-proof.json"):
        path = root / name
        if path.exists():
            quarantine.append(_quarantine_derived_path(root, path))
    invalid_path = root / "invalidated.json"
    prior = _read_object(invalid_path) if invalid_path.is_file() else {}
    invalidated_at = prior.get("invalidated_at")
    prior_quarantine: list[object] = []
    raw_prior_quarantine = prior.get("quarantined_derived_artifacts")
    if isinstance(raw_prior_quarantine, list):
        prior_quarantine.extend(raw_prior_quarantine)
    _write_json(
        invalid_path,
        {
            "protocol": "coding-router-smoke-invalid-v1",
            "valid": False,
            "invalidated_at": (invalidated_at if isinstance(invalidated_at, str) else _utc_now()),
            "reason": (
                "paid model calls completed without provider usage; exact spend, token capture, "
                "and the smoke cap cannot be proven"
            ),
            "paid_execution_allowed": False,
            "unknown_cost_attempts": [
                {
                    "scenario_id": outcome.scenario_id,
                    "model": outcome.model,
                    "attempt_number": outcome.attempt_number,
                    "artifact_dir": outcome.artifact_dir,
                    "completion_status": outcome.completion_status,
                    "model_cost_usd": None,
                }
                for outcome in unknown
            ],
            "quarantined_derived_artifacts": [*prior_quarantine, *quarantine],
        },
    )


def _drop_duplicate_retry_noops(
    root: Path,
    matrix_path: Path,
    ledger_path: Path,
    matrix: OutcomeMatrix,
) -> OutcomeMatrix:
    """Exclude deterministic Harbor resumes that made no new execution attempt."""
    seen: dict[tuple[str, str], set[str]] = {}
    kept: list[ScenarioOutcome] = []
    duplicates: list[dict[str, object]] = []
    duplicate_event_ids: set[str] = set()
    for outcome in sorted(
        matrix.outcomes,
        key=lambda row: (row.scenario_id, row.model, row.attempt_number),
    ):
        cell = (outcome.scenario_id, outcome.model)
        digest = _artifact_digest(outcome.artifact_dir)
        cell_digests = seen.setdefault(cell, set())
        if digest not in cell_digests:
            cell_digests.add(digest)
            kept.append(outcome)
            continue
        duplicates.append(
            {
                "scenario_id": outcome.scenario_id,
                "model": outcome.model,
                "attempt_number": outcome.attempt_number,
                "artifact_dir": outcome.artifact_dir,
                "artifact_digest": digest,
                "reason": (
                    "Harbor resumed an identical deterministic trial directory; no new "
                    "sandbox or provider call occurred"
                ),
            }
        )
        duplicate_event_ids.add(
            f"smoke:{outcome.benchmark}:{outcome.scenario_id}:"
            f"{outcome.model}:{outcome.attempt_number}"
        )
    if not duplicates:
        return matrix
    existing_path = root / "retry-noops.json"
    existing = _read_object(existing_path).get("rows") if existing_path.is_file() else []
    existing_rows = existing if isinstance(existing, list) else []
    by_key: dict[tuple[object, object, object], object] = {}
    for row in [*existing_rows, *duplicates]:
        if isinstance(row, dict):
            by_key[(row.get("scenario_id"), row.get("model"), row.get("attempt_number"))] = row
    _write_json(
        existing_path,
        {
            "protocol": "coding-router-smoke-retry-noops-v1",
            "rows": list(by_key.values()),
        },
    )
    matrix.outcomes = kept
    matrix.save(matrix_path)
    ledger_rows = [
        row for row in _ledger_rows(ledger_path) if row.get("event_id") not in duplicate_event_ids
    ]
    write_text_atomic(
        ledger_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in ledger_rows),
    )
    return matrix


def _fit(root: Path, matrix: OutcomeMatrix) -> None:
    policy_dir = root / "policy"
    policy_dir.mkdir(parents=True, exist_ok=True)
    policy_path = policy_dir / "policy.json"
    fitted = fit_knn_artifact(
        matrix,
        out_path=policy_path,
        matrix_source=str(root / "outcomes.json"),
        embedder=EmbedderSpec(kind="hashing", dim=1024),
        fit_ids=[_scenario_id(FIT_TASK)],
        z=0.5,
        rag_num=1,
        rag_thres=0.9,
        min_pairs=1,
        se_floor=True,
        floor_q=0.0,
    )
    heldout = evaluate_policy(fitted.policy, matrix, [_scenario_id(HELDOUT_TASK)])
    exact_spend = sum(
        outcome.cost_usd for outcome in matrix.outcomes if outcome.usage_accounting == "exact"
    )
    estimated_spend = sum(
        outcome.cost_usd for outcome in matrix.outcomes if outcome.usage_accounting == "estimated"
    )
    _write_json(
        root / "smoke-report.json",
        {
            "completed_at": _utc_now(),
            "cells": len(matrix.outcomes),
            "gradeable_cells": sum(outcome.reward is not None for outcome in matrix.outcomes),
            "model_spend_usd": _model_spend(matrix),
            "exact_model_spend_usd": exact_spend,
            "estimated_model_spend_usd": estimated_spend,
            "estimated_usage_cells": sum(
                outcome.usage_accounting == "estimated" for outcome in matrix.outcomes
            ),
            "fit_task": FIT_TASK,
            "heldout_task": HELDOUT_TASK,
            "fit": fitted.model_dump(mode="json"),
            "heldout": heldout.model_dump(mode="json"),
            "policy_path": str(policy_path.resolve()),
        },
    )


def preflight(root: Path, template_path: Path) -> None:
    """Resolve exact tasks and provider configs without launching a model or sandbox."""
    pool = load_pool(root.parent / "pool.toml")
    manifest = _read_object(root.parent / "tasks" / "terminal-bench-2.json")
    raw_tasks = manifest.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("Terminal-Bench manifest has no task rows")
    pins: dict[str, str] = {}
    for raw in raw_tasks:
        if not isinstance(raw, dict):
            continue
        task_id = raw.get("task_id")
        if not isinstance(task_id, str):
            continue
        pins[task_id] = f"git:{raw.get('git_url')}@{raw.get('git_commit_id')}"
    resolved: dict[str, object] = {}
    for arm in ARMS:
        entry = pool.entry(arm)
        for task_id in TASKS:
            scorer = asyncio.run(
                _scorer(
                    template_path,
                    root / "harbor" / entry.name / task_id,
                    task_id,
                    entry,
                )
            )
            if scorer.task_pins != {task_id: pins[task_id]}:
                raise ValueError(
                    f"{task_id} resolved to {scorer.task_pins}, expected {pins[task_id]}"
                )
            resolved[f"{arm}:{task_id}"] = {
                "provider": entry.provider_config().model_dump(mode="json"),
                "task_pin": pins[task_id],
                "attempts": scorer.request.attempts,
            }
    split = _read_object(root.parent / "splits" / "seed-0.json")
    terminal = split.get(BENCHMARK)
    if not isinstance(terminal, dict):
        raise ValueError("seed-0 split has no Terminal-Bench cohort")
    fit_ids = terminal.get("fit")
    heldout_ids = terminal.get("heldout")
    if (
        not isinstance(fit_ids, list)
        or not isinstance(heldout_ids, list)
        or FIT_TASK not in fit_ids
        or HELDOUT_TASK not in heldout_ids
    ):
        raise ValueError("smoke tasks do not preserve the declared fit and held-out roles")
    _write_json(
        root / "preflight.json",
        {
            "verified_at": _utc_now(),
            "paid_calls": 0,
            "fit_task": FIT_TASK,
            "heldout_task": HELDOUT_TASK,
            "cells": resolved,
        },
    )
    logger.info("preflight passed for %d exact cells; no paid calls made", len(resolved))


def run(root: Path, template_path: Path, *, interrupt_after_cells: int | None) -> None:
    """Execute or resume the exact four-cell gate."""
    for variable in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "E2B_API_KEY"):
        if not os.environ.get(variable):
            raise ValueError(f"{variable} is required for the integrated smoke")
    _require_e2b_capacity()
    pool = load_pool(root.parent / "pool.toml")
    matrix_path = root / "outcomes.json"
    ledger_path = root.parent / "spend-ledger.jsonl"
    root.mkdir(parents=True, exist_ok=True)
    matrix = _load_matrix(matrix_path, pool)
    matrix = _drop_duplicate_retry_noops(root, matrix_path, ledger_path, matrix)
    matrix = _normalize_existing_ungradeable_attempts(root, matrix_path, ledger_path, matrix)
    _invalidate_smoke_derivatives(root, matrix)
    prior = [
        {
            "scenario_id": outcome.scenario_id,
            "model": outcome.model,
            "attempt_number": outcome.attempt_number,
            "artifact_digest": _artifact_digest(outcome.artifact_dir),
        }
        for outcome in matrix.outcomes
        if outcome.reward is not None
    ]
    started_with = sum(_completed(matrix, task, arm) for arm in ARMS for task in TASKS)
    for arm in ARMS:
        for task_id in TASKS:
            _run_cell(
                root,
                template_path=template_path,
                task_id=task_id,
                entry=pool.entry(arm),
                matrix_path=matrix_path,
                ledger_path=ledger_path,
                pool=pool,
            )
            matrix = _load_matrix(matrix_path, pool)
            completed = sum(_completed(matrix, task, model) for model in ARMS for task in TASKS)
            if (
                interrupt_after_cells is not None
                and started_with < interrupt_after_cells <= completed
            ):
                _write_json(
                    root / "interruption.json",
                    {
                        "interrupted_at": _utc_now(),
                        "completed_cells": completed,
                        "reason": "intentional smoke resume checkpoint",
                    },
                )
                raise InterruptedError(
                    f"intentional checkpoint after {completed} completed smoke cells"
                )
    matrix = _load_matrix(matrix_path, pool)
    if sum(_completed(matrix, task, arm) for arm in ARMS for task in TASKS) != 4:
        raise RuntimeError("smoke matrix is incomplete")
    current = {
        (outcome.scenario_id, outcome.model, outcome.attempt_number): _artifact_digest(
            outcome.artifact_dir
        )
        for outcome in matrix.outcomes
        if outcome.reward is not None
    }
    unchanged = all(
        current.get((row["scenario_id"], row["model"], row["attempt_number"]))
        == row["artifact_digest"]
        for row in prior
    )
    _write_json(
        root / "resume-proof.json",
        {
            "verified_at": _utc_now(),
            "resumed_cells": len(prior),
            "unchanged": unchanged,
            "prior_cells": prior,
        },
    )
    _fit(root, matrix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(".wmo/experiments") / EXPERIMENT_ID / "smoke",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("wmo/distill/configs/tb2-harbor-job-template.yaml"),
    )
    parser.add_argument("--interrupt-after-cells", type=int)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight(args.root, args.template)
        return
    try:
        run(args.root, args.template, interrupt_after_cells=args.interrupt_after_cells)
    except InterruptedError as error:
        logger.warning("%s", error)
        raise SystemExit(75) from None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
