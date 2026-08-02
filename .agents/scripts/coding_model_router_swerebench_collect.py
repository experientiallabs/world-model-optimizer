"""Collect the complete external SWE-rebench matrix into fit-ready outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("coding-router-swerebench-collect")

PROTOCOL = "coding-router-swerebench-development-collection-v2"
CONFIRMATION_PROTOCOL = "coding-router-swerebench-confirmation-collection-v1"
POOLED_CONFIRMATION_PROTOCOL = "coding-router-pooled-uplift-confirmation-collection-v1"
DEVELOPMENT_EXECUTION_PROTOCOL = "coding-router-swerebench-development-execution-v1"
CONFIRMATION_EXECUTION_PROTOCOL = "coding-router-swerebench-confirmation-execution-v1"
POOLED_CONFIRMATION_EXECUTION_PROTOCOL = (
    "coding-router-pooled-uplift-confirmation-execution-v1"
)
CORPUS_SHA256 = "7d846b5576d15e68fd18ac21bfe0610cc1614b3b35ec0ae0cb8cfae0b82962c1"
CONFIRMATION_CORPUS_SHA256 = (
    "9798dd1e58be0d13331d097307670dc3fc3760ad211da20e6367666523f080a7"
)
POOLED_CONFIRMATION_CORPUS_SHA256 = (
    "6edd8ed4777d6bc48cf29f76a9fb4b9d60e3324908aa79d4d03df8617f6be825"
)
SMOKE_REPORT_SHA256 = "ee76a57040cbe7aaef692d2fc3f3df66d7a556cbf6dda74119e0802cb4230e13"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
SOURCE_TASKS = 200
MIN_RETAINED_TASKS = 190
USAGE_FIELDS = (
    "prompt_tokens",
    "cached_input_tokens",
    "completion_tokens",
    "reasoning_tokens",
)


@dataclass(frozen=True)
class CollectionPhase:
    """Frozen collection differences between development and confirmation."""

    name: str
    protocol: str
    execution_protocol: str
    corpus_sha256: str
    provenance: str
    reuse_smoke: bool
    requires_authorization: bool
    model: str
    arm_prefix: str
    prices_per_mtok: tuple[float, float, float]


DEVELOPMENT_PHASE = CollectionPhase(
    name="development",
    protocol=PROTOCOL,
    execution_protocol=DEVELOPMENT_EXECUTION_PROTOCOL,
    corpus_sha256=CORPUS_SHA256,
    provenance="development-matrix",
    reuse_smoke=True,
    requires_authorization=False,
    model="gpt-5.6-luna",
    arm_prefix="luna",
    prices_per_mtok=(1.0, 0.1, 6.0),
)
CONFIRMATION_PHASE = CollectionPhase(
    name="confirmation",
    protocol=CONFIRMATION_PROTOCOL,
    execution_protocol=CONFIRMATION_EXECUTION_PROTOCOL,
    corpus_sha256=CONFIRMATION_CORPUS_SHA256,
    provenance="confirmation-matrix",
    reuse_smoke=False,
    requires_authorization=True,
    model="gpt-5.6-luna",
    arm_prefix="luna",
    prices_per_mtok=(1.0, 0.1, 6.0),
)
POOLED_CONFIRMATION_PHASE = CollectionPhase(
    name="pooled-confirmation",
    protocol=POOLED_CONFIRMATION_PROTOCOL,
    execution_protocol=POOLED_CONFIRMATION_EXECUTION_PROTOCOL,
    corpus_sha256=POOLED_CONFIRMATION_CORPUS_SHA256,
    provenance="pooled-confirmation-matrix",
    reuse_smoke=False,
    requires_authorization=True,
    model="gpt-5.6-luna",
    arm_prefix="luna",
    prices_per_mtok=(1.0, 0.1, 6.0),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    return value


def _read_object(path: Path) -> dict[str, Any]:
    return _object(json.loads(path.read_text(encoding="utf-8")), str(path))


def _collection_phase(name: str) -> CollectionPhase:
    if name == DEVELOPMENT_PHASE.name:
        return DEVELOPMENT_PHASE
    if name == CONFIRMATION_PHASE.name:
        return CONFIRMATION_PHASE
    if name == POOLED_CONFIRMATION_PHASE.name:
        return POOLED_CONFIRMATION_PHASE
    raise ValueError(f"unknown collection phase: {name!r}")


def _launch_context(root: Path, phase: CollectionPhase) -> tuple[float, dict[str, object]]:
    launch_path = root / "launch.json"
    launch = _read_object(launch_path)
    if (
        launch.get("protocol") != phase.execution_protocol
        or launch.get("corpus_sha256") != phase.corpus_sha256
        or launch.get("model") != phase.model
        or launch.get("deep_swe_outcomes_accessed") is not False
        or launch.get("model_persisted") is not False
    ):
        raise ValueError(f"{phase.name} launch manifest is invalid")
    prior_spend = launch.get("prior_spend_usd")
    if (
        isinstance(prior_spend, bool)
        or not isinstance(prior_spend, (int, float))
        or not 0.0 <= float(prior_spend) < 20_000.0
    ):
        raise ValueError(f"{phase.name} launch has invalid prior spend")
    context: dict[str, object] = {"launch_sha256": _sha256(launch_path)}
    if phase.requires_authorization:
        authorization = launch.get("authorization")
        if (
            launch.get("confirmation_outcomes_accessed_before_launch") is not False
            or not isinstance(authorization, dict)
            or authorization.get("confirmation_corpus_sha256") != phase.corpus_sha256
        ):
            raise ValueError("confirmation launch lacks frozen authorization")
        context["authorization"] = authorization
    return float(prior_spend), context


def _usage(value: object, label: str) -> dict[str, int]:
    raw = _object(value, label)
    result: dict[str, int] = {}
    for field in USAGE_FIELDS:
        amount = raw.get(field)
        if not isinstance(amount, int) or isinstance(amount, bool) or amount < 0:
            raise ValueError(f"{label} has invalid {field}")
        result[field] = amount
    if result["reasoning_tokens"] > result["completion_tokens"]:
        raise ValueError(f"{label} reasoning exceeds completion tokens")
    return result


def _cost(usage: dict[str, int], phase: CollectionPhase = DEVELOPMENT_PHASE) -> float:
    input_rate, cached_input_rate, output_rate = phase.prices_per_mtok
    return (
        usage["prompt_tokens"] * input_rate / 1_000_000
        + usage["cached_input_tokens"] * cached_input_rate / 1_000_000
        + usage["completion_tokens"] * output_rate / 1_000_000
    )


def _smoke_cells(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if _sha256(path) != SMOKE_REPORT_SHA256:
        raise ValueError("smoke report hash mismatch")
    report = _read_object(path)
    if report.get("valid") is not True:
        raise ValueError("smoke report is not valid")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    archives = report.get("archives")
    if not isinstance(archives, list):
        raise ValueError("smoke report has no archives")
    for raw_archive in archives:
        archive = _object(raw_archive, "smoke archive")
        effort = archive.get("effort")
        cells = archive.get("cells")
        if effort not in {"xhigh", "max"} or not isinstance(cells, list):
            raise ValueError("smoke archive has invalid effort or cells")
        for raw_cell in cells:
            cell = _object(raw_cell, "smoke cell")
            task_id = cell.get("task_id")
            if not isinstance(task_id, str):
                raise ValueError("smoke cell has no task id")
            key = (task_id, str(effort))
            if key in result:
                raise ValueError(f"duplicate smoke cell {key}")
            result[key] = cell
    if len(result) != 4:
        raise ValueError("smoke report does not contain exactly four reused cells")
    return result


def _outcome(
    task: dict[str, Any],
    effort: str,
    attempt: int,
    cell: dict[str, Any],
    *,
    provenance: str,
    phase: CollectionPhase = DEVELOPMENT_PHASE,
) -> dict[str, object]:
    reward = cell.get("reward")
    if (
        not isinstance(reward, (int, float))
        or isinstance(reward, bool)
        or float(reward) not in {0.0, 1.0}
    ):
        raise ValueError("cell has invalid gradeable reward")
    usage = _usage(cell.get("usage"), "cell usage")
    return {
        "task_id": str(task["task_id"]),
        "repository": str(task["repository"]),
        "language": str(task["language"]),
        "prompt": str(task["prompt"]),
        "prompt_sha256": str(task["prompt_sha256"]),
        "arm": f"{phase.arm_prefix}-{effort}",
        "model": phase.model,
        "reasoning_effort": effort,
        "attempt_number": attempt,
        "reward": float(reward),
        "reward_provenance": cell.get("reward_provenance", "official verifier"),
        "official_verifier_reached": cell.get("official_verifier_reached", True),
        "cost_usd": _cost(usage, phase),
        "cost_provenance": "trace-derived frozen list-price estimate",
        "usage": usage,
        "provider_calls": int(cell.get("provider_calls", 0)),
        "stop_condition": cell.get("stop_condition"),
        "patch_sha256": cell.get("patch_sha256"),
        "provenance": provenance,
        "target_outcomes_used": False,
    }


def collect(
    root: Path,
    corpus_path: Path,
    smoke_report_path: Path | None,
    output: Path,
    *,
    phase_name: str = "development",
) -> None:
    """Validate retained task reports and drop whole infrastructure-missing tasks."""
    phase = _collection_phase(phase_name)
    if _sha256(corpus_path) != phase.corpus_sha256:
        raise ValueError(f"{phase.name} corpus hash mismatch")
    if phase.reuse_smoke and smoke_report_path is None:
        raise ValueError("development collection requires the frozen smoke report")
    if not phase.reuse_smoke and smoke_report_path is not None:
        raise ValueError("confirmation collection must not reuse a smoke report")
    prior_spend_usd, launch_context = _launch_context(root, phase)
    progress = _read_object(root / "progress.json")
    complete_tasks = progress.get("complete_tasks")
    excluded_count = progress.get("excluded_tasks", 0)
    if (
        progress.get("protocol") != phase.execution_protocol
        or not isinstance(complete_tasks, int)
        or not isinstance(excluded_count, int)
        or complete_tasks + excluded_count != SOURCE_TASKS
        or complete_tasks < MIN_RETAINED_TASKS
        or progress.get("failed_tasks") != 0
    ):
        raise ValueError("development matrix is not complete and failure-free")
    corpus = _read_object(corpus_path)
    raw_tasks = corpus.get("tasks")
    if not isinstance(raw_tasks, list) or len(raw_tasks) != SOURCE_TASKS:
        raise ValueError("development corpus does not contain 200 tasks")
    tasks = [_object(task, f"corpus task {index}") for index, task in enumerate(raw_tasks)]
    reused = _smoke_cells(smoke_report_path) if smoke_report_path is not None else {}
    outcomes: list[dict[str, object]] = []
    input_hashes: dict[str, dict[str, str]] = {}
    exclusions: list[dict[str, object]] = []
    reused_cells = 0
    excluded_infrastructure_cost = 0.0
    for index, task in enumerate(tasks):
        task_id = str(task["task_id"])
        task_dir = root / "tasks" / f"{index:04d}"
        state_path = task_dir / "state.json"
        state = _read_object(state_path)
        if state.get("protocol") != phase.execution_protocol or state.get("task_id") != task_id:
            raise ValueError(f"task {index} state identity changed")
        if state.get("stage") == "excluded-infrastructure":
            exclusion = _object(state.get("exclusion"), f"task {index} exclusion")
            if (
                exclusion.get("scope") != "whole-task"
                or exclusion.get("effort") not in EFFORTS
                or not isinstance(exclusion.get("reason"), str)
                or not isinstance(exclusion.get("evidence_sha256"), str)
                or len(exclusion["evidence_sha256"]) != 64
                or not isinstance(exclusion.get("usage"), dict)
                or not isinstance(exclusion.get("provider_calls"), int)
                or not isinstance(exclusion.get("observed_scientific_cells"), int)
                or exclusion.get("scientific_cells_rerun") != 0
            ):
                raise ValueError(f"task {index} has an invalid exclusion")
            effort = str(exclusion["effort"])
            exclusion_report = task_dir / f"{effort}.infrastructure-missing.json"
            exclusion_archive = task_dir / f"{effort}.infrastructure-missing.tar.gz"
            if (
                _sha256(exclusion_report) != exclusion.get("report_sha256")
                or _sha256(exclusion_archive) != exclusion.get("evidence_sha256")
            ):
                raise ValueError(f"task {index} exclusion evidence hash mismatch")
            exclusion_usage = _usage(
                exclusion.get("usage"), f"task {index} exclusion usage"
            )
            excluded_infrastructure_cost += _cost(exclusion_usage, phase)
            excluded_efforts = _object(
                state.get("efforts"), f"task {index} completed excluded efforts"
            )
            for completed_effort, raw_payload in excluded_efforts.items():
                payload = _object(
                    raw_payload,
                    f"task {index} completed excluded effort {completed_effort}",
                )
                excluded_infrastructure_cost += _cost(
                    _usage(
                        payload.get("usage"),
                        f"task {index} completed excluded effort usage",
                    ),
                    phase,
                )
            exclusions.append(
                {
                    "task_id": task_id,
                    "task_index": index,
                    "scope": "whole-task",
                    "effort": exclusion["effort"],
                    "reason": exclusion["reason"],
                    "evidence_sha256": exclusion.get("evidence_sha256"),
                    "scientific_cells_rerun": 0,
                }
            )
            input_hashes[task_id] = {
                "state": _sha256(state_path),
                "exclusion_report": _sha256(exclusion_report),
                "exclusion_archive": _sha256(exclusion_archive),
            }
            continue
        if state.get("stage") != "complete" or state.get("task_id") != task_id:
            raise ValueError(f"task {index} is not complete with frozen identity")
        efforts = _object(state.get("efforts"), f"task {index} efforts")
        input_hashes[task_id] = {"state": _sha256(state_path)}
        for effort in EFFORTS:
            payload = _object(efforts.get(effort), f"task {index} {effort} state")
            report_path = task_dir / f"{effort}.report.json"
            archive_path = task_dir / f"{effort}.tar.gz"
            report_sha = _sha256(report_path)
            archive_sha = _sha256(archive_path)
            if report_sha != payload.get("report_sha256"):
                raise ValueError(f"task {index} {effort} report hash mismatch")
            if archive_sha != payload.get("archive_sha256"):
                raise ValueError(f"task {index} {effort} archive hash mismatch")
            report = _read_object(report_path)
            if (
                report.get("valid") is not True
                or report.get("task_id") != task_id
                or report.get("effort") != effort
            ):
                raise ValueError(f"task {index} {effort} report identity mismatch")
            cells = report.get("cells")
            if not isinstance(cells, list):
                raise ValueError(f"task {index} {effort} report has no cells")
            attempt_cells: dict[int, dict[str, Any]] = {}
            for raw_cell in cells:
                cell = _object(raw_cell, f"task {index} {effort} cell")
                attempt = cell.get("attempt_number")
                if not isinstance(attempt, int) or isinstance(attempt, bool):
                    raise ValueError(f"task {index} {effort} has invalid attempt")
                if attempt in attempt_cells:
                    raise ValueError(f"task {index} {effort} duplicates attempt {attempt}")
                attempt_cells[attempt] = cell
            smoke_cell = reused.get((task_id, effort))
            if smoke_cell is not None:
                if 0 in attempt_cells or set(attempt_cells) != {1}:
                    raise ValueError(f"task {index} {effort} did not preserve smoke attempt zero")
                attempt_cells[0] = smoke_cell
            if set(attempt_cells) != {0, 1}:
                raise ValueError(f"task {index} {effort} attempts are incomplete")
            for attempt in (0, 1):
                provenance = (
                    "reused-valid-smoke"
                    if smoke_cell is not None and attempt == 0
                    else phase.provenance
                )
                outcomes.append(
                    _outcome(
                        task,
                        effort,
                        attempt,
                        attempt_cells[attempt],
                        provenance=provenance,
                        phase=phase,
                    )
                )
                if smoke_cell is not None and attempt == 0:
                    reused_cells += 1
            input_hashes[task_id][f"{effort}_report"] = report_sha
            input_hashes[task_id][f"{effort}_archive"] = archive_sha
    retained_tasks = SOURCE_TASKS - len(exclusions)
    expected_cells = retained_tasks * len(EFFORTS) * 2
    if retained_tasks < MIN_RETAINED_TASKS or len(exclusions) != excluded_count:
        raise ValueError("task exclusions violate the frozen coverage gate")
    if len(outcomes) != expected_cells:
        raise ValueError(f"expected {expected_cells} outcomes, found {len(outcomes)}")
    identities = {
        (row["task_id"], row["reasoning_effort"], row["attempt_number"])
        for row in outcomes
    }
    if len(identities) != expected_cells:
        raise ValueError("collected outcome identities are not unique")
    output.mkdir(parents=True, exist_ok=False)
    outcomes_path = output / "outcomes.jsonl"
    outcomes_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in outcomes),
        encoding="utf-8",
    )
    total_cost = math.fsum(float(row["cost_usd"]) for row in outcomes)
    reused_cost = math.fsum(
        float(row["cost_usd"])
        for row in outcomes
        if row["provenance"] == "reused-valid-smoke"
    )
    audit = {
        "protocol": phase.protocol,
        "valid": True,
        "source_tasks": SOURCE_TASKS,
        "tasks": retained_tasks,
        "retained_task_coverage": retained_tasks / SOURCE_TASKS,
        "excluded_tasks": exclusions,
        "efforts": list(EFFORTS),
        "model": phase.model,
        "arm_prefix": phase.arm_prefix,
        "attempts_per_effort": 2,
        "cells": expected_cells,
        "unique_cell_identities": len(identities),
        "reused_smoke_cells": reused_cells,
        "new_matrix_cells": expected_cells - reused_cells,
        "outcome_cost_usd": total_cost,
        "reused_smoke_cost_usd": reused_cost,
        "new_matrix_cost_usd": total_cost - reused_cost,
        "excluded_infrastructure_cost_usd": excluded_infrastructure_cost,
        "spent_matrix_cost_usd": total_cost - reused_cost
        + excluded_infrastructure_cost,
        "rough_cumulative_experiment_spend_usd": prior_spend_usd
        + total_cost
        - reused_cost
        + excluded_infrastructure_cost,
        "target_outcomes_used": False,
        "deep_swe_outcomes_accessed": False,
        "outcomes_sha256": _sha256(outcomes_path),
        "corpus_sha256": phase.corpus_sha256,
        "launch": launch_context,
        "input_hashes": input_hashes,
    }
    if phase.reuse_smoke:
        audit["smoke_report_sha256"] = SMOKE_REPORT_SHA256
    elif phase.requires_authorization:
        audit["confirmation_outcomes_accessed"] = True
        audit["confirmation_authorization_preserved"] = True
    (output / "completion-audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info(
        "collected cells=%d new_cost_usd=%.6f output_sha256=%s",
        len(outcomes),
        total_cost - reused_cost,
        audit["outcomes_sha256"],
    )


def main() -> None:
    """Parse paths and collect a complete matrix."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--smoke-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=(
            DEVELOPMENT_PHASE.name,
            CONFIRMATION_PHASE.name,
            POOLED_CONFIRMATION_PHASE.name,
        ),
        default=DEVELOPMENT_PHASE.name,
    )
    args = parser.parse_args()
    collect(
        args.root,
        args.corpus,
        args.smoke_report,
        args.output,
        phase_name=args.phase,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
