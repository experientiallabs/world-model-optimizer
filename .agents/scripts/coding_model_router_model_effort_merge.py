"""Merge the three frozen SWE-rebench model matrices with whole-task exclusions."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("coding-router-model-effort-merge")

PROTOCOL = "coding-router-model-effort-development-merge-v1"
CORPUS_SHA256 = "7d846b5576d15e68fd18ac21bfe0610cc1614b3b35ec0ae0cb8cfae0b82962c1"
LUNA_OUTCOMES_SHA256 = "5c2097116b03291f20bc33d6a376cb01d9a2e9fb182f46c482df5508b7140ee2"
LUNA_AUDIT_SHA256 = "ca20ebdc85bda0482e9c726a95c4216bc1b4acec63fe86db88ef4fc4431ab316"
FROZEN_PRIOR_SPEND_USD = 1_123.9297378
EFFORTS = ("low", "medium", "high", "xhigh", "max")
MODELS = {
    "luna": "gpt-5.6-luna",
    "terra": "gpt-5.6-terra",
    "sol": "gpt-5.6-sol",
}
ATTEMPTS = 2
SOURCE_TASKS = 200
MIN_RETAINED_TASKS = 190


@dataclass(frozen=True)
class MatrixSource:
    """One collected model matrix and its frozen identity."""

    prefix: str
    outcomes: Path
    audit: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
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


def _validate_source(source: MatrixSource) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    audit_sha = _sha256(source.audit)
    outcomes_sha = _sha256(source.outcomes)
    if source.prefix == "luna" and (
        audit_sha != LUNA_AUDIT_SHA256 or outcomes_sha != LUNA_OUTCOMES_SHA256
    ):
        raise ValueError("frozen Luna collection changed")
    audit = _read_object(source.audit)
    rows = _read_rows(source.outcomes)
    retained = audit.get("tasks")
    exclusions = audit.get("excluded_tasks")
    if (
        audit.get("valid") is not True
        or audit.get("source_tasks") != SOURCE_TASKS
        or not isinstance(retained, int)
        or not MIN_RETAINED_TASKS <= retained <= SOURCE_TASKS
        or not isinstance(exclusions, list)
        or len(exclusions) != SOURCE_TASKS - retained
        or audit.get("retained_task_coverage") != retained / SOURCE_TASKS
        or audit.get("cells") != retained * len(EFFORTS) * ATTEMPTS
        or audit.get("outcomes_sha256") != outcomes_sha
        or audit.get("target_outcomes_used") is not False
        or audit.get("deep_swe_outcomes_accessed") is not False
    ):
        raise ValueError(f"{source.prefix} collection audit is incomplete or unsafe")
    if source.prefix != "luna" and (
        audit.get("model") != MODELS[source.prefix]
        or audit.get("arm_prefix") != source.prefix
    ):
        raise ValueError(f"{source.prefix} collection model identity changed")
    return audit, rows


def _validate_row(row: dict[str, Any], prefix: str) -> tuple[str, str, int]:
    task_id = row.get("task_id")
    arm = row.get("arm")
    attempt = row.get("attempt_number")
    reward = row.get("reward")
    cost = row.get("cost_usd")
    expected_arms = {f"{prefix}-{effort}" for effort in EFFORTS}
    if (
        not isinstance(task_id, str)
        or not isinstance(arm, str)
        or arm not in expected_arms
        or not isinstance(attempt, int)
        or isinstance(attempt, bool)
        or not 0 <= attempt < ATTEMPTS
        or row.get("model") != MODELS[prefix]
        or row.get("target_outcomes_used") is not False
        or isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or float(reward) not in {0.0, 1.0}
        or isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(float(cost))
        or float(cost) < 0.0
    ):
        raise ValueError(f"invalid {prefix} outcome identity {(task_id, arm, attempt)!r}")
    return task_id, arm, attempt


def merge(
    corpus_path: Path,
    sources: tuple[MatrixSource, ...],
    output: Path,
) -> None:
    """Validate, whole-task intersect, and merge all frozen model matrices."""
    if _sha256(corpus_path) != CORPUS_SHA256:
        raise ValueError("development corpus changed")
    if tuple(source.prefix for source in sources) != tuple(MODELS):
        raise ValueError("model sources must be ordered Luna, Terra, Sol")
    raw_tasks = _read_object(corpus_path).get("tasks")
    if not isinstance(raw_tasks, list) or len(raw_tasks) != SOURCE_TASKS:
        raise ValueError("development corpus must contain 200 tasks")
    task_ids = {
        str(task["task_id"])
        for task in raw_tasks
        if isinstance(task, dict) and isinstance(task.get("task_id"), str)
    }
    if len(task_ids) != SOURCE_TASKS:
        raise ValueError("development task identities are invalid")

    validated: list[tuple[MatrixSource, dict[str, Any], list[dict[str, Any]]]] = []
    exclusion_sources: dict[str, list[dict[str, object]]] = {}
    for source in sources:
        audit, rows = _validate_source(source)
        validated.append((source, audit, rows))
        for raw in audit["excluded_tasks"]:
            if not isinstance(raw, dict) or raw.get("scope") != "whole-task":
                raise ValueError(f"{source.prefix} has an invalid exclusion")
            task_id = raw.get("task_id")
            if not isinstance(task_id, str) or task_id not in task_ids:
                raise ValueError(f"{source.prefix} excludes an unknown task")
            exclusion_sources.setdefault(task_id, []).append(
                {
                    "model": MODELS[source.prefix],
                    "reason": str(raw.get("reason")),
                    "evidence_sha256": str(raw.get("evidence_sha256")),
                    "scientific_cells_rerun": raw.get("scientific_cells_rerun"),
                }
            )
    excluded_ids = set(exclusion_sources)
    retained_ids = task_ids - excluded_ids
    if len(retained_ids) < MIN_RETAINED_TASKS:
        raise ValueError("cross-model whole-task coverage is below 95 percent")

    merged: list[dict[str, Any]] = []
    identities: set[tuple[str, str, int]] = set()
    input_hashes: dict[str, dict[str, str]] = {}
    new_spend = 0.0
    for source, audit, rows in validated:
        input_hashes[source.prefix] = {
            "outcomes_sha256": _sha256(source.outcomes),
            "completion_audit_sha256": _sha256(source.audit),
        }
        if source.prefix != "luna":
            spend = audit.get("spent_matrix_cost_usd")
            if isinstance(spend, bool) or not isinstance(spend, (int, float)):
                raise ValueError(f"{source.prefix} audit lacks matrix spend")
            new_spend += float(spend)
        for row in rows:
            identity = _validate_row(row, source.prefix)
            if identity[0] in excluded_ids:
                continue
            if identity in identities:
                raise ValueError(f"duplicate cross-model outcome {identity!r}")
            identities.add(identity)
            merged.append(row)

    expected = len(retained_ids) * len(MODELS) * len(EFFORTS) * ATTEMPTS
    if len(merged) != expected or len(identities) != expected:
        raise ValueError(f"expected {expected} dense outcomes, found {len(merged)}")
    merged.sort(key=lambda row: (str(row["task_id"]), str(row["arm"]), int(row["attempt_number"])))
    output.mkdir(parents=True, exist_ok=False)
    outcomes_path = output / "outcomes.jsonl"
    outcomes_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in merged),
        encoding="utf-8",
    )
    exclusions = [
        {
            "task_id": task_id,
            "scope": "whole-task",
            "sources": exclusion_sources[task_id],
            "scientific_cells_rerun": 0,
        }
        for task_id in sorted(excluded_ids)
    ]
    audit = {
        "protocol": PROTOCOL,
        "valid": True,
        "source_tasks": SOURCE_TASKS,
        "tasks": len(retained_ids),
        "retained_task_coverage": len(retained_ids) / SOURCE_TASKS,
        "excluded_tasks": exclusions,
        "models": MODELS,
        "efforts": list(EFFORTS),
        "arms": [f"{prefix}-{effort}" for prefix in MODELS for effort in EFFORTS],
        "attempts_per_arm": ATTEMPTS,
        "cells": expected,
        "unique_cell_identities": len(identities),
        "new_matrix_spend_usd": new_spend,
        "rough_cumulative_experiment_spend_usd": FROZEN_PRIOR_SPEND_USD + new_spend,
        "target_outcomes_used": False,
        "deep_swe_outcomes_accessed": False,
        "fitted_numeric_router_state_persisted": False,
        "corpus_sha256": CORPUS_SHA256,
        "outcomes_sha256": _sha256(outcomes_path),
        "inputs": input_hashes,
    }
    (output / "completion-audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "merged tasks=%d cells=%d new_spend_usd=%.6f outcomes_sha256=%s",
        len(retained_ids),
        expected,
        new_spend,
        audit["outcomes_sha256"],
    )


def main() -> None:
    """Parse frozen collection inputs and merge them."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    for prefix in MODELS:
        parser.add_argument(f"--{prefix}-outcomes", type=Path, required=True)
        parser.add_argument(f"--{prefix}-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = tuple(
        MatrixSource(
            prefix,
            getattr(args, f"{prefix}_outcomes"),
            getattr(args, f"{prefix}_audit"),
        )
        for prefix in MODELS
    )
    merge(args.corpus, sources, args.output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
