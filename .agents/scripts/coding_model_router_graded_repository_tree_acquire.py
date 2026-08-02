"""Acquire exact public Git trees and construct ephemeral repository features.

This command is intended for an internet-enabled E2B acquisition worker. It projects only the
five allowed columns from the pinned SWE-rebench parquet, validates them against the frozen task
manifest, and never loads patches, tests, verifier metadata, or install configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from coding_model_router_graded_repository_tree import (
    PROTOCOL,
    RawTreeEntry,
    feature_blocks,
    feature_views,
    validate_tree,
)

DATASET_REVISION = "475dd5e8703bb5fb22dd3c60b5d038b019eba1e0"
DATASET_PARQUET_SHA256 = (
    "0e0bf9355f892ad74ae98d4e1c404f39fd6654a8e351ee3e6ab162e4a64cd3ad"
)
DEVELOPMENT_CORPUS_SHA256 = (
    "48d88436a083b66972c25cd7d9439fd149c95bcf9caded2bab7f3b6453aea3d5"
)
COLLECTION_AUDIT_SHA256 = (
    "d256c23e6661d9b1ef232c74d52922b5c1ce83a69ca1c87e723c97266a71064b"
)
ALLOWED_COLUMNS = (
    "instance_id",
    "repo",
    "language",
    "problem_statement",
    "base_commit",
    "image_name",
)
MIN_DEVELOPMENT_COVERAGE = 0.95
EXPECTED_RETAINED_TASKS = 649
HTTP_ATTEMPTS = 3
HTTP_BACKOFF_SECONDS = (1.0, 3.0)

logger = logging.getLogger("coding-router-graded-repository-tree-acquire")


@dataclass(frozen=True)
class DatasetTask:
    """The only projected SWE-rebench fields eligible for acquisition."""

    task_id: str
    repository: str
    language: str
    prompt: str
    base_commit: str
    image_name: str


@dataclass(frozen=True)
class FeatureRow:
    """One ephemeral task feature row passed only to remote fitting."""

    task_id: str
    repository: str
    language: str
    base_commit: str
    structure: tuple[float, ...]
    localization: tuple[float, ...]
    prompt_shape: tuple[float, ...]


@dataclass(frozen=True)
class ProjectionResult:
    """Exact source rows plus label-free whole-task rejections."""

    tasks: tuple[DatasetTask, ...]
    failures: tuple[dict[str, str], ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return {str(key): item for key, item in value.items()}


def _manifest_tasks(path: Path, audit_path: Path) -> list[dict[str, Any]]:
    if _sha256(path) != DEVELOPMENT_CORPUS_SHA256:
        raise ValueError("development corpus changed")
    if _sha256(audit_path) != COLLECTION_AUDIT_SHA256:
        raise ValueError("development completion audit changed")
    manifest = _object(json.loads(path.read_text(encoding="utf-8")), "development corpus")
    audit = _object(json.loads(audit_path.read_text(encoding="utf-8")), "completion audit")
    raw = manifest.get("tasks")
    exclusions = audit.get("excluded_tasks")
    if not isinstance(raw, list) or len(raw) != 673 or not isinstance(exclusions, list):
        raise ValueError("development corpus or exclusions are invalid")
    excluded = {
        str(row["task_id"])
        for row in exclusions
        if isinstance(row, dict) and row.get("scope") == "whole-task"
    }
    tasks = [
        _object(row, "development task")
        for row in raw
        if isinstance(row, dict) and str(row.get("task_id")) not in excluded
    ]
    if len(tasks) != EXPECTED_RETAINED_TASKS:
        raise ValueError("retained development task count changed")
    return tasks


def _dataset_task(row: dict[str, Any]) -> DatasetTask:
    def required_string(column: str) -> str:
        value = row.get(column)
        if not isinstance(value, str) or not value:
            raise ValueError("projected dataset row contains an empty or non-string field")
        return value

    return DatasetTask(
        task_id=required_string("instance_id"),
        repository=required_string("repo"),
        language=required_string("language"),
        prompt=required_string("problem_statement"),
        base_commit=required_string("base_commit"),
        image_name=required_string("image_name"),
    )


def validate_projection(
    manifest_tasks: Sequence[dict[str, Any]],
    dataset_rows: Iterable[dict[str, Any]],
) -> ProjectionResult:
    """Validate an allowed-column-only dataset projection against the frozen manifest."""
    wanted = {str(task["task_id"]): task for task in manifest_tasks}
    selected: dict[str, DatasetTask] = {}
    failures: dict[str, dict[str, str]] = {}
    seen: set[str] = set()
    for row in dataset_rows:
        task_id = row.get("instance_id")
        if not isinstance(task_id, str) or task_id not in wanted:
            continue
        if task_id in seen:
            raise ValueError("pinned dataset repeats a retained task identity")
        seen.add(task_id)
        projected = _dataset_task(row)
        frozen = wanted[task_id]
        comparisons = {
            "repository": (projected.repository, frozen.get("repository")),
            "language": (projected.language.casefold(), str(frozen.get("language")).casefold()),
            "prompt": (projected.prompt, frozen.get("prompt")),
            "image": (projected.image_name, frozen.get("image_name")),
        }
        mismatches = [name for name, pair in comparisons.items() if pair[0] != pair[1]]
        if mismatches:
            failures[task_id] = {
                "task_id": task_id,
                "repository": str(frozen.get("repository", "")),
                "reason_type": f"source-identity-mismatch:{','.join(mismatches)}",
            }
        else:
            selected[task_id] = projected
    for task_id in sorted(set(wanted) - set(selected) - set(failures)):
        failures[task_id] = {
            "task_id": task_id,
            "repository": str(wanted[task_id].get("repository", "")),
            "reason_type": "source-row-missing",
        }
    ordered_tasks = tuple(
        selected[str(task["task_id"])]
        for task in manifest_tasks
        if str(task["task_id"]) in selected
    )
    ordered_failures = tuple(
        failures[str(task["task_id"])]
        for task in manifest_tasks
        if str(task["task_id"]) in failures
    )
    return ProjectionResult(ordered_tasks, ordered_failures)


def load_projected_dataset(parquet_path: Path) -> Iterable[dict[str, Any]]:
    """Yield only the preregistered columns from the exact pinned native parquet."""
    if _sha256(parquet_path) != DATASET_PARQUET_SHA256:
        raise ValueError("pinned native dataset parquet changed")
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(parquet_path)
    schema_names = set(parquet.schema_arrow.names)
    if not set(ALLOWED_COLUMNS) <= schema_names:
        raise ValueError("pinned dataset lacks an allowed projected column")
    for batch in parquet.iter_batches(columns=list(ALLOWED_COLUMNS), batch_size=128):
        for row in batch.to_pylist():
            yield _object(row, "projected dataset row")


def _github_payload(repository: str, base_commit: str, token: str | None) -> dict[str, Any]:
    encoded_repository = urllib.parse.quote(repository, safe="/")
    encoded_commit = urllib.parse.quote(base_commit, safe="")
    url = f"https://api.github.com/repos/{encoded_repository}/git/trees/{encoded_commit}?recursive=1"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "experiential-labs-coding-router-tree-v1",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(HTTP_ATTEMPTS):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read())
            return _object(payload, "GitHub tree response")
        except urllib.error.HTTPError as error:
            retryable = error.code in {429, 500, 502, 503, 504}
            if not retryable or attempt == HTTP_ATTEMPTS - 1:
                raise
        except (TimeoutError, urllib.error.URLError):
            if attempt == HTTP_ATTEMPTS - 1:
                raise
        time.sleep(HTTP_BACKOFF_SECONDS[attempt])
    raise AssertionError("GitHub retry loop did not return or raise")


def _raw_entries(payload: dict[str, Any]) -> tuple[RawTreeEntry, ...]:
    rows = payload.get("tree")
    if not isinstance(rows, list):
        raise ValueError("GitHub tree response lacks rows")
    entries = []
    for value in rows:
        row = _object(value, "GitHub tree row")
        path = row.get("path")
        object_type = row.get("type")
        mode = row.get("mode")
        size = row.get("size")
        if (
            not isinstance(path, str)
            or not isinstance(object_type, str)
            or not isinstance(mode, str)
            or isinstance(size, bool)
            or size is not None
            and not isinstance(size, int)
        ):
            raise ValueError("GitHub tree row has invalid allowed fields")
        entries.append(RawTreeEntry(path, object_type, mode, size))
    return tuple(entries)


def acquire_feature_rows(
    tasks: Sequence[DatasetTask], token: str | None
) -> tuple[list[FeatureRow], list[dict[str, str]]]:
    """Acquire exact trees and return ephemeral rows plus label-free failures."""
    rows: list[FeatureRow] = []
    failures: list[dict[str, str]] = []
    for index, task in enumerate(tasks, start=1):
        try:
            payload = _github_payload(task.repository, task.base_commit, token)
            files = validate_tree(
                _raw_entries(payload),
                truncated=payload.get("truncated") is not False,
            )
            blocks = feature_blocks(files, issue=task.prompt, language=task.language)
            views = feature_views(blocks)
            if tuple(len(view) for view in views) != (61, 100, 115):
                raise RuntimeError("frozen repository feature dimensions changed")
            rows.append(
                FeatureRow(
                    task_id=task.task_id,
                    repository=task.repository,
                    language=task.language,
                    base_commit=task.base_commit,
                    structure=tuple(float(value) for value in blocks.structure),
                    localization=tuple(float(value) for value in blocks.localization),
                    prompt_shape=tuple(float(value) for value in blocks.prompt_shape),
                )
            )
        except Exception as error:  # noqa: BLE001
            failures.append(
                {
                    "task_id": task.task_id,
                    "repository": task.repository,
                    "reason_type": type(error).__name__,
                }
            )
        logger.info(
            "tree_tasks_processed=%d tree_tasks_total=%d eligible=%d failures=%d",
            index,
            len(tasks),
            len(rows),
            len(failures),
        )
    return rows, failures


def _coverage_report(
    tasks: Sequence[DatasetTask],
    eligible_task_ids: set[str],
    failures: Sequence[dict[str, str]],
) -> dict[str, Any]:
    retained = eligible_task_ids
    by_language_total = Counter(task.language.casefold() for task in tasks)
    by_language_valid = Counter(
        task.language.casefold() for task in tasks if task.task_id in retained
    )
    repository_sizes = Counter(task.repository for task in tasks)

    def bucket(size: int) -> str:
        if size == 1:
            return "1"
        if size <= 4:
            return "2-4"
        return "5+"

    by_size_total = Counter(bucket(repository_sizes[task.repository]) for task in tasks)
    by_size_valid = Counter(
        bucket(repository_sizes[task.repository]) for task in tasks if task.task_id in retained
    )
    coverage = len(retained) / len(tasks)
    return {
        "protocol": f"{PROTOCOL}-acquisition-v1",
        "dataset_revision": DATASET_REVISION,
        "dataset_parquet_sha256": DATASET_PARQUET_SHA256,
        "tasks": len(tasks),
        "eligible_tasks": len(retained),
        "coverage": coverage,
        "coverage_gate": MIN_DEVELOPMENT_COVERAGE,
        "coverage_passed": coverage >= MIN_DEVELOPMENT_COVERAGE,
        "failures": list(failures),
        "by_language": {
            language: {
                "tasks": total,
                "eligible": by_language_valid[language],
                "coverage": by_language_valid[language] / total,
            }
            for language, total in sorted(by_language_total.items())
        },
        "by_repository_task_count": {
            size: {
                "tasks": total,
                "eligible": by_size_valid[size],
                "coverage": by_size_valid[size] / total,
            }
            for size, total in sorted(by_size_total.items())
        },
        "provider_calls": 0,
        "confirmation_outcomes_accessed": False,
        "deep_swe_outcomes_accessed": False,
        "outcomes_joined": False,
    }


def main() -> None:
    """Run exact development acquisition and emit ephemeral features plus aggregate coverage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-corpus", type=Path, required=True)
    parser.add_argument("--completion-audit", type=Path, required=True)
    parser.add_argument("--dataset-parquet", type=Path, required=True)
    parser.add_argument("--features-out", type=Path, required=True)
    parser.add_argument("--coverage-out", type=Path, required=True)
    args = parser.parse_args()
    manifest = _manifest_tasks(args.development_corpus, args.completion_audit)
    projection = validate_projection(manifest, load_projected_dataset(args.dataset_parquet))
    projected_by_id = {task.task_id: task for task in projection.tasks}
    coverage_tasks = tuple(
        DatasetTask(
            task_id=str(task["task_id"]),
            repository=str(task["repository"]),
            language=str(task["language"]),
            prompt=str(task["prompt"]),
            base_commit=(
                projected_by_id[str(task["task_id"])].base_commit
                if str(task["task_id"]) in projected_by_id
                else "source-rejected"
            ),
            image_name=str(task["image_name"]),
        )
        for task in manifest
    )
    source_ids = set(projected_by_id)
    if len(source_ids) / len(coverage_tasks) < MIN_DEVELOPMENT_COVERAGE:
        coverage = _coverage_report(
            coverage_tasks, source_ids, list(projection.failures)
        )
        coverage["source_projection_only"] = True
        args.coverage_out.write_text(
            json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise RuntimeError("pinned source projection missed the frozen coverage gate")
    rows, tree_failures = acquire_feature_rows(
        projection.tasks, os.environ.get("GITHUB_TOKEN")
    )
    failures = [*projection.failures, *tree_failures]
    coverage = _coverage_report(
        coverage_tasks, {row.task_id for row in rows}, failures
    )
    coverage["source_projection_only"] = False
    args.coverage_out.write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if coverage["coverage_passed"] is not True:
        raise RuntimeError("repository tree acquisition missed the frozen coverage gate")
    args.features_out.write_text(
        json.dumps([asdict(row) for row in rows], separators=(",", ":")),
        encoding="utf-8",
    )
    logger.info("repository tree acquisition passed coverage with %d rows", len(rows))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
