"""Screen a repeated external coding-agent matrix for honest routing headroom.

The script reads only public official result JSONs from the pinned ASSERT-KTH
artifact release. DeepSWE metadata is used only to remove exact task-id and
normalized-prompt overlap. DeepSWE outcomes are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow.parquet as pq

logger = logging.getLogger("coding-router-assert-kth-oracle")

DATASET = "ASSERT-KTH/agentic-evals-artifacts"
DATASET_REVISION = "5db0c4b69382d160a313d7ceaded915398c63e13"
TASK_DATASET = "princeton-nlp/SWE-bench_Verified"
TASK_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
TASK_PARQUET_PATH = "data/test-00000-of-00001.parquet"
ARMS = (
    "nano-agent-Qwen_Qwen3-32B-temp0",
    "nano-agent-Qwen_Qwen3-32B",
    "nano-agent-agentica-org_DeepSWE-Preview",
    "nano-agent-agentica-org_DeepSWE-Preview__temp0",
    "nano-agent-mistral_devstral-2512",
    "nano-agent-mistral_devstral-2512__temp0",
    "r2e-gym-Qwen_Qwen3-32B",
    "r2e-gym-Qwen_Qwen3-32B__temp0",
    "r2e-gym-agentica-org__DeepSWE-preview",
    "r2e-gym-agentica-org__DeepSWE-preview__temp0",
    "r2e-gym-mistral_devstral-2512",
    "r2e-gym-mistral_devstral-2512__temp0",
)
ATTEMPTS = 10
FIT_ATTEMPTS = 5
ATTEMPT_SPLITS = 400
BOOTSTRAPS_PER_SPLIT = 20
MIN_TASKS = 250
MIN_ARMS = 8
MIN_HEADROOM = 0.10
MIN_LOWER_BOUND = 0.05


@dataclass(frozen=True)
class Task:
    task_id: str
    repository: str
    problem_statement: str


@dataclass(frozen=True)
class Matrix:
    task_ids: tuple[str, ...]
    groups: np.ndarray
    arms: tuple[str, ...]
    rewards: np.ndarray


def _json_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "wmo-coding-router-oracle/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _json(url: str) -> object:
    return cast(object, json.loads(_json_bytes(url)))


def _tree_url(path: str) -> str:
    encoded = urllib.parse.quote(path, safe="/")
    return (
        f"https://huggingface.co/api/datasets/{DATASET}/tree/"
        f"{DATASET_REVISION}/{encoded}?recursive=true&expand=false&limit=1000"
    )


def _resolve_url(dataset: str, revision: str, path: str) -> str:
    encoded = urllib.parse.quote(path, safe="/")
    return f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{encoded}"


def _run_number(path: str) -> int:
    matches = re.findall(r"run_(\d+)", path)
    if not matches:
        raise ValueError(f"cannot infer run number from {path}")
    return int(matches[-1])


def _is_result_item(row: dict[str, Any]) -> bool:
    path = str(row.get("path", ""))
    size = int(row.get("size", 0) or 0)
    return (
        row.get("type") == "file"
        and path.endswith(".json")
        and 10_000 <= size <= 200_000
    )


def _result_paths(arm: str) -> dict[int, str]:
    rows = _json(_tree_url(arm))
    if not isinstance(rows, list):
        raise ValueError(f"Hub tree for {arm} is not a list")
    paths: dict[int, str] = {}
    for untyped in rows:
        if not isinstance(untyped, dict):
            continue
        row = cast(dict[str, Any], untyped)
        path = str(row.get("path", ""))
        if not _is_result_item(row):
            continue
        run = _run_number(path)
        if run in paths:
            raise ValueError(f"{arm} has multiple result JSONs for run {run}")
        paths[run] = path
    expected = set(range(ATTEMPTS))
    if set(paths) != expected:
        raise ValueError(
            f"{arm} result runs differ: expected={sorted(expected)} actual={sorted(paths)}"
        )
    return paths


def _outcomes(payload: dict[str, Any]) -> dict[str, float]:
    submitted_untyped = payload.get("submitted_ids")
    resolved_untyped = payload.get("resolved_ids")
    incomplete_untyped = payload.get("incomplete_ids", [])
    if (
        not isinstance(submitted_untyped, list)
        or not isinstance(resolved_untyped, list)
        or not isinstance(incomplete_untyped, list)
    ):
        raise ValueError("official result is missing submitted_ids or resolved_ids")
    submitted = [str(value) for value in submitted_untyped]
    incomplete = [str(value) for value in incomplete_untyped]
    gradeable = submitted + incomplete
    resolved = {str(value) for value in resolved_untyped}
    if len(gradeable) != len(set(gradeable)):
        raise ValueError("official result has duplicate submitted or incomplete task ids")
    if not resolved <= set(submitted):
        raise ValueError("resolved ids are not a subset of submitted ids")
    total = payload.get("total_instances")
    if not isinstance(total, int) or total != len(gradeable):
        raise ValueError(
            "official result total_instances disagrees with submitted plus incomplete ids"
        )
    return {task_id: float(task_id in resolved) for task_id in gradeable}


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _target_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise ValueError("target metadata must contain a rows list")
    return [
        cast(dict[str, Any], row)
        for row in cast(list[Any], payload["rows"])
        if isinstance(row, dict)
    ]


def _target_feature_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("target_reward_fields_accessed") is not False
        or payload.get("target_cost_fields_accessed") is not False
        or not isinstance(payload.get("rows"), list)
    ):
        raise ValueError("target feature view is not label-free")
    return [
        cast(dict[str, Any], row)
        for row in cast(list[Any], payload["rows"])
        if isinstance(row, dict)
    ]


def _target_texts(rows: list[dict[str, Any]]) -> set[str]:
    texts: set[str] = set()
    for row in rows:
        for key in ("problem_statement", "text", "description", "prompt"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                texts.add(_normalize(value))
    return texts


def _overlaps(task: Task, target_ids: set[str], target_texts: set[str]) -> bool:
    if task.task_id in target_ids:
        return True
    normalized = _normalize(task.problem_statement)
    return bool(
        normalized
        and any(normalized == target or normalized in target for target in target_texts)
    )


def _load_tasks(cache: Path) -> dict[str, Task]:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        cache.write_bytes(
            _json_bytes(_resolve_url(TASK_DATASET, TASK_REVISION, TASK_PARQUET_PATH))
        )
    table = pq.read_table(
        cache,
        columns=["instance_id", "repo", "problem_statement"],
    )
    columns = table.to_pydict()
    tasks: dict[str, Task] = {}
    for task_id, repository, statement in zip(
        columns["instance_id"],
        columns["repo"],
        columns["problem_statement"],
        strict=True,
    ):
        task = Task(str(task_id), str(repository), str(statement))
        tasks[task.task_id] = task
    if len(tasks) != table.num_rows:
        raise ValueError("task metadata has duplicate instance ids")
    return tasks


def _load_matrix(
    cache: Path,
    target_path: Path,
    target_feature_view_path: Path,
) -> tuple[Matrix, dict[str, Any]]:
    tasks = _load_tasks(cache / "swe-bench-verified.parquet")
    target_metadata = _target_rows(target_path)
    target_features = _target_feature_rows(target_feature_view_path)
    target = target_metadata + target_features
    target_ids = {str(row.get("id", row.get("instance_id", ""))) for row in target}
    target_texts = _target_texts(target)
    by_arm: dict[str, list[dict[str, float]]] = {}
    source_hashes: dict[str, str] = {}
    submitted_sets: list[set[str]] = []
    for arm in ARMS:
        runs: list[dict[str, float]] = []
        for run, path in sorted(_result_paths(arm).items()):
            raw = _json_bytes(_resolve_url(DATASET, DATASET_REVISION, path))
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError(f"{path} is not a JSON object")
            outcomes = _outcomes(cast(dict[str, Any], payload))
            runs.append(outcomes)
            submitted_sets.append(set(outcomes))
            source_hashes[f"{arm}/run_{run}"] = hashlib.sha256(raw).hexdigest()
        by_arm[arm] = runs
    common = set.intersection(*submitted_sets)
    missing_metadata = sorted(common - set(tasks))
    if missing_metadata:
        raise ValueError(f"{len(missing_metadata)} common tasks lack metadata")
    overlap_ids = {
        task_id
        for task_id in common
        if _overlaps(tasks[task_id], target_ids, target_texts)
    }
    retained = tuple(sorted(common - overlap_ids))
    groups = np.asarray([tasks[task_id].repository for task_id in retained], dtype=object)
    rewards = np.zeros((len(retained), len(ARMS), ATTEMPTS), dtype=np.float64)
    for arm_index, arm in enumerate(ARMS):
        for run, outcomes in enumerate(by_arm[arm]):
            rewards[:, arm_index, run] = [outcomes[task_id] for task_id in retained]
    matrix = Matrix(retained, groups, ARMS, rewards)
    audit = {
        "source_tasks": len(common),
        "target_task_ids": len(target_ids),
        "target_normalized_texts": len(target_texts),
        "overlap_tasks_removed": len(overlap_ids),
        "overlap_task_ids": sorted(overlap_ids),
        "retained_tasks": len(retained),
        "repositories": len(set(cast(list[str], groups.tolist()))),
        "source_result_sha256": source_hashes,
    }
    return matrix, audit


def _choose(
    fit: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, int]:
    global_means = np.mean(fit[indices], axis=0)
    order = np.lexsort((np.arange(fit.shape[1]), -global_means))
    rank = np.empty(fit.shape[1], dtype=np.int64)
    rank[order] = np.arange(fit.shape[1])
    adjusted = fit[indices] - rank[np.newaxis, :] * 1e-12
    chosen = np.argmax(adjusted, axis=1)
    return chosen, int(order[0])


def _headroom(
    matrix: Matrix,
    *,
    seed: int,
    attempt_splits: int = ATTEMPT_SPLITS,
    bootstraps_per_split: int = BOOTSTRAPS_PER_SPLIT,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    groups = sorted(set(cast(list[str], matrix.groups.tolist())))
    by_group = {
        group: np.flatnonzero(matrix.groups == group)
        for group in groups
    }
    split_rows: list[dict[str, Any]] = []
    bootstrap_headroom: list[float] = []
    for split in range(attempt_splits):
        permutation = rng.permutation(ATTEMPTS)
        fit_runs = np.sort(permutation[:FIT_ATTEMPTS])
        eval_runs = np.sort(permutation[FIT_ATTEMPTS:])
        fit = np.mean(matrix.rewards[:, :, fit_runs], axis=2)
        heldout = np.mean(matrix.rewards[:, :, eval_runs], axis=2)
        all_indices = np.arange(len(matrix.task_ids), dtype=np.int64)
        chosen, static = _choose(fit, all_indices)
        selected = heldout[all_indices, chosen]
        static_reward = heldout[:, static]
        split_rows.append(
            {
                "split": split,
                "fit_runs": fit_runs.tolist(),
                "eval_runs": eval_runs.tolist(),
                "oracle_reward": float(np.mean(selected)),
                "static_reward": float(np.mean(static_reward)),
                "headroom": float(np.mean(selected - static_reward)),
                "fit_selected_static": matrix.arms[static],
            }
        )
        for _ in range(bootstraps_per_split):
            sampled_groups = rng.choice(groups, size=len(groups), replace=True)
            sampled = np.concatenate([by_group[str(group)] for group in sampled_groups])
            bootstrap_chosen, bootstrap_static = _choose(fit, sampled)
            bootstrap_headroom.append(
                float(
                    np.mean(
                        heldout[sampled, bootstrap_chosen]
                        - heldout[sampled, bootstrap_static]
                    )
                )
            )
    headroom = np.asarray([float(row["headroom"]) for row in split_rows])
    bootstraps = np.asarray(bootstrap_headroom)
    full = np.mean(matrix.rewards, axis=2)
    full_indices = np.arange(len(matrix.task_ids), dtype=np.int64)
    naive_chosen, naive_static = _choose(full, full_indices)
    naive_headroom = float(
        np.mean(full[full_indices, naive_chosen] - full[:, naive_static])
    )
    interval = np.quantile(bootstraps, [0.025, 0.5, 0.975])
    static_counts: dict[str, int] = {}
    for row in split_rows:
        arm = str(row["fit_selected_static"])
        static_counts[arm] = static_counts.get(arm, 0) + 1
    return {
        "attempt_splits": attempt_splits,
        "fit_attempts": FIT_ATTEMPTS,
        "heldout_attempts": ATTEMPTS - FIT_ATTEMPTS,
        "repository_bootstraps_per_split": bootstraps_per_split,
        "mean_heldout_oracle_headroom": float(np.mean(headroom)),
        "heldout_oracle_headroom_95ci": [float(value) for value in interval],
        "naive_same_attempt_headroom": naive_headroom,
        "winner_curse_gap": float(naive_headroom - np.mean(headroom)),
        "fit_selected_static_counts": static_counts,
        "split_summaries": split_rows,
    }


def run(
    target_path: Path,
    target_feature_view_path: Path,
    output: Path,
    *,
    seed: int,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    matrix, audit = _load_matrix(
        output / "cache",
        target_path,
        target_feature_view_path,
    )
    oracle = _headroom(matrix, seed=seed)
    interval = cast(list[float], oracle["heldout_oracle_headroom_95ci"])
    gates = {
        "minimum_tasks": len(matrix.task_ids) >= MIN_TASKS,
        "minimum_complete_arms": len(matrix.arms) >= MIN_ARMS,
        "minimum_mean_headroom": (
            float(oracle["mean_heldout_oracle_headroom"]) >= MIN_HEADROOM
        ),
        "minimum_lower_bound": interval[0] > MIN_LOWER_BOUND,
        "dense_binary_matrix": (
            matrix.rewards.shape == (len(matrix.task_ids), len(matrix.arms), ATTEMPTS)
            and bool(np.all((matrix.rewards == 0.0) | (matrix.rewards == 1.0)))
        ),
    }
    report = {
        "protocol": {
            "dataset": DATASET,
            "dataset_revision": DATASET_REVISION,
            "task_dataset": TASK_DATASET,
            "task_revision": TASK_REVISION,
            "arms": list(matrix.arms),
            "attempts_per_arm": ATTEMPTS,
            "target_outcomes_used": False,
            "full_trajectories_downloaded": False,
            "gate": {
                "minimum_tasks": MIN_TASKS,
                "minimum_complete_arms": MIN_ARMS,
                "minimum_mean_headroom": MIN_HEADROOM,
                "minimum_lower_bound": MIN_LOWER_BOUND,
            },
        },
        "overlap_audit": audit,
        "matrix": {
            "tasks": len(matrix.task_ids),
            "repositories": len(set(cast(list[str], matrix.groups.tolist()))),
            "arms": len(matrix.arms),
            "attempts_per_arm": ATTEMPTS,
            "cells": int(matrix.rewards.size),
        },
        "oracle": oracle,
        "gates": gates,
        "passed": all(gates.values()),
    }
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "oracle gate complete tasks=%d arms=%d headroom=%.4f lower=%.4f passed=%s",
        len(matrix.task_ids),
        len(matrix.arms),
        float(oracle["mean_heldout_oracle_headroom"]),
        interval[0],
        report["passed"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-tasks", type=Path, required=True)
    parser.add_argument("--target-feature-view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()
    run(
        args.target_tasks,
        args.target_feature_view,
        args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
