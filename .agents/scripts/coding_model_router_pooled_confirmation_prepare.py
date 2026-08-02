"""Freeze the fresh pooled-uplift SWE-rebench confirmation cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path

import coding_model_router_swerebench_prepare as base

logger = logging.getLogger("coding-router-pooled-confirmation-prepare")

PROTOCOL = "coding-router-pooled-uplift-confirmation-cohort-v1"
SEED = 20_260_801
SOURCE_PARQUET_SHA256 = "7416e352008b35480c82610cf4f5edf160dd269ed6bf382a22bd4c17daed24b9"
POOLED_EXCLUSIONS_SHA256 = "4f05067a44fc69f1e53bb26f79da615a51119740fe9753a3e690434aa9e2b01f"
SWESMITH_TASKS_SHA256 = "9a4b3b749fb2123933335f9c4db41057247f49b37c53a7c075143b44e800aa7c"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(rows: list[dict[str, object]]) -> tuple[set[str], set[str], set[str]]:
    repositories = {base._canonical_repo(str(row["repository"])) for row in rows}
    prompts = {
        base._digest(base._normalize_text(str(row.get("prompt") or row.get("text"))))
        for row in rows
    }
    task_ids = {str(row.get("task_id") or row.get("id")) for row in rows}
    return repositories, prompts, task_ids


def _load_exclusions(
    pooled_path: Path,
    swesmith_path: Path,
) -> tuple[set[str], set[str], set[str], dict[str, object]]:
    """Load both frozen external-development exclusion surfaces."""
    if _sha256(pooled_path) != POOLED_EXCLUSIONS_SHA256:
        raise ValueError("pooled exclusion manifest hash changed")
    if _sha256(swesmith_path) != SWESMITH_TASKS_SHA256:
        raise ValueError("SWE-smith compact task hash changed")
    pooled = json.loads(pooled_path.read_text(encoding="utf-8"))
    pooled_rows = pooled.get("tasks")
    if not isinstance(pooled_rows, list) or len(pooled_rows) != 400:
        raise ValueError("pooled exclusion manifest must contain 400 tasks")
    swesmith_rows: list[dict[str, object]] = []
    for line in swesmith_path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("SWE-smith compact corpus contains a non-object row")
        swesmith_rows.append({str(key): item for key, item in value.items()})
    if len(swesmith_rows) != 1_551:
        raise ValueError("SWE-smith compact task count changed")
    pooled_repositories, pooled_prompts, pooled_ids = _metadata(pooled_rows)
    swesmith_repositories, swesmith_prompts, swesmith_ids = _metadata(swesmith_rows)
    return (
        pooled_repositories | swesmith_repositories,
        pooled_prompts | swesmith_prompts,
        pooled_ids | swesmith_ids,
        {
            "pooled_tasks": len(pooled_rows),
            "pooled_repositories": len(pooled_repositories),
            "swesmith_tasks": len(swesmith_rows),
            "swesmith_repositories": len(swesmith_repositories),
            "pooled_exclusions_sha256": _sha256(pooled_path),
            "swesmith_tasks_sha256": _sha256(swesmith_path),
        },
    )


def prepare(
    parquet: Path,
    target_index: Path,
    target_prompts: Path,
    pooled_exclusions: Path,
    swesmith_tasks: Path,
    output: Path,
) -> None:
    """Select one untouched label-free cohort under the frozen quotas."""
    if output.exists():
        raise FileExistsError(f"confirmation output already exists: {output}")
    if _sha256(parquet) != SOURCE_PARQUET_SHA256:
        raise ValueError("SWE-rebench source Parquet hash changed")
    target_repositories, target_prompt_hashes = base._load_target(
        target_index,
        target_prompts,
    )
    excluded_repositories, excluded_prompts, excluded_ids, exclusion_audit = (
        _load_exclusions(pooled_exclusions, swesmith_tasks)
    )
    source, source_counters = base._load_source(
        parquet,
        target_repositories,
        target_prompt_hashes,
    )
    counters: Counter[str] = Counter(source_counters)
    eligible: list[dict[str, object]] = []
    for row in source:
        if str(row["task_id"]) in excluded_ids:
            counters["excluded_development_task_id"] += 1
            continue
        if str(row["repository"]) in excluded_repositories:
            counters["excluded_development_repository"] += 1
            continue
        if str(row["prompt_sha256"]) in excluded_prompts:
            counters["excluded_development_prompt"] += 1
            continue
        eligible.append(row)
    counters["eligible_after_all_exclusions"] = len(eligible)
    base.SEED = SEED
    confirmation = base._select_split(eligible, "pooled-uplift-confirmation", set())
    repositories = {str(row["repository"]) for row in confirmation}
    prompts = {str(row["prompt_sha256"]) for row in confirmation}
    task_ids = {str(row["task_id"]) for row in confirmation}
    if len(confirmation) != sum(base.SPLIT_QUOTAS.values()):
        raise ValueError("fresh confirmation task count changed")
    if repositories & (target_repositories | excluded_repositories):
        raise ValueError("fresh confirmation repository overlap")
    if prompts & (target_prompt_hashes | excluded_prompts):
        raise ValueError("fresh confirmation prompt overlap")
    if task_ids & excluded_ids:
        raise ValueError("fresh confirmation task-id overlap")
    output.mkdir(parents=True)
    tasks_path = output / "confirmation-tasks.json"
    base._write_json(
        tasks_path,
        {"split": "pooled-uplift-confirmation", "tasks": confirmation},
    )
    manifest = {
        "protocol": PROTOCOL,
        "valid": True,
        "dataset": base.DATASET_ID,
        "dataset_revision": base.DATASET_REVISION,
        "source_parquet_sha256": _sha256(parquet),
        "seed": SEED,
        "split_quotas": base.SPLIT_QUOTAS,
        "max_tasks_per_repository": base.MAX_TASKS_PER_REPO,
        "confirmation_tasks": len(confirmation),
        "confirmation_repositories": len(repositories),
        "confirmation_language_counts": dict(
            sorted(Counter(str(row["language"]) for row in confirmation).items())
        ),
        "exclusion_audit": exclusion_audit,
        "selection_counters": dict(sorted(counters.items())),
        "target_repository_overlap": 0,
        "target_normalized_prompt_overlap": 0,
        "development_repository_overlap": 0,
        "development_normalized_prompt_overlap": 0,
        "development_task_id_overlap": 0,
        "target_reward_fields_accessed": False,
        "target_cost_fields_accessed": False,
        "source_gold_patch_fields_accessed_for_selection": False,
        "source_test_fields_accessed_for_selection": False,
        "confirmation_tasks_sha256": _sha256(tasks_path),
        "target_index_sha256": _sha256(target_index),
        "target_prompt_view_sha256": _sha256(target_prompts),
    }
    base._write_json(output / "manifest.json", manifest)
    logger.info(
        "frozen fresh confirmation tasks=%d repositories=%d task_sha256=%s",
        len(confirmation),
        len(repositories),
        manifest["confirmation_tasks_sha256"],
    )


def main() -> None:
    """Run the fresh confirmation cohort CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--target-index", type=Path, required=True)
    parser.add_argument("--target-prompts", type=Path, required=True)
    parser.add_argument("--pooled-exclusions", type=Path, required=True)
    parser.add_argument("--swesmith-tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(
        args.parquet,
        args.target_index,
        args.target_prompts,
        args.pooled_exclusions,
        args.swesmith_tasks,
        args.output,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
