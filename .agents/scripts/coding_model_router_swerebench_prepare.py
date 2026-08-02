"""Freeze disjoint SWE-rebench V2 development and confirmation cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

logger = logging.getLogger("coding-router-swerebench-prepare")

DATASET_ID = "PrimeIntellect/SWE-rebench-V2-Filtered-Verified"
DATASET_REVISION = "03cc767ee33126b7fc7890ad57047e9dd6914cca"
SEED = 20260731
SPLIT_QUOTAS = {
    "go": 60,
    "javascript": 10,
    "python": 60,
    "rust": 10,
    "typescript": 60,
}
MAX_TASKS_PER_REPO = 3
LANGUAGE_ALIASES = {"js": "javascript", "ts": "typescript"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _canonical_repo(value: str) -> str:
    return value.strip().lower().removesuffix(".git")


def _load_target(target_index: Path, target_prompts: Path) -> tuple[set[str], set[str]]:
    index = json.loads(target_index.read_text(encoding="utf-8"))
    prompts = json.loads(target_prompts.read_text(encoding="utf-8"))
    index_rows = index.get("rows")
    prompt_rows = prompts.get("rows")
    if not isinstance(index_rows, list) or len(index_rows) != 113:
        raise ValueError("target index must contain 113 label-free rows")
    if not isinstance(prompt_rows, list) or len(prompt_rows) != 113:
        raise ValueError("target prompt view must contain 113 label-free rows")
    if prompts.get("target_reward_fields_accessed") is not False:
        raise ValueError("target prompt view accessed rewards")
    if prompts.get("target_cost_fields_accessed") is not False:
        raise ValueError("target prompt view accessed costs")
    repositories = {_canonical_repo(str(row["repository"])) for row in index_rows}
    prompt_hashes = {_digest(_normalize_text(str(row["text"]))) for row in prompt_rows}
    return repositories, prompt_hashes


def _load_source(
    parquet: Path, target_repositories: set[str], target_prompt_hashes: set[str]
) -> tuple[list[dict[str, object]], dict[str, int]]:
    columns = [
        "base_commit",
        "created_at",
        "image_name",
        "instance_id",
        "language",
        "problem_statement",
        "repo",
    ]
    table = pq.read_table(parquet, columns=columns)
    counters: Counter[str] = Counter()
    eligible: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for raw in table.to_pylist():
        counters["source_rows"] += 1
        raw_language = str(raw.get("language") or "").strip().lower()
        language = LANGUAGE_ALIASES.get(raw_language, raw_language)
        if language not in SPLIT_QUOTAS:
            counters["excluded_language"] += 1
            continue
        task_id = str(raw.get("instance_id") or "").strip()
        repo = _canonical_repo(str(raw.get("repo") or ""))
        prompt = str(raw.get("problem_statement") or "").strip()
        if not task_id or not repo or not prompt:
            counters["excluded_missing_label_free_field"] += 1
            continue
        if task_id in seen_ids:
            raise ValueError(f"duplicate source task id: {task_id}")
        seen_ids.add(task_id)
        if repo in target_repositories:
            counters["excluded_target_repository"] += 1
            continue
        prompt_hash = _digest(_normalize_text(prompt))
        if prompt_hash in target_prompt_hashes:
            counters["excluded_target_prompt"] += 1
            continue
        eligible.append(
            {
                "base_commit": str(raw.get("base_commit") or ""),
                "created_at": str(raw.get("created_at") or ""),
                "image_name": str(raw.get("image_name") or ""),
                "language": language,
                "prompt": prompt,
                "prompt_characters": len(prompt),
                "prompt_lines": len(prompt.splitlines()),
                "prompt_sha256": prompt_hash,
                "repository": repo,
                "task_id": task_id,
            }
        )
    counters["eligible_rows"] = len(eligible)
    return eligible, dict(counters)


def _select_split(
    rows: list[dict[str, object]], split: str, forbidden_repositories: set[str]
) -> list[dict[str, object]]:
    by_language_repo: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        repo = str(row["repository"])
        if repo not in forbidden_repositories:
            by_language_repo[str(row["language"])][repo].append(row)
    selected: list[dict[str, object]] = []
    used_repositories: set[str] = set()
    for language, quota in SPLIT_QUOTAS.items():
        repositories = sorted(
            by_language_repo[language],
            key=lambda repo: _digest(f"{SEED}|{split}|{language}|{repo}"),
        )
        language_rows: list[dict[str, object]] = []
        for repo in repositories:
            if repo in used_repositories:
                continue
            candidates = sorted(
                by_language_repo[language][repo],
                key=lambda row: _digest(f"{SEED}|{split}|{row['task_id']}"),
            )
            take = min(MAX_TASKS_PER_REPO, quota - len(language_rows), len(candidates))
            if take:
                language_rows.extend(candidates[:take])
                used_repositories.add(repo)
            if len(language_rows) == quota:
                break
        if len(language_rows) != quota:
            raise ValueError(
                f"cannot fill {split} {language} quota: {len(language_rows)} of {quota}"
            )
        selected.extend(language_rows)
    return sorted(selected, key=lambda row: str(row["task_id"]))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    """Select and audit the two label-free source cohorts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--target-index", type=Path, required=True)
    parser.add_argument("--target-prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    target_repositories, target_prompt_hashes = _load_target(
        args.target_index, args.target_prompts
    )
    eligible, counters = _load_source(
        args.parquet, target_repositories, target_prompt_hashes
    )
    development = _select_split(eligible, "development", set())
    development_repositories = {str(row["repository"]) for row in development}
    confirmation = _select_split(eligible, "confirmation", development_repositories)
    confirmation_repositories = {str(row["repository"]) for row in confirmation}
    if development_repositories & confirmation_repositories:
        raise ValueError("development and confirmation repositories overlap")
    if (development_repositories | confirmation_repositories) & target_repositories:
        raise ValueError("source and target repositories overlap")

    development_path = args.output / "development-tasks.json"
    confirmation_path = args.output / "confirmation-tasks.json"
    _write_json(development_path, {"split": "development", "tasks": development})
    _write_json(confirmation_path, {"split": "confirmation", "tasks": confirmation})
    manifest = {
        "protocol": "coding-router-swerebench-effort-v1",
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "source_parquet_sha256": _sha256(args.parquet),
        "seed": SEED,
        "split_quotas": SPLIT_QUOTAS,
        "max_tasks_per_repository": MAX_TASKS_PER_REPO,
        "development_tasks": len(development),
        "development_repositories": len(development_repositories),
        "confirmation_tasks": len(confirmation),
        "confirmation_repositories": len(confirmation_repositories),
        "development_confirmation_repository_overlap": 0,
        "source_target_repository_overlap": 0,
        "source_target_normalized_prompt_overlap": 0,
        "target_tasks": 113,
        "target_reward_fields_accessed": False,
        "target_cost_fields_accessed": False,
        "source_gold_patch_fields_accessed_for_selection": False,
        "source_test_fields_accessed_for_selection": False,
        "source_selection_counters": counters,
        "development_language_counts": dict(
            sorted(Counter(str(row["language"]) for row in development).items())
        ),
        "confirmation_language_counts": dict(
            sorted(Counter(str(row["language"]) for row in confirmation).items())
        ),
        "development_tasks_sha256": _sha256(development_path),
        "confirmation_tasks_sha256": _sha256(confirmation_path),
        "target_index_sha256": _sha256(args.target_index),
        "target_prompt_view_sha256": _sha256(args.target_prompts),
    }
    _write_json(args.output / "manifest.json", manifest)
    logger.info(
        "frozen SWE-rebench cohorts development=%d confirmation=%d repos=%d/%d",
        len(development),
        len(confirmation),
        len(development_repositories),
        len(confirmation_repositories),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
