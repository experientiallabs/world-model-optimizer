"""Prepare a compact SWE-smith difficulty corpus from frozen trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

logger = logging.getLogger("coding-router-swesmith-prepare")

DATASET_ID = "SWE-bench/SWE-smith-trajectories"
DATASET_REVISION = "08e109b4a59eaeebf80e4675cd125d42e7ac99a4"
SHARDS = (
    (
        "tool-00000-of-00008.parquet",
        "ac76e9efe75978f83d4daec98a55604587dd58b0f1509364893b8778cbf5b487",
    ),
    (
        "tool-00001-of-00008.parquet",
        "2d3a4fcdb89bf4ec4485a1abc41d92645a446d115705cbe882dd5fe148afec3a",
    ),
    (
        "tool-00002-of-00008.parquet",
        "13a95587cfc2fac8e2d6130b25167c02362add1cf969888574328a49dc8ad319",
    ),
    (
        "tool-00003-of-00008.parquet",
        "677831f640fb14e5b7bc02fd2aa729c05cd2068750777eaf877084de03157594",
    ),
    (
        "tool-00004-of-00008.parquet",
        "e41ffbd3f9d1571917ac5b9f4d0108ded4799d9776f597a054203467135a5715",
    ),
    (
        "tool-00005-of-00008.parquet",
        "487a3a6bc8d00c4d2e6ace23da10a82c03349ada8b0f6758d5abf8c7f7fa61d4",
    ),
    (
        "tool-00006-of-00008.parquet",
        "c20d2fc45e67af5d5c6d4c91ca7dfeac69cbaf40442bc2bfe7bac528a99e4b59",
    ),
    (
        "tool-00007-of-00008.parquet",
        "29b684a990ec25345e9c7ade1b8287a6e41fa7fbabd4734f45420ed29280c868",
    ),
)
MESSAGE_COLUMNS = ("messages", "instance_id", "resolved", "model", "traj_id")
DESCRIPTION_PATTERN = re.compile(
    r"<(?:pr|issue)_description>\s*(.*?)\s*</(?:pr|issue)_description>",
    flags=re.DOTALL | re.IGNORECASE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _canonical_repo(value: str) -> str:
    return value.strip().lower().removesuffix(".git")


def _decode_messages(raw: str) -> list[dict[str, Any]]:
    """Decode one or two JSON string layers into a message sequence."""
    payload: object = json.loads(raw)
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError("trajectory messages are not a JSON object sequence")
    return payload


def _user_text(messages: list[dict[str, Any]]) -> str:
    """Extract only the initial user text from a trajectory."""
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            pieces = [
                str(item["text"])
                for item in content
                if isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            ]
            text = "\n".join(pieces).strip()
            if text:
                return text
        raise ValueError("initial user message has no text content")
    raise ValueError("trajectory has no initial user message")


def _problem_statement(user_text: str) -> str:
    """Remove the fixed SWE-smith wrapper when its description tag is present."""
    match = DESCRIPTION_PATTERN.search(user_text)
    if match:
        return match.group(1).strip()
    return user_text.strip()


def _repository(instance_id: str) -> str:
    prefix, separator, _ = instance_id.partition(".")
    if not separator or "__" not in prefix:
        raise ValueError(f"cannot derive repository from {instance_id!r}")
    owner, name = prefix.split("__", 1)
    return _canonical_repo(f"{owner}/{name}")


def _exclusions(
    target_view: Path, development_tasks: Path
) -> tuple[set[str], set[str], dict[str, str]]:
    target = json.loads(target_view.read_text(encoding="utf-8"))
    development = json.loads(development_tasks.read_text(encoding="utf-8"))
    target_rows = target.get("rows")
    development_rows = development.get("tasks")
    if not isinstance(target_rows, list) or len(target_rows) != 113:
        raise ValueError("target view must contain 113 label-free rows")
    if target.get("target_reward_fields_accessed") is not False:
        raise ValueError("target view accessed reward fields")
    if target.get("target_cost_fields_accessed") is not False:
        raise ValueError("target view accessed cost fields")
    if not isinstance(development_rows, list) or len(development_rows) < 190:
        raise ValueError("development task manifest is incomplete")
    repositories = {
        _canonical_repo(str(row["repository"]))
        for row in [*target_rows, *development_rows]
    }
    prompt_hashes = {
        hashlib.sha256(
            _normalized_text(str(row.get("text") or row.get("prompt"))).encode()
        ).hexdigest()
        for row in [*target_rows, *development_rows]
    }
    return repositories, prompt_hashes, {
        "target_view_sha256": _sha256(target_view),
        "development_tasks_sha256": _sha256(development_tasks),
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare(
    shard_dir: Path,
    target_view: Path,
    development_tasks: Path,
    output: Path,
) -> None:
    """Validate frozen shards and write the compact task-level source corpus."""
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    excluded_repositories, excluded_prompts, input_hashes = _exclusions(
        target_view, development_tasks
    )
    observed_shards: dict[str, str] = {}
    aggregates: dict[str, dict[str, Any]] = {}
    ambiguous_task_ids: set[str] = set()
    seen_trajectories: set[str] = set()
    counters: Counter[str] = Counter()
    for name, expected_hash in SHARDS:
        path = shard_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen shard: {path}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"frozen shard hash changed: {name}")
        observed_shards[name] = actual_hash
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=64, columns=list(MESSAGE_COLUMNS)):
            for raw in batch.to_pylist():
                counters["trajectory_rows"] += 1
                task_id = str(raw.get("instance_id") or "").strip()
                trajectory_id = str(raw.get("traj_id") or "").strip()
                model = str(raw.get("model") or "").strip()
                resolved = raw.get("resolved")
                messages_raw = raw.get("messages")
                if (
                    not task_id
                    or not trajectory_id
                    or not model
                    or not isinstance(resolved, bool)
                    or not isinstance(messages_raw, str)
                ):
                    raise ValueError("trajectory row lacks a required frozen field")
                if trajectory_id in seen_trajectories:
                    raise ValueError(f"duplicate trajectory id: {trajectory_id}")
                seen_trajectories.add(trajectory_id)
                prompt = _problem_statement(_user_text(_decode_messages(messages_raw)))
                prompt_hash = hashlib.sha256(_normalized_text(prompt).encode()).hexdigest()
                repository = _repository(task_id)
                row = aggregates.setdefault(
                    task_id,
                    {
                        "task_id": task_id,
                        "repository": repository,
                        "prompt": prompt,
                        "prompt_sha256": prompt_hash,
                        "successes": 0,
                        "trials": 0,
                        "model_trials": Counter(),
                        "model_successes": Counter(),
                    },
                )
                if row["repository"] != repository:
                    raise ValueError(f"task repository changed across trajectories: {task_id}")
                if row["prompt_sha256"] != prompt_hash:
                    ambiguous_task_ids.add(task_id)
                row["trials"] += 1
                row["successes"] += int(resolved)
                row["model_trials"][model] += 1
                row["model_successes"][model] += int(resolved)

    retained: list[dict[str, Any]] = []
    for task_id in sorted(aggregates):
        row = aggregates[task_id]
        if task_id in ambiguous_task_ids:
            counters["excluded_ambiguous_problem_statement"] += 1
            continue
        if row["repository"] in excluded_repositories:
            counters["excluded_repository_overlap"] += 1
            continue
        if row["prompt_sha256"] in excluded_prompts:
            counters["excluded_prompt_overlap"] += 1
            continue
        if row["trials"] < 3:
            counters["excluded_fewer_than_three_trajectories"] += 1
            continue
        retained.append(
            {
                "task_id": row["task_id"],
                "repository": row["repository"],
                "prompt": row["prompt"],
                "prompt_sha256": row["prompt_sha256"],
                "successes": row["successes"],
                "trials": row["trials"],
                "difficulty_target": (row["successes"] + 1) / (row["trials"] + 2),
                "model_trials": dict(sorted(row["model_trials"].items())),
                "model_successes": dict(sorted(row["model_successes"].items())),
            }
        )
    if len(retained) < 1_000:
        raise ValueError(f"too few retained SWE-smith tasks: {len(retained)}")
    tasks_path = output / "tasks.jsonl"
    tasks_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in retained),
        encoding="utf-8",
    )
    repository_count = len({str(row["repository"]) for row in retained})
    prompt_count = len({str(row["prompt_sha256"]) for row in retained})
    if prompt_count != len(retained):
        raise ValueError("retained SWE-smith prompts are not unique")
    manifest = {
        "protocol": "coding-router-swesmith-difficulty-corpus-v1",
        "valid": True,
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "split": "tool",
        "shard_sha256": observed_shards,
        "input_sha256": input_hashes,
        "trajectory_rows": counters["trajectory_rows"],
        "unique_trajectories": len(seen_trajectories),
        "source_tasks": len(aggregates),
        "ambiguous_problem_statement_tasks": len(ambiguous_task_ids),
        "retained_tasks": len(retained),
        "retained_repositories": repository_count,
        "retained_prompt_hashes": prompt_count,
        "selection_counters": dict(sorted(counters.items())),
        "minimum_trajectories_per_task": 3,
        "difficulty_target": "beta-binomial-posterior-mean-alpha1-beta1",
        "messages_field_read": True,
        "initial_user_text_used": True,
        "later_trajectory_turns_used": False,
        "patch_field_read": False,
        "target_reward_fields_accessed": False,
        "target_cost_fields_accessed": False,
        "fitted_model_persisted": False,
        "tasks_sha256": _sha256(tasks_path),
    }
    _write_json(output / "manifest.json", manifest)
    logger.info(
        "prepared SWE-smith tasks=%d repositories=%d trajectories=%d tasks_sha256=%s",
        len(retained),
        repository_count,
        len(seen_trajectories),
        manifest["tasks_sha256"],
    )


def main() -> None:
    """Run the compact SWE-smith corpus preparation CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--target-view", type=Path, required=True)
    parser.add_argument("--development-tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.shard_dir, args.target_view, args.development_tasks, args.output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
