"""Freeze an external graded SWE-rebench cohort without target outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

PROTOCOL = "coding-router-graded-swerebench-cohort-v1"
SOURCE_SHA256 = "0e0bf9355f892ad74ae98d4e1c404f39fd6654a8e351ee3e6ab162e4a64cd3ad"
MINED_SHA256 = "40f9a1b3ace2a592cbdfeac54b57db4f4638b3477df00693207ea45ab88f6caa"
TARGET_INDEX_SHA256 = "b0d25ec0e566c0391e4385a63343b92d5371b67f052e1c9062c9d226d9d18dd1"
TARGET_PROMPTS_SHA256 = "35ad33855f63f147b1861b58b59ad635f8860677b5d0d5e902c421029d78637b"
SEED = 20260801
MIN_TASKS = 900
IMAGE_PATTERN = re.compile(r"^docker\.io/swerebenchv2/[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+$")
TASK_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+$")
LANGUAGE_ALIASES = {"js": "javascript", "ts": "typescript"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _repo(value: str) -> str:
    return value.strip().lower().removesuffix(".git")


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return {str(key): item for key, item in value.items()}


def _target_exclusions(index_path: Path, prompts_path: Path) -> tuple[set[str], set[str]]:
    index = _read_object(index_path)
    prompts = _read_object(prompts_path)
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
    repositories = {_repo(str(row["repository"])) for row in index_rows}
    prompt_hashes = {_digest(_normalize_text(str(row["text"]))) for row in prompt_rows}
    return repositories, prompt_hashes


def _split(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Assign repositories to a deterministic approximately 70/30 split."""
    confirmation_repositories = {
        str(row["repository"])
        for row in rows
        if int(_digest(f"{SEED}|{row['repository']}")[:8], 16) % 10 < 3
    }
    development = [row for row in rows if row["repository"] not in confirmation_repositories]
    confirmation = [row for row in rows if row["repository"] in confirmation_repositories]
    share = len(development) / len(rows)
    if not 0.65 <= share <= 0.75:
        raise ValueError(f"repository split has invalid development share {share:.3f}")
    if {row["repository"] for row in development} & {
        row["repository"] for row in confirmation
    }:
        raise ValueError("repository split overlaps")
    return development, confirmation


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    """Prepare immutable label-free routing views and private verifier task rows."""
    if args.output.exists():
        raise FileExistsError(args.output)
    expected_hashes = {
        args.source: SOURCE_SHA256,
        args.mined: MINED_SHA256,
        args.target_index: TARGET_INDEX_SHA256,
        args.target_prompts: TARGET_PROMPTS_SHA256,
    }
    for path, expected in expected_hashes.items():
        if _sha256(path) != expected:
            raise ValueError(f"input changed: {path}")
    mined = json.loads(args.mined.read_text(encoding="utf-8"))
    if (
        not isinstance(mined, list)
        or len(mined) != 1_000
        or [row.get("rank") for row in mined if isinstance(row, dict)] != list(range(1, 1_001))
    ):
        raise ValueError("mined cohort is not the frozen ranked 1,000")
    mined_ids = [str(row["instance_id"]) for row in mined]
    if len(set(mined_ids)) != 1_000:
        raise ValueError("mined task identities repeat")

    columns = [
        "instance_id",
        "repo",
        "language",
        "image_name",
        "problem_statement",
        "base_commit",
        "test_patch",
        "patch",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "install_config",
    ]
    table = pq.read_table(args.source, columns=columns)
    table = table.filter(pc.is_in(table["instance_id"], value_set=pa.array(mined_ids)))
    source = {str(row["instance_id"]): row for row in table.to_pylist()}
    if set(source) != set(mined_ids):
        raise ValueError(f"source is missing {len(set(mined_ids) - set(source))} mined tasks")

    target_repositories, target_prompt_hashes = _target_exclusions(
        args.target_index,
        args.target_prompts,
    )
    counters: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for rank, task_id in enumerate(mined_ids, start=1):
        raw = source[task_id]
        repository = _repo(str(raw.get("repo") or ""))
        prompt = str(raw.get("problem_statement") or "").strip()
        prompt_hash = _digest(_normalize_text(prompt))
        image = str(raw.get("image_name") or "")
        f2p = list(raw.get("FAIL_TO_PASS") or [])
        install_config = raw.get("install_config")
        if repository in target_repositories:
            counters["excluded_target_repository"] += 1
            continue
        if prompt_hash in target_prompt_hashes:
            counters["excluded_target_prompt"] += 1
            continue
        if (
            not TASK_PATTERN.fullmatch(task_id)
            or not repository
            or not prompt
            or not IMAGE_PATTERN.fullmatch(image)
            or not f2p
            or not isinstance(install_config, dict)
            or not str(raw.get("test_patch") or "").strip()
        ):
            counters["excluded_invalid_verifier_task"] += 1
            continue
        language = LANGUAGE_ALIASES.get(
            str(raw.get("language") or "").strip().lower(),
            str(raw.get("language") or "").strip().lower(),
        )
        public = {
            "rank": rank,
            "task_id": task_id,
            "repository": repository,
            "language": language,
            "prompt": prompt,
            "prompt_sha256": prompt_hash,
            "image_name": image,
            "f2p_total": len(f2p),
        }
        rows.append(public)
        private_rows.append(
            {
                **public,
                "base_commit": str(raw.get("base_commit") or ""),
                "test_patch": str(raw["test_patch"]),
                "gold_patch": str(raw.get("patch") or ""),
                "fail_to_pass": f2p,
                "pass_to_pass": list(raw.get("PASS_TO_PASS") or []),
                "install_config": install_config,
            }
        )
    counters["retained"] = len(rows)
    if len(rows) < MIN_TASKS:
        raise ValueError(f"only {len(rows)} safe tasks survived")
    development, confirmation = _split(rows)
    split_by_id = {
        row["task_id"]: "development" for row in development
    } | {row["task_id"]: "confirmation" for row in confirmation}
    private_rows = [{**row, "split": split_by_id[row["task_id"]]} for row in private_rows]

    args.output.mkdir(parents=True)
    development_path = args.output / "development-tasks.json"
    confirmation_path = args.output / "confirmation-tasks.json"
    private_path = args.output / "verifier-tasks.jsonl"
    _write_json(development_path, {"split": "development", "tasks": development})
    _write_json(confirmation_path, {"split": "confirmation", "tasks": confirmation})
    private_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in private_rows),
        encoding="utf-8",
    )
    manifest = {
        "protocol": PROTOCOL,
        "source_dataset": "nebius/SWE-rebench-V2",
        "source_parquet_sha256": SOURCE_SHA256,
        "mined_selection_sha256": MINED_SHA256,
        "target_index_sha256": TARGET_INDEX_SHA256,
        "target_prompt_view_sha256": TARGET_PROMPTS_SHA256,
        "seed": SEED,
        "source_tasks": 1_000,
        "retained_tasks": len(rows),
        "development_tasks": len(development),
        "confirmation_tasks": len(confirmation),
        "development_share": len(development) / len(rows),
        "development_repositories": len({row["repository"] for row in development}),
        "confirmation_repositories": len({row["repository"] for row in confirmation}),
        "development_confirmation_repository_overlap": 0,
        "source_target_repository_overlap": 0,
        "source_target_normalized_prompt_overlap": 0,
        "selection_used_deepswe_outcomes": False,
        "router_inputs_include_gold_or_tests": False,
        "counters": dict(sorted(counters.items())),
        "language_counts": dict(sorted(Counter(row["language"] for row in rows).items())),
        "f2p_total": {
            "minimum": min(int(row["f2p_total"]) for row in rows),
            "maximum": max(int(row["f2p_total"]) for row in rows),
            "sum": sum(int(row["f2p_total"]) for row in rows),
        },
        "development_tasks_sha256": _sha256(development_path),
        "confirmation_tasks_sha256": _sha256(confirmation_path),
        "verifier_tasks_sha256": _sha256(private_path),
        "deep_swe_outcomes_accessed": False,
        "target_outcomes_used": False,
    }
    _write_json(args.output / "manifest.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mined", type=Path, required=True)
    parser.add_argument("--target-index", type=Path, required=True)
    parser.add_argument("--target-prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
