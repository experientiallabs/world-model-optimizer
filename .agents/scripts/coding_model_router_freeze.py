"""Freeze the execution-scored coding-model router protocol inputs."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import logging
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd
from harbor.models.job.config import DatasetConfig
from harbor.models.trial.config import TaskConfig
from pydantic import BaseModel, ConfigDict

from wmo.core.files import write_text_atomic

logger = logging.getLogger("coding-model-router-freeze")

EXPERIMENT_ID = "coding-router-20260728"
SOURCE_COMMIT = "c3267f1f9d5f35a14ad45b6a94b7b21d3b11c958"
SPLIT_SEEDS = (0, 1, 2, 3, 4)
FIT_FRACTION = 0.7
SWE_DATASET = "princeton-nlp/SWE-bench_Verified"
SWE_DATASET_COMMIT = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
SWE_HARBOR_DATASET = "swebench-verified"
SWE_HARBOR_VERSION = "1.0"
SWE_PARQUET_URL = (
    "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/"
    "resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet"
)
SWE_PARQUET_SHA256 = "a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd"


class Arm(BaseModel):
    """One frozen model and reasoning-effort candidate."""

    model_config = ConfigDict(frozen=True)

    name: str
    kind: str
    model: str
    model_type: str
    effort: str | None
    tier: str
    input_per_mtok: float
    cached_input_per_mtok: float
    output_per_mtok: float
    cache_write_per_mtok: float | None = None


ARMS = (
    Arm(
        name="oai-sol-max",
        kind="openai_responses",
        model="gpt-5.6-sol",
        model_type="gpt-5.6-sol",
        effort="max",
        tier="frontier",
        input_per_mtok=5.0,
        cached_input_per_mtok=0.5,
        output_per_mtok=30.0,
        cache_write_per_mtok=6.25,
    ),
    Arm(
        name="oai-sol-high",
        kind="openai_responses",
        model="gpt-5.6-sol",
        model_type="gpt-5.6-sol",
        effort="high",
        tier="frontier",
        input_per_mtok=5.0,
        cached_input_per_mtok=0.5,
        output_per_mtok=30.0,
        cache_write_per_mtok=6.25,
    ),
    Arm(
        name="oai-terra-max",
        kind="openai_responses",
        model="gpt-5.6-terra",
        model_type="gpt-5.6-terra",
        effort="max",
        tier="frontier",
        input_per_mtok=2.5,
        cached_input_per_mtok=0.25,
        output_per_mtok=15.0,
        cache_write_per_mtok=3.125,
    ),
    Arm(
        name="oai-terra-high",
        kind="openai_responses",
        model="gpt-5.6-terra",
        model_type="gpt-5.6-terra",
        effort="high",
        tier="frontier",
        input_per_mtok=2.5,
        cached_input_per_mtok=0.25,
        output_per_mtok=15.0,
        cache_write_per_mtok=3.125,
    ),
    Arm(
        name="oai-luna-high",
        kind="openai_responses",
        model="gpt-5.6-luna",
        model_type="gpt-5.6-luna",
        effort="high",
        tier="open",
        input_per_mtok=1.0,
        cached_input_per_mtok=0.1,
        output_per_mtok=6.0,
        cache_write_per_mtok=1.25,
    ),
    Arm(
        name="oai-gpt55-high",
        kind="openai_responses",
        model="gpt-5.5-2026-04-23",
        model_type="gpt-5.5",
        effort="high",
        tier="frontier",
        input_per_mtok=5.0,
        cached_input_per_mtok=0.5,
        output_per_mtok=30.0,
    ),
    Arm(
        name="oai-codex53-high",
        kind="openai_responses",
        model="gpt-5.3-codex",
        model_type="gpt-5.3-codex",
        effort="high",
        tier="frontier",
        input_per_mtok=1.75,
        cached_input_per_mtok=0.175,
        output_per_mtok=14.0,
    ),
    Arm(
        name="oai-mini54-high",
        kind="openai_responses",
        model="gpt-5.4-mini-2026-03-17",
        model_type="gpt-5.4-mini",
        effort="high",
        tier="open",
        input_per_mtok=0.75,
        cached_input_per_mtok=0.075,
        output_per_mtok=4.5,
    ),
    Arm(
        name="ant-fable-max",
        kind="anthropic",
        model="claude-fable-5",
        model_type="claude-fable-5",
        effort="max",
        tier="frontier",
        input_per_mtok=10.0,
        cached_input_per_mtok=1.0,
        output_per_mtok=50.0,
        cache_write_per_mtok=12.5,
    ),
    Arm(
        name="ant-opus5-max",
        kind="anthropic",
        model="claude-opus-5",
        model_type="claude-opus-5",
        effort="max",
        tier="frontier",
        input_per_mtok=5.0,
        cached_input_per_mtok=0.5,
        output_per_mtok=25.0,
        cache_write_per_mtok=6.25,
    ),
    Arm(
        name="ant-opus5-high",
        kind="anthropic",
        model="claude-opus-5",
        model_type="claude-opus-5",
        effort="high",
        tier="frontier",
        input_per_mtok=5.0,
        cached_input_per_mtok=0.5,
        output_per_mtok=25.0,
        cache_write_per_mtok=6.25,
    ),
    Arm(
        name="ant-sonnet5-high",
        kind="anthropic",
        model="claude-sonnet-5",
        model_type="claude-sonnet-5",
        effort="high",
        tier="frontier",
        input_per_mtok=3.0,
        cached_input_per_mtok=0.3,
        output_per_mtok=15.0,
        cache_write_per_mtok=3.75,
    ),
    Arm(
        name="ant-sonnet5-low",
        kind="anthropic",
        model="claude-sonnet-5",
        model_type="claude-sonnet-5",
        effort="low",
        tier="open",
        input_per_mtok=3.0,
        cached_input_per_mtok=0.3,
        output_per_mtok=15.0,
        cache_write_per_mtok=3.75,
    ),
    Arm(
        name="ant-haiku45",
        kind="anthropic",
        model="claude-haiku-4-5-20251001",
        model_type="claude-haiku-4-5",
        effort=None,
        tier="open",
        input_per_mtok=1.0,
        cached_input_per_mtok=0.1,
        output_per_mtok=5.0,
        cache_write_per_mtok=1.25,
    ),
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _fetch(url: str) -> bytes:
    if not url.startswith("https://huggingface.co/"):
        raise ValueError(f"refusing non-Hugging-Face URL: {url}")
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
        return response.read()


def _terminal_family(task_id: str) -> str:
    parts = task_id.split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else task_id


def _group_split(
    rows: list[tuple[str, str]],
    *,
    benchmark: str,
    seed: int,
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for task_id, group_id in rows:
        grouped[group_id].append(task_id)
    ordered = sorted(
        grouped,
        key=lambda group_id: (
            hashlib.sha256(f"{seed}:{benchmark}:{group_id}".encode()).digest(),
            group_id,
        ),
    )
    target = round(len(rows) * FIT_FRACTION)
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for group_id in ordered:
        size = len(grouped[group_id])
        additions = {
            count + size: selected + (group_id,)
            for count, selected in reachable.items()
            if count + size < len(rows)
        }
        for count, selected in additions.items():
            reachable.setdefault(count, selected)
    best_count = min(
        (count for count in reachable if count > 0),
        key=lambda count: (abs(count - target), count > target, count),
    )
    fit_groups = set(reachable[best_count])
    fit = [task_id for task_id, group_id in rows if group_id in fit_groups]
    heldout = [task_id for task_id, group_id in rows if group_id not in fit_groups]
    if not fit or not heldout:
        raise ValueError(f"{benchmark} seed {seed} produced an empty split")
    return {"fit": fit, "heldout": heldout}


async def _registry_tasks(
    name: str,
    version: str,
    cache_dir: Path,
) -> list[TaskConfig]:
    return await DatasetConfig(
        name=name,
        version=version,
        download_dir=cache_dir,
    ).get_task_configs()


async def _terminal_manifest(cache_dir: Path) -> dict[str, object]:
    configs = await _registry_tasks("terminal-bench", "2.0", cache_dir)
    tasks = [
        {
            "task_id": config.get_task_id().get_name(),
            "group": _terminal_family(config.get_task_id().get_name()),
            "git_url": config.git_url,
            "git_commit_id": config.git_commit_id,
            "path": str(config.path),
            "source": config.source,
        }
        for config in configs
    ]
    return {
        "benchmark": "terminal-bench-2",
        "registry_name": "terminal-bench",
        "registry_version": "2.0",
        "count": len(tasks),
        "tasks": tasks,
    }


def _swe_manifest(
    parquet: bytes,
    harbor_tasks: dict[str, TaskConfig],
) -> dict[str, object]:
    observed = _sha256(parquet)
    if observed != SWE_PARQUET_SHA256:
        raise ValueError(
            f"SWE-bench parquet changed: expected {SWE_PARQUET_SHA256}, observed {observed}"
        )
    frame = pd.read_parquet(io.BytesIO(parquet))
    columns = ["instance_id", "repo", "base_commit", "created_at", "difficulty"]
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"SWE-bench parquet is missing columns: {sorted(missing)}")
    instance_ids = {str(value) for value in frame["instance_id"].tolist()}
    if instance_ids != set(harbor_tasks):
        only_hf = sorted(instance_ids - set(harbor_tasks))
        only_harbor = sorted(set(harbor_tasks) - instance_ids)
        raise ValueError(
            "SWE-bench identities differ between Hugging Face and Harbor: "
            f"HF-only={only_hf[:5]}, Harbor-only={only_harbor[:5]}"
        )
    tasks = []
    for instance_id, repo, base_commit, created_at, difficulty in frame[columns].itertuples(
        index=False, name=None
    ):
        harbor = harbor_tasks[str(instance_id)]
        tasks.append(
            {
                "task_id": str(instance_id),
                "group": str(repo),
                "repo": str(repo),
                "base_commit": str(base_commit),
                "created_at": None if pd.isna(created_at) else str(created_at),
                "difficulty": None if pd.isna(difficulty) else str(difficulty),
                "harbor_git_url": harbor.git_url,
                "harbor_git_commit_id": harbor.git_commit_id,
                "harbor_path": str(harbor.path),
            }
        )
    return {
        "benchmark": "swe-bench-verified",
        "dataset": SWE_DATASET,
        "dataset_commit": SWE_DATASET_COMMIT,
        "parquet_sha256": SWE_PARQUET_SHA256,
        "harbor_registry_name": SWE_HARBOR_DATASET,
        "harbor_registry_version": SWE_HARBOR_VERSION,
        "count": len(tasks),
        "tasks": tasks,
    }


def _pool_toml() -> str:
    lines = [
        "# Generated by .agents/scripts/coding_model_router_freeze.py",
        f"# source_commit = {SOURCE_COMMIT}",
        "",
    ]
    for arm in ARMS:
        lines.extend(
            [
                "[[model]]",
                f'name = "{arm.name}"',
                f'kind = "{arm.kind}"',
                f'model = "{arm.model}"',
                f'model_type = "{arm.model_type}"',
                f'tier = "{arm.tier}"',
                f"input_per_mtok = {arm.input_per_mtok}",
                f"cached_input_per_mtok = {arm.cached_input_per_mtok}",
                f"output_per_mtok = {arm.output_per_mtok}",
            ]
        )
        if arm.cache_write_per_mtok is not None:
            lines.append(f"cache_write_per_mtok = {arm.cache_write_per_mtok}")
        if arm.effort is not None:
            lines.append(f'reasoning_effort = "{arm.effort}"')
        lines.append("")
    return "\n".join(lines)


def _task_rows(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError("manifest tasks must be a list of objects")
    rows: list[dict[str, object]] = []
    for row in value:
        if not isinstance(row, dict):
            raise ValueError("manifest tasks must be a list of objects")
        rows.append({str(key): item for key, item in row.items()})
    return rows


def freeze(out_dir: Path, cache_dir: Path) -> None:
    """Write manifests, grouped splits, model pool, and immutable digests."""
    tasks_dir = out_dir / "tasks"
    splits_dir = out_dir / "splits"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    terminal = asyncio.run(_terminal_manifest(cache_dir))
    swe_configs = asyncio.run(_registry_tasks(SWE_HARBOR_DATASET, SWE_HARBOR_VERSION, cache_dir))
    swe_by_id = {config.get_task_id().get_name(): config for config in swe_configs}
    parquet = _fetch(SWE_PARQUET_URL)
    swe = _swe_manifest(parquet, swe_by_id)

    terminal_path = tasks_dir / "terminal-bench-2.json"
    swe_path = tasks_dir / "swe-bench-verified.json"
    _write_json(terminal_path, terminal)
    _write_json(swe_path, swe)
    benchmark_rows = {
        "terminal-bench-2": [
            (str(row["task_id"]), str(row["group"])) for row in _task_rows(terminal["tasks"])
        ],
        "swe-bench-verified": [
            (str(row["task_id"]), str(row["group"])) for row in _task_rows(swe["tasks"])
        ],
    }
    split_digests: dict[str, str] = {}
    for seed in SPLIT_SEEDS:
        split = {
            "version": "group-subset-sha256-70-30-v1",
            "seed": seed,
            "fit_fraction": FIT_FRACTION,
            **{
                benchmark: _group_split(rows, benchmark=benchmark, seed=seed)
                for benchmark, rows in benchmark_rows.items()
            },
        }
        path = splits_dir / f"seed-{seed}.json"
        _write_json(path, split)
        split_digests[path.name] = _sha256(path.read_bytes())

    pool_path = out_dir / "pool.toml"
    write_text_atomic(pool_path, _pool_toml())
    ledger_path = out_dir / "spend-ledger.jsonl"
    if not ledger_path.exists():
        write_text_atomic(ledger_path, "")
    _write_json(
        out_dir / "freeze-summary.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "source_commit": SOURCE_COMMIT,
            "spend_ceiling_usd": None,
            "terminal_manifest_sha256": _sha256(terminal_path.read_bytes()),
            "swe_manifest_sha256": _sha256(swe_path.read_bytes()),
            "swe_parquet_sha256": _sha256(parquet),
            "pool_sha256": _sha256(pool_path.read_bytes()),
            "split_sha256": split_digests,
            "model_arms": len(ARMS),
            "terminal_tasks": terminal["count"],
            "swe_tasks": swe["count"],
        },
    )
    logger.info(
        "frozen %s: %s Terminal-Bench 2 tasks, %s SWE-bench Verified tasks, %d arms",
        EXPERIMENT_ID,
        terminal["count"],
        swe["count"],
        len(ARMS),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(".wmo/experiments") / EXPERIMENT_ID,
    )
    parser.add_argument(
        "--harbor-cache",
        type=Path,
        default=Path("/private/tmp/wmo-coding-router-harbor/tasks"),
    )
    args = parser.parse_args()
    freeze(args.out_dir, args.harbor_cache)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
