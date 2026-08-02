"""Freeze BeyondSWE task text and released Codex trace-burden labels.

This script is intended for remote experiment compute. DeepSWE metadata may be
read only to remove exact task-id and normalized-prompt overlap. DeepSWE outcomes
are never read. A second metadata file can similarly reserve an external
validation corpus such as Open-SWE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger("coding-router-beyondswe-prepare")

EXPECTED_AGENT = "codex"
EXPECTED_MODEL = "gpt-5.4"
EXPECTED_PROVIDER = "openai"


def _normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _read_jsonl(path: Path, *, key: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            task_id = raw.get(key)
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"{path}:{line_number} has no {key}")
            if task_id in rows:
                raise ValueError(f"{path} contains duplicate {key}={task_id}")
            rows[task_id] = {str(name): value for name, value in raw.items()}
    return rows


def _metadata(path: Path | None) -> tuple[set[str], set[str]]:
    if path is None:
        return set(), set()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        candidates = raw.get("rows", raw.get("tasks"))
    else:
        candidates = raw
    if not isinstance(candidates, list):
        raise ValueError(f"{path} has no task rows")
    ids: set[str] = set()
    prompts: set[str] = set()
    for item in candidates:
        if not isinstance(item, dict):
            continue
        task_id = item.get("task_id", item.get("id", item.get("instance_id")))
        text = item.get(
            "text",
            item.get("prompt", item.get("problem_statement")),
        )
        if isinstance(task_id, str) and task_id:
            ids.add(task_id)
        if isinstance(text, str) and text.strip():
            prompts.add(_normalize_text(text))
    return ids, prompts


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _integer(value: object, *, name: str) -> int:
    result = _number(value, name=name)
    if result < 0 or not result.is_integer():
        raise ValueError(f"{name} is not a nonnegative integer")
    return int(result)


def _optional_number(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _number(value, name=name)


def _joined_row(task: dict[str, Any], trace: dict[str, Any]) -> dict[str, object] | None:
    if trace.get("has_exception") is True:
        return None
    if trace.get("trajectory_available") is not True:
        raise ValueError(f"{trace.get('task_name')} has no trajectory")
    agent_info = trace.get("agent_info")
    if not isinstance(agent_info, dict):
        raise ValueError(f"{trace.get('task_name')} has no agent_info")
    model_info = agent_info.get("model_info")
    if not isinstance(model_info, dict):
        raise ValueError(f"{trace.get('task_name')} has no model_info")
    provenance = (
        str(agent_info.get("name")),
        str(model_info.get("name")),
        str(model_info.get("provider")),
    )
    if provenance != (EXPECTED_AGENT, EXPECTED_MODEL, EXPECTED_PROVIDER):
        raise ValueError(f"unexpected trace provenance {provenance}")
    reward = _number(trace.get("reward"), name="reward")
    if not 0.0 <= reward <= 1.0:
        raise ValueError(f"reward outside [0, 1]: {reward}")
    result = trace.get("agent_result")
    trajectory = trace.get("trajectory")
    if not isinstance(result, dict) or not isinstance(trajectory, dict):
        raise ValueError(f"{trace.get('task_name')} has incomplete trace metrics")
    final_metrics = trajectory.get("final_metrics")
    if not isinstance(final_metrics, dict):
        raise ValueError(f"{trace.get('task_name')} has no final_metrics")
    task_id = str(task["instance_id"])
    prompt = str(task.get("problem_statement") or task.get("task") or "")
    repo = str(task.get("repo") or "")
    language = str(task.get("language") or "").casefold()
    task_type = str(trace.get("task_type") or task.get("task") or "")
    if not prompt or not repo or not language or not task_type:
        raise ValueError(f"{task_id} has incomplete task metadata")
    steps = _integer(
        final_metrics.get("total_steps", trace.get("trajectory_step_count")),
        name="total_steps",
    )
    row = {
        "task_id": task_id,
        "text": prompt,
        "repo": repo,
        "language": language,
        "task_type": task_type,
        "dataset_id": str(task.get("dataset_id") or ""),
        "reward": reward,
        "failed": float(reward < 1.0),
        "cost_usd": _optional_number(result.get("cost_usd"), name="cost_usd"),
        "input_tokens": _integer(result.get("n_input_tokens"), name="n_input_tokens"),
        "output_tokens": _integer(
            result.get("n_output_tokens"),
            name="n_output_tokens",
        ),
        "cached_tokens": _integer(
            result.get("n_cache_tokens"),
            name="n_cache_tokens",
        ),
        "trajectory_steps": steps,
        "total_prompt_tokens": _integer(
            final_metrics.get("total_prompt_tokens"),
            name="total_prompt_tokens",
        ),
        "total_completion_tokens": _integer(
            final_metrics.get("total_completion_tokens"),
            name="total_completion_tokens",
        ),
        "total_cached_tokens": _integer(
            final_metrics.get("total_cached_tokens"),
            name="total_cached_tokens",
        ),
        "trace_sha256": str(trace.get("trajectory_sha256") or ""),
    }
    if isinstance(row["cost_usd"], (int, float)) and row["cost_usd"] < 0:
        raise ValueError(f"{task_id} has negative cost")
    return row


def prepare(
    tasks_path: Path,
    traces_path: Path,
    output: Path,
    *,
    target_metadata: Path | None,
    validation_metadata: Path | None,
    task_source_commit: str,
    trace_source_commit: str,
) -> dict[str, object]:
    tasks = _read_jsonl(tasks_path, key="instance_id")
    traces = _read_jsonl(traces_path, key="task_name")
    if tasks.keys() != traces.keys():
        raise ValueError(
            "BeyondSWE task and trajectory ids differ: "
            f"task_only={len(tasks.keys() - traces.keys())} "
            f"trace_only={len(traces.keys() - tasks.keys())}"
        )
    target_ids, target_prompts = _metadata(target_metadata)
    validation_ids, validation_prompts = _metadata(validation_metadata)
    selected: list[dict[str, object]] = []
    exclusions = Counter()
    for task_id in sorted(tasks):
        task = tasks[task_id]
        prompt = str(task.get("problem_statement") or task.get("task") or "")
        normalized = _normalize_text(prompt)
        if task_id in target_ids:
            exclusions["target_identity"] += 1
            continue
        if normalized and normalized in target_prompts:
            exclusions["target_prompt"] += 1
            continue
        if task_id in validation_ids:
            exclusions["validation_identity"] += 1
            continue
        if normalized and normalized in validation_prompts:
            exclusions["validation_prompt"] += 1
            continue
        row = _joined_row(task, traces[task_id])
        if row is None:
            exclusions["trace_exception"] += 1
            continue
        selected.append(row)
    if len(selected) < 100:
        raise ValueError(f"only {len(selected)} clean BeyondSWE traces remain")
    output.mkdir(parents=True, exist_ok=True)
    source_path = output / "source.json"
    source = {
        "protocol": "beyondswe-codex-trace-source-v1",
        "rows": selected,
    }
    _write_json(source_path, source)
    measured_costs = [
        _number(row["cost_usd"], name="selected cost")
        for row in selected
        if row["cost_usd"] is not None
    ]
    manifest = {
        "protocol": "beyondswe-codex-trace-source-manifest-v1",
        "task_source": "AweAI-Team/BeyondSWE",
        "task_source_commit": task_source_commit,
        "task_source_sha256": _sha256_path(tasks_path),
        "trace_source": "AweAI-Team/BeyondSWE/trajectories/GPT-5.4-XHigh/Codex",
        "trace_source_commit": trace_source_commit,
        "trace_source_sha256": _sha256_path(traces_path),
        "joined_tasks": len(tasks),
        "selected_tasks": len(selected),
        "excluded": dict(sorted(exclusions.items())),
        "task_type_mix": dict(Counter(str(row["task_type"]) for row in selected)),
        "language_mix": dict(Counter(str(row["language"]) for row in selected)),
        "mean_reward": sum(
            _number(row["reward"], name="selected reward") for row in selected
        )
        / len(selected),
        "measured_cost_tasks": len(measured_costs),
        "mean_cost_usd": (
            sum(measured_costs) / len(measured_costs) if measured_costs else None
        ),
        "target_metadata_path_used": target_metadata is not None,
        "validation_metadata_path_used": validation_metadata is not None,
        "target_outcomes_used": False,
        "target_overlap_retained": 0,
        "validation_overlap_retained": 0,
        "source_sha256": _sha256_path(source_path),
        "trace_provenance": {
            "agent": EXPECTED_AGENT,
            "model": EXPECTED_MODEL,
            "provider": EXPECTED_PROVIDER,
        },
    }
    _write_json(output / "manifest.json", manifest)
    logger.info(
        "BeyondSWE source frozen joined=%d selected=%d exclusions=%s",
        len(tasks),
        len(selected),
        dict(sorted(exclusions.items())),
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-metadata", type=Path)
    parser.add_argument("--validation-metadata", type=Path)
    parser.add_argument("--task-source-commit", required=True)
    parser.add_argument("--trace-source-commit", required=True)
    args = parser.parse_args()
    prepare(
        args.tasks,
        args.traces,
        args.output,
        target_metadata=args.target_metadata,
        validation_metadata=args.validation_metadata,
        task_source_commit=args.task_source_commit,
        trace_source_commit=args.trace_source_commit,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
