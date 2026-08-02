"""Unit tests for the BeyondSWE external trace source freezer."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _module() -> ModuleType:
    path = Path(__file__).with_name("coding_model_router_beyondswe_prepare.py")
    spec = importlib.util.spec_from_file_location("beyondswe_prepare", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _module()


def _task(task_id: str, text: str) -> dict[str, object]:
    return {
        "instance_id": task_id,
        "problem_statement": text,
        "repo": f"org/{task_id}",
        "language": "Python",
        "task": "DomainFix",
        "dataset_id": "source",
    }


def _trace(task_id: str, *, exception: bool = False) -> dict[str, object]:
    return {
        "task_name": task_id,
        "task_type": "DomainFix",
        "reward": None if exception else 0.5,
        "has_exception": exception,
        "trajectory_available": not exception,
        "agent_info": {
            "name": "codex",
            "model_info": {"name": "gpt-5.4", "provider": "openai"},
        },
        "agent_result": {
            "cost_usd": 1.25,
            "n_input_tokens": 10,
            "n_output_tokens": 5,
            "n_cache_tokens": 3,
        },
        "trajectory": {
            "final_metrics": {
                "total_steps": 7,
                "total_prompt_tokens": 100,
                "total_completion_tokens": 20,
                "total_cached_tokens": 60,
            }
        },
        "trajectory_sha256": "abc",
    }


def test_joined_row_preserves_dense_trace_labels() -> None:
    row = module._joined_row(_task("task-1", "Repair it"), _trace("task-1"))
    assert row is not None
    assert row["reward"] == 0.5
    assert row["failed"] == 1.0
    assert row["trajectory_steps"] == 7
    assert row["total_prompt_tokens"] == 100
    assert row["language"] == "python"


def test_exception_is_excluded_before_numeric_validation() -> None:
    row = module._joined_row(
        _task("task-1", "Repair it"),
        _trace("task-1", exception=True),
    )
    assert row is None


def test_doc_to_repo_uses_task_text_when_problem_statement_is_empty() -> None:
    task = _task("task-1", "")
    task["task"] = "Build the documented feature"
    row = module._joined_row(task, _trace("task-1"))
    assert row is not None
    assert row["text"] == "Build the documented feature"


def test_metadata_accepts_rows_and_tasks_shapes(tmp_path: Path) -> None:
    rows = tmp_path / "rows.json"
    rows.write_text(
        json.dumps({"rows": [{"id": "a", "text": " Fix   This "}]}),
        encoding="utf-8",
    )
    tasks = tmp_path / "tasks.json"
    tasks.write_text(
        json.dumps({"tasks": [{"task_id": "b", "prompt": "Repair that"}]}),
        encoding="utf-8",
    )
    assert module._metadata(rows) == ({"a"}, {"fix this"})
    assert module._metadata(tasks) == ({"b"}, {"repair that"})
