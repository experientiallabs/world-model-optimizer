"""Tests for official BigCodeBench score preparation and merging."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType

import pytest


def _module() -> ModuleType:
    path = Path(__file__).with_name("coding_model_router_bigcodebench_score.py")
    spec = importlib.util.spec_from_file_location("bigcodebench_score", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _module()


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_response_text_concatenates_output_fragments() -> None:
    response = {
        "output": [
            {
                "content": [
                    {"type": "output_text", "text": "def solve():\n"},
                    {"type": "output_text", "text": "    return 1\n"},
                ]
            }
        ]
    }
    assert module._response_text(response) == "def solve():\n    return 1\n"


def test_make_chunks_preserves_task_and_sample_order(tmp_path: Path) -> None:
    tasks = [{"task_id": f"task-{index}"} for index in range(3)]
    samples = [
        {"task_id": task["task_id"], "solution": f"solution-{sample}"}
        for task in tasks
        for sample in range(2)
    ]
    index = [
        {
            "line_number": line,
            "cell_id": f"cell-{line}",
            "task_id": sample["task_id"],
            "arm": "arm",
            "attempt": line % 2,
        }
        for line, sample in enumerate(samples)
    ]
    _write_jsonl(tmp_path / "tasks.jsonl", tasks)
    _write_jsonl(tmp_path / "samples.jsonl", samples)
    _write_jsonl(tmp_path / "sample-index.jsonl", index)
    module.make_chunks(tmp_path, tasks_per_chunk=2, cells_per_task=2)
    first = module._read_object(tmp_path / "score-chunks/chunk-00/chunk.json")
    second = module._read_object(tmp_path / "score-chunks/chunk-01/chunk.json")
    assert first["task_ids"] == ["task-0", "task-1"]
    assert first["samples"] == 4
    assert second["task_ids"] == ["task-2"]
    assert second["samples"] == 2


def test_make_chunks_rejects_incomplete_matrix(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "tasks.jsonl", [{"task_id": "task"}])
    _write_jsonl(tmp_path / "samples.jsonl", [{"task_id": "task", "solution": "x"}])
    _write_jsonl(tmp_path / "sample-index.jsonl", [])
    with pytest.raises(ValueError, match="incomplete"):
        module.make_chunks(tmp_path, tasks_per_chunk=1, cells_per_task=2)


def test_validate_official_result_requires_dense_exact_task_set() -> None:
    module._validate_official_result(
        {"eval": {"task-a": [{"status": "pass"}, {"status": "fail"}]}},
        ["task-a"],
        cells_per_task=2,
    )
    with pytest.raises(ValueError, match="task set differs"):
        module._validate_official_result(
            {"eval": {"task-b": [{}, {}]}},
            ["task-a"],
            cells_per_task=2,
        )
    with pytest.raises(ValueError, match="1 results"):
        module._validate_official_result(
            {"eval": {"task-a": [{"status": "pass"}]}},
            ["task-a"],
            cells_per_task=2,
        )


def test_merge_chunks_builds_dense_scores_in_global_order(tmp_path: Path) -> None:
    tasks = [{"task_id": "task-a"}, {"task_id": "task-b"}]
    index = [
        {
            "line_number": line,
            "cell_id": f"cell-{line}",
            "task_id": task_id,
            "arm": f"arm-{sample}",
            "attempt": sample,
        }
        for line, (task_id, sample) in enumerate(
            [("task-a", 0), ("task-a", 1), ("task-b", 0), ("task-b", 1)]
        )
    ]
    _write_jsonl(tmp_path / "tasks.jsonl", tasks)
    _write_jsonl(tmp_path / "sample-index.jsonl", index)
    for chunk_number, task_id in enumerate(["task-a", "task-b"]):
        chunk = tmp_path / f"score-chunks/chunk-{chunk_number:02d}"
        _write_jsonl(
            chunk / "samples.jsonl",
            [
                {"task_id": task_id, "solution": "a"},
                {"task_id": task_id, "solution": "b"},
            ],
        )
        module._write_object(
            chunk / "chunk.json",
            {
                "chunk": chunk_number,
                "task_ids": [task_id],
                "tasks": 1,
                "samples_sha256": module._sha256_file(chunk / "samples.jsonl"),
            },
        )
        module._write_object(
            chunk / "samples_eval_results.json",
            {
                "eval": {
                    task_id: [
                        {"status": "pass", "details": {}},
                        {"status": "fail", "details": {"case": "failed"}},
                    ]
                }
            },
        )
    module.merge_chunks(tmp_path, cells_per_task=2)
    scores = module._read_jsonl(tmp_path / "scores.jsonl")
    assert [row["cell_id"] for row in scores] == [
        "cell-0",
        "cell-1",
        "cell-2",
        "cell-3",
    ]
    assert [row["reward"] for row in scores] == [1.0, 0.0, 1.0, 0.0]
    assert module._read_object(tmp_path / "score-manifest.json")["cells"] == 4
