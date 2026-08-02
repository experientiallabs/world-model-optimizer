"""Tests for the Codeforces reasoning-effort matrix worker."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _module() -> ModuleType:
    path = Path(__file__).with_name("coding_model_router_codeforces_matrix.py")
    spec = importlib.util.spec_from_file_location("codeforces_matrix", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _module()


def test_schedule_is_dense_unique_and_deterministic() -> None:
    tasks = [{"task_id": "b"}, {"task_id": "a"}]
    first = module._schedule(tasks, attempts=2, seed=7)
    second = module._schedule(list(reversed(tasks)), attempts=2, seed=7)
    assert [cell.cell_id for cell in first] == [cell.cell_id for cell in second]
    assert len(first) == 2 * len(module.ARMS) * 2
    assert len({cell.cell_id for cell in first}) == len(first)


def test_extract_code_response_usage_and_cost() -> None:
    response = {
        "output": [{"content": [{"type": "output_text", "text": "```python\nprint(1)\n```"}]}],
        "usage": {
            "input_tokens": 1_000_000,
            "input_tokens_details": {"cached_tokens": 400_000},
            "output_tokens": 100_000,
            "output_tokens_details": {"reasoning_tokens": 70_000},
        },
    }
    assert module._extract_code(module._response_text(response)) == "print(1)\n"
    usage = module._usage(response)
    assert usage["reasoning_tokens"] == 70_000
    assert module._cost(usage) == 1.24


def test_append_only_state_resumes(tmp_path: Path) -> None:
    state = module.MatrixState(tmp_path, ceiling_usd=100.0, prior_spend_usd=3.0)
    state.reserve()
    state.persist({"cell_id": "a", "cost_usd": 0.2})
    resumed = module.MatrixState(tmp_path, ceiling_usd=100.0, prior_spend_usd=3.0)
    assert resumed.completed("a")
    assert resumed.spent == 0.2


def test_grade_returns_fraction_and_hashes(monkeypatch: pytest.MonkeyPatch) -> None:
    completed = iter(
        [
            SimpleNamespace(returncode=0, stdout="3\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="9\n", stderr=""),
        ]
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: next(completed),
    )
    task = {
        "task_id": "sum",
        "memory_limit_mb": 256,
        "time_limit_s": 2.0,
        "tests": [
            {"input": "1 2\n", "output": "3\n"},
            {"input": "4 5\n", "output": "10\n"},
        ],
    }
    reward, rows = module._grade("a,b=map(int,input().split());print(a+b)\n", task)
    assert reward == 0.5
    assert [row["passed"] for row in rows] == [True, False]
    assert all("stdout_sha256" in row for row in rows)


def test_corpus_information_boundary(tmp_path: Path) -> None:
    path = tmp_path / "corpus.json"
    path.write_text(
        json.dumps(
            {
                "target_outcomes_used": True,
                "published_generations_loaded": False,
                "tasks": [],
            }
        ),
        encoding="utf-8",
    )
    assert module._read_object(path)["target_outcomes_used"] is True


def test_default_and_corrected_output_limits_are_explicit() -> None:
    assert module.MAX_OUTPUT_TOKENS == 32_768
    assert 131_072 > module.MAX_OUTPUT_TOKENS
