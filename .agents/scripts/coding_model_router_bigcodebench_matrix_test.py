"""Tests for the BigCodeBench reasoning-effort matrix worker."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _module() -> ModuleType:
    path = Path(__file__).with_name("coding_model_router_bigcodebench_matrix.py")
    spec = importlib.util.spec_from_file_location("bigcodebench_matrix", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _module()


def _row(task_id: str, prompt: str, libs: list[str] | None = None) -> dict[str, object]:
    return {
        "task_id": task_id,
        "instruct_prompt": prompt,
        "entry_point": "solve",
        "libs": libs or [],
    }


def test_selection_keeps_hard_tasks_and_removes_target_overlap() -> None:
    full = [
        _row("hard-1", "Implement alpha", ["numpy"]),
        _row("hard-2", "Implement beta", ["pandas"]),
        _row("target-id", "Unrelated prompt"),
        _row("target-text", "  Fix THE widget "),
        _row("fill-1", "Implement gamma"),
        _row("fill-2", "Implement delta"),
    ]
    target = [
        {"id": "target-id", "text": "another target"},
        {"id": "deep-2", "text": "fix the widget"},
    ]
    selected, audit = module._select_tasks(
        full,
        {"hard-1", "hard-2"},
        target,
        limit=3,
        seed=17,
    )
    ids = [row["task_id"] for row in selected]
    assert ids[:2] == ["hard-1", "hard-2"]
    assert set(ids).isdisjoint({"target-id", "target-text"})
    assert audit["overlap_tasks_removed"] == 2
    assert audit["retained_hard_tasks"] == 2


def test_target_feature_view_rejects_label_access(tmp_path: Path) -> None:
    path = tmp_path / "target.json"
    payload = {
        "rows": [{"id": "target", "text": "Fix it"}],
        "target_reward_fields_accessed": False,
        "target_cost_fields_accessed": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert module._target_feature_rows(path) == payload["rows"]
    payload["target_cost_fields_accessed"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not label-free"):
        module._target_feature_rows(path)


def test_schedule_is_dense_unique_and_deterministic() -> None:
    tasks = [_row("task-b", "B"), _row("task-a", "A")]
    first = module._schedule(tasks, seed=9)
    second = module._schedule(list(reversed(tasks)), seed=9)
    assert [cell.cell_id for cell in first] == [cell.cell_id for cell in second]
    assert len(first) == 2 * len(module.ARMS) * module.ATTEMPTS
    assert len({cell.cell_id for cell in first}) == len(first)
    for task_id in {"task-a", "task-b"}:
        cells = [cell for cell in first if cell.task_id == task_id]
        assert {cell.arm for cell in cells} == set(module.ARMS)
        assert {cell.attempt for cell in cells} == set(range(module.ATTEMPTS))


def test_response_text_usage_and_cost_extraction() -> None:
    response = {
        "output": [
            {
                "content": [
                    {"type": "output_text", "text": "def solve():\n"},
                    {"type": "output_text", "text": "    return 1\n"},
                ]
            }
        ],
        "usage": {
            "input_tokens": 1_000_000,
            "input_tokens_details": {"cached_tokens": 400_000},
            "output_tokens": 100_000,
            "output_tokens_details": {"reasoning_tokens": 70_000},
        },
    }
    assert module._response_text(response) == "def solve():\n    return 1\n"
    usage = module._usage(response)
    assert usage["reasoning_tokens"] == 70_000
    assert module._cost(usage) == 1.24


def test_append_only_state_resumes_without_rewriting(tmp_path: Path) -> None:
    state = module.MatrixState(
        tmp_path,
        ceiling_usd=100.0,
        prior_spend_usd=2.0,
    )
    state.reserve()
    state.persist({"cell_id": "a", "cost_usd": 0.25})
    state.reserve()
    state.persist({"cell_id": "b", "cost_usd": 0.5})
    assert [row["cell_id"] for row in module._read_jsonl(state.path)] == ["a", "b"]
    resumed = module.MatrixState(
        tmp_path,
        ceiling_usd=100.0,
        prior_spend_usd=2.0,
    )
    assert resumed.completed("a")
    assert resumed.completed("b")
    assert resumed.spent == 0.75


def test_heldout_oracle_recovers_repeatable_complementarity() -> None:
    rewards = module.np.zeros(
        (8, len(module.ARMS), module.ATTEMPTS),
        dtype=module.np.float64,
    )
    rewards[:4, 0, :] = 1.0
    rewards[4:, 1, :] = 1.0
    report = module._heldout_oracle(
        rewards,
        ["family-a"] * 4 + ["family-b"] * 4,
        seed=7,
        bootstraps_per_split=4,
    )
    assert report["attempt_splits"] == 10
    assert report["mean_heldout_oracle_headroom"] == 0.5


def test_heldout_oracle_rejects_task_blind_ordering() -> None:
    rewards = module.np.zeros(
        (6, len(module.ARMS), module.ATTEMPTS),
        dtype=module.np.float64,
    )
    rewards[:, 0, :] = 1.0
    report = module._heldout_oracle(
        rewards,
        ["family-a"] * 3 + ["family-b"] * 3,
        seed=11,
        bootstraps_per_split=3,
    )
    assert report["mean_heldout_oracle_headroom"] == 0.0


def test_heldout_oracle_bootstraps_mean_across_attempt_splits() -> None:
    rng = module.np.random.default_rng(19)
    rewards = rng.integers(
        0,
        2,
        size=(6, len(module.ARMS), module.ATTEMPTS),
    ).astype(module.np.float64)
    report = module._heldout_oracle(
        rewards,
        ["one-family"] * len(rewards),
        seed=23,
        bootstraps_per_split=7,
    )
    mean = report["mean_heldout_oracle_headroom"]
    assert report["family_bootstraps"] == 7
    assert report["heldout_oracle_headroom_95ci"] == pytest.approx([mean] * 3)
