"""Unit tests for BeyondSWE trace-burden transfer fitting."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


def _module() -> ModuleType:
    path = Path(__file__).with_name("coding_model_router_beyondswe_fit.py")
    spec = importlib.util.spec_from_file_location("beyondswe_fit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _module()


def test_burden_increases_with_failure_and_trace_work() -> None:
    rows = [
        {
            "reward": 1.0,
            "trajectory_steps": 2,
            "total_prompt_tokens": 100,
            "total_completion_tokens": 10,
        },
        {
            "reward": 0.0,
            "trajectory_steps": 20,
            "total_prompt_tokens": 10_000,
            "total_completion_tokens": 1_000,
        },
    ]
    burden, labels = module._burden(rows)
    assert burden[1] > burden[0]
    assert set(labels) == {
        "reward_deficit",
        "log_steps",
        "log_prompt_tokens",
        "log_completion_tokens",
    }


def test_source_group_folds_have_no_repository_overlap() -> None:
    groups = [f"repo-{index // 2}" for index in range(20)]
    texts = [f"Repair task {index}" for index in range(20)]
    data = module.SourceData(
        task_ids=[f"task-{index}" for index in range(20)],
        groups=groups,
        texts=texts,
        structural=np.asarray([module._structural(text) for text in texts]),
        burden=np.linspace(-1.0, 1.0, 20),
        raw_labels={},
    )
    candidate = module.Candidate("structural", "structural", 1, 10.0)
    scores, audits = module._oof_source(data, candidate, seed=7)
    assert len(scores) == 20
    assert len(audits) == 5
    assert all(row["group_overlap"] == 0 for row in audits)


def test_matched_blind_comparison_uses_equal_strong_traffic() -> None:
    validation = module.ValidationData(
        task_ids=["a", "b", "c", "d", "e"],
        groups=["a", "b", "c", "d", "e"],
        texts=["a", "b", "c", "d", "e"],
        structural=np.zeros((5, 18)),
        cheap=np.zeros(5),
        strong=np.asarray([1.0, 0.0, 0.0, 0.0, 0.0]),
    )
    rows = module._operating_points(
        validation,
        np.asarray([5.0, 4.0, 3.0, 2.0, 1.0]),
    )
    gate = next(
        row
        for row in rows
        if row["strong_traffic_fraction"] == module.GATE_TRAFFIC_FRACTION
    )
    assert gate["strong_traffic_tasks"] == 1
    assert gate["router_reward"] == 0.2
    assert gate["matched_blind_reward"] == 0.04
    assert gate["advantage_vs_matched_blind"] > 0.0
