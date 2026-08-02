"""Focused tests for graded SWE-rebench WMO kNN fitting."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).parents[1] / ".agents" / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "coding_model_router_graded_swerebench_fit",
    SCRIPTS / "coding_model_router_graded_swerebench_fit.py",
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _data() -> module.Data:
    rewards = np.asarray(
        [
            [0.2, 0.4, 0.6, 0.8, 0.9, 1.0],
            [0.3, 0.5, 0.7, 0.8, 1.0, 0.9],
        ]
    )
    costs = np.asarray([[1, 2, 3, 4, 5, 10], [1, 2, 3, 4, 5, 10]], dtype=float)
    return module.Data(
        task_ids=["a", "b"],
        repositories=["repo-a", "repo-b"],
        texts=["text-a", "text-b"],
        rewards=rewards,
        costs=costs,
        rough_cumulative_spend_usd=1.0,
    )


def test_candidate_grid_covers_every_guard_and_frozen_knob() -> None:
    candidates = module.candidate_grid()
    assert len(candidates) == 480
    assert {candidate.guard for candidate in candidates} == set(module.ARMS)
    assert {candidate.pick_lam for candidate in candidates} == set(module.LAM_VALUES)


def test_fit_selected_static_uses_quality_then_cost() -> None:
    data = _data()
    assert module.ARMS[module._static_baseline(data)] == "luna-max"


def test_metrics_include_task_blind_control_and_static_dominance() -> None:
    metrics = module._metrics(_data(), np.asarray([0, 5]))
    assert "matched_blind_advantage" in metrics
    assert "dominated_by_static" in metrics
    assert metrics["quality_retention"] > 0.0


def test_metrics_reject_incomplete_routes() -> None:
    with pytest.raises(ValueError, match="cover every task"):
        module._metrics(_data(), np.asarray([0, -1]))


def test_development_gate_requires_positive_blind_advantage_in_every_seed() -> None:
    rows = [
        {
            "quality_retention": 0.96,
            "cost_savings": 0.5,
            "matched_blind_advantage": 0.02,
            "dominated_by_static": [],
        }
        for _ in module.SEEDS
    ]
    assert module._passes_development_gates(rows)
    rows[0]["matched_blind_advantage"] = -0.001
    rows[1]["matched_blind_advantage"] = 0.1
    assert not module._passes_development_gates(rows)


def test_repository_folds_have_zero_overlap() -> None:
    groups = [f"repo-{index}" for index in range(30) for _ in range(2)]
    folds = module.grouped_folds(groups, 11)
    for fold in range(module.FOLDS):
        train = {groups[index] for index in np.flatnonzero(folds != fold)}
        heldout = {groups[index] for index in np.flatnonzero(folds == fold)}
        assert train.isdisjoint(heldout)


def test_task_text_contains_only_pre_call_fields() -> None:
    text = module._task_text(
        {"repository": "owner/repo", "language": "python", "prompt": "repair bug"}
    )
    assert text == "repository=owner/repo\nlanguage=python\nrepair bug"
