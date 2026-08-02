"""Focused tests for graded confirmation analysis."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).parents[1] / ".agents" / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "coding_model_router_graded_swerebench_confirm_analyze",
    SCRIPTS / "coding_model_router_graded_swerebench_confirm_analyze.py",
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_repository_bootstrap_margin_passes_identical_quality() -> None:
    result = module._bootstrap_margin(
        ["repo-a", "repo-a", "repo-b"],
        np.asarray([1.0, 0.5, 0.25]),
        np.asarray([1.0, 0.5, 0.25]),
    )
    assert result["passed"] is True
    assert result["lower_95"] > 0.0


def test_repository_bootstrap_requires_task_signal_over_matched_blind() -> None:
    result = module._bootstrap_matched_blind(
        ["repo-a", "repo-a", "repo-b"],
        np.asarray([0.8, 0.7, 0.6]),
        np.asarray([0.7, 0.6, 0.5]),
    )
    assert result["passed"] is True
    assert result["lower_95"] > 0.0
    assert result["seed"] == module.MATCHED_BLIND_BOOTSTRAP_SEED
    tied = module._bootstrap_matched_blind(
        ["repo-a", "repo-b"],
        np.asarray([0.5, 0.5]),
        np.asarray([0.5, 0.5]),
    )
    assert tied["passed"] is False


def test_metrics_use_fit_selected_baseline() -> None:
    rewards = np.asarray([[0.2, 0.4], [0.3, 0.5]])
    costs = np.asarray([[1.0, 3.0], [1.0, 3.0]])
    original = module.ARMS
    vars(module)["ARMS"] = ("cheap", "guard")
    try:
        metrics = module._metrics(rewards, costs, np.asarray([0, 1]), baseline=1)
    finally:
        vars(module)["ARMS"] = original
    assert metrics["quality_retention"] == 0.35 / 0.45
    assert metrics["cost_savings"] > 0.0
