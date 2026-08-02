"""Tests for aggregate-only repository-tree primary selection."""

from __future__ import annotations

import copy

import pytest
from coding_model_router_graded_repository_tree_fit import PROTOCOL
from coding_model_router_graded_repository_tree_run import select_primary
from coding_model_router_graded_swerebench_fit import SEEDS


def _reports(*, retention: float, savings: float) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    for seed in SEEDS:
        metrics: list[dict[str, object]] = [
            {
                "key": f"candidate-{index}",
                "seed": seed,
                "quality_retention": retention,
                "cost_savings": savings,
                "matched_blind_advantage": 0.01,
                "dominated_by_static": [],
            }
            for index in range(3_510)
        ]
        reports.append(
            {
                "protocol": PROTOCOL,
                "valid": True,
                "seed": seed,
                "tasks": 640,
                "structures": 270,
                "operating_points": 3_510,
                "metrics": metrics,
                "inputs": {"features": "abc"},
                "provider_calls": 0,
            }
        )
    return reports


def test_primary_selection_requires_every_seed_gate() -> None:
    passing = select_primary(_reports(retention=0.96, savings=0.41))
    assert passing["primary_eligible_count"] == 3_510
    assert passing["requires_family_null"] is True
    assert passing["development_passed"] is False
    failing = select_primary(_reports(retention=0.94, savings=0.50))
    assert failing["primary_eligible_count"] == 0
    assert failing["requires_family_null"] is False


def test_primary_selection_rejects_mismatched_inputs() -> None:
    reports = _reports(retention=0.96, savings=0.41)
    changed = copy.deepcopy(reports)
    changed[-1]["inputs"] = {"features": "changed"}
    with pytest.raises(ValueError, match="different inputs"):
        select_primary(changed)


def test_primary_selection_rejects_missing_seed() -> None:
    with pytest.raises(ValueError, match="five"):
        select_primary(_reports(retention=0.96, savings=0.41)[:-1])
