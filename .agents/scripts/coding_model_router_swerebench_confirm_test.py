"""Tests for one-shot SWE-rebench external confirmation analysis."""

from __future__ import annotations

import coding_model_router_swerebench_confirm as confirm
import numpy as np


def _data() -> confirm.ConfirmationData:
    rewards = np.zeros((4, 5), dtype=np.float64)
    rewards[:2, 0] = 1.0
    rewards[2:, 4] = 1.0
    costs = np.tile(np.arange(1.0, 6.0, dtype=np.float64), (4, 1))
    return confirm.ConfirmationData(
        task_ids=[f"task-{index}" for index in range(4)],
        repositories=["repo-a", "repo-b", "repo-a", "repo-b"],
        rewards=rewards,
        costs=costs,
    )


def test_matched_blind_uses_identical_effort_counts() -> None:
    result = confirm._evaluate_route(
        _data(),
        np.asarray([0, 0, 4, 4], dtype=np.int64),
        bootstrap_resamples=500,
    )

    assert result["reward"] == 1.0
    assert result["matched_blind_reward"] == 0.5
    assert result["matched_blind_advantage"] == 0.5
    assert result["matched_blind_advantage_ci95_lower"] == 0.5
    assert result["primary_matched_blind_passed"] is True
    assert result["static_dominance_passed"] is True
    assert result["quality_retention_passed"] is True
    assert result["arm_counts"] == {
        "luna-low": 2,
        "luna-medium": 0,
        "luna-high": 0,
        "luna-xhigh": 0,
        "luna-max": 2,
    }


def test_shuffled_route_fails_primary_gate() -> None:
    result = confirm._evaluate_route(
        _data(),
        np.asarray([4, 4, 0, 0], dtype=np.int64),
        bootstrap_resamples=500,
    )

    assert result["reward"] == 0.0
    assert result["matched_blind_reward"] == 0.5
    assert result["matched_blind_advantage"] == -0.5
    assert result["primary_matched_blind_passed"] is False
    assert result["quality_retention_passed"] is False


def test_repository_bootstrap_is_deterministic() -> None:
    values = np.asarray([1.0, -0.5, 0.5, 0.25], dtype=np.float64)
    repositories = ["repo-a", "repo-a", "repo-b", "repo-b"]

    first = confirm._repository_bootstrap_lower(
        values,
        repositories,
        resamples=1_000,
    )
    second = confirm._repository_bootstrap_lower(
        values,
        repositories,
        resamples=1_000,
    )

    assert first == second
