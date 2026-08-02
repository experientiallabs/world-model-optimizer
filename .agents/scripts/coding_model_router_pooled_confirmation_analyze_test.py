"""Tests for pooled-uplift external confirmation analysis."""

from __future__ import annotations

import coding_model_router_pooled_confirmation_analyze as analyze
import numpy as np


def test_best_null_comparison_uses_outcome_best_frozen_null() -> None:
    rewards = np.zeros((4, 5), dtype=np.float64)
    rewards[:2, 2] = 1.0
    rewards[2:, 4] = 1.0
    data = analyze.ConfirmationData(
        task_ids=[f"task-{index}" for index in range(4)],
        repositories=["repo-a", "repo-b", "repo-a", "repo-b"],
        rewards=rewards,
        costs=np.ones_like(rewards),
    )
    real = np.asarray([2, 2, 4, 4], dtype=np.int64)
    nulls = np.tile(np.asarray([2, 4, 2, 4], dtype=np.int64), (analyze.NULL_COUNT, 1))
    nulls[7] = np.asarray([2, 4, 4, 2], dtype=np.int64)

    result = analyze._best_null_comparison(
        data,
        real,
        nulls,
        bootstrap_resamples=500,
    )

    assert result["best_null_index"] == 0
    assert result["best_null_matched_blind_advantage"] == 0.0
    assert result["real_minus_best_null_reward"] == 0.5
    assert result["real_minus_best_null_ci95_lower"] == 0.5
    assert result["passed"] is True


def test_null_comparison_allows_retained_traffic_shift() -> None:
    rewards = np.zeros((2, 5), dtype=np.float64)
    rewards[0, 2] = 1.0
    rewards[1, 4] = 1.0
    data = analyze.ConfirmationData(
        task_ids=["task-0", "task-1"],
        repositories=["repo-a", "repo-b"],
        rewards=rewards,
        costs=np.ones_like(rewards),
    )
    real = np.asarray([2, 4], dtype=np.int64)
    nulls = np.tile(np.asarray([2, 2], dtype=np.int64), (analyze.NULL_COUNT, 1))

    result = analyze._best_null_comparison(
        data,
        real,
        nulls,
        bootstrap_resamples=100,
    )

    assert result["best_null_index"] == 0
    assert result["best_null_reward"] == 0.5
    assert result["best_null_arm_counts"]["luna-high"] == 2
    assert result["real_minus_best_null_reward"] == 0.5
