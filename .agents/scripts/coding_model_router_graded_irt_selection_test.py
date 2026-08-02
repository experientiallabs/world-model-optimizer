"""Tests for conditional graded IRT robust selection helpers."""

from __future__ import annotations

import numpy as np
import pytest
from coding_model_router_graded_irt_selection import (
    quality_guarded_choices,
    repository_robust_margin,
    repository_robust_metrics,
)


def test_quality_guarded_choices_apply_cost_pressure_then_revert() -> None:
    probabilities = np.asarray(
        [
            [0.91, 0.92, 0.95],
            [0.88, 0.80, 0.95],
            [0.90, 0.90, 0.90],
        ],
        dtype=np.float64,
    )
    costs = np.asarray(
        [
            [1.0, 2.0, 10.0],
            [1.0, 2.0, 10.0],
            [1.0, 2.0, 10.0],
        ],
        dtype=np.float64,
    )
    choices = quality_guarded_choices(
        probabilities,
        costs,
        guard_arm=2,
        cost_penalty=0.1,
    )
    np.testing.assert_array_equal(choices, np.asarray([0, 2, 0]))


def test_repository_robust_metrics_lower_reward_and_raise_cost() -> None:
    rewards = np.asarray(
        [[0.9, 1.0], [0.7, 1.0], [0.2, 1.0], [0.4, 1.0]],
        dtype=np.float64,
    )
    costs = np.asarray(
        [[1.0, 4.0], [1.0, 4.0], [3.0, 4.0], [3.0, 4.0]],
        dtype=np.float64,
    )
    choices = np.zeros(4, dtype=np.int64)
    repositories = np.asarray(["a", "a", "b", "b"], dtype=object)
    metrics = repository_robust_metrics(
        rewards,
        costs,
        choices,
        repositories,
        radius=0.1,
    )
    np.testing.assert_allclose(metrics.repository_rewards, [0.8, 0.3])
    np.testing.assert_allclose(metrics.repository_costs, [1.0, 3.0])
    assert metrics.reward_lower_bound < 0.55
    assert metrics.cost_upper_bound > 2.0


def test_repository_robust_margin_preserves_paired_task_difference() -> None:
    margins = np.asarray([0.1, 0.3, -0.4, 0.0], dtype=np.float64)
    repositories = np.asarray(["a", "a", "b", "b"], dtype=object)
    result = repository_robust_margin(margins, repositories, radius=0.1)
    np.testing.assert_allclose(result.repository_margins, [0.2, -0.2])
    assert result.lower_bound < 0.0


@pytest.mark.parametrize(
    ("guard_arm", "cost_penalty", "quality_floor"),
    [(-1, 0.0, 0.95), (2, 0.0, 0.95), (1, -0.1, 0.95), (1, 0.0, 0.0)],
)
def test_quality_guarded_choices_reject_invalid_settings(
    guard_arm: int,
    cost_penalty: float,
    quality_floor: float,
) -> None:
    probabilities = np.asarray([[0.5, 0.8]], dtype=np.float64)
    costs = np.asarray([[1.0, 2.0]], dtype=np.float64)
    with pytest.raises(ValueError):
        quality_guarded_choices(
            probabilities,
            costs,
            guard_arm=guard_arm,
            cost_penalty=cost_penalty,
            quality_floor=quality_floor,
        )
