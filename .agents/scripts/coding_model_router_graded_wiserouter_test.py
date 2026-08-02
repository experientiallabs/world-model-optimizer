"""Tests for the graded workload-budget router helpers."""

from __future__ import annotations

import numpy as np
import pytest
from coding_model_router_graded_swerebench_fit import Data
from coding_model_router_graded_wiserouter import (
    NULL_COUNT,
    Candidate,
    ContextStatistics,
    _context_statistics,
    candidate_grid,
    deterministic_choices,
    higher_quantile,
    null_gate,
    permute_repository_blocks,
    solve_workload_policy,
)


def _data() -> Data:
    rewards = np.asarray(
        [
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.9],
            [0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
            [0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.2],
        ],
        dtype=np.float64,
    )
    costs = np.tile(np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 1.0]), (4, 1))
    return Data(
        task_ids=["repo-a__1", "repo-a__2", "repo-b__1", "repo-b__2"],
        repositories=["repo-a", "repo-a", "repo-b", "repo-b"],
        texts=[
            "repository=repo-a\nlanguage=python\nfix parser one",
            "repository=repo-a\nlanguage=python\nfix parser two",
            "repository=repo-b\nlanguage=python\nfix runtime one",
            "repository=repo-b\nlanguage=python\nfix runtime two",
        ],
        rewards=rewards,
        costs=costs,
        rough_cumulative_spend_usd=0.0,
    )


def test_candidate_grid_is_frozen_and_complete() -> None:
    values = candidate_grid()
    assert len(values) == 90
    assert len({value.key for value in values}) == 90
    assert values[0].key == "hash512-j8-shrink0-save0.4"
    assert values[-1].key == "hash2048-j32-shrink16-save0.6"


def test_context_statistics_shrink_toward_global_arm_means() -> None:
    data = _data()
    rewards = np.vstack(
        [
            np.repeat(data.rewards[[0]], 8, axis=0),
            np.repeat(data.rewards[[3]], 8, axis=0),
        ]
    )
    costs = np.repeat(data.costs[[0]], 16, axis=0)
    assignments = np.asarray([0] * 8 + [1] * 8, dtype=np.int64)
    no_shrink = _context_statistics(
        assignments,
        rewards,
        costs,
        contexts=2,
        shrinkage=0.0,
    )
    shrunk = _context_statistics(
        assignments,
        rewards,
        costs,
        contexts=2,
        shrinkage=4.0,
    )
    global_reward = float(np.mean(rewards[:, 0]))
    assert abs(shrunk.rewards[0, 0] - global_reward) < abs(
        no_shrink.rewards[0, 0] - global_reward
    )
    assert np.allclose(shrunk.context_probability, [0.5, 0.5])


def test_workload_lp_respects_budget_and_context_simplex() -> None:
    statistics = ContextStatistics(
        context_probability=np.asarray([0.5, 0.5]),
        rewards=np.asarray([[0.2, 0.9], [0.8, 0.9]]),
        costs=np.asarray([[0.1, 1.0], [0.1, 1.0]]),
        baseline_cost=1.0,
    )
    policy = solve_workload_policy(statistics, savings=0.4)
    assert np.allclose(np.sum(policy, axis=1), 1.0)
    assert float(np.sum(statistics.context_probability[:, None] * policy * statistics.costs)) <= (
        0.6 + 1e-8
    )
    assert policy[0, 0] < policy[1, 0]


def test_deterministic_choices_cover_exact_context_counts() -> None:
    task_ids = [f"task-{index}" for index in range(10)]
    indices = np.arange(10, dtype=np.int64)
    assignments = np.asarray([0] * 5 + [1] * 5, dtype=np.int64)
    policy = np.asarray([[0.6, 0.4], [0.2, 0.8]], dtype=np.float64)
    statistics = ContextStatistics(
        context_probability=np.asarray([0.5, 0.5]),
        rewards=np.asarray([[0.4, 0.8], [0.3, 0.9]]),
        costs=np.asarray([[0.1, 1.0], [0.1, 1.0]]),
        baseline_cost=1.0,
    )
    first = deterministic_choices(
        task_ids,
        indices,
        assignments,
        policy,
        statistics,
        seed=11,
        fold=0,
    )
    second = deterministic_choices(
        task_ids,
        indices,
        assignments,
        policy,
        statistics,
        seed=11,
        fold=0,
    )
    assert np.array_equal(first, second)
    assert np.bincount(first[:5], minlength=2).tolist() == [3, 2]
    assert np.bincount(first[5:], minlength=2).tolist() == [1, 4]


def test_repository_block_permutation_preserves_complete_rows() -> None:
    data = _data()
    permuted = permute_repository_blocks(data.rewards, data, seed=20_260_801)
    assert np.array_equal(permuted[:2], data.rewards[2:])
    assert np.array_equal(permuted[2:], data.rewards[:2])
    original_rows = sorted(tuple(row) for row in data.rewards)
    permuted_rows = sorted(tuple(row) for row in permuted)
    assert permuted_rows == original_rows


def test_higher_quantile_is_conservative() -> None:
    assert higher_quantile([0.0, 1.0, 2.0, 3.0], 0.5) == 2.0
    with pytest.raises(ValueError, match="quantile"):
        higher_quantile([], 0.5)


def test_null_gate_requires_four_positive_seed_margins() -> None:
    seeds = (11, 23, 37, 41, 59)
    real = [
        {"seed": seed, "matched_blind_advantage": 0.05 if seed != 59 else 0.01}
        for seed in seeds
    ]
    nulls = [
        [
            {"seed": seed, "matched_blind_advantage": 0.02}
            for seed in seeds
        ]
        for _ in range(NULL_COUNT)
    ]
    result = null_gate(real, nulls)
    assert result["passed"] is True
    assert result["real_minus_null95"] == pytest.approx(0.022)


def test_null_gate_rejects_incomplete_controls() -> None:
    real = [
        {"seed": seed, "matched_blind_advantage": 0.05}
        for seed in (11, 23, 37, 41, 59)
    ]
    with pytest.raises(ValueError, match="128"):
        null_gate(real, [])


def test_context_support_is_enforced() -> None:
    data = _data()
    with pytest.raises(ValueError, match="support"):
        _context_statistics(
            np.asarray([0, 0, 1, 1], dtype=np.int64),
            data.rewards,
            data.costs,
            contexts=2,
            shrinkage=0.0,
        )


def test_invalid_lp_budget_fails_closed() -> None:
    statistics = ContextStatistics(
        context_probability=np.asarray([1.0]),
        rewards=np.asarray([[0.4, 0.9]]),
        costs=np.asarray([[0.8, 1.0]]),
        baseline_cost=1.0,
    )
    with pytest.raises(RuntimeError, match="LP failed"):
        solve_workload_policy(statistics, savings=0.6)


def test_candidate_key_is_stable() -> None:
    candidate = Candidate(7, 512, 16, 4.0, 0.45)
    assert candidate.key == "hash512-j16-shrink4-save0.45"
