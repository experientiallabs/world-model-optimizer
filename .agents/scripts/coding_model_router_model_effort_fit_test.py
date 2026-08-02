"""Tests for frozen model-by-effort pair selection primitives."""

from __future__ import annotations

import coding_model_router_model_effort_fit as fit
import numpy as np


def test_grouped_folds_have_zero_repository_overlap() -> None:
    groups = ["repo-a", "repo-a", "repo-b", "repo-c", "repo-d", "repo-e", "repo-f"]
    folds = fit.grouped_folds(groups, 11)
    for fold in range(fit.FOLDS):
        train = {groups[index] for index in np.flatnonzero(folds != fold)}
        test = {groups[index] for index in np.flatnonzero(folds == fold)}
        assert train.isdisjoint(test)


def test_candidate_grid_keeps_model_and_effort_in_arm_identity() -> None:
    costs = np.tile(np.arange(1, len(fit.ARMS) + 1, dtype=np.float64), (20, 1))
    rewards = np.zeros_like(costs)
    rewards[:, -1] = 1.0
    candidates = fit.candidate_grid(rewards, costs)
    assert len(candidates) == 1_596
    assert candidates[0].cheap == 0
    assert candidates[0].expensive == len(fit.ARMS) - 1
    assert fit.ARMS[candidates[-1].expensive] == "sol-max"
    assert all(
        len(fit.ARMS) - 1 in {candidate.cheap, candidate.expensive}
        for candidate in candidates
    )


def test_metrics_matches_task_blind_traffic_exactly() -> None:
    rewards = np.zeros((10, len(fit.ARMS)))
    costs = np.ones_like(rewards)
    rewards[:, 0] = np.arange(10) % 2
    rewards[:, 1] = 1 - rewards[:, 0]
    costs[:, 0] = 1.0
    costs[:, 1] = 2.0
    data = fit.Data(
        task_ids=[f"task-{index}" for index in range(10)],
        repositories=[f"repo-{index}" for index in range(10)],
        texts=[f"text {index}" for index in range(10)],
        rewards=rewards,
        costs=costs,
    )
    choices = np.asarray([0, 1] * 5)
    metrics = fit._metrics(data, choices, 0, 1)
    assert metrics.expensive_traffic == 5
    assert metrics.blind_reward == 0.5
    assert metrics.blind_cost_usd == 1.5
