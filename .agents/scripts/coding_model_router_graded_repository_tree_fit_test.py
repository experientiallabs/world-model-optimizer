"""Tests for the frozen repository-tree learner grid and route construction."""

from __future__ import annotations

import numpy as np
import pytest
from coding_model_router_graded_repository_tree_fit import (
    StructureCandidate,
    align_features,
    cross_fitted_scores,
    operating_grid,
    ranked_choices,
    structure_grid,
)
from coding_model_router_graded_swerebench_fit import Data


def _data(tasks: int = 20) -> Data:
    task_ids = [f"repo-{index // 2}__task-{index}" for index in range(tasks)]
    repositories = [f"repo-{index // 2}" for index in range(tasks)]
    rewards = np.tile(np.linspace(0.2, 0.9, 6), (tasks, 1))
    rewards[:, 0] += np.linspace(0.0, 0.2, tasks)
    costs = np.tile(np.linspace(0.1, 1.0, 6), (tasks, 1))
    return Data(
        task_ids=task_ids,
        repositories=repositories,
        texts=[f"language=python\ntask {index}" for index in range(tasks)],
        rewards=rewards,
        costs=costs,
        rough_cumulative_spend_usd=1.0,
    )


def _rows(data: Data) -> list[dict[str, object]]:
    return [
        {
            "task_id": task_id,
            "repository": data.repositories[index],
            "language": "python",
            "base_commit": f"commit-{index}",
            "structure": [float(index + column) for column in range(61)],
            "localization": [float(index - column) for column in range(39)],
            "prompt_shape": [float((index + column) % 3) for column in range(15)],
        }
        for index, task_id in enumerate(data.task_ids)
    ]


def test_frozen_grids_are_complete_and_stable() -> None:
    structures = structure_grid()
    points = operating_grid()
    assert len(structures) == 270
    assert len(points) == 3_510
    assert structures[0].key == "luna-low-structure-leaves7-leaf10-l21"
    assert structures[-1].key == "luna-max-prompt-shape-leaves31-leaf50-l210"
    assert points[0].key.endswith("rank20")
    assert points[-1].key.endswith("rank80")


def test_feature_alignment_builds_nested_views_and_subsets() -> None:
    data = _data()
    features = align_features(data, _rows(data)[1:])
    assert len(features.data.task_ids) == 19
    assert [view.shape for view in features.views] == [(19, 61), (19, 100), (19, 115)]
    assert features.data.task_ids[0] == data.task_ids[1]


def test_feature_alignment_rejects_low_coverage_and_repository_mismatch() -> None:
    data = _data()
    with pytest.raises(ValueError, match="coverage"):
        align_features(data, _rows(data)[:-2])
    rows = _rows(data)
    rows[0]["repository"] = "wrong/repo"
    with pytest.raises(ValueError, match="mismatch"):
        align_features(data, rows)


def test_ranked_choices_have_exact_guard_traffic_and_stable_ties() -> None:
    task_ids = ["d", "c", "b", "a"]
    scores = np.ones(4, dtype=np.float64)
    choices = ranked_choices(task_ids, scores, cheap_index=0, threshold_rank=50)
    assert choices.tolist() == [0, 0, 5, 5]


def test_cross_fit_is_complete_and_repository_disjoint() -> None:
    data = _data(30)
    aligned = align_features(data, _rows(data))
    candidate = StructureCandidate(0, 0, 0, 7, 10, 1.0)
    first = cross_fitted_scores(aligned, candidate, seed=11)
    second = cross_fitted_scores(aligned, candidate, seed=11)
    assert first.shape == (30,)
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)


def test_invalid_feature_dimensions_fail_closed() -> None:
    data = _data()
    rows = _rows(data)
    rows[0]["structure"] = [1.0]
    with pytest.raises(ValueError, match="dimensions"):
        align_features(data, rows)
