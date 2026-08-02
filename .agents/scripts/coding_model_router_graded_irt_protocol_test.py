"""Tests for the conditional graded IRT protocol helpers."""

from __future__ import annotations

import numpy as np
import pytest
from coding_model_router_graded_irt_protocol import (
    cosine_knn_laplacian,
    repository_grouped_folds,
    shuffle_within_repositories,
)


def test_cosine_knn_laplacian_is_symmetric_and_deterministic() -> None:
    features = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=np.float64,
    )
    first = cosine_knn_laplacian(features, neighbors=1)
    repeated = cosine_knn_laplacian(features, neighbors=1)
    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_allclose(first, first.T)
    np.testing.assert_allclose(np.sum(first, axis=1), 0.0)
    assert np.all(np.diag(first) > 0.0)
    off_diagonal = first.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    assert np.all(off_diagonal <= 0.0)
    assert first[0, 1] < 0.0
    assert first[2, 3] < 0.0
    assert first[0, 2] == 0.0


@pytest.mark.parametrize(
    ("features", "neighbors"),
    [
        (np.asarray([[1.0], [0.0]]), 1),
        (np.asarray([[1.0], [np.nan]]), 1),
        (np.asarray([[1.0], [2.0]]), 0),
        (np.asarray([[1.0], [2.0]]), 2),
    ],
)
def test_cosine_knn_laplacian_rejects_invalid_inputs(
    features: np.ndarray,
    neighbors: int,
) -> None:
    with pytest.raises(ValueError):
        cosine_knn_laplacian(features, neighbors=neighbors)


def test_repository_grouped_folds_are_seeded_disjoint_and_complete() -> None:
    groups = np.asarray(
        [group for group in "abcdefghij" for _ in range(2)],
        dtype=object,
    )
    first = repository_grouped_folds(groups, n_splits=5, seed=11)
    repeated = repository_grouped_folds(groups, n_splits=5, seed=11)
    alternate = repository_grouped_folds(groups, n_splits=5, seed=23)

    assert len(first) == 5
    assert all(
        np.array_equal(left.test, right.test)
        for left, right in zip(first, repeated, strict=True)
    )
    assert any(
        not np.array_equal(left.test, right.test)
        for left, right in zip(first, alternate, strict=True)
    )
    seen = np.zeros(len(groups), dtype=np.int64)
    for fold in first:
        assert not set(groups[fold.train]) & set(groups[fold.test])
        seen[fold.test] += 1
        assert len(fold.test) == 4
    np.testing.assert_array_equal(seen, np.ones(len(groups), dtype=np.int64))


def test_repository_grouped_folds_balance_unequal_task_counts() -> None:
    groups = np.asarray(["a"] * 7 + ["b"] * 5 + ["c"] * 4 + ["d"] * 3 + ["e"] * 2)
    folds = repository_grouped_folds(groups, n_splits=3, seed=37)
    loads = [len(fold.test) for fold in folds]
    assert max(loads) - min(loads) <= 3


def test_shuffle_stays_within_repositories_and_is_deterministic() -> None:
    groups = np.asarray(["a", "a", "a", "b", "b", "b"], dtype=object)
    values = np.asarray(
        [[0, 10], [1, 11], [2, 12], [100, 110], [101, 111], [102, 112]],
        dtype=np.float64,
    )
    shuffled = shuffle_within_repositories(values, groups, seed=41)
    repeated = shuffle_within_repositories(values, groups, seed=41)
    np.testing.assert_array_equal(shuffled, repeated)
    assert not np.array_equal(shuffled, values)
    for group in ("a", "b"):
        indices = np.flatnonzero(groups == group)
        assert {tuple(row) for row in shuffled[indices]} == {
            tuple(row) for row in values[indices]
        }


@pytest.mark.parametrize(
    ("groups", "n_splits"),
    [
        (np.asarray([], dtype=object), 2),
        (np.asarray(["a", "a"], dtype=object), 2),
        (np.asarray(["a", "b"], dtype=object), 3),
    ],
)
def test_repository_grouped_folds_reject_invalid_inputs(
    groups: np.ndarray,
    n_splits: int,
) -> None:
    with pytest.raises(ValueError):
        repository_grouped_folds(groups, n_splits=n_splits, seed=11)
