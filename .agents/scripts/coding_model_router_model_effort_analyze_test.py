"""Tests for repository-bootstrap model-effort confirmation analysis."""

from __future__ import annotations

import coding_model_router_model_effort_analyze as analyze
import numpy as np


def test_bootstrap_keeps_repository_siblings_together() -> None:
    repositories = ["a", "a", "b", "c", "c"]
    values = np.asarray([1.0, 1.0, 0.0, 0.5, 0.5])
    group_index, weights = analyze._bootstrap_weights(repositories)
    means = analyze._bootstrap_means(values, group_index, weights)
    assert means.shape == (analyze.BOOTSTRAPS,)
    assert np.isfinite(means).all()
    assert 0.0 <= float(means.min()) <= float(means.max()) <= 1.0


def test_bootstrap_vectorizes_family_nulls() -> None:
    repositories = ["a", "a", "b", "c"]
    values = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ]
    )
    group_index, weights = analyze._bootstrap_weights(repositories)
    means = analyze._bootstrap_means(values, group_index, weights)
    assert means.shape == (analyze.BOOTSTRAPS, 2)
    assert np.isfinite(means).all()
