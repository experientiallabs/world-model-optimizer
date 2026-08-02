"""Pure protocol helpers for the conditional graded IRT router study.

The helpers in this module operate only on group identities and in-memory arrays. They provide
seeded repository-disjoint folds, a pre-call feature graph, and the within-repository
shuffled-label control without loading outcomes, fitting a model, or persisting state.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RepositoryFold:
    """One repository-disjoint train and test partition."""

    train: np.ndarray
    test: np.ndarray


def _validate_groups(groups: np.ndarray) -> tuple[str, ...]:
    """Return validated repository identities aligned with task rows."""
    if groups.ndim != 1 or not len(groups):
        raise ValueError("repository groups must be a nonempty vector")
    values = tuple(str(value) for value in groups)
    if any(not value for value in values):
        raise ValueError("repository groups must be nonempty strings")
    return values


def _seeded_digest(seed: int, *parts: str) -> bytes:
    """Return a stable seed-sensitive digest for protocol ordering."""
    value = ":".join([str(seed), *parts])
    return hashlib.sha256(value.encode()).digest()


def cosine_knn_laplacian(features: np.ndarray, *, neighbors: int = 8) -> np.ndarray:
    """Build the frozen symmetric cosine kNN graph Laplacian.

    Each feature row is L2 normalized. Directed neighbors use descending cosine similarity with
    stable task-index tie breaks. Edge weights are shifted cosine values in ``[0, 1]`` and the
    directed graph is converted to an undirected union by taking the maximum weight.
    """
    if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] < 1:
        raise ValueError("graph features must contain at least two task rows and one column")
    if not np.isfinite(features).all():
        raise ValueError("graph features must be finite")
    task_count = features.shape[0]
    if not 1 <= neighbors < task_count:
        raise ValueError("graph neighbor count must be between one and task_count minus one")
    norms = np.linalg.norm(features, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("graph feature rows must have positive norm")
    normalized = features / norms[:, None]
    similarities = np.clip(normalized @ normalized.T, -1.0, 1.0)
    np.fill_diagonal(similarities, -np.inf)
    nearest = np.argsort(-similarities, axis=1, kind="stable")[:, :neighbors]
    adjacency = np.zeros((task_count, task_count), dtype=np.float64)
    rows = np.repeat(np.arange(task_count, dtype=np.int64), neighbors)
    columns = nearest.ravel()
    weights = np.maximum((similarities[rows, columns] + 1.0) / 2.0, 1e-12)
    adjacency[rows, columns] = weights
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    return np.diag(np.sum(adjacency, axis=1)) - adjacency


def repository_grouped_folds(
    groups: np.ndarray,
    *,
    n_splits: int,
    seed: int,
) -> tuple[RepositoryFold, ...]:
    """Build balanced seeded folds without crossing repository boundaries."""
    values = _validate_groups(groups)
    by_group: dict[str, list[int]] = {}
    for index, group in enumerate(values):
        by_group.setdefault(group, []).append(index)
    if not 2 <= n_splits <= len(by_group):
        raise ValueError("split count must be between two and the repository count")

    ordered = sorted(
        by_group,
        key=lambda group: (
            -len(by_group[group]),
            _seeded_digest(seed, "group", group),
        ),
    )
    fold_groups: list[list[str]] = [[] for _ in range(n_splits)]
    fold_loads = np.zeros(n_splits, dtype=np.int64)
    for group in ordered:
        minimum = int(np.min(fold_loads))
        candidates = np.flatnonzero(fold_loads == minimum)
        fold = min(
            (int(candidate) for candidate in candidates),
            key=lambda candidate: _seeded_digest(
                seed,
                "fold",
                group,
                str(candidate),
            ),
        )
        fold_groups[fold].append(group)
        fold_loads[fold] += len(by_group[group])

    all_indices = np.arange(len(values), dtype=np.int64)
    folds: list[RepositoryFold] = []
    seen = np.zeros(len(values), dtype=np.int64)
    for selected in fold_groups:
        if not selected:
            raise RuntimeError("repository balancing produced an empty fold")
        selected_set = set(selected)
        test = np.asarray(
            [index for index, group in enumerate(values) if group in selected_set],
            dtype=np.int64,
        )
        train = np.setdiff1d(all_indices, test, assume_unique=True)
        if set(np.asarray(values, dtype=object)[train]) & set(
            np.asarray(values, dtype=object)[test]
        ):
            raise AssertionError("repository crossed a grouped fold")
        seen[test] += 1
        folds.append(RepositoryFold(train=train, test=test))
    if not np.all(seen == 1):
        raise RuntimeError("grouped folds do not cover every task exactly once")
    return tuple(folds)


def shuffle_within_repositories(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Permute complete task rows only among tasks from the same repository."""
    group_values = _validate_groups(groups)
    if values.ndim < 1 or values.shape[0] != len(group_values):
        raise ValueError("shuffled values must align with repository groups")
    result = values.copy()
    group_array = np.asarray(group_values, dtype=object)
    rng = np.random.default_rng(seed)
    for group in sorted(set(group_values)):
        indices = np.flatnonzero(group_array == group)
        if len(indices) > 1:
            shift = int(rng.integers(1, len(indices)))
            result[indices] = values[np.roll(indices, shift)]
    return result
