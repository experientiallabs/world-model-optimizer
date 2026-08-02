"""Frozen direct-substitution learner for repository-tree development features.

All fitted estimators and feature rows remain in memory. The module has no network or persistence
surface and is designed for no-internet E2B workers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from coding_model_router_graded_irt_protocol import repository_grouped_folds
from coding_model_router_graded_swerebench_fit import (
    ARMS,
    MIN_SAVINGS,
    QUALITY_RETENTION,
    SEEDS,
    Data,
    _metrics,
)
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

PROTOCOL = "coding-router-graded-repository-tree-fit-v1"
GUARD_INDEX = 5
CHEAP_INDICES = (0, 1, 2, 3, 4)
VIEW_NAMES = ("structure", "localization", "prompt-shape")
VIEW_DIMS = (61, 100, 115)
MAX_LEAVES = (7, 15, 31)
MIN_LEAF_SIZES = (10, 25, 50)
L2_VALUES = (1.0, 10.0)
THRESHOLD_RANKS = tuple(range(20, 81, 5))
FOLDS = 5


@dataclass(frozen=True)
class StructureCandidate:
    """One frozen cheap-arm, feature-view, and tree structure."""

    order: int
    cheap_index: int
    view_index: int
    max_leaves: int
    min_leaf_size: int
    l2: float

    @property
    def key(self) -> str:
        """Return the stable structure identity."""
        return (
            f"{ARMS[self.cheap_index]}-{VIEW_NAMES[self.view_index]}"
            f"-leaves{self.max_leaves}-leaf{self.min_leaf_size}-l2{self.l2:g}"
        )


@dataclass(frozen=True)
class OperatingPoint:
    """One structure and score-rank routing threshold."""

    structure: StructureCandidate
    threshold_rank: int

    @property
    def key(self) -> str:
        """Return the stable operating-point identity."""
        return f"{self.structure.key}-rank{self.threshold_rank}"


@dataclass(frozen=True)
class FeatureData:
    """Aligned in-memory development rows for the three frozen views."""

    data: Data
    languages: tuple[str, ...]
    base_commits: tuple[str, ...]
    views: tuple[np.ndarray, np.ndarray, np.ndarray]


def structure_grid() -> tuple[StructureCandidate, ...]:
    """Return the exact 270 frozen learner structures."""
    values = tuple(
        StructureCandidate(index, cheap, view, leaves, leaf_size, l2)
        for index, (cheap, view, leaves, leaf_size, l2) in enumerate(
            (cheap, view, leaves, leaf_size, l2)
            for cheap in CHEAP_INDICES
            for view in range(len(VIEW_NAMES))
            for leaves in MAX_LEAVES
            for leaf_size in MIN_LEAF_SIZES
            for l2 in L2_VALUES
        )
    )
    if len(values) != 270 or len({value.key for value in values}) != 270:
        raise AssertionError("repository-tree structure grid is incomplete or duplicated")
    return values


def operating_grid() -> tuple[OperatingPoint, ...]:
    """Return the exact 3,510 deterministic operating points."""
    values = tuple(
        OperatingPoint(structure, rank)
        for structure in structure_grid()
        for rank in THRESHOLD_RANKS
    )
    if len(values) != 3_510 or len({value.key for value in values}) != 3_510:
        raise AssertionError("repository-tree operating grid is incomplete or duplicated")
    return values


def align_features(data: Data, rows: Sequence[dict[str, Any]]) -> FeatureData:
    """Join label-free feature rows to development by exact identity only."""
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id or task_id in by_id:
            raise ValueError("feature rows contain an invalid or repeated task identity")
        by_id[task_id] = row
    retained = [index for index, task_id in enumerate(data.task_ids) if task_id in by_id]
    if len(retained) / len(data.task_ids) < 0.95:
        raise ValueError("feature rows miss the frozen development coverage gate")
    task_ids = [data.task_ids[index] for index in retained]
    ordered = [by_id[task_id] for task_id in task_ids]
    repositories = [data.repositories[index] for index in retained]
    for task_id, repository, row in zip(task_ids, repositories, ordered, strict=True):
        if row.get("repository") != repository:
            raise ValueError(f"feature repository mismatch for {task_id}")
    structure = np.asarray([row.get("structure") for row in ordered], dtype=np.float64)
    localization = np.asarray([row.get("localization") for row in ordered], dtype=np.float64)
    prompt = np.asarray([row.get("prompt_shape") for row in ordered], dtype=np.float64)
    views = (
        structure,
        np.concatenate([structure, localization], axis=1),
        np.concatenate([structure, localization, prompt], axis=1),
    )
    if any(
        view.shape != (len(retained), VIEW_DIMS[index])
        or not np.isfinite(view).all()
        for index, view in enumerate(views)
    ):
        raise ValueError("feature views have invalid frozen dimensions or values")
    retained_data = Data(
        task_ids=task_ids,
        repositories=repositories,
        texts=[data.texts[index] for index in retained],
        rewards=data.rewards[retained].copy(),
        costs=data.costs[retained].copy(),
        rough_cumulative_spend_usd=data.rough_cumulative_spend_usd,
    )
    languages = tuple(str(row.get("language", "")) for row in ordered)
    commits = tuple(str(row.get("base_commit", "")) for row in ordered)
    if any(not value for value in (*languages, *commits)):
        raise ValueError("feature rows lack language or base commit identity")
    return FeatureData(retained_data, languages, commits, views)


def cross_fitted_scores(
    features: FeatureData,
    candidate: StructureCandidate,
    *,
    seed: int,
    fit_rewards: np.ndarray | None = None,
) -> np.ndarray:
    """Predict Sol-minus-cheap reward out of fold for every retained task."""
    data = features.data
    rewards = data.rewards if fit_rewards is None else fit_rewards
    if rewards.shape != data.rewards.shape or not np.isfinite(rewards).all():
        raise ValueError("fit rewards must be a complete aligned matrix")
    matrix = features.views[candidate.view_index]
    target = rewards[:, GUARD_INDEX] - rewards[:, candidate.cheap_index]
    scores = np.full(len(data.task_ids), np.nan, dtype=np.float64)
    folds = repository_grouped_folds(
        np.asarray(data.repositories, dtype=object), n_splits=FOLDS, seed=seed
    )
    for fold in folds:
        scaler = StandardScaler().fit(matrix[fold.train])
        train = scaler.transform(matrix[fold.train])
        test = scaler.transform(matrix[fold.test])
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=candidate.max_leaves,
            min_samples_leaf=candidate.min_leaf_size,
            l2_regularization=candidate.l2,
            early_stopping=False,
            random_state=seed,
        ).fit(train, target[fold.train])
        scores[fold.test] = model.predict(test)
    if scores.shape != (len(data.task_ids),) or not np.isfinite(scores).all():
        raise RuntimeError("cross-fitted repository-tree scores are incomplete")
    return scores


def ranked_choices(
    task_ids: Sequence[str],
    scores: np.ndarray,
    *,
    cheap_index: int,
    threshold_rank: int,
) -> np.ndarray:
    """Turn a score percentile into deterministic complete two-arm routes."""
    if (
        scores.shape != (len(task_ids),)
        or not np.isfinite(scores).all()
        or cheap_index not in CHEAP_INDICES
        or threshold_rank not in THRESHOLD_RANKS
    ):
        raise ValueError("ranked route inputs are invalid")
    guard_count = int(np.ceil(len(task_ids) * (100 - threshold_rank) / 100.0))
    order = sorted(range(len(task_ids)), key=lambda index: (-scores[index], task_ids[index]))
    choices = np.full(len(task_ids), cheap_index, dtype=np.int64)
    choices[np.asarray(order[:guard_count], dtype=np.int64)] = GUARD_INDEX
    return choices


def evaluate_structure_seed(
    features: FeatureData,
    candidate: StructureCandidate,
    *,
    seed: int,
    fit_rewards: np.ndarray | None = None,
) -> tuple[dict[str, Any], ...]:
    """Evaluate all 13 routing ranks for one structure and grouped seed."""
    scores = cross_fitted_scores(
        features, candidate, seed=seed, fit_rewards=fit_rewards
    )
    return tuple(
        {
            "key": OperatingPoint(candidate, rank).key,
            "structure_order": candidate.order,
            "threshold_rank": rank,
            "seed": seed,
            **_metrics(
                features.data,
                ranked_choices(
                    features.data.task_ids,
                    scores,
                    cheap_index=candidate.cheap_index,
                    threshold_rank=rank,
                ),
            ),
        }
        for rank in THRESHOLD_RANKS
    )


def passes_primary_gates(seed_metrics: Sequence[dict[str, Any]]) -> bool:
    """Apply the frozen gates independently to every grouped seed."""
    return (
        len(seed_metrics) == len(SEEDS)
        and {int(row["seed"]) for row in seed_metrics} == set(SEEDS)
        and all(float(row["quality_retention"]) >= QUALITY_RETENTION for row in seed_metrics)
        and all(float(row["cost_savings"]) >= MIN_SAVINGS for row in seed_metrics)
        and all(float(row["matched_blind_advantage"]) > 0.0 for row in seed_metrics)
        and all(not row["dominated_by_static"] for row in seed_metrics)
    )
