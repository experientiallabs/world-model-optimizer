"""Run the conditional graded IRT study as an in-memory nested protocol.

This module combines the preregistered numeric, grouping, and robust-selection helpers. It has no
filesystem or serialization surface. Fitted coefficients and task probabilities remain inside the
remote process, while callers may retain only aggregate metrics and label-free arm choices.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from coding_model_router_graded_irt_core import (
    fit_projected_binomial_irt,
    predict_projected_probabilities,
)
from coding_model_router_graded_irt_protocol import (
    cosine_knn_laplacian,
    repository_grouped_folds,
    shuffle_within_repositories,
)
from coding_model_router_graded_irt_selection import (
    quality_guarded_choices,
    repository_robust_margin,
)

ARM_COUNT = 6
INNER_FOLDS = 5
OUTER_SEEDS = (11, 23, 37, 41, 59)
FEATURE_NAMES = (
    "signed-hash-512",
    "signed-hash-2048",
    "prompt-shape",
    "combined",
)
LATENT_DIMENSIONS = (2,)
REGULARIZATION_STRENGTHS = (0.1, 1.0, 10.0, 100.0)
GRAPH_STRENGTHS = (0.01, 0.1, 1.0)
COST_PENALTIES = (0.0, 0.005, 0.01, 0.02, 0.03)
KL_RADII = (0.0, 0.01, 0.03, 0.05, 0.1)
QUALITY_FLOOR = 0.95
MIN_SAVINGS = 0.40
MAX_REPOSITORY_LOSS = 0.10
MAX_ROUTE_P50_MS = 5.0
MAX_ROUTE_P95_MS = 20.0
MIN_ROUTE_DECISIONS = 10_000


@dataclass(frozen=True)
class IrtStructure:
    """One frozen feature-conditioned IRT structure."""

    order: int
    feature_name: str
    latent_dimension: int
    regularization: float
    monotone_luna: bool
    graph_l2: float

    @property
    def key(self) -> str:
        """Return a stable candidate identity."""
        ability = "monotone" if self.monotone_luna else "free"
        graph = f"graph{self.graph_l2:g}" if self.graph_l2 else "nograph"
        return (
            f"{self.feature_name}-d{self.latent_dimension}-r{self.regularization:g}-"
            f"{ability}-{graph}"
        )

    def coefficient_count(self, feature_count: int) -> int:
        """Return the ephemeral fitted coefficient count for tie breaking."""
        augmented = feature_count + 1
        return (
            ARM_COUNT * self.latent_dimension
            + augmented
            + augmented * self.latent_dimension
        )


@dataclass(frozen=True)
class OperatingPoint:
    """One frozen cost and distribution-shift operating point."""

    order: int
    cost_penalty: float
    kl_radius: float

    @property
    def key(self) -> str:
        """Return a stable operating-point identity."""
        return f"penalty{self.cost_penalty:g}-kl{self.kl_radius:g}"


@dataclass(frozen=True)
class CrossfitPredictions:
    """Ephemeral grouped out-of-fold predictions for one structure and seed."""

    probabilities: np.ndarray
    shuffled_probabilities: np.ndarray
    decision_costs: np.ndarray
    guard_arms: np.ndarray
    fit_iterations: int
    shuffled_fit_iterations: int


@dataclass(frozen=True)
class PolicyMetric:
    """Aggregate fit-only evidence for one seed and policy candidate."""

    seed: int
    structure_key: str
    operating_key: str
    structure_order: int
    operating_order: int
    coefficient_count: int
    latent_dimension: int
    cost_penalty: float
    kl_radius: float
    reward: float
    cost_per_task: float
    quality_retention: float
    cost_savings: float
    matched_blind_advantage: float
    shuffled_label_advantage: float
    robust_quality_margin: float
    robust_cost_margin: float
    worst_large_repository_loss: float
    dominated_by_static: bool
    eligible: bool

    @property
    def key(self) -> str:
        """Return the joint structure and operating-point identity."""
        return f"{self.structure_key}__{self.operating_key}"


@dataclass(frozen=True)
class RouteLatencyMetric:
    """One single-core online route audit with no inference-time network call."""

    p50_ms: float
    p95_ms: float
    decisions: int
    network_calls: int

    @property
    def eligible(self) -> bool:
        """Return whether the frozen latency and network gates pass."""
        return (
            np.isfinite(self.p50_ms)
            and np.isfinite(self.p95_ms)
            and self.p50_ms >= 0.0
            and self.p95_ms >= self.p50_ms
            and self.p50_ms < MAX_ROUTE_P50_MS
            and self.p95_ms < MAX_ROUTE_P95_MS
            and self.decisions >= MIN_ROUTE_DECISIONS
            and self.network_calls == 0
        )


@dataclass(frozen=True)
class NestedSelectionResult:
    """Five-seed fit-only selection output without fitted numeric state."""

    selected_key: str | None
    metrics: tuple[PolicyMetric, ...]
    eligible_count: int


@dataclass(frozen=True)
class FullRouteResult:
    """Label-free routes and aggregate diagnostics from one ephemeral full fit."""

    choices: np.ndarray
    guard_arm: int
    loss: float
    iterations: int
    coefficient_count: int


def frozen_structures() -> tuple[IrtStructure, ...]:
    """Enumerate the preregistered unconstrained, monotone, and graph ablations."""
    values: list[IrtStructure] = []
    for feature_name in FEATURE_NAMES:
        for latent_dimension in LATENT_DIMENSIONS:
            for regularization in REGULARIZATION_STRENGTHS:
                variants = ((False, 0.0), (True, 0.0)) + tuple(
                    (False, graph_l2) for graph_l2 in GRAPH_STRENGTHS
                )
                for monotone_luna, graph_l2 in variants:
                    values.append(
                        IrtStructure(
                            order=len(values),
                            feature_name=feature_name,
                            latent_dimension=latent_dimension,
                            regularization=regularization,
                            monotone_luna=monotone_luna,
                            graph_l2=graph_l2,
                        )
                    )
    result = tuple(values)
    if len(result) != 80 or len({value.key for value in result}) != len(result):
        raise AssertionError("frozen IRT structure grid is incomplete or duplicated")
    return result


def frozen_operating_points() -> tuple[OperatingPoint, ...]:
    """Enumerate the preregistered cost-penalty and forward-KL grid."""
    result = tuple(
        OperatingPoint(order, cost_penalty, kl_radius)
        for order, (cost_penalty, kl_radius) in enumerate(
            (cost_penalty, kl_radius)
            for cost_penalty in COST_PENALTIES
            for kl_radius in KL_RADII
        )
    )
    if len(result) != 25 or len({value.key for value in result}) != len(result):
        raise AssertionError("frozen IRT operating-point grid is incomplete or duplicated")
    return result


def _validate_matrix_inputs(
    passed: np.ndarray,
    total: np.ndarray,
    costs: np.ndarray,
    repositories: np.ndarray,
) -> np.ndarray:
    """Validate a dense six-arm development matrix and return task rewards."""
    if (
        passed.ndim != 2
        or passed.shape[1] != ARM_COUNT
        or total.shape != passed.shape
        or costs.shape != passed.shape
        or repositories.shape != (len(passed),)
    ):
        raise ValueError("graded development inputs must be aligned six-arm matrices")
    if not len(passed) or not np.isfinite(passed).all() or not np.isfinite(total).all():
        raise ValueError("graded counts must be nonempty and finite")
    if np.any(total <= 0.0) or np.any(passed < 0.0) or np.any(passed > total):
        raise ValueError("graded counts require 0 <= passed <= positive total")
    if not np.isfinite(costs).all() or np.any(costs < 0.0):
        raise ValueError("development costs must be finite and nonnegative")
    if any(not str(value) for value in repositories):
        raise ValueError("repository identities must be nonempty")
    return passed / total


def _best_static_arm(rewards: np.ndarray, costs: np.ndarray) -> int:
    """Select the strongest static arm with cost and frozen-order tie breaks."""
    mean_rewards = np.mean(rewards, axis=0)
    mean_costs = np.mean(costs, axis=0)
    return min(
        range(rewards.shape[1]),
        key=lambda arm: (-mean_rewards[arm], mean_costs[arm], arm),
    )


def crossfit_structure(
    features: np.ndarray,
    passed: np.ndarray,
    total: np.ndarray,
    costs: np.ndarray,
    repositories: np.ndarray,
    *,
    structure: IrtStructure,
    seed: int,
    n_splits: int = INNER_FOLDS,
) -> CrossfitPredictions:
    """Fit one structure inside seeded repository folds and predict every task once."""
    rewards = _validate_matrix_inputs(passed, total, costs, repositories)
    if features.ndim != 2 or features.shape[0] != len(passed) or not features.shape[1]:
        raise ValueError("pre-call features must align with development tasks")
    if not np.isfinite(features).all():
        raise ValueError("pre-call features must be finite")
    probabilities = np.full(passed.shape, np.nan, dtype=np.float64)
    shuffled_probabilities = np.full_like(probabilities, np.nan)
    decision_costs = np.full_like(probabilities, np.nan)
    guard_arms = np.full(len(passed), -1, dtype=np.int64)
    fit_iterations = 0
    shuffled_fit_iterations = 0
    for fold_index, fold in enumerate(
        repository_grouped_folds(repositories, n_splits=n_splits, seed=seed)
    ):
        train_features = features[fold.train]
        graph_laplacian = None
        if structure.graph_l2 > 0.0:
            if len(fold.train) <= 8:
                raise ValueError("frozen eight-neighbor graph needs at least nine fit tasks")
            graph_laplacian = cosine_knn_laplacian(train_features, neighbors=8)
        fit = fit_projected_binomial_irt(
            train_features,
            passed[fold.train],
            total[fold.train],
            structure.latent_dimension,
            projection_l2=structure.regularization,
            monotone_luna=structure.monotone_luna,
            graph_laplacian=graph_laplacian,
            graph_l2=structure.graph_l2,
        )
        probabilities[fold.test] = predict_projected_probabilities(
            fit,
            features[fold.test],
        )
        fit_iterations += fit.iterations

        count_rows = np.concatenate(
            [passed[fold.train], total[fold.train]],
            axis=1,
        )
        shuffled_rows = shuffle_within_repositories(
            count_rows,
            repositories[fold.train],
            seed=seed * 10_000 + fold_index,
        )
        shuffled_fit = fit_projected_binomial_irt(
            train_features,
            shuffled_rows[:, :ARM_COUNT],
            shuffled_rows[:, ARM_COUNT:],
            structure.latent_dimension,
            projection_l2=structure.regularization,
            monotone_luna=structure.monotone_luna,
            graph_laplacian=graph_laplacian,
            graph_l2=structure.graph_l2,
        )
        shuffled_probabilities[fold.test] = predict_projected_probabilities(
            shuffled_fit,
            features[fold.test],
        )
        shuffled_fit_iterations += shuffled_fit.iterations
        decision_costs[fold.test] = np.mean(costs[fold.train], axis=0)
        guard_arms[fold.test] = _best_static_arm(
            rewards[fold.train],
            costs[fold.train],
        )
    if (
        not np.isfinite(probabilities).all()
        or not np.isfinite(shuffled_probabilities).all()
        or not np.isfinite(decision_costs).all()
        or np.any(guard_arms < 0)
    ):
        raise RuntimeError("nested grouped predictions are incomplete")
    return CrossfitPredictions(
        probabilities=probabilities,
        shuffled_probabilities=shuffled_probabilities,
        decision_costs=decision_costs,
        guard_arms=guard_arms,
        fit_iterations=fit_iterations,
        shuffled_fit_iterations=shuffled_fit_iterations,
    )


def _large_repository_loss(
    routed_rewards: np.ndarray,
    guard_rewards: np.ndarray,
    repositories: np.ndarray,
) -> float:
    """Return the largest reward loss among repositories with at least five tasks."""
    losses = [
        float(np.mean(guard_rewards[repositories == repository]))
        - float(np.mean(routed_rewards[repositories == repository]))
        for repository in sorted({str(value) for value in repositories})
        if int(np.sum(repositories == repository)) >= 5
    ]
    return max(losses, default=0.0)


def evaluate_policy(
    crossfit: CrossfitPredictions,
    passed: np.ndarray,
    total: np.ndarray,
    costs: np.ndarray,
    repositories: np.ndarray,
    *,
    structure: IrtStructure,
    operating_point: OperatingPoint,
    seed: int,
    feature_count: int,
) -> PolicyMetric:
    """Apply the frozen guard and robust gates to one out-of-fold policy."""
    rewards = _validate_matrix_inputs(passed, total, costs, repositories)
    if (
        crossfit.probabilities.shape != rewards.shape
        or crossfit.shuffled_probabilities.shape != rewards.shape
        or crossfit.decision_costs.shape != rewards.shape
        or crossfit.guard_arms.shape != (len(rewards),)
    ):
        raise ValueError("crossfit predictions do not align with the development matrix")
    choices = np.empty(len(rewards), dtype=np.int64)
    shuffled_choices = np.empty_like(choices)
    for guard_arm in sorted(set(int(value) for value in crossfit.guard_arms)):
        selected = np.flatnonzero(crossfit.guard_arms == guard_arm)
        choices[selected] = quality_guarded_choices(
            crossfit.probabilities[selected],
            crossfit.decision_costs[selected],
            guard_arm=guard_arm,
            cost_penalty=operating_point.cost_penalty,
            quality_floor=QUALITY_FLOOR,
        )
        shuffled_choices[selected] = quality_guarded_choices(
            crossfit.shuffled_probabilities[selected],
            crossfit.decision_costs[selected],
            guard_arm=guard_arm,
            cost_penalty=operating_point.cost_penalty,
            quality_floor=QUALITY_FLOOR,
        )
    rows = np.arange(len(rewards), dtype=np.int64)
    routed_rewards = rewards[rows, choices]
    routed_costs = costs[rows, choices]
    guard_rewards = rewards[rows, crossfit.guard_arms]
    guard_costs = costs[rows, crossfit.guard_arms]
    shuffled_rewards = rewards[rows, shuffled_choices]
    traffic = np.bincount(choices, minlength=ARM_COUNT).astype(np.float64) / len(choices)
    matched_blind = rewards @ traffic
    reward = float(np.mean(routed_rewards))
    cost = float(np.mean(routed_costs))
    guard_reward = float(np.mean(guard_rewards))
    guard_cost = float(np.mean(guard_costs))
    static_rewards = np.mean(rewards, axis=0)
    static_costs = np.mean(costs, axis=0)
    dominated = any(
        static_rewards[arm] >= reward
        and static_costs[arm] <= cost
        and (static_rewards[arm] > reward or static_costs[arm] < cost)
        for arm in range(ARM_COUNT)
    )
    robust_quality = repository_robust_margin(
        routed_rewards - QUALITY_FLOOR * guard_rewards,
        repositories,
        radius=operating_point.kl_radius,
    ).lower_bound
    robust_cost = repository_robust_margin(
        (1.0 - MIN_SAVINGS) * guard_costs - routed_costs,
        repositories,
        radius=operating_point.kl_radius,
    ).lower_bound
    quality_retention = reward / guard_reward if guard_reward > 0.0 else 0.0
    cost_savings = 1.0 - cost / guard_cost if guard_cost > 0.0 else 0.0
    matched_advantage = float(np.mean(routed_rewards - matched_blind))
    shuffled_advantage = float(np.mean(routed_rewards - shuffled_rewards))
    worst_loss = _large_repository_loss(
        routed_rewards,
        guard_rewards,
        repositories,
    )
    eligible = (
        quality_retention >= QUALITY_FLOOR
        and cost_savings >= MIN_SAVINGS
        and matched_advantage > 0.0
        and shuffled_advantage > 0.0
        and robust_quality >= 0.0
        and robust_cost >= 0.0
        and worst_loss <= MAX_REPOSITORY_LOSS
        and not dominated
    )
    return PolicyMetric(
        seed=seed,
        structure_key=structure.key,
        operating_key=operating_point.key,
        structure_order=structure.order,
        operating_order=operating_point.order,
        coefficient_count=structure.coefficient_count(feature_count),
        latent_dimension=structure.latent_dimension,
        cost_penalty=operating_point.cost_penalty,
        kl_radius=operating_point.kl_radius,
        reward=reward,
        cost_per_task=cost,
        quality_retention=quality_retention,
        cost_savings=cost_savings,
        matched_blind_advantage=matched_advantage,
        shuffled_label_advantage=shuffled_advantage,
        robust_quality_margin=robust_quality,
        robust_cost_margin=robust_cost,
        worst_large_repository_loss=worst_loss,
        dominated_by_static=dominated,
        eligible=eligible and operating_point.kl_radius > 0.0,
    )


def select_nested_policy(
    feature_views: Mapping[str, np.ndarray],
    passed: np.ndarray,
    total: np.ndarray,
    costs: np.ndarray,
    repositories: np.ndarray,
    *,
    structures: Sequence[IrtStructure] | None = None,
    operating_points: Sequence[OperatingPoint] | None = None,
    seeds: Sequence[int] = OUTER_SEEDS,
    route_latency: Mapping[str, RouteLatencyMetric] | None = None,
) -> NestedSelectionResult:
    """Run the full fit-only nested search and select one common five-seed winner."""
    selected_structures = (
        frozen_structures() if structures is None else tuple(structures)
    )
    selected_points = (
        frozen_operating_points()
        if operating_points is None
        else tuple(operating_points)
    )
    selected_seeds = tuple(int(seed) for seed in seeds)
    if not selected_structures or not selected_points or not selected_seeds:
        raise ValueError("nested selection requires structures, operating points, and seeds")
    metrics = tuple(
        metric
        for seed in selected_seeds
        for metric in evaluate_nested_seed(
            feature_views,
            passed,
            total,
            costs,
            repositories,
            structures=selected_structures,
            operating_points=selected_points,
            seed=seed,
        )
    )
    return select_nested_metrics(
        metrics,
        structures=selected_structures,
        operating_points=selected_points,
        seeds=selected_seeds,
        route_latency=route_latency,
    )


def evaluate_nested_seed(
    feature_views: Mapping[str, np.ndarray],
    passed: np.ndarray,
    total: np.ndarray,
    costs: np.ndarray,
    repositories: np.ndarray,
    *,
    seed: int,
    structures: Sequence[IrtStructure] | None = None,
    operating_points: Sequence[OperatingPoint] | None = None,
) -> tuple[PolicyMetric, ...]:
    """Evaluate one independently shardable seed and return aggregate metrics only."""
    selected_structures = (
        frozen_structures() if structures is None else tuple(structures)
    )
    selected_points = (
        frozen_operating_points()
        if operating_points is None
        else tuple(operating_points)
    )
    if not selected_structures or not selected_points:
        raise ValueError("seed evaluation requires structures and operating points")
    metrics: list[PolicyMetric] = []
    for structure in selected_structures:
        if structure.feature_name not in feature_views:
            raise ValueError(f"missing frozen feature view: {structure.feature_name}")
        features = feature_views[structure.feature_name]
        crossfit = crossfit_structure(
            features,
            passed,
            total,
            costs,
            repositories,
            structure=structure,
            seed=seed,
        )
        metrics.extend(
            evaluate_policy(
                crossfit,
                passed,
                total,
                costs,
                repositories,
                structure=structure,
                operating_point=operating_point,
                seed=seed,
                feature_count=features.shape[1],
            )
            for operating_point in selected_points
        )
    return tuple(metrics)


def select_nested_metrics(
    metrics: Sequence[PolicyMetric],
    *,
    structures: Sequence[IrtStructure] | None = None,
    operating_points: Sequence[OperatingPoint] | None = None,
    seeds: Sequence[int] = OUTER_SEEDS,
    route_latency: Mapping[str, RouteLatencyMetric] | None = None,
) -> NestedSelectionResult:
    """Select from complete aggregate seed shards without any fitted numeric state."""
    selected_structures = (
        frozen_structures() if structures is None else tuple(structures)
    )
    selected_points = (
        frozen_operating_points()
        if operating_points is None
        else tuple(operating_points)
    )
    selected_seeds = tuple(int(seed) for seed in seeds)
    metric_values = tuple(metrics)
    if not selected_structures or not selected_points or not selected_seeds:
        raise ValueError("nested selection requires structures, operating points, and seeds")
    expected_keys = {
        f"{structure.key}__{operating_point.key}"
        for structure in selected_structures
        for operating_point in selected_points
    }
    by_key: dict[str, list[PolicyMetric]] = {}
    for metric in metric_values:
        by_key.setdefault(metric.key, []).append(metric)
    expected_seed_set = set(selected_seeds)
    if set(by_key) != expected_keys or any(
        len(values) != len(selected_seeds)
        or {value.seed for value in values} != expected_seed_set
        for values in by_key.values()
    ):
        raise ValueError("aggregate seed shards are incomplete, duplicated, or unexpected")
    eligible = [
        values
        for values in by_key.values()
        if all(value.eligible for value in values)
    ]
    latency = route_latency or {}
    latency_eligible = [
        values
        for values in eligible
        if values[0].key in latency and latency[values[0].key].eligible
    ]
    winner = None
    if latency_eligible:
        winner = min(
            latency_eligible,
            key=lambda values: (
                float(np.mean([value.cost_per_task for value in values])),
                -min(value.quality_retention for value in values),
                -min(value.matched_blind_advantage for value in values),
                -values[0].kl_radius,
                latency[values[0].key].p95_ms,
                values[0].coefficient_count,
                values[0].latent_dimension,
                values[0].structure_order,
                values[0].operating_order,
            ),
        )[0].key
    return NestedSelectionResult(
        selected_key=winner,
        metrics=metric_values,
        eligible_count=len(eligible),
    )


def fit_full_routes(
    development_features: np.ndarray,
    target_features: np.ndarray,
    passed: np.ndarray,
    total: np.ndarray,
    costs: np.ndarray,
    *,
    structure: IrtStructure,
    operating_point: OperatingPoint,
) -> FullRouteResult:
    """Fit ephemerally on all development rows and return label-free target choices."""
    repositories = np.asarray(["development"] * len(passed), dtype=object)
    rewards = _validate_matrix_inputs(passed, total, costs, repositories)
    if (
        development_features.ndim != 2
        or target_features.ndim != 2
        or development_features.shape[0] != len(passed)
        or development_features.shape[1] != target_features.shape[1]
        or not np.isfinite(development_features).all()
        or not np.isfinite(target_features).all()
    ):
        raise ValueError("development and target feature matrices must be finite and aligned")
    graph_laplacian = None
    if structure.graph_l2 > 0.0:
        graph_laplacian = cosine_knn_laplacian(development_features, neighbors=8)
    fit = fit_projected_binomial_irt(
        development_features,
        passed,
        total,
        structure.latent_dimension,
        projection_l2=structure.regularization,
        monotone_luna=structure.monotone_luna,
        graph_laplacian=graph_laplacian,
        graph_l2=structure.graph_l2,
    )
    probabilities = predict_projected_probabilities(fit, target_features)
    guard_arm = _best_static_arm(rewards, costs)
    mean_costs = np.mean(costs, axis=0)
    choices = quality_guarded_choices(
        probabilities,
        np.repeat(mean_costs[None, :], len(target_features), axis=0),
        guard_arm=guard_arm,
        cost_penalty=operating_point.cost_penalty,
        quality_floor=QUALITY_FLOOR,
    )
    return FullRouteResult(
        choices=choices,
        guard_arm=guard_arm,
        loss=fit.loss,
        iterations=fit.iterations,
        coefficient_count=structure.coefficient_count(development_features.shape[1]),
    )
