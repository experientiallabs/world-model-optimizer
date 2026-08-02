"""Tests for the in-memory conditional graded IRT nested protocol."""

from __future__ import annotations

import coding_model_router_graded_irt_nested as nested
import numpy as np
import pytest
from coding_model_router_graded_irt_nested import (
    CrossfitPredictions,
    IrtStructure,
    OperatingPoint,
    RouteLatencyMetric,
    evaluate_nested_seed,
    evaluate_policy,
    frozen_operating_points,
    frozen_structures,
    select_nested_metrics,
    select_nested_policy,
)


def _structure() -> IrtStructure:
    return IrtStructure(
        order=0,
        feature_name="signed-hash-512",
        latent_dimension=2,
        regularization=0.1,
        monotone_luna=False,
        graph_l2=0.0,
    )


def _operating_point() -> OperatingPoint:
    return OperatingPoint(order=0, cost_penalty=0.03, kl_radius=0.01)


def _latency() -> RouteLatencyMetric:
    return RouteLatencyMetric(
        p50_ms=0.5,
        p95_ms=1.0,
        decisions=10_000,
        network_calls=0,
    )


def _fixture() -> tuple[
    CrossfitPredictions,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rewards = np.full((10, 6), 0.5, dtype=np.float64)
    rewards[:, 5] = 0.9
    strong = np.asarray([0, 1, 2, 5, 6, 7], dtype=np.int64)
    weak = np.asarray([3, 4, 8, 9], dtype=np.int64)
    rewards[strong, 0] = 1.0
    rewards[weak, 0] = 0.8
    total = np.full_like(rewards, 10.0)
    passed = rewards * total
    costs = np.repeat(
        np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0, 10.0]], dtype=np.float64),
        len(rewards),
        axis=0,
    )
    probabilities = np.full_like(rewards, 0.5)
    probabilities[:, 5] = 0.9
    probabilities[strong, 0] = 0.95
    probabilities[weak, 0] = 0.8
    shuffled = probabilities.copy()
    shuffled[strong, 0] = 0.8
    shuffled[weak, 0] = 0.95
    crossfit = CrossfitPredictions(
        probabilities=probabilities,
        shuffled_probabilities=shuffled,
        decision_costs=costs,
        guard_arms=np.full(len(rewards), 5, dtype=np.int64),
        fit_iterations=25,
        shuffled_fit_iterations=25,
    )
    repositories = np.asarray(["a"] * 5 + ["b"] * 5, dtype=object)
    return crossfit, passed, total, costs, repositories


def test_frozen_grid_is_complete_unique_and_ordered() -> None:
    structures = frozen_structures()
    points = frozen_operating_points()
    assert len(structures) == 80
    assert len({value.key for value in structures}) == 80
    assert [value.order for value in structures] == list(range(80))
    assert len(points) == 25
    assert len({value.key for value in points}) == 25
    assert [value.order for value in points] == list(range(25))
    assert sum(value.monotone_luna for value in structures) == 16
    assert sum(value.graph_l2 > 0.0 for value in structures) == 48


def test_policy_gate_rewards_task_signal_over_identical_traffic() -> None:
    crossfit, passed, total, costs, repositories = _fixture()
    metric = evaluate_policy(
        crossfit,
        passed,
        total,
        costs,
        repositories,
        structure=_structure(),
        operating_point=_operating_point(),
        seed=11,
        feature_count=512,
    )
    assert metric.reward == pytest.approx(0.96)
    assert metric.cost_per_task == pytest.approx(4.6)
    assert metric.cost_savings == pytest.approx(0.54)
    assert metric.matched_blind_advantage == pytest.approx(0.048)
    assert metric.shuffled_label_advantage == pytest.approx(0.10)
    assert metric.robust_quality_margin > 0.0
    assert metric.robust_cost_margin > 0.0
    assert metric.eligible


def test_nominal_radius_is_reported_but_cannot_promote() -> None:
    crossfit, passed, total, costs, repositories = _fixture()
    metric = evaluate_policy(
        crossfit,
        passed,
        total,
        costs,
        repositories,
        structure=_structure(),
        operating_point=OperatingPoint(order=0, cost_penalty=0.03, kl_radius=0.0),
        seed=11,
        feature_count=512,
    )
    assert metric.robust_quality_margin > 0.0
    assert metric.robust_cost_margin > 0.0
    assert not metric.eligible


def test_crossfit_predicts_every_task_from_repository_disjoint_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Fit:
        iterations = 1

    fit_rows: list[set[int]] = []
    predicted_rows: list[int] = []

    def fake_fit(
        feature_values: np.ndarray,
        passed_values: np.ndarray,
        total_values: np.ndarray,
        latent_dimension: int,
        *,
        projection_l2: float,
        monotone_luna: bool,
        graph_laplacian: np.ndarray | None,
        graph_l2: float,
    ) -> _Fit:
        del (
            passed_values,
            total_values,
            latent_dimension,
            projection_l2,
            monotone_luna,
            graph_laplacian,
            graph_l2,
        )
        fit_rows.append({int(value) for value in feature_values[:, 0]})
        return _Fit()

    def fake_predict(fit: _Fit, feature_values: np.ndarray) -> np.ndarray:
        del fit
        predicted_rows.extend(int(value) for value in feature_values[:, 0])
        return np.repeat(
            np.linspace(0.4, 0.9, 6, dtype=np.float64)[None, :],
            len(feature_values),
            axis=0,
        )

    monkeypatch.setattr(nested, "fit_projected_binomial_irt", fake_fit)
    monkeypatch.setattr(nested, "predict_projected_probabilities", fake_predict)
    task_count = 20
    features = np.arange(task_count, dtype=np.float64)[:, None]
    total = np.full((task_count, 6), 10.0, dtype=np.float64)
    passed = np.repeat(
        np.asarray([[4.0, 5.0, 6.0, 7.0, 8.0, 9.0]], dtype=np.float64),
        task_count,
        axis=0,
    )
    costs = np.repeat(
        np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0, 10.0]], dtype=np.float64),
        task_count,
        axis=0,
    )
    repositories = np.asarray(
        [repository for repository in "abcdefghij" for _ in range(2)],
        dtype=object,
    )
    result = nested.crossfit_structure(
        features,
        passed,
        total,
        costs,
        repositories,
        structure=_structure(),
        seed=11,
    )
    assert len(fit_rows) == 10
    assert all(len(rows) == 16 for rows in fit_rows)
    assert sorted(predicted_rows) == sorted(list(range(task_count)) * 2)
    assert result.fit_iterations == 5
    assert result.shuffled_fit_iterations == 5
    assert np.isfinite(result.probabilities).all()
    assert np.isfinite(result.shuffled_probabilities).all()


def test_nested_selection_requires_one_policy_to_pass_every_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crossfit, passed, total, costs, repositories = _fixture()

    def fake_crossfit(
        features: np.ndarray,
        passed_values: np.ndarray,
        total_values: np.ndarray,
        cost_values: np.ndarray,
        repository_values: np.ndarray,
        *,
        structure: IrtStructure,
        seed: int,
        n_splits: int = nested.INNER_FOLDS,
    ) -> CrossfitPredictions:
        del (
            features,
            passed_values,
            total_values,
            cost_values,
            repository_values,
            structure,
            seed,
            n_splits,
        )
        return crossfit

    monkeypatch.setattr(nested, "crossfit_structure", fake_crossfit)
    stronger_radius = OperatingPoint(order=1, cost_penalty=0.03, kl_radius=0.1)
    result = select_nested_policy(
        {"signed-hash-512": np.ones((10, 2), dtype=np.float64)},
        passed,
        total,
        costs,
        repositories,
        structures=(_structure(),),
        operating_points=(_operating_point(), stronger_radius),
        route_latency={
            f"{_structure().key}__{_operating_point().key}": _latency(),
            f"{_structure().key}__{stronger_radius.key}": _latency(),
        },
    )
    assert result.selected_key == f"{_structure().key}__{stronger_radius.key}"
    assert result.eligible_count == 2
    assert len(result.metrics) == 10
    assert all(metric.eligible for metric in result.metrics)


def test_one_failed_seed_prevents_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    crossfit, passed, total, costs, repositories = _fixture()

    def seed_sensitive_crossfit(
        features: np.ndarray,
        passed_values: np.ndarray,
        total_values: np.ndarray,
        cost_values: np.ndarray,
        repository_values: np.ndarray,
        *,
        structure: IrtStructure,
        seed: int,
        n_splits: int = nested.INNER_FOLDS,
    ) -> CrossfitPredictions:
        del (
            features,
            passed_values,
            total_values,
            cost_values,
            repository_values,
            structure,
            n_splits,
        )
        if seed != 59:
            return crossfit
        return CrossfitPredictions(
            probabilities=crossfit.probabilities,
            shuffled_probabilities=crossfit.probabilities,
            decision_costs=crossfit.decision_costs,
            guard_arms=crossfit.guard_arms,
            fit_iterations=crossfit.fit_iterations,
            shuffled_fit_iterations=crossfit.shuffled_fit_iterations,
        )

    monkeypatch.setattr(nested, "crossfit_structure", seed_sensitive_crossfit)
    result = select_nested_policy(
        {"signed-hash-512": np.ones((10, 2), dtype=np.float64)},
        passed,
        total,
        costs,
        repositories,
        structures=(_structure(),),
        operating_points=(_operating_point(),),
        route_latency={
            f"{_structure().key}__{_operating_point().key}": _latency(),
        },
    )
    assert result.selected_key is None
    assert result.eligible_count == 0
    assert len(result.metrics) == 5


def test_complete_seed_shards_select_without_fitted_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crossfit, passed, total, costs, repositories = _fixture()

    def fake_crossfit(*args: object, **kwargs: object) -> CrossfitPredictions:
        del args, kwargs
        return crossfit

    monkeypatch.setattr(nested, "crossfit_structure", fake_crossfit)
    feature_views = {"signed-hash-512": np.ones((10, 2), dtype=np.float64)}
    seeds = (11, 23)
    shards = tuple(
        metric
        for seed in seeds
        for metric in evaluate_nested_seed(
            feature_views,
            passed,
            total,
            costs,
            repositories,
            structures=(_structure(),),
            operating_points=(_operating_point(),),
            seed=seed,
        )
    )
    result = select_nested_metrics(
        shards,
        structures=(_structure(),),
        operating_points=(_operating_point(),),
        seeds=seeds,
        route_latency={
            f"{_structure().key}__{_operating_point().key}": _latency(),
        },
    )
    assert result.selected_key == f"{_structure().key}__{_operating_point().key}"
    assert result.eligible_count == 1
    assert len(result.metrics) == 2


def test_seed_shard_selection_rejects_missing_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crossfit, passed, total, costs, repositories = _fixture()

    def fake_crossfit(*args: object, **kwargs: object) -> CrossfitPredictions:
        del args, kwargs
        return crossfit

    monkeypatch.setattr(nested, "crossfit_structure", fake_crossfit)
    shard = evaluate_nested_seed(
        {"signed-hash-512": np.ones((10, 2), dtype=np.float64)},
        passed,
        total,
        costs,
        repositories,
        structures=(_structure(),),
        operating_points=(_operating_point(),),
        seed=11,
    )
    with pytest.raises(ValueError, match="incomplete"):
        select_nested_metrics(
            shard,
            structures=(_structure(),),
            operating_points=(_operating_point(),),
            seeds=(11, 23),
        )


def test_nested_selection_requires_a_complete_latency_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crossfit, passed, total, costs, repositories = _fixture()

    def fake_crossfit(*args: object, **kwargs: object) -> CrossfitPredictions:
        del args, kwargs
        return crossfit

    monkeypatch.setattr(nested, "crossfit_structure", fake_crossfit)
    key = f"{_structure().key}__{_operating_point().key}"

    def select(
        route_latency: dict[str, RouteLatencyMetric] | None = None,
    ) -> nested.NestedSelectionResult:
        return select_nested_policy(
            {"signed-hash-512": np.ones((10, 2), dtype=np.float64)},
            passed,
            total,
            costs,
            repositories,
            structures=(_structure(),),
            operating_points=(_operating_point(),),
            seeds=(11,),
            route_latency=route_latency,
        )

    missing = select()
    assert missing.eligible_count == 1
    assert missing.selected_key is None

    slow = select(
        {
            key: RouteLatencyMetric(
                p50_ms=5.0,
                p95_ms=20.0,
                decisions=10_000,
                network_calls=0,
            )
        },
    )
    assert slow.eligible_count == 1
    assert slow.selected_key is None

    measured = select({key: _latency()})
    assert measured.selected_key == key
