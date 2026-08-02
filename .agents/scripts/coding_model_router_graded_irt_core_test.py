"""Tests for the ephemeral graded binomial IRT core."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import coding_model_router_graded_irt_core as core
from coding_model_router_graded_irt_core import (
    _initial_feature_parameters,
    _initial_parameters,
    _ridge_projection,
    binomial_irt_loss_and_gradient,
    feature_binomial_irt_loss_and_gradient,
    fit_binomial_irt,
    fit_feature_binomial_irt,
    fit_projected_binomial_irt,
    kl_robust_lower_bound,
    predict_feature_probabilities,
    predict_probabilities,
    predict_projected_probabilities,
)


def test_binomial_irt_gradient_matches_finite_difference() -> None:
    passed = np.asarray(
        [
            [0, 1, 3],
            [2, 3, 4],
            [1, 4, 6],
            [0, 2, 8],
        ],
        dtype=np.float64,
    )
    total = np.asarray(
        [
            [2, 2, 4],
            [5, 5, 5],
            [8, 8, 8],
            [10, 10, 10],
        ],
        dtype=np.float64,
    )
    latent_dimension = 2
    parameters = _initial_parameters(passed, total, latent_dimension)
    _, gradient = binomial_irt_loss_and_gradient(
        parameters,
        passed,
        total,
        latent_dimension,
    )
    epsilon = 1e-6
    numerical = np.zeros_like(parameters)
    for index in range(len(parameters)):
        upper = parameters.copy()
        lower = parameters.copy()
        upper[index] += epsilon
        lower[index] -= epsilon
        upper_loss = binomial_irt_loss_and_gradient(
            upper,
            passed,
            total,
            latent_dimension,
        )[0]
        lower_loss = binomial_irt_loss_and_gradient(
            lower,
            passed,
            total,
            latent_dimension,
        )[0]
        numerical[index] = (upper_loss - lower_loss) / (2.0 * epsilon)
    np.testing.assert_allclose(gradient, numerical, atol=1e-6, rtol=1e-5)


def test_binomial_likelihood_weights_exact_denominators() -> None:
    passed = np.zeros((2, 2), dtype=np.float64)
    total = np.asarray([[2, 2], [200, 200]], dtype=np.float64)
    latent_dimension = 1
    parameters = np.zeros(2 + 2 + 2, dtype=np.float64)
    _, gradient = binomial_irt_loss_and_gradient(
        parameters,
        passed,
        total,
        latent_dimension,
        ability_l2=0.0,
        difficulty_l2=0.0,
        discrimination_l2=0.0,
    )
    difficulty_gradient = gradient[2:4]
    assert difficulty_gradient[1] == pytest.approx(100.0 * difficulty_gradient[0])


def test_monotone_task_irt_gradient_matches_finite_difference() -> None:
    passed = np.asarray(
        [[0, 1, 2, 3, 4, 2], [1, 2, 3, 4, 5, 3], [2, 3, 4, 5, 5, 4], [3, 4, 5, 5, 5, 5]],
        dtype=np.float64,
    )
    total = np.full_like(passed, 5.0)
    latent_dimension = 2
    parameters = _initial_parameters(passed, total, latent_dimension)
    _, gradient = binomial_irt_loss_and_gradient(
        parameters,
        passed,
        total,
        latent_dimension,
        monotone_luna=True,
    )
    epsilon = 1e-6
    numerical = np.zeros_like(parameters)
    for index in range(len(parameters)):
        upper = parameters.copy()
        lower = parameters.copy()
        upper[index] += epsilon
        lower[index] -= epsilon
        upper_loss = binomial_irt_loss_and_gradient(
            upper,
            passed,
            total,
            latent_dimension,
            monotone_luna=True,
        )[0]
        lower_loss = binomial_irt_loss_and_gradient(
            lower,
            passed,
            total,
            latent_dimension,
            monotone_luna=True,
        )[0]
        numerical[index] = (upper_loss - lower_loss) / (2.0 * epsilon)
    np.testing.assert_allclose(gradient, numerical, atol=1e-6, rtol=1e-5)


def test_feature_binomial_irt_gradient_matches_finite_difference() -> None:
    features = np.asarray(
        [
            [-1.0, 0.2],
            [-0.4, 0.5],
            [0.3, -0.2],
            [1.1, 0.7],
        ],
        dtype=np.float64,
    )
    passed = np.asarray(
        [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
        ],
        dtype=np.float64,
    )
    total = np.full_like(passed, 5.0)
    latent_dimension = 2
    parameters = _initial_feature_parameters(
        features,
        passed,
        total,
        latent_dimension,
    )
    _, gradient = feature_binomial_irt_loss_and_gradient(
        parameters,
        features,
        passed,
        total,
        latent_dimension,
    )
    epsilon = 1e-6
    numerical = np.zeros_like(parameters)
    for index in range(len(parameters)):
        upper = parameters.copy()
        lower = parameters.copy()
        upper[index] += epsilon
        lower[index] -= epsilon
        upper_loss = feature_binomial_irt_loss_and_gradient(
            upper,
            features,
            passed,
            total,
            latent_dimension,
        )[0]
        lower_loss = feature_binomial_irt_loss_and_gradient(
            lower,
            features,
            passed,
            total,
            latent_dimension,
        )[0]
        numerical[index] = (upper_loss - lower_loss) / (2.0 * epsilon)
    np.testing.assert_allclose(gradient, numerical, atol=1e-6, rtol=1e-5)


def test_graph_regularized_feature_gradient_matches_finite_difference() -> None:
    features = np.asarray(
        [[-1.0, 0.2], [-0.4, 0.5], [0.3, -0.2], [1.1, 0.7]],
        dtype=np.float64,
    )
    passed = np.asarray(
        [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]],
        dtype=np.float64,
    )
    total = np.full_like(passed, 5.0)
    laplacian = np.asarray(
        [
            [1.0, -1.0, 0.0, 0.0],
            [-1.0, 2.0, -1.0, 0.0],
            [0.0, -1.0, 2.0, -1.0],
            [0.0, 0.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )
    latent_dimension = 2
    parameters = _initial_feature_parameters(features, passed, total, latent_dimension)
    _, gradient = feature_binomial_irt_loss_and_gradient(
        parameters,
        features,
        passed,
        total,
        latent_dimension,
        graph_laplacian=laplacian,
        graph_l2=0.3,
    )
    epsilon = 1e-6
    numerical = np.zeros_like(parameters)
    for index in range(len(parameters)):
        upper = parameters.copy()
        lower = parameters.copy()
        upper[index] += epsilon
        lower[index] -= epsilon
        upper_loss = feature_binomial_irt_loss_and_gradient(
            upper,
            features,
            passed,
            total,
            latent_dimension,
            graph_laplacian=laplacian,
            graph_l2=0.3,
        )[0]
        lower_loss = feature_binomial_irt_loss_and_gradient(
            lower,
            features,
            passed,
            total,
            latent_dimension,
            graph_laplacian=laplacian,
            graph_l2=0.3,
        )[0]
        numerical[index] = (upper_loss - lower_loss) / (2.0 * epsilon)
    np.testing.assert_allclose(gradient, numerical, atol=1e-6, rtol=1e-5)


def test_graph_regularization_requires_valid_laplacian() -> None:
    features = np.asarray([[-1.0], [1.0]], dtype=np.float64)
    passed = np.asarray([[0, 1], [1, 2]], dtype=np.float64)
    total = np.full_like(passed, 2.0)
    parameters = _initial_feature_parameters(features, passed, total, 1)
    invalid = np.asarray([[1.0, 1.0], [-1.0, 1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="symmetric"):
        feature_binomial_irt_loss_and_gradient(
            parameters,
            features,
            passed,
            total,
            1,
            graph_laplacian=invalid,
            graph_l2=0.1,
        )
    with pytest.raises(ValueError, match="requires"):
        feature_binomial_irt_loss_and_gradient(
            parameters,
            features,
            passed,
            total,
            1,
            graph_l2=0.1,
        )


def test_graph_regularized_feature_fit_is_finite() -> None:
    features = np.linspace(-1.5, 1.5, 12, dtype=np.float64)[:, None]
    abilities = np.linspace(-1.0, 1.0, 6, dtype=np.float64)
    probabilities = 1.0 / (1.0 + np.exp(-(abilities[None, :] - features)))
    total = np.full(probabilities.shape, 20.0)
    passed = np.rint(total * probabilities)
    adjacency = np.zeros((len(features), len(features)), dtype=np.float64)
    for index in range(len(features) - 1):
        adjacency[index, index + 1] = 1.0
        adjacency[index + 1, index] = 1.0
    laplacian = np.diag(np.sum(adjacency, axis=1)) - adjacency
    fit = fit_feature_binomial_irt(
        features,
        passed,
        total,
        latent_dimension=2,
        graph_laplacian=laplacian,
        graph_l2=0.1,
    )
    assert np.isfinite(fit.loss)
    assert fit.difficulty_weights.shape == (2,)


def test_monotone_feature_irt_gradient_matches_finite_difference() -> None:
    features = np.asarray(
        [[-1.0, 0.2], [-0.4, 0.5], [0.3, -0.2], [1.1, 0.7]],
        dtype=np.float64,
    )
    passed = np.asarray(
        [[0, 1, 2, 3, 4, 2], [1, 2, 3, 4, 5, 3], [2, 3, 4, 5, 5, 4], [3, 4, 5, 5, 5, 5]],
        dtype=np.float64,
    )
    total = np.full_like(passed, 5.0)
    latent_dimension = 2
    parameters = _initial_feature_parameters(features, passed, total, latent_dimension)
    _, gradient = feature_binomial_irt_loss_and_gradient(
        parameters,
        features,
        passed,
        total,
        latent_dimension,
        monotone_luna=True,
    )
    epsilon = 1e-6
    numerical = np.zeros_like(parameters)
    for index in range(len(parameters)):
        upper = parameters.copy()
        lower = parameters.copy()
        upper[index] += epsilon
        lower[index] -= epsilon
        upper_loss = feature_binomial_irt_loss_and_gradient(
            upper,
            features,
            passed,
            total,
            latent_dimension,
            monotone_luna=True,
        )[0]
        lower_loss = feature_binomial_irt_loss_and_gradient(
            lower,
            features,
            passed,
            total,
            latent_dimension,
            monotone_luna=True,
        )[0]
        numerical[index] = (upper_loss - lower_loss) / (2.0 * epsilon)
    np.testing.assert_allclose(gradient, numerical, atol=1e-6, rtol=1e-5)


def test_multidimensional_fit_recovers_arm_order() -> None:
    task_count = 36
    arm_count = 6
    difficulties = np.linspace(-1.5, 1.5, task_count)
    abilities = np.linspace(-1.2, 1.2, arm_count)
    probabilities = 1.0 / (1.0 + np.exp(-(abilities[None, :] - difficulties[:, None])))
    total = np.full((task_count, arm_count), 20.0)
    passed = np.rint(total * probabilities)
    fit = fit_binomial_irt(passed, total, latent_dimension=2)
    predicted = predict_probabilities(fit)
    assert np.all(np.diff(np.mean(predicted, axis=0)) > 0.0)
    assert np.isfinite(fit.loss)
    assert fit.abilities.shape == (arm_count, 2)
    assert fit.log_discriminations.shape == (task_count, 2)


def test_binomial_fit_continues_from_finite_iteration_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    passed = np.asarray([[0, 1], [1, 2]], dtype=np.float64)
    total = np.asarray([[2, 2], [2, 2]], dtype=np.float64)
    calls = 0

    def fake_minimize(
        objective: object,
        initial: np.ndarray,
        **kwargs: object,
    ) -> SimpleNamespace:
        del objective, kwargs
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            success=calls == 2,
            fun=1.0,
            x=np.asarray(initial, dtype=np.float64),
            nit=1_000 if calls == 1 else 7,
            message=(
                "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT"
                if calls == 1
                else "CONVERGENCE"
            ),
        )

    monkeypatch.setattr(core, "minimize", fake_minimize)

    fit = core.fit_binomial_irt(passed, total, latent_dimension=2)

    assert calls == 2
    assert fit.iterations == 1_007


def test_feature_fit_predicts_unseen_task_probabilities() -> None:
    train_features = np.linspace(-1.5, 1.5, 40, dtype=np.float64)[:, None]
    arm_abilities = np.linspace(-1.2, 1.2, 6, dtype=np.float64)
    logits = arm_abilities[None, :] - train_features
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    total = np.full(probabilities.shape, 40.0)
    passed = np.rint(total * probabilities)
    fit = fit_feature_binomial_irt(
        train_features,
        passed,
        total,
        latent_dimension=2,
    )
    test_features = np.asarray([[-1.0], [0.0], [1.0]], dtype=np.float64)
    predicted = predict_feature_probabilities(fit, test_features)
    assert predicted.shape == (3, 6)
    assert np.all(np.diff(np.mean(predicted, axis=0)) > 0.0)
    assert np.all(np.diff(predicted[:, -1]) < 0.0)


def test_monotone_feature_fit_orders_luna_capacity_only() -> None:
    train_features = np.linspace(-1.5, 1.5, 40, dtype=np.float64)[:, None]
    arm_abilities = np.asarray([-1.2, -0.7, -0.1, 0.5, 1.1, 0.2], dtype=np.float64)
    logits = arm_abilities[None, :] - train_features
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    total = np.full(probabilities.shape, 40.0)
    passed = np.rint(total * probabilities)
    fit = fit_feature_binomial_irt(
        train_features,
        passed,
        total,
        latent_dimension=2,
        monotone_luna=True,
    )
    assert np.all(np.diff(fit.abilities[:5, 0]) > 0.0)
    assert fit.abilities[5, 0] < fit.abilities[4, 0]
    predicted = predict_feature_probabilities(fit, np.asarray([[0.0]]))
    assert np.all(np.diff(predicted[0, :5]) > 0.0)


def test_graph_ridge_projection_satisfies_its_normal_equation() -> None:
    features = np.asarray(
        [[-1.0, 0.2], [-0.4, 0.5], [0.3, -0.2], [1.1, 0.7]],
        dtype=np.float64,
    )
    targets = np.asarray([-1.0, -0.2, 0.4, 1.2], dtype=np.float64)
    laplacian = np.asarray(
        [
            [1.0, -1.0, 0.0, 0.0],
            [-1.0, 2.0, -1.0, 0.0],
            [0.0, -1.0, 2.0, -1.0],
            [0.0, 0.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )
    regularization = 0.3
    graph_l2 = 0.2
    weights = _ridge_projection(
        features,
        targets,
        regularization=regularization,
        graph_laplacian=laplacian,
        graph_l2=graph_l2,
    )
    augmented = np.column_stack([features, np.ones(len(features))])
    predictions = augmented @ weights
    normal_gradient = (
        augmented.T @ (predictions - targets)
        + regularization * weights
        + graph_l2 * augmented.T @ laplacian @ predictions
    )
    np.testing.assert_allclose(normal_gradient, 0.0, atol=1e-9)


def test_projected_irt_predicts_unseen_tasks_without_task_state() -> None:
    train_features = np.linspace(-1.5, 1.5, 40, dtype=np.float64)[:, None]
    arm_abilities = np.asarray([-1.2, -0.7, -0.1, 0.5, 1.1, 0.2], dtype=np.float64)
    probabilities = 1.0 / (
        1.0 + np.exp(-(arm_abilities[None, :] - train_features))
    )
    total = np.full(probabilities.shape, 40.0)
    passed = np.rint(total * probabilities)
    fit = fit_projected_binomial_irt(
        train_features,
        passed,
        total,
        latent_dimension=2,
        projection_l2=0.1,
        monotone_luna=True,
    )
    predicted = predict_projected_probabilities(
        fit,
        np.asarray([[-1.0], [0.0], [1.0]], dtype=np.float64),
    )
    assert predicted.shape == (3, 6)
    assert np.isfinite(predicted).all()
    assert np.all(np.diff(fit.abilities[:5, 0]) > 0.0)
    assert fit.difficulty_weights.shape == (2,)
    assert fit.log_discrimination_weights.shape == (2, 2)


def test_kl_robust_lower_bound_matches_two_repository_search() -> None:
    values = np.asarray([0.2, 0.9], dtype=np.float64)
    weights = np.asarray([0.35, 0.65], dtype=np.float64)
    radius = 0.08
    robust = kl_robust_lower_bound(values, radius, weights)
    candidates = np.linspace(1e-6, 1.0 - 1e-6, 200_001)
    distributions = np.column_stack([candidates, 1.0 - candidates])
    divergences = np.sum(distributions * np.log(distributions / weights), axis=1)
    feasible = distributions[divergences <= radius]
    brute_force = float(np.min(feasible @ values))
    assert robust == pytest.approx(brute_force, abs=1e-5)


def test_kl_robust_lower_bound_respects_radius_extremes() -> None:
    values = np.asarray([0.1, 0.5, 0.9], dtype=np.float64)
    nominal = kl_robust_lower_bound(values, 0.0)
    moderate = kl_robust_lower_bound(values, 0.1)
    concentrated = kl_robust_lower_bound(values, np.log(3.0))
    assert nominal == pytest.approx(0.5)
    assert 0.1 < moderate < nominal
    assert concentrated == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("passed", "total"),
    [
        (np.asarray([[0.5, 1.0]]), np.asarray([[1.0, 1.0]])),
        (np.asarray([[2.0, 0.0]]), np.asarray([[1.0, 1.0]])),
        (np.asarray([[0.0, 0.0]]), np.asarray([[0.0, 1.0]])),
    ],
)
def test_binomial_fit_rejects_invalid_counts(passed: np.ndarray, total: np.ndarray) -> None:
    with pytest.raises(ValueError):
        fit_binomial_irt(passed, total, latent_dimension=2)
