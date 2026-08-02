"""Fit an ephemeral multidimensional IRT model to graded test counts.

This module owns only the numeric core for the conditional graded-router study. It has no
filesystem output surface, so fitted abilities, difficulties, and discriminations stay inside the
remote fit process that calls it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

DEFAULT_ABILITY_L2 = 0.01
DEFAULT_DIFFICULTY_L2 = 0.01
DEFAULT_DISCRIMINATION_L2 = 0.05
BINOMIAL_IRT_MAX_ITERATIONS_PER_PASS = 1_000
BINOMIAL_IRT_CONTINUATION_PASSES = 3
FROZEN_ARM_COUNT = 6
LUNA_ARM_COUNT = 5


@dataclass(frozen=True)
class BinomialIrtFit:
    """One ephemeral multidimensional item-response fit."""

    abilities: np.ndarray
    difficulties: np.ndarray
    log_discriminations: np.ndarray
    loss: float
    iterations: int


@dataclass(frozen=True)
class FeatureBinomialIrtFit:
    """One ephemeral feature-conditioned multidimensional item-response fit."""

    abilities: np.ndarray
    difficulty_weights: np.ndarray
    discrimination_weights: np.ndarray
    loss: float
    iterations: int


@dataclass(frozen=True)
class ProjectedBinomialIrtFit:
    """One ephemeral task-IRT fit projected onto pre-call features."""

    abilities: np.ndarray
    difficulty_weights: np.ndarray
    log_discrimination_weights: np.ndarray
    loss: float
    projection_loss: float
    iterations: int


def _validate_counts(passed: np.ndarray, total: np.ndarray) -> None:
    """Validate a dense task-by-arm matrix of exact binomial counts."""
    if passed.ndim != 2 or total.ndim != 2 or passed.shape != total.shape:
        raise ValueError("passed and total must be matching task-by-arm matrices")
    if not passed.size or not np.isfinite(passed).all() or not np.isfinite(total).all():
        raise ValueError("graded count matrices must be nonempty and finite")
    if np.any(total <= 0.0) or np.any(passed < 0.0) or np.any(passed > total):
        raise ValueError("graded counts require 0 <= passed <= positive total")
    if not np.array_equal(passed, np.rint(passed)) or not np.array_equal(total, np.rint(total)):
        raise ValueError("graded passed and total values must be integer counts")


def _logit(values: np.ndarray) -> np.ndarray:
    """Return a numerically bounded logit."""
    clipped = np.clip(values, 1e-4, 1.0 - 1e-4)
    return np.log(clipped / (1.0 - clipped))


def _validate_features(features: np.ndarray, task_count: int) -> None:
    """Validate a dense pre-call feature matrix."""
    if (
        task_count < 1
        or features.ndim != 2
        or features.shape[0] != task_count
        or features.shape[1] < 1
    ):
        raise ValueError("features must be a task-by-feature matrix aligned with counts")
    if not np.isfinite(features).all():
        raise ValueError("task features must be finite")


def _validate_graph_laplacian(
    graph_laplacian: np.ndarray | None,
    task_count: int,
    graph_l2: float,
) -> None:
    """Validate an optional fit-only task graph penalty."""
    if not np.isfinite(graph_l2) or graph_l2 < 0.0:
        raise ValueError("graph regularization strength must be finite and nonnegative")
    if graph_laplacian is None:
        if graph_l2 != 0.0:
            raise ValueError("positive graph regularization requires a graph Laplacian")
        return
    if graph_laplacian.shape != (task_count, task_count):
        raise ValueError("graph Laplacian must be a square task-by-task matrix")
    if not np.isfinite(graph_laplacian).all():
        raise ValueError("graph Laplacian must be finite")
    if not np.allclose(graph_laplacian, graph_laplacian.T, atol=1e-10):
        raise ValueError("graph Laplacian must be symmetric")
    if np.any(np.diag(graph_laplacian) < -1e-10):
        raise ValueError("graph Laplacian diagonal must be nonnegative")
    off_diagonal = graph_laplacian.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    if np.any(off_diagonal > 1e-10):
        raise ValueError("graph Laplacian off-diagonal entries must be nonpositive")
    if not np.allclose(np.sum(graph_laplacian, axis=1), 0.0, atol=1e-10):
        raise ValueError("graph Laplacian rows must sum to zero")


def _feature_parameter_shapes(
    arm_count: int,
    feature_count: int,
    latent_dimension: int,
) -> tuple[int, int, int]:
    """Return feature-conditioned parameter block sizes, including intercepts."""
    if arm_count < 2 or feature_count < 1 or latent_dimension < 1:
        raise ValueError("feature IRT dimensions require two arms, features, and latent dimensions")
    augmented_feature_count = feature_count + 1
    return (
        arm_count * latent_dimension,
        augmented_feature_count,
        augmented_feature_count * latent_dimension,
    )


def _unpack_feature_parameters(
    parameters: np.ndarray,
    arm_count: int,
    feature_count: int,
    latent_dimension: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """View one feature-conditioned optimizer vector as its three parameter blocks."""
    ability_size, difficulty_size, discrimination_size = _feature_parameter_shapes(
        arm_count,
        feature_count,
        latent_dimension,
    )
    if parameters.shape != (ability_size + difficulty_size + discrimination_size,):
        raise ValueError("feature IRT parameter vector has the wrong length")
    ability_end = ability_size
    difficulty_end = ability_end + difficulty_size
    abilities = parameters[:ability_end].reshape(arm_count, latent_dimension)
    difficulty_weights = parameters[ability_end:difficulty_end]
    discrimination_weights = parameters[difficulty_end:].reshape(
        feature_count + 1,
        latent_dimension,
    )
    return abilities, difficulty_weights, discrimination_weights


def _augment_features(features: np.ndarray) -> np.ndarray:
    """Append a deterministic intercept column to pre-call features."""
    return np.column_stack([features, np.ones(len(features), dtype=np.float64)])


def _softplus(values: np.ndarray) -> np.ndarray:
    """Return a stable positive discrimination transform."""
    return np.logaddexp(0.0, values) + 1e-6


def _monotone_luna_abilities(raw_abilities: np.ndarray) -> np.ndarray:
    """Map raw abilities to a monotone first coordinate for the five Luna arms."""
    if (
        raw_abilities.ndim != 2
        or raw_abilities.shape[0] != FROZEN_ARM_COUNT
        or raw_abilities.shape[1] < 1
    ):
        raise ValueError("monotone Luna abilities require the frozen six-arm roster")
    abilities = raw_abilities.copy()
    increments = _softplus(raw_abilities[1:LUNA_ARM_COUNT, 0])
    abilities[1:LUNA_ARM_COUNT, 0] = raw_abilities[0, 0] + np.cumsum(increments)
    return abilities


def _monotone_luna_raw_gradient(
    raw_abilities: np.ndarray,
    ability_gradient: np.ndarray,
) -> np.ndarray:
    """Apply the monotone Luna parameterization Jacobian to an ability gradient."""
    if raw_abilities.shape != ability_gradient.shape:
        raise ValueError("ability gradient must match raw abilities")
    _monotone_luna_abilities(raw_abilities)
    raw_gradient = ability_gradient.copy()
    luna_gradient = ability_gradient[:LUNA_ARM_COUNT, 0]
    suffix = np.cumsum(luna_gradient[::-1])[::-1]
    raw_gradient[0, 0] = suffix[0]
    raw_gradient[1:LUNA_ARM_COUNT, 0] = (
        expit(raw_abilities[1:LUNA_ARM_COUNT, 0]) * suffix[1:]
    )
    return raw_gradient


def _monotone_luna_initial(raw_abilities: np.ndarray) -> np.ndarray:
    """Convert initial direct abilities to the monotone raw parameterization."""
    if (
        raw_abilities.ndim != 2
        or raw_abilities.shape[0] != FROZEN_ARM_COUNT
        or raw_abilities.shape[1] < 1
    ):
        raise ValueError("monotone Luna initialization requires the frozen six-arm roster")
    raw = raw_abilities.copy()
    target = raw_abilities[:LUNA_ARM_COUNT, 0].copy()
    for index in range(1, LUNA_ARM_COUNT):
        target[index] = max(target[index], target[index - 1] + 0.01)
    raw[0, 0] = target[0]
    positive = np.maximum(np.diff(target) - 1e-6, 1e-6)
    raw[1:LUNA_ARM_COUNT, 0] = np.log(np.expm1(positive))
    return raw


def _tilted_distribution(
    values: np.ndarray,
    weights: np.ndarray,
    temperature: float,
) -> tuple[np.ndarray, float]:
    """Return the exponential lower-tail tilt and its forward KL divergence."""
    if temperature <= 0.0:
        raise ValueError("KL tilt temperature must be positive")
    scaled = -(values - float(np.min(values))) / temperature
    unnormalized = weights * np.exp(scaled)
    distribution = unnormalized / np.sum(unnormalized)
    positive = distribution > 0.0
    divergence = float(
        np.sum(
            distribution[positive]
            * np.log(distribution[positive] / weights[positive])
        )
    )
    return distribution, divergence


def kl_robust_lower_bound(
    values: np.ndarray,
    radius: float,
    weights: np.ndarray | None = None,
) -> float:
    """Minimize expected reward over a forward-KL ball around repository weights.

    Args:
        values: One reward estimate per repository.
        radius: Nonnegative KL radius.
        weights: Optional strictly positive nominal repository probabilities.

    Returns:
        The worst-case expected reward under ``KL(q || weights) <= radius``.
    """
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("KL robust values must be a nonempty finite vector")
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("KL radius must be finite and nonnegative")
    if weights is None:
        nominal = np.full(len(values), 1.0 / len(values), dtype=np.float64)
    else:
        if weights.shape != values.shape or not np.isfinite(weights).all():
            raise ValueError("KL weights must be finite and aligned with values")
        if np.any(weights <= 0.0):
            raise ValueError("KL weights must be strictly positive")
        weight_sum = float(np.sum(weights))
        if not np.isfinite(weight_sum):
            raise ValueError("KL weight sum must be finite")
        nominal = weights / weight_sum
    if radius == 0.0 or np.all(values == values[0]):
        return float(nominal @ values)

    minimum = float(np.min(values))
    minimum_mass = float(np.sum(nominal[values == minimum]))
    if radius >= -np.log(minimum_mass):
        return minimum

    upper = max(float(np.ptp(values)), 1e-6)
    _, divergence = _tilted_distribution(values, nominal, upper)
    while divergence > radius:
        upper *= 2.0
        _, divergence = _tilted_distribution(values, nominal, upper)
    lower = 0.0
    for _ in range(100):
        temperature = (lower + upper) / 2.0
        _, divergence = _tilted_distribution(values, nominal, temperature)
        if divergence > radius:
            lower = temperature
        else:
            upper = temperature
    distribution, _ = _tilted_distribution(values, nominal, upper)
    return float(distribution @ values)


def _parameter_shapes(
    task_count: int,
    arm_count: int,
    latent_dimension: int,
) -> tuple[int, int, int]:
    """Return flattened parameter block sizes."""
    if task_count < 1 or arm_count < 2 or latent_dimension < 1:
        raise ValueError("IRT dimensions require tasks, at least two arms, and latent_dimension")
    return (
        arm_count * latent_dimension,
        task_count,
        task_count * latent_dimension,
    )


def _unpack(
    parameters: np.ndarray,
    task_count: int,
    arm_count: int,
    latent_dimension: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """View one flat optimizer vector as abilities, difficulties, and log discriminations."""
    ability_size, difficulty_size, discrimination_size = _parameter_shapes(
        task_count,
        arm_count,
        latent_dimension,
    )
    if parameters.shape != (ability_size + difficulty_size + discrimination_size,):
        raise ValueError("IRT parameter vector has the wrong length")
    ability_end = ability_size
    difficulty_end = ability_end + difficulty_size
    abilities = parameters[:ability_end].reshape(arm_count, latent_dimension)
    difficulties = parameters[ability_end:difficulty_end]
    log_discriminations = parameters[difficulty_end:].reshape(task_count, latent_dimension)
    return abilities, difficulties, log_discriminations


def binomial_irt_loss_and_gradient(
    parameters: np.ndarray,
    passed: np.ndarray,
    total: np.ndarray,
    latent_dimension: int,
    *,
    ability_l2: float = DEFAULT_ABILITY_L2,
    difficulty_l2: float = DEFAULT_DIFFICULTY_L2,
    discrimination_l2: float = DEFAULT_DISCRIMINATION_L2,
    monotone_luna: bool = False,
) -> tuple[float, np.ndarray]:
    """Return exact binomial negative log likelihood and its analytic gradient.

    The likelihood is normalized by the number of fail-to-pass assertions, not the number of
    task-arm cells. A 100-test observation therefore contributes 100 times the evidence of an
    otherwise identical one-test observation.
    """
    _validate_counts(passed, total)
    if min(ability_l2, difficulty_l2, discrimination_l2) < 0.0:
        raise ValueError("IRT regularization strengths must be nonnegative")
    task_count, arm_count = passed.shape
    raw_abilities, difficulties, log_discriminations = _unpack(
        parameters,
        task_count,
        arm_count,
        latent_dimension,
    )
    abilities = (
        _monotone_luna_abilities(raw_abilities)
        if monotone_luna
        else raw_abilities
    )
    discriminations = np.exp(log_discriminations)
    logits = discriminations @ abilities.T - difficulties[:, None]
    assertion_count = float(np.sum(total))
    loss = float(np.sum(total * np.logaddexp(0.0, logits) - passed * logits) / assertion_count)
    loss += ability_l2 * float(np.mean(abilities**2))
    loss += difficulty_l2 * float(np.mean(difficulties**2))
    loss += discrimination_l2 * float(np.mean(log_discriminations**2))

    error = (total * expit(logits) - passed) / assertion_count
    ability_gradient = error.T @ discriminations
    ability_gradient += 2.0 * ability_l2 * abilities / abilities.size
    if monotone_luna:
        ability_gradient = _monotone_luna_raw_gradient(
            raw_abilities,
            ability_gradient,
        )
    difficulty_gradient = -np.sum(error, axis=1)
    difficulty_gradient += 2.0 * difficulty_l2 * difficulties / difficulties.size
    discrimination_gradient = (error @ abilities) * discriminations
    discrimination_gradient += (
        2.0 * discrimination_l2 * log_discriminations / log_discriminations.size
    )
    return loss, np.concatenate(
        [
            ability_gradient.ravel(),
            difficulty_gradient,
            discrimination_gradient.ravel(),
        ]
    )


def feature_binomial_irt_loss_and_gradient(
    parameters: np.ndarray,
    features: np.ndarray,
    passed: np.ndarray,
    total: np.ndarray,
    latent_dimension: int,
    *,
    ability_l2: float = DEFAULT_ABILITY_L2,
    feature_l2: float = DEFAULT_DIFFICULTY_L2,
    discrimination_l2: float = DEFAULT_DISCRIMINATION_L2,
    monotone_luna: bool = False,
    graph_laplacian: np.ndarray | None = None,
    graph_l2: float = 0.0,
    _graph_validated: bool = False,
) -> tuple[float, np.ndarray]:
    """Return exact binomial loss for a feature-conditioned IRT model.

    Difficulty and nonnegative discrimination are functions only of pre-call features. This
    permits grouped out-of-fold predictions for unseen repositories without retaining task-level
    fitted state.
    """
    _validate_counts(passed, total)
    _validate_features(features, len(passed))
    if not _graph_validated:
        _validate_graph_laplacian(graph_laplacian, len(passed), graph_l2)
    if min(ability_l2, feature_l2, discrimination_l2) < 0.0:
        raise ValueError("feature IRT regularization strengths must be nonnegative")
    _, arm_count = passed.shape
    raw_abilities, difficulty_weights, discrimination_weights = _unpack_feature_parameters(
        parameters,
        arm_count,
        features.shape[1],
        latent_dimension,
    )
    abilities = (
        _monotone_luna_abilities(raw_abilities)
        if monotone_luna
        else raw_abilities
    )
    augmented = _augment_features(features)
    difficulties = augmented @ difficulty_weights
    discrimination_linear = augmented @ discrimination_weights
    discriminations = _softplus(discrimination_linear)
    logits = discriminations @ abilities.T - difficulties[:, None]
    assertion_count = float(np.sum(total))
    loss = float(np.sum(total * np.logaddexp(0.0, logits) - passed * logits) / assertion_count)
    loss += ability_l2 * float(np.mean(abilities**2))
    loss += feature_l2 * float(np.mean(difficulty_weights**2))
    loss += discrimination_l2 * float(np.mean(discrimination_weights**2))
    if graph_laplacian is not None and graph_l2 > 0.0:
        loss += graph_l2 * float(
            difficulties @ graph_laplacian @ difficulties / len(difficulties)
        )

    error = (total * expit(logits) - passed) / assertion_count
    ability_gradient = error.T @ discriminations
    ability_gradient += 2.0 * ability_l2 * abilities / abilities.size
    if monotone_luna:
        ability_gradient = _monotone_luna_raw_gradient(
            raw_abilities,
            ability_gradient,
        )
    difficulty_gradient = augmented.T @ (-np.sum(error, axis=1))
    difficulty_gradient += 2.0 * feature_l2 * difficulty_weights / difficulty_weights.size
    if graph_laplacian is not None and graph_l2 > 0.0:
        difficulty_gradient += (
            2.0
            * graph_l2
            * augmented.T
            @ graph_laplacian
            @ difficulties
            / len(difficulties)
        )
    discrimination_linear_gradient = (error @ abilities) * expit(discrimination_linear)
    discrimination_gradient = augmented.T @ discrimination_linear_gradient
    discrimination_gradient += (
        2.0
        * discrimination_l2
        * discrimination_weights
        / discrimination_weights.size
    )
    return loss, np.concatenate(
        [
            ability_gradient.ravel(),
            difficulty_gradient,
            discrimination_gradient.ravel(),
        ]
    )


def _initial_feature_parameters(
    features: np.ndarray,
    passed: np.ndarray,
    total: np.ndarray,
    latent_dimension: int,
) -> np.ndarray:
    """Build a deterministic initializer without fitting per-task parameters."""
    _validate_counts(passed, total)
    _validate_features(features, len(passed))
    _, arm_count = passed.shape
    feature_count = features.shape[1]
    arm_rates = np.sum(passed, axis=0) / np.sum(total, axis=0)
    arm_logits = _logit(arm_rates)
    arm_logits -= np.mean(arm_logits)
    abilities = np.zeros((arm_count, latent_dimension), dtype=np.float64)
    abilities[:, 0] = arm_logits
    if latent_dimension > 1:
        offsets = np.linspace(-0.01, 0.01, arm_count, dtype=np.float64)
        for dimension in range(1, latent_dimension):
            abilities[:, dimension] = offsets * (dimension / latent_dimension)

    difficulty_weights = np.zeros(feature_count + 1, dtype=np.float64)
    task_rates = np.sum(passed, axis=1) / np.sum(total, axis=1)
    difficulty_weights[-1] = float(np.mean(-_logit(task_rates)))
    discrimination_weights = np.zeros(
        (feature_count + 1, latent_dimension),
        dtype=np.float64,
    )
    initial_discrimination = 1.0 / np.sqrt(float(latent_dimension))
    discrimination_weights[-1] = np.log(np.expm1(initial_discrimination))
    return np.concatenate(
        [
            abilities.ravel(),
            difficulty_weights,
            discrimination_weights.ravel(),
        ]
    )


def _initial_parameters(
    passed: np.ndarray,
    total: np.ndarray,
    latent_dimension: int,
) -> np.ndarray:
    """Build a deterministic bounded initializer from marginal graded pass rates."""
    task_count, arm_count = passed.shape
    arm_rates = np.sum(passed, axis=0) / np.sum(total, axis=0)
    arm_logits = _logit(arm_rates)
    arm_logits -= np.mean(arm_logits)
    abilities = np.zeros((arm_count, latent_dimension), dtype=np.float64)
    abilities[:, 0] = arm_logits
    if latent_dimension > 1:
        arm_offsets = np.linspace(-0.01, 0.01, arm_count, dtype=np.float64)
        for dimension in range(1, latent_dimension):
            abilities[:, dimension] = arm_offsets * (dimension / latent_dimension)
    task_rates = np.sum(passed, axis=1) / np.sum(total, axis=1)
    difficulties = -_logit(task_rates)
    difficulties -= np.mean(difficulties)
    initial_discrimination = -0.5 * np.log(float(latent_dimension))
    log_discriminations = np.full(
        (task_count, latent_dimension),
        initial_discrimination,
        dtype=np.float64,
    )
    return np.concatenate(
        [abilities.ravel(), difficulties, log_discriminations.ravel()]
    )


def fit_binomial_irt(
    passed: np.ndarray,
    total: np.ndarray,
    latent_dimension: int,
    *,
    ability_l2: float = DEFAULT_ABILITY_L2,
    difficulty_l2: float = DEFAULT_DIFFICULTY_L2,
    discrimination_l2: float = DEFAULT_DISCRIMINATION_L2,
    monotone_luna: bool = False,
) -> BinomialIrtFit:
    """Fit the ephemeral count-weighted model with bounded L-BFGS optimization."""
    _validate_counts(passed, total)
    task_count, arm_count = passed.shape
    initial = _initial_parameters(passed, total, latent_dimension)
    ability_size, difficulty_size, discrimination_size = _parameter_shapes(
        task_count,
        arm_count,
        latent_dimension,
    )
    bounds = (
        [(-8.0, 8.0)] * ability_size
        + [(-8.0, 8.0)] * difficulty_size
        + [(-3.0, 3.0)] * discrimination_size
    )
    if monotone_luna:
        raw_abilities, difficulties, log_discriminations = _unpack(
            initial,
            task_count,
            arm_count,
            latent_dimension,
        )
        initial = np.concatenate(
            [
                _monotone_luna_initial(raw_abilities).ravel(),
                difficulties,
                log_discriminations.ravel(),
            ]
        )
    objective = partial(
        binomial_irt_loss_and_gradient,
        ability_l2=ability_l2,
        difficulty_l2=difficulty_l2,
        discrimination_l2=discrimination_l2,
        monotone_luna=monotone_luna,
    )
    total_iterations = 0
    for _ in range(BINOMIAL_IRT_CONTINUATION_PASSES):
        result = minimize(
            objective,
            initial,
            args=(passed, total, latent_dimension),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": BINOMIAL_IRT_MAX_ITERATIONS_PER_PASS,
                "ftol": 1e-10,
                "gtol": 1e-7,
            },
        )
        total_iterations += int(result.nit)
        if result.success and np.isfinite(result.fun):
            break
        if (
            "ITERATIONS REACHED LIMIT" not in str(result.message)
            or not np.isfinite(result.fun)
            or not np.isfinite(result.x).all()
        ):
            raise RuntimeError(f"binomial IRT optimization failed: {result.message}")
        initial = np.asarray(result.x, dtype=np.float64)
    else:
        raise RuntimeError(f"binomial IRT optimization failed: {result.message}")
    raw_abilities, difficulties, log_discriminations = _unpack(
        np.asarray(result.x, dtype=np.float64),
        task_count,
        arm_count,
        latent_dimension,
    )
    abilities = (
        _monotone_luna_abilities(raw_abilities)
        if monotone_luna
        else raw_abilities
    )
    return BinomialIrtFit(
        abilities=abilities.copy(),
        difficulties=difficulties.copy(),
        log_discriminations=log_discriminations.copy(),
        loss=float(result.fun),
        iterations=total_iterations,
    )


def fit_feature_binomial_irt(
    features: np.ndarray,
    passed: np.ndarray,
    total: np.ndarray,
    latent_dimension: int,
    *,
    ability_l2: float = DEFAULT_ABILITY_L2,
    feature_l2: float = DEFAULT_DIFFICULTY_L2,
    discrimination_l2: float = DEFAULT_DISCRIMINATION_L2,
    monotone_luna: bool = False,
    graph_laplacian: np.ndarray | None = None,
    graph_l2: float = 0.0,
) -> FeatureBinomialIrtFit:
    """Fit a feature-conditioned model for ephemeral grouped evaluation.

    When ``monotone_luna`` is enabled, the first latent ability coordinate is constrained to
    increase from Luna low through Luna max. The sixth Sol arm remains unconstrained.
    """
    _validate_counts(passed, total)
    _validate_features(features, len(passed))
    _validate_graph_laplacian(graph_laplacian, len(passed), graph_l2)
    _, arm_count = passed.shape
    feature_count = features.shape[1]
    initial = _initial_feature_parameters(
        features,
        passed,
        total,
        latent_dimension,
    )
    ability_size, difficulty_size, discrimination_size = _feature_parameter_shapes(
        arm_count,
        feature_count,
        latent_dimension,
    )
    if monotone_luna:
        raw_abilities, difficulty_weights, discrimination_weights = (
            _unpack_feature_parameters(
                initial,
                arm_count,
                feature_count,
                latent_dimension,
            )
        )
        initial = np.concatenate(
            [
                _monotone_luna_initial(raw_abilities).ravel(),
                difficulty_weights,
                discrimination_weights.ravel(),
            ]
        )
    bounds = (
        [(-8.0, 8.0)] * ability_size
        + [(-8.0, 8.0)] * difficulty_size
        + [(-4.0, 4.0)] * discrimination_size
    )
    objective = partial(
        feature_binomial_irt_loss_and_gradient,
        ability_l2=ability_l2,
        feature_l2=feature_l2,
        discrimination_l2=discrimination_l2,
        monotone_luna=monotone_luna,
        graph_laplacian=graph_laplacian,
        graph_l2=graph_l2,
        _graph_validated=True,
    )
    result = minimize(
        objective,
        initial,
        args=(features, passed, total, latent_dimension),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 1_000, "ftol": 1e-10, "gtol": 1e-7},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"feature binomial IRT optimization failed: {result.message}")
    raw_abilities, difficulty_weights, discrimination_weights = _unpack_feature_parameters(
        np.asarray(result.x, dtype=np.float64),
        arm_count,
        feature_count,
        latent_dimension,
    )
    abilities = (
        _monotone_luna_abilities(raw_abilities)
        if monotone_luna
        else raw_abilities
    )
    return FeatureBinomialIrtFit(
        abilities=abilities.copy(),
        difficulty_weights=difficulty_weights.copy(),
        discrimination_weights=discrimination_weights.copy(),
        loss=float(result.fun),
        iterations=int(result.nit),
    )


def _ridge_projection(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    regularization: float,
    graph_laplacian: np.ndarray | None = None,
    graph_l2: float = 0.0,
) -> np.ndarray:
    """Project task latents onto pre-call features by exact dual ridge."""
    _validate_features(features, len(features))
    if targets.ndim not in (1, 2) or targets.shape[0] != len(features):
        raise ValueError("ridge projection targets must align with feature rows")
    if not np.isfinite(targets).all():
        raise ValueError("ridge projection targets must be finite")
    if not np.isfinite(regularization) or regularization <= 0.0:
        raise ValueError("ridge projection regularization must be finite and positive")
    _validate_graph_laplacian(graph_laplacian, len(features), graph_l2)
    augmented = _augment_features(features)
    gram = augmented @ augmented.T
    graph_operator = np.eye(len(features), dtype=np.float64)
    if graph_laplacian is not None and graph_l2 > 0.0:
        graph_operator += graph_l2 * graph_laplacian
    system = regularization * np.eye(len(features), dtype=np.float64)
    system += graph_operator @ gram
    dual = np.linalg.solve(system, targets)
    weights = augmented.T @ dual
    if not np.isfinite(weights).all():
        raise RuntimeError("ridge projection produced non-finite weights")
    return weights


def fit_projected_binomial_irt(
    features: np.ndarray,
    passed: np.ndarray,
    total: np.ndarray,
    latent_dimension: int,
    *,
    projection_l2: float,
    monotone_luna: bool = False,
    graph_laplacian: np.ndarray | None = None,
    graph_l2: float = 0.0,
) -> ProjectedBinomialIrtFit:
    """Fit exact task IRT once, then project its latents onto pre-call features."""
    _validate_counts(passed, total)
    _validate_features(features, len(passed))
    _validate_graph_laplacian(graph_laplacian, len(passed), graph_l2)
    task_fit = fit_binomial_irt(
        passed,
        total,
        latent_dimension,
        monotone_luna=monotone_luna,
    )
    difficulty_weights = _ridge_projection(
        features,
        task_fit.difficulties,
        regularization=projection_l2,
        graph_laplacian=graph_laplacian,
        graph_l2=graph_l2,
    )
    log_discrimination_weights = _ridge_projection(
        features,
        task_fit.log_discriminations,
        regularization=projection_l2,
    )
    augmented = _augment_features(features)
    difficulty_residual = augmented @ difficulty_weights - task_fit.difficulties
    discrimination_residual = (
        augmented @ log_discrimination_weights - task_fit.log_discriminations
    )
    projection_loss = float(
        np.mean(difficulty_residual**2) + np.mean(discrimination_residual**2)
    )
    return ProjectedBinomialIrtFit(
        abilities=task_fit.abilities.copy(),
        difficulty_weights=difficulty_weights.copy(),
        log_discrimination_weights=log_discrimination_weights.copy(),
        loss=task_fit.loss + projection_loss,
        projection_loss=projection_loss,
        iterations=task_fit.iterations,
    )


def predict_probabilities(fit: BinomialIrtFit) -> np.ndarray:
    """Predict fitted task-by-arm pass probabilities without serializing fit state."""
    if fit.abilities.ndim != 2 or fit.log_discriminations.ndim != 2:
        raise ValueError("IRT abilities and discriminations must be matrices")
    if fit.difficulties.shape != (len(fit.log_discriminations),):
        raise ValueError("IRT difficulty count does not match task discriminations")
    if fit.abilities.shape[1] != fit.log_discriminations.shape[1]:
        raise ValueError("IRT latent dimensions do not match")
    discriminations = np.exp(fit.log_discriminations)
    return expit(discriminations @ fit.abilities.T - fit.difficulties[:, None])


def predict_feature_probabilities(
    fit: FeatureBinomialIrtFit,
    features: np.ndarray,
) -> np.ndarray:
    """Predict arm probabilities for unseen tasks from pre-call features only."""
    if fit.abilities.ndim != 2 or fit.discrimination_weights.ndim != 2:
        raise ValueError("feature IRT abilities and discrimination weights must be matrices")
    if fit.difficulty_weights.ndim != 1:
        raise ValueError("feature IRT difficulty weights must be a vector")
    expected_feature_count = len(fit.difficulty_weights) - 1
    _validate_features(features, len(features))
    if features.shape[1] != expected_feature_count:
        raise ValueError("prediction features do not match fitted feature count")
    if fit.discrimination_weights.shape != (
        expected_feature_count + 1,
        fit.abilities.shape[1],
    ):
        raise ValueError("feature IRT fitted dimensions do not match")
    augmented = _augment_features(features)
    difficulties = augmented @ fit.difficulty_weights
    discriminations = _softplus(augmented @ fit.discrimination_weights)
    return expit(discriminations @ fit.abilities.T - difficulties[:, None])


def predict_projected_probabilities(
    fit: ProjectedBinomialIrtFit,
    features: np.ndarray,
) -> np.ndarray:
    """Predict unseen-task arm probabilities from projected task IRT latents."""
    if fit.abilities.ndim != 2 or fit.log_discrimination_weights.ndim != 2:
        raise ValueError("projected IRT abilities and discrimination weights must be matrices")
    if fit.difficulty_weights.ndim != 1:
        raise ValueError("projected IRT difficulty weights must be a vector")
    expected_feature_count = len(fit.difficulty_weights) - 1
    _validate_features(features, len(features))
    if features.shape[1] != expected_feature_count:
        raise ValueError("prediction features do not match projected feature count")
    if fit.log_discrimination_weights.shape != (
        expected_feature_count + 1,
        fit.abilities.shape[1],
    ):
        raise ValueError("projected IRT fitted dimensions do not match")
    augmented = _augment_features(features)
    difficulties = augmented @ fit.difficulty_weights
    log_discriminations = np.clip(
        augmented @ fit.log_discrimination_weights,
        -3.0,
        3.0,
    )
    discriminations = np.exp(log_discriminations)
    return expit(discriminations @ fit.abilities.T - difficulties[:, None])
