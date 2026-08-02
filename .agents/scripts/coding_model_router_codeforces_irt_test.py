"""Tests for the Codeforces graded-response item-response router."""

from __future__ import annotations

import numpy as np
from coding_model_router_codeforces_irt import (
    ARMS,
    Candidate,
    _choose,
    _fit_irt,
    _irt_loss_and_gradient,
)


def test_irt_gradient_matches_finite_difference() -> None:
    rewards = np.asarray(
        [
            [0.1, 0.2, 0.4, 0.7, 0.9],
            [0.0, 0.1, 0.3, 0.5, 0.8],
            [0.4, 0.5, 0.7, 0.8, 1.0],
        ],
        dtype=np.float64,
    )
    parameters = np.linspace(-0.4, 0.4, len(ARMS) + 2 * len(rewards))
    _, gradient = _irt_loss_and_gradient(parameters, rewards)
    epsilon = 1e-6
    numerical = np.zeros_like(parameters)
    for index in range(len(parameters)):
        upper = parameters.copy()
        lower = parameters.copy()
        upper[index] += epsilon
        lower[index] -= epsilon
        upper_loss = _irt_loss_and_gradient(upper, rewards)[0]
        lower_loss = _irt_loss_and_gradient(lower, rewards)[0]
        numerical[index] = (upper_loss - lower_loss) / (2.0 * epsilon)
    np.testing.assert_allclose(gradient, numerical, atol=1e-6, rtol=1e-5)


def test_graded_irt_recovers_arm_ability_order() -> None:
    difficulties = np.linspace(-1.5, 1.5, 24)
    discriminations = np.linspace(0.6, 1.8, 24)
    abilities = np.asarray([-1.2, -0.5, 0.1, 0.7, 1.4])
    rewards = 0.05 * np.round(
        20.0
        * (
            1.0
            / (
                1.0
                + np.exp(-discriminations[:, None] * (abilities[None, :] - difficulties[:, None]))
            )
        )
    )
    fitted = _fit_irt(rewards)
    assert np.all(np.diff(fitted.abilities) > 0.0)
    assert np.isfinite(fitted.loss)
    assert np.all(np.isfinite(fitted.difficulties))


def test_scalarizations_use_probability_and_cost() -> None:
    probabilities = np.asarray(
        [
            [0.88, 0.90, 0.91, 0.92, 0.93],
            [0.20, 0.40, 0.70, 0.91, 0.95],
        ]
    )
    costs = np.asarray([1.0, 2.0, 4.0, 8.0, 16.0])
    cost_first = Candidate(512, 1.0, "linear", 0.70)
    quality_first = Candidate(512, 1.0, "linear", 0.99)
    cheap_choices = _choose(probabilities, costs, cost_first)
    quality_choices = _choose(probabilities, costs, quality_first)
    assert cheap_choices[0] == 0
    assert quality_choices[1] >= cheap_choices[1]
    chebyshev = Candidate(512, 1.0, "chebyshev", 0.95)
    choices = _choose(probabilities, costs, chebyshev)
    assert choices.shape == (2,)
    assert np.all((choices >= 0) & (choices < len(ARMS)))
