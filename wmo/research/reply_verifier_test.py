"""Tests for the distilled reply verifier (ridge heads, pairwise design, selector adaptation)."""

from __future__ import annotations

import numpy as np
import pytest

from wmo.optimize.outcomes import ScenarioOutcome
from wmo.research.reply_verifier import (
    episode_key,
    fit_absolute,
    fit_pairwise,
    fit_projection,
    pairwise_design,
    scenario_folds,
    shuffled_rewards,
    verifier_selector,
)


def _outcome(sid: str, model: str, episode: int, reward: float) -> ScenarioOutcome:
    return ScenarioOutcome(
        scenario_id=sid, task=f"t-{sid}", model=model, episode=episode, reward=reward
    )


def _cells() -> tuple[list[list[ScenarioOutcome]], dict[tuple[str, str, int], np.ndarray]]:
    """Two cells. Dim 0 tracks reward WITHIN a cell; dim 1 is per-cell difficulty, constant inside.

    Difficulty also anti-correlates with reward across cells (the easy cell scores higher), which
    is the between/within confound the pairwise head is meant to be immune to.
    """
    cells = [
        [_outcome("easy", "m", 0, 0.0), _outcome("easy", "m", 1, 1.0)],
        [_outcome("hard", "m", 0, 0.0), _outcome("hard", "m", 1, 0.5)],
    ]
    vectors = {
        ("easy", "m", 0): np.array([0.0, 1.0]),
        ("easy", "m", 1): np.array([1.0, 1.0]),
        ("hard", "m", 0): np.array([0.0, 9.0]),
        ("hard", "m", 1): np.array([1.0, 9.0]),
    }
    return cells, vectors


def test_episode_key_identifies_one_rollout() -> None:
    assert episode_key(_outcome("s0", "cheap", 1, 0.5)) == ("s0", "cheap", 1)


def test_pairwise_design_is_sign_symmetric_and_drops_ties() -> None:
    cells, vectors = _cells()
    cells.append([_outcome("tied", "m", 0, 0.4), _outcome("tied", "m", 1, 0.4)])
    vectors[("tied", "m", 0)] = np.array([0.0, 3.0])
    vectors[("tied", "m", 1)] = np.array([1.0, 3.0])
    differences, gaps = pairwise_design(cells, vectors)
    # Two informative cells x both directions = 4 rows; the tied cell contributes nothing.
    assert differences.shape == (4, 2)
    assert gaps.tolist() == [-1.0, 1.0, -0.5, 0.5]
    # Sign symmetry: rows come in +/- pairs, so every column sums to zero.
    assert differences.sum(axis=0).tolist() == [0.0, 0.0]
    # Difficulty (dim 1) cancels within every cell, so the design cannot see it at all.
    assert differences[:, 1].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_pairwise_head_recovers_the_within_cell_direction() -> None:
    cells, vectors = _cells()
    differences, gaps = pairwise_design(cells, vectors)
    verifier = fit_pairwise(differences, gaps, alpha=1e-6)
    coefficients = np.asarray(verifier.coef)
    assert verifier.mode == "pairwise"
    assert verifier.intercept == 0.0
    assert coefficients[0] > 0.5  # the signal dimension
    assert coefficients[1] == pytest.approx(0.0, abs=1e-9)  # difficulty is invisible, not learned


def test_absolute_head_recovers_a_linear_reward_function() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 3))
    rewards = features @ np.array([1.0, -2.0, 0.0]) + 0.25
    verifier = fit_absolute(features, rewards, alpha=1e-8)
    assert verifier.mode == "absolute"
    assert verifier.train_rows == 200
    assert np.allclose(verifier.coef, [1.0, -2.0, 0.0], atol=1e-4)
    assert verifier.intercept == pytest.approx(0.25, abs=1e-4)


def test_ridge_alpha_shrinks_coefficients() -> None:
    rng = np.random.default_rng(1)
    features = rng.normal(size=(50, 4))
    rewards = features @ np.array([3.0, 0.0, 0.0, 0.0])
    loose = fit_absolute(features, rewards, alpha=1e-8)
    tight = fit_absolute(features, rewards, alpha=1e4)
    assert abs(tight.coef[0]) < abs(loose.coef[0])


def test_verifier_selector_ranks_the_higher_score_first() -> None:
    cells, vectors = _cells()
    differences, gaps = pairwise_design(cells, vectors)
    key = verifier_selector(fit_pairwise(differences, gaps, alpha=1e-6), vectors)
    # posthoc_bounds ranks LOWEST key first, so the better episode must get the smaller key.
    for episodes in cells:
        worse, better = episodes
        assert key(better) < key(worse)


def test_verifier_selector_is_neutral_on_a_missing_embedding() -> None:
    cells, vectors = _cells()
    differences, gaps = pairwise_design(cells, vectors)
    key = verifier_selector(fit_pairwise(differences, gaps, alpha=1e-6), vectors)
    assert key(_outcome("unseen", "m", 0, 0.0)) == (0.0,)


def test_projection_of_a_difference_equals_the_difference_of_projections() -> None:
    """The invariant that keeps a projected pairwise fit consistent with `score`."""
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(40, 12))
    projection = fit_projection(vectors, components=5)
    first, second = vectors[0], vectors[1]
    assert np.allclose(
        projection.apply_difference(first - second),
        projection.apply(first) - projection.apply(second),
    )
    # Subtracting the mean twice is the bug this guards: `apply` on a difference is NOT equal.
    assert not np.allclose(
        projection.apply(first - second), projection.apply_difference(first - second)
    )


def test_fit_projection_yields_orthonormal_components_and_reduces_width() -> None:
    rng = np.random.default_rng(2)
    projection = fit_projection(rng.normal(size=(30, 8)), components=3)
    components = np.asarray(projection.components)
    assert components.shape == (3, 8)
    assert np.allclose(components @ components.T, np.eye(3), atol=1e-9)
    assert projection.apply(rng.normal(size=(4, 8))).shape == (4, 3)
    # Asking for more components than the data supports is capped, not an error.
    assert len(fit_projection(rng.normal(size=(3, 8)), components=50).components) <= 3


def test_projected_pairwise_head_still_ranks_correctly() -> None:
    cells, vectors = _cells()
    differences, gaps = pairwise_design(cells, vectors)
    projection = fit_projection(np.asarray(list(vectors.values())), components=2)
    verifier = fit_pairwise(differences, gaps, alpha=1e-6, projection=projection)
    assert verifier.projection is not None
    assert len(verifier.coef) == 2  # the head lives in the projected basis
    key = verifier_selector(verifier, vectors)
    for episodes in cells:
        worse, better = episodes
        assert key(better) < key(worse)


def test_scenario_folds_partition_scenarios_without_splitting_any() -> None:
    scenarios = [f"s{index}" for index in range(23)]
    folds = scenario_folds(scenarios * 2, folds=5, seed=0)
    assert len(folds) == 5
    flat = [sid for fold in folds for sid in fold]
    assert sorted(flat) == sorted(scenarios)  # every scenario once, none duplicated across folds
    assert all(len(set(fold)) == len(fold) for fold in folds)


def test_scenario_folds_are_seed_deterministic() -> None:
    scenarios = [f"s{index}" for index in range(20)]
    assert scenario_folds(scenarios, 4, seed=7) == scenario_folds(scenarios, 4, seed=7)
    assert scenario_folds(scenarios, 4, seed=7) != scenario_folds(scenarios, 4, seed=8)


def test_shuffled_rewards_permutes_without_changing_the_multiset() -> None:
    rewards = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.0])
    permuted = shuffled_rewards(rewards, seed=3)
    assert sorted(permuted.tolist()) == sorted(rewards.tolist())
    assert permuted.tolist() != rewards.tolist()  # seed 3 does reorder this input
    assert rewards.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0, 1.0]  # input untouched
