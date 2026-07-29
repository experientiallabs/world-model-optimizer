"""Tests for the corner analyses' shared statistics, pinning the published conventions.

The sign-test pin reproduces the cycle-1 result note's number exactly (5 down, 2 up over the
7 tasks that moved gives two-sided p = 0.453), so the module and the published note can never
silently disagree.
"""

from __future__ import annotations

import pytest
from stats import (
    NOISE_FLOOR_REWARD,
    mean_with_ci,
    paired_delta,
    sign_agreement,
    sign_test_p,
    spearman_model_means,
)


def test_sign_test_matches_the_cycle1_result_note() -> None:
    assert sign_test_p(2, 5) == pytest.approx(58 / 128)


def test_sign_test_is_symmetric_and_exact_on_small_cases() -> None:
    assert sign_test_p(5, 2) == sign_test_p(2, 5)
    assert sign_test_p(0, 5) == pytest.approx(2 / 32)
    assert sign_test_p(1, 1) == pytest.approx(1.0)


def test_sign_test_refuses_zero_movers() -> None:
    with pytest.raises(ValueError, match="at least one scenario"):
        sign_test_p(0, 0)


def test_mean_with_ci_averages_per_scenario_not_per_episode() -> None:
    # Scenario a has three episodes, scenario b one: the mean must weight scenarios equally
    # (0.0 and 1.0 -> 0.5), not pool episodes (which would give 0.25).
    result = mean_with_ci({"a": [0.0, 0.0, 0.0], "b": [1.0]}, resamples=200, seed=7)
    assert result.mean == pytest.approx(0.5)
    assert result.n_scenarios == 2
    assert result.n_episodes == 4
    assert result.ci_low <= result.mean <= result.ci_high


def test_mean_with_ci_is_deterministic_under_a_seed() -> None:
    data = {f"s{i}": [i / 10.0] for i in range(10)}
    first = mean_with_ci(data, resamples=500, seed=3)
    second = mean_with_ci(data, resamples=500, seed=3)
    assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)


def test_mean_with_ci_rejects_empty_and_unscored_shapes() -> None:
    with pytest.raises(ValueError, match="no scenarios"):
        mean_with_ci({})
    with pytest.raises(ValueError, match="no rewards"):
        mean_with_ci({"a": []})


def test_paired_delta_pairs_on_the_intersection_and_counts_movement() -> None:
    arm = {"a": [1.0], "b": [0.0], "c": [1.0], "only-arm": [1.0]}
    anchor = {"a": [0.0], "b": [0.0], "c": [0.0], "only-anchor": [0.0]}
    result = paired_delta(arm, anchor, resamples=200, seed=1)
    assert result.n_pairs == 3  # the two one-sided scenarios never enter
    assert (result.n_up, result.n_down, result.n_tied) == (2, 0, 1)
    assert result.mean_delta == pytest.approx(2 / 3)
    assert result.sign_test_p == pytest.approx(sign_test_p(2, 0))
    assert not result.within_noise_floor
    assert set(result.scenario_deltas) == {"a", "b", "c"}


def test_verdict_requires_both_a_clear_ci_and_a_mean_outside_the_floor() -> None:
    scenarios = {f"s{i}": [1.0] for i in range(6)}
    zeros = {f"s{i}": [0.0] for i in range(6)}
    big = paired_delta(scenarios, zeros, resamples=100)
    assert big.verdict == "measurable"
    # Consistent direction but tiny magnitude: significant is not the same as headline-able.
    tiny = paired_delta(
        {f"s{i}": [0.51] for i in range(6)}, {f"s{i}": [0.5] for i in range(6)}, resamples=100
    )
    assert tiny.verdict == "within_noise_floor"
    # The cycle-1 shape: a large mean whose CI spans zero is noise, never a regression.
    mixed = paired_delta(
        {"a": [1.0], "b": [0.0], "c": [1.0], "d": [0.0]},
        {"a": [0.0], "b": [1.0], "c": [0.0], "d": [1.0]},
        resamples=200,
    )
    assert mixed.verdict == "no_effect"


def test_paired_delta_flags_the_noise_floor_and_handles_no_movement() -> None:
    inside = paired_delta({"a": [0.51], "b": [0.5]}, {"a": [0.5], "b": [0.5]}, resamples=100)
    assert inside.within_noise_floor
    assert inside.noise_floor == NOISE_FLOOR_REWARD
    frozen = paired_delta({"a": [0.5]}, {"a": [0.5]}, resamples=100)
    assert frozen.sign_test_p is None
    assert frozen.n_tied == 1


def test_paired_delta_refuses_disjoint_scenario_sets() -> None:
    with pytest.raises(ValueError, match="both sides"):
        paired_delta({"a": [1.0]}, {"b": [0.0]})


def test_sign_agreement_counts_direction_and_excludes_ties() -> None:
    a = {"s1": 0.2, "s2": -0.1, "s3": 0.3, "s4": 0.0}
    b = {"s1": 0.1, "s2": 0.2, "s3": 0.05, "s4": 0.5}
    result = sign_agreement(a, b)
    assert result.compared == 3  # s4 is a tie on side a
    assert result.agree == 2  # s1 and s3 agree, s2 disagrees
    assert result.ties_excluded == 1
    assert result.fraction == pytest.approx(2 / 3)


def test_sign_agreement_with_all_ties_reports_none_not_a_number() -> None:
    result = sign_agreement({"s1": 0.0}, {"s1": 0.4})
    assert result.fraction is None
    assert result.ties_excluded == 1


def test_spearman_is_exact_on_monotone_data_and_carries_its_caveat() -> None:
    up = spearman_model_means([(1.0, 10.0), (2.0, 20.0), (3.0, 30.0)])
    assert up.rho == pytest.approx(1.0)
    down = spearman_model_means([(1.0, 30.0), (2.0, 20.0), (3.0, 10.0)])
    assert down.rho == pytest.approx(-1.0)
    assert "descriptive only" in up.caveat
    assert "n=3" in up.caveat


def test_spearman_handles_ties_and_refuses_tiny_n() -> None:
    tied = spearman_model_means([(1.0, 5.0), (1.0, 5.0), (2.0, 9.0), (3.0, 7.0)])
    assert -1.0 <= tied.rho <= 1.0
    constant = spearman_model_means([(1.0, 5.0), (2.0, 5.0), (3.0, 5.0)])
    assert constant.rho == 0.0  # a constant side has no direction to correlate
    with pytest.raises(ValueError, match="vacuous"):
        spearman_model_means([(1.0, 2.0), (3.0, 4.0)])
