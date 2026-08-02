"""Tests for the cost/quality Pareto curve: pure offline aggregation, no spend."""

from __future__ import annotations

import pytest

from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.pareto import ParetoCurve, held_out_curve, pareto_curve
from wmo.optimize.policy import EmbedderSpec, RoutingPolicy
from wmo.providers.base import ProviderKind, TokenUsage
from wmo.providers.pool import PoolEntry


def _row(
    sid: str,
    model: str,
    *,
    reward: float | None,
    cost: float = 0.01,
    episode: int = 0,
) -> ScenarioOutcome:
    return ScenarioOutcome(
        scenario_id=sid,
        task=f"task for {sid}",
        model=model,
        episode=episode,
        reward=reward,
        success=reward is not None and reward >= 0.5,
        usage=TokenUsage(input_tokens=100, output_tokens=50),
        cost_usd=cost,
        call_seconds=[0.5, 0.5],
        error=None if reward is not None else "sandbox timeout",
    )


def _matrix(outcomes: list[ScenarioOutcome]) -> OutcomeMatrix:
    return OutcomeMatrix(
        pool=[
            PoolEntry(
                name="cheap",
                kind=ProviderKind.OPENAI,
                model="cheap-model",
                input_per_mtok=0.1,
                output_per_mtok=0.2,
            ),
            PoolEntry(
                name="mid",
                kind=ProviderKind.OPENAI,
                model="mid-model",
                input_per_mtok=1.0,
                output_per_mtok=2.0,
            ),
            PoolEntry(name="strong", kind=ProviderKind.ANTHROPIC, model="claude-fable-5"),
        ],
        outcomes=outcomes,
    )


def _three_model_matrix() -> OutcomeMatrix:
    # cheap: $0.01/completed at reward 0.5; mid: dominated (dearer than strong, worse);
    # strong: $0.06/completed at reward 1.0.
    return _matrix(
        [
            _row("s1", "cheap", reward=1.0, cost=0.01),
            _row("s2", "cheap", reward=0.0, cost=0.01),
            _row("s1", "mid", reward=0.5, cost=0.08),
            _row("s2", "mid", reward=0.5, cost=0.08),
            _row("s1", "strong", reward=1.0, cost=0.03),
            _row("s2", "strong", reward=1.0, cost=0.03),
        ]
    )


def test_frontier_marks_non_dominated_points_and_keeps_dominated_ones() -> None:
    curve = pareto_curve(_three_model_matrix(), judge="test-judge")

    by_id = {p.id: p for p in curve.points}
    assert by_id["cheap"].on_frontier  # cheapest per completed task
    assert by_id["strong"].on_frontier  # best reward
    assert not by_id["mid"].on_frontier  # dearer than strong AND worse: dominated
    assert len(curve.points) == 3  # dominated points stay visible, never dropped


def test_recommended_without_policy_is_the_frontiers_best_quality_point() -> None:
    curve = pareto_curve(_three_model_matrix(), judge="test-judge")

    assert curve.recommended == "strong"


def test_linear_policy_adds_one_routed_point_without_a_fake_dial() -> None:
    matrix = _three_model_matrix()
    policy = RoutingPolicy(
        kind="linear",
        default_model="strong",
        pool=matrix.pool,
        embedder=EmbedderSpec(dim=8),
        linear_weak_model="cheap",
        linear_strong_model="strong",
        linear_weak_weights=[0.0] * 8,
        linear_strong_weights=[0.0] * 8,
        linear_weak_bias=0.0,
        linear_strong_bias=1.0,
        linear_threshold=0.5,
    )

    curve = held_out_curve(matrix, policy, judge="test-judge")

    routed = [point for point in curve.points if point.kind == "routed"]
    assert len(routed) == 1
    assert routed[0].id == "routed"
    assert routed[0].dial is None
    assert routed[0].mix == {"strong": 2}
    assert curve.recommended == "routed"


def test_complete_flag_is_false_while_any_candidate_has_unscored_cells() -> None:
    outcomes = _three_model_matrix().outcomes + [_row("s3", "cheap", reward=None)]
    curve = pareto_curve(_matrix(outcomes), judge="test-judge")

    assert curve.complete is False


def test_full_matrix_reports_complete() -> None:
    curve = pareto_curve(_three_model_matrix(), judge="test-judge")

    assert curve.complete is True
    assert curve.n_scenarios == 2
    assert curve.judge == "test-judge"
    assert curve.provenance == "wm_simulated"


def test_scenario_restriction_applies_to_every_point() -> None:
    curve = pareto_curve(_three_model_matrix(), judge="test-judge", scenario_ids=["s1"])

    assert curve.n_scenarios == 1
    assert all(p.n_scenarios == 1 for p in curve.points)
    # On s1 alone, cheap completes 1 task at $0.01 and reward 1.0: it dominates strong.
    by_id = {p.id: p for p in curve.points}
    assert by_id["cheap"].on_frontier
    assert not by_id["strong"].on_frontier


def test_unmeasured_scenario_ids_are_refused() -> None:
    with pytest.raises(ValueError, match="never measured"):
        pareto_curve(_three_model_matrix(), judge="test-judge", scenario_ids=["ghost"])


def test_nothing_completed_point_is_unplaced_but_visible() -> None:
    outcomes = _three_model_matrix().outcomes + [
        _row("s1", "mid", reward=0.0, cost=0.05, episode=1),
    ]
    # Rebuild mid as all-failures: replace its rows with reward-0 episodes.
    outcomes = [o for o in outcomes if o.model != "mid"] + [
        _row("s1", "mid", reward=0.0, cost=0.05),
        _row("s2", "mid", reward=0.0, cost=0.05),
    ]
    curve = pareto_curve(_matrix(outcomes), judge="test-judge")

    mid = next(p for p in curve.points if p.id == "mid")
    assert mid.cost_per_completed_task_usd is None
    assert not mid.on_frontier


def test_serializes_for_the_wire() -> None:
    curve = pareto_curve(_three_model_matrix(), judge="test-judge")

    restored = ParetoCurve.model_validate_json(curve.model_dump_json())
    assert restored == curve


def test_survivorship_cannot_take_the_frontier() -> None:
    """An arm judged only on the episodes it survived must not dominate the band.

    Repro from the real tau2 grid: qwen3.5-9b lost most episodes to its own empty
    replies, aced the survivors, and its (cost, reward) beat the anchor measured on
    everything. Survivorship is not dominance: the under-covered point stays plotted
    and labeled, but never holds the frontier and is never recommended.
    """
    outcomes = [
        # "cheap" scores on only 1 of 4 scenarios (the rest unscored) and aces it.
        _row("s1", "cheap", reward=1.0, cost=0.001),
        _row("s2", "cheap", reward=None),
        _row("s3", "cheap", reward=None),
        _row("s4", "cheap", reward=None),
        # "strong" is measured on the whole band.
        _row("s1", "strong", reward=1.0, cost=0.03),
        _row("s2", "strong", reward=1.0, cost=0.03),
        _row("s3", "strong", reward=0.5, cost=0.03),
        _row("s4", "strong", reward=1.0, cost=0.03),
    ]
    curve = pareto_curve(_matrix(outcomes), judge="test-judge")

    by_id = {p.id: p for p in curve.points}
    assert not by_id["cheap"].frontier_eligible
    assert not by_id["cheap"].on_frontier
    assert by_id["strong"].on_frontier
    assert curve.recommended == "strong"
    assert not curve.complete
    assert "coverage" in curve.frontier_rule


def test_full_coverage_band_is_untouched_by_the_eligibility_rule() -> None:
    curve = pareto_curve(_three_model_matrix(), judge="test-judge")

    assert all(p.frontier_eligible for p in curve.points)


def test_a_pinned_policy_still_ships_the_workload_frontier() -> None:
    """A pin has no dial to replay, but the workload's frontier exists anyway.

    `route report` was skipping the pareto write for static policies, so an
    honestly-pinned endpoint shipped NO curve for the product's dial UI. The
    curve now carries the model points over the full matrix and recommends
    what the product mounts today: the pinned model itself.
    """
    matrix = _three_model_matrix()
    policy = RoutingPolicy(
        kind="static",
        default_model="strong",
        pool=[PoolEntry(name="strong", kind=ProviderKind.ANTHROPIC, model="claude-fable-5")],
    )

    curve = held_out_curve(matrix, policy, judge="test-judge", provenance="real_episode")

    assert curve.recommended == "strong"
    assert all(point.kind == "model" for point in curve.points)
    assert {point.id for point in curve.points} == {"cheap", "mid", "strong"}
    assert curve.provenance == "real_episode"


def test_an_ineligible_pin_is_not_recommended_by_its_own_curve() -> None:
    """The artifact must not contradict its own frontier rule.

    A pinned model whose coverage the rule disqualifies (the survivorship
    case) keeps the curve's honest recommendation instead of overriding it.
    """
    outcomes = [
        _row("s1", "cheap", reward=1.0, cost=0.001),
        _row("s2", "cheap", reward=None),
        _row("s3", "cheap", reward=None),
        _row("s4", "cheap", reward=None),
        _row("s1", "strong", reward=1.0, cost=0.03),
        _row("s2", "strong", reward=1.0, cost=0.03),
        _row("s3", "strong", reward=0.5, cost=0.03),
        _row("s4", "strong", reward=1.0, cost=0.03),
    ]
    policy = RoutingPolicy(
        kind="static",
        default_model="cheap",
        pool=[
            PoolEntry(
                name="cheap",
                kind=ProviderKind.OPENAI,
                model="m",
                input_per_mtok=1,
                output_per_mtok=1,
            )
        ],
    )

    curve = held_out_curve(_matrix(outcomes), policy, judge="test-judge", provenance="real_episode")

    by_id = {p.id: p for p in curve.points}
    assert not by_id["cheap"].frontier_eligible
    assert curve.recommended == "strong"  # the frontier's answer, not the ineligible pin


def test_unplaceable_points_keep_their_honest_coverage_flag() -> None:
    """frontier_eligible reports COVERAGE; placement is a different exclusion."""
    outcomes = [
        _row("s1", "cheap", reward=0.0, cost=0.01),
        _row("s2", "cheap", reward=0.0, cost=0.01),
        _row("s1", "strong", reward=1.0, cost=0.03),
        _row("s2", "strong", reward=1.0, cost=0.03),
    ]
    curve = pareto_curve(_matrix(outcomes), judge="test-judge")

    by_id = {p.id: p for p in curve.points}
    # cheap completed nothing: unplaceable, never on the frontier - but its
    # coverage is full, so the coverage flag must not misexplain it.
    assert by_id["cheap"].cost_per_completed_task_usd is None
    assert not by_id["cheap"].on_frontier
    assert by_id["cheap"].frontier_eligible
