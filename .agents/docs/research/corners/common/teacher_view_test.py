"""Tests for the teacher-verdict view, on a synthetic matrix with a known answer.

Pinned semantics (the handoff contract with wmo.optimize.teacher depends on them): the
baseline is the cheapest candidate by effective cost per COMPLETED task, gains are paired
per scenario, and "headroom" is only granted under the shared CI-and-noise-floor rule.
"""

from __future__ import annotations

import pytest
from teacher_view import cheapest_scored_candidate, teacher_verdict_rows

from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.providers.base import ProviderKind
from wmo.providers.pool import PoolEntry


def _entry(name: str) -> PoolEntry:
    return PoolEntry(
        name=name,
        kind=ProviderKind.OPENAI,
        model=f"custom-{name}",
        tier="open",
        input_per_mtok=0.1,
        output_per_mtok=0.2,
    )


def _row(sid: str, model: str, reward: float, cost: float) -> ScenarioOutcome:
    return ScenarioOutcome(
        scenario_id=sid,
        task=f"task {sid}",
        model=model,
        reward=reward,
        success=reward >= 0.5,
        cost_usd=cost,
    )


def _matrix() -> OutcomeMatrix:
    # cheap: solves half the scenarios at low cost. strong: solves everything, pricier.
    # peer: identical rewards to cheap (the cycle-1 shape: no headroom to distill).
    scenarios = [f"s{i}" for i in range(8)]
    outcomes: list[ScenarioOutcome] = []
    for i, sid in enumerate(scenarios):
        cheap_reward = 1.0 if i % 2 == 0 else 0.0
        outcomes.append(_row(sid, "cheap", cheap_reward, cost=0.01))
        outcomes.append(_row(sid, "strong", 1.0, cost=0.05))
        outcomes.append(_row(sid, "peer", cheap_reward, cost=0.04))
    return OutcomeMatrix(pool=[_entry("cheap"), _entry("strong"), _entry("peer")], outcomes=outcomes)


def test_cheapest_is_by_effective_cost_per_completed_task() -> None:
    # cheap completes 4 tasks at $0.08 total = $0.02/task; strong completes 8 at $0.40 =
    # $0.05/task; peer completes 4 at $0.32 = $0.08/task.
    assert cheapest_scored_candidate(_matrix()) == "cheap"


def test_headroom_goes_to_the_strong_model_and_the_peer_reads_no_effect() -> None:
    rows = {row.model: row for row in teacher_verdict_rows(_matrix())}
    assert rows["strong"].verdict == "headroom"
    assert rows["strong"].delta.mean_delta == pytest.approx(0.5)
    assert rows["cheap"].verdict == "baseline"
    # A peer with identical per-scenario rewards has nothing to distill: exactly cycle 1.
    assert rows["peer"].verdict == "no measurable effect"
    assert rows["peer"].delta.mean_delta == pytest.approx(0.0)


def test_rows_sort_best_gain_first_and_carry_cost() -> None:
    rows = teacher_verdict_rows(_matrix())
    assert [row.model for row in rows][0] == "strong"
    assert rows[0].cost_per_completed_task_usd == pytest.approx(0.05)
    assert all(row.baseline == "cheap" for row in rows)


def test_explicit_baseline_overrides_the_cheapest_rule() -> None:
    rows = {row.model: row for row in teacher_verdict_rows(_matrix(), baseline="strong")}
    assert rows["strong"].verdict == "baseline"
    assert rows["cheap"].verdict == "below baseline"


def test_a_matrix_where_nothing_completed_refuses_to_name_a_baseline() -> None:
    scenarios = [f"s{i}" for i in range(3)]
    matrix = OutcomeMatrix(
        pool=[_entry("cheap")],
        outcomes=[_row(sid, "cheap", 0.0, cost=0.01) for sid in scenarios],
    )
    with pytest.raises(ValueError, match="no cheapest candidate"):
        cheapest_scored_candidate(matrix)
