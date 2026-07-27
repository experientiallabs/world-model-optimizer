"""Tests for the routing ablation run records (uniform evaluation + explain blocks)."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.providers.base import ProviderKind, TokenUsage
from wmo.providers.pool import PoolEntry
from wmo.research.routing_runs import (
    Finish,
    RunRecord,
    append_run,
    evaluate_call_sequences,
    evaluate_choices,
    run_report,
)


def _matrix() -> OutcomeMatrix:
    pool = [
        PoolEntry(
            name="fast-cheap",
            kind=ProviderKind.OPENAI,
            model="fc",
            input_per_mtok=1.0,
            output_per_mtok=1.0,
        ),
        PoolEntry(
            name="slow-pricey",
            kind=ProviderKind.OPENAI,
            model="sp",
            input_per_mtok=1.0,
            output_per_mtok=1.0,
        ),
    ]
    outcomes = []
    for index in range(4):
        sid = f"s{index}"
        outcomes.append(
            ScenarioOutcome(
                scenario_id=sid,
                task=f"t{index}",
                model="fast-cheap",
                reward=1.0 if index < 3 else 0.0,
                success=index < 3,
                usage=TokenUsage(input_tokens=100, output_tokens=20),
                cost_usd=0.001,
                call_seconds=[0.5],
            )
        )
        outcomes.append(
            ScenarioOutcome(
                scenario_id=sid,
                task=f"t{index}",
                model="slow-pricey",
                reward=1.0,
                success=True,
                usage=TokenUsage(input_tokens=100, output_tokens=200),
                cost_usd=0.02,
                call_seconds=[2.0],
            )
        )
    return OutcomeMatrix(pool=pool, outcomes=outcomes)


def test_evaluate_choices_full_explain_block() -> None:
    matrix = _matrix()
    ids = matrix.scenario_ids()
    # Route s0/s1 to the fast-cheap model, s2/s3 to the slow-pricey one.
    result = evaluate_choices(
        matrix, ids, lambda sid: "fast-cheap" if sid in ("s0", "s1") else "slow-pricey"
    )
    assert result.accuracy == pytest.approx(1.0)
    assert result.cost_per_call == pytest.approx((0.001 + 0.001 + 0.02 + 0.02) / 4)
    assert result.latency_p50_s == pytest.approx((0.5 + 2.0) / 2)  # median of .5,.5,2,2
    assert result.model_mix == {"fast-cheap": 0.5, "slow-pricey": 0.5}
    assert result.tokens_by_model["fast-cheap"] == {"input": 200, "output": 40}
    assert result.tokens_by_model["slow-pricey"] == {"input": 200, "output": 400}
    assert result.per_model_latency_p50_s["fast-cheap"] == pytest.approx(0.5)
    assert result.per_model_latency_p50_s["slow-pricey"] == pytest.approx(2.0)


def test_run_record_persists_and_report_explains(tmp_path: Path) -> None:
    matrix = _matrix()
    ids = matrix.scenario_ids()
    routed = evaluate_choices(matrix, ids, lambda sid: "fast-cheap")
    baseline = evaluate_choices(matrix, ids, lambda sid: "slow-pricey")
    record = RunRecord(
        run_id="r1",
        ts="2026-07-24T00:00:00Z",
        matrix="synthetic",
        variant="static",
        params={"model": "fast-cheap"},
        split_seed=0,
        fit_scenarios=0,
        test_scenarios=len(ids),
        result=routed,
        baselines={"best_single": baseline},
    )
    runs = tmp_path / "runs.jsonl"
    append_run(record, runs)
    append_run(record, runs)
    assert len(runs.read_text().splitlines()) == 2

    report = run_report(record)
    # The explain block must make "cost down AND latency down" non-fishy: it shows the
    # mix's per-model latency and the blended token breakdown by model.
    assert "fast-cheap" in report
    assert "tokens by model" in report.lower()
    assert "latency" in report.lower()
    assert "vs best_single" in report.lower()


def test_cascade_escalates_and_sums_cost() -> None:
    matrix = _matrix()
    ids = matrix.scenario_ids()

    # Try fast-cheap first; escalate to slow-pricey only where it failed (s3 in _matrix).
    def cascade(sid: str, transcript: list) -> str | Finish:  # noqa: ANN001
        if not transcript:
            return "fast-cheap"
        if transcript[-1].model == "fast-cheap" and sid == "s3":
            return "slow-pricey"
        return Finish()

    result = evaluate_call_sequences(matrix, ids, cascade)
    assert result.accuracy == pytest.approx(1.0)  # the escalation rescued s3
    # s0-s2 cost one cheap call; s3 cost a cheap call PLUS a pricey one.
    assert result.cost_per_call == pytest.approx((0.001 * 4 + 0.02) / 4)
    assert result.calls_per_scenario == pytest.approx(5 / 4)
    assert result.model_mix == {"fast-cheap": 4 / 5, "slow-pricey": 1 / 5}


def test_best_of_two_consumes_episodes_in_order() -> None:
    pool = [
        PoolEntry(
            name="m",
            kind=ProviderKind.OPENAI,
            model="m",
            input_per_mtok=1.0,
            output_per_mtok=1.0,
        )
    ]
    outcomes = [
        ScenarioOutcome(
            scenario_id="s0",
            task="t",
            model="m",
            episode=episode,
            reward=reward,
            cost_usd=0.01,
            usage=TokenUsage(input_tokens=10, output_tokens=10),
        )
        for episode, reward in ((0, 0.0), (1, 1.0))
    ]
    matrix = OutcomeMatrix(pool=pool, outcomes=outcomes)

    # Oracle best-of-2: call m twice, keep the better episode (upper-bound simulation).
    def best_of_two(sid: str, transcript: list) -> str | Finish:  # noqa: ANN001
        if len(transcript) < 2:
            return "m"
        rewards = [o.reward for o in transcript]
        return Finish(pick=rewards.index(max(rewards)))

    result = evaluate_call_sequences(matrix, ["s0"], best_of_two)
    assert result.accuracy == pytest.approx(1.0)  # picked episode 1, not episode 0
    assert result.cost_per_call == pytest.approx(0.02)  # but paid for BOTH episodes
    assert result.calls_per_scenario == pytest.approx(2.0)

    # A third call to the same model has no stored episode left: scenario goes unscored.
    def best_of_three(sid: str, transcript: list) -> str | Finish:  # noqa: ANN001
        return "m" if len(transcript) < 3 else Finish()

    with pytest.raises(ValueError, match="no scored outcomes"):
        evaluate_call_sequences(matrix, ["s0"], best_of_three)


def test_runaway_policy_hits_max_calls() -> None:
    matrix = _matrix()
    with pytest.raises(ValueError, match="max_calls"):
        evaluate_call_sequences(matrix, ["s0"], lambda sid, transcript: "fast-cheap", max_calls=1)
