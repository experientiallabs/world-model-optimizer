"""Tests for the closed-loop outcome matrix types (the routing optimizer's training data)."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome, split_router_scenarios
from wmo.providers.base import ProviderKind, TokenUsage
from wmo.providers.pool import PoolEntry


def _outcome(
    scenario_id: str, model: str, *, reward: float = 0.5, episode: int = 0
) -> ScenarioOutcome:
    return ScenarioOutcome(
        scenario_id=scenario_id,
        task="do the thing",
        model=model,
        benchmark="terminal-bench-2",
        episode=episode,
        attempt_number=episode + 1,
        reward=reward,
        success=reward >= 0.5,
        critique="ok",
        steps=3,
        tool_calls=3,
        stop_reason="agent_done",
        usage=TokenUsage(input_tokens=100, output_tokens=50),
        cost_usd=0.01,
        call_seconds=[0.2, 0.3, 0.25],
        wall_seconds=12.5,
        completion_status="scored",
        artifact_dir="/artifacts/cell",
        replies=["{}", "{}", '{"done": true}'],
    )


def _matrix() -> OutcomeMatrix:
    entries = [
        PoolEntry(name="fable-5", kind=ProviderKind.ANTHROPIC, model="claude-fable-5"),
        PoolEntry(name="haiku-4-5", kind=ProviderKind.ANTHROPIC, model="claude-haiku-4-5"),
    ]
    outcomes = [
        _outcome("s1", "fable-5", reward=0.9),
        _outcome("s2", "fable-5", reward=0.7),
        _outcome("s1", "haiku-4-5", reward=0.4),
        _outcome("s2", "haiku-4-5", reward=0.8),
    ]
    return OutcomeMatrix(pool=entries, outcomes=outcomes)


def test_matrix_accessors() -> None:
    matrix = _matrix()
    assert matrix.model_names() == ["fable-5", "haiku-4-5"]
    assert matrix.scenario_ids() == ["s1", "s2"]
    assert matrix.mean_reward("fable-5") == pytest.approx(0.8)
    assert matrix.mean_reward("haiku-4-5") == pytest.approx(0.6)
    assert [o.model for o in matrix.for_scenario("s1")] == ["fable-5", "haiku-4-5"]


def test_router_split_is_deterministic_disjoint_and_order_preserving() -> None:
    ids = [f"scenario-{index}" for index in range(10)]
    split = split_router_scenarios(ids)

    assert len(split.fit_ids) == 7
    assert len(split.report_ids) == 3
    assert set(split.fit_ids).isdisjoint(split.report_ids)
    assert [sid for sid in ids if sid in split.fit_ids] == list(split.fit_ids)
    reordered = list(reversed(ids))
    rerun = split_router_scenarios(reordered)
    assert set(rerun.fit_ids) == set(split.fit_ids)
    assert set(rerun.report_ids) == set(split.report_ids)


def test_router_split_refuses_a_claim_with_no_holdout() -> None:
    with pytest.raises(ValueError, match="at least 2 scenarios"):
        split_router_scenarios(["only-one"])


def test_matrix_round_trips_through_json(tmp_path: Path) -> None:
    matrix = _matrix()
    path = tmp_path / "outcomes.json"
    matrix.save(path)
    loaded = OutcomeMatrix.load(path)
    assert loaded == matrix


def test_estimated_usage_provenance_round_trips(tmp_path: Path) -> None:
    matrix = _matrix()
    matrix.outcomes[0].usage_accounting = "estimated"
    matrix.outcomes[0].usage_estimate_method = "trace-char-prefix-4k-overhead-v1"
    path = tmp_path / "outcomes.json"

    matrix.save(path)
    loaded = OutcomeMatrix.load(path)

    assert loaded.outcomes[0].usage_accounting == "estimated"
    assert loaded.outcomes[0].usage_estimate_method == "trace-char-prefix-4k-overhead-v1"


def test_mean_reward_unknown_model_errors() -> None:
    with pytest.raises(KeyError, match="fable-5"):
        _matrix().mean_reward("nope")


def test_outcomes_must_name_pool_models() -> None:
    # A ghost model used to reach the fitter and die on a bare KeyError; the matrix names it.
    matrix = _matrix()
    with pytest.raises(ValueError, match="ghost-model"):
        OutcomeMatrix(
            pool=matrix.pool,
            outcomes=[*matrix.outcomes, _outcome("s1", "ghost-model")],
        )


def test_measured_compression_reads_the_arm_off_the_scored_rows() -> None:
    matrix = OutcomeMatrix(
        pool=[
            PoolEntry(
                name="a",
                kind=ProviderKind.OPENAI,
                model="a",
                input_per_mtok=1.0,
                output_per_mtok=1.0,
            )
        ],
        outcomes=[
            ScenarioOutcome(
                scenario_id="s1",
                task="t",
                model="a",
                reward=1.0,
                compressor_id="truncate",
                compressor_version="1",
                aggressiveness=0.5,
            )
        ],
    )
    config = matrix.measured_compression()
    assert config is not None
    assert config.compressor_id == "truncate"
    assert config.aggressiveness == 0.5


def test_a_matrix_with_no_compression_fields_reads_as_the_uncompressed_arm() -> None:
    # Every matrix captured before D-COMPRESS existed, which must keep fitting exactly as before.
    matrix = OutcomeMatrix(
        pool=[
            PoolEntry(
                name="a",
                kind=ProviderKind.OPENAI,
                model="a",
                input_per_mtok=1.0,
                output_per_mtok=1.0,
            )
        ],
        outcomes=[ScenarioOutcome(scenario_id="s1", task="t", model="a", reward=1.0)],
    )
    assert matrix.measured_compression() is None


def test_a_matrix_that_mixes_arms_refuses_to_name_one() -> None:
    # Two arms in one file: the rows are not comparable, so no single policy can be fitted
    # from them and picking a winner here would hide that.
    matrix = OutcomeMatrix(
        pool=[
            PoolEntry(
                name="a",
                kind=ProviderKind.OPENAI,
                model="a",
                input_per_mtok=1.0,
                output_per_mtok=1.0,
            )
        ],
        outcomes=[
            ScenarioOutcome(scenario_id="s1", task="t", model="a", reward=1.0),
            ScenarioOutcome(
                scenario_id="s2",
                task="t",
                model="a",
                reward=1.0,
                compressor_id="truncate",
                compressor_version="1",
                aggressiveness=0.5,
            ),
        ],
    )
    with pytest.raises(ValueError, match="mixes compression configs"):
        matrix.measured_compression()


def test_an_unscored_row_does_not_decide_the_arm() -> None:
    # An unscored episode produced no reward, so whatever it ran under cannot bias a fit.
    matrix = OutcomeMatrix(
        pool=[
            PoolEntry(
                name="a",
                kind=ProviderKind.OPENAI,
                model="a",
                input_per_mtok=1.0,
                output_per_mtok=1.0,
            )
        ],
        outcomes=[
            ScenarioOutcome(scenario_id="s1", task="t", model="a", reward=1.0),
            ScenarioOutcome(
                scenario_id="s2",
                task="t",
                model="a",
                reward=None,
                error="provider throttled",
                compressor_id="truncate",
                compressor_version="1",
                aggressiveness=0.5,
            ),
        ],
    )
    assert matrix.measured_compression() is None
