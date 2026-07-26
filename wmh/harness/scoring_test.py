"""Tests for the evaluator-neutral scoring contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from wmh.harness.doc import HarnessDoc
from wmh.harness.scoring import (
    RewardMode,
    ScoreCell,
    ScoreReport,
    ScoreRequest,
    reward_passed,
)


def _cell(task_id: str, attempt: int, reward: float, mode: RewardMode = "raw") -> ScoreCell:
    return ScoreCell(
        task_id=task_id,
        attempt=attempt,
        reward=reward,
        passed=reward_passed(reward, mode),
        artifact_dir=f"/jobs/{task_id}__x{attempt}",
    )


def _report(cells: tuple[ScoreCell, ...], *, tasks: tuple[str, ...], attempts: int) -> ScoreReport:
    return ScoreReport(
        doc_hash=HarnessDoc.baseline().doc_hash,
        request=ScoreRequest(task_ids=tasks, attempts=attempts),
        reward_mode="raw",
        cells=cells,
    )


def test_reward_modes_follow_the_frozen_selection_protocol() -> None:
    # raw: passed iff exactly 1.0; positive-binary: passed iff strictly positive.
    assert reward_passed(1.0, "raw")
    assert not reward_passed(0.99, "raw")
    assert not reward_passed(0.0, "raw")
    assert reward_passed(0.01, "positive-binary")
    assert reward_passed(1.0, "positive-binary")
    assert not reward_passed(0.0, "positive-binary")


def test_score_weights_tasks_equally_and_keeps_raw_rewards() -> None:
    report = _report(
        (
            _cell("a", 1, 1.0),
            _cell("a", 2, 1.0),
            _cell("b", 1, 0.25),
            _cell("b", 2, 1.0),
        ),
        tasks=("a", "b"),
        attempts=2,
    )
    assert report.score == pytest.approx(0.75)  # mean of per-task means: (1.0 + 0.5) / 2
    assert report.pass_rate == pytest.approx(0.75)
    assert [cell.reward for cell in report.by_task()["b"]] == [0.25, 1.0]
    assert report.by_task()["b"][0].artifact_dir == "/jobs/b__x1"


def test_report_rejects_missing_duplicate_and_extra_cells() -> None:
    with pytest.raises(ValidationError, match="missing"):
        _report((_cell("a", 1, 1.0),), tasks=("a", "b"), attempts=1)
    with pytest.raises(ValidationError, match="duplicate"):
        _report((_cell("a", 1, 1.0), _cell("a", 1, 0.0)), tasks=("a",), attempts=1)
    with pytest.raises(ValidationError, match="extra"):
        _report((_cell("a", 1, 1.0), _cell("a", 2, 1.0)), tasks=("a",), attempts=1)


def test_cells_canonicalize_and_reject_invalid_rewards() -> None:
    report = _report(
        (_cell("b", 1, 0.0), _cell("a", 1, 1.0)),
        tasks=("a", "b"),
        attempts=1,
    )
    assert [cell.task_id for cell in report.cells] == ["a", "b"]
    with pytest.raises(ValidationError):
        _cell("a", 1, 1.5)
    with pytest.raises(ValidationError):
        _cell("a", 1, float("nan"))
    with pytest.raises(ValidationError, match="not boolean"):
        ScoreCell(task_id="a", attempt=1, reward=True, passed=True)  # type: ignore[arg-type]


def test_request_rejects_empty_duplicate_and_boolean_inputs() -> None:
    with pytest.raises(ValidationError, match="nonempty"):
        ScoreRequest(task_ids=(), attempts=1)
    with pytest.raises(ValidationError, match="unique"):
        ScoreRequest(task_ids=("a", "a"), attempts=1)
    with pytest.raises(ValidationError, match="not boolean"):
        ScoreRequest(task_ids=("a",), attempts=True)  # type: ignore[arg-type]


def test_telemetry_fields_default_and_keep_the_score_contract_unchanged() -> None:
    # A cell built the old way carries no telemetry; the score/pass_rate contract is untouched.
    report = _report(
        (_cell("a", 1, 1.0), _cell("b", 1, 0.0)),
        tasks=("a", "b"),
        attempts=1,
    )
    for cell in report.cells:
        assert cell.turns is None
        assert cell.calls is None
        assert cell.input_tokens is None
        assert cell.output_tokens is None
        assert cell.stop_reason == ""
        assert cell.hit_turn_cap is False
        assert cell.hit_timeout is False
    assert report.score == pytest.approx(0.5)
    assert report.pass_rate == pytest.approx(0.5)
    # No cell has telemetry, so the aggregate is the explicit empty marker (not a bare {}).
    assert report.telemetry_summary() == {"n_with_telemetry": 0}


def _telemetry_cell(
    task_id: str,
    attempt: int,
    *,
    turns: int,
    input_tokens: int,
    output_tokens: int,
    hit_turn_cap: bool = False,
    hit_timeout: bool = False,
) -> ScoreCell:
    return ScoreCell(
        task_id=task_id,
        attempt=attempt,
        reward=1.0,
        passed=True,
        turns=turns,
        calls=turns,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        stop_reason="submitted",
        hit_turn_cap=hit_turn_cap,
        hit_timeout=hit_timeout,
    )


def test_telemetry_summary_aggregates_only_cells_that_carry_telemetry() -> None:
    report = _report(
        (
            _telemetry_cell("a", 1, turns=5, input_tokens=100, output_tokens=10, hit_turn_cap=True),
            _telemetry_cell("a", 2, turns=8, input_tokens=200, output_tokens=20, hit_timeout=True),
            _telemetry_cell("b", 1, turns=3, input_tokens=300, output_tokens=90),
            # No-telemetry cell: skipped by the aggregate (turns is None).
            _cell("b", 2, 1.0),
        ),
        tasks=("a", "b"),
        attempts=2,
    )
    summary = report.telemetry_summary()
    assert summary["n_with_telemetry"] == 3  # the fourth cell has no telemetry
    assert summary["turn_cap_hit_rate"] == pytest.approx(1 / 3)
    assert summary["timeout_rate"] == pytest.approx(1 / 3)
    assert summary["input_tokens_median"] == 200  # median of [100, 200, 300]
    assert summary["output_tokens_median"] == 20  # median of [10, 20, 90]
    # p90 by nearest rank over [10, 20, 90]: ceil(0.9*3)=3 -> the largest value.
    assert summary["output_tokens_p90"] == 90


def test_telemetry_summary_tolerates_turns_present_but_token_fields_absent() -> None:
    # A timed-out episode can record turns/stop_reason with no worker_usage: medians stay None,
    # but the rates and count still reflect the cell (turns is not None).
    report = _report(
        (
            ScoreCell(
                task_id="a",
                attempt=1,
                reward=0.0,
                passed=False,
                turns=4,
                stop_reason="cancelled-by-harbor-timeout",
                hit_timeout=True,
            ),
        ),
        tasks=("a",),
        attempts=1,
    )
    summary = report.telemetry_summary()
    assert summary["n_with_telemetry"] == 1
    assert summary["timeout_rate"] == pytest.approx(1.0)
    assert summary["turn_cap_hit_rate"] == pytest.approx(0.0)
    assert summary["input_tokens_median"] is None
    assert summary["output_tokens_median"] is None
    assert summary["output_tokens_p90"] is None
