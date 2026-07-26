"""Tests for `_chunk_scored_datums`, the cross-tokenizer scoring orchestration.

Scope: which datums reach the teacher and why the rest are dropped. The
alignment itself (byte offsets, island location, partition intersection,
advantage math) is covered by `wmh/distill/xtoken/*_test.py`, so the render and
plan calls are stubbed here. What must be right at this layer is that every drop
is counted under its own reason -- a run that trains on 3 of 64 trajectories and
reports only "coverage was low" cannot be debugged.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pytest

import wmh.distill.loop as loop_module
from wmh.distill.data import TrainDatum
from wmh.distill.loop import _chunk_scored_datums
from wmh.distill.rendering import ChatRendering
from wmh.distill.tokens import TrialRecord
from wmh.distill.xtoken.plan import SurfaceTokenizer
from wmh.distill.xtoken.teacher_render import TemplateTokenizer
from wmh.providers.tinker import TokenSpan


def _datum(trial: str, *, fragment: int = 0, loss_tokens: int = 4) -> TrainDatum:
    """A datum with `loss_tokens` trainable positions after a 2-token prompt."""
    total = 2 + loss_tokens
    return TrainDatum(
        trial_name=trial,
        fragment_index=fragment,
        model_input_tokens=list(range(10, 10 + total)),
        loss_mask=[0.0, 0.0] + [1.0] * loss_tokens,
        sampled_logprobs=[0.0, 0.0] + [-0.5] * loss_tokens,
    )


def _record(trial: str, *, canonical: bool = True) -> TrialRecord:
    return TrialRecord(
        task_id=trial,
        attempt=1,
        trial_name=trial,
        reward=0.0,
        passed=False,
        spans=[
            TokenSpan(
                call_index=0,
                prompt_token_ids=[10, 11],
                sampled_token_ids=[12, 13, 14, 15],
                sampled_logprobs=[-0.5] * 4,
                delta_start=0 if canonical else None,
                delta_messages=[] if canonical else None,
            )
        ],
        artifact_dir=f"/trials/{trial}",
    )


# The render, plan, and replay steps are monkeypatched wholesale by `stubs`, so
# nothing ever touches these three; they exist to keep the call typed.
_RENDERING = cast("ChatRendering", object())
_STUDENT_TOKENIZER = cast("SurfaceTokenizer", object())
_TEACHER_TOKENIZER = cast("TemplateTokenizer", object())


class _Plan:
    """Stands in for a ChunkPlan; only these two attributes are read here."""

    def __init__(self, *, chunks: int, scored: int) -> None:
        self.chunks = list(range(chunks))
        self.scored_student_tokens = scored


class _Render:
    def __init__(self, *, islands: int, tokens: int = 6) -> None:
        self.islands = list(range(islands))
        self.token_ids = list(range(100, 100 + tokens))


class _Teacher:
    """Records what it was asked to score; optionally fails on demand."""

    def __init__(self, *, fail_on: set[int] | None = None) -> None:
        self.calls: list[list[int]] = []
        self._fail_on = fail_on or set()

    def score(self, token_ids: Sequence[int]) -> list[float | None]:
        index = len(self.calls)
        self.calls.append(list(token_ids))
        if index in self._fail_on:
            raise RuntimeError("teacher endpoint returned garbage")
        return [None] + [-0.25] * (len(token_ids) - 1)


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy-path render/plan: one island, one chunk covering every loss token."""
    # Mirrors the real contract: None if ANY span lost its canonical messages,
    # since one re-render fallback disqualifies the whole trial.
    monkeypatch.setattr(
        loop_module,
        "reconstruct_conversation",
        lambda spans, _r: (
            _Replay() if spans and all(span.delta_messages is not None for span in spans) else None
        ),
    )
    monkeypatch.setattr(loop_module, "render_for_teacher", lambda *_a, **_k: _Render(islands=1))
    monkeypatch.setattr(
        loop_module, "build_chunk_plan", lambda *_a, **_k: _Plan(chunks=2, scored=4)
    )


class _Replay:
    messages: list[object] = []
    tools: list[object] | None = None


def test_scores_every_datum_when_each_trial_is_a_single_fragment(stubs: None) -> None:
    datums = [_datum("a"), _datum("b")]
    records = [_record("a"), _record("b")]
    teacher = _Teacher()

    kept, plans, rows, stats = _chunk_scored_datums(
        datums, records, _RENDERING, _STUDENT_TOKENIZER, _TEACHER_TOKENIZER, teacher
    )

    assert [d.trial_name for d in kept] == ["a", "b"]
    assert len(plans) == len(rows) == 2
    assert stats.scored == 2
    # Coverage is over TRAINABLE tokens: 4 scored of 4 loss tokens, per datum.
    assert stats.scored_student_tokens == 8
    assert stats.loss_tokens == 8
    assert stats.coverage == 1.0
    assert len(teacher.calls) == 2


def test_a_trial_split_into_fragments_is_skipped_not_guessed(stubs: None) -> None:
    """Two datums from one trial cannot be paired against its single replay.

    `reconstruct_conversation` returns one conversation per TRIAL, so fragment 1
    would be aligned against fragment 0's assistant turn. That mis-pairing is
    silent -- the advantages would be wrong, not absent -- so the datums are
    dropped and counted instead.
    """
    datums = [_datum("a", fragment=0), _datum("a", fragment=1), _datum("b")]
    records = [_record("a"), _record("b")]
    teacher = _Teacher()

    kept, _plans, _rows, stats = _chunk_scored_datums(
        datums, records, _RENDERING, _STUDENT_TOKENIZER, _TEACHER_TOKENIZER, teacher
    )

    assert [d.trial_name for d in kept] == ["b"]
    assert stats.fragmented == 2
    assert stats.scored == 1
    # The dropped datums still count toward the denominator: their tokens were
    # collected and paid for, and coverage must not flatter itself by ignoring
    # trajectories it failed to use.
    assert stats.loss_tokens == 12
    assert stats.coverage == pytest.approx(4 / 12)
    assert len(teacher.calls) == 1


def test_a_lost_canonical_history_is_counted_as_no_replay(stubs: None) -> None:
    datums = [_datum("a"), _datum("b")]
    records = [_record("a", canonical=False), _record("b")]
    teacher = _Teacher()

    kept, _plans, _rows, stats = _chunk_scored_datums(
        datums, records, _RENDERING, _STUDENT_TOKENIZER, _TEACHER_TOKENIZER, teacher
    )

    assert [d.trial_name for d in kept] == ["b"]
    assert stats.no_replay == 1


def test_a_datum_with_no_matching_record_is_counted_not_crashed(stubs: None) -> None:
    """A datum whose trial record is missing must not raise a KeyError mid-step."""
    kept, _plans, _rows, stats = _chunk_scored_datums(
        [_datum("orphan")], [], _RENDERING, _STUDENT_TOKENIZER, _TEACHER_TOKENIZER, _Teacher()
    )

    assert kept == []
    assert stats.no_replay == 1


def test_empty_islands_and_empty_chunks_are_counted_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two failures have different causes and must not be conflated.

    No islands means the teacher's render did not contain the student's text at
    all (a template or escaping problem). No chunks means the text was found but
    the two tokenizations shared no usable boundary inside it (an alignment
    problem). Collapsing them would point at the wrong fix.
    """
    monkeypatch.setattr(loop_module, "reconstruct_conversation", lambda *_a: _Replay())
    monkeypatch.setattr(
        loop_module,
        "render_for_teacher",
        lambda _tok, messages, _tools=None: _Render(islands=0 if messages == "none" else 1),
    )
    monkeypatch.setattr(
        loop_module, "build_chunk_plan", lambda *_a, **_k: _Plan(chunks=0, scored=0)
    )

    _kept, _plans, _rows, stats = _chunk_scored_datums(
        [_datum("a")],
        [_record("a")],
        _RENDERING,
        _STUDENT_TOKENIZER,
        _TEACHER_TOKENIZER,
        _Teacher(),
    )
    assert (stats.no_islands, stats.no_chunks, stats.scored) == (0, 1, 0)

    monkeypatch.setattr(loop_module, "render_for_teacher", lambda *_a, **_k: _Render(islands=0))
    _kept, _plans, _rows, stats = _chunk_scored_datums(
        [_datum("a")],
        [_record("a")],
        _RENDERING,
        _STUDENT_TOKENIZER,
        _TEACHER_TOKENIZER,
        _Teacher(),
    )
    assert (stats.no_islands, stats.no_chunks, stats.scored) == (1, 0, 0)


def test_one_failed_scoring_call_does_not_lose_the_other_trajectories(stubs: None) -> None:
    """A single bad endpoint response must cost one trajectory, not the step."""
    datums = [_datum("a"), _datum("b"), _datum("c")]
    records = [_record("a"), _record("b"), _record("c")]
    teacher = _Teacher(fail_on={1})

    kept, plans, rows, stats = _chunk_scored_datums(
        datums, records, _RENDERING, _STUDENT_TOKENIZER, _TEACHER_TOKENIZER, teacher
    )

    assert [d.trial_name for d in kept] == ["a", "c"]
    assert stats.scoring_failed == 1
    assert stats.scored == 2
    # Kept datums, plans and rows must stay index-aligned after a mid-batch drop:
    # a row paired with the wrong datum is exactly the silent corruption this
    # whole path exists to avoid.
    assert len(kept) == len(plans) == len(rows) == 2


def test_coverage_is_zero_rather_than_undefined_for_an_empty_batch() -> None:
    _kept, _plans, _rows, stats = _chunk_scored_datums(
        [], [], _RENDERING, _STUDENT_TOKENIZER, _TEACHER_TOKENIZER, _Teacher()
    )

    assert stats.scored == 0
    assert stats.coverage == 0.0
