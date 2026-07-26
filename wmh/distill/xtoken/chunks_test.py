"""Tests for chunk plans and cross-tokenizer chunk advantages."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from wmh.distill.config import DistillConfig
from wmh.distill.data import TrainDatum
from wmh.distill.xtoken.chunks import (
    ChunkPlan,
    ChunkSpan,
    _chunk_totals,
    attach_chunk_advantages,
)


def _cfg(*, center: bool = True, clip: float = 4.0) -> DistillConfig:
    """A minimal valid run config with the two knobs this module reads."""
    return DistillConfig.model_validate(
        {
            "student": {"base_model": "Qwen/Qwen3.6-27B"},
            "teacher": {"model": "zai-org/GLM-5.2"},
            "harbor": {"job_template": "job.yaml"},
            "train": {"center_advantages": center, "advantage_clip": clip},
        }
    )


def _datum(
    loss_positions: set[int],
    total_tokens: int,
    *,
    sampled_logprob: float = 0.0,
    trial_name: str = "t",
    fragment_index: int = 0,
) -> TrainDatum:
    """A datum whose named positions are loss positions with a fixed logprob."""
    return TrainDatum(
        trial_name=trial_name,
        fragment_index=fragment_index,
        model_input_tokens=list(range(total_tokens)),
        loss_mask=[1.0 if i in loss_positions else 0.0 for i in range(total_tokens)],
        sampled_logprobs=[
            sampled_logprob if i in loss_positions else 0.0 for i in range(total_tokens)
        ],
    )


def test_chunk_span_rejects_student_position_zero() -> None:
    # Position 0's advantage is dropped by the next-token shift, so a chunk
    # there would silently lose influence.
    with pytest.raises(ValidationError):
        ChunkSpan(student_start=0, student_end=2, teacher_start=1, teacher_end=2)


def test_chunk_span_rejects_teacher_position_zero() -> None:
    with pytest.raises(ValidationError):
        ChunkSpan(student_start=1, student_end=2, teacher_start=0, teacher_end=1)


def test_chunk_span_rejects_empty_range() -> None:
    with pytest.raises(ValidationError):
        ChunkSpan(student_start=3, student_end=3, teacher_start=1, teacher_end=2)


def test_chunk_plan_rejects_overlapping_student_ranges() -> None:
    with pytest.raises(ValidationError, match="sorted and non-overlapping"):
        ChunkPlan(
            trial_name="t",
            fragment_index=0,
            teacher_token_count=10,
            chunks=[
                ChunkSpan(student_start=1, student_end=4, teacher_start=1, teacher_end=2),
                ChunkSpan(student_start=3, student_end=6, teacher_start=2, teacher_end=3),
            ],
        )


def test_chunk_plan_rejects_overlapping_teacher_ranges() -> None:
    with pytest.raises(ValidationError, match="sorted and non-overlapping"):
        ChunkPlan(
            trial_name="t",
            fragment_index=0,
            teacher_token_count=10,
            chunks=[
                ChunkSpan(student_start=1, student_end=2, teacher_start=1, teacher_end=5),
                ChunkSpan(student_start=2, student_end=3, teacher_start=4, teacher_end=6),
            ],
        )


def test_chunk_plan_rejects_range_past_teacher_sequence() -> None:
    with pytest.raises(ValidationError, match="past the teacher sequence"):
        ChunkPlan(
            trial_name="t",
            fragment_index=0,
            teacher_token_count=3,
            chunks=[ChunkSpan(student_start=1, student_end=2, teacher_start=1, teacher_end=5)],
        )


def test_validate_against_rejects_chunk_over_context_position() -> None:
    # THE filler bug: sampled_logprobs is 0.0 padding at mask-0 positions, so a
    # chunk straddling the transition would fold zeros into the student sum.
    datum = _datum({1, 2}, 5)
    plan = ChunkPlan(
        trial_name="t",
        fragment_index=0,
        teacher_token_count=5,
        chunks=[ChunkSpan(student_start=1, student_end=4, teacher_start=1, teacher_end=3)],
    )
    with pytest.raises(ValueError, match="context position"):
        plan.validate_against(datum)


def test_validate_against_rejects_mismatched_datum_identity() -> None:
    datum = _datum({1}, 3, trial_name="other")
    plan = ChunkPlan(
        trial_name="t",
        fragment_index=0,
        teacher_token_count=3,
        chunks=[ChunkSpan(student_start=1, student_end=2, teacher_start=1, teacher_end=2)],
    )
    with pytest.raises(ValueError, match="paired with their datum"):
        plan.validate_against(datum)


def test_single_token_chunk_reproduces_per_token_advantage() -> None:
    # With 1-to-1 chunks and centering off, the chunk advantage must equal the
    # same-tokenizer formula teacher_lp - sampled_lp.
    datum = _datum({1, 2}, 4, sampled_logprob=-1.0)
    plan = ChunkPlan(
        trial_name="t",
        fragment_index=0,
        teacher_token_count=4,
        chunks=[
            ChunkSpan(student_start=1, student_end=2, teacher_start=1, teacher_end=2),
            ChunkSpan(student_start=2, student_end=3, teacher_start=2, teacher_end=3),
        ],
    )
    row = [None, -0.5, -3.0, None]
    attached, stats = attach_chunk_advantages([datum], [plan], [row], _cfg(center=False))
    assert len(attached) == 1
    assert attached[0].advantages[1] == pytest.approx(-0.5 - -1.0)
    assert attached[0].advantages[2] == pytest.approx(-3.0 - -1.0)
    assert stats.scored_loss_tokens == 2
    assert stats.coverage_rate == 1.0


def test_chunk_influence_equals_kl_gap_regardless_of_length() -> None:
    # A chunk covering 4 student tokens against 1 teacher token must contribute
    # exactly (teacher_sum - student_sum) in total, not 4x it.
    datum = _datum({1, 2, 3, 4}, 6, sampled_logprob=-0.25)
    plan = ChunkPlan(
        trial_name="t",
        fragment_index=0,
        teacher_token_count=6,
        chunks=[ChunkSpan(student_start=1, student_end=5, teacher_start=1, teacher_end=2)],
    )
    row = [None, -2.0, None, None, None, None]
    attached, _ = attach_chunk_advantages([datum], [plan], [row], _cfg(center=False))
    total = sum(attached[0].advantages)
    # teacher_sum -2.0, student_sum 4 * -0.25 = -1.0, gap = -1.0
    assert total == pytest.approx(-1.0)
    # And it is spread evenly across the chunk's student tokens.
    assert attached[0].advantages[1:5] == pytest.approx([-0.25] * 4)


def test_centering_does_not_invert_a_long_chunk() -> None:
    """Regression: token-level centering flips the sign of long chunks.

    Chunk A covers 10 student tokens with a raw gap of +1.0; chunk B covers
    1000 with a raw gap of +1.5. Centering over CHUNK TOTALS keeps B (the
    larger gap) above A. Centering per TOKEN would give A +0.975 and B -0.975,
    reversing the ordering and training the larger gap backwards.
    """
    short_positions = set(range(1, 11))
    long_positions = set(range(11, 1011))
    datum = _datum(short_positions | long_positions, 1012)
    plan = ChunkPlan(
        trial_name="t",
        fragment_index=0,
        teacher_token_count=3,
        chunks=[
            ChunkSpan(student_start=1, student_end=11, teacher_start=1, teacher_end=2),
            ChunkSpan(student_start=11, student_end=1011, teacher_start=2, teacher_end=3),
        ],
    )
    # sampled logprobs are 0.0, so teacher logprob IS the gap for each chunk.
    row = [None, 1.0, 1.5]
    attached, _ = attach_chunk_advantages([datum], [plan], [row], _cfg(center=True))
    advantages = attached[0].advantages
    short_total = sum(advantages[1:11])
    long_total = sum(advantages[11:1011])
    # Mean chunk total is 1.25, so A -> -0.25 and B -> +0.25.
    assert short_total == pytest.approx(-0.25)
    assert long_total == pytest.approx(0.25)
    # The invariant that matters: the larger raw gap stays the larger total.
    assert long_total > short_total
    # And centering is zero-sum across chunks.
    assert short_total + long_total == pytest.approx(0.0)


def test_unscored_loss_tokens_stay_exactly_zero_after_centering() -> None:
    # Structural tokens the teacher render has no counterpart for must keep
    # advantage 0.0. Advantage 0.0 IS the mask on the wire, so a centered
    # nudge off zero would train on noise.
    datum = _datum({1, 2, 3}, 5)
    plan = ChunkPlan(
        trial_name="t",
        fragment_index=0,
        teacher_token_count=3,
        chunks=[ChunkSpan(student_start=1, student_end=2, teacher_start=1, teacher_end=2)],
    )
    row = [None, 2.0, None]
    attached, stats = attach_chunk_advantages([datum], [plan], [row], _cfg(center=True))
    advantages = attached[0].advantages
    assert advantages[2] == 0.0
    assert advantages[3] == 0.0
    assert stats.unscored_loss_tokens == 2
    assert stats.scored_loss_tokens == 1
    assert stats.coverage_rate == pytest.approx(1 / 3)


def test_clip_bounds_the_per_token_value_and_is_counted() -> None:
    datum = _datum({1}, 3)
    plan = ChunkPlan(
        trial_name="t",
        fragment_index=0,
        teacher_token_count=2,
        chunks=[ChunkSpan(student_start=1, student_end=2, teacher_start=1, teacher_end=2)],
    )
    attached, stats = attach_chunk_advantages(
        [datum], [plan], [[None, 99.0]], _cfg(center=False, clip=4.0)
    )
    assert attached[0].advantages[1] == pytest.approx(4.0)
    assert stats.clipped_chunks == 1


def test_chunk_reverse_kl_is_student_minus_teacher() -> None:
    datum = _datum({1}, 3, sampled_logprob=-2.0)
    plan = ChunkPlan(
        trial_name="t",
        fragment_index=0,
        teacher_token_count=2,
        chunks=[ChunkSpan(student_start=1, student_end=2, teacher_start=1, teacher_end=2)],
    )
    _, stats = attach_chunk_advantages([datum], [plan], [[None, -0.5]], _cfg(center=True))
    # mean(student_lp - teacher_lp) = -2.0 - -0.5 = -1.5, and centering must
    # not move the measurement.
    assert stats.chunk_reverse_kl == pytest.approx(-1.5)


def test_wrong_length_teacher_row_drops_the_datum() -> None:
    datum = _datum({1}, 3)
    plan = ChunkPlan(
        trial_name="t",
        fragment_index=0,
        teacher_token_count=5,
        chunks=[ChunkSpan(student_start=1, student_end=2, teacher_start=1, teacher_end=2)],
    )
    attached, stats = attach_chunk_advantages([datum], [plan], [[None, 1.0]], _cfg())
    assert attached == []
    assert stats.mismatch_drops == 1


def test_none_at_a_needed_teacher_position_drops_the_datum() -> None:
    datum = _datum({1}, 3)
    plan = ChunkPlan(
        trial_name="t",
        fragment_index=0,
        teacher_token_count=3,
        chunks=[ChunkSpan(student_start=1, student_end=2, teacher_start=1, teacher_end=3)],
    )
    attached, stats = attach_chunk_advantages([datum], [plan], [[None, 1.0, None]], _cfg())
    assert attached == []
    assert stats.mismatch_drops == 1


def test_datum_with_no_covered_loss_token_is_dropped() -> None:
    datum = _datum({1, 2}, 4)
    plan = ChunkPlan(trial_name="t", fragment_index=0, teacher_token_count=4, chunks=[])
    attached, stats = attach_chunk_advantages([datum], [plan], [[None, 1.0, 1.0, 1.0]], _cfg())
    assert attached == []
    assert stats.empty_coverage_drops == 1
    assert stats.datums == 0


def test_length_mismatch_between_datums_and_plans_is_a_caller_bug() -> None:
    datum = _datum({1}, 3)
    with pytest.raises(ValueError, match="exactly one plan per datum"):
        attach_chunk_advantages([datum], [], [[None, 1.0, 1.0]], _cfg())


def test_length_mismatch_between_datums_and_rows_is_a_caller_bug() -> None:
    datum = _datum({1}, 3)
    plan = ChunkPlan(
        trial_name="t",
        fragment_index=0,
        teacher_token_count=3,
        chunks=[ChunkSpan(student_start=1, student_end=2, teacher_start=1, teacher_end=2)],
    )
    with pytest.raises(ValueError, match="exactly one row per datum"):
        attach_chunk_advantages([datum], [plan], [], _cfg())


def test_centering_is_across_the_whole_batch() -> None:
    # Two datums, one chunk each, gaps +1.0 and +3.0; centering uses both.
    first = _datum({1}, 3, trial_name="a")
    second = _datum({1}, 3, trial_name="b")
    plans = [
        ChunkPlan(
            trial_name=name,
            fragment_index=0,
            teacher_token_count=2,
            chunks=[ChunkSpan(student_start=1, student_end=2, teacher_start=1, teacher_end=2)],
        )
        for name in ("a", "b")
    ]
    rows = [[None, 1.0], [None, 3.0]]
    attached, stats = attach_chunk_advantages([first, second], plans, rows, _cfg(center=True))
    assert attached[0].advantages[1] == pytest.approx(-1.0)
    assert attached[1].advantages[1] == pytest.approx(1.0)
    assert stats.advantage_mean == pytest.approx(0.0)
    assert stats.chunks == 2


def test_clip_none_disables_clipping() -> None:
    """A clip of None must train unclipped rather than raise.

    The same-tokenizer lane widened `train.advantage_clip` to optional in its own
    branch. This module must already tolerate that when the branches meet, or a
    valid config would raise TypeError deep inside a training step. Exercised
    against `_chunk_totals` directly, because THIS branch's config still requires
    a float, and widening a shared config's semantics is not this lane's call.
    """
    datum = _datum({1, 2}, 4, sampled_logprob=-1.0)
    plan = ChunkPlan(
        trial_name="t",
        fragment_index=0,
        teacher_token_count=3,
        chunks=[
            ChunkSpan(student_start=1, student_end=2, teacher_start=1, teacher_end=2),
            ChunkSpan(student_start=2, student_end=3, teacher_start=2, teacher_end=3),
        ],
    )
    row: list[float | None] = [None, 99.0, -0.5]
    unclipped = _chunk_totals(datum, plan, row, None)
    assert unclipped is not None
    totals, clipped = unclipped
    # The full +100.0 gap survives, and nothing is counted as clipped.
    assert totals[0] == pytest.approx(100.0)
    assert clipped == 0
    # The same input WITH a clip bounds it, proving the branch is live.
    bounded = _chunk_totals(datum, plan, row, 4.0)
    assert bounded is not None
    assert bounded[0][0] == pytest.approx(4.0)
    assert bounded[1] == 1
