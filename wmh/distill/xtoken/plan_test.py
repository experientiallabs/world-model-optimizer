"""Tests for joining a student datum to a teacher render.

The byte-boundary fallback exists because the DP refuses long, poorly-anchored
reasoning islands and dropping them cost 65,169 of 102,346 student tokens in one
live step. These tests pin that a refusal keeps full coverage, that the fallback
agrees with the DP where both apply, and that byte-misaligned pairs are still
rejected however they were produced.
"""

from __future__ import annotations

from wmh.distill.data import TrainDatum
from wmh.distill.xtoken.aligner import align_tokens
from wmh.distill.xtoken.plan import (
    _pair_is_byte_aligned,
    boundary_partition,
    sampled_runs,
)


def _ends(pieces: list[bytes]) -> list[int]:
    """Cumulative byte-end offsets for a token sequence."""
    out: list[int] = []
    total = 0
    for piece in pieces:
        total += len(piece)
        out.append(total)
    return out


def test_sampled_runs_finds_each_contiguous_loss_span() -> None:
    datum = TrainDatum(
        trial_name="t",
        fragment_index=0,
        model_input_tokens=list(range(10)),
        loss_mask=[0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        sampled_logprobs=[0.0] * 10,
    )
    assert sampled_runs(datum) == [(1, 3), (5, 8)]


def test_sampled_runs_handles_a_run_ending_at_the_last_token() -> None:
    datum = TrainDatum(
        trial_name="t",
        fragment_index=0,
        model_input_tokens=[1, 2, 3],
        loss_mask=[0.0, 1.0, 1.0],
        sampled_logprobs=[0.0, -1.0, -1.0],
    )
    assert sampled_runs(datum) == [(1, 3)]


def test_boundary_partition_cuts_only_at_shared_boundaries() -> None:
    # Same 6 bytes both sides: student 'ab|cd|ef', teacher 'abc|d|ef'.
    # Shared boundaries are at bytes 4 and 6, so the first two student tokens
    # pair with the first two teacher tokens, then 'ef' pairs one to one.
    student = _ends([b"ab", b"cd", b"ef"])
    teacher = _ends([b"abc", b"d", b"ef"])
    pairs = boundary_partition(student, teacher, 0, 0, (0, 3), (0, 3))
    assert [(p.student_start, p.student_end, p.teacher_start, p.teacher_end) for p in pairs] == [
        (0, 2, 0, 2),
        (2, 3, 2, 3),
    ]
    # Only the one-to-one cell claims exactness.
    assert [p.exact for p in pairs] == [False, True]


def test_boundary_partition_covers_every_token() -> None:
    student = _ends([b"aa", b"bb", b"cc", b"dd"])
    teacher = _ends([b"a", b"abb", b"ccd", b"d"])
    pairs = boundary_partition(student, teacher, 0, 0, (0, 4), (0, 4))
    # Coverage is total on both sides: no token is left out.
    assert pairs[0].student_start == 0
    assert pairs[-1].student_end == 4
    assert pairs[0].teacher_start == 0
    assert pairs[-1].teacher_end == 4
    spans = [(p.student_start, p.student_end) for p in pairs]
    assert all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1))


def test_boundary_partition_honours_island_origins() -> None:
    # The island starts 3 bytes into the student span and 5 into the render.
    student = _ends([b"XXX", b"ab", b"cd"])
    teacher = _ends([b"YYYYY", b"abc", b"d"])
    pairs = boundary_partition(student, teacher, 3, 5, (1, 3), (1, 3))
    assert [(p.student_start, p.student_end, p.teacher_start, p.teacher_end) for p in pairs] == [
        (1, 3, 1, 3)
    ]


def test_fallback_agrees_with_the_dp_where_both_apply() -> None:
    # A small byte-identical case the DP handles comfortably: the exact partition
    # and the DP must produce the same cells, which is what makes the fallback a
    # safe substitute rather than a different objective.
    student_pieces = [b"ls", b" -", b"la", b" |", b" wc"]
    teacher_pieces = [b"l", b"s -", b"la ", b"|", b" wc"]
    dp = align_tokens(
        [piece.decode() for piece in student_pieces],
        [piece.decode() for piece in teacher_pieces],
    )
    assert dp is not None
    exact = boundary_partition(
        _ends(student_pieces),
        _ends(teacher_pieces),
        0,
        0,
        (0, len(student_pieces)),
        (0, len(teacher_pieces)),
    )
    as_tuples = [(p.student_start, p.student_end, p.teacher_start, p.teacher_end) for p in exact]
    dp_tuples = [(p.student_start, p.student_end, p.teacher_start, p.teacher_end) for p in dp]
    assert as_tuples == dp_tuples


def test_byte_alignment_guard_rejects_a_shifted_pair() -> None:
    # The reviewer's repro: student '..' + '.....' against a 5-way teacher split
    # of the same 7 bytes. Pairing student token 1 with teacher tokens 0-2 covers
    # bytes 2..7 against 0..5, which must be refused.
    student = _ends([b"..", b"....."])
    teacher = _ends([b"...", b".", b".", b".", b"."])
    assert _pair_is_byte_aligned(student, teacher, 0, 0, (1, 2), (0, 3)) is False
    assert _pair_is_byte_aligned(student, teacher, 0, 0, (0, 2), (0, 5)) is True


def test_byte_alignment_guard_accepts_every_fallback_pair() -> None:
    # Whatever the fallback emits must survive the guard by construction, since
    # both are derived from the same shared byte boundaries.
    student_pieces = [b"aa", b"bb", b"cc", b"dd", b"ee"]
    teacher_pieces = [b"a", b"abb", b"c", b"cdd", b"ee"]
    student = _ends(student_pieces)
    teacher = _ends(teacher_pieces)
    pairs = boundary_partition(student, teacher, 0, 0, (0, 5), (0, 5))
    assert pairs
    for pair in pairs:
        assert _pair_is_byte_aligned(
            student,
            teacher,
            0,
            0,
            (pair.student_start, pair.student_end),
            (pair.teacher_start, pair.teacher_end),
        )


def test_boundary_partition_on_a_single_shared_boundary_is_one_cell() -> None:
    # Worst case: the only shared boundary is the island end, so the whole island
    # becomes one coarse chunk. Still full coverage, just blunt signal.
    student = _ends([b"abc", b"de"])
    teacher = _ends([b"ab", b"cde"])
    pairs = boundary_partition(student, teacher, 0, 0, (0, 2), (0, 2))
    assert [(p.student_start, p.student_end, p.teacher_start, p.teacher_end) for p in pairs] == [
        (0, 2, 0, 2)
    ]
    assert pairs[0].exact is False
