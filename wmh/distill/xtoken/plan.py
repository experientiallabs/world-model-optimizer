"""Build a `ChunkPlan` by joining a student datum to a teacher render.

This is the seam where the two token spaces meet. The student side is a
`TrainDatum` whose `loss_mask` marks the tokens it actually sampled; the
teacher side is a `TeacherRender` of the same conversation under the teacher's
chat template, carrying the byte-identical content islands. `build_chunk_plan`
locates each island inside the student's sampled bytes, aligns the two token
sequences within it, and emits `ChunkSpan`s in the absolute index spaces that
`attach_chunk_advantages` expects.

Three index spaces are in play and confusing them is the main hazard, so each
is named explicitly here:

- DATUM space: positions in `datum.model_input_tokens` (context deltas plus
  sampled spans). `ChunkSpan.student_start/end` live here.
- SPAN space: positions within one contiguous sampled run, which is what byte
  offsets are computed over (a span's bytes are self-contained; the context
  tokens around it are not part of the assistant's text).
- TEACHER space: positions in `teacher_render.token_ids`.

Alignment happens in SPAN space and is shifted into DATUM space on the way out.
The Nth contiguous sampled run corresponds to the Nth assistant message, which
is how islands find their span.
"""

from __future__ import annotations

import logging
from typing import Protocol

from wmh.distill.data import TrainDatum
from wmh.distill.xtoken.aligner import AlignedPair, align_tokens
from wmh.distill.xtoken.byte_offsets import ByteOffsetTokenizer, span_byte_ends
from wmh.distill.xtoken.canonical import canonical_token
from wmh.distill.xtoken.chunks import ChunkPlan, ChunkSpan
from wmh.distill.xtoken.teacher_render import TeacherRender

logger = logging.getLogger(__name__)


class SurfaceTokenizer(ByteOffsetTokenizer, Protocol):
    """A tokenizer that also exposes surface forms, which the aligner compares.

    Structural, like the protocol it extends: a teacher-side
    `TemplateTokenizer` and the tests' deterministic fakes both satisfy it
    without inheriting from it.
    """


def sampled_runs(datum: TrainDatum) -> list[tuple[int, int]]:
    """The contiguous runs of loss positions in a datum, in DATUM space.

    Each run is one assistant turn's sampled tokens: `_merge_trial_spans`
    interleaves context deltas (mask 0.0) with sampled spans (mask 1.0), so a
    maximal run of 1.0 is exactly one turn's output.

    Args:
        datum: The datum to scan.

    Returns:
        Half-open `(start, end)` ranges, in order.
    """
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for position, weight in enumerate(datum.loss_mask):
        if weight == 1.0 and start is None:
            start = position
        elif weight != 1.0 and start is not None:
            runs.append((start, position))
            start = None
    if start is not None:
        runs.append((start, len(datum.loss_mask)))
    return runs


def _token_range_for_bytes(
    byte_ends: list[int], byte_start: int, byte_end: int
) -> tuple[int, int] | None:
    """The half-open token range whose tokens lie wholly inside a byte range.

    A token covers `[ends[i - 1], ends[i])`. Tokens straddling either edge mix
    island bytes with framing bytes and are excluded: their logprob is not
    comparable to anything on the other side.
    """
    first: int | None = None
    last: int | None = None
    previous = 0
    for index, boundary in enumerate(byte_ends):
        if previous >= byte_start and boundary <= byte_end:
            if first is None:
                first = index
            last = index
        previous = boundary
    if first is None or last is None:
        return None
    return first, last + 1


def _byte_range(ends: list[int], start: int, stop: int, origin: int) -> tuple[int, int]:
    """The byte range a half-open token range covers, relative to `origin`.

    `ends[i]` is the byte offset just past token i, so the range starts where
    token `start - 1` ended (or at 0) and finishes where token `stop - 1` ends.
    """
    begin = ends[start - 1] if start > 0 else 0
    return begin - origin, ends[stop - 1] - origin


def _pair_is_byte_aligned(
    student_ends: list[int],
    teacher_ends: list[int],
    student_origin: int,
    teacher_origin: int,
    pair_student: tuple[int, int],
    pair_teacher: tuple[int, int],
) -> bool:
    """Whether a pair covers the SAME bytes of the island on both sides.

    The island text is byte-identical across the two renders, so corresponding
    tokens must occupy identical byte ranges measured from the island's start.
    This is the guard the aligner does not apply to its DP pairs: it accepts a
    block move on joined-text equality alone, so when the true cell is wider
    than `max_comb_len` the optimal path can pair a coincidentally identical
    token from ELSEWHERE in the text (plus gap moves) and label it exact. That
    pair would have the teacher scoring different text than the student
    produced, which is silent gradient corruption rather than a crash.

    Checking here rather than trusting the aligner makes the failure
    unreachable regardless of what the DP returns, and costs two subtractions
    per pair.
    """
    student_span = _byte_range(student_ends, *pair_student, student_origin)
    teacher_span = _byte_range(teacher_ends, *pair_teacher, teacher_origin)
    return student_span == teacher_span


def boundary_partition(
    student_ends: list[int],
    teacher_ends: list[int],
    student_origin: int,
    teacher_origin: int,
    student_range: tuple[int, int],
    teacher_range: tuple[int, int],
) -> list[AlignedPair]:
    """The coarsest common refinement of two tokenizations of the same bytes.

    Both sides tokenize byte-identical island text, so they partition one byte
    interval and a chunk boundary is legitimate exactly where the two partitions
    agree. Cutting at every shared boundary and nowhere else is the finest
    alignment that never pairs different bytes: correct by construction, no
    scoring function, and O(n + m).

    This is the fallback for when the DP refuses an island. Refusing happens on
    long, poorly-anchored reasoning (measured live: a 4M-cell budget is 150x too
    small for a 24k-by-26k island, because repetitive math reasoning has too few
    unique n-grams to anchor on), and dropping the island instead cost 65,169 of
    102,346 student tokens in a single step. Falling back keeps full coverage.

    Args:
        student_ends: Cumulative byte offset per student token, span-relative.
        teacher_ends: Cumulative byte offset per teacher token, render-relative.
        student_origin: Byte offset where the island starts on the student side.
        teacher_origin: Byte offset where the island starts on the teacher side.
        student_range: Half-open student token range covering the island.
        teacher_range: Half-open teacher token range covering the island.

    Returns:
        Pairs in the SAME index space as the inputs (absolute token indices),
        with `exact` set only for one-to-one cells.
    """
    s_first, s_last = student_range
    t_first, t_last = teacher_range
    student_cuts = {student_ends[index] - student_origin: index for index in range(s_first, s_last)}
    teacher_cuts = {teacher_ends[index] - teacher_origin: index for index in range(t_first, t_last)}
    shared = sorted(set(student_cuts) & set(teacher_cuts))
    pairs: list[AlignedPair] = []
    student_start = s_first
    teacher_start = t_first
    for cut in shared:
        student_end = student_cuts[cut] + 1
        teacher_end = teacher_cuts[cut] + 1
        if student_end <= student_start or teacher_end <= teacher_start:
            continue
        pairs.append(
            AlignedPair(
                student_start=student_start,
                student_end=student_end,
                teacher_start=teacher_start,
                teacher_end=teacher_end,
                exact=(student_end - student_start == 1 and teacher_end - teacher_start == 1),
            )
        )
        student_start = student_end
        teacher_start = teacher_end
    return pairs


def build_chunk_plan(
    datum: TrainDatum,
    render: TeacherRender,
    student_tokenizer: SurfaceTokenizer,
    teacher_tokenizer: SurfaceTokenizer,
    *,
    max_cells: int | None = None,
) -> ChunkPlan:
    """Align one datum's sampled tokens against a teacher render.

    Args:
        datum: The student datum, with `loss_mask` marking sampled tokens.
        render: The same conversation rendered under the teacher's template.
        student_tokenizer: Tokenizer for the datum's ids.
        teacher_tokenizer: Tokenizer for `render.token_ids`.
        max_cells: Optional DP cell budget override, forwarded to the aligner.

    Returns:
        The plan. Islands that cannot be located, spans whose bytes cannot be
        reconstructed, and alignments that blow the cell budget are all left
        UNCOVERED rather than guessed at, so their tokens keep advantage 0.0.
        Coverage is therefore the metric to watch, and it is derivable from the
        returned plan against `datum.loss_token_count`.
    """
    runs = sampled_runs(datum)
    teacher_bytes = span_byte_ends(teacher_tokenizer, render.token_ids)
    if teacher_bytes is None:
        logger.warning(
            "cannot reconstruct teacher byte offsets for trial %s fragment %d; no chunk "
            "can be scored for it",
            datum.trial_name,
            datum.fragment_index,
        )
        return ChunkPlan(
            trial_name=datum.trial_name,
            fragment_index=datum.fragment_index,
            teacher_token_count=len(render.token_ids),
            chunks=[],
        )
    teacher_ends, _teacher_raw = teacher_bytes
    teacher_surfaces = teacher_tokenizer.convert_ids_to_tokens(list(render.token_ids))

    # The Nth assistant message maps to the Nth sampled run. Islands carry the
    # message index, so group them and walk both in order.
    assistant_indices = sorted({island.message_index for island in render.islands})
    chunks: list[ChunkSpan] = []
    fallback_islands = 0
    for run_index, (run_start, run_end) in enumerate(runs):
        if run_index >= len(assistant_indices):
            break
        message_index = assistant_indices[run_index]
        span_ids = datum.model_input_tokens[run_start:run_end]
        student_bytes = span_byte_ends(student_tokenizer, span_ids)
        if student_bytes is None:
            logger.warning(
                "cannot reconstruct student byte offsets for trial %s run %d; leaving it unscored",
                datum.trial_name,
                run_index,
            )
            continue
        student_ends, student_raw = student_bytes
        student_surfaces = student_tokenizer.convert_ids_to_tokens(list(span_ids))
        cursor = 0
        for island in render.islands:
            if island.message_index != message_index:
                continue
            needle = island.text.encode("utf-8")
            found = student_raw.find(needle, cursor)
            if found < 0:
                logger.debug(
                    "island of kind %s is not present in the student's sampled bytes for "
                    "trial %s run %d; leaving it unscored",
                    island.kind,
                    datum.trial_name,
                    run_index,
                )
                continue
            cursor = found + len(needle)
            student_range = _token_range_for_bytes(student_ends, found, cursor)
            if student_range is None:
                continue
            s_first, s_last = student_range
            t_first, t_last = island.teacher_start, island.teacher_end
            pairs = align_tokens(
                [canonical_token(surface or "") for surface in student_surfaces[s_first:s_last]],
                [canonical_token(surface or "") for surface in teacher_surfaces[t_first:t_last]],
                **({} if max_cells is None else {"max_cells": max_cells}),
            )
            if pairs is None:
                # The DP refused this island (cell budget). Do NOT drop it: the
                # bytes are identical on both sides, so the exact partition is
                # computable directly and keeps the island fully covered.
                pairs = boundary_partition(
                    student_ends,
                    teacher_ends,
                    found,
                    island.byte_start,
                    (s_first, s_last),
                    (t_first, t_last),
                )
                fallback_islands += 1
                logger.info(
                    "alignment exceeded its cell budget for trial %s run %d island %s "
                    "(%d student tokens); used the exact byte-boundary partition instead, "
                    "yielding %d chunk(s)",
                    datum.trial_name,
                    run_index,
                    island.kind,
                    s_last - s_first,
                    len(pairs),
                )
                # boundary_partition already returns absolute indices, so the
                # island-relative shift the DP path applies must not be reapplied.
                pairs = [
                    AlignedPair(
                        student_start=pair.student_start - s_first,
                        student_end=pair.student_end - s_first,
                        teacher_start=pair.teacher_start - t_first,
                        teacher_end=pair.teacher_end - t_first,
                        exact=pair.exact,
                    )
                    for pair in pairs
                ]
            misaligned = 0
            for pair in pairs:
                # Reject any pair whose two sides do not cover the same island
                # bytes, whatever the aligner claimed about it.
                if not _pair_is_byte_aligned(
                    student_ends,
                    teacher_ends,
                    found,
                    island.byte_start,
                    (s_first + pair.student_start, s_first + pair.student_end),
                    (t_first + pair.teacher_start, t_first + pair.teacher_end),
                ):
                    misaligned += 1
                    continue
                student_start = run_start + s_first + pair.student_start
                student_end = run_start + s_first + pair.student_end
                teacher_start = t_first + pair.teacher_start
                teacher_end = t_first + pair.teacher_end
                # A pair with nothing on one side is an insertion or deletion:
                # there is no counterpart to compare, so it carries no signal.
                if student_end <= student_start or teacher_end <= teacher_start:
                    continue
                # Student position 0's advantage is dropped by the next-token
                # shift, and teacher position 0 has no context to score from.
                if student_start < 1 or teacher_start < 1:
                    continue
                chunks.append(
                    ChunkSpan(
                        student_start=student_start,
                        student_end=student_end,
                        teacher_start=teacher_start,
                        teacher_end=teacher_end,
                        exact=pair.exact,
                    )
                )
            if misaligned:
                logger.warning(
                    "dropped %d of %d aligned pair(s) for trial %s island %s: the two sides "
                    "covered different island bytes, so the teacher would have scored text "
                    "the student did not produce",
                    misaligned,
                    len(pairs),
                    datum.trial_name,
                    island.kind,
                )
    if fallback_islands:
        logger.info(
            "trial %s: %d island(s) aligned by exact byte boundaries after the DP refused",
            datum.trial_name,
            fallback_islands,
        )
    return ChunkPlan(
        trial_name=datum.trial_name,
        fragment_index=datum.fragment_index,
        teacher_token_count=len(render.token_ids),
        chunks=chunks,
    )
