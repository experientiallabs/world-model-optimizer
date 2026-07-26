"""Cross-tokenizer token alignment: anchor-bounded Needleman-Wunsch over token spans.

`align_tokens` takes the student's token surface forms and the teacher's token
surface forms for the same text and returns the spans that cover the same
content, which is what `ChunkPlan` needs. The alignment is a monotone sequence of
`AlignedPair`s; tokens no pair covers are deliberately left out (an uncovered
student position keeps advantage 0.0, which IS the mask on the wire).

WHAT THE ALIGNMENT IS, exactly. When both token sequences spell the same bytes
(the case this package is built for: `teacher_render` reports teacher ranges
covering byte-identical message content), the two tokenizations are two
partitions of one byte interval, and the right answer is their coarsest common
refinement: cut at every byte offset where a student boundary coincides with a
teacher boundary. That answer is computable directly from byte offsets, with no
search, which is what makes this module testable rather than merely plausible
(`aligner_test.py` computes it independently and demands the DP reproduce it
exactly on real Qwen3.6 and GLM-5.2 tokenizations).

The scoring is chosen so the DP's optimum IS that partition. A matching block of
k student tokens against l teacher tokens scores
`combination_score_multiplier * (max(k, l) + 1)`, with a 1-to-1 match scoring
`exact_match_score` instead. Splitting any matching block into r >= 2 matching
sub-blocks then scores at least `c * (max(k, l) + r)`, strictly more than the
merged block's `c * (max(k, l) + 1)`, because `sum_i max(k_i, l_i) >= max(k, l)`.
So finer always wins and the optimum cannot be coarser than the common
refinement; and it cannot be finer either, since a block move requires the two
sides' text to be equal. `exact_match_score >= 2 * combination_score_multiplier`
is what keeps a 1-to-1 match from being the one block that prefers to stay
merged, and is enforced.

TEXT EQUALITY ALONE IS NOT ENOUGH, and this is the subtlest thing in the module.
A block move whose two sides merely SPELL the same characters can still pair
content from two different places in the text, and the path can reach it over
gap moves that quietly skip the tokens in between. It happens exactly when the
true cell is wider than `max_comb_len` on one side, so the honest cell is
inexpressible and some spuriously equal token from elsewhere scores better than
mismatching. Measured on random 40-character texts at `max_comb_len` 8, 300
trials produced 7 pairs that were labelled exact while covering different bytes
on the two sides, and on a two-symbol alphabet 188 of 200 trials did. Inside the
expressible regime (every cell of the reference partition at most `max_comb_len`
tokens on both sides) there were 0 disagreements over about 5,800 trials, so the
scoring argument above is sound and this is purely about the inexpressible
regime. Every candidate span therefore also has to be BOUNDARY CONSISTENT:
`_boundary_codes` labels each token boundary with its offset in the part of the
text the two sides agree on, and a span is only accepted when its start and end
boundaries carry the same label on both sides. Anchors, DP blocks, and coalesced
regions all go through the same check, so a wrong pair cannot be emitted at all
rather than being caught downstream.

The agreeing part is a REGION, not the whole span. Two renders of one turn
usually agree over a long prefix and a long suffix and differ somewhere in the
middle, so a single global "are these two texts equal" flag would switch the
guard off for the whole span the moment one token differs anywhere, which is
backwards. `_comparable_regions` therefore measures the shared prefix and the
shared suffix separately and the guard keeps protecting both of them; only spans
that touch the differing middle go unchecked, and those are reported non-exact
anyway.

Three engineering constraints drive the rest, all of them measured rather than
assumed (upstream NeMo-RL's aligner gets all three wrong):

- MEMORY. The traceback is a numpy int8 move code per cell, never an object
  array. `np.full((n + 1, m + 1), "", dtype=object)` at n = 24,852 is 4.9 GB of
  pointers before a single move is written.
- A CELL BUDGET ACROSS THE WHOLE CALL. At this kernel's measured 4.4e-7 s per
  cell, one unanchored 3,171 x 3,600 span is about 5 seconds and a full 25k x 25k
  sequence is 3 hours. `align_tokens` sums the cells of every anchor-bounded gap
  BEFORE running any DP and returns None past `max_cells`, so the caller masks
  that span instead of hanging.
- ANCHORING IS LOAD BEARING, not an optimization. Unique n-gram anchors cut the
  problem into gaps between pinned matches; without them the budget above is
  blown by any realistic trajectory. Measured on this repo's own source rendered
  through both tokenizers, a 6,382 x 5,992 pair whose full matrix is 38.3 million
  cells aligns exactly in 0.034 s because the anchors leave almost nothing to
  search.
"""

from __future__ import annotations

import logging

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from wmh.distill.xtoken.canonical import canonicalize_sequence

logger = logging.getLogger(__name__)

MAX_DP_CELLS = 4_000_000
"""Default cell budget for one `align_tokens` call.

At the measured `SECONDS_PER_CELL` this is about 2 s of DP per call, which is the
most a rollout-scoring step can spend on one span before the aligner, rather than
the model, becomes the bottleneck.
"""

SECONDS_PER_CELL = 5.0e-7
"""Measured wall clock per DP cell for this kernel, used to report the budget.

Measured over anchor-free 300 x 300, 600 x 600, and 1,200 x 1,200 alignments at
`max_comb_len` 4 (4.75e-7, 4.96e-7, 4.98e-7 s/cell); the block scan is a
constant factor per cell, so the rate does not drift with size. It went up from
4.4e-7 when the boundary-consistency check landed: one code comparison per cell
buys the guarantee that a pair covers the same bytes on both sides, which is
worth 13%.
"""

MAX_COMBINATION_LEN = 10
"""Largest `max_comb_len` the int8 move encoding can represent.

Move codes run to `4 + max_comb_len ** 2 - 1`, so 10 fits in int8 with room to
spare while 12 would overflow.
"""

_MOVE_NONE = 0
"""Unset cell; only ever read at the origin, which the traceback never enters."""

_MOVE_STUDENT_GAP = 1
"""Consume one student token with no counterpart; emits no pair."""

_MOVE_TEACHER_GAP = 2
"""Consume one teacher token with no counterpart; emits no pair."""

_MOVE_MISMATCH = 3
"""Pair one student token with one teacher token whose text differs."""

_MOVE_BLOCK_BASE = 4
"""Block moves start here: `_MOVE_BLOCK_BASE + (k - 1) * max_comb_len + (l - 1)`."""


class AlignedPair(BaseModel):
    """One aligned span: a student token range and the teacher range for the same text.

    Both ranges are half-open and 0-based INTO THE INPUT LISTS passed to
    `align_tokens` (not into the canonicalized sequences the DP runs over), so a
    caller can index its own token lists directly.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    student_start: int = Field(ge=0)
    student_end: int = Field(gt=0)
    teacher_start: int = Field(ge=0)
    teacher_end: int = Field(gt=0)
    exact: bool
    """Whether the two ranges' canonical text is identical. False means the
    aligner paired them across a mismatch, which the loss must treat as a
    lower-confidence chunk (`ChunkSpan.exact`)."""

    @model_validator(mode="after")
    def _check_ranges(self) -> AlignedPair:
        """Reject an empty or inverted range on either side.

        A pair always covers at least one token per side: an unmatched token gets
        no pair at all, which is how it stays out of the loss.
        """
        if self.student_end <= self.student_start:
            raise ValueError(
                f"student range [{self.student_start}, {self.student_end}) is empty or "
                "inverted; an aligned pair must cover at least one student token, and an "
                "unmatched token is left out of the alignment entirely"
            )
        if self.teacher_end <= self.teacher_start:
            raise ValueError(
                f"teacher range [{self.teacher_start}, {self.teacher_end}) is empty or "
                "inverted; an aligned pair must cover at least one teacher token, and an "
                "unmatched token is left out of the alignment entirely"
            )
        return self


class _Pair(BaseModel):
    """An aligned span in CANONICAL index space, before the remap to input indices."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    student_start: int
    student_end: int
    teacher_start: int
    teacher_end: int
    exact: bool


def _check_parameters(
    *,
    exact_match_score: float,
    combination_score_multiplier: float,
    gap_penalty: float,
    max_comb_len: int,
    anchor_length: int,
    max_cells: int,
) -> None:
    """Reject scoring parameters that break the alignment's correctness argument.

    Raises:
        ValueError: If a parameter is outside the range the DP is proved over.
            These are caller bugs (a bad config), not per-span evidence, so they
            raise instead of returning None.
    """
    if exact_match_score <= 0.0:
        raise ValueError(
            f"exact_match_score is {exact_match_score}; it must be positive, or a matching "
            "token pair scores no better than a mismatch. Use the default 3.0"
        )
    if combination_score_multiplier <= 0.0:
        raise ValueError(
            f"combination_score_multiplier is {combination_score_multiplier}; it must be "
            "positive, or a matching many-to-one span scores no better than a gap. Use the "
            "default 1.5"
        )
    if exact_match_score < 2.0 * combination_score_multiplier:
        raise ValueError(
            f"exact_match_score {exact_match_score} is below 2 * combination_score_multiplier "
            f"({2.0 * combination_score_multiplier}); at those weights merging two adjacent "
            "1-to-1 matches into one combined span scores HIGHER than keeping them apart, so "
            "the optimum stops being the finest shared-boundary partition. Raise "
            "exact_match_score or lower combination_score_multiplier"
        )
    if gap_penalty >= 0.0:
        raise ValueError(
            f"gap_penalty is {gap_penalty}; it must be negative, or the DP is free to skip "
            "every token. Use the default -1.5"
        )
    if not 1 <= max_comb_len <= MAX_COMBINATION_LEN:
        raise ValueError(
            f"max_comb_len is {max_comb_len}; it must be between 1 and {MAX_COMBINATION_LEN} "
            "so the traceback move code fits in the int8 array. Use the default 4, and raise "
            "it only if real spans need wider combined blocks"
        )
    if anchor_length < 1:
        raise ValueError(
            f"anchor_length is {anchor_length}; it must be at least 1 (the length of the unique "
            "n-gram that pins an anchor). Use the default 3"
        )
    if max_cells < 1:
        raise ValueError(
            f"max_cells is {max_cells}; it must be at least 1. Use the default {MAX_DP_CELLS}"
        )


class _ComparableRegions(BaseModel):
    """The prefix and the suffix over which two canonical texts spell the same bytes.

    Inside either region the two sides' character offsets stand for the same
    content, so an offset comparison PROVES whether a candidate span pairs the
    same bytes. Between the regions the texts genuinely differ and no offset
    comparison means anything, which is why this is two lengths rather than one
    global flag: one differing token in the middle must not disable the guard
    over the parts that do agree.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    student_length: int
    """Characters in the student's canonical text."""

    teacher_length: int
    """Characters in the teacher's canonical text."""

    prefix_length: int
    """Characters both texts share from the start; 0 when they differ at once."""

    suffix_length: int
    """Characters both texts share from the end, clamped so the two regions
    cannot overlap on either side."""


def _comparable_regions(student_text: str, teacher_text: str) -> _ComparableRegions:
    """Measure the shared prefix and shared suffix of two canonical texts.

    Args:
        student_text: The student's canonical tokens, joined.
        teacher_text: The teacher's canonical tokens, joined.

    Returns:
        The regions. When the two texts are equal the prefix covers everything
        and the suffix is empty, which is the byte-identical case this package is
        built for: then every boundary is checked.
    """
    prefix_length = 0
    for student_character, teacher_character in zip(student_text, teacher_text, strict=False):
        if student_character != teacher_character:
            break
        prefix_length += 1
    suffix_length = 0
    for student_character, teacher_character in zip(
        reversed(student_text), reversed(teacher_text), strict=False
    ):
        if student_character != teacher_character:
            break
        suffix_length += 1
    suffix_length = min(
        suffix_length,
        len(student_text) - prefix_length,
        len(teacher_text) - prefix_length,
    )
    return _ComparableRegions(
        student_length=len(student_text),
        teacher_length=len(teacher_text),
        prefix_length=prefix_length,
        suffix_length=suffix_length,
    )


def _boundary_codes(
    offsets: list[int], total: int, regions: _ComparableRegions
) -> list[int | None]:
    """Label every token boundary with a code that is comparable across sides.

    Two boundaries carrying the same code stand for the same position in the text
    both sides agree on, so a span whose start and end boundaries match on both
    sides covers the same bytes on both sides. A boundary in the differing middle
    gets None, meaning "no claim": it compares equal to anything, so the guard
    never rejects a span merely because the aligner cannot check it.

    Args:
        offsets: Cumulative character offset before each token, plus the total,
            as `_character_offsets` returns it.
        total: Characters in this side's canonical text.
        regions: The shared prefix and suffix lengths for the two sides.

    Returns:
        One code per entry of `offsets`. A boundary in the shared suffix codes as
        the negation of its distance from the END, minus one; one in the shared
        prefix codes as its offset from the START, so a prefix code can never
        collide with a suffix code (a span that crosses the differing middle is
        rejected rather than silently accepted).

    The SUFFIX test comes first, and the order is load-bearing. Testing the
    prefix first makes `0 <= prefix_length` true unconditionally, so offset 0
    always claims prefix code 0 even when the two texts differ at their very
    first character and `prefix_length` is therefore 0. Both sides then claim
    code 0 for positions that hold different content, and the guard accepts a
    span pairing them. Concretely, aligning ["xx","hello","world"] against
    ["hello","world"] paired "xxhello" with "hello" and destroyed the legitimate
    exact "hello"/"hello" pair; plan.py's byte check then dropped the bad pair,
    so those student tokens lost coverage entirely (measured 32.6% versus 35.6%
    student-token coverage over 544 such trials). With the suffix tested first,
    the shorter side's offset 0 falls inside the shared suffix and codes from the
    END, so it no longer collides with the longer side's true start.
    """
    codes: list[int | None] = []
    for offset in offsets:
        if total - offset <= regions.suffix_length:
            codes.append(-(total - offset) - 1)
        elif offset <= regions.prefix_length:
            codes.append(offset)
        else:
            codes.append(None)
    return codes


def _codes_compatible(student_code: int | None, teacher_code: int | None) -> bool:
    """Whether two boundary codes can stand for the same position in the text."""
    return student_code is None or teacher_code is None or student_code == teacher_code


def _span_compatible(
    student_codes: list[int | None],
    teacher_codes: list[int | None],
    *,
    student_start: int,
    student_end: int,
    teacher_start: int,
    teacher_end: int,
) -> bool:
    """Whether a candidate span's two sides sit at the same place in the text.

    Args:
        student_codes: Boundary codes for the student side, indexed by token
            boundary (so `len(student_tokens) + 1` entries).
        teacher_codes: Boundary codes for the teacher side.
        student_start: First student token of the span.
        student_end: One past the last student token of the span.
        teacher_start: First teacher token of the span.
        teacher_end: One past the last teacher token of the span.

    Returns:
        True when both the opening and the closing boundary are compatible. Both
        ends are checked because a span may open inside the shared prefix and
        close beyond it, in which case only the closing boundary is wrong.
    """
    return _codes_compatible(
        student_codes[student_start], teacher_codes[teacher_start]
    ) and _codes_compatible(student_codes[student_end], teacher_codes[teacher_end])


def _joined_spans(tokens: list[str], max_comb_len: int) -> list[list[str]]:
    """Joined text of every span of up to `max_comb_len` tokens, by span length.

    Args:
        tokens: Canonical token surface forms.
        max_comb_len: Longest span to precompute.

    Returns:
        `joined` where `joined[k][i] == "".join(tokens[i - k : i])` for
        `k >= 1` and `i >= k`; other entries are the empty string and are never
        read (the DP bounds k by i). Index 0 is unused so `k` reads directly.
    """
    count = len(tokens)
    joined: list[list[str]] = [[""] * (count + 1) for _ in range(max_comb_len + 1)]
    for index in range(1, count + 1):
        token = tokens[index - 1]
        joined[1][index] = token
        for length in range(2, min(index, max_comb_len) + 1):
            joined[length][index] = joined[length - 1][index - 1] + token
    return joined


def _align_dp(
    student: list[str],
    teacher: list[str],
    *,
    student_codes: list[int | None],
    teacher_codes: list[int | None],
    exact_match_score: float,
    combination_score_multiplier: float,
    gap_penalty: float,
    max_comb_len: int,
) -> list[_Pair]:
    """Needleman-Wunsch over token spans, with an int8 traceback.

    Moves are: a 1-to-1 pair whose text differs (`-exact_match_score`), a gap on
    either side (`gap_penalty`, emitting no pair), and a k-to-l block whose
    joined text matches AND whose two sides sit at the same place in the text
    (`exact_match_score` when k = l = 1, else
    `combination_score_multiplier * (max(k, l) + 1)`). Requiring the second
    condition is what stops the DP from pairing a spuriously equal token from
    elsewhere in the text, reached over gap moves, when the honest cell is wider
    than `max_comb_len`; see the module docstring for the measured rate.

    Args:
        student: Canonical student tokens for this segment.
        teacher: Canonical teacher tokens for this segment.
        student_codes: Boundary codes for this segment's student tokens, one per
            boundary (`len(student) + 1` entries), from `_boundary_codes`.
        teacher_codes: Boundary codes for this segment's teacher tokens.
        exact_match_score: Score for a matching 1-to-1 pair; a mismatched pair
            scores its negation.
        combination_score_multiplier: Per-`max(k, l)` score for a matching block.
        gap_penalty: Score for skipping one token on either side.
        max_comb_len: Longest block, in tokens, on either side.

    Returns:
        The pairs on the maximum-score path, in order, in this segment's own
        index space. Gap moves contribute no pair.
    """
    rows = len(student)
    columns = len(teacher)
    if not rows or not columns:
        return []
    student_joined = _joined_spans(student, max_comb_len)
    teacher_joined = _joined_spans(teacher, max_comb_len)
    # int8, never dtype=object: upstream's object traceback is 4.9 GB of pointers
    # at 24,852 tokens and 34.4 GB at 65,534, before any move is written.
    moves = np.full((rows + 1, columns + 1), _MOVE_NONE, dtype=np.int8)
    # Only the last `max_comb_len` score rows are ever read, so the score table
    # is a rolling window of plain Python lists: it keeps memory at O(max_comb_len
    # * columns) and scalar reads are several times faster than numpy indexing
    # inside this hot loop.
    previous: list[list[float]] = [[index * gap_penalty for index in range(columns + 1)]]
    mismatch_score = -exact_match_score
    block_scores = [
        [
            combination_score_multiplier * (max(student_span, teacher_span) + 1)
            for teacher_span in range(max_comb_len + 1)
        ]
        for student_span in range(max_comb_len + 1)
    ]
    if max_comb_len >= 1:
        block_scores[1][1] = exact_match_score
    for row_index in range(1, rows + 1):
        current = [0.0] * (columns + 1)
        current[0] = row_index * gap_penalty
        moves[row_index, 0] = _MOVE_STUDENT_GAP
        above = previous[0]
        span_limit = min(row_index, max_comb_len)
        row_code = student_codes[row_index]
        for column_index in range(1, columns + 1):
            best = above[column_index - 1] + mismatch_score
            best_move = _MOVE_MISMATCH
            column_limit = min(column_index, max_comb_len)
            column_code = teacher_codes[column_index]
            # Hoisted out of the block scan: every block ending in this cell
            # shares this closing boundary, so an incompatible one rules them all
            # out at once. Inlined rather than calling `_codes_compatible`
            # because this runs once per DP cell.
            if row_code is None or column_code is None or row_code == column_code:
                for student_span in range(1, span_limit + 1):
                    text = student_joined[student_span][row_index]
                    start_code = student_codes[row_index - student_span]
                    back = previous[student_span - 1]
                    scores = block_scores[student_span]
                    for teacher_span in range(1, column_limit + 1):
                        if teacher_joined[teacher_span][column_index] != text:
                            continue
                        if not _codes_compatible(
                            start_code, teacher_codes[column_index - teacher_span]
                        ):
                            continue
                        candidate = back[column_index - teacher_span] + scores[teacher_span]
                        if candidate > best:
                            best = candidate
                            best_move = (
                                _MOVE_BLOCK_BASE
                                + (student_span - 1) * max_comb_len
                                + (teacher_span - 1)
                            )
            candidate = above[column_index] + gap_penalty
            if candidate > best:
                best = candidate
                best_move = _MOVE_STUDENT_GAP
            candidate = current[column_index - 1] + gap_penalty
            if candidate > best:
                best = candidate
                best_move = _MOVE_TEACHER_GAP
            current[column_index] = best
            moves[row_index, column_index] = best_move
        previous.insert(0, current)
        del previous[max_comb_len:]

    pairs: list[_Pair] = []
    row_index, column_index = rows, columns
    while row_index > 0 and column_index > 0:
        move = int(moves[row_index, column_index])
        if move == _MOVE_STUDENT_GAP:
            row_index -= 1
            continue
        if move == _MOVE_TEACHER_GAP:
            column_index -= 1
            continue
        if move == _MOVE_MISMATCH:
            pairs.append(
                _Pair(
                    student_start=row_index - 1,
                    student_end=row_index,
                    teacher_start=column_index - 1,
                    teacher_end=column_index,
                    exact=False,
                )
            )
            row_index -= 1
            column_index -= 1
            continue
        offset = move - _MOVE_BLOCK_BASE
        student_span = offset // max_comb_len + 1
        teacher_span = offset % max_comb_len + 1
        pairs.append(
            _Pair(
                student_start=row_index - student_span,
                student_end=row_index,
                teacher_start=column_index - teacher_span,
                teacher_end=column_index,
                exact=True,
            )
        )
        row_index -= student_span
        column_index -= teacher_span
    pairs.reverse()
    return pairs


def _unique_ngram_anchors(
    student: list[str], teacher: list[str], anchor_length: int
) -> list[tuple[int, int]]:
    """Positions of every n-gram that occurs exactly once on each side.

    Args:
        student: Canonical student tokens.
        teacher: Canonical teacher tokens.
        anchor_length: n, the n-gram length.

    Returns:
        `(student_index, teacher_index)` pairs, sorted by student index. A
        repeated n-gram is not an anchor: the whole point is that its position is
        unambiguous on both sides.
    """
    student_positions: dict[tuple[str, ...], list[int]] = {}
    teacher_positions: dict[tuple[str, ...], list[int]] = {}
    for index in range(len(student) - anchor_length + 1):
        ngram = tuple(student[index : index + anchor_length])
        student_positions.setdefault(ngram, []).append(index)
    for index in range(len(teacher) - anchor_length + 1):
        ngram = tuple(teacher[index : index + anchor_length])
        teacher_positions.setdefault(ngram, []).append(index)
    anchors: list[tuple[int, int]] = []
    for ngram in student_positions.keys() & teacher_positions.keys():
        student_hits = student_positions[ngram]
        teacher_hits = teacher_positions[ngram]
        if len(student_hits) == 1 and len(teacher_hits) == 1:
            anchors.append((student_hits[0], teacher_hits[0]))
    anchors.sort()
    return anchors


def _character_offsets(tokens: list[str]) -> list[int]:
    """Cumulative character offset before each token, plus the total at the end."""
    offsets = [0] * (len(tokens) + 1)
    total = 0
    for index, token in enumerate(tokens):
        total += len(token)
        offsets[index + 1] = total
    return offsets


def _select_anchors(
    student: list[str],
    teacher: list[str],
    anchor_length: int,
    *,
    student_codes: list[int | None],
    teacher_codes: list[int | None],
) -> list[tuple[int, int]]:
    """Pick a monotone, non-overlapping chain of anchors, greedily by student index.

    An anchor whose two sides carry different boundary codes provably pairs
    different content (the n-gram is spelled the same but occurs elsewhere in the
    text) and is dropped. Upstream keeps such anchors and mis-aligns everything
    between them. The codes are per-region, so this still fires inside the parts
    of a partially mismatched span where the two texts do agree.

    Args:
        student: Canonical student tokens.
        teacher: Canonical teacher tokens.
        anchor_length: n-gram length for an anchor.
        student_codes: Boundary codes for the student side.
        teacher_codes: Boundary codes for the teacher side.

    Returns:
        Selected `(student_index, teacher_index)` anchors, sorted, with both
        sides strictly increasing and no two anchors overlapping.
    """
    selected: list[tuple[int, int]] = []
    next_student = 0
    next_teacher = 0
    for student_index, teacher_index in _unique_ngram_anchors(student, teacher, anchor_length):
        if student_index < next_student or teacher_index < next_teacher:
            continue
        if not _span_compatible(
            student_codes,
            teacher_codes,
            student_start=student_index,
            student_end=student_index + anchor_length,
            teacher_start=teacher_index,
            teacher_end=teacher_index + anchor_length,
        ):
            continue
        selected.append((student_index, teacher_index))
        next_student = student_index + anchor_length
        next_teacher = teacher_index + anchor_length
    return selected


class _Segment(BaseModel):
    """One stretch of both sequences: either a pinned anchor or a gap to align.

    Segments tile both sequences in order with no overlap, so the alignment is
    the concatenation of the segments' pairs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    student_start: int
    student_end: int
    teacher_start: int
    teacher_end: int
    is_anchor: bool
    """Anchors are runs of identical tokens, already aligned 1-to-1; gaps go to the DP."""

    @property
    def cells(self) -> int:
        """DP cells this segment costs (0 for an anchor, which needs no DP)."""
        if self.is_anchor:
            return 0
        return (self.student_end - self.student_start + 1) * (
            self.teacher_end - self.teacher_start + 1
        )


def _plan_segments(
    student_count: int,
    teacher_count: int,
    anchors: list[tuple[int, int]],
    anchor_length: int,
) -> list[_Segment]:
    """Tile both sequences into alternating gap and anchor segments, in order."""
    segments: list[_Segment] = []
    student_cursor = 0
    teacher_cursor = 0
    for student_index, teacher_index in anchors:
        if student_index > student_cursor or teacher_index > teacher_cursor:
            segments.append(
                _Segment(
                    student_start=student_cursor,
                    student_end=student_index,
                    teacher_start=teacher_cursor,
                    teacher_end=teacher_index,
                    is_anchor=False,
                )
            )
        segments.append(
            _Segment(
                student_start=student_index,
                student_end=student_index + anchor_length,
                teacher_start=teacher_index,
                teacher_end=teacher_index + anchor_length,
                is_anchor=True,
            )
        )
        student_cursor = student_index + anchor_length
        teacher_cursor = teacher_index + anchor_length
    if student_cursor < student_count or teacher_cursor < teacher_count:
        segments.append(
            _Segment(
                student_start=student_cursor,
                student_end=student_count,
                teacher_start=teacher_cursor,
                teacher_end=teacher_count,
                is_anchor=False,
            )
        )
    return segments


def _coalesce_mismatches(
    pairs: list[_Pair],
    student: list[str],
    teacher: list[str],
    *,
    student_codes: list[int | None],
    teacher_codes: list[int | None],
) -> list[_Pair]:
    """Merge each mismatched region between exact pairs into one span, then recheck it.

    A run of mismatched 1-to-1 pairs plus the gaps around it is what the DP falls
    back to for a span whose nearest shared boundaries are further apart than
    `max_comb_len` tokens, and every pair in it is individually wrong. The whole
    region between two exact pairs is merged, gapped tokens included: those
    tokens are inside the mismatched byte range, so leaving them out would both
    lose coverage and make the merged text unequal. Merged, the region's two
    sides usually spell exactly the same text, which turns a run of wrong pairs
    into one correct coarse pair. A region that still does not match stays one
    non-exact pair, which is honest: the caller can score it as a
    lower-confidence chunk or drop it.

    A merged region only claims `exact` when its two sides both spell the same
    text AND sit at the same place in it: merging is the one step that can widen
    a span, so it has to satisfy the same boundary check as every other pair or
    it becomes a second way to emit a wrong exact pair.

    Args:
        pairs: Pairs from the DP, in order, in canonical index space.
        student: Canonical student tokens.
        teacher: Canonical teacher tokens.
        student_codes: Boundary codes for the student side.
        teacher_codes: Boundary codes for the teacher side.

    Returns:
        The pairs with mismatched regions coalesced. Exact pairs pass through
        untouched, so on byte-identical input this is a no-op.
    """
    out: list[_Pair] = []
    index = 0
    count = len(pairs)
    student_cursor = 0
    teacher_cursor = 0
    while index < count:
        pair = pairs[index]
        if pair.exact:
            out.append(pair)
            student_cursor = pair.student_end
            teacher_cursor = pair.teacher_end
            index += 1
            continue
        last = index
        while last + 1 < count and not pairs[last + 1].exact:
            last += 1
        student_end = pairs[last + 1].student_start if last + 1 < count else len(student)
        teacher_end = pairs[last + 1].teacher_start if last + 1 < count else len(teacher)
        merged_student = "".join(student[student_cursor:student_end])
        merged_teacher = "".join(teacher[teacher_cursor:teacher_end])
        merged_exact = merged_student == merged_teacher and _span_compatible(
            student_codes,
            teacher_codes,
            student_start=student_cursor,
            student_end=student_end,
            teacher_start=teacher_cursor,
            teacher_end=teacher_end,
        )
        out.append(
            _Pair(
                student_start=student_cursor,
                student_end=student_end,
                teacher_start=teacher_cursor,
                teacher_end=teacher_end,
                exact=merged_exact,
            )
        )
        student_cursor = student_end
        teacher_cursor = teacher_end
        index = last + 1
    return out


def align_tokens(
    student_tokens: list[str],
    teacher_tokens: list[str],
    *,
    exact_match_score: float = 3.0,
    combination_score_multiplier: float = 1.5,
    gap_penalty: float = -1.5,
    max_comb_len: int = 4,
    anchor_length: int = 3,
    max_cells: int = MAX_DP_CELLS,
) -> list[AlignedPair] | None:
    """Align two tokenizations of the same text, span by span.

    Canonicalizes both sides (`wmh.distill.xtoken.canonical`), pins unique
    n-gram anchors, runs the span DP inside each anchor-bounded gap, coalesces
    runs of mismatched pairs, and maps every index back to the INPUT lists.

    When the two sequences spell the same bytes the result is exactly their
    coarsest common refinement: one pair per byte range delimited by boundaries
    both tokenizers share (see the module docstring for why the DP's optimum is
    that partition, and `aligner_test.py` for the independent check on real
    Qwen3.6 and GLM-5.2 tokenizations).

    Args:
        student_tokens: Student token surface forms, in order.
        teacher_tokens: Teacher token surface forms for the same text, in order.
        exact_match_score: Score for a matching 1-to-1 pair. Must be at least
            twice `combination_score_multiplier`.
        combination_score_multiplier: Per-`max(k, l)` score for a matching
            k-to-l block.
        gap_penalty: Score for leaving one token unpaired. Must be negative.
        max_comb_len: Longest matching block, in tokens, on either side. Spans
            whose shared boundaries are further apart than this are recovered by
            the mismatch coalescing step instead.
        anchor_length: n-gram length for anchoring. Longer anchors are safer and
            rarer; 3 is what the measured sequences need.
        max_cells: Budget on total DP cells across every gap in this call.

    Returns:
        The aligned pairs, in order, strictly increasing and non-overlapping on
        both sides, with indices into `student_tokens` and `teacher_tokens`. Two
        empty inputs, or one empty input, give an empty list. Returns None when
        the anchor-bounded DP would still exceed `max_cells`: the caller must
        mask that span rather than wait for it.

    Raises:
        ValueError: If a scoring parameter is outside the range the alignment is
            proved over (a config bug, unlike the budget, which is data).
    """
    _check_parameters(
        exact_match_score=exact_match_score,
        combination_score_multiplier=combination_score_multiplier,
        gap_penalty=gap_penalty,
        max_comb_len=max_comb_len,
        anchor_length=anchor_length,
        max_cells=max_cells,
    )
    if not student_tokens or not teacher_tokens:
        return []
    student_canon, student_spans = canonicalize_sequence(student_tokens)
    teacher_canon, teacher_spans = canonicalize_sequence(teacher_tokens)
    student_offsets = _character_offsets(student_canon)
    teacher_offsets = _character_offsets(teacher_canon)
    regions = _comparable_regions("".join(student_canon), "".join(teacher_canon))
    student_codes = _boundary_codes(student_offsets, regions.student_length, regions)
    teacher_codes = _boundary_codes(teacher_offsets, regions.teacher_length, regions)
    anchors = _select_anchors(
        student_canon,
        teacher_canon,
        anchor_length,
        student_codes=student_codes,
        teacher_codes=teacher_codes,
    )
    segments = _plan_segments(len(student_canon), len(teacher_canon), anchors, anchor_length)
    cells = sum(segment.cells for segment in segments)
    if cells > max_cells:
        logger.warning(
            "skipping cross-tokenizer alignment for a %d-token student span against a "
            "%d-token teacher span: the %d unique %d-gram anchor(s) leave %d DP cell(s) "
            "across %d gap(s), over the %d-cell budget (about %.1fs of DP). Mask this span, "
            "or align per message island so anchors can bound it; raise max_cells only if "
            "the wall clock is acceptable",
            len(student_tokens),
            len(teacher_tokens),
            len(anchors),
            anchor_length,
            cells,
            sum(1 for segment in segments if not segment.is_anchor),
            max_cells,
            cells * SECONDS_PER_CELL,
        )
        return None

    pairs: list[_Pair] = []
    for segment in segments:
        if segment.is_anchor:
            # An anchor is a run of identical tokens, so it splits into 1-to-1
            # pairs rather than one coarse pair: that is the finest partition of
            # it, and it is what the byte-boundary reference expects.
            for offset in range(segment.student_end - segment.student_start):
                pairs.append(
                    _Pair(
                        student_start=segment.student_start + offset,
                        student_end=segment.student_start + offset + 1,
                        teacher_start=segment.teacher_start + offset,
                        teacher_end=segment.teacher_start + offset + 1,
                        exact=True,
                    )
                )
            continue
        for pair in _align_dp(
            student_canon[segment.student_start : segment.student_end],
            teacher_canon[segment.teacher_start : segment.teacher_end],
            student_codes=student_codes[segment.student_start : segment.student_end + 1],
            teacher_codes=teacher_codes[segment.teacher_start : segment.teacher_end + 1],
            exact_match_score=exact_match_score,
            combination_score_multiplier=combination_score_multiplier,
            gap_penalty=gap_penalty,
            max_comb_len=max_comb_len,
        ):
            pairs.append(
                _Pair(
                    student_start=pair.student_start + segment.student_start,
                    student_end=pair.student_end + segment.student_start,
                    teacher_start=pair.teacher_start + segment.teacher_start,
                    teacher_end=pair.teacher_end + segment.teacher_start,
                    exact=pair.exact,
                )
            )

    pairs = _coalesce_mismatches(
        pairs,
        student_canon,
        teacher_canon,
        student_codes=student_codes,
        teacher_codes=teacher_codes,
    )
    return [
        AlignedPair(
            student_start=student_spans[pair.student_start][0],
            student_end=student_spans[pair.student_end - 1][1],
            teacher_start=teacher_spans[pair.teacher_start][0],
            teacher_end=teacher_spans[pair.teacher_end - 1][1],
            exact=pair.exact,
        )
        for pair in pairs
    ]
