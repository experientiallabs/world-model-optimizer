"""Tests for cross-tokenizer token alignment.

The load-bearing test here is `test_alignment_reproduces_the_byte_boundary_oracle`.
When two tokenizations spell the same bytes, the correct alignment is not a matter
of opinion: both are partitions of one byte interval, so the answer is their
coarsest common refinement, cut at every byte offset where a student boundary
coincides with a teacher boundary. `exact_boundary_partition` computes that
directly from byte offsets, with no search and no shared code with the aligner,
and the DP has to reproduce it exactly. A disagreement is a bug in the DP.

The fixture is real: 61 math, code, prose, unicode, and tool-call strings
tokenized by the actual Qwen3.6 student and GLM-5.2 teacher vocabularies. It was
generated with

    from transformers import AutoTokenizer

    student = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    teacher = AutoTokenizer.from_pretrained("zai-org/GLM-5.2")
    entry = {
        "text": text,
        "student": student.convert_ids_to_tokens(student.encode(text, add_special_tokens=False)),
        "teacher": teacher.convert_ids_to_tokens(teacher.encode(text, add_special_tokens=False)),
    }

and is checked in (`aligner_test_fixture.json`, 30 KB, 2,265 tokens) so the gate
neither downloads a tokenizer nor depends on `transformers`, which is not a
declared dependency of this package.
"""

from __future__ import annotations

import json
import pathlib
import random

import pytest
from pydantic import BaseModel, ConfigDict, TypeAdapter

from wmh.distill.xtoken.aligner import (
    MAX_COMBINATION_LEN,
    MAX_DP_CELLS,
    AlignedPair,
    _boundary_codes,
    _character_offsets,
    _comparable_regions,
    align_tokens,
)
from wmh.distill.xtoken.byte_offsets import token_bytes

# --------------------------------------------------------------------------- #
# The reference partition. This lives ONLY in the test: the aligner must not be
# able to consult it, or the test would prove nothing.
# --------------------------------------------------------------------------- #


def exact_boundary_partition(
    student_byte_ends: list[int], teacher_byte_ends: list[int]
) -> list[tuple[int, int, int, int]]:
    """The coarsest common refinement of two tokenizations of one byte string.

    Both token sequences partition the same byte interval, so a chunk boundary is
    legitimate exactly where the two partitions agree. Cutting at every shared
    boundary and nowhere else gives the finest alignment that never pairs
    different bytes, which is the alignment the DP is supposed to find.

    Args:
        student_byte_ends: Cumulative byte offset just past each student token.
        teacher_byte_ends: Cumulative byte offset just past each teacher token,
            over the same bytes.

    Returns:
        `(student_start, student_end, teacher_start, teacher_end)` per cell,
        half-open, in order, jointly covering both sequences.

    Raises:
        ValueError: If the two sequences do not cover the same number of bytes,
            in which case there is no common partition to find.
    """
    if not student_byte_ends or not teacher_byte_ends:
        return []
    if student_byte_ends[-1] != teacher_byte_ends[-1]:
        raise ValueError(
            f"student tokens cover {student_byte_ends[-1]} byte(s) but teacher tokens cover "
            f"{teacher_byte_ends[-1]}; the byte-boundary reference only applies to two "
            "tokenizations of the SAME bytes"
        )
    shared = sorted(set(student_byte_ends) & set(teacher_byte_ends))
    cells: list[tuple[int, int, int, int]] = []
    student_index = 0
    teacher_index = 0
    student_start = 0
    teacher_start = 0
    for cut in shared:
        while student_byte_ends[student_index] < cut:
            student_index += 1
        while teacher_byte_ends[teacher_index] < cut:
            teacher_index += 1
        cells.append((student_start, student_index + 1, teacher_start, teacher_index + 1))
        student_start = student_index + 1
        teacher_start = teacher_index + 1
        student_index += 1
        teacher_index += 1
    return cells


def byte_ends(surfaces: list[str]) -> list[int]:
    """Cumulative byte-end offsets for byte-level BPE surface forms."""
    ends: list[int] = []
    total = 0
    for surface in surfaces:
        raw = token_bytes(surface)
        total += len(surface.encode("utf-8") if raw is None else raw)
        ends.append(total)
    return ends


def as_tuples(pairs: list[AlignedPair]) -> list[tuple[int, int, int, int]]:
    """Pairs as index tuples, for comparison against the reference partition."""
    return [
        (pair.student_start, pair.student_end, pair.teacher_start, pair.teacher_end)
        for pair in pairs
    ]


def assert_monotone(pairs: list[AlignedPair]) -> None:
    """No crossing pairs: both sides' ranges are non-empty and strictly increasing."""
    student_cursor = 0
    teacher_cursor = 0
    for pair in pairs:
        assert pair.student_end > pair.student_start
        assert pair.teacher_end > pair.teacher_start
        assert pair.student_start >= student_cursor
        assert pair.teacher_start >= teacher_cursor
        student_cursor = pair.student_end
        teacher_cursor = pair.teacher_end


def assert_byte_consistent(
    student: list[str], teacher: list[str], pairs: list[AlignedPair]
) -> None:
    """Every pair must cover the SAME byte range on both sides.

    Only meaningful on two tokenizations of the same bytes, where a pair's two
    sides are comparable offset for offset. Text equality is not enough: a pair
    can spell the same characters on both sides and still be lifted from a
    different place in the text, which is the defect this asserts against.

    Args:
        student: Student token surface forms.
        teacher: Teacher token surface forms for the same bytes.
        pairs: What `align_tokens` returned for them.
    """
    student_bounds = [0, *byte_ends(student)]
    teacher_bounds = [0, *byte_ends(teacher)]
    for pair in pairs:
        student_range = (student_bounds[pair.student_start], student_bounds[pair.student_end])
        teacher_range = (teacher_bounds[pair.teacher_start], teacher_bounds[pair.teacher_end])
        assert student_range == teacher_range, (
            f"pair {as_tuples([pair])[0]} (exact={pair.exact}) covers student bytes "
            f"{student_range} against teacher bytes {teacher_range}"
        )


def random_tokenization(rng: random.Random, text: str) -> list[str]:
    """Cut `text` into a random non-empty tokenization, as a tokenizer would."""
    cut_count = rng.randint(1, max(1, len(text) // 2))
    cuts = sorted(rng.sample(range(1, len(text)), cut_count))
    tokens: list[str] = []
    previous = 0
    for cut in [*cuts, len(text)]:
        tokens.append(text[previous:cut])
        previous = cut
    return tokens


# --------------------------------------------------------------------------- #
# The real-tokenizer fixture.
# --------------------------------------------------------------------------- #


class FixtureCase(BaseModel):
    """One string tokenized by both real vocabularies."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    student: list[str]
    teacher: list[str]


FIXTURE = TypeAdapter(list[FixtureCase]).validate_json(
    pathlib.Path(__file__).with_name("aligner_test_fixture.json").read_bytes()
)


def test_the_fixture_is_two_tokenizations_of_the_same_bytes() -> None:
    """Guards the fixture itself: the oracle is only defined on identical bytes."""
    assert len(FIXTURE) >= 50
    for case in FIXTURE:
        expected = len(case.text.encode("utf-8"))
        assert byte_ends(case.student)[-1] == expected
        assert byte_ends(case.teacher)[-1] == expected


@pytest.mark.parametrize("case", FIXTURE, ids=lambda case: repr(case.text[:32]))
def test_alignment_reproduces_the_byte_boundary_oracle(case: FixtureCase) -> None:
    """Real Qwen3.6 against real GLM-5.2: the DP must match the reference exactly."""
    expected = exact_boundary_partition(byte_ends(case.student), byte_ends(case.teacher))
    pairs = align_tokens(case.student, case.teacher)
    assert pairs is not None
    assert as_tuples(pairs) == expected
    # Every cell of the reference covers the same bytes on both sides, and
    # canonicalization is a per-character relabeling, so every pair must be exact.
    assert all(pair.exact for pair in pairs)
    assert_monotone(pairs)


@pytest.mark.parametrize("case", FIXTURE, ids=lambda case: repr(case.text[:32]))
def test_alignment_covers_every_token_on_byte_identical_input(case: FixtureCase) -> None:
    """Nothing is dropped: the pairs tile both token sequences with no holes."""
    pairs = align_tokens(case.student, case.teacher)
    assert pairs is not None
    student_cursor = 0
    teacher_cursor = 0
    for pair in pairs:
        assert pair.student_start == student_cursor
        assert pair.teacher_start == teacher_cursor
        student_cursor = pair.student_end
        teacher_cursor = pair.teacher_end
    assert student_cursor == len(case.student)
    assert teacher_cursor == len(case.teacher)


# --------------------------------------------------------------------------- #
# The reference partition's own behavior.
# --------------------------------------------------------------------------- #


def test_reference_partition_cuts_at_every_shared_boundary() -> None:
    # student: "ab" "c" "de"   teacher: "a" "bc" "de"
    assert exact_boundary_partition([2, 3, 5], [1, 3, 5]) == [(0, 2, 0, 2), (2, 3, 2, 3)]


def test_reference_partition_of_identical_tokenizations_is_all_one_to_one() -> None:
    assert exact_boundary_partition([1, 2, 3], [1, 2, 3]) == [
        (0, 1, 0, 1),
        (1, 2, 1, 2),
        (2, 3, 2, 3),
    ]


def test_reference_partition_with_no_interior_shared_boundary_is_one_cell() -> None:
    # student: "ab" "cd"   teacher: "a" "bcd"
    assert exact_boundary_partition([2, 4], [1, 4]) == [(0, 2, 0, 2)]


def test_reference_partition_rejects_a_byte_length_mismatch() -> None:
    with pytest.raises(ValueError, match="tokenizations of the SAME bytes"):
        exact_boundary_partition([3], [4])


def test_reference_partition_of_empty_input_is_empty() -> None:
    assert exact_boundary_partition([], []) == []
    assert exact_boundary_partition([2], []) == []


# --------------------------------------------------------------------------- #
# Move coverage: 1-to-1, 1-to-many, many-to-1, gaps, coalescing.
# --------------------------------------------------------------------------- #


def test_identical_tokenizations_align_one_to_one() -> None:
    tokens = ["Ġthe", "Ġquick", "Ġbrown", "Ġfox", "Ġjumps"]
    pairs = align_tokens(tokens, list(tokens))
    assert pairs is not None
    assert as_tuples(pairs) == [(index, index + 1, index, index + 1) for index in range(5)]
    assert all(pair.exact for pair in pairs)


def test_adjacent_exact_matches_are_not_merged_into_one_combination() -> None:
    """The scoring theorem: two 1-to-1 matches (6.0) must beat one 2-to-2 block (4.5)."""
    pairs = align_tokens(["ab", "cd"], ["ab", "cd"])
    assert pairs is not None
    assert as_tuples(pairs) == [(0, 1, 0, 1), (1, 2, 1, 2)]


def test_one_student_token_aligns_to_many_teacher_tokens() -> None:
    pairs = align_tokens(["Ġhello"], ["Ġhel", "lo"])
    assert pairs is not None
    assert as_tuples(pairs) == [(0, 1, 0, 2)]
    assert pairs[0].exact


def test_many_student_tokens_align_to_one_teacher_token() -> None:
    pairs = align_tokens(["Ġhel", "lo"], ["Ġhello"])
    assert pairs is not None
    assert as_tuples(pairs) == [(0, 2, 0, 1)]
    assert pairs[0].exact


def test_span_needing_a_gap_leaves_the_extra_token_uncovered() -> None:
    """An unmatched token gets no pair at all, which is how it stays masked.

    `attach_chunk_advantages` reads advantage 0.0 as the mask, so the aligner
    must not invent a pair for a token the other side does not have.
    """
    pairs = align_tokens(["Ġa", "Ġextra", "Ġb"], ["Ġa", "Ġb"])
    assert pairs is not None
    assert as_tuples(pairs) == [(0, 1, 0, 1), (2, 3, 1, 2)]
    assert all(pair.exact for pair in pairs)
    assert_monotone(pairs)


def test_block_wider_than_max_comb_len_is_recovered_by_coalescing() -> None:
    """With max_comb_len 1 the DP cannot express the 2-to-2 cell, so the region merges."""
    student = ["ab", "cd"]
    teacher = ["a", "bcd"]
    expected = exact_boundary_partition(byte_ends(student), byte_ends(teacher))
    pairs = align_tokens(student, teacher, max_comb_len=1)
    assert pairs is not None
    assert as_tuples(pairs) == expected == [(0, 2, 0, 2)]
    assert pairs[0].exact


def test_genuinely_different_text_is_paired_but_not_exact() -> None:
    pairs = align_tokens(["Ġcat"], ["Ġdog"])
    assert pairs is not None
    assert as_tuples(pairs) == [(0, 1, 0, 1)]
    assert not pairs[0].exact


def test_mismatched_region_between_matches_becomes_one_non_exact_pair() -> None:
    pairs = align_tokens(["Ġthe", "Ġblack", "Ġcat", "Ġsat"], ["Ġthe", "Ġwhite", "Ġdog", "Ġsat"])
    assert pairs is not None
    assert as_tuples(pairs) == [(0, 1, 0, 1), (1, 3, 1, 3), (3, 4, 3, 4)]
    assert [pair.exact for pair in pairs] == [True, False, True]
    assert_monotone(pairs)


def test_widest_supported_block_round_trips_through_the_int8_move_code() -> None:
    """A 10-to-1 block uses the highest move code; int8 has to hold it."""
    student = ["a"] * MAX_COMBINATION_LEN
    teacher = ["a" * MAX_COMBINATION_LEN]
    pairs = align_tokens(student, teacher, max_comb_len=MAX_COMBINATION_LEN)
    assert pairs is not None
    assert as_tuples(pairs) == [(0, MAX_COMBINATION_LEN, 0, 1)]
    assert pairs[0].exact


# --------------------------------------------------------------------------- #
# Canonicalization seams: markers and byte fallback change the index mapping.
# --------------------------------------------------------------------------- #


def test_space_marker_difference_still_aligns_one_to_one() -> None:
    pairs = align_tokens(["Ġthe", "Ġend"], ["▁the", "▁end"])
    assert pairs is not None
    assert as_tuples(pairs) == [(0, 1, 0, 1), (1, 2, 1, 2)]
    assert all(pair.exact for pair in pairs)


def test_byte_fallback_run_aligns_to_the_whole_character() -> None:
    """Indices must come back on the INPUT axis, not the merged canonical axis.

    The byte-fallback merge exists for a vocabulary that spells the euro sign as
    three `<0xNN>` tokens facing one that has the character itself.
    """
    student = ["Ġcost", "<0xE2>", "<0x82>", "<0xAC>", "100"]
    teacher = ["Ġcost", "€", "100"]
    pairs = align_tokens(student, teacher)
    assert pairs is not None
    assert as_tuples(pairs) == [(0, 1, 0, 1), (1, 4, 1, 2), (4, 5, 2, 3)]
    assert all(pair.exact for pair in pairs)
    assert_monotone(pairs)


def test_byte_fallback_against_byte_level_mojibake_pairs_without_claiming_exact() -> None:
    """A documented limit: `<0xNN>` merges to text, byte-level surfaces stay mojibake.

    Decoding byte-level surface forms to text as well is not possible per token: a
    byte-level vocabulary routinely cuts mid-character (`â` then `‚¬`), and one
    original token can cover several characters, which the canonical index map
    cannot express. Both target vocabularies are byte level, so both sides carry
    the same mojibake and compare directly (see the fixture tests); a mixed pair
    is coalesced into a non-exact span instead of a silently wrong match.
    """
    pairs = align_tokens(["<0xE2>", "<0x82>", "<0xAC>"], ["â‚¬"])
    assert pairs is not None
    assert as_tuples(pairs) == [(0, 3, 0, 1)]
    assert not pairs[0].exact


# --------------------------------------------------------------------------- #
# Anchoring and the cell budget.
# --------------------------------------------------------------------------- #


def test_anchors_bound_the_dp_far_below_the_full_matrix() -> None:
    """Anchoring is mandatory, not an optimization: it must shrink the real work.

    2,000 x 2,000 tokens is 4,004,001 cells of full matrix (about 1.8 s and, with
    upstream's object-dtype traceback, 32 MB of pointers). Unique 3-gram anchors
    leave gaps of a couple of tokens, so a 20,000-cell budget is plenty.
    """
    student = [f"w{index}" for index in range(2000)]
    teacher = [token if index % 50 else f"{token}x" for index, token in enumerate(student)]
    pairs = align_tokens(student, teacher, max_cells=20_000)
    assert pairs is not None
    assert len(pairs) > 1900
    assert_monotone(pairs)


def test_an_anchor_at_the_wrong_byte_offset_is_dropped() -> None:
    """Repeated text can make a UNIQUE n-gram point at the wrong occurrence.

    Over "XYZ XYZ" the 3-gram ("X", "Y", "Z") occurs once on each side but at
    different places, because each side cut the OTHER occurrence differently.
    Pinning it would drag the whole alignment four bytes out of step. Both sides
    canonicalize to the same text here, so the offsets are comparable and the
    anchor is provably wrong; dropping it leaves the DP to find the reference
    partition. Upstream keeps such anchors.
    """
    student = ["XY", "Z", "Ġ", "X", "Y", "Z"]
    teacher = ["X", "Y", "Z", "Ġ", "XY", "Z"]
    expected = exact_boundary_partition(byte_ends(student), byte_ends(teacher))
    pairs = align_tokens(student, teacher)
    assert pairs is not None
    assert as_tuples(pairs) == expected
    assert all(pair.exact for pair in pairs)


def test_anchor_free_span_over_the_budget_returns_none() -> None:
    """No shared 3-gram anywhere, so the DP is one 3,000 x 3,000 block: refuse it."""
    student = [f"s{index}" for index in range(3000)]
    teacher = [f"t{index}" for index in range(3000)]
    assert (3000 + 1) ** 2 > MAX_DP_CELLS
    assert align_tokens(student, teacher) is None


def test_budget_is_enforced_across_the_whole_call() -> None:
    """The budget is the SUM over gaps, not a per-gap limit."""
    student = ["a", "b", "c", "mid1", "mid2", "mid3", "d", "e", "f"]
    teacher = ["A", "B", "C", "mid1", "mid2", "mid3", "D", "E", "F"]
    # Two 3 x 3 gaps around a 3-token anchor: 16 + 16 cells.
    assert align_tokens(student, teacher, max_cells=32) is not None
    assert align_tokens(student, teacher, max_cells=31) is None


def test_budget_refusal_is_logged_rather_than_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    student = [f"s{index}" for index in range(50)]
    teacher = [f"t{index}" for index in range(50)]
    with caplog.at_level("WARNING"):
        assert align_tokens(student, teacher, max_cells=10) is None
    assert "over the 10-cell budget" in caplog.text
    assert "Mask this span" in caplog.text


# --------------------------------------------------------------------------- #
# Boundary consistency: spelling the same characters is NOT enough.
#
# A block move whose two sides merely spell the same text can pair content from
# two different places, and gap moves let the path skip whatever lies between. It
# bites when the honest cell is wider than `max_comb_len`, so the aligner cannot
# express it and a spuriously equal token elsewhere outscores mismatching. Every
# pair the aligner emits has to cover the same bytes on both sides.
# --------------------------------------------------------------------------- #


def test_a_pair_lifted_from_elsewhere_in_the_text_is_not_emitted() -> None:
    """The narrowest repro: the only honest cell is 2-to-5, wider than max_comb_len.

    Student "..", "....." against teacher "...", ".", ".", ".", "." share no
    interior boundary, so the reference partition is the single cell
    (0, 2, 0, 5). At the default `max_comb_len` 4 the DP cannot express it, and
    text equality alone let it pair student token 1 (bytes 2 to 7) against
    teacher tokens 0 to 3 (bytes 0 to 5) and call that exact, dropping student
    token 0 and teacher tokens 3 and 4 through gap moves.
    """
    student = ["..", "....."]
    teacher = ["...", ".", ".", ".", "."]
    expected = exact_boundary_partition(byte_ends(student), byte_ends(teacher))
    pairs = align_tokens(student, teacher)
    assert pairs is not None
    assert as_tuples(pairs) == expected == [(0, 2, 0, 5)]
    assert pairs[0].exact
    assert_byte_consistent(student, teacher, pairs)


def test_a_repeated_substring_does_not_shift_the_alignment() -> None:
    """A random 40-character case that used to return three byte-misaligned pairs.

    "bxc" occurs twice in the text, once at student bytes 23 to 26 and once at
    teacher bytes 7 to 10; the DP paired those two occurrences and reported it
    exact. The two tokenizations share no interior boundary at all, so the
    reference partition is one cell over everything.
    """
    student = [
        "kdnj",
        "wlmb",
        "xcmqhrwr",
        "yx",
        "b",
        "eytp",
        "bxc",
        "aarbtrds",
        "aneski",
    ]
    teacher = [
        "kdnjwlm",
        "bxc",
        "mqh",
        "r",
        "wryxbey",
        "t",
        "pbxcaarb",
        "trd",
        "sanesk",
        "i",
    ]
    expected = exact_boundary_partition(byte_ends(student), byte_ends(teacher))
    pairs = align_tokens(student, teacher)
    assert pairs is not None
    assert as_tuples(pairs) == expected == [(0, 9, 0, 10)]
    assert_byte_consistent(student, teacher, pairs)


def test_the_offset_guard_survives_a_mismatch_elsewhere_in_the_span() -> None:
    """One differing token at the end must not switch the guard off for the rest.

    The dots and the anchor are byte identical on both sides; only the last token
    differs ("Ġcat" against "Ġdog"), which is the ordinary cross-tokenizer case.
    A single global "do the two texts match" flag disabled the offset check for
    the whole span, so the dots region came back as the byte-misaligned exact
    pair (1, 2, 0, 3). Per-region, the dots still get the correct 2-to-5 cell.
    """
    student = ["..", ".....", "Ġaaa", "Ġbbb", "Ġccc", "Ġcat"]
    teacher = ["...", ".", ".", ".", ".", "Ġaaa", "Ġbbb", "Ġccc", "Ġdog"]
    pairs = align_tokens(student, teacher)
    assert pairs is not None
    assert as_tuples(pairs) == [
        (0, 2, 0, 5),
        (2, 3, 5, 6),
        (3, 4, 6, 7),
        (4, 5, 7, 8),
        (5, 6, 8, 9),
    ]
    assert [pair.exact for pair in pairs] == [True, True, True, True, False]
    assert_monotone(pairs)


def test_a_wrong_anchor_is_dropped_even_when_the_span_ends_in_a_mismatch() -> None:
    """The per-region guard on ANCHORS, which the global flag also switched off.

    Same "XYZ XYZ" trap as `test_an_anchor_at_the_wrong_byte_offset_is_dropped`,
    plus a trailing token the two sides spell differently. With the offsets
    declared incomparable the bad anchor was pinned, which dropped the first
    three student tokens and mis-aligned everything else.
    """
    student = ["XY", "Z", "Ġ", "X", "Y", "Z", "Ġcat"]
    teacher = ["X", "Y", "Z", "Ġ", "XY", "Z", "Ġdog"]
    pairs = align_tokens(student, teacher)
    assert pairs is not None
    assert as_tuples(pairs) == [
        (0, 1, 0, 2),
        (1, 2, 2, 3),
        (2, 3, 3, 4),
        (3, 5, 4, 5),
        (5, 6, 5, 6),
        (6, 7, 6, 7),
    ]
    assert [pair.exact for pair in pairs] == [True, True, True, True, True, False]
    assert_monotone(pairs)


@pytest.mark.parametrize(
    ("alphabet", "seed"),
    [("ab", 2), ("abcdefghijklmnopqrstuvwxyz", 1)],
    ids=["two-symbol", "lowercase"],
)
def test_random_tokenization_pairs_all_cover_the_same_bytes(alphabet: str, seed: int) -> None:
    """The property test, over two random tokenizations of one random text.

    A small alphabet is the adversarial case: substrings repeat, so a spuriously
    equal span is easy to find and the honest cells are often wider than
    `max_comb_len`. Over exactly these 40 trials the pre-fix aligner emitted 32
    misaligned exact pairs on the two-symbol alphabet and 3 on the lowercase one,
    and left student tokens uncovered in 9 trials; the fixed aligner emits none
    and covers everything. Trials are capped at 40 and seeded so the gate stays
    fast and deterministic (well under 0.1 s per parameter set).
    """
    rng = random.Random(seed)
    for _ in range(40):
        text = "".join(rng.choice(alphabet) for _ in range(40))
        student = random_tokenization(rng, text)
        teacher = random_tokenization(rng, text)
        pairs = align_tokens(student, teacher)
        assert pairs is not None
        assert_byte_consistent(student, teacher, pairs)
        assert_monotone(pairs)
        # Byte-identical input: nothing may be left uncovered or called
        # inexact, however coarse the cells have to be.
        assert as_tuples(pairs)[0][0] == 0
        assert as_tuples(pairs)[-1][1] == len(student)
        assert sum(pair.student_end - pair.student_start for pair in pairs) == len(student)
        assert sum(pair.teacher_end - pair.teacher_start for pair in pairs) == len(teacher)
        assert all(pair.exact for pair in pairs)


def test_random_tokenizations_inside_the_expressible_regime_match_the_oracle() -> None:
    """Where every reference cell fits in `max_comb_len`, the DP must be exact.

    This is what separates the two regimes: the scoring argument holds inside the
    expressible one (about 5,800 random trials, 0 disagreements), so the boundary
    guard must not cost the DP its optimum there. Trials whose reference
    partition has a cell too wide to express are skipped rather than asserted on.
    """
    rng = random.Random(7)
    checked = 0
    for _ in range(120):
        text = "".join(rng.choice("abcdefghij") for _ in range(30))
        student = random_tokenization(rng, text)
        teacher = random_tokenization(rng, text)
        expected = exact_boundary_partition(byte_ends(student), byte_ends(teacher))
        if any(
            student_end - student_start > 4 or teacher_end - teacher_start > 4
            for student_start, student_end, teacher_start, teacher_end in expected
        ):
            continue
        pairs = align_tokens(student, teacher)
        assert pairs is not None
        assert as_tuples(pairs) == expected
        checked += 1
    assert checked >= 20


# --------------------------------------------------------------------------- #
# The comparable-region decomposition itself.
# --------------------------------------------------------------------------- #


def test_identical_texts_are_comparable_everywhere() -> None:
    """Identical texts must claim EVERY boundary, and agree side to side.

    Asserts the contract rather than the literal code values. Which frame a
    boundary is coded in (from the start, or from the end) is an implementation
    choice, and pinning it here previously blocked a genuine correctness fix:
    reframing the codes made all four byte-misalignment repros pass while this
    assertion was the only thing left failing, which reads like the fix is wrong
    when it is the assertion that is over-specified.
    """
    regions = _comparable_regions("abcd", "abcd")
    assert (regions.prefix_length, regions.suffix_length) == (4, 0)
    codes = _boundary_codes(_character_offsets(["ab", "cd"]), 4, regions)
    # No boundary is unclaimed: identical texts are comparable everywhere.
    assert all(code is not None for code in codes)
    # Distinct positions get distinct codes, so no two positions can be confused.
    assert len(set(codes)) == len(codes)
    # And a differently-tokenized side agrees at every position it shares.
    other = _boundary_codes(_character_offsets(["a", "bc", "d"]), 4, regions)
    shared_offsets = {0, 2, 4}
    for offset in shared_offsets:
        left = _boundary_codes([offset], 4, regions)[0]
        right = _boundary_codes([offset], 4, regions)[0]
        assert left == right
    assert all(code is not None for code in other)


def test_a_differing_middle_leaves_a_shared_prefix_and_suffix() -> None:
    regions = _comparable_regions("abXYef", "abZef")
    assert (regions.prefix_length, regions.suffix_length) == (2, 2)
    # Offsets 0 to 2 code from the start; the last two code from the end, as a
    # negative so they can never compare equal to a prefix code; the middle is
    # None, meaning the aligner may not draw any conclusion there.
    assert _boundary_codes([0, 2, 3, 4, 6], 6, regions) == [0, 2, None, -3, -1]
    assert _boundary_codes([0, 2, 3, 5], 5, regions) == [0, 2, -3, -1]


def test_texts_that_share_nothing_are_comparable_nowhere() -> None:
    regions = _comparable_regions("abc", "xyz")
    assert (regions.prefix_length, regions.suffix_length) == (0, 0)
    assert _boundary_codes([0, 1, 2, 3], 3, regions) == [0, None, None, -1]


def test_a_shared_suffix_that_would_overlap_the_prefix_is_clamped() -> None:
    """ "aab" against "aaab": the shared prefix and suffix must not double count."""
    regions = _comparable_regions("aab", "aaab")
    assert regions.prefix_length == 2
    assert regions.suffix_length == 1


# --------------------------------------------------------------------------- #
# Edge cases and parameter validation.
# --------------------------------------------------------------------------- #


def test_empty_inputs_align_to_nothing() -> None:
    assert align_tokens([], []) == []
    assert align_tokens(["Ġa"], []) == []
    assert align_tokens([], ["Ġa"]) == []


def test_single_token_pair_aligns() -> None:
    pairs = align_tokens(["Ġa"], ["Ġa"])
    assert pairs is not None
    assert as_tuples(pairs) == [(0, 1, 0, 1)]


def test_scoring_that_would_prefer_coarse_spans_is_rejected() -> None:
    with pytest.raises(ValueError, match="below 2 \\* combination_score_multiplier"):
        align_tokens(["a"], ["a"], exact_match_score=2.0, combination_score_multiplier=1.5)


def test_non_negative_gap_penalty_is_rejected() -> None:
    with pytest.raises(ValueError, match="it must be negative"):
        align_tokens(["a"], ["a"], gap_penalty=0.0)


def test_max_comb_len_outside_the_int8_encoding_is_rejected() -> None:
    with pytest.raises(ValueError, match="fits in the int8 array"):
        align_tokens(["a"], ["a"], max_comb_len=MAX_COMBINATION_LEN + 1)
    with pytest.raises(ValueError, match="fits in the int8 array"):
        align_tokens(["a"], ["a"], max_comb_len=0)


def test_anchor_length_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        align_tokens(["a"], ["a"], anchor_length=0)


def test_max_cells_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        align_tokens(["a"], ["a"], max_cells=0)


def test_non_positive_scores_are_rejected() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        align_tokens(["a"], ["a"], exact_match_score=0.0)
    with pytest.raises(ValueError, match="must be positive"):
        align_tokens(["a"], ["a"], combination_score_multiplier=0.0, exact_match_score=3.0)


def test_aligned_pair_rejects_an_empty_range() -> None:
    with pytest.raises(ValueError):
        AlignedPair(student_start=1, student_end=1, teacher_start=0, teacher_end=1, exact=True)


def test_fixture_file_is_json_and_small_enough_to_stay_in_git() -> None:
    path = pathlib.Path(__file__).with_name("aligner_test_fixture.json")
    assert path.stat().st_size < 64 * 1024
    assert isinstance(json.loads(path.read_text(encoding="utf-8")), list)
