"""Tests for canonical token surface forms."""

from __future__ import annotations

from wmh.distill.xtoken.canonical import canonical_token, canonicalize_sequence


def test_byte_level_space_marker_becomes_a_plain_space() -> None:
    assert canonical_token("Ġthe") == " the"


def test_sentencepiece_space_marker_becomes_a_plain_space() -> None:
    assert canonical_token("▁the") == " the"


def test_literal_underscore_is_not_treated_as_a_space_marker() -> None:
    """See `_MARKER_TRANSLATION`: in both target vocabularies `_` is an underscore.

    Rewriting it would make `get_name` an exact match for `get name` and mangle
    every special token that has one.
    """
    assert canonical_token("_the") == "_the"
    assert canonical_token("Ġ__init__") == " __init__"
    assert canonical_token("<|im_end|>") == "<|im_end|>"


def test_byte_level_newline_marker_becomes_a_newline() -> None:
    assert canonical_token("Ċ") == "\n"
    assert canonical_token("ĊĊ") == "\n\n"
    assert canonical_token("Ġ.Ċ") == " .\n"


def test_special_token_passes_through_unchanged() -> None:
    assert canonical_token("<|im_end|>") == "<|im_end|>"


def test_empty_token_stays_empty() -> None:
    assert canonical_token("") == ""


def test_marker_rewrite_is_position_independent() -> None:
    """The canonical text of a byte string must not depend on where it was cut.

    This is the property the aligner's correctness rests on: two tokenizations of
    one byte string have to canonicalize to ONE text. A prefix-only rewrite (only
    rewriting a marker when it starts a token) breaks it, and underscores in code
    are where it breaks first.
    """
    for left, right in (
        (["a", "_foo"], ["a_", "foo"]),
        (["__", "init", "__"], ["__init__"]),
        (["Ġ", "the"], ["Ġthe"]),
        (["ĠĊ", "x"], ["Ġ", "Ċx"]),
    ):
        left_canon, _ = canonicalize_sequence(left)
        right_canon, _ = canonicalize_sequence(right)
        assert "".join(left_canon) == "".join(right_canon)


def test_ascii_byte_fallback_token_becomes_its_character() -> None:
    assert canonical_token("<0x41>") == "A"
    assert canonical_token("<0x0A>") == "\n"


def test_ascii_byte_fallback_space_matches_the_space_markers() -> None:
    """`<0x20>` and `Ġ` stand for the same byte, so they must canonicalize alike."""
    assert canonical_token("<0x20>") == canonical_token("Ġ") == " "


def test_high_byte_fallback_token_is_left_for_sequence_merging() -> None:
    """A lone lead or continuation byte is not a character, so it is not guessed at."""
    assert canonical_token("<0xE2>") == "<0xE2>"
    assert canonical_token("<0x82>") == "<0x82>"


def test_byte_fallback_lookalikes_are_not_treated_as_bytes() -> None:
    for token in ("<0x>", "<0xZZ>", "<0x123>", "<0x41", "0x41>"):
        assert canonical_token(token) == token


def test_byte_fallback_run_merges_into_one_character() -> None:
    canonical, spans = canonicalize_sequence(["<0xE2>", "<0x82>", "<0xAC>"])
    assert canonical == ["€"]
    assert spans == [(0, 3)]


def test_byte_fallback_run_merges_into_several_characters() -> None:
    canonical, spans = canonicalize_sequence(
        ["<0xC3>", "<0xA9>", "<0xE2>", "<0x82>", "<0xAC>", "<0x41>"]
    )
    assert canonical == ["é", "€", "A"]
    assert spans == [(0, 2), (2, 5), (5, 6)]


def test_byte_fallback_run_between_ordinary_tokens_keeps_the_index_map_exact() -> None:
    canonical, spans = canonicalize_sequence(["Ġcost", "<0xE2>", "<0x82>", "<0xAC>", "100"])
    assert canonical == [" cost", "€", "100"]
    assert spans == [(0, 1), (1, 4), (4, 5)]


def test_undecodable_byte_keeps_its_original_surface_form() -> None:
    """0xFF starts no UTF-8 character; inventing chr(0xFF) would invent content."""
    canonical, spans = canonicalize_sequence(["<0xFF>", "<0x41>"])
    assert canonical == ["<0xFF>", "A"]
    assert spans == [(0, 1), (1, 2)]


def test_truncated_byte_fallback_run_keeps_the_incomplete_lead_byte() -> None:
    canonical, spans = canonicalize_sequence(["<0xE2>", "<0x82>"])
    assert canonical == ["<0xE2>", "<0x82>"]
    assert spans == [(0, 1), (1, 2)]


def test_byte_fallback_merge_matches_a_single_byte_level_token() -> None:
    """The point of the merge: a euro sign is one token on one side, three on the other."""
    fallback, _ = canonicalize_sequence(["<0xE2>", "<0x82>", "<0xAC>"])
    whole, _ = canonicalize_sequence(["€"])
    assert fallback == whole


def test_empty_sequence_canonicalizes_to_nothing() -> None:
    assert canonicalize_sequence([]) == ([], [])


def test_sequence_without_byte_fallback_is_the_per_token_map() -> None:
    tokens = ["Ġdef", "Ġcanonical", "_token", "(", "token", "):", "Ċ", "ĠĠĠĠ", "return"]
    canonical, spans = canonicalize_sequence(tokens)
    assert canonical == [canonical_token(token) for token in tokens]
    assert spans == [(index, index + 1) for index in range(len(tokens))]


def test_index_map_covers_every_input_token_exactly_once() -> None:
    tokens = ["Ġa", "<0xE2>", "<0x82>", "<0xAC>", "<0xFF>", "b", "<0xC3>", "<0xA9>"]
    canonical, spans = canonicalize_sequence(tokens)
    assert len(canonical) == len(spans)
    assert spans[0][0] == 0
    assert spans[-1][1] == len(tokens)
    for earlier, later in zip(spans, spans[1:], strict=False):
        assert earlier[1] == later[0]
        assert earlier[0] < earlier[1]
