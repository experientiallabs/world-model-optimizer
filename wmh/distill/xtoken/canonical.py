"""Canonical token surface forms, so two vocabularies can be string-compared.

The chunk aligner compares a Qwen3.6 token's surface form against a GLM-5.2
token's surface form as STRINGS. That only works if the two vocabularies spell
the same bytes the same way, and they do not always: a space is `\\u0120` in the
byte-level BPE lineage and `\\u2581` in SentencePiece, and a newline is `\\u010a`
byte-level and a literal newline elsewhere. A SentencePiece vocabulary can also
emit one token PER BYTE for a character outside its vocabulary
(`<0xE2><0x82><0xAC>` for the euro sign), so the same character is one token on
one side and three on the other.

Two properties make this module's normalization safe for the aligner, and both
are easy to lose:

1. It is POSITION INDEPENDENT. Every rewrite is a per-character relabeling
   applied wherever the character occurs, never "only at the start of a token".
   A prefix-only rewrite (upstream's `token.startswith("_")` rule) makes the
   canonical text depend on where the tokenizer happened to cut: `['a', '_b']`
   would canonicalize to `"a b"` while `['a_', 'b']` canonicalizes to `"a_b"`,
   even though both spell the same bytes. The aligner's whole correctness
   argument is that two tokenizations of one byte string canonicalize to ONE
   text, so a split-dependent rewrite would silently break alignment on exactly
   the inputs that are supposed to be easiest.

2. It never DROPS content. Upstream additionally maps `"\\u0120,"` to `","`,
   deleting a space byte; two tokenizations that cut around that space then
   disagree about the text they cover. Nothing here changes the length of what a
   token stands for except the byte-fallback merge, which is reported through the
   index map.

Rule 1 is also why the marker set stops at characters that can ONLY mean a space
or a newline; see `_MARKER_TRANSLATION` for the measured reason a plain `_` is
not one of them.

Rule 1 also bounds what this module can do. It does NOT decode byte-level surface
forms to text (`\\u00e2\\u201a\\u00ac` stays as it is rather than becoming a euro
sign): a byte-level vocabulary cuts mid-character freely, so decoding per token
would decode `\\u00e2\\u201a\\u00ac` and leave a neighbouring `\\u00e2` alone,
which is exactly the position dependence rule 1 forbids, and one token covering
several characters cannot be split without breaking the index map below. Both
vocabularies this package aligns are byte level, so they carry the SAME mojibake
and compare directly; the byte-fallback merge is what bridges a SentencePiece
vocabulary to a vocabulary that has the character itself.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_MARKER_TRANSLATION = str.maketrans(
    {
        "Ġ": " ",  # byte-level BPE (GPT-2 lineage): the space byte
        "▁": " ",  # SentencePiece: the word-start space marker
        "Ċ": "\n",  # byte-level BPE: the newline byte
    }
)
"""Marker characters that unambiguously stand for a space or a newline.

A plain `_` is NOT here, though upstream rewrites it to a space. It is a space
marker in a few hand-rolled SentencePiece exports and a literal underscore
everywhere else, and the two are indistinguishable from the surface form alone.
Measured on the vocabularies this package aligns, U+2581 appears in 0 of 248,077
Qwen3.6 tokens and 0 of 154,856 GLM-5.2 tokens (both spell a space as U+0120),
while a literal underscore appears in 5,683 and 5,715 tokens respectively. So the
rewrite would never once be correct here, and it would cost real precision:
`<|im_end|>` would canonicalize to `<|im end|>`, and a `get_name` span would
become an EXACT match for a `get name` span, feeding the loss a chunk whose two
sides do not spell the same bytes. A vocabulary that really does use `_` as its
space marker has to be fixed at the tokenizer boundary (rewrite `_` to U+2581
when loading it), where the convention is known, rather than guessed at per
token.
"""

_BYTE_FALLBACK_LENGTH = 6
"""Length of a SentencePiece byte-fallback token, `<0xNN>`."""

_MAX_UTF8_CHAR_BYTES = 4


def _byte_fallback_value(token: str) -> int | None:
    """The byte a `<0xNN>` byte-fallback token stands for, else None.

    Args:
        token: A token surface form.

    Returns:
        The byte value 0..255, or None when `token` is not a byte-fallback
        token. The check is deliberately strict about length and the hex digits
        so an ordinary token that merely looks similar (`<0x>` in a code
        snippet) is left alone.
    """
    if len(token) != _BYTE_FALLBACK_LENGTH or not token.startswith("<0x") or token[-1] != ">":
        return None
    try:
        return int(token[3:5], 16)
    except ValueError:
        return None


def canonical_token(token: str) -> str:
    """Normalize one token's surface form to the text it stands for.

    Args:
        token: A token surface form from `convert_ids_to_tokens`.

    Returns:
        The canonical form: space markers become a plain space, the byte-level
        newline becomes a newline, and an ASCII byte-fallback token becomes its
        character. A byte-fallback token for a byte >= 0x80 is returned
        UNCHANGED, because a lone lead or continuation byte is not a character;
        `canonicalize_sequence` is what merges those runs. Special and added
        tokens (`<|im_end|>`) contain no marker characters and pass through.
    """
    if not token:
        return token
    value = _byte_fallback_value(token)
    if value is not None:
        if value < 0x80:
            return chr(value).translate(_MARKER_TRANSLATION)
        return token
    return token.translate(_MARKER_TRANSLATION)


def _decode_byte_run(raw: bytes) -> list[tuple[str | None, int]]:
    """Split a run of raw bytes into whole UTF-8 characters, greedily.

    Args:
        raw: The bytes of a maximal run of consecutive byte-fallback tokens, one
            byte per token, in order.

    Returns:
        One `(text, byte_length)` entry per output character. `text` is None
        when the bytes at that position do not start a valid UTF-8 character, in
        which case `byte_length` is 1 and the caller keeps the original
        `<0xNN>` token so it cannot spuriously match anything.
    """
    pieces: list[tuple[str | None, int]] = []
    position = 0
    while position < len(raw):
        decoded: str | None = None
        length = 1
        for candidate in range(1, _MAX_UTF8_CHAR_BYTES + 1):
            if position + candidate > len(raw):
                break
            try:
                text = raw[position : position + candidate].decode("utf-8")
            except UnicodeDecodeError:
                continue
            decoded = text
            length = candidate
            break
        pieces.append((decoded, length))
        position += length
    return pieces


def canonicalize_sequence(tokens: list[str]) -> tuple[list[str], list[tuple[int, int]]]:
    """Canonicalize a whole token sequence, merging byte-fallback runs.

    Byte-fallback merging is why this cannot be a per-token map: three
    `<0xNN>` tokens can stand for ONE character, so the canonical sequence is
    shorter than the input and every index the aligner produces has to be
    translated back.

    Args:
        tokens: Token surface forms, in order.

    Returns:
        `(canonical, spans)` where `canonical[k]` is the k-th canonical token
        and `spans[k]` is the half-open range `[start, end)` of ORIGINAL token
        indices it was built from. `spans` is strictly increasing,
        non-overlapping, and covers `range(len(tokens))` exactly, so a caller
        can map any canonical range back to an original range with
        `spans[first][0]` and `spans[last - 1][1]`.
    """
    canonical: list[str] = []
    spans: list[tuple[int, int]] = []
    index = 0
    count = len(tokens)
    while index < count:
        value = _byte_fallback_value(tokens[index])
        if value is None:
            canonical.append(canonical_token(tokens[index]))
            spans.append((index, index + 1))
            index += 1
            continue
        run: list[int] = []
        while index + len(run) < count:
            byte_value = _byte_fallback_value(tokens[index + len(run)])
            if byte_value is None:
                break
            run.append(byte_value)
        for text, length in _decode_byte_run(bytes(run)):
            start = index
            index += length
            if text is None:
                # An undecodable byte keeps its original surface form: pretending
                # it is chr(byte) would invent a character the text never had.
                canonical.append(tokens[start])
            else:
                canonical.append(text.translate(_MARKER_TRANSLATION))
            spans.append((start, index))
    return canonical, spans
