"""Segment-SCOPED compression: compress tool observations only, never dialogue or the task.

C2's rounds measured why whole-context compression is the wrong shape for agent episodes:
on tau dialogue it lost 6.4 to 16.8 points of quality per cluster and INVERTED cost
(derailed episodes run longer than the tokens saved), while round 0 measured the same
mechanism on the routing channel. The compressible mass in an agent episode is the tool
observations (the literature puts them at ~84% of agent-turn tokens); the task statement
and the dialogue are where the meaning lives. This module scopes any registered compressor
to exactly that split.

A `ScopedCompressor` wraps an inner compressor and applies it ONLY to the observation span
of TOOL_CALL history entries in the eval path's rendered turn (`wmo.env.llm_agent's`
`TASK: / EPISODE SO FAR: / N. call -> observation` format, which `_CompressingProvider`
hands over as the user segment). Everything else passes through byte-exact:

- the TASK line and ENVIRONMENT NOTES (the ruled seam protection: the task/question
  segment is never compressed),
- `message:` entries and their observations (dialogue turns, both directions),
- the action text of tool calls (verbatim tool syntax),
- the trailing instruction line.

Append-stability is inherited per entry: each observation is compressed in isolation with
a deterministic inner compressor, and a history entry already rendered is re-rendered
byte-identically on later turns, so the compressed transcript grows append-only exactly
like the unscoped stage. The parse is pinned to `_render_turn`'s format by a test that
renders through the real function; if the format changes, the test breaks loudly instead
of the scope silently protecting nothing.

Scoped ids are `scoped-<inner>` (e.g. `scoped-truncate`, `scoped-llmlingua2-endpoint`),
registered as factories so the inner compressor's own construction rules (env-gated for
the endpoint) keep applying. A segment that does not parse as a rendered turn passes
through UNCOMPRESSED: outside the eval path's format the scope cannot tell observations
from dialogue, and the fail-safe direction is compressing nothing.
"""

from __future__ import annotations

import re
import time

from wmo.optimize.compression import (
    CompressionConfig,
    CompressionResult,
    Compressor,
    estimate_tokens,
    get_compressor,
    register_compressor_factory,
)

# One history entry in the rendered turn: "N. <call> -> <observation>[ [ERROR]]". Entries are
# located by their index prefix at line start; an entry runs to the next entry or the final
# instruction line. Observations may span lines (history_chars truncation preserves newlines).
_ENTRY_START = re.compile(r"(?m)^\d+\. ")
_INSTRUCTION_LINE = "Your next move (JSON only):"
_SEPARATOR = " -> "
_DIALOGUE_PREFIX = "message: "
_ERROR_SUFFIX = " [ERROR]"

SCOPE_PREFIX = "scoped-"


def _split_rendered_turn(segment: str) -> tuple[str, list[str], str] | None:
    """(head, entries, tail) of a rendered turn, or None when it is not one.

    `head` is everything through the `EPISODE SO FAR:` line; `entries` are the history
    entries; `tail` is the trailing instruction line. A turn with no history section returns
    None (nothing in scope to compress).
    """
    if _INSTRUCTION_LINE not in segment:
        return None
    body, _, _ = segment.rpartition(_INSTRUCTION_LINE)
    starts = [match.start() for match in _ENTRY_START.finditer(body)]
    if not starts:
        return None
    head = body[: starts[0]]
    entries = [body[start:end] for start, end in zip(starts, [*starts[1:], len(body)], strict=True)]
    return head, entries, _INSTRUCTION_LINE + segment.rpartition(_INSTRUCTION_LINE)[2]


def _observation_span(entry: str) -> tuple[str, str, str] | None:
    """(prefix, observation, suffix) of a TOOL entry, or None when the entry is protected.

    The prefix keeps the index and the verbatim call text plus the separator; the suffix
    keeps the error mark and the entry's trailing newline. Dialogue entries (`message:`
    actions) return None: their observations are the other party's words, not tool output.
    """
    marker = entry.find(_SEPARATOR)
    if marker < 0:
        return None
    call = entry[:marker]
    call_text = call.split(". ", 1)[1] if ". " in call else call
    if call_text.startswith(_DIALOGUE_PREFIX):
        return None
    observation = entry[marker + len(_SEPARATOR) :]
    suffix = ""
    stripped = observation.rstrip("\n")
    trailing_newlines = observation[len(stripped) :]
    if stripped.endswith(_ERROR_SUFFIX):
        stripped = stripped[: -len(_ERROR_SUFFIX)]
        suffix = _ERROR_SUFFIX
    return entry[: marker + len(_SEPARATOR)], stripped, suffix + trailing_newlines


class ScopedCompressor:
    """The scope wrapper: inner compression on tool observations, byte-exact everything else."""

    def __init__(self, inner: Compressor) -> None:
        self._inner = inner
        self.id = f"{SCOPE_PREFIX}{inner.id}"
        # The scope rule is part of the bytes-out identity, exactly like the inner version:
        # the same inner compressor under a different scope emits different bytes.
        self.version = f"{inner.version}-scope1"
        self.append_stable = inner.append_stable

    def compress(self, segments: list[str], config: CompressionConfig) -> CompressionResult:
        started = time.monotonic()
        inner_config = config.model_copy(update={"compressor_id": self._inner.id})

        parsed: list[tuple[str, list[str], str] | None] = [
            _split_rendered_turn(segment) for segment in segments
        ]
        spans: list[tuple[int, int, str, str, str]] = []  # (segment, entry, prefix, obs, suffix)
        for segment_index, split in enumerate(parsed):
            if split is None:
                continue
            for entry_index, entry in enumerate(split[1]):
                span = _observation_span(entry)
                if span is not None and span[1].strip():
                    spans.append((segment_index, entry_index, *span))

        inner_cost = 0.0
        replacements: dict[tuple[int, int], str] = {}
        if spans:
            result = self._inner.compress([span[3] for span in spans], inner_config)
            inner_cost = result.cost_usd
            for (segment_index, entry_index, prefix, _obs, suffix), compressed in zip(
                spans, result.segments, strict=True
            ):
                replacements[(segment_index, entry_index)] = prefix + compressed + suffix

        out: list[str] = []
        for segment_index, (segment, split) in enumerate(zip(segments, parsed, strict=True)):
            if split is None:
                out.append(segment)  # not a rendered turn: fail safe, compress nothing
                continue
            head, entries, tail = split
            rebuilt = [
                replacements.get((segment_index, entry_index), entry)
                for entry_index, entry in enumerate(entries)
            ]
            out.append(head + "".join(rebuilt) + tail)

        return CompressionResult(
            segments=out,
            tokens_in_raw=sum(estimate_tokens(segment) for segment in segments),
            tokens_in_compressed=sum(estimate_tokens(segment) for segment in out),
            latency_s=time.monotonic() - started,  # inner ran inline, so wall time covers it
            cost_usd=inner_cost,
        )


def _scoped_factory(inner_id: str):  # noqa: ANN202 - CompressorFactory
    def build() -> Compressor:
        return ScopedCompressor(get_compressor(inner_id))

    return build


# The round-3 arms. Factories, so the endpoint compressor's env-gated construction (and its
# fail-closed behavior without WMO_COMPRESSOR_* config) applies unchanged behind the scope.
register_compressor_factory(f"{SCOPE_PREFIX}truncate", _scoped_factory("truncate"))
register_compressor_factory(
    f"{SCOPE_PREFIX}llmlingua2-endpoint", _scoped_factory("llmlingua2-endpoint")
)
