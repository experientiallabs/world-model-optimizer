"""Provider-dialect parity over committed golden stream fixtures.

The golden fixtures are the contract: raw provider stream bytes in, the exact
canonical event sequence out. The native (Rust) normalizer is checked against
the goldens through ``normalize_stream_fixture``.
"""

from __future__ import annotations

import json
import zlib
from collections.abc import Sequence
from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import (
    GatewayEvent,
    GatewayEventKind,
)


def _sse(payload: JsonObject) -> bytes:
    """Encode one JSON object as a complete SSE data event."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


GEMINI_GOLDEN_CHUNKS: tuple[bytes, ...] = (
    _sse({"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]}),
    _sse(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"thought": True, "text": "hidden reasoning"},
                            {"text": "lo"},
                        ]
                    }
                }
            ]
        }
    ),
    _sse(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "id": "call-1",
                                    "name": "lookup",
                                    "args": {"city": "Zürich", "count": 2},
                                }
                            }
                        ]
                    }
                }
            ]
        }
    ),
    _sse(
        {
            "candidates": [{"finishReason": "STOP"}],
            "usageMetadata": {
                "promptTokenCount": 11,
                "candidatesTokenCount": 5,
                "cachedContentTokenCount": 2,
                "thoughtsTokenCount": 3,
            },
        }
    ),
)

_GEMINI_RAW_ARGUMENTS = '{"city":"Zürich","count":2}'

GEMINI_GOLDEN_EVENTS: tuple[JsonObject, ...] = (
    {"kind": "text_delta", "text": "Hel"},
    {"kind": "text_delta", "text": "lo"},
    {"kind": "tool_call_started", "index": 0, "call_id": "call-1", "name": "lookup"},
    {"kind": "tool_arguments_delta", "index": 0, "text": _GEMINI_RAW_ARGUMENTS},
    {
        "kind": "tool_call_completed",
        "index": 0,
        "call_id": "call-1",
        "name": "lookup",
        "raw_arguments": _GEMINI_RAW_ARGUMENTS,
    },
    {
        # Gemini thoughts are additive on the wire, so the engine folds the 3
        # thought tokens into the output total and reports them as its subset.
        "kind": "usage",
        "input_tokens": 11,
        "output_tokens": 8,
        "cached_input_tokens": 2,
        "reasoning_tokens": 3,
    },
    {"kind": "completed"},
)

GEMINI_REFUSAL_CHUNKS: tuple[bytes, ...] = (_sse({"candidates": [{"finishReason": "SAFETY"}]}),)

# A prompt-level block as Google delivers it (production capture shape,
# 2026-09-04): one frame, no candidates, the block named on promptFeedback,
# and usageMetadata counting the processed prompt.
GEMINI_PROMPT_BLOCK_CHUNKS: tuple[bytes, ...] = (
    _sse(
        {
            "promptFeedback": {
                "blockReason": "PROHIBITED_CONTENT",
                "safetyRatings": [
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "probability": "HIGH"}
                ],
            },
            "usageMetadata": {"promptTokenCount": 42, "totalTokenCount": 42},
        }
    ),
)

GEMINI_PROMPT_BLOCK_EVENTS: tuple[JsonObject, ...] = (
    {
        "kind": "usage",
        "input_tokens": 42,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "reasoning_tokens": None,
    },
    {
        "kind": "failed",
        "failure_class": "refusal",
        # PROHIBITED_CONTENT names the content-policy category.
        "safe_message": "provider refused the request: content policy",
        "refusal_reason": "content_policy",
    },
)

GEMINI_INCOMPLETE_CHUNKS: tuple[bytes, ...] = (
    _sse({"candidates": [{"content": {"parts": [{"text": "cut"}]}}]}),
    _sse(
        {
            "candidates": [{"finishReason": "MAX_TOKENS"}],
            "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 9},
        }
    ),
)

GEMINI_INCOMPLETE_EVENTS: tuple[JsonObject, ...] = (
    {"kind": "text_delta", "text": "cut"},
    {
        "kind": "usage",
        "input_tokens": 4,
        "output_tokens": 9,
        "cached_input_tokens": 0,
        "reasoning_tokens": None,
    },
    {"kind": "incomplete"},
)


def _native_normalized(dialect: str, chunks: Sequence[bytes]) -> JsonObject:
    """Run raw chunks through the Rust frame decoder and dialect normalizer.

    Args:
        dialect: Native dialect identifier from the wire profile.
        chunks: Raw provider stream bytes in arrival order.

    Returns:
        The decoded ``{"events": [...], "failure": ...}`` fixture result.
    """
    native = pytest.importorskip("exp_gateway_native")
    argument = json.dumps([chunk.decode("latin-1") for chunk in chunks])
    return json.loads(native.normalize_stream_fixture(dialect, argument))


def _simplified(event: GatewayEvent) -> JsonObject:
    """Project one python gateway event onto the golden fixture vocabulary.

    Args:
        event: Normalized event from a python provider mapper.

    Returns:
        The content-bearing fields in the shared fixture shape.
    """
    if event.kind is GatewayEventKind.TEXT_DELTA:
        return {"kind": "text_delta", "text": event.text_delta}
    if event.kind is GatewayEventKind.REFUSAL_DELTA:
        return {"kind": "refusal_delta", "text": event.text_delta}
    if event.kind is GatewayEventKind.REASONING_SUMMARY_DELTA:
        return {
            "kind": "reasoning_summary_delta",
            "output_index": event.reasoning_summary_output_index,
            "summary_index": event.reasoning_summary_index,
            "text": event.text_delta,
        }
    if event.kind is GatewayEventKind.TOOL_CALL_STARTED:
        return {
            "kind": "tool_call_started",
            "index": event.tool_call_index,
            "call_id": event.tool_call_id,
            "name": event.tool_name,
        }
    if event.kind is GatewayEventKind.TOOL_ARGUMENTS_DELTA:
        return {
            "kind": "tool_arguments_delta",
            "index": event.tool_call_index,
            "text": event.raw_arguments_delta,
        }
    if event.kind is GatewayEventKind.TOOL_CALL_COMPLETED:
        call = event.tool_call
        assert call is not None
        return {
            "kind": "tool_call_completed",
            "index": event.tool_call_index,
            "call_id": call.call_id,
            "name": call.name,
            "raw_arguments": call.raw_arguments,
        }
    if event.kind is GatewayEventKind.USAGE:
        usage = event.usage
        assert usage is not None
        return {
            "kind": "usage",
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
        }
    if event.kind is GatewayEventKind.COMPLETED:
        return {"kind": "completed"}
    if event.kind is GatewayEventKind.INCOMPLETE:
        return {"kind": "incomplete"}
    assert event.failure is not None
    failed: JsonObject = {
        "kind": "failed",
        "failure_class": event.failure.failure_class.value,
        "safe_message": event.failure.safe_message,
    }
    if event.failure.refusal_reason is not None:
        failed["refusal_reason"] = event.failure.refusal_reason.value
    return failed


def test_native_gemini_normalizer_matches_the_golden_fixture() -> None:
    """The Rust normalizer reproduces the committed canonical event sequence."""
    result = _native_normalized("gemini_generate_content", GEMINI_GOLDEN_CHUNKS)
    assert result["failure"] is None
    assert result["events"] == list(GEMINI_GOLDEN_EVENTS)

    incomplete = _native_normalized("gemini_generate_content", GEMINI_INCOMPLETE_CHUNKS)
    assert incomplete["failure"] is None
    assert incomplete["events"] == list(GEMINI_INCOMPLETE_EVENTS)

    refusal = _native_normalized("gemini_generate_content", GEMINI_REFUSAL_CHUNKS)
    assert refusal["failure"] is None
    assert refusal["events"] == [
        {
            "kind": "failed",
            "failure_class": "refusal",
            "safe_message": "provider refused the request: content policy",
            "refusal_reason": "content_policy",
        }
    ]


def test_native_gemini_normalizer_classifies_googles_error_envelope() -> None:
    """Google's error envelope on the stream is the provider declaring failure,
    classified by what it says: an overloaded model is a throttle (fail over,
    Retry-After), never a malformed stream end and never a synthesized
    completion after prior output. A genuine fault stays provider_internal."""
    envelope = _sse(
        {
            "error": {
                "code": 503,
                "message": "The model is overloaded. Please try again later.",
                "status": "UNAVAILABLE",
            }
        }
    )
    failed = {
        "kind": "failed",
        "failure_class": "throttled",
        "safe_message": (
            "provider throttled the request; retry after the delay in the Retry-After header"
        ),
    }
    alone = _native_normalized("gemini_generate_content", (envelope,))
    assert alone["failure"] is None
    assert alone["events"] == [failed]
    after_output = _native_normalized(
        "gemini_generate_content", (GEMINI_GOLDEN_CHUNKS[0], envelope)
    )
    assert after_output["failure"] is None
    assert after_output["events"] == [{"kind": "text_delta", "text": "Hel"}, failed]
    internal = _native_normalized(
        "gemini_generate_content",
        (
            _sse(
                {
                    "error": {
                        "code": 500,
                        "message": "Internal error encountered.",
                        "status": "INTERNAL",
                    }
                }
            ),
        ),
    )
    assert internal["events"] == [
        {
            "kind": "failed",
            "failure_class": "provider_internal",
            "safe_message": "provider stream failed",
        }
    ]


def test_native_gemini_normalizer_refuses_a_blocked_prompt() -> None:
    """A prompt Google blocks arrives with no candidates at all. It is the
    provider's refusal (the same terminal a SAFETY finish produces, after the
    usage it reported), never a stream that "ended without a terminal event"
    to be retried and failed over."""
    result = _native_normalized("gemini_generate_content", GEMINI_PROMPT_BLOCK_CHUNKS)
    assert result["failure"] is None
    assert result["events"] == list(GEMINI_PROMPT_BLOCK_EVENTS)


def test_native_gemini_normalizer_completes_a_clean_end_after_content() -> None:
    """A stream that closes after content, with no finishReason frame, is a
    normal completion: Gemini legitimately ends some streams that way, and the
    real answer must not be thrown away as malformed."""
    result = _native_normalized("gemini_generate_content", GEMINI_GOLDEN_CHUNKS[:1])
    assert result["failure"] is None
    assert result["events"] == [
        {"kind": "text_delta", "text": "Hel"},
        {"kind": "completed"},
    ]


def test_native_gemini_normalizer_fails_streams_that_produced_no_content() -> None:
    """A stream that closes having emitted NO content at all stays malformed:
    a usage-only trailer with no candidates has no answer to complete."""
    trailer = _sse({"usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 0}})
    result = _native_normalized("gemini_generate_content", (trailer,))
    assert result["events"] == []
    failure = result["failure"]
    assert isinstance(failure, dict)
    assert failure["failure_class"] == "malformed_response"


def test_native_gemini_normalizer_recovers_a_partial_before_an_abnormal_frame() -> None:
    """Content, then a structurally malformed frame: the partial answer is kept
    and the turn ends `incomplete` (an early-termination finish reason) with the
    last-seen usage folded, never discarded as malformed. Gemini uniquely ends
    legitimate turns abnormally, so a break after content is a truncated answer,
    not corruption."""
    chunks = (
        _sse({"candidates": [{"content": {"parts": [{"text": "partial"}]}}]}),
        _sse({"usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 3}}),
        # A non-string text part after content: malformed mid-stream frame.
        _sse({"candidates": [{"content": {"parts": [{"text": 5}]}}]}),
    )
    result = _native_normalized("gemini_generate_content", chunks)
    assert result["failure"] is None
    assert result["events"] == [
        {"kind": "text_delta", "text": "partial"},
        {
            "kind": "usage",
            "input_tokens": 9,
            "output_tokens": 3,
            "cached_input_tokens": 0,
            "reasoning_tokens": None,
        },
        {"kind": "incomplete"},
    ]


def test_native_gemini_normalizer_reclassifies_a_pre_content_abnormal_frame() -> None:
    """A malformed frame before any content is a Gemini abnormal end with
    nothing to salvage: it is reclassified from a hard malformed reject to a
    retryable transport failure (retry the lane, then fail over), and no content
    is emitted."""
    chunk = _sse({"candidates": [{"content": {"parts": [{"text": 5}]}}]})
    result = _native_normalized("gemini_generate_content", (chunk,))
    assert result["events"] == []
    failure = result["failure"]
    assert isinstance(failure, dict)
    assert failure["failure_class"] == "transport"


def _eventstream_message(name: str, payload: JsonObject, *, exception: bool = False) -> bytes:
    """Encode one AWS event-stream message the way Bedrock frames its stream.

    Args:
        name: Event-type (or exception-type) header value.
        payload: JSON payload object.
        exception: Whether to frame the message as a service exception.

    Returns:
        One complete binary event-stream message with valid checksums.
    """
    headers = [
        (":message-type", "exception" if exception else "event"),
        (":exception-type" if exception else ":event-type", name),
    ]
    block = b""
    for header_name, value in headers:
        block += bytes([len(header_name)]) + header_name.encode()
        block += bytes([7]) + len(value).to_bytes(2, "big") + value.encode()
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    total = 12 + len(block) + len(body) + 4
    prelude = total.to_bytes(4, "big") + len(block).to_bytes(4, "big")
    message = prelude + zlib.crc32(prelude).to_bytes(4, "big") + block + body
    return message + zlib.crc32(message).to_bytes(4, "big")


BEDROCK_GOLDEN_ENVELOPES: tuple[tuple[str, JsonObject], ...] = (
    ("messageStart", {"role": "assistant"}),
    ("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "Hel"}}),
    ("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "lo"}}),
    ("contentBlockStop", {"contentBlockIndex": 0}),
    (
        "contentBlockStart",
        {
            "contentBlockIndex": 1,
            "start": {"toolUse": {"toolUseId": "call-1", "name": "lookup"}},
        },
    ),
    ("contentBlockDelta", {"contentBlockIndex": 1, "delta": {"toolUse": {"input": '{"city":'}}}),
    (
        "contentBlockDelta",
        {"contentBlockIndex": 1, "delta": {"toolUse": {"input": '"Zürich"}'}}},
    ),
    ("contentBlockStop", {"contentBlockIndex": 1}),
    ("messageStop", {"stopReason": "tool_use"}),
    (
        "metadata",
        {
            "usage": {
                "inputTokens": 9,
                "outputTokens": 4,
                "cacheReadInputTokens": 2,
                "cacheWriteInputTokens": 1,
            },
            "metrics": {"latencyMs": 12},
        },
    ),
)

BEDROCK_GOLDEN_EVENTS: tuple[JsonObject, ...] = (
    {"kind": "text_delta", "text": "Hel"},
    {"kind": "text_delta", "text": "lo"},
    {"kind": "tool_call_started", "index": 1, "call_id": "call-1", "name": "lookup"},
    {"kind": "tool_arguments_delta", "index": 1, "text": '{"city":'},
    {"kind": "tool_arguments_delta", "index": 1, "text": '"Zürich"}'},
    {
        "kind": "tool_call_completed",
        "index": 1,
        "call_id": "call-1",
        "name": "lookup",
        "raw_arguments": '{"city":"Zürich"}',
    },
    {
        "kind": "usage",
        "input_tokens": 12,
        "output_tokens": 4,
        "cached_input_tokens": 2,
        "reasoning_tokens": None,
    },
    {"kind": "completed"},
)

BEDROCK_REFUSAL_ENVELOPES: tuple[tuple[str, JsonObject], ...] = (
    ("messageStop", {"stopReason": "guardrail_intervened"}),
    ("metadata", {"usage": {"inputTokens": 3, "outputTokens": 0}}),
)


def test_native_bedrock_normalizer_matches_the_golden_fixture() -> None:
    """The Rust event-stream decoder and normalizer reproduce the goldens."""
    chunks = [_eventstream_message(name, payload) for name, payload in BEDROCK_GOLDEN_ENVELOPES]
    result = _native_normalized("bedrock_converse_stream", chunks)
    assert result["failure"] is None
    assert result["events"] == list(BEDROCK_GOLDEN_EVENTS)

    refusal_chunks = [
        _eventstream_message(name, payload) for name, payload in BEDROCK_REFUSAL_ENVELOPES
    ]
    refusal = _native_normalized("bedrock_converse_stream", refusal_chunks)
    assert refusal["failure"] is None
    assert refusal["events"] == [
        {
            "kind": "usage",
            "input_tokens": 3,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "reasoning_tokens": None,
        },
        {
            "kind": "failed",
            "failure_class": "refusal",
            # guardrail_intervened is a content-policy verdict.
            "safe_message": "provider refused the request: content policy",
            "refusal_reason": "content_policy",
        },
    ]

    throttled = _native_normalized(
        "bedrock_converse_stream",
        [_eventstream_message("throttlingException", {"message": "x"}, exception=True)],
    )
    assert throttled["failure"] is None
    assert throttled["events"] == [
        {
            "kind": "failed",
            "failure_class": "throttled",
            "safe_message": "provider throttled the request",
        }
    ]


def test_native_bedrock_normalizer_fails_corrupt_and_truncated_frames() -> None:
    """Checksum corruption and mid-message truncation fail as malformed."""
    good = _eventstream_message("messageStart", {"role": "assistant"})
    corrupt = good[:-1] + bytes([good[-1] ^ 0xFF])
    result = _native_normalized("bedrock_converse_stream", [corrupt])
    failure = result["failure"]
    assert isinstance(failure, dict)
    assert failure["failure_class"] == "malformed_response"

    truncated = _native_normalized("bedrock_converse_stream", [good[: len(good) - 3]])
    failure = truncated["failure"]
    assert isinstance(failure, dict)
    assert failure["failure_class"] == "malformed_response"


ANTHROPIC_THINKING_CHUNKS: tuple[bytes, ...] = (
    _sse(
        {
            "type": "message_start",
            "message": {"usage": {"input_tokens": 6, "cache_read_input_tokens": 2}},
        }
    ),
    _sse(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        }
    ),
    _sse(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "step one"},
        }
    ),
    _sse(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "c2ln"},
        }
    ),
    _sse({"type": "content_block_stop", "index": 0}),
    _sse(
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "redacted_thinking", "data": "b3BhcXVl"},
        }
    ),
    _sse({"type": "content_block_stop", "index": 1}),
    _sse(
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "text", "text": ""},
        }
    ),
    _sse(
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "text_delta", "text": "Hi"},
        }
    ),
    _sse({"type": "content_block_stop", "index": 2}),
    _sse(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 9},
        }
    ),
    _sse({"type": "message_stop"}),
)

ANTHROPIC_THINKING_EVENTS: tuple[JsonObject, ...] = (
    {"kind": "thinking_delta", "index": 0, "text": "step one"},
    {"kind": "thinking_signature", "index": 0, "signature": "c2ln"},
    {"kind": "redacted_thinking", "index": 1, "data": "b3BhcXVl"},
    # Every Anthropic text block opens with its boundary event so the
    # Messages encoder mirrors the provider's block structure (citations
    # attach per block); block-less encoders ignore it.
    {"kind": "text_block_started", "index": 2},
    {"kind": "text_delta", "text": "Hi"},
    {
        "kind": "usage",
        "input_tokens": 8,
        "output_tokens": 9,
        "cached_input_tokens": 2,
        # Anthropic reports thinking inside output_tokens with no separate
        # count, so the reasoning subset stays unknown.
        "reasoning_tokens": None,
    },
    {"kind": "completed"},
)


def test_native_anthropic_normalizer_emits_thinking_events() -> None:
    """Extended-thinking frames normalize to dedicated events, never silence."""
    result = _native_normalized("anthropic_messages", ANTHROPIC_THINKING_CHUNKS)
    assert result["failure"] is None
    assert result["events"] == list(ANTHROPIC_THINKING_EVENTS)


# Captured from a live api.anthropic.com tool_use stream (2026-08-28,
# claude-haiku-4-5, ids neutralized): the real wire pads data lines with
# trailing whitespace inside the JSON, opens tool_use blocks with a `caller`
# object, nests a `cache_creation` breakdown in usage, carries
# `stop_details`, and emits one empty leading `input_json_delta`.
ANTHROPIC_LIVE_TOOL_FRAMES: tuple[bytes, ...] = (
    b'event: message_start\ndata: {"type":"message_start","message":{"model":"claude-haiku-4-5",'
    b'"id":"msg_fixture","type":"message","role":"assistant","content":[],"stop_reason":null,'
    b'"stop_sequence":null,"stop_details":null,"usage":{"input_tokens":663,'
    b'"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"cache_creation":'
    b'{"ephemeral_5m_input_tokens":0,"ephemeral_1h_input_tokens":0},"output_tokens":12,'
    b'"service_tier":"standard","inference_geo":"not_available"}}       }\n\n',
    b'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
    b'"content_block":{"type":"tool_use","id":"toolu_fixture","name":"get_weather",'
    b'"input":{},"caller":{"type":"direct"}}     }\n\n',
    b'event: ping\ndata: {"type": "ping"}\n\n',
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    b'"delta":{"type":"input_json_delta","partial_json":""}     }\n\n',
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    b'"delta":{"type":"input_json_delta","partial_json":"{\\"city\\""}   }\n\n',
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    b'"delta":{"type":"input_json_delta","partial_json":": \\"Paris\\"}"}           }\n\n',
    b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0          }\n\n',
    b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"tool_use",'
    b'"stop_sequence":null,"stop_details":null},"usage":{"input_tokens":663,'
    b'"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":33}  }\n\n',
    b'event: message_stop\ndata: {"type":"message_stop"           }\n\n',
)

ANTHROPIC_LIVE_TOOL_EVENTS: tuple[JsonObject, ...] = (
    {"kind": "tool_call_started", "index": 0, "call_id": "toolu_fixture", "name": "get_weather"},
    {"kind": "tool_arguments_delta", "index": 0, "text": ""},
    {"kind": "tool_arguments_delta", "index": 0, "text": '{"city"'},
    {"kind": "tool_arguments_delta", "index": 0, "text": ': "Paris"}'},
    {
        "kind": "tool_call_completed",
        "index": 0,
        "call_id": "toolu_fixture",
        "name": "get_weather",
        "raw_arguments": '{"city": "Paris"}',
    },
    {
        "kind": "usage",
        "input_tokens": 663,
        "output_tokens": 33,
        "cached_input_tokens": 0,
        "reasoning_tokens": None,
    },
    {"kind": "completed"},
)


def test_native_anthropic_normalizer_decodes_the_live_tool_use_wire() -> None:
    """The real captured tool_use wire decodes to the canonical event stream."""
    result = _native_normalized("anthropic_messages", ANTHROPIC_LIVE_TOOL_FRAMES)
    assert result["failure"] is None
    assert result["events"] == list(ANTHROPIC_LIVE_TOOL_EVENTS)


# Captured live (2026-08-28, claude-haiku-4-5, ids neutralized): a
# zero-argument tool call streams exactly one EMPTY input_json_delta and no
# other fragments, so completion must default the accumulated arguments to
# the canonical empty object instead of failing the stream as malformed.
ANTHROPIC_LIVE_ZERO_ARG_FRAMES: tuple[bytes, ...] = (
    b'event: message_start\ndata: {"type":"message_start","message":{"model":"claude-haiku-4-5",'
    b'"id":"msg_fixture","type":"message","role":"assistant","content":[],"stop_reason":null,'
    b'"stop_sequence":null,"stop_details":null,"usage":{"input_tokens":550,'
    b'"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":21}}      }\n\n',
    b'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
    b'"content_block":{"type":"tool_use","id":"toolu_fixture","name":"get_time",'
    b'"input":{},"caller":{"type":"direct"}}      }\n\n',
    b'event: ping\ndata: {"type": "ping"}\n\n',
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    b'"delta":{"type":"input_json_delta","partial_json":""}            }\n\n',
    b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0     }\n\n',
    b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"tool_use",'
    b'"stop_sequence":null,"stop_details":null},"usage":{"input_tokens":550,'
    b'"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":37}}\n\n',
    b'event: message_stop\ndata: {"type":"message_stop"               }\n\n',
)


def test_native_anthropic_normalizer_completes_a_zero_argument_tool_call() -> None:
    """The live zero-argument wire completes with {} instead of malforming.

    Production incident (2026-08-28): every zero-argument tool (the shape
    most coding agents ship) failed with malformed_response because the
    accumulated raw arguments stayed empty.
    """
    result = _native_normalized("anthropic_messages", ANTHROPIC_LIVE_ZERO_ARG_FRAMES)
    assert result["failure"] is None
    assert result["events"] == [
        {"kind": "tool_call_started", "index": 0, "call_id": "toolu_fixture", "name": "get_time"},
        {"kind": "tool_arguments_delta", "index": 0, "text": ""},
        # The completion-time seed streams before the completed call so every
        # downstream byte verification matches the accumulated fragments.
        {"kind": "tool_arguments_delta", "index": 0, "text": "{}"},
        {
            "kind": "tool_call_completed",
            "index": 0,
            "call_id": "toolu_fixture",
            "name": "get_time",
            "raw_arguments": "{}",
        },
        {
            "kind": "usage",
            "input_tokens": 550,
            "output_tokens": 37,
            "cached_input_tokens": 0,
            "reasoning_tokens": None,
        },
        {"kind": "completed"},
    ]


# Captured from a live api.anthropic.com web_search stream (2026-08-31,
# claude-haiku-4-5, ids neutralized, results trimmed to one and encrypted
# payloads shortened; structure and frame order are the real wire): the
# server_tool_use block streams input like a client tool but with an
# srvtoolu_ id, the whole web_search_tool_result block (caller field
# included) rides its start frame, the answer text block opens with an empty
# citations array and its citations_delta arrives BEFORE the first
# text_delta, and the terminal usage reports the true post-search input
# total (12284) that the message_start count (2230) severely undercounts.
ANTHROPIC_LIVE_WEB_SEARCH_FRAMES: tuple[bytes, ...] = (
    b'event: message_start\ndata: {"type":"message_start","message":{"model":"claude-haiku-4-5",'
    b'"id":"msg_fixture","type":"message","role":"assistant","content":[],"stop_reason":null,'
    b'"stop_sequence":null,"stop_details":null,"usage":{"input_tokens":2230,'
    b'"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"cache_creation":'
    b'{"ephemeral_5m_input_tokens":0,"ephemeral_1h_input_tokens":0},"output_tokens":25,'
    b'"service_tier":"standard","inference_geo":"not_available"}}         }\n\n',
    b'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
    b'"content_block":{"type":"server_tool_use","id":"srvtoolu_fixture","name":"web_search",'
    b'"input":{}}            }\n\n',
    b'event: ping\ndata: {"type": "ping"}\n\n',
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    b'"delta":{"type":"input_json_delta","partial_json":""}             }\n\n',
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    b'"delta":{"type":"input_json_delta","partial_json":"{\\"query\\": \\"c"}}\n\n',
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    b'"delta":{"type":"input_json_delta","partial_json":"urrent stable Python\\"}"}  }\n\n',
    b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0 }\n\n',
    b'event: content_block_start\ndata: {"type":"content_block_start","index":1,'
    b'"content_block":{"type":"web_search_tool_result","tool_use_id":"srvtoolu_fixture",'
    b'"content":[{"type":"web_search_result","title":"Python versions",'
    b'"url":"https://www.python.org/doc/versions/","encrypted_content":"Et8QCioIExgC",'
    b'"page_age":"March 12, 2026"}],"caller":{"type":"direct"}}        }\n\n',
    b'event: content_block_stop\ndata: {"type":"content_block_stop","index":1    }\n\n',
    b'event: content_block_start\ndata: {"type":"content_block_start","index":2,'
    b'"content_block":{"citations":[],"type":"text","text":""}}\n\n',
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":2,'
    b'"delta":{"type":"citations_delta","citation":{"type":"web_search_result_location",'
    b'"cited_text":"Python 3.14.7, released on 5 August 2026",'
    b'"url":"https://www.python.org/doc/versions/","title":"Python versions",'
    b'"encrypted_index":"Eo8BCioIExgC"}}  }\n\n',
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":2,'
    b'"delta":{"type":"text_delta","text":"The"}       }\n\n',
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":2,'
    b'"delta":{"type":"text_delta","text":" current stable Python version is 3.14.7."} }\n\n',
    b'event: content_block_stop\ndata: {"type":"content_block_stop","index":2 }\n\n',
    b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn",'
    b'"stop_sequence":null,"stop_details":null},"usage":{"input_tokens":12284,'
    b'"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":103,'
    b'"server_tool_use":{"web_search_requests":1,"web_fetch_requests":0}}     }\n\n',
    b'event: message_stop\ndata: {"type":"message_stop"  }\n\n',
)

ANTHROPIC_LIVE_WEB_SEARCH_EVENTS: tuple[JsonObject, ...] = (
    {
        "kind": "server_tool_use_started",
        "index": 0,
        "call_id": "srvtoolu_fixture",
        "name": "web_search",
    },
    {"kind": "server_tool_arguments_delta", "index": 0, "text": ""},
    {"kind": "server_tool_arguments_delta", "index": 0, "text": '{"query": "c'},
    {"kind": "server_tool_arguments_delta", "index": 0, "text": 'urrent stable Python"}'},
    {
        "kind": "server_tool_use_completed",
        "index": 0,
        "call_id": "srvtoolu_fixture",
        "name": "web_search",
        "raw_arguments": '{"query": "current stable Python"}',
    },
    {
        "kind": "server_tool_result",
        "index": 1,
        "block": (
            '{"type":"web_search_tool_result","tool_use_id":"srvtoolu_fixture",'
            '"content":[{"type":"web_search_result","title":"Python versions",'
            '"url":"https://www.python.org/doc/versions/","encrypted_content":"Et8QCioIExgC",'
            '"page_age":"March 12, 2026"}],"caller":{"type":"direct"}}'
        ),
    },
    {"kind": "text_block_started", "index": 2},
    {
        "kind": "citation_delta",
        "index": 2,
        "citation": (
            '{"type":"web_search_result_location",'
            '"cited_text":"Python 3.14.7, released on 5 August 2026",'
            '"url":"https://www.python.org/doc/versions/","title":"Python versions",'
            '"encrypted_index":"Eo8BCioIExgC"}'
        ),
    },
    {"kind": "text_delta", "text": "The"},
    {"kind": "text_delta", "text": " current stable Python version is 3.14.7."},
    {
        "kind": "usage",
        # The terminal usage report supersedes the start-frame input legs:
        # the model re-reads fetched results as input.
        "input_tokens": 12284,
        "output_tokens": 103,
        "cached_input_tokens": 0,
        "reasoning_tokens": None,
    },
    {"kind": "completed"},
)


def test_native_anthropic_normalizer_decodes_the_live_web_search_wire() -> None:
    """The captured WebSearch wire decodes to dedicated server-tool events.

    Production incident (2026-08-31 class): server_tool_use blocks were
    skipped as unknown, so their input_json_delta frames failed the whole
    stream as malformed, and the request itself 400d at decode.
    """
    result = _native_normalized("anthropic_messages", ANTHROPIC_LIVE_WEB_SEARCH_FRAMES)
    assert result["failure"] is None
    assert result["events"] == list(ANTHROPIC_LIVE_WEB_SEARCH_EVENTS)


def test_native_responses_preserves_multi_message_status_phase_and_idless_call() -> None:
    """OpenAI 3.x output item shapes survive normalization and public encoding."""
    from openai.types.responses.response import Response
    from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
    from openai.types.responses.response_output_item_done_event import (
        ResponseOutputItemDoneEvent,
    )
    from openai.types.responses.response_output_message import ResponseOutputMessage

    chunks = (
        _sse(
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": "rs-incomplete",
                    "type": "reasoning",
                    "summary": [],
                    "status": "in_progress",
                },
            }
        ),
        _sse(
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": "rs-incomplete",
                    "type": "reasoning",
                    "summary": [],
                    "encrypted_content": "opaque",
                    "status": "incomplete",
                },
            }
        ),
        _sse(
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {
                    "id": "msg-commentary",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "status": "in_progress",
                    "phase": "commentary",
                },
            }
        ),
        _sse(
            {
                "type": "response.output_text.delta",
                "output_index": 1,
                "item_id": "msg-commentary",
                "content_index": 0,
                "delta": "Checking.",
            }
        ),
        _sse(
            {
                "type": "response.output_item.done",
                "output_index": 1,
                "item": {
                    "id": "msg-commentary",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Checking.",
                            "annotations": [],
                        }
                    ],
                    "status": "incomplete",
                    "phase": "commentary",
                },
            }
        ),
        _sse(
            {
                "type": "response.output_item.added",
                "output_index": 2,
                "item": {
                    "type": "function_call",
                    "call_id": "call-required",
                    "name": "lookup",
                    "arguments": "",
                    "status": "in_progress",
                },
            }
        ),
        _sse(
            {
                "type": "response.output_item.done",
                "output_index": 2,
                "item": {
                    "type": "function_call",
                    "call_id": "call-required",
                    "name": "lookup",
                    "arguments": '{"query":"x"}',
                    "status": "incomplete",
                },
            }
        ),
        _sse(
            {
                "type": "response.output_item.added",
                "output_index": 3,
                "item": {
                    "id": "msg-final",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "status": "in_progress",
                    "phase": "final_answer",
                },
            }
        ),
        _sse(
            {
                "type": "response.output_text.delta",
                "output_index": 3,
                "item_id": "msg-final",
                "content_index": 0,
                "delta": "Done.",
            }
        ),
        _sse(
            {
                "type": "response.output_item.done",
                "output_index": 3,
                "item": {
                    "id": "msg-final",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Done.", "annotations": []}],
                    "status": "completed",
                    "phase": "final_answer",
                },
            }
        ),
        _sse(
            {
                "type": "response.incomplete",
                "response": {
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                },
            }
        ),
    )
    normalized = _native_normalized("openai_responses", chunks)
    assert normalized["failure"] is None
    events = cast(list[JsonObject], normalized["events"])
    completed = [event for event in events if event["kind"] == "provider_output_item_completed"]
    completed_fields = [
        (event["output_index"], event.get("status"), event.get("phase")) for event in completed
    ]
    assert completed_fields == [
        (0, "incomplete", None),
        (1, "incomplete", "commentary"),
        (2, "incomplete", None),
        (3, "completed", "final_answer"),
    ]

    native = pytest.importorskip("exp_gateway_native")
    events_json = json.dumps(events)
    body = json.loads(
        native.completed_responses_fixture(
            "request-official",
            "gpt-5.6-sol",
            1_700_000_000.0,
            "{}",
            events_json,
        )
    )
    parsed = Response.model_validate(body)
    assert [item.type for item in parsed.output] == [
        "reasoning",
        "message",
        "function_call",
        "message",
    ]
    commentary = cast(ResponseOutputMessage, parsed.output[1])
    call = cast(ResponseFunctionToolCall, parsed.output[2])
    final = cast(ResponseOutputMessage, parsed.output[3])
    assert commentary.status == "incomplete"
    assert commentary.phase == "commentary"
    assert call.id is None
    assert call.call_id == "call-required"
    assert call.status == "incomplete"
    assert final.phase == "final_answer"

    frames = native.encode_responses_fixture(
        "request-official",
        "gpt-5.6-sol",
        1_700_000_000.0,
        "{}",
        events_json,
    )
    payloads = [json.loads(frame.split("data: ", 1)[1]) for frame in frames if "data: " in frame]
    done_payloads = [
        payload for payload in payloads if payload["type"] == "response.output_item.done"
    ]
    for payload in done_payloads:
        ResponseOutputItemDoneEvent.model_validate(payload)
    assert not any(
        payload["type"].startswith("response.function_call_arguments") for payload in payloads
    )


def test_native_responses_serves_hosted_tool_items_end_to_end() -> None:
    """Hosted-tool output items (web_search_call, mcp_call) pass through the
    normalizer, the aggregated body, and the public stream verbatim, with
    URL-citation annotations attached to the answer's text part.

    Production incident (2026-09-04): the `response.output_item.added` frame
    for a web_search_call killed the whole stream as malformed_response
    post-dispatch across three orgs.
    """
    from openai.types.responses.response import Response
    from openai.types.responses.response_output_item_done_event import (
        ResponseOutputItemDoneEvent,
    )

    web_search_done = {
        "id": "ws_1",
        "type": "web_search_call",
        "status": "completed",
        "action": {"type": "search", "query": "current stable Python"},
    }
    mcp_done = {
        "id": "mcp_1",
        "type": "mcp_call",
        "server_label": "deepwiki",
        "name": "ask_question",
        "arguments": '{"q": "pi"}',
        "output": "3.14159",
        "status": "completed",
    }
    chunks = (
        _sse(
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"id": "ws_1", "type": "web_search_call", "status": "in_progress"},
            }
        ),
        _sse(
            {
                "type": "response.web_search_call.searching",
                "item_id": "ws_1",
                "output_index": 0,
                "sequence_number": 4,
            }
        ),
        _sse({"type": "response.output_item.done", "output_index": 0, "item": web_search_done}),
        _sse(
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {
                    "id": "mcp_1",
                    "type": "mcp_call",
                    "server_label": "deepwiki",
                    "name": "ask_question",
                    "arguments": "",
                },
            }
        ),
        _sse(
            {
                "type": "response.mcp_call_arguments.delta",
                "item_id": "mcp_1",
                "output_index": 1,
                "delta": '{"q": "pi"}',
                "sequence_number": 8,
            }
        ),
        _sse({"type": "response.output_item.done", "output_index": 1, "item": mcp_done}),
        _sse(
            {
                "type": "response.output_item.added",
                "output_index": 2,
                "item": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "status": "in_progress",
                },
            }
        ),
        _sse(
            {
                "type": "response.output_text.delta",
                "output_index": 2,
                "item_id": "msg_1",
                "content_index": 0,
                "delta": "Python 3.14.7.",
            }
        ),
        _sse(
            {
                "type": "response.output_text.annotation.added",
                "output_index": 2,
                "item_id": "msg_1",
                "content_index": 0,
                "annotation_index": 0,
                "annotation": {
                    "type": "url_citation",
                    "url": "https://www.python.org/doc/versions/",
                    "title": "Python versions",
                    "start_index": 0,
                    "end_index": 14,
                },
            }
        ),
        _sse(
            {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "usage": {"input_tokens": 320, "output_tokens": 41, "total_tokens": 361},
                },
            }
        ),
    )
    normalized = _native_normalized("openai_responses", chunks)
    assert normalized["failure"] is None
    events = cast(list[JsonObject], normalized["events"])
    kinds = [event["kind"] for event in events]
    assert kinds == [
        "hosted_tool_item_started",
        "hosted_tool_item_progress",
        "hosted_tool_item_completed",
        "hosted_tool_item_started",
        "hosted_tool_item_progress",
        "hosted_tool_item_completed",
        "provider_output_item_started",
        "provider_text_delta",
        "provider_text_annotation",
        "provider_output_item_completed",
        "usage",
        "completed",
    ]

    native = pytest.importorskip("exp_gateway_native")
    events_json = json.dumps(events)
    body = json.loads(
        native.completed_responses_fixture(
            "request-hosted",
            "gpt-5.6-sol",
            1_700_000_000.0,
            "{}",
            events_json,
        )
    )
    parsed = Response.model_validate(body)
    assert [item.type for item in parsed.output] == ["web_search_call", "mcp_call", "message"]
    assert body["output"][0] == web_search_done
    assert body["output"][1] == mcp_done
    message = body["output"][2]
    assert message["content"][0]["text"] == "Python 3.14.7."
    assert message["content"][0]["annotations"][0]["type"] == "url_citation"
    assert parsed.usage is not None and parsed.usage.input_tokens == 320

    frames = native.encode_responses_fixture(
        "request-hosted",
        "gpt-5.6-sol",
        1_700_000_000.0,
        "{}",
        events_json,
    )
    payloads = [json.loads(frame.split("data: ", 1)[1]) for frame in frames if "data: " in frame]
    searching = next(
        payload for payload in payloads if payload["type"] == "response.web_search_call.searching"
    )
    # The public frame is re-stamped by the gateway: its own monotonic
    # sequence, its own output index; the provider's payload fields survive.
    assert searching["output_index"] == 0
    assert searching["item_id"] == "ws_1"
    sequence_numbers = [payload["sequence_number"] for payload in payloads]
    assert sequence_numbers == sorted(set(sequence_numbers))
    for payload in payloads:
        if payload["type"] == "response.output_item.done":
            ResponseOutputItemDoneEvent.model_validate(payload)
    annotation_added = next(
        payload
        for payload in payloads
        if payload["type"] == "response.output_text.annotation.added"
    )
    assert annotation_added["annotation"]["url"] == "https://www.python.org/doc/versions/"


def test_native_responses_serves_a_budget_truncated_function_call_as_incomplete() -> None:
    """A function call the provider itself cut at max_output_tokens serves as
    an incomplete response with the truncated item intact, never a 502.

    Frame shapes captured live from api.openai.com (gpt-6-astra, 2026-09-05):
    `function_call_arguments.done` carries the PARTIAL bytes, the item's own
    status is `incomplete`, and the terminal is `response.incomplete`.
    Production incident (2026-09-05, ~5/min): the partial arguments failed the
    strict JSON completion contract and killed the stream post-dispatch.
    """
    from openai.types.responses.response import Response

    truncated_args = '{"city":"Paris","country":"France","units":"metric'
    chunks = (
        _sse(
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": "fc_astra",
                    "type": "function_call",
                    "status": "in_progress",
                    "arguments": "",
                    "call_id": "call_astra",
                    "name": "get_weather",
                },
            }
        ),
        _sse(
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_astra",
                "output_index": 0,
                "delta": truncated_args,
            }
        ),
        _sse(
            {
                "type": "response.function_call_arguments.done",
                "item_id": "fc_astra",
                "output_index": 0,
                "arguments": truncated_args,
            }
        ),
        _sse(
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": "fc_astra",
                    "type": "function_call",
                    "status": "incomplete",
                    "arguments": truncated_args,
                    "call_id": "call_astra",
                    "name": "get_weather",
                },
            }
        ),
        _sse(
            {
                "type": "response.incomplete",
                "response": {
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "usage": {"input_tokens": 62, "output_tokens": 24, "total_tokens": 86},
                },
            }
        ),
    )
    normalized = _native_normalized("openai_responses", chunks)
    assert normalized["failure"] is None
    events = cast(list[JsonObject], normalized["events"])
    kinds = [event["kind"] for event in events]
    assert "tool_call_completed" not in kinds, kinds
    assert kinds[-1] == "incomplete"

    native = pytest.importorskip("exp_gateway_native")
    events_json = json.dumps(events)
    body = json.loads(
        native.completed_responses_fixture(
            "request-astra",
            "gpt-6-astra",
            1_700_000_000.0,
            "{}",
            events_json,
        )
    )
    parsed = Response.model_validate(body)
    assert parsed.status == "incomplete"
    item = body["output"][0]
    # The caller sees the provider's honest truncation: the item at its own
    # incomplete status with the partial argument bytes, like OpenAI's wire.
    assert item["type"] == "function_call"
    assert item["status"] == "incomplete"
    assert item["arguments"] == truncated_args
