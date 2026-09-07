"""Round-trip and rejection tests for the Anthropic Messages decoder."""

from __future__ import annotations

from typing import cast

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.common.models.content import MAXIMUM_DOCUMENTS_PER_REQUEST
from exp.runtime.anthropic_protocol.requests import decode_messages
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayNamedToolChoice,
    RedactedThinkingBlock,
    ThinkingBlock,
)
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.wire_messages import anthropic_blocks
from exp.runtime.openai_protocol.errors import OpenAIProtocolError

_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)
"""One valid single-pixel PNG, base64 encoded."""


def _body(**overrides: JsonValue) -> JsonObject:
    """Return one minimal valid Messages body with overrides applied."""
    payload: JsonObject = {
        "model": "coding",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "hi"}],
    }
    payload.update(overrides)
    return payload


def test_decode_full_request_is_lossless() -> None:
    """Every supported field lands on the canonical request."""
    decoded = decode_messages(
        _body(
            system=[{"type": "text", "text": "be terse"}, {"type": "text", "text": "and kind"}],
            temperature=0.5,
            top_p=0.9,
            stop_sequences=["STOP", "STOP", "END"],
            stream=True,
            tools=[
                {
                    "name": "search",
                    "description": "look things up",
                    "input_schema": {"type": "object"},
                }
            ],
            tool_choice={"type": "tool", "name": "search", "disable_parallel_tool_use": True},
            metadata={"user_id": "user-1"},
        )
    )
    request = decoded.request
    assert decoded.alias == "coding"
    assert request.surface == GatewayApiSurface.MESSAGES
    assert request.messages[0].role == "system"
    assert request.messages[0].content == "be terse\n\nand kind"
    assert request.messages[1].role == "user"
    assert request.maximum_output_tokens == 128
    assert request.temperature == 0.5
    assert request.top_p == 0.9
    assert request.stop == ("STOP", "END")
    assert request.stream is True
    assert request.include_usage is True
    assert request.tools[0].name == "search"
    assert request.tools[0].parameters == {"type": "object"}
    assert request.tool_choice == GatewayNamedToolChoice(name="search")
    assert request.parallel_tool_calls is False
    assert request.metadata == {"user_id": "user-1"}
    assert request.idempotency_key is None
    assert request.client_request_id is None


def test_decode_splits_tool_results_and_keeps_assistant_tool_calls() -> None:
    """A mixed history turn splits into ordered canonical messages."""
    decoded = decode_messages(
        _body(
            messages=[
                {"role": "user", "content": "run the tool"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "on it"},
                        {
                            "type": "tool_use",
                            "id": "call-1",
                            "name": "search",
                            "input": {"q": "x"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-1",
                            "content": [{"type": "text", "text": "found it"}],
                        },
                        {"type": "text", "text": "now answer"},
                    ],
                },
            ]
        )
    )
    roles = [message.role for message in decoded.request.messages]
    assert roles == ["user", "assistant", "tool", "user"]
    assistant = decoded.request.messages[1]
    assert assistant.content == "on it"
    assert assistant.tool_calls[0].call_id == "call-1"
    assert assistant.tool_calls[0].raw_arguments == '{"q":"x"}'
    tool = decoded.request.messages[2]
    assert tool.tool_call_id == "call-1"
    assert tool.content == "found it"
    assert decoded.request.messages[3].content == "now answer"


def test_decode_drops_only_nonsemantic_cache_control() -> None:
    """A cache hint may be omitted without changing the requested model behavior."""
    decoded = decode_messages(
        _body(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "hi",
                            "cache_control": {"type": "ephemeral", "ttl": "5m"},
                        }
                    ],
                }
            ],
        )
    )
    assert decoded.request.messages[0].content == "hi"
    assert decoded.request.metadata == {}


def test_output_config_is_carried_verbatim_and_maps_canonical_effort() -> None:
    """The caller's output_config survives byte-for-byte and its effort rides
    the shared reasoning_effort field (Claude Code sends {"effort": ...} by
    default; accepted live without a beta, 2026-08-30)."""
    decoded = decode_messages(_body(output_config={"effort": "high"}))
    assert decoded.request.provider_output_config == {"effort": "high"}
    assert decoded.request.reasoning_effort == "high"
    # A non-canonical (future provider) effort stays verbatim-only: the
    # provider decides it, the gateway does not reject it.
    future = decode_messages(_body(output_config={"effort": "hyperdrive"}))
    assert future.request.provider_output_config == {"effort": "hyperdrive"}
    assert future.request.reasoning_effort is None
    assert decode_messages(_body()).request.provider_output_config is None


def test_thinking_config_is_carried_verbatim() -> None:
    """The caller's thinking object survives byte-for-byte on the canonical request."""
    config: JsonObject = {"type": "enabled", "budget_tokens": 1024}
    decoded = decode_messages(_body(thinking=config))
    assert decoded.request.provider_thinking_config == config
    assert decode_messages(_body()).request.provider_thinking_config is None

    with pytest.raises(OpenAIProtocolError) as excinfo:
        decode_messages(_body(thinking={"type": "enabled"}))
    assert excinfo.value.detail.param == "thinking"
    with pytest.raises(OpenAIProtocolError):
        decode_messages(_body(thinking={"type": "adaptive", "budget_tokens": 64}))


def test_interleaved_thinking_turn_keeps_its_block_order_for_replay() -> None:
    """A thinking turn carries its blocks in the caller's order alongside the
    flattened fields, so the Anthropic wire can replay it byte-for-byte:
    interleaved thinking puts thinking between tool_use blocks, and the
    provider refuses a reordered latest assistant message."""
    blocks: list[JsonObject] = [
        {"type": "thinking", "thinking": "plan", "signature": "sig-a"},
        {"type": "tool_use", "id": "call-1", "name": "read", "input": {"path": "a"}},
        {"type": "thinking", "thinking": "next", "signature": "sig-b"},
        {"type": "text", "text": "reading"},
        {"type": "tool_use", "id": "call-2", "name": "read", "input": {"path": "b"}},
    ]
    decoded = decode_messages(
        _body(
            messages=[{"role": "user", "content": "go"}, {"role": "assistant", "content": blocks}]
        )
    )
    assistant = decoded.request.messages[1]
    assert assistant.provider_anthropic_blocks == tuple(blocks)
    assert [block.kind for block in assistant.provider_reasoning] == ["thinking", "thinking"]
    assert [call.call_id for call in assistant.tool_calls] == ["call-1", "call-2"]
    role, wire = anthropic_blocks(assistant)
    assert role == "assistant"
    assert wire == blocks

    # An empty text block carrying Claude Code's cache marker drops (the wire
    # rejects it) and its breakpoint lands on the closest prior block that can
    # carry one, skipping the signed thinking block.
    marked = decode_messages(
        _body(
            messages=[
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": [
                        *blocks[:3],
                        {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}},
                        *blocks[3:],
                    ],
                },
            ]
        )
    )
    _role, migrated = anthropic_blocks(marked.request.messages[1])
    assert len(migrated) == len(blocks)
    assert migrated[1] == {**blocks[1], "cache_control": {"type": "ephemeral"}}
    assert migrated[2] == blocks[2]
    assert [block["type"] for block in migrated] == [block["type"] for block in blocks]

    # A turn without thinking has no signatures to protect and stays flattened.
    plain = decode_messages(
        _body(
            messages=[
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": [{"type": "text", "text": "ok"}, blocks[1]]},
            ]
        )
    )
    assert plain.request.messages[1].provider_anthropic_blocks is None

    # Reasoning narrowed away (nothing left to verify) falls back to the
    # flattened emission rather than replaying thinking the rung dropped.
    stripped = assistant.model_copy(update={"provider_reasoning": ()})
    _role, fallback = anthropic_blocks(stripped)
    assert [block["type"] for block in fallback] == ["text", "tool_use", "tool_use"]


def test_thinking_history_blocks_ride_the_opaque_carrier_in_order() -> None:
    """Assistant reasoning history translates losslessly with byte-exact signatures."""
    decoded = decode_messages(
        _body(
            messages=[
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "private", "signature": "sig=="},
                        {"type": "redacted_thinking", "data": "opaque=="},
                        {"type": "text", "text": "done"},
                        {
                            "type": "tool_use",
                            "id": "call-1",
                            "name": "search",
                            "input": {},
                        },
                    ],
                },
            ]
        )
    )
    assistant = decoded.request.messages[1]
    assert assistant.content == "done"
    assert assistant.tool_calls[0].call_id == "call-1"
    blocks = assistant.provider_reasoning
    assert [block.kind for block in blocks] == ["thinking", "redacted_thinking"]
    thinking, redacted = blocks
    assert isinstance(thinking, ThinkingBlock)
    assert thinking.text == "private"
    assert thinking.signature == "sig=="
    assert isinstance(redacted, RedactedThinkingBlock)
    assert redacted.data == "opaque=="

    # A thinking-only assistant turn (cut off mid-thinking) is legal history.
    only = decode_messages(
        _body(
            messages=[
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "partial"}],
                },
                {"role": "user", "content": "continue"},
            ]
        )
    )
    assert only.request.messages[1].provider_reasoning[0].kind == "thinking"

    with pytest.raises(OpenAIProtocolError, match="only valid in assistant messages"):
        decode_messages(
            _body(
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "thinking", "thinking": "private"}],
                    }
                ]
            )
        )


@pytest.mark.parametrize(
    ("overrides", "param_fragment"),
    [
        ({"service_tier": "auto"}, "service_tier"),
        ({"container": "c"}, "container"),
        ({"unknown_field": 1}, "unknown_field"),
    ],
)
def test_unsupported_and_unknown_top_level_fields_are_rejected(
    overrides: JsonObject, param_fragment: str
) -> None:
    """Unsupported and unknown fields answer a loud field-specific 400."""
    with pytest.raises(OpenAIProtocolError) as excinfo:
        decode_messages(_body(**overrides))
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail.param == param_fragment


def test_top_k_is_preserved_for_route_specific_validation() -> None:
    """The official Messages top-k field reaches the shared route contract."""
    decoded = decode_messages(_body(top_k=5))
    assert decoded.request.top_k == 5


def test_missing_max_tokens_is_rejected_with_its_field() -> None:
    """max_tokens is required by the Anthropic protocol."""
    payload = _body()
    del payload["max_tokens"]
    with pytest.raises(OpenAIProtocolError) as excinfo:
        decode_messages(payload)
    assert excinfo.value.detail.param == "max_tokens"


def test_image_blocks_are_retained_in_caller_order() -> None:
    """An image block rides the canonical parts beside its text."""
    decoded = decode_messages(
        _body(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": _PNG_BASE64,
                            },
                        },
                    ],
                }
            ]
        )
    )
    message = decoded.request.messages[-1]
    assert message.content == "what is this"
    assert [part.kind for part in message.content_parts] == ["text", "image"]
    assert message.images[0].data == _PNG_BASE64


def test_a_cache_marker_on_an_image_block_is_retained() -> None:
    """A breakpoint the caller placed on the image reaches the wire."""
    decoded = decode_messages(
        _body(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": _PNG_BASE64,
                            },
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ]
        )
    )
    assert decoded.request.messages[-1].images[0].cache_control == {"type": "ephemeral"}


def test_an_empty_text_block_beside_an_image_never_re_emits() -> None:
    """An attachment's empty text block never reaches the Anthropic wire."""
    decoded = decode_messages(
        _body(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ""},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": _PNG_BASE64,
                            },
                        },
                        {"type": "text", "text": "read it", "cache_control": {"type": "ephemeral"}},
                    ],
                }
            ]
        )
    )
    message = decoded.request.messages[-1]
    assert [part.kind for part in message.content_parts] == ["image", "text"]
    _role, blocks = anthropic_blocks(message)
    assert blocks == [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": _PNG_BASE64},
        },
        {"type": "text", "text": "read it", "cache_control": {"type": "ephemeral"}},
    ]


def test_a_tool_result_image_re_emits_as_the_exact_block_run() -> None:
    """An Anthropic rung round-trips the tool screenshot losslessly."""
    decoded = decode_messages(
        _body(messages=_tool_result_image_messages(leading_text="tool said:"))
    )
    role, blocks = anthropic_blocks(decoded.request.messages[-1])
    assert role == "user"
    assert blocks == [
        {
            "type": "tool_result",
            "tool_use_id": "call-1",
            "content": [
                {"type": "text", "text": "tool said:"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": _PNG_BASE64,
                    },
                },
            ],
        }
    ]


def test_a_cache_marker_on_a_tool_result_text_block_round_trips() -> None:
    """A breakpoint on an inner text block re-emits with the block run.

    Claude Code marks the last block of recent user turns; in an agent loop
    that block can be a text sub-block inside an image-bearing tool_result,
    and losing it would silently un-cache the conversation prefix.
    """
    decoded = decode_messages(
        _body(
            messages=[
                {"role": "user", "content": "read the screenshot"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "call-1", "name": "computer", "input": {}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-1",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "tool said:",
                                    "cache_control": {"type": "ephemeral"},
                                },
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": _PNG_BASE64,
                                    },
                                },
                            ],
                        }
                    ],
                },
            ]
        )
    )
    _role, blocks = anthropic_blocks(decoded.request.messages[-1])
    assert blocks == [
        {
            "type": "tool_result",
            "tool_use_id": "call-1",
            "content": [
                {
                    "type": "text",
                    "text": "tool said:",
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": _PNG_BASE64,
                    },
                },
            ],
        }
    ]


def test_malformed_image_source_is_rejected() -> None:
    """An image the gateway cannot forward is rejected at its own path."""
    with pytest.raises(OpenAIProtocolError) as excinfo:
        decode_messages(
            _body(
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "image", "source": {"type": "base64", "data": "x"}}],
                    }
                ]
            )
        )
    assert excinfo.value.detail.param == "messages.0.content.0.source"


def test_document_block_inside_tool_result_is_rejected() -> None:
    """Nested unsupported blocks inside tool results are rejected loudly."""
    with pytest.raises(OpenAIProtocolError) as excinfo:
        decode_messages(
            _body(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "call-1",
                                "content": [{"type": "document", "source": {}}],
                            }
                        ],
                    }
                ]
            )
        )
    assert "document blocks are not supported" in excinfo.value.detail.message


def _tool_result_image_messages(*, leading_text: str | None = None) -> list[JsonObject]:
    """The owner-reported repro: text turn, tool_use, tool_result with an image."""
    inner: list[JsonObject] = []
    if leading_text is not None:
        inner.append({"type": "text", "text": leading_text})
    inner.append(
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": _PNG_BASE64},
        }
    )
    return [
        {"role": "user", "content": "read the screenshot"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call-1", "name": "computer", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": inner}],
        },
    ]


def test_image_blocks_inside_tool_results_ride_the_tool_message_parts() -> None:
    """A tool screenshot decodes losslessly instead of 400ing the session.

    Anthropic's real API accepts image sub-blocks in tool_result content, and
    Claude Code's Read-on-image and computer-use tools emit them routinely;
    the block is baked into history, so rejecting it wedges every later turn.
    """
    decoded = decode_messages(
        _body(messages=_tool_result_image_messages(leading_text="tool said:"))
    )
    tool_message = decoded.request.messages[-1]
    assert tool_message.role == "tool"
    assert tool_message.tool_call_id == "call-1"
    assert tool_message.content == "tool said:"
    assert [part.kind for part in tool_message.content_parts] == ["text", "image"]
    assert tool_message.images[0].data == _PNG_BASE64


def test_an_image_only_tool_result_decodes_with_empty_content() -> None:
    """The exact wedged-session repro: content is one bare image block."""
    decoded = decode_messages(_body(messages=_tool_result_image_messages()))
    tool_message = decoded.request.messages[-1]
    assert tool_message.role == "tool"
    assert tool_message.content == ""
    assert [part.kind for part in tool_message.content_parts] == ["image"]


def test_a_text_only_tool_result_retains_no_content_parts() -> None:
    """Existing text-only results serialize and digest exactly as before."""
    decoded = decode_messages(
        _body(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-1",
                            "content": [{"type": "text", "text": "plain"}],
                        }
                    ],
                }
            ]
        )
    )
    tool_message = decoded.request.messages[-1]
    assert tool_message.content == "plain"
    assert tool_message.content_parts == ()


def test_an_unknown_tool_result_sub_block_names_its_index_and_type() -> None:
    """A union miss must name the offending block, never the string arm.

    The misleading "content.str: Input should be a valid string" rendering
    sent an entire diagnosis chain toward the caller's request shape when the
    real problem was one unsupported sub-block in the list arm.
    """
    with pytest.raises(OpenAIProtocolError) as excinfo:
        decode_messages(
            _body(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "call-1",
                                "content": [{"type": "mystery"}],
                            }
                        ],
                    }
                ]
            )
        )
    assert excinfo.value.detail.param == "messages.0.content.0.content.0"
    assert "unsupported block type 'mystery'" in excinfo.value.detail.message
    assert ".str" not in (excinfo.value.detail.param or "")


def test_a_union_miss_never_reports_the_string_arm() -> None:
    """A structurally bad list block reports its own path, not content.str."""
    with pytest.raises(OpenAIProtocolError) as excinfo:
        decode_messages(
            _body(messages=[{"role": "user", "content": [{"type": "text", "text": 7}]}])
        )
    param = excinfo.value.detail.param or ""
    assert ".str" not in param
    assert param.startswith("messages.0.content.0")


_PDF_BASE64 = "JVBERi0xLjQKJSBtaW5pbWFsIHBkZgo="
"""One short PDF header, base64 encoded."""


def _pdf_block(data: str = _PDF_BASE64, **extra: JsonValue) -> JsonObject:
    """Build one base64 Anthropic PDF document block with optional extra fields."""
    block: JsonObject = {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": data},
    }
    block.update(extra)
    return block


def test_document_blocks_are_retained_in_caller_order_with_interleaved_text() -> None:
    """PDF blocks ride the canonical parts at their positions among the text."""
    decoded = decode_messages(
        _body(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "first: "},
                        _pdf_block(title="one.pdf"),
                        {"type": "text", "text": " second: "},
                        _pdf_block("JVBERi0xLjcK"),
                        {"type": "text", "text": " compare them"},
                    ],
                }
            ]
        )
    )
    message = decoded.request.messages[-1]
    assert message.content == "first:  second:  compare them"
    assert [part.kind for part in message.content_parts] == [
        "text",
        "document",
        "text",
        "document",
        "text",
    ]
    documents = decoded.request.documents
    assert [document.data for document in documents] == [_PDF_BASE64, "JVBERi0xLjcK"]
    assert [document.name for document in documents] == ["one.pdf", None]
    assert documents[0].media_type == "application/pdf"


def test_a_document_url_source_and_cache_marker_are_retained() -> None:
    """A remote document and a breakpoint placed on it both reach the canonical part."""
    decoded = decode_messages(
        _body(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {"type": "url", "url": "https://example.com/brief.pdf"},
                            "cache_control": {"type": "ephemeral"},
                            "citations": {"enabled": False},
                        },
                        {"type": "text", "text": "summarize"},
                    ],
                }
            ]
        )
    )
    document = decoded.request.documents[0]
    assert document.url == "https://example.com/brief.pdf"
    assert document.data is None
    assert document.cache_control == {"type": "ephemeral"}


def test_document_sent_once_survives_a_multi_turn_thread() -> None:
    """A PDF in an earlier user turn is retained when later turns reference it."""
    decoded = decode_messages(
        _body(
            messages=[
                {
                    "role": "user",
                    "content": [_pdf_block(), {"type": "text", "text": "what is the title"}],
                },
                {"role": "assistant", "content": "Minimal PDF."},
                {"role": "user", "content": "and the page count?"},
            ]
        )
    )
    assert [len(message.documents) for message in decoded.request.messages] == [1, 0, 0]
    assert decoded.request.messages[-1].content_parts == ()


@pytest.mark.parametrize(
    ("block", "param"),
    [
        (
            {
                "type": "document",
                "source": {"type": "base64", "media_type": "text/plain", "data": "aGk="},
            },
            "messages.0.content.0.source",
        ),
        (
            {
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": "!!"},
            },
            "messages.0.content.0.source",
        ),
        (
            {"type": "document", "source": {"type": "url", "url": "ftp://example.com/a.pdf"}},
            "messages.0.content.0.source",
        ),
        (_pdf_block(citations={"enabled": True}), "messages.0.content.0.citations"),
    ],
)
def test_unservable_document_blocks_are_rejected_at_their_path(
    block: JsonObject, param: str
) -> None:
    """A document the gateway cannot forward is rejected loudly, never dropped."""
    with pytest.raises(OpenAIProtocolError) as excinfo:
        decode_messages(_body(messages=[{"role": "user", "content": [block]}]))
    assert excinfo.value.detail.param == param


def test_files_api_document_sources_decode_to_anthropic_handles() -> None:
    """A ``file`` source becomes an Anthropic-scoped handle, never bytes or a URL."""
    decoded = decode_messages(
        _body(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "document", "source": {"type": "file", "file_id": "file_011abc"}}
                    ],
                }
            ]
        )
    )
    (document,) = decoded.request.documents
    assert document.handle is not None
    assert document.handle.provider == "anthropic"
    assert document.handle.reference == "file_011abc"
    assert document.data is None and document.url is None


def test_malformed_files_api_ids_are_rejected() -> None:
    """A ``file`` source whose id is not an Anthropic Files id fails closed."""
    with pytest.raises(OpenAIProtocolError):
        decode_messages(
            _body(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "document", "source": {"type": "file", "file_id": "file-1"}}
                        ],
                    }
                ]
            )
        )


def test_assistant_document_blocks_are_rejected() -> None:
    """Only a caller message may carry a document."""
    with pytest.raises(OpenAIProtocolError, match="only valid in user messages"):
        decode_messages(_body(messages=[{"role": "assistant", "content": [_pdf_block()]}]))


def test_too_many_documents_are_rejected() -> None:
    """The per-request document ceiling fails closed with the ceiling named."""
    with pytest.raises(OpenAIProtocolError, match="at most 5 documents"):
        decode_messages(
            _body(
                messages=[
                    {
                        "role": "user",
                        "content": [_pdf_block() for _ in range(MAXIMUM_DOCUMENTS_PER_REQUEST + 1)],
                    }
                ]
            )
        )


def test_role_misplaced_blocks_and_empty_content_are_rejected() -> None:
    """Blocks are validated against their legal roles and non-empty turns."""
    with pytest.raises(OpenAIProtocolError, match="only valid in assistant messages"):
        decode_messages(
            _body(
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "tool_use", "id": "call-1", "name": "n", "input": {}}],
                    }
                ]
            )
        )
    with pytest.raises(OpenAIProtocolError, match="only valid in user messages"):
        decode_messages(
            _body(
                messages=[
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_result", "tool_use_id": "call-1"}],
                    }
                ]
            )
        )
    with pytest.raises(OpenAIProtocolError, match="must not be empty"):
        decode_messages(_body(messages=[{"role": "user", "content": ""}]))
    with pytest.raises(OpenAIProtocolError, match="must contain text"):
        decode_messages(_body(messages=[{"role": "assistant", "content": []}]))


def test_tool_choice_forms_and_stop_sequence_validation() -> None:
    """Every tool-choice form normalizes; bad stop sequences are rejected."""
    assert decode_messages(_body(tool_choice={"type": "auto"})).request.tool_choice == "auto"
    assert decode_messages(_body(tool_choice={"type": "none"})).request.tool_choice == "none"
    required = decode_messages(
        _body(
            tool_choice={"type": "any"},
            tools=[{"name": "search", "input_schema": {}}],
        )
    )
    assert required.request.tool_choice == "required"
    with pytest.raises(OpenAIProtocolError, match="requires a name"):
        decode_messages(_body(tool_choice={"type": "tool"}))
    with pytest.raises(OpenAIProtocolError, match="non-empty"):
        decode_messages(_body(stop_sequences=[""]))


def test_invalid_json_shape_errors_carry_a_dotted_field_path() -> None:
    """Wire validation errors name the offending field path."""
    with pytest.raises(OpenAIProtocolError) as excinfo:
        decode_messages(_body(max_tokens=0))
    assert excinfo.value.detail.param == "max_tokens"
    with pytest.raises(OpenAIProtocolError) as excinfo:
        decode_messages(_body(messages=[]))
    assert excinfo.value.detail.param == "messages"


def test_tool_result_error_state_is_preserved_on_the_canonical_message() -> None:
    """is_error travels on the canonical tool message without touching digests."""
    decoded = decode_messages(
        _body(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-1",
                            "content": "boom",
                            "is_error": True,
                        }
                    ],
                }
            ]
        )
    )
    tool = decoded.request.messages[0]
    assert tool.role == "tool"
    assert tool.tool_is_error is True
    plain = decode_messages(
        _body(
            messages=[
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call-1"}]}
            ]
        )
    )
    assert plain.request.messages[0].tool_is_error is False


def test_context_management_is_carried_verbatim_and_shallow_validated() -> None:
    """Claude Code's context-editing config survives byte-for-byte.

    Production incident (real Claude Code CLI, 2026-08-29): the field was a
    conscious UNSUPPORTED and every default-configured session 400ed.
    Validation is deliberately shallow (an object) because the nested shape
    is an evolving provider beta the gateway forwards verbatim.
    """
    config: JsonObject = {
        "edits": [
            {
                "type": "clear_tool_uses_20250919",
                "trigger": {"type": "input_tokens", "value": 30000},
                "keep": {"type": "tool_uses", "value": 3},
            }
        ]
    }
    decoded = decode_messages(_body(context_management=config))
    assert decoded.request.context_management == config
    assert decode_messages(_body()).request.context_management is None

    with pytest.raises(OpenAIProtocolError) as raised:
        decode_messages(_body(context_management="clear"))
    assert raised.value.detail.param == "context_management"


def test_thinking_display_is_carried_verbatim() -> None:
    """The adaptive display disposition rides the verbatim thinking config
    (Claude Code sends {"type": "adaptive", "display": "omitted"} by
    default; accepted live without a beta, 2026-08-30)."""
    config: JsonObject = {"type": "adaptive", "display": "omitted"}
    decoded = decode_messages(_body(thinking=config))
    assert decoded.request.provider_thinking_config == config


def test_mid_conversation_system_turn_decodes_positionally() -> None:
    """A system message after conversation start keeps its role and order."""
    decoded = decode_messages(
        _body(
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "answer in uppercase",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
            ]
        )
    )
    assert [message.role for message in decoded.request.messages] == ["user", "system"]
    assert decoded.request.messages[1].content == "answer in uppercase"


def test_the_captured_claude_code_request_shape_decodes_losslessly() -> None:
    """Regression fixture: the field shapes real Claude Code (2.1.251) sends
    by default, trimmed from a live capture (2026-08-29). Every top-level
    field and block shape from the capture appears here."""
    decoded = decode_messages(
        {
            "model": "claude-fable-5",
            "max_tokens": 64000,
            "stream": True,
            "system": [
                {
                    "type": "text",
                    "text": "You are Claude Code.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "thinking": {"type": "adaptive", "display": "omitted"},
            "output_config": {"effort": "high"},
            "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
            "metadata": {"user_id": "device-hash-redacted"},
            "tools": [
                {
                    "name": "Bash",
                    "description": "Run a shell command",
                    "input_schema": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                }
            ],
            "messages": [
                {"role": "user", "content": "Run ls, then count the entries."},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "count", "signature": "sig=="},
                        {
                            "type": "tool_use",
                            "id": "toolu_01",
                            "name": "Bash",
                            "input": {"command": "ls"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_01",
                            "content": "file_a.txt",
                            "is_error": False,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "Available agent types trimmed.",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
            ],
        }
    )
    request = decoded.request
    assert [message.role for message in request.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "system",
    ]
    assert request.reasoning_effort == "high"
    assert request.provider_output_config == {"effort": "high"}
    assert request.provider_thinking_config == {"type": "adaptive", "display": "omitted"}
    assert request.context_management is not None
    assert request.metadata == {"user_id": "device-hash-redacted"}
    assert request.ignored_parameters == ()


def test_diagnostics_and_speed_are_carried_verbatim_and_shallow_validated() -> None:
    """Claude Code's conditional diagnostics and fast-mode fields decode.

    Production incident (real Claude Code CLI, 2026-08-29): ``diagnostics``
    was undecided and every diagnostics-carrying session 400ed with "The
    parameter 'diagnostics' is not supported". Both fields are accepted by
    the provider behind their beta headers (verified live 2026-08-30), so
    the gateway carries them verbatim; validation stays shallow because the
    shapes are evolving provider betas.
    """
    decoded = decode_messages(_body(diagnostics={"previous_message_id": None}, speed="fast"))
    assert decoded.request.diagnostics == {"previous_message_id": None}
    assert decoded.request.speed == "fast"
    assert decode_messages(_body()).request.diagnostics is None
    assert decode_messages(_body()).request.speed is None

    with pytest.raises(OpenAIProtocolError) as raised:
        decode_messages(_body(diagnostics="on"))
    assert raised.value.detail.param == "diagnostics"


def test_caller_beta_tokens_partition_into_allowlist_and_disclosures() -> None:
    """The caller anthropic-beta header forwards only allowlisted tokens.

    Claude Code activates the 1M context window with a caller-sent
    ``context-1m-2025-08-07`` token (captured live 2026-08-30); without
    forwarding it the provider serves 200K and long sessions fail. Every
    non-allowlisted token drops with a per-token disclosure, never a
    rejection and never a blind forward.
    """
    header = (
        "claude-code-20250219,context-1m-2025-08-07,interleaved-thinking-2025-05-14,"
        "thinking-token-count-2026-05-13,fallback-credit-2026-06-01"
    )
    decoded = decode_messages(_body(), anthropic_beta=header)
    assert decoded.request.provider_beta_tokens == (
        "context-1m-2025-08-07",
        "interleaved-thinking-2025-05-14",
    )
    assert decoded.request.ignored_parameters == (
        "anthropic-beta.claude-code-20250219",
        "anthropic-beta.thinking-token-count-2026-05-13",
        "anthropic-beta.fallback-credit-2026-06-01",
    )
    assert decode_messages(_body()).request.provider_beta_tokens == ()

    with pytest.raises(OpenAIProtocolError) as raised:
        decode_messages(_body(), anthropic_beta="bad\nvalue")
    assert raised.value.detail.param == "anthropic-beta"


def test_a_real_toolset_tool_description_over_8k_decodes() -> None:
    """Tool descriptions bound generously: a real Claude Code toolset
    carried a description past the earlier 8k cap and 400ed with
    "Invalid value for 'tools.1.description'" while the provider accepts
    40k-character descriptions live (verified 2026-08-30)."""
    tools = [
        {"name": "small", "description": "x", "input_schema": {"type": "object"}},
        {"name": "large", "description": "y" * 40_000, "input_schema": {"type": "object"}},
    ]
    decoded = decode_messages(_body(tools=tools))
    description = decoded.request.tools[1].description
    assert description is not None and len(description) == 40_000


def test_decode_accepts_the_live_eager_input_streaming_tool_shape() -> None:
    """Live-captured 2026-08-30: a production Claude Code session sent a tool
    carrying ``eager_input_streaming`` and got "Invalid value for
    'tools.0.eager_input_streaming'" while api.anthropic.com accepts the field
    bare (no beta header). The exact wire shape stays accepted."""
    decoded = decode_messages(
        _body(
            stream=True,
            tools=[
                {
                    "name": "Bash",
                    "description": "Executes a bash command and returns its output.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                    "eager_input_streaming": True,
                }
            ],
        )
    )
    tool = decoded.request.tools[0]
    assert tool.eager_input_streaming is True
    assert decoded.request.ignored_parameters == ()


def test_decode_carries_every_provider_native_tool_annotation() -> None:
    """The provider-native tool annotations land on the canonical tool
    (each accepted bare by the live API, verified 2026-08-30)."""
    decoded = decode_messages(
        _body(
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get weather.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    "eager_input_streaming": False,
                    "defer_loading": False,
                    "allowed_callers": ["code_execution_20260120"],
                    "input_examples": [{"city": "Paris"}],
                }
            ]
        )
    )
    tool = decoded.request.tools[0]
    assert tool.strict is True
    assert tool.cache_control == {"type": "ephemeral", "ttl": "1h"}
    assert tool.eager_input_streaming is False
    assert tool.defer_loading is False
    assert tool.allowed_callers == ("code_execution_20260120",)
    assert tool.input_examples == ({"city": "Paris"},)

    bare = decode_messages(
        _body(tools=[{"name": "get_weather", "input_schema": {"type": "object"}}])
    ).request.tools[0]
    assert bare.strict is False
    assert bare.cache_control is None
    assert bare.eager_input_streaming is None
    assert bare.defer_loading is None
    assert bare.allowed_callers is None
    assert bare.input_examples is None


def test_decode_carries_top_level_cache_control_and_inference_geo() -> None:
    """Top-level auto-caching and the inference region ride their carriers
    verbatim (both accepted bare by the live API, verified 2026-08-30)."""
    decoded = decode_messages(_body(cache_control={"type": "ephemeral"}, inference_geo="us"))
    assert decoded.request.provider_cache_control == {"type": "ephemeral"}
    assert decoded.request.inference_geo == "us"
    absent = decode_messages(_body()).request
    assert absent.provider_cache_control is None
    assert absent.inference_geo is None

    with pytest.raises(OpenAIProtocolError) as excinfo:
        decode_messages(_body(cache_control={"type": "persistent"}))
    assert excinfo.value.detail.param == "cache_control.type"


@pytest.mark.parametrize(
    "field",
    ["user_profile_id", "fallbacks", "fallback_credit_token", "betas"],
)
def test_route_identity_and_delegation_fields_stay_consciously_rejected(field: str) -> None:
    """Fallback model swaps, body-borne beta opt-ins, and third-party
    attribution are recorded rejections, each answered by its named 400."""
    with pytest.raises(OpenAIProtocolError) as excinfo:
        decode_messages(_body(**{field: "x"}))
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail.param == field


def test_validation_errors_state_the_expectation_not_only_the_field() -> None:
    """A strict-decode 400 names the field and what was expected there."""
    with pytest.raises(OpenAIProtocolError) as unknown:
        decode_messages(_body(tools=[{"name": "t", "input_schema": {}, "eager_streaming": True}]))
    assert unknown.value.detail.param == "tools.0.eager_streaming"
    assert "Unknown parameter 'tools.0.eager_streaming'" in unknown.value.detail.message

    with pytest.raises(OpenAIProtocolError) as invalid:
        decode_messages(_body(tools=[{"name": "t", "input_schema": {}, "strict": "maybe"}]))
    assert invalid.value.detail.param == "tools.0.strict"
    assert "Invalid value for 'tools.0.strict'" in invalid.value.detail.message
    assert "bool" in invalid.value.detail.message


def test_the_prod_failing_web_search_tool_shape_decodes() -> None:
    """The exact Claude Code WebSearch entry decodes and carries verbatim.

    Production incident (2026-08-31 class): the strict custom-tool model
    400d every session with WebSearch enabled because the server tool entry
    carries no input_schema.
    """
    server_entry: JsonObject = {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 8,
    }
    decoded = decode_messages(
        _body(
            tools=[
                {"name": "Bash", "description": "run", "input_schema": {"type": "object"}},
                server_entry,
            ],
            tool_choice={"type": "auto"},
        )
    )
    request = decoded.request
    assert tuple(tool.name for tool in request.tools) == ("Bash",)
    # The carried entry equals the raw payload object byte-for-byte.
    assert request.provider_server_tools == (server_entry,)


def test_every_verified_web_search_version_decodes() -> None:
    """All live-verified web_search versions pass the accept table."""
    for tool_type in ("web_search_20250305", "web_search_20260209", "web_search_20260318"):
        decoded = decode_messages(_body(tools=[{"type": tool_type, "name": "web_search"}]))
        entries = decoded.request.provider_server_tools
        assert len(entries) == 1 and entries[0]["type"] == tool_type


def test_unserved_server_tool_types_are_rejected_by_name() -> None:
    """A classified-but-unserved or unknown server tool type 400s loudly."""
    for tool_type in ("web_fetch_20250910", "code_execution_20250522", "someday_20990101"):
        with pytest.raises(OpenAIProtocolError) as error:
            decode_messages(_body(tools=[{"type": tool_type, "name": "t"}]))
        assert error.value.status_code == 400
        assert tool_type in error.value.detail.message
        assert "web_search_20250305" in error.value.detail.message


def test_malformed_server_tool_entries_are_rejected() -> None:
    """A non-pattern type or a nameless server entry stays a validation error."""
    for entry in (
        {"type": "Web Search!", "name": "web_search"},
        {"type": "web_search_20250305"},
    ):
        with pytest.raises(OpenAIProtocolError) as error:
            decode_messages(_body(tools=[entry]))
        assert error.value.status_code == 400


def _echoed_web_search_turn() -> list[JsonObject]:
    """The assistant blocks a served WebSearch turn echoes back (live shape)."""
    return [
        {
            "type": "server_tool_use",
            "id": "srvtoolu_fixture",
            "name": "web_search",
            "input": {"query": "current stable Python"},
        },
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_fixture",
            "content": [
                {
                    "type": "web_search_result",
                    "title": "Python versions",
                    "url": "https://www.python.org/doc/versions/",
                    "encrypted_content": "Et8QCioIExgC",
                    "page_age": "March 12, 2026",
                }
            ],
            "caller": {"type": "direct"},
        },
        {
            "citations": [
                {
                    "type": "web_search_result_location",
                    "cited_text": "Python 3.14.7, released on 5 August 2026",
                    "url": "https://www.python.org/doc/versions/",
                    "title": "Python versions",
                    "encrypted_index": "Eo8BCioIExgC",
                }
            ],
            "type": "text",
            "text": "The current stable Python version is 3.14.7.",
        },
    ]


def test_echoed_server_tool_turn_rides_the_verbatim_block_carrier() -> None:
    """A turn-2 echo decodes into ordered verbatim per-block messages.

    The echoed turn (server_tool_use, web_search_tool_result, cited text)
    must round-trip byte-faithfully: every block, extras and provider-issued
    encrypted payloads included, becomes one whole-message carrier at its
    position.
    """
    blocks = _echoed_web_search_turn()
    decoded = decode_messages(
        _body(
            messages=[
                {"role": "user", "content": "search please"},
                {"role": "assistant", "content": blocks},
                {"role": "user", "content": "thanks, just the version"},
            ],
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 2}],
        )
    )
    messages = decoded.request.messages
    assert [message.role for message in messages] == [
        "user",
        "assistant",
        "assistant",
        "assistant",
        "user",
    ]
    carried = [
        message.provider_anthropic_block
        for message in messages
        if message.provider_anthropic_block is not None
    ]
    assert carried == blocks
    assert messages[-1].content == "thanks, just the version"


def test_cited_text_splits_around_plain_assistant_text_in_order() -> None:
    """Plain text merges as content while cited text carries verbatim."""
    decoded = decode_messages(
        _body(
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me check."},
                        {
                            "type": "text",
                            "text": "It is 3.14.7.",
                            "citations": [{"type": "web_search_result_location"}],
                        },
                        {"type": "text", "text": "Anything else?"},
                    ],
                },
                {"role": "user", "content": "no"},
            ]
        )
    )
    assistant = decoded.request.messages[1:4]
    assert assistant[0].content == "Let me check."
    assert assistant[1].provider_anthropic_block == {
        "type": "text",
        "text": "It is 3.14.7.",
        "citations": [{"type": "web_search_result_location"}],
    }
    assert assistant[2].content == "Anything else?"


def test_uncited_citation_shapes_stay_on_the_plain_text_path() -> None:
    """The SDK accumulator's null and empty citations decode as plain text."""
    for citations in (None, []):
        decoded = decode_messages(
            _body(
                messages=[
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "ok", "citations": citations}],
                    },
                    {"role": "user", "content": "next"},
                ]
            )
        )
        assistant = decoded.request.messages[1]
        assert assistant.content == "ok"
        assert assistant.provider_anthropic_block is None


def test_server_tool_blocks_and_citations_are_assistant_only() -> None:
    """Server-tool output shapes in a user turn are rejected with the field."""
    for content in (
        [{"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}}],
        [{"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": []}],
        [{"type": "text", "text": "hi", "citations": [{"type": "char_location"}]}],
    ):
        with pytest.raises(OpenAIProtocolError) as error:
            decode_messages(_body(messages=[{"role": "user", "content": content}]))
        assert error.value.status_code == 400
        assert "assistant" in error.value.detail.message


def test_decode_carries_block_level_cache_markers_like_a_live_claude_code_turn() -> None:
    """P0 (captured live 2026-09-01): Claude Code marks two of its three
    system blocks and the last text block of the last user turn; agent loops
    also mark tool_result breakpoints. Flattening dropped every marker, so
    nothing through the gateway was ever cacheable (measured cache_read=0
    across whole sessions, ~10x input billing)."""
    decoded = decode_messages(
        _body(
            system=[
                {"type": "text", "text": "You are Claude Code."},
                {"type": "text", "text": "Short block.", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "Long env block.", "cache_control": {"type": "ephemeral"}},
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "context"},
                        {
                            "type": "text",
                            "text": "do the thing",
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "call-1", "name": "Bash", "input": {}}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-1",
                            "content": "ok",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
            ],
        )
    )
    system = decoded.request.messages[0]
    assert system.role == "system"
    assert system.content == "You are Claude Code.\n\nShort block.\n\nLong env block."
    assert system.provider_text_blocks == (
        {"type": "text", "text": "You are Claude Code."},
        {"type": "text", "text": "Short block.", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "Long env block.", "cache_control": {"type": "ephemeral"}},
    )
    user = decoded.request.messages[1]
    assert user.content == "contextdo the thing"
    assert user.provider_text_blocks == (
        {"type": "text", "text": "context"},
        {"type": "text", "text": "do the thing", "cache_control": {"type": "ephemeral"}},
    )
    tool = decoded.request.messages[3]
    assert tool.role == "tool"
    assert tool.cache_control == {"type": "ephemeral"}

    # A markerless request carries nothing: payloads stay byte-identical.
    plain = decode_messages(_body(system=[{"type": "text", "text": "You are terse."}])).request
    assert plain.messages[0].provider_text_blocks == ()
    assert plain.messages[1].provider_text_blocks == ()


def test_video_block_is_rejected_with_a_surface_hint() -> None:
    """The Messages wire defines no video block, so one is refused loudly."""
    with pytest.raises(OpenAIProtocolError) as excinfo:
        decode_messages(
            _body(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what happens"},
                            {
                                "type": "video",
                                "source": {"type": "url", "url": "https://example.com/a.mp4"},
                            },
                        ],
                    }
                ]
            )
        )
    assert "video blocks are not supported" in excinfo.value.detail.message


def test_audio_block_is_rejected_with_a_surface_hint() -> None:
    """The Messages wire defines no audio block, so one is refused by name, not dropped."""
    for body in (
        _body(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is said"},
                        {"type": "audio", "source": {"type": "base64", "data": "UklGRg=="}},
                    ],
                }
            ]
        ),
        _body(
            messages=[
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "t1", "name": "f", "input": {}}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [
                                {"type": "audio", "source": {"type": "base64", "data": "UklGRg=="}}
                            ],
                        }
                    ],
                },
            ]
        ),
    ):
        with pytest.raises(OpenAIProtocolError) as excinfo:
            decode_messages(body)
        assert "audio blocks are not supported" in excinfo.value.detail.message
        assert "input_audio" not in excinfo.value.detail.message
        assert "Chat Completions" in excinfo.value.detail.message


def test_a_screenshot_history_beyond_twenty_images_still_decodes() -> None:
    """A long agent session's image history clears validation and route shaping.

    Screenshots are baked into the transcript, so a per-request image ceiling
    the history can grow into wedges the session: the live incident hit
    "a request carries at most 20 images" once its screenshot history passed
    20, and every replay failed forever. The Anthropic API itself accepts 100
    images per request, so 21 must decode; beyond the API ceiling the named
    rejection stays (the provider would refuse anyway, less clearly).
    """
    image_block: dict[str, object] = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": _PNG_BASE64},
    }
    decoded = decode_messages(_body(messages=[{"role": "user", "content": [image_block] * 21}]))
    assert len(decoded.request.images) == 21

    with pytest.raises(OpenAIProtocolError, match="at most 100 images"):
        decode_messages(_body(messages=[{"role": "user", "content": [image_block] * 101}]))


def _openai_reasoning_profile() -> GatewayWireProfile:
    """Return one OpenAI Responses profile with the standard effort ladder."""
    return GatewayWireProfile(
        dialect="openai_responses",
        url="https://api.openai.com/v1/responses",
        model_id="gpt-5.6-sol",
        supports_reasoning=True,
        reasoning_wire_format="openai_responses",
        supported_reasoning_efforts=("none", "low", "medium", "high"),
    )


def test_a_claude_code_thinking_request_serves_on_an_openai_route() -> None:
    """The Messages thinking channel translates end to end, disclosed.

    Driven through the real /v1/messages decode surface and the admission
    sequence: route shaping still rejects the config by name, the coercion
    layer translates it, and the coerced request reaches the OpenAI payload
    as a reasoning effort.
    """
    import pytest as _pytest

    from exp.runtime.models.providers.capability_policy import coerce_generation_parameters
    from exp.runtime.models.providers.errors import ProviderParameterError
    from exp.runtime.models.providers.streaming_requests import (
        dialect_stream_payload,
        route_generation_parameter_requests,
    )

    decoded = decode_messages(
        _body(
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "enabled", "budget_tokens": 8192},
        )
    )
    profile = _openai_reasoning_profile()
    with _pytest.raises(ProviderParameterError, match="thinking"):
        route_generation_parameter_requests((profile,), decoded.request)

    coercion = coerce_generation_parameters((profile,), decoded.request)
    assert coercion is not None
    assert coercion.disclosures == ("thinking->reasoning_effort:medium",)
    _public, provider = route_generation_parameter_requests((profile,), coercion.request)
    payload = dialect_stream_payload(profile, provider)
    assert payload["reasoning"] == {"effort": "medium"}
    assert "thinking" not in payload


def test_a_failed_tool_result_serves_on_an_openai_route() -> None:
    """The exact live repro (is_error:true on a GPT route) now serves.

    Previously: "The parameter 'messages.content.is_error' is not supported
    by this model route", which 400-killed a Claude Code session the moment
    any tool call failed. The flag folds into the result text, disclosed.
    """
    from exp.runtime.models.providers.streaming_requests import (
        dialect_stream_payload,
        route_generation_parameter_requests,
    )

    decoded = decode_messages(
        _body(
            max_tokens=64,
            tools=[
                {
                    "name": "run",
                    "description": "run a command",
                    "input_schema": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                }
            ],
            messages=[
                {"role": "user", "content": "run false"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_01",
                            "name": "run",
                            "input": {"cmd": "false"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_01",
                            "content": "exit 1",
                            "is_error": True,
                        }
                    ],
                },
            ],
        )
    )
    profile = _openai_reasoning_profile()
    public, provider = route_generation_parameter_requests((profile,), decoded.request)

    assert "messages.content.is_error->content" in public.ignored_parameters
    payload = dialect_stream_payload(profile, provider)
    items = [item for item in cast("list[JsonObject]", payload["input"])]
    outputs = [item for item in items if item.get("type") == "function_call_output"]
    assert outputs == [
        {"type": "function_call_output", "call_id": "toolu_01", "output": "[tool error] exit 1"}
    ]


def test_the_claude_code_model_probe_serves_on_an_openai_route() -> None:
    """The exact post-/model probe (max_tokens: 1) rides the provider floor."""
    from exp.runtime.models.providers.streaming_requests import (
        dialect_stream_payload,
        route_generation_parameter_requests,
    )

    decoded = decode_messages(_body(max_tokens=1, messages=[{"role": "user", "content": "hi"}]))
    profile = _openai_reasoning_profile()
    public, provider = route_generation_parameter_requests((profile,), decoded.request)

    assert "max_tokens->16" in public.ignored_parameters
    payload = dialect_stream_payload(profile, provider)
    assert payload["max_output_tokens"] == 16


def test_decode_names_a_duplicate_tool_use_id_instead_of_crashing() -> None:
    """A canonical-contract violation in a replayed turn is a named 400.

    Two ``tool_use`` blocks sharing one id in a single assistant turn violate
    the canonical assistant-message contract during turn translation, after
    the wire models have already passed. That exception must map to the
    turn-specific protocol error every other invalid shape gets: before the
    mapping it escaped decode as an unclassified 500 whose "retry the
    request" guidance is wrong for caller-shaped input.
    """
    with pytest.raises(OpenAIProtocolError, match="messages.1.*unique") as rejected:
        decode_messages(
            _body(
                messages=[
                    {"role": "user", "content": "t"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "tool_use", "id": "dup", "name": "a", "input": {}},
                            {"type": "tool_use", "id": "dup", "name": "a", "input": {}},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "dup", "content": "x"},
                            {"type": "tool_result", "tool_use_id": "dup", "content": "y"},
                        ],
                    },
                ],
                tools=[{"name": "a", "description": "d", "input_schema": {"type": "object"}}],
            )
        )
    assert rejected.value.status_code == 400
    assert rejected.value.detail.param == "messages.1"


def test_forced_tool_choice_decodes_to_the_canonical_forced_forms() -> None:
    """``any`` and a named ``tool`` reach the canonical forced forms admission
    narrows and coerces on; ``auto`` and ``none`` stay open selectors."""
    tools = [{"name": "lookup", "input_schema": {"type": "object"}}]
    forced_any = decode_messages(_body(tool_choice={"type": "any"}, tools=tools))
    assert forced_any.request.tool_choice == "required"
    forced_named = decode_messages(
        _body(tool_choice={"type": "tool", "name": "lookup"}, tools=tools)
    )
    assert forced_named.request.tool_choice == GatewayNamedToolChoice(name="lookup")
    assert decode_messages(
        _body(tool_choice={"type": "auto"}, tools=tools)
    ).request.tool_choice == ("auto")
