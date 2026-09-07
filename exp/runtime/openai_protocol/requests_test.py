"""Tests for shared Chat Completions and Responses request decoding."""

from __future__ import annotations

import base64
from collections.abc import Callable
from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject, sha256_json
from exp.common.models.content import (
    GEMINI_FILE_URI_PREFIX,
    MAXIMUM_DOCUMENTS_PER_REQUEST,
    AudioContentPart,
    MediaHandle,
    VideoContentPart,
)
from exp.runtime.gateway.contracts import (
    EncryptedReasoningBlock,
    GatewayApiSurface,
    GatewayMessage,
    GatewayNamedToolChoice,
    GatewayRequest,
)
from exp.runtime.gateway.reasoning_carrier import FIREWORKS_REASONING_CONTENT_PREFIX
from exp.runtime.models.providers.streaming_requests import openai_responses_stream_payload
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.model_adapter import model_request
from exp.runtime.openai_protocol.requests import (
    DecodedGatewayRequest,
    decode_chat,
    decode_embeddings,
    decode_responses,
)


def test_chat_decoder_preserves_every_supported_semantic_field() -> None:
    """Chat conversion retains roles, raw tools, strict schema, controls, usage, and metadata."""
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {"role": "developer", "content": "Follow policy."},
                {"role": "user", "content": [{"type": "text", "text": "Call weather."}]},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-one",
                            "type": "function",
                            "function": {
                                "name": "weather",
                                "arguments": '{ "city" : "Zürich" }',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-one", "content": "sunny"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "description": "Read weather",
                        "parameters": {"type": "object"},
                        "strict": True,
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "weather"}},
            "parallel_tool_calls": True,
            "max_completion_tokens": 123,
            "stop": ["END", "STOP"],
            "temperature": 0.2,
            "top_p": 1,
            "reasoning_effort": "high",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {"type": "object"},
                    "strict": True,
                },
            },
            "stream": True,
            "stream_options": {"include_usage": True},
            "metadata": {"cohort": "test"},
        },
        idempotency_key="operation-one",
        client_request_id="operation-one",
    )

    request = decoded.request
    assert decoded.alias == "coding"
    assert request.surface == GatewayApiSurface.CHAT_COMPLETIONS
    assert tuple(message.role for message in request.messages) == (
        "developer",
        "user",
        "assistant",
        "tool",
    )
    assert request.messages[2].tool_calls[0].raw_arguments == '{ "city" : "Zürich" }'
    assert request.tools[0].strict
    assert isinstance(request.tool_choice, GatewayNamedToolChoice)
    assert request.tool_choice.name == "weather"
    assert request.maximum_output_tokens == 123
    assert request.maximum_output_tokens_parameter == "max_completion_tokens"
    assert request.stop == ("END", "STOP")
    assert request.temperature == 0.2
    assert request.top_p == 1.0
    assert request.reasoning_effort == "high"
    assert request.structured_text is not None and request.structured_text.strict
    assert request.include_usage
    assert request.metadata == {"cohort": "test"}


def test_chat_decoder_carries_json_object_as_its_own_mode() -> None:
    """response_format json_object rides json_object_output, not a permissive schema,
    and is not disclosed as a translation."""
    request = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "reply as json"}],
            "response_format": {"type": "json_object"},
        }
    ).request
    assert request.json_object_output is True
    assert request.structured_text is None
    assert request.ignored_parameters == ()


def test_chat_decoder_admits_sampling_penalties() -> None:
    """frequency_penalty/presence_penalty are admitted at the ingress (adapted per rung
    downstream) rather than rejected as unsupported fields."""
    request = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "frequency_penalty": 0.5,
            "presence_penalty": -0.25,
        }
    ).request
    assert request.frequency_penalty == 0.5
    assert request.presence_penalty == -0.25


def test_chat_legacy_max_tokens_reaches_native_responses_as_max_output_tokens() -> None:
    """A Chat request using legacy max_tokens serves a native Responses max_output_tokens.

    Chat clients (playground and agents) commonly send the legacy max_tokens field. On a
    direct OpenAI deployment the native Responses API rejects max_tokens and wants
    max_output_tokens, so the canonical request must translate the field and the native
    payload must never carry max_tokens.
    """
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 256,
            "stream": True,
        }
    )

    assert decoded.request.maximum_output_tokens == 256
    assert decoded.request.maximum_output_tokens_parameter == "max_tokens"
    payload = openai_responses_stream_payload(
        "gpt-fixture",
        decoded.request,
        supports_temperature=True,
        reasoning_effort=None,
    )
    assert payload["max_output_tokens"] == 256
    assert "max_tokens" not in payload


def test_responses_decoder_preserves_continuation_and_distinct_wire_shapes() -> None:
    """Responses conversion keeps instructions, item history, named tools, and structured text."""
    decoded = decode_responses(
        {
            "model": "coding",
            "instructions": "Use tools.",
            "input": [
                {"type": "message", "role": "user", "content": "Weather?"},
                {
                    "type": "function_call",
                    "call_id": "call-one",
                    "name": "weather",
                    "arguments": '{"city":"Paris"}',
                },
                {"type": "function_call_output", "call_id": "call-one", "output": "sunny"},
            ],
            "previous_response_id": "resp_previous",
            "tools": [
                {
                    "type": "function",
                    "name": "weather",
                    "parameters": {"type": "object"},
                    "strict": True,
                }
            ],
            "tool_choice": {"type": "function", "name": "weather"},
            "parallel_tool_calls": False,
            "max_output_tokens": 321,
            "temperature": 0.4,
            "reasoning": {
                "effort": "high",
                "generate_summary": "concise",
                "summary": "concise",
            },
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "schema": {"type": "object"},
                    "strict": True,
                }
            },
            "stream": True,
            "metadata": {"cohort": "test"},
        },
        client_request_id="operation-two",
    )

    request = decoded.request
    assert decoded.developer_messages_param == "instructions"
    assert request.surface == GatewayApiSurface.RESPONSES
    assert tuple(message.role for message in request.messages) == (
        "developer",
        "user",
        "assistant",
        "tool",
    )
    assert request.previous_response_id == "resp_previous"
    assert request.reasoning_effort == "high"
    assert request.reasoning_summary == "concise"
    assert request.reasoning_summary_parameters == (
        "reasoning.generate_summary",
        "reasoning.summary",
    )
    assert request.ignored_parameters == ()
    assert request.messages[2].tool_calls[0].raw_arguments == '{"city":"Paris"}'
    assert request.parallel_tool_calls is False
    assert request.maximum_output_tokens == 321
    assert request.maximum_output_tokens_parameter == "max_output_tokens"
    assert request.structured_text is not None
    assert request.client_request_id == "operation-two"


def test_responses_decoder_tracks_developer_input_origin() -> None:
    """Capability errors can identify an input developer role without inventing instructions."""
    decoded = decode_responses(
        {
            "model": "coding",
            "input": [
                {"type": "message", "role": "user", "content": "hello"},
                {"type": "message", "role": "developer", "content": "follow policy"},
            ],
        }
    )

    assert decoded.developer_messages_param == "input.1.role"


def test_responses_decoder_rejects_conflicting_reasoning_summary_aliases() -> None:
    """Current and deprecated summary selectors cannot request different outputs."""
    with pytest.raises(OpenAIProtocolError) as captured:
        decode_responses(
            {
                "model": "coding",
                "input": "hello",
                "reasoning": {"summary": "concise", "generate_summary": "detailed"},
            }
        )

    assert captured.value.detail.code == "invalid_parameter"


@pytest.mark.parametrize(
    ("decoder", "payload", "param"),
    (
        (
            decode_responses,
            {"model": "coding", "input": "x", "background": True},
            "background",
        ),
        (
            decode_chat,
            {"model": "coding", "messages": [{"role": "user", "content": "x"}], "future": 1},
            "future",
        ),
    ),
)
def test_unknown_and_excluded_fields_fail_with_exact_param(
    decoder: Callable[[JsonObject], DecodedGatewayRequest], payload: JsonObject, param: str
) -> None:
    """Closed manifests reject excluded and future SDK fields before dispatch."""
    with pytest.raises(OpenAIProtocolError) as captured:
        decoder(payload)
    assert captured.value.detail.code == "unsupported_parameter"
    assert captured.value.detail.param == param


def test_chat_decoder_accepts_echoed_assistant_message_with_empty_sdk_fields() -> None:
    """Assistant messages echoed verbatim from a prior gateway response must decode.

    The gateway's own Chat responses and official SDK message dumps carry
    refusal, annotations, audio, function_call, and a possibly null tool_calls
    key; a tool-call continuation sends that message back unchanged.
    """
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {"role": "user", "content": "Weather in Paris?"},
                {
                    "role": "assistant",
                    "content": None,
                    "refusal": None,
                    "annotations": [],
                    "audio": None,
                    "function_call": None,
                    "tool_calls": [
                        {
                            "id": "call-one",
                            "type": "function",
                            "function": {"name": "weather", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-one", "content": "sunny"},
                {
                    "role": "assistant",
                    "content": "It is sunny.",
                    "refusal": None,
                    "tool_calls": None,
                },
                {"role": "user", "content": "Thanks."},
            ],
        }
    )
    assert decoded.request.messages[1].tool_calls[0].name == "weather"
    assert decoded.request.messages[3].content == "It is sunny."
    assert decoded.request.messages[3].tool_calls == ()


def test_chat_decoder_accepts_a_verbatim_litellm_message_dump() -> None:
    """A LiteLLM ``Message.model_dump()`` echoed back on the next turn decodes.

    LiteLLM stamps ``provider_specific_fields`` (an object), plus null
    ``thinking_blocks``, ``reasoning_items`` and ``images``, on every assistant
    message; a Terminus-2 port that keeps the message object resends all of
    them (Akhara, 2026-09-05). The object is carried for disclosure and never
    forwarded; the empty forms decode like the SDK's own empty keys.
    """
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {"role": "user", "content": "Reply with a command."},
                {
                    "content": '{"command": "ls"}',
                    "role": "assistant",
                    "tool_calls": None,
                    "function_call": None,
                    "provider_specific_fields": {"refusal": None},
                    "reasoning_content": "The user wants a listing.",
                    "thinking_blocks": None,
                    "reasoning_items": None,
                    "annotations": None,
                    "audio": None,
                    "images": None,
                },
                {"role": "user", "content": "Output: a.txt"},
            ],
        }
    )
    echoed = decoded.request.messages[1]
    assert echoed.content == '{"command": "ls"}'
    assert echoed.provider_specific_fields == {"refusal": None}
    assert "provider_specific_fields" not in echoed.model_dump(mode="json")

    # An empty object is the common stamp and carries nothing to disclose.
    empty = decode_chat(
        {
            "model": "coding",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "x", "provider_specific_fields": {}},
                {"role": "user", "content": "next"},
            ],
        }
    )
    assert empty.request.messages[1].provider_specific_fields is None


def test_chat_decoder_preserves_only_a_gateway_issued_reasoning_carrier() -> None:
    """A Fireworks continuation stays encrypted until authorized admission."""
    deployment = base64.urlsafe_b64encode(b"fireworks-rung").rstrip(b"=").decode()
    envelope = base64.urlsafe_b64encode(b"x" * 32).rstrip(b"=").decode()
    carrier = f"{FIREWORKS_REASONING_CONTENT_PREFIX}{deployment}:{envelope}"
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {"role": "user", "content": "Use a tool"},
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": carrier,
                    "tool_calls": [
                        {
                            "id": "call-one",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-one", "content": "done"},
            ],
        }
    )

    block = decoded.request.messages[1].provider_reasoning[0]
    assert block.kind == "sealed_reasoning_content"
    assert block.carrier == carrier
    assert block.deployment_hint == "fireworks-rung"
    adapted = model_request(decoded.request)
    assert adapted.messages[1].assistant_action is not None
    assert "provider_reasoning" not in adapted.messages[1].assistant_action.model_dump()


def test_chat_decoder_rejects_duplicate_assistant_tool_call_ids() -> None:
    """Two active calls cannot share the result-linkage identity."""
    with pytest.raises(OpenAIProtocolError) as captured:
        decode_chat(
            {
                "model": "coding",
                "messages": [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-one",
                                "type": "function",
                                "function": {"name": "first", "arguments": "{}"},
                            },
                            {
                                "id": "call-one",
                                "type": "function",
                                "function": {"name": "second", "arguments": "{}"},
                            },
                        ],
                    }
                ],
            }
        )

    assert captured.value.detail.param == "messages.0"


def test_chat_decoder_accepts_plaintext_reasoning_as_exposed_history() -> None:
    """Plaintext ``reasoning_content`` decodes as caller-owned exposed history.

    An exposure-gated rung (Tencent/DeepSeek) returns plaintext reasoning on
    every non-tool turn; a Terminus/Harbor loop echoes it back verbatim. The
    decoder carries it as an ``exposed_reasoning_content`` block — route
    admission, not the decoder, decides which rungs may replay it.
    """
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {"role": "user", "content": "run ls"},
                {
                    "role": "assistant",
                    "content": '{"command": "ls"}',
                    "reasoning_content": "The user wants a directory listing.",
                },
                {"role": "user", "content": "a.txt b.txt"},
            ],
        }
    )
    block = decoded.request.messages[1].provider_reasoning[0]
    assert block.kind == "exposed_reasoning_content"
    assert block.content == "The user wants a directory listing."
    # A reasoning-only assistant turn (content null — an exposed rung's
    # length-cut thinking turn, echoed exactly as returned) decodes too.
    reasoning_only = decode_chat(
        {
            "model": "coding",
            "messages": [
                {"role": "user", "content": "think"},
                {"role": "assistant", "content": None, "reasoning_content": "ran out of room"},
                {"role": "user", "content": "continue"},
            ],
        }
    )
    assert reasoning_only.request.messages[1].content is None
    assert (
        reasoning_only.request.messages[1].provider_reasoning[0].kind == "exposed_reasoning_content"
    )
    # Plaintext on a tool-call turn is caller-owned history too: AI-SDK
    # clients re-serialize a reasoning part onto the same assistant message
    # as its tool calls, and exposure-gated providers emit reasoning_content
    # on tool turns. It decodes like any other plaintext (previously a 400
    # that wedged every cross-model session); the sealed-carrier bond keeps
    # its strict path for text presented AS a carrier.
    tool_turn = decode_chat(
        {
            "model": "coding",
            "messages": [
                {"role": "user", "content": "look it up"},
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "plaintext on a tool turn",
                    "tool_calls": [
                        {
                            "id": "call-one",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-one", "content": "done"},
            ],
        }
    )
    tool_turn_message = tool_turn.request.messages[1]
    assert tool_turn_message.provider_reasoning[0].kind == "exposed_reasoning_content"
    assert tool_turn_message.tool_calls[0].name == "lookup"
    # An empty string is not reasoning; it names its field.
    with pytest.raises(OpenAIProtocolError) as raised:
        decode_chat(
            {
                "model": "coding",
                "messages": [{"role": "assistant", "content": "x", "reasoning_content": ""}],
            }
        )
    assert raised.value.detail.param == "messages.0.reasoning_content"


@pytest.mark.parametrize(
    "reasoning_content",
    (
        FIREWORKS_REASONING_CONTENT_PREFIX,
        f"{FIREWORKS_REASONING_CONTENT_PREFIX}not-base64:payload",
    ),
)
def test_chat_decoder_rejects_unbound_or_malformed_reasoning_content(
    reasoning_content: str,
) -> None:
    """A value under a known carrier prefix must parse as that carrier."""
    with pytest.raises(OpenAIProtocolError) as raised:
        decode_chat(
            {
                "model": "coding",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "x",
                        "reasoning_content": reasoning_content,
                    }
                ],
            }
        )
    assert raised.value.detail.param == "messages.0.reasoning_content"


def test_chat_decoder_still_rejects_populated_unsupported_message_fields() -> None:
    """A populated refusal, annotation, or LiteLLM carrier in history stays rejected."""
    for extra in (
        {"refusal": "no"},
        {"annotations": [{"type": "url_citation"}]},
        {"thinking_blocks": [{"type": "thinking", "thinking": "x", "signature": "y"}]},
        {"reasoning_items": [{"type": "reasoning"}]},
        {"images": [{"image_url": {"url": "https://example.test/a.png"}}]},
        {"provider_specific_fields": "not-an-object"},
    ):
        with pytest.raises(OpenAIProtocolError) as captured:
            decode_chat(
                {
                    "model": "coding",
                    "messages": [{"role": "assistant", "content": "x", **extra}],
                }
            )
        param = captured.value.detail.param
        assert param is not None and param.startswith("messages.0.")


def test_invalid_tool_arguments_and_divergent_operation_headers_are_specific() -> None:
    """Malformed history names its field; independent identity headers both decode."""
    with pytest.raises(OpenAIProtocolError) as arguments:
        decode_chat(
            {
                "model": "coding",
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call-one",
                                "type": "function",
                                "function": {"name": "tool", "arguments": "{"},
                            }
                        ],
                    }
                ],
            }
        )
    assert arguments.value.detail.param == "messages.0.tool_calls.0.function.arguments"

    # Idempotency-Key names one retriable operation; X-Client-Request-Id is
    # session correlation identity (Codex sends its session id on every
    # request of a session), so divergent values decode side by side.
    decoded = decode_responses(
        {"model": "coding", "input": "x"},
        idempotency_key="one",
        client_request_id="two",
    )
    assert decoded.request.idempotency_key == "one"
    assert decoded.request.client_request_id == "two"


def _assert_no_cache_control(decoded: DecodedGatewayRequest) -> None:
    """Require cache_control to be absent from canonical and provider-bound messages."""
    for message in decoded.request.messages:
        assert "cache_control" not in message.model_dump(mode="json")
    adapted = model_request(decoded.request)
    for message in adapted.messages:
        assert "cache_control" not in message.model_dump(mode="json")


def test_chat_decoder_drops_opencode_message_cache_control() -> None:
    """OpenCode Chat Completions annotate messages with Anthropic cache_control.

    The live failure is Invalid value for 'messages.0.cache_control' because the
    closed Chat wire model forbids that nested field before routing. Supported
    ephemeral forms must decode and never reach the canonical or provider-bound
    message.
    """
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "system",
                    "content": "You are concise.",
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "role": "user",
                    "content": "hello",
                    "cache_control": {"type": "ephemeral", "ttl": "5m"},
                },
                {
                    "role": "assistant",
                    "content": "hi",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                },
                {"role": "user", "content": "again", "cache_control": None},
            ],
            "top_p": 1,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    )

    assert tuple(message.content for message in decoded.request.messages) == (
        "You are concise.",
        "hello",
        "hi",
        "again",
    )
    assert decoded.request.top_p == 1.0
    assert decoded.request.stream
    assert decoded.request.include_usage
    _assert_no_cache_control(decoded)


def test_chat_decoder_drops_opencode_text_part_cache_control() -> None:
    """OpenCode openai-compatible conversion can put cache_control on text parts.

    applyCaching marks the last content part, and @ai-sdk/openai-compatible
    keeps that annotation on the text block when the user message has more
    than one part.
    """
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "prefix "},
                        {
                            "type": "text",
                            "text": "cached suffix",
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                }
            ],
        }
    )

    assert decoded.request.messages[0].content == "prefix cached suffix"
    _assert_no_cache_control(decoded)


@pytest.mark.parametrize(
    "cache_control",
    (
        {"type": "persistent"},
        {"type": "ephemeral", "ttl": "2h"},
        {"type": "ephemeral", "ttl": None},
        {"type": "ephemeral", "extra": True},
        "ephemeral",
        1,
        [],
    ),
)
def test_chat_decoder_rejects_malformed_message_cache_control(cache_control: object) -> None:
    """Unsupported cache_control shapes fail at messages.<index>.cache_control."""
    with pytest.raises(OpenAIProtocolError) as captured:
        decode_chat(
            {
                "model": "coding",
                "messages": [{"role": "user", "content": "hello", "cache_control": cache_control}],
            }
        )
    assert captured.value.detail.code == "invalid_parameter"
    assert captured.value.detail.param == "messages.0.cache_control"
    assert captured.value.detail.message == "Invalid value for 'messages.0.cache_control'."


def test_chat_decoder_rejects_malformed_text_part_cache_control() -> None:
    """Unsupported text-part cache_control stays a field-specific invalid value."""
    with pytest.raises(OpenAIProtocolError) as captured:
        decode_chat(
            {
                "model": "coding",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "hello",
                                "cache_control": {"type": "persistent"},
                            }
                        ],
                    }
                ],
            }
        )
    assert captured.value.detail.code == "invalid_parameter"
    assert captured.value.detail.param == "messages.0.content.0.cache_control"


def test_chat_decoder_still_rejects_unknown_nested_message_fields() -> None:
    """Dropping cache_control must not weaken unrelated unknown nested fields."""
    with pytest.raises(OpenAIProtocolError) as captured:
        decode_chat(
            {
                "model": "coding",
                "messages": [
                    {
                        "role": "user",
                        "content": "hello",
                        "cache_control": {"type": "ephemeral"},
                        "providerOptions": {},
                    }
                ],
            }
        )
    param = captured.value.detail.param
    assert param == "messages.0.providerOptions"


def test_chat_decoder_still_rejects_unknown_text_part_fields_with_cache_control() -> None:
    """A valid text-part cache_control drop still rejects other extra part keys."""
    with pytest.raises(OpenAIProtocolError) as captured:
        decode_chat(
            {
                "model": "coding",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "hello",
                                "cache_control": {"type": "ephemeral"},
                                "future": 1,
                            }
                        ],
                    }
                ],
            }
        )
    param = captured.value.detail.param
    assert param is not None and param.startswith("messages.0.content")


def test_chat_decoder_accepts_opencode_nucleus_and_usage_stream_shape() -> None:
    """OpenCode Chat Completions send top_p=1 with streamed usage and must decode losslessly."""
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "hello"}],
            "top_p": 1,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    )

    assert decoded.request.top_p == 1.0
    assert decoded.request.stream
    assert decoded.request.include_usage


def test_chat_decoder_rejects_out_of_range_top_p() -> None:
    """Nucleus sampling stays inside the official [0, 1] interval."""
    with pytest.raises(OpenAIProtocolError) as captured:
        decode_chat(
            {
                "model": "coding",
                "messages": [{"role": "user", "content": "hello"}],
                "top_p": 1.5,
            }
        )
    assert captured.value.detail.code == "invalid_parameter"
    assert captured.value.detail.param == "top_p"


def test_responses_decoder_preserves_top_p() -> None:
    """Responses accepts the provider-supported nucleus-sampling control."""
    decoded = decode_responses({"model": "coding", "input": "hello", "top_p": 1})
    assert decoded.request.top_p == 1


def test_chat_decoder_rejects_unprojectable_top_logprobs() -> None:
    """Alternate-token probability output is rejected at the public boundary."""
    with pytest.raises(OpenAIProtocolError) as raised:
        decode_chat(
            {
                "model": "coding",
                "messages": [{"role": "user", "content": "hello"}],
                "logprobs": True,
                "top_logprobs": 5,
            }
        )
    assert raised.value.detail.code == "unsupported_parameter"
    assert raised.value.detail.param == "top_logprobs"


def test_chat_decoder_accepts_store_false_opt_out() -> None:
    """OpenAI-style agents hardcode store:false and must not be rejected.

    The gateway never retains Chat output, so store:false is a satisfied no-op
    that decodes losslessly (it is not forwarded onto the canonical request).
    """
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "hello"}],
            "store": False,
        }
    )
    assert decoded.request.maximum_output_tokens is None
    assert len(decoded.request.messages) == 1


def test_chat_decoder_rejects_store_true_retention_request() -> None:
    """store:true asks the gateway to retain output, which it never does."""
    with pytest.raises(OpenAIProtocolError) as captured:
        decode_chat(
            {
                "model": "coding",
                "messages": [{"role": "user", "content": "hello"}],
                "store": True,
            }
        )
    assert captured.value.detail.code == "invalid_parameter"
    assert captured.value.detail.param == "store"


def test_chat_decoder_preserves_logprobs_for_route_validation() -> None:
    """The route gate distinguishes a semantic true request from a false no-op."""
    for value in (True, False):
        decoded = decode_chat(
            {
                "model": "coding",
                "messages": [{"role": "user", "content": "hello"}],
                "logprobs": value,
            }
        )
        assert decoded.request.logprobs is value


def test_chat_decoder_preserves_gateway_top_k_extension() -> None:
    """The provider-neutral top-k extension survives official SDK validation."""
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "hello"}],
            "top_k": 40,
        }
    )
    assert decoded.request.top_k == 40


def test_empty_responses_input_is_a_public_protocol_error() -> None:
    """Canonical validation failures do not leak internal Pydantic exceptions."""
    with pytest.raises(OpenAIProtocolError) as captured:
        decode_responses({"model": "coding", "input": []})
    assert captured.value.detail.code == "invalid_parameter"
    assert captured.value.detail.param == "messages"


def test_chat_decoder_captures_end_user_attribution_and_cache_hint() -> None:
    """safety_identifier/user/prompt_cache_key are captured; the label prefers safety_identifier."""
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "safety_identifier": "sha256:abc",
            "user": "legacy-user",
            "prompt_cache_key": "prompt_v1:sess",
        }
    )
    request = decoded.request
    assert request.safety_identifier == "sha256:abc"
    assert request.user == "legacy-user"
    assert request.prompt_cache_key == "prompt_v1:sess"
    # The current safety_identifier wins over the deprecated user for attribution.
    assert request.attribution_label == "sha256:abc"


def test_chat_attribution_label_falls_back_to_deprecated_user() -> None:
    """With no safety_identifier, the deprecated user field still labels the request."""
    decoded = decode_chat(
        {"model": "coding", "messages": [{"role": "user", "content": "hi"}], "user": "u-1"}
    )
    assert decoded.request.safety_identifier is None
    assert decoded.request.attribution_label == "u-1"


def test_prompt_cache_key_is_never_an_attribution_label() -> None:
    """prompt_cache_key is a cache-routing hint, not an end-user identity."""
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "prompt_cache_key": "pck",
        }
    )
    assert decoded.request.prompt_cache_key == "pck"
    assert decoded.request.attribution_label is None


def test_responses_decoder_captures_end_user_attribution() -> None:
    """The Responses surface captures the same attribution/cache fields as Chat."""
    decoded = decode_responses(
        {
            "model": "coding",
            "input": "hi",
            "safety_identifier": "sid-9",
            "prompt_cache_key": "pck-1",
        }
    )
    request = decoded.request
    assert request.safety_identifier == "sid-9"
    assert request.prompt_cache_key == "pck-1"
    assert request.attribution_label == "sid-9"


def test_responses_decoder_accepts_the_codex_request_shape() -> None:
    """store:false, include, ultra effort, and replayed reasoning all decode."""
    decoded = decode_responses(
        {
            "model": "coding",
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"effort": "ultra", "summary": "auto"},
            "prompt_cache_key": "codex-session-1",
            "input": [
                {"type": "message", "role": "user", "content": "run the tool"},
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "planned"}],
                    "encrypted_content": "blob==",
                },
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call-1",
                    "name": "search",
                    "arguments": "{}",
                },
                {
                    "type": "reasoning",
                    "id": "rs_2",
                    "summary": [],
                    "encrypted_content": "second-blob==",
                },
                {"type": "function_call_output", "call_id": "call-1", "output": "found"},
            ],
        }
    )
    request = decoded.request
    assert request.response_store is False
    assert request.include_encrypted_reasoning is True
    assert request.reasoning_effort == "ultra"
    assert request.reasoning_summary == "auto"
    assert request.prompt_cache_key == "codex-session-1"
    assistant = request.messages[1]
    assert assistant.role == "assistant"
    assert assistant.tool_calls[0].call_id == "call-1"
    blocks = assistant.provider_reasoning
    assert len(blocks) == 1
    assert blocks[0].kind == "encrypted_reasoning"
    assert blocks[0].id == "rs_1"
    assert blocks[0].encrypted_content == "blob=="
    trailing_block = request.messages[2].provider_reasoning[0]
    assert trailing_block.kind == "encrypted_reasoning"
    assert trailing_block.id == "rs_2"
    assert trailing_block.encrypted_content == "second-blob=="
    payload = openai_responses_stream_payload(
        "gpt-fixture",
        request,
        supports_temperature=False,
        supports_reasoning=True,
    )
    payload_input = cast(list[JsonObject], payload["input"])
    assert payload_input[1:] == [
        {
            "type": "reasoning",
            "summary": [],
            "encrypted_content": "blob==",
        },
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call-1",
            "name": "search",
            "arguments": "{}",
        },
        {
            "type": "reasoning",
            "summary": [],
            "encrypted_content": "second-blob==",
        },
        {"type": "function_call_output", "call_id": "call-1", "output": "found"},
    ]


def test_responses_decoder_rejects_reasoning_without_item_id() -> None:
    """Opaque reasoning replay requires the provider-issued item identity."""
    with pytest.raises(OpenAIProtocolError) as raised:
        decode_responses(
            {
                "model": "coding",
                "input": [{"type": "reasoning", "summary": [], "encrypted_content": "blob=="}],
            }
        )

    assert raised.value.status_code == 400
    assert raised.value.detail.param == "input.0.id"


def test_responses_decoder_replays_output_message_in_provider_order() -> None:
    """A stateless output transcript keeps reasoning, message, and call order."""
    decoded = decode_responses(
        {
            "model": "coding",
            "input": [
                {
                    "type": "reasoning",
                    "id": "rs_0",
                    "status": "completed",
                    "summary": [],
                    "encrypted_content": "opaque",
                },
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "I will look that up.",
                            "annotations": [],
                            "logprobs": [],
                        }
                    ],
                },
                {
                    "type": "function_call",
                    "id": "fc_2",
                    "call_id": "call-2",
                    "name": "lookup",
                    "arguments": '{ "q" : "x" }',
                    "status": "completed",
                },
                {"type": "function_call_output", "call_id": "call-2", "output": "found"},
            ],
        }
    )
    payload = openai_responses_stream_payload(
        "gpt-fixture",
        decoded.request,
        supports_temperature=False,
        supports_reasoning=True,
    )
    payload_input = cast(list[JsonObject], payload["input"])
    assert [(item["type"], item.get("id")) for item in payload_input[:3]] == [
        ("reasoning", None),
        ("message", "msg_1"),
        ("function_call", "fc_2"),
    ]
    message_content = cast(list[JsonObject], payload_input[1]["content"])
    assert message_content[0]["text"] == "I will look that up."
    # A replayed reasoning item never carries status (the provider rejects
    # it: "Unknown parameter", verified live 2026-08-29); other item types
    # keep it.
    assert "status" not in payload_input[0]
    assert payload_input[2]["arguments"] == '{ "q" : "x" }'
    assert payload_input[2]["status"] == "completed"


def test_responses_decoder_preserves_multiple_official_output_message_phases() -> None:
    """Official SDK output messages keep distinct identity, status, phase, and order."""
    from openai.types.responses.response_output_message import ResponseOutputMessage

    commentary = ResponseOutputMessage.model_validate(
        {
            "type": "message",
            "id": "msg_commentary",
            "role": "assistant",
            "status": "incomplete",
            "phase": "commentary",
            "content": [
                {
                    "type": "output_text",
                    "text": "I am checking.",
                    "annotations": [],
                }
            ],
        }
    ).model_dump(mode="json", exclude_none=True)
    final = ResponseOutputMessage.model_validate(
        {
            "type": "message",
            "id": "msg_final",
            "role": "assistant",
            "status": "completed",
            "phase": "final_answer",
            "content": [
                {
                    "type": "output_text",
                    "text": "Done.",
                    "annotations": [],
                }
            ],
        }
    ).model_dump(mode="json", exclude_none=True)

    decoded = decode_responses(
        {
            "model": "coding",
            "input": [
                commentary,
                {
                    "type": "function_call",
                    "call_id": "call-between",
                    "name": "lookup",
                    "arguments": "{}",
                    "status": "incomplete",
                },
                final,
            ],
        }
    )

    assistant_messages = tuple(
        message for message in decoded.request.messages if message.role == "assistant"
    )
    assert [message.provider_item_id for message in assistant_messages] == [
        "msg_commentary",
        None,
        "msg_final",
    ]
    assert [message.provider_phase for message in assistant_messages] == [
        "commentary",
        None,
        "final_answer",
    ]
    call = assistant_messages[1].tool_calls[0]
    assert call.call_id == "call-between"
    assert call.provider_item_id is None
    assert call.provider_output_index == 1
    assert call.provider_status == "incomplete"

    payload = openai_responses_stream_payload(
        "gpt-fixture",
        decoded.request,
        supports_temperature=False,
        supports_reasoning=True,
    )
    replay = cast(list[JsonObject], payload["input"])
    assert [(item["type"], item.get("id")) for item in replay] == [
        ("message", "msg_commentary"),
        ("function_call", None),
        ("message", "msg_final"),
    ]
    assert replay[0]["phase"] == "commentary"
    assert replay[0]["status"] == "incomplete"
    assert replay[1]["call_id"] == "call-between"
    assert replay[1]["status"] == "incomplete"
    assert replay[2]["phase"] == "final_answer"


@pytest.mark.parametrize(
    ("item", "param"),
    (
        (
            {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": ""},
            "input.0.encrypted_content",
        ),
        (
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "",
                "arguments": "{}",
            },
            "input.0.name",
        ),
    ),
)
def test_responses_decoder_reports_the_specific_malformed_output_item_field(
    item: JsonObject,
    param: str,
) -> None:
    """Union validation never leaks implementation branches such as input.str."""
    with pytest.raises(OpenAIProtocolError) as raised:
        decode_responses({"model": "coding", "input": [item]})

    assert raised.value.status_code == 400
    assert raised.value.detail.param == param


def test_responses_decoder_orders_function_calls_without_optional_item_ids() -> None:
    """A legacy ID-less call keeps its provider output index beside identified calls."""
    decoded = decode_responses(
        {
            "model": "coding",
            "input": [
                {
                    "type": "reasoning",
                    "id": "rs_0",
                    "summary": [],
                    "encrypted_content": "opaque",
                },
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "first",
                    "arguments": '{ "position" : 1 }',
                },
                {
                    "type": "function_call",
                    "id": "fc_2",
                    "call_id": "call-2",
                    "name": "second",
                    "arguments": '{"position":2}',
                },
            ],
        }
    )
    calls = decoded.request.messages[0].tool_calls
    assert [(call.provider_item_id, call.provider_output_index) for call in calls] == [
        (None, 1),
        ("fc_2", 2),
    ]
    payload = openai_responses_stream_payload(
        "gpt-fixture",
        decoded.request,
        supports_temperature=False,
        supports_reasoning=True,
    )
    payload_input = cast(list[JsonObject], payload["input"])
    assert [(item["type"], item.get("id")) for item in payload_input] == [
        # The reasoning id never crosses upstream: encrypted_content is
        # cryptographically bound to the provider's original item id, and an
        # id-less item verifies against the id embedded in the payload.
        ("reasoning", None),
        ("function_call", None),
        ("function_call", "fc_2"),
    ]
    assert [item["arguments"] for item in payload_input[1:]] == [
        '{ "position" : 1 }',
        '{"position":2}',
    ]


@pytest.mark.parametrize("reasoning_first", [True, False])
def test_responses_decoder_groups_fireworks_carrier_with_all_tool_calls(
    reasoning_first: bool,
) -> None:
    """One carrier and contiguous calls reconstruct their exact assistant turn in any order."""
    deployment = base64.urlsafe_b64encode(b"fireworks-rung").rstrip(b"=").decode()
    envelope = base64.urlsafe_b64encode(b"x" * 32).rstrip(b"=").decode()
    carrier = f"{FIREWORKS_REASONING_CONTENT_PREFIX}{deployment}:{envelope}"

    reasoning = {
        "type": "reasoning",
        "id": "rs_fireworks",
        "summary": [],
        "encrypted_content": carrier,
    }
    message = {
        "type": "message",
        "role": "assistant",
        "content": "I will check.",
    }
    assistant_items = [reasoning, message] if reasoning_first else [message, reasoning]
    decoded = decode_responses(
        {
            "model": "coding",
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "input": [
                *assistant_items,
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "first",
                    "arguments": "{}",
                },
                {
                    "type": "function_call",
                    "call_id": "call-2",
                    "name": "second",
                    "arguments": '{"value":2}',
                },
                {"type": "function_call_output", "call_id": "call-1", "output": "one"},
                {"type": "function_call_output", "call_id": "call-2", "output": "two"},
            ],
        }
    )

    assistant = decoded.request.messages[0]
    assert assistant.content == "I will check."
    assert tuple(call.call_id for call in assistant.tool_calls) == ("call-1", "call-2")
    assert assistant.provider_reasoning[0].kind == "sealed_reasoning_content"
    assert assistant.provider_reasoning[0].carrier == carrier


def test_responses_decoder_rejects_malformed_gateway_carrier() -> None:
    """A carrier-prefixed item cannot fall back to native opaque replay."""
    with pytest.raises(OpenAIProtocolError) as raised:
        decode_responses(
            {
                "model": "coding",
                "input": [
                    {
                        "type": "reasoning",
                        "id": "rs_malformed",
                        "summary": [],
                        "encrypted_content": f"{FIREWORKS_REASONING_CONTENT_PREFIX}broken",
                    }
                ],
            }
        )

    assert raised.value.detail.param == "input.0.encrypted_content"


def test_responses_decoder_keeps_orphaned_reasoning_as_its_own_turn() -> None:
    """Trailing reasoning with no assistant successor stays a standalone turn."""
    decoded = decode_responses(
        {
            "model": "coding",
            "input": [
                {"type": "message", "role": "user", "content": "go"},
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [],
                    "encrypted_content": "blob==",
                },
            ],
        }
    )
    trailing = decoded.request.messages[-1]
    assert trailing.role == "assistant"
    assert trailing.content is None
    block = trailing.provider_reasoning[0]
    assert block.kind == "encrypted_reasoning"
    assert block.encrypted_content == "blob=="


def test_responses_decoder_rejects_unknown_include_paths() -> None:
    """Only the encrypted reasoning include selector is honored."""
    with pytest.raises(OpenAIProtocolError) as raised:
        decode_responses(
            {
                "model": "coding",
                "include": ["message.output_text.logprobs"],
                "input": "hi",
            }
        )
    assert raised.value.detail.param == "include"

    # An id-only reasoning item (store=true replay, encrypted_content is
    # SDK-optional) is no longer a 400: it carries verbatim to the native
    # Responses wire and the provider judges resolvability.
    decoded = decode_responses(
        {
            "model": "coding",
            "input": [{"type": "reasoning", "id": "rs_1", "summary": []}],
        }
    )
    native = decoded.request.messages[-1].provider_native_item
    assert native == {"type": "reasoning", "id": "rs_1", "summary": []}


def test_chat_decoder_accepts_the_ultra_reasoning_effort() -> None:
    """The wire model owns effort validation ahead of the installed SDK literal."""
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "ultra",
        }
    )
    assert decoded.request.reasoning_effort == "ultra"

    with pytest.raises(OpenAIProtocolError) as raised:
        decode_chat(
            {
                "model": "coding",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "extreme",
            }
        )
    assert raised.value.detail.param == "reasoning_effort"


def test_responses_decoder_accepts_reasoning_context_and_names_rejections() -> None:
    """reasoning.context decodes verbatim; unknown values and fields 400 by name.

    Customer repro: pydantic_ai's OpenAIResponsesModel sends
    reasoning={"effort": ..., "context": "all_turns"} on the gpt-5.6 family
    and OpenAI-direct accepts it, so the gateway must too.
    """
    for value in ("auto", "current_turn", "all_turns"):
        decoded = decode_responses(
            {
                "model": "coding",
                "input": "hi",
                "reasoning": {"effort": "high", "context": value},
            }
        )
        assert decoded.request.reasoning_context == value
    assert decode_responses({"model": "coding", "input": "hi"}).request.reasoning_context is None

    with pytest.raises(OpenAIProtocolError) as invalid_value:
        decode_responses(
            {
                "model": "coding",
                "input": "hi",
                "reasoning": {"context": "every_turn"},
            }
        )
    assert invalid_value.value.detail.param == "reasoning.context"

    # The consciously rejected reasoning.mode field keeps its named 400.
    with pytest.raises(OpenAIProtocolError) as rejected_field:
        decode_responses(
            {
                "model": "coding",
                "input": "hi",
                "reasoning": {"mode": "pro"},
            }
        )
    assert rejected_field.value.detail.param == "reasoning.mode"


def test_responses_decoder_accepts_verbatim_echoes_of_prior_output_items() -> None:
    """Turn-2 input echoing turn-1 output items verbatim must decode.

    Customer repro (Codex continuation model, 2026-08-28): a function_call
    output item carries {arguments, call_id, id, name, status, type}; echoing
    it with its function_call_output failed 400 on the echo-only ``id`` and
    ``status`` markers. Every item shape below mirrors what this gateway (and
    OpenAI-direct) actually emit.
    """
    decoded = decode_responses(
        {
            "model": "coding",
            "input": [
                {"type": "message", "role": "user", "content": "Weather in Paris?"},
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "plan"}],
                    "content": [{"type": "reasoning_text", "text": "thinking"}],
                    "encrypted_content": "blob==",
                    "status": "completed",
                },
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call-1",
                    "name": "get_weather",
                    "arguments": "{}",
                    "status": "completed",
                },
                {
                    "type": "function_call_output",
                    "id": "fco_1",
                    "call_id": "call-1",
                    "output": "sunny",
                    "status": "completed",
                },
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "It is sunny.",
                            "annotations": [],
                            "logprobs": [],
                        }
                    ],
                },
                {"type": "message", "role": "user", "content": "And tomorrow?"},
            ],
        }
    )
    roles = [message.role for message in decoded.request.messages]
    assert roles == ["user", "assistant", "tool", "assistant", "user"]
    assistant_call = decoded.request.messages[1]
    assert assistant_call.tool_calls[0].call_id == "call-1"
    assert assistant_call.provider_reasoning[0].kind == "encrypted_reasoning"
    assert decoded.request.messages[3].content == "It is sunny."


def test_responses_union_errors_name_the_item_field_not_the_branch() -> None:
    """A bad echoed item names its field, never a union branch like input.str."""
    with pytest.raises(OpenAIProtocolError) as raised:
        decode_responses(
            {
                "model": "coding",
                "input": [
                    {"type": "message", "role": "user", "content": "hi"},
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "get_weather",
                        "arguments": "{}",
                        "caller": "direct",
                    },
                ],
            }
        )
    assert raised.value.detail.param == "input.1.caller"


def test_every_chat_cache_control_placement_follows_its_classified_decision() -> None:
    """Each classified cache_control placement decodes per the manifest table.

    Customer repro (opencode, 2026-08-28): the @ai-sdk stack lands the hint
    inside a ``tool_calls`` entry when the message's last content part is a
    tool call, and the closed wire model 400ed every Claude-family multi-turn
    tool session with no client-side workaround.
    """
    from exp.runtime.openai_protocol.manifest import CHAT_CACHE_CONTROL_PLACEMENTS

    assert CHAT_CACHE_CONTROL_PLACEMENTS == {
        "messages": "validated_and_dropped",
        "messages.content": "validated_and_dropped",
        "messages.tool_calls": "validated_and_forwarded_to_anthropic_tool_use",
    }
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "read both files",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    "cache_control": {"type": "ephemeral", "ttl": "5m"},
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path":"a.txt"}'},
                        },
                        {
                            "id": "call-2",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path":"b.txt"}'},
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "a"},
                {"role": "tool", "tool_call_id": "call-2", "content": "b"},
            ],
        }
    )
    calls = decoded.request.messages[1].tool_calls
    assert calls[0].cache_control is None
    assert calls[1].cache_control == {"type": "ephemeral"}
    # Message- and part-level hints stay validated-and-dropped.
    assert "cache_control" not in decoded.request.messages[0].model_dump(mode="json")

    with pytest.raises(OpenAIProtocolError) as raised:
        decode_chat(
            {
                "model": "coding",
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": "{}"},
                                "cache_control": {"type": "persistent"},
                            }
                        ],
                    }
                ],
            }
        )
    assert "cache_control" in str(raised.value.detail.param)


def test_empty_tool_call_arguments_decode_as_the_canonical_empty_object() -> None:
    """A zero-argument echo with arguments '' decodes as {} on both surfaces.

    Mirrors the streaming completion seed: no provider wire accepts empty
    argument bytes, and the @ai-sdk stack normally sends "{}", so the empty
    string is normalized instead of 400ing the continuation.
    """
    chat = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "get_time", "arguments": ""},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "noon"},
            ],
        }
    )
    call = chat.request.messages[0].tool_calls[0]
    assert call.arguments == {}
    assert call.raw_arguments == "{}"

    responses = decode_responses(
        {
            "model": "coding",
            "input": [
                {"type": "function_call", "call_id": "call-1", "name": "get_time", "arguments": ""},
                {"type": "function_call_output", "call_id": "call-1", "output": "noon"},
            ],
        }
    )
    assert responses.request.messages[0].tool_calls[0].raw_arguments == "{}"


def test_the_captured_codex_request_shape_decodes_losslessly() -> None:
    """Regression fixture: the field shapes real Codex (0.151.0) sends by
    default, trimmed from a live capture (2026-08-29). Native items carry
    byte-for-byte; non-assistant message ids are accepted and dropped;
    assistant echoes carry id+phase without status."""
    additional_tools = {
        "type": "additional_tools",
        "id": "at_fixture",
        "role": "developer",
        "tools": [
            {
                "type": "namespace",
                "name": "functions",
                "description": "",
                "tools": [
                    {"type": "custom", "name": "exec", "description": "Run JavaScript"},
                    {
                        "type": "function",
                        "name": "followup_task",
                        "description": "Send a follow-up task",
                        "parameters": {"type": "object", "properties": {}},
                    },
                ],
            }
        ],
    }
    custom_call = {
        "type": "custom_tool_call",
        "id": "ctc_fixture",
        "status": "completed",
        "call_id": "call_fixture",
        "name": "exec",
        "input": 'const r = await tools.exec_command({cmd:"ls"});',
    }
    custom_output = {
        "type": "custom_tool_call_output",
        "id": "ctco_fixture",
        "call_id": "call_fixture",
        "output": "[{'type': 'input_text', 'text': 'file_a.txt'}]",
    }
    decoded = decode_responses(
        {
            "model": "gpt-5.6-sol",
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"effort": "low", "context": "all_turns"},
            "text": {"verbosity": "low"},
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "prompt_cache_key": "session-fixture",
            "client_metadata": {"thread_id": "thread-fixture"},
            "input": [
                additional_tools,
                {
                    "type": "message",
                    "id": "msg_dev_fixture",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "You are Codex."}],
                },
                {
                    "type": "message",
                    "id": "msg_user_fixture",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Run ls."}],
                },
                {
                    "type": "message",
                    "id": "msg_echo_fixture",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Listing now."}],
                    "phase": "commentary",
                },
                custom_call,
                custom_output,
            ],
        }
    )
    request = decoded.request
    assert request.ignored_parameters == ()
    assert request.text_verbosity == "low"
    assert request.client_metadata == {"thread_id": "thread-fixture"}
    assert request.reasoning_effort == "low"
    roles = [message.role for message in request.messages]
    assert roles == ["developer", "developer", "user", "assistant", "assistant", "tool"]
    natives = [
        message.provider_native_item
        for message in request.messages
        if message.provider_native_item is not None
    ]
    assert natives == [additional_tools, custom_call, custom_output]
    # Non-assistant ids drop; the assistant echo retains identity with
    # status OPTIONAL.
    developer = request.messages[1]
    assert developer.provider_item_id is None
    echo = request.messages[3]
    assert echo.provider_item_id == "msg_echo_fixture"
    assert echo.provider_status is None
    assert echo.provider_phase == "commentary"


def test_the_captured_codex_reasoning_echo_with_null_content_decodes() -> None:
    """Regression fixture: the third request of a real Codex (0.151.0)
    session, trimmed from a live capture (2026-08-29). After a
    custom_tool_call round Codex echoes the reasoning output item with an
    explicit ``content: null``; the provider accepts that request, so the
    gateway must decode it instead of rejecting ``input.N.content``."""
    reasoning_echo = {
        "type": "reasoning",
        "id": "rs_fixture",
        "summary": [],
        "content": None,
        "encrypted_content": "gAAAAABfixture",
    }
    decoded = decode_responses(
        {
            "model": "gpt-5.6-sol",
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"effort": "max", "context": "all_turns"},
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "prompt_cache_key": "session-fixture",
            "client_metadata": {"thread_id": "thread-fixture"},
            "input": [
                {
                    "type": "message",
                    "id": "msg_user_fixture",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Read data.txt."}],
                },
                reasoning_echo,
                {
                    "type": "custom_tool_call",
                    "id": "ctc_fixture",
                    "status": "completed",
                    "call_id": "call_fixture",
                    "name": "exec",
                    "input": 'const r = await tools.exec_command({cmd:"cat data.txt"});',
                },
                {
                    "type": "custom_tool_call_output",
                    "id": "ctco_fixture",
                    "call_id": "call_fixture",
                    "output": [
                        {"type": "input_text", "text": "Script completed\n"},
                        {"type": "input_text", "text": "gateway test file\n"},
                    ],
                },
            ],
        }
    )
    request = decoded.request
    roles = [message.role for message in request.messages]
    assert roles == ["user", "assistant", "assistant", "tool"]
    carrier = request.messages[1].provider_reasoning
    assert len(carrier) == 1
    assert isinstance(carrier[0], EncryptedReasoningBlock)
    assert carrier[0].encrypted_content == "gAAAAABfixture"


def test_decode_errors_name_the_expected_shape_against_the_arriving_type() -> None:
    """Union rejections say what shape the field expected and what arrived,
    at type level only, matching the provider's own error style."""
    with pytest.raises(OpenAIProtocolError) as content:
        decode_responses(
            {
                "model": "coding",
                "input": [
                    {
                        "type": "reasoning",
                        "id": "rs_bad",
                        "encrypted_content": "gAAAAABfixture",
                        "content": 7,
                    }
                ],
            }
        )
    assert content.value.detail.param == "input.0.content"
    assert content.value.detail.message == (
        "Invalid value for 'input.0.content': expected an array, but got an integer instead."
    )

    with pytest.raises(OpenAIProtocolError) as summary:
        decode_responses(
            {
                "model": "coding",
                "input": [
                    {
                        "type": "reasoning",
                        "id": "rs_bad",
                        "encrypted_content": "gAAAAABfixture",
                        "summary": None,
                    }
                ],
            }
        )
    assert summary.value.detail.param == "input.0.summary"
    assert summary.value.detail.message == (
        "Invalid value for 'input.0.summary': expected an array, but got null instead."
    )


_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)
"""One valid single-pixel PNG, base64 encoded."""


def test_chat_decoder_retains_image_parts_in_caller_order() -> None:
    """A chat image part is kept beside its text in the order the caller sent."""
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{_PNG_BASE64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": "be brief"},
                    ],
                }
            ],
        }
    )
    message = decoded.request.messages[0]
    assert message.content == "what is thisbe brief"
    assert [part.kind for part in message.content_parts] == ["text", "image", "text"]
    image = decoded.request.images[0]
    assert image.data == _PNG_BASE64
    assert image.media_type == "image/png"
    assert image.detail == "high"


def test_an_empty_text_part_beside_an_image_drops() -> None:
    """A client's empty text part never reaches a wire that rejects one."""
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ""},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{_PNG_BASE64}"},
                        },
                        {"type": "text", "text": "read it"},
                    ],
                }
            ],
        }
    )
    message = decoded.request.messages[0]
    assert message.content == "read it"
    assert [part.kind for part in message.content_parts] == ["image", "text"]


def test_responses_decoder_retains_input_image_parts() -> None:
    """A Responses ``input_image`` survives decoding as a canonical image."""
    decoded = decode_responses(
        {
            "model": "coding",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "describe"},
                        {
                            "type": "input_image",
                            "image_url": "https://example.com/cat.png",
                            "detail": "auto",
                        },
                    ],
                }
            ],
        }
    )
    assert [part.kind for part in decoded.request.messages[0].content_parts] == ["text", "image"]
    assert decoded.request.images[0].url == "https://example.com/cat.png"


def test_text_only_chat_messages_keep_no_content_parts() -> None:
    """Text-only requests decode exactly as before, with no retained parts."""
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        }
    )
    assert decoded.request.messages[0].content_parts == ()
    assert decoded.request.images == ()


def test_images_change_the_canonical_request_digest() -> None:
    """An image changes what the model is asked, so it changes replay identity."""
    text_only = decode_chat(
        {"model": "coding", "messages": [{"role": "user", "content": "what is this"}]}
    )
    with_image = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{_PNG_BASE64}"},
                        },
                    ],
                }
            ],
        }
    )
    assert sha256_json(text_only.request) != sha256_json(with_image.request)


def test_malformed_chat_image_url_is_rejected_with_its_field() -> None:
    """An unusable image carrier names the exact offending request field."""
    with pytest.raises(OpenAIProtocolError) as error:
        decode_chat(
            {
                "model": "coding",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "ftp://example.com/a.png"}}
                        ],
                    }
                ],
            }
        )
    assert error.value.detail.param == "messages.0.content.0.image_url"


def test_assistant_image_parts_are_rejected() -> None:
    """Only a caller message may carry an image."""
    with pytest.raises(OpenAIProtocolError):
        decode_chat(
            {
                "model": "coding",
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{_PNG_BASE64}"},
                            }
                        ],
                    }
                ],
            }
        )


_PDF_BASE64 = "JVBERi0xLjQKJSBtaW5pbWFsIHBkZgo="
"""One short PDF header, base64 encoded."""

_PDF_DATA_URL = f"data:application/pdf;base64,{_PDF_BASE64}"
"""The same PDF as an OpenAI ``file_data`` value."""


def _chat_file(file_data: str = _PDF_DATA_URL, filename: str | None = "brief.pdf") -> JsonObject:
    """Build one Chat Completions ``file`` content part."""
    file: JsonObject = {"file_data": file_data}
    if filename is not None:
        file["filename"] = filename
    return {"type": "file", "file": file}


def test_chat_decoder_retains_file_parts_interleaved_with_text() -> None:
    """Chat ``file`` parts keep their positions among the caller's text."""
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "first: "},
                        _chat_file(),
                        {"type": "text", "text": " second: "},
                        _chat_file("JVBERi0xLjcK", filename=None),
                        {"type": "text", "text": " compare"},
                    ],
                }
            ],
        }
    )
    message = decoded.request.messages[0]
    assert message.content == "first:  second:  compare"
    assert [part.kind for part in message.content_parts] == [
        "text",
        "document",
        "text",
        "document",
        "text",
    ]
    documents = decoded.request.documents
    assert [document.data for document in documents] == [_PDF_BASE64, "JVBERi0xLjcK"]
    assert [document.name for document in documents] == ["brief.pdf", None]
    assert all(document.media_type == "application/pdf" for document in documents)


def test_chat_file_sent_once_survives_a_multi_turn_thread() -> None:
    """A PDF in an earlier user turn is retained when later turns reference it."""
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {"role": "user", "content": [_chat_file(), {"type": "text", "text": "title?"}]},
                {"role": "assistant", "content": "Minimal PDF."},
                {"role": "user", "content": "page count?"},
            ],
        }
    )
    assert [len(message.documents) for message in decoded.request.messages] == [1, 0, 0]


def test_responses_decoder_retains_input_file_parts() -> None:
    """Responses ``input_file`` decodes inline data and remote URLs separately."""
    decoded = decode_responses(
        {
            "model": "coding",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "compare"},
                        {"type": "input_file", "filename": "a.pdf", "file_data": _PDF_DATA_URL},
                        {"type": "input_file", "file_url": "https://example.com/b.pdf"},
                    ],
                }
            ],
        }
    )
    message = decoded.request.messages[0]
    assert [part.kind for part in message.content_parts] == ["text", "document", "document"]
    inline, remote = decoded.request.documents
    assert (inline.data, inline.name, inline.url) == (_PDF_BASE64, "a.pdf", None)
    assert (remote.data, remote.url) == (None, "https://example.com/b.pdf")


def test_documents_change_the_canonical_request_digest() -> None:
    """Document bytes, names, and order all change what the model is asked."""

    def digest(*parts: JsonObject) -> str:
        """Digest one chat request whose user turn carries the given parts."""
        decoded = decode_chat(
            {
                "model": "coding",
                "messages": [{"role": "user", "content": [{"type": "text", "text": "q"}, *parts]}],
            }
        )
        return sha256_json(decoded.request)

    text_only = digest()
    one = digest(_chat_file())
    other_bytes = digest(_chat_file("JVBERi0xLjcK"))
    renamed = digest(_chat_file(filename="other.pdf"))
    assert len({text_only, one, other_bytes, renamed}) == 4
    assert digest(_chat_file(), _chat_file("JVBERi0xLjcK")) != digest(
        _chat_file("JVBERi0xLjcK"), _chat_file()
    )


@pytest.mark.parametrize(
    ("part", "param"),
    [
        (_chat_file("data:text/plain;base64,aGk="), "messages.0.content.0.file.file_data"),
        (_chat_file("!!not base64"), "messages.0.content.0.file.file_data"),
        (
            {"type": "file", "file": {"file_id": "not-a-file-id"}},
            "messages.0.content.0.file.file_id",
        ),
        ({"type": "image_url", "image_url": {}}, "messages.0.content.0.image_url.url"),
    ],
)
def test_unservable_chat_file_parts_are_rejected_at_their_field(
    part: JsonObject, param: str
) -> None:
    """A part the gateway cannot forward names its real field (the union tag
    that doubles as the payload key is reported once), never drops."""
    with pytest.raises(OpenAIProtocolError) as error:
        decode_chat({"model": "coding", "messages": [{"role": "user", "content": [part]}]})
    assert error.value.detail.param == param


@pytest.mark.parametrize(
    "part",
    [
        {"type": "input_file", "file_url": "ftp://example.com/a.pdf"},
        {"type": "input_file", "file_data": _PDF_DATA_URL, "file_url": "https://example.com/a.pdf"},
        {"type": "input_file", "filename": "a.pdf"},
        {"type": "input_file", "file_id": "file_1"},
    ],
)
def test_unservable_responses_input_file_parts_are_rejected(part: JsonObject) -> None:
    """Responses files need exactly one servable carrier."""
    with pytest.raises(OpenAIProtocolError):
        decode_responses({"model": "coding", "input": [{"role": "user", "content": [part]}]})


def test_assistant_file_parts_are_rejected() -> None:
    """Only a caller message may carry a document."""
    with pytest.raises(OpenAIProtocolError):
        decode_chat(
            {"model": "coding", "messages": [{"role": "assistant", "content": [_chat_file()]}]}
        )


def test_too_many_chat_files_are_rejected() -> None:
    """The per-request document ceiling fails closed with the ceiling named."""
    with pytest.raises(OpenAIProtocolError, match="at most 5 documents"):
        decode_chat(
            {
                "model": "coding",
                "messages": [
                    {
                        "role": "user",
                        "content": [_chat_file() for _ in range(MAXIMUM_DOCUMENTS_PER_REQUEST + 1)],
                    }
                ],
            }
        )


def test_non_function_tool_declarations_carry_verbatim_at_their_positions() -> None:
    """Regression fixture: the top-level tools array real Codex (0.151.0)
    sends by default on gpt-5.x models, trimmed from a live capture
    (2026-09-01). Every non-function declaration type api.openai.com accepts
    with a plain key (custom, namespace, web_search, tool_search; each
    verified live 2026-09-01) carries byte-for-byte with its position;
    function declarations keep the strict typed profile."""
    function_tool = {
        "type": "function",
        "name": "exec_command",
        "description": "Execute shell commands",
        "strict": False,
        "parameters": {"type": "object", "properties": {}},
    }
    custom_tool = {
        "type": "custom",
        "name": "apply_patch",
        "description": "Use the `apply_patch` tool to edit files.",
        "format": {
            "type": "grammar",
            "syntax": "lark",
            "definition": 'start: begin_patch hunk+ end_patch\nbegin_patch: "*** Begin Patch"',
        },
    }
    namespace_tool = {
        "type": "namespace",
        "name": "multi_agent_v1",
        "description": "Tools for spawning and managing sub-agents.",
        "tools": [
            {
                "type": "function",
                "name": "close_agent",
                "description": "Close an agent.",
                "strict": False,
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    }
    web_search_tool = {"type": "web_search", "external_web_access": False}
    tool_search_tool = {
        "type": "tool_search",
        "description": "Search for additional tools.",
        "parameters": {"type": "object", "properties": {}},
        "execution": {"type": "server"},
    }
    decoded = decode_responses(
        {
            "model": "gpt-5.2",
            "store": False,
            "stream": True,
            "tool_choice": "required",
            "input": "Run ls.",
            "tools": [
                function_tool,
                custom_tool,
                namespace_tool,
                web_search_tool,
                tool_search_tool,
            ],
        }
    )
    request = decoded.request
    assert [tool.name for tool in request.tools] == ["exec_command"]
    assert [(entry.index, entry.tool) for entry in request.provider_native_tools] == [
        (1, custom_tool),
        (2, namespace_tool),
        (3, web_search_tool),
        (4, tool_search_tool),
    ]


def test_a_malformed_function_tool_declaration_still_fails_closed() -> None:
    """The opaque carrier accepts only non-function types; a function
    declaration missing its name is a named validation error, never an
    opaque forward."""
    with pytest.raises(OpenAIProtocolError) as rejection:
        decode_responses(
            {
                "model": "gpt-5.2",
                "input": "hi",
                "tools": [{"type": "function", "description": "nameless"}],
            }
        )
    assert rejection.value.status_code == 400
    assert "tools" in (rejection.value.detail.param or "")


def test_embeddings_decoder_preserves_supported_fields() -> None:
    """Embeddings conversion keeps the alias, inputs, dimensions, encoding, and attribution."""
    decoded = decode_embeddings(
        {
            "model": "text-embedding-3-small",
            "input": "hello world",
            "dimensions": 256,
            "encoding_format": "float",
            "user": "end-user-7",
        }
    )

    assert decoded.alias == "text-embedding-3-small"
    assert decoded.request.surface == GatewayApiSurface.EMBEDDINGS
    assert decoded.request.inputs == ("hello world",)
    assert decoded.request.dimensions == 256
    assert decoded.request.encoding_format == "float"
    assert decoded.request.user == "end-user-7"


def test_embeddings_decoder_accepts_a_list_of_inputs() -> None:
    """An array of texts decodes in order with defaults for the optional fields."""
    decoded = decode_embeddings({"model": "m", "input": ["first", "second"]})

    assert decoded.request.inputs == ("first", "second")
    assert decoded.request.dimensions is None
    assert decoded.request.encoding_format is None
    assert decoded.request.user is None


def test_embeddings_decoder_rejects_unknown_and_streaming_fields() -> None:
    """A field outside the closed embeddings manifest is a named 400, never silently dropped."""
    with pytest.raises(OpenAIProtocolError) as rejection:
        decode_embeddings({"model": "m", "input": "x", "stream": True})
    assert rejection.value.status_code == 400
    assert "stream" in rejection.value.detail.message


def test_embeddings_decoder_rejects_token_array_input() -> None:
    """Pre-tokenized id arrays pass official validation but this text surface rejects them."""
    with pytest.raises(OpenAIProtocolError) as rejection:
        decode_embeddings({"model": "m", "input": [1, 2, 3]})
    assert rejection.value.status_code == 400
    assert "input" in (rejection.value.detail.param or "")


def test_embeddings_decoder_rejects_empty_and_malformed_inputs() -> None:
    """Empty strings, an empty array, a missing input, and bad options each fail with a param."""
    cases: tuple[tuple[JsonObject, str], ...] = (
        ({"model": "m", "input": ""}, "input"),
        ({"model": "m", "input": []}, "input"),
        ({"model": "m"}, "input"),
        ({"model": "m", "input": "x", "dimensions": 0}, "dimensions"),
        ({"model": "m", "input": "x", "encoding_format": "weird"}, "encoding_format"),
    )
    for payload, param in cases:
        with pytest.raises(OpenAIProtocolError) as rejection:
            decode_embeddings(payload)
        assert rejection.value.status_code == 400
        assert param in (rejection.value.detail.param or "")


_MP4_BASE64 = "AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDE="
"""A base64 prefix of an MP4 ``ftyp`` box, enough for a carrier fixture."""


def test_chat_decoder_retains_video_parts_in_caller_order() -> None:
    """A chat ``video_url`` part is kept beside its text and images in caller order."""
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "first "},
                        {"type": "video_url", "video_url": {"url": "https://example.com/a.mp4"}},
                        {"type": "text", "text": "then "},
                        {
                            "type": "video_url",
                            "video_url": {"url": f"data:video/webm;base64,{_MP4_BASE64}"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{_PNG_BASE64}"},
                        },
                        {"type": "text", "text": "compare"},
                    ],
                }
            ],
        }
    )
    message = decoded.request.messages[0]
    assert message.content == "first then compare"
    assert [part.kind for part in message.content_parts] == [
        "text",
        "video",
        "text",
        "video",
        "image",
        "text",
    ]
    remote, inline = decoded.request.videos
    assert remote.url == "https://example.com/a.mp4"
    assert inline.media_type == "video/webm"
    assert inline.data == _MP4_BASE64
    assert len(decoded.request.images) == 1
    assert [part.kind for part in model_request(decoded.request).messages[0].content_parts] == [
        part.kind for part in message.content_parts
    ]


def test_videos_change_the_canonical_request_digest() -> None:
    """A video changes what the model is asked, so it changes replay identity."""
    text_only = decode_chat(
        {"model": "coding", "messages": [{"role": "user", "content": "what happens"}]}
    )
    with_video = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what happens"},
                        {
                            "type": "video_url",
                            "video_url": {"url": f"data:video/mp4;base64,{_MP4_BASE64}"},
                        },
                    ],
                }
            ],
        }
    )
    other_video = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what happens"},
                        {
                            "type": "video_url",
                            "video_url": {"url": f"data:video/mp4;base64,{_MP4_BASE64[:-4]}"},
                        },
                    ],
                }
            ],
        }
    )
    digests = {
        sha256_json(text_only.request),
        sha256_json(with_video.request),
        sha256_json(other_video.request),
    }
    assert len(digests) == 3
    assert "video" in with_video.request.model_dump_json()


def test_malformed_chat_video_url_is_rejected_with_its_field() -> None:
    """An unusable video carrier names the exact offending request field."""
    for url in ("ftp://example.com/a.mp4", f"data:video/x-matroska;base64,{_MP4_BASE64}"):
        with pytest.raises(OpenAIProtocolError) as error:
            decode_chat(
                {
                    "model": "coding",
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "video_url", "video_url": {"url": url}}],
                        }
                    ],
                }
            )
        assert error.value.detail.param == "messages.0.content.0.video_url"


def test_a_request_carries_at_most_the_video_ceiling() -> None:
    """The eleventh video in one request is refused rather than dropped."""
    part: JsonObject = {"type": "video_url", "video_url": {"url": "https://example.com/a.mp4"}}
    decode_chat({"model": "coding", "messages": [{"role": "user", "content": [part] * 10}]})
    with pytest.raises(OpenAIProtocolError):
        decode_chat({"model": "coding", "messages": [{"role": "user", "content": [part] * 11}]})
    with pytest.raises(ValueError, match="at most 10 videos"):
        GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=(
                GatewayMessage(
                    role="user",
                    content="",
                    content_parts=tuple(
                        VideoContentPart(url="https://example.com/a.mp4") for _ in range(11)
                    ),
                ),
            ),
        )


def test_assistant_video_parts_are_rejected() -> None:
    """Only a caller message may carry a video."""
    with pytest.raises(OpenAIProtocolError):
        decode_chat(
            {
                "model": "coding",
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "video_url", "video_url": {"url": "https://example.com/a.mp4"}}
                        ],
                    }
                ],
            }
        )


def test_responses_surface_defines_no_video_part() -> None:
    """The Responses wire has no video content type, so one is refused, not dropped."""
    with pytest.raises(OpenAIProtocolError):
        decode_responses(
            {
                "model": "coding",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "describe"},
                            {
                                "type": "video_url",
                                "video_url": {"url": "https://example.com/a.mp4"},
                            },
                        ],
                    }
                ],
            }
        )
    with pytest.raises(OpenAIProtocolError):
        decode_responses(
            {
                "model": "coding",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_video", "video_url": "https://example.com/a.mp4"}
                        ],
                    }
                ],
            }
        )


def test_responses_decoder_wraps_file_ids_as_openai_handles() -> None:
    """``input_image.file_id`` and ``input_file.file_id`` become OpenAI handles in order."""
    decoded = decode_responses(
        {
            "model": "coding",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "file_id": "file-img", "detail": "low"},
                        {"type": "input_text", "text": "and this"},
                        {"type": "input_file", "file_id": "file-doc"},
                    ],
                }
            ],
        }
    )
    message = decoded.request.messages[0]
    assert [part.kind for part in message.content_parts] == ["image", "text", "document"]
    (image,) = decoded.request.images
    (document,) = decoded.request.documents
    assert image.handle == MediaHandle(provider="openai", reference="file-img")
    assert (image.detail, image.data, image.url) == ("low", None, None)
    assert document.handle == MediaHandle(provider="openai", reference="file-doc")
    assert decoded.request.media_handles == (image.handle, document.handle)


def test_chat_decoder_wraps_file_part_ids_as_openai_handles() -> None:
    """A Chat ``file.file_id`` becomes an OpenAI handle beside inline siblings."""
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        _chat_file(),
                        {"type": "file", "file": {"file_id": "file-doc", "filename": "b.pdf"}},
                        {"type": "text", "text": "compare"},
                    ],
                }
            ],
        }
    )
    inline, handled = decoded.request.documents
    assert inline.data == _PDF_BASE64 and inline.handle is None
    assert handled.handle == MediaHandle(provider="openai", reference="file-doc")
    assert handled.name == "b.pdf"


def test_provider_uris_in_url_fields_decode_as_handles() -> None:
    """``s3://``, ``gs://``, and Gemini Files URIs are handles on every URL field."""
    gemini = f"{GEMINI_FILE_URI_PREFIX}abc123"

    def chat(part: JsonObject) -> GatewayRequest:
        """Decode one Chat request carrying a single media part and a text run."""
        return decode_chat(
            {
                "model": "coding",
                "messages": [
                    {"role": "user", "content": [part, {"type": "text", "text": "describe"}]}
                ],
            }
        ).request

    (image,) = chat({"type": "image_url", "image_url": {"url": "s3://bkt/cat.png"}}).images
    assert image.handle == MediaHandle(provider="bedrock", reference="s3://bkt/cat.png")
    assert image.media_type == "image/png" and image.url is None
    (video,) = chat({"type": "video_url", "video_url": {"url": "gs://bkt/clip.mp4"}}).videos
    assert video.handle == MediaHandle(provider="vertex", reference="gs://bkt/clip.mp4")
    assert video.media_type == "video/mp4"
    responses = decode_responses(
        {
            "model": "coding",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_file", "file_url": gemini},
                        {"type": "input_image", "image_url": gemini},
                    ],
                }
            ],
        }
    )
    assert [handle.provider for handle in responses.request.media_handles] == ["gemini", "gemini"]
    assert responses.request.media_handles[0].reference == gemini


def test_handles_from_two_providers_in_one_request_are_rejected() -> None:
    """No route can resolve both, so the decoder refuses the request as a whole."""
    with pytest.raises(OpenAIProtocolError, match="same provider"):
        decode_chat(
            {
                "model": "coding",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "s3://bkt/cat.png"}},
                            {"type": "video_url", "video_url": {"url": "gs://bkt/clip.mp4"}},
                        ],
                    }
                ],
            }
        )


@pytest.mark.parametrize(
    ("part", "suffix"),
    [
        (
            {"type": "input_image", "file_id": "file-img", "image_url": "https://x.test/a.png"},
            "content.0.input_image",
        ),
        ({"type": "input_image", "file_id": "not-openai"}, "content.0.file_id"),
        (
            {"type": "input_file", "file_id": "file-a", "file_data": _PDF_DATA_URL},
            "content.0.input_file",
        ),
        ({"type": "input_image", "image_url": "s3://bkt/no-suffix"}, "content.0.image_url"),
    ],
)
def test_malformed_or_doubled_responses_handles_are_rejected(part: JsonObject, suffix: str) -> None:
    """A handle beside another carrier, a foreign id, or a suffixless object fails at its field."""
    with pytest.raises(OpenAIProtocolError) as error:
        decode_responses({"model": "coding", "input": [{"role": "user", "content": [part]}]})
    assert error.value.detail.param is not None
    assert error.value.detail.param.endswith(suffix)


_WAV_BASE64 = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQAAAAA="
"""A 44-byte WAV header with an empty data chunk, base64 encoded."""


def _input_audio(data: str = _WAV_BASE64, audio_format: str = "wav") -> JsonObject:
    """Build one Chat ``input_audio`` content part."""
    return {"type": "input_audio", "input_audio": {"data": data, "format": audio_format}}


def test_chat_decoder_retains_audio_parts_in_caller_order() -> None:
    """A chat ``input_audio`` part is kept beside its text and images in caller order."""
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "first "},
                        _input_audio(),
                        {"type": "text", "text": "then "},
                        _input_audio("SUQzBAAAAAAAAA==", "mp3"),
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{_PNG_BASE64}"},
                        },
                        {"type": "text", "text": "compare"},
                    ],
                }
            ],
        }
    )
    message = decoded.request.messages[0]
    assert message.content == "first then compare"
    assert [part.kind for part in message.content_parts] == [
        "text",
        "audio",
        "text",
        "audio",
        "image",
        "text",
    ]
    wav, mp3 = decoded.request.audios
    assert (wav.media_type, wav.data, wav.audio_format()) == ("audio/wav", _WAV_BASE64, "wav")
    assert (mp3.media_type, mp3.audio_format()) == ("audio/mpeg", "mp3")
    assert len(decoded.request.images) == 1
    assert [part.kind for part in model_request(decoded.request).messages[0].content_parts] == [
        part.kind for part in message.content_parts
    ]


def test_audio_changes_the_canonical_request_digest() -> None:
    """A clip changes what the model is asked, so it changes replay identity."""
    text_only = decode_chat(
        {"model": "coding", "messages": [{"role": "user", "content": "what is said"}]}
    )
    with_audio = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "what is said"}, _input_audio()],
                }
            ],
        }
    )
    other_audio = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is said"},
                        _input_audio(_WAV_BASE64, "mp3"),
                    ],
                }
            ],
        }
    )
    digests = {
        sha256_json(text_only.request),
        sha256_json(with_audio.request),
        sha256_json(other_audio.request),
    }
    assert len(digests) == 3
    assert "audio" in with_audio.request.model_dump_json()


def test_malformed_chat_input_audio_is_rejected_with_its_field() -> None:
    """An unusable clip names the exact offending request field."""
    for part in (_input_audio(audio_format="flac"), _input_audio(data="not base64!")):
        with pytest.raises(OpenAIProtocolError) as error:
            decode_chat({"model": "coding", "messages": [{"role": "user", "content": [part]}]})
        assert error.value.detail.param == "messages.0.content.0.input_audio"
        assert "'wav' or 'mp3'" in error.value.detail.message


def test_a_request_carries_at_most_the_audio_ceiling() -> None:
    """The eleventh clip in one request is refused rather than dropped."""
    part = _input_audio()
    decode_chat({"model": "coding", "messages": [{"role": "user", "content": [part] * 10}]})
    with pytest.raises(OpenAIProtocolError):
        decode_chat({"model": "coding", "messages": [{"role": "user", "content": [part] * 11}]})
    with pytest.raises(ValueError, match="at most 10 audio clips"):
        GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=(
                GatewayMessage(
                    role="user",
                    content="",
                    content_parts=tuple(
                        AudioContentPart(media_type="audio/wav", data=_WAV_BASE64)
                        for _ in range(11)
                    ),
                ),
            ),
        )


def test_assistant_audio_parts_are_rejected() -> None:
    """Only a caller message may carry a clip."""
    with pytest.raises(OpenAIProtocolError):
        decode_chat(
            {"model": "coding", "messages": [{"role": "assistant", "content": [_input_audio()]}]}
        )


def test_responses_surface_refuses_audio_by_name() -> None:
    """The Responses API serves no audio input, so a clip is refused, not dropped."""
    with pytest.raises(OpenAIProtocolError) as error:
        decode_responses(
            {
                "model": "coding",
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": "describe"}, _input_audio()],
                    }
                ],
            }
        )
    assert error.value.detail.param == "input.0.content.1.input_audio"
    assert "Chat Completions" in error.value.detail.message


def test_copilot_hardcoded_no_op_values_decode_on_both_openai_surfaces() -> None:
    """The exact VS Code Copilot custom-endpoint shapes decode end to end.

    Wire-captured from VS Code 1.136 (2026-09-02): Copilot hardcodes ``n: 1``
    (with ``stream_options.include_usage``, ``temperature: 0.1``,
    ``top_p: 1``) on every Chat request and ``truncation: "disabled"`` plus
    ``prompt_cache_options: {"mode": "implicit"}`` on every Responses
    request; each rejection blocked the whole lane. The values are accepted
    as already satisfied (this gateway serves one completion, never
    truncates, and caches implicitly on served routes), never forwarded, and
    disclose nothing because nothing is ignored.
    """
    chat = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0.1,
            "top_p": 1,
            "n": 1,
        }
    )
    assert chat.request.temperature == 0.1
    assert chat.request.include_usage is True
    assert chat.request.ignored_parameters == ()
    assert "n" not in chat.request.model_dump(mode="json")

    responses = decode_responses(
        {
            "model": "coding",
            "input": "hello",
            "truncation": "disabled",
            "prompt_cache_options": {"mode": "implicit"},
        }
    )
    assert responses.request.ignored_parameters == ()
    dumped = responses.request.model_dump(mode="json")
    assert "truncation" not in dumped
    assert "prompt_cache_options" not in dumped


@pytest.mark.parametrize(
    ("decoder", "payload", "param", "fragment"),
    (
        (
            decode_chat,
            {"model": "coding", "messages": [{"role": "user", "content": "x"}], "n": 2},
            "n",
            "default of 1",
        ),
        (
            decode_responses,
            {"model": "coding", "input": "x", "truncation": "auto"},
            "truncation",
            "never truncates",
        ),
        (
            decode_responses,
            {"model": "coding", "input": "x", "prompt_cache_options": {"mode": "explicit"}},
            "prompt_cache_options.mode",
            "implicit",
        ),
        (
            decode_responses,
            {
                "model": "coding",
                "input": "x",
                "prompt_cache_options": {"mode": "implicit", "ttl": "5m"},
            },
            "prompt_cache_options.ttl",
            "",
        ),
    ),
)
def test_non_default_values_of_accepted_no_op_fields_stay_named_rejections(
    decoder: Callable[[JsonObject], DecodedGatewayRequest],
    payload: JsonObject,
    param: str,
    fragment: str,
) -> None:
    """Only the semantically satisfied value of each field is accepted."""
    with pytest.raises(OpenAIProtocolError) as captured:
        decoder(payload)
    assert captured.value.detail.code == "invalid_parameter"
    assert captured.value.detail.param == param
    assert fragment in captured.value.detail.message


def test_service_tier_decodes_on_both_openai_surfaces_and_rejects_unknown_values() -> None:
    """Doubleword's tier passthrough (PR #728): valid tiers land on the
    carrier for BYOK forwarding; unknown values (including Anthropic's
    'fast', which is a speed selector, not an OpenAI tier) reject by name."""
    chat = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "x"}],
            "service_tier": "flex",
        }
    )
    assert chat.request.service_tier == "flex"
    assert "service_tier" not in chat.request.model_dump(mode="json")
    responses = decode_responses({"model": "coding", "input": "x", "service_tier": "priority"})
    assert responses.request.service_tier == "priority"

    invalid_cases: tuple[tuple[JsonObject, Callable[[JsonObject], DecodedGatewayRequest]], ...] = (
        (
            {
                "model": "coding",
                "messages": [{"role": "user", "content": "x"}],
                "service_tier": "fast",
            },
            decode_chat,
        ),
        ({"model": "coding", "input": "x", "service_tier": "turbo"}, decode_responses),
    )
    for payload, decoder in invalid_cases:
        with pytest.raises(OpenAIProtocolError) as captured:
            decoder(payload)
        assert captured.value.detail.param == "service_tier"


def test_a_namespaced_function_call_round_trips_its_namespace_verbatim() -> None:
    """The exact Codex namespaced wire shape reaches the provider unchanged.

    OpenAI rejects a namespaced call replayed without its namespace ("Missing
    namespace for function_call 'spawn_agent'. It does not exist in the
    default namespace. Round-trip the model's function_call item with its
    namespace field included."), and the item is baked into the caller's
    history, so dropping or rejecting the field wedges every later turn.
    """
    decoded = decode_responses(
        {
            "model": "coding",
            "input": [
                {"type": "message", "role": "user", "content": "spawn a worker"},
                {
                    "type": "function_call",
                    "call_id": "call_x",
                    "name": "spawn_agent",
                    "namespace": "collaboration",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_x",
                    "name": "spawn_agent",
                    "namespace": "collaboration",
                    "output": "spawned",
                },
            ],
        }
    )
    call = decoded.request.messages[1].tool_calls[0]
    assert call.provider_namespace == "collaboration"
    tool_message = decoded.request.messages[2]
    assert tool_message.provider_tool_name == "spawn_agent"
    assert tool_message.provider_tool_namespace == "collaboration"

    payload = openai_responses_stream_payload(
        "gpt-fixture", decoded.request, supports_temperature=False
    )
    payload_input = cast(list[JsonObject], payload["input"])
    assert payload_input[-2] == {
        "type": "function_call",
        "call_id": "call_x",
        "name": "spawn_agent",
        "arguments": "{}",
        "namespace": "collaboration",
    }
    assert payload_input[-1] == {
        "type": "function_call_output",
        "call_id": "call_x",
        "output": "spawned",
        "name": "spawn_agent",
        "namespace": "collaboration",
    }


def test_a_namespace_free_function_call_keeps_its_exact_wire_shape() -> None:
    """Histories from before namespaced tools re-emit byte-identically."""
    decoded = decode_responses(
        {
            "model": "coding",
            "input": [
                {"type": "function_call", "call_id": "call_p", "name": "f", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_p", "output": "ok"},
            ],
        }
    )
    assert decoded.request.messages[0].tool_calls[0].provider_namespace is None
    assert decoded.request.messages[1].provider_tool_name is None
    assert decoded.request.messages[1].provider_tool_namespace is None
    payload = openai_responses_stream_payload(
        "gpt-fixture", decoded.request, supports_temperature=False
    )
    payload_input = cast(list[JsonObject], payload["input"])
    assert payload_input[-2] == {
        "type": "function_call",
        "call_id": "call_p",
        "name": "f",
        "arguments": "{}",
    }
    assert payload_input[-1] == {
        "type": "function_call_output",
        "call_id": "call_p",
        "output": "ok",
    }


def test_a_malformed_function_call_namespace_names_its_field() -> None:
    """An unusable namespace value reports its own input location."""
    with pytest.raises(OpenAIProtocolError) as error:
        decode_responses(
            {
                "model": "coding",
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "call_x",
                        "name": "spawn_agent",
                        "namespace": "",
                        "arguments": "{}",
                    }
                ],
            }
        )
    assert error.value.detail.param == "input.0.namespace"


def test_hosted_tool_item_echoes_decode_as_verbatim_native_items() -> None:
    """A stateless turn-2 request echoes prior hosted-tool output items
    (web_search_call, mcp_call, ...) and their caller-authored outputs; each
    decodes as a shallow native item carried byte-for-byte at its position,
    with provider-authored items on the assistant role and caller-authored
    outputs on the tool role."""
    web_search = {
        "type": "web_search_call",
        "id": "ws_1",
        "status": "completed",
        "action": {"type": "search", "query": "current stable Python"},
    }
    mcp_call = {
        "type": "mcp_call",
        "id": "mcp_1",
        "server_label": "deepwiki",
        "name": "ask_question",
        "arguments": '{"q": "pi"}',
        "output": "3.14159",
        "status": "completed",
    }
    computer_output = {
        "type": "computer_call_output",
        "call_id": "call_c1",
        "output": {"type": "computer_screenshot", "image_url": "https://example.com/shot.png"},
    }
    decoded = decode_responses(
        {
            "model": "gpt-5.6-sol",
            "input": [
                {"role": "user", "content": "search then act"},
                web_search,
                mcp_call,
                computer_output,
                {"role": "user", "content": "continue"},
            ],
        }
    )
    messages = decoded.request.messages
    assert [message.role for message in messages] == [
        "user",
        "assistant",
        "assistant",
        "tool",
        "user",
    ]
    natives = [
        message.provider_native_item
        for message in messages
        if message.provider_native_item is not None
    ]
    assert natives == [web_search, mcp_call, computer_output]

    payload = openai_responses_stream_payload(
        "gpt-5.6-sol", decoded.request, supports_temperature=False
    )
    assert payload["input"] == [
        {"role": "user", "content": "search then act"},
        web_search,
        mcp_call,
        computer_output,
        {"role": "user", "content": "continue"},
    ]


def test_annotation_bearing_assistant_echoes_decode_and_drop_on_replay() -> None:
    """A cited web-search answer served by this gateway carries populated
    ``annotations`` on its ``output_text`` part; the caller resends it
    verbatim on turn 2 and the echo must decode (a 400 here wedges every
    later turn of the session). The display-only annotations are validated
    and dropped on replay, like echoed reasoning summary parts."""
    decoded = decode_responses(
        {
            "model": "gpt-5.6-sol",
            "input": [
                {"role": "user", "content": "search"},
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Python 3.14.7.",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url": "https://www.python.org/doc/versions/",
                                    "title": "Python versions",
                                    "start_index": 0,
                                    "end_index": 14,
                                }
                            ],
                            "logprobs": [],
                        }
                    ],
                },
            ],
        }
    )
    echo = decoded.request.messages[1]
    assert echo.role == "assistant"
    assert echo.content == "Python 3.14.7."
    payload = openai_responses_stream_payload(
        "gpt-5.6-sol", decoded.request, supports_temperature=False
    )
    replayed = cast("list[JsonObject]", payload["input"])[1]
    content = cast("list[JsonObject]", replayed["content"])
    assert content[0]["annotations"] == []


def test_hosted_tool_echo_annotations_require_a_typed_object() -> None:
    """Malformed annotation echoes stay a named 400, never a silent accept."""
    with pytest.raises(OpenAIProtocolError) as rejected:
        decode_responses(
            {
                "model": "gpt-5.6-sol",
                "input": [
                    {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "x",
                                "annotations": [{"url": "https://example.com"}],
                            }
                        ],
                    },
                ],
            }
        )
    assert rejected.value.status_code == 400


def test_responses_decoder_names_a_duplicate_call_id_instead_of_crashing() -> None:
    """A canonical-contract violation in replayed history is a named 400.

    Two echoed ``function_call`` items sharing one ``call_id`` violate the
    canonical assistant-message contract during history reconstruction,
    after the wire models have already passed. That exception must map to
    the field-specific protocol error every other invalid shape gets: before
    the mapping it escaped decode as an unclassified 500 whose "retry the
    request" guidance is wrong for caller-shaped input.
    """
    with pytest.raises(OpenAIProtocolError) as rejected:
        decode_responses(
            {
                "model": "gpt-5.6-sol",
                "tools": [{"type": "function", "name": "a", "parameters": {"type": "object"}}],
                "input": [
                    {"role": "user", "content": "t"},
                    {"type": "function_call", "call_id": "dup", "name": "a", "arguments": "{}"},
                    {"type": "function_call", "call_id": "dup", "name": "a", "arguments": "{}"},
                    {"type": "function_call_output", "call_id": "dup", "output": "x"},
                    {"type": "function_call_output", "call_id": "dup", "output": "y"},
                ],
            }
        )
    assert rejected.value.status_code == 400
    assert rejected.value.detail.param == "input"
    assert "tool call IDs must be unique" in rejected.value.detail.message


def test_a_caller_attributed_function_call_round_trips_its_caller_verbatim() -> None:
    """The SDK 3.0 programmatic tool-calling attribution reaches the provider unchanged.

    ``caller`` (for example ``{"type": "program", "id": ...}``) arrives on
    function_call, function_call_output, and custom_tool_call items; each is
    baked into the caller's history, so dropping or rejecting the field would
    wedge every later turn of a session the provider itself serves.
    """
    caller = {"type": "program", "caller_id": "call_prog"}
    decoded = decode_responses(
        {
            "model": "coding",
            "input": [
                {"type": "message", "role": "user", "content": "run the program"},
                {
                    "type": "function_call",
                    "call_id": "call_c",
                    "name": "lookup",
                    "caller": caller,
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_c",
                    "caller": caller,
                    "output": "found",
                },
                {
                    "type": "custom_tool_call",
                    "call_id": "call_free",
                    "name": "freeform",
                    "caller": caller,
                    "input": "raw text",
                },
            ],
        }
    )
    call = decoded.request.messages[1].tool_calls[0]
    assert call.provider_caller == caller
    tool_message = decoded.request.messages[2]
    assert tool_message.provider_tool_caller == caller
    native = decoded.request.messages[3].provider_native_item
    assert native is not None and native["caller"] == caller

    payload = openai_responses_stream_payload(
        "gpt-fixture", decoded.request, supports_temperature=False
    )
    payload_input = cast(list[JsonObject], payload["input"])
    assert payload_input[-3] == {
        "type": "function_call",
        "call_id": "call_c",
        "name": "lookup",
        "arguments": "{}",
        "caller": caller,
    }
    assert payload_input[-2] == {
        "type": "function_call_output",
        "call_id": "call_c",
        "output": "found",
        "caller": caller,
    }
    assert payload_input[-1] == {
        "type": "custom_tool_call",
        "call_id": "call_free",
        "name": "freeform",
        "caller": caller,
        "input": "raw text",
    }


def test_a_caller_free_function_call_keeps_its_exact_wire_shape() -> None:
    """Histories from before programmatic tool calling re-emit byte-identically."""
    decoded = decode_responses(
        {
            "model": "coding",
            "input": [
                {"type": "function_call", "call_id": "call_p", "name": "f", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_p", "output": "ok"},
            ],
        }
    )
    assert decoded.request.messages[0].tool_calls[0].provider_caller is None
    assert decoded.request.messages[1].provider_tool_caller is None
    payload = openai_responses_stream_payload(
        "gpt-fixture", decoded.request, supports_temperature=False
    )
    payload_input = cast(list[JsonObject], payload["input"])
    assert "caller" not in payload_input[-2]
    assert "caller" not in payload_input[-1]


def test_a_list_valued_function_output_maps_onto_canonical_tool_parts() -> None:
    """A tool that returns an image beside text serves instead of a 400.

    The SDK types ``function_call_output.output`` as a union of plain text
    and a content-part list; the list form decodes onto the canonical tool
    message (text and image parts) and re-emits as the typed list, so the
    history round-trips on the native Responses wire.
    """
    image_url = "data:image/png;base64,aGk="
    decoded = decode_responses(
        {
            "model": "coding",
            "input": [
                {"type": "function_call", "call_id": "call_s", "name": "shot", "arguments": "{}"},
                {
                    "type": "function_call_output",
                    "call_id": "call_s",
                    "output": [
                        {"type": "input_text", "text": "screenshot:"},
                        {"type": "input_image", "image_url": image_url},
                    ],
                },
            ],
        }
    )
    tool_message = decoded.request.messages[-1]
    assert tool_message.content == "screenshot:"
    assert [part.kind for part in tool_message.content_parts] == ["text", "image"]

    payload = openai_responses_stream_payload(
        "gpt-fixture", decoded.request, supports_temperature=False
    )
    payload_input = cast(list[JsonObject], payload["input"])
    assert payload_input[-1]["type"] == "function_call_output"
    output_value = cast(list[JsonObject], payload_input[-1]["output"])
    assert output_value[0] == {"type": "input_text", "text": "screenshot:"}
    assert output_value[1]["type"] == "input_image"
    assert output_value[1]["image_url"] == image_url


def test_an_all_text_list_function_output_flattens_to_the_plain_string_form() -> None:
    """A text-only part list keeps the exact pre-list message and wire shape."""
    decoded = decode_responses(
        {
            "model": "coding",
            "input": [
                {"type": "function_call", "call_id": "call_t", "name": "f", "arguments": "{}"},
                {
                    "type": "function_call_output",
                    "call_id": "call_t",
                    "output": [
                        {"type": "input_text", "text": "part one "},
                        {"type": "input_text", "text": "part two"},
                    ],
                },
            ],
        }
    )
    tool_message = decoded.request.messages[-1]
    assert tool_message.content == "part one part two"
    assert tool_message.content_parts == ()
    payload = openai_responses_stream_payload(
        "gpt-fixture", decoded.request, supports_temperature=False
    )
    payload_input = cast(list[JsonObject], payload["input"])
    assert payload_input[-1]["output"] == "part one part two"


def test_a_file_part_inside_a_function_output_is_rejected_by_name() -> None:
    """The canonical tool message carries text and image parts only."""
    with pytest.raises(OpenAIProtocolError) as error:
        decode_responses(
            {
                "model": "coding",
                "input": [
                    {"type": "function_call", "call_id": "call_f", "name": "f", "arguments": "{}"},
                    {
                        "type": "function_call_output",
                        "call_id": "call_f",
                        "output": [
                            {
                                "type": "input_file",
                                "file_data": "data:application/pdf;base64,aGk=",
                            }
                        ],
                    },
                ],
            }
        )
    assert error.value.status_code == 400
    assert error.value.detail.param == "input.1.output"


def test_a_custom_tool_call_output_list_carries_verbatim() -> None:
    """The freeform result item is opaque: a list output rides the raw item."""
    output_value = [{"type": "input_text", "text": "raw"}]
    decoded = decode_responses(
        {
            "model": "coding",
            "input": [
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call_free",
                    "output": output_value,
                }
            ],
        }
    )
    native = decoded.request.messages[-1].provider_native_item
    assert native is not None and native["output"] == output_value


def test_a_tool_message_name_round_trips_on_both_openai_wires() -> None:
    """The legacy role:"function" name on a tool result serves instead of a 400.

    hermes-agent (and other agent frameworks) sends name:"<function>" on
    every tool-result message; the provider serves the shape (probed live
    2026-09-05), and the field is baked into history, so the previous
    "Invalid value for 'messages.N.name'" rejection wedged every session on
    its first tool call.
    """
    from exp.runtime.models.providers.streaming_requests import (
        openai_compatible_stream_payload,
        openai_responses_stream_payload,
    )

    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {"role": "user", "content": "read it"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_probe1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_probe1",
                    "name": "read_file",
                    "content": "contents",
                },
            ],
        }
    )
    tool_message = decoded.request.messages[-1]
    assert tool_message.provider_tool_name == "read_file"

    chat_payload = openai_compatible_stream_payload("chat-fixture", decoded.request)
    chat_messages = cast(list[JsonObject], chat_payload["messages"])
    assert chat_messages[-1] == {
        "role": "tool",
        "content": "contents",
        "tool_call_id": "call_probe1",
        "name": "read_file",
    }

    responses_payload = openai_responses_stream_payload(
        "gpt-fixture", decoded.request, supports_temperature=False
    )
    responses_input = cast(list[JsonObject], responses_payload["input"])
    assert responses_input[-1]["name"] == "read_file"


def test_a_name_free_tool_message_keeps_its_exact_chat_wire_shape() -> None:
    """Histories without the legacy attribution re-emit byte-identically."""
    from exp.runtime.models.providers.streaming_requests import openai_compatible_stream_payload

    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            ],
        }
    )
    assert decoded.request.messages[-1].provider_tool_name is None
    chat_payload = openai_compatible_stream_payload("chat-fixture", decoded.request)
    chat_messages = cast(list[JsonObject], chat_payload["messages"])
    assert chat_messages[-1] == {"role": "tool", "content": "ok", "tool_call_id": "call_1"}


def test_a_name_on_a_non_tool_message_stays_a_named_400() -> None:
    """The participant-name field on other roles keeps the explicit rejection."""
    with pytest.raises(OpenAIProtocolError) as error:
        decode_chat(
            {
                "model": "coding",
                "messages": [{"role": "user", "content": "hi", "name": "alice"}],
            }
        )
    assert error.value.status_code == 400
    assert error.value.detail.param == "messages.0"
    assert "name is valid only for tool messages" in str(error.value.detail.message)


def test_replayed_reasoning_content_degrades_instead_of_wedging_cross_model_sessions() -> None:
    """The prod OpenCode + gpt-6-astra wedge: rc in history serves everywhere.

    A session that touched a reasoning-exposed rung (or whose AI-SDK client
    re-serializes reasoning parts) carries plaintext reasoning_content in its
    transcript — on tool-call turns too. A non-exposed route now drops the
    block with disclosure and dispatches without it; an exposed route
    forwards it verbatim beside the tool calls. Previously both repro shapes
    400d ("must be a gateway-issued carrier on an assistant tool-call turn" /
    "carries plaintext reasoning"), killing the session the moment it
    switched models.
    """
    from exp.runtime.models.providers.base import GatewayWireProfile
    from exp.runtime.models.providers.streaming_requests import (
        dialect_stream_payload,
        route_generation_parameter_requests,
    )

    astra = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://astra.test/v1",
        model_id="gpt-6-astra",
        supports_reasoning=True,
        reasoning_wire_format="reasoning_effort",
    )
    exposed = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://tokenhub-intl.tencentcloudmaas.com/v1",
        model_id="hy4-preview",
        supports_reasoning=True,
        reasoning_wire_format="reasoning",
        reasoning_output_exposed=True,
    )

    # Repro A (the exact prod screenshot shape): rc beside tool_calls.
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {"role": "user", "content": "what time is it"},
                {
                    "role": "assistant",
                    "reasoning_content": "The user wants the time; call get_time.",
                    "tool_calls": [
                        {
                            "id": "call_rc1",
                            "type": "function",
                            "function": {"name": "get_time", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_rc1", "content": "12:00"},
                {"role": "user", "content": "thanks, and the date?"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_time",
                        "description": "d",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        }
    )
    public, provider = route_generation_parameter_requests((astra,), decoded.request)
    assert (
        "messages.reasoning_content->dropped(unsupported_by_provider)" in public.ignored_parameters
    )
    payload = dialect_stream_payload(astra, provider.model_copy(update={"stream": True}))
    astra_messages = cast(list[JsonObject], payload["messages"])
    assert all("reasoning_content" not in message for message in astra_messages)
    assert astra_messages[1]["tool_calls"]

    # The same history onto an exposed route forwards the plaintext verbatim
    # beside the tool calls (the provider's own wire shape) — no demand for a
    # sealed carrier on history the gateway never issued.
    public_exposed, provider_exposed = route_generation_parameter_requests(
        (exposed,), decoded.request
    )
    assert not any("reasoning_content" in path for path in public_exposed.ignored_parameters)
    exposed_payload = dialect_stream_payload(
        exposed, provider_exposed.model_copy(update={"stream": True})
    )
    exposed_messages = cast(list[JsonObject], exposed_payload["messages"])
    assert exposed_messages[1]["reasoning_content"] == "The user wants the time; call get_time."
    assert exposed_messages[1]["tool_calls"]

    # Repro B: rc on a plain assistant turn, non-exposed route.
    decoded_b = decode_chat(
        {
            "model": "coding",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "A", "reasoning_content": "thinking..."},
                {"role": "user", "content": "again"},
            ],
        }
    )
    public_b, provider_b = route_generation_parameter_requests((astra,), decoded_b.request)
    assert (
        "messages.reasoning_content->dropped(unsupported_by_provider)"
        in public_b.ignored_parameters
    )
    payload_b = dialect_stream_payload(astra, provider_b.model_copy(update={"stream": True}))
    plain_messages = cast(list[JsonObject], payload_b["messages"])
    assert all("reasoning_content" not in message for message in plain_messages)


def test_a_forged_carrier_prefix_on_a_tool_turn_never_decodes_as_plaintext() -> None:
    """The sealed-carrier boundary holds: a spoofed carrier is a named 400.

    Caller plaintext on tool-call turns decodes as caller-owned exposed
    history, but text carrying a gateway carrier PREFIX must parse as the
    genuine gateway-issued carrier or reject — it never falls back to the
    plaintext path, so untrusted input cannot be interpreted as (or
    substituted for) gateway-issued reasoning bound to the calls.
    """
    with pytest.raises(OpenAIProtocolError) as forged:
        decode_chat(
            {
                "model": "coding",
                "messages": [
                    {"role": "user", "content": "look it up"},
                    {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": (
                            "x-experiential-fireworks-reasoning-v2:not-a-real-carrier"
                        ),
                        "tool_calls": [
                            {
                                "id": "call-one",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call-one", "content": "done"},
                ],
            }
        )
    assert forged.value.detail.param == "messages.1.reasoning_content"
    assert "gateway-issued carrier" in str(forged.value.detail.message)


def test_forced_tool_choice_decodes_canonically_on_both_openai_surfaces() -> None:
    """Chat ``required``/named-function and Responses ``required``/named-function
    both reach the canonical forced forms admission narrows and coerces on."""
    tools_chat = [
        {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
    ]
    chat_required = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "go"}],
            "tools": tools_chat,
            "tool_choice": "required",
        }
    )
    assert chat_required.request.tool_choice == "required"
    chat_named = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "go"}],
            "tools": tools_chat,
            "tool_choice": {"type": "function", "function": {"name": "lookup"}},
        }
    )
    assert chat_named.request.tool_choice == GatewayNamedToolChoice(name="lookup")

    tools_responses = [{"type": "function", "name": "lookup", "parameters": {"type": "object"}}]
    responses_required = decode_responses(
        {"model": "coding", "input": "go", "tools": tools_responses, "tool_choice": "required"}
    )
    assert responses_required.request.tool_choice == "required"
    responses_named = decode_responses(
        {
            "model": "coding",
            "input": "go",
            "tools": tools_responses,
            "tool_choice": {"type": "function", "name": "lookup"},
        }
    )
    assert responses_named.request.tool_choice == GatewayNamedToolChoice(name="lookup")


def test_tool_description_bounds_are_uniform_and_named_on_both_surfaces() -> None:
    """65,536-char descriptions serve; 65,537 is a self-explanatory named 400.

    Prod report: an 8,292-char tool description 400d every agentic turn at
    the old 8,192 bound while the provider itself serves 66,000+ (probed
    live 2026-09-05). The bound now matches the Messages surface and the
    canonical GatewayToolDefinition, and the over-limit rejection states the
    limit and the arriving length instead of forcing the caller to bisect.
    """
    reporter_sized = "x" * 8_292
    at_bound = "x" * 65_536
    over_bound = "x" * 65_537

    for description in (reporter_sized, at_bound):
        decoded = decode_chat(
            {
                "model": "coding",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "t",
                            "description": description,
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            }
        )
        assert decoded.request.tools[0].description == description

    with pytest.raises(OpenAIProtocolError) as chat_over:
        decode_chat(
            {
                "model": "coding",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "t",
                            "description": over_bound,
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            }
        )
    assert chat_over.value.detail.param == "tools.0.function.description"
    assert "at most 65,536 characters" in str(chat_over.value.detail.message)
    assert "65,537" in str(chat_over.value.detail.message)

    decoded = decode_responses(
        {
            "model": "coding",
            "input": "hi",
            "tools": [
                {
                    "type": "function",
                    "name": "t",
                    "description": at_bound,
                    "parameters": {"type": "object"},
                }
            ],
        }
    )
    assert decoded.request.tools[0].description == at_bound
    with pytest.raises(OpenAIProtocolError) as responses_over:
        decode_responses(
            {
                "model": "coding",
                "input": "hi",
                "tools": [
                    {
                        "type": "function",
                        "name": "t",
                        "description": over_bound,
                        "parameters": {"type": "object"},
                    }
                ],
            }
        )
    assert responses_over.value.detail.param == "tools.0.description"
    assert "at most 65,536 characters" in str(responses_over.value.detail.message)


def test_structured_format_description_bounds_are_uniform_and_named() -> None:
    """The response_format and text.format description bounds match the tools'."""
    at_bound = "x" * 65_536
    over_bound = "x" * 65_537

    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "shape",
                    "description": at_bound,
                    "schema": {"type": "object"},
                },
            },
        }
    )
    assert decoded.request.structured_text is not None
    assert decoded.request.structured_text.description == at_bound
    with pytest.raises(OpenAIProtocolError) as chat_over:
        decode_chat(
            {
                "model": "coding",
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "shape",
                        "description": over_bound,
                        "schema": {"type": "object"},
                    },
                },
            }
        )
    assert chat_over.value.detail.param == "response_format.json_schema.description"
    assert "at most 65,536 characters" in str(chat_over.value.detail.message)

    decoded = decode_responses(
        {
            "model": "coding",
            "input": "hi",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "shape",
                    "description": at_bound,
                    "schema": {"type": "object"},
                }
            },
        }
    )
    assert decoded.request.structured_text is not None
    assert decoded.request.structured_text.description == at_bound
    with pytest.raises(OpenAIProtocolError) as responses_over:
        decode_responses(
            {
                "model": "coding",
                "input": "hi",
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "shape",
                        "description": over_bound,
                        "schema": {"type": "object"},
                    }
                },
            }
        )
    assert responses_over.value.detail.param == "text.format.description"
    assert "at most 65,536 characters" in str(responses_over.value.detail.message)
