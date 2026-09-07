"""Tests for launch-provider streaming request payload translation."""

from typing import Literal, cast

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models.content import (
    AudioContentPart,
    DocumentContentPart,
    ImageContentPart,
    MediaHandle,
    TextContentPart,
    VideoContentPart,
)
from exp.common.models.model import ReasoningEffort, ToolCall
from exp.runtime.gateway.contracts import (
    ExposedReasoningContentBlock,
    GatewayApiSurface,
    GatewayMessage,
    GatewayNamedToolChoice,
    GatewayProviderNativeTool,
    GatewayRequest,
    GatewayToolDefinition,
    StructuredTextFormat,
    ThinkingBlock,
)
from exp.runtime.models.providers.anthropic_tool_compat import (
    anthropic_rejects_assistant_prefill,
)
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.bedrock_requests import converse_body
from exp.runtime.models.providers.errors import (
    ProviderCapabilityError,
    ProviderParameterError,
    UnsupportedReasoningEffortError,
)
from exp.runtime.models.providers.gemini_requests import gemini_generate_request
from exp.runtime.models.providers.generation_route_compat import (
    compatible_generation_parameter_profile_indexes,
)
from exp.runtime.models.providers.messages_payloads import JSON_OBJECT_SYSTEM_INSTRUCTION
from exp.runtime.models.providers.streaming_requests import (
    TOOL_RESULT_IMAGE_DROP_DISCLOSURE,
    TOOL_RESULT_IMAGE_PLACEHOLDER,
    anthropic_messages_stream_payload,
    bedrock_converse_stream_payload,
    dialect_stream_payload,
    gemini_generate_content_stream_payload,
    openai_compatible_stream_payload,
    openai_responses_stream_payload,
    route_generation_parameter_requests,
)
from exp.runtime.openai_protocol.model_adapter import model_request

_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)
"""One valid single-pixel PNG, base64 encoded."""


def _chat_request(
    *,
    temperature: float | None = None,
    top_p: float | None = None,
) -> GatewayRequest:
    """Build one Chat Completions streaming request with optional sampling fields.

    Args:
        temperature: Optional sampling temperature to include.
        top_p: Optional nucleus-sampling mass to include.

    Returns:
        A streaming Chat request carrying the supplied sampling fields.
    """
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hello"),),
        temperature=temperature,
        top_p=top_p,
        stream=True,
        include_usage=True,
    )


def _fireworks_responses_request(
    *,
    tool_choice: Literal["auto", "none", "required"] | GatewayNamedToolChoice | None = None,
) -> GatewayRequest:
    """Build a store-disabled Fireworks Responses request that declares one tool."""
    return GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="hello"),),
        tools=(GatewayToolDefinition(name="lookup", parameters={"type": "object"}),),
        tool_choice=tool_choice,
        response_store=False,
        stream=True,
        include_usage=True,
    )


def _fireworks_profile() -> GatewayWireProfile:
    """Return the exact public Fireworks Chat endpoint profile."""
    return GatewayWireProfile(
        dialect="openai_compatible",
        url="https://api.fireworks.ai/inference/v1/chat/completions",
        model_id="accounts/fireworks/models/deepseek-v4-flash-0731",
        fireworks_reasoning_route_sha256="f" * 64,
    )


def test_fireworks_tools_disabled_needs_no_continuation_channel() -> None:
    """Declared tools plus tool_choice none remains a guaranteed text-only turn."""
    profile = _fireworks_profile()
    request = _fireworks_responses_request(tool_choice="none")

    route_generation_parameter_requests((profile,), request)
    payload = dialect_stream_payload(profile, request)

    assert payload["tool_choice"] == "none"


@pytest.mark.parametrize(
    "tool_choice",
    (None, "auto", "required", GatewayNamedToolChoice(name="lookup")),
)
def test_fireworks_tools_that_can_run_require_continuation(
    tool_choice: Literal["auto", "none", "required"] | GatewayNamedToolChoice | None,
) -> None:
    """Every selector able to emit a tool call still fails closed without retention."""
    profile = _fireworks_profile()
    request = _fireworks_responses_request(tool_choice=tool_choice)

    with pytest.raises(ProviderParameterError, match="continuation"):
        route_generation_parameter_requests((profile,), request)
    with pytest.raises(ProviderParameterError, match="continuation"):
        dialect_stream_payload(profile, request)


def test_fireworks_continuation_gate_accepts_no_tools() -> None:
    """A request without tools needs no continuation channel."""
    profile = _fireworks_profile()
    request = _fireworks_responses_request()

    safe = request.model_copy(update={"tools": ()})
    route_generation_parameter_requests((profile,), safe)
    assert dialect_stream_payload(profile, safe)["stream"] is True


@pytest.mark.parametrize(
    "update",
    ({"response_store": True}, {"include_encrypted_reasoning": True}),
)
def test_fireworks_continuation_channels_are_accepted(
    update: dict[str, bool],
) -> None:
    """Server retention and encrypted carriers each provide a safe continuation channel."""
    profile = _fireworks_profile()
    request = _fireworks_responses_request().model_copy(update=update)

    route_generation_parameter_requests((profile,), request)
    assert dialect_stream_payload(profile, request)["stream"] is True


def test_fireworks_encrypted_reasoning_uses_the_authenticated_carrier() -> None:
    """Encrypted reasoning selects the authenticated gateway carrier channel."""
    profile = _fireworks_profile()
    request = _fireworks_responses_request().model_copy(
        update={"include_encrypted_reasoning": True}
    )

    route_generation_parameter_requests((profile,), request)
    assert dialect_stream_payload(profile, request)["stream"] is True


def test_openai_compatible_stream_payload_forwards_top_p_and_usage() -> None:
    """Streaming Chat payloads preserve nucleus sampling and always request usage."""
    payload = openai_compatible_stream_payload(
        "exact-model",
        _chat_request(top_p=1.0, temperature=0.2),
    )

    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 1.0


def _json_object_request() -> GatewayRequest:
    """One Chat request in schema-free JSON-object mode with a leading system turn."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(
            GatewayMessage(role="system", content="You are terse."),
            GatewayMessage(role="user", content="hello"),
        ),
        json_object_output=True,
    )


def test_openai_compatible_payload_passes_json_object_through_natively() -> None:
    """json_object mode is the native Chat response_format on OpenAI-compatible rungs
    (OpenAI, Azure, OpenRouter, Fireworks); no schema is invented."""
    payload = openai_compatible_stream_payload("exact-model", _json_object_request())

    assert payload["response_format"] == {"type": "json_object"}


def test_openai_responses_payload_passes_json_object_through_natively() -> None:
    """A Chat json_object served by a Responses rung uses the native text.format mode."""
    payload = openai_responses_stream_payload(
        "exact-model", _json_object_request(), supports_temperature=True
    )

    assert payload["text"] == {"format": {"type": "json_object"}}


def test_anthropic_payload_carries_json_object_as_a_trailing_system_instruction() -> None:
    """Anthropic has no JSON mode: the caller's system prompt is preserved first and the
    JSON-object instruction is appended; no output_config schema is emitted."""
    payload = anthropic_messages_stream_payload("claude", _json_object_request())

    assert payload["system"] == "You are terse.\n\n" + JSON_OBJECT_SYSTEM_INSTRUCTION
    assert "output_config" not in payload
    assert payload["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]


def test_anthropic_payload_json_object_without_caller_system_prompt() -> None:
    """With no caller system turn, the instruction is the whole system prompt."""
    request = _json_object_request().model_copy(
        update={"messages": (GatewayMessage(role="user", content="hello"),)}
    )
    payload = anthropic_messages_stream_payload("claude", request)

    assert payload["system"] == JSON_OBJECT_SYSTEM_INSTRUCTION


def test_gemini_payload_requests_json_mime_type_without_a_schema() -> None:
    """Gemini json_object mode sets responseMimeType only; no empty responseJsonSchema."""
    payload = gemini_generate_content_stream_payload("gemini", _json_object_request())

    generation = payload["generationConfig"]
    assert isinstance(generation, dict)
    assert generation["responseMimeType"] == "application/json"
    assert "responseJsonSchema" not in generation


def test_bedrock_payload_carries_json_object_as_a_trailing_system_instruction() -> None:
    """Converse has no JSON mode: the instruction trails the caller's system blocks and no
    outputConfig textFormat schema is emitted."""
    payload = bedrock_converse_stream_payload("anthropic.claude", _json_object_request())

    assert payload["system"] == [
        {"text": "You are terse."},
        {"text": JSON_OBJECT_SYSTEM_INSTRUCTION},
    ]
    assert "outputConfig" not in payload


def test_json_object_mode_is_absent_from_every_dialect_without_the_flag() -> None:
    """A plain Chat request emits no JSON-object artifacts on any dialect."""
    request = _json_object_request().model_copy(update={"json_object_output": False})

    assert "response_format" not in openai_compatible_stream_payload("m", request)
    assert "text" not in openai_responses_stream_payload("m", request, supports_temperature=True)
    assert anthropic_messages_stream_payload("m", request)["system"] == "You are terse."
    gemini_generation = gemini_generate_content_stream_payload("m", request).get(
        "generationConfig", {}
    )
    assert isinstance(gemini_generation, dict)
    assert "responseMimeType" not in gemini_generation
    assert bedrock_converse_stream_payload("m", request)["system"] == [{"text": "You are terse."}]


def test_json_object_mode_is_admitted_on_every_supported_dialect() -> None:
    """Route admission keeps json_object off the strict-schema check: a mixed route of
    Anthropic, Gemini, Bedrock and OpenAI-compatible rungs is fully compatible and the
    provider request keeps the mode."""
    profiles = (
        GatewayWireProfile(dialect="anthropic_messages", url="https://a.test", model_id="claude"),
        GatewayWireProfile(dialect="gemini_generate_content", url="https://g.test", model_id="g"),
        GatewayWireProfile(dialect="bedrock_converse_stream", url="https://b.test", model_id="b"),
        GatewayWireProfile(dialect="openai_compatible", url="https://o.test", model_id="gpt"),
    )
    request = _json_object_request()

    assert compatible_generation_parameter_profile_indexes(profiles, request) == (0, 1, 2, 3)
    public_request, provider_request = route_generation_parameter_requests(profiles, request)
    assert public_request.json_object_output is True
    assert provider_request.json_object_output is True
    assert provider_request.structured_text is None
    for profile in profiles:
        assert dialect_stream_payload(profile, provider_request)


def test_json_object_mode_rejection_names_response_format_type() -> None:
    """A route with a dialect that cannot honor json_object fails on the public field the
    caller sent, never on an internal strict-schema path."""
    # Every implemented dialect honors json_object, so the profile validator has to be
    # bypassed to reach the defensive route check.
    unknown = GatewayWireProfile(dialect="openai_compatible", url="https://u.test", model_id="m")
    object.__setattr__(unknown, "dialect", "unknown_wire")

    with pytest.raises(ProviderParameterError) as raised:
        route_generation_parameter_requests((unknown,), _json_object_request())
    assert raised.value.param == "response_format.type"
    assert raised.value.code == "unsupported_parameter"
    assert "response_format.type" in str(raised.value)
    assert "strict" not in str(raised.value)


def test_openai_compatible_stream_payload_omits_absent_top_p() -> None:
    """An omitted nucleus parameter stays off the wire instead of being invented."""
    payload = openai_compatible_stream_payload("exact-model", _chat_request())

    assert "top_p" not in payload
    assert "temperature" not in payload


def test_openai_compatible_stream_payload_omits_unproven_controls() -> None:
    """Unknown compatible routes drop top-k and logprobs instead of guessing wire support."""
    request = _chat_request().model_copy(update={"top_k": 40, "logprobs": True, "top_logprobs": 5})
    payload = openai_compatible_stream_payload("exact-model", request)
    assert "top_k" not in payload
    assert "logprobs" not in payload
    assert "top_logprobs" not in payload


def test_openai_compatible_stream_payload_ignores_logprobs_even_when_flagged() -> None:
    """Logprob controls stay off the wire until normalized output can preserve them."""
    request = _chat_request().model_copy(update={"top_k": 40, "logprobs": True, "top_logprobs": 5})
    payload = openai_compatible_stream_payload(
        "exact-model",
        request,
        supports_top_k=True,
        supports_logprobs=True,
    )
    assert payload["top_k"] == 40
    assert "logprobs" not in payload
    assert "top_logprobs" not in payload


def test_anthropic_stream_payload_omits_logprobs_even_when_flagged() -> None:
    """Anthropic's native Messages lane never receives an OpenAI logprob field."""
    request = _chat_request().model_copy(update={"logprobs": True, "top_logprobs": 5})
    payload = anthropic_messages_stream_payload(
        "claude-sonnet-5",
        request,
        supports_logprobs=True,
    )
    assert "logprobs" not in payload
    assert "top_logprobs" not in payload


def _budgeted_haiku_profile() -> GatewayWireProfile:
    """Return the Anthropic budgeted-enabled reasoning profile for haiku-4-5."""
    return GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://api.anthropic.com/v1/messages",
        model_id="claude-haiku-4-5",
        supports_temperature=True,
        supports_reasoning=True,
        reasoning_wire_format="anthropic_adaptive",
        sampling_requires_reasoning_none=True,
        maximum_output_tokens=64_000,
    )


def test_haiku_multi_turn_budgeted_thinking_is_honored_end_to_end() -> None:
    """A caller budget config plus replayed thinking blocks survive verbatim."""
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(
            GatewayMessage(role="user", content="hi"),
            GatewayMessage(
                role="assistant",
                content="prior answer",
                provider_reasoning=(ThinkingBlock(text="prior reasoning", signature="sig-1"),),
            ),
            GatewayMessage(role="user", content="again"),
        ),
        provider_thinking_config={"type": "enabled", "budget_tokens": 2_048},
        maximum_output_tokens=8_000,
        stream=True,
    )
    payload = anthropic_messages_stream_payload(
        "claude-haiku-4-5",
        request,
        supports_temperature=True,
        supports_reasoning=True,
    )
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 2_048}
    messages = payload["messages"]
    assert isinstance(messages, list)
    assistant = next(message for message in messages if message["role"] == "assistant")
    content = assistant["content"]
    assert isinstance(content, list)
    # The signed thinking block re-emits byte-exact, leading the assistant turn.
    assert content[0] == {"type": "thinking", "thinking": "prior reasoning", "signature": "sig-1"}


def test_haiku_thinking_off_temperature_is_honored_under_the_srn_hatch() -> None:
    """With thinking off (no config), haiku's srn hatch keeps temperature.

    haiku has no "none" on its effort ladder, so the srn hatch opens on the
    absence of any thinking budget instead — thinking-off traffic that sets
    temperature keeps it rather than regressing to a silent drop.
    """
    profile = _budgeted_haiku_profile()
    honored = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="hello"),),
        temperature=0.5,
        maximum_output_tokens=1_000,
        stream=True,
    )
    _public, provider = route_generation_parameter_requests((profile,), honored)
    assert provider.temperature == 0.5
    payload = anthropic_messages_stream_payload(
        "claude-haiku-4-5",
        provider,
        supports_temperature=True,
        supports_reasoning=True,
    )
    assert payload["temperature"] == 0.5

    # Contrast: with thinking ENABLED, Anthropic requires temperature 1, so the
    # same control is a disclosed drop, not a rejection — srn is per request.
    thinking_on = honored.model_copy(
        update={"provider_thinking_config": {"type": "enabled", "budget_tokens": 2_048}}
    )
    public_on, provider_on = route_generation_parameter_requests((profile,), thinking_on)
    assert provider_on.temperature is None
    assert "temperature->dropped(set_reasoning_effort_none)" in public_on.ignored_parameters


def test_haiku_bare_effort_realizes_as_a_token_budget_at_the_payload_seam() -> None:
    """An effort on a budgeted-enabled-only model emits enabled+budget, never adaptive.

    haiku rejects ``thinking.type: adaptive`` and ``output_config.effort`` by
    name; an effort reaching the build is the caller's depth intent (or the
    route's pinned default), realized as the derived token budget — the same
    wire realization the effort generation gets via the adaptive object.
    """
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="hello"),),
        reasoning_effort="medium",
        maximum_output_tokens=8_000,
        stream=True,
    )
    payload = anthropic_messages_stream_payload(
        "claude-haiku-4-5",
        request,
        supports_reasoning=True,
    )
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 4_000}
    assert "output_config" not in payload

    # The route's pinned catalog default (no caller effort) realizes the same way.
    pinned = anthropic_messages_stream_payload(
        "claude-haiku-4-5",
        request.model_copy(update={"reasoning_effort": None}),
        supports_reasoning=True,
        reasoning_effort="medium",
    )
    assert pinned["thinking"] == {"type": "enabled", "budget_tokens": 4_000}
    assert "output_config" not in pinned

    # Effort "none" keeps thinking off entirely.
    off = anthropic_messages_stream_payload(
        "claude-haiku-4-5",
        request.model_copy(update={"reasoning_effort": "none"}),
        supports_reasoning=True,
    )
    assert "thinking" not in off

    # A ceiling too small for any legal budget keeps thinking off rather than
    # emitting an illegal budget the provider would reject.
    tight = anthropic_messages_stream_payload(
        "claude-haiku-4-5",
        request.model_copy(update={"maximum_output_tokens": 1_024}),
        supports_reasoning=True,
    )
    assert "thinking" not in tight


def test_haiku_caller_output_config_effort_is_stripped_at_the_payload_seam() -> None:
    """A caller-seeded output_config.effort never reaches a budgeted-only model."""
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="hello"),),
        reasoning_effort="medium",
        provider_output_config={"effort": "medium", "format": {"type": "text"}},
        maximum_output_tokens=8_000,
        stream=True,
    )
    payload = anthropic_messages_stream_payload(
        "claude-haiku-4-5",
        request,
        supports_reasoning=True,
    )
    # The depth intent rides the budget; the by-name-rejected key is gone and
    # unrelated output_config keys survive verbatim.
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 4_000}
    assert payload["output_config"] == {"format": {"type": "text"}}

    # An UNRECOGNIZED effort never mapped to reasoning_effort, so it has no
    # budget realization; it stays verbatim and the provider's own by-name
    # rejection is the honest outcome, never a silent thinking-off answer.
    future = request.model_copy(
        update={
            "reasoning_effort": None,
            "provider_output_config": {"effort": "hyperdrive"},
        }
    )
    payload = anthropic_messages_stream_payload(
        "claude-haiku-4-5",
        future,
        supports_reasoning=True,
    )
    assert "thinking" not in payload
    assert payload["output_config"] == {"effort": "hyperdrive"}


def test_haiku_adaptive_config_is_rejected_by_name_before_dispatch() -> None:
    """An adaptive config on a budgeted-only route raises pre-dispatch.

    The named rejection is what lets the admit loop offer the disclosed
    adaptive->enabled(budget) coercion; forwarding verbatim would surface the
    provider's own opaque 400 instead (which never fails over).
    """
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="hello"),),
        provider_thinking_config={"type": "adaptive"},
        maximum_output_tokens=8_000,
        stream=True,
    )
    with pytest.raises(ProviderParameterError) as excinfo:
        route_generation_parameter_requests((_budgeted_haiku_profile(),), request)
    assert excinfo.value.param == "thinking.type"


def test_openai_compatible_stream_payload_omits_reasoning_without_route_capability() -> None:
    """A compatible route never receives a reasoning field without explicit capability proof."""
    request = _chat_request().model_copy(update={"reasoning_effort": "high"})

    payload = openai_compatible_stream_payload(
        "cloud-opus-5",
        request,
        reasoning_effort="medium",
    )

    assert "reasoning_effort" not in payload


def test_openai_compatible_stream_payload_forwards_reasoning_with_route_capability() -> None:
    """A compatible route receives the requested effort only when its row opts in."""
    request = _chat_request().model_copy(update={"reasoning_effort": "high"})

    payload = openai_compatible_stream_payload(
        "reasoning-model",
        request,
        supports_reasoning=True,
        reasoning_effort="medium",
    )

    assert payload["reasoning_effort"] == "high"


def test_openai_compatible_stream_payload_honors_token_limit_key() -> None:
    """The output-token ceiling lands on the configured wire field only."""
    request = _chat_request().model_copy(update={"maximum_output_tokens": 64})

    default_payload = openai_compatible_stream_payload("exact-model", request)
    azure_payload = openai_compatible_stream_payload(
        "exact-model", request, token_limit_key="max_completion_tokens"
    )

    assert default_payload["max_tokens"] == 64
    assert "max_completion_tokens" not in default_payload
    assert azure_payload["max_completion_tokens"] == 64
    assert "max_tokens" not in azure_payload


def test_openai_responses_stream_payload_forwards_top_p_when_sampling_is_open() -> None:
    """Native Responses streaming preserves nucleus sampling on models that accept it."""
    payload = openai_responses_stream_payload(
        "exact-model",
        _chat_request(top_p=1.0, temperature=0.2),
        supports_temperature=True,
        supports_reasoning=False,
        reasoning_effort=None,
    )

    assert payload["stream"] is True
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 1.0


def test_openai_responses_stream_payload_omits_top_p_when_sampling_is_pinned() -> None:
    """Pinned-sampling Responses models omit caller top_p instead of failing the request."""
    payload = openai_responses_stream_payload(
        "exact-model",
        _chat_request(top_p=1.0),
        supports_temperature=False,
        supports_reasoning=True,
        reasoning_effort="xhigh",
    )
    assert "top_p" not in payload


def test_openai_responses_stream_payload_omits_absent_top_p() -> None:
    """Native Responses streaming does not invent a nucleus-sampling value."""
    payload = openai_responses_stream_payload(
        "exact-model",
        _chat_request(),
        supports_temperature=True,
        supports_reasoning=False,
        reasoning_effort=None,
    )

    assert "top_p" not in payload


def test_openai_responses_stream_payload_omits_reasoning_without_route_capability() -> None:
    """An unsupported native route drops a caller reasoning request before upstream dispatch."""
    request = _chat_request().model_copy(update={"reasoning_effort": "high"})

    payload = openai_responses_stream_payload(
        "cloud-opus-5",
        request,
        supports_temperature=True,
        reasoning_effort="medium",
    )

    assert "reasoning" not in payload


def test_openai_responses_stream_payload_forwards_reasoning_with_route_capability() -> None:
    """A verified native Responses route receives the caller's requested effort."""
    request = _chat_request().model_copy(update={"reasoning_effort": "high"})

    payload = openai_responses_stream_payload(
        "gpt-5.6-luna",
        request,
        supports_temperature=False,
        supports_reasoning=True,
        reasoning_effort="medium",
    )

    assert payload["reasoning"] == {"effort": "high"}


def test_openai_responses_stream_payload_forwards_reasoning_summary() -> None:
    """A native reasoning route receives the current Responses summary selector."""
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="hello"),),
        reasoning_summary="detailed",
        reasoning_summary_parameters=("reasoning.generate_summary",),
    )

    payload = openai_responses_stream_payload(
        "gpt-5.6-luna",
        request,
        supports_temperature=False,
        supports_reasoning=True,
    )

    assert payload["reasoning"] == {"summary": "detailed"}


def test_route_rejects_reasoning_summary_outside_native_responses() -> None:
    """A fallback without summary output support rejects the exact caller alias."""
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="hello"),),
        reasoning_summary="concise",
        reasoning_summary_parameters=(
            "reasoning.generate_summary",
            "reasoning.summary",
        ),
    )
    profiles = (
        GatewayWireProfile(
            dialect="openai_responses",
            url="https://openai.test",
            supports_reasoning=True,
            reasoning_wire_format="openai_responses",
        ),
        GatewayWireProfile(
            dialect="openai_compatible",
            url="https://fallback.test",
            supports_reasoning=True,
            reasoning_wire_format="reasoning",
        ),
    )

    with pytest.raises(ProviderParameterError) as raised:
        route_generation_parameter_requests(profiles, request)

    assert raised.value.code == "unsupported_parameter"
    assert raised.value.param == "reasoning.generate_summary"


def _tool_image_message() -> GatewayMessage:
    """One tool message carrying a screenshot beside its text."""
    return GatewayMessage(
        role="tool",
        tool_call_id="call-1",
        content="tool said:",
        content_parts=(
            TextContentPart(text="tool said:"),
            ImageContentPart(media_type="image/png", data="aGk="),
        ),
    )


def test_a_mixed_route_degrades_tool_result_images_with_disclosure() -> None:
    """A non-Anthropic rung cannot express a tool-result image, so the route
    substitutes positional placeholder text and discloses the drop instead of
    rejecting a block the caller cannot remove from history."""
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="go"), _tool_image_message()),
    )
    profiles = (
        GatewayWireProfile(dialect="anthropic_messages", url="https://a.test"),
        GatewayWireProfile(dialect="openai_compatible", url="https://b.test"),
    )

    public_request, provider_request = route_generation_parameter_requests(profiles, request)

    tool_message = provider_request.messages[-1]
    assert tool_message.content_parts == ()
    assert tool_message.content == "tool said:" + TOOL_RESULT_IMAGE_PLACEHOLDER
    assert TOOL_RESULT_IMAGE_DROP_DISCLOSURE in public_request.ignored_parameters
    # The public request keeps the caller's original history.
    assert public_request.messages[-1].images


def test_an_all_anthropic_route_keeps_tool_result_images() -> None:
    """Single-dialect Anthropic routes carry the screenshot verbatim."""
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="go"), _tool_image_message()),
    )
    profiles = (GatewayWireProfile(dialect="anthropic_messages", url="https://a.test"),)

    public_request, provider_request = route_generation_parameter_requests(profiles, request)

    assert provider_request.messages[-1].images
    assert TOOL_RESULT_IMAGE_DROP_DISCLOSURE not in public_request.ignored_parameters


def test_route_accepts_reasoning_summary_on_native_anthropic() -> None:
    """Anthropic thinking reaches the summary channel, so the route serves the field."""
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="hello"),),
        reasoning_summary="auto",
        reasoning_summary_parameters=("reasoning.summary",),
    )
    profiles = (
        GatewayWireProfile(
            dialect="anthropic_messages",
            url="https://anthropic.test",
            model_id="claude-opus-5",
            supports_reasoning=True,
            reasoning_wire_format="anthropic_adaptive",
        ),
    )

    public_request, provider_request = route_generation_parameter_requests(profiles, request)

    assert public_request.reasoning_summary == "auto"
    assert provider_request.reasoning_summary == "auto"
    assert "reasoning" not in anthropic_messages_stream_payload(
        "claude-opus-5",
        provider_request,
        supports_reasoning=True,
    )


def test_reasoning_summary_narrows_a_mixed_claude_waterfall() -> None:
    """A Claude route serves the summary on Anthropic instead of failing on its fallback."""
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="hello"),),
        reasoning_summary="auto",
        reasoning_summary_parameters=("reasoning.summary",),
    )
    profiles = (
        GatewayWireProfile(
            dialect="anthropic_messages",
            url="https://anthropic.test",
            model_id="claude-opus-5",
            supports_reasoning=True,
            reasoning_wire_format="anthropic_adaptive",
        ),
        GatewayWireProfile(
            dialect="openai_compatible",
            url="https://openrouter.test",
            model_id="anthropic/claude-opus-5",
            supports_reasoning=True,
            reasoning_wire_format="reasoning",
        ),
    )

    assert compatible_generation_parameter_profile_indexes(profiles, request) == (0,)


@pytest.mark.parametrize(
    ("field", "value"),
    [("temperature", 0.2), ("top_p", 0.8)],
)
def test_route_generation_controls_use_the_whole_waterfall_intersection(
    field: str, value: float | int
) -> None:
    """One incompatible fallback drops an explicit sampling control with disclosure.

    temperature/top_p are genuinely unsupported on the fallback (declared False):
    the control is dropped route-wide and disclosed, never a 400 (the model
    still answers with its own default). A value outside a SUPPORTING route's
    range still rejects (covered separately).
    """
    request = _chat_request().model_copy(update={field: value})
    profiles = (
        GatewayWireProfile(
            dialect="openai_compatible",
            url="https://first.test",
            model_id="openai/gpt-5",
            supports_temperature=True,
            supports_top_p=True,
            supports_top_k=True,
            supports_reasoning=True,
            reasoning_wire_format="reasoning",
        ),
        GatewayWireProfile(
            dialect="anthropic_messages",
            url="https://fallback.test",
            model_id="claude-sonnet-4-6",
            supports_temperature=False,
            supports_top_p=False,
            supports_top_k=False,
            supports_reasoning=True,
            reasoning_wire_format="anthropic_adaptive",
        ),
    )

    public_request, provider_request = route_generation_parameter_requests(profiles, request)

    assert getattr(provider_request, field) is None
    assert f"{field}->dropped(unsupported_by_provider)" in public_request.ignored_parameters


def test_top_k_narrows_to_a_supporting_rung_then_drops_when_none_support() -> None:
    """top_k prefers a rung that carries it, and is dropped+disclosed (not rejected)
    only when no rung on the committed route accepts it."""
    supporting = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://first.test",
        model_id="provider/top-k",
        supports_top_k=True,
        minimum_top_k=1,
        maximum_top_k=100,
    )
    unsupporting = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://azure.test",
        model_id="provider/no-top-k",
        supports_top_k=False,
    )
    request = _chat_request().model_copy(update={"top_k": 20})

    # Selection prefers the rung that honors top_k.
    assert compatible_generation_parameter_profile_indexes((supporting, unsupporting), request) == (
        0,
    )

    # A committed route with no supporting rung serves by dropping+disclosing.
    public_request, provider_request = route_generation_parameter_requests((unsupporting,), request)
    assert provider_request.top_k is None
    assert public_request.ignored_parameters == ("top_k->dropped(unsupported_by_provider)",)

    # A route where every rung supports it honors the value.
    _public, honored = route_generation_parameter_requests((supporting,), request)
    assert honored.top_k == 20


def test_penalties_honor_where_supported_and_drop_with_disclosure_otherwise() -> None:
    """Sampling penalties are honored (emitted) where every rung supports them, and
    dropped+disclosed (a soft preference) where a rung does not — never rejected."""
    supporting = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://p.test",
        model_id="m",
        supports_frequency_penalty=True,
        supports_presence_penalty=True,
    )
    unsupporting = GatewayWireProfile(
        dialect="openai_compatible", url="https://q.test", model_id="m2"
    )
    request = _chat_request().model_copy(
        update={"frequency_penalty": 0.5, "presence_penalty": -0.3}
    )

    # Every rung supports them → honored and emitted on the payload.
    public_all, provider_all = route_generation_parameter_requests((supporting,), request)
    assert public_all.ignored_parameters == ()
    payload = openai_compatible_stream_payload(
        "m",
        provider_all,
        supports_frequency_penalty=True,
        supports_presence_penalty=True,
    )
    assert payload["frequency_penalty"] == 0.5
    assert payload["presence_penalty"] == -0.3

    # A rung without support → dropped + disclosed, never rejected.
    public_drop, provider_drop = route_generation_parameter_requests(
        (supporting, unsupporting), request
    )
    assert provider_drop.frequency_penalty is None
    assert provider_drop.presence_penalty is None
    assert public_drop.ignored_parameters == (
        "frequency_penalty->dropped(unsupported_by_provider)",
        "presence_penalty->dropped(unsupported_by_provider)",
    )


def test_penalty_flag_on_a_non_emitting_dialect_still_drops_and_discloses() -> None:
    """A supports_*_penalty flag stamped on a dialect that cannot EMIT penalties
    (only openai_compatible does) must NOT claim honored — it drops+discloses, never a
    silent undisclosed omission (a catalog could mis-stamp e.g. an openai_responses rung)."""
    responses_rung = GatewayWireProfile(
        dialect="openai_responses",
        url="https://r.test",
        model_id="gpt-5",
        supports_frequency_penalty=True,
        supports_presence_penalty=True,
    )
    request = _chat_request().model_copy(
        update={"frequency_penalty": 0.5, "presence_penalty": -0.2}
    )

    public, provider = route_generation_parameter_requests((responses_rung,), request)

    assert provider.frequency_penalty is None
    assert provider.presence_penalty is None
    assert public.ignored_parameters == (
        "frequency_penalty->dropped(unsupported_by_provider)",
        "presence_penalty->dropped(unsupported_by_provider)",
    )


def test_route_rejects_effort_not_preserved_by_the_whole_waterfall() -> None:
    """An exact effort mismatch fails locally instead of clamping on a fallback."""
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="hello"),),
        reasoning_effort="minimal",
    )
    profiles = (
        GatewayWireProfile(
            dialect="openai_responses",
            url="https://first.test",
            model_id="gpt-5",
            supports_reasoning=True,
            reasoning_wire_format="openai_responses",
        ),
        GatewayWireProfile(
            dialect="openai_responses",
            url="https://fallback.test",
            model_id="gpt-5-pro",
            supports_reasoning=True,
            reasoning_wire_format="openai_responses",
        ),
    )

    with pytest.raises(UnsupportedReasoningEffortError) as raised:
        route_generation_parameter_requests(profiles, request)

    assert raised.value.param == "reasoning.effort"
    assert str(raised.value) == (
        "The effort parameter value 'minimal' is not supported by this model route. "
        "Supported values: 'high'."
    )


def test_route_rejects_effort_when_reasoning_is_unsupported() -> None:
    """Reasoning effort is semantic and is never silently dropped."""
    request = _chat_request().model_copy(update={"reasoning_effort": "high"})

    with pytest.raises(UnsupportedReasoningEffortError) as raised:
        route_generation_parameter_requests(
            (
                GatewayWireProfile(
                    dialect="openai_compatible",
                    url="https://provider.test",
                ),
            ),
            request,
        )

    assert raised.value.param == "reasoning_effort"
    assert "not supported by this model route" in str(raised.value)


def test_route_rejects_an_unsupported_configured_effort_without_clamping() -> None:
    """An invalid operator pin fails locally even when the caller omits effort."""
    with pytest.raises(UnsupportedReasoningEffortError) as raised:
        route_generation_parameter_requests(
            (
                GatewayWireProfile(
                    dialect="openai_responses",
                    url="https://provider.test",
                    model_id="gpt-5.6-sol",
                    supports_reasoning=True,
                    reasoning_wire_format="openai_responses",
                    reasoning_effort="ultra",
                    reasoning_effort_required=True,
                ),
            ),
            _chat_request(),
        )

    assert raised.value.param == "reasoning_effort"
    assert "Supported values: 'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'" in str(
        raised.value
    )


def test_route_preserves_each_required_reasoning_default_across_fallbacks() -> None:
    """An omitted caller effort is injected independently on each required wire."""
    profiles = (
        GatewayWireProfile(
            dialect="openai_compatible",
            url="https://first.test",
            model_id="provider/first",
            supports_reasoning=True,
            reasoning_wire_format="reasoning",
            reasoning_effort="high",
            supported_reasoning_efforts=("medium", "high"),
            reasoning_effort_required=True,
        ),
        GatewayWireProfile(
            dialect="openai_compatible",
            url="https://fallback.test",
            model_id="provider/fallback",
            supports_reasoning=True,
            reasoning_wire_format="reasoning",
            reasoning_effort="medium",
            supported_reasoning_efforts=("medium", "high"),
            reasoning_effort_required=True,
        ),
    )

    public, provider = route_generation_parameter_requests(profiles, _chat_request())

    assert public.reasoning_effort is None
    assert provider.reasoning_effort is None
    assert dialect_stream_payload(profiles[0], provider)["reasoning"] == {"effort": "high"}
    assert dialect_stream_payload(profiles[1], provider)["reasoning"] == {"effort": "medium"}


def test_explicit_provider_effort_set_accepts_exact_values_and_rejects_others() -> None:
    """Catalog-published OpenRouter values replace the unknown-family singleton."""
    profile = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://openrouter.test",
        model_id="z-ai/glm-5.3",
        supports_reasoning=True,
        reasoning_wire_format="reasoning",
        reasoning_effort="max",
        supported_reasoning_efforts=("low", "high", "max"),
        reasoning_effort_required=True,
    )

    _public, provider = route_generation_parameter_requests(
        (profile,),
        _chat_request().model_copy(update={"reasoning_effort": "high"}),
    )
    assert provider.reasoning_effort == "high"

    with pytest.raises(UnsupportedReasoningEffortError) as raised:
        route_generation_parameter_requests(
            (profile,),
            _chat_request().model_copy(update={"reasoning_effort": "medium"}),
        )
    assert "Supported values: 'low', 'high', 'max'." in str(raised.value)


def test_required_reasoning_default_is_sent_but_optional_default_is_not() -> None:
    """Only a provider-mandated default may add an omitted reasoning field."""
    required = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://required.test",
        model_id="z-ai/glm-5.3",
        supports_reasoning=True,
        reasoning_wire_format="reasoning",
        reasoning_effort="max",
        supported_reasoning_efforts=("low", "high", "max"),
        reasoning_effort_required=True,
    )
    optional = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://optional.test",
        model_id="deepseek/deepseek-v4-flash",
        supports_reasoning=True,
        reasoning_wire_format="reasoning",
        reasoning_effort="high",
        supported_reasoning_efforts=("high", "xhigh"),
    )

    public, provider = route_generation_parameter_requests((required,), _chat_request())
    assert public.reasoning_effort is None
    assert provider.reasoning_effort is None
    assert dialect_stream_payload(required, provider)["reasoning"] == {"effort": "max"}
    optional_public, optional_provider = route_generation_parameter_requests(
        (optional,), _chat_request()
    )
    assert optional_public.reasoning_effort is None
    assert optional_provider.reasoning_effort is None
    assert "reasoning" not in dialect_stream_payload(optional, optional_provider)

    mixed_public, mixed_provider = route_generation_parameter_requests(
        (required, optional), _chat_request()
    )
    assert mixed_public.reasoning_effort is None
    assert mixed_provider.reasoning_effort is None
    assert dialect_stream_payload(required, mixed_provider)["reasoning"] == {"effort": "max"}
    assert "reasoning" not in dialect_stream_payload(optional, mixed_provider)


def test_generation_parameter_selection_skips_incompatible_fallbacks() -> None:
    """An exact caller value keeps usable rungs instead of poisoning the waterfall."""
    profiles = (
        GatewayWireProfile(
            dialect="openai_compatible",
            url="https://azure.test",
            model_id="DeepSeek-V4-Flash",
        ),
        GatewayWireProfile(
            dialect="openai_compatible",
            url="https://openrouter.test",
            model_id="deepseek/deepseek-v4-flash",
            supports_reasoning=True,
            reasoning_wire_format="reasoning",
            supported_reasoning_efforts=("high", "xhigh"),
        ),
    )
    request = _chat_request().model_copy(update={"reasoning_effort": "high"})

    assert compatible_generation_parameter_profile_indexes(profiles, request) == (1,)


def test_generation_parameter_selection_accepts_different_required_defaults() -> None:
    """Omitted effort keeps independently valid rungs with different wire defaults."""
    profiles = (
        GatewayWireProfile(
            dialect="openai_compatible",
            url="https://first.test",
            model_id="provider/first",
            supports_reasoning=True,
            reasoning_wire_format="reasoning",
            reasoning_effort="high",
            supported_reasoning_efforts=("medium", "high"),
            reasoning_effort_required=True,
        ),
        GatewayWireProfile(
            dialect="openai_compatible",
            url="https://fallback.test",
            model_id="provider/fallback",
            supports_reasoning=True,
            reasoning_wire_format="reasoning",
            reasoning_effort="medium",
            supported_reasoning_efforts=("medium", "high"),
            reasoning_effort_required=True,
        ),
    )

    assert compatible_generation_parameter_profile_indexes(profiles, _chat_request()) == (0, 1)


def test_generation_parameter_selection_uses_each_default_for_sampling() -> None:
    """Conditional sampling narrows against each required default independently."""
    profiles = (
        GatewayWireProfile(
            dialect="openai_compatible",
            url="https://reasoning.test",
            model_id="provider/reasoning",
            supports_reasoning=True,
            reasoning_wire_format="reasoning",
            reasoning_effort="high",
            supported_reasoning_efforts=("none", "high"),
            reasoning_effort_required=True,
            sampling_requires_reasoning_none=True,
        ),
        GatewayWireProfile(
            dialect="openai_compatible",
            url="https://sampling.test",
            model_id="provider/sampling",
            supports_reasoning=True,
            reasoning_wire_format="reasoning",
            reasoning_effort="none",
            supported_reasoning_efforts=("none", "high"),
            reasoning_effort_required=True,
            sampling_requires_reasoning_none=True,
        ),
    )
    request = _chat_request().model_copy(update={"temperature": 0.5})

    assert compatible_generation_parameter_profile_indexes(profiles, request) == (1,)


def test_generation_parameter_selection_serves_with_drop_when_no_rung_honors() -> None:
    """When every rung is a reasoning route that blocks sampling at this effort, all
    rungs stay selectable (they serve by dropping+disclosing), not rejected."""
    profiles = tuple(
        GatewayWireProfile(
            dialect="openai_compatible",
            url=f"https://reasoning{position}.test",
            model_id=f"provider/reasoning-{position}",
            supports_reasoning=True,
            reasoning_wire_format="reasoning",
            reasoning_effort="high",
            supported_reasoning_efforts=("none", "high"),
            reasoning_effort_required=True,
            sampling_requires_reasoning_none=True,
        )
        for position in range(2)
    )
    request = _chat_request().model_copy(update={"temperature": 0.5})

    # No rung honors temperature at effort=high, so both remain serviceable and the
    # value is dropped+disclosed on whichever serves (rather than a hard rejection).
    assert compatible_generation_parameter_profile_indexes(profiles, request) == (0, 1)


def test_non_strict_json_schema_narrows_away_from_a_schema_closing_rung() -> None:
    """A real non-strict json_schema narrows to a rung that serves open schemas, and
    rejects on the strict path only when every rung is a schema-closing (Anthropic)
    dialect. This check is reserved for actual schemas; json_object mode never reaches it."""
    open_request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hi"),),
        structured_text=StructuredTextFormat(
            name="answer", json_schema={"type": "object"}, strict=False
        ),
    )
    anthropic = GatewayWireProfile(
        dialect="anthropic_messages", url="https://a.test", model_id="claude"
    )
    openai = GatewayWireProfile(dialect="openai_compatible", url="https://o.test", model_id="gpt")

    # Narrows away from the Anthropic rung to the open-capable one.
    assert compatible_generation_parameter_profile_indexes((anthropic, openai), open_request) == (
        1,
    )

    # No open-capable rung → reject, not a silent inversion.
    with pytest.raises(ProviderParameterError) as raised:
        route_generation_parameter_requests((anthropic,), open_request)
    assert raised.value.param == "response_format.json_schema.strict"
    assert raised.value.code == "unsupported_parameter"

    # A STRICT schema on the same Anthropic rung is NOT narrowed away (it is closed by
    # the #733 coercion downstream instead), so it stays compatible.
    strict_request = open_request.model_copy(
        update={
            "structured_text": StructuredTextFormat(
                name="answer", json_schema={"type": "object"}, strict=True
            )
        }
    )
    assert compatible_generation_parameter_profile_indexes((anthropic,), strict_request) == (0,)


def test_route_accepts_anthropic_max_effort_without_translation() -> None:
    """The provider's documented max level is preserved exactly."""
    request = _chat_request().model_copy(update={"reasoning_effort": "max"})
    profile = GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://provider.test",
        model_id="claude-opus-5",
        supports_reasoning=True,
        reasoning_wire_format="anthropic_adaptive",
    )

    public_request, provider_request = route_generation_parameter_requests((profile,), request)

    assert public_request.reasoning_effort == "max"
    assert provider_request.reasoning_effort == "max"


def test_route_shaping_omits_tool_controls_when_no_tools_exist() -> None:
    """No-op tool selectors never reach providers that reject them without schemas."""
    request = _chat_request().model_copy(
        update={"tool_choice": "none", "parallel_tool_calls": False}
    )

    public_request, provider_request = route_generation_parameter_requests(
        (GatewayWireProfile(dialect="openai_compatible", url="https://provider.test"),),
        request,
    )

    assert public_request.ignored_parameters == ("tool_choice", "parallel_tool_calls")
    assert provider_request.tool_choice is None
    assert provider_request.parallel_tool_calls is None


def test_mixed_route_keeps_the_prompt_cache_marker_when_any_rung_is_anthropic() -> None:
    """The cache marker survives a mixed waterfall so the winning Anthropic rung
    still caches; only an all-non-Anthropic route drops it. Dropping it the
    moment one fallback rung was non-Anthropic billed every turn's full context
    uncached (~10x on a large system prompt)."""
    request = _chat_request().model_copy(update={"provider_cache_control": {"type": "ephemeral"}})
    anthropic = GatewayWireProfile(
        dialect="anthropic_messages", url="https://a.test", model_id="claude-fable-5"
    )
    fallback = GatewayWireProfile(dialect="openai_compatible", url="https://b.test")

    # Mixed route with an Anthropic rung: kept, not disclosed.
    public_request, provider_request = route_generation_parameter_requests(
        (anthropic, fallback), request
    )
    assert provider_request.provider_cache_control == {"type": "ephemeral"}
    assert "cache_control" not in public_request.ignored_parameters

    # No rung can cache: dropped with disclosure.
    public_only, provider_only = route_generation_parameter_requests((fallback,), request)
    assert provider_only.provider_cache_control is None
    assert "cache_control" in public_only.ignored_parameters


def test_route_shaping_omits_parallel_control_when_tool_choice_disables_tools() -> None:
    """A parallel selector cannot affect a turn whose tool choice is none."""
    request = _chat_request().model_copy(
        update={
            "tools": (GatewayToolDefinition(name="search", parameters={"type": "object"}),),
            "tool_choice": "none",
            "parallel_tool_calls": False,
        }
    )

    public_request, provider_request = route_generation_parameter_requests(
        (GatewayWireProfile(dialect="gemini_generate_content", url="https://provider.test"),),
        request,
    )

    assert public_request.ignored_parameters == ("parallel_tool_calls",)
    assert provider_request.parallel_tool_calls is None


@pytest.mark.parametrize(
    "dialect",
    ("gemini_generate_content", "bedrock_converse_stream"),
)
def test_route_emulates_parallel_control_when_the_dialect_has_no_toggle(dialect: str) -> None:
    """A wire without a parallel-tool control never loses the caller's semantics.

    `false` is honoured by the data plane (one tool call per turn) and
    disclosed; `true` is the provider's own default and is dropped with its
    own disclosure. Neither is a 400 any more.
    """
    tools = (GatewayToolDefinition(name="search", parameters={"type": "object"}),)
    profiles = (GatewayWireProfile(dialect=dialect, url="https://provider.test"),)

    sequential = _chat_request().model_copy(update={"tools": tools, "parallel_tool_calls": False})
    public_request, provider_request = route_generation_parameter_requests(profiles, sequential)
    assert provider_request.parallel_tool_calls is None
    assert provider_request.serialize_tool_calls is True
    assert (
        "parallel_tool_calls->emulated(serialized_by_gateway)" in public_request.ignored_parameters
    )

    parallel = _chat_request().model_copy(update={"tools": tools, "parallel_tool_calls": True})
    public_request, provider_request = route_generation_parameter_requests(profiles, parallel)
    assert provider_request.parallel_tool_calls is None
    assert provider_request.serialize_tool_calls is False
    assert "parallel_tool_calls->dropped(provider_default)" in public_request.ignored_parameters


def test_mixed_route_keeps_parallel_control_for_the_rungs_that_carry_it() -> None:
    """Route-wide shaping only fires when NO rung has the control.

    With one native rung and one toggle-less rung the field stays on the
    provider request: the native rung forwards it verbatim and the toggle-less
    rung is shaped per rung at dispatch, so a capable deployment never loses
    the caller's native control to another rung's limitation.
    """
    tools = (GatewayToolDefinition(name="search", parameters={"type": "object"}),)
    profiles = (
        GatewayWireProfile(dialect="openai_compatible", url="https://native.test"),
        GatewayWireProfile(dialect="openai_responses", url="https://toggle-less.test"),
    )
    sequential = _chat_request().model_copy(update={"tools": tools, "parallel_tool_calls": False})
    public_request, provider_request = route_generation_parameter_requests(profiles, sequential)
    assert provider_request.parallel_tool_calls is False
    assert provider_request.serialize_tool_calls is False
    assert not any("parallel_tool_calls" in note for note in public_request.ignored_parameters)


def test_route_rejects_non_strict_schema_on_a_strict_only_provider() -> None:
    """Constrained decoding cannot silently strengthen a non-strict request."""
    request = _chat_request().model_copy(
        update={
            "structured_text": StructuredTextFormat(
                name="answer",
                json_schema={"type": "object"},
                strict=False,
            )
        }
    )

    with pytest.raises(ProviderParameterError) as raised:
        route_generation_parameter_requests(
            (
                GatewayWireProfile(
                    dialect="anthropic_messages",
                    url="https://provider.test",
                ),
            ),
            request,
        )

    assert raised.value.code == "unsupported_parameter"
    assert raised.value.param == "response_format.json_schema.strict"


def test_route_rejects_semantic_tool_selectors_without_a_matching_schema() -> None:
    """Required and named choices cannot be reduced to harmless no-ops."""
    profile = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://provider.test",
    )
    with pytest.raises(ProviderParameterError) as required:
        route_generation_parameter_requests(
            (profile,),
            _chat_request().model_copy(update={"tool_choice": "required"}),
        )
    assert required.value.code == "invalid_parameter"
    assert required.value.param == "tool_choice"

    with pytest.raises(ProviderParameterError) as named:
        route_generation_parameter_requests(
            (profile,),
            _chat_request().model_copy(
                update={
                    "tools": (GatewayToolDefinition(name="search", parameters={"type": "object"}),),
                    "tool_choice": GatewayNamedToolChoice(name="missing"),
                }
            ),
        )
    assert named.value.code == "invalid_parameter"
    assert named.value.param == "tool_choice"


def test_route_rejects_out_of_range_provider_controls() -> None:
    """Provider-specific ranges produce a local field-specific error."""
    request = _chat_request(temperature=1.5)

    with pytest.raises(ProviderParameterError) as raised:
        route_generation_parameter_requests(
            (
                GatewayWireProfile(
                    dialect="gemini_generate_content",
                    url="https://provider.test",
                    maximum_temperature=1.0,
                    supports_top_k=True,
                    maximum_top_k=100,
                ),
            ),
            request,
        )

    assert raised.value.code == "invalid_parameter"
    assert raised.value.param == "temperature"
    assert "between 0.0 and 1.0" in str(raised.value)


def test_srn_sampling_drops_and_discloses_instead_of_rejecting() -> None:
    """On a reasoning route (srn), temperature/top_p sent with reasoning on is dropped
    and disclosed, not rejected: the model accepts sampling, just not at this effort."""
    profile = GatewayWireProfile(
        dialect="openai_responses",
        url="https://provider.test",
        model_id="gpt-5.1",
        supports_reasoning=True,
        reasoning_wire_format="openai_responses",
        reasoning_effort="medium",
        supports_temperature=True,
        supports_top_p=True,
        sampling_requires_reasoning_none=True,
    )
    public_request, provider_request = route_generation_parameter_requests(
        (profile,), _chat_request(temperature=0.2, top_p=0.9)
    )

    # The caller value survives on the public copy for reflection, is dropped from
    # the provider payload, and both drops are disclosed with an actionable reason.
    assert public_request.temperature == 0.2
    assert public_request.top_p == 0.9
    assert provider_request.temperature is None
    assert provider_request.top_p is None
    assert public_request.ignored_parameters == (
        "temperature->dropped(set_reasoning_effort_none)",
        "top_p->dropped(set_reasoning_effort_none)",
    )


def test_srn_sampling_is_honored_at_explicit_none_reasoning() -> None:
    """The sampling hatch stays open: reasoning_effort=none honors temperature/top_p."""
    profile = GatewayWireProfile(
        dialect="openai_responses",
        url="https://provider.test",
        model_id="gpt-5.1",
        supports_reasoning=True,
        reasoning_wire_format="openai_responses",
        reasoning_effort="medium",
        supports_temperature=True,
        supports_top_p=True,
        sampling_requires_reasoning_none=True,
    )
    request = _chat_request(temperature=0.2, top_p=0.9).model_copy(
        update={"reasoning_effort": "none"}
    )
    public_request, provider_request = route_generation_parameter_requests((profile,), request)

    assert public_request.temperature == 0.2
    assert provider_request.temperature == 0.2
    assert provider_request.top_p == 0.9
    assert provider_request.reasoning_effort == "none"
    assert public_request.ignored_parameters == ()


def test_temperature_narrows_to_a_honoring_rung_over_an_srn_rung() -> None:
    """A mixed route [srn rung + a plain rung that honors sampling] narrows temperature/
    top_p to the honoring rung rather than dropping them on the srn rung — preserving the
    caller's value when a rung can serve it."""
    srn_rung = GatewayWireProfile(
        dialect="openai_responses",
        url="https://srn.test",
        model_id="gpt-5.1",
        supports_reasoning=True,
        reasoning_wire_format="openai_responses",
        reasoning_effort="medium",
        supports_temperature=True,
        supports_top_p=True,
        sampling_requires_reasoning_none=True,
    )
    plain_rung = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://plain.test",
        model_id="provider/plain",
        supports_temperature=True,
        supports_top_p=True,
    )
    request = _chat_request(temperature=0.5, top_p=0.9)

    # The srn rung would drop sampling at effort=medium; the plain rung honors it, so
    # selection narrows to the honoring rung.
    assert compatible_generation_parameter_profile_indexes((srn_rung, plain_rung), request) == (1,)


def test_genuinely_unsupported_sampling_drops_with_its_own_disclosure() -> None:
    """A route that never declares temperature (Anthropic constrained [1,1]) drops it
    and says so — there is nothing to honor at any effort, so the disclosure names
    the provider, not a reasoning-effort remedy, and the model still answers."""
    profile = GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://provider.test",
        model_id="claude-constrained",
        supports_temperature=False,
    )
    public_request, provider_request = route_generation_parameter_requests(
        (profile,), _chat_request(temperature=0.2)
    )

    assert provider_request.temperature is None
    assert "temperature->dropped(unsupported_by_provider)" in public_request.ignored_parameters
    assert "temperature->dropped(set_reasoning_effort_none)" not in (
        public_request.ignored_parameters
    )


def test_thinking_default_enable_resolves_the_required_default_effort() -> None:
    """A level-less enable resolves to the route's required default effort."""
    profile = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://p.test",
        model_id="provider/reasoner",
        supports_reasoning=True,
        reasoning_wire_format="reasoning",
        reasoning_effort="medium",
        supported_reasoning_efforts=("none", "low", "medium", "high"),
        reasoning_effort_required=True,
    )
    request = _chat_request().model_copy(update={"thinking_default_enable": True})

    _public, provider = route_generation_parameter_requests((profile,), request)

    assert provider.reasoning_effort == "medium"


def test_thinking_default_enable_falls_back_to_the_lowest_non_none_effort() -> None:
    """When the model requires no default, a level-less enable picks the lowest tier."""
    profile = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://p.test",
        model_id="provider/optional-reasoner",
        supports_reasoning=True,
        reasoning_wire_format="reasoning",
        supported_reasoning_efforts=("none", "low", "high"),
    )
    request = _chat_request().model_copy(update={"thinking_default_enable": True})

    _public, provider = route_generation_parameter_requests((profile,), request)

    assert provider.reasoning_effort == "low"


def test_thinking_default_enable_on_a_non_reasoning_route_surfaces() -> None:
    """A route that supports no reasoning effort cannot enable thinking → rejects."""
    profile = GatewayWireProfile(
        dialect="openai_compatible", url="https://p.test", model_id="provider/plain"
    )
    request = _chat_request().model_copy(update={"thinking_default_enable": True})

    with pytest.raises(ProviderParameterError) as raised:
        route_generation_parameter_requests((profile,), request)

    assert raised.value.code == "unsupported_parameter"


def test_payload_builder_rejects_conditional_sampling_without_admission() -> None:
    """Direct provider use retains the same local guard as gateway admission."""
    with pytest.raises(ProviderParameterError, match="reasoning_effort is 'none'"):
        openai_responses_stream_payload(
            "gpt-5.1",
            _chat_request(temperature=0.2).model_copy(update={"reasoning_effort": "medium"}),
            supports_temperature=True,
            supports_reasoning=True,
            sampling_requires_reasoning_none=True,
        )


def test_route_admits_stop_on_native_responses_and_keeps_it_off_the_wire() -> None:
    """The Responses API has no stop field: the route admits stop for data-plane emulation.

    The canonical request keeps the caller's sequences (the wire entry hands
    them to the native relay) while the provider payload never carries them.
    """
    request = _chat_request().model_copy(update={"stop": ("DONE", "</severity>")})

    public_request, provider_request = route_generation_parameter_requests(
        (GatewayWireProfile(dialect="openai_responses", url="https://provider.test"),),
        request,
    )

    assert provider_request.stop == ("DONE", "</severity>")
    assert "stop" not in public_request.ignored_parameters
    payload = openai_responses_stream_payload(
        "gpt-5.6-luna",
        provider_request,
        supports_temperature=True,
    )
    assert "stop" not in payload
    assert "stop_sequences" not in payload


def test_responses_logprob_request_is_rejected_instead_of_ignored() -> None:
    """A requested probability result cannot disappear during normalization."""
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="hello"),),
        logprobs=True,
        top_logprobs=5,
        ignored_parameters=("top_logprobs",),
    )

    with pytest.raises(ProviderParameterError) as raised:
        route_generation_parameter_requests(
            (GatewayWireProfile(dialect="openai_responses", url="https://provider.test"),),
            request,
        )

    assert raised.value.code == "unsupported_parameter"
    assert raised.value.param == "top_logprobs"


def test_false_logprobs_is_disclosed_and_removed_as_a_noop() -> None:
    """An explicit false selector is harmless but never leaks to brittle providers."""
    request = _chat_request().model_copy(update={"logprobs": False})

    public_request, provider_request = route_generation_parameter_requests(
        (GatewayWireProfile(dialect="openai_compatible", url="https://provider.test"),),
        request,
    )

    assert public_request.ignored_parameters == ("logprobs",)
    assert provider_request.logprobs is None


def test_route_rejects_temperature_outside_any_provider_range() -> None:
    """A cross-provider route rejects a value one wire cannot preserve."""
    request = _chat_request(temperature=1.5)
    profiles = (
        GatewayWireProfile(dialect="openai_responses", url="https://openai.test"),
        GatewayWireProfile(
            dialect="bedrock_converse_stream",
            url="https://bedrock.test",
            maximum_temperature=1.0,
        ),
    )

    with pytest.raises(ProviderParameterError) as raised:
        route_generation_parameter_requests(profiles, request)

    assert raised.value.code == "invalid_parameter"
    assert raised.value.param == "temperature"


@pytest.mark.parametrize(
    "public_parameter",
    ("max_tokens", "max_completion_tokens", "max_output_tokens"),
)
def test_route_rejects_output_ceiling_above_smallest_waterfall_limit(
    public_parameter: str,
) -> None:
    """A caller token ceiling is checked against every known deployment limit."""
    request = _chat_request().model_copy(
        update={
            "maximum_output_tokens": 65_000,
            "maximum_output_tokens_parameter": public_parameter,
        }
    )
    profiles = (
        GatewayWireProfile(
            dialect="openai_compatible",
            url="https://first.test",
            maximum_output_tokens=128_000,
        ),
        GatewayWireProfile(
            dialect="openai_compatible",
            url="https://fallback.test",
            maximum_output_tokens=64_000,
        ),
    )

    with pytest.raises(ProviderParameterError) as raised:
        route_generation_parameter_requests(profiles, request)

    assert raised.value.code == "invalid_parameter"
    assert raised.value.param == public_parameter
    assert "maximum of 64000" in str(raised.value)


def test_route_supplies_anthropic_required_max_tokens_within_every_rung_limit() -> None:
    """An omitted public limit becomes one safe route-wide Anthropic default."""
    request = _chat_request()
    profiles = (
        GatewayWireProfile(
            dialect="anthropic_messages",
            url="https://anthropic.test",
            maximum_output_tokens=8_192,
        ),
        GatewayWireProfile(
            dialect="openai_compatible",
            url="https://fallback.test",
            maximum_output_tokens=2_048,
        ),
    )

    public_request, provider_request = route_generation_parameter_requests(profiles, request)

    assert public_request.maximum_output_tokens is None
    assert provider_request.maximum_output_tokens == 2_048


def test_reasoning_effort_uses_each_provider_native_wire_shape() -> None:
    """The same public control translates without leaking an OpenAI field across providers."""
    request = _chat_request().model_copy(update={"reasoning_effort": "high"})

    openrouter = dialect_stream_payload(
        GatewayWireProfile(
            dialect="openai_compatible",
            url="https://openrouter.test",
            model_id="anthropic/claude-opus-5",
            supports_reasoning=True,
            reasoning_wire_format="reasoning",
        ),
        request,
    )
    anthropic = dialect_stream_payload(
        GatewayWireProfile(
            dialect="anthropic_messages",
            url="https://anthropic.test",
            model_id="claude-sonnet-4-6",
            supports_reasoning=True,
            reasoning_wire_format="anthropic_adaptive",
        ),
        request,
    )
    gemini = dialect_stream_payload(
        GatewayWireProfile(
            dialect="gemini_generate_content",
            url="https://gemini.test",
            model_id="gemini-3.7-flash",
            supports_reasoning=True,
            reasoning_wire_format="gemini_thinking",
        ),
        request,
    )

    assert openrouter["reasoning"] == {"effort": "high"}
    assert "reasoning_effort" not in openrouter
    assert anthropic["thinking"] == {"type": "adaptive"}
    assert anthropic["output_config"] == {"effort": "high"}
    assert "reasoning" not in anthropic
    generation = cast("dict[str, object]", gemini["generationConfig"])
    assert generation["thinkingConfig"] == {"thinkingLevel": "HIGH"}


@pytest.mark.parametrize(
    "profile",
    (
        GatewayWireProfile(dialect="anthropic_messages", url="https://anthropic.test"),
        GatewayWireProfile(dialect="gemini_generate_content", url="https://gemini.test"),
        GatewayWireProfile(dialect="bedrock_converse_stream", url="https://bedrock.test"),
        GatewayWireProfile(dialect="openai_compatible", url="https://compatible.test"),
    ),
)
def test_non_native_reasoning_profiles_never_emit_openai_reasoning_fields(
    profile: GatewayWireProfile,
) -> None:
    """Every non-Responses provider path drops OpenAI reasoning controls by default."""
    request = _chat_request().model_copy(update={"reasoning_effort": "high"})

    payload = dialect_stream_payload(profile, request)

    assert "reasoning" not in payload
    assert "reasoning_effort" not in payload


def test_anthropic_messages_stream_payload_forwards_top_p() -> None:
    """Native Anthropic streaming preserves nucleus sampling on the Messages wire."""
    payload = anthropic_messages_stream_payload(
        "exact-model",
        _chat_request(top_p=1.0, temperature=0.2),
    )

    assert payload["stream"] is True
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 1.0


def test_anthropic_messages_stream_payload_omits_absent_top_p() -> None:
    """Native Anthropic streaming does not invent a nucleus-sampling value."""
    payload = anthropic_messages_stream_payload("exact-model", _chat_request())

    assert "top_p" not in payload
    assert "temperature" not in payload


def test_anthropic_payload_merges_reasoning_schema_and_tool_controls() -> None:
    """Anthropic receives its native strict-output and parallel-tool wire shapes."""
    request = _chat_request().model_copy(
        update={
            "tools": (
                GatewayToolDefinition(
                    name="lookup",
                    description="Find a record.",
                    parameters={"type": "object"},
                    strict=True,
                ),
            ),
            "parallel_tool_calls": False,
            "reasoning_effort": "high",
            "structured_text": StructuredTextFormat(
                name="answer",
                json_schema={"type": "object", "additionalProperties": False},
            ),
        }
    )

    payload = anthropic_messages_stream_payload(
        "claude-opus-5",
        request,
        supports_reasoning=True,
    )

    assert payload["tools"] == [
        {
            "name": "lookup",
            "description": "Find a record.",
            "input_schema": {"type": "object"},
            "strict": True,
        }
    ]
    assert payload["tool_choice"] == {
        "type": "auto",
        "disable_parallel_tool_use": True,
    }
    assert payload["thinking"] == {"type": "adaptive"}
    assert payload["output_config"] == {
        "effort": "high",
        "format": {
            "type": "json_schema",
            "schema": {"type": "object", "additionalProperties": False},
        },
    }


def test_anthropic_messages_stream_payload_round_trips_tool_error_state() -> None:
    """A failed tool result keeps is_error on the Anthropic wire only when set."""
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(
            GatewayMessage(role="tool", content="boom", tool_call_id="call-1", tool_is_error=True),
            GatewayMessage(role="tool", content="fine", tool_call_id="call-2"),
        ),
        stream=True,
        include_usage=True,
    )
    payload = anthropic_messages_stream_payload("exact-model", request)
    messages = cast("list[dict[str, object]]", payload["messages"])
    blocks = cast("list[dict[str, object]]", messages[0]["content"])
    assert blocks[0] == {
        "type": "tool_result",
        "tool_use_id": "call-1",
        "content": "boom",
        "is_error": True,
    }
    # An ordinary result stays byte-identical to the pre-existing payload shape.
    assert blocks[1] == {"type": "tool_result", "tool_use_id": "call-2", "content": "fine"}


def test_an_empty_assistant_text_never_reaches_the_anthropic_wire() -> None:
    """A tool-call-only assistant turn omits the empty text block."""
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(
            GatewayMessage(role="user", content="look it up"),
            GatewayMessage(
                role="assistant",
                content="",
                tool_calls=(
                    ToolCall(call_id="call-1", name="lookup", arguments={}, raw_arguments="{}"),
                ),
            ),
            GatewayMessage(role="tool", content="found", tool_call_id="call-1"),
        ),
        stream=True,
        include_usage=True,
    )
    payload = anthropic_messages_stream_payload("exact-model", request)
    messages = cast("list[dict[str, object]]", payload["messages"])
    blocks = cast("list[dict[str, object]]", messages[1]["content"])
    assert [block["type"] for block in blocks] == ["tool_use"]


def test_tool_error_state_folds_with_disclosure_off_the_anthropic_wire() -> None:
    """A fallback rung folds the error flag into text instead of rejecting.

    The flag is baked into history on every failed tool call, so the old
    named 400 wedged whole sessions; the canonical flag stays on the shared
    request (Anthropic rungs still emit it natively) and the fold is
    disclosed route-wide.
    """
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(
            GatewayMessage(role="tool", content="boom", tool_call_id="call-1", tool_is_error=True),
        ),
    )
    anthropic = GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://anthropic.test",
    )

    public_request, provider_request = route_generation_parameter_requests((anthropic,), request)
    assert public_request.messages[0].tool_is_error is True
    assert provider_request.messages[0].tool_is_error is True
    assert "messages.content.is_error->content" not in public_request.ignored_parameters

    mixed_public, mixed_provider = route_generation_parameter_requests(
        (
            anthropic,
            GatewayWireProfile(
                dialect="openai_compatible",
                url="https://fallback.test",
            ),
        ),
        request,
    )
    assert "messages.content.is_error->content" in mixed_public.ignored_parameters
    assert mixed_provider.messages[0].tool_is_error is True


def test_gemini_stream_payload_matches_the_provider_client_builder() -> None:
    """The gemini dialect payload is the exact generateContent body the python
    provider client sends, built through the same shared converter."""
    request = _chat_request(temperature=0.3)
    payload = gemini_generate_content_stream_payload("gemini-2.5-pro", request)

    assert payload == gemini_generate_request("gemini-2.5-pro", model_request(request))
    assert payload["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
    generation = cast("dict[str, object]", payload["generationConfig"])
    assert isinstance(generation, dict)
    assert generation["temperature"] == 0.3
    # Streaming is selected by the streamGenerateContent route, not the body.
    assert "stream" not in payload


def test_gemini_generation_forwards_top_p_and_top_k_only_when_declared() -> None:
    """Gemini's camel-case sampling names are gated independently."""
    request = _chat_request().model_copy(update={"top_p": 0.8, "top_k": 20})
    payload = gemini_generate_content_stream_payload(
        "gemini-2.5-pro",
        request,
        supports_top_k=True,
    )
    generation = cast("dict[str, object]", payload["generationConfig"])
    assert isinstance(generation, dict)
    assert generation["topP"] == 0.8
    assert generation["topK"] == 20


def test_gemini_generation_forwards_stop_and_strict_json_schema() -> None:
    """Gemini stop and structured output controls use generationConfig names."""
    schema: JsonObject = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }
    request = _chat_request().model_copy(
        update={
            "stop": ("DONE",),
            "structured_text": StructuredTextFormat(name="answer", json_schema=schema),
        }
    )

    payload = gemini_generate_content_stream_payload("gemini-2.5-pro", request)
    generation = cast("dict[str, object]", payload["generationConfig"])

    assert generation["stopSequences"] == ["DONE"]
    assert generation["responseMimeType"] == "application/json"
    assert generation["responseJsonSchema"] == schema


def test_openai_responses_stream_payload_ignores_logprobs_even_when_flagged() -> None:
    """Responses logprob controls are accepted but not sent without output projection."""
    request = _chat_request().model_copy(update={"top_logprobs": 5})
    payload = openai_responses_stream_payload(
        "exact-model",
        request,
        supports_temperature=True,
        supports_logprobs=True,
    )

    assert "top_logprobs" not in payload


def test_dialect_dispatch_builds_the_gemini_payload() -> None:
    """The shared dialect dispatch routes gemini_generate_content correctly."""
    profile = GatewayWireProfile(
        dialect="gemini_generate_content",
        url="https://example.invalid/models/gemini-2.5-pro:streamGenerateContent?alt=sse",
        model_id="gemini-2.5-pro",
    )
    payload = dialect_stream_payload(profile, _chat_request())

    assert payload["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]


def test_bedrock_stream_payload_matches_the_provider_builder_without_model_id() -> None:
    """The bedrock dialect body is the shared Converse payload with the
    ``modelId`` routing key removed: on the REST route the model travels in
    the URL path, not the body."""
    request = _chat_request(temperature=0.2)
    payload = bedrock_converse_stream_payload("us.anthropic.claude-sonnet-4-5", request)

    assert payload == converse_body(model_request(request))
    assert "modelId" not in payload
    assert payload["messages"] == [{"role": "user", "content": [{"text": "hello"}]}]
    inference = payload["inferenceConfig"]
    assert isinstance(inference, dict)
    assert inference["temperature"] == 0.2


def test_bedrock_stream_payload_forwards_stop_schema_and_strict_tools() -> None:
    """Converse receives every certified semantic control on its native fields."""
    schema: JsonObject = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }
    request = _chat_request().model_copy(
        update={
            "stop": ("DONE",),
            "tools": (
                GatewayToolDefinition(
                    name="lookup",
                    description="Find a record.",
                    parameters={"type": "object"},
                    strict=True,
                ),
            ),
            "structured_text": StructuredTextFormat(
                name="answer",
                description="Return one answer.",
                json_schema=schema,
            ),
        }
    )

    payload = bedrock_converse_stream_payload("exact-model", request)

    assert payload["inferenceConfig"] == {"stopSequences": ["DONE"]}
    assert payload["toolConfig"] == {
        "tools": [
            {
                "toolSpec": {
                    "name": "lookup",
                    "description": "Find a record.",
                    "inputSchema": {"json": {"type": "object"}},
                    "strict": True,
                }
            }
        ]
    }
    assert payload["outputConfig"] == {
        "textFormat": {
            "type": "json_schema",
            "structure": {
                "jsonSchema": {
                    "schema": '{"properties":{"answer":{"type":"string"}},"type":"object"}',
                    "name": "answer",
                    "description": "Return one answer.",
                }
            },
        }
    }


def test_dialect_dispatch_builds_the_bedrock_payload() -> None:
    """The shared dialect dispatch routes bedrock_converse_stream correctly."""
    profile = GatewayWireProfile(
        dialect="bedrock_converse_stream",
        url="https://bedrock-runtime.us-east-1.amazonaws.com/model/m/converse-stream",
        model_id="us.anthropic.claude-sonnet-4-5",
        signs_request_body=True,
    )
    payload = dialect_stream_payload(profile, _chat_request())

    assert "modelId" not in payload
    assert payload["messages"] == [{"role": "user", "content": [{"text": "hello"}]}]


def _thinking_history_request(**overrides: object) -> GatewayRequest:
    """Build one Messages request replaying thinking history blocks."""
    from exp.runtime.gateway.contracts import RedactedThinkingBlock, ThinkingBlock

    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(
            GatewayMessage(role="user", content="go"),
            GatewayMessage(
                role="assistant",
                content="done",
                provider_reasoning=(
                    ThinkingBlock(text="private", signature="sig=="),
                    RedactedThinkingBlock(data="opaque=="),
                ),
            ),
            GatewayMessage(role="user", content="continue"),
        ),
        stream=True,
        include_usage=True,
    )
    return request.model_copy(update=dict(overrides)) if overrides else request


def test_anthropic_payload_carries_verbatim_thinking_config_over_adaptive() -> None:
    """The caller's exact thinking object wins over the catalog effort default."""
    config: JsonObject = {"type": "enabled", "budget_tokens": 2048}
    request = _thinking_history_request(provider_thinking_config=config)
    payload = anthropic_messages_stream_payload(
        "claude-fable-5",
        request,
        supports_reasoning=True,
        reasoning_effort="high",
    )
    assert payload["thinking"] == config
    # The adaptive default and its effort stay off the wire under an
    # explicit caller configuration.
    assert "output_config" not in payload

    adaptive = anthropic_messages_stream_payload(
        "claude-fable-5",
        _thinking_history_request(),
        supports_reasoning=True,
        reasoning_effort="high",
    )
    assert adaptive["thinking"] == {"type": "adaptive"}
    assert adaptive["output_config"] == {"effort": "high"}


def test_anthropic_payload_replays_thinking_blocks_first_and_verbatim() -> None:
    """Thinking history leads the assistant turn with byte-exact signatures."""
    payload = anthropic_messages_stream_payload("claude-fable-5", _thinking_history_request())
    messages = cast(list[JsonObject], payload["messages"])
    assistant_blocks = cast(list[JsonObject], messages[1]["content"])
    assert assistant_blocks[0] == {
        "type": "thinking",
        "thinking": "private",
        "signature": "sig==",
    }
    assert assistant_blocks[1] == {"type": "redacted_thinking", "data": "opaque=="}
    assert assistant_blocks[2] == {"type": "text", "text": "done"}


def test_route_shaping_rejects_thinking_by_name_so_admission_can_coerce() -> None:
    """History thinking blocks and a live config both reject at SHAPING on a
    non-Anthropic route: blocks are signed provider state no translation can
    carry, and the config's named rejection is what lets the admit loop offer
    the disclosed thinking->reasoning_effort coercion without stealing
    narrowing preference from an Anthropic rung (coverage for the coercion
    itself lives in capability_policy_test)."""
    anthropic = GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://anthropic.test",
        reasoning_wire_format="anthropic_adaptive",
    )
    fallback = GatewayWireProfile(dialect="openai_compatible", url="https://fallback.test")

    for request in (
        _thinking_history_request(),
        _chat_request().model_copy(
            update={
                "surface": GatewayApiSurface.MESSAGES,
                "provider_thinking_config": {"type": "enabled", "budget_tokens": 1024},
            }
        ),
    ):
        route_generation_parameter_requests((anthropic,), request)
        with pytest.raises(ProviderParameterError) as raised:
            route_generation_parameter_requests((anthropic, fallback), request)
        assert raised.value.param == "thinking"
        assert raised.value.code == "unsupported_parameter"


def _encrypted_reasoning_request() -> GatewayRequest:
    """Build one Codex-shaped Responses request with encrypted reasoning replay."""
    from exp.runtime.gateway.contracts import EncryptedReasoningBlock

    return GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(
            GatewayMessage(role="user", content="go"),
            GatewayMessage(
                role="assistant",
                content="done",
                provider_reasoning=(
                    EncryptedReasoningBlock(id="rs_1", encrypted_content="blob=="),
                ),
            ),
            GatewayMessage(role="user", content="continue"),
        ),
        response_store=False,
        include_encrypted_reasoning=True,
        stream=True,
        include_usage=True,
    )


def test_responses_payload_forwards_include_and_replays_reasoning_items() -> None:
    """Encrypted reasoning replays ahead of its assistant message, store stays false."""
    payload = openai_responses_stream_payload(
        "gpt-5.6-sol",
        _encrypted_reasoning_request(),
        supports_temperature=True,
        supports_reasoning=True,
    )
    assert payload["store"] is False
    assert payload["include"] == ["reasoning.encrypted_content"]
    items = cast(list[JsonObject], payload["input"])
    # The item id is never forwarded: the provider binds encrypted_content
    # to its ORIGINAL item id and callers echo this gateway's minted public
    # ids, so any forwarded id fails verification (live 2026-08-29:
    # "Encrypted content item_id did not match"); an id-less item verifies
    # against the id embedded in the payload itself.
    assert items[1] == {
        "type": "reasoning",
        "summary": [],
        "encrypted_content": "blob==",
    }
    assert items[2] == {"role": "assistant", "content": "done"}


def test_responses_payload_acquires_internal_reasoning_for_gateway_continuation() -> None:
    """Gateway-stored reasoning stays available even when the caller hides it."""
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="go"),),
        tools=(GatewayToolDefinition(name="lookup", parameters={"type": "object"}),),
    )

    retained = openai_responses_stream_payload(
        "gpt-5.6-sol",
        request,
        supports_temperature=True,
        supports_reasoning=True,
    )
    assert retained["store"] is False
    assert retained["include"] == ["reasoning.encrypted_content"]

    caller_opt_out = openai_responses_stream_payload(
        "gpt-5.6-sol",
        request.model_copy(update={"response_store": False}),
        supports_temperature=True,
        supports_reasoning=True,
    )
    assert "include" not in caller_opt_out

    stored = openai_responses_stream_payload(
        "gpt-5.6-sol",
        request.model_copy(update={"response_store": True}),
        supports_temperature=True,
        supports_reasoning=True,
    )
    assert stored["include"] == ["reasoning.encrypted_content"]

    caller_public = openai_responses_stream_payload(
        "gpt-5.6-sol",
        request.model_copy(update={"response_store": False, "include_encrypted_reasoning": True}),
        supports_temperature=True,
        supports_reasoning=True,
    )
    assert caller_public["include"] == ["reasoning.encrypted_content"]

    non_reasoning = openai_responses_stream_payload(
        "plain-model",
        request,
        supports_temperature=True,
        supports_reasoning=False,
    )
    assert "include" not in non_reasoning


def test_route_rejects_encrypted_reasoning_outside_native_responses() -> None:
    """Encrypted reasoning requires every waterfall rung to speak native Responses."""
    responses = GatewayWireProfile(
        dialect="openai_responses",
        url="https://openai.test",
        supports_reasoning=True,
        reasoning_wire_format="openai_responses",
    )
    fallback = GatewayWireProfile(dialect="openai_compatible", url="https://fallback.test")
    request = _encrypted_reasoning_request()

    route_generation_parameter_requests((responses,), request)
    with pytest.raises(ProviderParameterError) as raised:
        route_generation_parameter_requests((responses, fallback), request)
    assert raised.value.param == "include"
    assert raised.value.code == "unsupported_parameter"


def test_fireworks_stateless_carrier_include_survives_route_shaping() -> None:
    """The gateway-issued Fireworks carrier is distinct from native OpenAI reasoning."""
    fireworks = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://api.fireworks.ai/inference/v1/chat/completions",
        model_id="accounts/fireworks/models/deepseek-v4-flash-0731",
        fireworks_reasoning_route_sha256="f" * 64,
    )
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="use a tool"),),
        tools=(GatewayToolDefinition(name="lookup", parameters={"type": "object"}),),
        response_store=False,
        include_encrypted_reasoning=True,
        stream=True,
        include_usage=True,
    )

    route_generation_parameter_requests((fireworks,), request)
    payload = dialect_stream_payload(fireworks, request)

    assert payload["model"] == "accounts/fireworks/models/deepseek-v4-flash-0731"


def test_mixed_native_and_fireworks_reasoning_channels_fail_closed() -> None:
    """One include selector cannot promise two incompatible carrier authorities."""
    responses = GatewayWireProfile(
        dialect="openai_responses",
        url="https://openai.test",
        supports_reasoning=True,
        reasoning_wire_format="openai_responses",
    )
    fireworks = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://api.fireworks.ai/inference/v1/chat/completions",
        fireworks_reasoning_route_sha256="f" * 64,
    )
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="go"),),
        response_store=False,
        include_encrypted_reasoning=True,
    )

    with pytest.raises(ProviderParameterError) as raised:
        route_generation_parameter_requests((responses, fireworks), request)
    assert raised.value.param == "include"


def test_gpt_56_efforts_match_the_provider_and_ultra_rejects_loud() -> None:
    """The gpt-5.6 family accepts the provider-verified seven efforts.

    Live check 2026-08-28: gpt-5.6-sol and gpt-5.6-codex reject "ultra" by
    name and accept none through xhigh plus max, so admission mirrors the
    provider exactly: decode still accepts the token, and the route rejects
    it with the supported list before any dispatch.
    """
    codex = GatewayWireProfile(
        dialect="openai_responses",
        url="https://openai.test",
        model_id="gpt-5.6-sol",
        supports_reasoning=True,
        reasoning_wire_format="openai_responses",
    )

    def request_with(effort: str) -> GatewayRequest:
        """Build one Responses request carrying the given effort."""
        return GatewayRequest(
            surface=GatewayApiSurface.RESPONSES,
            messages=(GatewayMessage(role="user", content="go"),),
            reasoning_effort=cast("ReasoningEffort", effort),
        )

    max_request = request_with("max")
    route_generation_parameter_requests((codex,), max_request)
    payload = openai_responses_stream_payload(
        "gpt-5.6-sol",
        max_request,
        supports_temperature=True,
        supports_reasoning=True,
    )
    assert payload["reasoning"] == {"effort": "max"}

    with pytest.raises(UnsupportedReasoningEffortError) as raised:
        route_generation_parameter_requests((codex,), request_with("ultra"))
    assert "'ultra'" in str(raised.value)
    assert "'max'" in str(raised.value)


def test_reasoning_context_passes_through_verbatim_and_narrows_per_rung() -> None:
    """reasoning.context forwards untouched and admits only native Responses rungs."""
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="go"),),
        reasoning_effort="high",
        reasoning_context="all_turns",
    )
    payload = openai_responses_stream_payload(
        "gpt-5.6-luna",
        request,
        supports_temperature=True,
        supports_reasoning=True,
    )
    assert payload["reasoning"] == {"effort": "high", "context": "all_turns"}

    responses = GatewayWireProfile(
        dialect="openai_responses",
        url="https://openai.test",
        model_id="gpt-5.6-luna",
        supports_reasoning=True,
        reasoning_wire_format="openai_responses",
    )
    # The fallback accepts the effort, so context is the one blocker.
    fallback = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://fallback.test",
        model_id="gpt-5.2",
        supports_reasoning=True,
        reasoning_wire_format="reasoning_effort",
    )
    route_generation_parameter_requests((responses,), request)
    # Per-rung narrowing keeps the compatible rung; the whole-route error
    # names the field only when no rung qualifies.
    assert compatible_generation_parameter_profile_indexes((responses, fallback), request) == (0,)
    with pytest.raises(ProviderParameterError) as raised:
        route_generation_parameter_requests((fallback,), request)
    assert raised.value.param == "reasoning.context"
    assert raised.value.code == "unsupported_parameter"


def test_anthropic_tools_omit_an_absent_description() -> None:
    """Anthropic 400s an explicit null description, so the key stays absent.

    Live repro (2026-08-28): a tool without a description produced
    "tools.0.custom.description: Input should be a valid string" and every
    tools request on the route failed before streaming.
    """
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="go"),),
        tools=(
            GatewayToolDefinition(name="bare", parameters={"type": "object"}),
            GatewayToolDefinition(
                name="described", description="Look up.", parameters={"type": "object"}
            ),
        ),
        stream=True,
        include_usage=True,
    )
    payload = anthropic_messages_stream_payload("claude-fable-5", request)
    tools = cast(list[JsonObject], payload["tools"])
    assert "description" not in tools[0]
    assert tools[1]["description"] == "Look up."


def _thinking_config_request(config: JsonObject) -> GatewayRequest:
    """Build one Messages request carrying a verbatim thinking config."""
    return GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="go"),),
        provider_thinking_config=config,
        stream=True,
        include_usage=True,
    )


def _anthropic_profile(model_id: str) -> GatewayWireProfile:
    """Build one reasoning-capable Anthropic wire profile."""
    return GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://anthropic.test",
        model_id=model_id,
        supports_reasoning=True,
        reasoning_wire_format="anthropic_adaptive",
        reasoning_effort="medium",
    )


def test_enabled_thinking_translates_to_adaptive_on_adaptive_only_models() -> None:
    """The adaptive-only generation rejects caller enabled configs, so the
    route translates to adaptive and DISCLOSES the dropped token budget
    instead of silently mapping or silently failing."""
    request = _thinking_config_request({"type": "enabled", "budget_tokens": 2048})
    public, provider = route_generation_parameter_requests(
        (_anthropic_profile("claude-fable-5"),), request
    )
    assert "thinking.budget_tokens" in public.ignored_parameters
    assert provider.provider_thinking_config == {"type": "adaptive"}
    payload = anthropic_messages_stream_payload(
        "claude-fable-5",
        provider,
        supports_reasoning=True,
        reasoning_effort="medium",
    )
    assert payload["thinking"] == {"type": "adaptive"}
    assert payload["output_config"] == {"effort": "medium"}
    # The translation stays explicit on routes that pin no effort, so the
    # caller's request to think never degrades to an implicit provider default.
    bare = anthropic_messages_stream_payload("claude-fable-5", provider)
    assert bare["thinking"] == {"type": "adaptive"}
    assert "output_config" not in bare


def test_enabled_thinking_stays_verbatim_on_budget_capable_models() -> None:
    """The pre-adaptive generation still honors the caller's exact config."""
    config: JsonObject = {"type": "enabled", "budget_tokens": 2048}
    request = _thinking_config_request(config)
    public, provider = route_generation_parameter_requests(
        (_anthropic_profile("claude-haiku-4-5"),), request
    )
    assert "thinking.budget_tokens" not in public.ignored_parameters
    assert provider.provider_thinking_config == config
    payload = anthropic_messages_stream_payload(
        "claude-haiku-4-5",
        provider,
        supports_reasoning=True,
        reasoning_effort="medium",
    )
    assert payload["thinking"] == config


def test_disabled_thinking_rejects_by_name_on_adaptive_only_models() -> None:
    """Thinking cannot be turned off on the adaptive generation; silently
    letting the model think anyway would bill the caller for reasoning they
    explicitly disabled."""
    request = _thinking_config_request({"type": "disabled"})
    with pytest.raises(ProviderParameterError) as raised:
        route_generation_parameter_requests((_anthropic_profile("claude-sonnet-5"),), request)
    assert raised.value.param == "thinking.type"
    assert raised.value.code == "unsupported_parameter"

    # The pre-adaptive generation still accepts an explicit disabled config.
    _public, provider = route_generation_parameter_requests(
        (_anthropic_profile("claude-opus-4-6"),), _thinking_config_request({"type": "disabled"})
    )
    assert provider.provider_thinking_config == {"type": "disabled"}


def test_tool_call_cache_hint_forwards_to_anthropic_and_discloses_elsewhere() -> None:
    """The validated cache hint reaches only the tool_use block that can honor it."""
    from exp.common.core.artifacts import sha256_json

    hinted = ToolCall(
        call_id="call-2",
        name="read_file",
        arguments={"path": "b.txt"},
        cache_control={"type": "ephemeral"},
    )
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(
            GatewayMessage(role="user", content="read"),
            GatewayMessage(
                role="assistant",
                tool_calls=(
                    ToolCall(call_id="call-1", name="read_file", arguments={"path": "a.txt"}),
                    hinted,
                ),
            ),
            GatewayMessage(role="tool", content="a", tool_call_id="call-1"),
            GatewayMessage(role="tool", content="b", tool_call_id="call-2"),
        ),
        stream=True,
        include_usage=True,
    )

    anthropic_payload = anthropic_messages_stream_payload("claude-fable-5", request)
    messages = cast(list[JsonObject], anthropic_payload["messages"])
    blocks = cast(list[JsonObject], messages[1]["content"])
    assert "cache_control" not in blocks[0]
    assert blocks[1]["cache_control"] == {"type": "ephemeral"}

    # OpenAI-family wires never see the hint; the route discloses the no-op.
    chat_payload = openai_compatible_stream_payload("gpt-5.5", request)
    import json

    assert "cache_control" not in json.dumps(chat_payload)
    public, _provider = route_generation_parameter_requests(
        (GatewayWireProfile(dialect="openai_compatible", url="https://openai.test"),),
        request,
    )
    assert "messages.tool_calls.cache_control" in public.ignored_parameters
    anthropic_public, _provider = route_generation_parameter_requests(
        (GatewayWireProfile(dialect="anthropic_messages", url="https://anthropic.test"),),
        request,
    )
    assert "messages.tool_calls.cache_control" not in anthropic_public.ignored_parameters

    # The hint never perturbs digests, artifacts, or replay identity.
    bare = hinted.model_copy(update={"cache_control": None})
    assert sha256_json(hinted) == sha256_json(bare)


def test_context_management_forwards_on_anthropic_and_discloses_elsewhere() -> None:
    """Anthropic-native context editing forwards verbatim with its beta
    header; any other route drops it with disclosure, never a rejection
    (Claude Code sends the field by default)."""
    from exp.runtime.models.providers.wire_messages import (
        ANTHROPIC_CONTEXT_MANAGEMENT_BETA,
        anthropic_request_headers,
    )

    config: JsonObject = {
        "edits": [
            {
                "type": "clear_tool_uses_20250919",
                "trigger": {"type": "input_tokens", "value": 30000},
                "keep": {"type": "tool_uses", "value": 3},
            }
        ]
    }
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="go"),),
        context_management=config,
        stream=True,
        include_usage=True,
    )
    payload = anthropic_messages_stream_payload("claude-fable-5", request)
    assert payload["context_management"] == config

    headers = anthropic_request_headers({"x-api-key": "k"}, request)
    assert headers["anthropic-beta"] == ANTHROPIC_CONTEXT_MANAGEMENT_BETA
    merged = anthropic_request_headers({"x-api-key": "k", "anthropic-beta": "other-beta"}, request)
    assert merged["anthropic-beta"] == f"other-beta,{ANTHROPIC_CONTEXT_MANAGEMENT_BETA}"
    bare = anthropic_request_headers(
        {"x-api-key": "k"},
        GatewayRequest(
            surface=GatewayApiSurface.MESSAGES,
            messages=(GatewayMessage(role="user", content="go"),),
        ),
    )
    assert "anthropic-beta" not in bare

    anthropic = GatewayWireProfile(dialect="anthropic_messages", url="https://anthropic.test")
    fallback = GatewayWireProfile(dialect="openai_compatible", url="https://fallback.test")
    public, provider = route_generation_parameter_requests((anthropic,), request)
    assert "context_management" not in public.ignored_parameters
    assert provider.context_management == config
    public, provider = route_generation_parameter_requests((fallback,), request)
    assert "context_management" in public.ignored_parameters
    assert provider.context_management is None


def test_tool_names_the_anthropic_wire_rejects_narrow_out_before_dispatch() -> None:
    """A dotted tool name is refused by name on Anthropic and Bedrock rungs
    (the provider's own 400 named tools.0.custom.name after dispatch, one
    org, 2026-09-07) and passes untouched on wires that accept it."""
    request = _chat_request().model_copy(
        update={
            "tools": (
                GatewayToolDefinition(name="search", parameters={"type": "object"}),
                GatewayToolDefinition(name="web.search", parameters={"type": "object"}),
            )
        }
    )
    anthropic = GatewayWireProfile(
        dialect="anthropic_messages", url="https://anthropic.test", model_id="claude-sonnet-4-5"
    )
    bedrock = GatewayWireProfile(
        dialect="bedrock_converse_stream",
        url="https://bedrock.test",
        model_id="anthropic.claude-sonnet-4-5-v1:0",
    )
    for profile in (anthropic, bedrock):
        with pytest.raises(ProviderParameterError) as rejected:
            route_generation_parameter_requests((profile,), request)
        assert rejected.value.param == "tools[1].name"
        assert "web.search" not in str(rejected.value)
    compatible = GatewayWireProfile(dialect="openai_compatible", url="https://fw.test")
    public, _provider = route_generation_parameter_requests((compatible,), request)
    assert public.ignored_parameters == ()


def test_assistant_prefill_narrows_out_rungs_whose_model_rejects_it() -> None:
    """A trailing assistant turn is refused BEFORE dispatch on the Anthropic
    releases that 400 it (live 2026-09-07: 4.6+ and every 5-generation
    release), in any id spelling, and passes everywhere else."""
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(
            GatewayMessage(role="user", content="Say a colour."),
            GatewayMessage(role="assistant", content="The colour is"),
        ),
        stream=True,
        include_usage=True,
    )
    rejecting = GatewayWireProfile(
        dialect="anthropic_messages", url="https://anthropic.test", model_id="claude-opus-5"
    )
    bedrock = GatewayWireProfile(
        dialect="bedrock_converse_stream",
        url="https://bedrock.test",
        model_id="anthropic.claude-fable-5-1-20260901-v1:0",
    )
    accepting = GatewayWireProfile(
        dialect="anthropic_messages", url="https://anthropic.test", model_id="claude-sonnet-4-5"
    )
    for profile in (rejecting, bedrock):
        with pytest.raises(ProviderParameterError) as prefill:
            route_generation_parameter_requests((profile,), request)
        assert prefill.value.param == "messages"
        assert "assistant prefill" in str(prefill.value)
        assert profile.model_id in str(prefill.value)
    public, _provider = route_generation_parameter_requests((accepting,), request)
    assert public.ignored_parameters == ()
    # Exact releases only: a later point release is NOT assumed from its
    # generation, while a dated snapshot and a Bedrock suffix inherit.
    assert anthropic_rejects_assistant_prefill("claude-opus-5-20260901")
    assert anthropic_rejects_assistant_prefill("anthropic.claude-sonnet-4-6-v1:0")
    assert not anthropic_rejects_assistant_prefill("claude-opus-5-1")
    assert not anthropic_rejects_assistant_prefill("claude-sonnet-5-2")
    assert not anthropic_rejects_assistant_prefill("claude-sonnet-4-5")
    # The same conversation ending in a user turn passes on the rejecting rung.
    user_last = request.model_copy(
        update={"messages": (*request.messages, GatewayMessage(role="user", content="go on"))}
    )
    route_generation_parameter_requests((rejecting,), user_last)


def test_mid_conversation_system_stays_positional_on_capable_wires() -> None:
    """A system turn after conversation start keeps its position on the
    Anthropic and OpenAI wires and narrows out instruction-hoisting rungs
    (Claude Code appends one by default; accepted live 2026-08-30)."""
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(
            GatewayMessage(role="system", content="lead instructions"),
            GatewayMessage(role="user", content="hi"),
            GatewayMessage(role="system", content="answer in uppercase"),
        ),
        stream=True,
        include_usage=True,
    )
    payload = anthropic_messages_stream_payload("claude-fable-5", request)
    assert payload["system"] == "lead instructions"
    assert payload["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "system", "content": [{"type": "text", "text": "answer in uppercase"}]},
    ]

    responses = openai_responses_stream_payload(
        "gpt-5.6-sol",
        request.model_copy(update={"surface": GatewayApiSurface.RESPONSES}),
        supports_temperature=False,
    )
    assert responses["instructions"] == "lead instructions"
    assert responses["input"] == [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "answer in uppercase"},
    ]

    anthropic = GatewayWireProfile(dialect="anthropic_messages", url="https://anthropic.test")
    gemini = GatewayWireProfile(dialect="gemini_generate_content", url="https://gemini.test")
    with pytest.raises(ProviderParameterError) as hoisting:
        route_generation_parameter_requests((anthropic, gemini), request)
    assert hoisting.value.param == "messages"
    public, _provider = route_generation_parameter_requests((anthropic,), request)
    assert public.ignored_parameters == ()


def test_output_config_seeds_the_payload_and_engine_keys_fill_gaps() -> None:
    """The caller's output_config wins byte-for-byte; the engine's derived
    effort and format only fill absent keys, so the two sources cannot
    fight (accepted live without a beta, 2026-08-30)."""
    caller = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="go"),),
        reasoning_effort="high",
        provider_output_config={"effort": "high", "future_key": 1},
        stream=True,
        include_usage=True,
    )
    payload = anthropic_messages_stream_payload(
        "claude-fable-5", caller, supports_reasoning=True, reasoning_effort="low"
    )
    # Caller effort survives verbatim over the catalog-pinned "low".
    assert payload["output_config"] == {"effort": "high", "future_key": 1}
    assert payload["thinking"] == {"type": "adaptive"}

    pinned_only = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="go"),),
        provider_output_config={"future_key": 1},
        stream=True,
        include_usage=True,
    )
    filled = anthropic_messages_stream_payload(
        "claude-fable-5", pinned_only, supports_reasoning=True, reasoning_effort="low"
    )
    assert filled["output_config"] == {"future_key": 1, "effort": "low"}


def test_output_config_discloses_on_routes_that_cannot_honor_it() -> None:
    """An effort-only canonical config rides reasoning_effort everywhere; any
    richer config drops with disclosure when a rung is not Anthropic."""
    anthropic = GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://anthropic.test",
        model_id="claude-fable-5",
        supports_reasoning=True,
        reasoning_wire_format="anthropic_adaptive",
    )
    fallback = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://fallback.test",
        model_id="gpt-5.1",
        supports_reasoning=True,
        reasoning_wire_format="reasoning_effort",
    )
    effort_only = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="go"),),
        reasoning_effort="high",
        provider_output_config={"effort": "high"},
        stream=True,
        include_usage=True,
    )
    public, provider = route_generation_parameter_requests((anthropic, fallback), effort_only)
    assert public.ignored_parameters == ()
    assert provider.reasoning_effort == "high"

    richer = effort_only.model_copy(
        update={"provider_output_config": {"effort": "high", "format": {"type": "json_schema"}}}
    )
    public, provider = route_generation_parameter_requests((anthropic, fallback), richer)
    assert "output_config" in public.ignored_parameters
    assert provider.provider_output_config is None


def test_native_items_require_a_homogeneous_responses_route_and_reemit_verbatim() -> None:
    """Codex-native input items forward byte-for-byte on native Responses
    rungs at their exact position; any other rung in the route is a named
    rejection (dropping tool definitions would silently degrade the agent)."""
    native_item: JsonObject = {
        "type": "custom_tool_call",
        "id": "ctc_1",
        "call_id": "call_1",
        "name": "exec",
        "input": "const r = 1;",
    }
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(
            GatewayMessage(role="user", content="go"),
            GatewayMessage(role="assistant", provider_native_item=native_item),
        ),
        stream=True,
        include_usage=True,
    )
    payload = openai_responses_stream_payload("gpt-5.6-sol", request, supports_temperature=False)
    assert payload["input"] == [{"role": "user", "content": "go"}, native_item]

    responses = GatewayWireProfile(dialect="openai_responses", url="https://openai.test")
    chat = GatewayWireProfile(dialect="openai_compatible", url="https://chat.test")
    public, _provider = route_generation_parameter_requests((responses,), request)
    assert public.ignored_parameters == ()
    with pytest.raises(ProviderParameterError) as mixed:
        route_generation_parameter_requests((responses, chat), request)
    assert mixed.value.param == "input"


def test_client_metadata_and_verbosity_forward_native_and_disclose_elsewhere() -> None:
    """Codex's telemetry object and verbosity hint ride the native Responses
    wire verbatim and drop with disclosure on any other route."""
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="go"),),
        text_verbosity="low",
        client_metadata={"thread_id": "t-1"},
        stream=True,
        include_usage=True,
    )
    payload = openai_responses_stream_payload("gpt-5.6-sol", request, supports_temperature=False)
    assert payload["client_metadata"] == {"thread_id": "t-1"}
    assert payload["text"] == {"verbosity": "low"}

    responses = GatewayWireProfile(dialect="openai_responses", url="https://openai.test")
    chat = GatewayWireProfile(dialect="openai_compatible", url="https://chat.test")
    public, provider = route_generation_parameter_requests((responses,), request)
    assert public.ignored_parameters == ()
    public, provider = route_generation_parameter_requests((responses, chat), request)
    assert set(public.ignored_parameters) == {"client_metadata", "text.verbosity"}
    assert provider.client_metadata is None
    assert provider.text_verbosity is None


def test_diagnostics_speed_and_betas_forward_on_anthropic_and_disclose_elsewhere() -> None:
    """The conditional Claude Code carriers ride Anthropic rungs verbatim
    with their required beta tokens merged into one header; a route with
    any other rung drops each with disclosure, never a rejection."""
    from exp.runtime.models.providers.wire_messages import (
        ANTHROPIC_DIAGNOSTICS_BETA,
        ANTHROPIC_FAST_MODE_BETA,
        anthropic_request_headers,
    )

    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="go"),),
        diagnostics={"previous_message_id": "msg_prior"},
        speed="fast",
        provider_beta_tokens=("context-1m-2025-08-07",),
        stream=True,
        include_usage=True,
    )
    payload = anthropic_messages_stream_payload("claude-fable-5", request)
    assert payload["diagnostics"] == {"previous_message_id": "msg_prior"}
    assert payload["speed"] == "fast"

    headers = anthropic_request_headers({"anthropic-beta": "operator-token"}, request)
    assert headers["anthropic-beta"] == (
        "operator-token,context-1m-2025-08-07,"
        f"{ANTHROPIC_DIAGNOSTICS_BETA},{ANTHROPIC_FAST_MODE_BETA}"
    )

    anthropic = GatewayWireProfile(dialect="anthropic_messages", url="https://anthropic.test")
    fallback = GatewayWireProfile(dialect="openai_compatible", url="https://fallback.test")
    public, provider = route_generation_parameter_requests((anthropic,), request)
    assert public.ignored_parameters == ()
    assert provider.diagnostics == {"previous_message_id": "msg_prior"}
    assert provider.speed == "fast"
    assert provider.provider_beta_tokens == ("context-1m-2025-08-07",)

    mixed_public, mixed_provider = route_generation_parameter_requests(
        (anthropic, fallback), request
    )
    assert set(mixed_public.ignored_parameters) == {
        "diagnostics",
        "speed",
        "anthropic-beta.context-1m-2025-08-07",
    }
    assert mixed_provider.diagnostics is None
    assert mixed_provider.speed is None
    assert mixed_provider.provider_beta_tokens == ()


def test_tool_annotations_and_top_carriers_forward_on_anthropic_and_disclose_elsewhere() -> None:
    """Provider-native tool annotations and inference region reach only the
    Anthropic wire; any other rung drops each with a per-field disclosure,
    never a rejection (a production Claude Code session sent
    ``eager_input_streaming`` and was 400ed). The top-level cache marker is the
    exception: it is cost-only and honored on any Anthropic rung, so a mixed
    waterfall keeps it rather than billing every turn uncached."""
    from exp.runtime.gateway.contracts import GatewayToolDefinition

    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="run"),),
        tools=(
            GatewayToolDefinition(
                name="Bash",
                description="Executes a bash command.",
                parameters={"type": "object"},
                cache_control={"type": "ephemeral"},
                eager_input_streaming=True,
                defer_loading=False,
                allowed_callers=("code_execution_20260120",),
                input_examples=({"command": "ls"},),
            ),
        ),
        provider_cache_control={"type": "ephemeral"},
        inference_geo="us",
        stream=True,
        include_usage=True,
    )

    payload = anthropic_messages_stream_payload("claude-fable-5", request)
    tool = cast(list[JsonObject], payload["tools"])[0]
    assert tool["cache_control"] == {"type": "ephemeral"}
    assert tool["eager_input_streaming"] is True
    assert tool["defer_loading"] is False
    assert tool["allowed_callers"] == ["code_execution_20260120"]
    assert tool["input_examples"] == [{"command": "ls"}]
    assert payload["cache_control"] == {"type": "ephemeral"}
    assert payload["inference_geo"] == "us"

    # None of these accepted fields require a beta token (verified live
    # 2026-08-30), so the dispatch headers stay untouched.
    from exp.runtime.models.providers.wire_messages import anthropic_request_headers

    assert anthropic_request_headers({}, request) == {}

    # OpenAI-family wires never see the annotations; the route discloses
    # each dropped field by name.
    import json

    chat_payload = openai_compatible_stream_payload("gpt-5.5", request)
    serialized = json.dumps(chat_payload)
    for marker in (
        "cache_control",
        "eager_input_streaming",
        "defer_loading",
        "allowed_callers",
        "input_examples",
        "inference_geo",
    ):
        assert marker not in serialized

    anthropic = GatewayWireProfile(dialect="anthropic_messages", url="https://anthropic.test")
    fallback = GatewayWireProfile(dialect="openai_compatible", url="https://fallback.test")
    public, provider = route_generation_parameter_requests((anthropic,), request)
    assert public.ignored_parameters == ()
    assert provider.provider_cache_control == {"type": "ephemeral"}
    assert provider.inference_geo == "us"

    mixed_public, mixed_provider = route_generation_parameter_requests(
        (anthropic, fallback), request
    )
    # The top-level cache marker is COST-only and honored on the Anthropic rung,
    # so a mixed waterfall keeps it (dropping it billed every turn uncached);
    # the behavioral annotations still drop with disclosure on the non-Anthropic
    # rungs.
    assert set(mixed_public.ignored_parameters) == {
        "inference_geo",
        "tools.cache_control",
        "tools.eager_input_streaming",
        "tools.defer_loading",
        "tools.allowed_callers",
        "tools.input_examples",
    }
    assert mixed_provider.provider_cache_control == {"type": "ephemeral"}
    assert mixed_provider.inference_geo is None


def _web_search_messages_request(
    *,
    tool_choice: Literal["auto", "none", "required"] | GatewayNamedToolChoice | None = None,
    echoed_block: bool = False,
) -> GatewayRequest:
    """Build one Messages request carrying a verbatim web_search server tool."""
    messages: tuple[GatewayMessage, ...] = (GatewayMessage(role="user", content="search"),)
    if echoed_block:
        messages += (
            GatewayMessage(
                role="assistant",
                provider_anthropic_block={
                    "type": "server_tool_use",
                    "id": "srvtoolu_1",
                    "name": "web_search",
                    "input": {"query": "python"},
                },
            ),
            GatewayMessage(role="user", content="and now?"),
        )
    return GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=messages,
        provider_server_tools=(
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 8},
        ),
        tool_choice=tool_choice,
        maximum_output_tokens=256,
        maximum_output_tokens_parameter="max_tokens",
        stream=True,
        include_usage=True,
    )


def _anthropic_web_search_profile(url: str = "https://anthropic.test") -> GatewayWireProfile:
    """Return one native Anthropic Messages wire profile."""
    return GatewayWireProfile(
        dialect="anthropic_messages",
        url=url,
        model_id="claude-haiku-4-5",
        reasoning_wire_format="anthropic_adaptive",
    )


def test_server_tools_reject_mixed_routes_by_name() -> None:
    """A route with any non-Anthropic rung cannot serve server tools.

    Rejection, not disclosure-drop: silently removing a search capability
    the caller asked for would falsify every answer that needed it.
    """
    profiles = (
        _anthropic_web_search_profile(),
        GatewayWireProfile(dialect="openai_compatible", url="https://fallback.test"),
    )
    for request in (
        _web_search_messages_request(),
        _web_search_messages_request(echoed_block=True).model_copy(
            update={"provider_server_tools": ()}
        ),
    ):
        with pytest.raises(ProviderParameterError) as raised:
            route_generation_parameter_requests(profiles, request)
        assert raised.value.code == "unsupported_parameter"
        assert raised.value.param == "tools"
        # The rejection names the exact tool, its Claude Code label, and the
        # Anthropic-only rule, so a Claude Code user knows what to change.
        message = str(raised.value)
        assert "'web_search' (Claude Code's WebSearch)" in message
        assert "Anthropic only allows them on Anthropic (Claude) models" in message
        assert "Switch to a Claude model alias" in message


def test_server_tools_keep_tool_choice_on_an_anthropic_route() -> None:
    """Server tools count as tool definitions for the no-op selector rule."""
    profiles = (
        _anthropic_web_search_profile(),
        _anthropic_web_search_profile("https://anthropic-b.test"),
    )
    for tool_choice in ("auto", "required", GatewayNamedToolChoice(name="web_search")):
        request = _web_search_messages_request(tool_choice=tool_choice)
        public_request, provider_request = route_generation_parameter_requests(profiles, request)
        assert "tool_choice" not in public_request.ignored_parameters
        assert provider_request.tool_choice == tool_choice
        assert provider_request.provider_server_tools == request.provider_server_tools


def test_anthropic_payload_appends_server_tools_verbatim_after_custom_tools() -> None:
    """Server tool entries re-emit byte-for-byte after the converted tools."""
    request = _web_search_messages_request(tool_choice="auto").model_copy(
        update={
            "tools": (GatewayToolDefinition(name="Bash", parameters={"type": "object"}),),
        }
    )
    payload = anthropic_messages_stream_payload("claude-haiku-4-5", request)
    assert payload["tools"] == [
        {"name": "Bash", "input_schema": {"type": "object"}},
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 8},
    ]
    assert payload["tool_choice"] == {"type": "auto"}


def test_anthropic_payload_serves_a_server_tool_only_toolset() -> None:
    """A request whose only tools are server tools still sends a tools array."""
    payload = anthropic_messages_stream_payload("claude-haiku-4-5", _web_search_messages_request())
    assert payload["tools"] == [
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 8}
    ]


def test_anthropic_payload_reemits_echoed_server_blocks_in_order() -> None:
    """Echoed server-tool blocks re-emit verbatim at their positions."""
    cited_text: JsonObject = {
        "citations": [{"type": "web_search_result_location", "encrypted_index": "Eo8B"}],
        "type": "text",
        "text": "It is 3.14.7.",
    }
    result_block: JsonObject = {
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_1",
        "content": [{"type": "web_search_result", "encrypted_content": "Et8Q"}],
        "caller": {"type": "direct"},
    }
    request = _web_search_messages_request(echoed_block=True)
    messages = (
        request.messages[:2]
        + (
            GatewayMessage(role="assistant", provider_anthropic_block=result_block),
            GatewayMessage(role="assistant", provider_anthropic_block=cited_text),
        )
        + request.messages[2:]
    )
    payload = anthropic_messages_stream_payload(
        "claude-haiku-4-5", request.model_copy(update={"messages": messages})
    )
    wire_messages = cast(list[JsonObject], payload["messages"])
    assert [message["role"] for message in wire_messages] == ["user", "assistant", "user"]
    # Consecutive assistant carrier messages merge back into one turn with
    # the exact echoed block order.
    assert wire_messages[1]["content"] == [
        {
            "type": "server_tool_use",
            "id": "srvtoolu_1",
            "name": "web_search",
            "input": {"query": "python"},
        },
        result_block,
        cited_text,
    ]


def test_block_cache_markers_reach_the_anthropic_wire_and_survive_mixed_routes() -> None:
    """The caller's block structure re-emits exactly where markers exist and
    only there; markerless payloads stay byte-identical, and per the #699
    rule a mixed waterfall keeps the markers for its Anthropic rung."""
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(
            GatewayMessage(
                role="system",
                content="You are Claude Code.\n\nLong env block.",
                provider_text_blocks=(
                    {"type": "text", "text": "You are Claude Code."},
                    {
                        "type": "text",
                        "text": "Long env block.",
                        "cache_control": {"type": "ephemeral"},
                    },
                ),
            ),
            GatewayMessage(
                role="user",
                content="contextdo the thing",
                provider_text_blocks=(
                    {"type": "text", "text": "context"},
                    {
                        "type": "text",
                        "text": "do the thing",
                        "cache_control": {"type": "ephemeral"},
                    },
                ),
            ),
            GatewayMessage(role="assistant", content="ran it"),
            GatewayMessage(
                role="tool",
                content="ok",
                tool_call_id="call-1",
                cache_control={"type": "ephemeral"},
            ),
        ),
        stream=True,
        include_usage=True,
    )
    payload = anthropic_messages_stream_payload("claude-fable-5", request)
    # The canonical blank-line separator folds into the following block
    # (the provider rejects whitespace-only blocks), so the system TEXT
    # equals the unmarked join with markers on their blocks.
    assert payload["system"] == [
        {"type": "text", "text": "You are Claude Code."},
        {"type": "text", "text": "\n\nLong env block.", "cache_control": {"type": "ephemeral"}},
    ]
    messages = cast(list[JsonObject], payload["messages"])
    user_blocks = cast(list[JsonObject], messages[0]["content"])
    assert user_blocks == [
        {"type": "text", "text": "context"},
        {"type": "text", "text": "do the thing", "cache_control": {"type": "ephemeral"}},
    ]
    tool_blocks = cast(list[JsonObject], messages[2]["content"])
    assert tool_blocks[0]["cache_control"] == {"type": "ephemeral"}

    # Markerless requests keep the exact pre-change wire shape.
    plain = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(
            GatewayMessage(role="system", content="a\n\nb"),
            GatewayMessage(role="user", content="hi"),
        ),
        stream=True,
        include_usage=True,
    )
    plain_payload = anthropic_messages_stream_payload("claude-fable-5", plain)
    assert plain_payload["system"] == "a\n\nb"
    plain_messages = cast(list[JsonObject], plain_payload["messages"])
    assert plain_messages[0]["content"] == [{"type": "text", "text": "hi"}]

    anthropic = GatewayWireProfile(dialect="anthropic_messages", url="https://anthropic.test")
    fallback = GatewayWireProfile(dialect="openai_compatible", url="https://fallback.test")
    mixed_public, mixed_provider = route_generation_parameter_requests(
        (anthropic, fallback), request
    )
    assert "messages.content.cache_control" not in mixed_public.ignored_parameters
    assert mixed_provider.messages[0].provider_text_blocks
    foreign_public, _foreign_provider = route_generation_parameter_requests((fallback,), request)
    assert "messages.content.cache_control" in foreign_public.ignored_parameters


def test_litellm_provider_specific_fields_are_dropped_with_disclosure_on_every_route() -> None:
    """A populated LiteLLM stamp never reaches a provider and is always disclosed.

    No wire has a field for it, so unlike cache markers the disclosure does
    not depend on the route's dialects; an empty stamp decodes to nothing and
    discloses nothing.
    """
    stamped = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(
            GatewayMessage(role="user", content="Reply with a command."),
            GatewayMessage(
                role="assistant",
                content='{"command": "ls"}',
                provider_specific_fields={"refusal": None},
            ),
            GatewayMessage(role="user", content="Output: a.txt"),
        ),
    )
    compatible = GatewayWireProfile(dialect="openai_compatible", url="https://hy4.test")
    anthropic = GatewayWireProfile(dialect="anthropic_messages", url="https://anthropic.test")
    for profiles in ((compatible,), (anthropic,), (compatible, anthropic)):
        public, provider = route_generation_parameter_requests(profiles, stamped)
        assert "messages.provider_specific_fields" in public.ignored_parameters
        payload = openai_compatible_stream_payload("hy4-preview", provider)
        messages = cast(list[JsonObject], payload["messages"])
        assert all("provider_specific_fields" not in message for message in messages)
    plain = stamped.model_copy(
        update={
            "messages": tuple(
                message.model_copy(update={"provider_specific_fields": None})
                for message in stamped.messages
            )
        }
    )
    public, _provider = route_generation_parameter_requests((compatible,), plain)
    assert "messages.provider_specific_fields" not in public.ignored_parameters


def test_prompt_cache_affinity_key_is_forwarded_only_where_the_rung_routes_by_it() -> None:
    """The derived key rides ``prompt_cache_key`` on flagged rungs and nowhere else.

    The caller's own ``prompt_cache_key`` is never the dispatched value: only
    the admission-derived ``provider_prompt_cache_key`` is, and only when the
    builder is told the rung's provider routes by it.
    """
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hi"),),
        prompt_cache_key="caller-session",
        provider_prompt_cache_key="xpl-derived",
        stream=True,
        include_usage=True,
    )
    forwarded = openai_compatible_stream_payload(
        "hy4-preview", request, forwards_prompt_cache_key=True
    )
    assert forwarded["prompt_cache_key"] == "xpl-derived"
    withheld = openai_compatible_stream_payload("hy4-preview", request)
    assert "prompt_cache_key" not in withheld
    responses = openai_responses_stream_payload(
        "gpt-5.5", request, supports_temperature=True, forwards_prompt_cache_key=True
    )
    assert responses["prompt_cache_key"] == "xpl-derived"
    responses_withheld = openai_responses_stream_payload(
        "gpt-5.5", request, supports_temperature=True
    )
    assert "prompt_cache_key" not in responses_withheld
    # No derived key (admission found neither a caller key nor a stem): nothing
    # is dispatched even on a flagged rung.
    unkeyed = request.model_copy(update={"provider_prompt_cache_key": None})
    assert "prompt_cache_key" not in openai_compatible_stream_payload(
        "hy4-preview", unkeyed, forwards_prompt_cache_key=True
    )


def test_block_cache_markers_survive_a_multimodal_user_turn() -> None:
    """An image in the marked turn must not cost the caller its cache prefix.

    Claude Code marks the last text block of recent user turns, and a turn
    that attaches a screenshot is exactly such a turn: the image joins the
    caller's blocks at its position and the marker re-emits with its text.
    """
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(
            GatewayMessage(
                role="user",
                content="contextdescribe this",
                content_parts=(
                    TextContentPart(text="context"),
                    ImageContentPart(media_type="image/png", data=_PNG_BASE64),
                    TextContentPart(text="describe this"),
                ),
                provider_text_blocks=(
                    {"type": "text", "text": "context"},
                    {
                        "type": "text",
                        "text": "describe this",
                        "cache_control": {"type": "ephemeral"},
                    },
                ),
            ),
        ),
        stream=True,
        include_usage=True,
    )
    payload = anthropic_messages_stream_payload("claude-fable-5", request)
    messages = cast(list[JsonObject], payload["messages"])
    assert messages[0]["content"] == [
        {"type": "text", "text": "context"},
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": _PNG_BASE64},
        },
        {"type": "text", "text": "describe this", "cache_control": {"type": "ephemeral"}},
    ]


def test_marked_system_prompt_keeps_the_exact_unmarked_text_bytes() -> None:
    """Marked and unmarked payloads carry byte-identical system TEXT.

    Cache markers must never change the instructions the model reads: with a
    marked top-level system followed by a leading system-role turn, the
    block-path text (blocks concatenated in order) equals the unmarked
    joined string exactly, separator included, and the only difference is
    the markers themselves.
    """

    def request(marked: bool) -> GatewayRequest:
        blocks: tuple[JsonObject, ...] = (
            (
                {"type": "text", "text": "You are Claude Code."},
                {
                    "type": "text",
                    "text": "Long env block.",
                    "cache_control": {"type": "ephemeral"},
                },
            )
            if marked
            else ()
        )
        return GatewayRequest(
            surface=GatewayApiSurface.MESSAGES,
            messages=(
                GatewayMessage(
                    role="system",
                    content="You are Claude Code.\n\nLong env block.",
                    provider_text_blocks=blocks,
                ),
                GatewayMessage(role="system", content="Leading turn instruction."),
                GatewayMessage(role="user", content="hi"),
            ),
            stream=True,
            include_usage=True,
        )

    unmarked_payload = anthropic_messages_stream_payload("claude-fable-5", request(False))
    marked_payload = anthropic_messages_stream_payload("claude-fable-5", request(True))
    unmarked_system = cast(str, unmarked_payload["system"])
    marked_system = cast(list[JsonObject], marked_payload["system"])
    assert "".join(str(block["text"]) for block in marked_system) == unmarked_system
    marked_controls = [block.get("cache_control") for block in marked_system]
    assert marked_controls.count({"type": "ephemeral"}) == 1
    assert marked_system[1] == {
        "type": "text",
        "text": "\n\nLong env block.",
        "cache_control": {"type": "ephemeral"},
    }


def test_native_tool_declarations_require_a_homogeneous_responses_route() -> None:
    """Non-function tool declarations (custom, namespace, web_search,
    tool_search) forward byte-for-byte at their caller positions on native
    Responses rungs; any other rung in the route is a named rejection
    (dropping an agent's tool definitions would silently degrade it)."""
    custom_tool: JsonObject = {
        "type": "custom",
        "name": "apply_patch",
        "description": "Edit files.",
        "format": {"type": "grammar", "syntax": "lark", "definition": "start: x"},
    }
    web_search_tool: JsonObject = {"type": "web_search", "external_web_access": False}
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="go"),),
        tools=(
            GatewayToolDefinition(name="exec_command", parameters={"type": "object"}),
            GatewayToolDefinition(name="view_image", parameters={"type": "object"}),
        ),
        provider_native_tools=(
            GatewayProviderNativeTool(index=1, tool=custom_tool),
            GatewayProviderNativeTool(index=3, tool=web_search_tool),
        ),
        stream=True,
        include_usage=True,
    )
    payload = openai_responses_stream_payload("gpt-5.2", request, supports_temperature=False)
    tools = payload["tools"]
    assert isinstance(tools, list)
    assert [entry.get("type") for entry in tools if isinstance(entry, dict)] == [
        "function",
        "custom",
        "function",
        "web_search",
    ]
    assert tools[1] == custom_tool
    assert tools[3] == web_search_tool

    responses = GatewayWireProfile(dialect="openai_responses", url="https://openai.test")
    chat = GatewayWireProfile(dialect="openai_compatible", url="https://chat.test")
    public, _provider = route_generation_parameter_requests((responses,), request)
    assert public.ignored_parameters == ()
    with pytest.raises(ProviderParameterError) as mixed:
        route_generation_parameter_requests((responses, chat), request)
    assert mixed.value.param == "tools"


def test_native_tool_declarations_count_as_tools_for_capability_preflight() -> None:
    """A rung that declares no tool support rejects a native-tools-only
    request locally instead of dispatching a known-unsupported call."""
    from exp.common.models import GatewayDeploymentCapabilities, ModelCapabilities
    from exp.runtime.models.providers.protocol import preflight_gateway_request

    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="go"),),
        provider_native_tools=(GatewayProviderNativeTool(index=0, tool={"type": "web_search"}),),
        stream=True,
        include_usage=True,
    )
    capabilities = GatewayDeploymentCapabilities(supports_streaming=True)
    with pytest.raises(ProviderCapabilityError) as rejection:
        preflight_gateway_request(
            request,
            capabilities,
            model_capabilities=ModelCapabilities(supports_tools=False),
        )
    assert rejection.value.capability == "function_tools"


_MP4_BASE64 = "AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDE="
"""A base64 prefix of an MP4 ``ftyp`` box, enough for a carrier fixture."""


def _video_request(
    surface: GatewayApiSurface = GatewayApiSurface.CHAT_COMPLETIONS,
    *,
    remote: bool = False,
) -> GatewayRequest:
    """Build one streaming request with text on both sides of a video."""
    video = (
        VideoContentPart(url="https://example.com/clip.mp4")
        if remote
        else VideoContentPart(media_type="video/mp4", data=_MP4_BASE64)
    )
    return GatewayRequest(
        surface=surface,
        messages=(
            GatewayMessage(
                role="user",
                content="watch this: what happens?",
                content_parts=(
                    TextContentPart(text="watch this: "),
                    video,
                    TextContentPart(text="what happens?"),
                ),
            ),
        ),
        stream=True,
        include_usage=True,
    )


def _tiered_request(surface: GatewayApiSurface) -> GatewayRequest:
    """Build one streaming request carrying an explicit service tier."""
    return GatewayRequest(
        surface=surface,
        messages=(GatewayMessage(role="user", content="hello"),),
        service_tier="flex",
        stream=True,
        include_usage=True,
    )


def test_openai_compatible_payload_carries_video_url_parts_in_order() -> None:
    """The OpenRouter and Fireworks wire gets a ``video_url`` part between its text."""
    payload = openai_compatible_stream_payload("qwen3-omni", _video_request())
    messages = cast(list[JsonObject], payload["messages"])
    assert messages[0]["content"] == [
        {"type": "text", "text": "watch this: "},
        {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{_MP4_BASE64}"}},
        {"type": "text", "text": "what happens?"},
    ]
    remote = openai_compatible_stream_payload("qwen3-omni", _video_request(remote=True))
    remote_content = cast(
        list[JsonObject], cast(list[JsonObject], remote["messages"])[0]["content"]
    )
    assert remote_content[1] == {
        "type": "video_url",
        "video_url": {"url": "https://example.com/clip.mp4"},
    }


def test_gemini_payload_carries_inline_and_file_video_parts() -> None:
    """Gemini gets ``inline_data`` for bytes and ``file_data`` for a fetched URI."""
    payload = gemini_generate_content_stream_payload("gemini-2.5-flash", _video_request())
    assert payload["contents"] == [
        {
            "role": "user",
            "parts": [
                {"text": "watch this: "},
                {"inline_data": {"mime_type": "video/mp4", "data": _MP4_BASE64}},
                {"text": "what happens?"},
            ],
        }
    ]
    remote = gemini_generate_content_stream_payload("gemini-2.5-flash", _video_request(remote=True))
    parts = cast(list[JsonObject], cast(list[JsonObject], remote["contents"])[0]["parts"])
    assert parts[1] == {"file_data": {"file_uri": "https://example.com/clip.mp4"}}


def test_bedrock_payload_carries_a_video_block_and_declines_a_url() -> None:
    """Converse gets a ``video`` block for bytes and narrows past a URL it cannot fetch."""
    payload = bedrock_converse_stream_payload("us.amazon.nova-lite-v1:0", _video_request())
    assert payload["messages"] == [
        {
            "role": "user",
            "content": [
                {"text": "watch this: "},
                {"video": {"format": "mp4", "source": {"bytes": _MP4_BASE64}}},
                {"text": "what happens?"},
            ],
        }
    ]
    with pytest.raises(ProviderCapabilityError, match="video_url_input"):
        bedrock_converse_stream_payload("us.amazon.nova-lite-v1:0", _video_request(remote=True))


def test_wires_without_a_video_carrier_narrow_past_the_rung() -> None:
    """Responses and Anthropic payloads refuse a video instead of dropping it."""
    with pytest.raises(ProviderCapabilityError, match="video_input"):
        openai_responses_stream_payload(
            "gpt-fixture",
            _video_request(),
            supports_temperature=True,
            supports_reasoning=False,
            reasoning_effort=None,
        )
    with pytest.raises(ProviderCapabilityError, match="video_input"):
        anthropic_messages_stream_payload("claude-fable-5", _video_request())


_PDF_BASE64 = "JVBERi0xLjQKJSBtaW5pbWFsIHBkZgo="
"""One short PDF header, base64 encoded."""


def _document_request(surface: GatewayApiSurface) -> GatewayRequest:
    """Build one streaming request with two PDFs interleaved with text."""
    return GatewayRequest(
        surface=surface,
        messages=(
            GatewayMessage(
                role="user",
                content="first: second: compare",
                content_parts=(
                    TextContentPart(text="first:"),
                    DocumentContentPart(data=_PDF_BASE64, name="a.pdf"),
                    TextContentPart(text=" second:"),
                    DocumentContentPart(data="JVBERi0xLjcK"),
                    TextContentPart(text=" compare"),
                ),
            ),
        ),
        stream=True,
        include_usage=True,
    )


def test_openai_chat_payload_carries_file_parts_in_caller_order() -> None:
    """The Chat wire carries each PDF as a ``file`` part at its position."""
    payload = openai_compatible_stream_payload(
        "exact-model", _document_request(GatewayApiSurface.CHAT_COMPLETIONS)
    )
    messages = cast(list[JsonObject], payload["messages"])
    assert messages[0]["content"] == [
        {"type": "text", "text": "first:"},
        {
            "type": "file",
            "file": {
                "filename": "a.pdf",
                "file_data": f"data:application/pdf;base64,{_PDF_BASE64}",
            },
        },
        {"type": "text", "text": " second:"},
        {
            "type": "file",
            "file": {
                "filename": "document.pdf",
                "file_data": "data:application/pdf;base64,JVBERi0xLjcK",
            },
        },
        {"type": "text", "text": " compare"},
    ]


def test_openai_responses_payload_carries_input_file_parts_in_caller_order() -> None:
    """The Responses wire carries each PDF as an ``input_file`` part at its position."""
    payload = openai_responses_stream_payload(
        "gpt-5.2",
        _document_request(GatewayApiSurface.RESPONSES),
        supports_temperature=False,
    )
    items = cast(list[JsonObject], payload["input"])
    assert items[0]["content"] == [
        {"type": "input_text", "text": "first:"},
        {
            "type": "input_file",
            "filename": "a.pdf",
            "file_data": f"data:application/pdf;base64,{_PDF_BASE64}",
        },
        {"type": "input_text", "text": " second:"},
        {
            "type": "input_file",
            "filename": "document.pdf",
            "file_data": "data:application/pdf;base64,JVBERi0xLjcK",
        },
        {"type": "input_text", "text": " compare"},
    ]


def test_anthropic_payload_carries_document_blocks_in_caller_order() -> None:
    """The Messages wire carries each PDF as a ``document`` block at its position."""
    payload = anthropic_messages_stream_payload(
        "claude-fable-5", _document_request(GatewayApiSurface.MESSAGES)
    )
    messages = cast(list[JsonObject], payload["messages"])
    assert messages[0]["content"] == [
        {"type": "text", "text": "first:"},
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": _PDF_BASE64,
            },
            "title": "a.pdf",
        },
        {"type": "text", "text": " second:"},
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": "JVBERi0xLjcK",
            },
        },
        {"type": "text", "text": " compare"},
    ]


def test_anthropic_files_handles_add_the_files_api_beta_header() -> None:
    """A request carrying an Anthropic Files handle dispatches with the beta token."""
    from exp.runtime.models.providers.wire_messages import (
        ANTHROPIC_FILES_API_BETA,
        anthropic_request_headers,
    )

    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(
            GatewayMessage(
                role="user",
                content="describe",
                content_parts=(
                    ImageContentPart(handle=MediaHandle(provider="anthropic", reference="file_a")),
                    TextContentPart(text="describe"),
                ),
            ),
        ),
    )
    headers = anthropic_request_headers({"x-api-key": "k"}, request)
    assert headers["anthropic-beta"] == ANTHROPIC_FILES_API_BETA
    merged = anthropic_request_headers(
        {"x-api-key": "k", "anthropic-beta": ANTHROPIC_FILES_API_BETA}, request
    )
    assert merged["anthropic-beta"] == ANTHROPIC_FILES_API_BETA


_WAV_BASE64 = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQAAAAA="
"""A 44-byte WAV header with an empty data chunk, base64 encoded."""


def _audio_request(
    surface: GatewayApiSurface = GatewayApiSurface.CHAT_COMPLETIONS,
) -> GatewayRequest:
    """Build one streaming request with text on both sides of a clip."""
    return GatewayRequest(
        surface=surface,
        messages=(
            GatewayMessage(
                role="user",
                content="listen: what is said?",
                content_parts=(
                    TextContentPart(text="listen: "),
                    AudioContentPart(media_type="audio/wav", data=_WAV_BASE64),
                    TextContentPart(text="what is said?"),
                ),
            ),
        ),
        stream=True,
        include_usage=True,
    )


def test_openai_compatible_payload_carries_input_audio_parts_in_order() -> None:
    """The Chat wire (OpenRouter, Azure, gpt-audio) gets ``input_audio`` between its text."""
    payload = openai_compatible_stream_payload("gpt-audio-mini", _audio_request())
    messages = cast(list[JsonObject], payload["messages"])
    assert messages[0]["content"] == [
        {"type": "text", "text": "listen: "},
        {"type": "input_audio", "input_audio": {"data": _WAV_BASE64, "format": "wav"}},
        {"type": "text", "text": "what is said?"},
    ]


def test_gemini_payload_carries_inline_audio_parts() -> None:
    """Gemini gets ``inline_data`` with the audio MIME type at the clip's position."""
    payload = gemini_generate_content_stream_payload("gemini-3-flash-preview", _audio_request())
    assert payload["contents"] == [
        {
            "role": "user",
            "parts": [
                {"text": "listen: "},
                {"inline_data": {"mime_type": "audio/wav", "data": _WAV_BASE64}},
                {"text": "what is said?"},
            ],
        }
    ]


def test_wires_without_an_audio_carrier_narrow_past_the_rung() -> None:
    """Responses, Anthropic, and Bedrock payloads refuse a clip instead of dropping it."""
    with pytest.raises(ProviderCapabilityError, match="audio_input"):
        openai_responses_stream_payload(
            "gpt-fixture",
            _audio_request(),
            supports_temperature=True,
            supports_reasoning=False,
            reasoning_effort=None,
        )
    with pytest.raises(ProviderCapabilityError, match="audio_input"):
        anthropic_messages_stream_payload("claude-fable-5", _audio_request())
    with pytest.raises(ProviderCapabilityError, match="audio_input"):
        bedrock_converse_stream_payload("us.amazon.nova-lite-v1:0", _audio_request())


@pytest.mark.parametrize(
    ("dialect", "surface"),
    (
        ("openai_compatible", GatewayApiSurface.CHAT_COMPLETIONS),
        ("openai_responses", GatewayApiSurface.RESPONSES),
    ),
)
def test_service_tier_forwards_on_byok_tier_preserving_dialects(
    dialect: str,
    surface: GatewayApiSurface,
) -> None:
    """Both OpenAI wire dialects carry the caller's tier verbatim on BYOK rungs.

    Adapted from the tier-preserving-dialect test: forwarding now also
    requires tenant-owned (BYOK) credentials, because on host-funded rungs
    the tier changes what the provider charges (flex discounted, priority
    premium) while the gateway bills catalog rates.
    """
    byok = GatewayWireProfile(
        dialect=dialect,
        url="https://provider.test",
        model_id="model-x",
        billing_customer_managed=True,
    )
    payload = dialect_stream_payload(byok, _tiered_request(surface))
    assert payload["service_tier"] == "flex"

    # The same dialect on house-funded credentials serves WITHOUT the tier:
    # a structural non-emission, never a decline, so the rung stays in the
    # route as a fallback.
    hosted = GatewayWireProfile(dialect=dialect, url="https://provider.test", model_id="model-x")
    hosted_payload = dialect_stream_payload(hosted, _tiered_request(surface))
    assert "service_tier" not in hosted_payload


@pytest.mark.parametrize(
    "dialect",
    ("anthropic_messages", "gemini_generate_content", "bedrock_converse_stream"),
)
def test_service_tier_declines_dialects_without_a_wire_field(dialect: str) -> None:
    """A rung that would drop the tier silently rejects it as a capability."""
    profile = GatewayWireProfile(
        dialect=dialect,
        url="https://provider.test",
        model_id="model-x",
    )
    request = _tiered_request(GatewayApiSurface.CHAT_COMPLETIONS)

    with pytest.raises(ProviderCapabilityError) as excinfo:
        dialect_stream_payload(profile, request)

    assert excinfo.value.capability == "service_tier"
    # The same rungs serve as soon as the tier is gone.
    untiered = request.model_copy(update={"service_tier": None})
    assert dialect_stream_payload(profile, untiered)


def test_service_tier_route_shaping_forwards_on_byok_and_discloses_elsewhere() -> None:
    """A route with no BYOK OpenAI rung strips the tier with disclosure.

    Route shaping keeps the certified waterfall intact: a mixed route with
    one eligible rung keeps the tier (the eligible rung emits it, the
    house-funded rung serves untiered), and a route with no eligible rung
    drops the tier up front so every rung serves and the drop is disclosed
    through the same 'service_tier' entry the coercion path uses.
    """
    request = _tiered_request(GatewayApiSurface.CHAT_COMPLETIONS)
    byok = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://byok.test",
        billing_customer_managed=True,
    )
    hosted = GatewayWireProfile(dialect="openai_compatible", url="https://house.test")
    public, provider = route_generation_parameter_requests((byok, hosted), request)
    assert "service_tier" not in public.ignored_parameters
    assert provider.service_tier == "flex"

    hosted_public, hosted_provider = route_generation_parameter_requests((hosted,), request)
    assert "service_tier" in hosted_public.ignored_parameters
    assert hosted_provider.service_tier is None

    # A BYOK rung on a non-OpenAI wire cannot carry the OpenAI tier either.
    anthropic_byok = GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://anthropic.test",
        billing_customer_managed=True,
    )
    foreign_public, foreign_provider = route_generation_parameter_requests(
        (anthropic_byok,), request
    )
    assert "service_tier" in foreign_public.ignored_parameters
    assert foreign_provider.service_tier is None


def test_service_tier_forwarding_is_per_card_not_lane_level() -> None:
    """A mixed route emits the tier only on the candidate that can BILL it.

    Regression for the underbill window: the route admits because ONE host rung
    carries a flex card, but a fallback host rung carded only for `priority`
    must NOT forward the flex tier (its `service_tier_pricing_enabled` is True).
    If it did and were selected, the provider would run flex (discounted) while
    the gateway billed the base rate. Route shaping keeps the tier alive for the
    flex-carded rung; the priority-only rung strips it at its own payload build,
    so forward and bill agree on whichever candidate is selected.
    """
    request = _tiered_request(GatewayApiSurface.CHAT_COMPLETIONS)
    flex_carded = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://flex.test",
        model_id="model-x",
        service_tier_pricing_enabled=True,
        service_tier_cards=frozenset({"flex"}),
    )
    priority_only = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://priority.test",
        model_id="model-x",
        service_tier_pricing_enabled=True,
        service_tier_cards=frozenset({"priority"}),
    )
    public, provider = route_generation_parameter_requests((flex_carded, priority_only), request)
    # The tier survives route shaping (the flex-carded rung can bill it).
    assert "service_tier" not in public.ignored_parameters
    assert provider.service_tier == "flex"
    # The selected candidate decides emission: flex-carded emits, priority-only
    # strips (no flex card -> no underbill if it is the one that serves).
    assert dialect_stream_payload(flex_carded, provider)["service_tier"] == "flex"
    assert "service_tier" not in dialect_stream_payload(priority_only, provider)


def test_service_tier_scale_strips_on_host_lane_without_rejecting() -> None:
    """`scale` (a valid tier we do not price as opt-in) strips on a host lane.

    A flex-carded host rung carries a card for flex only; a `scale` request is
    not rejected and not forwarded — it is stripped with disclosure and the
    provider runs its default at the base rate (billing-safe).
    """
    scale_request = _tiered_request(GatewayApiSurface.CHAT_COMPLETIONS).model_copy(
        update={"service_tier": "scale"}
    )
    flex_carded = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://flex.test",
        model_id="model-x",
        service_tier_pricing_enabled=True,
        service_tier_cards=frozenset({"flex"}),
    )
    public, provider = route_generation_parameter_requests((flex_carded,), scale_request)
    assert "service_tier" in public.ignored_parameters
    assert provider.service_tier is None
    assert "service_tier" not in dialect_stream_payload(flex_carded, provider)


def test_narrowing_surfaces_the_first_rung_rejection_not_the_route_shape() -> None:
    """When no rung serves, the caller sees the first rung's own field-scoped reason.

    The trigger pairs a ``disabled`` config (the adaptive-only Anthropic rung
    rejects it by name) with replayed thinking blocks (the fallback rung
    rejects those), so each ordering surfaces its own first rung's error.
    """
    anthropic = GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://anthropic.test",
        model_id="claude-opus-5",
        supports_reasoning=True,
        reasoning_wire_format="anthropic_adaptive",
    )
    fallback = GatewayWireProfile(dialect="openai_compatible", url="https://fallback.test")
    request = _thinking_history_request().model_copy(
        update={"provider_thinking_config": {"type": "disabled"}}
    )

    with pytest.raises(ProviderParameterError) as raised:
        compatible_generation_parameter_profile_indexes((anthropic, fallback), request)
    assert raised.value.param == "thinking.type"

    with pytest.raises(ProviderParameterError) as reordered:
        compatible_generation_parameter_profile_indexes((fallback, anthropic), request)
    assert reordered.value.param == "thinking"


def _openai_reasoning_profile(
    efforts: tuple[ReasoningEffort, ...] = ("none", "low", "medium", "high"),
) -> GatewayWireProfile:
    """Return one OpenAI Responses profile with an explicit effort ladder."""
    return GatewayWireProfile(
        dialect="openai_responses",
        url="https://api.openai.com/v1/responses",
        model_id="gpt-5.6-sol",
        supports_reasoning=True,
        reasoning_wire_format="openai_responses",
        supported_reasoning_efforts=efforts,
    )


def _messages_request(
    *,
    thinking: JsonObject | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    maximum_output_tokens: int | None = None,
) -> GatewayRequest:
    """Build one minimal Messages-surface request for translation fixtures."""
    return GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="hi"),),
        provider_thinking_config=thinking,
        reasoning_effort=reasoning_effort,
        maximum_output_tokens=maximum_output_tokens,
        stream=True,
    )


def test_replayed_thinking_blocks_still_reject_on_a_non_anthropic_route() -> None:
    """Signed history blocks replay only on the wire that issued them."""
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(
            GatewayMessage(role="user", content="hi"),
            GatewayMessage(
                role="assistant",
                content="prior",
                provider_reasoning=(ThinkingBlock(text="deep", signature="sig-1"),),
            ),
        ),
        stream=True,
    )
    with pytest.raises(ProviderParameterError, match="extended-thinking"):
        route_generation_parameter_requests((_openai_reasoning_profile(),), request)


def _tool_error_request(*, content: str = "exit 1") -> GatewayRequest:
    """Build one failed-tool-turn request (the Claude Code repro shape)."""
    return GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(
            GatewayMessage(role="user", content="run false"),
            GatewayMessage(
                role="assistant",
                tool_calls=(ToolCall(call_id="toolu_01", name="run", arguments={"cmd": "false"}),),
            ),
            GatewayMessage(
                role="tool",
                content=content,
                tool_call_id="toolu_01",
                tool_is_error=True,
            ),
        ),
        tools=(GatewayToolDefinition(name="run", parameters={"type": "object"}),),
        maximum_output_tokens=64,
        stream=True,
    )


def test_a_failed_tool_result_folds_into_text_on_every_non_anthropic_wire() -> None:
    """is_error survives as a text prefix instead of a session-killing 400.

    Claude Code sets the flag on every failed tool call and the block is
    baked into history, so a rejection wedges the session; a silent drop
    would misstate that the invocation failed.
    """
    request = _tool_error_request()
    for profile in (
        _openai_reasoning_profile(),
        GatewayWireProfile(
            dialect="openai_compatible",
            url="https://api.example.test/v1/chat/completions",
            model_id="plain-chat",
        ),
        GatewayWireProfile(
            dialect="gemini_generate_content",
            url="https://generativelanguage.googleapis.com/v1beta/models/g:streamGenerateContent",
            model_id="gemini-3.4-pro",
        ),
        GatewayWireProfile(
            dialect="bedrock_converse_stream",
            url="https://bedrock.example.test/model/m/converse-stream",
            model_id="us.amazon.nova-2-lite-v1:0",
        ),
    ):
        public, provider = route_generation_parameter_requests((profile,), request)
        assert "messages.content.is_error->content" in public.ignored_parameters
        payload = dialect_stream_payload(profile, provider)
        rendered = str(payload)
        assert "[tool error] exit 1" in rendered, profile.dialect


def test_a_failed_tool_result_stays_a_native_flag_on_an_anthropic_route() -> None:
    """The one wire with a native error flag keeps it verbatim, undisclosed."""
    profile = GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://api.anthropic.com/v1/messages",
        model_id="claude-opus-5",
    )
    public, provider = route_generation_parameter_requests((profile,), _tool_error_request())

    assert "messages.content.is_error->content" not in public.ignored_parameters
    payload = dialect_stream_payload(profile, provider)
    rendered = str(payload)
    assert "'is_error': True" in rendered
    assert "[tool error]" not in rendered


def test_the_tool_error_fold_never_accumulates_across_replays() -> None:
    """The fold derives from the canonical flag, never from prior folded text.

    A caller replays its ORIGINAL content each turn (it never sees the
    provider payload), so a two-build round of the same request must emit
    exactly one prefix, and caller text that happens to start with the
    marker is not treated as a prior fold.
    """
    profile = _openai_reasoning_profile()
    request = _tool_error_request()
    _public_one, provider = route_generation_parameter_requests((profile,), request)
    first = dialect_stream_payload(profile, provider)
    second = dialect_stream_payload(profile, provider)
    assert first == second
    rendered = str(first)
    assert rendered.count("[tool error]") == 1

    # Caller-authored text carrying the literal marker is content, not a fold.
    marked = _tool_error_request(content="[tool error] exit 1")
    _public_two, provider = route_generation_parameter_requests((profile,), marked)
    rendered = str(dialect_stream_payload(profile, provider))
    assert rendered.count("[tool error] [tool error] exit 1") == 1


def test_a_messages_probe_below_the_openai_floor_rides_the_floor() -> None:
    """Claude Code's max_tokens:1 probe serves instead of a provider 400."""
    profile = _openai_reasoning_profile()
    public, provider = route_generation_parameter_requests(
        (profile,), _messages_request(maximum_output_tokens=1)
    )

    assert provider.maximum_output_tokens == 16
    assert "max_tokens->16" in public.ignored_parameters
    payload = dialect_stream_payload(profile, provider)
    assert payload["max_output_tokens"] == 16


def test_the_openai_floor_leaves_anthropic_routes_and_other_surfaces_alone() -> None:
    """The clamp is a Messages-surface translation, not a global rewrite."""
    anthropic = GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://api.anthropic.com/v1/messages",
        model_id="claude-opus-5",
    )
    _public, provider = route_generation_parameter_requests(
        (anthropic,), _messages_request(maximum_output_tokens=1)
    )
    assert provider.maximum_output_tokens == 1

    chat_surface = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hi"),),
        maximum_output_tokens=1,
        stream=True,
    )
    public, provider = route_generation_parameter_requests(
        (
            GatewayWireProfile(
                dialect="openai_compatible",
                url="https://api.example.test/v1/chat/completions",
                model_id="plain-chat",
            ),
        ),
        chat_surface,
    )
    assert provider.maximum_output_tokens == 1
    assert "max_tokens->16" not in public.ignored_parameters

    # A rung whose declared output ceiling sits below the provider floor
    # cannot ride it: the gateway never dispatches above a rung's declared
    # capability just to satisfy the floor, and a homogeneous native
    # Responses route then rejects the sub-minimum value by name at
    # admission instead of dispatching a request the provider will 400
    # opaquely post-commit.
    capped = GatewayWireProfile(
        dialect="openai_responses",
        url="https://api.openai.com/v1/responses",
        model_id="tiny-cap",
        maximum_output_tokens=8,
    )
    with pytest.raises(ProviderParameterError) as rejected:
        route_generation_parameter_requests((capped,), _messages_request(maximum_output_tokens=1))
    assert "at least 16" in str(rejected.value)


def _exposed_reasoning_request(
    surface: GatewayApiSurface = GatewayApiSurface.CHAT_COMPLETIONS,
) -> GatewayRequest:
    """A Terminus-shaped loop: plaintext reasoning on a prior non-tool assistant turn."""
    return GatewayRequest(
        surface=surface,
        messages=(
            GatewayMessage(role="user", content="run ls"),
            GatewayMessage(
                role="assistant",
                content='{"command": "ls"}',
                provider_reasoning=(
                    ExposedReasoningContentBlock(content="The user wants a directory listing."),
                ),
            ),
            GatewayMessage(role="user", content="a.txt b.txt"),
        ),
        stream=True,
    )


def _exposed_profile(
    *, exposed: bool, url: str = "https://tokenhub-intl.tencentcloudmaas.com/v1"
) -> GatewayWireProfile:
    return GatewayWireProfile(
        dialect="openai_compatible",
        url=url,
        model_id="hy4-preview",
        supports_reasoning=True,
        reasoning_wire_format="reasoning",
        reasoning_output_exposed=exposed,
    )


def test_plaintext_reasoning_replays_only_on_an_exposing_rung() -> None:
    """The exposing rung forwards the caller's plaintext verbatim; others omit it."""
    request = _exposed_reasoning_request()
    forwarded = openai_compatible_stream_payload(
        "hy4-preview", request, reasoning_output_exposed=True
    )
    messages = forwarded["messages"]
    assert isinstance(messages, list)
    assert messages[1]["reasoning_content"] == "The user wants a directory listing."
    omitted = openai_compatible_stream_payload("hy4-preview", request)
    omitted_messages = omitted["messages"]
    assert isinstance(omitted_messages, list)
    assert "reasoning_content" not in omitted_messages[1]
    # Other wires drop the block instead of raising; narrowing disclosed it.
    anthropic = anthropic_messages_stream_payload("claude-haiku-4-5", request)
    anthropic_messages = anthropic["messages"]
    assert isinstance(anthropic_messages, list)
    assert anthropic_messages[1]["content"] == [{"type": "text", "text": '{"command": "ls"}'}]


def test_plaintext_reasoning_route_gate_discloses_or_forwards() -> None:
    """No exposing rung -> disclosed drop; mixed -> disclosed drop; all exposing -> silent.

    Plaintext reasoning is baked into the transcript (an earlier exposed-rung
    turn or a client re-serialization), so a route that cannot replay it
    degrades instead of rejecting — previously a 400 that killed every
    session the moment it switched from an exposed model to any other.
    """
    request = _exposed_reasoning_request()
    public, provider = route_generation_parameter_requests(
        (_exposed_profile(exposed=False),), request
    )
    assert (
        "messages.reasoning_content->dropped(unsupported_by_provider)" in public.ignored_parameters
    )
    payload = openai_compatible_stream_payload("hy4-preview", provider)
    payload_messages = cast("list[JsonObject]", payload["messages"])
    assert all("reasoning_content" not in message for message in payload_messages)

    public, _provider = route_generation_parameter_requests(
        (
            _exposed_profile(exposed=True),
            _exposed_profile(exposed=False, url="https://openrouter.test/v1"),
        ),
        request,
    )
    assert (
        "messages.reasoning_content->dropped(unsupported_by_provider)" in public.ignored_parameters
    )

    public, _provider = route_generation_parameter_requests(
        (_exposed_profile(exposed=True),), request
    )
    assert public.ignored_parameters == ()


def test_plaintext_reasoning_prefers_the_exposing_rung_on_a_mixed_waterfall() -> None:
    """A rung that carries the reasoning is exact; a rung that would drop it is a fallback."""
    request = _exposed_reasoning_request()
    profiles = (
        _exposed_profile(exposed=False, url="https://openrouter.test/v1"),
        _exposed_profile(exposed=True),
    )
    assert compatible_generation_parameter_profile_indexes(profiles, request) == (1,)
    # Without plaintext history both rungs are exact and the order is preserved.
    plain = request.model_copy(update={"messages": (GatewayMessage(role="user", content="hi"),)})
    assert compatible_generation_parameter_profile_indexes(profiles, plain) == (0, 1)


def test_hosted_tool_echoes_require_a_homogeneous_responses_route_and_reemit_verbatim() -> None:
    """Echoed hosted-tool items (web_search_call, mcp_call, their outputs)
    forward byte-for-byte on native Responses rungs at their exact position;
    any other rung in the route is a named rejection, mirroring the Anthropic
    server-tool rule: silently dropping provider-executed history would wedge
    the session."""
    hosted_item: JsonObject = {
        "type": "web_search_call",
        "id": "ws_1",
        "status": "completed",
        "action": {"type": "search", "query": "current stable Python"},
    }
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(
            GatewayMessage(role="user", content="go"),
            GatewayMessage(role="assistant", provider_native_item=hosted_item),
            GatewayMessage(role="user", content="and now?"),
        ),
        stream=True,
        include_usage=True,
    )
    payload = openai_responses_stream_payload("gpt-5.6-sol", request, supports_temperature=False)
    assert payload["input"] == [
        {"role": "user", "content": "go"},
        hosted_item,
        {"role": "user", "content": "and now?"},
    ]

    responses = GatewayWireProfile(dialect="openai_responses", url="https://openai.test")
    chat = GatewayWireProfile(dialect="openai_compatible", url="https://chat.test")
    public, _provider = route_generation_parameter_requests((responses,), request)
    assert public.ignored_parameters == ()
    with pytest.raises(ProviderParameterError) as mixed:
        route_generation_parameter_requests((responses, chat), request)
    assert mixed.value.param == "input"
    assert "hosted tool items" in str(mixed.value)


def test_route_rejects_max_output_tokens_below_the_responses_minimum() -> None:
    """A sub-16 output ceiling on a homogeneous native Responses route fails
    at admission with the documented minimum named, instead of dispatching a
    request the provider will 400 opaquely post-commit."""
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="go"),),
        maximum_output_tokens=5,
        maximum_output_tokens_parameter="max_output_tokens",
        stream=True,
        include_usage=True,
    )
    responses = GatewayWireProfile(dialect="openai_responses", url="https://openai.test")
    with pytest.raises(ProviderParameterError) as rejected:
        route_generation_parameter_requests((responses,), request)
    assert rejected.value.param == "max_output_tokens"
    assert "at least 16" in str(rejected.value)

    # The documented minimum itself dispatches.
    at_minimum = request.model_copy(update={"maximum_output_tokens": 16})
    public, _provider = route_generation_parameter_requests((responses,), at_minimum)
    assert public.ignored_parameters == ()

    # A mixed route keeps dispatching: the bound is OpenAI's, not the route's.
    anthropic = GatewayWireProfile(dialect="anthropic_messages", url="https://a.test")
    public, _provider = route_generation_parameter_requests((responses, anthropic), request)
    assert public.ignored_parameters == ()


def test_an_all_responses_route_keeps_tool_result_images() -> None:
    """The native Responses wire carries tool-result images itself (the SDK
    function_call_output part list), so a homogeneous Responses route keeps
    the screenshot verbatim like an all-Anthropic route does."""
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="go"), _tool_image_message()),
    )
    profiles = (GatewayWireProfile(dialect="openai_responses", url="https://r.test"),)

    public_request, provider_request = route_generation_parameter_requests(profiles, request)

    assert provider_request.messages[-1].images
    assert TOOL_RESULT_IMAGE_DROP_DISCLOSURE not in public_request.ignored_parameters


def test_a_responses_and_chat_route_still_degrades_tool_result_images() -> None:
    """A chat fallback rung has no tool-image carrier, so the mixed route
    keeps the disclosed placeholder degrade."""
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="go"), _tool_image_message()),
    )
    profiles = (
        GatewayWireProfile(dialect="openai_responses", url="https://r.test"),
        GatewayWireProfile(dialect="openai_compatible", url="https://c.test"),
    )

    public_request, provider_request = route_generation_parameter_requests(profiles, request)

    assert provider_request.messages[-1].content_parts == ()
    assert TOOL_RESULT_IMAGE_DROP_DISCLOSURE in public_request.ignored_parameters


def _attributed_history_request() -> GatewayRequest:
    """One replayed turn whose call, and result, carry Responses attribution."""
    return GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(
            GatewayMessage(role="user", content="go"),
            GatewayMessage(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        call_id="call-1",
                        name="spawn_agent",
                        arguments={},
                        provider_namespace="collaboration",
                        provider_caller={"type": "program", "caller_id": "call_prog"},
                    ),
                ),
            ),
            GatewayMessage(
                role="tool",
                tool_call_id="call-1",
                content="done",
                provider_tool_name="spawn_agent",
                provider_tool_namespace="collaboration",
            ),
        ),
    )


def test_tool_call_attribution_drops_with_disclosure_off_the_responses_wire() -> None:
    """A chat rung rebuilds the call without its namespace/caller attribution;
    the call still executes with its exact name and arguments, so the route
    discloses the per-field drop instead of rejecting a history the caller
    cannot rewrite."""
    profiles = (GatewayWireProfile(dialect="openai_compatible", url="https://c.test"),)

    public_request, _provider = route_generation_parameter_requests(
        profiles, _attributed_history_request()
    )

    assert (
        "messages.tool_calls.namespace->dropped(unsupported_by_provider)"
        in public_request.ignored_parameters
    )
    assert (
        "messages.tool_calls.caller->dropped(unsupported_by_provider)"
        in public_request.ignored_parameters
    )
    assert (
        "messages.tool_results.attribution->dropped(unsupported_by_provider)"
        in public_request.ignored_parameters
    )


def test_tool_call_attribution_is_undisclosed_on_a_homogeneous_responses_route() -> None:
    """The native wire re-emits the attribution verbatim: nothing is dropped."""
    profiles = (GatewayWireProfile(dialect="openai_responses", url="https://r.test"),)

    public_request, _provider = route_generation_parameter_requests(
        profiles, _attributed_history_request()
    )

    assert not any("attribution" in path for path in public_request.ignored_parameters)
    assert not any("tool_calls.namespace" in path for path in public_request.ignored_parameters)
    assert not any("tool_calls.caller" in path for path in public_request.ignored_parameters)


def test_a_messages_surface_effort_rejection_matches_the_client_recovery_latch() -> None:
    """The 400 names output_config.effort and phrases the miss recoverably.

    Claude Code sends output_config.effort on every request and auto-recovers
    (drops the field and retries) only when the message contains "effort
    parameter" and "not support" or the param names output_config.effort; any
    other phrasing wedges every turn on an effort-constrained route (#795).
    """
    profiles = (
        GatewayWireProfile(
            dialect="openai_compatible",
            url="https://c.test",
            model_id="hermes-4-405b",
            supports_reasoning=True,
            reasoning_wire_format="reasoning_effort",
            supported_reasoning_efforts=("medium",),
        ),
    )
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="hi"),),
        reasoning_effort="high",
    )

    with pytest.raises(UnsupportedReasoningEffortError) as raised:
        route_generation_parameter_requests(profiles, request)

    assert raised.value.param == "output_config.effort"
    assert "effort parameter" in str(raised.value)
    assert "not supported" in str(raised.value)


def test_route_refuses_a_whole_empty_user_turn_before_an_anthropic_dispatch() -> None:
    """The Anthropic wire rejects empty text content blocks post-dispatch
    ("text content blocks must be non-empty"; 2026-09-05, six orgs on
    claude-fable routes); a user turn that is entirely empty has nothing to
    send and dropping the whole message would change conversation structure,
    so it is refused by name pre-dispatch."""
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(
            GatewayMessage(role="user", content="hello"),
            GatewayMessage(role="assistant", content="hi"),
            GatewayMessage(role="user", content=""),
        ),
        maximum_output_tokens=256,
        stream=True,
        include_usage=True,
    )
    anthropic = GatewayWireProfile(dialect="anthropic_messages", url="https://a.test")
    with pytest.raises(ProviderParameterError) as rejected:
        route_generation_parameter_requests((anthropic,), request)
    assert rejected.value.param == "messages"
    assert "empty or whitespace-only content" in str(rejected.value)

    # A whitespace-only user turn is the same refusal: the wire rejects it
    # ("text content blocks must contain non-whitespace text", live
    # 2026-09-05) and a user message cannot dispatch as an empty array.
    whitespace = request.model_copy(
        update={"messages": (*request.messages[:2], GatewayMessage(role="user", content="\n "))}
    )
    with pytest.raises(ProviderParameterError) as rejected_whitespace:
        route_generation_parameter_requests((anthropic,), whitespace)
    assert rejected_whitespace.value.param == "messages"

    # An all-empty cache-marked block run is the same empty turn (the blocks
    # must flatten to the content, so all-empty blocks imply empty content)
    # and takes the same refusal instead of falling through to the builder:
    # the admission check keys on empty content alone, so cache markers on
    # the empty blocks change nothing. The breakpoint migration cannot save
    # this turn either (there is no retained block to carry the marker), so
    # refusal is the only honest outcome.
    block_runs: tuple[tuple[JsonObject, ...], ...] = (
        ({"type": "text", "text": ""},),
        ({"type": "text", "text": "", "cache_control": {"type": "ephemeral"}},),
        (
            {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": ""},
        ),
    )
    for blocks in block_runs:
        marked = request.model_copy(
            update={
                "messages": (
                    GatewayMessage(role="user", content="hello"),
                    GatewayMessage(role="assistant", content="hi"),
                    GatewayMessage(role="user", content="", provider_text_blocks=blocks),
                )
            }
        )
        with pytest.raises(ProviderParameterError) as rejected:
            route_generation_parameter_requests((anthropic,), marked)
        assert rejected.value.param == "messages"

    # A non-Anthropic route keeps serving the shape it can carry.
    chat = GatewayWireProfile(dialect="openai_compatible", url="https://chat.test")
    public, _provider = route_generation_parameter_requests((chat,), request)
    assert public.ignored_parameters == ()


def test_chat_surface_sub_16_output_ceiling_rides_the_openai_floor() -> None:
    """A Chat-surface max_tokens below OpenAI's minimum translated onto an
    OpenAI rung rides the disclosed 16-token floor (the Messages-surface
    contract), instead of dispatching a value the provider 400s opaquely
    post-commit (2026-09-05 stragglers)."""
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hi"),),
        maximum_output_tokens=1,
        maximum_output_tokens_parameter="max_tokens",
        stream=True,
        include_usage=True,
    )
    responses = GatewayWireProfile(dialect="openai_responses", url="https://openai.test")
    chat_rung = GatewayWireProfile(dialect="openai_compatible", url="https://chat.test")
    # The floor applies on all-OpenAI routes AND mixed OpenAI routes, so no
    # rung dispatches the sub-minimum value.
    for route in ((responses,), (responses, chat_rung)):
        public, provider = route_generation_parameter_requests(route, request)
        assert provider.maximum_output_tokens == 16
        assert "max_tokens->16" in public.ignored_parameters

    # The disclosure names the caller's own parameter.
    completion_request = request.model_copy(
        update={"maximum_output_tokens_parameter": "max_completion_tokens"}
    )
    public, provider = route_generation_parameter_requests((responses,), completion_request)
    assert provider.maximum_output_tokens == 16
    assert "max_completion_tokens->16" in public.ignored_parameters


def test_a_tool_result_name_is_undisclosed_on_openai_wire_routes() -> None:
    """Both OpenAI wires carry the legacy tool-result name; nothing drops."""
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(
            GatewayMessage(role="user", content="go"),
            GatewayMessage(
                role="tool",
                tool_call_id="call-1",
                content="ok",
                provider_tool_name="read_file",
            ),
        ),
    )
    profiles = (
        GatewayWireProfile(dialect="openai_compatible", url="https://c.test"),
        GatewayWireProfile(dialect="openai_responses", url="https://r.test"),
    )

    public_request, _provider = route_generation_parameter_requests(profiles, request)

    assert not any("tool_results" in path for path in public_request.ignored_parameters)


def test_a_tool_result_name_drops_with_disclosure_off_the_openai_wires() -> None:
    """A rung with no name slot (Anthropic tool_result) discloses the drop."""
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(
            GatewayMessage(role="user", content="go"),
            GatewayMessage(
                role="tool",
                tool_call_id="call-1",
                content="ok",
                provider_tool_name="read_file",
            ),
        ),
    )
    profiles = (
        GatewayWireProfile(dialect="openai_compatible", url="https://c.test"),
        GatewayWireProfile(dialect="anthropic_messages", url="https://a.test"),
    )

    public_request, _provider = route_generation_parameter_requests(profiles, request)

    assert (
        "messages.tool_results.name->dropped(unsupported_by_provider)"
        in public_request.ignored_parameters
    )
    # Namespace/caller attribution stays a Responses-only concern.
    assert not any("attribution" in path for path in public_request.ignored_parameters)
