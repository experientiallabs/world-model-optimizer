"""Native streaming payload builders for the OpenAI-family dialects.

Split from ``streaming_requests`` for the module line budget, mirroring the
Messages-family split in ``messages_payloads``: the native Responses and
OpenAI-compatible Chat builders live here; ``dialect_stream_payload`` in
``dialect_dispatch`` remains the single dispatch seam.
"""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject
from exp.common.models import ChatMaxTokensField
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.models.providers.errors import (
    ProviderResponseError,
)
from exp.runtime.models.providers.fireworks import prepare_gateway_reasoning_history
from exp.runtime.models.providers.reasoning_compat import (
    openai_reasoning_effort,
    require_sampling_reasoning_compatibility,
)
from exp.runtime.models.providers.wire_messages import (
    add_openai_tools,
    openai_chat_message,
    responses_items,
)

_INPUT_MESSAGE_ROLES = frozenset({"user", "system", "developer"})


def _without_input_message_status(item: JsonObject) -> JsonObject:
    """Drop ``status`` from a replayed input message; every other item is verbatim."""
    if (
        "status" in item
        and item.get("type") in (None, "message")
        and item.get("role") in _INPUT_MESSAGE_ROLES
    ):
        return {key: value for key, value in item.items() if key != "status"}
    return item


def openai_responses_stream_payload(
    model_id: str,
    request: GatewayRequest,
    *,
    supports_temperature: bool,
    supports_top_p: bool | None = None,
    supports_top_k: bool = False,
    supports_logprobs: bool = False,
    supports_reasoning: bool = False,
    reasoning_effort: str | None = None,
    sampling_requires_reasoning_none: bool = False,
    forwards_service_tier: bool = False,
    forwards_prompt_cache_key: bool = False,
) -> JsonObject:
    """Translate one canonical request to native streaming Responses JSON.

    Args:
        model_id: Exact OpenAI model identifier.
        request: Canonical gateway request.
        supports_temperature: Whether this exact model accepts explicit temperature.
        supports_reasoning: Whether this exact model accepts the reasoning parameter.
        reasoning_effort: Optional catalog-pinned reasoning effort.

    Returns:
        Native Responses request with storage disabled and streaming enabled.

    Raises:
        ProviderResponseError: An instruction message has no text.
    """
    # The Responses API has no stop field. Caller stop sequences never reach
    # this wire; the native data plane emulates them from the wire entry's
    # ``stop_sequences`` (see ``deployment_wire_entry``), cutting the stream
    # at the first match and reporting a stop-sequence terminal.
    instructions: list[str] = []
    items: list[JsonObject] = []
    for message in request.messages:
        if message.provider_native_item is not None:
            # Codex-native input items (tool namespaces, freeform tool
            # history, hosted tool echoes) re-emit byte-for-byte at their
            # position; route admission already required every rung to speak
            # this wire. The one exception is an input MESSAGE carrying the
            # output-only ``status`` a client copied from a prior response:
            # the input-message schema has no such field and OpenAI answers
            # 400 "Unknown parameter: 'input[N].status'". Hosted tool items
            # keep theirs (their schema defines it).
            items.append(_without_input_message_status(message.provider_native_item))
        elif message.role in {"system", "developer"}:
            if message.content is None:
                raise ProviderResponseError("instruction messages require text")
            # Leading instructions ride the instructions field; one arriving
            # after conversation began keeps its position as an input item.
            if items:
                items.append({"role": message.role, "content": message.content})
            else:
                instructions.append(message.content)
        else:
            items.extend(responses_items(message))
    # Upstream storage stays disabled regardless of the caller's `store`
    # selector: continuation state is gateway-owned, the gateway never
    # references a provider-stored response, and disabled storage is what
    # makes the provider return encrypted reasoning content.
    payload: JsonObject = {
        "model": model_id,
        "input": items,
        "store": False,
        "stream": True,
    }
    response_store = request.response_store
    if request.include_encrypted_reasoning or supports_reasoning and response_store is not False:
        payload["include"] = ["reasoning.encrypted_content"]
    if instructions:
        payload["instructions"] = "\n\n".join(instructions)
    add_openai_tools(payload, request, responses=True)
    if request.parallel_tool_calls is not None:
        payload["parallel_tool_calls"] = request.parallel_tool_calls
    if request.client_metadata is not None:
        # Opaque client telemetry, forwarded verbatim (standard Responses
        # surface: accepted live with a plain API key, 2026-08-29).
        payload["client_metadata"] = request.client_metadata
    text_payload: JsonObject = {}
    if request.text_verbosity is not None:
        text_payload["verbosity"] = request.text_verbosity
    if request.structured_text is not None:
        format_payload: JsonObject = {
            "type": "json_schema",
            "name": request.structured_text.name,
            "schema": request.structured_text.json_schema,
            "strict": request.structured_text.strict,
        }
        if request.structured_text.description is not None:
            format_payload["description"] = request.structured_text.description
        text_payload["format"] = format_payload
    elif request.json_object_output:
        text_payload["format"] = {"type": "json_object"}
    if text_payload:
        payload["text"] = text_payload
    if request.maximum_output_tokens is not None:
        payload["max_output_tokens"] = request.maximum_output_tokens
    effective_reasoning_effort = request.reasoning_effort or reasoning_effort
    require_sampling_reasoning_compatibility(
        reasoning_effort=effective_reasoning_effort,
        sampling_requires_reasoning_none=sampling_requires_reasoning_none,
        temperature_requested=request.temperature is not None,
        top_p_requested=request.top_p is not None,
    )
    if request.temperature is not None and supports_temperature:
        payload["temperature"] = request.temperature
    top_p_supported = supports_temperature if supports_top_p is None else supports_top_p
    if request.top_p is not None and top_p_supported:
        payload["top_p"] = request.top_p
    if request.service_tier is not None and forwards_service_tier:
        # BYOK-only: the caller pays this provider directly, so their tier
        # selection (and its pricing) is between them and the provider.
        payload["service_tier"] = request.service_tier
    if request.provider_prompt_cache_key is not None and forwards_prompt_cache_key:
        # Tenant-namespaced cache-affinity key (derived at admission): it
        # changes cache-hit cost, never semantics, so rungs that do not route
        # by it omit it structurally with no decline and no disclosure.
        payload["prompt_cache_key"] = request.provider_prompt_cache_key
    # Native OpenAI Responses has no top-k request field. Never trust a
    # mistaken route declaration to send this extension to the API.
    del supports_top_k
    # Responses output normalization has no probability representation. Keep
    # the shared capability argument, but ignore logprob controls before send.
    del supports_logprobs
    reasoning: JsonObject = {}
    if supports_reasoning and effective_reasoning_effort is not None:
        reasoning["effort"] = openai_reasoning_effort(model_id, effective_reasoning_effort)
    if supports_reasoning and request.reasoning_summary is not None:
        reasoning["summary"] = request.reasoning_summary
    if supports_reasoning and request.reasoning_context is not None:
        # Forwarded verbatim: the value controls provider-side re-rendering
        # of prior turns' reasoning and has no gateway semantics.
        reasoning["context"] = request.reasoning_context
    if reasoning:
        payload["reasoning"] = reasoning
    return payload


def openai_compatible_stream_payload(
    model_id: str,
    request: GatewayRequest,
    *,
    token_limit_key: ChatMaxTokensField = "max_tokens",
    supports_temperature: bool = True,
    supports_top_p: bool | None = None,
    supports_top_k: bool = False,
    supports_frequency_penalty: bool = False,
    supports_presence_penalty: bool = False,
    supports_logprobs: bool = False,
    supports_reasoning: bool = False,
    reasoning_wire_format: str = "reasoning_effort",
    reasoning_effort: str | None = None,
    sampling_requires_reasoning_none: bool = False,
    fireworks_reasoning_route_sha256: str | None = None,
    hunyuan_reasoning_route_sha256: str | None = None,
    reasoning_output_exposed: bool = False,
    forwards_service_tier: bool = False,
    forwards_prompt_cache_key: bool = False,
) -> JsonObject:
    """Translate one canonical request to streaming Chat Completions JSON.

    Args:
        model_id: Exact provider model identifier.
        request: Canonical gateway request.
        token_limit_key: Wire field carrying the output-token ceiling. Azure OpenAI
            reasoning deployments reject ``max_tokens`` and require
            ``max_completion_tokens``.
        supports_temperature: Whether this exact model accepts explicit sampling controls.
        supports_reasoning: Whether this exact model accepts a reasoning control.
        reasoning_wire_format: Provider field used for normalized reasoning effort.
        reasoning_effort: Optional catalog-pinned reasoning effort.
        reasoning_output_exposed: Whether this rung replays the caller's plaintext
            ``reasoning_content`` history verbatim (exposure-gated Tencent/DeepSeek rung).

    Returns:
        Chat Completions request that always asks the provider for terminal usage.
    """
    # A rung is a preserved-thinking route under exactly one provider scheme;
    # its block must name that route to forward. Fireworks additionally toggles
    # the wire-native `reasoning_history` field, which Hunyuan does not use.
    reasoning_route_sha256 = fireworks_reasoning_route_sha256 or hunyuan_reasoning_route_sha256
    messages, active_reasoning = prepare_gateway_reasoning_history(
        request.messages,
        route_sha256=reasoning_route_sha256,
    )
    payload: JsonObject = {
        "model": model_id,
        "messages": [
            openai_chat_message(
                message,
                reasoning_route_sha256=reasoning_route_sha256,
                reasoning_output_exposed=reasoning_output_exposed,
            )
            for message in messages
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if active_reasoning and fireworks_reasoning_route_sha256 is not None:
        payload["reasoning_history"] = "interleaved"
    add_openai_tools(payload, request, responses=False)
    if request.parallel_tool_calls is not None:
        payload["parallel_tool_calls"] = request.parallel_tool_calls
    if request.structured_text is not None:
        schema: JsonObject = {
            "name": request.structured_text.name,
            "schema": request.structured_text.json_schema,
            "strict": request.structured_text.strict,
        }
        if request.structured_text.description is not None:
            schema["description"] = request.structured_text.description
        payload["response_format"] = {"type": "json_schema", "json_schema": schema}
    elif request.json_object_output:
        payload["response_format"] = {"type": "json_object"}
    if request.maximum_output_tokens is not None:
        payload[token_limit_key] = request.maximum_output_tokens
    effective_reasoning_effort = request.reasoning_effort or reasoning_effort
    require_sampling_reasoning_compatibility(
        reasoning_effort=effective_reasoning_effort,
        sampling_requires_reasoning_none=sampling_requires_reasoning_none,
        temperature_requested=request.temperature is not None,
        top_p_requested=request.top_p is not None,
    )
    if request.temperature is not None and supports_temperature:
        payload["temperature"] = request.temperature
    top_p_supported = supports_temperature if supports_top_p is None else supports_top_p
    if request.top_p is not None and top_p_supported:
        payload["top_p"] = request.top_p
    if request.top_k is not None and supports_top_k:
        payload["top_k"] = request.top_k
    if request.frequency_penalty is not None and supports_frequency_penalty:
        payload["frequency_penalty"] = request.frequency_penalty
    if request.presence_penalty is not None and supports_presence_penalty:
        payload["presence_penalty"] = request.presence_penalty
    # Compatible streaming responses also normalize logprobs to null, so an
    # accepted public control is intentionally ignored until projection exists.
    del supports_logprobs
    if request.stop:
        payload["stop"] = list(request.stop)
    if request.service_tier is not None and forwards_service_tier:
        # BYOK-only, matching the native Responses lane.
        payload["service_tier"] = request.service_tier
    if request.provider_prompt_cache_key is not None and forwards_prompt_cache_key:
        # Cache-affinity key, matching the native Responses lane.
        payload["prompt_cache_key"] = request.provider_prompt_cache_key
    if supports_reasoning and effective_reasoning_effort is not None:
        if reasoning_wire_format == "reasoning":
            payload["reasoning"] = {"effort": effective_reasoning_effort}
        elif reasoning_wire_format == "reasoning_effort":
            payload["reasoning_effort"] = openai_reasoning_effort(
                model_id, effective_reasoning_effort
            )
    return payload
