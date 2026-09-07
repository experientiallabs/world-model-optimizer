"""Replay identity over canonical gateway requests and their excluded carriers."""

from __future__ import annotations

from typing import assert_never

from exp.common.core.artifacts import JsonObject, Sha256, sha256_json
from exp.runtime.gateway.contracts import EncryptedReasoningBlock, GatewayRequest
from exp.runtime.gateway.embeddings_contracts import EmbeddingsRequest, ServingRequest
from exp.runtime.gateway.images_contracts import ImagesRequest


def provider_replay_authority(request: GatewayRequest) -> JsonObject | None:
    """Collect the provider-significant input excluded from serialization.

    The carriers (replayed reasoning, native items, verbatim provider
    configurations) are excluded from model serialization so immutable
    artifacts and pre-existing request digests stay byte-identical, but the
    provider still reads them as input. Replay identity folds this envelope
    into :func:`canonical_request_sha256`, and reservation adds its bytes to
    the conservative input bound so an excluded carrier can never push a
    request across a pricing threshold unreserved.

    Args:
        request: Canonical gateway request as decoded from the public wire.

    Returns:
        The envelope of excluded provider input, or ``None`` when the plain
        serialization already covers everything.
    """
    replay: list[JsonObject] = []
    for message_index, message in enumerate(request.messages):
        authority: JsonObject = {"message_index": message_index}
        if message.provider_item_id is not None:
            authority["provider_item_id"] = message.provider_item_id
            authority["provider_output_index"] = message.provider_output_index
            authority["provider_status"] = message.provider_status
            authority["provider_phase"] = message.provider_phase
        if message.provider_native_item is not None:
            authority["provider_native_item"] = message.provider_native_item
        if message.provider_anthropic_block is not None:
            authority["provider_anthropic_block"] = message.provider_anthropic_block
        if message.provider_reasoning:
            blocks: list[JsonObject] = []
            for block in message.provider_reasoning:
                serialized = block.model_dump(mode="json")
                if isinstance(block, EncryptedReasoningBlock):
                    serialized["output_index"] = block.output_index
                    serialized["status"] = block.status
                blocks.append(serialized)
            authority["provider_reasoning"] = blocks
        retained_calls: list[JsonObject] = []
        for tool_call_index, call in enumerate(message.tool_calls):
            if (
                call.raw_arguments is None
                and call.provider_item_id is None
                and call.provider_output_index is None
                and call.provider_status is None
                and call.provider_namespace is None
                and call.provider_caller is None
            ):
                continue
            retained_call: JsonObject = {
                "tool_call_index": tool_call_index,
                "call_id": call.call_id,
                "name": call.name,
                "raw_arguments": call.raw_arguments,
                "provider_item_id": call.provider_item_id,
                "provider_output_index": call.provider_output_index,
                "provider_status": call.provider_status,
            }
            # Joins only when present so every namespace-free request keeps
            # its exact pre-existing digest.
            if call.provider_namespace is not None:
                retained_call["provider_namespace"] = call.provider_namespace
            if call.provider_caller is not None:
                retained_call["provider_caller"] = call.provider_caller
            retained_calls.append(retained_call)
        if retained_calls:
            authority["tool_calls"] = retained_calls
        if message.tool_is_error:
            authority["tool_is_error"] = True
        if message.provider_tool_name is not None:
            authority["provider_tool_name"] = message.provider_tool_name
        if message.provider_tool_namespace is not None:
            authority["provider_tool_namespace"] = message.provider_tool_namespace
        if message.provider_tool_caller is not None:
            authority["provider_tool_caller"] = message.provider_tool_caller
        if len(authority) > 1:
            replay.append(authority)
    retained_tools: list[JsonObject] = []
    for tool_index, tool in enumerate(request.tools):
        if not tool.has_anthropic_tool_carriers():
            continue
        retained_tool: JsonObject = {"tool_index": tool_index, "name": tool.name}
        if tool.eager_input_streaming is not None:
            retained_tool["eager_input_streaming"] = tool.eager_input_streaming
        if tool.defer_loading is not None:
            retained_tool["defer_loading"] = tool.defer_loading
        if tool.allowed_callers is not None:
            retained_tool["allowed_callers"] = list(tool.allowed_callers)
        if tool.input_examples is not None:
            retained_tool["input_examples"] = list(tool.input_examples)
        retained_tools.append(retained_tool)
    if (
        not replay
        and not retained_tools
        and request.provider_thinking_config is None
        and request.reasoning_context is None
        and request.context_management is None
        and request.provider_output_config is None
        and request.diagnostics is None
        and request.speed is None
        and request.inference_geo is None
        and request.service_tier is None
        and not request.json_object_output
        and not request.provider_beta_tokens
        and not request.provider_server_tools
        and not request.provider_native_tools
    ):
        return None
    envelope: JsonObject = {
        "provider_replay": replay,
        "provider_thinking_config": request.provider_thinking_config,
        "reasoning_context": request.reasoning_context,
        "context_management": request.context_management,
        "provider_output_config": request.provider_output_config,
    }
    # The newer Messages carriers join the envelope only when present, so
    # every request decoded before they existed keeps its exact digest.
    if request.diagnostics is not None:
        envelope["diagnostics"] = request.diagnostics
    if request.speed is not None:
        envelope["speed"] = request.speed
    if request.inference_geo is not None:
        envelope["inference_geo"] = request.inference_geo
    if request.service_tier is not None:
        # A provider tier changes pricing and scheduling for the same body.
        envelope["service_tier"] = request.service_tier
    if request.json_object_output:
        # Schema-free JSON mode changes the answer shape for the same body.
        envelope["json_object_output"] = True
    if retained_tools:
        envelope["tools"] = retained_tools
    if request.provider_beta_tokens:
        envelope["provider_beta_tokens"] = list(request.provider_beta_tokens)
    if request.provider_server_tools:
        envelope["provider_server_tools"] = list(request.provider_server_tools)
    if request.provider_native_tools:
        envelope["provider_native_tools"] = [
            {"index": entry.index, "tool": entry.tool} for entry in request.provider_native_tools
        ]
    return envelope


def canonical_request_sha256(request: ServingRequest) -> Sha256:
    """Digest one canonical request including excluded provider replay authority.

    A caller operation key reused with different replayed reasoning must be
    a rejected conflict, never a silent replay of the earlier response. A
    request with no carrier digests exactly as its plain serialization, so
    every request decoded before the carriers existed keeps its identity.

    The embeddings and images surfaces have no messages, tools, or provider
    carriers, so they digest exactly as their plain serialization.

    Args:
        request: Canonical serving request as decoded from the public wire.

    Returns:
        The stable canonical request digest.
    """
    match request:
        case EmbeddingsRequest() | ImagesRequest():
            return sha256_json(request)
        case GatewayRequest():
            envelope = provider_replay_authority(request)
            if envelope is None:
                return sha256_json(request)
            return sha256_json({"request_sha256": sha256_json(request), **envelope})
        case _:  # pragma: no cover - exhaustive over the ServingRequest union.
            assert_never(request)
