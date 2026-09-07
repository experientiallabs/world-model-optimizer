"""Native Bedrock Converse payload construction shared by both engines.

The blocking provider client (`bedrock.py`) and the gateway dialect builders
(`streaming_requests.py`) build the identical Converse request from this one
module, so the two callers cannot drift at the Bedrock wire boundary. This
module stays free of streaming imports on purpose: the shared dialect
builders import it without creating a cycle through the streaming stack.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from typing import cast

from exp.common.core.artifacts import JsonObject
from exp.common.models import ModelMessage, ModelRequest, ToolChoice
from exp.runtime.models.providers.audios import reject_audio_part
from exp.runtime.models.providers.documents import bedrock_document_block
from exp.runtime.models.providers.errors import ProviderParameterError
from exp.runtime.models.providers.images import bedrock_image_block
from exp.runtime.models.providers.videos import bedrock_video_block

BEDROCK_MAXIMUM_INLINE_MEDIA_BYTES = 25_000_000
"""Converse accepts inline media only while the whole request payload stays
under 25 MB. Per-file gateway ceilings do not bound the sum, so a request that
carries inline media is measured as the complete serialized body: media, text,
system prompts, tools, and output configuration together."""


def converse_request(
    model_id: str,
    request: ModelRequest,
    *,
    supports_temperature: bool = True,
    supports_top_p: bool = True,
    supports_top_k: bool = False,
    supports_logprobs: bool = False,
    stop_sequences: Sequence[str] = (),
    structured_output_name: str | None = None,
    structured_output_description: str | None = None,
    structured_output_schema: JsonObject | None = None,
    strict_tool_names: Collection[str] = (),
    json_object_instruction: str | None = None,
) -> JsonObject:
    """Translate one EXP request into boto Converse keyword arguments.

    Args:
        model_id: Exact foundation-model or inference-profile ID sent as the
            boto ``modelId`` routing key.
        request: Typed EXP request.
        supports_temperature: Whether this exact deployment accepts temperature.
        supports_top_p: Whether this exact deployment accepts top-p sampling.
        supports_top_k: Whether this exact deployment accepts top-k sampling.
        supports_logprobs: Reserved response-projection capability flag.
        stop_sequences: Exact stop strings admitted for the selected route.
        structured_output_name: Optional name for a strict JSON output contract.
        structured_output_description: Optional description for that output contract.
        structured_output_schema: Strict JSON schema admitted for structured output.
        strict_tool_names: Tool definitions whose schemas Bedrock must enforce.
        json_object_instruction: Optional trailing system text requesting
            schema-free JSON output (Converse has no native JSON mode).

    Returns:
        Keyword arguments accepted by ``bedrock-runtime`` Converse.

    Raises:
        ValueError: A message cannot be represented without dropping tool context.
    """
    return {
        "modelId": model_id,
        **converse_body(
            request,
            supports_temperature=supports_temperature,
            supports_top_p=supports_top_p,
            supports_top_k=supports_top_k,
            supports_logprobs=supports_logprobs,
            stop_sequences=stop_sequences,
            structured_output_name=structured_output_name,
            structured_output_description=structured_output_description,
            structured_output_schema=structured_output_schema,
            strict_tool_names=strict_tool_names,
            json_object_instruction=json_object_instruction,
        ),
    }


def converse_body(
    request: ModelRequest,
    *,
    supports_temperature: bool = True,
    supports_top_p: bool = True,
    supports_top_k: bool = False,
    supports_logprobs: bool = False,
    stop_sequences: Sequence[str] = (),
    structured_output_name: str | None = None,
    structured_output_description: str | None = None,
    structured_output_schema: JsonObject | None = None,
    strict_tool_names: Collection[str] = (),
    json_object_instruction: str | None = None,
) -> JsonObject:
    """Translate one EXP request into the Converse wire document.

    The body carries no routing key: boto callers splice ``modelId`` beside it
    and the ConverseStream REST route carries the model in the URL path.

    Args:
        request: Typed EXP request.
        supports_temperature: Whether this exact deployment accepts temperature.
        supports_top_p: Whether this exact deployment accepts top-p sampling.
        supports_top_k: Whether this exact deployment accepts top-k sampling.
        supports_logprobs: Reserved response-projection capability flag.
        stop_sequences: Exact stop strings admitted for the selected route.
        structured_output_name: Optional name for a strict JSON output contract.
        structured_output_description: Optional description for that output contract.
        structured_output_schema: Strict JSON schema admitted for structured output.
        strict_tool_names: Tool definitions whose schemas Bedrock must enforce.
        json_object_instruction: Optional trailing system text requesting
            schema-free JSON output (Converse has no native JSON mode).

    Returns:
        The native Converse request document.

    Raises:
        ValueError: A message cannot be represented without dropping tool context.
        ProviderParameterError: The request carries inline media and its
            complete serialized body exceeds the Converse payload ceiling.
    """
    # Converse has no provider-neutral logprobs field. Keep the flag in the
    # shared signature so all provider lanes use one capability contract, but
    # omit the request until response projection exists.
    del supports_logprobs
    system: list[JsonObject] = []
    messages: list[JsonObject] = []

    def push(role: str, content: list[JsonObject]) -> None:
        """Append or merge one Converse message while preserving adjacent same-role blocks."""
        if messages and messages[-1]["role"] == role:
            existing = cast("list[JsonObject]", messages[-1]["content"])
            existing.extend(content)
            return
        messages.append({"role": role, "content": content})

    for message in request.messages:
        if message.role == "system":
            if message.content is None:
                raise ValueError("system messages need text content")
            system.append({"text": message.content})
            continue
        if message.role == "tool":
            push(
                "user",
                [
                    {
                        "toolResult": {
                            "toolUseId": message.tool_call_id or "",
                            "content": [{"text": message.content or ""}],
                        }
                    }
                ],
            )
            continue
        push(
            "assistant" if message.role == "assistant" else "user",
            _message_blocks(message),
        )

    if json_object_instruction is not None:
        system.append({"text": json_object_instruction})
    payload: JsonObject = {"messages": messages}
    inference = _inference_config(
        request,
        supports_temperature=supports_temperature,
        supports_top_p=supports_top_p,
        stop_sequences=stop_sequences,
    )
    if inference:
        payload["inferenceConfig"] = inference
    if request.top_k is not None and supports_top_k:
        # Converse exposes model-specific controls at the request root, not
        # inside inferenceConfig. The route flag is explicit because support
        # varies by foundation model.
        payload["additionalModelRequestFields"] = {"top_k": request.top_k}
    if system:
        payload["system"] = system
    if structured_output_schema is not None:
        json_schema: JsonObject = {
            "schema": json.dumps(
                structured_output_schema,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        }
        if structured_output_name is not None:
            json_schema["name"] = structured_output_name
        if structured_output_description is not None:
            json_schema["description"] = structured_output_description
        payload["outputConfig"] = {
            "textFormat": {
                "type": "json_schema",
                "structure": {"jsonSchema": json_schema},
            }
        }
    tool_config = _tool_config(request, strict_tool_names=strict_tool_names)
    if tool_config is not None:
        payload["toolConfig"] = tool_config
    _require_inline_media_within_payload(request, payload)
    return payload


def _carries_inline_media(request: ModelRequest) -> bool:
    """Return whether any message carries an inline image, video, or document."""
    return any(
        part.kind != "text" and part.data is not None
        for message in request.messages
        for part in message.content_parts
    )


def _require_inline_media_within_payload(request: ModelRequest, payload: JsonObject) -> None:
    """Reject an inline-media request whose complete body exceeds the Converse payload cap.

    Text-only requests are not measured here: the ceiling is documented for
    inline media, and text limits are enforced by the model's token budget.

    Args:
        request: Typed EXP request whose user messages may carry inline media.
        payload: The fully assembled Converse body about to be dispatched.

    Raises:
        ProviderParameterError: The serialized body exceeds
            ``BEDROCK_MAXIMUM_INLINE_MEDIA_BYTES``.
    """
    if not _carries_inline_media(request):
        return
    encoded = len(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode())
    if encoded > BEDROCK_MAXIMUM_INLINE_MEDIA_BYTES:
        raise ProviderParameterError(
            message=(
                "This model route accepts requests of at most 25 MB including inline "
                "image and video data. Send fewer or smaller files or less text."
            ),
            param="messages",
            code="invalid_parameter",
        )


def _multimodal_blocks(message: ModelMessage) -> list[JsonObject]:
    """Emit one multimodal user turn in caller order.

    Documents are named by their one-based position within the turn when the
    caller sent no filename, since Converse requires a name on every block.
    Empty text parts drop because Converse rejects an empty text block.
    """
    blocks: list[JsonObject] = []
    document_ordinal = 0
    for part in message.content_parts:
        if part.kind == "image":
            blocks.append(bedrock_image_block(part))
        elif part.kind == "document":
            document_ordinal += 1
            blocks.append(bedrock_document_block(part, document_ordinal))
        elif part.kind == "video":
            blocks.append(bedrock_video_block(part))
        elif part.kind == "audio":
            blocks.append(reject_audio_part(part))
        elif part.text:
            blocks.append({"text": part.text})
    return blocks


def _message_blocks(message: ModelMessage) -> list[JsonObject]:
    """Convert one user or assistant message into Converse content blocks.

    Args:
        message: One visible user or assistant history message.

    Returns:
        Ordered native content blocks.

    Raises:
        ValueError: The message cannot be represented without dropping context.
    """
    if message.role == "user" and message.assistant_action is not None:
        raise ValueError("user messages cannot carry assistant actions")
    if message.role == "user" and message.content is None:
        raise ValueError("user messages need text content")
    if message.content_parts:
        return _multimodal_blocks(message)
    blocks: list[JsonObject] = []
    action = message.assistant_action
    text = message.content if message.content is not None else action.content if action else None
    if text:
        blocks.append({"text": text})
    if action is not None:
        for call in action.tool_calls:
            blocks.append(
                {
                    "toolUse": {
                        "toolUseId": call.call_id,
                        "name": call.name,
                        "input": dict(call.arguments),
                    }
                }
            )
    if not blocks:
        raise ValueError(f"{message.role} messages need text or a tool call")
    return blocks


def _inference_config(
    request: ModelRequest,
    *,
    supports_temperature: bool,
    supports_top_p: bool,
    stop_sequences: Sequence[str] = (),
) -> JsonObject:
    """Return Converse inference controls without inventing omitted sampling fields."""
    inference: JsonObject = {}
    if request.maximum_output_tokens is not None:
        inference["maxTokens"] = request.maximum_output_tokens
    if request.temperature is not None and supports_temperature:
        inference["temperature"] = request.temperature
    if request.top_p is not None and supports_top_p:
        inference["topP"] = request.top_p
    if stop_sequences:
        inference["stopSequences"] = list(stop_sequences)
    return inference


def _tool_config(
    request: ModelRequest,
    *,
    strict_tool_names: Collection[str] = (),
) -> JsonObject | None:
    """Return Converse tool configuration, or omit it when tools are disabled."""
    if request.tool_choice == "none" or not request.tools:
        return None
    known_tool_names = {tool.name for tool in request.tools}
    unknown_strict_tools = set(strict_tool_names) - known_tool_names
    if unknown_strict_tools:
        raise ValueError("strict Bedrock tools must name request tool definitions")
    tools: list[JsonObject] = []
    for tool in request.tools:
        tool_spec: JsonObject = {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": {"json": tool.input_schema},
        }
        if tool.name in strict_tool_names:
            tool_spec["strict"] = True
        tools.append({"toolSpec": tool_spec})
    config: JsonObject = {"tools": tools}
    if request.tool_choice == "required":
        config["toolChoice"] = {"any": {}}
    elif isinstance(request.tool_choice, ToolChoice):
        config["toolChoice"] = {"tool": {"name": request.tool_choice.name}}
    return config
