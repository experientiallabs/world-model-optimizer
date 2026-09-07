"""Decode Chat Completions and Responses into canonical serving requests."""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from typing import Literal, cast

from openai.types import EmbeddingCreateParams
from openai.types.chat.completion_create_params import CompletionCreateParams
from openai.types.responses.response_create_params import ResponseCreateParams
from pydantic import BaseModel, Field, JsonValue, TypeAdapter, ValidationError
from pydantic_core import ErrorDetails

from exp.common.core.artifacts import ContractModel, JsonObject
from exp.common.models.model import ToolCall
from exp.runtime.gateway.compatibility import (
    CompatibilityDisposition,
    CompatibilityManifest,
)
from exp.runtime.gateway.contracts import (
    EncryptedReasoningBlock,
    ExposedReasoningContentBlock,
    GatewayApiSurface,
    GatewayMessage,
    GatewayNamedToolChoice,
    GatewayProviderNativeTool,
    GatewayRequest,
    GatewayToolDefinition,
    SealedReasoningContentBlock,
)
from exp.runtime.gateway.embeddings_contracts import EmbeddingsRequest
from exp.runtime.gateway.reasoning_carrier import (
    FIREWORKS_REASONING_CONTENT_PREFIX,
    parse_reasoning_content_carrier,
    scheme_for_carrier,
)
from exp.runtime.openai_protocol.cache_control import (
    drop_opencode_cache_control,
)
from exp.runtime.openai_protocol.enable_thinking import translate_enable_thinking
from exp.runtime.openai_protocol.errors import OpenAIProtocolError, invalid_field, unsupported_field
from exp.runtime.openai_protocol.manifest import (
    CHAT_MANIFEST,
    EMBEDDINGS_MANIFEST,
    RESPONSES_MANIFEST,
    disposition_map,
)
from exp.runtime.openai_protocol.media_parts import message_content
from exp.runtime.openai_protocol.responses_input import (
    ReplayedFunctionCall,
    ReplayedFunctionOutput,
    ReplayedInput,
    ReplayedMessage,
    ReplayedNativeItem,
    ReplayedReasoning,
    responses_input_messages,
)
from exp.runtime.openai_protocol.structured_text import (
    chat_json_object_output,
    chat_structured_text,
    responses_structured_text,
)
from exp.runtime.openai_protocol.wire_models import (
    HOSTED_TOOL_ITEM_TYPES_ASSISTANT,
    HOSTED_TOOL_ITEM_TYPES_TOOL,
    _AdditionalToolsItem,
    _AssistantToolCall,
    _ChatRequest,
    _ChatTool,
    _CustomToolCall,
    _CustomToolCallOutput,
    _EmbeddingsRequest,
    _FunctionCall,
    _HostedToolItemEcho,
    _Message,
    _ResponseFunctionCall,
    _ResponseMessage,
    _ResponseReasoningItem,
    _ResponsesInputItem,
    _ResponsesRequest,
    _ResponseTool,
)

_CHAT_OFFICIAL = TypeAdapter(CompletionCreateParams)
_RESPONSES_OFFICIAL = TypeAdapter(ResponseCreateParams)
# object-parametrized: EmbeddingCreateParams is one TypedDict, unlike the chat/responses unions.
_EMBEDDINGS_OFFICIAL: TypeAdapter[object] = TypeAdapter[object](EmbeddingCreateParams)
_TEXT_PART_TYPES = frozenset({"text", "input_text", "output_text"})


class DecodedGatewayRequest(ContractModel):
    """Public alias plus its canonical provider-neutral request."""

    alias: str = Field(min_length=1, max_length=256)
    request: GatewayRequest
    developer_messages_param: str | None = None


class DecodedEmbeddingsRequest(ContractModel):
    """Public alias plus its canonical embeddings request.

    Distinct from :class:`DecodedGatewayRequest` because the embeddings surface
    carries its own message-less, non-streaming request contract.
    """

    alias: str = Field(min_length=1, max_length=256)
    request: EmbeddingsRequest


_CHAT_MESSAGE_EXTENSION_KEYS = frozenset(
    {
        "reasoning_content",
        "provider_specific_fields",
        "thinking_blocks",
        "reasoning_items",
        "images",
    }
)
"""Message keys the strict wire model owns; hidden from official validation.

``reasoning_content`` is the authenticated Chat extension; the other four are
LiteLLM's message-dump keys, which ``_Message`` admits only in their empty (or,
for ``provider_specific_fields``, dropped-and-disclosed) forms.
"""


def _without_chat_message_extensions(payload: JsonObject) -> JsonObject:
    """Hide the wire-model-owned message keys from official OpenAI validation."""
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        return payload
    changed = False
    messages: list[JsonValue] = []
    for raw_message in raw_messages:
        if isinstance(raw_message, dict) and _CHAT_MESSAGE_EXTENSION_KEYS & raw_message.keys():
            messages.append(
                {
                    key: value
                    for key, value in raw_message.items()
                    if key not in _CHAT_MESSAGE_EXTENSION_KEYS
                }
            )
            changed = True
        else:
            messages.append(raw_message)
    return {**payload, "messages": messages} if changed else payload


def decode_chat(
    payload: JsonObject,
    *,
    idempotency_key: str | None = None,
    client_request_id: str | None = None,
) -> DecodedGatewayRequest:
    """Decode one Chat Completions body without silently dropping fields.

    OpenCode may attach an Anthropic-style ``cache_control`` annotation on
    Chat messages and on text content parts. Supported ephemeral forms are
    validated and removed before official OpenAI validation and before
    canonical conversion. Other unknown nested fields stay rejected.

    Args:
        payload: Parsed JSON request body.
        idempotency_key: Optional standard caller operation identity.
        client_request_id: Optional gateway client request identity.

    Returns:
        Public alias and lossless canonical gateway request.

    Raises:
        OpenAIProtocolError: The body is invalid, unknown, or unsupported.
    """
    payload = drop_opencode_cache_control(payload)
    _validate_manifest(payload, CHAT_MANIFEST)
    # The installed SDK's effort literal lags the newest provider tier
    # ("ultra"), so the strict wire model owns reasoning validation.
    _validate_official(
        _CHAT_OFFICIAL,
        _without_chat_message_extensions(payload),
        extension_fields={"top_k", "reasoning_effort"},
    )
    request = _validate_wire(_ChatRequest, payload)
    idempotency_key, client_request_id = _validated_operation_headers(
        idempotency_key, client_request_id
    )
    maximum = request.max_completion_tokens or request.max_tokens
    stop = (
        ()
        if request.stop is None
        else (request.stop,)
        if isinstance(request.stop, str)
        else request.stop
    )
    thinking = translate_enable_thinking(request)
    try:
        canonical = GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=_messages(request.messages, "messages"),
            tools=tuple(_chat_tool(tool) for tool in request.tools),
            tool_choice=_chat_tool_choice(request.tool_choice),
            parallel_tool_calls=request.parallel_tool_calls,
            structured_text=chat_structured_text(request.response_format),
            json_object_output=chat_json_object_output(request.response_format),
            ignored_parameters=thinking.disclosures,
            maximum_output_tokens=maximum,
            maximum_output_tokens_parameter=(
                "max_completion_tokens"
                if request.max_completion_tokens is not None
                else "max_tokens"
                if request.max_tokens is not None
                else None
            ),
            stop=stop,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            logprobs=request.logprobs,
            top_logprobs=request.top_logprobs,
            reasoning_effort=thinking.reasoning_effort,
            thinking_default_enable=thinking.thinking_default_enable,
            stream=request.stream,
            include_usage=(
                request.stream_options is not None and request.stream_options.include_usage
            ),
            metadata=request.metadata,
            safety_identifier=request.safety_identifier,
            user=request.user,
            prompt_cache_key=request.prompt_cache_key,
            service_tier=request.service_tier,
            idempotency_key=idempotency_key,
            client_request_id=client_request_id,
        )
    except ValidationError as exc:
        raise _validation_protocol_error(exc) from exc
    return DecodedGatewayRequest(alias=request.model, request=canonical)


def decode_embeddings(payload: JsonObject) -> DecodedEmbeddingsRequest:
    """Decode one Embeddings body into the canonical embeddings surface.

    The embeddings surface has no idempotency protocol yet: keyed replay is a
    future add, so an inbound ``Idempotency-Key`` header is ignored upstream
    rather than keying this decode (which therefore takes no header arguments).

    Args:
        payload: Parsed JSON request body.

    Returns:
        Public alias and canonical embeddings request.

    Raises:
        OpenAIProtocolError: The body is invalid, unknown, or unsupported.
    """
    _validate_manifest(payload, EMBEDDINGS_MANIFEST)
    _validate_official(_EMBEDDINGS_OFFICIAL, payload)
    request = _validate_wire(_EmbeddingsRequest, payload)
    inputs = (request.input,) if isinstance(request.input, str) else request.input
    try:
        canonical = EmbeddingsRequest(
            inputs=inputs,
            dimensions=request.dimensions,
            encoding_format=request.encoding_format,
            user=request.user,
        )
    except ValidationError as exc:
        raise _validation_protocol_error(exc) from exc
    return DecodedEmbeddingsRequest(alias=request.model, request=canonical)


def decode_responses(
    payload: JsonObject,
    *,
    idempotency_key: str | None = None,
    client_request_id: str | None = None,
) -> DecodedGatewayRequest:
    """Decode one Responses body into the distinct canonical surface.

    Args:
        payload: Parsed JSON request body.
        idempotency_key: Optional standard caller operation identity.
        client_request_id: Optional gateway client request identity.

    Returns:
        Public alias and lossless canonical gateway request.

    Raises:
        OpenAIProtocolError: The body is invalid, unknown, or unsupported.
    """
    _validate_manifest(payload, RESPONSES_MANIFEST)
    # The installed SDK's effort literal lags the newest provider tier
    # ("ultra"), so the strict wire model owns reasoning validation.
    request = _validate_wire(_ResponsesRequest, payload)
    official_probe = dict(payload)
    if isinstance(raw := payload.get("input"), list):
        # The installed SDK lags the live surface on echoed output items:
        # it has no message `phase` and requires `status` alongside `id`,
        # while real Codex echoes carry id+phase and omit status, and it
        # requires reasoning `content` to be an array while Codex echoes an
        # explicit null that the provider accepts (both captured
        # 2026-08-29). The strict wire model owns those contracts, so the
        # official probe sees a normalized item.
        adapted: list[JsonValue] = []
        for index, entry in enumerate(cast("list[JsonValue]", raw)):
            if isinstance(entry, dict):
                entry = _official_image_details(entry, f"input.{index}")
            if isinstance(entry, dict) and entry.get("type") == "message":
                item = {key: value for key, value in entry.items() if key != "phase"}
                if item.get("id") is not None and "status" not in item:
                    item["status"] = "completed"
                adapted.append(item)
            elif (
                isinstance(entry, dict)
                and entry.get("type") == "reasoning"
                and "content" in entry
                and entry.get("content") is None
            ):
                adapted.append({key: value for key, value in entry.items() if key != "content"})
            else:
                adapted.append(entry)
        official_probe["input"] = adapted
    _validate_official(
        _RESPONSES_OFFICIAL,
        official_probe,
        extension_fields={"top_k", "reasoning", "client_metadata"},
    )
    include_encrypted_reasoning = _include_encrypted_reasoning(request.include)
    idempotency_key, client_request_id = _validated_operation_headers(
        idempotency_key, client_request_id
    )
    raw_input = payload.get("input")
    raw_tools = payload.get("tools")
    function_tools: list[GatewayToolDefinition] = []
    native_tools: list[GatewayProviderNativeTool] = []
    for tool_index, declared in enumerate(request.tools):
        if isinstance(declared, _ResponseTool):
            function_tools.append(_response_tool(declared))
        else:
            # The raw caller declaration, not the re-serialized wire model,
            # so the native rung receives it byte-for-byte at its position.
            assert isinstance(raw_tools, list)
            native_tools.append(
                GatewayProviderNativeTool(
                    index=tool_index,
                    tool=cast("JsonObject", raw_tools[tool_index]),
                )
            )
    replayed_items = cast("list[JsonObject]", raw_input) if isinstance(raw_input, list) else ()
    try:
        messages = list(_response_input_messages(request.input, raw_items=replayed_items))
    except ValidationError as exc:
        # History reconstruction folds echoed items into canonical messages,
        # so a canonical-contract violation (such as duplicate call_ids in one
        # assistant segment) first surfaces here, past the wire models. It is
        # caller-shaped input all the same: name the rule instead of letting
        # the exception escape as an unclassified 500.
        detail = exc.errors(include_url=False)[0]
        raise invalid_field(
            "input",
            "Invalid value for 'input': " + detail["msg"].removeprefix("Value error, ") + ".",
        ) from exc
    if request.instructions is not None:
        messages.insert(0, GatewayMessage(role="developer", content=request.instructions))
    try:
        canonical = GatewayRequest(
            surface=GatewayApiSurface.RESPONSES,
            messages=tuple(messages),
            tools=tuple(function_tools),
            provider_native_tools=tuple(native_tools),
            tool_choice=_responses_tool_choice(request.tool_choice),
            parallel_tool_calls=request.parallel_tool_calls,
            structured_text=responses_structured_text(request.text),
            maximum_output_tokens=request.max_output_tokens,
            maximum_output_tokens_parameter=(
                "max_output_tokens" if request.max_output_tokens is not None else None
            ),
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            logprobs=(True if request.top_logprobs is not None else None),
            top_logprobs=request.top_logprobs,
            reasoning_effort=(request.reasoning.effort if request.reasoning is not None else None),
            reasoning_context=(
                request.reasoning.context if request.reasoning is not None else None
            ),
            reasoning_summary=(
                request.reasoning.summary or request.reasoning.generate_summary
                if request.reasoning is not None
                else None
            ),
            reasoning_summary_parameters=(
                tuple(
                    path
                    for path, value in (
                        ("reasoning.generate_summary", request.reasoning.generate_summary),
                        ("reasoning.summary", request.reasoning.summary),
                    )
                    if value is not None
                )
                if request.reasoning is not None
                else ()
            ),
            text_verbosity=(request.text.verbosity if request.text is not None else None),
            client_metadata=request.client_metadata,
            response_store=request.store,
            include_encrypted_reasoning=include_encrypted_reasoning,
            stream=request.stream,
            previous_response_id=request.previous_response_id,
            metadata=request.metadata,
            safety_identifier=request.safety_identifier,
            user=request.user,
            prompt_cache_key=request.prompt_cache_key,
            service_tier=request.service_tier,
            idempotency_key=idempotency_key,
            client_request_id=client_request_id,
        )
    except ValidationError as exc:
        raise _validation_protocol_error(exc) from exc
    developer_messages_param = None
    if request.instructions is not None:
        developer_messages_param = "instructions"
    elif not isinstance(request.input, str):
        developer_index = next(
            (
                index
                for index, item in enumerate(request.input)
                if isinstance(item, _ResponseMessage) and item.role == "developer"
            ),
            None,
        )
        if developer_index is not None:
            developer_messages_param = f"input.{developer_index}.role"
    return DecodedGatewayRequest(
        alias=request.model,
        request=canonical,
        developer_messages_param=developer_messages_param,
    )


def _official_image_details(entry: JsonObject, param: str) -> JsonObject:
    """Default the detail level of every ``input_image`` part of one item.

    The Responses surface treats ``input_image.detail`` as optional and
    resolves an omitted level to ``auto``, while the installed SDK marks the
    field required. Only the official probe sees the resolved default: the
    strict wire model owns the real contract and keeps an unstated level
    unstated on the provider wire. An ``input_audio`` part is refused by name.
    """
    content = entry.get("content")
    if not isinstance(content, list):
        return entry
    parts: list[JsonValue] = []
    for index, part in enumerate(cast("list[JsonValue]", content)):
        if isinstance(part, dict) and part.get("type") == "input_audio":
            raise unsupported_field(
                f"{param}.content.{index}.input_audio",
                message="Audio input is not available on Responses; use Chat Completions.",
            )
        if isinstance(part, dict) and part.get("type") == "input_image" and "detail" not in part:
            parts.append({**part, "detail": "auto"})
        else:
            parts.append(part)
    return {**entry, "content": parts}


def _validate_manifest(payload: JsonObject, manifest: CompatibilityManifest) -> None:
    """Reject unsupported and unknown top-level fields before responder work."""
    decisions = disposition_map(manifest)
    for field in payload:
        disposition = decisions.get(field)
        if disposition is None or disposition == CompatibilityDisposition.UNSUPPORTED:
            raise unsupported_field(field)


def _validate_official(
    adapter: TypeAdapter[object],
    payload: JsonObject,
    *,
    extension_fields: Collection[str] = frozenset(),
) -> None:
    """Run the installed official SDK request schema before gateway narrowing."""
    try:
        official_payload = {
            key: value for key, value in payload.items() if key not in extension_fields
        }
        adapter.validate_python(official_payload)
    except ValidationError as exc:
        raise _validation_protocol_error(exc) from exc


def _validate_wire[ModelT: BaseModel](model: type[ModelT], payload: JsonObject) -> ModelT:
    """Validate one strict private wire model with a field-specific public error."""
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise _validation_protocol_error(exc) from exc


_LOCATION_NOISE = {"body", "non-streaming", "streaming"}
_UNION_BRANCH_TYPES = {"str", "int", "float", "bool", "list", "tuple", "dict", "NoneType"}
_OUTPUT_ITEM_VARIANTS = {
    "message",
    "function_call",
    "function_call_output",
    "reasoning",
    "additional_tools",
    "custom_tool_call",
    "custom_tool_call_output",
    # Hosted-tool echo variants share the same union-branch label shape.
    *HOSTED_TOOL_ITEM_TYPES_TOOL,
    *HOSTED_TOOL_ITEM_TYPES_ASSISTANT,
}


def _cleaned_location(location: tuple[str | int, ...]) -> tuple[str, ...]:
    """Drop pydantic union-branch labels so the path names request fields."""
    cleaned: list[str] = []
    for part in location:
        text = str(part)
        if text in _LOCATION_NOISE:
            continue
        # Typed-dict union branches are labeled with their class name, which
        # no request field ever shares: every public field is lower case.
        if isinstance(part, str) and (
            part.startswith("_") or "[" in text or text in _UNION_BRANCH_TYPES or text[:1].isupper()
        ):
            continue
        if text in _OUTPUT_ITEM_VARIANTS and cleaned and cleaned[-1].isdigit():
            continue
        # A discriminated part whose tag is also its payload field name
        # (``file.file``, ``image_url.image_url``) reports the tag once.
        if cleaned and cleaned[-1] == text and not text.isdigit():
            continue
        cleaned.append(text)
    return tuple(cleaned)


_WIRE_TYPE_NAMES = {
    "str": "a string",
    "int": "an integer",
    "float": "a number",
    "bool": "a boolean",
    "list": "an array",
    "tuple": "an array",
    "dict": "an object",
    "NoneType": "null",
}
"""JSON-shape names for python input types, used in expected/got messages."""

_EXPECTED_BY_ERROR_TYPE = {
    "string_type": "a string",
    "string_too_short": "a non-empty string",
    "int_type": "an integer",
    "int_parsing": "an integer",
    "float_type": "a number",
    "float_parsing": "a number",
    "bool_type": "a boolean",
    "list_type": "an array",
    "tuple_type": "an array",
    "dict_type": "an object",
    "model_type": "an object",
    "model_attributes_type": "an object",
    "missing": "a value",
    "none_required": "null",
}
"""Shape-level expectations for the pydantic error types worth naming."""


def _shape_message(param: str, details: list[ErrorDetails]) -> str | None:
    """Describe what shape a field expected versus what arrived.

    Only structural facts appear: expectations come from this gateway's own
    wire models and the got side is the JSON type of the caller's value,
    never the value itself and never provider prose.
    """
    expected: list[str] = []
    got: str | None = None
    for detail in details:
        if detail["type"] == "string_too_long":
            # The bound and the arriving LENGTH are both display-safe facts
            # (the value itself is never echoed); stating them saves the
            # caller from bisecting the ceiling out of a bare rejection.
            context = detail.get("ctx") or {}
            maximum = context.get("max_length")
            value = detail.get("input")
            if isinstance(maximum, int) and isinstance(value, str):
                return (
                    f"Invalid value for '{param}': expected at most "
                    f"{maximum:,} characters, but got {len(value):,}."
                )
        phrase = _EXPECTED_BY_ERROR_TYPE.get(detail["type"])
        if detail["type"] in {"literal_error", "enum"}:
            context = detail.get("ctx") or {}
            allowed = context.get("expected")
            if isinstance(allowed, str):
                phrase = f"one of {allowed}"
        if phrase is not None and phrase not in expected:
            expected.append(phrase)
        # A missing-field complaint carries the parent object as its input,
        # so it contributes no honest "got" type.
        if detail["type"] != "missing" and "input" in detail:
            got = _WIRE_TYPE_NAMES.get(type(detail["input"]).__name__, got)
    if not expected:
        return None
    description = " or ".join(expected)
    if got is not None:
        return f"Invalid value for '{param}': expected {description}, but got {got} instead."
    return f"Invalid value for '{param}': expected {description}."


def _validation_protocol_error(error: ValidationError) -> OpenAIProtocolError:
    """Convert Pydantic locations into stable dotted OpenAI ``param`` paths.

    Union validation reports every branch's complaints. Errors group by
    their branch (the location minus its final field segment); among the
    most field-specific groups, the branch the caller actually meant is the
    one with the fewest complaints, so its deepest cleaned location names
    the real field (an echoed item's ``input.1.caller``), never a union
    branch label such as ``input.str``. The chosen field's own complaints
    then name the expected shape against the arriving JSON type.
    """
    groups: dict[tuple[str | int, ...], list[tuple[tuple[str, ...], ErrorDetails]]] = {}
    for detail in error.errors(include_url=False):
        groups.setdefault(tuple(detail["loc"][:-1]), []).append(
            (_cleaned_location(detail["loc"]), detail)
        )
    if not groups:
        return invalid_field("body")
    deepest = max(len(location) for members in groups.values() for location, _ in members)
    candidates = [
        members
        for members in groups.values()
        if any(len(location) == deepest for location, _ in members)
    ]
    best = min(candidates, key=len)
    location = max((cleaned for cleaned, _ in best), key=len, default=())
    param = ".".join(location) or "body"
    details = [detail for cleaned, detail in best if cleaned == location]
    if param == "body":
        # A whole-request rule (such as the attachment count ceiling) has no
        # field of its own, so its own wording is the only useful message.
        for detail in details:
            if detail["type"] == "value_error":
                return invalid_field(param, detail["msg"].removeprefix("Value error, ") + ".")
    for detail in details:
        # A field validator's own wording states this gateway's exact value
        # constraint (only our wire models raise these, so the text is
        # display-safe and never echoes the caller's value).
        if detail["type"] == "value_error":
            return invalid_field(
                param,
                f"Invalid value for {param!r}: "
                + detail["msg"].removeprefix("Value error, ")
                + ".",
            )
    return invalid_field(param, _shape_message(param, details))


def _validated_operation_headers(
    idempotency_key: str | None, client_request_id: str | None
) -> tuple[str | None, str | None]:
    """Validate the two caller identity headers as independent concepts.

    ``Idempotency-Key`` names one retriable operation and is the only header
    that keys replay and duplicate detection. ``X-Client-Request-Id`` is a
    caller correlation identity: Codex sends its session id there on every
    request of a session (captured live 2026-08-29), and the provider serves
    those requests without deduplication, so treating it as an operation key
    would reject the second request of every real session as a conflict. It
    is echoed on responses and scopes route affinity only, and the two
    headers may therefore carry different values.
    """
    for name, value in (
        ("Idempotency-Key", idempotency_key),
        ("X-Client-Request-Id", client_request_id),
    ):
        if value is not None and (
            not value or len(value) > 512 or any(ord(char) < 32 for char in value)
        ):
            raise invalid_field(name, f"{name} must be a non-empty display-safe value.")
    return idempotency_key, client_request_id


def _messages(messages: tuple[_Message, ...], prefix: str) -> tuple[GatewayMessage, ...]:
    """Convert ordered wire messages while retaining raw assistant arguments."""
    converted: list[GatewayMessage] = []
    for message_index, message in enumerate(messages):
        calls = tuple(
            _tool_call(call, f"{prefix}.{message_index}.tool_calls.{call_index}.function.arguments")
            for call_index, call in enumerate(message.history_tool_calls)
        )
        provider_reasoning: tuple[
            SealedReasoningContentBlock | ExposedReasoningContentBlock, ...
        ] = ()
        if message.reasoning_content is not None:
            param = f"{prefix}.{message_index}.reasoning_content"
            # The scheme is fixed by the carrier's own opaque prefix. A known
            # prefix MUST parse as that provider's carrier. Text under no known
            # prefix is the plaintext an exposure-gated rung itself returned on
            # a non-tool turn (Tencent/DeepSeek): it decodes as caller-owned
            # history and route admission decides which rungs may carry it.
            scheme = scheme_for_carrier(message.reasoning_content)
            if scheme is None:
                # Plaintext reasoning is caller-owned history on ANY assistant
                # turn, tool-call turns included: AI-SDK clients re-serialize a
                # reasoning part onto the same assistant message as its tool
                # calls, and exposure-gated providers themselves emit
                # reasoning_content on tool turns. The field is baked into the
                # transcript, so rejecting it wedges every session that ever
                # touched a reasoning-exposed model (the sealed-carrier bond
                # applies only to text presented AS a gateway-issued carrier,
                # which keeps its strict path below).
                try:
                    provider_reasoning = (
                        ExposedReasoningContentBlock(content=message.reasoning_content),
                    )
                except ValidationError as exc:
                    raise invalid_field(
                        param,
                        f"'{param}' must be non-empty plaintext reasoning within the size bound "
                        "or a gateway-issued carrier.",
                    ) from exc
            else:
                try:
                    provider_reasoning = (
                        parse_reasoning_content_carrier(message.reasoning_content, scheme=scheme),
                    )
                except ValueError as exc:
                    raise invalid_field(
                        param, f"'{param}' must be a gateway-issued carrier."
                    ) from exc
        content, content_parts = message_content(
            message.content, f"{prefix}.{message_index}.content"
        )
        converted.append(
            GatewayMessage(
                role=message.role,
                content=content,
                content_parts=content_parts,
                tool_call_id=message.tool_call_id,
                tool_calls=calls,
                provider_tool_name=message.name,
                provider_reasoning=provider_reasoning,
                # An empty object is the common LiteLLM stamp and carries
                # nothing to disclose; only a populated one is a dropped field.
                provider_specific_fields=message.provider_specific_fields or None,
            )
        )
    return tuple(converted)


def _tool_call(call: _AssistantToolCall, param: str) -> ToolCall:
    """Parse one complete tool call while retaining its exact raw argument string."""
    # Some SDK stacks echo a zero-argument call as an empty string; the
    # canonical empty object mirrors the streaming completion seed, since no
    # provider wire accepts empty argument bytes.
    raw_arguments = call.function.arguments or "{}"
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise invalid_field(param, f"'{param}' must encode one JSON object.") from exc
    if not isinstance(parsed, dict):
        raise invalid_field(param, f"'{param}' must encode one JSON object.")
    return ToolCall(
        call_id=call.id,
        name=call.function.name,
        arguments=cast(JsonObject, parsed),
        raw_arguments=raw_arguments,
        cache_control=(
            call.cache_control.model_dump(mode="json", exclude_none=True)
            if call.cache_control is not None
            else None
        ),
    )


def _chat_tool(tool: _ChatTool) -> GatewayToolDefinition:
    """Convert one Chat function tool without weakening strictness."""
    return GatewayToolDefinition(
        name=tool.function.name,
        description=tool.function.description,
        parameters=tool.function.parameters,
        strict=tool.function.strict,
    )


def _response_tool(tool: _ResponseTool) -> GatewayToolDefinition:
    """Convert one Responses function tool without weakening strictness."""
    return GatewayToolDefinition(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
        strict=bool(tool.strict),
    )


def _chat_tool_choice(
    value: JsonValue,
) -> Literal["auto", "none", "required"] | GatewayNamedToolChoice | None:
    """Normalize Chat tool-choice strings and named-function objects."""
    if value is None:
        return None
    if isinstance(value, str) and value in {"auto", "none", "required"}:
        return cast(Literal["auto", "none", "required"], value)
    if isinstance(value, dict):
        function = value.get("function")
        if value.get("type") == "function" and isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str):
                return GatewayNamedToolChoice(name=name)
    raise invalid_field("tool_choice")


def _responses_tool_choice(
    value: JsonValue,
) -> Literal["auto", "none", "required"] | GatewayNamedToolChoice | None:
    """Normalize Responses tool-choice strings and named-function objects."""
    if value is None:
        return None
    if isinstance(value, str) and value in {"auto", "none", "required"}:
        return cast(Literal["auto", "none", "required"], value)
    if isinstance(value, dict) and value.get("type") == "function":
        name = value.get("name")
        if isinstance(name, str):
            return GatewayNamedToolChoice(name=name)
    raise invalid_field("tool_choice")


def _include_encrypted_reasoning(include: tuple[str, ...] | None) -> bool:
    """Validate the closed ``include`` selector list.

    Args:
        include: Raw caller include paths.

    Returns:
        Whether the caller asked for ``reasoning.encrypted_content``.

    Raises:
        OpenAIProtocolError: An include path is not supported by this gateway.
    """
    if include is None:
        return False
    for path in include:
        if path != "reasoning.encrypted_content":
            raise invalid_field(
                "include",
                f"The include path {path!r} is not supported by this gateway. "
                "Only 'reasoning.encrypted_content' is available.",
            )
    return bool(include)


def _response_input_messages(
    value: str | tuple[_ResponsesInputItem, ...],
    *,
    raw_items: Sequence[JsonObject] = (),
) -> tuple[GatewayMessage, ...]:
    """Validate replay details and reconstruct OpenAI or Fireworks history."""
    if isinstance(value, str):
        return responses_input_messages(value)
    replayed: list[ReplayedInput] = []
    for index, item in enumerate(value):
        if isinstance(
            item,
            (_AdditionalToolsItem, _CustomToolCall, _CustomToolCallOutput, _HostedToolItemEcho),
        ):
            # The raw caller item, not the re-serialized wire model, so the
            # native rung receives the item byte-for-byte.
            if isinstance(item, _HostedToolItemEcho):
                native_role = "tool" if item.type in HOSTED_TOOL_ITEM_TYPES_TOOL else "assistant"
            elif isinstance(item, _CustomToolCall):
                native_role = "assistant"
            elif isinstance(item, _CustomToolCallOutput):
                native_role = "tool"
            else:
                native_role = "developer"
            replayed.append(
                ReplayedNativeItem(index=index, role=native_role, item=raw_items[index])
            )
        elif isinstance(item, _ResponseReasoningItem):
            if item.encrypted_content is None:
                # A store=true flow replays reasoning by item id alone (the
                # SDK marks encrypted_content optional); only the issuing
                # native Responses wire can resolve the id, so the item is
                # carried verbatim like a hosted-tool item and the provider
                # judges resolvability, rather than rejecting SDK-legal
                # input the provider itself may serve.
                replayed.append(
                    ReplayedNativeItem(index=index, role="assistant", item=raw_items[index])
                )
                continue
            if item.encrypted_content.startswith(FIREWORKS_REASONING_CONTENT_PREFIX):
                try:
                    block: EncryptedReasoningBlock | SealedReasoningContentBlock = (
                        parse_reasoning_content_carrier(item.encrypted_content)
                    )
                except ValueError as exc:
                    raise invalid_field(
                        f"input.{index}.encrypted_content",
                        "Responses encrypted_content must be a gateway-issued carrier.",
                    ) from exc
            else:
                block = EncryptedReasoningBlock(
                    id=item.id,
                    encrypted_content=item.encrypted_content,
                    output_index=index,
                    status=item.status,
                )
            replayed.append(ReplayedReasoning(index=index, block=block))
        elif isinstance(item, _ResponseMessage):
            converted = _messages((item,), f"input.{index}")
            if converted and item.role == "assistant":
                converted = (
                    converted[0].model_copy(
                        update={
                            "provider_item_id": item.id,
                            "provider_output_index": index if item.id is not None else None,
                            "provider_status": item.status,
                            "provider_phase": item.phase,
                        }
                    ),
                    *converted[1:],
                )
            replayed.append(ReplayedMessage(index=index, message=converted[0]))
        elif isinstance(item, _ResponseFunctionCall):
            wire_call = _AssistantToolCall(
                id=item.call_id,
                function=_FunctionCall(name=item.name, arguments=item.arguments),
            )
            replayed.append(
                ReplayedFunctionCall(
                    index=index,
                    call=_tool_call(wire_call, f"input.{index}.arguments").model_copy(
                        update={
                            "provider_item_id": item.id,
                            "provider_output_index": index,
                            "provider_status": item.status,
                            "provider_namespace": item.namespace,
                            "provider_caller": item.caller,
                        }
                    ),
                )
            )
        else:
            if isinstance(item.output, str):
                output_text, output_parts = item.output, ()
            else:
                # The SDK list form: text and image parts map onto the
                # canonical tool message (the tool-message contract carries
                # exactly those two kinds); any other kind is a named 400
                # because a tool result has no canonical carrier for it and
                # dropping it would misstate what the tool returned.
                output_text, output_parts = message_content(item.output, f"input.{index}.output")
                unsupported = next(
                    (part for part in output_parts if part.kind not in ("text", "image")),
                    None,
                )
                if unsupported is not None:
                    raise unsupported_field(
                        f"input.{index}.output",
                        message=(
                            "function_call_output.output supports text and image parts "
                            f"only; this list carries a {unsupported.kind!r} part."
                        ),
                    )
                output_text = output_text or ""
            replayed.append(
                ReplayedFunctionOutput(
                    index=index,
                    call_id=item.call_id,
                    output=output_text,
                    name=item.name,
                    namespace=item.namespace,
                    caller=item.caller,
                    content_parts=output_parts,
                )
            )
    return responses_input_messages(tuple(replayed))
