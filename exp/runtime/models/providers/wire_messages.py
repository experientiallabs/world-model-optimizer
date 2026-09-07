"""Canonical gateway message translation to each provider wire vocabulary.

These translators are the one place a canonical :class:`GatewayMessage`
(including its opaque provider-reasoning carrier) becomes provider content
items or blocks; every streaming payload builder composes them so the two
engines cannot drift at the message boundary.
"""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import (
    TOOL_ERROR_TEXT_PREFIX,
    GatewayMessage,
    GatewayNamedToolChoice,
    GatewayRequest,
)
from exp.runtime.models.providers.audios import openai_chat_audio_part, reject_audio_part
from exp.runtime.models.providers.documents import (
    anthropic_document_block,
    openai_chat_document_part,
    responses_document_part,
)
from exp.runtime.models.providers.errors import ProviderResponseError
from exp.runtime.models.providers.images import (
    anthropic_image_block,
    openai_chat_image_part,
    responses_image_part,
)
from exp.runtime.models.providers.videos import openai_chat_video_part, reject_video_part


def responses_items(message: GatewayMessage) -> list[JsonObject]:
    """Translate one non-instruction gateway message to Responses input items."""
    if message.role == "tool":
        output_value: str | list[JsonObject]
        if message.content_parts:
            # The SDK list form round-trips: text and image parts re-emit as
            # typed output parts (the image encoder is the same one user
            # messages use). A failed invocation folds its error prefix in as
            # a leading text part, derived from the canonical flag each time
            # so replays never accumulate prefixes.
            output_value = []
            for part in message.content_parts:
                if part.kind == "text":
                    output_value.append({"type": "input_text", "text": part.text})
                elif part.kind == "image":
                    output_value.append(responses_image_part(part))
                else:
                    # The canonical contract restricts tool messages to text
                    # and image parts; anything else here is a gateway bug.
                    raise ProviderResponseError("tool results carry only text and image parts")
            if message.tool_is_error:
                output_value.insert(0, {"type": "input_text", "text": TOOL_ERROR_TEXT_PREFIX})
        else:
            output_value = message.folded_tool_error_content()
        output_item: JsonObject = {
            "type": "function_call_output",
            "call_id": message.tool_call_id or "",
            "output": output_value,
        }
        # Tool attribution round-trips verbatim: present stays present and
        # absent stays absent, so pre-namespace histories are unchanged.
        if message.provider_tool_name is not None:
            output_item["name"] = message.provider_tool_name
        if message.provider_tool_namespace is not None:
            output_item["namespace"] = message.provider_tool_namespace
        if message.provider_tool_caller is not None:
            output_item["caller"] = message.provider_tool_caller
        return [output_item]
    if message.role == "user":
        if message.content_parts:
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": part.text}
                        if part.kind == "text"
                        else responses_image_part(part)
                        if part.kind == "image"
                        else reject_video_part(part)
                        if part.kind == "video"
                        else reject_audio_part(part)
                        if part.kind == "audio"
                        else responses_document_part(part)
                        for part in message.content_parts
                    ],
                }
            ]
        return [{"role": "user", "content": message.content or ""}]
    if message.role != "assistant":
        raise ProviderResponseError("unsupported Responses message role")
    items: list[JsonObject] = []
    indexed_items: list[tuple[int, JsonObject]] = []
    for block in message.provider_reasoning:
        if block.kind == "exposed_reasoning_content":
            # Plaintext reasoning replays only on an exposure-gated Chat rung;
            # route narrowing disclosed the drop for this wire.
            continue
        if block.kind != "encrypted_reasoning":
            # Anthropic thinking cannot replay on the OpenAI wire; route
            # admission rejects the combination before dispatch.
            raise ProviderResponseError("thinking blocks cannot replay on the Responses wire")
        # Reasoning items precede the assistant action they belong to, and
        # the encrypted payload is the round-trip authority; the display-only
        # summary is deliberately empty on replay. The item id and status are
        # NEVER forwarded (verified live 2026-08-29): the provider
        # cryptographically binds encrypted_content to its ORIGINAL item id
        # and rejects any mismatch ("Encrypted content item_id did not
        # match") while an id-less item verifies against the id embedded in
        # the payload itself, and a replayed reasoning item with `status` is
        # rejected outright ("Unknown parameter: 'input[N].status'") even
        # though function_call and message items accept it.
        item: JsonObject = {
            "type": "reasoning",
            "summary": [],
            "encrypted_content": block.encrypted_content,
        }
        if block.output_index is None:
            items.append(item)
        else:
            indexed_items.append((block.output_index, item))
    if message.content is not None or message.provider_item_id is not None:
        if message.provider_output_index is None:
            items.append({"role": "assistant", "content": message.content})
        else:
            if message.provider_item_id is None:
                raise ProviderResponseError("Responses assistant output omitted its provider ID")
            output_message: JsonObject = {
                "type": "message",
                "id": message.provider_item_id,
                "role": "assistant",
                "status": message.provider_status or "completed",
                "content": (
                    [
                        {
                            "type": "output_text",
                            "text": message.content,
                            "annotations": [],
                            "logprobs": [],
                        }
                    ]
                    if message.content is not None
                    else []
                ),
            }
            if message.provider_phase is not None:
                output_message["phase"] = message.provider_phase
            indexed_items.append((message.provider_output_index, output_message))
    for call in message.tool_calls:
        item: JsonObject = {
            "type": "function_call",
            "call_id": call.call_id,
            "name": call.name,
            "arguments": call.arguments_json(),
        }
        # The provider rejects a namespaced call replayed without its
        # namespace ("Missing namespace for function_call ..."), so the
        # retained value re-emits verbatim; absent stays absent.
        if call.provider_namespace is not None:
            item["namespace"] = call.provider_namespace
        # SDK 3.0 programmatic tool-calling attribution round-trips verbatim
        # like the namespace; absent stays absent.
        if call.provider_caller is not None:
            item["caller"] = call.provider_caller
        if call.provider_item_id is not None:
            item["id"] = call.provider_item_id
        if call.provider_status is not None:
            item["status"] = call.provider_status
        if call.provider_output_index is None:
            items.append(item)
        else:
            indexed_items.append((call.provider_output_index, item))
    if indexed_items:
        output_indexes = tuple(index for index, _item in indexed_items)
        if len(output_indexes) != len(set(output_indexes)):
            raise ProviderResponseError("Responses output items repeated a provider index")
        items[:0] = [item for _index, item in sorted(indexed_items)]
    return items


def retained_cache_marked_blocks(
    blocks: tuple[JsonObject, ...] | list[JsonObject],
) -> list[JsonObject]:
    """Drop empty text blocks while keeping their cache breakpoints.

    The wire rejects empty text blocks, but Claude Code lands its
    prompt-cache marker on the LAST block of a turn, which can be empty;
    dropping the block must not drop the breakpoint or the whole prefix
    bills uncached (the block-cache incident class). A displaced marker
    lands on the closest retained block before it (an empty block adds no
    bytes, so that boundary is byte-identical), or on the first retained
    block after it when nothing precedes (a slightly wider, still valid
    breakpoint). Adjacent duplicate markers collapse: one marker per
    boundary suffices.
    """
    retained: list[JsonObject] = []
    displaced: object | None = None
    for block in blocks:
        if block.get("text"):
            kept = dict(block)
            if displaced is not None and "cache_control" not in kept:
                kept["cache_control"] = displaced
            displaced = None
            retained.append(kept)
        elif "cache_control" in block:
            if retained:
                if "cache_control" not in retained[-1]:
                    retained[-1] = {**retained[-1], "cache_control": block["cache_control"]}
            else:
                displaced = block["cache_control"]
    return retained


def _anthropic_multimodal_blocks(message: GatewayMessage) -> list[JsonObject]:
    """Emit one multimodal user turn in caller order, markers intact.

    The cache-marked run holds the caller's text blocks verbatim, one per
    retained text part and in the same order, so a marker on the last text
    block of a turn that also carries an image still re-emits: dropping it
    would make the whole prefix uncacheable on exactly the turns Claude Code
    marks.

    Args:
        message: A user message carrying at least one image part.

    Returns:
        The ordered Anthropic content blocks for the turn.
    """
    marked = retained_cache_marked_blocks(message.provider_text_blocks)
    blocks: list[JsonObject] = []
    text_index = 0
    for part in message.content_parts:
        if part.kind == "image":
            blocks.append(anthropic_image_block(part))
            continue
        if part.kind == "video":
            blocks.append(reject_video_part(part))
            continue
        if part.kind == "audio":
            blocks.append(reject_audio_part(part))
            continue
        if part.kind == "document":
            blocks.append(anthropic_document_block(part))
            continue
        if not part.text:
            # This wire rejects empty text content blocks post-dispatch
            # ("text content blocks must be non-empty"), and an empty part
            # carries nothing, so it drops loss-free; the turn's attachment
            # guarantees the content array stays non-empty.
            continue
        blocks.append(
            marked[text_index] if text_index < len(marked) else {"type": "text", "text": part.text}
        )
        text_index += 1
    return blocks


_UNMARKABLE_REPLAY_BLOCKS = frozenset({"thinking", "redacted_thinking"})


def ordered_blocks_with_markers(blocks: tuple[JsonObject, ...]) -> list[JsonObject]:
    """Replay an ordered turn: empty text drops, its cache marker survives.

    Same rule as :func:`retained_cache_marked_blocks`, extended to a turn that
    mixes block kinds: a displaced marker lands on the closest retained block
    before it that can carry one (text or tool_use; the wire refuses markers
    on thinking blocks), else on the first such block after it. Signed
    blocks themselves are never touched.
    """
    retained: list[JsonObject] = []
    displaced: object | None = None

    def markable(block: JsonObject) -> bool:
        return block.get("type") not in _UNMARKABLE_REPLAY_BLOCKS

    for block in blocks:
        if block.get("type") == "text" and not block.get("text"):
            marker = block.get("cache_control")
            if marker is None:
                continue
            carrier = next(
                (index for index in range(len(retained) - 1, -1, -1) if markable(retained[index])),
                None,
            )
            if carrier is not None:
                if "cache_control" not in retained[carrier]:
                    retained[carrier] = {**retained[carrier], "cache_control": marker}
            else:
                displaced = marker
            continue
        kept = dict(block)
        if displaced is not None and markable(kept) and "cache_control" not in kept:
            kept["cache_control"] = displaced
            displaced = None
        retained.append(kept)
    return retained


def _reasoning_intact(message: GatewayMessage) -> bool:
    """Whether the flattened reasoning still holds every thinking block the caller sent.

    Admission may strip or narrow reasoning blocks for a rung; the verbatim
    order is only replayed when it would say exactly what the flattened
    fields say.
    """
    assert message.provider_anthropic_blocks is not None
    sent = sum(
        1
        for block in message.provider_anthropic_blocks
        if block.get("type") in {"thinking", "redacted_thinking"}
    )
    kept = sum(
        1 for block in message.provider_reasoning if block.kind in {"thinking", "redacted_thinking"}
    )
    return sent == kept and sent > 0


def anthropic_blocks(message: GatewayMessage) -> tuple[str, list[JsonObject]]:
    """Translate one non-instruction gateway message to Anthropic content blocks."""
    if message.role == "tool":
        result: JsonObject = {
            "type": "tool_result",
            "tool_use_id": message.tool_call_id or "",
            "content": message.content or "",
        }
        if message.content_parts:
            # A tool screenshot re-emits as the caller's exact block run:
            # image parts become image blocks in their original positions.
            # The canonical model restricts tool messages to these two kinds.
            # Empty text parts drop loss-free (the wire rejects empty text
            # blocks); an all-empty run keeps the flattened string content.
            run: list[JsonObject] = []
            for part in message.content_parts:
                if part.kind == "image":
                    run.append(anthropic_image_block(part))
                elif part.kind == "text" and part.text:
                    block: JsonObject = {"type": "text", "text": part.text}
                    if part.cache_control is not None:
                        block["cache_control"] = part.cache_control
                    run.append(block)
            if run:
                result["content"] = run
        # Only the Anthropic wire can express a failed tool invocation; the
        # marker is emitted solely when set so existing payloads are unchanged.
        if message.tool_is_error:
            result["is_error"] = True
        # A caller cache marker on the tool result re-emits with its block:
        # this is where Claude Code's conversation breakpoints usually land.
        if message.cache_control is not None:
            result["cache_control"] = message.cache_control
        return ("user", [result])
    if message.role == "user":
        if message.content_parts:
            # The caller's exact interleaving is preserved: an image before
            # its question reads differently from one after it.
            return "user", _anthropic_multimodal_blocks(message)
        marked_run = retained_cache_marked_blocks(message.provider_text_blocks)
        if marked_run:
            # The cache-marked run re-emits the caller's blocks with empty
            # ones dropped loss-free (the wire rejects them and they carry
            # nothing) and their breakpoints migrated to a retained
            # neighbor; the flattened content stays canonical elsewhere.
            return "user", marked_run
        return "user", [{"type": "text", "text": message.content or ""}]
    if message.role != "assistant":
        raise ProviderResponseError("unsupported Anthropic message role")
    if message.provider_anthropic_block is not None:
        # An echoed server-tool block re-emits byte-for-byte at its position;
        # route admission guarantees this dispatch is an Anthropic rung.
        return "assistant", [message.provider_anthropic_block]
    if message.provider_anthropic_blocks is not None and _reasoning_intact(message):
        # A thinking turn replays in the caller's own block order: Anthropic
        # verifies the latest assistant message against its signatures, and
        # interleaved thinking puts thinking between tool_use blocks, an
        # order the flattened fields below cannot express.
        return "assistant", ordered_blocks_with_markers(message.provider_anthropic_blocks)
    blocks: list[JsonObject] = []
    for reasoning in message.provider_reasoning:
        if reasoning.kind == "exposed_reasoning_content":
            # Plaintext reasoning replays only on an exposure-gated Chat rung;
            # route narrowing disclosed the drop for this wire.
            continue
        # Thinking blocks lead the assistant turn (the Anthropic contract)
        # and re-emit verbatim: the signature must round-trip byte-exact.
        if reasoning.kind == "thinking":
            thinking: JsonObject = {"type": "thinking", "thinking": reasoning.text}
            if reasoning.signature is not None:
                thinking["signature"] = reasoning.signature
            blocks.append(thinking)
        elif reasoning.kind == "redacted_thinking":
            blocks.append({"type": "redacted_thinking", "data": reasoning.data})
        else:
            # OpenAI encrypted reasoning cannot replay on the Anthropic wire;
            # route admission rejects the combination before dispatch.
            raise ProviderResponseError("encrypted reasoning cannot replay on the Anthropic wire")
    if message.provider_text_blocks:
        # Empty marked blocks drop with their breakpoints migrated, exactly
        # as on user turns; the wire rejects an empty text block anywhere.
        blocks.extend(retained_cache_marked_blocks(message.provider_text_blocks))
    elif message.content:
        blocks.append({"type": "text", "text": message.content})
    for call in message.tool_calls:
        tool_use: JsonObject = {
            "type": "tool_use",
            "id": call.call_id,
            "name": call.name,
            "input": call.arguments,
        }
        # A validated caller cache hint forwards only on this wire, which
        # defines tool_use-block prompt caching natively.
        if call.cache_control is not None:
            tool_use["cache_control"] = call.cache_control
        blocks.append(tool_use)
    if blocks and all(
        block.get("type") == "text" and not str(block.get("text", "")).strip() for block in blocks
    ):
        # The wire rejects a text block that is empty ("text content blocks
        # must be non-empty") and a turn whose text is all whitespace ("must
        # contain non-whitespace text"), while it accepts an EMPTY assistant
        # content array in any position and whitespace text beside a
        # tool_use or another text block (all verified live 2026-09-05 on
        # fable-5-1 and sonnet-4-6). An assistant turn that carries only
        # empty or whitespace text therefore dispatches as an empty array:
        # the turn stays in place, so conversation structure is preserved,
        # and nothing the model could read is lost. A cache marker on the
        # dropped run is re-homed by the payload builder onto the closest
        # retained block of a neighboring turn.
        blocks = []
    return "assistant", blocks


ANTHROPIC_CONTEXT_MANAGEMENT_BETA = "context-management-2025-06-27"
"""Beta token Anthropic requires before it accepts ``context_management``."""

ANTHROPIC_DIAGNOSTICS_BETA = "cache-diagnosis-2026-04-07"
"""Beta token Anthropic requires before it accepts ``diagnostics``
(verified live 2026-08-30: the field alone is "Extra inputs are not
permitted"; with this token it is accepted)."""

ANTHROPIC_FAST_MODE_BETA = "fast-mode-2026-02-01"
"""Beta token Anthropic requires before it accepts ``speed``
(verified live 2026-08-30)."""

ANTHROPIC_FILES_API_BETA = "files-api-2025-04-14"
"""Beta token Anthropic requires before a ``file`` source resolves an uploaded file."""


def anthropic_request_headers(
    profile_headers: dict[str, str],
    request: GatewayRequest,
) -> dict[str, str]:
    """Return the per-request Anthropic headers for one dispatch.

    ``context_management``, ``diagnostics``, and ``speed`` are each served
    behind an ``anthropic-beta`` token (each verified live: the bare field
    is "Extra inputs are not permitted"), so their tokens join the
    connection's static headers exactly when the request carries the field.
    An Anthropic Files handle likewise needs the Files API token.
    Allowlisted caller-forwarded tokens (``request.provider_beta_tokens``,
    e.g. the 1M context window) merge the same way. The merged list keeps
    operator tokens first, then caller tokens, then field-required tokens,
    deduped in that order.

    Args:
        profile_headers: The connection's static wire headers.
        request: Canonical request about to be dispatched.

    Returns:
        Headers to send verbatim for this request.
    """
    headers = dict(profile_headers)
    required: list[str] = list(request.provider_beta_tokens)
    if request.context_management is not None:
        required.append(ANTHROPIC_CONTEXT_MANAGEMENT_BETA)
    if request.diagnostics is not None:
        required.append(ANTHROPIC_DIAGNOSTICS_BETA)
    if request.speed is not None:
        required.append(ANTHROPIC_FAST_MODE_BETA)
    if any(handle.provider == "anthropic" for handle in request.media_handles):
        required.append(ANTHROPIC_FILES_API_BETA)
    if not required:
        return headers
    existing = headers.get("anthropic-beta")
    tokens = [token for token in (existing.split(",") if existing else []) if token]
    for token in required:
        if token not in tokens:
            tokens.append(token)
    headers["anthropic-beta"] = ",".join(tokens)
    return headers


def openai_chat_message(
    message: GatewayMessage,
    *,
    reasoning_route_sha256: str | None = None,
    reasoning_output_exposed: bool = False,
) -> JsonObject:
    """Translate one gateway message to OpenAI Chat wire JSON.

    A canonical ``developer`` message is emitted as ``system`` on this wire.
    OpenAI's Chat Completions reference gives the two roles one definition
    ("Developer-provided instructions that the model should follow, regardless
    of messages sent by the user"), distinguished only by the model generation
    each was introduced for (``developer`` replaces ``system`` from o1 on, and
    those models accept ``system`` by converting it), so the fold is lossless
    and needs no disclosure. It applies to every provider on the Chat wire
    because this dialect never carries direct OpenAI traffic (that is the
    Responses dialect) and the third-party OpenAI-compatible servers behind
    it (Azure AI Foundry, DeepSeek, vLLM, and the like) enumerate the classic
    roles only and 400 on ``developer`` by name. The Responses builder keeps
    ``developer`` verbatim, since that wire defines the role.

    ``reasoning_route_sha256`` is the active preserved-thinking route identity
    for this rung (Fireworks or Hunyuan); an unsealed ``reasoning_content``
    block forwards to the provider only when it names that exact route.
    ``reasoning_output_exposed`` marks a rung whose plaintext reasoning the
    caller may replay verbatim (an ``exposed_reasoning_content`` block); any
    other rung omits that block, which route narrowing already disclosed.
    """
    if message.role == "tool":
        tool_payload: JsonObject = {
            "role": "tool",
            "content": message.folded_tool_error_content(),
            "tool_call_id": message.tool_call_id or "",
        }
        # Legacy tool-result name attribution round-trips on the OpenAI
        # wires: present stays present (the provider serves it) and absent
        # stays absent, so name-free histories keep their exact wire bytes.
        if message.provider_tool_name is not None:
            tool_payload["name"] = message.provider_tool_name
        return tool_payload
    payload: JsonObject = {
        "role": "system" if message.role == "developer" else message.role,
        "content": (
            [
                {"type": "text", "text": part.text}
                if part.kind == "text"
                else openai_chat_image_part(part)
                if part.kind == "image"
                else openai_chat_video_part(part)
                if part.kind == "video"
                else openai_chat_audio_part(part)
                if part.kind == "audio"
                else openai_chat_document_part(part)
                for part in message.content_parts
            ]
            if message.content_parts
            else message.content or ""
        ),
    }
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments_json()},
            }
            for call in message.tool_calls
        ]
    if message.provider_reasoning:
        if len(message.provider_reasoning) != 1:
            raise ProviderResponseError("Chat reasoning history requires exactly one carrier")
        block = message.provider_reasoning[0]
        if block.kind == "exposed_reasoning_content":
            if reasoning_output_exposed:
                payload["reasoning_content"] = block.content
            return payload
        if (
            block.kind != "reasoning_content"
            or reasoning_route_sha256 is None
            or block.route_sha256 != reasoning_route_sha256
        ):
            raise ProviderResponseError("reasoning carrier belongs to a different Chat route")
        payload["reasoning_content"] = block.content
    return payload


def add_openai_tools(
    payload: JsonObject,
    request: GatewayRequest,
    *,
    responses: bool,
) -> None:
    """Add Responses-native or Chat-native tools and tool choice in place."""
    if request.tools or request.provider_native_tools:
        if responses:
            declared: list[JsonObject] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    "strict": tool.strict,
                }
                for tool in request.tools
            ]
            if request.provider_native_tools:
                # Re-emit each verbatim declaration at its caller position;
                # decode assigns contiguous indexes over one tools array, so
                # the converted function tools exactly fill the gaps.
                natives = {entry.index: entry.tool for entry in request.provider_native_tools}
                functions = iter(declared)
                declared = [
                    natives[position] if position in natives else next(functions)
                    for position in range(len(declared) + len(natives))
                ]
            payload["tools"] = declared
        elif request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                        "strict": tool.strict,
                    },
                }
                for tool in request.tools
            ]
    if request.tool_choice is not None:
        if isinstance(request.tool_choice, GatewayNamedToolChoice):
            payload["tool_choice"] = (
                {"type": "function", "name": request.tool_choice.name}
                if responses
                else {
                    "type": "function",
                    "function": {"name": request.tool_choice.name},
                }
            )
        else:
            payload["tool_choice"] = request.tool_choice
