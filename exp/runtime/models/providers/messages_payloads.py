"""Native streaming payload builders for the Messages-family dialects.

Split from ``streaming_requests`` for the module line budget: the Anthropic,
Gemini, and Bedrock builders live here; ``dialect_stream_payload`` in
``dialect_dispatch`` remains the single dispatch seam.
"""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import (
    GatewayNamedToolChoice,
    GatewayRequest,
)
from exp.runtime.models.providers.anthropic_tool_compat import (
    anthropic_rejects_forced_tool_choice,
    anthropic_strict_schema_unsupported,
)
from exp.runtime.models.providers.bedrock_requests import converse_body
from exp.runtime.models.providers.errors import (
    ProviderCapabilityError,
    ProviderParameterError,
    ProviderResponseError,
)
from exp.runtime.models.providers.gemini_requests import gemini_generate_request
from exp.runtime.models.providers.reasoning_compat import (
    anthropic_budgeted_enabled_only,
    anthropic_reasoning_effort,
    anthropic_thinking_budget_tokens,
)
from exp.runtime.models.providers.wire_messages import (
    anthropic_blocks,
    retained_cache_marked_blocks,
)
from exp.runtime.openai_protocol.model_adapter import model_request as gateway_model_request

JSON_OBJECT_SYSTEM_INSTRUCTION = (
    "Output format: the caller will pass your entire reply to a strict JSON "
    "parser, so it must be exactly one raw JSON object. Begin the reply with "
    "'{' as the very first character and end it with '}' as the very last. "
    "Do not begin with ```json or any code fence, do not use markdown, and do "
    "not add any words before or after the object. A reply that starts with "
    "anything other than '{' is a failure."
)
"""System instruction that carries ``json_object_output`` on wires with no native JSON mode."""


def anthropic_messages_stream_payload(
    model_id: str,
    request: GatewayRequest,
    *,
    supports_temperature: bool = True,
    supports_top_p: bool = True,
    supports_top_k: bool = False,
    supports_logprobs: bool = False,
    supports_reasoning: bool = False,
    reasoning_effort: str | None = None,
) -> JsonObject:
    """Translate one canonical request to native streaming Messages JSON.

    Args:
        model_id: Exact Anthropic model identifier.
        request: Canonical gateway request.

    Returns:
        Native Messages request with streaming enabled.

    Raises:
        ProviderCapabilityError: A ``strict`` tool schema uses a keyword the
            provider's strict validator rejects (``strict_tools``), or the
            request forces a tool the model or its thinking mode cannot force
            (``forced_tool_choice``); both let route admission prefer a rung
            that honors the request and otherwise coerce with disclosure.
        ProviderResponseError: Instruction or message content is malformed.
    """
    # Anthropic Messages has no compatible logprob request/response surface in
    # this adapter. Keep the shared route signature for capability plumbing,
    # but never put an OpenAI-shaped field on the Anthropic wire.
    del supports_logprobs
    system_parts: list[tuple[str, tuple[JsonObject, ...]]] = []
    messages: list[JsonObject] = []
    displaced_marker: JsonObject | None = None
    for message in request.messages:
        if message.role in {"system", "developer"}:
            if message.content is None:
                raise ProviderResponseError("instruction messages require text")
            # Leading instructions ride the top-level system field; a system
            # turn after conversation began is a first-class mid-conversation
            # message on this wire (the provider enforces its own placement
            # rules), so its position is preserved verbatim.
            if messages:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            list(message.provider_text_blocks)
                            if message.provider_text_blocks
                            else [{"type": "text", "text": message.content}]
                        ),
                    }
                )
            else:
                system_parts.append((message.content, message.provider_text_blocks))
            continue
        role, blocks = anthropic_blocks(message)
        if not blocks:
            # An assistant turn with no readable text dispatches as an empty
            # array (accepted live) and so has no block of its own to carry a
            # caller cache marker. The breakpoint migrates across the turn
            # boundary by the same rule the block-run helper applies within a
            # turn: onto the closest retained block before it (the empty turn
            # adds no readable bytes, so the cached prefix is the same one),
            # else onto the first retained block after it.
            marker = _cache_marker(message.provider_text_blocks)
            if marker is not None and not _mark_last_block(messages, marker):
                displaced_marker = marker
        elif displaced_marker is not None:
            blocks = _mark_first_block(blocks, displaced_marker)
            displaced_marker = None
        if messages and messages[-1].get("role") == role:
            existing = messages[-1].get("content")
            if not isinstance(existing, list):
                raise ProviderResponseError("Anthropic message content is malformed")
            existing.extend(blocks)
        else:
            messages.append({"role": role, "content": blocks})
    if request.json_object_output:
        # Anthropic has no schema-free JSON mode, so the caller's intent rides
        # the system prompt as a trailing instruction.
        system_parts.append((JSON_OBJECT_SYSTEM_INSTRUCTION, ()))
    payload: JsonObject = {
        "model": model_id,
        "messages": messages,
        "max_tokens": request.maximum_output_tokens or 4096,
        "stream": True,
    }
    if system_parts:
        if any(blocks for _, blocks in system_parts):
            # A cache-marked system prompt re-emits the caller's blocks with
            # their markers: block-level markers are the only way the
            # provider caches the prompt. A marker must change cost and
            # nothing else, so the emitted text equals the unmarked joined
            # string byte-for-byte. The provider rejects whitespace-only
            # text blocks (verified live 2026-09-01), so each canonical
            # blank-line separator is folded into the FOLLOWING block's text
            # (between parts, and between a part's blocks when its canonical
            # content joined them that way); markers stay on their blocks
            # and the first block stays byte-exact. Markerless requests keep
            # the joined string so their payloads stay byte-identical.
            system_blocks: list[JsonObject] = []

            def emit(block: JsonObject, *, separated: bool) -> None:
                """Append one block, folding in a leading separator if due."""
                if separated:
                    block = {**block, "text": "\n\n" + str(block.get("text", ""))}
                system_blocks.append(block)

            for content, blocks in system_parts:
                part_leads = bool(system_blocks)
                if not blocks:
                    emit({"type": "text", "text": content}, separated=part_leads)
                    continue
                adjacent = "".join(str(block.get("text", "")) for block in blocks)
                inner_separated = adjacent != content
                for position, block in enumerate(blocks):
                    emit(
                        dict(block),
                        separated=(part_leads if position == 0 else inner_separated),
                    )
            # The wire rejects an empty system text block anywhere and a
            # system prompt whose text is all whitespace ("system: text
            # content blocks must be non-empty" / "must contain non-whitespace
            # text", verified live 2026-09-05). Empty blocks drop with their
            # breakpoints migrated; a prompt left with no readable text is
            # omitted, since an absent system field is what it says.
            retained_system = retained_cache_marked_blocks(system_blocks)
            if any(str(block.get("text", "")).strip() for block in retained_system):
                payload["system"] = retained_system
        else:
            joined_system = "\n\n".join(content for content, _ in system_parts)
            if joined_system.strip():
                payload["system"] = joined_system
    if request.tools:
        tools: list[JsonObject] = []
        for tool in request.tools:
            translated: JsonObject = {
                "name": tool.name,
                "input_schema": tool.parameters,
            }
            # Anthropic rejects an explicit null description ("Input should
            # be a valid string"), so an absent description stays absent.
            if tool.description is not None:
                translated["description"] = tool.description
            if tool.strict:
                # The strict validator compiles the schema into a grammar and
                # 400s by name on keywords it cannot express (verified live
                # 2026-09-05: ``maxItems`` on every current model). Declining
                # here keeps the schema intact and lets admission drop only
                # ``strict`` when no rung can honor it.
                if anthropic_strict_schema_unsupported(tool.parameters) is not None:
                    raise ProviderCapabilityError(capability="strict_tools")
                translated["strict"] = True
            # Anthropic-native tool annotations forward verbatim on this
            # wire only; the provider owns their validity rules. An absent
            # value stays absent so provider defaults keep applying.
            if tool.cache_control is not None:
                translated["cache_control"] = tool.cache_control
            if tool.eager_input_streaming is not None:
                translated["eager_input_streaming"] = tool.eager_input_streaming
            if tool.defer_loading is not None:
                translated["defer_loading"] = tool.defer_loading
            if tool.allowed_callers is not None:
                translated["allowed_callers"] = list(tool.allowed_callers)
            if tool.input_examples is not None:
                translated["input_examples"] = list(tool.input_examples)
            tools.append(translated)
        payload["tools"] = tools
    if request.provider_server_tools:
        # Server tools re-emit verbatim after the converted custom tools (an
        # accepted ordering deviation); route admission guarantees this
        # dispatch is an Anthropic rung, which owns their validity rules.
        server_entries = [dict(entry) for entry in request.provider_server_tools]
        existing_tools = payload.get("tools")
        if isinstance(existing_tools, list):
            existing_tools.extend(server_entries)
        else:
            payload["tools"] = server_entries
    tool_choice: JsonObject | None = None
    if request.tool_choice is not None:
        if isinstance(request.tool_choice, GatewayNamedToolChoice):
            tool_choice = {"type": "tool", "name": request.tool_choice.name}
        else:
            mapping = {"auto": "auto", "none": "none", "required": "any"}
            tool_choice = {"type": mapping[request.tool_choice]}
    if request.parallel_tool_calls is not None:
        tool_choice = tool_choice or {"type": "auto"}
        tool_choice["disable_parallel_tool_use"] = not request.parallel_tool_calls
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if request.temperature is not None and supports_temperature:
        payload["temperature"] = request.temperature
    if request.top_p is not None and supports_top_p:
        payload["top_p"] = request.top_p
    if request.top_k is not None and supports_top_k:
        payload["top_k"] = request.top_k
    effective_reasoning_effort = request.reasoning_effort or reasoning_effort
    # The caller's verbatim output_config seeds the object; engine-derived
    # keys only fill gaps, so the caller always wins byte-for-byte. A
    # canonical caller effort also rides request.reasoning_effort (decode
    # maps it), which keeps the engine's effective effort equal to the
    # caller's and structurally removes the two-sources fight.
    output_config: JsonObject = (
        dict(request.provider_output_config) if request.provider_output_config is not None else {}
    )
    if request.context_management is not None:
        # Anthropic-native context editing forwards byte-for-byte; the
        # required beta header joins the dispatch via
        # anthropic_request_headers.
        payload["context_management"] = request.context_management
    if request.diagnostics is not None:
        # Same treatment: verbatim object, beta header via
        # anthropic_request_headers.
        payload["diagnostics"] = request.diagnostics
    if request.speed is not None:
        payload["speed"] = request.speed
    if request.provider_cache_control is not None:
        # The top-level automatic caching marker forwards byte-for-byte; the
        # provider accepts it bare (verified live 2026-08-30).
        payload["cache_control"] = request.provider_cache_control
    if request.inference_geo is not None:
        payload["inference_geo"] = request.inference_geo
    # A budgeted-enabled-only model (haiku-4-5) rejects ``thinking.type:
    # adaptive`` and ``output_config.effort`` by NAME: its reasoning dial is a
    # token budget, not the effort ladder. An effort reaching this seam is the
    # caller's depth intent, so it is realized as a derived budget — the same
    # wire realization the effort generation gets via the adaptive object.
    budgeted_only = supports_reasoning and anthropic_budgeted_enabled_only(model_id)
    if budgeted_only and "effort" in output_config and request.reasoning_effort is not None:
        # A recognized caller effort rides request.reasoning_effort too (decode
        # maps it) and is realized as the token budget below, so the by-name-
        # rejected output_config key comes off the wire. An UNRECOGNIZED effort
        # never mapped, has no budget realization, and stays verbatim — the
        # provider's own by-name rejection is the honest outcome, never a
        # silent thinking-off answer.
        output_config.pop("effort")
    if request.provider_thinking_config is not None:
        # The caller's exact thinking configuration wins over the catalog's
        # adaptive default and travels verbatim, so budget semantics are
        # never reinterpreted by the gateway. An adaptive config (caller-sent
        # or route-translated) still composes with the route's pinned effort,
        # exactly like a request that carried no thinking config.
        payload["thinking"] = request.provider_thinking_config
        if (
            request.provider_thinking_config.get("type") == "adaptive"
            and supports_reasoning
            and not budgeted_only
            and effective_reasoning_effort is not None
            and "effort" not in output_config
        ):
            output_config["effort"] = anthropic_reasoning_effort(
                model_id, effective_reasoning_effort
            )
    elif budgeted_only and effective_reasoning_effort not in (None, "none"):
        # No legal budget under the output ceiling means thinking stays off;
        # the model still answers, and route narrowing already disclosed any
        # sampling interplay. output_config.effort is never emitted here.
        budget = anthropic_thinking_budget_tokens(request.maximum_output_tokens)
        if budget is not None:
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
    elif supports_reasoning and not budgeted_only and effective_reasoning_effort is not None:
        payload["thinking"] = {"type": "adaptive"}
        if "effort" not in output_config:
            output_config["effort"] = anthropic_reasoning_effort(
                model_id, effective_reasoning_effort
            )
    if request.structured_text is not None and "format" not in output_config:
        output_config["format"] = {
            "type": "json_schema",
            "schema": request.structured_text.json_schema,
        }
    if output_config:
        payload["output_config"] = output_config
    if request.stop:
        payload["stop_sequences"] = list(request.stop)
    _require_forced_tool_choice_support(model_id, request, payload)
    return payload


_UNMARKABLE_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})
"""Content block types the wire refuses to carry a ``cache_control`` marker on."""


def _cache_marker(blocks: tuple[JsonObject, ...]) -> JsonObject | None:
    """Return the first caller cache marker in a block run, if any."""
    for block in blocks:
        marker = block.get("cache_control")
        if isinstance(marker, dict):
            return marker
    return None


def _mark_last_block(messages: list[JsonObject], marker: JsonObject) -> bool:
    """Attach ``marker`` to the closest earlier emitted block that can carry one.

    Collapsed (empty) turns are skipped on the way back: consecutive empty
    assistant turns all sit on the same cache boundary, so every marker they
    carried collapses onto the same retained block instead of deferring to a
    later one, which would cache a larger prefix than the caller asked for.

    Returns:
        Whether a block took the marker; a block already marked counts, since
        one marker per boundary suffices.
    """
    for emitted in reversed(messages):
        content = emitted.get("content")
        if not isinstance(content, list):
            return False
        if not content:
            continue
        last = content[-1]
        if not isinstance(last, dict) or last.get("type") in _UNMARKABLE_BLOCK_TYPES:
            return False
        if "cache_control" not in last:
            content[-1] = {**last, "cache_control": marker}
        return True
    return False


def _mark_first_block(blocks: list[JsonObject], marker: JsonObject) -> list[JsonObject]:
    """Return ``blocks`` with ``marker`` on the first block that can carry one."""
    for index, block in enumerate(blocks):
        if block.get("type") in _UNMARKABLE_BLOCK_TYPES:
            continue
        if "cache_control" not in block:
            return [*blocks[:index], {**block, "cache_control": marker}, *blocks[index + 1 :]]
        return blocks
    return blocks


def _require_forced_tool_choice_support(
    model_id: str,
    request: GatewayRequest,
    payload: JsonObject,
) -> None:
    """Decline a forced ``tool_choice`` this rung is known to reject.

    Two provider rules apply (both verified live 2026-09-05). Fable 5.1 and
    Mythos 5.1 answer ``any`` and ``tool`` with a 400 on every request, even
    with no thinking config. Every model rejects a forced choice beside a
    budgeted ``thinking: enabled`` config ("Thinking may not be enabled when
    tool_choice forces tool use"), whether the caller sent that config or the
    rung derived it from an effort; adaptive thinking carries a forced choice
    fine. ``auto`` and ``none`` are never affected.

    Raises:
        ProviderCapabilityError: ``forced_tool_choice`` when the built payload
            would be rejected.
    """
    forced = request.tool_choice == "required" or isinstance(
        request.tool_choice, GatewayNamedToolChoice
    )
    if not forced:
        return
    thinking = payload.get("thinking")
    budgeted_thinking = isinstance(thinking, dict) and thinking.get("type") == "enabled"
    if budgeted_thinking or anthropic_rejects_forced_tool_choice(model_id):
        raise ProviderCapabilityError(capability="forced_tool_choice")


def gemini_generate_content_stream_payload(
    model_id: str,
    request: GatewayRequest,
    *,
    supports_temperature: bool = True,
    supports_top_p: bool = True,
    supports_top_k: bool = False,
    supports_logprobs: bool = False,
    supports_reasoning: bool = False,
    reasoning_effort: str | None = None,
) -> JsonObject:
    """Translate one canonical request to the native streamGenerateContent JSON.

    The payload is built by the exact converter the Gemini provider client
    uses (canonical request through the shared model adapter, then the native
    generateContent builder), so both engines send one identical body. Gemini
    streaming needs no body-level stream flag: streaming is selected by the
    ``streamGenerateContent`` route in the wire profile URL.

    Args:
        model_id: Exact Gemini model identifier; travels in the route path.
        request: Canonical gateway request.

    Returns:
        Native generation request for the SSE streaming route.

    Raises:
        ProviderCapabilityError: The request carries an image URL this wire
            cannot fetch, so route selection can narrow past this rung.
        ProviderResponseError: A message cannot preserve its tool linkage on
            Gemini's wire.
    """
    try:
        return gemini_generate_request(
            model_id,
            gateway_model_request(request),
            supports_temperature=supports_temperature,
            supports_top_p=supports_top_p,
            supports_top_k=supports_top_k,
            supports_logprobs=supports_logprobs,
            supports_reasoning=supports_reasoning,
            reasoning_effort=reasoning_effort,
            stop_sequences=request.stop,
            response_json_schema=(
                request.structured_text.json_schema if request.structured_text is not None else None
            ),
            json_object_output=request.json_object_output,
        )
    except (ProviderParameterError, ProviderCapabilityError):
        raise
    except ValueError as exc:
        raise ProviderResponseError(str(exc)) from exc


def bedrock_converse_stream_payload(
    model_id: str,
    request: GatewayRequest,
    *,
    supports_temperature: bool = True,
    supports_top_p: bool = True,
    supports_top_k: bool = False,
    supports_logprobs: bool = False,
) -> JsonObject:
    """Translate one canonical request to the native ConverseStream REST body.

    The body is built by the exact converter the Bedrock provider client
    uses (canonical request through the shared model adapter, then the shared
    Converse body builder), so both engines send one identical document. On
    the REST route the model travels in the URL path, never the body, and
    streaming is selected by the ``converse-stream`` route itself.

    Args:
        model_id: Exact Bedrock model or inference-profile identifier; it
            travels in the wire profile URL and keeps the dispatch signature.
        request: Canonical gateway request.

    Returns:
        Native Converse request body for the streaming REST route.

    Raises:
        ProviderCapabilityError: The request carries an image URL this wire
            cannot fetch, so route selection can narrow past this rung.
        ProviderParameterError: Inline media exceeds the Converse payload
            ceiling, so route selection can narrow past this rung.
        ProviderResponseError: A message cannot be represented without
            dropping tool context.
    """
    del model_id
    try:
        return converse_body(
            gateway_model_request(request),
            supports_temperature=supports_temperature,
            supports_top_p=supports_top_p,
            supports_top_k=supports_top_k,
            supports_logprobs=supports_logprobs,
            stop_sequences=request.stop,
            structured_output_name=(
                request.structured_text.name if request.structured_text is not None else None
            ),
            structured_output_description=(
                request.structured_text.description if request.structured_text is not None else None
            ),
            structured_output_schema=(
                request.structured_text.json_schema if request.structured_text is not None else None
            ),
            strict_tool_names=tuple(tool.name for tool in request.tools if tool.strict),
            json_object_instruction=(
                JSON_OBJECT_SYSTEM_INSTRUCTION if request.json_object_output else None
            ),
        )
    except (ProviderParameterError, ProviderCapabilityError):
        raise
    except ValueError as exc:
        raise ProviderResponseError(str(exc)) from exc
