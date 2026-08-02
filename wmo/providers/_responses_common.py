"""Shared structured request and response translation for OpenAI's Responses API.

The agent runtime speaks the OpenAI-compatible Chat Completions shape because that is pi's
transport contract. Responses is a different wire protocol: assistant tool calls and tool
results are top-level input items, tool definitions are flat, and output tokens use a different
field name. This module is the single stateless bridge between those contracts.
"""

from __future__ import annotations

import json
from typing import Any, cast

from llm_waterfall import ChatRequest, ChatResponse
from pydantic import JsonValue


def responses_request(
    request: ChatRequest,
    model: str,
    *,
    reasoning_effort: str | None = None,
    service_tier: str | None = None,
    allow_sampling: bool = True,
) -> dict[str, object]:
    """Translate one structured chat request into a native Responses API payload.

    Args:
        request: Validated OpenAI-compatible request emitted by the agent runtime.
        model: Provider-controlled model or deployment name.
        reasoning_effort: Provider-controlled reasoning effort. Caller request extras cannot
            override this value.
        service_tier: Provider-controlled service tier. Caller request extras cannot override
            this value.
        allow_sampling: Whether compatible non-reasoning models may receive ``temperature`` and
            ``top_p`` from the agent request.

    Returns:
        Keyword arguments suitable for ``client.responses.create``.

    Raises:
        ValueError: If the request contains a non-text message or malformed structured option.
    """
    payload: dict[str, object] = {
        "model": model,
        "input": _responses_input(request),
        # pi asks its OpenAI-compatible endpoint for a stream. The Python runtime needs one
        # complete Response object, so the provider boundary deliberately terminates streaming.
        "stream": False,
        # Stateless replay is the harness contract and avoids accumulating provider-side state.
        "store": _boolean_extra(request, "store", default=False),
        # The adapter reconstructs every turn from chat history instead of using a prior response
        # id. Request the opaque reasoning payload even when the caller leaves effort at the model
        # default, since reasoning models still need that item replayed before their tool calls.
        "include": ["reasoning.encrypted_content"],
    }

    max_output_tokens = (
        request.max_completion_tokens
        if request.max_completion_tokens is not None
        else request.max_tokens
    )
    if max_output_tokens is not None:
        payload["max_output_tokens"] = max_output_tokens

    parallel_tool_calls = _optional_boolean_extra(request, "parallel_tool_calls")
    if parallel_tool_calls is not None:
        payload["parallel_tool_calls"] = parallel_tool_calls

    if request.tool_choice is not None:
        payload["tool_choice"] = _responses_tool_choice(request.tool_choice)

    if request.tools:
        payload["tools"] = [
            {
                "type": "function",
                "name": tool.function.name,
                "description": tool.function.description,
                "parameters": tool.function.parameters,
                **({"strict": tool.function.strict} if tool.function.strict is not None else {}),
            }
            for tool in request.tools
        ]

    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}
    elif allow_sampling:
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        top_p = _optional_number_extra(request, "top_p")
        if top_p is not None:
            payload["top_p"] = top_p

    if service_tier is not None:
        payload["service_tier"] = service_tier
    return payload


def responses_response(raw: dict[str, object]) -> ChatResponse:
    """Translate one native Responses object into the structured chat response contract.

    Args:
        raw: JSON-mode dump of an OpenAI SDK ``Response``.

    Returns:
        A single-choice response consumable by the pi OpenAI-compatible bridge.

    Raises:
        ValueError: If a failed response or malformed function call cannot be represented safely.
    """
    _require_completed_response(raw)

    text_parts: list[str] = []
    tool_calls: list[dict[str, object]] = []
    reasoning_details: list[dict[str, str]] = []
    pending_reasoning: list[dict[str, object]] = []
    recognized_output = False
    output = raw.get("output")
    if not isinstance(output, list):
        raise ValueError("Responses API completed without an output array")
    for item_value in output:
        item = _object_dict(item_value)
        if item is None:
            raise ValueError("Responses API output item must be an object")
        _require_completed_output_item(item)
        item_type = item.get("type")
        if item_type == "reasoning":
            pending_reasoning.append(item)
        elif item_type == "message":
            recognized_output = True
            _append_message_text(item, text_parts)
        elif item_type == "function_call":
            recognized_output = True
            tool_call = _chat_tool_call(item)
            call_id = cast("str", tool_call["id"])
            if pending_reasoning:
                reasoning_details.append(_encrypted_reasoning_detail(pending_reasoning, call_id))
            pending_reasoning.clear()
            tool_calls.append(tool_call)
        else:
            raise ValueError(f"unsupported Responses output item type {item_type!r}")
    if not recognized_output:
        raise ValueError("Responses API completed without a message or function call")

    message: dict[str, object] = {
        "role": "assistant",
        "content": "".join(text_parts),
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning_details:
        # Pi's OpenAI-completions parser stores each opaque detail on the matching tool call's
        # thoughtSignature and re-emits it in the next request's assistant message.
        message["reasoning_details"] = reasoning_details

    response: dict[str, object] = {
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": _finish_reason(raw, has_tool_calls=bool(tool_calls)),
            }
        ]
    }
    model = raw.get("model")
    if isinstance(model, str):
        response["model"] = model

    usage = _object_dict(raw.get("usage"))
    if usage is not None:
        input_details = _object_dict(usage.get("input_tokens_details")) or {}
        output_details = _object_dict(usage.get("output_tokens_details")) or {}
        response["usage"] = {
            "prompt_tokens": _usage_count(usage.get("input_tokens")),
            "completion_tokens": _usage_count(usage.get("output_tokens")),
            "prompt_tokens_details": {
                "cached_tokens": _usage_count(input_details.get("cached_tokens"))
            },
            "completion_tokens_details": {
                "reasoning_tokens": _usage_count(output_details.get("reasoning_tokens"))
            },
        }

    service_tier = raw.get("service_tier")
    if isinstance(service_tier, str):
        response["service_tier"] = service_tier
    return ChatResponse.model_validate(response)


def complete_chat(
    responses: object,
    model: str,
    request: ChatRequest,
    *,
    reasoning_effort: str | None = None,
    service_tier: str | None = None,
    allow_sampling: bool = True,
) -> ChatResponse:
    """Run a structured chat turn against an OpenAI SDK Responses resource.

    Args:
        responses: ``client.responses`` from either OpenAI or an OpenAI-compatible client.
        model: Provider-controlled model or deployment name.
        request: Validated structured request from the agent runtime.
        reasoning_effort: Provider-controlled reasoning effort.
        service_tier: Provider-controlled service tier.
        allow_sampling: Whether compatible non-reasoning models may receive sampling fields.

    Returns:
        Provider-neutral structured chat response.
    """
    # OpenAI models create() as a broad TypedDict union whose exact surface changes between SDK
    # releases. The payload is validated by the narrow translation above, so keep that churn at
    # this one SDK boundary instead of leaking Any through the runtime contract.
    resource = cast("Any", responses)
    sdk_response = resource.create(
        **responses_request(
            request,
            model,
            reasoning_effort=reasoning_effort,
            service_tier=service_tier,
            allow_sampling=allow_sampling,
        )
    )
    raw = cast("dict[str, object]", sdk_response.model_dump(mode="json"))
    return responses_response(raw)


def _responses_input(request: ChatRequest) -> list[dict[str, object]]:
    """Translate ordered chat history into stateless Responses input items."""
    items: list[dict[str, object]] = []
    for message in request.messages:
        if message.role == "tool":
            if message.tool_call_id is None:
                raise ValueError("Responses tool result requires tool_call_id")
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": _chat_text(message.content) or "",
                }
            )
            continue

        text = _chat_text(message.content)
        if text is not None:
            items.append({"role": message.role, "content": text})

        if message.role != "assistant" and message.tool_calls:
            raise ValueError("Responses function calls must belong to an assistant message")
        reasoning_by_call = _reasoning_items_by_call(message)
        replayed_reasoning_ids: set[str] = set()
        for tool_call in message.tool_calls or []:
            for reasoning in reasoning_by_call.get(tool_call.id, []):
                reasoning_id = cast("str", reasoning["id"])
                if reasoning_id not in replayed_reasoning_ids:
                    items.append(reasoning)
                    replayed_reasoning_ids.add(reasoning_id)
            items.append(
                {
                    "type": "function_call",
                    "call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                }
            )
    return items


def _reasoning_items_by_call(message: object) -> dict[str, list[dict[str, object]]]:
    """Decode Pi's opaque thought signatures into validated Responses reasoning items."""
    extras = getattr(message, "model_extra", None)
    if not isinstance(extras, dict) or extras.get("reasoning_details") is None:
        return {}
    details = extras["reasoning_details"]
    if not isinstance(details, list):
        raise ValueError("Responses reasoning_details must be an array")
    calls = getattr(message, "tool_calls", None) or []
    call_ids = {call.id for call in calls}
    by_call: dict[str, list[dict[str, object]]] = {}
    for detail_value in details:
        detail = _object_dict(detail_value)
        if detail is None:
            raise ValueError("Responses reasoning detail must be an object")
        if detail.get("type") != "reasoning.encrypted":
            continue
        call_id = detail.get("id")
        data = detail.get("data")
        if not isinstance(call_id, str) or call_id not in call_ids:
            raise ValueError("encrypted Responses reasoning detail has no matching tool call")
        if not isinstance(data, str):
            raise ValueError("encrypted Responses reasoning detail data must be JSON text")
        try:
            decoded = json.loads(data)
        except json.JSONDecodeError as error:
            raise ValueError(
                "encrypted Responses reasoning detail contains invalid JSON"
            ) from error
        decoded_items = decoded if isinstance(decoded, list) else [decoded]
        if not decoded_items:
            raise ValueError("encrypted Responses reasoning detail contains no reasoning items")
        by_call.setdefault(call_id, []).extend(
            _validated_reasoning_item(item) for item in decoded_items
        )
    return by_call


def _chat_text(content: JsonValue) -> str | None:
    """Flatten the text-only content forms emitted by OpenAI-compatible agent SDKs."""
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block_value in content:
            block = _object_dict(block_value)
            if block is None or not isinstance(block.get("text"), str):
                raise ValueError("Responses adapter supports text chat content only")
            parts.append(cast("str", block["text"]))
        return "".join(parts)
    raise ValueError("Responses adapter supports text chat content only")


def _responses_tool_choice(choice: JsonValue) -> object:
    """Translate Chat Completions tool choice into the Responses API shape."""
    if isinstance(choice, str):
        if choice not in ("auto", "none", "required"):
            raise ValueError(f"unsupported Responses tool_choice {choice!r}")
        return choice
    choice_dict = _object_dict(choice)
    if choice_dict is None or choice_dict.get("type") != "function":
        raise ValueError("Responses tool_choice must name a function or use auto/none/required")
    function = _object_dict(choice_dict.get("function"))
    name = function.get("name") if function is not None else choice_dict.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Responses function tool_choice requires a name")
    return {"type": "function", "name": name}


def _append_message_text(item: dict[str, object], parts: list[str]) -> None:
    """Append native output-text blocks from one Responses message item."""
    content = item.get("content")
    if not isinstance(content, list):
        raise ValueError("Responses message output content must be an array")
    for block_value in content:
        block = _object_dict(block_value)
        if block is None:
            raise ValueError("Responses message content block must be an object")
        if block.get("type") == "output_text" and isinstance(block.get("text"), str):
            parts.append(cast("str", block["text"]))
        elif block.get("type") == "refusal" and isinstance(block.get("refusal"), str):
            parts.append(cast("str", block["refusal"]))
        else:
            raise ValueError(f"unsupported Responses message block type {block.get('type')!r}")


def _chat_tool_call(item: dict[str, object]) -> dict[str, object]:
    """Map one native Responses function call while preserving its exact call id."""
    call_id = item.get("call_id")
    name = item.get("name")
    arguments = item.get("arguments")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("Responses function_call is missing call_id")
    if not isinstance(name, str) or not name:
        raise ValueError("Responses function_call is missing name")
    if not isinstance(arguments, str):
        raise ValueError("Responses function_call arguments must be a JSON string")
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _validated_reasoning_item(value: object) -> dict[str, object]:
    """Return the safe stateless subset of one encrypted Responses reasoning item."""
    item = _object_dict(value)
    if item is None or item.get("type") != "reasoning":
        raise ValueError("encrypted Responses reasoning data is not a reasoning item")
    item_id = item.get("id")
    encrypted_content = item.get("encrypted_content")
    summary = item.get("summary", [])
    if not isinstance(item_id, str) or not item_id:
        raise ValueError("encrypted Responses reasoning item is missing id")
    if not isinstance(encrypted_content, str) or not encrypted_content:
        raise ValueError("Responses reasoning item is missing encrypted_content")
    if not isinstance(summary, list):
        raise ValueError("Responses reasoning item summary must be an array")
    return {
        "type": "reasoning",
        "id": item_id,
        "summary": summary,
        "encrypted_content": encrypted_content,
    }


def _encrypted_reasoning_detail(items: list[dict[str, object]], call_id: str) -> dict[str, str]:
    """Pack ordered native reasoning items into Pi's one thoughtSignature per tool call."""
    validated = [_validated_reasoning_item(item) for item in items]
    return {
        "type": "reasoning.encrypted",
        "id": call_id,
        "data": json.dumps(validated, separators=(",", ":"), sort_keys=True),
    }


def _require_completed_response(raw: dict[str, object]) -> None:
    """Reject every non-completed top-level status before exposing partial tool calls."""
    status = raw.get("status")
    if status == "completed":
        return
    if status == "incomplete":
        details = _object_dict(raw.get("incomplete_details"))
        reason = details.get("reason") if details is not None else None
        raise ValueError(
            f"Responses API returned incomplete response: {reason or 'unknown reason'}"
        )
    error = _object_dict(raw.get("error"))
    diagnostics: list[str] = []
    if error is not None:
        code = error.get("code")
        message = error.get("message")
        if isinstance(code, str) and code:
            diagnostics.append(f"code={code}")
        if isinstance(message, str) and message:
            diagnostics.append(f"message={message}")
    suffix = f" ({', '.join(diagnostics)})" if diagnostics else ""
    raise ValueError(f"Responses API returned non-completed status {status!r}{suffix}")


def _require_completed_output_item(item: dict[str, object]) -> None:
    """Reject partial output items even on an otherwise malformed completed response."""
    status = item.get("status")
    if status not in (None, "completed"):
        raise ValueError(
            f"Responses API returned {item.get('type')!r} output with status {status!r}"
        )


def _finish_reason(raw: dict[str, object], *, has_tool_calls: bool) -> str:
    """Map Responses status metadata to the OpenAI-compatible finish reason."""
    del raw
    return "tool_calls" if has_tool_calls else "stop"


def _object_dict(value: object) -> dict[str, object] | None:
    """Narrow a JSON object without imposing mutable mapping APIs on callers."""
    return cast("dict[str, object]", value) if isinstance(value, dict) else None


def _usage_count(value: object) -> int:
    """Read a non-boolean integer usage counter, defaulting absent values to zero."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _boolean_extra(request: ChatRequest, name: str, *, default: bool) -> bool:
    """Read and validate one boolean extra from the forward-compatible request surface."""
    value = (request.model_extra or {}).get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"Responses {name} must be a boolean")
    return value


def _optional_boolean_extra(request: ChatRequest, name: str) -> bool | None:
    """Read one optional boolean request extra."""
    extras = request.model_extra or {}
    if name not in extras or extras[name] is None:
        return None
    return _boolean_extra(request, name, default=False)


def _optional_number_extra(request: ChatRequest, name: str) -> float | int | None:
    """Read one optional non-boolean numeric request extra."""
    value = (request.model_extra or {}).get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Responses {name} must be numeric")
    return value
