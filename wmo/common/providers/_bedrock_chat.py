"""Structured tool-calling translation for Bedrock Converse providers."""

from __future__ import annotations

import json
from typing import cast

from pydantic import JsonValue

from wmo.common.vendor.waterfall import ChatRequest, ChatResponse


def converse_request(request: ChatRequest, model: str) -> dict[str, object]:
    """Translate the provider-neutral structured contract to Bedrock Converse."""
    system: list[dict[str, str]] = []
    messages: list[dict[str, object]] = []
    has_tool_history = False

    def push(role: str, content: list[dict[str, object]]) -> None:
        if messages and messages[-1]["role"] == role:
            existing = cast("list[dict[str, object]]", messages[-1]["content"])
            existing.extend(content)
        else:
            messages.append({"role": role, "content": content})

    for message in request.messages:
        if message.role in ("system", "developer"):
            text = _chat_text(message.content)
            if text:
                system.append({"text": text})
            continue
        if message.role == "tool":
            has_tool_history = True
            push(
                "user",
                [
                    {
                        "toolResult": {
                            "toolUseId": message.tool_call_id or "",
                            "content": [{"text": _chat_text(message.content)}],
                        }
                    }
                ],
            )
            continue
        blocks: list[dict[str, object]] = []
        text = _chat_text(message.content)
        if text:
            blocks.append({"text": text})
        for tool_call in message.tool_calls or []:
            has_tool_history = True
            try:
                arguments = json.loads(tool_call.function.arguments)
            except ValueError:
                arguments = {}
            blocks.append(
                {
                    "toolUse": {
                        "toolUseId": tool_call.id,
                        "name": tool_call.function.name,
                        "input": arguments,
                    }
                }
            )
        if blocks:
            push("assistant" if message.role == "assistant" else "user", blocks)

    max_tokens = request.max_tokens or request.max_completion_tokens or 4096
    inference: dict[str, float | int] = {"maxTokens": max_tokens}
    if request.temperature is not None:
        inference["temperature"] = request.temperature
    result: dict[str, object] = {
        "modelId": model,
        "messages": messages,
        "inferenceConfig": inference,
    }
    if system:
        result["system"] = system
    choice = request.tool_choice
    if request.tools:
        tools = [
            {
                "toolSpec": {
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "inputSchema": {"json": tool.function.parameters},
                }
            }
            for tool in request.tools
        ]
        tool_config: dict[str, object] = {"tools": tools}
        if choice == "required":
            tool_config["toolChoice"] = {"any": {}}
        elif choice == "none" and has_tool_history:
            # Caller disabled tools for this turn but history has toolUse/toolResult
            # blocks. Bedrock requires toolConfig to be present; lock it to auto so
            # the model can interpret the history without being allowed to call new
            # tools (auto is the least-permissive mode Converse supports here).
            tool_config["toolChoice"] = {"auto": {}}
        elif isinstance(choice, dict):
            function = choice.get("function")
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                tool_config["toolChoice"] = {"tool": {"name": function["name"]}}
        if choice != "none" or has_tool_history:
            result["toolConfig"] = tool_config
    elif has_tool_history:
        # No tools on this turn but replayed history carries toolUse/toolResult
        # blocks. Bedrock rejects the request without toolConfig, so attach an
        # empty tools list in auto mode as the minimum-viable placeholder.
        result["toolConfig"] = {"tools": [], "toolChoice": {"auto": {}}}
    return result


def converse_response(raw: object, model: str) -> ChatResponse:
    """Translate a Bedrock Converse response to the structured provider contract."""
    response = cast("dict[str, object]", raw)
    output = cast("dict[str, object]", response["output"])
    message_data = cast("dict[str, object]", output["message"])
    blocks = cast("list[dict[str, object]]", message_data["content"])
    text = "".join(str(block["text"]) for block in blocks if "text" in block)
    tool_calls: list[dict[str, object]] = []
    for block in blocks:
        use = block.get("toolUse")
        if not isinstance(use, dict):
            continue
        tool_calls.append(
            {
                "id": str(use.get("toolUseId", "")),
                "type": "function",
                "function": {
                    "name": str(use.get("name", "")),
                    "arguments": json.dumps(use.get("input", {})),
                },
            }
        )
    stop_reason = str(response.get("stopReason", "end_turn"))
    finish_reason = {
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "content_filtered": "content_filter",
        "guardrail_intervened": "content_filter",
    }.get(stop_reason, "stop")
    message: dict[str, object] = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    usage = response.get("usage")
    usage_data = usage if isinstance(usage, dict) else {}
    # Converse reports `inputTokens` EXCLUDING the cached prefix, so the two
    # cache legs must be added back or every cached call under-reports its
    # input by the whole prefix (the `complete` path in bedrock.py does the
    # same normalization). The read leg rides the OpenAI-compatible
    # `prompt_tokens_details.cached_tokens` shape; the write leg rides the
    # Anthropic-style `cache_creation_input_tokens` field, so both price at
    # their cache tiers.
    cache_read = _token_count(usage_data.get("cacheReadInputTokens"))
    cache_write = _token_count(usage_data.get("cacheWriteInputTokens"))
    return ChatResponse.model_validate(
        {
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": _token_count(usage_data.get("inputTokens"))
                + cache_read
                + cache_write,
                "completion_tokens": _token_count(usage_data.get("outputTokens")),
                "prompt_tokens_details": {"cached_tokens": cache_read},
                # Raw top-level read count alongside, mirroring the direct
                # Anthropic translator, so both wire shapes carry the split.
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_write,
            },
        }
    )


def _token_count(value: object) -> int:
    """A Converse usage counter, or 0 for an absent/malformed one."""
    return value if isinstance(value, int) else 0


def _chat_text(content: JsonValue) -> str:
    """Flatten the text-bearing forms used by OpenAI-compatible chat messages."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return "" if content is None else str(content)
