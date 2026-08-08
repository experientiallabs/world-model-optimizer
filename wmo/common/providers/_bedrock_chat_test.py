"""Tests for structured Bedrock Converse translation."""

from typing import cast

import pytest

from wmo.common.providers._bedrock_chat import converse_request, converse_response
from wmo.common.vendor.waterfall import ChatRequest


def test_converse_round_trip_preserves_tools_results_and_usage() -> None:
    request = ChatRequest.model_validate(
        {
            "messages": [
                {"role": "system", "content": "use tools"},
                {"role": "user", "content": "list files"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": '{"command":"ls"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "a.txt"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "run a command",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "tool_choice": "required",
            "max_completion_tokens": 1024,
        }
    )

    wire = converse_request(request, "model-id")

    assert wire["modelId"] == "model-id"
    assert wire["inferenceConfig"] == {"maxTokens": 1024}
    assert wire["system"] == [{"text": "use tools"}]
    tool_config = wire["toolConfig"]
    assert isinstance(tool_config, dict)
    assert cast("dict[str, object]", tool_config)["toolChoice"] == {"any": {}}

    response = converse_response(
        {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": "checking"},
                        {
                            "toolUse": {
                                "toolUseId": "call-2",
                                "name": "bash",
                                "input": {"command": "pwd"},
                            }
                        },
                    ],
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 42, "outputTokens": 7},
        },
        "model-id",
    )

    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.tool_calls is not None
    assert response.choices[0].message.tool_calls[0].function.name == "bash"
    assert response.token_usage().input_tokens == 42


@pytest.mark.parametrize("stop_reason", ["content_filtered", "guardrail_intervened"])
def test_converse_response_preserves_filtered_stops(stop_reason: str) -> None:
    """Blocked Bedrock turns cannot look like successful assistant stops."""
    response = converse_response(
        {
            "output": {"message": {"role": "assistant", "content": []}},
            "stopReason": stop_reason,
            "usage": {"inputTokens": 5, "outputTokens": 0},
        },
        "model-id",
    )

    assert response.choices[0].finish_reason == "content_filter"


def test_converse_response_normalizes_cached_prompt_tokens() -> None:
    """Converse excludes the cache legs from inputTokens; the mapping adds them back.

    Without the normalization a cached tool-calling call under-reports its input
    by the entire cached prefix — under-billing, not just a lost discount. The
    read leg rides `prompt_tokens_details.cached_tokens`, the shape the serving
    log prices at the cache-read rate.
    """
    response = converse_response(
        {
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 100,
                "outputTokens": 5,
                "cacheReadInputTokens": 900,
                "cacheWriteInputTokens": 50,
            },
        },
        "model-id",
    )

    assert response.usage is not None
    assert response.usage.prompt_tokens == 1050
    assert response.usage.prompt_tokens_details is not None
    assert response.usage.prompt_tokens_details.cached_tokens == 900
    usage = response.token_usage()
    assert usage.input_tokens == 1050
    assert usage.cached_input_tokens == 900
    assert usage.cache_write_input_tokens == 50


def test_converse_response_carries_the_cache_split_serving_reads() -> None:
    """Converse reports cache reads/writes beside inputTokens; dropping them priced
    cached tokens at the full input rate (the Anthropic path's overcharge, mirrored)."""
    raw = {
        "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 100,
            "outputTokens": 5,
            "cacheReadInputTokens": 9000,
            "cacheWriteInputTokens": 500,
        },
    }

    response = converse_response(raw, "us.anthropic.claude-opus-4-8")

    assert response.usage is not None
    extra = response.usage.model_extra or {}
    assert response.usage.prompt_tokens == 9600  # reads/writes fold into the total
    assert extra.get("cache_read_input_tokens") == 9000  # raw wire-shape parity
    assert response.usage.cache_creation_input_tokens == 500  # typed: the priced write leg
    counts = response.token_usage()
    assert counts.cached_input_tokens == 9000
    assert counts.cache_write_input_tokens == 500


def test_converse_omits_tools_for_initial_none_choice() -> None:
    request = ChatRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "answer without tools"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "run a command",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "tool_choice": "none",
        }
    )

    wire = converse_request(request, "model-id")

    assert "toolConfig" not in wire


def test_converse_retains_tools_for_none_choice_with_tool_history() -> None:
    request = ChatRequest.model_validate(
        {
            "messages": [
                {"role": "user", "content": "list files"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": '{"command":"ls"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "a.txt"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "run a command",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "tool_choice": "none",
        }
    )

    wire = converse_request(request, "model-id")

    tool_config = wire["toolConfig"]
    assert isinstance(tool_config, dict)
    tool_config_data = cast("dict[str, object]", tool_config)
    # Locked to auto so the model can interpret history but cannot call new tools.
    assert tool_config_data["toolChoice"] == {"auto": {}}
    assert cast("list[dict[str, object]]", tool_config_data["tools"])[0]["toolSpec"] == {
        "name": "bash",
        "description": "run a command",
        "inputSchema": {"json": {"type": "object"}},
    }


def test_converse_retains_toolconfig_for_toolless_replay_with_history() -> None:
    """Replayed tool history without current tools still needs a toolConfig placeholder.

    Bedrock rejects requests whose message history contains toolUse or toolResult
    blocks if toolConfig is absent, even when no tools are passed for this turn.
    """
    request = ChatRequest.model_validate(
        {
            "messages": [
                {"role": "user", "content": "list files"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": '{"command":"ls"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "a.txt"},
                {"role": "user", "content": "now summarise"},
            ],
            # No tools on this turn.
        }
    )

    wire = converse_request(request, "model-id")

    tool_config = wire["toolConfig"]
    assert isinstance(tool_config, dict)
    tool_config_data = cast("dict[str, object]", tool_config)
    assert tool_config_data["tools"] == []
    assert tool_config_data["toolChoice"] == {"auto": {}}

