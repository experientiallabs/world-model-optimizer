"""Tests for provider-neutral structured Responses API translation."""

import json

import pytest
from llm_waterfall import ChatRequest

from wmo.providers._responses_common import (
    responses_request,
    responses_response,
)


def test_responses_request_translates_exact_pi_first_turn() -> None:
    """The pi bridge request becomes a native, non-streaming Responses call."""
    request = ChatRequest.model_validate(
        {
            "model": "worker",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": [{"type": "text", "text": "go"}]},
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
            "store": False,
            "max_completion_tokens": 4096,
            "temperature": 0.7,
            "top_p": 0.9,
            "parallel_tool_calls": True,
            "tool_choice": "required",
            "reasoning": {"effort": "minimal"},
            "service_tier": "flex",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "read",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                        "strict": False,
                    },
                }
            ],
        }
    )

    payload = responses_request(
        request,
        "gpt-5.5-deployment",
        reasoning_effort="high",
        service_tier="priority",
    )

    assert payload == {
        "model": "gpt-5.5-deployment",
        "input": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "go"},
        ],
        "stream": False,
        "store": False,
        "max_output_tokens": 4096,
        "parallel_tool_calls": True,
        "tool_choice": "required",
        "reasoning": {"effort": "high"},
        "include": ["reasoning.encrypted_content"],
        "service_tier": "priority",
        "tools": [
            {
                "type": "function",
                "name": "read_file",
                "description": "read",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                "strict": False,
            }
        ],
    }
    assert "messages" not in payload
    assert "max_completion_tokens" not in payload
    assert "stream_options" not in payload
    assert "temperature" not in payload
    assert "top_p" not in payload


def test_responses_request_translates_tool_result_turn_with_exact_call_id() -> None:
    """A stateless replay preserves the assistant call and matching tool result."""
    request = ChatRequest.model_validate(
        {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "developer", "content": "be concise"},
                {"role": "user", "content": [{"type": "text", "text": "go"}]},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-abc",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"x"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "contents", "tool_call_id": "call-abc"},
            ],
            "max_tokens": 512,
            "store": True,
            "temperature": 0.25,
            "top_p": 0.8,
            "tool_choice": {"type": "function", "function": {"name": "read_file"}},
        }
    )

    payload = responses_request(request, "gpt-5.5")

    assert payload == {
        "model": "gpt-5.5",
        "input": [
            {"role": "system", "content": "system"},
            {"role": "developer", "content": "be concise"},
            {"role": "user", "content": "go"},
            {
                "type": "function_call",
                "call_id": "call-abc",
                "name": "read_file",
                "arguments": '{"path":"x"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call-abc",
                "output": "contents",
            },
        ],
        "stream": False,
        "store": True,
        "include": ["reasoning.encrypted_content"],
        "max_output_tokens": 512,
        "temperature": 0.25,
        "top_p": 0.8,
        "tool_choice": {"type": "function", "name": "read_file"},
    }


def test_responses_response_preserves_text_multiple_tools_usage_and_tier() -> None:
    """Native Responses output maps back to one OpenAI-compatible chat choice."""
    response = responses_response(
        {
            "id": "resp-1",
            "model": "gpt-5.5-2026-05-01",
            "status": "completed",
            "service_tier": "priority",
            "output": [
                {
                    "type": "reasoning",
                    "id": "reasoning-1",
                    "summary": [],
                    "encrypted_content": "ciphertext-1",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "Checking "},
                        {"type": "output_text", "text": "both."},
                    ],
                },
                {
                    "type": "function_call",
                    "call_id": "call-a",
                    "name": "read_file",
                    "arguments": '{"path":"a"}',
                    "status": "completed",
                },
                {
                    "type": "function_call",
                    "call_id": "call-b",
                    "name": "read_file",
                    "arguments": '{"path":"b"}',
                    "status": "completed",
                },
            ],
            "usage": {
                "input_tokens": 321,
                "output_tokens": 45,
                "total_tokens": 366,
                "input_tokens_details": {"cached_tokens": 120},
                "output_tokens_details": {"reasoning_tokens": 30},
            },
        }
    )

    assert response.model == "gpt-5.5-2026-05-01"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 321
    assert response.usage.completion_tokens == 45
    assert response.usage.model_extra == {
        "prompt_tokens_details": {"cached_tokens": 120},
        "completion_tokens_details": {"reasoning_tokens": 30},
    }
    assert response.wire_payload()["service_tier"] == "priority"
    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.content == "Checking both."
    assert choice.message.tool_calls is not None
    assert [call.id for call in choice.message.tool_calls] == ["call-a", "call-b"]
    assert [call.function.name for call in choice.message.tool_calls] == [
        "read_file",
        "read_file",
    ]
    assert [call.function.arguments for call in choice.message.tool_calls] == [
        '{"path":"a"}',
        '{"path":"b"}',
    ]
    details = choice.message.model_extra
    assert details is not None
    assert details["reasoning_details"] == [
        {
            "type": "reasoning.encrypted",
            "id": "call-a",
            "data": json.dumps(
                [
                    {
                        "type": "reasoning",
                        "id": "reasoning-1",
                        "summary": [],
                        "encrypted_content": "ciphertext-1",
                    }
                ],
                separators=(",", ":"),
                sort_keys=True,
            ),
        }
    ]


def test_ordered_encrypted_reasoning_round_trips_through_pi_chat_history() -> None:
    """A stateless second turn replays every ordered reasoning item through Pi once."""
    first = responses_response(
        {
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "id": "reasoning-1",
                    "summary": [],
                    "encrypted_content": "ciphertext-1",
                },
                {
                    "type": "reasoning",
                    "id": "reasoning-2",
                    "summary": [{"type": "summary_text", "text": "second"}],
                    "encrypted_content": "ciphertext-2",
                },
                {
                    "type": "function_call",
                    "call_id": "call-a",
                    "name": "read_file",
                    "arguments": '{"path":"a"}',
                    "status": "completed",
                },
                {
                    "type": "function_call",
                    "call_id": "call-b",
                    "name": "read_file",
                    "arguments": '{"path":"b"}',
                    "status": "completed",
                },
            ],
        }
    )
    wire_choices = first.wire_payload()["choices"]
    assert isinstance(wire_choices, list)
    wire_choice = wire_choices[0]
    assert isinstance(wire_choice, dict)
    assistant = wire_choice["message"]
    assert isinstance(assistant, dict)
    reasoning_details = assistant["reasoning_details"]
    assert isinstance(reasoning_details, list)
    assert len(reasoning_details) == 1
    detail = reasoning_details[0]
    assert isinstance(detail, dict)
    data = detail["data"]
    assert isinstance(data, str)
    assert [item["id"] for item in json.loads(data)] == ["reasoning-1", "reasoning-2"]
    request = ChatRequest.model_validate(
        {
            "messages": [
                {"role": "user", "content": "inspect"},
                assistant,
                {"role": "tool", "tool_call_id": "call-a", "content": "A"},
                {"role": "tool", "tool_call_id": "call-b", "content": "B"},
            ],
            "store": False,
        }
    )

    payload = responses_request(request, "gpt-5.5", reasoning_effort="high")

    assert payload["input"] == [
        {"role": "user", "content": "inspect"},
        {"role": "assistant", "content": ""},
        {
            "type": "reasoning",
            "id": "reasoning-1",
            "summary": [],
            "encrypted_content": "ciphertext-1",
        },
        {
            "type": "reasoning",
            "id": "reasoning-2",
            "summary": [{"type": "summary_text", "text": "second"}],
            "encrypted_content": "ciphertext-2",
        },
        {
            "type": "function_call",
            "call_id": "call-a",
            "name": "read_file",
            "arguments": '{"path":"a"}',
        },
        {
            "type": "function_call",
            "call_id": "call-b",
            "name": "read_file",
            "arguments": '{"path":"b"}',
        },
        {"type": "function_call_output", "call_id": "call-a", "output": "A"},
        {"type": "function_call_output", "call_id": "call-b", "output": "B"},
    ]
    assert payload["include"] == ["reasoning.encrypted_content"]


def test_duplicate_parallel_reasoning_details_replay_once() -> None:
    """Pi/provider compatibility duplicates cannot multiply a large encrypted trace."""
    reasoning = {
        "type": "reasoning",
        "id": "reasoning-1",
        "summary": [],
        "encrypted_content": "ciphertext-1",
    }
    data = json.dumps(reasoning, separators=(",", ":"), sort_keys=True)
    request = ChatRequest.model_validate(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-a",
                            "function": {"name": "read_file", "arguments": "{}"},
                        },
                        {
                            "id": "call-b",
                            "function": {"name": "read_file", "arguments": "{}"},
                        },
                    ],
                    "reasoning_details": [
                        {"type": "reasoning.encrypted", "id": "call-a", "data": data},
                        {"type": "reasoning.encrypted", "id": "call-b", "data": data},
                    ],
                }
            ]
        }
    )

    payload = responses_request(request, "gpt-5.5", reasoning_effort="high")

    inputs = payload["input"]
    assert isinstance(inputs, list)
    assert sum(item == reasoning for item in inputs) == 1


@pytest.mark.parametrize("status", [None, "queued", "in_progress", "cancelled", "unknown"])
def test_non_completed_responses_fail_closed(status: str | None) -> None:
    with pytest.raises(ValueError, match="non-completed status"):
        responses_response({"status": status, "output": []})


def test_failed_response_includes_provider_diagnostics() -> None:
    with pytest.raises(ValueError, match="code=bad_request, message=invalid tool"):
        responses_response(
            {
                "status": "failed",
                "error": {"code": "bad_request", "message": "invalid tool"},
                "output": [],
            }
        )


def test_incomplete_response_never_exposes_a_partial_tool_call() -> None:
    with pytest.raises(ValueError, match="incomplete response: max_output_tokens"):
        responses_response(
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "write_file",
                        "arguments": "{",
                        "status": "incomplete",
                    }
                ],
            }
        )


def test_completed_response_rejects_an_incomplete_output_item() -> None:
    with pytest.raises(ValueError, match="output with status 'incomplete'"):
        responses_response(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "write_file",
                        "arguments": "{",
                        "status": "incomplete",
                    }
                ],
            }
        )


def test_completed_response_requires_recognized_output() -> None:
    with pytest.raises(ValueError, match="without a message or function call"):
        responses_response({"status": "completed", "output": []})
