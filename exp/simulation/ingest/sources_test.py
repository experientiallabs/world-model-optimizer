"""Tests for declared trace-source resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import JsonValue

from exp.simulation.ingest.sources import (
    CANONICAL_TRACE_SOURCES,
    TraceSourceError,
    load_trace_source,
)

_USER: JsonValue = {"role": "user", "content": "plan the trip"}
_ASSISTANT: JsonValue = {"role": "assistant", "content": "here is the plan"}

_MINIMAL_VENDOR_EXPORTS: dict[str, JsonValue] = {
    "braintrust": {
        "events": [
            {
                "id": "row-0",
                "span_id": "span-0",
                "root_span_id": "root-1",
                "span_attributes": {"type": "llm", "name": "chat"},
                "metrics": {"start": 1_772_000_000.0, "end": 1_772_000_001.0},
                "metadata": {"provider": "openai", "model": "gpt-test"},
                "input": [_USER],
                "output": _ASSISTANT,
            }
        ]
    },
    "datadog": [
        {
            "trace_id": "trace-1",
            "span_id": "span-0",
            "name": "answer",
            "start": 1_788_516_258_420_588_000,
            "duration": 130_000,
            "meta": {
                "span.kind": "llm",
                "model_name": "gpt-test",
                "model_provider": "openai",
            },
            "metrics": {"input_tokens": 10, "output_tokens": 20},
            "meta_struct": {
                "_llmobs": {
                    "meta": {
                        "input": {"messages": [_USER]},
                        "output": {"content": "here is the plan"},
                        "span": {"kind": "llm"},
                    }
                }
            },
        }
    ],
    "langfuse": {
        "id": "trace-1",
        "timestamp": "2026-02-01T00:00:00Z",
        "input": {"messages": [_USER]},
        "metadata": {"provider": "openai"},
        "observations": [
            {
                "id": "obs-0",
                "traceId": "trace-1",
                "type": "GENERATION",
                "name": "answer",
                "startTime": "2026-02-01T00:00:00Z",
                "endTime": "2026-02-01T00:00:01Z",
                "model": "gpt-test",
                "input": [_USER],
                "output": _ASSISTANT,
            }
        ],
    },
    "langsmith": {
        "runs": [
            {
                "id": "run-0",
                "trace_id": "trace-1",
                "run_type": "llm",
                "name": "ChatOpenAI",
                "start_time": "2026-03-01T00:00:00Z",
                "end_time": "2026-03-01T00:00:01Z",
                "inputs": {"messages": [_USER]},
                "outputs": {"generations": [[{"text": "here is the plan"}]]},
                "extra": {"metadata": {"ls_provider": "openai", "ls_model_name": "gpt-test"}},
            }
        ]
    },
    "mastra": {
        "spans": [
            {
                "traceId": "trace-1",
                "id": "span-0",
                "type": "model_generation",
                "name": "generate",
                "startTime": "2026-04-01T00:00:00Z",
                "endTime": "2026-04-01T00:00:01Z",
                "attributes": {"provider": "openai", "model": "gpt-test"},
                "input": {"messages": [_USER]},
                "output": {"text": "here is the plan"},
            }
        ]
    },
    "otel-genai": [
        {
            "trace_id": "9" * 32,
            "span_id": f"{1:016x}",
            "name": "agent.model_call",
            "start_time": "2026-06-01T00:00:00Z",
            "end_time": "2026-06-01T00:00:01Z",
            "attributes": {
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": "openai",
                "gen_ai.request.model": "gpt-test",
                "gen_ai.input.messages": json.dumps([_USER]),
                "gen_ai.output.messages": json.dumps([_ASSISTANT]),
            },
        }
    ],
    "phoenix": [
        {
            "context": {"trace_id": "trace-1", "span_id": "span-0"},
            "name": "ChatCompletion",
            "start_time": "2026-05-01T00:00:00Z",
            "end_time": "2026-05-01T00:00:01Z",
            "attributes": {
                "openinference": {"span": {"kind": "LLM"}},
                "llm": {
                    "provider": "openai",
                    "model_name": "gpt-test",
                    "input_messages": [{"message": _USER}],
                    "output_messages": [{"message": _ASSISTANT}],
                },
            },
        }
    ],
}


def _chat_json_export(path: Path) -> Path:
    """Write one minimal chat conversation export.

    Args:
        path: Directory receiving the export.

    Returns:
        Path of the written export.
    """
    export = path / "chat.json"
    export.write_text(
        json.dumps(
            {
                "conversation_id": "conversation-1",
                "messages": [
                    {"role": "user", "content": "plan the trip"},
                    {"role": "assistant", "content": "here is the plan"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return export


def test_declared_sources_are_the_supported_set() -> None:
    """The declared source names are exactly the supported normalizers, sorted."""
    assert CANONICAL_TRACE_SOURCES == (
        "braintrust",
        "chat-json",
        "datadog",
        "langfuse",
        "langsmith",
        "mastra",
        "otel-genai",
        "otlp",
        "phoenix",
        "posthog",
    )


def test_load_trace_source_dispatches_to_the_declared_loader(tmp_path: Path) -> None:
    """A declared source normalizes its own export through one canonical loader."""
    result = load_trace_source("chat-json", _chat_json_export(tmp_path))

    assert len(result.traces) == 1
    assert result.traces[0].source.identity.kind == "file"
    assert result.issues == ()


@pytest.mark.parametrize("source", sorted(_MINIMAL_VENDOR_EXPORTS))
def test_load_trace_source_wires_each_declared_vendor_to_its_own_loader(
    source: str,
    tmp_path: Path,
) -> None:
    """Each declared file vendor normalizes its own export and labels its provenance.

    Args:
        source: Declared canonical source format under test.
        tmp_path: Directory receiving the vendor export.
    """
    export = tmp_path / f"{source}.json"
    export.write_text(json.dumps(_MINIMAL_VENDOR_EXPORTS[source]), encoding="utf-8")

    result = load_trace_source(source, export)

    assert len(result.traces) == 1
    assert result.issues == ()
    assert result.traces[0].source.identity.source_id.startswith(f"{source}:")


def test_load_trace_source_normalizes_the_declared_name(tmp_path: Path) -> None:
    """Surrounding whitespace and letter case do not change the resolved loader."""
    result = load_trace_source("  Chat-JSON ", _chat_json_export(tmp_path))

    assert len(result.traces) == 1


def test_load_trace_source_preserves_a_hosted_durable_source_label(tmp_path: Path) -> None:
    """The canonical resolver passes a caller-owned label through instead of the local path."""
    result = load_trace_source(
        "chat-json",
        _chat_json_export(tmp_path),
        source_id="platform-trace-source:source-123",
    )

    assert result.traces[0].source.identity.source_id == "platform-trace-source:source-123"


def test_load_trace_source_rejects_an_undeclared_source(tmp_path: Path) -> None:
    """An unsupported name fails closed and lists the supported names."""
    with pytest.raises(TraceSourceError, match="unsupported trace source 'weave'"):
        load_trace_source("weave", _chat_json_export(tmp_path))


def test_load_trace_source_reports_the_source_that_failed(tmp_path: Path) -> None:
    """A source-specific failure is raised as one seam error naming the declared source."""
    with pytest.raises(TraceSourceError, match="chat-json normalization failed"):
        load_trace_source("chat-json", tmp_path / "absent.json")
