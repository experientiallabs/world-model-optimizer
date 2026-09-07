"""Tests for Datadog LLM Observability span export normalization."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import JsonValue

from exp.simulation.ingest.datadog import DATADOG_SOURCE
from exp.simulation.ingest.sources import CANONICAL_TRACE_SOURCES, load_trace_source

_START_NS = 1_788_516_258_420_588_000


def _llm_span(
    *,
    trace_id: JsonValue = "trace-1",
    span_id: JsonValue = "span-1",
    parent_id: JsonValue | None = None,
    name: str = "answer",
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    inputs: JsonValue | None = None,
    outputs: JsonValue | None = None,
) -> dict[str, JsonValue]:
    """Return one Datadog LLM span with nanosecond timing and model identity.

    Args:
        trace_id: Declared trace identifier.
        span_id: Declared span identifier.
        parent_id: Declared parent span identifier, when any.
        name: Declared span name.
        model: Declared model name.
        provider: Declared provider name.
        inputs: Declared span input, defaulting to a user message list.
        outputs: Declared span output, defaulting to an assistant completion.

    Returns:
        One Datadog span object.
    """
    span: dict[str, JsonValue] = {
        "trace_id": trace_id,
        "span_id": span_id,
        "name": name,
        "resource": name,
        "service": "agent",
        "type": "llm",
        "start": _START_NS,
        "duration": 130_000,
        "meta": {
            "span.kind": "llm",
            "model_name": model,
            "model_provider": provider,
            "ml_app": "experiential-check",
        },
        "metrics": {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
        "error": 0,
    }
    span["meta_struct"] = {
        "_llmobs": {
            "meta": {
                "input": inputs
                if inputs is not None
                else {"messages": [{"role": "user", "content": "Where is my order?"}]},
                "output": outputs
                if outputs is not None
                else {"messages": [{"role": "assistant", "content": "It ships tomorrow."}]},
                "span": {"kind": "llm"},
                "model_name": model,
                "model_provider": provider,
            },
            "metrics": {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
            "tags": {"ml_app": "experiential-check"},
        }
    }
    if parent_id is not None:
        span["parent_id"] = parent_id
    return span


def _tool_span(
    *,
    trace_id: JsonValue = "trace-1",
    span_id: JsonValue = "span-2",
    parent_id: JsonValue = "span-1",
    name: str = "lookup_order",
) -> dict[str, JsonValue]:
    """Return one Datadog tool span that reports its arguments and result."""
    span = _llm_span(trace_id=trace_id, span_id=span_id, parent_id=parent_id, name=name)
    assert isinstance(span["meta"], dict)
    assert isinstance(span["meta_struct"], dict)
    span["meta"]["span.kind"] = "tool"
    llmobs = span["meta_struct"]["_llmobs"]
    assert isinstance(llmobs, dict)
    meta = llmobs["meta"]
    assert isinstance(meta, dict)
    meta["span"] = {"kind": "tool"}
    meta["input"] = {"value": '{"order": "A1"}'}
    meta["output"] = {"value": "ships tomorrow"}
    del meta["model_name"]
    del meta["model_provider"]
    return span


def test_load_datadog_file_keeps_completion_and_model_identity(tmp_path: Path) -> None:
    """The model span keeps its completion text and declared provider model."""
    path = tmp_path / "datadog.json"
    path.write_text(
        json.dumps([_llm_span(), _tool_span(), _llm_span(span_id="span-3")]),
        encoding="utf-8",
    )

    result = DATADOG_SOURCE.load(path)

    assert result.issues == ()
    assert len(result.traces) == 1
    assert result.traces[0].task == "Where is my order?"
    completions = [
        str(span.attributes.get("gen_ai.completion") or "") for span in result.traces[0].spans
    ]
    assert any("It ships tomorrow." in completion for completion in completions)
    assert result.traces[0].spans[0].attributes["gen_ai.request.model"] == "gpt-4o-mini"


def test_load_datadog_file_preserves_usage_when_declared(tmp_path: Path) -> None:
    """Token accounting declared on LLM spans is retained on the model span."""
    path = tmp_path / "datadog.json"
    path.write_text(json.dumps([_llm_span()]), encoding="utf-8")

    result = DATADOG_SOURCE.load(path)

    usage = result.traces[0].spans[0].usage
    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens) == (12, 8)


def test_load_datadog_file_tool_result_is_paired_with_requesting_call(tmp_path: Path) -> None:
    """A tool span is normalized as a tool_result paired with the earlier model call."""
    path = tmp_path / "datadog.json"
    path.write_text(
        json.dumps([_llm_span(span_id="span-1"), _tool_span(span_id="span-2", parent_id="span-1")]),
        encoding="utf-8",
    )

    result = DATADOG_SOURCE.load(path)

    tool_names = [span.attributes.get("gen_ai.tool.name") for span in result.traces[0].spans]
    assert "lookup_order" in tool_names
    assert len(result.traces[0].spans) == 2
    tool_span = next(
        span
        for span in result.traces[0].spans
        if span.attributes.get("gen_ai.tool.name") == "lookup_order"
    )
    assert json.loads(str(tool_span.attributes["gen_ai.tool.call.arguments"])) == {"order": "A1"}
    assert tool_span.attributes["gen_ai.tool.message"] == "ships tomorrow"


def test_load_datadog_file_ignores_orchestration_spans(tmp_path: Path) -> None:
    """Task and retrieval spans carry no direct evidence and are ignored."""
    ignored = _llm_span(span_id="span-99", name="retrieve")
    assert isinstance(ignored["meta"], dict)
    ignored["meta"]["span.kind"] = "retrieval"
    assert isinstance(ignored["meta_struct"], dict)
    llmobs = ignored["meta_struct"]["_llmobs"]
    assert isinstance(llmobs, dict)
    meta = llmobs["meta"]
    assert isinstance(meta, dict)
    meta["span"] = {"kind": "retrieval"}
    path = tmp_path / "datadog.json"
    path.write_text(json.dumps([_llm_span(span_id="span-1"), ignored]), encoding="utf-8")

    result = DATADOG_SOURCE.load(path)

    assert len(result.traces[0].spans) == 1


def test_load_datadog_file_keeps_workflow_agent_evidence(tmp_path: Path) -> None:
    """A workflow span with input and output is kept as agent-level evidence."""
    workflow = {
        "trace_id": "trace-1",
        "span_id": "span-root",
        "name": "chat",
        "type": "llm",
        "start": _START_NS,
        "duration": 264_000,
        "meta_struct": {
            "_llmobs": {
                "meta": {
                    "input": {"value": "Where is my order?"},
                    "output": {"value": "It ships tomorrow."},
                    "span": {"kind": "workflow"},
                },
                "tags": {"ml_app": "experiential-check"},
            }
        },
    }
    path = tmp_path / "datadog.json"
    path.write_text(
        json.dumps([workflow, _llm_span(span_id="span-1", parent_id="span-root")]),
        encoding="utf-8",
    )

    result = DATADOG_SOURCE.load(path)

    assert result.issues == ()
    assert result.traces[0].task == "Where is my order?"
    assert len(result.traces[0].spans) == 2


def test_load_datadog_file_accepts_v04_trace_arrays(tmp_path: Path) -> None:
    """A v0.4 array of trace arrays is flattened into one trace."""
    path = tmp_path / "datadog.json"
    path.write_text(json.dumps([[_llm_span(span_id="span-7")]]), encoding="utf-8")

    result = DATADOG_SOURCE.load(path)

    assert len(result.traces) == 1
    assert result.traces[0].spans[0].attributes["gen_ai.request.model"] == "gpt-4o-mini"


def test_load_datadog_file_accepts_jsonl(tmp_path: Path) -> None:
    """JSONL with one span per line is supported."""
    path = tmp_path / "datadog.jsonl"
    spans = [
        _llm_span(trace_id="trace-2", span_id="span-a"),
        _tool_span(trace_id="trace-2", span_id="span-b", parent_id="span-a"),
    ]
    path.write_text("\n".join(json.dumps(span) for span in spans), encoding="utf-8")

    result = DATADOG_SOURCE.load(path)

    assert len(result.traces) == 1
    assert len(result.traces[0].spans) == 2


def test_load_datadog_file_marks_failed_span(tmp_path: Path) -> None:
    """A span with a nonzero error flag retains the declared failure message."""
    span = _tool_span(span_id="span-err")
    span["error"] = 1
    assert isinstance(span["meta"], dict)
    span["meta"]["error.message"] = "not refundable"
    path = tmp_path / "datadog.json"
    path.write_text(json.dumps([_llm_span(span_id="span-1"), span]), encoding="utf-8")

    result = DATADOG_SOURCE.load(path)

    assert result.issues == ()
    failed = [span for span in result.traces[0].spans if span.failure is not None]
    assert len(failed) == 1
    assert failed[0].failure is not None
    assert failed[0].failure.message == "not refundable"


def test_load_datadog_file_accepts_flat_meta_shape(tmp_path: Path) -> None:
    """Spans without meta_struct read kind, model, and messages from flat meta."""
    span: dict[str, JsonValue] = {
        "trace_id": 12345,
        "span_id": 67890,
        "name": "answer",
        "resource": "answer",
        "start": _START_NS,
        "duration": 130_000,
        "meta": {
            "span.kind": "llm",
            "model_name": "gpt-4o-mini",
            "model_provider": "openai",
            "input.messages": json.dumps([{"role": "user", "content": "Where is my order?"}]),
            "output.messages": json.dumps([{"role": "assistant", "content": "It ships tomorrow."}]),
        },
        "metrics": {"input_tokens": 12, "output_tokens": 8},
        "error": 0,
    }
    path = tmp_path / "datadog.json"
    path.write_text(json.dumps([span]), encoding="utf-8")

    result = DATADOG_SOURCE.load(path)

    assert result.issues == ()
    assert result.traces[0].task == "Where is my order?"
    usage = result.traces[0].spans[0].usage
    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens) == (12, 8)


def test_datadog_is_registered_as_canonical_source() -> None:
    """The Datadog source is discoverable through the canonical source table."""
    assert "datadog" in CANONICAL_TRACE_SOURCES
    assert "datadog" in sorted(CANONICAL_TRACE_SOURCES)


def test_load_trace_source_dispatches_to_datadog(tmp_path: Path) -> None:
    """The generic loader routes ``datadog`` to the Datadog adapter."""
    path = tmp_path / "datadog.json"
    path.write_text(json.dumps([_llm_span()]), encoding="utf-8")

    result = load_trace_source("datadog", path)

    assert len(result.traces) == 1


def test_load_datadog_file_rejects_unsupported_shape(tmp_path: Path) -> None:
    """A payload without any Datadog record keys is reported as an exclusion."""
    path = tmp_path / "datadog.json"
    path.write_text(json.dumps({"unknown": 1}), encoding="utf-8")

    result = DATADOG_SOURCE.load(path)

    assert result.traces == ()
    assert len(result.issues) == 1
