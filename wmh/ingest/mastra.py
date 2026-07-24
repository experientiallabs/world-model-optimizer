"""Mastra adapter: turn a Mastra AI-tracing export into `Trace`s.

[Mastra](https://mastra.ai) (a TypeScript agent framework) records agent runs as **AI-tracing
spans** (`ExportedSpan`), typed by `type`, that share a `traceId`. The span id field is `id`, times
are `startTime`/`endTime`, and errors ride on a structured `errorInfo` object:

    {"traceId": "t1", "id": "s2", "parentSpanId": "s1",
     "name": "modelGeneration", "type": "model_generation",
     "input": [{"role": "user", "content": "what's the weather in Paris?"}],
     "output": {"role": "assistant",
                "toolCalls": [{"toolCallId": "c1", "toolName": "getWeather",
                               "input": {"city": "Paris"}}]},  # AI SDK v5 uses `input` (v4: `args`)
     "attributes": {"model": "gpt-4o"}, "startTime": "2026-01-01T00:00:01.000Z"}
    {"traceId": "t1", "id": "s3", "name": "getWeather", "type": "tool_call",
     "input": {"city": "Paris"}, "output": "18C and sunny", "startTime": "..."}

Because this is not an OTLP/OpenInference span shape, the adapter overrides `spans_from_payload`
(like `wmh.ingest.langfuse`) and emits `SpanRecord`s in the **OTel-GenAI vocabulary** so the shared
classifier/normalizer (`wmh.ingest.normalize`) does the pairing/state/metadata work:

  - a `model_generation` span (or the pre-rename `llm_generation`) whose `output` carries tool calls
    -> one `chat` action span per call (`gen_ai.tool.name` + `gen_ai.tool.call.arguments`). Mastra
    exposes tool calls as either the AI SDK `toolCalls` (`{toolCallId, toolName, input}` on v5;
    `args` on v4) or the OpenAI `tool_calls` (`{id, function:{name, arguments}}`); both are handled.
    (Tool calls may instead appear only as separate `tool_call` spans — that case is covered below.)
  - a `tool_call` / `mcp_tool_call` span -> an `execute_tool` result span (`gen_ai.tool.message`
    from `output`), also carrying name/args (from `name`/`input`) so a standalone tool span still
    pairs; the normalizer backfills the action's name/args from here if the model span lacked them.
  - a `model_generation` with no tool call -> a plain `chat` message span (`gen_ai.completion`).
  - `agent_run` / `workflow_*` / `model_chunk` / `model_step` / `generic` container/noise spans are
    skipped (no `(action) -> observation` step); an `agent_run`/`model_generation` input supplies
    the task.
  - a span with an `errorInfo` (or an error status) sets `status_error=True`.

Spans order by `startTime`/`startedAt` (ISO-8601 or datetime -> a monotonic ordinal, index if none).

Accepted file shapes (`from_file`): a single span, a JSON array of spans, a wrapper
(`{"spans": [...]}` / `{"traces": [...]}` / `{"data": [...]}`), or JSONL. Grouping is by `traceId`.

Pull: `_pull_payloads` fetches from a running Mastra server's observability API
(`{base}/api/observability/traces`), with the base URL passed as `--project` (or `$MASTRA_URL`). The
response is handed to the same flexible span extractor, so it tolerates the server's wrapper shape.
"""

from __future__ import annotations

import hashlib
import os

import httpx
from pydantic import JsonValue

from wmh.core.types import JsonObject
from wmh.ingest.adapter import VendorPull, register_adapter
from wmh.ingest.base import BaseTraceAdapter
from wmh.ingest.normalize import SpanRecord, SyntheticSpanEmitter, as_text, iso_to_ordinal

# Mastra self-hosts, so the "vendor" is a server base URL (dev default http://localhost:4111).
_MASTRA_URL_ENV = "MASTRA_URL"

# `type` values (normalized lowercase). LLM/agent turns vs tool executions; the rest are
# containers/noise that carry no standalone step. Mastra renamed LLM spans to "model" spans
# (changelog 2025-11-01): `model_generation` is current, `llm_generation` is the pre-rename name we
# still accept. `model_chunk`/`model_step`/`agent_run`/`workflow_*`/`generic` are not steps.
_LLM_TYPES = frozenset({"model_generation", "llm_generation"})
_TOOL_TYPES = frozenset({"tool_call", "mcp_tool_call"})


def _as_str(value: JsonValue) -> str:
    return value if isinstance(value, str) else ""


def _span_type(span: JsonObject) -> str:
    for key in ("spanType", "span_type", "type"):
        value = span.get(key)
        if isinstance(value, str) and value:
            return value.lower()
    return ""


def _start_ordinal(span: JsonObject, fallback: int) -> int:
    """Monotonic ordering key from the span's start time (shared helper; UTC-safe)."""
    for key in ("startTime", "startedAt", "start_time"):
        value = span.get(key)
        if value is not None:
            return iso_to_ordinal(value, fallback)
    return fallback


def _is_error(span: JsonObject) -> bool:
    """A non-empty `errorInfo`/`error`, or an error status, marks the span as failed."""
    for key in ("errorInfo", "error"):
        value = span.get(key)
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, dict) and value:
            return True
    status = span.get("status")
    return isinstance(status, str) and status.upper() in {"ERROR", "FAILED"}


def _tool_calls(output: JsonValue) -> list[JsonObject]:
    """Extract tool calls from an llm span `output` (AI SDK `toolCalls` or OpenAI `tool_calls`)."""
    calls: list[JsonObject] = []
    candidates: list[JsonValue] = [output]
    if isinstance(output, dict):
        # Mastra may nest the assistant message under output.message / output.response.
        for key in ("message", "response"):
            nested = output.get(key)
            if nested is not None:
                candidates.append(nested)
    for candidate in candidates:
        if isinstance(candidate, dict):
            for key in ("toolCalls", "tool_calls"):
                raw = candidate.get(key)
                if isinstance(raw, list):
                    calls.extend(tc for tc in raw if isinstance(tc, dict))
        elif isinstance(candidate, list):
            for message in candidate:
                if isinstance(message, dict):
                    for key in ("toolCalls", "tool_calls"):
                        raw = message.get(key)
                        if isinstance(raw, list):
                            calls.extend(tc for tc in raw if isinstance(tc, dict))
    return calls


def _call_name_args(tool_call: JsonObject) -> tuple[str, str]:
    """(name, raw-arguments-json) from a Mastra AI-SDK or OpenAI-shaped tool call."""
    fn = tool_call.get("function")
    if isinstance(fn, dict):  # OpenAI shape: {"function": {"name", "arguments": "<json str>"}}
        name = fn.get("name")
        args = fn.get("arguments")
    else:  # AI SDK shape: v5 {"toolName", "input": {...}, "toolCallId"}; v4 used "args".
        name = tool_call.get("toolName") or tool_call.get("name")
        args = tool_call.get("input")  # AI SDK v5
        if args is None:
            args = tool_call.get("args")  # AI SDK v4
        if args is None:
            args = tool_call.get("arguments")
    name_s = name if isinstance(name, str) else ""
    args_s = args if isinstance(args, str) else as_text(args)
    return name_s, args_s


def _tool_span_name(span: JsonObject) -> str:
    """A tool name for a tool_call span: explicit attribute first, else the span name."""
    attributes = span.get("attributes")
    if isinstance(attributes, dict):
        for key in ("toolName", "tool_name"):
            value = attributes.get(key)
            if isinstance(value, str) and value:
                return value
    name = span.get("name")
    return name if isinstance(name, str) else ""


def _first_user_text(value: JsonValue) -> str | None:
    """First user message text from a span `input` (a messages list), else the input as text."""
    if isinstance(value, list):
        for message in value:
            if isinstance(message, dict) and _as_str(message.get("role")).lower() in {
                "user",
                "human",
            }:
                content = message.get("content")
                if content is not None:
                    return as_text(content)
        return None
    if isinstance(value, dict):
        for key in ("prompt", "input", "query", "message"):
            inner = value.get(key)
            if isinstance(inner, str) and inner:
                return inner
        return None
    if isinstance(value, str) and value:
        return value
    return None


def _completion_text(output: JsonValue) -> str:
    """Render an llm span `output` to text: an assistant message content, else the whole output."""
    if isinstance(output, dict):
        for key in ("text", "content"):
            value = output.get(key)
            if isinstance(value, str) and value:
                return value
    return as_text(output)


class MastraAdapter(BaseTraceAdapter):
    """Map a Mastra AI-tracing export into normalized `Trace`s. No SDK."""

    name = "mastra"

    def spans_from_payload(self, payload: JsonValue) -> list[SpanRecord]:
        raw_spans = self._spans(payload)
        by_trace: dict[str, list[JsonObject]] = {}
        for span in raw_spans:
            by_trace.setdefault(self._trace_id(span), []).append(span)
        spans: list[SpanRecord] = []
        for trace_id, trace_spans in by_trace.items():
            spans.extend(self._spans_for_trace(trace_id, trace_spans))
        return spans

    def _spans(self, payload: JsonValue) -> list[JsonObject]:
        """Normalize a payload into a flat list of Mastra span objects.

        Accepts a single span, a bare list, or a wrapper (`{"spans"|"traces"|"data": [...]}`). A
        wrapper item may itself be a trace object holding its own `spans`, so we recurse.
        """
        if isinstance(payload, list):
            out: list[JsonObject] = []
            for item in payload:
                out.extend(self._spans(item))
            return out
        if not isinstance(payload, dict):
            return []
        for wrapper_key in ("spans", "traces", "data"):
            inner = payload.get(wrapper_key)
            if isinstance(inner, list):
                out = []
                for item in inner:
                    out.extend(self._spans(item))
                return out
        # A bare span carries a trace id and a type (Mastra's span id field is `id`).
        if any(key in payload for key in ("traceId", "type", "spanType", "id", "spanId")):
            return [payload]
        return []

    def _trace_id(self, span: JsonObject) -> str:
        for key in ("traceId", "trace_id"):
            value = span.get(key)
            if isinstance(value, str) and value:
                return value
        sid = span.get("id") or span.get("spanId") or span.get("span_id")
        if isinstance(sid, str) and sid:
            return sid
        return hashlib.sha256(as_text(span).encode()).hexdigest()[:32]

    def _spans_for_trace(self, trace_id: str, raw_spans: list[JsonObject]) -> list[SpanRecord]:
        indexed = list(enumerate(raw_spans))
        indexed.sort(key=lambda pair: (_start_ordinal(pair[1], pair[0]), pair[0]))

        task: str | None = None
        for _, span in indexed:
            task = _first_user_text(span.get("input"))
            if task is not None:
                break

        emitter = SyntheticSpanEmitter(
            trace_id, first_attrs={"gen_ai.prompt": task} if task is not None else None
        )
        for _, span in indexed:
            stype = _span_type(span)
            error = _is_error(span)
            if stype in _LLM_TYPES:
                calls = _tool_calls(span.get("output"))
                if calls:
                    for tool_call in calls:
                        name, args = _call_name_args(tool_call)
                        emitter.emit(
                            {"gen_ai.tool.name": name, "gen_ai.tool.call.arguments": args},
                            tool=False,
                            error=error,
                        )
                else:
                    emitter.emit(
                        {"gen_ai.completion": _completion_text(span.get("output"))},
                        tool=False,
                        error=error,
                    )
            elif stype in _TOOL_TYPES:
                emitter.emit(
                    {
                        "gen_ai.tool.name": _tool_span_name(span),
                        "gen_ai.tool.call.arguments": as_text(span.get("input")),
                        "gen_ai.tool.message": as_text(span.get("output")),
                    },
                    tool=True,
                    error=error,
                )
            # agent_run / workflow_* / llm_chunk / generic -> no standalone step.
        return emitter.spans

    def _pull_payloads(self, pull: VendorPull) -> list[JsonValue]:
        """Fetch AI-tracing spans from a running Mastra server's observability API.

        `pull.project` (else `$MASTRA_URL`) is the Mastra server base URL, e.g.
        `http://localhost:4111`. Fetches `{base}/api/observability/traces` and hands the response to
        the flexible span extractor (which tolerates the server's `{traces|spans: [...]}` wrapper).
        """
        base = (pull.project or os.environ.get(_MASTRA_URL_ENV) or "").rstrip("/")
        if not base:
            raise ValueError(
                f"mastra pull needs the server URL: pass --project <base-url> or set "
                f"${_MASTRA_URL_ENV}"
            )
        headers = {"Authorization": f"Bearer {pull.api_key}"} if pull.api_key else {}
        params: dict[str, str] = {}
        if pull.limit is not None:
            params["perPage"] = str(pull.limit)
        resp = httpx.get(
            f"{base}/api/observability/traces", headers=headers, params=params, timeout=60.0
        )
        resp.raise_for_status()
        return [resp.json()]


register_adapter(MastraAdapter())
