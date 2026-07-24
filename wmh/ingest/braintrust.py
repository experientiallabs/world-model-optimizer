"""Braintrust adapter: turn a Braintrust span-row export into `Trace`s.

Braintrust does NOT export OTLP spans. It logs **spans as rows** in an experiment or project log;
the export API (`GET /v1/project_logs/{id}/fetch`, `GET /v1/experiment/{id}/fetch`) and the SDK
return one row per span, where a *trace* is the set of rows sharing a `root_span_id`:

    {"span_id": "s2", "root_span_id": "r1", "span_parents": ["s1"],
     "span_attributes": {"name": "llm", "type": "llm"},
     "input": [{"role": "user", "content": "what's the weather in Paris?"}],
     "output": {"role": "assistant",
                "tool_calls": [{"id": "c1",
                    "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
     "metadata": {"model": "gpt-4o"}, "created": "2026-01-01T00:00:01.000Z", "error": null}

Because this is not an OTLP/OpenInference span shape, the adapter overrides `spans_from_payload`
(like `wmh.ingest.langfuse`) and re-emits each row in the **OTel-GenAI vocabulary** so the shared
classifier/normalizer (`wmh.ingest.normalize`) does the pairing/state/metadata work:

  - The row's `span_attributes.type` classifies it. A row of type `llm`/`task`/`function`/`chain`
    whose `output` carries OpenAI-style `tool_calls` becomes one `chat` **action** span per call
    (`gen_ai.tool.name` + `gen_ai.tool.call.arguments`).
  - A row of type `tool` (or a tool-like row carrying `output`) becomes an `execute_tool` **result**
    span (`gen_ai.tool.message`), also carrying name/args so a standalone tool row still pairs.
  - An `llm`/`task`/`function` row with no tool call becomes a plain `chat` message span
    (`gen_ai.completion`).
  - A non-null `error` marks the row's step as an error (`status_error=True`).

Rows are grouped by `root_span_id` (the trace key; `span_id` is the per-span key) and ordered by the
`created` ISO-8601 timestamp -> a monotonic ordinal (list index when absent). Only monotonicity
within a trace matters — the normalizer sorts by `start_nano`. The first user message in the first
row's `input` becomes the step `task` (`gen_ai.prompt`); the row-level `metadata` round-trips via
`wmh.trace.metadata`.

Accepted file shapes (`from_file`): a single span row, a JSON array of rows, an API page wrapper
(`{"events": [...]}` or `{"data": [...]}`), or JSONL (one row per line).

Pull: live pull via the Braintrust SDK is not implemented; export to a file and use `from_file`. The
`BaseTraceAdapter` default raises a friendly error pointing at `--file`.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid

import httpx
from pydantic import JsonValue

from wmh.core.types import JsonObject
from wmh.ingest.adapter import VendorPull, register_adapter
from wmh.ingest.base import BaseTraceAdapter
from wmh.ingest.normalize import (
    SpanRecord,
    as_text,
    iso_to_ordinal,
    openai_call_name_args,
    openai_tool_calls,
)

# Braintrust REST API. The fetch endpoint returns `{"events": [span rows]}`; the project list
# resolves a human project name to its id. Overridable for self-hosted deployments.
_API_BASE = os.environ.get("BRAINTRUST_API_URL", "https://api.braintrust.dev").rstrip("/")
_API_KEY_ENV = "BRAINTRUST_API_KEY"

# Row types (`span_attributes.type`) that represent a model/agent turn (a potential action), vs a
# tool execution (a result). Braintrust's common types are: "llm", "tool", "function", "task",
# "score", "eval". We treat anything tool-like as a result and the rest of the "thinking" types as
# llm-ish; unknown types fall back to a content-based guess.
_LLM_TYPES = frozenset({"llm", "task", "function", "chain", "agent"})
_TOOL_TYPES = frozenset({"tool"})


def _as_str(value: JsonValue) -> str:
    return value if isinstance(value, str) else ""


def _looks_like_uuid(value: str) -> bool:
    """True if `value` is a canonical UUID (Braintrust project ids), so we skip the name lookup."""
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


def _start_ordinal(row: JsonObject, fallback: int) -> int:
    """Monotonic ordering key from the row's `created` timestamp (shared helper; UTC-safe)."""
    return iso_to_ordinal(row.get("created"), fallback)


def _row_type(row: JsonObject) -> str:
    """The span type from `span_attributes.type` (Braintrust's row kind)."""
    attrs = row.get("span_attributes")
    if isinstance(attrs, dict):
        value = attrs.get("type")
        if isinstance(value, str):
            return value.lower()
    return ""


def _row_name(row: JsonObject) -> str:
    """A display/tool name from `span_attributes.name` (or top-level `name`)."""
    attrs = row.get("span_attributes")
    if isinstance(attrs, dict):
        value = attrs.get("name")
        if isinstance(value, str) and value:
            return value
    name = row.get("name")
    return name if isinstance(name, str) else ""


def _is_error(row: JsonObject) -> bool:
    """A non-null/non-empty `error` marks the row as a failed step."""
    error = row.get("error")
    if error is None:
        return False
    if isinstance(error, str):
        return bool(error.strip())
    return bool(error)


def _first_user_text(value: JsonValue) -> str | None:
    """The first user message text from an `input` (message list), else the input rendered as text.

    Braintrust `input` is commonly a list of chat messages; the task is the first user turn. When
    `input` is not a message list, the whole input (rendered) stands in as the task.
    """
    if isinstance(value, list):
        for message in value:
            if isinstance(message, dict) and message.get("role") == "user":
                return as_text(message.get("content"))
        return None
    if value is None:
        return None
    return as_text(value)


def _completion_text(output: JsonValue) -> str:
    """Render an llm `output` to text: an assistant message's `content`, else the whole output."""
    if isinstance(output, dict):
        content = output.get("content")
        if content is not None:
            return as_text(content)
    return as_text(output)


class BraintrustAdapter(BaseTraceAdapter):
    """Map a Braintrust span-row export into normalized `Trace`s. No SDK."""

    name = "braintrust"

    def _pull_payloads(self, pull: VendorPull) -> list[JsonValue]:
        """Fetch span rows live from the Braintrust REST API.

        `pull.project` is a project name or id; `pull.api_key` (else `$BRAINTRUST_API_KEY`) auths.
        Resolves a name to an id via `/v1/project`, then fetches `/v1/project_logs/{id}/fetch`,
        whose `{"events": [...]}` body is handed straight to `spans_from_payload`.
        """
        api_key = pull.api_key or os.environ.get(_API_KEY_ENV)
        if not api_key:
            raise ValueError(
                f"braintrust pull needs an API key: pass --api-key or set ${_API_KEY_ENV}"
            )
        if not pull.project:
            raise ValueError("braintrust pull needs --project (a project name or id)")
        headers = {"Authorization": f"Bearer {api_key}"}
        project_id = self._resolve_project_id(pull.project, headers)
        params: dict[str, str] = {}
        if pull.limit is not None:
            params["limit"] = str(pull.limit)
        resp = httpx.get(
            f"{_API_BASE}/v1/project_logs/{project_id}/fetch",
            headers=headers,
            params=params,
            timeout=60.0,
        )
        resp.raise_for_status()
        return [resp.json()]

    def _resolve_project_id(self, project: str, headers: dict[str, str]) -> str:
        """Resolve a project name to its id (a UUID-shaped `project` is used as the id directly).

        Filters by `project_name` server-side rather than listing+scanning, so it does not miss
        projects past an arbitrary page limit.
        """
        if _looks_like_uuid(project):
            return project
        resp = httpx.get(
            f"{_API_BASE}/v1/project",
            headers=headers,
            params={"project_name": project, "limit": "100"},
            timeout=30.0,
        )
        resp.raise_for_status()
        body = resp.json()
        objects = body.get("objects", []) if isinstance(body, dict) else []
        for obj in objects:
            if isinstance(obj, dict) and obj.get("name") == project:
                pid = obj.get("id")
                if isinstance(pid, str):
                    return pid
        raise ValueError(f"braintrust project {project!r} not found for this API key")

    def spans_from_payload(self, payload: JsonValue) -> list[SpanRecord]:
        """Map a Braintrust export (a row, a list, or a page wrapper) to `SpanRecord`s."""
        rows = self._rows(payload)
        # Group rows by their trace key (`root_span_id`), preserving first-seen order of traces.
        by_trace: dict[str, list[JsonObject]] = {}
        for row in rows:
            by_trace.setdefault(self._trace_id(row), []).append(row)
        spans: list[SpanRecord] = []
        for trace_id, trace_rows in by_trace.items():
            spans.extend(self._spans_for_trace(trace_id, trace_rows))
        return spans

    def _rows(self, payload: JsonValue) -> list[JsonObject]:
        """Normalize a payload into a flat list of Braintrust span-row objects.

        Accepts a single row, a bare list of rows, or an API page wrapper (`{"events": [...]}` /
        `{"data": [...]}` as returned by the fetch endpoints).
        """
        if isinstance(payload, list):
            out: list[JsonObject] = []
            for item in payload:
                out.extend(self._rows(item))
            return out
        if not isinstance(payload, dict):
            return []
        for wrapper_key in ("events", "data"):
            inner = payload.get(wrapper_key)
            if isinstance(inner, list):
                out = []
                for item in inner:
                    out.extend(self._rows(item))
                return out
        # A bare span row is identified by its grouping/identity keys.
        if any(key in payload for key in ("root_span_id", "span_id", "span_attributes")):
            return [payload]
        return []

    def _trace_id(self, row: JsonObject) -> str:
        """The trace grouping key: `root_span_id`, then `span_id`, else a hash (stable fallback)."""
        for key in ("root_span_id", "span_id"):
            value = row.get(key)
            if isinstance(value, str) and value:
                return value
        return hashlib.sha256(as_text(row).encode()).hexdigest()[:32]

    def _spans_for_trace(self, trace_id: str, rows: list[JsonObject]) -> list[SpanRecord]:
        # Order by `created`; ties (or absent timestamps) keep input order via the index fallback.
        indexed = list(enumerate(rows))
        indexed.sort(key=lambda pair: (_start_ordinal(pair[1], pair[0]), pair[0]))

        # The task is the first user message across the trace's rows (in time order).
        task: str | None = None
        for _, row in indexed:
            task = _first_user_text(row.get("input"))
            if task is not None:
                break
        # Trace metadata: the first row's metadata object (in time order).
        meta_obj: JsonObject = {}
        for _, row in indexed:
            meta = row.get("metadata")
            if isinstance(meta, dict) and meta:
                meta_obj = meta
                break

        spans: list[SpanRecord] = []
        ordinal = 0

        def emit(attrs: JsonObject, *, tool: bool, error: bool = False) -> None:
            nonlocal ordinal
            if ordinal == 0:
                if task is not None:
                    attrs.setdefault("gen_ai.prompt", task)
                if meta_obj:
                    attrs.setdefault("wmh.trace.metadata", json.dumps(meta_obj))
            spans.append(
                SpanRecord(
                    trace_id=trace_id,
                    span_id=f"{trace_id[:12]}{ordinal:06x}{'t' if tool else 'a'}",
                    name="execute_tool" if tool else "chat",
                    start_nano=ordinal,
                    attributes={
                        "gen_ai.operation.name": "execute_tool" if tool else "chat",
                        **attrs,
                    },
                    status_error=error,
                )
            )
            ordinal += 1

        for _, row in indexed:
            rtype = _row_type(row)
            error = _is_error(row)
            output = row.get("output")
            calls = openai_tool_calls(output) if rtype not in _TOOL_TYPES else []
            if calls:
                # An llm/task/function row that issued tool calls: one `chat` action span per call.
                # The RESULT comes from the sibling `tool` row below (the normalizer pairs the
                # nearest following execute_tool span), so we do NOT synthesize a result here.
                for tool_call in calls:
                    name, args = openai_call_name_args(tool_call)
                    emit(
                        {"gen_ai.tool.name": name, "gen_ai.tool.call.arguments": args},
                        tool=False,
                        error=error,
                    )
            elif rtype in _TOOL_TYPES or (rtype not in _LLM_TYPES and self._is_tool_like(row)):
                # A `tool` row (or an unknown-typed tool-like row) = a tool execution: an
                # `execute_tool` result span. We carry name/args too so a standalone tool row (no
                # preceding llm action) still pairs; the normalizer backfills the action otherwise.
                emit(
                    {
                        "gen_ai.tool.name": _row_name(row),
                        "gen_ai.tool.call.arguments": as_text(row.get("input")),
                        "gen_ai.tool.message": as_text(output),
                    },
                    tool=True,
                    error=error,
                )
            elif rtype in _LLM_TYPES or output is not None:
                # A plain llm/task turn (no tool call): a message action, no observation.
                emit({"gen_ai.completion": _completion_text(output)}, tool=False, error=error)
            # Rows with no usable output and no tool semantics (e.g. score/eval) are ignored.
        return spans

    def _is_tool_like(self, row: JsonObject) -> bool:
        """An unknown-typed row is a tool execution when it has a name and I/O (input/output)."""
        if row.get("output") is None and row.get("input") is None:
            return False
        return bool(_row_name(row))


register_adapter(BraintrustAdapter())
