"""PostHog adapter: turn PostHog LLM-observability events into `Trace`s.

PostHog captures LLM traces as analytics **events** (not OTLP spans). Its LLM-observability schema
emits, per agent run, a set of events sharing `properties.$ai_trace_id`:

  - `$ai_generation` — one LLM call. Prompt in `properties.$ai_input` (a messages list), completion
    in `properties.$ai_output_choices` (a list of choice messages). A choice message that issues a
    tool call carries it OpenAI-style in `tool_calls`.
  - `$ai_span` — a non-LLM step (often a tool execution). Its input is `properties.$ai_input_state`
    and its result `properties.$ai_output_state`; the tool name is `properties.$ai_span_name`.
  - `$ai_trace` — a trace-root summary event (its `$ai_input`/`$ai_output_state` may hold the
    overall task/result); it carries no step of its own.
  - `$ai_is_error` (bool) on any event marks that step as an error.

Because this is not an OTLP/OpenInference span shape, the adapter overrides `spans_from_payload`
(like `wmh.ingest.langfuse`) and emits `SpanRecord`s in the **OTel-GenAI vocabulary** so the shared
classifier/normalizer (`wmh.ingest.normalize`) does the pairing/state/metadata work:

  - a `$ai_generation` whose output choices carry tool calls -> one `chat` action span per call
    (`gen_ai.tool.name` + `gen_ai.tool.call.arguments`); the result is paired with the sibling
    `$ai_span` tool event by the normalizer.
  - a `$ai_generation` with no tool call -> a plain `chat` message span (`gen_ai.completion`).
  - a tool-like `$ai_span` -> an `execute_tool` span (`gen_ai.tool.message` from the
    `$ai_output_state`), also carrying name/args so a standalone tool event still pairs.

Events are ordered by `timestamp` (ISO-8601 -> a monotonic ordinal, list index when absent). The
first user message across the trace's `$ai_input`s becomes the task (`gen_ai.prompt`).

Accepted file shapes (`from_file`): a single event, a JSON array of events, a query-result wrapper
(`{"results": [...]}` from the HogQL/events API), or JSONL (one event per line). Grouping is by
`properties.$ai_trace_id` (falling back to `$ai_span_id`/event id).

Pull: live pull via the PostHog query API is implemented in `_pull_payloads` (HogQL over `events`);
export to a file and use `from_file` if you prefer.
"""

from __future__ import annotations

import hashlib
import json
import os

import httpx
from pydantic import JsonValue

from wmh.core.types import JsonObject
from wmh.ingest.adapter import VendorPull, register_adapter
from wmh.ingest.base import BaseTraceAdapter
from wmh.ingest.normalize import (
    SpanRecord,
    SyntheticSpanEmitter,
    as_text,
    iso_to_ordinal,
    openai_call_name_args,
)

# PostHog API. `$AI_*` events are queried via HogQL over the `events` table. Host is region-specific
# (US: us.posthog.com, EU: eu.posthog.com), so it is configurable.
_API_HOST = os.environ.get("POSTHOG_HOST", "https://us.posthog.com").rstrip("/")
_API_KEY_ENV = "POSTHOG_API_KEY"


def _as_str(value: JsonValue) -> str:
    return value if isinstance(value, str) else ""


def _props(event: JsonObject) -> JsonObject:
    """An event's `properties` dict (PostHog nests the $ai_* fields there)."""
    props = event.get("properties")
    return props if isinstance(props, dict) else {}


def _start_ordinal(event: JsonObject, fallback: int) -> int:
    """Monotonic ordering key from the event `timestamp` (shared helper; UTC-safe)."""
    return iso_to_ordinal(event.get("timestamp"), fallback)


def _event_name(event: JsonObject) -> str:
    name = event.get("event")
    return name if isinstance(name, str) else ""


def _is_error(props: JsonObject) -> bool:
    flag = props.get("$ai_is_error")
    if isinstance(flag, bool):
        return flag
    error = props.get("$ai_error")
    if isinstance(error, str):
        return bool(error.strip())
    return bool(error)


def _tool_calls(choices: JsonValue) -> list[JsonObject]:
    """Tool calls from `$ai_output_choices` (a list of choice messages).

    PostHog's NORMALIZED format puts a tool call as a `content` PART `{"type": "function",
    "function": {"name", "arguments"}}` (arguments is a JSON object, not a string), NOT in a
    `tool_calls` array. We collect both: content-parts with `type == "function"` (the normalized
    shape) and a `tool_calls` array (raw-OpenAI passthrough); `openai_call_name_args` reads either.
    """
    calls: list[JsonObject] = []
    if not isinstance(choices, list):
        return calls
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        raw = choice.get("tool_calls")
        if isinstance(raw, list):
            calls.extend(tc for tc in raw if isinstance(tc, dict))
        content = choice.get("content")
        if isinstance(content, list):
            calls.extend(
                part
                for part in content
                if isinstance(part, dict) and part.get("type") == "function"
            )
    return calls


def _content_text(content: JsonValue) -> str:
    """Assistant text from a choice `content`: a plain string, or a parts list where text lives in
    `{"type": "text", "text": ...}` parts (PostHog's normalized multi-part shape). Non-text parts
    (e.g. `type == "function"` tool calls) are skipped."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str) and text:
                    texts.append(text)
        return "\n".join(texts)
    return ""


def _choices_text(choices: JsonValue) -> str:
    """Assistant text from `$ai_output_choices` (first choice's content), else the whole list."""
    if isinstance(choices, list):
        for choice in choices:
            if isinstance(choice, dict):
                text = _content_text(choice.get("content"))
                if text:
                    return text
    return as_text(choices)


def _first_user_text(messages: JsonValue) -> str | None:
    """First user/human message content in an `$ai_input` messages list."""
    if not isinstance(messages, list):
        return None
    for message in messages:
        if isinstance(message, dict) and _as_str(message.get("role")).lower() in {"user", "human"}:
            content = message.get("content")
            if content is not None:
                return as_text(content)
    return None


class PostHogAdapter(BaseTraceAdapter):
    """Map PostHog LLM-observability events into normalized `Trace`s."""

    name = "posthog"

    def spans_from_payload(self, payload: JsonValue) -> list[SpanRecord]:
        events = self._events(payload)
        by_trace: dict[str, list[JsonObject]] = {}
        for event in events:
            by_trace.setdefault(self._trace_id(event), []).append(event)
        spans: list[SpanRecord] = []
        for trace_id, trace_events in by_trace.items():
            spans.extend(self._spans_for_trace(trace_id, trace_events))
        return spans

    def _events(self, payload: JsonValue) -> list[JsonObject]:
        """Normalize a payload into a flat list of PostHog event objects.

        Accepts a single event, a bare list, or a query wrapper (`{"results": [...]}`).
        """
        if isinstance(payload, list):
            out: list[JsonObject] = []
            for item in payload:
                out.extend(self._events(item))
            return out
        if not isinstance(payload, dict):
            return []
        for wrapper_key in ("results", "events"):
            inner = payload.get(wrapper_key)
            if isinstance(inner, list):
                out = []
                for item in inner:
                    out.extend(self._events(item))
                return out
        if "event" in payload or "properties" in payload:
            return [payload]
        return []

    def _trace_id(self, event: JsonObject) -> str:
        props = _props(event)
        for key in ("$ai_trace_id", "$ai_span_id"):
            value = props.get(key)
            if isinstance(value, str) and value:
                return value
        eid = event.get("id") or event.get("uuid")
        if isinstance(eid, str) and eid:
            return eid
        return hashlib.sha256(as_text(event).encode()).hexdigest()[:32]

    def _spans_for_trace(self, trace_id: str, events: list[JsonObject]) -> list[SpanRecord]:
        indexed = list(enumerate(events))
        indexed.sort(key=lambda pair: (_start_ordinal(pair[1], pair[0]), pair[0]))

        task: str | None = None
        for _, event in indexed:
            task = _first_user_text(_props(event).get("$ai_input"))
            if task is not None:
                break

        emitter = SyntheticSpanEmitter(
            trace_id, first_attrs={"gen_ai.prompt": task} if task is not None else None
        )
        for _, event in indexed:
            name = _event_name(event)
            props = _props(event)
            error = _is_error(props)
            if name == "$ai_generation":
                calls = _tool_calls(props.get("$ai_output_choices"))
                if calls:
                    for tool_call in calls:
                        tool_name, args = openai_call_name_args(tool_call)
                        emitter.emit(
                            {"gen_ai.tool.name": tool_name, "gen_ai.tool.call.arguments": args},
                            tool=False,
                            error=error,
                        )
                else:
                    emitter.emit(
                        {"gen_ai.completion": _choices_text(props.get("$ai_output_choices"))},
                        tool=False,
                        error=error,
                    )
            elif name == "$ai_span" and self._span_is_tool(props):
                emitter.emit(
                    {
                        "gen_ai.tool.name": _as_str(props.get("$ai_span_name")),
                        "gen_ai.tool.call.arguments": as_text(props.get("$ai_input_state")),
                        "gen_ai.tool.message": as_text(props.get("$ai_output_state")),
                    },
                    tool=True,
                    error=error,
                )
            # $ai_trace (root summary) and other events carry no standalone step.
        return emitter.spans

    def _span_is_tool(self, props: JsonObject) -> bool:
        """An `$ai_span` is a tool execution when it has an output/input state or a span name."""
        return (
            props.get("$ai_output_state") is not None
            or props.get("$ai_input_state") is not None
            or bool(_as_str(props.get("$ai_span_name")))
        )

    def _pull_payloads(self, pull: VendorPull) -> list[JsonValue]:
        """Query PostHog for `$ai_*` events via HogQL and return them for normalization.

        `pull.project` is the PostHog project id; `pull.api_key` (else `$POSTHOG_API_KEY`) is a
        personal API key. Host is `$POSTHOG_HOST` (region-specific). Fetches recent `$ai_*` events.
        """
        api_key = pull.api_key or os.environ.get(_API_KEY_ENV)
        if not api_key:
            raise ValueError(
                f"posthog pull needs an API key: pass --api-key or set ${_API_KEY_ENV}"
            )
        if not pull.project:
            raise ValueError("posthog pull needs --project (the PostHog project id)")
        limit = pull.limit if pull.limit is not None else 1000
        query = (
            "select event, properties, timestamp from events "
            "where event like '$ai_%' order by timestamp asc limit " + str(int(limit))
        )
        # Trailing slash required: PostHog's DRF router 301-redirects `/query` -> `/query/`, and the
        # redirect drops the POST body (httpx follows as a GET), so hit the canonical URL directly.
        resp = httpx.post(
            f"{_API_HOST}/api/projects/{pull.project}/query/",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"query": {"kind": "HogQLQuery", "query": query}},
            timeout=60.0,
        )
        resp.raise_for_status()
        body = resp.json()
        # HogQL returns {"results": [[event, properties, timestamp], ...], "columns": [...]}.
        results = body.get("results", []) if isinstance(body, dict) else []
        events: list[JsonValue] = []
        for row in results:
            if isinstance(row, list) and len(row) >= 3:
                props = row[1]
                if isinstance(props, str):
                    try:
                        props = json.loads(props)
                    except json.JSONDecodeError:
                        props = {}
                events.append({"event": row[0], "properties": props, "timestamp": row[2]})
        return [events]


register_adapter(PostHogAdapter())
