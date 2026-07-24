"""Shared span-to-`Trace` normalizer — the one core every span-based adapter reuses.

Most agent-observability providers (Arize/Phoenix, Langfuse, LangSmith, Braintrust) export spans
that follow either the OpenTelemetry **GenAI** semantic conventions (`gen_ai.*`) or the
**OpenInference** conventions (`llm.*`, `tool.*`, `input.value`/`output.value`, `openinference.span.
kind`). Rather than write a bespoke parser per provider, an adapter normalizes its raw export into a
flat list of `SpanRecord`s and calls `spans_to_traces()` here. The attribute *keys* differ across
conventions, so the field extractors below look in both vocabularies (GenAI first, OpenInference as
fallback).

Pipeline:
  raw OTLP/OpenInference payload --(collect_spans / a provider's own transform)--> list[SpanRecord]
  list[SpanRecord] --(spans_to_traces)--> list[Trace]

`spans_to_traces` groups by `trace_id`, orders each group by start time, and pairs each LLM/agent
span (an Action) with the following tool/execution span (its Observation), mirroring how a real
agent step reads: `(state, action) -> observation`. The optional `wmh.*` enrichment attributes
(`wmh.state.*` -> `Step.state_before`, `wmh.trace.metadata` -> `Trace.metadata`) are honored on any
span, so a faithfully captured trace round-trips for open-loop replay.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from pydantic import BaseModel, Field, JsonValue

from wmh.core.types import Action, ActionKind, EnvState, JsonObject, Observation, Step, Trace

# --- attribute vocabularies (GenAI semconv first, OpenInference fallback) ---------------------

# Operation/kind markers.
_LLM_OPS = frozenset({"chat", "text_completion", "invoke_agent", "generate_content"})
_TOOL_OPS = frozenset({"execute_tool"})
# OpenInference span kinds (attribute `openinference.span.kind`).
_OI_LLM_KINDS = frozenset({"LLM", "AGENT", "CHAIN"})
_OI_TOOL_KINDS = frozenset({"TOOL"})

# A tool call's name, in priority order across conventions.
_TOOL_NAME_KEYS = ("gen_ai.tool.name", "tool.name", "tool_call.function.name")
# A tool call's serialized arguments, in priority order.
_TOOL_ARG_KEYS = (
    "gen_ai.tool.call.arguments",
    "gen_ai.tool.arguments",
    "gen_ai.tool.input",
    "gen_ai.request.arguments",
    "tool_call.function.arguments",
    "input.value",
    "input",
)
# A tool execution's output, in priority order.
_TOOL_OUTPUT_KEYS = (
    "gen_ai.tool.message",
    "gen_ai.tool.output",
    "gen_ai.tool.call.result",
    "gen_ai.tool.result",
    "gen_ai.completion",
    "output.value",
    "output",
)
# The originating prompt / task, in priority order.
_PROMPT_KEYS = ("gen_ai.prompt", "input.value", "llm.input_messages")
# An LLM message completion, in priority order.
_COMPLETION_KEYS = ("gen_ai.completion", "output.value", "llm.output_messages")
# Presence of any of these marks a span as an LLM/agent span when no explicit op/kind is set.
_LLM_PRESENCE_KEYS = (
    "gen_ai.request.model",
    "gen_ai.completion",
    "gen_ai.prompt",
    "llm.model_name",
    "llm.input_messages",
)

# Optional `wmh.*` enrichment keys (a strict superset of any semconv).
_STATE_STRUCTURED_KEY = "wmh.state.structured"
_STATE_SCRATCHPAD_KEY = "wmh.state.scratchpad"
_TRACE_METADATA_KEY = "wmh.trace.metadata"


class SpanRecord(BaseModel):
    """A flattened span with attributes decoded to plain JSON — the normalizer's input unit.

    Adapters either build these directly (from a provider's own event shape) or via
    `collect_spans` (from an OTLP/OpenInference-JSON payload).
    """

    trace_id: str
    span_id: str = ""
    parent_span_id: str = ""
    name: str = ""
    start_nano: int = 0
    end_nano: int = 0
    attributes: JsonObject = Field(default_factory=dict)
    status_error: bool = False


class SyntheticSpanEmitter:
    """Builds an ordered synthetic `gen_ai.*` span sequence for one reconstructed trace.

    Adapters whose export is not itself a span stream (message lists, provider event rows) rebuild
    the agent's step sequence by calling `emit` per action/observation in order. The emitter owns
    the shared bookkeeping every such adapter repeated: a monotonic ordinal that drives both the
    synthetic span id and `start_nano`, and the `chat`/`execute_tool` operation-name stamp. Any
    `first_attrs` (e.g. the trace-level `gen_ai.prompt` task and `wmh.trace.metadata`) are merged
    into the first span only, without clobbering keys the caller already set.
    """

    def __init__(self, trace_id: str, *, first_attrs: JsonObject | None = None) -> None:
        self.trace_id = trace_id
        self.spans: list[SpanRecord] = []
        self._first_attrs = first_attrs or {}

    def emit(self, attrs: JsonObject, *, tool: bool, error: bool = False) -> None:
        """Append one span; `tool` picks the operation (tool execution vs LLM chat)."""
        ordinal = len(self.spans)
        if ordinal == 0:
            for key, value in self._first_attrs.items():
                attrs.setdefault(key, value)
        op = "execute_tool" if tool else "chat"
        self.spans.append(
            SpanRecord(
                trace_id=self.trace_id,
                span_id=f"{self.trace_id[:12]}{ordinal:06x}{'t' if tool else 'a'}",
                name=op,
                start_nano=ordinal,
                attributes={"gen_ai.operation.name": op, **attrs},
                status_error=error,
            )
        )


# --- value coercion ---------------------------------------------------------------------------


def iso_to_ordinal(value: object, fallback: int) -> int:
    """Map an ISO-8601 timestamp (or a `datetime`) to epoch microseconds; `fallback` if unusable.

    Accepts a string OR a `datetime`/pandas `Timestamp` (Phoenix's `get_spans_dataframe` yields
    datetimes, not ISO strings). A naive value (no tz offset — e.g. LangSmith's `2026-01-01T00:00`)
    is treated as **UTC**, not the machine's local zone, so ordering is reproducible across hosts.
    Only monotonicity within a trace matters (spans_to_traces sorts by start_nano), so microsecond
    precision and a list-index fallback are plenty. Shared by every row adapter's timestamp order.
    """
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value:
        text = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return fallback
    else:
        return fallback
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    try:
        return int(parsed.timestamp() * 1_000_000)
    except (ValueError, OverflowError, OSError):
        return fallback


def to_int(value: JsonValue) -> int:
    """Coerce an OTLP numeric/string to int; bool (an int subclass) is treated as non-numeric."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def as_text(value: JsonValue) -> str:
    """Render a value as a JSON-clean string: strings pass through, else compact JSON (no repr)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _as_str(value: JsonValue) -> str:
    return value if isinstance(value, str) else ""


# --- OpenAI chat-completion tool-call shape ---------------------------------------------------


def openai_tool_calls(output: JsonValue) -> list[JsonObject]:
    """Extract OpenAI-style tool calls from an `output` (a message object or a message list)."""
    if isinstance(output, dict):
        raw = output.get("tool_calls")
        if isinstance(raw, list):
            return [tc for tc in raw if isinstance(tc, dict)]
    if isinstance(output, list):
        calls: list[JsonObject] = []
        for message in output:
            if isinstance(message, dict):
                raw = message.get("tool_calls")
                if isinstance(raw, list):
                    calls.extend(tc for tc in raw if isinstance(tc, dict))
        return calls
    return []


def openai_call_name_args(tool_call: JsonObject) -> tuple[str, str]:
    """(name, raw-arguments-json) from a tool call in OpenAI-nested or flattened shape.

    Arguments are usually a JSON *string* (OpenAI) but may be an object; either way the returned
    value is a string the span carries, which the normalizer's `_tool_args` re-parses.
    """
    fn = tool_call.get("function")
    if isinstance(fn, dict):
        name = fn.get("name")
        args = fn.get("arguments")
    else:
        name = tool_call.get("name")
        args = tool_call.get("arguments")
    name_s = name if isinstance(name, str) else ""
    args_s = args if isinstance(args, str) else as_text(args)
    return name_s, args_s


# --- OTLP / OpenInference AnyValue decoding ---------------------------------------------------


def any_value(value: JsonValue) -> JsonValue:
    """Decode an OTLP `AnyValue` (`{"stringValue": ...}` etc.) to a plain JSON value."""
    if not isinstance(value, dict):
        return value
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        return to_int(value["intValue"])
    if "doubleValue" in value:
        return value["doubleValue"]
    if "boolValue" in value:
        return value["boolValue"]
    if "arrayValue" in value:
        arr = value["arrayValue"]
        values = arr.get("values") if isinstance(arr, dict) else None
        return [any_value(v) for v in values] if isinstance(values, list) else []
    if "kvlistValue" in value:
        kv = value["kvlistValue"]
        values = kv.get("values") if isinstance(kv, dict) else None
        return attrs_to_dict(values) if isinstance(values, list) else {}
    return value


def attrs_to_dict(attrs: JsonValue) -> JsonObject:
    """Turn an OTLP attribute list (`[{"key", "value": <AnyValue>}, ...]`) into a flat dict.

    Also accepts an already-flat `{key: value}` mapping (some providers export that shape), in which
    case values are returned as-is.
    """
    out: JsonObject = {}
    if isinstance(attrs, dict):
        for key, val in attrs.items():
            if isinstance(key, str):
                out[key] = val
        return out
    if not isinstance(attrs, list):
        return out
    for attr in attrs:
        if isinstance(attr, dict):
            key = attr.get("key")
            if isinstance(key, str):
                out[key] = any_value(attr.get("value"))
    return out


def parse_span(raw: JsonValue) -> SpanRecord | None:
    """Parse one OTLP-JSON span object into a `SpanRecord` (None if it lacks a trace id)."""
    if not isinstance(raw, dict):
        return None
    trace_id = raw.get("traceId")
    if not isinstance(trace_id, str) or not trace_id:
        return None
    status = raw.get("status")
    status_error = False
    if isinstance(status, dict):
        code = status.get("code")
        status_error = code in (2, "STATUS_CODE_ERROR")
    return SpanRecord(
        trace_id=trace_id,
        span_id=_as_str(raw.get("spanId")),
        parent_span_id=_as_str(raw.get("parentSpanId")),
        name=_as_str(raw.get("name")),
        start_nano=to_int(raw.get("startTimeUnixNano")),
        end_nano=to_int(raw.get("endTimeUnixNano")),
        attributes=attrs_to_dict(raw.get("attributes")),
        status_error=status_error,
    )


def collect_spans(obj: JsonValue) -> list[SpanRecord]:
    """Walk an OTLP-JSON payload, a list of payloads/spans, or a bare span into `SpanRecord`s."""
    spans: list[SpanRecord] = []
    if isinstance(obj, list):
        for item in obj:
            spans.extend(collect_spans(item))
        return spans
    if not isinstance(obj, dict):
        return spans
    if "resourceSpans" in obj:
        resource_spans = obj["resourceSpans"]
        if isinstance(resource_spans, list):
            for resource_span in resource_spans:
                spans.extend(_spans_in_resource(resource_span))
        return spans
    parsed = parse_span(obj)
    if parsed is not None:
        spans.append(parsed)
    return spans


def _spans_in_resource(resource_span: JsonValue) -> list[SpanRecord]:
    spans: list[SpanRecord] = []
    if not isinstance(resource_span, dict):
        return spans
    scope_spans = resource_span.get("scopeSpans")
    if not isinstance(scope_spans, list):
        return spans
    for scope_span in scope_spans:
        if not isinstance(scope_span, dict):
            continue
        raw_spans = scope_span.get("spans")
        if not isinstance(raw_spans, list):
            continue
        for raw in raw_spans:
            parsed = parse_span(raw)
            if parsed is not None:
                spans.append(parsed)
    return spans


# --- classification + field extraction --------------------------------------------------------


def _first(attrs: JsonObject, keys: tuple[str, ...]) -> JsonValue:
    for key in keys:
        value = attrs.get(key)
        if value is not None:
            return value
    return None


def _operation(span: SpanRecord) -> str:
    op = span.attributes.get("gen_ai.operation.name")
    return op if isinstance(op, str) else ""


def _oi_kind(span: SpanRecord) -> str:
    kind = span.attributes.get("openinference.span.kind")
    return kind if isinstance(kind, str) else ""


def is_tool_span(span: SpanRecord) -> bool:
    op = _operation(span)
    if op in _TOOL_OPS:
        return True
    if op in _LLM_OPS:
        return False
    kind = _oi_kind(span)
    if kind in _OI_TOOL_KINDS:
        return True
    if kind in _OI_LLM_KINDS:
        return False
    return span.name.startswith("execute_tool")


def is_llm_span(span: SpanRecord) -> bool:
    op = _operation(span)
    if op in _LLM_OPS:
        return True
    if op in _TOOL_OPS:
        return False
    kind = _oi_kind(span)
    if kind in _OI_LLM_KINDS:
        return True
    if kind in _OI_TOOL_KINDS:
        return False
    return any(span.attributes.get(key) is not None for key in _LLM_PRESENCE_KEYS)


def _coerce_args(raw: JsonValue) -> JsonObject:
    """Coerce a tool call's arguments (a dict, a JSON string, or a scalar) to an arguments dict."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        if not raw.strip():
            return {}  # empty/blank serialized args (e.g. a no-arg tool call) -> no arguments
        try:
            parsed: JsonValue = json.loads(raw)
        except json.JSONDecodeError:
            return {"value": raw}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {"value": raw}


def _tool_args(attrs: JsonObject) -> JsonObject:
    return _coerce_args(_first(attrs, _TOOL_ARG_KEYS))


# OpenInference flattens an LLM-emitted tool call across INDEXED attribute keys, e.g.
#   llm.output_messages.0.message.tool_calls.0.tool_call.function.name      = "get_user"
#   llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments = '{"id": "u1"}'
# so the tool call lives on the LLM span itself, not a static `tool.name` key.
_OI_TOOL_NAME_SUFFIX = ".tool_call.function.name"
_OI_TOOL_ARGS_SUFFIX = ".tool_call.function.arguments"


def _openinference_tool_call(attrs: JsonObject) -> tuple[str, JsonValue] | None:
    """The lowest-indexed OpenInference tool call `(name, raw_args)` on an LLM span, if any."""
    name_keys = sorted(k for k in attrs if k.endswith(_OI_TOOL_NAME_SUFFIX))
    for name_key in name_keys:
        name = attrs.get(name_key)
        if isinstance(name, str) and name:
            args_key = name_key[: -len(_OI_TOOL_NAME_SUFFIX)] + _OI_TOOL_ARGS_SUFFIX
            return name, attrs.get(args_key)
    return None


def action_from_llm_span(span: SpanRecord) -> Action:
    attrs = span.attributes
    tool_name = _first(attrs, _TOOL_NAME_KEYS)
    if isinstance(tool_name, str) and tool_name:
        return Action(kind=ActionKind.TOOL_CALL, name=tool_name, arguments=_tool_args(attrs))
    oi_call = _openinference_tool_call(attrs)
    if oi_call is not None:
        name, raw_args = oi_call
        return Action(kind=ActionKind.TOOL_CALL, name=name, arguments=_coerce_args(raw_args))
    completion = _first(attrs, _COMPLETION_KEYS)
    content = _first(attrs, _PROMPT_KEYS) if completion is None else completion
    return Action(kind=ActionKind.MESSAGE, content=as_text(content))


def tool_call_action_from_tool_span(span: SpanRecord) -> Action:
    name = _first(span.attributes, _TOOL_NAME_KEYS)
    return Action(
        kind=ActionKind.TOOL_CALL,
        name=name if isinstance(name, str) and name else None,
        arguments=_tool_args(span.attributes),
    )


def observation_from_tool_span(span: SpanRecord) -> Observation:
    content = as_text(_first(span.attributes, _TOOL_OUTPUT_KEYS))
    return Observation(content=content, is_error=span.status_error)


def _trace_task(spans: list[SpanRecord]) -> str | None:
    for span in spans:
        prompt = _first(span.attributes, _PROMPT_KEYS)
        if prompt is not None:
            return as_text(prompt)
    return None


def _state_before(span: SpanRecord) -> EnvState:
    """Read an optional `wmh.state.*` snapshot off a span (empty when absent)."""
    attrs = span.attributes
    structured = attrs.get(_STATE_STRUCTURED_KEY)
    if isinstance(structured, str):
        try:
            decoded: JsonValue = json.loads(structured)
        except json.JSONDecodeError:
            decoded = {}
        structured = decoded
    scratchpad = attrs.get(_STATE_SCRATCHPAD_KEY)
    return EnvState(
        structured=structured if isinstance(structured, dict) else {},
        scratchpad=scratchpad if isinstance(scratchpad, str) else "",
    )


def _trace_metadata(spans: list[SpanRecord]) -> JsonObject:
    """First `wmh.trace.metadata` object across a trace's spans."""
    for span in spans:
        raw = span.attributes.get(_TRACE_METADATA_KEY)
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                decoded: JsonValue = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, dict):
                return decoded
    return {}


def _build_steps(spans: list[SpanRecord]) -> list[Step]:
    """Pair ordered Action spans with their following Observation spans into Steps."""
    task = _trace_task(spans)
    steps: list[Step] = []
    pending: Action | None = None
    pending_ids: list[str] = []
    pending_state = EnvState()

    def flush(
        action: Action, observation: Observation, span_ids: list[str], state: EnvState
    ) -> None:
        steps.append(
            Step(
                action=action,
                observation=observation,
                state_before=state,
                task=task,
                raw_span_ids=span_ids,
            )
        )

    for span in spans:
        if is_tool_span(span):
            observation = observation_from_tool_span(span)
            if pending is None:
                action = tool_call_action_from_tool_span(span)
                flush(action, observation, [span.span_id], _state_before(span))
            else:
                # The LLM span usually carries the call's name/args; backfill from the tool span
                # only when it didn't. Derive the tool-span action once to avoid re-parsing.
                if pending.kind == ActionKind.TOOL_CALL and (
                    not pending.arguments or pending.name is None
                ):
                    from_tool = tool_call_action_from_tool_span(span)
                    if not pending.arguments:
                        pending.arguments = from_tool.arguments
                    if pending.name is None:
                        pending.name = from_tool.name
                flush(pending, observation, [*pending_ids, span.span_id], pending_state)
            pending, pending_ids, pending_state = None, [], EnvState()
        elif is_llm_span(span):
            if pending is not None:
                flush(pending, Observation(content=""), pending_ids, pending_state)
            pending, pending_ids = action_from_llm_span(span), [span.span_id]
            pending_state = _state_before(span)
        # Non-agent spans are ignored.

    if pending is not None:
        flush(pending, Observation(content=""), pending_ids, pending_state)
    return steps


def spans_to_traces(spans: list[SpanRecord], *, source: str) -> list[Trace]:
    """Group spans by trace id, order each group by start time, and build `Trace`s.

    This is the shared tail every span-based adapter calls after producing `SpanRecord`s.
    """
    by_trace: dict[str, list[SpanRecord]] = {}
    for span in spans:
        by_trace.setdefault(span.trace_id, []).append(span)
    # Sorting each group by (start_nano, span_id) leaves group[0] as that trace's earliest span, so
    # we reuse it as the inter-trace sort key rather than re-scanning.
    ordered: list[tuple[int, Trace]] = []
    for group in by_trace.values():
        group.sort(key=lambda s: (s.start_nano, s.span_id))
        trace = Trace(
            trace_id=group[0].trace_id,
            steps=_build_steps(group),
            source=source,
            metadata=_trace_metadata(group),
        )
        ordered.append((group[0].start_nano, trace))
    ordered.sort(key=lambda pair: pair[0])
    return [trace for _, trace in ordered]
