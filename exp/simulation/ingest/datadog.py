"""Normalize Datadog LLM Observability span exports into canonical trace evidence.

Datadog APM spans carry ``trace_id``, ``span_id``, ``parent_id``, ``name``,
``resource``, ``start`` (nanoseconds since epoch) and ``duration``
(nanoseconds), plus ``meta``, ``metrics``, and ``error``. LLM Observability
spans add ``meta_struct._llmobs`` with ``meta.span.kind``, ``input``/``output``
(messages or value), ``model_name``/``model_provider``, ``metrics`` token
counts, and ``tags``. Exports arrive as a span array, a v0.4 array of trace
arrays, a ``spans``/``traces`` envelope, or JSONL with one span per line.

Spans map to canonical evidence by what they observe:

- ``llm`` becomes model calls, including the tool calls their output requests,
- ``tool`` becomes tool results paired with the earlier requesting call,
- ``workflow`` and ``agent`` become agent-level evidence when they declare
  input or output, and are ignored otherwise,
- ``task``, ``retrieval``, ``embedding``, and other orchestration types are
  not converted.

Model identity comes from ``model_name``/``model_provider`` and is retained
only when the export declares both; usage comes from ``input_tokens`` and
``output_tokens``.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.simulation.ingest.vendor_observations import (
    VendorObservation,
    VendorTokenUsage,
    declared_completion_text,
    declared_error_message,
    declared_model_identity,
    declared_tool_calls,
    declared_usage,
)
from exp.simulation.ingest.vendor_records import (
    first_text,
    first_user_text,
    json_text,
    json_value,
    required_text,
    source_interval,
)
from exp.simulation.ingest.vendor_source import VendorSource, record_flattener
from exp.simulation.ingest.vendor_trace import approved_extensions

VENDOR = "datadog"

_MODEL_KINDS = frozenset({"llm"})
_TOOL_KINDS = frozenset({"tool"})
_AGENT_KINDS = frozenset({"workflow", "agent"})
_MODEL_KEYS = ("model_name", "modelName", "model")
_PROVIDER_KEYS = ("model_provider", "modelProvider", "provider", "llm.system")
_ERROR_KEYS = ("error.message", "error_message", "errorMessage")


def _span_observation(span: JsonObject, ordinal: int) -> tuple[VendorObservation, ...]:
    """Convert one Datadog span to a declared model, tool-result, or agent observation.

    Args:
        span: Datadog span record.
        ordinal: Source order position for the emitted observation.

    Returns:
        Declared observation, or nothing for orchestration-only span kinds.

    Raises:
        VendorTraceFormatError: The span lacks identity, timing, or tool evidence.
    """
    llmobs = _llmobs_meta(span)
    kind = _span_kind(span, llmobs)
    if kind not in _MODEL_KINDS | _TOOL_KINDS | _AGENT_KINDS:
        return ()
    source_trace_id = _trace_id(span, llmobs)
    source_span_id = _span_id(span)
    started_at, ended_at = _interval(span)
    attributes = _attributes(span)
    inputs = _io(span, llmobs, "input")
    outputs = _io(span, llmobs, "output")
    extensions = _extensions(span, attributes)
    failure = _failure_message(span, attributes)
    parent = _parent_id(span, llmobs)
    if kind in _TOOL_KINDS:
        return (
            VendorObservation(
                source_trace_id=source_trace_id,
                source_span_id=source_span_id,
                ordinal=ordinal,
                started_at=started_at,
                ended_at=ended_at,
                kind="tool_result",
                source_parent_span_id=parent,
                request_text=first_user_text(inputs),
                tool_name=_tool_name(span, attributes),
                tool_arguments=json_text(inputs),
                tool_message=declared_completion_text(outputs),
                tool_call_id=first_text(attributes, ("toolCallId", "tool_call_id")),
                failure_message=failure,
                extensions=extensions,
            ),
        )
    if kind in _AGENT_KINDS:
        if inputs is None and outputs is None:
            return ()
        completion = declared_completion_text(outputs)
        if not completion and first_user_text(inputs) is None:
            return ()
        return (
            VendorObservation(
                source_trace_id=source_trace_id,
                source_span_id=source_span_id,
                ordinal=ordinal,
                started_at=started_at,
                ended_at=ended_at,
                kind="agent",
                source_parent_span_id=parent,
                request_text=first_user_text(inputs),
                completion_text=completion or None,
                failure_message=failure,
                extensions=extensions,
            ),
        )
    model, declared_model = declared_model_identity(
        {**attributes, **_identity_fields(span, llmobs)},
        model_keys=_MODEL_KEYS,
        provider_keys=_PROVIDER_KEYS,
    )
    return (
        VendorObservation(
            source_trace_id=source_trace_id,
            source_span_id=source_span_id,
            ordinal=ordinal,
            started_at=started_at,
            ended_at=ended_at,
            kind="model",
            source_parent_span_id=parent,
            request_text=first_user_text(inputs),
            input_messages=_input_messages(inputs),
            completion_text=declared_completion_text(outputs) or None,
            tool_calls=declared_tool_calls(outputs),
            model=model,
            usage=_usage(span, llmobs, attributes),
            failure_message=failure,
            declared_attributes=(
                {} if declared_model is None else {"gen_ai.request.model": declared_model}
            ),
            extensions=extensions,
        ),
    )


def _llmobs_meta(span: JsonObject) -> JsonObject:
    """Return the decoded Datadog LLM Observability payload for one span.

    Args:
        span: Datadog span record.

    Returns:
        The ``meta_struct._llmobs`` object with ``meta``, ``metrics``, and
        ``tags``, or an empty mapping when the span declares none.
    """
    meta_struct = span.get("meta_struct", span.get("metaStruct"))
    if not isinstance(meta_struct, dict):
        return {}
    llmobs = meta_struct.get("_llmobs")
    if isinstance(llmobs, dict):
        return llmobs
    return {}


def _span_kind(span: JsonObject, llmobs: JsonObject) -> str:
    """Return the lowercase declared Datadog span kind.

    Args:
        span: Datadog span record.
        llmobs: Decoded LLM Observability payload.

    Returns:
        Lowercase span kind, empty when the span declares none.
    """
    meta = llmobs.get("meta")
    if isinstance(meta, dict):
        inner = meta.get("span")
        if isinstance(inner, dict):
            kind = inner.get("kind")
            if isinstance(kind, str) and kind.strip():
                return kind.strip().casefold()
    for key in ("span.kind", "span_kind", "kind"):
        value = first_text(span, (key,))
        if value:
            return value.casefold()
    attributes = span.get("meta")
    if isinstance(attributes, dict):
        for key in ("span.kind", "span_kind", "kind"):
            raw = attributes.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip().casefold()
    return ""


def _trace_id(span: JsonObject, llmobs: JsonObject) -> str:
    """Read the Datadog trace identifier.

    Args:
        span: Datadog span record.
        llmobs: Decoded LLM Observability payload.

    Returns:
        Declared trace identifier.

    Raises:
        VendorTraceFormatError: The span declares no trace identifier.
    """
    for key in ("trace_id", "traceId", "traceID"):
        value = span.get(key)
        if isinstance(value, str) and value.strip():
            return required_text(value, "Datadog trace_id")
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
    trace_id = llmobs.get("trace_id", llmobs.get("traceId"))
    if isinstance(trace_id, str) and trace_id.strip():
        return required_text(trace_id, "Datadog trace_id")
    return required_text(None, "Datadog trace_id")


def _span_id(span: JsonObject) -> str:
    """Read the Datadog span identifier.

    Args:
        span: Datadog span record.

    Returns:
        Declared span identifier.

    Raises:
        VendorTraceFormatError: The span declares no span identifier.
    """
    for key in ("span_id", "spanId", "spanID", "id"):
        value = span.get(key)
        if isinstance(value, str) and value.strip():
            return required_text(value, "Datadog span_id")
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
    return required_text(None, "Datadog span_id")


def _parent_id(span: JsonObject, llmobs: JsonObject) -> str | None:
    """Return the declared Datadog parent span identifier, if any.

    Args:
        span: Datadog span record.
        llmobs: Decoded LLM Observability payload.

    Returns:
        Declared parent span identifier, or ``None`` when absent.
    """
    for key in ("parent_id", "parentId", "parent_span_id"):
        value = span.get(key)
        if isinstance(value, str) and value.strip() and value.strip() != "undefined":
            return value.strip()
        if isinstance(value, int) and not isinstance(value, bool) and value != 0:
            return str(value)
    parent_id = llmobs.get("parent_id", llmobs.get("parentId"))
    if isinstance(parent_id, str) and parent_id.strip() and parent_id.strip() != "undefined":
        return parent_id.strip()
    return None


def _attributes(span: JsonObject) -> JsonObject:
    """Return the Datadog span metadata object.

    Args:
        span: Datadog span record.

    Returns:
        Metadata mapping, empty when the span declares none.
    """
    value = span.get("meta", span.get("attributes"))
    if isinstance(value, dict):
        return value
    return {}


def _identity_fields(span: JsonObject, llmobs: JsonObject) -> JsonObject:
    """Return the declared Datadog model and provider names.

    Args:
        span: Datadog span record.
        llmobs: Decoded LLM Observability payload.

    Returns:
        Mapping with the declared model and provider values, when present.
    """
    fields: JsonObject = {}
    meta = llmobs.get("meta")
    candidates: list[JsonObject] = [span]
    if isinstance(meta, dict):
        candidates.append(meta)
    attributes = _attributes(span)
    candidates.append(attributes)
    for source in candidates:
        for key in _MODEL_KEYS:
            value = source.get(key)
            if isinstance(value, str) and value.strip() and key not in fields:
                fields[key] = value.strip()
        for key in _PROVIDER_KEYS:
            value = source.get(key)
            if isinstance(value, str) and value.strip() and key not in fields:
                fields[key] = value.strip()
    return fields


def _io(span: JsonObject, llmobs: JsonObject, which: str) -> JsonValue | None:
    """Read the declared Datadog span input or output.

    Datadog wraps scalar payloads as ``{"value": ...}``; the wrapper is removed
    so tool arguments, results, and completion text keep their declared shape.

    Args:
        span: Datadog span record.
        llmobs: Decoded LLM Observability payload.
        which: ``"input"`` or ``"output"``.

    Returns:
        Decoded input or output, or ``None`` when the span declares neither.
    """
    meta = llmobs.get("meta")
    if isinstance(meta, dict):
        value = meta.get(which)
        if value is not None:
            return _unwrap(value)
    for key in (which, f"{which}s"):
        value = json_value(span.get(key))
        if value is not None:
            return _unwrap(value)
    attributes = _attributes(span)
    for key in (which, f"{which}s", f"{which}.messages", f"{which}_messages"):
        if key in attributes:
            value = json_value(attributes[key])
            if value is not None:
                return _unwrap(value)
    return None


def _unwrap(value: JsonValue | None) -> JsonValue | None:
    """Remove the Datadog scalar ``{"value": ...}`` wrapper, when present.

    Args:
        value: Decoded span input or output.

    Returns:
        The wrapped payload, or the value unchanged when it is not a wrapper.
    """
    if isinstance(value, dict) and set(value) == {"value"}:
        return json_value(value["value"])
    return value


def _input_messages(inputs: JsonValue | None) -> JsonValue | None:
    """Return the declared model input messages for one Datadog model span.

    Args:
        inputs: Decoded span input.

    Returns:
        The declared message list, or the raw input when the span declares no list.
    """
    if isinstance(inputs, dict):
        for key in ("messages", "prompt", "input", "value"):
            value = inputs.get(key)
            if isinstance(value, list):
                return value
    if isinstance(inputs, list):
        return inputs
    return inputs


def _tool_name(span: JsonObject, attributes: JsonObject) -> str:
    """Read the executed tool name from a Datadog tool span.

    Args:
        span: Datadog span record.
        attributes: Declared span metadata.

    Returns:
        Declared tool name.

    Raises:
        VendorTraceFormatError: The span declares no tool name.
    """
    name = first_text(attributes, ("toolName", "tool_name", "name"))
    if name is not None:
        return name
    for key in ("name", "resource"):
        value = span.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return required_text(None, "Datadog tool span name")


def _interval(span: JsonObject) -> tuple[datetime, datetime]:
    """Read the source interval from Datadog nanosecond or ISO fields.

    Args:
        span: Datadog span record.

    Returns:
        Source start and end instants, equal when the span declares no end.

    Raises:
        VendorTraceFormatError: The span declares no readable start time.
    """
    start_ns = span.get("start", span.get("start_ns"))
    if isinstance(start_ns, int | float) and not isinstance(start_ns, bool):
        start_seconds = float(start_ns) / 1_000_000_000
        duration_ns = span.get("duration", span.get("duration_ns"))
        end_seconds: float | None = None
        if isinstance(duration_ns, int | float) and not isinstance(duration_ns, bool):
            end_seconds = start_seconds + float(duration_ns) / 1_000_000_000
        else:
            end_ns = span.get("end", span.get("end_ns"))
            if isinstance(end_ns, int | float) and not isinstance(end_ns, bool):
                end_seconds = float(end_ns) / 1_000_000_000
        return source_interval(
            start_seconds,
            end_seconds,
            start_label="Datadog span start",
            end_label="Datadog span end",
        )
    return source_interval(
        span.get("start_time", span.get("startTime")),
        span.get("end_time", span.get("endTime")),
        start_label="Datadog span start_time",
        end_label="Datadog span end_time",
    )


def _usage(span: JsonObject, llmobs: JsonObject, attributes: JsonObject) -> VendorTokenUsage | None:
    """Read declared token accounting from a Datadog span.

    Args:
        span: Datadog span record.
        llmobs: Decoded LLM Observability payload.
        attributes: Declared span metadata.

    Returns:
        Declared usage, or ``None`` when the span declares no complete accounting.

    Raises:
        VendorTraceFormatError: A declared token count is not a non-negative integer.
    """
    candidates: list[JsonValue | None] = [span.get("metrics")]
    metrics = llmobs.get("metrics")
    candidates.append(metrics if isinstance(metrics, dict) else None)
    candidates.append(attributes.get("metrics"))
    candidates.append(attributes.get("usage"))
    candidates.append(attributes)
    for candidate in candidates:
        usage = declared_usage(candidate)
        if usage is not None:
            return usage
    return None


def _failure_message(span: JsonObject, attributes: JsonObject) -> str | None:
    """Read a Datadog span failure message when the span reports an error.

    Args:
        span: Datadog span record.
        attributes: Declared span metadata.

    Returns:
        Declared error text, or ``None`` when the span reports no error.
    """
    error = span.get("error")
    if isinstance(error, int) and not isinstance(error, bool) and error != 0:
        return (
            declared_error_message({**span, **attributes}, keys=_ERROR_KEYS, label="Datadog span")
            or f"Datadog span failed with error {error}"
        )
    if isinstance(error, str) and error.strip() and error.strip() != "0":
        return (
            declared_error_message({**span, **attributes}, keys=_ERROR_KEYS, label="Datadog span")
            or f"Datadog span failed with error {error.strip()}"
        )
    failure = declared_error_message(span, keys=_ERROR_KEYS, label="Datadog span")
    if failure is not None:
        return failure
    return declared_error_message(attributes, keys=_ERROR_KEYS, label="Datadog span")


def _extensions(span: JsonObject, attributes: JsonObject) -> JsonObject:
    """Read approved EXP extensions and the Datadog application from one span.

    Args:
        span: Datadog span record.
        attributes: Declared span metadata.

    Returns:
        Approved extension attributes for this record.
    """
    extensions = approved_extensions(span)
    extensions.update(approved_extensions(attributes))
    llmobs = _llmobs_meta(span)
    tags = llmobs.get("tags")
    if isinstance(tags, dict):
        extensions.update(approved_extensions(tags))
        app = tags.get("ml_app", tags.get("mlApp"))
        if isinstance(app, str) and app.strip() and "exp.datadog.app" not in extensions:
            extensions["exp.datadog.app"] = app.strip()
    return extensions


DATADOG_SOURCE: VendorSource[JsonObject] = VendorSource(
    vendor=VENDOR,
    records=record_flattener(
        vendor=VENDOR,
        wrapper_keys=("spans", "traces", "data", "results", "items", "events"),
        record_keys=("trace_id", "traceId", "span_id", "spanId"),
    ),
    convert=_span_observation,
)
