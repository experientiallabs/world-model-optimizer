"""LangSmith adapter: turn a LangSmith run-tree export into `Trace`s.

LangSmith (LangChain's tracing product) does NOT export OTLP spans. It models a trace as a tree of
**runs**, where each run is a node typed by `run_type`
(`tool | chain | llm | retriever | embedding | prompt | parser`). An exported run (from
`POST /api/v1/runs/query` — the list endpoint is a POST with a JSON filter body, NOT a GET — or the
SDK `Client.list_runs`) looks roughly like:

    {"id": "<run uuid>", "trace_id": "<trace grouping uuid>", "parent_run_id": "<uuid|null>",
     "run_type": "llm" | "tool" | "chain" | "retriever" | "embedding" | "prompt" | "parser",
     "name": "ChatOpenAI",
     "inputs": {...}, "outputs": {...},
     "start_time": "2026-01-01T00:00:00.000000", "end_time": "...",
     "error": null | "<traceback str>", "extra": {...}}

Because this is not an OTLP/OpenInference span shape, the adapter overrides `spans_from_payload`
(like `wmh.ingest.messages` / `wmh.ingest.langfuse`) and emits `SpanRecord`s in the **OTel-GenAI
vocabulary** so the shared classifier/normalizer (`wmh.ingest.normalize`) does the pairing /
state / metadata work. Each run maps to zero or more spans:

  - `run_type == "llm"`: if the outputs carry tool calls, emit one `chat` ACTION span per call with
    `{"gen_ai.tool.name", "gen_ai.tool.call.arguments"}`; otherwise emit one plain `chat` message
    span with `{"gen_ai.completion": <text>}`.
  - `run_type == "tool"`: emit one `execute_tool` RESULT span with
    `{"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": <name>,
      "gen_ai.tool.message": <outputs as text>}`.  The normalizer pairs it with the preceding action
    span (the llm run's tool call), backfilling name/args from here if the action lacked them.
  - `run_type in {"chain", "retriever", ...}`: skipped (not directly actionable). A run we cannot
    interpret is skipped rather than crashing the ingest.

Where LangChain hides tool calls (these paths are version-dependent and best-effort; we dig all of
them and degrade gracefully):
  - `outputs["generations"][i]["message"]["kwargs"]["tool_calls"]`  (LCEL ChatGeneration dump)
  - `outputs["generations"][i]["message"]["kwargs"]["additional_kwargs"]["tool_calls"]`  (OpenAI)
  - `outputs["generations"][i][j]["message"]...`  (nested list-of-lists generations)
  - `outputs["tool_calls"]` / `outputs["message"]...`  (flatter dumps)
A LangChain `tool_calls` entry is either the normalized shape `{"name", "args": {...}, "id"}` or the
OpenAI shape `{"id", "function": {"name", "arguments": "<json str>"}}`; both are handled.

The LLM completion text is dug from `generations[i].text`, `generations[i].message.kwargs.content`,
or `outputs["output"|"content"|"text"]`. A tool run's output is dug from
`outputs["output"]`, else the whole `outputs`. The task (`gen_ai.prompt`) is the first human/user
input, dug from a run's `inputs` (chat `messages`, a `messages` list, or `inputs["input"]`).

Ordering: runs are ordered by `start_time` (ISO-8601 -> epoch microseconds); only monotonicity
within a trace matters (the normalizer sorts by `start_nano`), so an absent/unparseable timestamp
degrades to the run's list index.

Accepted file shapes (`from_file`): a single run object, a JSON array of runs, a `{"runs": [...]}`
wrapper, or JSONL (one run per line). Grouping is by `trace_id` (falling back to `id` for a root
run that omits it).

Pull: a live pull via the `langsmith` SDK is not implemented here; export to a file and use
`from_file`. The `BaseTraceAdapter` default raises a friendly error pointing at `--file`, so the
config gate stays SDK-free.
"""

from __future__ import annotations

import hashlib

from pydantic import JsonValue

from wmh.core.types import JsonObject
from wmh.ingest.adapter import register_adapter
from wmh.ingest.base import BaseTraceAdapter
from wmh.ingest.normalize import SpanRecord, SyntheticSpanEmitter, as_text, iso_to_ordinal


def _as_str(value: JsonValue) -> str:
    return value if isinstance(value, str) else ""


def _is_error(run: JsonObject) -> bool:
    """A non-null/non-empty `error` marks the run as failed; an empty string is NOT an error.

    Some LangSmith dumps set `error: ""` on a successful run, so a bare `is not None` check would
    misclassify it as a failure.
    """
    error = run.get("error")
    if error is None:
        return False
    if isinstance(error, str):
        return bool(error.strip())
    return bool(error)


def _start_ordinal(run: JsonObject, fallback: int) -> int:
    """Monotonic ordering key from the run's `start_time` (shared helper; UTC-safe)."""
    return iso_to_ordinal(run.get("start_time"), fallback)


def _generations(outputs: JsonObject) -> list[JsonObject]:
    """Flatten `outputs["generations"]` (a list, or a list-of-lists) into generation dicts."""
    raw = outputs.get("generations")
    if not isinstance(raw, list):
        return []
    flat: list[JsonObject] = []
    for item in raw:
        if isinstance(item, dict):
            flat.append(item)
        elif isinstance(item, list):  # list-of-lists (one inner list per prompt)
            flat.extend(g for g in item if isinstance(g, dict))
    return flat


def _message_kwargs(generation: JsonObject) -> JsonObject:
    """The `message.kwargs` dict of a ChatGeneration dump (empty when absent)."""
    message = generation.get("message")
    if isinstance(message, dict):
        kwargs = message.get("kwargs")
        if isinstance(kwargs, dict):
            return kwargs
    return {}


def _tool_calls_in(container: JsonObject) -> list[JsonObject]:
    """Pull `tool_calls` from a dict, checking `tool_calls` then `additional_kwargs.tool_calls`."""
    calls: list[JsonObject] = []
    raw = container.get("tool_calls")
    if isinstance(raw, list):
        calls.extend(tc for tc in raw if isinstance(tc, dict))
    extra = container.get("additional_kwargs")
    if isinstance(extra, dict):
        raw_extra = extra.get("tool_calls")
        if isinstance(raw_extra, list):
            calls.extend(tc for tc in raw_extra if isinstance(tc, dict))
    return calls


def _llm_tool_calls(outputs: JsonObject) -> list[JsonObject]:
    """Dig tool calls out of an llm run's `outputs`, across the common LangChain locations."""
    calls: list[JsonObject] = []
    for generation in _generations(outputs):
        calls.extend(_tool_calls_in(_message_kwargs(generation)))
    # Flatter dumps: outputs.tool_calls or outputs.message.kwargs.tool_calls.
    calls.extend(_tool_calls_in(outputs))
    message = outputs.get("message")
    if isinstance(message, dict):
        kwargs = message.get("kwargs")
        if isinstance(kwargs, dict):
            calls.extend(_tool_calls_in(kwargs))
    return calls


def _call_name_args(tool_call: JsonObject) -> tuple[str, str]:
    """(name, raw-arguments-json) from a LangChain-normalized or OpenAI-shaped tool call."""
    fn = tool_call.get("function")
    if isinstance(fn, dict):  # OpenAI shape: {"function": {"name", "arguments": "<json str>"}}
        name = fn.get("name")
        args = fn.get("arguments")
    else:  # LangChain-normalized shape: {"name", "args": {...}, "id"}
        name = tool_call.get("name")
        args = tool_call.get("args")
        if args is None:
            args = tool_call.get("arguments")
    name_s = name if isinstance(name, str) else ""
    args_s = args if isinstance(args, str) else as_text(args)
    return name_s, args_s


def _llm_completion(outputs: JsonObject) -> str:
    """Dig the assistant text out of an llm run's `outputs` (best-effort across dump shapes)."""
    for generation in _generations(outputs):
        text = generation.get("text")
        if isinstance(text, str) and text:
            return text
        content = _message_kwargs(generation).get("content")
        if isinstance(content, str) and content:
            return content
    for key in ("output", "content", "text"):
        value = outputs.get(key)
        if isinstance(value, str) and value:
            return value
    return as_text(outputs)


def _tool_output_text(outputs: JsonValue) -> str:
    """A tool run's result text: `outputs["output"]` if present, else the whole outputs."""
    if isinstance(outputs, dict):
        value = outputs.get("output")
        if value is not None:
            return as_text(value)
    return as_text(outputs)


def _tool_run_name(run: JsonObject) -> str:
    """A tool name for a `tool` run: explicit fields first, else the run name."""
    for key in ("tool_name", "name"):
        value = run.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _first_user_text(inputs: JsonValue) -> str | None:
    """Dig the first human/user input out of a run's `inputs` (the trace task)."""
    if isinstance(inputs, str):
        return inputs or None
    if not isinstance(inputs, dict):
        return None
    messages = inputs.get("messages")
    text = _first_user_in_messages(messages)
    if text is not None:
        return text
    for key in ("input", "question", "query", "text"):
        value = inputs.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _first_user_in_messages(messages: JsonValue) -> str | None:
    """First human/user message content in a (possibly nested) LangChain messages list."""
    if not isinstance(messages, list):
        return None
    for message in messages:
        # Nested list-of-lists (one inner list per prompt) — recurse.
        if isinstance(message, list):
            found = _first_user_in_messages(message)
            if found is not None:
                return found
            continue
        if not isinstance(message, dict):
            continue
        role = _message_role(message)
        content = _message_content(message)
        if role in {"human", "user"} and content:
            return content
    return None


def _message_role(message: JsonObject) -> str:
    """Role of a LangChain/OpenAI message dict (`role`, `type`, or a serialized class id)."""
    for key in ("role", "type"):
        value = message.get(key)
        if isinstance(value, str) and value:
            return value.lower()
    # Serialized LangChain message: {"id": [..., "HumanMessage"], "kwargs": {...}}.
    cid = message.get("id")
    if isinstance(cid, list) and cid:
        last = cid[-1]
        if isinstance(last, str):
            return last.lower().removesuffix("message")
    return ""


def _message_content(message: JsonObject) -> str:
    """Text content of a LangChain/OpenAI message dict (top-level or under `kwargs`)."""
    content = message.get("content")
    if isinstance(content, str) and content:
        return content
    kwargs = message.get("kwargs")
    if isinstance(kwargs, dict):
        nested = kwargs.get("content")
        if isinstance(nested, str) and nested:
            return nested
    return ""


class LangSmithAdapter(BaseTraceAdapter):
    """Map a LangSmith run-tree export into normalized `Trace`s. No SDK; pure JSON."""

    name = "langsmith"

    def spans_from_payload(self, payload: JsonValue) -> list[SpanRecord]:
        """Map one payload (single run, list, `{runs:[...]}`) to ordered `SpanRecord`s by trace."""
        runs = self._runs(payload)
        # Group by trace_id so we can assign a per-trace monotonic ordinal and set the task once.
        by_trace: dict[str, list[JsonObject]] = {}
        for run in runs:
            by_trace.setdefault(self._trace_id(run), []).append(run)

        spans: list[SpanRecord] = []
        for trace_id, trace_runs in by_trace.items():
            spans.extend(self._spans_for_trace(trace_id, trace_runs))
        return spans

    def _runs(self, payload: JsonValue) -> list[JsonObject]:
        """Normalize a payload into a flat list of run objects.

        Accepts a single run, a bare list of runs, or a `{"runs": [...]}` wrapper.
        """
        if isinstance(payload, list):
            out: list[JsonObject] = []
            for item in payload:
                out.extend(self._runs(item))
            return out
        if not isinstance(payload, dict):
            return []
        wrapped = payload.get("runs")
        if isinstance(wrapped, list) and "run_type" not in payload:
            out = []
            for item in wrapped:
                out.extend(self._runs(item))
            return out
        # A run object: it has an id (and usually run_type). Be permissive.
        if "id" in payload or "run_type" in payload:
            return [payload]
        return []

    def _spans_for_trace(self, trace_id: str, runs: list[JsonObject]) -> list[SpanRecord]:
        # Order by start_time; ties (or absent timestamps) keep input order via the index fallback.
        indexed = list(enumerate(runs))
        indexed.sort(key=lambda pair: (_start_ordinal(pair[1], pair[0]), pair[0]))

        task = self._trace_task([run for _, run in indexed])

        emitter = SyntheticSpanEmitter(
            trace_id, first_attrs={"gen_ai.prompt": task} if task is not None else None
        )
        for _, run in indexed:
            run_type = _as_str(run.get("run_type")).lower()
            error = _is_error(run)
            outputs = run.get("outputs")
            out_obj: JsonObject = outputs if isinstance(outputs, dict) else {}

            if run_type == "llm":
                calls = _llm_tool_calls(out_obj)
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
                        {"gen_ai.completion": _llm_completion(out_obj)}, tool=False, error=error
                    )
            elif run_type == "tool":
                emitter.emit(
                    {
                        "gen_ai.tool.name": _tool_run_name(run),
                        "gen_ai.tool.message": _tool_output_text(outputs),
                    },
                    tool=True,
                    error=error,
                )
            # chain / retriever / unknown run types are not directly actionable -> skipped.
        return emitter.spans

    def _trace_task(self, runs: list[JsonObject]) -> str | None:
        """First human/user input across a trace's runs (ordered) -> the task text."""
        for run in runs:
            text = _first_user_text(run.get("inputs"))
            if text is not None:
                return text
        return None

    def _trace_id(self, run: JsonObject) -> str:
        """Grouping key: `trace_id`, else the run's own `id` (a root run may omit trace_id)."""
        for key in ("trace_id", "id"):
            value = run.get(key)
            if isinstance(value, str) and value:
                return value
        return hashlib.sha256(as_text(run).encode()).hexdigest()[:32]


register_adapter(LangSmithAdapter())
