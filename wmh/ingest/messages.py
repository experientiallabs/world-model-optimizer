"""Chat / tool-call converter: turn recorded LLM conversations into `Trace`s (no SDK, no spans).

Not every source is a span exporter. The most universal trace people already have is a list of chat
messages with tool calls — the OpenAI Chat Completions shape, which LangChain, the Anthropic SDK
(after a light dump), and most agent frameworks can emit:

    {"messages": [
        {"role": "user", "content": "what's the weather in Paris?"},
        {"role": "assistant", "content": "let me check",
         "tool_calls": [{"id": "c1", "function": {"name": "get_weather",
                                                  "arguments": "{\"city\": \"Paris\"}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "18C and sunny"},
        {"role": "assistant", "content": "It's 18C and sunny in Paris."}
    ]}

This adapter maps each assistant **tool call** to an Action and the matching `role:"tool"` message
(by `tool_call_id`, else the next tool message in order) to its Observation — exactly the
`(action) -> observation` step the harness scores. A trailing assistant message with no tool call
becomes a final message Step with an empty observation. The first user message is the trace `task`.

It builds `SpanRecord`s in the OTel-GenAI vocabulary and hands them to the shared normalizer, so it
reuses the same pairing/state/metadata logic as every other adapter rather than re-implementing it.

Accepted file shapes (`from_file`):
  - a single conversation object `{"messages": [...]}` (optionally `{"id"/"trace_id", "metadata"}`)
  - a JSON array of such conversation objects
  - JSONL: one conversation object per line
  - a bare list of messages `[{"role": ...}, ...]` (treated as one conversation)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import JsonValue

from wmh.core.types import JsonObject, Trace
from wmh.ingest.adapter import VendorPull, register_adapter
from wmh.ingest.normalize import (
    SpanRecord,
    as_text,
    openai_call_name_args,
    spans_to_traces,
)


def _hash_id(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]


def _tool_calls(message: JsonObject) -> list[JsonObject]:
    raw = message.get("tool_calls")
    if not isinstance(raw, list):
        return []
    return [tc for tc in raw if isinstance(tc, dict)]


def _spans_for_conversation(
    messages: list[JsonValue], trace_id: str, metadata: JsonObject
) -> list[SpanRecord]:
    """Build ordered action/observation SpanRecords (GenAI vocab) for one conversation."""
    # Index tool results by tool_call_id; fall back to consumption in order for results lacking one.
    results_by_id: dict[str, str] = {}
    ordered_results: list[str] = []
    task: str | None = None
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "user" and task is None:
            task = as_text(m.get("content"))
        if role == "tool":
            content = as_text(m.get("content"))
            tcid = m.get("tool_call_id")
            if isinstance(tcid, str):
                results_by_id[tcid] = content
            ordered_results.append(content)

    spans: list[SpanRecord] = []
    ordinal = 0
    unmatched = list(ordered_results)

    def _emit(attrs: JsonObject, *, tool: bool, error: bool = False) -> None:
        nonlocal ordinal
        if ordinal == 0 and task is not None:
            attrs.setdefault("gen_ai.prompt", task)
        if ordinal == 0 and metadata:
            attrs.setdefault("wmh.trace.metadata", json.dumps(metadata))
        spans.append(
            SpanRecord(
                trace_id=trace_id,
                span_id=f"{trace_id[:12]}{ordinal:06x}{'t' if tool else 'a'}",
                name="execute_tool" if tool else "chat",
                start_nano=ordinal,
                attributes={"gen_ai.operation.name": "execute_tool" if tool else "chat", **attrs},
                status_error=error,
            )
        )
        ordinal += 1

    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        calls = _tool_calls(m)
        if calls:
            for tc in calls:
                name, args = openai_call_name_args(tc)
                _emit({"gen_ai.tool.name": name, "gen_ai.tool.call.arguments": args}, tool=False)
                tcid = tc.get("id")
                if isinstance(tcid, str) and tcid in results_by_id:
                    result = results_by_id[tcid]
                    if result in unmatched:
                        unmatched.remove(result)
                elif unmatched:
                    result = unmatched.pop(0)
                else:
                    result = ""
                _emit({"gen_ai.tool.message": result}, tool=True)
        else:
            # A plain assistant message turn (e.g. the final answer): a message Action, no tool obs.
            content = m.get("content")
            if content is not None:
                _emit({"gen_ai.completion": as_text(content)}, tool=False)
    return spans


def _conversation_records(payload: JsonValue) -> list[tuple[str, list[JsonValue], JsonObject]]:
    """Normalize a payload into a list of (trace_id, messages, metadata) conversations."""
    out: list[tuple[str, list[JsonValue], JsonObject]] = []
    if isinstance(payload, list):
        # Either a list of conversation objects, or a bare message list (one conversation).
        if payload and all(isinstance(x, dict) and "role" in x for x in payload):
            out.append((_hash_id(as_text(payload)), payload, {}))
            return out
        for item in payload:
            out.extend(_conversation_records(item))
        return out
    if not isinstance(payload, dict):
        return out
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return out
    tid = payload.get("trace_id") or payload.get("id")
    trace_id = tid if isinstance(tid, str) and tid else _hash_id(as_text(messages))
    # 32-hex normalize so trace ids are uniform regardless of source id format.
    if len(trace_id) != 32 or any(c not in "0123456789abcdef" for c in trace_id.lower()):
        trace_id = _hash_id(trace_id)
    meta = payload.get("metadata")
    metadata: JsonObject = meta if isinstance(meta, dict) else {}
    out.append((trace_id, messages, metadata))
    return out


class ChatMessagesAdapter:
    """Convert recorded chat/tool-call conversations (OpenAI-style) into `Trace`s. No SDK."""

    name = "chat-json"

    def from_file(self, path: str) -> list[Trace]:
        text = Path(path).read_text(encoding="utf-8")
        payloads: list[JsonValue] = []
        try:
            payloads.append(json.loads(text))
        except json.JSONDecodeError:
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payloads.append(json.loads(stripped))
                except json.JSONDecodeError:
                    continue
        spans: list[SpanRecord] = []
        for payload in payloads:
            for trace_id, messages, metadata in _conversation_records(payload):
                spans.extend(_spans_for_conversation(messages, trace_id, metadata))
        return spans_to_traces(spans, source=f"chat-json:{path}")

    def from_vendor(self, pull: VendorPull) -> list[Trace]:
        raise ValueError(
            "chat-json converts local conversation files; it has no vendor API. "
            "Use `from_file` with an exported messages JSON/JSONL."
        )


register_adapter(ChatMessagesAdapter())
