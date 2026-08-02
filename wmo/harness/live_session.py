# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""Interactive live sessions: the host engine that drives one long-lived pi runner.

A *session* differs from an *episode* (`runner_link.RunnerLink`) in three ways: it is multi-turn
(the user sends messages, steers a running turn, and interrupts, over one persistent transcript),
its tools execute for REAL against a live filesystem/shell rather than being answered by a
world-model simulation, and it emits a stream of typed events (assistant text, tool calls, tool
output, state changes) that a UI renders live. Everything else is the RunnerLink model unchanged:
the worker LLM is answered host-side so credentials never reach the runner, and the wire is the
same length-prefixed / base64-line frame `Channel`.

The engine is transport- and platform-agnostic. It answers `llm_request` frames host-side via a
fully-configured `ToolCallingProvider`'s `complete_chat` (Bedrock or Azure — provider-agnostic per
wmo #142) or an injected `worker_fn` (tests), and answers `tool_request` frames with an injected
`ToolExecutor` (the platform runs bash/read_file/write_file against the session's E2B sandbox
filesystem; a CLI would run them against a local directory). Because the host answers every
`llm_request` and every `tool_request`, it *is* the trusted observer of what the agent did — the
feed is derived from the frames the host itself produced/answered, so a compromised runner cannot
narrate a different story than the actions it actually took.

New session frames (additive to the RunnerLink vocabulary; unknown types are ignored on both
sides, so eval episodes are unaffected):

  host -> runner
    session_start {session_id, system, tools, files, turn_cap, max_output_tokens, temperature,
                   conversation_scope}
    user_message  {msg_id, text}
    abort         {reason}
    ping          {nonce}
    shutdown                       (reused; ends the runner process)
  runner -> host
    hello         {...}            (reused)
    llm_request   {req_id, openai_body}   (reused)
    tool_request  {req_id, name, arguments}  (reused)
    state         {status: "idle"|"running", turns, reason?, cleared_steers?, msg_id?}
    pong          {nonce}
    episode_error {note}           (reused; a fatal runner error ends the session)
"""

from __future__ import annotations

import queue
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from llm_waterfall import ChatRequest, ChatResponse
from pydantic import JsonValue

from wmo.core.types import JsonObject
from wmo.harness.runner_link import Channel, TokenUsage, WorkerFn, params_schema
from wmo.harness.runtime import DEFAULT_MAX_OUTPUT_TOKENS
from wmo.harness.tools import READ_SKILL, SUBMIT, ToolSpec
from wmo.providers.base import ToolCallingProvider, structured_token_usage

# The action budget a single user turn may spend before the runner is told to stop — the live
# analogue of `HostEpisode.max_env_actions`, so a champion optimized under that pressure behaves
# the same way. It resets on every user message.
DEFAULT_ACTIONS_PER_TURN = 40
# Turns a single user message may run before the runner aborts it (distinct from a user interrupt).
# 3x the doc default (20) — real tasks against a live filesystem legitimately run longer than the
# short world-model-simulation tasks the harness was scored on.
DEFAULT_TURN_CAP = 60

# --------------------------------------------------------------------------------------------------
# Events: the typed stream the engine emits; a UI renders these, a store persists them.
# --------------------------------------------------------------------------------------------------
EventKind = Literal[
    "user_message",
    "assistant_message",
    "tool_call",
    "tool_output",
    "tool_result",
    "submit",
    "state",
    "error",
]
ConversationScope = Literal["session", "turn"]


@dataclass(frozen=True)
class SessionEvent:
    """One thing that happened in a session, in the order the host observed it."""

    kind: EventKind
    payload: JsonObject


EventSink = Callable[[SessionEvent], None]

# (name, arguments, emit_output) -> (content, is_error). `emit_output(stream, chunk)` streams
# partial stdout/stderr so the UI shows a long command's output as it runs; the returned content is
# the final (already length-capped by the executor) observation the agent sees.
OutputEmitter = Callable[[str, str], None]
ToolExecutor = Callable[[str, JsonObject, OutputEmitter], "ToolOutcome"]


@dataclass(frozen=True)
class ToolOutcome:
    """The result of executing one real tool call: what the agent sees, and whether it failed."""

    content: str
    is_error: bool = False
    truncated: bool = False


# --------------------------------------------------------------------------------------------------
# Inbox: user intents the driver feeds in from another thread, drained into frames on each pump.
# --------------------------------------------------------------------------------------------------
@dataclass
class _UserMessage:
    text: str
    msg_id: str


@dataclass
class _Interrupt:
    reason: str = "user_interrupt"


@dataclass
class _End:
    pass


_Intent = _UserMessage | _Interrupt | _End


class LiveSession:
    """Drives one interactive pi session over a `Channel`, emitting a typed event stream.

    Lifecycle: `start()` sends `session_start` and waits for the runner's first `state:idle`; then
    the driver loops calling `pump(timeout)` while feeding intents via `send_user_message`,
    `interrupt`, and `end` (thread-safe — a driver may enqueue from a command-poll thread). `pump`
    drains the inbox to frames, then processes one inbound frame (answering `llm_request` /
    `tool_request`, emitting events). `closed` flips once the runner process ends or `end()` is
    honored. `worker_usage` accumulates the metered worker-LLM spend across the whole session.
    """

    def __init__(
        self,
        channel: Channel,
        *,
        tools: list[ToolSpec],
        execute_tool: ToolExecutor,
        on_event: EventSink,
        files: dict[str, str] | None = None,
        system_prompt: str = "",
        skill_bodies: dict[str, str] | None = None,
        provider: ToolCallingProvider | None = None,
        worker_fn: WorkerFn | None = None,
        actions_per_turn: int = DEFAULT_ACTIONS_PER_TURN,
        turn_cap: int = DEFAULT_TURN_CAP,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float = 0.7,
        conversation_scope: ConversationScope = "session",
    ) -> None:
        self._channel = channel
        self._tools = list(tools)
        self._execute_tool = execute_tool
        self._on_event = on_event
        self._files = dict(files or {})
        self._system_prompt = system_prompt
        self._skill_bodies = dict(skill_bodies or {})
        if self._skill_bodies and READ_SKILL.name not in {tool.name for tool in self._tools}:
            self._tools.append(READ_SKILL)
        # The worker LLM is answered host-side by a fully-configured provider's
        # `complete_chat` (Bedrock or Azure — provider-agnostic per #142), or an
        # injected `worker_fn` (tests). Left unset, an `llm_request` is answered
        # with an error instead of crashing the host.
        if worker_fn is not None:
            self._worker_fn: WorkerFn | None = worker_fn
        elif provider is not None:
            self._worker_fn = provider.complete_chat
        else:
            self._worker_fn = None
        self._actions_per_turn = actions_per_turn
        self._turn_cap = turn_cap
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be >= 1")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        if conversation_scope not in ("session", "turn"):
            raise ValueError("conversation_scope must be 'session' or 'turn'")
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._conversation_scope = conversation_scope

        self._inbox: queue.Queue[_Intent] = queue.Queue()
        self._session_id = uuid.uuid4().hex
        self._status: str = "starting"
        self._closed = False
        self._failure_message: str | None = None
        self._actions_this_turn = 0
        self._aborting = False
        self._pending_ping: str | None = None
        self.worker_usage = TokenUsage()

    # -- lifecycle -------------------------------------------------------------------------------

    @property
    def status(self) -> str:
        """The runner's last-known agent state: "starting" | "idle" | "running" | "ended"."""
        return self._status

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def failure_message(self) -> str | None:
        """Return the runner or channel error that closed this session, when one exists."""
        return self._failure_message

    def start(self, hello_timeout: float = 60.0) -> None:
        """Send `session_start` and block until the runner reports its first idle state."""
        self._channel.send(
            {
                "type": "session_start",
                "session_id": self._session_id,
                "system": self._system_prompt,
                "tools": self._tool_specs(),
                "files": self._files,
                "turn_cap": self._turn_cap,
                "max_output_tokens": self._max_output_tokens,
                "temperature": self._temperature,
                "conversation_scope": self._conversation_scope,
            }
        )
        # The runner sends `state:idle` once the agent is constructed and ready for the first
        # message; surface a fatal construction error (bad champion code) instead of hanging.
        deadline = _Deadline(hello_timeout)
        while not deadline.expired():
            frame = self._recv(deadline.remaining())
            if frame is None:
                break
            if self._handle_frame(frame):
                if self._status in ("idle", "running"):
                    return
                if self._closed:
                    break
        if self._failure_message is not None:
            raise RuntimeError(f"live session runner did not become ready: {self._failure_message}")
        raise RuntimeError("live session runner did not become ready")

    # -- driver-facing intents (thread-safe) -----------------------------------------------------

    def send_user_message(self, text: str) -> str:
        """Queue a user message; returns the msg_id echoed on its `user_message` event."""
        msg_id = uuid.uuid4().hex
        self._inbox.put(_UserMessage(text=text, msg_id=msg_id))
        return msg_id

    def interrupt(self, reason: str = "user_interrupt") -> None:
        """Queue an interrupt: abort the current run (does not end the session)."""
        self._inbox.put(_Interrupt(reason=reason))

    def flush_pending_intents(self) -> None:
        """Send queued controls without consuming another runner frame.

        The driver uses this when it must abort and retire a session immediately. Calling
        :meth:`pump` would also read one inbound frame, which could start another synchronous
        provider or tool operation after cancellation was already observed.
        """
        if not self._closed:
            self._drain_inbox()

    def end(self) -> None:
        """Queue a graceful end: abort any run, then shut the runner down."""
        self._inbox.put(_End())

    def ping(self) -> None:
        """Send a liveness ping; a returned `pong` proves the runner event loop is alive."""
        nonce = uuid.uuid4().hex
        self._pending_ping = nonce
        self._safe_send({"type": "ping", "nonce": nonce})

    # -- pump ------------------------------------------------------------------------------------

    def pump(self, timeout: float = 0.2) -> bool:
        """Drain queued intents to frames, then process one inbound frame. False once closed."""
        if self._closed:
            return False
        self._drain_inbox()
        if self._closed:
            return False
        frame = self._recv(timeout)
        if frame is None:
            return not self._closed
        self._handle_frame(frame)
        return not self._closed

    # -- internals -------------------------------------------------------------------------------

    def _drain_inbox(self) -> None:
        while True:
            try:
                intent = self._inbox.get_nowait()
            except queue.Empty:
                return
            if isinstance(intent, _UserMessage):
                self._actions_this_turn = 0
                # NB: do not clear `_aborting` here — a new message can be drained before
                # the cancelled turn's in-flight submit frame is read, and clearing now
                # would let that stale submit emit. The runner's next `state` frame (the
                # real turn boundary) clears it in `_on_state`.
                self._emit("user_message", {"msg_id": intent.msg_id, "text": intent.text})
                self._safe_send(
                    {"type": "user_message", "msg_id": intent.msg_id, "text": intent.text}
                )
            elif isinstance(intent, _Interrupt):
                # Mark the current turn as aborting so a `submit` tool_request that was
                # already in flight when the interrupt fired does not emit a final submit
                # event: the host, not the runner, owns event emission, so this gate is
                # the only place the race can be closed. Cleared at the next turn boundary.
                self._aborting = True
                self._safe_send({"type": "abort", "reason": intent.reason})
            else:  # _End
                self._safe_send({"type": "abort", "reason": "shutdown"})
                self._safe_send({"type": "shutdown"})
                self._mark_closed("ended")

    def _handle_frame(self, frame: JsonObject) -> bool:
        kind = frame.get("type")
        if kind == "llm_request":
            self._answer_llm(frame)
        elif kind == "tool_request":
            self._answer_tool(frame)
        elif kind == "state":
            self._on_state(frame)
        elif kind == "pong":
            if frame.get("nonce") == self._pending_ping:
                self._pending_ping = None
        elif kind == "episode_error":
            note = frame.get("note")
            self._failure_message = note if isinstance(note, str) else "runner error"
            self._emit("error", {"message": self._failure_message})
            self._mark_closed("failed")
        elif kind == "hello":
            pass  # the pool already consumed the handshake; a duplicate is harmless
        # unknown frames are ignored (forward-compatible)
        return kind is not None

    def _answer_llm(self, frame: JsonObject) -> None:
        req_id = frame.get("req_id")
        body = frame.get("openai_body")
        if self._worker_fn is None:
            self._emit("error", {"message": "live session has no worker configured"})
            self._safe_send(
                {"type": "llm_response", "req_id": req_id, "error": "no worker configured"}
            )
            return
        started = time.perf_counter()
        self.worker_usage.calls += 1
        try:
            request_body = dict(body) if isinstance(body, dict) else {}
            request_body["temperature"] = self._temperature
            request = ChatRequest.model_validate(request_body)
            completion = self._worker_fn(request)
            self._meter(completion)
            self._emit_assistant(completion)
            self._safe_send(
                {"type": "llm_response", "req_id": req_id, "completion": completion.wire_payload()}
            )
        except Exception as exc:  # noqa: BLE001 - report to the runner, never crash the host
            self._emit("error", {"message": f"worker LLM error: {exc}"})
            self._safe_send({"type": "llm_response", "req_id": req_id, "error": str(exc)})
        finally:
            self.worker_usage.call_seconds.append(time.perf_counter() - started)

    def _answer_tool(self, frame: JsonObject) -> None:
        req_id = frame.get("req_id")
        name = frame.get("name")
        name = name if isinstance(name, str) else ""
        args = frame.get("arguments")
        args = cast("JsonObject", args) if isinstance(args, dict) else {}
        call_id = uuid.uuid4().hex

        if name == SUBMIT.name:
            # If the turn is being interrupted, do NOT emit a final submit for it —
            # the answer belongs to a cancelled run. Still respond so the runner's
            # submit tool does not hang; the aborted run ends via `state:idle`.
            if not self._aborting:
                answer = args.get("answer")
                self._emit("submit", {"answer": answer if isinstance(answer, str) else ""})
            self._respond_tool(req_id, "submitted", is_error=False)
            return

        if self._aborting:
            # The turn is being interrupted; do NOT run a real side-effecting tool
            # (bash / write_file) for a cancelled run. Respond so the runner's tool
            # call does not hang; the aborted run ends via `state:idle`.
            self._respond_tool(req_id, "interrupted", is_error=True)
            return

        self._emit("tool_call", {"call_id": call_id, "name": name, "arguments": args})
        known = {t.name for t in self._tools}
        if name not in known:
            outcome = ToolOutcome(content=f"tool {name!r} not available", is_error=True)
        elif name == READ_SKILL.name:
            outcome = self._read_skill(args)
        elif self._actions_this_turn >= self._actions_per_turn:
            outcome = ToolOutcome(content="environment action budget exhausted", is_error=True)
        else:
            self._actions_this_turn += 1

            def emit_output(stream: str, chunk: str) -> None:
                if chunk:
                    self._emit("tool_output", {"call_id": call_id, "stream": stream, "text": chunk})

            try:
                outcome = self._execute_tool(name, args, emit_output)
            except Exception as exc:  # noqa: BLE001 - a tool failure must not crash the host loop
                # A transient sandbox/FS error becomes an error result, so the runner
                # always gets a tool_response and pump() keeps running.
                outcome = ToolOutcome(content=f"tool {name!r} failed: {exc}", is_error=True)
        self._emit(
            "tool_result",
            {
                "call_id": call_id,
                "content": outcome.content,
                "is_error": outcome.is_error,
                "truncated": outcome.truncated,
            },
        )
        self._respond_tool(req_id, outcome.content, is_error=outcome.is_error)

    def _read_skill(self, args: JsonObject) -> ToolOutcome:
        raw_name = args.get("name")
        name = raw_name if isinstance(raw_name, str) else ""
        body = self._skill_bodies.get(name)
        if body is None:
            return ToolOutcome(content=f"no skill named {name!r}", is_error=True)
        return ToolOutcome(content=body)

    def _respond_tool(self, req_id: JsonValue, content: str, *, is_error: bool) -> None:
        self._safe_send(
            {"type": "tool_response", "req_id": req_id, "content": content, "is_error": is_error}
        )

    def _on_state(self, frame: JsonObject) -> None:
        status = frame.get("status")
        if isinstance(status, str):
            self._status = status
        # Only the terminal `idle` frame is the cancelled turn's true boundary. The
        # runner emits `state:running` at each prompt start; clearing on that (or any
        # non-idle frame) could re-enable submit emission while the turn is still
        # aborting, letting a stale in-flight `submit` surface as a final answer.
        if self._status == "idle":
            self._aborting = False
        payload: JsonObject = {"status": self._status}
        for key in ("turns", "reason", "msg_id"):
            if key in frame:
                payload[key] = frame[key]
        cleared = frame.get("cleared_steers")
        if isinstance(cleared, list) and cleared:
            payload["cleared_steers"] = cleared
        self._emit("state", payload)

    def _emit_assistant(self, completion: ChatResponse) -> None:
        if not completion.choices:
            return
        text = completion.choices[0].message.content
        if isinstance(text, str) and text.strip():
            self._emit("assistant_message", {"text": text})

    def _meter(self, completion: ChatResponse) -> None:
        reported = structured_token_usage(completion)
        self.worker_usage.input_tokens += reported.input_tokens
        self.worker_usage.output_tokens += reported.output_tokens
        self.worker_usage.cached_input_tokens += reported.cached_input_tokens
        self.worker_usage.cache_write_input_tokens += reported.cache_write_input_tokens
        self.worker_usage.reasoning_tokens += reported.reasoning_tokens
        self.worker_usage.call_input_tokens.append(reported.input_tokens)
        self.worker_usage.call_output_tokens.append(reported.output_tokens)
        self.worker_usage.call_cached_input_tokens.append(reported.cached_input_tokens)
        self.worker_usage.call_cache_write_input_tokens.append(reported.cache_write_input_tokens)

    def _tool_specs(self) -> list[JsonObject]:
        return [
            {"name": t.name, "description": t.description, "parameters": params_schema(t)}
            for t in self._tools
        ]

    def _emit(self, kind: EventKind, payload: JsonObject) -> None:
        try:
            self._on_event(SessionEvent(kind=kind, payload=payload))
        except Exception:  # noqa: BLE001 - a sink error must never stop the session loop
            pass

    def _recv(self, timeout: float | None) -> JsonObject | None:
        try:
            frame = _recv_with_timeout(self._channel, timeout)
        except TimeoutError:
            return None
        except Exception as exc:  # noqa: BLE001 - a dead runner ends the session, not the process
            if not self._closed:
                self._failure_message = str(exc)
                self._emit("error", {"message": self._failure_message})
                self._mark_closed("failed")
            return None
        if frame is None and not self._closed:
            self._mark_closed("ended")
        return frame

    def _safe_send(self, frame: JsonObject) -> None:
        if self._closed:
            return
        try:
            self._channel.send(frame)
        except Exception as exc:  # noqa: BLE001 - a broken channel ends the session cleanly
            self._failure_message = f"channel send failed: {exc}"
            self._emit("error", {"message": self._failure_message})
            self._mark_closed("failed")

    def _mark_closed(self, status: str) -> None:
        self._closed = True
        self._status = status


def _recv_with_timeout(channel: Channel, timeout: float | None) -> JsonObject | None:
    """Call `channel.recv`, passing `timeout` when the channel supports it (E2BStdioChannel does).

    The bare `Channel` protocol has a no-arg `recv`; the E2B channel accepts an optional timeout so
    the pump can wake to flush the inbox even while the runner is thinking. A channel without the
    parameter blocks — acceptable for in-process test doubles that always have a frame ready.
    """
    recv: Any = channel.recv
    try:
        return cast("JsonObject | None", recv(timeout))
    except TypeError:
        return cast("JsonObject | None", recv())


@dataclass
class _Deadline:
    seconds: float
    _remaining: float = field(init=False)

    def __post_init__(self) -> None:
        import time

        self._end = time.monotonic() + self.seconds

    def remaining(self) -> float:
        import time

        return max(0.0, self._end - time.monotonic())

    def expired(self) -> bool:
        return self.remaining() <= 0.0
