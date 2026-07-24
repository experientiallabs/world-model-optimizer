"""RunnerLink: the transport that replaces per-episode SSH + reverse tunnel for the pi runner.

The control plane (this process) holds the model credentials and the world-model session state; a
long-lived pi *runner* — local, on nucbox, or any remote box — dials the host and blocks reading
frames. One episode is driven over one bidirectional frame channel: the host sends an
`episode_start`, then answers the two callbacks the runner pushes up — `llm_request` (the worker
LLM completion, produced host-side so no creds ever reach the runner) and `tool_request` (the
environment tool call, routed to the `AgentEnvironment` / world model) — until `done`.

Frames are length-prefixed JSON (4-byte big-endian length + UTF-8 body) over a raw socket, so the
transport adds ZERO dependency on either side (Python stdlib here; Node stdlib in the runner). The
episode-driving logic is decoupled from the socket behind the `Channel` protocol so a scripted
in-process peer can exercise the whole broker offline (see runner_link_test.py).

The link is provider-neutral: the caller supplies a structured tool-calling provider, which owns
authentication, routing, wire translation, retries, and failover. RunnerLink only validates and
brokers frames plus environment tool calls.
"""

from __future__ import annotations

import json
import struct
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, cast

from llm_waterfall import ChatRequest, ChatResponse

from wmh.core.types import Action, ActionKind, EnvState, JsonObject, Observation, Step
from wmh.harness.environment import AgentEnvironment, is_env_action
from wmh.harness.runtime import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_TURNS,
    RunResult,
    RuntimeCancelled,
    StopReason,
    TokenUsage,
)
from wmh.harness.skills import SkillLibrary
from wmh.harness.tools import READ_SKILL, ToolSpec
from wmh.providers.base import ToolCallingProvider

DEFAULT_MAX_ENV_ACTIONS = 40
DEFAULT_CANCEL_POLL_INTERVAL_S = 0.5


# --------------------------------------------------------------------------------------------------
# Wire framing: length-prefixed JSON over a raw socket (stdlib only, both sides).
# --------------------------------------------------------------------------------------------------
class _SupportsSocket(Protocol):
    def sendall(self, data: bytes) -> None: ...
    def recv(self, n: int) -> bytes: ...


def write_frame(sock: _SupportsSocket, frame: JsonObject) -> None:
    """Send one JSON frame: 4-byte big-endian length prefix + UTF-8 body."""
    body = json.dumps(frame).encode("utf-8")
    sock.sendall(struct.pack(">I", len(body)) + body)


def read_frame(sock: _SupportsSocket) -> JsonObject | None:
    """Read one framed JSON message, or None if the peer closed the connection cleanly."""
    header = _recv_exactly(sock, 4)
    if header is None:
        return None
    (length,) = struct.unpack(">I", header)
    body = _recv_exactly(sock, length)
    if body is None:
        return None
    return cast("JsonObject", json.loads(body))


def _recv_exactly(sock: _SupportsSocket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


class Channel(Protocol):
    """A bidirectional frame channel to the runner peer (a socket, or a test double)."""

    def send(self, frame: JsonObject) -> None: ...
    def recv(self, timeout: float | None = None) -> JsonObject | None: ...


class SocketChannel:
    """A `Channel` backed by a connected socket, using the length-prefixed JSON framing."""

    def __init__(self, sock: _SupportsSocket) -> None:
        self._sock = sock
        self._recv_buffer = bytearray()

    def send(self, frame: JsonObject) -> None:
        write_frame(self._sock, frame)

    def recv(self, timeout: float | None = None) -> JsonObject | None:
        settimeout = getattr(self._sock, "settimeout", None)
        gettimeout = getattr(self._sock, "gettimeout", None)
        previous = gettimeout() if timeout is not None and callable(gettimeout) else None
        if timeout is not None and callable(settimeout):
            settimeout(timeout)
        try:
            if not self._fill_recv_buffer(4):
                return None
            (length,) = struct.unpack(">I", self._recv_buffer[:4])
            if not self._fill_recv_buffer(4 + length):
                return None
            body = bytes(self._recv_buffer[4 : 4 + length])
            del self._recv_buffer[: 4 + length]
            return cast("JsonObject", json.loads(body))
        finally:
            if timeout is not None and callable(settimeout):
                settimeout(previous)

    def _fill_recv_buffer(self, size: int) -> bool:
        """Read through ``size`` bytes while preserving partial frames across timed polls."""
        while len(self._recv_buffer) < size:
            chunk = self._sock.recv(size - len(self._recv_buffer))
            if not chunk:
                return False
            self._recv_buffer += chunk
        return True


# The process-wide runner channel doc.runtime(PI_TRANSPORT=link) drives. A search/eval sets it once
# (its runner connection is process-scoped infra), so create_harness's internal doc.runtime() calls
# reach the runner without threading a channel through every signature; cleared at teardown.
_ACTIVE_CHANNEL: Channel | None = None


def set_active_channel(channel: Channel | None) -> None:
    global _ACTIVE_CHANNEL
    _ACTIVE_CHANNEL = channel


def active_channel() -> Channel | None:
    return _ACTIVE_CHANNEL


def params_schema(tool: ToolSpec) -> JsonObject:
    """A JSON-schema `parameters` object for a tool, as the model's function-calling API expects."""
    props: JsonObject = {
        name: {"type": "string", "description": desc} for name, desc in tool.arguments.items()
    }
    return {"type": "object", "properties": props, "required": list(tool.arguments)}


def resolve_env_tool_call(
    action: Action,
    *,
    tools: list[ToolSpec],
    skills: SkillLibrary,
    environment: AgentEnvironment,
    env_calls: int,
    max_env_actions: int,
) -> tuple[Observation, bool]:
    """Resolve one tool call to its Observation under the shared runtime tool-routing rules.

    Applies the same precedence every pi episode uses: an unknown tool, the skill reader, an
    exhausted environment-action budget, and a non-environment tool each short-circuit to an
    observation before the environment is touched; anything else executes against `environment`.
    Returns the observation plus whether it consumed an environment-action slot, so the caller can
    advance its own `env_calls` counter (kept caller-side because episodes own their mutable state).
    """
    name = action.name
    if name not in {t.name for t in tools}:
        return Observation(content=f"tool {name!r} not available", is_error=True), False
    if name == READ_SKILL.name:
        raw_name = action.arguments.get("name")
        skill_name = raw_name if isinstance(raw_name, str) else ""
        skill = skills.get(skill_name)
        if skill is None:
            return Observation(content=f"no skill named {skill_name!r}", is_error=True), False
        return Observation(content=skill.body), False
    if env_calls >= max_env_actions:
        return Observation(content="environment action budget exhausted", is_error=True), False
    if not is_env_action(action):
        return Observation(content=f"tool {name!r} not available", is_error=True), False
    return environment.execute(action), True


# --------------------------------------------------------------------------------------------------
# Host-side episode: environment tool routing, budget, and transcript recording.
# --------------------------------------------------------------------------------------------------
@dataclass
class HostEpisode:
    """Per-episode host state: routes tool calls to the environment under a budget, records Steps.

    Same budget/step-recording contract the SSH shim's `_Episode` had, minus the HTTP specifics.
    """

    instruction: str
    tools: list[ToolSpec]
    environment: AgentEnvironment
    skills: SkillLibrary = field(default_factory=SkillLibrary)
    max_env_actions: int = DEFAULT_MAX_ENV_ACTIONS
    steps: list[Step] = field(default_factory=list)
    answer: str = ""
    _env_calls: int = 0

    def tool_specs(self) -> list[JsonObject]:
        return [
            {"name": t.name, "description": t.description, "parameters": params_schema(t)}
            for t in self.tools
        ]

    def run_tool(self, name: str, arguments: JsonObject) -> JsonObject:
        """Answer one runtime/environment tool call under AgentRuntime-compatible semantics."""
        action = Action(kind=ActionKind.TOOL_CALL, name=name, arguments=arguments)
        obs, consumed = resolve_env_tool_call(
            action,
            tools=self.tools,
            skills=self.skills,
            environment=self.environment,
            env_calls=self._env_calls,
            max_env_actions=self.max_env_actions,
        )
        if consumed:
            self._env_calls += 1
        self.steps.append(
            Step(action=action, observation=obs, state_before=EnvState(), task=self.instruction)
        )
        return {"content": obs.content, "is_error": obs.is_error}


# The worker function the host uses to answer llm_request frames; injectable for tests.
WorkerFn = Callable[[ChatRequest], ChatResponse]


class RunnerLink:
    """Drives one pi episode over a `Channel` to the runner peer.

    Sends `episode_start`, then answers `llm_request` (worker LLM, host-side) and `tool_request`
    (environment) frames until `done`/`episode_error`, returning a `RunResult` shaped exactly like
    the other runtimes. One `RunnerLink.run` == one episode; concurrent episodes multiplex over the
    same channel by `episode_id` (a later migration step).
    """

    def __init__(
        self,
        channel: Channel,
        *,
        tools: list[ToolSpec] | None = None,
        provider: ToolCallingProvider | None = None,
        worker_fn: WorkerFn | None = None,
        files: dict[str, str] | None = None,
        system_prompt: str = "",
        max_env_actions: int = DEFAULT_MAX_ENV_ACTIONS,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float = 0.7,
        skills: SkillLibrary | None = None,
        episode_timeout_s: float | None = None,
        should_cancel: Callable[[], bool] | None = None,
        cancel_poll_interval_s: float = DEFAULT_CANCEL_POLL_INTERVAL_S,
    ) -> None:
        self._channel = channel
        # Tools bound at construction make RunnerLink satisfy the runtime contract closed-loop eval
        # drives — `run(task_id, instruction, environment)` — while `run(..., tools=...)` still lets
        # a caller (or the conformance tests) override per episode.
        self._tools = list(tools or [])
        self._skills = skills if skills is not None else SkillLibrary()
        if len(self._skills) and READ_SKILL.name not in {tool.name for tool in self._tools}:
            self._tools.append(READ_SKILL)
        if worker_fn is None and provider is None:
            raise ValueError("RunnerLink needs a ToolCallingProvider or worker_fn")
        # worker_fn lets tests answer llm_request without a real provider.
        if worker_fn is not None:
            self._worker_fn = worker_fn
        else:
            assert provider is not None
            self._worker_fn = provider.complete_chat
        self._files = files or {}
        self._system_prompt = system_prompt
        self._max_env_actions = max_env_actions
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be >= 1")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        if episode_timeout_s is not None and episode_timeout_s <= 0:
            raise ValueError("episode_timeout_s must be positive when set")
        if cancel_poll_interval_s <= 0:
            raise ValueError("cancel_poll_interval_s must be positive")
        self._max_turns = max_turns
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._episode_timeout_s = episode_timeout_s
        self._should_cancel = should_cancel
        self._cancel_poll_interval_s = cancel_poll_interval_s

    def run(
        self,
        task_id: str,
        instruction: str,
        environment: AgentEnvironment,
        *,
        tools: list[ToolSpec] | None = None,
    ) -> RunResult:
        episode_tools = list(tools) if tools is not None else list(self._tools)
        if len(self._skills) and READ_SKILL.name not in {tool.name for tool in episode_tools}:
            episode_tools.append(READ_SKILL)
        episode = HostEpisode(
            instruction=instruction,
            tools=episode_tools,
            environment=environment,
            skills=self._skills,
            max_env_actions=self._max_env_actions,
        )
        episode_id = uuid.uuid4().hex
        usage = TokenUsage()
        self._check_cancelled(usage)
        deadline = (
            time.monotonic() + self._episode_timeout_s
            if self._episode_timeout_s is not None
            else None
        )

        def send_frame(frame: JsonObject) -> RunResult | None:
            try:
                self._channel.send(frame)
            except Exception:
                self._check_cancelled(usage)
                if deadline is not None and time.monotonic() >= deadline:
                    return self._budget_result(task_id, episode, instruction, usage)
                raise
            self._check_cancelled(usage)
            if deadline is not None and time.monotonic() >= deadline:
                return self._budget_result(task_id, episode, instruction, usage)
            return None

        stopped = send_frame(
            {
                "type": "episode_start",
                "episode_id": episode_id,
                "task_id": task_id,
                "instruction": instruction,
                "system": self._system_prompt,
                "tools": episode.tool_specs(),
                "files": self._files,
                "max_env_actions": self._max_env_actions,
                "max_turns": self._max_turns,
                "max_output_tokens": self._max_output_tokens,
                "temperature": self._temperature,
                "episode_timeout_s": self._episode_timeout_s,
            }
        )
        if stopped is not None:
            return stopped
        while True:
            self._check_cancelled(usage)
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return self._budget_result(task_id, episode, instruction, usage)
            recv_timeout = remaining
            if self._should_cancel is not None:
                recv_timeout = (
                    self._cancel_poll_interval_s
                    if recv_timeout is None
                    else min(recv_timeout, self._cancel_poll_interval_s)
                )
            try:
                frame = self._channel.recv(timeout=recv_timeout)
            except TimeoutError:
                self._check_cancelled(usage)
                if deadline is not None and time.monotonic() >= deadline:
                    return self._budget_result(task_id, episode, instruction, usage)
                if self._should_cancel is not None:
                    continue
                raise
            except Exception:
                self._check_cancelled(usage)
                if deadline is not None and time.monotonic() >= deadline:
                    return self._budget_result(task_id, episode, instruction, usage)
                raise
            self._check_cancelled(usage)
            if deadline is not None and time.monotonic() >= deadline:
                return self._budget_result(task_id, episode, instruction, usage)
            if frame is None:  # channel closed before the episode finished
                return self._error_result(
                    task_id, episode, instruction, "runner channel closed", usage=usage
                )
            kind = frame.get("type")
            if kind == "llm_request":
                response = self._llm_response(episode_id, frame, usage)
                self._check_cancelled(usage)
                if deadline is not None and time.monotonic() >= deadline:
                    return self._budget_result(task_id, episode, instruction, usage)
                # A send timeout is transport failure with an uncertain delivery state. Let it
                # propagate so the owning runtime retires rather than sending a second response.
                stopped = send_frame(response)
                if stopped is not None:
                    return stopped
            elif kind == "tool_request":
                name = frame.get("name")
                args = frame.get("arguments")
                obs = episode.run_tool(
                    name if isinstance(name, str) else "",
                    args if isinstance(args, dict) else {},
                )
                self._check_cancelled(usage)
                if deadline is not None and time.monotonic() >= deadline:
                    return self._budget_result(task_id, episode, instruction, usage)
                stopped = send_frame(
                    {
                        "type": "tool_response",
                        "episode_id": episode_id,
                        "req_id": frame.get("req_id"),
                        **obs,
                    }
                )
                if stopped is not None:
                    return stopped
            elif kind == "done":
                answer = frame.get("answer")
                episode.answer = answer if isinstance(answer, str) else ""
                return RunResult(
                    task_id=task_id,
                    steps=episode.steps,
                    stop_reason=StopReason.SUBMITTED,
                    answer=episode.answer,
                    turns=len(episode.steps),
                    worker_usage=usage if usage.calls else None,
                )
            elif kind == "episode_error":
                note = frame.get("note")
                return self._error_result(
                    task_id,
                    episode,
                    instruction,
                    note if isinstance(note, str) else "runner error",
                    usage=usage,
                )
            # unknown frame types are ignored (forward-compatible)

    def _check_cancelled(self, usage: TokenUsage) -> None:
        if self._should_cancel is not None and self._should_cancel():
            raise RuntimeCancelled(
                "runtime episode cancelled",
                worker_usage=(usage.model_copy() if usage.calls else None),
            )

    def _budget_result(
        self,
        task_id: str,
        episode: HostEpisode,
        instruction: str,
        usage: TokenUsage,
    ) -> RunResult:
        assert self._episode_timeout_s is not None
        return self._error_result(
            task_id,
            episode,
            instruction,
            f"evaluation episode exceeded {self._episode_timeout_s:g}s wall budget",
            stop=StopReason.BUDGET,
            usage=usage,
        )

    def _llm_response(self, episode_id: str, frame: JsonObject, usage: TokenUsage) -> JsonObject:
        req_id = frame.get("req_id")
        body = frame.get("openai_body")
        try:
            # The runner owns message/tool serialization, while HarnessDoc owns sampling policy.
            # Override any runner default at the final host boundary before the real model call.
            request_body = dict(body) if isinstance(body, dict) else {}
            request_body["temperature"] = self._temperature
            request = ChatRequest.model_validate(request_body)
            completion = self._worker_fn(request)
            # Meter the worker leg from the provider's structured response.
            usage.calls += 1
            reported = completion.token_usage()
            usage.input_tokens += reported.input_tokens
            usage.output_tokens += reported.output_tokens
            response: JsonObject = {
                "type": "llm_response",
                "episode_id": episode_id,
                "req_id": req_id,
                "completion": completion.wire_payload(),
            }
        except Exception as exc:  # noqa: BLE001 - report to the runner, never crash the host
            response = {
                "type": "llm_response",
                "episode_id": episode_id,
                "req_id": req_id,
                "error": str(exc),
            }
        return response

    @staticmethod
    def _error_result(
        task_id: str,
        episode: HostEpisode,
        instruction: str,
        note: str,
        *,
        stop: StopReason | None = None,
        usage: TokenUsage | None = None,
    ) -> RunResult:
        resolved_stop = stop or (StopReason.MAX_TURNS if episode.steps else StopReason.ERROR)
        episode.steps.append(
            Step(
                action=Action(kind=ActionKind.MESSAGE, content="(runner link)"),
                observation=Observation(content=note, is_error=True),
                state_before=EnvState(),
                task=instruction,
            )
        )
        return RunResult(
            task_id=task_id,
            steps=episode.steps,
            stop_reason=resolved_stop,
            answer="",
            turns=len(episode.steps),
            worker_usage=usage if usage is not None and usage.calls else None,
        )
