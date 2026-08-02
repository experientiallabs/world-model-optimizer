"""Tests for running ordinary agents inside a persistent filesystem project."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest
from llm_waterfall import ChatResponse

from wmo.agents import project as project_module
from wmo.agents.meta import meta_agent
from wmo.agents.optimizer import optimizer_agent
from wmo.agents.project import AgentProject
from wmo.core.types import JsonObject
from wmo.harness import e2b_sandbox as e2b_sandbox_module
from wmo.harness.doc import TOOL_POLICY_ID, HarnessDoc
from wmo.harness.e2b_sandbox import SandboxCleanupError, SandboxUsage
from wmo.harness.live_session import SessionEvent
from wmo.harness.runtime import HarnessSearchCancelled
from wmo.harness.source_tree import HarnessSourceFile, HarnessSourceTree
from wmo.providers.base import ProviderConfig, ProviderKind


class _Files:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def write(self, path: str, data: str) -> object:
        self.values[path] = data
        return None

    def read(
        self,
        path: str,
        *,
        request_timeout: float | None = None,
        gzip: bool = False,
    ) -> str:
        del request_timeout, gzip
        return self.values[path]


class _Output:
    def __init__(self, *, stdout: str = "", stderr: str = "", exit_code: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


class _Commands:
    def __init__(self, files: _Files | None = None) -> None:
        self.runs: list[str] = []
        self.files = files

    def run(self, cmd: str, background: bool | None = None, **kwargs: object) -> _Output:
        del background, kwargs
        self.runs.append(cmd)
        if cmd.startswith("find ") and self.files is not None:
            paths = sorted(self.files.values)
            return _Output(stdout="\0".join(paths) + ("\0" if paths else ""))
        return _Output()

    def send_stdin(self, pid: int, data: str, request_timeout: float | None = None) -> object:
        del pid, data, request_timeout
        return None


class _Sandbox:
    def __init__(self) -> None:
        self.files = _Files()
        self.commands = _Commands(self.files)
        self.killed = False
        self.network_updates: list[dict[str, object]] = []

    def set_timeout(self, timeout: int) -> None:
        del timeout

    def update_network(self, network: dict[str, object]) -> None:
        self.network_updates.append(network)

    def kill(self, request_timeout: float | None = None) -> object:
        del request_timeout
        self.killed = True
        return None


class _FlakyKillSandbox(_Sandbox):
    def __init__(self, *, failures: int) -> None:
        super().__init__()
        self.failures = failures
        self.kill_attempts = 0

    def kill(self, request_timeout: float | None = None) -> object:
        del request_timeout
        self.kill_attempts += 1
        if self.kill_attempts <= self.failures:
            raise RuntimeError("control plane unavailable")
        self.killed = True
        return None


class _Channel:
    def __init__(self) -> None:
        self.inbound: list[JsonObject] = [
            {"type": "state", "status": "idle"},
            {"type": "state", "status": "running"},
            {
                "type": "tool_request",
                "req_id": 1,
                "name": "write_file",
                "arguments": {"path": "/home/user/project/result.txt", "content": "done"},
            },
            {
                "type": "tool_request",
                "req_id": 2,
                "name": "submit",
                "arguments": {"answer": "finished"},
            },
            {"type": "state", "status": "idle", "reason": "completed"},
        ]
        self.sent: list[JsonObject] = []
        self.closed = False

    def send(self, frame: JsonObject) -> None:
        self.sent.append(frame)

    def recv(self, timeout: float | None = None) -> JsonObject | None:
        del timeout
        return self.inbound.pop(0) if self.inbound else None

    def close(self) -> None:
        self.closed = True


class _Provider:
    config = ProviderConfig(kind=ProviderKind.BEDROCK, model="worker")

    def complete_chat(self, request: object) -> ChatResponse:
        del request
        raise AssertionError("scripted channel never requests the worker")


class _MeteredProvider(_Provider):
    def complete_chat(self, request: object) -> ChatResponse:
        del request
        return ChatResponse.model_validate(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "working"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 7},
            }
        )


class _FailingProvider(_Provider):
    def complete_chat(self, request: object) -> ChatResponse:
        del request
        raise RuntimeError("provider down")


def test_default_project_channel_enables_durable_outbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = _Sandbox()
    channel = _Channel()
    captured: dict[str, object] = {}

    def start(sandbox_arg: object, **kwargs: object) -> _Channel:
        captured["sandbox"] = sandbox_arg
        captured.update(kwargs)
        return channel

    monkeypatch.setattr(project_module, "start_live_runner", start)

    assert project_module._start_channel(sandbox, "/project") is channel  # noqa: SLF001
    assert captured == {
        "sandbox": sandbox,
        "workspace": "/project",
        "durable_outbox": True,
    }


def test_project_preserves_files_and_runs_through_live_session() -> None:
    sandbox = _Sandbox()
    channel = _Channel()
    project = AgentProject(
        sandbox,
        channel_factory=lambda sandbox, workspace: channel,
        owns_sandbox=False,
    )

    project.write_text("history/round-1.json", "{}")
    result = project.run(meta_agent(), _Provider(), "produce a result", timeout=1)

    assert project.read_text("history/round-1.json") == "{}"
    assert project.read_text("result.txt") == "done"
    assert result.answer == "finished"
    [session_start] = [frame for frame in channel.sent if frame["type"] == "session_start"]
    assert session_start["max_output_tokens"] == meta_agent().max_output_tokens() == 16384
    assert session_start["conversation_scope"] == "turn"
    assert project._session is not None  # noqa: SLF001 - runtime wiring contract
    assert project._session._actions_per_turn == meta_agent().max_turns() == 60  # noqa: SLF001
    assert any(frame["type"] == "user_message" for frame in channel.sent)
    assert channel.closed is False
    assert sandbox.commands.runs[:2] == [
        "mkdir -p /home/user/project",
        "mkdir -p /home/user/project/history",
    ]
    project.close()
    assert channel.closed is True


def test_project_grants_agent_writes_to_exact_files_only() -> None:
    """A turn grant contains agent writes without constraining trusted host writes."""
    sandbox = _Sandbox()
    channel = _Channel()
    channel.inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "running"},
        {
            "type": "tool_request",
            "req_id": 1,
            "name": "write_file",
            "arguments": {
                "path": "/home/user/project/context/round-1/parent.json",
                "content": "poisoned",
            },
        },
        {
            "type": "tool_request",
            "req_id": 2,
            "name": "write_file",
            "arguments": {
                "path": "/home/user/project/proposals/round-1/proposal-01.json",
                "content": "candidate",
            },
        },
        {
            "type": "tool_request",
            "req_id": 3,
            "name": "submit",
            "arguments": {"answer": "done"},
        },
        {"type": "state", "status": "idle", "reason": "completed"},
    ]
    project = AgentProject(
        sandbox,
        channel_factory=lambda sandbox, workspace: channel,
        owns_sandbox=False,
    )
    project.write_text("context/round-1/parent.json", "trusted")
    host_write_pending = [True]

    def on_event(event: SessionEvent) -> None:
        if event.kind != "tool_call" or not host_write_pending:
            return
        host_write_pending.clear()
        project.write_text("context/round-1/host-note.txt", "host-authored")

    result = project.run(
        meta_agent(),
        _Provider(),
        "produce one proposal",
        timeout=1,
        on_event=on_event,
        writable_files=["proposals/round-1/proposal-01.json"],
    )

    assert result.answer == "done"
    assert project.read_text("context/round-1/parent.json") == "trusted"
    assert project.read_text("context/round-1/host-note.txt") == "host-authored"
    assert project.read_text("proposals/round-1/proposal-01.json") == "candidate"
    tool_results = [event for event in result.events if event.kind == "tool_result"]
    assert [event.payload["is_error"] for event in tool_results] == [True, False]
    assert "not writable in this project turn" in str(tool_results[0].payload["content"])


def test_project_write_grant_resets_between_turns_on_one_session() -> None:
    """A restricted turn cannot narrow a later backward-compatible unrestricted turn."""
    sandbox = _Sandbox()
    channel = _Channel()
    channel.inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "running"},
        {
            "type": "tool_request",
            "req_id": 1,
            "name": "write_file",
            "arguments": {"path": "memory.txt", "content": "blocked"},
        },
        {
            "type": "tool_request",
            "req_id": 2,
            "name": "submit",
            "arguments": {"answer": "restricted"},
        },
        {"type": "state", "status": "idle", "reason": "completed"},
        {"type": "state", "status": "running"},
        {
            "type": "tool_request",
            "req_id": 3,
            "name": "write_file",
            "arguments": {"path": "memory.txt", "content": "unrestricted"},
        },
        {
            "type": "tool_request",
            "req_id": 4,
            "name": "submit",
            "arguments": {"answer": "second"},
        },
        {"type": "state", "status": "idle", "reason": "completed"},
    ]
    project = AgentProject(
        sandbox,
        channel_factory=lambda sandbox, workspace: channel,
        owns_sandbox=False,
    )
    project.write_text("memory.txt", "original")
    agent = meta_agent()
    provider = _Provider()

    first = project.run(agent, provider, "restricted", timeout=1, writable_files=[])
    assert first.answer == "restricted"
    assert project.read_text("memory.txt") == "original"

    second = project.run(agent, provider, "unrestricted", timeout=1)

    assert second.answer == "second"
    assert project.read_text("memory.txt") == "unrestricted"
    assert [frame["type"] for frame in channel.sent].count("session_start") == 1


def test_owned_project_disables_internet_before_the_agent_turn() -> None:
    order: list[str] = []

    class _OrderedSandbox(_Sandbox):
        def update_network(self, network: dict[str, object]) -> None:
            super().update_network(network)
            order.append("network-locked")

    class _OrderedChannel(_Channel):
        def send(self, frame: JsonObject) -> None:
            if frame["type"] == "session_start":
                order.append("session-start")
            super().send(frame)

    sandbox = _OrderedSandbox()
    channel = _OrderedChannel()
    project = AgentProject(
        sandbox,
        channel_factory=lambda sandbox, workspace: channel,
    )

    project.run(meta_agent(), _Provider(), "produce a result", timeout=1)

    assert sandbox.network_updates == [{"allow_internet_access": False}]
    assert order[:2] == ["network-locked", "session-start"]
    project.close()


def test_project_retries_one_context_write_after_e2b_disconnect() -> None:
    """A transient context-file transport drop cannot fan out into a failed proposal batch."""

    class _DisconnectOnceCommands(_Commands):
        def __init__(self) -> None:
            super().__init__()
            self.disconnect_next = False
            self.attempts = 0

        def run(self, cmd: str, background: bool | None = None, **kwargs: object) -> _Output:
            self.attempts += 1
            if self.disconnect_next:
                self.disconnect_next = False
                raise RuntimeError("Server disconnected")
            return super().run(cmd, background=background, **kwargs)

    sandbox = _Sandbox()
    commands = _DisconnectOnceCommands()
    sandbox.commands = commands
    project = AgentProject(
        sandbox,
        channel_factory=lambda sandbox, workspace: _Channel(),
        sandbox_factory=lambda: pytest.fail("an idempotent write must not replace the sandbox"),
    )
    attempts_before = commands.attempts
    commands.disconnect_next = True

    project.write_text("context/round-0003/parent.json", '{"round": 3}')

    assert commands.attempts - attempts_before == 2
    assert project.read_text("context/round-0003/parent.json") == '{"round": 3}'
    assert project.usage().count == 1
    assert sandbox.killed is False


def test_project_retries_one_context_write_after_closed_http2_connection() -> None:
    """A stale E2B HTTP/2 connection cannot invalidate an entire proposal batch."""

    class _ClosedHttp2OnceCommands(_Commands):
        def __init__(self) -> None:
            super().__init__()
            self.close_next = False
            self.attempts = 0

        def run(self, cmd: str, background: bool | None = None, **kwargs: object) -> _Output:
            self.attempts += 1
            if self.close_next:
                self.close_next = False
                raise RuntimeError(
                    "Invalid input ConnectionInputs.SEND_DATA in state ConnectionState.CLOSED"
                )
            return super().run(cmd, background=background, **kwargs)

    sandbox = _Sandbox()
    commands = _ClosedHttp2OnceCommands()
    sandbox.commands = commands
    project = AgentProject(
        sandbox,
        channel_factory=lambda sandbox, workspace: _Channel(),
        sandbox_factory=lambda: pytest.fail("a fresh HTTP/2 request must reuse the project"),
    )
    attempts_before = commands.attempts
    commands.close_next = True

    project.write_text("context/round-0007/parent.json", '{"round": 7}')

    assert commands.attempts - attempts_before == 2
    assert project.read_text("context/round-0007/parent.json") == '{"round": 7}'
    assert project.usage().count == 1
    assert sandbox.killed is False


def test_project_replaces_owned_sandbox_after_repeated_closed_http2_writes() -> None:
    """An exhausted context-write retry reaches the project's bounded sandbox fallback."""

    class _ClosedHttp2Commands(_Commands):
        closed = False

        def run(self, cmd: str, background: bool | None = None, **kwargs: object) -> _Output:
            if self.closed:
                raise RuntimeError(
                    "Invalid input ConnectionInputs.SEND_DATA in state ConnectionState.CLOSED"
                )
            return super().run(cmd, background=background, **kwargs)

    original = _Sandbox()
    original_commands = _ClosedHttp2Commands()
    original.commands = original_commands
    replacement = _Sandbox()
    project = AgentProject(original, sandbox_factory=lambda: replacement)
    project.write_text("history/round-0006.json", '{"kept": true}')
    original_commands.closed = True

    project.write_text("context/round-0007/parent.json", '{"round": 7}')

    assert original.killed is True
    assert replacement.killed is False
    assert project.usage().count == 2
    assert replacement.files.values["/home/user/project/history/round-0006.json"] == (
        '{"kept": true}'
    )
    assert replacement.files.values["/home/user/project/context/round-0007/parent.json"] == (
        '{"round": 7}'
    )


def test_context_write_recovery_restarts_an_idle_project_session() -> None:
    """A poisoned next-round write preserves the archive and resumes on a fresh live session."""

    class _ClosedHttp2Commands(_Commands):
        closed = False

        def run(self, cmd: str, background: bool | None = None, **kwargs: object) -> _Output:
            if self.closed:
                raise RuntimeError(
                    "Invalid input ConnectionInputs.SEND_DATA in state ConnectionState.CLOSED"
                )
            return super().run(cmd, background=background, **kwargs)

    original = _Sandbox()
    original_commands = _ClosedHttp2Commands()
    original.commands = original_commands
    replacement = _Sandbox()
    original_channel = _Channel()
    replacement_channel = _Channel()
    project = AgentProject(
        original,
        channel_factory=lambda sandbox, workspace: (
            replacement_channel if sandbox is replacement else original_channel
        ),
        sandbox_factory=lambda: replacement,
    )
    agent = meta_agent()
    provider = _Provider()
    project.write_text("history/round-0006.json", '{"kept": true}')
    first = project.run(agent, provider, "round 6", timeout=1)
    original_commands.closed = True

    project.write_text("context/round-0007/parent.json", '{"round": 7}')
    second = project.run(agent, provider, "round 7", timeout=1)

    assert first.answer == second.answer == "finished"
    assert original_channel.closed is True
    assert replacement_channel.closed is False
    assert original.killed is True
    assert replacement.killed is False
    assert project.usage().count == 2
    assert project.read_text("history/round-0006.json") == '{"kept": true}'
    assert project.read_text("context/round-0007/parent.json") == '{"round": 7}'
    assert [frame["type"] for frame in original_channel.sent].count("user_message") == 1
    assert [frame["type"] for frame in replacement_channel.sent].count("user_message") == 1


def test_project_does_not_retry_non_transport_context_write_failure() -> None:
    """A real filesystem failure still propagates after exactly one attempt."""

    class _FailOnceCommands(_Commands):
        def __init__(self) -> None:
            super().__init__()
            self.fail_next = False
            self.attempts = 0

        def run(self, cmd: str, background: bool | None = None, **kwargs: object) -> _Output:
            self.attempts += 1
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("permission denied")
            return super().run(cmd, background=background, **kwargs)

    sandbox = _Sandbox()
    commands = _FailOnceCommands()
    sandbox.commands = commands
    project = AgentProject(sandbox, channel_factory=lambda sandbox, workspace: _Channel())
    attempts_before = commands.attempts
    commands.fail_next = True

    with pytest.raises(RuntimeError, match="permission denied"):
        project.write_text("context/round-0003/parent.json", '{"round": 3}')

    assert commands.attempts - attempts_before == 1


def test_project_reuses_one_agent_runner_across_fresh_project_turns() -> None:
    sandbox = _Sandbox()
    channel = _Channel()
    channel.inbound.extend(
        [
            {"type": "state", "status": "running"},
            {
                "type": "tool_request",
                "req_id": 3,
                "name": "submit",
                "arguments": {"answer": "second"},
            },
            {"type": "state", "status": "idle", "reason": "completed"},
        ]
    )
    starts = 0

    def channel_factory(sandbox: object, workspace: str) -> _Channel:
        nonlocal starts
        del sandbox, workspace
        starts += 1
        return channel

    project = AgentProject(sandbox, channel_factory=channel_factory, owns_sandbox=False)
    agent = meta_agent()
    provider = _Provider()

    first = project.run(agent, provider, "first turn", timeout=1)
    second = project.run(agent, provider, "second turn", timeout=1)

    assert first.answer == "finished"
    assert second.answer == "second"
    assert starts == 1
    assert [frame["type"] for frame in channel.sent].count("session_start") == 1
    assert [frame["type"] for frame in channel.sent].count("user_message") == 2
    assert (
        next(frame for frame in channel.sent if frame["type"] == "session_start")[
            "conversation_scope"
        ]
        == "turn"
    )


def test_project_surfaces_a_mid_turn_runner_error() -> None:
    sandbox = _Sandbox()
    channel = _Channel()
    channel.inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "running"},
        {"type": "episode_error", "note": "worker bridge disconnected"},
    ]
    project = AgentProject(
        sandbox,
        channel_factory=lambda sandbox, workspace: channel,
        owns_sandbox=False,
    )

    with pytest.raises(RuntimeError, match="worker bridge disconnected"):
        project.run(meta_agent(), _Provider(), "produce a result", timeout=1)


def test_project_promotes_a_worker_error_after_the_runner_returns_idle() -> None:
    """A provider failure cannot look like a normally completed project turn."""
    sandbox = _Sandbox()
    channel = _Channel()
    channel.inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "running"},
        {"type": "llm_request", "req_id": 1, "openai_body": {"messages": []}},
        {"type": "state", "status": "idle", "reason": "completed"},
    ]
    events: list[SessionEvent] = []
    project = AgentProject(
        sandbox,
        channel_factory=lambda sandbox, workspace: channel,
        owns_sandbox=False,
    )

    with pytest.raises(
        RuntimeError,
        match="project agent session failed: worker LLM error: provider down",
    ):
        project.run(
            meta_agent(),
            _FailingProvider(),
            "produce a result",
            timeout=1,
            on_event=events.append,
        )

    error = next(event for event in events if event.kind == "error")
    assert error.payload == {"message": "worker LLM error: provider down"}
    response = next(frame for frame in channel.sent if frame["type"] == "llm_response")
    assert response["error"] == "provider down"


def test_project_does_not_retry_a_provider_error_that_looks_like_transport() -> None:
    """Provider text cannot borrow the project transport's retry ownership."""

    class _TransportLookingProvider(_Provider):
        def __init__(self) -> None:
            self.calls = 0

        def complete_chat(self, request: object) -> ChatResponse:
            del request
            self.calls += 1
            raise RuntimeError("Server disconnected without sending a response")

    failed = _Channel()
    failed.inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "running"},
        {"type": "llm_request", "req_id": 1, "openai_body": {"messages": []}},
        {"type": "state", "status": "idle", "reason": "completed"},
    ]
    recovered = _Channel()
    channels = iter([failed, recovered])
    starts = 0

    def channel_factory(sandbox: object, workspace: str) -> _Channel:
        nonlocal starts
        del sandbox, workspace
        starts += 1
        return next(channels)

    provider = _TransportLookingProvider()
    project = AgentProject(
        _Sandbox(),
        channel_factory=channel_factory,
        owns_sandbox=False,
    )

    with pytest.raises(
        RuntimeError,
        match="worker LLM error: Server disconnected without sending a response",
    ):
        project.run(meta_agent(), provider, "produce a result", timeout=1)

    assert provider.calls == 1
    assert starts == 1
    assert failed.closed is False


def test_project_ignores_idle_until_the_new_turn_reports_running() -> None:
    """A stale idle frame cannot complete a newly queued project turn."""
    channel = _Channel()
    channel.inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "idle", "reason": "stale_abort"},
        {"type": "state", "status": "running"},
        {
            "type": "tool_request",
            "req_id": 1,
            "name": "submit",
            "arguments": {"answer": "fresh"},
        },
        {"type": "state", "status": "idle", "reason": "completed"},
    ]
    project = AgentProject(
        _Sandbox(),
        channel_factory=lambda sandbox, workspace: channel,
        owns_sandbox=False,
    )

    result = project.run(meta_agent(), _Provider(), "produce a result", timeout=1)

    assert result.answer == "fresh"


def test_project_timeout_retires_the_session_before_the_next_turn() -> None:
    """A late abort boundary cannot leak from a timed-out turn into its successor."""

    class _HangingChannel(_Channel):
        def __init__(self) -> None:
            super().__init__()
            self.inbound = [
                {"type": "state", "status": "idle"},
                {"type": "state", "status": "running"},
            ]

        def recv(self, timeout: float | None = None) -> JsonObject | None:
            if self.inbound:
                return super().recv(timeout)
            raise TimeoutError

    hanging = _HangingChannel()
    recovered = _Channel()
    channels = iter([hanging, recovered])
    starts = 0

    def channel_factory(sandbox: object, workspace: str) -> _Channel:
        nonlocal starts
        del sandbox, workspace
        starts += 1
        return next(channels)

    project = AgentProject(
        _Sandbox(),
        channel_factory=channel_factory,
        owns_sandbox=False,
    )
    agent = meta_agent()
    provider = _Provider()

    with pytest.raises(TimeoutError, match="project agent did not finish"):
        project.run(agent, provider, "timed-out turn", timeout=0.001)
    result = project.run(agent, provider, "next turn", timeout=1)

    assert hanging.closed is True
    assert result.answer == "finished"
    assert starts == 2


def test_project_cancellation_interrupts_and_retires_the_active_session() -> None:
    """Cancellation after one blocking provider pump cannot start a second model call."""

    class _CountingProvider(_MeteredProvider):
        def __init__(self) -> None:
            self.calls = 0

        def complete_chat(self, request: object) -> ChatResponse:
            self.calls += 1
            return super().complete_chat(request)

    channel = _Channel()
    channel.inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "running"},
        {"type": "llm_request", "req_id": 1, "openai_body": {"messages": []}},
        {"type": "llm_request", "req_id": 2, "openai_body": {"messages": []}},
    ]
    provider = _CountingProvider()
    project = AgentProject(
        _Sandbox(),
        channel_factory=lambda sandbox, workspace: channel,
        owns_sandbox=False,
    )

    with pytest.raises(HarnessSearchCancelled, match="cancelled"):
        project.run(
            meta_agent(),
            provider,
            "produce a result",
            timeout=1,
            should_cancel=lambda: provider.calls >= 1,
        )

    assert provider.calls == 1
    assert channel.closed is True
    assert any(
        frame.get("type") == "abort" and frame.get("reason") == "harness_search_cancelled"
        for frame in channel.sent
    )


@pytest.mark.parametrize("reason", ["aborted", "turn_limit"])
def test_project_promotes_unsuccessful_terminal_turn_reasons(reason: str) -> None:
    """A bounded/aborted agent turn cannot masquerade as a completed proposal turn."""
    channel = _Channel()
    channel.inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "running"},
        {"type": "state", "status": "idle", "reason": reason},
    ]
    project = AgentProject(
        _Sandbox(),
        channel_factory=lambda sandbox, workspace: channel,
        owns_sandbox=False,
    )

    with pytest.raises(RuntimeError, match=f"turn ended with reason: {reason}"):
        project.run(meta_agent(), _Provider(), "produce a result", timeout=1)


def test_project_restarts_one_live_session_after_transport_disconnect() -> None:
    """A dropped runner stream retries once without replacing project storage."""

    class _DisconnectedChannel(_Channel):
        def __init__(self) -> None:
            super().__init__()
            self.inbound = [
                {"type": "state", "status": "idle"},
                {"type": "state", "status": "running"},
            ]

        def recv(self, timeout: float | None = None) -> JsonObject | None:
            if self.inbound:
                return super().recv(timeout)
            raise RuntimeError("Server disconnected")

    sandbox = _Sandbox()
    disconnected = _DisconnectedChannel()
    recovered = _Channel()
    channels = iter([disconnected, recovered])
    project = AgentProject(
        sandbox,
        channel_factory=lambda sandbox, workspace: next(channels),
        owns_sandbox=False,
    )
    project.write_text("history/round-1.json", '{"kept": true}')

    result = project.run(meta_agent(), _Provider(), "produce a result", timeout=1)

    assert result.answer == "finished"
    assert disconnected.closed is True
    assert recovered.closed is False
    assert project.read_text("history/round-1.json") == '{"kept": true}'
    assert [frame["type"] for frame in disconnected.sent].count("user_message") == 1
    assert [frame["type"] for frame in recovered.sent].count("user_message") == 1


def test_project_retries_a_transient_channel_send_failure() -> None:
    """E2B stdin timeouts are transport failures even after LiveSession stringifies them."""

    class _SendFailureChannel(_Channel):
        def send(self, frame: JsonObject) -> None:
            if frame.get("type") == "user_message":
                raise RuntimeError("request timed out")
            super().send(frame)

    failed = _SendFailureChannel()
    failed.inbound = [{"type": "state", "status": "idle"}]
    recovered = _Channel()
    channels = iter([failed, recovered])
    project = AgentProject(
        _Sandbox(),
        channel_factory=lambda sandbox, workspace: next(channels),
        owns_sandbox=False,
    )

    result = project.run(meta_agent(), _Provider(), "produce a result", timeout=1)

    assert result.answer == "finished"
    assert failed.closed is True
    assert recovered.closed is False


def test_project_retries_an_initial_session_start_socket_failure() -> None:
    """The direct LiveSession.start send reaches the same bounded recovery path."""

    class _StartFailureChannel(_Channel):
        def send(self, frame: JsonObject) -> None:
            if frame.get("type") == "session_start":
                raise RuntimeError("failed to send a frame to the E2B runner")
            super().send(frame)

    failed = _StartFailureChannel()
    recovered = _Channel()
    channels = iter([failed, recovered])
    project = AgentProject(
        _Sandbox(),
        channel_factory=lambda sandbox, workspace: next(channels),
        owns_sandbox=False,
    )

    result = project.run(meta_agent(), _Provider(), "produce a result", timeout=1)

    assert result.answer == "finished"
    assert failed.closed is True
    assert recovered.closed is False


def test_project_replaces_owned_sandbox_after_durable_outbox_failure() -> None:
    """A corrupt durable transport still reaches the bounded fresh-sandbox fallback."""

    class _CorruptOutboxChannel(_Channel):
        def __init__(self) -> None:
            super().__init__()
            self.inbound = [
                {"type": "state", "status": "idle"},
                {"type": "state", "status": "running"},
            ]

        def recv(self, timeout: float | None = None) -> JsonObject | None:
            if self.inbound:
                return super().recv(timeout)
            raise RuntimeError("durable outbox frame 4 unavailable after 5s")

    original = _Sandbox()
    replacement = _Sandbox()
    failed = _CorruptOutboxChannel()
    recovered = _Channel()
    project = AgentProject(
        original,
        channel_factory=lambda sandbox, workspace: recovered if sandbox is replacement else failed,
        sandbox_factory=lambda: replacement,
    )

    result = project.run(meta_agent(), _Provider(), "produce a result", timeout=1)

    assert result.answer == "finished"
    assert original.killed is True
    assert replacement.killed is False
    assert project.usage().count == 2


def test_denied_agent_write_never_enters_the_replayed_project_mirror() -> None:
    """A sandbox replacement replays trusted bytes, not a rejected agent overwrite."""

    class _DisconnectedAfterDeniedWrite(_Channel):
        def __init__(self) -> None:
            super().__init__()
            self.inbound = [
                {"type": "state", "status": "idle"},
                {"type": "state", "status": "running"},
                {
                    "type": "tool_request",
                    "req_id": 1,
                    "name": "write_file",
                    "arguments": {
                        "path": "context/round-1/parent.json",
                        "content": "poisoned",
                    },
                },
            ]

        def recv(self, timeout: float | None = None) -> JsonObject | None:
            if self.inbound:
                return super().recv(timeout)
            raise RuntimeError("Server disconnected")

    original = _Sandbox()
    replacement = _Sandbox()
    failed = _DisconnectedAfterDeniedWrite()
    recovered = _Channel()
    project = AgentProject(
        original,
        channel_factory=lambda sandbox, workspace: recovered if sandbox is replacement else failed,
        sandbox_factory=lambda: replacement,
    )
    parent_path = "context/round-1/parent.json"
    project.write_text(parent_path, "trusted")

    result = project.run(
        meta_agent(),
        _Provider(),
        "produce a result",
        timeout=1,
        writable_files=["result.txt"],
    )

    assert result.answer == "finished"
    assert original.killed is True
    assert replacement.files.values[f"/home/user/project/{parent_path}"] == "trusted"
    assert project.read_text(parent_path) == "trusted"
    assert project.read_text("result.txt") == "done"


def test_project_counts_worker_usage_from_failed_and_recovered_attempts() -> None:
    """A logical run reports tokens spent before and after its bounded recovery."""

    class _DisconnectedAfterLlmChannel(_Channel):
        def __init__(self) -> None:
            super().__init__()
            self.inbound = [
                {"type": "state", "status": "idle"},
                {"type": "state", "status": "running"},
                {"type": "llm_request", "req_id": 1, "openai_body": {"messages": []}},
            ]

        def recv(self, timeout: float | None = None) -> JsonObject | None:
            if self.inbound:
                return super().recv(timeout)
            raise RuntimeError("Server disconnected")

    failed = _DisconnectedAfterLlmChannel()
    recovered = _Channel()
    recovered.inbound.insert(
        2, {"type": "llm_request", "req_id": 3, "openai_body": {"messages": []}}
    )
    channels = iter([failed, recovered])
    project = AgentProject(
        _Sandbox(),
        channel_factory=lambda sandbox, workspace: next(channels),
        owns_sandbox=False,
    )

    result = project.run(meta_agent(), _MeteredProvider(), "produce a result", timeout=1)

    assert result.answer == "finished"
    assert result.worker_usage.calls == 2
    assert result.worker_usage.input_tokens == 10
    assert result.worker_usage.output_tokens == 14
    assert result.worker_usage.call_input_tokens == [5, 5]
    assert result.worker_usage.call_output_tokens == [7, 7]


def test_project_replaces_owned_sandbox_and_restores_files_after_disconnect() -> None:
    """A poisoned E2B transport is replaced without losing the project archive."""

    class _DisconnectedChannel(_Channel):
        def __init__(self) -> None:
            super().__init__()
            self.inbound = [
                {"type": "state", "status": "idle"},
                {"type": "state", "status": "running"},
            ]

        def recv(self, timeout: float | None = None) -> JsonObject | None:
            if self.inbound:
                return super().recv(timeout)
            raise RuntimeError("Server disconnected")

    original = _Sandbox()
    replacement = _Sandbox()
    disconnected = _DisconnectedChannel()
    recovered = _Channel()
    replacement_calls = 0

    def sandbox_factory() -> _Sandbox:
        nonlocal replacement_calls
        replacement_calls += 1
        return replacement

    project = AgentProject(
        original,
        channel_factory=lambda sandbox, workspace: (
            recovered if sandbox is replacement else disconnected
        ),
        sandbox_factory=sandbox_factory,
    )
    project.write_text("history/round-1.json", '{"kept": true}')

    result = project.run(meta_agent(), _Provider(), "produce a result", timeout=1)

    archived = "/home/user/project/history/round-1.json"
    assert result.answer == "finished"
    assert replacement_calls == 1
    assert original.killed is True
    assert replacement.killed is False
    assert replacement.files.values[archived] == '{"kept": true}'
    assert project.read_text("history/round-1.json") == '{"kept": true}'
    assert project.usage().count == 2


def test_project_meters_overlapping_replacement_sandbox_lifetimes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replacement bootstrap time bills both the old and new live sandboxes."""
    ticks = iter([0.0, 10.0, 15.0, 30.0])
    monkeypatch.setattr(project_module.time, "monotonic", lambda: next(ticks))
    original = _Sandbox()
    replacement = _Sandbox()
    project = AgentProject(original, sandbox_factory=lambda: replacement)

    project._replace_sandbox()  # noqa: SLF001
    usage = project.usage()

    assert usage.count == 2
    assert usage.seconds == 35.0  # old: 0..15 plus replacement: 10..30


def test_project_meters_a_replacement_that_fails_during_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A created sandbox is billable even if replay fails before it becomes active."""

    class _BrokenCommands(_Commands):
        def run(self, cmd: str, background: bool | None = None, **kwargs: object) -> _Output:
            del cmd, background, kwargs
            raise RuntimeError("restore failed")

    ticks = iter([0.0, 10.0, 20.0, 30.0])
    monkeypatch.setattr(project_module.time, "monotonic", lambda: next(ticks))
    original = _Sandbox()
    replacement = _Sandbox()
    replacement.commands = _BrokenCommands()
    project = AgentProject(original, sandbox_factory=lambda: replacement)

    with pytest.raises(RuntimeError, match="restore failed"):
        project._replace_sandbox()  # noqa: SLF001
    usage = project.usage()

    assert replacement.killed is True
    assert usage.count == 2
    assert usage.seconds == 40.0  # original: 0..30 plus failed replacement: 10..20


def test_project_initialization_failure_releases_only_an_owned_sandbox() -> None:
    """Constructor setup cannot orphan an owned project or kill a caller-owned lease."""

    class _BrokenCommands(_Commands):
        def run(self, cmd: str, background: bool | None = None, **kwargs: object) -> _Output:
            del cmd, background, kwargs
            raise RuntimeError("workspace setup failed")

    owned = _Sandbox()
    owned.commands = _BrokenCommands()
    with pytest.raises(RuntimeError, match="workspace setup failed"):
        AgentProject(owned)
    assert owned.killed is True

    caller_owned = _Sandbox()
    caller_owned.commands = _BrokenCommands()
    with pytest.raises(RuntimeError, match="workspace setup failed"):
        AgentProject(caller_owned, owns_sandbox=False)
    assert caller_owned.killed is False


def test_project_failed_close_keeps_usage_live_and_retries_every_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed meta-project kill remains billable and retryable instead of looking final."""
    now = [0.0]
    monkeypatch.setattr(project_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(e2b_sandbox_module.time, "sleep", lambda delay: None)
    sandbox = _FlakyKillSandbox(failures=3)
    project = AgentProject(sandbox)

    now[0] = 5.0
    with pytest.raises(SandboxCleanupError, match="1 of 1") as raised:
        project.close()
    assert sandbox.kill_attempts == 3
    assert raised.value.resource == "meta_project_sandbox"
    assert raised.value.sandbox_usage == SandboxUsage(count=1, seconds=5.0)
    assert project.usage().seconds == 5.0

    now[0] = 8.0
    assert project.usage().seconds == 8.0
    project.close()
    assert sandbox.kill_attempts == 4
    assert sandbox.killed is True
    assert project.usage().seconds == 8.0
    project.close()


def test_project_retains_old_and_replacement_leases_after_failed_retirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recovery cannot drop the old sandbox handle when its kill is unproven."""
    now = [0.0]
    monkeypatch.setattr(project_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(e2b_sandbox_module.time, "sleep", lambda delay: None)
    original = _FlakyKillSandbox(failures=3)
    replacement = _Sandbox()
    project = AgentProject(original, sandbox_factory=lambda: replacement)

    now[0] = 10.0
    with pytest.raises(SandboxCleanupError, match="cleanup failed"):
        project._replace_sandbox()  # noqa: SLF001

    now[0] = 20.0
    usage = project.usage()
    assert usage.count == 2
    assert usage.seconds == 30.0  # old: 0..20 plus replacement: 10..20

    project.close()
    assert original.kill_attempts == 4
    assert original.killed is True
    assert replacement.killed is True
    assert project.usage().seconds == 30.0


def test_project_retries_a_clean_premature_session_end() -> None:
    """A clean EOF before the turn boundary is a recoverable runner lifecycle loss."""

    ended = _Channel()
    ended.inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "running"},
    ]
    recovered = _Channel()
    channels = iter([ended, recovered])
    project = AgentProject(
        _Sandbox(),
        channel_factory=lambda sandbox, workspace: next(channels),
        owns_sandbox=False,
    )

    result = project.run(meta_agent(), _Provider(), "produce a result", timeout=1)

    assert result.answer == "finished"
    assert ended.closed is True
    assert recovered.closed is False


def test_project_rejects_paths_that_escape_its_workspace() -> None:
    project = AgentProject(_Sandbox(), channel_factory=lambda sandbox, workspace: _Channel())

    try:
        project.write_text("../escape", "no")
    except ValueError as error:
        assert "relative project path" in str(error)
    else:
        raise AssertionError("path traversal should fail")


def test_agent_file_tools_reject_paths_outside_the_project() -> None:
    """Absolute and traversing agent paths cannot reach runner or sibling files."""
    project = AgentProject(_Sandbox(), channel_factory=lambda sandbox, workspace: _Channel())

    for path in ("/home/user/runner.js", "../runner.js"):
        outcome = project._execute_tool("read_file", {"path": path}, lambda stream, data: None)
        assert outcome.is_error is True
        assert "escapes project workspace" in outcome.content


@pytest.mark.parametrize("tool", ["read_skill"])
def test_project_rejects_agents_with_uncontained_tools(tool: str) -> None:
    """The project still rejects capabilities outside its isolated tool allowlist.

    `bash` moved INTO the allowlist for the harbor optimizer's proposer turns (it executes in
    the project sandbox as the default user), so only genuinely unrouted tools remain rejected.
    """
    base = meta_agent()
    uncontained = HarnessDoc(
        name="uncontained",
        surfaces=[
            surface.model_copy(update={"content": f"{tool}\nsubmit"})
            if surface.id == TOOL_POLICY_ID
            else surface
            for surface in base.surfaces
        ],
    )
    project = AgentProject(_Sandbox(), channel_factory=lambda sandbox, workspace: _Channel())

    with pytest.raises(ValueError, match=f"uncontained tools: {tool}"):
        project.run(uncontained, _Provider(), "escape", timeout=1)


# -- bash, stage/snapshot, and single-attempt runs (the harbor proposer surface) --------------


class _ScriptedCommands(_Commands):
    """Commands fake with scripted responses for bash and snapshot commands."""

    def __init__(self, files: _Files | None = None) -> None:
        super().__init__(files)
        self.responses: list[object] = []

    def run(self, cmd: str, background: bool | None = None, **kwargs: object) -> _Output:
        if self.responses and (cmd.startswith(("cd ", "python3 -c"))):
            self.runs.append(cmd)
            item = self.responses.pop(0)
            if isinstance(item, Exception):
                raise item
            assert isinstance(item, _Output)
            return item
        return super().run(cmd, background, **kwargs)


class _ExitError(RuntimeError):
    """E2B's CommandExitException shape: a raised failure carrying the finished streams."""

    def __init__(self, *, stdout: str = "", stderr: str = "", exit_code: int = 1) -> None:
        super().__init__(f"exit {exit_code}")
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


def _scripted_project(responses: list[object]) -> tuple[AgentProject, _ScriptedCommands]:
    sandbox = _Sandbox()
    commands = _ScriptedCommands(sandbox.files)
    sandbox.commands = commands
    project = AgentProject(
        sandbox, channel_factory=lambda sandbox, workspace: _Channel(), owns_sandbox=False
    )
    commands.responses = responses
    return project, commands


def test_bash_tool_reports_nonzero_exit_as_output_not_a_crash() -> None:
    project, commands = _scripted_project(
        [_ExitError(stdout="partial", stderr="boom", exit_code=2)]
    )

    outcome = project._execute_tool("bash", {"command": "false"}, lambda stream, data: None)

    assert outcome.is_error is True
    assert "partial" in outcome.content
    assert "boom" in outcome.content
    assert "[exit 2]" in outcome.content
    command = commands.runs[-1]
    assert command.startswith("cd /home/user/project && timeout --kill-after=10s 60s ")
    assert "bash --noprofile --norc -c false" in command


def test_run_bash_truncates_streams_head_and_tail() -> None:
    big = "a" * 20_000 + "MIDDLE" + "b" * 20_000
    project, _commands = _scripted_project([_Output(stdout=big)])

    result = project.run_bash("cat big")

    assert result.exit_code == 0
    assert "characters truncated" in result.stdout
    assert result.stdout.startswith("aaa")
    assert result.stdout.endswith("bbb")
    assert "MIDDLE" not in result.stdout


def test_run_bash_propagates_transport_failures() -> None:
    project, _commands = _scripted_project([RuntimeError("connection reset by peer")])

    with pytest.raises(RuntimeError, match="connection reset"):
        project.run_bash("ls")


def test_stage_and_snapshot_source_tree_roundtrip() -> None:
    tree = HarnessSourceTree(
        files=(
            HarnessSourceFile(path="SYSTEM.md", content="hello"),
            HarnessSourceFile(path="src/agent-loop.ts", content="export {};"),
        )
    )
    payload = json.dumps(
        {
            "files": [
                {
                    "path": item.path,
                    "content_base64": base64.b64encode(item.content.encode()).decode(),
                }
                for item in tree.files
            ]
        }
    )
    project, commands = _scripted_project([_Output(stdout=payload)])

    project.stage_source_tree(tree, "candidate")
    assert commands.files is not None
    assert commands.files.values["/home/user/project/candidate/SYSTEM.md"] == "hello"

    snapshot = project.snapshot_source_tree("candidate", max_files=10, max_bytes=1_000)
    assert snapshot == tree
    command = commands.runs[-1]
    assert command.startswith("python3 -c")
    assert command.endswith("/home/user/project/candidate 10 1000 1024")


def test_snapshot_source_tree_surfaces_in_sandbox_failures() -> None:
    project, _commands = _scripted_project(
        [_Output(stderr="directory exceeds the 10 file bound", exit_code=2)]
    )

    with pytest.raises(RuntimeError, match="10 file bound"):
        project.snapshot_source_tree("candidate", max_files=10, max_bytes=1_000)


def test_snapshot_script_walks_regular_files_and_counts_skipped_entries(
    tmp_path: Path,
) -> None:
    """The trusted in-sandbox walk: sorted regular files, symlinks REPORTED, bounds."""
    (tmp_path / "src").mkdir()
    (tmp_path / "SYSTEM.md").write_text("hi", encoding="utf-8")
    (tmp_path / "src" / "loop.ts").write_text("x", encoding="utf-8")
    (tmp_path / "link.md").symlink_to(tmp_path / "SYSTEM.md")
    (tmp_path / "link-dir").symlink_to(tmp_path / "src")
    script = project_module._SNAPSHOT_SOURCE_TREE_SCRIPT

    done = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), "10", "1000", "1024"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(done.stdout)
    assert [item["path"] for item in data["files"]] == ["SYSTEM.md", "src/loop.ts"]
    # Non-regular entries are counted, never silently dropped from the candidate.
    assert data["skipped"] == ["link-dir", "link.md"]

    over = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), "10", "1", "1024"],
        capture_output=True,
        text=True,
    )
    assert over.returncode == 2
    assert "byte bound" in over.stderr


def test_snapshot_source_tree_rejects_candidates_with_non_regular_entries() -> None:
    payload = json.dumps({"files": [], "skipped": ["candidate/link.md"]})
    project, _commands = _scripted_project([_Output(stdout=payload)])

    with pytest.raises(RuntimeError, match="candidate/link.md"):
        project.snapshot_source_tree("candidate", max_files=10, max_bytes=1_000)


def test_run_retry_recoverable_false_is_single_attempt_and_closes_the_session() -> None:
    """A recoverable transport death must not replay a paid proposal turn."""

    class _DisconnectChannel(_Channel):
        def __init__(self) -> None:
            super().__init__()
            self.inbound = [
                {"type": "state", "status": "idle"},
                {"type": "state", "status": "running"},
            ]

        def recv(self, timeout: float | None = None) -> JsonObject | None:
            if self.inbound:
                return super().recv(timeout)
            raise RuntimeError("server disconnected")

    replacement = _Sandbox()
    project = AgentProject(
        _Sandbox(),
        channel_factory=lambda sandbox, workspace: _DisconnectChannel(),
        sandbox_factory=lambda: pytest.fail("retry_recoverable=False must not replace"),
    )

    with pytest.raises(RuntimeError, match="server disconnected"):
        project.run(meta_agent(), _Provider(), "propose", timeout=1, retry_recoverable=False)

    assert replacement.killed is False
    assert project._session is None  # the poisoned session is never reused by a later run


def test_bash_capable_agents_must_run_single_attempt() -> None:
    """Shell writes bypass the replayable host mirror, so a transparently retried turn could
    replay onto a replacement sandbox missing them; run() rejects that combination outright
    instead of silently downgrading the retry."""
    project = AgentProject(_Sandbox(), channel_factory=lambda sandbox, workspace: _Channel())

    with pytest.raises(ValueError, match="retry_recoverable=False"):
        project.run(optimizer_agent(), _Provider(), "propose", timeout=1)

    result = project.run(
        optimizer_agent(), _Provider(), "propose", timeout=1, retry_recoverable=False
    )
    assert result.answer == "finished"
