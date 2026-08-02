"""Persistent E2B filesystem projects driven by the shared pi session runtime."""

from __future__ import annotations

import base64
import contextlib
import shlex
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol

from pydantic import BaseModel

from wmo.core.types import JsonObject
from wmo.harness.doc import HarnessDoc
from wmo.harness.e2b_sandbox import (
    SandboxCleanupError,
    SandboxFactory,
    SandboxHandle,
    SandboxUsage,
    create_sandbox,
    default_sandbox_factory,
    kill_sandbox,
)
from wmo.harness.live_session import (
    DEFAULT_ACTIONS_PER_TURN,
    LiveSession,
    SessionEvent,
    ToolOutcome,
)
from wmo.harness.pi_e2b import start_live_runner
from wmo.harness.runner_link import Channel, TokenUsage
from wmo.harness.runtime import HarnessSearchCancelled
from wmo.harness.source_tree import MAX_SOURCE_PATH_BYTES, HarnessSourceFile, HarnessSourceTree
from wmo.harness.tools import resolve_tools
from wmo.providers.base import ToolCallingProvider

PROJECT_WORKSPACE = "/home/user/project"
DEFAULT_PROJECT_TIMEOUT_S = 21_600
DEFAULT_SOURCE_TREE_MAX_FILES = 1_024
DEFAULT_SOURCE_TREE_MAX_BYTES = 8 * 1024 * 1024
_OUTPUT_CAP = 16_000
# Combined stdout+stderr budget for one bash command (head+tail truncated per stream).
_BASH_OUTPUT_CAP = 32 * 1024
_BASH_TIMEOUT_S = 60.0
_PROJECT_TOOLS = frozenset({"bash", "read_file", "write_file", "submit"})
_RECOVERABLE_SESSION_MARKERS = (
    "server disconnected",
    "connection reset",
    "connection closed",
    "broken pipe",
    "remoteprotocolerror",
    "readerror",
    "pi runner process exited",
    "pi live runner process exited",
    "durable outbox",
    "durable runner",
    "failed to send a frame to the e2b runner",
    "session ended before completing its turn",
    "live session runner did not become ready",
    "channel send failed",
)

# One trusted in-sandbox walk turning a project directory into bounded {path: base64} JSON.
# Only regular files are captured; every non-regular entry (symlink, socket, pipe) is COUNTED
# in the payload's `skipped` list so the host can reject a candidate that would otherwise be
# silently thinner than what the agent built. Path and byte bounds are enforced in-sandbox so
# the host never downloads more than the declared budget.
_SNAPSHOT_SOURCE_TREE_SCRIPT = r"""
import base64
import json
import os
import stat
import sys


def fail(message):
    sys.stderr.write(message + "\n")
    raise SystemExit(2)


root = sys.argv[1]
max_files = int(sys.argv[2])
max_bytes = int(sys.argv[3])
max_path_bytes = int(sys.argv[4])
if not os.path.isdir(root):
    fail("source directory does not exist: " + root)

files = []
skipped = []
total_bytes = 0
for current, directories, names in os.walk(root, topdown=True, followlinks=False):
    directories.sort()
    names.sort()
    for name in directories:
        path = os.path.join(current, name)
        if not stat.S_ISDIR(os.lstat(path).st_mode):
            skipped.append(os.path.relpath(path, root).replace(os.sep, "/"))
    for name in names:
        path = os.path.join(current, name)
        relative = os.path.relpath(path, root).replace(os.sep, "/")
        if not stat.S_ISREG(os.lstat(path).st_mode):
            skipped.append(relative)
            continue
        if len(relative.encode("utf-8")) > max_path_bytes:
            fail("file path exceeds " + str(max_path_bytes) + " bytes: " + relative)
        if len(files) >= max_files:
            fail("directory exceeds the " + str(max_files) + " file bound")
        with open(path, "rb") as source:
            content = source.read(max_bytes - total_bytes + 1)
        total_bytes += len(content)
        if total_bytes > max_bytes:
            fail("directory exceeds the " + str(max_bytes) + " byte bound")
        files.append(
            {
                "path": relative,
                "content_base64": base64.b64encode(content).decode("ascii"),
            }
        )

json.dump({"files": files, "skipped": sorted(skipped)}, sys.stdout, separators=(",", ":"))
"""


@dataclass(frozen=True)
class ProjectBashResult:
    """One finished project bash command: bounded streams plus its exit code."""

    stdout: str
    stderr: str
    exit_code: int


class _EncodedSourceFile(BaseModel):
    """One regular snapshot file encoded for bounded JSON transport."""

    path: str
    content_base64: str


class _EncodedSourceSnapshot(BaseModel):
    """The trusted sandbox script's wire representation."""

    files: tuple[_EncodedSourceFile, ...]
    # Non-regular entries (symlinks, sockets, pipes) the walk refused to capture. A candidate
    # containing them is invalid, not silently thinner, so the host raises on a nonzero count.
    skipped: tuple[str, ...] = ()


class ChannelFactory(Protocol):
    """Start one fresh runner channel in a project's sandbox."""

    def __call__(self, sandbox: SandboxHandle, workspace: str) -> Channel: ...


@dataclass(frozen=True)
class AgentProjectRun:
    """Result of one agent turn inside a project."""

    answer: str
    events: tuple[SessionEvent, ...]
    worker_usage: TokenUsage


class _ProjectAgentTurnError(RuntimeError):
    """A worker/provider error reported by a live agent turn, not its transport."""


class AgentProject:
    """A persistent filesystem that can run project-scoped pi agents.

    The project owns environment state, while :class:`LiveSession` owns ordinary agent execution.
    Repeated ``run`` calls for the same agent and provider reuse one live session and runner, while
    each outer project task gets a fresh model transcript. The project filesystem is the durable
    memory shared across those tasks.
    Changing the agent harness or provider starts a new session against the same filesystem.
    """

    def __init__(
        self,
        sandbox: SandboxHandle,
        *,
        workspace: str = PROJECT_WORKSPACE,
        channel_factory: ChannelFactory | None = None,
        sandbox_factory: SandboxFactory | None = None,
        owns_sandbox: bool = True,
    ) -> None:
        self._sandbox = sandbox
        self.workspace = workspace.rstrip("/")
        self._channel_factory = channel_factory or _start_channel
        # Replacing a caller-owned sandbox would exceed this object's authority. Injected test or
        # application sandboxes still get the bounded fresh-session retry in the same filesystem.
        self._sandbox_factory = sandbox_factory if owns_sandbox else None
        self._owns_sandbox = owns_sandbox
        self._active_sandbox_started_at = time.monotonic()
        self._retired_sandbox_seconds = 0.0
        self._sandbox_count = 1
        # A lease remains live until E2B confirms its kill. Replacement failures retain both
        # handles here so usage keeps accruing and close() can retry every unproven teardown.
        self._live_sandboxes: dict[int, tuple[SandboxHandle, float]] = {
            id(sandbox): (sandbox, self._active_sandbox_started_at)
        }
        self._closing = False
        self._finished_at: float | None = None
        # Keep an in-process mirror of mediated writes so a dead E2B transport can be replaced
        # without discarding the prior proposals that make this a persistent meta-agent project.
        self._file_contents: dict[str, str] = {}
        self._channel: Channel | None = None
        self._session: LiveSession | None = None
        self._session_agent_hash: str | None = None
        self._session_provider: ToolCallingProvider | None = None
        self._network_locked_sandbox_id: int | None = None
        self._active_event_sink: Callable[[SessionEvent], None] | None = None
        # ``None`` preserves the historical unrestricted project-tool behavior. A concrete set is
        # one logical run's exact, project-relative write grant; it is cleared even when the turn
        # fails so a reused live session cannot inherit the preceding turn's authority.
        self._active_writable_files: frozenset[str] | None = None
        self._retired_worker_usage = TokenUsage()
        try:
            self._initialize_sandbox(self._sandbox)
        except Exception as error:
            if self._owns_sandbox:
                try:
                    self._retire_sandbox(self._sandbox)
                except SandboxCleanupError as cleanup_error:
                    raise cleanup_error from error
            raise

    @classmethod
    def create(
        cls,
        *,
        timeout: float = DEFAULT_PROJECT_TIMEOUT_S,
        template: str | None = None,
        api_key: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> AgentProject:
        """Create one owned E2B project sandbox."""
        factory = default_sandbox_factory(
            timeout=timeout,
            template=template,
            api_key=api_key,
            metadata=metadata,
        )
        sandbox = create_sandbox(factory)
        return cls(sandbox, sandbox_factory=factory)

    def write_text(self, path: str, content: str) -> None:
        """Write one project-relative file without allowing path traversal."""
        if self._closing:
            raise RuntimeError("cannot write to a closed project")
        absolute = self._absolute_path(path)
        try:
            self._write_sandbox_file(self._sandbox, absolute, content)
        except Exception as error:
            # Proposer context is written before ``run()``, so its recovery loop cannot own an
            # exhausted control-plane retry. Replace an owned, transport-poisoned sandbox once,
            # replay the established mirror, and then apply this idempotent overwrite there.
            if self._sandbox_factory is None or not _is_recoverable_transport_error(error):
                raise
            try:
                self._replace_sandbox()
                self._write_sandbox_file(self._sandbox, absolute, content)
            except Exception as recovery_error:
                raise RuntimeError(
                    f"{error}; fresh project sandbox recovery failed: {recovery_error}"
                ) from recovery_error
        self._file_contents[self._relative_path(absolute)] = content

    def read_text(self, path: str) -> str:
        """Read one project-relative file."""
        if self._closing:
            raise RuntimeError("cannot read from a closed project")
        absolute = self._absolute_path(path)
        relative = self._relative_path(absolute)
        try:
            content = self._sandbox.files.read(absolute)
        except Exception:
            if relative in self._file_contents:
                return self._file_contents[relative]
            raise
        self._file_contents[relative] = content
        return content

    def run_bash(self, command: str, *, timeout_s: float = _BASH_TIMEOUT_S) -> ProjectBashResult:
        """Run one bounded shell command in the workspace as the default sandbox user.

        The command runs under ``timeout --kill-after`` inside a profile-free bash with the
        project workspace as its working directory. A nonzero exit is a RESULT, not an
        exception: E2B raises on nonzero exits with the streams attached to the exception, so
        that shape is unwrapped here. Only a transport failure (no exit code) propagates.
        Streams are head+tail truncated to a fixed combined budget.
        """
        if self._closing:
            raise RuntimeError("cannot run bash in a closed project")
        budget_s = max(1, int(timeout_s))
        script = (
            f"cd {shlex.quote(self.workspace)} && "
            f"timeout --kill-after=10s {budget_s}s "
            f"bash --noprofile --norc -c {shlex.quote(command)}"
        )
        result: object
        try:
            result = self._sandbox.commands.run(script, timeout=budget_s + 30)
        except Exception as error:  # noqa: BLE001 - E2B raises finished nonzero exits
            exit_code = getattr(error, "exit_code", None)
            if isinstance(exit_code, bool) or not isinstance(exit_code, int):
                raise  # a transport failure, not a finished command
            result = error
        return ProjectBashResult(
            stdout=_head_tail(str(getattr(result, "stdout", "") or ""), _BASH_OUTPUT_CAP // 2),
            stderr=_head_tail(str(getattr(result, "stderr", "") or ""), _BASH_OUTPUT_CAP // 2),
            exit_code=int(getattr(result, "exit_code", 0) or 0),
        )

    def stage_source_tree(
        self,
        tree: HarnessSourceTree,
        dest: str,
        *,
        max_files: int = DEFAULT_SOURCE_TREE_MAX_FILES,
        max_bytes: int = DEFAULT_SOURCE_TREE_MAX_BYTES,
    ) -> None:
        """Write one bounded, validated source tree under a project-relative directory."""
        tree.validate_bounds(max_files=max_files, max_bytes=max_bytes)
        self._absolute_path(dest)  # containment: reject absolute or traversing destinations
        for item in tree.files:
            self.write_text(f"{dest}/{item.path}", item.content)

    def snapshot_source_tree(
        self,
        directory: str,
        *,
        max_files: int = DEFAULT_SOURCE_TREE_MAX_FILES,
        max_bytes: int = DEFAULT_SOURCE_TREE_MAX_BYTES,
    ) -> HarnessSourceTree:
        """Capture one project directory as a bounded, validated host-side source tree.

        A single in-sandbox python walk emits every regular file as base64 JSON on stdout;
        the host decodes and validates it into a :class:`HarnessSourceTree`. Bound breaches,
        non-UTF-8 content, invalid paths, and non-regular entries (symlinks, sockets, pipes)
        all raise with the offending path in the message: a candidate is captured exactly or
        rejected, never silently thinner.
        """
        if self._closing:
            raise RuntimeError("cannot snapshot a source tree in a closed project")
        if isinstance(max_files, bool) or not isinstance(max_files, int) or max_files < 1:
            raise ValueError("max_files must be a positive integer")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        absolute = self._absolute_path(directory)
        command = (
            f"python3 -c {shlex.quote(_SNAPSHOT_SOURCE_TREE_SCRIPT)} {shlex.quote(absolute)} "
            f"{max_files} {max_bytes} {MAX_SOURCE_PATH_BYTES}"
        )
        try:
            result = self._sandbox.commands.run(command, timeout=120)
        except Exception as error:  # noqa: BLE001 - E2B raises command failures
            detail = str(getattr(error, "stderr", "") or error)
            raise RuntimeError(
                f"project source snapshot failed: {_head_tail(detail, 4_096)}"
            ) from error
        exit_code = int(getattr(result, "exit_code", 0) or 0)
        if exit_code != 0:
            stderr = str(getattr(result, "stderr", "") or "snapshot command failed")
            raise RuntimeError(f"project source snapshot failed: {_head_tail(stderr, 4_096)}")
        try:
            tree = _decode_source_tree_snapshot(str(getattr(result, "stdout", "")))
        except ValueError as error:
            raise RuntimeError(f"project source snapshot is invalid: {error}") from error
        tree.validate_bounds(max_files=max_files, max_bytes=max_bytes)
        return tree

    def run(
        self,
        agent: HarnessDoc,
        provider: ToolCallingProvider,
        instruction: str,
        *,
        timeout: float = DEFAULT_PROJECT_TIMEOUT_S,
        on_event: Callable[[SessionEvent], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        writable_files: Collection[str] | None = None,
        retry_recoverable: bool = True,
    ) -> AgentProjectRun:
        """Run one turn of an ordinary agent against this persistent project.

        A transient runner-channel disconnect retries the turn once. Owned E2B
        projects replace a transport-poisoned sandbox and replay their mirrored
        filesystem first; injected test projects keep the sandbox and replace
        only the ordinary live session. ``writable_files`` optionally grants the
        agent's ``write_file`` tool access to exact project-relative files for
        this logical run. Omitting it preserves unrestricted project writes;
        an empty collection denies every agent write. Host ``write_text`` calls
        are not constrained by an agent turn's grant. ``retry_recoverable=False``
        makes the run exactly one agent attempt: a transport failure closes the
        session and propagates instead of transparently replaying paid work.
        A bash-capable agent REQUIRES it: shell writes bypass the replayable
        host mirror, so a replayed turn would land on a replacement sandbox
        missing those edits, and that mismatch is rejected here, not silently
        downgraded.
        """
        if self._closing:
            raise RuntimeError("cannot run an agent in a closed project")
        _check_cancelled(should_cancel)
        if self._active_event_sink is not None:
            raise RuntimeError("a project agent turn is already running")
        unsupported = set(agent.tools()) - _PROJECT_TOOLS
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"project agents cannot use uncontained tools: {names}")
        if retry_recoverable and "bash" in agent.tools():
            raise ValueError(
                "bash-capable project agents must run with retry_recoverable=False: shell "
                "writes bypass the replayable host file mirror, so a transparently retried "
                "turn could replay onto a replacement sandbox missing those edits"
            )
        write_grant = self._normalize_writable_files(writable_files)
        usage_before = self._total_worker_usage()
        self._active_writable_files = write_grant
        max_attempts = 2 if retry_recoverable else 1
        try:
            for attempt in range(max_attempts):
                try:
                    result = self._run_turn(
                        agent,
                        provider,
                        instruction,
                        timeout=timeout,
                        on_event=on_event,
                        should_cancel=should_cancel,
                    )
                    usage_after = self._total_worker_usage()
                    return AgentProjectRun(
                        answer=result.answer,
                        events=result.events,
                        worker_usage=_usage_delta(usage_after, usage_before),
                    )
                except HarnessSearchCancelled:
                    raise
                except Exception as error:
                    if not _is_recoverable_session_error(error):
                        raise
                    if attempt + 1 >= max_attempts:
                        # A transport-poisoned session is never reused by a later logical run,
                        # even when the caller deliberately owns recovery at a higher level.
                        self._close_agent_session()
                        raise
                    if self._sandbox_factory is None:
                        self._close_agent_session()
                        continue
                    try:
                        self._replace_sandbox()
                    except Exception as recovery_error:
                        raise RuntimeError(
                            f"{error}; fresh project sandbox recovery failed: {recovery_error}"
                        ) from recovery_error
            raise AssertionError("unreachable")
        finally:
            self._active_writable_files = None

    def _run_turn(
        self,
        agent: HarnessDoc,
        provider: ToolCallingProvider,
        instruction: str,
        *,
        timeout: float,
        on_event: Callable[[SessionEvent], None] | None,
        should_cancel: Callable[[], bool] | None,
    ) -> AgentProjectRun:
        """Execute one attempt using the compatible ordinary live session."""
        session = self._ensure_session(agent, provider)
        events: list[SessionEvent] = []
        answer = ""
        turn_started = False
        turn_running = False
        turn_finished = False
        turn_terminal_reason: str | None = None
        turn_error: str | None = None

        def sink(event: SessionEvent) -> None:
            nonlocal answer, turn_error, turn_finished, turn_running, turn_terminal_reason
            events.append(event)
            if event.kind == "submit":
                submitted = event.payload.get("answer")
                answer = submitted if isinstance(submitted, str) else ""
            elif event.kind == "error" and turn_error is None:
                message = event.payload.get("message")
                turn_error = message if isinstance(message, str) else "project agent session error"
            elif turn_started and event.kind == "state":
                status = event.payload.get("status")
                if status == "running":
                    turn_running = True
                elif status == "idle" and turn_running:
                    turn_finished = True
                    reason = event.payload.get("reason")
                    turn_terminal_reason = reason if isinstance(reason, str) else None
            if on_event is not None:
                on_event(event)

        self._active_event_sink = sink
        try:
            session.send_user_message(instruction)
            turn_started = True
            deadline = time.monotonic() + timeout
            while not turn_finished:
                self._cancel_turn_if_requested(session, should_cancel)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    session.interrupt("project_run_timeout")
                    session.flush_pending_intents()
                    # An abort acknowledgement can arrive after this deadline. Retiring the
                    # session prevents that stale idle boundary from completing the next turn.
                    self._close_agent_session()
                    raise TimeoutError(f"project agent did not finish within {timeout:g}s")
                running = session.pump(timeout=min(0.5, remaining))
                # A pump can synchronously run one provider completion. Observe cancellation as
                # soon as it returns, before consuming a second model or tool request.
                self._cancel_turn_if_requested(session, should_cancel)
                if not running and not turn_finished:
                    if session.failure_message is not None:
                        raise RuntimeError(
                            f"project agent session failed: {session.failure_message}"
                        )
                    raise RuntimeError("project agent session ended before completing its turn")
            if turn_error is not None:
                raise _ProjectAgentTurnError(f"project agent session failed: {turn_error}")
            if turn_terminal_reason in {"aborted", "turn_limit"}:
                raise _ProjectAgentTurnError(
                    f"project agent turn ended with reason: {turn_terminal_reason}"
                )
        finally:
            self._active_event_sink = None
        return AgentProjectRun(answer=answer, events=tuple(events), worker_usage=TokenUsage())

    def _cancel_turn_if_requested(
        self,
        session: LiveSession,
        should_cancel: Callable[[], bool] | None,
    ) -> None:
        """Abort and retire the active session at one cooperative cancellation boundary."""
        if should_cancel is None or not should_cancel():
            return
        session.interrupt("harness_search_cancelled")
        with contextlib.suppress(Exception):
            session.flush_pending_intents()
        self._close_agent_session()
        raise HarnessSearchCancelled("harness search cancelled")

    def _ensure_session(self, agent: HarnessDoc, provider: ToolCallingProvider) -> LiveSession:
        """Return the compatible live session, starting one when the harness changed."""
        if (
            self._session is not None
            and not self._session.closed
            and self._session_agent_hash == agent.doc_hash
            and self._session_provider is provider
        ):
            return self._session
        self._close_agent_session()
        channel = self._channel_factory(self._sandbox, self.workspace)
        try:
            # Runner bootstrap has completed in channel_factory, but no agent-controlled source
            # has been imported yet. Remove egress before session_start materializes that code.
            self._lock_project_network()
            skills = agent.skills()
            session = LiveSession(
                channel,
                tools=resolve_tools(agent.tools()),
                execute_tool=self._execute_tool,
                on_event=self._emit_session_event,
                files={
                    surface.path: surface.content for surface in agent.code_files() if surface.path
                },
                system_prompt=agent.assembled_prompt(),
                skill_bodies={skill.name: skill.body for skill in skills},
                provider=provider,
                # Project agents explore a durable filesystem and can legitimately need one
                # project action per model turn. Never let LiveSession's generic 40-action default
                # silently undercut a harness that explicitly raises its turn budget.
                actions_per_turn=max(DEFAULT_ACTIONS_PER_TURN, agent.max_turns()),
                turn_cap=agent.max_turns(),
                max_output_tokens=agent.max_output_tokens(),
                temperature=agent.temperature(),
                # Project files are durable memory. Replaying every prior project task in the
                # model transcript only duplicates that state and eventually collapses pi's
                # available output budget as context fills.
                conversation_scope="turn",
            )
            session.start()
        except Exception:
            close = getattr(channel, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()
            raise
        self._channel = channel
        self._session = session
        self._session_agent_hash = agent.doc_hash
        self._session_provider = provider
        return session

    def _lock_project_network(self) -> None:
        """Remove internet egress before untrusted project evidence can drive tools."""
        if not self._owns_sandbox or self._network_locked_sandbox_id == id(self._sandbox):
            return
        update_network = getattr(self._sandbox, "update_network", None)
        if not callable(update_network):
            raise RuntimeError("owned project sandbox cannot disable internet access")
        update_network({"allow_internet_access": False})
        self._network_locked_sandbox_id = id(self._sandbox)

    def _emit_session_event(self, event: SessionEvent) -> None:
        """Route session events to the currently active project turn."""
        if self._active_event_sink is not None:
            self._active_event_sink(event)

    def _close_agent_session(self) -> None:
        """Close the current agent session without touching the project filesystem."""
        session = self._session
        channel = self._channel
        self._session = None
        self._channel = None
        self._session_agent_hash = None
        self._session_provider = None
        if session is not None:
            self._retired_worker_usage.input_tokens += session.worker_usage.input_tokens
            self._retired_worker_usage.output_tokens += session.worker_usage.output_tokens
            self._retired_worker_usage.cached_input_tokens += (
                session.worker_usage.cached_input_tokens
            )
            self._retired_worker_usage.cache_write_input_tokens += (
                session.worker_usage.cache_write_input_tokens
            )
            self._retired_worker_usage.reasoning_tokens += session.worker_usage.reasoning_tokens
            self._retired_worker_usage.calls += session.worker_usage.calls
            self._retired_worker_usage.call_seconds.extend(session.worker_usage.call_seconds)
            self._retired_worker_usage.call_input_tokens.extend(
                session.worker_usage.call_input_tokens
            )
            self._retired_worker_usage.call_output_tokens.extend(
                session.worker_usage.call_output_tokens
            )
            self._retired_worker_usage.call_cached_input_tokens.extend(
                session.worker_usage.call_cached_input_tokens
            )
            self._retired_worker_usage.call_cache_write_input_tokens.extend(
                session.worker_usage.call_cache_write_input_tokens
            )
        close = getattr(channel, "close", None)
        if callable(close):
            with contextlib.suppress(Exception):
                close()
        elif session is not None and not session.closed:
            # Test/local channels without an owned close hook still get the protocol-level end.
            # Real project channels close the runner directly above so cancellation never waits
            # for two durable abort/shutdown acknowledgements from an unreachable process.
            with contextlib.suppress(Exception):
                session.end()
                session.pump(timeout=0)

    def usage(self) -> SandboxUsage:
        """Return this project's sandbox lifetime meter."""
        now = time.monotonic()
        active_seconds = sum(
            max(0.0, now - started_at) for _sandbox, started_at in self._live_sandboxes.values()
        )
        return SandboxUsage(
            count=self._sandbox_count,
            seconds=self._retired_sandbox_seconds + active_seconds,
        )

    def _total_worker_usage(self) -> TokenUsage:
        """Return worker usage across retired and currently attached live sessions."""
        current = self._session.worker_usage if self._session is not None else TokenUsage()
        return TokenUsage(
            input_tokens=self._retired_worker_usage.input_tokens + current.input_tokens,
            output_tokens=self._retired_worker_usage.output_tokens + current.output_tokens,
            cached_input_tokens=(
                self._retired_worker_usage.cached_input_tokens + current.cached_input_tokens
            ),
            cache_write_input_tokens=(
                self._retired_worker_usage.cache_write_input_tokens
                + current.cache_write_input_tokens
            ),
            reasoning_tokens=(
                self._retired_worker_usage.reasoning_tokens + current.reasoning_tokens
            ),
            calls=self._retired_worker_usage.calls + current.calls,
            call_seconds=[
                *self._retired_worker_usage.call_seconds,
                *current.call_seconds,
            ],
            call_input_tokens=[
                *self._retired_worker_usage.call_input_tokens,
                *current.call_input_tokens,
            ],
            call_output_tokens=[
                *self._retired_worker_usage.call_output_tokens,
                *current.call_output_tokens,
            ],
            call_cached_input_tokens=[
                *self._retired_worker_usage.call_cached_input_tokens,
                *current.call_cached_input_tokens,
            ],
            call_cache_write_input_tokens=[
                *self._retired_worker_usage.call_cache_write_input_tokens,
                *current.call_cache_write_input_tokens,
            ],
        )

    def close(self) -> None:
        """Release every owned lease, retaining unproven kills for a later retry."""
        if self._finished_at is not None:
            return
        self._closing = True
        self._close_agent_session()
        if not self._owns_sandbox:
            finished_at = time.monotonic()
            for _sandbox, started_at in self._live_sandboxes.values():
                self._retired_sandbox_seconds += max(0.0, finished_at - started_at)
            self._live_sandboxes.clear()
            self._finished_at = finished_at
            return

        leases = list(self._live_sandboxes.values())
        failures: list[SandboxCleanupError] = []
        for sandbox, _started_at in leases:
            try:
                self._retire_sandbox(sandbox)
            except SandboxCleanupError as error:
                failures.append(error)
        if failures:
            raise SandboxCleanupError(
                "failed to prove cleanup for "
                f"{len(failures)} of {len(leases)} "
                "meta-project E2B sandboxes",
                resource="meta_project_sandbox",
                sandbox_usage=self.usage(),
            ) from failures[0]
        self._finished_at = time.monotonic()

    def __enter__(self) -> AgentProject:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _absolute_path(self, path: str) -> str:
        candidate = PurePosixPath(path)
        if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
            raise ValueError(f"expected a relative project path, got {path!r}")
        return f"{self.workspace}/{candidate.as_posix()}"

    def _relative_path(self, absolute: str) -> str:
        """Return one already-contained absolute path relative to the project root."""
        return PurePosixPath(absolute).relative_to(PurePosixPath(self.workspace)).as_posix()

    def _initialize_sandbox(self, sandbox: SandboxHandle) -> None:
        """Create the workspace and replay the authoritative project-file mirror."""
        sandbox.commands.run(f"mkdir -p {shlex.quote(self.workspace)}", timeout=30)
        for relative, content in self._file_contents.items():
            absolute = f"{self.workspace}/{relative}"
            self._write_sandbox_file(sandbox, absolute, content)

    @staticmethod
    def _write_sandbox_file(sandbox: SandboxHandle, absolute: str, content: str) -> None:
        directory = str(PurePosixPath(absolute).parent)
        for attempt in range(2):
            try:
                sandbox.commands.run(f"mkdir -p {shlex.quote(directory)}", timeout=30)
                sandbox.files.write(absolute, content)
                return
            except Exception as error:  # noqa: BLE001 - classify the E2B transport boundary
                # Both operations are idempotent: replaying ``mkdir -p`` and the same overwrite is
                # safe even when the first request reached E2B but its response was disconnected.
                # Keep the live project sandbox/session intact for a one-off control-plane drop.
                if attempt > 0 or not _is_recoverable_transport_error(error):
                    raise

    def _replace_sandbox(self) -> None:
        """Replace a transport-poisoned sandbox while retaining every project file."""
        factory = self._sandbox_factory
        if factory is None:
            raise RuntimeError("project sandbox replacement is unavailable")
        # Required durable files are synchronously mirrored by write_text/write_file. Bash is
        # explicitly scratch-only, so recovery never scans or replays an unbounded agent-created
        # tree before honoring cancellation or replacing a poisoned transport.
        replacement = create_sandbox(factory)
        replacement_started_at = time.monotonic()
        self._sandbox_count += 1
        self._live_sandboxes[id(replacement)] = (replacement, replacement_started_at)
        try:
            self._initialize_sandbox(replacement)
        except Exception as error:
            try:
                self._retire_sandbox(replacement)
            except SandboxCleanupError as cleanup_error:
                raise cleanup_error from error
            raise

        previous = self._sandbox
        self._close_agent_session()
        self._active_sandbox_started_at = replacement_started_at
        self._sandbox = replacement
        self._network_locked_sandbox_id = None
        if self._owns_sandbox:
            self._retire_sandbox(previous)

    def _retire_sandbox(self, sandbox: SandboxHandle) -> None:
        """Finalize one lease only after E2B confirms that it is gone."""
        lease = self._live_sandboxes.get(id(sandbox))
        if lease is None:
            return
        kill_sandbox(sandbox)
        retired_at = time.monotonic()
        _handle, started_at = self._live_sandboxes.pop(id(sandbox))
        self._retired_sandbox_seconds += max(0.0, retired_at - started_at)

    def _execute_tool(
        self,
        name: str,
        arguments: JsonObject,
        emit: Callable[[str, str], None],
    ) -> ToolOutcome:
        del emit  # Project tools return one bounded observation; they do not stream output.
        try:
            if name == "bash":
                # Agent bash writes bypass the host-side file mirror, so a sandbox replacement
                # would silently lose them. That is sound only because run() ENFORCES that a
                # bash-capable agent runs with retry_recoverable=False: a dead transport ends
                # the turn instead of replaying it onto a replacement sandbox.
                result = self.run_bash(str(arguments.get("command", "")))
                body = result.stdout
                if result.stderr:
                    body = f"{body}\n[stderr]\n{result.stderr}" if body else result.stderr
                if result.exit_code != 0:
                    marker = f"[exit {result.exit_code}]"
                    body = f"{body}\n{marker}" if body else marker
                return ToolOutcome(content=body, is_error=result.exit_code != 0)
            if name == "read_file":
                path = self._tool_path(str(arguments.get("path", "")))
                relative = self._relative_path(path)
                try:
                    content = self._sandbox.files.read(path)
                except Exception:
                    if relative not in self._file_contents:
                        raise
                    content = self._file_contents[relative]
                else:
                    self._file_contents[relative] = content
                return _capped(content)
            if name == "write_file":
                path = self._tool_path(str(arguments.get("path", "")))
                relative = self._relative_path(path)
                if (
                    self._active_writable_files is not None
                    and relative not in self._active_writable_files
                ):
                    raise PermissionError(
                        f"path is not writable in this project turn: {relative!r}"
                    )
                content = str(arguments.get("content", ""))
                self._write_sandbox_file(self._sandbox, path, content)
                self._file_contents[relative] = content
                return ToolOutcome(content=f"wrote {path}")
        except Exception as error:  # noqa: BLE001 - tool errors are agent observations
            return ToolOutcome(content=f"{name} failed: {error}", is_error=True)
        return ToolOutcome(content=f"tool {name!r} not available", is_error=True)

    def _tool_path(self, path: str) -> str:
        """Resolve an agent-supplied path while containing it to the project."""
        candidate = PurePosixPath(path)
        workspace = PurePosixPath(self.workspace)
        if candidate.is_absolute():
            try:
                candidate = candidate.relative_to(workspace)
            except ValueError as error:
                raise ValueError(f"path escapes project workspace: {path!r}") from error
        if not candidate.parts or ".." in candidate.parts:
            raise ValueError(f"path escapes project workspace: {path!r}")
        return str(workspace / candidate)

    def _normalize_writable_files(
        self, writable_files: Collection[str] | None
    ) -> frozenset[str] | None:
        """Normalize one optional exact-file grant to project-relative paths."""
        if writable_files is None:
            return None
        return frozenset(self._relative_path(self._absolute_path(path)) for path in writable_files)


def _start_channel(sandbox: SandboxHandle, workspace: str) -> Channel:
    # Project turns can be separated by long evaluation waves. Their ordinary live runner writes
    # every semantic output frame to a sequenced E2B outbox before stdout, so the shared
    # LiveSession can replay a dropped command stream without replacing the agent, transcript, or
    # project sandbox. Platform live sessions keep start_live_runner's established stdio default.
    return start_live_runner(sandbox, workspace=workspace, durable_outbox=True)


def _is_recoverable_session_error(error: Exception) -> bool:
    """Return whether one fresh live session may recover this transport failure."""
    if isinstance(error, _ProjectAgentTurnError):
        return False
    return _is_recoverable_transport_error(error)


def _is_recoverable_transport_error(error: Exception) -> bool:
    """Return whether one idempotent E2B transport operation may be retried once."""
    error_type = type(error)
    if error_type.__module__ == "e2b.exceptions" and error_type.__name__ == "TimeoutException":
        return True
    text = str(error).lower()
    # httpcore can race an E2B HTTP/2 GOAWAY with request body delivery. h2 then surfaces a raw
    # ProtocolError instead of httpx's usual transport wrapper. The pool will not reassign that
    # unavailable closed connection, so the next idempotent control-plane request opens a fresh
    # one. Match the state-machine shape rather than every h2 ProtocolError: malformed responses
    # remain fatal.
    if "invalid input connectioninputs." in text and "connectionstate.closed" in text:
        return True
    return any(marker in text for marker in _RECOVERABLE_SESSION_MARKERS)


def _check_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    """Fail before creating or retrying a project turn when search cancellation is already set."""
    if should_cancel is not None and should_cancel():
        raise HarnessSearchCancelled("harness search cancelled")


def _usage_delta(after: TokenUsage, before: TokenUsage) -> TokenUsage:
    """Subtract cumulative usage snapshots for one logical project run."""
    return TokenUsage(
        input_tokens=after.input_tokens - before.input_tokens,
        output_tokens=after.output_tokens - before.output_tokens,
        cached_input_tokens=after.cached_input_tokens - before.cached_input_tokens,
        cache_write_input_tokens=(after.cache_write_input_tokens - before.cache_write_input_tokens),
        reasoning_tokens=after.reasoning_tokens - before.reasoning_tokens,
        calls=after.calls - before.calls,
        call_seconds=after.call_seconds[len(before.call_seconds) :],
        call_input_tokens=after.call_input_tokens[len(before.call_input_tokens) :],
        call_output_tokens=after.call_output_tokens[len(before.call_output_tokens) :],
        call_cached_input_tokens=after.call_cached_input_tokens[
            len(before.call_cached_input_tokens) :
        ],
        call_cache_write_input_tokens=after.call_cache_write_input_tokens[
            len(before.call_cache_write_input_tokens) :
        ],
    )


def _capped(content: str, *, is_error: bool = False) -> ToolOutcome:
    if len(content) <= _OUTPUT_CAP:
        return ToolOutcome(content=content, is_error=is_error)
    half = _OUTPUT_CAP // 2
    marker = f"\n... {len(content) - _OUTPUT_CAP} characters truncated ...\n"
    return ToolOutcome(
        content=f"{content[:half]}{marker}{content[-half:]}",
        is_error=is_error,
        truncated=True,
    )


def _head_tail(content: str, cap: int) -> str:
    """Keep both ends of one stream within `cap` characters, with an omission marker."""
    if len(content) <= cap:
        return content
    half = cap // 2
    omitted = len(content) - cap
    return f"{content[:half]}\n... {omitted} characters truncated ...\n{content[-half:]}"


def _decode_source_tree_snapshot(payload: str) -> HarnessSourceTree:
    """Decode bounded base64 JSON from the trusted snapshot script."""
    encoded = _EncodedSourceSnapshot.model_validate_json(payload)
    if encoded.skipped:
        names = ", ".join(encoded.skipped)
        raise ValueError(
            f"directory contains {len(encoded.skipped)} non-regular entr(y/ies) that cannot "
            f"be part of a portable source tree: {names}; replace them with regular files"
        )
    files: list[HarnessSourceFile] = []
    for item in encoded.files:
        try:
            content = base64.b64decode(item.content_base64, validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError(f"file {item.path!r} is not valid UTF-8 text") from error
        files.append(HarnessSourceFile(path=item.path, content=content))
    return HarnessSourceTree(files=tuple(files))
