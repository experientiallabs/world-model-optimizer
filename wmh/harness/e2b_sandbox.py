"""E2B sandbox plumbing for the e2b harness backend: protocol slice, creation, retries.

An E2B microVM is where a `pi-node` harness *process* executes under `backend="e2b"` — the
environment its tool calls hit stays whatever `AgentEnvironment` the eval binds (normally the
world-model simulation). This module owns only the sandbox mechanics: the exact protocol slice of
`e2b.Sandbox` the harness uses (so tests substitute fakes), the lazy-SDK default factory, and
capacity-shaped creation retries with fixed (1, 3, 9) s delays — the
`wmh.providers.retry.RetryingProvider` precedent, no RNG in scoring paths. The e2b SDK stays an
optional extra (`uv sync --extra e2b`).
"""

from __future__ import annotations

import importlib
import os
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Protocol, cast, runtime_checkable

from pydantic import BaseModel


class SandboxUsage(BaseModel):
    """E2B spend metrics for a pool: how many sandboxes ran, and their total lifetime seconds.

    Seconds are wall-clock sandbox lifetimes (create -> kill; live sandboxes count up to now),
    the unit E2B bills on. Pricing is the caller's concern (deployment-specific instance rates);
    this is the raw meter.
    """

    count: int = 0
    seconds: float = 0.0


E2B_API_KEY_ENV = "E2B_API_KEY"
E2B_TEMPLATE_ENV = "WMH_E2B_TEMPLATE"

# Sandbox lifetime. The sandbox only hosts the harness process (tool calls are answered by the
# environment host-side), so the bound is episode wall-time, not command time.
DEFAULT_SANDBOX_TIMEOUT_S = 900.0

# Fixed delays before each retry of sandbox creation (RetryingProvider precedent: 1s, 3s, 9s).
_CREATE_DELAYS = (1.0, 3.0, 9.0)
# Teardown is normally one cheap request. Two short deterministic retries cover a stale HTTP/2
# connection without adding latency to the success path or hiding a sandbox whose release cannot
# be proved.
_KILL_DELAYS = (0.1, 0.5)
_KILL_REQUEST_TIMEOUT_S = 5.0


class SandboxCleanupError(RuntimeError):
    """An E2B sandbox may still be live after bounded teardown retries."""

    def __init__(
        self,
        message: str,
        *,
        resource: str = "e2b_sandbox",
        sandbox_usage: SandboxUsage | None = None,
    ) -> None:
        super().__init__(message)
        self.resource = resource
        self.sandbox_usage = sandbox_usage


@runtime_checkable
class CommandOutput(Protocol):
    """The result slice of a finished sandbox command (e2b's `CommandResult` shape).

    `runtime_checkable` because e2b's `CommandExitException` *is* a `CommandResult` (non-zero
    exits raise instead of returning) — an isinstance check against this protocol recognizes it
    without importing the SDK.
    """

    stdout: str
    stderr: str
    exit_code: int


class CommandHandle(Protocol):
    """A background sandbox command (e2b's handle): stdin by pid, iteration yields stream events.

    Iteration events are `(stdout, stderr, pty)` chunks; `E2BPiRuntime` drives the RunnerLink
    frame stream over one.
    """

    @property
    def pid(self) -> int: ...

    def __iter__(self) -> Iterator[tuple[str | None, str | None, str | None]]: ...


class SandboxCommands(Protocol):
    """The `sandbox.commands` slice: run/connect commands and inject stdin."""

    def run(
        self,
        cmd: str,
        background: bool | None = None,
        *,
        envs: dict[str, str] | None = None,
        stdin: bool | None = None,
        timeout: float | None = None,
    ) -> CommandOutput | CommandHandle: ...

    def connect(
        self,
        pid: int,
        *,
        timeout: float | None = None,
    ) -> CommandHandle: ...

    def send_stdin(
        self,
        pid: int,
        data: str,
        request_timeout: float | None = None,
    ) -> object: ...

    def list(self, request_timeout: float | None = None) -> Sequence[SandboxProcess]: ...

    def kill(self, pid: int, request_timeout: float | None = None) -> object: ...


class SandboxProcess(Protocol):
    """The running-process field used to classify a durable runner stream EOF."""

    @property
    def pid(self) -> int: ...


class SandboxFiles(Protocol):
    """The `sandbox.files` slice: whole-file read and write."""

    def write(self, path: str, data: str) -> object: ...

    def read(
        self,
        path: str,
        *,
        request_timeout: float | None = None,
        gzip: bool = False,
    ) -> str: ...


@runtime_checkable
class SandboxHandle(Protocol):
    """The exact slice of `e2b.Sandbox` the harness uses, so tests substitute fakes."""

    @property
    def commands(self) -> SandboxCommands: ...

    @property
    def files(self) -> SandboxFiles: ...

    def set_timeout(self, timeout: int) -> None: ...

    def kill(self, request_timeout: float | None = None) -> object: ...


# Opens one sandbox. The default factory calls the real SDK; tests inject fakes.
SandboxFactory = Callable[[], SandboxHandle]


def default_sandbox_factory(
    *,
    api_key: str | None = None,
    template: str | None = None,
    timeout: float = DEFAULT_SANDBOX_TIMEOUT_S,
    metadata: dict[str, str] | None = None,
) -> SandboxFactory:
    """A factory creating real E2B sandboxes (lazy SDK import; key from arg or $E2B_API_KEY).

    `metadata` tags the sandbox at create time (e.g. `{"session_id": …}`) so an out-of-band sweep
    (`Sandbox.list`) can find and reap an orphaned sandbox whose owning process died — the live
    session driver relies on this for cost-leak reconciliation.
    """

    def make() -> SandboxHandle:
        try:
            e2b = importlib.import_module("e2b")
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "the e2b SDK is not installed; run `uv sync --extra e2b` to use the "
                "e2b harness backend"
            ) from exc
        key = api_key or os.environ.get(E2B_API_KEY_ENV)
        if not key:
            raise RuntimeError(f"set ${E2B_API_KEY_ENV} to run the harness in E2B sandboxes")
        chosen = template or os.environ.get(E2B_TEMPLATE_ENV) or None
        if metadata:
            sandbox = e2b.Sandbox.create(
                template=chosen, timeout=int(timeout), api_key=key, metadata=metadata
            )
        else:
            sandbox = e2b.Sandbox.create(template=chosen, timeout=int(timeout), api_key=key)
        # The SDK object satisfies the protocol slice structurally; cast rather than pin the
        # SDK's full (much wider) signatures into the protocol.
        return cast("SandboxHandle", sandbox)

    return make


def create_sandbox(factory: SandboxFactory) -> SandboxHandle:
    """Open one sandbox via `factory`, retrying capacity errors with fixed (1, 3, 9) s delays."""
    for delay in _CREATE_DELAYS:
        try:
            return factory()
        except Exception as exc:  # noqa: BLE001 - classified below; non-capacity re-raises
            if not _is_retryable_create_error(exc):
                raise
            time.sleep(delay)
    return factory()  # final attempt: let any error propagate


def kill_sandbox(sandbox: SandboxHandle) -> None:
    """Kill one sandbox with bounded retries, failing closed when release is unproven.

    A successful call, a falsey SDK result, or an explicit already-gone response all mean there
    is no live resource left to meter. Other exceptions are retried twice; exhausting that bound
    raises :class:`SandboxCleanupError` so callers cannot report clean cancellation while the
    sandbox may still be billable.
    """
    for delay in _KILL_DELAYS:
        try:
            sandbox.kill(request_timeout=_KILL_REQUEST_TIMEOUT_S)
            return
        except Exception as error:  # noqa: BLE001 - E2B SDK errors are optional/import-free here
            if _is_already_gone_error(error):
                return
            time.sleep(delay)
    try:
        sandbox.kill(request_timeout=_KILL_REQUEST_TIMEOUT_S)
    except Exception as error:  # noqa: BLE001 - promote the bounded cleanup failure uniformly
        if _is_already_gone_error(error):
            return
        sandbox_id = getattr(sandbox, "sandbox_id", None) or getattr(sandbox, "id", None)
        identity = f" {sandbox_id!r}" if sandbox_id is not None else ""
        raise SandboxCleanupError(
            f"E2B sandbox{identity} cleanup failed after {len(_KILL_DELAYS) + 1} attempts: {error}"
        ) from error


def _is_retryable_create_error(exc: Exception) -> bool:
    """True for capacity-shaped creation failures (rate limit / no capacity / 5xx).

    Matched by exception name and message so fakes need no SDK import; anything else (auth,
    bad template, missing key) fails immediately — retrying those only hides real bugs.
    """
    if type(exc).__name__ == "RateLimitException":  # e2b's 429
        return True
    text = str(exc).lower()
    if "rate limit" in text or "capacity" in text or "too many requests" in text:
        return True
    return any(code in text for code in ("429", "500", "502", "503", "504"))


def _is_already_gone_error(exc: Exception) -> bool:
    """Whether a failed kill explicitly proves the sandbox no longer exists."""
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "sandbox not found",
            "sandbox is not found",
            "sandbox already killed",
            "sandbox has been killed",
            "sandbox already closed",
            "sandbox has expired",
        )
    )
