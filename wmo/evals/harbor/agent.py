"""Harbor agent bridge for running an exact WMO harness document.

Harbor instantiates `WmoHarborAgent` from an import path plus JSON kwargs, so one serialized
candidate (the `HarnessDoc`), one provider config, and the execution knobs travel through
harbor's own trial machinery unchanged. The agent rebuilds the runtime host-side and drives it
against `HarborAgentEnvironment`, a synchronous adapter over harbor's async task environment:
the real container is the environment; the worker LLM and tool routing stay host-side.

Two properties here are load-bearing for the optimizer:

- The WMO transcript (`wmo-run.json` in harbor's logs_dir) is written in a ``finally``, so a
  trial cancelled by harbor's agent timeout still leaves whatever steps executed plus a
  ``stop_reason: "cancelled-by-harbor-timeout"`` marker. Timeout trials are the most informative
  failures a proposer sees; losing their transcripts would blind it.
- Episodes and cleanup run on a dedicated process-wide `ThreadPoolExecutor` sized at least as
  large as agent concurrency, never on ``asyncio.to_thread``'s default executor: with high agent
  concurrency the default pool (min(32, cpus + 4)) would queue episodes whose harbor timeout
  clocks are already running.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import shlex
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig

from wmo.core.types import Action, ActionKind, JsonObject, Observation
from wmo.harness.doc import HarnessDoc
from wmo.harness.environment import is_env_action
from wmo.harness.runtime import (
    DEFAULT_EVAL_EPISODE_TIMEOUT_S,
    RunResult,
    validate_episode_timeout_s,
)
from wmo.providers.base import Provider, ProviderConfig
from wmo.providers.registry import get_provider
from wmo.providers.retry import wrap_provider_with_retries

WMO_HARBOR_AGENT_VERSION = "1"
WMO_HARBOR_AGENT_IMPORT_PATH = "wmo.evals.harbor.agent:WmoHarborAgent"
# Keep every task command finite and leave cleanup headroom beneath the local Pi process cap. The
# local shim also joins active request handlers, so a late-starting command cannot outlive runtime
# return; E2B Pi already blocks synchronously on the same bridge.
MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC = 240
# Default size of the dedicated episode executor. It must stay >= the harbor agent concurrency
# plus in-flight cleanup, or queued episodes burn their harbor timeout budget before starting.
DEFAULT_EPISODE_WORKERS = 64
_TRACE_FILENAME = "wmo-run.json"
_CANCELLED_STOP_REASON = "cancelled-by-harbor-timeout"
_WRITE_COMMAND = (
    'mkdir -p -- "$(dirname -- "$WMO_FILE_PATH")" && '
    'printf \'%s\' "$WMO_FILE_CONTENT_B64" | base64 -d > "$WMO_FILE_PATH"'
)

_EXECUTOR_LOCK = threading.Lock()
_EPISODE_EXECUTOR: ThreadPoolExecutor | None = None
_EPISODE_EXECUTOR_WORKERS = 0


def _episode_executor(min_workers: int) -> ThreadPoolExecutor:
    """The process-wide executor for episode and cleanup work, grown to `min_workers`.

    One executor serves every concurrently running WmoHarborAgent in this process. When a job
    asks for more workers than the current pool has, a larger pool replaces it; the old pool
    keeps draining its in-flight work and is dropped without shutdown (process-lifetime infra).
    """
    global _EPISODE_EXECUTOR, _EPISODE_EXECUTOR_WORKERS
    with _EXECUTOR_LOCK:
        if _EPISODE_EXECUTOR is None or _EPISODE_EXECUTOR_WORKERS < min_workers:
            _EPISODE_EXECUTOR = ThreadPoolExecutor(
                max_workers=min_workers,
                thread_name_prefix="wmo-harbor-episode",
            )
            _EPISODE_EXECUTOR_WORKERS = min_workers
        return _EPISODE_EXECUTOR


class HarborAgentEnvironment:
    """Expose Harbor's async task environment through WMO's synchronous protocol.

    Every executed step is also recorded so a cancelled episode can still persist the partial
    transcript the optimizer's proposer feeds on.
    """

    def __init__(
        self,
        event_loop: asyncio.AbstractEventLoop,
        environment: BaseEnvironment,
        *,
        command_timeout_sec: int = MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC,
    ) -> None:
        self._event_loop = event_loop
        self._environment = environment
        self._command_timeout_sec = _validate_command_timeout_sec(command_timeout_sec)
        self._recorded_steps: list[JsonObject] = []

    def execute(self, action: Action) -> Observation:
        """Execute one supported WMO tool in Harbor's owned task environment.

        A command that times out or dies on a transport error is a CANDIDATE outcome, not an
        infrastructure failure: it becomes an error observation the agent can react to. Letting
        it escape as an exception would kill the episode before verification, turn the whole
        candidate into an unscoreable HarborRewardMissingError, and (because the pruner deletes
        reward-less failed trials) make the deterministic job dir re-run it forever.
        """
        try:
            observation = self._execute(action)
        except Exception as exc:  # noqa: BLE001 - env failures are episode feedback, never fatal
            observation = Observation(
                content=f"environment command failed: {type(exc).__name__}: {exc}",
                is_error=True,
            )
        self._recorded_steps.append(
            {
                "action": action.model_dump(mode="json"),
                "observation": observation.model_dump(mode="json"),
            }
        )
        return observation

    def recorded_steps(self) -> list[JsonObject]:
        """The steps executed so far (the partial-transcript source on cancellation)."""
        return list(self._recorded_steps)

    def close(self) -> None:
        """Leave lifecycle ownership with Harbor."""

    def _execute(self, action: Action) -> Observation:
        if action.kind is not ActionKind.TOOL_CALL or not is_env_action(action):
            return Observation(content=f"tool {action.name!r} not available", is_error=True)
        arguments = action.arguments or {}
        if action.name == "bash":
            command = _string_argument(arguments, "command")
            if command is None:
                return _invalid_arguments("bash", "command must be a string")
            return _command_observation(self._exec(command))
        if action.name == "read_file":
            path = _string_argument(arguments, "path", nonempty=True)
            if path is None:
                return _invalid_arguments("read_file", "path must be a nonempty string")
            return _command_observation(self._exec(f"cat -- {shlex.quote(path)}"))
        if action.name == "write_file":
            path = _string_argument(arguments, "path", nonempty=True)
            content = _string_argument(arguments, "content")
            if path is None or content is None:
                return _invalid_arguments(
                    "write_file", "path must be nonempty and content must be a string"
                )
            result = self._exec(
                _WRITE_COMMAND,
                env={
                    "WMO_FILE_PATH": path,
                    "WMO_FILE_CONTENT_B64": base64.b64encode(content.encode()).decode(),
                },
            )
            observation = _command_observation(result)
            if not observation.is_error:
                return Observation(
                    content=f"wrote {path}",
                    metadata=observation.metadata,
                )
            return observation
        return Observation(content=f"tool {action.name!r} not available", is_error=True)

    def _exec(self, command: str, *, env: dict[str, str] | None = None) -> ExecResult:
        future = asyncio.run_coroutine_threadsafe(
            self._environment.exec(
                command,
                env=env,
                timeout_sec=self._command_timeout_sec,
            ),
            self._event_loop,
        )
        return future.result()


class WmoHarborAgent(BaseAgent):
    """Run the serialized WMO candidate while Harbor owns tasks and verification."""

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        logger: logging.Logger | None = None,
        mcp_servers: list[MCPServerConfig] | None = None,
        skills_dir: str | None = None,
        *,
        command_timeout_sec: int = MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC,
        extra_env: dict[str, str] | None = None,
        harness: JsonObject,
        provider_config: JsonObject,
        harness_backend: Literal["local", "e2b"] = "local",
        e2b_template: str | None = None,
        episode_timeout_sec: float = DEFAULT_EVAL_EPISODE_TIMEOUT_S,
        episode_workers: int = DEFAULT_EPISODE_WORKERS,
        context_window: int | None = None,
    ) -> None:
        if extra_env:
            raise ValueError("WMO Harbor evaluation does not inject agent environment variables")
        super().__init__(
            logs_dir=logs_dir,
            model_name=model_name,
            logger=logger,
            mcp_servers=mcp_servers,
            skills_dir=skills_dir,
            extra_env=extra_env,
        )
        if harness_backend not in ("local", "e2b"):
            raise ValueError("harness_backend must be local or e2b")
        if harness_backend == "local" and e2b_template is not None:
            raise ValueError("e2b_template requires harness_backend='e2b'")
        try:
            self._episode_timeout_sec = validate_episode_timeout_s(episode_timeout_sec)
        except ValueError as error:
            raise ValueError("episode_timeout_sec must be a finite positive number") from error
        if context_window is not None and (
            isinstance(context_window, bool) or not isinstance(context_window, int)
        ):
            raise ValueError("context_window must be an integer number of tokens")
        if context_window is not None and context_window < 1024:
            raise ValueError("context_window must be at least 1024 tokens")
        if isinstance(episode_workers, bool) or not isinstance(episode_workers, int):
            raise ValueError("episode_workers must be a positive integer")
        if episode_workers < 1:
            raise ValueError("episode_workers must be a positive integer")
        self._harness = HarnessDoc.model_validate(harness)
        self._provider_config = ProviderConfig.model_validate(provider_config)
        expected_model_name = f"{self._provider_config.kind.value}/{self._provider_config.model}"
        if model_name != expected_model_name:
            raise ValueError(
                f"Harbor model identity must be {expected_model_name!r}, got {model_name!r}"
            )
        self._provider = self._build_provider(self._provider_config)
        self._command_timeout_sec = _validate_command_timeout_sec(command_timeout_sec)
        self._harness_backend = harness_backend
        self._e2b_template = e2b_template
        self._episode_workers = episode_workers
        self._context_window = context_window

    def _build_provider(self, config: ProviderConfig) -> Provider:
        """Construct the worker provider; the one seam subclasses may override.

        Called exactly once, from ``__init__``, after ``BaseAgent`` has set
        ``self.logs_dir`` and the validated provider config and model identity are in
        place, so an override can key per-trial state (e.g. a token sink named after
        the harbor trial) off the logs directory.

        Args:
            config: The validated worker provider config.

        Returns:
            The provider the episode runtime will drive. Overrides must keep the
            retry contract by wrapping their provider with
            ``wrap_provider_with_retries``.
        """
        # Retry-wrap the worker provider: Bedrock disables botocore's own retries, so one
        # unwrapped ThrottlingException would otherwise kill a whole trial.
        return wrap_provider_with_retries(get_provider(config))

    def _build_provider(self, config: ProviderConfig) -> Provider:
        """Construct the worker provider; the one seam subclasses may override.

        Called exactly once, from ``__init__``, after ``BaseAgent`` has set
        ``self.logs_dir`` and the validated provider config and model identity are in
        place, so an override can key per-trial state (e.g. a token sink named after
        the harbor trial) off the logs directory.

        Args:
            config: The validated worker provider config.

        Returns:
            The provider the episode runtime will drive. Overrides must keep the
            retry contract by wrapping their provider with
            ``wrap_provider_with_retries``.
        """
        # Retry-wrap the worker provider: Bedrock disables botocore's own retries, so one
        # unwrapped ThrottlingException would otherwise kill a whole trial.
        return wrap_provider_with_retries(get_provider(config))

    @staticmethod
    def name() -> str:
        return "wmo-harness"

    def version(self) -> str:
        return WMO_HARBOR_AGENT_VERSION

    async def setup(self, environment: BaseEnvironment) -> None:
        """Use Harbor's already-started task environment without installing another agent."""
        del environment

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Run the candidate in a dedicated worker thread and always persist its WMO trace."""
        context.metadata = {"candidate_doc_hash": self._harness.doc_hash}
        cancel_requested = threading.Event()
        # Cancellation is cooperative and best-effort on the local backend: the local SSH
        # pi-node runtime has no should_cancel hook, so a cancelled local episode runs to its
        # own node/SSH bound before the shield below releases. The e2b backend honors it.
        runtime = self._harness.runtime(
            self._provider,
            backend=self._harness_backend,
            e2b_template=self._e2b_template,
            # The wall budget applies to every backend: the local SSH transport used to hardcode
            # `timeout 300 node`, so a configured budget was silently ignored there.
            episode_timeout_s=self._episode_timeout_sec,
            context_window=self._context_window,
            # A real task environment is mutable, so an E2B transport failure must not replay
            # the whole episode against already-mutated state. Local Pi has no replay wrapper.
            transport_retries=0 if self._harness_backend == "e2b" else None,
            should_cancel=cancel_requested.is_set,
        )
        bridge = HarborAgentEnvironment(
            asyncio.get_running_loop(),
            environment,
            command_timeout_sec=self._command_timeout_sec,
        )
        task_id = str(self.context_id or self.session_id or "harbor-task")
        loop = asyncio.get_running_loop()
        executor = _episode_executor(self._episode_workers)
        run_task = asyncio.ensure_future(
            loop.run_in_executor(executor, lambda: runtime.run(task_id, instruction, bridge))
        )
        result: RunResult | None = None
        try:
            # Harbor enforces its agent timeout by cancelling this coroutine. Shield the
            # worker so cancellation cannot detach a still-running harness from the task
            # environment that Harbor is about to verify.
            result = await asyncio.shield(run_task)
        except asyncio.CancelledError:
            cancel_requested.set()
            abort = getattr(runtime, "abort", None)
            try:
                if callable(abort):
                    await self._cleanup_uncancellable(abort, executor, what="abort")
            finally:
                await _wait_for_quiescence(run_task)
            raise
        finally:
            try:
                close = getattr(runtime, "close", None)
                if callable(close):
                    await self._cleanup_uncancellable(close, executor, what="close")
            finally:
                bridge.close()
                # The trace write lives inside this inner finally: a harbor-timeout cancellation
                # (the most informative failure class) must still leave the partial transcript
                # for the proposer, even when cleanup itself was re-cancelled above. Synchronous
                # small-file I/O, so cancellation cannot interrupt the write itself.
                self._write_trace(
                    task_id,
                    instruction,
                    run_task,
                    bridge,
                    cancelled=cancel_requested.is_set(),
                )
        _populate_context(context, result)

    async def _cleanup_uncancellable(
        self,
        call: Callable[[], object],
        executor: ThreadPoolExecutor,
        *,
        what: str,
    ) -> None:
        """Run one cleanup step to completion; its failures never replace the episode outcome.

        A pool-close/abort error (e.g. SandboxCleanupError) after a finished episode would
        otherwise abort the trial pre-verification and discard a real result. Cancellation
        semantics are preserved: a re-cancellation observed during cleanup still re-raises.
        """
        try:
            await _run_uncancellable(call, executor)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - cleanup is best-effort; the run outcome wins
            self.logger.warning("harbor agent %s cleanup failed; continuing", what, exc_info=True)

    def _write_trace(
        self,
        task_id: str,
        instruction: str,
        run_task: asyncio.Future[RunResult],
        bridge: HarborAgentEnvironment,
        *,
        cancelled: bool,
    ) -> None:
        """Persist the full RunResult when one exists, else the partial episode evidence."""
        payload = _trace_payload(
            task_id,
            instruction,
            run_task,
            bridge,
            cancelled=cancelled,
        )
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            (self.logs_dir / _TRACE_FILENAME).write_text(payload, encoding="utf-8")
        except OSError:
            if not cancelled and not run_task.cancelled() and run_task.exception() is None:
                raise  # a healthy trial must not silently lose its transcript
            self.logger.warning("failed to persist the partial WMO trace", exc_info=True)


def _trace_payload(
    task_id: str,
    instruction: str,
    run_task: asyncio.Future[RunResult],
    bridge: HarborAgentEnvironment,
    *,
    cancelled: bool,
) -> str:
    if run_task.done() and not run_task.cancelled() and run_task.exception() is None:
        return (
            run_task.result()
            .model_copy(update={"instruction": instruction})
            .model_dump_json(indent=2)
        )
    error = None if not run_task.done() or run_task.cancelled() else run_task.exception()
    stop_reason = (
        _CANCELLED_STOP_REASON
        if cancelled
        else f"agent-exception:{type(error).__name__}"
        if error is not None
        else _CANCELLED_STOP_REASON
    )
    steps = bridge.recorded_steps()
    partial: JsonObject = {
        "task_id": task_id,
        "instruction": instruction,
        "steps": steps,
        "stop_reason": stop_reason,
        "answer": "",
        "turns": len(steps),
        "partial": True,
    }
    if error is not None:
        partial["error"] = f"{type(error).__name__}: {error}"
    usage = getattr(error, "worker_usage", None)
    if usage is not None:
        partial["worker_usage"] = usage.model_dump(mode="json")
    return json.dumps(partial, indent=2, ensure_ascii=False)


async def _wait_for_quiescence(run_task: asyncio.Future[RunResult]) -> None:
    """Drain a shielded runtime even if the owning Harbor task is cancelled again."""
    await _wait_until_done(run_task)
    if not run_task.cancelled():
        run_task.exception()


async def _run_uncancellable[T](
    call: Callable[[], T],
    executor: ThreadPoolExecutor,
) -> T:
    """Run blocking cleanup to completion despite repeated coroutine cancellation."""
    loop = asyncio.get_running_loop()
    cleanup_task = asyncio.ensure_future(loop.run_in_executor(executor, call))
    cancelled = await _wait_until_done(cleanup_task)
    if cancelled:
        # Re-deliver the cancellation; a cleanup failure is secondary (consume it so the
        # event loop never logs a never-retrieved exception).
        if not cleanup_task.cancelled():
            cleanup_task.exception()
        raise asyncio.CancelledError
    return cleanup_task.result()


async def _wait_until_done[T](task: asyncio.Future[T]) -> bool:
    """Wait without propagating the child result or cancelling it with the waiter."""
    cancelled = False
    while not task.done():
        try:
            await asyncio.wait({task})
        except asyncio.CancelledError:
            cancelled = True
    return cancelled


def _populate_context(context: AgentContext, result: RunResult | None) -> None:
    if result is None:
        return
    usage = result.worker_usage
    if usage is not None:
        context.n_input_tokens = usage.input_tokens
        context.n_output_tokens = usage.output_tokens
    metadata = dict(context.metadata or {})
    metadata.update(
        {
            "stop_reason": result.stop_reason.value,
            "turns": result.turns,
        }
    )
    context.metadata = metadata


def _string_argument(
    arguments: Mapping[str, object],
    name: str,
    *,
    nonempty: bool = False,
) -> str | None:
    value = arguments.get(name)
    if not isinstance(value, str) or (nonempty and not value):
        return None
    return value


def _invalid_arguments(tool: str, message: str) -> Observation:
    return Observation(content=f"invalid {tool} arguments: {message}", is_error=True)


# A real task environment can emit observations no model context can use (a rendered 52 MiB
# image via read_file, verified live). Two separate hazards, one cap:
#   - transport: an unbounded observation travels the whole worker transport as one frame and
#     kills the runner channel mid-episode,
#   - context: the cap must be small against the MODEL's window, not just the wire. The former
#     262,144 chars is roughly 75,000 tokens, so ONE observation could exceed a whole 65,536-token
#     serving window by itself; `gcode-to-text` died at step 1 in every attempt of every model
#     because its first natural move returns 262,227 chars.
# 10,000 is parity with the reference terminus-2 agent's own single-observation cap
# (`_limit_output_length(..., max_bytes=10000)`), which is the scaffold the published numbers were
# measured with. Truncation keeps the MIDDLE out and both ends in: head carries the format
# signature, tail carries any summary a command prints last, and the explicit marker tells the
# model to narrow its command rather than leaving it to infer a silent cut.
MAX_OBSERVATION_CHARS = 10_000


def _elision_marker(omitted: int) -> str:
    """The explicit middle-elision notice, which also tells the model what to do about it."""
    return (
        f"\n... [{omitted} characters truncated from the middle; command output exceeded "
        f"{MAX_OBSERVATION_CHARS} characters, so narrow the command or page through the "
        "output] ...\n"
    )


def _bounded_observation_text(content: str) -> str:
    """One observation clipped to MAX_OBSERVATION_CHARS, eliding the middle with a marker.

    The marker itself is inside the budget: its length is reserved using `len(content)` as an upper
    bound on the omitted count, so its digit count can never grow past the reservation and push the
    result over the cap.
    """
    if len(content) <= MAX_OBSERVATION_CHARS:
        return content
    keep = max(MAX_OBSERVATION_CHARS - len(_elision_marker(len(content))), 2)
    head = keep - keep // 2
    tail = keep // 2
    return content[:head] + _elision_marker(len(content) - head - tail) + content[-tail:]


def _command_observation(result: ExecResult) -> Observation:
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    content = _bounded_observation_text(stdout + stderr)
    if result.return_code != 0:
        content += f"\n[exit {result.return_code}]"
    return Observation(
        content=content,
        is_error=result.return_code != 0,
        metadata={"return_code": result.return_code},
    )


def _validate_command_timeout_sec(value: int) -> int:
    """Validate the evaluator-owned finite task-command policy."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC
    ):
        raise ValueError(
            f"command_timeout_sec must be an integer in [1, {MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC}]"
        )
    return value
