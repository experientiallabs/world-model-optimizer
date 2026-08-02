"""Tests for the Harbor bridge that runs an exact WMO harness document."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

import pytest
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.agent.context import AgentContext

from wmo.core.types import Action, ActionKind
from wmo.evals.harbor.agent import (
    MAX_OBSERVATION_CHARS,
    HarborAgentEnvironment,
    WmoHarborAgent,
    _bounded_observation_text,
)
from wmo.harness.doc import HarnessDoc
from wmo.harness.runtime import RunResult, RuntimeCancelled, StopReason, TokenUsage
from wmo.providers.base import Provider, ProviderConfig, ProviderKind
from wmo.providers.retry import RetryingProvider


class _Environment:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str] | None, int | None]] = []
        self.responses: dict[str, str] = {}

    async def exec(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        **_kwargs: object,
    ) -> ExecResult:
        self.calls.append((command, env, timeout_sec))
        if command in self.responses:
            return ExecResult(stdout=self.responses[command], stderr="", return_code=0)
        if command == "sleep 999":
            raise TimeoutError("command exceeded 240s")
        if command == "false":
            return ExecResult(stdout="", stderr="failed\n", return_code=7)
        if command.startswith("cat --"):
            return ExecResult(stdout="contents\n", stderr="", return_code=0)
        return ExecResult(stdout="ok\n", stderr="", return_code=0)


class _FakeProvider:
    config = ProviderConfig(kind=ProviderKind.BEDROCK, model="worker-model", region="us-west-2")


def _provider_config() -> ProviderConfig:
    return ProviderConfig(kind=ProviderKind.BEDROCK, model="worker-model", region="us-west-2")


def _agent(
    tmp_path: Path,
    *,
    harness: HarnessDoc | None = None,
    model_name: str = "bedrock/worker-model",
    harness_backend: Literal["local", "e2b"] = "local",
    e2b_template: str | None = None,
    episode_timeout_sec: float = 300.0,
    episode_workers: int = 64,
    context_window: int | None = None,
    extra_env: dict[str, str] | None = None,
) -> WmoHarborAgent:
    return WmoHarborAgent(
        logs_dir=tmp_path,
        model_name=model_name,
        harness=(harness or HarnessDoc.baseline()).model_dump(mode="json"),
        provider_config=_provider_config().model_dump(mode="json"),
        harness_backend=harness_backend,
        e2b_template=e2b_template,
        episode_timeout_sec=episode_timeout_sec,
        episode_workers=episode_workers,
        context_window=context_window,
        extra_env=extra_env,
    )


def test_harbor_environment_routes_supported_tools_and_records_steps() -> None:
    async def run() -> None:
        environment = _Environment()
        bridge = HarborAgentEnvironment(
            asyncio.get_running_loop(), cast("BaseEnvironment", environment)
        )

        bash = await asyncio.to_thread(
            bridge.execute,
            Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": "false"}),
        )
        write = await asyncio.to_thread(
            bridge.execute,
            Action(
                kind=ActionKind.TOOL_CALL,
                name="write_file",
                arguments={"path": "out/data.txt", "content": "hello"},
            ),
        )
        unknown = await asyncio.to_thread(
            bridge.execute,
            Action(kind=ActionKind.TOOL_CALL, name="rm_rf", arguments={}),
        )

        assert bash.is_error is True
        assert bash.metadata["return_code"] == 7
        assert "failed" in bash.content
        assert write.content == "wrote out/data.txt"
        assert unknown.is_error is True
        assert environment.calls[1][1] is not None
        assert environment.calls[1][1]["WMO_FILE_CONTENT_B64"] == "aGVsbG8="
        assert {call[2] for call in environment.calls} == {240}
        steps = [cast("dict[str, dict[str, object]]", step) for step in bridge.recorded_steps()]
        assert [step["action"]["name"] for step in steps] == ["bash", "write_file", "rm_rf"]
        assert steps[0]["observation"]["is_error"] is True

    asyncio.run(run())


def test_agent_runs_the_exact_candidate_on_the_dedicated_executor_and_persists_its_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = HarnessDoc.baseline("candidate")
    environment = _Environment()
    provider = _FakeProvider()
    observed: dict[str, object] = {}

    class _Runtime:
        def run(
            self,
            task_id: str,
            instruction: str,
            task_environment: HarborAgentEnvironment,
        ) -> RunResult:
            observed["thread"] = threading.current_thread().name
            observed["instruction"] = instruction
            observation = task_environment.execute(
                Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": "pwd"})
            )
            assert observation.content == "ok\n"
            return RunResult(
                task_id=task_id,
                stop_reason=StopReason.SUBMITTED,
                answer="done",
                turns=3,
                worker_usage=TokenUsage(input_tokens=11, output_tokens=7, calls=2),
            )

        def close(self) -> None:
            observed["closed"] = True

    def runtime(
        self: HarnessDoc,
        actual_provider: object,
        *,
        backend: str = "local",
        e2b_template: str | None = None,
        episode_timeout_s: float | None = None,
        transport_retries: int | None = None,
        **_kwargs: object,
    ) -> _Runtime:
        observed["candidate_hash"] = self.doc_hash
        observed["provider"] = actual_provider
        observed["backend"] = backend
        observed["template"] = e2b_template
        observed["episode_timeout_s"] = episode_timeout_s
        observed["transport_retries"] = transport_retries
        return _Runtime()

    monkeypatch.setattr("wmo.evals.harbor.agent.get_provider", lambda _config: provider)
    monkeypatch.setattr(HarnessDoc, "runtime", runtime)
    agent = _agent(
        tmp_path,
        harness=candidate,
        harness_backend="e2b",
        e2b_template="runner-template",
        episode_timeout_sec=12_000,
    )
    context = AgentContext()

    asyncio.run(agent.run("solve it", cast("BaseEnvironment", environment), context))

    assert observed["candidate_hash"] == candidate.doc_hash
    assert observed["backend"] == "e2b"
    assert observed["template"] == "runner-template"
    assert observed["episode_timeout_s"] == 12_000
    # Real environments are side-effectful: whole-episode transport replay must be off.
    assert observed["transport_retries"] == 0
    assert observed["closed"] is True
    # The episode ran on the dedicated executor, never asyncio.to_thread's default pool.
    assert str(observed["thread"]).startswith("wmo-harbor-episode")
    # The worker provider is retry-wrapped (Bedrock disables botocore retries; one raw
    # ThrottlingException must not kill a trial).
    wrapped = observed["provider"]
    assert isinstance(wrapped, RetryingProvider)
    assert wrapped.config is provider.config
    assert context.n_input_tokens == 11
    assert context.metadata == {
        "candidate_doc_hash": candidate.doc_hash,
        "stop_reason": "submitted",
        "turns": 3,
    }
    trace = json.loads((tmp_path / "wmo-run.json").read_text())
    assert trace["instruction"] == "solve it"
    assert trace["answer"] == "done"
    assert trace["stop_reason"] == "submitted"


def test_environment_exec_failures_become_error_observations_not_episode_deaths() -> None:
    """A timed-out or transport-dead task command is a candidate outcome the agent sees as an
    error observation; escaping as an exception would skip verification and make the whole
    candidate unscoreable (and, with the pruner, re-run forever)."""

    async def run() -> None:
        environment = _Environment()
        bridge = HarborAgentEnvironment(
            asyncio.get_running_loop(), cast("BaseEnvironment", environment)
        )
        observation = await asyncio.to_thread(
            bridge.execute,
            Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": "sleep 999"}),
        )
        assert observation.is_error is True
        assert "environment command failed: TimeoutError" in observation.content
        # The failed step still reaches the transcript the proposer reads.
        [step] = [cast("dict[str, dict[str, object]]", s) for s in bridge.recorded_steps()]
        assert step["action"]["arguments"] == {"command": "sleep 999"}
        assert step["observation"]["is_error"] is True

    asyncio.run(run())


def test_oversized_command_output_is_truncated_before_it_reaches_the_channel() -> None:
    """A real environment can emit observations no model can use (a 52 MiB rendered image via
    read_file, observed live); unbounded, one such frame kills the worker transport
    mid-episode. The bridge bounds every observation with an explicit head+tail marker."""

    async def run() -> None:
        environment = _Environment()
        oversized = "x" * (MAX_OBSERVATION_CHARS + 10_000)
        environment.responses["cat -- /app/image.ppm"] = oversized
        bridge = HarborAgentEnvironment(
            asyncio.get_running_loop(), cast("BaseEnvironment", environment)
        )
        observation = await asyncio.to_thread(
            bridge.execute,
            Action(
                kind=ActionKind.TOOL_CALL, name="read_file", arguments={"path": "/app/image.ppm"}
            ),
        )
        assert observation.is_error is False
        assert len(observation.content) <= MAX_OBSERVATION_CHARS
        assert "characters truncated" in observation.content
        assert observation.content.startswith("x" * 100)
        assert observation.content.endswith("x" * 100)

    asyncio.run(run())


def test_the_observation_cap_fits_inside_a_model_context_window() -> None:
    """The cap protects the MODEL's context, not just the wire.

    At the old 262,144 chars (roughly 75,000 tokens) ONE observation could exceed a whole
    65,536-token serving window by itself, and `gcode-to-text` died at step 1 in every attempt of
    every model because its first natural move returns 262,227 chars. 10,000 is parity with the
    reference terminus-2 agent's own single-observation cap.
    """
    assert MAX_OBSERVATION_CHARS == 10_000
    # Roughly 4 chars per token: one capped observation must stay a small slice of the window.
    assert MAX_OBSERVATION_CHARS / 4 < 0.1 * 65_536


def test_a_bounded_observation_keeps_both_ends_and_elides_the_middle() -> None:
    """Head carries the format signature, tail carries any trailing summary."""
    content = "HEAD" + ("m" * (MAX_OBSERVATION_CHARS * 3)) + "TAIL"
    bounded = _bounded_observation_text(content)

    assert len(bounded) <= MAX_OBSERVATION_CHARS
    assert bounded.startswith("HEAD")
    assert bounded.endswith("TAIL")
    assert "truncated from the middle" in bounded
    # Only the middle is gone; nothing is rewritten or reordered.
    assert "m" * 100 in bounded


def test_a_short_observation_is_returned_untouched() -> None:
    assert _bounded_observation_text("small output") == "small output"
    exact = "y" * MAX_OBSERVATION_CHARS
    assert _bounded_observation_text(exact) == exact


def test_cleanup_failure_never_masks_the_episode_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A SandboxCleanupError-class failure from close() must not abort a verified-able trial."""

    class _Runtime:
        def run(
            self,
            task_id: str,
            _instruction: str,
            _environment: HarborAgentEnvironment,
        ) -> RunResult:
            return RunResult(task_id=task_id, stop_reason=StopReason.SUBMITTED, answer="done")

        def close(self) -> None:
            raise RuntimeError("failed to prove cleanup for 1 sandbox")

    monkeypatch.setattr("wmo.evals.harbor.agent.get_provider", lambda _config: _FakeProvider())
    monkeypatch.setattr(HarnessDoc, "runtime", lambda *_args, **_kwargs: _Runtime())
    agent = _agent(tmp_path)
    context = AgentContext()

    asyncio.run(agent.run("solve it", cast("BaseEnvironment", _Environment()), context))

    assert context.metadata is not None
    assert context.metadata["stop_reason"] == "submitted"
    trace = json.loads((tmp_path / "wmo-run.json").read_text())
    assert trace["answer"] == "done"  # the full result survived the failed cleanup


def test_harbor_timeout_cancellation_still_persists_the_partial_transcript(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The trace write lives in a finally: timeout trials must not vanish for the proposer."""
    started = threading.Event()
    events: list[str] = []

    class _Runtime:
        def __init__(self, should_cancel: Callable[[], bool]) -> None:
            self._should_cancel = should_cancel

        def run(
            self,
            _task_id: str,
            _instruction: str,
            task_environment: HarborAgentEnvironment,
        ) -> RunResult:
            task_environment.execute(
                Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": "pwd"})
            )
            started.set()
            while not self._should_cancel():
                time.sleep(0.001)
            raise RuntimeCancelled(
                worker_usage=TokenUsage(input_tokens=5, output_tokens=2, calls=1)
            )

        def abort(self) -> None:
            events.append("abort")

        def close(self) -> None:
            events.append("close")

    def runtime(
        _self: HarnessDoc,
        _provider: object,
        *,
        should_cancel: Callable[[], bool] | None = None,
        **_kwargs: object,
    ) -> _Runtime:
        assert should_cancel is not None
        return _Runtime(should_cancel)

    monkeypatch.setattr("wmo.evals.harbor.agent.get_provider", lambda _config: _FakeProvider())
    monkeypatch.setattr(HarnessDoc, "runtime", runtime)
    agent = _agent(tmp_path)

    async def drive() -> None:
        task = asyncio.create_task(
            agent.run("solve it", cast("BaseEnvironment", _Environment()), AgentContext())
        )
        while not started.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(drive())

    assert events == ["abort", "close"]
    trace = json.loads((tmp_path / "wmo-run.json").read_text())
    assert trace["partial"] is True
    assert trace["stop_reason"] == "cancelled-by-harbor-timeout"
    assert [step["action"]["name"] for step in trace["steps"]] == ["bash"]
    assert trace["worker_usage"] == {
        "input_tokens": 5,
        "output_tokens": 2,
        "cached_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "reasoning_tokens": 0,
        "calls": 1,
        "call_seconds": [],
        "call_input_tokens": [],
        "call_output_tokens": [],
        "call_cached_input_tokens": [],
        "call_cache_write_input_tokens": [],
    }
    assert "RuntimeCancelled" in trace["error"]


def test_agent_exception_persists_a_partial_transcript_with_the_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Runtime:
        def run(
            self,
            _task_id: str,
            _instruction: str,
            _environment: HarborAgentEnvironment,
        ) -> RunResult:
            raise RuntimeError("sandbox died")

        def close(self) -> None:
            return None

    monkeypatch.setattr("wmo.evals.harbor.agent.get_provider", lambda _config: _FakeProvider())
    monkeypatch.setattr(HarnessDoc, "runtime", lambda *_args, **_kwargs: _Runtime())
    agent = _agent(tmp_path)

    with pytest.raises(RuntimeError, match="sandbox died"):
        asyncio.run(agent.run("solve it", cast("BaseEnvironment", _Environment()), AgentContext()))

    trace = json.loads((tmp_path / "wmo-run.json").read_text())
    assert trace["partial"] is True
    assert trace["stop_reason"] == "agent-exception:RuntimeError"
    assert trace["error"] == "RuntimeError: sandbox died"


def test_build_provider_hook_runs_with_logs_dir_set_and_feeds_the_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The provider-construction seam: an override sees the validated config after
    BaseAgent has set logs_dir, and its provider is the exact one the runtime drives
    (the distill agent keys its per-trial token sink off both)."""
    observed: dict[str, object] = {}
    hook_provider = _FakeProvider()

    class _HookAgent(WmoHarborAgent):
        def _build_provider(self, config: ProviderConfig) -> Provider:
            observed["logs_dir"] = self.logs_dir
            observed["config"] = config
            return cast("Provider", hook_provider)

    class _Runtime:
        def run(
            self,
            task_id: str,
            _instruction: str,
            _environment: HarborAgentEnvironment,
        ) -> RunResult:
            return RunResult(task_id=task_id, stop_reason=StopReason.SUBMITTED, answer="done")

        def close(self) -> None:
            return None

    def runtime(_self: HarnessDoc, provider: object, **_kwargs: object) -> _Runtime:
        observed["runtime_provider"] = provider
        return _Runtime()

    monkeypatch.setattr(HarnessDoc, "runtime", runtime)
    agent = _HookAgent(
        logs_dir=tmp_path,
        model_name="bedrock/worker-model",
        harness=HarnessDoc.baseline().model_dump(mode="json"),
        provider_config=_provider_config().model_dump(mode="json"),
    )

    assert observed["logs_dir"] == tmp_path
    assert observed["config"] == _provider_config()

    asyncio.run(agent.run("solve it", cast("BaseEnvironment", _Environment()), AgentContext()))
    assert observed["runtime_provider"] is hook_provider


def test_agent_rejects_invalid_construction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="environment variables"):
        _agent(tmp_path, extra_env={"TOKEN": "secret"})
    with pytest.raises(ValueError, match="model identity"):
        _agent(tmp_path, model_name="openai/other-model")
    with pytest.raises(ValueError, match="requires harness_backend='e2b'"):
        _agent(tmp_path, e2b_template="tmpl")
    # episode_timeout_sec is no longer e2b-only: the local SSH transport applies it as the remote
    # node timeout instead of the fixed 300s it used to hardcode. Only nonsense values are rejected.
    with pytest.raises(ValueError, match="episode_timeout_sec"):
        _agent(tmp_path, episode_timeout_sec=0)
    with pytest.raises(ValueError, match="context_window"):
        _agent(tmp_path, context_window=16)
    with pytest.raises(ValueError, match="episode_workers"):
        _agent(tmp_path, episode_workers=0)
