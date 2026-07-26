"""Tests for the token-recording distill harbor agent (no real tinker SDK)."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path
from typing import cast

import pytest
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.agent.context import AgentContext

from wmh.distill.agents import (
    WMH_DISTILL_HARBOR_AGENT_IMPORT_PATH,
    WmhDistillHarborAgent,
)
from wmh.harness.doc import HarnessDoc
from wmh.harness.runtime import RunResult, StopReason
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.providers.openai import OpenAIProvider
from wmh.providers.retry import RetryingToolCallingProvider
from wmh.providers.tinker import TinkerChatProvider, TokenSpan


def _tinker_config() -> ProviderConfig:
    return ProviderConfig(
        kind=ProviderKind.TINKER,
        model_type="Qwen/Qwen3-8B",
        model="tinker://run/weights/0",
    )


def _logs_dir(tmp_path: Path, trial_name: str = "task-a__x1Y2z3") -> Path:
    # Harbor's single-step layout: {job_dir}/{trial_name}/agent is the agent logs dir.
    return tmp_path / "jobs" / "wmh-abc" / trial_name / "agent"


def _agent(
    tmp_path: Path,
    *,
    provider_config: ProviderConfig | None = None,
    token_sink_dir: str | None = None,
    trial_name: str = "task-a__x1Y2z3",
) -> WmhDistillHarborAgent:
    config = provider_config or _tinker_config()
    return WmhDistillHarborAgent(
        logs_dir=_logs_dir(tmp_path, trial_name),
        model_name=f"{config.kind.value}/{config.model}",
        harness=HarnessDoc.baseline().model_dump(mode="json"),
        provider_config=config.model_dump(mode="json"),
        token_sink_dir=(
            token_sink_dir if token_sink_dir is not None else str(tmp_path / "tokens" / "step-0000")
        ),
    )


class _Environment:
    async def exec(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        **_kwargs: object,
    ) -> ExecResult:
        del command, env, timeout_sec
        return ExecResult(stdout="ok\n", stderr="", return_code=0)


def test_import_path_constant_resolves_to_the_agent_class() -> None:
    module_name, _, attribute = WMH_DISTILL_HARBOR_AGENT_IMPORT_PATH.partition(":")
    module = importlib.import_module(module_name)
    assert getattr(module, attribute) is WmhDistillHarborAgent


def test_sink_filename_derives_from_the_harbor_trial_name(tmp_path: Path) -> None:
    agent = _agent(tmp_path, trial_name="task-b__q6Rn5Jd")

    expected = tmp_path / "tokens" / "step-0000" / "task-b__q6Rn5Jd.jsonl"
    assert agent.token_sink_path == expected
    assert expected.parent.is_dir()

    # The provider keeps the base retry contract and wraps the tinker provider,
    # whose recorder writes through to the derived sink path.
    wrapped = agent._provider
    assert isinstance(wrapped, RetryingToolCallingProvider)
    inner = wrapped._provider
    assert isinstance(inner, TinkerChatProvider)
    recorder = inner._recorder
    assert recorder is not None
    recorder.record(
        TokenSpan(
            call_index=0,
            prompt_token_ids=[1, 2],
            sampled_token_ids=[65],
            sampled_logprobs=[-0.25],
        )
    )
    [line] = expected.read_text(encoding="utf-8").splitlines()
    assert json.loads(line)["sampled_token_ids"] == [65]


def test_construction_never_imports_the_tinker_sdk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider construction stays lazy: the SDK loads only at first sample."""
    monkeypatch.setitem(sys.modules, "tinker", None)
    monkeypatch.setitem(sys.modules, "tinker_cookbook", None)
    agent = _agent(tmp_path)
    assert isinstance(agent._provider, RetryingToolCallingProvider)


def test_served_teacher_generates_without_token_capture(tmp_path: Path) -> None:
    """A generation-only trial (the gate's served teacher baseline) needs no sink.

    The served teacher cannot report the ids it sampled, so the trial gets the
    base bridge's plain retry-wrapped provider, records nothing, and is scored on
    its verifier reward alone.
    """
    config = ProviderConfig(
        kind=ProviderKind.OPENAI,
        model="accounts/fireworks/models/glm-5p2",
        endpoint="https://api.fireworks.ai/inference/v1",
    )
    sink_dir = tmp_path / "tokens" / "step-0000"

    agent = _agent(tmp_path, provider_config=config, token_sink_dir=str(sink_dir))

    assert agent.token_sink_path is None
    wrapped = agent._provider
    assert isinstance(wrapped, RetryingToolCallingProvider)
    inner = wrapped._provider
    assert isinstance(inner, OpenAIProvider)
    assert inner.config.endpoint == "https://api.fireworks.ai/inference/v1"
    # Nothing was prepared for a recorder that does not exist.
    assert not sink_dir.exists()


def test_empty_token_sink_dir_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="token_sink_dir must be a nonempty path string"):
        _agent(tmp_path, token_sink_dir="")


def test_stale_sink_from_a_pruned_attempt_is_removed(tmp_path: Path) -> None:
    """A pruned trial dir can re-run under the same name; its old sink must not
    prepend stale spans (load_trial_spans rejects call_index resets)."""
    sink_dir = tmp_path / "tokens" / "step-0000"
    sink_dir.mkdir(parents=True)
    stale = sink_dir / "task-a__x1Y2z3.jsonl"
    stale.write_text('{"stale": true}\n', encoding="utf-8")

    agent = _agent(tmp_path, token_sink_dir=str(sink_dir))

    assert agent.token_sink_path == stale
    assert not stale.exists()


def test_run_drives_the_runtime_with_the_recording_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The inherited run() plumbing feeds the subclass-built provider to the runtime."""
    observed: dict[str, object] = {}

    class _Runtime:
        def run(
            self,
            task_id: str,
            _instruction: str,
            _environment: object,
        ) -> RunResult:
            return RunResult(task_id=task_id, stop_reason=StopReason.SUBMITTED, answer="done")

        def close(self) -> None:
            observed["closed"] = True

    def runtime(_self: HarnessDoc, provider: object, **_kwargs: object) -> _Runtime:
        observed["provider"] = provider
        return _Runtime()

    monkeypatch.setattr(HarnessDoc, "runtime", runtime)
    agent = _agent(tmp_path)

    asyncio.run(agent.run("solve it", cast("BaseEnvironment", _Environment()), AgentContext()))

    assert observed["provider"] is agent._provider
    assert observed["closed"] is True
    trace = json.loads((_logs_dir(tmp_path) / "wmh-run.json").read_text(encoding="utf-8"))
    assert trace["stop_reason"] == "submitted"


def test_accepts_every_keyword_its_base_harbor_agent_accepts() -> None:
    """The distill agent must not silently drop a kwarg harbor forwards to its base class.

    Harbor's `AgentFactory` calls `agent_class(logs_dir=..., model_name=..., **kwargs)`, so a
    keyword the base `WmhHarborAgent` grew but this subclass never declared raises `TypeError`
    inside trial construction — every trial dies at `_init_agent`, and the batch reports that as
    infra failure rather than as a code defect. That is how `context_window` (added when the
    served window stopped being hardcoded) killed a whole 48-episode probe wave while the test
    suite stayed green.
    """
    import inspect

    from wmh.evals.harbor.agent import WmhHarborAgent

    def keywords(cls: type) -> set[str]:
        return {
            name
            for name, p in inspect.signature(cls.__init__).parameters.items()
            if p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD) and name != "self"
        }

    missing = keywords(WmhHarborAgent) - keywords(WmhDistillHarborAgent)
    assert not missing, (
        f"distill agent drops base-class keywords harbor forwards: {sorted(missing)}"
    )
