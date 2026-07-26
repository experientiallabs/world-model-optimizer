"""CLI tests for `wmh eval --mode closed-loop`: harness-backend plumbing, driven via CliRunner.

Scoring is faked at the `ClosedLoopEval` seam — these tests pin the WIRING: the `@<model>` label,
backend-appropriate concurrency defaults, the always-required world model, runtime construction
and teardown for the e2b backend, and flag validation. No sandbox (or model) is ever touched.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from wmh.cli import app
from wmh.config.settings import ModelRole, ModelsSettings, ProjectSettings, save_settings
from wmh.evals.closed_loop import ClosedLoopReport, TaskOutcome, WorldModelSource
from wmh.evals.gold import GoldJudge, GoldVerdict
from wmh.evals.tasks import TaskSpec
from wmh.harness.doc import RUNTIME_KIND_ID, TOOL_POLICY_ID, HarnessDoc, Surface, SurfaceKind
from wmh.harness.environment import AgentEnvironment
from wmh.harness.pi_e2b import E2BPiRuntime
from wmh.harness.runtime import AgentRuntime, RunResult, Runtime
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind

eval_cl_module = importlib.import_module("wmh.cli.eval_closed_loop")
model_roles_module = importlib.import_module("wmh.cli.model_roles")

runner = CliRunner()

_Progress = Callable[[str, int, GoldVerdict], None] | None


class _Provider:
    """A do-nothing provider: scoring is faked, so no LLM role is ever exercised."""

    def __init__(self, config: ProviderConfig | None = None) -> None:
        self.config = config or ProviderConfig(kind=ProviderKind.BEDROCK, model="m")

    def complete(self, system: str, messages: list[Message], **kw: object) -> Completion:
        raise NotImplementedError

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self) -> object:
        raise NotImplementedError


class _FakeStore:
    """Resolves any name to a fixed model dir, so the label/`load_world_model` wiring is pinned."""

    def __init__(self, root: str) -> None:
        self.root = root

    def resolve(self, name: str | None) -> Path:
        return Path("/models/wm-alpha")


def _pi_doc() -> HarnessDoc:
    return HarnessDoc(
        name="pi",
        surfaces=[
            Surface(id="prompt:core", kind=SurfaceKind.PROMPT, content="p"),
            Surface(id=TOOL_POLICY_ID, kind=SurfaceKind.TOOL_POLICY, content="bash\nsubmit"),
            Surface(id=RUNTIME_KIND_ID, kind=SurfaceKind.PARAM, content="pi-node"),
            Surface(
                id="code:src-agent-ts", kind=SurfaceKind.CODE, path="src/agent.ts", content="// a"
            ),
        ],
    )


def _tasks_file(tmp_path: Path) -> str:
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        '{"task_id": "t1", "instruction": "do it", "gold": ["done"]}\n', encoding="utf-8"
    )
    return str(path)


def _report(label: str, k: int) -> ClosedLoopReport:
    outcome = TaskOutcome(task_id="t1", success_rate=1.0, mean_fraction=1.0, passes=k)
    return ClosedLoopReport(
        label=label, success_rate=1.0, mean_fraction=1.0, k=k, per_task={"t1": outcome}
    )


def _invoke(tmp_path: Path, *extra: str) -> Result:
    return runner.invoke(
        app,
        [
            "eval",
            _tasks_file(tmp_path),
            "--mode",
            "closed-loop",
            "--root",
            str(tmp_path / ".wmh"),
            *extra,
        ],
    )


def _patch_seams(monkeypatch: pytest.MonkeyPatch, seen: dict[str, object]) -> object:
    """Fake the world-model store/loader and `ClosedLoopEval`, recording the eval's wiring."""
    wm = object()
    world_provider = _Provider()

    class _FakeEval:
        def __init__(
            self,
            tasks: list[TaskSpec],
            environment: WorldModelSource,
            agent_provider: object,
            judge: GoldJudge,
            *,
            label: str,
            k: int,
            concurrency: int,
            runtime: Runtime | None,
            on_progress: _Progress,
        ) -> None:
            seen.update(
                {
                    "world_model": environment.world_model,
                    "agent_provider": agent_provider,
                    "label": label,
                    "concurrency": concurrency,
                    "runtime": runtime,
                }
            )
            self._label = label
            self._k = k

        def run(self) -> ClosedLoopReport:
            return _report(self._label, self._k)

    monkeypatch.setattr(eval_cl_module, "WorldModelStore", _FakeStore)
    monkeypatch.setattr(eval_cl_module, "load_world_model", lambda d: (wm, world_provider))
    monkeypatch.setattr(eval_cl_module, "ClosedLoopEval", _FakeEval)
    seen["world_provider"] = world_provider
    return wm


def test_eval_local_default_scores_the_world_model_sequentially(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}
    wm = _patch_seams(monkeypatch, seen)

    result = _invoke(tmp_path)  # --harness-backend defaults to local

    assert result.exit_code == 0, result.output
    assert seen["world_model"] is wm
    assert seen["agent_provider"] is seen["world_provider"]
    runtime = seen["runtime"]
    assert isinstance(runtime, AgentRuntime)
    assert runtime._provider is seen["world_provider"]
    assert seen["label"] == "baseline@wm-alpha"  # the label always names the model
    assert seen["concurrency"] == 1  # local default: the sequential loop, unchanged
    flat = " ".join(result.output.split())
    assert "world model" in flat and "wm-alpha" in flat
    assert "OVERALL" in result.output


def test_eval_agent_role_from_settings_drives_the_agent_under_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`[models.agent]` must survive the create-to-eval reproduction path."""
    seen: dict[str, object] = {}
    _patch_seams(monkeypatch, seen)
    save_settings(
        ProjectSettings(
            models=ModelsSettings(
                agent=ModelRole(
                    provider="openai",
                    model="Qwen/Qwen3.5-9B",
                    endpoint="http://127.0.0.1:8002/v1",
                )
            )
        ),
        tmp_path / ".wmh",
    )
    agent_sentinel = _Provider(
        ProviderConfig(
            kind=ProviderKind.OPENAI,
            model="Qwen/Qwen3.5-9B",
            endpoint="http://127.0.0.1:8002/v1",
        )
    )

    def fake_get_provider(_config: ProviderConfig) -> _Provider:
        return agent_sentinel

    monkeypatch.setattr(model_roles_module, "get_provider", fake_get_provider)

    result = _invoke(tmp_path)

    assert result.exit_code == 0, result.output
    assert seen["agent_provider"] is agent_sentinel
    runtime = seen["runtime"]
    assert isinstance(runtime, AgentRuntime)
    assert runtime._provider is agent_sentinel
    assert seen["label"] == "baseline[agent=openai:Qwen/Qwen3.5-9B]@wm-alpha"
    flat = " ".join(result.output.split())
    assert "agent model openai:Qwen/Qwen3.5-9B" in flat


def test_eval_always_requires_a_world_model(tmp_path: Path) -> None:
    # No model built under the tmp root: every backend must fail as a usage error.
    result = _invoke(tmp_path)
    assert result.exit_code == 2
    assert not isinstance(result.exception, (FileNotFoundError, ValueError))


def test_eval_rejects_unknown_harness_backend(tmp_path: Path) -> None:
    result = _invoke(tmp_path, "--harness-backend", "banana")
    assert result.exit_code == 2  # usage error, not a traceback
    assert "choose local or e2b" in result.output


def test_eval_e2b_without_a_harness_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}
    _patch_seams(monkeypatch, seen)
    result = _invoke(tmp_path, "--harness-backend", "e2b")
    assert result.exit_code == 2  # the baseline loop has no harness process to move
    assert "pass --harness" in result.output
    assert "label" not in seen  # rejected before any eval ran


def test_eval_e2b_with_a_non_pi_harness_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}
    _patch_seams(monkeypatch, seen)
    monkeypatch.setattr(
        eval_cl_module, "_load_harness", lambda name, root: HarnessDoc.baseline("plain")
    )
    result = _invoke(tmp_path, "--harness", "plain", "--harness-backend", "e2b")
    assert result.exit_code == 2  # doc.runtime's ValueError surfaces as a usage error
    assert "use backend='local'" in " ".join(result.output.split())
    assert "label" not in seen


def test_eval_e2b_runs_the_pi_harness_in_parallel_and_closes_its_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}
    wm = _patch_seams(monkeypatch, seen)

    class _FakePiRuntime(E2BPiRuntime):
        """isinstance-narrows as the real thing; skips the pool machinery entirely."""

        def __init__(self) -> None:  # deliberately NOT calling super: no pool, no sandbox
            self.closes = 0

        def run(self, task_id: str, instruction: str, environment: AgentEnvironment) -> RunResult:
            raise NotImplementedError  # scoring is faked; no episode ever runs

        def close(self) -> None:
            self.closes += 1

    fake_runtime = _FakePiRuntime()

    class _FakePiDoc:
        """The slice of HarnessDoc the CLI touches, with a recording runtime factory."""

        name = "pi"
        version = 3

        def runtime_kind(self) -> str:
            return "pi-node"

        def max_turns(self) -> int:
            return 20

        def runtime(
            self,
            provider: object,
            *,
            backend: str = "local",
            e2b_template: str | None = None,
        ) -> _FakePiRuntime:
            seen.update({"backend": backend, "e2b_template": e2b_template})
            return fake_runtime

    monkeypatch.setattr(eval_cl_module, "_load_harness", lambda name, root: _FakePiDoc())

    result = _invoke(
        tmp_path, "--harness", "pi", "--harness-backend", "e2b", "--e2b-template", "tmpl-x"
    )

    assert result.exit_code == 0, result.output
    assert seen["world_model"] is wm  # the env stays the world model on the e2b backend
    assert seen["label"] == "pi-v3@wm-alpha"  # the label still names the model, never "e2b"
    assert seen["concurrency"] == 0  # e2b default: every (task, attempt) cell at once
    assert seen["backend"] == "e2b"
    assert seen["e2b_template"] == "tmpl-x"
    assert seen["runtime"] is fake_runtime
    assert fake_runtime.closes == 1  # the eval tears down the runtime's private sandbox pool
    flat = " ".join(result.output.split())
    assert "E2B sandboxes" in flat and "wm-alpha" in flat


def test_eval_local_rejects_parallel_pi_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local pi runtimes are single-episode resources: local + --eval-concurrency>1 must not run."""
    monkeypatch.setattr(eval_cl_module, "WorldModelStore", _FakeStore)
    monkeypatch.setattr(eval_cl_module, "load_world_model", lambda d: (object(), _Provider()))
    monkeypatch.setattr(eval_cl_module, "_load_harness", lambda name, root: _pi_doc())

    result = _invoke(tmp_path, "--harness", "pi", "--eval-concurrency", "2")

    assert result.exit_code == 2  # usage error, before any rollout
    assert "one episode at a time" in result.output
