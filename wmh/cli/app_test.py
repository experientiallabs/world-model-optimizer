"""Tests for the CLI: command surface + build/list/play driven via CliRunner (fake provider)."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
from pathlib import Path
from typing import cast

import pytest
import typer
from typer.testing import CliRunner

from wmh.cli import app
from wmh.cli.app import _CONCURRENCY_ISOLATION_FLAGS
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind, verify_via_ping

# `wmh.cli`'s `app` attribute (the Typer object) shadows the `wmh.cli.app` submodule on
# plain `import wmh.cli.app as ...`; go through importlib to monkeypatch module globals.
cli_app_module = importlib.import_module("wmh.cli.app")

runner = CliRunner()


class FakeProvider:
    """Canned world-model JSON for rollouts/steps; a fixed prompt for GEPA reflection."""

    def __init__(self) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="opus")
        self.systems: list[str] = []  # system prompt of every complete() call, for assertions

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        self.systems.append(system)
        if "improve the system prompt" in system:
            return Completion(text="IMPROVED ENV PROMPT")
        if "grade a world model" in system:
            return Completion(
                text=(
                    '{"format": 0.5, "factuality": 0.5, "consistency": 0.5, '
                    '"realism": 0.5, "quality": 0.5, "critique": "be more specific"}'
                )
            )
        return Completion(text='{"output": "user u1 found", "is_error": false}')

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        # The pre-build verify guard pings through this; delegate to the shared ping so the fake
        # reports ok without hitting a real backend.
        return verify_via_ping(self)


def _traces_file(tmp_path) -> str:  # noqa: ANN001 - pytest fixture path
    span_llm = {
        "traceId": "a" * 32,
        "spanId": "s1",
        "name": "chat",
        "startTimeUnixNano": 1,
        "attributes": [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
            {"key": "gen_ai.tool.name", "value": {"stringValue": "get_user"}},
            {"key": "gen_ai.tool.call.arguments", "value": {"stringValue": '{"id": "u1"}'}},
            {"key": "gen_ai.prompt", "value": {"stringValue": "look up u1"}},
        ],
    }
    span_tool = {
        "traceId": "a" * 32,
        "spanId": "s2",
        "name": "execute_tool",
        "startTimeUnixNano": 2,
        "attributes": [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}},
            {"key": "gen_ai.tool.message", "value": {"stringValue": "found u1"}},
        ],
    }
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(span_llm) + "\n" + json.dumps(span_tool) + "\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def patched_provider(monkeypatch) -> None:  # noqa: ANN001 - pytest fixture
    """Swap the real provider registry for the fake everywhere the CLI constructs one.

    Each module binds `get_provider` at its own import time (build.py for the build pipeline,
    loader.py for serve/demo/play), so patch every module-level name plus the registry the lazy
    imports read.
    """
    import sys

    import wmh.providers as providers_pkg
    import wmh.providers.registry as registry
    import wmh.providers.waterfall as waterfall_mod

    fake = FakeProvider()
    # `wmh.engine.__init__` rebinds the name `build` to the function, shadowing the submodule
    # attribute, so reach module objects through sys.modules rather than attribute access.
    monkeypatch.setattr(sys.modules["wmh.engine.build"], "get_provider", lambda config: fake)
    # loader.py (serve/demo/play) and the CLI construct through the chain-aware seam.
    monkeypatch.setattr(
        sys.modules["wmh.engine.loader"], "provider_or_chain", lambda config, **kw: fake
    )
    monkeypatch.setattr(providers_pkg, "get_provider", lambda config: fake)
    monkeypatch.setattr(providers_pkg, "provider_or_chain", lambda config, **kw: fake)
    # The pre-build verify guard pings via verify_all/verify_embedder, which construct providers
    # through the registry's own get_provider — patch that too so the guard sees the fake, and
    # patch the name waterfall.py bound at import for its no-chain-file passthrough.
    monkeypatch.setattr(registry, "get_provider", lambda config: fake)
    monkeypatch.setattr(waterfall_mod, "get_provider", lambda config: fake)


def _build(root, name: str, tmp_path) -> None:  # noqa: ANN001 - pytest fixture paths
    result = runner.invoke(
        app,
        [
            "build",
            "--name",
            name,
            "--file",
            _traces_file(tmp_path),
            "--root",
            str(root),
            "--provider",
            "bedrock",
            "--fidelity",
            "low",
        ],
    )
    assert result.exit_code == 0, result.output


def test_build_writes_model_card(patched_provider, tmp_path) -> None:  # noqa: ANN001
    from wmh.config.card import load_card

    root = tmp_path / ".wmh"
    _build(root, "tau2-airline", tmp_path)
    card = load_card(root / "models" / "tau2-airline")
    assert card is not None
    assert card.name == "tau2-airline"
    assert card.corpus.traces is not None and card.corpus.traces > 0
    assert card.corpus.steps > 0
    assert card.provider == "bedrock"
    assert card.built_at is not None


def test_build_survives_card_write_failure(patched_provider, monkeypatch, tmp_path) -> None:  # noqa: ANN001
    # The card is additive metadata: a write failure must not fail an otherwise-complete build.
    def _boom(card, model_dir) -> None:  # noqa: ANN001
        raise OSError("disk full")

    monkeypatch.setattr(cli_app_module, "save_card", _boom)
    root = tmp_path / ".wmh"
    _build(root, "tau2-airline", tmp_path)  # asserts exit_code == 0 internally
    assert (root / "models" / "tau2-airline" / "config.toml").exists()


def test_cli_exposes_the_small_command_set() -> None:
    names = {cmd.name for cmd in app.registered_commands}
    core = {"build", "list", "serve", "demo", "eval", "play", "download", "knowledge"}
    platform = {"login", "logout", "status", "push", "pull", "run"}
    assert names == core | platform


def test_knowledge_command_prints_path_and_files(tmp_path) -> None:  # noqa: ANN001 - fixture
    from wmh.config import save_config
    from wmh.config.config import HarnessConfig
    from wmh.engine.knowledge import KnowledgeBase

    root = tmp_path / ".wmh"
    model_dir = root / "models" / "airline"
    save_config(HarnessConfig(), root=model_dir)
    KnowledgeBase(model_dir / "knowledge").write_file("rules.md", "- gate: auth required")

    result = runner.invoke(app, ["knowledge", "--name", "airline", "--root", str(root)])
    assert result.exit_code == 0, result.output
    assert "knowledge" in result.output  # the folder path (the real editing surface)
    assert "rules.md" in result.output
    assert "gate: auth required" in result.output


def test_knowledge_command_without_kb_says_how_to_enable(tmp_path) -> None:  # noqa: ANN001
    from wmh.config import save_config
    from wmh.config.config import HarnessConfig

    root = tmp_path / ".wmh"
    save_config(HarnessConfig(), root=root / "models" / "airline")
    result = runner.invoke(app, ["knowledge", "--name", "airline", "--root", str(root)])
    assert result.exit_code == 0, result.output
    assert "empty" in result.output.lower()


@pytest.mark.parametrize("args", [[], ["providers"], ["examples"], ["config"]])
def test_bare_invocation_shows_help(args: list[str]) -> None:
    result = runner.invoke(app, args)
    assert "Missing command" not in result.output
    assert "Usage:" in result.output
    assert "--help" in result.output
    # Bare invocation keeps the usage-error exit code (click >=8.2), unlike explicit --help
    # which exits 0 — scripts can still tell "asked for help" from "forgot the command".
    assert result.exit_code == 2


def test_build_rejects_invalid_name_flag_with_friendly_error(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(
        app,
        ["build", "--name", "tau/bench", "--file", _traces_file(tmp_path), "--no-interactive"],
    )
    assert result.exit_code == 2  # usage error, not a ValueError traceback
    assert "invalid world model name" in result.output


def test_examples_run_rejects_invalid_name_with_friendly_error() -> None:
    result = runner.invoke(app, ["examples", "run", "tau bench"])
    assert result.exit_code == 2  # usage error, not a ValueError traceback
    assert "unknown example" in result.output


def test_serve_rejects_invalid_name_with_friendly_error(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(app, ["serve", "--name", "tau bench", "--root", str(tmp_path / ".wmh")])
    assert result.exit_code == 2  # usage error, not a ValueError traceback
    assert "invalid world model name" in result.output


def test_examples_discovery_skips_unresolvable_names(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    # A dir whose name validate_name rejects can never be run, so list (and the "available:"
    # hint in the unknown-example error) must not advertise it.
    for dirname in ("good-example", "tau bench"):
        example = tmp_path / dirname
        example.mkdir()
        (example / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(cli_app_module, "_benchmark_roots", lambda: (tmp_path,))

    listed = runner.invoke(app, ["examples", "list"])
    assert listed.exit_code == 0, listed.output
    assert "good-example" in listed.output
    assert "tau bench" not in listed.output

    unknown = runner.invoke(app, ["examples", "run", "nope"])
    assert unknown.exit_code == 2
    assert "available: good-example" in unknown.output


def test_main_entry_loads_dotenv_before_dispatch(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    # The persistence half of the wizard's credential flow: keys saved to .env must be back in
    # os.environ on the next `wmh` invocation (main), and importing the module must NOT load.
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("WMH_TEST_MAIN_VAR=loaded\n", encoding="utf-8")
    monkeypatch.delenv("WMH_TEST_MAIN_VAR", raising=False)
    monkeypatch.setattr(cli_app_module, "app", lambda: None)
    cli_app_module.main()
    assert os.environ["WMH_TEST_MAIN_VAR"] == "loaded"


def test_demo_replays_a_sampled_scenario_open_loop(patched_provider, tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmh"
    _build(root, "demo-model", tmp_path)
    result = runner.invoke(
        app,
        [
            "demo",
            "--name",
            "demo-model",
            "--root",
            str(root),
            "--traces",
            _traces_file(tmp_path),
            "--seed",
            "0",
            "--steps",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "replaying scenario" in result.output
    assert "predicted" in result.output
    assert "actual" in result.output
    assert "exact matches" in result.output


def test_retry_narrator_dedupes_identical_failures_and_counts_down(monkeypatch) -> None:  # noqa: ANN001
    from rich.console import Console as RichConsole

    _RetryNarrator = cli_app_module._RetryNarrator

    console = RichConsole(force_terminal=False, no_color=True, width=100)

    class Boto(Exception):
        def __init__(self, code: str) -> None:
            super().__init__("An error occurred (reached max retries: 1)")
            self.response = {"Error": {"Code": code, "Message": "Bedrock is unable"}}

    class FakeStatus:
        def __init__(self) -> None:
            self.updates: list[str] = []

        def update(self, text: str) -> None:
            self.updates.append(text)

    monkeypatch.setattr(cli_app_module.time, "sleep", lambda _s: None)
    narrator = _RetryNarrator(console)
    status = FakeStatus()
    narrator.attach(status, "busy")
    with console.capture() as cap:
        narrator.on_retry(1, 3, 1.0, Boto("ServiceUnavailableException"))
        narrator.sleep(1.0)
        narrator.on_retry(2, 3, 3.0, Boto("ServiceUnavailableException"))  # same failure: silent
        narrator.sleep(3.0)
        narrator.on_retry(3, 3, 9.0, Boto("ThrottlingException"))  # different: printed
    out = cap.get()
    assert out.count("provider hiccup") == 2  # deduped consecutive identical failures
    assert "ServiceUnavailableException: Bedrock is unable" in out
    assert "reached max retries" not in out  # transport chatter stripped
    assert "retry 2/3 — waiting 3s…" in " ".join(status.updates)  # inline countdown
    assert status.updates[-1] == "busy"  # spinner text restored after the wait


def test_providers_subcommand_is_registered() -> None:
    group_names = {group.name for group in app.registered_groups}
    assert "providers" in group_names
    assert "examples" in group_names
    assert "config" in group_names


def test_config_telemetry_command_manages_project_settings(tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmh"

    disabled = runner.invoke(app, ["config", "telemetry", "disable", "--root", str(root)])
    assert disabled.exit_code == 0, disabled.output
    assert "telemetry disabled" in disabled.output
    assert "enabled = false" in (root / "settings.toml").read_text(encoding="utf-8")

    status = runner.invoke(app, ["config", "telemetry", "--root", str(root)])
    assert status.exit_code == 0, status.output
    assert "telemetry disabled" in status.output

    enabled = runner.invoke(app, ["config", "telemetry", "enable", "--root", str(root)])
    assert enabled.exit_code == 0, enabled.output
    assert "telemetry enabled" in enabled.output


def test_examples_list_shows_task_folders() -> None:
    result = runner.invoke(app, ["examples", "list"])
    assert result.exit_code == 0, result.output
    assert "tau-bench" in result.output
    assert "swe-bench" in result.output
    assert "terminal-tasks" in result.output


def test_examples_run_invokes_task_launcher(monkeypatch) -> None:  # noqa: ANN001
    seen: dict[str, object] = {}

    def fake_run(command: list[str], *, cwd: object, check: bool) -> subprocess.CompletedProcess:
        seen["command"] = command
        seen["cwd"] = cwd
        seen["check"] = check
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = runner.invoke(app, ["examples", "run", "tau-bench", "--", "--trace", "0"])

    assert result.exit_code == 0, result.output
    command = cast(list[str], seen["command"])
    assert command[0].endswith("environment-capture/tau-bench/run.sh")
    assert command[1:] == ["--trace", "0"]
    assert str(seen["cwd"]).endswith("environment-capture/tau-bench")
    assert seen["check"] is False


def test_eval_trace_file_command_still_scores(patched_provider, tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(
        app,
        ["eval", _traces_file(tmp_path), "--no-rag"],
    )

    assert result.exit_code == 0, result.output
    assert "OVERALL" in result.output
    assert "fidelity=0.500" in result.output


def test_eval_pins_the_judge_off_the_failover_chain(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    # World-model calls may fail over (provider_or_chain); the judge is the metric and must stay
    # pinned to the single requested backend — a judge that silently switches models mid-run
    # makes fidelity numbers incomparable.
    import wmh.providers as providers_pkg

    chain = FakeProvider()
    pinned = FakeProvider()
    configs: list[ProviderConfig] = []

    def provider_or_chain(config: ProviderConfig, **kw) -> FakeProvider:  # noqa: ANN003
        configs.append(config)
        return chain

    def get_provider(config: ProviderConfig) -> FakeProvider:
        configs.append(config)
        return pinned

    monkeypatch.setattr(providers_pkg, "provider_or_chain", provider_or_chain)
    monkeypatch.setattr(providers_pkg, "get_provider", get_provider)

    result = runner.invoke(app, ["eval", _traces_file(tmp_path), "--no-rag"])

    assert result.exit_code == 0, result.output
    judge_systems_chain = [s for s in chain.systems if "grade a world model" in s]
    judge_systems_pinned = [s for s in pinned.systems if "grade a world model" in s]
    assert judge_systems_chain == []  # the chain never judges
    assert judge_systems_pinned  # every judge call went to the pinned backend
    prediction_systems = [s for s in chain.systems if "grade a world model" not in s]
    assert prediction_systems  # predictions went through the chain
    assert [config.model for config in configs] == [
        "us.anthropic.claude-opus-4-8",
        "us.anthropic.claude-opus-4-8",
    ]
    assert all(config.model_type == "claude-opus-4-8" for config in configs)


def test_eval_pinned_judge_builds_its_own_provider(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    """--judge-model pins the grader to its own provider config (vs riding the serve model)."""
    import wmh.providers as providers_pkg
    import wmh.providers.waterfall as waterfall_mod

    requested: list[tuple[str, str]] = []

    def recording_get_provider(config):  # noqa: ANN001, ANN202
        requested.append((config.kind.value, config.model))
        return FakeProvider()

    monkeypatch.setattr(providers_pkg, "get_provider", recording_get_provider)
    # the serve model rides the chain seam; with no chain file it passes through to
    # get_provider, so record that construction too.
    monkeypatch.setattr(waterfall_mod, "get_provider", recording_get_provider)
    result = runner.invoke(
        app,
        [
            "eval",
            _traces_file(tmp_path),
            "--no-rag",
            "--model",
            "serve-model",
            "--judge-model",
            "pinned-judge",
            "--judge-provider",
            "bedrock",
        ],
    )
    assert result.exit_code == 0, result.output
    assert ("bedrock", "serve-model") in requested
    assert ("bedrock", "pinned-judge") in requested


def test_eval_grid_threads_judge_flags_into_run_grid(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    """`wmh eval grid` honors --judge-model/--judge-provider/--judge-region (not just file mode)."""
    from wmh.evals.grid import GridResult

    examples_root = tmp_path / "examples"
    evals_dir = examples_root / "tiny-task" / "evals"
    evals_dir.mkdir(parents=True)
    (examples_root / "tiny-task" / "traces.otel.jsonl").write_text(
        Path(_traces_file(tmp_path)).read_text(encoding="utf-8"), encoding="utf-8"
    )
    (evals_dir / "default.toml").write_text(
        'description = "Tiny suite"\nfiles = ["../traces.otel.jsonl"]\ntrain_split = 0.5\n',
        encoding="utf-8",
    )

    captured: dict[str, str | None] = {}

    def fake_run_grid(**kwargs):  # noqa: ANN003, ANN202
        captured.update(
            {k: kwargs[k] for k in ("judge_model", "judge_provider", "judge_region")}
        )
        return GridResult(
            suite="tiny-task/default",
            judge_model=kwargs["judge_model"],
            judge_provider=kwargs["judge_provider"],
            train_split=0.5,
            top_k=5,
            seed=0,
            sample_turns="all",
        )

    monkeypatch.setattr(cli_app_module, "run_grid", fake_run_grid)
    monkeypatch.setattr(cli_app_module, "plot_grid", lambda *a, **kw: None)
    result = runner.invoke(
        app,
        [
            "eval",
            "grid",
            "tiny-task",
            "--examples-root",
            str(examples_root),
            "--results-root",
            str(tmp_path / ".wmh" / "evals"),
            "--judge-model",
            "pinned-grid-judge",
            "--judge-provider",
            "openai",
            "--judge-region",
            "eu-west-1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured == {
        "judge_model": "pinned-grid-judge",
        "judge_provider": "openai",
        "judge_region": "eu-west-1",
    }

    # unset -> the canonical pinned default, not the serve model
    captured.clear()
    result = runner.invoke(
        app,
        [
            "eval",
            "grid",
            "tiny-task",
            "--examples-root",
            str(examples_root),
            "--results-root",
            str(tmp_path / ".wmh" / "evals"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["judge_model"] == cli_app_module._GRID_JUDGE_DEFAULT


def test_eval_suite_list_run_and_results(patched_provider, tmp_path) -> None:  # noqa: ANN001
    examples_root = tmp_path / "examples"
    task_dir = examples_root / "tiny-task"
    evals_dir = task_dir / "evals"
    evals_dir.mkdir(parents=True)
    trace_path = task_dir / "traces.otel.jsonl"
    trace_path.write_text(
        Path(_traces_file(tmp_path)).read_text(encoding="utf-8"), encoding="utf-8"
    )
    (evals_dir / "default.toml").write_text(
        "\n".join(
            [
                'description = "Tiny deterministic suite"',
                'files = ["../traces.otel.jsonl"]',
                "train_split = 0.5",
            ]
        ),
        encoding="utf-8",
    )

    listed = runner.invoke(app, ["eval", "list", "--examples-root", str(examples_root)])
    assert listed.exit_code == 0, listed.output
    assert "tiny-task/default" in listed.output

    results_root = tmp_path / ".wmh" / "evals"
    ran = runner.invoke(
        app,
        [
            "eval",
            "run",
            "tiny-task",
            "--examples-root",
            str(examples_root),
            "--results-root",
            str(results_root),
        ],
    )
    assert ran.exit_code == 0, ran.output
    assert "wrote eval result" in ran.output
    result_files = list(results_root.glob("tiny-task/default/*.json"))
    assert len(result_files) == 1
    payload = json.loads(result_files[0].read_text(encoding="utf-8"))
    assert payload["suite"] == "tiny-task/default"
    assert payload["report"]["overall_fidelity"] == 0.5
    assert set(payload["report"]["per_file"]) == {"tiny-task"}

    summarized = runner.invoke(
        app,
        [
            "eval",
            "results",
            "tiny-task",
            "--examples-root",
            str(examples_root),
            "--results-root",
            str(results_root),
        ],
    )
    assert summarized.exit_code == 0, summarized.output
    assert "tiny-task/default" in summarized.output
    assert "0.500" in summarized.output


def test_build_then_list_shows_named_model(patched_provider, tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmh"
    _build(root, "tau2-airline", tmp_path)

    # The artifact lands under <root>/models/<name>/.
    assert (root / "models" / "tau2-airline" / "config.toml").exists()

    listed = runner.invoke(app, ["list", "--root", str(root)])
    assert listed.exit_code == 0, listed.output
    assert "tau2-airline" in listed.output


def test_list_empty_project_is_friendly(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(app, ["list", "--root", str(tmp_path / ".wmh")])
    assert result.exit_code == 0
    assert "no world models" in result.output


def test_play_repl_steps_and_quits(patched_provider, tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmh"
    _build(root, "default", tmp_path)

    # Feed one tool call then quit; the world model's canned observation should surface.
    result = runner.invoke(
        app,
        ["play", "--root", str(root), "--task", "look up users"],
        input='get_user {"id": "u1"}\n:quit\n',
    )
    assert result.exit_code == 0, result.output
    assert "user u1 found" in result.output


def test_build_interactive_wizard_creates_model(
    patched_provider,  # noqa: ANN001 - pytest fixture
    tmp_path,  # noqa: ANN001 - pytest fixture
    monkeypatch,  # noqa: ANN001 - pytest fixture
) -> None:
    root = tmp_path / ".wmh"
    for var in ("AWS_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(var, "test-cred")  # creds present: no interactive key prompts
    # --interactive forces the wizard even under CliRunner (non-TTY); feed each answer line in
    # prompt order: name, trace source (select), file, provider (select), model (select), region
    # (bedrock only), judge model (select), budget, embedder (select). The offline 'hashing'
    # embedder skips the embed-model prompt; phi dim isn't prompted. Selects pick by index.
    answers = "\n".join(
        [
            "wizard-built",
            "",  # trace source: accept the default (otel-genai)
            _traces_file(tmp_path),
            "3",  # provider: bedrock (order: openai, anthropic, bedrock, azure, ...)
            "1",  # model: us.anthropic.claude-opus-4-8
            "us-east-1",
            "",  # judge model: accept the bedrock default (dated haiku)
            "1",  # fidelity: low (RAG only)
            "1",  # embedder: hashing
        ]
    )
    result = runner.invoke(
        app, ["build", "--interactive", "--root", str(root)], input=answers + "\n"
    )
    assert result.exit_code == 0, result.output
    assert (root / "models" / "wizard-built" / "config.toml").exists()


def test_build_non_interactive_without_source_errors(tmp_path) -> None:  # noqa: ANN001
    # No --file/--vendor and --no-interactive: should fail fast rather than hang on input.
    result = runner.invoke(app, ["build", "--no-interactive", "--root", str(tmp_path / ".wmh")])
    assert result.exit_code != 0


def test_build_aborts_when_provider_sdk_missing(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    """A missing SDK must abort the build before any rollouts, with the `uv sync` extra hint.

    Regression: previously the ModuleNotFoundError was swallowed inside GEPA and the build
    "succeeded" with a useless held-out-0.0 model.
    """
    import sys

    from wmh.providers.base import VerifyResult

    appmod = sys.modules["wmh.cli.app"]
    monkeypatch.setattr(
        appmod,
        "verify_all",
        lambda configs: [
            VerifyResult(
                ok=False,
                kind=configs[0].kind,
                model=configs[0].model,
                detail="No module named 'boto3'",
            )
        ],
    )
    root = tmp_path / ".wmh"
    result = runner.invoke(
        app, ["build", "--name", "x", "--file", _traces_file(tmp_path), "--root", str(root)]
    )
    assert result.exit_code == 1
    assert "run `uv sync` to install the provider SDKs" in result.output
    # Aborted before building: no artifact written.
    assert not (root / "models" / "x" / "config.toml").exists()


def test_play_unknown_model_errors(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(app, ["play", "--name", "nope", "--root", str(tmp_path / ".wmh")])
    assert result.exit_code != 0
    # A clean usage error, not an uncaught FileNotFoundError traceback.
    assert not isinstance(result.exception, FileNotFoundError)
    assert "nope" in result.output


def test_demo_unknown_model_is_clean_error(patched_provider, tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmh"
    _build(root, "airline", tmp_path)
    result = runner.invoke(app, ["demo", "--name", "ghost", "--root", str(root)])
    assert result.exit_code != 0
    # Resolved through _load_model -> _resolve_name; must surface as a usage error, not a traceback.
    assert not isinstance(result.exception, (FileNotFoundError, ValueError))


def test_providers_verify_unknown_model_is_clean_error(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(
        app, ["providers", "verify", "--name", "ghost", "--root", str(tmp_path / ".wmh")]
    )
    assert result.exit_code != 0
    assert not isinstance(result.exception, FileNotFoundError)


def test_providers_verify_empty_project_is_friendly(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(app, ["providers", "verify", "--root", str(tmp_path / ".wmh")])
    assert result.exit_code == 0
    assert "no world models built yet" in result.output


def test_providers_verify_reports_built_model_provider(patched_provider, tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmh"
    _build(root, "airline", tmp_path)
    result = runner.invoke(app, ["providers", "verify", "--root", str(root)])
    assert result.exit_code == 0, result.output
    # The bedrock provider configured at build time shows up in the verify report.
    assert "bedrock" in result.output


def test_research_concurrency_rejects_level_above_scenarios_fixed_n() -> None:
    # Fixed-N: a level above --scenarios would silently cap concurrency at N and duplicate the
    # N-worker point, so it must fail fast (guard fires before any suite/corpus resolution).
    result = runner.invoke(
        app,
        ["research", "concurrency", "any-suite", "--scenarios", "4", "--levels", "1,2,4,8"],
    )
    assert result.exit_code != 0
    assert "levels goes up to 8" in result.output


def test_research_concurrency_allows_levels_up_to_scenarios() -> None:
    # levels == scenarios is fine; the guard must not fire (a later stage may still fail).
    result = runner.invoke(
        app,
        ["research", "concurrency", "any-suite", "--scenarios", "8", "--levels", "1,2,4,8"],
    )
    assert "levels goes up to" not in result.output


def test_swe_bench_concurrency_forces_cache_shared() -> None:
    # swe-bench's fixed-N sweep must force --cache-shared (build shared base+env once, cold-build
    # the per-instance image each level) — NOT --no-family-purge, which would rebuild the base per
    # scenario and let concurrent workers clobber each other's shared images.
    forced = _CONCURRENCY_ISOLATION_FLAGS["swe-bench"]
    assert "--cache-shared" in forced
    assert "--no-family-purge" not in forced


def test_research_concurrency_rejects_non_integer_levels() -> None:
    # A typo in --levels must produce a friendly BadParameter, not a raw int() traceback.
    result = runner.invoke(
        app,
        ["research", "concurrency", "any-suite", "--scenarios", "8", "--levels", "1,2,foo,8"],
    )
    assert result.exit_code != 0
    assert "--levels must be a comma-separated list of integers" in result.output


def test_research_concurrency_rejects_bad_select() -> None:
    result = runner.invoke(
        app,
        [
            "research",
            "concurrency",
            "any-suite",
            "--select",
            "bogus",
            "--scenarios",
            "4",
            "--levels",
            "1,2,4",
        ],
    )
    assert result.exit_code != 0
    assert "--select must be one of" in result.output


def test_scenario_role_llms_resolve_from_settings(monkeypatch) -> None:  # noqa: ANN001
    from wmh.config.settings import ModelRole, ModelsSettings, ProjectSettings

    made: list[ProviderConfig] = []

    def fake_get_provider(config: ProviderConfig) -> ProviderConfig:
        made.append(config)
        return config  # identity provider: assertions read the config directly

    monkeypatch.setattr(cli_app_module.providers, "get_provider", fake_get_provider)
    monkeypatch.setattr(
        cli_app_module,
        "load_settings",
        lambda: ProjectSettings(
            models=ModelsSettings(
                worker=ModelRole(provider="azure", model="gpt-5.4", endpoint="https://x/v1"),
                judge=ModelRole(
                    provider="bedrock", model="us.anthropic.claude-opus-4-8", region="us-east-2"
                ),
            )
        ),
    )
    summary, worker, judge = cli_app_module._scenario_role_llms(None, None, None)
    assert summary is worker  # unset summary falls back to the worker role
    assert cast(ProviderConfig, worker).model == "gpt-5.4"
    assert cast(ProviderConfig, worker).endpoint == "https://x/v1"
    assert cast(ProviderConfig, judge).model == "us.anthropic.claude-opus-4-8"
    assert cast(ProviderConfig, judge).region == "us-east-2"
    assert len(made) == 2  # worker constructed once and shared with summary


def test_scenario_role_llms_cli_flags_pin_every_role(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(cli_app_module.providers, "get_provider", lambda config: config)
    summary, worker, judge = cli_app_module._scenario_role_llms("bedrock", "some-model", None)
    assert summary is worker
    assert worker is judge
    assert cast(ProviderConfig, worker).model == "some-model"


def test_scenario_role_llms_default_when_nothing_configured(monkeypatch) -> None:  # noqa: ANN001
    from wmh.config.settings import ProjectSettings

    monkeypatch.setattr(cli_app_module.providers, "get_provider", lambda config: config)
    monkeypatch.setattr(cli_app_module, "load_settings", lambda: ProjectSettings())
    summary, worker, judge = cli_app_module._scenario_role_llms(None, None, None)
    assert summary is worker
    assert worker is judge
    assert cast(ProviderConfig, worker).model == "us.anthropic.claude-opus-4-8"


def test_download_fetches_named_benchmarks(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    fetched: list[tuple[str, bool]] = []

    def fake_fetch(name: str, *, force: bool = False, on_progress=None) -> Path:  # noqa: ANN001
        fetched.append((name, force))
        return tmp_path / name / "traces.otel.jsonl"

    monkeypatch.setattr(cli_app_module, "fetch_corpus", fake_fetch)
    monkeypatch.setattr(cli_app_module, "corpus_path", lambda name: tmp_path / name / "missing")
    result = runner.invoke(app, ["download", "bird-sql", "dabstep", "--force"])
    assert result.exit_code == 0, result.output
    assert fetched == [("bird-sql", True), ("dabstep", True)]
    assert "fetched" in result.output


def test_download_all_expands_to_the_manifest(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    fetched: list[str] = []
    monkeypatch.setattr(
        cli_app_module,
        "fetch_corpus",
        lambda name, force=False, on_progress=None: fetched.append(name) or tmp_path,
    )
    monkeypatch.setattr(cli_app_module, "corpus_path", lambda name: tmp_path / name / "missing")
    result = runner.invoke(app, ["download", "all"])
    assert result.exit_code == 0, result.output
    assert fetched == sorted(cli_app_module.CORPORA)


def test_download_unknown_benchmark_is_a_usage_error(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.setattr(cli_app_module, "corpus_path", lambda name: tmp_path / name / "missing")
    result = runner.invoke(app, ["download", "nope"])
    assert result.exit_code != 0
    assert "no published corpus" in result.output


def test_download_picker_lists_published_and_fetches_choice(
    monkeypatch,  # noqa: ANN001
    tmp_path: Path,
) -> None:
    from environment_capture.hub import PublishedCorpus

    published = [
        PublishedCorpus(
            benchmark="gaia2",
            repo_id="experiential-labs/wmh-gaia2-traces",
            last_modified="2026-07-06",
        )
    ]
    fetched: list[str] = []
    monkeypatch.setattr(cli_app_module, "published_corpora", lambda: published)
    monkeypatch.setattr(
        cli_app_module,
        "fetch_corpus",
        lambda name, force=False, on_progress=None: fetched.append(name) or tmp_path,
    )
    monkeypatch.setattr(cli_app_module, "corpus_path", lambda name: tmp_path / name / "missing")
    result = runner.invoke(app, ["download"], input="1\n")
    assert result.exit_code == 0, result.output
    assert fetched == ["gaia2"]
    assert "not downloaded" in result.output  # picker showed local status


def test_grid_output_paths_never_collide() -> None:
    # Regression: `--out foo.json` must NOT make the chart PNG overwrite the just-written result
    # JSON. The JSON and PNG always get distinct suffixes off the same stem.
    from wmh.cli.app import _grid_output_paths

    default = Path("/tmp/grid/suite-run.json")
    for out in ("foo.json", "foo.png", "foo", "dir/bar.json"):
        json_path, png_path = _grid_output_paths(out, default)
        assert json_path.suffix == ".json"
        assert png_path.suffix == ".png"
        assert json_path != png_path  # the bug: these were equal for `--out foo.json`
        assert json_path.stem == png_path.stem == Path(out).stem
    # No --out: fall back to the default JSON dest + its .png sibling.
    json_path, png_path = _grid_output_paths(None, default)
    assert json_path == default
    assert png_path == default.with_suffix(".png")


def test_parse_model_specs_validates_provider_and_resolves_model() -> None:
    from wmh.cli.app import _parse_model_specs

    specs = _parse_model_specs(
        "Opus 4.8:bedrock:us.anthropic.claude-opus-4-8,Qwen:openai:qwen-agentworld-35b-a3b"
    )
    assert [(s.label, s.provider, s.model) for s in specs] == [
        ("Opus 4.8", "bedrock", "us.anthropic.claude-opus-4-8"),  # exact wire id preserved
        ("Qwen", "openai", "qwen-agentworld-35b-a3b"),  # self-hosted id passes through unchanged
    ]
    # A bad provider fails at parse time with a clear message, not deep inside run_grid.
    with pytest.raises(typer.BadParameter, match="unknown provider"):
        _parse_model_specs("X:notaprovider:m")
    # Malformed entry (wrong arity) still rejected.
    with pytest.raises(typer.BadParameter, match="bad --models entry"):
        _parse_model_specs("Opus:bedrock")
