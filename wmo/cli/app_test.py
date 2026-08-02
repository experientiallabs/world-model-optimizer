"""Tests for the CLI: command surface + build/list/play driven via CliRunner (fake provider)."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import time
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import typer
from pydantic import ValidationError
from typer.testing import CliRunner

from wmo.cli import app, pool_registry
from wmo.cli.app import _CONCURRENCY_ISOLATION_FLAGS
from wmo.cli.pool_registry import read_pool_entries
from wmo.config import (
    FIDELITY_TIERS,
    FidelityTier,
    HarnessConfig,
    ModelInfo,
    ModelRole,
    WorldModelStore,
    load_config,
    load_settings,
    save_settings,
)
from wmo.core.types import Action, ActionKind, Observation, Step, Trace
from wmo.engine.build import DEFAULT_TRAIN_SPLIT, split_traces, split_traces_3way
from wmo.engine.eval_suites import EvalSuiteConfig
from wmo.ingest import VendorPull
from wmo.providers.base import (
    Completion,
    EmbedderKind,
    Message,
    ProviderConfig,
    ProviderKind,
    VerifyResult,
    verify_via_ping,
)
from wmo.providers.openrouter_pricing import CATALOG_PATH_ENV, PriceCatalog

# The exact text the tinker provider raises: the CLI hint has to recognise THAT
# message, not a paraphrase of it.
from wmo.providers.tinker import _MISSING_TINKER_EXTRA
from wmo.tracking.pricing import ModelPrice

# `wmo.cli`'s `app` attribute (the Typer object) shadows the `wmo.cli.app` submodule on
# plain `import wmo.cli.app as ...`; go through importlib to monkeypatch module globals.
cli_app_module = importlib.import_module("wmo.cli.app")

runner = CliRunner()


def _flat(text: str) -> str:
    """Collapse rich wrapping (and typer's error-box borders) for substring asserts."""
    return " ".join(text.replace("│", " ").split())


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


def _squashed(text: str) -> str:
    """Whitespace-free view of rich output, so a boxed+wrapped message still matches a substring.

    Typer renders usage errors inside a panel that hard-wraps at the terminal width, which splits
    long paths and command hints across lines; dropping whitespace and the box rules puts them
    back together. Callers squash the expected string the same way.
    """
    return "".join(ch for ch in text if not ch.isspace() and ch not in "│┃")


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

    import wmo.providers as providers_pkg
    import wmo.providers.registry as registry
    import wmo.providers.waterfall as waterfall_mod

    fake = FakeProvider()
    # `wmo.engine.__init__` rebinds the name `build` to the function, shadowing the submodule
    # attribute, so reach module objects through sys.modules rather than attribute access.
    monkeypatch.setattr(sys.modules["wmo.engine.build"], "get_provider", lambda config: fake)
    # loader.py (serve/demo/play) and the CLI construct through the chain-aware seam.
    monkeypatch.setattr(
        sys.modules["wmo.engine.loader"], "provider_or_chain", lambda config, **kw: fake
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


def test_build_uses_configured_worker_provider(patched_provider, tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmo"
    settings = load_settings(root)
    settings.models.worker = ModelRole(provider="openai", model="gpt-5.4-mini")
    save_settings(settings, root)

    result = runner.invoke(
        app,
        [
            "build",
            "--name",
            "configured",
            "--file",
            _traces_file(tmp_path),
            "--fidelity",
            "low",
            "--root",
            str(root),
        ],
    )

    assert result.exit_code == 0, result.output
    config = load_config(root / "models" / "configured")
    assert config.serve_provider is ProviderKind.OPENAI
    assert config.serve_provider_config().model_type == "gpt-5.4-mini"


def test_build_explicit_model_keeps_configured_azure_connection(
    patched_provider: None, tmp_path: Path
) -> None:
    root = tmp_path / ".wmo"
    settings = load_settings(root)
    settings.models.worker = ModelRole(
        provider="azure",
        model="gpt-5.5",
        endpoint="https://azure.example/v1",
        deployment="configured-deployment",
        api_version="2026-01-01",
    )
    save_settings(settings, root)

    result = runner.invoke(
        app,
        [
            "build",
            "--name",
            "explicit-model",
            "--file",
            _traces_file(tmp_path),
            "--provider",
            "azure",
            "--model",
            "gpt-5.5",
            "--fidelity",
            "low",
            "--root",
            str(root),
        ],
    )

    assert result.exit_code == 0, result.output
    config = load_config(root / "models" / "explicit-model")
    provider = config.serve_provider_config()
    assert provider.kind is ProviderKind.AZURE_OPENAI
    assert provider.model_type == "gpt-5.5"
    assert provider.endpoint == "https://azure.example/v1"
    assert provider.deployment == "configured-deployment"
    assert provider.api_version == "2026-01-01"


def test_build_wizard_does_not_reuse_connection_for_changed_provider(
    patched_provider: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / ".wmo"
    settings = load_settings(root)
    settings.models.worker = ModelRole(
        provider="azure",
        model="gpt-5.4",
        endpoint="https://azure.example/v1",
        deployment="configured-deployment",
        api_version="2026-01-01",
    )
    save_settings(settings, root)

    def switch_provider(_console, params):  # noqa: ANN001, ANN202
        return params.model_copy(
            update={
                "name": "wizard-switch",
                "file": _traces_file(tmp_path),
                "provider": "openai",
                "model": "gpt-5.4-mini",
                "region": None,
            }
        )

    monkeypatch.setattr("wmo.cli.ui.run_build_wizard", switch_provider)

    result = runner.invoke(app, ["build", "--interactive", "--root", str(root)])

    assert result.exit_code == 0, result.output
    config = load_config(root / "models" / "wizard-switch")
    provider = config.serve_provider_config()
    assert provider.kind is ProviderKind.OPENAI
    assert provider.endpoint is None
    assert provider.deployment is None
    assert provider.api_version is None


def test_build_writes_model_card(patched_provider, tmp_path) -> None:  # noqa: ANN001
    from wmo.config.card import load_card

    root = tmp_path / ".wmo"
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

    monkeypatch.setattr("wmo.config.card.save_card", _boom)
    root = tmp_path / ".wmo"
    _build(root, "tau2-airline", tmp_path)  # asserts exit_code == 0 internally
    assert (root / "models" / "tau2-airline" / "config.toml").exists()


def test_cli_exposes_the_small_command_set() -> None:
    names = {cmd.name for cmd in app.registered_commands}
    core = {
        "build",
        "ingest",
        "list",
        "serve",
        "demo",
        "eval",
        "play",
        "download",
        "knowledge",
    }
    platform = {"login", "logout", "status", "push", "pull", "run"}
    assert names == core | platform
    # `optimize` is a GROUP now (harness today; route and training-type optimizers join it).
    groups = {group.name for group in app.registered_groups}
    assert "optimize" in groups


def test_knowledge_command_prints_path_and_files(tmp_path) -> None:  # noqa: ANN001 - fixture
    from wmo.config import save_config
    from wmo.config.config import HarnessConfig
    from wmo.engine.knowledge import KnowledgeBase

    root = tmp_path / ".wmo"
    model_dir = root / "models" / "airline"
    save_config(HarnessConfig(), root=model_dir)
    KnowledgeBase(model_dir / "knowledge").write_file("rules.md", "- gate: auth required")

    result = runner.invoke(app, ["knowledge", "--name", "airline", "--root", str(root)])
    assert result.exit_code == 0, result.output
    assert "knowledge" in result.output  # the folder path (the real editing surface)
    assert "rules.md" in result.output
    assert "gate: auth required" in result.output


def test_knowledge_command_prints_bracketed_markdown_verbatim(tmp_path) -> None:  # noqa: ANN001
    """Knowledge is hand-edited markdown, so rich must not read its brackets as style tags.

    Unescaped, `[/items]` raised MarkupError (the command died on ordinary content) and both
    `list[str]` and the link text were silently deleted from the rendered output.
    """
    from wmo.config import save_config
    from wmo.config.config import HarnessConfig
    from wmo.engine.knowledge import KnowledgeBase

    root = tmp_path / ".wmo"
    model_dir = root / "models" / "airline"
    save_config(HarnessConfig(), root=model_dir)
    KnowledgeBase(model_dir / "knowledge").write_file(
        "schemas.md",
        "Use the XML close marker [/items] to end a list.\n"
        "reservations: list[str]\n"
        "See [the docs](https://example.com) for details.",
    )

    result = runner.invoke(app, ["knowledge", "--name", "airline", "--root", str(root)])
    assert result.exit_code == 0, result.exception
    assert "[/items]" in result.output
    assert "list[str]" in result.output
    assert "[the docs](https://example.com)" in result.output


def test_knowledge_command_without_kb_says_how_to_enable(tmp_path) -> None:  # noqa: ANN001
    # The empty state used to print a directory that does not exist and say "drop *.md files in
    # this folder" without naming the flag that seeds one, so this now pins both halves.
    from wmo.config import save_config
    from wmo.config.config import HarnessConfig

    root = tmp_path / ".wmo"
    save_config(HarnessConfig(), root=root / "models" / "airline")
    result = runner.invoke(app, ["knowledge", "--name", "airline", "--root", str(root)])
    assert result.exit_code == 0, result.output
    flat = _squashed(result.output)
    assert _squashed("does not exist yet") in flat  # the printed dir is absent, and it says so
    assert "--knowledge" in flat  # the exact build flag that creates one


def test_knowledge_command_flags_a_kb_the_model_ignores(tmp_path) -> None:  # noqa: ANN001
    """Files under `knowledge/` are inert unless the model was built with `--knowledge`."""
    from wmo.config import save_config
    from wmo.config.config import HarnessConfig
    from wmo.engine.knowledge import KnowledgeBase

    root = tmp_path / ".wmo"
    model_dir = root / "models" / "airline"
    save_config(HarnessConfig(), root=model_dir)  # knowledge=False, the build default
    KnowledgeBase(model_dir / "knowledge").write_file("rules.md", "- gate: auth required")

    result = runner.invoke(app, ["knowledge", "--name", "airline", "--root", str(root)])

    assert result.exit_code == 0, result.output
    flat = _squashed(result.output)
    assert "inert" in flat
    assert "--knowledge" in flat  # names the flag that activates them
    assert _squashed("gate: auth required") in flat  # the files are still shown


def test_knowledge_command_stays_quiet_when_the_kb_is_live(tmp_path) -> None:  # noqa: ANN001
    from wmo.config import save_config
    from wmo.config.config import HarnessConfig
    from wmo.engine.knowledge import KnowledgeBase

    root = tmp_path / ".wmo"
    model_dir = root / "models" / "airline"
    save_config(HarnessConfig(knowledge=True), root=model_dir)
    KnowledgeBase(model_dir / "knowledge").write_file("rules.md", "- gate: auth required")

    result = runner.invoke(app, ["knowledge", "--name", "airline", "--root", str(root)])

    assert result.exit_code == 0, result.output
    assert "inert" not in result.output


def test_knowledge_resolves_a_shipped_example_like_demo_and_play(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    """`wmo knowledge` must see the same models `wmo demo`/`wmo play` resolve, examples included."""
    from wmo.config import save_config
    from wmo.config.config import HarnessConfig
    from wmo.engine.knowledge import KnowledgeBase

    example = tmp_path / "airline-bench"
    model_dir = example / "models" / "airline"
    save_config(HarnessConfig(knowledge=True), root=model_dir)
    (example / "traces.otel.jsonl").write_text("", encoding="utf-8")
    KnowledgeBase(model_dir / "knowledge").write_file("rules.md", "- gate: auth required")
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(cli_app_module, "_benchmark_roots", lambda: (tmp_path,))
    monkeypatch.chdir(project)

    result = runner.invoke(app, ["knowledge", "--name", "airline"])

    assert result.exit_code == 0, result.output
    assert "gate: auth required" in result.output


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["build", "--help"], "[deprecated] alias for --source"),
        (["eval", "--help"], "`[models.agent]` selects a distinct agent provider"),
        (["providers", "set", "--help"], "settings.toml` as `[models.worker]`"),
        (["providers", "verify", "--help"], "the `[models.<role>]` roles in"),
        (["scenarios", "build", "--help"], "settings.toml [models.worker|judge|summary]."),
    ],
)
def test_help_keeps_the_bracketed_pointer_it_exists_to_teach(
    argv: list[str], expected: str
) -> None:
    """Typer renders help through rich markup, which swallows an unescaped `[...]` whole.

    Each of these is the only pointer in that help text to where the setting lives (or, for
    `--vendor`, the only sign that the option is deprecated), so a swallowed pair is silent
    misinformation.
    """
    result = runner.invoke(app, argv)
    assert result.exit_code == 0, result.output
    rendered = " ".join(result.output.replace("│", " ").split())
    assert expected in rendered


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


def test_build_rejects_the_reserved_harbor_name(tmp_path) -> None:  # noqa: ANN001
    """`harbor` is the optimize environment literal, so no world model may claim it."""
    result = runner.invoke(
        app,
        ["build", "--name", "harbor", "--file", _traces_file(tmp_path), "--no-interactive"],
    )
    assert result.exit_code == 2
    assert "reserved" in result.output


def test_examples_run_rejects_invalid_name_with_friendly_error() -> None:
    result = runner.invoke(app, ["examples", "run", "tau bench"])
    assert result.exit_code == 2  # usage error, not a ValueError traceback
    assert "unknown example" in result.output


def test_serve_rejects_invalid_name_with_friendly_error(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(app, ["serve", "--name", "tau bench", "--root", str(tmp_path / ".wmo")])
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


def _flat(output: str) -> str:
    """Rich wraps error panels; flatten box drawing and newlines before matching a message."""
    return " ".join(output.replace("│", " ").split())


def test_examples_data_only_bundle_is_marked_and_points_at_wmo_build(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    # What `wmo download` fetches: a corpus with no launcher. `list` must say so, and `run` must
    # name what the bundle IS for instead of dead-ending on "no run.sh launcher".
    bundle = tmp_path / "demo-corpus"
    bundle.mkdir()
    (bundle / "traces.otel.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(cli_app_module, "_benchmark_roots", lambda: (tmp_path,))

    listed = runner.invoke(app, ["examples", "list"])
    assert listed.exit_code == 0, listed.output
    assert "demo-corpus" in listed.output
    assert "data only" in listed.output

    result = runner.invoke(app, ["examples", "run", "demo-corpus"])
    assert result.exit_code == 2, result.output
    output = _flat(result.output)
    assert "data-only bundle" in output
    assert "wmo build --file" in output
    assert "--name demo-corpus" in output


@pytest.mark.skipif(sys.platform == "win32", reason="Windows ignores the Unix exec bit")
def test_examples_run_rejects_a_non_executable_launcher(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    # A run.sh that lost its exec bit (archive, checkout) must be a usage error naming chmod,
    # not a PermissionError traceback out of subprocess.
    example = tmp_path / "noexec"
    example.mkdir()
    launcher = example / "run.sh"
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    launcher.chmod(0o644)
    monkeypatch.setattr(cli_app_module, "_benchmark_roots", lambda: (tmp_path,))

    result = runner.invoke(app, ["examples", "run", "noexec"])

    assert result.exit_code == 2, result.output  # usage error, not a traceback
    assert "chmod +x" in _flat(result.output)


@pytest.mark.parametrize(
    "script",
    [
        pytest.param("echo hi\n", id="no-shebang"),  # execve -> ENOEXEC
        pytest.param("#!/nonexistent/interp\n", id="missing-interpreter"),  # execve -> ENOENT
    ],
)
def test_examples_run_reports_a_launcher_that_cannot_be_started(
    tmp_path,  # noqa: ANN001
    monkeypatch,  # noqa: ANN001
    script: str,
) -> None:
    # X_OK passes but the kernel still refuses to exec, so `subprocess.run` raises before there
    # is any exit code to forward. That must not surface as an OSError traceback either.
    example = tmp_path / "unstartable"
    example.mkdir()
    launcher = example / "run.sh"
    launcher.write_text(script, encoding="utf-8")
    launcher.chmod(0o755)
    monkeypatch.setattr(cli_app_module, "_benchmark_roots", lambda: (tmp_path,))

    result = runner.invoke(app, ["examples", "run", "unstartable"])

    assert result.exit_code == 2, result.output  # usage error, not a traceback
    assert not isinstance(result.exception, OSError), result.exception
    output = _flat(result.output)
    assert "could not start" in output
    assert "head -1" in output


def test_config_help_does_not_reuse_the_harness_group_name() -> None:
    # `wmo harness` is a different group managing a different object; `wmo config` manages the
    # project's own settings file.
    result = runner.invoke(app, ["config", "--help"])
    assert result.exit_code == 0, result.output
    output = _flat(result.output)
    assert "project-local wmo settings" in output
    assert "harness" not in output


def test_serve_help_names_the_openai_endpoint_and_a_real_example_root() -> None:
    # The OpenAI-compatible surface is what README step 3 exists for, and examples/tau-bench
    # moved to packages/environment-capture/ — the help must name both correctly.
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0, result.output
    output = _flat(result.output)
    assert "/v1/chat/completions" in output
    assert "examples/tau-bench" not in output
    assert "packages/environment-capture/tau-bench" in output


def test_main_entry_loads_dotenv_before_dispatch(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    # The persistence half of the wizard's credential flow: keys saved to .env must be back in
    # os.environ on the next `wmo` invocation (main), and importing the module must NOT load.
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("WMO_TEST_MAIN_VAR=loaded\n", encoding="utf-8")
    monkeypatch.delenv("WMO_TEST_MAIN_VAR", raising=False)
    monkeypatch.setattr(cli_app_module, "app", lambda: None)
    cli_app_module.main()
    assert os.environ["WMO_TEST_MAIN_VAR"] == "loaded"


def test_demo_replays_a_sampled_scenario_open_loop(patched_provider, tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmo"
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


@pytest.mark.parametrize("steps", ["0", "-1"])
def test_demo_rejects_a_non_positive_step_budget(patched_provider, tmp_path, steps) -> None:  # noqa: ANN001
    # `--steps 0` used to slice the trace to [] and index [-1] on it: an IndexError traceback.
    root = tmp_path / ".wmo"
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
            "--steps",
            steps,
            "--no-prompt",
        ],
    )
    assert result.exit_code == 2, result.output
    assert not isinstance(result.exception, IndexError)
    assert "--steps" in result.output


def test_demo_missing_explicit_traces_names_the_path_given(patched_provider, tmp_path) -> None:  # noqa: ANN001
    # The old message blamed the model's default location and told you to pass the flag you just
    # passed; an explicit path that does not exist must name that path.
    root = tmp_path / ".wmo"
    _build(root, "demo-model", tmp_path)
    typo = tmp_path / "does-not-exist.jsonl"
    result = runner.invoke(
        app, ["demo", "--name", "demo-model", "--root", str(root), "--traces", str(typo)]
    )
    assert result.exit_code == 2, result.output
    assert _squashed(str(typo)) in _squashed(result.output)


def test_demo_without_a_corpus_names_the_file_to_pass(patched_provider, tmp_path) -> None:  # noqa: ANN001
    """A build keeps no copy of its corpus, so the default can never resolve after `wmo build`."""
    root = tmp_path / ".wmo"
    _build(root, "demo-model", tmp_path)
    result = runner.invoke(app, ["demo", "--name", "demo-model", "--root", str(root)])
    assert result.exit_code == 2, result.output
    flat = _squashed(result.output)
    assert _squashed("keeps no copy of the corpus it read") in flat
    assert _squashed("--traces <that file>") in flat
    assert _squashed(str(root / "models" / "demo-model" / "traces.otel.jsonl")) in flat


def test_demo_finds_a_corpus_stored_beside_the_artifact(patched_provider, tmp_path) -> None:  # noqa: ANN001
    """serve and `optimize route sweep` read <model_dir>/traces.otel.jsonl; demo must too."""
    root = tmp_path / ".wmo"
    _build(root, "demo-model", tmp_path)
    corpus = Path(_traces_file(tmp_path)).read_text(encoding="utf-8")
    (root / "models" / "demo-model" / "traces.otel.jsonl").write_text(corpus, encoding="utf-8")

    result = runner.invoke(
        app, ["demo", "--name", "demo-model", "--root", str(root), "--seed", "0", "--steps", "1"]
    )

    assert result.exit_code == 0, result.output
    assert "replaying scenario" in result.output


def test_demo_provider_failure_is_a_clean_error_not_a_traceback(
    patched_provider: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The likeliest first-run failure: the serve provider has no credentials."""
    import openai

    import wmo.providers as providers_pkg

    root = tmp_path / ".wmo"
    _build(root, "demo-model", tmp_path)

    class _NoCredentials(FakeProvider):
        def complete(self, system, messages, *, temperature=0.7, max_tokens=8192):  # noqa: ANN001, ANN202
            raise openai.OpenAIError("Missing credentials. Please pass an `api_key`")

    monkeypatch.setattr(providers_pkg, "get_provider", lambda config: _NoCredentials())

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
            "--steps",
            "1",
            "--no-prompt",
        ],
    )

    assert result.exit_code == 1, result.output
    assert not isinstance(result.exception, openai.OpenAIError)
    flat = _squashed(result.output)
    assert _squashed("Missing credentials") in flat
    assert _squashed("wmo providers verify") in flat


def test_demo_off_a_terminal_calls_an_outage_an_outage(
    patched_provider: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 429 that outlived the retries is not a credentials problem, and must not claim to be.

    CliRunner is never a terminal, so this is exactly the CI/redirected path: the interactive
    branch (offer a different provider) is unreachable and the capacity error falls through to
    the setup-error renderer, which used to print `check ... your credentials`.
    """
    import httpx
    import openai

    import wmo.providers as providers_pkg

    root = tmp_path / ".wmo"
    _build(root, "demo-model", tmp_path)
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")

    class _RateLimited(FakeProvider):
        def complete(self, system, messages, *, temperature=0.7, max_tokens=8192):  # noqa: ANN001, ANN202
            raise openai.RateLimitError(
                "Rate limit reached for gpt-4o",
                response=httpx.Response(429, request=request),
                body=None,
            )

    monkeypatch.setattr(providers_pkg, "get_provider", lambda config: _RateLimited())

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
            "--steps",
            "1",
            "--no-prompt",
        ],
    )

    assert result.exit_code == 1, result.output
    assert not isinstance(result.exception, openai.RateLimitError)  # still no traceback
    flat = _squashed(result.output)
    assert _squashed("is out of capacity") in flat
    assert _squashed("not a credentials problem") in flat
    assert _squashed("re-run `wmo demo --name demo-model`") in flat  # the exact next command
    assert "credentialsareset" not in flat  # the setup hint, squashed; wrong diagnosis here
    assert "wmoprovidersverify" not in flat  # it would only re-report the same outage


def test_demo_keeps_the_traceback_for_a_wmo_bug(patched_provider, tmp_path, monkeypatch) -> None:  # noqa: ANN001
    """Only backend SDK failures are re-rendered; our own bugs must not be dressed up as setup."""
    root = tmp_path / ".wmo"
    _build(root, "demo-model", tmp_path)
    monkeypatch.setattr(
        "wmo.engine.demo.run_demo",
        lambda *a, **kw: (_ for _ in ()).throw(KeyError("internal")),
    )

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
            "--steps",
            "1",
            "--no-prompt",
        ],
    )

    assert isinstance(result.exception, KeyError)


def test_play_refuses_a_serve_provider_it_cannot_prepare(
    patched_provider: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`prepare()` is free and offline, so a missing credential fails before the first step."""
    import wmo.providers as providers_pkg

    root = tmp_path / ".wmo"
    _build(root, "demo-model", tmp_path)

    class _Unpreparable(FakeProvider):
        def prepare(self) -> None:
            raise RuntimeError("Missing credentials. Set OPENAI_API_KEY")

    monkeypatch.setattr(providers_pkg, "get_provider", lambda config: _Unpreparable())

    result = runner.invoke(app, ["play", "--name", "demo-model", "--root", str(root)])

    assert result.exit_code == 1, result.output
    flat = _squashed(result.output)
    assert _squashed("Missing credentials") in flat
    assert _squashed("wmo providers verify") in flat


def _example_model(root: Path, example: str, name: str) -> Path:
    """A shipped-example artifact: <root>/<example>/models/<name>/ plus the example's corpus."""
    from wmo.config import save_config
    from wmo.config.config import HarnessConfig

    example_dir = root / example
    (example_dir / "traces.otel.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (example_dir / "traces.otel.jsonl").write_text("", encoding="utf-8")
    model_dir = example_dir / "models" / name
    save_config(HarnessConfig(), root=model_dir)
    return model_dir


def test_demo_root_spelling_does_not_change_what_is_discovered(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    # Discovery used to be gated on `root != ".wmo"` string equality, so `./.wmo` (or the `.wmo/`
    # shell tab-completion types) silently searched a different set than the identical `.wmo`.
    _example_model(tmp_path, "airline-bench", "airline")
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(cli_app_module, "_benchmark_roots", lambda: (tmp_path,))
    monkeypatch.chdir(project)

    outputs = [
        _squashed(runner.invoke(app, ["demo", "--name", "ghost", "--root", spelling]).output)
        for spelling in (".wmo", "./.wmo", ".wmo/")
    ]

    assert all(_squashed("airline (airline-bench example)") in out for out in outputs), outputs


def test_demo_lists_a_shadowed_example_distinguishably(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    # A local build of the same name used to appear twice in the list, and --name could not say
    # which one was meant.
    from wmo.config import save_config
    from wmo.config.config import HarnessConfig

    _example_model(tmp_path, "airline-bench", "airline")
    project = tmp_path / "project"
    save_config(HarnessConfig(), root=project / ".wmo" / "models" / "airline")
    _example_model(tmp_path, "retail-bench", "retail")
    monkeypatch.setattr(cli_app_module, "_benchmark_roots", lambda: (tmp_path,))
    monkeypatch.chdir(project)

    result = runner.invoke(app, ["demo"])

    assert result.exit_code == 2, result.output
    flat = _squashed(result.output)
    assert _squashed("airline (local)") in flat
    assert _squashed("airline (airline-bench example)") in flat
    assert _squashed(f"--root {tmp_path / 'airline-bench'}") in flat


def test_demo_with_nothing_built_points_at_a_command_that_exists(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    # The old hint named `examples/tau-bench`, a path that ships in neither the wheel nor the repo.
    empty_root = tmp_path / "no-examples"
    empty_root.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(cli_app_module, "_benchmark_roots", lambda: (empty_root,))
    monkeypatch.chdir(project)

    result = runner.invoke(app, ["demo"])

    assert result.exit_code == 2, result.output
    flat = _squashed(result.output)
    assert "examples/tau-bench" not in flat
    assert "wmodownload" in flat
    assert _squashed("wmo build --file <traces> --name <name>") in flat


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
    root = tmp_path / ".wmo"

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


def test_providers_set_verifies_and_saves_local_worker(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmo"
    checked: list[ProviderConfig] = []

    def verify(configs: list[ProviderConfig]) -> list[VerifyResult]:
        checked.extend(configs)
        return [VerifyResult(ok=True, kind=config.kind, model=config.model) for config in configs]

    monkeypatch.setattr("wmo.providers.verify_all", verify)
    result = runner.invoke(
        app,
        [
            "providers",
            "set",
            "--provider",
            "openai",
            "--model",
            "gpt-5.4-mini",
            "--endpoint",
            "https://models.example/v1",
            "--root",
            str(root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert checked[0].kind is ProviderKind.OPENAI
    worker = load_settings(root).models.worker
    assert worker is not None
    assert worker.provider == "openai"
    assert worker.model == "gpt-5.4-mini"
    assert worker.endpoint == "https://models.example/v1"


def _accept_every_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub both live pings so `providers set` reaches, and gets through, pool registration.

    Two seams, because the command proves two different things: `verify_all` proves the worker
    provider, and `verify_pool_entry` proves each routing candidate over its own route.
    """
    monkeypatch.setattr(
        "wmo.providers.verify_all",
        lambda configs: [
            VerifyResult(ok=True, kind=config.kind, model=config.model) for config in configs
        ],
    )
    monkeypatch.setattr(
        pool_registry,
        "verify_pool_entry",
        lambda entry: VerifyResult(ok=True, kind=entry.kind, model=entry.model),
    )


def _seed_openrouter_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the OpenRouter price resolver at a fixture catalog (the suite never fetches)."""
    catalog = PriceCatalog(
        fetched_at=time.time(),
        source="test fixture",
        prices={
            "anthropic/claude-sonnet-4.5": ModelPrice(input_per_mtok=3.0, output_per_mtok=15.0)
        },
    )
    path = tmp_path / "openrouter-prices.json"
    path.write_text(catalog.model_dump_json(), encoding="utf-8")
    monkeypatch.setenv(CATALOG_PATH_ENV, str(path))


def test_providers_set_registers_pool_models_beside_the_settings_it_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The blocker this command exists to remove: nothing wrote `pool.toml` for an ordinary
    # provider model, so a router had no candidates without hand-authored TOML.
    _accept_every_provider(monkeypatch)
    _seed_openrouter_catalog(tmp_path, monkeypatch)
    root = tmp_path / ".wmo"

    result = runner.invoke(
        app,
        [
            "providers",
            "set",
            "--provider",
            "openrouter",
            "--model",
            "anthropic/claude-sonnet-4.5",
            "--pool-model",
            "anthropic/claude-sonnet-4.5",
            "--tier",
            "open",
            "--root",
            str(root),
        ],
    )

    assert result.exit_code == 0, result.output
    worker = load_settings(root).models.worker
    assert worker is not None and worker.provider == "openrouter"
    entries = read_pool_entries(root / "pool.toml")
    assert [(entry.name, entry.model, entry.tier) for entry in entries] == [
        ("claude-sonnet-4.5", "anthropic/claude-sonnet-4.5", "open")
    ]
    # Priced from the published catalog, so the roster never reports $0 for this candidate.
    assert entries[0].price().input_per_mtok == 3.0


def test_providers_set_registers_into_an_explicit_pool_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _accept_every_provider(monkeypatch)
    roster = tmp_path / "rosters" / "candidates.toml"

    result = runner.invoke(
        app,
        [
            "providers",
            "set",
            "--provider",
            "bedrock",
            "--model",
            "claude-opus-4-8",
            "--pool-model",
            "claude-haiku-4-5",
            "--pool",
            str(roster),
            "--root",
            str(tmp_path / ".wmo"),
        ],
    )

    assert result.exit_code == 0, result.output
    entry = read_pool_entries(roster)[0]
    assert entry.kind is ProviderKind.BEDROCK
    # Resolved through the built-in registry, so the entry carries the callable runtime id.
    assert entry.model == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert entry.model_type == "claude-haiku-4-5"


def test_providers_set_refuses_a_pool_model_it_cannot_price(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A candidate with no price silently costs $0, and a cost-aware policy routes everything to
    # it. Non-interactively there is nobody to ask, so the command has to refuse.
    _accept_every_provider(monkeypatch)
    root = tmp_path / ".wmo"

    result = runner.invoke(
        app,
        [
            "providers",
            "set",
            "--provider",
            "openai",
            "--model",
            "gpt-5.4-mini",
            "--pool-model",
            "some-unlisted-model",
            "--root",
            str(root),
        ],
    )

    assert result.exit_code != 0
    assert "no built-in price" in result.output
    assert not (root / "pool.toml").exists()


def test_providers_set_prices_a_pool_model_from_the_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _accept_every_provider(monkeypatch)
    root = tmp_path / ".wmo"

    result = runner.invoke(
        app,
        [
            "providers",
            "set",
            "--provider",
            "openai",
            "--model",
            "gpt-5.4-mini",
            "--endpoint",
            "https://vllm.example/v1",
            "--pool-model",
            "qwen3-32b",
            "--input-per-mtok",
            "0.1",
            "--output-per-mtok",
            "0.4",
            "--api-key-env",
            "WMO_ENDPOINT_API_KEY",
            "--root",
            str(root),
        ],
    )

    assert result.exit_code == 0, result.output
    entry = read_pool_entries(root / "pool.toml")[0]
    assert (entry.input_per_mtok, entry.output_per_mtok) == (0.1, 0.4)
    assert entry.endpoint == "https://vllm.example/v1"
    assert entry.api_key_env == "WMO_ENDPOINT_API_KEY"


def test_providers_set_rejects_an_unknown_tier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _accept_every_provider(monkeypatch)

    result = runner.invoke(
        app,
        [
            "providers",
            "set",
            "--provider",
            "openai",
            "--model",
            "gpt-5.4-mini",
            "--tier",
            "cheap",
            "--root",
            str(tmp_path / ".wmo"),
        ],
    )

    assert result.exit_code != 0
    assert "frontier, open" in result.output


def test_providers_set_never_guesses_an_azure_deployment_for_the_pool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The worker config fills an Azure deployment in from the model id when none was given; a
    # pool entry must not inherit that guess, because Azure sends the deployment as the request
    # model and a guessed name addresses a route that does not exist.
    _accept_every_provider(monkeypatch)
    root = tmp_path / ".wmo"

    result = runner.invoke(
        app,
        [
            "providers",
            "set",
            "--provider",
            "azure",
            "--model",
            "gpt-5.5",
            "--pool-model",
            "gpt-5.5",
            "--root",
            str(root),
        ],
    )

    assert result.exit_code != 0
    assert "azure needs --deployment" in result.output
    assert not (root / "pool.toml").exists()
    # The worker role is still saved with its derived deployment: only the pool is strict.
    worker = load_settings(root).models.worker
    assert worker is not None and worker.deployment == "gpt-5.5"


def test_providers_set_registers_a_named_azure_deployment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _accept_every_provider(monkeypatch)
    root = tmp_path / ".wmo"

    result = runner.invoke(
        app,
        [
            "providers",
            "set",
            "--provider",
            "azure",
            "--model",
            "gpt-5.5",
            "--deployment",
            "chat-prod",
            "--api-version",
            "2025-01-01-preview",
            "--pool-model",
            "gpt-5.5",
            "--root",
            str(root),
        ],
    )

    assert result.exit_code == 0, result.output
    entry = read_pool_entries(root / "pool.toml")[0]
    assert (entry.name, entry.model, entry.deployment) == ("chat-prod", "gpt-5.5", "chat-prod")
    assert entry.api_version == "2025-01-01-preview"


def test_providers_set_refuses_a_pool_model_that_cannot_be_called(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A --pool-model can differ from the verified worker in model, endpoint, deployment and
    # credential, so the worker's ping proves nothing about it. Registering it unproved would
    # surface as a 401 inside a paid `route sweep`, after the candidates ahead of it were billed.
    _accept_every_provider(monkeypatch)
    monkeypatch.setattr(
        pool_registry,
        "verify_pool_entry",
        lambda entry: VerifyResult(
            ok=False, kind=entry.kind, model=entry.model, detail="401 unauthorized"
        ),
    )
    root = tmp_path / ".wmo"

    result = runner.invoke(
        app,
        [
            "providers",
            "set",
            "--provider",
            "openai",
            "--model",
            "gpt-5.4-mini",
            "--pool-model",
            "gpt-5.4",
            "--root",
            str(root),
        ],
    )

    assert result.exit_code != 0
    assert "not callable" in result.output
    assert not (root / "pool.toml").exists()


def test_providers_set_rejects_half_a_price_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A pool entry prices both token tiers or neither. Half a pair would be dropped silently in
    # the interactive flow and rejected as an invalid entry with --pool-model, so it is refused
    # up front, where the message can name the flag.
    _accept_every_provider(monkeypatch)
    root = tmp_path / ".wmo"

    result = runner.invoke(
        app,
        [
            "providers",
            "set",
            "--provider",
            "openai",
            "--model",
            "gpt-5.4-mini",
            "--pool-model",
            "qwen3-32b",
            "--input-per-mtok",
            "0.1",
            "--root",
            str(root),
        ],
    )

    assert result.exit_code != 0
    assert "both --input-per-mtok and --output-per-mtok" in result.output
    assert not (root / "pool.toml").exists()


def test_providers_set_without_pool_flags_leaves_scripted_runs_untouched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The pre-existing contract: `--provider` + `--model` prompts for nothing and writes only
    # settings. Registration is an addition, never something a script trips over.
    _accept_every_provider(monkeypatch)
    root = tmp_path / ".wmo"

    result = runner.invoke(
        app,
        [
            "providers",
            "set",
            "--provider",
            "openai",
            "--model",
            "gpt-5.4-mini",
            "--root",
            str(root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert load_settings(root).models.worker is not None
    assert not (root / "pool.toml").exists()
    assert "Register models" not in result.output


def test_providers_set_does_not_save_a_failed_provider(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmo"
    monkeypatch.setattr(
        "wmo.providers.verify_all",
        lambda configs: [
            VerifyResult(
                ok=False,
                kind=configs[0].kind,
                model=configs[0].model,
                detail="bad key",
            )
        ],
    )

    result = runner.invoke(
        app,
        [
            "providers",
            "set",
            "--provider",
            "openai",
            "--model",
            "gpt-5.4-mini",
            "--root",
            str(root),
        ],
    )

    assert result.exit_code == 1
    assert "bad key" in result.output
    assert load_settings(root).models.worker is None


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
    assert Path(command[0]).as_posix().endswith("environment-capture/tau-bench/run.sh")
    assert command[1:] == ["--trace", "0"]
    assert Path(str(seen["cwd"])).as_posix().endswith("environment-capture/tau-bench")
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
    import wmo.providers as providers_pkg

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
    traces = _traces_file(tmp_path)
    # No settings.toml here, so the asserted bedrock ids are the no-role-configured fallback.
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["eval", traces, "--no-rag"])

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


def _record_eval_providers(monkeypatch: pytest.MonkeyPatch) -> list[ProviderConfig]:
    """Capture every ProviderConfig `wmo eval` builds, and answer with the fake provider."""
    seen: list[ProviderConfig] = []
    fake = FakeProvider()

    def record(config: ProviderConfig, **_kwargs: object) -> FakeProvider:
        seen.append(config)
        return fake

    monkeypatch.setattr("wmo.providers.provider_or_chain", record)
    monkeypatch.setattr("wmo.providers.get_provider", record)
    return seen


def _write_worker_role(root: Path, provider: str, model: str) -> None:
    settings = load_settings(root)
    settings.models.worker = ModelRole(provider=provider, model=model)
    save_settings(settings, root)


def test_eval_uses_configured_worker_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `wmo providers set` (step 1 of getting started) writes [models.worker]. eval used to score
    # against a hardcoded bedrock/claude-opus-4-8 regardless, so an OpenAI-only project got a
    # 0.000 fidelity at exit 0 from a provider it never configured.
    traces = _traces_file(tmp_path)
    _write_worker_role(tmp_path / ".wmo", "openai", "gpt-5.4-mini")
    monkeypatch.chdir(tmp_path)
    seen = _record_eval_providers(monkeypatch)

    result = runner.invoke(app, ["eval", traces, "--no-rag"])

    assert result.exit_code == 0, result.output
    assert {config.kind for config in seen} == {ProviderKind.OPENAI}
    assert {config.model for config in seen} == {"gpt-5.4-mini"}
    # The report is only comparable across runs on the same model, so eval names the backend.
    assert "scoring with openai (gpt-5.4-mini)" in result.output


def test_eval_provider_flag_overrides_the_configured_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    traces = _traces_file(tmp_path)
    _write_worker_role(tmp_path / ".wmo", "openai", "gpt-5.4-mini")
    monkeypatch.chdir(tmp_path)
    seen = _record_eval_providers(monkeypatch)

    result = runner.invoke(app, ["eval", traces, "--no-rag", "--provider", "bedrock"])

    assert result.exit_code == 0, result.output
    assert {config.kind for config in seen} == {ProviderKind.BEDROCK}
    # A --provider naming another backend drops the role's model: gpt-5.4-mini is not on bedrock.
    assert {config.model for config in seen} == {"us.anthropic.claude-opus-4-8"}


def test_eval_suite_run_records_the_resolved_worker_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The result JSON must name the model that produced the number, which with no flags is the
    # configured role rather than anything on the command line.
    examples_root = tmp_path / "examples"
    evals_dir = examples_root / "tiny-task" / "evals"
    evals_dir.mkdir(parents=True)
    (examples_root / "tiny-task" / "traces.otel.jsonl").write_text(
        Path(_traces_file(tmp_path)).read_text(encoding="utf-8"), encoding="utf-8"
    )
    (evals_dir / "default.toml").write_text(
        'files = ["../traces.otel.jsonl"]\ntrain_split = 0.5\n', encoding="utf-8"
    )
    _write_worker_role(tmp_path / ".wmo", "openai", "gpt-5.4-mini")
    monkeypatch.chdir(tmp_path)
    seen = _record_eval_providers(monkeypatch)
    results_root = tmp_path / ".wmo" / "evals"

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
    assert {config.kind for config in seen} == {ProviderKind.OPENAI}
    payload = json.loads(next(iter(results_root.glob("tiny-task/default/*.json"))).read_text())
    assert payload["config"]["provider"] == "openai"
    assert payload["config"]["model"] == "gpt-5.4-mini"


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

    results_root = tmp_path / ".wmo" / "evals"
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


def test_eval_out_parent_is_created_before_the_eval_runs(patched_provider, tmp_path) -> None:  # noqa: ANN001
    # The report is written last; a missing parent used to blow up with FileNotFoundError AFTER
    # the (paid) eval had finished, discarding it.
    destination = tmp_path / "nodir" / "deeper" / "report.json"

    result = runner.invoke(
        app, ["eval", _traces_file(tmp_path), "--no-rag", "--out", str(destination)]
    )

    assert result.exit_code == 0, result.output
    assert destination.exists()
    assert "Traceback" not in result.output


def test_eval_out_pointing_at_a_directory_is_a_usage_error(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(
        app, ["eval", _traces_file(tmp_path), "--no-rag", "--out", str(tmp_path)]
    )

    assert result.exit_code == 2  # usage error, not an IsADirectoryError traceback
    assert "is a directory" in _flat(result.output)


def test_eval_on_a_directory_is_a_usage_error(tmp_path) -> None:  # noqa: ANN001
    corpus_dir = tmp_path / "benchmark"
    corpus_dir.mkdir()

    result = runner.invoke(app, ["eval", str(corpus_dir)])

    assert result.exit_code == 2  # usage error, not an IsADirectoryError traceback
    flat = _flat(result.output)
    assert "is a directory" in flat
    # Rich may soft-wrap long Windows paths mid-token (`traces.otel.j` / `sonl`); strip spaces.
    assert "traces.otel.jsonl" in flat.replace(" ", "")  # names the file to pass instead


def test_eval_file_with_no_traces_fails_instead_of_scoring_zero(tmp_path) -> None:  # noqa: ANN001
    # A tasks.jsonl (or any non-OTel export) used to print a plausible
    # "OVERALL fidelity=0.000 over 0 held-out steps" scorecard and exit 0.
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text('{"task_id": "t1", "instruction": "do it"}\n', encoding="utf-8")

    result = runner.invoke(app, ["eval", str(tasks)])

    assert result.exit_code == 2, result.output
    flat = _flat(result.output)
    assert "no OTel GenAI traces" in flat
    assert "--mode closed-loop" in flat
    assert "OVERALL" not in flat


def test_eval_run_suite_with_no_traces_fails_instead_of_persisting_zero(tmp_path) -> None:  # noqa: ANN001
    # A suite result is durable: a zero-step run used to be saved and then resurface in
    # `wmo eval results` as a real 0.000 measurement.
    evals_dir = tmp_path / "examples" / "tiny-task" / "evals"
    evals_dir.mkdir(parents=True)
    (evals_dir.parent / "traces.otel.jsonl").write_text(
        '{"task_id": "t1", "instruction": "do it"}\n', encoding="utf-8"
    )
    (evals_dir / "default.toml").write_text('files = ["../traces.otel.jsonl"]\n', encoding="utf-8")
    results_root = tmp_path / ".wmo" / "evals"

    result = runner.invoke(
        app,
        [
            "eval",
            "run",
            "tiny-task",
            "--examples-root",
            str(tmp_path / "examples"),
            "--results-root",
            str(results_root),
        ],
    )

    assert result.exit_code == 2, result.output
    flat = _flat(result.output)
    assert "no OTel GenAI traces" in flat
    assert "OVERALL" not in flat
    assert not list(results_root.rglob("*.json"))  # nothing persisted


def test_eval_run_suite_listing_no_files_is_a_usage_error(tmp_path) -> None:  # noqa: ANN001
    evals_dir = tmp_path / "examples" / "tiny-task" / "evals"
    evals_dir.mkdir(parents=True)
    (evals_dir / "default.toml").write_text("files = []\n", encoding="utf-8")

    result = runner.invoke(
        app, ["eval", "run", "tiny-task", "--examples-root", str(tmp_path / "examples")]
    )

    assert result.exit_code == 2, result.output
    assert "lists no trace files" in _flat(result.output)


def test_eval_run_unknown_suite_is_a_usage_error(tmp_path) -> None:  # noqa: ANN001
    examples_root = tmp_path / "examples"
    examples_root.mkdir()

    for tokens in (["run", "nosuite"], ["grid", "nosuite"]):
        result = runner.invoke(app, ["eval", *tokens, "--examples-root", str(examples_root)])
        assert result.exit_code == 2, result.output  # usage error, not a ValueError traceback
        assert "unknown eval suite 'nosuite'" in _flat(result.output)


def test_eval_unknown_chain_is_a_usage_error(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)  # no .wmo/fallback.toml here

    result = runner.invoke(app, ["eval", _traces_file(tmp_path), "--chain", "nope"])

    assert result.exit_code == 2  # usage error, not a ValueError traceback
    assert "fallback.toml does not exist" in _flat(result.output)


def test_eval_rejects_closed_loop_only_flags_in_open_loop(tmp_path) -> None:  # noqa: ANN001
    # The README's closed-loop command minus `--mode closed-loop` used to silently drop every
    # closed-loop flag and run a different (paid) evaluation.
    result = runner.invoke(
        app,
        [
            "eval",
            _traces_file(tmp_path),
            "--harness",
            "nosuchharness",
            "--k",
            "7",
            "--harness-backend",
            "e2b",
        ],
    )

    assert result.exit_code == 2, result.output
    flat = _flat(result.output)
    assert "--k, --harness, --harness-backend" in flat
    assert "--mode closed-loop" in flat


def test_eval_threshold_belongs_to_the_agreement_flow(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(app, ["eval", _traces_file(tmp_path), "--threshold", "0.9"])

    assert result.exit_code == 2, result.output
    assert "wmo eval agreement" in _flat(result.output)


def _hide_viz_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make only matplotlib/seaborn look uninstalled, as on a core `pip install`."""
    real_find_spec = importlib.util.find_spec

    def find_spec(name: str, package: str | None = None) -> ModuleSpec | None:
        if name in cli_app_module._VIZ_MODULES:
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(cli_app_module.importlib.util, "find_spec", find_spec)


def test_eval_grid_flows_name_the_viz_extra_when_it_is_missing(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    # matplotlib/seaborn ship in the optional [viz] extra; the plotting import used to escape as a
    # raw ModuleNotFoundError that never named it (for `eval grid`, after the whole paid grid).
    _hide_viz_extra(monkeypatch)
    result_json = tmp_path / "grid.json"
    result_json.write_text("{}", encoding="utf-8")

    for tokens in (["grid-plot", str(result_json)], ["grid-heatmap", str(result_json)]):
        result = runner.invoke(app, ["eval", *tokens])
        assert result.exit_code == 2, result.output
        assert "uv sync --extra viz" in _flat(result.output)


def test_research_plot_commands_name_the_viz_extra_when_it_is_missing(monkeypatch) -> None:  # noqa: ANN001
    _hide_viz_extra(monkeypatch)

    for argv in (
        ["research", "plot-concurrency", "missing.json"],
        ["research", "plot-concurrency-combined", "a.json", "b.json"],
    ):
        result = runner.invoke(app, argv)
        assert result.exit_code == 2, result.output
        assert "uv sync --extra viz" in _flat(result.output)


def test_eval_help_lists_every_dispatched_flow() -> None:
    result = runner.invoke(app, ["eval", "--help"])

    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    # The grid family owns six of this command's options, so its tokens must be discoverable.
    for token in ("grid <suite>", "grid-plot", "grid-heatmap", "agreement"):
        assert token in flat
    # Suites moved out of examples/ long ago; the help must not send readers to the wrong dir.
    assert "packages/environment-capture" in flat


def test_build_then_list_shows_named_model(patched_provider, tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmo"
    _build(root, "tau2-airline", tmp_path)

    # The artifact lands under <root>/models/<name>/.
    assert (root / "models" / "tau2-airline" / "config.toml").exists()

    listed = runner.invoke(app, ["list", "--root", str(root)])
    assert listed.exit_code == 0, listed.output
    assert "tau2-airline" in listed.output


def test_list_empty_project_is_friendly(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(app, ["list", "--root", str(tmp_path / ".wmo")])
    assert result.exit_code == 0
    assert "no world models" in result.output
    # --root defaults to a cwd-relative `.wmo`, so "nothing built" and "wrong directory" read
    # the same unless the empty listing says where it looked. Asserted on the tail of the path
    # only: rich wraps a long tmp_path across lines.
    flat = _flat(result.output)
    assert "no world models built under" in flat
    assert str(Path(".wmo") / "models") in flat


def test_list_rejects_a_file_as_root(tmp_path) -> None:  # noqa: ANN001
    # `--root traces.jsonl` used to report a healthy empty project; a file can never hold models/.
    corpus = tmp_path / "traces.jsonl"
    corpus.write_text("{}\n", encoding="utf-8")
    result = runner.invoke(app, ["list", "--root", str(corpus)])
    assert result.exit_code == 2
    assert "is a file, not a project dir" in _flat(result.output)


def test_list_shows_an_unreadable_artifact_as_a_row(patched_provider, tmp_path) -> None:  # noqa: ANN001
    # One artifact this CLI cannot parse (a bundle from a newer CLI, a hand edit) used to
    # traceback the whole listing; the healthy models beside it must still be listed.
    root = tmp_path / ".wmo"
    _build(root, "alpha-healthy", tmp_path)
    broken = root / "models" / "zz-broken"
    broken.mkdir(parents=True)
    (broken / "config.toml").write_text("this is not toml =", encoding="utf-8")

    result = runner.invoke(app, ["list", "--root", str(root)])

    assert result.exit_code == 0, result.output
    assert result.exception is None  # a bad row, not an escaped TOMLDecodeError
    assert "alpha-healthy" in result.output
    assert "unreadable" in result.output
    assert "zz-broken" in result.output
    assert "is not valid TOML" in _flat(result.output)


def _write_broken_model(root: Path, name: str) -> None:
    (root / "models" / name).mkdir(parents=True)
    (root / "models" / name / "config.toml").write_text("this is not toml =", encoding="utf-8")


def test_the_model_picker_offers_only_readable_artifacts(
    patched_provider: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `list_info` now hands back a row for an artifact it could not read, so the picker has to
    # drop those rather than offer a choice that dead-ends the moment it is picked.
    root = tmp_path / ".wmo"
    _build(root, "alpha-healthy", tmp_path)
    _build(root, "beta-healthy", tmp_path)
    _write_broken_model(root, "zz-broken")
    offered: list[str] = []

    def fake_select_model(console: object, infos: list[ModelInfo]) -> str:
        offered.extend(info.name for info in infos)
        return infos[0].name

    monkeypatch.setattr(cli_app_module, "_console", SimpleNamespace(is_terminal=True))
    monkeypatch.setattr("wmo.cli.ui.select_model", fake_select_model)

    assert cli_app_module._resolve_name(WorldModelStore(root), None) == "alpha-healthy"
    assert offered == ["alpha-healthy", "beta-healthy"]


def test_the_model_picker_reports_when_nothing_is_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / ".wmo"
    for name in ("one-broken", "two-broken"):
        _write_broken_model(root, name)
    monkeypatch.setattr(cli_app_module, "_console", SimpleNamespace(is_terminal=True))

    with pytest.raises(typer.BadParameter, match="no readable world model"):
        cli_app_module._resolve_name(WorldModelStore(root), None)


def test_play_repl_steps_and_quits(patched_provider, tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmo"
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
    root = tmp_path / ".wmo"
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
    result = runner.invoke(app, ["build", "--no-interactive", "--root", str(tmp_path / ".wmo")])
    assert result.exit_code != 0


def _flat(output: str) -> str:
    """CliRunner output with rich's panel borders and line wrapping removed, for substrings."""
    return " ".join(output.replace("│", " ").split())


def _pull_trace(trace_id: str, *, usable: bool) -> Trace:
    """One single-step trace. `usable=False` makes it degenerate (empty observation)."""
    return Trace(
        trace_id=trace_id,
        source="otel-genai:vendor",
        steps=[
            Step(
                action=Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={"id": "u1"}),
                observation=Observation(content="found u1" if usable else ""),
            )
        ],
    )


def _many_traces_file(tmp_path, count: int) -> str:  # noqa: ANN001 - pytest fixture path
    """`count` copies of the single-trace export, each under its own trace id."""
    base = Path(_traces_file(tmp_path)).read_text(encoding="utf-8").splitlines()
    lines: list[str] = []
    for i in range(count):
        for line in base:
            lines.append(json.dumps({**json.loads(line), "traceId": f"{i:032d}"}))
    path = tmp_path / "many.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_build_with_a_name_but_no_trace_source_is_a_usage_error(tmp_path) -> None:  # noqa: ANN001
    # `wmo list` prints `wmo build --name <name>` as its empty-state hint, and every non-TTY
    # (CI, piped output) takes the scriptable path. It used to reach the ingest seam and raise
    # a raw ValueError; the guard only fired when --name was ALSO omitted.
    result = runner.invoke(
        app, ["build", "--name", "x", "--root", str(tmp_path / ".wmo"), "--no-interactive"]
    )
    assert result.exit_code == 2
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "provide --file <export> or --pull" in _flat(result.output)


def test_list_empty_state_names_a_trace_export(tmp_path) -> None:  # noqa: ANN001
    # The hint must be a runnable command: --name alone is a usage error (test above).
    result = runner.invoke(app, ["list", "--root", str(tmp_path / ".wmo")])
    assert result.exit_code == 0
    assert "--file" in result.output


def test_build_missing_trace_file_is_a_usage_error(tmp_path) -> None:  # noqa: ANN001
    # A typo'd --file used to die with a raw FileNotFoundError from the adapter, and only after
    # the provider ping had already run.
    result = runner.invoke(
        app,
        [
            "build",
            "--name",
            "x",
            "--file",
            str(tmp_path / "nope.jsonl"),
            "--root",
            str(tmp_path / ".wmo"),
            "--no-interactive",
        ],
    )
    assert result.exit_code == 2
    assert not isinstance(result.exception, FileNotFoundError)
    assert "trace file not found" in _flat(result.output)
    # Rejected at the argument boundary: no provider was pinged.
    assert "verifying" not in result.output


def test_build_rejects_a_directory_as_the_trace_file(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(
        app,
        [
            "build",
            "--name",
            "x",
            "--file",
            str(tmp_path),
            "--root",
            str(tmp_path / ".wmo"),
            "--no-interactive",
        ],
    )
    assert result.exit_code == 2
    assert not isinstance(result.exception, IsADirectoryError)
    assert "not a directory" in _flat(result.output)


def test_build_rejects_the_postgres_source_and_names_wmo_ingest(tmp_path) -> None:  # noqa: ANN001
    # `postgres` passes the adapter-name validator but can never work here: build has no
    # --dsn/--table (those live on `wmo ingest`), so it must be rejected at the boundary.
    result = runner.invoke(
        app,
        [
            "build",
            "--name",
            "x",
            "--file",
            _traces_file(tmp_path),
            "--source",
            "postgres",
            "--root",
            str(tmp_path / ".wmo"),
            "--no-interactive",
        ],
    )
    assert result.exit_code == 2
    assert not isinstance(result.exception, ValueError)
    flat = _flat(result.output)
    assert "wmo ingest --source postgres --dsn <dsn> --table <table>" in flat
    assert "--dsn" not in _flat(runner.invoke(app, ["build", "--help"]).output)


def test_build_wrong_source_names_the_detected_format(patched_provider, tmp_path) -> None:  # noqa: ANN001
    # A chat-json export under the silent `--source otel-genai` default ingested nothing and
    # raised ValueError('no traces ingested; nothing to build') as a traceback.
    chat = tmp_path / "chat.json"
    chat.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}), encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "build",
            "--name",
            "x",
            "--file",
            str(chat),
            "--root",
            str(tmp_path / ".wmo"),
            "--provider",
            "bedrock",
            "--fidelity",
            "low",
            "--no-interactive",
        ],
    )
    assert result.exit_code == 2
    assert not isinstance(result.exception, ValueError)
    flat = _flat(result.output)
    assert "--source otel-genai" in flat
    assert "it looks like chat-json" in flat
    # The path itself is wrapped by rich, so assert on the command, not the rendered path.
    assert "wmo ingest --file" in flat


def test_build_empty_trace_file_names_source_and_ingest(patched_provider, tmp_path) -> None:  # noqa: ANN001
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "build",
            "--name",
            "x",
            "--file",
            str(empty),
            "--root",
            str(tmp_path / ".wmo"),
            "--provider",
            "bedrock",
            "--fidelity",
            "low",
            "--no-interactive",
        ],
    )
    assert result.exit_code == 2
    assert not isinstance(result.exception, ValueError)
    flat = _flat(result.output)
    assert "--source otel-genai" in flat
    assert "wmo ingest --file" in flat


def test_build_limit_caps_a_file_corpus(patched_provider, tmp_path) -> None:  # noqa: ANN001
    # --limit was wired only into the VendorPull branch, so it was silently ignored for --file
    # builds while `wmo ingest --limit` capped both transports.
    from wmo.config.card import load_card

    root = tmp_path / ".wmo"
    result = runner.invoke(
        app,
        [
            "build",
            "--name",
            "capped",
            "--file",
            _many_traces_file(tmp_path, 6),
            "--limit",
            "2",
            "--root",
            str(root),
            "--provider",
            "bedrock",
            "--fidelity",
            "low",
        ],
    )
    assert result.exit_code == 0, result.output
    card = load_card(root / "models" / "capped")
    assert card is not None
    assert card.corpus.traces == 2


def test_build_pull_limit_is_a_fetch_cap_applied_once(
    patched_provider,  # noqa: ANN001 - pytest fixture
    monkeypatch,  # noqa: ANN001 - pytest fixture
    tmp_path,  # noqa: ANN001 - pytest fixture path
) -> None:
    """A pull spends `--limit` vendor-side, so `--drop-degenerate` can leave fewer than N.

    `wmo.ingest.base.from_vendor` slices to `pull.limit` before `build` ever sees the corpus,
    so re-applying the same cap after the degenerate filter cannot restore the dropped traces —
    it would only read as a promise of N usable traces that this transport cannot keep. Pinning
    both halves: the adapter receives the cap, and `build` is not handed it a second time.
    """
    from wmo.config.card import load_card

    seen: list[VendorPull] = []
    passed_to_build: dict[str, object] = {}

    class _CappingAdapter:
        """Mimics `base.from_vendor`: alternating junk/usable traces, sliced at `pull.limit`."""

        name = "otel-genai"

        def from_vendor(self, pull: VendorPull) -> list[Trace]:
            seen.append(pull)
            traces = [_pull_trace(f"{i:032d}", usable=bool(i % 2)) for i in range(6)]
            return traces if pull.limit is None else traces[: pull.limit]

    # `wmo.engine.build` is shadowed by the `build` function re-exported from
    # `wmo.engine.__init__`, so attribute / `import wmo.engine.build` resolve the function.
    # Reach the submodule only through importlib / sys.modules.
    import importlib

    engine_build = importlib.import_module("wmo.engine.build")
    real_run_build = engine_build.build

    def _spy(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202 - passthrough spy
        passed_to_build.update(kwargs)
        return real_run_build(*args, **kwargs)

    monkeypatch.setattr(engine_build, "get_adapter", lambda name: _CappingAdapter())
    monkeypatch.setattr(engine_build, "build", _spy)
    root = tmp_path / ".wmo"
    result = runner.invoke(
        app,
        [
            "build",
            "--name",
            "pulled",
            "--pull",
            "--limit",
            "4",
            "--drop-degenerate",
            "--root",
            str(root),
            "--provider",
            "bedrock",
            "--fidelity",
            "low",
        ],
    )
    assert result.exit_code == 0, result.output
    assert [p.limit for p in seen] == [4]  # spent at fetch…
    assert passed_to_build["limit"] is None  # …and not a second time after the filter
    card = load_card(root / "models" / "pulled")
    assert card is not None
    assert card.corpus.traces == 2  # 4 fetched, 2 of them degenerate; the cap cannot refill


def test_build_rejects_a_limit_below_one(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(
        app,
        [
            "build",
            "--name",
            "x",
            "--file",
            _traces_file(tmp_path),
            "--limit",
            "0",
            "--root",
            str(tmp_path / ".wmo"),
            "--no-interactive",
        ],
    )
    assert result.exit_code == 2
    assert "--limit must be at least 1" in _flat(result.output)


def test_build_unknown_chain_is_a_usage_error(tmp_path) -> None:  # noqa: ANN001
    """`--chain` with no `.wmo/fallback.toml` must say how to create the file, not traceback.

    Deliberately no `patched_provider`: that fixture stubs `providers.provider_or_chain`, which
    is the seam under test. Chain resolution runs before the provider ping, so nothing here
    reaches the network (`wmo/conftest.py` points the chain path at an empty tmp dir).
    """
    result = runner.invoke(
        app,
        [
            "build",
            "--name",
            "x",
            "--file",
            _traces_file(tmp_path),
            "--chain",
            "fast",
            "--root",
            str(tmp_path / ".wmo"),
            "--provider",
            "bedrock",
            "--fidelity",
            "low",
            "--no-interactive",
        ],
    )
    assert result.exit_code == 2
    assert not isinstance(result.exception, ValueError)
    flat = _flat(result.output)
    assert "chain 'fast' requested but" in flat
    assert "fallback.toml" in flat
    assert "[[chain.<name>]] rung tables" in flat
    assert "docs/reference/failover.md" in flat


def test_build_model_default_follows_the_provider(patched_provider, tmp_path) -> None:  # noqa: ANN001
    # --provider openai with no --model used to persist the Anthropic id `claude-opus-4-8`.
    root = tmp_path / ".wmo"
    result = runner.invoke(
        app,
        [
            "build",
            "--name",
            "oa",
            "--file",
            _traces_file(tmp_path),
            "--provider",
            "openai",
            "--fidelity",
            "low",
            "--root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.output
    config = load_config(root / "models" / "oa")
    assert config.serve_provider is ProviderKind.OPENAI
    assert config.serve_provider_config().model_type == "gpt-5.5"


def test_build_requires_a_model_for_a_provider_without_a_default(tmp_path) -> None:  # noqa: ANN001
    # openrouter/tinker/openai_responses have no curated model list: ask rather than guess,
    # matching `wmo providers set`'s scriptable contract.
    result = runner.invoke(
        app,
        [
            "build",
            "--name",
            "x",
            "--file",
            _traces_file(tmp_path),
            "--provider",
            "openrouter",
            "--root",
            str(tmp_path / ".wmo"),
            "--no-interactive",
        ],
    )
    assert result.exit_code == 2
    assert "--provider openrouter has no default serve model" in _flat(result.output)


def test_build_aborts_when_provider_sdk_missing(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    """A missing SDK must abort the build before any rollouts, with the `uv sync` extra hint.

    Regression: previously the ModuleNotFoundError was swallowed inside GEPA and the build
    "succeeded" with a useless held-out-0.0 model.
    """
    from wmo.providers.base import VerifyResult

    monkeypatch.setattr(
        "wmo.providers.verify_all",
        lambda configs: [
            VerifyResult(
                ok=False,
                kind=configs[0].kind,
                model=configs[0].model,
                detail="No module named 'boto3'",
            )
        ],
    )
    root = tmp_path / ".wmo"
    result = runner.invoke(
        app, ["build", "--name", "x", "--file", _traces_file(tmp_path), "--root", str(root)]
    )
    assert result.exit_code == 1
    assert "run `uv sync` to install the provider SDKs" in result.output
    # Aborted before building: no artifact written.
    assert not (root / "models" / "x" / "config.toml").exists()


def test_play_unknown_model_errors(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(app, ["play", "--name", "nope", "--root", str(tmp_path / ".wmo")])
    assert result.exit_code != 0
    # A clean usage error, not an uncaught FileNotFoundError traceback.
    assert not isinstance(result.exception, FileNotFoundError)
    assert "nope" in result.output


def test_demo_unknown_model_is_clean_error(patched_provider, tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmo"
    _build(root, "airline", tmp_path)
    result = runner.invoke(app, ["demo", "--name", "ghost", "--root", str(root)])
    assert result.exit_code != 0
    # Resolved through _load_model -> _resolve_name; must surface as a usage error, not a traceback.
    assert not isinstance(result.exception, (FileNotFoundError, ValueError))


def test_providers_verify_unknown_model_is_clean_error(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(
        app, ["providers", "verify", "--name", "ghost", "--root", str(tmp_path / ".wmo")]
    )
    assert result.exit_code != 0
    assert not isinstance(result.exception, FileNotFoundError)


# Hand-editing `.wmo/settings.toml` is documented (docs/reference/closed_loop.md), and a file
# written by an older CLI outlives an upgrade, so every command that reads it has to fail as a
# usage error naming the file, never as a tomllib/pydantic traceback.
_BROKEN_SETTINGS = pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ('[models.worker\nprovider = "openai"\n', "is not valid TOML"),
        ('[models]\nworker = "openai"\n', "does not match the current settings schema"),
    ],
    ids=["malformed-toml", "schema-invalid"],
)


def _write_settings(tmp_path: Path, payload: str) -> Path:
    root = tmp_path / ".wmo"
    root.mkdir(parents=True, exist_ok=True)
    (root / "settings.toml").write_text(payload, encoding="utf-8")
    return root


@_BROKEN_SETTINGS
@pytest.mark.parametrize(
    "argv",
    [
        ["providers", "verify"],
        ["config", "telemetry", "status"],
        ["config", "telemetry", "enable"],
        ["providers", "set", "--provider", "openai", "--model", "gpt-5.4"],
    ],
    ids=["verify", "telemetry-status", "telemetry-enable", "providers-set"],
)
def test_broken_settings_is_a_usage_error_not_a_traceback(
    tmp_path: Path, payload: str, expected: str, argv: list[str]
) -> None:
    root = _write_settings(tmp_path, payload)

    result = runner.invoke(app, [*argv, "--root", str(root)])

    assert result.exit_code == 2
    assert not isinstance(result.exception, ValueError)  # BadParameter, not a leaked loader raise
    flat = _flat(result.output)
    assert expected in flat
    assert "settings.toml" in flat
    assert "delete it and re-run `wmo providers set`" in flat


@_BROKEN_SETTINGS
def test_providers_set_rejects_a_bad_provider_before_reading_settings(
    tmp_path: Path, payload: str, expected: str
) -> None:
    # The caller's own argument is wrong too; the error must be about the argument they typed.
    root = _write_settings(tmp_path, payload)

    result = runner.invoke(
        app, ["providers", "set", "--provider", "bogus", "--model", "x", "--root", str(root)]
    )

    assert result.exit_code == 2
    assert "unknown provider 'bogus'" in _flat(result.output)
    assert expected not in _flat(result.output)


def test_providers_verify_unreadable_model_config_is_clean_error(tmp_path: Path) -> None:
    # An artifact copied in by hand (or extracted from a bundle a newer CLI wrote) can hold a
    # config.toml this CLI cannot parse; the command whose job is reporting configuration
    # problems must report that one too.
    broken = tmp_path / ".wmo" / "models" / "foo"
    broken.mkdir(parents=True)
    (broken / "config.toml").write_text("this is not toml [[[\n", encoding="utf-8")

    result = runner.invoke(app, ["providers", "verify", "--root", str(tmp_path / ".wmo")])

    assert result.exit_code == 2
    assert not isinstance(result.exception, ValueError)
    flat = _flat(result.output)
    # Path separators and rich soft-wraps vary by OS; match the durable pieces.
    assert "foo" in flat and "config.toml" in flat.replace(" ", "")
    assert "is not valid TOML" in flat
    assert "re-run `wmo build`" in flat


def _record_verify_all(monkeypatch: pytest.MonkeyPatch, pinged: list[ProviderConfig]) -> None:
    """Record exactly which providers `providers verify` decided to ping, and report them ok."""

    def fake_verify_all(configs: list[ProviderConfig]) -> list[VerifyResult]:
        pinged.extend(configs)
        return [VerifyResult(ok=True, kind=c.kind, model=c.model) for c in configs]

    monkeypatch.setattr("wmo.providers.verify_all", fake_verify_all)


def test_providers_verify_nothing_configured_is_actionable(tmp_path: Path) -> None:
    # Nothing to check at all is a usage problem, not a pass: say which command fixes it and
    # exit non-zero so a setup script does not read silence as "credentials are fine".
    result = runner.invoke(app, ["providers", "verify", "--root", str(tmp_path / ".wmo")])
    assert result.exit_code == 1
    assert "nothing configured" in result.output
    assert "wmo providers set" in result.output


def test_providers_verify_without_a_world_model_checks_settings_roles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The repro this fixes: verifying credentials is what you do BEFORE `wmo build` (which
    # aborts outright on bad ones), so an unbuilt project must still check what it has.
    root = tmp_path / ".wmo"
    settings = load_settings(root)
    settings.models.worker = ModelRole(provider="openai", model="gpt-5.4-mini")
    settings.models.judge = ModelRole(provider="bedrock", model="claude-opus-4-8")
    save_settings(settings, root)
    pinged: list[ProviderConfig] = []
    _record_verify_all(monkeypatch, pinged)

    result = runner.invoke(app, ["providers", "verify", "--root", str(root)])

    assert result.exit_code == 0, result.output
    assert not (root / "models").exists()
    # Both configured roles are pinged, the bedrock one at its runtime id (settings hold the
    # canonical type), and each line names the role that asked for it.
    assert [(c.kind, c.model) for c in pinged] == [
        (ProviderKind.OPENAI, "gpt-5.4-mini"),
        (ProviderKind.BEDROCK, "us.anthropic.claude-opus-4-8"),
    ]
    assert "ok openai (gpt-5.4-mini) (models.worker)" in result.output
    assert "models.judge" in result.output
    # The embed path belongs to a built model: skipped with a note, not fatal.
    assert "embed path: skipped" in result.output


def test_providers_verify_reports_a_role_failure_with_its_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / ".wmo"
    settings = load_settings(root)
    settings.models.worker = ModelRole(provider="bedrock", model="claude-opus-4-8")
    save_settings(settings, root)
    monkeypatch.setattr(
        "wmo.providers.verify_all",
        lambda configs: [
            VerifyResult(
                ok=False,
                kind=c.kind,
                model=c.model,
                # Rich markup in raw provider error text must not be interpreted.
                detail="denied [foo]",
            )
            for c in configs
        ],
    )

    result = runner.invoke(app, ["providers", "verify", "--root", str(root)])

    assert result.exit_code == 0, result.output
    assert "fail bedrock" in result.output
    assert "[foo]" in result.output
    assert "AWS_ACCESS_KEY_ID" in result.output


def test_providers_verify_missing_optional_sdk_points_at_the_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # tinker's SDK is an optional extra, and its ImportError text replaces the "No module named"
    # wording the hint used to key on, so the hint said "check your credentials" on a failure
    # that has nothing to do with credentials.
    root = tmp_path / ".wmo"
    settings = load_settings(root)
    settings.models.worker = ModelRole(provider="tinker", model="Qwen/Qwen3-8B")
    save_settings(settings, root)
    monkeypatch.setattr(
        "wmo.providers.verify_all",
        lambda configs: [
            VerifyResult(ok=False, kind=c.kind, model=c.model, detail=_MISSING_TINKER_EXTRA)
            for c in configs
        ],
    )

    result = runner.invoke(app, ["providers", "verify", "--root", str(root)])

    flat = _flat(result.output)
    # pip is the documented install path, so the extra must be reachable without a checkout,
    # and the `[distill]` must survive rich markup rather than being read as a style tag.
    assert "pip install 'world-model-optimizer[distill]'" in flat
    assert "credentials are set" not in flat


def test_providers_verify_reports_built_model_provider(patched_provider, tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmo"
    _build(root, "airline", tmp_path)
    result = runner.invoke(app, ["providers", "verify", "--root", str(root)])
    assert result.exit_code == 0, result.output
    # The bedrock provider configured at build time shows up in the verify report.
    assert "bedrock" in result.output


def test_providers_verify_checks_a_built_model_embed_path(
    patched_provider: None, tmp_path: Path
) -> None:
    # A provider-backed embedder is verified alongside the completion provider; that check is
    # the half a world model is genuinely needed for.
    root = tmp_path / ".wmo"
    result = runner.invoke(
        app,
        [
            "build",
            "--name",
            "airline",
            "--file",
            _traces_file(tmp_path),
            "--root",
            str(root),
            "--provider",
            "bedrock",
            "--embed-provider",
            "bedrock",
            "--embed-model",
            "amazon.titan-embed-text-v2:0",
            "--fidelity",
            "low",
        ],
    )
    assert result.exit_code == 0, result.output

    result = runner.invoke(app, ["providers", "verify", "--root", str(root)])

    assert result.exit_code == 0, result.output
    assert "embed:bedrock (amazon.titan-embed-text-v2:0)" in result.output
    assert "embed path: skipped" not in result.output


def test_providers_verify_pings_a_role_shared_with_a_built_model_once(
    patched_provider: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Dedup spans both sources, so pointing the worker role at the model you already built does
    # not bill a second ping, and the built artifact's own config is the one exercised.
    root = tmp_path / ".wmo"
    _build(root, "airline", tmp_path)
    built = load_config(str(root / "models" / "airline")).providers[0]
    settings = load_settings(root)
    settings.models.worker = ModelRole(provider=built.kind.value, model=built.model)
    save_settings(settings, root)
    pinged: list[ProviderConfig] = []
    _record_verify_all(monkeypatch, pinged)

    result = runner.invoke(app, ["providers", "verify", "--root", str(root)])

    assert result.exit_code == 0, result.output
    assert [(c.kind, c.model) for c in pinged] == [(built.kind, built.model)]
    assert "(airline, models.worker)" in result.output


def test_providers_verify_does_not_collapse_two_regions_of_one_model(
    patched_provider: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Same kind and model, different region: two different backends that fail independently.
    # Collapsing them would ping one and report the OTHER as verified, which is a false pass on
    # exactly the credential question this command exists to answer.
    root = tmp_path / ".wmo"
    _build(root, "airline", tmp_path)
    built = load_config(str(root / "models" / "airline")).providers[0]
    assert built.region is None
    settings = load_settings(root)
    settings.models.worker = ModelRole(
        provider=built.kind.value, model=built.model, region="eu-west-1"
    )
    save_settings(settings, root)
    pinged: list[ProviderConfig] = []
    _record_verify_all(monkeypatch, pinged)

    result = runner.invoke(app, ["providers", "verify", "--root", str(root)])

    assert result.exit_code == 0, result.output
    assert [c.region for c in pinged] == [None, "eu-west-1"]


def test_providers_verify_does_not_collapse_two_azure_deployments(
    patched_provider: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Two roles on one Azure model, each behind its own operator-named deployment and endpoint.
    root = tmp_path / ".wmo"
    settings = load_settings(root)
    settings.models.worker = ModelRole(
        provider="azure", model="gpt-5.5", endpoint="https://a.example", deployment="dep-a"
    )
    settings.models.judge = ModelRole(
        provider="azure", model="gpt-5.5", endpoint="https://b.example", deployment="dep-b"
    )
    save_settings(settings, root)
    pinged: list[ProviderConfig] = []
    _record_verify_all(monkeypatch, pinged)

    result = runner.invoke(app, ["providers", "verify", "--root", str(root)])

    assert result.exit_code == 0, result.output
    assert [(c.endpoint, c.deployment) for c in pinged] == [
        ("https://a.example", "dep-a"),
        ("https://b.example", "dep-b"),
    ]


def test_providers_verify_checks_both_embed_models_on_one_backend(
    patched_provider: None, tmp_path: Path
) -> None:
    # Two world models sharing a completion backend but embedding through different models: one
    # completion ping, and BOTH embed paths checked (embed_model is what the embed call sends).
    root = tmp_path / ".wmo"
    for model_name, embed_model in (
        ("a", "amazon.titan-embed-text-v1"),
        ("b", "amazon.titan-embed-text-v2:0"),
    ):
        built = runner.invoke(
            app,
            [
                "build",
                "--name",
                model_name,
                "--file",
                _traces_file(tmp_path),
                "--root",
                str(root),
                "--provider",
                "bedrock",
                "--embed-provider",
                "bedrock",
                "--embed-model",
                embed_model,
                "--fidelity",
                "low",
            ],
        )
        assert built.exit_code == 0, built.output

    result = runner.invoke(app, ["providers", "verify", "--root", str(root)])

    assert result.exit_code == 0, result.output
    assert "embed:bedrock (amazon.titan-embed-text-v1)" in result.output
    assert "embed:bedrock (amazon.titan-embed-text-v2:0)" in result.output
    # The shared completion provider is still pinged once, under both model names.
    assert result.output.count("ok bedrock (") == 1


def test_providers_verify_name_scopes_the_report_to_one_world_model(
    patched_provider: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `--name` answers "is THIS model's provider reachable?"; pulling the project's roles in
    # would bill for a question the caller did not ask.
    root = tmp_path / ".wmo"
    _build(root, "airline", tmp_path)
    settings = load_settings(root)
    settings.models.worker = ModelRole(provider="openai", model="gpt-5.4-mini")
    save_settings(settings, root)
    pinged: list[ProviderConfig] = []
    _record_verify_all(monkeypatch, pinged)

    result = runner.invoke(app, ["providers", "verify", "--name", "airline", "--root", str(root)])

    assert result.exit_code == 0, result.output
    assert [c.kind for c in pinged] == [ProviderKind.BEDROCK]
    assert "models.worker" not in result.output


def test_research_concurrency_uses_the_configured_worker_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The environment LLM is the same worker-role call every other command makes; it used to be
    # pinned to bedrock/claude-opus-4-8 whatever the project configured.
    examples_root = tmp_path / "examples"
    evals_dir = examples_root / "tiny-task" / "evals"
    evals_dir.mkdir(parents=True)
    (examples_root / "tiny-task" / "traces.otel.jsonl").write_text(
        Path(_traces_file(tmp_path)).read_text(encoding="utf-8"), encoding="utf-8"
    )
    (evals_dir / "default.toml").write_text(
        'files = ["../traces.otel.jsonl"]\ntrain_split = 0.5\n', encoding="utf-8"
    )
    _write_worker_role(tmp_path / ".wmo", "openai", "gpt-5.4-mini")
    monkeypatch.chdir(tmp_path)
    # get_provider is the identity here, so calling the runner's factory yields its config.
    built: list[ProviderConfig] = []
    monkeypatch.setattr("wmo.providers.get_provider", lambda config: config)
    monkeypatch.setattr(
        "wmo.research.concurrency_run.build_world_runner",
        lambda factory, prompt, demos, selected: built.append(factory()),
    )
    monkeypatch.setattr(
        "wmo.research.run_concurrency_scaling",
        lambda *a, **kw: SimpleNamespace(benchmark="", best_speedup=lambda: None),
    )

    result = runner.invoke(
        app,
        [
            "research",
            "concurrency",
            "tiny-task",
            "--examples-root",
            str(examples_root),
            "--side",
            "world",
            "--scenarios",
            "1",
            "--levels",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert built[0].kind is ProviderKind.OPENAI
    assert built[0].model == "gpt-5.4-mini"


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


def _framed(output: str) -> str:
    """One flat line of a rich-framed message: the box wraps and pads what it renders."""
    return " ".join(output.replace("│", " ").split())


def test_research_concurrency_unknown_suite_is_clean_error(tmp_path) -> None:  # noqa: ANN001
    # The suite argument must fail like every other argument on this command: a framed usage
    # error naming the next command, not the raw ValueError traceback out of resolve_eval_suite.
    result = runner.invoke(
        app,
        ["research", "concurrency", "nosuchsuite", "--examples-root", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert not isinstance(result.exception, ValueError)
    flat = _framed(result.output)
    assert "unknown eval suite 'nosuchsuite'" in flat
    assert "`wmo eval list` prints the suites" in flat


def test_scenarios_build_missing_file_is_clean_error(tmp_path) -> None:  # noqa: ANN001
    # A typo in --file is the likeliest first-run mistake; it must not be a FileNotFoundError.
    result = runner.invoke(app, ["scenarios", "build", "--file", str(tmp_path / "nope.jsonl")])
    assert result.exit_code != 0
    assert not isinstance(result.exception, FileNotFoundError)
    flat = _framed(result.output)
    assert "does not exist" in flat
    assert "`wmo download <benchmark>`" in flat


def test_scenarios_build_directory_file_is_clean_error(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(app, ["scenarios", "build", "--file", str(tmp_path)])
    assert result.exit_code != 0
    assert not isinstance(result.exception, IsADirectoryError)
    assert "is a directory" in _framed(result.output)


def test_scenarios_build_help_documents_the_accepted_embed_provider() -> None:
    # The help must name values the command accepts: EmbedderKind spells Azure "azure".
    result = runner.invoke(app, ["scenarios", "build", "--help"])
    assert result.exit_code == 0
    flat = _framed(result.output)
    documented = flat[flat.index("Facet embedder") : flat.index("--embed-model")]
    for kind in EmbedderKind:
        assert kind.value in documented, kind
    assert "azure_openai" not in documented


def test_scenarios_verify_missing_scenario_set_is_clean_error(tmp_path) -> None:  # noqa: ANN001
    corpus = tmp_path / "traces.jsonl"
    corpus.write_text("", encoding="utf-8")
    result = runner.invoke(
        app,
        ["scenarios", "verify", str(tmp_path / "nope.json"), "--file", str(corpus)],
    )
    assert result.exit_code != 0
    assert not isinstance(result.exception, FileNotFoundError)
    flat = _framed(result.output)
    assert "does not exist" in flat
    assert "`wmo scenarios build --file <traces.jsonl> --out" in flat


def test_scenarios_verify_missing_corpus_is_clean_error(tmp_path) -> None:  # noqa: ANN001
    scenarios_file = tmp_path / "scenarios.json"
    scenarios_file.write_text('{"scenarios": []}', encoding="utf-8")
    result = runner.invoke(
        app,
        ["scenarios", "verify", str(scenarios_file), "--file", str(tmp_path / "nope.jsonl")],
    )
    assert result.exit_code != 0
    assert not isinstance(result.exception, FileNotFoundError)
    flat = _framed(result.output)
    assert "--file" in flat and "does not exist" in flat


def test_scenarios_verify_malformed_scenario_set_is_clean_error(tmp_path) -> None:  # noqa: ANN001
    # Pydantic's ValidationError points at pydantic's docs; the user needs the command that
    # writes this artifact instead.
    scenarios_file = tmp_path / "scenarios.json"
    scenarios_file.write_text("not json\n", encoding="utf-8")
    corpus = tmp_path / "traces.jsonl"
    corpus.write_text("", encoding="utf-8")
    result = runner.invoke(app, ["scenarios", "verify", str(scenarios_file), "--file", str(corpus)])
    assert result.exit_code != 0
    assert not isinstance(result.exception, ValidationError)
    flat = _framed(result.output)
    assert "is not a scenario set written by `wmo scenarios build`" in flat
    assert "errors.pydantic.dev" not in flat


def test_scenarios_verify_non_utf8_scenario_set_is_clean_error(tmp_path) -> None:  # noqa: ANN001
    # Regression (Greptile P1): `ScenarioSet.load` reads with encoding="utf-8", so binary bytes
    # raise UnicodeDecodeError *before* pydantic and slipped past the ValidationError handler.
    scenarios_file = tmp_path / "scenarios.json"
    scenarios_file.write_bytes(b"\xff\xfe\x00binary")
    corpus = tmp_path / "traces.jsonl"
    corpus.write_text("", encoding="utf-8")
    result = runner.invoke(app, ["scenarios", "verify", str(scenarios_file), "--file", str(corpus)])
    assert result.exit_code != 0
    assert not isinstance(result.exception, UnicodeDecodeError)
    assert "is not UTF-8 text" in _framed(result.output)


def test_scenarios_build_non_utf8_corpus_is_clean_error(tmp_path) -> None:  # noqa: ANN001
    # Regression (Greptile P1): a path that exists still fails inside the adapter's read.
    corpus = tmp_path / "traces.jsonl"
    corpus.write_bytes(b"\xff\xfe\x00binary")
    result = runner.invoke(app, ["scenarios", "build", "--file", str(corpus)])
    assert result.exit_code != 0
    assert not isinstance(result.exception, UnicodeDecodeError)
    flat = _framed(result.output)
    assert "--file" in flat and "is not UTF-8 text" in flat


@pytest.mark.skipif(sys.platform == "win32", reason="chmod(0) does not revoke read on Windows")
def test_scenarios_build_unreadable_corpus_is_clean_error(tmp_path) -> None:  # noqa: ANN001
    # Regression (Greptile P1): chmod-000 raised PermissionError out of Path.read_text.
    corpus = tmp_path / "traces.jsonl"
    corpus.write_text("", encoding="utf-8")
    corpus.chmod(0)
    try:
        result = runner.invoke(app, ["scenarios", "build", "--file", str(corpus)])
    finally:
        corpus.chmod(0o644)
    assert result.exit_code != 0
    assert not isinstance(result.exception, PermissionError)
    flat = _framed(result.output)
    assert "could not be read" in flat
    assert "shows its owner and mode" in flat


def test_research_concurrency_malformed_suite_file_omits_the_listing_hint(tmp_path) -> None:  # noqa: ANN001
    # Regression (Greptile P2): a direct `.toml` selector that exists resolves past name lookup
    # and fails on its own contents, so `wmo eval list` would point away from the broken file.
    suite_file = tmp_path / "broken.toml"
    suite_file.write_text("this is not = toml [[[\n", encoding="utf-8")
    result = runner.invoke(app, ["research", "concurrency", str(suite_file)])
    assert result.exit_code != 0
    assert not isinstance(result.exception, ValueError)
    flat = _framed(result.output)
    assert "is not valid TOML" in flat
    assert "wmo eval list" not in flat


def test_scenario_role_llms_resolve_from_settings(monkeypatch) -> None:  # noqa: ANN001
    from wmo.config.settings import ModelRole, ModelsSettings, ProjectSettings

    made: list[ProviderConfig] = []

    def fake_get_provider(config: ProviderConfig) -> ProviderConfig:
        made.append(config)
        return config  # identity provider: assertions read the config directly

    monkeypatch.setattr("wmo.providers.get_provider", fake_get_provider)
    monkeypatch.setattr(
        cli_app_module,
        "load_settings_or_abort",
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
    from wmo.config.settings import ProjectSettings

    monkeypatch.setattr("wmo.providers.get_provider", lambda config: config)
    monkeypatch.setattr(cli_app_module, "load_settings_or_abort", lambda: ProjectSettings())
    summary, worker, judge = cli_app_module._scenario_role_llms("bedrock", "some-model", None)
    assert summary is worker
    assert worker is judge
    assert cast(ProviderConfig, worker).model == "some-model"


def test_scenario_role_llms_model_flag_keeps_the_configured_provider(monkeypatch) -> None:  # noqa: ANN001
    # Half a flag pair used to complete from bedrock, so `--model gpt-5.5` on an OpenAI project
    # asked bedrock for an OpenAI model id.
    from wmo.config.settings import ModelRole, ModelsSettings, ProjectSettings

    monkeypatch.setattr("wmo.providers.get_provider", lambda config: config)
    monkeypatch.setattr(
        cli_app_module,
        "load_settings_or_abort",
        lambda: ProjectSettings(
            models=ModelsSettings(worker=ModelRole(provider="openai", model="gpt-5.4-mini"))
        ),
    )
    _summary, worker, _judge = cli_app_module._scenario_role_llms(None, "gpt-5.5", None)
    assert cast(ProviderConfig, worker).kind is ProviderKind.OPENAI
    assert cast(ProviderConfig, worker).model == "gpt-5.5"


def test_scenario_role_llms_default_when_nothing_configured(monkeypatch) -> None:  # noqa: ANN001
    from wmo.config.settings import ProjectSettings

    monkeypatch.setattr("wmo.providers.get_provider", lambda config: config)
    monkeypatch.setattr(cli_app_module, "load_settings_or_abort", lambda: ProjectSettings())
    summary, worker, judge = cli_app_module._scenario_role_llms(None, None, None)
    assert summary is worker
    assert worker is judge
    assert cast(ProviderConfig, worker).model == "us.anthropic.claude-opus-4-8"


def test_worker_role_provider_config_falls_back_to_bedrock(monkeypatch) -> None:  # noqa: ANN001
    from wmo.config.settings import ProjectSettings

    monkeypatch.setattr(cli_app_module, "load_settings_or_abort", lambda: ProjectSettings())
    config = cli_app_module._worker_role_provider_config(None, None, None)
    assert config.kind is ProviderKind.BEDROCK
    assert config.model == "us.anthropic.claude-opus-4-8"


def _azure_worker_settings(monkeypatch: pytest.MonkeyPatch, deployment: str | None) -> None:
    from wmo.config.settings import ModelRole, ModelsSettings, ProjectSettings

    monkeypatch.setattr(
        cli_app_module,
        "load_settings_or_abort",
        lambda: ProjectSettings(
            models=ModelsSettings(
                worker=ModelRole(
                    provider="azure",
                    model="gpt-5.4",
                    endpoint="https://azure.example/v1",
                    deployment=deployment,
                )
            )
        ),
    )


def test_worker_role_provider_config_model_flag_keeps_the_role_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The endpoint describes the BACKEND, not the model, so swapping the model keeps it.
    _azure_worker_settings(monkeypatch, "prod-54-canary")

    config = cli_app_module._worker_role_provider_config(
        None, "gpt-5.5", None, deployment="prod-55-canary"
    )

    assert config.kind is ProviderKind.AZURE_OPENAI
    assert config.model == "gpt-5.5"
    assert config.endpoint == "https://azure.example/v1"
    assert config.deployment == "prod-55-canary"


def test_worker_role_provider_config_refuses_an_azure_model_swap_without_a_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # On Azure the wire `model` IS the deployment name, so the role's deployment names the model
    # being replaced. Keeping it would call gpt-5.4 while reporting gpt-5.5; guessing `gpt-5.5`
    # 404s on a resource that names deployments anything else, and `wmo eval` turns that into a
    # silent fidelity=0.000 at exit 0. So refuse, naming the command that fixes it.
    _azure_worker_settings(monkeypatch, "prod-54-canary")

    with pytest.raises(typer.BadParameter) as excinfo:
        cli_app_module._worker_role_provider_config(None, "gpt-5.5", None)

    message = str(excinfo.value)
    assert "prod-54-canary" in message
    assert "wmo providers set --provider azure --model gpt-5.5 --deployment <deployment>" in message


def test_worker_role_provider_config_allows_an_azure_model_swap_the_deployment_already_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A resource whose deployments are named after their models has already answered the question.
    _azure_worker_settings(monkeypatch, "gpt-5.5")

    config = cli_app_module._worker_role_provider_config(None, "gpt-5.5", None)

    assert config.model == "gpt-5.5"
    assert config.deployment == "gpt-5.5"


def test_worker_role_provider_config_derives_a_deployment_the_role_never_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Nothing configured to contradict, so fall back to the model type as
    # `_worker_provider_config` does rather than refusing over a value that was never there.
    _azure_worker_settings(monkeypatch, None)

    config = cli_app_module._worker_role_provider_config(None, "gpt-5.5", None)

    assert config.deployment == "gpt-5.5"


def test_worker_role_provider_config_keeps_a_custom_deployment_for_the_same_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Re-stating the role's own model is not a model change: an operator's deployment name is not
    # derivable from the model id, so re-deriving it here would break a working config.
    _azure_worker_settings(monkeypatch, "prod-54-canary")

    config = cli_app_module._worker_role_provider_config(None, "gpt-5.4", None)

    assert config.deployment == "prod-54-canary"


def test_worker_role_provider_config_provider_flag_uses_that_backends_flagship(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A --provider naming another backend must take its model from THAT backend's catalog:
    # pairing --provider openai with bedrock's claude-opus-4-8 sends OpenAI a model it has never
    # heard of, so the command fails instead of running on the backend the user selected.
    from wmo.config.settings import ModelRole, ModelsSettings, ProjectSettings

    monkeypatch.setattr(
        cli_app_module,
        "load_settings_or_abort",
        lambda: ProjectSettings(
            models=ModelsSettings(worker=ModelRole(provider="bedrock", model="claude-sonnet-4-6"))
        ),
    )

    config = cli_app_module._worker_role_provider_config("openai", None, None)

    assert config.kind is ProviderKind.OPENAI
    assert config.model == "gpt-5.6-sol"


def test_worker_role_provider_config_demands_a_model_for_a_catalog_less_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # openrouter/tinker publish no built-in rows — nothing can derive an operator's route or
    # weights path — so the fix is to say which model, not to guess one.
    from wmo.config.settings import ProjectSettings

    monkeypatch.setattr(cli_app_module, "load_settings_or_abort", lambda: ProjectSettings())

    with pytest.raises(typer.BadParameter) as excinfo:
        cli_app_module._worker_role_provider_config("openrouter", None, None)

    assert "pass --model <model>" in str(excinfo.value)
    assert "wmo providers set --provider openrouter --model <model>" in str(excinfo.value)


def test_download_fetches_named_benchmarks(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    fetched: list[tuple[str, bool]] = []

    def fake_fetch(name: str, *, force: bool = False, on_progress=None) -> Path:  # noqa: ANN001
        fetched.append((name, force))
        return tmp_path / name / "traces.otel.jsonl"

    monkeypatch.setattr("wmo.hub.fetch_corpus", fake_fetch)
    monkeypatch.setattr("wmo.hub.corpus_path", lambda name: tmp_path / name / "missing")
    result = runner.invoke(app, ["download", "bird-sql", "dabstep", "--force"])
    assert result.exit_code == 0, result.output
    assert fetched == [("bird-sql", True), ("dabstep", True)]
    assert "fetched" in result.output


def test_download_all_expands_to_the_published_list(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    # `all` means "everything actually on the Hub" (live list), not the static registry — a
    # registry entry that isn't published yet would 404.
    fetched: list[str] = []
    published = [SimpleNamespace(benchmark=n, last_modified=None) for n in ("a-bench", "b-bench")]
    monkeypatch.setattr("wmo.hub.published_corpora", lambda: published)
    monkeypatch.setattr(
        "wmo.hub.fetch_corpus",
        lambda name, force=False, on_progress=None: fetched.append(name) or tmp_path,
    )
    monkeypatch.setattr("wmo.hub.corpus_path", lambda name: tmp_path / name / "missing")
    result = runner.invoke(app, ["download", "all"])
    assert result.exit_code == 0, result.output
    assert fetched == ["a-bench", "b-bench"]


def test_download_multi_skips_a_404_and_fetches_the_rest(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    # One unpublished dataset must not abort the remaining downloads (it used to kill `all`
    # mid-loop, alphabetically stranding everything after the 404).
    import urllib.error

    fetched: list[str] = []

    def fetch(name, force=False, on_progress=None):  # noqa: ANN001, ANN202
        if name == "broken":
            raise urllib.error.HTTPError("https://hub/x", 404, "nf", None, None)  # ty: ignore[invalid-argument-type]
        fetched.append(name)
        return tmp_path

    monkeypatch.setattr("wmo.hub.fetch_corpus", fetch)
    monkeypatch.setattr("wmo.hub.corpus_path", lambda name: tmp_path / name / "missing")
    result = runner.invoke(app, ["download", "a-bench", "broken", "z-bench"])
    assert fetched == ["a-bench", "z-bench"]  # kept going past the 404
    assert result.exit_code != 0  # ...but the failure is still reported at the end
    assert "broken" in result.output


def test_download_all_offline_skips_the_unpublished_and_still_succeeds(  # noqa: ANN201
    monkeypatch,  # noqa: ANN001
    tmp_path: Path,
):
    # Offline, `all` falls back to the local registry. That registry names bundles registered
    # here so the write side knows how to publish them but never pushed, and the Hub can only
    # answer 401 for those — which used to turn an otherwise complete `wmo download all` into a
    # failed command over something the user cannot act on. The fallback is the published
    # subset, and it says what it dropped.
    import urllib.error

    from wmo.hub import CORPORA, downloadable_benchmarks

    unpublished = sorted(n for n, spec in CORPORA.items() if not spec.published)
    assert unpublished, "this test is meaningless once every registered corpus is published"
    fetched: list[str] = []

    def no_catalogue(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("wmo.hub.published_corpora", no_catalogue)
    monkeypatch.setattr(
        "wmo.hub.fetch_corpus",
        lambda name, force=False, on_progress=None: fetched.append(name) or tmp_path,
    )
    monkeypatch.setattr("wmo.hub.corpus_path", lambda name: tmp_path / name / "missing")
    result = runner.invoke(app, ["download", "all"])
    assert result.exit_code == 0, result.output  # no failure over an unpushed registry entry
    assert fetched == downloadable_benchmarks()
    for name in unpublished:
        assert name not in fetched
        assert name in result.output  # the narrowing is announced, never silent


def test_download_multi_keeps_going_past_a_truncated_transfer(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    # A file still short after `fetch_corpus`'s own per-file retries raises OSError, which used
    # to escape the loop's per-item handling and kill the command — so a bundle the Hub served
    # badly stranded every benchmark queued behind it, exactly like the 404 above once did.
    fetched: list[str] = []

    def fetch(name, force=False, on_progress=None):  # noqa: ANN001, ANN202
        if name == "short":
            raise OSError("traces.otel.jsonl: downloaded 6 bytes but the Hub tree lists 4096")
        fetched.append(name)
        return tmp_path

    monkeypatch.setattr("wmo.hub.fetch_corpus", fetch)
    monkeypatch.setattr("wmo.hub.corpus_path", lambda name: tmp_path / name / "missing")
    result = runner.invoke(app, ["download", "a-bench", "short", "z-bench"])
    assert fetched == ["a-bench", "z-bench"]  # kept going past the short transfer
    assert result.exit_code != 0  # ...but the failure is still reported at the end
    assert "short" in result.output


def test_download_of_one_bundle_reports_a_truncated_transfer_as_a_failure(  # noqa: ANN201
    monkeypatch,  # noqa: ANN001
    tmp_path: Path,
):
    # Alone it is a runtime failure, not a usage error: the name was fine, the transfer was not.
    def fetch(name, force=False, on_progress=None):  # noqa: ANN001, ANN202
        raise OSError("traces.otel.jsonl: 6 bytes, tree lists 4096 — truncated transfer")

    monkeypatch.setattr("wmo.hub.fetch_corpus", fetch)
    monkeypatch.setattr("wmo.hub.corpus_path", lambda name: tmp_path / name / "missing")
    result = runner.invoke(app, ["download", "dabstep"])
    assert result.exit_code == 1
    assert "truncated transfer" in result.output
    assert "Invalid value" not in result.output


def test_download_multi_reports_an_unknown_name_without_stranding_the_rest(  # noqa: ANN201
    monkeypatch,  # noqa: ANN001
    tmp_path: Path,
):
    # Same defect class, decided offline before the network is touched: one bad name in a
    # hand-typed list used to abort the command before the good ones were attempted.
    fetched: list[str] = []

    from wmo.hub import CORPORA

    def fetch(name, force=False, on_progress=None):  # noqa: ANN001, ANN202
        if name not in CORPORA:
            raise ValueError(f"{name!r} has no published corpus (available: dabstep)")
        fetched.append(name)
        return tmp_path

    monkeypatch.setattr("wmo.hub.fetch_corpus", fetch)
    monkeypatch.setattr("wmo.hub.corpus_path", lambda name: tmp_path / name / "missing")
    result = runner.invoke(app, ["download", "nope", "dabstep"])
    assert fetched == ["dabstep"]
    assert result.exit_code != 0
    assert "nope" in result.output


def test_download_failure_names_every_repo_id_it_tried(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    # A fetch tries more than one dataset repo name (the wmh -> wmo rename), so a bare "404"
    # cannot be acted on: the report must say which ids were looked for. The CLI reads that off
    # plain HTTPError attributes rather than the subclass, so the report survives any fetcher
    # that raises a stock HTTPError.
    import urllib.error
    from http.client import HTTPMessage

    from wmo.hub import CorpusRepoUnavailable, candidate_repo_ids

    attempts = [
        (repo_id, urllib.error.HTTPError(f"https://hub/{repo_id}", 404, "nf", HTTPMessage(), None))
        for repo_id in candidate_repo_ids("dabstep")
    ]

    def fetch(name, force=False, on_progress=None):  # noqa: ANN001, ANN202
        raise CorpusRepoUnavailable(name, "main", attempts)

    monkeypatch.setattr("wmo.hub.fetch_corpus", fetch)
    monkeypatch.setattr("wmo.hub.corpus_path", lambda name: tmp_path / name / "missing")
    result = runner.invoke(app, ["download", "dabstep"])
    assert result.exit_code != 0
    for repo_id in candidate_repo_ids("dabstep"):
        assert repo_id in result.output


def test_download_unknown_benchmark_is_a_usage_error(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.setattr("wmo.hub.corpus_path", lambda name: tmp_path / name / "missing")
    result = runner.invoke(app, ["download", "nope"])
    assert result.exit_code != 0
    assert "no published corpus" in result.output


def test_download_picker_lists_published_and_fetches_choice(
    monkeypatch,  # noqa: ANN001
    tmp_path: Path,
) -> None:
    from wmo.hub import PublishedCorpus

    published = [
        PublishedCorpus(
            benchmark="gaia2",
            repo_id="experiential-labs/wmo-gaia2-traces",
            last_modified="2026-07-06",
        )
    ]
    fetched: list[str] = []
    monkeypatch.setattr("wmo.hub.published_corpora", lambda: published)
    monkeypatch.setattr(
        "wmo.hub.fetch_corpus",
        lambda name, force=False, on_progress=None: fetched.append(name) or tmp_path,
    )
    monkeypatch.setattr("wmo.hub.corpus_path", lambda name: tmp_path / name / "missing")
    result = runner.invoke(app, ["download"], input="1\n")
    assert result.exit_code == 0, result.output
    assert fetched == ["gaia2"]
    assert "not downloaded" in result.output  # picker showed local status


def test_grid_output_paths_never_collide() -> None:
    # Regression: `--out foo.json` must NOT make the chart PNG overwrite the just-written result
    # JSON. The JSON and PNG always get distinct suffixes off the same stem.
    from wmo.cli.app import _grid_output_paths

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
    from wmo.cli.app import _parse_model_specs

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


def _build_cli_train_split_default() -> float:
    """The `--train-split` default `wmo build` registers, read off the Typer option itself."""
    option = inspect.signature(cli_app_module.build).parameters["train_split"].default
    return cast(float, option.default)


def _eval_cli_train_split_default() -> float:
    """The train split `wmo eval` resolves when the user passes no `--train-split`."""
    # `_eval_options` is the resolver under test: it is where `wmo eval` turns "no flag given"
    # into a concrete split, so asserting on its output is asserting on the real default.
    options = cli_app_module._eval_options(
        prompt_file=None,
        train_split=None,
        embed_dim=None,
        rag=None,
        sample_turns=None,
        seed=None,
        top_k=None,
    )
    return options.train_split


def test_build_and_eval_share_one_default_train_split() -> None:
    # `wmo build` and `wmo eval` cut the SAME deterministic trace-id hash line of the SAME corpus,
    # so their defaults are not two independent knobs: they are one number. They drifted apart
    # (build 0.8, eval 0.7), which leaked the [0.7, 0.8) band of GEPA's training traces into every
    # default eval as "held-out". Both must read `DEFAULT_TRAIN_SPLIT` and nothing else.
    assert _build_cli_train_split_default() == DEFAULT_TRAIN_SPLIT
    assert _eval_cli_train_split_default() == DEFAULT_TRAIN_SPLIT
    # Pinned to each other too, so a future edit to one alone is a failure and not a silent leak.
    assert _build_cli_train_split_default() == _eval_cli_train_split_default()
    # Suites and the Python-API config default sit on the same line and must agree as well.
    assert EvalSuiteConfig().train_split == DEFAULT_TRAIN_SPLIT
    assert HarnessConfig().train_split == DEFAULT_TRAIN_SPLIT


def test_default_eval_holdout_contains_no_build_training_trace() -> None:
    # The measurement-validity invariant behind the shared constant, checked end to end on the
    # real split functions: nothing GEPA trained on may be scored as held-out.
    traces = [Trace(trace_id=f"trace-{i}") for i in range(400)]
    build_split = _build_cli_train_split_default()
    # Mirrors `wmo.engine.build.build`: train / val / test on one hash line.
    gepa_train, _val, _test = split_traces_3way(traces, build_split, (1.0 - build_split) / 2)
    # Mirrors `wmo.evals.open_loop.evaluate_files` on the ad hoc `wmo eval <file>` path.
    _eval_train, holdout = split_traces(traces, _eval_cli_train_split_default())
    assert gepa_train, "sanity: the corpus must actually produce a training split"
    assert holdout, "sanity: the corpus must actually produce a holdout"
    leaked = {t.trace_id for t in gepa_train} & {t.trace_id for t in holdout}
    assert leaked == set(), f"{len(leaked)} GEPA training traces scored as held-out"


def test_build_defaults_to_the_free_fidelity_tier(patched_provider, tmp_path) -> None:  # noqa: ANN001
    """A plain `wmo build` must not spend on search.

    The default was `medium`, which runs GEPA plus a cheap-lever config search; in one observed
    build that was 73% of total spend. `low` is `estimate_only` with `gepa_budget=0` and no
    config search, so the documented quickstart no longer bills a first-time user by default.
    `auto_fidelity.json` records the difference: the low tier writes a signature estimate with
    no `scores`, while every searching tier writes the scores it paid for.
    """
    spec = FIDELITY_TIERS[FidelityTier.LOW]
    assert spec.gepa_budget == 0 and spec.config_search is False, "low is no longer the free tier"

    root = tmp_path / ".wmo"
    result = runner.invoke(
        app,
        [
            "build",
            "--name",
            "defaulted",
            "--file",
            _traces_file(tmp_path),
            "--root",
            str(root),
            "--provider",
            "bedrock",
        ],
    )

    assert result.exit_code == 0, result.output
    auto = json.loads(
        (root / "models" / "defaulted" / "auto_fidelity.json").read_text(encoding="utf-8")
    )
    assert not auto.get("scores"), f"default build paid for a config search: {auto['scores']}"
