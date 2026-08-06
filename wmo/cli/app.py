"""`wmo` CLI — ingestion UI and operator console for the harness.

Deliberately small. The lifecycle is:
    providers verify -> build -> list -> serve / demo / play
`build` creates the project artifact directory itself, so there is no separate init step. World
models are named (`--name`), stored under `<root>/models/<name>/`, and listed with `wmo list`.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import random
import time
import urllib.error
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, cast
from uuid import uuid4

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.filesize import decimal
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table

from wmo.cli.defer import add_deferred_typer
from wmo.cli.model_roles import configured_role_configs, load_settings_or_abort
from wmo.config import (
    ARTIFACT_DIR,
    DEFAULT_MODEL_NAME,
    FIDELITY_TIERS,
    PROVIDER_ENV_VARS,
    ArtifactPaths,
    FidelityTier,
    HarnessConfig,
    ModelRole,
    WorldModelStore,
    load_config,
    load_env_file,
    normalize_name,
    save_settings,
    set_telemetry_enabled,
    settings_path,
    validate_name,
)

if TYPE_CHECKING:
    import wmo.cli.pool_registry as pool_registry
    from wmo.core.types import JsonObject, Trace
    from wmo.engine.eval_suites import EvalSuite
    from wmo.engine.world_model import WorldModel
    from wmo.evals.grid import ModelSpec
    from wmo.evals.open_loop import EvalReport
    from wmo.providers import ProviderConfig, ProviderKind, VerifyResult
    from wmo.providers.base import Embedder, Provider
    from wmo.providers.models import ProviderModel
    from wmo.providers.pool import Tier
    from wmo.scenarios import ScenarioSet


app = typer.Typer(
    help="Run agents, build world models from traces, and optimize agent harnesses.",
    no_args_is_help=True,
)


providers_app = typer.Typer(help="Manage and verify LLM providers.", no_args_is_help=True)
# "harness" here would collide with the `wmo harness` group, which manages a different object.
config_app = typer.Typer(help="Manage project-local wmo settings.", no_args_is_help=True)
research_app = typer.Typer(
    help="Research experiments over the harness (scaling laws, ablations).", no_args_is_help=True
)
scenarios_app = typer.Typer(
    help="Construct and verify representative eval scenario sets from traces.",
    no_args_is_help=True,
)
app.add_typer(providers_app, name="providers")
app.add_typer(config_app, name="config")
app.add_typer(research_app, name="research")
app.add_typer(scenarios_app, name="scenarios")
add_deferred_typer(
    app,
    name="optimize",
    module="wmo.cli.optimize_app",
    attr="optimize_app",
    help=(
        "Optimizers behind one switch. `model` is the staged one-command path (preflight, "
        "sweep, fit, tune, report); `route` is those steps individually; `distill` trains "
        "an adapter."
    ),
    known_names=("route", "distill", "model"),
)
add_deferred_typer(
    app,
    name="runs",
    module="wmo.cli.runs_app",
    attr="runs_app",
    help="See and steer optimization runs: list, show, tail, stop, retry, backfill.",
    known_names=("list", "show", "tail", "stop", "retry", "backfill"),
)


def _register_ingest() -> None:
    from wmo.cli.ingest_cmd import ingest as _ingest_command

    app.command("ingest")(_ingest_command)


def _register_side_commands() -> None:
    from wmo.cli.platform_cmds import register as register_platform_commands

    register_platform_commands(app)


_register_ingest()
_register_side_commands()
_console = Console()
_CHECK = "[green]✓[/green]"

# Module-level singleton: a typer.Argument call can't be a default inline (ruff B008).
# Every dispatched flow is listed: the `grid*` family owns six of this command's options
# (--val-frac, --models, --gepa-prompts, --dataset-label, --limit-traces, --judge-model), so
# leaving its tokens out of the help made those flags reference an undiscoverable subcommand.
_EVAL_TOKENS = typer.Argument(
    None,
    help="Trace files to score, or eval flow: list | run <suite> | results optional-suite | "
    "grid <suite> | grid-plot <result.json>... | grid-heatmap <result.json>... | "
    "agreement <a.json> <b.json>.",
)
# Repeatable option default hoisted out of the signature (ruff B008 forbids the call inline).
_RESEARCH_REAL_ARG = typer.Option(
    None, "--real-arg", help="Extra arg forwarded to the real sandbox run.sh; repeat for several."
)
_RESEARCH_PLOT_REPORT = typer.Argument(..., help="ConcurrencyScalingReport JSON file to plot.")
_RESEARCH_PLOT_COMBINED = typer.Argument(
    ..., help="ConcurrencyScalingReport JSONs to overlay (one per benchmark)."
)
# Flags each real runner needs so concurrent scenarios stay COLD/FRESH (as if on separate machines)
# and measure the TRUE standup cost. swe-bench: `--mode build` forces the honest from-source standup
# (base image + conda/pip env install + repo clone/checkout/install, minutes/env) instead of pulling
# SWE-bench's prebuilt image (~16s, which under-counts the real environment cost ~15-30x); plus
# `--no-family-purge` keeps each instance's own build cold without deleting sibling runs' images.
# terminal-tasks needs nothing — it already builds a per-run unique image tag under concurrency.
# tau-bench's real side is in-process (no docker), so it is always isolated.
_CONCURRENCY_ISOLATION_FLAGS: dict[str, tuple[str, ...]] = {
    # --cache-shared: build the shared base+env images once, but cold-build the per-instance image
    # at every level (the marginal per-scenario standup). Correct for this fixed-N sweep, where the
    # same scenarios repeat across levels — plain --no-family-purge would rebuild the 660MB base for
    # every scenario and let concurrent workers clobber each other's base image. (--cache-shared
    # implies no family purge.)
    "swe-bench": ("--mode", "build", "--cache-shared"),
}

_DOWNLOAD_BENCHMARKS = typer.Argument(
    None, help="Benchmark bundles to download, or 'all'. Omit for a picker."
)
# Repeatable option default hoisted out of the signature (ruff B008 forbids the call inline).
_POOL_MODEL_OPTION = typer.Option(
    None,
    "--pool-model",
    help="Register this model id as a routing candidate, with no prompts; repeat for several.",
)


@dataclass(frozen=True)
class _EvalOptions:
    prompt_file: str | None
    train_split: float
    embed_dim: int
    use_rag: bool
    sample_turns: str
    seed: int
    top_k: int
    knowledge: bool
    reasoning: bool


@config_app.command("telemetry")
def config_telemetry(
    action: str = typer.Argument("status", help="status | enable | disable"),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir holding local settings."),
) -> None:
    """View or change project-local usage telemetry settings."""
    normalized = action.lower()
    if normalized not in ("status", "enable", "disable"):
        raise typer.BadParameter("action must be one of: status, enable, disable")
    # Read through the guarded loader first: `set_telemetry_enabled` reads the same file to
    # preserve the rest of it, so a corrupt settings.toml must fail here as a usage error naming
    # the file rather than as a tomllib traceback from inside the write.
    settings = load_settings_or_abort(root)
    if normalized != "status":
        settings = set_telemetry_enabled(normalized == "enable", root)
    state = "enabled" if settings.telemetry.enabled else "disabled"
    _console.print(f"telemetry {state} ({settings_path(root)})")


def _worker_provider_config(
    provider: str,
    model: str,
    region: str | None,
    *,
    endpoint: str | None = None,
    deployment: str | None = None,
    api_version: str | None = None,
) -> ProviderConfig:
    """Resolve the provider settings used by the built-in worker agent."""
    from wmo.providers import ProviderKind

    config = _provider_config(provider, model, region)
    if endpoint is not None:
        config = config.model_copy(update={"endpoint": endpoint})
    if config.kind is ProviderKind.AZURE_OPENAI:
        config = config.model_copy(
            update={
                "deployment": deployment or config.model_type or config.model,
                "api_version": api_version or "2024-05-01-preview",
            }
        )
    return config


@providers_app.command("set")
def providers_set(
    provider: str = typer.Option(None, "--provider", help="Provider for the local worker agent."),
    model: str = typer.Option(None, "--model", help="Canonical model type for the worker."),
    region: str = typer.Option(None, "--region", help="AWS region for Bedrock."),
    endpoint: str = typer.Option(None, "--endpoint", help="OpenAI-compatible API endpoint."),
    deployment: str = typer.Option(None, "--deployment", help="Azure deployment name."),
    api_version: str = typer.Option(None, "--api-version", help="Azure API version."),
    pool: str = typer.Option(
        None,
        "--pool",
        help="Candidate pool TOML the router picks from (default: <root>/pool.toml).",
    ),
    pool_model: list[str] | None = _POOL_MODEL_OPTION,
    api_key_env: str = typer.Option(
        None, "--api-key-env", help="Env var holding the pool entries' API key (multi-account)."
    ),
    tier: str = typer.Option(None, "--tier", help="Pool entry tier: frontier | open."),
    input_per_mtok: float = typer.Option(
        None,
        "--input-per-mtok",
        min=0.0,
        help="Prompt-token price of every --pool-model, USD per 1M tokens.",
    ),
    output_per_mtok: float = typer.Option(
        None,
        "--output-per-mtok",
        min=0.0,
        help="Completion-token price of every --pool-model, USD per 1M tokens.",
    ),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir holding local settings."),
) -> None:
    """Choose the local worker's provider, and register the models the router can choose from.

    Two things a project needs, in one command. The worker provider lands in
    `<root>/settings.toml` as `\\[models.worker]`, exactly as before. Then, on a terminal, this
    offers to register models as ROUTING CANDIDATES in `<root>/pool.toml`, the roster
    `wmo optimize route` selects over: pick a backend, search its catalog, and answer only what
    that backend needs (an Azure deployment; a price for a model with no published one, never
    for OpenRouter, which self-prices). Re-run it to add another provider's models beside the
    ones already registered.

    Scripts keep the old contract: with `--provider` and `--model` both given, nothing is
    prompted. Registering non-interactively is `--pool-model <id>`, repeated per model, with
    `--input-per-mtok`/`--output-per-mtok` when the model has no published price.

    A locally hosted OpenAI-compatible server (Ollama, vLLM, llama.cpp) registers through the
    same command: `--provider openai --endpoint http://localhost:11434/v1` with `--model` and
    `--pool-model` naming what that server serves. Self-hosted candidates are priced explicitly
    at $0 per Mtok unless the price flags say otherwise, and the interactive openai pass asks
    for the endpoint URL and lists the server's own models.
    """
    import wmo.cli.pool_registry as pool_registry
    from wmo.cli.ui import select_provider_and_model
    from wmo.providers import verify_all

    if tier is not None and tier not in pool_registry.TIERS:
        raise typer.BadParameter(f"--tier must be one of: {', '.join(pool_registry.TIERS)}")
    if (input_per_mtok is None) != (output_per_mtok is None):
        # A pool entry takes both prices or neither, so half a pair would be silently dropped
        # (interactively) or rejected as an invalid entry (with --pool-model).
        raise typer.BadParameter(
            "set both --input-per-mtok and --output-per-mtok, or neither; a pool entry prices "
            "prompt and completion tokens together"
        )
    if provider is not None:
        # Reject a bad --provider before reading the project's settings, so the argument the
        # caller typed is what the error is about even in a project whose settings.toml is broken.
        _provider_kind(provider)
    existing = load_settings_or_abort(root).models.worker
    used_picker = _console.is_terminal and (provider is None or model is None)
    if used_picker:
        provider, model, region = select_provider_and_model(
            _console,
            lambda text: _console.input(text),
            lambda text: _console.input(text, password=True),
            default_provider=provider or (existing.provider if existing else None),
            default_model=model or (existing.model if existing else None),
            default_region=region or (existing.region if existing else None),
            interactive=True,
            check=lambda cfg: verify_all(
                [
                    _worker_provider_config(
                        cfg.kind.value,
                        cfg.model_type or cfg.model,
                        cfg.region,
                        endpoint=endpoint,
                        deployment=deployment,
                        api_version=api_version,
                    )
                ]
            )[0],
        )
    if provider is None or model is None:
        raise typer.BadParameter(
            "provide --provider and --model, or run `wmo providers set` in a terminal"
        )
    config = _worker_provider_config(
        provider,
        model,
        region,
        endpoint=endpoint,
        deployment=deployment,
        api_version=api_version,
    )
    if not used_picker:
        result = verify_all([config])[0]
        if not result.ok:
            detail = escape(result.detail or "unknown error")
            _console.print(f"[red]provider verification failed[/red]: {detail}")
            raise typer.Exit(1)

    settings = load_settings_or_abort(root)
    settings.models.worker = ModelRole(
        provider=config.kind.value,
        model=config.model_type or model,
        region=config.region,
        endpoint=config.endpoint,
        deployment=config.deployment,
        api_version=config.api_version,
    )
    save_settings(settings, root)
    _console.print(
        f"[green]set[/green] local worker provider to {config.kind.value} "
        f"({config.model_type or model}) in {settings_path(root)}"
    )
    _register_pool_models(
        pool_path=pool_registry.pool_path_for(root, pool),
        kind=config.kind,
        pool_model=list(pool_model or []),
        options=pool_registry.EntryOptions(
            endpoint=config.endpoint,
            region=config.region,
            # The RAW flags, not the worker config's: `_worker_provider_config` fills an Azure
            # deployment in from the model id when none was given, and a pool entry must never
            # inherit a guessed deployment name (nothing can derive an operator's own).
            deployment=deployment,
            api_version=api_version,
            api_key_env=api_key_env,
            tier=cast("Tier", tier) if tier is not None else "frontier",
            input_per_mtok=input_per_mtok,
            output_per_mtok=output_per_mtok,
        ),
        interactive=used_picker,
    )


def _register_pool_models(
    *,
    pool_path: Path,
    kind: ProviderKind,
    pool_model: list[str],
    options: pool_registry.EntryOptions,
    interactive: bool,
) -> None:
    """Register routing candidates after the worker provider is saved.

    Explicit `--pool-model` ids win and register with no prompts, so a script gets the same
    roster a person would build by hand. Otherwise the registry is OFFERED only on the run that
    already prompted (a bare `wmo providers set` at a terminal): a scripted
    `--provider ... --model ...` invocation keeps its exact pre-existing behavior and asks
    nothing.
    """
    import wmo.cli.pool_registry as pool_registry

    if pool_model:
        pool_registry.register_model_ids(
            _console, pool_path=pool_path, kind=kind, model_ids=pool_model, options=options
        )
        return
    if not interactive:
        return
    pool_registry.run_pool_registry(
        _console,
        lambda text: _console.input(text),
        lambda text: _console.input(text, password=True),
        pool_path=pool_path,
        default_kind=kind,
        options=options,
    )


@providers_app.command("verify")
def providers_verify(
    name: str = typer.Option(None, "--name", help="Verify one model's providers (default: all)."),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir."),
) -> None:
    """Ping every configured provider (completion + embed path) and report status.

    Two sources count as "configured", because checking that credentials work is what you do
    BEFORE spending anything on `wmo build`: the `\\[models.<role>]` roles in
    `<root>/settings.toml` (what `wmo providers set` writes), and the providers persisted inside
    every built world model. Completion providers are deduped by kind+model across both, so a
    role naming the same backend as a built model costs one ping, not two. The phi embed path
    belongs to a BUILT model, so on a project with nothing built that half is skipped with a
    note instead of aborting the command; otherwise every distinct provider-backed embedder is
    checked (the offline hashing embedder needs no credentials and is skipped). `--name` scopes
    the whole report to that one world model.
    """
    from wmo.providers import verify_all, verify_embedder
    from wmo.providers.base import EmbedderKind

    store = WorldModelStore(root)
    names = [name] if name is not None else store.list_names()
    configs: list[HarnessConfig] = []
    for model_name in names:
        try:
            # Reading the artifact is inside the guard too: a model dir that exists but whose
            # config.toml is corrupt or unreadable is a bad artifact, not an internal error.
            configs.append(load_config(str(store.resolve(model_name))))
        except (FileNotFoundError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc

    # Dedup identical completion calls across all selected models, then across the settings
    # roles. World models come FIRST so a config present in both is pinged with the built
    # artifact's own copy, exactly as before roles were a source.
    labelled = [
        (model_name, pc)
        for model_name, cfg in zip(names, configs, strict=True)
        for pc in cfg.providers
    ]
    # `--name` asks about one world model; widening it to the project's roles would answer a
    # question the caller did not ask (and bill for it).
    if name is None:
        labelled += [(f"models.{role}", pc) for role, pc in configured_role_configs(root)]
    index: dict[str, int] = {}
    provider_configs: list[ProviderConfig] = []
    sources: list[list[str]] = []
    for label, pc in labelled:
        key = _completion_identity(pc)
        if key not in index:
            index[key] = len(provider_configs)
            provider_configs.append(pc)
            sources.append([])
        if label not in sources[index[key]]:
            sources[index[key]].append(label)

    if not provider_configs:
        _console.print(
            "[yellow]nothing configured[/yellow]; run `wmo providers set` to choose a provider, "
            "or `wmo build --name <name>` to build a world model"
        )
        raise typer.Exit(1)

    for result, origin in zip(verify_all(provider_configs), sources, strict=True):
        _print_verify_result(result, origin)

    if not configs:
        _console.print(
            "[dim]embed path: skipped, no world model built yet "
            "(the embedder is chosen by `wmo build`)[/dim]"
        )
        return

    # Verify each distinct provider-backed embed path (skip the in-process embedders: hashing
    # and local have no provider to verify).
    embed_seen: set[str] = set()
    for model_name, config in zip(names, configs, strict=True):
        if config.embed_provider in (EmbedderKind.HASHING, EmbedderKind.LOCAL):
            continue
        embed_config = config.embed_provider_config()
        key = embed_config.model_dump_json()
        if key in embed_seen:
            continue
        embed_seen.add(key)
        result = verify_embedder(embed_config)
        _print_verify_result(result, [model_name], prefix="embed:")


def _completion_identity(config: ProviderConfig) -> str:
    """What makes two provider configs the SAME completion call, for `providers verify` dedup.

    Everything the config carries except the embed-only fields, which do not reach a completion
    request. Keying on kind+model alone would collapse one model served from two Bedrock regions,
    or one Azure model behind two deployments or endpoints, into a single ping, and then report
    the config that was never called as verified on the other one's result. Deriving the key
    from the whole config rather than a hand-listed subset means a field added later splits the
    key by default instead of silently widening that collapse.
    """
    return config.model_copy(update={"embed_model": None, "embed_dim": None}).model_dump_json()


def _print_verify_result(result: VerifyResult, sources: list[str], *, prefix: str = "") -> None:
    """Print one `providers verify` line, plus the next step to take when the ping failed.

    `sources` names what asked for this provider (world model names, `models.<role>` settings
    roles) so a failure points at the thing to fix. The detail and the hint are escaped: they
    carry raw provider error text and pip extras (`...[distill]`), and an unescaped `[...]` in
    either would be read as rich markup and silently dropped from the report.
    """
    mark = "[green]ok[/green]" if result.ok else "[red]fail[/red]"
    origin = f" [dim]({', '.join(sources)})[/dim]" if sources else ""
    detail = f" {escape(result.detail)}" if result.detail else ""
    _console.print(f"{mark} {prefix}{result.kind.value} ({result.model}){origin}{detail}")
    if not result.ok:
        hint = escape(_credential_hint(result.kind, result.detail))
        _console.print(f"  [yellow]{hint}[/yellow]")


# The serve provider `wmo build` falls back to when neither `--provider` nor a configured worker
# role names one.
_BUILD_PROVIDER = "bedrock"
# Sources that read a live database rather than an export. `wmo build` has no transport flags for
# them (`VendorPull.dsn/table` are only reachable from `wmo ingest`), so it rejects them up front
# instead of failing inside the adapter with advice about flags it does not define.
_DB_SOURCES = ("postgres",)


def _check_build_source(source: str) -> None:
    """Reject a `--source` that `wmo build` cannot drive, naming the `wmo ingest` two-step."""
    from wmo.ingest import list_adapters

    if source not in list_adapters():
        raise typer.BadParameter(
            f"unknown --source {source!r}; choose one of: {', '.join(list_adapters())}"
        )
    if source in _DB_SOURCES:
        raise typer.BadParameter(
            f"--source {source} reads a live database, which `wmo build` has no connection flags "
            f"for: normalize first with `wmo ingest --source {source} --dsn <dsn> --table <table> "
            "--out traces.otel.jsonl`, then `wmo build --file traces.otel.jsonl`"
        )


def _check_build_file(file: str) -> None:
    """Reject a missing/unreadable `--file` before the provider ping, not inside the adapter."""
    path = Path(file)
    if not path.exists():
        raise typer.BadParameter(
            f"trace file not found: {file} (export one with `wmo ingest --file <export> --out "
            f"{file}`, or pass an existing path)"
        )
    if path.is_dir():
        raise typer.BadParameter(f"--file must be a trace export, not a directory: {file}")


def _empty_corpus_error(file: str | None, source: str) -> typer.BadParameter:
    """Explain an empty ingest: usually `--source` does not match the export's real format."""
    from wmo.ingest.base import load_payloads
    from wmo.ingest.detect import detect_format

    if file is None:
        return typer.BadParameter(
            f"the --source {source} pull returned no traces; widen --since/--limit or check the "
            "--project"
        )
    detected: str | None = None
    try:
        detected = detect_format(load_payloads(Path(file).read_text(encoding="utf-8")))
    except (ValueError, OSError):
        detected = None
    if detected is not None and detected != source:
        return typer.BadParameter(
            f"no traces ingested from {file} with --source {source}: it looks like {detected}. "
            f"Retry with `--source {detected}`, or let `wmo ingest --file {file} --out "
            "traces.otel.jsonl` auto-detect the format and build from its output."
        )
    return typer.BadParameter(
        f"no traces ingested from {file} with --source {source} (build never auto-detects): "
        f"check that the export carries agent steps and that --source names its format, or run "
        f"`wmo ingest --file {file} --out traces.otel.jsonl` to auto-detect and normalize it "
        "first"
    )


def _chain_bad_parameter(err: ValueError) -> typer.BadParameter:
    """Render a failover-chain resolution failure as a usage error naming the file to write."""
    from wmo.providers.waterfall import FALLBACK_CONFIG_PATH

    return typer.BadParameter(
        f"{err}. Write {FALLBACK_CONFIG_PATH} as [[chain.<name>]] rung tables (one table per "
        "fallback rung, keys kind/model/profile/region; format in docs/reference/failover.md), "
        "or drop --chain to serve from --provider alone"
    )


def _provider_or_chain_or_abort(config: ProviderConfig, chain: str | None) -> Provider:
    """`providers.provider_or_chain`, with chain-resolution errors as clean usage errors."""
    import wmo.providers as providers

    try:
        return providers.provider_or_chain(config, chain=chain)
    except ValueError as err:
        raise _chain_bad_parameter(err) from None


@app.command("build")
def build(
    name: str = typer.Option(None, "--name", help="Name for this world model."),
    source: str = typer.Option(
        "otel-genai",
        "--source",
        help="Trace source adapter, pinned (build never auto-detects): otel-genai, chat-json, "
        "braintrust, phoenix, langfuse, langsmith, posthog, mastra. To auto-detect an export's "
        "format, normalize it first with `wmo ingest --file <export>`.",
    ),
    file: str = typer.Option(None, "--file", help="Path to an exported traces file for --source."),
    pull: bool = typer.Option(
        False, "--pull", help="Pull traces live from the source's vendor API (instead of --file)."
    ),
    project: str = typer.Option(None, "--project", help="Vendor project/workspace id (--pull)."),
    api_key: str = typer.Option(None, "--api-key", help="Vendor API key (else env var)."),
    since: str = typer.Option(None, "--since", help="Only pull traces since this ISO timestamp."),
    limit: int = typer.Option(
        None,
        "--limit",
        help="Only use the first N traces (caps a --file export too, not just --pull); cost "
        "control for a first build over a large corpus. A --pull is capped at fetch time, so "
        "with --drop-degenerate it can yield fewer than N; a --file export yields N usable.",
    ),
    vendor: str = typer.Option(
        None, "--vendor", help="\\[deprecated] alias for --source <name> --pull."
    ),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir holding all world models."),
    provider: str = typer.Option(
        None, "--provider", help=f"Provider that serves the model (default: {_BUILD_PROVIDER})."
    ),
    model: str = typer.Option(
        None, help="Canonical serve model type (default: the provider's suggested model)."
    ),
    judge_model: str = typer.Option(
        None, "--judge-model", help="Canonical GEPA judge model type (default: cheap per provider)."
    ),
    region: str = typer.Option(None, help="AWS region (Bedrock)."),
    fidelity: str = typer.Option(
        "low",
        help="Build effort (all searching tiers are floored at low's estimate — more effort "
        "never ships worse than low): low (default; free — the estimated-best config, no "
        "search) | medium (+light GEPA + cheap-lever search) | high (+GEPA + config search) | "
        "max (deep GEPA + full config search). Searching tiers cost real money: one observed "
        "medium build spent 73% of its total on GEPA. The chosen config serves under "
        "`--max-fidelity`.",
    ),
    chain: str = typer.Option(
        None, "--chain", help="Named failover chain from .wmo/fallback.toml (default: its default)."
    ),
    train_split: float = typer.Option(
        0.8,
        help="Train/held-out ratio for GEPA's internal split (lower = bigger valset). Shares "
        "`wmo eval`'s default: both cut the same trace-id hash line, so a mismatch leaks train "
        "traces into the eval holdout.",
    ),
    embed_provider: str = typer.Option(
        "hashing", help="phi embedder: hashing (offline) | bedrock | openai | azure."
    ),
    embed_model: str = typer.Option(None, help="Embeddings model id / Azure embedding deployment."),
    embed_dim: int = typer.Option(512, help="phi dimensionality (index + query must agree)."),
    knowledge: bool = typer.Option(
        False,
        "--knowledge/--no-knowledge",
        help="Seed a knowledge base (rules/entities/schemas markdown) from the train traces.",
    ),
    reasoning: bool = typer.Option(
        False,
        "--reasoning/--no-reasoning",
        help="Serve with the deliberate-then-answer output contract.",
    ),
    grounder: str = typer.Option(
        "none",
        help="Web grounding for unknown entities: none | brave (needs BRAVE_SEARCH_API_KEY).",
    ),
    drop_degenerate: bool = typer.Option(
        False,
        "--drop-degenerate",
        help="Drop all-empty-observation traces (failed captures) before building "
        "(swe-bench is ~66% such junk).",
    ),
    interactive: bool = typer.Option(
        None,
        "--interactive/--no-interactive",
        help="Guided creation wizard. Default: on at a TTY when inputs are missing.",
    ),
    endpoint: str = typer.Option(None, "--endpoint", help="OpenAI-compatible API endpoint base URL (e.g. https://generativelanguage.googleapis.com/v1beta/openai/). Sets WMO_ENDPOINT_API_KEY as the credential."),
) -> None:
    """Ingest traces (file upload or vendor SDK pull) and build a named world model.

    Stores the artifact under `<root>/models/<name>/`: ingest -> normalize -> split(train/test) ->
    embed/index -> GEPA optimize -> write. Re-running with the same `--name` rebuilds it.

    With no `--name`/`--file` on an interactive terminal, this launches a guided creation wizard;
    pass `--no-interactive` (or any of those flags) to stay fully scriptable.
    """
    # `--vendor <name>` is the deprecated alias for `--source <name> --pull`: it names the source
    # adapter and implies a live pull.
    import wmo.providers as providers
    from wmo.cli.ui import (
        BuildParams,
        RichBuildReporter,
        build_summary_panel,
        judge_model_default,
        run_build_wizard,
        serve_model_default,
    )
    from wmo.config.card import make_build_card, save_card
    from wmo.engine.build import EmptyCorpusError
    from wmo.engine.build import build as run_build
    from wmo.engine.grounding import GROUNDER_KINDS
    from wmo.ingest import VendorPull
    from wmo.providers import ProviderKind
    from wmo.providers.base import EmbedderKind
    from wmo.retrieval import get_embedder
    from wmo.telemetry import BuildTelemetryStats, TelemetryBuildReporter, capture_build_completed
    from wmo.tracking import MeteredProvider, Phase, RunTracker, classify_build_call, save_run

    if vendor:
        source = vendor
        pull = True

    if endpoint is not None and not endpoint.strip():
        raise typer.BadParameter(
            '--endpoint is empty; give the OpenAI-compatible base URL, '
            'or drop the flag to use the provider default endpoint'
        )

    # Decide whether to run the wizard: explicit flag wins; otherwise auto when at a TTY and the
    # essential inputs (a name and a trace source — a file or a live pull) were not supplied.
    needs_input = name is None or (file is None and not pull)
    use_wizard = interactive if interactive is not None else (_console.is_terminal and needs_input)

    configured_worker = load_settings_or_abort(root).models.resolve("worker")
    use_configured_worker = configured_worker is not None and (
        provider is None or provider == configured_worker.provider
    )
    # Settle the serve provider here so the model default can follow it. A single hard-coded
    # Anthropic default wrote artifacts configured to call e.g. OpenAI with `claude-opus-4-8`.
    resolved_provider = (
        provider or (configured_worker.provider if configured_worker else None) or _BUILD_PROVIDER
    )
    resolved_model = (
        model
        or (configured_worker.model if use_configured_worker and configured_worker else None)
        or serve_model_default(resolved_provider)
    )
    if resolved_model is None and not use_wizard:
        raise typer.BadParameter(
            f"--provider {resolved_provider} has no default serve model: pass `--model <id>` "
            f"(`wmo build --provider {resolved_provider} --model <id> --file <export>`)"
        )
    params = BuildParams(
        name=name or DEFAULT_MODEL_NAME,
        source=source,
        file=file,
        pull=pull,
        project=project,
        api_key=api_key,
        since=since,
        limit=limit,
        provider=provider or (configured_worker.provider if configured_worker else None),
        # A wizard run re-picks provider and model from its own per-provider lists, so an
        # unresolved default here is only an unused suggestion; the flag path errored above.
        model=resolved_model or "",
        region=(
            region
            or (configured_worker.region if use_configured_worker and configured_worker else None)
        ),
        fidelity=fidelity,
        train_split=train_split,
        judge_model=judge_model,
        embed_provider=embed_provider,
        embed_model=embed_model,
        embed_dim=embed_dim,
        endpoint=endpoint,
    )
    if use_wizard:
        params = run_build_wizard(_console, params)
    elif params.file is None and not params.pull:
        # This guard also required `name is None`, so `wmo build --name x` (verbatim the
        # empty-state hint `wmo list` prints) fell through to a raw ValueError from the ingest
        # seam instead. A name is not a corpus: what is missing is the trace source.
        raise typer.BadParameter(
            "provide --file <export> or --pull (with --source), or run `wmo build` interactively"
        )
    if params.file and params.pull:
        raise typer.BadParameter("pass either --file or --pull, not both")
    _check_build_source(params.source)
    if params.file is not None:
        _check_build_file(params.file)
    if params.limit is not None and params.limit < 1:
        raise typer.BadParameter(f"--limit must be at least 1, got {params.limit}")
    # The wizard always resolves a provider; the flag path keeps its historical default.
    params.provider = params.provider or _BUILD_PROVIDER
    # The wizard may replace the configured worker's provider. Re-evaluate the match before
    # carrying provider-specific connection fields into the build config.
    use_configured_worker = (
        configured_worker is not None and params.provider == configured_worker.provider
    )

    # Flag-supplied names get the same whitespace-to-dash normalization as the wizard.
    params.name = normalize_name(params.name)
    try:
        validate_name(params.name)
    except ValueError as err:
        raise typer.BadParameter(str(err)) from None
    if params.name == "harbor":
        raise typer.BadParameter(
            "world model name 'harbor' is reserved: `wmo optimize harness <agent> harbor` selects "
            "the harbor benchmark environment; choose another name"
        )
    try:
        tier = FidelityTier(params.fidelity)
    except ValueError:
        tiers = ", ".join(t.value for t in FidelityTier)
        raise typer.BadParameter(
            f"unknown fidelity {params.fidelity!r}; choose one of: {tiers}"
        ) from None
    spec = FIDELITY_TIERS[tier]
    try:
        serve_provider = ProviderKind(params.provider)
    except ValueError:
        kinds = ", ".join(k.value for k in ProviderKind)
        raise typer.BadParameter(
            f"unknown provider {params.provider!r}; choose one of: {kinds}"
        ) from None
    try:
        embed_kind = EmbedderKind(params.embed_provider)
    except ValueError:
        kinds = ", ".join(k.value for k in EmbedderKind)
        raise typer.BadParameter(
            f"unknown embed provider {params.embed_provider!r}; choose one of: {kinds}"
        ) from None
    # A provider-backed embedder needs an embeddings model; fail fast, not deep inside embed().
    # The in-process kinds need none: hashing has no model, local carries its own default.
    if embed_kind not in (EmbedderKind.HASHING, EmbedderKind.LOCAL) and not params.embed_model:
        raise typer.BadParameter(
            f"--embed-provider {embed_kind.value} requires --embed-model "
            "(the embeddings model id / Azure embedding deployment)"
        )

    store = WorldModelStore(root)
    model_dir = str(store.model_dir(params.name))
    # Provider wiring (reuse-vs-separate embed config) lives in HarnessConfig.for_build, not here.
    config = HarnessConfig.for_build(
        serve_provider=serve_provider,
        serve_model=params.model,
        region=params.region,
        embed_provider=embed_kind,
        embed_model=params.embed_model,
        embed_dim=params.embed_dim,
        gepa_budget=spec.gepa_budget,
        train_split=params.train_split,
        judge_model=params.judge_model or judge_model_default(params.provider, params.model),
        trace_adapter=params.source,
        endpoint=params.endpoint,
    )
    if use_configured_worker and configured_worker is not None:
        config.providers[0] = config.providers[0].model_copy(
            update={
                "endpoint": configured_worker.endpoint,
                "deployment": configured_worker.deployment,
                "api_version": configured_worker.api_version,
            }
        )
    if grounder not in GROUNDER_KINDS:
        raise typer.BadParameter(
            f"unknown grounder {grounder!r}; choose one of: {', '.join(GROUNDER_KINDS)}"
        )
    # Agentic-mode flags (CLI-only, not in the wizard): persisted to config.toml so serve/load
    # pick them up; knowledge additionally seeds knowledge/ during this build.
    config.knowledge = knowledge
    config.reasoning = reasoning
    config.grounder = grounder
    # Fail fast: ping the serve provider (and the embed path, if provider-backed) before spending
    # any rollouts. A missing SDK or bad creds otherwise surfaces only deep inside GEPA, which
    # silently swallows it and "succeeds" with a useless held-out-0.0 model.
    if not use_wizard:
        # The wizard already live-pinged the serve provider and embedder inline.
        _verify_or_abort(config, chain=chain)

    # Meter the build at the provider boundary; `classify_build_call` splits judge vs GEPA by
    # system prompt. Rollouts/reflection may ride the failover chain, but the judge (GEPA's
    # fitness metric) is PINNED to the single configured backend — a judge that silently switches
    # models mid-build scores candidates on different scales. Both wrappers share one tracker,
    # so cost/tokens still land in a single run record.
    tracker = RunTracker(run_id=uuid.uuid4().hex, kind="build")
    metered = MeteredProvider(
        _provider_or_chain_or_abort(config.serve_provider_config(), chain),
        tracker,
        classify=classify_build_call,
    )
    metered_judge = metered
    if config.judge_model and config.judge_model != config.serve_provider_config().model:
        judge_cfg = config.serve_provider_config().model_copy(update={"model": config.judge_model})
        metered_judge = MeteredProvider(
            providers.get_provider(judge_cfg), tracker, classify=classify_build_call
        )
    build_stats = BuildTelemetryStats()
    with tracker.timed(), RichBuildReporter(_console, params.name) as reporter:
        try:
            result = run_build(
                config,
                file=None if params.pull else params.file,
                vendor=(
                    VendorPull(
                        api_key=params.api_key,
                        project=params.project,
                        since=params.since,
                        limit=params.limit,
                    )
                    if params.pull
                    else None
                ),
                root=model_dir,
                serve_provider=metered,
                judge_provider=metered_judge,
                embedder=get_embedder(config),
                reporter=TelemetryBuildReporter(reporter, build_stats),
                max_fidelity=spec.config_search,
                fidelity_budget=spec.search_budget,
                full_search=spec.full_ladder,
                cheap_search=spec.cheap_frontier_only,
                estimate_only=spec.estimate_only,
                drop_degenerate=drop_degenerate,
                gepa_val_cap=spec.gepa_val_cap or None,
                # One cap per build, never both: a pull is already sliced to `--limit` inside
                # `from_vendor`, so repeating it post-filter cannot restore what
                # `--drop-degenerate` removed and would only read as a promise of N usable
                # traces that the pull transport cannot keep.
                limit=None if params.pull else params.limit,
            )
        except EmptyCorpusError:
            # The one ingest outcome a user causes and can fix: the export does not parse under
            # the pinned --source. Only this type is caught, so real build bugs still surface.
            raise _empty_corpus_error(None if params.pull else params.file, params.source) from None
    record = tracker.record_summary()
    save_run(record, ArtifactPaths(model_dir).runs)
    # The card is additive metadata; a write failure (disk full, permissions) must not make an
    # otherwise-complete build exit non-zero and then block retries with "already exists".
    try:
        save_card(
            make_build_card(
                name=params.name,
                provider=params.provider,
                model_id=params.model,
                traces=build_stats.input_trace_count,
                steps=build_stats.input_step_count,
                built_at=datetime.now(UTC).isoformat(),
                source=Path(params.file).name if params.file else params.vendor,
            ),
            model_dir,
        )
    except OSError as err:
        _console.print(f"[yellow]warning[/yellow]: could not write card.json: {err}")
    capture_build_completed(
        stats=build_stats,
        gepa_budget=spec.gepa_budget,
        rollouts_used=result.metrics.rollouts_used,
        frontier_size=len(result.frontier),
        record=record,
        root=root,
    )

    _console.print(build_summary_panel(store.info(params.name), model_dir))
    auto_report = Path(model_dir) / "auto_fidelity.json"
    if auto_report.exists():
        auto = json.loads(auto_report.read_text(encoding="utf-8"))
        # The low tier writes an estimate with no scores (no search ran); higher tiers write
        # the searched scores. Render each honestly rather than an empty "(; 0 traces)".
        if auto.get("scores"):
            scores = ", ".join(f"{k}={v:.3f}" for k, v in auto["scores"].items())
            provenance = f"searched: {scores}; {auto['val_traces']} held-out traces"
        else:
            provenance = "signature estimate — no search"
        _console.print(
            f"[bold]max-fidelity config[/bold]: [bold]{auto['winner_label']}[/bold] "
            f"({provenance}) — activate with `wmo serve --max-fidelity` / "
            f"`wmo play --max-fidelity`"
        )
    _console.print(
        f"[bold]run[/bold] {record.run_id[:8]}: {record.duration_seconds:.1f}s, "
        f"{record.total.total_tokens} tokens, ${record.total.cost_usd:.4f} "
        f"({record.total.calls} calls)"
    )
    for phase in (Phase.GEPA, Phase.JUDGE):
        bucket = record.by_phase.get(phase)
        if bucket is not None:
            _console.print(
                f"  {phase.value}: {bucket.total_tokens} tokens, "
                f"${bucket.cost_usd:.4f} ({bucket.calls} calls)"
            )


def _verify_or_abort(config: HarnessConfig, chain: str | None = None) -> None:
    """Ping the serve provider (and any provider-backed embedder) and abort on failure.

    Runs before any rollouts so a missing SDK or bad creds fails loudly and immediately, instead of
    being swallowed inside GEPA and yielding a useless model. Raises `typer.Exit(1)` with an
    actionable hint (`uv sync` for a missing SDK; "check creds / model id" otherwise).
    """
    import wmo.providers as providers
    from wmo.providers import verify_all, verify_embedder
    from wmo.providers.base import EmbedderKind

    checks = [(config.serve_provider_config(), False)]
    if config.embed_provider not in (EmbedderKind.HASHING, EmbedderKind.LOCAL):
        checks.append((config.embed_provider_config(), True))

    failed = False
    for cfg, is_embed in checks:
        label = f"embed:{cfg.kind.value}" if is_embed else cfg.kind.value
        serve_provider = None if is_embed else _provider_or_chain_or_abort(cfg, chain)
        if is_embed:
            _console.print(f"verifying {label}…")
            result = verify_embedder(cfg)
        elif isinstance(serve_provider, providers.WaterfallProvider):
            # Verify the provider the build will actually use — with a chain active, that means
            # pinging every rung (a broken fallback must fail here, not hours into the build).
            label = f"{label} chain (.wmo/fallback.toml)"
            _console.print(f"verifying {label}…")
            result = serve_provider.verify()
        else:
            _console.print(f"verifying {label}…")
            result = verify_all([cfg])[0]
        if result.ok:
            _console.print(f"  {_CHECK} {label} ({result.model}) reachable")
            continue
        failed = True
        _console.print(f"  [red]✗ {label} ({result.model}) failed[/red]: {result.detail}")
        hint = escape(_credential_hint(cfg.kind, result.detail))
        _console.print(f"    [yellow]{hint}[/yellow]")
    if failed:
        raise typer.Exit(1)


# Keyed by ProviderKind.value so this module stays import-light (no providers import at load).
_PROVIDER_EXTRAS: dict[str, str] = {"tinker": "distill"}
"""Providers whose SDK ships in an optional extra, keyed by kind value so the hint can name it.

Every other provider's SDK is a core dependency, so "the module is missing" means something
different for them (a stale env) than it does here (an install step the user has not run yet).
"""


def _missing_sdk(detail: str) -> bool:
    """Does a failed ping's detail mean "the SDK is absent" rather than "the creds are wrong"?

    Two shapes reach here: the raw ImportError text of a core SDK ("No module named 'boto3'"), and
    an optional extra's own message, which replaces that text with its install hint (see
    `wmo.providers.tinker.check_tinker_prerequisites`) and so never contains the module wording.
    """
    return "No module named" in detail or "SDK is not installed" in detail


def _credential_hint(kind: ProviderKind, detail: str) -> str:
    """The next step for a failed provider ping: install the SDKs, or fix creds/model id.

    Shared by the pre-build guard and `wmo providers verify` so both name the same env vars.
    """
    if _missing_sdk(detail):
        extra = _PROVIDER_EXTRAS.get(kind.value)
        if extra is not None:
            return (
                f"run `pip install 'world-model-optimizer[{extra}]'` (or `uv sync --extra {extra}` "
                "in a checkout), then re-run `wmo providers verify`"
            )
        # The rest are core deps; a missing module means the env is stale or hand-rolled.
        return "run `uv sync` to install the provider SDKs"
    envs = ", ".join(PROVIDER_ENV_VARS.get(kind, []))
    hint = f" ({envs})" if envs else ""
    return f"check the model id and that your credentials are set{hint}"


@app.command("list")
def list_models(root: str = typer.Option(ARTIFACT_DIR, help="Project dir to list.")) -> None:
    """List every world model built under the project dir.

    An empty listing names the directory it searched, because `--root` defaults to a
    cwd-relative `.wmo` and "nothing here" and "wrong directory" look identical otherwise.
    An artifact that cannot be read is listed as `unreadable` with its reason, so one broken
    `config.toml` costs you that one row instead of the whole listing.
    """
    from wmo.cli.ui import models_table

    if Path(root).is_file():
        raise typer.BadParameter(
            f"--root {root} is a file, not a project dir; pass the dir holding models/ "
            f"(the default is `{ARTIFACT_DIR}`)"
        )
    store = WorldModelStore(root)
    infos = store.list_info()
    if not infos:
        # Name the trace export too: `wmo build --name <name>` alone has no corpus to build from.
        _console.print(
            f"[yellow]no world models built under {store.models_dir}[/yellow]; run "
            "`wmo build --name <name> --file <traces export>`"
        )
        return
    _console.print(models_table(infos))
    for info in infos:
        if info.error is not None:
            _console.print(f"  [red]✗ {info.name}[/red]: {escape(info.error)}")


@app.command("download")
def download(
    benchmarks: list[str] = _DOWNLOAD_BENCHMARKS,
    force: bool = typer.Option(False, "--force", help="Overwrite existing local files."),
) -> None:
    """Download benchmark data bundles (traces, task data, prebuilt model, evals) from the Hub.

    With no arguments, lists the org's published datasets (live, via the Hub API) and offers a
    picker. Bundles land in `environment-capture-data/<benchmark>/` under the current directory;
    set `ENVCAP_DATA_ROOT` to put them somewhere else. Existing local files are kept unless
    `--force`.

    A bundle arrives ready to use, not just ready to build from: its `models/` are prebuilt world
    models `wmo play`/`demo`/`eval --mode closed-loop` resolve by name, and its `evals/` are the
    named suites `wmo eval list` discovers.
    """
    from wmo.cli.ui import select_option
    from wmo.hub import corpus_path, published_corpora

    selected = list(benchmarks or [])
    if selected == ["all"]:
        selected = _all_downloadable()
    if not selected:
        try:
            published = published_corpora()
        except urllib.error.URLError as exc:
            raise typer.BadParameter(
                f"could not list the Hub's published datasets ({exc.reason}); check the "
                "connection, or pass benchmark names directly, e.g. `wmo download bird-sql`"
            ) from exc
        if not published:
            raise typer.BadParameter(
                "no published corpora found on the Hub; "
                "pass benchmark names directly, e.g. `wmo download bird-sql`"
            )
        notes = {}
        for corpus in published:
            local = (corpus_path(corpus.benchmark)).exists()
            state = "local copy present" if local else "not downloaded"
            when = f", updated {corpus.last_modified}" if corpus.last_modified else ""
            notes[corpus.benchmark] = f"{state}{when}"
        choices = [corpus.benchmark for corpus in published]
        picked = select_option(
            _console, "Download which data bundle?", [*choices, "all"], notes=notes
        )
        selected = choices if picked == "all" else [picked]
    failures: list[str] = []
    for name in selected:
        existing = corpus_path(name).exists()
        try:
            path = _fetch_with_progress(name, force=force)
        except urllib.error.HTTPError as exc:
            # One unpublished/broken dataset must not abort the REST of a multi-download:
            # record it, keep fetching, and fail (with every name) at the end. The reason is
            # quoted rather than summarized here: a fetch tries more than one dataset repo id
            # and only the error it raises knows which ones the Hub refused.
            reason = f"the Hub answered {exc.code} for {exc.url} ({exc.reason})"
            note = f"Hub answered {exc.code}"
        except urllib.error.URLError as exc:
            # The connection itself is down, which is NOT a verdict on one bundle: everything
            # queued behind it would fail identically, so stop instead of printing the same
            # reason once per benchmark. (Checked before the OSError branch below, which it
            # would otherwise be swallowed by — URLError is an OSError.)
            raise typer.BadParameter(
                f"{name}: could not reach the Hub ({exc.reason}); check the connection and re-run"
                " — fetches resume file-by-file"
            ) from exc
        except ValueError as exc:
            # An unknown name, decided offline before the network is touched. Asked for on its
            # own it stays a plain usage error, because wrapping `wmo download nope` in "some
            # datasets could not be downloaded" buries the answer to the common typo.
            if len(selected) == 1:
                raise typer.BadParameter(str(exc)) from exc
            reason = note = str(exc)
        except OSError as exc:
            # A transfer still truncated after `fetch_corpus`'s own per-file retries. It says
            # nothing about the bundles queued behind it, so in a list it joins the end-of-run
            # report rather than stranding them; alone it is a runtime failure, not a usage
            # error, so it exits 1 with the reason instead of `Invalid value:`.
            if len(selected) == 1:
                _console.print(f"[red]✗ could not download {name}[/red]: {escape(str(exc))}")
                raise typer.Exit(1) from exc
            reason = note = str(exc)
        else:
            state = "kept local" if existing and not force else "fetched"
            _console.print(f"{_CHECK} {state} [bold]{name}[/bold] -> {path}")
            continue
        failures.append(f"{name}: {reason}")
        _console.print(f"[yellow]skipping {name}: {escape(note)}[/yellow]")
    if failures:
        # No cause is asserted here: the list now collects unknown names, Hub refusals and
        # broken transfers alike, and each line carries the reason it actually failed for.
        raise typer.BadParameter(
            "some datasets could not be downloaded (`wmo download` with no arguments lists "
            "what the Hub publishes):\n  " + "\n  ".join(failures)
        )


def _all_downloadable() -> list[str]:
    """The bundles `wmo download all` should fetch, live Hub list preferred.

    The Hub's own listing is authoritative, so it is tried first. Offline the local registry
    answers instead — but only the entries it marks as published. The whole registry is the
    wrong answer: it names bundles registered here so the write side knows how to publish them,
    which the Hub can only answer 401 for, and one of those turns an otherwise complete
    `wmo download all` into a failed command over something the user cannot act on.

    Both narrowings are announced. A quiet substitution of a stale local list for the live one,
    or a quiet drop of a registered benchmark, reads afterwards as "everything was fetched".
    """
    from wmo.hub import CORPORA, downloadable_benchmarks, published_corpora

    try:
        return sorted(corpus.benchmark for corpus in published_corpora())
    except urllib.error.URLError as exc:
        selected = downloadable_benchmarks()
        _console.print(
            f"[yellow]could not list the Hub's published datasets ({exc.reason}); falling back "
            "to the bundles this release knows about[/yellow]"
        )
        skipped = sorted(set(CORPORA) - set(selected))
        if skipped:
            _console.print(
                f"[yellow]not downloading {', '.join(skipped)}: registered here but never "
                "pushed to the Hub, so there is nothing to fetch[/yellow]"
            )
        return selected


def _fetch_with_progress(name: str, *, force: bool) -> Path:
    """fetch_corpus with a live per-file progress bar (hidden when nothing needs downloading).

    The bar counts FILES because that is what the wait is made of: a bundle is one request per
    file, so a byte-weighted bar sat at 97% for 98% of the download. The bundle's size stays on
    screen as description text, which is where a constant belongs.
    """
    from wmo.hub import fetch_corpus

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed:.0f}/{task.total:.0f} files"),
        console=_console,
        transient=True,
    ) as progress:
        task_id = progress.add_task(f"downloading {name}", total=None, visible=False)

        def on_progress(done: float, total: float, byte_total: int) -> None:
            progress.update(
                task_id,
                completed=done,
                total=total or None,
                description=f"downloading {name} ({decimal(byte_total)})",
                visible=True,
            )

        return fetch_corpus(name, force=force, on_progress=on_progress)


@app.command("serve")
def serve(
    name: list[str] = typer.Option(  # noqa: B008 - typer reads option defaults at definition time
        None, "--name", help="World model(s) to serve. Repeatable; default: all built ones."
    ),
    port: int = typer.Option(8000, help="Port for the local backend."),
    root: list[str] = typer.Option(  # noqa: B008 - typer reads option defaults at definition time
        [ARTIFACT_DIR],
        "--root",
        help="Project dir(s) to serve from. Repeatable; server-side builds land in the first.",
    ),
    max_fidelity: bool = typer.Option(
        False,
        "--max-fidelity",
        help="Serve with the online extras on: the build-measured winning config when the "
        "artifact has one, otherwise every extra it supports. Default: pure RAG.",
    ),
) -> None:
    """Run the local FastAPI backend so agents can step against world models over HTTP.

    Serves every built model by default, or just the `--name` ones, from one or more roots
    (e.g. `--root .wmo --root environment-capture-data/tau-bench`). Two surfaces are
    exposed: the world-model step API, namespaced `/world_models/{name}/sessions` and
    `.../step`; and, for every served model whose dir carries a `policy.json` (written by
    `wmo optimize route fit --out` or `wmo optimize model`), the OpenAI-compatible endpoint
    `POST /v1/chat/completions` with `model="<name>"`, listed by `GET /v1/models`.
    """

    import uvicorn

    from wmo.serving.server import create_app

    names = list(name) if name else None
    # Bad --name input (unsafe segment, unknown model, nothing built) is a usage error,
    # not a traceback; load the models before uvicorn takes over the process.
    try:
        server_app = create_app(list(root), names=names, max_fidelity=max_fidelity)
    except (ValueError, FileNotFoundError) as err:
        raise typer.BadParameter(str(err)) from None
    uvicorn.run(server_app, host="127.0.0.1", port=port)


# Charting deps ship in the optional `viz` extra, so `wmo.evals.grid_plot` /
# `wmo.research.concurrency_plot` import them lazily. Every chart-writing flow probes for them
# UP FRONT instead: `wmo eval grid` otherwise spent the whole (paid) grid and only then died on
# `import matplotlib` with a raw traceback that never named the extra.
_VIZ_MODULES = ("matplotlib", "seaborn")


def _require_viz_extra() -> None:
    """Usage error naming the `viz` extra when the charting deps are not installed."""
    missing = [name for name in _VIZ_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        raise typer.BadParameter(
            f"charts need the `viz` extra ({', '.join(missing)} not installed); run "
            "`uv sync --extra viz` (or `pip install 'world-model-optimizer[viz]'`) and retry"
        )


def _prepare_out_path(out: str | None) -> None:
    """Validate `--out` and create its parent directory BEFORE any (paid) eval work runs.

    Reports are written last, so a `--out` under a missing directory used to surface as a raw
    FileNotFoundError that discarded a finished run. `_eval_run_suite`/`_eval_run_grid` already
    created the parent at write time; doing it here makes every eval flow behave the same and
    fail before it spends anything.
    """
    if out is None:
        return
    path = Path(out)
    if path.is_dir():
        raise typer.BadParameter(
            f"--out {path} is a directory; pass the file to write (e.g. `--out {path}/report.json`)"
        )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        raise typer.BadParameter(f"cannot create --out directory {path.parent}: {err}") from None


def _resolve_eval_suite_or_usage(selector: str, examples_roots: list[str]) -> EvalSuite:
    """Resolve a suite name, turning the unknown/ambiguous `ValueError` into a usage error.

    Same contract as `_resolve_example`: a typo prints a clean box listing the available suites,
    never a traceback.
    """
    from wmo.engine.eval_suites import resolve_eval_suite

    try:
        return resolve_eval_suite(selector, examples_roots)
    except ValueError as err:
        raise typer.BadParameter(str(err)) from None


# Options only the closed-loop mode reads. In open-loop they used to be accepted and silently
# dropped, so the README's closed-loop command minus `--mode closed-loop` quietly ran a different
# (paid) evaluation. Same guard shape as `wmo optimize harness`'s world-model/harbor flag split.
_CLOSED_LOOP_ONLY_FLAGS = (
    ("name", "--name"),
    ("root", "--root"),
    ("k", "--k"),
    ("max_turns", "--max-turns"),
    ("harness", "--harness"),
    ("harness_backend", "--harness-backend"),
    ("eval_concurrency", "--eval-concurrency"),
    ("e2b_template", "--e2b-template"),
)


@app.command("eval")
def eval_(  # noqa: A001 - `eval` is the user-facing command name; the builtin isn't used here
    ctx: typer.Context,
    tokens: list[str] | None = _EVAL_TOKENS,
    mode: str = typer.Option(
        "open-loop",
        "--mode",
        help="open-loop: replay traces, score per-step fidelity (default). "
        "closed-loop: a live agent runs tasks with the world model as its environment.",
    ),
    prompt_file: str | None = typer.Option(
        None, "--prompt", help="Prompt file; default=BASE_ENV_PROMPT."
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="Provider running the model. Default: the worker role `wmo providers set` wrote to "
        "`.wmo/settings.toml`, else bedrock.",
    ),
    model: str | None = typer.Option(
        None,
        help="Canonical model type. Default: the configured worker role's model, else the "
        "flagship of whichever provider is in play (bedrock/claude-opus-4-8).",
    ),
    region: str | None = typer.Option(None, help="AWS region (Bedrock)."),
    chain: str | None = typer.Option(
        None, "--chain", help="Named failover chain from .wmo/fallback.toml (default: its default)."
    ),
    train_split: float | None = typer.Option(
        None,
        help="Train/holdout ratio per file (default: 0.8, or suite config). "
        "Must match the ratio the model was built with, or the holdout contains train traces.",
    ),
    val_frac: float | None = typer.Option(
        None,
        "--val-frac",
        help="`wmo eval grid`: validation fraction reserved for GEPA in a 3-way split so cells "
        "score only the leak-free test band (default: (1 - train_split)/2, matching build).",
    ),
    embed_dim: int | None = typer.Option(
        None, help="phi dimensionality for the offline embedder (default: 512, or suite config)."
    ),
    rag: bool | None = typer.Option(
        None, "--rag/--no-rag", help="Enable retrieval, or disable it for zero-shot replay."
    ),
    sample_turns: str | None = typer.Option(
        None, help="Turns scored per trace: all | sampled (5). Default: all, or suite config."
    ),
    seed: int | None = typer.Option(None, help="Seed for reproducible turn sampling."),
    top_k: int | None = typer.Option(
        None, help="Retrieved demos per step (default: 5, or suite config)."
    ),
    knowledge: bool | None = typer.Option(
        None,
        "--knowledge/--no-knowledge",
        help="Seed a knowledge base from the train split and render it into every prediction.",
    ),
    reasoning: bool | None = typer.Option(
        None,
        "--reasoning/--no-reasoning",
        help="Deliberate-then-answer output contract (explicit reasoning pass).",
    ),
    out: str | None = typer.Option(None, help="Optional path to write the full JSON report."),
    examples_root: str | None = typer.Option(
        None,
        help="Directory containing eval suites. Default: the downloaded benchmark data root "
        "(environment-capture-data/, where the shipped suites live).",
    ),
    results_root: str = typer.Option(
        f"{ARTIFACT_DIR}/evals", help="Local directory for named eval result JSON."
    ),
    limit: int = typer.Option(20, help="Rows to show for `wmo eval results`."),
    models: str | None = typer.Option(
        None,
        help="`wmo eval grid`: comma-separated Label:provider:model cells "
        "(e.g. 'Opus 4.8:bedrock:us.anthropic.claude-opus-4-8,GPT-5.5:openai:gpt-5.5').",
    ),
    gepa_prompts: str | None = typer.Option(
        None, help="`wmo eval grid`: dir of <label>.txt evolved prompts (enables +GEPA cells)."
    ),
    dataset_label: str | None = typer.Option(
        None, help="`wmo eval grid`: dataset name for the chart subtitle (default: suite id)."
    ),
    limit_traces: int | None = typer.Option(
        None, help="`wmo eval grid`: cap test traces (dry-run). Default: all."
    ),
    judge_model: str = typer.Option(
        "us.anthropic.claude-opus-4-8", help="`wmo eval grid`: pinned judge model (Bedrock)."
    ),
    name: str | None = typer.Option(
        None, "--name", help="World model for --mode closed-loop (default: the only built one)."
    ),
    root: str = typer.Option(
        ARTIFACT_DIR, help="Project dir holding world models (--mode closed-loop)."
    ),
    k: int = typer.Option(
        3, min=1, help="Closed-loop passes per task (means reported, never 1-pass)."
    ),
    max_turns: int | None = typer.Option(
        None, min=1, help="Closed-loop agent turn cap (default: 20, or the harness's own)."
    ),
    threshold: float = typer.Option(
        0.5, help="Agreement pass threshold on a task's k-pass success rate."
    ),
    harness: str | None = typer.Option(
        None, "--harness", help="Stored harness to run for `closed-loop` (default: baseline)."
    ),
    harness_backend: str = typer.Option(
        "local",
        "--harness-backend",
        help="Where the closed-loop harness PROCESS runs: local (in/from this process) or e2b "
        "(the pi-node harness inside pooled E2B sandboxes). The environment is always the "
        "world model.",
    ),
    eval_concurrency: int | None = typer.Option(
        None,
        "--eval-concurrency",
        min=0,
        help="Closed-loop (task, attempt) cells run at once. Default: 1 for local; "
        "0 (= all cells at once) for e2b.",
    ),
    e2b_template: str | None = typer.Option(
        None,
        "--e2b-template",
        envvar="WMO_E2B_TEMPLATE",
        help="Prebaked E2B sandbox template for --harness-backend e2b (default: "
        "$WMO_E2B_TEMPLATE; without one, sandboxes bootstrap node + the pi runner deps on "
        "first use).",
    ),
) -> None:
    """Score reconstruction fidelity, or run named example-local eval suites.

    Flows:
    - `wmo eval <trace files...>`: ad hoc replay scoring (open-loop, teacher-forced — the
      default mode).
    - `wmo eval <tasks.jsonl> --mode closed-loop`: a live agent runs tasks WITH the world model
      as its environment. `\\[models.agent]` selects a distinct agent provider when configured;
      otherwise the agent shares the world model's provider. `--harness-backend e2b` moves the
      pi-node harness process into pooled E2B sandboxes while the environment stays the world
      model. Score task success against gold assertions (see docs/reference/closed_loop.md).
    - `wmo eval list`: list named suites under `environment-capture-data/<task>/evals/`.
    - `wmo eval run <suite>`: run a suite and save a local JSON result.
    - `wmo eval results optional-suite`: summarize local suite results.
    - `wmo eval grid <suite>`: run the model x condition grid for a suite and chart it (needs the
      `viz` extra; consumes --models/--gepa-prompts/--dataset-label/--limit-traces/--judge-model
      and --val-frac).
    - `wmo eval grid-plot <result.json>...`: re-chart saved grid results, merged (`viz` extra).
    - `wmo eval grid-heatmap <result.json>...`: chart saved grid results as one cross-benchmark
      heatmap (`viz` extra).
    - `wmo eval agreement <a.json> <b.json>`: compare two closed-loop reports task-by-task
      (e.g. world-model vs real environment) — the outcome-agreement validity check.

    Open-loop scoring runs on the worker role `wmo providers set` writes to `.wmo/settings.toml`
    (bedrock/claude-opus-4-8 when no role is configured); `--provider`/`--model` override it.
    """
    from wmo.cli.eval_closed_loop import run_agreement, run_closed_loop
    from wmo.cli.ui import explicit_param as _explicit
    from wmo.telemetry import capture_eval_completed

    args = tokens or []
    suite_roots = (
        [str(root) for root in _benchmark_roots()] if examples_root is None else [examples_root]
    )
    if mode not in ("open-loop", "closed-loop"):
        raise typer.BadParameter(f"unknown --mode {mode!r}; choose open-loop or closed-loop")
    # Reject flags this flow will never read, before anything runs: a silently dropped
    # --harness/--k means the user paid for an open-loop run they did not ask for.
    if mode != "closed-loop":
        inapplicable = [flag for param, flag in _CLOSED_LOOP_ONLY_FLAGS if _explicit(ctx, param)]
        if inapplicable:
            raise typer.BadParameter(
                f"{', '.join(inapplicable)} apply only to `--mode closed-loop`; add that flag "
                "or drop them"
            )
    if (not args or args[0] != "agreement") and _explicit(ctx, "threshold"):
        raise typer.BadParameter(
            "--threshold applies only to `wmo eval agreement <a.json> <b.json>`; drop it"
        )
    # --out is written last (after the paid work); make sure it is writable first.
    _prepare_out_path(out)
    if mode == "closed-loop":
        if len(args) != 1 or args[0] in ("list", "run", "results", "agreement"):
            raise typer.BadParameter("usage: wmo eval <tasks.jsonl> --mode closed-loop")
        run_closed_loop(
            _console,
            tasks_file=args[0],
            name=name,
            root=root,
            k=k,
            max_turns=max_turns,
            out=out,
            harness=harness,
            harness_backend=harness_backend,
            eval_concurrency=eval_concurrency,
            e2b_template=e2b_template,
        )
        return
    if args and args[0] == "agreement":
        if len(args) != 3:
            raise typer.BadParameter("usage: wmo eval agreement <report_a.json> <report_b.json>")
        run_agreement(_console, report_a=args[1], report_b=args[2], threshold=threshold)
        return
    if args and args[0] == "list":
        if len(args) != 1:
            raise typer.BadParameter("usage: wmo eval list")
        _eval_list(suite_roots)
        return
    if args and args[0] == "results":
        if len(args) > 2:
            raise typer.BadParameter("usage: wmo eval results [suite]")
        suite_filter = args[1] if len(args) == 2 else None
        _eval_results(results_root, suite_roots, suite_filter, limit=limit)
        return
    if args and args[0] == "grid":
        if len(args) != 2:
            raise typer.BadParameter("usage: wmo eval grid <suite>")
        _eval_run_grid(
            args[1],
            examples_roots=suite_roots,
            results_root=results_root,
            models=models,
            gepa_prompts=gepa_prompts,
            dataset_label=dataset_label,
            limit_traces=limit_traces,
            judge_model=judge_model,
            region=region,
            train_split=train_split,
            seed=seed,
            top_k=top_k,
            embed_dim=embed_dim,
            sample_turns=sample_turns,
            val_frac=val_frac,
            out=out,
        )
        return
    if args and args[0] == "grid-plot":
        if len(args) < 2:
            raise typer.BadParameter("usage: wmo eval grid-plot <result.json> [<result.json>...]")
        _eval_grid_plot(args[1:], out=out, dataset_label=dataset_label)
        return
    if args and args[0] == "grid-heatmap":
        if len(args) < 2:
            raise typer.BadParameter(
                "usage: wmo eval grid-heatmap <result.json> [<result.json>...]"
            )
        _eval_grid_heatmap(args[1:], out=out)
        return
    if args and args[0] == "run":
        if len(args) != 2:
            raise typer.BadParameter("usage: wmo eval run <suite>")
        _eval_run_suite(
            args[1],
            examples_roots=suite_roots,
            results_root=results_root,
            prompt_file=prompt_file,
            provider_config=_worker_role_provider_config(provider, model, region),
            train_split=train_split,
            embed_dim=embed_dim,
            rag=rag,
            sample_turns=sample_turns,
            seed=seed,
            top_k=top_k,
            knowledge=knowledge,
            reasoning=reasoning,
            out=out,
        )
        return
    if not args:
        raise typer.BadParameter(
            "provide trace files, or use `wmo eval list`, `wmo eval run <suite>`, "
            "`wmo eval results`, `wmo eval grid <suite>`, `wmo eval grid-plot <result.json>`, "
            "`wmo eval grid-heatmap <result.json>`, or `wmo eval agreement <a.json> <b.json>`"
        )

    options = _eval_options(
        prompt_file=prompt_file,
        train_split=train_split,
        embed_dim=embed_dim,
        rag=rag,
        sample_turns=sample_turns,
        seed=seed,
        top_k=top_k,
        knowledge=knowledge,
        reasoning=reasoning,
    )
    report = _run_eval_files(
        [Path(f) for f in args],
        options,
        provider_config=_worker_role_provider_config(provider, model, region),
        chain=chain,
    )
    _require_scorable_steps(
        report,
        args,
        next_step=f"check the export with `wmo ingest --file {args[0]}`, or add "
        "`--mode closed-loop` for a tasks file",
    )
    _print_eval_report(report)
    if out:
        _write_ad_hoc_eval_report(Path(out), report)
    capture_eval_completed(
        mode="ad_hoc",
        file_count=len(args),
        scored_step_count=report.total_valid,
        rag_enabled=options.use_rag,
        sample_turns=options.sample_turns,
        train_split=options.train_split,
        top_k=options.top_k,
        root=ARTIFACT_DIR,
    )


def _eval_list(examples_roots: list[str]) -> None:
    from wmo.engine.eval_suites import discover_eval_suites

    suites = discover_eval_suites(examples_roots)
    if not suites:
        _console.print("[yellow]no eval suites found[/yellow]")
        return
    table = Table(title="Eval suites")
    table.add_column("Suite", no_wrap=True)
    table.add_column("Files")
    table.add_column("Split")
    table.add_column("Description")
    for suite in suites:
        table.add_row(
            suite.id,
            ", ".join(suite.config.files),
            f"{suite.config.train_split:.2f}",
            suite.config.description or "",
        )
    _console.print(table)


def _eval_results(
    results_root: str,
    examples_roots: list[str],
    suite_filter: str | None,
    *,
    limit: int,
) -> None:
    from wmo.engine.eval_suites import list_eval_results, resolve_eval_suite

    resolved_suite = suite_filter
    if suite_filter is not None:
        try:
            resolved_suite = resolve_eval_suite(suite_filter, examples_roots).id
        except ValueError:
            resolved_suite = suite_filter
    summaries = list_eval_results(results_root, resolved_suite, limit=limit)
    if not summaries:
        _console.print("[yellow]no eval results found[/yellow]")
        return
    table = Table(title="Eval results")
    table.add_column("Suite", no_wrap=True)
    table.add_column("Run")
    table.add_column("Started")
    table.add_column("Model")
    table.add_column("Fidelity", justify="right")
    table.add_column("Steps", justify="right")
    table.add_column("Path")
    for summary in summaries:
        steps = str(summary.total_steps)
        if summary.total_invalid:
            steps += f" ({summary.total_invalid} inv)"
        table.add_row(
            summary.suite,
            summary.run_id[:8],
            summary.started_at,
            summary.model,
            f"{summary.overall_fidelity:.3f}±{summary.overall_std:.3f}",
            steps,
            str(summary.path),
        )
    _console.print(table)


# Default grid: one bar family per serving model. Qwen-AgentWorld is self-hosted (openai-compatible
# vLLM via OPENAI_BASE_URL); the rest are frontier models the registry builds directly.
_DEFAULT_GRID_MODELS = (
    "Opus 4.8:bedrock:us.anthropic.claude-opus-4-8",
    "Haiku 4.5:bedrock:us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "GPT-5.5:openai:gpt-5.5",
    "GPT-5.4 Mini:openai:gpt-5.4-mini",
    "Qwen-AgentWorld:openai:Qwen/Qwen-AgentWorld-35B-A3B",
)


def _parse_model_specs(models: str | None) -> list[ModelSpec]:
    """Parse "Label:provider:model[,...]" into ModelSpecs (default set when None).

    The provider is validated and the model resolved through the shared catalog
    (`resolve_provider_model`), so a friendly model type resolves to its canonical wire id while an
    unknown (self-hosted) id passes through unchanged, and a bad provider fails at parse time.
    """
    from wmo.evals.grid import ModelSpec
    from wmo.providers import ProviderKind
    from wmo.providers.models import resolve_provider_model

    raw = models.split(",") if models else list(_DEFAULT_GRID_MODELS)
    specs: list[ModelSpec] = []
    for entry in raw:
        parts = entry.split(":", 2)
        if len(parts) != 3 or not all(p.strip() for p in parts):
            raise typer.BadParameter(f"bad --models entry {entry!r}; want 'Label:provider:model'")
        label, provider_str, model_str = (p.strip() for p in parts)
        try:
            kind = ProviderKind(provider_str)
        except ValueError:
            kinds = ", ".join(k.value for k in ProviderKind)
            raise typer.BadParameter(
                f"unknown provider {provider_str!r} in --models {entry!r}; want one of {kinds}"
            ) from None
        specs.append(ModelSpec(label, kind.value, resolve_provider_model(kind, model_str).model_id))
    return specs


def _grid_output_paths(out: str | None, default_json: Path) -> tuple[Path, Path]:
    """Result-JSON and chart-PNG destinations for a grid run.

    With `--out`, the JSON and PNG share the stem but ALWAYS take distinct suffixes, so passing
    `--out foo.json` can never make the PNG write clobber the result JSON at the same path (and
    `--out foo.png` still lands the JSON next to it). Without `--out`, use the default JSON dest and
    its `.png` sibling.
    """
    if out is None:
        return default_json, default_json.with_suffix(".png")
    base = Path(out)
    return base.with_suffix(".json"), base.with_suffix(".png")


def _eval_run_grid(  # noqa: PLR0913 - a CLI seam threading grid options; each maps to one flag
    selector: str,
    *,
    examples_roots: list[str],
    results_root: str,
    models: str | None,
    gepa_prompts: str | None,
    dataset_label: str | None,
    limit_traces: int | None,
    judge_model: str,
    region: str | None,
    train_split: float | None,
    seed: int | None,
    top_k: int | None,
    embed_dim: int | None,
    sample_turns: str | None,
    val_frac: float | None,
    out: str | None,
) -> None:
    """Run the model x condition grid for a suite, write result JSON + a fidelity bar chart PNG."""
    from wmo.engine.prompts import BASE_ENV_PROMPT
    from wmo.evals.grid import run_grid
    from wmo.evals.grid_plot import plot_grid

    suite = _resolve_eval_suite_or_usage(selector, examples_roots)
    # The chart is the last thing written, so probe its deps before the grid spends anything.
    _require_viz_extra()
    specs = _parse_model_specs(models)
    prompt_dir = Path(gepa_prompts) if gepa_prompts else None
    prompt_map: dict[str, str] | None = None
    if prompt_dir is not None:
        prompt_map = {
            s.label: str(prompt_dir / f"{s.label}.txt")
            for s in specs
            if (prompt_dir / f"{s.label}.txt").exists()
        }
    cfg = suite.config
    result = run_grid(
        suite_name=suite.id,
        files=[str(p) for p in suite.resolve_files()],
        models=specs,
        gepa_prompts=prompt_map,
        base_prompt=BASE_ENV_PROMPT,
        judge_provider="bedrock",
        judge_model=judge_model,
        judge_region=region,
        train_split=train_split if train_split is not None else cfg.train_split,
        val_frac=val_frac,
        top_k=top_k if top_k is not None else cfg.top_k,
        seed=seed if seed is not None else cfg.seed,
        sample_turns=sample_turns or cfg.sample_turns,
        embed_dim=embed_dim if embed_dim is not None else cfg.embed_dim,
        max_holdout_traces=limit_traces,
    )
    for cell in result.cells:
        cost = f" ${cell.cost_usd:.2f}" if cell.cost_usd else ""
        _console.print(
            f"  {cell.model_label:16} {cell.condition_label:14} "
            f"fidelity={cell.fidelity:.3f} err_flag={cell.error_flag_acc:.3f} "
            f"n={cell.n_steps}{cost}"
        )
    run_id = uuid4().hex
    default_dest = Path(results_root) / "grid" / f"{suite.name}-{run_id}.json"
    dest, png = _grid_output_paths(out, default_dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    _console.print(f"wrote grid result -> {dest}")
    png.parent.mkdir(parents=True, exist_ok=True)
    plot_grid(
        result,
        png,
        dataset_label=dataset_label or suite.id,
        n_test_traces=result.total_test_traces,
    )
    _console.print(f"wrote grid chart  -> {png}")


def _eval_grid_plot(paths: list[str], *, out: str | None, dataset_label: str | None) -> None:
    """Merge one or more grid result JSONs and render a single combined fidelity chart.

    Lets a self-hosted model's grid (run in its own process, since its OpenAI base URL is
    process-global) be combined with the API-model grid into one chart - and re-plots any saved
    result without re-running the eval.
    """
    from wmo.evals.grid import GridResult, merge_results
    from wmo.evals.grid_plot import plot_grid

    _require_viz_extra()
    results = [GridResult.model_validate_json(Path(p).read_text(encoding="utf-8")) for p in paths]
    merged = merge_results(results)
    for cell in merged.cells:
        cost = f" ${cell.cost_usd:.2f}" if cell.cost_usd else ""
        _console.print(
            f"  {cell.model_label:16} {cell.condition_label:14} "
            f"fidelity={cell.fidelity:.3f} err_flag={cell.error_flag_acc:.3f} "
            f"n={cell.n_steps}{cost}"
        )
    # Force a .png suffix so `--out foo.json` writes a real PNG file, never a PNG mislabeled .json.
    png = Path(out).with_suffix(".png") if out else Path(paths[0]).with_suffix(".merged.png")
    plot_grid(
        merged,
        png,
        dataset_label=dataset_label or merged.suite,
        n_test_traces=merged.total_test_traces,
    )
    _console.print(f"wrote merged grid chart -> {png}")


def _eval_grid_heatmap(paths: list[str], *, out: str | None) -> None:
    """Render the whole grid as one heatmap from result JSONs (merged per suite).

    Accepts any mix of API/Qwen result JSONs; same-suite results are merged into one 5-model row
    set, then all suites become the heatmap's columns (rows = model x condition).
    """
    from wmo.evals.grid import GridResult, merge_results
    from wmo.evals.grid_plot import plot_grid_heatmap

    _require_viz_extra()
    by_suite: dict[str, list[GridResult]] = {}
    for p in paths:
        res = GridResult.model_validate_json(Path(p).read_text(encoding="utf-8"))
        by_suite.setdefault(res.suite, []).append(res)
    merged = {suite: merge_results(rs) for suite, rs in by_suite.items()}
    png = Path(out).with_suffix(".png") if out else Path("grid-heatmap.png")
    plot_grid_heatmap(merged, png)
    _console.print(f"wrote grid heatmap -> {png} ({len(merged)} benchmarks)")


def _eval_run_suite(
    selector: str,
    *,
    examples_roots: list[str],
    results_root: str,
    prompt_file: str | None,
    provider_config: ProviderConfig,
    train_split: float | None,
    embed_dim: int | None,
    rag: bool | None,
    sample_turns: str | None,
    seed: int | None,
    top_k: int | None,
    knowledge: bool | None,
    reasoning: bool | None,
    out: str | None,
) -> None:
    from wmo.engine.eval_suites import result_path
    from wmo.telemetry import capture_eval_completed, settings_root_from_results_root

    suite = _resolve_eval_suite_or_usage(selector, examples_roots)
    suite_prompt = suite.resolve_prompt()
    options = _eval_options(
        prompt_file=prompt_file or (str(suite_prompt) if suite_prompt is not None else None),
        train_split=train_split if train_split is not None else suite.config.train_split,
        embed_dim=embed_dim if embed_dim is not None else suite.config.embed_dim,
        rag=rag if rag is not None else not suite.config.no_rag,
        sample_turns=sample_turns or suite.config.sample_turns,
        seed=seed if seed is not None else suite.config.seed,
        top_k=top_k if top_k is not None else suite.config.top_k,
        knowledge=knowledge if knowledge is not None else suite.config.knowledge,
        reasoning=reasoning if reasoning is not None else suite.config.reasoning,
    )
    files = suite.resolve_files()
    if not files:
        raise typer.BadParameter(
            f"eval suite {suite.id} lists no trace files; set `files` in {suite.path}"
        )
    report = _run_eval_files(files, options, provider_config=provider_config)
    # A suite run is saved under --results-root and resurfaces in `wmo eval results`, so a
    # zero-step scorecard must fail here rather than persist as a real 0.000 measurement.
    _require_scorable_steps(
        report,
        [str(path) for path in files],
        next_step=f"check the corpus `files` in {suite.path} with `wmo ingest --file {files[0]}`",
    )
    _print_eval_report(report)

    run_id = uuid4().hex
    started_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    destination = Path(out) if out else result_path(results_root, suite, run_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "started_at": started_at,
        "suite": suite.id,
        "suite_path": str(suite.path),
        "suite_config": suite.config.model_dump(mode="json"),
        # The RESOLVED backend, not the raw flags: with the flags omitted these come from the
        # configured worker role, and a result JSON that recorded `null` could not say which
        # model produced the fidelity number.
        "config": {
            "provider": provider_config.kind.value,
            "model": provider_config.model_type or provider_config.model,
            "region": provider_config.region,
            "prompt": options.prompt_file,
            "files": [str(path) for path in files],
            "train_split": options.train_split,
            "top_k": options.top_k,
            "sample_turns": options.sample_turns,
            "seed": options.seed,
            "rag": options.use_rag,
            "embed_dim": options.embed_dim,
            "knowledge": options.knowledge,
            "reasoning": options.reasoning,
        },
        "report": _eval_report_payload(report),
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _console.print(f"wrote eval result -> {destination}")
    capture_eval_completed(
        mode="suite",
        file_count=len(files),
        scored_step_count=report.total_valid,
        rag_enabled=options.use_rag,
        sample_turns=options.sample_turns,
        train_split=options.train_split,
        top_k=options.top_k,
        root=settings_root_from_results_root(results_root),
    )


def _eval_options(
    *,
    prompt_file: str | None,
    train_split: float | None,
    embed_dim: int | None,
    rag: bool | None,
    sample_turns: str | None,
    seed: int | None,
    top_k: int | None,
    knowledge: bool | None = None,
    reasoning: bool | None = None,
) -> _EvalOptions:
    from wmo.engine.build import DEFAULT_TRAIN_SPLIT

    split = DEFAULT_TRAIN_SPLIT if train_split is None else train_split
    dim = 512 if embed_dim is None else embed_dim
    retrieval = True if rag is None else rag
    turns = "all" if sample_turns is None else sample_turns
    rng_seed = 0 if seed is None else seed
    demos = 5 if top_k is None else top_k
    if not 0.0 < split < 1.0:
        raise typer.BadParameter("--train-split must be between 0 and 1")
    if dim <= 0:
        raise typer.BadParameter("--embed-dim must be positive")
    if demos < 0:
        raise typer.BadParameter("--top-k must be >= 0")
    if turns not in {"all", "sampled"}:
        raise typer.BadParameter("--sample-turns must be one of: all, sampled")
    return _EvalOptions(
        prompt_file=prompt_file,
        train_split=split,
        embed_dim=dim,
        use_rag=retrieval,
        sample_turns=turns,
        seed=rng_seed,
        top_k=demos,
        knowledge=bool(knowledge),
        reasoning=bool(reasoning),
    )


def _run_eval_files(
    files: list[Path],
    options: _EvalOptions,
    *,
    provider_config: ProviderConfig,
    chain: str | None = None,
) -> EvalReport:
    import wmo.providers as providers
    from wmo.engine.prompts import BASE_ENV_PROMPT
    from wmo.evals.open_loop import OpenLoopEval
    from wmo.optimize.judge import RubricJudge
    from wmo.retrieval import HashingEmbedder

    for path in files:
        if not path.exists():
            raise typer.BadParameter(f"trace file not found: {path}")
        if not path.is_file():
            raise typer.BadParameter(
                f"not a trace file: {path} is a directory; pass the export itself "
                f"(e.g. `wmo eval {path}/traces.otel.jsonl`)"
            )
    # Name the backend: the number below is only comparable against runs on the same model, and
    # with no flags the model comes from settings rather than the command line.
    _console.print(
        f"scoring with {provider_config.kind.value} "
        f"({provider_config.model_type or provider_config.model})"
    )
    # A missing/unknown chain, or a multi-chain fallback.toml with no `default`, is a usage
    # error: the message already names the file, so it must not arrive as a traceback.
    try:
        llm = providers.provider_or_chain(provider_config, chain=chain)
    except (ValueError, FileNotFoundError) as err:
        raise typer.BadParameter(str(err)) from None
    if isinstance(llm, providers.WaterfallProvider):
        _console.print("failover chain active (.wmo/fallback.toml) — world-model calls only")
    prompt = (
        Path(options.prompt_file).read_text(encoding="utf-8")
        if options.prompt_file
        else BASE_ENV_PROMPT
    )
    embedder = HashingEmbedder(dim=options.embed_dim) if options.use_rag else None
    # The judge is the metric: it stays PINNED to the single requested backend and never rides
    # the failover chain — a judge that silently switches models mid-run makes fidelity numbers
    # incomparable across steps. World-model prediction calls (above) may fail over freely.
    scorer = RubricJudge(providers.get_provider(provider_config))
    evaluation = OpenLoopEval(
        files,
        prompt,
        llm,
        scorer,
        embedder=embedder,
        train_split=options.train_split,
        top_k=options.top_k,
        sample_turns=options.sample_turns,
        seed=options.seed,
        knowledge=options.knowledge,
        reasoning=options.reasoning,
    )
    return evaluation.run()


def _require_scorable_steps(report: EvalReport, paths: list[str], *, next_step: str) -> None:
    """Refuse to print (or persist) a 0.000 scorecard for input that held nothing to score.

    `evaluate_files` skips a file that yields no OTel GenAI traces, so a wrong file type, a
    tasks.jsonl missing `--mode closed-loop`, or a train_split that leaves no holdout all used to
    render as a plausible `OVERALL fidelity=0.000 over 0 held-out steps` at exit 0. `wmo eval run
    <suite>` additionally saved that scorecard as a durable result, so it resurfaced later in
    `wmo eval results`; both flows now stop here instead.

    `next_step` is the caller's flow-specific remedy, since the ad-hoc path can suggest
    `--mode closed-loop` for a tasks file and the suite path cannot.
    """
    if report.total_steps:
        return
    listed = ", ".join(paths)
    if not report.per_file:
        raise typer.BadParameter(f"no OTel GenAI traces in {listed}; {next_step}")
    raise typer.BadParameter(
        f"no held-out steps to score in {listed}; lower `--train-split` (currently reserving "
        "every trace for training) or pass a larger corpus"
    )


def _print_eval_report(report: EvalReport) -> None:
    for name, rep in report.per_file.items():
        _console.print(f"  {name:28} {rep.summary()}")
    invalid = f" ({report.total_invalid} judge-invalid excluded)" if report.total_invalid else ""
    _console.print(
        f"[bold]OVERALL[/bold] fidelity={report.overall_fidelity:.3f}±{report.overall_std:.3f} "
        f"over {report.total_steps} held-out steps{invalid}"
    )


def _write_ad_hoc_eval_report(path: Path, report: EvalReport) -> None:
    path.write_text(
        json.dumps({n: r.model_dump(mode="json") for n, r in report.per_file.items()}, indent=2),
        encoding="utf-8",
    )
    _console.print(f"wrote full report -> {path}")


def _eval_report_payload(report: EvalReport) -> JsonObject:
    from wmo.optimize.judge import JUDGE_VERSION

    return {
        "judge_version": JUDGE_VERSION,
        "overall_fidelity": report.overall_fidelity,
        "overall_std": report.overall_std,
        "total_steps": report.total_steps,
        "total_invalid": report.total_invalid,
        "per_file": {name: rep.model_dump(mode="json") for name, rep in report.per_file.items()},
    }


@app.command("knowledge")
def knowledge_(
    name: str = typer.Option(None, "--name", help="World model (default: the only one)."),
    root: str = typer.Option(
        ARTIFACT_DIR,
        help="Project dir. Shipped example models are found too while this is the default "
        "project dir; point it elsewhere to search that root alone.",
    ),
) -> None:
    """Show a model's knowledge base: the env's canonical facts, a folder of editable markdown.

    The printed directory IS the editing interface, open it in any editor. `rules.md`/
    `entities.md`/`schemas.md` are seeded at build (with knowledge enabled); `learned.md` collects
    the env's own cross-session notes; `grounded.md` caches web-search groundings. Models are
    resolved exactly as `wmo demo`/`wmo play` resolve them, so a shipped example needs no `--root`.
    """
    from wmo.engine.knowledge import KnowledgeBase

    store_root, resolved = _resolve_model_any(name, root)
    model_dir = WorldModelStore(str(store_root)).resolve(resolved)
    kb = KnowledgeBase(ArtifactPaths(model_dir).knowledge)
    _console.print(f"[bold]{escape(str(kb.directory))}[/bold]")
    build_with_knowledge = f"`wmo build --name {resolved} --file <traces> --knowledge`"
    if kb.is_empty:
        state = "is empty" if kb.directory.is_dir() else "does not exist yet"
        _console.print(
            f"(that directory {state}) seed one with {build_with_knowledge}, which extracts "
            "rules.md / entities.md / schemas.md from the train traces"
        )
        return
    if not load_config(model_dir).knowledge:
        # The files are on disk but `WorldModel.load` gates the KB on config.knowledge, so at the
        # default fidelity nothing here reaches the env prompt. Say so instead of printing them
        # back as if they were live.
        _console.print(
            f"[yellow]inert[/yellow]: {resolved!r} was built without a knowledge base, so "
            "`wmo demo` / `wmo play` / `wmo serve` never render these files into the env "
            f"prompt. Activate them by rebuilding with {build_with_knowledge}, or by setting "
            f"`knowledge = true` in {escape(str(ArtifactPaths(model_dir).config))}."
        )
    # Knowledge is hand-edited markdown: `[/items]`, `list[str]` and `[text](url)` are ordinary
    # content, so it is escaped rather than parsed as rich markup (which would crash on the
    # first, and silently delete the other two).
    for file_name, content in kb.files().items():
        _console.print(f"\n[bold]## {escape(file_name)}[/bold]")
        _console.print(escape(content.strip()))


@scenarios_app.command("build")
def scenarios_build(
    file: str = typer.Option(..., "--file", help="Path to exported traces (OTLP-JSON / JSONL)."),
    out: str = typer.Option("scenarios.json", "--out", help="Where to write the scenario set."),
    budget: int = typer.Option(20, help="Number of scenarios to construct."),
    k: int = typer.Option(None, help="Cluster count (default: sqrt(corpus size))."),
    limit: int = typer.Option(None, help="Only use the first N ingested traces (cost control)."),
    provider: str = typer.Option(
        None,
        "--provider",
        help=(
            "Pin ONE LLM for every role (facets/naming/synthesis/validation). When omitted, "
            "roles resolve from .wmo/settings.toml \\[models.worker|judge|summary]."
        ),
    ),
    model: str = typer.Option(None, help="Model id (pins all roles, like --provider)."),
    region: str = typer.Option(None, help="AWS region (Bedrock)."),
    embed_provider: str = typer.Option(
        "hashing",
        help=(
            "Facet embedder: hashing (offline but lexical-only; clusters by wording, not "
            "meaning; prefer a semantic embedder for real corpora) | local (in-process "
            "Qwen3, semantic and credential-free; pass --embed-dim 1024) | bedrock | "
            "openai | azure."
        ),
    ),
    embed_model: str = typer.Option(None, help="Embeddings model id / Azure deployment."),
    embed_dim: int = typer.Option(512, help="Embedding dimensionality."),
    seed: int = typer.Option(0, help="Clustering seed."),
) -> None:
    """Distill a trace corpus into a representative scenario set (facets -> cluster -> select).

    Writes a `ScenarioSet` JSON: scenarios (task, seed state, checklist, weight, provenance),
    the named clusters they came from, and the corpus-coverage number that justifies them.
    """
    from wmo.scenarios import FacetExtractor, ScenarioBuildConfig, build_scenario_set

    traces = _ingest_scenario_corpus(file)
    if limit is not None:
        traces = traces[:limit]
    if not traces:
        raise typer.BadParameter(f"no traces ingested from {file}")
    summary_llm, worker_llm, judge_llm = _scenario_role_llms(provider, model, region)
    embedder = _resolve_scenario_embedder(embed_provider, embed_model, embed_dim, region)

    _console.print(f"extracting facets for {len(traces)} traces…")
    facets = FacetExtractor(summary_llm).extract_all(traces)
    config = ScenarioBuildConfig(budget=budget, k=k, seed=seed)
    scenario_set = build_scenario_set(
        traces, facets, worker_llm, embedder, config, judge_provider=judge_llm
    )
    scenario_set.save(out)

    table = Table(title="Scenario set")
    table.add_column("Cluster", no_wrap=True)
    table.add_column("Scenario task")
    table.add_column("Weight", justify="right")
    table.add_column("Source", no_wrap=True)
    for scenario in scenario_set.scenarios:
        source = scenario.failure_category or scenario.source_outcome.value
        table.add_row(scenario.cluster_name, scenario.task[:80], f"{scenario.weight:.3f}", source)
    _console.print(table)
    _console.print(
        f"{len(scenario_set.scenarios)} scenarios from {scenario_set.corpus_traces} traces; "
        f"coverage {scenario_set.corpus_coverage:.0%} at tau={scenario_set.coverage_tau} -> {out}"
    )


@scenarios_app.command("verify")
def scenarios_verify(
    scenarios_file: str = typer.Argument(..., help="Scenario set JSON from `wmo scenarios build`."),
    file: str = typer.Option(..., "--file", help="Source trace corpus (for back-agreement)."),
    name: str = typer.Option(None, "--name", help="World model to roll against."),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir holding world models."),
    provider: str = typer.Option(None, "--provider", help="Override serve provider kind."),
    model: str = typer.Option(None, help="Override canonical serve model type."),
    region: str = typer.Option(None, help="AWS region (Bedrock)."),
    max_steps: int = typer.Option(12, help="Rollout step budget per scenario."),
    drop: bool = typer.Option(False, "--drop", help="Write back only verified scenarios."),
) -> None:
    """Closed-loop verification: back-agreement on source traces + solvability rollouts.

    Loads the world model (optionally overriding its serve provider with a cheaper model), rolls a
    baseline LLM agent on every scenario, and grades episodes against each scenario's checklist.
    With `--drop`, unverified scenarios are removed from the set in place.
    """
    import wmo.providers as providers
    from wmo.engine.world_model import WorldModel
    from wmo.env.llm_agent import LLMAgent
    from wmo.providers.retry import wrap_provider_with_retries
    from wmo.scenarios import ChecklistJudge, verify_scenarios

    scenario_set = _load_scenario_set(scenarios_file)
    traces = _ingest_scenario_corpus(file)
    if provider is not None or model is not None:
        store = WorldModelStore(root)
        model_dir = store.resolve(_resolve_name(store, name))
        override = _worker_role_provider_config(provider, model, region)
        llm = wrap_provider_with_retries(providers.get_provider(override))
        world_model = WorldModel.load(str(model_dir), llm)
    else:
        world_model, _resolved_name, llm = _load_model(name, root)

    # The rollout agent takes the worker role and the grader the judge role when configured in
    # settings (judge should differ in family from the generator); both fall back to the world
    # model's serve provider, which was the only behavior before roles existed.
    worker_config = _role_provider_config("worker", region)
    judge_config = _role_provider_config("judge", region)
    agent_llm = providers.get_provider(worker_config) if worker_config else llm
    judge_llm = providers.get_provider(judge_config) if judge_config else llm
    report = verify_scenarios(
        scenario_set,
        traces,
        world_model,
        LLMAgent(agent_llm),
        ChecklistJudge(judge_llm),
        max_steps=max_steps,
    )
    table = Table(title="Scenario verification")
    table.add_column("Scenario", no_wrap=True)
    table.add_column("Back-agree")
    table.add_column("Solvable")
    table.add_column("Pass rate", justify="right")
    for verdict in report.verdicts:
        if verdict.back_agreement is None:
            agree = "-"
        else:
            agree = "yes" if verdict.back_agreement else "NO"
        table.add_row(
            verdict.scenario_id,
            agree,
            "yes" if verdict.solvable else "NO",
            f"{verdict.rollout_pass_rate:.2f}",
        )
    _console.print(table)
    _console.print(
        f"back-agreement {report.back_agreement_rate:.0%}, solvable {report.solvable_rate:.0%} "
        f"over {len(report.verdicts)} scenarios"
    )
    if drop:
        verified = {v.scenario_id for v in report.verdicts if v.ok}
        scenario_set.retain(verified)
        scenario_set.save(scenarios_file)
        _console.print(
            f"kept {len(scenario_set.scenarios)} verified scenarios "
            f"(weights renormalized, coverage reset) -> {scenarios_file}"
        )


def _ingest_scenario_corpus(file: str) -> list[Trace]:
    """Ingest a `--file` trace corpus for the scenarios commands as a validated CLI input.

    `--file` is the only required option on `wmo scenarios build`, so a mistyped path is the
    likeliest first-run mistake. Guard it here rather than letting `Path.read_text` raise, which
    reaches the user as a stdlib FileNotFoundError/IsADirectoryError traceback.
    """
    from wmo.ingest import get_adapter

    path = Path(file)
    if path.is_dir():
        raise typer.BadParameter(
            f"--file {file} is a directory; point it at the trace file itself, "
            f"e.g. `--file {Path(file) / 'traces.otel.jsonl'}`"
        )
    if not path.exists():
        raise typer.BadParameter(
            f"--file {file} does not exist; pass an exported OTel-GenAI corpus, or fetch a "
            "benchmark one with `wmo download <benchmark>`"
        )
    try:
        return get_adapter("otel-genai").from_file(file)
    except (OSError, UnicodeDecodeError) as exc:
        raise _unreadable_input(f"--file {file}", path, exc) from exc


def _load_scenario_set(scenarios_file: str) -> ScenarioSet:
    """Load a `ScenarioSet` argument, reporting a missing or malformed file as a usage error.

    The raw failures are a stdlib FileNotFoundError/IsADirectoryError and a pydantic
    ValidationError that sends the user to pydantic's docs; neither says the file is supposed to
    be the output of `wmo scenarios build`.
    """
    from wmo.scenarios import ScenarioSet

    path = Path(scenarios_file)
    build_hint = (
        f"build one with `wmo scenarios build --file <traces.jsonl> --out {scenarios_file}`"
    )
    if path.is_dir():
        raise typer.BadParameter(
            f"scenario set {scenarios_file} is a directory; pass the JSON file written by "
            "`wmo scenarios build --out <scenarios.json>`"
        )
    if not path.exists():
        raise typer.BadParameter(f"scenario set {scenarios_file} does not exist; {build_hint}")
    try:
        return ScenarioSet.load(path)
    except ValidationError as exc:
        raise typer.BadParameter(
            f"{scenarios_file} is not a scenario set written by `wmo scenarios build` "
            f"({exc.error_count()} validation error(s), first: {exc.errors()[0]['msg']}); "
            f"{build_hint}"
        ) from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise _unreadable_input(f"scenario set {scenarios_file}", path, exc) from exc


def _unreadable_input(
    label: str, path: Path, exc: OSError | UnicodeDecodeError
) -> typer.BadParameter:
    """Report the two read failures an exists/is-dir check cannot predict as a usage error.

    A path that passes the shape checks can still fail inside the read: no permission on it, or
    bytes that are not UTF-8 (a compressed or binary export). Both otherwise reach the user as a
    stdlib traceback, which is the thing these guards exist to prevent.
    """
    if isinstance(exc, UnicodeDecodeError):
        return typer.BadParameter(
            f"{label} is not UTF-8 text; pass the decompressed JSON/JSONL export "
            f"(`file {path}` says what it actually is)"
        )
    return typer.BadParameter(
        f"{label} could not be read ({exc.strerror or exc}); "
        f"`ls -l {path}` shows its owner and mode"
    )


def _provider_kind(provider: str) -> ProviderKind:
    """The `ProviderKind` a `--provider` flag names, as a usage error when it names none."""
    from wmo.providers import ProviderKind

    try:
        return ProviderKind(provider)
    except ValueError:
        kinds = ", ".join(k.value for k in ProviderKind)
        raise typer.BadParameter(f"unknown provider {provider!r}; choose one of: {kinds}") from None


def _provider_config(provider: str, model: str, region: str | None) -> ProviderConfig:
    from wmo.providers import ProviderConfig
    from wmo.providers.models import resolve_provider_model

    kind = _provider_kind(provider)
    spec = resolve_provider_model(kind, model)
    return ProviderConfig(
        kind=kind,
        model_type=spec.model_type,
        model=spec.model_id,
        region=region,
    )


# The backend a worker-role command falls back to when the project configured no
# `[models.worker]` role at all. Never a substitute for a configured role.
_DEFAULT_WORKER_PROVIDER = "bedrock"
_DEFAULT_WORKER_MODEL = "claude-opus-4-8"


def _default_model_for_provider(kind: ProviderKind) -> str:
    """`kind`'s flagship: the model to run when neither a flag nor a role named one.

    A default model belongs to ONE backend — pairing `--provider openai` with bedrock's
    `claude-opus-4-8` sends a model OpenAI has never heard of. `openrouter` and `tinker` publish
    no built-in rows (nothing can derive an operator's route or weights path), so they must be
    told which model to run.
    """
    from wmo.providers.models import model_types_for_provider

    catalog = model_types_for_provider(kind)
    if not catalog:
        raise typer.BadParameter(
            f"provider {kind.value!r} has no default model; pass --model <model>, or run "
            f"`wmo providers set --provider {kind.value} --model <model>` to configure the "
            f"worker role"
        )
    return catalog[0]


def _role_provider_config(role: str, region: str | None) -> ProviderConfig | None:
    """ProviderConfig for a settings-defined model role, or None when the role isn't configured.

    Roles live in `.wmo/settings.toml` under `[models.worker|judge|summary]`; unset judge/summary
    fall back to worker (see `ModelsSettings.resolve`). A role's stored region wins over the
    generic `--region` flag — the flag also feeds the embedder, and e.g. a judge pinned to the
    one region where its model is enabled must not follow it.
    """
    configured = load_settings_or_abort().models.resolve(role)
    if configured is None:
        return None
    config = _provider_config(configured.provider, configured.model, configured.region or region)
    return config.model_copy(
        update={"endpoint": configured.endpoint, "deployment": configured.deployment}
    )


def _azure_deployment_for_model(
    configured: ProviderConfig, spec: ProviderModel, deployment: str | None
) -> str:
    """The Azure deployment to invoke after `--model` moved the role off its configured one.

    On Azure the wire `model` IS the deployment name (`AzureOpenAIProvider._deployment`), so a
    role's deployment names the model being replaced. Keeping it would call the old model and
    report the new one. Guessing the new one is no better: an operator's deployment name is not
    derivable from a model id, and a wrong guess 404s on every prediction, which `wmo eval`
    reports as a silent `fidelity=0.000` at exit 0 — the defect this whole path exists to stop.
    So a model swap on Azure has to be told which deployment serves it.
    """
    if deployment is not None:
        return deployment
    if configured.deployment is None:
        # Nothing configured to contradict, so derive it from the model as
        # `_worker_provider_config` does; the role could not have been called without one anyway.
        return spec.model_type
    if configured.deployment in (spec.model_type, spec.model_id):
        return configured.deployment
    raise typer.BadParameter(
        f"the configured azure worker serves {configured.model} from deployment "
        f"{configured.deployment!r}, and on Azure the deployment name is what is actually "
        f"invoked, so --model {spec.model_type} needs the deployment that serves it. Run "
        f"`wmo providers set --provider azure --model {spec.model_type} "
        f"--deployment <deployment>` to point the worker role at it."
    )


def _worker_role_provider_config(
    provider: str | None,
    model: str | None,
    region: str | None,
    *,
    deployment: str | None = None,
) -> ProviderConfig:
    """The backend for a worker-role call: explicit flags, then the worker role, then the default.

    `wmo providers set` writes `[models.worker]` and is step 1 of the documented getting-started
    path, so a command that ignored it would run against a provider the user never configured.
    Each field falls back independently, and a `--provider` naming a DIFFERENT backend than the
    configured role drops that role's model and connection fields, which belong to the backend it
    replaced — the model then comes from the NEW backend's catalog, never from bedrock's.
    """
    from wmo.providers import ProviderKind
    from wmo.providers.models import resolve_provider_model

    configured = _role_provider_config("worker", region)
    if configured is None or (provider is not None and provider != configured.kind.value):
        kind = _provider_kind(provider or _DEFAULT_WORKER_PROVIDER)
        config = _provider_config(kind.value, model or _default_model_for_provider(kind), region)
    elif model is None:
        config = configured
    else:
        spec = resolve_provider_model(configured.kind, model)
        if spec.model_id == configured.model:
            # Re-stating the role's own model is not a model change: leave its connection alone.
            config = configured
        else:
            update: dict[str, object] = {"model_type": spec.model_type, "model": spec.model_id}
            if configured.kind is ProviderKind.AZURE_OPENAI:
                update["deployment"] = _azure_deployment_for_model(configured, spec, deployment)
            config = configured.model_copy(update=update)
    if deployment is not None:
        config = config.model_copy(update={"deployment": deployment})
    return config


def _scenario_role_llms(
    provider: str | None, model: str | None, region: str | None
) -> tuple[Provider, Provider, Provider]:
    """(summary, worker, judge) providers for scenario construction.

    Explicit `--provider`/`--model` flags pin ALL roles to that one model (the pre-roles
    behavior); half a pair completes from the configured worker role rather than from bedrock.
    Otherwise each role resolves from `.wmo/settings.toml`, falling back to worker, then to the
    built-in default. Judging benefits from a different family than the worker: a same-family
    judge carries self-preference bias toward the generator's outputs.
    """
    import wmo.providers as providers

    if provider is not None or model is not None:
        llm = providers.get_provider(_worker_role_provider_config(provider, model, region))
        return llm, llm, llm
    default = _provider_config(_DEFAULT_WORKER_PROVIDER, _DEFAULT_WORKER_MODEL, region)
    cache: dict[str, Provider] = {}
    by_role: dict[str, Provider] = {}
    for role in ("summary", "worker", "judge"):
        config = _role_provider_config(role, region) or default
        key = f"{config.kind.value}:{config.model}:{config.endpoint}:{config.region}"
        if key not in cache:
            cache[key] = providers.get_provider(config)
        by_role[role] = cache[key]
    return by_role["summary"], by_role["worker"], by_role["judge"]


def _resolve_scenario_embedder(
    embed_provider: str, embed_model: str | None, embed_dim: int, region: str | None
) -> Embedder:
    import wmo.providers as providers
    from wmo.providers import ProviderConfig
    from wmo.providers.base import EmbedderKind
    from wmo.retrieval import HashingEmbedder

    try:
        kind = EmbedderKind(embed_provider)
    except ValueError:
        kinds = ", ".join(k.value for k in EmbedderKind)
        raise typer.BadParameter(
            f"unknown embed provider {embed_provider!r}; choose one of: {kinds}"
        ) from None
    if kind is EmbedderKind.HASHING:
        return HashingEmbedder(dim=embed_dim)
    if kind is EmbedderKind.LOCAL:
        from wmo.providers.local_embed import LocalEmbedder

        return LocalEmbedder(embed_model, dim=embed_dim)
    if not embed_model:
        raise typer.BadParameter(
            f"--embed-provider {kind.value} requires --embed-model "
            "(the embeddings model id / Azure embedding deployment)"
        )
    return providers.get_provider(
        ProviderConfig(
            kind=kind.provider_kind(),
            model=embed_model,
            embed_model=embed_model,
            embed_dim=embed_dim,
            region=region,
        )
    )


@app.command("demo")
def demo(
    name: str = typer.Option(None, "--name", help="World model to demo (default: pick one)."),
    root: str = typer.Option(
        ARTIFACT_DIR,
        help="Project dir. Shipped example models are found too while this is the default "
        "project dir; point it elsewhere to search that root alone.",
    ),
    steps: int = typer.Option(5, min=1, help="Max scenario steps to replay."),
    traces: str = typer.Option(
        None,
        "--traces",
        help="Trace file to sample the scenario from. Default: the model's own "
        "traces.otel.jsonl, which only a Hub-downloaded model or a shipped example has, since a "
        "build keeps no copy of the corpus it read.",
    ),
    seed: int = typer.Option(None, help="Seed for the scenario sample (default: random)."),
    show_prompt: bool = typer.Option(
        True,
        "--show-prompt/--no-prompt",
        help="Print the exact env prompt the world model sees for the first step.",
    ),
    max_fidelity: bool = typer.Option(
        False, "--max-fidelity", help="Run with the online extras on (default: pure RAG)."
    ),
) -> None:
    """Replay a randomly sampled recorded scenario against the world model, open loop."""

    import wmo.providers as providers
    from wmo.cli.ui import select_provider_and_model
    from wmo.engine.build import ingest
    from wmo.engine.demo import run_demo
    from wmo.engine.world_model import WorldModel
    from wmo.providers import verify_all
    from wmo.providers.retry import wrap_provider_with_retries
    from wmo.utils.waterfall import is_capacity_error

    wm, resolved_name, _provider, model_root = _load_model_any(
        name, root, max_fidelity=max_fidelity
    )
    model_dir = WorldModelStore(str(model_root)).resolve(resolved_name)
    traces_file = _demo_traces(model_dir, resolved_name, traces)
    config = load_config(model_dir)  # the model dir holds its own HarnessConfig
    candidates = [t for t in ingest(config, file=str(traces_file)) if t.steps]
    if not candidates:
        raise typer.BadParameter(f"{traces_file} contains no replayable traces")
    trace = random.Random(seed).choice(candidates)

    total = min(steps, len(trace.steps))
    _console.print(
        f"replaying scenario [bold]{trace.trace_id}[/bold] against [bold]{resolved_name}[/bold] "
        f"(open loop, {total} of {len(trace.steps)} steps)…"
    )
    if trace.steps[0].task:
        _console.print(f"[dim]task: {escape(trace.steps[0].task)}[/dim]")
    if show_prompt:
        _console.print(
            Panel(
                escape(_first_prompt(wm, trace)),
                title="[dim]env prompt (step 1) — what the world model sees[/dim]",
                border_style="bright_black",
            )
        )

    done: list = []  # DemoStep results stream in as each prediction lands

    def _print_result(i: int, n: int, demo_step) -> None:  # noqa: ANN001 - engine DemoStep
        done.append(demo_step)
        action = demo_step.action
        call = (
            f"{action.name} {json.dumps(action.arguments)}"
            if action.name
            else (action.content or "")[:120]
        )
        verdict = (
            "[green]exact match[/green]" if demo_step.exact_match else "[yellow]differs[/yellow]"
        )
        _console.print(f"\n[bold]step {i}/{n}[/bold]  [cyan]{escape(call)}[/cyan]  {verdict}")
        _console.print(f"  [green]predicted[/green]: {escape(demo_step.predicted.content)}")
        _console.print(f"  [dim]actual[/dim]:    {escape(demo_step.actual.content)}")
        note = demo_step.predicted.metadata.get("state_note")
        if isinstance(note, str) and note.strip():
            _console.print(f"  [dim]model note: {escape(note.strip())}[/dim]")

    while True:
        try:
            busy = "[dim]world model predicting…[/dim]"
            with _console.status(busy, spinner="dots") as status:
                _NARRATOR.attach(status, busy)

                def _on_step(i: int, n: int) -> None:
                    text = f"[dim]world model predicting step {i}/{n}…[/dim]"
                    _NARRATOR.busy = text
                    status.update(text)

                try:
                    run_demo(
                        wm,
                        trace,
                        max_steps=steps,
                        on_step=_on_step,
                        on_result=_print_result,
                        skip=len(done),
                    )
                finally:
                    _NARRATOR.detach()
            break
        except Exception as exc:  # noqa: BLE001 - classified below
            out_of_capacity = is_capacity_error(exc)
            if not out_of_capacity or not _console.is_terminal:
                # A backend that refused the request (missing/expired credentials, a model id it
                # does not serve) is a user-fixable setup error, not a wmo bug: render it with the
                # same hint `wmo providers verify` prints. Anything else keeps its traceback.
                if not _is_provider_sdk_error(exc):
                    raise
                _exit_serve_provider_failure(
                    resolved_name,
                    config.serve_provider_config(),
                    exc,
                    out_of_capacity=out_of_capacity,
                    retry=f"`wmo demo --name {resolved_name}`",
                )
            # Retries are exhausted and the backend is still down: offer to re-point the model
            # at a different provider (same picker as the build wizard) and RESUME from the
            # failed step — completed steps stay done.
            _console.print(f"\n[red]serve provider is still failing[/red]: {_short_error(exc)}")
            _console.print("[yellow]pick a different provider to continue the demo[/yellow]")
            provider_name, model_type, region = select_provider_and_model(
                _console,
                lambda text: _console.input(text),
                lambda text: _console.input(text, password=True),
                default_provider=None,
                default_model=None,
                default_region=None,
                interactive=True,
                check=lambda cfg: verify_all([cfg])[0],
            )
            switched = _provider_config(provider_name, model_type, region)
            provider = wrap_provider_with_retries(
                providers.get_provider(switched), on_retry=_NARRATOR.on_retry, sleep=_NARRATOR.sleep
            )
            wm = WorldModel.load(str(model_dir), provider, telemetry_root=str(model_root))
            _console.print(
                f"[dim]resuming from step {len(done) + 1} with "
                f"{provider_name} ({model_type})…[/dim]"
            )
    matches = sum(1 for d in done if d.exact_match)
    _console.print(f"\n{matches}/{len(done)} exact matches (run `wmo eval` for judged fidelity)")


def _first_prompt(wm: WorldModel, trace) -> str:  # noqa: ANN001 - core Trace
    """Render the first step's env prompt on a throwaway session (display only)."""
    probe = wm.new_session(task=trace.steps[0].task)
    try:
        return wm.render_step_prompt(probe.id, trace.steps[0].action)
    finally:
        wm.end_session(probe.id)


@app.command("play")
def play(
    name: str = typer.Option(None, "--name", help="World model to play (default: pick one)."),
    task: str = typer.Option(None, "--task", help="Task to seed the session with."),
    root: str = typer.Option(
        ARTIFACT_DIR,
        help="Project dir. Shipped example models are found too while this is the default "
        "project dir; point it elsewhere to search that root alone.",
    ),
    max_fidelity: bool = typer.Option(
        False, "--max-fidelity", help="Run with the online extras on (default: pure RAG)."
    ),
) -> None:
    """Step into the environment yourself: type actions, the world model returns observations."""
    from wmo.cli.ui import run_play_repl

    wm, resolved_name, _provider, _model_root = _load_model_any(
        name, root, max_fidelity=max_fidelity
    )
    suggestions = _action_suggestions(wm)
    run_play_repl(_console, wm, resolved_name, task, suggestions=suggestions)


def _demo_traces(model_dir: Path, resolved_name: str, explicit: str | None) -> Path:
    """The corpus `wmo demo` samples its scenario from: explicit `--traces`, else the model's own.

    Default resolution is `local_traces_path`, the same one `wmo serve` and `wmo optimize route
    sweep` use: a Hub-downloaded copy inside the artifact, else the sibling corpus of the shipped
    example layout. A build keeps NO copy of the corpus it read, so a plain `wmo build` leaves
    neither, and the failure has to name the file to pass rather than the directory it searched.
    """
    from wmo.serving.traces_source import TRACES_FILENAME, local_traces_path

    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise typer.BadParameter(f"no trace file at {path} (--traces)")
        return path
    found = local_traces_path(model_dir)
    if found is not None:
        return found
    raise typer.BadParameter(
        f"no trace corpus for world model {resolved_name!r}: looked for "
        f"{model_dir / TRACES_FILENAME} and {model_dir.parent.parent / TRACES_FILENAME}. "
        "A build keeps no copy of the corpus it read, so pass the file `wmo build --file` was "
        f"given: `wmo demo --name {resolved_name} --traces <that file>`"
    )


def _action_suggestions(wm: WorldModel, n: int = 3) -> list[str]:
    """Real action lines sampled from the model's corpus, to seed the play prompt."""
    suggestions: list[str] = []
    for step in wm.sample_steps(8):
        action = step.action
        if action.name:
            args = json.dumps(action.arguments) if action.arguments else ""
            line = f"{action.name} {args}".strip()
        elif action.content:
            line = f"say {action.content[:60]}"
        else:
            continue
        if line not in suggestions:
            suggestions.append(line)
        if len(suggestions) >= n:
            break
    return suggestions


def _load_model_any(name: str | None, root: str, *, max_fidelity: bool = False):  # noqa: ANN202
    """Resolve a model across the project dir AND shipped examples, then load it."""
    store_root, resolved = _resolve_model_any(name, root)
    wm, resolved, provider = _load_model(resolved, str(store_root), max_fidelity=max_fidelity)
    return wm, resolved, provider, store_root


def _model_candidates(root: str) -> list[tuple[str, Path, str]]:
    """Every reachable artifact as `(label, store_root, name)`, local builds first.

    The label disambiguates same-named artifacts (`tau-bench (local)` vs `tau-bench (tau-bench
    example)`), so every message that enumerates choices must print labels, not bare names.
    """
    candidates: list[tuple[str, Path, str]] = []
    candidates.extend((f"{n} (local)", Path(root), n) for n in WorldModelStore(root).list_names())
    for example_dir in _discover_examples():
        example_store = WorldModelStore(example_dir)
        candidates.extend(
            (f"{n} ({example_dir.name} example)", example_dir, n)
            for n in example_store.list_names()
        )
    return candidates


def _is_default_project_dir(root: str) -> bool:
    """Whether `--root` still points at the default project dir, however it was spelled.

    Comparing the raw string against `.wmo` made `--root ./.wmo` and `--root .wmo/` (what shell
    tab-completion types) silently mean something different from `--root .wmo`, so resolve both.
    """
    return Path(root).resolve() == Path(ARTIFACT_DIR).resolve()


def _resolve_model_any(name: str | None, root: str) -> tuple[Path, str]:
    """Which artifact a read command should open, as `(store_root, resolved_name)`.

    A `--root` pointing somewhere other than the default project dir keeps single-root behavior.
    Otherwise the search spans `<root>/models/*` plus the downloaded `<data root>/*/models/*`,
    so `demo`, `play` and `knowledge` all agree on
    which models exist.
    """
    if not _is_default_project_dir(root):
        return Path(root), _resolve_name(WorldModelStore(root), name)

    candidates = _model_candidates(root)
    if name is not None:
        matched = [c for c in candidates if c[2] == name]
        if not matched:
            have = ", ".join(c[0] for c in candidates) or "none built"
            raise typer.BadParameter(f"no world model named {name!r} (have: {have})")
        # Prefer the local build over a same-named example artifact.
        _label, store_root, resolved = matched[0]
    elif not candidates:
        raise typer.BadParameter(
            "no world models found; build one with `wmo build --file <traces> --name <name>`, "
            "or fetch a published benchmark corpus first with `wmo download tau-bench`"
        )
    elif len(candidates) == 1:
        _label, store_root, resolved = candidates[0]
    elif _console.is_terminal:
        labels = [c[0] for c in candidates]
        chosen = _select_from(labels)
        _label, store_root, resolved = candidates[labels.index(chosen)]
    else:
        have = ", ".join(c[0] for c in candidates)
        raise typer.BadParameter(
            f"multiple world models ({have}); pass --name{_shadow_hint(candidates)}"
        )
    return store_root, resolved


def _shadow_hint(candidates: list[tuple[str, Path, str]]) -> str:
    """Name the flag that reaches a shipped example a same-named local build shadows.

    `--name` cannot separate the two (the local build always wins), so a listing that shows the
    name twice has to say which flag can: `--root <the example dir>`.
    """
    names = [c[2] for c in candidates]
    shadowed = [c for c in candidates if names.count(c[2]) > 1 and not c[0].endswith("(local)")]
    if not shadowed:
        return ""
    label, store_root, shadowed_name = shadowed[0]
    return (
        f" (--name {shadowed_name} takes the local build; for '{label}' pass --root {store_root})"
    )


def _select_from(labels: list[str]) -> str:
    """Interactive picker over pre-rendered labels (arrow keys on a TTY)."""
    from wmo.cli import ui as _ui  # package-internal: reuse the wizard's picker machinery

    return _ui._select(
        _console,
        lambda text: _console.input(text),
        "Select a world model",
        labels,
        None,
        interactive=True,
    )


def _resolve_name(store: WorldModelStore, name: str | None) -> str:
    """Resolve which model to run: explicit `--name`, an interactive picker, or the sole model.

    With `--name`, validate it exists. Otherwise, when several models are built on an interactive
    terminal, show a numbered picker; on a non-TTY (or a single model) defer to `store.resolve`,
    which returns the lone model or raises a helpful "pass --name" error. Store errors
    (unknown/ambiguous name) are turned into a clean `typer.BadParameter` rather than a traceback.
    """
    from wmo.cli.ui import select_model

    try:
        if name is not None:
            store.resolve(name)  # validates existence, raising a friendly error if missing
            return name
        # Only enumerate full model summaries when we actually need the picker (>1 model on a TTY).
        # `list_names` is cheap (a dir scan); `list_info` reads every config/metrics/frontier file.
        if _console.is_terminal and len(store.list_names()) > 1:
            # An artifact `list_info` reports as unreadable cannot be run, so keep it off the
            # menu; `wmo list` is where its reason is printed.
            readable = [info for info in store.list_info() if info.error is None]
            if not readable:
                raise ValueError(
                    f"no readable world model under {store.models_dir}; "
                    "run `wmo list` to see what is wrong with each one"
                )
            return select_model(_console, readable)
        return store.resolve(None).name
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def _benchmark_roots() -> tuple[Path, ...]:
    """Every root holding self-contained task dirs.

    Benchmark data is not vendored in this repo: `wmo download` writes bundles through
    `wmo.hub`, which owns where they land (`$ENVCAP_DATA_ROOT` if set, else
    `environment-capture-data/` under the working directory). Deriving the root from
    `corpus_path` instead of hardcoding one keeps discovery pointed wherever download wrote,
    including when the override moves it.
    """
    # Imported here per this module's deferred-import rule (#373): the CLI's startup
    # latency budget forbids eager imports, and this function runs only on discovery.
    from wmo.hub import CORPORA, corpus_path

    # corpus_path is `<root>/<benchmark>/traces.otel.jsonl`, so its grandparent is the root.
    return (corpus_path(next(iter(CORPORA))).parent.parent,)


def _discover_examples() -> list[Path]:
    found: list[Path] = []
    for root in _benchmark_roots():
        if not root.exists():
            continue
        found.extend(
            path
            for path in root.iterdir()
            if path.is_dir()
            and _is_safe_example_name(path.name)
            and ((path / "traces.otel.jsonl").exists() or (path / "run.sh").exists())
        )
    return sorted(found)


def _is_safe_example_name(name: str) -> bool:
    """Whether `name` is resolvable at all — discovery must not surface what lookup would reject.

    A downloaded dir whose name `validate_name` rejects can never be named on a command line, so
    listing it as a model candidate or in an "available:" hint would only offer a dead end.
    """
    try:
        validate_name(name)
    except ValueError:
        return False
    return True


def _resolve_example(name: str) -> Path:
    # An unsafe segment (spaces, path separators, ...) is simply not an example name; fall
    # through to the same "unknown example" usage error instead of a ValueError traceback.
    try:
        safe = validate_name(name)
    except ValueError:
        safe = None
    if safe is not None:
        matches = [root / safe for root in _benchmark_roots() if (root / safe).is_dir()]
        if len(matches) > 1:
            found = ", ".join(str(path) for path in matches)
            raise typer.BadParameter(f"example {name!r} exists in multiple roots: {found}")
        if matches:
            return matches[0]
    available = ", ".join(path.name for path in _discover_examples())
    hint = f" (available: {available})" if available else ""
    raise typer.BadParameter(f"unknown example {name!r}{hint}")


@research_app.command("concurrency")
def research_concurrency(
    suite: str = typer.Argument(
        ..., help="Eval suite / example name (e.g. tau-bench) — its corpus + config are reused."
    ),
    scenarios: int = typer.Option(16, "--scenarios", help="Batch size N held fixed across levels."),
    levels: str = typer.Option(
        "1,2,4,8,16", "--levels", help="Comma-separated concurrency levels (baseline first)."
    ),
    trials: int = typer.Option(1, "--trials", help="Timed repeats per level (for error bars)."),
    select: str = typer.Option(
        "random",
        "--select",
        help="Which held-out scenarios to draw: random (default — a representative sample) | "
        "simplest (fewest steps) | longest. simplest/longest order by step count to check whether "
        "a result is robust to the trace sample; the biased draws must not be the default.",
    ),
    select_seed: int = typer.Option(0, "--select-seed", help="Seed for --select random."),
    side: str = typer.Option(
        "both", "--side", help="both = differential | world = WM-only | real = sandbox-only."
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="Provider running the model. Default: the worker role `wmo providers set` wrote to "
        "`.wmo/settings.toml`, else bedrock.",
    ),
    model: str | None = typer.Option(
        None,
        help="Model id (environment LLM). Default: the configured worker role's model, else the "
        "flagship of whichever provider is in play (bedrock/claude-opus-4-8).",
    ),
    region: str | None = typer.Option(None, help="AWS region (Bedrock)."),
    deployment: str | None = typer.Option(None, help="Azure OpenAI deployment name."),
    api_version: str | None = typer.Option(None, help="Azure OpenAI API version."),
    endpoint: str | None = typer.Option(None, help="Azure OpenAI / custom base URL."),
    real_arg: list[str] | None = _RESEARCH_REAL_ARG,
    real_timeout: float | None = typer.Option(
        None, "--real-timeout", help="Abort a real sandbox run after N seconds."
    ),
    out: str | None = typer.Option(None, help="Path to write the ConcurrencyScalingReport JSON."),
    examples_root: str | None = typer.Option(
        None, help="Examples dir. Default: the downloaded benchmark data root."
    ),
) -> None:
    """Measure the concurrency scaling law: batch wall-clock vs. how many scenarios run at once.

    Reconstructs a fixed batch of N held-out scenarios from `suite`'s corpus at each concurrency
    level, timing the world-model batch and (with `--side both`) the matching real-sandbox batch,
    to give the time differential T_real(W)/T_world(W). Reuses the suite's corpus + config
    (train_split, top_k, prompt) and the example's `run.sh`. See `wmo.research.concurrency_run`.
    """
    import wmo.providers as providers
    from wmo.engine.build import split_traces
    from wmo.engine.eval_suites import resolve_eval_suite
    from wmo.engine.prompts import BASE_ENV_PROMPT
    from wmo.ingest import get_adapter
    from wmo.research import Side, run_concurrency_scaling
    from wmo.research.concurrency_run import build_real_runner, build_world_runner
    from wmo.research.concurrency_scaling import ConcurrencyPoint
    from wmo.retrieval import EmbeddingRetriever, HashingEmbedder
    from wmo.retrieval.leakfree import DemoRetriever

    if scenarios < 1:
        raise typer.BadParameter("--scenarios must be at least 1")
    try:
        which = Side(side)
    except ValueError:
        allowed = ", ".join(s.value for s in Side)
        raise typer.BadParameter(f"--side must be one of: {allowed}") from None
    # Resolve (and so validate) the environment LLM up front, not lazily inside a worker thread
    # mid-sweep. Omitted flags come from the configured worker role.
    # `--deployment` goes IN rather than on top: on Azure it is what a `--model` swap needs to
    # know, and layering it afterwards would reject a run the user had already answered for.
    env_config = _worker_role_provider_config(provider, model, region, deployment=deployment)
    connection = {
        field: value
        for field, value in (("api_version", api_version), ("endpoint", endpoint))
        if value is not None
    }
    if connection:
        env_config = env_config.model_copy(update=connection)
    if select not in ("simplest", "longest", "random"):
        raise typer.BadParameter("--select must be one of: simplest, longest, random")
    try:
        level_list = [int(x) for x in levels.split(",") if x.strip()]
    except ValueError:
        raise typer.BadParameter(
            f"--levels must be a comma-separated list of integers, got {levels!r}"
        ) from None
    if not level_list:
        raise typer.BadParameter("--levels must list at least one concurrency level")
    if any(level < 1 for level in level_list):
        raise typer.BadParameter("--levels must be positive concurrency counts")
    # Every level runs the same N scenarios, so a level above N would silently cap its effective
    # concurrency at N and just duplicate the N-worker point — a misleading flat tail.
    if max(level_list) > scenarios:
        raise typer.BadParameter(
            f"--levels goes up to {max(level_list)} but only --scenarios {scenarios} run at each "
            f"level, so W>{scenarios} would just duplicate W={scenarios}. Raise --scenarios to "
            f"{max(level_list)} or drop levels above {scenarios}."
        )

    suite_roots = (
        [str(root) for root in _benchmark_roots()] if examples_root is None else [examples_root]
    )
    # An unresolvable suite is a bad argument, like every other input validated above; the
    # ValueError carries the available names, so only the next command has to be added. A direct
    # `.toml` selector that exists resolves past name lookup and fails on its own contents, so
    # listing the discoverable suites there would point away from the file that needs repairing.
    try:
        resolved = resolve_eval_suite(suite, suite_roots)
    except ValueError as exc:
        direct = Path(suite)
        resolved_by_name = not (direct.suffix == ".toml" and direct.exists())
        hint = "; `wmo eval list` prints the suites" if resolved_by_name else ""
        raise typer.BadParameter(f"{exc}{hint}") from exc
    files = resolved.resolve_files()
    missing = [f for f in files if not f.exists()]
    if not files or missing:
        raise typer.BadParameter(f"suite {suite!r} has no trace corpus at {missing or files}")
    adapter = get_adapter("otel-genai")
    traces = [t for f in files for t in adapter.from_file(str(f))]
    if not traces:
        raise typer.BadParameter(f"suite {suite!r} ingested no traces")

    # Draw N held-out scenarios, so both sides replay the SAME held-out traces by index — keep N
    # within the held-out pool so the split stays aligned. `--select` picks the drawing rule:
    # `random` (default — a representative sample) or `simplest`/`longest`, which order by step
    # count to check a result is robust to the trace sample rather than an artifact of it.
    train, holdout = split_traces(traces, resolved.config.train_split)
    pool = holdout or traces
    need = scenarios
    if need > len(pool):
        raise typer.BadParameter(
            f"need {need} scenario(s) but suite {suite!r} has only {len(pool)} held-out trace(s); "
            "lower --scenarios/--levels or grow the corpus"
        )
    indexed = list(enumerate(pool))
    if select == "random":
        random.Random(select_seed).shuffle(indexed)
    else:  # simplest | longest — order by step count, ascending or descending
        indexed.sort(key=lambda item: len(item[1].steps), reverse=(select == "longest"))
    selected = indexed[:need]

    # Prompt + retrieval mirror the suite's eval config, so the timed reconstruction is the same
    # work `wmo eval run <suite>` scores.
    suite_prompt = resolved.resolve_prompt()
    prompt = (
        suite_prompt.read_text(encoding="utf-8") if suite_prompt is not None else BASE_ENV_PROMPT
    )
    use_rag = not resolved.config.no_rag
    embedder = HashingEmbedder(dim=resolved.config.embed_dim) if use_rag else None
    demos = DemoRetriever(
        EmbeddingRetriever(embedder) if embedder is not None else None,
        train if use_rag else [],
        top_k=resolved.config.top_k,
    )

    def provider_factory() -> Provider:
        return providers.get_provider(env_config)

    world_runner = build_world_runner(provider_factory, prompt, demos, selected)
    real_runner = None
    if which in (Side.BOTH, Side.REAL):
        example_dir = _resolve_example(resolved.example)
        real_extra = list(real_arg or [])
        # Force each runner's cold-standup + isolation flags so the real side pays its TRUE from-
        # source standup and concurrent sandboxes don't clobber each other's docker state (see
        # _CONCURRENCY_ISOLATION_FLAGS). Skipped if the caller passed their own real-runner args.
        # Prepend the forced cold-standup/isolation flags; any user `--real-arg` comes AFTER so it
        # can still override (argparse takes the last value), but the forced flags are never dropped
        # just because the user passed an unrelated arg.
        forced = _CONCURRENCY_ISOLATION_FLAGS.get(resolved.example, ())
        if forced:
            real_extra = list(forced) + real_extra
            _console.print(
                f"[yellow]{resolved.example}: real runner uses {' '.join(forced)} "
                "(true from-source cold standup, isolated per concurrent scenario)[/yellow]"
            )
        real_runner = build_real_runner(
            example_dir,
            selected,
            train_split=resolved.config.train_split,
            extra_args=real_extra,
            timeout=real_timeout,
        )

    batch_desc = f"batch of {len(selected)}"
    _console.print(
        f"\n[bold]concurrency scaling[/bold] {suite}: {batch_desc} held-out "
        f"scenario(s), levels={level_list}, side={which.value}, trials={trials}\n"
    )

    def _progress(point: ConcurrencyPoint) -> None:
        _console.print(f"  {point.summary()}")

    report = run_concurrency_scaling(
        world_runner,
        real_runner,
        levels=level_list,
        scenarios=len(selected),
        trials=trials,
        side=which,
        on_point=_progress,
    )
    report.benchmark = resolved.example

    # Speedup vs. W=1 is a like-for-like ratio because every level runs the same fixed-N batch.
    best = report.best_speedup()
    if best is not None and best.speedup:
        _console.print(
            f"\n[bold]best world-model speedup[/bold]: {best.speedup:.2f}x at concurrency "
            f"{best.level} (efficiency {best.efficiency:.0%})"
        )
    if out:
        Path(out).write_text(report.model_dump_json(indent=2), encoding="utf-8")
        _console.print(f"wrote report -> {out}")


@research_app.command("plot-concurrency")
def research_plot_concurrency(
    report: str = _RESEARCH_PLOT_REPORT,
    out: str = typer.Option("concurrency_scaling.png", "--out", help="Output image path."),
    title: str = typer.Option("Concurrency scaling law", "--title", help="Figure title."),
) -> None:
    """Render a concurrency scaling-law report JSON to a figure (needs the `viz` extra).

    Draws batch wall-clock vs. concurrency (log-log, mean±std), the world-model speedup vs.
    ideal-linear, and the T_real/T_world differential when the report has both sides.

    `concurrency_plot` is imported here (not at module scope) because it pulls in matplotlib/pandas
    from the optional `viz` extra, and the harness runtime must not require them.
    `_require_viz_extra`
    turns a missing extra into a usage error naming `uv sync --extra viz`, so the import itself
    never surfaces a raw ModuleNotFoundError traceback.
    """
    _require_viz_extra()
    from wmo.research.concurrency_plot import render_report

    try:
        written = render_report(report, out, title=title)
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _console.print(f"wrote figure -> {written}")


@research_app.command("plot-concurrency-combined")
def research_plot_concurrency_combined(
    reports: list[str] = _RESEARCH_PLOT_COMBINED,
    out_speedup: str = typer.Option(
        "concurrency_speedup.png", "--out-speedup", help="Output path for the speed-up figure."
    ),
    out_cost: str = typer.Option(
        "concurrency_cost.png", "--out-cost", help="Output path for the cost figure."
    ),
) -> None:
    """Render the cross-benchmark comparison as two standalone figures (needs the `viz` extra).

    Overlays several benchmarks so their RELATIVE differences read at a glance, split into two
    self-contained images: the speed-up figure (how many times faster the world model is than the
    real environment, per benchmark) and the cost figure (the reconstruction-vs-real-setup cost that
    explains it). Pass the per-benchmark report JSONs (`wmo research concurrency <suite> --out ...`)
    in the order you want them coloured.

    Imported lazily for the same reason as `plot-concurrency` (the optional matplotlib viz extra),
    and guarded by the same `_require_viz_extra` pre-flight.
    """
    _require_viz_extra()
    from wmo.research.concurrency_plot import render_cost, render_speedup

    try:
        speedup_path = render_speedup(reports, out_speedup)
        cost_path = render_cost(reports, out_cost)
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _console.print(f"wrote figures -> {speedup_path}, {cost_path}")


# The pre-split coding-router lab's own numbers on this matrix, quoted beside the wmo protocol's
# so a reader sees the decision-rule difference instead of hunting for it: its router picked the
# CHEAPEST arm whose predicted solve odds cleared a threshold (cost first, quality as the
# constraint), measured over 6 seeded 80/20 repo splits.
_DEEPSWE_LAB_REFERENCE = "cost ratio median 3.18x (range 0.90-5.25), graded delta median -0.015"


@research_app.command("deepswe-holdout")
def research_deepswe_holdout(
    bundle: str = typer.Argument(
        ...,
        help="Converted DeepSWE bundle dir: matrix.json + task_embeddings.npy + "
        "scenario_groups.json (written by `wmo optimize route convert-deepswe`).",
    ),
    fallback: str = typer.Option(
        "claude-opus-5@high",
        "--fallback",
        help="Baseline pool arm the guard pins (the matrix's strongest single arm).",
    ),
    splits: int = typer.Option(
        6,
        "--splits",
        min=1,
        help="Independent repo-grouped fit/report splits (deterministic, salted 0..n-1); the "
        "median over them is the headline, since one 30% holdout of ~90 repos can be lucky.",
    ),
    cost_quality: float = typer.Option(
        0.25,
        "--cost-quality",
        min=0.0,
        max=1.0,
        help="Dial position to evaluate (0.25 = the shipped default; 1.0 = max savings). The "
        "champion optimizes quality first, so cost movement lives on this dial.",
    ),
) -> None:
    """Repo-grouped router holdout on the converted DeepSWE matrix, offline and credential-free.

    The honesty protocol this benchmark needs: DeepSWE tasks from one repository share code, so
    fit and report sides split by REPOSITORY (`split_router_scenarios_grouped`), never by row.
    Per split: fit wmo's champion kNN on the fit repos (recorded local-model vectors, guard
    pinned to `--fallback`), set the dial, and report the held-out paired headline against
    always-`--fallback`. Prints per-split rows, the medians, and the pre-split lab's reference
    numbers for the same matrix under its different (cheapest-arm-above-threshold) decision
    rule; a gap against that reference is a finding about the rules, not noise.
    """
    import tempfile
    from datetime import UTC, datetime
    from statistics import median

    from rich.table import Table

    from wmo.optimize.routing.embedding_cache import CachedTaskEmbedder
    from wmo.optimize.routing.knn import apply_cost_quality, fit_knn_artifact
    from wmo.optimize.routing.outcomes import OutcomeMatrix, split_router_scenarios_grouped
    from wmo.optimize.routing.policy import EmbedderSpec
    from wmo.optimize.routing.report import build_report

    root = Path(bundle)
    for name in ("matrix.json", "task_embeddings.npy", "scenario_groups.json"):
        if not (root / name).is_file():
            raise typer.BadParameter(
                f"{root / name} is missing; point at a bundle written by "
                "`wmo optimize route convert-deepswe` (or the downloaded published bundle)"
            )
    matrix = OutcomeMatrix.load(root / "matrix.json")
    groups: dict[str, str] = json.loads((root / "scenario_groups.json").read_text(encoding="utf-8"))
    built = CachedTaskEmbedder(matrix, root / "task_embeddings.npy")
    spec = EmbedderSpec(kind="local", dim=built.dim)

    table = Table(show_header=True, title=f"DeepSWE repo-grouped holdout (dial {cost_quality:g})")
    for column in ("split", "fit/report tasks", "repos", "base", "routed", "delta", "x cheaper"):
        table.add_column(column, justify="right")
    ratios: list[float] = []
    deltas: list[float] = []
    for index in range(splits):
        salt = "" if index == 0 else str(index)
        try:
            split = split_router_scenarios_grouped(matrix.scenario_ids(), groups, salt=salt)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        with tempfile.TemporaryDirectory(prefix="deepswe-holdout-") as scratch:
            try:
                fitted = fit_knn_artifact(
                    matrix,
                    out_path=Path(scratch) / "policy.json",
                    matrix_source=str(root / "matrix.json"),
                    embedder=spec,
                    fit_ids=list(split.fit_ids),
                    fallback=fallback,
                    built=built,
                )
            except ValueError as exc:
                raise typer.BadParameter(str(exc)) from exc
            policy = apply_cost_quality(fitted.policy, cost_quality)
            report = build_report(
                matrix,
                policy,
                baseline=fallback,
                endpoint="research-deepswe-holdout",
                generated_at=datetime.now(UTC).isoformat(),
                scenario_label="on held-out DeepSWE tasks from repositories the fit never saw",
                built=built,
            )
        headline = report.headline
        ratio = (
            headline.baseline_cost_per_run_usd / headline.cost_per_run_usd
            if headline.cost_per_run_usd > 0.0
            else float("inf")
        )
        delta = headline.accuracy - headline.baseline_accuracy
        ratios.append(ratio)
        deltas.append(delta)
        fit_repos = len({groups[sid] for sid in split.fit_ids})
        report_repos = len({groups[sid] for sid in split.report_ids})
        table.add_row(
            salt or "0",
            f"{len(split.fit_ids)}/{len(split.report_ids)}",
            f"{fit_repos}/{report_repos}",
            f"{headline.baseline_accuracy:.3f}",
            f"{headline.accuracy:.3f}",
            f"{delta:+.3f}",
            f"{ratio:.2f}",
        )
    _console.print(table)
    _console.print(
        f"[bold]median over {splits} grouped splits[/bold]: cost ratio "
        f"{median(ratios):.2f}x (range {min(ratios):.2f}-{max(ratios):.2f}), "
        f"graded delta {median(deltas):+.3f}"
    )
    _console.print(
        f"[dim]pre-split lab reference on this matrix, its cheapest-arm-above-threshold rule, "
        f"6 seeded 80/20 repo splits: {_DEEPSWE_LAB_REFERENCE}. wmo's guarded champion leaves "
        "the baseline only on paired quality evidence, so at low dial it buys parity rather "
        "than savings; slide --cost-quality toward 1.0 for the savings end.[/dim]"
    )


def _short_error(exc: Exception) -> str:
    """The error's code + service message, without transport chatter.

    botocore's text ("... (reached max retries: 1) ...") reads as OUR retry state and confuses
    the narration; the structured code + message is what the user needs.
    """
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error", {})
        code, message = error.get("Code"), error.get("Message") or ""
        if code:
            return f"{code}: {message}".rstrip(": ")[:110]
    return str(exc).splitlines()[0][:110]


class _RetryNarrator:
    """Console narration for RetryingProvider: hiccup lines + an inline countdown.

    The hiccup line prints only when the failure CHANGES (a stream of identical throttles says
    it once); while a rich status is attached (demo), the wait counts down in place as
    "retry k/3 — waiting Ns…" and then hands the spinner back to the busy text.
    """

    def __init__(self, console: Console) -> None:
        self._console = console
        self._status = None  # rich Status while a spinner context is active
        self.busy = ""
        self._last_error: str | None = None
        self._attempt = 0
        self._total = 0

    def attach(self, status, busy: str) -> None:  # noqa: ANN001 - rich Status
        self._status = status
        self.busy = busy

    def detach(self) -> None:
        self._status = None
        self._last_error = None

    def on_retry(self, attempt: int, total: int, delay: float, exc: Exception) -> None:
        detail = _short_error(exc)
        if detail != self._last_error:
            self._console.print(f"  [yellow]provider hiccup: {escape(detail)}[/yellow]")
            self._last_error = detail
        self._attempt, self._total = attempt, total
        if self._status is None:
            self._console.print(f"  [yellow]retry {attempt}/{total} in {delay:.0f}s…[/yellow]")

    def sleep(self, delay: float) -> None:
        remaining = int(delay)
        while remaining > 0:
            if self._status is not None:
                self._status.update(
                    f"[yellow]retry {self._attempt}/{self._total} — waiting {remaining}s…[/yellow]"
                )
            time.sleep(1)
            remaining -= 1
        if self._status is not None:
            self._status.update(self.busy)


_NARRATOR = _RetryNarrator(_console)


def _load_model(name: str | None, root: str, *, max_fidelity: bool = False):  # noqa: ANN202
    """Resolve + load a named world model (or the single built one) with its serve provider.

    The serve provider comes from the MODEL'S OWN config (the one it was built to serve on),
    wrapped so transient capacity errors retry with narrated exponential backoff instead of
    dying. `max_fidelity` = the online extras (see `WorldModel.load`); default is pure RAG.
    Returns `(world_model, resolved_name, provider)`.
    """
    import wmo.providers as providers
    from wmo.engine.world_model import WorldModel
    from wmo.providers.retry import wrap_provider_with_retries

    store = WorldModelStore(root)
    resolved_name = _resolve_name(store, name)
    model_dir = store.resolve(resolved_name)
    config = load_config(model_dir)
    serve_config = config.serve_provider_config()
    backend = providers.get_provider(serve_config)
    _prepare_serve_provider_or_exit(backend, serve_config)
    provider = wrap_provider_with_retries(
        backend,
        on_retry=_NARRATOR.on_retry,
        sleep=_NARRATOR.sleep,
    )
    world_model = WorldModel.load(
        str(model_dir), provider, telemetry_root=store.root, max_fidelity=max_fidelity
    )
    return world_model, resolved_name, provider


def _prepare_serve_provider_or_exit(provider: Provider, config: ProviderConfig) -> None:
    """Resolve the serve backend's local prerequisites before the first step, or exit cleanly.

    Every backend builds its SDK client lazily, so a missing SDK or an unset credential otherwise
    surfaces as the SDK's own exception mid-rollout, which Typer renders as a traceback.
    `PreparableProvider.prepare` is the free, offline seam `wmo optimize route sweep` already
    pre-flights with (no backend's `prepare` touches the network), so the interactive commands
    fail here with the same hint `wmo providers verify` prints. Bedrock and tinker document a
    residual gap they cannot close locally; those still fail on the first call.
    """
    from wmo.providers.base import PreparableProvider

    if not isinstance(provider, PreparableProvider):
        return
    try:
        provider.prepare()
    except Exception as exc:  # noqa: BLE001 - every backend raises its own SDK's type here
        _console.print(
            f"[red]✗ {config.kind.value} ({config.model}) unusable[/red]: {escape(str(exc))}"
        )
        _console.print(f"  [yellow]{escape(_credential_hint(config.kind, str(exc)))}[/yellow]")
        _console.print("  [yellow]then re-check with `wmo providers verify`[/yellow]")
        raise typer.Exit(1) from exc


def _exit_serve_provider_failure(
    resolved_name: str,
    config: ProviderConfig,
    exc: Exception,
    *,
    out_of_capacity: bool,
    retry: str,
) -> NoReturn:
    """Render a serve-provider failure as the setup error or the outage it actually is, then exit.

    The two need opposite next steps, and printing the wrong one costs the user the debugging
    session: a 429/503/timeout has already survived `RetryingProvider`'s backoff, so the
    credentials it took to get that far are demonstrably fine and `wmo providers verify` will
    just report the same outage. Only a refusal (bad key, unserved model id) earns the
    credential hint.
    """
    verb = "is out of capacity" if out_of_capacity else "failed"
    _console.print(
        f"\n[red]✗ the serve provider for {resolved_name!r} "
        f"({config.kind.value}, {config.model}) {verb}[/red]: {escape(_short_error(exc))}"
    )
    if out_of_capacity:
        _console.print(
            "  [yellow]retries are exhausted and the backend is still refusing; this is an "
            "outage or a rate limit, not a credentials problem[/yellow]"
        )
        _console.print(
            f"  [yellow]re-run {retry} once it recovers, or run it on a terminal to switch "
            "provider and resume from the failed step[/yellow]"
        )
    else:
        _console.print(f"  [yellow]{escape(_credential_hint(config.kind, str(exc)))}[/yellow]")
        _console.print("  [yellow]then re-check with `wmo providers verify`[/yellow]")
    raise typer.Exit(1) from exc


# Top-level modules whose exceptions mean "the backend refused", not "wmo has a bug": the provider
# SDKs plus the HTTP transports they raise through. waterfall gates its capacity/client
# classification on the same set; this one answers a different question (is a credential hint the
# right thing to print?), so it stays a local literal rather than importing a private name.
_PROVIDER_SDK_MODULES = frozenset(
    {"anthropic", "boto3", "botocore", "httpcore", "httpx", "openai", "tinker"}
)


def _is_provider_sdk_error(exc: Exception) -> bool:
    """Whether `exc` was DEFINED by a provider SDK (bad creds, bad model id, refused request).

    Deliberately narrow: wmo's own bugs must keep their traceback, so only exceptions whose class
    (or a base of it) comes from a backend SDK are re-rendered as a setup error.
    """
    return any(
        (getattr(klass, "__module__", "") or "").split(".")[0] in _PROVIDER_SDK_MODULES
        for klass in type(exc).__mro__
    )


if __name__ == "__main__":
    app()


def _quiet_http_logs() -> None:
    """Cap noisy per-request loggers at WARNING.

    The openai SDK (via httpx) logs one INFO line per API call; logging handlers write to the
    real stderr and bypass the live display's redirection, so during a build each request would
    scroll the GEPA activity region and litter orphaned frame headers across the terminal.
    """
    for name in ("httpx", "httpcore", "openai", "botocore", "urllib3", "anthropic"):
        logging.getLogger(name).setLevel(logging.WARNING)


def main() -> None:
    """CLI entry point: load `.env` from the working directory (so wizard-saved provider keys
    persist across sessions), then dispatch. Kept out of import time so importing the module
    never mutates os.environ."""
    load_env_file()
    _quiet_http_logs()
    app()
