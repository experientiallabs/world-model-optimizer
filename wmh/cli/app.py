"""`wmh` CLI — ingestion UI and operator console for the harness.

Deliberately small. The lifecycle is:
    providers verify -> build -> list -> serve / demo / play
`build` creates the project artifact directory itself, so there is no separate init step. World
models are named (`--name`), stored under `<root>/models/<name>/`, and listed with `wmh list`.
"""

from __future__ import annotations

import json
import logging
import random
import subprocess
import time
import urllib.error
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import typer
import uvicorn
from environment_capture.hub import (
    CORPORA,
    corpus_path,
    fetch_corpus,
    published_corpora,
)
from llm_waterfall import is_capacity_error
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn
from rich.table import Table

import wmh.providers as providers
from wmh.cli.agent_session import register as register_agent_session_commands
from wmh.cli.eval_closed_loop import run_agreement, run_closed_loop
from wmh.cli.harness_app import harness_app
from wmh.cli.platform_cmds import register as register_platform_commands
from wmh.cli.ui import (
    BuildParams,
    RichBuildReporter,
    build_summary_panel,
    judge_model_default,
    models_table,
    run_build_wizard,
    run_play_repl,
    select_model,
    select_option,
    select_provider_and_model,
)
from wmh.config import (
    ARTIFACT_DIR,
    DEFAULT_MODEL_NAME,
    FIDELITY_TIERS,
    PROVIDER_ENV_VARS,
    ArtifactPaths,
    FidelityTier,
    HarnessConfig,
    WorldModelStore,
    load_config,
    load_env_file,
    load_settings,
    normalize_name,
    set_telemetry_enabled,
    settings_path,
    validate_name,
)
from wmh.config.card import make_build_card, save_card
from wmh.config.manifest import MANIFEST_FILE, ManifestMismatch, VerifyResult, verify_manifest
from wmh.core.types import JsonObject
from wmh.engine.build import build as run_build
from wmh.engine.build import ingest, split_traces
from wmh.engine.demo import run_demo
from wmh.engine.eval_suites import (
    discover_eval_suites,
    list_eval_results,
    resolve_eval_suite,
    result_path,
)
from wmh.engine.grounding import GROUNDER_KINDS
from wmh.engine.knowledge import KnowledgeBase
from wmh.engine.prompts import BASE_ENV_PROMPT
from wmh.engine.world_model import WorldModel
from wmh.env.llm_agent import LLMAgent
from wmh.evals.grid import GridResult, ModelSpec, merge_results, run_grid
from wmh.evals.grid_plot import plot_grid, plot_grid_heatmap
from wmh.evals.open_loop import EvalReport, OpenLoopEval
from wmh.ingest import VendorPull, get_adapter, list_adapters
from wmh.optimize.judge import JUDGE_VERSION, RubricJudge
from wmh.providers import ProviderConfig, ProviderKind, verify_all, verify_embedder
from wmh.providers.base import Embedder, EmbedderKind, Provider
from wmh.providers.models import resolve_provider_model
from wmh.providers.retry import wrap_provider_with_retries
from wmh.research import Side, run_concurrency_scaling
from wmh.research.concurrency_run import build_real_runner, build_world_runner
from wmh.research.concurrency_scaling import ConcurrencyPoint
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder, get_embedder
from wmh.retrieval.leakfree import DemoRetriever
from wmh.scenarios import (
    ChecklistJudge,
    FacetExtractor,
    ScenarioBuildConfig,
    ScenarioSet,
    build_scenario_set,
    verify_scenarios,
)
from wmh.serving.server import create_app
from wmh.telemetry import (
    BuildTelemetryStats,
    TelemetryBuildReporter,
    capture_build_completed,
    capture_eval_completed,
    settings_root_from_results_root,
)
from wmh.tracking import MeteredProvider, Phase, RunTracker, classify_build_call, save_run

app = typer.Typer(
    help="World Model Harness: a frontier LLM acts as your agent's environment.",
    no_args_is_help=True,
)


providers_app = typer.Typer(help="Manage and verify LLM providers.", no_args_is_help=True)
examples_app = typer.Typer(
    help="List and launch self-contained task examples.", no_args_is_help=True
)
config_app = typer.Typer(help="Manage local harness config.", no_args_is_help=True)
research_app = typer.Typer(
    help="Research experiments over the harness (scaling laws, ablations).", no_args_is_help=True
)
scenarios_app = typer.Typer(
    help="Construct and verify representative eval scenario sets from traces.",
    no_args_is_help=True,
)
app.add_typer(providers_app, name="providers")
app.add_typer(examples_app, name="examples")
app.add_typer(config_app, name="config")
app.add_typer(research_app, name="research")
app.add_typer(scenarios_app, name="scenarios")
app.add_typer(harness_app, name="harness")
register_platform_commands(app)
register_agent_session_commands(app)
_console = Console()
_CHECK = "[green]✓[/green]"

# Module-level singleton: a typer.Argument call can't be a default inline (ruff B008).
_EVAL_TOKENS = typer.Argument(
    None,
    help="Trace files to score, or eval flow: list | run <suite> | results optional-suite.",
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
    if normalized == "status":
        settings = load_settings(root)
    elif normalized == "enable":
        settings = set_telemetry_enabled(True, root)
    elif normalized == "disable":
        settings = set_telemetry_enabled(False, root)
    else:
        raise typer.BadParameter("action must be one of: status, enable, disable")
    state = "enabled" if settings.telemetry.enabled else "disabled"
    _console.print(f"telemetry {state} ({settings_path(root)})")


@providers_app.command("verify")
def providers_verify(
    name: str = typer.Option(None, "--name", help="Verify one model's providers (default: all)."),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir."),
) -> None:
    """Ping every configured provider (completion + embed path) and report status.

    Gathers provider configs from the built world models (one `--name`, or all of them, deduped by
    kind+model), so a brand-new project with nothing built yet has nothing to verify. The phi embed
    path of each model is checked too, unless it is the offline (creds-free) hashing embedder.
    """
    store = WorldModelStore(root)
    names = [name] if name is not None else store.list_names()
    if not names:
        _console.print("[yellow]no world models built yet[/yellow]; run `wmh build --name <name>`")
        return
    configs: list[HarnessConfig] = []
    for model_name in names:
        try:
            model_dir = str(store.resolve(model_name))
        except (FileNotFoundError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
        configs.append(load_config(model_dir))

    # Dedup completion providers by kind+model across all selected models.
    seen: set[tuple[str, str]] = set()
    providers: list[ProviderConfig] = []
    for config in configs:
        for pc in config.providers:
            key = (pc.kind.value, pc.model)
            if key not in seen:
                seen.add(key)
                providers.append(pc)
    for result in verify_all(providers):
        mark = "[green]ok[/green]" if result.ok else "[red]fail[/red]"
        _console.print(f"{mark} {result.kind.value} ({result.model}) {result.detail}")

    # Verify each distinct provider-backed embed path (skip the offline hashing embedder).
    embed_seen: set[tuple[str, str]] = set()
    for config in configs:
        if config.embed_provider is EmbedderKind.HASHING:
            continue
        embed_config = config.embed_provider_config()
        key = (embed_config.kind.value, embed_config.model)
        if key in embed_seen:
            continue
        embed_seen.add(key)
        result = verify_embedder(embed_config)
        mark = "[green]ok[/green]" if result.ok else "[red]fail[/red]"
        _console.print(f"{mark} embed:{result.kind.value} ({result.model}) {result.detail}")


@examples_app.command("list")
def examples_list() -> None:
    """List self-contained example tasks."""
    examples = _discover_examples()
    if not examples:
        _console.print("[yellow]no examples found[/yellow]")
        return
    for example in examples:
        _console.print(example.name)


@examples_app.command(
    "run",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def examples_run(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Example task name."),
) -> None:
    """Run an example's local launcher, forwarding any extra args after `--`."""
    example_dir = _resolve_example(name)
    runner = example_dir / "run.sh"
    if not runner.exists():
        raise typer.BadParameter(f"example {name!r} has no run.sh launcher")
    result = subprocess.run([str(runner), *ctx.args], cwd=example_dir, check=False)
    raise typer.Exit(result.returncode)


@app.command("build")
def build(
    name: str = typer.Option(None, "--name", help="Name for this world model."),
    source: str = typer.Option(
        "otel-genai",
        "--source",
        help="Trace source adapter: otel-genai, chat-json, braintrust, phoenix, langfuse, "
        "langsmith, posthog, mastra.",
    ),
    file: str = typer.Option(None, "--file", help="Path to an exported traces file for --source."),
    pull: bool = typer.Option(
        False, "--pull", help="Pull traces live from the source's vendor API (instead of --file)."
    ),
    project: str = typer.Option(None, "--project", help="Vendor project/workspace id (--pull)."),
    api_key: str = typer.Option(None, "--api-key", help="Vendor API key (else env var)."),
    since: str = typer.Option(None, "--since", help="Only pull traces since this ISO timestamp."),
    limit: int = typer.Option(None, "--limit", help="Max number of traces to pull."),
    vendor: str = typer.Option(
        None, "--vendor", help="[deprecated] alias for --source <name> --pull."
    ),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir holding all world models."),
    provider: str = typer.Option(
        None, "--provider", help="Provider that serves the model (default: bedrock)."
    ),
    model: str = typer.Option("claude-opus-4-8", help="Canonical serve model type."),
    judge_model: str = typer.Option(
        None, "--judge-model", help="Canonical GEPA judge model type (default: cheap per provider)."
    ),
    region: str = typer.Option(None, help="AWS region (Bedrock)."),
    fidelity: str = typer.Option(
        "medium",
        help="Build effort (all searching tiers are floored at low's estimate — more effort "
        "never ships worse than low): low (free; the estimated-best config, no search) | "
        "medium (+light GEPA + cheap-lever search) | high (+GEPA + config search) | max (deep "
        "GEPA + full config search). The chosen config serves under `--max-fidelity`.",
    ),
    chain: str = typer.Option(
        None, "--chain", help="Named failover chain from .wmh/fallback.toml (default: its default)."
    ),
    train_split: float = typer.Option(
        0.8, help="Train/held-out ratio for GEPA's internal split (lower = bigger valset)."
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
) -> None:
    """Ingest traces (file upload or vendor SDK pull) and build a named world model.

    Stores the artifact under `<root>/models/<name>/`: ingest -> normalize -> split(train/test) ->
    embed/index -> GEPA optimize -> write. Re-running with the same `--name` rebuilds it.

    With no `--name`/`--file` on an interactive terminal, this launches a guided creation wizard;
    pass `--no-interactive` (or any of those flags) to stay fully scriptable.
    """
    # `--vendor <name>` is the deprecated alias for `--source <name> --pull`: it names the source
    # adapter and implies a live pull.
    if vendor:
        source = vendor
        pull = True

    # Decide whether to run the wizard: explicit flag wins; otherwise auto when at a TTY and the
    # essential inputs (a name and a trace source — a file or a live pull) were not supplied.
    needs_input = name is None or (file is None and not pull)
    use_wizard = interactive if interactive is not None else (_console.is_terminal and needs_input)

    params = BuildParams(
        name=name or DEFAULT_MODEL_NAME,
        source=source,
        file=file,
        pull=pull,
        project=project,
        api_key=api_key,
        since=since,
        limit=limit,
        provider=provider,
        model=model,
        region=region,
        fidelity=fidelity,
        train_split=train_split,
        judge_model=judge_model,
        embed_provider=embed_provider,
        embed_model=embed_model,
        embed_dim=embed_dim,
    )
    if use_wizard:
        params = run_build_wizard(_console, params)
    elif name is None and file is None and not pull:
        raise typer.BadParameter(
            "provide --file <export> or --pull (with --source), or run `wmh build` interactively"
        )
    if params.file and params.pull:
        raise typer.BadParameter("pass either --file or --pull, not both")
    if params.source not in list_adapters():
        raise typer.BadParameter(
            f"unknown --source {params.source!r}; choose one of: {', '.join(list_adapters())}"
        )
    # The wizard always resolves a provider; the flag path keeps its historical default.
    params.provider = params.provider or "bedrock"

    # Flag-supplied names get the same whitespace-to-dash normalization as the wizard.
    params.name = normalize_name(params.name)
    try:
        validate_name(params.name)
    except ValueError as err:
        raise typer.BadParameter(str(err)) from None
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
    if embed_kind is not EmbedderKind.HASHING and not params.embed_model:
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
        providers.provider_or_chain(config.serve_provider_config(), chain=chain),
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
        )
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
            f"({provenance}) — activate with `wmh serve --max-fidelity` / "
            f"`wmh play --max-fidelity`"
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
    checks = [(config.serve_provider_config(), False)]
    if config.embed_provider is not EmbedderKind.HASHING:
        checks.append((config.embed_provider_config(), True))

    failed = False
    for cfg, is_embed in checks:
        label = f"embed:{cfg.kind.value}" if is_embed else cfg.kind.value
        serve_provider = None if is_embed else providers.provider_or_chain(cfg, chain=chain)
        if is_embed:
            _console.print(f"verifying {label}…")
            result = verify_embedder(cfg)
        elif isinstance(serve_provider, providers.WaterfallProvider):
            # Verify the provider the build will actually use — with a chain active, that means
            # pinging every rung (a broken fallback must fail here, not hours into the build).
            label = f"{label} chain (.wmh/fallback.toml)"
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
        if "No module named" in result.detail:
            # SDKs are core deps; a missing module means the env is stale or hand-rolled.
            _console.print("    [yellow]run `uv sync` to install the provider SDKs[/yellow]")
        else:
            envs = ", ".join(PROVIDER_ENV_VARS.get(cfg.kind, []))
            hint = f" ({envs})" if envs else ""
            _console.print(
                f"    [yellow]check the model id and that your credentials are set{hint}[/yellow]"
            )
    if failed:
        raise typer.Exit(1)


@app.command("list")
def list_models(root: str = typer.Option(ARTIFACT_DIR, help="Project dir to list.")) -> None:
    """List every world model built under the project dir."""
    infos = WorldModelStore(root).list_info()
    if not infos:
        _console.print("[yellow]no world models built yet[/yellow]; run `wmh build --name <name>`")
        return
    _console.print(models_table(infos))


@app.command("verify")
def verify(
    name: str = typer.Option(
        None, "--name", help="World model to verify (default: all built ones)."
    ),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir."),
) -> None:
    """Check artifact integrity: re-hash every file and compare against the stored manifest.

    Detects corruption, partial writes, and files modified after the build. Models built
    before this feature have no manifest and are reported as unverified (not an error).
    Exits with code 1 if any file fails verification.
    """
    store = WorldModelStore(root)
    names = [name] if name is not None else store.list_names()
    if not names:
        _console.print("[yellow]no world models built yet[/yellow]; run `wmh build --name <name>`")
        return

    all_ok = True
    for model_name in names:
        try:
            model_dir = store.resolve(model_name)
        except (FileNotFoundError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc

        _console.print(f"[bold]{model_name}[/bold]")
        if not (model_dir / MANIFEST_FILE).exists():
            _console.print(
                "  [yellow]⚠ no manifest — rebuild with `wmh build` to enable integrity checks"
                "[/yellow]"
            )
            continue

        try:
            result: VerifyResult = verify_manifest(model_dir)
        except ManifestMismatch as exc:
            _console.print(f"  [red]✗ manifest unreadable: {exc}[/red]")
            all_ok = False
            continue

        for f in result.files:
            if f.ok:
                _console.print(f"  {_CHECK} {f.name}")
            else:
                all_ok = False
                _console.print(f"  [red]✗ {f.name}[/red]  ({f.detail})")

        if result.ok:
            _console.print("  [green]artifact integrity verified[/green]")
        else:
            _console.print(
                f"  [red]artifact appears corrupted "
                f"({result.failed}/{result.total} files failed) — "
                f"rebuild with `wmh build --name {model_name}`[/red]"
            )

    if not all_ok:
        raise typer.Exit(1)


@app.command("download")
def download(
    benchmarks: list[str] = _DOWNLOAD_BENCHMARKS,
    force: bool = typer.Option(False, "--force", help="Overwrite existing local files."),
) -> None:
    """Download benchmark data bundles (trace corpus + task data) from the Hub.

    With no arguments, lists the org's published datasets (live, via the Hub API) and offers a
    picker. Bundles land in `packages/environment-capture/<benchmark>/`; existing local files
    are kept unless `--force`.
    """
    selected = list(benchmarks or [])
    if selected == ["all"]:
        # Prefer the Hub's live list: the static registry can name corpora that aren't
        # published yet (a 404 mid-loop used to abort the remaining downloads).
        try:
            selected = sorted(corpus.benchmark for corpus in published_corpora())
        except urllib.error.URLError:
            selected = sorted(CORPORA)
    if not selected:
        try:
            published = published_corpora()
        except urllib.error.URLError as exc:
            raise typer.BadParameter(
                f"could not list the Hub's published datasets ({exc.reason}); check the "
                "connection, or pass benchmark names directly, e.g. `wmh download bird-sql`"
            ) from exc
        if not published:
            raise typer.BadParameter(
                "no published corpora found on the Hub; "
                "pass benchmark names directly, e.g. `wmh download bird-sql`"
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
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        except urllib.error.HTTPError as exc:
            # One unpublished/broken dataset must not abort the REST of a multi-download:
            # record it, keep fetching, and fail (with every name) at the end.
            failures.append(f"{name}: the Hub answered {exc.code} for {exc.url}")
            _console.print(f"[yellow]skipping {name}: Hub answered {exc.code}[/yellow]")
            continue
        except urllib.error.URLError as exc:
            raise typer.BadParameter(
                f"{name}: could not reach the Hub ({exc.reason}); check the connection and re-run"
                " — fetches resume file-by-file"
            ) from exc
        state = "kept local" if existing and not force else "fetched"
        _console.print(f"{_CHECK} {state} [bold]{name}[/bold] -> {path}")
    if failures:
        raise typer.BadParameter(
            "some datasets could not be downloaded (unpublished? `wmh download` with no "
            "arguments lists what is):\n  " + "\n  ".join(failures)
        )


def _fetch_with_progress(name: str, *, force: bool) -> Path:
    """fetch_corpus with a live byte progress bar (hidden when nothing needs downloading)."""
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        console=_console,
        transient=True,
    ) as progress:
        task_id = progress.add_task(f"downloading {name}", total=None, visible=False)

        def on_progress(done: int, total: int) -> None:
            progress.update(task_id, completed=done, total=total or None, visible=True)

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
    (e.g. `--root .wmh --root examples/tau-bench`). Routes are namespaced:
    `/world_models/{name}/sessions` and `.../step`.
    """
    names = list(name) if name else None
    # Bad --name input (unsafe segment, unknown model, nothing built) is a usage error,
    # not a traceback; load the models before uvicorn takes over the process.
    try:
        server_app = create_app(list(root), names=names, max_fidelity=max_fidelity)
    except (ValueError, FileNotFoundError) as err:
        raise typer.BadParameter(str(err)) from None
    uvicorn.run(server_app, host="127.0.0.1", port=port)


@app.command("eval")
def eval_(  # noqa: A001 - `eval` is the user-facing command name; the builtin isn't used here
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
    provider: str = typer.Option("bedrock", "--provider", help="Provider running the model."),
    model: str = typer.Option("claude-opus-4-8", help="Canonical model type."),
    region: str | None = typer.Option(None, help="AWS region (Bedrock)."),
    chain: str | None = typer.Option(
        None, "--chain", help="Named failover chain from .wmh/fallback.toml (default: its default)."
    ),
    train_split: float | None = typer.Option(
        None, help="Train/holdout ratio per file (default: 0.7, or suite config)."
    ),
    val_frac: float | None = typer.Option(
        None,
        "--val-frac",
        help="`wmh eval grid`: validation fraction reserved for GEPA in a 3-way split so cells "
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
        None, help="Directory containing example eval suites. Default: repo-local examples/."
    ),
    results_root: str = typer.Option(
        f"{ARTIFACT_DIR}/evals", help="Local directory for named eval result JSON."
    ),
    limit: int = typer.Option(20, help="Rows to show for `wmh eval results`."),
    models: str | None = typer.Option(
        None,
        help="`wmh eval grid`: comma-separated Label:provider:model cells "
        "(e.g. 'Opus 4.8:bedrock:us.anthropic.claude-opus-4-8,GPT-5.5:openai:gpt-5.5').",
    ),
    gepa_prompts: str | None = typer.Option(
        None, help="`wmh eval grid`: dir of <label>.txt evolved prompts (enables +GEPA cells)."
    ),
    dataset_label: str | None = typer.Option(
        None, help="`wmh eval grid`: dataset name for the chart subtitle (default: suite id)."
    ),
    limit_traces: int | None = typer.Option(
        None, help="`wmh eval grid`: cap test traces (dry-run). Default: all."
    ),
    judge_model: str = typer.Option(
        "us.anthropic.claude-opus-4-8", help="`wmh eval grid`: pinned judge model (Bedrock)."
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
        envvar="WMH_E2B_TEMPLATE",
        help="Prebaked E2B sandbox template for --harness-backend e2b (default: "
        "$WMH_E2B_TEMPLATE; without one, sandboxes bootstrap node + the pi runner deps on "
        "first use).",
    ),
) -> None:
    """Score reconstruction fidelity, or run named example-local eval suites.

    Flows:
    - `wmh eval <trace files...>`: ad hoc replay scoring (open-loop, teacher-forced — the
      default mode).
    - `wmh eval <tasks.jsonl> --mode closed-loop`: a live agent runs tasks WITH the world model
      as its environment; `--harness-backend e2b` moves the pi-node harness process into pooled
      E2B sandboxes (the env stays the world model, all cells in parallel); score task success
      against gold assertions (see docs/reference/closed_loop.md).
    - `wmh eval list`: list named suites under `examples/<task>/evals/`.
    - `wmh eval run <suite>`: run a suite and save a local JSON result.
    - `wmh eval results optional-suite`: summarize local suite results.
    - `wmh eval agreement <a.json> <b.json>`: compare two closed-loop reports task-by-task
      (e.g. world-model vs real environment) — the outcome-agreement validity check.
    """
    args = tokens or []
    suite_roots = (
        [str(root) for root in _benchmark_roots()] if examples_root is None else [examples_root]
    )
    if mode not in ("open-loop", "closed-loop"):
        raise typer.BadParameter(f"unknown --mode {mode!r}; choose open-loop or closed-loop")
    if mode == "closed-loop":
        if len(args) != 1 or args[0] in ("list", "run", "results", "agreement"):
            raise typer.BadParameter("usage: wmh eval <tasks.jsonl> --mode closed-loop")
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
            raise typer.BadParameter("usage: wmh eval agreement <report_a.json> <report_b.json>")
        run_agreement(_console, report_a=args[1], report_b=args[2], threshold=threshold)
        return
    if args and args[0] == "list":
        if len(args) != 1:
            raise typer.BadParameter("usage: wmh eval list")
        _eval_list(suite_roots)
        return
    if args and args[0] == "results":
        if len(args) > 2:
            raise typer.BadParameter("usage: wmh eval results [suite]")
        suite_filter = args[1] if len(args) == 2 else None
        _eval_results(results_root, suite_roots, suite_filter, limit=limit)
        return
    if args and args[0] == "grid":
        if len(args) != 2:
            raise typer.BadParameter("usage: wmh eval grid <suite>")
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
            raise typer.BadParameter("usage: wmh eval grid-plot <result.json> [<result.json>...]")
        _eval_grid_plot(args[1:], out=out, dataset_label=dataset_label)
        return
    if args and args[0] == "grid-heatmap":
        if len(args) < 2:
            raise typer.BadParameter(
                "usage: wmh eval grid-heatmap <result.json> [<result.json>...]"
            )
        _eval_grid_heatmap(args[1:], out=out)
        return
    if args and args[0] == "run":
        if len(args) != 2:
            raise typer.BadParameter("usage: wmh eval run <suite>")
        _eval_run_suite(
            args[1],
            examples_roots=suite_roots,
            results_root=results_root,
            prompt_file=prompt_file,
            provider=provider,
            model=model,
            region=region,
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
            "provide trace files, or use `wmh eval list`, `wmh eval run <suite>`, "
            "or `wmh eval results`"
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
        provider=provider,
        model=model,
        chain=chain,
        region=region,
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
    suite = resolve_eval_suite(selector, examples_roots)
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
    provider: str,
    model: str,
    region: str | None,
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
    suite = resolve_eval_suite(selector, examples_roots)
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
    report = _run_eval_files(files, options, provider=provider, model=model, region=region)
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
        "config": {
            "provider": provider,
            "model": model,
            "region": region,
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
    split = 0.7 if train_split is None else train_split
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
    provider: str,
    model: str,
    region: str | None,
    chain: str | None = None,
) -> EvalReport:
    for path in files:
        if not path.exists():
            raise typer.BadParameter(f"trace file not found: {path}")
    provider_config = _provider_config(provider, model, region)
    llm = providers.provider_or_chain(provider_config, chain=chain)
    if isinstance(llm, providers.WaterfallProvider):
        _console.print("failover chain active (.wmh/fallback.toml) — world-model calls only")
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
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir."),
) -> None:
    """Show a model's knowledge base: the env's canonical facts, a folder of editable markdown.

    The printed directory IS the editing interface — open it in any editor. `rules.md`/
    `entities.md`/`schemas.md` are seeded at build (with knowledge enabled); `learned.md` collects
    the env's own cross-session notes; `grounded.md` caches web-search groundings.
    """
    store = WorldModelStore(root)
    resolved = _resolve_name(store, name)
    kb = KnowledgeBase(ArtifactPaths(store.resolve(resolved)).knowledge)
    _console.print(f"[bold]{kb.directory}[/bold]")
    if kb.is_empty:
        _console.print(
            "(empty — enable knowledge at build, drop *.md files in this folder, "
            "or PUT files via the serving API)"
        )
        return
    for file_name, content in kb.files().items():
        _console.print(f"\n[bold]## {file_name}[/bold]")
        _console.print(content.strip())


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
            "roles resolve from .wmh/settings.toml [models.worker|judge|summary]."
        ),
    ),
    model: str = typer.Option(None, help="Model id (pins all roles, like --provider)."),
    region: str = typer.Option(None, help="AWS region (Bedrock)."),
    embed_provider: str = typer.Option(
        "hashing",
        help=(
            "Facet embedder: hashing (offline but lexical-only — clusters by wording, not "
            "meaning; prefer a semantic embedder for real corpora) | bedrock | openai | "
            "azure_openai."
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
    traces = get_adapter("otel-genai").from_file(file)
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
    scenarios_file: str = typer.Argument(..., help="Scenario set JSON from `wmh scenarios build`."),
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
    scenario_set = ScenarioSet.load(scenarios_file)
    traces = get_adapter("otel-genai").from_file(file)
    if provider is not None or model is not None:
        store = WorldModelStore(root)
        model_dir = store.resolve(_resolve_name(store, name))
        override = _provider_config(provider or "bedrock", model or "claude-opus-4-8", region)
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


def _provider_config(provider: str, model: str, region: str | None) -> ProviderConfig:
    try:
        kind = ProviderKind(provider)
    except ValueError:
        kinds = ", ".join(k.value for k in ProviderKind)
        raise typer.BadParameter(f"unknown provider {provider!r}; choose one of: {kinds}") from None
    spec = resolve_provider_model(kind, model)
    return ProviderConfig(
        kind=kind,
        model_type=spec.model_type,
        model=spec.model_id,
        region=region,
    )


_SCENARIO_DEFAULT_PROVIDER = "bedrock"
_SCENARIO_DEFAULT_MODEL = "claude-opus-4-8"


def _role_provider_config(role: str, region: str | None) -> ProviderConfig | None:
    """ProviderConfig for a settings-defined model role, or None when the role isn't configured.

    Roles live in `.wmh/settings.toml` under `[models.worker|judge|summary]`; unset judge/summary
    fall back to worker (see `ModelsSettings.resolve`). A role's stored region wins over the
    generic `--region` flag — the flag also feeds the embedder, and e.g. a judge pinned to the
    one region where its model is enabled must not follow it.
    """
    configured = load_settings().models.resolve(role)
    if configured is None:
        return None
    config = _provider_config(configured.provider, configured.model, configured.region or region)
    return config.model_copy(
        update={"endpoint": configured.endpoint, "deployment": configured.deployment}
    )


def _scenario_role_llms(
    provider: str | None, model: str | None, region: str | None
) -> tuple[Provider, Provider, Provider]:
    """(summary, worker, judge) providers for scenario construction.

    Explicit `--provider`/`--model` flags pin ALL roles to that one model (the pre-roles
    behavior). Otherwise each role resolves from `.wmh/settings.toml`, falling back to worker,
    then to the built-in default. Judging benefits from a different family than the worker —
    a same-family judge carries self-preference bias toward the generator's outputs.
    """
    if provider is not None or model is not None:
        config = _provider_config(
            provider or _SCENARIO_DEFAULT_PROVIDER, model or _SCENARIO_DEFAULT_MODEL, region
        )
        llm = providers.get_provider(config)
        return llm, llm, llm
    default = _provider_config(_SCENARIO_DEFAULT_PROVIDER, _SCENARIO_DEFAULT_MODEL, region)
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
    try:
        kind = EmbedderKind(embed_provider)
    except ValueError:
        kinds = ", ".join(k.value for k in EmbedderKind)
        raise typer.BadParameter(
            f"unknown embed provider {embed_provider!r}; choose one of: {kinds}"
        ) from None
    if kind is EmbedderKind.HASHING:
        return HashingEmbedder(dim=embed_dim)
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
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir (example models are found too)."),
    steps: int = typer.Option(5, help="Max scenario steps to replay."),
    traces: str = typer.Option(
        None, "--traces", help="Trace file to sample the scenario from (default: the model's)."
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
    wm, resolved_name, _provider, model_root = _load_model_any(
        name, root, max_fidelity=max_fidelity
    )
    traces_file = Path(traces) if traces else _traces_for_root(model_root)
    if traces_file is None or not traces_file.exists():
        raise typer.BadParameter(
            f"no trace file found for {resolved_name!r} under {model_root}; pass --traces"
        )
    model_dir = WorldModelStore(str(model_root)).resolve(resolved_name)
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
            if not is_capacity_error(exc) or not _console.is_terminal:
                raise
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
    _console.print(f"\n{matches}/{len(done)} exact matches (run `wmh eval` for judged fidelity)")


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
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir (example models are found too)."),
    max_fidelity: bool = typer.Option(
        False, "--max-fidelity", help="Run with the online extras on (default: pure RAG)."
    ),
) -> None:
    """Step into the environment yourself: type actions, the world model returns observations."""
    wm, resolved_name, _provider, _model_root = _load_model_any(
        name, root, max_fidelity=max_fidelity
    )
    suggestions = _action_suggestions(wm)
    run_play_repl(_console, wm, resolved_name, task, suggestions=suggestions)


def _traces_for_root(model_root: Path) -> Path | None:
    """The trace corpus that built the models under `model_root`, when it ships alongside."""
    candidate = model_root / "traces.otel.jsonl"
    return candidate if candidate.exists() else None


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
    """Resolve a model across the project dir AND shipped examples, then load it.

    An explicit non-default `--root` keeps the old single-root behavior. Otherwise the picker
    spans `<root>/models/*` plus `examples/*/models/*`, labeling each with its source.
    """
    if root != ARTIFACT_DIR:
        wm, resolved, provider = _load_model(name, root, max_fidelity=max_fidelity)
        return wm, resolved, provider, Path(root)

    candidates: list[tuple[str, Path, str]] = []  # (label, store_root, name)
    local = WorldModelStore(root)
    candidates.extend((f"{n} (local)", Path(root), n) for n in local.list_names())
    for example_dir in _discover_examples():
        example_store = WorldModelStore(example_dir)
        candidates.extend(
            (f"{n} ({example_dir.name} example)", example_dir, n)
            for n in example_store.list_names()
        )

    if name is not None:
        matched = [c for c in candidates if c[2] == name]
        if not matched:
            have = ", ".join(c[2] for c in candidates) or "none built"
            raise typer.BadParameter(f"no world model named {name!r} (have: {have})")
        # Prefer the local build over a same-named example artifact.
        label, store_root, resolved = matched[0]
    elif not candidates:
        raise typer.BadParameter(
            "no world models found; run `wmh build` or try an example (examples/tau-bench)"
        )
    elif len(candidates) == 1:
        label, store_root, resolved = candidates[0]
    elif _console.is_terminal:
        labels = [c[0] for c in candidates]
        chosen = _select_from(labels)
        label, store_root, resolved = candidates[labels.index(chosen)]
    else:
        have = ", ".join(c[2] for c in candidates)
        raise typer.BadParameter(f"multiple world models ({have}); pass --name")

    wm, resolved, provider = _load_model(resolved, str(store_root), max_fidelity=max_fidelity)
    return wm, resolved, provider, store_root


def _select_from(labels: list[str]) -> str:
    """Interactive picker over pre-rendered labels (arrow keys on a TTY)."""
    from wmh.cli import ui as _ui  # package-internal: reuse the wizard's picker machinery

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
    try:
        if name is not None:
            store.resolve(name)  # validates existence, raising a friendly error if missing
            return name
        # Only enumerate full model summaries when we actually need the picker (>1 model on a TTY).
        # `list_names` is cheap (a dir scan); `list_info` reads every config/metrics/frontier file.
        if _console.is_terminal and len(store.list_names()) > 1:
            return select_model(_console, store.list_info())
        return store.resolve(None).name
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def _benchmark_roots() -> tuple[Path, ...]:
    """Every root holding self-contained task dirs: examples/ + the capture member's data dirs."""
    repo = Path(__file__).resolve().parents[2]
    return (repo / "examples", repo / "packages" / "environment-capture")


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
    """Whether `name` would resolve via `wmh examples run` — keeps list/hint/run in agreement."""
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
    provider: str = typer.Option("bedrock", "--provider", help="Provider running the model."),
    model: str = typer.Option("us.anthropic.claude-opus-4-8", help="Model id (environment LLM)."),
    region: str | None = typer.Option(None, help="AWS region (Bedrock)."),
    deployment: str | None = typer.Option(None, help="Azure OpenAI deployment name."),
    api_version: str | None = typer.Option(None, help="Azure OpenAI API version."),
    endpoint: str | None = typer.Option(None, help="Azure OpenAI / custom base URL."),
    real_arg: list[str] | None = _RESEARCH_REAL_ARG,
    real_timeout: float | None = typer.Option(
        None, "--real-timeout", help="Abort a real sandbox run after N seconds."
    ),
    out: str | None = typer.Option(None, help="Path to write the ConcurrencyScalingReport JSON."),
    examples_root: str | None = typer.Option(None, help="Examples dir. Default: repo-local."),
) -> None:
    """Measure the concurrency scaling law: batch wall-clock vs. how many scenarios run at once.

    Reconstructs a fixed batch of N held-out scenarios from `suite`'s corpus at each concurrency
    level, timing the world-model batch and (with `--side both`) the matching real-sandbox batch,
    to give the time differential T_real(W)/T_world(W). Reuses the suite's corpus + config
    (train_split, top_k, prompt) and the example's `run.sh`. See `wmh.research.concurrency_run`.
    """
    if scenarios < 1:
        raise typer.BadParameter("--scenarios must be at least 1")
    try:
        which = Side(side)
    except ValueError:
        allowed = ", ".join(s.value for s in Side)
        raise typer.BadParameter(f"--side must be one of: {allowed}") from None
    # Validate the provider up front, not lazily inside a worker thread mid-sweep.
    try:
        provider_kind = ProviderKind(provider)
    except ValueError:
        kinds = ", ".join(k.value for k in ProviderKind)
        raise typer.BadParameter(f"unknown provider {provider!r}; choose one of: {kinds}") from None
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
    resolved = resolve_eval_suite(suite, suite_roots)
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
    # work `wmh eval run <suite>` scores.
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
        return providers.get_provider(
            ProviderConfig(
                kind=provider_kind,
                model=model,
                region=region,
                deployment=deployment,
                api_version=api_version,
                endpoint=endpoint,
            )
        )

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
    from the optional `viz` extra — the harness runtime must not require them. A missing extra
    raises a plain ImportError naming the module (`uv sync --extra viz`); it is not caught.
    """
    from wmh.research.concurrency_plot import render_report

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
    explains it). Pass the per-benchmark report JSONs (`wmh research concurrency <suite> --out ...`)
    in the order you want them coloured.

    Imported lazily for the same reason as `plot-concurrency` (the optional matplotlib viz extra).
    """
    from wmh.research.concurrency_plot import render_cost, render_speedup

    try:
        speedup_path = render_speedup(reports, out_speedup)
        cost_path = render_cost(reports, out_cost)
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _console.print(f"wrote figures -> {speedup_path}, {cost_path}")


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
    store = WorldModelStore(root)
    resolved_name = _resolve_name(store, name)
    model_dir = store.resolve(resolved_name)
    config = load_config(model_dir)
    provider = wrap_provider_with_retries(
        providers.get_provider(config.serve_provider_config()),
        on_retry=_NARRATOR.on_retry,
        sleep=_NARRATOR.sleep,
    )
    world_model = WorldModel.load(
        str(model_dir), provider, telemetry_root=store.root, max_fidelity=max_fidelity
    )
    return world_model, resolved_name, provider


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
