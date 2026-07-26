"""Terminal UX for the `wmh` CLI: guided creation/selection flows, an animated build pipeline, and
the interactive play REPL.

Everything that talks to `rich` lives here so the engine stays headless. Responsibilities:

- `run_build_wizard` interactively fills in any missing `wmh build` inputs (name, traces, provider,
  region, budget), so a bare `wmh build` becomes a guided creation flow.
- `select_model` shows a numbered picker so a user can choose which built world model to run when
  `--name` is omitted and several exist.
- `RichBuildReporter` implements `wmh.engine.reporting.BuildReporter`, turning build events into a
  guided, animated pipeline (stage lines + a live GEPA rollout progress bar) on a TTY, and into
  plain one-line-per-event output when piped (non-TTY), so logs stay legible.
- `run_play_repl` drives the human-in-the-loop demo: the user types actions, the world model
  answers, and the evolving session state (scratchpad + history) is rendered each turn.
"""

from __future__ import annotations

import os
import sys
from collections import deque
from collections.abc import Callable

import click
import typer
from pydantic import BaseModel
from rich.console import Console, Group
from rich.control import Control
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.segment import ControlType
from rich.table import Table
from rich.text import Text

from wmh.config import (
    PROVIDER_ENV_VARS,
    ModelInfo,
    normalize_name,
    upsert_env_var,
    validate_name,
)
from wmh.core.types import Action, ActionKind, EnvState, Session
from wmh.engine.play import PlayTurn, parse_action, play_turn
from wmh.engine.world_model import WorldModel
from wmh.providers import verify_all, verify_embedder
from wmh.providers.base import ProviderConfig, ProviderKind, VerifyResult
from wmh.providers.models import resolve_provider_model

# A reader takes a fully-rendered prompt string and returns the user's typed line.
PromptReader = Callable[[str], str]

# Stage glyphs reused by the animated and plain reporters.
_CHECK = "[green]✓[/green]"

# Rows in the GEPA activity window (the fixed-height inner-loop stream under the progress bar).
_ACTIVITY_WINDOW_LINES = 8

# Serve providers offered in the wizard picker, with the model ids each supports. The first model
# in each list is the suggested default. Keep these in sync with the provider backends.
_PROVIDER_MODELS: dict[str, list[str]] = {
    "openai": ["gpt-5.5", "gpt-5.5-pro", "gpt-5.4", "gpt-5.4-mini"],
    "anthropic": [
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    ],
    # Keep this picker curated to inference profiles verified by the normal interactive flow.
    # The full canonical registry remains available to flags and programmatic callers.
    "bedrock": ["claude-opus-4-8", "claude-opus-4-7", "claude-haiku-4-5"],
    # openai_responses (the Responses API) stays flag-only (`wmh build --provider
    # openai_responses`); the wizard list keeps to the four everyday backends.
    "azure": ["gpt-5.5", "gpt-5.4"],
}
_DEFAULT_REGIONS: dict[str, str] = {"bedrock": "us-east-1"}

# Default GEPA judge model per provider: a cheap, fast model of the same backend. Providers
# without an entry judge on the serve model itself.
_JUDGE_MODEL_DEFAULTS: dict[str, str] = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-5.4-mini",
    "bedrock": "claude-haiku-4-5",
}


def judge_model_default(provider: str | None, serve_model: str) -> str:
    """The GEPA judge model when none was chosen: cheap per-provider, else the serve model."""
    return _JUDGE_MODEL_DEFAULTS.get(provider or "", serve_model)


# Embedders offered in the wizard, with the embeddings-model ids each provider-backed one supports
# (None = the offline hashing embedder, no model). First entry is the suggested default.
_EMBEDDERS: dict[str, list[str] | None] = {
    "hashing": None,
    "openai": ["text-embedding-3-small", "text-embedding-3-large"],
    "bedrock": ["amazon.titan-embed-text-v2:0"],
    "azure": ["text-embedding-3-small", "text-embedding-3-large"],
}


class BuildParams(BaseModel):
    """The fully-resolved inputs for a build, as collected by the creation wizard."""

    name: str
    # Trace source: which adapter ingests, and where the traces come from. `source` is a registered
    # adapter name (otel-genai, chat-json, braintrust, phoenix, langfuse, langsmith, posthog,
    # mastra). A source is read from a `file` export or `pull`ed live from its vendor API
    # (`project`/`api_key`, optionally windowed by `since`/`limit`).
    source: str = "otel-genai"
    file: str | None = None
    pull: bool = False
    project: str | None = None
    api_key: str | None = None
    since: str | None = None
    limit: int | None = None
    # Deprecated alias for `--source <name> --pull`; kept so old invocations keep working.
    vendor: str | None = None
    # None = not chosen yet: the wizard suggests the first provider with credentials present,
    # and the non-interactive path falls back to bedrock.
    provider: str | None = None
    model: str = "claude-opus-4-8"
    # Canonical GEPA judge model type; None = pick the cheap per-provider default.
    judge_model: str | None = None
    region: str | None = None
    fidelity: str = "medium"
    train_split: float = 0.8
    embed_provider: str = "hashing"
    embed_model: str | None = None
    embed_dim: int = 512


# Trace sources offered in the build wizard: name -> (label, can-pull-live). File-only sources
# (otel-genai/chat-json) read an export; the rest can also pull live from a vendor API. Any adapter
# registered in `wmh.ingest` is usable via `--source` even if it isn't surfaced here.
_SOURCES: dict[str, tuple[str, bool]] = {
    "otel-genai": ("OpenTelemetry GenAI spans (file)", False),
    "chat-json": ("Chat / tool-call log, OpenAI-style (file)", False),
    "braintrust": ("Braintrust", True),
    "phoenix": ("Arize Phoenix", True),
    "langfuse": ("Langfuse", True),
    "langsmith": ("LangSmith", True),
    "posthog": ("PostHog", True),
    "mastra": ("Mastra", True),
}


def _collect_source(
    console: Console,
    ask: PromptReader,
    ask_secret: PromptReader,
    defaults: BuildParams,
    interactive: bool,
) -> tuple[str, str | None, bool, str | None, str | None]:
    """Pick the trace source and where its traces come from.

    Returns `(source, file, pull, project, api_key)`. The source is a registered adapter; traces
    are read from a `file` export or pulled live (`pull`, with `project` + a masked `api_key`). A
    source/file/pull already supplied via flags is honored without re-prompting.
    """
    # A source supplied entirely via flags (a file, a pull, or a non-default --source) is accepted
    # as-is; otherwise prompt for the adapter.
    source = defaults.source
    if source == "otel-genai" and not defaults.file and not defaults.pull:
        names = list(_SOURCES)
        labels = [_SOURCES[n][0] for n in names]
        chosen = _select(
            console, ask, "Trace source", labels, _SOURCES[source][0], interactive=interactive
        )
        source = names[labels.index(chosen)]

    can_pull = _SOURCES.get(source, ("", False))[1]
    file = defaults.file
    pull = defaults.pull
    project = defaults.project
    api_key = defaults.api_key

    # A pull-capable vendor with no file already given: offer file-vs-pull.
    if can_pull and not file and not pull:
        mode = _select(
            console,
            ask,
            "Read traces from",
            ["exported file", "live API pull"],
            None,
            interactive=interactive,
        )
        pull = mode == "live API pull"

    if pull:
        if not project:
            project = _prompt_text(console, ask, f"{source} project / workspace", None) or None
        # Read the key without echo (it's a secret); blank falls back to the source's env var.
        if not api_key:
            api_key = ask_secret(f"{source} API key (blank = use env var): ").strip() or None
        return source, None, True, project, api_key

    # File source: require a path, re-prompting on empty input rather than erroring out.
    while not file:
        file = _prompt_text(
            console,
            ask,
            f"Path to the {source} export to ingest",
            None,
            example="packages/environment-capture/tau-bench/traces.otel.jsonl",
        )
        if not file:
            console.print("[red]a trace export path is required[/red]")
    return source, file, False, project, api_key


def select_provider_and_model(
    console: Console,
    ask: PromptReader,
    ask_secret: PromptReader,
    *,
    default_provider: str | None,
    default_model: str | None,
    default_region: str | None,
    interactive: bool,
    check: Callable[[ProviderConfig], VerifyResult],
) -> tuple[str, str, str | None]:
    """The wizard's serve-provider block: pick provider + model (+ region), creds, live verify.

    Providers with credentials present are annotated and the first becomes the suggested default
    (none otherwise); a failed live ping loops back to the picker with the failed pick as the
    retry default. Also reused by `wmh demo`'s switch-provider flow. Returns
    (provider, model, region).
    """
    providers = list(_PROVIDER_MODELS)
    with_creds = [p for p in providers if _has_credentials(p)]
    # Name the actual variable so a key inherited from the shell (e.g. exported in ~/.zshrc)
    # is traceable — "api key exists" alone reads as a mystery when .env doesn't have it.
    notes = {p: _creds_note(p) for p in with_creds}
    provider_default = default_provider or (with_creds[0] if with_creds else None)
    while True:
        provider = _select(
            console,
            ask,
            "Serve provider",
            providers,
            provider_default,
            interactive=interactive,
            notes=notes,
        )
        _ensure_credentials(console, ask_secret, provider)
        model = _select(
            console,
            ask,
            "Serve model type",
            _PROVIDER_MODELS[provider],
            default_model,
            interactive=interactive,
            collapsed=2,
        )
        region = None
        if provider == "bedrock":
            region_default = default_region or _DEFAULT_REGIONS.get(provider)
            region = _prompt_text(console, ask, "AWS region", region_default) or None
        # Live ping now, not at the end: a bad key or model id loops straight back to the
        # picker (the failed pick becomes the suggested retry default).
        console.print(f"verifying {provider}…")
        model_spec = resolve_provider_model(ProviderKind(provider), model)
        ping = check(
            ProviderConfig(
                kind=ProviderKind(provider),
                model_type=model_spec.model_type,
                model=model_spec.model_id,
                region=region,
            )
        )
        if ping.ok:
            console.print(f"  {_CHECK} {provider} ({escape(model)}) reachable")
            return provider, model, region
        console.print(
            f"  [red]✗ {provider} ({escape(model)}) failed[/red]: {escape(ping.detail or '')}"
        )
        console.print("  [yellow]fix the credentials or pick a different provider/model[/yellow]")
        provider_default = provider


def run_build_wizard(
    console: Console,
    defaults: BuildParams,
    reader: PromptReader | None = None,
    verify: Callable[[ProviderConfig], VerifyResult] | None = None,
    verify_embed: Callable[[ProviderConfig], VerifyResult] | None = None,
) -> BuildParams:
    """Guided creation flow: prompt for each build input, pre-filled with `defaults`.

    Returns a resolved `BuildParams`. Any value already set in `defaults` (i.e. passed as a flag)
    becomes the suggested default the user can accept with Enter. The serve provider and any
    provider-backed embedder are live-pinged right after their model pick — a failure loops back
    to the picker instead of surfacing after the wizard. `verify`/`verify_embed` exist so tests
    can stub the pings. Raises `ValueError` if no trace source (file or vendor) is provided.
    """
    interactive = reader is None
    check = verify if verify is not None else (lambda cfg: verify_all([cfg])[0])
    check_embed = verify_embed if verify_embed is not None else verify_embedder
    base_ask = reader if reader is not None else (lambda text: console.input(text))
    base_secret = (
        reader if reader is not None else (lambda text: console.input(text, password=True))
    )

    def _eof_aborts(read: PromptReader) -> PromptReader:
        # Exhausted piped stdin (or Ctrl-D) aborts the wizard cleanly ("Aborted.", exit 1)
        # instead of leaking an EOFError traceback from whichever prompt was being read.
        def wrapped(text: str) -> str:
            try:
                return read(text)
            except EOFError:
                raise typer.Abort() from None

        return wrapped

    ask = _eof_aborts(base_ask)
    ask_secret = _eof_aborts(base_secret)

    console.print(
        Panel(
            "Let's create a world model. Press Enter to accept the [dim]default[/dim] in brackets.",
            title="[bold cyan]wmh build[/bold cyan]",
            border_style="cyan",
        )
    )

    # Whitespace is dash-joined ('tau bench' -> 'tau-bench') rather than rejected; a name
    # that is still unsafe (path separators, ...) re-prompts with the validation message
    # instead of escaping as a ValueError traceback. An invalid name arriving via --name is
    # dropped from the suggested default — otherwise Enter would re-offer it forever.
    try:
        name_default = validate_name(normalize_name(defaults.name))
    except ValueError:
        name_default = None
    while True:
        raw_name = _prompt_text(console, ask, "Name this world model", name_default)
        name = normalize_name(raw_name)
        try:
            validate_name(name)
        except ValueError as err:
            console.print(f"[red]{escape(str(err))}[/red]")
            continue
        if name != raw_name:
            console.print(f"  [dim]using[/dim] {escape(name)}")
        break

    # Trace source: pick which adapter ingests, then where its traces come from — a file export or a
    # live pull from the vendor's API (with project + masked API key). A source already supplied via
    # flags (--file/--pull/--source) is kept without re-prompting.
    source, file, pull, project, api_key = _collect_source(
        console, ask, ask_secret, defaults, interactive
    )

    provider, model, region = select_provider_and_model(
        console,
        ask,
        ask_secret,
        default_provider=defaults.provider,
        default_model=defaults.model,
        default_region=defaults.region,
        interactive=interactive,
        check=check,
    )

    # GEPA judge model: defaults to a cheap model of the same provider (haiku / gpt-5.4-mini);
    # picking the serve model itself is always on the list. Same inline verify + retry.
    judge_default = defaults.judge_model or judge_model_default(provider, model)
    judge_options = list(dict.fromkeys([judge_default, model, *_PROVIDER_MODELS[provider]]))
    judge_notes = {model: "same as serve"} if model != judge_default else {}
    while True:
        judge_model = _select(
            console,
            ask,
            "GEPA judge model type",
            judge_options,
            judge_default,
            interactive=interactive,
            notes=judge_notes,
            collapsed=2,
        )
        if judge_model == model:
            break  # the serve model was just verified
        console.print(f"verifying {provider} (judge)…")
        judge_spec = resolve_provider_model(ProviderKind(provider), judge_model)
        ping = check(
            ProviderConfig(
                kind=ProviderKind(provider),
                model_type=judge_spec.model_type,
                model=judge_spec.model_id,
                region=region,
            )
        )
        if ping.ok:
            console.print(f"  {_CHECK} {provider} ({escape(judge_model)}) reachable")
            break
        console.print(
            f"  [red]✗ {provider} ({escape(judge_model)}) failed[/red]: {escape(ping.detail or '')}"
        )
        console.print("  [yellow]pick a different judge model[/yellow]")
        judge_default = judge_model

    # Build effort is a tier, not an iteration count (raw budgets live in the Python API).
    # `low` picks the corpus-signature's estimated-best config for free; the searching tiers
    # spend more to find (and verify) a config that beats that estimate — each is floored at the
    # low estimate, so more build effort never ships worse than low. The chosen config activates
    # at serve time with `--max-fidelity` (a plain `wmh serve` stays pure RAG).
    console.print(
        "  [dim]low estimates the best config free; medium/high/max search harder for one that "
        "beats it (floored at low). Serve the chosen config with --max-fidelity.[/dim]"
    )
    fidelity_notes = {
        "low": "free · the estimated-best config for your traces, no search",
        "medium": "+ light prompt optimization & a cheap-lever search over low",
        "high": "+ full config search (knowledge/verify/grounding) — recommended",
        "max": "+ deep optimization & exhaustive search — highest cost, to be certain",
    }
    fidelity = _select(
        console,
        ask,
        "Fidelity",
        ["low", "medium", "high", "max"],
        defaults.fidelity,
        interactive=interactive,
        notes=fidelity_notes,
    )

    # Retrieval phi embedder: offline lexical hashing is the measured-best default at EVERY
    # tier (semantic phi lost to char-trigram hashing on all benchmarks — PR #72's matrix and
    # the tier ladder's tau decline agree; command outputs are predicted by literal token
    # overlap). Provider-backed semantic embeddings stay available as an explicit choice for
    # experiments. Phi dimensionality keeps its default (index and query embedders must agree).
    embed_default = defaults.embed_provider
    while True:
        embed_provider = _select(
            console,
            ask,
            "Embedder",
            list(_EMBEDDERS),
            embed_default,
            interactive=interactive,
            collapsed=2,
        )
        embed_model = defaults.embed_model
        embed_models = _EMBEDDERS[embed_provider]
        if embed_models is None:
            break  # offline hashing embedder: nothing to verify
        if embed_provider != provider:
            _ensure_credentials(console, ask_secret, embed_provider)
        embed_model = _select(
            console,
            ask,
            "Embeddings model id",
            embed_models,
            embed_model or embed_models[0],
            interactive=interactive,
        )
        # Same inline ping for the embed path, with the embeddings model and phi dimension
        # stamped on (mirrors HarnessConfig.embed_provider_config).
        console.print(f"verifying embed:{embed_provider}…")
        ping = check_embed(
            ProviderConfig(
                kind=ProviderKind(embed_provider),
                model=embed_model,
                embed_model=embed_model,
                embed_dim=defaults.embed_dim,
                region=region if embed_provider == "bedrock" else None,
            )
        )
        if ping.ok:
            console.print(f"  {_CHECK} embed:{embed_provider} ({escape(embed_model)}) reachable")
            break
        console.print(
            f"  [red]✗ embed:{embed_provider} ({escape(embed_model)}) failed[/red]: "
            f"{escape(ping.detail or '')}"
        )
        console.print("  [yellow]fix the credentials or pick a different embedder[/yellow]")
        embed_default = embed_provider

    return BuildParams(
        name=name,
        source=source,
        file=file,
        pull=pull,
        project=project,
        api_key=api_key,
        since=defaults.since,
        limit=defaults.limit,
        provider=provider,
        model=model,
        judge_model=judge_model,
        region=region,
        fidelity=fidelity,
        train_split=defaults.train_split,
        embed_provider=embed_provider,
        embed_model=embed_model,
        embed_dim=defaults.embed_dim,
    )


def _picker_fits(console: Console, row_count: int) -> bool:
    """Whether the arrow-key picker can run: a real TTY and every row fits on screen at once
    (the repaint moves the cursor up over the block, which breaks if the block scrolled)."""
    return console.is_terminal and sys.stdin.isatty() and row_count + 2 <= console.size.height


def _split_keys(raw: str) -> list[str]:
    """Split one getchar() read into individual key sequences.

    Fast key repeat (holding an arrow) can deliver several escape sequences in a single raw
    read; treating the batch as one unknown sequence would drop them all as inert.
    """
    keys: list[str] = []
    i = 0
    while i < len(raw):
        if raw[i] != "\x1b":
            keys.append(raw[i])
            i += 1
            continue
        if raw[i : i + 2] == "\x1b[":
            j = i + 2
            while j < len(raw) and not ("@" <= raw[j] <= "~"):
                j += 1
            keys.append(raw[i : j + 1])
            i = j + 1
        elif raw[i : i + 2] == "\x1bO":
            keys.append(raw[i : i + 3])
            i += 3
        else:
            keys.append("\x1b")
            i += 1
    return keys


def _decode_key(seq: str) -> str:
    """Map one click.getchar() sequence to a picker key.

    getchar returns a whole escape sequence per call ('\x1b[A'), so nothing can desync: plain
    and application-mode arrows decode to 'up'/'down', any other escape sequence (modified
    arrows, PgUp, Delete, bare ESC) is the inert 'esc', and a plain character passes through.
    """
    if seq in ("\x1b[A", "\x1bOA"):
        return "up"
    if seq in ("\x1b[B", "\x1bOB"):
        return "down"
    if seq.startswith("\x1b"):
        return "esc"
    return seq


def _step_selection(key: str, index: int, count: int) -> tuple[int, bool]:
    """Picker key reducer: next highlighted index and whether the selection was accepted.

    Arrows (and vi j/k) move with wraparound, Enter accepts the highlight, a digit jump-selects
    that option; anything else is inert.
    """
    if key in ("up", "k"):
        return (index - 1) % count, False
    if key in ("down", "j"):
        return (index + 1) % count, False
    if key in ("\r", "\n"):
        return index, True
    # ASCII-decimal only: '²'.isdigit() is True but int('²') raises.
    if key.isascii() and key.isdecimal() and 1 <= int(key) <= count:
        return int(key) - 1, True
    return index, False


def _arrow_select(
    console: Console, rows: list[str], index: int, hidden_rows: list[str] | None = None
) -> int:
    """Drive an up/down picker over pre-rendered `rows`; return the chosen index.

    Keys come from click.getchar(): raw mode handled portably (termios on Unix, msvcrt on
    Windows), one whole escape sequence per call, Ctrl-C raised as KeyboardInterrupt for
    click's usual clean abort. EOF (closed stdin) aborts rather than spinning.

    `hidden_rows` collapse behind a dim "… N more" row; navigating (or jump-selecting) onto it
    reveals the rest in place. The returned index is into rows + hidden_rows.
    """
    visible = list(rows)
    hidden = list(hidden_rows or [])
    painted_height = 0

    def current_rows() -> list[str]:
        more = [f"[dim]… {len(hidden)} more[/dim]"] if hidden else []
        return visible + more

    def paint(current: int) -> None:
        nonlocal painted_height
        shown = current_rows()
        if painted_height:
            console.control(Control.move(y=-painted_height))
        for i, row in enumerate(shown):
            console.control(Control((ControlType.ERASE_IN_LINE, 2)))
            pointer = "[bold cyan]\u276f[/bold cyan]" if i == current else " "
            console.print(f" {pointer} {row}", highlight=False, no_wrap=True, overflow="ellipsis")
        painted_height = len(shown)

    while True:
        paint(index)
        try:
            seq = click.getchar()
        except EOFError:
            raise typer.Abort() from None
        if seq == "":
            raise typer.Abort()
        for key in _split_keys(seq):
            index, accepted = _step_selection(_decode_key(key), index, len(current_rows()))
            if hidden and index == len(visible):
                # Landed on the "more" row: reveal the rest in place; the highlight stays
                # put, now on the first revealed option.
                visible.extend(hidden)
                hidden = []
                accepted = False
            if accepted:
                paint(index)
                return index


def _select(
    console: Console,
    ask: PromptReader,
    label: str,
    options: list[str],
    default: str | None,
    *,
    interactive: bool = False,
    notes: dict[str, str] | None = None,
    collapsed: int | None = None,
) -> str:
    """Pick one of `options`: an arrow-key picker on a real TTY, else a numbered prompt.

    The Enter-default is `default` when present in `options`; a non-matching default falls
    back to the first option, and None means no Enter-default at all (the provider picker: a
    default exists only when creds do; the model picker: the choice is always explicit).
    `notes` adds a dim annotation after an option's name (e.g. "api key exists"). `collapsed`
    shows only the first N options in the arrow picker behind a "… more" row (the numbered
    fallback always lists everything, so scripted input is unaffected).
    """
    if default in options:
        chosen_default = default
    elif default is not None:
        chosen_default = options[0]
    else:
        chosen_default = None
    notes = notes or {}

    def row(opt: str) -> str:
        note = f"  [dim]({notes[opt]})[/dim]" if opt in notes else ""
        marker = "  [dim](default)[/dim]" if opt == chosen_default else ""
        return f"{escape(opt)}{note}{marker}"

    if interactive and _picker_fits(console, len(options)):
        console.print(f"[bold]{label}[/bold] [dim](up/down + Enter)[/dim]:")
        start = options.index(chosen_default) if chosen_default is not None else 0
        rows = [row(opt) for opt in options]
        if collapsed is not None and start < collapsed < len(options):
            return options[_arrow_select(console, rows[:collapsed], start, rows[collapsed:])]
        return options[_arrow_select(console, rows, start)]

    console.print(f"[bold]{label}[/bold]:")
    for i, opt in enumerate(options, start=1):
        console.print(f"  [cyan]{i}[/cyan]. {row(opt)}")
    prompt = f"[dim]\\[{escape(chosen_default)}][/dim] > " if chosen_default is not None else "> "
    while True:
        raw = ask(prompt).strip()
        if not raw and chosen_default is not None:
            return chosen_default
        choice = _parse_int(raw)
        if choice is not None and 1 <= choice <= len(options):
            return options[choice - 1]
        if raw in options:  # allow typing the option name directly
            return raw
        console.print(f"[red]pick 1-{len(options)} or an option name[/red]")


def _provider_env_vars(provider: str) -> list[str]:
    """The env vars `provider` reads its credentials from ([] for unknown/offline kinds)."""
    try:
        return PROVIDER_ENV_VARS[ProviderKind(provider)]
    except (ValueError, KeyError):
        return []


def _creds_note(provider: str) -> str:
    """Picker annotation for a provider whose credentials are present, naming what was found."""
    env_vars = _provider_env_vars(provider)
    return f"{env_vars[0]} set" if len(env_vars) == 1 else "creds set"


def _has_credentials(provider: str) -> bool:
    """Offline presence check: every credential env var for `provider` is set (not validated)."""
    env_vars = _provider_env_vars(provider)
    return bool(env_vars) and all(os.environ.get(var) for var in env_vars)


def _ensure_credentials(console: Console, ask_secret: PromptReader, provider: str) -> None:
    """Prompt for any missing credential env vars and persist entered values to `.env`.

    Presence only — the live ping that confirms the creds actually work happens once before
    the build (see `wmh build`). Enter skips a var, leaving it to the shell environment.
    """
    for var in _provider_env_vars(provider):
        if os.environ.get(var):
            console.print(f"  {_CHECK} {var} is set")
            continue
        prompt = f"  [bold]{var}[/bold] [dim](saved to .env; Enter to skip)[/dim]: "
        value = ask_secret(prompt).strip()
        if value:
            try:
                upsert_env_var(var, value)
            except (ValueError, OSError) as err:
                # Persistence refused (symlinked .env, O_NOFOLLOW's ELOOP on a swapped link,
                # unwritable dir, ...): the session still gets the credential.
                os.environ[var] = value
                console.print(f"  [yellow]{var} not saved: {escape(str(err))}[/yellow]")
            else:
                console.print(f"  {_CHECK} {var} saved to .env")
        else:
            console.print(f"  [yellow]{var} still unset[/yellow]")


def select_option(
    console: Console,
    label: str,
    options: list[str],
    *,
    notes: dict[str, str] | None = None,
    reader: PromptReader | None = None,
) -> str:
    """Pick one of `options` (arrow-key picker on a TTY, numbered prompt otherwise)."""
    ask = reader if reader is not None else (lambda text: console.input(text))
    return _select(console, ask, label, options, None, interactive=True, notes=notes)


def select_model(
    console: Console, infos: list[ModelInfo], reader: PromptReader | None = None
) -> str:
    """Show a numbered picker and return the chosen model name.

    Re-prompts on invalid input. With a single model it returns that name without prompting.
    """
    if len(infos) == 1:
        return infos[0].name
    ask = reader if reader is not None else (lambda text: console.input(text))
    notes = {
        info.name: f"held-out {info.held_out_accuracy:.2f}"
        for info in infos
        if info.held_out_accuracy is not None
    }
    return _select(
        console,
        ask,
        "Select a world model",
        [info.name for info in infos],
        None,
        interactive=reader is None,
        notes=notes,
    )


def _prompt_text(
    console: Console,
    ask: PromptReader,
    label: str,
    default: str | None,
    *,
    example: str | None = None,
) -> str:
    # Escape interpolated values: "default" or anything with [...] is valid rich markup and would
    # otherwise be swallowed (rendered invisibly) instead of shown. A prompt with no default can
    # carry a grey `example` hint so the user sees the expected shape of the answer.
    if default:
        suffix = f" [dim]\\[{escape(default)}][/dim]"
    elif example:
        suffix = f" [dim](e.g. {escape(example)})[/dim]"
    else:
        suffix = ""
    value = ask(f"[bold]{label}[/bold]{suffix}: ").strip()
    return value or (default or "")


def _prompt_int(console: Console, ask: PromptReader, label: str, default: int) -> int:
    while True:
        raw = ask(f"[bold]{label}[/bold] [dim]\\[{default}][/dim]: ").strip()
        if not raw:
            return default
        value = _parse_int(raw)
        if value is not None and value >= 0:
            return value
        console.print("[red]enter a non-negative whole number[/red]")


def _parse_int(raw: str) -> int | None:
    """Parse a base-10 integer, or None. Unlike `str.isdigit`, this rejects unicode digit
    characters (e.g. superscripts) that `isdigit()` accepts but `int()` rejects with ValueError."""
    try:
        return int(raw)
    except ValueError:
        return None


class RichBuildReporter:
    """A `BuildReporter` that renders the build as a guided pipeline.

    On a TTY it shows stage lines and a live progress bar for GEPA rollouts (with the running
    held-out score). When output is piped (`console.is_terminal` is false) it degrades to a single
    plain line per event — no spinners, no carriage returns — so captured logs stay readable.
    """

    def __init__(self, console: Console, model_name: str) -> None:
        self._console = console
        self._name = model_name
        self._tty = console.is_terminal
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        self._live: Live | None = None
        self._activity: deque[str] = deque(maxlen=_ACTIVITY_WINDOW_LINES)

    def ingest_done(self, traces: int, steps: int) -> None:
        self._stage(f"ingested {traces} traces → normalized {steps} steps")

    def split_done(self, train: int, val: int, test: int) -> None:
        self._stage(f"split {train} train / {val} val / {test} test traces")

    def index_done(self, steps: int) -> None:
        self._stage(f"indexed {steps} steps into the replay buffer")

    def optimize_start(self, budget: int) -> None:
        self._stage(f"optimizing env prompt with GEPA (budget {budget} metric calls)")
        if self._tty and budget > 0:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("{task.fields[score]}"),
                TimeElapsedColumn(),
                console=self._console,
            )
            self._task_id = self._progress.add_task(
                "GEPA metric calls", total=budget, score="score n/a"
            )
            # One Live region holding the fixed-height activity window + the bar: GEPA's inner
            # loop (proposals, selections, judge notes) streams inside the window instead of
            # scrolling the terminal. Not transient: the last frame (bar at 100% + final
            # activity) stays in scrollback above the GEPA-done stage line. get_renderable (not
            # per-event update()) so only the refresh thread repaints, and stray prints are
            # redirected through the region — anything written straight to the terminal mid-Live
            # scrolls the region and leaves orphaned frame headers behind.
            self._live = Live(
                get_renderable=self._render_optimize,
                console=self._console,
                refresh_per_second=4,
                transient=False,
                redirect_stdout=True,
                redirect_stderr=True,
            )
            self._live.start()

    def _render_optimize(self) -> Group:
        lines = list(self._activity)
        pad = [""] * (_ACTIVITY_WINDOW_LINES - len(lines))
        body = Text("\n".join(lines + pad), style="dim", no_wrap=True, overflow="ellipsis")
        window = Panel(
            body,
            title="[dim]GEPA activity[/dim]",
            border_style="bright_black",
            height=_ACTIVITY_WINDOW_LINES + 2,
        )
        assert self._progress is not None
        return Group(window, self._progress)

    def activity(self, line: str) -> None:
        # Width-safe: judge critiques can contain combining marks / double-width glyphs whose
        # cell width the terminal and rich disagree on; one such line in the live region makes
        # every repaint's cursor math drift, shedding an orphaned frame header per refresh.
        text = "".join(ch if " " <= ch <= "~" else "?" for ch in line.strip())
        if not text:
            return
        if self._live is not None:
            self._activity.append(text)  # the Live's refresh thread picks this up
        # Non-TTY logs keep their sparse heartbeat; streaming every inner-loop line would flood.

    def rollout(self, done: int, budget: int, score: float | None) -> None:
        label = f"avg fidelity {score:.3f}" if score is not None else "score n/a"
        if self._progress is not None and self._task_id is not None:
            # The budget is an estimate (GEPA treats max_metric_calls as a soft cap and can
            # overshoot). Never pin a live run at 100% — rich freezes the elapsed clock once
            # completed == total, which reads as "stuck". Grow the total instead.
            total = budget if done < budget else done + 1
            self._progress.update(self._task_id, completed=done, total=total, score=label)
        elif not self._tty:
            # Non-TTY: emit a sparse heartbeat so long runs still show life without flooding logs.
            if done == 1 or done % 10 == 0 or done >= budget:
                progress = (
                    f"{done}/{budget}" if done <= budget else f"{done} (budget target {budget})"
                )
                self._console.print(f"  GEPA metric call {progress} ({label})")

    def optimize_done(self, held_out_accuracy: float, frontier_size: int, rollouts: int) -> None:
        if self._progress is not None:
            if self._task_id is not None and rollouts > 0:
                # 100% exactly when GEPA is actually done: the endpoint can't be predicted (soft
                # cap + per-iteration costs decided at runtime), so the bar snaps to the ACTUAL
                # final call count on the completion event instead of guessing during the run.
                self._progress.update(self._task_id, completed=rollouts, total=rollouts)
            if self._live is not None:
                self._live.refresh()  # final frame: bar snapped to 100%
                self._live.stop()
                self._live = None
            self._progress = None
            self._task_id = None
        self._stage(
            f"GEPA done: val {held_out_accuracy:.3f} (selection sample), "
            f"{frontier_size} frontier candidates, {rollouts} rollouts used"
        )

    def _stage(self, message: str) -> None:
        self._console.print(f"{_CHECK} {message}")

    def close(self) -> None:
        """Stop the live display if it is still running (e.g. the build raised mid-GEPA)."""
        if self._live is not None:
            self._live.stop()
            self._live = None
            self._progress = None
            self._task_id = None
            return
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task_id = None

    def __enter__(self) -> RichBuildReporter:
        return self

    def __exit__(self, *exc: object) -> None:
        # Always tear down the live Progress so an exception during the build doesn't leave a
        # spinning bar that corrupts the terminal.
        self.close()


def build_summary_panel(info: ModelInfo, root: str) -> Panel:
    """A tidy panel summarizing a freshly built world model (shown after `wmh build`)."""
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="bold")
    table.add_column()
    table.add_row("name", info.name)
    table.add_row("artifact", root)
    table.add_row("serve provider", f"{info.serve_provider} ({info.serve_model})")
    if info.held_out_accuracy is not None:
        table.add_row("held-out accuracy", f"{info.held_out_accuracy:.3f}")
    if info.rollouts_used is not None:
        table.add_row("rollouts used", str(info.rollouts_used))
    if info.frontier_size is not None:
        table.add_row("frontier candidates", str(info.frontier_size))
    return Panel(
        table,
        title=f"[bold green]world model ready: {info.name}[/bold green]",
        subtitle="serve it with `wmh serve` or step into it with `wmh play`",
        border_style="green",
    )


def models_table(infos: list[ModelInfo]) -> Table:
    """A table of every built world model (for `wmh list`)."""
    table = Table(title="world models")
    table.add_column("name", style="bold")
    table.add_column("serve provider")
    table.add_column("held-out", justify="right")
    table.add_column("rollouts", justify="right")
    table.add_column("frontier", justify="right")
    for info in infos:
        table.add_row(
            info.name,
            f"{info.serve_provider} ({info.serve_model})",
            "-" if info.held_out_accuracy is None else f"{info.held_out_accuracy:.3f}",
            "-" if info.rollouts_used is None else str(info.rollouts_used),
            "-" if info.frontier_size is None else str(info.frontier_size),
        )
    return table


# --- interactive play REPL -----------------------------------------------------------------------

_PLAY_HELP = (
    "[bold]You are the agent.[/bold] Type an action and the world model answers:\n"
    '  [cyan]get_user {"id": "u1"}[/cyan]   a tool call with JSON arguments\n'
    "  [cyan]list_flights[/cyan]            a tool call with no arguments\n"
    "  [cyan]say I am stuck[/cyan]          a free-text message to the environment\n"
    "Commands: [cyan]:state[/cyan] show session state  ·  [cyan]:help[/cyan]  ·  "
    "[cyan]:quit[/cyan] (or Ctrl-D) to exit"
)


_AGENT_PROMPT = "[bold]agent>[/bold] "


def run_play_repl(
    console: Console,
    world_model: WorldModel,
    model_name: str,
    task: str | None,
    reader: PromptReader | None = None,
    suggestions: list[str] | None = None,
    seed_state: EnvState | None = None,
) -> None:
    """Run the human-in-the-loop demo against `world_model`.

    `reader` is an optional `PromptReader` (`(prompt_text) -> line`) used to source input — injected
    in tests, defaults to the console's prompt. The loop ends on `:quit`, EOF, or KeyboardInterrupt.
    """
    ask = reader if reader is not None else console.input
    session = world_model.new_session(task=task, seed_state=seed_state)
    body = _PLAY_HELP
    if suggestions:
        sampled = "\n".join(f"  [cyan]{escape(line)}[/cyan]" for line in suggestions)
        body = (
            "You are the agent. Type an action and the world model answers.\n"
            "Real actions from this model's traces to try:\n"
            f"{sampled}\n"
            "Commands: :state show session state  \u00b7  :help  \u00b7  :quit (or Ctrl-D) to exit"
        )
    console.print(
        Panel(
            body,
            title=f"[bold]playing[/bold] {model_name}",
            subtitle=f"task: {task}" if task else "no task set",
            border_style="cyan",
        )
    )

    while True:
        try:
            line = ask(_AGENT_PROMPT)
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            return
        line = line.strip()
        if not line:
            continue
        if line in {":quit", ":q", ":exit"}:
            console.print("[dim]bye[/dim]")
            return
        if line in {":help", ":h"}:
            console.print(_PLAY_HELP)
            continue
        if line == ":state":
            _render_state(console, world_model.get_session(session.id))
            continue
        _handle_action(console, world_model, session.id, line)


def _handle_action(console: Console, world_model: WorldModel, session_id: str, line: str) -> None:
    """Parse + step one typed action, rendering the observation (or a friendly error).

    A failed step (e.g. a provider/network error) is reported and swallowed so the REPL keeps the
    session alive instead of crashing the whole interactive run.
    """
    try:
        action = parse_action(line)
    except ValueError as exc:
        console.print(f"[red]parse error[/red]: {exc}")
        return
    try:
        with console.status("[dim]world model thinking…[/dim]", spinner="dots"):
            turn = play_turn(world_model, session_id, action)
    except Exception as exc:  # noqa: BLE001 - keep the REPL alive; surface the failure to the user
        console.print(f"[red]step failed[/red]: {exc}")
        return
    _render_turn(console, turn)


def _render_turn(console: Console, turn: PlayTurn) -> None:
    console.print(f"[bold cyan]→ you[/bold cyan]: {_action_text(turn.action)}")
    style = "red" if turn.observation.is_error else "green"
    label = "error" if turn.observation.is_error else "observation"
    console.print(
        Panel(
            turn.observation.content or "[dim](empty)[/dim]",
            title=f"[bold]{label}[/bold]",
            border_style=style,
        )
    )


def _render_state(console: Console, session: Session) -> None:
    scratchpad = session.state.scratchpad or "[dim](empty)[/dim]"
    body = f"[bold]task[/bold]: {session.task or '(none)'}\n"
    body += f"[bold]turns[/bold]: {len(session.history)}\n\n"
    body += f"[bold]scratchpad[/bold]:\n{scratchpad}"
    console.print(Panel(body, title="session state", border_style="blue"))


def _action_text(action: Action) -> str:
    if action.kind == ActionKind.TOOL_CALL:
        return f"{action.name}({action.arguments})"
    return f'message: "{action.content}"'
