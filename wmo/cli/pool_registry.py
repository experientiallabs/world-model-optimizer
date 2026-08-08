"""The interactive model registry behind `wmo providers set`: filling `.wmo/pool.toml`.

Two questions used to be unconnected. `wmo providers set` answered "which provider runs my local
worker agent" and wrote `[models.worker]` into settings; the routing optimizer read a completely
separate `pool.toml` that nothing wrote for an ordinary provider model, so a user who wanted five
OpenRouter models and five Azure deployments had to hand-author TOML and get `kind`, `model`,
prices, `deployment`, `api_version`, and `api_key_env` right unaided. This module makes the one
command answer both: after a provider is picked and verified, it offers to register models from
it (or from any other backend) as routing candidates.

Design rules this module holds to:

- **Nothing hardcoded.** Model ids come from `wmo.common.providers.catalog`, which reads
  OpenRouter's published catalog and WMO's canonical registry. A typed id works for every kind.
- **One writer.** Entries go through `wmo.common.providers.pool.upsert_pool_entry`, so registration
  inherits its cross-process lock, its whole-roster revalidation, and its atomic write. This
  module never opens the roster for writing.
- **Ask only what the kind needs.** Azure is asked for a deployment and an api-version; an
  OpenRouter entry is asked for neither, and for no price either, because OpenRouter self-prices
  from its published catalog. A price is prompted exactly when the entry would otherwise report
  $0.
- **Idempotent.** A model already in the roster is shown as such and re-registers under its
  existing handle, so a second pass adds the new provider's models beside the first pass's
  instead of duplicating or clobbering them.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

import typer
from pydantic import BaseModel, ConfigDict, ValidationError
from rich.console import Console
from rich.markup import escape
from rich.table import Table

import wmo.cli.pool_registry as _self
from wmo.cli.model_roles import DEFAULT_AZURE_API_VERSION
from wmo.cli.ui import creds_note, ensure_credentials, has_credentials, select_option
from wmo.common.config import PROVIDER_ENV_VARS
from wmo.common.providers.base import ProviderKind, VerifyResult
from wmo.common.providers.pool import Tier

if TYPE_CHECKING:
    from wmo.cli.ui import PromptReader
    from wmo.common.providers.catalog import CatalogModel, ProviderCatalog
    from wmo.common.providers.pool import PoolEntry

logger = logging.getLogger(__name__)

# The live provider ping, as a seam: one cheap completion against a built candidate. Injected
# so tests never reach a backend, mirroring `run_build_wizard`'s `verify` parameter.
ProviderCheck = Callable[["PoolEntry"], VerifyResult]

POOL_FILENAME = "pool.toml"
"""The roster's name inside a project root, so `--root <dir>` keeps settings and pool together."""

TIERS: tuple[Tier, ...] = ("frontier", "open")
"""The `PoolEntry.tier` vocabulary, offered as a picker so the value is never mistyped."""

_MAX_LISTED = 20
"""Rows one search shows. OpenRouter publishes 338 models, so an unfiltered dump is not a list."""

_CHECK = "[green]✓[/green]"


class EntryOptions(BaseModel):
    """Backend knobs a registration applies to every entry it writes in one pass.

    Seeded from `wmo providers set`'s own flags for the provider being set, then refined by the
    prompts. Everything here is shared across the models chosen in one pass; anything that
    genuinely differs per model (an Azure deployment name, a price) is asked per model instead.
    """

    model_config = ConfigDict(extra="forbid")

    endpoint: str | None = None
    region: str | None = None
    deployment: str | None = None
    api_version: str | None = None
    api_key_env: str | None = None
    tier: Tier = "frontier"
    input_per_mtok: float | None = None
    output_per_mtok: float | None = None


def pool_path_for(root: str | Path, override: str | None = None) -> Path:
    """Where this project's roster lives: `--pool` when given, else `<root>/pool.toml`.

    Derived from the same root as `settings.toml` so one `--root` moves the whole project, and
    equal to `wmo.common.providers.pool.DEFAULT_POOL_PATH` at the default root, which is what the
    routing commands read.
    """
    return Path(override) if override else Path(root) / POOL_FILENAME


def read_pool_entries(path: Path) -> list[PoolEntry]:
    """The roster's entries, or an empty list when there is no readable roster yet.

    Tolerant on purpose: this feeds display and duplicate detection, and a roster that cannot be
    parsed is `upsert_pool_entry`'s error to raise, with its own actionable message. Swallowing
    it here would only pre-empt that with a worse one.
    """
    from wmo.common.providers.pool import load_pool

    try:
        return list(load_pool(path).models)
    except (FileNotFoundError, ValueError) as exc:
        logger.debug("no readable model pool at %s: %s", path, exc)
        return []


def print_pool(console: Console, path: Path, entries: list[PoolEntry]) -> None:
    """Show the roster the router chooses from, so a second pass adds to a visible starting set."""
    from wmo.common.providers.pool import is_local_endpoint

    if not entries:
        console.print(f"[dim]routing pool[/dim] {path} [dim]is empty[/dim]")
        return
    table = Table(title=f"routing pool ({path})", title_justify="left", header_style="bold")
    table.add_column("handle")
    table.add_column("kind")
    table.add_column("model")
    table.add_column("$/Mtok in/out", justify="right")
    table.add_column("tier")
    for entry in entries:
        price = entry.price()
        # "(local)" is derived from the endpoint URL, never stored: the wire kind stays plain
        # openai, and copy is the only place the distinction exists.
        kind = entry.kind.value
        if is_local_endpoint(entry.endpoint):
            kind = f"{kind} (local)"
        elif entry.endpoint:
            kind = f"{kind} (custom endpoint)"
        table.add_row(
            entry.name,
            kind,
            entry.deployment or entry.model,
            f"{price.input_per_mtok:g} / {price.output_per_mtok:g}",
            entry.tier if entry.enabled else f"{entry.tier} (off)",
            style=None if entry.enabled else "dim",
        )
    console.print(table)


def run_pool_registry(
    console: Console,
    ask: PromptReader,
    ask_secret: PromptReader,
    *,
    pool_path: Path,
    default_kind: ProviderKind,
    options: EntryOptions,
    verify: ProviderCheck | None = None,
) -> int:
    """Offer to register routing candidates, one provider at a time, until the user is done.

    The interactive half of the registry. Shows the roster first (a re-run has to start from what
    is already there), then loops: pick a backend, pick models from its catalog, answer only what
    that backend needs, write. `options` seeds the FIRST pass, because it carries the flags that
    described `default_kind`'s backend; a different backend picked later starts from defaults so
    an Azure api-version can never leak onto an OpenRouter entry.

    Args:
        console: Where the roster and progress are rendered.
        ask: Reads one prompt line.
        ask_secret: Reads one prompt line without echo (credential entry).
        pool_path: The roster TOML to write.
        default_kind: The provider just configured; the suggested backend for the first pass.
        options: Backend knobs from the command's own flags, applied to the first pass.
        verify: The live provider ping, one per pass; None uses the real one.

    Returns:
        How many entries were written across every pass.
    """
    print_pool(console, pool_path, read_pool_entries(pool_path))
    if not _ask_yes_no(console, ask, "Register models the router can choose from?", default=True):
        console.print(
            f"[dim]skipped; run `wmo providers set` again any time to add candidates to "
            f"{pool_path}[/dim]"
        )
        return 0
    written = 0
    kind = _pick_kind(console, ask, default=default_kind)
    while True:
        if kind is default_kind:
            pass_options = options
        else:
            # A backend the command's flags never described: start from defaults, and make sure
            # its own credentials are present before offering to register anything against it.
            pass_options = EntryOptions()
            ensure_credentials(console, ask_secret, kind.value)
        written += register_from_provider(
            console, ask, pool_path=pool_path, kind=kind, options=pass_options, verify=verify
        )
        if not _ask_yes_no(console, ask, "Register models from another provider?", default=False):
            break
        kind = _pick_kind(console, ask, default=kind)
    print_pool(console, pool_path, read_pool_entries(pool_path))
    return written


def register_from_provider(
    console: Console,
    ask: PromptReader,
    *,
    pool_path: Path,
    kind: ProviderKind,
    options: EntryOptions,
    verify: ProviderCheck | None = None,
) -> int:
    """One pass: choose `kind`'s models, answer its requirements, write them. Returns the count.

    `verify` is the live ping (None uses the real one). It runs once per CANDIDATE rather than
    once per pass, because a pass is not one route: on Azure every candidate names a different
    operator-chosen deployment, so a first deployment that answers proves nothing about the
    second. `wmo providers verify` bills a call per model for the same reason.
    """
    from wmo.common.providers.catalog import list_provider_models

    if kind is ProviderKind.OPENAI:
        # The one kind that can point at a self-hosted server (Ollama, vLLM, llama.cpp), so the
        # endpoint question comes BEFORE the catalog: a set URL swaps the built-in registry for
        # what that server actually serves.
        options = _ask_endpoint(console, ask, options)
    if kind is ProviderKind.OPENAI and options.endpoint:
        endpoint_cat_fn = cast(Any, _self.endpoint_catalog)
        catalog = endpoint_cat_fn(options.endpoint)
    else:
        catalog = list_provider_models(kind)
    _describe_catalog(console, catalog)
    chosen = choose_models(
        console, ask, catalog, read_pool_entries(pool_path), endpoint=options.endpoint
    )
    if not chosen:
        console.print("[dim]nothing selected[/dim]")
        return 0
    shared = _ask_shared_options(console, ask, kind, options)
    written = 0
    for model in chosen:
        if _register_one(
            console, ask, pool_path=pool_path, kind=kind, model=model, options=shared, verify=verify
        ):
            written += 1
    return written


def _ask_endpoint(console: Console, ask: PromptReader, options: EntryOptions) -> EntryOptions:
    """Ask which server the OpenAI pass talks to: OpenAI's own API when nothing is set.

    Answered once per pass, because every model registered against a server shares its URL. The
    stored value is exactly what was typed (a platform serving inside a container translates
    known-local hostnames at ITS boundary; the pool file never carries the translated form).

    The blank answer keeps whatever is in effect, so its meaning is stated per case: with no
    seeded endpoint blank means OpenAI's own API, and with one (`--endpoint`) blank keeps it,
    with `-` as the explicit way back to the hosted API. Without that escape a session started
    with `--endpoint` could never register a hosted candidate at all.
    """
    from wmo.common.providers.pool import is_local_endpoint

    if options.endpoint:
        hint = f"blank = keep {escape(options.endpoint)}; '-' = OpenAI's own API"
    else:
        hint = "blank = OpenAI's own API; e.g. http://localhost:11434/v1 for Ollama"
    raw = _read(ask, f"[bold]OpenAI-compatible endpoint URL[/bold] [dim]({hint})[/dim]: ")
    if raw == "-":
        return options.model_copy(update={"endpoint": None})
    endpoint = raw or options.endpoint
    if not endpoint:
        return options.model_copy(update={"endpoint": None})
    if is_local_endpoint(endpoint):
        console.print(f"  [dim]{escape(endpoint)} reads as a locally hosted server[/dim]")
    return options.model_copy(update={"endpoint": endpoint})


def _verify_candidate(
    console: Console,
    ask: PromptReader,
    *,
    entry: PoolEntry,
    verify: ProviderCheck | None,
) -> bool:
    """Ping one candidate, and say whether to register it.

    `wmo providers set` already bills a ping to prove the WORKER provider works, and the roster
    is the artifact where an unproved candidate is most expensive: `wmo optimize route sweep`
    pays for every candidate ahead of the one that 401s before it finds out. So each candidate
    the registry is about to write pays the same cheap ping, over its own route (its endpoint,
    deployment, region) and its own `api_key_env` account.

    A failure does not discard the pick silently: it is reported and the user decides, because a
    roster is legitimately built on a machine that does not hold the key.
    """
    console.print(f"verifying {entry.kind.value} ({escape(entry.deployment or entry.model)})...")
    result = (verify or verify_pool_entry)(entry)
    label = f"{entry.kind.value} ({escape(entry.deployment or entry.model)})"
    if result.ok:
        console.print(f"  {_CHECK} {label} reachable")
        return True
    console.print(f"  [red]x {label} failed[/red]: {escape(result.detail or 'unknown error')}")
    return _ask_yes_no(console, ask, "Register it anyway?", default=False)


def verify_pool_entry(entry: PoolEntry) -> VerifyResult:
    """One cheap live completion against a candidate exactly as the router would call it.

    Goes through `pool_provider` rather than `verify_all` so the entry's own `api_key_env`
    account is the one proved: a multi-account roster that verified the default credential
    would have proved nothing about the key the candidate will actually send.

    Never raises: a backend that refuses to be built at all (an unset `api_key_env`, Bedrock
    handed a key) comes back as a failure with its own message, like any other.
    """
    from wmo.common.providers.pool import pool_provider

    try:
        return pool_provider(entry).verify()
    except ValueError as exc:
        return VerifyResult(ok=False, kind=entry.kind, model=entry.model, detail=str(exc))


def register_model_ids(
    console: Console,
    *,
    pool_path: Path,
    kind: ProviderKind,
    model_ids: list[str],
    options: EntryOptions,
    verify: ProviderCheck | None = None,
) -> int:
    """Register `model_ids` with no prompts, for `--pool-model` and scripts. Returns the count.

    Every id is resolved against `kind`'s catalog first, so a scripted registration picks up the
    same canonical model type and published price an interactive one would; an id the catalog
    does not list is registered verbatim, exactly as a typed id is.

    Every candidate is then live-pinged over its own route, exactly as the interactive flow does.
    A `--pool-model` can differ from the verified worker in model, deployment, endpoint, and
    credential, so the worker's ping proves nothing about it, and a script is precisely where
    nobody is watching for the 401 that would otherwise surface inside a paid `route sweep`. The
    whole batch is built and proved BEFORE anything is written, so a bad third id cannot leave
    half a roster behind.

    Azure is the one backend this cannot do in bulk. On the wire its deployment name IS the
    model, deployment names are chosen by whoever created them, and nothing in a catalog can
    guess one: derived from the model id they would silently address deployments that do not
    exist, and one `--deployment` shared across ids would point several candidates at one route.
    So a scripted Azure registration is one explicitly named deployment per invocation.

    Raises:
        typer.BadParameter: An entry cannot be built (a missing price, an Azure deployment), it
            cannot be called, the roster refuses it, or Azure was given no deployment or several
            models for one. Non-interactively there is nobody to ask, so this fails loudly rather
            than writing a candidate that does not work.
    """
    from wmo.common.providers.catalog import CatalogModel, list_provider_models
    from wmo.common.providers.pool import is_local_endpoint, static_requirements

    _check_azure_deployment(kind, model_ids, options)
    if _self_hosted(kind, options) and (options.input_per_mtok is None) != (
        options.output_per_mtok is None
    ):
        # Both or neither, exactly like the CLI flag guard (`wmo providers set` enforces it for
        # flags; this covers direct library callers): silently zero-filling the missing half
        # would discard a declared rate.
        raise typer.BadParameter(
            "set both input and output prices, or neither; a half-supplied pair would silently "
            "zero the declared half"
        )
    if _self_hosted(kind, options) and options.input_per_mtok is None:
        if not is_local_endpoint(options.endpoint):
            # A REMOTE OpenAI-compatible server (Together, Groq, a LiteLLM proxy) bills real
            # money; only a locally hosted one defaults to free. Demanding the price here keeps
            # the loud behavior main had for unpriced candidates.
            raise typer.BadParameter(
                f"'{options.endpoint}' is not a locally hosted server, so its price cannot be "
                "assumed to be $0; pass --input-per-mtok and --output-per-mtok (USD per 1M "
                "tokens) for the candidates it serves"
            )
        # The scripted twin of the interactive default: a LOCAL candidate is priced by the
        # operator or it is explicitly free, never priced off a shadowed built-in id.
        console.print(
            f"  [dim]locally hosted endpoint {escape(options.endpoint or '')}: pricing at "
            "$0/Mtok (override with --input-per-mtok/--output-per-mtok)[/dim]"
        )
        options = options.model_copy(update={"input_per_mtok": 0.0, "output_per_mtok": 0.0})
    if _self_hosted(kind, options):
        endpoint_cat_fn = cast(Any, _self.endpoint_catalog)
        catalog = endpoint_cat_fn(options.endpoint or "")
    else:
        catalog = list_provider_models(kind)
    check = verify or verify_pool_entry
    existing = read_pool_entries(pool_path)
    taken = {entry.name for entry in existing}
    planned: list[PoolEntry] = []
    for model_id in model_ids:
        model = catalog.find(model_id) or CatalogModel(id=model_id)
        name = _existing_handle(existing, kind, model, options) or _unique_handle(
            _handle_base(kind, model, options), taken
        )
        taken.add(name)
        try:
            entry = build_pool_entry(name=name, kind=kind, model=model, options=options)
        except (ValidationError, ValueError) as exc:
            raise typer.BadParameter(
                f"cannot register '{model.id}' from {kind.value}: {exc}"
            ) from exc
        problems = static_requirements(entry)
        if problems:
            raise typer.BadParameter(
                f"cannot register '{model.id}' from {kind.value}: {'; '.join(problems)}"
            )
        result = check(entry)
        if not result.ok:
            raise typer.BadParameter(
                f"'{model.id}' from {kind.value} is not callable "
                f"({result.detail or 'unknown error'}); fix the credential, endpoint, or id, or "
                "leave --pool-model off and register it from the interactive picker, which lets "
                "you keep an unreachable candidate deliberately"
            )
        planned.append(entry)
    return sum(1 for entry in planned if _write(console, entry, pool_path))


def _check_azure_deployment(
    kind: ProviderKind, model_ids: list[str], options: EntryOptions
) -> None:
    """Refuse a scripted Azure registration that cannot name the deployment it will call.

    Raises:
        typer.BadParameter: No `--deployment`, or one `--deployment` for several models.
    """
    if kind is not ProviderKind.AZURE_OPENAI or not model_ids:
        return
    if options.deployment is None:
        raise typer.BadParameter(
            "azure needs --deployment alongside --pool-model: an Azure deployment name is chosen "
            "by whoever created it and is what every request names as its model, so it cannot be "
            "derived from a model id (deriving it would register a route that does not exist)"
        )
    if len(model_ids) > 1:
        raise typer.BadParameter(
            "one --deployment cannot describe several --pool-model ids on azure: the deployment "
            "name IS the model on the wire, so several ids sharing it would all call the same "
            "deployment; run this once per deployment"
        )


def choose_models(
    console: Console,
    ask: PromptReader,
    catalog: ProviderCatalog,
    existing: list[PoolEntry],
    endpoint: str | None = None,
) -> list[CatalogModel]:
    """Search-and-pick over one catalog: type a term to filter, numbers to toggle, blank to finish.

    A picker over OpenRouter's 338 models cannot be a scrollable list, so search comes first and
    only the matches are numbered. One loop serves every backend: with no catalog to search (a
    Tinker base model) a typed line IS the id, and with a catalog a term that matches nothing
    offers to take the line as a literal id, so a model published after this release is never
    unreachable.

    Args:
        console: Where matches and the running selection are rendered.
        ask: Reads one prompt line.
        catalog: The backend's offerable models.
        existing: The roster as it stands, used to annotate what is already registered.
        endpoint: The server this pass registers against (None = the backend's own API), part
            of candidate identity: the same model id on two servers is two candidates.

    Returns:
        The chosen models, in selection order.
    """
    from wmo.common.providers.catalog import CatalogModel

    by_target = {
        _target(entry.kind, entry.model, entry.deployment, entry.endpoint): entry
        for entry in existing
    }
    selected: dict[str, CatalogModel] = {}
    shown: list[CatalogModel] = catalog.models[:_MAX_LISTED]
    if catalog.models:
        _list_matches(console, catalog, shown, len(catalog.models), by_target, selected, endpoint)
    while True:
        raw = _read(ask, "[bold]models[/bold] [dim](search / numbers / blank when done)[/dim]> ")
        if not raw:
            return list(selected.values())
        picks = _parse_picks(raw)
        if picks is not None:
            # All-or-nothing: "1 99" over twelve rows is a typo, and toggling the valid half
            # would register a model nobody chose while looking like it worked.
            if all(1 <= pick <= len(shown) for pick in picks):
                for index in picks:
                    _toggle(console, selected, shown[index - 1])
            else:
                console.print(f"  [red]pick 1-{len(shown)} from the rows above[/red]")
            continue
        listed = catalog.find(raw)
        if listed is not None:
            _toggle(console, selected, listed)
            continue
        if not catalog.models:
            _toggle(console, selected, CatalogModel(id=raw))
            continue
        matches = catalog.search(raw)
        if matches:
            shown = matches[:_MAX_LISTED]
            _list_matches(console, catalog, shown, len(matches), by_target, selected, endpoint)
            continue
        console.print(f"  [yellow]no {catalog.kind.value} model matches[/yellow] {escape(raw)}")
        if _ask_yes_no(console, ask, f"Register '{raw}' as a literal model id?", default=False):
            _toggle(console, selected, CatalogModel(id=raw))


def build_pool_entry(
    *,
    name: str,
    kind: ProviderKind,
    model: CatalogModel,
    options: EntryOptions,
) -> PoolEntry:
    """One resolved candidate as a `PoolEntry`, with only the fields its kind actually uses.

    Backend knobs are applied per kind rather than passed through wholesale: an Azure deployment
    and api-version belong on an Azure entry and nowhere else, a Bedrock region likewise, and
    Bedrock refuses an `api_key_env` outright (it authenticates with AWS credentials), so that
    field is dropped there instead of being written and rejected at load.

    Raises:
        pydantic.ValidationError: The entry is not a valid candidate, most often a model with no
            built-in price and no declared one.
    """
    from wmo.common.providers.pool import PoolEntry

    azure = kind is ProviderKind.AZURE_OPENAI
    bedrock = kind is ProviderKind.BEDROCK
    return PoolEntry(
        name=name,
        kind=kind,
        model=model.id,
        model_type=model.model_type,
        endpoint=options.endpoint,
        deployment=(options.deployment or model.id) if azure else None,
        api_version=(options.api_version or DEFAULT_AZURE_API_VERSION) if azure else None,
        region=options.region if bedrock else None,
        api_key_env=None if bedrock else options.api_key_env,
        tier=options.tier,
        input_per_mtok=options.input_per_mtok,
        output_per_mtok=options.output_per_mtok,
    )


def needs_price(kind: ProviderKind, model: CatalogModel) -> bool:
    """Whether this candidate has to be GIVEN a price, i.e. would otherwise be costed at $0.

    Deliberately the same two questions `PoolEntry._validate_price` asks, in the same order, so
    the prompt appears exactly when the entry would be rejected without an answer: is the model
    in the built-in pricing table, and failing that, does OpenRouter publish a price for it? The
    catalog row's own `price` is display data and is not consulted, because a row priced by its
    model TYPE would still leave the entry, which is keyed on the runtime id, unpriced.

    The OpenRouter lookup reads the same cached table the picker listed from, so it costs no
    second fetch.
    """
    from wmo.common.observability.pricing import price_for
    from wmo.common.providers.openrouter_pricing import resolve_price as resolve_openrouter_price

    if price_for(model.id) is not None:
        return False
    if kind is ProviderKind.OPENROUTER:
        return resolve_openrouter_price(model.id).price is None
    return True


def _register_one(
    console: Console,
    ask: PromptReader,
    *,
    pool_path: Path,
    kind: ProviderKind,
    model: CatalogModel,
    options: EntryOptions,
    verify: ProviderCheck | None,
) -> bool:
    """Ask what this one model still needs, prove it answers, name it, write it.

    In that order: the connection details come first because the ping needs them, the ping comes
    before the handle prompt because there is no point naming a candidate that cannot be called,
    and the handle comes last because it is the only answer the roster keys on.

    Returns:
        True when the roster gained or changed this entry, False when it was skipped.
    """
    from wmo.common.providers.pool import static_requirements

    per_model = _ask_per_model_options(console, ask, kind, model, options)
    try:
        probe = build_pool_entry(name="probe", kind=kind, model=model, options=per_model)
    except (ValidationError, ValueError) as exc:
        console.print(f"  [red]skipped {escape(model.id)}[/red]: {escape(str(exc))}")
        return False
    problems = static_requirements(probe)
    if problems:
        console.print(f"  [red]skipped {escape(model.id)}[/red]: {escape('; '.join(problems))}")
        return False
    if not _verify_candidate(console, ask, entry=probe, verify=verify):
        return False
    entries = read_pool_entries(pool_path)
    taken = {entry.name: entry for entry in entries}
    known = _existing_handle(entries, kind, model, per_model)
    default_name = known or _unique_handle(_handle_base(kind, model, per_model), set(taken))
    while True:
        name = _ask_text(console, ask, f"Handle for {model.id}", default_name)
        clash = taken.get(name)
        if clash is None or name == known:
            break
        console.print(
            f"  [yellow]{escape(name)} already names[/yellow] {clash.kind.value} "
            f"{escape(clash.deployment or clash.model)}[yellow]; replacing it rewrites "
            f"{pool_path} and drops its comments[/yellow]"
        )
        if _ask_yes_no(console, ask, f"Replace '{name}'?", default=False):
            break
    return _write(console, probe.model_copy(update={"name": name}), pool_path)


def _write(console: Console, entry: PoolEntry, pool_path: Path) -> bool:
    """Hand one entry to the roster's only writer and report what it did.

    An entry the roster already holds VERBATIM is not written at all. That is what makes a re-run
    free: `upsert_pool_entry` treats a same-name entry as a replacement, which re-renders the
    whole file, so writing an identical entry would reorder the roster and drop its comments to
    change nothing.

    Returns:
        True when the roster changed, False when it was already right or the write was refused.
    """
    from wmo.common.core.locks import FileLockTimeout
    from wmo.common.providers.pool import upsert_pool_entry

    existing = {row.name: row for row in read_pool_entries(pool_path)}.get(entry.name)
    if existing is not None and not existing.enabled:
        # `enabled = false` is an explicit operator edit; a re-run of a registration script
        # must not silently put the candidate back into selection. Re-enabling is the same
        # one-line file edit that disabled it.
        entry = entry.model_copy(update={"enabled": False})
        console.print(
            f"  [dim]{escape(entry.name)} is disabled in the roster (enabled = false); "
            "keeping it disabled[/dim]"
        )
    price = entry.price()
    summary = (
        f"[bold]{escape(entry.name)}[/bold] ({entry.kind.value} "
        f"{escape(entry.deployment or entry.model)}, "
        f"${price.input_per_mtok:g}/${price.output_per_mtok:g} per Mtok)"
    )
    if entry in read_pool_entries(pool_path):
        console.print(f"  {_CHECK} {summary} [dim]already registered, unchanged[/dim]")
        return False
    try:
        written = upsert_pool_entry(entry, pool_path)
    except FileLockTimeout as exc:
        # Another writer is in the way; nothing about the entry is wrong, so say to retry rather
        # than sending the user back through the prompts.
        console.print(f"  [red]pool busy[/red] {escape(str(exc))}")
        return False
    except ValueError as exc:
        console.print(f"  [red]skipped {escape(entry.name)}[/red]: {escape(str(exc))}")
        return False
    # `rewritten`, not `replaced`: normalizing a legacy inline-form roster drops the comments on
    # an ADD too, and reporting that as a plain "added" would delete them silently.
    note = (
        " [dim](the roster was rewritten, so its comments are gone)[/dim]"
        if written.rewritten
        else ""
    )
    console.print(f"  {_CHECK} {'replaced' if written.replaced else 'added'} {summary}{note}")
    return True


def _describe_catalog(console: Console, catalog: ProviderCatalog) -> None:
    """One line saying where this backend's model list came from, before the picker opens."""
    from wmo.common.providers.catalog import CatalogSource

    match catalog.source:
        case CatalogSource.PUBLISHED:
            origin = "published catalog"
        case CatalogSource.BUILT_IN:
            origin = "built-in registry"
        case CatalogSource.NONE:
            origin = "no catalog"
    detail = f": {catalog.detail}" if catalog.detail else ""
    console.print(f"[bold]{catalog.kind.value}[/bold] [dim]{origin}{escape(detail)}[/dim]")


def _list_matches(
    console: Console,
    catalog: ProviderCatalog,
    shown: list[CatalogModel],
    total: int,
    by_target: dict[tuple[str, str, str, str], PoolEntry],
    selected: dict[str, CatalogModel],
    endpoint: str | None = None,
) -> None:
    """Number the current matches, marking what is already registered and already picked."""
    for index, model in enumerate(shown, start=1):
        registered = by_target.get(
            _target(catalog.kind, model.id, None, endpoint)
        ) or by_target.get(_target(catalog.kind, model.id, model.id, endpoint))
        notes = []
        if registered is not None:
            notes.append(f"in pool as {registered.name}")
        if model.id in selected:
            notes.append("selected")
        suffix = f"  [dim]({', '.join(notes)})[/dim]" if notes else ""
        console.print(f"  [cyan]{index}[/cyan]. {escape(model.label())}{suffix}")
    if total > len(shown):
        console.print(f"  [dim]... {total - len(shown)} more; refine the search[/dim]")


def _toggle(console: Console, selected: dict[str, CatalogModel], model: CatalogModel) -> None:
    """Add or remove one model from the running selection, echoing which happened."""
    if selected.pop(model.id, None) is not None:
        console.print(f"  [dim]- {escape(model.id)}[/dim]")
        return
    selected[model.id] = model
    console.print(f"  [green]+[/green] {escape(model.id)}")


def _parse_picks(raw: str) -> list[int] | None:
    """`raw` as row numbers, or None when it is not a list of them (so the caller searches).

    Whether they are IN range is the caller's call, because "1 99" is a mis-typed selection worth
    saying so about, while "4o" is a search term. ASCII decimals only: "4o".isdecimal() is False,
    but so is a superscript that `int()` would reject after `isdigit()` accepted it.
    """
    tokens = raw.replace(",", " ").split()
    if not tokens or not all(token.isascii() and token.isdecimal() for token in tokens):
        return None
    return [int(token) for token in tokens]


def _pick_kind(console: Console, ask: PromptReader, *, default: ProviderKind) -> ProviderKind:
    """Choose which backend to register from, annotated with whose credentials are present."""
    kinds = [kind.value for kind in ProviderKind]
    notes = {kind: creds_note(kind) for kind in kinds if has_credentials(kind)}
    chosen = select_option(
        console, "Register models from", kinds, notes=notes, default=default.value, reader=ask
    )
    return ProviderKind(chosen)


def _ask_shared_options(
    console: Console,
    ask: PromptReader,
    kind: ProviderKind,
    options: EntryOptions,
) -> EntryOptions:
    """Ask the questions that apply to every model of this pass: tier, account, region."""
    tier = select_option(console, "Tier", list(TIERS), default=options.tier, reader=ask)
    updates: dict[str, str | None] = {"tier": tier}
    if kind is ProviderKind.BEDROCK:
        # Bedrock authenticates with AWS credentials, so it takes a region and refuses a key var.
        updates["region"] = (
            _ask_text(console, ask, "AWS region (blank = AWS_REGION)", options.region or "") or None
        )
    else:
        default_env = _default_key_env(kind)
        updates["api_key_env"] = (
            _ask_text(
                console,
                ask,
                f"Env var holding the API key (blank = {default_env})",
                options.api_key_env or "",
            )
            or None
        )
    return options.model_copy(update=updates)


def _ask_per_model_options(
    console: Console,
    ask: PromptReader,
    kind: ProviderKind,
    model: CatalogModel,
    options: EntryOptions,
) -> EntryOptions:
    """Ask what THIS model needs and nothing else: an Azure deployment, or a missing price."""
    from wmo.common.providers.pool import is_local_endpoint

    updates: dict[str, str | float | None] = {}
    if kind is ProviderKind.AZURE_OPENAI:
        # On the wire Azure's model IS the deployment name, and every deployment is named by the
        # operator, so neither field can be derived from the catalog.
        updates["deployment"] = _ask_text(
            console, ask, f"Azure deployment serving {model.id}", options.deployment or model.id
        )
        updates["api_version"] = _ask_text(
            console, ask, "Azure api-version", options.api_version or DEFAULT_AZURE_API_VERSION
        )
    # Either side missing, not just the input side: `PoolEntry` refuses a half pair outright, so
    # suppressing the prompts on a half-supplied price would skip the model instead of asking
    # for the number it is missing.
    priced_by_flags = options.input_per_mtok is not None and options.output_per_mtok is not None
    if not priced_by_flags and _self_hosted(kind, options):
        # A self-hosted server ALWAYS takes an explicit price, even when the id shadows a
        # built-in one (someone serving `gpt-4o` locally must not inherit OpenAI's hosted
        # rate). The blank-answer default is 0 ONLY for a locally hosted server (no marginal
        # per-token bill); a remote OpenAI-compatible endpoint (Together, Groq, a proxy) bills
        # real money, so its prompt insists on a typed number.
        local = is_local_endpoint(options.endpoint)
        console.print(
            f"  [dim]{escape(model.id)} is served by "
            f"{escape(options.endpoint or 'a custom endpoint')}; built-in prices do not apply."
            + (
                " 0 = free (local inference); declare an amortized rate to make cost-aware "
                "routing weigh it[/dim]"
                if local
                else " Enter what this server bills you (0 only if it is genuinely free)[/dim]"
            )
        )
        default = 0.0 if local else None
        updates["input_per_mtok"] = _ask_price(
            console, ask, "Input price, USD per 1M tokens", default=default
        )
        updates["output_per_mtok"] = _ask_price(
            console, ask, "Output price, USD per 1M tokens", default=default
        )
    elif not priced_by_flags and needs_price(kind, model):
        console.print(
            f"  [yellow]{escape(model.id)} has no published or built-in price[/yellow]"
            "[dim]; an unpriced candidate reports $0 and a cost-aware policy routes"
            " everything to it[/dim]"
        )
        updates["input_per_mtok"] = _ask_price(console, ask, "Input price, USD per 1M tokens")
        updates["output_per_mtok"] = _ask_price(console, ask, "Output price, USD per 1M tokens")
    return options.model_copy(update=updates) if updates else options


def _self_hosted(kind: ProviderKind, options: EntryOptions) -> bool:
    """Whether this pass registers against an operator-run OpenAI-compatible server."""
    return kind is ProviderKind.OPENAI and options.endpoint is not None


def _default_key_env(kind: ProviderKind) -> str:
    """The variable this backend reads by default, named so the blank answer is not a mystery."""
    env_vars = PROVIDER_ENV_VARS.get(kind, [])
    return env_vars[0] if env_vars else "the backend default"


def _target(
    kind: ProviderKind, model: str, deployment: str | None, endpoint: str | None
) -> tuple[str, str, str, str]:
    """What makes two entries the SAME callable candidate, for idempotent re-registration.

    The endpoint is part of identity: `gpt-oss:20b` on this machine's Ollama and the same id on
    a teammate's vLLM box are two different candidates, and collapsing them would silently
    delete one server's entry, exactly like the multi-account case `_existing_handle` guards.
    """
    return (kind.value, model.lower(), (deployment or model).lower(), _endpoint_key(endpoint))


def _endpoint_key(endpoint: str | None) -> str:
    """One endpoint URL as an identity key: equivalent spellings compare equal.

    Scheme and host are case-insensitive on the wire and a default port is the same route as
    no port, so `HTTP://LocalHost:80/v1` and `http://localhost/v1` must not become two
    candidates a sweep pays for twice. The path keeps its case (paths are case-sensitive), and
    the query rides along verbatim: two proxy endpoints differing only by a routing query
    parameter are two different candidates. A URL urlsplit cannot parse, or whose port is not
    a number, compares as its raw lowercased text rather than crashing the registration flow.
    """
    if not endpoint:
        return ""
    try:
        parts = urlsplit(endpoint.rstrip("/"))
        port_number = parts.port  # raises ValueError on a non-numeric or out-of-range port
    except ValueError:
        return endpoint.rstrip("/").lower()
    if parts.hostname is None:
        return endpoint.rstrip("/").lower()
    scheme = parts.scheme.lower()
    default_port = {"http": 80, "https": 443}.get(scheme)
    port = "" if port_number in (None, default_port) else f":{port_number}"
    query = f"?{parts.query}" if parts.query else ""
    # Embedded userinfo is part of the route (two proxies differing only by user or
    # password are two candidates); it rides verbatim, case-sensitive like the path.
    userinfo, at, _ = parts.netloc.rpartition("@")
    user = f"{userinfo}{at}" if at else ""
    return f"{scheme}://{user}{parts.hostname.lower()}{port}{parts.path}{query}"


def _existing_handle(
    entries: list[PoolEntry],
    kind: ProviderKind,
    model: CatalogModel,
    options: EntryOptions,
) -> str | None:
    """The handle this exact candidate already has in the roster, else None.

    Identity includes `api_key_env`, because a multi-account pool deliberately carries the same
    model twice under two credentials; collapsing those would silently delete one account's
    candidate.
    """
    deployment = options.deployment if kind is ProviderKind.AZURE_OPENAI else None
    wanted = _target(kind, model.id, deployment, options.endpoint)
    for entry in entries:
        if _target(entry.kind, entry.model, entry.deployment, entry.endpoint) != wanted:
            continue
        if entry.api_key_env == (None if kind is ProviderKind.BEDROCK else options.api_key_env):
            return entry.name
    return None


def _handle_base(kind: ProviderKind, model: CatalogModel, options: EntryOptions) -> str:
    """A readable default handle: the deployment name, the model type, or the id's last segment.

    OpenRouter ids are `vendor/model` and Bedrock ids carry routing prefixes, so the raw id makes
    a poor handle for the thing policy artifacts and request logs are keyed on.
    """
    if kind is ProviderKind.AZURE_OPENAI and options.deployment:
        base = options.deployment
    else:
        base = model.model_type or model.id.rsplit("/", maxsplit=1)[-1]
    cleaned = "".join(char if char.isalnum() or char in "._-" else "-" for char in base).strip("-")
    return cleaned or model.id


def _unique_handle(base: str, taken: set[str]) -> str:
    """`base`, or `base-2`, `base-3`, ... : a suggested handle never silently replaces an entry."""
    if base not in taken:
        return base
    suffix = 2
    while f"{base}-{suffix}" in taken:
        suffix += 1
    return f"{base}-{suffix}"


def _ask_text(console: Console, ask: PromptReader, label: str, default: str) -> str:
    """Prompt for a line, showing `default` and returning it when the answer is blank."""
    suffix = f" [dim]\\[{escape(default)}][/dim]" if default else ""
    return _read(ask, f"[bold]{label}[/bold]{suffix}: ") or default


def _ask_price(
    console: Console, ask: PromptReader, label: str, default: float | None = None
) -> float:
    """Prompt until a non-negative price is given.

    `default` is taken on a blank answer; None means there is no sane default to fall back on
    (a hosted model with no published price), so the prompt insists.
    """
    suffix = f" [dim]\\[{default:g}][/dim]" if default is not None else ""
    while True:
        raw = _read(ask, f"[bold]{label}[/bold]{suffix}: ")
        if not raw and default is not None:
            return default
        try:
            value = float(raw)
        except ValueError:
            value = -1.0
        if value >= 0.0:
            return value
        console.print("  [red]enter a non-negative number of USD per 1M tokens[/red]")


def _ask_yes_no(console: Console, ask: PromptReader, question: str, *, default: bool) -> bool:
    """Prompt for yes/no, with `default` taken on a blank answer."""
    hint = "Y/n" if default else "y/N"
    while True:
        raw = _read(ask, f"[bold]{question}[/bold] [dim]({hint})[/dim]: ").lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        console.print("  [red]answer y or n[/red]")


def _read(ask: PromptReader, prompt: str) -> str:
    """Read one prompt line, turning exhausted input into a clean abort instead of a traceback."""
    try:
        return ask(prompt).strip()
    except EOFError:
        raise typer.Abort() from None


def __getattr__(name: str) -> object:
    """Lazy module attribute resolution for deferred CLI imports."""
    if name == "endpoint_catalog":
        from wmo.common.providers.catalog import endpoint_catalog

        globals()["endpoint_catalog"] = endpoint_catalog
        return endpoint_catalog
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


