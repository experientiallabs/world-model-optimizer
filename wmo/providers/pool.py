"""Candidate model pool: the editable roster the routing optimizer selects over.

The pool is one operator-owned TOML file (default `.wmo/pool.toml`, like `.wmo/fallback.toml`),
one `[[model]]` table per candidate. Swapping the roster is editing that file; nothing else in
the harness hardcodes candidate models.

Trust note: the pool file is trusted local config, unlike a model bundle's `config.toml`. That is
why entries may name `api_key_env` (which environment variable holds that account's API key) and
`pool_provider` resolves it and hands the key to `get_provider` as an explicit argument: the
explicit-argument channel is unreachable from bundle-controlled config, so a bundle can never
choose which env var gets read or where its value gets sent.

Pricing: entries for models in the built-in `wmo.tracking.pricing` table need no price fields;
anything else must declare `input_per_mtok`/`output_per_mtok` so downstream cost numbers stay
honest (an unpriced candidate would silently report $0). `cached_input_per_mtok` is the provider
cache-READ price and `cache_write_per_mtok` the cache-WRITE price, carried for cache-aware
routing and compression costs.

`kind = "openrouter"` entries are the one exception, because OpenRouter publishes prices for
every model it fronts: an entry that declares none resolves them from that catalog at load
(`wmo.providers.openrouter_pricing`) and keeps them, so a pool entry needs only a model id and
downstream artifacts still record exact numbers. Offline with nothing cached, the entry falls
back to the same "declare the prices" error, with the catalog failure named.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal, NamedTuple
from urllib.parse import urlsplit

import tomli_w
from llm_waterfall import ChatMaxTokensField
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic_core import ErrorDetails

from wmo.core.files import write_text_atomic
from wmo.core.locks import DEFAULT_LOCK_TIMEOUT_S, file_write_lock
from wmo.core.types import JsonObject
from wmo.providers.base import (
    PreparableProvider,
    Provider,
    ProviderConfig,
    ProviderKind,
    TokenUsage,
)
from wmo.providers.openrouter_pricing import resolve_price as resolve_openrouter_price
from wmo.providers.registry import get_provider
from wmo.tracking.pricing import ModelPrice, price_for, request_price_multipliers

DEFAULT_POOL_PATH = Path(".wmo/pool.toml")

# Kept as a module constant, rather than defaulted at the call, so a test can shorten the wait
# for a CLI command that exposes no timeout flag (`wmo/cli/route_app_test.py`).
POOL_LOCK_TIMEOUT_S = DEFAULT_LOCK_TIMEOUT_S

# D-REPORT ModelRef vocabulary: "frontier" anchors the improvement report's comparison; "open"
# models carry the run-10x-more-for-the-same-budget story.
Tier = Literal["frontier", "open"]
ReasoningEffort = Literal["none", "low", "medium", "high", "max", "xhigh"]


class PoolEntry(BaseModel):
    """One candidate model. `name` is the stable handle policy artifacts and request logs key on.

    `extra="forbid"`: a typo like `api_key_evn` must fail at load, not surface as a 401 at
    request time with no hint (same policy as `.wmo/fallback.toml`'s rungs).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    kind: ProviderKind
    model: str = Field(min_length=1)  # provider runtime id (on Azure: the base model id)
    # Canonical model identity for capability resolution, when `model` is a runtime id that does
    # not carry it: a distilled student's `model` is a `tinker://` weights path, and only its base
    # model determines which request shape and sampling params the backend accepts.
    model_type: str | None = None
    # The output-budget parameter this backend accepts. Built-in models resolve it from the
    # catalog; a self-hosted or beta OpenAI-compatible server that wants the classic `max_tokens`
    # (Tinker's serving endpoint does) declares it here, because the catalog has never heard of it.
    chat_max_tokens_field: ChatMaxTokensField = "max_completion_tokens"
    endpoint: str | None = None
    endpoint_env: str | None = None
    deployment: str | None = None  # Azure deployment name
    deployment_env: str | None = None
    api_version: str | None = None  # Azure api-version
    region: str | None = None  # AWS Bedrock region (bedrock entries only)
    api_key_env: str | None = None  # env var holding this entry's API key (multi-account pools)
    tier: Tier = "frontier"
    # The roster's per-candidate toggle: `enabled = false` keeps the entry (its handle, prices,
    # and comments) but takes it out of everything that CHOOSES models: sweeps, fits, pins, and
    # the platform's endpoint-creation defaults. A disabled entry still validates and still
    # resolves in policies already fitted with it, so flipping the flag never strands an
    # artifact that recorded the entry while it was on.
    enabled: bool = True
    # One effort dial across vendors: OpenAI-family backends forward it as
    # `reasoning.effort` (dispatching through their Responses client), Anthropic as
    # adaptive thinking's `output_config.effort` (low|medium|high|max, probed live
    # 2026-07-29). OpenAI reasoning models may also expose xhigh. Two entries differing only
    # in effort are two ARMS with one runtime model id, which is the router comparison's premise.
    reasoning_effort: ReasoningEffort | None = None
    input_per_mtok: float | None = None
    output_per_mtok: float | None = None
    cached_input_per_mtok: float | None = None  # provider cache-read price, USD per 1M tokens
    cache_write_per_mtok: float | None = None  # provider cache-write price, USD per 1M tokens

    @model_validator(mode="after")
    def _validate_reasoning_effort_route(self) -> PoolEntry:
        # Fail at LOAD, not at the first request of a sweep: Bedrock's Converse API has
        # no effort dial on any path, so an effort-dialed Bedrock entry is a mis-mapped
        # arm that would burn a preflight and then refuse every cell.
        if self.reasoning_effort is not None and self.kind == ProviderKind.BEDROCK:
            raise ValueError(
                f"pool entry {self.name!r}: reasoning_effort is not supported on bedrock "
                "(Converse has no effort dial); route effort-dialed Claude through the "
                "direct anthropic kind instead"
            )
        if self.reasoning_effort == "xhigh" and self.kind == ProviderKind.ANTHROPIC:
            raise ValueError(
                f"pool entry {self.name!r}: Anthropic adaptive thinking supports effort through "
                "max, not xhigh; use max or route an xhigh-capable model through its OpenAI "
                "compatible provider"
            )
        return self

    @model_validator(mode="after")
    def _validate_price(self) -> PoolEntry:
        if (self.input_per_mtok is None) != (self.output_per_mtok is None):
            raise ValueError(
                f"pool model '{self.name}': set both input_per_mtok and output_per_mtok, or neither"
            )
        catalog_note = ""
        if self.input_per_mtok is None and self.kind is ProviderKind.OPENROUTER:
            catalog_note = self._resolve_openrouter_price()
        if (
            self.input_per_mtok is None
            and self.kind is ProviderKind.OPENAI
            and self.endpoint is not None
        ):
            # A self-hosted server (openai kind + explicit endpoint) always takes an explicit
            # price: the built-in table describes OpenAI's HOSTED rates, and an entry whose
            # model id shadows a hosted one (someone serving "gpt-4o" locally) would otherwise
            # silently bill sweeps, routing, and metering at the wrong server's price.
            raise ValueError(
                f"pool model '{self.name}': '{self.model}' is served by a custom endpoint "
                f"({self.endpoint}), so built-in prices do not apply; add input_per_mtok and "
                "output_per_mtok to its entry (0 and 0 for free local inference)"
            )
        if self.input_per_mtok is None and price_for(self.model) is None:
            raise ValueError(
                f"pool model '{self.name}': '{self.model}' has no built-in price;{catalog_note} "
                "add input_per_mtok and output_per_mtok (USD per 1M tokens) to its pool entry"
            )
        if self.endpoint is not None and self.endpoint_env is not None:
            raise ValueError(f"pool model '{self.name}': set endpoint or endpoint_env, not both")
        if self.deployment is not None and self.deployment_env is not None:
            raise ValueError(
                f"pool model '{self.name}': set deployment or deployment_env, not both"
            )
        if (
            self.kind is ProviderKind.AZURE_OPENAI
            and self.deployment is None
            and self.deployment_env is None
        ):
            # Without this the entry loads fine and the first request routed to it 500s
            # from AzureOpenAIProvider._deployment(); load is the validation boundary.
            raise ValueError(
                f"pool model '{self.name}': azure entries need `deployment` (the Azure "
                "deployment name to call), or `deployment_env` naming the variable that holds it"
            )
        if self.kind is ProviderKind.BEDROCK and self.api_key_env is not None:
            # Same boundary: `BedrockProvider.__init__` refuses an explicit key, and a sweep
            # constructs providers lazily per cell, so this would abort mid-run after the
            # candidates ahead of it had already been paid for.
            raise ValueError(
                f"pool model '{self.name}': bedrock authenticates with AWS credentials "
                "(profile/role), not an API key; drop api_key_env from this entry"
            )
        return self

    def _resolve_openrouter_price(self) -> str:
        """Stamp this entry with OpenRouter's published price, returning "" or why it could not.

        Reached only from `_validate_price`, and only for an OpenRouter entry that declared no
        price: no other provider's config validation touches the catalog, and nothing fetches
        at import time.

        Stamping (rather than looking the price up per call) is what makes a fitted policy
        stable. `RoutingPolicy.pool` and `OutcomeMatrix.pool` serialize these entries verbatim,
        so the numbers a policy was fitted under travel with it and re-validating that artifact
        finds the prices already set, which short-circuits this method entirely. A later vendor
        price change cannot silently re-price a policy already in production; refitting or a
        fresh `load_pool` is what picks it up.

        Returns:
            An empty string on success, else a clause naming the catalog failure, ready to be
            spliced into the caller's "declare the prices explicitly" error.
        """
        resolution = resolve_openrouter_price(self.model)
        if resolution.price is None:
            return f" {resolution.detail};"
        self.input_per_mtok = resolution.price.input_per_mtok
        self.output_per_mtok = resolution.price.output_per_mtok
        # Cache tiers are published per model too, but an explicit entry value stays the
        # operator's (a negotiated rate must not be overwritten by the public list price).
        if self.cached_input_per_mtok is None:
            self.cached_input_per_mtok = resolution.price.cache_read_per_mtok
        if self.cache_write_per_mtok is None:
            self.cache_write_per_mtok = resolution.price.cache_write_per_mtok
        return ""

    def price(self) -> ModelPrice:
        """This entry's price row: the explicit override, else the built-in pricing table."""
        if self.input_per_mtok is not None and self.output_per_mtok is not None:
            return ModelPrice(
                input_per_mtok=self.input_per_mtok, output_per_mtok=self.output_per_mtok
            )
        price = price_for(self.model)
        if price is None:  # unreachable after validation; keep the failure loud, not $0
            raise ValueError(f"pool model '{self.name}': no price available for '{self.model}'")
        return price

    def cost_usd(self, usage: TokenUsage) -> float:
        """Effective USD cost of `usage` priced by THIS entry's row (overrides included).

        Cache-adjusted on both tiers: cache-read tokens (`usage.cached_input_tokens`) bill at
        `cached_input_per_mtok` and cache-write tokens (`usage.cache_write_input_tokens`) at
        `cache_write_per_mtok`; each tier falls back to the built-in price row's rate when the
        entry carries no override, and to the full input rate when the row has no tier either
        (never silently free). The global `wmo.tracking.pricing.cost_usd` only knows the
        built-in table; pool entries with explicit prices must be costed here or they would
        silently read $0. This aggregate form cannot infer per-request long-context tiers;
        callers holding one provider request use `call_cost_usd`, and multi-call paths sum it
        at the request boundary.
        """
        price = self.price()
        read = min(usage.cached_input_tokens, usage.input_tokens)
        write = min(usage.cache_write_input_tokens, usage.input_tokens - read)
        read_rate = self.cached_input_per_mtok
        if read_rate is None:
            read_rate = (
                price.cache_read_per_mtok
                if price.cache_read_per_mtok is not None
                else price.input_per_mtok
            )
        write_rate = self.cache_write_per_mtok
        if write_rate is None:
            write_rate = (
                price.cache_write_per_mtok
                if price.cache_write_per_mtok is not None
                else price.input_per_mtok
            )
        return (
            (usage.input_tokens - read - write) * price.input_per_mtok
            + read * read_rate
            + write * write_rate
            + usage.output_tokens * price.output_per_mtok
        ) / 1_000_000

    def call_cost_usd(self, usage: TokenUsage) -> float:
        """Price one provider request, including model-specific context tiers."""
        base = self.price()
        read = min(usage.cached_input_tokens, usage.input_tokens)
        write = min(usage.cache_write_input_tokens, usage.input_tokens - read)
        read_rate = self.cached_input_per_mtok
        if read_rate is None:
            read_rate = (
                base.cache_read_per_mtok
                if base.cache_read_per_mtok is not None
                else base.input_per_mtok
            )
        write_rate = self.cache_write_per_mtok
        if write_rate is None:
            write_rate = (
                base.cache_write_per_mtok
                if base.cache_write_per_mtok is not None
                else base.input_per_mtok
            )
        input_multiplier, output_multiplier = request_price_multipliers(
            self.model_type or self.model,
            usage.input_tokens,
        )
        return (
            (usage.input_tokens - read - write) * base.input_per_mtok * input_multiplier
            + read * read_rate * input_multiplier
            + write * write_rate * input_multiplier
            + usage.output_tokens * base.output_per_mtok * output_multiplier
        ) / 1_000_000

    def provider_config(self) -> ProviderConfig:
        return ProviderConfig(
            kind=self.kind,
            model=self.model,
            model_type=self.model_type,
            chat_max_tokens_field=self.chat_max_tokens_field,
            endpoint=self._env_backed_value(
                literal=self.endpoint,
                env_name=self.endpoint_env,
                field="endpoint",
            ),
            deployment=self._env_backed_value(
                literal=self.deployment,
                env_name=self.deployment_env,
                field="deployment",
            ),
            api_version=self.api_version,
            reasoning_effort=self.reasoning_effort,
            region=self.region,
        )

    def _env_backed_value(
        self,
        *,
        literal: str | None,
        env_name: str | None,
        field: str,
    ) -> str | None:
        """Resolve a non-secret pool reference without serializing its value."""
        if literal is not None:
            return literal
        if env_name is None:
            return None
        value = os.environ.get(env_name)
        if not value:
            raise ValueError(
                f"pool model '{self.name}': environment variable {env_name} for {field} "
                "is unset or empty"
            )
        return value


class ModelPool(BaseModel):
    """The full candidate roster, as loaded from one pool TOML."""

    models: list[PoolEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_names(self) -> ModelPool:
        seen: set[str] = set()
        for entry in self.models:
            if entry.name in seen:
                raise ValueError(f"pool model '{entry.name}' is declared twice; names are handles")
            seen.add(entry.name)
        return self

    def entry(self, name: str) -> PoolEntry:
        for candidate in self.models:
            if candidate.name == name:
                return candidate
        available = ", ".join(m.name for m in self.models)
        raise KeyError(f"no pool model named '{name}'; available: {available}")

    def enabled_models(self) -> list[PoolEntry]:
        """The candidates that participate in model selection (`enabled` not flipped off).

        Everything that CHOOSES models reads this instead of `models`: `wmo optimize route
        sweep`'s preflight, `wmo optimize route pin`, and the platform's endpoint-creation
        defaults. Loading and validation stay on the full roster, so a disabled entry keeps
        failing loudly on a typo rather than silently rotting until it is re-enabled.
        """
        return [entry for entry in self.models if entry.enabled]


# Hostnames that mean "this machine": how display and the platform's serving boundary recognize
# a locally hosted, OpenAI-compatible endpoint (Ollama, vLLM, llama.cpp) without a new provider
# kind. The wire behavior is identical to any custom endpoint; only copy and, in a container,
# host translation care.
_LOCAL_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"})  # noqa: S104


def is_local_endpoint(endpoint: str | None) -> bool:
    """Whether `endpoint` points at this machine (see `_LOCAL_HOSTNAMES`; `*.local` counts too).

    Never raises: this feeds display copy and price defaulting, and `urlsplit` raises
    `ValueError` on malformed bracket URLs (`http://[::1:8000`), which must read as "not
    local", not crash the roster table or the endpoint prompt.
    """
    if not endpoint:
        return False
    try:
        host = urlsplit(endpoint).hostname
    except ValueError:
        return False
    if host is None:
        return False
    return host.lower() in _LOCAL_HOSTNAMES or host.lower().endswith(".local")


def load_pool(path: Path = DEFAULT_POOL_PATH) -> ModelPool:
    """Load and validate the pool file at `path`.

    Every failure names the file and the command that writes it. Callers turn what this raises
    straight into the user's error (`wmo optimize model`, `wmo optimize route sweep`), so a bare
    `tomllib`/pydantic message would reach an operator as a schema dump with a pydantic.dev URL,
    no path, and nothing to type next. The roster is an operator-edited file, and the whole
    point of the message is to say which file and how to repair it. A file that exists but
    declares no candidate is answered like a missing one: with the path and the commands that
    write an entry.

    Args:
        path: The pool TOML to read.

    Returns:
        The validated roster.

    Raises:
        FileNotFoundError: No file at `path`.
        ValueError: The file is not valid TOML, declares no `[[model]]` table, or has an entry
            that fails validation; the message names the path and each failing field.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"no model pool file at {path}; register candidates with `wmo providers set` (or "
            "write the file by hand: one [[model]] table per candidate, fields: name, kind, "
            "model, and for non-built-in models input_per_mtok/output_per_mtok; "
            "endpoint/deployment/api_version/api_key_env as the backend needs)"
        )
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(
            f"pool file {path} is not valid TOML ({exc}); fix the file, or move it aside and "
            "re-register the candidates with `wmo providers set`"
        ) from exc
    entries = data.get("model", [])
    if not isinstance(entries, list):
        # `[model]` declares ONE table; the roster is an array of tables, which needs the doubled
        # brackets. Worth naming, since it is a typo on the very syntax the missing-file message
        # above recommends.
        raise ValueError(
            f"{path} declares a single [model] table; the candidate pool is an array of tables, "
            "so each candidate needs DOUBLED brackets: [[model]]"
        )
    if not entries:
        raise ValueError(
            f"no [[model]] tables in {path}, so the candidate pool is empty; register a hosted "
            "model with `wmo providers set <provider>`, add a distilled student with "
            "`wmo optimize route student <run-dir> --input-per-mtok <p> --output-per-mtok <p>`, "
            "or write one [[model]] table per candidate by hand"
        )
    try:
        return ModelPool.model_validate({"models": entries})
    except ValidationError as exc:
        # `loc` is ("models", <table index>, <field>); the leading "models" is this function's
        # own wrapper key and means nothing to someone reading their TOML, so drop it and name
        # the [[model]] table by the position they can count to in the file.
        details = "; ".join(_entry_error(err) for err in exc.errors())
        raise ValueError(
            f"pool file {path} is not a valid model pool ({details}); fix the entry, or "
            "re-register the candidate with `wmo providers set`"
        ) from exc


def _entry_error(error: ErrorDetails) -> str:
    """One pydantic pool error as `[[model]] N field: message`, for a TOML-file reader."""
    # A whole-pool failure (the duplicate-name rule) carries no entry position at all.
    location = list(error["loc"][1:])
    index = location[0] if location else None
    if not isinstance(index, int):
        return error["msg"]
    field = ".".join(str(part) for part in location[1:])
    table = f"[[model]] {index + 1}"
    return f"{table} {field}: {error['msg']}" if field else f"{table}: {error['msg']}"


def pool_api_key(entry: PoolEntry) -> str | None:
    """One entry's own API key, read from its `api_key_env`; None means backend defaults.

    Separate from `pool_provider` so a caller about to spend money on a WHOLE pool can resolve
    every candidate's credentials up front (`wmo optimize route sweep` does, before its cost
    confirmation) instead of discovering an unset variable at the first cell of a candidate it
    already paid to reach. The lookup is local `os.environ`, so it costs nothing to run early.

    Raises:
        ValueError: `api_key_env` names a variable that is unset or empty.
    """
    if not entry.api_key_env:
        return None
    api_key = os.environ.get(entry.api_key_env)
    if not api_key:
        raise ValueError(
            f"pool model '{entry.name}': environment variable {entry.api_key_env} is unset "
            "or empty; export that account's API key or drop api_key_env to use the "
            "backend's default credentials"
        )
    return api_key


class PoolWrite(NamedTuple):
    """What one `upsert_pool_entry` did to the file, for a caller that has to report it.

    Two independent facts, because the CLI says a different thing for each and a bool cannot
    carry both. `replaced` answers "was an entry of this name already there", which is the
    question `wmo optimize route student` prompts about BEFORE writing. `rewritten` answers
    "were the operator's comments dropped", which a replacement always does and an ADD can now
    also do, when the roster had to be normalized out of the legacy inline form.
    """

    replaced: bool
    rewritten: bool


def upsert_pool_entry(
    entry: PoolEntry,
    path: Path = DEFAULT_POOL_PATH,
    *,
    lock_timeout_s: float | None = None,
) -> PoolWrite:
    """Add `entry` to the pool roster at `path`, replacing any entry with the same name.

    The one write path for the roster, so a trained model becomes routable without hand-editing
    TOML. Entries already in the file are carried through as the exact tables they were written
    as: re-serializing them through `PoolEntry` would stamp every default (`tier`,
    `chat_max_tokens_field`) into a file an operator maintains by hand.

    Adding a NEW entry appends its one `[[model]]` section and touches nothing else, so comments,
    key order, and spacing all survive byte for byte. REPLACING an entry has to remove the old
    table, which means re-rendering the file through `tomllib` -> `tomli_w`: that keeps every
    entry's fields but DROPS comments, because neither library round-trips them. Callers that can
    prompt should say so before replacing (`wmo optimize route student` does).

    One kind of ADD re-renders too, and so also drops comments: a roster written as the inline
    array `model = [ {...} ]` rather than as `[[model]]` sections cannot be appended to (the
    section header would be a second top-level `model` key), so it is normalized on the next
    write. Releases up to 0.2.1 could write that form; a file this command has written since is
    always in section form and always takes the append path. That case is why the return value
    reports `rewritten` separately from `replaced`: an add that silently deleted the comments
    recording which account each row bills to, while printing a plain "added", is not acceptable.

    Whichever body is produced, it is parsed back and required to read as exactly the intended
    roster before it is committed (`_parses_back`), so no write can leave the pool unloadable.

    The merged roster is validated as a whole `ModelPool` BEFORE anything is written, and the
    write itself goes through a temp file, so a rejected entry can never leave the pool
    unloadable. That matters more than it looks: `load_pool` is what serving and the routing
    optimizer both call, so a broken pool file takes every endpoint down, not just the candidate
    being added.

    The whole read-validate-write cycle runs under an exclusive cross-process lock (see
    `wmo.core.locks.file_write_lock`), so two racing registrations both land. Without it each reads
    the same roster and the later write erases the earlier entry, while both commands report
    success: a model an operator registered is simply not in the pool, and nothing says so.

    Args:
        entry: The candidate to add, or to replace an existing entry of the same name with.
        path: The pool TOML. Created, with its parent directory, when absent.
        lock_timeout_s: Seconds to wait for another writer's lock before giving up; the default
            is `POOL_LOCK_TIMEOUT_S`.

    Returns:
        What the write did: see `PoolWrite`.

    Raises:
        ValueError: If `path` exists but is not a readable pool file, if it is already an invalid
            roster before this entry is added, or if adding `entry` would make it one.
        FileLockTimeout: If another writer holds the roster's lock for the whole wait.
    """
    timeout_s = POOL_LOCK_TIMEOUT_S if lock_timeout_s is None else lock_timeout_s
    with file_write_lock(path, what="the model pool", timeout_s=timeout_s):
        return _upsert_locked(entry, path)


def _upsert_locked(entry: PoolEntry, path: Path) -> PoolWrite:
    """The read-validate-write cycle of `upsert_pool_entry`, with the roster's lock held."""
    tables = _raw_tables(path)
    if tables:
        # Validate what is ALREADY there first, so a pre-existing bad row is reported as a
        # pre-existing bad row. Otherwise a typo an operator left in some other entry surfaces as
        # "adding 'student' would make this an invalid pool", pointing at the flags they just got
        # right instead of at the line that is actually wrong.
        try:
            ModelPool.model_validate({"models": tables})
        except ValidationError as exc:
            raise ValueError(
                f"pool file {path} is already invalid, before adding '{entry.name}': {exc}"
            ) from exc
    kept = [table for table in tables if table.get("name") != entry.name]
    replaced = len(kept) != len(tables)
    # exclude_defaults keeps the written table as small as the hand-written ones around it, and
    # exclude_none drops the backend knobs this kind does not use. mode="json" turns the
    # ProviderKind enum into the plain string TOML can hold.
    table = entry.model_dump(mode="json", exclude_defaults=True, exclude_none=True)
    merged = [*kept, table]
    try:
        ModelPool.model_validate({"models": merged})
    except ValidationError as exc:
        raise ValueError(f"adding '{entry.name}' would make {path} an invalid pool: {exc}") from exc
    # The common case is a first registration, and appending the one new section byte-preserves
    # everything already in the file: an operator's comments, key order, and spacing. Rewriting
    # would silently destroy the comments that say which account a row bills to. A replacement
    # cannot append (removing the old table means re-rendering), and neither can a file whose
    # roster is not in section form, so both of those fall through to the re-render below.
    existed = path.is_file()
    body = None
    if not replaced and existed:
        body = _parses_back(_appended_body(path.read_text(encoding="utf-8"), table), merged)
    # Creating the file is not a rewrite: there was nothing in it to lose, and reporting one
    # would put "its comments are gone" on a first registration.
    rewritten = body is None and existed
    if body is None:
        body = _parses_back(_render_sections(merged), merged)
    if body is None:  # the re-render is what every other path falls back TO; it has no fallback
        raise ValueError(
            f"refusing to write {path}: the roster rendered to TOML that does not read back as "
            f"the {len(merged)} intended entries. This is a bug in wmo, not in your file, which "
            "is untouched; the likely cause is a pool field whose value is not a plain scalar"
        )
    write_text_atomic(path, body)
    return PoolWrite(replaced=replaced, rewritten=rewritten)


def _parses_back(body: str, merged: list[JsonObject]) -> str | None:
    """`body` if it reads back as exactly the roster `merged`, else None.

    The commit gate, checked on EVERY write rather than only on the append path, because the two
    ways to render this file fail differently and both fail silently. Appending to a roster
    written as the inline array `model = [ {...} ]` (the form releases up to 0.2.1 could produce)
    is a duplicate top-level key, so the file stops loading. And `tomli_w.dumps` on a table with a
    non-scalar value emits a `[key]` header, which under a hand-written `[[model]]` parses as a
    SIBLING top-level table: the field silently vanishes from the entry with no error at all,
    which is worse. `PoolEntry`'s all-scalar schema is what rules the second one out today, and
    this is what keeps it ruled out if that ever changes.

    Cheap enough to be unconditional: one `tomllib` parse of a file that holds a handful of
    entries, once per registration, under a lock that is already held.
    """
    try:
        parsed = tomllib.loads(body)
    except tomllib.TOMLDecodeError:
        return None
    return body if parsed.get("model") == merged else None


def _render_sections(tables: list[JsonObject]) -> str:
    """`tables` as one `[[model]]` section each, blank-line separated.

    The section headers are written HERE rather than left to `tomli_w.dumps({"model": tables})`,
    which does not reliably produce them. `tomli_w` picks the rendering per value: an array of
    tables comes out as `[[model]]` sections only when at least one of them fails
    `is_suitable_inline_table`, which is a LINE-LENGTH heuristic (100 characters). A roster of
    short entries (`{name, kind, model}` for an OpenAI model) renders instead as the inline array
    `model = [ {...} ]`. That form is valid TOML and still loads, but it cannot be APPENDED to:
    adding a second `model = [...]` (or a `[[model]]` section) is a duplicate top-level key, so
    every later `load_pool` fails with `Cannot overwrite a value` and the roster has to be
    repaired by hand. Verbose entries (an Azure row carrying deployment/api_version/api_key_env)
    cross 100 characters and come out as sections, which is what made the corruption look
    intermittent rather than what it was: data-dependent, and reliably hit by the smallest entry
    anyone registers.

    Writing the header ourselves takes that heuristic out of the write path entirely, in both
    directions: the file this command creates is one an append can extend, so the byte-preserving
    add path in `_appended_body` stays reachable no matter how short the entries are.

    Correct only while every table is a flat dict of scalars, which `PoolEntry` is: `tomli_w.dumps`
    on a table holding a nested value emits a `[key]` header for it, and under a hand-written
    `[[model]]` that header reads as a SIBLING top-level table, so the field leaves the entry with
    no error raised. Nothing here enforces that; `_parses_back` is what catches it at the commit
    point, for this and for the whole-roster render alike.
    """
    return "\n".join(f"[[model]]\n{tomli_w.dumps(table)}" for table in tables)


def _appended_body(existing: str, table: JsonObject) -> str:
    """`existing` plus one `[[model]]` section for `table`, separated by a blank line.

    A pure string builder: whether the result is SAFE to write is `_parses_back`'s question, asked
    at the commit point for every render rather than only for this one. Appending is what
    byte-preserves an operator's comments, and it works against any roster already in section
    form, which `_render_sections` guarantees for every file this command has written. A roster
    still in the inline `model = [ {...} ]` form from 0.2.1 or earlier is the case that fails: a
    section header appended to one is a duplicate top-level `model` key, the caller falls back to
    a full re-render, and the file is normalized to section form so the next add can append again.

    Args:
        existing: The current file contents, already known to parse (`_raw_tables` read it).
        table: The new entry's table, as rendered into the file.

    Returns:
        The full file contents to write, unverified.
    """
    if existing.endswith("\n\n"):
        separator = ""
    else:
        separator = "\n" if existing.endswith("\n") else "\n\n"
    return f"{existing}{separator}{_render_sections([table])}"


def _raw_tables(path: Path) -> list[JsonObject]:
    """The `[[model]]` tables in `path`, exactly as written; empty when the file does not exist."""
    if not path.is_file():
        return []
    try:
        parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(
            f"pool file {path} is not valid TOML ({exc}); fix the file, or move it aside to "
            "start a fresh roster"
        ) from exc
    tables = parsed.get("model", [])
    if not isinstance(tables, list) or not all(isinstance(table, dict) for table in tables):
        raise ValueError(
            f"pool file {path} does not hold an array of [[model]] tables; every candidate is its "
            "own [[model]] table (fields: name, kind, model, and prices for non-built-in models)"
        )
    return [dict(table) for table in tables]


def pool_provider(entry: PoolEntry) -> Provider:
    """Construct the provider for one pool entry, resolving its per-account API key.

    Construction is side-effect free for every backend (`wmo.providers.registry`): each one only
    stores its config and defers the SDK import, the credential read, and the client to its first
    request. Which is also why construction alone is a weak pre-flight: use
    `prepare_pool_provider` when the point is to learn whether a candidate can be CALLED.

    A construction failure is re-raised naming the entry and its kind: a backend that refuses to
    be built ("Bedrock authenticates with AWS credentials, not an API key") knows nothing about
    the pool it came from, and the caller is usually looping over candidates.

    Raises:
        ValueError: The entry names an unset `api_key_env`, or its backend refuses this config.
    """
    api_key = pool_api_key(entry)
    try:
        return get_provider(entry.provider_config(), api_key=api_key)
    except ValueError as exc:
        raise ValueError(f"pool model '{entry.name}' (kind={entry.kind.value}): {exc}") from exc


def static_requirements(entry: PoolEntry) -> list[str]:
    """What one entry's KIND needs from the entry alone, worded for the file it is edited in.

    Read before any SDK is imported and before any client is built, so an entry that could never
    be called fails on its own config. Each item is derived from the backend's request path, not
    guessed:

    - azure needs `deployment` (on the wire, Azure's `model` IS the deployment name, so
      `AzureOpenAIProvider._deployment` refuses without it) and `api_version` (its client cannot be
      built without one). `endpoint` is NOT required here: it legitimately comes from
      AZURE_OPENAI_ENDPOINT, which the provider resolves.
    - tinker needs a base model name, because `TinkerChatProvider` resolves its renderer and
      tokenizer from `ProviderConfig.model_type` and a pool entry has no field that fills it, so a
      `tinker://` weights path in `model` can never render a prompt.
    - bedrock, openai, openai_responses, anthropic and openrouter need nothing from the entry: a
      region (for bedrock) or a credential (for the rest) is what they need, and those resolve
      from the local environment, not from this file. openrouter also needs no price, because an
      entry that declares none resolves one from its published catalog at load.

    Kept out of `PoolEntry`'s own validation on purpose, unlike the two rules that are there: a
    saved `OutcomeMatrix` carries its pool inline, so tightening load-time validation would
    retroactively refuse matrices whose entries only ever supplied prices. Being callable matters
    where a candidate is about to be called.

    Returns:
        One human-readable complaint per unmet requirement; empty when the entry is complete.
    """
    match entry.kind:
        case ProviderKind.AZURE_OPENAI:
            missing: list[str] = []
            if entry.deployment is None:
                missing.append(
                    "`deployment` (the Azure deployment name every request names as its model)"
                )
            if entry.api_version is None:
                missing.append(
                    "`api_version` (AzureOpenAIProvider cannot build its client without one)"
                )
            return [f"azure entries need {item}" for item in missing]
        case ProviderKind.TINKER:
            if entry.model.startswith("tinker://"):
                return [
                    "a tinker entry's `model` cannot be a tinker:// weights path: the renderer and "
                    "tokenizer resolve from the BASE model name, which a pool entry has no field "
                    "to carry, so name the base model (e.g. 'Qwen/Qwen3-8B') instead"
                ]
            return []
        case (
            ProviderKind.BEDROCK
            | ProviderKind.OPENAI
            | ProviderKind.OPENAI_RESPONSES
            | ProviderKind.ANTHROPIC
            | ProviderKind.OPENROUTER
        ):
            return []


def prepare_pool_provider(entry: PoolEntry) -> Provider:
    """Construct one entry's provider AND resolve every prerequisite that needs no request.

    The pre-flight seam for a caller about to spend money on a whole roster (`wmo optimize route
    sweep`). `pool_provider` alone is too weak for that: every backend builds its SDK client
    lazily, so an uninstalled SDK extra, an unset credential, or a region that resolves nowhere
    still lands at that candidate's FIRST CALL, after the candidates ahead of it have been paid
    for. This runs the kind's `static_requirements` and then `PreparableProvider.prepare`, which
    forces the lazy client to be built where that is free.

    Free, and provably so: no backend's `prepare` issues a request (see each one's docstring).
    Verifying a candidate over the wire is deliberately NOT done here, because `wmo providers
    verify` bills a real call per model, which a pre-flight that runs before spend is authorized
    may not do. Two backends therefore keep a documented residual gap, and callers should say so:
    bedrock cannot resolve AWS CREDENTIALS locally (building the client walks a chain that reaches
    the instance-metadata endpoint over the network, and succeeds with no credentials anyway), and
    tinker cannot resolve SERVICE REACHABILITY (constructing the client connects and pins a
    server-side session). Both stay first-call failures.

    Returns:
        The prepared provider. Callers that only wanted the check may discard it; the sweep does,
        because `evaluate_pool` builds its own per cell to keep per-episode provider state fresh.

    Raises:
        ValueError: This entry cannot be used, named with its kind and what to do about it.
    """
    problems = static_requirements(entry)
    if problems:
        raise ValueError(
            f"pool model '{entry.name}' (kind={entry.kind.value}): {'; '.join(problems)}"
        )
    provider = pool_provider(entry)
    if isinstance(provider, PreparableProvider):
        try:
            provider.prepare()
        except Exception as exc:  # noqa: BLE001 - every backend raises its own SDK's type here
            # Re-raised as one usage error naming the entry: the SDK's own message says what is
            # missing, and the caller is looping over candidates in a file it wants to edit.
            raise ValueError(f"pool model '{entry.name}' (kind={entry.kind.value}): {exc}") from exc
    return provider
