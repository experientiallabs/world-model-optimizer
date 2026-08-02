"""Provider interface and shared config/value types."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

from llm_waterfall import ChatMaxTokensField, ChatRequest, ChatResponse
from pydantic import BaseModel, Field


class ProviderKind(StrEnum):
    """Which backend serves a provider's completions."""

    ANTHROPIC = "anthropic"  # Opus 4.8 direct
    BEDROCK = "bedrock"  # Claude 4.8 via AWS
    AZURE_OPENAI = "azure"  # GPT 5.5 via the Azure OpenAI service
    OPENAI = "openai"  # GPT 5.5 direct
    OPENAI_RESPONSES = "openai_responses"  # GPT 5.x direct via the Responses API
    OPENROUTER = "openrouter"  # any OpenRouter-fronted model, on one OPENROUTER_API_KEY
    TINKER = "tinker"  # Tinker LoRA sampling for distillation rollouts (lazy distill extra)


class EmbedderKind(StrEnum):
    """Which embedder supplies phi for retrieval.

    `HASHING` is the offline, zero-config default (no creds, no network). The other three map 1:1 to
    the same-named `ProviderKind` and use that backend's embeddings API. Anthropic is intentionally
    absent — it has no embeddings API; configure `BEDROCK`/`OPENAI`/`AZURE_OPENAI` (or `HASHING`).
    """

    HASHING = "hashing"  # offline HashingEmbedder (default)
    BEDROCK = "bedrock"  # Titan on AWS Bedrock
    OPENAI = "openai"  # OpenAI embeddings
    AZURE_OPENAI = "azure"  # Azure OpenAI embedding deployment

    def provider_kind(self) -> ProviderKind:
        """The ProviderKind backing this embedder. Raises for `HASHING` (no provider)."""
        if self is EmbedderKind.HASHING:
            raise ValueError("HASHING is the offline embedder; it has no backing provider")
        return ProviderKind(self.value)


Role = Literal["user", "assistant"]


class Message(BaseModel):
    """One turn of a completion request."""

    role: Role
    content: str


class TokenUsage(BaseModel):
    """Tokens one call billed, with the cache-read and cache-write subsets broken out."""

    input_tokens: int = 0
    output_tokens: int = 0
    # Prompt tokens served from the provider's cache (a SUBSET of input_tokens, never in
    # addition to it). Cache reads bill at a discounted rate, so effective-cost accounting
    # (PoolEntry.cost_usd, D-METERING records) needs the split; providers that don't report
    # cache usage leave it 0, which prices the whole prompt at the full input rate.
    # Providers whose APIs report cache reads BESIDE the input count (Anthropic Messages,
    # Bedrock Converse) normalize at the construction site: input_tokens = fresh + cache_read.
    cached_input_tokens: int = 0
    # Prompt tokens WRITTEN to the provider's cache this call (Anthropic
    # cache_creation_input_tokens / Bedrock cacheWriteInputTokens). Same subset contract as
    # cached_input_tokens: a subset of input_tokens, disjoint from the read subset, normalized
    # at the construction site (input_tokens = fresh + cache_read + cache_write). Writes bill
    # at a PREMIUM (Anthropic 5m TTL: 1.25x input), so cache-adjusted cost needs this split
    # too; providers that don't report writes (OpenAI charges no write premium) leave it 0.
    cache_write_input_tokens: int = 0
    # Reasoning tokens are a SUBSET of output_tokens. They bill at the output rate, but keeping
    # the split is necessary for per-arm reasoning-effort analysis.
    reasoning_tokens: int = 0


def structured_token_usage(response: ChatResponse) -> TokenUsage:
    """Project a structured response onto cache-aware and reasoning-aware counters."""
    counts = response.token_usage()
    cached = 0
    cache_write = 0
    reasoning = 0
    if response.usage is not None:
        extras = response.usage.model_extra or {}
        prompt_details = extras.get("prompt_tokens_details")
        if isinstance(prompt_details, dict):
            cached = _nonnegative_int(prompt_details.get("cached_tokens"))
            cache_write = _nonnegative_int(prompt_details.get("cache_write_tokens"))
        completion_details = extras.get("completion_tokens_details")
        if isinstance(completion_details, dict):
            reasoning = _nonnegative_int(completion_details.get("reasoning_tokens"))
    return TokenUsage(
        input_tokens=counts.input_tokens,
        output_tokens=counts.output_tokens,
        cached_input_tokens=min(cached, counts.input_tokens),
        cache_write_input_tokens=min(cache_write, max(0, counts.input_tokens - cached)),
        reasoning_tokens=min(reasoning, counts.output_tokens),
    )


def _nonnegative_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


class Completion(BaseModel):
    """One completion: its text, what it cost in tokens, and which model served it."""

    text: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    # The model that actually served, when the provider is a failover chain and a fallback took
    # the call. None (the norm) means "the configured model" — metering falls back to config.
    # min_length=1 keeps "" impossible, so `completion.model or config.model` is exact.
    model: str | None = Field(default=None, min_length=1)


DEFAULT_MAX_TOKENS = 8192


class VerifyResult(BaseModel):
    """Outcome of a provider credential/reachability check, per `wmo providers verify`."""

    ok: bool
    kind: ProviderKind
    model: str
    detail: str = ""


class ProviderConfig(BaseModel):
    """Everything needed to construct one provider.

    Credentials are read from the environment by default (keys named per backend); the explicit
    backend knobs below override. The env var names are documented in `wmo.config`.
    """

    kind: ProviderKind
    # Canonical, provider-independent identity. ``model`` remains the exact
    # provider runtime id for SDK calls and old persisted configs.
    model_type: str | None = None
    model: str
    embed_model: str | None = None  # embeddings model id / Azure embedding deployment
    embed_dim: int | None = None  # requested embedding dimension (Titan v2, text-embedding-3-*)
    # Backend knobs (only some apply per kind):
    endpoint: str | None = None  # Azure OpenAI / custom base URL
    region: str | None = None  # AWS Bedrock region
    deployment: str | None = None  # Azure OpenAI deployment name
    api_version: str | None = None  # Azure OpenAI API version
    reasoning_effort: str | None = None  # OpenAI Responses reasoning.effort
    # The serialized default stays stable for persisted configs. When callers do not explicitly
    # set this field, built-in models resolve it from the canonical ProviderModel catalog.
    chat_max_tokens_field: ChatMaxTokensField = "max_completion_tokens"

    def resolved_chat_max_tokens_field(self) -> ChatMaxTokensField:
        """Return the output-token field accepted by this configured model."""
        # Local import avoids a module cycle: the model catalog imports ProviderKind above.
        from wmo.providers.models import resolve_chat_max_tokens_field

        model = self.model_type or self.model
        return resolve_chat_max_tokens_field(
            self.kind,
            model,
            fallback=self.chat_max_tokens_field,
        )

    def resolved_chat_forward_temperature(self) -> bool:
        """Return whether this configured model accepts chat temperature."""
        # Local import avoids a module cycle: the model catalog imports ProviderKind above.
        from wmo.providers.models import resolve_provider_model

        # Explicit OpenAI-compatible endpoints are user-owned sampling servers even when their
        # configured model label happens to match a built-in reasoning model.
        if self.kind is ProviderKind.OPENAI and self.endpoint is not None:
            return True
        model = self.model_type or self.model
        return resolve_provider_model(self.kind, model).forward_temperature


def normalize_chat_temperature(
    request: ChatRequest,
    *,
    forward_temperature: bool,
) -> ChatRequest:
    """Apply one model's sampling capability without mutating the provider-neutral request."""
    if forward_temperature or request.temperature is None:
        return request
    return request.model_copy(update={"temperature": None})


@runtime_checkable
class Embedder(Protocol):
    """The embedding half of a provider (phi in DreamGym).

    Retrieval depends only on this narrower capability, so it accepts either a full `Provider` or a
    standalone local embedder (`wmo.retrieval.embedders.HashingEmbedder`) without requiring creds.
    """

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class EmbeddingResult(BaseModel):
    """Vectors plus provider-reported billable usage for one embedding request."""

    vectors: list[list[float]]
    usage: TokenUsage = Field(default_factory=TokenUsage)
    model: str = Field(min_length=1)


@runtime_checkable
class UsageReportingEmbedder(Protocol):
    """Optional embedding seam for callers that must meter semantic routing."""

    def embed_with_usage(self, texts: list[str]) -> EmbeddingResult: ...


@runtime_checkable
class Provider(Protocol):
    """The single interface all four backends implement."""

    config: ProviderConfig

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion:
        """Generate a completion. Used by the world model, GEPA, and the judge."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts for retrieval (phi in DreamGym). May delegate to a sibling embed model."""
        ...

    def verify(self) -> VerifyResult:
        """Cheap creds/model check run on startup (`wmo providers verify`)."""
        ...


class StreamChunk(BaseModel):
    """One increment of a streamed completion.

    A stream is zero or more delta chunks followed by exactly one terminal chunk (`done=True`)
    carrying the call's `TokenUsage` (and, for failover chains, the model that actually served).
    Putting usage on the terminal chunk keeps streamed traffic meterable at the provider
    boundary exactly like `complete()`.
    """

    delta: str = ""
    done: bool = False
    usage: TokenUsage | None = None
    model: str | None = Field(default=None, min_length=1)


@runtime_checkable
class StreamingProvider(Protocol):
    """Provider capability for native token streaming.

    Separate from :class:`Provider` for the same reason as :class:`ToolCallingProvider`: the
    world model, judge, and optimizers consume whole completions, while the serving endpoint
    must forward tokens as the upstream model emits them (a synthesized stream would hold the
    full response before the first byte, which defeats streaming). Every shipped backend
    implements it natively.
    """

    def stream(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Iterator[StreamChunk]:
        """Yield delta chunks as the model emits them, then one terminal chunk with usage."""
        ...


@runtime_checkable
class ToolCallingProvider(Protocol):
    """Provider capability for full structured agent requests.

    This stays separate from :class:`Provider`: world-model, judge, and prompt-optimization
    callers need only text, while agent runtimes must preserve tool schemas, tool calls, tool
    results, finish reasons, and usage end to end.
    """

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        """Return one non-streaming structured chat completion."""
        ...


UNPARSED_TOOL_CALLS_KEY = "wmo_unparsed_tool_calls"
"""Extra `ChatChoice` key: parse errors for tool calls the provider could not decode.

A provider whose wire format is raw sampled text (the Tinker student, decoded through a chat
template renderer) can emit a turn whose tool call is malformed or was cut off at the output cap.
Dropping it silently makes the turn look like plain prose, which is how a truncated `write_file`
became a "submitted" episode with reward 0. Providers set this list so the pi runner can feed the
parser's complaint back to the model instead of ending the episode; `ChatChoice` allows extras, so
it rides `wire_payload()` to the runner untouched."""


@runtime_checkable
class PreparableProvider(Protocol):
    """Provider capability for resolving a backend's prerequisites WITHOUT issuing a request.

    Every backend builds its SDK client lazily on first use, so constructing a provider proves
    almost nothing: a missing SDK extra, an unset credential, or an incomplete config still
    surfaces at the first call. This capability exists for the one caller shape that cannot live
    with that, code about to spend money on a whole roster of models (`wmo optimize route sweep`'s
    pre-flight), where a backend that can never be called must be a usage error at the boundary
    instead of a mid-run abort with the earlier candidates already paid for.

    `prepare()` must stay free: it may import the SDK, read the environment and the local AWS/SDK
    credential files, and build the client, and it must NOT touch the network, because a
    pre-flight that bills a call cannot run before spend is authorized. A backend whose client
    cannot be built without a request prepares the locally knowable part only and documents what
    it deliberately leaves for the first call (see `wmo.providers.bedrock` and
    `wmo.providers.tinker`). Kept separate from :class:`Provider` for that reason: not every
    backend can honor the whole contract.
    """

    def prepare(self) -> None:
        """Resolve every prerequisite knowable locally, or raise saying what is missing."""
        ...


@runtime_checkable
class ContextWindowProvider(Protocol):
    """Provider capability for reporting the context window its model is actually served with.

    Kept separate from :class:`Provider` because most backends have no single answer (an API key
    fronts many models). Implement it where the served window is a property of the deployment, so
    the pi runner can calibrate its context guard to the real number instead of a guess.
    """

    def context_window(self) -> int | None:
        """The served context window in tokens, or None when it cannot be determined."""
        ...


# One read-only instance reused across every verify() ping (complete() never mutates messages).
_PING_MESSAGES: list[Message] = [Message(role="user", content="ping")]

# Ping output budget. Reasoning models (GPT-5.x) spend output tokens on reasoning before any
# visible text, and OpenAI 400s ("max_tokens or model output limit was reached") when the budget
# can't cover it — which reads like bad credentials. Non-reasoning models stop after a token or
# two regardless, so the headroom costs nothing there.
PING_MAX_TOKENS = 2048

# Belt-and-suspenders for the above: if a reasoning model spends even the larger ping budget on
# reasoning before emitting output, the resulting error still PROVES the model is reachable (auth
# ok, model exists). Treat these markers as reachable so `verify` passes instead of reporting fail.
_REACHABLE_ERROR_MARKERS = (
    "max_tokens",
    "max_output_tokens",
    "output limit was reached",
    "finish the message because",
)


def verify_via_ping(provider: Provider, ping: Callable[[], object] | None = None) -> VerifyResult:
    """Shared `verify()`: one cheap short completion, reporting failure as ok=False.

    Every backend's verify() is identical apart from its kind/model (both on the config), so they
    all delegate here. `ping` substitutes the probe call when the config dispatches through a
    route plain `complete` would not exercise (e.g. a structured Responses branch). Never
    raises; `verify_all` relies on that to not crash startup.
    """
    cfg = provider.config
    try:
        if ping is None:
            provider.complete("", _PING_MESSAGES, max_tokens=PING_MAX_TOKENS)
        else:
            ping()
    except Exception as exc:  # noqa: BLE001 - verify reports failure, never raises
        # A max-tokens/output-limit error confirms reachability: the request reached the model
        # (auth + model id are valid) and only failed because a reasoning model consumed the 1-token
        # ping budget before producing output. Anything else (auth, missing model, network) is a
        # real failure.
        msg = str(exc).lower()
        if any(marker in msg for marker in _REACHABLE_ERROR_MARKERS):
            return VerifyResult(ok=True, kind=cfg.kind, model=cfg.model)
        return VerifyResult(ok=False, kind=cfg.kind, model=cfg.model, detail=str(exc))
    return VerifyResult(ok=True, kind=cfg.kind, model=cfg.model)
