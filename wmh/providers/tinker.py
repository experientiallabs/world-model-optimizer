"""Tinker sampling provider: serves the pi agent from a Tinker LoRA student.

During distillation rollouts the student model lives on Tinker. This provider
renders each structured chat request to token ids with the base model's
cookbook renderer, samples from a Tinker sampling client, and parses the
sampled tokens back into an OpenAI-style chat response for the pi bridge.

Every successful completion records a `TokenSpan` (exact prompt ids, sampled
ids, and per-token logprobs) into an optional `TokenRecorder`. Downstream
training consumes ONLY these recorded ids (tokens-in tokens-out); text is
never re-encoded, and a sample without per-token logprobs fails loudly.

Each span also carries the canonical message DELTA it added to the episode's
conversation plus that call's tool schemas, so a cross-tokenizer teacher can
re-render the same multi-turn conversation with its own chat template
(`wmh.distill.tokens.reconstruct_conversation`). Only the delta is stored: a
full history per call would be quadratic in text, and real agentic prompts
reach tens of thousands of tokens. The delta boundary is the SAME boundary the
suffix render uses, so the message and token deltas cannot disagree.

Multi-turn prompts are built incrementally from the episode's own token
history: agents re-serialize earlier assistant turns (reformatted tool-call
JSON, collapsed think framing), so re-rendering the full history never
byte-matches the tokens actually sampled and every turn would fragment into
its own training datum. Instead the next prompt is (previous prompt + raw
sampled ids + a rendered suffix of only the new messages); a genuine history
edit falls back to a full re-render and is counted on the recorder.

`config.model_type` carries the base model name (renderer and tokenizer
identity); `config.model` carries either a `tinker://` sampler-weights path or
a base model name for an untrained student. The tinker SDK is an optional
extra imported lazily (`uv sync --extra distill`), same contract as e2b.

Every SDK call is deadline-bounded (`wmh.distill.deadlines`): a wedged
session raises a retryable `TinkerDeadlineError` instead of hanging, and the
provider drops its lazily built sampling client on expiry so the retry
wrapper's next attempt heals through a fresh session.

SDK clients are shared process-wide (`shared_service_client` and
`shared_sampling_client`): the SDK's service client starts a heartbeat task
that strongly references its internal holder, so every constructed client
lives for the rest of the process and keeps one server-side session alive.
Building a fresh client per trial therefore leaks sessions until the service
rejects new session creation with capacity errors (observed live at ~240
cumulative sessions). One `tinker.ServiceClient` plus one `SamplingClient`
per exact model string serve every consumer instead; the SDK documents
`SamplingClient` as thread-safe, and all per-episode state (`TokenRecorder`,
incremental prompt state, renderer) stays on each provider instance.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from llm_waterfall.types import ChatChoice, ChatMessage, ChatTool, ChatUsage
from pydantic import BaseModel, Field, model_validator

from wmh.distill.deadlines import TinkerDeadlineError, call_with_deadline, wait_with_deadline
from wmh.distill.rendering import (
    ChatRendering,
    ParsedAssistantMessage,
    RendererTokenizer,
    build_renderer,
)
from wmh.providers.base import (
    DEFAULT_MAX_TOKENS,
    UNPARSED_TOOL_CALLS_KEY,
    ChatRequest,
    ChatResponse,
    Completion,
    Message,
    ProviderConfig,
    TokenUsage,
    VerifyResult,
    verify_via_ping,
)

if TYPE_CHECKING:
    import tinker

logger = logging.getLogger(__name__)

TINKER_API_KEY_ENV = "TINKER_API_KEY"

_MISSING_TINKER_EXTRA = (
    "the tinker SDK is not installed; run `uv sync --extra distill` to use the tinker provider"
)

# The tinker SamplingParams default; used when a structured request carries no
# temperature (pi normally stamps one on every request).
_DEFAULT_CHAT_TEMPERATURE = 1.0

_WEDGE_REBUILD_THRESHOLD = 3
"""Consecutive deadline expiries before the process-wide service client is rebuilt.

Not 1: an isolated expiry is ordinary under load, and each rebuild pins another server-side
session against a ~240 cumulative cap. Not 10: two live runs burned 30+ minutes and 96
consecutive expiries producing nothing while a fresh process reached the same service in 1.7s."""

_shared_lock = threading.Lock()
"""Guards the process-wide service client and sampling-client cache below."""

_shared_service: tinker.ServiceClient | None = None
_shared_samplers: dict[str, tinker.SamplingClient] = {}
"""Process-wide `SamplingClient` cache, keyed by the exact model string."""


def shared_service_client() -> tinker.ServiceClient:
    """The process-wide `tinker.ServiceClient`, constructed at most once.

    The SDK's service client starts a heartbeat task that strongly references
    its internal holder, so every constructed client lives (and pins one live
    server-side session) for the rest of the process. Constructing one per
    trial therefore leaks sessions until the service rejects new session
    creation with capacity errors; every wmh consumer (the rollout provider,
    the teacher scorer, the distill loop) shares this single client instead.

    The API key is checked on every call, before the cache, so a missing key
    stays an actionable error rather than an SDK-internal auth failure.

    Returns:
        The shared service client.

    Raises:
        ImportError: If the tinker SDK is not installed (the distill extra).
        RuntimeError: If TINKER_API_KEY is missing from the environment.
        TinkerDeadlineError: If construction exceeds the connect deadline
            (nothing is cached, so the next call rebuilds).
    """
    try:
        import tinker
    except ImportError as exc:
        raise ImportError(_MISSING_TINKER_EXTRA) from exc
    if not os.environ.get(TINKER_API_KEY_ENV):
        raise RuntimeError(
            f"{TINKER_API_KEY_ENV} is not set in the environment; set it to "
            "your Tinker API key to use the tinker provider"
        )
    global _shared_service
    with _shared_lock:
        if _shared_service is None:
            _shared_service = call_with_deadline("connect", tinker.ServiceClient)
        return _shared_service


def rebuild_shared_service_client() -> None:
    """Discard the process-wide service client so the next call builds a fresh session.

    Last resort for a wedge that survives sampling-client replacement. The SDK can drop a
    long-lived session into an internal JWT-refresh/heartbeat retry loop where calls neither
    return nor raise; our deadlines convert that into repeated expiries, and `_drop_wedged_sampler`
    rebuilds the SamplingClient -- but from THIS client. If the wedge lives here, every
    replacement inherits it and the run expires forever while a brand-new process talks to the
    same service in 1.7s. Measured live: two runs sat at 96 and 9 consecutive expiries with zero
    rollouts for 30+ minutes, while an independent probe created a client in 0.7s and sampled a
    4,000-token prompt in 1.7s.

    Deliberately NOT called on every expiry. Each `ServiceClient` pins one live server-side
    session for the life of the process (its heartbeat strongly references it, so it is never
    collected), and the service rejects new sessions at roughly 240 cumulative -- so rebuilding
    freely trades a wedge for a capacity wall. Callers must require several CONSECUTIVE expiries
    first. The cached sampling clients are dropped with it: they were built from the discarded
    client and would keep its wedge.
    """
    global _shared_service
    with _shared_lock:
        if _shared_service is None and not _shared_samplers:
            return
        logger.warning(
            "rebuilding the process-wide tinker service client after repeated deadline "
            "expiries; %d cached sampling client(s) are dropped with it. Each rebuild pins "
            "another server-side session for this process, so this must stay rare",
            len(_shared_samplers),
        )
        _shared_service = None
        _shared_samplers.clear()


def shared_sampling_client(model: str) -> tinker.SamplingClient:
    """The process-wide sampling client for one model, from the shared cache.

    Keyed by the exact model string: a `tinker://` sampler-weights path or a
    base model name. The SDK documents `SamplingClient` as thread-safe, so a
    single client per model serves every concurrent trial; per-episode state
    (`TokenRecorder`, incremental prompt state, renderer) stays on each
    provider instance. Entries live for the rest of the process (the SDK pins
    every constructed client regardless); a wedged entry is replaced through
    `evict_shared_sampling_client`.

    Args:
        model: The exact model string to sample from.

    Returns:
        The cached (or newly built and cached) sampling client.

    Raises:
        ImportError: If the tinker SDK is not installed (the distill extra).
        RuntimeError: If TINKER_API_KEY is missing from the environment.
        TinkerDeadlineError: If client construction exceeds the connect
            deadline (nothing is cached, so the next call rebuilds).
    """
    service = shared_service_client()
    with _shared_lock:
        cached = _shared_samplers.get(model)
        if cached is not None:
            return cached

        def build() -> tinker.SamplingClient:
            if model.startswith("tinker://"):
                return service.create_sampling_client(model_path=model)
            return service.create_sampling_client(base_model=model)

        client = call_with_deadline("connect", build)
        _shared_samplers[model] = client
        return client


_served_context_windows: dict[str, int] | None = None
"""Cached `model_name -> max_context_length` from the service's own capabilities."""


def served_context_window(model: str) -> int | None:
    """The context window the Tinker service actually serves `model` with, or None.

    Read from the service's `get_server_capabilities()` (`SupportedModel.max_context_length`), not
    from a table in this repo: the context tier is part of the served model identity (the catalog
    names carry a `:262144`-style suffix) and hardcoding any number here is exactly how a 128k
    assumption against a 64k deployment produced 118 context-overflow 400s in one run. The lookup
    tolerates the model string carrying a checkpoint/tier suffix the catalog name does not.

    Args:
        model: The model string the sampler was built with.

    Returns:
        The served window in tokens, or None when the service does not list the model (or the
        capability probe is unavailable).

    Raises:
        ImportError: If the tinker SDK is not installed (the distill extra).
        RuntimeError: If TINKER_API_KEY is missing from the environment.
    """
    global _served_context_windows
    with _shared_lock:
        cached = _served_context_windows
    if cached is None:
        service = shared_service_client()
        capabilities = call_with_deadline("connect", service.get_server_capabilities)
        cached = {
            supported.model_name: supported.max_context_length
            for supported in capabilities.supported_models
            if supported.model_name and supported.max_context_length
        }
        with _shared_lock:
            _served_context_windows = cached
    direct = cached.get(model)
    if direct is not None:
        return direct
    # A sampler path (tinker://...) or a name carrying an extra suffix still identifies one served
    # model; match on the longest listed name the model string starts with.
    prefixes = [name for name in cached if model.startswith(name)]
    if not prefixes:
        return None
    return cached[max(prefixes, key=len)]


def evict_shared_sampling_client(model: str, client: tinker.SamplingClient) -> None:
    """Drop one model's cached sampling client if it is still `client`.

    The wedged-session heal path: a client that blew a deadline is wedged for
    EVERY user sharing it, so callers evict the cache entry (not merely their
    own reference) and the next `shared_sampling_client(model)` builds a
    fresh session. The identity check keeps a late eviction from discarding a
    replacement another caller already rebuilt.

    Args:
        model: The cache key the wedged client was built under.
        client: The wedged client itself, identity-compared to the entry.
    """
    with _shared_lock:
        if _shared_samplers.get(model) is client:
            del _shared_samplers[model]
            logger.info(
                "evicted the shared tinker sampling client for %s; the next "
                "user builds a fresh session",
                model,
            )


class TokenSpan(BaseModel):
    """The exact tokens one sampling call consumed and produced.

    This is the tokens-in-tokens-out ground truth for one completion: training
    data is assembled from these ids verbatim, never from re-encoded text.

    The optional `delta_start`, `delta_messages`, and `tools` fields carry the
    CANONICAL conversation beside the tokens, so a cross-tokenizer teacher can
    re-render the same conversation with its own chat template instead of
    guessing at the student template's framing. They are optional for
    backward compatibility: sinks recorded before they existed still load, and
    `wmh.distill.tokens.reconstruct_conversation` reports honestly (None) that
    such a sink cannot be replayed.
    """

    call_index: int
    """0-based index of the successful completion within one episode."""

    prompt_token_ids: list[int]
    sampled_token_ids: list[int]
    sampled_logprobs: list[float]
    """Sampler-assigned logprob for each sampled token, aligned one to one."""

    delta_start: int | None = None
    """Index into this call's full message list at which `delta_messages`
    begins, i.e. the exact boundary the prompt tokens were built at:

    - 0: the episode's FIRST call, whose prompt renders the whole history, so
      the delta is that whole history.
    - `len(previous call's messages) + 1`: the ordinary incremental case; the
      prompt is (previous prompt + raw sampled ids + rendered suffix of the
      delta), skipping the caller's echo of the previous assistant turn.

    None when this call's prompt is not a clean suffix extension of the
    previous one (a reused prompt after a discarded turn, or the
    full-re-render fallback of a genuine history edit), where a concatenated
    delta would describe a different conversation than the one sampled, and on
    sinks recorded before this field existed."""

    delta_messages: list[ChatMessage] | None = None
    """The canonical messages this call ADDED relative to the previous call
    (`messages[delta_start:]`), never the whole history: per-call full history
    would be quadratic in text and real TB2 tool results are large (a single
    observed prompt was 55,222 tokens). Concatenating the deltas mirrors the
    token-side prefix merge exactly, because `delta_start` IS the boundary the
    prompt suffix was rendered at. None exactly when `delta_start` is None (see
    there), which is distinct from `[]`, a genuinely empty delta."""

    tools: list[ChatTool] = Field(default_factory=list)
    """The tool schemas this call rendered with; empty when none were rendered
    (including `tool_choice="none"`). Recorded on EVERY span, not just call 0:
    schemas are small and normally constant, and a mid-episode tool change
    (which fragments the prompt anyway) then stays visible per call rather
    than being silently attributed to the first one."""

    @model_validator(mode="after")
    def _check_alignment(self) -> TokenSpan:
        if len(self.sampled_logprobs) != len(self.sampled_token_ids):
            raise ValueError(
                f"sampled_logprobs length {len(self.sampled_logprobs)} does not match "
                f"sampled_token_ids length {len(self.sampled_token_ids)}"
            )
        if (self.delta_start is None) != (self.delta_messages is None):
            raise ValueError(
                "delta_start and delta_messages must be recorded together (the delta "
                "boundary is meaningless without its messages, and vice versa); leave "
                "both unset only for spans recorded before the conversation fields existed"
            )
        if self.delta_start is not None and self.delta_start < 0:
            raise ValueError(
                f"delta_start {self.delta_start} is negative; it is an index into the "
                "call's message list, so 0 (a full re-render) is the minimum"
            )
        return self


class TokenRecorder:
    """Collects the `TokenSpan`s of ONE episode/trial.

    Ownership contract: one recorder per episode, driven by a single thread
    (the pi agent issues its completions sequentially). Create a fresh
    recorder (and a fresh sink file) per episode so `call_index` starts at 0.

    Args:
        jsonl_path: Optional sink; every recorded span is appended and flushed
            immediately, so a killed trial still leaves its captured spans on
            disk.
    """

    def __init__(self, jsonl_path: Path | None = None) -> None:
        self._spans: list[TokenSpan] = []
        self._jsonl_path = jsonl_path
        self._fallbacks = 0

    def __len__(self) -> int:
        return len(self._spans)

    def record(self, span: TokenSpan) -> None:
        """Append one span, writing through to the jsonl sink when configured."""
        self._spans.append(span)
        if self._jsonl_path is not None:
            with self._jsonl_path.open("a", encoding="utf-8") as sink:
                sink.write(span.model_dump_json() + "\n")
                sink.flush()

    def spans(self) -> list[TokenSpan]:
        """A snapshot copy of the spans recorded so far."""
        return list(self._spans)

    @property
    def fallback_count(self) -> int:
        """How many prompts fell back to a full re-render (each one fragments)."""
        return self._fallbacks

    def record_fallback(self) -> None:
        """Count one incremental-prompt fallback (genuine mid-episode history edit)."""
        self._fallbacks += 1


class SampledSequenceLike(Protocol):
    """The slice of a sampled sequence the provider consumes."""

    @property
    def tokens(self) -> list[int]:
        """Sampled token ids."""
        ...

    @property
    def logprobs(self) -> list[float] | None:
        """Per-token logprobs aligned with `tokens`, or None if unavailable."""
        ...


class TinkerSampler(Protocol):
    """The sampling call the provider makes, in token-id terms.

    `wmh.distill.fake_tinker.FakeSamplingClient` satisfies this directly;
    real `tinker.SamplingClient`s are adapted via `SdkSampler`.
    """

    def sample(
        self,
        prompt_token_ids: list[int],
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str] | list[int] | None = None,
    ) -> SampledSequenceLike:
        """Sample one sequence conditioned on the prompt token ids."""
        ...


@runtime_checkable
class _TokenizerSource(Protocol):
    """A sampling client that can supply the base model's tokenizer."""

    def get_tokenizer(self) -> RendererTokenizer: ...


class SdkSampler:
    """Adapts a real `tinker.SamplingClient` to the `TinkerSampler` seam.

    The provider's lazy path builds this itself; callers that already hold a
    sampling client (e.g. the distill loop refreshing student weights via
    `save_weights_and_get_sampling_client`) wrap it in this before injecting.
    """

    def __init__(self, client: tinker.SamplingClient) -> None:
        self._client = client

    @property
    def sdk_client(self) -> tinker.SamplingClient:
        """The wrapped SDK client (shared-cache eviction compares its identity)."""
        return self._client

    def sample(
        self,
        prompt_token_ids: list[int],
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str] | list[int] | None = None,
    ) -> SampledSequenceLike:
        """Run one deadline-bounded sample and return the single sampled sequence.

        Raises:
            TinkerDeadlineError: If the sample deadline expires (the session
                is likely wedged; the caller should retry with a fresh one).
        """
        import tinker

        future = self._client.sample(
            prompt=tinker.ModelInput.from_ints(prompt_token_ids),
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=max_tokens, temperature=temperature, stop=stop
            ),
        )
        return wait_with_deadline("sample", future).sequences[0]

    def get_tokenizer(self) -> RendererTokenizer:
        """The HF tokenizer for the client's base model (deadline-bounded fetch)."""
        # HF stubs type decode as `str | list[str]` depending on the input
        # shape; for the list[int] calls renderers make it is always str.
        return cast("RendererTokenizer", call_with_deadline("connect", self._client.get_tokenizer))


class _BuiltPrompt(BaseModel):
    """One built prompt plus the message boundary it was built at (internal).

    `delta_start` is the single source of truth for the delta boundary: the
    token suffix is rendered from it and the recorded `TokenSpan.delta_messages`
    is sliced at it, so the message delta and the token delta cannot disagree.
    """

    token_ids: list[int]
    """The exact prompt ids to sample from."""

    delta_start: int | None
    """Index of the first incoming message this prompt adds over the previous call.

    None when the prompt is not a clean suffix extension of the previous one (a
    reused prompt or a full-re-render fallback), so no message delta can be
    recorded for the span (see `TokenSpan.delta_messages`).
    """


class _SampledTurn(BaseModel):
    """One successful render-sample-parse round trip (internal)."""

    prompt_token_ids: list[int]
    sampled_token_ids: list[int]
    parsed: ParsedAssistantMessage


@dataclass
class _PromptState:
    """The provider's last successful call, for incremental prompt extension.

    Shares the recorder's single-episode ownership: one provider serves one
    episode sequentially, so the next call's history normally extends this
    call's message list by the assistant echo plus new tool/user messages.
    """

    messages: list[ChatMessage]
    """Snapshot of the message list the last prompt was built from."""

    tool_signature: str | None
    """Normalized digest of the tool schemas the last prompt rendered with."""

    prompt_tokens: list[int]
    """The exact prompt ids sent on the last call (incremental or full)."""

    sampled_tokens: list[int]
    """The raw sampled ids of the last call, including any end-of-turn token."""


def _tool_signature(tools: list[ChatTool] | None) -> str | None:
    """A normalized digest of tool schemas; None when no tools are rendered."""
    if not tools:
        return None
    return json.dumps([tool.model_dump(mode="json") for tool in tools], sort_keys=True)


def _first_incompatible_index(
    previous: list[ChatMessage], incoming: list[ChatMessage]
) -> int | None:
    """Where `incoming` stops being a tolerant extension of `previous`, or None.

    The shared region must match role for role and count for count. Assistant
    turns are compared by role only: the agent re-serializes the provider's
    own turns (parsed and reformatted tool calls, collapsed think framing), so
    their text never byte-matches and the provider's token history is the
    ground truth for them. System, user, and tool messages must match exactly
    by content and tool linkage. When `incoming` is longer, the first new
    message must be the assistant echo of the provider's last sampled turn.

    Returns:
        None when compatible; otherwise the index of the first message that
        breaks the extension (a genuine history edit or compaction).
    """
    if len(incoming) < len(previous):
        return len(incoming)
    for index, (prev, cur) in enumerate(zip(previous, incoming, strict=False)):
        if prev.role != cur.role:
            return index
        if prev.role == "assistant":
            continue
        if prev.content != cur.content:
            return index
        if prev.tool_call_id != cur.tool_call_id:
            return index
        if (prev.model_extra or {}).get("name") != (cur.model_extra or {}).get("name"):
            return index
    if len(incoming) > len(previous) and incoming[len(previous)].role != "assistant":
        return len(previous)
    return None


class TinkerChatProvider:
    """Serves completions from a Tinker-hosted LoRA student during distillation.

    Args:
        config: Provider config; `model_type` is the base model name and
            `model` is the `tinker://` sampler-weights path (or a base model
            name for an untrained student).
        sampling_client: Optional injected sampler (tests use the fakes in
            `wmh.distill.fake_tinker`; wrap a real `tinker.SamplingClient` in
            `SdkSampler`). When None, a real client is fetched lazily from
            the process-wide shared cache (`shared_sampling_client`, keyed by
            `config.model`; one client per model string serves every
            concurrent trial, and the SDK documents `SamplingClient` as
            thread-safe) and EVICTED from that cache after a
            `TinkerDeadlineError` so every future user rebuilds through a
            fresh session; an injected client bypasses the cache entirely and
            is never dropped.
        renderer: Optional injected rendering. When None, it is built lazily
            from the base model name and the sampling client's tokenizer.
        recorder: Optional per-episode span recorder; when present, every
            successful completion records exactly one `TokenSpan`.
        api_key: Accepted for `get_provider`'s explicit-credential channel and
            REJECTED when set. Tinker authenticates through the process-wide
            shared `ServiceClient` (`TINKER_API_KEY`), which is cached per
            process rather than per credential, so a per-entry key cannot be
            honored here. Failing loudly beats silently sampling on a
            different account than the pool entry named.
    """

    def __init__(
        self,
        config: ProviderConfig,
        *,
        sampling_client: TinkerSampler | None = None,
        renderer: ChatRendering | None = None,
        recorder: TokenRecorder | None = None,
        api_key: str | None = None,
    ) -> None:
        if api_key is not None:
            raise ValueError(
                "the tinker provider does not accept an explicit api_key: it authenticates "
                f"through the process-wide shared service client, which reads {TINKER_API_KEY_ENV} "
                "from the environment; drop api_key_env from the pool entry and export "
                f"{TINKER_API_KEY_ENV} instead"
            )
        self.config = config
        self._sampler = sampling_client
        # Only a client the provider built itself may be dropped and rebuilt
        # after a deadline expiry; an injected one cannot be reconstructed.
        self._owns_sampler = sampling_client is None
        self._consecutive_expiries = 0
        self._rendering = renderer
        self._recorder = recorder
        self._prompt_state: _PromptState | None = None

    def _base_model_name(self) -> str:
        base = self.config.model_type or self.config.model
        if base.startswith("tinker://"):
            if self.config.model_type:
                raise ValueError(
                    "config.model_type is a tinker:// weights path; set model_type to "
                    "the base model name (e.g. 'Qwen/Qwen3-8B') so the renderer and "
                    "tokenizer can be resolved (weights paths belong in config.model)"
                )
            raise ValueError(
                "config.model is a tinker:// weights path and config.model_type is "
                "unset; set model_type to the base model name (e.g. 'Qwen/Qwen3-8B') "
                "so the renderer and tokenizer can be resolved"
            )
        return base

    def context_window(self) -> int | None:
        """The served context window for this student, from the service's capabilities.

        Satisfies `wmh.providers.base.ContextWindowProvider`, which is how the pi runner calibrates
        its context guard to the real deployment instead of a hardcoded number. Resolved from
        `config.model_type` (the base/catalog name that names the context tier) and falling back to
        `config.model`; None when the service does not list either.
        """
        for candidate in (self.config.model_type, self.config.model):
            if not candidate or candidate.startswith("tinker://"):
                continue
            window = served_context_window(candidate)
            if window is not None:
                return window
        return served_context_window(self.config.model)

    def _get_sampler(self) -> TinkerSampler:
        if self._sampler is None:
            self._sampler = self._build_sdk_sampler()
        return self._sampler

    def _build_sdk_sampler(self) -> TinkerSampler:
        # Lazy: the SDK import and the API-key check happen inside the shared
        # cache path (`shared_sampling_client`), so nothing touches tinker
        # before the first completion. Sharing keeps session creation bounded
        # by distinct model strings instead of by trial count.
        return SdkSampler(shared_sampling_client(self.config.model))

    def _drop_wedged_sampler(self) -> None:
        """Forget (and evict from the shared cache) a wedged sampling client.

        A wedged session keeps timing out while a freshly built one heals
        (observed live), so dropping here makes the retry wrapper's next
        attempt rebuild through `_get_sampler`. The shared cache entry is
        evicted too, not just this provider's reference: the client is shared
        process-wide, so leaving it cached would hand the same wedged session
        to every future trial. An injected client is never dropped: the
        provider cannot rebuild what it did not build.

        After `_WEDGE_REBUILD_THRESHOLD` CONSECUTIVE expiries this escalates and
        discards the process-wide service client too. Replacing only the sampling client
        cannot heal a wedge that lives one level up, because the replacement is built from
        the same service client -- the failure mode that stalled two live runs at 96 and 9
        consecutive expiries with zero rollouts for 30+ minutes while a fresh process
        sampled the same model in 1.7s. The counter resets on any success
        (`note_healthy_call`), so only an unbroken run of failures escalates.
        """
        if self._owns_sampler and self._sampler is not None:
            self._consecutive_expiries += 1
            logger.warning(
                "dropping the tinker sampling client after a deadline expiry (%d consecutive); "
                "the next attempt builds a fresh session",
                self._consecutive_expiries,
            )
            if isinstance(self._sampler, SdkSampler):
                evict_shared_sampling_client(self.config.model, self._sampler.sdk_client)
            self._sampler = None
            if self._consecutive_expiries >= _WEDGE_REBUILD_THRESHOLD:
                rebuild_shared_service_client()
                self._consecutive_expiries = 0

    def note_healthy_call(self) -> None:
        """Reset the consecutive-expiry counter after a call that succeeded.

        Without this the counter is cumulative, and a run that expires occasionally over
        hours would eventually rebuild the service client for no reason -- each rebuild
        pinning another server-side session against a ~240 cap.
        """
        self._consecutive_expiries = 0

    def _get_rendering(self) -> ChatRendering:
        if self._rendering is None:
            base_model = self._base_model_name()
            sampler = self._get_sampler()
            if not isinstance(sampler, _TokenizerSource):
                raise RuntimeError(
                    "the injected sampling client exposes no get_tokenizer(); pass "
                    "renderer= explicitly when constructing TinkerChatProvider with "
                    "a custom sampling client"
                )
            self._rendering = build_renderer(base_model, sampler.get_tokenizer())
        return self._rendering

    def _build_prompt(
        self, messages: list[ChatMessage], tools: list[ChatTool] | None
    ) -> _BuiltPrompt:
        """Build the prompt ids, extending the episode's own token history when possible.

        When the previous call's message list is a tolerant prefix of the
        incoming one (see `_first_incompatible_index`) and the tool schemas
        are unchanged, the prompt is the previous prompt plus the raw sampled
        ids plus a rendered suffix of only the NEW messages, so it extends
        (previous prompt + previous sample) verbatim as a token prefix and the
        episode merges into one training datum. An identical-length compatible
        history (the caller discarded the last turn and re-asks) reuses the
        previous prompt unchanged. Anything else (a genuine history edit or
        compaction) falls back to a full re-render, which is counted on the
        recorder because every fallback fragments the episode's datums.

        Returns:
            The prompt ids plus the message index the suffix render started at
            (0 for the episode's first call, None when the prompt was reused or
            fully re-rendered), so the span records the message delta over
            exactly the SAME boundary as the token delta.
        """
        rendering = self._get_rendering()
        state = self._prompt_state
        if state is None:
            return _BuiltPrompt(
                token_ids=rendering.build_generation_prompt(messages, tools), delta_start=0
            )
        signature = _tool_signature(tools)
        mismatch = _first_incompatible_index(state.messages, messages)
        if mismatch is None and signature == state.tool_signature:
            if len(messages) == len(state.messages):
                return _BuiltPrompt(token_ids=list(state.prompt_tokens), delta_start=None)
            # One boundary for both deltas: messages[delta_start:] is exactly the
            # region render_suffix turns into tokens (index len(state.messages) is
            # the caller's echo of the previous sampled turn, already in tokens).
            delta_start = len(state.messages) + 1
            suffix = rendering.render_suffix(
                messages,
                delta_start,
                tools,
                previous_sampled_ids=state.sampled_tokens,
            )
            return _BuiltPrompt(
                token_ids=state.prompt_tokens + state.sampled_tokens + suffix,
                delta_start=delta_start,
            )
        if self._recorder is not None:
            self._recorder.record_fallback()
        if mismatch is not None:
            logger.info(
                "incremental prompt fallback: incoming message %d does not extend the "
                "previous call's history (genuine edit or compaction); re-rendering the "
                "full prompt, which fragments this episode's training datums",
                mismatch,
            )
        else:
            logger.info(
                "incremental prompt fallback: the tool schemas changed since the previous "
                "call; re-rendering the full prompt, which fragments this episode's "
                "training datums"
            )
        return _BuiltPrompt(
            token_ids=rendering.build_generation_prompt(messages, tools), delta_start=None
        )

    def _sample_turn(
        self,
        messages: list[ChatMessage],
        tools: list[ChatTool] | None,
        *,
        temperature: float,
        max_tokens: int,
    ) -> _SampledTurn:
        """Render, sample, parse, and (on success) record exactly one span."""
        try:
            rendering = self._get_rendering()
            built = self._build_prompt(messages, tools)
            prompt_ids = built.token_ids
            sequence = self._get_sampler().sample(
                prompt_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=rendering.stop_sequences,
            )
        except TinkerDeadlineError:
            # The session is likely wedged; drop it so the retry wrapper's
            # next attempt rebuilds fresh. No span was recorded (recording
            # happens only after the whole completion succeeds, below).
            self._drop_wedged_sampler()
            raise
        # The call returned, so whatever wedge the expiry counter was tracking has cleared.
        # Without this reset the count is cumulative and a run that expires occasionally over
        # hours would eventually rebuild the service client for no reason.
        self.note_healthy_call()
        sampled_ids = list(sequence.tokens)
        logprobs = sequence.logprobs
        if logprobs is None or len(logprobs) != len(sampled_ids):
            got = (
                "no logprobs"
                if logprobs is None
                else f"{len(logprobs)} logprobs for {len(sampled_ids)} tokens"
            )
            raise RuntimeError(
                f"tinker sampling returned {got}; per-token logprobs are required "
                "for tokens-in-tokens-out training and are never fabricated"
            )
        parsed = rendering.parse_response(sampled_ids)
        # Update the incremental state and record only after the whole completion
        # succeeded, so a failure that an outer retry wrapper re-invokes never
        # leaves a span (or stale prompt state) behind.
        self._prompt_state = _PromptState(
            messages=list(messages),
            tool_signature=_tool_signature(tools),
            prompt_tokens=prompt_ids,
            sampled_tokens=sampled_ids,
        )
        if self._recorder is not None:
            # Deep copies: the agent owns these message objects and may mutate
            # them after this call returns, while the recorder's in-memory spans
            # outlive the call (the jsonl sink is already a snapshot).
            delta_start = built.delta_start
            delta_messages = (
                None
                if delta_start is None
                else [message.model_copy(deep=True) for message in messages[delta_start:]]
            )
            self._recorder.record(
                TokenSpan(
                    call_index=len(self._recorder),
                    prompt_token_ids=prompt_ids,
                    sampled_token_ids=sampled_ids,
                    sampled_logprobs=list(logprobs),
                    delta_start=delta_start,
                    delta_messages=delta_messages,
                    tools=[tool.model_copy(deep=True) for tool in tools] if tools else [],
                )
            )
        return _SampledTurn(
            prompt_token_ids=prompt_ids, sampled_token_ids=sampled_ids, parsed=parsed
        )

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        """Serve one structured agent request from the student sampler."""
        temperature = (
            request.temperature if request.temperature is not None else _DEFAULT_CHAT_TEMPERATURE
        )
        max_tokens = request.max_completion_tokens or request.max_tokens or DEFAULT_MAX_TOKENS
        if (request.model_extra or {}).get("stop") is not None:
            logger.debug(
                "ignoring request-supplied stop sequences; the renderer's stop "
                "sequences are authoritative for token-exact sampling"
            )
        # tool_choice is honored where token sampling can express it ("none" renders
        # without tool schemas) and rejected loudly where it cannot ("required" or a
        # named function would need constrained decoding the sampler does not offer).
        tools = request.tools
        choice = request.tool_choice
        if choice == "none":
            tools = None
        elif choice is not None and choice != "auto":
            raise ValueError(
                f"unsupported tool_choice {choice!r}: the tinker provider samples raw "
                "tokens and cannot force the student to call a tool; use 'auto', "
                "'none', or omit tool_choice"
            )
        turn = self._sample_turn(
            request.messages, tools, temperature=temperature, max_tokens=max_tokens
        )
        parsed = turn.parsed
        if parsed.tool_calls:
            finish_reason = "tool_calls"
        elif parsed.stopped:
            finish_reason = "stop"
        else:
            finish_reason = "length"
        message = ChatMessage(
            role="assistant",
            content=parsed.text or None,
            tool_calls=parsed.tool_calls or None,
        )
        # An unreadable tool call travels WITH the completion so the agent scaffold can feed the
        # parser's complaint back as an observation. Without it, a dropped call is indistinguishable
        # from prose and ends the episode as a clean-looking submission.
        choice_extra: dict[str, list[str]] = (
            {UNPARSED_TOOL_CALLS_KEY: list(parsed.unparsed_errors)}
            if parsed.unparsed_errors
            else {}
        )
        return ChatResponse(
            choices=[
                ChatChoice(index=0, message=message, finish_reason=finish_reason, **choice_extra)
            ],
            usage=ChatUsage(
                prompt_tokens=len(turn.prompt_token_ids),
                completion_tokens=len(turn.sampled_token_ids),
            ),
            model=self.config.model,
        )

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion:
        """Plain-text completion through the same render-sample-parse machinery."""
        chat_messages: list[ChatMessage] = []
        if system:
            chat_messages.append(ChatMessage(role="system", content=system))
        chat_messages.extend(ChatMessage(role=m.role, content=m.content) for m in messages)
        turn = self._sample_turn(
            chat_messages, None, temperature=temperature, max_tokens=max_tokens
        )
        return Completion(
            text=turn.parsed.text,
            usage=TokenUsage(
                input_tokens=len(turn.prompt_token_ids),
                output_tokens=len(turn.sampled_token_ids),
            ),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Tinker has no embeddings API; configure a dedicated embedder instead."""
        raise ValueError(
            "the tinker provider has no embeddings API; configure a dedicated "
            "embedder (hashing, openai, azure, or bedrock) for retrieval instead"
        )

    def verify(self) -> VerifyResult:
        """One-token sample through the real render+sample path (never recorded)."""
        return verify_via_ping(self, ping=self._ping)

    def _ping(self) -> None:
        try:
            rendering = self._get_rendering()
            prompt_ids = rendering.build_generation_prompt(
                [ChatMessage(role="user", content="ping")]
            )
            self._get_sampler().sample(
                prompt_ids, max_tokens=1, temperature=0.0, stop=rendering.stop_sequences
            )
        except TinkerDeadlineError:
            self._drop_wedged_sampler()
            raise
