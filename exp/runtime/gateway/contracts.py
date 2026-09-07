"""Immutable gateway request, target, event, failure, and compatibility contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from exp.common.core.artifacts import ArtifactId, ContractModel, JsonObject, Sha256
from exp.common.models.content import (
    AudioContentPart,
    DocumentContentPart,
    ImageContentPart,
    MediaHandle,
    MessageContentPart,
    VideoContentPart,
    require_attachment_ceilings,
)
from exp.common.models.gateway_catalog import (
    DeploymentId,
    ExactModelId,
    ExactModelPoolId,
    FailoverMode,
)
from exp.common.models.model import MAXIMUM_TOOL_CALL_ID_CHARACTERS, ReasoningEffort, ToolCall
from exp.runtime.gateway.reasoning_blocks import (
    EncryptedReasoningBlock as EncryptedReasoningBlock,
)
from exp.runtime.gateway.reasoning_blocks import (
    ExposedReasoningContentBlock as ExposedReasoningContentBlock,
)
from exp.runtime.gateway.reasoning_blocks import (
    OpaqueReasoningContentBlock as OpaqueReasoningContentBlock,
)
from exp.runtime.gateway.reasoning_blocks import (
    ProviderReasoningBlock as ProviderReasoningBlock,
)
from exp.runtime.gateway.reasoning_blocks import (
    RedactedThinkingBlock as RedactedThinkingBlock,
)
from exp.runtime.gateway.reasoning_blocks import (
    SealedReasoningContentBlock as SealedReasoningContentBlock,
)
from exp.runtime.gateway.reasoning_blocks import (
    ThinkingBlock as ThinkingBlock,
)
from exp.runtime.gateway.stream_contracts import (
    GatewayEvent as GatewayEvent,
)
from exp.runtime.gateway.stream_contracts import (
    GatewayEventKind as GatewayEventKind,
)
from exp.runtime.gateway.stream_contracts import (
    GatewayFailure as GatewayFailure,
)
from exp.runtime.gateway.stream_contracts import (
    GatewayFailureClass as GatewayFailureClass,
)
from exp.runtime.gateway.stream_contracts import (
    GatewayUsage as GatewayUsage,
)

GatewayAliasName = ArtifactId
OrganizationId = ArtifactId
IdentityId = ArtifactId
VirtualKeyId = ArtifactId
GatewayAliasRevisionId = ArtifactId
ProjectRef = ArtifactId
ActivationRef = ArtifactId
RequestId = ArtifactId
AttemptId = ArtifactId


class DirectTarget(ContractModel):
    """An alias target that resolves directly to one exact-model pool."""

    kind: Literal["direct"] = "direct"
    pool_id: ExactModelPoolId


class ProjectTarget(ContractModel):
    """An alias target that selects through one immutable EXP router activation."""

    kind: Literal["project"] = "project"
    project_ref: ProjectRef
    activation_ref: ActivationRef
    catalog_sha256: Sha256


GatewayTarget = Annotated[DirectTarget | ProjectTarget, Field(discriminator="kind")]


class GatewayApiSurface(StrEnum):
    """Public endpoint family used by one canonical request."""

    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"
    MESSAGES = "messages"
    EMBEDDINGS = "embeddings"
    IMAGES = "images"


class GatewayToolDefinition(ContractModel):
    """One caller-defined function tool with its exact JSON Schema declaration.

    The description bound is deliberately generous: both providers accept
    40k-character tool descriptions live (verified 2026-08-30), and real
    Claude Code toolsets exceeded the earlier 8k bound. The request-body
    size cap remains the effective total limit.
    """

    name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=65_536)
    parameters: JsonObject
    strict: bool = False
    cache_control: JsonObject | None = Field(default=None, exclude=True)
    """Validated caller prompt-caching hint attached to this tool definition,
    forwarded onto the native Anthropic tool block and dropped with
    disclosure on other wires. Like ``ToolCall.cache_control``, a cache hint
    changes cost, not semantics: it joins neither serialization nor replay
    identity."""
    eager_input_streaming: bool | None = Field(default=None, exclude=True)
    """Verbatim Anthropic fine-grained tool-input streaming selector, sent
    conditionally by Claude Code and accepted bare by the provider (verified
    live 2026-08-30, no beta header). Excluded from serialization (tool
    digests predate it); a present value joins replay identity through
    :func:`canonical_request_sha256`, like every carrier below."""
    defer_loading: bool | None = Field(default=None, exclude=True)
    """Verbatim Anthropic tool-search deferred-loading selector; the provider
    owns the cross-tool validity rules (verified live 2026-08-30: ``false``
    is a no-op and an all-deferred toolset is the provider's own 400)."""
    allowed_callers: tuple[str, ...] | None = Field(default=None, exclude=True)
    """Verbatim Anthropic programmatic-tool-calling caller allowlist,
    accepted bare by the provider even without a companion server tool
    (verified live 2026-08-30), which stays the combination authority."""
    input_examples: tuple[JsonObject, ...] | None = Field(default=None, exclude=True)
    """Verbatim Anthropic example tool inputs.

    Accepted bare by the provider (verified live 2026-08-30). Examples add
    provider-visible prompt content, so a present value is excluded from
    serialization and joins replay identity through
    :func:`canonical_request_sha256`; reservation counts its bytes with the
    rest of the replay envelope.
    """

    def has_anthropic_tool_carriers(self) -> bool:
        """Whether any Anthropic-native tool carrier is present on this tool."""
        return (
            self.eager_input_streaming is not None
            or self.defer_loading is not None
            or self.allowed_callers is not None
            or self.input_examples is not None
        )


class StructuredTextFormat(ContractModel):
    """A strict structured-text output schema requested by the caller."""

    name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=65_536)
    json_schema: JsonObject
    strict: bool = True


class GatewayProviderNativeTool(ContractModel):
    """One verbatim non-function OpenAI Responses tool declaration.

    Codex ships ``custom`` (freeform grammar), ``namespace`` (nested tool tree),
    ``web_search``, and ``tool_search`` declarations whose shapes exist on no
    other wire; each is validated shallowly at decode and re-emitted byte-for-byte
    on native Responses rungs only, with the provider owning the declaration's
    internal shape (each type captured live from Codex 0.151.0 and accepted with
    a plain API key, 2026-09-01). ``index`` is the declaration's position in the
    caller's ``tools`` array so re-emission preserves the caller's interleaving.
    """

    index: int = Field(ge=0)
    tool: JsonObject


class GatewayNamedToolChoice(ContractModel):
    """A request to require one named caller-defined function."""

    name: str = Field(min_length=1, max_length=256)


class GatewayMessage(ContractModel):
    """One canonical gateway message preserving developer and tool-call identity."""

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | None = None
    tool_call_id: str | None = Field(
        default=None, min_length=1, max_length=MAXIMUM_TOOL_CALL_ID_CHARACTERS
    )
    tool_calls: tuple[ToolCall, ...] = ()
    tool_is_error: bool = Field(default=False, exclude=True)
    """Whether this tool result reports a failed tool invocation.

    Only the Anthropic Messages surface can express it (``tool_result.is_error``),
    and only the Anthropic upstream dialect can emit it back, so route
    admission requires every waterfall rung to use that dialect. OpenAI-family
    wires cannot represent the flag and are rejected instead of dropping it. Like
    ``ToolCall.raw_arguments``, the field is deliberately excluded from model
    serialization so request digests, replay identity, and immutable
    artifacts are unaffected by it.
    """
    provider_specific_fields: JsonObject | None = Field(default=None, exclude=True)
    """LiteLLM's per-message ``provider_specific_fields`` bookkeeping, when echoed.

    Accepted so a client that replays LiteLLM message dumps verbatim keeps
    working; no provider wire takes the object, so admission always drops it
    with a ``messages.provider_specific_fields`` disclosure. Excluded from
    serialization like the other carried-but-never-forwarded message fields.
    """
    provider_reasoning: tuple[ProviderReasoningBlock, ...] = Field(default=(), exclude=True)
    """Ordered opaque provider-reasoning blocks carried on assistant turns.

    Thinking and redacted-thinking blocks exist only on the Anthropic wire;
    encrypted reasoning items exist only on the OpenAI Responses wire. Route
    admission therefore requires every waterfall rung to speak the one
    dialect that can replay them, mirroring ``tool_is_error``. Like that
    flag, the carrier is excluded from model serialization so immutable
    artifacts and carrier-free request digests are unperturbed; requests that
    do carry it join replay identity through
    :func:`canonical_request_sha256`, so a caller operation key reused with
    different reasoning is a rejected conflict, never a silent replay.
    """
    provider_item_id: str | None = Field(default=None, min_length=1, max_length=256, exclude=True)
    provider_output_index: int | None = Field(default=None, ge=0, exclude=True)
    provider_status: Literal["in_progress", "completed", "incomplete"] | None = Field(
        default=None,
        exclude=True,
    )
    provider_phase: Literal["commentary", "final_answer"] | None = Field(
        default=None,
        exclude=True,
    )
    """OpenAI Responses assistant-message phase retained for exact replay."""
    provider_tool_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        exclude=True,
    )
    provider_tool_namespace: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        exclude=True,
    )
    """Tool-result attribution replayed on a Responses ``function_call_output``.

    Codex serializes an optional ``name`` and ``namespace`` on the outputs of
    namespaced tool calls; both re-emit verbatim on the rebuilt item
    (mirroring ``ToolCall.provider_namespace`` on the call side) and are
    excluded from serialization like the other replay carriers, joining
    replay identity explicitly through :func:`canonical_request_sha256`.
    ``provider_tool_name`` also carries the Chat surface's legacy
    ``role: "tool"`` ``name`` (the old ``role: "function"`` attribution many
    agent frameworks still send; the provider serves it, probed live
    2026-09-05), re-emitted on both OpenAI wires and dropped with disclosure
    elsewhere.
    """
    provider_tool_caller: JsonObject | None = Field(default=None, exclude=True)
    """Opaque SDK 3.0 ``caller`` attribution on a ``function_call_output``.

    Programmatic tool calling attributes the result to the program that
    invoked the call; the object's internal shape is an evolving provider
    surface, so it is validated only as an object and re-emitted verbatim on
    the rebuilt item (mirroring ``ToolCall.provider_caller`` on the call
    side). Excluded from serialization like the other replay carriers,
    joining replay identity explicitly through
    :func:`canonical_request_sha256`.
    """
    provider_native_item: JsonObject | None = Field(default=None, exclude=True)
    """One verbatim OpenAI Responses input item the gateway carries opaquely.

    Codex ships tool definitions and freeform tool history as native input
    items (``additional_tools``, ``custom_tool_call``,
    ``custom_tool_call_output``), and hosted-tool turns echo their
    provider-executed items (``web_search_call``, ``mcp_call``,
    ``code_interpreter_call``, their outputs, ...); every such shape exists
    on no other wire, so the item is validated shallowly at decode and
    re-emitted byte-for-byte at its position on native Responses rungs only.
    A message carrying it carries nothing else. Excluded from serialization
    like the other carriers so item-free digests are unperturbed; a present
    item joins replay identity through :func:`canonical_request_sha256`.
    """
    provider_anthropic_blocks: tuple[JsonObject, ...] | None = Field(default=None, exclude=True)
    """The caller's assistant content blocks in their ORIGINAL order, when a
    thinking block is among them.

    The flattened fields (``content``, ``tool_calls``, ``provider_reasoning``)
    lose the order of blocks within one assistant turn; the Anthropic wire
    re-emits them as thinking, then text, then tool_use. With interleaved
    thinking a turn is [thinking, tool_use, thinking, text, tool_use ...], and
    Anthropic verifies the LATEST assistant message byte-for-byte against the
    signatures it issued: a reordered turn is refused as "thinking or
    redacted_thinking blocks in the latest assistant message cannot be
    modified" (134 requests / 48h on one Messages-surface client,
    2026-09-07). The Anthropic wire replays these verbatim when they are
    present and the flattened reasoning was not narrowed; every other wire
    keeps reading the flattened fields. Excluded from serialization like the
    other carriers.
    """
    provider_anthropic_block: JsonObject | None = Field(default=None, exclude=True)
    """One verbatim Anthropic content block the gateway carries opaquely.

    Server tools return ``server_tool_use`` and ``web_search_tool_result``
    blocks, plus citation-bearing ``text`` blocks (citations exist only as
    server-tool output), whose shapes exist on no other wire; a caller
    echoing them in history gets each carried shallowly at its position and
    re-emitted byte-for-byte on native Anthropic rungs only (route admission
    mirrors ``provider_native_item``). Decode splits the assistant turn at
    block boundaries so re-emission preserves the exact block order. A
    message carrying it carries nothing else. Excluded from serialization
    like the other carriers so block-free digests are unperturbed; a present
    block joins replay identity through :func:`canonical_request_sha256`.
    """
    provider_text_blocks: tuple[JsonObject, ...] = Field(default=(), exclude=True)
    """This message's verbatim Anthropic text blocks when one carries a
    prompt-cache marker.

    Claude Code marks system blocks and the last text block of recent user
    turns (captured live 2026-09-01); flattening them to one plain string
    strips every marker, so nothing the caller sends is ever cacheable and
    long sessions bill full input each turn (measured ~10x). When present,
    the blocks' concatenated text equals ``content`` exactly and Anthropic
    rungs re-emit them verbatim; other wires keep the flattened string and
    disclose the dropped markers. A cache hint changes cost, not semantics,
    so like the other cache carriers this joins neither serialization nor
    replay identity.
    """
    content_parts: tuple[MessageContentPart, ...] = ()
    """Ordered caller content parts for a message that carries attachments.

    Empty on every text-only message, so a text-only request serializes and
    digests exactly as before attachments existed. When present, the text parts
    concatenate to ``content`` byte-for-byte and at least one attachment (image,
    video, audio, or document) is included, so a route that cannot carry it is
    rejected at admission instead of silently serving the text alone. Attachments
    change what the model sees, so the field is serialized and joins request identity.
    """
    cache_control: JsonObject | None = Field(default=None, exclude=True)
    """Validated caller prompt-caching marker on this tool-result message.

    Claude Code marks the last block of recent user turns, which in an agent
    loop is usually a ``tool_result``; the split tool message carries the
    marker onto the re-emitted block on Anthropic rungs. Cost, not
    semantics: never in digests or replay identity.
    """

    @model_validator(mode="after")
    def _require_role_coherence(self) -> GatewayMessage:
        """Reject payload fields that do not belong to the selected message role.

        Returns:
            The validated canonical message.

        Raises:
            ValueError: Content, tool linkage, or assistant calls are incoherent.
        """
        if self.provider_native_item is not None:
            if (
                self.content is not None
                or self.tool_calls
                or self.provider_reasoning
                or self.provider_item_id is not None
                or self.tool_call_id is not None
                or self.tool_is_error
            ):
                raise ValueError("a native provider item carries the whole message")
            return self
        if self.provider_anthropic_block is not None:
            if (
                self.content is not None
                or self.tool_calls
                or self.provider_reasoning
                or self.provider_item_id is not None
                or self.tool_call_id is not None
                or self.tool_is_error
            ):
                raise ValueError("a native Anthropic block carries the whole message")
            return self
        if (
            self.content is None
            and not self.tool_calls
            and not self.provider_reasoning
            and self.provider_item_id is None
        ):
            raise ValueError("gateway messages need content, tool calls, or reasoning blocks")
        if self.role != "assistant" and self.tool_calls:
            raise ValueError("tool_calls are valid only for assistant messages")
        if self.role != "assistant" and self.provider_reasoning:
            raise ValueError("provider reasoning blocks are valid only for assistant messages")
        if self.role != "assistant" and (
            self.provider_item_id is not None
            or self.provider_output_index is not None
            or self.provider_status is not None
            or self.provider_phase is not None
        ):
            raise ValueError("provider output identity is valid only for assistant messages")
        if (self.provider_item_id is None) != (self.provider_output_index is None):
            raise ValueError("provider item ID and output index must be retained together")
        if self.provider_status is not None and self.provider_item_id is None:
            raise ValueError("provider output status requires retained item identity")
        if self.provider_phase is not None and self.provider_item_id is None:
            raise ValueError("provider output phase requires retained item identity")
        call_ids = tuple(call.call_id for call in self.tool_calls)
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("assistant tool call IDs must be unique")
        if self.role == "tool" and self.tool_call_id is None:
            raise ValueError("tool messages require tool_call_id")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError("tool_call_id is valid only for tool messages")
        if self.role != "tool" and (
            self.provider_tool_name is not None
            or self.provider_tool_namespace is not None
            or self.provider_tool_caller is not None
        ):
            raise ValueError("tool-result attribution is valid only for tool messages")
        if self.role != "tool" and self.tool_is_error:
            raise ValueError("tool_is_error is valid only for tool messages")
        if self.cache_control is not None and self.role != "tool":
            raise ValueError("message cache_control is valid only for tool messages")
        if self.provider_text_blocks:
            if self.role == "tool":
                raise ValueError("text blocks are not valid for tool messages")
            # The carrier never changes semantics: its text must flatten to
            # this message's canonical content (message runs join adjacent
            # parts directly; system blocks join with one blank line).
            texts = [str(block.get("text", "")) for block in self.provider_text_blocks]
            if (self.content or "") not in ("".join(texts), "\n\n".join(texts)):
                raise ValueError("provider text blocks must flatten to the message content")
        if self.content_parts:
            # Tool messages carry attachments too: Anthropic tool_result blocks
            # accept image sub-blocks (tool screenshots), and the block is baked
            # into caller history, so the canonical model must be able to hold it.
            if self.role not in ("user", "tool"):
                raise ValueError("content parts are valid only for user and tool messages")
            if all(part.kind == "text" for part in self.content_parts):
                raise ValueError("content parts are retained only for multimodal messages")
            if self.role == "tool" and any(
                part.kind not in ("text", "image") for part in self.content_parts
            ):
                raise ValueError("tool messages carry only text and image parts")
            texts = [part.text for part in self.content_parts if part.kind == "text"]
            if (self.content or "") != "".join(texts):
                raise ValueError("content parts must flatten to the message content")
        return self

    @property
    def images(self) -> tuple[ImageContentPart, ...]:
        """Return this message's retained image parts in caller order."""
        return tuple(part for part in self.content_parts if part.kind == "image")

    @property
    def videos(self) -> tuple[VideoContentPart, ...]:
        """Return this message's retained video parts in caller order."""
        return tuple(part for part in self.content_parts if part.kind == "video")

    @property
    def documents(self) -> tuple[DocumentContentPart, ...]:
        """Return this message's retained document parts in caller order."""
        return tuple(part for part in self.content_parts if part.kind == "document")

    def folded_tool_error_content(self) -> str:
        """Return this tool result's text with ``tool_is_error`` folded in.

        Only the Anthropic wire has a native ``tool_result.is_error`` field;
        every other wire re-states the flag in the one channel it has (the
        result text, prefixed with :data:`TOOL_ERROR_TEXT_PREFIX`) so the
        model still learns the invocation failed. The fold derives from the
        canonical flag on each request, never from previously folded text, so
        a replayed history can never accumulate prefixes.
        """
        content = self.content or ""
        if self.tool_is_error:
            return f"{TOOL_ERROR_TEXT_PREFIX}{content}"
        return content


TOOL_ERROR_TEXT_PREFIX = "[tool error] "
"""Prefix folding Anthropic's ``tool_result.is_error`` into plain result text
on wires without a native error flag (see
:meth:`GatewayMessage.folded_tool_error_content`)."""


class GatewayRequest(ContractModel):
    """Lossless canonical request shared by protocol and provider implementations."""

    surface: GatewayApiSurface
    messages: tuple[GatewayMessage, ...] = Field(min_length=1)
    tools: tuple[GatewayToolDefinition, ...] = ()
    tool_choice: Literal["auto", "none", "required"] | GatewayNamedToolChoice | None = None
    parallel_tool_calls: bool | None = None
    structured_text: StructuredTextFormat | None = None
    maximum_output_tokens: int | None = Field(default=None, gt=0)
    maximum_output_tokens_parameter: (
        Literal["max_tokens", "max_completion_tokens", "max_output_tokens"] | None
    ) = Field(default=None, exclude=True)
    """Exact caller field normalized into ``maximum_output_tokens``."""
    stop: tuple[str, ...] = ()
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    logprobs: bool | None = None
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    reasoning_effort: ReasoningEffort | None = None
    # Level-less enable-thinking; the route seam resolves the concrete effort.
    thinking_default_enable: bool = False
    reasoning_summary: Literal["auto", "concise", "detailed"] | None = None
    reasoning_summary_parameters: tuple[
        Literal["reasoning.generate_summary", "reasoning.summary"], ...
    ] = Field(default=(), exclude=True)
    """Exact caller selector paths normalized into ``reasoning_summary``."""
    provider_thinking_config: JsonObject | None = Field(default=None, exclude=True)
    """Verbatim caller ``thinking`` configuration from the Messages surface.

    The object is opaque to the gateway: it is validated against the closed
    wire profile at decode time and then forwarded byte-for-byte to the
    Anthropic upstream, overriding the catalog's adaptive default. Excluded
    from serialization like the other Anthropic-only carriers so
    config-free digests are unperturbed; a present config joins replay
    identity through :func:`canonical_request_sha256`.
    """
    context_management: JsonObject | None = Field(default=None, exclude=True)
    """Verbatim caller ``context_management`` from the Messages surface.

    Anthropic's native context-editing configuration (Claude Code sends it
    by default). The object is deliberately validated only as an object and
    forwarded byte-for-byte with the required beta header on Anthropic
    rungs: the shape is an evolving provider beta, and a closed model here
    would recreate the reject-what-real-clients-send incident class.
    Excluded from serialization like the other Anthropic-only carriers so
    config-free digests are unperturbed; a present value joins replay
    identity through :func:`canonical_request_sha256`.
    """
    diagnostics: JsonObject | None = Field(default=None, exclude=True)
    """Verbatim caller ``diagnostics`` from the Messages surface.

    Anthropic's diagnostics-correlation object (Claude Code sends
    ``{"previous_message_id": ...}`` conditionally). Validated only as an
    object and forwarded byte-for-byte with the required beta header on
    Anthropic rungs, dropped with disclosure elsewhere; the shape is an
    evolving provider beta, so validation stays shallow. Excluded from
    serialization; a present value joins replay identity through
    :func:`canonical_request_sha256`.
    """
    speed: str | None = Field(default=None, max_length=64, exclude=True)
    """Verbatim caller ``speed`` selector from the Messages surface.

    Anthropic's fast-mode selector (Claude Code sends ``"fast"``; accepted
    live behind its beta header, 2026-08-30). Bounded but deliberately not
    enumerated: the value set is an evolving provider surface. Forwarded
    with the required beta header on Anthropic rungs and dropped with
    disclosure elsewhere. Fast-mode output is provider-priced at a premium,
    so a present value joins replay identity through
    :func:`canonical_request_sha256`.
    """
    provider_cache_control: JsonObject | None = Field(default=None, exclude=True)
    """Verbatim caller top-level ``cache_control`` from the Messages surface.

    Anthropic's automatic prompt-caching marker for the last cacheable
    block (accepted bare, verified live 2026-08-30). Forwarded byte-for-byte
    on Anthropic rungs and dropped with disclosure elsewhere. Like the
    tool-call cache hint, it changes cost, not semantics, so it deliberately
    joins NEITHER serialization NOR replay identity: two requests differing
    only here are the same request.
    """
    inference_geo: str | None = Field(default=None, max_length=64, exclude=True)
    """Verbatim caller ``inference_geo`` selector from the Messages surface.

    Anthropic's inference-region selector (accepted bare, verified live
    2026-08-30). Bounded but deliberately not enumerated: the region set is
    an evolving provider surface. Forwarded verbatim on Anthropic rungs and
    dropped with disclosure elsewhere. Where inference runs is a
    caller-visible processing commitment, so a present value joins replay
    identity through :func:`canonical_request_sha256`.
    """
    provider_beta_tokens: tuple[str, ...] = Field(default=(), exclude=True)
    """Allowlisted caller ``anthropic-beta`` tokens from the Messages surface.

    Only tokens on the decoder's explicit forward allowlist appear here (a
    caller header is operator-trust surface and is never blind-forwarded);
    the rest are dropped at decode with an ``anthropic-beta.<token>``
    disclosure. Forwarded tokens merge with the gateway's own per-field
    injections on Anthropic rungs and are dropped with disclosure
    elsewhere. Tokens change provider behavior and pricing (the 1M context
    window rides one), so present tokens join replay identity through
    :func:`canonical_request_sha256`.
    """
    response_store: bool | None = None
    """Caller ``store`` selector from the Responses surface.

    ``False`` skips gateway-side continuation retention for the produced
    response; ``True`` and absent keep the default retention behavior.
    """
    include_encrypted_reasoning: bool = False
    """Whether the caller asked for ``include=["reasoning.encrypted_content"]``."""
    reasoning_context: Literal["auto", "current_turn", "all_turns"] | None = Field(
        default=None, exclude=True
    )
    """Caller ``reasoning.context`` selector from the Responses surface.

    Controls whether the model re-renders prior turns' reasoning. Forwarded
    verbatim to native Responses rungs. Excluded from model serialization so
    context-free request digests stay byte-identical to pre-field traffic; a
    present value joins replay identity through
    :func:`canonical_request_sha256`.
    """
    text_verbosity: Literal["low", "medium", "high"] | None = None
    """Caller ``text.verbosity`` selector from the Responses surface."""
    client_metadata: JsonObject | None = Field(default=None, exclude=True)
    """Verbatim caller ``client_metadata`` from the Responses surface.

    Opaque client telemetry (Codex sends it by default), forwarded verbatim
    on native Responses rungs and dropped with disclosure elsewhere. It is
    semantically inert, so unlike the other carriers it deliberately joins
    NEITHER serialization nor replay identity: two requests differing only
    here are the same request.
    """
    provider_output_config: JsonObject | None = Field(default=None, exclude=True)
    """Verbatim caller ``output_config`` from the Messages surface.

    Anthropic's native output configuration (Claude Code sends
    ``{"effort": ...}`` by default). A canonical ``effort`` value also maps
    into ``reasoning_effort`` so the shared effort machinery applies; the
    raw object forwards byte-for-byte on Anthropic rungs with caller keys
    winning over engine-derived ones. Excluded from serialization like the
    other Anthropic-only carriers; a present value joins replay identity
    through :func:`canonical_request_sha256`.
    """
    provider_native_tools: tuple[GatewayProviderNativeTool, ...] = Field(default=(), exclude=True)
    """Verbatim non-function OpenAI Responses tool declarations.

    See :class:`GatewayProviderNativeTool`. Rungs that are not native
    Responses cannot serve these, so route admission rejects by name instead
    of silently dropping a capability the caller asked for. Excluded from
    serialization like the other carriers so declaration-free digests are
    unperturbed; present entries join replay identity through
    :func:`canonical_request_sha256`.
    """
    provider_server_tools: tuple[JsonObject, ...] = Field(default=(), exclude=True)
    """Verbatim Anthropic server-tool entries from the Messages ``tools`` array.

    Server tools (``web_search_20250305``-style typed entries with no
    ``input_schema``) execute at the provider; their per-type configuration
    is an evolving provider surface, so each entry is validated shallowly at
    decode and re-emitted byte-for-byte AFTER the converted custom tools on
    native Anthropic rungs only (an accepted ordering deviation from the
    caller's interleaving). Other rungs cannot execute them, so route
    admission rejects by name instead of silently dropping a capability the
    caller asked for. Excluded from serialization like the other carriers so
    server-tool-free digests are unperturbed; present entries join replay
    identity through :func:`canonical_request_sha256`.
    """
    stream: bool = False
    include_usage: bool = False
    previous_response_id: str | None = Field(default=None, min_length=1, max_length=256)
    metadata: JsonObject = Field(default_factory=dict)
    # End-user attribution / cache hints from the OpenAI request. Captured for
    # gateway-side attribution and never forwarded verbatim. `safety_identifier`
    # is the current stable end-user identifier; `user` its deprecated predecessor;
    # `prompt_cache_key` a same-prefix cache-routing hint (never an identity) that
    # reaches the provider only as the namespaced `provider_prompt_cache_key`.
    safety_identifier: str | None = Field(default=None, max_length=1024)
    user: str | None = Field(default=None, max_length=1024)
    prompt_cache_key: str | None = Field(default=None, max_length=1024)
    provider_prompt_cache_key: str | None = Field(default=None, max_length=128, exclude=True)
    """Tenant-namespaced cache-affinity key dispatched to rungs that route by it.

    Derived at admission (``prompt_cache_affinity.provider_prompt_cache_key``)
    from the caller's ``prompt_cache_key`` or, absent one, from the
    conversation stem (the leading system/developer messages, else the first
    user turn), so every request sharing a cacheable prefix lands on the
    provider cache node that holds it while the caller's raw key never leaves
    the gateway. Excluded from serialization: it is routing state, never
    request identity.
    """
    service_tier: str | None = Field(default=None, max_length=64, exclude=True)
    """Caller provider processing tier, forwarded only on BYOK OpenAI-family
    rungs (routing and billing rules live at streaming_requests and
    capability_policy). Excluded from serialization so tier-free digests are
    unperturbed; a present value joins replay identity through
    :func:`canonical_request_sha256`: the same body at a different tier is a
    different provider price and schedule."""
    ignored_parameters: tuple[str, ...] = Field(default=(), exclude=True)
    """The caller sent ``parallel_tool_calls: false`` and at least one admitted
    rung has no such wire control: the data plane serializes those rungs' tool
    calls to one per turn instead. Disclosed through ``ignored_parameters``."""
    serialize_tool_calls: bool = Field(default=False, exclude=True)
    """Disclosed compatibility decisions applied to this request.

    A plain field path names a control accepted but intentionally omitted
    from provider dispatch; a ``path->effective`` entry (for example
    ``reasoning_effort->high`` or ``tools.strict->false``) names a disclosed
    coercion the route applied when no deployment preserved the caller's
    exact value. Coercions are never silent: each entry here also logs and
    counts in the admission metrics.
    """
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=512)
    client_request_id: str | None = Field(default=None, min_length=1, max_length=512)

    @property
    def attribution_label(self) -> str | None:
        """The end-user attribution label for this request, per the OpenAI spec.

        Prefers the current `safety_identifier`; falls back to the deprecated
        `user` field for older clients. `prompt_cache_key` is deliberately never
        used here — it is a cache-routing hint, not an end-user identity.

        Returns:
            The attribution label, or None when the caller sent neither field.
        """
        return self.safety_identifier or self.user

    @property
    def images(self) -> tuple[ImageContentPart, ...]:
        """Return every image this request carries, in message and part order."""
        return tuple(image for message in self.messages for image in message.images)

    @property
    def videos(self) -> tuple[VideoContentPart, ...]:
        """Return every video this request carries, in message and part order."""
        return tuple(video for message in self.messages for video in message.videos)

    @property
    def audios(self) -> tuple[AudioContentPart, ...]:
        """Return every audio clip this request carries, in message and part order."""
        parts = (part for message in self.messages for part in message.content_parts)
        return tuple(part for part in parts if part.kind == "audio")

    @property
    def documents(self) -> tuple[DocumentContentPart, ...]:
        """Return every document this request carries, in message and part order."""
        return tuple(part for message in self.messages for part in message.documents)

    @property
    def media_handles(self) -> tuple[MediaHandle, ...]:
        """Return every provider media handle this request carries, in caller order."""
        return tuple(
            part.handle
            for message in self.messages
            for part in message.content_parts
            if isinstance(part, (ImageContentPart, VideoContentPart, DocumentContentPart))
            and part.handle is not None
        )

    @field_validator("stop")
    @classmethod
    def _require_unique_stop_sequences(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject empty or repeated stop sequences while preserving caller order.

        Args:
            value: Requested stop strings.

        Returns:
            The unchanged validated stop sequence.

        Raises:
            ValueError: A stop is empty or repeated.
        """
        if any(not item for item in value):
            raise ValueError("stop sequences must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("stop sequences must not repeat")
        return value

    @model_validator(mode="after")
    def _require_coherent_tools(self) -> GatewayRequest:
        """Require named and required tool choices to reference available tools.

        Returns:
            The validated canonical request.

        Raises:
            ValueError: Tool definitions or tool choice are incoherent.
        """
        names = tuple(tool.name for tool in self.tools)
        if len(set(names)) != len(names):
            raise ValueError("gateway tool names must not repeat")
        # Server tools are addressable by tool_choice too; the provider owns
        # cross-set name rules for the verbatim entries.
        server_names = tuple(
            str(entry["name"]) for entry in self.provider_server_tools if "name" in entry
        )
        if (
            isinstance(self.tool_choice, GatewayNamedToolChoice)
            and self.tool_choice.name not in names
            and self.tool_choice.name not in server_names
        ):
            raise ValueError("named gateway tool choice must name a request tool")
        has_tools = bool(self.tools or self.provider_server_tools or self.provider_native_tools)
        if self.tool_choice == "required" and not has_tools:
            raise ValueError("required gateway tool choice needs at least one tool")
        if self.include_usage and not self.stream:
            raise ValueError("include_usage is valid only for streaming requests")
        parts = (part for message in self.messages for part in message.content_parts)
        require_attachment_ceilings(parts)
        if len({handle.provider for handle in self.media_handles}) > 1:
            raise ValueError(
                "media handles in one request must all name the same provider; "
                "no single route can resolve handles from two providers"
            )
        if self.reasoning_summary is not None and self.surface != GatewayApiSurface.RESPONSES:
            raise ValueError("reasoning_summary is valid only for Responses requests")
        if self.response_store is not None and self.surface != GatewayApiSurface.RESPONSES:
            raise ValueError("response_store is valid only for Responses requests")
        if self.include_encrypted_reasoning and self.surface != GatewayApiSurface.RESPONSES:
            raise ValueError("include_encrypted_reasoning is valid only for Responses requests")
        if self.reasoning_context is not None and self.surface != GatewayApiSurface.RESPONSES:
            raise ValueError("reasoning_context is valid only for Responses requests")
        if self.provider_thinking_config is not None and self.surface != GatewayApiSurface.MESSAGES:
            raise ValueError("provider_thinking_config is valid only for Messages requests")
        if self.provider_output_config is not None and self.surface != GatewayApiSurface.MESSAGES:
            raise ValueError("provider_output_config is valid only for Messages requests")
        if self.text_verbosity is not None and self.surface != GatewayApiSurface.RESPONSES:
            raise ValueError("text_verbosity is valid only for Responses requests")
        if self.client_metadata is not None and self.surface != GatewayApiSurface.RESPONSES:
            raise ValueError("client_metadata is valid only for Responses requests")
        if self.context_management is not None and self.surface != GatewayApiSurface.MESSAGES:
            raise ValueError("context_management is valid only for Messages requests")
        if self.diagnostics is not None and self.surface != GatewayApiSurface.MESSAGES:
            raise ValueError("diagnostics is valid only for Messages requests")
        if self.speed is not None and self.surface != GatewayApiSurface.MESSAGES:
            raise ValueError("speed is valid only for Messages requests")
        if self.provider_cache_control is not None and self.surface != GatewayApiSurface.MESSAGES:
            raise ValueError("provider_cache_control is valid only for Messages requests")
        if self.inference_geo is not None and self.surface != GatewayApiSurface.MESSAGES:
            raise ValueError("inference_geo is valid only for Messages requests")
        if (
            any(tool.has_anthropic_tool_carriers() for tool in self.tools)
            and self.surface != GatewayApiSurface.MESSAGES
        ):
            raise ValueError("Anthropic tool carriers are valid only for Messages requests")
        if self.provider_beta_tokens and self.surface != GatewayApiSurface.MESSAGES:
            raise ValueError("provider_beta_tokens are valid only for Messages requests")
        if self.service_tier is not None and self.surface == GatewayApiSurface.MESSAGES:
            raise ValueError("service_tier is not valid for Messages requests")
        if self.provider_server_tools and self.surface != GatewayApiSurface.MESSAGES:
            raise ValueError("provider_server_tools are valid only for Messages requests")
        if self.provider_native_tools and self.surface != GatewayApiSurface.RESPONSES:
            raise ValueError("provider_native_tools are valid only for Responses requests")
        if self.provider_native_tools:
            # Positions must tile one tools array with the converted function
            # tools exactly, so native re-emission is total by construction.
            positions = tuple(entry.index for entry in self.provider_native_tools)
            declaration_count = len(self.tools) + len(positions)
            if len(set(positions)) != len(positions) or any(
                position >= declaration_count for position in positions
            ):
                raise ValueError(
                    "provider_native_tools positions must be distinct indexes "
                    "into the caller's tools array"
                )
        if self.maximum_output_tokens_parameter is not None and self.maximum_output_tokens is None:
            raise ValueError("maximum output parameter requires a maximum output value")
        if self.reasoning_summary_parameters and self.reasoning_summary is None:
            raise ValueError("reasoning summary parameter paths require a summary selector")
        if len(set(self.reasoning_summary_parameters)) != len(self.reasoning_summary_parameters):
            raise ValueError("reasoning summary parameter paths must not repeat")
        return self


class ProjectSelection(ContractModel):
    """One frozen learned-router selection resolved before provider execution."""

    exact_model_id: ExactModelId
    selected_alias: ArtifactId
    activation_ref: ActivationRef
    fallback_reason: str | None = Field(default=None, max_length=512)


class AuthorizationSnapshot(ContractModel):
    """Immutable authority and alias target frozen before learned model selection."""

    request_id: RequestId
    organization_id: OrganizationId
    identity_id: IdentityId
    virtual_key_id: VirtualKeyId
    alias: GatewayAliasName
    alias_revision_id: GatewayAliasRevisionId
    target: GatewayTarget
    surface: GatewayApiSurface
    catalog_sha256: Sha256
    canonical_request_sha256: Sha256
    caller_operation_sha256: Sha256 | None = None
    refusal_failover: bool = False
    deadline_monotonic: float = Field(gt=0)
    app_referer: str | None = Field(default=None, max_length=2_048)
    """Caller-supplied ``HTTP-Referer`` app identity, content-free and never a credential."""
    app_title: str | None = Field(default=None, max_length=256)
    """Caller-supplied ``X-Title`` app label used only for content-free app attribution."""
    attribution_label: str | None = Field(default=None, max_length=1024)
    """End-user attribution from the OpenAI ``safety_identifier`` (or deprecated
    ``user``) request field: content-free and never a credential."""
    fair_share_weight: int = Field(default=1, ge=1, le=1_000_000)
    """Relative weight of this organization for fair-share rung admission.

    Populated by the hosted store's ``authorize_request`` from its own org data
    (paying tiers heavier than promo/free); the default 1 gives every caller an
    equal share, which is byte-identical to pre-fair-share behavior. Read only
    on rungs whose ``GatewayRungDispatchPolicy.fair_share`` is authored on.
    """


class ExecutionSnapshot(ContractModel):
    """Route-bound request plan created only after exact-model selection."""

    authorization: AuthorizationSnapshot
    exact_model_id: ExactModelId
    pool_id: ExactModelPoolId
    deployment_ids: tuple[DeploymentId, ...] = Field(min_length=1)
    # The pool's per-model failover policy, carried onto the route so the
    # per-attempt retry/failover decision can honor it. Defaults to the
    # historical maximize_availability.
    failover_mode: FailoverMode = "maximize_availability"
