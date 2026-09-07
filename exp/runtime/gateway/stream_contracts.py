"""Provider-neutral stream outcome contracts: usage, events, and failures.

Split from :mod:`exp.runtime.gateway.contracts` for the module line budget;
that module re-exports every name here, so import paths are unchanged.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject
from exp.common.models.model import MAXIMUM_TOOL_CALL_ID_CHARACTERS, ToolCall


class GatewayUsage(ContractModel):
    """Normalized token counts and invoked tool names from one provider attempt.

    Cached-input and reasoning counts are subsets of the total input and output counts when
    present. They identify differently priced portions of those totals and must not be added a
    second time by callers.

    A terminal event may carry only ``tool_names`` when the provider omits token usage. In that
    case both token totals remain unknown instead of being represented as zero.
    """

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    cache_creation_input_tokens: int | None = Field(default=None, ge=0)
    """Cache-write tokens inside the input total (Anthropic-only today),
    present only when the provider reported a nonzero count; billing keeps
    using the folded input total."""
    reasoning_tokens: int | None = Field(default=None, ge=0)
    tool_names: tuple[str, ...] = ()
    """Invoked tool names in first-use order, names only and never arguments."""

    @model_validator(mode="after")
    def _require_complete_tokens_or_tool_names(self) -> GatewayUsage:
        """Require complete token totals unless this is tool-only terminal metadata."""
        totals = (self.input_tokens, self.output_tokens)
        if (totals[0] is None) != (totals[1] is None):
            raise ValueError("input and output token counts must be reported together")
        if totals[0] is None:
            if (
                self.cached_input_tokens is not None
                or self.cache_creation_input_tokens is not None
                or self.reasoning_tokens is not None
            ):
                raise ValueError("token detail counts require input and output totals")
            if not self.tool_names:
                raise ValueError("usage requires token totals or invoked tool names")
        return self

    @property
    def has_token_counts(self) -> bool:
        """Return whether both provider token totals are known."""
        return self.input_tokens is not None and self.output_tokens is not None


class GatewayEventKind(StrEnum):
    """Provider-neutral semantic and terminal stream event categories."""

    TEXT_DELTA = "text_delta"
    REFUSAL_DELTA = "refusal_delta"
    REASONING_SUMMARY_DELTA = "reasoning_summary_delta"
    THINKING_DELTA = "thinking_delta"
    THINKING_SIGNATURE = "thinking_signature"
    REDACTED_THINKING = "redacted_thinking"
    ENCRYPTED_REASONING = "encrypted_reasoning"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_ARGUMENTS_DELTA = "tool_arguments_delta"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    USAGE = "usage"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


class GatewayEvent(ContractModel):
    """One ordered provider-neutral stream event, including raw tool fragments."""

    kind: GatewayEventKind
    sequence_number: int = Field(ge=0)
    text_delta: str | None = None
    reasoning_summary_output_index: int | None = Field(default=None, ge=0)
    reasoning_summary_index: int | None = Field(default=None, ge=0)
    reasoning_item_id: str | None = Field(default=None, min_length=1, max_length=256)
    reasoning_block_index: int | None = Field(default=None, ge=0)
    """Provider content-block (or output-item) index grouping reasoning events."""
    thinking_signature: str | None = None
    redacted_thinking_data: str | None = None
    encrypted_content: str | None = None
    tool_call_index: int | None = Field(default=None, ge=0)
    tool_call_id: str | None = Field(
        default=None, min_length=1, max_length=MAXIMUM_TOOL_CALL_ID_CHARACTERS
    )
    tool_name: str | None = Field(default=None, min_length=1, max_length=256)
    raw_arguments_delta: str | None = None
    tool_call: ToolCall | None = None
    usage: GatewayUsage | None = None
    failure: GatewayFailure | None = None

    @model_validator(mode="after")
    def _require_event_payload(self) -> GatewayEvent:
        """Require each event kind to carry its one relevant payload.

        Returns:
            The validated stream event.

        Raises:
            ValueError: The selected event kind lacks its required payload.
        """
        if self.kind in {GatewayEventKind.TEXT_DELTA, GatewayEventKind.REFUSAL_DELTA}:
            if self.text_delta is None:
                raise ValueError("text and refusal deltas require text_delta")
        elif self.kind == GatewayEventKind.REASONING_SUMMARY_DELTA:
            if (
                self.text_delta is None
                or self.reasoning_summary_output_index is None
                or self.reasoning_summary_index is None
                or self.reasoning_item_id is None
            ):
                raise ValueError("reasoning summary deltas require item, output, summary, and text")
        elif self.kind == GatewayEventKind.THINKING_DELTA:
            if self.text_delta is None or self.reasoning_block_index is None:
                raise ValueError("thinking deltas require block index and text")
        elif self.kind == GatewayEventKind.THINKING_SIGNATURE:
            if self.thinking_signature is None or self.reasoning_block_index is None:
                raise ValueError("thinking signatures require block index and signature")
        elif self.kind == GatewayEventKind.REDACTED_THINKING:
            if self.redacted_thinking_data is None or self.reasoning_block_index is None:
                raise ValueError("redacted thinking requires block index and data")
        elif self.kind == GatewayEventKind.ENCRYPTED_REASONING:
            if (
                self.encrypted_content is None
                or self.reasoning_block_index is None
                or self.reasoning_item_id is None
            ):
                raise ValueError("encrypted reasoning requires item, block index, and content")
        elif self.kind == GatewayEventKind.TOOL_CALL_STARTED:
            if self.tool_call_index is None or self.tool_call_id is None or self.tool_name is None:
                raise ValueError("tool-call start requires index, ID, and name")
        elif self.kind == GatewayEventKind.TOOL_ARGUMENTS_DELTA:
            if self.tool_call_index is None or self.raw_arguments_delta is None:
                raise ValueError("tool argument delta requires index and raw fragment")
        elif self.kind == GatewayEventKind.TOOL_CALL_COMPLETED and self.tool_call is None:
            raise ValueError("tool-call completion requires the complete tool call")
        elif self.kind == GatewayEventKind.USAGE:
            if self.usage is None or not self.usage.has_token_counts:
                raise ValueError("usage event requires complete normalized token usage")
        elif self.kind == GatewayEventKind.FAILED and self.failure is None:
            raise ValueError("failed event requires a normalized failure")
        return self


class GatewayFailureClass(StrEnum):
    """Stable failure classes shared by provider execution and the public protocol."""

    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    QUOTA_EXCEEDED = "quota_exceeded"
    THROTTLED = "throttled"
    TRANSPORT = "transport"
    TIMEOUT = "timeout"
    PROVIDER_AUTHENTICATION = "provider_authentication"
    PROVIDER_NOT_FOUND = "provider_not_found"
    # The provider ACCOUNT cannot pay for the request (trial quota exhausted,
    # billing not enabled): operator-actionable deadness that fails over in
    # every mode. Distinct from QUOTA_EXCEEDED, the CALLER's gateway credit.
    PROVIDER_QUOTA = "provider_quota"
    REFUSAL = "refusal"
    MALFORMED_RESPONSE = "malformed_response"
    PROVIDER_INTERNAL = "provider_internal"
    CANCELLED = "cancelled"
    GUARDRAIL = "guardrail"
    INTERNAL = "internal"
    # A transient control-plane condition (a rolling deploy building the
    # authorized catalog revision) that the caller should simply retry. Unlike
    # INTERNAL it is not a bug signal and does not page; unlike a provider class
    # it never opens a deployment circuit.
    UNAVAILABLE = "unavailable"


class GatewayRefusalReason(StrEnum):
    """The bounded category of a provider refusal, mirroring the native
    ``RefusalReason``.

    A refusal answer names WHICH policy declined the content as a closed
    vocabulary, so a client can branch on it and the control plane can count
    refusals by reason without parsing the free-form provider detail. The
    caller never sees the provider's own prose, only the fixed category.
    """

    CYBER_POLICY = "cyber_policy"
    CBRN = "cbrn"
    CONTENT_POLICY = "content_policy"
    RECITATION = "recitation"
    DATA_INSPECTION = "data_inspection"
    UNSPECIFIED = "unspecified"


class GatewayFailure(ContractModel):
    """Sanitized failure with retry and failover eligibility already classified."""

    failure_class: GatewayFailureClass
    safe_message: str = Field(min_length=1, max_length=2_048)
    retryable_same_deployment: bool = False
    failover_eligible: bool = False
    safe_details: JsonObject = Field(default_factory=dict)
    rejected_parameter: str | None = Field(default=None, min_length=1, max_length=128)
    """Validated provider-named parameter path; never provider prose."""
    provider_detail: str | None = Field(default=None, min_length=1, max_length=240)
    """Provider explanation of a client error, relayed only for that class."""
    retry_after_seconds: int | None = Field(default=None, ge=1)
    """The failure is the caller's own provider configuration: a rejected
    credential or exhausted account on their customer-managed (BYOK) rung. The
    class keeps its ladder semantics; the ledger files it as the caller's
    invalid request and the terminal answer is their 400."""
    customer_owned: bool = False
    """Known wait before a retry can dispatch (a throttle window's remainder).

    When present on a throttled failure, the public mapping advertises this
    value as ``Retry-After`` instead of its fixed default, so the header and
    the message never tell the caller two different waits.
    """
    refusal_reason: GatewayRefusalReason | None = None
    """The bounded refusal category, present only on a ``REFUSAL`` failure.

    Set from the provider's own code and sentence and carried on the public
    error and the settlement argument, so the caller reads the category and
    the control plane counts refusals by reason without parsing detail."""
