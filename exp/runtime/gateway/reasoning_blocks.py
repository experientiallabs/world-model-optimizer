"""Provider reasoning blocks replayed verbatim through the gateway.

Each block carries provider-issued reasoning state (Anthropic thinking
signatures, OpenAI encrypted reasoning, Fireworks reasoning carriers) that
must reach its provider byte-exact. ``ProviderReasoningBlock`` is the
discriminated union carried on canonical gateway messages.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from exp.common.core.artifacts import ContractModel, Sha256


class ThinkingBlock(ContractModel):
    """One verbatim Anthropic extended-thinking block from assistant history.

    ``signature`` is an opaque cryptographic value the provider issued with
    the block; it must round-trip byte-exact or the provider rejects the
    replayed turn, so it is never normalized or re-encoded.
    """

    kind: Literal["thinking"] = "thinking"
    text: str = ""
    signature: str | None = None


class RedactedThinkingBlock(ContractModel):
    """One opaque Anthropic redacted-thinking block from assistant history."""

    kind: Literal["redacted_thinking"] = "redacted_thinking"
    data: str


class EncryptedReasoningBlock(ContractModel):
    """One opaque OpenAI Responses reasoning item replayed with the input.

    ``encrypted_content`` is the provider-issued opaque payload a stateless
    caller (``store: false``) replays so the model can resume its own prior
    reasoning; it must reach the provider byte-exact.
    """

    kind: Literal["encrypted_reasoning"] = "encrypted_reasoning"
    id: str = Field(min_length=1, max_length=256)
    encrypted_content: str = Field(min_length=1)
    output_index: int | None = Field(default=None, ge=0, exclude=True)
    status: Literal["in_progress", "completed", "incomplete"] | None = Field(
        default=None,
        exclude=True,
    )


class OpaqueReasoningContentBlock(ContractModel):
    """Authenticated Fireworks reasoning retained only inside the gateway.

    The provider-issued text is never accepted directly from a caller. Public
    decoding creates a sealed block, and admission replaces it with this
    plaintext form only after authenticating the carrier against the exact
    current deployment and credential authority.
    """

    kind: Literal["reasoning_content"] = "reasoning_content"
    route_sha256: Sha256
    content: str = Field(min_length=1, max_length=8 * 1024 * 1024)
    carrier_size_bytes: int = Field(default=0, ge=0, exclude=True)


class SealedReasoningContentBlock(ContractModel):
    """One bounded, still-encrypted Fireworks continuation carrier."""

    kind: Literal["sealed_reasoning_content"] = "sealed_reasoning_content"
    carrier: str = Field(min_length=1)
    deployment_hint: str = Field(min_length=1, max_length=256)


class ExposedReasoningContentBlock(ContractModel):
    """Caller-replayed plaintext reasoning for an exposure-gated rung.

    A rung stamped ``reasoning_output_exposed`` (Tencent Hunyuan, DeepSeek)
    accepts caller-owned ``reasoning_content`` on assistant history, including
    tool-call turns. Preserve an explicitly empty string: providers can require
    the field's presence even when that turn performed no reasoning. Missing
    or null fields produce no block, and unsupported rungs omit exposed blocks
    with a disclosure. Gateway-issued sealed carriers retain their separate
    authenticated contract.
    """

    kind: Literal["exposed_reasoning_content"] = "exposed_reasoning_content"
    content: str = Field(max_length=8 * 1024 * 1024)


ProviderReasoningBlock = Annotated[
    ThinkingBlock
    | RedactedThinkingBlock
    | EncryptedReasoningBlock
    | OpaqueReasoningContentBlock
    | SealedReasoningContentBlock
    | ExposedReasoningContentBlock,
    Field(discriminator="kind"),
]
