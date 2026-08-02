"""Canonical model types and provider-specific runtime identifiers."""

from __future__ import annotations

from llm_waterfall import ChatMaxTokensField
from pydantic import BaseModel, ConfigDict

from wmo.providers.base import ProviderKind


class ProviderModel(BaseModel):
    """One canonical model type as exposed by a concrete provider.

    ``model_type`` is the provider-independent identity used in product and
    configuration surfaces. ``model_id`` is the provider-specific value sent
    over the wire. ``chat_max_tokens_field`` records which output-token field
    the model accepts on OpenAI-compatible chat requests. ``forward_temperature``
    records whether structured chat requests may send the sampling parameter at
    all. Keeping these together prevents provider details from leaking into
    product catalogs.
    """

    model_config = ConfigDict(frozen=True)

    provider: ProviderKind
    model_type: str
    model_id: str
    chat_max_tokens_field: ChatMaxTokensField = "max_completion_tokens"
    forward_temperature: bool = True


_MODELS: tuple[ProviderModel, ...] = (
    ProviderModel(
        provider=ProviderKind.OPENAI,
        model_type="gpt-5.6-sol",
        model_id="gpt-5.6-sol",
        forward_temperature=False,
    ),
    ProviderModel(
        provider=ProviderKind.OPENAI,
        model_type="gpt-5.6-terra",
        model_id="gpt-5.6-terra",
        forward_temperature=False,
    ),
    ProviderModel(
        provider=ProviderKind.OPENAI,
        model_type="gpt-5.6-luna",
        model_id="gpt-5.6-luna",
        forward_temperature=False,
    ),
    ProviderModel(
        provider=ProviderKind.OPENAI,
        model_type="gpt-5.5",
        model_id="gpt-5.5",
        forward_temperature=False,
    ),
    ProviderModel(
        provider=ProviderKind.OPENAI,
        model_type="gpt-5.5-pro",
        model_id="gpt-5.5-pro",
        forward_temperature=False,
    ),
    ProviderModel(
        provider=ProviderKind.OPENAI,
        model_type="gpt-5.4",
        model_id="gpt-5.4",
        forward_temperature=False,
    ),
    ProviderModel(
        provider=ProviderKind.OPENAI,
        model_type="gpt-5.4-mini",
        model_id="gpt-5.4-mini",
        forward_temperature=False,
    ),
    ProviderModel(
        provider=ProviderKind.OPENAI_RESPONSES,
        model_type="gpt-5.6-sol",
        model_id="gpt-5.6-sol",
        forward_temperature=False,
    ),
    ProviderModel(
        provider=ProviderKind.OPENAI_RESPONSES,
        model_type="gpt-5.6-terra",
        model_id="gpt-5.6-terra",
        forward_temperature=False,
    ),
    ProviderModel(
        provider=ProviderKind.OPENAI_RESPONSES,
        model_type="gpt-5.6-luna",
        model_id="gpt-5.6-luna",
        forward_temperature=False,
    ),
    ProviderModel(
        provider=ProviderKind.OPENAI_RESPONSES,
        model_type="gpt-5.5",
        model_id="gpt-5.5",
        forward_temperature=False,
    ),
    ProviderModel(
        provider=ProviderKind.OPENAI_RESPONSES,
        model_type="gpt-5.5-pro",
        model_id="gpt-5.5-pro",
        forward_temperature=False,
    ),
    ProviderModel(
        provider=ProviderKind.OPENAI_RESPONSES,
        model_type="gpt-5.4",
        model_id="gpt-5.4",
        forward_temperature=False,
    ),
    ProviderModel(
        provider=ProviderKind.OPENAI_RESPONSES,
        model_type="gpt-5.4-mini",
        model_id="gpt-5.4-mini",
        forward_temperature=False,
    ),
    ProviderModel(
        provider=ProviderKind.ANTHROPIC,
        model_type="claude-fable-5",
        model_id="claude-fable-5",
        forward_temperature=False,
    ),
    ProviderModel(
        provider=ProviderKind.ANTHROPIC,
        model_type="claude-sonnet-5",
        model_id="claude-sonnet-5",
        forward_temperature=False,
    ),
    ProviderModel(
        provider=ProviderKind.ANTHROPIC,
        model_type="claude-opus-4-8",
        model_id="claude-opus-4-8",
        forward_temperature=False,
    ),
    ProviderModel(
        provider=ProviderKind.ANTHROPIC,
        model_type="claude-opus-4-7",
        model_id="claude-opus-4-7",
    ),
    ProviderModel(
        provider=ProviderKind.ANTHROPIC,
        model_type="claude-sonnet-4-6",
        model_id="claude-sonnet-4-6",
    ),
    ProviderModel(
        provider=ProviderKind.ANTHROPIC,
        model_type="claude-haiku-4-5",
        model_id="claude-haiku-4-5",
        forward_temperature=False,
    ),
    ProviderModel(
        provider=ProviderKind.ANTHROPIC,
        model_type="claude-opus-5",
        model_id="claude-opus-5",
        # Like Opus 4.8, Opus 5 rejects forwarded sampling parameters.
        forward_temperature=False,
    ),
    ProviderModel(
        provider=ProviderKind.BEDROCK,
        model_type="claude-opus-4-8",
        model_id="us.anthropic.claude-opus-4-8",
        # Opus 4.8 dropped sampling parameters. Bedrock rejects a forwarded
        # temperature with a ValidationException instead of ignoring it.
        forward_temperature=False,
    ),
    ProviderModel(
        provider=ProviderKind.BEDROCK,
        model_type="claude-opus-4-7",
        model_id="us.anthropic.claude-opus-4-7",
    ),
    ProviderModel(
        provider=ProviderKind.BEDROCK,
        model_type="claude-sonnet-4-6",
        model_id="us.anthropic.claude-sonnet-4-6",
    ),
    ProviderModel(
        provider=ProviderKind.BEDROCK,
        model_type="claude-haiku-4-5",
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
    ),
    ProviderModel(
        provider=ProviderKind.BEDROCK,
        model_type="claude-opus-5",
        # Inference-profile id verified live on this account (converse OK, 2026-07-28).
        model_id="us.anthropic.claude-opus-5",
        forward_temperature=False,
    ),
    ProviderModel(provider=ProviderKind.BEDROCK, model_type="glm-5", model_id="zai.glm-5"),
    ProviderModel(
        provider=ProviderKind.BEDROCK,
        model_type="qwen3-vl-235b-a22b",
        model_id="qwen.qwen3-vl-235b-a22b",
    ),
    ProviderModel(
        provider=ProviderKind.BEDROCK,
        model_type="gpt-oss-120b",
        model_id="openai.gpt-oss-120b-1:0",
    ),
    # Azure uses deployment names at runtime. These defaults deliberately
    # match the canonical type; callers with custom deployment names override
    # ProviderConfig.deployment without changing model identity.
    ProviderModel(
        provider=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model_id="gpt-5.5",
        forward_temperature=False,
    ),
    ProviderModel(
        provider=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.4",
        model_id="gpt-5.4",
        forward_temperature=False,
    ),
    ProviderModel(
        provider=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.4-mini",
        model_id="gpt-5.4-mini",
        forward_temperature=False,
    ),
    ProviderModel(
        provider=ProviderKind.AZURE_OPENAI,
        model_type="deepseek-v4-pro",
        model_id="deepseek-v4-pro",
        chat_max_tokens_field="max_tokens",
    ),
    ProviderModel(
        provider=ProviderKind.AZURE_OPENAI,
        model_type="kimi-k2.6",
        model_id="kimi-k2.6",
        chat_max_tokens_field="max_tokens",
    ),
)


def model_types_for_provider(provider: ProviderKind) -> tuple[str, ...]:
    """Return canonical model types offered by ``provider`` in catalog order."""
    return tuple(spec.model_type for spec in _MODELS if spec.provider is provider)


def resolve_provider_model(provider: ProviderKind, model: str) -> ProviderModel:
    """Resolve a canonical model type or known runtime id for ``provider``.

    Unknown values remain valid as custom/self-hosted model types whose wire id
    is identical. This preserves WMO's open-ended provider contract while
    canonicalizing every model in the built-in catalog.
    """
    # Case-insensitive: Azure deployments are often created with vendor casing
    # ("DeepSeek-V4-Pro") while the catalog rows are lowercase.
    wanted = model.lower()
    for spec in _MODELS:
        if spec.provider is provider and wanted in (spec.model_type.lower(), spec.model_id.lower()):
            return spec
    return ProviderModel(provider=provider, model_type=model, model_id=model)


def resolve_chat_max_tokens_field(
    provider: ProviderKind,
    model: str,
    *,
    fallback: ChatMaxTokensField = "max_completion_tokens",
) -> ChatMaxTokensField:
    """Resolve a known model contract, or preserve a custom endpoint's fallback."""
    # Case-insensitive for the same reason as resolve_provider_model: Azure deployments
    # carry vendor casing ("DeepSeek-V4-Pro") while catalog rows are lowercase.
    wanted = model.lower()
    for spec in _MODELS:
        if spec.provider is provider and wanted in (spec.model_type.lower(), spec.model_id.lower()):
            return spec.chat_max_tokens_field
    return fallback
