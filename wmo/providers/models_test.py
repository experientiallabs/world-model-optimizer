"""Tests for canonical model types and provider runtime ids."""

from wmo.providers.base import ProviderConfig, ProviderKind
from wmo.providers.models import model_types_for_provider, resolve_provider_model


def test_same_model_type_resolves_to_provider_specific_ids() -> None:
    """Claude keeps one identity while direct and Bedrock wire ids differ."""
    direct = resolve_provider_model(ProviderKind.ANTHROPIC, "claude-opus-4-8")
    bedrock = resolve_provider_model(ProviderKind.BEDROCK, "claude-opus-4-8")

    assert direct.model_type == bedrock.model_type == "claude-opus-4-8"
    assert direct.model_id == "claude-opus-4-8"
    assert bedrock.model_id == "us.anthropic.claude-opus-4-8"


def test_model_catalog_owns_temperature_compatibility() -> None:
    """Sampling compatibility follows the canonical model across callers."""
    opus = resolve_provider_model(ProviderKind.BEDROCK, "claude-opus-4-8")
    opus_runtime_id = resolve_provider_model(ProviderKind.BEDROCK, "us.anthropic.claude-opus-4-8")
    sonnet = resolve_provider_model(ProviderKind.BEDROCK, "claude-sonnet-4-6")

    assert opus.forward_temperature is False
    assert opus_runtime_id.forward_temperature is False
    assert sonnet.forward_temperature is True
    assert resolve_provider_model(ProviderKind.OPENAI, "gpt-5.5").forward_temperature is False
    assert (
        resolve_provider_model(ProviderKind.OPENAI_RESPONSES, "gpt-5.6-sol").forward_temperature
        is False
    )
    assert (
        resolve_provider_model(ProviderKind.ANTHROPIC, "claude-sonnet-5").forward_temperature
        is False
    )
    assert (
        resolve_provider_model(ProviderKind.AZURE_OPENAI, "gpt-5.4-mini").forward_temperature
        is False
    )
    assert (
        resolve_provider_model(ProviderKind.AZURE_OPENAI, "deepseek-v4-pro").forward_temperature
        is True
    )


def test_runtime_id_resolves_back_to_canonical_model_type() -> None:
    """Known provider ids never become a second public model identity."""
    resolved = resolve_provider_model(
        ProviderKind.BEDROCK, "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    )

    assert resolved.model_type == "claude-haiku-4-5"
    assert resolved.model_id == "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def test_provider_catalog_exposes_only_canonical_model_types() -> None:
    assert model_types_for_provider(ProviderKind.BEDROCK)[:4] == (
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    )


def test_current_frontier_models_are_in_direct_provider_catalogs() -> None:
    assert model_types_for_provider(ProviderKind.OPENAI_RESPONSES)[:3] == (
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
    )
    assert model_types_for_provider(ProviderKind.ANTHROPIC)[:2] == (
        "claude-fable-5",
        "claude-sonnet-5",
    )


def test_azure_models_declare_their_chat_token_parameter() -> None:
    """Each built-in Azure model owns its compatible output-token field."""
    expected = {
        "gpt-5.5": "max_completion_tokens",
        "gpt-5.4": "max_completion_tokens",
        "gpt-5.4-mini": "max_completion_tokens",
        "deepseek-v4-pro": "max_tokens",
        "kimi-k2.6": "max_tokens",
    }

    actual = {
        model_type: resolve_provider_model(
            ProviderKind.AZURE_OPENAI, model_type
        ).chat_max_tokens_field
        for model_type in expected
    }

    assert actual == expected


def test_unknown_custom_model_round_trips() -> None:
    resolved = resolve_provider_model(ProviderKind.OPENAI, "my-fine-tune")
    assert resolved.model_type == "my-fine-tune"
    assert resolved.model_id == "my-fine-tune"
    assert resolved.chat_max_tokens_field == "max_completion_tokens"


def test_explicit_openai_compatible_endpoint_keeps_sampling_capability() -> None:
    config = ProviderConfig(
        kind=ProviderKind.OPENAI,
        model="gpt-5.5",
        endpoint="http://localhost:8001/v1",
    )

    assert config.resolved_chat_forward_temperature() is True


def test_provider_config_resolves_model_contract_before_custom_deployment() -> None:
    """Canonical model type, not an opaque Azure deployment name, selects parameters."""
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model="customer-gpt-deployment",
        deployment="customer-gpt-deployment",
    )

    assert config.chat_max_tokens_field == "max_completion_tokens"
    assert "chat_max_tokens_field" not in config.model_fields_set
    assert config.resolved_chat_max_tokens_field() == "max_completion_tokens"


def test_provider_config_allows_an_explicit_custom_endpoint_override() -> None:
    """Unknown OpenAI-compatible servers can override the catalog default."""
    config = ProviderConfig(
        kind=ProviderKind.OPENAI,
        model="legacy-compatible-model",
        chat_max_tokens_field="max_tokens",
    )

    assert "chat_max_tokens_field" in config.model_fields_set
    assert config.resolved_chat_max_tokens_field() == "max_tokens"


def test_persisted_default_does_not_override_a_known_model_contract() -> None:
    """Serialized defaults remain fallbacks; known catalog metadata still wins."""
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="kimi-k2.6",
        model="customer-kimi-deployment",
    )

    loaded = ProviderConfig.model_validate(config.model_dump(mode="json"))

    assert loaded.chat_max_tokens_field == "max_completion_tokens"
    assert loaded.resolved_chat_max_tokens_field() == "max_tokens"
