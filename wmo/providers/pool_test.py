"""Tests for the candidate model pool (schema, loading, pricing, provider construction)."""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from filelock import FileLock
from pydantic import ValidationError

from wmo.core.locks import FileLockTimeout
from wmo.providers import pool as pool_module
from wmo.providers.azure_openai import AzureOpenAIProvider
from wmo.providers.base import ProviderConfig, ProviderKind, TokenUsage
from wmo.providers.openrouter import OPENROUTER_API_KEY_ENV, OpenRouterProvider
from wmo.providers.openrouter_pricing import CATALOG_PATH_ENV, PriceCatalog
from wmo.providers.pool import (
    DEFAULT_POOL_PATH,
    PoolEntry,
    load_pool,
    pool_api_key,
    pool_provider,
    prepare_pool_provider,
    static_requirements,
    upsert_pool_entry,
)
from wmo.providers.registry import get_provider
from wmo.tracking.pricing import ModelPrice

_POOL_TOML = """
[[model]]
name = "deepseek-v4-pro"
kind = "azure"
model = "DeepSeek-V4-Pro"
deployment = "DeepSeek-V4-Pro"
endpoint = "https://silen-resource.services.ai.azure.com"
api_version = "2024-10-21"
api_key_env = "AZURE_SILEN_RESOURCE_API_KEY"
tier = "open"
input_per_mtok = 1.2
output_per_mtok = 4.8
cached_input_per_mtok = 0.12
cache_write_per_mtok = 1.5

[[model]]
name = "gpt-5.5"
kind = "azure"
model = "gpt-5.5"
deployment = "gpt-5.5"
endpoint = "https://google-sheets.openai.azure.com"
api_version = "2024-10-21"
api_key_env = "AZURE_GOOGLE_SHEETS_API_KEY"

[[model]]
name = "fable"
kind = "anthropic"
model = "claude-fable-5"
"""


def _write_pool(tmp_path: Path, text: str = _POOL_TOML) -> Path:
    path = tmp_path / "pool.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_pool_parses_entries(tmp_path: Path) -> None:
    pool = load_pool(_write_pool(tmp_path))
    assert [m.name for m in pool.models] == ["deepseek-v4-pro", "gpt-5.5", "fable"]
    deepseek = pool.entry("deepseek-v4-pro")
    assert deepseek.kind is ProviderKind.AZURE_OPENAI
    assert deepseek.tier == "open"
    assert deepseek.api_key_env == "AZURE_SILEN_RESOURCE_API_KEY"
    assert deepseek.cached_input_per_mtok == 0.12
    assert deepseek.cache_write_per_mtok == 1.5
    # Entries default to the frontier tier (the D-REPORT ModelRef vocabulary).
    assert pool.entry("fable").tier == "frontier"


def test_load_pool_missing_file_says_what_to_create(tmp_path: Path) -> None:
    missing = tmp_path / "nope.toml"
    with pytest.raises(FileNotFoundError, match=r"\[\[model\]\]") as excinfo:
        load_pool(missing)
    assert "wmo providers set" in str(excinfo.value), "the command that writes the file"
    assert DEFAULT_POOL_PATH == Path(".wmo/pool.toml")


def test_duplicate_names_rejected(tmp_path: Path) -> None:
    extra = '\n[[model]]\nname = "fable"\nkind = "anthropic"\nmodel = "claude-fable-5"\n'
    dupe = _POOL_TOML + extra
    with pytest.raises(ValueError, match="fable"):
        load_pool(_write_pool(tmp_path, dupe))


def test_empty_pool_rejected(tmp_path: Path) -> None:
    """And it names the file and a command that writes an entry, as the missing case does.

    Falling through to pydantic here printed `List should have at least 1 item ... [too_short]`
    plus an errors.pydantic.dev URL: no path, no remedy. Every caller only ever shows `str(exc)`.
    """
    path = _write_pool(tmp_path, "# no models\n")
    with pytest.raises(ValueError, match="wmo providers set") as excinfo:
        load_pool(path)
    message = str(excinfo.value)
    assert str(path) in message
    assert "wmo optimize route student" in message
    assert "too_short" not in message


def test_single_bracket_model_table_says_to_double_the_brackets(tmp_path: Path) -> None:
    # `[model]` is a typo on the very syntax the missing-file message recommends, and pydantic
    # answered it with `Input should be a valid list`.
    with pytest.raises(ValueError, match=r"\[\[model\]\]"):
        load_pool(_write_pool(tmp_path, '[model]\nname = "fable"\n'))


def test_a_pool_that_is_not_valid_toml_names_the_file(tmp_path: Path) -> None:
    path = _write_pool(tmp_path, "model = \n")
    with pytest.raises(ValueError) as excinfo:
        load_pool(path)
    message = str(excinfo.value)
    assert str(path) in message
    assert "not valid TOML" in message


def test_an_invalid_entry_names_the_file_the_table_and_the_field(tmp_path: Path) -> None:
    path = _write_pool(tmp_path, _POOL_TOML + '\n[[model]]\nname = "half"\nkind = "openai"\n')
    with pytest.raises(ValueError) as excinfo:
        load_pool(path)
    message = str(excinfo.value)
    assert str(path) in message
    assert "[[model]] 4 model" in message, "the table's position in the file, and the field"
    assert "errors.pydantic.dev" not in message


def test_unknown_model_requires_explicit_price() -> None:
    with pytest.raises(ValueError, match="input_per_mtok"):
        PoolEntry(name="glm", kind=ProviderKind.OPENAI, model="FW-GLM-5.2")


def test_price_must_be_set_as_a_pair() -> None:
    with pytest.raises(ValueError, match="both"):
        PoolEntry(name="glm", kind=ProviderKind.OPENAI, model="FW-GLM-5.2", input_per_mtok=1.0)


def test_price_falls_back_to_builtin_table() -> None:
    entry = PoolEntry(name="fable", kind=ProviderKind.ANTHROPIC, model="claude-fable-5")
    price = entry.price()
    assert price.input_per_mtok == 10.0
    assert price.output_per_mtok == 50.0


def test_price_override_wins_over_table() -> None:
    entry = PoolEntry(
        name="fable-discount",
        kind=ProviderKind.ANTHROPIC,
        model="claude-fable-5",
        input_per_mtok=1.0,
        output_per_mtok=2.0,
    )
    assert entry.price().input_per_mtok == 1.0


def test_provider_config_maps_backend_knobs() -> None:
    entry = PoolEntry(
        name="gpt",
        kind=ProviderKind.AZURE_OPENAI,
        model="gpt-5.5",
        deployment="gpt-5.5",
        endpoint="https://google-sheets.openai.azure.com",
        api_version="2024-10-21",
        reasoning_effort="high",
    )
    config = entry.provider_config()
    assert config.kind is ProviderKind.AZURE_OPENAI
    assert config.model == "gpt-5.5"
    assert config.deployment == "gpt-5.5"
    assert config.endpoint == "https://google-sheets.openai.azure.com"
    assert config.api_version == "2024-10-21"
    assert config.reasoning_effort == "high"


def test_provider_config_resolves_endpoint_and_deployment_env_without_serializing_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WMO_TEST_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("WMO_TEST_DEPLOYMENT", "private-deployment")
    entry = PoolEntry(
        name="gpt",
        kind=ProviderKind.AZURE_OPENAI,
        model="gpt-5.5",
        endpoint_env="WMO_TEST_ENDPOINT",
        deployment_env="WMO_TEST_DEPLOYMENT",
    )

    config = entry.provider_config()

    assert config.endpoint == "https://example.openai.azure.com"
    assert config.deployment == "private-deployment"
    dumped = entry.model_dump_json()
    assert "private-deployment" not in dumped
    assert "https://example.openai.azure.com" not in dumped


def test_env_backed_endpoint_and_deployment_are_required_at_config_time() -> None:
    entry = PoolEntry(
        name="gpt",
        kind=ProviderKind.AZURE_OPENAI,
        model="gpt-5.5",
        endpoint_env="WMO_MISSING_ENDPOINT",
        deployment_env="WMO_MISSING_DEPLOYMENT",
    )

    with pytest.raises(ValueError, match="WMO_MISSING_ENDPOINT"):
        entry.provider_config()


def test_pool_rejects_literal_and_env_reference_for_the_same_backend_field() -> None:
    with pytest.raises(ValidationError, match="endpoint or endpoint_env"):
        PoolEntry(
            name="gpt",
            kind=ProviderKind.AZURE_OPENAI,
            model="gpt-5.5",
            endpoint="https://example.openai.azure.com",
            endpoint_env="WMO_TEST_ENDPOINT",
            deployment="deployment",
        )


def test_pool_provider_requires_named_env_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AZURE_SILEN_RESOURCE_API_KEY", raising=False)
    pool = load_pool(_write_pool(tmp_path))
    with pytest.raises(ValueError, match="AZURE_SILEN_RESOURCE_API_KEY"):
        pool_provider(pool.entry("deepseek-v4-pro"))


def test_pool_provider_passes_explicit_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_SILEN_RESOURCE_API_KEY", "sk-pool-test")
    pool = load_pool(_write_pool(tmp_path))
    provider = pool_provider(pool.entry("deepseek-v4-pro"))
    assert isinstance(provider, AzureOpenAIProvider)
    # The explicit key is the trusted channel: the client authenticates with the entry's own
    # account key even though the endpoint differs from AZURE_OPENAI_ENDPOINT (which would
    # otherwise downgrade auth to the WMO_ENDPOINT_API_KEY placeholder).
    client = provider._get_client()  # noqa: SLF001 - asserting the wired credential
    assert client.api_key == "sk-pool-test"


def test_pool_provider_names_the_entry_when_a_backend_refuses_its_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A backend that refuses to be built says nothing about WHICH candidate did it, and callers
    # loop over a whole pool (`wmo optimize route sweep` constructs all of them as a pre-flight),
    # so the entry name and kind have to survive into the message an operator reads.
    monkeypatch.setenv("WMO_POOL_TEST_KEY", "sk-present")
    entry = PoolEntry(
        name="student",
        kind=ProviderKind.TINKER,
        model="Qwen/Qwen3-8B",
        api_key_env="WMO_POOL_TEST_KEY",  # set: this is a backend refusal, not a missing key
        input_per_mtok=0.1,
        output_per_mtok=0.2,
    )
    with pytest.raises(ValueError, match=r"pool model 'student' \(kind=tinker\)") as failure:
        pool_provider(entry)
    # The backend's own advice is preserved, not replaced by the identification.
    assert "TINKER_API_KEY" in str(failure.value)


def test_static_requirements_pass_a_complete_entry() -> None:
    # Nothing is required of the kinds whose only prerequisite (a credential, a region) lives in
    # the environment rather than the entry, and a complete azure entry is complete.
    assert (
        static_requirements(
            PoolEntry(name="fable", kind=ProviderKind.ANTHROPIC, model="claude-fable-5")
        )
        == []
    )
    complete_azure = PoolEntry(
        name="gpt",
        kind=ProviderKind.AZURE_OPENAI,
        model="gpt-5.5",
        deployment="gpt-5.5",
        api_version="2024-10-21",
    )
    assert static_requirements(complete_azure) == []


def test_static_requirements_name_the_azure_api_version() -> None:
    # `AzureOpenAIProvider._get_client` refuses without an api-version, and that check runs inside
    # the FIRST call: a swept candidate would abort mid-run. Knowable from the entry alone.
    entry = PoolEntry(
        name="gpt",
        kind=ProviderKind.AZURE_OPENAI,
        model="gpt-5.5",
        deployment="gpt-5.5",
    )
    assert [problem for problem in static_requirements(entry) if "api_version" in problem]


def test_static_requirements_reject_a_tinker_weights_path() -> None:
    # A `tinker://` path can never render a prompt from a pool entry: the renderer and tokenizer
    # resolve from `ProviderConfig.model_type`, and a pool entry has no field that fills it.
    entry = PoolEntry(
        name="student",
        kind=ProviderKind.TINKER,
        model="tinker://abc/sampler_weights/42",
        input_per_mtok=0.1,
        output_per_mtok=0.2,
    )
    problems = static_requirements(entry)
    assert len(problems) == 1
    assert "model_type" not in problems[0]  # worded for the pool file, not the provider config
    assert "base model" in problems[0]


def test_prepare_pool_provider_forces_the_lazy_client(monkeypatch: pytest.MonkeyPatch) -> None:
    # The point of the seam: an azure entry with no endpoint (and no AZURE_OPENAI_ENDPOINT)
    # CONSTRUCTS fine, because `__init__` only stores the config, and fails only when the client is
    # built. `prepare_pool_provider` builds it, without a request, and names the entry.
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    entry = PoolEntry(
        name="gpt-azure",
        kind=ProviderKind.AZURE_OPENAI,
        model="gpt-5.5",
        deployment="gpt-5.5",
        api_version="2024-10-21",
    )
    assert isinstance(pool_provider(entry), AzureOpenAIProvider)  # construction alone says nothing
    with pytest.raises(ValueError, match=r"pool model 'gpt-azure' \(kind=azure\)") as failure:
        prepare_pool_provider(entry)
    assert "AZURE_OPENAI_ENDPOINT" in str(failure.value)  # the backend's own advice survives


def test_prepare_pool_provider_returns_a_usable_entrys_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WMO_POOL_TEST_KEY", "sk-present")
    entry = PoolEntry(
        name="gpt-azure",
        kind=ProviderKind.AZURE_OPENAI,
        model="gpt-5.5",
        deployment="gpt-5.5",
        endpoint="https://example.openai.azure.com",
        api_version="2024-10-21",
        api_key_env="WMO_POOL_TEST_KEY",
    )
    provider = prepare_pool_provider(entry)
    assert isinstance(provider, AzureOpenAIProvider)


def test_pool_api_key_checks_credentials_without_building_a_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The seam a caller about to spend on a whole pool uses to check every candidate up front:
    # same verdict as `pool_provider`, no provider constructed and no network client touched.
    pool = load_pool(_write_pool(tmp_path))
    monkeypatch.delenv("AZURE_SILEN_RESOURCE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="AZURE_SILEN_RESOURCE_API_KEY"):
        pool_api_key(pool.entry("deepseek-v4-pro"))
    monkeypatch.setenv("AZURE_SILEN_RESOURCE_API_KEY", "sk-pool-test")
    assert pool_api_key(pool.entry("deepseek-v4-pro")) == "sk-pool-test"
    # An entry with no api_key_env uses the backend's default credentials, and says so with None.
    assert pool_api_key(pool.entry("fable")) is None


def test_pool_entry_unknown_name_lists_available(tmp_path: Path) -> None:
    pool = load_pool(_write_pool(tmp_path))
    with pytest.raises(KeyError, match="deepseek-v4-pro"):
        pool.entry("not-a-model")


def test_get_provider_rejects_api_key_for_bedrock() -> None:
    config = ProviderConfig(kind=ProviderKind.BEDROCK, model="us.anthropic.claude-opus-4-8")
    with pytest.raises(ValueError, match="[Bb]edrock"):
        get_provider(config, api_key="sk-nope")


def test_cost_usd_is_cache_adjusted() -> None:
    entry = PoolEntry(
        name="cached",
        kind=ProviderKind.AZURE_OPENAI,
        model="gpt-5.5",
        deployment="gpt-5.5",
        input_per_mtok=10.0,
        output_per_mtok=20.0,
        cached_input_per_mtok=1.0,
    )
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=0, cached_input_tokens=400_000)
    # 600k fresh @ $10/M + 400k cached @ $1/M = $6.40 - never the $10 list price.
    assert entry.cost_usd(usage) == pytest.approx(6.4)


def test_call_cost_usd_applies_openai_long_context_tier_to_explicit_prices() -> None:
    entry = PoolEntry(
        name="gpt55",
        kind=ProviderKind.OPENAI_RESPONSES,
        model="gpt-5.5-2026-04-23",
        model_type="gpt-5.5",
        input_per_mtok=5.0,
        output_per_mtok=30.0,
        cached_input_per_mtok=0.5,
    )
    usage = TokenUsage(
        input_tokens=300_000,
        output_tokens=10_000,
        cached_input_tokens=100_000,
    )
    assert entry.cost_usd(usage) == pytest.approx(1.35)
    assert entry.call_cost_usd(usage) == pytest.approx(2.55)


def test_cost_usd_without_cache_price_bills_cached_tokens_at_full_rate() -> None:
    entry = PoolEntry(
        name="no-cache-price",
        kind=ProviderKind.AZURE_OPENAI,
        model="gpt-5.5",
        deployment="gpt-5.5",
        input_per_mtok=10.0,
        output_per_mtok=20.0,
    )
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=0, cached_input_tokens=400_000)
    assert entry.cost_usd(usage) == pytest.approx(10.0)  # honest fallback, never free


def test_cost_usd_bills_cache_writes_at_entry_override() -> None:
    entry = PoolEntry(
        name="write-priced",
        kind=ProviderKind.AZURE_OPENAI,
        model="gpt-5.5",
        deployment="gpt-5.5",
        input_per_mtok=10.0,
        output_per_mtok=20.0,
        cache_write_per_mtok=12.5,
    )
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=0, cache_write_input_tokens=400_000)
    # 600k fresh @ $10/M + 400k written @ $12.5/M = 6.0 + 5.0 = $11.00.
    assert entry.cost_usd(usage) == pytest.approx(11.0)


def test_cost_usd_falls_back_to_builtin_cache_tiers() -> None:
    # An entry with NO explicit prices uses the built-in table, including its cache tiers
    # (fable: reads 0.1x -> $1/M, writes 1.25x -> $12.5/M on a $10/M input rate).
    entry = PoolEntry(name="fable", kind=ProviderKind.ANTHROPIC, model="claude-fable-5")
    usage = TokenUsage(
        input_tokens=1_000_000,
        output_tokens=0,
        cached_input_tokens=300_000,
        cache_write_input_tokens=200_000,
    )
    # 500k fresh @ $10 + 300k read @ $1 + 200k write @ $12.5 = 5.0 + 0.3 + 2.5 = $7.80.
    assert entry.cost_usd(usage) == pytest.approx(7.8)


def test_bedrock_entry_pins_region() -> None:
    entry = PoolEntry(
        name="opus-4-8",
        kind=ProviderKind.BEDROCK,
        model="us.anthropic.claude-opus-4-8",
        region="us-east-1",
    )
    assert entry.provider_config().region == "us-east-1"


def test_unknown_pool_keys_fail_at_load() -> None:
    # A typo like api_key_evn must fail at load, not surface as a 401 at request time.
    with pytest.raises(ValidationError, match="api_key_evn"):
        PoolEntry.model_validate(
            {
                "name": "typo",
                "kind": "anthropic",
                "model": "claude-haiku-4-5",
                "api_key_evn": "SOME_KEY",
            }
        )


def test_azure_entry_requires_deployment() -> None:
    with pytest.raises(ValidationError, match="deployment"):
        PoolEntry(
            name="no-deploy",
            kind=ProviderKind.AZURE_OPENAI,
            model="gpt-5.5",
            input_per_mtok=1.0,
            output_per_mtok=2.0,
        )


def _student_entry(name: str = "student") -> PoolEntry:
    """A distilled-student shaped entry: a weights path the catalog cannot resolve."""
    return PoolEntry(
        name=name,
        kind=ProviderKind.OPENAI,
        model="tinker://weights/abc123",
        model_type="Qwen/Qwen3-30B-A3B",
        chat_max_tokens_field="max_tokens",
        endpoint="https://tinker.example/oai/api/v1",
        api_key_env="TINKER_API_KEY",
        tier="open",
        input_per_mtok=0.1,
        output_per_mtok=0.4,
    )


def test_provider_config_forwards_model_type_and_max_tokens_field() -> None:
    """Both new fields must reach ProviderConfig, or a routed student 400s on every call.

    `model` is a tinker:// weights path the built-in catalog cannot resolve, so capability
    resolution has only `model_type` and the explicit field to go on.
    """
    config = _student_entry().provider_config()

    assert config.model_type == "Qwen/Qwen3-30B-A3B"
    assert config.chat_max_tokens_field == "max_tokens"
    assert config.resolved_chat_max_tokens_field() == "max_tokens"


def test_pool_entry_defaults_keep_the_built_in_contract() -> None:
    """An entry that says nothing keeps the catalog's answer, so this cannot regress GPT-5.x."""
    entry = PoolEntry(name="gpt", kind=ProviderKind.OPENAI, model="gpt-5.5")

    assert entry.model_type is None
    assert entry.provider_config().resolved_chat_max_tokens_field() == "max_completion_tokens"


def test_upsert_pool_entry_creates_the_file_and_appends(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "pool.toml"

    assert upsert_pool_entry(_student_entry(), path).replaced is False

    assert [m.name for m in load_pool(path).models] == ["student"]
    entry = load_pool(path).entry("student")
    assert entry.model_type == "Qwen/Qwen3-30B-A3B"
    assert entry.chat_max_tokens_field == "max_tokens"
    assert entry.api_key_env == "TINKER_API_KEY"


def test_upsert_pool_entry_replaces_by_name_and_keeps_the_others(tmp_path: Path) -> None:
    path = tmp_path / "pool.toml"
    path.write_text(_POOL_TOML, encoding="utf-8")
    before = [m.name for m in load_pool(path).models]

    assert upsert_pool_entry(_student_entry(), path).replaced is False
    assert upsert_pool_entry(_student_entry(), path).replaced is True

    after = [m.name for m in load_pool(path).models]
    assert after == [*before, "student"]  # replaced in place, nothing duplicated or dropped


def test_upsert_pool_entry_does_not_stamp_defaults_onto_existing_entries(tmp_path: Path) -> None:
    """Hand-maintained entries must not gain every default just because a student was added.

    Byte-level preservation of the whole file is covered by
    `test_upsert_pool_entry_appends_without_touching_a_byte_of_the_existing_file`; this checks the
    narrower property that survives even a REPLACEMENT, which has to re-render the roster.
    """
    path = tmp_path / "pool.toml"
    path.write_text(_POOL_TOML, encoding="utf-8")

    upsert_pool_entry(_student_entry(), path)

    written = path.read_text(encoding="utf-8")
    assert 'tier = "frontier"' not in written  # the default was never stamped onto gpt-5.5
    assert "chat_max_tokens_field" in written  # but the student's explicit value is there
    gpt = load_pool(path).entry("gpt-5.5")
    assert gpt.input_per_mtok is None  # still priced from the built-in table, as authored


def test_upsert_pool_entry_names_the_file_when_it_is_not_valid_toml(tmp_path: Path) -> None:
    path = tmp_path / "pool.toml"
    path.write_text("[[model]\nname = broken", encoding="utf-8")

    with pytest.raises(ValueError, match=r"is not valid TOML"):
        upsert_pool_entry(_student_entry(), path)


def test_upsert_pool_entry_rejects_a_file_that_is_not_a_model_array(tmp_path: Path) -> None:
    path = tmp_path / "pool.toml"
    path.write_text('model = "not-an-array"\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"\[\[model\]\] tables"):
        upsert_pool_entry(_student_entry(), path)


def test_upsert_pool_entry_uses_a_process_private_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two concurrent upserts must not be able to rename each other's half-written file.

    A shared fixed staging path turns a lost update into a CORRUPT roster: writer A can rename
    B's partially written file into place. A name unique per call bounds the damage to
    last-writer-wins. The property belongs to `wmo.core.files.write_bytes_atomic` and is covered
    directly in `wmo/core/files_test.py`; this pins that the roster's write actually goes through
    it, which is the reason it was documented on this function in the first place.
    """
    renamed: list[str] = []
    real_replace = Path.replace

    def _record(self: Path, target: object) -> Path:
        renamed.append(self.name)
        return real_replace(self, target)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(Path, "replace", _record)
    path = tmp_path / "pool.toml"
    upsert_pool_entry(_student_entry(), path)
    upsert_pool_entry(_student_entry("other"), path)

    assert len(set(renamed)) == 2, f"the staging name repeated across calls: {renamed}"
    assert all(name.startswith(".pool.toml.") for name in renamed)
    assert list(tmp_path.glob("*.partial")) == []  # the staging file is gone once renamed


def test_upsert_pool_entry_leaves_no_temp_behind_when_the_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pool.toml"
    path.write_text(_POOL_TOML, encoding="utf-8")
    real_replace = Path.replace

    def _boom(self: Path, target: object) -> Path:
        if self.suffix == ".partial":
            raise OSError("disk full")
        return real_replace(self, target)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(Path, "replace", _boom)
    with pytest.raises(OSError, match="disk full"):
        upsert_pool_entry(_student_entry(), path)

    assert list(tmp_path.glob("*.partial")) == []
    assert load_pool(path).models  # the roster is untouched and still loadable


_COMMENTED_POOL = """# Roster for the support endpoint. Keep this file under review.
[[model]]
name = "gpt-5.5"
kind = "azure"
model = "gpt-5.5"
deployment = "gpt-5.5"
# billed to the prod OpenAI account
api_key_env = "AZURE_PROD_KEY"

# Cheap tier. DO NOT delete: the savings figure is quoted against this row.
[[model]]
name = "haiku"
kind = "anthropic"
model = "claude-haiku-4-5"
"""


def test_upsert_pool_entry_appends_without_touching_a_byte_of_the_existing_file(
    tmp_path: Path,
) -> None:
    """An operator's comments say which account a row bills to; losing them silently is not ok."""
    path = tmp_path / "pool.toml"
    path.write_text(_COMMENTED_POOL, encoding="utf-8")

    assert upsert_pool_entry(_student_entry(), path).replaced is False

    written = path.read_text(encoding="utf-8")
    assert written.startswith(_COMMENTED_POOL)  # byte-identical prefix, comments included
    assert "DO NOT delete" in written
    assert "billed to the prod OpenAI account" in written
    assert [m.name for m in load_pool(path).models] == ["gpt-5.5", "haiku", "student"]


def test_upsert_pool_entry_append_round_trips_when_the_file_has_no_trailing_newline(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pool.toml"
    path.write_text(
        '[[model]]\nname = "haiku"\nkind = "anthropic"\nmodel = "claude-haiku-4-5"',
        encoding="utf-8",
    )

    upsert_pool_entry(_student_entry(), path)

    assert [m.name for m in load_pool(path).models] == ["haiku", "student"]


def _short_openai(name: str) -> PoolEntry:
    """The smallest entry anyone registers: three fields, well under 100 rendered characters."""
    return PoolEntry(name=name, kind=ProviderKind.OPENAI, model="gpt-5.5")


def test_upsert_pool_entry_keeps_the_roster_loadable_across_two_short_registrations(
    tmp_path: Path,
) -> None:
    """The direct regression: two minimal OpenAI rows used to write a duplicate top-level key.

    `tomli_w` emits `[[model]]` sections only when a table is too long to inline (a 100-character
    heuristic), so `model = [ {...} ]` came out for short entries and appending a second one
    produced invalid TOML. Every later `load_pool` then failed, which takes serving and the
    routing optimizer down, not just the candidate being added. This fixture must stay MINIMAL: a
    verbose Azure row crosses the heuristic and passes with or without the fix.
    """
    path = tmp_path / "pool.toml"

    assert upsert_pool_entry(_short_openai("gpt-5.5"), path).replaced is False
    assert upsert_pool_entry(_short_openai("gpt-5.4-mini"), path).replaced is False

    assert [m.name for m in load_pool(path).models] == ["gpt-5.5", "gpt-5.4-mini"]


@pytest.mark.parametrize(
    "second",
    [
        pytest.param(_short_openai("gpt-5.4-mini"), id="short-openai"),
        pytest.param(
            PoolEntry(
                name="azure-gpt",
                kind=ProviderKind.AZURE_OPENAI,
                model="gpt-5.5",
                deployment="gpt-5.5",
                endpoint="https://example.openai.azure.com",
                api_version="2024-10-21",
                api_key_env="AZURE_EXAMPLE_KEY",
            ),
            id="long-azure",
        ),
    ],
)
def test_upsert_pool_entry_round_trips_whatever_shape_the_entry_renders_to(
    tmp_path: Path, second: PoolEntry
) -> None:
    """Parametrized over both sides of the length heuristic so it cannot silently come back."""
    path = tmp_path / "pool.toml"

    upsert_pool_entry(_short_openai("gpt-5.5"), path)
    upsert_pool_entry(second, path)

    assert [m.name for m in load_pool(path).models] == ["gpt-5.5", second.name]


def test_upsert_pool_entry_appends_short_entries_indefinitely(tmp_path: Path) -> None:
    """Three-plus sequential adds: the roster stays loadable and keeps every entry, in order."""
    path = tmp_path / "pool.toml"
    names = ["gpt-5.5", "gpt-5.4-mini", "haiku", "sonnet"]

    for name in names:
        assert upsert_pool_entry(_short_openai(name), path).replaced is False

    assert [m.name for m in load_pool(path).models] == names


def test_upsert_pool_entry_writes_a_file_a_later_append_can_preserve(tmp_path: Path) -> None:
    """Creating the file must not cost the NEXT add its comment preservation.

    The inline array form loads fine, so the corruption is only half the story: a file in that
    form cannot be appended to at all, and every later add has to re-render it and drop the
    operator's comments. Writing sections from the first entry on is what keeps the byte-preserving
    path reachable for a roster of short entries.
    """
    path = tmp_path / "pool.toml"
    upsert_pool_entry(_short_openai("gpt-5.5"), path)
    annotated = path.read_text(encoding="utf-8") + "\n# billed to the research account\n"
    path.write_text(annotated, encoding="utf-8")

    upsert_pool_entry(_short_openai("gpt-5.4-mini"), path)

    written = path.read_text(encoding="utf-8")
    assert written.startswith(annotated)  # byte-identical prefix, comment included
    assert [m.name for m in load_pool(path).models] == ["gpt-5.5", "gpt-5.4-mini"]


_LEGACY_INLINE_POOL = """model = [
    { name = "gpt-5.5", kind = "openai", model = "gpt-5.5" },
]
"""


def test_upsert_pool_entry_normalizes_a_legacy_inline_pool_file(tmp_path: Path) -> None:
    """Rosters written by releases up to 0.2.1 are one inline array, which cannot be appended to.

    Adding a `[[model]]` section to one is the same duplicate-key corruption in the other
    direction, so the upgrade path is a full re-render. It costs that file its comments once, and
    leaves it in the section form every later add can extend.
    """
    path = tmp_path / "pool.toml"
    path.write_text(_LEGACY_INLINE_POOL, encoding="utf-8")

    assert upsert_pool_entry(_short_openai("haiku"), path).replaced is False

    assert [m.name for m in load_pool(path).models] == ["gpt-5.5", "haiku"]
    assert "[[model]]" in path.read_text(encoding="utf-8")  # normalized, so the next add appends


def test_upsert_pool_entry_reports_the_rewrite_that_normalizing_a_legacy_pool_costs(
    tmp_path: Path,
) -> None:
    """An ADD that drops the operator's comments must SAY so; the CLI prints the note from this.

    Normalizing an inline-form roster is a re-render, so the comments recording which account
    each row bills to are gone. Reporting that as a plain "added" (which is what keying the
    note off `replaced` alone does) deletes them silently and unrecoverably.
    """
    path = tmp_path / "pool.toml"
    path.write_text(f"# bills to research\n{_LEGACY_INLINE_POOL}", encoding="utf-8")

    written = upsert_pool_entry(_short_openai("haiku"), path)

    assert written.replaced is False  # nothing of that name was there
    assert written.rewritten is True  # but the file was re-rendered anyway, so say so
    assert "bills to research" not in path.read_text(encoding="utf-8")


def test_upsert_pool_entry_reports_no_rewrite_for_a_plain_append(tmp_path: Path) -> None:
    """The common case must not print the comment warning, or it stops meaning anything."""
    path = tmp_path / "pool.toml"
    upsert_pool_entry(_short_openai("gpt-5.5"), path)

    written = upsert_pool_entry(_short_openai("gpt-5.4-mini"), path)

    assert (written.replaced, written.rewritten) == (False, False)


def test_upsert_pool_entry_does_not_call_creating_the_file_a_rewrite(tmp_path: Path) -> None:
    """A first registration has no comments to lose, so it must not claim any were dropped."""
    written = upsert_pool_entry(_short_openai("gpt-5.5"), tmp_path / "pool.toml")

    assert (written.replaced, written.rewritten) == (False, False)


def test_upsert_pool_entry_refuses_to_write_a_roster_that_does_not_read_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The commit gate, on the re-render path that has no fallback.

    `_render_sections` hand-writes the `[[model]]` header, which is only correct while every
    table is a flat dict of scalars. A nested value makes `tomli_w` emit a `[key]` header that
    TOML reads as a SIBLING top-level table, so the field leaves the entry with no error raised:
    silent data loss, not a parse failure.

    `PoolEntry`'s `extra="forbid"` is the first line of defence and rejects such a field before
    the renderer ever sees it, so the renderer is stubbed here to produce what tomli_w would
    produce if that schema were ever relaxed. What is under test is the commit gate on the
    RE-RENDER path, which is the one with no fallback: the append path could still fall back to
    re-rendering, but a bad re-render used to be written straight out.
    """
    path = tmp_path / "pool.toml"

    def _hoists_a_sub_table(tables: list[dict[str, object]]) -> str:
        # Exactly what tomli_w does with a nested value today: the sub-table gets its own header,
        # which under the [[model]] above it parses as a SIBLING, not as a field of the entry.
        return "".join(
            f"[[model]]\nname = {table['name']!r}\n\n[rate_limits]\nrpm = 60\n" for table in tables
        )

    monkeypatch.setattr(pool_module, "_render_sections", _hoists_a_sub_table)
    with pytest.raises(ValueError, match="does not read back as"):
        upsert_pool_entry(_short_openai("gpt-5.5"), path)

    assert not path.exists()  # nothing was committed
    assert list(tmp_path.glob("*.partial")) == []


def test_load_pool_names_the_file_when_it_is_not_valid_toml(tmp_path: Path) -> None:
    """A decode error reaches the user through `SweepError(str(exc))`, so it must carry the path.

    Bare, `tomllib`'s "Cannot overwrite a value (at line 7, column 2)" surfaces under typer's
    `Invalid value:`, which reads as a bad CLI argument rather than a corrupt pool file.
    """
    path = tmp_path / "pool.toml"
    path.write_text('model = [ { name = "a" } ]\n\nmodel = [ { name = "b" } ]\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"is not valid TOML") as caught:
        load_pool(path)

    assert str(path) in str(caught.value)


def test_upsert_pool_entry_blames_a_pre_existing_bad_row_on_the_file_not_the_new_entry(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pool.toml"
    path.write_text(
        '[[model]]\nname = "mine"\nkind = "openai"\nmodel = "some/self-hosted-model"\n',
        encoding="utf-8",
    )

    original = path.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match=r"already invalid, before adding 'student'"):
        upsert_pool_entry(_student_entry(), path)

    # Validate-before-write: a rejected upsert must leave the roster exactly as it was, and must
    # not strand a temp file beside it.
    assert path.read_text(encoding="utf-8") == original
    assert list(path.parent.glob("*.tmp")) == []


def test_upsert_pool_entry_keeps_both_registrations_when_two_threads_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two concurrent writers adding different entries must BOTH land in the roster.

    An unlocked read-modify-write loses one of them and reports success to both, so a model an
    operator registered is simply absent from the pool with nothing saying so.

    The race is forced, not hoped for: every read of the roster parks on a barrier, so both
    writers provably hold the SAME snapshot before either writes, which is exactly the
    interleaving that drops an entry. With the write lock held across the whole cycle the second
    writer cannot reach its first read until the first writer has finished, so the barrier times
    out (its wait is far shorter than the lock's), the reads happen in sequence, and the second
    writer merges onto the roster it can now see.
    """
    path = tmp_path / "pool.toml"
    path.write_text(_COMMENTED_POOL, encoding="utf-8")
    read_together = threading.Barrier(2, timeout=1.0)
    real_read = Path.read_text

    def _park_on_every_roster_read(self: Path, **kwargs: object) -> str:
        text = real_read(self, **kwargs)  # ty: ignore[invalid-argument-type]
        if self.name == path.name:
            with contextlib.suppress(threading.BrokenBarrierError):
                read_together.wait()
        return text

    monkeypatch.setattr(Path, "read_text", _park_on_every_roster_read)
    failures: list[BaseException] = []

    def _register(name: str) -> None:
        try:
            upsert_pool_entry(_student_entry(name), path)
        except BaseException as exc:  # noqa: BLE001 - reported by the main thread's assertions
            failures.append(exc)

    writers = [
        threading.Thread(target=_register, args=(name,), name=name)
        for name in ("student-a", "student-b")
    ]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join(timeout=60)
        assert not writer.is_alive(), f"{writer.name} never finished; the lock wait is not bounded"
    monkeypatch.undo()  # the assertions below read the roster without parking on the barrier

    registered = sorted(entry.name for entry in load_pool(path).models)
    assert registered == ["gpt-5.5", "haiku", "student-a", "student-b"], (
        f"a concurrent registration was lost (writer errors: {failures})"
    )
    assert failures == []
    assert "DO NOT delete" in path.read_text(encoding="utf-8")  # appends, so comments survive
    # The lock file is the only extra artifact: no half-written temp survives a race.
    assert sorted(child.name for child in tmp_path.iterdir()) == ["pool.toml", "pool.toml.lock"]


# The child of `test_upsert_pool_entry_keeps_every_registration_from_concurrent_processes`: one
# real `wmo` process registering one candidate. It signals readiness, then spins until the parent
# releases every writer at once, so the processes collide on the roster instead of queueing.
_WRITER_PROGRAM = """
import sys
import time
from pathlib import Path

from wmo.providers.base import ProviderKind
from wmo.providers.pool import PoolEntry, upsert_pool_entry

pool_path, name, ready_path, start_path = (
    Path(sys.argv[1]),
    sys.argv[2],
    Path(sys.argv[3]),
    Path(sys.argv[4]),
)
entry = PoolEntry(
    name=name,
    kind=ProviderKind.OPENAI,
    model="tinker://weights/" + name,
    input_per_mtok=0.1,
    output_per_mtok=0.4,
)
ready_path.write_text("ready", encoding="utf-8")
while not start_path.exists():
    time.sleep(0.001)
upsert_pool_entry(entry, pool_path)
"""


def test_upsert_pool_entry_keeps_every_registration_from_concurrent_processes(
    tmp_path: Path,
) -> None:
    """The cross-process half: separate `wmo optimize route student` runs must not erase each other.

    The threaded test above pins the interleaving; this one proves the lock is held against other
    PROCESSES, which is the situation an operator actually hits (two terminals, or a script
    registering several students at once). Nothing is patched here, so the loss it guards against
    is timing-dependent: with the roster unlocked, eight writers released together keep whichever
    entry landed last and drop the rest.
    """
    path = tmp_path / "pool.toml"
    path.write_text(_COMMENTED_POOL, encoding="utf-8")
    start = tmp_path / "start"
    names = [f"student-{index}" for index in range(8)]
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[2])}
    writers = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                _WRITER_PROGRAM,
                str(path),
                name,
                str(tmp_path / f"ready-{name}"),
                str(start),
            ],
            env=environment,
            stderr=subprocess.PIPE,
            text=True,
        )
        for name in names
    ]
    try:
        deadline = time.monotonic() + 60.0
        while not all((tmp_path / f"ready-{name}").exists() for name in names):
            assert time.monotonic() < deadline, "the concurrent writers never became ready"
            time.sleep(0.01)
        start.write_text("go", encoding="utf-8")
        for writer in writers:
            stderr = writer.communicate(timeout=60.0)[1]
            assert writer.returncode == 0, stderr
    finally:
        for writer in writers:
            if writer.poll() is None:
                writer.kill()

    assert sorted(entry.name for entry in load_pool(path).models) == sorted(
        ["gpt-5.5", "haiku", *names]
    )
    assert "DO NOT delete" in path.read_text(encoding="utf-8")
    assert list(tmp_path.glob("*.tmp")) == []


def test_upsert_pool_entry_reports_a_stuck_writer_instead_of_hanging(tmp_path: Path) -> None:
    """A held lock must fail with an actionable message inside the bound, never wedge the CLI."""
    path = tmp_path / "pool.toml"
    path.write_text(_COMMENTED_POOL, encoding="utf-8")
    holder = FileLock(path.with_name(f"{path.name}.lock"))
    holder.acquire()
    try:
        with pytest.raises(FileLockTimeout, match=r"writing the model pool at .*pool\.toml"):
            upsert_pool_entry(_student_entry(), path, lock_timeout_s=0.05)
    finally:
        holder.release()

    assert [entry.name for entry in load_pool(path).models] == ["gpt-5.5", "haiku"]  # untouched
    assert list(tmp_path.glob("*.tmp")) == []
    # Once the holder is gone the lock is free again: the OS owns that release, so a leftover lock
    # FILE is never a held lock and cannot wedge the next run.
    assert upsert_pool_entry(_student_entry(), path, lock_timeout_s=0.05).replaced is False


def test_upsert_pool_entry_releases_the_lock_when_it_rejects_the_roster(tmp_path: Path) -> None:
    """A rejected upsert must not leave the roster locked, or one bad row wedges every later run."""
    path = tmp_path / "pool.toml"
    path.write_text(
        '[[model]]\nname = "mine"\nkind = "openai"\nmodel = "some/self-hosted-model"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"already invalid"):
        upsert_pool_entry(_student_entry(), path, lock_timeout_s=0.05)

    path.write_text(_COMMENTED_POOL, encoding="utf-8")  # the operator fixes the row
    assert upsert_pool_entry(_student_entry(), path, lock_timeout_s=0.05).replaced is False


# --- OpenRouter entries: priced from the published catalog, not by hand -----------------------

_OPENROUTER_POOL = """
[[model]]
name = "or-sonnet"
kind = "openrouter"
model = "anthropic/claude-sonnet-4"

[[model]]
name = "or-glm-free"
kind = "openrouter"
model = "z-ai/glm-4.6:free"
tier = "open"
"""


_SONNET_PRICE = ModelPrice(
    input_per_mtok=3.0,
    output_per_mtok=15.0,
    cache_read_per_mtok=0.3,
    cache_write_per_mtok=3.75,
)
_FREE_PRICE = ModelPrice(input_per_mtok=0.0, output_per_mtok=0.0)


def _catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prices: dict[str, ModelPrice] | None = None,
) -> Path:
    """Point price resolution at a fixture catalog on disk (the suite never fetches)."""
    path = tmp_path / "openrouter-prices.json"
    catalog = PriceCatalog(
        fetched_at=time.time(),
        source="test fixture",
        prices=prices
        if prices is not None
        else {"anthropic/claude-sonnet-4": _SONNET_PRICE, "z-ai/glm-4.6:free": _FREE_PRICE},
    )
    path.write_text(catalog.model_dump_json(), encoding="utf-8")
    monkeypatch.setenv(CATALOG_PATH_ENV, str(path))
    return path


def test_openrouter_entry_needs_only_a_model_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The launch promise: a pool entry is a name, a kind, and a model id. Both tiers, including
    # the cache rates, come from OpenRouter's published catalog.
    _catalog(tmp_path, monkeypatch)
    pool = load_pool(_write_pool(tmp_path, _OPENROUTER_POOL))

    entry = pool.entry("or-sonnet")
    assert entry.price().input_per_mtok == 3.0
    assert entry.price().output_per_mtok == 15.0
    assert entry.cached_input_per_mtok == pytest.approx(0.3)
    assert entry.cache_write_per_mtok == pytest.approx(3.75)
    # 500k fresh @ $3 + 500k cached @ $0.30 = 1.5 + 0.15 = $1.65.
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=0, cached_input_tokens=500_000)
    assert entry.cost_usd(usage) == pytest.approx(1.65)
    assert pool.entry("or-glm-free").price().input_per_mtok == 0.0


def test_openrouter_entry_offline_falls_back_to_the_explicit_price_error(tmp_path: Path) -> None:
    # No cache and no network (the conftest fetch stub refuses): the entry must fail with the
    # ordinary "declare the prices" instruction, and say WHY the automatic route did not apply.
    # `load_pool` re-raises pydantic's ValidationError as a plain ValueError (still what every
    # caller catches) so the message can carry the file path and a next command instead of a
    # schema dump; the entry's own instruction has to survive that rewrap intact.
    with pytest.raises(ValueError) as excinfo:
        load_pool(_write_pool(tmp_path, _OPENROUTER_POOL))
    message = str(excinfo.value)
    assert "OpenRouter price catalog" in message
    assert "unreachable" in message
    assert "add input_per_mtok and output_per_mtok" in message


def test_openrouter_entry_with_explicit_prices_keeps_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A negotiated rate is the operator's, and the catalog is never consulted for it (the
    # fixture below prices the same model very differently, and must lose).
    _catalog(tmp_path, monkeypatch)
    entry = PoolEntry(
        name="or-sonnet",
        kind=ProviderKind.OPENROUTER,
        model="anthropic/claude-sonnet-4",
        input_per_mtok=1.0,
        output_per_mtok=2.0,
    )
    assert entry.price().input_per_mtok == 1.0
    assert entry.cached_input_per_mtok is None


def test_a_priced_entry_is_never_repriced_by_a_later_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The persistence property behind a fitted policy: the resolved numbers live ON the entry,
    # and RoutingPolicy/OutcomeMatrix serialize entries verbatim. Re-validating a persisted
    # entry against a catalog that has since doubled its price must not move it.
    _catalog(tmp_path, monkeypatch)
    fitted = load_pool(_write_pool(tmp_path, _OPENROUTER_POOL)).entry("or-sonnet")
    snapshot = fitted.model_dump_json()

    _catalog(
        tmp_path,
        monkeypatch,
        {"anthropic/claude-sonnet-4": ModelPrice(input_per_mtok=6.0, output_per_mtok=30.0)},
    )
    reloaded = PoolEntry.model_validate_json(snapshot)

    assert reloaded.input_per_mtok == 3.0
    assert reloaded.price().output_per_mtok == 15.0


def test_openrouter_pool_entry_resolves_to_the_openrouter_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _catalog(tmp_path, monkeypatch)
    monkeypatch.setenv("OPENROUTER_ACCOUNT_B_KEY", "sk-or-account-b")
    per_account = (
        '[[model]]\nname = "or-glm-free"\nkind = "openrouter"\nmodel = "z-ai/glm-4.6:free"\n'
        'api_key_env = "OPENROUTER_ACCOUNT_B_KEY"\n'
    )
    pool = load_pool(_write_pool(tmp_path, per_account))

    provider = pool_provider(pool.entry("or-glm-free"))

    assert isinstance(provider, OpenRouterProvider)
    assert provider.config.kind is ProviderKind.OPENROUTER
    assert provider.config.model == "z-ai/glm-4.6:free"
    assert provider._get_client().api_key == "sk-or-account-b"  # noqa: SLF001 - asserting wiring


def test_openrouter_provider_falls_back_to_the_shared_account_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No api_key_env on the entry: the single-key launch path, straight from the environment.
    _catalog(tmp_path, monkeypatch)
    monkeypatch.setenv(OPENROUTER_API_KEY_ENV, "sk-or-shared")
    pool = load_pool(_write_pool(tmp_path, _OPENROUTER_POOL))

    provider = pool_provider(pool.entry("or-sonnet"))

    assert isinstance(provider, OpenRouterProvider)
    assert provider._get_client().api_key == "sk-or-shared"  # noqa: SLF001 - asserting wiring


def test_bedrock_entry_rejects_api_key_env_at_load() -> None:
    # `BedrockProvider.__init__` refuses an explicit key, and providers are built lazily per
    # eval cell: caught at load this is a config typo, caught at the first cell it aborts a
    # paid-for sweep. Same boundary as the azure `deployment` rule above.
    with pytest.raises(ValidationError, match="api_key_env"):
        PoolEntry(
            name="claude-bedrock",
            kind=ProviderKind.BEDROCK,
            model="us.anthropic.claude-opus-4-8",
            api_key_env="AWS_SOMETHING",
            input_per_mtok=1.0,
            output_per_mtok=2.0,
        )


_LOCAL_POOL = """
[[model]]
name = "qwen3-4b-local"
kind = "openai"
model = "qwen3:4b"
endpoint = "http://localhost:11434/v1"
input_per_mtok = 0.0
output_per_mtok = 0.0

[[model]]
name = "fable"
kind = "anthropic"
model = "claude-fable-5"
enabled = false
"""


def test_zero_priced_local_entry_validates_and_costs_zero(tmp_path: Path) -> None:
    # A self-hosted candidate's marginal price is genuinely $0; the EXPLICIT pair is what
    # separates "declared free" from the unpriced-entry accident `_validate_price` rejects.
    pool = load_pool(_write_pool(tmp_path, _LOCAL_POOL))
    entry = pool.entry("qwen3-4b-local")
    assert entry.kind is ProviderKind.OPENAI
    assert entry.endpoint == "http://localhost:11434/v1"
    price = entry.price()
    assert (price.input_per_mtok, price.output_per_mtok) == (0.0, 0.0)
    assert entry.cost_usd(TokenUsage(input_tokens=100_000, output_tokens=20_000)) == 0.0


def test_enabled_defaults_on_and_enabled_models_filters(tmp_path: Path) -> None:
    pool = load_pool(_write_pool(tmp_path, _LOCAL_POOL))
    assert pool.entry("qwen3-4b-local").enabled is True
    assert pool.entry("fable").enabled is False
    assert [entry.name for entry in pool.enabled_models()] == ["qwen3-4b-local"]


def test_disabled_entry_still_validates_loudly(tmp_path: Path) -> None:
    # The toggle removes an entry from selection, not from validation: a typo in a disabled
    # entry must not rot silently until the flag is flipped back.
    broken = _LOCAL_POOL.replace('model = "claude-fable-5"', 'model = ""')
    with pytest.raises(ValueError, match="model"):
        load_pool(_write_pool(tmp_path, broken))


@pytest.mark.parametrize(
    ("endpoint", "local"),
    [
        ("http://localhost:11434/v1", True),
        ("http://127.0.0.1:8001/v1", True),
        ("http://host.docker.internal:11434/v1", True),
        ("http://silens-mac.local:11434/v1", True),
        ("https://api.openai.com/v1", False),
        ("https://40.80.93.150:8443", False),
        (None, False),
        ("", False),
    ],
)
def test_is_local_endpoint(endpoint: str | None, local: bool) -> None:
    assert pool_module.is_local_endpoint(endpoint) is local


def test_an_endpoint_backed_entry_never_inherits_a_hosted_price() -> None:
    # "gpt-5.4" has a built-in (hosted) price, but this entry is served by a custom endpoint:
    # the hosted rate is the wrong server's price, so the entry must demand its own.
    with pytest.raises(ValidationError, match="custom endpoint"):
        PoolEntry(
            name="gpt-at-home",
            kind=ProviderKind.OPENAI,
            model="gpt-5.4",
            endpoint="http://localhost:8001/v1",
        )


def test_is_local_endpoint_never_raises_on_a_malformed_bracket_url() -> None:
    # urlsplit raises ValueError on `http://[::1:8000`; display copy must read it as
    # "not local", not crash the roster table.
    assert pool_module.is_local_endpoint("http://[::1:8000/v1") is False


def test_pool_entry_threads_reasoning_effort_into_the_provider_config() -> None:
    """Two entries differing only in effort are two arms over one runtime model."""
    entry = PoolEntry(
        name="sonnet-5@max",
        kind=ProviderKind.ANTHROPIC,
        model="claude-sonnet-5",
        reasoning_effort="max",
    )

    assert entry.provider_config().reasoning_effort == "max"
    assert (
        PoolEntry(name="sonnet-5", kind=ProviderKind.ANTHROPIC, model="claude-sonnet-5")
        .provider_config()
        .reasoning_effort
        is None
    )


def test_pool_entry_preserves_openai_xhigh_reasoning_effort() -> None:
    entry = PoolEntry(
        name="gpt-5.4@xhigh",
        kind=ProviderKind.OPENAI,
        model="gpt-5.4",
        reasoning_effort="xhigh",
    )

    assert entry.provider_config().reasoning_effort == "xhigh"
    assert entry.model_dump()["reasoning_effort"] == "xhigh"


def test_reasoning_effort_validates_at_load_not_first_request() -> None:
    """A bad effort value or a Bedrock effort entry must fail when the pool loads."""
    with pytest.raises(ValidationError, match="reasoning_effort"):
        # model_validate: the wire shape a TOML actually carries.
        PoolEntry.model_validate(
            {
                "name": "sonnet-5@warp",
                "kind": "anthropic",
                "model": "claude-sonnet-5",
                "reasoning_effort": "warp",
            }
        )
    with pytest.raises(ValidationError, match="Converse has no effort dial"):
        PoolEntry(
            name="opus-4-8@max",
            kind=ProviderKind.BEDROCK,
            model="us.anthropic.claude-opus-4-8",
            reasoning_effort="max",
        )
    with pytest.raises(ValidationError, match="supports effort through max"):
        PoolEntry(
            name="sonnet-5@xhigh",
            kind=ProviderKind.ANTHROPIC,
            model="claude-sonnet-5",
            reasoning_effort="xhigh",
        )
