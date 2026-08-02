"""Focused no-spend tests for the coding-router world-model pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

import pytest
from coding_model_router_world_model import (
    AZURE_DEPLOYMENT_ENV,
    AZURE_ENDPOINT_ENV,
    AZURE_KEY_ENV,
    _azure_provider,
    _build,
    _complete_event,
    _ledger_rows,
    _persisted_config,
    _reserve_event,
    _simulated_outcome,
    _spent_reserved,
)

from wmo.engine.world_model import WorldModel
from wmo.env.closed_loop import PoolCell
from wmo.env.scenarios import Scenario
from wmo.optimize.outcomes import ScenarioOutcome
from wmo.providers.base import (
    Completion,
    Message,
    Provider,
    ProviderConfig,
    ProviderKind,
    TokenUsage,
    VerifyResult,
)
from wmo.providers.pool import PoolEntry


class _NoCallProvider:
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model="gpt-5.5",
        deployment="unit-test-deployment",
    )

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        raise AssertionError("the low-fidelity build must not call the provider")

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("the low-fidelity build uses the injected hashing embedder")

    def verify(self) -> VerifyResult:
        raise AssertionError("the direct build path must not issue a verification ping")


def _freeze(root: Path, ceiling: float) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "freeze-summary.json").write_text(
        json.dumps({"spend_ceiling_usd": ceiling}),
        encoding="utf-8",
    )


def test_persisted_config_keeps_azure_location_and_secret_values_out() -> None:
    rendered = _persisted_config().model_dump(mode="json")
    provider = cast("list[dict[str, object]]", rendered["providers"])[0]
    assert provider["deployment"] == f"env:{AZURE_DEPLOYMENT_ENV}"
    assert provider["endpoint"] is None
    assert "api_key" not in json.dumps(rendered).lower()


def test_azure_provider_passes_runtime_config_without_mutating_generic_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(AZURE_DEPLOYMENT_ENV, "configured-deployment")
    monkeypatch.setenv(AZURE_ENDPOINT_ENV, "https://configured.example")
    monkeypatch.setenv(AZURE_KEY_ENV, "configured-key")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    captured: dict[str, object] = {}

    def fake_get_provider(
        config: ProviderConfig,
        *,
        api_key: str | None = None,
    ) -> Provider:
        captured["config"] = config
        captured["api_key"] = api_key
        return cast("Provider", _NoCallProvider())

    monkeypatch.setattr(
        "coding_model_router_world_model.get_provider",
        fake_get_provider,
    )
    _azure_provider()

    config = cast("ProviderConfig", captured["config"])
    assert config.endpoint == "https://configured.example"
    assert config.deployment == "configured-deployment"
    assert captured["api_key"] == "configured-key"
    assert "AZURE_OPENAI_ENDPOINT" not in os.environ
    assert "AZURE_OPENAI_API_KEY" not in os.environ


def test_spend_ledger_requires_ceiling_reservation_and_single_completion(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="no authorized positive spend ceiling"):
        _reserve_event(tmp_path, "event", 5.0, "test")

    _freeze(tmp_path, 10.0)
    _reserve_event(
        tmp_path,
        "event",
        5.0,
        "test",
        provider="openai",
        model="test-model",
        benchmark="test-benchmark",
    )
    rows = _ledger_rows(tmp_path / "spend-ledger.jsonl")
    assert _spent_reserved(rows) == (0.0, 5.0)
    assert rows[0]["provider"] == "openai"
    assert rows[0]["model"] == "test-model"
    assert rows[0]["benchmark"] == "test-benchmark"

    with pytest.raises(ValueError, match="already exists"):
        _reserve_event(tmp_path, "event", 5.0, "test")

    _complete_event(
        tmp_path,
        "event",
        {"phase": "test", "model_cost_usd": 1.25},
    )
    rows = _ledger_rows(tmp_path / "spend-ledger.jsonl")
    assert _spent_reserved(rows) == (1.25, 0.0)
    with pytest.raises(ValueError, match="no active spend reservation"):
        _complete_event(
            tmp_path,
            "event",
            {"phase": "test", "model_cost_usd": 1.25},
        )


def test_reservation_refuses_to_cross_frozen_ceiling(tmp_path: Path) -> None:
    _freeze(tmp_path, 4.99)
    with pytest.raises(ValueError, match=r"above the frozen \$4.99 ceiling"):
        _reserve_event(tmp_path, "event", 5.0, "test")


def test_low_fidelity_build_is_zero_call_and_releases_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _freeze(tmp_path, 100.0)
    world_root = tmp_path / "world-model"
    corpus = world_root / "corpus.otel.jsonl"
    corpus.parent.mkdir()
    corpus.write_text('{"span": 1}\n', encoding="utf-8")

    from coding_model_router_world_model import _sha256

    (world_root / "prepare.json").write_text(
        json.dumps({"corpus_sha256": _sha256(corpus)}),
        encoding="utf-8",
    )

    def fake_build(*args: object, **kwargs: object) -> None:
        artifact = Path(cast("str", kwargs["root"]))
        artifact.mkdir(parents=True)
        (artifact / "config.toml").write_text("top_k = 5\n", encoding="utf-8")

    monkeypatch.setattr(
        "coding_model_router_world_model._azure_provider",
        lambda: cast("Provider", _NoCallProvider()),
    )
    monkeypatch.setattr(
        "coding_model_router_world_model.build_world_model",
        fake_build,
    )

    _build(tmp_path)

    usage = json.loads((world_root / "build-usage.json").read_text(encoding="utf-8"))
    assert usage["total"]["calls"] == 0
    rows = _ledger_rows(tmp_path / "spend-ledger.jsonl")
    assert _spent_reserved(rows) == (0.0, 0.0)
    assert rows[0]["completion_status"] == "built"


def test_simulated_cell_uses_request_level_long_context_pricing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = PoolEntry(
        name="oai-gpt55-high",
        kind=ProviderKind.OPENAI_RESPONSES,
        model="gpt-5.5-2026-04-23",
        model_type="gpt-5.5",
        input_per_mtok=5.0,
        cached_input_per_mtok=0.5,
        output_per_mtok=30.0,
    )
    cell = PoolCell(
        entry=entry,
        scenario=Scenario(
            task="repair the repository",
            provenance=["terminal-bench-2:task"],
        ),
        episode=0,
    )

    def fake_run_cell(*args: object, **kwargs: object) -> ScenarioOutcome:
        return ScenarioOutcome(
            scenario_id="terminal-bench-2:task",
            task="repair the repository",
            model=entry.name,
            reward=1.0,
            success=True,
            usage=TokenUsage(
                input_tokens=300_000,
                output_tokens=10_000,
                cached_input_tokens=100_000,
            ),
            call_seconds=[1.0],
            call_input_tokens=[300_000],
            call_output_tokens=[10_000],
            call_cached_input_tokens=[100_000],
            call_cache_write_input_tokens=[0],
        )

    monkeypatch.setattr(
        "coding_model_router_world_model.run_cell",
        fake_run_cell,
    )
    outcome, usage = _simulated_outcome(
        cell,
        world_model=cast("WorldModel", object()),
        tools_hint="bash",
        attempt=1,
    )

    assert usage is None
    assert outcome.cost_usd == pytest.approx(2.55)
    assert outcome.completion_status == "simulated_scored"
