"""Tests for `load_world_model`: it forwards the load knobs and returns the serve provider."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import wmh.engine.loader as loader_module
from wmh.config import HarnessConfig, save_config
from wmh.engine.loader import load_world_model
from wmh.providers.base import ProviderConfig, ProviderKind


def _bedrock_config() -> HarnessConfig:
    """A minimal config whose serve provider resolves (loader builds it before loading)."""
    return HarnessConfig(
        providers=[ProviderConfig(kind=ProviderKind.BEDROCK, model="m")],
        serve_provider=ProviderKind.BEDROCK,
    )


def _record_load(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], object]:
    """Patch out the provider factory and `WorldModel.load`; return the captured kwargs + provider.

    Keeps the test hermetic: `load_config` still runs on a real saved config, but no live provider
    is constructed and no artifact is loaded. The returned dict records exactly what the loader
    forwarded into `WorldModel.load`.
    """
    sentinel_provider = object()
    sentinel_wm = object()
    captured: dict[str, Any] = {}

    def fake_provider_or_chain(config: object) -> object:
        return sentinel_provider

    def fake_load(model_dir: str, provider: object, **kwargs: object) -> object:
        captured["model_dir"] = model_dir
        captured["provider"] = provider
        captured["kwargs"] = kwargs
        return sentinel_wm

    monkeypatch.setattr(loader_module, "provider_or_chain", fake_provider_or_chain)
    monkeypatch.setattr(loader_module.WorldModel, "load", staticmethod(fake_load))
    return captured, sentinel_provider


def test_forwards_knowledge_dir_and_returns_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / ".wmh"
    save_config(_bedrock_config(), root=root)
    captured, sentinel_provider = _record_load(monkeypatch)
    override = tmp_path / "run-knowledge"

    wm, provider = load_world_model(root, knowledge_dir=override, max_fidelity=True)

    assert provider is sentinel_provider
    assert wm is not None
    assert captured["kwargs"]["knowledge_dir"] == override
    assert captured["kwargs"]["max_fidelity"] is True


def test_defaults_knowledge_dir_to_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / ".wmh"
    save_config(_bedrock_config(), root=root)
    captured, _ = _record_load(monkeypatch)

    load_world_model(root)

    assert captured["kwargs"]["knowledge_dir"] is None
    assert captured["kwargs"]["max_fidelity"] is False
