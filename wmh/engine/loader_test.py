"""Tests for integrity checks performed while loading a world-model artifact."""

from __future__ import annotations

import importlib
import warnings
from pathlib import Path

from wmh.config import HarnessConfig, save_config, write_manifest
from wmh.providers.base import ProviderConfig, ProviderKind

loader_module = importlib.import_module("wmh.engine.loader")


def _artifact(root: Path) -> Path:
    """Create the minimal config artifact needed before the loader is mocked."""
    config = HarnessConfig(
        providers=[ProviderConfig(kind=ProviderKind.BEDROCK, model="test-model")],
        serve_provider=ProviderKind.BEDROCK,
    )
    save_config(config, root=root)
    payload = root / "prompts" / "optimized.txt"
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_text("prompt", encoding="utf-8")
    write_manifest(root, [root / "config.toml", payload])
    return payload


def test_load_world_model_is_silent_for_a_clean_manifest(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    """A verified artifact reaches the normal loader without an integrity warning."""
    _artifact(tmp_path)
    provider = object()
    monkeypatch.setattr(loader_module, "provider_or_chain", lambda config: provider)
    monkeypatch.setattr(loader_module.WorldModel, "load", lambda *args, **kwargs: object())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model, loaded_provider = loader_module.load_world_model(tmp_path)

    assert model is not None
    assert loaded_provider is provider
    assert not caught


def test_load_world_model_warns_for_a_tampered_manifest(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    """A bad artifact remains loadable but warns operators before it can serve traffic."""
    config = _artifact(tmp_path)
    config.write_text("tampered", encoding="utf-8")
    monkeypatch.setattr(loader_module, "provider_or_chain", lambda config: object())
    monkeypatch.setattr(loader_module.WorldModel, "load", lambda *args, **kwargs: object())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loader_module.load_world_model(tmp_path)

    assert len(caught) == 1
    assert "artifact integrity check failed" in str(caught[0].message)
