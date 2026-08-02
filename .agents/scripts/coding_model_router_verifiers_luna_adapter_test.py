"""Tests for the pinned verifiers Luna compatibility adapter."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    path = Path(__file__).with_name("coding_model_router_verifiers_luna_adapter.py")
    spec = importlib.util.spec_from_file_location("verifiers_luna_adapter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adapt_source_maps_only_luna_token_limit() -> None:
    module = _load_module()
    adapted = module.adapt_source("prefix\n" + module.UPSTREAM_SOURCE + "suffix\n")
    assert 'if model == "gpt-5.6-luna":' in adapted
    assert 'body.pop("max_tokens", None)' in adapted
    assert 'overrides["max_completion_tokens"] = max_tokens' in adapted
    assert adapted.startswith("prefix\n")
    assert adapted.endswith("suffix\n")


def test_adapt_source_rejects_unpinned_shape() -> None:
    module = _load_module()
    try:
        module.adapt_source("class ChatDialect:\n    pass\n")
    except ValueError as error:
        assert "override block" in str(error)
    else:
        raise AssertionError("adapter accepted an unpinned ChatDialect")
