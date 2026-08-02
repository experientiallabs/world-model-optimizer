"""Tests for the pinned mini-swe-agent Responses adapter."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    path = Path(__file__).with_name("coding_model_router_verifiers_responses_adapter.py")
    spec = importlib.util.spec_from_file_location("verifiers_responses_adapter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adapt_source_selects_responses_model_class() -> None:
    module = _load_module()
    source = "before\n" + module.UPSTREAM_MODEL_CLASS_LINE + "\nafter\n"
    adapted = module.adapt_source(source)
    assert module.UPSTREAM_MODEL_CLASS_LINE not in adapted
    assert module.ADAPTED_MODEL_CLASS_LINE in adapted
    assert adapted.startswith("before\n")
    assert adapted.endswith("\nafter\n")


def test_adapt_source_rejects_unpinned_shape() -> None:
    module = _load_module()
    try:
        module.adapt_source("class MiniSWEAgentHarness:\n    pass\n")
    except ValueError as error:
        assert "model-class line" in str(error)
    else:
        raise AssertionError("adapter accepted an unpinned harness")
