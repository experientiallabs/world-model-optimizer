"""Focused tests for the graded confirmation collector phase."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).parents[1] / ".agents" / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "coding_model_router_graded_swerebench_confirm_collect",
    SCRIPTS / "coding_model_router_graded_swerebench_confirm_collect.py",
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_confirmation_collection_is_separate_from_development() -> None:
    assert module.PROTOCOL.endswith("confirmation-collection-v1")
    assert module.EXECUTION_PROTOCOL.endswith("confirmation-execution-v1")
    assert module.CORPUS_SHA256 != module.collector.CORPUS_SHA256
