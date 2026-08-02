"""Focused tests for graded confirmation authorization."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).parents[1] / ".agents" / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "coding_model_router_graded_swerebench_confirm",
    SCRIPTS / "coding_model_router_graded_swerebench_confirm.py",
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_confirmation_uses_separate_frozen_phase() -> None:
    assert module.PROTOCOL.endswith("confirmation-execution-v1")
    assert module.TASKS == 320
    assert module.CORPUS_SHA256 != module.runner.CORPUS_SHA256
