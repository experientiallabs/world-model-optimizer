"""Focused tests for graded SWE-rebench collection helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).parents[1] / ".agents" / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "coding_model_router_graded_swerebench_collect",
    SCRIPTS / "coding_model_router_graded_swerebench_collect.py",
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_cost_uses_model_specific_frozen_prices() -> None:
    usage = {
        "prompt_tokens": 1_000_000,
        "cached_input_tokens": 1_000_000,
        "completion_tokens": 1_000_000,
        "reasoning_tokens": 500_000,
    }
    assert module._cost("gpt-5.6-luna", usage) == 7.1
    assert module._cost("gpt-5.6-sol", usage) == 35.5


def test_exclusion_is_whole_task_and_never_rerun() -> None:
    state = {
        "stage": "excluded-audit-artifact-loss",
        "exclusion": {
            "scope": "whole-task",
            "reason": "validator rejected official no-change trace",
            "observed_scientific_cells": 1,
            "scientific_cells_rerun": 0,
            "provider_usage_recoverable": False,
        },
    }
    assert module._excluded(state)
    state["exclusion"]["scientific_cells_rerun"] = 1
    assert not module._excluded(state)


def test_arm_roster_preserves_reasoning_effort_axis() -> None:
    assert module.ARMS == (
        "luna-low",
        "luna-medium",
        "luna-high",
        "luna-xhigh",
        "luna-max",
        "sol-max",
    )
