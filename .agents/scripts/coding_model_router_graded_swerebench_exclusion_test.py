"""Tests for permanent whole-task exclusions in the graded experiment."""

from __future__ import annotations

from types import ModuleType

import coding_model_router_graded_swerebench_collect as collector
import coding_model_router_graded_swerebench_execute as executor
import pytest


@pytest.mark.parametrize("module", [collector, executor])
@pytest.mark.parametrize(
    ("stage", "reason"),
    [
        (
            "excluded-audit-artifact-loss",
            "validator rejected official no-change trace",
        ),
        (
            "excluded-ungradeable-scientific-cell",
            "official trace lacked a graded reward after one frozen attempt",
        ),
        (
            "excluded-ungradeable-scientific-cell",
            "scientific artifact became irrecoverable after E2B transport loss",
        ),
        (
            "excluded-ungradeable-scientific-cell",
            "official graded trace became irrecoverable after missing usage audit failure",
        ),
    ],
)
def test_permanent_whole_task_exclusion(module: ModuleType, stage: str, reason: str) -> None:
    state = {
        "stage": stage,
        "exclusion": {
            "scope": "whole-task",
            "reason": reason,
            "arm": "luna-high",
            "observed_scientific_cells": 1,
            "scientific_cells_rerun": 0,
            "provider_usage_recoverable": False,
        },
    }

    assert module._excluded(state) is True


@pytest.mark.parametrize("module", [collector, executor])
def test_exclusion_rejects_any_rerun(module: ModuleType) -> None:
    state = {
        "stage": "excluded-ungradeable-scientific-cell",
        "exclusion": {
            "scope": "whole-task",
            "reason": "official trace lacked a graded reward after one frozen attempt",
            "arm": "luna-high",
            "observed_scientific_cells": 1,
            "scientific_cells_rerun": 1,
            "provider_usage_recoverable": False,
        },
    }

    assert module._excluded(state) is False
