"""Tests for selected model-effort confirmation collection."""

from __future__ import annotations

import coding_model_router_model_effort_confirm_collect as selected
import coding_model_router_swerebench_collect as collector


def test_configure_binds_exact_selected_arm() -> None:
    original_efforts = collector.EFFORTS
    original_phase = collector.DEVELOPMENT_PHASE
    try:
        arm = selected.configure("gpt-5.6-luna", "max")
        assert arm == "luna-max"
        assert collector.EFFORTS == ("max",)
        assert collector.DEVELOPMENT_PHASE.requires_authorization is True
        assert collector.DEVELOPMENT_PHASE.model == "gpt-5.6-luna"
    finally:
        collector.EFFORTS = original_efforts
        collector.DEVELOPMENT_PHASE = original_phase
