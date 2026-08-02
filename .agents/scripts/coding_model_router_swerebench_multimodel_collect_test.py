"""Tests for generalized SWE-rebench matrix collection."""

from __future__ import annotations

import coding_model_router_swerebench_collect as collector
import coding_model_router_swerebench_multimodel_collect as multimodel


def test_configure_binds_sol_collection() -> None:
    original = collector.DEVELOPMENT_PHASE
    try:
        multimodel.configure("gpt-5.6-sol")
        phase = collector.DEVELOPMENT_PHASE
        assert phase.model == "gpt-5.6-sol"
        assert phase.arm_prefix == "sol"
        assert phase.requires_authorization is False
        assert phase.reuse_smoke is False
        assert collector._cost(
            {
                "prompt_tokens": 1_000_000,
                "cached_input_tokens": 1_000_000,
                "completion_tokens": 1_000_000,
                "reasoning_tokens": 0,
            },
            phase,
        ) == 35.5
    finally:
        collector.DEVELOPMENT_PHASE = original


def test_outcome_uses_model_and_arm_prefix() -> None:
    original = collector.DEVELOPMENT_PHASE
    try:
        multimodel.configure("gpt-5.6-terra")
        phase = collector.DEVELOPMENT_PHASE
        outcome = collector._outcome(
            {
                "task_id": "owner__repo-1",
                "repository": "owner/repo",
                "language": "python",
                "prompt": "Fix it",
                "prompt_sha256": "a" * 64,
            },
            "xhigh",
            0,
            {
                "reward": 1.0,
                "usage": {
                    "prompt_tokens": 10,
                    "cached_input_tokens": 20,
                    "completion_tokens": 30,
                    "reasoning_tokens": 10,
                },
            },
            provenance="test",
            phase=phase,
        )
        assert outcome["arm"] == "terra-xhigh"
        assert outcome["model"] == "gpt-5.6-terra"
    finally:
        collector.DEVELOPMENT_PHASE = original
