"""Tests for cross-model matrix merge validation."""

from __future__ import annotations

import coding_model_router_model_effort_merge as merge
import pytest


def test_validate_row_accepts_model_effort_identity() -> None:
    identity = merge._validate_row(
        {
            "task_id": "owner__repo-1",
            "arm": "terra-xhigh",
            "attempt_number": 1,
            "model": "gpt-5.6-terra",
            "reward": 1.0,
            "cost_usd": 0.25,
            "target_outcomes_used": False,
        },
        "terra",
    )
    assert identity == ("owner__repo-1", "terra-xhigh", 1)


def test_validate_row_rejects_collapsed_model_identity() -> None:
    with pytest.raises(ValueError, match="invalid sol outcome identity"):
        merge._validate_row(
            {
                "task_id": "owner__repo-1",
                "arm": "sol-high",
                "attempt_number": 0,
                "model": "gpt-5.6-luna",
                "reward": 0.0,
                "cost_usd": 0.1,
                "target_outcomes_used": False,
            },
            "sol",
        )
