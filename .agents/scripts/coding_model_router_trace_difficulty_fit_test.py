"""Focused tests for the external trace-difficulty fitter."""

from __future__ import annotations

import json

import coding_model_router_model_effort_fit as base
import coding_model_router_trace_difficulty_fit as trace_fit
import numpy as np


def test_initial_user_text_ignores_assistant_and_tool_content() -> None:
    raw = json.dumps(
        [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "wrapper\n<pr_description>task only</pr_description>\nmore",
                    }
                ],
            },
            {"role": "assistant", "content": "private reasoning"},
            {"role": "tool", "content": "repository output"},
        ]
    )
    assert trace_fit._initial_user_text(raw) == "task only"


def test_choices_routes_frozen_easiest_fraction_to_alternate() -> None:
    alternate = base.ARMS.index("luna-low")
    scores = np.asarray([0.1, 0.9, 0.3, 0.7], dtype=np.float64)
    choices = trace_fit._choices(scores, alternate, 50)
    assert choices.tolist() == [
        base.ARMS.index("sol-max"),
        alternate,
        base.ARMS.index("sol-max"),
        alternate,
    ]


def test_null_choices_preserve_repository_blocks_and_arm_traffic() -> None:
    choices = np.asarray([0, 0, 1, 1, 2, 3], dtype=np.int64)
    repositories = ["a", "a", "b", "b", "c", "d"]
    null = trace_fit._null_choices(choices, repositories, 20_260_801)
    assert sorted(null.tolist()) == sorted(choices.tolist())
    assert null[0] == null[1]
    assert null[2] == null[3]
