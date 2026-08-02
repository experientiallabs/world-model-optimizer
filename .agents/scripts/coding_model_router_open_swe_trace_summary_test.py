"""Tests for compact Open-SWE trajectory burden summaries."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _module() -> ModuleType:
    path = Path(__file__).with_name("coding_model_router_open_swe_trace_summary.py")
    spec = importlib.util.spec_from_file_location(
        "coding_model_router_open_swe_trace_summary",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row() -> dict[str, object]:
    return {
        "instance_id": "owner__repo-1",
        "repo": "owner/repo",
        "language": "python",
        "trajectory_id": "trace-1",
        "resolved": 1,
        "trajectory": [
            {
                "role": "assistant",
                "content": "secret analysis",
                "reasoning_content": "private reasoning",
                "tool_calls": [
                    {
                        "function": {
                            "name": "terminal",
                            "arguments": "pytest -q secret_test.py",
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "edit",
                "reasoning_content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "terminal",
                            "arguments": "pytest -q secret_test.py",
                        }
                    },
                    {
                        "function": {
                            "name": "str_replace_editor",
                            "arguments": "private patch",
                        }
                    },
                ],
            },
        ],
        "metadata": {
            "reference_patch": {"patch": "must not survive", "num_modified_lines": 99},
            "model_patch": {
                "patch": "also secret",
                "num_modified_files": 2,
                "num_modified_lines": 7,
            },
        },
    }


def test_summary_keeps_counts_and_discards_raw_content() -> None:
    module = _module()
    summary = module._summarize_row(
        _row(),
        source_path="data/minimax_m25_openhands_trajectories/train-00000-of-00020.parquet",
        source_sha256="a" * 64,
        source_row=3,
    )
    assert summary["scaffold"] == "openhands"
    assert summary["model_mode"] == "minimax_m25"
    assert summary["messages"] == 2
    assert summary["assistant_turns"] == 2
    assert summary["tool_calls"] == 3
    assert summary["distinct_tools"] == 2
    assert summary["repeated_calls"] == 1
    assert summary["max_repeated_call_run"] == 2
    assert summary["test_calls"] == 2
    assert summary["edit_calls"] == 1
    assert summary["model_patch_files"] == 2
    assert summary["model_patch_lines"] == 7
    serialized = json.dumps(summary)
    for forbidden in ("secret analysis", "private reasoning", "private patch", "must not survive"):
        assert forbidden not in serialized


def test_summary_rejects_ungradeable_rows() -> None:
    module = _module()
    row = _row()
    row["resolved"] = None
    with pytest.raises(ValueError, match="resolved"):
        module._summarize_row(
            row,
            source_path="data/qwen35_sweagent_trajectories/train-00000-of-00018.parquet",
            source_sha256="b" * 64,
            source_row=0,
        )


def test_partition_rejects_unknown_layout() -> None:
    module = _module()
    with pytest.raises(ValueError, match="unrecognized"):
        module._partition("data/unknown/train.parquet")
