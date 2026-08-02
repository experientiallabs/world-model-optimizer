"""Tests for sealing a multi-model zero-trace worker failure."""

from __future__ import annotations

import coding_model_router_swerebench_multimodel_exclude as exclude


def test_report_accepts_exact_zero_trace_exit_137() -> None:
    state = {
        "stage": "failed",
        "task_id": "owner__repo-1",
        "error": "expected 2 outer rows, found 0",
        "efforts": {},
        "sandbox_attempts": [
            {
                "sandbox_id": "sandbox-1",
                "terminated": False,
                "effort_processes": {
                    "low": {
                        "scientific_command_starts": 1,
                        "completed": True,
                        "exit_code": 137,
                    }
                },
            }
        ],
    }
    report = exclude._report(state, "owner__repo-1", "low", "gpt-5.6-terra")
    assert report["durable_trace_rows"] == 0
    assert report["scientific_cells_rerun"] == 0
    assert report["model"] == "gpt-5.6-terra"
