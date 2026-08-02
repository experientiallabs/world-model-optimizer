"""Tests for zero-rerun pooled-confirmation worker exclusion."""

from __future__ import annotations

import coding_model_router_pooled_confirmation_exclude as exclude


def _state() -> dict[str, object]:
    return {
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


def test_report_seals_zero_trace_worker_without_rerun() -> None:
    report = exclude._report(_state(), "owner__repo-1", "low")
    assert report["valid"] is True
    assert report["observed_scientific_cells"] == 0
    assert report["scientific_cells_rerun"] == 0
    assert report["provider_calls"] == 0
    assert report["usage_provenance"].startswith("zero trace-derived lower bound")


def test_report_rejects_a_second_scientific_start() -> None:
    state = _state()
    state["sandbox_attempts"][0]["effort_processes"]["low"][
        "scientific_command_starts"
    ] = 2
    try:
        exclude._report(state, "owner__repo-1", "low")
    except ValueError as error:
        assert "zero-trace" in str(error)
    else:
        raise AssertionError("a repeated scientific command was accepted")
