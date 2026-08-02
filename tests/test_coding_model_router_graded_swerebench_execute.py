"""Focused tests for the graded SWE-rebench executor."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).parents[1] / ".agents" / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "coding_model_router_graded_swerebench_execute",
    SCRIPTS / "coding_model_router_graded_swerebench_execute.py",
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _call(model: str = "gpt-5.6-luna", effort: str = "high") -> dict[str, object]:
    return {
        "model": model,
        "endpoint": "/responses",
        "sampling": {"reasoning_effort": effort, "max_tokens": 32768},
        "usage": {
            "prompt_tokens": 10,
            "cached_input_tokens": 20,
            "completion_tokens": 30,
            "reasoning_tokens": 15,
        },
    }


def _run_validator(
    tmp_path: Path,
    *,
    reward: float,
    f2p_total: int = 4,
    patch: str | None = "diff",
) -> subprocess.CompletedProcess[str]:
    validator = tmp_path / "validate.py"
    traces = tmp_path / "traces.jsonl"
    report = tmp_path / "report.json"
    validator.write_text(module.REMOTE_VALIDATOR, encoding="utf-8")
    trace = {
        "task": {
            "data": {
                "name": "owner__repo-1",
                "fail_to_pass": [f"test-{index}" for index in range(f2p_total)],
            }
        },
        "verifiers": {"commit": module.VERIFIERS_COMMIT},
        "rewards": {"solved": {"score": reward}},
        "timing": {"scoring": {"start": 1.0, "end": 2.0}},
        "calls": [_call()],
        "info": {"patch": patch},
        "ok": True,
        "errors": [],
        "stop_condition": "agent_completed",
    }
    traces.write_text(json.dumps({"ok": True, "traces": [trace]}) + "\n")
    return subprocess.run(
        [
            sys.executable,
            str(validator),
            "--traces",
            str(traces),
            "--task",
            "owner__repo-1",
            "--arm",
            "luna-high",
            "--model",
            "gpt-5.6-luna",
            "--effort",
            "high",
            "--f2p-total",
            str(f2p_total),
            "--output",
            str(report),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_validator_accepts_graded_f2p_fraction(tmp_path: Path) -> None:
    result = _run_validator(tmp_path, reward=0.75)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["reward"] == 0.75
    assert report["f2p_passed"] == 3
    assert report["f2p_total"] == 4
    assert report["reward_provenance"] == "official graded F2P verifier"


def test_validator_rejects_reward_inconsistent_with_denominator(tmp_path: Path) -> None:
    result = _run_validator(tmp_path, reward=0.7)
    assert result.returncode != 0
    assert "inconsistent with the F2P denominator" in result.stderr


def test_validator_accepts_officially_scored_no_change_trace(tmp_path: Path) -> None:
    result = _run_validator(tmp_path, reward=0.0, patch=None)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["patch_bytes"] == 0
    assert report["patch_provenance"] == "official trace reported no source changes"


def test_config_freezes_one_local_task_and_model_effort() -> None:
    config = module._config(
        module.Arm("sol-max", "gpt-5.6-sol", "max"),
        "/home/user/task.json",
        "/home/user/output",
    )
    assert 'model = "gpt-5.6-sol"' in config
    assert 'reasoning_effort = "max"' in config
    assert "num_rollouts = 1" in config
    assert 'task_json = "/home/user/task.json"' in config
    assert "filter_fn" not in config


def test_arm_order_rotates_without_changing_roster() -> None:
    assert module._arm_order(0) == module.ARMS
    assert module._arm_order(1)[0] == module.ARMS[1]
    assert set(module._arm_order(5)) == set(module.ARMS)


def test_exclusion_requires_one_lost_cell_and_zero_reruns() -> None:
    state = {
        "stage": "excluded-audit-artifact-loss",
        "exclusion": {
            "scope": "whole-task",
            "reason": "validator rejected official no-change trace",
            "arm": "luna-medium",
            "observed_scientific_cells": 1,
            "scientific_cells_rerun": 0,
            "provider_usage_recoverable": False,
        },
    }
    assert module._excluded(state)
    state["exclusion"]["scientific_cells_rerun"] = 1
    assert not module._excluded(state)
