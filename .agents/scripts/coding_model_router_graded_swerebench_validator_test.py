"""Tests for the standalone graded SWE-rebench trace validator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from coding_model_router_graded_swerebench_execute import REMOTE_VALIDATOR


def _call(
    *,
    usage: dict[str, int] | None,
    error: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "model": "gpt-5.6-sol",
        "endpoint": "/responses",
        "sampling": {"reasoning_effort": "max", "max_tokens": 32768},
        "usage": usage,
        "error": error,
        "request_summary": "repair the repository and run tests",
    }


def _validate(tmp_path: Path, calls: list[dict[str, object]]) -> dict[str, object]:
    script = tmp_path / "validate.py"
    traces = tmp_path / "traces.jsonl"
    output = tmp_path / "report.json"
    script.write_text(REMOTE_VALIDATOR, encoding="utf-8")
    trace = {
        "ok": True,
        "task": {"data": {"name": "example__repo-1", "fail_to_pass": ["test_one"]}},
        "verifiers": {"commit": "f6e420b9908ae14d625f079881f13c15011ee1c9"},
        "calls": calls,
        "rewards": {"solved": {"score": 1.0}},
        "errors": [],
        "stop_condition": "completed",
        "info": {"patch": ""},
        "timing": {"scoring": {"start": 1.0, "end": 2.0}},
    }
    traces.write_text(
        json.dumps({"ok": True, "traces": [trace], "errors": []}) + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--traces",
            str(traces),
            "--task",
            "example__repo-1",
            "--arm",
            "sol-max",
            "--model",
            "gpt-5.6-sol",
            "--effort",
            "max",
            "--f2p-total",
            "1",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    value = json.loads(output.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return {str(key): item for key, item in value.items()}


def test_official_reward_survives_missing_usage_with_labeled_estimate(tmp_path: Path) -> None:
    report = _validate(tmp_path, [_call(usage=None)])
    assert report["reward"] == 1.0
    assert report["provider_inference_calls"] == 1
    assert report["usage_provenance"] == (
        "mixed exact and conservative trace-derived token estimate"
    )
    estimated = report["estimated_usage_calls"]
    assert isinstance(estimated, list) and len(estimated) == 1
    usage = report["usage"]
    assert isinstance(usage, dict)
    prompt_tokens = usage.get("prompt_tokens")
    assert isinstance(prompt_tokens, int) and prompt_tokens > 4_096


def test_provider_rejection_keeps_zero_usage_without_hiding_exact_calls(tmp_path: Path) -> None:
    exact = {
        "prompt_tokens": 100,
        "cached_input_tokens": 10,
        "completion_tokens": 20,
        "reasoning_tokens": 5,
    }
    report = _validate(
        tmp_path,
        [
            _call(usage=exact),
            _call(usage=None, error={"type": "BadRequest", "status_code": 400}),
        ],
    )
    assert report["provider_inference_calls"] == 1
    assert report["estimated_usage_calls"] == []
    errors = report["provider_errors"]
    assert isinstance(errors, list) and len(errors) == 1
    usage = report["usage"]
    assert isinstance(usage, dict)
    assert usage.get("prompt_tokens") == 100
