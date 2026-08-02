"""Tests for strict SWE-rebench development trace validation."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import coding_model_router_swerebench_execute as execute


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _call() -> dict[str, object]:
    return {
        "model": "gpt-5.6-luna",
        "endpoint": "/responses",
        "sampling": {"reasoning_effort": "xhigh", "max_tokens": 32768},
        "usage": {
            "prompt_tokens": 10,
            "cached_input_tokens": 20,
            "completion_tokens": 30,
            "reasoning_tokens": 15,
        },
    }


def _run_validator(tmp_path: Path, trace: dict[str, object]) -> subprocess.CompletedProcess[str]:
    validator = tmp_path / "validate.py"
    traces = tmp_path / "traces.jsonl"
    report = tmp_path / "report.json"
    validator.write_text(execute.REMOTE_VALIDATOR, encoding="utf-8")
    traces.write_text(
        json.dumps(
            {"ok": False, "errors": [], "traces": [trace]},
            ensure_ascii=False,
        )
        + "\n"
    )
    return subprocess.run(
        [
            sys.executable,
            str(validator),
            "--traces",
            str(traces),
            "--task",
            "owner__repo-1",
            "--effort",
            "xhigh",
            "--expected",
            "1",
            "--attempt-offset",
            "0",
            "--output",
            str(report),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _timeout_trace() -> dict[str, object]:
    return {
        "task": {"data": {"name": "owner__repo-1"}},
        "verifiers": {"commit": "f6e420b9908ae14d625f079881f13c15011ee1c9"},
        "rewards": {},
        "timing": {"scoring": {"start": 0.0, "end": 0.0}},
        "calls": [_call()],
        "info": {"patch": None},
        "ok": False,
        "errors": [
            {"type": "HarnessError", "message": "agent timeout: rollout exceeded its 900s budget"}
        ],
        "stop_condition": "error",
    }


def test_validator_accepts_post_execution_agent_timeout_as_zero(tmp_path: Path) -> None:
    result = _run_validator(tmp_path, _timeout_trace())
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    cell = report["cells"][0]
    assert cell["reward"] == 0.0
    assert cell["reward_provenance"] == "gradeable post-execution agent timeout"
    assert cell["official_verifier_reached"] is False
    assert cell["patch_sha256"] is None
    assert cell["scoring_seconds"] is None
    max_turns = _timeout_trace()
    max_turns["stop_condition"] = "max_turns"
    result = _run_validator(tmp_path, max_turns)
    assert result.returncode == 0, result.stderr


def test_validator_rejects_unrecognized_unscored_error(tmp_path: Path) -> None:
    trace = _timeout_trace()
    trace["errors"] = [{"type": "HarnessError", "message": "unexpected failure"}]
    result = _run_validator(tmp_path, trace)
    assert result.returncode != 0
    assert "lacks an official binary reward" in result.stderr


def test_validator_accepts_post_execution_mini_swe_agent_exit_137_as_zero(
    tmp_path: Path,
) -> None:
    trace = _timeout_trace()
    trace["errors"] = [
        {
            "type": "HarnessError",
            "message": (
                "harness 'mini_swe_agent' exited 137: "
                "Warning: Input is not a terminal (fd=0)."
            ),
        }
    ]
    result = _run_validator(tmp_path, trace)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    cell = report["cells"][0]
    assert cell["reward"] == 0.0
    assert cell["reward_provenance"] == (
        "gradeable post-execution mini-swe-agent exit 137"
    )
    assert cell["official_verifier_reached"] is False


def test_validator_rejects_other_harness_exit_codes(tmp_path: Path) -> None:
    trace = _timeout_trace()
    trace["errors"] = [
        {"type": "HarnessError", "message": "harness 'mini_swe_agent' exited 1"}
    ]
    result = _run_validator(tmp_path, trace)
    assert result.returncode != 0
    assert "lacks an official binary reward" in result.stderr


def test_validator_preserves_unicode_line_separator_inside_patch(tmp_path: Path) -> None:
    trace = {
        "task": {"data": {"name": "owner__repo-1"}},
        "verifiers": {"commit": "f6e420b9908ae14d625f079881f13c15011ee1c9"},
        "rewards": {"solved": {"score": 1.0}},
        "timing": {"scoring": {"start": 1.0, "end": 2.0}},
        "calls": [_call()],
        "info": {"patch": "before\u2028after"},
        "ok": True,
        "errors": [],
        "stop_condition": "agent_completed",
    }
    result = _run_validator(tmp_path, trace)
    assert result.returncode == 0, result.stderr


def test_whole_task_exclusion_requires_audited_zero_reruns() -> None:
    state = {
        "stage": "excluded-infrastructure",
        "exclusion": {
            "scope": "whole-task",
            "effort": "low",
            "reason": "official verifier scoring timeout after completed inference",
            "evidence_sha256": "a" * 64,
            "usage": {},
            "provider_calls": 2,
            "observed_scientific_cells": 2,
            "scientific_cells_rerun": 0,
        },
    }
    assert execute._task_excluded(state)
    state["exclusion"]["scientific_cells_rerun"] = 1
    assert not execute._task_excluded(state)


def test_durable_eval_starts_once_and_polls_atomic_exit_marker(tmp_path: Path) -> None:
    class FakeCommands:
        def __init__(self) -> None:
            self.starts: list[str] = []
            self.polls = 0

        def run(self, command: str, *, timeout: int) -> SimpleNamespace:
            if "nohup bash" in command:
                assert timeout == 120
                self.starts.append(command)
                return SimpleNamespace(stdout="314\n")
            assert timeout == 60
            self.polls += 1
            return SimpleNamespace(stdout="stopped")

    class FakeFiles:
        def __init__(self, commands: FakeCommands) -> None:
            self.commands = commands
            self.writes: dict[str, str] = {}

        def write(self, path: str, content: str) -> None:
            self.writes[path] = content

        def exists(self, path: str, *, request_timeout: int) -> bool:
            assert path == "/remote/xhigh.eval-exit-status"
            assert request_timeout == 60
            return self.commands.polls > 0

        def read(self, path: str) -> str:
            assert path == "/remote/xhigh.eval-exit-status"
            return "0\n"

    commands = FakeCommands()
    files = FakeFiles(commands)
    sandbox = SimpleNamespace(
        sandbox_id="sandbox-1",
        commands=commands,
        files=files,
    )
    state = {"stage": "running-xhigh"}
    attempt: dict[str, object] = {}
    state_path = tmp_path / "state.json"

    result, active = execute._run_durable_eval(
        sandbox,
        "scientific-eval --frozen",
        effort="xhigh",
        exit_status_path="/remote/xhigh.eval-exit-status",
        state=state,
        state_path=state_path,
        attempt=attempt,
        timeout=5,
        poll_interval=0,
    )

    assert active is sandbox
    assert result.exit_code == 0
    assert len(commands.starts) == 1
    assert commands.starts[0].count("nohup bash") == 1
    wrapper = files.writes["/remote/xhigh.eval-exit-status.wrapper.sh"]
    assert wrapper.count("scientific-eval --frozen") == 1
    process = attempt["effort_processes"]["xhigh"]
    assert process["pid"] == 314
    assert process["scientific_command_starts"] == 1
    assert process["completed"] is True


def test_confirmation_never_reuses_development_smoke_cells() -> None:
    task_id = next(iter(execute.REUSED_TASKS))
    assert execute._new_rollouts(task_id, "xhigh") == (1, 1)
    assert execute._new_rollouts(task_id, "xhigh", reuse_smoke=False) == (2, 0)


def test_pooled_confirmation_has_frozen_phase_and_raised_concurrency() -> None:
    phase = execute._execution_phase("pooled-confirmation")
    assert phase is execute.POOLED_CONFIRMATION_PHASE
    assert phase.metadata_owner == "coding-router-v42"
    assert phase.reuse_smoke is False
    assert execute._max_concurrency(phase) == 200
    assert execute._max_concurrency(execute.CONFIRMATION_PHASE) == 100


def test_recovery_helpers_can_load_executor_without_sys_modules_registration() -> None:
    path = Path(execute.__file__)
    spec = importlib.util.spec_from_file_location("detached_swerebench_execute", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.DEVELOPMENT_PHASE.protocol == execute.PROTOCOL


def test_confirmation_authorization_is_content_addressed(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    corpus = tmp_path / "confirmation.json"
    tasks = [
        {
            "task_id": f"owner__repo-{index}",
            "repository": f"owner/repo-{index}",
            "prompt_sha256": f"{index:064x}",
        }
        for index in range(200)
    ]
    _write_json(corpus, {"tasks": tasks})
    corpus_hash = execute._sha256(corpus)
    monkeypatch.setattr(execute, "CONFIRMATION_CORPUS_SHA256", corpus_hash)

    development_audit = tmp_path / "completion-audit.json"
    _write_json(
        development_audit,
        {
            "valid": True,
            "retained_task_coverage": 0.98,
            "rough_cumulative_experiment_spend_usd": 638.5,
            "target_outcomes_used": False,
            "deep_swe_outcomes_accessed": False,
        },
    )
    audit_hash = execute._sha256(development_audit)
    fit_output = tmp_path / "fit"
    fit_output.mkdir()
    lock = fit_output / "selection-lock.json"
    _write_json(
        lock,
        {
            "collection_audit_sha256": audit_hash,
            "confirmation_corpus_sha256": corpus_hash,
            "target_outcomes_used": False,
            "deep_swe_outcomes_accessed": False,
            "confirmation_outcomes_accessed": False,
        },
    )
    rows = [
        {
            "task_id": task["task_id"],
            "reasoning_effort": "high",
            "target_outcomes_used": False,
            "deep_swe_outcomes_accessed": False,
        }
        for task in tasks
    ]
    routes = fit_output / "confirmation-routes.jsonl"
    shuffled = fit_output / "confirmation-shuffled-routes.jsonl"
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    routes.write_text(content, encoding="utf-8")
    shuffled.write_text(content, encoding="utf-8")
    _write_json(
        fit_output / "route-audit.json",
        {
            "selection_lock_sha256": execute._sha256(lock),
            "confirmation_routes_sha256": execute._sha256(routes),
            "shuffled_routes_sha256": execute._sha256(shuffled),
            "latency": {"passed": True, "p95_ms": 1.0},
            "fitted_numeric_state_persisted": False,
            "target_outcomes_used": False,
            "deep_swe_outcomes_accessed": False,
            "confirmation_outcomes_accessed": False,
        },
    )
    _write_json(
        fit_output / "development-report.json",
        {
            "development_passed": True,
            "confirmation_authorized": True,
            "confirmation_routes_written": True,
            "target_outcomes_used": False,
            "deep_swe_outcomes_accessed": False,
            "confirmation_outcomes_accessed": False,
            "inputs": {
                "collection_audit_sha256": audit_hash,
                "confirmation_corpus_sha256": corpus_hash,
            },
        },
    )

    spend, hashes = execute._confirmation_authorization(
        fit_output,
        development_audit,
        corpus,
    )

    assert spend == 638.5
    assert hashes["development_audit_sha256"] == audit_hash
    assert hashes["confirmation_corpus_sha256"] == corpus_hash
