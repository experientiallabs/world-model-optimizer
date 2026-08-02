"""Offline regression tests for coding-router failure and evidence handling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import coding_model_router_matrix as matrix_runner
import coding_model_router_smoke as smoke_runner
import pytest

from wmo.harness.scoring import ScoreCell
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.providers.base import ProviderKind, TokenUsage
from wmo.providers.pool import ModelPool, PoolEntry


class _SandboxPage:
    def __init__(self, active_pages: list[int]) -> None:
        self.active_pages = active_pages
        self.index = 0

    @property
    def has_next(self) -> bool:
        return self.index < len(self.active_pages)

    def next_items(self) -> list[object]:
        active = self.active_pages[self.index]
        self.index += 1
        return [object()] * active


class _SandboxClient:
    page = _SandboxPage([0])

    @classmethod
    def list(cls, *, limit: int) -> _SandboxPage:
        assert limit == smoke_runner.E2B_LIST_PAGE_SIZE
        return cls.page


def test_expanded_e2b_account_cap_is_frozen_across_runners() -> None:
    assert smoke_runner.E2B_ACCOUNT_CAP == 1000
    assert matrix_runner.E2B_ACCOUNT_CAP == smoke_runner.E2B_ACCOUNT_CAP
    assert smoke_runner.E2B_LIST_PAGE_SIZE == 100
    assert matrix_runner.E2B_LIST_PAGE_SIZE == smoke_runner.E2B_LIST_PAGE_SIZE


def _entry() -> PoolEntry:
    return PoolEntry(
        name="test-arm",
        kind=ProviderKind.OPENAI,
        model="test-model",
        input_per_mtok=1.0,
        output_per_mtok=2.0,
    )


def test_smoke_capacity_gate_accepts_a_legitimate_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _SandboxClient.page = _SandboxPage([100] * 9 + [99])
    monkeypatch.setattr(smoke_runner, "Sandbox", _SandboxClient)

    smoke_runner._require_e2b_capacity()


@pytest.mark.parametrize(
    "active_pages",
    [
        [100] * 10,
        [100] * 10 + [1],
    ],
)
def test_smoke_capacity_gate_rejects_full_account(
    monkeypatch: pytest.MonkeyPatch,
    active_pages: list[int],
) -> None:
    _SandboxClient.page = _SandboxPage(active_pages)
    monkeypatch.setattr(smoke_runner, "Sandbox", _SandboxClient)

    with pytest.raises(RuntimeError, match="account cap"):
        smoke_runner._require_e2b_capacity()


def _artifact(root: Path) -> Path:
    artifact = root / "artifact"
    trace = artifact / "agent" / "wmo-run.json"
    trace.parent.mkdir(parents=True)
    trace.write_text(
        json.dumps(
            {
                "instruction": "repair the repository",
                "stop_reason": "error",
                "steps": [
                    {
                        "action": {
                            "kind": "message",
                            "content": "(pi runtime)",
                            "name": None,
                            "arguments": {},
                        },
                        "observation": {
                            "content": (
                                "remote materialize failed (rc=255): Host key verification failed."
                            ),
                            "is_error": True,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return artifact


def _cell(artifact: Path) -> ScoreCell:
    return ScoreCell(
        task_id="task",
        attempt=1,
        reward=0.0,
        passed=False,
        artifact_dir=str(artifact),
        infra_failed=False,
    )


def _unmetered_artifact(root: Path, *, stop_reason: str = "submitted") -> Path:
    artifact = root / "artifact"
    trace = artifact / "agent" / "wmo-run.json"
    trace.parent.mkdir(parents=True)
    trace.write_text(
        json.dumps(
            {
                "instruction": "repair the repository",
                "stop_reason": stop_reason,
                "steps": [
                    {
                        "action": {
                            "kind": "tool_call",
                            "name": "bash",
                            "arguments": {"command": "true"},
                        },
                        "observation": {"content": "", "is_error": False},
                    }
                ],
                "worker_usage": None,
            }
        ),
        encoding="utf-8",
    )
    return artifact


def _metered_artifact(root: Path, *, stop_reason: str) -> Path:
    artifact = root / "artifact"
    trace = artifact / "agent" / "wmo-run.json"
    trace.parent.mkdir(parents=True)
    trace.write_text(
        json.dumps(
            {
                "instruction": "repair the repository",
                "stop_reason": stop_reason,
                "steps": [],
                "worker_usage": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "cached_input_tokens": 0,
                    "cache_write_input_tokens": 0,
                    "reasoning_tokens": 0,
                    "calls": 1,
                    "call_seconds": [0.1],
                    "call_input_tokens": [10],
                    "call_output_tokens": [2],
                    "call_cached_input_tokens": [0],
                    "call_cache_write_input_tokens": [0],
                },
            }
        ),
        encoding="utf-8",
    )
    return artifact


def test_smoke_infrastructure_stop_is_ungradeable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact(tmp_path)
    monkeypatch.setattr(smoke_runner, "_wall_seconds", lambda path: 0.0)

    outcome = smoke_runner._outcome(
        _cell(artifact),
        entry=_entry(),
        logical_attempt=1,
        artifact_dir=artifact,
    )

    assert outcome.reward is None
    assert outcome.completion_status == "infrastructure_failure"
    assert outcome.failure_class == "infrastructure"
    assert outcome.error == ("remote materialize failed (rc=255): Host key verification failed.")


def test_full_matrix_infrastructure_stop_is_ungradeable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact(tmp_path)
    monkeypatch.setattr(matrix_runner, "_wall_seconds", lambda path: 0.0)

    outcome = matrix_runner._outcome(
        _cell(artifact),
        benchmark="terminal-bench-2",
        entry=_entry(),
        attempt=1,
        artifact_dir=artifact,
    )

    assert outcome.reward is None
    assert outcome.completion_status == "infrastructure_failure"
    assert outcome.failure_class == "infrastructure"


@pytest.mark.parametrize("runner", [smoke_runner, matrix_runner])
@pytest.mark.parametrize(
    "stop_reason",
    [
        "budget",
        "max_turns",
        "no_action",
        "no_tool_call",
        "output_truncated",
        "unparsed_tool_call",
    ],
)
def test_metered_agent_failure_remains_a_gradeable_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: object,
    stop_reason: str,
) -> None:
    artifact = _metered_artifact(tmp_path, stop_reason=stop_reason)
    monkeypatch.setattr(runner, "_wall_seconds", lambda path: 0.0)
    if runner is smoke_runner:
        outcome = smoke_runner._outcome(
            _cell(artifact),
            entry=_entry(),
            logical_attempt=1,
            artifact_dir=artifact,
        )
    else:
        outcome = matrix_runner._outcome(
            _cell(artifact),
            benchmark="terminal-bench-2",
            entry=_entry(),
            attempt=1,
            artifact_dir=artifact,
        )

    assert outcome.reward == 0.0
    assert outcome.completion_status == "scored_agent_failure"
    assert outcome.failure_class == "agent_failure"


@pytest.mark.parametrize("runner", [smoke_runner, matrix_runner])
def test_metered_provider_output_truncation_is_gradeable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: object,
) -> None:
    artifact = _metered_artifact(tmp_path, stop_reason="provider_error")
    cell = _cell(artifact).model_copy(update={"infra_failed": True})
    trace_path = artifact / "agent" / "wmo-run.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace["steps"] = [
        {
            "action": {"kind": "message", "content": "", "name": None, "arguments": {}},
            "observation": {
                "content": (
                    "worker LLM proxy error: Responses API returned incomplete response: "
                    "max_output_tokens"
                ),
                "is_error": True,
            },
        }
    ]
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    monkeypatch.setattr(runner, "_wall_seconds", lambda path: 0.0)
    if runner is smoke_runner:
        outcome = smoke_runner._outcome(
            cell,
            entry=_entry(),
            logical_attempt=1,
            artifact_dir=artifact,
        )
    else:
        outcome = matrix_runner._outcome(
            cell,
            benchmark="terminal-bench-2",
            entry=_entry(),
            attempt=1,
            artifact_dir=artifact,
        )

    assert outcome.reward == 0.0
    assert outcome.completion_status == "scored_agent_failure"
    assert outcome.failure_class == "agent_failure"
    assert outcome.error is None


@pytest.mark.parametrize("runner", [smoke_runner, matrix_runner])
def test_provider_error_before_model_execution_is_infrastructure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: object,
) -> None:
    artifact = tmp_path / "artifact"
    trace_path = artifact / "agent" / "wmo-run.json"
    trace_path.parent.mkdir(parents=True)
    trace_path.write_text(
        json.dumps(
            {
                "instruction": "repair the repository",
                "stop_reason": "provider_error",
                "steps": [],
                "worker_usage": None,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "_wall_seconds", lambda path: 0.0)
    if runner is smoke_runner:
        outcome = smoke_runner._outcome(
            _cell(artifact),
            entry=_entry(),
            logical_attempt=1,
            artifact_dir=artifact,
        )
    else:
        outcome = matrix_runner._outcome(
            _cell(artifact),
            benchmark="terminal-bench-2",
            entry=_entry(),
            attempt=1,
            artifact_dir=artifact,
        )

    assert outcome.reward is None
    assert outcome.completion_status == "infrastructure_failure"
    assert outcome.failure_class == "infrastructure"


@pytest.mark.parametrize("runner", [smoke_runner, matrix_runner])
def test_submitted_cell_without_worker_usage_uses_labeled_estimate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: object,
) -> None:
    artifact = _unmetered_artifact(tmp_path)
    monkeypatch.setattr(runner, "_wall_seconds", lambda path: 0.0)
    if runner is smoke_runner:
        outcome = smoke_runner._outcome(
            _cell(artifact),
            entry=_entry(),
            logical_attempt=2,
            artifact_dir=artifact,
        )
    else:
        outcome = matrix_runner._outcome(
            _cell(artifact),
            benchmark="terminal-bench-2",
            entry=_entry(),
            attempt=2,
            artifact_dir=artifact,
        )

    assert outcome.reward == 0.0
    assert outcome.completion_status == "scored_failure"
    assert outcome.failure_class == "task_failure"
    assert outcome.usage_accounting == "estimated"
    assert outcome.usage_estimate_method == "trace-char-prefix-4k-overhead-v1"
    assert outcome.cost_usd > 0
    assert outcome.error is None


def test_existing_infrastructure_zero_is_normalized_before_resume(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    root = tmp_path / "smoke"
    matrix_path = root / "outcomes.json"
    ledger_path = tmp_path / "spend-ledger.jsonl"
    entry = _entry()
    measured = OutcomeMatrix(
        pool=[entry],
        outcomes=[
            ScenarioOutcome(
                scenario_id="terminal-bench-2:task",
                task="repair the repository",
                model=entry.name,
                benchmark="terminal-bench-2",
                reward=0.0,
                completion_status="scored_failure",
                failure_class="scaffold",
                artifact_dir=str(artifact),
            )
        ],
    )

    normalized = smoke_runner._normalize_existing_ungradeable_attempts(
        root,
        matrix_path,
        ledger_path,
        measured,
    )

    outcome = normalized.outcomes[0]
    assert outcome.reward is None
    assert outcome.completion_status == "infrastructure_failure"
    assert outcome.failure_class == "infrastructure"
    assert outcome.error == ("remote materialize failed (rc=255): Host key verification failed.")
    assert Path(outcome.artifact_dir).is_dir()
    assert json.loads(ledger_path.read_text(encoding="utf-8"))["completion_status"] == (
        "infrastructure_failure"
    )


def test_existing_unmetered_score_is_normalized_to_labeled_estimate(tmp_path: Path) -> None:
    artifact = _unmetered_artifact(tmp_path)
    root = tmp_path / "smoke"
    matrix_path = root / "outcomes.json"
    ledger_path = tmp_path / "spend-ledger.jsonl"
    entry = _entry()
    measured = OutcomeMatrix(
        pool=[entry],
        outcomes=[
            ScenarioOutcome(
                scenario_id="terminal-bench-2:task",
                task="repair the repository",
                model=entry.name,
                benchmark="terminal-bench-2",
                attempt_number=2,
                reward=1.0,
                success=True,
                completion_status="scored_pass",
                artifact_dir=str(artifact),
            )
        ],
    )

    normalized = smoke_runner._normalize_existing_ungradeable_attempts(
        root,
        matrix_path,
        ledger_path,
        measured,
    )

    outcome = normalized.outcomes[0]
    assert outcome.reward == 1.0
    assert outcome.failure_class == ""
    assert outcome.usage_accounting == "estimated"
    assert outcome.cost_usd > 0
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["model_cost_usd"] == outcome.cost_usd
    assert ledger["model_cost_accounting_status"] == "estimated_from_trace"


def test_existing_unmetered_scaffold_is_reclassified_as_unknown_cost(tmp_path: Path) -> None:
    artifact = _unmetered_artifact(tmp_path, stop_reason="max_turns")
    root = tmp_path / "smoke"
    matrix_path = root / "outcomes.json"
    ledger_path = tmp_path / "spend-ledger.jsonl"
    entry = _entry()
    measured = OutcomeMatrix(
        pool=[entry],
        outcomes=[
            ScenarioOutcome(
                scenario_id="terminal-bench-2:task",
                task="repair the repository",
                model=entry.name,
                benchmark="terminal-bench-2",
                attempt_number=2,
                reward=None,
                completion_status="scaffold_failure",
                failure_class="scaffold",
                artifact_dir=str(artifact),
            )
        ],
    )

    normalized = smoke_runner._normalize_existing_ungradeable_attempts(
        root,
        matrix_path,
        ledger_path,
        measured,
    )

    assert normalized.outcomes[0].failure_class == "metering"
    assert json.loads(ledger_path.read_text(encoding="utf-8"))["model_cost_usd"] is None


def test_archive_keeps_distinct_artifacts_for_the_same_logical_attempt(tmp_path: Path) -> None:
    root = tmp_path / "smoke"
    first = _artifact(tmp_path / "first")
    second = _unmetered_artifact(tmp_path / "second")

    first_archive = smoke_runner._archive_infra(
        root,
        first,
        task_id="task",
        arm="arm",
        attempt=2,
    )
    second_archive = smoke_runner._archive_infra(
        root,
        second,
        task_id="task",
        arm="arm",
        attempt=2,
    )

    assert first_archive != second_archive
    assert smoke_runner._artifact_digest(str(first_archive)) != smoke_runner._artifact_digest(
        str(second_archive)
    )


def test_unmetered_smoke_quarantines_derived_policy_and_reports(tmp_path: Path) -> None:
    root = tmp_path / "smoke"
    (root / "policy").mkdir(parents=True)
    (root / "policy" / "policy.json").write_text("{}", encoding="utf-8")
    (root / "smoke-report.json").write_text('{"valid": true}', encoding="utf-8")
    (root / "resume-proof.json").write_text('{"unchanged": true}', encoding="utf-8")
    entry = _entry()
    matrix = OutcomeMatrix(
        pool=[entry],
        outcomes=[
            ScenarioOutcome(
                scenario_id="terminal-bench-2:task",
                task="repair",
                model=entry.name,
                benchmark="terminal-bench-2",
                attempt_number=2,
                reward=None,
                completion_status="metering_failure",
                failure_class="metering",
                artifact_dir="/evidence/attempt-2",
            )
        ],
    )

    smoke_runner._invalidate_smoke_derivatives(root, matrix)
    smoke_runner._invalidate_smoke_derivatives(root, matrix)

    assert not (root / "policy").exists()
    assert not (root / "smoke-report.json").exists()
    assert not (root / "resume-proof.json").exists()
    invalid = json.loads((root / "invalidated.json").read_text(encoding="utf-8"))
    assert invalid["valid"] is False
    assert invalid["paid_execution_allowed"] is False
    assert invalid["unknown_cost_attempts"][0]["model_cost_usd"] is None
    assert len(invalid["quarantined_derived_artifacts"]) == 3
    assert all(
        Path(row["quarantined_path"]).exists() for row in invalid["quarantined_derived_artifacts"]
    )
    report = next(
        row
        for row in invalid["quarantined_derived_artifacts"]
        if row["original_path"].endswith("smoke-report.json")
    )
    assert report["sha256"] == hashlib.sha256(b'{"valid": true}').hexdigest()


def test_unknown_historical_cost_does_not_block_a_new_estimated_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _entry()
    pool = ModelPool(models=[entry])
    root = tmp_path / "smoke"
    matrix_path = root / "outcomes.json"
    ledger_path = tmp_path / "spend-ledger.jsonl"
    OutcomeMatrix(
        pool=[entry],
        outcomes=[
            ScenarioOutcome(
                scenario_id="terminal-bench-2:prior",
                task="prior",
                model=entry.name,
                benchmark="terminal-bench-2",
                reward=None,
                completion_status="metering_failure",
                failure_class="metering",
            )
        ],
    ).save(matrix_path)
    monkeypatch.setattr(smoke_runner, "ARMS", (entry.name,))
    artifact = _unmetered_artifact(tmp_path / "new")

    class FakeScorer:
        def score(self, agent: object) -> object:
            del agent
            return type(
                "ScoreResult",
                (),
                {
                    "cells": [
                        ScoreCell(
                            task_id="next",
                            attempt=1,
                            reward=0.0,
                            passed=False,
                            artifact_dir=str(artifact),
                            infra_failed=False,
                        )
                    ]
                },
            )()

    async def fake_scorer(*args: object, **kwargs: object) -> FakeScorer:
        del args, kwargs
        return FakeScorer()

    monkeypatch.setattr(smoke_runner, "_scorer", fake_scorer)
    monkeypatch.setattr(smoke_runner, "_wall_seconds", lambda path: 0.0)

    smoke_runner._run_cell(
        root,
        template_path=tmp_path / "unused.yaml",
        task_id="next",
        entry=entry,
        matrix_path=matrix_path,
        ledger_path=ledger_path,
        pool=pool,
    )

    outcome = next(
        row
        for row in OutcomeMatrix.load(matrix_path).outcomes
        if row.scenario_id == "terminal-bench-2:next"
    )
    assert outcome.reward == 0.0
    assert outcome.usage_accounting == "estimated"


def test_full_matrix_rejects_an_invalid_smoke_before_paid_work(tmp_path: Path) -> None:
    smoke_root = tmp_path / "smoke"
    smoke_root.mkdir(parents=True)
    (smoke_root / "invalidated.json").write_text(
        json.dumps({"valid": False}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="replacement smoke"):
        matrix_runner._require_valid_smoke(tmp_path)


def test_full_matrix_accepts_complete_metered_smoke_evidence(tmp_path: Path) -> None:
    smoke_root = tmp_path / "smoke"
    artifact = smoke_root / "official-artifact"
    artifact.mkdir(parents=True)
    (artifact / "result.json").write_text("{}", encoding="utf-8")
    entries = [
        PoolEntry(
            name="oai-luna-high",
            kind=ProviderKind.OPENAI_RESPONSES,
            model="gpt-5.6-luna",
            input_per_mtok=1.0,
            output_per_mtok=6.0,
        ),
        PoolEntry(
            name="ant-haiku45",
            kind=ProviderKind.ANTHROPIC,
            model="claude-haiku-4-5-20251001",
            input_per_mtok=1.0,
            output_per_mtok=5.0,
        ),
    ]
    outcomes = [
        ScenarioOutcome(
            scenario_id=f"terminal-bench-2:{task_id}",
            task=task_id,
            model=entry.name,
            benchmark="terminal-bench-2",
            reward=1.0,
            success=True,
            usage=TokenUsage(input_tokens=10, output_tokens=2),
            cost_usd=0.001,
            call_seconds=[0.1],
            call_input_tokens=[10],
            call_output_tokens=[2],
            call_cached_input_tokens=[0],
            call_cache_write_input_tokens=[0],
            completion_status="scored_pass",
            artifact_dir=str(artifact),
        )
        for task_id in matrix_runner.SMOKE_TASKS
        for entry in entries
    ]
    OutcomeMatrix(pool=entries, outcomes=outcomes).save(smoke_root / "outcomes.json")
    (smoke_root / "smoke-report.json").write_text(
        json.dumps(
            {
                "gradeable_cells": 4,
                "fit_task": matrix_runner.SMOKE_TASKS[0],
                "heldout_task": matrix_runner.SMOKE_TASKS[1],
                "model_spend_usd": 0.004,
            }
        ),
        encoding="utf-8",
    )
    (smoke_root / "resume-proof.json").write_text(
        json.dumps({"unchanged": True, "resumed_cells": 2}),
        encoding="utf-8",
    )
    (smoke_root / "policy").mkdir()
    (smoke_root / "policy" / "policy.json").write_text("{}", encoding="utf-8")

    matrix_runner._require_valid_smoke(tmp_path)


def test_identical_harbor_resume_does_not_consume_an_attempt(tmp_path: Path) -> None:
    first_artifact = _artifact(tmp_path / "first")
    duplicate_artifact = _artifact(tmp_path / "duplicate")
    root = tmp_path / "smoke"
    matrix_path = root / "outcomes.json"
    ledger_path = tmp_path / "spend-ledger.jsonl"
    entry = _entry()
    outcomes = [
        ScenarioOutcome(
            scenario_id="terminal-bench-2:task",
            task="repair the repository",
            model=entry.name,
            benchmark="terminal-bench-2",
            attempt_number=attempt,
            episode=attempt - 1,
            reward=None,
            completion_status="scaffold_failure",
            failure_class="scaffold",
            artifact_dir=str(artifact),
        )
        for attempt, artifact in ((1, first_artifact), (2, duplicate_artifact))
    ]
    measured = OutcomeMatrix(pool=[entry], outcomes=outcomes)
    matrix_path.parent.mkdir(parents=True)
    measured.save(matrix_path)
    ledger_path.write_text(
        "".join(
            json.dumps(
                {
                    "event_id": (
                        f"smoke:terminal-bench-2:terminal-bench-2:task:{entry.name}:{attempt}"
                    ),
                    "status": "completed",
                    "model_cost_usd": 0.0,
                }
            )
            + "\n"
            for attempt in (1, 2)
        ),
        encoding="utf-8",
    )

    deduplicated = smoke_runner._drop_duplicate_retry_noops(
        root,
        matrix_path,
        ledger_path,
        measured,
    )

    assert [row.attempt_number for row in deduplicated.outcomes] == [1]
    assert len(ledger_path.read_text(encoding="utf-8").splitlines()) == 1
    audit = json.loads((root / "retry-noops.json").read_text(encoding="utf-8"))
    assert audit["rows"][0]["attempt_number"] == 2
