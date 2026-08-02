"""Offline tests for staged coding-router matrix scheduling."""

from __future__ import annotations

import hashlib
import json
import threading
from collections import Counter
from pathlib import Path

import coding_model_router_matrix as matrix_runner
import pytest
from coding_model_router_analyze import _develop
from coding_model_router_matrix import (
    BENCHMARKS,
    CELL_SPEND_RESERVATION_USD,
    DEFAULT_VERIFIER_TIMEOUT_S,
    FAST_DEV_ARMS,
    FAST_DEV_BENCHMARK,
    FAST_DEV_STAGE,
    FAST_DEV_TASK_COUNT,
    FULL_STAGE,
    HARBOR_TASK_CACHE,
    SPLIT_SEEDS,
    TERMINAL_FULL_STAGE,
    BudgetExhausted,
    RunState,
    _failure_class,
    _fast_dev_task_ids,
    _job_template,
    _outcome,
    _run_cells,
    _stage_cell_specs,
)

from wmo.harness.scoring import ScoreCell
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.providers.base import ProviderKind
from wmo.providers.pool import ModelPool, PoolEntry


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _pool() -> ModelPool:
    names = (*FAST_DEV_ARMS, "extra-arm")
    return ModelPool(
        models=[
            PoolEntry(
                name=name,
                kind=ProviderKind.OPENAI,
                model=f"test-{name}",
                input_per_mtok=1.0,
                output_per_mtok=2.0,
            )
            for name in names
        ]
    )


def _root(tmp_path: Path) -> Path:
    terminal_tasks = [f"task-{index:02d}" for index in range(16)]
    _write_json(
        tmp_path / "tasks" / f"{FAST_DEV_BENCHMARK}.json",
        {"tasks": [{"task_id": task_id} for task_id in terminal_tasks]},
    )
    _write_json(
        tmp_path / "tasks" / "swe-bench-verified.json",
        {"tasks": [{"task_id": "swe-task"}]},
    )
    common_fit = terminal_tasks[:14]
    for seed in SPLIT_SEEDS:
        _write_json(
            tmp_path / "splits" / f"seed-{seed}.json",
            {
                FAST_DEV_BENCHMARK: {
                    "fit": [*common_fit, terminal_tasks[14 + seed % 2]],
                    "heldout": [],
                }
            },
        )
    return tmp_path


def test_fast_dev_tasks_are_deterministic_all_seed_fit_rows(tmp_path: Path) -> None:
    root = _root(tmp_path)
    common = [f"task-{index:02d}" for index in range(14)]
    expected = sorted(
        common,
        key=lambda task_id: (
            hashlib.sha256(f"fast-dev-v1:{task_id}".encode()).hexdigest(),
            task_id,
        ),
    )[:FAST_DEV_TASK_COUNT]

    assert _fast_dev_task_ids(root) == expected


def test_fast_dev_stage_is_exact_reusable_anchor_tranche(tmp_path: Path) -> None:
    root = _root(tmp_path)
    specs = _stage_cell_specs(root, _pool(), FAST_DEV_STAGE)

    assert len(specs) == FAST_DEV_TASK_COUNT * len(FAST_DEV_ARMS)
    assert {benchmark for benchmark, _, _ in specs} == {FAST_DEV_BENCHMARK}
    assert {entry.name for _, _, entry in specs} == set(FAST_DEV_ARMS)


def test_full_stage_keeps_every_benchmark_task_and_model(tmp_path: Path) -> None:
    root = _root(tmp_path)
    pool = _pool()
    specs = _stage_cell_specs(root, pool, FULL_STAGE)

    expected_tasks = 16 + 1
    assert len(specs) == expected_tasks * len(pool.models)
    assert {benchmark for benchmark, _, _ in specs} == set(BENCHMARKS)


def test_terminal_full_stage_keeps_every_terminal_task_and_model(tmp_path: Path) -> None:
    root = _root(tmp_path)
    pool = _pool()
    specs = _stage_cell_specs(root, pool, TERMINAL_FULL_STAGE)

    assert len(specs) == 16 * len(pool.models)
    assert {benchmark for benchmark, _, _ in specs} == {FAST_DEV_BENCHMARK}
    assert {entry.name for _, _, entry in specs} == {entry.name for entry in pool.models}


def test_job_template_uses_experiment_task_cache(tmp_path: Path) -> None:
    template = _job_template(FAST_DEV_BENCHMARK, tmp_path / "jobs")

    assert template.datasets[0].download_dir == HARBOR_TASK_CACHE
    assert template.verifier.override_timeout_sec == DEFAULT_VERIFIER_TIMEOUT_S


def test_post_execution_error_with_official_reward_is_gradeable() -> None:
    cell = ScoreCell(
        task_id="task",
        attempt=1,
        reward=0.0,
        passed=False,
        note="completed with AgentTimeoutError",
    )

    assert _failure_class(cell, "error", provider_execution_started=True) == "agent_failure"


def test_pre_execution_error_remains_infrastructure() -> None:
    cell = ScoreCell(
        task_id="task",
        attempt=1,
        reward=0.0,
        passed=False,
        note="failed before provider execution",
    )

    assert _failure_class(cell, "error", provider_execution_started=False) == "infrastructure"


def test_post_execution_infrastructure_failure_estimates_missing_usage(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "trial"
    _write_json(
        artifact / "agent" / "wmo-run.json",
        {
            "instruction": "repair the repository",
            "stop_reason": "agent-exception:TimeoutExpired",
            "turns": 1,
            "steps": [
                {
                    "action": {"kind": "tool_call", "name": "bash"},
                    "observation": {"content": "partial result"},
                }
            ],
        },
    )
    cell = ScoreCell(
        task_id="task",
        attempt=1,
        reward=0.0,
        passed=False,
        note="verifier produced no reward",
        infra_failed=True,
    )

    outcome = _outcome(
        cell,
        benchmark=FAST_DEV_BENCHMARK,
        entry=_pool().models[0],
        attempt=1,
        artifact_dir=artifact,
    )

    assert outcome.reward is None
    assert outcome.failure_class == "infrastructure"
    assert outcome.usage_accounting == "estimated"
    assert outcome.usage_estimate_method == "trace-char-prefix-4k-overhead-v1"
    assert outcome.usage.input_tokens > 0
    assert outcome.cost_usd > 0


def test_conservative_unknown_cost_debit_counts_against_ceiling(tmp_path: Path) -> None:
    (tmp_path / "spend-ledger.jsonl").write_text(
        json.dumps(
            {
                "event_id": "invalid-smoke:paid",
                "model_cost_accounting_status": "missing_provider_usage",
                "model_cost_usd": None,
                "budget_debit_usd": 300.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    state = RunState(root=tmp_path / "full", pool=_pool(), ceiling_usd=20_000.0)

    assert state.spent_and_reserved() == (300.0, 0.0)


def test_unreconciled_unknown_cost_still_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "spend-ledger.jsonl").write_text(
        json.dumps(
            {
                "event_id": "invalid-smoke:paid",
                "model_cost_accounting_status": "missing_provider_usage",
                "model_cost_usd": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    state = RunState(root=tmp_path / "full", pool=_pool(), ceiling_usd=20_000.0)

    with pytest.raises(BudgetExhausted, match="no authorized conservative budget debit"):
        state.spent_and_reserved()


def test_explicit_stale_reservation_recovery_preserves_debit_and_frees_id(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    pool = _pool()
    event_id = f"full:terminal-bench-2:task:{FAST_DEV_ARMS[0]}:1"
    (root / "spend-ledger.jsonl").write_text(
        json.dumps(
            {
                "event_id": event_id,
                "scenario_id": "terminal-bench-2:task",
                "model": FAST_DEV_ARMS[0],
                "attempt_number": 1,
                "status": "reserved",
                "reserved_usd": CELL_SPEND_RESERVATION_USD,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    state = RunState(root=root / "full", pool=pool, ceiling_usd=20_000.0)

    assert state.recover_stale_reservations() == 1
    assert state.spent_and_reserved() == (CELL_SPEND_RESERVATION_USD, 0.0)
    assert state.ledger[0]["event_id"] == (f"interrupted:{event_id}:stale-reservation-recovery")
    assert state.reserve("terminal-bench-2", "task", FAST_DEV_ARMS[0], 1) == event_id
    assert state.spent_and_reserved() == (
        CELL_SPEND_RESERVATION_USD,
        CELL_SPEND_RESERVATION_USD,
    )


def test_scheduler_retries_temporary_reservation_pressure_after_headroom_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root(tmp_path)
    pool = _pool()
    state = RunState(root=root / "full", pool=pool, ceiling_usd=20_000.0)
    entry = pool.models[0]
    calls: Counter[str] = Counter()
    completed: list[str] = []
    first_budget_check = threading.Event()

    def fake_run_cell(
        state: RunState,
        *,
        benchmark: str,
        task_id: str,
        entry: PoolEntry,
        timeout_s: float,
        verifier_timeout_s: float,
    ) -> None:
        del state, benchmark, entry, timeout_s, verifier_timeout_s
        calls[task_id] += 1
        if task_id == "task-00":
            assert first_budget_check.wait(timeout=1.0)
        elif task_id == "task-01" and calls[task_id] == 1:
            first_budget_check.set()
            raise BudgetExhausted("temporary in-flight reservation pressure")
        completed.append(task_id)

    monkeypatch.setattr(matrix_runner, "_run_cell", fake_run_cell)
    deferred = _run_cells(
        state,
        cells=[
            (FAST_DEV_BENCHMARK, "task-00", entry),
            (FAST_DEV_BENCHMARK, "task-01", entry),
            (FAST_DEV_BENCHMARK, "task-02", entry),
        ],
        concurrency=2,
        timeout_s=1.0,
        verifier_timeout_s=1.0,
    )

    assert calls == Counter({"task-01": 2, "task-00": 1, "task-02": 1})
    assert sorted(completed) == ["task-00", "task-01", "task-02"]
    assert deferred == 0


def test_scheduler_stops_when_budget_blocks_with_nothing_in_flight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root(tmp_path)
    pool = _pool()
    state = RunState(root=root / "full", pool=pool, ceiling_usd=20_000.0)
    entry = pool.models[0]
    calls = 0

    def fake_run_cell(
        state: RunState,
        *,
        benchmark: str,
        task_id: str,
        entry: PoolEntry,
        timeout_s: float,
        verifier_timeout_s: float,
    ) -> None:
        nonlocal calls
        del state, benchmark, task_id, entry, timeout_s, verifier_timeout_s
        calls += 1
        raise BudgetExhausted("true ceiling exhaustion")

    monkeypatch.setattr(matrix_runner, "_run_cell", fake_run_cell)
    deferred = _run_cells(
        state,
        cells=[(FAST_DEV_BENCHMARK, "task-00", entry)],
        concurrency=1,
        timeout_s=1.0,
        verifier_timeout_s=1.0,
    )

    assert calls == 1
    assert deferred == 1


@pytest.mark.parametrize(
    "stop_reason",
    ["provider_error", "error", "agent-exception:AgentTimeoutError"],
)
def test_existing_post_execution_failures_keep_one_gradeable_row(
    tmp_path: Path,
    stop_reason: str,
) -> None:
    root = _root(tmp_path)
    pool = _pool()
    outcomes: list[ScenarioOutcome] = []
    ledger_rows: list[dict[str, object]] = []
    for attempt in range(1, 4):
        artifact = tmp_path / "full" / "infra-attempts" / f"attempt-{attempt}"
        _write_json(
            artifact / "agent" / "wmo-run.json",
            {
                "instruction": "repair the repository",
                "stop_reason": stop_reason,
                "steps": [{"action": {"kind": "message"}, "observation": {}}],
                "worker_usage": {
                    "calls": 1,
                    "input_tokens": 10,
                    "output_tokens": 2,
                },
            },
        )
        (artifact / "verifier").mkdir(parents=True)
        (artifact / "verifier" / "reward.txt").write_text("0\n", encoding="utf-8")
        outcomes.append(
            ScenarioOutcome(
                scenario_id="terminal-bench-2:task",
                task="repair the repository",
                model=FAST_DEV_ARMS[0],
                benchmark="terminal-bench-2",
                reward=None,
                success=False,
                stop_reason=stop_reason,
                attempt_number=attempt,
                completion_status="infrastructure_failure",
                failure_class="infrastructure",
                artifact_dir=str(artifact),
                cost_usd=0.1,
            )
        )
        ledger_rows.append(
            {
                "event_id": f"full:terminal-bench-2:task:{FAST_DEV_ARMS[0]}:{attempt}",
                "scenario_id": "terminal-bench-2:task",
                "model": FAST_DEV_ARMS[0],
                "attempt_number": attempt,
                "status": "completed",
                "completion_status": "infrastructure_failure",
                "failure_class": "infrastructure",
                "model_cost_usd": 0.1,
            }
        )
    full = root / "full"
    full.mkdir(exist_ok=True)
    OutcomeMatrix(pool=pool.models, outcomes=outcomes).save(full / "outcomes.json")
    (root / "spend-ledger.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in ledger_rows),
        encoding="utf-8",
    )

    state = RunState(root=full, pool=pool, ceiling_usd=20_000.0)

    repaired = sorted(state.matrix.outcomes, key=lambda outcome: outcome.attempt_number)
    assert repaired[0].reward == 0.0
    assert repaired[0].completion_status == "scored_agent_failure"
    assert repaired[0].failure_class == "agent_failure"
    assert [row.completion_status for row in repaired[1:]] == [
        "excluded_protocol_retry",
        "excluded_protocol_retry",
    ]
    assert [row.failure_class for row in repaired[1:]] == [
        "protocol_retry_excluded",
        "protocol_retry_excluded",
    ]
    assert all(row.usage_accounting == "estimated" for row in repaired)
    assert all(row.usage.input_tokens > 0 for row in repaired)
    assert all(row.cost_usd > 0 for row in repaired)
    assert state.completed("terminal-bench-2", "task", FAST_DEV_ARMS[0]) is True
    assert [row["completion_status"] for row in state.ledger] == [
        "scored_agent_failure",
        "excluded_protocol_retry",
        "excluded_protocol_retry",
    ]
    assert all(
        row["model_cost_accounting_status"] == "estimated_from_trace" for row in state.ledger
    )


def test_earlier_metered_provider_failure_excludes_later_retry(tmp_path: Path) -> None:
    root = _root(tmp_path)
    pool = _pool()
    first_artifact = tmp_path / "full" / "infra-attempts" / "attempt-1"
    _write_json(
        first_artifact / "agent" / "wmo-run.json",
        {
            "instruction": "repair the repository",
            "stop_reason": "provider_error",
            "steps": [
                {
                    "action": {"kind": "message"},
                    "observation": {
                        "is_error": True,
                        "content": "incomplete response: max_output_tokens",
                    },
                }
            ],
            "worker_usage": {
                "calls": 1,
                "input_tokens": 10,
                "output_tokens": 2,
            },
        },
    )
    outcomes = [
        ScenarioOutcome(
            scenario_id="terminal-bench-2:task",
            task="repair the repository",
            model=FAST_DEV_ARMS[0],
            benchmark="terminal-bench-2",
            attempt_number=1,
            reward=None,
            success=False,
            stop_reason="provider_error",
            completion_status="infrastructure_failure",
            failure_class="infrastructure",
            artifact_dir=str(first_artifact),
            cost_usd=0.1,
        ),
        ScenarioOutcome(
            scenario_id="terminal-bench-2:task",
            task="repair the repository",
            model=FAST_DEV_ARMS[0],
            benchmark="terminal-bench-2",
            attempt_number=2,
            reward=0.0,
            success=False,
            stop_reason="submitted",
            completion_status="scored_failure",
            failure_class="task_failure",
            artifact_dir=str(tmp_path / "full" / "attempt-2"),
            cost_usd=0.1,
        ),
    ]
    full = root / "full"
    full.mkdir(exist_ok=True)
    OutcomeMatrix(pool=pool.models, outcomes=outcomes).save(full / "outcomes.json")
    (root / "spend-ledger.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "event_id": f"full:terminal-bench-2:task:{FAST_DEV_ARMS[0]}:{attempt}",
                    "scenario_id": "terminal-bench-2:task",
                    "model": FAST_DEV_ARMS[0],
                    "attempt_number": attempt,
                    "status": "completed",
                    "model_cost_usd": 0.1,
                }
            )
            + "\n"
            for attempt in (1, 2)
        ),
        encoding="utf-8",
    )

    state = RunState(root=full, pool=pool, ceiling_usd=20_000.0)

    repaired = sorted(state.matrix.outcomes, key=lambda outcome: outcome.attempt_number)
    assert repaired[0].reward == 0.0
    assert repaired[0].completion_status == "scored_agent_failure"
    assert repaired[0].failure_class == "agent_failure"
    assert repaired[1].reward is None
    assert repaired[1].completion_status == "excluded_protocol_retry"
    assert repaired[1].failure_class == "protocol_retry_excluded"
    assert state.completed("terminal-bench-2", "task", FAST_DEV_ARMS[0]) is True


def test_develop_fits_diagnostic_policy_without_outer_heldout(tmp_path: Path) -> None:
    root = _root(tmp_path)
    pool = _pool()
    rewards = {
        "oai-sol-high": (1.0, 0.20),
        "oai-luna-high": (0.5, 0.05),
        "ant-opus5-high": (0.875, 0.25),
        "ant-haiku45": (0.375, 0.04),
    }
    outcomes: list[ScenarioOutcome] = []
    for task_index, task_id in enumerate(_fast_dev_task_ids(root)):
        for entry in pool.models:
            if entry.name not in FAST_DEV_ARMS:
                continue
            quality, cost = rewards[entry.name]
            reward = float((task_index % 8) / 8 < quality)
            outcomes.append(
                ScenarioOutcome(
                    scenario_id=f"{FAST_DEV_BENCHMARK}:{task_id}",
                    task=f"repair fixture {task_id}",
                    model=entry.name,
                    benchmark=FAST_DEV_BENCHMARK,
                    reward=reward,
                    success=bool(reward),
                    cost_usd=cost,
                    call_seconds=[1.0],
                )
            )
    matrix_path = root / "full" / "outcomes.json"
    matrix_path.parent.mkdir(parents=True)
    OutcomeMatrix(pool=pool.models, outcomes=outcomes).save(matrix_path)

    _develop(root)

    report = json.loads((root / "analysis" / "fast-dev-report.json").read_text())
    assert report["diagnostic_only"] is True
    assert report["promotion_evidence"] is False
    assert report["outer_heldout_rows_read"] == 0
    assert len(report["fit_ids"]) == 8
    assert len(report["replay_ids"]) == 4
    assert (root / "analysis" / "fast-dev-policy" / "policy.json").is_file()
