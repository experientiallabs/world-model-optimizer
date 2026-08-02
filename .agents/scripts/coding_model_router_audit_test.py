"""Tests for the coding-router completion auditor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from coding_model_router_audit import FINAL_REPORT_ROWS, audit


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _complete_fixture(root: Path) -> None:
    pool = root / "pool.toml"
    pool.parent.mkdir(parents=True, exist_ok=True)
    pool.write_text('[[models]]\nname = "one"\n', encoding="utf-8")
    task_paths = {
        benchmark: root / "tasks" / f"{benchmark}.json"
        for benchmark in ("terminal-bench-2", "swe-bench-verified")
    }
    for benchmark, path in task_paths.items():
        _write_json(
            path,
            {"count": 1, "tasks": [{"task_id": f"{benchmark}-task", "group": "g"}]},
        )
    split_paths = [root / "splits" / f"seed-{seed}.json" for seed in range(5)]
    for seed, path in enumerate(split_paths):
        _write_json(path, {"seed": seed})
    _write_json(
        root / "freeze-summary.json",
        {
            "experiment_id": "coding-router-20260728",
            "source_commit": "abc123",
            "spend_ceiling_usd": 500.0,
            "model_arms": 2,
            "pool_sha256": _sha256(pool),
            "terminal_manifest_sha256": _sha256(task_paths["terminal-bench-2"]),
            "swe_manifest_sha256": _sha256(task_paths["swe-bench-verified"]),
            "split_sha256": {path.name: _sha256(path) for path in split_paths},
        },
    )

    smoke_rows = []
    for task in ("break-filter-js-from-html", "log-summary-date-ranges"):
        for arm in ("oai-luna-high", "ant-haiku45"):
            smoke_rows.append(
                {
                    "scenario_id": f"terminal-bench-2:{task}",
                    "model": arm,
                    "reward": 1.0,
                    "cost_usd": 0.01,
                    "call_seconds": [1.0],
                    "call_input_tokens": [10],
                    "call_output_tokens": [2],
                }
            )
    _write_json(root / "smoke" / "outcomes.json", {"outcomes": smoke_rows})
    _write_json(root / "smoke" / "smoke-report.json", {"gradeable_cells": 4})
    _write_json(root / "smoke" / "resume-proof.json", {"unchanged": True, "resumed_cells": 2})
    _write_json(root / "smoke" / "policy" / "policy.json", {"kind": "knn"})

    full_rows = [
        {"scenario_id": f"scenario-{index}", "model": "model", "reward": 1.0} for index in range(4)
    ]
    full_outcomes = root / "full" / "outcomes.json"
    _write_json(full_outcomes, {"outcomes": full_rows})
    _write_json(
        root / "full" / "summary.json",
        {
            "stage": "full",
            "cells_expected": 4,
            "gradeable_cells": 4,
        },
    )
    _write_json(
        root / "analysis" / "matrix-validation.json",
        {"status": "complete", "gradeable_cells": 4},
    )
    (root / "spend-ledger.jsonl").write_text(
        json.dumps({"event_id": "cell:1", "status": "completed", "model_cost_usd": 1.25}) + "\n",
        encoding="utf-8",
    )

    lock_path = root / "analysis" / "selection-lock.json"
    outer_path = root / "analysis" / "outer-results.json"
    _write_json(lock_path, {"seeds": [{"seed": seed} for seed in range(5)]})
    _write_json(
        outer_path,
        {
            "seeds": [{"seed": seed} for seed in range(5)],
            "pareto": [{"quality": 0.9, "cost_per_task": 1.0}],
            "promoted": False,
            "all_seed_promotion_gates": [False] * 5,
            "paired_cluster_bootstrap": {"retention_lower_95": 0.9},
            "capability_slices": {
                "repository-level-bug-fixing": {
                    "seeds_observed": 5,
                    "points": {},
                }
            },
            "one_at_a_time_ablations": {
                "benchmark_stratified": "ablation:benchmark_stratified",
                "missing_fit_coverage_0.8": "ablation:missing_fit_coverage_0.8",
                "missing_fit_coverage_1.0_control": "guarded_knn",
                "latency_only_static": "latency_only",
                "production_eligible": False,
            },
        },
    )
    _write_json(
        root / "analysis" / "evaluation-complete.json",
        {
            "heldout_evaluated": True,
            "selection_lock_sha256": _sha256(lock_path),
            "outer_results_sha256": _sha256(outer_path),
        },
    )
    _write_json(
        root / "analysis" / "deployable" / "policy.json",
        {"kind": "knn", "knn_bank_path": "policy_knn_bank.npz"},
    )
    (root / "analysis" / "deployable" / "policy_knn_bank.npz").write_bytes(b"bank")

    _write_json(root / "serving" / "prepare.json", {"protocol": "coding-router-serving-v1"})
    _write_json(
        root / "serving" / "result-1.json",
        {
            "completion_status": "passed",
            "requests": 8,
            "fallback_gate": "novelty-abstain",
            "affinity_reason": "sticky: conversation affinity",
            "cache_aware_credit_usd": 0.001,
        },
    )

    _write_json(root / "world-model" / "prepare.json", {})
    _write_json(root / "world-model" / "build-usage.json", {})
    _write_json(root / "world-model" / "simulated" / "completion.json", {"complete": True})
    _write_json(root / "world-model" / "simulated" / "outcomes.json", {"outcomes": []})
    _write_json(
        root / "world-model" / "comparison.json",
        {
            "protocol": "coding-world-model-compare-v1",
            "binary_agreement": 0.9,
            "false_positive_rate": 0.1,
            "false_negative_rate": 0.1,
            "calibration": [],
            "candidate_rank_spearman": 0.8,
            "best_single_agreement": 1.0,
            "selected_model_agreement": 0.8,
            "guard_decision_agreement": 0.9,
            "promotion_agreement": True,
        },
    )
    rows = "\n".join(
        f"| {row.title()} | 1 | 1 | 1 | 1 | 1 | 1/1 | pass |" for row in FINAL_REPORT_ROWS
    )
    (root / "final-report.md").write_text(
        "| Policy | Quality | Quality retained | Cost/task | Cost savings | Completion | "
        "Latency p50/p95 | Verdict |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"{rows}\n\nThe target was not achieved.\n",
        encoding="utf-8",
    )


def test_empty_root_fails_closed(tmp_path: Path) -> None:
    result = audit(tmp_path)

    assert result.completion_status == "incomplete"
    assert result.ready_for_material_paid_execution is False
    assert result.target_achieved is None
    assert result.conservative_budget_debit_usd == 0.0
    assert result.estimated_model_spend_usd == 0.0
    assert result.blocking_requirements


def test_complete_evidence_can_conclude_target_not_reached(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)

    result = audit(tmp_path)

    assert result.completion_status == "complete"
    assert result.target_achieved is False
    assert result.ready_for_material_paid_execution is True
    assert result.known_model_spend_usd == 1.25
    assert result.conservative_budget_debit_usd == 0.0
    assert result.estimated_model_spend_usd == 0.0
    assert not result.blocking_requirements


def test_unknown_cost_event_blocks_completion(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)
    with (tmp_path / "spend-ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "event_id": "cell:2",
                    "status": "completed",
                    "model_cost_usd": None,
                }
            )
            + "\n"
        )

    result = audit(tmp_path)

    assert result.completion_status == "incomplete"
    assert result.unknown_cost_events == 1
    assert "spend_ledger" in result.blocking_requirements


def test_estimated_model_cost_is_reported_separately(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)
    (tmp_path / "spend-ledger.jsonl").write_text(
        json.dumps(
            {
                "event_id": "cell:estimated",
                "status": "completed",
                "model_cost_usd": 2.5,
                "model_cost_accounting_status": "estimated_from_trace",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = audit(tmp_path)

    assert result.completion_status == "complete"
    assert result.known_model_spend_usd == 0.0
    assert result.estimated_model_spend_usd == 2.5
    assert "spend_ledger" not in result.blocking_requirements


def test_conservative_debit_reconciles_but_does_not_relabel_unknown_cost(
    tmp_path: Path,
) -> None:
    _complete_fixture(tmp_path)
    with (tmp_path / "spend-ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "event_id": "invalid-smoke:paid",
                    "status": "completed",
                    "model_cost_usd": None,
                    "budget_debit_usd": 300.0,
                }
            )
            + "\n"
        )

    result = audit(tmp_path)

    assert result.completion_status == "complete"
    assert result.known_model_spend_usd == 1.25
    assert result.estimated_model_spend_usd == 0.0
    assert result.conservative_budget_debit_usd == 300.0
    assert result.unknown_cost_events == 1
