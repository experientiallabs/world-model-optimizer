"""Tests for the coding-router final report renderer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from coding_model_router_report import build_report, write_report


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(
    *,
    quality: float,
    cost: float,
    mix: dict[str, float],
    route_away: float = 0.0,
    guard_reversion: float = 0.0,
    novelty: float = 0.0,
) -> dict[str, object]:
    return {
        "quality": quality,
        "cost_per_task": cost,
        "total_cost": cost * 10,
        "effective_cost_per_success": cost / max(quality, 0.1),
        "success_rate": quality,
        "latency_p50_s": 10.0,
        "latency_p95_s": 20.0,
        "model_mix": mix,
        "route_away_rate": route_away,
        "guard_reversion_rate": guard_reversion,
        "novelty_abstention_rate": novelty,
        "per_benchmark": {
            benchmark: {
                "quality": quality,
                "cost_per_task": cost,
                "success_rate": quality,
                "scenarios": 5.0,
            }
            for benchmark in ("terminal-bench-2", "swe-bench-verified")
        },
        "scenarios": 10,
    }


def _complete_fixture(root: Path) -> None:
    outcomes_path = root / "full" / "outcomes.json"
    _write_json(
        outcomes_path,
        {
            "outcomes": [
                {
                    "scenario_id": "terminal-bench-2:one",
                    "model": "frontier",
                    "reward": 1.0,
                    "cost_usd": 1.0,
                    "usage_accounting": "exact",
                },
                {
                    "scenario_id": "swe-bench-verified:two",
                    "model": "cheap",
                    "reward": 1.0,
                    "cost_usd": 0.2,
                    "usage_accounting": "estimated",
                },
            ]
        },
    )
    lock_path = root / "analysis" / "selection-lock.json"
    _write_json(
        lock_path,
        {
            "matrix_sha256": _sha256(outcomes_path),
            "deployment_consensus_baseline": "frontier",
        },
    )
    points = {
        "best_single": _metric(quality=1.0, cost=1.0, mix={"frontier": 1.0}),
        "cheapest_single": _metric(quality=0.7, cost=0.2, mix={"cheap": 1.0}),
        "unguarded_knn": _metric(
            quality=0.93,
            cost=0.4,
            mix={"frontier": 0.3, "cheap": 0.7},
            route_away=0.7,
        ),
        "guarded_knn": _metric(
            quality=0.96,
            cost=0.5,
            mix={"frontier": 0.4, "cheap": 0.6},
            route_away=0.6,
            guard_reversion=0.2,
            novelty=0.1,
        ),
        "oracle": _metric(
            quality=1.0,
            cost=0.3,
            mix={"frontier": 0.2, "cheap": 0.8},
            route_away=0.8,
        ),
    }
    outer_path = root / "analysis" / "outer-results.json"
    _write_json(
        outer_path,
        {
            "matrix_sha256": _sha256(outcomes_path),
            "selection_lock_sha256": _sha256(lock_path),
            "seeds": [{"seed": seed, "points": points} for seed in range(5)],
            "pareto": [
                {
                    "id": "guarded_knn",
                    "quality": 0.96,
                    "cost_per_task": 0.5,
                    "on_frontier": True,
                }
            ],
            "promoted": True,
            "paired_cluster_bootstrap": {"retention_lower_95": 0.951},
            "capability_slices": {
                "repository-level-bug-fixing": {
                    "seeds_observed": 5,
                    "points": {
                        "guarded_knn": {
                            "quality": 0.96,
                            "baseline_quality": 1.0,
                            "quality_retained": 0.96,
                            "absolute_quality_delta": -0.04,
                            "cost_per_task": 0.5,
                            "baseline_cost_per_task": 1.0,
                            "cost_savings": 0.5,
                            "success_rate": 0.96,
                            "latency_p50_s": 10.0,
                            "latency_p95_s": 20.0,
                            "scenarios": 5.0,
                            "model_mix": {"frontier": 0.4, "cheap": 0.6},
                        }
                    },
                }
            },
            "one_at_a_time_ablations": {
                "benchmark_stratified": "ablation:benchmark_stratified",
                "missing_fit_coverage_0.8": "ablation:missing_fit_coverage_0.8",
                "missing_fit_coverage_1.0_control": "guarded_knn",
                "latency_only_static": "latency_only",
                "production_eligible": False,
            },
            "deployment_consensus_baseline": "frontier",
            "deployment_consensus_config": {"neighbors": 8},
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
    _write_json(
        root / "world-model" / "comparison.json",
        {
            "protocol": "coding-world-model-compare-v1",
            "promotion_agreement": True,
            "deployment_consensus_config_agreement": True,
            "deployment_consensus_baseline_agreement": True,
        },
    )


def test_report_fails_closed_without_complete_evidence(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="required evidence is missing"):
        build_report(tmp_path)

    assert not (tmp_path / "final-report.md").exists()
    assert not (tmp_path / "final-summary.json").exists()


def test_report_renders_required_rows_and_traceable_summary(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)

    report_path, summary_path = write_report(tmp_path)

    report = report_path.read_text(encoding="utf-8")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "| Best Single |" in report
    assert "| Cheapest Single |" in report
    assert "| Unguarded Router |" in report
    assert "| Guarded Router |" in report
    assert "| Selected Pareto Point" in report
    assert "| Oracle Upper Bound |" in report
    assert "The target was achieved." in report
    assert "Cost comparisons are approximate" in report
    assert "| repository-level-bug-fixing | 5/5 |" in report
    assert summary["target_achieved"] is True
    assert summary["selected_policy"]["id"] == "guarded_knn"
    assert summary["world_model_same_deployment_decision"] is True
    assert summary["cost_accounting"]["estimated_cells"] == 1
    assert summary["unsafe_capabilities"] == []


def test_report_rejects_changed_outer_results(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)
    outer_path = tmp_path / "analysis" / "outer-results.json"
    outer = json.loads(outer_path.read_text(encoding="utf-8"))
    outer["promoted"] = False
    _write_json(outer_path, outer)

    with pytest.raises(ValueError, match="completion evidence"):
        build_report(tmp_path)
