"""Tests for full-source BigCodeBench deployment refitting."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load(name: str) -> ModuleType:
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fit = _load("coding_model_router_bigcodebench_fit")
select = _load("coding_model_router_bigcodebench_select")
run = _load("coding_model_router_bigcodebench_select_run")
_load("coding_model_router_bigcodebench_lock")
_load("coding_model_router_bigcodebench_knn_audit")
_load("coding_model_router_bigcodebench_evaluate")
_load("coding_model_router_bigcodebench_numeric_audit")
_load("coding_model_router_bigcodebench_promote")
module = _load("coding_model_router_bigcodebench_deploy")


def _candidate(name: str = "consensus") -> object:
    config_json, config_sha256 = fit.canonical_candidate_config(
        {
            "alpha": 1.0,
            "dim": 256,
            "estimator": "ridge",
            "family": "ordinal",
            "lam": 0.0,
            "learning_rate": 0.0,
            "max_features": "",
            "max_leaf_nodes": 0,
            "min_samples_leaf": 0,
            "n_estimators": 0,
            "prior_strength": 0.0,
            "z": 0.0,
        }
    )
    return run.CandidateRecord.model_construct(
        family="ordinal",
        name=name,
        order=0,
        config_json=config_json,
        config_sha256=config_sha256,
    )


def _report(
    seed: int,
    baseline_arm: str,
    rewards: dict[str, float],
    *,
    candidate: object | None = None,
) -> object:
    controls = [
        run.ControlRecord.model_construct(
            kind="static",
            name=f"static-{arm}",
            reward=rewards[arm],
            cost_usd=float(index + 1),
            arm_counts={},
        )
        for index, arm in enumerate(fit.ARMS)
    ]
    return run.SeedFitReport.model_construct(
        seed=seed,
        baseline_arm=baseline_arm,
        controls=controls,
        candidates=[candidate or _candidate()],
    )


def test_deployment_baseline_uses_majority_before_quality() -> None:
    rewards = {arm: 0.9 for arm in fit.ARMS}
    reports = [_report(seed, "luna-high" if seed < 3 else "luna-max", rewards) for seed in range(5)]
    assert module.deployment_baseline(reports) == "luna-high"


def test_deployment_baseline_breaks_count_tie_by_mean_quality() -> None:
    reports = []
    baselines = ["luna-high", "luna-high", "luna-max", "luna-max", "luna-low"]
    for seed, baseline in enumerate(baselines):
        rewards = {arm: 0.5 for arm in fit.ARMS}
        rewards["luna-high"] = 0.80
        rewards["luna-max"] = 0.85
        reports.append(_report(seed, baseline, rewards))
    assert module.deployment_baseline(reports) == "luna-max"


def test_consensus_record_requires_identical_candidate_identity() -> None:
    candidate = _candidate()
    reports = [
        _report(seed, "luna-high", {arm: 0.8 for arm in fit.ARMS}, candidate=candidate)
        for seed in range(5)
    ]
    consensus = fit.DeploymentConsensus.model_construct(
        family=candidate.family,
        name=candidate.name,
        order=candidate.order,
        config_json=candidate.config_json,
        config_sha256=candidate.config_sha256,
    )
    lock = fit.SelectionLock.model_construct(deployment_consensus=consensus)
    assert module.consensus_record(lock, reports) is candidate
    reports[-1] = reports[-1].model_copy(update={"candidates": [_candidate(name="different")]})
    with pytest.raises(ValueError, match="lacks one exact consensus candidate"):
        module.consensus_record(lock, reports)


def test_seed_reports_must_reproduce_the_selection_lock() -> None:
    candidate = _candidate()
    reports = []
    selections = []
    for seed in range(5):
        report = _report(
            seed,
            "luna-high",
            {arm: 0.8 for arm in fit.ARMS},
            candidate=candidate,
        ).model_copy(
            update={
                "code_commit": "a" * 40,
                "tasks_sha256": "b" * 64,
                "scores_sha256": "c" * 64,
                "outcomes_sha256": "d" * 64,
                "oracle_report_sha256": "e" * 64,
                "fit_tasks": 240,
                "heldout_tasks": 60,
                "fit_ids_sha256": f"{seed}" * 64,
                "heldout_ids_sha256": f"{seed + 1}" * 64,
                "baseline_fit_reward": 0.8,
                "baseline_fit_cost_usd": 1.0,
                "selected_name": candidate.name,
            }
        )
        reports.append(report)
        selections.append(
            fit.SeedSelection.model_construct(
                seed=seed,
                fit_tasks=240,
                heldout_tasks=60,
                fit_ids_sha256=report.fit_ids_sha256,
                heldout_ids_sha256=report.heldout_ids_sha256,
                baseline_arm="luna-high",
                baseline_fit_reward=0.8,
                baseline_fit_cost_usd=1.0,
                selected=fit.LockedCandidate.model_construct(
                    family=candidate.family,
                    name=candidate.name,
                    config_json=candidate.config_json,
                    config_sha256=candidate.config_sha256,
                ),
            )
        )
    lock = fit.SelectionLock.model_construct(
        code_commit="a" * 40,
        tasks_sha256="b" * 64,
        scores_sha256="c" * 64,
        outcomes_sha256="d" * 64,
        oracle_report_sha256="e" * 64,
        seeds=selections,
    )
    module.require_reports_match_lock(lock, reports)
    reports[-1] = reports[-1].model_copy(update={"baseline_fit_cost_usd": 2.0})
    with pytest.raises(ValueError, match="differs from the selection lock"):
        module.require_reports_match_lock(lock, reports)


def test_deployment_report_enforces_the_latency_gate() -> None:
    with pytest.raises(ValueError, match="less than 20"):
        module.DeploymentArtifactReport(
            code_commit="a" * 40,
            selection_lock_sha256="b" * 64,
            external_promotion_sha256="c" * 64,
            family="ordinal",
            candidate_name="candidate",
            config_sha256="d" * 64,
            baseline_arm="luna-high",
            deployment_seed=module.DEPLOYMENT_SEED,
            fit_tasks=300,
            artifact_kind="numeric-router",
            artifact_sha256="e" * 64,
            artifact_bytes=1,
            decisions=10_000,
            latency_p50_ms=1.0,
            latency_p95_ms=20.0,
            latency_passed=True,
        )
