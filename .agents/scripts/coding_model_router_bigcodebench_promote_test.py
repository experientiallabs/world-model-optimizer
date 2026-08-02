"""Tests for deterministic BigCodeBench external promotion auditing."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
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
evaluate = _load("coding_model_router_bigcodebench_evaluate")
module = _load("coding_model_router_bigcodebench_promote")


def _data() -> object:
    tasks = 30
    rewards = np.zeros((tasks, len(fit.ARMS), fit.ATTEMPTS), dtype=np.float64)
    rewards[:15, 0, :] = 1.0
    rewards[15:, 4, :] = 1.0
    costs = np.broadcast_to(
        np.asarray([0.001, 0.002, 0.003, 0.004, 0.005])[None, :, None],
        rewards.shape,
    ).copy()
    return fit.FitData(
        task_ids=[f"task-{index}" for index in range(tasks)],
        groups=[f"group-{index // 2}" for index in range(tasks)],
        texts=[
            f"sql query {index}" if index < 15 else f"async await {index}" for index in range(tasks)
        ],
        is_hard=np.zeros(tasks, dtype=np.bool_),
        rewards=rewards,
        costs=costs,
    )


def _evidence(tmp_path: Path) -> tuple[object, list[object], str]:
    data = _data()
    spec = select.CandidateSpec("ordinal", "ridge", 512, 0, alpha=1.0)
    config_json, config_sha256 = fit.canonical_candidate_config(spec.config())
    code_commit = "a" * 40
    lock_sha256 = "b" * 64
    selections = []
    reports = []
    for split in fit.outer_splits(data.groups):
        fit_ids_sha256, heldout_ids_sha256 = fit.seed_split_provenance(data, split)
        baseline = fit.fit_selected_static(data, split.train_indices)
        candidate = fit.LockedCandidate(
            family="ordinal",
            name=spec.name,
            config_json=config_json,
            config_sha256=config_sha256,
            fit_reward=0.9,
            fit_cost_usd=0.003,
            matched_blind_reward=0.5,
            latency_p95_ms=1.0,
            artifact_bytes=1_024,
        )
        selections.append(
            fit.SeedSelection(
                seed=split.seed,
                fit_tasks=len(split.train_indices),
                heldout_tasks=len(split.test_indices),
                fit_ids_sha256=fit_ids_sha256,
                heldout_ids_sha256=heldout_ids_sha256,
                baseline_arm=baseline.name,
                baseline_fit_reward=baseline.reward,
                baseline_fit_cost_usd=baseline.cost_usd,
                selected=candidate,
            )
        )
        reports.append(
            evaluate.seed_heldout_report(
                data,
                split,
                spec,
                code_commit=code_commit,
                selection_lock_sha256=lock_sha256,
                seed_fit_report_sha256=f"{split.seed}" * 64,
                winner_audit_sha256=f"{split.seed + 1}" * 64,
                candidate_config_sha256=config_sha256,
                work_dir=tmp_path / f"seed-{split.seed}",
            )
        )
    consensus = fit.DeploymentConsensus(
        family="ordinal",
        name=spec.name,
        order=spec.order,
        config_json=config_json,
        config_sha256=config_sha256,
        mean_fit_reward=0.9,
        mean_fit_cost_usd=0.003,
        mean_matched_blind_reward=0.5,
        mean_baseline_reward=0.9,
        minimum_seed_retention=1.0,
        fit_quality_feasible=True,
    )
    lock = fit.SelectionLock(
        protocol="bigcodebench-fit-only-selection-v1",
        tasks_sha256="c" * 64,
        scores_sha256="d" * 64,
        outcomes_sha256="e" * 64,
        oracle_report_sha256="f" * 64,
        code_commit=code_commit,
        seeds=selections,
        deployment_consensus=consensus,
    )
    return lock, reports, lock_sha256


def test_external_promotion_is_deterministic_and_target_safe(tmp_path: Path) -> None:
    lock, reports, lock_sha256 = _evidence(tmp_path)
    first = module.analyze_external_promotion(
        lock,
        lock_sha256,
        reports,
        samples=200,
    )
    second = module.analyze_external_promotion(
        lock,
        lock_sha256,
        reports,
        samples=200,
    )
    assert first == second
    assert len(first.seed_gates) == 5
    assert len(first.control_gates) == 4
    assert first.target_outcomes_used is False


def test_external_promotion_rejects_tampered_aggregate(tmp_path: Path) -> None:
    lock, reports, lock_sha256 = _evidence(tmp_path)
    report = reports[0]
    reports[0] = report.model_copy(
        update={"router": report.router.model_copy(update={"reward": 0.123})}
    )
    with pytest.raises(ValueError, match="aggregate differs"):
        module.analyze_external_promotion(
            lock,
            lock_sha256,
            reports,
            samples=200,
        )


def test_grouped_interval_requires_multiple_families() -> None:
    with pytest.raises(ValueError, match="at least two"):
        module._interval(
            ["one", "one"],
            np.asarray([1.0, 0.0]),
            np.asarray([1.0, 1.0]),
            kind="delta",
            samples=100,
            seed=1,
        )


def test_external_promotion_report_is_immutable_and_resumable(tmp_path: Path) -> None:
    lock, reports, lock_sha256 = _evidence(tmp_path)
    output = tmp_path / "external-promotion.json"
    first = module.write_external_promotion_report(
        lock,
        selection_lock_sha256=lock_sha256,
        reports=reports,
        output=output,
        samples=200,
    )
    original = output.read_bytes()
    second = module.write_external_promotion_report(
        lock,
        selection_lock_sha256=lock_sha256,
        reports=reports,
        output=output,
        samples=200,
    )
    assert first == second
    assert output.read_bytes() == original
