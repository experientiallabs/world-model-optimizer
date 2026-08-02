"""Tests for leakage-safe BigCodeBench outer-heldout replay."""

from __future__ import annotations

import hashlib
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
select_run = _load("coding_model_router_bigcodebench_select_run")
lock_module = _load("coding_model_router_bigcodebench_lock")
module = _load("coding_model_router_bigcodebench_evaluate")


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


def _split() -> tuple[np.ndarray, np.ndarray]:
    return np.arange(24, dtype=np.int64), np.arange(24, 30, dtype=np.int64)


def test_non_knn_routes_do_not_depend_on_heldout_rewards(tmp_path: Path) -> None:
    data = _data()
    train, heldout = _split()
    spec = select.CandidateSpec("ordinal", "ridge", 512, 0, alpha=1.0)
    initial = module.replay_outer_heldout(
        data,
        train,
        heldout,
        spec,
        seed=7,
        work_dir=tmp_path / "initial",
    )
    changed_rewards = data.rewards.copy()
    changed_rewards[heldout] = 1.0 - changed_rewards[heldout]
    changed = fit.FitData(
        task_ids=data.task_ids,
        groups=data.groups,
        texts=data.texts,
        is_hard=data.is_hard,
        rewards=changed_rewards,
        costs=data.costs,
    )
    replay = module.replay_outer_heldout(
        changed,
        train,
        heldout,
        spec,
        seed=7,
        work_dir=tmp_path / "changed",
    )
    assert np.array_equal(initial.choices, replay.choices)
    assert initial.value.reward != replay.value.reward


def test_knn_routes_do_not_depend_on_heldout_rewards(tmp_path: Path) -> None:
    data = _data()
    train, heldout = _split()
    spec = select.KnnCandidateSpec(512, 8, 0.9, 0.0, 3, 0)
    initial = module.replay_outer_heldout(
        data,
        train,
        heldout,
        spec,
        seed=7,
        work_dir=tmp_path / "initial",
    )
    changed_rewards = data.rewards.copy()
    changed_rewards[heldout] = 1.0 - changed_rewards[heldout]
    changed = fit.FitData(
        task_ids=data.task_ids,
        groups=data.groups,
        texts=data.texts,
        is_hard=data.is_hard,
        rewards=changed_rewards,
        costs=data.costs,
    )
    replay = module.replay_outer_heldout(
        changed,
        train,
        heldout,
        spec,
        seed=7,
        work_dir=tmp_path / "changed",
    )
    assert np.array_equal(initial.choices, replay.choices)


def test_outer_replay_rejects_group_overlap(tmp_path: Path) -> None:
    data = _data()
    spec = select.CandidateSpec("ordinal", "ridge", 512, 0, alpha=1.0)
    with pytest.raises(ValueError, match="group crossed"):
        module.replay_outer_heldout(
            data,
            np.arange(23),
            np.arange(23, 30),
            spec,
            seed=7,
            work_dir=tmp_path,
        )


def test_every_frozen_base_candidate_round_trips_from_lock() -> None:
    candidates = [*select.candidate_grid(), *select.knn_candidate_grid()]
    for candidate in candidates:
        config_json, _ = fit.canonical_candidate_config(candidate.config())
        rebuilt = module.candidate_spec_from_lock(
            "knn" if isinstance(candidate, select.KnnCandidateSpec) else candidate.family,
            config_json,
            name=candidate.name,
            order=candidate.order,
        )
        assert rebuilt == candidate


def test_economic_knn_candidate_round_trips_from_lock() -> None:
    candidate = select.KnnCandidateSpec(
        2_048,
        32,
        0.95,
        1.0,
        16,
        1_027,
        guard_model="luna-low",
        guard_mode="asymmetric",
        pick_lam=0.03,
    )
    config_json, _ = fit.canonical_candidate_config(candidate.config())
    rebuilt = module.candidate_spec_from_lock(
        "knn",
        config_json,
        name=candidate.name,
        order=candidate.order,
    )
    assert rebuilt == candidate


def test_seed_report_contains_exact_heldout_controls(tmp_path: Path) -> None:
    data = _data()
    train, heldout = _split()
    split = fit.TaskSplit(seed=0, train_indices=train, test_indices=heldout)
    spec = select.CandidateSpec("ordinal", "ridge", 512, 0, alpha=1.0)
    _, config_sha256 = fit.canonical_candidate_config(spec.config())
    report = module.seed_heldout_report(
        data,
        split,
        spec,
        code_commit="a" * 40,
        selection_lock_sha256="b" * 64,
        seed_fit_report_sha256="c" * 64,
        winner_audit_sha256="d" * 64,
        candidate_config_sha256=config_sha256,
        work_dir=tmp_path,
    )
    assert report.heldout_tasks == len(heldout)
    assert len(report.controls) == 9
    assert {control.kind for control in report.controls} == {
        "static",
        "matched-task-blind",
        "random",
        "cost-only",
        "shuffled-label",
    }
    assert all(sum(control.arm_counts.values()) == len(heldout) for control in report.controls)


def test_seed_report_rejects_missing_control(tmp_path: Path) -> None:
    data = _data()
    train, heldout = _split()
    split = fit.TaskSplit(seed=0, train_indices=train, test_indices=heldout)
    spec = select.CandidateSpec("ordinal", "ridge", 512, 0, alpha=1.0)
    _, config_sha256 = fit.canonical_candidate_config(spec.config())
    report = module.seed_heldout_report(
        data,
        split,
        spec,
        code_commit="a" * 40,
        selection_lock_sha256="b" * 64,
        seed_fit_report_sha256="c" * 64,
        winner_audit_sha256="d" * 64,
        candidate_config_sha256=config_sha256,
        work_dir=tmp_path,
    )
    value = report.model_dump()
    value["controls"] = value["controls"][:-1]
    with pytest.raises(ValueError, match="at least 9 items"):
        module.SeedHeldoutReport.model_validate(value)


def test_five_seed_outer_replay_is_resumable_and_lock_bound(tmp_path: Path) -> None:
    data = _data()
    selected_spec = select.candidate_grid()[0]
    specs = [
        *select.candidate_grid(),
        *select.knn_candidate_grid(),
        *select.knn_economic_grid(select.knn_candidate_grid()[0]),
    ]
    candidates = []
    for spec in specs:
        config_json, config_sha256 = fit.canonical_candidate_config(spec.config())
        candidates.append(
            select_run.CandidateRecord(
                family="knn" if isinstance(spec, select.KnnCandidateSpec) else spec.family,
                name=spec.name,
                order=spec.order,
                config_json=config_json,
                config_sha256=config_sha256,
                fit_reward=0.8,
                fit_cost_usd=0.003,
                matched_blind_reward=0.5,
                matched_blind_cost_usd=0.003,
                baseline_reward=0.8,
                baseline_cost_usd=0.005,
            )
        )
    assert len(candidates) == 1_028
    selected = next(candidate for candidate in candidates if candidate.name == selected_spec.name)
    code_commit = "a" * 40
    report_paths = []
    audit_paths = []
    selections = []
    for split in fit.outer_splits(data.groups):
        fit_ids_sha256, heldout_ids_sha256 = fit.seed_split_provenance(data, split)
        fit_tasks = len(split.train_indices)
        controls = []
        for arm_index, arm in enumerate(fit.ARMS):
            controls.append(
                select_run.ControlRecord(
                    kind="static",
                    name=f"static-{arm}",
                    reward=0.8,
                    cost_usd=0.005,
                    arm_counts={
                        value: fit_tasks if index == arm_index else 0
                        for index, value in enumerate(fit.ARMS)
                    },
                )
            )
        mixed_counts = {
            arm: fit_tasks // len(fit.ARMS) + (index < fit_tasks % len(fit.ARMS))
            for index, arm in enumerate(fit.ARMS)
        }
        controls.extend(
            [
                select_run.ControlRecord(
                    kind="matched-task-blind",
                    name="selected-matched-task-blind",
                    reward=0.5,
                    cost_usd=0.003,
                    arm_counts=mixed_counts,
                ),
                select_run.ControlRecord(
                    kind="random",
                    name="seeded-uniform-random",
                    reward=0.5,
                    cost_usd=0.003,
                    arm_counts=mixed_counts,
                ),
                select_run.ControlRecord(
                    kind="cost-only",
                    name="fit-cost-only",
                    reward=0.5,
                    cost_usd=0.001,
                    arm_counts={
                        arm: fit_tasks if index == 0 else 0 for index, arm in enumerate(fit.ARMS)
                    },
                ),
                select_run.ControlRecord(
                    kind="shuffled-label",
                    name="selected-shuffled-labels",
                    reward=0.5,
                    cost_usd=0.003,
                    arm_counts=mixed_counts,
                ),
            ]
        )
        report = select_run.SeedFitReport(
            seed=split.seed,
            code_commit=code_commit,
            tasks_sha256="b" * 64,
            scores_sha256="c" * 64,
            outcomes_sha256="d" * 64,
            oracle_report_sha256="e" * 64,
            fit_tasks=fit_tasks,
            heldout_tasks=len(split.test_indices),
            fit_ids_sha256=fit_ids_sha256,
            heldout_ids_sha256=heldout_ids_sha256,
            baseline_arm="luna-max",
            baseline_fit_reward=0.8,
            baseline_fit_cost_usd=0.005,
            candidates=candidates,
            controls=controls,
            selected_name=selected.name,
        )
        report_path = tmp_path / f"fit-{split.seed}.json"
        report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
        audit = lock_module.SeedWinnerAudit(
            seed=split.seed,
            seed_report_sha256=report_sha256,
            candidate_name=selected.name,
            config_sha256=selected.config_sha256,
            artifact_kind="numeric-router",
            artifact_sha256=str(split.seed) * 64,
            artifact_bytes=1_024,
            decisions=10_000,
            latency_p50_ms=0.5,
            latency_p95_ms=1.0,
            latency_passed=True,
        )
        audit_path = tmp_path / f"audit-{split.seed}.json"
        audit_path.write_text(audit.model_dump_json(indent=2) + "\n", encoding="utf-8")
        report_paths.append(report_path)
        audit_paths.append(audit_path)
        selections.append(
            fit.SeedSelection(
                seed=split.seed,
                fit_tasks=fit_tasks,
                heldout_tasks=len(split.test_indices),
                fit_ids_sha256=fit_ids_sha256,
                heldout_ids_sha256=heldout_ids_sha256,
                baseline_arm="luna-max",
                baseline_fit_reward=0.8,
                baseline_fit_cost_usd=0.005,
                selected=fit.LockedCandidate(
                    family=selected.family,
                    name=selected.name,
                    config_json=selected.config_json,
                    config_sha256=selected.config_sha256,
                    fit_reward=selected.fit_reward,
                    fit_cost_usd=selected.fit_cost_usd,
                    matched_blind_reward=selected.matched_blind_reward,
                    latency_p95_ms=1.0,
                    artifact_bytes=1_024,
                ),
            )
        )
    consensus = fit.DeploymentConsensus(
        family=selected.family,
        name=selected.name,
        order=selected.order,
        config_json=selected.config_json,
        config_sha256=selected.config_sha256,
        mean_fit_reward=0.8,
        mean_fit_cost_usd=0.003,
        mean_matched_blind_reward=0.5,
        mean_baseline_reward=0.8,
        minimum_seed_retention=1.0,
        fit_quality_feasible=True,
    )
    lock = fit.SelectionLock(
        protocol="bigcodebench-fit-only-selection-v1",
        tasks_sha256="b" * 64,
        scores_sha256="c" * 64,
        outcomes_sha256="d" * 64,
        oracle_report_sha256="e" * 64,
        code_commit=code_commit,
        seeds=selections,
        deployment_consensus=consensus,
    )
    output_dir = tmp_path / "heldout"
    first = module.write_outer_heldout_reports(
        data,
        lock,
        selection_lock_sha256="f" * 64,
        report_paths=report_paths,
        audit_paths=audit_paths,
        output_dir=output_dir,
    )
    original = [(output_dir / f"seed-{seed}.json").read_bytes() for seed in range(5)]
    second = module.write_outer_heldout_reports(
        data,
        lock,
        selection_lock_sha256="f" * 64,
        report_paths=report_paths,
        audit_paths=audit_paths,
        output_dir=output_dir,
    )
    assert first == second
    assert original == [(output_dir / f"seed-{seed}.json").read_bytes() for seed in range(5)]
