"""Tests for durable remote BigCodeBench seed-selection reports."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


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
module = _load("coding_model_router_bigcodebench_select_run")


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


def _validation(
    spec: object,
    *,
    reward: float,
    cost: float,
    baseline_reward: float = 0.8,
) -> object:
    value = fit.PolicyValue(reward, cost, reward - 0.01, cost, {fit.ARMS[0]: 1})
    baseline = fit.PolicyValue(
        baseline_reward,
        1.0,
        baseline_reward,
        1.0,
        {fit.ARMS[-1]: 1},
    )
    return select.CandidateValidation(
        spec=spec,
        value=value,
        baseline=baseline,
        metric=fit.CandidateMetric(spec.name, reward, cost, 0.0, 0, spec.order),
    )


def _controls(fit_tasks: int = 240) -> list[object]:
    kinds_and_names = [
        *(("static", f"static-{arm}") for arm in fit.ARMS),
        ("matched-task-blind", "selected-matched-task-blind"),
        ("random", "seeded-uniform-random"),
        ("cost-only", "fit-cost-only"),
        ("shuffled-label", "selected-shuffled-labels"),
    ]
    return [
        module.ControlRecord(
            kind=kind,
            name=name,
            reward=0.7,
            cost_usd=0.3,
            arm_counts={arm: fit_tasks if arm == fit.ARMS[0] else 0 for arm in fit.ARMS},
        )
        for kind, name in kinds_and_names
    ]


def test_family_winner_uses_the_frozen_quality_floor() -> None:
    non_knn = _validation(
        select.CandidateSpec("ordinal", "ridge", 512, 0, alpha=1.0),
        reward=0.77,
        cost=0.6,
    )
    knn = _validation(
        select.KnnCandidateSpec(512, 8, 0.9, 0.5, 8, 576),
        reward=0.76,
        cost=0.4,
    )
    assert module.select_family_winner(non_knn, knn) is knn


def test_candidate_record_canonicalizes_config() -> None:
    result = _validation(
        select.KnnCandidateSpec(512, 8, 0.9, 0.5, 8, 576),
        reward=0.76,
        cost=0.4,
    )
    record = module.candidate_record(result)
    assert record.family == "knn"
    assert record.name == result.spec.name
    assert len(record.config_sha256) == 64
    assert record.fit_reward == 0.76


def test_fit_controls_include_label_destruction_and_effort_mixing(tmp_path: Path) -> None:
    spec = select.CandidateSpec("ordinal", "ridge", 512, 0, alpha=1.0)
    selected = select.evaluate_candidate_oof(_data(), np.arange(30), spec, seed=1)
    controls = module.fit_controls(
        _data(),
        np.arange(30),
        selected,
        seed=1,
        work_dir=tmp_path,
    )
    assert len(controls) == 9
    assert {control.kind for control in controls} == {
        "static",
        "matched-task-blind",
        "random",
        "cost-only",
        "shuffled-label",
    }


def test_seed_report_rejects_duplicate_candidate_names() -> None:
    result = _validation(
        select.KnnCandidateSpec(512, 8, 0.9, 0.5, 8, 576),
        reward=0.76,
        cost=0.4,
    )
    record = module.candidate_record(result)
    values = [record.model_copy(update={"name": f"candidate-{index}"}) for index in range(1_028)]
    values[-1] = values[0]
    with np.testing.assert_raises(ValueError):
        module.SeedFitReport(
            seed=0,
            code_commit="a" * 40,
            tasks_sha256="b" * 64,
            scores_sha256="c" * 64,
            outcomes_sha256="d" * 64,
            oracle_report_sha256="e" * 64,
            fit_tasks=240,
            heldout_tasks=60,
            fit_ids_sha256="f" * 64,
            heldout_ids_sha256="0" * 64,
            baseline_arm=fit.ARMS[-1],
            baseline_fit_reward=0.8,
            baseline_fit_cost_usd=1.0,
            candidates=values,
            controls=_controls(),
            selected_name=values[0].name,
        )


def test_seed_report_rejects_incomplete_control_traffic() -> None:
    result = _validation(
        select.KnnCandidateSpec(512, 8, 0.9, 0.5, 8, 576),
        reward=0.76,
        cost=0.4,
    )
    record = module.candidate_record(result)
    values = [record.model_copy(update={"name": f"candidate-{index}"}) for index in range(1_028)]
    controls = _controls()
    controls[-1] = controls[-1].model_copy(update={"arm_counts": {arm: 0 for arm in fit.ARMS}})
    with np.testing.assert_raises(ValueError):
        module.SeedFitReport(
            seed=0,
            code_commit="a" * 40,
            tasks_sha256="b" * 64,
            scores_sha256="c" * 64,
            outcomes_sha256="d" * 64,
            oracle_report_sha256="e" * 64,
            fit_tasks=240,
            heldout_tasks=60,
            fit_ids_sha256="f" * 64,
            heldout_ids_sha256="0" * 64,
            baseline_arm=fit.ARMS[-1],
            baseline_fit_reward=0.8,
            baseline_fit_cost_usd=1.0,
            candidates=values,
            controls=controls,
            selected_name=values[0].name,
        )
