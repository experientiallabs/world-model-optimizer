"""Tests for grouped fit-only BigCodeBench candidate selection."""

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
module = _load("coding_model_router_bigcodebench_select")


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
            f"sql select join query {index}" if index < 15 else f"async coroutine await {index}"
            for index in range(tasks)
        ],
        is_hard=np.zeros(tasks, dtype=np.bool_),
        rewards=rewards,
        costs=costs,
    )


def test_candidate_grid_is_complete_and_unique() -> None:
    candidates = module.candidate_grid()
    assert len(candidates) == 576
    assert len({candidate.name for candidate in candidates}) == 576
    assert {candidate.family for candidate in candidates} == {
        "ordinal",
        "doubly-robust",
        "empirical-bayes",
    }
    knn = module.knn_candidate_grid()
    assert len(knn) == 432
    assert len({candidate.name for candidate in knn}) == 432
    economic = module.knn_economic_grid(knn[0])
    assert len(economic) == 20
    assert len({candidate.name for candidate in economic}) == 20
    assert {candidate.guard_model for candidate in economic} == set(fit.ARMS)
    assert {candidate.pick_lam for candidate in economic} == {0.0, 0.01, 0.02, 0.03}


def test_grouped_oof_candidate_has_no_unfilled_routes() -> None:
    spec = module.CandidateSpec(
        family="ordinal",
        estimator="ridge",
        dim=512,
        alpha=1.0,
        order=0,
    )
    result = module.evaluate_candidate_oof(
        _data(),
        np.arange(30),
        spec,
        seed=3,
    )
    assert result.value.arm_counts.values()
    assert sum(result.value.arm_counts.values()) == 30
    assert result.baseline.reward > 0.0


def test_small_fit_only_search_selects_one_candidate() -> None:
    candidates = [
        module.CandidateSpec(
            family="ordinal",
            estimator="ridge",
            dim=512,
            alpha=alpha,
            order=order,
        )
        for order, alpha in enumerate((0.1, 10.0))
    ]
    selected, results = module.select_non_knn_candidate(
        _data(),
        np.arange(30),
        candidates,
        seed=4,
    )
    assert len(results) == 2
    assert selected in results


def test_small_knn_search_reuses_one_bank_per_fold(tmp_path: Path) -> None:
    candidates = [
        module.KnnCandidateSpec(512, 8, 0.9, z, 8, order)
        for order, z in enumerate((0.0, 0.5), start=576)
    ]
    selected, results = module.select_knn_candidate(
        _data(),
        np.arange(30),
        candidates,
        seed=2,
        work_dir=tmp_path,
    )
    assert len(results) == 2
    assert selected in results
    assert len(list(tmp_path.rglob("*.bank.npz"))) == 5


def test_knn_economic_refinement_is_fit_only_and_keeps_base_feasible(tmp_path: Path) -> None:
    base, _ = module.select_knn_candidate(
        _data(),
        np.arange(30),
        [module.KnnCandidateSpec(512, 8, 0.9, 0.0, 8, 576)],
        seed=1,
        work_dir=tmp_path / "base",
    )
    selected, refinements = module.select_knn_economic_refinement(
        _data(),
        np.arange(30),
        base,
        seed=1,
        work_dir=tmp_path / "economic",
    )
    assert len(refinements) == 20
    assert selected in [base, *refinements]
    assert selected.value.reward >= 0.95 * selected.baseline.reward
    assert len(list((tmp_path / "economic").rglob("*.bank.npz"))) == 5
