"""Focused scientific-integrity tests for the coding-router analysis runner."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
from coding_model_router_analyze import (
    BENCHMARKS,
    INNER_FOLDS,
    RouterConfig,
    _aggregate_capability_slices,
    _best_single,
    _capability_slice_ids,
    _inner_evaluate,
    _missing_fit_matrix,
    _seed_evaluation,
)

from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.providers.pool import PoolEntry


def _pool() -> list[PoolEntry]:
    return [
        PoolEntry(
            name=name,
            kind="openai",
            model=f"test-{name}",
            input_per_mtok=1.0 if name == "a" else 0.5,
            output_per_mtok=1.0,
        )
        for name in ("a", "b")
    ]


def _outcome(scenario_id: str, model: str, reward: float) -> ScenarioOutcome:
    return ScenarioOutcome(
        scenario_id=scenario_id,
        task=(f"pre-call task {scenario_id} compile write- model fix-secret-leak-recovery"),
        model=model,
        benchmark=scenario_id.split(":", 1)[0],
        reward=reward,
        success=bool(reward),
        cost_usd=1.0 if model == "a" else 0.5,
        call_seconds=[1.0],
    )


def test_best_single_uses_equal_benchmark_weights() -> None:
    outcomes: list[ScenarioOutcome] = []
    tb_id = "terminal-bench-2:tb"
    for model, reward in (("a", 1.0), ("b", 0.4)):
        outcomes.append(_outcome(tb_id, model, reward))
    ids = [tb_id]
    for index in range(9):
        scenario_id = f"swe-bench-verified:swe-{index}"
        ids.append(scenario_id)
        for model, reward in (("a", 0.0), ("b", 0.4)):
            outcomes.append(_outcome(scenario_id, model, reward))
    matrix = OutcomeMatrix(pool=_pool(), outcomes=outcomes)
    assert _best_single(matrix, ids) == "a"


def _group_for_fold(seed: int, benchmark: str, fold: int) -> str:
    for index in range(10_000):
        group = f"group-{fold}-{index}"
        digest = hashlib.sha256(f"inner-v1:{seed}:{benchmark}:{group}".encode()).digest()
        if int.from_bytes(digest[:8], "big") % INNER_FOLDS == fold:
            return group
    raise AssertionError("failed to find a deterministic fold group")


def _nested_fixture() -> tuple[
    OutcomeMatrix,
    dict[str, dict[str, dict[str, object]]],
    list[str],
]:
    outcomes: list[ScenarioOutcome] = []
    manifests: dict[str, dict[str, dict[str, object]]] = {benchmark: {} for benchmark in BENCHMARKS}
    ids: list[str] = []
    for benchmark in BENCHMARKS:
        for fold in range(INNER_FOLDS):
            task_id = f"task-{fold}"
            scenario_id = f"{benchmark}:{task_id}"
            ids.append(scenario_id)
            manifests[benchmark][task_id] = {
                "task_id": task_id,
                "group": _group_for_fold(0, benchmark, fold),
            }
            outcomes.extend(
                (
                    _outcome(scenario_id, "a", 0.8),
                    _outcome(scenario_id, "b", 0.6 + 0.05 * (fold % 2)),
                )
            )
    return OutcomeMatrix(pool=_pool(), outcomes=outcomes), manifests, ids


def test_inner_selection_excludes_rows_outside_outer_fit(tmp_path: Path) -> None:
    matrix, manifests, fit_ids = _nested_fixture()
    config = RouterConfig(neighbors=4, min_pairs=3)
    fit_router, fit_baseline = _inner_evaluate(
        matrix,
        manifests,
        root=tmp_path,
        seed=0,
        fit_ids=fit_ids,
        baseline="a",
        config=config,
    )

    leaked = "terminal-bench-2:outer-heldout"
    manifests["terminal-bench-2"]["outer-heldout"] = {
        "task_id": "outer-heldout",
        "group": "outer-heldout",
    }
    matrix_with_heldout = matrix.model_copy(
        update={
            "outcomes": [
                *matrix.outcomes,
                _outcome(leaked, "a", 0.0),
                _outcome(leaked, "b", 1.0),
            ]
        }
    )
    rerun_router, rerun_baseline = _inner_evaluate(
        matrix_with_heldout,
        manifests,
        root=tmp_path,
        seed=0,
        fit_ids=fit_ids,
        baseline="a",
        config=config,
    )
    assert rerun_router == fit_router
    assert rerun_baseline == fit_baseline
    assert fit_router.scenarios == len(fit_ids)
    assert fit_baseline.quality == pytest.approx(0.8)


def test_capability_slices_use_only_pre_call_task_information() -> None:
    scenario_ids = (
        "swe-bench-verified:repo-fix",
        "terminal-bench-2:compile-build-system",
        "terminal-bench-2:write-compiler",
        "terminal-bench-2:pytorch-model-inference",
        "terminal-bench-2:fix-secret-leak-recovery",
    )
    outcomes = [
        _outcome(scenario_id, model, 1.0) for scenario_id in scenario_ids for model in ("a", "b")
    ]
    matrix = OutcomeMatrix(pool=_pool(), outcomes=outcomes)

    slices = _capability_slice_ids(matrix)

    assert set(slices) == {
        "build-and-dependency",
        "code-generation-and-translation",
        "data-ml-and-scientific",
        "debugging-and-test-repair",
        "long-context",
        "repository-level-bug-fixing",
        "security-and-recovery",
        "terminal-operation-and-tool-use",
    }
    assert scenario_ids[0] in slices["repository-level-bug-fixing"]
    assert scenario_ids[1] in slices["build-and-dependency"]
    assert scenario_ids[2] in slices["code-generation-and-translation"]
    assert scenario_ids[3] in slices["data-ml-and-scientific"]
    assert scenario_ids[4] in slices["security-and-recovery"]


def test_missing_cell_ablation_preserves_baseline_and_exact_coverage() -> None:
    matrix, _manifests, ids = _nested_fixture()

    masked = _missing_fit_matrix(
        matrix,
        ids,
        baseline="a",
        seed=0,
        coverage=0.8,
    )

    by_model = {
        model: {outcome.scenario_id for outcome in masked.outcomes if outcome.model == model}
        for model in ("a", "b")
    }
    assert by_model["a"] == set(ids)
    assert len(by_model["b"]) == int(len(ids) * 0.8)
    assert masked == _missing_fit_matrix(
        matrix,
        ids,
        baseline="a",
        seed=0,
        coverage=0.8,
    )


def test_seed_evaluation_persists_capability_and_robustness_ablations(
    tmp_path: Path,
) -> None:
    matrix, _manifests, _ids = _nested_fixture()
    split: dict[str, dict[str, list[str]]] = {}
    for benchmark in BENCHMARKS:
        task_ids = [f"task-{index}" for index in range(INNER_FOLDS)]
        split[benchmark] = {
            "fit": task_ids[:3],
            "heldout": task_ids[3:],
        }
    split_path = tmp_path / "splits" / "seed-0.json"
    split_path.parent.mkdir(parents=True)
    split_path.write_text(json.dumps(split), encoding="utf-8")

    result = _seed_evaluation(
        matrix,
        tmp_path,
        seed=0,
        lock_row={
            "baseline": "a",
            "config": asdict(RouterConfig(neighbors=4, min_pairs=3)),
        },
    )

    points = result["points"]
    assert isinstance(points, dict)
    assert "latency_only" in points
    assert "ablation:benchmark_stratified" in points
    assert "ablation:missing_fit_coverage_0.8" in points
    slices = result["capability_slices"]
    assert isinstance(slices, dict)
    assert set(slices) == {
        "build-and-dependency",
        "code-generation-and-translation",
        "data-ml-and-scientific",
        "debugging-and-test-repair",
        "long-context",
        "repository-level-bug-fixing",
        "security-and-recovery",
        "terminal-operation-and-tool-use",
    }
    aggregate = _aggregate_capability_slices([{**result, "seed": seed} for seed in range(5)])
    assert aggregate["repository-level-bug-fixing"]["seeds_observed"] == 5
