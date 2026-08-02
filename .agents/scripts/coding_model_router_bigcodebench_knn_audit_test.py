"""Tests for native WMO BigCodeBench kNN artifact auditing."""

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
run = _load("coding_model_router_bigcodebench_select_run")
module = _load("coding_model_router_bigcodebench_knn_audit")


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


def _record() -> object:
    spec = select.KnnCandidateSpec(512, 8, 0.9, 0.5, 8, 576)
    value = fit.PolicyValue(0.8, 0.3, 0.79, 0.3, {fit.ARMS[0]: 30})
    result = select.CandidateValidation(
        spec=spec,
        value=value,
        baseline=value,
        metric=fit.CandidateMetric(spec.name, 0.8, 0.3, 0.0, 0, spec.order),
    )
    return run.candidate_record(result)


def test_knn_winner_builds_small_native_artifact_and_routes(tmp_path: Path) -> None:
    data = _data()
    audit = module.audit_knn_winner(
        data,
        np.arange(24),
        _record(),
        baseline_arm=fit.ARMS[0],
        artifact_dir=tmp_path / "artifact",
        decisions=50,
    )
    assert audit.decisions == 50
    assert audit.artifact_bytes > 0
    assert audit.network_calls_per_route == 0
    assert not audit.foundation_model_weights_persisted
    assert (tmp_path / "artifact" / "policy.json").is_file()
    assert (tmp_path / "artifact" / "knn-bank.npz").is_file()


def test_lock_audit_rejects_latency_below_required_decision_count() -> None:
    with np.testing.assert_raises(ValueError):
        module.SeedWinnerAudit(
            seed=0,
            seed_report_sha256="a" * 64,
            candidate_name="candidate",
            config_sha256="b" * 64,
            artifact_kind="wmo-knn",
            artifact_sha256="c" * 64,
            sidecar_sha256="d" * 64,
            artifact_bytes=1_024,
            decisions=9_999,
            latency_p50_ms=1.0,
            latency_p95_ms=2.0,
            latency_passed=True,
        )
