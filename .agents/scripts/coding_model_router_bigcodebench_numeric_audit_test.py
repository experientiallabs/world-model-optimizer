"""Tests for fitted BigCodeBench numeric-router artifacts."""

from __future__ import annotations

from pathlib import Path

import coding_model_router_bigcodebench_fit as fit
import coding_model_router_bigcodebench_numeric_audit as module
import coding_model_router_bigcodebench_select as select
import joblib
import numpy as np
import pytest
from coding_model_router_bigcodebench_fit import FitData
from coding_model_router_bigcodebench_select import CandidateSpec

pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)


def _data() -> FitData:
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
        is_hard=np.asarray([index % 3 == 0 for index in range(tasks)], dtype=np.bool_),
        rewards=rewards,
        costs=costs,
    )


@pytest.mark.parametrize(
    "spec",
    [
        select.CandidateSpec("ordinal", "ridge", 512, 0, alpha=1.0),
        select.CandidateSpec(
            "ordinal",
            "extra-trees",
            512,
            1,
            n_estimators=200,
            min_samples_leaf=5,
            max_features="sqrt",
        ),
        select.CandidateSpec(
            "doubly-robust",
            "ridge",
            512,
            2,
            alpha=1.0,
            lam=0.01,
        ),
        select.CandidateSpec(
            "doubly-robust",
            "histogram",
            512,
            3,
            max_leaf_nodes=7,
            learning_rate=0.03,
            min_samples_leaf=10,
            lam=0.01,
        ),
        select.CandidateSpec(
            "empirical-bayes",
            "ridge",
            512,
            4,
            alpha=1.0,
            prior_strength=5.0,
            z=1.0,
        ),
    ],
)
def test_numeric_artifact_matches_frozen_candidate_and_round_trips(
    tmp_path: Path,
    spec: CandidateSpec,
) -> None:
    data = _data()
    train = np.arange(24, dtype=np.int64)
    test = np.arange(24, 30, dtype=np.int64)
    _, config_sha256 = fit.canonical_candidate_config(spec.config())
    payload = module.fit_numeric_payload(
        data,
        train,
        spec,
        config_sha256=config_sha256,
        seed=7,
    )
    actual = module.numeric_choices(
        payload,
        spec,
        [data.texts[int(index)] for index in test],
        [data.groups[int(index)] for index in test],
        [bool(data.is_hard[int(index)]) for index in test],
    )
    features = fit.feature_matrix(data, dim=spec.dim, scale_indices=train)
    expected = select._candidate_choices(
        spec,
        data,
        train,
        test,
        features[train],
        features[test],
        seed=7,
    )
    assert np.array_equal(actual, expected)
    artifact = tmp_path / f"{spec.order}.joblib"
    joblib.dump(payload, artifact, compress=3, protocol=5)
    restored = joblib.load(artifact)
    assert np.array_equal(
        actual,
        module.numeric_choices(
            restored,
            spec,
            [data.texts[int(index)] for index in test],
            [data.groups[int(index)] for index in test],
            [bool(data.is_hard[int(index)]) for index in test],
        ),
    )
