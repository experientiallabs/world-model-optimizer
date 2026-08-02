"""Unit tests for the pooled cross-attempt uplift fitter."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


def _load_module() -> ModuleType:
    directory = Path(__file__).parent
    sys.path.insert(0, str(directory))
    path = directory / "coding_model_router_pooled_uplift_fit.py"
    spec = importlib.util.spec_from_file_location("pooled_uplift_fit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_module()


def _data() -> object:
    rewards = np.zeros((4, 5, 2), dtype=np.float64)
    costs = np.ones((4, 5, 2), dtype=np.float64)
    for attempt in range(2):
        rewards[0, module.HIGH_INDEX, attempt] = 1.0
        rewards[1, module.MAX_INDEX, attempt] = 1.0
        rewards[2, module.HIGH_INDEX, attempt] = 1.0
        rewards[3, module.MAX_INDEX, attempt] = 1.0
        costs[:, module.HIGH_INDEX, attempt] = 1.0
        costs[:, module.MAX_INDEX, attempt] = 2.0
    return module.PooledData(
        task_ids=["a1", "a2", "b1", "b2"],
        repositories=["a", "a", "b", "b"],
        languages=["python"] * 4,
        texts=["one", "two", "three", "four"],
        rewards=rewards,
        costs=costs,
    )


def test_candidate_grid_matches_frozen_family() -> None:
    candidates = module._candidate_grid()
    assert len(candidates) == 25
    assert candidates[0].key == "direct_ridge-hash512-a1"
    assert candidates[17].key == "two_head_ridge-hash8192-a100"
    assert candidates[-1].key == "extra-trees-256-leaf20"


def test_route_sends_larger_uplift_to_max() -> None:
    choices = module._route(np.asarray([-0.1, 0.2, 0.8]), threshold=0.2)
    assert choices.tolist() == [module.HIGH_INDEX, module.MAX_INDEX, module.MAX_INDEX]


def test_attempt_metrics_uses_traffic_matched_blind() -> None:
    data = _data()
    choices = np.asarray(
        [module.HIGH_INDEX, module.MAX_INDEX, module.HIGH_INDEX, module.MAX_INDEX]
    )
    metrics = module._attempt_metrics(data, choices, np.arange(4), attempt=0)
    assert metrics.reward == 1.0
    assert metrics.blind_reward == 0.5
    assert metrics.advantage == 0.5
    assert metrics.cost_usd == 1.5


def test_aggregate_metrics_preserves_attempt_specific_traffic() -> None:
    data = _data()
    choices = np.asarray(
        [
            [module.HIGH_INDEX, module.HIGH_INDEX],
            [module.MAX_INDEX, module.MAX_INDEX],
            [module.HIGH_INDEX, module.HIGH_INDEX],
            [module.MAX_INDEX, module.MAX_INDEX],
        ]
    )
    indices = np.arange(4)
    metrics = module._aggregate_metrics(data, choices, [(indices, 0), (indices, 1)])
    assert metrics.reward == 1.0
    assert metrics.blind_reward == 0.5
    assert len(metrics.task_advantages) == 8


def test_block_null_preserves_repository_pairs() -> None:
    data = _data()
    scores = np.asarray([1.0, 2.0, 3.0, 4.0])
    values = module._block_permutations(
        data,
        np.arange(4),
        scores,
        seed=module.SEED,
    )
    assert values.shape == (module.NULL_COUNT, 4)
    for row in values:
        assert tuple(row[:2]) in {(1.0, 2.0), (3.0, 4.0)}
        assert tuple(row[2:]) in {(1.0, 2.0), (3.0, 4.0)}
