"""Unit tests for the SWE-smith null-penalized fitter."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np


def _load_module() -> ModuleType:
    directory = Path(__file__).parent
    sys.path.insert(0, str(directory))
    path = directory / "coding_model_router_swesmith_null_fit.py"
    spec = importlib.util.spec_from_file_location("swesmith_null_fit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_module()


def test_scorer_grid_matches_frozen_family() -> None:
    candidates = module._scorer_grid()
    assert len(candidates) == 10
    assert candidates[0].key == "structural-only-a100"
    assert candidates[-1].key == "charhash8192-a100"


def test_structural_features_are_fixed_and_finite() -> None:
    values = module._structural(
        "Fix the Python dependency traceback in src/pkg/file.py and run tests"
    )
    assert len(values) == 15
    assert all(math.isfinite(value) for value in values)
    assert values[10] == 1.0
    assert values[8] > 0.0


def test_route_sends_lower_scores_to_max() -> None:
    choices = module._route(np.asarray([0.1, 0.5, 0.9]), threshold=0.5)
    assert choices.tolist() == [module.MAX_INDEX, module.MAX_INDEX, module.HIGH_INDEX]


def test_null_permutation_preserves_equal_length_repository_blocks() -> None:
    source = SimpleNamespace(
        repositories=["a", "a", "b", "b", "c"],
        languages=["python", "python", "python", "python", "go"],
    )
    scores = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    nulls = module._null_scores(source, scores)
    assert nulls.shape == (module.NULL_COUNT, 5)
    for row in nulls:
        assert tuple(row[:2]) in {(1.0, 2.0), (3.0, 4.0)}
        assert tuple(row[2:4]) in {(1.0, 2.0), (3.0, 4.0)}
        assert row[4] == 5.0


def test_metrics_compares_route_to_traffic_matched_blind() -> None:
    rewards = np.zeros((2, 5), dtype=np.float64)
    costs = np.ones((2, 5), dtype=np.float64)
    rewards[0, module.HIGH_INDEX] = 1.0
    rewards[1, module.MAX_INDEX] = 1.0
    costs[:, module.HIGH_INDEX] = 1.0
    costs[:, module.MAX_INDEX] = 2.0
    source = SimpleNamespace(data=SimpleNamespace(rewards=rewards, costs=costs))
    choices = np.asarray([module.HIGH_INDEX, module.MAX_INDEX])
    metrics = module._metrics(source, choices, np.asarray([0, 1]))
    assert metrics.reward == 1.0
    assert metrics.blind_reward == 0.5
    assert metrics.advantage == 0.5
    assert metrics.cost_usd == 1.5
