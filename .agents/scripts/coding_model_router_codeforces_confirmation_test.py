"""Tests for the frozen Codeforces confirmation evaluator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


def _module() -> ModuleType:
    path = Path(__file__).with_name("coding_model_router_codeforces_confirmation.py")
    spec = importlib.util.spec_from_file_location("codeforces_confirmation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _module()


def test_score_features_uses_frozen_development_scale() -> None:
    rewards = np.ones((2, len(module.ARMS)))
    data = module.Data(
        task_ids=["1/C", "2/D"],
        groups=["1", "2"],
        texts=["solve a graph", "solve a string"],
        structural=np.asarray([[1.0, 2.0], [3.0, 6.0]]),
        rewards=rewards,
        costs=rewards,
    )
    scale = module._scale(data)
    features = module._score_features(data, scale)
    assert scale.tolist() == [1.0, 2.0]
    assert features.shape == (2, module.DIMENSION + 2)


def test_frozen_candidate_matches_protocol() -> None:
    assert module.FROZEN_CANDIDATE == "direct-hash512-a10-t0"
    assert module.DIMENSION == 512
    assert module.ALPHA == 10.0
    assert module.THRESHOLD == 0.0
