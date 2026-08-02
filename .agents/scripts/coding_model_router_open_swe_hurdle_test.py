"""Unit tests for the nested Open-SWE zero-inflated uplift fitter."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np


def _module() -> ModuleType:
    path = Path(__file__).with_name("coding_model_router_open_swe_hurdle.py")
    spec = importlib.util.spec_from_file_location("open_swe_hurdle", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _module()


def _data() -> object:
    size = 20
    texts = [f"Repair issue {index}" for index in range(size)]
    return module.Data(
        task_ids=[f"task-{index}" for index in range(size)],
        groups=np.asarray([f"repo-{index // 2}" for index in range(size)], dtype=object),
        texts=texts,
        structural=np.asarray([module._structural(text) for text in texts]),
        auxiliary=np.linspace(-1.0, 1.0, size),
        cheap=np.zeros(size),
        strong=np.asarray([1.0 if index % 3 == 0 else 0.0 for index in range(size)]),
        cheap_attempts=np.ones(size),
        strong_attempts=np.ones(size),
    )


def test_precision_is_harmonic_attempt_weight() -> None:
    data = cast(Any, _data())
    weights = data.precision
    assert np.allclose(weights, np.ones(len(weights)))


def test_grouped_oof_has_no_repository_overlap() -> None:
    data = cast(Any, _data())
    candidate = module.Candidate(
        "structural-direct",
        "structural",
        "direct-ridge",
    )
    features = module._features(data, candidate)
    scores, audits = module._oof(
        data,
        features,
        np.arange(len(data.task_ids)),
        candidate,
        seed=7,
    )
    assert len(scores) == len(data.task_ids)
    assert all(row["group_overlap"] == 0 for row in audits)


def test_hurdle_score_prefers_predicted_positive_discordance() -> None:
    data = cast(Any, _data())
    candidate = module.Candidate(
        "structural-hurdle",
        "structural",
        "hurdle",
    )
    features = module._features(data, candidate)
    train = np.arange(15)
    test = np.arange(15, 20)
    scores = module._fit_predict(
        data,
        features,
        train,
        test,
        candidate,
        seed=11,
    )
    assert scores.shape == (5,)
    assert np.all(np.isfinite(scores))


def test_traffic_matched_advantage_rewards_correct_top_pick() -> None:
    size = 5
    data = module.Data(
        task_ids=[str(index) for index in range(size)],
        groups=np.asarray([str(index) for index in range(size)], dtype=object),
        texts=["x"] * size,
        structural=np.zeros((size, 27)),
        auxiliary=np.zeros(size),
        cheap=np.zeros(size),
        strong=np.asarray([1.0, 0.0, 0.0, 0.0, 0.0]),
        cheap_attempts=np.ones(size),
        strong_attempts=np.ones(size),
    )
    advantage = module._advantage(
        data,
        np.arange(size),
        np.asarray([5.0, 4.0, 3.0, 2.0, 1.0]),
    )
    assert advantage > 0.0
