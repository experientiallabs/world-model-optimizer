"""Unit tests for the external trace-burden-aware router."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _module() -> ModuleType:
    root = Path(__file__).parent
    sys.path.insert(0, str(root))
    path = root / "coding_model_router_open_swe_cost_aware.py"
    spec = importlib.util.spec_from_file_location("open_swe_cost_aware", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _module()


def test_penalty_reorders_equal_uplift_by_trace_burden() -> None:
    base = module.np.asarray([1.0, 1.0])
    auxiliary = module.np.asarray([1.0, -1.0])
    scores = module._scores(base, auxiliary, 0.1)
    assert scores[1] > scores[0]


def test_selection_retains_reward_then_minimizes_proxy_cost() -> None:
    rows = [
        module.Metrics(0.0, 0.0100, 0.002, 1.1, 0.1),
        module.Metrics(0.05, 0.0095, -0.001, 0.9, 0.09),
        module.Metrics(0.20, 0.0080, -0.004, 0.7, 0.08),
    ]
    selected = module._select(rows)
    assert selected.penalty == 0.05


def test_selection_falls_back_to_best_reward_when_none_positive() -> None:
    rows = [
        module.Metrics(0.0, -0.01, 0.0, 1.0, 0.0),
        module.Metrics(0.1, -0.02, -0.1, 0.8, 0.0),
    ]
    assert module._select(rows).penalty == 0.0
