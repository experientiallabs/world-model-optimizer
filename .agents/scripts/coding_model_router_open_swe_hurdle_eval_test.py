"""Unit tests for the frozen Open-SWE to DeepSWE confirmation evaluator."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


def _module() -> ModuleType:
    script_root = Path(__file__).parent
    sys.path.insert(0, str(script_root))
    path = script_root / "coding_model_router_open_swe_hurdle_eval.py"
    spec = importlib.util.spec_from_file_location("open_swe_hurdle_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _module()


def test_target_metadata_uses_public_title_and_description(tmp_path: Path) -> None:
    path = tmp_path / "tasks.json"
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "id": "task-1",
                        "problem_title": "Repair cache",
                        "display_description": "Fix invalidation behavior.",
                        "repository": "org/repo",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    ids, texts, groups = module._target_metadata(path)
    assert ids == ["task-1"]
    assert texts == ["Repair cache\nFix invalidation behavior."]
    assert groups == ["org/repo"]


def test_feature_view_ignores_reward_and_cost_fields(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.json"
    tasks.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "id": "task-1",
                        "problem_title": "Repair cache",
                        "display_description": "Fix invalidation behavior.",
                        "repository": "org/repo",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "outcomes": [
                    {
                        "scenario_id": "task-1",
                        "task": "Full runtime request",
                        "reward": {"must": "not be read"},
                        "cost_usd": {"must": "not be read"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    view = module._target_feature_view(matrix, tasks, tmp_path)
    payload = json.loads(view.read_text(encoding="utf-8"))
    assert payload["target_reward_fields_accessed"] is False
    assert payload["rows"][0]["text"] == "Full runtime request"


def test_random_control_uses_exact_router_traffic() -> None:
    result = module._random_control(
        np.zeros(5),
        np.asarray([1.0, 0.0, 0.0, 0.0, 0.0]),
        np.ones(5),
        np.full(5, 2.0),
        routed_reward=0.2,
        routed_cost=6.0,
        strong_count=1,
        seed=7,
    )
    assert result["strong_tasks"] == 1
    assert result["expected_reward"] == 0.04
    assert result["expected_cost_usd"] == 6.0


def test_repository_bootstrap_is_finite() -> None:
    result = module._bootstrap_vs_mixture(
        ["a", "a", "b", "c"],
        np.asarray([1.0, 0.0, 1.0, 0.0]),
        np.asarray([2.0, 1.0, 2.0, 1.0]),
        np.full(4, 0.25),
        np.full(4, 1.25),
        seed=7,
    )
    assert np.all(np.isfinite(result["reward_delta_95ci"]))
    assert np.all(np.isfinite(result["cost_delta_95ci_usd"]))
