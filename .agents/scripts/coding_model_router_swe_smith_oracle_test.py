"""Tests for the broad SWE-smith held-out-format oracle."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


def _module() -> ModuleType:
    if "duckdb" not in sys.modules and importlib.util.find_spec("duckdb") is None:
        sys.modules["duckdb"] = ModuleType("duckdb")
    path = Path(__file__).with_name("coding_model_router_swe_smith_oracle.py")
    spec = importlib.util.spec_from_file_location("coding_model_router_swe_smith_oracle", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_estimate_scores_task_specific_choices_on_other_format() -> None:
    module = _module()
    weak = np.asarray([[1.0, 1.0], [0.0, 0.0]])
    strong = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    strong_delta, headroom = module._estimate(np.asarray([0, 1]), weak, strong)
    assert strong_delta == 0.0
    assert headroom == 0.5


def test_oracle_keeps_repository_bootstrap_and_small_cohort_gate() -> None:
    module = _module()
    compact = [
        {"instance_id": f"task-{index}", "repo": f"repo-{index // 2}"}
        for index in range(20)
    ]
    cells = []
    for index in range(20):
        for prompt_format in module.FORMATS:
            cells.extend(
                [
                    {
                        "instance_id": f"task-{index}",
                        "repo": f"repo-{index // 2}",
                        "prompt_format": prompt_format,
                        "arm": "weak",
                        "reward": float(index % 2 == 0),
                    },
                    {
                        "instance_id": f"task-{index}",
                        "repo": f"repo-{index // 2}",
                        "prompt_format": prompt_format,
                        "arm": "strong",
                        "reward": float(index % 2 == 1),
                    },
                ]
            )
    report = module._oracle(compact, cells)
    assert report["tasks"] == 20
    assert report["repositories"] == 10
    assert report["mean_cross_format_oracle_headroom"] == 0.5
    assert report["gates"]["minimum_tasks"] is False
    assert report["gates"]["minimum_repositories"] is False
    assert report["passed"] is False
