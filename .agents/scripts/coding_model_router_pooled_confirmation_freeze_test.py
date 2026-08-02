"""Unit tests for pooled-uplift confirmation route freezing."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


def _load_module() -> ModuleType:
    directory = Path(__file__).parent
    sys.path.insert(0, str(directory))
    path = directory / "coding_model_router_pooled_confirmation_freeze.py"
    spec = importlib.util.spec_from_file_location("pooled_confirmation_freeze", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_module()


def test_selected_candidate_matches_development_lock() -> None:
    candidate = module._selected_candidate()
    assert candidate.key == module.SELECTED_KEY
    assert candidate.family == "direct_ridge"
    assert candidate.dim == 8_192
    assert candidate.alpha == 10.0


def test_write_routes_separates_real_and_null_rows(tmp_path: Path) -> None:
    tasks = [
        {"task_id": "a", "repository": "org/a", "language": "python"},
        {"task_id": "b", "repository": "org/b", "language": "go"},
    ]
    path = tmp_path / "routes.jsonl"
    module._write_routes(
        path,
        tasks,
        np.asarray([module.HIGH_INDEX, module.MAX_INDEX]),
        null_index=7,
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["arm"] for row in rows] == ["luna-high", "luna-max"]
    assert [row["null_index"] for row in rows] == [7, 7]
    assert all(row["target_outcomes_used"] is False for row in rows)
