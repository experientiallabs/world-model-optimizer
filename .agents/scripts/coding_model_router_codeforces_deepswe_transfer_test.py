"""Tests for the sealed Codeforces to DeepSWE transfer boundary."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _module() -> ModuleType:
    scripts = Path(__file__).parent
    sys.path.insert(0, str(scripts))
    path = scripts / "coding_model_router_codeforces_deepswe_transfer.py"
    spec = importlib.util.spec_from_file_location("codeforces_deepswe_transfer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _module()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_target_structural_preserves_frozen_width() -> None:
    values = module._target_structural("Input\n```python\narray graph\n```")
    assert len(values) == 21
    assert values[3:6] == [0.0, 0.0, 0.0]
    assert values[-4:] == [0.0, 0.0, 0.0, 0.0]


def test_freeze_uses_only_label_free_target_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = tmp_path / "development.json"
    outcomes = tmp_path / "outcomes.jsonl"
    corpus.write_text("source corpus\n")
    outcomes.write_text("source outcomes\n")
    rewards = np.asarray(
        [
            [0.4, 0.5, 0.6, 0.7, 0.8],
            [0.8, 0.7, 0.6, 0.5, 0.4],
            [0.5, 0.6, 0.7, 0.8, 0.9],
            [0.9, 0.8, 0.7, 0.6, 0.5],
        ]
    )
    data = module.Data(
        task_ids=["a", "b", "c", "d"],
        groups=["a", "b", "c", "d"],
        texts=["array", "graph", "string", "tree"],
        structural=np.zeros((4, 21)),
        rewards=rewards,
        costs=np.tile(np.arange(1.0, 6.0), (4, 1)),
    )
    monkeypatch.setattr(module, "load_data", lambda *args, **kwargs: data)
    confirmation = tmp_path / "confirmation.json"
    confirmation.write_text(
        json.dumps(
            {
                "frozen_candidate": module.FROZEN_CANDIDATE,
                "deep_swe_evaluation_authorized": True,
                "target_outcomes_used": False,
                "target_embeddings_used": False,
                "no_persisted_fitted_model": True,
                "confirmation_gate": {"passed": True},
                "inputs": {
                    "development_corpus_sha256": _sha256(corpus),
                    "development_outcomes_sha256": _sha256(outcomes),
                },
            }
        )
    )
    target = tmp_path / "target.json"
    target.write_text(
        json.dumps(
            {
                "protocol": "deepswe-label-free-task-feature-view-v2",
                "target_reward_fields_accessed": False,
                "target_cost_fields_accessed": False,
                "rows": [
                    {"id": f"task-{index}", "text": f"Fix array {index}", "repository": f"r{index}"}
                    for index in range(113)
                ],
            }
        )
    )
    output = tmp_path / "freeze"
    report = module.freeze_routes(
        development_corpus=corpus,
        development_outcomes=outcomes,
        confirmation_report=confirmation,
        target_feature_view=target,
        output_dir=output,
    )
    assert report["tasks"] == 113
    assert report["target_outcomes_used"] is False
    assert report["no_persisted_fitted_model"] is True
    assert len((output / "target-decisions.jsonl").read_text().splitlines()) == 113


def test_evaluate_opens_only_frozen_five_effort_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "BOOTSTRAPS", 100)
    decisions = tmp_path / "decisions.jsonl"
    rows = [
        {
            "task_id": f"task-{index}",
            "repository": f"repo-{index}",
            "source_arm": "luna-low" if index == 0 else "luna-high",
            "target_arm": module.TARGET_ARM["luna-low" if index == 0 else "luna-high"],
        }
        for index in range(113)
    ]
    decisions.write_text("".join(json.dumps(row) + "\n" for row in rows))
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "protocol": "codeforces-to-deepswe-direct-effort-route-freeze-v1",
                "decisions_sha256": _sha256(decisions),
                "target_outcomes_used": False,
                "target_reward_fields_accessed": False,
                "target_cost_fields_accessed": False,
            }
        )
    )
    matrix = tmp_path / "matrix.json"
    matrix_rows = [
        {
            "scenario_id": task_id,
            "model": module.TARGET_ARM[arm],
            "reward": float(1.0 if arm == selected else 0.5),
            "cost_usd": float(arm_index + 1),
        }
        for task_id, selected in (("task-0", "luna-low"), ("task-1", "luna-high"))
        for arm_index, arm in enumerate(module.ARMS)
    ]
    matrix.write_text(json.dumps({"outcomes": matrix_rows}))
    report = module.evaluate_frozen_routes(
        freeze_path=freeze,
        decisions_path=decisions,
        target_matrix=matrix,
        output_dir=tmp_path / "evaluation",
    )
    assert report["target_tasks"] == 2
    assert report["router"]["reward"] == 1.0
    assert report["target_outcomes_used_for_fit"] is False
    assert report["target_evaluation_count"] == 1
