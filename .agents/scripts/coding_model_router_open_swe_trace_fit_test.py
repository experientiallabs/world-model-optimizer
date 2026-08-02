"""Tests for nested Open-SWE trajectory-distillation helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from scipy import sparse


def _module() -> ModuleType:
    path = Path(__file__).with_name("coding_model_router_open_swe_trace_fit.py")
    spec = importlib.util.spec_from_file_location("coding_model_router_open_swe_trace_fit", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_burden_signature_is_agent_neutral_and_finite() -> None:
    module = _module()
    partitions = {
        ("openhands", "weak"): [[1.0, 3.0], [3.0, 7.0]],
        ("openhands", "strong"): [[2.0, 4.0], [4.0, 8.0]],
        ("sweagent", "weak"): [[1.0, 2.0], [5.0, 6.0]],
    }
    signature = module._burden_signature(partitions)
    assert len(signature) == 6
    assert np.all(np.isfinite(signature))
    reversed_signature = module._burden_signature(dict(reversed(list(partitions.items()))))
    assert signature == pytest.approx(reversed_signature)


def test_select_point_finds_minimum_quality_constrained_traffic() -> None:
    module = _module()
    scores = np.asarray([4.0, 3.0, 2.0, 1.0])
    cheap = np.asarray([0.0, 0.0, 1.0, 1.0])
    strong = np.ones(4)
    point = module._select_point(scores, cheap, strong, 0.75)
    assert point.strong_count == 1
    assert point.traffic == pytest.approx(0.25)
    assert point.reward == pytest.approx(0.75)
    assert point.retention == pytest.approx(0.75)


def test_route_is_deterministic_under_ties() -> None:
    module = _module()
    scores = np.ones(4)
    cheap = np.zeros(4)
    strong = np.ones(4)
    reward, selected = module._route(scores, cheap, strong, 0.5)
    assert selected.tolist() == [True, True, False, False]
    assert reward.tolist() == [1.0, 1.0, 0.0, 0.0]


def test_burden_signature_rejects_one_partition() -> None:
    module = _module()
    with pytest.raises(ValueError, match="two agent partitions"):
        module._burden_signature({("openhands", "weak"): [[1.0]]})


def test_target_metadata_loader_accepts_only_label_free_rows(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "target.json"
    path.write_text(
        """{
          "target_cost_fields_accessed": false,
          "target_reward_fields_accessed": false,
          "rows": [{"id": "task-1", "repository": "owner/repo", "text": "Fix  spacing"}]
        }""",
        encoding="utf-8",
    )
    ids, texts, audit = module._load_target_metadata(path)
    assert ids == {"task-1"}
    assert texts == {"fix spacing"}
    assert audit["target_rows"] == 1


def test_target_metadata_loader_rejects_outcomes(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "target.json"
    path.write_text(
        """{
          "target_cost_fields_accessed": false,
          "target_reward_fields_accessed": false,
          "rows": [{
            "id": "task-1", "repository": "owner/repo", "text": "Fix it", "reward": 1
          }]
        }""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="label-free"):
        module._load_target_metadata(path)


def test_all_candidate_families_produce_grouped_finite_predictions() -> None:
    module = _module()
    rng = np.random.default_rng(7)
    tasks = 100
    data = module.Data(
        task_ids=[str(index) for index in range(tasks)],
        repos=np.asarray([f"repo-{index // 5}" for index in range(tasks)], dtype=object),
        texts=["task"] * tasks,
        cheap=rng.random(tasks),
        strong=rng.random(tasks),
        cheap_attempts=np.ones(tasks),
        strong_attempts=np.ones(tasks),
        burden=rng.random((tasks, 6)),
        attempt_rows={},
        overlap_audit={},
    )
    features = sparse.random(tasks, 64, density=0.2, random_state=7, format="csr")
    indices = np.arange(tasks)
    train, test = module._group_splits(indices, data.repos, 5)[0]
    for family in ("direct", "distilled", "concat"):
        candidate = module.Candidate(family, 2_048, 1.0, 1.0)
        out_of_fold = module._candidate_oof(data, features, indices, candidate)
        heldout = module._fit_candidate(data, features, train, test, candidate)
        assert out_of_fold.shape == (tasks,)
        assert heldout.shape == (len(test),)
        assert np.all(np.isfinite(out_of_fold))
        assert np.all(np.isfinite(heldout))
