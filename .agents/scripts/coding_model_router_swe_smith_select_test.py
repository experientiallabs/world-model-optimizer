"""Tests for broad SWE-smith fit-only selection."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


def _module() -> ModuleType:
    scripts = Path(__file__).parent
    sys.path.insert(0, str(scripts))
    path = scripts / "coding_model_router_swe_smith_select.py"
    spec = importlib.util.spec_from_file_location("coding_model_router_swe_smith_select", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_knn_guard_uses_only_supported_similar_neighbors() -> None:
    module = _module()
    train_features = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )
    heldout_features = np.asarray([[1.0, 0.0]])
    uplift = np.asarray([1.0, 1.0, -1.0, -1.0])
    supported = module.KnnConfig(2, 2, 0.8, 0.0, 2)
    unsupported = module.KnnConfig(2, 2, 0.8, 0.0, 3)
    scores = module._knn_fold_scores(
        train_features,
        heldout_features,
        uplift,
        [supported, unsupported],
    )
    assert scores[supported.name].tolist() == [1.0]
    assert scores[unsupported.name].tolist() == [-1_000_000.0]


def test_prepare_splits_preserves_frozen_heldout_ids(tmp_path: Path) -> None:
    module = _module()
    rows = [
        {
            "instance_id": f"task-{index}",
            "repo": f"repo-{index // 2}",
            "text": f"prompt {index}",
            "cheap_reward": 0.0,
            "cheap_attempts": 2,
            "strong_reward": 1.0,
            "strong_attempts": 2,
        }
        for index in range(4)
    ]
    source = tmp_path / "source.json"
    source.write_text(json.dumps(rows) + "\n", encoding="utf-8")
    splits = [
        {
            "seed": seed,
            "heldout_ids": [f"task-{seed % 4}"],
            "heldout_ids_sha256": f"digest-{seed}",
        }
        for seed in range(5)
    ]
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "selected_cohort": {
                    "cohort_sha256": "cohort",
                    "tasks": [{"instance_id": row["instance_id"]} for row in rows],
                    "splits": splits,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    module.EXPECTED_SOURCE_SHA256 = _sha256(source)
    module.EXPECTED_FREEZE_SHA256 = _sha256(freeze)
    output = tmp_path / "splits"
    module._prepare_splits(argparse.Namespace(source=source, freeze=freeze, output=output))
    seed_zero_fit = json.loads((output / "seed-0-fit.json").read_text(encoding="utf-8"))
    seed_zero_heldout = json.loads((output / "seed-0-heldout.json").read_text(encoding="utf-8"))
    assert [row["instance_id"] for row in seed_zero_heldout] == ["task-0"]
    assert {row["instance_id"] for row in seed_zero_fit} == {"task-1", "task-2", "task-3"}
    manifest = json.loads((output / "split-manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["seeds"]) == 5
    assert manifest["target_outcomes_used"] is False


def test_shuffle_is_deterministic_and_stays_within_repository() -> None:
    module = _module()
    weak = np.asarray([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
    strong = weak + 100.0
    groups = ["a", "a", "a", "b", "b", "b"]
    first_weak, first_strong = module._shuffle_within_groups(
        weak,
        strong,
        groups,
        seed=10_000,
    )
    second_weak, second_strong = module._shuffle_within_groups(
        weak,
        strong,
        groups,
        seed=10_000,
    )
    assert first_weak.tolist() == second_weak.tolist()
    assert first_strong.tolist() == second_strong.tolist()
    assert set(first_weak[:3]) == {0.0, 1.0, 2.0}
    assert set(first_weak[3:]) == {10.0, 11.0, 12.0}
    assert (first_strong - first_weak).tolist() == [100.0] * 6


def test_frozen_controls_are_present_but_not_candidates() -> None:
    module = _module()
    candidates = module._candidate_specs()
    controls = module._control_specs()
    assert all(not module.autoresearch._is_control(spec) for spec in candidates)
    assert {spec.name for spec in controls} == {
        "task-blind-uplift",
        "hash2048-ridge-heads-shuffled-a1",
        "word128-ridge-uplift-shuffled-a10",
        "structural-irt-shuffled-a1",
    }


def test_fit_source_must_match_seed_manifest(tmp_path: Path) -> None:
    module = _module()
    source = tmp_path / "seed-2-fit.json"
    source.write_text("[]\n", encoding="utf-8")
    manifest = tmp_path / "split-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "swe-smith-broad-source-splits-v1",
                "freeze_sha256": module.EXPECTED_FREEZE_SHA256,
                "source_sha256": module.EXPECTED_SOURCE_SHA256,
                "target_outcomes_used": False,
                "seeds": [{"seed": 2, "fit_sha256": _sha256(source)}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert module._verify_fit_source(source, manifest, 2) == _sha256(source)
    source.write_text("[{}]\n", encoding="utf-8")
    try:
        module._verify_fit_source(source, manifest, 2)
    except ValueError as error:
        assert "differs from the split manifest" in str(error)
    else:
        raise AssertionError("changed fit source was accepted")
