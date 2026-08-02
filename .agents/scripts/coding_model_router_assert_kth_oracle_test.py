"""Tests for the ASSERT-KTH repeated-attempt oracle gate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _module() -> ModuleType:
    path = Path(__file__).with_name("coding_model_router_assert_kth_oracle.py")
    spec = importlib.util.spec_from_file_location("assert_kth_oracle", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _module()


def test_run_number_uses_last_run_marker() -> None:
    path = "arm/traj_model_run_7/model_eval_run_7.json"
    assert module._run_number(path) == 7


def test_result_classifier_accepts_evaluator_provenance_with_preds_name() -> None:
    assert module._is_result_item(
        {
            "type": "file",
            "path": "arm/run_0/model.run_0__preds.jsonl__hash.json",
            "size": 46_537,
        }
    )
    assert not module._is_result_item(
        {
            "type": "file",
            "path": "arm/run_0/trajectory_summary.json",
            "size": 3_088_697,
        }
    )


def test_outcomes_score_every_nonresolved_submission_zero() -> None:
    outcomes = module._outcomes(
        {
            "total_instances": 4,
            "submitted_ids": ["a", "b", "c"],
            "incomplete_ids": ["d"],
            "resolved_ids": ["b"],
            "error_ids": ["c"],
        }
    )
    assert outcomes == {"a": 0.0, "b": 1.0, "c": 0.0, "d": 0.0}


def test_overlap_uses_id_and_embedded_normalized_problem_text() -> None:
    task = module.Task("repo__repo-1", "repo/repo", "Fix   the Widget")
    assert module._overlaps(task, {"repo__repo-1"}, set())
    assert module._overlaps(task, set(), {"issue header fix the widget footer"})
    assert not module._overlaps(task, set(), {"fix a different widget"})


def test_target_feature_view_must_be_label_free(tmp_path: Path) -> None:
    path = tmp_path / "target-features.json"
    path.write_text(
        module.json.dumps(
            {
                "rows": [{"id": "target-1", "text": "Fix the widget"}],
                "target_reward_fields_accessed": False,
                "target_cost_fields_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    assert module._target_feature_rows(path) == [
        {"id": "target-1", "text": "Fix the widget"}
    ]
    payload = module.json.loads(path.read_text(encoding="utf-8"))
    payload["target_reward_fields_accessed"] = True
    path.write_text(module.json.dumps(payload), encoding="utf-8")
    try:
        module._target_feature_rows(path)
    except ValueError as error:
        assert "not label-free" in str(error)
    else:
        raise AssertionError("target feature view with rewards should fail")


def test_heldout_oracle_recovers_repeatable_complementarity() -> None:
    np = module.np
    rewards = np.zeros((8, 2, 10), dtype=np.float64)
    rewards[:4, 0, :] = 1.0
    rewards[4:, 1, :] = 1.0
    matrix = module.Matrix(
        tuple(f"task-{index}" for index in range(8)),
        np.asarray(["repo-a"] * 4 + ["repo-b"] * 4, dtype=object),
        ("arm-a", "arm-b"),
        rewards,
    )
    report = module._headroom(
        matrix,
        seed=7,
        attempt_splits=8,
        bootstraps_per_split=4,
    )
    assert report["mean_heldout_oracle_headroom"] == 0.5
    assert report["naive_same_attempt_headroom"] == 0.5


def test_choose_returns_one_arm_for_each_resampled_task() -> None:
    np = module.np
    fit = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    sampled = np.asarray([0, 0, 3], dtype=np.int64)
    chosen, _ = module._choose(fit, sampled)
    assert chosen.tolist() == [0, 0, 1]


def test_heldout_oracle_rejects_task_blind_arm_ordering() -> None:
    np = module.np
    rewards = np.zeros((6, 2, 10), dtype=np.float64)
    rewards[:, 0, :] = 1.0
    matrix = module.Matrix(
        tuple(f"task-{index}" for index in range(6)),
        np.asarray(["repo-a"] * 3 + ["repo-b"] * 3, dtype=object),
        ("best", "worse"),
        rewards,
    )
    report = module._headroom(
        matrix,
        seed=11,
        attempt_splits=6,
        bootstraps_per_split=3,
    )
    assert report["mean_heldout_oracle_headroom"] == 0.0
    assert report["naive_same_attempt_headroom"] == 0.0
