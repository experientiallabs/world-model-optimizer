"""Tests for the graded SWE-rebench to DeepSWE sealed transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import coding_model_router_graded_swerebench_deepswe_transfer as transfer
import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_freeze_uses_confirmed_source_and_label_free_target(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    corpus = tmp_path / "development.json"
    outcomes = tmp_path / "development.jsonl"
    audit = tmp_path / "audit.json"
    for path in (corpus, outcomes, audit):
        path.write_text(path.name + "\n", encoding="utf-8")
    candidate = transfer.Candidate(order=0, guard="luna-high", k=8, z=1.0, pick_lam=0.02)
    fit = tmp_path / "fit.json"
    _json(
        fit,
        {
            "protocol": transfer.FIT_PROTOCOL,
            "valid": True,
            "development_passed": True,
            "deep_swe_outcomes_accessed": False,
            "target_outcomes_used": False,
            "fitted_numeric_state_persisted": False,
            "development": {
                "selected": {
                    "candidate": candidate.key,
                    "configuration": {
                        "order": candidate.order,
                        "guard": candidate.guard,
                        "k": candidate.k,
                        "z": candidate.z,
                        "pick_lam": candidate.pick_lam,
                    },
                }
            },
            "frontiers": {"fit_selected_static": "luna-max"},
        },
    )
    confirmation = tmp_path / "confirmation.json"
    _json(
        confirmation,
        {
            "protocol": transfer.CONFIRMATION_PROTOCOL,
            "valid": True,
            "confirmation_passed": True,
            "selected_candidate": candidate.key,
            "target_outcomes_used": False,
            "deep_swe_outcomes_accessed": False,
            "confirmation_outcomes_accessed_once": True,
            "input_sha256": {"fit_report": _sha256(fit)},
        },
    )
    feature_view = tmp_path / "target-view.json"
    _json(
        feature_view,
        {
            "protocol": "deepswe-label-free-task-feature-view-v2",
            "target_reward_fields_accessed": False,
            "target_cost_fields_accessed": False,
            "rows": [
                {
                    "id": f"task-{index}",
                    "repository": f"repo-{index % 10}",
                    "text": f"Fix bug {index}",
                }
                for index in range(113)
            ],
        },
    )
    monkeypatch.setattr(transfer, "TARGET_FEATURE_VIEW_SHA256", _sha256(feature_view))
    monkeypatch.setattr(transfer, "candidate_grid", lambda: (candidate,))
    data = SimpleNamespace(texts=["development one", "development two"])
    monkeypatch.setattr(transfer, "load_data", lambda *args: data)
    monkeypatch.setattr(
        transfer,
        "_embed",
        lambda texts, *args: (np.ones((len(texts), 4)), {"dimension": 4}),
    )
    monkeypatch.setattr(
        transfer,
        "_freeze_routes",
        lambda selected, source, tasks, *args: {
            "routes": [
                {
                    "task_id": task["task_id"],
                    "repository": task["repository"],
                    "arm": "sol-max" if index == 0 else "luna-high",
                }
                for index, task in enumerate(tasks)
            ],
            "route_decision_latency_ms": {"p95": 0.4},
        },
    )
    output = tmp_path / "freeze"
    report = transfer.freeze_routes(
        argparse.Namespace(
            output=output,
            fit_report=fit,
            confirmation_report=confirmation,
            development_corpus=corpus,
            development_outcomes=outcomes,
            development_audit=audit,
            target_feature_view=feature_view,
            embedding_model=tmp_path / "model.onnx",
            tokenizer=tmp_path / "tokenizer.json",
        )
    )

    assert report["tasks"] == 113
    assert report["arm_counts"]["sol-max"] == 1
    assert report["deep_swe_outcomes_accessed"] is False
    assert report["fitted_numeric_state_persisted"] is False
    assert "mini_swe_agent_gpt_5_6_sol_max" in (output / "target-decisions.jsonl").read_text()


def test_evaluate_opens_hash_pinned_six_arm_matrix_once(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    decisions = tmp_path / "target-decisions.jsonl"
    decision_rows = [
        {
            "task_id": f"task-{index}",
            "repository": f"repo-{index}",
            "source_arm": "sol-max" if index == 0 else "luna-high",
            "target_arm": transfer.TARGET_ARM["sol-max" if index == 0 else "luna-high"],
        }
        for index in range(113)
    ]
    decisions.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in decision_rows),
        encoding="utf-8",
    )
    freeze = tmp_path / "freeze.json"
    _json(
        freeze,
        {
            "protocol": transfer.FREEZE_PROTOCOL,
            "decisions_sha256": _sha256(decisions),
            "fit_selected_static": "luna-max",
            "target_outcomes_used": False,
            "deep_swe_outcomes_accessed": False,
            "fitted_numeric_state_persisted": False,
        },
    )
    matrix = tmp_path / "matrix.json"
    rows = [
        {
            "scenario_id": task_id,
            "model": transfer.TARGET_ARM[arm],
            "reward": float(1.0 if arm == selected else 0.5 if arm == "luna-max" else 0.0),
            "cost_usd": float(arm_index + 1),
        }
        for task_id, selected in (("task-0", "sol-max"), ("task-1", "luna-high"))
        for arm_index, arm in enumerate(transfer.ARMS)
    ]
    _json(matrix, {"outcomes": rows})
    monkeypatch.setattr(transfer, "TARGET_MATRIX_SHA256", _sha256(matrix))
    monkeypatch.setattr(
        transfer,
        "_cluster_interval",
        lambda groups, values: {
            "mean": float(np.mean(values)),
            "lower_95": 0.1,
            "median": 0.2,
            "upper_95": 0.3,
            "repositories": len(set(groups)),
            "seed": 1,
            "draws": 10,
        },
    )
    output = tmp_path / "evaluation"
    report = transfer.evaluate_routes(
        argparse.Namespace(
            output=output,
            freeze=freeze,
            decisions=decisions,
            target_matrix=matrix,
        )
    )

    assert report["target_tasks"] == 2
    assert report["router"]["reward"] == 1.0
    assert report["router"]["best_reward_hit_rate"] == 1.0
    assert report["router"]["cost_efficiency_gain_vs_fit_static"] > 1.0
    assert report["target_evaluation_count"] == 1
    assert report["target_outcomes_used_for_fit"] is False
    assert report["input_sha256"]["target_matrix"] == _sha256(matrix)
