"""Tests for the sealed SWE-rebench to DeepSWE transfer boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import coding_model_router_swerebench_deepswe_transfer as transfer
import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_freeze_fits_source_only_and_uses_label_free_target(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    corpus = tmp_path / "development.json"
    outcomes = tmp_path / "outcomes.jsonl"
    audit = tmp_path / "audit.json"
    corpus.write_text("external corpus\n", encoding="utf-8")
    outcomes.write_text("external outcomes\n", encoding="utf-8")
    audit.write_text("external audit\n", encoding="utf-8")
    candidate = transfer.Candidate(
        family="direct",
        order=0,
        dim=512,
        alpha=1.0,
        threshold=0.0,
    )
    config = candidate.config()
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    lock = tmp_path / "selection-lock.json"
    _write_json(
        lock,
        {
            "selected_key": candidate.key,
            "selected_config": config,
            "selected_config_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "development_corpus_sha256": _sha256(corpus),
            "development_outcomes_sha256": _sha256(outcomes),
            "collection_audit_sha256": _sha256(audit),
            "target_outcomes_used": False,
            "deep_swe_outcomes_accessed": False,
            "confirmation_outcomes_accessed": False,
        },
    )
    confirmation = tmp_path / "confirmation-report.json"
    _write_json(
        confirmation,
        {
            "protocol": transfer.CONFIRMATION_PROTOCOL,
            "confirmation_passed": True,
            "target_outcomes_used": False,
            "deep_swe_outcomes_accessed": False,
            "inputs": {"selection_lock_sha256": _sha256(lock)},
        },
    )
    feature_view = tmp_path / "target-view.json"
    _write_json(
        feature_view,
        {
            "protocol": "deepswe-label-free-task-feature-view-v2",
            "target_reward_fields_accessed": False,
            "target_cost_fields_accessed": False,
            "rows": [
                {
                    "id": f"task-{index}",
                    "text": f"Fix issue {index}",
                    "repository": f"repo-{index % 10}",
                }
                for index in range(113)
            ],
        },
    )
    source = SimpleNamespace(data=SimpleNamespace(rewards=np.zeros((2, 5))))
    monkeypatch.setattr(transfer, "candidate_grid", lambda: (candidate,))
    monkeypatch.setattr(transfer, "load_source", lambda *args: source)
    monkeypatch.setattr(
        transfer,
        "_fit_router",
        lambda *args, **kwargs: lambda task: 2,
    )
    monkeypatch.setattr(
        transfer,
        "_latency",
        lambda *args: {"passed": True, "p95_ms": 1.0},
    )

    report = transfer.freeze_routes(
        development_corpus=corpus,
        development_outcomes=outcomes,
        development_audit=audit,
        selection_lock=lock,
        confirmation_report=confirmation,
        target_feature_view=feature_view,
        output=tmp_path / "freeze",
    )

    assert report["tasks"] == 113
    assert report["arm_counts"]["luna-high"] == 113
    assert report["target_outcomes_used"] is False
    assert report["fitted_numeric_state_persisted"] is False


def test_evaluate_opens_hash_pinned_target_once(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    decisions = tmp_path / "target-decisions.jsonl"
    decision_rows = [
        {
            "task_id": f"task-{index}",
            "repository": f"repo-{index}",
            "source_arm": "luna-low" if index == 0 else "luna-high",
            "target_arm": transfer.TARGET_ARM[
                "luna-low" if index == 0 else "luna-high"
            ],
        }
        for index in range(113)
    ]
    decisions.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in decision_rows),
        encoding="utf-8",
    )
    freeze = tmp_path / "target-route-freeze.json"
    _write_json(
        freeze,
        {
            "protocol": "swerebench-to-deepswe-effort-route-freeze-v1",
            "decisions_sha256": _sha256(decisions),
            "target_outcomes_used": False,
            "deep_swe_outcomes_accessed": False,
            "fitted_numeric_state_persisted": False,
        },
    )
    matrix = tmp_path / "matrix.json"
    matrix_rows = [
        {
            "scenario_id": task_id,
            "model": transfer.TARGET_ARM[arm],
            "reward": float(1.0 if arm == selected else 0.0),
            "cost_usd": float(arm_index + 1),
        }
        for task_id, selected in (("task-0", "luna-low"), ("task-1", "luna-high"))
        for arm_index, arm in enumerate(transfer.ARMS)
    ]
    _write_json(matrix, {"outcomes": matrix_rows})
    monkeypatch.setattr(transfer, "TARGET_MATRIX_SHA256", _sha256(matrix))
    monkeypatch.setattr(
        transfer,
        "_cluster_interval",
        lambda groups, values: (float(np.mean(values)), 0.1, 0.2),
    )

    report = transfer.evaluate_routes(
        freeze=freeze,
        decisions=decisions,
        target_matrix=matrix,
        output=tmp_path / "evaluation",
    )

    assert report["target_tasks"] == 2
    assert report["router"]["reward"] == 1.0
    assert report["target_evaluation_count"] == 1
    assert report["target_outcomes_used_for_fit"] is False
