"""Tests for the external-only SWE-rebench effort fitter."""

from __future__ import annotations

import json
from pathlib import Path

import coding_model_router_swerebench_fit as fit
import numpy as np
import pytest
from coding_model_router_codeforces_fit import ARMS, Data


def _source() -> fit.SourceData:
    task_ids = [f"task-{index}" for index in range(10)]
    groups = [f"repo-{index // 2}" for index in range(10)]
    rewards = np.zeros((10, len(ARMS), fit.ATTEMPTS), dtype=np.float64)
    costs = np.empty_like(rewards)
    for arm in range(len(ARMS)):
        rewards[:, arm, :] = float(arm >= 2)
        costs[:, arm, :] = 0.01 * (arm + 1)
    data = Data(
        task_ids=task_ids,
        groups=groups,
        texts=[f"repository={groups[index]}\nfix bug {index}" for index in range(10)],
        structural=np.asarray(
            [[float(index), *([1.0] * 14)] for index in range(10)]
        ),
        rewards=rewards.mean(axis=2),
        costs=costs.mean(axis=2),
    )
    return fit.SourceData(
        data=data,
        raw_rewards=rewards,
        raw_costs=costs,
        languages=["Python"] * 10,
        repositories=groups,
    )


def test_candidate_grid_is_complete_and_has_no_similarity_floor() -> None:
    candidates = fit.candidate_grid()
    assert len(candidates) == 1_389
    assert len({candidate.key for candidate in candidates}) == len(candidates)
    knn = [candidate for candidate in candidates if candidate.family == "knn"]
    assert {candidate.rag_num for candidate in knn} == {8, 16, 32, 64}
    assert {candidate.z for candidate in knn} == {0.0, 0.5, 1.0, 1.645, 2.0}
    assert {candidate.pick_lam for candidate in knn} == {0.0, 0.01, 0.02, 0.03}
    assert {candidate.config()["rag_thres"] for candidate in knn} == {0.95}
    assert {candidate.config()["floor_q"] for candidate in knn} == {0.0}
    assert {candidate.config()["floor_sim"] for candidate in knn} == {None}


def test_grouped_folds_have_zero_repository_overlap() -> None:
    source = _source()
    for train, test in fit._folds(source.data):
        assert set(np.asarray(source.data.groups)[train]).isdisjoint(
            set(np.asarray(source.data.groups)[test])
        )


def test_load_source_drops_whole_infrastructure_missing_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_path = tmp_path / "corpus.json"
    outcomes_path = tmp_path / "outcomes.jsonl"
    audit_path = tmp_path / "audit.json"
    tasks = [
        {
            "task_id": f"task-{index}",
            "repository": f"repo-{index}",
            "language": "Python",
            "prompt": f"fix bug {index}",
        }
        for index in range(fit.TASKS)
    ]
    corpus_path.write_text(json.dumps({"tasks": tasks}), encoding="utf-8")
    monkeypatch.setattr(
        fit,
        "DEVELOPMENT_CORPUS_SHA256",
        fit._sha256(corpus_path),
    )
    rows = [
        {
            "task_id": task["task_id"],
            "arm": arm,
            "attempt_number": attempt,
            "model": "gpt-5.6-luna",
            "reward": float(arm_index >= 2),
            "cost_usd": 0.01 * (arm_index + 1),
            "target_outcomes_used": False,
        }
        for task in tasks[1:]
        for arm_index, arm in enumerate(ARMS)
        for attempt in range(fit.ATTEMPTS)
    ]
    outcomes_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    audit = {
        "valid": True,
        "source_tasks": fit.TASKS,
        "tasks": fit.TASKS - 1,
        "retained_task_coverage": (fit.TASKS - 1) / fit.TASKS,
        "excluded_tasks": [
            {
                "task_id": "task-0",
                "scope": "whole-task",
                "scientific_cells_rerun": 0,
            }
        ],
        "cells": len(rows),
        "target_outcomes_used": False,
        "deep_swe_outcomes_accessed": False,
        "outcomes_sha256": fit._sha256(outcomes_path),
    }
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    source = fit.load_source(corpus_path, outcomes_path, audit_path)
    assert len(source.data.task_ids) == fit.TASKS - 1
    assert "task-0" not in source.data.task_ids
    assert source.raw_rewards.shape == (fit.TASKS - 1, len(ARMS), fit.ATTEMPTS)


def test_direct_full_fit_router_returns_a_frozen_effort() -> None:
    source = _source()
    candidate = fit.Candidate(
        family="direct",
        order=0,
        dim=512,
        alpha=1.0,
        threshold=0.02,
    )
    route = fit._fit_text_router(
        source,
        candidate,
        label_rewards=source.data.rewards,
    )
    choice = route(
        {
            "repository": "heldout/repo",
            "language": "Python",
            "prompt": "Fix the failing parser test.",
        }
    )
    assert 0 <= choice < len(ARMS)


def test_within_repository_permutation_never_crosses_groups() -> None:
    source = _source()
    labels = source.data.rewards.copy()
    for index in range(len(labels)):
        labels[index] = index
    permutable = fit.SourceData(
        data=Data(
            task_ids=source.data.task_ids,
            groups=source.data.groups,
            texts=source.data.texts,
            structural=source.data.structural,
            rewards=labels,
            costs=source.data.costs,
        ),
        raw_rewards=source.raw_rewards,
        raw_costs=source.raw_costs,
        languages=source.languages,
        repositories=source.repositories,
    )
    shuffled = fit._permuted_labels(permutable)
    for index, group in enumerate(source.data.groups):
        allowed = {
            float(member)
            for member, candidate_group in enumerate(source.data.groups)
            if candidate_group == group
        }
        assert float(shuffled[index, 0]) in allowed
