"""Offline tests for the DeepSWE model and reasoning-effort optimizer."""

from __future__ import annotations

import json
from pathlib import Path

from coding_model_router_deepswe import (
    _cell_lookup,
    _load_deepswe,
    _repository_folds,
    _repository_split,
    _static_frontier_arm,
    _verify_serving,
)

from wmo.optimize.policy import RoutingPolicy


def _source(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    data = root / "data" / "deepswe"
    tasks_root = data / "deep-swe-main" / "tasks"
    arms = [
        ("mini_swe_agent_gpt_5_6_luna_low", "gpt-5-6-luna", "low"),
        ("mini_swe_agent_gpt_5_6_luna_max", "gpt-5-6-luna", "max"),
        ("mini_swe_agent_gpt_5_6_terra_high", "gpt-5-6-terra", "high"),
    ]
    task_rows = [
        {"id": "a", "repository": "repo/a"},
        {"id": "b", "repository": "repo/a"},
        {"id": "c", "repository": "repo/c"},
        {"id": "d", "repository": "repo/d"},
        {"id": "drop", "repository": "repo/drop"},
    ]
    trial_rows = []
    for task in task_rows:
        task_id = task["id"]
        task_dir = tasks_root / task_id
        task_dir.mkdir(parents=True)
        (task_dir / "instruction.md").write_text(f"repair {task_id}", encoding="utf-8")
        for arm_index, (arm, model, effort) in enumerate(arms):
            if task_id == "drop" and arm_index == 0:
                continue
            trial_rows.append(
                {
                    "config": arm,
                    "task_name": task_id,
                    "included_in_score": True,
                    "model": model,
                    "reasoning_effort": effort,
                    "f2p_passed": (80, 86, 90)[arm_index],
                    "f2p_total": 100,
                    "cost_usd": 1.0 + arm_index,
                }
            )
    data.mkdir(parents=True, exist_ok=True)
    (data / "trials.json").write_text(
        json.dumps({"n_trials": len(trial_rows), "rows": trial_rows}),
        encoding="utf-8",
    )
    (data / "tasks.json").write_text(
        json.dumps({"n_tasks": len(task_rows), "rows": task_rows}),
        encoding="utf-8",
    )
    results = root / "results"
    results.mkdir()
    (results / "deepswe_embeddings.json").write_text(
        json.dumps({row["id"]: [1.0, float(index)] for index, row in enumerate(task_rows)}),
        encoding="utf-8",
    )
    return root


def test_loader_keeps_effort_arms_and_drops_incomplete_tasks_not_arms(tmp_path: Path) -> None:
    loaded, embedder = _load_deepswe(_source(tmp_path))

    assert loaded.matrix.scenario_ids() == ["a", "b", "c", "d"]
    assert loaded.dropped_tasks == ("drop",)
    assert len(loaded.matrix.pool) == 3
    assert {entry.reasoning_effort for entry in loaded.matrix.pool} == {"low", "high", "max"}
    assert all(entry.model.startswith("gpt-5.6-") for entry in loaded.matrix.pool)
    assert len(loaded.matrix.outcomes) == 12
    assert embedder.dim == 2


def test_repository_splits_never_leak_a_repository(tmp_path: Path) -> None:
    loaded, _embedder = _load_deepswe(_source(tmp_path))
    ids = loaded.matrix.scenario_ids()

    fit, heldout = _repository_split(ids, loaded.groups, seed=11)
    assert set(fit).isdisjoint(heldout)
    assert {loaded.groups[item] for item in fit}.isdisjoint(
        {loaded.groups[item] for item in heldout}
    )

    for inner_fit, inner_heldout in _repository_folds(fit, loaded.groups, seed=1011):
        assert {loaded.groups[item] for item in inner_fit}.isdisjoint(
            {loaded.groups[item] for item in inner_heldout}
        )


def test_static_frontier_selects_cheapest_arm_above_relative_quality_floor(
    tmp_path: Path,
) -> None:
    loaded, _embedder = _load_deepswe(_source(tmp_path))
    cells = _cell_lookup(loaded.matrix)

    selected = _static_frontier_arm(
        loaded.matrix,
        loaded.matrix.scenario_ids(),
        quality_floor=0.95,
        cells=cells,
    )

    assert selected == "mini_swe_agent_gpt_5_6_luna_max"


def test_static_policy_survives_wmo_http_serving_with_reasoning_effort(
    tmp_path: Path,
) -> None:
    loaded, _embedder = _load_deepswe(_source(tmp_path))
    selected = "mini_swe_agent_gpt_5_6_luna_max"
    policy_path = tmp_path / "policy.json"
    RoutingPolicy(
        kind="static",
        default_model=selected,
        pool=loaded.matrix.pool,
    ).save(policy_path)

    proof = _verify_serving(policy_path, tmp_path)

    assert proof["status"] == "passed"
    assert proof["routed_arm"] == selected
    assert proof["provider_model"] == "gpt-5.6-luna"
    assert proof["reasoning_effort"] == "max"
    assert proof["paid_calls"] == 0
