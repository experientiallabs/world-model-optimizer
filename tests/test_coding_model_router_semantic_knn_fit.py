"""Focused tests for the external semantic kNN experiment runner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).parents[1] / ".agents" / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "coding_model_router_semantic_knn_fit",
    SCRIPTS / "coding_model_router_semantic_knn_fit.py",
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_frozen_grid_and_pool_keep_model_effort_arms() -> None:
    candidates = module.candidate_grid()
    assert len(candidates) == 640
    assert len({candidate.key for candidate in candidates}) == 640
    assert {candidate.guard for candidate in candidates} == set(module.GUARDS)
    pool = module._pool()
    assert [entry.name for entry in pool] == list(module.base.ARMS)
    assert {entry.reasoning_effort for entry in pool} == set(module.base.EFFORTS)
    assert all(entry.input_per_mtok > 0 and entry.output_per_mtok > 0 for entry in pool)


def test_cached_embedder_rejects_missing_or_ambiguous_texts() -> None:
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    embedder = module.CachedEmbedder(["a", "b"], vectors)
    assert embedder.embed(["b", "a"]) == [[0.0, 1.0], [1.0, 0.0]]
    with pytest.raises(KeyError, match="absent"):
        embedder.embed(["c"])
    with pytest.raises(ValueError, match="unique"):
        module.CachedEmbedder(["a", "a"], vectors)


def test_metrics_use_sol_max_and_punish_static_dominance() -> None:
    task_count = 4
    arm_count = len(module.base.ARMS)
    rewards = np.zeros((task_count, arm_count), dtype=np.float64)
    costs = np.ones_like(rewards)
    sol_max = module.base.ARMS.index("sol-max")
    luna_low = module.base.ARMS.index("luna-low")
    rewards[:, sol_max] = 1.0
    costs[:, sol_max] = 2.0
    rewards[:, luna_low] = 0.5
    costs[:, luna_low] = 0.2
    data = module.base.Data(
        task_ids=[str(index) for index in range(task_count)],
        repositories=[f"repo-{index}" for index in range(task_count)],
        texts=[f"task-{index}" for index in range(task_count)],
        rewards=rewards,
        costs=costs,
    )
    metrics = module._metrics(data, np.asarray([sol_max] * task_count, dtype=np.int64))
    assert metrics["quality_retention"] == 1.0
    assert metrics["cost_savings"] == 0.0
    assert metrics["matched_blind_advantage"] == 0.0
    assert metrics["dominated_by_static"] == []


def test_embedding_rejects_wrong_frozen_content_length(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    tokenizer = tmp_path / "tokenizer.json"
    model.write_bytes(b"wrong")
    tokenizer.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="content length"):
        module._embed(["task"], model_path=model, tokenizer_path=tokenizer)
