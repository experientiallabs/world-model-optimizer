"""Focused tests for the public trace interaction representation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).parents[1] / ".agents" / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "coding_model_router_trace_interaction_fit",
    SCRIPTS / "coding_model_router_trace_interaction_fit.py",
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_extracts_only_canonical_pre_call_task() -> None:
    messages = json.dumps(
        [
            {
                "role": "user",
                "content": "wrapper\n<pr_description>repair this bug</pr_description>\nmore",
            },
            {"role": "assistant", "content": "private trajectory"},
        ]
    )
    assert module._initial_user_text(messages) == "repair this bug"


def _trace_data(count: int = 120) -> module.TraceData:
    models = module.SOURCE_MODELS
    x = np.linspace(0.05, 0.95, count)
    rewards = [
        {
            models[0]: float(value),
            models[1]: float(1.0 - value),
            models[2]: float(0.2 + 0.5 * value),
        }
        for value in x
    ]
    return module.TraceData(
        texts=[f"task-{index}" for index in range(count)],
        groups=[f"owner__repo-{index % 20}" for index in range(count)],
        model_rewards=rewards,
        keys=[(f"owner__repo-{index % 20}.task-{index}", str(index)) for index in range(count)],
        provenance={},
    )


def test_targets_keep_generic_and_all_dense_pairs() -> None:
    targets = module.build_targets(_trace_data())
    assert targets[0].name == "generic-resolution"
    assert not targets[0].pair_specific
    assert len(targets) == 4
    assert all(len(target.indices) == 120 for target in targets)
    assert sum(target.pair_specific for target in targets) == 3


def test_trace_only_grouped_selection_finds_linear_pair_signal() -> None:
    data = _trace_data()
    x = np.linspace(-1.0, 1.0, len(data.texts))
    vectors = np.column_stack([x, x**2, np.sin(x)]).astype(np.float32)
    alpha, grid = module.select_alpha(data, module.build_targets(data), vectors)
    assert alpha in module.RIDGE_ALPHAS
    selected = next(row for row in grid if row["alpha"] == alpha)
    assert selected["eligible"]
    assert selected["positive_pair_targets"] >= 2
    assert all(seed["mean_spearman"] > 0.0 for seed in selected["seeds"])


def test_interaction_vectors_are_normalized_and_weights_are_not_returned() -> None:
    data = _trace_data()
    x = np.linspace(-1.0, 1.0, len(data.texts))
    source = np.column_stack([x, x**2, np.sin(x)]).astype(np.float32)
    route = source[:7].copy()
    vectors, report = module.fit_interaction_vectors(
        data,
        module.build_targets(data),
        source,
        route,
        alpha=10.0,
    )
    assert vectors.shape == (7, 4)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)
    assert report["dimensions"] == 4
    assert "weights" not in report
