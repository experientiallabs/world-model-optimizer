"""Tests for the LLMRouterBench adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wmo.research.llmrouterbench import load_llmrouterbench

_MODELS = ["model-a", "model-b"]


def _write(
    root: Path, dataset: str, model: str, records: list[dict], *, hybrid: bool = False
) -> None:
    d = root / dataset / ("hybrid/" + model if hybrid else model)
    d.mkdir(parents=True, exist_ok=True)
    # Mirror the shipped summary-dict wrapper (a bare list also occurs; the loader takes both).
    payload = {"records": records, "model_name": model} if hybrid else records
    (d / f"{dataset}-{model}-20260101_000000.json").write_text(json.dumps(payload))


def _rec(index: int, score: float, cost: float) -> dict:
    return {
        "index": index,
        "origin_query": f"question {index}",
        "prompt": "p",
        "score": score,
        "cost": cost,
        "prompt_tokens": 10,
        "completion_tokens": 5,
    }


def test_load_joins_on_index_and_keeps_shared(tmp_path: Path) -> None:
    _write(tmp_path, "aime", "model-a", [_rec(1, 1.0, 0.02), _rec(2, 0.0, 0.02)])
    _write(tmp_path, "aime", "model-b", [_rec(1, 0.0, 0.001)], hybrid=True)  # index 2 missing
    matrix = load_llmrouterbench(tmp_path, models=_MODELS)
    assert matrix.scenario_ids() == ["aime:1"]  # index 2 dropped: not covered by model-b
    row = next(o for o in matrix.outcomes if o.model == "model-a")
    assert row.reward == 1.0
    assert row.cost_usd == 0.02
    assert row.task == "question 1"


def test_datasets_missing_a_model_are_skipped(tmp_path: Path) -> None:
    _write(tmp_path, "aime", "model-a", [_rec(1, 1.0, 0.02)])
    _write(tmp_path, "aime", "model-b", [_rec(1, 0.0, 0.001)])
    _write(tmp_path, "gpqa", "model-a", [_rec(1, 1.0, 0.02)])  # no model-b records
    matrix = load_llmrouterbench(tmp_path, models=_MODELS)
    assert matrix.scenario_ids() == ["aime:1"]


def test_empty_root_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no dataset"):
        load_llmrouterbench(tmp_path, models=_MODELS)


def test_duplicate_task_texts_are_dropped(tmp_path: Path) -> None:
    # Same origin_query under two datasets: only the first survives (leakage control).
    _write(tmp_path, "aime", "model-a", [_rec(1, 1.0, 0.02)])
    _write(tmp_path, "aime", "model-b", [_rec(1, 0.0, 0.001)])
    _write(tmp_path, "aime2", "model-a", [_rec(1, 1.0, 0.02)])
    _write(tmp_path, "aime2", "model-b", [_rec(1, 0.0, 0.001)])
    matrix = load_llmrouterbench(tmp_path, models=_MODELS)
    assert matrix.scenario_ids() == ["aime:1"]
