"""Guard the pinned gui-tasks scenario sets against corpus drift and leakage.

Locks the invariants the RL arms depend on: train is capped, eval is EVERY unique test-split task,
the two are disjoint at BOTH the task-text and trace-id levels, every provenance id resolves to a
real trace, and every scenario is task-text tier (`rubric: null`). It also re-derives the sets from
the committed corpus and asserts the committed files still match — a corpus append that shifts the
split fails here instead of silently in training.

`examples/` is off the root gate; run explicitly:
    uv run pytest examples/gui-tasks/rl/pin_scenarios_test.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from wmh.env import scenarios_from_traces
from wmh.ingest import get_adapter

_DIR = Path(__file__).resolve().parent


def _load_pin():
    spec = importlib.util.spec_from_file_location("gui_pin", _DIR / "pin_scenarios.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_pinned_sets_are_leak_free_and_well_formed() -> None:
    pin = _load_pin()
    train = _read(pin.TRAIN_OUT)
    ev = _read(pin.EVAL_OUT)

    # shape: every scenario carries a task, provenance, category, and null rubric (task-text tier).
    for row in (*train, *ev):
        assert isinstance(row["task"], str) and row["task"].strip()
        assert isinstance(row["provenance"], list) and row["provenance"]
        assert isinstance(row["category"], str) and row["category"]
        assert row["rubric"] is None

    assert len(train) <= pin.TRAIN_CAP

    # disjointness at the task-text level (a policy must never train on an eval task) ...
    train_tasks = {r["task"] for r in train}
    eval_tasks = {r["task"] for r in ev}
    assert train_tasks.isdisjoint(eval_tasks)

    # ... and at the trace-id level (no shared provenance across the two sets).
    train_ids = {tid for r in train for tid in r["provenance"]}
    eval_ids = {tid for r in ev for tid in r["provenance"]}
    assert train_ids.isdisjoint(eval_ids)

    # every provenance id resolves to a real trace in the committed corpus.
    traces = get_adapter("otel-genai").from_file(str(pin._TRACES_PATH))
    corpus_ids = {t.trace_id for t in traces}
    assert (train_ids | eval_ids) <= corpus_ids


def test_committed_files_match_a_fresh_derivation() -> None:
    """Re-deriving from the committed corpus reproduces eval exactly and keeps train within the cap.

    Eval = ALL unique test-split tasks, so it must reproduce byte-for-byte task-wise. Train is a
    seeded stratified sample of the leak-filtered train pool; we assert it stays a valid subset of
    that pool at the pinned size rather than re-running the sampler (the sampler IS the pin).
    """
    pin = _load_pin()
    traces = get_adapter("otel-genai").from_file(str(pin._TRACES_PATH))
    train_split, _val, test_split = pin.split_traces_3way(traces, 0.8, 0.1)

    fresh_eval_tasks = {s.task for s in scenarios_from_traces(test_split)}
    committed_eval_tasks = {r["task"] for r in _read(pin.EVAL_OUT)}
    assert committed_eval_tasks == fresh_eval_tasks

    pool_tasks = {s.task for s in scenarios_from_traces(train_split)} - fresh_eval_tasks
    committed_train_tasks = {r["task"] for r in _read(pin.TRAIN_OUT)}
    assert committed_train_tasks <= pool_tasks
    assert len(committed_train_tasks) == min(pin.TRAIN_CAP, len(pool_tasks))


def test_tools_inventory_is_train_derived() -> None:
    pin = _load_pin()
    traces = get_adapter("otel-genai").from_file(str(pin._TRACES_PATH))
    train_split, _val, _test = pin.split_traces_3way(traces, 0.8, 0.1)
    committed = json.loads(pin.TOOLS_OUT.read_text(encoding="utf-8"))
    assert committed == pin._tool_inventory(train_split)
    assert committed, "tool inventory is non-empty"
