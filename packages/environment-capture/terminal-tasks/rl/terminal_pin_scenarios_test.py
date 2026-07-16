"""Guard the pinned terminal-tasks scenario sets against corpus drift and leakage.

Locks the invariants the RL arms depend on: train is capped, eval is EVERY unique test-split task,
the two are disjoint at BOTH the task-text and trace-id levels, every provenance id resolves to a
real trace, and every scenario is task-text tier (`rubric: null`). It also re-derives the sets from
the committed corpus and asserts the committed files still match.

`examples/` is off the root gate; run explicitly:
    uv run pytest packages/environment-capture/terminal-tasks/rl/pin_scenarios_test.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from wmh.env import scenarios_from_traces
from wmh.ingest import get_adapter

_DIR = Path(__file__).resolve().parent

# The trace corpora moved off-repo (huggingface: experiential-labs/wmh-*-traces).
# Corpus-reading tests skip when it isn't downloaded; the committed-file guards
# (well-formedness, train/eval disjointness) always run.
_needs_corpus = pytest.mark.skipif(
    not (_DIR.parent / "traces.otel.jsonl").exists(),
    reason="corpus not downloaded (see packages/environment-capture/README.md)",
)


def _load_pin() -> object:
    spec = importlib.util.spec_from_file_location(
        "terminal_pin", _DIR / "terminal_pin_scenarios.py"
    )
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

    for row in (*train, *ev):
        assert isinstance(row["task"], str) and row["task"].strip()
        assert isinstance(row["provenance"], list) and row["provenance"]
        assert isinstance(row["category"], str) and row["category"]
        assert row["rubric"] is None

    assert len(train) <= pin.TRAIN_CAP

    train_tasks = {r["task"] for r in train}
    eval_tasks = {r["task"] for r in ev}
    assert train_tasks.isdisjoint(eval_tasks)

    train_ids = {tid for r in train for tid in r["provenance"]}
    eval_ids = {tid for r in ev for tid in r["provenance"]}
    assert train_ids.isdisjoint(eval_ids)


@_needs_corpus
def test_pins_are_a_subset_of_the_corpus() -> None:
    pin = _load_pin()
    train_ids = {tid for r in _read(pin.TRAIN_OUT) for tid in r["provenance"]}
    eval_ids = {tid for r in _read(pin.EVAL_OUT) for tid in r["provenance"]}
    traces = get_adapter("otel-genai").from_file(str(pin._TRACES_PATH))
    corpus_ids = {t.trace_id for t in traces}
    assert (train_ids | eval_ids) <= corpus_ids


@_needs_corpus
def test_committed_files_match_a_fresh_derivation() -> None:
    pin = _load_pin()
    traces = get_adapter("otel-genai").from_file(str(pin._TRACES_PATH))
    train_split, _val, test_split = pin.split_traces_3way(traces, 0.8, 0.1)

    fresh_eval_tasks = {s.task for s in scenarios_from_traces(test_split)}
    assert {r["task"] for r in _read(pin.EVAL_OUT)} == fresh_eval_tasks

    pool_tasks = {s.task for s in scenarios_from_traces(train_split)} - fresh_eval_tasks
    committed_train_tasks = {r["task"] for r in _read(pin.TRAIN_OUT)}
    assert committed_train_tasks <= pool_tasks
    assert len(committed_train_tasks) == min(pin.TRAIN_CAP, len(pool_tasks))


@_needs_corpus
def test_tools_inventory_is_train_derived() -> None:
    pin = _load_pin()
    traces = get_adapter("otel-genai").from_file(str(pin._TRACES_PATH))
    train_split, _val, _test = pin.split_traces_3way(traces, 0.8, 0.1)
    committed = json.loads(pin.TOOLS_OUT.read_text(encoding="utf-8"))
    assert committed == pin._tool_inventory(train_split)
    assert committed, "tool inventory is non-empty"
