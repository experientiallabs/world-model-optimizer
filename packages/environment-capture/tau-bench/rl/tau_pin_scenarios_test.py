"""Guard the committed pinned scenario files: shape, counts, and train/eval leakage."""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _load_jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (_HERE / name).read_text().splitlines() if line.strip()]


def test_scenario_files_parse_and_have_expected_counts() -> None:
    train = _load_jsonl("scenarios_train.jsonl")
    eval_ = _load_jsonl("scenarios_eval.jsonl")
    assert 90 <= len(train) <= 150, f"train scenario count drifted: {len(train)}"
    assert len(eval_) == 20, f"eval scenario count drifted: {len(eval_)}"
    for record in train + eval_:
        assert record["task"].strip(), "scenario with empty task"
        assert record["provenance"], "scenario without provenance trace_ids"
        assert record["domain"] in ("airline", "retail", "telecom"), record["domain"]


def test_no_train_eval_leakage() -> None:
    """The policy must never train on an eval prompt: no shared task text OR trace ids."""
    train = _load_jsonl("scenarios_train.jsonl")
    eval_ = _load_jsonl("scenarios_eval.jsonl")
    assert not {r["task"] for r in train} & {r["task"] for r in eval_}, (
        "task text shared between train and eval scenarios"
    )
    train_ids = {tid for r in train for tid in r["provenance"]}
    eval_ids = {tid for r in eval_ for tid in r["provenance"]}
    assert not train_ids & eval_ids, "provenance trace_ids shared between train and eval"


def test_tools_json_covers_all_domains() -> None:
    tools = json.loads((_HERE / "tools.json").read_text())
    assert set(tools) == {"airline", "retail", "telecom"}
    markers = {
        "airline": "book_reservation",
        "retail": "cancel_pending_order",
        "telecom": "check_network_status",
    }
    for domain, marker in markers.items():
        assert marker in tools[domain], f"missing {marker} in {domain}"
        for name, arg_keys in tools[domain].items():
            assert isinstance(arg_keys, list), (domain, name)
