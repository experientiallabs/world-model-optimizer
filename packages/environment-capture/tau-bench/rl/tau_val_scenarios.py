"""Pin the tau VAL-split confirmatory scenario set (D90): the uncontaminated pass.

Every protocol choice in BENCH-B2 (pins, substrate, checkpoint rules, judge config) was
iterated against the same 20-task eval split, so the eval rows answer "did training work"
but not "did we overfit our protocol to our eval set". The val split (`split_traces_3way`'s
middle band) was never used for training, tuning, pins, or any decision; this script derives
its scenarios once, with the exact machinery of `tau_pin_scenarios.py` (same split call,
same provenance identity, same v2 substrate join), and the confirmatory pass runs ONE
pre-declared pass per arm on the result.

Contamination rules, stricter than the train pin (both directions):
- drop any val scenario whose task text appears in the pinned TRAIN scenarios (the
  policies trained on that task), keyed by the committed scenarios_train.jsonl, and
- drop any val scenario whose task text appears in the pinned EVAL scenarios (every
  protocol decision saw those tasks), keyed by the committed scenarios_eval.jsonl.
Dedup within val: identical task texts keep only the first by provenance order (the WM
eval is scenario-keyed; duplicate tasks would double-weight one task).

Output: scenarios_val_v2.jsonl next to the other pins ({task, provenance, domain,
seed_state, rubric}) — byte-stable on re-run over the same corpus.

Run from the repo root:
    uv run python packages/environment-capture/tau-bench/rl/tau_val_scenarios.py
"""

from __future__ import annotations

import json
from pathlib import Path

from tau_scenarios_v2 import upgrade

from wmh.config import load_config
from wmh.core.types import Trace
from wmh.engine import ingest, split_traces_3way
from wmh.env import scenarios_from_traces

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent / "models" / "tau-bench"
_TRACES_PATH = _HERE.parent / "traces.otel.jsonl"
VAL_OUT = _HERE / "scenarios_val_v2.jsonl"


def _domain(trace: Trace) -> str:
    value = trace.metadata.get("domain")
    return value if isinstance(value, str) and value else "unknown"


def _pinned_tasks(path: Path) -> set[str]:
    return {
        json.loads(line)["task"] for line in path.read_text().splitlines() if line.strip()
    }


def main() -> int:
    config = load_config(str(_MODEL_DIR))
    traces = ingest(config, file=str(_TRACES_PATH))
    by_trace = {t.trace_id: t for t in traces}
    _train, val, _test = split_traces_3way(traces, 0.8, 0.1)

    seen_tasks = _pinned_tasks(_HERE / "scenarios_train.jsonl") | _pinned_tasks(
        _HERE / "scenarios_eval.jsonl"
    )
    val_scenarios = sorted(scenarios_from_traces(val), key=lambda s: s.provenance[0])
    kept, dropped_seen, dropped_dup = [], 0, 0
    val_tasks: set[str] = set()
    for s in val_scenarios:
        if s.task in seen_tasks:
            dropped_seen += 1
            continue
        if s.task in val_tasks:
            dropped_dup += 1
            continue
        val_tasks.add(s.task)
        kept.append(s)

    rows = [
        {
            "task": s.task,
            "provenance": s.provenance,
            "domain": _domain(by_trace[s.provenance[0]]),
        }
        for s in kept
    ]
    upgraded = upgrade(rows, by_trace)
    with VAL_OUT.open("w") as f:
        for row in upgraded:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    counts: dict[str, int] = {}
    for row in upgraded:
        counts[row["domain"]] = counts.get(row["domain"], 0) + 1
    with_rubric = sum(1 for r in upgraded if r["rubric"])
    print(  # noqa: T201 - CLI output
        f"val split: {len(val)} traces -> {len(val_scenarios)} scenarios; "
        f"dropped {dropped_seen} train/eval-task overlaps + {dropped_dup} in-val dups -> "
        f"{len(upgraded)} pinned ({with_rubric} with gold rubrics) {dict(sorted(counts.items()))}"
    )
    print(f"wrote {VAL_OUT.name}")  # noqa: T201 - CLI output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
