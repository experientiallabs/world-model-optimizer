"""Pin the shared train/eval scenario sets for the terminal-tasks RL arms.

Ports the tau-bench pinning pattern (`packages/environment-capture/tau-bench/rl/pin_scenarios.py`, D26) to the
terminal-tasks corpus so every training arm trains on the SAME train scenarios and evaluates on
the SAME held-out scenarios. Both sets are derived deterministically from the committed corpus and
written to committed JSONL files; the training chats load the FILES, never re-derive (a corpus
append would silently shift a re-derived list).

- split: whole-trace `split_traces_3way(traces, 0.8, 0.1)` (stable blake2b hash of trace_id)
- eval scenarios: EVERY unique task in the test split (no sampling — the eval set must never
  look chosen)
- train scenarios: the train split's unique tasks, minus any task that also appears in the eval
  set (task-level leakage filter — dropped FROM TRAIN), then capped at TRAIN_CAP with a fixed
  seed, stratified by `task_category` so no category dominates the cap
- identity: each line carries `provenance` (source trace_ids); consumers key on provenance,
  never on line number
- rubric: `null` for every scenario. The terminal corpus has NO gold criteria (it records the
  real command outputs, not a task-level pass/fail rubric), so per D67 these scenarios are
  task-text tier; we do not invent rubrics.
- tools.json: the tool inventory (name -> argument keys) derived from the TRAIN split only, pinned
  for the same reason the scenarios are. Terminal tools are not category-specific, so this is a
  single flat inventory (unlike tau's per-domain split).

Reads the corpus straight through the OTel-GenAI adapter (the corpus is always that shape) rather
than `load_config` on a built model dir — pinning derives from the committed corpus, so it must not
depend on a `wmh build` artifact existing.

Run from the repo root:  uv run python packages/environment-capture/terminal-tasks/rl/pin_scenarios.py
Idempotent: re-running on the same corpus rewrites byte-identical files.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from wmh.core.types import ActionKind, Trace
from wmh.engine import split_traces_3way
from wmh.env import Scenario, scenarios_from_traces
from wmh.ingest import get_adapter

_HERE = Path(__file__).resolve().parent
_TRACES_PATH = _HERE.parent / "traces.otel.jsonl"
TRAIN_CAP = 150
SEED = 4405  # the repo's benchmark-convention seed (D12 lineage)

TRAIN_OUT = _HERE / "scenarios_train.jsonl"
EVAL_OUT = _HERE / "scenarios_eval.jsonl"
TOOLS_OUT = _HERE / "tools.json"


def _category(trace: Trace) -> str:
    value = trace.metadata.get("task_category")
    return value if isinstance(value, str) and value else "unknown"


def _tool_inventory(train: list[Trace]) -> dict[str, list[str]]:
    """tool name -> sorted argument keys, from the TRAIN split only (leak-free)."""
    tools: dict[str, set[str]] = defaultdict(set)
    for trace in train:
        for step in trace.steps:
            if step.action.kind is ActionKind.TOOL_CALL and step.action.name:
                tools[step.action.name].update(step.action.arguments)
    return {name: sorted(args) for name, args in sorted(tools.items())}


def _stratified_cap(scenarios: list[Scenario], by_trace: dict[str, Trace]) -> list[Scenario]:
    """Cap to TRAIN_CAP with per-category proportional sampling (fixed seed, order-stable)."""
    if len(scenarios) <= TRAIN_CAP:
        return sorted(scenarios, key=lambda s: s.provenance[0])
    groups: dict[str, list[Scenario]] = defaultdict(list)
    for scenario in scenarios:
        groups[_category(by_trace[scenario.provenance[0]])].append(scenario)
    rng = random.Random(SEED)
    picked: list[Scenario] = []
    remaining = TRAIN_CAP
    for i, (_name, group) in enumerate(sorted(groups.items())):
        # proportional share of the cap; the last category absorbs the rounding remainder
        if i == len(groups) - 1:
            share = remaining
        else:
            share = round(TRAIN_CAP * len(group) / len(scenarios))
        share = min(share, len(group), remaining)
        picked.extend(rng.sample(group, share))
        remaining -= share
    # deterministic output order: by first provenance trace_id
    return sorted(picked, key=lambda s: s.provenance[0])


def _write(path: Path, scenarios: list[Scenario], by_trace: dict[str, Trace]) -> None:
    lines = [
        json.dumps(
            {
                "task": s.task,
                "provenance": s.provenance,
                "category": _category(by_trace[s.provenance[0]]),
                "rubric": None,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        for s in scenarios
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    traces = get_adapter("otel-genai").from_file(str(_TRACES_PATH))
    by_trace = {t.trace_id: t for t in traces}
    train, val, test = split_traces_3way(traces, 0.8, 0.1)

    eval_scenarios = sorted(scenarios_from_traces(test), key=lambda s: s.provenance[0])
    # Identical task prompts can land in different splits (the split hashes trace_ids, not tasks).
    # The policy must never train on an eval task, so any train scenario whose task also appears in
    # the eval set is dropped — from TRAIN, keeping the eval set at full size.
    eval_tasks = {s.task for s in eval_scenarios}
    train_all = scenarios_from_traces(train)
    train_pool = [s for s in train_all if s.task not in eval_tasks]
    n_leak = len(train_all) - len(train_pool)
    train_scenarios = _stratified_cap(train_pool, by_trace)

    _write(TRAIN_OUT, train_scenarios, by_trace)
    _write(EVAL_OUT, eval_scenarios, by_trace)
    TOOLS_OUT.write_text(
        json.dumps(_tool_inventory(train), indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )

    def _counts(scenarios: list[Scenario]) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for s in scenarios:
            counts[_category(by_trace[s.provenance[0]])] += 1
        return dict(sorted(counts.items()))

    print(f"corpus: {len(traces)} traces -> train {len(train)} / val {len(val)} / test {len(test)}")
    print(f"train scenarios: {len(train_scenarios)} (cap {TRAIN_CAP}, {n_leak} leak-dropped)")
    print(f"  {_counts(train_scenarios)}")
    print(f"eval scenarios:  {len(eval_scenarios)} (ALL of test)")
    print(f"  {_counts(eval_scenarios)}")
    print(f"wrote {TRAIN_OUT.name}, {EVAL_OUT.name}, {TOOLS_OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
