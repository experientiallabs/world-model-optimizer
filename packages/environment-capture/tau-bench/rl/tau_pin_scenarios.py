"""Pin the shared BENCH-B scenario sets for the tau-bench RL arms.

All six training arms (ICL, SFT, PPO, REINFORCE++, GRPO, SDPO — three chats) must train on the
SAME train scenarios and evaluate on the SAME held-out scenarios, or the comparison rows mean
nothing. This script derives both sets deterministically from the committed corpus and writes
them to JSONL files that are committed alongside it; the training chats load the FILES, never
re-derive (a corpus append would silently shift a re-derived list).

- split: whole-trace `split_traces_3way(traces, 0.8, 0.1)` (stable blake2b hash of trace_id)
- train scenarios: capped at TRAIN_CAP, sampled with a fixed seed, stratified by tau domain
  (airline/retail/telecom, read from trace metadata) so no domain dominates the cap
- eval scenarios: EVERY test-split scenario (no sampling — the eval set must never look chosen)
- identity: each line carries `provenance` (source trace_ids); consumers key on provenance,
  never on line number
- tools.json: the per-domain tool inventory (name -> argument keys) derived from the TRAIN
  split only, pinned for the same reason the scenarios are — every arm's agent must see the
  same tool list, and a corpus append must not silently change it

Run from the repo root:  uv run python packages/environment-capture/tau-bench/rl/pin_scenarios.py
Idempotent: re-running on the same corpus rewrites byte-identical files.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from wmh.config import load_config
from wmh.core.types import ActionKind, Trace
from wmh.engine import ingest, split_traces_3way
from wmh.env import Scenario, scenarios_from_traces

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent / "models" / "tau-bench"
_TRACES_PATH = _HERE.parent / "traces.otel.jsonl"
TRAIN_CAP = 150
SEED = 4405  # the repo's benchmark-convention seed (D12 lineage)

TRAIN_OUT = _HERE / "scenarios_train.jsonl"
EVAL_OUT = _HERE / "scenarios_eval.jsonl"
TOOLS_OUT = _HERE / "tools.json"


def _tool_inventory(train: list[Trace]) -> dict[str, dict[str, list[str]]]:
    """domain -> {tool name -> sorted argument keys}, from the TRAIN split only (leak-free)."""
    tools: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for trace in train:
        domain = _domain(trace)
        for step in trace.steps:
            if step.action.kind is ActionKind.TOOL_CALL and step.action.name:
                tools[domain][step.action.name].update(step.action.arguments)
    return {
        domain: {name: sorted(args) for name, args in sorted(by_name.items())}
        for domain, by_name in sorted(tools.items())
    }


def _domain(trace: Trace) -> str:
    value = trace.metadata.get("domain")
    return value if isinstance(value, str) and value else "unknown"


def _stratified_cap(scenarios: list[Scenario], by_trace: dict[str, Trace]) -> list[Scenario]:
    """Cap to TRAIN_CAP with per-domain proportional sampling (fixed seed, order-stable)."""
    if len(scenarios) <= TRAIN_CAP:
        return scenarios
    groups: dict[str, list[Scenario]] = defaultdict(list)
    for scenario in scenarios:
        groups[_domain(by_trace[scenario.provenance[0]])].append(scenario)
    rng = random.Random(SEED)
    picked: list[Scenario] = []
    remaining = TRAIN_CAP
    for i, (_name, group) in enumerate(sorted(groups.items())):
        # proportional share of the cap; the last domain absorbs rounding remainder
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
                "domain": _domain(by_trace[s.provenance[0]]),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        for s in scenarios
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    config = load_config(str(_MODEL_DIR))
    traces = ingest(config, file=str(_TRACES_PATH))
    by_trace = {t.trace_id: t for t in traces}
    train, _val, test = split_traces_3way(traces, 0.8, 0.1)

    eval_scenarios = sorted(scenarios_from_traces(test), key=lambda s: s.provenance[0])
    # The same tau task was captured in multiple traces, and identical task prompts can land in
    # different splits (the split hashes trace_ids, not tasks). The policy must never train on an
    # eval task, so any train scenario whose task also appears in the eval set is dropped —
    # from TRAIN, keeping the eval set at full size.
    eval_tasks = {s.task for s in eval_scenarios}
    train_pool = [s for s in scenarios_from_traces(train) if s.task not in eval_tasks]
    train_scenarios = _stratified_cap(train_pool, by_trace)

    _write(TRAIN_OUT, train_scenarios, by_trace)
    _write(EVAL_OUT, eval_scenarios, by_trace)
    TOOLS_OUT.write_text(
        json.dumps(_tool_inventory(train), indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )

    def _counts(scenarios: list[Scenario]) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for s in scenarios:
            counts[_domain(by_trace[s.provenance[0]])] += 1
        return dict(sorted(counts.items()))

    split_note = f"train {len(train)} / val {len(_val)} / test {len(test)}"
    print(f"corpus: {len(traces)} traces -> {split_note}")
    print(f"train scenarios: {len(train_scenarios)} (cap {TRAIN_CAP}) {_counts(train_scenarios)}")
    print(f"eval scenarios:  {len(eval_scenarios)} (ALL of test) {_counts(eval_scenarios)}")
    print(f"wrote {TRAIN_OUT.name}, {EVAL_OUT.name}, {TOOLS_OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
