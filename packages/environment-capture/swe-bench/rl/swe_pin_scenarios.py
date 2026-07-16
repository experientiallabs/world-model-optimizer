"""Pin the shared train/eval scenario sets for the swe-bench RL arms.

Ports the tau-bench pinning pattern (`packages/environment-capture/tau-bench/rl/pin_scenarios.py`, D26) to the
swe-bench corpus so every training arm trains on the SAME train scenarios and evaluates on the
SAME held-out scenarios. Both sets are derived deterministically from the committed corpus and
written to committed JSONL files; the training chats load the FILES, never re-derive (a corpus
append would silently shift a re-derived list).

- split: whole-trace `split_traces_3way(traces, 0.8, 0.1)` (stable blake2b hash of trace_id)
- eval scenarios: EVERY unique task in the test split (no sampling — the eval set must never
  look chosen)
- train scenarios: the train split's unique tasks, minus any task that also appears in the eval
  set (task-level leakage filter — dropped FROM TRAIN), then capped at TRAIN_CAP with a fixed
  seed, stratified by `repo` (parsed from the SWE-bench `instance_id`, e.g.
  `django__django-11099` -> `django__django`) so no repo dominates the cap
- identity: each line carries `provenance` (source trace_ids) AND the SWE-bench `instance_id`;
  consumers key on these, never on line number
- rubric: a compact JSON string of the instance's SWE-bench GOLD, joined on `instance_id` from
  the committed `swe_gold.json` cache (built by `fetch_gold.py` from SWE-bench Verified):
  `{"fail_to_pass": [...], "pass_to_pass_count": N, "repo": ..., "base_commit": ...}`. The
  `FAIL_TO_PASS` tests ARE the grading gold (the tests the correct patch flips to green); they
  ride verbatim. `PASS_TO_PASS` can be hundreds of tests, so only its count rides. The join is
  asserted 100% (a pinned instance missing from the cache aborts, never a silent null). The
  corpus's own `submission` (the agent's predicted diff) and `exit_status` (run outcome) are NOT
  gold and are never used as the rubric.
- tools.json: the tool inventory (name -> argument keys) derived from the TRAIN split only, pinned
  for the same reason the scenarios are. A single flat inventory (tools are not repo-specific).

Reads the corpus straight through the OTel-GenAI adapter (the corpus is always that shape) rather
than `load_config` on a built model dir — pinning derives from the committed corpus, so it must not
depend on a `wmh build` artifact existing.

Run from the repo root:  uv run python packages/environment-capture/swe-bench/rl/pin_scenarios.py
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
GOLD_PATH = _HERE / "swe_gold.json"  # SWE-bench Verified gold cache (see fetch_gold.py)


def _load_gold() -> dict[str, dict[str, object]]:
    """The committed SWE-bench Verified gold cache: instance_id -> gold fields."""
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))


def _rubric(instance_id: str, gold: dict[str, dict[str, object]]) -> str:
    """Compact JSON string of an instance's gold criteria (FAIL_TO_PASS verbatim + P2P count)."""
    g = gold[instance_id]
    return json.dumps(
        {
            "fail_to_pass": g["fail_to_pass"],
            "pass_to_pass_count": g["pass_to_pass_count"],
            "repo": g["repo"],
            "base_commit": g["base_commit"],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _instance_id(trace: Trace) -> str:
    value = trace.metadata.get("instance_id")
    return value if isinstance(value, str) and value else ""


def _repo(trace: Trace) -> str:
    """The SWE-bench repo, parsed from `instance_id` (`<repo>-<number>`)."""
    iid = _instance_id(trace)
    return iid.rsplit("-", 1)[0] if "-" in iid else (iid or "unknown")


def _tool_inventory(train: list[Trace]) -> dict[str, list[str]]:
    """tool name -> sorted argument keys, from the TRAIN split only (leak-free)."""
    tools: dict[str, set[str]] = defaultdict(set)
    for trace in train:
        for step in trace.steps:
            if step.action.kind is ActionKind.TOOL_CALL and step.action.name:
                tools[step.action.name].update(step.action.arguments)
    return {name: sorted(args) for name, args in sorted(tools.items())}


def _stratified_cap(scenarios: list[Scenario], by_trace: dict[str, Trace]) -> list[Scenario]:
    """Cap to TRAIN_CAP with per-repo proportional sampling (fixed seed, order-stable)."""
    if len(scenarios) <= TRAIN_CAP:
        return sorted(scenarios, key=lambda s: s.provenance[0])
    groups: dict[str, list[Scenario]] = defaultdict(list)
    for scenario in scenarios:
        groups[_repo(by_trace[scenario.provenance[0]])].append(scenario)
    rng = random.Random(SEED)
    picked: list[Scenario] = []
    remaining = TRAIN_CAP
    for i, (_name, group) in enumerate(sorted(groups.items())):
        # proportional share of the cap; the last repo absorbs the rounding remainder
        if i == len(groups) - 1:
            share = remaining
        else:
            share = round(TRAIN_CAP * len(group) / len(scenarios))
        share = min(share, len(group), remaining)
        picked.extend(rng.sample(group, share))
        remaining -= share
    # deterministic output order: by first provenance trace_id
    return sorted(picked, key=lambda s: s.provenance[0])


def _write(
    path: Path,
    scenarios: list[Scenario],
    by_trace: dict[str, Trace],
    gold: dict[str, dict[str, object]],
) -> None:
    lines = [
        json.dumps(
            {
                "task": s.task,
                "provenance": s.provenance,
                "instance_id": _instance_id(by_trace[s.provenance[0]]),
                "repo": _repo(by_trace[s.provenance[0]]),
                "rubric": _rubric(_instance_id(by_trace[s.provenance[0]]), gold),
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

    # Join every pinned scenario to gold on instance_id; abort on any miss (never a silent null).
    gold = _load_gold()
    pinned_ids = {
        _instance_id(by_trace[s.provenance[0]]) for s in (*train_scenarios, *eval_scenarios)
    }
    missing = sorted(pinned_ids - gold.keys())
    if missing:
        raise SystemExit(
            f"{len(missing)} pinned instance(s) missing from {GOLD_PATH.name}: {missing}; "
            f"re-run fetch_gold.py to refresh the SWE-bench Verified gold cache"
        )

    _write(TRAIN_OUT, train_scenarios, by_trace, gold)
    _write(EVAL_OUT, eval_scenarios, by_trace, gold)
    TOOLS_OUT.write_text(
        json.dumps(_tool_inventory(train), indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )

    def _counts(scenarios: list[Scenario]) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for s in scenarios:
            counts[_repo(by_trace[s.provenance[0]])] += 1
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
