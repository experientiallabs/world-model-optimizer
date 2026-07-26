"""Probe: does seeding the WM with the scenario's case facts kill environment luck?

B2 measured (D62): replaying an IDENTICAL action sequence into fresh train-WM sessions scores
with stdev ~0.34 at the 0.7 provider default and ~0.24 even at temperature 0 — the world model
imagines different case circumstances per session (tau scenarios carry no world state), so the
same actions succeed in one imagined world and fail in another. Temperature pins token
sampling; it cannot pin the imagined world.

This probe tests the deeper fix: `seed_state` (already in the API: `new_session(seed_state=)`,
`Env.reset(seed_state=)` — deliberately unused by scenarios v1). For one pinned train scenario,
we replay its source trace's recorded action sequence into N fresh sessions two ways:

- UNSEEDED: task only (today's behavior — the D62 lottery)
- SEEDED: task + `EnvState.scratchpad` holding the case facts extracted from the source
  trace's recorded observations (re-creating the fixed per-task DB that real tau2 has)

and compare reward spread. If seeded stdev drops well below the temperature-0 floor (~0.24),
scenario-pinned world state is the principled env-luck fix and becomes the core of
ScenarioSuite v2 (D19/D64): a scenario = task + seed_state, not task alone.

Run:  uv run python packages/environment-capture/tau-bench/rl/env_luck_probe.py [--sessions N] [--min-steps A --max-steps B]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from wmh.config import load_config
from wmh.core.types import Action, EnvState, Trace
from wmh.engine import ingest, split_traces_3way
from wmh.engine.world_model import WorldModel
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.providers.registry import get_provider
from wmh.providers.waterfall import WaterfallProvider

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent / "models" / "tau-bench"
_TRACES_PATH = _HERE.parent / "traces.otel.jsonl"
_TRAIN_SCENARIOS = _HERE / "scenarios_train.jsonl"

HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
OPUS = "us.anthropic.claude-opus-4-8"
JUDGE_POOLS = ((None, "us-east-1"), (None, "us-west-2"), (None, "us-east-2"))
FACT_CHARS = 500  # per recorded observation folded into the seed scratchpad
_WRITE_PREFIXES = (
    "book_", "cancel_", "exchange_", "return_", "update_", "modify_", "transfer_",
    "send_", "refuel", "enable_", "disable_", "set_", "toggle_", "pay_",
)


def _build_wm() -> WorldModel:
    env_provider = get_provider(
        ProviderConfig(kind=ProviderKind.BEDROCK, model=HAIKU, region="us-east-1")
    )
    judge = WaterfallProvider(
        [ProviderConfig(kind=ProviderKind.BEDROCK, model=OPUS, region=r) for _p, r in JUDGE_POOLS]
    )
    return WorldModel.load(str(_MODEL_DIR), env_provider, reward_provider=judge)


def _pick_probe_trace(min_steps: int, max_steps: int) -> tuple[str, Trace]:
    """A pinned train scenario whose source trace is a COMPLETE short episode.

    Replaying a complete episode gives the judge something that can plausibly score high —
    a prefix of a long episode scores ~0 in every imagined world (floor effect, no variance
    to measure)."""
    config = load_config(str(_MODEL_DIR))
    traces = ingest(config, file=str(_TRACES_PATH))
    train, _val, _test = split_traces_3way(traces, 0.8, 0.1)
    by_id = {t.trace_id: t for t in train}
    for line in _TRAIN_SCENARIOS.read_text(encoding="utf-8").splitlines():
        scenario = json.loads(line)
        trace = by_id.get(scenario["provenance"][0])
        if (
            trace is not None
            and min_steps <= len(trace.steps) <= max_steps
            and float(trace.metadata.get("reward") or 0.0) >= 1.0  # real tau2 marked it solved
            # The task must RESOLVE via tool calls: conversation-resolved tau episodes (e.g.
            # complaint negotiation) drop their resolution turns in conversion, so their
            # tool-call skeleton legitimately judges 0 in every world — a floor, not luck.
            and any(
                (st.action.name or "").startswith(_WRITE_PREFIXES) for st in trace.steps
            )
        ):
            return scenario["task"], trace
    raise SystemExit(f"no pinned train scenario with a {min_steps}-{max_steps} step source trace")


def _seed_state_from(trace: Trace, n_steps: int) -> EnvState:
    """Case facts: what the recorded environment actually returned for this task's world."""
    facts = []
    for step in trace.steps[:n_steps]:
        name = step.action.name or "message"
        facts.append(f"- {name} -> {step.observation.content[:FACT_CHARS]}")
    scratchpad = (
        "FIXED CASE FACTS for this task's world (the environment's database is exactly this; "
        "stay consistent with it):\n" + "\n".join(facts)
    )
    return EnvState(scratchpad=scratchpad)


def _replay(
    wm: WorldModel,
    task: str,
    actions: list[Action],
    seed_state: EnvState | None,
    rubric: str | None = None,
) -> tuple[float, str]:
    # Methodology guards (both bugs found in review — each contaminates the variance being
    # measured): (1) deep-copy the seed so the WM's in-place scratchpad appends don't leak one
    # session's imagined notes into the next session's starting state; (2) enrich=False so a
    # session's PREDICTED observations never enter the shared retrieval buffer as demos for
    # later sessions (order-dependent self-reinforcement would bias stdev low).
    seed = seed_state.model_copy(deep=True) if seed_state is not None else None
    session = wm.new_session(task=task, seed_state=seed, enrich=False)
    for action in actions:
        wm.step(session.id, action)
    score = wm.score_session(session.id, rubric=rubric)
    wm.end_session(session.id)
    return score.reward, score.critique


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--min-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=6)
    args = parser.parse_args()

    task, trace = _pick_probe_trace(args.min_steps, args.max_steps)
    actions = [s.action for s in trace.steps]  # the COMPLETE recorded episode
    seed = _seed_state_from(trace, len(trace.steps))
    wm = _build_wm()
    print(f"probe trace {trace.trace_id} | task[:100]: {task[:100]!r}")
    print(f"replaying {len(actions)} recorded actions x {args.sessions} sessions per condition\n")

    # the scenario's success rubric = real tau2's gold criteria, stored by the converter
    rubric_raw = trace.metadata.get("gold")  # the converter stores tau2 evaluation_criteria here
    rubric = json.dumps(rubric_raw, ensure_ascii=False) if rubric_raw else None
    print(f"rubric available: {bool(rubric)}\n")
    results: dict[str, list[float]] = {"unseeded": [], "seeded": [], "seeded+rubric": []}
    for label, state, rub in (
        ("unseeded", None, None),
        ("seeded", seed, None),
        ("seeded+rubric", seed, rubric),
    ):
        for i in range(args.sessions):
            reward, critique = _replay(wm, task, actions, state, rub)
            results[label].append(reward)
            # a zero from a broken judge reply must be visible, not silently averaged in
            print(f"  {label} session {i + 1}: reward={reward:.2f} | {critique[:110]!r}")

    print()
    for label, rewards in results.items():
        mean = statistics.fmean(rewards)
        std = statistics.pstdev(rewards)
        shown = ", ".join(f"{r:.2f}" for r in rewards)
        print(f"{label:9s} rewards=[{shown}] mean={mean:.3f} stdev={std:.3f}")
    print("\n(D62 baselines: stdev 0.34 @ temp 0.7, 0.24 @ temp 0; real tau2 scored this "
          "episode reward=1.0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
