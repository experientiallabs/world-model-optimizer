"""Export train-split terminal episodes as neutral JSONL for the SFT comparator (D86 orders).

The offline-SFT comparator imitates the recorded terminal agent (Opus captures). The claas-verl side
(`claas/benchmarks/wm_tau/sft.py`) turns these episodes into chat-format examples using
the SAME system prompt / message shapes / compression as the wm_tau rollout scaffold, so
the SFT row and the RL rows see byte-compatible prompts.

Leakage rule (same as terminal_pin_scenarios.py, D26): any train trace whose task text appears in
the pinned eval scenario set is dropped entirely — the policy must never train on an eval
prompt, whether as a scenario or as a recorded demonstration. Task identity comes from
`wmh.env.scenarios.trace_task`, the same helper the pin scripts use (via
`scenarios_from_traces`), so the two filters cannot drift apart.

Output (gitignored artifact root): .wmh/rl/sft_episodes.jsonl, one `SftEpisode` per line.

Run from the repo root:  uv run python packages/environment-capture/terminal-tasks/rl/terminal_export_sft.py [out.jsonl]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from wmh.config import load_config
from wmh.core.types import JsonObject, Trace
from wmh.engine import ingest, split_traces_3way
from wmh.env.scenarios import trace_task

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent / "models" / "terminal-tasks"
_TRACES_PATH = _HERE.parent / "traces.otel.jsonl"
_EVAL_SCENARIOS = _HERE / "scenarios_eval.jsonl"
# _HERE = .../packages/environment-capture/tau-bench/rl -> parents[3] = repo root.
_DEFAULT_OUT = _HERE.parents[3] / ".wmh" / "rl" / "terminal_sft_episodes.jsonl"


class SftStep(BaseModel):
    """One recorded tool call and the environment's observation."""

    name: str
    arguments: JsonObject = Field(default_factory=dict)
    observation: str
    is_error: bool = False


class SftEpisode(BaseModel):
    """One recorded terminal episode: the unit the SFT dataset builder consumes."""

    trace_id: str
    task: str
    domain: str = "unknown"
    steps: list[SftStep] = Field(default_factory=list)


def episodes_from_traces(traces: list[Trace], eval_tasks: set[str]) -> list[SftEpisode]:
    """Neutral episode records for SFT, dropping any trace whose task is in the eval set."""
    episodes: list[SftEpisode] = []
    for trace in traces:
        task = trace_task(trace)
        if task is None or task in eval_tasks:
            continue
        steps = [
            SftStep(
                name=step.action.name,
                arguments=step.action.arguments,
                observation=step.observation.content,
                is_error=step.observation.is_error,
            )
            for step in trace.steps
            if step.action.name is not None
        ]
        if not steps:
            continue
        domain = trace.metadata.get("domain")
        episodes.append(
            SftEpisode(
                trace_id=trace.trace_id,
                task=task,
                domain=domain if isinstance(domain, str) else "unknown",
                steps=steps,
            )
        )
    return episodes


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_OUT
    config = load_config(str(_MODEL_DIR))
    traces = ingest(config, file=str(_TRACES_PATH))
    train, _val, _test = split_traces_3way(traces, 0.8, 0.1)
    eval_tasks = {
        json.loads(line)["task"]
        for line in _EVAL_SCENARIOS.read_text().splitlines()
        if line.strip()
    }
    episodes = episodes_from_traces(train, eval_tasks)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for episode in episodes:
            f.write(episode.model_dump_json() + "\n")
    n_steps = sum(len(e.steps) for e in episodes)
    dropped = len(train) - len(episodes)
    print(
        f"wrote {out}: {len(episodes)} episodes / {n_steps} steps "
        f"(dropped {dropped} of {len(train)} train traces: eval-task overlap or empty)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
