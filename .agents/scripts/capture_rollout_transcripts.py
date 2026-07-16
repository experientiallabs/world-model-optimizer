"""Capture verbatim rollout transcripts for a curated subset of the K=8 scenario set.

Re-runs verification for hand-picked scenarios (covering agree/solvable quadrants) with a
recording provider wrapper, persisting the EXACT text exchanged: agent completions, world-model
env prompts + observations, and checklist-judge verdicts. Output feeds the proof-of-work artifact.

Usage (from the repo root):
    uv run python .agents/scripts/capture_rollout_transcripts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / ".agents" / "scripts"))

from run_scenario_e2e import NOVA_LITE, OUT_DIR, TRACES, WM_DIR, bedrock  # noqa: E402

from wmh.core.types import ActionKind  # noqa: E402
from wmh.engine.world_model import WorldModel  # noqa: E402
from wmh.env.base import WorldModelEnv  # noqa: E402
from wmh.env.episode import run_episode  # noqa: E402
from wmh.env.llm_agent import AGENT_SYSTEM, LLMAgent  # noqa: E402
from wmh.ingest import get_adapter  # noqa: E402
from wmh.scenarios import ChecklistJudge, ScenarioSet, trace_digest  # noqa: E402
from wmh.scenarios.verification import CHECKLIST_SYSTEM  # noqa: E402

# Curated to cover every verification quadrant + distinct clusters (from the e2e run verdicts).
CURATED = [
    ("scenario-1f0ca0031cd4", "back-agreement PASS / solvable PASS"),
    ("scenario-8fb0686ede3a", "back-agreement PASS / solvable PASS"),
    ("scenario-16e5f88274b8", "back-agreement FAIL / solvable PASS"),
    ("scenario-f3bad4bd68e6", "back-agreement PASS / solvable FAIL"),
    ("scenario-182d0811b7d0", "back-agreement FAIL / solvable FAIL"),
]
MAX_STEPS = 5


class RecordingProvider:
    """Wraps a provider, recording every (tag, system, user, reply) exchange verbatim."""

    def __init__(self, inner):  # noqa: ANN001
        self.config = inner.config
        self._inner = inner
        self.calls: list[dict[str, str]] = []

    def complete(self, system, messages, **kwargs):  # noqa: ANN001, ANN003, ANN201
        completion = self._inner.complete(system, messages, **kwargs)
        if system == AGENT_SYSTEM:
            tag = "agent"
        elif system == CHECKLIST_SYSTEM:
            tag = "judge"
        else:
            tag = "env"
        self.calls.append(
            {"tag": tag, "system": system, "user": messages[0].content, "reply": completion.text}
        )
        return completion

    def embed(self, texts):  # noqa: ANN001, ANN201
        return self._inner.embed(texts)

    def verify(self):  # noqa: ANN201
        return self._inner.verify()


def main() -> None:
    pool = ScenarioSet.load(OUT_DIR / "scenario_pool_tau_bench.json")
    by_id = {s.scenario_id: s for s in pool.scenarios}
    traces = {t.trace_id: t for t in get_adapter("otel-genai").from_file(str(TRACES))}

    provider = RecordingProvider(bedrock(NOVA_LITE))
    world_model = WorldModel.load(str(WM_DIR), provider, telemetry_root=str(REPO / ".wmh"))
    judge = ChecklistJudge(provider)
    agent = LLMAgent(provider)

    transcripts = []
    for scenario_id, quadrant in CURATED:
        scenario = by_id[scenario_id]
        source = traces[scenario.provenance[0]]
        print(f"== {scenario_id} ({quadrant}) ==", flush=True)

        # Back-agreement: judge the SOURCE trajectory against the generated checklist.
        provider.calls = []
        source_result = judge.score(scenario.task, scenario.checklist, source.steps)
        back_call = provider.calls[-1]
        recorded_reward = source.metadata.get("reward")

        # Solvability rollout, frozen (eval steps must not enrich the index).
        provider.calls = []
        with world_model.frozen():
            episode = run_episode(
                WorldModelEnv(world_model),
                agent,
                scenario.task,
                seed_state=scenario.seed_state,
                max_steps=MAX_STEPS,
            )
        rollout_calls = list(provider.calls)

        provider.calls = []
        rollout_result = judge.score(scenario.task, scenario.checklist, episode.steps)
        rollout_judge_call = provider.calls[-1]

        # Pair raw calls to parsed steps: each step is one agent call (+ one env call unless DONE).
        steps = []
        agent_calls = [c for c in rollout_calls if c["tag"] == "agent"]
        env_calls = [c for c in rollout_calls if c["tag"] == "env"]
        for i, step in enumerate(episode.steps):
            steps.append(
                {
                    "agent_reply_raw": agent_calls[i]["reply"] if i < len(agent_calls) else "",
                    "action": {
                        "kind": step.action.kind.value,
                        "name": step.action.name,
                        "arguments": step.action.arguments,
                        "content": step.action.content,
                    },
                    "env_prompt_user": env_calls[i]["user"] if i < len(env_calls) else "",
                    "env_reply_raw": env_calls[i]["reply"] if i < len(env_calls) else "",
                    "observation": step.observation.content,
                    "is_error": step.observation.is_error,
                }
            )
        done = len(agent_calls) > len(episode.steps)
        final_agent_reply = agent_calls[len(episode.steps)]["reply"] if done else None

        recorded_success = (
            None if not isinstance(recorded_reward, (int, float)) else float(recorded_reward) >= 1.0
        )
        transcripts.append(
            {
                "scenario_id": scenario_id,
                "quadrant": quadrant,
                "cluster_name": scenario.cluster_name,
                "weight": scenario.weight,
                "task": scenario.task,
                "seed_state": scenario.seed_state.scratchpad,
                "checklist": scenario.checklist,
                "provenance": scenario.provenance,
                "source_domain": source.metadata.get("domain"),
                "source_task_id": source.metadata.get("task_id"),
                "source_digest": trace_digest(source),
                "back_agreement": {
                    "judge_prompt": back_call["user"],
                    "judge_reply_raw": back_call["reply"],
                    "judge_success": source_result.success,
                    "recorded_reward": recorded_reward,
                    "recorded_success": recorded_success,
                    "agrees": None
                    if recorded_success is None
                    else source_result.success == recorded_success,
                },
                "rollout": {
                    "steps": steps,
                    "stop_reason": episode.stop_reason.value,
                    "agent_done_reply_raw": final_agent_reply,
                    "judge_prompt": rollout_judge_call["user"],
                    "judge_reply_raw": rollout_judge_call["reply"],
                    "passed": rollout_result.passed,
                    "success": rollout_result.success,
                    "critique": rollout_result.critique,
                },
            }
        )
        print(
            f"   back-agree={transcripts[-1]['back_agreement']['agrees']} "
            f"solvable={rollout_result.success} steps={len(episode.steps)}",
            flush=True,
        )

    out = OUT_DIR / "rollout_transcripts_tau_bench.json"
    out.write_text(
        json.dumps(
            {
                "prompts": {"agent_system": AGENT_SYSTEM, "judge_system": CHECKLIST_SYSTEM},
                "transcripts": transcripts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
