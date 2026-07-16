"""Smoke-test all 5 RL-method data paths + the HTTP training seam against the tau-bench WM.

Each downstream training chat (SFT + PPO + REINFORCE++, GRPO + SDPO, ICL) will consume the same
wmh-side machinery this script exercises end-to-end at tiny scale:

    1. ICL       -- closed-loop episode via `run_episode` + `WorldModelEnv`, plus
                    prior-episode critique injection into the next-episode agent prompt.
    2. SFT       -- chat-format dataset builder over recorded train traces
                    (one JSON example per step: {messages, completion}).
    3. PPO / R++ -- single-rollout batch: n=1 rollout per scenario, scored by
                    `WorldModel.score_session` (reward, step_rewards).
    4. GRPO      -- group-relative advantages: n=2 rollouts per scenario, grouped by
                    task_id; advantage = reward - group_mean.
    5. SDPO      -- one rollout scored, then the {task, rollout, critique} feedback
                    record (critique = EpisodeScore.critique).

Finally we hit the FastAPI serving layer in-process with `TestClient` and drive
session -> step -> score -> delete, since claas-verl and other training scaffolds will
talk to that HTTP surface, not the Python objects.

Budget: <~30 Bedrock haiku calls, max_steps=3, 2 scenarios. Under the benchmark dir so it's
excluded from the ruff/ty gate but we keep it clean. Run with:

    uv run python packages/environment-capture/tau-bench/rl/smoke.py

from the repo root. AWS creds come from `~/.aws` (default profile), region us-east-1.
"""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from fastapi.testclient import TestClient

from wmh.config import load_config
from wmh.core.parsing import extract_json_object
from wmh.core.render import render_action, render_demo
from wmh.core.types import Action, ActionKind, EnvState, JsonObject, Step, Trace
from wmh.engine import ingest, split_traces_3way
from wmh.engine.world_model import WorldModel
from wmh.env import DONE_SIGNAL, WorldModelEnv, run_episode
from wmh.env.scenarios import Scenario, scenarios_from_traces
from wmh.optimize.reward import EpisodeScore
from wmh.providers import get_provider
from wmh.providers.base import Message, Provider, ProviderConfig, ProviderKind
from wmh.serving.server import create_app

# --- config ---------------------------------------------------------------

# __file__ = .../packages/environment-capture/tau-bench/rl/smoke.py -> parents[4] = repo root.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_MODEL_DIR = _REPO_ROOT / "packages" / "environment-capture" / "tau-bench" / "models" / "tau-bench"
_TRACES_PATH = _REPO_ROOT / "packages" / "environment-capture" / "tau-bench" / "traces.otel.jsonl"

# Haiku is cheap and honours all Bedrock knobs; we override the model's built-in Opus so a
# ~30-call smoke doesn't cost Opus money. WorldModel.load reads the artifact index + prompt
# unchanged; only the *runtime* provider swaps.
# The Bedrock inference-profile id needs the dated version suffix (undated `us.anthropic.
# claude-haiku-4-5` returns ValidationException). Pricing strips the suffix, so cost still
# lines up with the `claude-haiku-4-5` price row.
_HAIKU_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
_REGION = "us-east-1"

MAX_STEPS = 3
NUM_SCENARIOS = 2
GRPO_ROLLOUTS_PER_SCENARIO = 2
WORLD_MODEL_NAME = "tau-bench"


# --- result plumbing ------------------------------------------------------


@dataclass
class MethodResult:
    name: str
    passed: bool
    detail: str = ""
    error: str = ""


@dataclass
class Report:
    results: list[MethodResult] = field(default_factory=list)

    def record(self, r: MethodResult) -> None:
        self.results.append(r)
        icon = "PASS" if r.passed else "FAIL"
        print(f"\n[{icon}] {r.name}: {r.detail or r.error}")

    def summary(self) -> str:
        width = max(len(r.name) for r in self.results)
        lines = ["", "=" * (width + 14), "SMOKE SUMMARY", "=" * (width + 14)]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"  {r.name.ljust(width)}   {status}   {r.detail or r.error}")
        lines.append("=" * (width + 14))
        return "\n".join(lines)


def _run(report: Report, name: str, fn: Callable[[], str]) -> None:
    """Run one path; on any exception, mark FAIL and keep going so we see every method's status."""
    print(f"\n--- {name} ---")
    try:
        detail = fn()
        report.record(MethodResult(name=name, passed=True, detail=detail))
    except Exception as exc:  # noqa: BLE001 - smoke reports rather than crashes on first failure
        tb = traceback.format_exc()
        report.record(MethodResult(name=name, passed=False, error=f"{type(exc).__name__}: {exc}"))
        print(tb)


# --- world model + agent construction ------------------------------------


def _load_wm_with_haiku() -> tuple[WorldModel, Provider]:
    """Load the built tau-bench WM but swap in Bedrock haiku for BOTH serve and reward providers."""
    config = load_config(str(_MODEL_DIR))  # read stored embed_dim, top_k, artifact layout
    _ = config  # config is consumed inside WorldModel.load; we override the provider only
    haiku_cfg = ProviderConfig(kind=ProviderKind.BEDROCK, model=_HAIKU_MODEL, region=_REGION)
    provider = get_provider(haiku_cfg)
    wm = WorldModel.load(str(_MODEL_DIR), provider, reward_provider=provider)
    return wm, provider


def _load_scenarios() -> tuple[list[Trace], list[Trace], list[Scenario]]:
    """Ingest the tau-bench corpus and split 80/10/10 (train / val / test).

    We only need a couple of scenarios, but running through the full split proves the whole
    train/test surface still works after the RL seam landed.
    """
    config = load_config(str(_MODEL_DIR))
    traces = ingest(config, file=str(_TRACES_PATH))
    train, _val, _test = split_traces_3way(traces, 0.8, 0.1)
    scenarios = scenarios_from_traces(train)[:NUM_SCENARIOS]
    if len(scenarios) < NUM_SCENARIOS:
        raise RuntimeError(
            f"need >= {NUM_SCENARIOS} scenarios from the train split; got {len(scenarios)}"
        )
    return train, _test, scenarios


# --- an agent that talks to the WM ---------------------------------------


class HaikuToolCallAgent:
    """Ask haiku for one tool call per step; fall back to a MESSAGE action on parse failure.

    Same shape as `wmh.engine.demo._propose_action`: JSON-only tool-call reply, temperature 0.
    """

    def __init__(self, provider: Provider, examples: list[Step]) -> None:
        self._provider = provider
        self._examples = examples

    def act(self, task: str | None, state: EnvState, history: list[Step]) -> Action:
        # Feed the model the task and (very brief) history — enough for a smoke tool call.
        history_text = (
            "\n".join(
                f"step {i + 1} ACTION: {render_action(s.action)}\n"
                f"step {i + 1} OBSERVATION: {s.observation.content[:200]}"
                for i, s in enumerate(history[-3:])
            )
            or "(no prior steps)"
        )
        example_block = "\n\n".join(render_demo(e) for e in self._examples[:2]) or "(no examples)"
        user = (
            "You role-play a tau-bench airline/retail/telecom customer-service agent. Emit one "
            "next tool call as a JSON object and nothing else:\n"
            '{"name": "<tool>", "arguments": {<json>}}\n\n'
            "If you believe the task is complete, instead return exactly the string <DONE>.\n\n"
            f"TASK:\n{task}\n\n"
            f"HISTORY:\n{history_text}\n\n"
            f"REFERENCE EXAMPLES:\n{example_block}\n\n"
            "Your single tool call (JSON only) or <DONE>:"
        )
        completion = self._provider.complete(
            "You role-play an agent. Reply with JSON only or <DONE>.",
            [Message(role="user", content=user)],
        )
        text = completion.text.strip()
        if text == DONE_SIGNAL:
            return Action(kind=ActionKind.MESSAGE, content=DONE_SIGNAL)
        raw = extract_json_object(text)
        if raw is not None:
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict) and "name" in obj:
                    args = obj.get("arguments") or {}
                    if isinstance(args, dict):
                        return Action(
                            kind=ActionKind.TOOL_CALL,
                            name=str(obj["name"]),
                            arguments=_as_json_object(args),
                        )
            except (json.JSONDecodeError, TypeError):
                pass
        return Action(kind=ActionKind.MESSAGE, content=text[:500])


def _as_json_object(d: dict) -> JsonObject:
    return {str(k): v for k, v in d.items()}


# --- helper: step the WM manually (used by PPO/GRPO/SDPO) ----------------


def _rollout_direct(
    wm: WorldModel, agent: HaikuToolCallAgent, task: str, max_steps: int
) -> tuple[str, list[Step], EpisodeScore]:
    """New WM session, agent loop, score, return (session_id, steps, score).

    `run_episode` uses its own session inside a WorldModelEnv (opened at reset, closed at return),
    so it isn't usable when we need `wm.score_session(session_id)` on the SAME session afterwards.
    This helper is the pattern PPO/GRPO/SDPO batches will use: new_session -> loop -> score ->
    end_session, keeping the reward tracker's SERVE/JUDGE split intact.
    """
    session = wm.new_session(task=task)
    for _ in range(max_steps):
        try:
            action = agent.act(task, session.state, session.history)
        except Exception as exc:  # noqa: BLE001 - one bad agent call kills only this rollout
            print(f"  agent.act raised: {type(exc).__name__}: {exc}")
            break
        if action.kind is ActionKind.MESSAGE and action.content == DONE_SIGNAL:
            break
        try:
            wm.step(session.id, action)
        except Exception as exc:  # noqa: BLE001 - env exceptions become episode-end, not test-end
            print(f"  wm.step raised: {type(exc).__name__}: {exc}")
            break
    score = wm.score_session(session.id)
    steps = list(session.history)
    wm.end_session(session.id)
    return session.id, steps, score


# --- method 1: ICL --------------------------------------------------------


def _method_icl(wm: WorldModel, provider: Provider, scenarios: list[Scenario]) -> str:
    """Closed-loop episode via `run_episode`; then show critique injection into a next-episode prompt."""
    scenario = scenarios[0]
    print(f"scenario.task[:120]: {scenario.task[:120]!r}")
    examples = wm.sample_steps(2)
    agent = HaikuToolCallAgent(provider, examples)
    env = WorldModelEnv(wm)
    result = run_episode(env, agent, scenario.task, max_steps=MAX_STEPS)
    print(f"  stop_reason={result.stop_reason.value} steps={len(result.steps)}")
    if not result.steps:
        raise RuntimeError("ICL: episode produced 0 steps; agent likely errored on step 1")
    # Now score the *just-completed* run by hand-building the same score call. run_episode
    # closed its env session, so we spin one up over the recorded steps to score - or simpler:
    # rebuild a session, replay steps into history (no LLM calls) and score. To keep this
    # smoke honest and cheap, we score a FRESH one-step rollout that reuses the agent instead.
    _sid, _steps, score = _rollout_direct(wm, agent, scenario.task, max_steps=1)
    print(f"  scored followup rollout: reward={score.reward:.2f} success={score.success}")
    print(f"  critique[:200]: {score.critique[:200]!r}")

    # ICL's whole point: fold the critique into the NEXT-episode prompt so the agent learns
    # in-context. Build the augmented prompt string and print — no second episode needed.
    augmented = (
        f"TASK:\n{scenario.task}\n\n"
        f"PREVIOUS ATTEMPT CRITIQUE (from the reward judge, use as guidance):\n{score.critique}\n\n"
        f"Take a fresh next tool call reflecting the critique."
    )
    print(f"  augmented next-episode prompt (first 300 chars):\n    {augmented[:300]!r}")
    return (
        f"episode steps={len(result.steps)}, follow-up reward={score.reward:.2f}, critique injected"
    )


# --- method 2: SFT --------------------------------------------------------


def _method_sft(train_traces: list[Trace]) -> str:
    """Build a chat-format SFT dataset from the first 2 train traces: one example per step."""
    if len(train_traces) < 2:
        raise RuntimeError("SFT: need >= 2 train traces")
    dataset: list[dict] = []
    for trace in train_traces[:2]:
        task = _first_task(trace)
        for i, step in enumerate(trace.steps):
            prior = trace.steps[:i]
            history_lines = []
            for j, ps in enumerate(prior[-3:], start=1):
                history_lines.append(f"turn {j} action: {render_action(ps.action)}")
                history_lines.append(f"turn {j} observation: {ps.observation.content[:200]}")
            user_content = (
                f"TASK: {task}\n\n"
                f"PRIOR TURNS:\n{chr(10).join(history_lines) or '(none)'}\n\n"
                "Emit the next agent tool call as JSON:"
            )
            completion = json.dumps(
                {"name": step.action.name, "arguments": step.action.arguments},
                sort_keys=True,
            )
            dataset.append(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a tau-bench customer-service agent.",
                        },
                        {"role": "user", "content": user_content},
                    ],
                    "completion": completion,
                }
            )
    print(f"  dataset size: {len(dataset)} examples")
    print("  example[0]:")
    print(json.dumps(dataset[0], indent=2)[:600])
    return f"{len(dataset)} SFT examples built from 2 train traces"


def _first_task(trace: Trace) -> str:
    for step in trace.steps:
        if step.task:
            return step.task
    return "(no task recorded)"


# --- method 3: PPO / REINFORCE++ -----------------------------------------


def _method_ppo(wm: WorldModel, provider: Provider, scenarios: list[Scenario]) -> str:
    """n=1 rollout per scenario, WM-scored. Reports the per-scenario batch record."""
    examples = wm.sample_steps(2)
    agent = HaikuToolCallAgent(provider, examples)
    batch = []
    for scenario in scenarios:
        _sid, steps, score = _rollout_direct(wm, agent, scenario.task, max_steps=MAX_STEPS)
        entry = {
            "task_id": scenario.provenance[0] if scenario.provenance else "?",
            "task": scenario.task[:80],
            "rollout_steps": len(steps),
            "reward": score.reward,
            "success": score.success,
            "step_rewards": score.step_rewards,
        }
        batch.append(entry)
        print(f"  rollout: {entry}")
    return f"{len(batch)} single-rollout PPO/R++ examples"


# --- method 4: GRPO -------------------------------------------------------


def _method_grpo(wm: WorldModel, provider: Provider, scenarios: list[Scenario]) -> str:
    """n=2 rollouts per scenario; group-relative advantages = reward - group_mean."""
    examples = wm.sample_steps(2)
    agent = HaikuToolCallAgent(provider, examples)
    groups: dict[str, list[dict]] = {}
    for scenario in scenarios:
        task_id = scenario.provenance[0] if scenario.provenance else "?"
        group = groups.setdefault(task_id, [])
        for k in range(GRPO_ROLLOUTS_PER_SCENARIO):
            _sid, steps, score = _rollout_direct(wm, agent, scenario.task, max_steps=MAX_STEPS)
            group.append(
                {
                    "rollout_index": k,
                    "reward": score.reward,
                    "step_rewards": score.step_rewards,
                    "steps": len(steps),
                }
            )
    for task_id, group in groups.items():
        mean = sum(r["reward"] for r in group) / max(len(group), 1)
        for r in group:
            r["advantage"] = r["reward"] - mean
        print(f"  group {task_id}: mean_reward={mean:.3f}")
        for r in group:
            print(
                f"    rollout {r['rollout_index']}: reward={r['reward']:.2f} adv={r['advantage']:+.3f}"
            )
    n_rollouts = sum(len(g) for g in groups.values())
    return f"{len(groups)} groups, {n_rollouts} rollouts, group-relative advantages computed"


# --- method 5: SDPO -------------------------------------------------------


def _method_sdpo(wm: WorldModel, provider: Provider, scenarios: list[Scenario]) -> str:
    """One scored rollout -> {task, rollout, critique} feedback record (SDPO's teacher signal)."""
    scenario = scenarios[0]
    examples = wm.sample_steps(2)
    agent = HaikuToolCallAgent(provider, examples)
    _sid, steps, score = _rollout_direct(wm, agent, scenario.task, max_steps=MAX_STEPS)
    record = {
        "task": scenario.task,
        "rollout": [
            {
                "action": {
                    "kind": s.action.kind.value,
                    "name": s.action.name,
                    "arguments": s.action.arguments,
                    "content": s.action.content,
                },
                "observation": {
                    "content": s.observation.content[:200],
                    "is_error": s.observation.is_error,
                },
            }
            for s in steps
        ],
        "critique": score.critique,
        "reward": score.reward,
    }
    print(f"  SDPO record (critique[:200]): {record['critique'][:200]!r}")
    print(f"  rollout_steps={len(record['rollout'])} reward={record['reward']:.2f}")
    return f"1 SDPO feedback record built (rollout_steps={len(steps)})"


# --- HTTP smoke via TestClient -------------------------------------------


def _method_http(wm: WorldModel) -> str:
    """Serve the wm via `create_app` and drive session -> step -> score -> delete over HTTP."""
    app = create_app(world_models={WORLD_MODEL_NAME: wm})
    client = TestClient(app)

    resp = client.post(
        f"/world_models/{WORLD_MODEL_NAME}/sessions",
        json={"task": "Look up reservation EHGLP3 for emma_kim_9957."},
    )
    assert resp.status_code == 200, resp.text
    session_id = resp.json()["session_id"]
    print(f"  session_id={session_id}")

    action = {
        "kind": "tool_call",
        "name": "get_reservation_details",
        "arguments": {"reservation_id": "EHGLP3"},
        "content": None,
    }
    resp = client.post(
        f"/world_models/{WORLD_MODEL_NAME}/sessions/{session_id}/step",
        json={"action": action},
    )
    assert resp.status_code == 200, resp.text
    obs = resp.json()["observation"]
    print(f"  step observation.content[:200]: {obs['content'][:200]!r}")

    resp = client.post(f"/world_models/{WORLD_MODEL_NAME}/sessions/{session_id}/score")
    assert resp.status_code == 200, resp.text
    score = resp.json()
    print(f"  score: reward={score['reward']:.2f} success={score['success']}")
    print(f"  critique[:200]: {score['critique'][:200]!r}")

    resp = client.delete(f"/world_models/{WORLD_MODEL_NAME}/sessions/{session_id}")
    assert resp.status_code == 200, resp.text
    usage = resp.json()
    print(f"  final usage: {usage}")

    return f"HTTP session {session_id[:8]} ok (reward={score['reward']:.2f})"


# --- main -----------------------------------------------------------------


def main() -> int:
    print("wmh RL smoke: loading tau-bench world model with Bedrock haiku (us-east-1)...")
    t0 = time.monotonic()
    wm, provider = _load_wm_with_haiku()
    train, _test, scenarios = _load_scenarios()
    print(
        f"  loaded wm + {len(train)} train traces + {len(scenarios)} scenarios ({time.monotonic() - t0:.1f}s)"
    )

    report = Report()
    _run(report, "1_ICL", lambda: _method_icl(wm, provider, scenarios))
    _run(report, "2_SFT", lambda: _method_sft(train))
    _run(report, "3_PPO_REINFORCE", lambda: _method_ppo(wm, provider, scenarios))
    _run(report, "4_GRPO", lambda: _method_grpo(wm, provider, scenarios))
    _run(report, "5_SDPO", lambda: _method_sdpo(wm, provider, scenarios))
    _run(report, "6_HTTP", lambda: _method_http(wm))

    print(report.summary())
    return 0 if all(r.passed for r in report.results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
