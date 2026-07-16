"""Episode reward judging: the RL-side reward signal served by the harness.

An `EpisodeRewardJudge` scores one finished rollout (task + the steps the agent took against the
world model) in a single LLM call, returning every signal the RL algorithms need at once:

- `reward` / `success` — the scalar episode outcome (GRPO group advantages, PPO/REINFORCE++ returns)
- `critique` — natural-language feedback (SDPO's tokenized teacher signal)
- `step_rewards` — per-step progress scores (dense credit assignment / diagnostics)

One call per rollout keeps reward cheap and internally consistent (the same judgement produces the
scalar and the per-step breakdown). The judge sees ONLY the task and the rollout — never a gold
trace — so alternative strategies that accomplish the task score well.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field, ValidationError

from wmh.core.parsing import extract_json_object
from wmh.core.render import render_action
from wmh.core.types import Step
from wmh.providers.base import Message, Provider

REWARD_JUDGE_SYSTEM = """You are a strict evaluator of agent rollouts in a simulated environment.
You are given a TASK and the sequence of steps an agent took (each step: the agent's action and
the environment's response). Judge whether the agent accomplished the task.

Score honestly and strictly:
- An agent that reached the goal state through ANY valid strategy succeeds.
- An agent that claims success without the environment confirming it does NOT succeed.
- Errors the agent recovered from should reduce step scores, not the episode score.

Reply with ONLY a JSON object:
{"success": <bool: did the rollout accomplish the task>,
 "reward": <float 0..1: graded task completion>,
 "step_rewards": [<float 0..1 per step, in order: did this step make progress toward the task>],
 "critique": "<2-4 sentences: what the agent did well / where it went wrong / what it should have
 done instead. Written as feedback TO the agent.>"}
The step_rewards array must have exactly one entry per step shown."""


class EpisodeScore(BaseModel):
    """Everything the reward judge extracts from one rollout, in one call."""

    reward: float  # graded episode-level task completion, 0..1
    success: bool  # binary task completion (GRPO/REINFORCE++ binary reward)
    critique: str  # feedback to the agent; SDPO's tokenized teacher signal
    step_rewards: list[float] = Field(default_factory=list)  # per-step progress, 0..1 each


class _RawEpisodeScore(BaseModel):
    """Lenient view of the judge's JSON before clamping/normalization."""

    reward: float
    success: bool
    critique: str = ""
    step_rewards: list[float] = Field(default_factory=list)


class EpisodeRewardJudge:
    """Scores finished rollouts with a single judge-LLM call. See module docstring."""

    def __init__(self, provider: Provider) -> None:
        self._provider = provider

    def score(
        self, task: str | None, steps: list[Step], rubric: str | None = None
    ) -> EpisodeScore:
        """Judge one rollout; robust to malformed judge replies (flagged zero score, never raises).

        `steps` is the episode in order (e.g. `EpisodeResult.steps` from `run_episode`, or a served
        session's history). An empty rollout scores 0 without calling the LLM.

        `rubric`: the scenario's success criteria (e.g. real tau2's gold `evaluation_criteria`),
        anchors the verdict: without it the judge infers success conditions from the task text
        alone, and that interpretation varies run to run (D65 source 2: the same pinned-world
        rollout judged 0.75 and 0.0 across sessions on a date-semantics reading). The rubric names
        WHAT must be true; the judge still never sees a gold trajectory of HOW.
        """
        if not steps:
            return EpisodeScore(
                reward=0.0, success=False, critique="Empty rollout: no steps to judge."
            )
        user = _build_reward_prompt(task, steps, rubric)
        completion = self._provider.complete(
            REWARD_JUDGE_SYSTEM, [Message(role="user", content=user)], temperature=0.0
        )
        return _parse_episode_score(completion.text, n_steps=len(steps))


def _build_reward_prompt(task: str | None, steps: list[Step], rubric: str | None = None) -> str:
    """Render the rollout for the judge: task, rubric (when given), then each step in order."""
    lines = [f"TASK: {task or '(none given)'}"]
    if rubric:
        lines += ["", "SUCCESS RUBRIC (authoritative — judge against THESE criteria):", rubric]
    lines += ["", f"ROLLOUT ({len(steps)} steps):"]
    for i, step in enumerate(steps, start=1):
        observation = step.observation
        flag = " [ERROR]" if observation.is_error else ""
        lines.append(f"--- step {i} ---")
        lines.append(f"ACTION: {render_action(step.action)}")
        if step.action.arguments:
            lines.append(f"ARGUMENTS: {json.dumps(step.action.arguments, sort_keys=True)}")
        lines.append(f"OBSERVATION{flag}: {observation.content}")
    return "\n".join(lines)


def _parse_episode_score(text: str, n_steps: int) -> EpisodeScore:
    """Parse the judge reply; normalize step_rewards to exactly `n_steps` entries.

    Malformed replies become a flagged zero score rather than raising, so one bad judge reply
    can't abort a training batch. Missing/short step_rewards are padded with 0.0; extras dropped.
    """
    raw = extract_json_object(text)
    if raw is not None:
        try:
            parsed = _RawEpisodeScore.model_validate_json(raw)
        except ValidationError:
            parsed = None
        if parsed is not None:
            step_rewards = [_clamp(r) for r in parsed.step_rewards[:n_steps]]
            step_rewards += [0.0] * (n_steps - len(step_rewards))
            return EpisodeScore(
                reward=_clamp(parsed.reward),
                success=parsed.success,
                critique=parsed.critique.strip(),
                step_rewards=step_rewards,
            )
    return EpisodeScore(
        reward=0.0,
        success=False,
        critique=f"Unparseable reward-judge reply; treated as failure. Raw: {text.strip()[:200]}",
        step_rewards=[0.0] * n_steps,
    )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
