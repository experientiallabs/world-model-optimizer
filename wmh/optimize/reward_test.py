"""Tests for the episode reward judge."""

from __future__ import annotations

import json

from wmh.core.types import Action, ActionKind, Observation, Step
from wmh.optimize.reward import EpisodeRewardJudge, EpisodeScore, _build_reward_prompt
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind


def _step(content: str = "ok", is_error: bool = False) -> Step:
    return Step(
        action=Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={"id": "u1"}),
        observation=Observation(content=content, is_error=is_error),
    )


class FakeProvider:
    """Returns a canned completion text; records the prompts it saw."""

    def __init__(self, reply: str) -> None:
        self.config = ProviderConfig(kind=ProviderKind.ANTHROPIC, model="m")
        self._reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        self.calls.append((system, messages[0].content))
        return Completion(text=self._reply)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


def test_scores_parse_clamp_and_truncate() -> None:
    reply = json.dumps(
        {"success": True, "reward": 1.7, "step_rewards": [0.5, 2.0, -1.0, 0.9], "critique": "good"}
    )
    judge = EpisodeRewardJudge(FakeProvider(reply))
    score = judge.score("book a flight", [_step(), _step()])
    assert score.success is True
    assert score.reward == 1.0  # clamped
    assert score.step_rewards == [0.5, 1.0]  # clamped + truncated to n_steps
    assert score.critique == "good"


def test_short_step_rewards_are_padded() -> None:
    reply = json.dumps({"success": False, "reward": 0.3, "step_rewards": [0.3], "critique": "meh"})
    judge = EpisodeRewardJudge(FakeProvider(reply))
    score = judge.score("task", [_step(), _step(), _step()])
    assert score.step_rewards == [0.3, 0.0, 0.0]


def test_unparseable_reply_is_flagged_zero_not_raise() -> None:
    judge = EpisodeRewardJudge(FakeProvider("the agent did great, five stars"))
    score = judge.score("task", [_step()])
    assert score.reward == 0.0
    assert score.success is False
    assert "Unparseable" in score.critique
    assert score.step_rewards == [0.0]


def test_empty_rollout_scores_zero_without_llm_call() -> None:
    provider = FakeProvider("{}")
    score = EpisodeRewardJudge(provider).score("task", [])
    expected = EpisodeScore(reward=0.0, success=False, critique="Empty rollout: no steps to judge.")
    assert score == expected
    assert provider.calls == []


def test_prompt_shows_task_actions_observations_and_error_flags() -> None:
    prompt = _build_reward_prompt("find the reservation", [_step("not found", is_error=True)])
    assert "TASK: find the reservation" in prompt
    assert "get_user" in prompt
    assert '"id": "u1"' in prompt
    assert "[ERROR]" in prompt
    assert "not found" in prompt


def test_judge_never_sees_gold_trace() -> None:
    """The prompt must contain only task + rollout — no reference/gold observation channel."""
    provider = FakeProvider(json.dumps({"success": True, "reward": 1.0, "critique": ""}))
    EpisodeRewardJudge(provider).score("task", [_step()])
    _system, user = provider.calls[0]
    assert "gold" not in user.lower()
    assert "reference" not in user.lower()


def test_rubric_reaches_the_judge_prompt_and_shapes_the_system() -> None:
    """A scenario's success rubric anchors the judgement (D65 source-2 fix)."""
    provider = FakeProvider(
        json.dumps({"success": True, "reward": 1.0, "step_rewards": [1.0], "critique": "ok"})
    )
    rubric = "Task is complete iff reservation is moved to 2024-05-24 and paid by gift card."
    EpisodeRewardJudge(provider).score("change my flight", [_step()], rubric=rubric)
    system, user = provider.calls[0]
    assert rubric in user
    assert "SUCCESS RUBRIC" in user
    # without a rubric the prompt must not carry an empty rubric section
    provider2 = FakeProvider(
        json.dumps({"success": True, "reward": 1.0, "step_rewards": [1.0], "critique": "ok"})
    )
    EpisodeRewardJudge(provider2).score("change my flight", [_step()])
    _s2, user2 = provider2.calls[0]
    assert "SUCCESS RUBRIC" not in user2
