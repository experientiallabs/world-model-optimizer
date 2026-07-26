"""Tests for feedback-directed verification tasks (strict parse, retry once, no fallback)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from wmh.core.types import JsonValue
from wmh.evals.tasks import TaskSpec
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.scenarios.feedback import FeedbackSynthesis, synthesize_verification_tasks


class FakeProvider:
    """Returns queued completion texts in order; records every prompt for assertions."""

    def __init__(self, replies: list[str]) -> None:
        self.config = ProviderConfig(kind=ProviderKind.ANTHROPIC, model="m")
        self._replies = list(replies)
        self.systems: list[str] = []
        self.requests: list[str] = []

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        self.systems.append(system)
        self.requests.append(messages[0].content)
        if not self._replies:
            raise AssertionError("provider called more times than replies were queued")
        return Completion(text=self._replies.pop(0))

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


def _task(task_id: str, instruction: str = "List my open GitHub PRs.") -> dict[str, JsonValue]:
    return {
        "task_id": task_id,
        "instruction": instruction,
        "gold": [
            "The agent called the GitHub API to list pull requests",
            "The agent reported PR #42 'Fix login bug' from acme/webapp",
        ],
    }


def _reply(tasks: list[dict[str, JsonValue]], **extra: JsonValue) -> str:
    payload: dict[str, JsonValue] = {"tasks": tasks, "knowledge_notes": "GitHub notes"}
    payload.update(extra)
    return json.dumps(payload)


def test_happy_path_parses_into_feedback_synthesis() -> None:
    reply = _reply(
        [
            _task("feedback-verify-01"),
            _task("feedback-verify-02", "Look up nonexistent repo acme/ghost."),
        ],
        knowledge_notes="## GitHub\n- repo acme/webapp has PR #42 'Fix login bug'",
    )
    provider = FakeProvider(["Here you go:\n```json\n" + reply + "\n```"])
    result = synthesize_verification_tasks(
        "you should have access to my GitHub",
        provider=provider,
        count=3,
        environment_context="The user works in the acme org.",
    )
    assert isinstance(result, FeedbackSynthesis)
    assert isinstance(result.tasks, tuple)
    assert [t.task_id for t in result.tasks] == ["feedback-verify-01", "feedback-verify-02"]
    assert all(isinstance(t, TaskSpec) for t in result.tasks)
    assert result.tasks[0].instruction == "List my open GitHub PRs."
    assert len(result.tasks[0].gold) == 2
    assert result.knowledge_notes == "## GitHub\n- repo acme/webapp has PR #42 'Fix login bug'"
    # One clean call: the feedback, the caller's context, and the count all reached the model.
    assert len(provider.requests) == 1
    assert "you should have access to my GitHub" in provider.requests[0]
    assert "The user works in the acme org." in provider.requests[0]
    assert "between 1 and 3 tasks" in provider.systems[0]


def test_unparseable_reply_retries_once_then_raises() -> None:
    provider = FakeProvider(["not json at all", "still not json"])
    with pytest.raises(ValueError, match="after one retry.*no JSON object"):
        synthesize_verification_tasks("add GitHub access", provider=provider)
    assert len(provider.requests) == 2
    # The retry carries the first validation error back to the model.
    assert "rejected" in provider.requests[1]
    assert "no JSON object" in provider.requests[1]


def test_duplicate_task_ids_trigger_retry_then_succeed() -> None:
    bad = _reply([_task("feedback-verify-01"), _task("feedback-verify-01")])
    good = _reply([_task("feedback-verify-01")])
    provider = FakeProvider([bad, good])
    result = synthesize_verification_tasks("add GitHub access", provider=provider)
    assert len(provider.requests) == 2
    assert "duplicate task_id 'feedback-verify-01'" in provider.requests[1]
    assert [t.task_id for t in result.tasks] == ["feedback-verify-01"]


def test_empty_gold_triggers_retry_then_raises_when_still_invalid() -> None:
    empty = _task("feedback-verify-01")
    empty["gold"] = []
    whitespace = _task("feedback-verify-01")
    whitespace["gold"] = ["   ", ""]
    provider = FakeProvider([_reply([empty]), _reply([whitespace])])
    with pytest.raises(ValueError, match="no gold assertions"):
        synthesize_verification_tasks("add GitHub access", provider=provider)
    assert len(provider.requests) == 2


def test_empty_instruction_triggers_retry() -> None:
    bad = _reply([_task("feedback-verify-01", instruction="   ")])
    good = _reply([_task("feedback-verify-01")])
    provider = FakeProvider([bad, good])
    result = synthesize_verification_tasks("add GitHub access", provider=provider)
    assert len(provider.requests) == 2
    assert "empty instruction" in provider.requests[1]
    assert result.tasks[0].instruction == "List my open GitHub PRs."


def test_count_is_enforced_and_named_in_the_prompt() -> None:
    three = _reply([_task(f"feedback-verify-0{i}") for i in (1, 2, 3)])
    two = _reply([_task("feedback-verify-01"), _task("feedback-verify-02")])
    provider = FakeProvider([three, two])
    result = synthesize_verification_tasks("add GitHub access", provider=provider, count=2)
    assert len(result.tasks) == 2
    assert "between 1 and 2 tasks" in provider.systems[0]
    assert "expected between 1 and 2 tasks, got 3" in provider.requests[1]


def test_zero_tasks_is_invalid() -> None:
    provider = FakeProvider([_reply([]), _reply([])])
    with pytest.raises(ValueError, match="between 1 and 3 tasks, got 0"):
        synthesize_verification_tasks("add GitHub access", provider=provider)


def test_knowledge_notes_defaults_to_empty_when_omitted() -> None:
    reply = json.dumps({"tasks": [_task("feedback-verify-01")]})
    provider = FakeProvider([reply])
    result = synthesize_verification_tasks("add GitHub access", provider=provider)
    assert result.knowledge_notes == ""


def test_blank_feedback_and_bad_count_raise_without_calling_provider() -> None:
    provider = FakeProvider([])
    with pytest.raises(ValueError, match="nonempty"):
        synthesize_verification_tasks("   ", provider=provider)
    with pytest.raises(ValueError, match="count must be at least 1"):
        synthesize_verification_tasks("add GitHub access", provider=provider, count=0)
    assert provider.requests == []


def test_result_model_is_frozen() -> None:
    provider = FakeProvider([_reply([_task("feedback-verify-01")])])
    result = synthesize_verification_tasks("add GitHub access", provider=provider)
    with pytest.raises(ValidationError):
        result.knowledge_notes = "mutated"  # type: ignore[misc]
