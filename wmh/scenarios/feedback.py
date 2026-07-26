"""Feedback-directed verification tasks: turn one piece of user feedback into a must-pass gate.

One piece of natural-language feedback about an agent ("you should have access to my GitHub")
names a capability the current harness lacks. An LLM converts that feedback into a small set of
verification `TaskSpec`s that exercise the NEW capability directly, plus the knowledge notes the
world-model environment needs to simulate it plausibly. The tasks are the acceptance gate for a
feedback-directed one-step harness improvement: they run closed-loop against the world model and
are judged by their gold assertions, so an improvement only counts when the new capability
demonstrably works.

Unlike the trace-driven `ScenarioSynthesizer` (which degrades gracefully because a corpus has
thousands of traces), a gate synthesized from one sentence of feedback has no fallback that is
still meaningful: an unusable reply is retried once with the validation error, then raised.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wmh.core.parsing import extract_json_object
from wmh.evals.tasks import TaskSpec
from wmh.providers.base import Message, Provider

FEEDBACK_SYNTHESIS_SYSTEM = """You design verification tasks for an AI agent. A user gave one
piece of feedback naming a capability the agent should have but currently lacks. Write tasks
that PROVE the new capability works once the agent's harness is improved.

Respond with ONLY a JSON object, no prose around it:
{{"tasks": [{{"task_id": "feedback-verify-01", "instruction": "<self-contained task a user would
give the agent>", "gold": ["<concrete assertion>", "..."]}}, ...],
 "knowledge_notes": "<markdown facts the environment simulator needs>"}}

Rules:
- Produce between 1 and {count} tasks.
- Each instruction is a self-contained task a real user would give the agent, and it must
  exercise the NEW capability the feedback asks for. Never write a meta-task about editing the
  agent's configuration, prompt, or tools: the task is what a user asks AFTER the capability
  exists.
- Each gold list holds 2 to 5 concrete, outcome-focused assertions that a judge can check from
  the run transcript alone: the agent invoked the new capability, it surfaced the correct data,
  and it did not fabricate results it never retrieved.
- Tasks must differ from each other. When the task budget allows more than one, cover the happy
  path plus at least one edge or error path (for example: authentication missing, or the
  requested entity does not exist).
- knowledge_notes gives the world-model environment simulator the facts it needs to answer the
  new tool calls consistently: tool or API names, request/response shapes, and realistic sample
  entities with concrete names, ids, and values that the gold assertions can reference. Use an
  empty string only when the environment truly needs nothing new.
- task_id values must be unique, contain only characters safe in kebab-case or snake_case, and
  start with "feedback-verify-"."""


class FeedbackSynthesis(BaseModel):
    """Verification tasks plus the environment knowledge needed to simulate the new capability.

    `knowledge_notes` is markdown destined for the world-model environment's knowledge base: the
    tool names, request/response shapes, and sample entities the simulator needs to answer the
    new capability's tool calls consistently. Empty when nothing is needed.
    """

    model_config = ConfigDict(frozen=True)

    tasks: tuple[TaskSpec, ...]
    knowledge_notes: str = ""


class _RawTask(BaseModel):
    task_id: str
    instruction: str
    gold: list[str] = Field(default_factory=list)


class _RawFeedbackSynthesis(BaseModel):
    tasks: list[_RawTask] = Field(default_factory=list)
    knowledge_notes: str = ""


def synthesize_verification_tasks(
    feedback: str,
    *,
    provider: Provider,
    count: int = 3,
    environment_context: str = "",
) -> FeedbackSynthesis:
    """Synthesize the must-pass verification tasks for one piece of user feedback.

    Args:
        feedback: One piece of natural-language user feedback about the agent, naming the
            capability it should gain (e.g. "you should have access to my GitHub").
        provider: LLM provider used to write the tasks.
        count: Maximum number of tasks to synthesize (the model returns 1 to `count`).
        environment_context: Optional caller-supplied prose about the environment (e.g. a
            knowledge-base excerpt) that grounds the tasks in existing entities.

    Returns:
        The validated tasks plus the knowledge notes the environment simulator needs.

    Raises:
        ValueError: When `feedback` is blank, `count` is not positive, or the model failed to
            produce a valid reply after one retry. There is no silent fallback: these tasks gate
            a harness change, so a degraded task set must fail loudly, not pass vacuously.
    """
    if not feedback.strip():
        raise ValueError("feedback must be a nonempty description of the desired capability")
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")
    system = FEEDBACK_SYNTHESIS_SYSTEM.format(count=count)
    request = _render_request(feedback, environment_context)
    completion = provider.complete(
        system,
        [Message(role="user", content=request)],
        temperature=0.0,
        max_tokens=4096,
    )
    try:
        return _parse_synthesis(completion.text, count=count)
    except ValueError as first_error:
        retry_request = (
            f"{request}\n\nYour previous reply was rejected: {first_error}\n"
            "Reply again with ONLY the corrected JSON object."
        )
        completion = provider.complete(
            system,
            [Message(role="user", content=retry_request)],
            temperature=0.0,
            max_tokens=4096,
        )
        try:
            return _parse_synthesis(completion.text, count=count)
        except ValueError as second_error:
            raise ValueError(
                "could not synthesize verification tasks from the feedback after one retry: "
                f"{second_error}"
            ) from second_error


def _render_request(feedback: str, environment_context: str) -> str:
    """Render the user message: the feedback, grounded by any caller-supplied context."""
    sections = [f"User feedback about the agent:\n{feedback.strip()}"]
    if environment_context.strip():
        sections.append(f"Known environment context:\n{environment_context.strip()}")
    return "\n\n".join(sections)


def _parse_synthesis(text: str, *, count: int) -> FeedbackSynthesis:
    """Parse and validate one model reply; raise ValueError describing the first defect found."""
    raw = extract_json_object(text)
    if raw is None:
        raise ValueError("reply contains no JSON object")
    try:
        parsed = _RawFeedbackSynthesis.model_validate_json(raw)
    except ValidationError as exc:
        raise ValueError(f"reply JSON does not match the contract: {exc}") from exc
    if not 1 <= len(parsed.tasks) <= count:
        raise ValueError(f"expected between 1 and {count} tasks, got {len(parsed.tasks)}")
    tasks: list[TaskSpec] = []
    seen_ids: set[str] = set()
    for index, raw_task in enumerate(parsed.tasks, start=1):
        try:
            spec = TaskSpec(
                task_id=raw_task.task_id.strip(),
                instruction=raw_task.instruction.strip(),
                gold=[assertion.strip() for assertion in raw_task.gold if assertion.strip()],
            )
        except ValidationError as exc:
            raise ValueError(f"task {index} is not a valid TaskSpec: {exc}") from exc
        if not spec.task_id:
            raise ValueError(f"task {index} has an empty task_id")
        if not spec.instruction:
            raise ValueError(f"task {spec.task_id!r} has an empty instruction")
        if not spec.gold:
            raise ValueError(f"task {spec.task_id!r} has no gold assertions")
        if spec.task_id in seen_ids:
            raise ValueError(f"duplicate task_id {spec.task_id!r}")
        seen_ids.add(spec.task_id)
        tasks.append(spec)
    return FeedbackSynthesis(tasks=tuple(tasks), knowledge_notes=parsed.knowledge_notes.strip())
