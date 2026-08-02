"""`AgentRuntime`: the minimal agent loop that drives closed-loop rollouts.

A plain, owned while-loop: build the system prompt, ask the agent model for one action,
dispatch it to the environment, append the observation, repeat until `submit` or the turn cap. The
loop is deliberately fixed and small — closed-loop eval tests the *world model*, so the agent must
be a constant: any divergence is then attributable to the world model alone.

Every run yields a `RunResult` whose `steps` are `wmo.core.types.Step`s, so transcripts render with
the same types the rest of the harness uses.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from wmo.core.types import Action, ActionKind, EnvState, Observation, Step
from wmo.harness.e2b_sandbox import SandboxUsage
from wmo.harness.environment import AgentEnvironment, is_env_action
from wmo.harness.skills import SkillLibrary
from wmo.harness.tools import (
    DEFAULT_TOOLS,
    READ_SKILL,
    SUBMIT,
    ToolCall,
    parse_tool_call,
    render_tools,
    resolve_tools,
    to_action,
)
from wmo.providers.base import Message, Provider

JSON_PROTOCOL_CLAUSE = """Every reply MUST be a single JSON object and nothing else:
{"tool": "<tool name>", "arguments": {<the tool's arguments>}}"""
"""The JSON-action calling convention `parse_tool_call` implements.

Only the runtimes that actually parse JSON replies (`AgentRuntime`, `CodeRuntime`) may show this
to the model. The pi runtimes pass STRUCTURED tool schemas and their renderer emits its own
(XML-ish) tool-call notation, so carrying this clause there states a protocol that is not in use;
`strip_json_protocol_clause` removes it on that path."""

DEFAULT_SYSTEM_PROMPT = f"""You are a capable command-line agent working inside a Linux environment.
You are given a task. Accomplish it by taking ONE action at a time.

{JSON_PROTOCOL_CLAUSE}

Work in small, verifiable steps: inspect state, act, check the result, then continue. When the
task is done, call `submit` with your answer. Prefer composing small bash commands over guessing."""


def strip_json_protocol_clause(prompt: str) -> str:
    """Return `prompt` without the JSON-action clause, for structured tool calling.

    A no-op when the clause is absent (an optimizer-edited prompt surface, or a seed harness that
    never carried it), so this is safe to apply to any authored prompt.

    Args:
        prompt: The assembled system prompt.

    Returns:
        The prompt with the JSON-action paragraph (and the blank line it owned) removed.
    """
    if JSON_PROTOCOL_CLAUSE not in prompt:
        return prompt
    return prompt.replace(f"\n\n{JSON_PROTOCOL_CLAUSE}", "").replace(JSON_PROTOCOL_CLAUSE, "")


DEFAULT_MAX_TURNS = 20  # small shell tasks converge well before this; raise for longer horizons
# Per-call output budget used by the pi runtimes. This remains separate from the turn cap: a
# reasoning model can exhaust one response before it emits a tool call even when many turns remain.
DEFAULT_MAX_OUTPUT_TOKENS = 4096

# A score cell must finish much sooner than the E2B lease-renewal safety cap. This wall budget is
# host-enforced by RunnerLink and may be exceeded only by one already-running provider/tool call.
# It lives here (not pi_e2b) so timeout policy validates without importing the optional e2b path.
DEFAULT_EVAL_EPISODE_TIMEOUT_S = 300.0

# Per-observation cap in the judge-facing transcript. Generous rather than tight: gold evidence
# routinely lives deep in long outputs (`cat` of a produced file, `ls -R`), and truncating it away
# turns real successes into judged failures.
TRANSCRIPT_OBS_CHARS = 2000

_NUDGE = (
    "[ERROR] that reply was not a single valid JSON tool call. Reply with EXACTLY one JSON "
    'object: {"tool": "<tool name>", "arguments": {...}}'
)


def validate_episode_timeout_s(value: object) -> float:
    """Return one finite positive episode timeout, rejecting booleans and non-numbers.

    The single entry-point validation for the episode wall budget: `HarnessDoc.runtime()` and
    the runtimes call this once instead of re-validating the number at every layer.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("episode_timeout_s must be a finite positive number")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("episode_timeout_s must be a finite positive number")
    return timeout


class HarnessSearchCancelled(RuntimeError):
    """The caller requested that an optimizer search stop before its next costly phase.

    ``worker_usage`` aggregates every completed score wave plus the partial
    wave interrupted by cancellation. ``sandbox_usage`` is populated by
    ``create_harness`` after its evaluator pool has been closed. Together they
    let a caller persist already-incurred runtime spend even though
    cancellation intentionally produces no partial search result.
    """

    def __init__(
        self,
        message: str = "harness search cancelled",
        *,
        worker_usage: TokenUsage | None = None,
        sandbox_usage: SandboxUsage | None = None,
    ) -> None:
        super().__init__(message)
        self.worker_usage = worker_usage
        self.sandbox_usage = sandbox_usage


class RuntimeCancelled(RuntimeError):
    """The caller cancelled an in-flight runtime episode.

    This is deliberately distinct from a normal ``RunResult`` stop reason: evaluation must not
    send cancelled cells to a judge, and transport owners must not retry them as infrastructure
    failures. Runtimes that cannot interrupt an active provider call raise this immediately after
    that bounded call returns. Self-metering runtimes attach the worker tokens
    incurred before that boundary so an owning evaluator can aggregate them
    while it drains sibling episodes.
    """

    def __init__(
        self,
        message: str = "runtime episode cancelled",
        *,
        worker_usage: TokenUsage | None = None,
    ) -> None:
        super().__init__(message)
        self.worker_usage = worker_usage


@runtime_checkable
class Runtime(Protocol):
    """What closed-loop eval drives: any object that can run one episode against an environment.

    `AgentRuntime` (the fixed loop) and `CodeRuntime` (the harness's own program) both satisfy
    this; evaluation code depends on the shape, never the implementation.
    """

    def run(self, task_id: str, instruction: str, environment: AgentEnvironment) -> RunResult: ...


class StopReason(StrEnum):
    """Why one episode ended.

    The four `pi`-specific members below exist because a single `done` frame used to cover four
    completely different events, all of them scored as clean submissions: a real `submit`, a
    prose-only turn, a turn truncated at the output-token cap, and a tool call the renderer could
    not parse. Anything other than `SUBMITTED` is a SCAFFOLD loss, not a measured task failure
    (see `wmo.distill.rollouts.RolloutStats.scaffold_loss_rate`).
    """

    SUBMITTED = "submitted"  # the agent called submit
    MAX_TURNS = "max_turns"  # hit the turn cap without submitting
    NO_ACTION = "no_action"  # the agent produced no parseable tool call
    ERROR = "error"  # harness code raised (CodeRuntime episodes only)
    BUDGET = "budget"  # harness code exhausted an episode budget (CodeRuntime episodes only)
    NO_TOOL_CALL = "no_tool_call"  # prose-only turns exhausted the nudge budget
    OUTPUT_TRUNCATED = "output_truncated"  # last turn was cut at the output-token cap
    UNPARSED_TOOL_CALL = "unparsed_tool_call"  # the renderer could not parse the emitted call
    PROVIDER_ERROR = "provider_error"  # the worker LLM call kept failing (e.g. context overflow)
    UNKNOWN_DONE_REASON = "unknown_done_reason"  # a `done` frame carried no reason we recognize


SCAFFOLD_LOSS_STOP_REASONS = frozenset(
    {
        StopReason.MAX_TURNS,
        StopReason.NO_ACTION,
        StopReason.ERROR,
        StopReason.BUDGET,
        StopReason.NO_TOOL_CALL,
        StopReason.OUTPUT_TRUNCATED,
        StopReason.UNPARSED_TOOL_CALL,
        StopReason.PROVIDER_ERROR,
        StopReason.UNKNOWN_DONE_REASON,
    }
)
"""Every stop reason that is NOT a self-declared completion.

An episode that ends on one of these was cut off by the harness, so its reward measures where the
guillotine fell rather than what the model can do. Exactly the complement of `SUBMITTED`, spelled
out so a new member cannot silently join the "looks like a completion" side."""

MAX_NONACTION_TURNS = 3
"""Consecutive non-action assistant turns a pi runner nudges through before giving up.

The reference terminus-2 agent never ends an episode on a parse failure: it salvages what it can,
feeds the parser's complaint back, and keeps going. Three is the bounded version of that: enough
for a model to recover from one malformed emission or one truncation, few enough that a model
which has genuinely stopped acting does not burn the whole wall budget on prose. Mirrored in
`pi_entry/runner_termination.ts`, which is where the loop actually runs."""


class TokenUsage(BaseModel):
    """Worker-LLM token spend, aggregated over one or more episodes.

    Populated by runtimes whose LLM calls bypass the `Provider` abstraction (the pi worker path
    answers `llm_request` frames with raw native tool-calling requests), so callers can meter the
    agent leg the way provider-wrapped runtimes are metered. `None` on a result means the runtime
    does not report usage — not that the run was free.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    reasoning_tokens: int = 0
    calls: int = 0
    call_seconds: list[float] = Field(default_factory=list)
    # Per-call counters preserve pricing boundaries that aggregate episode totals cannot.
    # OpenAI long-context rates, for example, are selected by one request's prompt width rather
    # than by the sum over an agent episode. Empty on legacy/runtime paths that only meter totals.
    call_input_tokens: list[int] = Field(default_factory=list)
    call_output_tokens: list[int] = Field(default_factory=list)
    call_cached_input_tokens: list[int] = Field(default_factory=list)
    call_cache_write_input_tokens: list[int] = Field(default_factory=list)


def combine_usage(parts: Iterable[TokenUsage | None]) -> TokenUsage | None:
    """Sum the reported usages; `None` when nothing reported (order-independent)."""
    reported = [p for p in parts if p is not None]
    if not reported:
        return None
    return TokenUsage(
        input_tokens=sum(p.input_tokens for p in reported),
        output_tokens=sum(p.output_tokens for p in reported),
        cached_input_tokens=sum(p.cached_input_tokens for p in reported),
        cache_write_input_tokens=sum(p.cache_write_input_tokens for p in reported),
        reasoning_tokens=sum(p.reasoning_tokens for p in reported),
        calls=sum(p.calls for p in reported),
        call_seconds=[seconds for p in reported for seconds in p.call_seconds],
        call_input_tokens=[tokens for p in reported for tokens in p.call_input_tokens],
        call_output_tokens=[tokens for p in reported for tokens in p.call_output_tokens],
        call_cached_input_tokens=[
            tokens for p in reported for tokens in p.call_cached_input_tokens
        ],
        call_cache_write_input_tokens=[
            tokens for p in reported for tokens in p.call_cache_write_input_tokens
        ],
    )


class RunResult(BaseModel):
    """The outcome of one rollout: the transcript, why it stopped, and any answer."""

    task_id: str
    instruction: str = ""
    steps: list[Step] = Field(default_factory=list)
    stop_reason: StopReason
    answer: str = ""
    turns: int = 0
    worker_usage: TokenUsage | None = None  # set by runtimes that answer the worker LLM directly

    def transcript(self) -> str:
        """A compact judge-readable transcript of the run."""
        lines: list[str] = []
        for i, step in enumerate(self.steps, 1):
            act = step.action
            desc = act.name or (act.content or "")
            if act.kind == ActionKind.TOOL_CALL and act.arguments:
                desc = f"{act.name} {act.arguments}"
            lines.append(f"[{i}] {act.kind.value}: {desc}")
            lines.append(f"    -> {step.observation.content[:TRANSCRIPT_OBS_CHARS]}")
        return "\n".join(lines)


class AgentRuntime:
    """Drives the fixed agent loop against one `AgentEnvironment`.

    `provider` is the *agent* model — a separate role from the world model serving the simulated
    environment (they may be the same backend). The prompt/tools/limits are parameters so the
    harness layer can construct configured runtimes, but within a run they never change.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        tools: list[str] | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        temperature: float = 0.7,
        skills: SkillLibrary | None = None,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        self._provider = provider
        self._system_prompt = system_prompt
        self._skills = skills if skills is not None else SkillLibrary()
        tool_names = list(tools) if tools is not None else list(DEFAULT_TOOLS)
        # A skill-bearing harness needs read_skill for progressive disclosure; add it implicitly so
        # harness config files don't have to remember the plumbing tool.
        if len(self._skills) and READ_SKILL.name not in tool_names:
            tool_names.append(READ_SKILL.name)
        self._tools = resolve_tools(tool_names)
        self._max_turns = max_turns
        self._temperature = temperature

    def run(self, task_id: str, instruction: str, environment: AgentEnvironment) -> RunResult:
        messages: list[Message] = [Message(role="user", content=f"TASK: {instruction}")]
        steps: list[Step] = []
        state = EnvState()
        nudged = False

        for turn in range(1, self._max_turns + 1):
            completion = self._provider.complete(
                self._full_system_prompt(), messages, temperature=self._temperature
            )
            reply = completion.text.strip()
            call = parse_tool_call(reply)
            if call is None:
                # One recovery nudge per run (symmetric with the unavailable-tool path, which also
                # feeds an error back): at nonzero temperature a single malformed reply is agent
                # noise, and aborting the rollout would charge it to the world model's score.
                if not nudged:
                    nudged = True
                    messages.append(Message(role="assistant", content=reply))
                    messages.append(Message(role="user", content=_NUDGE))
                    continue
                return self._result(task_id, steps, StopReason.NO_ACTION, turns=turn)

            if call.tool == SUBMIT.name:
                answer = _str_arg(call, "answer")
                steps.append(
                    _step(to_action(call), Observation(content=answer), state, instruction)
                )
                return self._result(task_id, steps, StopReason.SUBMITTED, answer=answer, turns=turn)

            action, observation = self._dispatch(call, environment)
            steps.append(_step(action, observation, state, instruction))
            state = _advance(state, observation)
            messages.append(Message(role="assistant", content=reply))
            messages.append(Message(role="user", content=_observation_text(observation)))

        return self._result(task_id, steps, StopReason.MAX_TURNS, turns=self._max_turns)

    def _dispatch(
        self, call: ToolCall, environment: AgentEnvironment
    ) -> tuple[Action, Observation]:
        """Route one non-submit call: read_skill handled here, env tools to the environment."""
        action = to_action(call)
        if call.tool not in {t.name for t in self._tools}:
            return action, Observation(content=f"tool {call.tool!r} not available", is_error=True)
        if call.tool == READ_SKILL.name:
            name = _str_arg(call, "name")
            skill = self._skills.get(name)
            if skill is None:
                return action, Observation(content=f"no skill named {name!r}", is_error=True)
            return action, Observation(content=skill.body)
        if not is_env_action(action):
            return action, Observation(content=f"tool {call.tool!r} not available", is_error=True)
        return action, environment.execute(action)

    def _full_system_prompt(self) -> str:
        prompt = f"{self._system_prompt}\n\n## Tools\n{render_tools(self._tools)}"
        index = self._skills.render_index()
        if index:
            prompt += f"\n\n## Your skills (read a body with read_skill)\n{index}"
        return prompt

    def _result(
        self,
        task_id: str,
        steps: list[Step],
        stop_reason: StopReason,
        *,
        answer: str = "",
        turns: int,
    ) -> RunResult:
        return RunResult(
            task_id=task_id, steps=steps, stop_reason=stop_reason, answer=answer, turns=turns
        )


def _str_arg(call: ToolCall, key: str) -> str:
    value = call.arguments.get(key)
    return value if isinstance(value, str) else ""


def _step(action: Action, observation: Observation, state: EnvState, instruction: str) -> Step:
    return Step(action=action, observation=observation, state_before=state, task=instruction)


def _advance(state: EnvState, observation: Observation) -> EnvState:
    """Carry a one-line note forward into the next step's state (mirrors WorldModel scratchpad)."""
    note = observation.metadata.get("state_note")
    if isinstance(note, str) and note.strip():
        prefix = f"{state.scratchpad}\n" if state.scratchpad else ""
        return EnvState(structured=state.structured, scratchpad=f"{prefix}- {note.strip()}")
    return state


def _observation_text(observation: Observation) -> str:
    tag = "ERROR" if observation.is_error else "OK"
    return f"[{tag}] {observation.content}"
