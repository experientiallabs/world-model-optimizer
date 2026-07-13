"""`wmh demo`: replay a real recorded scenario against the world model, open loop.

A randomly sampled trace supplies the task and the agent's recorded actions; the world model
predicts each observation while the session is teacher-forced with the recorded one
(`step_open_loop`), so every prediction is conditioned on the real trajectory. The result shows
predicted vs. actual side by side — the harness working end-to-end without needing a live agent.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from wmh.core.types import Action, EnvState, Observation, Trace
from wmh.engine.world_model import WorldModel


def seed_state_for(trace: Trace) -> EnvState | None:
    """The first step's recorded env state, when it carries something worth seeding.

    Returns `None` for an empty snapshot (no structured config and no scratchpad) so the session
    falls back to `new_session`'s default `EnvState()` rather than being handed a pointless empty
    seed. Most provider traces leave `state_before` empty; benchmarks that capture a compact env
    state (e.g. bird-sql) populate it, and seeding it lets the replay start from the real initial
    state instead of the world model hallucinating it.
    """
    if not trace.steps:
        return None
    state = trace.steps[0].state_before
    return state if (state.structured or state.scratchpad) else None


class DemoStep(BaseModel):
    """One replayed step: the recorded action with the predicted and recorded observations."""

    action: Action
    predicted: Observation
    actual: Observation

    @property
    def exact_match(self) -> bool:
        return self.predicted.content.strip() == self.actual.content.strip()


class DemoReplay(BaseModel):
    """The rendered scenario replay for `wmh demo`."""

    trace_id: str
    task: str | None
    steps: list[DemoStep]
    first_env_prompt: str  # the exact prompt the world model saw for the first step


def run_demo(
    world_model: WorldModel,
    trace: Trace,
    max_steps: int = 5,
    on_step: Callable[[int, int], None] | None = None,
    on_result: Callable[[int, int, DemoStep], None] | None = None,
    skip: int = 0,
) -> DemoReplay:
    """Replay up to `max_steps` of `trace` open-loop and collect predicted vs. actual.

    `on_step(i, total)` fires before each prediction and `on_result(i, total, step)` right
    after it, so callers can stream the interaction as it happens. `skip` resumes a replay
    mid-scenario (e.g. after a provider switch): the skipped prefix seeds the session from the
    recorded trajectory without any predictions, and the returned steps cover only the rest.
    """
    if not trace.steps:
        raise ValueError(f"trace {trace.trace_id!r} has no steps to replay")
    task = trace.steps[0].task
    session = world_model.new_session(task=task, seed_state=seed_state_for(trace))
    replayed = trace.steps[:max_steps]
    if skip:
        world_model.seed_session(session.id, replayed[:skip])
    first_env_prompt = world_model.render_step_prompt(
        session.id, replayed[min(skip, len(replayed) - 1)].action
    )

    steps: list[DemoStep] = []
    for i, step in enumerate(replayed[skip:], start=skip + 1):
        if on_step is not None:
            on_step(i, len(replayed))
        predicted = world_model.step_open_loop(session.id, step.action, step.observation)
        result = DemoStep(action=step.action, predicted=predicted, actual=step.observation)
        steps.append(result)
        if on_result is not None:
            on_result(i, len(replayed), result)
    return DemoReplay(
        trace_id=trace.trace_id, task=task, steps=steps, first_env_prompt=first_env_prompt
    )
