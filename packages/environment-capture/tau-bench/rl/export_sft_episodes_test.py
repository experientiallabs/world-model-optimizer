"""Tests for the SFT episode exporter's leakage filter and record shape."""

from __future__ import annotations

from export_sft_episodes import SftStep, episodes_from_traces

from wmh.core.types import Action, ActionKind, Observation, Step, Trace


def _trace(trace_id: str, task: str, tool: str = "get_user_details") -> Trace:
    return Trace(
        trace_id=trace_id,
        steps=[
            Step(
                task=task,
                action=Action(kind=ActionKind.TOOL_CALL, name=tool, arguments={"user_id": "u1"}),
                observation=Observation(content="ok", is_error=False),
            )
        ],
        metadata={"domain": "airline"},
    )


def test_drops_eval_task_traces_and_keeps_shape() -> None:
    traces = [_trace("t1", "train task"), _trace("t2", "eval task"), _trace("t3", "train task")]
    episodes = episodes_from_traces(traces, eval_tasks={"eval task"})
    assert [e.trace_id for e in episodes] == ["t1", "t3"]
    episode = episodes[0]
    assert episode.task == "train task"
    assert episode.domain == "airline"
    assert episode.steps == [
        SftStep(
            name="get_user_details",
            arguments={"user_id": "u1"},
            observation="ok",
            is_error=False,
        )
    ]


def test_skips_taskless_and_stepless_traces() -> None:
    taskless = Trace(
        trace_id="t0",
        steps=[
            Step(
                task="",
                action=Action(kind=ActionKind.TOOL_CALL, name="x", arguments={}),
                observation=Observation(content="ok"),
            )
        ],
        metadata={},
    )
    messages_only = Trace(
        trace_id="t1",
        steps=[
            Step(
                task="talk",
                action=Action(kind=ActionKind.MESSAGE, content="hello"),
                observation=Observation(content="hi"),
            )
        ],
        metadata={},
    )
    assert episodes_from_traces([taskless, messages_only], eval_tasks=set()) == []
