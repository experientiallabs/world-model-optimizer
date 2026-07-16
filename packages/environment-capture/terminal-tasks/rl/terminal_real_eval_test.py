"""Tests for the real-terminal eval harness: policy loop, record shaping, error handling.

No docker and no network: the endpoint is a scripted `httpx.MockTransport`, the bash executor is a
fake, and `evaluate` is driven with a fake `run_one`.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
from terminal_real_eval import NUDGE, Scenario, evaluate, run_policy_loop

from wmh.core.types import Action, ActionKind, JsonObject, Observation, Step
from wmh.optimize.reward import EpisodeScore

TOOLS: list[JsonObject] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command in the task shell.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]


def _tool_turn(command: str, call_id: str = "c1") -> dict:
    """An assistant message that calls bash with `command` (OpenAI tool-call shape)."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": command})},
            }
        ],
    }


def _text_turn(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _client(scripted: list[dict], calls: list[dict]) -> httpx.Client:
    """A client whose endpoint replays `scripted` messages in order, recording request bodies."""

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        message = scripted[min(len(calls) - 1, len(scripted) - 1)]
        return httpx.Response(200, json={"choices": [{"message": message}]})

    return httpx.Client(transport=httpx.MockTransport(handler))


def _run(
    scripted: list[dict],
    execute: Callable[[str], tuple[str, int]],
    *,
    max_steps: int = 20,
) -> tuple[list[Step], list[dict]]:
    calls: list[dict] = []
    client = _client(scripted, calls)
    steps = run_policy_loop(
        "do the thing",
        client=client,
        api_base="http://test/v1",
        api_key="k",
        model="ckpt-1",
        tools=TOOLS,
        temperature=1.0,
        max_steps=max_steps,
        execute=execute,
    )
    return steps, calls


def test_policy_loop_executes_tool_and_ends_after_nudge() -> None:
    executed: list[str] = []

    def execute(command: str) -> tuple[str, int]:
        executed.append(command)
        return "hi\n", 0

    scripted = [
        _tool_turn("echo hi"),
        _text_turn("Looks done to me."),  # no tool call -> triggers the single nudge
        _text_turn("Confirmed complete."),  # second toolless turn -> episode ends
    ]
    steps, calls = _run(scripted, execute)

    assert executed == ["echo hi"]
    assert len(steps) == 1
    assert steps[0].action.kind is ActionKind.TOOL_CALL
    assert steps[0].action.name == "bash"
    assert steps[0].action.arguments == {"command": "echo hi"}
    assert steps[0].observation.content == "hi\n"
    assert not steps[0].observation.is_error

    # tool turn + nudge turn + ending turn = 3 endpoint calls
    assert len(calls) == 3
    # the model carries the tool output and then the nudge into later requests
    assert any(m.get("content") == "hi\n" and m["role"] == "tool" for m in calls[1]["messages"])
    assert any(m.get("content") == NUDGE and m["role"] == "user" for m in calls[2]["messages"])
    # payload carries the model name and the derived tool spec
    assert calls[0]["model"] == "ckpt-1"
    assert calls[0]["tools"][0]["function"]["name"] == "bash"


def test_nonzero_exit_is_prefixed_and_flagged() -> None:
    def execute(command: str) -> tuple[str, int]:
        return "boom\n", 2

    scripted = [_tool_turn("false"), _text_turn("giving up"), _text_turn("done")]
    steps, calls = _run(scripted, execute)

    assert steps[0].observation.content == "[exit 2] boom\n"
    assert steps[0].observation.is_error
    # the prefixed output is what gets fed back to the policy
    assert any(m.get("content") == "[exit 2] boom\n" for m in calls[1]["messages"])


def test_max_steps_stops_a_nonstopping_policy() -> None:
    def execute(command: str) -> tuple[str, int]:
        return "", 0

    # every turn calls a tool, so only the step cap can end it
    steps, calls = _run([_tool_turn("sleep 0")], execute, max_steps=3)
    assert len(steps) == 3
    assert len(calls) == 3


def test_evaluate_builds_records_keyed_by_provenance() -> None:
    scenarios = [
        Scenario(task="t1", provenance=["aaa"], category="x"),
        Scenario(task="t2", provenance=["bbb"], category="y"),  # this one crashes
    ]
    good_steps = [
        Step(
            action=Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": "ls"}),
            observation=Observation(content="ok"),
        )
    ]

    def run_one(task: str) -> tuple[EpisodeScore, list[Step]]:
        if task == "t2":
            raise RuntimeError("docker boom")
        return EpisodeScore(reward=1.0, success=True, critique="good"), good_steps

    records = evaluate(scenarios, run_one, trials=2, concurrency=2)

    assert len(records) == 4  # 2 scenarios x 2 trials
    good = [r for r in records if r.scenario_id == "aaa"]
    bad = [r for r in records if r.scenario_id == "bbb"]
    assert {r.rollout_index for r in good} == {0, 1}
    assert all(r.env == "real-terminal" for r in records)
    assert all(
        r.success and r.reward == 1.0 and r.steps == 1 and r.errors == [] and r.critique == "good"
        for r in good
    )
    assert all(not r.success and r.steps == 0 and r.errors for r in bad)
    assert "docker boom" in bad[0].errors[0]


def test_long_command_output_is_truncated_head_and_tail() -> None:
    """Unbounded outputs overflow the policy context (400s); cap keeps head + tail."""
    from terminal_real_eval import MAX_OUTPUT_CHARS

    big = "A" * 10_000 + "TAIL"

    def execute(command: str) -> tuple[str, int]:
        return big, 0

    scripted = [_tool_turn("cat big"), _text_turn("done")]
    steps, calls = _run(scripted, execute)

    content = steps[0].observation.content
    assert len(content) < MAX_OUTPUT_CHARS + 200
    assert content.startswith("A" * 100)
    assert content.endswith("TAIL")
    assert "chars truncated" in content
    # the truncated form (not the raw dump) is what returns to the policy
    assert any("chars truncated" in str(m.get("content", "")) for m in calls[1]["messages"])


def test_malformed_tool_arguments_are_normalized_on_replay() -> None:
    """WM-trained ckpts emit sloppy argument JSON; replaying it verbatim 400s vLLM."""
    import json as json_mod

    executed = []

    def execute(command: str) -> tuple[str, int]:
        executed.append(command)
        return "ok", 0

    bad_args = '{"command": "ls /tmp"} trailing-garbage'
    bad_turn = _tool_turn("ignored")
    bad_turn["tool_calls"][0]["function"]["arguments"] = bad_args
    scripted = [bad_turn, _text_turn("done"), _text_turn("done")]
    steps, calls = _run(scripted, execute)
    # the parsed command still executed, and the REPLAYED arguments are clean JSON
    assert executed and "ls /tmp" in executed[0]
    replayed = next(
        m for m in calls[1]["messages"] if m["role"] == "assistant" and m.get("tool_calls")
    )
    json_mod.loads(replayed["tool_calls"][0]["function"]["arguments"])  # must not raise


def test_terminal_scenarios_v2_attach_seed_state_only() -> None:
    """Terminal v2 adds seed_state; rubric stays as pinned (null — no gold criteria)."""
    from terminal_scenarios_v2 import upgrade

    from wmh.core.types import Action, ActionKind, Observation, Step, Trace

    trace = Trace(
        trace_id="t1",
        steps=[
            Step(
                task="x",
                action=Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": "ls"}),
                observation=Observation(content="a.txt"),
            )
        ],
        metadata={},
    )
    rows = [{"task": "x", "provenance": ["t1"], "category": "Misc", "rubric": None}]
    (v2,) = upgrade(rows, {"t1": trace})
    assert "bash -> a.txt" in v2["seed_state"]["scratchpad"]
    assert v2["rubric"] is None
