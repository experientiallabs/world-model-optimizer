"""Tests for joining harbor trial rewards with recorded token spans."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from llm_waterfall.types import (
    ChatFunctionCall,
    ChatFunctionDefinition,
    ChatMessage,
    ChatTool,
    ChatToolCall,
)
from pydantic import ValidationError

from wmh.distill.rendering import ParsedAssistantMessage
from wmh.distill.tokens import (
    TrialRecord,
    assemble_harbor_trial_records,
    assemble_trial_records,
    load_trial_rollout_spans,
    load_trial_spans,
    read_terminus_stop_reason,
    read_trial_stop_reason,
    reconstruct_conversation,
)
from wmh.harness.scoring import GradedTests, ScoreCell
from wmh.providers.tinker import TokenRecorder, TokenSpan


def _span(call_index: int) -> TokenSpan:
    return TokenSpan(
        call_index=call_index,
        prompt_token_ids=[1, 2, call_index],
        sampled_token_ids=[65, 66],
        sampled_logprobs=[-0.5, -1.5],
    )


_BASH_TOOL = ChatTool(
    function=ChatFunctionDefinition(
        name="bash", description="run bash", parameters={"type": "object"}
    )
)


class _ScriptedParser:
    """A `SampledTurnParser` that parses sampled ids from a canned table.

    Keeps the replay tests independent of any real renderer: the mapping from
    token ids to an assistant turn is exactly what a renderer supplies.
    """

    def __init__(self, table: dict[tuple[int, ...], ParsedAssistantMessage]) -> None:
        self._table = table

    def parse_response(self, sampled_ids: list[int]) -> ParsedAssistantMessage:
        parsed = self._table.get(tuple(sampled_ids))
        assert parsed is not None, f"no scripted parse for sampled ids {sampled_ids}"
        return parsed


def _cell(task_id: str, attempt: int, *, reward: float, artifact_dir: Path) -> ScoreCell:
    return ScoreCell(
        task_id=task_id,
        attempt=attempt,
        reward=reward,
        passed=reward == 1.0,
        artifact_dir=str(artifact_dir),
    )


def _write_trace(trial_dir: Path, payload: str) -> None:
    agent_dir = trial_dir / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "wmh-run.json").write_text(payload, encoding="utf-8")


def test_load_trial_spans_round_trips_the_recorder_sink_format(tmp_path: Path) -> None:
    """The reader is coupled to TokenRecorder's as-built sink: write through it."""
    recorder = TokenRecorder(jsonl_path=tmp_path / "task-a__x1.jsonl")
    recorder.record(_span(0))
    recorder.record(_span(1))

    spans = load_trial_spans(tmp_path, "task-a__x1")

    assert spans == recorder.spans()
    assert [span.call_index for span in spans] == [0, 1]


def test_load_trial_spans_missing_sink_is_empty_not_an_error(tmp_path: Path) -> None:
    assert load_trial_spans(tmp_path, "task-a__never-ran") == []


def test_load_trial_spans_tolerates_blank_lines(tmp_path: Path) -> None:
    sink = tmp_path / "t.jsonl"
    sink.write_text(_span(0).model_dump_json() + "\n\n", encoding="utf-8")
    assert len(load_trial_spans(tmp_path, "t")) == 1


def test_load_trial_spans_corrupt_line_is_actionable(tmp_path: Path) -> None:
    sink = tmp_path / "t.jsonl"
    sink.write_text(_span(0).model_dump_json() + '\n{"call_index": "nope"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"line 2 of .*t\.jsonl"):
        load_trial_spans(tmp_path, "t")
    with pytest.raises(ValueError, match="delete this step's token sink directory"):
        load_trial_spans(tmp_path, "t")


def test_load_trial_spans_rejects_a_sink_appended_by_two_recorders(tmp_path: Path) -> None:
    """A call_index reset means two episodes shared one sink; spans can no longer be
    attributed to the reported trial, so the reader refuses instead of guessing."""
    sink = tmp_path / "t.jsonl"
    lines = [_span(0).model_dump_json(), _span(1).model_dump_json(), _span(0).model_dump_json()]
    sink.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"call_index sequence \[0, 1, 0\]"):
        load_trial_spans(tmp_path, "t")


def test_load_trial_spans_rejects_a_gap_in_the_sequence(tmp_path: Path) -> None:
    sink = tmp_path / "t.jsonl"
    lines = [_span(0).model_dump_json(), _span(2).model_dump_json()]
    sink.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 0..1"):
        load_trial_spans(tmp_path, "t")


def _tool_call() -> ChatToolCall:
    return ChatToolCall(
        id="call_0", function=ChatFunctionCall(name="bash", arguments='{"cmd": "ls"}')
    )


def _tool_use_episode() -> tuple[list[TokenSpan], _ScriptedParser]:
    """Two spans of one tool-using episode, shaped exactly as the provider records them."""
    spans = [
        TokenSpan(
            call_index=0,
            prompt_token_ids=[1, 2, 3],
            sampled_token_ids=[10, 11],
            sampled_logprobs=[-0.1, -0.2],
            delta_start=0,
            delta_messages=[
                ChatMessage(role="system", content="be terse"),
                ChatMessage(role="user", content="list files"),
            ],
            tools=[_BASH_TOOL],
        ),
        TokenSpan(
            call_index=1,
            prompt_token_ids=[1, 2, 3, 10, 11, 4],
            sampled_token_ids=[12, 13],
            sampled_logprobs=[-0.3, -0.4],
            # Only the NEW message: the caller's echo of the assistant turn is
            # the previous span's sampled ids, never repeated here.
            delta_start=3,
            delta_messages=[
                ChatMessage.model_validate(
                    {"role": "tool", "content": "a.txt b.txt", "tool_call_id": "call_0"}
                )
            ],
            tools=[_BASH_TOOL],
        ),
    ]
    parser = _ScriptedParser(
        {
            (10, 11): ParsedAssistantMessage(text="on it", tool_calls=[_tool_call()], stopped=True),
            (12, 13): ParsedAssistantMessage(text="a.txt b.txt is all", stopped=True),
        }
    )
    return spans, parser


def test_load_trial_spans_round_trips_the_canonical_messages_and_tools(tmp_path: Path) -> None:
    """The teacher-facing fields must survive the recorder's jsonl sink verbatim."""
    spans, _ = _tool_use_episode()
    recorder = TokenRecorder(jsonl_path=tmp_path / "task-a__x1.jsonl")
    for span in spans:
        recorder.record(span)

    loaded = load_trial_spans(tmp_path, "task-a__x1")

    assert loaded == spans
    assert loaded[0].delta_messages is not None
    assert [message.role for message in loaded[0].delta_messages] == ["system", "user"]
    assert loaded[1].delta_messages is not None
    assert loaded[1].delta_messages[0].tool_call_id == "call_0"
    assert loaded[1].tools is not None
    assert loaded[1].tools[0].function.name == "bash"


def test_load_trial_spans_still_reads_a_sink_without_the_canonical_messages(
    tmp_path: Path,
) -> None:
    """Real sinks on disk carry only the four original keys; they MUST still load."""
    old_format = [
        {
            "call_index": 0,
            "prompt_token_ids": [1, 2],
            "sampled_token_ids": [65, 66],
            "sampled_logprobs": [-0.5, -1.5],
        },
        {
            "call_index": 1,
            "prompt_token_ids": [1, 2, 65, 66, 7],
            "sampled_token_ids": [67],
            "sampled_logprobs": [-0.25],
        },
    ]
    sink = tmp_path / "t.jsonl"
    sink.write_text("".join(json.dumps(item) + "\n" for item in old_format), encoding="utf-8")

    spans = load_trial_spans(tmp_path, "t")

    assert [span.call_index for span in spans] == [0, 1]
    assert spans[0].sampled_logprobs == [-0.5, -1.5]
    assert all(span.delta_messages is None for span in spans)
    assert all(span.tools == [] for span in spans)


def test_reconstruct_conversation_replays_a_multi_turn_tool_use_episode() -> None:
    spans, parser = _tool_use_episode()

    replay = reconstruct_conversation(spans, parser)

    assert replay is not None
    assert [message.role for message in replay.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert replay.messages[1].content == "list files"
    first_assistant = replay.messages[2]
    assert first_assistant.content == "on it"
    assert first_assistant.tool_calls is not None
    assert first_assistant.tool_calls[0].function.name == "bash"
    assert first_assistant.tool_calls[0].function.arguments == '{"cmd": "ls"}'
    assert replay.messages[3].tool_call_id == "call_0"
    assert replay.messages[4].content == "a.txt b.txt is all"
    assert replay.messages[4].tool_calls is None
    # The planner pairs each sampled span with the message its tokens produced.
    assert replay.assistant_index_by_span == {0: 2, 1: 4}
    assert replay.tools == [_BASH_TOOL]


def test_reconstruct_conversation_returns_none_for_an_old_sink(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An old sink has no canonical messages: degrade honestly, never guess one."""
    sink = tmp_path / "t.jsonl"
    sink.write_text(
        json.dumps(
            {
                "call_index": 0,
                "prompt_token_ids": [1, 2],
                "sampled_token_ids": [10, 11],
                "sampled_logprobs": [-0.5, -1.5],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    spans = load_trial_spans(tmp_path, "t")
    _, parser = _tool_use_episode()

    with caplog.at_level("WARNING", logger="wmh.distill.tokens"):
        assert reconstruct_conversation(spans, parser) is None

    assert any("carries no delta_messages" in record.message for record in caplog.records)
    assert any("Re-run the rollout step" in record.message for record in caplog.records)


def test_reconstruct_conversation_returns_none_without_spans() -> None:
    _, parser = _tool_use_episode()
    assert reconstruct_conversation([], parser) is None


def test_reconstruct_conversation_returns_none_when_spans_disagree_about_tools(
    caplog: pytest.LogCaptureFixture,
) -> None:
    spans, parser = _tool_use_episode()
    spans[1].tools = []

    with caplog.at_level("WARNING", logger="wmh.distill.tokens"):
        assert reconstruct_conversation(spans, parser) is None

    assert any("different tool schemas" in record.message for record in caplog.records)


def test_read_trial_stop_reason_full_and_partial_traces(tmp_path: Path) -> None:
    full = tmp_path / "task-a__x1"
    _write_trace(
        full,
        json.dumps({"task_id": "t", "steps": [], "stop_reason": "submitted", "turns": 1}),
    )
    partial = tmp_path / "task-a__x2"
    _write_trace(
        partial,
        json.dumps({"stop_reason": "cancelled-by-harbor-timeout", "partial": True}),
    )
    assert read_trial_stop_reason(full) == "submitted"
    assert read_trial_stop_reason(partial) == "cancelled-by-harbor-timeout"


def test_read_trial_stop_reason_falls_back_to_the_trial_root(tmp_path: Path) -> None:
    trial = tmp_path / "task-a__x3"
    trial.mkdir()
    (trial / "wmh-run.json").write_text(json.dumps({"stop_reason": "max_turns"}), encoding="utf-8")
    assert read_trial_stop_reason(trial) == "max_turns"


def test_read_trial_stop_reason_missing_or_unreadable_is_none(tmp_path: Path) -> None:
    missing = tmp_path / "task-a__gone"
    assert read_trial_stop_reason(missing) is None

    unreadable = tmp_path / "task-a__bad"
    _write_trace(unreadable, "{not json")
    assert read_trial_stop_reason(unreadable) is None

    non_string = tmp_path / "task-a__odd"
    _write_trace(non_string, json.dumps({"stop_reason": 7}))
    assert read_trial_stop_reason(non_string) is None


def test_assemble_joins_spans_and_stop_reasons_by_trial_name(tmp_path: Path) -> None:
    sink_dir = tmp_path / "tokens"
    sink_dir.mkdir()
    trials_dir = tmp_path / "job"

    solved = trials_dir / "task-a__s1"
    _write_trace(solved, json.dumps({"stop_reason": "submitted"}))
    recorder = TokenRecorder(jsonl_path=sink_dir / "task-a__s1.jsonl")
    recorder.record(_span(0))
    recorder.record(_span(1))

    # This trial died before its first completion: no sink file, no trace.
    dead = trials_dir / "task-b__d1"
    dead.mkdir(parents=True)

    cells = [
        _cell("task-a", 1, reward=1.0, artifact_dir=solved),
        _cell("task-b", 1, reward=0.0, artifact_dir=dead),
    ]

    records = assemble_trial_records(cells, sink_dir)

    assert [record.trial_name for record in records] == ["task-a__s1", "task-b__d1"]
    assert records[0].task_id == "task-a"
    assert records[0].attempt == 1
    assert records[0].reward == 1.0
    assert records[0].passed is True
    assert records[0].spans == recorder.spans()
    assert records[0].stop_reason == "submitted"
    assert records[0].artifact_dir == str(solved)
    # The span-less trial is recorded, not dropped: its reward is real signal.
    assert records[1].spans == []
    assert records[1].stop_reason is None
    assert records[1].passed is False


def test_assemble_accepts_an_injected_stop_reason_reader(tmp_path: Path) -> None:
    seen: list[Path] = []

    def reader(artifact_dir: Path) -> str | None:
        seen.append(artifact_dir)
        return "custom"

    trial = tmp_path / "job" / "task-a__s1"
    records = assemble_trial_records(
        [_cell("task-a", 1, reward=0.0, artifact_dir=trial)],
        tmp_path / "tokens",
        read_stop_reason=reader,
    )
    assert records[0].stop_reason == "custom"
    assert seen == [trial]


def test_assemble_rejects_cells_without_an_artifact_dir(tmp_path: Path) -> None:
    cell = ScoreCell(task_id="task-a", attempt=1, reward=0.0, passed=False, artifact_dir="")
    with pytest.raises(ValueError, match="carries no artifact dir"):
        assemble_trial_records([cell], tmp_path)


def test_assemble_carries_the_cells_graded_test_breakdown(tmp_path: Path) -> None:
    graded = tmp_path / "job" / "task-a__s1"
    ungraded = tmp_path / "job" / "task-b__s1"
    graded.mkdir(parents=True)
    ungraded.mkdir(parents=True)
    cells = [
        _cell("task-a", 1, reward=0.0, artifact_dir=graded).model_copy(
            update={"tests": GradedTests(passed=1, resolved=2)}
        ),
        _cell("task-b", 1, reward=0.0, artifact_dir=ungraded),
    ]

    records = assemble_trial_records(cells, tmp_path / "tokens")

    assert records[0].tests == GradedTests(passed=1, resolved=2)
    assert records[0].graded_score == 0.5
    assert records[0].reward == 0.0  # the binary reward rides along unchanged
    # No report on the cell means no graded score on the record: absent, not zero.
    assert records[1].tests is None
    assert records[1].graded_score is None


def test_a_record_written_before_the_graded_field_still_loads() -> None:
    """The in-flight-run contract: an artifact on disk predates `tests` and must still validate."""
    restored = TrialRecord.model_validate_json(
        '{"task_id": "task-a", "attempt": 1, "trial_name": "task-a__s1", "reward": 1.0, '
        '"passed": true, "spans": [], "stop_reason": "submitted", "infra_failed": false, '
        '"artifact_dir": "/tmp/job/task-a__s1"}'
    )

    assert restored.tests is None
    assert restored.graded_score is None
    assert restored.passed is True


def test_trial_record_validation() -> None:
    record = TrialRecord(
        task_id="task-a",
        attempt=1,
        trial_name="task-a__s1",
        reward=1.0,
        passed=True,
        artifact_dir="/tmp/job/task-a__s1",
    )
    assert record.spans == []
    assert record.stop_reason is None
    with pytest.raises(ValidationError):
        TrialRecord(
            task_id="task-a",
            attempt=0,
            trial_name="task-a__s1",
            reward=1.0,
            passed=True,
            artifact_dir="x",
        )
    with pytest.raises(ValidationError):
        TrialRecord(
            task_id="task-a",
            attempt=1,
            trial_name="task-a__s1",
            reward=1.5,
            passed=True,
            artifact_dir="x",
        )


# --- harbor terminus-2 rollout details (the distillation span source) ----------------------------


def _write_result(trial_dir: Path, payload: dict[str, object]) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")


def _rollout_detail(turns: int) -> dict[str, object]:
    return {
        "prompt_token_ids": [[1, 2, index] for index in range(turns)],
        "completion_token_ids": [[65, 66] for _ in range(turns)],
        "logprobs": [[-0.5, -1.5] for _ in range(turns)],
    }


def _trajectory(*, complete: bool) -> dict[str, object]:
    tool_calls = (
        [{"tool_call_id": "c1", "function_name": "mark_task_complete", "arguments": {}}]
        if complete
        else [{"tool_call_id": "c1", "function_name": "bash_command", "arguments": {}}]
    )
    return {
        "steps": [
            {"step_id": 1, "source": "user"},
            {"step_id": 2, "source": "agent", "tool_calls": tool_calls},
        ]
    }


def test_load_trial_rollout_spans_reads_harbors_per_turn_token_ids(tmp_path: Path) -> None:
    """The ids the sampler issued reach training verbatim, straight out of result.json."""
    trial = tmp_path / "task-a__s1"
    _write_result(trial, {"agent_result": {"rollout_details": [_rollout_detail(3)]}})

    spans = load_trial_rollout_spans(trial)

    assert [span.call_index for span in spans] == [0, 1, 2]
    assert [span.prompt_token_ids for span in spans] == [[1, 2, 0], [1, 2, 1], [1, 2, 2]]
    assert all(span.sampled_token_ids == [65, 66] for span in spans)
    assert all(span.sampled_logprobs == [-0.5, -1.5] for span in spans)


def test_load_trial_rollout_spans_uses_only_the_main_agent_segment(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Subagent segments are separate conversations; their prompts extend nothing."""
    trial = tmp_path / "task-a__s1"
    _write_result(
        trial,
        {"agent_result": {"rollout_details": [_rollout_detail(2), _rollout_detail(1)]}},
    )

    with caplog.at_level("WARNING"):
        spans = load_trial_rollout_spans(trial)

    assert len(spans) == 2
    assert "Summarization must stay OFF" in caplog.text


def test_load_trial_rollout_spans_refuses_to_pair_desynchronized_turn_lists(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A turn that returned no completion desynchronizes harbor's three per-turn lists.

    Nothing can realign them afterwards, so the trial contributes no training data at all
    rather than training the wrong prompts against the wrong completions.
    """
    trial = tmp_path / "task-a__s1"
    detail = _rollout_detail(3)
    detail["completion_token_ids"] = [[65, 66], [65, 66]]
    _write_result(trial, {"agent_result": {"rollout_details": [detail]}})

    with caplog.at_level("WARNING"):
        assert load_trial_rollout_spans(trial) == []
    assert "cannot be aligned" in caplog.text


def test_load_trial_rollout_spans_refuses_logprobs_that_do_not_match_their_tokens(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    trial = tmp_path / "task-a__s1"
    detail = _rollout_detail(1)
    detail["logprobs"] = [[-0.5]]
    _write_result(trial, {"agent_result": {"rollout_details": [detail]}})

    with caplog.at_level("WARNING"):
        assert load_trial_rollout_spans(trial) == []
    assert "no training data" in caplog.text


def test_load_trial_rollout_spans_missing_or_empty_evidence_is_empty_not_an_error(
    tmp_path: Path,
) -> None:
    assert load_trial_rollout_spans(tmp_path / "never-ran") == []
    unreadable = tmp_path / "task-a__s1"
    unreadable.mkdir()
    (unreadable / "result.json").write_text("{not json", encoding="utf-8")
    assert load_trial_rollout_spans(unreadable) == []
    no_details = tmp_path / "task-b__s1"
    _write_result(no_details, {"agent_result": {"rollout_details": None}})
    assert load_trial_rollout_spans(no_details) == []


def test_read_terminus_stop_reason_maps_the_recorded_exception(tmp_path: Path) -> None:
    timed_out = tmp_path / "task-a__s1"
    _write_result(
        timed_out,
        {
            "exception_info": {"exception_type": "AgentTimeoutError"},
            "agent_result": {"metadata": {"n_episodes": 4}},
        },
    )
    assert read_terminus_stop_reason(timed_out, max_turns=100) == "budget"

    overflowed = tmp_path / "task-b__s1"
    _write_result(overflowed, {"exception_info": {"exception_type": "ContextLengthExceededError"}})
    assert read_terminus_stop_reason(overflowed, max_turns=100) == "provider_error"

    other = tmp_path / "task-c__s1"
    _write_result(other, {"exception_info": {"exception_type": "EnvironmentStartTimeoutError"}})
    assert read_terminus_stop_reason(other, max_turns=100) == "error"


def test_read_terminus_stop_reason_reads_completion_from_the_trajectory(tmp_path: Path) -> None:
    trial = tmp_path / "task-a__s1"
    _write_result(trial, {"agent_result": {"metadata": {"n_episodes": 7}}})
    (trial / "agent").mkdir()
    (trial / "agent" / "trajectory.json").write_text(
        json.dumps(_trajectory(complete=True)), encoding="utf-8"
    )
    assert read_terminus_stop_reason(trial, max_turns=100) == "submitted"

    (trial / "agent" / "trajectory.json").write_text(
        json.dumps(_trajectory(complete=False)), encoding="utf-8"
    )
    # Ran, never claimed completion, never hit the cap: terminus-2 only leaves that loop when
    # its tmux session dies, which is a harness failure rather than a task verdict.
    assert read_terminus_stop_reason(trial, max_turns=100) == "error"


def test_read_terminus_stop_reason_prefers_the_turn_cap_at_the_boundary(tmp_path: Path) -> None:
    """A completion claim on the last allowed turn still reads as a scaffold loss."""
    trial = tmp_path / "task-a__s1"
    _write_result(trial, {"agent_result": {"metadata": {"n_episodes": 10}}})
    (trial / "agent").mkdir()
    (trial / "agent" / "trajectory.json").write_text(
        json.dumps(_trajectory(complete=True)), encoding="utf-8"
    )
    assert read_terminus_stop_reason(trial, max_turns=10) == "max_turns"


def test_read_terminus_stop_reason_is_none_without_readable_evidence(tmp_path: Path) -> None:
    assert read_terminus_stop_reason(tmp_path / "never-ran", max_turns=100) is None
    no_episodes = tmp_path / "task-a__s1"
    _write_result(no_episodes, {"agent_result": {"metadata": {}}})
    assert read_terminus_stop_reason(no_episodes, max_turns=100) is None


def test_assemble_harbor_trial_records_joins_rewards_with_harbor_spans(tmp_path: Path) -> None:
    solved = tmp_path / "job" / "task-a__s1"
    _write_result(
        solved,
        {
            "agent_result": {
                "rollout_details": [_rollout_detail(2)],
                "metadata": {"n_episodes": 2},
            }
        },
    )
    (solved / "agent").mkdir()
    (solved / "agent" / "trajectory.json").write_text(
        json.dumps(_trajectory(complete=True)), encoding="utf-8"
    )
    dead = tmp_path / "job" / "task-b__d1"
    dead.mkdir(parents=True)

    records = assemble_harbor_trial_records(
        [
            _cell("task-a", 1, reward=1.0, artifact_dir=solved),
            _cell("task-b", 1, reward=0.0, artifact_dir=dead),
        ],
        max_turns=100,
    )

    assert [record.trial_name for record in records] == ["task-a__s1", "task-b__d1"]
    assert len(records[0].spans) == 2
    assert records[0].stop_reason == "submitted"
    assert records[0].artifact_dir == str(solved)
    # A trial that left no evidence is kept with empty spans, never dropped.
    assert records[1].spans == []
    assert records[1].stop_reason is None
