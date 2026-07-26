"""Tests for the Tinker provider: span recording, response shape, lazy imports.

Everything runs against the deterministic fakes in `wmh.distill.fake_tinker`
plus a minimal char-level `ChatRendering`; the real tinker SDK is never
touched (several tests pin that by poisoning `sys.modules`).
"""

from __future__ import annotations

import ast
import json
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, NoReturn, cast

import pytest
from llm_waterfall.types import (
    ChatFunctionCall,
    ChatFunctionDefinition,
    ChatMessage,
    ChatRequest,
    ChatTool,
    ChatToolCall,
)
from pydantic import JsonValue, ValidationError

import wmh.distill.rendering as rendering_module
import wmh.providers.tinker as tinker_module
from wmh.config.config import PROVIDER_ENV_VARS
from wmh.distill.config import (
    DistillConfig,
    HarborConfig,
    RolloutConfig,
    StudentConfig,
    TeacherConfig,
    TrainConfig,
)
from wmh.distill.data import build_datums
from wmh.distill.deadlines import TinkerDeadlineError
from wmh.distill.fake_tinker import FakeSampledSequence, FakeSamplingClient, FakeTokenizer
from wmh.distill.rendering import ParsedAssistantMessage
from wmh.distill.tokens import TrialRecord, reconstruct_conversation
from wmh.providers.base import (
    UNPARSED_TOOL_CALLS_KEY,
    ContextWindowProvider,
    Message,
    Provider,
    ProviderConfig,
    ProviderKind,
    ToolCallingProvider,
)
from wmh.providers.registry import get_provider
from wmh.providers.retry import wrap_provider_with_retries
from wmh.providers.tinker import (
    TINKER_API_KEY_ENV,
    SdkSampler,
    TinkerChatProvider,
    TinkerSampler,
    TokenRecorder,
    TokenSpan,
)

if TYPE_CHECKING:
    import tinker


class _MiniRendering:
    """Minimal char-level ChatRendering that drives the provider without the cookbook."""

    def __init__(self) -> None:
        self._tok = FakeTokenizer()

    @property
    def stop_sequences(self) -> list[str] | list[int]:
        # Newline: outside the fake sampler's printable-ASCII token range, so
        # deterministic samples always run to max_tokens.
        return [ord("\n")]

    def build_generation_prompt(
        self, messages: list[ChatMessage], tools: list[ChatTool] | None = None
    ) -> list[int]:
        lines: list[str] = []
        if tools:
            lines.append("tools: " + ",".join(tool.function.name for tool in tools))
        for message in messages:
            content = message.content if isinstance(message.content, str) else ""
            lines.append(f"{message.role}: {content}")
        lines.append("assistant:")
        return self._tok.encode("\n".join(lines))

    def render_suffix(
        self,
        messages: list[ChatMessage],
        delta_start: int,
        tools: list[ChatTool] | None = None,
        *,
        previous_sampled_ids: list[int],
    ) -> list[int]:
        del tools, previous_sampled_ids
        lines = [""]
        for message in messages[delta_start:]:
            content = message.content if isinstance(message.content, str) else ""
            lines.append(f"{message.role}: {content}")
        lines.append("assistant:")
        return self._tok.encode("\n".join(lines))

    def decode(self, token_ids: list[int]) -> str:
        return self._tok.decode(token_ids)

    def decode_with_specials(self, token_ids: list[int]) -> str:
        # The char-level fake has no special tokens to strip, so the raw
        # decode IS the specials-preserving decode.
        return self._tok.decode(token_ids)

    def parse_response(self, sampled_ids: list[int]) -> ParsedAssistantMessage:
        return ParsedAssistantMessage(
            text=self._tok.decode(sampled_ids), tool_calls=[], stopped=False
        )


class _ToolCallRendering(_MiniRendering):
    """Parses every sample into one fixed tool call (tool-call shape tests)."""

    def parse_response(self, sampled_ids: list[int]) -> ParsedAssistantMessage:
        del sampled_ids
        call = ChatToolCall(
            id="call_0", function=ChatFunctionCall(name="bash", arguments='{"cmd": "ls"}')
        )
        return ParsedAssistantMessage(text="", tool_calls=[call], stopped=True)


class _FlakySampler:
    """Raises on the first sample() calls, then delegates to a fake sampler."""

    def __init__(self, inner: FakeSamplingClient, failures: int = 1) -> None:
        self._inner = inner
        self._failures = failures

    def sample(
        self,
        prompt_token_ids: list[int],
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str] | list[int] | None = None,
    ) -> FakeSampledSequence:
        if self._failures > 0:
            self._failures -= 1
            raise RuntimeError("simulated sampler outage")
        return self._inner.sample(
            prompt_token_ids, max_tokens=max_tokens, temperature=temperature, stop=stop
        )


def _config() -> ProviderConfig:
    return ProviderConfig(
        kind=ProviderKind.TINKER,
        model_type="Qwen/Qwen3-8B",
        model="tinker://run/weights/0",
    )


def _request(max_tokens: int = 16) -> ChatRequest:
    return ChatRequest(
        messages=[
            ChatMessage(role="system", content="be terse"),
            ChatMessage(role="user", content="hi"),
        ],
        temperature=0.7,
        max_tokens=max_tokens,
    )


def _as_dict(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, dict)
    return value


def _as_list(value: JsonValue) -> list[JsonValue]:
    assert isinstance(value, list)
    return value


def _span(call_index: int = 0) -> TokenSpan:
    return TokenSpan(
        call_index=call_index,
        prompt_token_ids=[1, 2],
        sampled_token_ids=[65, 66],
        sampled_logprobs=[-0.5, -1.5],
    )


def test_token_span_requires_aligned_logprobs() -> None:
    with pytest.raises(ValidationError, match="does not match"):
        TokenSpan(
            call_index=0,
            prompt_token_ids=[1],
            sampled_token_ids=[2, 3],
            sampled_logprobs=[-0.1],
        )


def test_token_span_canonical_message_fields_default_for_old_sinks() -> None:
    """Spans on disk from earlier runs carry only the four original keys."""
    span = TokenSpan.model_validate_json(
        json.dumps(
            {
                "call_index": 3,
                "prompt_token_ids": [1],
                "sampled_token_ids": [2],
                "sampled_logprobs": [-0.5],
            }
        )
    )
    assert span.delta_start is None
    assert span.delta_messages is None
    assert span.tools == []


def test_token_span_round_trips_the_canonical_message_fields() -> None:
    tool = ChatTool(
        function=ChatFunctionDefinition(
            name="bash", description="run bash", parameters={"type": "object"}
        )
    )
    span = TokenSpan(
        call_index=0,
        prompt_token_ids=[1],
        sampled_token_ids=[2],
        sampled_logprobs=[-0.5],
        delta_start=0,
        delta_messages=[
            ChatMessage(role="user", content="hi"),
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ChatToolCall(
                        id="call_0",
                        function=ChatFunctionCall(name="bash", arguments='{"cmd": "ls"}'),
                    )
                ],
            ),
        ],
        tools=[tool],
    )
    restored = TokenSpan.model_validate_json(span.model_dump_json())
    assert restored == span
    assert restored.delta_messages is not None
    assert restored.delta_messages[1].tool_calls is not None
    assert restored.delta_messages[1].tool_calls[0].function.arguments == '{"cmd": "ls"}'
    assert restored.tools == [tool]


def test_recorder_snapshot_is_a_copy() -> None:
    recorder = TokenRecorder()
    recorder.record(_span())
    snapshot = recorder.spans()
    snapshot.clear()
    assert len(recorder) == 1
    assert recorder.spans()[0].call_index == 0


def test_recorder_jsonl_sink_written_incrementally(tmp_path: Path) -> None:
    sink = tmp_path / "spans.jsonl"
    recorder = TokenRecorder(jsonl_path=sink)
    recorder.record(_span(0))
    assert len(sink.read_text(encoding="utf-8").splitlines()) == 1
    recorder.record(_span(1))
    lines = sink.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert [item["call_index"] for item in parsed] == [0, 1]
    assert parsed[0]["sampled_token_ids"] == [65, 66]
    assert parsed[0]["sampled_logprobs"] == [-0.5, -1.5]


def test_complete_chat_shape_and_span_matches_issued_sample() -> None:
    fake = FakeSamplingClient(seed="student-v0")
    recorder = TokenRecorder()
    provider = TinkerChatProvider(
        _config(), sampling_client=fake, renderer=_MiniRendering(), recorder=recorder
    )

    response = provider.complete_chat(_request(max_tokens=16))

    choice = response.choices[0]
    assert choice.message.role == "assistant"
    assert isinstance(choice.message.content, str)
    assert choice.message.content
    assert choice.message.tool_calls is None
    # _MiniRendering never reports a stop signal, so truncation reads as length.
    assert choice.finish_reason == "length"
    expected_prompt = _MiniRendering().build_generation_prompt(_request().messages)
    assert response.usage is not None
    assert response.usage.prompt_tokens == len(expected_prompt)
    assert response.usage.completion_tokens == 16
    assert response.model == "tinker://run/weights/0"

    # TITO: the recorded span is byte-identical to what the sampler issued.
    assert len(recorder) == 1
    span = recorder.spans()[0]
    issued = fake.issued[0]
    assert span.call_index == 0
    assert span.prompt_token_ids == list(issued.prompt_ids) == expected_prompt
    assert span.sampled_token_ids == list(issued.sampled_ids)
    assert span.sampled_logprobs == list(issued.logprobs)

    wire = response.wire_payload()
    wire_message = _as_dict(_as_dict(_as_list(wire["choices"])[0])["message"])
    assert wire_message["role"] == "assistant"
    assert _as_dict(wire["usage"])["completion_tokens"] == 16


def test_complete_chat_tool_calls_in_openai_format() -> None:
    provider = TinkerChatProvider(
        _config(), sampling_client=FakeSamplingClient(seed="s"), renderer=_ToolCallRendering()
    )
    response = provider.complete_chat(_request())
    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.tool_calls is not None
    assert choice.message.tool_calls[0].function.name == "bash"

    wire = response.wire_payload()
    wire_message = _as_dict(_as_dict(_as_list(wire["choices"])[0])["message"])
    wire_call = _as_list(wire_message["tool_calls"])[0]
    assert wire_call == {
        "id": "call_0",
        "type": "function",
        "function": {"name": "bash", "arguments": '{"cmd": "ls"}'},
    }
    # Empty text serializes as an absent content key, like OpenAI's null content.
    assert "content" not in wire_message


def test_tool_choice_none_renders_without_tool_schemas() -> None:
    recorder = TokenRecorder()
    rendering = _MiniRendering()
    provider = TinkerChatProvider(
        _config(),
        sampling_client=FakeSamplingClient(seed="s"),
        renderer=rendering,
        recorder=recorder,
    )
    request = _request()
    request.tools = [
        ChatTool(
            function=ChatFunctionDefinition(
                name="bash", description="run bash", parameters={"type": "object"}
            )
        )
    ]
    request.tool_choice = "none"
    provider.complete_chat(request)
    prompt_text = rendering.decode(recorder.spans()[0].prompt_token_ids)
    assert "tools:" not in prompt_text
    # The span records the schemas actually rendered, so a teacher re-render
    # sees the same (empty) tool surface the student saw.
    assert recorder.spans()[0].tools == []


@pytest.mark.parametrize("choice", ["required", {"type": "function", "function": {"name": "bash"}}])
def test_unsupported_tool_choice_raises_actionable_error(choice: JsonValue) -> None:
    provider = TinkerChatProvider(
        _config(), sampling_client=FakeSamplingClient(seed="s"), renderer=_MiniRendering()
    )
    request = _request()
    request.tool_choice = choice
    with pytest.raises(ValueError, match="tool_choice"):
        provider.complete_chat(request)


def test_span_recorded_once_per_success_and_not_on_failure() -> None:
    recorder = TokenRecorder()
    provider = TinkerChatProvider(
        _config(),
        sampling_client=_FlakySampler(FakeSamplingClient(seed="s"), failures=1),
        renderer=_MiniRendering(),
        recorder=recorder,
    )
    # First attempt fails mid-sampling: an outer retry wrapper would re-invoke
    # complete_chat, and the failed attempt must not leave a span behind.
    with pytest.raises(RuntimeError, match="outage"):
        provider.complete_chat(_request())
    assert len(recorder) == 0

    provider.complete_chat(_request())
    assert [span.call_index for span in recorder.spans()] == [0]

    provider.complete_chat(_request())
    assert [span.call_index for span in recorder.spans()] == [0, 1]


class _FramedRendering(_MiniRendering):
    """Scripted renderer with explicit per-message framing and suffix rendering.

    Each message renders as `<role>content|calls=name:args</>`, the generation
    header is `<assistant>`, and `</>` is the end-of-turn framing. Tool-call
    arguments render VERBATIM, so a caller that echoes an assistant turn with
    reformatted JSON spacing changes the re-rendered tokens, exactly the live
    defect the incremental prompt construction exists to absorb.
    """

    def _segment(self, message: ChatMessage) -> str:
        content = message.content if isinstance(message.content, str) else ""
        calls = ""
        if message.tool_calls:
            calls = "|calls=" + ";".join(
                f"{call.function.name}:{call.function.arguments}" for call in message.tool_calls
            )
        return f"<{message.role}>{content}{calls}</>"

    def build_generation_prompt(
        self, messages: list[ChatMessage], tools: list[ChatTool] | None = None
    ) -> list[int]:
        prefix = ""
        if tools:
            prefix = "<tools>" + ",".join(tool.function.name for tool in tools) + "</>"
        body = "".join(self._segment(message) for message in messages)
        return self._tok.encode(prefix + body + "<assistant>")

    def render_suffix(
        self,
        messages: list[ChatMessage],
        delta_start: int,
        tools: list[ChatTool] | None = None,
        *,
        previous_sampled_ids: list[int],
    ) -> list[int]:
        del tools
        end_of_turn = self._tok.encode("</>")
        tokens: list[int] = []
        if previous_sampled_ids[-len(end_of_turn) :] != end_of_turn:
            tokens.extend(end_of_turn)
        for message in messages[delta_start:]:
            tokens.extend(self._tok.encode(self._segment(message)))
        tokens.extend(self._tok.encode("<assistant>"))
        return tokens

    def parse_response(self, sampled_ids: list[int]) -> ParsedAssistantMessage:
        text = self._tok.decode(sampled_ids)
        stopped = text.endswith("</>")
        return ParsedAssistantMessage(text=text.removesuffix("</>"), tool_calls=[], stopped=stopped)


class _ScriptedSampler:
    """Returns canned token sequences in order, recording every prompt."""

    def __init__(self, texts: list[str]) -> None:
        self._tok = FakeTokenizer()
        self._outputs = [self._tok.encode(text) for text in texts]
        self.prompts: list[list[int]] = []

    def sample(
        self,
        prompt_token_ids: list[int],
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str] | list[int] | None = None,
    ) -> FakeSampledSequence:
        del max_tokens, temperature, stop
        self.prompts.append(list(prompt_token_ids))
        tokens = self._outputs.pop(0)
        return FakeSampledSequence(tokens=tokens, logprobs=[-0.5] * len(tokens), stop_reason="stop")


def _distill_cfg() -> DistillConfig:
    return DistillConfig(
        student=StudentConfig(base_model="Qwen/Qwen3-8B"),
        teacher=TeacherConfig(model="Qwen/Qwen3-32B"),
        harbor=HarborConfig(job_template="job.yaml"),
        rollout=RolloutConfig(),
        train=TrainConfig(),
    )


def _trial(recorder: TokenRecorder) -> TrialRecord:
    return TrialRecord(
        task_id="task-a",
        attempt=1,
        trial_name="task-a__x1",
        reward=1.0,
        passed=True,
        spans=recorder.spans(),
        stop_reason="submitted",
        artifact_dir="/tmp/jobs/task-a__x1",
    )


def _framed_provider(texts: list[str]) -> tuple[TinkerChatProvider, TokenRecorder]:
    recorder = TokenRecorder()
    provider = TinkerChatProvider(
        _config(),
        sampling_client=_ScriptedSampler(texts),
        renderer=_FramedRendering(),
        recorder=recorder,
    )
    return provider, recorder


def _chat(provider: TinkerChatProvider, messages: list[ChatMessage]) -> ChatMessage:
    response = provider.complete_chat(
        ChatRequest(messages=messages, temperature=0.0, max_tokens=64)
    )
    return response.choices[0].message


_HISTORY = [
    ChatMessage(role="system", content="be terse"),
    ChatMessage(role="user", content="list files"),
]


def _echo(arguments: str) -> ChatMessage:
    return ChatMessage(
        role="assistant",
        content="ok",
        tool_calls=[
            ChatToolCall(id="call_0", function=ChatFunctionCall(name="bash", arguments=arguments))
        ],
    )


def _tool_result(content: str) -> ChatMessage:
    return ChatMessage.model_validate(
        {"role": "tool", "content": content, "tool_call_id": "call_0"}
    )


def test_multi_turn_verbatim_echo_prompts_extend_and_merge() -> None:
    # Regression: when the caller echoes the assistant turn verbatim, prompts
    # are prefix-extending AND identical to a full re-render, validating that
    # the suffix composition matches full-render framing.
    rendering = _FramedRendering()
    provider, recorder = _framed_provider(['ok|calls=bash:{"cmd": "ls"}</>', "done</>"])
    _chat(provider, _HISTORY)
    extended = [*_HISTORY, _echo('{"cmd": "ls"}'), _tool_result("a.txt b.txt")]
    _chat(provider, extended)

    first, second = recorder.spans()
    episode = first.prompt_token_ids + first.sampled_token_ids
    assert second.prompt_token_ids[: len(episode)] == episode
    assert second.prompt_token_ids == rendering.build_generation_prompt(extended)
    datums, stats = build_datums([_trial(recorder)], _distill_cfg())
    assert len(datums) == 1
    assert stats.fragments == 0
    assert recorder.fallback_count == 0


def test_reformatted_assistant_echo_still_extends_and_merges() -> None:
    # The live defect: the agent re-serializes the assistant turn (different
    # JSON spacing in tool_calls), so a full re-render would NOT extend the
    # sampled tokens; the incremental prompt must still extend and merge.
    rendering = _FramedRendering()
    provider, recorder = _framed_provider(['ok|calls=bash:{"cmd": "ls"}</>', "done</>"])
    _chat(provider, _HISTORY)
    extended = [*_HISTORY, _echo('{"cmd":"ls"}'), _tool_result("a.txt b.txt")]
    _chat(provider, extended)

    first, second = recorder.spans()
    episode = first.prompt_token_ids + first.sampled_token_ids
    # The defect is real in this scripted world: a full re-render diverges.
    assert rendering.build_generation_prompt(extended)[: len(episode)] != episode
    # The fix: the incrementally built prompt still extends the token history.
    assert second.prompt_token_ids[: len(episode)] == episode
    datums, stats = build_datums([_trial(recorder)], _distill_cfg())
    assert len(datums) == 1
    assert stats.fragments == 0
    assert stats.fragmentation_rate == 0.0
    assert recorder.fallback_count == 0


def test_genuine_history_edit_falls_back_and_fragments(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rendering = _FramedRendering()
    provider, recorder = _framed_provider(["a</>", "b</>", "c</>"])
    _chat(provider, _HISTORY)
    second_messages = [*_HISTORY, _echo('{"cmd":"ls"}'), _tool_result("a.txt")]
    _chat(provider, second_messages)
    # A changed tool-result message mid-history is a genuine edit: fall back.
    edited = [
        *_HISTORY,
        _echo('{"cmd":"ls"}'),
        _tool_result("EDITED"),
        ChatMessage(role="assistant", content="b"),
        _tool_result("more"),
    ]
    with caplog.at_level("INFO", logger="wmh.providers.tinker"):
        _chat(provider, edited)

    assert recorder.fallback_count == 1
    assert any("incoming message 3" in record.message for record in caplog.records)
    spans = recorder.spans()
    # The fallback prompt is a correct full render of the edited history.
    assert spans[2].prompt_token_ids == rendering.build_generation_prompt(edited)
    # A re-rendered prompt is not a delta over the previous call, so the span
    # records no canonical messages and replay degrades instead of lying.
    assert spans[2].delta_messages is None
    assert reconstruct_conversation(spans, rendering) is None
    datums, stats = build_datums([_trial(recorder)], _distill_cfg())
    assert len(datums) == 2
    assert stats.fragments == 1


def test_max_tokens_truncation_gets_end_of_turn_framing() -> None:
    tok = FakeTokenizer()
    provider, recorder = _framed_provider(["par", "done</>"])
    first_response = _chat(provider, _HISTORY)
    assert first_response.content == "par"
    extended = [
        *_HISTORY,
        ChatMessage(role="assistant", content="par"),
        ChatMessage(role="user", content="continue"),
    ]
    _chat(provider, extended)

    first, second = recorder.spans()
    episode = first.prompt_token_ids + first.sampled_token_ids
    assert second.prompt_token_ids[: len(episode)] == episode
    # The suffix supplies the missing end-of-turn framing before the new message.
    expected_suffix = tok.encode("</>" + "<user>continue</>" + "<assistant>")
    assert second.prompt_token_ids[len(episode) :] == expected_suffix
    datums, stats = build_datums([_trial(recorder)], _distill_cfg())
    assert len(datums) == 1
    assert stats.fragments == 0


def test_spans_record_the_canonical_message_delta_not_the_full_history() -> None:
    """TB2 blocker: the sink must carry the messages a cross-tokenizer teacher
    re-renders, and only the DELTA (full history per call is quadratic in text)."""
    provider, recorder = _framed_provider(['ok|calls=bash:{"cmd": "ls"}</>', "done</>"])
    _chat(provider, _HISTORY)
    extended = [*_HISTORY, _echo('{"cmd": "ls"}'), _tool_result("a.txt b.txt")]
    _chat(provider, extended)

    first, second = recorder.spans()
    assert first.delta_messages == _HISTORY
    # The caller's assistant echo is NOT in the delta: the previous span's
    # sampled ids are that turn's ground truth.
    assert second.delta_messages == [_tool_result("a.txt b.txt")]


def test_message_delta_boundary_matches_the_token_prefix_boundary() -> None:
    """The recorded delta must start exactly where the token prefix stops; a
    disagreement would silently mis-describe the sampled conversation."""
    rendering = _FramedRendering()
    provider, recorder = _framed_provider(['ok|calls=bash:{"cmd": "ls"}</>', "done</>"])
    _chat(provider, _HISTORY)
    extended = [*_HISTORY, _echo('{"cmd":"ls"}'), _tool_result("a.txt b.txt")]
    _chat(provider, extended)

    first, second = recorder.spans()
    assert second.delta_messages is not None
    delta_start = len(extended) - len(second.delta_messages)
    assert second.delta_messages == extended[delta_start:]
    episode = first.prompt_token_ids + first.sampled_token_ids
    assert second.prompt_token_ids[: len(episode)] == episode
    # The token suffix is exactly the render of the messages from that boundary.
    assert second.prompt_token_ids[len(episode) :] == rendering.render_suffix(
        extended, delta_start, None, previous_sampled_ids=first.sampled_token_ids
    )


def test_recorded_spans_replay_the_episode_conversation() -> None:
    """End to end: recorder sink -> reconstruct_conversation -> the real conversation."""
    provider, recorder = _framed_provider(['ok|calls=bash:{"cmd": "ls"}</>', "done</>"])
    first_reply = _chat(provider, _HISTORY)
    extended = [*_HISTORY, _echo('{"cmd":"ls"}'), _tool_result("a.txt b.txt")]
    second_reply = _chat(provider, extended)

    replay = reconstruct_conversation(recorder.spans(), _FramedRendering())

    assert replay is not None
    assert [message.role for message in replay.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert replay.messages[:2] == _HISTORY
    assert replay.messages[3] == _tool_result("a.txt b.txt")
    assert replay.assistant_index_by_span == {0: 2, 1: 4}
    # The replayed assistant turns are the turns the provider actually served.
    assert replay.messages[2].content == first_reply.content
    assert replay.messages[4].content == second_reply.content
    assert replay.tools is None


def test_re_asked_identical_history_reuses_the_exact_prompt() -> None:
    provider, recorder = _framed_provider(["a</>", "b</>"])
    _chat(provider, _HISTORY)
    _chat(provider, _HISTORY)
    first, second = recorder.spans()
    assert second.prompt_token_ids == first.prompt_token_ids
    assert recorder.fallback_count == 0
    # A reused prompt drops the previous sampled turn, so concatenated deltas
    # would describe a conversation that never happened: record none.
    assert second.delta_messages is None
    assert reconstruct_conversation(recorder.spans(), _FramedRendering()) is None


def test_tool_schema_change_mid_episode_falls_back() -> None:
    provider, recorder = _framed_provider(["a</>", "b</>"])
    tool = ChatTool(
        function=ChatFunctionDefinition(
            name="bash", description="run bash", parameters={"type": "object"}
        )
    )
    request = ChatRequest(messages=list(_HISTORY), temperature=0.0, max_tokens=64)
    request.tools = [tool]
    provider.complete_chat(request)
    follow_up = ChatRequest(
        messages=[*_HISTORY, ChatMessage(role="assistant", content="a"), _tool_result("out")],
        temperature=0.0,
        max_tokens=64,
    )
    provider.complete_chat(follow_up)
    assert recorder.fallback_count == 1
    # Each span carries the schemas its own prompt rendered with.
    assert recorder.spans()[0].tools == [tool]
    assert recorder.spans()[1].tools == []


def test_complete_plain_text_uses_same_machinery() -> None:
    recorder = TokenRecorder()
    rendering = _MiniRendering()
    provider = TinkerChatProvider(
        _config(),
        sampling_client=FakeSamplingClient(seed="s"),
        renderer=rendering,
        recorder=recorder,
    )
    completion = provider.complete(
        "sys prompt", [Message(role="user", content="do it")], temperature=0.5, max_tokens=8
    )
    span = recorder.spans()[0]
    assert completion.text == rendering.decode(span.sampled_token_ids)
    assert completion.usage.input_tokens == len(span.prompt_token_ids)
    assert completion.usage.output_tokens == 8
    # The system prompt travels as a leading system message.
    assert "system: sys prompt" in rendering.decode(span.prompt_token_ids)


def test_embed_raises_actionable_error() -> None:
    provider = TinkerChatProvider(_config(), sampling_client=FakeSamplingClient(seed="s"))
    with pytest.raises(ValueError, match="embedder"):
        provider.embed(["text"])


def test_verify_ok_via_fakes_and_never_records() -> None:
    recorder = TokenRecorder()
    provider = TinkerChatProvider(
        _config(),
        sampling_client=FakeSamplingClient(seed="s"),
        renderer=_MiniRendering(),
        recorder=recorder,
    )
    result = provider.verify()
    assert result.ok is True
    assert result.kind is ProviderKind.TINKER
    assert result.model == "tinker://run/weights/0"
    assert len(recorder) == 0


def test_registry_constructs_provider_without_touching_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Poison the SDK modules: construction and fake-backed completions must
    # never import tinker or tinker_cookbook.
    monkeypatch.setitem(sys.modules, "tinker", None)
    monkeypatch.setitem(sys.modules, "tinker_cookbook", None)
    provider = get_provider(_config())
    assert isinstance(provider, TinkerChatProvider)
    assert isinstance(provider, Provider)
    assert isinstance(provider, ToolCallingProvider)

    injected = TinkerChatProvider(
        _config(), sampling_client=FakeSamplingClient(seed="s"), renderer=_MiniRendering()
    )
    response = injected.complete_chat(_request())
    assert response.choices[0].message.role == "assistant"


def test_missing_extra_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tinker", None)
    provider = TinkerChatProvider(_config(), renderer=_MiniRendering())
    with pytest.raises(ImportError, match="uv sync --extra distill"):
        provider.complete_chat(_request())


def test_missing_api_key_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("tinker")
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    provider = TinkerChatProvider(_config(), renderer=_MiniRendering())
    # The message must state the problem (not set) and the remedy (set it).
    with pytest.raises(RuntimeError, match="TINKER_API_KEY is not set"):
        provider.complete_chat(_request())


def test_tinker_path_without_model_type_is_actionable() -> None:
    config = ProviderConfig(kind=ProviderKind.TINKER, model="tinker://run/weights/0")
    provider = TinkerChatProvider(config, sampling_client=FakeSamplingClient(seed="s"))
    with pytest.raises(ValueError, match="model_type is unset"):
        provider.complete_chat(_request())


def test_tinker_path_in_model_type_names_the_swapped_field() -> None:
    # The swapped-fields mistake (weights path in model_type) must not claim that
    # model_type is unset; the message points at the field that actually holds the path.
    config = ProviderConfig(
        kind=ProviderKind.TINKER,
        model="Qwen/Qwen3-8B",
        model_type="tinker://run/weights/0",
    )
    provider = TinkerChatProvider(config, sampling_client=FakeSamplingClient(seed="s"))
    with pytest.raises(ValueError, match="weights paths belong in config.model"):
        provider.complete_chat(_request())


def test_injected_sampler_without_tokenizer_requires_renderer() -> None:
    provider = TinkerChatProvider(_config(), sampling_client=FakeSamplingClient(seed="s"))
    with pytest.raises(RuntimeError, match="renderer="):
        provider.complete_chat(_request())


def _module_scope_import_roots(path: Path) -> set[str]:
    """Top-level import roots of a module (TYPE_CHECKING blocks excluded)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            roots.add(node.module.split(".")[0])
    return roots


def test_module_scope_never_imports_the_distill_extra() -> None:
    # The tinker/tinker-cookbook SDKs are optional; module import must stay
    # lazy so the provider modules load without the distill extra installed.
    for module in (tinker_module, rendering_module):
        assert module.__file__ is not None
        roots = _module_scope_import_roots(Path(module.__file__))
        assert not roots & {"tinker", "tinker_cookbook"}, module.__name__


# --- deadlines: wedged sessions become retryable errors with fresh clients ----------------------


class _WedgedSampler:
    """A sampler whose every call reports a deadline expiry (a wedged session)."""

    def __init__(self) -> None:
        self.calls = 0

    def sample(
        self,
        prompt_token_ids: list[int],
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str] | list[int] | None = None,
    ) -> NoReturn:
        del prompt_token_ids, max_tokens, temperature, stop
        self.calls += 1
        raise TinkerDeadlineError("sample", elapsed_s=0.05, deadline_s=0.05)


def test_sampling_deadline_drops_and_rebuilds_the_lazy_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = TokenRecorder()
    provider = TinkerChatProvider(_config(), renderer=_MiniRendering(), recorder=recorder)
    builds: list[TinkerSampler] = []

    def build_sampler() -> TinkerSampler:
        sampler: TinkerSampler = _WedgedSampler() if not builds else FakeSamplingClient(seed="s")
        builds.append(sampler)
        return sampler

    monkeypatch.setattr(provider, "_build_sdk_sampler", build_sampler)

    with pytest.raises(TinkerDeadlineError, match="timed out"):
        provider.complete_chat(_request())
    # The timed-out call recorded no span, and the retry wrapper's next
    # attempt (simulated by calling again) builds a fresh client and succeeds.
    assert len(recorder) == 0
    response = provider.complete_chat(_request())
    assert response.choices[0].message.role == "assistant"
    assert len(builds) == 2
    assert [span.call_index for span in recorder.spans()] == [0]


def test_injected_sampling_client_is_never_dropped_on_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An injected client cannot be rebuilt; poison the SDK so any accidental
    # rebuild attempt would fail loudly instead of hitting the network.
    monkeypatch.setitem(sys.modules, "tinker", None)
    sampler = _WedgedSampler()
    provider = TinkerChatProvider(_config(), sampling_client=sampler, renderer=_MiniRendering())
    for _ in range(2):
        with pytest.raises(TinkerDeadlineError):
            provider.complete_chat(_request())
    assert sampler.calls == 2


class _NeverResolvingFuture:
    """Mimics the SDK future of a wedged session: result(timeout) honors the timeout."""

    def __init__(self) -> None:
        self._never = threading.Event()

    def result(self, timeout: float | None = None) -> NoReturn:
        self._never.wait(timeout)
        raise TimeoutError(f"fake future gave up after {timeout}s")


def test_sdk_sampler_bounds_the_sample_future(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("tinker")
    monkeypatch.setenv("WMH_TINKER_DEADLINE_SAMPLE", "0.05")

    class _WedgedClient:
        def sample(
            self, prompt: object, num_samples: int, sampling_params: object
        ) -> _NeverResolvingFuture:
            del prompt, num_samples, sampling_params
            return _NeverResolvingFuture()

    sampler = SdkSampler(cast("tinker.SamplingClient", _WedgedClient()))
    with pytest.raises(TinkerDeadlineError, match="tinker sample timed out"):
        sampler.sample([1, 2, 3], max_tokens=4, temperature=1.0)


# --- process-wide client sharing: the session-leak regression -----------------------------------


@dataclass
class _FakeSdkState:
    """Counters the fake tinker SDK module tracks for the sharing tests.

    `live_sessions` is deliberately never decremented: the real SDK's
    heartbeat task strongly references each constructed client's holder, so
    every construction pins a live server-side session for the rest of the
    process. `session_cap` models the service refusing new session creation
    with a capacity-shaped error once too many are live.
    """

    session_cap: int
    live_sessions: int = 0
    service_clients: int = 0
    sampling_clients: int = 0
    wedge_first_sampler: bool = False
    wedge_sampler_construction: bool = False


@dataclass(frozen=True)
class _FakeSdkSequence:
    tokens: list[int]
    logprobs: list[float]


@dataclass(frozen=True)
class _FakeSdkSampleResult:
    sequences: list[_FakeSdkSequence]


class _ReadySdkFuture:
    """A fake SDK future whose result is immediately available."""

    def __init__(self, value: _FakeSdkSampleResult) -> None:
        self._value = value

    def result(self, timeout: float | None = None) -> _FakeSdkSampleResult:
        del timeout
        return self._value


class _TimedOutSdkFuture:
    """A wedged session's future: reports the timeout immediately (fast tests)."""

    def result(self, timeout: float | None = None) -> NoReturn:
        raise TimeoutError(f"fake future gave up after {timeout}s")


class _FakeSdkSamplingClient:
    """Just enough of `tinker.SamplingClient` for `SdkSampler` to drive."""

    def __init__(self, wedged: bool) -> None:
        self._wedged = wedged

    def sample(
        self, prompt: object, num_samples: int, sampling_params: object
    ) -> _ReadySdkFuture | _TimedOutSdkFuture:
        del prompt, num_samples, sampling_params
        if self._wedged:
            return _TimedOutSdkFuture()
        return _ReadySdkFuture(
            _FakeSdkSampleResult(
                sequences=[_FakeSdkSequence(tokens=[65, 66], logprobs=[-0.1, -0.2])]
            )
        )

    def get_tokenizer(self) -> FakeTokenizer:
        return FakeTokenizer()


class _FakeModelInput:
    """Stands in for tinker.ModelInput (the provider only calls from_ints)."""

    @classmethod
    def from_ints(cls, token_ids: list[int]) -> list[int]:
        return list(token_ids)


class _FakeSamplingParams:
    """Stands in for tinker.SamplingParams (constructed, never read)."""

    def __init__(
        self,
        max_tokens: int,
        temperature: float,
        stop: list[str] | list[int] | None = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stop = stop


class _FakeTinkerModule:
    """A sys.modules-injectable stand-in for the tinker SDK module."""

    ServiceClient: type
    ModelInput: type[_FakeModelInput]
    SamplingParams: type[_FakeSamplingParams]

    def __init__(self, service_client: type) -> None:
        self.ServiceClient = service_client
        self.ModelInput = _FakeModelInput
        self.SamplingParams = _FakeSamplingParams


def _install_fake_sdk(
    monkeypatch: pytest.MonkeyPatch,
    *,
    session_cap: int,
    wedge_first_sampler: bool = False,
    wedge_sampler_construction: bool = False,
) -> _FakeSdkState:
    """Install a fake tinker module that meters sessions like the live outage."""
    state = _FakeSdkState(
        session_cap=session_cap,
        wedge_first_sampler=wedge_first_sampler,
        wedge_sampler_construction=wedge_sampler_construction,
    )

    class _FakeServiceClient:
        def __init__(self) -> None:
            state.service_clients += 1
            state.live_sessions += 1

        def create_sampling_client(
            self, model_path: str | None = None, base_model: str | None = None
        ) -> _FakeSdkSamplingClient:
            assert (model_path is None) != (base_model is None)
            if state.wedge_sampler_construction:
                threading.Event().wait()  # a fully blocking construction, never returns
            state.live_sessions += 1
            if state.live_sessions > state.session_cap:
                raise RuntimeError(
                    "429 too many requests: session capacity exceeded "
                    f"({state.live_sessions} live sessions, cap {state.session_cap})"
                )
            state.sampling_clients += 1
            wedged = state.wedge_first_sampler and state.sampling_clients == 1
            return _FakeSdkSamplingClient(wedged)

    fake = _FakeTinkerModule(_FakeServiceClient)
    monkeypatch.setitem(sys.modules, "tinker", cast("ModuleType", fake))
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    # Give the test its own empty process-wide cache (restored afterwards).
    monkeypatch.setattr(tinker_module, "_shared_service", None)
    monkeypatch.setattr(tinker_module, "_shared_samplers", {})
    monkeypatch.setattr(tinker_module, "build_renderer", _stub_build_renderer)
    return state


def _stub_build_renderer(base_model: str, tokenizer: object) -> _MiniRendering:
    """The cookbook renderer stub for the fake base models."""
    del base_model, tokenizer
    return _MiniRendering()


def test_session_cap_regression_shares_one_service_client_across_trials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The outage: a fresh ServiceClient + sampling client per trial pinned a
    # live session each until the service refused new session creation with
    # capacity errors (~240 cumulative). M >> N per-trial providers must all
    # succeed because the SDK clients are shared process-wide.
    state = _install_fake_sdk(monkeypatch, session_cap=5)
    weights_config = _config()
    base_config = ProviderConfig(kind=ProviderKind.TINKER, model="Qwen/Qwen3-8B")

    for trial in range(40):
        provider = get_provider(weights_config if trial % 2 == 0 else base_config)
        assert isinstance(provider, TinkerChatProvider)
        response = provider.complete_chat(_request())
        assert response.choices[0].message.role == "assistant"

    assert state.service_clients == 1
    # Live sessions are bounded by distinct model strings, not by trial count:
    # the one service client plus one sampling client per model.
    assert state.sampling_clients == 2
    assert state.live_sessions == 1 + 2


def test_deadline_evicts_the_shared_cache_so_every_user_heals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_fake_sdk(monkeypatch, session_cap=100, wedge_first_sampler=True)

    first = TinkerChatProvider(_config())
    with pytest.raises(TinkerDeadlineError, match="timed out"):
        first.complete_chat(_request())
    # The wedged client was EVICTED from the shared cache (not merely dropped
    # from this provider), so a different provider builds a fresh session
    # instead of inheriting the wedged one.
    assert _config().model not in tinker_module._shared_samplers
    second = TinkerChatProvider(_config())
    assert second.complete_chat(_request()).choices[0].message.role == "assistant"
    assert state.sampling_clients == 2
    # The first provider's next attempt heals through the same fresh client.
    assert first.complete_chat(_request()).choices[0].message.role == "assistant"
    assert state.sampling_clients == 2  # reused from the cache, not rebuilt again
    assert state.service_clients == 1


def test_shared_sampling_client_construction_is_deadline_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_sdk(monkeypatch, session_cap=100, wedge_sampler_construction=True)
    monkeypatch.setenv("WMH_TINKER_DEADLINE_CONNECT", "0.05")

    with pytest.raises(TinkerDeadlineError, match="tinker connect timed out"):
        tinker_module.shared_sampling_client("tinker://run/weights/0")
    # Nothing was cached, so a later attempt rebuilds instead of returning a
    # half-constructed entry.
    assert "tinker://run/weights/0" not in tinker_module._shared_samplers


class _UnparsedRendering(_MiniRendering):
    """Reports a tool call the renderer could not read (a genuine format error)."""

    def parse_response(self, sampled_ids: list[int]) -> ParsedAssistantMessage:
        del sampled_ids
        return ParsedAssistantMessage(
            text="<tool_call>\n<function=submit>",
            tool_calls=[],
            stopped=True,
            unparsed_errors=["Malformed Qwen3.5 tool call XML"],
        )


def test_an_unreadable_tool_call_travels_with_the_completion() -> None:
    """The parse error must reach the agent scaffold, not just a log line.

    A dropped tool call used to be indistinguishable from prose, so the runner ended the episode as
    a clean submission with reward 0. The scaffold now feeds this back as an observation.
    """
    provider = TinkerChatProvider(
        _config(), sampling_client=FakeSamplingClient(seed="s"), renderer=_UnparsedRendering()
    )

    response = provider.complete_chat(_request())

    choice = response.choices[0]
    assert choice.message.tool_calls is None
    extra = choice.model_extra or {}
    assert extra[UNPARSED_TOOL_CALLS_KEY] == ["Malformed Qwen3.5 tool call XML"]
    # It survives the wire hop to the pi runner, which reads it off the choice.
    wire_choice = _as_dict(_as_list(response.wire_payload()["choices"])[0])
    assert wire_choice[UNPARSED_TOOL_CALLS_KEY] == ["Malformed Qwen3.5 tool call XML"]


def test_a_clean_completion_carries_no_unparsed_key() -> None:
    provider = TinkerChatProvider(
        _config(), sampling_client=FakeSamplingClient(seed="s"), renderer=_ToolCallRendering()
    )

    response = provider.complete_chat(_request())

    assert UNPARSED_TOOL_CALLS_KEY not in (response.choices[0].model_extra or {})
    assert UNPARSED_TOOL_CALLS_KEY not in _as_dict(_as_list(response.wire_payload()["choices"])[0])


class _Supported:
    """One `SupportedModel`-shaped entry from the service capabilities response."""

    def __init__(self, model_name: str, max_context_length: int) -> None:
        self.model_name = model_name
        self.max_context_length = max_context_length


class _Capabilities:
    def __init__(self, supported_models: list[_Supported]) -> None:
        self.supported_models = supported_models


def test_the_served_context_window_comes_from_the_service_not_a_local_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pi runner calibrates its context guard to this, so it must be the real deployment.

    The context tier is part of the served model identity (the catalog names carry a `:262144`-style
    suffix); a hardcoded 128,000 against a smaller deployment produced 118 context-overflow 400s in
    one run, with the guard never firing before the error.
    """

    class _Service:
        def __init__(self) -> None:
            self.calls = 0

        def get_server_capabilities(self) -> _Capabilities:
            self.calls += 1
            return _Capabilities(
                [
                    _Supported("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:peft:262144", 262144),
                    _Supported("Qwen/Qwen3-8B", 32768),
                ]
            )

    service = _Service()
    monkeypatch.setattr(tinker_module, "_served_context_windows", None)
    monkeypatch.setattr(tinker_module, "shared_service_client", lambda: service)

    provider = TinkerChatProvider(
        _config(), sampling_client=FakeSamplingClient(seed="s"), renderer=_MiniRendering()
    )
    assert provider.context_window() == 32768

    nemotron = ProviderConfig(
        kind=ProviderKind.TINKER,
        model_type="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:peft:262144",
        model="tinker://run/weights/7",
    )
    big = TinkerChatProvider(
        nemotron, sampling_client=FakeSamplingClient(seed="s"), renderer=_MiniRendering()
    )
    assert big.context_window() == 262144
    # Capabilities are fetched once per process, not per episode.
    assert service.calls == 1


def test_an_unlisted_model_reports_no_window_rather_than_a_guess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Service:
        def get_server_capabilities(self) -> _Capabilities:
            return _Capabilities([_Supported("some/other-model", 8192)])

    monkeypatch.setattr(tinker_module, "_served_context_windows", None)
    monkeypatch.setattr(tinker_module, "shared_service_client", lambda: _Service())

    provider = TinkerChatProvider(
        _config(), sampling_client=FakeSamplingClient(seed="s"), renderer=_MiniRendering()
    )
    assert provider.context_window() is None


def test_the_retry_wrapper_forwards_the_context_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wrapper that swallowed the capability would silently send the runner to its fallback."""

    class _Service:
        def get_server_capabilities(self) -> _Capabilities:
            return _Capabilities([_Supported("Qwen/Qwen3-8B", 32768)])

    monkeypatch.setattr(tinker_module, "_served_context_windows", None)
    monkeypatch.setattr(tinker_module, "shared_service_client", lambda: _Service())

    wrapped = wrap_provider_with_retries(
        TinkerChatProvider(
            _config(), sampling_client=FakeSamplingClient(seed="s"), renderer=_MiniRendering()
        )
    )
    assert isinstance(wrapped, ContextWindowProvider)
    assert wrapped.context_window() == 32768


def test_repeated_expiries_escalate_to_rebuilding_the_service_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replacing only the sampling client cannot heal a wedge one level up.

    Live failure this pins: two runs sat at 96 and 9 CONSECUTIVE deadline expiries with zero
    rollouts for 30+ minutes, each expiry dutifully rebuilding the SamplingClient -- from the
    same wedged process-wide ServiceClient -- while an independent process reached the same
    service in 1.7s.
    """
    rebuilds: list[bool] = []
    monkeypatch.setattr(
        tinker_module, "rebuild_shared_service_client", lambda: rebuilds.append(True)
    )
    provider = TinkerChatProvider(_config(), renderer=_MiniRendering())

    for _ in range(tinker_module._WEDGE_REBUILD_THRESHOLD - 1):
        provider._sampler = _FlakySampler(FakeSamplingClient("wedge-probe"), failures=0)
        provider._drop_wedged_sampler()
    assert rebuilds == [], "escalated before the threshold"

    provider._sampler = _FlakySampler(FakeSamplingClient("wedge-probe"), failures=0)
    provider._drop_wedged_sampler()
    assert rebuilds == [True], "did not escalate at the threshold"


def test_a_successful_call_resets_the_expiry_streak(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only an UNBROKEN run of failures may escalate.

    A cumulative counter would eventually rebuild the service client on a healthy run that
    merely expires occasionally, and every rebuild pins another server-side session against a
    ~240 cumulative cap.
    """
    rebuilds: list[bool] = []
    monkeypatch.setattr(
        tinker_module, "rebuild_shared_service_client", lambda: rebuilds.append(True)
    )
    provider = TinkerChatProvider(_config(), renderer=_MiniRendering())

    for _ in range(tinker_module._WEDGE_REBUILD_THRESHOLD * 3):
        provider._sampler = _FlakySampler(FakeSamplingClient("wedge-probe"), failures=0)
        provider._drop_wedged_sampler()
        provider.note_healthy_call()

    assert rebuilds == [], "a reset streak must never escalate"


def test_an_explicit_api_key_is_rejected_rather_than_silently_ignored() -> None:
    """`get_provider(..., api_key=...)` promises the backend authenticates with exactly
    that key. Tinker cannot: its `ServiceClient` is cached per PROCESS off
    `TINKER_API_KEY`, not per credential, so honoring the argument silently would sample
    on whichever account the environment names."""
    with pytest.raises(ValueError, match="does not accept an explicit api_key"):
        get_provider(_config(), api_key="sk-pool-entry")


def test_provider_env_vars_names_the_key_the_provider_actually_reads() -> None:
    # Without the entry the `providers verify` hint drops the "(TINKER_API_KEY)" clue and
    # the CLI never prompts for the key; the equality pins the literal in config.py to the
    # name the provider reads, so the two cannot drift.
    assert PROVIDER_ENV_VARS[ProviderKind.TINKER] == [TINKER_API_KEY_ENV]


def test_get_provider_can_construct_tinker_without_an_explicit_key() -> None:
    # Registering ProviderKind.TINKER puts this provider on get_provider's trusted-credential
    # path, which calls backend(config, api_key=api_key) for EVERY kind. Before __init__ grew
    # the parameter this was an unconditional TypeError the type checker caught and no test did.
    assert isinstance(get_provider(_config()), TinkerChatProvider)
