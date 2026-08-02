"""RunnerLink conformance tests: drive the whole broker offline via a scripted runner peer.

No socket, no node, no Bedrock. A `FakeChannel` plays the runner side (emits tool_request /
llm_request / done frames and records what the host sent back); a fake `AgentEnvironment` stands in
for the world model; `worker_fn` is injected so the worker-LLM callback needs no provider.
"""

from __future__ import annotations

import logging
import time
from typing import Any, cast

import pytest
from llm_waterfall import ChatRequest, ChatResponse

from wmo.core.types import Action, JsonObject, Observation
from wmo.harness import runner_link as runner_link_module
from wmo.harness.runner_link import RunnerLink, SocketChannel, read_frame, write_frame
from wmo.harness.runtime import SCAFFOLD_LOSS_STOP_REASONS, RuntimeCancelled, StopReason
from wmo.harness.skills import Skill, SkillLibrary
from wmo.harness.tools import SUBMIT, TOOL_REGISTRY


class _Env:
    def __init__(self) -> None:
        self.actions: list[Action] = []

    def execute(self, action: Action) -> Observation:
        self.actions.append(action)
        return Observation(content=f"ran {action.name}")

    def close(self) -> None:
        pass


class _FakeChannel:
    """Plays the runner peer: recv() yields scripted frames in order; send() records host output."""

    def __init__(self, script: list) -> None:
        self.sent: list = []
        self._script = list(script)

    def send(self, frame: JsonObject) -> None:
        self.sent.append(frame)

    def recv(self, timeout: float | None = None) -> JsonObject | None:
        del timeout
        return self._script.pop(0) if self._script else None


class _ResponseSendTimeoutChannel(_FakeChannel):
    """Records the attempted response, then fails every response transport send."""

    def send(self, frame: JsonObject) -> None:
        self.sent.append(frame)
        if frame.get("type") == "llm_response":
            raise TimeoutError("transport send timed out")


class _EpisodeStartTimeoutChannel(_FakeChannel):
    def send(self, frame: JsonObject) -> None:
        self.sent.append(frame)
        raise TimeoutError("episode start send timed out")


class _TimedWaitChannel(_FakeChannel):
    """Play a prefix, then model a timed wait with no runner frame."""

    def __init__(self, script: list) -> None:
        super().__init__(script)
        self.recv_timeouts: list[float | None] = []

    def recv(self, timeout: float | None = None) -> JsonObject | None:
        self.recv_timeouts.append(timeout)
        if self._script:
            return self._script.pop(0)
        if timeout is not None:
            time.sleep(timeout)
            raise TimeoutError("no frame before poll deadline")
        return None


def _tools() -> list:
    return [TOOL_REGISTRY["bash"], SUBMIT]


def _completion(
    content: str = "ok",
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
    cache_write_input_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> ChatResponse:
    return ChatResponse.model_validate(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "prompt_tokens_details": {
                    "cached_tokens": cached_input_tokens,
                    "cache_write_tokens": cache_write_input_tokens,
                },
                "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
            },
        }
    )


def _link(channel: _FakeChannel, **kw) -> RunnerLink:  # noqa: ANN003
    # worker_fn returns a fixed completion so the llm_request path needs no provider.
    return RunnerLink(channel, worker_fn=lambda request: _completion(), **kw)


def _sent(channel: _FakeChannel, kind: str) -> list:
    # cast to Any so deep-indexing assertions on frame payloads stay readable in tests.
    return [cast(Any, f) for f in channel.sent if f.get("type") == kind]


# --- frame codec ---
class _PipeSock:
    def __init__(self) -> None:
        self.buf = bytearray()

    def sendall(self, data: bytes) -> None:
        self.buf += data

    def recv(self, n: int) -> bytes:
        if not self.buf:
            return b""
        chunk = bytes(self.buf[:n])
        del self.buf[:n]
        return chunk


class _TimedPipeSock(_PipeSock):
    """Timeout after a partial frame, then resume from the remaining bytes."""

    def __init__(self) -> None:
        super().__init__()
        self._recv_calls = 0
        self._timeout: float | None = None

    def gettimeout(self) -> float | None:
        return self._timeout

    def settimeout(self, timeout: float | None) -> None:
        self._timeout = timeout

    def recv(self, n: int) -> bytes:
        self._recv_calls += 1
        if self._recv_calls == 1:
            return super().recv(min(2, n))  # half the length header
        if self._recv_calls == 2:
            raise TimeoutError("fragment stalled")
        return super().recv(n)


def test_frame_codec_roundtrip_and_eof() -> None:
    sock = _PipeSock()
    hello: JsonObject = {"type": "hello", "n": 1, "s": "x" * 5000}
    done: JsonObject = {"type": "done", "reason": "submit", "answer": "café"}
    write_frame(sock, hello)
    write_frame(sock, done)
    assert read_frame(sock) == hello
    assert read_frame(sock) == done
    assert read_frame(sock) is None  # clean EOF


def test_socket_channel_preserves_a_partial_frame_across_timed_polls() -> None:
    sock = _TimedPipeSock()
    frame: JsonObject = {"type": "done", "reason": "submit", "answer": "fragmented"}
    write_frame(sock, frame)
    channel = SocketChannel(sock)

    with pytest.raises(TimeoutError, match="fragment stalled"):
        channel.recv(timeout=0.01)

    assert channel.recv(timeout=0.01) == frame
    assert sock.gettimeout() is None  # each bounded poll restores the caller's socket setting


# --- episode broker ---
def test_episode_start_carries_task_tools_and_limits() -> None:
    ch = _FakeChannel([{"type": "done", "reason": "submit", "answer": "x"}])
    _link(
        ch,
        system_prompt="sys",
        files={"src/agent.ts": "// a"},
        max_turns=7,
        max_output_tokens=16384,
        temperature=0.25,
        episode_timeout_s=12.5,
    ).run("t1", "do it", _Env(), tools=_tools())
    start = _sent(ch, "episode_start")
    assert len(start) == 1
    s = start[0]
    assert s["instruction"] == "do it" and s["system"] == "sys"
    assert s["files"] == {"src/agent.ts": "// a"}
    assert s["max_turns"] == 7
    assert s["max_output_tokens"] == 16384
    assert s["temperature"] == 0.25
    assert s["episode_timeout_s"] == 12.5
    assert {t["name"] for t in s["tools"]} >= {"bash", "submit"}


def test_tool_request_routes_to_env_and_records_step() -> None:
    env = _Env()
    script = [
        {"type": "tool_request", "req_id": 1, "name": "bash", "arguments": {"command": "ls"}},
        {"type": "done", "reason": "submit", "answer": "done-42"},
    ]
    ch = _FakeChannel(script)
    result = _link(ch).run("t1", "do it", env, tools=_tools())
    assert result.stop_reason is StopReason.SUBMITTED
    assert result.answer == "done-42"
    assert [a.name for a in env.actions] == ["bash"]  # the host WM answered the call
    tr = _sent(ch, "tool_response")
    assert tr[0]["content"] == "ran bash" and tr[0]["is_error"] is False
    assert tr[0]["req_id"] == 1  # correlation id echoed
    assert len(result.steps) == 1


def test_env_action_budget_enforced() -> None:
    env = _Env()
    script = [
        {"type": "tool_request", "req_id": i, "name": "bash", "arguments": {}} for i in range(4)
    ]
    script.append({"type": "done", "reason": "submit", "answer": "ok"})
    ch = _FakeChannel(script)
    result = _link(ch, max_env_actions=2).run("t1", "x", env, tools=_tools())
    assert len(env.actions) == 2  # only budgeted calls reached the environment
    responses = _sent(ch, "tool_response")
    assert responses[2]["is_error"] is True and "budget" in responses[2]["content"]
    assert result.stop_reason is StopReason.SUBMITTED


def test_llm_request_answered_via_worker_fn() -> None:
    calls: list[ChatRequest] = []

    def worker(request: ChatRequest) -> ChatResponse:
        calls.append(request)
        return _completion("hi")

    script = [
        {"type": "llm_request", "req_id": 7, "openai_body": {"messages": [{"role": "user"}]}},
        {"type": "done", "reason": "submit", "answer": "fin"},
    ]
    ch = _FakeChannel(script)
    RunnerLink(ch, worker_fn=worker).run("t1", "x", _Env(), tools=_tools())
    assert len(calls) == 1  # the worker callback fired host-side
    resp = _sent(ch, "llm_response")[0]
    assert resp["req_id"] == 7
    assert resp["completion"]["choices"][0]["message"]["content"] == "hi"


def test_harness_temperature_overrides_the_runner_request_before_the_worker_call() -> None:
    calls: list[ChatRequest] = []

    def worker(request: ChatRequest) -> ChatResponse:
        calls.append(request)
        return _completion()

    ch = _FakeChannel(
        [
            {
                "type": "llm_request",
                "req_id": 1,
                "openai_body": {"messages": [], "temperature": 1.75},
            },
            {"type": "done", "reason": "submit", "answer": "ok"},
        ]
    )
    RunnerLink(ch, worker_fn=worker, temperature=0.35).run("t1", "x", _Env(), tools=_tools())

    assert [request.temperature for request in calls] == [0.35]


def test_skill_library_implicitly_adds_and_serves_read_skill_without_env_budget() -> None:
    skill = Skill(name="count-words", description="count words", body="wc -w <path>")
    script = [
        {
            "type": "tool_request",
            "req_id": 1,
            "name": "read_skill",
            "arguments": {"name": "count-words"},
        },
        {
            "type": "tool_request",
            "req_id": 2,
            "name": "read_skill",
            "arguments": {"name": "ghost"},
        },
        {"type": "done", "reason": "submit", "answer": "ok"},
    ]
    channel = _FakeChannel(script)
    env = _Env()
    result = RunnerLink(
        channel,
        tools=_tools(),
        worker_fn=lambda request: _completion(),
        skills=SkillLibrary([skill]),
        max_env_actions=0,
    ).run("t1", "x", env)

    advertised = {tool["name"] for tool in _sent(channel, "episode_start")[0]["tools"]}
    assert "read_skill" in advertised
    responses = _sent(channel, "tool_response")
    assert responses[0]["content"] == "wc -w <path>" and responses[0]["is_error"] is False
    assert responses[1]["content"] == "no skill named 'ghost'"
    assert responses[1]["is_error"] is True
    assert [step.action.name for step in result.steps] == ["read_skill", "read_skill"]
    assert env.actions == []


def test_worker_fn_error_is_reported_not_crashed() -> None:
    def boom(request: ChatRequest) -> ChatResponse:
        del request
        raise RuntimeError("provider down")

    script = [
        {"type": "llm_request", "req_id": 1, "openai_body": {}},
        {"type": "done", "reason": "submit", "answer": "ok"},
    ]
    ch = _FakeChannel(script)
    result = RunnerLink(ch, worker_fn=boom).run("t1", "x", _Env(), tools=_tools())
    responses = _sent(ch, "llm_response")
    assert len(responses) == 1
    resp = responses[0]
    assert "provider down" in resp["error"]  # surfaced to the runner, host survives
    assert result.stop_reason is StopReason.SUBMITTED


def test_worker_fn_error_is_logged_at_warning_with_type_and_bounded_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The live outage: a provider that failed every call was surfaced only to
    # the runner, so the host logs showed clean zero-turn episodes. The host
    # must warn (bounded, no traceback) before returning the error frame.
    long_detail = "session capacity exceeded " + "x" * 2000

    def boom(request: ChatRequest) -> ChatResponse:
        del request
        raise RuntimeError(long_detail)

    script = [
        {"type": "llm_request", "req_id": 1, "openai_body": {}},
        {"type": "done", "reason": "submit", "answer": "ok"},
    ]
    ch = _FakeChannel(script)
    with caplog.at_level(logging.WARNING, logger="wmo.harness.runner_link"):
        result = RunnerLink(ch, worker_fn=boom).run("t1", "x", _Env(), tools=_tools())

    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and "worker completion failed" in record.getMessage()
    ]
    assert len(warnings) == 1
    assert "RuntimeError" in warnings[0]
    assert "session capacity exceeded" in warnings[0]
    assert "(truncated)" in warnings[0]
    assert len(warnings[0]) < 700  # bounded at warning level
    # The error frame still flows to the runner and the episode still ends.
    responses = _sent(ch, "llm_response")
    assert len(responses) == 1
    assert "session capacity exceeded" in responses[0]["error"]
    assert result.stop_reason is StopReason.SUBMITTED


def test_llm_response_send_timeout_propagates_without_error_response_retry() -> None:
    """A transport send failure is not mislabeled as a provider failure or sent twice."""
    script = [{"type": "llm_request", "req_id": 1, "openai_body": {}}]
    ch = _ResponseSendTimeoutChannel(script)

    with pytest.raises(TimeoutError, match="transport send timed out"):
        _link(ch).run("t1", "x", _Env(), tools=_tools())

    responses = _sent(ch, "llm_response")
    assert len(responses) == 1
    assert "completion" in responses[0]
    assert "error" not in responses[0]


def test_send_failure_after_wall_deadline_returns_budget_instead_of_transport_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter([0.0, 2.0])  # deadline starts at 1; the failed start send returns after it
    monkeypatch.setattr(runner_link_module.time, "monotonic", lambda: next(ticks))
    ch = _EpisodeStartTimeoutChannel([])

    result = _link(ch, episode_timeout_s=1.0).run("t1", "x", _Env(), tools=_tools())

    assert result.stop_reason is StopReason.BUDGET
    assert "wall budget" in result.transcript()
    assert len(_sent(ch, "episode_start")) == 1


def test_cancellation_during_worker_call_suppresses_the_llm_response() -> None:
    cancelled = False

    def worker(request: ChatRequest) -> ChatResponse:
        nonlocal cancelled
        del request
        cancelled = True
        return _completion(input_tokens=17, output_tokens=5)

    ch = _FakeChannel([{"type": "llm_request", "req_id": 1, "openai_body": {}}])
    with pytest.raises(RuntimeCancelled, match="cancelled") as raised:
        RunnerLink(ch, worker_fn=worker, should_cancel=lambda: cancelled).run(
            "t1", "x", _Env(), tools=_tools()
        )

    assert _sent(ch, "llm_response") == []
    assert raised.value.worker_usage is not None
    assert raised.value.worker_usage.input_tokens == 17
    assert raised.value.worker_usage.output_tokens == 5
    assert raised.value.worker_usage.calls == 1


def test_wall_deadline_during_worker_call_suppresses_the_llm_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter([0.0, 0.0, 0.0, 0.0, 2.0])
    monkeypatch.setattr(runner_link_module.time, "monotonic", lambda: next(ticks))
    ch = _FakeChannel([{"type": "llm_request", "req_id": 1, "openai_body": {}}])

    result = RunnerLink(
        ch,
        worker_fn=lambda request: _completion(input_tokens=5),
        episode_timeout_s=1.0,
    ).run("t1", "x", _Env(), tools=_tools())

    assert result.stop_reason is StopReason.BUDGET
    assert result.worker_usage is not None and result.worker_usage.calls == 1
    assert _sent(ch, "llm_response") == []


def test_cancellation_during_environment_call_suppresses_the_tool_response() -> None:
    cancelled = False

    class CancellingEnv(_Env):
        def execute(self, action: Action) -> Observation:
            nonlocal cancelled
            observation = super().execute(action)
            cancelled = True
            return observation

    ch = _FakeChannel([{"type": "tool_request", "req_id": 1, "name": "bash", "arguments": {}}])
    with pytest.raises(RuntimeCancelled, match="cancelled"):
        _link(ch, should_cancel=lambda: cancelled).run("t1", "x", CancellingEnv(), tools=_tools())

    assert _sent(ch, "tool_response") == []


def test_wall_deadline_during_environment_call_suppresses_the_tool_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter([0.0, 0.0, 0.0, 0.0, 2.0])
    monkeypatch.setattr(runner_link_module.time, "monotonic", lambda: next(ticks))
    ch = _FakeChannel([{"type": "tool_request", "req_id": 1, "name": "bash", "arguments": {}}])

    result = _link(ch, episode_timeout_s=1.0).run("t1", "x", _Env(), tools=_tools())

    assert result.stop_reason is StopReason.BUDGET
    assert "ran bash" in result.transcript()
    assert _sent(ch, "tool_response") == []


def test_channel_close_without_done_reports_error() -> None:
    ch = _FakeChannel(
        [{"type": "tool_request", "req_id": 1, "name": "bash", "arguments": {}}]  # then EOF
    )
    result = _link(ch).run("t1", "x", _Env(), tools=_tools())
    assert result.stop_reason is StopReason.MAX_TURNS  # a step ran, so not a bare ERROR
    assert result.steps[-1].observation.is_error


def test_episode_wall_budget_returns_partial_usage_and_budget_stop() -> None:
    ch = _TimedWaitChannel(
        [
            {"type": "llm_request", "req_id": 1, "openai_body": {}},
            {"type": "tool_request", "req_id": 2, "name": "bash", "arguments": {}},
        ]
    )
    env = _Env()
    result = RunnerLink(
        ch,
        worker_fn=lambda request: _completion("working", input_tokens=11, output_tokens=3),
        episode_timeout_s=0.01,
    ).run("t1", "x", env, tools=_tools())

    assert result.stop_reason is StopReason.BUDGET
    assert result.worker_usage is not None
    assert result.worker_usage.calls == 1
    assert result.worker_usage.input_tokens == 11
    assert [action.name for action in env.actions] == ["bash"]
    assert "ran bash" in result.transcript()
    assert "wall budget" in result.transcript()
    assert ch.recv_timeouts[-1] is not None


def test_cooperative_cancellation_interrupts_a_timed_runner_wait() -> None:
    checks = 0

    def should_cancel() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 4

    ch = _TimedWaitChannel([])
    with pytest.raises(RuntimeCancelled, match="cancelled"):
        RunnerLink(
            ch,
            worker_fn=lambda request: _completion(),
            episode_timeout_s=60,
            should_cancel=should_cancel,
            cancel_poll_interval_s=0.01,
        ).run("t1", "x", _Env(), tools=_tools())

    assert ch.recv_timeouts == [0.01]


def test_episode_error_frame_reports_error() -> None:
    ch = _FakeChannel([{"type": "episode_error", "note": "pi fatal"}])
    result = _link(ch).run("t1", "x", _Env(), tools=_tools())
    assert result.stop_reason is StopReason.ERROR  # no steps -> hard error
    assert "pi fatal" in result.steps[-1].observation.content


def test_tools_bound_at_construction_satisfy_runtime_contract() -> None:
    # RunnerLink(tools=...) is a drop-in runtime: run(task_id, instruction, env), no tools kwarg.
    env = _Env()
    script = [
        {"type": "tool_request", "req_id": 1, "name": "bash", "arguments": {"command": "ls"}},
        {"type": "done", "reason": "submit", "answer": "done"},
    ]
    ch = _FakeChannel(script)
    link = RunnerLink(ch, tools=_tools(), worker_fn=lambda request: _completion())
    result = link.run("t1", "do it", env)  # no tools= : uses the constructor's
    assert result.stop_reason is StopReason.SUBMITTED
    assert [a.name for a in env.actions] == ["bash"]
    assert {t["name"] for t in _sent(ch, "episode_start")[0]["tools"]} >= {"bash", "submit"}


def test_multiple_episodes_over_one_channel() -> None:
    # A persistent channel drives episodes sequentially (what closed-loop eval / the search do).
    script = [
        {"type": "done", "reason": "submit", "answer": "a1"},  # episode 1
        {"type": "done", "reason": "submit", "answer": "a2"},  # episode 2
    ]
    ch = _FakeChannel(script)
    link = _link(ch, tools=_tools())
    r1 = link.run("t1", "first", _Env())
    r2 = link.run("t2", "second", _Env())
    assert (r1.answer, r2.answer) == ("a1", "a2")
    starts = _sent(ch, "episode_start")
    assert len(starts) == 2
    assert starts[0]["episode_id"] != starts[1]["episode_id"]  # fresh id per episode
    assert (starts[0]["instruction"], starts[1]["instruction"]) == ("first", "second")


def test_doc_runtime_dispatches_runner_link_under_pi_transport_link() -> None:
    import os as _os

    from wmo.harness.doc import (
        MAX_OUTPUT_TOKENS_ID,
        MAX_TURNS_ID,
        RUNTIME_KIND_ID,
        TOOL_POLICY_ID,
        HarnessDoc,
        Surface,
        SurfaceKind,
    )
    from wmo.harness.runner_link import RunnerLink, set_active_channel
    from wmo.providers.base import Provider, ProviderConfig, ProviderKind

    class _P:
        config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")

        def complete(self, *a, **k) -> object:  # noqa: ANN002, ANN003
            raise NotImplementedError

        def complete_chat(self, request: ChatRequest) -> ChatResponse:
            return _completion()

        def embed(self, texts) -> list:  # noqa: ANN001
            return [[0.0] for _ in texts]

        def verify(self) -> object:
            raise NotImplementedError

    doc = HarnessDoc(
        name="pi",
        surfaces=[
            Surface(id="prompt:core", kind=SurfaceKind.PROMPT, content="p"),
            Surface(id=TOOL_POLICY_ID, kind=SurfaceKind.TOOL_POLICY, content="bash\nsubmit"),
            Surface(id=RUNTIME_KIND_ID, kind=SurfaceKind.PARAM, content="pi-node"),
            Surface(id=MAX_TURNS_ID, kind=SurfaceKind.PARAM, content="7"),
            Surface(id=MAX_OUTPUT_TOKENS_ID, kind=SurfaceKind.PARAM, content="16384"),
            Surface(
                id="code:src-agent-ts", kind=SurfaceKind.CODE, path="src/agent.ts", content="// a"
            ),
        ],
    )
    provider = cast(Provider, _P())
    prev = _os.environ.get("PI_TRANSPORT")
    _os.environ["PI_TRANSPORT"] = "link"
    try:
        set_active_channel(None)
        try:
            doc.runtime(provider)
            raise AssertionError("expected RuntimeError with no active channel")
        except RuntimeError as exc:
            assert "no active runner channel" in str(exc)
        set_active_channel(_FakeChannel([]))
        runtime = doc.runtime(provider)
        assert isinstance(runtime, RunnerLink)
        assert runtime._max_turns == 7  # noqa: SLF001 - document parameter reaches the link frame
        assert runtime._max_output_tokens == 16384  # noqa: SLF001 - same agent model contract
    finally:
        set_active_channel(None)
        if prev is None:
            _os.environ.pop("PI_TRANSPORT", None)
        else:
            _os.environ["PI_TRANSPORT"] = prev


def test_worker_usage_accumulates_across_llm_requests() -> None:
    """Each answered llm_request adds its completion usage to RunResult.worker_usage."""
    replies = iter(
        [
            _completion(
                "a",
                input_tokens=100,
                output_tokens=7,
                cached_input_tokens=40,
                cache_write_input_tokens=10,
                reasoning_tokens=3,
            ),
            _completion("b"),
        ]
    )
    script = [
        {"type": "llm_request", "req_id": 1, "openai_body": {}},
        {"type": "llm_request", "req_id": 2, "openai_body": {}},
        {"type": "done", "reason": "submit", "answer": "fin"},
    ]
    ch = _FakeChannel(script)
    result = RunnerLink(ch, worker_fn=lambda request: next(replies)).run(
        "t1", "x", _Env(), tools=_tools()
    )
    assert result.worker_usage is not None
    assert result.worker_usage.calls == 2
    assert result.worker_usage.input_tokens == 100
    assert result.worker_usage.output_tokens == 7
    assert result.worker_usage.cached_input_tokens == 40
    assert result.worker_usage.cache_write_input_tokens == 10
    assert result.worker_usage.reasoning_tokens == 3
    assert len(result.worker_usage.call_seconds) == 2
    assert all(seconds >= 0 for seconds in result.worker_usage.call_seconds)
    assert result.worker_usage.call_input_tokens == [100, 0]
    assert result.worker_usage.call_output_tokens == [7, 0]
    assert result.worker_usage.call_cached_input_tokens == [40, 0]
    assert result.worker_usage.call_cache_write_input_tokens == [10, 0]
    # No llm_request at all -> usage stays None (not zero: the runtime reported nothing).
    quiet = RunnerLink(
        _FakeChannel([{"type": "done", "reason": "submit", "answer": "ok"}]),
        worker_fn=lambda request: _completion(),
    ).run("t2", "x", _Env(), tools=_tools())
    assert quiet.worker_usage is None


# --- termination disambiguation (the pi scaffold audit's headline defect) ---
#
# Every `done` frame used to map to StopReason.SUBMITTED, so a real submit, a prose-only turn, a
# turn truncated at the output-token cap, and a tool call the renderer dropped were
# indistinguishable and all scored reward 0 as clean completions. 88.8% of Super trials and 92.2%
# of Ultra trials ended that way.
@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("submit", StopReason.SUBMITTED),
        ("no_tool_call", StopReason.NO_TOOL_CALL),
        ("output_truncated", StopReason.OUTPUT_TRUNCATED),
        ("unparsed_tool_call", StopReason.UNPARSED_TOOL_CALL),
        ("provider_error", StopReason.PROVIDER_ERROR),
        ("max_turns", StopReason.MAX_TURNS),
    ],
)
def test_done_reason_maps_to_a_distinct_stop_reason(reason: str, expected: StopReason) -> None:
    ch = _FakeChannel([{"type": "done", "reason": reason, "answer": "text"}])
    result = _link(ch).run("t1", "do it", _Env(), tools=_tools())
    assert result.stop_reason is expected


def test_only_an_explicit_submit_reads_as_a_completion() -> None:
    """No non-submit termination may land in SUBMITTED; that is what hid the scaffold loss."""
    for reason in ("no_tool_call", "output_truncated", "unparsed_tool_call", "provider_error"):
        ch = _FakeChannel([{"type": "done", "reason": reason, "answer": "raw reasoning"}])
        result = _link(ch).run("t1", "do it", _Env(), tools=_tools())
        assert result.stop_reason is not StopReason.SUBMITTED, reason
        assert result.stop_reason in SCAFFOLD_LOSS_STOP_REASONS


def test_done_without_a_reason_is_a_scaffold_loss_not_a_completion(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A stale runner must not read as success.

    Failing toward SUBMITTED here would re-create the original defect one level up: every
    episode from a runner older than the host reads as a clean submit, so `scaffold_loss_rate`
    reads ~0 and a probe whose only job is to count scaffold losses passes.
    """
    ch = _FakeChannel([{"type": "done", "answer": "x"}])
    with caplog.at_level(logging.WARNING, logger=runner_link_module.__name__):
        result = _link(ch).run("t1", "do it", _Env(), tools=_tools())
    assert result.stop_reason is StopReason.UNKNOWN_DONE_REASON
    assert result.stop_reason in SCAFFOLD_LOSS_STOP_REASONS
    assert "predates termination reporting" in caplog.text


def test_done_with_an_unrecognized_reason_is_a_scaffold_loss() -> None:
    """A reason the host does not know is unknown, not success."""
    ch = _FakeChannel([{"type": "done", "reason": "wat", "answer": "x"}])
    result = _link(ch).run("t1", "do it", _Env(), tools=_tools())
    assert result.stop_reason is StopReason.UNKNOWN_DONE_REASON
    assert result.stop_reason in SCAFFOLD_LOSS_STOP_REASONS


def test_episode_start_carries_the_served_context_window() -> None:
    """pi's context guard must be calibrated to the real window, not a hardcoded 128k."""
    ch = _FakeChannel([{"type": "done", "reason": "submit", "answer": "x"}])
    _link(ch, context_window=65_536).run("t1", "do it", _Env(), tools=_tools())
    assert _sent(ch, "episode_start")[0]["context_window"] == 65_536


def test_context_window_is_resolved_from_the_provider_when_unset() -> None:
    class _WindowProvider:
        def complete_chat(self, request: ChatRequest) -> ChatResponse:
            del request
            return _completion()

        def context_window(self) -> int | None:
            return 262_144

    ch = _FakeChannel([{"type": "done", "reason": "submit", "answer": "x"}])
    RunnerLink(ch, provider=cast(Any, _WindowProvider())).run("t1", "do it", _Env(), tools=_tools())
    assert _sent(ch, "episode_start")[0]["context_window"] == 262_144


def test_a_failing_context_window_probe_never_breaks_the_episode(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _AngryProvider:
        def complete_chat(self, request: ChatRequest) -> ChatResponse:
            del request
            return _completion()

        def context_window(self) -> int | None:
            raise RuntimeError("capability probe unavailable")

    ch = _FakeChannel([{"type": "done", "reason": "submit", "answer": "x"}])
    with caplog.at_level(logging.WARNING, logger=runner_link_module.__name__):
        result = RunnerLink(ch, provider=cast(Any, _AngryProvider())).run(
            "t1", "do it", _Env(), tools=_tools()
        )
    assert result.stop_reason is StopReason.SUBMITTED
    assert _sent(ch, "episode_start")[0]["context_window"] is None
    assert "could not resolve the served context window" in caplog.text


@pytest.mark.parametrize("value", [True, 512, "65536"])
def test_context_window_rejects_values_that_would_miscalibrate_the_guard(value: object) -> None:
    with pytest.raises(ValueError, match="context_window"):
        _link(_FakeChannel([]), context_window=value)
