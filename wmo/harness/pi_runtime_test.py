"""PiRuntime unit tests: schema + tool routing/budget, offline (no ssh/node)."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from socket import socket
from typing import cast

import pytest
from llm_waterfall import ChatRequest, ChatResponse

from wmo.core.types import Action, Observation
from wmo.harness import pi_runtime as pi_runtime_module
from wmo.harness.doc import (
    MAX_OUTPUT_TOKENS_ID,
    MAX_TURNS_ID,
    RUNTIME_KIND_ID,
    TEMPERATURE_ID,
    TOOL_POLICY_ID,
    HarnessDoc,
    Surface,
    SurfaceKind,
)
from wmo.harness.pi_runtime import (
    PiRuntime,
    _Episode,
    _params_schema,
    _ShimHandler,
    _ShimServer,
)
from wmo.harness.runtime import StopReason
from wmo.harness.skills import Skill, SkillLibrary
from wmo.harness.tools import SUBMIT, TOOL_REGISTRY
from wmo.providers.base import Provider


class _Env:
    def __init__(self) -> None:
        self.actions: list[Action] = []

    def execute(self, action: Action) -> Observation:
        self.actions.append(action)
        return Observation(content=f"ran {action.name}")

    def close(self) -> None:
        pass


class _Provider:
    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        del request
        return ChatResponse.model_validate(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )


def _episode(
    env: _Env, *, budget: int = 40, skills: SkillLibrary | None = None, temperature: float = 0.7
) -> _Episode:
    return _Episode(
        instruction="do it",
        system_prompt="sys",
        tools=[TOOL_REGISTRY["bash"], SUBMIT],
        provider=_Provider(),
        environment=env,
        temperature=temperature,
        skills=skills if skills is not None else SkillLibrary(),
        max_env_actions=budget,
        max_turns=7,
        max_output_tokens=16384,
    )


def test_task_json_shape() -> None:
    ep = _episode(_Env())
    import json

    tj = json.loads(json.dumps(ep.task_json()))
    assert tj["instruction"] == "do it" and tj["system"] == "sys"
    assert tj["max_turns"] == 7
    assert tj["max_output_tokens"] == 16384
    names = {t["name"] for t in tj["tools"]}
    assert "bash" in names and "submit" in names
    schema = json.loads(json.dumps(_params_schema(TOOL_REGISTRY["bash"])))
    assert schema["type"] == "object" and "command" in schema["properties"]


def test_tool_routing_records_steps_and_rejects_unknown() -> None:
    env = _Env()
    ep = _episode(env)
    ok = ep.run_tool("bash", {"command": "ls"})
    assert ok == {"content": "ran bash", "is_error": False}
    bad = ep.run_tool("rm_rf", {})
    assert bad["is_error"] is True and env.actions == [ep.steps[0].action]
    assert len(ep.steps) == 2  # both recorded (the transcript the judge sees)


def test_env_action_budget_is_enforced() -> None:
    env = _Env()
    ep = _episode(env, budget=2)
    for _ in range(4):
        ep.run_tool("bash", {"command": "true"})
    assert len(env.actions) == 2  # only the budgeted calls reached the environment
    assert "budget exhausted" in ep.steps[-1].observation.content


def test_worker_request_uses_document_temperature() -> None:
    request = _episode(_Env(), temperature=0.35).worker_request(
        {"messages": [], "temperature": 1.75}
    )
    assert request.temperature == 0.35


def test_worker_completion_usage_keeps_exact_per_call_counters() -> None:
    ep = _episode(_Env())
    completion = ChatResponse.model_validate(
        {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_tokens_details": {
                    "cached_tokens": 40,
                    "cache_write_tokens": 10,
                },
                "completion_tokens_details": {"reasoning_tokens": 7},
            },
        }
    )

    ep.record_worker_completion(completion, 0.25)

    assert ep.worker_usage.model_dump(mode="json") == {
        "input_tokens": 100,
        "output_tokens": 20,
        "cached_input_tokens": 40,
        "cache_write_input_tokens": 10,
        "reasoning_tokens": 7,
        "calls": 1,
        "call_seconds": [0.25],
        "call_input_tokens": [100],
        "call_output_tokens": [20],
        "call_cached_input_tokens": [40],
        "call_cache_write_input_tokens": [10],
    }


def test_error_result_keeps_partial_worker_usage() -> None:
    ep = _episode(_Env())
    ep.record_worker_failure(0.5)

    result = PiRuntime._error_result(  # noqa: SLF001 - partial spend is the contract
        "task",
        ep,
        "do it",
        "provider failed",
        StopReason.PROVIDER_ERROR,
    )

    assert result.worker_usage is not None
    assert result.worker_usage.calls == 1
    assert result.worker_usage.call_seconds == [0.5]


def test_read_skill_is_runtime_local_and_does_not_consume_environment_budget() -> None:
    env = _Env()
    skills = SkillLibrary(
        [Skill(name="count-words", description="count words", body="wc -w <path>")]
    )
    ep = _episode(env, budget=0, skills=skills)
    ep.tools.append(TOOL_REGISTRY["read_skill"])

    found = ep.run_tool("read_skill", {"name": "count-words"})
    missing = ep.run_tool("read_skill", {"name": "ghost"})

    assert found == {"content": "wc -w <path>", "is_error": False}
    assert missing == {"content": "no skill named 'ghost'", "is_error": True}
    assert env.actions == []


def test_local_shim_close_waits_for_active_environment_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler_started = threading.Event()
    release_handler = threading.Event()
    close_finished = threading.Event()
    server = _ShimServer(("127.0.0.1", 0), _ShimHandler, bind_and_activate=False)

    def blocking_request(_request: object, _client_address: object) -> None:
        handler_started.set()
        assert release_handler.wait(timeout=2)

    monkeypatch.setattr(server, "process_request_thread", blocking_request)
    server.process_request(cast("socket", object()), ("127.0.0.1", 1))
    assert handler_started.wait(timeout=1)

    def close_server() -> None:
        server.server_close()
        close_finished.set()

    close_thread = threading.Thread(target=close_server)
    close_thread.start()
    try:
        time.sleep(0.05)
        assert _ShimServer.daemon_threads is False
        assert close_finished.is_set() is False
        assert close_thread.is_alive()
    finally:
        release_handler.set()
        close_thread.join(timeout=2)

    assert close_thread.is_alive() is False
    assert close_finished.is_set()


def test_doc_dispatches_pi_runtime_for_pi_node_kind() -> None:
    from wmo.providers.base import ProviderConfig, ProviderKind

    class _P:
        config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")

        def complete(self, *a, **k) -> object:  # noqa: ANN002, ANN003
            raise NotImplementedError

        def complete_chat(self, request: ChatRequest) -> ChatResponse:
            return _Provider().complete_chat(request)

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
            Surface(id=TEMPERATURE_ID, kind=SurfaceKind.PARAM, content="0.35"),
            Surface(
                id="skill:count-words",
                kind=SurfaceKind.SKILL,
                content=Skill(
                    name="count-words", description="count words", body="wc -w <path>"
                ).to_markdown(),
            ),
            Surface(
                id="code:src-agent-ts",
                kind=SurfaceKind.CODE,
                path="src/agent.ts",
                content="// agent",
            ),
        ],
    )
    assert doc.runtime_kind() == "pi-node"
    assert [s.path for s in doc.code_files()] == ["src/agent.ts"]
    from typing import cast

    runtime = doc.runtime(cast("Provider", _P()))
    assert isinstance(runtime, PiRuntime)
    assert runtime._max_turns == 7  # noqa: SLF001 - document parameter reaches entry.ts
    assert runtime._max_output_tokens == 16384  # noqa: SLF001 - same agent model contract
    assert runtime._temperature == 0.35  # noqa: SLF001 - same worker sampling contract
    assert {tool.name for tool in runtime._tools} >= {  # noqa: SLF001 - runtime plumbing
        "bash",
        "submit",
        "read_skill",
    }


# --- termination disambiguation on the SSH shim path (audit defect 1) ----------------------------
def test_signal_json_reports_the_hosts_view_of_the_last_completion() -> None:
    """entry.ts reads GET /signal to classify why it finished, so the host must record it.

    The shim owns the provider call, so only it can see finish_reason "length" (truncated at the
    output cap) and the renderer's tool-call parse errors; entry.ts sees neither on its own.
    """
    ep = _episode(_Env())
    assert ep.signal_json() == {
        "finish_reason": "",
        "unparsed_tool_calls": [],
        "provider_error": "",
        "tool_call_turns": 0,
    }
    ep.finish_reason = "length"
    ep.unparsed_tool_calls = ["Unexpected trailing content inside <function> block"]
    ep.tool_call_turns = 3
    assert ep.signal_json() == {
        "finish_reason": "length",
        "unparsed_tool_calls": ["Unexpected trailing content inside <function> block"],
        "provider_error": "",
        "tool_call_turns": 3,
    }


def test_task_json_carries_the_served_context_window() -> None:
    ep = _Episode(
        instruction="do it",
        system_prompt="sys",
        tools=[SUBMIT],
        provider=_Provider(),
        environment=_Env(),
        temperature=0.7,
        skills=SkillLibrary(),
        max_env_actions=40,
        max_turns=7,
        max_output_tokens=16384,
        context_window=262_144,
    )
    assert ep.task_json()["context_window"] == 262_144


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("submit", StopReason.SUBMITTED),
        ("no_tool_call", StopReason.NO_TOOL_CALL),
        ("output_truncated", StopReason.OUTPUT_TRUNCATED),
        ("unparsed_tool_call", StopReason.UNPARSED_TOOL_CALL),
    ],
)
def test_run_maps_the_shim_done_reason_to_a_distinct_stop_reason(
    monkeypatch: pytest.MonkeyPatch, reason: str, expected: StopReason
) -> None:
    """Only an explicit submit is a completion, on this transport too."""
    runtime = PiRuntime(
        cast("Provider", _Provider()),
        files={"src/agent.ts": "// a"},
        tools=[TOOL_REGISTRY["bash"], SUBMIT],
        port=8899,
    )
    monkeypatch.setattr(PiRuntime, "_materialize", lambda self, workdir: None)

    def fake_run_node(self: PiRuntime, port: int, workdir: str) -> tuple[int, str]:
        del self, port, workdir
        # Stand in for entry.ts POSTing /done with its classified reason.
        episode.answer = "text"
        episode.done_reason = reason
        episode.done.set()
        return 0, ""

    episode = _Episode(
        instruction="do it",
        system_prompt="",
        tools=[SUBMIT],
        provider=_Provider(),
        environment=_Env(),
        temperature=0.7,
        skills=SkillLibrary(),
        max_env_actions=40,
        max_turns=7,
        max_output_tokens=4096,
    )
    monkeypatch.setattr(PiRuntime, "_run_node", fake_run_node)
    monkeypatch.setattr(
        "wmo.harness.pi_runtime._Episode", lambda **kwargs: _capture_episode(episode, kwargs)
    )

    result = runtime.run("t1", "do it", _Env())
    assert result.stop_reason is expected


def _capture_episode(episode: _Episode, kwargs: dict[str, object]) -> _Episode:
    """Return the test's episode so the fake node run can drive it."""
    del kwargs
    return episode


def test_the_node_wall_budget_comes_from_the_configured_episode_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`timeout 300 node` was hardcoded, so a configured budget was silently ignored and 31% of
    long TerminalBench-2 trials died on that clock."""
    commands: list[list[str]] = []

    class _Completed:
        returncode = 0
        stderr = ""

    def fake_run(command: list[str], **kwargs: object) -> _Completed:
        commands.append(command)
        assert kwargs["timeout"] == pytest.approx(1800.0 + 60.0)
        assert (
            pi_runtime_module._NODE_HARD_KILL_AFTER_S  # noqa: SLF001 - timeout contract
            < pi_runtime_module._NODE_TEARDOWN_GRACE_S  # noqa: SLF001 - timeout contract
        )
        return _Completed()

    runtime = PiRuntime(
        cast("Provider", _Provider()),
        files={"src/agent.ts": "// a"},
        tools=[SUBMIT],
        port=8898,
        episode_timeout_s=1800.0,
    )
    monkeypatch.setattr("wmo.harness.pi_runtime.subprocess.run", fake_run)

    code, _note = runtime._run_node(8898, "~/pi-run/ep-8898")  # noqa: SLF001

    assert code == 0
    assert "StrictHostKeyChecking=accept-new" in commands[0]
    assert (
        "timeout --kill-after=10 1800 node --experimental-strip-types entry.ts" in commands[0][-1]
    )


@pytest.mark.parametrize(
    ("budget", "expected"),
    [(0.5, 1), (1.9, 2), (1800.0, 1800)],
)
def test_a_fractional_node_wall_budget_rounds_up_instead_of_truncating(
    monkeypatch: pytest.MonkeyPatch, budget: float, expected: int
) -> None:
    """A fractional budget must never truncate, and never reach `timeout 0`.

    GNU coreutils reads `timeout 0` as "no timeout at all", so truncating a sub-second budget
    would remove the node wall entirely and leave termination to the outer SSH deadline plus
    its teardown grace, an order of magnitude past what the caller asked for.
    """
    commands: list[list[str]] = []

    class _Completed:
        returncode = 0
        stderr = ""

    def fake_run(command: list[str], **kwargs: object) -> _Completed:
        del kwargs
        commands.append(command)
        return _Completed()

    runtime = PiRuntime(
        cast("Provider", _Provider()),
        files={"src/agent.ts": "// a"},
        tools=[SUBMIT],
        port=8899,
        episode_timeout_s=budget,
    )
    monkeypatch.setattr("wmo.harness.pi_runtime.subprocess.run", fake_run)

    runtime._run_node(8899, "~/pi-run/ep-8899")  # noqa: SLF001

    assert (
        f"timeout --kill-after=10 {expected} node --experimental-strip-types entry.ts"
        in commands[0][-1]
    )


def test_default_runtime_ports_and_workdirs_are_isolated_across_parallel_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Independent local runtimes must not share the default shim port or runner directory."""
    barrier = threading.Barrier(2)
    endpoints: list[tuple[int, str]] = []
    endpoint_lock = threading.Lock()

    monkeypatch.setattr(PiRuntime, "_materialize", lambda self, workdir: None)

    def fake_run_node(self: PiRuntime, port: int, workdir: str) -> tuple[int, str]:
        del self
        with endpoint_lock:
            endpoints.append((port, workdir))
        barrier.wait(timeout=5)
        return 0, ""

    monkeypatch.setattr(PiRuntime, "_run_node", fake_run_node)
    runtimes = [
        PiRuntime(
            cast("Provider", _Provider()),
            files={"src/agent.ts": "// a"},
            tools=[SUBMIT],
        )
        for _ in range(2)
    ]

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda runtime: runtime.run("t1", "do it", _Env()), runtimes))

    assert len(results) == 2
    assert len({port for port, _workdir in endpoints}) == 2
    assert len({workdir for _port, workdir in endpoints}) == 2
    assert all(port > 0 and workdir.endswith(f"ep-{port}") for port, workdir in endpoints)


def test_the_shared_termination_policy_is_materialized_next_to_entry_ts() -> None:
    """entry.ts imports ./runner_termination.ts, so the SSH blob must carry it."""
    assert Path(pi_runtime_module._TERMINATION_TS).is_file()  # noqa: SLF001 - deploy contract
    source = Path(pi_runtime_module.__file__).read_text(encoding="utf-8")
    assert '"runner_termination.ts": _read(_TERMINATION_TS)' in source
