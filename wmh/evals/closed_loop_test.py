"""End-to-end closed-loop tests: scripted agent + world model + judge, no network.

One provider plays all three roles (agent, world model, gold judge) by inspecting the system
prompt — the same fake-provider pattern the engine tests use.
"""

from __future__ import annotations

import threading

import pytest

from wmh.core.types import Action, ActionKind, EnvState, Observation, Step
from wmh.engine.world_model import WorldModel
from wmh.evals.closed_loop import (
    ClosedLoopReport,
    WorldModelEnvironment,
    WorldModelSource,
    evaluate_closed_loop,
    evaluate_with_env,
)
from wmh.evals.gold import AssertionResult, GoldJudge, GoldVerdict
from wmh.evals.tasks import TaskSpec
from wmh.harness.e2b_sandbox import SandboxCleanupError
from wmh.harness.environment import AgentEnvironment, is_env_action
from wmh.harness.runtime import RunResult, RuntimeCancelled, StopReason, TokenUsage
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder


class RoleProvider:
    """Plays agent, world model, and gold judge, keyed off the system prompt."""

    def __init__(self, *, judge_passes: bool = True) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")
        self._judge_passes = judge_passes

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        if "grade whether an agent completed a task" in system:
            passed = "true" if self._judge_passes else "false"
            return Completion(
                text='{"assertions": [{"assertion": "did it", "passed": '
                + passed
                + ', "why": "x"}], "passed": '
                + passed
                + "}"
            )
        if system.startswith("You are a capable command-line agent"):
            return Completion(
                text='{"tool": "submit", "arguments": {"answer": "the answer is 42"}}'
            )
        # world model
        return Completion(text='{"output": "ok", "is_error": false}')

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201 - test fake never calls it
        raise NotImplementedError


def _model(provider: RoleProvider) -> WorldModel:
    return WorldModel(provider, EmbeddingRetriever(HashingEmbedder(dim=16)))


def _wm(provider: RoleProvider) -> WorldModelSource:
    """The environment under test: a locally loaded world model driven by `provider`."""
    return WorldModelSource(_model(provider))


def test_gold_judge_no_assertions_trivially_passes() -> None:
    verdict = GoldJudge(RoleProvider()).score("task", "answer", "transcript", [])
    assert verdict == GoldVerdict.trivially_passed()


def test_closed_loop_scores_success_over_k_passes() -> None:
    provider = RoleProvider(judge_passes=True)
    tasks = [TaskSpec(task_id="q1", instruction="answer it", gold=["did it"])]
    report = evaluate_closed_loop(tasks, _wm(provider), provider, GoldJudge(provider), k=3)
    assert report.k == 3
    assert report.success_rate == 1.0
    outcome = report.per_task["q1"]
    assert outcome.passes == 3
    assert len(outcome.attempts) == 3
    assert all(attempt.answer == "the answer is 42" for attempt in outcome.attempts)
    assert all(attempt.stop_reason == StopReason.SUBMITTED for attempt in outcome.attempts)
    assert all("submit" in attempt.transcript for attempt in outcome.attempts)


def test_closed_loop_reports_failure_when_judge_rejects() -> None:
    provider = RoleProvider(judge_passes=False)
    tasks = [TaskSpec(task_id="q1", instruction="answer it", gold=["did it"])]
    report = evaluate_closed_loop(tasks, _wm(provider), provider, GoldJudge(provider), k=2)
    assert report.success_rate == 0.0
    assert report.per_task["q1"].mean_fraction == 0.0


def test_world_model_environment_steps_and_ends_session() -> None:
    provider = RoleProvider()
    wm = _model(provider)
    env = WorldModelEnvironment(wm, task="do a thing")
    session_id = env.session_id
    obs = env.execute(Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": "ls"}))
    assert obs.content == "ok"
    env.close()
    # The session is released on close (WorldModel.end_session drops it).
    try:
        wm.get_session(session_id)
    except KeyError:
        pass
    else:  # pragma: no cover
        raise AssertionError("session should be gone after close()")
    env.close()  # idempotent


def test_rollouts_do_not_enrich_the_retrieval_buffer() -> None:
    """A rollout's PREDICTED steps must not become retrieval demos for later rollouts."""
    provider = RoleProvider()
    wm = _model(provider)
    before = len(wm.sample_steps(1000))
    env = WorldModelEnvironment(wm, task="do a thing")
    env.execute(Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": "ls"}))
    env.execute(Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": "pwd"}))
    env.close()
    assert len(wm.sample_steps(1000)) == before  # buffer unchanged: eval sessions don't enrich
    # Serve-time sessions still enrich by default.
    session = wm.new_session(task="serve")
    wm.step(session.id, Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": "x"}))
    assert len(wm.sample_steps(1000)) == before + 1


def test_is_env_action_gates_tool_calls() -> None:
    assert is_env_action(Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={}))
    assert not is_env_action(Action(kind=ActionKind.TOOL_CALL, name="submit", arguments={}))
    assert not is_env_action(Action(kind=ActionKind.MESSAGE, content="hi"))


def test_evaluate_rejects_k_below_one() -> None:
    provider = RoleProvider()
    tasks = [TaskSpec(task_id="q", instruction="x", gold=[])]
    try:
        evaluate_closed_loop(tasks, _wm(provider), provider, GoldJudge(provider), k=0)
    except ValueError as exc:
        assert "k must be >= 1" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for k=0")


class _StaticEnv:
    """Env fake: answers every action with a constant observation; tracks close()."""

    def __init__(self) -> None:
        self.closed = False

    def execute(self, action: Action) -> Observation:
        return Observation(content="ok")

    def close(self) -> None:
        self.closed = True


class _ScriptedRuntime:
    """Runtime fake: deterministic answer keyed off the task_id (pure in its inputs)."""

    def run(self, task_id: str, instruction: str, environment: AgentEnvironment) -> RunResult:
        environment.execute(
            Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": "x"})
        )
        answer = "pass" if task_id.endswith("-pass") else "fail"
        return RunResult(task_id=task_id, stop_reason=StopReason.SUBMITTED, answer=answer)


class _AnswerJudge(GoldJudge):
    """Judge fake: the verdict is read straight off the runtime's answer (no LLM call)."""

    def __init__(self) -> None:
        super().__init__(RoleProvider())

    def score(self, instruction: str, answer: str, transcript: str, gold: list[str]) -> GoldVerdict:
        passed = answer == "pass"
        return GoldVerdict(passed=passed, fraction=1.0 if passed else 0.0, rationale=answer)


def _parity_tasks() -> list[TaskSpec]:
    return [
        TaskSpec(task_id="a-pass", instruction="do a", gold=["g"]),
        TaskSpec(task_id="b-fail", instruction="do b", gold=["g"]),
    ]


def test_concurrent_report_matches_sequential() -> None:
    """concurrency=0 (all cells) and =2 (capped) must reproduce the sequential report exactly."""

    def run(concurrency: int) -> ClosedLoopReport:
        return evaluate_with_env(
            _parity_tasks(),
            lambda task: _StaticEnv(),
            _ScriptedRuntime(),
            _AnswerJudge(),
            label="parity",
            k=3,
            concurrency=concurrency,
        )

    sequential = run(1)
    assert sequential.per_task["a-pass"].success_rate == 1.0
    assert sequential.per_task["b-fail"].success_rate == 0.0
    for concurrency in (0, 2):
        assert run(concurrency).model_dump() == sequential.model_dump()


def test_rollout_evidence_is_bounded_without_losing_trace_tail() -> None:
    class LongTraceRuntime:
        def run(self, task_id: str, instruction: str, environment: AgentEnvironment) -> RunResult:
            del environment
            steps = [
                Step(
                    action=Action(
                        kind=ActionKind.TOOL_CALL,
                        name="bash",
                        arguments={"command": f"step-{index}"},
                    ),
                    observation=Observation(content=f"obs-{index}-" + ("x" * 2_000)),
                    state_before=EnvState(),
                    task=instruction,
                )
                for index in range(30)
            ]
            return RunResult(
                task_id=task_id,
                steps=steps,
                stop_reason=StopReason.SUBMITTED,
                answer="pass",
                turns=len(steps),
            )

    report = evaluate_with_env(
        [TaskSpec(task_id="a-pass", instruction="long", gold=["g"])],
        lambda _task: _StaticEnv(),
        LongTraceRuntime(),
        _AnswerJudge(),
        k=1,
    )

    trace = report.per_task["a-pass"].attempts[0].transcript
    assert len(trace) < 13_000
    assert "trace characters omitted" in trace
    assert "step-0" in trace
    assert "step-29" in trace


def test_rollout_and_judge_evidence_is_safe_for_utf8_project_files() -> None:
    class InvalidTextRuntime:
        def run(self, task_id: str, instruction: str, environment: AgentEnvironment) -> RunResult:
            del instruction, environment
            return RunResult(
                task_id=task_id,
                stop_reason=StopReason.SUBMITTED,
                answer="before\x00after",
                steps=[
                    Step(
                        action=Action(kind=ActionKind.MESSAGE, content="before\ud800after"),
                        observation=Observation(content="before\udcffafter"),
                    )
                ],
            )

    report = evaluate_with_env(
        [TaskSpec(task_id="t", instruction="i", gold=["g"])],
        lambda _task: _StaticEnv(),
        InvalidTextRuntime(),
        _AnswerJudge(),
        k=1,
    )

    replacement = "\N{REPLACEMENT CHARACTER}"
    attempt = report.per_task["t"].attempts[0]
    assert attempt.answer == f"before{replacement}after"
    assert "\x00" not in attempt.transcript
    assert not any(0xD800 <= ord(char) <= 0xDFFF for char in attempt.transcript)

    assertion = AssertionResult(
        assertion="before\x00after",
        passed=False,
        why="before\ud800after",
    )
    verdict = GoldVerdict(rationale="before\udcffafter", assertions=[assertion])
    assert verdict.assertions[0].assertion == f"before{replacement}after"
    assert verdict.assertions[0].why == f"before{replacement}after"
    assert verdict.rationale == f"before{replacement}after"


def test_concurrency_zero_overlaps_all_cells() -> None:
    """With concurrency=0 every (task, attempt) cell is in flight at once: a barrier sized to the
    full cell count releases only under true parallelism (any smaller pool would deadlock and trip
    the barrier timeout)."""
    tasks = [
        TaskSpec(task_id="a-pass", instruction="do a", gold=["g"]),
        TaskSpec(task_id="b-pass", instruction="do b", gold=["g"]),
    ]
    k = 2
    barrier = threading.Barrier(len(tasks) * k)

    class BarrierRuntime:
        def run(self, task_id: str, instruction: str, environment: AgentEnvironment) -> RunResult:
            barrier.wait(timeout=30)  # BrokenBarrierError (test failure) unless all cells overlap
            return RunResult(task_id=task_id, stop_reason=StopReason.SUBMITTED, answer="pass")

    report = evaluate_with_env(
        tasks, lambda task: _StaticEnv(), BarrierRuntime(), _AnswerJudge(), k=k, concurrency=0
    )
    assert report.success_rate == 1.0
    assert barrier.broken is False


def test_on_progress_fires_once_per_cell_from_the_coordinating_thread() -> None:
    events: list[tuple[str, int, bool]] = []
    coordinator = threading.get_ident()
    callback_threads: set[int] = set()

    def on_progress(task_id: str, attempt: int, verdict: GoldVerdict) -> None:
        callback_threads.add(threading.get_ident())
        events.append((task_id, attempt, verdict.passed))

    evaluate_with_env(
        _parity_tasks(),
        lambda task: _StaticEnv(),
        _ScriptedRuntime(),
        _AnswerJudge(),
        k=3,
        concurrency=0,
        on_progress=on_progress,
    )
    assert len(events) == 6  # exactly once per (task, attempt) cell
    assert set(events) == {
        (task.task_id, attempt, task.task_id.endswith("-pass"))
        for task in _parity_tasks()
        for attempt in (1, 2, 3)
    }
    assert callback_threads == {coordinator}  # serial UI stream, never from worker threads


def test_rollout_exception_propagates_and_does_not_hang() -> None:
    class ExplodingRuntime:
        def run(self, task_id: str, instruction: str, environment: AgentEnvironment) -> RunResult:
            if task_id == "boom":
                raise RuntimeError("sandbox died")
            return RunResult(task_id=task_id, stop_reason=StopReason.SUBMITTED, answer="pass")

    tasks = [
        TaskSpec(task_id="boom", instruction="x", gold=["g"]),
        TaskSpec(task_id="ok-pass", instruction="y", gold=["g"]),
    ]
    with pytest.raises(RuntimeError, match="sandbox died"):
        evaluate_with_env(
            tasks, lambda task: _StaticEnv(), ExplodingRuntime(), _AnswerJudge(), k=2, concurrency=0
        )


@pytest.mark.parametrize("concurrency", [1, 0])
def test_cancellation_after_rollout_skips_the_judge(concurrency: int) -> None:
    cancelled = threading.Event()

    class CancellingRuntime:
        def run(self, task_id: str, instruction: str, environment: AgentEnvironment) -> RunResult:
            cancelled.set()
            return RunResult(task_id=task_id, stop_reason=StopReason.SUBMITTED, answer="pass")

    class CountingJudge(_AnswerJudge):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def score(
            self, instruction: str, answer: str, transcript: str, gold: list[str]
        ) -> GoldVerdict:
            self.calls += 1
            return super().score(instruction, answer, transcript, gold)

    judge = CountingJudge()
    with pytest.raises(RuntimeCancelled, match="cancelled"):
        evaluate_with_env(
            [TaskSpec(task_id="cancel", instruction="x", gold=["g"])],
            lambda task: _StaticEnv(),
            CancellingRuntime(),
            judge,
            k=1,
            concurrency=concurrency,
            should_cancel=cancelled.is_set,
        )

    assert judge.calls == 0


@pytest.mark.parametrize("concurrency", [1, 0])
def test_budget_result_with_partial_transcript_is_still_judged(concurrency: int) -> None:
    class BudgetRuntime:
        def run(self, task_id: str, instruction: str, environment: AgentEnvironment) -> RunResult:
            return RunResult(
                task_id=task_id,
                stop_reason=StopReason.BUDGET,
                steps=[
                    Step(
                        action=Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={}),
                        observation=Observation(content="partial work"),
                        state_before=EnvState(),
                        task=instruction,
                    )
                ],
            )

    class CapturingJudge(_AnswerJudge):
        def __init__(self) -> None:
            super().__init__()
            self.transcripts: list[str] = []

        def score(
            self, instruction: str, answer: str, transcript: str, gold: list[str]
        ) -> GoldVerdict:
            self.transcripts.append(transcript)
            return super().score(instruction, answer, transcript, gold)

    judge = CapturingJudge()
    evaluate_with_env(
        [TaskSpec(task_id="budget", instruction="x", gold=["g"])],
        lambda task: _StaticEnv(),
        BudgetRuntime(),
        judge,
        k=1,
        concurrency=concurrency,
    )

    assert len(judge.transcripts) == 1
    assert "partial work" in judge.transcripts[0]


def test_rollout_exception_drains_other_started_cells_before_returning() -> None:
    """A failed wave cannot terminalize while another cell still owns billable work."""
    release = threading.Event()
    blocker_started = threading.Event()
    failure_raised = threading.Event()
    blocker_finished = threading.Event()
    abort_called = threading.Event()
    done = threading.Event()
    errors: list[BaseException] = []

    class HalfStuckRuntime:
        def run(self, task_id: str, instruction: str, environment: AgentEnvironment) -> RunResult:
            if task_id == "stuck":
                blocker_started.set()
                release.wait(timeout=30)
                blocker_finished.set()
                return RunResult(task_id=task_id, stop_reason=StopReason.SUBMITTED, answer="pass")
            blocker_started.wait(timeout=30)  # raise only once the slow cell is truly in flight
            failure_raised.set()
            raise RuntimeError("sandbox died")

        def abort(self) -> None:
            abort_called.set()

    tasks = [
        TaskSpec(task_id="stuck", instruction="x", gold=["g"]),
        TaskSpec(task_id="boom", instruction="y", gold=["g"]),
    ]

    def evaluate() -> None:
        try:
            evaluate_with_env(
                tasks,
                lambda task: _StaticEnv(),
                HalfStuckRuntime(),
                _AnswerJudge(),
                k=1,
                concurrency=0,
            )
        except BaseException as error:  # noqa: BLE001 - assert exact failure below
            errors.append(error)
        finally:
            done.set()

    worker = threading.Thread(target=evaluate)
    worker.start()
    try:
        assert failure_raised.wait(timeout=5)
        assert abort_called.wait(timeout=5)
        assert not done.wait(timeout=0.1)
        assert not blocker_finished.is_set()
        release.set()
        assert done.wait(timeout=5)
        assert blocker_finished.is_set()
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        assert str(errors[0]) == "sandbox died"
    finally:
        release.set()
        worker.join(timeout=5)


def test_external_cancellation_aborts_blocked_cells_before_joining() -> None:
    """Caller polling can stop cells that have not reached runtime cancellation checks."""
    started = threading.Event()
    released = threading.Event()
    abort_called = threading.Event()
    cancelled = threading.Event()
    done = threading.Event()
    errors: list[BaseException] = []

    class BlockedRuntime:
        def run(self, task_id: str, instruction: str, environment: AgentEnvironment) -> RunResult:
            del instruction, environment
            started.set()
            released.wait(timeout=30)
            return RunResult(task_id=task_id, stop_reason=StopReason.SUBMITTED, answer="pass")

        def abort(self) -> None:
            abort_called.set()
            released.set()

    def evaluate() -> None:
        try:
            evaluate_with_env(
                [TaskSpec(task_id="blocked", instruction="x", gold=["g"])],
                lambda task: _StaticEnv(),
                BlockedRuntime(),
                _AnswerJudge(),
                k=1,
                concurrency=0,
                should_cancel=cancelled.is_set,
            )
        except BaseException as error:  # noqa: BLE001 - assert exact type below
            errors.append(error)
        finally:
            done.set()

    worker = threading.Thread(target=evaluate)
    worker.start()
    try:
        assert started.wait(timeout=5)
        cancelled.set()
        assert abort_called.wait(timeout=5)
        assert done.wait(timeout=5)
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeCancelled)
    finally:
        released.set()
        worker.join(timeout=5)


def test_abort_cleanup_is_rechecked_after_worker_release_drain() -> None:
    """A transient abort failure does not survive a clean post-drain retry."""
    blocker_started = threading.Event()
    first_abort = threading.Event()
    abort_calls = 0

    class RetryCleanRuntime:
        def run(self, task_id: str, instruction: str, environment: AgentEnvironment) -> RunResult:
            del instruction, environment
            if task_id == "blocked":
                blocker_started.set()
                first_abort.wait(timeout=30)
                return RunResult(
                    task_id=task_id,
                    stop_reason=StopReason.SUBMITTED,
                    answer="pass",
                )
            blocker_started.wait(timeout=30)
            raise RuntimeError("cell failed")

        def abort(self) -> None:
            nonlocal abort_calls
            abort_calls += 1
            first_abort.set()
            if abort_calls == 1:
                raise SandboxCleanupError("first cleanup attempt failed")

    with pytest.raises(RuntimeError, match="cell failed"):
        evaluate_with_env(
            [
                TaskSpec(task_id="blocked", instruction="x", gold=["g"]),
                TaskSpec(task_id="failed", instruction="y", gold=["g"]),
            ],
            lambda task: _StaticEnv(),
            RetryCleanRuntime(),
            _AnswerJudge(),
            k=1,
            concurrency=0,
        )
    assert abort_calls == 2


def test_rollout_cancellation_drains_already_started_cells() -> None:
    """Cancellation waits for bounded in-flight work so its usage can finalize."""
    blocker_started = threading.Event()
    cancellation_raised = threading.Event()
    release = threading.Event()
    late_finished = threading.Event()
    done = threading.Event()
    errors: list[BaseException] = []

    class CancellingRuntime:
        def run(self, task_id: str, instruction: str, environment: AgentEnvironment) -> RunResult:
            del instruction, environment
            if task_id == "stuck":
                blocker_started.set()
                release.wait(timeout=30)
                late_finished.set()
                return RunResult(
                    task_id=task_id,
                    stop_reason=StopReason.SUBMITTED,
                    answer="pass",
                    worker_usage=TokenUsage(input_tokens=11, output_tokens=3, calls=1),
                )
            blocker_started.wait(timeout=30)
            cancellation_raised.set()
            raise RuntimeCancelled(
                "cancelled",
                worker_usage=TokenUsage(input_tokens=7, output_tokens=2, calls=1),
            )

    def evaluate() -> None:
        try:
            evaluate_with_env(
                [
                    TaskSpec(task_id="stuck", instruction="x", gold=["g"]),
                    TaskSpec(task_id="cancel", instruction="y", gold=["g"]),
                ],
                lambda task: _StaticEnv(),
                CancellingRuntime(),
                _AnswerJudge(),
                k=1,
                concurrency=0,
            )
        except BaseException as error:  # noqa: BLE001 - assert exact type below
            errors.append(error)
        finally:
            done.set()

    worker = threading.Thread(target=evaluate)
    worker.start()
    try:
        assert cancellation_raised.wait(timeout=5)
        assert not done.wait(timeout=0.1)
        assert not late_finished.is_set()
        release.set()
        assert done.wait(timeout=5)
        assert late_finished.is_set()
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeCancelled)
        usage = errors[0].worker_usage
        assert usage is not None
        assert usage.input_tokens == 18
        assert usage.output_tokens == 5
        assert usage.calls == 2
    finally:
        release.set()
        worker.join(timeout=5)


def test_evaluate_closed_loop_passes_concurrency_through() -> None:
    provider = RoleProvider(judge_passes=True)
    tasks = [TaskSpec(task_id="q1", instruction="answer it", gold=["did it"])]
    report = evaluate_closed_loop(
        tasks, _wm(provider), provider, GoldJudge(provider), k=2, concurrency=2
    )
    assert report.success_rate == 1.0
    assert report.per_task["q1"].passes == 2


def test_report_aggregates_worker_usage_from_self_metering_runtimes() -> None:
    """Cells that report worker usage sum into the report; none reported -> None."""
    from wmh.harness.runtime import TokenUsage

    class MeteredRuntime:
        def run(self, task_id: str, instruction: str, environment: AgentEnvironment) -> RunResult:
            return RunResult(
                task_id=task_id,
                stop_reason=StopReason.SUBMITTED,
                answer="pass",
                worker_usage=TokenUsage(input_tokens=10, output_tokens=3, calls=2),
            )

    tasks = [TaskSpec(task_id="a-pass", instruction="x", gold=["g"])]
    for concurrency in (1, 0):  # both aggregation paths
        report = evaluate_with_env(
            tasks,
            lambda task: _StaticEnv(),
            MeteredRuntime(),
            _AnswerJudge(),
            k=2,
            concurrency=concurrency,
        )
        assert report.worker_usage is not None
        assert report.worker_usage.input_tokens == 20  # 2 cells x 10
        assert report.worker_usage.output_tokens == 6
        assert report.worker_usage.calls == 4
    # Provider-wrapped runtimes report nothing -> the report says None, not zero.
    silent = evaluate_with_env(
        tasks, lambda task: _StaticEnv(), _ScriptedRuntime(), _AnswerJudge(), k=1
    )
    assert silent.worker_usage is None
