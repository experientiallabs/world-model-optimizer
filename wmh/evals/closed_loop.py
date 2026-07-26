"""Closed-loop scoring: run a live agent on tasks against an environment, judge task success.

Open-loop eval (`wmh/engine/eval.py`, the default `wmh eval` mode) replays recorded steps
teacher-forced and scores per-step fidelity. This module is the closed-loop counterpart
(`wmh eval --mode closed-loop`): for each task, the agent loop runs to completion (submit or turn
cap) and the `GoldJudge` scores the transcript against the task's gold assertions. Per the repo's
eval convention, every task runs **k=3 passes** and metrics are means over the passes — never
single-pass.

The environment is a factory parameter: `evaluate_closed_loop` binds it to the world model
(`WorldModelEnvironment`), and any real execution backend can bind the same core through the
`AgentEnvironment` protocol, producing a directly comparable report (see
`wmh.evals.agreement`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from statistics import fmean, pstdev

from pydantic import BaseModel, Field

from wmh.core.text import normalize_durable_text
from wmh.core.types import Action, Observation
from wmh.engine.world_model import WorldModel
from wmh.evals.gold import GoldJudge, GoldVerdict
from wmh.evals.tasks import TaskSpec
from wmh.harness.environment import AgentEnvironment, EnvironmentSource
from wmh.harness.runtime import (
    AgentRuntime,
    RunResult,
    Runtime,
    RuntimeCancelled,
    StopReason,
    TokenUsage,
    combine_usage,
)
from wmh.providers.base import Provider

DEFAULT_K = 3  # eval-reporting convention: every metric is the mean of k passes, never single-pass
_ROLLOUT_EVIDENCE_CHARS = 12_000
_ROLLOUT_ANSWER_CHARS = 4_000

# Opens a fresh environment for one task. The world-model backend and any real backend both fit
# this shape, which is what lets the SAME scoring core measure simulation and reality.
EnvFactory = Callable[[TaskSpec], AgentEnvironment]


class WorldModelEnvironment:
    """A simulated environment: actions are answered by the world model, not a real shell.

    Wraps one `WorldModel` session, so the agent loop drives closed-loop eval exactly as it would
    drive a real environment. Sessions are explicitly ended on `close` so batch rollouts don't
    accumulate resident session state in the model.
    """

    def __init__(self, world_model: WorldModel, task: str) -> None:
        self._wm = world_model
        # enrich=False: this rollout's PREDICTED steps must not enter the retrieval buffer, or
        # k=2 retrieves k=1's hallucinations as demos and scores become order-dependent.
        self._session = world_model.new_session(task=task, enrich=False)
        self._closed = False

    @property
    def session_id(self) -> str:
        return self._session.id

    def execute(self, action: Action) -> Observation:
        return self._wm.step(self._session.id, action)

    def close(self) -> None:
        if not self._closed:
            self._wm.end_session(self._session.id)
            self._closed = True


class WorldModelSource:
    """The `EnvironmentSource` backed by a world model loaded in this process."""

    def __init__(self, world_model: WorldModel, *, label: str = "world-model") -> None:
        """Wrap `world_model`; `label` names it in reports and CLI output."""
        self._wm = world_model
        self._label = label

    @property
    def label(self) -> str:
        return self._label

    @property
    def world_model(self) -> WorldModel:
        """The wrapped model, for callers that need its provider or corpus."""
        return self._wm

    def open(self, task: str) -> AgentEnvironment:
        return WorldModelEnvironment(self._wm, task=task)

    @contextmanager
    def frozen(self) -> Iterator[None]:
        """Freeze the model for the duration: concurrent rollouts must not mutate its index."""
        with self._wm.frozen():
            yield


class RolloutEvidence(BaseModel):
    """The execution evidence behind one judged pass.

    Aggregate scores are sufficient for ranking, but not for improving a harness: the proposer
    needs to see what the agent actually tried, what the environment returned, and why the run
    stopped. Keeping this alongside each verdict lets every optimizer backend expose the same
    trace-level feedback without reaching into a platform-specific rollout store.
    """

    answer: str = ""
    transcript: str = ""
    stop_reason: StopReason
    turns: int = 0


class TaskOutcome(BaseModel):
    """One task's closed-loop result across k passes."""

    task_id: str
    success_rate: float = 0.0  # fraction of k passes that fully passed gold
    mean_fraction: float = 0.0  # mean fraction-of-assertions across passes (partial credit)
    passes: int = 0
    verdicts: list[GoldVerdict] = Field(default_factory=list)
    attempts: list[RolloutEvidence] = Field(default_factory=list)


class ClosedLoopReport(BaseModel):
    """A closed-loop scorecard over a task suite.

    `label` names what produced the report (a world model name, or a real environment) so two
    reports compared by `compute_agreement` stay identifiable.
    """

    label: str = ""
    success_rate: float = 0.0  # mean over tasks of per-task pass rate
    mean_fraction: float = 0.0  # mean over tasks of mean assertion-fraction (denser signal)
    success_std: float = 0.0  # spread of per-task success rates
    k: int = DEFAULT_K
    per_task: dict[str, TaskOutcome] = Field(default_factory=dict)
    # Aggregate worker-LLM spend from runtimes that meter it themselves (the pi worker path);
    # None when every rollout came from a provider-wrapped runtime (metered upstream).
    worker_usage: TokenUsage | None = None

    @property
    def headline(self) -> float:
        """The `EvalResult` headline: end-to-end task success."""
        return self.success_rate

    def summary(self) -> str:
        return (
            f"success_rate={self.success_rate:.3f}±{self.success_std:.3f} "
            f"assertion_fraction={self.mean_fraction:.3f} "
            f"({len(self.per_task)} tasks, k={self.k})"
        )


def evaluate_with_env(
    tasks: list[TaskSpec],
    make_env: EnvFactory,
    runtime: Runtime,
    judge: GoldJudge,
    *,
    label: str = "",
    k: int = DEFAULT_K,
    concurrency: int = 1,
    on_progress: Callable[[str, int, GoldVerdict], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> ClosedLoopReport:
    """Score the agent on `tasks` against whatever env `make_env` opens, k passes per task.

    `concurrency` is how many (task, attempt) cells run at once: the default 1 keeps the
    sequential loop (world-model behavior unchanged), 0 runs every cell simultaneously (the E2B
    one-sandbox-per-rollout backend), and N>1 caps the pool at N. The report is identical either
    way: verdicts are collected by cell index and aggregated per task in attempt order, and
    `on_progress` always fires from the calling thread so UI callbacks see a serial stream.
    """
    if k < 1:
        raise ValueError("k must be >= 1 (metrics are means over k passes)")
    per_task: dict[str, TaskOutcome] = {}
    usages: list[TokenUsage | None] = []
    if concurrency != 0 and concurrency <= 1:
        try:
            for task in tasks:
                verdicts: list[GoldVerdict] = []
                attempts: list[RolloutEvidence] = []
                for attempt in range(k):
                    _check_cancelled(should_cancel)
                    result = _run_once(task, make_env, runtime)
                    # Record the episode before the next cancellation boundary. If
                    # cancellation landed during the bounded worker call, RunnerLink
                    # raises with that episode's partial usage instead.
                    usages.append(result.worker_usage)
                    attempts.append(_rollout_evidence(result))
                    _check_cancelled(should_cancel)
                    verdict = judge.score(
                        task.instruction, result.answer, result.transcript(), task.gold
                    )
                    _check_cancelled(should_cancel)
                    verdicts.append(verdict)
                    if on_progress is not None:
                        on_progress(task.task_id, attempt + 1, verdict)
                successes = [1.0 if v.passed else 0.0 for v in verdicts]
                per_task[task.task_id] = TaskOutcome(
                    task_id=task.task_id,
                    success_rate=fmean(successes),
                    mean_fraction=fmean(v.fraction for v in verdicts),
                    passes=k,
                    verdicts=verdicts,
                    attempts=attempts,
                )
        except RuntimeCancelled as error:
            error.worker_usage = combine_usage([*usages, error.worker_usage])
            raise
    else:
        by_cell, usages, evidence_by_cell = _run_cells_concurrently(
            tasks,
            make_env,
            runtime,
            judge,
            k=k,
            concurrency=concurrency,
            on_progress=on_progress,
            should_cancel=should_cancel,
        )
        for index, task in enumerate(tasks):
            verdicts = by_cell[index * k : (index + 1) * k]  # cells are task-major, attempt-minor
            attempts = evidence_by_cell[index * k : (index + 1) * k]
            successes = [1.0 if v.passed else 0.0 for v in verdicts]
            per_task[task.task_id] = TaskOutcome(
                task_id=task.task_id,
                success_rate=fmean(successes),
                mean_fraction=fmean(v.fraction for v in verdicts),
                passes=k,
                verdicts=verdicts,
                attempts=attempts,
            )

    task_rates = [o.success_rate for o in per_task.values()]
    return ClosedLoopReport(
        label=label,
        success_rate=fmean(task_rates) if task_rates else 0.0,
        mean_fraction=fmean(o.mean_fraction for o in per_task.values()) if per_task else 0.0,
        success_std=pstdev(task_rates) if len(task_rates) > 1 else 0.0,
        k=k,
        per_task=per_task,
        worker_usage=combine_usage(usages),
    )


def evaluate_closed_loop(
    tasks: list[TaskSpec],
    environment: EnvironmentSource,
    agent_provider: Provider,
    judge: GoldJudge,
    *,
    label: str = "world-model",
    k: int = DEFAULT_K,
    concurrency: int = 1,
    runtime: Runtime | None = None,
    on_progress: Callable[[str, int, GoldVerdict], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> ClosedLoopReport:
    """Score the fixed agent on `tasks` against `environment` (`wmh eval --mode closed-loop`).

    With `concurrency != 1` the environment serves many rollouts at once, so the whole eval runs
    under `environment.frozen()` (the `scenario_fidelity.score_matrix` precedent): sessions are
    already independent (`enrich=False`), and freezing a local world model keeps parallel stepping
    from mutating the shared retrieval index mid-eval. Sequential behavior is unchanged, and a
    hosted environment (whose state lives server-side) freezes to a no-op.
    """

    def _evaluate() -> ClosedLoopReport:
        return evaluate_with_env(
            tasks,
            lambda task: environment.open(task.instruction),
            runtime if runtime is not None else AgentRuntime(agent_provider),
            judge,
            label=label,
            k=k,
            concurrency=concurrency,
            on_progress=on_progress,
            should_cancel=should_cancel,
        )

    if concurrency == 1:
        return _evaluate()
    with environment.frozen():
        return _evaluate()


class ClosedLoopEval:
    """The closed-loop `Evaluation`: a live agent runs tasks with the simulation as its env."""

    def __init__(
        self,
        tasks: list[TaskSpec],
        environment: EnvironmentSource,
        agent_provider: Provider,
        judge: GoldJudge,
        *,
        label: str = "world-model",
        k: int = DEFAULT_K,
        concurrency: int = 1,
        runtime: Runtime | None = None,
        on_progress: Callable[[str, int, GoldVerdict], None] | None = None,
    ) -> None:
        self._tasks = tasks
        self._environment = environment
        self._agent_provider = agent_provider
        self._judge = judge
        self._label = label
        self._k = k
        self._concurrency = concurrency
        self._runtime = runtime
        self._on_progress = on_progress

    def run(self) -> ClosedLoopReport:
        return evaluate_closed_loop(
            self._tasks,
            self._environment,
            self._agent_provider,
            self._judge,
            label=self._label,
            k=self._k,
            concurrency=self._concurrency,
            runtime=self._runtime,
            on_progress=self._on_progress,
        )


def _run_once(task: TaskSpec, make_env: EnvFactory, runtime: Runtime) -> RunResult:
    """One rollout: a fresh environment per attempt, always closed."""
    env = make_env(task)
    try:
        return runtime.run(task.task_id, task.instruction, env)
    finally:
        env.close()


def _rollout_evidence(result: RunResult) -> RolloutEvidence:
    """Freeze the proposer-facing parts of a run before its environment is gone."""
    return RolloutEvidence(
        answer=_bounded_evidence_text(
            normalize_durable_text(result.answer),
            _ROLLOUT_ANSWER_CHARS,
            label="answer",
        ),
        transcript=_bounded_evidence_text(
            normalize_durable_text(result.transcript()),
            _ROLLOUT_EVIDENCE_CHARS,
            label="trace",
        ),
        stop_reason=result.stop_reason,
        turns=result.turns,
    )


def _bounded_evidence_text(content: str, limit: int, *, label: str) -> str:
    """Retain the beginning and terminal behavior without unbounded report memory."""
    if len(content) <= limit:
        return content
    head = limit // 2
    tail = limit - head
    omitted = len(content) - limit
    return f"{content[:head]}\n... ({omitted} {label} characters omitted) ...\n{content[-tail:]}"


def _run_cells_concurrently(
    tasks: list[TaskSpec],
    make_env: EnvFactory,
    runtime: Runtime,
    judge: GoldJudge,
    *,
    k: int,
    concurrency: int,
    on_progress: Callable[[str, int, GoldVerdict], None] | None,
    should_cancel: Callable[[], bool] | None,
) -> tuple[list[GoldVerdict], list[TokenUsage | None], list[RolloutEvidence]]:
    """Run every (task, attempt) cell on a thread pool; verdicts return in cell order.

    Cell order is task-major, attempt-minor — the exact order the sequential loop visits — so the
    caller can slice per task and aggregate deterministically. `on_progress` fires from THIS
    thread as futures land (gepa.py precedent: UI callbacks must be a serial stream). A rollout or
    judge call that raises is a real failure: pending cells are cancelled and the exception
    propagates after owned work drains — never swallowed into a verdict. Any failure drains
    already-started cells before returning while cancelling cells
    that have not started: those threads can still own billable provider calls, rollout
    persistence, and sandbox leases, so the caller must not terminalize the enclosing run while
    they remain active. Every cell releases its environment through ``_run_once``'s ``finally``.
    """
    cells = [(task, attempt) for task in tasks for attempt in range(k)]
    if not cells:
        return [], [], []
    max_workers = len(cells) if concurrency == 0 else min(concurrency, len(cells))
    slots: list[GoldVerdict | None] = [None] * len(cells)
    usage_slots: list[TokenUsage | None] = [None] * len(cells)
    evidence_slots: list[RolloutEvidence | None] = [None] * len(cells)

    def run_cell(
        task: TaskSpec,
    ) -> tuple[GoldVerdict, TokenUsage | None, RolloutEvidence]:
        _check_cancelled(should_cancel)
        result = _run_once(task, make_env, runtime)
        try:
            _check_cancelled(should_cancel)
            verdict = judge.score(task.instruction, result.answer, result.transcript(), task.gold)
            _check_cancelled(should_cancel)
        except RuntimeCancelled as error:
            # The rollout finished before cancellation, even though its verdict
            # did not. Preserve that worker spend for the coordinator's drain.
            error.worker_usage = combine_usage([result.worker_usage, error.worker_usage])
            raise
        return verdict, result.worker_usage, _rollout_evidence(result)

    pool = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = {pool.submit(run_cell, task): i for i, (task, _attempt) in enumerate(cells)}
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
            # Do not rely on a cell reaching RunnerLink before noticing API cancellation: all
            # cells may still be inside cold E2B startup. This caller-side poll aborts the shared
            # runtime below, which closes registered sandboxes before the ownership drain.
            _check_cancelled(should_cancel)
            for future in sorted(done, key=futures.__getitem__):
                index = futures[future]
                verdict, usage, evidence = (
                    future.result()
                )  # a rollout/judge exception propagates here
                slots[index] = verdict
                usage_slots[index] = usage
                evidence_slots[index] = evidence
                if on_progress is not None:
                    task, attempt = cells[index]
                    on_progress(task.task_id, attempt + 1, verdict)
    except BaseException as error:
        cancelled = isinstance(error, RuntimeCancelled)
        if not cancelled and isinstance(error, Exception) and should_cancel is not None:
            try:
                cancelled = should_cancel()
            except Exception:  # noqa: BLE001 - preserve the cell's original failure
                pass
        abort_error: BaseException | None = None
        abort = getattr(runtime, "abort", None)
        if callable(abort):
            try:
                abort()
            except BaseException as candidate:  # noqa: BLE001 - cleanup must survive the drain
                abort_error = candidate
        pool.shutdown(wait=True, cancel_futures=True)
        if abort_error is not None and callable(abort):
            # A worker's release() during the drain may have retried and proved the exact lease
            # whose first abort failed. Re-run the idempotent close before declaring a leak.
            try:
                abort()
            except BaseException as candidate:  # noqa: BLE001 - this is the final cleanup proof
                abort_error = candidate
            else:
                abort_error = None
        cancelled_error: RuntimeCancelled | None = None
        if cancelled:
            # Every started future is done after shutdown(wait=True). Collect
            # successful episode usage and partial usage carried by cancelled
            # episodes exactly once, including futures the coordinator had not
            # observed before the cancellation boundary.
            partial_usages: list[TokenUsage | None] = []
            for future in futures:
                if future.cancelled():
                    continue
                try:
                    _verdict, usage, _evidence = future.result()
                except RuntimeCancelled as candidate:
                    partial_usages.append(candidate.worker_usage)
                except BaseException:  # noqa: BLE001 - original error remains authoritative
                    continue
                else:
                    partial_usages.append(usage)
            if isinstance(error, RuntimeCancelled):
                cancelled_error = error
            else:
                cancelled_error = RuntimeCancelled("runtime evaluation cancelled")
            cancelled_error.worker_usage = combine_usage(partial_usages)
        if abort_error is not None:
            raise abort_error from (cancelled_error or error)
        if cancelled_error is not None and cancelled_error is not error:
            raise cancelled_error from error
        raise
    else:
        pool.shutdown(wait=True)
    verdicts: list[GoldVerdict] = []
    attempts: list[RolloutEvidence] = []
    for slot, evidence in zip(slots, evidence_slots, strict=True):
        if slot is None:  # pragma: no cover - every future completed, or we raised above
            raise RuntimeError("a cell completed without producing a verdict")
        if evidence is None:  # pragma: no cover - same future produced both values atomically
            raise RuntimeError("a cell completed without preserving rollout evidence")
        verdicts.append(slot)
        attempts.append(evidence)
    return verdicts, usage_slots, attempts


def _check_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel is not None and should_cancel():
        raise RuntimeCancelled("runtime evaluation cancelled")
