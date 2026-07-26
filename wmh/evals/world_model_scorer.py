"""Closed-loop world-model projection into evaluator-neutral harness score reports.

The world-model sibling of :class:`wmh.evals.harbor.scorer.HarborScorer`: both implement the
`wmh.harness.scoring.Scorer` protocol, but this one runs the fixed agent against a SIMULATED
environment. Every environment response comes from the world model, a :class:`GoldJudge` grades
each rollout transcript against the task's gold assertions, and each (task, attempt) verdict
becomes one :class:`ScoreCell`. Rollouts reuse the merged closed-loop machinery
(`wmh.evals.closed_loop.evaluate_closed_loop`) rather than reimplementing them.

Unlike the harbor scorer there is no per-trial artifact directory on the host filesystem: the
world model produces evidence in memory, not a job directory the proposer can re-read, so every
cell carries an empty `artifact_dir` and a short diagnostic `note`. The proposer marks those
gaps with `NO-EVIDENCE.md`; that is the accepted local-backend behavior. `reward_mode` is
carried on the report for record only: the closed-loop judge already decided pass/fail per its
gold semantics, so `passed` comes from the judge and is never re-thresholded here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from wmh.engine.world_model import WorldModel
from wmh.evals.closed_loop import (
    ClosedLoopReport,
    RolloutEvidence,
    evaluate_closed_loop,
)
from wmh.evals.gold import GoldJudge, GoldVerdict
from wmh.evals.tasks import TaskSpec
from wmh.harness.doc import HarnessDoc
from wmh.harness.runtime import HarnessSearchCancelled, Runtime, RuntimeCancelled
from wmh.harness.scoring import (
    MAX_CELL_NOTE_CHARS,
    RewardMode,
    ScoreCell,
    ScoreReport,
    ScoreRequest,
)
from wmh.providers.base import Provider

# The rollout evidence rationale is folded into a bounded cell note; keep it well under the
# frozen MAX_CELL_NOTE_CHARS so the leading diagnostic prefix always survives.
_NOTE_RATIONALE_CHARS = 1_500


class ClosedLoopEvaluate(Protocol):
    """The exact rollout seam `evaluate_closed_loop` satisfies.

    The scorer owns request minting and cell projection; the seam owns rollouts and judging.
    Tests inject a fake here so no provider, world model, or sandbox is exercised.
    """

    def __call__(
        self,
        tasks: list[TaskSpec],
        world_model: WorldModel,
        agent_provider: Provider,
        judge: GoldJudge,
        *,
        label: str,
        k: int,
        concurrency: int,
        runtime: Runtime | None,
        on_progress: Callable[[str, int, GoldVerdict], None] | None,
        should_cancel: Callable[[], bool] | None,
    ) -> ClosedLoopReport: ...


class WorldModelScorer:
    """Evaluate exact harness candidates closed-loop against a simulated environment.

    Each `score` call runs `request.attempts` passes per configured task with the world model
    answering every environment action, judges each rollout with the configured `GoldJudge`,
    and projects the per-task verdicts onto the exact requested cell matrix. The backend is
    local only: `evaluate_closed_loop` runs the fixed `agent_provider` through an
    `AgentRuntime` in-process, so there is no e2b harness backend, no sandbox pool, and no
    execution digest.
    """

    def __init__(
        self,
        *,
        world_model: WorldModel,
        tasks: Sequence[TaskSpec],
        agent_provider: Provider,
        judge: GoldJudge,
        attempts: int,
        reward_mode: RewardMode = "positive-binary",
        eval_concurrency: int = 1,
        should_cancel: Callable[[], bool] | None = None,
        evaluate: ClosedLoopEvaluate | None = None,
    ) -> None:
        task_list = [TaskSpec.model_validate(task.model_dump(mode="python")) for task in tasks]
        if not task_list:
            raise ValueError("tasks must be nonempty")
        ids = [task.task_id for task in task_list]
        duplicates = sorted({task_id for task_id in ids if ids.count(task_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate task_id(s): {duplicates}")
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
            raise ValueError("attempts must be a positive integer")
        if (
            isinstance(eval_concurrency, bool)
            or not isinstance(eval_concurrency, int)
            or eval_concurrency < 0
        ):
            raise ValueError("eval_concurrency must be an integer >= 0 (0 runs every cell at once)")
        if reward_mode not in ("raw", "positive-binary"):
            raise ValueError("reward_mode must be raw or positive-binary")
        self._world_model = world_model
        self._tasks = tuple(task_list)
        self._tasks_by_id = {task.task_id: task for task in task_list}
        self._task_ids = tuple(task.task_id for task in task_list)
        self._agent_provider = agent_provider
        self._judge = judge
        self._attempts = attempts
        self._reward_mode: RewardMode = reward_mode
        self._eval_concurrency = eval_concurrency
        self._should_cancel = should_cancel
        self._evaluate: ClosedLoopEvaluate = (
            evaluate if evaluate is not None else evaluate_closed_loop
        )

    @property
    def request(self) -> ScoreRequest:
        """The exact task-by-attempt matrix every `score` call evaluates (configured order)."""
        return ScoreRequest(task_ids=self._task_ids, attempts=self._attempts)

    @property
    def reward_mode(self) -> RewardMode:
        """The frozen reward interpretation carried on every report (record only)."""
        return self._reward_mode

    def score(
        self,
        doc: HarnessDoc,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> ScoreReport:
        """Run one candidate closed-loop and project its judged verdicts into cells.

        The scorer always evaluates its full configured suite: main's `ScoreRequest` carries no
        subset, and `optimize` always calls `score(doc)` with the scorer's own request.
        """
        cancel = should_cancel if should_cancel is not None else self._should_cancel
        _check_cancelled(cancel)
        try:
            report = self._evaluate(
                list(self._tasks),
                self._world_model,
                self._agent_provider,
                self._judge,
                label=doc.name,
                k=self._attempts,
                concurrency=self._eval_concurrency,
                runtime=None,
                on_progress=None,
                should_cancel=cancel,
            )
        except RuntimeCancelled as error:
            # Cancelled cells are not scoreable outcomes: convert at the scorer boundary so
            # population.optimize propagates it as a search cancellation, never a partial report.
            raise HarnessSearchCancelled(
                "harness score cancelled", worker_usage=error.worker_usage
            ) from error
        return self._project(doc, report)

    def _project(self, doc: HarnessDoc, report: ClosedLoopReport) -> ScoreReport:
        """Map the closed-loop report onto the exact requested cell matrix, fail-closed."""
        cells: list[ScoreCell] = []
        for task_id in self._task_ids:
            outcome = report.per_task.get(task_id)
            if outcome is None:
                raise ValueError(f"closed-loop report is missing task {task_id!r}")
            if len(outcome.verdicts) != self._attempts or len(outcome.attempts) != self._attempts:
                raise ValueError(
                    f"closed-loop report for task {task_id!r} does not carry exactly "
                    f"{self._attempts} judged attempt(s)"
                )
            for attempt in range(1, self._attempts + 1):
                verdict = outcome.verdicts[attempt - 1]
                evidence = outcome.attempts[attempt - 1]
                cells.append(
                    ScoreCell(
                        task_id=task_id,
                        attempt=attempt,
                        # verdict.fraction is already in [0, 1]; passed comes straight from the
                        # judge's gold decision and is never re-thresholded by reward_mode.
                        reward=verdict.fraction,
                        passed=verdict.passed,
                        artifact_dir="",
                        note=_cell_note(verdict, evidence),
                    )
                )
        return ScoreReport(
            doc_hash=doc.doc_hash,
            request=self.request,
            reward_mode=self._reward_mode,
            cells=tuple(cells),
        )


def _cell_note(verdict: GoldVerdict, evidence: RolloutEvidence) -> str:
    """One short diagnostic per cell, bounded to the frozen cell-note length."""
    rationale = verdict.rationale
    if len(rationale) > _NOTE_RATIONALE_CHARS:
        rationale = rationale[:_NOTE_RATIONALE_CHARS] + " ..."
    note = (
        f"passed={verdict.passed} fraction={verdict.fraction:.3f} "
        f"stop={evidence.stop_reason.value} turns={evidence.turns}: {rationale}"
    )
    return note[:MAX_CELL_NOTE_CHARS]


def _check_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel is not None and should_cancel():
        raise HarnessSearchCancelled("harness score cancelled")
