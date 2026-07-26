"""Hermetic WorldModelScorer tests: a role-playing provider and a stubbed rollout seam.

Two layers: the round-trip test drives the REAL `evaluate_closed_loop` machinery with one
`RoleProvider` playing agent, world model, and judge (the `closed_loop_test` pattern), while the
projection tests inject a `FakeEvaluate` seam so cell mapping and report shape are pinned without
any rollout, provider, world model, or sandbox.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest

from wmh.engine.world_model import WorldModel
from wmh.evals.closed_loop import ClosedLoopReport, RolloutEvidence, TaskOutcome
from wmh.evals.gold import AssertionResult, GoldJudge, GoldVerdict
from wmh.evals.tasks import TaskSpec
from wmh.evals.world_model_scorer import ClosedLoopEvaluate, WorldModelScorer
from wmh.harness.doc import HarnessDoc
from wmh.harness.runtime import HarnessSearchCancelled, Runtime, RuntimeCancelled, StopReason
from wmh.harness.scoring import RewardMode
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder


class RoleProvider:
    """Plays agent, world model, and gold judge, keyed off the system prompt."""

    def __init__(self, *, judge_passes: bool = True, model: str = "m") -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model=model)
        self._judge_passes = judge_passes

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        del messages, temperature, max_tokens
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
        return Completion(text='{"output": "ok", "is_error": false}')

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201 - test fake never calls it
        raise NotImplementedError


def _wm(provider: RoleProvider) -> WorldModel:
    return WorldModel(provider, EmbeddingRetriever(HashingEmbedder(dim=16)))


def _tasks() -> list[TaskSpec]:
    return [
        TaskSpec(task_id="q1", instruction="answer it", gold=["did it"]),
        TaskSpec(task_id="q2", instruction="answer it again", gold=["did it"]),
    ]


def _seam_tasks() -> list[TaskSpec]:
    return [
        TaskSpec(task_id="a", instruction="do a", gold=["g"]),
        TaskSpec(task_id="b", instruction="do b", gold=["g"]),
        TaskSpec(task_id="c", instruction="do c", gold=["g"]),
    ]


def _make_scorer(
    *,
    provider: RoleProvider | None = None,
    tasks: list[TaskSpec] | None = None,
    attempts: int = 2,
    reward_mode: RewardMode = "positive-binary",
    eval_concurrency: int = 1,
    should_cancel: Callable[[], bool] | None = None,
    evaluate: ClosedLoopEvaluate | None = None,
) -> WorldModelScorer:
    resolved = provider if provider is not None else RoleProvider()
    return WorldModelScorer(
        world_model=_wm(resolved),
        tasks=tasks if tasks is not None else _tasks(),
        agent_provider=resolved,
        judge=GoldJudge(resolved),
        attempts=attempts,
        reward_mode=reward_mode,
        eval_concurrency=eval_concurrency,
        should_cancel=should_cancel,
        evaluate=evaluate,
    )


class FakeEvaluate:
    """Rollout seam stub: records every call and fabricates a `ClosedLoopReport` per plan."""

    def __init__(
        self,
        *,
        fractions: dict[str, list[float]] | None = None,
        drop_last_verdict: bool = False,
        extra_task_id: str | None = None,
        raises: BaseException | None = None,
    ) -> None:
        self.fractions = fractions
        self.drop_last_verdict = drop_last_verdict
        self.extra_task_id = extra_task_id
        self.raises = raises
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        tasks: list[TaskSpec],
        world_model: WorldModel,
        agent_provider: object,
        judge: GoldJudge,
        *,
        label: str,
        k: int,
        concurrency: int,
        runtime: Runtime | None,
        on_progress: object,
        should_cancel: Callable[[], bool] | None,
    ) -> ClosedLoopReport:
        del world_model, agent_provider, judge, on_progress, should_cancel
        self.calls.append(
            {
                "task_ids": [task.task_id for task in tasks],
                "label": label,
                "k": k,
                "concurrency": concurrency,
                "runtime": runtime,
            }
        )
        if self.raises is not None:
            raise self.raises
        per_task = {task.task_id: self._outcome(task, k) for task in tasks}
        if self.extra_task_id is not None:
            extra = TaskSpec(task_id=self.extra_task_id, instruction="extra", gold=["g"])
            per_task[extra.task_id] = self._outcome(extra, k)
        return ClosedLoopReport(label=label, k=k, per_task=per_task)

    def _outcome(self, task: TaskSpec, k: int) -> TaskOutcome:
        fracs = (self.fractions or {}).get(task.task_id, [1.0] * k)
        verdicts = [
            GoldVerdict(
                passed=fraction == 1.0,
                fraction=fraction,
                assertions=[AssertionResult(assertion="g", passed=fraction == 1.0, why="w")],
                rationale=f"{task.task_id} attempt {attempt} rationale",
            )
            for attempt, fraction in enumerate(fracs, 1)
        ]
        if self.drop_last_verdict:
            verdicts = verdicts[:-1]
        attempts = [
            RolloutEvidence(
                answer=f"answer-{task.task_id}-{attempt}",
                transcript=f"[1] tool_call: bash cmd-{task.task_id}-{attempt}\n    -> ok",
                stop_reason=StopReason.SUBMITTED,
                turns=attempt,
            )
            for attempt in range(1, len(fracs) + 1)
        ]
        return TaskOutcome(
            task_id=task.task_id,
            success_rate=sum(1.0 for verdict in verdicts if verdict.passed) / max(len(fracs), 1),
            mean_fraction=sum(verdict.fraction for verdict in verdicts) / max(len(fracs), 1),
            passes=k,
            verdicts=verdicts,
            attempts=attempts,
        )


# -- request property ---------------------------------------------------------------------------


def test_request_is_the_full_suite_in_configured_order() -> None:
    scorer = _make_scorer(attempts=2)
    request = scorer.request
    assert request.task_ids == ("q1", "q2")
    assert request.attempts == 2
    # request is a property, stable across reads, and preserves configured order.
    reversed_scorer = _make_scorer(tasks=list(reversed(_tasks())), attempts=1)
    assert reversed_scorer.request.task_ids == ("q2", "q1")


# -- end to end through the real closed-loop machinery -----------------------------------------


def test_score_round_trips_through_real_rollouts() -> None:
    scorer = _make_scorer(attempts=2)
    candidate = HarnessDoc.baseline("candidate")

    report = scorer.score(candidate)

    assert report.doc_hash == candidate.doc_hash
    assert report.request == scorer.request
    assert report.reward_mode == "positive-binary"
    assert [(cell.task_id, cell.attempt, cell.reward, cell.passed) for cell in report.cells] == [
        ("q1", 1, 1.0, True),
        ("q1", 2, 1.0, True),
        ("q2", 1, 1.0, True),
        ("q2", 2, 1.0, True),
    ]
    assert report.score == 1.0
    assert report.pass_rate == 1.0
    assert all(cell.artifact_dir == "" for cell in report.cells)
    assert all(cell.note.startswith("passed=True fraction=1.000") for cell in report.cells)


def test_failed_judgement_maps_to_zero_reward_cells() -> None:
    scorer = _make_scorer(provider=RoleProvider(judge_passes=False), attempts=1)
    report = scorer.score(HarnessDoc.baseline())
    assert report.score == 0.0
    assert all(not cell.passed for cell in report.cells)
    assert all(cell.reward == 0.0 for cell in report.cells)


# -- cell mapping through the stubbed seam ------------------------------------------------------


def test_cells_map_fraction_passed_and_one_indexed_attempts() -> None:
    fake = FakeEvaluate(fractions={"a": [1.0, 0.5], "b": [0.0, 1.0], "c": [1.0, 1.0]})
    scorer = _make_scorer(tasks=_seam_tasks(), attempts=2, evaluate=fake)
    candidate = HarnessDoc.baseline("candidate")

    report = scorer.score(candidate)

    assert [(cell.task_id, cell.attempt, cell.reward, cell.passed) for cell in report.cells] == [
        ("a", 1, 1.0, True),
        ("a", 2, 0.5, False),
        ("b", 1, 0.0, False),
        ("b", 2, 1.0, True),
        ("c", 1, 1.0, True),
        ("c", 2, 1.0, True),
    ]
    # report.score is the mean of per-task pass RATES: a=0.5, b=0.5, c=1.0 -> 2/3.
    assert report.score == pytest.approx(2.0 / 3.0)
    by_task = report.by_task()
    assert set(by_task) == {"a", "b", "c"}
    half_credit = next(cell for cell in report.cells if (cell.task_id, cell.attempt) == ("a", 2))
    assert "fraction=0.500" in half_credit.note
    assert "a attempt 2 rationale" in half_credit.note
    assert len(half_credit.note) <= 2_000
    assert len(fake.calls) == 1
    assert fake.calls[0]["k"] == 2
    assert fake.calls[0]["concurrency"] == 1
    assert fake.calls[0]["label"] == "candidate"
    assert fake.calls[0]["task_ids"] == ["a", "b", "c"]
    assert fake.calls[0]["runtime"] is None


def test_passed_is_the_judge_decision_never_re_thresholded() -> None:
    # A partial fraction that "passes" positive-binary thresholding must still be marked failed:
    # passed comes straight from the judge, not from reward_passed(fraction, reward_mode).
    fake = FakeEvaluate(fractions={"a": [0.5]})
    scorer = _make_scorer(
        tasks=_seam_tasks()[:1], attempts=1, reward_mode="positive-binary", evaluate=fake
    )
    report = scorer.score(HarnessDoc.baseline())
    [cell] = report.cells
    assert cell.reward == 0.5
    assert cell.passed is False


# -- report validation --------------------------------------------------------------------------


@pytest.mark.parametrize("defect", ["drop_last_verdict", "missing_task"])
def test_malformed_closed_loop_reports_are_rejected(defect: str) -> None:
    fake = FakeEvaluate(
        drop_last_verdict=defect == "drop_last_verdict",
        fractions={"a": [1.0, 1.0], "b": [], "c": [1.0, 1.0]} if defect == "missing_task" else None,
    )
    scorer = _make_scorer(tasks=_seam_tasks(), attempts=2, evaluate=fake)
    with pytest.raises(ValueError):
        scorer.score(HarnessDoc.baseline())


def test_extra_report_task_is_ignored_only_requested_cells_are_projected() -> None:
    fake = FakeEvaluate(extra_task_id="zz")
    scorer = _make_scorer(tasks=_seam_tasks(), attempts=1, evaluate=fake)
    report = scorer.score(HarnessDoc.baseline())
    assert {cell.task_id for cell in report.cells} == {"a", "b", "c"}


# -- cancellation ------------------------------------------------------------------------------


def test_cancellation_before_rollout_raises_without_evaluating() -> None:
    fake = FakeEvaluate()
    scorer = _make_scorer(tasks=_seam_tasks(), attempts=1, evaluate=fake)
    with pytest.raises(HarnessSearchCancelled):
        scorer.score(HarnessDoc.baseline(), should_cancel=lambda: True)
    assert fake.calls == []


def test_runtime_cancelled_is_converted_at_the_scorer_boundary() -> None:
    fake = FakeEvaluate(raises=RuntimeCancelled("cell cancelled mid-wave"))
    scorer = _make_scorer(tasks=_seam_tasks(), attempts=1, evaluate=fake)
    with pytest.raises(HarnessSearchCancelled):
        scorer.score(HarnessDoc.baseline())


def test_constructor_should_cancel_is_the_default_seam() -> None:
    fake = FakeEvaluate()
    scorer = _make_scorer(
        tasks=_seam_tasks(), attempts=1, evaluate=fake, should_cancel=lambda: True
    )
    with pytest.raises(HarnessSearchCancelled):
        scorer.score(HarnessDoc.baseline())
    assert fake.calls == []


# -- constructor validation ---------------------------------------------------------------------


def test_constructor_rejects_bad_configuration() -> None:
    with pytest.raises(ValueError, match="tasks must be nonempty"):
        _make_scorer(tasks=[])
    with pytest.raises(ValueError, match="duplicate task_id"):
        _make_scorer(tasks=[_tasks()[0], _tasks()[0]])
    with pytest.raises(ValueError, match="attempts must be a positive integer"):
        _make_scorer(attempts=0)
    with pytest.raises(ValueError, match="attempts must be a positive integer"):
        _make_scorer(attempts=cast("int", True))
    with pytest.raises(ValueError, match="eval_concurrency"):
        _make_scorer(eval_concurrency=-1)
    with pytest.raises(ValueError, match="eval_concurrency"):
        _make_scorer(eval_concurrency=cast("int", True))
    with pytest.raises(ValueError, match="reward_mode"):
        _make_scorer(reward_mode=cast("RewardMode", "binary"))


def test_scorer_snapshots_tasks_against_caller_mutation() -> None:
    tasks = _tasks()
    scorer = _make_scorer(tasks=tasks, attempts=1)
    before = scorer.request
    tasks[0].instruction = "mutated after construction"
    assert scorer.request == before
