"""Behavioral tests for the feedback-directed one-step improvement gate.

The tests drive the REAL `wmh.harness.population.optimize` loop through a tmp run dir with a
fake main `Scorer` (request property + `score(doc)`) and a fake main `CandidateProposer`
(`propose(population, *, slot, should_cancel)`), so the gate decisions are asserted on real
`ScoreReport` cells (`reward`/`passed`) and `result.population`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest

from wmh.harness.doc import HarnessDoc
from wmh.harness.improve import (
    ImproveGate,
    ImproveOutcome,
    improve_harness,
    suite_score,
    verification_results,
)
from wmh.harness.population import (
    CandidateProposal,
    CandidateProposalError,
    EvaluatedCandidate,
)
from wmh.harness.scoring import ScoreCell, ScoreReport, ScoreRequest
from wmh.harness.source_tree import HarnessSourceFile, HarnessSourceTree

# One preplanned score call: task_id -> one (reward, passed) pair per attempt.
_CellPlan = Mapping[str, Sequence[tuple[float, bool]]]


class _Scorer:
    """Yields one preplanned per-task cell matrix per score call (keyed by doc hash).

    The population loop scores the seed first, then each candidate in slot order, so the plans
    are consumed in that same order. Each report's doc_hash is taken from the scored doc so
    `EvaluatedCandidate` accepts it.
    """

    def __init__(
        self,
        task_ids: Sequence[str],
        attempts: int,
        outcomes: Sequence[_CellPlan],
    ) -> None:
        self._task_ids = tuple(task_ids)
        self._attempts = attempts
        self._outcomes = list(outcomes)
        self.calls: list[str] = []

    @property
    def request(self) -> ScoreRequest:
        return ScoreRequest(task_ids=self._task_ids, attempts=self._attempts)

    def score(
        self, doc: HarnessDoc, *, should_cancel: Callable[[], bool] | None = None
    ) -> ScoreReport:
        del should_cancel
        self.calls.append(doc.doc_hash)
        per_task = self._outcomes.pop(0)
        cells: list[ScoreCell] = []
        for task_id in self._task_ids:
            attempt_outcomes = per_task[task_id]
            assert len(attempt_outcomes) == self._attempts
            for attempt, (reward, passed) in enumerate(attempt_outcomes, start=1):
                cells.append(
                    ScoreCell(
                        task_id=task_id,
                        attempt=attempt,
                        reward=reward,
                        passed=passed,
                        note="planned cell",
                    )
                )
        return ScoreReport(
            doc_hash=doc.doc_hash,
            request=self.request,
            reward_mode="positive-binary",
            cells=tuple(cells),
        )


class _Proposer:
    """Returns one preplanned source tree (or raises an error) per slot."""

    def __init__(self, outcomes: Sequence[HarnessSourceTree | CandidateProposalError]) -> None:
        self._outcomes = list(outcomes)
        self.slots: list[int] = []

    def propose(
        self,
        population: Sequence[EvaluatedCandidate],
        *,
        slot: int,
        should_cancel: Callable[[], bool] | None = None,
    ) -> CandidateProposal:
        del population, should_cancel
        self.slots.append(slot)
        outcome = self._outcomes.pop(0)
        candidate_id = f"candidate-{slot:04d}"
        if isinstance(outcome, CandidateProposalError):
            raise outcome
        return CandidateProposal(
            candidate_id=candidate_id,
            source=outcome,
            candidate=outcome.to_doc(candidate_id),
        )


def _source(prompt: str) -> HarnessSourceTree:
    return HarnessSourceTree(
        files=(
            HarnessSourceFile(path="SYSTEM.md", content=prompt),
            HarnessSourceFile(
                path="config.toml",
                content='[harness]\ntools = ["bash", "submit"]\nruntime_kind = "pi-node"\n',
            ),
        )
    )


def _uniform(task_outcomes: Mapping[str, tuple[float, bool]], attempts: int) -> _CellPlan:
    return {task_id: [outcome] * attempts for task_id, outcome in task_outcomes.items()}


_SUITE = ("suite-a", "suite-b")
_VERIFY = ("verify-1",)
_ALL_IDS = ("suite-a", "suite-b", "verify-1")


def _improve(
    *,
    scorer: _Scorer,
    proposer: _Proposer,
    run_dir: Path,
    iterations: int,
    suite: Sequence[str] = _SUITE,
    verification: Sequence[str] = _VERIFY,
    feedback: str = "the agent should have access to my GitHub",
    gate: ImproveGate | None = None,
) -> ImproveOutcome:
    return improve_harness(
        seed=_source("seed"),
        feedback=feedback,
        suite_task_ids=suite,
        verification_task_ids=verification,
        scorer=scorer,
        proposer=proposer,
        run_dir=run_dir,
        iterations=iterations,
        gate=gate if gate is not None else ImproveGate(),
    )


def test_gate_accepts_candidate_within_margin_with_all_verification_passing(
    tmp_path: Path,
) -> None:
    scorer = _Scorer(
        _ALL_IDS,
        3,
        [
            _uniform({"suite-a": (0.8, True), "suite-b": (0.8, True), "verify-1": (0.0, False)}, 3),
            {
                "suite-a": [(0.75, True)] * 3,
                "suite-b": [(0.75, True)] * 3,
                "verify-1": [(1.0, True), (1.0, True), (0.0, False)],  # 2 of 3: strict majority
            },
        ],
    )
    proposer = _Proposer([_source("improved")])

    outcome = _improve(scorer=scorer, proposer=proposer, run_dir=tmp_path / "run", iterations=1)

    assert outcome.accepted is True
    assert outcome.selected is not None
    assert outcome.selected.candidate_id == "candidate-0001"
    assert outcome.seed_suite_score == pytest.approx(0.8)
    assert outcome.candidate_suite_score == pytest.approx(0.75)
    [verification] = outcome.verification
    assert verification.task_id == "verify-1"
    assert verification.passed is True
    assert verification.pass_count == 2
    assert verification.attempts == 3
    assert "candidate-0001" in outcome.reason


def test_gate_rejects_suite_regression_beyond_margin(tmp_path: Path) -> None:
    scorer = _Scorer(
        _ALL_IDS,
        3,
        [
            _uniform({"suite-a": (0.8, True), "suite-b": (0.8, True), "verify-1": (0.0, False)}, 3),
            # Suite mean 0.6 is below the floor 0.72 = 0.9 * 0.8 even though verification passes.
            _uniform({"suite-a": (0.6, True), "suite-b": (0.6, True), "verify-1": (1.0, True)}, 3),
        ],
    )
    proposer = _Proposer([_source("regressed")])

    outcome = _improve(scorer=scorer, proposer=proposer, run_dir=tmp_path / "run", iterations=1)

    assert outcome.accepted is False
    assert outcome.selected is None
    assert outcome.candidate_suite_score is None
    assert outcome.verification == ()
    assert "suite regression beyond margin" in outcome.reason
    assert "0.600000" in outcome.reason
    assert "0.720000" in outcome.reason


def test_gate_rejects_when_a_verification_task_fails_majority(tmp_path: Path) -> None:
    scorer = _Scorer(
        _ALL_IDS,
        3,
        [
            _uniform({"suite-a": (0.8, True), "suite-b": (0.8, True), "verify-1": (0.0, False)}, 3),
            {
                "suite-a": [(0.9, True)] * 3,
                "suite-b": [(0.9, True)] * 3,
                "verify-1": [(1.0, True), (0.0, False), (0.0, False)],  # 1 of 3: no majority
            },
        ],
    )
    proposer = _Proposer([_source("half done")])

    outcome = _improve(scorer=scorer, proposer=proposer, run_dir=tmp_path / "run", iterations=1)

    assert outcome.accepted is False
    assert outcome.selected is None
    assert "verification tasks failed" in outcome.reason
    assert "verify-1" in outcome.reason
    assert "1/3" in outcome.reason


def test_selection_prefers_highest_suite_score_over_optimizer_blended_best(
    tmp_path: Path,
) -> None:
    scorer = _Scorer(
        _ALL_IDS,
        3,
        [
            _uniform({"suite-a": (0.5, True), "suite-b": (0.5, True), "verify-1": (0.0, False)}, 3),
            # Higher suite (0.9) but a partial verify (2 of 3) -> blended per-task pass rate 0.889.
            {
                "suite-a": [(0.9, True)] * 3,
                "suite-b": [(0.9, True)] * 3,
                "verify-1": [(1.0, True), (1.0, True), (0.0, False)],
            },
            # Lower suite (0.7) but all verify passes -> blended pass rate 1.0 (optimizer's best).
            _uniform({"suite-a": (0.7, True), "suite-b": (0.7, True), "verify-1": (1.0, True)}, 3),
        ],
    )
    proposer = _Proposer([_source("high suite"), _source("high verification")])

    outcome = _improve(scorer=scorer, proposer=proposer, run_dir=tmp_path / "run", iterations=2)

    # The optimizer's blended best is candidate-0002 (mean per-task pass rate 1.0 > 0.889); our
    # gate picks the higher-suite candidate-0001 (0.9 > 0.7) instead.
    assert outcome.result.best.candidate_id == "candidate-0002"
    assert outcome.accepted is True
    assert outcome.selected is not None
    assert outcome.selected.candidate_id == "candidate-0001"
    assert outcome.candidate_suite_score == pytest.approx(0.9)


def test_selection_tie_on_suite_score_keeps_the_earliest_candidate(tmp_path: Path) -> None:
    plan = _uniform({"suite-a": (0.8, True), "suite-b": (0.8, True), "verify-1": (1.0, True)}, 1)
    scorer = _Scorer(
        _ALL_IDS,
        1,
        [
            _uniform({"suite-a": (0.8, True), "suite-b": (0.8, True), "verify-1": (0.0, False)}, 1),
            plan,
            plan,
        ],
    )
    proposer = _Proposer([_source("first tie"), _source("second tie")])

    outcome = _improve(scorer=scorer, proposer=proposer, run_dir=tmp_path / "run", iterations=2)

    assert outcome.accepted is True
    assert outcome.selected is not None
    assert outcome.selected.candidate_id == "candidate-0001"


def test_seed_only_population_reports_no_valid_proposals(tmp_path: Path) -> None:
    scorer = _Scorer(
        _ALL_IDS,
        3,
        [
            _uniform({"suite-a": (0.8, True), "suite-b": (0.8, True), "verify-1": (0.0, False)}, 3),
        ],
    )
    proposer = _Proposer([CandidateProposalError("candidate-0001", "agent turn did not submit")])

    outcome = _improve(scorer=scorer, proposer=proposer, run_dir=tmp_path / "run", iterations=1)

    assert outcome.accepted is False
    assert "no valid proposals" in outcome.reason
    assert outcome.selected is None
    assert outcome.candidate_suite_score is None
    assert outcome.verification == ()
    assert [item.candidate_id for item in outcome.result.population] == ["candidate-0000"]
    assert outcome.seed_suite_score == pytest.approx(0.8)


def _report(task_ids: Sequence[str], attempts: int, plan: _CellPlan) -> ScoreReport:
    scorer = _Scorer(task_ids, attempts, [plan])
    doc = _source("unit").to_doc("candidate-0000")
    return scorer.score(doc)


def test_suite_score_means_only_matching_cells_and_rejects_empty_selection() -> None:
    report = _report(
        ("suite-a", "suite-b", "verify-1"),
        2,
        {
            "suite-a": [(1.0, True), (0.0, False)],
            "suite-b": [(0.5, True), (0.5, True)],
            "verify-1": [(0.0, False), (0.0, False)],
        },
    )

    assert suite_score(report, ("suite-a", "suite-b")) == pytest.approx(0.5)
    assert suite_score(report, ("suite-a",)) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="no cells"):
        suite_score(report, ("absent-task",))
    with pytest.raises(ValueError, match="no cells"):
        suite_score(report, ())


def test_verification_results_apply_the_strict_majority_rule() -> None:
    half = _report(("verify-1",), 2, {"verify-1": [(1.0, True), (0.0, False)]})
    [result] = verification_results(half, ("verify-1",))
    assert result.passed is False  # exactly half (1 of 2) is not a strict majority
    assert result.pass_count == 1
    assert result.attempts == 2

    majority = _report(("verify-1",), 3, {"verify-1": [(1.0, True), (1.0, True), (0.0, False)]})
    [result] = verification_results(majority, ("verify-1",))
    assert result.passed is True
    assert result.pass_count == 2
    assert result.attempts == 3

    with pytest.raises(ValueError, match="absent-task"):
        verification_results(majority, ("absent-task",))


def test_improve_validates_task_sets_and_request_before_any_proposal(tmp_path: Path) -> None:
    proposer = _Proposer([])
    valid_scorer = _Scorer(_ALL_IDS, 1, [])

    with pytest.raises(ValueError, match="disjoint"):
        _improve(
            scorer=valid_scorer,
            proposer=proposer,
            run_dir=tmp_path / "a",
            iterations=1,
            verification=("suite-a",),
        )
    with pytest.raises(ValueError, match="suite_task_ids must be nonempty"):
        _improve(
            scorer=valid_scorer, proposer=proposer, run_dir=tmp_path / "b", iterations=1, suite=()
        )
    with pytest.raises(ValueError, match="verification_task_ids must be nonempty"):
        _improve(
            scorer=valid_scorer,
            proposer=proposer,
            run_dir=tmp_path / "c",
            iterations=1,
            verification=(),
        )
    with pytest.raises(ValueError, match="feedback"):
        _improve(
            scorer=valid_scorer,
            proposer=proposer,
            run_dir=tmp_path / "d",
            iterations=1,
            feedback="   ",
        )

    reordered_scorer = _Scorer(("verify-1", "suite-a", "suite-b"), 1, [])
    with pytest.raises(ValueError, match="suite tasks followed by the verification tasks"):
        _improve(scorer=reordered_scorer, proposer=proposer, run_dir=tmp_path / "e", iterations=1)
    assert proposer.slots == []


def test_margin_gate_validates_its_fraction() -> None:
    assert ImproveGate().suite_margin == pytest.approx(0.10)
    assert ImproveGate(suite_margin=0.0).suite_margin == 0.0
    for invalid in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError):
            ImproveGate(suite_margin=invalid)
