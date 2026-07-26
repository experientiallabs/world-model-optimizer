"""End-to-end create-loop tests: one provider plays all four roles, no network, no entropy.

Extends the closed-loop test pattern (`closed_loop_test.RoleProvider`) with a fourth role: the
meta-agent, keyed on `MUTATE_SYSTEM`'s distinctive phrase. The agent role is the FALLBACK after
the judge/meta/world-model markers, because a variant's system prompt is exactly what the search
rewrites — no marker on it is stable across generations. The fake judge echoes the gold assertions
verbatim from its prompt (the real judge is scored by text-matching those echoes) and passes or
fails a run based on the submitted answer, so seed and child scores can genuinely differ.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import ClassVar

import pytest
from llm_waterfall import ChatRequest, ChatResponse

from wmh.core.types import JsonObject
from wmh.engine.world_model import WorldModel
from wmh.evals.closed_loop import ClosedLoopReport, TaskOutcome, WorldModelSource
from wmh.evals.gold import AssertionResult, GoldJudge, GoldVerdict
from wmh.evals.tasks import TaskSpec
from wmh.harness import create as create_module
from wmh.harness.create import (
    CreateResult,
    HarnessSearchCancelled,
    ProposalRecord,
    cluster_failures,
    create_harness,
    select_failure_cluster,
)
from wmh.harness.delta import FailureSignature, GateRecord, HarnessDelta
from wmh.harness.doc import HarnessDoc
from wmh.harness.e2b_sandbox import SandboxUsage
from wmh.harness.environment import EnvironmentSource
from wmh.harness.mutate import parse_delta
from wmh.harness.proposer import ProposalFailure, ProviderDeltaProposer
from wmh.harness.runtime import Runtime
from wmh.providers.base import Completion, Message, Provider, ProviderConfig, ProviderKind
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder

_CAREFUL_PROMPT = "You are a careful agent. Verify the state of the system before submitting."


def _meta_reply(parent: HarnessDoc, new_prompt: str) -> str:
    """A well-formed delta against `parent`, preconditioned on its actual prompt hash."""
    core = parent.surface("prompt:core")
    assert core is not None
    return json.dumps(
        {
            "expected_effect": "the failing tasks flip to pass",
            "preconditions": {"prompt:core": core.content_hash},
            "ops": [
                {
                    "op": "replace",
                    "surface_id": "prompt:core",
                    "content": new_prompt,
                    "rationale": "make the agent verify before submitting",
                }
            ],
        }
    )


class RoleProvider:
    """Plays agent, world model, gold judge, and meta-agent, keyed off the system prompt."""

    def __init__(
        self,
        *,
        meta_reply: str = "not json at all",
        judge_fn: Callable[[str], bool] | None = None,
    ) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")
        self._meta_reply = meta_reply
        self.meta_users: list[str] = []  # every proposer prompt, for history assertions
        # Default: a run passes iff the agent submitted the verified answer.
        self._judge_fn = judge_fn if judge_fn is not None else lambda u: "done-verified" in u

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        user = messages[-1].content
        if "grade whether an agent completed a task" in system:
            passed = self._judge_fn(user)
            results = [
                {"assertion": a, "passed": passed, "why": "x"} for a in _gold_assertions(user)
            ]
            return Completion(text=json.dumps({"assertions": results, "passed": passed}))
        if "meta-agent improving an agent harness" in system:
            self.meta_users.append(user)
            return Completion(text=self._meta_reply)
        if "You ARE the environment" in system:
            return Completion(text='{"output": "ok", "is_error": false}')
        # Fallback: the agent role. What it submits depends on the prompt the variant carries.
        if "careful agent" in system:
            answer = "done-verified"
        elif "broken agent" in system:
            answer = "done-broken"
        else:
            answer = "done"
        return Completion(text=json.dumps({"tool": "submit", "arguments": {"answer": answer}}))

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

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201 - test fake never calls it
        raise NotImplementedError


def _gold_assertions(user: str) -> list[str]:
    """The gold list the judge prompt carries, echoed back verbatim."""
    _, _, tail = user.partition("GOLD ASSERTIONS")
    return [line[2:] for line in tail.splitlines() if line.startswith("- ")]


def _wm(provider: RoleProvider) -> WorldModelSource:
    """The environment under search: a locally loaded world model driven by `provider`."""
    return WorldModelSource(WorldModel(provider, EmbeddingRetriever(HashingEmbedder(dim=16))))


def _tasks() -> list[TaskSpec]:
    return [TaskSpec(task_id="t1", instruction="answer it", gold=["the work was verified"])]


def _run(
    provider: RoleProvider,
    *,
    iterations: int = 1,
    k: int = 3,
    proposal_batch_size: int = 1,
    holdout: list[TaskSpec] | None = None,
    on_progress: Callable[[int, str, float, bool], None] | None = None,
    on_note: Callable[[str], None] | None = None,
    on_proposal: Callable[[ProposalRecord], None] | None = None,
    on_accept: Callable[[HarnessDoc, HarnessDelta, float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> CreateResult:
    return create_harness(
        "winner",
        HarnessDoc.baseline("seed"),
        _tasks(),
        _wm(provider),
        provider,
        ProviderDeltaProposer(provider),
        GoldJudge(provider),
        iterations=iterations,
        proposal_batch_size=proposal_batch_size,
        k=k,
        holdout=holdout,
        on_progress=on_progress,
        on_note=on_note,
        on_proposal=on_proposal,
        on_accept=on_accept,
        should_cancel=should_cancel,
    )


def test_create_accepts_improving_delta_and_promotes_suite() -> None:
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT))
    progress: list[tuple[int, str, float, bool]] = []
    result = _run(provider, on_progress=lambda i, n, r, a: progress.append((i, n, r, a)))

    assert result.skipped == 0
    assert result.best_score == 1.0
    assert result.best.name == "winner"
    assert result.best.system_prompt() == _CAREFUL_PROMPT
    assert progress == [(0, "seed", 0.0, True), (1, "winner-i1-p1", 1.0, True)]

    [delta] = result.archive.deltas
    assert delta.verdict is not None and delta.verdict.accepted
    assert delta.verdict.full_delta == 1.0
    assert delta.verdict.holdout_delta is None
    assert "1/1 tasks now pass" in delta.verdict.reason
    # The trigger came from deterministic clustering of the seed's failures.
    assert delta.trigger.mechanism == "the work was verified"
    assert delta.trigger.task_ids == ["t1"]
    # The newly-passing task promoted into the regression suite.
    assert result.suite == ["t1"]
    # Reports are keyed by content: seed and child doc hashes, k=3 passes each.
    assert set(result.reports) == {seed.doc_hash, delta.child_doc_hash}
    assert all(r.k == 3 for r in result.reports.values())


def test_archive_reconstructs_children_by_folding_deltas() -> None:
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT))
    result = _run(provider)
    [delta] = result.archive.deltas
    assert delta.child_doc_hash is not None
    rebuilt = result.archive.reconstruct(delta.child_doc_hash)
    assert rebuilt.surfaces == result.best.surfaces
    with pytest.raises(ValueError, match="not in this archive"):
        result.archive.reconstruct("0" * 32)


def test_create_rejects_regressing_delta_and_keeps_champion() -> None:
    # The seed already passes; the proposed prompt makes the agent submit a broken answer.
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(
        meta_reply=_meta_reply(seed, "You are a broken agent."),
        judge_fn=lambda user: "done-broken" not in user,
    )
    result = _run(provider)

    assert result.skipped == 0
    [delta] = result.archive.deltas
    assert delta.verdict is not None and not delta.verdict.accepted
    assert "full split" in delta.verdict.reason
    assert result.archive.accepted() == []
    # The champion never moved: the winner is the (renamed) seed at its original score.
    assert result.best_score == 1.0
    assert result.best.system_prompt() == seed.system_prompt()
    assert result.suite == ["t1"]  # and the suite kept the seed's win
    # An all-pass parent gets the generalization trigger, not a fabricated failure.
    assert delta.trigger.mechanism == "none: all tasks pass"


def test_create_skips_unusable_proposals() -> None:
    provider = RoleProvider(meta_reply="not json at all")
    result = _run(provider, iterations=2)
    assert result.skipped == 2
    assert result.archive.deltas == []  # nothing to audit: no delta object ever existed
    assert result.best.name == "winner"  # even a search with no children yields the renamed seed
    # Every iteration is recorded even when its proposal dies before producing anything.
    assert [(r.iteration, r.outcome) for r in result.proposal_records] == [
        (1, "unusable"),
        (2, "unusable"),
    ]


def test_create_stops_before_the_next_expensive_phase_when_cancelled() -> None:
    provider = RoleProvider(meta_reply="not json at all")
    checks = 0

    def should_cancel() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 2

    with pytest.raises(HarnessSearchCancelled):
        _run(provider, should_cancel=should_cancel)

    assert provider.meta_users == []


def test_create_passes_cancellation_into_a_batched_provider_proposer() -> None:
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT))

    with pytest.raises(HarnessSearchCancelled, match="cancelled"):
        _run(
            provider,
            proposal_batch_size=3,
            should_cancel=lambda: len(provider.meta_users) >= 1,
        )

    assert len(provider.meta_users) == 1


def test_create_never_converts_explicit_proposer_cancellation_to_failures() -> None:
    class _CancellingMetaProvider(RoleProvider):
        def complete(
            self,
            system: str,
            messages: list[Message],
            *,
            temperature: float = 0.7,
            max_tokens: int = 2048,
        ) -> Completion:
            if "meta-agent improving an agent harness" in system:
                raise HarnessSearchCancelled("harness search cancelled")
            return super().complete(
                system,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    with pytest.raises(HarnessSearchCancelled, match="cancelled"):
        _run(_CancellingMetaProvider())


def test_cancellation_wins_before_accepted_lineage_and_callback_mutate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT))
    cancelled = False
    crowned: list[str] = []
    gate_delta = create_module.gate_delta

    def cancelling_gate(
        delta: HarnessDelta,
        *,
        child: ClosedLoopReport,
        champion: ClosedLoopReport,
        best_full: float,
        suite: list[str],
        child_holdout: ClosedLoopReport | None = None,
        champion_holdout: ClosedLoopReport | None = None,
    ) -> GateRecord:
        nonlocal cancelled
        verdict = gate_delta(
            delta,
            child=child,
            champion=champion,
            best_full=best_full,
            suite=suite,
            child_holdout=child_holdout,
            champion_holdout=champion_holdout,
        )
        if verdict.accepted:
            cancelled = True
        return verdict

    monkeypatch.setattr(create_module, "gate_delta", cancelling_gate)

    with pytest.raises(HarnessSearchCancelled, match="cancelled"):
        _run(
            provider,
            should_cancel=lambda: cancelled,
            on_accept=lambda doc, delta, score: crowned.append(doc.doc_hash),
        )

    assert crowned == []


def test_create_audits_invalid_delta_without_spending_eval() -> None:
    stale = json.dumps(
        {
            "expected_effect": "x",
            "preconditions": {"prompt:core": "0" * 32},
            "ops": [
                {
                    "op": "replace",
                    "surface_id": "prompt:core",
                    "content": _CAREFUL_PROMPT,
                    "rationale": "r",
                }
            ],
        }
    )
    provider = RoleProvider(meta_reply=stale)
    result = _run(provider)
    assert result.skipped == 1
    [delta] = result.archive.deltas
    assert delta.verdict is not None and not delta.verdict.accepted
    assert delta.verdict.reason.startswith("invalid before eval")
    assert delta.child_doc_hash is None  # it never applied, so it never produced a doc
    assert len(result.reports) == 1  # only the seed was ever scored


def test_holdout_regression_rejects_a_full_split_win() -> None:
    # The delta fixes the main task but breaks the held-out one: tiers 1-2 pass, tier 3 rejects.
    seed = HarnessDoc.baseline("seed")

    def judge(user: str) -> bool:
        if "the holdout task" in user:
            return "done-verified" not in user  # holdout passes only for the seed's plain answer
        return "done-verified" in user

    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT), judge_fn=judge)
    holdout = [TaskSpec(task_id="h1", instruction="the holdout task", gold=["the base flow works"])]
    result = _run(provider, holdout=holdout)

    [delta] = result.archive.deltas
    assert delta.verdict is not None and not delta.verdict.accepted
    assert delta.verdict.full_delta == 1.0  # it really did win the full split...
    assert delta.verdict.holdout_delta == -1.0  # ...and really did regress held-out
    assert "held-out regressed" in delta.verdict.reason
    assert result.best_score == 0.0  # champion stays the seed
    assert set(result.holdout_reports) == {seed.doc_hash, delta.child_doc_hash}


# -- deterministic failure clustering ---------------------------------------------------------


def _failing(task_id: str, unmet: list[str]) -> TaskOutcome:
    verdict = GoldVerdict(
        passed=False,
        fraction=0.0,
        assertions=[AssertionResult(assertion=a, passed=False, why="w") for a in unmet],
    )
    return TaskOutcome(
        task_id=task_id, success_rate=0.0, mean_fraction=0.0, passes=2, verdicts=[verdict, verdict]
    )


def test_cluster_failures_groups_by_shared_assertions() -> None:
    report = ClosedLoopReport(
        per_task={
            "t1": _failing("t1", ["a", "b"]),
            "t2": _failing("t2", ["b", "c"]),
            "t3": _failing("t3", ["z"]),
            "t4": TaskOutcome(task_id="t4", success_rate=1.0, mean_fraction=1.0, passes=2),
            "t5": _failing("t5", []),  # unparseable judge: no per-assertion detail
        }
    )
    tasks = [TaskSpec(task_id=t, instruction=t) for t in ("t1", "t2", "t3", "t4", "t5")]
    clusters = cluster_failures(report, tasks)
    assert [c.task_ids for c in clusters] == [["t1", "t2"], ["t5"], ["t3"]]
    # t1+t2 connect through shared assertion "b", which also labels the mechanism.
    assert clusters[0].mechanism == "b"
    assert clusters[0].unmet_assertions == ["a", "b", "c"]
    assert clusters[1].mechanism == "run failed without per-assertion verdicts"
    assert clusters[2].mechanism == "z"


def test_cluster_failures_empty_when_everything_passes() -> None:
    report = ClosedLoopReport(
        per_task={"t1": TaskOutcome(task_id="t1", success_rate=1.0, mean_fraction=1.0, passes=3)}
    )
    assert cluster_failures(report, [TaskSpec(task_id="t1", instruction="i")]) == []


def test_select_failure_cluster_rotates_equally_sized_failures() -> None:
    clusters = [
        FailureSignature(mechanism=mechanism, task_ids=[f"t-{mechanism}"])
        for mechanism in ("a", "b", "c")
    ]
    counts: dict[tuple[str, str, tuple[str, ...]], int] = {}
    selected: list[str] = []
    for _ in range(4):
        cluster = select_failure_cluster(clusters, counts, parent_doc_hash="parent")
        selected.append(cluster.mechanism)
        key = ("parent", cluster.mechanism, tuple(cluster.task_ids))
        counts[key] = counts.get(key, 0) + 1

    assert selected == ["a", "b", "c", "a"]


def test_create_rotates_failure_evidence_after_a_screened_batch() -> None:
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(meta_reply=_useless_meta_reply(seed), judge_fn=lambda _user: False)
    tasks = [
        TaskSpec(task_id="t1", instruction="first failure", gold=["alpha assertion"]),
        TaskSpec(task_id="t2", instruction="second failure", gold=["beta assertion"]),
    ]

    create_harness(
        "winner",
        seed,
        tasks,
        _wm(provider),
        provider,
        ProviderDeltaProposer(provider),
        GoldJudge(provider),
        iterations=2,
        k=1,
    )

    assert "[TARGET] t1" in provider.meta_users[0]
    assert "[other] t2" in provider.meta_users[0]
    assert "[TARGET] t2" in provider.meta_users[1]
    assert "[other] t1" in provider.meta_users[1]


def test_create_does_not_discount_a_cluster_when_the_proposer_failed() -> None:
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(judge_fn=lambda _user: False)
    tasks = [
        TaskSpec(task_id="t1", instruction="first failure", gold=["alpha assertion"]),
        TaskSpec(task_id="t2", instruction="second failure", gold=["beta assertion"]),
    ]

    class FailingProposer:
        def __init__(self) -> None:
            self.triggers: list[FailureSignature] = []

        def propose_batch(  # noqa: PLR0913 - mirrors the proposer protocol
            self,
            parent: HarnessDoc,
            trigger: FailureSignature,
            evidence: str,
            *,
            history: list[HarnessDelta],
            count: int,
            should_cancel: Callable[[], bool] | None = None,
        ) -> list[HarnessDelta | ProposalFailure | None]:
            del parent, evidence, history, should_cancel
            self.triggers.append(trigger)
            return [ProposalFailure(reason="temporary transport failure")] * count

    proposer = FailingProposer()
    create_harness(
        "winner",
        seed,
        tasks,
        _wm(provider),
        provider,
        proposer,
        GoldJudge(provider),
        iterations=2,
        k=1,
    )

    assert [trigger.task_ids for trigger in proposer.triggers] == [["t1"], ["t1"]]


def test_create_does_not_discount_a_cluster_when_every_delta_is_invalid() -> None:
    """Parsed deltas spend no cluster allocation until one can enter evaluation."""
    seed = HarnessDoc.baseline("seed")
    stale = json.dumps(
        {
            "expected_effect": "fix the selected failure",
            "preconditions": {"prompt:core": "0" * 32},
            "ops": [
                {
                    "op": "replace",
                    "surface_id": "prompt:core",
                    "content": _CAREFUL_PROMPT,
                    "rationale": "exercise the invalid-before-eval path",
                }
            ],
        }
    )
    provider = RoleProvider(meta_reply=stale, judge_fn=lambda _user: False)
    tasks = [
        TaskSpec(task_id="t1", instruction="first failure", gold=["alpha assertion"]),
        TaskSpec(task_id="t2", instruction="second failure", gold=["beta assertion"]),
    ]

    result = create_harness(
        "winner",
        seed,
        tasks,
        _wm(provider),
        provider,
        ProviderDeltaProposer(provider),
        GoldJudge(provider),
        iterations=2,
        k=1,
    )

    assert result.skipped == 2
    assert "[TARGET] t1" in provider.meta_users[0]
    assert "[TARGET] t1" in provider.meta_users[1]
    assert "[other] t2" in provider.meta_users[0]
    assert "[other] t2" in provider.meta_users[1]


# -- staged verification: screening + history ---------------------------------------------------


def _useless_meta_reply(parent: HarnessDoc) -> str:
    """A well-formed delta that changes wording but cannot fix the failing task."""
    return _meta_reply(parent, "You are an agent. Do the task.")


def test_screen_rejects_delta_that_does_not_improve_its_trigger() -> None:
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(meta_reply=_useless_meta_reply(seed))
    progress: list[tuple[int, str, float, bool]] = []
    result = _run(provider, on_progress=lambda i, n, r, a: progress.append((i, n, r, a)))

    assert result.screened == 1 and result.skipped == 0
    [delta] = result.archive.deltas
    assert delta.verdict is not None and not delta.verdict.accepted
    assert delta.verdict.reason.startswith("screened out")
    # The dead iteration is still a first-class record, with its screen means attached.
    [record] = result.proposal_records
    assert record.iteration == 1 and record.outcome == "screened"
    assert record.screen_child is not None and record.screen_parent is not None
    # The cheap screen replaced the full eval: only the seed has a full-split report. The
    # iteration still emits one unchanged champion checkpoint.
    assert len(result.reports) == 1
    assert progress == [(0, "seed", 0.0, True), (1, "seed", 0.0, False)]


def test_screen_uses_assertion_fraction_to_admit_partial_improvement() -> None:
    seed = HarnessDoc.baseline("seed")

    class PartialJudgeProvider(RoleProvider):
        def complete(
            self,
            system: str,
            messages: list[Message],
            *,
            temperature: float = 0.7,
            max_tokens: int = 2048,
        ) -> Completion:
            if "grade whether an agent completed a task" not in system:
                return super().complete(
                    system,
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            user = messages[-1].content
            improved = "done-verified" in user
            assertions = _gold_assertions(user)
            results = [
                {
                    "assertion": assertion,
                    "passed": improved and index == 0,
                    "why": "one subgoal improved" if improved and index == 0 else "still missing",
                }
                for index, assertion in enumerate(assertions)
            ]
            return Completion(text=json.dumps({"assertions": results, "passed": False}))

    provider = PartialJudgeProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT))
    tasks = [
        TaskSpec(
            task_id="t1",
            instruction="complete both parts",
            gold=["part one complete", "part two complete"],
        )
    ]

    result = create_harness(
        "winner",
        seed,
        tasks,
        _wm(provider),
        provider,
        ProviderDeltaProposer(provider),
        GoldJudge(provider),
        iterations=1,
        k=1,
    )

    assert result.screened == 0
    [record] = result.proposal_records
    assert record.outcome == "scored"
    assert record.screen_child == record.screen_parent == 0.0
    assert record.screen_parent_fraction == 0.0
    assert record.screen_child_fraction == 0.5


def test_gate_rejects_target_partial_lift_when_full_split_partial_credit_regresses() -> None:
    """Dense screening is a prefilter; the authoritative full gate protects other tasks."""
    delta = HarnessDelta.model_construct(
        trigger=FailureSignature(mechanism="target", task_ids=["target"])
    )
    champion = ClosedLoopReport(
        success_rate=0.0,
        mean_fraction=0.45,
        per_task={
            "target": TaskOutcome(task_id="target", success_rate=0.0, mean_fraction=0.0),
            "other": TaskOutcome(task_id="other", success_rate=0.0, mean_fraction=0.9),
        },
    )
    child = ClosedLoopReport(
        success_rate=0.0,
        mean_fraction=0.25,
        per_task={
            "target": TaskOutcome(task_id="target", success_rate=0.0, mean_fraction=0.5),
            "other": TaskOutcome(task_id="other", success_rate=0.0, mean_fraction=0.0),
        },
    )

    verdict = create_module.gate_delta(
        delta,
        child=child,
        champion=champion,
        best_full=0.0,
        suite=[],
    )

    assert verdict.accepted is False
    assert verdict.full_delta == 0.0
    assert verdict.full_fraction_delta == pytest.approx(-0.2)
    assert "full-split assertion fraction regressed" in verdict.reason


def test_gate_accepts_binary_tie_with_nonregressing_global_partial_progress() -> None:
    delta = HarnessDelta.model_construct(
        trigger=FailureSignature(mechanism="target", task_ids=["target"])
    )
    champion = ClosedLoopReport(
        success_rate=0.0,
        mean_fraction=0.1,
        per_task={
            "target": TaskOutcome(task_id="target", success_rate=0.0, mean_fraction=0.0),
            "other": TaskOutcome(task_id="other", success_rate=0.0, mean_fraction=0.2),
        },
    )
    child = ClosedLoopReport(
        success_rate=0.0,
        mean_fraction=0.35,
        per_task={
            "target": TaskOutcome(task_id="target", success_rate=0.0, mean_fraction=0.5),
            "other": TaskOutcome(task_id="other", success_rate=0.0, mean_fraction=0.2),
        },
    )

    verdict = create_module.gate_delta(
        delta,
        child=child,
        champion=champion,
        best_full=0.0,
        suite=[],
    )

    assert verdict.accepted is True
    assert verdict.full_fraction_delta == pytest.approx(0.25)


def test_search_records_screen_and_full_trace_feedback_for_project_proposers() -> None:
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT))

    class RecordingProposer(ProviderDeltaProposer):
        def __init__(self, wrapped: Provider) -> None:
            super().__init__(wrapped)
            self.evaluations: list[tuple[str, str]] = []

        def record_evaluation(self, delta: HarnessDelta, *, stage: str, content: str) -> None:
            del delta
            self.evaluations.append((stage, content))

    proposer = RecordingProposer(provider)
    create_harness(
        "winner",
        seed,
        _tasks(),
        _wm(provider),
        provider,
        proposer,
        GoldJudge(provider),
        iterations=1,
        k=1,
    )

    assert [stage for stage, _content in proposer.evaluations] == ["screen", "full"]
    assert all("Execution transcript" in content for _stage, content in proposer.evaluations)
    assert all("Judge feedback" in content for _stage, content in proposer.evaluations)


def test_feedback_persistence_failure_does_not_abort_scored_search() -> None:
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT))
    notes: list[str] = []

    class BrokenFeedbackProposer(ProviderDeltaProposer):
        def record_evaluation(self, delta: HarnessDelta, *, stage: str, content: str) -> None:
            del delta, stage, content
            raise RuntimeError("project filesystem disconnected")

    result = create_harness(
        "winner",
        seed,
        _tasks(),
        _wm(provider),
        provider,
        BrokenFeedbackProposer(provider),
        GoldJudge(provider),
        iterations=1,
        k=1,
        on_note=notes.append,
    )

    assert result.best_score == 1.0
    assert len(result.proposal_records) == 1
    assert any("screen feedback could not be persisted" in note for note in notes)
    assert any("full feedback could not be persisted" in note for note in notes)


def test_feedback_persistence_preserves_explicit_cancellation() -> None:
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT))

    class CancellingFeedbackProposer(ProviderDeltaProposer):
        def record_evaluation(self, delta: HarnessDelta, *, stage: str, content: str) -> None:
            del delta, stage, content
            raise HarnessSearchCancelled("harness search cancelled")

    with pytest.raises(HarnessSearchCancelled, match="cancelled"):
        create_harness(
            "winner",
            seed,
            _tasks(),
            _wm(provider),
            provider,
            CancellingFeedbackProposer(provider),
            GoldJudge(provider),
            iterations=1,
            k=1,
        )


def test_rejected_history_reaches_the_next_proposal() -> None:
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(meta_reply=_useless_meta_reply(seed))
    _run(provider, iterations=2)
    assert len(provider.meta_users) == 2
    assert "Previous attempts" not in provider.meta_users[0]
    assert "Previous attempts" in provider.meta_users[1]
    assert "screened out" in provider.meta_users[1]  # the verdict itself is the lesson


# -- code deltas end to end ----------------------------------------------------------------------


def _code_meta_reply(parent: HarnessDoc) -> str:
    from wmh.harness.doc import CODE_RUNTIME_ID

    code_surface = parent.surface(CODE_RUNTIME_ID)
    assert code_surface is not None
    new_code = (
        "def run(kit):\n"
        '    kit.execute("bash", {"command": "verify the work"})\n'
        '    return "done-verified"\n'
    )
    return json.dumps(
        {
            "expected_effect": "the failing task flips to pass",
            "preconditions": {CODE_RUNTIME_ID: code_surface.content_hash},
            "ops": [
                {
                    "op": "replace",
                    "surface_id": CODE_RUNTIME_ID,
                    "content": new_code,
                    "rationale": "verify via the environment before submitting",
                }
            ],
        }
    )


def test_code_delta_passes_screen_and_gate_end_to_end() -> None:
    from wmh.harness.doc import CODE_RUNTIME_ID, code_baseline

    seed = code_baseline("seed")
    provider = RoleProvider(meta_reply=_code_meta_reply(seed))
    result = create_harness(
        "winner",
        seed,
        _tasks(),
        _wm(provider),
        provider,
        ProviderDeltaProposer(provider),
        GoldJudge(provider),
        iterations=1,
        k=3,
    )
    assert result.screened == 0 and result.skipped == 0
    [delta] = result.archive.deltas
    assert delta.verdict is not None and delta.verdict.accepted
    assert [op.surface_id for op in delta.ops] == [CODE_RUNTIME_ID]
    assert result.best_score == 1.0
    winner_code = result.best.surface(CODE_RUNTIME_ID)
    assert winner_code is not None and "done-verified" in winner_code.content


def test_broken_code_delta_is_rejected_before_any_eval() -> None:
    from wmh.harness.doc import CODE_RUNTIME_ID, code_baseline

    seed = code_baseline("seed")
    code_surface = seed.surface(CODE_RUNTIME_ID)
    assert code_surface is not None
    broken = json.dumps(
        {
            "expected_effect": "x",
            "preconditions": {CODE_RUNTIME_ID: code_surface.content_hash},
            "ops": [
                {
                    "op": "replace",
                    "surface_id": CODE_RUNTIME_ID,
                    "content": "def run(kit:\n",
                    "rationale": "r",
                }
            ],
        }
    )
    provider = RoleProvider(meta_reply=broken)
    result = create_harness(
        "winner",
        seed,
        _tasks(),
        _wm(provider),
        provider,
        ProviderDeltaProposer(provider),
        GoldJudge(provider),
        iterations=1,
        k=2,
    )
    assert result.skipped == 1
    [delta] = result.archive.deltas
    assert delta.verdict is not None and "does not compile" in delta.verdict.reason
    assert len(result.reports) == 1  # only the seed was ever scored


# -- confirmation re-runs -----------------------------------------------------------------------


def test_narrow_failing_tiers_eligibility() -> None:
    from wmh.harness.create import narrow_failing_tiers
    from wmh.harness.delta import GateRecord

    def record(**kw) -> GateRecord:  # noqa: ANN003
        return GateRecord(accepted=False, reason="r", **kw)

    # Narrow holdout veto on a full-split win -> retry that tier.
    narrow = record(full_delta=0.05, holdout_delta=-0.1)
    assert narrow_failing_tiers(narrow, k=5, n_suite=4, n_holdout=4) == ["holdout"]
    # A wide veto is a real regression, not noise: ineligible.
    wide = record(full_delta=0.05, holdout_delta=-1.0)
    assert narrow_failing_tiers(wide, k=5, n_suite=4, n_holdout=4) is None
    # No full-split win: nothing to confirm.
    no_win = record(full_delta=0.0, holdout_delta=-0.1)
    assert narrow_failing_tiers(no_win, k=5, n_suite=4, n_holdout=4) is None
    # Both tiers narrowly failing -> both retried.
    both = record(full_delta=0.05, suite_delta=-0.05, holdout_delta=-0.1)
    assert narrow_failing_tiers(both, k=5, n_suite=8, n_holdout=4) == ["suite", "holdout"]
    # Confirmation of one binary veto cannot erase a separate dense-signal veto.
    dense_veto = record(
        full_delta=0.05,
        suite_delta=-0.05,
        holdout_delta=0.0,
        holdout_fraction_delta=-0.2,
    )
    assert narrow_failing_tiers(dense_veto, k=5, n_suite=8, n_holdout=4) is None
    # Accepted verdicts are never retried.
    ok = GateRecord(accepted=True, reason="r", full_delta=0.05)
    assert narrow_failing_tiers(ok, k=5, n_suite=4, n_holdout=4) is None


def test_flaky_holdout_veto_is_overturned_by_confirmation() -> None:
    # The child genuinely fixes the train task; the holdout task fails exactly ONE child
    # attempt (judge flakiness). The initial k-pass gate vetoes; the 2k re-measurement of
    # child AND champion overturns it.
    seed = HarnessDoc.baseline("seed")
    child_h1_calls = {"n": 0}

    def judge(user: str) -> bool:
        if "the holdout task" in user:
            if "done-verified" in user:  # the child's answer style
                child_h1_calls["n"] += 1
                return child_h1_calls["n"] != 1  # fail only the first child attempt
            return True  # the seed always passes holdout
        return "done-verified" in user  # train task needs the careful child

    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT), judge_fn=judge)
    holdout = [TaskSpec(task_id="h1", instruction="the holdout task", gold=["the base flow works"])]
    result = _run(provider, holdout=holdout, k=5)

    assert result.confirmations == 1
    [delta] = result.archive.deltas
    assert delta.verdict is not None and delta.verdict.accepted
    assert "veto overturned" in delta.verdict.reason
    assert "initially: rejected" in delta.verdict.reason
    assert result.best_score == 1.0  # the win was kept


def test_wide_holdout_regression_skips_confirmation() -> None:
    # Same setup as the second-iteration holdout test: the child ALWAYS fails held-out. -1.0 is far
    # beyond the narrow margin, so no re-measurement is spent and the plain rejection stands.
    seed = HarnessDoc.baseline("seed")

    def judge(user: str) -> bool:
        if "the holdout task" in user:
            return "done-verified" not in user
        return "done-verified" in user

    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT), judge_fn=judge)
    holdout = [TaskSpec(task_id="h1", instruction="the holdout task", gold=["the base flow works"])]
    result = _run(provider, holdout=holdout)
    assert result.confirmations == 0
    [delta] = result.archive.deltas
    assert delta.verdict is not None and not delta.verdict.accepted
    assert "held-out regressed" in delta.verdict.reason


def test_confirmed_suite_overturn_still_faces_the_holdout_tier() -> None:
    # A suite veto narrow enough to overturn must NOT smuggle the child past held-out
    # verification: here the suite flake clears on re-measurement but the child genuinely
    # regresses held-out, so the final verdict is a holdout rejection.
    seed = HarnessDoc.baseline("seed")
    suite_flake = {"n": 0}

    def judge(user: str) -> bool:
        if "the holdout task" in user:
            return "done-verified" not in user  # child ALWAYS fails held-out (wide, real)
        if "suite task" in user:
            if "done-verified" in user:
                suite_flake["n"] += 1
                return suite_flake["n"] != 1  # one flaky failure for the child
            return True  # seed masters the suite task
        return "done-verified" in user  # the trigger task needs the careful child

    tasks = [
        TaskSpec(task_id="t1", instruction="answer it", gold=["the work was verified"]),
        TaskSpec(task_id="s1", instruction="the suite task", gold=["steady state holds"]),
    ]
    holdout = [TaskSpec(task_id="h1", instruction="the holdout task", gold=["the base flow works"])]
    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT), judge_fn=judge)
    result = create_harness(
        "winner",
        seed,
        tasks,
        _wm(provider),
        provider,
        ProviderDeltaProposer(provider),
        GoldJudge(provider),
        iterations=1,
        k=5,
        holdout=holdout,
    )
    [delta] = result.archive.deltas
    assert delta.verdict is not None and not delta.verdict.accepted
    # The holdout tier was measured (not bypassed) and its regression is the rejection.
    assert delta.verdict.holdout_delta is not None and delta.verdict.holdout_delta < 0
    assert result.best_score == pytest.approx(0.5)  # champion stayed the seed


# -- harness backends: local (in-process) vs e2b (the pi process in pooled sandboxes) ------------


def _pi_seed() -> HarnessDoc:
    from wmh.harness.doc import RUNTIME_KIND_ID, TOOL_POLICY_ID, Surface, SurfaceKind

    return HarnessDoc(
        name="seed",
        surfaces=[
            Surface(id="prompt:core", kind=SurfaceKind.PROMPT, content="p"),
            Surface(id=TOOL_POLICY_ID, kind=SurfaceKind.TOOL_POLICY, content="bash\nsubmit"),
            Surface(id=RUNTIME_KIND_ID, kind=SurfaceKind.PARAM, content="pi-node"),
            Surface(
                id="code:src-agent-ts", kind=SurfaceKind.CODE, path="src/agent.ts", content="// a"
            ),
        ],
    )


def _canned_report(rate: float, *, k: int = 3) -> ClosedLoopReport:
    outcome = TaskOutcome(task_id="t1", success_rate=rate, mean_fraction=rate, passes=k)
    return ClosedLoopReport(
        label="x", success_rate=rate, mean_fraction=rate, k=k, per_task={"t1": outcome}
    )


class _ScriptedPoolChannel:
    """Plays the runner peer for one pooled episode: a tool_request, then done.

    The same frame script `runner_link_test._FakeChannel` speaks; recv() hands frames to the
    real `RunnerLink`, send() records what the host answered — the tool_response content is how
    a test observes WHO answered the tool call.
    """

    def __init__(self) -> None:
        self.sent: list[JsonObject] = []
        self._script: list[JsonObject] = [
            {
                "type": "tool_request",
                "req_id": 1,
                "name": "bash",
                "arguments": {"command": "verify the work"},
            },
            {"type": "done", "answer": "done-verified"},
        ]

    def send(self, frame: JsonObject) -> None:
        self.sent.append(frame)

    def recv(self, timeout: float | None = None) -> JsonObject | None:
        del timeout
        return self._script.pop(0) if self._script else None


class _FakePool:
    """Stands in for `E2BSandboxPool`: no sandboxes, one scripted runner channel per acquire."""

    instances: ClassVar[list[_FakePool]] = []

    def __init__(
        self,
        *,
        template: str | None = None,
        api_key: str | None = None,
        metadata: dict[str, str] | None = None,
        sandbox_factory: object = None,
        hello_timeout: float = 0.0,
        episode_timeout_s: float = 300.0,
    ) -> None:
        self.template = template
        self.metadata = metadata
        self.episode_timeout_s = episode_timeout_s
        self.channels: list[_ScriptedPoolChannel] = []
        self.releases: list[bool] = []
        self.retire_idle_calls = 0
        self.closes = 0
        self.close_failures = 0
        _FakePool.instances.append(self)

    def assert_episode_timeout(self, episode_timeout_s: float) -> None:
        del episode_timeout_s  # the fake pool accepts any budget

    def usage(self) -> SandboxUsage:
        return SandboxUsage(count=len(self.channels), seconds=1.5 * len(self.channels))

    def acquire(self) -> tuple[object, _ScriptedPoolChannel]:
        channel = _ScriptedPoolChannel()
        self.channels.append(channel)
        return object(), channel

    def release(self, sandbox: object, channel: object, *, healthy: bool) -> None:
        self.releases.append(healthy)

    def retire_idle(self) -> int:
        self.retire_idle_calls += 1
        return 0

    def close(self) -> None:
        self.closes += 1
        if self.closes <= self.close_failures:
            from wmh.harness.e2b_sandbox import SandboxCleanupError

            raise SandboxCleanupError("evaluator cleanup unproven")


@pytest.fixture
def fake_pool_cls(monkeypatch: pytest.MonkeyPatch) -> type[_FakePool]:
    """Patch the pool at its source module (create_harness imports it lazily from there)."""
    _FakePool.instances = []
    monkeypatch.setattr("wmh.harness.pi_e2b.E2BSandboxPool", _FakePool)
    return _FakePool


def test_unknown_harness_backend_is_rejected() -> None:
    from typing import Literal, cast

    provider = RoleProvider()
    # Dynamic callers (the platform's optimizer passes a plain str) can hand in anything;
    # the runtime guard, not the type annotation, is what this test pins.
    bogus = cast("Literal['local', 'e2b']", "banana")
    with pytest.raises(ValueError, match="choose local or e2b"):
        create_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _tasks(),
            _wm(provider),
            provider,
            ProviderDeltaProposer(provider),
            GoldJudge(provider),
            harness_backend=bogus,
        )


def test_e2b_backend_rejects_non_pi_node_seeds() -> None:
    """e2b moves the pi-node harness PROCESS into sandboxes; in-process seeds must fail early."""
    provider = RoleProvider()
    with pytest.raises(ValueError, match="use harness_backend='local'"):
        create_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _tasks(),
            _wm(provider),
            provider,
            ProviderDeltaProposer(provider),
            GoldJudge(provider),
            harness_backend="e2b",
        )


def test_local_backend_rejects_parallel_pi_node_scoring() -> None:
    """Local pi runtimes are single-episode (one port/workdir/channel): local stays sequential.

    The guard fires per-doc at scoring time, before any rollout, so a parallel request fails
    loudly instead of colliding episodes.
    """
    provider = RoleProvider()
    with pytest.raises(ValueError, match="one episode at a time"):
        create_harness(
            "winner",
            _pi_seed(),
            _tasks(),
            _wm(provider),
            provider,
            ProviderDeltaProposer(provider),
            GoldJudge(provider),
            eval_concurrency=2,
        )


def test_e2b_backend_scores_against_the_world_model_through_the_shared_pool(
    monkeypatch: pytest.MonkeyPatch, fake_pool_cls: type[_FakePool]
) -> None:
    """harness_backend='e2b': the pi process lives in pooled sandboxes, the env stays the WM.

    The pool is faked (its channels play the runner peer), `evaluate_closed_loop` is the real
    one wrapped only to record the concurrency each eval was asked for — so every scripted
    tool_request is really brokered by `RunnerLink` into `WorldModelEnvironment`, and the
    tool_response carries the world model's marker reply ("ok" from the RoleProvider env role).
    """
    provider = RoleProvider()  # default judge passes on the runner's "done-verified" answer
    concurrencies: list[int] = []
    real_evaluate = create_module.evaluate_closed_loop

    def spying_evaluate(
        tasks: list[TaskSpec],
        environment: EnvironmentSource,
        agent_provider: Provider,
        judge: GoldJudge,
        *,
        label: str,
        k: int,
        concurrency: int,
        runtime: Runtime | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> ClosedLoopReport:
        concurrencies.append(concurrency)
        return real_evaluate(
            tasks,
            environment,
            agent_provider,
            judge,
            label=label,
            k=k,
            concurrency=concurrency,
            runtime=runtime,
            should_cancel=should_cancel,
        )

    monkeypatch.setattr(create_module, "evaluate_closed_loop", spying_evaluate)

    result = create_harness(
        "winner",
        _pi_seed(),
        _tasks(),
        _wm(provider),
        provider,
        ProviderDeltaProposer(provider),
        GoldJudge(provider),
        iterations=0,  # the seed eval alone exercises the whole scoring path
        k=3,
        harness_backend="e2b",
        e2b_template="tmpl-1",
        e2b_metadata={"optimizer_run_id": "run-1", "purpose": "evaluation"},
    )

    # The judge passed the runner's submitted answer: the eval genuinely ran end to end.
    assert result.best_score == 1.0
    assert concurrencies == [0]  # e2b default: every (task, attempt) cell at once
    [pool] = fake_pool_cls.instances  # ONE shared pool for the whole search
    assert pool.template == "tmpl-1"
    assert pool.metadata == {"optimizer_run_id": "run-1", "purpose": "evaluation"}
    # One finally owns teardown and mutates the returned model with the finalized meter.
    assert pool.closes == 1
    assert result.sandbox_usage is not None
    assert result.sandbox_usage.count == len(pool.channels)  # the fake meters per acquire
    assert len(pool.channels) == 3  # one pooled runner episode per (task, attempt) cell
    assert pool.releases == [True, True, True]  # healthy episodes return their sandboxes
    for channel in pool.channels:
        kinds = [f.get("type") for f in channel.sent]
        assert kinds == ["episode_start", "tool_response"]
        response = channel.sent[1]
        # The WORLD MODEL answered the tool: "ok" is the RoleProvider env-role marker reply.
        assert response.get("content") == "ok" and response.get("is_error") is False


def test_e2b_pool_is_closed_exactly_once_when_the_search_raises(
    monkeypatch: pytest.MonkeyPatch, fake_pool_cls: type[_FakePool]
) -> None:
    provider = RoleProvider()
    observed_usage: list[SandboxUsage] = []

    def exploding_evaluate(*args: object, **kwargs: object) -> ClosedLoopReport:
        raise RuntimeError("boom mid-eval")

    monkeypatch.setattr(create_module, "evaluate_closed_loop", exploding_evaluate)
    with pytest.raises(RuntimeError, match="boom mid-eval"):
        create_harness(
            "winner",
            _pi_seed(),
            _tasks(),
            _wm(provider),
            provider,
            ProviderDeltaProposer(provider),
            GoldJudge(provider),
            harness_backend="e2b",
            on_sandbox_usage=observed_usage.append,
        )
    [pool] = fake_pool_cls.instances
    assert pool.closes == 1  # the try/finally tears the pool down even on failure
    assert observed_usage == [SandboxUsage(count=0, seconds=0.0)]


def test_e2b_cleanup_failure_replaces_cancellation_and_withholds_final_usage(
    monkeypatch: pytest.MonkeyPatch, fake_pool_cls: type[_FakePool]
) -> None:
    """Cancellation cannot look clean when evaluator release remains unproven."""
    from wmh.harness.e2b_sandbox import SandboxCleanupError
    from wmh.harness.runtime import RuntimeCancelled

    provider = RoleProvider()
    observed_usage: list[SandboxUsage] = []

    def cancelled_evaluate(*args: object, **kwargs: object) -> ClosedLoopReport:
        del args, kwargs
        [pool] = fake_pool_cls.instances
        pool.close_failures = 1
        raise RuntimeCancelled("runtime episode cancelled")

    monkeypatch.setattr(create_module, "evaluate_closed_loop", cancelled_evaluate)

    with pytest.raises(SandboxCleanupError, match="cleanup unproven") as raised:
        create_harness(
            "winner",
            _pi_seed(),
            _tasks(),
            _wm(provider),
            provider,
            ProviderDeltaProposer(provider),
            GoldJudge(provider),
            harness_backend="e2b",
            on_sandbox_usage=observed_usage.append,
        )

    assert isinstance(raised.value.__context__, HarnessSearchCancelled)
    assert observed_usage == []


def test_runtime_cancellation_aborts_the_wave_without_judging_and_closes_pool(
    monkeypatch: pytest.MonkeyPatch, fake_pool_cls: type[_FakePool]
) -> None:
    from wmh.harness.pi_e2b import E2BPiRuntime
    from wmh.harness.runtime import RuntimeCancelled

    provider = RoleProvider()
    callback_seen = False

    def should_cancel() -> bool:
        return False

    def cancelled_evaluate(*args: object, **kwargs: object) -> ClosedLoopReport:
        nonlocal callback_seen
        runtime = kwargs.get("runtime")
        assert isinstance(runtime, E2BPiRuntime)
        callback_seen = runtime._should_cancel is should_cancel  # noqa: SLF001
        raise RuntimeCancelled("runtime episode cancelled")

    monkeypatch.setattr(create_module, "evaluate_closed_loop", cancelled_evaluate)

    with pytest.raises(HarnessSearchCancelled, match="cancelled") as raised:
        create_harness(
            "winner",
            _pi_seed(),
            _tasks(),
            _wm(provider),
            provider,
            ProviderDeltaProposer(provider),
            GoldJudge(provider),
            harness_backend="e2b",
            should_cancel=should_cancel,
        )

    assert callback_seen
    [pool] = fake_pool_cls.instances
    assert pool.closes == 1
    assert raised.value.sandbox_usage == SandboxUsage(count=0, seconds=0.0)


def test_cancellation_carries_completed_and_partial_wave_worker_usage(
    monkeypatch: pytest.MonkeyPatch, fake_pool_cls: type[_FakePool]
) -> None:
    """The public cancellation result owns all worker spend without a partial CreateResult."""
    from wmh.harness.runtime import RuntimeCancelled, TokenUsage

    seed = _pi_seed()
    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT))
    evaluate_calls = 0

    def cancel_second_wave(*args: object, **kwargs: object) -> ClosedLoopReport:
        nonlocal evaluate_calls
        del args
        evaluate_calls += 1
        if evaluate_calls == 1:
            k = kwargs.get("k", 3)
            assert isinstance(k, int)
            return _canned_report(0.5, k=k).model_copy(
                update={"worker_usage": TokenUsage(input_tokens=100, output_tokens=10, calls=2)}
            )
        raise RuntimeCancelled(
            "runtime episode cancelled",
            worker_usage=TokenUsage(input_tokens=7, output_tokens=2, calls=1),
        )

    monkeypatch.setattr(create_module, "evaluate_closed_loop", cancel_second_wave)

    with pytest.raises(HarnessSearchCancelled, match="cancelled") as raised:
        create_harness(
            "winner",
            seed,
            _tasks(),
            _wm(provider),
            provider,
            ProviderDeltaProposer(provider),
            GoldJudge(provider),
            iterations=1,
            harness_backend="e2b",
        )

    assert evaluate_calls == 2
    assert raised.value.worker_usage == TokenUsage(input_tokens=107, output_tokens=12, calls=3)
    assert raised.value.sandbox_usage == SandboxUsage(count=0, seconds=0.0)
    [pool] = fake_pool_cls.instances
    assert pool.closes == 1


def test_e2b_pool_retires_idle_runners_once_per_proposal_batch(
    monkeypatch: pytest.MonkeyPatch, fake_pool_cls: type[_FakePool]
) -> None:
    """Round boundaries rotate eval streams without rotating between sibling proposals."""
    provider = RoleProvider()
    monkeypatch.setattr(
        create_module,
        "evaluate_closed_loop",
        lambda *a, **k: _canned_report(0.5, k=k.get("k", 3)),
    )

    result = create_harness(
        "winner",
        _pi_seed(),
        _tasks(),
        _wm(provider),
        provider,
        ProviderDeltaProposer(provider),
        GoldJudge(provider),
        iterations=2,
        proposal_batch_size=3,
        harness_backend="e2b",
    )

    assert result.iterations == 2 and len(result.proposal_records) == 6
    [pool] = fake_pool_cls.instances
    assert pool.retire_idle_calls == 2  # once per batch, never between its three siblings


def test_eval_concurrency_overrides_both_backend_defaults(
    monkeypatch: pytest.MonkeyPatch, fake_pool_cls: type[_FakePool]
) -> None:
    """An explicit eval_concurrency reaches the scorer; unset local keeps the sequential default."""
    provider = RoleProvider()
    concurrencies: list[int] = []

    def fake_evaluate(
        tasks: list[TaskSpec],
        environment: EnvironmentSource,
        agent_provider: Provider,
        judge: GoldJudge,
        *,
        label: str,
        k: int,
        concurrency: int,
        runtime: Runtime | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> ClosedLoopReport:
        del should_cancel
        concurrencies.append(concurrency)
        return _canned_report(1.0, k=k)

    monkeypatch.setattr(create_module, "evaluate_closed_loop", fake_evaluate)

    def run(seed: HarnessDoc, *, harness_backend: str, eval_concurrency: int | None) -> None:
        create_harness(
            "winner",
            seed,
            _tasks(),
            _wm(provider),
            provider,
            ProviderDeltaProposer(provider),
            GoldJudge(provider),
            iterations=0,  # score the seed only: one eval call per run
            harness_backend="local" if harness_backend == "local" else "e2b",
            eval_concurrency=eval_concurrency,
        )

    run(HarnessDoc.baseline("seed"), harness_backend="local", eval_concurrency=None)
    run(HarnessDoc.baseline("seed"), harness_backend="local", eval_concurrency=4)
    run(_pi_seed(), harness_backend="e2b", eval_concurrency=2)
    assert concurrencies == [1, 4, 2]  # local defaults sequential; explicit caps pass through


def test_create_sums_worker_usage_across_score_waves(
    monkeypatch: pytest.MonkeyPatch, fake_pool_cls: type[_FakePool]
) -> None:
    """CreateResult.worker_usage is the sum of every score wave's report.worker_usage.

    Regression: the pi worker path self-meters tokens on each ClosedLoopReport, but the search
    dropped them on the floor (the accumulator list was declared and summed, never appended to),
    so CreateResult.worker_usage came back None and the platform's worker cost booked $0.00
    despite real agent LLM spend. Seed + one screened child = two waves here.
    """
    from wmh.harness.runtime import TokenUsage

    provider = RoleProvider(meta_reply=_meta_reply(HarnessDoc.baseline("seed"), _CAREFUL_PROMPT))

    def fake_evaluate(
        tasks: list[TaskSpec],
        environment: EnvironmentSource,
        agent_provider: Provider,
        judge: GoldJudge,
        *,
        label: str,
        k: int,
        concurrency: int,
        runtime: Runtime | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> ClosedLoopReport:
        del should_cancel
        report = _canned_report(0.5, k=k)
        return report.model_copy(
            update={"worker_usage": TokenUsage(input_tokens=100, output_tokens=10, calls=2)}
        )

    monkeypatch.setattr(create_module, "evaluate_closed_loop", fake_evaluate)

    result = create_harness(
        "winner",
        _pi_seed(),
        _tasks(),
        _wm(provider),
        provider,
        ProviderDeltaProposer(provider),
        GoldJudge(provider),
        iterations=1,
        harness_backend="e2b",
    )

    # Before the fix this was None (each wave's usage was never accumulated); now it sums
    # every wave. At least the seed wave ran (2 calls / 100in / 10out per wave), and the totals
    # hold that exact per-call ratio however many waves the search took.
    assert result.worker_usage is not None
    assert result.worker_usage.calls >= 2
    assert result.worker_usage.calls % 2 == 0
    assert result.worker_usage.input_tokens == 50 * result.worker_usage.calls
    assert result.worker_usage.output_tokens == 5 * result.worker_usage.calls


def test_create_worker_usage_is_none_when_no_wave_reports_it(
    monkeypatch: pytest.MonkeyPatch, fake_pool_cls: type[_FakePool]
) -> None:
    """Local runtimes don't self-meter: worker_usage stays None (never a zero-token TokenUsage)."""
    provider = RoleProvider()

    monkeypatch.setattr(
        create_module,
        "evaluate_closed_loop",
        lambda *a, **k: _canned_report(1.0, k=k.get("k", 3)),
    )

    result = create_harness(
        "winner",
        HarnessDoc.baseline("seed"),
        _tasks(),
        _wm(provider),
        provider,
        ProviderDeltaProposer(provider),
        GoldJudge(provider),
        iterations=0,
        harness_backend="local",
    )

    assert result.worker_usage is None


def test_e2b_rejects_a_delta_that_abandons_the_pi_runtime(
    monkeypatch: pytest.MonkeyPatch, fake_pool_cls: type[_FakePool]
) -> None:
    """A candidate that flips param:runtime-kind is archived invalid, not a run-aborting raise.

    Regression (Greptile P1): `doc.runtime(backend="e2b")` raises for non-pi-node docs; a meta
    proposal that rewrote the runtime-kind surface escaped the invalid-delta handling and
    aborted the whole search.
    """
    seed = _pi_seed()
    kind = seed.surface("param:runtime-kind")
    assert kind is not None
    escape = json.dumps(
        {
            "expected_effect": "run in-process instead",
            "preconditions": {"param:runtime-kind": kind.content_hash},
            "ops": [
                {
                    "op": "replace",
                    "surface_id": "param:runtime-kind",
                    "content": "kit-python",
                    "rationale": "abandon the pi runtime",
                }
            ],
        }
    )
    provider = RoleProvider(meta_reply=escape)
    monkeypatch.setattr(
        create_module,
        "evaluate_closed_loop",
        lambda *a, **k: _canned_report(0.5, k=k.get("k", 3)),
    )

    result = create_harness(
        "winner",
        seed,
        _tasks(),
        _wm(provider),
        provider,
        ProviderDeltaProposer(provider),
        GoldJudge(provider),
        iterations=1,
        harness_backend="e2b",
    )

    assert result.skipped == 1  # the escape delta was rejected, not fatal
    [delta] = result.archive.deltas
    assert delta.verdict is not None and not delta.verdict.accepted
    assert "pi-node only" in delta.verdict.reason
    assert result.best_score == 0.5  # the seed stayed champion and the search finished


class _MetaExplodingProvider(RoleProvider):
    """RoleProvider whose meta-agent calls raise (an API rejecting the request outright)."""

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        if "meta-agent improving an agent harness" in system:
            msg = "max_tokens above model output limit"
            raise RuntimeError(msg)
        return super().complete(system, messages, temperature=temperature, max_tokens=max_tokens)


def test_proposer_call_failure_skips_the_iteration_not_the_run() -> None:
    """A meta-provider exception (output-cap rejection, rate limit) costs one iteration.

    Same contract as an unusable reply, but narrated with the error; the search must not
    abort on the first provider fault.
    """
    provider = _MetaExplodingProvider()
    notes: list[str] = []
    result = _run(provider, iterations=2, on_note=notes.append)

    assert result.skipped == 2
    assert result.best.name == "winner"  # the seed still wins; the run completed
    assert len(notes) == 2
    assert all("proposer call failed" in note for note in notes)
    assert all("max_tokens above model output limit" in note for note in notes)
    assert [(r.iteration, r.outcome) for r in result.proposal_records] == [
        (1, "proposer_error"),
        (2, "proposer_error"),
    ]


class _SequencedMetaProvider(RoleProvider):
    """RoleProvider whose meta-agent replies follow a script, one per proposal call."""

    def __init__(self, replies: list[str]) -> None:
        super().__init__()
        self._replies = replies

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        if "meta-agent improving an agent harness" in system:
            self.meta_users.append(messages[-1].content)
            reply = self._replies[min(len(self.meta_users) - 1, len(self._replies) - 1)]
            return Completion(text=reply)
        return super().complete(system, messages, temperature=temperature, max_tokens=max_tokens)


class _RankedMetaProvider(_SequencedMetaProvider):
    """Give weak and strong proposal prompts distinct, deterministic task scores."""

    def __init__(self, replies: list[str]) -> None:
        super().__init__(replies)
        self._judge_fn = lambda user: (
            "done-strong" in user or ("done-weak" in user and "task one" in user)
        )

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        special = (
            "grade whether an agent completed a task",
            "meta-agent improving an agent harness",
            "You ARE the environment",
        )
        if not any(marker in system for marker in special):
            if "strong agent" in system:
                answer = "done-strong"
            elif "weak agent" in system:
                answer = "done-weak"
            else:
                answer = "done"
            return Completion(text=json.dumps({"tool": "submit", "arguments": {"answer": answer}}))
        return super().complete(
            system,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class _ParentRecordingProposer:
    """Propose against the live parent and remember each iteration's parent hash."""

    def __init__(self) -> None:
        self.parent_hashes: list[str] = []

    def propose_batch(
        self,
        parent: HarnessDoc,
        trigger: FailureSignature,
        evidence: str,
        *,
        history: list[HarnessDelta],
        count: int,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[HarnessDelta | ProposalFailure | None]:
        del evidence, history, should_cancel
        self.parent_hashes.append(parent.doc_hash)
        prompt = f"{_CAREFUL_PROMPT} Iteration {len(self.parent_hashes)}."
        proposal = parse_delta(parent, trigger, _meta_reply(parent, prompt))
        assert proposal is not None and count == 1
        return [proposal]


class _FeedbackRecordingProposer:
    """Return scripted siblings and retain the final evaluation feedback for each."""

    def __init__(self, prompts: list[str]) -> None:
        self.prompts = prompts
        self.feedback: list[tuple[str, str, str]] = []

    def propose_batch(
        self,
        parent: HarnessDoc,
        trigger: FailureSignature,
        evidence: str,
        *,
        history: list[HarnessDelta],
        count: int,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[HarnessDelta | ProposalFailure | None]:
        del evidence, history, should_cancel
        assert count == len(self.prompts)
        proposals: list[HarnessDelta | ProposalFailure | None] = [
            parse_delta(parent, trigger, _meta_reply(parent, prompt)) for prompt in self.prompts
        ]
        assert all(proposal is not None for proposal in proposals)
        return proposals

    def record_evaluation(self, delta: HarnessDelta, *, stage: str, content: str) -> None:
        self.feedback.append((delta.delta_id, stage, content))


def test_proposal_batch_is_generated_before_siblings_are_evaluated() -> None:
    """One iteration expands one parent into independently tracked sibling candidates."""
    seed = HarnessDoc.baseline("seed")
    provider = _SequencedMetaProvider(
        [
            _meta_reply(seed, _CAREFUL_PROMPT),
            _meta_reply(seed, f"{_CAREFUL_PROMPT} Double-check the result."),
        ]
    )

    result = _run(provider, iterations=1, proposal_batch_size=2)

    assert len(provider.meta_users) == 2
    assert [(record.iteration, record.proposal_index) for record in result.proposal_records] == [
        (1, 1),
        (1, 2),
    ]
    assert [record.candidate for record in result.proposal_records] == [
        "winner-i1-p1",
        "winner-i1-p2",
    ]
    assert len(result.archive.deltas) == 2


def test_iteration_batch_commits_one_winner_and_one_progress_point() -> None:
    """Three eligible siblings yield one deterministic winner and one champion update."""
    seed = HarnessDoc.baseline("seed")
    provider = _SequencedMetaProvider(
        [_meta_reply(seed, f"{_CAREFUL_PROMPT} Candidate {index}.") for index in range(1, 4)]
    )
    progress: list[tuple[int, str, float, bool]] = []
    proposals: list[ProposalRecord] = []
    crowned: list[str] = []

    result = create_harness(
        "winner",
        seed,
        _tasks(),
        _wm(provider),
        provider,
        ProviderDeltaProposer(provider),
        GoldJudge(provider),
        iterations=1,
        proposal_batch_size=3,
        on_progress=lambda i, n, r, a: progress.append((i, n, r, a)),
        on_proposal=proposals.append,
        on_accept=lambda doc, delta, score: crowned.append(doc.name),
    )

    assert result.iterations == 1
    assert len(result.proposal_records) == 3
    assert [(record.iteration, record.proposal_index) for record in result.proposal_records] == [
        (1, 1),
        (1, 2),
        (1, 3),
    ]
    assert [record.gate_eligible for record in result.proposal_records] == [True, True, True]
    assert [record.selected for record in result.proposal_records] == [True, False, False]
    assert len(proposals) == 3
    assert crowned == ["winner-i1-p1"]
    assert progress == [
        (0, "seed", 0.0, True),
        (1, "winner-i1-p1", 1.0, True),
    ]
    assert [delta.verdict.accepted for delta in result.archive.deltas if delta.verdict] == [
        True,
        False,
        False,
    ]


def test_duplicate_sibling_is_archived_without_duplicate_evaluation() -> None:
    """The search boundary rejects duplicate sibling deltas before spending another screen."""
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT))

    result = _run(provider, iterations=1, proposal_batch_size=2)

    assert result.skipped == 1
    assert len(result.archive.deltas) == 2
    assert [record.outcome for record in result.proposal_records] == ["scored", "invalid"]
    assert [record.selected for record in result.proposal_records] == [True, False]
    assert "already-proposed" in (result.proposal_records[1].reason or "")
    assert len(result.reports) == 2  # seed plus the first sibling, never the duplicate


def test_cancellation_during_later_sibling_commits_no_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later-sibling cancellation cannot publish an earlier sibling as the winner."""
    seed = HarnessDoc.baseline("seed")
    provider = _SequencedMetaProvider(
        [
            _meta_reply(seed, _CAREFUL_PROMPT),
            _meta_reply(seed, f"{_CAREFUL_PROMPT} Another candidate."),
        ]
    )
    cancelled = False
    gate_delta = create_module.gate_delta

    def cancel_after_first_gate(
        delta: HarnessDelta,
        *,
        child: ClosedLoopReport,
        champion: ClosedLoopReport,
        best_full: float,
        suite: list[str],
        child_holdout: ClosedLoopReport | None = None,
        champion_holdout: ClosedLoopReport | None = None,
    ) -> GateRecord:
        nonlocal cancelled
        verdict = gate_delta(
            delta,
            child=child,
            champion=champion,
            best_full=best_full,
            suite=suite,
            child_holdout=child_holdout,
            champion_holdout=champion_holdout,
        )
        cancelled = True
        return verdict

    monkeypatch.setattr(create_module, "gate_delta", cancel_after_first_gate)
    progress: list[tuple[int, str, float, bool]] = []
    crowned: list[str] = []
    proposals: list[ProposalRecord] = []

    with pytest.raises(HarnessSearchCancelled, match="cancelled"):
        _run(
            provider,
            proposal_batch_size=2,
            should_cancel=lambda: cancelled,
            on_progress=lambda i, n, r, a: progress.append((i, n, r, a)),
            on_accept=lambda doc, delta, score: crowned.append(doc.doc_hash),
            on_proposal=proposals.append,
        )

    assert progress == [(0, "seed", 0.0, True)]
    assert crowned == []
    assert proposals == []


@pytest.mark.parametrize(
    ("prompts", "selected"),
    [
        (["You are a weak agent.", "You are a strong agent."], [False, True]),
        (["You are a strong agent.", "You are a weak agent."], [True, False]),
    ],
)
def test_iteration_selects_best_eligible_score_independent_of_proposal_order(
    prompts: list[str], selected: list[bool]
) -> None:
    """Full success outranks assertion fraction and proposal order within a frozen batch."""
    seed = HarnessDoc.baseline("seed")
    provider = _RankedMetaProvider([_meta_reply(seed, prompt) for prompt in prompts])
    tasks = [
        TaskSpec(task_id="t1", instruction="task one", gold=["the work completed"]),
        TaskSpec(task_id="t2", instruction="task two", gold=["the work completed"]),
    ]
    crowned: list[str] = []

    result = create_harness(
        "winner",
        seed,
        tasks,
        _wm(provider),
        provider,
        ProviderDeltaProposer(provider),
        GoldJudge(provider),
        iterations=1,
        proposal_batch_size=2,
        k=1,
        on_accept=lambda doc, delta, score: crowned.append(doc.system_prompt()),
    )

    assert result.best_score == 1.0
    assert result.best.system_prompt() == "You are a strong agent."
    assert [record.gate_eligible for record in result.proposal_records] == [True, True]
    assert [record.selected for record in result.proposal_records] == selected
    assert crowned == ["You are a strong agent."]
    accepted = result.archive.accepted()
    assert len(accepted) == 1
    winner_hash = next(
        record.candidate_doc_hash for record in result.proposal_records if record.selected
    )
    assert accepted[0].child_doc_hash == winner_hash
    assert winner_hash is not None
    assert result.archive.reconstruct(winner_hash).system_prompt() == "You are a strong agent."
    loser_hash = next(
        record.candidate_doc_hash for record in result.proposal_records if not record.selected
    )
    assert loser_hash is not None
    with pytest.raises(ValueError, match="not in this archive"):
        result.archive.reconstruct(loser_hash)
    loser_delta = next(
        delta
        for delta in result.archive.deltas
        if delta.verdict is not None and not delta.verdict.accepted
    )
    assert loser_delta.verdict is not None
    assert "gate eligible but not selected" in loser_delta.verdict.reason


def test_full_feedback_records_final_batch_selection() -> None:
    """Eligible losers teach the proposer that ranking, not the gate, rejected them."""
    provider = _RankedMetaProvider([])
    proposer = _FeedbackRecordingProposer(["You are a weak agent.", "You are a strong agent."])
    tasks = [
        TaskSpec(task_id="t1", instruction="task one", gold=["the work completed"]),
        TaskSpec(task_id="t2", instruction="task two", gold=["the work completed"]),
    ]

    result = create_harness(
        "winner",
        HarnessDoc.baseline("seed"),
        tasks,
        _wm(provider),
        provider,
        proposer,
        GoldJudge(provider),
        iterations=1,
        proposal_batch_size=2,
        k=1,
    )

    loser = next(
        delta
        for delta in result.archive.deltas
        if delta.verdict is not None and not delta.verdict.accepted
    )
    winner = result.archive.accepted()[0]
    loser_feedback = next(
        content
        for delta_id, stage, content in proposer.feedback
        if delta_id == loser.delta_id and stage == "full"
    )
    winner_feedback = next(
        content
        for delta_id, stage, content in proposer.feedback
        if delta_id == winner.delta_id and stage == "full"
    )
    assert "gate eligible but not selected" in loser_feedback
    assert "gate eligible but not selected" not in winner_feedback


def test_sibling_holdout_gates_use_frozen_iteration_champion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A selected-looking first sibling cannot become the second sibling's gate baseline."""
    seed = HarnessDoc.baseline("seed")
    provider = _SequencedMetaProvider(
        [
            _meta_reply(seed, _CAREFUL_PROMPT),
            _meta_reply(seed, f"{_CAREFUL_PROMPT} Another candidate."),
        ]
    )
    provider._judge_fn = lambda user: (
        True if "the holdout task" in user else "done-verified" in user
    )
    holdout = [TaskSpec(task_id="h1", instruction="the holdout task", gold=["still works"])]
    gate_delta = create_module.gate_delta
    holdout_gate_champions: list[tuple[float, float]] = []

    def capture_gate_baselines(
        delta: HarnessDelta,
        *,
        child: ClosedLoopReport,
        champion: ClosedLoopReport,
        best_full: float,
        suite: list[str],
        child_holdout: ClosedLoopReport | None = None,
        champion_holdout: ClosedLoopReport | None = None,
    ) -> GateRecord:
        if child_holdout is not None and champion_holdout is not None:
            holdout_gate_champions.append((champion.success_rate, champion_holdout.success_rate))
        return gate_delta(
            delta,
            child=child,
            champion=champion,
            best_full=best_full,
            suite=suite,
            child_holdout=child_holdout,
            champion_holdout=champion_holdout,
        )

    monkeypatch.setattr(create_module, "gate_delta", capture_gate_baselines)

    result = _run(provider, proposal_batch_size=2, holdout=holdout)

    assert holdout_gate_champions == [(0.0, 1.0), (0.0, 1.0)]
    assert [record.gate_eligible for record in result.proposal_records] == [True, True]
    assert [record.selected for record in result.proposal_records] == [True, False]


def test_next_iteration_proposes_from_previous_iteration_winner() -> None:
    """The selected winner is the next parent, with no stepping-stone parent pool."""
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider()
    proposer = _ParentRecordingProposer()

    result = create_harness(
        "winner",
        seed,
        _tasks(),
        _wm(provider),
        provider,
        proposer,
        GoldJudge(provider),
        iterations=2,
        proposal_batch_size=1,
    )

    assert len(result.archive.accepted()) == 2
    first_winner_hash = result.proposal_records[0].candidate_doc_hash
    assert proposer.parent_hashes == [seed.doc_hash, first_winner_hash]
    assert [record.selected for record in result.proposal_records] == [True, True]


def test_on_accept_delivers_the_new_champion_the_moment_it_is_crowned() -> None:
    """Accepted champions stream out live, so callers can persist them in real time."""
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT))
    crowned: list[tuple[str, bool, float]] = []
    result = _run(
        provider,
        on_accept=lambda doc, delta, score: crowned.append(
            (doc.system_prompt(), delta.verdict is not None and delta.verdict.accepted, score)
        ),
    )

    assert result.best_score == 1.0
    [(prompt, verdict_accepted, score)] = crowned
    assert prompt == _CAREFUL_PROMPT  # the actual champion doc, not a name or hash
    assert verdict_accepted is True  # the delta arrives with its verdict already attached
    assert score == 1.0


def test_dead_iteration_ends_early_and_the_search_moves_on() -> None:
    """A dead proposal costs its iteration cheaply; the next iteration proceeds normally.

    Iteration 1's proposal is unusable; iteration 2 proposes the genuine fix. Both appear
    in the records, and the scored one keeps its own iteration number.
    """
    seed = HarnessDoc.baseline("seed")
    provider = _SequencedMetaProvider(["garbage, not json", _meta_reply(seed, _CAREFUL_PROMPT)])
    progress: list[tuple[int, str, float, bool]] = []
    result = _run(
        provider, iterations=2, on_progress=lambda i, n, r, a: progress.append((i, n, r, a))
    )

    assert result.skipped == 1
    assert result.best_score == 1.0
    assert result.best.system_prompt() == _CAREFUL_PROMPT
    assert [e[0] for e in progress] == [0, 1, 2]
    assert progress[1] == (1, "seed", 0.0, False)
    assert [(r.iteration, r.outcome) for r in result.proposal_records] == [
        (1, "unusable"),
        (2, "scored"),
    ]
    scored = result.proposal_records[-1]
    assert scored.selected is True and scored.score == 1.0
    assert scored.candidate == "winner-i2-p1"


def test_skipped_proposals_narrate_through_on_note() -> None:
    """Every proposal that dies before scoring narrates itself.

    Regression: a run whose proposals were all unusable (e.g. truncated meta replies on huge
    pi code surfaces) emitted NO progress events at all; five iterations looked like one.
    """
    provider = RoleProvider(meta_reply="truncated garbage that is not json")
    notes: list[str] = []
    result = _run(provider, iterations=3, on_note=notes.append)

    assert result.skipped == 3
    assert [note.split(":")[0] for note in notes] == [
        "iteration 1/3",
        "iteration 2/3",
        "iteration 3/3",
    ]
    assert all("proposal unusable" in note for note in notes)


def test_dead_notes_precede_final_proposal_records_and_iteration_checkpoint() -> None:
    """Eager diagnostics precede the batch's ordered records and champion checkpoint."""
    events: list[str] = []

    result = _run(
        RoleProvider(meta_reply="truncated garbage that is not json"),
        proposal_batch_size=2,
        on_progress=lambda iteration, name, score, changed: events.append(f"progress:{iteration}"),
        on_note=lambda message: events.append(f"note:{message.split(':', 1)[0]}"),
        on_proposal=lambda record: events.append(f"proposal:{record.proposal_index}"),
    )

    assert result.skipped == 2
    assert events == [
        "progress:0",
        "note:iteration 1/1 proposal 1/2",
        "note:iteration 1/1 proposal 2/2",
        "proposal:1",
        "proposal:2",
        "progress:1",
    ]
