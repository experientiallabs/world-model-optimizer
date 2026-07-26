"""Tests for the rollout collector against a stubbed harbor scorer."""

from __future__ import annotations

import ast
import json
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from harbor.models.job.config import JobConfig

import wmh.distill.rollouts as rollouts_module
from wmh.distill.config import (
    DistillConfig,
    HarborConfig,
    StudentConfig,
    TeacherConfig,
    TrainConfig,
)
from wmh.distill.rollouts import (
    HARBOR_TERMINUS_2_AGENT_IMPORT_PATH,
    collect_rollouts,
    rollout_stats,
    terminus_2_agent_kwargs,
)
from wmh.distill.tokens import TrialRecord
from wmh.evals.harbor.scorer import HarborScorer
from wmh.harness.doc import HarnessDoc
from wmh.harness.scoring import GradedTests, ScoreCell, ScoreReport, ScoreRequest
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.providers.tinker import TokenSpan

_TASK_IDS = ("task-a", "task-b")
_GROUP_SIZE = 2


def _provider_config() -> ProviderConfig:
    return ProviderConfig(
        kind=ProviderKind.TINKER,
        model_type="Qwen/Qwen3-8B",
        model="tinker://run/weights/4",
    )


def _write_template(tmp_path: Path) -> Path:
    template_path = tmp_path / "job-template.yaml"
    template_path.write_text(
        "job_name: template\n"
        "jobs_dir: /tmp/overridden-by-the-collector\n"
        "n_concurrent_trials: 1\n"
        "datasets:\n"
        f"- path: {tmp_path / 'tasks'}\n"
        "agents:\n"
        "- {}\n",
        encoding="utf-8",
    )
    return template_path


def _cfg(
    tmp_path: Path,
    *,
    backend: str = "local",
    trial_concurrency: int = 3,
) -> DistillConfig:
    return DistillConfig(
        student=StudentConfig(base_model="Qwen/Qwen3-4B"),
        teacher=TeacherConfig(model="Qwen/Qwen3-8B"),
        harbor=HarborConfig(
            job_template=str(_write_template(tmp_path)),
            backend="e2b" if backend == "e2b" else "local",
            reward_key="reward",
        ),
        train=TrainConfig(group_size=_GROUP_SIZE, trial_concurrency=trial_concurrency),
    )


def _span(call_index: int) -> TokenSpan:
    return TokenSpan(
        call_index=call_index,
        prompt_token_ids=[1, 2, call_index],
        sampled_token_ids=[70, 71],
        sampled_logprobs=[-0.5, -1.0],
    )


def _report(harness: HarnessDoc, trials_dir: Path) -> ScoreReport:
    """The canned (task x attempt) matrix: task-a passes both, task-b fails both."""
    cells = []
    for task_id in _TASK_IDS:
        for attempt in (1, 2):
            reward = 1.0 if task_id == "task-a" else 0.0
            cells.append(
                ScoreCell(
                    task_id=task_id,
                    attempt=attempt,
                    reward=reward,
                    passed=reward == 1.0,
                    artifact_dir=str(trials_dir / f"{task_id}__s{attempt}"),
                )
            )
    return ScoreReport(
        doc_hash=harness.doc_hash,
        request=ScoreRequest(task_ids=_TASK_IDS, attempts=_GROUP_SIZE),
        reward_mode="raw",
        cells=tuple(cells),
    )


class _StubScorer:
    """Stands in for a created HarborScorer; returns the canned report."""

    def __init__(self, report: ScoreReport) -> None:
        self.report = report
        self.score_calls: list[tuple[str, Callable[[], bool] | None]] = []
        self.jobs_dir: Path | None = None

    def candidate_job_dir(self, doc: HarnessDoc) -> Path:
        """Mirror HarborScorer's deterministic per-candidate job dir."""
        assert self.jobs_dir is not None
        return self.jobs_dir / f"wmh-{doc.doc_hash[:12]}"

    def score(
        self,
        doc: HarnessDoc,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> ScoreReport:
        self.score_calls.append((doc.doc_hash, should_cancel))
        return self.report


class _CreateCapture:
    """Monkeypatch target for HarborScorer.create; records every construction."""

    def __init__(self, stub: _StubScorer) -> None:
        self.stub = stub
        self.templates: list[JobConfig] = []
        self.task_ids: list[list[str]] = []
        self.kwargs: list[dict[str, object]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        capture = self

        async def fake_create(
            _cls: type[HarborScorer],
            job_template: JobConfig,
            task_ids: Sequence[str],
            **kwargs: object,
        ) -> _StubScorer:
            capture.templates.append(job_template)
            capture.task_ids.append(list(task_ids))
            capture.kwargs.append(dict(kwargs))
            capture.stub.jobs_dir = job_template.jobs_dir
            return capture.stub

        monkeypatch.setattr(HarborScorer, "create", classmethod(fake_create))


def _write_terminus_trial(
    trial_dir: Path,
    *,
    turns: int,
    n_episodes: int | None = None,
    exception_type: str | None = None,
    complete: bool = False,
    completion_length: int = 2,
) -> None:
    """Write the harbor artifacts a finished terminus-2 trial leaves behind."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    sampled = [70 + index for index in range(completion_length)]
    logprobs = [-0.5 * (index + 1) for index in range(completion_length)]
    detail: dict[str, object] = {
        "prompt_token_ids": [[1, 2, index] for index in range(turns)],
        "completion_token_ids": [list(sampled) for _ in range(turns)],
        "logprobs": [list(logprobs) for _ in range(turns)],
    }
    result: dict[str, object] = {
        "agent_result": {
            "rollout_details": [detail],
            "metadata": {"n_episodes": turns if n_episodes is None else n_episodes},
        }
    }
    if exception_type is not None:
        result["exception_info"] = {"exception_type": exception_type}
    (trial_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    if complete:
        agent_dir = trial_dir / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        (agent_dir / "trajectory.json").write_text(
            json.dumps(
                {
                    "steps": [
                        {"step_id": 1, "source": "user"},
                        {
                            "step_id": 2,
                            "source": "agent",
                            "tool_calls": [
                                {
                                    "tool_call_id": "call_1_task_complete",
                                    "function_name": "mark_task_complete",
                                    "arguments": {},
                                }
                            ],
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )


def test_collect_rollouts_wires_terminus_2_and_joins_harbor_rollout_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    trials_dir = tmp_path / "trials"
    harness = HarnessDoc.baseline()
    stub = _StubScorer(_report(harness, trials_dir))
    capture = _CreateCapture(stub)
    capture.install(monkeypatch)
    cfg = _cfg(tmp_path)

    # Harbor's own per-trial evidence for 3 of the 4 trials; the fourth never wrote one.
    for trial_name in ("task-a__s2", "task-b__s1"):
        _write_terminus_trial(trials_dir / trial_name, turns=2)
    _write_terminus_trial(trials_dir / "task-a__s1", turns=2, complete=True)

    def cancel_poll() -> bool:
        return False

    records, stats = collect_rollouts(
        4,
        _TASK_IDS,
        cfg,
        harness,
        _provider_config(),
        run_dir,
        should_cancel=cancel_poll,
    )

    # The scorer was created on a FRESH per-step jobs dir with harbor's terminus-2.
    [template] = capture.templates
    assert template.jobs_dir == run_dir / "harbor" / "step-0004"
    assert capture.task_ids == [list(_TASK_IDS)]
    [kwargs] = capture.kwargs
    assert kwargs["agent_import_path"] == HARBOR_TERMINUS_2_AGENT_IMPORT_PATH
    # The base model names the renderer/tokenizer; the sampler path is the checkpoint.
    assert kwargs["agent_model_name"] == "Qwen/Qwen3-8B"
    assert kwargs["extra_agent_kwargs"] == {
        "llm_backend": "tinker",
        "llm_kwargs": {
            "max_tokens": cfg.sampling.max_tokens,
            "context_limit": cfg.rollout.context_budget_tokens,
            "output_limit": cfg.sampling.max_tokens,
            "model_path": "tinker://run/weights/4",
        },
        "collect_rollout_details": True,
        "temperature": cfg.sampling.temperature,
        "max_turns": cfg.rollout.max_turns,
        "enable_summarize": False,
        "suppress_max_turns_warning": True,
    }
    assert kwargs["attempts"] == _GROUP_SIZE
    assert kwargs["reward_key"] == "reward"
    assert kwargs["harness_backend"] == "local"
    assert kwargs["task_environment"] == "docker"
    assert kwargs["agent_concurrency"] == 1
    assert kwargs["provider_config"] == _provider_config()
    # Terminus-2 has no wall clock of its own, so the budget rides on harbor's agent timeout.
    [agent] = template.agents
    assert agent.override_timeout_sec == pytest.approx(cfg.rollout.episode_timeout_s)
    assert agent.max_timeout_sec is None

    # Cancellation flows through to the blocking score call.
    assert stub.score_calls == [(harness.doc_hash, cancel_poll)]

    # Spans come from each trial dir's result.json; the evidence-less trial is kept, not dropped.
    assert [record.trial_name for record in records] == [
        "task-a__s1",
        "task-a__s2",
        "task-b__s1",
        "task-b__s2",
    ]
    assert all(len(record.spans) == 2 for record in records[:3])
    assert [span.sampled_token_ids for span in records[0].spans] == [[70, 71], [70, 71]]
    assert [span.call_index for span in records[0].spans] == [0, 1]
    assert records[0].stop_reason == "submitted"
    assert records[1].stop_reason == "error"  # ran, never claimed completion
    assert records[3].spans == []
    assert records[3].stop_reason is None
    assert stats.trials == 4
    assert stats.trials_with_spans == 3
    assert stats.empty_span_trials == 1
    assert stats.solve_rate == 0.5


def test_terminus_2_agent_kwargs_omit_model_path_for_a_base_model_provider(
    tmp_path: Path,
) -> None:
    """The teacher samples a base model directly: TinkerLLM must get no checkpoint path."""
    cfg = _cfg(tmp_path)
    base = ProviderConfig(
        kind=ProviderKind.TINKER, model="Qwen/Qwen3-8B", model_type="Qwen/Qwen3-8B"
    )
    llm_kwargs = terminus_2_agent_kwargs(cfg, base)["llm_kwargs"]
    assert isinstance(llm_kwargs, dict)
    assert "model_path" not in llm_kwargs


def test_terminus_2_agent_kwargs_pick_the_renderer_per_base_model(tmp_path: Path) -> None:
    """The escape hatch for a reasoning renderer terminus-2 cannot use as shipped.

    Keyed per base model because the teacher's own rollouts (warmup, teacher baseline) sample a
    DIFFERENT base model than the student, and a Nemotron Nano and a Nemotron Ultra need
    different renderer names.
    """
    cfg = _cfg(tmp_path)
    cfg = cfg.model_copy(
        update={
            "rollout": cfg.rollout.model_copy(
                update={"renderers": {"Qwen/Qwen3-8B": "wmh/qwen3_verbatim"}}
            )
        }
    )
    student = terminus_2_agent_kwargs(cfg, _provider_config())["llm_kwargs"]
    assert isinstance(student, dict)
    assert student["renderer_name"] == "wmh/qwen3_verbatim"

    other = ProviderConfig(
        kind=ProviderKind.TINKER, model="Qwen/Qwen3-32B", model_type="Qwen/Qwen3-32B"
    )
    unmapped = terminus_2_agent_kwargs(cfg, other)["llm_kwargs"]
    assert isinstance(unmapped, dict)
    assert "renderer_name" not in unmapped  # auto-discovered by TinkerLLM


def test_terminus_2_agent_kwargs_register_the_wmh_verbatim_renderers(tmp_path: Path) -> None:
    """Terminus-2 resolves `renderer_name` from the cookbook's global registry.

    This is the single call site every rollout path (training, warmup collection, eval waves)
    shares, so a config naming `wmh/...` only resolves because building the kwargs registered it.
    """
    from tinker_cookbook.renderers import get_registered_renderer_names, unregister_renderer

    from wmh.distill.renderers import VERBATIM_RENDERERS

    for name in VERBATIM_RENDERERS:
        unregister_renderer(name)
    assert not set(VERBATIM_RENDERERS) & set(get_registered_renderer_names())

    terminus_2_agent_kwargs(_cfg(tmp_path), _provider_config())

    assert set(VERBATIM_RENDERERS) <= set(get_registered_renderer_names())


def test_terminus_2_agent_kwargs_reject_a_non_tinker_provider(tmp_path: Path) -> None:
    """Only Tinker sampling records the token ids the loss trains on."""
    cfg = _cfg(tmp_path)
    hosted = ProviderConfig(kind=ProviderKind.OPENAI, model="gpt-4o", model_type="gpt-4o")
    with pytest.raises(ValueError, match="must sample through Tinker"):
        terminus_2_agent_kwargs(cfg, hosted)


def test_each_step_gets_a_fresh_jobs_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Job dirs are keyed by doc hash only; reusing one jobs dir across steps would
    resume step N's trials as step N+1's results with stale-weights tokens."""
    run_dir = tmp_path / "run"
    harness = HarnessDoc.baseline()
    stub = _StubScorer(_report(harness, tmp_path / "trials"))
    capture = _CreateCapture(stub)
    capture.install(monkeypatch)
    cfg = _cfg(tmp_path)

    collect_rollouts(4, _TASK_IDS, cfg, harness, _provider_config(), run_dir)
    collect_rollouts(5, _TASK_IDS, cfg, harness, _provider_config(), run_dir)

    assert [template.jobs_dir for template in capture.templates] == [
        run_dir / "harbor" / "step-0004",
        run_dir / "harbor" / "step-0005",
    ]
    # Spans live inside harbor's own trial dirs now, so no WMH sink dir is created.
    assert not (run_dir / "tokens").exists()


def _write_recorded_job_config(candidate_dir: Path, provider_config: ProviderConfig) -> None:
    """Persist the slice of a harbor config.json the stale-policy check reads."""
    candidate_dir.mkdir(parents=True, exist_ok=True)
    payload = {"agents": [{"kwargs": {"provider_config": provider_config.model_dump(mode="json")}}]}
    (candidate_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")


def test_stale_policy_job_dir_is_wiped_before_scoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A job dir left by a previous session under DIFFERENT sampler weights can
    never pass the scorer's strict resume check, and its trials sampled another
    policy: the collector wipes it whole (spans included, since they live in the
    trial dirs) so the step re-runs from the current weights instead of dying on
    resume."""
    run_dir = tmp_path / "run"
    harness = HarnessDoc.baseline()
    stub = _StubScorer(_report(harness, tmp_path / "trials"))
    capture = _CreateCapture(stub)
    capture.install(monkeypatch)
    cfg = _cfg(tmp_path)

    candidate = run_dir / "harbor" / "step-0004" / f"wmh-{harness.doc_hash[:12]}"
    old_provider = _provider_config().model_copy(update={"model": "tinker://run/weights/OLD"})
    _write_recorded_job_config(candidate, old_provider)
    _write_terminus_trial(candidate / "task-a__s1", turns=2)  # a stale completed trial

    collect_rollouts(4, _TASK_IDS, cfg, harness, _provider_config(), run_dir)

    assert not candidate.exists()  # the stale-policy job dir, and its spans, wiped whole
    assert len(stub.score_calls) == 1


def test_matching_policy_job_dir_is_kept_for_harbor_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same provider identity (e.g. the teacher's stable ref, which carries no
    session nonce): the dir is left alone so harbor's native trial-level
    resume re-runs only what is missing."""
    run_dir = tmp_path / "run"
    harness = HarnessDoc.baseline()
    stub = _StubScorer(_report(harness, tmp_path / "trials"))
    capture = _CreateCapture(stub)
    capture.install(monkeypatch)
    cfg = _cfg(tmp_path)

    candidate = run_dir / "harbor" / "step-0004" / f"wmh-{harness.doc_hash[:12]}"
    _write_recorded_job_config(candidate, _provider_config())
    completed_trial = candidate / "task-a__s1"
    _write_terminus_trial(completed_trial, turns=1)

    collect_rollouts(4, _TASK_IDS, cfg, harness, _provider_config(), run_dir)

    assert completed_trial.is_dir()
    assert (completed_trial / "result.json").exists()


def test_unreadable_job_config_is_left_for_the_scorer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable config.json is not evidence of another policy; the dir is
    left untouched so the scorer raises its own actionable error rather than
    the collector destroying evidence silently."""
    run_dir = tmp_path / "run"
    harness = HarnessDoc.baseline()
    stub = _StubScorer(_report(harness, tmp_path / "trials"))
    capture = _CreateCapture(stub)
    capture.install(monkeypatch)
    cfg = _cfg(tmp_path)

    candidate = run_dir / "harbor" / "step-0004" / f"wmh-{harness.doc_hash[:12]}"
    candidate.mkdir(parents=True)
    (candidate / "config.json").write_text("{not json", encoding="utf-8")

    collect_rollouts(4, _TASK_IDS, cfg, harness, _provider_config(), run_dir)

    assert (candidate / "config.json").exists()


def test_e2b_backend_routes_environment_and_concurrency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    harness = HarnessDoc.baseline()
    stub = _StubScorer(_report(harness, tmp_path / "trials"))
    capture = _CreateCapture(stub)
    capture.install(monkeypatch)
    cfg = _cfg(tmp_path, backend="e2b", trial_concurrency=6)

    collect_rollouts(0, _TASK_IDS, cfg, harness, _provider_config(), run_dir)

    [kwargs] = capture.kwargs
    assert kwargs["harness_backend"] == "e2b"
    assert kwargs["task_environment"] == "e2b"
    assert kwargs["agent_concurrency"] == 6


def test_negative_step_index_is_rejected(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="step_index must be >= 0"):
        collect_rollouts(
            -1, _TASK_IDS, cfg, HarnessDoc.baseline(), _provider_config(), tmp_path / "run"
        )


def test_template_load_failures_are_actionable(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    harness = HarnessDoc.baseline()

    missing = cfg.model_copy(deep=True)
    missing.harbor.job_template = str(tmp_path / "nowhere.yaml")
    with pytest.raises(ValueError, match="cannot load the harbor job template"):
        collect_rollouts(0, _TASK_IDS, missing, harness, _provider_config(), tmp_path / "run")

    not_mapping = tmp_path / "list.yaml"
    not_mapping.write_text("- 1\n- 2\n", encoding="utf-8")
    listy = cfg.model_copy(deep=True)
    listy.harbor.job_template = str(not_mapping)
    with pytest.raises(ValueError, match="must be a mapping"):
        collect_rollouts(0, _TASK_IDS, listy, harness, _provider_config(), tmp_path / "run")

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("agents: nope\n", encoding="utf-8")
    broken = cfg.model_copy(deep=True)
    broken.harbor.job_template = str(invalid)
    with pytest.raises(ValueError, match="invalid harbor job template"):
        collect_rollouts(0, _TASK_IDS, broken, harness, _provider_config(), tmp_path / "run")


def test_missing_harbor_extra_names_the_extra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "harbor.models.job.config", None)
    cfg = _cfg(tmp_path)
    with pytest.raises(ImportError, match="uv sync --extra harbor"):
        collect_rollouts(
            0, _TASK_IDS, cfg, HarnessDoc.baseline(), _provider_config(), tmp_path / "run"
        )


def _trial(
    task_id: str,
    *,
    passed: bool,
    spans: bool,
    stop_reason: str | None = "submitted",
    infra_failed: bool = False,
    tests: GradedTests | None = None,
) -> TrialRecord:
    return TrialRecord(
        task_id=task_id,
        attempt=1,
        trial_name=f"{task_id}__s1",
        reward=1.0 if passed else 0.0,
        passed=passed,
        spans=[_span(0)] if spans else [],
        stop_reason=stop_reason,
        infra_failed=infra_failed,
        tests=tests,
        artifact_dir=f"/trials/{task_id}",
    )


def test_rollout_stats_is_a_pure_function_of_the_records() -> None:
    """Stats recomputed from persisted records match a live batch's shape."""
    stats = rollout_stats(
        [
            _trial("task-a", passed=True, spans=True),
            _trial("task-b", passed=False, spans=True),
            _trial("task-c", passed=False, spans=False),
            _trial("task-d", passed=True, spans=True),
        ],
        max_tokens=4096,
    )
    assert stats.trials == 4
    assert stats.trials_with_spans == 3
    assert stats.solve_rate == 0.5
    assert stats.empty_span_trials == 1

    empty = rollout_stats([], max_tokens=4096)
    assert empty.trials == 0
    assert empty.solve_rate == 0.0
    assert empty.empty_span_trials == 0


def _sized_trial(task_id: str, *, lengths: Sequence[int], canonical: bool) -> TrialRecord:
    """A trial whose spans sample `lengths[i]` tokens each, at logprob -1.0.

    `canonical=False` leaves `delta_messages` None on the LAST span only, which
    is the realistic shape: a re-render fallback happens at one turn, not the
    whole episode, and `reconstruct_conversation` still refuses the trial whole.
    """
    spans = [
        TokenSpan(
            call_index=index,
            prompt_token_ids=[1, 2, index],
            sampled_token_ids=[70] * length,
            sampled_logprobs=[-1.0] * length,
            delta_start=(None if (not canonical and index == len(lengths) - 1) else 0),
            delta_messages=(None if (not canonical and index == len(lengths) - 1) else []),
        )
        for index, length in enumerate(lengths)
    ]
    return TrialRecord(
        task_id=task_id,
        attempt=1,
        trial_name=f"{task_id}__s1",
        reward=0.0,
        passed=False,
        spans=spans,
        artifact_dir=f"/trials/{task_id}",
    )


def test_rollout_stats_reports_the_length_distribution_not_just_its_mean() -> None:
    """Percentiles separate a shortening policy from a bimodal one.

    Both batches below have the same mean, so a mean-only metric cannot tell
    them apart -- but one is a tight distribution and the other has collapsed
    into a terminate-immediately mode plus a never-terminate mode, which is the
    failure this metric exists to catch.
    """
    tight = rollout_stats(
        [_sized_trial(f"t{i}", lengths=[100], canonical=True) for i in range(4)], max_tokens=4096
    )
    assert tight.mean_sampled_tokens == 100.0
    assert (tight.p50_sampled_tokens, tight.p99_sampled_tokens) == (100, 100)

    bimodal = rollout_stats(
        [_sized_trial("a", lengths=[2], canonical=True)]
        + [_sized_trial("b", lengths=[2], canonical=True)]
        + [_sized_trial("c", lengths=[198], canonical=True)]
        + [_sized_trial("d", lengths=[198], canonical=True)],
        max_tokens=4096,
    )
    assert bimodal.mean_sampled_tokens == 100.0
    assert bimodal.p50_sampled_tokens == 198
    assert bimodal.max_sampled_tokens == 198
    # Same mean, wildly different shape: the spread is the tell.
    assert bimodal.p50_sampled_tokens != tight.p50_sampled_tokens

    # Per-trial totals sum a multi-turn episode's spans rather than reporting
    # each call separately, since termination is an episode-level property.
    multi = rollout_stats(
        [_sized_trial("m", lengths=[10, 20, 30], canonical=True)], max_tokens=4096
    )
    assert multi.mean_sampled_tokens == 60.0
    assert multi.max_sampled_tokens == 60


def test_rollout_stats_entropy_estimate_is_the_mean_negated_logprob() -> None:
    """At temperature 1.0 this is a Monte Carlo estimate of per-token entropy."""
    stats = rollout_stats([_sized_trial("a", lengths=[3], canonical=True)], max_tokens=4096)
    assert stats.entropy_estimate == pytest.approx(1.0)

    # Mixed magnitudes average over TOKENS, not over trials, so a long rollout
    # weighs proportionally -- entropy is a per-token quantity.
    mixed = rollout_stats(
        [
            _trial("short", passed=False, spans=True),  # 2 tokens at -0.5, -1.0
            _sized_trial("long", lengths=[6], canonical=True),  # 6 tokens at -1.0
        ],
        max_tokens=4096,
    )
    assert mixed.entropy_estimate == pytest.approx((0.5 + 1.0 + 6 * 1.0) / 8)

    assert rollout_stats([], max_tokens=4096).entropy_estimate == 0.0


def test_rollout_stats_counts_trials_the_cross_tokenizer_path_will_refuse() -> None:
    """One non-canonical span anywhere disqualifies its whole trial.

    `reconstruct_conversation` returns None for a trial if ANY span lost its
    delta_messages, so the count is per TRIAL, not per span -- a trial with one
    bad span out of four contributes nothing to teacher scoring.
    """
    stats = rollout_stats(
        [
            _sized_trial("ok", lengths=[5, 5], canonical=True),
            _sized_trial("bad", lengths=[5, 5, 5], canonical=False),
        ],
        max_tokens=4096,
    )
    assert stats.trials_without_delta == 1
    assert stats.trials_with_spans == 2

    # A trial with no spans at all is already counted as empty_span_trials and
    # must not be double-counted here.
    with_empty = rollout_stats([_trial("none", passed=False, spans=False)], max_tokens=4096)
    assert with_empty.empty_span_trials == 1
    assert with_empty.trials_without_delta == 0


def test_module_scope_never_imports_the_harbor_extra() -> None:
    """The collector module must stay importable without the harbor extra."""
    assert rollouts_module.__file__ is not None
    tree = ast.parse(Path(rollouts_module.__file__).read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            roots.add(node.module.split(".")[0])
    assert not roots & {"harbor", "yaml"}
    # wmh.distill.agents and wmh.evals.harbor import harbor at module scope, and
    # wmh.distill.renderers subclasses tinker-cookbook classes at module scope; the
    # collector may only pull them inside a guarded lazy block.
    banned_wmh = {"wmh.distill.agents", "wmh.distill.renderers", "wmh.evals"}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            assert not any(
                node.module == name or node.module.startswith(name + ".") for name in banned_wmh
            )


# --- infrastructure failures are not task failures (audit defect 5) -----------------------------
def test_infra_failed_trials_are_excluded_from_the_solve_rate() -> None:
    """A trial whose sandbox never existed has no verifier evidence, so it cannot be a 0.

    `missing_reward="zero"` keeps a stand-in 0.0 for advantage estimation, but reporting it as a
    task failure is how three Super `student-before` baselines were published as 0.0% from 51/51
    E2B rate-limited trials, then used as the no-regression leg of a promotion gate.
    """
    stats = rollout_stats(
        [
            _trial("task-a", passed=True, spans=True),
            _trial("task-b", passed=False, spans=True),
            _trial("task-c", passed=False, spans=False, stop_reason=None, infra_failed=True),
            _trial("task-d", passed=False, spans=False, stop_reason=None, infra_failed=True),
        ],
        max_tokens=4096,
    )
    assert stats.trials == 4
    assert stats.executed_trials == 2
    assert stats.infra_failed_trials == 2
    # 1 of the 2 trials that actually ran passed.
    assert stats.solve_rate == 0.5
    # What advantage estimation sees, kept alongside instead of standing in for the above.
    assert stats.raw_solve_rate == 0.25


def test_an_all_infra_failed_batch_is_a_null_measurement_not_a_zero() -> None:
    stats = rollout_stats(
        [
            _trial("task-a", passed=False, spans=False, stop_reason=None, infra_failed=True),
            _trial("task-b", passed=False, spans=False, stop_reason=None, infra_failed=True),
        ],
        max_tokens=4096,
    )
    assert stats.trials == 2
    assert stats.executed_trials == 0
    assert stats.infra_failed_trials == 2
    assert stats.solve_rate == 0.0
    assert stats.scaffold_loss_rate == 0.0


# --- observability: the number that was 88.8% and invisible (audit defect 5/6) -------------------
def test_scaffold_loss_rate_counts_every_episode_the_harness_cut_off() -> None:
    stats = rollout_stats(
        [
            _trial("task-a", passed=True, spans=True, stop_reason="submitted"),
            _trial("task-b", passed=False, spans=True, stop_reason="max_turns"),
            _trial("task-c", passed=False, spans=True, stop_reason="no_tool_call"),
            _trial("task-d", passed=False, spans=True, stop_reason="output_truncated"),
            _trial("task-e", passed=False, spans=True, stop_reason="unparsed_tool_call"),
            _trial("task-f", passed=False, spans=True, stop_reason="budget"),
            _trial("task-g", passed=False, spans=True, stop_reason="provider_error"),
            _trial("task-h", passed=False, spans=True, stop_reason=None),
        ],
        max_tokens=4096,
    )
    # Only the explicit submit is a completion, so 6 of the 7 trials that reported a stop reason
    # were cut off. `task-h` has no readable trace: we do not know whether it submitted, so it is
    # neither a loss nor a success and is excluded from the rate rather than assumed guilty. It
    # cannot hide, though — it is surfaced as the `unknown` bucket asserted just below, and it
    # still counts in `trials`. Assuming it a loss would let a batch of dead sandboxes report
    # "100% scaffold loss", which reads as "the scaffold cut off working episodes" when in fact
    # nothing ever ran; that distinction is the whole point of `infra_failed_trials`.
    assert stats.scaffold_loss_rate == pytest.approx(6 / 7)
    assert stats.stop_reason_counts == {
        "budget": 1,
        "max_turns": 1,
        "no_tool_call": 1,
        "output_truncated": 1,
        "provider_error": 1,
        "submitted": 1,
        "unknown": 1,
        "unparsed_tool_call": 1,
    }


def test_a_fully_submitting_batch_reports_no_scaffold_loss() -> None:
    stats = rollout_stats(
        [
            _trial("task-a", passed=True, spans=True),
            _trial("task-b", passed=False, spans=True),
        ],
        max_tokens=4096,
    )
    assert stats.scaffold_loss_rate == 0.0
    assert stats.stop_reason_counts == {"submitted": 2}


def test_infra_failures_never_inflate_the_scaffold_loss_rate() -> None:
    """A trial that never ran is an infra problem, not a scaffold-termination problem."""
    stats = rollout_stats(
        [
            _trial("task-a", passed=True, spans=True, stop_reason="submitted"),
            _trial("task-b", passed=False, spans=False, stop_reason=None, infra_failed=True),
        ],
        max_tokens=4096,
    )
    assert stats.executed_trials == 1
    assert stats.scaffold_loss_rate == 0.0


# --- budgets threaded from config (audit defects 2 and 3) ---------------------------------------
def test_collect_rollouts_threads_the_configured_budgets_into_the_scorer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wall budget and the served context window must come from the run config.

    Neither was passed before: every rollout inherited the 300s evaluation default (which ended
    31% of Super trials on the clock) and every pi runner assumed a 128,000-token window against a
    smaller deployment (118 context-overflow 400s in one run).
    """
    harness = HarnessDoc.baseline()
    trials_dir = tmp_path / "run" / "harbor" / "step-0000" / f"wmh-{harness.doc_hash[:12]}"
    capture = _CreateCapture(_StubScorer(_report(harness, trials_dir)))
    capture.install(monkeypatch)
    cfg = _cfg(tmp_path)
    cfg = cfg.model_copy(
        update={
            "rollout": cfg.rollout.model_copy(
                update={"episode_timeout_s": 2400.0, "context_budget_tokens": 240_000}
            )
        }
    )

    collect_rollouts(0, _TASK_IDS, cfg, harness, _provider_config(), tmp_path / "run")

    kwargs = capture.kwargs[0]
    assert kwargs["episode_timeout_s"] == pytest.approx(2400.0)
    assert kwargs["context_window"] == 240_000
    # The training path still tolerates a dead trial rather than aborting a long run.
    assert kwargs["missing_reward"] == "zero"


def test_submit_then_ungradeable_is_a_scaffold_success_not_a_loss() -> None:
    """The scaffold rate must not borrow solve_rate's gradeable denominator.

    An episode that reached `submit` and then had its VERIFIER time out did everything the
    scaffold is measured on; only its task outcome is unknown. Sharing one denominator dropped
    it out of a rate it belongs in as a success, which inflated the rate. On the live 48-episode
    probe that read 15.22% over 46 gradeable trials when the true scaffold loss was 14.58%.
    """
    stats = rollout_stats(
        [
            _trial("task-a", passed=True, spans=True, stop_reason="submitted"),
            _trial("task-b", passed=False, spans=True, stop_reason="submitted", infra_failed=True),
            _trial("task-c", passed=False, spans=True, stop_reason="max_turns"),
        ],
        max_tokens=4096,
    )
    # 3 trials reported a stop reason, 2 of them submitted -> one real scaffold loss.
    assert stats.scaffold_loss_rate == pytest.approx(1 / 3)
    # ...while the ungradeable one is still excluded from the solve rate.
    assert stats.executed_trials == 2
    assert stats.infra_failed_trials == 1
    assert stats.solve_rate == pytest.approx(1 / 2)


# --- the graded rate's denominator (a missing test report is not a 0.0) --------------------------
def test_the_graded_rate_averages_only_trials_that_have_a_test_report() -> None:
    """Same trials as the binary rate, read at test resolution.

    The two hidden-progress trials here are the shape 9 of 46 probe trials had: reward 0, one of
    two tests passing. Binary reports 1/4; graded reports 0.5, and the difference is the entire
    reason this metric exists.
    """
    stats = rollout_stats(
        [
            _trial("task-a", passed=True, spans=True, tests=GradedTests(passed=2, resolved=2)),
            _trial("task-b", passed=False, spans=True, tests=GradedTests(passed=1, resolved=2)),
            _trial("task-c", passed=False, spans=True, tests=GradedTests(passed=1, resolved=2)),
            _trial("task-d", passed=False, spans=True, tests=GradedTests(passed=0, resolved=1)),
        ],
        max_tokens=4096,
    )

    assert stats.solve_rate == 0.25
    assert stats.graded_trials == 4
    assert stats.graded_solve_rate == pytest.approx(0.5)


def test_an_ungradeable_trial_is_excluded_from_the_graded_rate_too() -> None:
    """Rule parity with `solve_rate`: a verifier that never spoke wrote no CTRF report either.

    The one exception the fixture forces: an `infra_failed` trial that somehow does carry a report
    is still excluded, because the aggregation rule is the measurement status, not the file.
    """
    stats = rollout_stats(
        [
            _trial("task-a", passed=True, spans=True, tests=GradedTests(passed=2, resolved=2)),
            _trial("task-b", passed=False, spans=True, tests=GradedTests(passed=1, resolved=2)),
            _trial(
                "task-c",
                passed=False,
                spans=True,
                stop_reason="submitted",
                infra_failed=True,
                tests=None,
            ),
            _trial(
                "task-d",
                passed=False,
                spans=False,
                stop_reason=None,
                infra_failed=True,
                tests=GradedTests(passed=0, resolved=4),
            ),
        ],
        max_tokens=4096,
    )

    assert (stats.trials, stats.executed_trials, stats.infra_failed_trials) == (4, 2, 2)
    assert stats.graded_trials == 2
    assert stats.graded_solve_rate == pytest.approx(0.75)  # (1.0 + 0.5) / 2, the two gradeable ones


def test_a_gradeable_trial_with_no_test_report_leaves_the_graded_denominator() -> None:
    """A reward without a parseable report is graded-absent, never graded-zero.

    Counting it as 0.0 would drag the graded rate below the binary one it is meant to refine, so
    the denominators are allowed to differ and `graded_trials` is what says they did.
    """
    stats = rollout_stats(
        [
            _trial("task-a", passed=True, spans=True, tests=GradedTests(passed=1, resolved=1)),
            _trial("task-b", passed=False, spans=True, tests=None),
        ],
        max_tokens=4096,
    )

    assert stats.executed_trials == 2
    assert stats.solve_rate == 0.5
    assert stats.graded_trials == 1
    assert stats.graded_solve_rate == 1.0  # not 0.5: the unreported trial is not a zero


def test_a_batch_with_no_test_reports_at_all_is_a_null_graded_measurement() -> None:
    stats = rollout_stats(
        [
            _trial("task-a", passed=True, spans=True),
            _trial("task-b", passed=False, spans=True),
        ],
        max_tokens=4096,
    )

    assert stats.solve_rate == 0.5
    assert stats.graded_trials == 0
    # 0.0 with a zero denominator: callers read the count, exactly as with `solve_rate`.
    assert stats.graded_solve_rate == 0.0
    assert rollout_stats([], max_tokens=4096).graded_solve_rate == 0.0


# --- the truncation tripwire: harbor's own guard is unreachable ----------------------------------


def _trial_with_spans(task_id: str, sampled_lengths: Sequence[int]) -> TrialRecord:
    """A trial whose turns sampled exactly the given token counts."""
    return TrialRecord(
        task_id=task_id,
        attempt=1,
        trial_name=f"{task_id}__s1",
        reward=1.0,
        passed=True,
        spans=[
            TokenSpan(
                call_index=index,
                prompt_token_ids=[1, 2, index],
                sampled_token_ids=[70] * length,
                sampled_logprobs=[-0.5] * length,
            )
            for index, length in enumerate(sampled_lengths)
        ],
        stop_reason="submitted",
        artifact_dir=f"/trials/{task_id}",
    )


def test_harbors_output_truncation_guard_can_never_fire() -> None:
    """Why wmh has to count truncation itself (`RolloutStats.truncated_spans`).

    `harbor/llms/tinker.py` raises `OutputLengthExceededError` under
    `if not parse_success and len(completion_tokens) >= self._max_tokens`, but
    `parse_success` is a `ParseTermination` StrEnum and every member is a
    non-empty string, so `not parse_success` is always False. If the cookbook
    ever gives a member a falsy value, harbor starts raising and this counter
    stops being the only signal, which is worth knowing about.
    """
    from tinker_cookbook.renderers.base import ParseTermination

    assert all(bool(member) for member in ParseTermination)


def test_a_span_that_sampled_the_whole_output_cap_is_counted_as_truncated() -> None:
    """The turn was cut off mid-answer, and nothing upstream reports it."""
    stats = rollout_stats(
        [
            _trial_with_spans("task-a", [12, 4096, 30]),
            _trial_with_spans("task-b", [12, 30]),
            _trial_with_spans("task-c", [4096, 4096]),
        ],
        max_tokens=4096,
    )

    assert stats.truncated_spans == 3
    assert stats.truncated_span_trials == 2
    # Trials, spans and solve rate are unaffected: a truncated turn is still a real trial.
    assert stats.trials == 3
    assert stats.trials_with_spans == 3


def test_no_span_reaching_the_cap_reports_no_truncation() -> None:
    stats = rollout_stats([_trial_with_spans("task-a", [4095, 10])], max_tokens=4096)
    assert stats.truncated_spans == 0
    assert stats.truncated_span_trials == 0


def test_the_truncation_count_reads_the_cap_it_is_given() -> None:
    """A batch is only truncated relative to the cap it sampled under."""
    records = [_trial_with_spans("task-a", [512, 200])]
    assert rollout_stats(records, max_tokens=4096).truncated_spans == 0
    assert rollout_stats(records, max_tokens=512).truncated_spans == 1


def test_collect_rollouts_warns_when_turns_hit_the_output_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The tripwire has to be loud: harbor swallows truncation silently.

    Harbor's `OutputLengthExceededError` guard is unreachable, so a turn cut off
    at the cap reaches the agent as a half-written action and the episode goes on.
    The collector is the only place that can say so.
    """
    from wmh.distill.config import SamplingConfig

    run_dir = tmp_path / "run"
    trials_dir = tmp_path / "trials"
    harness = HarnessDoc.baseline()
    stub = _StubScorer(_report(harness, trials_dir))
    _CreateCapture(stub).install(monkeypatch)
    cap = 16
    cfg = _cfg(tmp_path).model_copy(update={"sampling": SamplingConfig(max_tokens=cap)})

    _write_terminus_trial(trials_dir / "task-a__s1", turns=2, completion_length=cap)
    for trial_name in ("task-a__s2", "task-b__s1", "task-b__s2"):
        _write_terminus_trial(trials_dir / trial_name, turns=2, completion_length=cap - 1)

    with caplog.at_level(logging.WARNING, logger="wmh.distill.rollouts"):
        _records, stats = collect_rollouts(0, _TASK_IDS, cfg, harness, _provider_config(), run_dir)

    assert stats.truncated_spans == 2
    assert stats.truncated_span_trials == 1
    assert any(
        "sampled the full sampling.max_tokens = 16" in record.getMessage()
        for record in caplog.records
    )
