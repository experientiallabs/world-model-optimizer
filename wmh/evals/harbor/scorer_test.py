"""Tests for projecting harbor job results into harness score reports (fakes only)."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import DatasetConfig, JobConfig, RetryConfig
from harbor.models.job.result import JobResult, JobStats
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.models.trial.result import AgentInfo, ExceptionInfo, ModelInfo, TrialResult
from harbor.models.verifier.result import VerifierResult

import wmh.evals.harbor.scorer as scorer_module
from wmh.evals.harbor.agent import WMH_HARBOR_AGENT_IMPORT_PATH
from wmh.evals.harbor.e2b_template_policy import WMH_HARBOR_E2B_ENVIRONMENT_IMPORT_PATH
from wmh.evals.harbor.scorer import (
    HarborJobRunner,
    HarborRewardMissingError,
    HarborRun,
    HarborScorer,
)
from wmh.harness.doc import HarnessDoc
from wmh.harness.scoring import RewardMode
from wmh.providers.base import ProviderConfig, ProviderKind

_JOB_ID = UUID("00000000-0000-4000-8000-000000000001")
_SUFFIXES = ("a7Hm2Ks", "m4Vx8Pa", "z9Tc3Wb", "q6Rn5Jd")


def _provider() -> ProviderConfig:
    return ProviderConfig(kind=ProviderKind.BEDROCK, model="worker-model", region="us-west-2")


def _tasks(tmp_path: Path, task_ids: tuple[str, ...]) -> list[TaskConfig]:
    return [TaskConfig(path=tmp_path / "tasks" / task_id, source="tasks") for task_id in task_ids]


def _job_template(
    tmp_path: Path, *, backend: EnvironmentType = EnvironmentType.DOCKER
) -> JobConfig:
    return JobConfig(
        job_name="template",
        jobs_dir=tmp_path / "jobs",
        n_concurrent_trials=4,
        environment=EnvironmentConfig(type=backend),
        agents=[AgentConfig()],
        datasets=[DatasetConfig(path=tmp_path / "tasks")],
    )


def _trial(
    tmp_path: Path,
    task_id: str,
    attempt: int,
    *,
    reward: float | None,
    exception: str | None = None,
) -> TrialResult:
    name = f"{task_id}__{_SUFFIXES[attempt - 1]}"
    now = datetime.now(UTC)
    return TrialResult(
        task_name=task_id,
        trial_name=name,
        trial_uri=f"file://{tmp_path}/{name}",
        task_id=TaskConfig(path=tmp_path / "tasks" / task_id).get_task_id(),
        source="tasks",
        task_checksum="c" * 64,
        config=TrialConfig(task=TaskConfig(path=tmp_path / "tasks" / task_id), job_id=_JOB_ID),
        agent_info=AgentInfo(
            name="wmh-harness",
            version="1",
            model_info=ModelInfo(name="worker-model", provider="bedrock"),
        ),
        verifier_result=(
            None if reward is None else VerifierResult.model_construct(rewards={"reward": reward})
        ),
        exception_info=(
            None
            if exception is None
            else ExceptionInfo(
                exception_type=exception,
                exception_message="failed",
                exception_traceback="trace",
                occurred_at=now,
            )
        ),
        started_at=now,
        finished_at=now,
    )


class _Runner:
    """Materializes trial dirs under the candidate's deterministic job dir, like harbor.

    `transcripts` maps a trial name to the exact `agent/wmh-run.json` bytes harbor would have
    written for that episode; unlisted trials get no transcript (the common no-telemetry case).
    """

    def __init__(
        self, trials: list[TrialResult], *, transcripts: dict[str, str] | None = None
    ) -> None:
        self.trials = trials
        self.transcripts = transcripts or {}
        self.configs: list[JobConfig] = []

    def run(
        self,
        config: JobConfig,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> HarborRun:
        del should_cancel
        self.configs.append(config)
        job_dir = config.jobs_dir / config.job_name
        for trial in self.trials:
            trial_dir = job_dir / trial.trial_name
            trial_dir.mkdir(parents=True, exist_ok=True)
            (trial_dir / "result.json").write_text(trial.model_dump_json(), encoding="utf-8")
            transcript = self.transcripts.get(trial.trial_name)
            if transcript is not None:
                agent_dir = trial_dir / "agent"
                agent_dir.mkdir(parents=True, exist_ok=True)
                (agent_dir / "wmh-run.json").write_text(transcript, encoding="utf-8")
        now = datetime.now(UTC)
        result = JobResult(
            id=_JOB_ID,
            started_at=now,
            finished_at=now,
            n_total_trials=len(self.trials),
            stats=JobStats.from_trial_results(self.trials, n_total_trials=len(self.trials)),
            trial_results=self.trials,
        )
        return HarborRun(result=result, job_dir=job_dir)


def _scorer(
    tmp_path: Path,
    trials: list[TrialResult],
    *,
    task_ids: tuple[str, ...] = ("task-a", "task-b"),
    attempts: int = 1,
    reward_mode: RewardMode = "raw",
    agent_concurrency: int | None = None,
    transcripts: dict[str, str] | None = None,
) -> tuple[HarborScorer, _Runner]:
    runner = _Runner(trials, transcripts=transcripts)
    scorer = HarborScorer(
        job_template=_job_template(tmp_path),
        tasks=_tasks(tmp_path, task_ids),
        provider_config=_provider(),
        reward_mode=reward_mode,
        attempts=attempts,
        harness_backend="e2b",
        e2b_template="pi-template",
        agent_concurrency=agent_concurrency,
        runner=runner,
    )
    return scorer, runner


def test_scorer_projects_rewards_and_injects_the_exact_candidate(tmp_path: Path) -> None:
    candidate = HarnessDoc.baseline("candidate")
    trials = [
        _trial(tmp_path, "task-a", 1, reward=1.0),
        _trial(tmp_path, "task-b", 1, reward=0.0),
    ]
    scorer, runner = _scorer(tmp_path, trials)

    report = scorer.score(candidate)

    assert report.doc_hash == candidate.doc_hash
    assert report.score == 0.5
    by_task = report.by_task()
    assert by_task["task-a"][0].passed is True
    assert by_task["task-b"][0].passed is False
    assert by_task["task-a"][0].note == "completed"
    expected_dir = tmp_path / "jobs" / f"wmh-{candidate.doc_hash[:12]}"
    assert by_task["task-a"][0].artifact_dir == str(expected_dir / trials[0].trial_name)

    [config] = runner.configs
    # Deterministic per-candidate job dir: the harbor-native trial-resume key.
    assert config.jobs_dir / config.job_name == scorer.candidate_job_dir(candidate)
    assert config.job_name == f"wmh-{candidate.doc_hash[:12]}"
    [agent] = config.agents
    assert agent.import_path == WMH_HARBOR_AGENT_IMPORT_PATH
    assert agent.model_name == "bedrock/worker-model"
    assert agent.kwargs["harness"] == candidate.model_dump(mode="json")
    assert agent.kwargs["harness_backend"] == "e2b"
    assert agent.kwargs["e2b_template"] == "pi-template"
    assert agent.kwargs["episode_workers"] >= 2 * config.n_concurrent_trials
    assert config.retry == RetryConfig(max_retries=0)
    assert [task.get_task_id().get_name() for task in config.tasks] == ["task-a", "task-b"]
    assert config.datasets == []


def test_failed_trial_with_a_written_reward_is_a_scored_cell_not_an_infra_halt(
    tmp_path: Path,
) -> None:
    """AgentTimeoutError with reward 0 is a CANDIDATE outcome; a missing reward is not."""
    scored = [
        _trial(tmp_path, "task-a", 1, reward=0.0, exception="AgentTimeoutError"),
        _trial(tmp_path, "task-b", 1, reward=1.0),
    ]
    scorer, _runner = _scorer(tmp_path, scored)
    report = scorer.score(HarnessDoc.baseline())
    cell = report.by_task()["task-a"][0]
    assert cell.reward == 0.0
    assert cell.passed is False
    assert cell.note == "completed with AgentTimeoutError"

    missing = [
        _trial(tmp_path, "task-a", 1, reward=None, exception="RuntimeError"),
        _trial(tmp_path, "task-b", 1, reward=1.0),
    ]
    scorer, _runner = _scorer(tmp_path, missing)
    with pytest.raises(HarborRewardMissingError, match="no verifier reward"):
        scorer.score(HarnessDoc.baseline())


def test_outcome_shaped_verifier_failures_score_zero_instead_of_halting(tmp_path: Path) -> None:
    """harbor classifies these as outcome-shaped (its retry exclude list): the verifier ran
    against the candidate's artifacts and terminally produced no reward. Scoring 0 matches the
    benchmark's absent-reward-file semantics and keeps a deterministic verifier timeout from
    wedging the boundary in a raise/prune/re-run loop."""
    trials = [
        _trial(tmp_path, "task-a", 1, reward=None, exception="VerifierTimeoutError"),
        _trial(tmp_path, "task-b", 1, reward=1.0),
    ]
    scorer, _runner = _scorer(tmp_path, trials, reward_mode="positive-binary")
    report = scorer.score(HarnessDoc.baseline())
    cell = report.by_task()["task-a"][0]
    assert cell.reward == 0.0
    assert cell.passed is False
    assert cell.note == "completed with VerifierTimeoutError"
    assert report.score == 0.5

    misconfigured = [
        _trial(tmp_path, "task-a", 1, reward=1.0),
        _trial(tmp_path, "task-b", 1, reward=1.0),
    ]
    scorer, _runner = _scorer(tmp_path, misconfigured)
    scorer._reward_key = "grade"  # a clean trial + wrong key is a config error, not infra
    with pytest.raises(HarborRewardMissingError, match=r"available reward keys: \['reward'\]"):
        scorer.score(HarnessDoc.baseline())


def test_positive_binary_mode_passes_on_any_positive_reward_and_keeps_raw_values(
    tmp_path: Path,
) -> None:
    trials = [
        _trial(tmp_path, "task-a", 1, reward=0.25),
        _trial(tmp_path, "task-b", 1, reward=0.0),
    ]
    scorer, _runner = _scorer(tmp_path, trials, reward_mode="positive-binary")
    report = scorer.score(HarnessDoc.baseline())
    cell = report.by_task()["task-a"][0]
    assert cell.reward == 0.25  # raw reward untouched
    assert cell.passed is True
    assert report.score == 0.5


def _trial_name(task_id: str, attempt: int) -> str:
    return f"{task_id}__{_SUFFIXES[attempt - 1]}"


def test_scorer_populates_telemetry_from_the_wmh_transcript(tmp_path: Path) -> None:
    """Turns/calls/tokens/stop_reason lift from agent/wmh-run.json; hit_turn_cap fires when the
    episode reached the scored doc's cap and hit_timeout on a harbor-timeout stop_reason."""
    candidate = HarnessDoc.baseline("candidate")  # baseline max_turns == 20
    trials = [
        _trial(tmp_path, "task-a", 1, reward=1.0),
        _trial(tmp_path, "task-b", 1, reward=0.0, exception="AgentTimeoutError"),
    ]
    transcripts = {
        # A clean submitted episode well under the turn cap.
        _trial_name("task-a", 1): json.dumps(
            {
                "task_id": "task-a",
                "stop_reason": "submitted",
                "turns": 7,
                "worker_usage": {"input_tokens": 1200, "output_tokens": 340, "calls": 7},
            }
        ),
        # A timed-out partial episode that reached the turn cap.
        _trial_name("task-b", 1): json.dumps(
            {
                "task_id": "task-b",
                "stop_reason": "cancelled-by-harbor-timeout",
                "turns": 20,
                "partial": True,
                "worker_usage": {"input_tokens": 9000, "output_tokens": 512, "calls": 20},
            }
        ),
    }
    scorer, _runner = _scorer(tmp_path, trials, transcripts=transcripts)

    report = scorer.score(candidate)

    clean = report.by_task()["task-a"][0]
    assert clean.turns == 7
    assert clean.calls == 7
    assert clean.input_tokens == 1200
    assert clean.output_tokens == 340
    assert clean.stop_reason == "submitted"
    assert clean.hit_turn_cap is False
    assert clean.hit_timeout is False

    timed_out = report.by_task()["task-b"][0]
    assert timed_out.turns == 20
    assert timed_out.hit_turn_cap is True  # turns >= baseline max_turns (20)
    assert timed_out.hit_timeout is True  # cancelled-by-harbor-timeout stop_reason
    assert timed_out.input_tokens == 9000

    # The candidate-level aggregate is now computable straight off the report.
    summary = report.telemetry_summary()
    assert summary["n_with_telemetry"] == 2
    assert summary["turn_cap_hit_rate"] == pytest.approx(0.5)
    assert summary["timeout_rate"] == pytest.approx(0.5)


def test_missing_or_unparseable_transcript_is_non_fatal_and_defaults_telemetry(
    tmp_path: Path,
) -> None:
    """A missing/unparseable wmh-run.json must never break scoring: the cell scores normally with
    every telemetry field defaulted (turns None, flags False), and the aggregate is empty."""
    candidate = HarnessDoc.baseline("candidate")
    trials = [
        _trial(tmp_path, "task-a", 1, reward=1.0),  # no transcript at all
        _trial(tmp_path, "task-b", 1, reward=0.0),  # unparseable transcript
    ]
    transcripts = {_trial_name("task-b", 1): "{not valid json"}
    scorer, _runner = _scorer(tmp_path, trials, transcripts=transcripts)

    report = scorer.score(candidate)  # must not raise

    assert report.score == pytest.approx(0.5)  # scoring unaffected
    for cell in report.cells:
        assert cell.turns is None
        assert cell.calls is None
        assert cell.input_tokens is None
        assert cell.output_tokens is None
        assert cell.stop_reason == ""
        assert cell.hit_turn_cap is False
        assert cell.hit_timeout is False
    assert report.telemetry_summary() == {"n_with_telemetry": 0}


def test_timeout_exception_marks_hit_timeout_even_without_a_transcript(tmp_path: Path) -> None:
    """A cancelled episode whose transcript was lost still reports hit_timeout from the trial
    exception alone, so timeout pressure is not silently dropped."""
    candidate = HarnessDoc.baseline("candidate")
    trials = [
        _trial(tmp_path, "task-a", 1, reward=0.0, exception="AgentTimeoutError"),
        _trial(tmp_path, "task-b", 1, reward=1.0),
    ]
    scorer, _runner = _scorer(tmp_path, trials)  # no transcripts written

    report = scorer.score(candidate)

    timed_out = report.by_task()["task-a"][0]
    assert timed_out.turns is None  # no transcript, so no turn/token telemetry
    assert timed_out.hit_timeout is True  # inferred from the AgentTimeoutError exception
    assert report.by_task()["task-b"][0].hit_timeout is False


def test_entry_prunes_only_unscoreable_trial_dirs(tmp_path: Path) -> None:
    """Missing/unparseable result.json or exception-without-reward dirs are pruned; scoreable
    completed trials survive so harbor's resume re-runs only what is broken."""
    candidate = HarnessDoc.baseline()
    scorer, runner = _scorer(
        tmp_path,
        [
            _trial(tmp_path, "task-a", 1, reward=1.0),
            _trial(tmp_path, "task-b", 1, reward=0.0),
        ],
    )
    job_dir = scorer.candidate_job_dir(candidate)

    # Distinct names from anything the fake runner writes, so survival/pruning is attributable
    # to the entry prune alone.
    keep = _trial(tmp_path, "task-a", 2, reward=0.0, exception="AgentTimeoutError")
    (job_dir / keep.trial_name).mkdir(parents=True)
    (job_dir / keep.trial_name / "result.json").write_text(keep.model_dump_json(), encoding="utf-8")
    unparseable = job_dir / "task-b__broken1"
    unparseable.mkdir()
    (unparseable / "result.json").write_text("{not json", encoding="utf-8")
    missing_result = job_dir / "task-b__broken2"
    missing_result.mkdir()
    crashed = _trial(tmp_path, "task-b", 3, reward=None, exception="RuntimeError")
    (job_dir / crashed.trial_name).mkdir()
    (job_dir / crashed.trial_name / "result.json").write_text(
        crashed.model_dump_json(), encoding="utf-8"
    )
    verifier_timeout = _trial(tmp_path, "task-b", 4, reward=None, exception="VerifierTimeoutError")
    (job_dir / verifier_timeout.trial_name).mkdir()
    (job_dir / verifier_timeout.trial_name / "result.json").write_text(
        verifier_timeout.model_dump_json(), encoding="utf-8"
    )

    scorer.score(candidate)

    assert (job_dir / keep.trial_name).is_dir()  # written reward: a kept candidate outcome
    assert not unparseable.exists()
    assert not missing_result.exists()
    assert not (job_dir / crashed.trial_name).exists()
    # Outcome-shaped verifier failure: scored 0, so re-running it would loop forever.
    assert (job_dir / verifier_timeout.trial_name).is_dir()
    assert runner.configs  # the job still ran after pruning


def test_backend_and_template_conflicts_are_rejected_not_rewritten(tmp_path: Path) -> None:
    tasks = _tasks(tmp_path, ("task-a",))
    with pytest.raises(ValueError, match="task_environment='e2b' was requested"):
        HarborScorer(
            job_template=_job_template(tmp_path, backend=EnvironmentType.DOCKER),
            tasks=tasks,
            provider_config=_provider(),
            task_environment="e2b",
            harness_backend="e2b",
            agent_concurrency=1,
        )
    with pytest.raises(ValueError, match="task_environment='docker' was requested"):
        HarborScorer(
            job_template=_job_template(tmp_path, backend=EnvironmentType.E2B),
            tasks=tasks,
            provider_config=_provider(),
            task_environment="docker",
            agent_concurrency=1,
        )
    template = _job_template(tmp_path)
    template.environment.import_path = "somewhere.else:Env"
    template.environment.type = None
    with pytest.raises(ValueError, match="owns task-environment routing"):
        HarborScorer(
            job_template=template,
            tasks=tasks,
            provider_config=_provider(),
            agent_concurrency=1,
        )


def test_builtin_e2b_environment_is_routed_through_the_paced_subclass(tmp_path: Path) -> None:
    template = _job_template(tmp_path, backend=EnvironmentType.E2B)
    template.environment.kwargs = {"keep": "me"}
    scorer = HarborScorer(
        job_template=template,
        tasks=_tasks(tmp_path, ("task-a",)),
        provider_config=_provider(),
        task_environment="e2b",
        harness_backend="e2b",
        agent_concurrency=1,
        runner=_Runner([]),
    )
    config = scorer._candidate_job(HarnessDoc.baseline())
    assert config.environment.type is None
    assert config.environment.import_path == WMH_HARBOR_E2B_ENVIRONMENT_IMPORT_PATH
    assert config.environment.kwargs == {"keep": "me"}  # options survive the routing rewrite


def test_sensitive_env_values_survive_scorer_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """harbor redacts sensitive-named env values on every model_dump when the literal differs
    from os.environ; the scorer's re-validation must never route env-bearing sections through
    that serializer."""
    monkeypatch.delenv("TASK_API_KEY", raising=False)
    monkeypatch.delenv("GRADER_TOKEN", raising=False)
    template = _job_template(tmp_path)
    template.environment.env = {"TASK_API_KEY": "literal-secret-12345"}
    template.verifier.env = {"GRADER_TOKEN": "grader-secret-6789"}
    scorer = HarborScorer(
        job_template=template,
        tasks=_tasks(tmp_path, ("task-a",)),
        provider_config=_provider(),
        harness_backend="e2b",
        agent_concurrency=1,
        runner=_Runner([]),
    )
    config = scorer._candidate_job(HarnessDoc.baseline())
    assert config.environment.env == {"TASK_API_KEY": "literal-secret-12345"}
    assert config.verifier.env == {"GRADER_TOKEN": "grader-secret-6789"}


def test_candidate_job_tasks_drop_their_dataset_source(tmp_path: Path) -> None:
    """Candidate jobs pin tasks directly with no datasets; a surviving dataset source name
    poisons harbor's metrics defaultdict (empty entry for the unknown dataset) and crashes
    its per-trial display hook with IndexError."""
    scorer = HarborScorer(
        job_template=_job_template(tmp_path),
        tasks=_tasks(tmp_path, ("task-a", "task-b")),
        provider_config=_provider(),
        agent_concurrency=1,
        runner=_Runner([]),
    )
    config = scorer._candidate_job(HarnessDoc.baseline())
    assert [task.source for task in config.tasks] == [None, None]


def test_candidate_job_is_revalidated_after_model_copy(tmp_path: Path) -> None:
    """model_copy(update=) skips validation; the scorer must re-validate the exact config."""
    template = _job_template(tmp_path)
    template.agents = [AgentConfig(n_concurrent=4)]
    with pytest.raises(ValueError, match="cannot exceed"):
        HarborScorer(
            job_template=template,
            tasks=_tasks(tmp_path, ("task-a",)),
            provider_config=_provider(),
            harness_backend="e2b",
            # Re-validation catches the now-inconsistent concurrency pair.
            agent_concurrency=2,
        )


def test_template_ownership_and_local_concurrency_rules(tmp_path: Path) -> None:
    tasks = _tasks(tmp_path, ("task-a",))
    template = _job_template(tmp_path)
    template.retry = RetryConfig(max_retries=3)
    with pytest.raises(ValueError, match="through harbor_retries"):
        HarborScorer(
            job_template=template,
            tasks=tasks,
            provider_config=_provider(),
            agent_concurrency=1,
        )
    with pytest.raises(ValueError, match="agent concurrency 1"):
        HarborScorer(
            job_template=_job_template(tmp_path),
            tasks=tasks,
            provider_config=_provider(),
            harness_backend="local",
        )
    owned = _job_template(tmp_path)
    owned.agents = [AgentConfig(import_path="x.y:Z")]
    with pytest.raises(ValueError, match="owns agent identity"):
        HarborScorer(
            job_template=owned,
            tasks=tasks,
            provider_config=_provider(),
            agent_concurrency=1,
        )
    with pytest.raises(ValueError, match="command_timeout_sec must be an integer"):
        HarborScorer(
            job_template=_job_template(tmp_path),
            tasks=tasks,
            provider_config=_provider(),
            harness_backend="e2b",
            agent_concurrency=1,
            command_timeout_sec=0,
        )


def test_harbor_retries_thread_into_retry_config_with_default_exclusions(
    tmp_path: Path,
) -> None:
    scorer = HarborScorer(
        job_template=_job_template(tmp_path),
        tasks=_tasks(tmp_path, ("task-a",)),
        provider_config=_provider(),
        harness_backend="e2b",
        agent_concurrency=1,
        harbor_retries=2,
        runner=_Runner([]),
    )
    config = scorer._candidate_job(HarnessDoc.baseline())
    assert config.retry.max_retries == 2
    # Harbor's default exclude list keeps candidate-outcome exceptions unretried.
    assert config.retry.exclude_exceptions is not None
    assert {
        "AgentTimeoutError",
        "VerifierTimeoutError",
        "RewardFileNotFoundError",
        "VerifierOutputParseError",
    } <= config.retry.exclude_exceptions


def test_concurrent_scores_of_one_candidate_are_rejected(tmp_path: Path) -> None:
    """The entry prune is destructive; a second in-process score of the same doc must not be
    able to delete the first one's in-flight trials."""
    candidate = HarnessDoc.baseline()
    trials = [
        _trial(tmp_path, "task-a", 1, reward=1.0),
        _trial(tmp_path, "task-b", 1, reward=1.0),
    ]
    scorer, _inner = _scorer(tmp_path, trials)
    reentered: list[str] = []

    class _ReentrantRunner(_Runner):
        def run(
            self,
            config: JobConfig,
            *,
            should_cancel: Callable[[], bool] | None = None,
        ) -> HarborRun:
            with pytest.raises(RuntimeError, match="already being scored"):
                scorer.score(candidate)
            reentered.append(config.job_name)
            return super().run(config)

    scorer._runner = _ReentrantRunner(trials)
    report = scorer.score(candidate)
    assert reentered == [f"wmh-{candidate.doc_hash[:12]}"]
    assert report.doc_hash == candidate.doc_hash
    # The guard releases after the score; the same candidate is scoreable again.
    scorer._runner = _Runner(trials)
    scorer.score(candidate)


def test_job_dir_with_a_different_recorded_config_is_never_pruned(tmp_path: Path) -> None:
    """Harbor refuses to resume a dir whose config changed; pruning it first would only
    destroy transcripts. Raise before touching anything."""
    candidate = HarnessDoc.baseline()
    scorer, runner = _scorer(
        tmp_path,
        [
            _trial(tmp_path, "task-a", 1, reward=1.0),
            _trial(tmp_path, "task-b", 1, reward=1.0),
        ],
    )
    job_dir = scorer.candidate_job_dir(candidate)
    other = scorer._candidate_job(candidate).model_copy(update={"n_attempts": 7})
    job_dir.mkdir(parents=True)
    (job_dir / "config.json").write_text(other.model_dump_json(), encoding="utf-8")
    broken = job_dir / "task-a__stale00"
    broken.mkdir()
    (broken / "result.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="different job config"):
        scorer.score(candidate)
    assert broken.is_dir()  # nothing was deleted
    assert runner.configs == []  # and nothing ran


def test_should_cancel_is_polled_during_the_running_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A multi-hour harbor boundary must be cancellable mid-job through the Scorer contract."""
    polled = threading.Event()

    class _HangingJob:
        job_dir = tmp_path / "jobs" / "wmh-hanging"

        @classmethod
        async def create(cls, _config: JobConfig) -> _HangingJob:
            return cls()

        async def run(self) -> JobResult:
            await asyncio.sleep(3600)
            raise AssertionError("unreachable")

    def cancel_requested() -> bool:
        polled.set()
        return True

    monkeypatch.setattr(scorer_module, "Job", _HangingJob)
    runner = scorer_module.HarborJobRunner(poll_interval_s=0.01)
    with pytest.raises(scorer_module.HarnessSearchCancelled, match="mid-job"):
        runner.run(_job_template(tmp_path), should_cancel=cancel_requested)
    assert polled.is_set()


def test_sync_runner_works_with_and_without_a_running_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _job_template(tmp_path)
    now = datetime.now(UTC)
    result = JobResult(
        id=_JOB_ID, started_at=now, finished_at=now, n_total_trials=0, stats=JobStats()
    )

    class _FakeJob:
        job_dir = tmp_path / "jobs" / "template"

        @classmethod
        async def create(cls, _config: JobConfig) -> _FakeJob:
            return cls()

        async def run(self) -> JobResult:
            return result

    monkeypatch.setattr(scorer_module, "Job", _FakeJob)
    runner = HarborJobRunner()
    assert runner.run(config).result is result

    async def nested() -> HarborRun:
        return runner.run(config)

    assert asyncio.run(nested()).result is result


def test_cancellation_is_observed_before_any_spend(tmp_path: Path) -> None:
    scorer, runner = _scorer(tmp_path, [])
    with pytest.raises(scorer_module.HarnessSearchCancelled):
        scorer.score(HarnessDoc.baseline(), should_cancel=lambda: True)
    assert runner.configs == []


def test_wmh_import_pulls_neither_harbor_nor_e2b() -> None:
    """The packaging contract of both extras: `import wmh` (and wmh.evals) stays clean, and
    even the harbor subpackage never imports the e2b SDK (that loads only through harbor's
    environment factory when a job actually targets E2B)."""
    code = (
        "import sys\n"
        "import wmh, wmh.evals\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('harbor', 'e2b')]\n"
        "assert not bad, f'eager optional imports: {bad}'\n"
        "import wmh.evals.harbor\n"
        "assert 'harbor' in sys.modules\n"
        "bad = [m for m in sys.modules if m.split('.')[0] == 'e2b']\n"
        "assert not bad, f'harbor subpackage must not import the e2b SDK: {bad}'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
