"""Score harness candidates on real benchmark tasks through harbor.

`HarborScorer` implements the `wmh.harness.scoring.Scorer` protocol: one exact `HarnessDoc`
candidate becomes one harbor job (the WMH agent bridge + a pinned task list), harbor owns the
task environments and the verifier lifecycle, and the verifier rewards project into a
`ScoreReport`. Harbor's own job directory is the artifact record, and each cell points at its
trial directory; nothing is re-read, re-hashed, or copied.

Two operational behaviors matter most here:

- **Trial-level resume.** Each candidate gets a deterministic job directory
  (`jobs_dir/wmh-<doc_hash12>`), and before running, trial directories whose result.json is
  missing/unparseable or that failed without a verifier reward are pruned. Harbor's native
  resume then keeps completed trials and re-runs only what is missing, so a crashed or
  interrupted boundary re-pays a handful of trials instead of the whole matrix.
- **Candidate outcomes vs infra failures.** A trial that raised (e.g. AgentTimeoutError) but
  still carries a written verifier reward is a CANDIDATE outcome: it becomes a scored cell with
  a note. The scorer raises (`HarborRewardMissingError`) only when verifier evidence or the
  configured reward is absent; that is an infrastructure failure no reward can stand in for.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import shutil
import threading
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Self, TypedDict

from harbor import Job
from harbor.models.agent.name import AgentName
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import JobConfig, RetryConfig
from harbor.models.job.result import JobResult
from harbor.models.trial.config import AgentConfig, TaskConfig
from harbor.models.trial.paths import TrialPaths
from harbor.models.trial.result import TrialResult

from wmh.evals.harbor.agent import (
    DEFAULT_EPISODE_WORKERS,
    MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC,
    WMH_HARBOR_AGENT_IMPORT_PATH,
)
from wmh.evals.harbor.e2b_template_policy import WMH_HARBOR_E2B_ENVIRONMENT_IMPORT_PATH
from wmh.evals.harbor.tasks import resolve_harbor_tasks
from wmh.harness.doc import HarnessDoc
from wmh.harness.e2b_sandbox import resolve_e2b_template
from wmh.harness.project_proposer import TRIAL_AGENT_DIR, WMH_RUN_FILENAME
from wmh.harness.runtime import (
    DEFAULT_EVAL_EPISODE_TIMEOUT_S,
    HarnessSearchCancelled,
    validate_episode_timeout_s,
)
from wmh.harness.scoring import RewardMode, ScoreCell, ScoreReport, ScoreRequest, reward_passed
from wmh.providers.base import ProviderConfig

logger = logging.getLogger(__name__)

TaskEnvironment = Literal["docker", "e2b"]
HarnessBackend = Literal["local", "e2b"]

# In-process registry of job dirs with a score() in flight: the entry prune is destructive, so
# concurrent scores of the same candidate must be rejected, not interleaved.
_ACTIVE_GUARD = threading.Lock()
_ACTIVE_JOB_DIRS: set[Path] = set()


class HarborRewardMissingError(RuntimeError):
    """Verifier evidence or the configured reward key is absent from a finished trial.

    This is the one condition the scorer refuses to score around: a missing reward is an
    infrastructure failure (verifier never ran, reward file lost), not a candidate outcome.
    """


@dataclass(frozen=True)
class HarborRun:
    """One completed harbor job and its output directory."""

    result: JobResult
    job_dir: Path


class HarborRunner(Protocol):
    """Synchronous execution seam for one harbor job (fakes replace it in tests)."""

    def run(
        self,
        config: JobConfig,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> HarborRun: ...


class HarborJobRunner:
    """Run harbor's async Python API from synchronous optimizer code.

    `should_cancel` is polled every `poll_interval_s` while the job runs; observing it cancels
    the harbor job task (harbor cancels its in-flight trials and persists what it can; the
    scorer's entry prune makes the interrupted boundary resumable) and raises
    `HarnessSearchCancelled`, so a multi-hour boundary stays cancellable through the Scorer
    contract instead of only at its edges.
    """

    def __init__(self, *, poll_interval_s: float = 2.0) -> None:
        if not poll_interval_s > 0:
            raise ValueError("poll_interval_s must be positive")
        self._poll_interval_s = poll_interval_s

    def run(
        self,
        config: JobConfig,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> HarborRun:
        async def run_job() -> HarborRun:
            job = await Job.create(config)
            job_task = asyncio.ensure_future(job.run())
            if should_cancel is None:
                return HarborRun(result=await job_task, job_dir=job.job_dir)
            while True:
                done, _pending = await asyncio.wait({job_task}, timeout=self._poll_interval_s)
                if done:
                    return HarborRun(result=job_task.result(), job_dir=job.job_dir)
                if should_cancel():
                    job_task.cancel()
                    await asyncio.wait({job_task})
                    if not job_task.cancelled():
                        job_task.exception()  # consume; cancellation is the outcome
                    raise HarnessSearchCancelled("harbor scoring cancelled mid-job")

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(run_job())
        # Called from inside an event loop (e.g. an async CLI): run the job on its own loop in
        # one worker thread instead of failing on nested asyncio.run.
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(lambda: asyncio.run(run_job())).result()


class HarborScorer:
    """Evaluate exact harness candidates through harbor's verifier lifecycle."""

    def __init__(
        self,
        *,
        job_template: JobConfig,
        tasks: Sequence[TaskConfig],
        provider_config: ProviderConfig,
        reward_key: str = "reward",
        reward_mode: RewardMode = "raw",
        attempts: int = 1,
        task_environment: TaskEnvironment = "docker",
        harness_backend: HarnessBackend = "local",
        e2b_template: str | None = None,
        episode_timeout_s: float = DEFAULT_EVAL_EPISODE_TIMEOUT_S,
        command_timeout_sec: int = MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC,
        agent_concurrency: int | None = None,
        harbor_retries: int = 0,
        runner: HarborRunner | None = None,
    ) -> None:
        if not tasks:
            raise ValueError("HarborScorer requires at least one resolved task")
        task_ids = [task.get_task_id().get_name() for task in tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("HarborScorer requires unique task ids")
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
            raise ValueError("attempts must be a positive integer")
        if not reward_key:
            raise ValueError("reward_key must be nonempty")
        if reward_mode not in ("raw", "positive-binary"):
            raise ValueError("reward_mode must be raw or positive-binary")
        if harness_backend not in ("local", "e2b"):
            raise ValueError("harness_backend must be local or e2b")
        if harness_backend == "local" and e2b_template is not None:
            raise ValueError("e2b_template requires harness_backend='e2b'")
        episode_timeout_s = validate_episode_timeout_s(episode_timeout_s)
        if harness_backend == "local" and episode_timeout_s != DEFAULT_EVAL_EPISODE_TIMEOUT_S:
            raise ValueError("episode_timeout_s requires harness_backend='e2b'")
        if isinstance(harbor_retries, bool) or not isinstance(harbor_retries, int):
            raise ValueError("harbor_retries must be a nonnegative integer")
        if harbor_retries < 0:
            raise ValueError("harbor_retries must be a nonnegative integer")
        if agent_concurrency is not None and (
            isinstance(agent_concurrency, bool)
            or not isinstance(agent_concurrency, int)
            or agent_concurrency < 1
        ):
            raise ValueError("agent_concurrency must be a positive integer")
        if (
            isinstance(command_timeout_sec, bool)
            or not isinstance(command_timeout_sec, int)
            or not 1 <= command_timeout_sec <= MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC
        ):
            raise ValueError(
                "command_timeout_sec must be an integer in "
                f"[1, {MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC}]"
            )
        _validate_job_template(job_template)
        environment = _route_task_environment(job_template, task_environment)
        effective_concurrency = agent_concurrency or job_template.n_concurrent_trials
        if harness_backend == "local" and effective_concurrency > 1:
            raise ValueError(
                "local harness execution requires agent concurrency 1 (the local pi runner "
                "shares one runner dir); use harness_backend='e2b' for parallel trials"
            )
        # model_copy(update=) skips validation, so the copied config is re-validated as a whole.
        self._job_template = _revalidated_job_config(
            job_template.model_copy(
                update={
                    "environment": environment,
                    "datasets": [],
                    "tasks": [],
                    "n_concurrent_trials": effective_concurrency,
                    "quiet": True,
                    "retry": RetryConfig(max_retries=harbor_retries),
                },
                deep=True,
            )
        )
        self._tasks = [TaskConfig.model_validate(task.model_dump(mode="python")) for task in tasks]
        self._task_ids = tuple(task_ids)
        self._provider_config = ProviderConfig.model_validate(
            provider_config.model_dump(mode="python")
        )
        self._reward_key = reward_key
        self._reward_mode: RewardMode = reward_mode
        self._attempts = attempts
        self._harness_backend: HarnessBackend = harness_backend
        self._episode_timeout_s = episode_timeout_s
        self._command_timeout_sec = command_timeout_sec
        # The dedicated episode executor must never be smaller than agent concurrency (episodes
        # + uncancellable cleanup share it), or queued episodes burn harbor timeout budget.
        self._episode_workers = max(DEFAULT_EPISODE_WORKERS, 2 * effective_concurrency)
        if harness_backend == "e2b":
            resolved_template = resolve_e2b_template(e2b_template)
            # "" pins "no template" so the agent process cannot drift onto an ambient
            # $WMH_E2B_TEMPLATE that differs from what this scorer resolved.
            self._e2b_template: str | None = (
                resolved_template if resolved_template is not None else ""
            )
        else:
            self._e2b_template = None
        self._runner = runner or HarborJobRunner()

    @classmethod
    async def create(
        cls,
        job_template: JobConfig,
        task_ids: Sequence[str],
        *,
        provider_config: ProviderConfig,
        reward_key: str = "reward",
        reward_mode: RewardMode = "raw",
        attempts: int = 1,
        task_environment: TaskEnvironment = "docker",
        harness_backend: HarnessBackend = "local",
        e2b_template: str | None = None,
        episode_timeout_s: float = DEFAULT_EVAL_EPISODE_TIMEOUT_S,
        command_timeout_sec: int = MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC,
        agent_concurrency: int | None = None,
        harbor_retries: int = 0,
        runner: HarborRunner | None = None,
    ) -> Self:
        """Resolve the exact tasks and construct a scorer that can incur spend.

        `job_template` supplies the run directory (`jobs_dir`), the task environment config,
        and harbor tuning (timeouts, concurrency); it must carry exactly one dataset, no direct
        tasks, and an untouched default agent + retry config (the scorer owns those).
        """
        if len(job_template.datasets) != 1 or job_template.tasks:
            raise ValueError("HarborScorer requires exactly one dataset and no direct tasks")
        tasks = await resolve_harbor_tasks(job_template.datasets[0], task_ids)
        return cls(
            job_template=job_template,
            tasks=tasks,
            provider_config=provider_config,
            reward_key=reward_key,
            reward_mode=reward_mode,
            attempts=attempts,
            task_environment=task_environment,
            harness_backend=harness_backend,
            e2b_template=e2b_template,
            episode_timeout_s=episode_timeout_s,
            command_timeout_sec=command_timeout_sec,
            agent_concurrency=agent_concurrency,
            harbor_retries=harbor_retries,
            runner=runner,
        )

    @property
    def request(self) -> ScoreRequest:
        """The exact task-by-attempt matrix every `score` call evaluates."""
        return ScoreRequest(task_ids=self._task_ids, attempts=self._attempts)

    @property
    def reward_mode(self) -> RewardMode:
        """The frozen reward interpretation this scorer applies."""
        return self._reward_mode

    @property
    def task_pins(self) -> dict[str, str]:
        """One stable provenance pin per resolved task, keyed by task id.

        Git tasks pin their resolved commit and package tasks their name@ref, so a caller can
        record the exact task identity a run was scored against and detect a dataset that
        re-resolves differently on resume. Local-path tasks pin only their resolved path (the
        weaker identity is deliberate: hashing arbitrary task dirs is not this scorer's job).
        """
        pins: dict[str, str] = {}
        for task in self._tasks:
            task_id = task.get_task_id().get_name()
            if task.is_package_task():
                pins[task_id] = f"package:{task.name}@{task.ref or 'latest'}"
            elif task.is_git_task():
                pins[task_id] = f"git:{task.git_url}@{task.git_commit_id}"
            else:
                pins[task_id] = f"path:{task.path}"
        return pins

    def candidate_job_dir(self, doc: HarnessDoc) -> Path:
        """The deterministic job directory one candidate's trials live in (resume key)."""
        return self._job_template.jobs_dir / f"wmh-{doc.doc_hash[:12]}"

    def score(
        self,
        doc: HarnessDoc,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> ScoreReport:
        """Run one candidate through harbor and project its verifier rewards."""
        if should_cancel is not None and should_cancel():
            raise HarnessSearchCancelled("harbor scoring cancelled before job start")
        config = self._candidate_job(doc)
        job_dir = (config.jobs_dir / config.job_name).resolve()
        # The entry prune is destructive; two concurrent scores of one candidate in this
        # process would delete each other's in-flight trials through the shared job dir.
        with _ACTIVE_GUARD:
            if job_dir in _ACTIVE_JOB_DIRS:
                raise RuntimeError(
                    f"candidate {doc.doc_hash[:12]} is already being scored (job dir {job_dir}); "
                    "wait for the in-flight score to finish"
                )
            _ACTIVE_JOB_DIRS.add(job_dir)
        try:
            _assert_job_dir_resumable(job_dir, config)
            pruned = _prune_invalid_trial_dirs(job_dir, reward_key=self._reward_key)
            if pruned:
                logger.info(
                    "pruned %d invalid trial dir(s) under %s; harbor resume re-runs only those",
                    pruned,
                    job_dir,
                )
            run = self._runner.run(config, should_cancel=should_cancel)
        finally:
            with _ACTIVE_GUARD:
                _ACTIVE_JOB_DIRS.discard(job_dir)
        return self._project(doc, run)

    def _candidate_job(self, doc: HarnessDoc) -> JobConfig:
        template_agent = self._job_template.agents[0]
        # Constructor, not model_copy(update=): construction runs AgentConfig's validators.
        agent_fields = {name: getattr(template_agent, name) for name in AgentConfig.model_fields}
        agent_fields.update(
            {
                "name": None,
                "import_path": WMH_HARBOR_AGENT_IMPORT_PATH,
                "model_name": f"{self._provider_config.kind.value}/{self._provider_config.model}",
                "skills": [],
                "env": {},
                "mcp_servers": [],
                "kwargs": {
                    "harness": doc.model_dump(mode="json"),
                    "provider_config": self._provider_config.model_dump(mode="json"),
                    "harness_backend": self._harness_backend,
                    "e2b_template": self._e2b_template,
                    "command_timeout_sec": self._command_timeout_sec,
                    "episode_timeout_sec": self._episode_timeout_s,
                    "episode_workers": self._episode_workers,
                },
            }
        )
        agent = AgentConfig(**agent_fields)
        config = self._job_template.model_copy(
            update={
                # Deterministic (NOT uuid-suffixed): rescoring the same candidate resumes its
                # completed trials through harbor's native trial resume.
                "job_name": f"wmh-{doc.doc_hash[:12]}",
                "n_attempts": self._attempts,
                # source names the dataset a task came from, but candidate jobs carry no
                # datasets: harbor's Job._refresh_metrics_for_eval indexes its metrics
                # defaultdict by source, creating an empty entry for the unknown name that
                # its display hook later crashes on with IndexError. Adhoc tasks (source
                # None) use the always-present "adhoc" metrics entry.
                "tasks": [
                    task.model_copy(deep=True, update={"source": None}) for task in self._tasks
                ],
                "agents": [agent],
            },
            deep=True,
        )
        # model_copy(update=) skips validation: re-validate the exact config harbor will run.
        return _revalidated_job_config(config)

    def _project(self, doc: HarnessDoc, run: HarborRun) -> ScoreReport:
        result = run.result
        if result.finished_at is None:
            raise ValueError("harbor job did not finish")
        expected = len(self._task_ids) * self._attempts
        if len(result.trial_results) != expected:
            raise ValueError(
                f"harbor returned {len(result.trial_results)} trials; expected {expected}"
            )
        grouped: defaultdict[str, list[TrialResult]] = defaultdict(list)
        for trial in result.trial_results:
            task_id = trial.task_id.get_name()
            if task_id not in self._task_ids:
                raise ValueError(f"harbor returned an unexpected task {task_id!r}")
            if trial.finished_at is None:
                raise ValueError(f"harbor trial {trial.trial_name!r} did not finish")
            grouped[task_id].append(trial)
        wrong_counts = {
            task_id: len(grouped[task_id])
            for task_id in self._task_ids
            if len(grouped[task_id]) != self._attempts
        }
        if wrong_counts:
            raise ValueError(f"harbor task matrix is incomplete: counts={wrong_counts}")

        max_turns = doc.max_turns()
        cells: list[ScoreCell] = []
        for task_id in self._task_ids:
            trials = sorted(grouped[task_id], key=lambda trial: trial.trial_name)
            for attempt, trial in enumerate(trials, 1):
                reward = _official_reward(trial, reward_key=self._reward_key)
                trial_dir = run.job_dir / trial.trial_name
                telemetry = _trial_telemetry(trial, trial_dir, max_turns=max_turns)
                cells.append(
                    ScoreCell(
                        task_id=task_id,
                        attempt=attempt,
                        reward=reward,
                        passed=reward_passed(reward, self._reward_mode),
                        artifact_dir=str(trial_dir),
                        note=_trial_note(trial),
                        **telemetry,
                    )
                )
        return ScoreReport(
            doc_hash=doc.doc_hash,
            request=self.request,
            reward_mode=self._reward_mode,
            cells=tuple(cells),
        )


def _revalidated_job_config(config: JobConfig) -> JobConfig:
    """Re-run JobConfig validation without serializing env-bearing sections.

    model_copy(update=) skips validation, and a model_dump round-trip is NOT safe here:
    harbor's EnvironmentConfig/VerifierConfig env field serializers templatize and redact
    sensitive-named values on every dump (harbor.utils.env.templatize_sensitive_env) with no
    disabling context, so dumping would silently corrupt a literal secret whose value differs
    from os.environ. Reconstructing from field values re-runs every JobConfig validator while
    nested models pass through by reference, never through their serializers.
    """
    return JobConfig(**{name: getattr(config, name) for name in JobConfig.model_fields})


def _validate_job_template(job_template: JobConfig) -> None:
    """Reject template shapes the scorer would otherwise have to silently rewrite."""
    if len(job_template.agents) != 1:
        raise ValueError("HarborScorer requires exactly one agent template")
    template = job_template.agents[0]
    if any(
        (
            template.name not in (None, AgentName.ORACLE.value),
            template.import_path is not None,
            template.model_name is not None,
            bool(template.skills),
            bool(template.env),
            bool(template.mcp_servers),
            bool(template.kwargs),
        )
    ):
        raise ValueError(
            "HarborScorer owns agent identity, model, skills, environment, and kwargs; "
            "leave the template agent unset"
        )
    if job_template.install_only:
        raise ValueError("HarborScorer cannot use an install-only harbor job")
    if job_template.verifier.disable:
        raise ValueError("HarborScorer requires harbor verification")
    if job_template.retry != RetryConfig():
        raise ValueError(
            "HarborScorer owns the retry policy; configure retries through harbor_retries"
        )


def _route_task_environment(job_template: JobConfig, task_environment: TaskEnvironment) -> object:
    """Validate the template/backend combination; rewrite type -> import_path only for E2B.

    Conflicts are REJECTED, never rewritten: a docker template with a requested e2b task
    environment (or vice versa) means the caller's config and intent disagree.
    """
    environment = job_template.environment
    if task_environment not in ("docker", "e2b"):
        raise ValueError("task_environment must be docker or e2b")
    if environment.import_path is not None:
        if environment.type is not None:
            raise ValueError("harbor environment cannot set both type and import_path")
        if (
            task_environment == "e2b"
            and environment.import_path == WMH_HARBOR_E2B_ENVIRONMENT_IMPORT_PATH
        ):
            return environment.model_copy(deep=True)
        raise ValueError(
            "HarborScorer owns task-environment routing; set environment.type and pass "
            "task_environment instead of import_path"
        )
    if task_environment == "docker":
        if environment.type is not EnvironmentType.DOCKER:
            raise ValueError(
                f"job template declares a {environment.type} task environment but "
                "task_environment='docker' was requested; make them agree"
            )
        return environment.model_copy(deep=True)
    if environment.type is not EnvironmentType.E2B:
        raise ValueError(
            f"job template declares a {environment.type} task environment but "
            "task_environment='e2b' was requested; make them agree"
        )
    # The consistent combination: route harbor's built-in E2B type through WMH's paced
    # subclass (qualified aliases, single-submit builds, create pacing). Options survive.
    return environment.model_copy(
        update={"type": None, "import_path": WMH_HARBOR_E2B_ENVIRONMENT_IMPORT_PATH},
        deep=True,
    )


def _assert_job_dir_resumable(job_dir: Path, config: JobConfig) -> None:
    """Refuse to touch a job dir whose recorded config differs from the one about to run.

    Harbor itself refuses to resume such a dir (FileExistsError), so pruning it first would
    only destroy transcripts of a run that can never be resumed by this config anyway.
    JobConfig equality ignores job_name/debug, matching harbor's own resume check.
    """
    config_path = job_dir / "config.json"
    if not config_path.exists():
        return
    try:
        existing = JobConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(
            f"candidate job dir {job_dir} has an unreadable config.json; refusing to prune or "
            "resume it. Move or delete the directory to rerun this candidate"
        ) from error
    if existing != config:
        raise ValueError(
            f"candidate job dir {job_dir} was produced by a different job config; harbor would "
            "refuse to resume it. Move or delete the directory to rerun this candidate"
        )


def _prune_invalid_trial_dirs(job_dir: Path, *, reward_key: str) -> int:
    """Delete trial dirs harbor's resume would either crash on or wrongly keep.

    Harbor keeps any trial whose result.json parses; a trial that died with exception_info and
    no verifier reward would therefore be "kept" as an unscoreable cell forever. Pruning it (and
    unreadable ones) makes harbor re-run exactly those trials: cheap trial-level resume of a
    crashed or interrupted boundary.
    """
    if not job_dir.is_dir():
        return 0
    pruned = 0
    for trial_dir in sorted(job_dir.iterdir()):
        if not trial_dir.is_dir():
            continue
        if _trial_dir_is_scoreable(trial_dir, reward_key=reward_key):
            continue
        shutil.rmtree(trial_dir)
        pruned += 1
    return pruned


def _trial_dir_is_scoreable(trial_dir: Path, *, reward_key: str) -> bool:
    result_path = TrialPaths(trial_dir).result_path
    try:
        trial = TrialResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if trial.exception_info is None:
        return True
    if trial.exception_info.exception_type in _VERIFIER_OUTCOME_EXCEPTIONS:
        # Scored 0 by _official_reward; re-running a deterministic verifier failure would
        # loop forever without changing the outcome.
        return True
    return _trial_reward(trial, reward_key=reward_key) is not None


def _trial_reward(trial: TrialResult, *, reward_key: str) -> float | None:
    verifier = trial.verifier_result
    rewards = None if verifier is None else verifier.rewards
    if rewards is None or reward_key not in rewards:
        return None
    value = rewards[reward_key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    reward = float(value)
    if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
        return None
    return reward


# Verifier failures harbor itself classifies as outcome-shaped (its default retry exclude
# list): the verifier ran against the candidate's artifacts and terminally failed to produce
# a reward. Scoring these 0 matches the benchmark's own semantics (an absent reward file is a
# failed task) and keeps a deterministic verifier timeout from wedging the boundary in a
# raise -> prune -> identical re-run loop.
_VERIFIER_OUTCOME_EXCEPTIONS = frozenset(
    {
        "VerifierTimeoutError",
        "RewardFileNotFoundError",
        "RewardFileEmptyError",
        "VerifierOutputParseError",
    }
)


def _official_reward(trial: TrialResult, *, reward_key: str) -> float:
    """The verifier's written reward; absence without an outcome-shaped cause is infra."""
    verifier = trial.verifier_result
    rewards = None if verifier is None else verifier.rewards
    if rewards is None or reward_key not in rewards:
        exception = trial.exception_info
        if exception is not None and exception.exception_type in _VERIFIER_OUTCOME_EXCEPTIONS:
            return 0.0
        available = sorted(rewards or {})
        raise HarborRewardMissingError(
            f"harbor trial {trial.trial_name!r} has no verifier reward {reward_key!r} "
            f"(available reward keys: {available or 'none'}); either the verifier never "
            "produced evidence or reward_key is misconfigured for this task set"
        )
    value = rewards[reward_key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"harbor reward {reward_key!r} must be numeric")
    reward = float(value)
    if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
        raise ValueError(f"harbor reward {reward_key!r} must be finite and in [0, 1]")
    return reward


def _trial_note(trial: TrialResult) -> str:
    exception = trial.exception_info
    return "completed" if exception is None else f"completed with {exception.exception_type}"


# Stop-reason markers a WMH transcript writes when the episode was cut short by a timeout: the
# harbor agent-timeout cancellation marker and the max-turns sentinel are both interesting to the
# proposer. See wmh/evals/harbor/agent.py (_CANCELLED_STOP_REASON) and StopReason.
_TIMEOUT_STOP_REASON_SUBSTRINGS = ("timeout", "cancelled")
# Exception types that mean the WMH episode itself timed out (harbor's agent timeout), distinct
# from the verifier-outcome exceptions scored 0 above.
_TIMEOUT_EXCEPTIONS = frozenset({"AgentTimeoutError", "TimeoutError"})


class _TrialTelemetry(TypedDict):
    """The optional per-episode telemetry fields projected onto one ScoreCell."""

    turns: int | None
    calls: int | None
    input_tokens: int | None
    output_tokens: int | None
    stop_reason: str
    hit_turn_cap: bool
    hit_timeout: bool


def _trial_telemetry(trial: TrialResult, trial_dir: Path, *, max_turns: int) -> _TrialTelemetry:
    """Best-effort per-episode token telemetry from the trial's WMH transcript.

    Non-fatal by contract: a missing, absent, or unparseable wmh-run.json (or one without
    worker_usage) leaves every field defaulted and never raises, so a broken transcript cannot
    break scoring. Reading it is the only new I/O in the projection, and it is cheap relative to
    the harbor job that produced it.
    """
    telemetry: _TrialTelemetry = {
        "turns": None,
        "calls": None,
        "input_tokens": None,
        "output_tokens": None,
        "stop_reason": "",
        "hit_turn_cap": False,
        "hit_timeout": False,
    }
    transcript_path = trial_dir / TRIAL_AGENT_DIR / WMH_RUN_FILENAME
    try:
        raw = json.loads(transcript_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # No transcript, or unreadable/invalid JSON: leave defaults and mark a timeout from the
        # trial exception alone, so a cancelled episode with a lost transcript still reports it.
        telemetry["hit_timeout"] = _exception_is_timeout(trial)
        return telemetry
    if not isinstance(raw, dict):
        telemetry["hit_timeout"] = _exception_is_timeout(trial)
        return telemetry

    turns = _as_optional_int(raw.get("turns"))
    telemetry["turns"] = turns
    stop_reason = raw.get("stop_reason")
    telemetry["stop_reason"] = stop_reason if isinstance(stop_reason, str) else ""
    usage = raw.get("worker_usage")
    if isinstance(usage, dict):
        telemetry["calls"] = _as_optional_int(usage.get("calls"))
        telemetry["input_tokens"] = _as_optional_int(usage.get("input_tokens"))
        telemetry["output_tokens"] = _as_optional_int(usage.get("output_tokens"))
    telemetry["hit_turn_cap"] = turns is not None and turns >= max_turns
    telemetry["hit_timeout"] = _stop_reason_is_timeout(telemetry["stop_reason"]) or (
        _exception_is_timeout(trial)
    )
    return telemetry


def _as_optional_int(value: object) -> int | None:
    """Coerce a transcript field to a nonnegative int, or None when it is not one."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _stop_reason_is_timeout(stop_reason: str) -> bool:
    lowered = stop_reason.lower()
    return any(marker in lowered for marker in _TIMEOUT_STOP_REASON_SUBSTRINGS)


def _exception_is_timeout(trial: TrialResult) -> bool:
    exception = trial.exception_info
    return exception is not None and exception.exception_type in _TIMEOUT_EXCEPTIONS
