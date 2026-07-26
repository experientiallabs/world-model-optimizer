"""Offline tests for the distillation orchestrator against the tinker fakes.

The whole loop runs without any SDK or network: the injected service client
wraps `wmh.distill.fake_tinker` (converting the loop's `TrainDatum`s into
`FakeDatum`s so the fake training client's TITO assertion sees every batch),
`collect_rollouts` is monkeypatched with a collector that samples real spans
from the CURRENT fake sampler weights, and `build_renderer` is monkeypatched
with a stub since no cookbook renderer exists for the fake base model.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NoReturn, cast, get_args

import pytest
from llm_waterfall.types import ChatMessage, ChatTool

import wmh.distill.loop as loop_module
import wmh.providers.tinker as providers_tinker

if TYPE_CHECKING:
    import tinker
from wmh.core.types import JsonObject
from wmh.distill.config import (
    BudgetConfig,
    DistillConfig,
    EvalConfig,
    GateConfig,
    HarborConfig,
    OffPolicyConfig,
    PricingConfig,
    RolloutConfig,
    SamplingConfig,
    StudentConfig,
    TeacherConfig,
    TrainConfig,
    TripwireConfig,
    WarmupConfig,
)
from wmh.distill.data import AdvantageStats, TrainDatum, attach_advantages
from wmh.distill.deadlines import TinkerDeadlineError
from wmh.distill.fake_tinker import (
    FakeDatum,
    FakeSamplingClient,
    FakeServiceClient,
    FakeTokenizer,
    FakeTrainingClient,
)
from wmh.distill.loop import (
    _WARMUP_STREAMS,
    ADVANTAGE_LOSS_BY_MODE,
    EVAL_ROLLOUTS_DIR,
    IMPORTANCE_SAMPLING_LOSS,
    MAX_CONSECUTIVE_EMPTY_STEPS,
    PPO_LOSS,
    SDK_GRAD_NORM_METRIC_NAMES,
    SDK_LOSS_METRIC_NAMES,
    STUDENT_AFTER_EVAL,
    STUDENT_BEFORE_EVAL,
    TEACHER_BASELINE_EVAL,
    WARMUP_ROLLOUTS_DIR,
    DistillBudgetError,
    DistillDegenerationError,
    DistillEmptyBatchError,
    DistillEvalReport,
    DistillNullEvalError,
    DistillProgress,
    DistillResult,
    DistillSamplingClient,
    OptimStepOutput,
    SdkSamplingClient,
    SdkServiceClient,
    SdkTrainingClient,
    StepMetrics,
    StudentSampler,
    TaskSampler,
    TrainStepOutput,
    WarmupMetrics,
    pin_rollout_params,
    resume_command,
    run_distillation,
    sdk_metric_value,
    tinker_provider_config,
)
from wmh.distill.offpolicy import CROSS_ENTROPY_LOSS, OffPolicyMetrics
from wmh.distill.rollouts import RolloutStats, rollout_stats
from wmh.distill.samples import SampleRollout
from wmh.distill.store import AdapterStore, DistillRunStore
from wmh.distill.teacher import EncodingTokenizer
from wmh.distill.tokens import TrialRecord
from wmh.distill.tracking import DistillTracker
from wmh.harness.doc import HarnessDoc
from wmh.harness.runtime import StopReason
from wmh.harness.scoring import GradedTests
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.providers.tinker import SampledSequenceLike, TokenSpan

_NAME = "distill-loop-test"
_TRAIN_IDS = ("task-a", "task-b", "task-c", "task-d")
_HOLDOUT_IDS = ("hold-a", "hold-b")
_STUDENT = "fake/student-4b"
_TEACHER = "fake/teacher-70b"


# -- fakes and shims -------------------------------------------------------------------------


class _StubRendering:
    """Just enough `ChatRendering` for the loop's preflight prompt."""

    @property
    def stop_sequences(self) -> list[str]:
        return []

    def build_generation_prompt(
        self, messages: list[ChatMessage], tools: list[ChatTool] | None = None
    ) -> list[int]:
        del tools
        text = messages[-1].content
        return [ord(ch) for ch in (text if isinstance(text, str) else "ping")]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token) for token in token_ids)

    def decode_with_specials(self, token_ids: list[int]) -> str:
        return "".join(chr(token) for token in token_ids)

    def parse_response(self, sampled_ids: list[int]) -> None:
        raise NotImplementedError("the loop never parses responses during preflight")


def _fake_build_renderer(base_model: str, tokenizer: EncodingTokenizer) -> _StubRendering:
    del base_model, tokenizer
    return _StubRendering()


def _number(row: JsonObject, key: str) -> float:
    """Read one numeric metrics-row value with the narrowing ty needs."""
    value = row[key]
    assert isinstance(value, int | float) and not isinstance(value, bool), (key, value)
    return float(value)


def _fake_datum(datum: TrainDatum) -> FakeDatum:
    """Convert a loop datum to the fake client's shifted next-token layout.

    Mirrors `to_tinker_sft_datums` for topk-CE replicas: their attached
    candidate targets and renormalized weights ride in place of the
    next-token targets and the loss mask, and the fake datum is flagged so
    the fake training client applies the input-side TITO check.
    """
    tokens = datum.model_input_tokens
    targets = datum.target_tokens if datum.is_topk_replica else tokens
    weights = datum.target_weights if datum.is_topk_replica else datum.loss_mask
    return FakeDatum(
        model_input_tokens=tokens[:-1],
        target_tokens=targets[1:],
        weights=weights[1:],
        advantages=datum.advantages[1:],
        logprobs=datum.sampled_logprobs[1:],
        topk=datum.is_topk_replica,
    )


class _Training:
    """`DistillTrainingClient` shim converting loop datums to `FakeDatum`s.

    Args:
        inner: The fake client doing the work; its `calls` list is the
            per-client ordered call log.
        service: The owning `_Service`. Its `events` log is appended to on
            `load_state` so tests can assert nothing (a sampling client, a
            second training client) came between creation and the restore, and
            its cross-session `crash_on_cross_entropy` counter is what lets a
            test interrupt a cross_entropy phase mid-schedule.
        wedge_load_state: Mirror the live pathology: `load_state` reaches the
            service (initializing the model) and then blows its deadline
            client-side, so a retry on THIS client can only be rejected.
    """

    def __init__(
        self,
        inner: FakeTrainingClient,
        service: _Service,
        *,
        wedge_load_state: bool = False,
    ) -> None:
        self.inner = inner
        self.load_state_calls: list[str] = []
        self._service = service
        self._events = service.events
        self._wedge_load_state = wedge_load_state

    def get_tokenizer(self) -> FakeTokenizer:
        return self.inner.get_tokenizer()

    def forward_backward(self, datums: Sequence[TrainDatum], loss_fn: str) -> TrainStepOutput:
        if loss_fn == CROSS_ENTROPY_LOSS:
            self._service.cross_entropy_calls += 1
            if self._service.crash_on_cross_entropy == self._service.cross_entropy_calls:
                # Raised BEFORE the work, so the crashed pass trains nothing.
                raise RuntimeError("injected cross_entropy crash")
        # Mirror SdkTrainingClient: extract the loss from the SDK-shaped
        # output's metrics dict through the same suffix-tolerant helper.
        output = self.inner.forward_backward([_fake_datum(datum) for datum in datums], loss_fn)
        return TrainStepOutput(loss=sdk_metric_value(output.metrics, SDK_LOSS_METRIC_NAMES))

    def optim_step(self, learning_rate: float) -> OptimStepOutput:
        response = self.inner.optim_step(learning_rate)
        return OptimStepOutput(
            grad_norm=sdk_metric_value(response.metrics, SDK_GRAD_NORM_METRIC_NAMES)
        )

    def save_state(self) -> str:
        return self.inner.save_state()

    def load_state(self, path: str) -> None:
        self.load_state_calls.append(path)
        self._events.append("load_state")
        self.inner.load_state(path)
        if self._wedge_load_state:
            raise TinkerDeadlineError("load_state", elapsed_s=600.0, deadline_s=600.0)

    def save_weights_for_sampler(self, name: str) -> str:
        return self.inner.save_weights_for_sampler(name)


class _Service:
    """`DistillServiceClient` over the fakes.

    Every call builds a FRESH training client, exactly like the real service
    (one new server-side model per session): saved states and registered
    sampler paths live on the shared `FakeServiceClient`, so the artifacts a
    prior session wrote outlive its client the way real tinker:// paths do,
    and a resumed session's client starts uninitialized (the only state in
    which tinker accepts `load_state`).
    """

    def __init__(self) -> None:
        self.inner = FakeServiceClient()
        self.trainings: list[_Training] = []
        self.events: list[str] = []
        """Ordered log of client creations and restores across every session."""

        self.wedged_load_state_clients = 0
        """The first N training clients blow their load_state deadline."""

        self.cross_entropy_calls = 0
        """Cross_entropy forward_backward calls across every session's client."""

        self.crash_on_cross_entropy: int | None = None
        """1-based index of the cross_entropy pass that raises instead of training."""

    def create_lora_training_client(self, base_model: str, rank: int = 32) -> _Training:
        training = _Training(
            self.inner.create_lora_training_client(base_model, rank),
            self,
            wedge_load_state=len(self.trainings) < self.wedged_load_state_clients,
        )
        self.events.append("create_training_client")
        self.trainings.append(training)
        return training

    @property
    def training(self) -> _Training | None:
        """The most recently created training client (None before the first)."""
        return self.trainings[-1] if self.trainings else None

    def create_sampling_client(self, model_path: str) -> DistillSamplingClient:
        self.events.append("create_sampling_client")
        return self.inner.create_sampling_client(model_path)


@dataclass(frozen=True)
class _RolloutCall:
    """One recorded `collect_rollouts` invocation."""

    step_index: int
    task_ids: tuple[str, ...]
    attempts: int
    provider_model: str
    run_dir: Path
    doc_hash: str


class _FakeRollouts:
    """Offline `collect_rollouts`: samples real spans from the current weights.

    Every trial makes two prefix-extending sampling calls through the service
    client for the provider's model path, so student trials carry spans the
    fake training client's ledger knows about (the TITO ground truth). Tasks
    at even positions pass, so every batch's solve rate is 0.5.
    """

    def __init__(self, service: _Service) -> None:
        self.service = service
        self.calls: list[_RolloutCall] = []
        self.fabricate_spans = False
        self.teacher_fail_all = False
        """When True, every trial of the teacher provider fails (warmup skip path)."""

        self.fail_on_train_step: int | None = None
        """Raise on the TRAIN batch of this step (a crash between phases)."""

        self.empty_span_train_steps: set[int] = set()
        """TRAIN steps whose every trial records zero spans (a dead provider)."""

        self.zero_trial_train_steps: set[int] = set()
        """TRAIN steps whose batch returns no trials at all (harbor scored nothing)."""

        self.stop_reason = StopReason.SUBMITTED.value
        """The stop reason every trial records; anything else is a scaffold loss."""

        self.infra_fail_all = False
        """When True, every trial is an infra failure (the E2B sandbox-cap outage)."""

        self.omit_test_reports = False
        """When True, no trial carries a CTRF test breakdown (a verifier that graded but wrote no
        readable report), so every graded rate must report a null measurement, not a 0.0."""

        self.tokens_per_call = 5
        """Tokens each of an episode's two sampling calls generates by default."""

        self.tokens_per_call_by_train_step: dict[int, int] = {}
        """Per-train-step override of `tokens_per_call`: a length collapse.

        Episode length is exactly 2 x this (no stop sequences), so a step set to
        2 pools to 0.4x the default batch's 10 tokens/episode (the tripwire's
        warn band) and a step set to 1 pools to 0.2x (its kill band)."""

        self.entropy_by_train_step: dict[int, float] = {}
        """Per-train-step override of every sampled logprob's magnitude.

        The fake sampler's own logprobs are hash-derived (about 2.05 nats
        pooled); pinning them makes an entropy RATIO exact. Only the recorded
        span logprobs change, never the token ids, so the fake training client's
        TITO assertion still sees the sampler's real tokens."""

    def __call__(
        self,
        step_index: int,
        task_ids: Sequence[str],
        cfg: DistillConfig,
        harness: HarnessDoc,
        provider_config: ProviderConfig,
        run_dir: Path,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> tuple[list[TrialRecord], RolloutStats]:
        del should_cancel
        is_train_batch = (
            run_dir.parent.name != "eval-rollouts" and run_dir.name != WARMUP_ROLLOUTS_DIR
        )
        if is_train_batch and step_index == self.fail_on_train_step:
            raise RuntimeError("injected rollout crash")
        # A dead student provider: every trial ends span-less ("submitted"
        # with zero turns), exactly what swallowed worker failures produce.
        empty_batch = is_train_batch and step_index in self.empty_span_train_steps
        if is_train_batch and step_index in self.zero_trial_train_steps:
            self.calls.append(
                _RolloutCall(
                    step_index=step_index,
                    task_ids=tuple(task_ids),
                    attempts=cfg.train.group_size,
                    provider_model=provider_config.model,
                    run_dir=run_dir,
                    doc_hash=harness.doc_hash,
                )
            )
            return [], rollout_stats([], max_tokens=4096)
        client = self.service.create_sampling_client(provider_config.model)
        records: list[TrialRecord] = []
        for task_index, task_id in enumerate(task_ids):
            for attempt in range(1, cfg.train.group_size + 1):
                passed = task_index % 2 == 0 and not empty_batch
                if self.teacher_fail_all and provider_config.model == _TEACHER:
                    passed = False
                trial_name = f"{task_id}__s{attempt}"
                spans = (
                    []
                    if empty_batch
                    else self._spans(
                        client, task_id, step_index, attempt, is_train_batch=is_train_batch
                    )
                )
                records.append(
                    TrialRecord(
                        task_id=task_id,
                        attempt=attempt,
                        trial_name=trial_name,
                        reward=1.0 if passed else 0.0,
                        passed=passed,
                        spans=spans,
                        stop_reason=None if self.infra_fail_all else self.stop_reason,
                        infra_failed=self.infra_fail_all,
                        # Two tests per task, the probe's own most common shape: a failing trial
                        # still passes one of them, so the graded rate is strictly above the binary
                        # one. An infra failure has no verifier evidence, so it has no report.
                        tests=(
                            None
                            if self.infra_fail_all or self.omit_test_reports
                            else GradedTests(passed=2 if passed else 1, resolved=2)
                        ),
                        artifact_dir=str(
                            run_dir / "harbor" / f"step-{step_index:04d}" / trial_name
                        ),
                    )
                )
        # The real aggregator, so the fake cannot drift from the stats the loop reads.
        stats = rollout_stats(records, max_tokens=4096)
        self.calls.append(
            _RolloutCall(
                step_index=step_index,
                task_ids=tuple(task_ids),
                attempts=cfg.train.group_size,
                provider_model=provider_config.model,
                run_dir=run_dir,
                doc_hash=harness.doc_hash,
            )
        )
        return records, stats

    def _spans(
        self,
        client: DistillSamplingClient,
        task_id: str,
        step_index: int,
        attempt: int,
        *,
        is_train_batch: bool = True,
    ) -> list[TokenSpan]:
        if self.fabricate_spans:
            # Token ids no sampler ever issued: a TITO violation by construction.
            return [
                TokenSpan(
                    call_index=0,
                    prompt_token_ids=[65, 66, 67],
                    sampled_token_ids=[1, 2, 3],
                    sampled_logprobs=[-0.1, -0.2, -0.3],
                )
            ]
        max_tokens = self.tokens_per_call
        entropy: float | None = None
        if is_train_batch:
            max_tokens = self.tokens_per_call_by_train_step.get(step_index, max_tokens)
            entropy = self.entropy_by_train_step.get(step_index)
        prompt = [ord(ch) for ch in f"{task_id}:{step_index}:{attempt}:"]
        first = client.sample(prompt, max_tokens=max_tokens, temperature=0.7)
        assert first.logprobs is not None
        follow = [*prompt, *first.tokens, *(ord(ch) for ch in "|obs|")]
        second = client.sample(follow, max_tokens=max_tokens, temperature=0.7)
        assert second.logprobs is not None

        def logprobs(issued: Sequence[float]) -> list[float]:
            return [-entropy] * len(issued) if entropy is not None else list(issued)

        return [
            TokenSpan(
                call_index=0,
                prompt_token_ids=prompt,
                sampled_token_ids=list(first.tokens),
                sampled_logprobs=logprobs(first.logprobs),
            ),
            TokenSpan(
                call_index=1,
                prompt_token_ids=follow,
                sampled_token_ids=list(second.tokens),
                sampled_logprobs=logprobs(second.logprobs),
            ),
        ]


@dataclass
class _Env:
    """One test's wired-up offline environment."""

    service: _Service
    rollouts: _FakeRollouts
    run_dir: Path
    adapters: AdapterStore
    progress: list[DistillProgress] = field(default_factory=list)


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Env:
    service = _Service()
    rollouts = _FakeRollouts(service)
    monkeypatch.setattr(loop_module, "collect_rollouts", rollouts)
    monkeypatch.setattr(loop_module, "build_renderer", _fake_build_renderer)
    return _Env(
        service=service,
        rollouts=rollouts,
        run_dir=tmp_path / "run",
        adapters=AdapterStore(tmp_path / ".wmh"),
    )


def _cfg(*, budget_max: float | None = None, pricing: PricingConfig | None = None) -> DistillConfig:
    return DistillConfig(
        student=StudentConfig(base_model=_STUDENT, lora_rank=8),
        teacher=TeacherConfig(model=_TEACHER),
        harbor=HarborConfig(job_template="unused-by-the-stubbed-collector.yaml"),
        train=TrainConfig(
            steps=3,
            tasks_per_batch=2,
            group_size=2,
            learning_rate=1e-4,
            sampler_refresh_every=1,
            save_state_every=2,
            trial_concurrency=2,
        ),
        eval=EvalConfig(every=2, tasks=2, k=1),
        gate=GateConfig(k=1),
        pricing=pricing
        if pricing is not None
        else PricingConfig(
            student_prefill=1.0, student_sample=1.0, student_train=1.0, teacher_prefill=1.0
        ),
        budget=BudgetConfig(max_usd=budget_max),
    )


def _train_priced_cfg(budget_max: float) -> DistillConfig:
    """A config whose spend lands ONLY on the student_train meter.

    Baselines never charge student_train, so the run deterministically
    survives the baselines and hits the cap at the first training step's
    budget check.
    """
    return _cfg(
        budget_max=budget_max,
        pricing=PricingConfig(
            student_prefill=0.0, student_sample=0.0, student_train=1e9, teacher_prefill=0.0
        ),
    )


def _warmup_cfg(
    *,
    warmup_steps: int = 2,
    rollouts_per_task: int = 2,
    keep: Literal["passed", "all"] = "passed",
    warmup_lr: float | None = 5e-5,
    budget_max: float | None = None,
    pricing: PricingConfig | None = None,
    trajectories_from: str | None = None,
) -> DistillConfig:
    """The 3-step config with the supervised warmup phase enabled."""
    return _cfg(budget_max=budget_max, pricing=pricing).model_copy(
        update={
            "warmup": WarmupConfig(
                steps=warmup_steps,
                rollouts_per_task=rollouts_per_task,
                keep=keep,
                learning_rate=warmup_lr,
                trajectories_from=trajectories_from,
            )
        }
    )


def _offpolicy_cfg(
    *,
    epochs: int = 2,
    minibatch_datums: int = 0,
    rollouts_per_task: int = 2,
    keep: Literal["passed", "all"] = "passed",
    learning_rate: float | None = 5e-5,
    shuffle_seed: int | None = None,
    checkpoint_every: int = 1,
    budget_max: float | None = None,
    pricing: PricingConfig | None = None,
    trajectories_from: str | None = None,
) -> DistillConfig:
    """The 3-step config with off-policy distillation enabled."""
    return _cfg(budget_max=budget_max, pricing=pricing).model_copy(
        update={
            "offpolicy": OffPolicyConfig(
                epochs=epochs,
                minibatch_datums=minibatch_datums,
                rollouts_per_task=rollouts_per_task,
                keep=keep,
                learning_rate=learning_rate,
                shuffle_seed=shuffle_seed,
                checkpoint_every=checkpoint_every,
                trajectories_from=trajectories_from,
            )
        }
    )


def _offpolicy_rows(run_dir: Path) -> list[JsonObject]:
    rows = DistillRunStore(run_dir).read_metrics()
    return [row for row in rows if row.get("phase") == "offpolicy"]


def _warmup_calls(env: _Env) -> list[_RolloutCall]:
    warmup_dir = env.run_dir / WARMUP_ROLLOUTS_DIR
    return [call for call in env.rollouts.calls if call.run_dir == warmup_dir]


def _eval_run_counts(env: _Env) -> dict[str, int]:
    """How many collector batches each eval key actually ran (reuse skips runs)."""
    eval_root = env.run_dir / EVAL_ROLLOUTS_DIR
    counts: dict[str, int] = {}
    for call in env.rollouts.calls:
        if call.run_dir.parent == eval_root:
            counts[call.run_dir.name] = counts.get(call.run_dir.name, 0) + 1
    return counts


def _loss_fns(env: _Env) -> list[str]:
    """Every trained batch's loss across every session's training client, in order."""
    assert env.service.trainings
    return [
        loss
        for training in env.service.trainings
        for _, loss in training.inner.forward_backward_calls
    ]


def _run(
    env: _Env,
    cfg: DistillConfig,
    *,
    resume: bool = False,
    tracker: DistillTracker | None = None,
    cli_agent: str | None = None,
) -> DistillResult:
    return run_distillation(
        _NAME,
        cfg,
        HarnessDoc.baseline(),
        _TRAIN_IDS,
        _HOLDOUT_IDS,
        env.run_dir,
        resume=resume,
        on_progress=env.progress.append,
        service_client=env.service,
        adapter_store=env.adapters,
        tracker=tracker,
        cli_agent=cli_agent,
    )


@dataclass(frozen=True)
class _TrackedSummary:
    """One recorded `log_summary` call."""

    gate_accepted: bool
    gate_reason: str
    teacher_solve_rate: float
    student_before_solve_rate: float
    student_after_solve_rate: float
    total_usd: float
    steps_completed: int


class _RecordingTracker:
    """A `DistillTracker` that records every call for assertions."""

    def __init__(self) -> None:
        self.steps: list[tuple[int, StepMetrics]] = []
        self.offpolicy_steps: list[tuple[int, OffPolicyMetrics]] = []
        self.warmup_steps: list[tuple[int, WarmupMetrics]] = []
        self.evals: list[tuple[str, float, int | None]] = []
        self.graded_evals: list[tuple[str, float | None]] = []
        """Every `log_eval`'s graded companion (None when the batch measured none)."""

        self.samples: list[tuple[str, int | None, list[SampleRollout]]] = []
        self.summaries: list[_TrackedSummary] = []
        self.finish_calls = 0

    def log_step(self, step: int, metrics: StepMetrics) -> None:
        self.steps.append((step, metrics))

    def log_offpolicy_step(self, offpolicy_step: int, metrics: OffPolicyMetrics) -> None:
        self.offpolicy_steps.append((offpolicy_step, metrics))

    def log_warmup_step(self, warmup_step: int, metrics: WarmupMetrics) -> None:
        self.warmup_steps.append((warmup_step, metrics))

    def log_eval(
        self,
        name: str,
        solve_rate: float,
        step: int | None,
        *,
        graded_solve_rate: float | None = None,
    ) -> None:
        self.evals.append((name, solve_rate, step))
        self.graded_evals.append((name, graded_solve_rate))

    def log_samples(self, kind: str, step: int | None, samples: list[SampleRollout]) -> None:
        self.samples.append((kind, step, list(samples)))

    def log_summary(
        self,
        *,
        gate_accepted: bool,
        gate_reason: str,
        teacher_solve_rate: float,
        student_before_solve_rate: float,
        student_after_solve_rate: float,
        total_usd: float,
        steps_completed: int,
    ) -> None:
        self.summaries.append(
            _TrackedSummary(
                gate_accepted=gate_accepted,
                gate_reason=gate_reason,
                teacher_solve_rate=teacher_solve_rate,
                student_before_solve_rate=student_before_solve_rate,
                student_after_solve_rate=student_after_solve_rate,
                total_usd=total_usd,
                steps_completed=steps_completed,
            )
        )

    def finish(self) -> None:
        self.finish_calls += 1


# -- the 3-step end-to-end run ---------------------------------------------------------------


def test_three_step_run_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = _setup(tmp_path, monkeypatch)

    result = _run(env, _cfg())

    assert result.steps_completed == 3
    assert result.gate.accepted
    assert result.adapter_version == 1
    assert result.final_sampler_path.startswith("tinker://fake/sampler/")
    assert result.run_dir == str(env.run_dir)
    assert result.spend.total_usd > 0.0

    # Metrics rows: one per step, carrying the loss/health/meter numbers.
    store = DistillRunStore(env.run_dir)
    rows = store.read_metrics()
    assert [row["step"] for row in rows] == [0, 1, 2]
    for row in rows:
        assert row["solve_rate"] == 0.5
        # The graded companion beside it: every failing trial still passed 1 of its 2 tests, which
        # binary cannot see at all. Its own denominator rides along.
        assert row["graded_solve_rate"] == 0.75
        assert row["graded_trials"] == 4
        assert row["fragmentation_rate"] == 0.0
        assert row["datums"] == 4  # 2 tasks x 2 attempts, each merged to one datum
        assert isinstance(row["reverse_kl_per_token"], float)
        # RL metrics: rewards are binary here, so the mean equals solve_rate;
        # the fake backend reports a deterministic loss and grad norm.
        assert row["reward_mean"] == 0.5
        assert isinstance(row["advantage_mean"], float)
        assert _number(row, "advantage_std") >= 0.0
        assert 0.0 <= _number(row, "clip_fraction") <= 1.0
        assert isinstance(row["pg_loss"], float) and row["pg_loss"] > 0.0
        assert isinstance(row["grad_norm"], float) and row["grad_norm"] > 0.0
        for key in (
            "usd",
            "student_prefill_tokens",
            "student_cached_prefill_tokens",
            "student_sample_tokens",
            "student_train_tokens",
            "teacher_prefill_tokens",
        ):
            assert _number(row, key) > 0
    # Teacher-in-harness billing (teacher_sample plus cached teacher prefill)
    # happens only in the pre-step teacher baseline, which folds into row 0;
    # training steps score the teacher in one full-price request per datum.
    assert _number(rows[0], "teacher_sample_tokens") > 0
    assert _number(rows[0], "teacher_cached_prefill_tokens") > 0
    for row in rows[1:]:
        assert _number(row, "teacher_sample_tokens") == 0
        assert _number(row, "teacher_cached_prefill_tokens") == 0
    # Cumulative spend: the first row folds in the pre-step baseline spend
    # exactly, every later row advances by exactly its own delta, and the
    # finalize eval charges after the last row (so the ledger total is higher).
    assert _number(rows[0], "cumulative_usd") == pytest.approx(_number(rows[0], "usd"))
    for previous, row in zip(rows, rows[1:], strict=False):
        assert _number(row, "cumulative_usd") == pytest.approx(
            _number(previous, "cumulative_usd") + _number(row, "usd")
        )
    assert _number(rows[-1], "cumulative_usd") < result.spend.total_usd

    # TITO held through every forward_backward: the fake training client
    # asserts spans against its ledger BEFORE recording a call, so three
    # recorded importance_sampling batches mean three passing TITO checks.
    training = env.service.training
    assert training is not None
    batches = training.inner.forward_backward_calls
    assert len(batches) == 3
    assert all(loss_fn == "importance_sampling" for _, loss_fn in batches)
    assert all(len(batch) == 4 for batch, _ in batches)
    assert training.inner.optim_step_lrs == [1e-4] * 3

    # Cadences: refresh_every=1 gives every training step its own sampler path.
    train_calls = [call for call in env.rollouts.calls if call.run_dir == env.run_dir]
    assert [call.step_index for call in train_calls] == [0, 1, 2]
    assert all(call.attempts == 2 for call in train_calls)
    assert len({call.provider_model for call in train_calls}) == 3
    # save_state_every=2 checkpoints after step 1; finalize checkpoints step 2.
    assert [checkpoint.step for checkpoint in store.checkpoints()] == [1, 2]
    latest = store.latest_checkpoint()
    assert latest is not None
    assert latest.sampler_path == result.final_sampler_path

    # Evals: two holdout baselines, one interim train-subsample eval after
    # step 1 (eval.every=2), and the final holdout student-after eval.
    eval_calls = [call for call in env.rollouts.calls if call.run_dir != env.run_dir]
    assert [call.run_dir.name for call in eval_calls] == [
        "baseline-teacher",
        "baseline-student-before",
        "step-0001",
        "student-after",
    ]
    teacher_call, before_call, interim_call, after_call = eval_calls
    assert teacher_call.provider_model == _TEACHER
    assert teacher_call.task_ids == _HOLDOUT_IDS
    assert teacher_call.attempts == 1  # gate.k
    assert before_call.provider_model == train_calls[0].provider_model
    assert len(interim_call.task_ids) == 2
    assert set(interim_call.task_ids) <= set(_TRAIN_IDS)
    assert after_call.provider_model == result.final_sampler_path
    assert after_call.task_ids == _HOLDOUT_IDS
    for key in ("baseline-teacher", "baseline-student-before", "step-0001", "student-after"):
        assert (store.evals_dir / f"{key}.json").is_file()

    # Terminal artifacts: config snapshot, gate, model card, handoff, adapter.
    assert store.config_path.is_file()
    gate = json.loads(store.gate_path.read_text(encoding="utf-8"))
    assert gate["accepted"] is True
    assert gate["teacher_solve_rate"] == 0.5
    assert result.final_sampler_path in store.handoff_path.read_text(encoding="utf-8")
    assert env.adapters.versions(_NAME) == [1]
    assert env.adapters.aliases(_NAME) == {"champion": 1}
    card = env.adapters.resolve(_NAME)
    assert card.base_model == _STUDENT
    assert card.teacher_model == _TEACHER
    assert card.sampler_path == result.final_sampler_path
    assert card.state_path == result.final_state_path
    assert card.gate is not None and card.gate.accepted

    phases = {event.phase for event in env.progress}
    assert {"preflight", "baseline", "rollouts", "training", "eval", "finalize", "gate"} <= phases

    # Every rollout batch (train and eval) ran the SAME pinned document: the
    # seed doc with [sampling]/[rollout] written into its param surfaces.
    pinned_hash = pin_rollout_params(HarnessDoc.baseline(), _cfg()).doc_hash
    assert {call.doc_hash for call in env.rollouts.calls} == {pinned_hash}

    # The spend ledger tracked every charge, including the finalize eval that
    # no metrics row ever carries.
    assert store.read_spend() == pytest.approx(result.spend.total_usd)


def test_eval_reports_record_the_graded_rate_beside_the_binary_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every persisted eval report carries the graded rate and its own denominator.

    The holdout is what a TerminalBench-2 comparison is read off, so the report on disk (not just
    the dashboard) has to carry the resolution: binary 0.5 here, graded 0.75, over the same trials.
    """
    env = _setup(tmp_path, monkeypatch)

    _run(env, _cfg())

    store = DistillRunStore(env.run_dir)
    for key in ("baseline-teacher", "baseline-student-before", "step-0001", "student-after"):
        report = DistillEvalReport.model_validate_json(
            (store.evals_dir / f"{key}.json").read_text(encoding="utf-8")
        )
        assert report.solve_rate == 0.5, key
        assert report.graded_solve_rate == 0.75, key
        # Every gradeable trial had a test report here, so the denominators agree.
        assert report.graded_trials == report.executed_trials == report.trials, key


def test_a_run_whose_verifiers_write_no_test_report_records_no_graded_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Graded-absent must stay distinguishable from graded-zero all the way out to the artifacts.

    Every trial here is graded by the verifier (real binary rewards) but leaves no readable CTRF
    report, so `graded_trials` is 0 everywhere, nothing is charted, and the placeholder 0.0 is only
    ever readable next to that zero denominator.
    """
    env = _setup(tmp_path, monkeypatch)
    env.rollouts.omit_test_reports = True
    tracker = _RecordingTracker()

    _run(env, _cfg(), tracker=tracker)

    store = DistillRunStore(env.run_dir)
    for row in store.read_metrics():
        assert row["solve_rate"] == 0.5  # the binary rate is unaffected
        assert row["graded_trials"] == 0
        assert row["graded_solve_rate"] == 0.0
    report = DistillEvalReport.model_validate_json(
        (store.evals_dir / "student-after.json").read_text(encoding="utf-8")
    )
    assert (report.solve_rate, report.graded_trials, report.graded_solve_rate) == (0.5, 0, 0.0)
    assert all(graded is None for _name, graded in tracker.graded_evals)
    # The operator's line says it in words rather than printing "graded 0.000".
    training = [event.message for event in env.progress if event.phase == "training"]
    assert training
    assert all("no graded score (no readable test report)" in message for message in training)


def test_a_baseline_imported_from_a_run_without_graded_scores_charts_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An older run's report has `graded_trials == 0`, which is absent, not 0.0.

    Charting its placeholder 0.0 would put a fabricated graded rate beside a real binary one.
    """
    env = _setup(tmp_path, monkeypatch)
    teacher_src = _write_prior_baseline(
        tmp_path / "prior" / "evals" / "baseline-teacher.json",
        name=TEACHER_BASELINE_EVAL,
        provider_model=_TEACHER,
        base_model=_TEACHER,
    )
    tracker = _RecordingTracker()

    _run(env, _reuse_cfg(teacher_from=teacher_src), tracker=tracker)

    graded = dict(tracker.graded_evals)
    assert graded["baseline-teacher"] is None  # imported, predates the metric
    assert graded["student-after"] == 0.75  # measured here
    copied = DistillEvalReport.model_validate_json(
        (DistillRunStore(env.run_dir).evals_dir / "baseline-teacher.json").read_text(
            encoding="utf-8"
        )
    )
    assert (copied.graded_solve_rate, copied.graded_trials) == (0.0, 0)


def _token_totals(datums: Sequence[TrainDatum]) -> tuple[int, int]:
    """A datum list's (loss, context) token totals."""
    loss = sum(datum.loss_token_count for datum in datums)
    return loss, sum(len(datum.model_input_tokens) for datum in datums) - loss


class _DroppingTeacherRows:
    """`attach_advantages` under a teacher that misaligns one row per step.

    Blanks the first datum's teacher logprobs, which is exactly what a
    wrong-length compute_logprobs response looks like by the time it reaches
    the datum builder, then delegates to the REAL `attach_advantages` so the
    drop and its post-drop stats are the production ones. Both the pre-drop
    and the trained token totals are recorded per step for the assertions.
    """

    def __init__(self) -> None:
        self.built: list[tuple[int, int]] = []
        """Per step, the (loss, context) totals of the batch the teacher scored."""

        self.trained: list[tuple[int, int]] = []
        """Per step, the (loss, context) totals of the batch that trained."""

    def __call__(
        self,
        datums: Sequence[TrainDatum],
        teacher_logprobs: Sequence[Sequence[float | None]],
        cfg: DistillConfig,
    ) -> tuple[list[TrainDatum], AdvantageStats]:
        rows: list[Sequence[float | None]] = [
            [] if index == 0 else row for index, row in enumerate(teacher_logprobs)
        ]
        trained, stats = attach_advantages(datums, rows, cfg)
        self.built.append(_token_totals(datums))
        self.trained.append(_token_totals(trained))
        return trained, stats


def test_metrics_row_describes_the_post_teacher_drop_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-step token counts are the TRAINED batch's, never the pre-drop one.

    Regression for a row that mixed the two accounting stages: `loss_tokens`
    and `context_tokens` came from `build_datums` (before the teacher's
    misaligned rows were dropped) while `clipped_tokens` and `clip_fraction`
    came from `attach_advantages` (after), so the step read bigger than it
    trained and `clipped_tokens / loss_tokens` disagreed with the reported
    `clip_fraction`.
    """
    env = _setup(tmp_path, monkeypatch)
    dropping = _DroppingTeacherRows()
    monkeypatch.setattr(loop_module, "attach_advantages", dropping)
    base = _cfg()
    # A clip bound this tight clips every trained loss token, pinning
    # clip_fraction at exactly 1.0: only the post-drop denominator reproduces
    # it, so the ratio check below is a real check.
    cfg = base.model_copy(update={"train": base.train.model_copy(update={"advantage_clip": 1e-9})})

    _run(env, cfg)

    rows = [row for row in DistillRunStore(env.run_dir).read_metrics() if "phase" not in row]
    assert len(rows) == 3
    assert len(dropping.trained) == len(rows)
    for row, trained, built in zip(rows, dropping.trained, dropping.built, strict=True):
        loss_tokens, context_tokens = trained
        # The teacher really dropped a datum, and it really carried tokens.
        assert row["mismatch_drops"] == 1
        assert row["datums"] == 3  # 4 built, 1 dropped
        assert built[0] > loss_tokens and built[1] > context_tokens
        # The row reports the batch that trained, not the batch that was scored.
        assert _number(row, "loss_tokens") == loss_tokens
        assert _number(row, "context_tokens") == context_tokens
        assert _number(row, "student_train_tokens") == loss_tokens + context_tokens
        # ... so the clip counters share the row's own denominator.
        assert _number(row, "clipped_tokens") == loss_tokens
        assert _number(row, "clip_fraction") == 1.0
        assert _number(row, "clipped_tokens") / _number(row, "loss_tokens") == pytest.approx(
            _number(row, "clip_fraction")
        )
        # The build-stage counters keep describing the build stage.
        assert _number(row, "overflow_drops") == 0
        assert _number(row, "overlong_drops") == 0


def test_tracker_sees_every_step_eval_summary_and_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    tracker = _RecordingTracker()

    result = _run(env, _cfg(), tracker=tracker)

    # One log_step per training step, with the SAME rows the store persisted.
    assert [step for step, _ in tracker.steps] == [0, 1, 2]
    persisted = DistillRunStore(env.run_dir).read_metrics()
    for (step, metrics), row in zip(tracker.steps, persisted, strict=True):
        assert row == {"step": step, **metrics.model_dump(mode="json")}

    # Every recorded eval report was tracked under its store key: the two
    # pre-training baselines (no step), the interim eval after step 1
    # (eval.every=2), and the finalize student-after eval.
    assert tracker.evals == [
        ("baseline-teacher", 0.5, None),
        ("baseline-student-before", 0.5, None),
        ("step-0001", 0.5, 1),
        ("student-after", 0.5, 2),
    ]
    # Each one also carries its graded companion, so the dashboard charts both series.
    assert tracker.graded_evals == [
        ("baseline-teacher", 0.75),
        ("baseline-student-before", 0.75),
        ("step-0001", 0.75),
        ("student-after", 0.75),
    ]

    (summary,) = tracker.summaries
    assert summary.gate_accepted is True
    assert summary.gate_reason == result.gate.reason
    assert summary.teacher_solve_rate == 0.5
    assert summary.student_before_solve_rate == 0.5
    assert summary.student_after_solve_rate == 0.5
    assert summary.total_usd == pytest.approx(result.spend.total_usd)
    assert summary.steps_completed == 3

    assert tracker.finish_calls == 1


def test_sample_rollouts_land_in_files_and_the_tracker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every batch kind writes samples/<name>.md and reaches log_samples."""
    env = _setup(tmp_path, monkeypatch)
    tracker = _RecordingTracker()

    _run(env, _warmup_cfg(), tracker=tracker)

    # One samples file per batch: the two holdout baselines, the warmup
    # collection, the three training steps, the interim eval after step 1
    # (eval.every=2), and the finalize student-after eval.
    samples_dir = DistillRunStore(env.run_dir).samples_dir
    assert {path.name for path in samples_dir.iterdir()} == {
        "eval-baseline-teacher.md",
        "eval-baseline-student-before.md",
        "warmup.md",
        "step-0000.md",
        "step-0001.md",
        "eval-step-0001.md",
        "step-0002.md",
        "eval-student-after.md",
    }
    # The default N=2 renders the first two span-bearing trials, each with
    # its outcome header and the decoded episode body.
    step0 = (samples_dir / "step-0000.md").read_text(encoding="utf-8")
    assert step0.count("### trial ") == 2
    assert "reward:" in step0
    assert "stop reason: submitted" in step0
    assert "episode tokens:" in step0

    # The tracker saw the same batches, in run order, under their kinds.
    assert [(kind, step) for kind, step, _ in tracker.samples] == [
        ("eval-baseline-teacher", None),
        ("eval-baseline-student-before", None),
        ("warmup", None),
        ("train", 0),
        ("train", 1),
        ("eval-step-0001", 1),
        ("train", 2),
        ("eval-student-after", 2),
    ]
    for _, _, samples in tracker.samples:
        assert len(samples) == 2
        for sample in samples:
            assert sample.text.startswith(f"### trial {sample.trial_name}\n")


def test_log_sample_rollouts_zero_disables_files_and_tracker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    tracker = _RecordingTracker()
    cfg = _cfg()
    cfg = cfg.model_copy(update={"train": cfg.train.model_copy(update={"log_sample_rollouts": 0})})

    _run(env, cfg, tracker=tracker)

    assert not DistillRunStore(env.run_dir).samples_dir.exists()
    assert tracker.samples == []


def test_tracker_finish_fires_on_the_budget_abort_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    tracker = _RecordingTracker()

    with pytest.raises(DistillBudgetError):
        _run(env, _train_priced_cfg(budget_max=1.0), tracker=tracker)

    # Step 0 completed (its row was tracked before the abort), the gate was
    # never reached, and the tracker was still finished.
    assert [step for step, _ in tracker.steps] == [0]
    assert [name for name, _, _ in tracker.evals] == [
        "baseline-teacher",
        "baseline-student-before",
    ]
    assert tracker.summaries == []
    assert tracker.finish_calls == 1


def test_rollout_params_are_pinned_from_the_config(
    caplog: pytest.LogCaptureFixture,
) -> None:
    doc = HarnessDoc.baseline()
    cfg = _cfg().model_copy(
        update={
            "rollout": RolloutConfig(max_turns=7),
            "sampling": SamplingConfig(temperature=0.5, max_tokens=256),
        }
    )

    with caplog.at_level(logging.WARNING, logger="wmh.distill.loop"):
        pinned = pin_rollout_params(doc, cfg)

    assert pinned.max_turns() == 7
    assert pinned.temperature() == pytest.approx(0.5)
    assert pinned.max_output_tokens() == 256
    # Off-1.0 temperatures bias the reverse-KL advantages; the pinning warns.
    assert "temperature" in caplog.text and "1.0" in caplog.text
    # Pure function of (doc, cfg): repeated pinning yields the identical identity.
    assert pin_rollout_params(doc, cfg).doc_hash == pinned.doc_hash
    assert pinned.doc_hash != doc.doc_hash

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="wmh.distill.loop"):
        neutral = pin_rollout_params(doc, _cfg())  # default temperature 1.0
    assert neutral.temperature() == pytest.approx(1.0)
    assert "temperature" not in caplog.text


def test_a_tito_violation_fails_the_forward_backward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fabricated spans (ids no sampler issued) must die in the TITO assertion."""
    env = _setup(tmp_path, monkeypatch)
    env.rollouts.fabricate_spans = True
    with pytest.raises(AssertionError, match="TITO violation"):
        _run(env, _cfg())


def test_teacher_in_harness_episodes_charge_the_teacher_sample_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The warmup collection and the teacher baseline bill sampling, not prefill.

    Every other price is zero (the cached rates derive 20% of zero), so the
    whole run's spend is exactly the teacher-in-harness sampled volume at the
    teacher_sample rate.
    """
    env = _setup(tmp_path, monkeypatch)
    pricing = PricingConfig(
        student_prefill=0.0,
        student_sample=0.0,
        student_train=0.0,
        teacher_prefill=0.0,
        teacher_sample=1e6,  # $1 per token, so USD equals the token count
    )

    result = _run(env, _warmup_cfg(pricing=pricing))

    lines = {line.meter: line for line in result.spend.lines}
    # Teacher-in-harness trials: the gate baseline (2 holdout tasks x gate.k=1)
    # plus the warmup collection (4 train tasks x 2 attempts); the stub
    # collector samples 2 calls x 5 tokens per trial.
    expected_sampled = (2 * 1 + 4 * 2) * 2 * 5
    assert lines["teacher_sample"].tokens == expected_sampled
    assert result.spend.total_usd == pytest.approx(float(expected_sampled))


# -- baseline reuse across runs ----------------------------------------------------------------


def _write_prior_baseline(
    path: Path,
    *,
    name: str,
    provider_model: str,
    base_model: str | None,
    task_ids: Sequence[str] = _HOLDOUT_IDS,
    attempts: int = 1,
    solve_rate: float = 0.5,
) -> Path:
    """Fabricate a prior run's recorded baseline eval report on disk."""
    report = DistillEvalReport(
        name=name,
        provider_model=provider_model,
        base_model=base_model,
        task_ids=list(task_ids),
        attempts=attempts,
        trials=attempts * len(task_ids),
        solve_rate=solve_rate,
        empty_span_trials=0,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


def _reuse_cfg(
    *, teacher_from: Path | None = None, student_from: Path | None = None
) -> DistillConfig:
    """The 3-step config with baseline reuse configured and interim evals off."""
    return _cfg().model_copy(
        update={
            "eval": EvalConfig(
                every=0,
                tasks=2,
                k=1,
                teacher_baseline_from=None if teacher_from is None else str(teacher_from),
                student_baseline_from=None if student_from is None else str(student_from),
            )
        }
    )


def test_baseline_reuse_skips_the_baseline_trials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both baselines load from prior reports, so no trials run for either eval."""
    env = _setup(tmp_path, monkeypatch)
    teacher_src = _write_prior_baseline(
        tmp_path / "prior" / "evals" / "baseline-teacher.json",
        name=TEACHER_BASELINE_EVAL,
        provider_model=_TEACHER,
        base_model=_TEACHER,
        solve_rate=0.5,
    )
    student_src = _write_prior_baseline(
        tmp_path / "prior" / "evals" / "baseline-student-before.json",
        name=STUDENT_BEFORE_EVAL,
        provider_model="tinker://fake/sampler/prior-run-step-0000",
        base_model=_STUDENT,
        solve_rate=0.25,
    )
    tracker = _RecordingTracker()

    result = _run(
        env, _reuse_cfg(teacher_from=teacher_src, student_from=student_src), tracker=tracker
    )

    # The rollout collector never ran for either baseline: the only eval batch
    # is the finalize student-after, and the teacher never sampled at all.
    eval_calls = [call for call in env.rollouts.calls if call.run_dir != env.run_dir]
    assert [call.run_dir.name for call in eval_calls] == ["student-after"]
    assert all(call.provider_model != _TEACHER for call in env.rollouts.calls)

    # The gate consumed the reused solve rates (teacher 0.5, before 0.25;
    # the stub collector's student-after solves 0.5).
    assert result.gate.teacher_solve_rate == pytest.approx(0.5)
    assert result.gate.student_before_solve_rate == pytest.approx(0.25)
    assert result.gate.accepted

    # Logged and tracked as usual (no step, like any pre-training baseline).
    assert ("baseline-teacher", 0.5, None) in tracker.evals
    assert ("baseline-student-before", 0.25, None) in tracker.evals


def test_reused_baseline_is_copied_with_a_provenance_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    teacher_src = _write_prior_baseline(
        tmp_path / "prior" / "evals" / "baseline-teacher.json",
        name=TEACHER_BASELINE_EVAL,
        provider_model=_TEACHER,
        base_model=_TEACHER,
    )

    _run(env, _reuse_cfg(teacher_from=teacher_src))

    evals_dir = DistillRunStore(env.run_dir).evals_dir
    copied = DistillEvalReport.model_validate_json(
        (evals_dir / "baseline-teacher.json").read_text(encoding="utf-8")
    )
    assert copied.source is not None
    assert str(teacher_src) in copied.source
    assert "eval.teacher_baseline_from" in copied.source
    assert copied.solve_rate == pytest.approx(0.5)
    assert copied.task_ids == list(_HOLDOUT_IDS)
    # The student baseline was NOT configured for reuse: it ran trials here,
    # carries no provenance note, and records the base model a later run's
    # student_baseline_from validation needs.
    before = DistillEvalReport.model_validate_json(
        (evals_dir / "baseline-student-before.json").read_text(encoding="utf-8")
    )
    assert before.source is None
    assert before.base_model == _STUDENT


def test_baseline_reuse_rejects_a_task_id_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    src = _write_prior_baseline(
        tmp_path / "prior" / "evals" / "baseline-teacher.json",
        name=TEACHER_BASELINE_EVAL,
        provider_model=_TEACHER,
        base_model=_TEACHER,
        task_ids=("hold-a", "hold-z"),
    )

    with pytest.raises(ValueError) as excinfo:
        _run(env, _reuse_cfg(teacher_from=src))

    message = str(excinfo.value)
    assert "eval.teacher_baseline_from" in message
    assert "hold-b" in message  # missing from the report
    assert "hold-z" in message  # extra in the report
    # The invalid report was never copied into this run's evals/.
    assert not (DistillRunStore(env.run_dir).evals_dir / "baseline-teacher.json").exists()


def test_baseline_reuse_rejects_too_few_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    src = _write_prior_baseline(
        tmp_path / "prior" / "evals" / "baseline-teacher.json",
        name=TEACHER_BASELINE_EVAL,
        provider_model=_TEACHER,
        base_model=_TEACHER,
        attempts=1,
    )
    cfg = _reuse_cfg(teacher_from=src).model_copy(update={"gate": GateConfig(k=2)})

    with pytest.raises(ValueError) as excinfo:
        _run(env, cfg)

    message = str(excinfo.value)
    assert "eval.teacher_baseline_from" in message
    assert "1 attempt(s)" in message
    assert "gate.k is 2" in message


def test_student_baseline_reuse_rejects_a_base_model_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    src = _write_prior_baseline(
        tmp_path / "prior" / "evals" / "baseline-student-before.json",
        name=STUDENT_BEFORE_EVAL,
        provider_model="tinker://fake/sampler/prior-run-step-0000",
        base_model="fake/other-student",
    )

    with pytest.raises(ValueError) as excinfo:
        _run(env, _reuse_cfg(student_from=src))

    message = str(excinfo.value)
    assert "eval.student_baseline_from" in message
    assert "fake/other-student" in message
    assert _STUDENT in message


def test_student_baseline_reuse_requires_a_recorded_base_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A student report without base_model cannot be validated, so it is rejected."""
    env = _setup(tmp_path, monkeypatch)
    src = _write_prior_baseline(
        tmp_path / "prior" / "evals" / "baseline-student-before.json",
        name=STUDENT_BEFORE_EVAL,
        provider_model="tinker://fake/sampler/prior-run-step-0000",
        base_model=None,
    )

    with pytest.raises(ValueError, match="records no base_model"):
        _run(env, _reuse_cfg(student_from=src))


# -- budget abort and resume -----------------------------------------------------------------


def test_budget_abort_persists_state_and_the_resume_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)

    with pytest.raises(DistillBudgetError) as excinfo:
        _run(env, _train_priced_cfg(budget_max=1.0))

    error = excinfo.value
    expected_command = resume_command(_NAME, env.run_dir)
    assert error.resume_command == expected_command
    assert expected_command in str(error)
    assert "budget.max_usd" in str(error)
    assert error.max_usd == 1.0
    assert error.spent_usd > 1.0

    # Step 0 completed (its metrics row exists) and its state was checkpointed.
    store = DistillRunStore(env.run_dir)
    assert [row["step"] for row in store.read_metrics()] == [0]
    latest = store.latest_checkpoint()
    assert latest is not None
    assert latest.step == 0
    # The checkpointed state is real: a fresh training client can restore it
    # (the only client the live service would accept a restore on).
    env.service.create_lora_training_client(_STUDENT).load_state(latest.state_path)
    # The run never reached the gate.
    assert not store.gate_path.exists()


def test_resume_command_prints_the_agent_string_as_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run seeded from a stored version is invoked as 'name@ref'; the printed
    resume command must carry that exact string or the CLI's resume conflict
    check rejects the one command the abort message tells the user to run."""
    env = _setup(tmp_path, monkeypatch)

    with pytest.raises(DistillBudgetError) as excinfo:
        _run(env, _train_priced_cfg(budget_max=1.0), cli_agent=f"{_NAME}@v3")

    expected = resume_command(f"{_NAME}@v3", env.run_dir)
    assert excinfo.value.resume_command == expected
    assert expected in str(excinfo.value)


def test_resume_rejects_a_changed_gate_k(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Baselines recorded at one k must not gate a student-after measured at
    another; the resume refuses instead of comparing different estimators."""
    env = _setup(tmp_path, monkeypatch)
    with pytest.raises(DistillBudgetError):
        _run(env, _train_priced_cfg(budget_max=1.0))

    changed = _cfg().model_copy(update={"gate": GateConfig(k=2)})
    with pytest.raises(RuntimeError, match="measured at k=1"):
        _run(env, changed, resume=True)


def test_resume_rejects_a_changed_teacher_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The teacher baseline is reusable precisely because the teacher identity
    is stable; swapping the teacher mid-run would gate against a stale rate."""
    env = _setup(tmp_path, monkeypatch)
    with pytest.raises(DistillBudgetError):
        _run(env, _train_priced_cfg(budget_max=1.0))

    changed = _cfg().model_copy(update={"teacher": TeacherConfig(model="other/teacher")})
    with pytest.raises(RuntimeError, match="mid-run model swap"):
        _run(env, changed, resume=True)


def test_zero_trial_steps_count_toward_the_empty_streak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batch that scores no trials at all is at least as dead as an all-empty
    one; it must extend the dead-provider streak, not reset it."""
    env = _setup(tmp_path, monkeypatch)
    env.rollouts.zero_trial_train_steps = {0}
    env.rollouts.empty_span_train_steps = {1}

    with pytest.raises(DistillEmptyBatchError) as excinfo:
        _run(env, _cfg())

    assert excinfo.value.consecutive_steps == MAX_CONSECUTIVE_EMPTY_STEPS


def test_two_consecutive_all_empty_steps_abort_with_the_resume_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dead-provider guard: all-empty steps must not burn the whole budget."""
    env = _setup(tmp_path, monkeypatch)
    env.rollouts.empty_span_train_steps = {0, 1}

    with pytest.raises(DistillEmptyBatchError) as excinfo:
        _run(env, _cfg())

    error = excinfo.value
    assert error.consecutive_steps == MAX_CONSECUTIVE_EMPTY_STEPS
    expected_command = resume_command(_NAME, env.run_dir)
    assert error.resume_command == expected_command
    assert expected_command in str(error)
    assert "no completions" in str(error)
    assert "runner logs" in str(error)
    # Both plausible causes are named, including the E2B concurrency cap that reads as a
    # model bug when trials silently die at sandbox creation.
    assert "wmh e2b reap" in str(error)
    # Exactly two empty train batches ran: the abort fired at the second,
    # before a third batch could spend more on span-less rollouts.
    train_steps = [call.step_index for call in env.rollouts.calls if call.run_dir == env.run_dir]
    assert train_steps == [0, 1]
    # Artifacts persisted: both steps' metrics rows and a resumable checkpoint.
    store = DistillRunStore(env.run_dir)
    rows = store.read_metrics()
    assert [row["step"] for row in rows] == [0, 1]
    assert all(_number(row, "datums") == 0 for row in rows)
    assert all(_number(row, "empty_span_trials") == _number(row, "trials") for row in rows)
    latest = store.latest_checkpoint()
    assert latest is not None
    assert latest.step == 1


def test_a_non_empty_step_resets_the_empty_batch_streak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    env.rollouts.empty_span_train_steps = {0, 2}  # the healthy step 1 sits between

    result = _run(env, _cfg())

    # Two non-consecutive empty steps never abort: the run finishes all steps.
    assert result.steps_completed == 3
    store = DistillRunStore(env.run_dir)
    rows = store.read_metrics()
    assert [row["step"] for row in rows] == [0, 1, 2]
    assert _number(rows[0], "datums") == 0
    assert _number(rows[1], "datums") > 0
    assert _number(rows[2], "datums") == 0
    # A step that trained nothing surfaces no backend or advantage metrics
    # (absent, never fabricated); the healthy step carries them all.
    for row in (rows[0], rows[2]):
        assert row["pg_loss"] is None
        assert row["grad_norm"] is None
        assert row["advantage_mean"] is None
        assert row["advantage_std"] is None
        assert _number(row, "clip_fraction") == 0.0
        assert _number(row, "reward_mean") == 0.0  # trials ran; every one failed
    assert isinstance(rows[1]["pg_loss"], float)
    assert isinstance(rows[1]["grad_norm"], float)
    assert isinstance(rows[1]["advantage_mean"], float)


def test_resume_continues_the_step_count_and_reuses_baselines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    with pytest.raises(DistillBudgetError):
        _run(env, _train_priced_cfg(budget_max=1.0))
    baseline_calls = [
        call for call in env.rollouts.calls if call.run_dir.name.startswith("baseline")
    ]
    assert len(baseline_calls) == 2

    # Resume with the budget lifted (the documented recovery path).
    result = _run(env, _cfg(), resume=True)

    assert result.steps_completed == 3
    assert result.gate.accepted
    store = DistillRunStore(env.run_dir)
    assert [row["step"] for row in store.read_metrics()] == [0, 1, 2]
    training = env.service.training
    assert training is not None
    latest_before_resume = store.checkpoints()[0]
    assert training.load_state_calls == [latest_before_resume.state_path]
    # Baselines were reused from the recorded eval payloads, not re-run.
    baseline_calls = [
        call for call in env.rollouts.calls if call.run_dir.name.startswith("baseline")
    ]
    assert len(baseline_calls) == 2
    # Training resumed at step 1: across both sessions each step ran exactly once.
    train_steps = [call.step_index for call in env.rollouts.calls if call.run_dir == env.run_dir]
    assert train_steps == [0, 1, 2]
    # Prior-session spend was restored from the spend ledger, and the resumed
    # session's rows carry cumulative totals that INCLUDE it.
    assert result.spend.prior_usd > 0.0
    for row in store.read_metrics()[1:]:
        assert _number(row, "cumulative_usd") > result.spend.prior_usd


def test_resume_loads_state_as_the_first_call_on_a_fresh_training_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tinker accepts LoadWeights only on an uninitialized model.

    Anything on the training client before the restore (a sampler-weights
    save, a forward_backward, a sampling client derived from it) makes the
    resume impossible: the live service answers "LoadWeights can only be
    called on uninitialized models" and the run dies having lost every
    trained step. Preflight therefore runs AFTER the restore, which is also
    more correct: it validates the resumed weights, not the bare base model.
    """
    env = _setup(tmp_path, monkeypatch)
    with pytest.raises(DistillBudgetError):
        _run(env, _train_priced_cfg(budget_max=1.0))
    store = DistillRunStore(env.run_dir)
    latest = store.latest_checkpoint()
    assert latest is not None
    clients_before = len(env.service.trainings)
    events_before = len(env.service.events)
    progress_before = len(env.progress)

    result = _run(env, _cfg(), resume=True)

    assert result.steps_completed == 3
    resumed = env.service.trainings[clients_before]
    assert resumed.load_state_calls == [latest.state_path]
    # The restore is the client's FIRST call, and the sampler refresh (the
    # first weights save, which would have closed the door) comes after it.
    assert resumed.inner.calls[:2] == ["load_state", "save_weights_for_sampler"]
    # Nothing went through the SERVICE first either: neither the teacher nor
    # the student sampling client was created before the restore.
    assert env.service.events[events_before : events_before + 2] == [
        "create_training_client",
        "load_state",
    ]
    # Preflight still gated the resumed session before any spend, and it ran
    # on the restored weights (after load_state, per the call log above).
    phases = [progress.phase for progress in env.progress[progress_before:]]
    assert phases[0] == "preflight"
    assert phases.index("preflight") < phases.index("training")


def test_resume_retries_a_timed_out_load_state_on_a_fresh_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A restore abandoned at its deadline may have initialized the model.

    That is what killed three live resumes: the adapter re-issued load_state
    on the same client and the service rejected it. The retry must open a
    fresh, still-uninitialized client instead.
    """
    env = _setup(tmp_path, monkeypatch)
    with pytest.raises(DistillBudgetError):
        _run(env, _train_priced_cfg(budget_max=1.0))
    store = DistillRunStore(env.run_dir)
    latest = store.latest_checkpoint()
    assert latest is not None
    clients_before = len(env.service.trainings)
    env.service.wedged_load_state_clients = clients_before + 1  # only the next one wedges

    result = _run(env, _cfg(), resume=True)

    assert result.steps_completed == 3
    wedged, retried = env.service.trainings[clients_before : clients_before + 2]
    assert wedged.load_state_calls == [latest.state_path]
    assert retried.load_state_calls == [latest.state_path]
    assert retried.inner.calls[0] == "load_state"
    # The wedged client was abandoned, not trained on.
    assert wedged.inner.forward_backward_calls == []
    assert retried.inner.forward_backward_calls != []


def test_fresh_run_never_loads_state_and_preflights_before_the_first_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fresh path is unchanged: no restore at all, preflight before spend."""
    env = _setup(tmp_path, monkeypatch)

    result = _run(env, _cfg())

    assert result.steps_completed == 3
    training = env.service.training
    assert training is not None
    assert training.load_state_calls == []
    assert "load_state" not in training.inner.calls
    assert "load_state" not in env.service.events
    assert env.service.events[0] == "create_training_client"
    # Preflight came first: its progress events precede every baseline and
    # training event, and its tokenizer fetch precedes the first trained batch.
    phases = [progress.phase for progress in env.progress]
    assert phases[0] == "preflight"
    assert phases.index("preflight") < phases.index("baseline") < phases.index("training")
    calls = training.inner.calls
    assert calls.index("get_tokenizer") < calls.index("forward_backward")


def test_resume_restores_spend_charged_between_metrics_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 'cannot spend the budget twice' regression (spend ledger).

    A budget abort during the holdout baselines leaves NO metrics row (rows
    land only when a training step completes), so a resume that derived prior
    spend from metrics.jsonl would restore $0 and happily spend budget.max_usd
    all over again. The ledger is written on every charge, so the resumed
    meter must equal the pre-abort meter exactly.
    """
    env = _setup(tmp_path, monkeypatch)
    cfg = _cfg(budget_max=1e-6)  # the teacher baseline alone exceeds the cap

    with pytest.raises(DistillBudgetError) as first:
        _run(env, cfg)

    store = DistillRunStore(env.run_dir)
    assert store.read_metrics() == []  # nothing for a metrics-derived resume to see
    assert first.value.spent_usd > 0.0
    assert store.read_spend() == pytest.approx(first.value.spent_usd)
    calls_after_abort = len(env.rollouts.calls)

    # Resuming with the SAME cap must abort immediately at the restored meter
    # (prior spend + nothing new), not run a fresh budget.max_usd worth.
    with pytest.raises(DistillBudgetError) as second:
        _run(env, cfg, resume=True)
    assert second.value.spent_usd == pytest.approx(first.value.spent_usd)
    assert len(env.rollouts.calls) == calls_after_abort  # no new spend before the abort

    # The documented recovery (raise the cap, resume) carries the prior spend
    # forward into the final accounting.
    result = _run(env, _cfg(), resume=True)
    assert result.spend.prior_usd == pytest.approx(first.value.spent_usd)
    assert result.spend.total_usd == pytest.approx(first.value.spent_usd + result.spend.session_usd)
    assert store.read_spend() == pytest.approx(result.spend.total_usd)


def test_fresh_run_into_a_used_run_dir_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    _run(env, _cfg())
    with pytest.raises(ValueError, match="resume=True"):
        _run(env, _cfg())


def test_resume_with_an_empty_run_dir_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="nothing to resume"):
        _run(env, _cfg(), resume=True)


# -- the degeneration tripwires (entropy and generation length) --------------------------------


def _tripwire_cfg(*, steps: int = 4, tripwire: TripwireConfig | None = None) -> DistillConfig:
    """A config for the tripwire tests: no interim evals, N training steps.

    Each step's batch is 2 tasks x 2 attempts = 4 episodes of 2 calls x 5
    tokens, so the healthy pooled baseline is exactly 10 sampled tokens per
    episode and `_FakeRollouts.tokens_per_call_by_train_step` sets exact ratios
    against it.
    """
    base = _cfg()
    return base.model_copy(
        update={
            "train": base.train.model_copy(update={"steps": steps}),
            "eval": EvalConfig(every=0, tasks=2, k=1),
            "tripwire": tripwire if tripwire is not None else TripwireConfig(),
        }
    )


def _rows_by_step(env: _Env) -> list[JsonObject]:
    return [row for row in DistillRunStore(env.run_dir).read_metrics() if "phase" not in row]


def test_the_first_training_step_arms_the_baseline_once_and_persists_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The baseline is measured at step 0, written to the run manifest, and never
    re-measured: every later row is judged against that one number."""
    env = _setup(tmp_path, monkeypatch)

    _run(env, _tripwire_cfg(steps=3))

    baseline = DistillRunStore(env.run_dir).read_tripwire_baseline()
    assert baseline is not None
    assert baseline.step == 0
    assert baseline.episodes == 4  # 2 tasks x 2 attempts, all span-bearing
    assert baseline.sampled_tokens == 40  # x 2 calls x 5 tokens
    assert baseline.mean_generation_tokens == pytest.approx(10.0)
    rows = _rows_by_step(env)
    assert [row["step"] for row in rows] == [0, 1, 2]
    # Row 0 IS the baseline (ratio exactly 1.0), and every later row carries the
    # same baseline rather than its own.
    assert _number(rows[0], "entropy_per_token") == pytest.approx(baseline.entropy_per_token)
    assert _number(rows[0], "entropy_ratio") == pytest.approx(1.0)
    assert _number(rows[0], "generation_tokens_ratio") == pytest.approx(1.0)
    for row in rows:
        assert _number(row, "entropy_baseline") == pytest.approx(baseline.entropy_per_token)
        assert _number(row, "generation_tokens_baseline") == pytest.approx(10.0)
        assert _number(row, "mean_generation_tokens") == pytest.approx(10.0)


def test_an_absolutely_flat_run_never_fires_because_the_baseline_is_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The false-alarm regression this whole design exists for.

    Every step here sits at 0.05 nats/token and 2 sampled tokens per episode:
    numbers the sibling lane's ABSOLUTE thresholds would kill on immediately,
    including at step 0 before a single gradient step. Because the baseline is
    the run's own first step, nothing fires and the run finishes.
    """
    env = _setup(tmp_path, monkeypatch)
    steps = 4
    env.rollouts.entropy_by_train_step = dict.fromkeys(range(steps), 0.05)
    env.rollouts.tokens_per_call_by_train_step = dict.fromkeys(range(steps), 1)

    result = _run(env, _tripwire_cfg(steps=steps))

    assert result.steps_completed == steps
    rows = _rows_by_step(env)
    for row in rows:
        assert _number(row, "entropy_per_token") == pytest.approx(0.05)
        assert _number(row, "mean_generation_tokens") == pytest.approx(2.0)
        assert _number(row, "entropy_ratio") == pytest.approx(1.0)
        assert _number(row, "generation_tokens_ratio") == pytest.approx(1.0)


def test_a_warn_level_step_warns_and_keeps_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A single warn-level step names the metric, the baseline, the value and the
    ratio, and costs the run nothing else."""
    env = _setup(tmp_path, monkeypatch)
    # 2 of 5 tokens per call: 4 sampled tokens per episode against a 10-token
    # baseline, i.e. exactly the 0.5 warn fraction for length. Entropy stays put.
    env.rollouts.tokens_per_call_by_train_step = {1: 2}

    with caplog.at_level(logging.WARNING, logger="wmh.distill.loop"):
        result = _run(env, _tripwire_cfg(steps=3))

    assert result.steps_completed == 3
    assert "degeneration tripwire at step 1" in caplog.text
    assert "mean_generation_tokens 4 is 0.40x this run's baseline 10" in caplog.text
    assert "tripwire.length_warn_frac" in caplog.text
    rows = _rows_by_step(env)
    assert _number(rows[1], "generation_tokens_ratio") == pytest.approx(0.4)
    # Warn level only: the streak counter never armed, so nothing aborted.
    assert "before the run aborts" not in caplog.text


def test_two_consecutive_kill_level_steps_abort_with_the_resume_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The kill path mirrors the budget abort: checkpoint first, then a typed
    error carrying the exact resume command."""
    env = _setup(tmp_path, monkeypatch)
    # 1 of 5 tokens per call: 2 sampled tokens per episode, 0.2x baseline, under
    # the 0.25 length kill fraction.
    env.rollouts.tokens_per_call_by_train_step = {1: 1, 2: 1}

    with pytest.raises(DistillDegenerationError) as excinfo:
        _run(env, _tripwire_cfg(steps=4))

    error = excinfo.value
    assert error.consecutive_steps == 2
    assert [breach.metric for breach in error.breaches] == ["mean_generation_tokens"]
    assert all(breach.level == "kill" for breach in error.breaches)
    expected_command = resume_command(_NAME, env.run_dir)
    assert error.resume_command == expected_command
    assert expected_command in str(error)
    assert "0.20x this run's baseline" in str(error)
    # Exactly three train batches ran: the abort fired at the second kill-level
    # step, before a fourth could spend more on a degenerate policy.
    train_steps = [call.step_index for call in env.rollouts.calls if call.run_dir == env.run_dir]
    assert train_steps == [0, 1, 2]
    # Artifacts persisted, exactly as the budget path does it.
    store = DistillRunStore(env.run_dir)
    assert [row["step"] for row in _rows_by_step(env)] == [0, 1, 2]
    latest = store.latest_checkpoint()
    assert latest is not None
    assert latest.step == 2
    env.service.create_lora_training_client(_STUDENT).load_state(latest.state_path)
    assert not store.gate_path.exists()


def test_non_consecutive_kill_level_steps_never_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One bad batch is a small task draw; the streak resets on a healthy step."""
    env = _setup(tmp_path, monkeypatch)
    env.rollouts.tokens_per_call_by_train_step = {1: 1, 3: 1}  # healthy step 2 between

    result = _run(env, _tripwire_cfg(steps=4))

    assert result.steps_completed == 4
    rows = _rows_by_step(env)
    assert _number(rows[1], "generation_tokens_ratio") == pytest.approx(0.2)
    assert _number(rows[2], "generation_tokens_ratio") == pytest.approx(1.0)
    assert _number(rows[3], "generation_tokens_ratio") == pytest.approx(0.2)


def test_an_unmeasurable_step_neither_breaches_nor_clears_the_streak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An all-empty batch is not evidence of health, so it must not launder a
    collapse; it also cannot fabricate a breach out of a zero."""
    env = _setup(tmp_path, monkeypatch)
    env.rollouts.tokens_per_call_by_train_step = {1: 1, 3: 1}
    env.rollouts.empty_span_train_steps = {2}  # measures nothing between the two kills

    with pytest.raises(DistillDegenerationError) as excinfo:
        _run(env, _tripwire_cfg(steps=5))

    assert excinfo.value.consecutive_steps == 2
    rows = _rows_by_step(env)
    assert [row["step"] for row in rows] == [0, 1, 2, 3]
    assert rows[2]["entropy_per_token"] is None
    assert rows[2]["entropy_ratio"] is None


def test_an_entropy_collapse_alone_aborts_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mode collapse at unchanged episode length: the signal a KL curve hides."""
    env = _setup(tmp_path, monkeypatch)
    env.rollouts.entropy_by_train_step = {0: 1.0, 1: 0.2, 2: 0.2}

    with pytest.raises(DistillDegenerationError) as excinfo:
        _run(env, _tripwire_cfg(steps=4))

    assert [breach.metric for breach in excinfo.value.breaches] == ["entropy_per_token"]
    rows = _rows_by_step(env)
    assert _number(rows[0], "entropy_per_token") == pytest.approx(1.0)
    assert _number(rows[1], "entropy_ratio") == pytest.approx(0.2)
    # Length never moved, so only the entropy leg fired.
    assert _number(rows[1], "generation_tokens_ratio") == pytest.approx(1.0)


def test_resume_reuses_the_recorded_baseline_instead_of_re_measuring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure mode that makes a resumed run's tripwire blind.

    Session 1 measures a healthy baseline at step 0 and aborts on budget.
    Session 2 resumes into steps that have collapsed. If it re-baselined on its
    own first step (the already-collapsed step 1), every ratio would read 1.0
    and nothing would ever fire again; reusing the recorded baseline kills the
    run instead.
    """
    env = _setup(tmp_path, monkeypatch)
    with pytest.raises(DistillBudgetError):
        _run(env, _train_priced_cfg(budget_max=1.0))

    store = DistillRunStore(env.run_dir)
    healthy = store.read_tripwire_baseline()
    assert healthy is not None
    assert healthy.step == 0
    assert healthy.mean_generation_tokens == pytest.approx(10.0)

    env.rollouts.tokens_per_call_by_train_step = {1: 1, 2: 1}
    with pytest.raises(DistillDegenerationError) as excinfo:
        _run(env, _tripwire_cfg(steps=4), resume=True)

    assert excinfo.value.consecutive_steps == 2
    # The manifest still records the step-0 baseline, untouched by the resume.
    assert store.read_tripwire_baseline() == healthy
    resumed_rows = [row for row in _rows_by_step(env) if _number(row, "step") > 0]
    assert [row["step"] for row in resumed_rows] == [1, 2]
    for row in resumed_rows:
        assert _number(row, "generation_tokens_baseline") == pytest.approx(10.0)
        assert _number(row, "generation_tokens_ratio") == pytest.approx(0.2)


def test_a_disabled_tripwire_still_measures_but_never_aborts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`enabled = false` silences the warning and the abort; the metrics, the
    baseline, and the ratios stay, because they cost nothing."""
    env = _setup(tmp_path, monkeypatch)
    env.rollouts.tokens_per_call_by_train_step = {1: 1, 2: 1}

    result = _run(env, _tripwire_cfg(steps=3, tripwire=TripwireConfig(enabled=False)))

    assert result.steps_completed == 3
    assert DistillRunStore(env.run_dir).read_tripwire_baseline() is not None
    rows = _rows_by_step(env)
    assert _number(rows[1], "generation_tokens_ratio") == pytest.approx(0.2)
    assert _number(rows[2], "generation_tokens_ratio") == pytest.approx(0.2)


def test_the_progress_line_carries_both_health_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)

    _run(env, _tripwire_cfg(steps=2))

    training = [event.message for event in env.progress if event.phase == "training"]
    assert training
    assert all("entropy/token" in message for message in training)
    assert all("gen tokens/episode 10" in message for message in training)
    assert "1.00x baseline" in training[0]
    # Both solve rates on the operator's line, each with its own denominator.
    assert all("solve rate 0.50" in message for message in training)
    assert all("graded 0.750 over 4 graded trial(s)" in message for message in training)


def test_an_all_empty_first_step_leaves_the_tripwire_unarmed_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dead batch is no baseline: step 0 measures nothing (its row carries no
    entropy at all), and step 1 arms the tripwire instead."""
    env = _setup(tmp_path, monkeypatch)
    env.rollouts.empty_span_train_steps = {0}

    _run(env, _tripwire_cfg(steps=3))

    baseline = DistillRunStore(env.run_dir).read_tripwire_baseline()
    assert baseline is not None
    assert baseline.step == 1
    rows = _rows_by_step(env)
    assert rows[0]["entropy_per_token"] is None
    assert rows[0]["entropy_baseline"] is None
    assert _number(rows[1], "entropy_ratio") == pytest.approx(1.0)


# -- off-policy distillation -------------------------------------------------------------------


def test_offpolicy_trains_the_teacher_corpus_then_opd_proceeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    tracker = _RecordingTracker()

    result = _run(env, _offpolicy_cfg(), tracker=tracker)

    assert result.steps_completed == 3
    assert result.gate.accepted

    # The corpus collection ran the TEACHER over the full train split under the
    # shared cross_entropy rollout root, rollouts_per_task attempts per task.
    (collection,) = _warmup_calls(env)
    assert collection.provider_model == _TEACHER
    assert collection.task_ids == _TRAIN_IDS
    assert collection.attempts == 2

    # Two full-batch cross_entropy epochs over the kept datums (tasks at even
    # positions pass: 2 tasks x 2 attempts = 4 kept trials, one datum each),
    # then the three importance_sampling OPD steps. The off-policy LR applies
    # to the off-policy steps only.
    training = env.service.training
    assert training is not None
    assert _loss_fns(env) == ["cross_entropy"] * 2 + ["importance_sampling"] * 3
    ce_batches = [
        batch for batch, loss in training.inner.forward_backward_calls if loss == "cross_entropy"
    ]
    assert all(len(batch) == 4 for batch in ce_batches)
    assert training.inner.optim_step_lrs == [5e-5, 5e-5, 1e-4, 1e-4, 1e-4]

    # OPD step 0 sampled the trained student: the forced post-phase refresh
    # published fresh weights, distinct from what the student-before baseline saw.
    train_calls = [call for call in env.rollouts.calls if call.run_dir == env.run_dir]
    before_call = next(
        call for call in env.rollouts.calls if call.run_dir.name == STUDENT_BEFORE_EVAL
    )
    assert "offpolicy" in train_calls[0].provider_model
    assert train_calls[0].provider_model != before_call.provider_model

    rows = _offpolicy_rows(env.run_dir)
    assert [row["step"] for row in rows] == [0, 1]
    assert [row["epoch"] for row in rows] == [0, 1]
    for row in rows:
        assert row["epochs"] == 2
        assert row["minibatch"] == 0
        assert row["planned_steps"] == 2
        assert row["trials"] == 8
        assert row["kept_trials"] == 4
        assert row["solve_rate"] == 0.5
        assert row["corpus_datums"] == 4
        assert row["datums"] == 4
        assert _number(row, "learning_rate") == pytest.approx(5e-5)
        assert _number(row, "student_train_tokens") > 0
        assert _number(row, "usd") > 0
    # The teacher collection's charge folds into the first row only.
    assert _number(rows[0], "teacher_sample_tokens") > 0
    assert _number(rows[1], "teacher_sample_tokens") == 0

    assert [step for step, _ in tracker.offpolicy_steps] == [0, 1]
    for (step, metrics), row in zip(tracker.offpolicy_steps, rows, strict=True):
        assert row == {"step": step, **metrics.model_dump(mode="json")}

    store = DistillRunStore(env.run_dir)
    record = store.read_offpolicy()
    assert record is not None
    assert record.epochs == 2
    assert record.steps == 2
    assert record.trials == 8
    assert record.kept_trials == 4
    assert record.datums == 4
    assert record.skipped_reason is None
    assert record.state_path is not None
    assert record.sampler_path == train_calls[0].provider_model
    # The terminal record supersedes the cursor, so no stale cursor is left.
    assert store.read_offpolicy_cursor() is None
    assert "offpolicy" in {event.phase for event in env.progress}


def test_offpolicy_minibatches_split_each_epoch_into_optimizer_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two datums per step over a 4-datum corpus is 2 optimizer steps an epoch."""
    env = _setup(tmp_path, monkeypatch)

    result = _run(env, _offpolicy_cfg(epochs=2, minibatch_datums=2))

    assert result.steps_completed == 3
    training = env.service.training
    assert training is not None
    assert _loss_fns(env) == ["cross_entropy"] * 4 + ["importance_sampling"] * 3
    ce_batches = [
        batch for batch, loss in training.inner.forward_backward_calls if loss == "cross_entropy"
    ]
    assert [len(batch) for batch in ce_batches] == [2, 2, 2, 2]
    assert training.inner.optim_step_lrs == [5e-5] * 4 + [1e-4] * 3

    rows = _offpolicy_rows(env.run_dir)
    assert [(row["step"], row["epoch"], row["minibatch"]) for row in rows] == [
        (0, 0, 0),
        (1, 0, 1),
        (2, 1, 0),
        (3, 1, 1),
    ]
    # Each row meters ITS minibatch, not the whole corpus.
    assert all(row["datums"] == 2 and row["corpus_datums"] == 4 for row in rows)
    record = DistillRunStore(env.run_dir).read_offpolicy()
    assert record is not None
    assert record.steps == 4


def test_offpolicy_resumes_from_its_cursor_after_an_interrupted_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted phase continues at the next minibatch, not from epoch 0.

    The whole point of the cursor: the resumed session restores the state saved
    at the last checkpointed step, reuses the teacher collection it already
    paid for, and trains exactly the steps the schedule has left.
    """
    env = _setup(tmp_path, monkeypatch)
    cfg = _offpolicy_cfg(epochs=3, minibatch_datums=1)  # 4 datums x 3 epochs = 12 steps
    env.service.crash_on_cross_entropy = 5  # dies entering step 4

    with pytest.raises(RuntimeError, match="injected cross_entropy crash"):
        _run(env, cfg)

    store = DistillRunStore(env.run_dir)
    assert store.read_offpolicy() is None  # no terminal record: the phase is unfinished
    cursor = store.read_offpolicy_cursor()
    assert cursor is not None
    assert cursor.steps_completed == 4
    assert (cursor.epoch, cursor.minibatch) == (1, 0)
    assert cursor.datums == 4
    assert [row["step"] for row in _offpolicy_rows(env.run_dir)] == [0, 1, 2, 3]
    assert _loss_fns(env).count("cross_entropy") == 4

    env.service.crash_on_cross_entropy = None
    result = _run(env, cfg, resume=True)

    assert result.steps_completed == 3
    # The resumed session restored the cursor's weights and trained only the
    # 8 remaining steps: 12 cross_entropy passes total, never 4 + 12.
    training = env.service.training
    assert training is not None
    assert training.load_state_calls == [cursor.state_path]
    assert _loss_fns(env).count("cross_entropy") == 12
    # The teacher corpus was reused from this run's own manifest, not re-collected.
    assert len(_warmup_calls(env)) == 1
    assert [row["step"] for row in _offpolicy_rows(env.run_dir)] == list(range(12))
    record = store.read_offpolicy()
    assert record is not None
    assert record.epochs == 3
    assert record.steps == 12
    assert store.read_offpolicy_cursor() is None


def test_a_stale_offpolicy_cursor_is_refused_rather_than_retrained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cursor from a differently sized corpus names different minibatches."""
    env = _setup(tmp_path, monkeypatch)
    cfg = _offpolicy_cfg(epochs=3, minibatch_datums=1)
    env.service.crash_on_cross_entropy = 5

    with pytest.raises(RuntimeError, match="injected cross_entropy crash"):
        _run(env, cfg)

    store = DistillRunStore(env.run_dir)
    stale = store.read_offpolicy_cursor()
    assert stale is not None
    store.write_offpolicy_cursor(stale.model_copy(update={"datums": stale.datums + 1}))

    env.service.crash_on_cross_entropy = None
    with pytest.raises(RuntimeError, match="no longer names the same minibatches"):
        _run(env, cfg, resume=True)


def test_offpolicy_zero_passing_trials_skips_to_pure_opd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Nothing to train on degrades the run to pure OPD, never aborts it."""
    env = _setup(tmp_path, monkeypatch)
    env.rollouts.teacher_fail_all = True

    with caplog.at_level(logging.WARNING, logger="wmh.distill.loop"):
        result = _run(env, _offpolicy_cfg())

    assert "pure on-policy distillation" in caplog.text
    assert result.steps_completed == 3
    assert _loss_fns(env) == ["importance_sampling"] * 3
    assert len(_warmup_calls(env)) == 1  # the collection itself did run

    rows = _offpolicy_rows(env.run_dir)
    assert len(rows) == 1
    assert rows[0]["trials"] == 8
    assert rows[0]["kept_trials"] == 0
    assert rows[0]["datums"] == 0
    record = DistillRunStore(env.run_dir).read_offpolicy()
    assert record is not None
    assert record.epochs == 0
    assert record.steps == 0
    assert record.skipped_reason is not None
    assert "keep='passed'" in record.skipped_reason
    assert record.state_path is None and record.sampler_path is None


def test_offpolicy_loads_a_source_runs_collection_without_collecting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    source_dir = tmp_path / "source-run"
    env.run_dir = source_dir
    _run(env, _offpolicy_cfg())
    ce_before = _loss_fns(env).count("cross_entropy")
    assert ce_before == 2

    env.run_dir = tmp_path / "target-run"
    result = _run(env, _offpolicy_cfg(trajectories_from=str(source_dir)))

    assert result.steps_completed == 3
    assert _warmup_calls(env) == []  # the target collected nothing
    assert _loss_fns(env).count("cross_entropy") == ce_before + 2
    record = DistillRunStore(env.run_dir).read_offpolicy()
    assert record is not None
    assert record.trials == 8
    assert record.kept_trials == 4


def test_offpolicy_load_with_a_mismatched_teacher_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    source_dir = tmp_path / "source-run"
    env.run_dir = source_dir
    _run(env, _offpolicy_cfg())
    ce_before = _loss_fns(env).count("cross_entropy")

    env.run_dir = tmp_path / "target-run"
    cfg = _offpolicy_cfg(trajectories_from=str(source_dir)).model_copy(
        update={"teacher": TeacherConfig(model="fake/other-teacher-70b")}
    )
    with pytest.raises(ValueError, match="sampled by teacher .* this run's teacher"):
        _run(env, cfg)

    assert _loss_fns(env).count("cross_entropy") == ce_before  # nothing trained


def test_offpolicy_shuffle_reorders_epochs_without_dropping_datums(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A seeded shuffle changes the order inside each epoch, never the coverage."""
    env = _setup(tmp_path, monkeypatch)

    _run(env, _offpolicy_cfg(epochs=2, minibatch_datums=1, shuffle_seed=3))

    training = env.service.training
    assert training is not None
    ce_batches = [
        batch for batch, loss in training.inner.forward_backward_calls if loss == "cross_entropy"
    ]
    assert len(ce_batches) == 8  # 4 datums x 2 epochs, one datum per step
    assert all(len(batch) == 1 for batch in ce_batches)
    # Each epoch covers the whole corpus exactly once; the two epochs' token
    # sequences are the same multiset, ordered differently.
    epochs = [
        [tuple(batch[0].model_input_tokens) for batch in ce_batches[:4]],
        [tuple(batch[0].model_input_tokens) for batch in ce_batches[4:]],
    ]
    assert sorted(epochs[0]) == sorted(epochs[1])
    assert len(set(epochs[0])) == 4
    assert epochs[0] != epochs[1]


# -- the supervised warmup phase ---------------------------------------------------------------


def test_warmup_trains_on_passing_teacher_trials_then_opd_proceeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    tracker = _RecordingTracker()

    result = _run(env, _warmup_cfg(), tracker=tracker)

    assert result.steps_completed == 3
    assert result.gate.accepted

    # The warmup collection ran the TEACHER on the full train split under the
    # isolated warmup-rollouts root, rollouts_per_task attempts per task.
    (warmup_call,) = _warmup_calls(env)
    assert warmup_call.provider_model == _TEACHER
    assert warmup_call.task_ids == _TRAIN_IDS
    assert warmup_call.attempts == 2
    assert warmup_call.step_index == 0

    # Two cross_entropy passes over the passed-filter datums (tasks at even
    # positions pass: 2 tasks x 2 attempts = 4 kept trials, one datum each),
    # then the three importance_sampling OPD steps; every cross_entropy batch
    # cleared the fake's TITO check on TEACHER-issued spans. The warmup LR
    # applies to the warmup passes only.
    training = env.service.training
    assert training is not None
    assert _loss_fns(env) == ["cross_entropy"] * 2 + ["importance_sampling"] * 3
    ce_batches = [
        batch for batch, loss in training.inner.forward_backward_calls if loss == "cross_entropy"
    ]
    assert all(len(batch) == 4 for batch in ce_batches)
    assert training.inner.optim_step_lrs == [5e-5, 5e-5, 1e-4, 1e-4, 1e-4]

    # OPD step 0 sampled the WARMED student: the post-warmup forced refresh
    # produced a fresh sampler path, distinct from the pre-warmup weights the
    # student-before baseline sampled.
    train_calls = [call for call in env.rollouts.calls if call.run_dir == env.run_dir]
    before_call = next(
        call for call in env.rollouts.calls if call.run_dir.name == STUDENT_BEFORE_EVAL
    )
    assert "warmup" in train_calls[0].provider_model
    assert train_calls[0].provider_model != before_call.provider_model

    # Metrics: one phase-tagged warmup row per warmup step, then the OPD rows
    # (their step indices restart at 0, so the phase key is the discriminator).
    store = DistillRunStore(env.run_dir)
    rows = store.read_metrics()
    warmup_rows = [row for row in rows if row.get("phase") == "warmup"]
    step_rows = [row for row in rows if "phase" not in row]
    assert [row["step"] for row in warmup_rows] == [0, 1]
    assert [row["step"] for row in step_rows] == [0, 1, 2]
    for row in warmup_rows:
        assert row["trials"] == 8
        assert row["kept_trials"] == 4
        assert row["solve_rate"] == 0.5
        assert row["datums"] == 4
        assert _number(row, "learning_rate") == pytest.approx(5e-5)
        assert _number(row, "student_train_tokens") > 0
        assert _number(row, "usd") > 0
    # The teacher collection's charge folds into warmup row 0 only, billed as
    # a teacher-in-harness batch: sampled tokens at teacher_sample plus
    # per-request prefill (unique full-rate, repeats cached).
    assert _number(warmup_rows[0], "teacher_prefill_tokens") > 0
    assert _number(warmup_rows[0], "teacher_cached_prefill_tokens") > 0
    assert _number(warmup_rows[0], "teacher_sample_tokens") > 0
    assert _number(warmup_rows[1], "teacher_prefill_tokens") == 0
    assert _number(warmup_rows[1], "teacher_cached_prefill_tokens") == 0
    assert _number(warmup_rows[1], "teacher_sample_tokens") == 0

    # The tracker saw the same warmup rows the store persisted.
    assert [step for step, _ in tracker.warmup_steps] == [0, 1]
    for (step, metrics), row in zip(tracker.warmup_steps, warmup_rows, strict=True):
        assert row == {"step": step, **metrics.model_dump(mode="json")}

    # The completion marker records the phase, and warmup never lands in the
    # checkpoint manifest (checkpoint steps drive the resume step count).
    record = store.read_warmup()
    assert record is not None
    assert record.steps == 2
    assert record.trials == 8
    assert record.kept_trials == 4
    assert record.datums == 4
    assert record.skipped_reason is None
    assert record.state_path is not None
    assert record.sampler_path == train_calls[0].provider_model
    assert [checkpoint.step for checkpoint in store.checkpoints()] == [1, 2]

    assert "warmup" in {event.phase for event in env.progress}


def test_warmup_zero_passing_trials_skips_to_pure_opd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Zero passing teacher trials degrade the run to pure OPD, never abort it."""
    env = _setup(tmp_path, monkeypatch)
    env.rollouts.teacher_fail_all = True

    with caplog.at_level(logging.WARNING, logger="wmh.distill.loop"):
        result = _run(env, _warmup_cfg())

    assert "pure on-policy distillation" in caplog.text
    assert result.steps_completed == 3
    assert _loss_fns(env) == ["importance_sampling"] * 3  # nothing to warm up on
    assert len(_warmup_calls(env)) == 1  # the collection itself did run

    store = DistillRunStore(env.run_dir)
    warmup_rows = [row for row in store.read_metrics() if row.get("phase") == "warmup"]
    assert len(warmup_rows) == 1
    assert warmup_rows[0]["trials"] == 8
    assert warmup_rows[0]["kept_trials"] == 0
    assert warmup_rows[0]["datums"] == 0
    record = store.read_warmup()
    assert record is not None
    assert record.steps == 0
    assert record.skipped_reason is not None
    assert "keep='passed'" in record.skipped_reason
    assert record.state_path is None and record.sampler_path is None


def test_warmup_keep_all_trains_on_failing_trials_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    env.rollouts.teacher_fail_all = True  # nothing passes; keep="all" still trains

    result = _run(env, _warmup_cfg(warmup_steps=1, keep="all"))

    assert result.steps_completed == 3
    assert _loss_fns(env) == ["cross_entropy"] + ["importance_sampling"] * 3
    training = env.service.training
    assert training is not None
    (ce_batch, _) = training.inner.forward_backward_calls[0]
    assert len(ce_batch) == 8  # every trial kept: 4 tasks x 2 attempts
    record = DistillRunStore(env.run_dir).read_warmup()
    assert record is not None
    assert record.kept_trials == 8
    assert record.datums == 8


def test_resume_never_reruns_a_completed_warmup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A finalize interrupted after the gate eval resumes without re-running
    warmup or the student-after eval. (Resuming a fully COMPLETED run is
    refused outright: see test_resume_of_a_completed_run_is_refused.)"""
    env = _setup(tmp_path, monkeypatch)
    cfg = _warmup_cfg()
    _run(env, cfg)
    assert len(_warmup_calls(env)) == 1
    assert _loss_fns(env).count("cross_entropy") == 2

    store = DistillRunStore(env.run_dir)
    after_evals_before = _eval_run_counts(env)[STUDENT_AFTER_EVAL]
    # Simulate a crash between the student-after eval and the gate verdict.
    store.gate_path.unlink()

    result = _run(env, cfg, resume=True)

    assert result.steps_completed == 3
    # Neither the teacher collection nor the SFT passes ran again.
    assert len(_warmup_calls(env)) == 1
    assert _loss_fns(env).count("cross_entropy") == 2
    # The recorded student-after eval was reused, not re-spent.
    assert _eval_run_counts(env)[STUDENT_AFTER_EVAL] == after_evals_before


def test_resume_of_a_completed_run_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resuming a run whose gate verdict is recorded would re-spend the holdout
    eval and could promote a duplicate adapter version; it must refuse."""
    env = _setup(tmp_path, monkeypatch)
    cfg = _warmup_cfg()
    _run(env, cfg)

    with pytest.raises(RuntimeError, match="already completed"):
        _run(env, cfg, resume=True)


def test_budget_abort_mid_warmup_reruns_warmup_whole_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted warmup holds no completion marker, so it re-runs whole."""
    env = _setup(tmp_path, monkeypatch)
    capped = _warmup_cfg(
        budget_max=1.0,
        # Only student_train is priced, so the run survives the baselines and
        # the teacher collection, then the first warmup pass blows the cap.
        pricing=PricingConfig(
            student_prefill=0.0, student_sample=0.0, student_train=1e9, teacher_prefill=0.0
        ),
    )

    with pytest.raises(DistillBudgetError):
        _run(env, capped)

    store = DistillRunStore(env.run_dir)
    assert store.read_warmup() is None  # no marker: the phase never finished
    assert len(_warmup_calls(env)) == 1
    assert _loss_fns(env).count("cross_entropy") == 1  # the aborted first pass

    result = _run(env, _warmup_cfg(), resume=True)  # cap lifted: the documented recovery

    assert result.steps_completed == 3
    # The resumed session re-collected and re-trained the whole warmup phase.
    assert len(_warmup_calls(env)) == 2
    assert _loss_fns(env).count("cross_entropy") == 1 + 2
    record = store.read_warmup()
    assert record is not None
    assert record.steps == 2


def test_resume_restores_post_warmup_state_when_no_step_checkpoint_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash between warmup and the first step checkpoint keeps the warmup.

    Without the warmup record's state_path restore, this resume would start
    OPD from the COLD student and silently lose the warmup it already paid
    for (no step checkpoint exists yet for load_state to use).
    """
    env = _setup(tmp_path, monkeypatch)
    env.rollouts.fail_on_train_step = 0
    cfg = _warmup_cfg()

    with pytest.raises(RuntimeError, match="injected rollout crash"):
        _run(env, cfg)

    store = DistillRunStore(env.run_dir)
    record = store.read_warmup()
    assert record is not None
    assert record.state_path is not None
    assert store.latest_checkpoint() is None  # nothing step-level to restore

    env.rollouts.fail_on_train_step = None
    result = _run(env, cfg, resume=True)

    assert result.steps_completed == 3
    training = env.service.training
    assert training is not None
    assert training.load_state_calls == [record.state_path]
    assert len(_warmup_calls(env)) == 1  # warmup itself was not re-run


# -- shared warmup-collection loading (warmup.trajectories_from) ------------------------------


def test_warmup_collection_writes_the_trials_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The collection persists every assembled trial, pre keep-filter."""
    env = _setup(tmp_path, monkeypatch)

    _run(env, _warmup_cfg())

    manifest = DistillRunStore(env.run_dir).read_warmup_trials()
    assert manifest is not None
    assert manifest.teacher_model == _TEACHER
    # Unfiltered: all 4 train tasks x 2 attempts, failing trials included,
    # so a loading run may apply a different keep filter.
    assert len(manifest.records) == 8
    assert sum(1 for record in manifest.records if record.passed) == 4
    assert all(record.spans for record in manifest.records)


def test_warmup_loads_a_source_runs_collection_without_collecting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    source_dir = tmp_path / "source-run"
    env.run_dir = source_dir
    _run(env, _warmup_cfg())
    ce_before = _loss_fns(env).count("cross_entropy")
    assert ce_before == 2

    env.run_dir = tmp_path / "target-run"
    result = _run(env, _warmup_cfg(trajectories_from=str(source_dir)))

    assert result.steps_completed == 3
    # No teacher collection ran for the target: the only warmup-rollouts
    # batch across both runs is the source run's.
    assert _warmup_calls(env) == []
    warmup_batches = [c for c in env.rollouts.calls if c.run_dir.name == WARMUP_ROLLOUTS_DIR]
    assert len(warmup_batches) == 1
    assert warmup_batches[0].run_dir == source_dir / WARMUP_ROLLOUTS_DIR
    # The CE training passes still ran per the target run's config.
    assert _loss_fns(env).count("cross_entropy") == ce_before + 2
    record = DistillRunStore(env.run_dir).read_warmup()
    assert record is not None
    assert record.steps == 2
    assert record.trials == 8
    assert record.kept_trials == 4
    assert record.datums == 4
    # OPD step 0 sampled the warmed student, exactly like a collecting run.
    train_calls = [call for call in env.rollouts.calls if call.run_dir == env.run_dir]
    assert "warmup" in train_calls[0].provider_model


def test_warmup_load_charges_no_teacher_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loading is free: the source run already paid for the collection."""
    env = _setup(tmp_path, monkeypatch)
    source_dir = tmp_path / "source-run"
    env.run_dir = source_dir
    source_result = _run(env, _warmup_cfg())
    source_rows = [
        row for row in DistillRunStore(source_dir).read_metrics() if row.get("phase") == "warmup"
    ]

    env.run_dir = tmp_path / "target-run"
    result = _run(env, _warmup_cfg(trajectories_from=str(source_dir)))

    target_rows = [
        row for row in DistillRunStore(env.run_dir).read_metrics() if row.get("phase") == "warmup"
    ]
    # Warmup row 0 folds everything charged since run start: both runs pay
    # identical baseline volumes (the fakes sample fixed-length sequences for
    # the same tasks), and only the SOURCE run pays the teacher collection on
    # top, so every teacher meter delta shrinks strictly when loading.
    for meter in (
        "teacher_prefill_tokens",
        "teacher_cached_prefill_tokens",
        "teacher_sample_tokens",
    ):
        assert 0 < _number(target_rows[0], meter) < _number(source_rows[0], meter)
    assert _number(target_rows[0], "usd") < _number(source_rows[0], "usd")
    assert result.spend.total_usd < source_result.spend.total_usd


def test_warmup_load_with_a_mismatched_teacher_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Warmup SFT must train on THIS run's teacher, so a foreign manifest fails."""
    env = _setup(tmp_path, monkeypatch)
    source_dir = tmp_path / "source-run"
    env.run_dir = source_dir
    _run(env, _warmup_cfg())
    ce_before = _loss_fns(env).count("cross_entropy")

    env.run_dir = tmp_path / "target-run"
    cfg = _warmup_cfg(trajectories_from=str(source_dir)).model_copy(
        update={"teacher": TeacherConfig(model="fake/other-teacher-70b")}
    )
    with pytest.raises(ValueError, match="sampled by teacher .* this run's teacher"):
        _run(env, cfg)

    assert _loss_fns(env).count("cross_entropy") == ce_before  # nothing trained


def test_warmup_load_applies_the_keep_filter_at_load_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The manifest is unfiltered, so keep= may differ from the source run's.

    An all-failing source collection degrades a keep="passed" loader to pure
    OPD (the same warning path as a fresh collection), while a keep="all"
    loader trains on every loaded trial.
    """
    env = _setup(tmp_path, monkeypatch)
    env.rollouts.teacher_fail_all = True
    source_dir = tmp_path / "source-run"
    env.run_dir = source_dir
    _run(env, _warmup_cfg())  # degrades itself, but the manifest is written pre-filter
    assert _loss_fns(env).count("cross_entropy") == 0

    env.run_dir = tmp_path / "target-passed"
    with caplog.at_level(logging.WARNING, logger="wmh.distill.loop"):
        result = _run(env, _warmup_cfg(trajectories_from=str(source_dir)))

    assert result.steps_completed == 3
    assert "pure on-policy distillation" in caplog.text
    assert _loss_fns(env).count("cross_entropy") == 0
    record = DistillRunStore(env.run_dir).read_warmup()
    assert record is not None
    assert record.steps == 0
    assert record.kept_trials == 0
    assert record.skipped_reason is not None
    assert "loaded" in record.skipped_reason

    env.run_dir = tmp_path / "target-all"
    result_all = _run(
        env, _warmup_cfg(warmup_steps=1, keep="all", trajectories_from=str(source_dir))
    )

    assert result_all.steps_completed == 3
    assert _loss_fns(env).count("cross_entropy") == 1
    record_all = DistillRunStore(env.run_dir).read_warmup()
    assert record_all is not None
    assert record_all.kept_trials == 8
    assert record_all.datums == 8


def test_warmup_load_without_a_manifest_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="no warmup trial manifest"):
        _run(env, _warmup_cfg(trajectories_from=str(tmp_path / "not-a-run")))


# -- preflight failure paths -----------------------------------------------------------------


def test_missing_api_key_without_an_injected_client_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """service_client=None requires the key (or the extra) before any client is built."""
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    with pytest.raises((RuntimeError, ImportError), match="TINKER_API_KEY|--extra distill"):
        run_distillation(
            _NAME,
            _cfg(),
            HarnessDoc.baseline(),
            _TRAIN_IDS,
            _HOLDOUT_IDS,
            tmp_path / "run",
        )


class _ExplodingScorer:
    """A sampling client whose every call fails (a retired teacher model)."""

    def sample(
        self, prompt_token_ids: list[int], *, max_tokens: int, temperature: float
    ) -> SampledSequenceLike:
        raise RuntimeError("model not found: the teacher was retired")

    def compute_logprobs(self, token_ids: list[int]) -> list[float | None]:
        raise RuntimeError("model not found: the teacher was retired")


def test_teacher_ping_failure_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    original = env.service.create_sampling_client

    def patched(model_path: str) -> DistillSamplingClient:
        if model_path == _TEACHER:
            return _ExplodingScorer()
        return original(model_path)

    monkeypatch.setattr(env.service, "create_sampling_client", patched)
    with pytest.raises(RuntimeError, match="teacher preflight ping failed") as excinfo:
        _run(env, _cfg())
    assert "the teacher was retired" in str(excinfo.value)


class _SkewedClient:
    """Delegates sampling but perturbs recomputed logprobs (scoring-path drift)."""

    def __init__(self, inner: DistillSamplingClient) -> None:
        self._inner = inner

    def sample(
        self, prompt_token_ids: list[int], *, max_tokens: int, temperature: float
    ) -> SampledSequenceLike:
        return self._inner.sample(prompt_token_ids, max_tokens=max_tokens, temperature=temperature)

    def compute_logprobs(self, token_ids: list[int]) -> list[float | None]:
        return [
            None if logprob is None else logprob + 1.0
            for logprob in self._inner.compute_logprobs(token_ids)
        ]


def test_tito_recompute_disagreement_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    original = env.service.create_sampling_client

    def patched(model_path: str) -> DistillSamplingClient:
        client = original(model_path)
        if model_path.startswith("tinker://"):
            return _SkewedClient(client)
        return client

    monkeypatch.setattr(env.service, "create_sampling_client", patched)
    with pytest.raises(RuntimeError, match="TITO recompute disagreement"):
        _run(env, _cfg())


class _PerturbedClient:
    """Delegates sampling but shifts recomputed logprobs by per-index offsets.

    Offsets are applied cyclically over the recomputed sequence, letting tests
    model zero-mean kernel noise (small alternating offsets) or one
    catastrophic position (a single large offset).
    """

    def __init__(self, inner: DistillSamplingClient, offsets: Sequence[float]) -> None:
        self._inner = inner
        self._offsets = list(offsets)

    def sample(
        self, prompt_token_ids: list[int], *, max_tokens: int, temperature: float
    ) -> SampledSequenceLike:
        return self._inner.sample(prompt_token_ids, max_tokens=max_tokens, temperature=temperature)

    def compute_logprobs(self, token_ids: list[int]) -> list[float | None]:
        recomputed = self._inner.compute_logprobs(token_ids)
        return [
            None if lp is None else lp + self._offsets[index % len(self._offsets)]
            for index, lp in enumerate(recomputed)
        ]


def test_tito_recompute_tolerates_zero_mean_kernel_noise() -> None:
    """Small alternating sampler/scorer drift (the observed live regime) passes."""
    inner = FakeSamplingClient("tito-noise-probe")
    noisy = _PerturbedClient(inner, [0.08, -0.08])
    loop_module.tito_recompute_check(noisy, [1, 2, 3, 4])


def test_tito_recompute_single_catastrophic_position_fails() -> None:
    """One multi-nat outlier position trips the per-position bound."""
    inner = FakeSamplingClient("tito-spike-probe")
    prompt = [1, 2, 3, 4]
    probe = inner.sample(prompt, max_tokens=16, temperature=1.0)
    spike_index = len(prompt) + len(list(probe.tokens)) - 1
    offsets = [0.0] * spike_index + [2.0]
    spiky = _PerturbedClient(FakeSamplingClient("tito-spike-probe"), offsets)
    with pytest.raises(RuntimeError, match="per-position bound"):
        loop_module.tito_recompute_check(spiky, prompt)


class _OffsetTokenizer:
    """Encodes every character one id higher than the student's tokenizer."""

    def encode(self, text: str) -> list[int]:
        return [ord(ch) + 1 for ch in text]


class _WrongTokenizerTeacher:
    """A teacher client that CAN supply a tokenizer, and it disagrees."""

    def __init__(self, inner: DistillSamplingClient) -> None:
        self._inner = inner

    def sample(
        self, prompt_token_ids: list[int], *, max_tokens: int, temperature: float
    ) -> SampledSequenceLike:
        return self._inner.sample(prompt_token_ids, max_tokens=max_tokens, temperature=temperature)

    def compute_logprobs(self, token_ids: list[int]) -> list[float | None]:
        return self._inner.compute_logprobs(token_ids)

    def get_tokenizer(self) -> _OffsetTokenizer:
        return _OffsetTokenizer()


def test_tokenizer_fingerprint_mismatch_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    original = env.service.create_sampling_client

    def patched(model_path: str) -> DistillSamplingClient:
        client = original(model_path)
        if model_path == _TEACHER:
            return _WrongTokenizerTeacher(client)
        return client

    monkeypatch.setattr(env.service, "create_sampling_client", patched)
    with pytest.raises(ValueError, match="tokenizer fingerprint mismatch"):
        _run(env, _cfg())


# -- input validation and the task sampler ---------------------------------------------------


def test_overlapping_splits_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="BOTH splits"):
        run_distillation(
            _NAME,
            _cfg(),
            HarnessDoc.baseline(),
            ("task-a", "task-b"),
            ("task-b", "hold-a"),
            tmp_path / "run",
            service_client=_Service(),
        )


def test_tinker_provider_config_is_the_one_shape_every_rollout_samples_through() -> None:
    """The student sampler's provider config IS the shared helper's output.

    The helper is also the seam a rollout-only caller (the scaffold probe in
    `.agents/distill/probe_scaffold.py`) uses to sample the BASE student
    without opening a training client, so `model == base_model` must be legal
    and must carry the same renderer identity.
    """
    base = tinker_provider_config(_STUDENT, _STUDENT)
    assert base.kind is ProviderKind.TINKER
    assert base.model == _STUDENT and base.model_type == _STUDENT

    service = _Service()
    sampler = StudentSampler(service, service.create_lora_training_client(_STUDENT, 8), _NAME)
    path = sampler.refresh(0)

    assert path.startswith("tinker://")
    assert sampler.provider_config(_STUDENT) == tinker_provider_config(path, _STUDENT)
    # The teacher side is the same call with the teacher's identity and renderer.
    assert tinker_provider_config(_TEACHER, _TEACHER).model_type == _TEACHER


def test_task_sampler_is_seeded_unique_and_covering() -> None:
    ids = ["a", "b", "c", "d", "e"]
    first = TaskSampler(ids, seed=7)
    second = TaskSampler(ids, seed=7)
    batches = [first.next_batch(2) for _ in range(6)]
    assert batches == [second.next_batch(2) for _ in range(6)]  # deterministic
    for batch in batches:
        assert len(batch) == 2
        assert len(set(batch)) == 2  # unique within a batch
    # The cycle visits every task before repeating any: the first three
    # batches (six slots over a five-task split) cover the whole split.
    assert {task for batch in batches[:3] for task in batch} == set(ids)
    # Oversized requests clamp to the split size.
    assert TaskSampler(["only"], seed=0).next_batch(5) == ["only"]
    with pytest.raises(ValueError, match="duplicates"):
        TaskSampler(["a", "a"], seed=0)
    with pytest.raises(ValueError, match="empty"):
        TaskSampler([], seed=0)


# -- SdkTrainingClient deadlines: retry-once vs abort per call idempotency -----------------------


class _NeverResolvingFuture:
    """Mimics the SDK future of a wedged session: result(timeout) honors the timeout."""

    def __init__(self) -> None:
        self._never = threading.Event()

    def result(self, timeout: float | None = None) -> NoReturn:
        self._never.wait(timeout)
        raise TimeoutError(f"fake future gave up after {timeout}s")


@dataclass(frozen=True)
class _SavedArtifact:
    path: str


class _ReadyFuture:
    """A fake SDK future whose result is immediately available."""

    def __init__(self, value: object = None) -> None:
        self._value = value

    def result(self, timeout: float | None = None) -> object:
        del timeout
        return self._value


class _WedgedOnceTrainingClient:
    """Fake tinker.TrainingClient: the FIRST call of each method wedges, retries succeed."""

    def __init__(self) -> None:
        self.forward_backward_calls = 0
        self.optim_step_calls = 0
        self.save_state_names: list[str] = []
        self.load_state_paths: list[str] = []
        self.save_weights_names: list[str] = []

    def forward_backward(
        self, datums: object, loss_fn: object
    ) -> _NeverResolvingFuture | _ReadyFuture:
        del datums, loss_fn
        self.forward_backward_calls += 1
        return _NeverResolvingFuture()

    def optim_step(self, params: object) -> _NeverResolvingFuture | _ReadyFuture:
        del params
        self.optim_step_calls += 1
        return _NeverResolvingFuture()

    def save_state(self, name: str) -> _NeverResolvingFuture | _ReadyFuture:
        self.save_state_names.append(name)
        if len(self.save_state_names) == 1:
            return _NeverResolvingFuture()
        return _ReadyFuture(_SavedArtifact(path=f"tinker://fake/state/{name}"))

    def load_state(self, path: str) -> _NeverResolvingFuture | _ReadyFuture:
        self.load_state_paths.append(path)
        if len(self.load_state_paths) == 1:
            return _NeverResolvingFuture()
        return _ReadyFuture()

    def save_weights_for_sampler(self, name: str) -> _NeverResolvingFuture | _ReadyFuture:
        self.save_weights_names.append(name)
        if len(self.save_weights_names) == 1:
            return _NeverResolvingFuture()
        return _ReadyFuture(_SavedArtifact(path=f"tinker://fake/sampler/{name}"))


class _ReadyStateTrainingClient:
    """Fake tinker.TrainingClient whose state and weights futures resolve at once.

    The counterpart to `_WedgedOnceTrainingClient`: nothing wedges here, so
    tests can drive real call SEQUENCES through the adapter.
    """

    def __init__(self) -> None:
        self.save_state_names: list[str] = []
        self.load_state_paths: list[str] = []
        self.save_weights_names: list[str] = []
        self.tokenizer_calls = 0

    def get_tokenizer(self) -> FakeTokenizer:
        self.tokenizer_calls += 1
        return FakeTokenizer()

    def save_state(self, name: str) -> _ReadyFuture:
        self.save_state_names.append(name)
        return _ReadyFuture(_SavedArtifact(path=f"tinker://fake/state/{name}"))

    def load_state(self, path: str) -> _ReadyFuture:
        self.load_state_paths.append(path)
        return _ReadyFuture()

    def save_weights_for_sampler(self, name: str) -> _ReadyFuture:
        self.save_weights_names.append(name)
        return _ReadyFuture(_SavedArtifact(path=f"tinker://fake/sampler/{name}"))


def _sdk_training_client(
    fake: _WedgedOnceTrainingClient | _ReadyStateTrainingClient,
) -> SdkTrainingClient:
    return SdkTrainingClient(cast("tinker.TrainingClient", fake))


def _short_deadlines(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_var in (
        "WMH_TINKER_DEADLINE_FORWARD_BACKWARD",
        "WMH_TINKER_DEADLINE_OPTIM_STEP",
        "WMH_TINKER_DEADLINE_SAVE_STATE",
        "WMH_TINKER_DEADLINE_LOAD_STATE",
        "WMH_TINKER_DEADLINE_SAVE_WEIGHTS_FOR_SAMPLER",
        "WMH_TINKER_DEADLINE_CONNECT",
    ):
        monkeypatch.setenv(env_var, "0.05")


def test_save_weights_for_sampler_retries_once_under_a_fresh_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _short_deadlines(monkeypatch)
    fake = _WedgedOnceTrainingClient()
    client = _sdk_training_client(fake)

    path = client.save_weights_for_sampler("run-step-0001")

    # Retried exactly once, and the retry saved under a distinct "-r1" name so
    # a first attempt that completed server-side after abandonment cannot collide.
    assert len(fake.save_weights_names) == 2
    first, second = fake.save_weights_names
    assert first.startswith("run-step-0001-")
    assert second == f"{first}-r1"
    assert path == f"tinker://fake/sampler/{second}"


def test_save_state_retries_once_with_a_fresh_name(monkeypatch: pytest.MonkeyPatch) -> None:
    _short_deadlines(monkeypatch)
    fake = _WedgedOnceTrainingClient()
    client = _sdk_training_client(fake)

    path = client.save_state()

    assert len(fake.save_state_names) == 2
    first, second = fake.save_state_names
    assert first != second  # the retry advanced the per-session counter
    assert first.endswith("-state-0000")
    assert second.endswith("-state-0001")
    assert path == f"tinker://fake/state/{second}"


def test_load_state_deadline_aborts_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """The live resume blocker: a retried load_state can only be rejected.

    Tinker accepts LoadWeights only on an uninitialized model and the
    abandoned first request keeps running server-side, so re-issuing it on the
    same client raised "LoadWeights can only be called on uninitialized
    models" and killed three resumes. Exactly one call now, and the retry
    happens on a fresh client (see the _open_training_client tests).
    """
    _short_deadlines(monkeypatch)
    fake = _WedgedOnceTrainingClient()
    client = _sdk_training_client(fake)

    with pytest.raises(TinkerDeadlineError, match="tinker load_state timed out"):
        client.load_state("tinker://fake/state/x")
    assert fake.load_state_paths == ["tinker://fake/state/x"]


def test_load_state_after_an_initializing_call_is_refused() -> None:
    """The adapter fails fast instead of spending a request the service rejects.

    Every call that initializes the model server-side closes the door on
    LoadWeights, so a sampler save, a state save, or an earlier restore each
    make the next load_state a local error naming the fix.
    """
    after_weights = _ReadyStateTrainingClient()
    client = _sdk_training_client(after_weights)
    client.save_weights_for_sampler("s0")
    with pytest.raises(RuntimeError, match="load_state must be the first call"):
        client.load_state("tinker://fake/state/x")

    after_save = _ReadyStateTrainingClient()
    client = _sdk_training_client(after_save)
    client.save_state()
    with pytest.raises(RuntimeError, match="load_state must be the first call"):
        client.load_state("tinker://fake/state/x")

    after_restore = _ReadyStateTrainingClient()
    client = _sdk_training_client(after_restore)
    client.load_state("tinker://fake/state/first")
    with pytest.raises(RuntimeError, match="load_state must be the first call"):
        client.load_state("tinker://fake/state/x")

    # No refused restore reached the service.
    assert after_weights.load_state_paths == []
    assert after_save.load_state_paths == []
    assert after_restore.load_state_paths == ["tinker://fake/state/first"]


def test_get_tokenizer_stays_legal_before_a_restore() -> None:
    """The tokenizer is metadata (SDK GetInfo), so it never blocks LoadWeights."""
    fake = _ReadyStateTrainingClient()
    client = _sdk_training_client(fake)

    client.get_tokenizer()
    client.load_state("tinker://fake/state/x")

    assert fake.load_state_paths == ["tinker://fake/state/x"]


def _attached_datum() -> TrainDatum:
    return TrainDatum(
        trial_name="task-a__x1",
        fragment_index=0,
        model_input_tokens=[1, 2, 3],
        loss_mask=[0.0, 1.0, 1.0],
        sampled_logprobs=[0.0, -0.5, -0.7],
        advantages=[0.0, 0.1, 0.2],
    )


def test_forward_backward_deadline_aborts_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # NOT idempotent: gradients may have accumulated server-side before the
    # deadline fired, so a retry could count the batch twice. Exactly one call.
    pytest.importorskip("tinker")
    _short_deadlines(monkeypatch)
    fake = _WedgedOnceTrainingClient()
    client = _sdk_training_client(fake)

    with pytest.raises(TinkerDeadlineError, match="tinker forward_backward timed out"):
        client.forward_backward([_attached_datum()], loss_fn="importance_sampling")
    assert fake.forward_backward_calls == 1


def test_optim_step_deadline_aborts_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    # NOT idempotent: the step may have been applied server-side; a retry
    # would double-step the optimizer. Exactly one call.
    pytest.importorskip("tinker")
    _short_deadlines(monkeypatch)
    fake = _WedgedOnceTrainingClient()
    client = _sdk_training_client(fake)

    with pytest.raises(TinkerDeadlineError, match="tinker optim_step timed out"):
        client.optim_step(1e-5)
    assert fake.optim_step_calls == 1


# -- SdkTrainingClient metric extraction (what the SDK actually exposes) --------------------------


def test_sdk_metric_value_matches_bare_and_suffixed_names() -> None:
    """Server metric keys carry a ':reduction' suffix; both spellings match."""
    assert sdk_metric_value({"total_loss:sum": 2.0}, SDK_LOSS_METRIC_NAMES) == 2.0
    assert sdk_metric_value({"total_loss": 1.5}, SDK_LOSS_METRIC_NAMES) == 1.5
    assert sdk_metric_value({"loss:mean": 0.25}, SDK_LOSS_METRIC_NAMES) == 0.25
    assert sdk_metric_value({"grad_norm:mean": 0.5}, SDK_GRAD_NORM_METRIC_NAMES) == 0.5
    # Unrelated keys (e.g. the documented MoE diagnostics), an empty dict, and
    # the OptimStepResponse's metrics=None all surface nothing.
    assert sdk_metric_value({"e_frac_with_tokens:mean": 1.0}, SDK_LOSS_METRIC_NAMES) is None
    assert sdk_metric_value({}, SDK_GRAD_NORM_METRIC_NAMES) is None
    assert sdk_metric_value(None, SDK_LOSS_METRIC_NAMES) is None


@dataclass(frozen=True)
class _FwdBwdOutput:
    """The ForwardBackwardOutput slice the adapter reads (no typed loss exists)."""

    metrics: dict[str, float]


@dataclass(frozen=True)
class _OptimResponse:
    """The OptimStepResponse slice the adapter reads."""

    metrics: dict[str, float] | None


class _ReadyTrainingClient:
    """Fake tinker.TrainingClient whose futures resolve immediately."""

    def __init__(
        self, fwdbwd_metrics: dict[str, float], optim_metrics: dict[str, float] | None
    ) -> None:
        self._fwdbwd_metrics = fwdbwd_metrics
        self._optim_metrics = optim_metrics

    def forward_backward(self, datums: object, loss_fn: object) -> _ReadyFuture:
        del datums, loss_fn
        return _ReadyFuture(_FwdBwdOutput(metrics=self._fwdbwd_metrics))

    def optim_step(self, params: object) -> _ReadyFuture:
        del params
        return _ReadyFuture(_OptimResponse(metrics=self._optim_metrics))


def test_sdk_training_client_extracts_reported_metrics() -> None:
    pytest.importorskip("tinker")
    fake = _ReadyTrainingClient(
        fwdbwd_metrics={"total_loss:sum": 1.25}, optim_metrics={"grad_norm:mean": 3.5}
    )
    client = SdkTrainingClient(cast("tinker.TrainingClient", fake))
    output = client.forward_backward([_attached_datum()], loss_fn="importance_sampling")
    assert output.loss == 1.25
    assert client.optim_step(1e-5).grad_norm == 3.5


def test_sdk_training_client_surfaces_none_when_the_service_reports_nothing() -> None:
    """No fabricated values: absent metrics stay None end to end."""
    pytest.importorskip("tinker")
    fake = _ReadyTrainingClient(fwdbwd_metrics={}, optim_metrics=None)
    client = SdkTrainingClient(cast("tinker.TrainingClient", fake))
    output = client.forward_backward([_attached_datum()], loss_fn="importance_sampling")
    assert output.loss is None
    assert client.optim_step(1e-5).grad_norm is None


def test_client_construction_is_deadline_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    # Sampling-client construction now goes through the process-wide shared
    # cache (bounded there; see wmh/providers/tinker_test.py), so the loop's
    # remaining direct construction is the training client.
    _short_deadlines(monkeypatch)
    never = threading.Event()

    class _WedgedService:
        def create_lora_training_client(self, base_model: str, rank: int) -> NoReturn:
            del base_model, rank
            never.wait()
            raise AssertionError("unreachable: the event is never set")

    service = SdkServiceClient(cast("tinker.ServiceClient", _WedgedService()))
    with pytest.raises(TinkerDeadlineError, match="tinker connect timed out"):
        service.create_lora_training_client(_STUDENT, rank=8)


def test_loop_sampling_adapter_deadline_evicts_the_shared_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The loop's per-refresh sampling clients come from the process-wide
    # shared cache; a deadline expiry must evict the entry so every future
    # user (the harbor trial providers included) rebuilds a fresh session.
    monkeypatch.setenv("WMH_TINKER_DEADLINE_CONNECT", "0.05")
    never = threading.Event()

    class _WedgedSamplingClient:
        def get_tokenizer(self) -> NoReturn:
            never.wait()
            raise AssertionError("unreachable: the event is never set")

    wedged = cast("tinker.SamplingClient", _WedgedSamplingClient())
    path = "tinker://fake/sampler/x"
    monkeypatch.setattr(providers_tinker, "_shared_samplers", {path: wedged})
    adapter = SdkSamplingClient(wedged, model=path)

    with pytest.raises(TinkerDeadlineError, match="tinker connect timed out"):
        adapter.get_tokenizer()

    assert path not in providers_tinker._shared_samplers


# -- the ppo loss mode (raw-gap advantages, ratio clipping in the loss) --------------------------


def _ppo_cfg(
    *,
    steps: int = 2,
    advantage_clip: float | None = None,
    center_advantages: bool = False,
) -> DistillConfig:
    """The OpenClaw-RL / Slime objective: raw gap under Tinker's ppo loss."""
    base = _cfg()
    return base.model_copy(
        update={
            "train": base.train.model_copy(
                update={
                    "steps": steps,
                    "loss": "ppo",
                    "advantage_clip": advantage_clip,
                    "center_advantages": center_advantages,
                }
            )
        }
    )


def test_ppo_two_step_run_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The full loop in ppo mode: wire loss, raw-gap advantages, TITO, gate.

    The datums are the importance_sampling datums (`to_tinker_datums`); what
    changes is the loss name on the wire and the fact that nothing reshapes
    the advantage, so `clip_fraction` is 0 for the whole run and
    `advantage_mean` is the mean teacher-minus-student gap.
    """
    env = _setup(tmp_path, monkeypatch)

    result = _run(env, _ppo_cfg())

    assert result.steps_completed == 2
    assert result.gate.accepted
    # Every batch went out under the ppo loss, and the fake training client
    # asserts TITO before recording a call, so two recorded batches are two
    # passing checks. No datum is flagged topk: the wire format is unchanged.
    assert _loss_fns(env) == ["ppo"] * 2
    training = env.service.training
    assert training is not None
    for batch, _ in training.inner.forward_backward_calls:
        assert len(batch) == 4
        assert not any(datum.topk for datum in batch)
        assert all(datum.advantages for datum in batch)

    rows = [row for row in DistillRunStore(env.run_dir).read_metrics() if "phase" not in row]
    assert [row["step"] for row in rows] == [0, 1]
    for row in rows:
        assert row["loss"] == "ppo"
        # Unclipped: no token can hit a bound that is not set.
        assert _number(row, "clipped_tokens") == 0
        assert _number(row, "clip_fraction") == 0.0
        # Uncentered, so the mean advantage is the objective itself: the
        # negated reverse KL over the same trained tokens.
        assert _number(row, "advantage_mean") == pytest.approx(
            -_number(row, "reverse_kl_per_token")
        )
        assert _number(row, "advantage_mean") != 0.0
        assert _number(row, "advantage_std") >= 0.0
        assert isinstance(row["pg_loss"], float)
        assert isinstance(row["grad_norm"], float)


def test_ppo_mode_still_catches_fabricated_spans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Token-in-token-out is enforced under ppo exactly as under the default."""
    env = _setup(tmp_path, monkeypatch)
    env.rollouts.fabricate_spans = True
    with pytest.raises(AssertionError, match="TITO violation"):
        _run(env, _ppo_cfg())


@pytest.mark.parametrize(
    ("mode", "wire_loss"),
    [
        ("importance_sampling", "importance_sampling"),
        ("ppo", "ppo"),
        ("topk_ce", "cross_entropy"),
    ],
)
def test_every_loss_mode_dispatches_to_its_wire_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, wire_loss: str
) -> None:
    """`train.loss` decides the wire `loss_fn`, and the row records the mode.

    Two of the three modes share the wire format and differ only in the loss
    name (the service clips the ratio for ppo); topk_ce trains replicas under
    cross_entropy. A metrics row that only carried the wire name could not
    tell topk_ce from the supervised warmup, so it carries the mode.
    """
    env = _setup(tmp_path, monkeypatch)
    base = _cfg()
    cfg = base.model_copy(
        update={"train": base.train.model_copy(update={"steps": 1, "loss": mode, "topk": 2})}
    )

    _run(env, cfg)

    assert _loss_fns(env) == [wire_loss]
    rows = [row for row in DistillRunStore(env.run_dir).read_metrics() if "phase" not in row]
    assert [row["loss"] for row in rows] == [mode]


def test_every_configurable_loss_mode_is_wired_to_a_wire_loss() -> None:
    """No `train.loss` value may reach the step loop without a dispatch.

    The config Literal is the source of truth for the modes; a value added
    there but not to `ADVANTAGE_LOSS_BY_MODE` (or to the topk_ce branch) would
    otherwise fail with a bare KeyError at the first training step, after the
    run has already paid for rollouts and teacher scoring.
    """
    modes = set(get_args(TrainConfig.model_fields["loss"].annotation))
    assert modes == {"importance_sampling", "ppo", "topk_ce"}
    assert set(ADVANTAGE_LOSS_BY_MODE) == modes - {"topk_ce"}
    assert set(ADVANTAGE_LOSS_BY_MODE.values()) == {IMPORTANCE_SAMPLING_LOSS, PPO_LOSS}


def test_advantage_clip_off_versus_on_shows_up_in_the_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clipping is observable end to end, and 0.0 clip_fraction means OFF.

    A bound tight enough that every gap hits it drives clip_fraction to 1.0
    and shrinks the trained advantages to the bound; with no bound nothing is
    counted and the raw gaps ride through. Without the contrast, an unset
    clip and a never-biting clip would read identically on the dashboard.
    """
    clipped_env = _setup(tmp_path / "clipped", monkeypatch)
    _run(clipped_env, _ppo_cfg(steps=1, advantage_clip=1e-6))
    clipped_rows = [
        row for row in DistillRunStore(clipped_env.run_dir).read_metrics() if "phase" not in row
    ]
    (clipped_row,) = clipped_rows
    assert _number(clipped_row, "clipped_tokens") > 0
    assert _number(clipped_row, "clip_fraction") == pytest.approx(1.0)
    assert abs(_number(clipped_row, "advantage_mean")) <= 1e-6

    raw_env = _setup(tmp_path / "raw", monkeypatch)
    _run(raw_env, _ppo_cfg(steps=1))
    (raw_row,) = [
        row for row in DistillRunStore(raw_env.run_dir).read_metrics() if "phase" not in row
    ]
    assert _number(raw_row, "clipped_tokens") == 0
    assert _number(raw_row, "clip_fraction") == 0.0
    assert abs(_number(raw_row, "advantage_mean")) > 1e-6


def test_centering_on_flattens_the_advantage_mean_the_metric_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With centering ON the mean is ~0 by construction; OFF it reads the gap.

    This is why the default moved: `advantage_mean` is how the run is read,
    and under centering it can only ever say 0.
    """
    centered_env = _setup(tmp_path / "centered", monkeypatch)
    _run(centered_env, _ppo_cfg(steps=1, center_advantages=True))
    (centered_row,) = [
        row for row in DistillRunStore(centered_env.run_dir).read_metrics() if "phase" not in row
    ]
    assert _number(centered_row, "advantage_mean") == pytest.approx(0.0, abs=1e-9)
    # The spread survives centering, so the step still carries signal.
    assert _number(centered_row, "advantage_std") > 0.0

    raw_env = _setup(tmp_path / "raw", monkeypatch)
    _run(raw_env, _ppo_cfg(steps=1))
    (raw_row,) = [
        row for row in DistillRunStore(raw_env.run_dir).read_metrics() if "phase" not in row
    ]
    assert _number(raw_row, "advantage_mean") != pytest.approx(0.0, abs=1e-9)


# -- the topk_ce loss mode -----------------------------------------------------------------------


def _topk_cfg(k: int = 3) -> DistillConfig:
    """A 2-step config training the weighted top-k cross_entropy loss."""
    base = _cfg()
    return base.model_copy(
        update={"train": base.train.model_copy(update={"steps": 2, "loss": "topk_ce", "topk": k})}
    )


def test_topk_ce_two_step_run_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The full loop in topk_ce mode: metrics, meters, KL, TITO, and the gate."""
    env = _setup(tmp_path, monkeypatch)
    k = 3

    result = _run(env, _topk_cfg(k))

    assert result.steps_completed == 2
    assert result.gate.accepted
    assert result.adapter_version == 1

    # Every training batch trained cross_entropy over k rank replicas per
    # source datum (2 tasks x 2 attempts, each merged to one datum = 4
    # sources), and every replica cleared the fake's input-side TITO check
    # (candidate targets are teacher-proposed; the input stays sampled tokens).
    training = env.service.training
    assert training is not None
    batches = training.inner.forward_backward_calls
    assert [loss for _, loss in batches] == ["cross_entropy"] * 2
    for batch, _ in batches:
        assert len(batch) == 4 * k
        assert all(datum.topk for datum in batch)
        # Rank replicas of one source share the model input, and at every
        # loss-weighted target index their weights sum to 1 (the renormalized
        # teacher distribution over the k candidates).
        for start in range(0, len(batch), k):
            replicas = batch[start : start + k]
            assert len({tuple(replica.model_input_tokens) for replica in replicas}) == 1
            for index in range(len(replicas[0].target_tokens)):
                weights = [replica.weights[index] for replica in replicas]
                if any(weight != 0.0 for weight in weights):
                    assert sum(weights) == pytest.approx(1.0)
    assert training.inner.optim_step_lrs == [1e-4] * 2

    # Metrics rows: replica-count datums, the reverse-KL metric still present
    # (from the same prefill request's realized logprobs), and no advantage
    # metrics (nothing is clipped or centered in this mode).
    store = DistillRunStore(env.run_dir)
    rows = [row for row in store.read_metrics() if "phase" not in row]
    assert [row["step"] for row in rows] == [0, 1]
    for row in rows:
        assert row["datums"] == 4 * k
        assert row["mismatch_drops"] == 0
        assert isinstance(row["reverse_kl_per_token"], float)
        assert row["advantage_mean"] is None
        assert row["advantage_std"] is None
        assert _number(row, "clip_fraction") == 0.0
        assert _number(row, "clipped_tokens") == 0
        assert isinstance(row["pg_loss"], float)
        assert isinstance(row["grad_norm"], float)
        # Billing honesty: the student_train meter charged k x the CE volume
        # (each replica carries the full sequence: loss + context tokens).
        expected_train = k * (_number(row, "loss_tokens") + _number(row, "context_tokens"))
        assert _number(row, "student_train_tokens") == expected_train
        assert _number(row, "teacher_prefill_tokens") > 0
        assert _number(row, "usd") > 0

    # The spend ledger saw everything, as in the default mode.
    assert store.read_spend() == pytest.approx(result.spend.total_usd)


def test_topk_ce_mode_survives_fabricated_span_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fabricated INPUT spans still die in topk mode: the relaxation moved the
    TITO check to the model input, it did not remove it."""
    env = _setup(tmp_path, monkeypatch)
    env.rollouts.fabricate_spans = True
    with pytest.raises(AssertionError, match="TITO violation in topk datum"):
        _run(env, _topk_cfg())


# --- observability: the scaffold-loss rate reaches the metrics row and the tracker ---------------
def test_metrics_rows_carry_the_scaffold_loss_rate_and_stop_reason_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run whose episodes never submit must SAY so, in the row and on the dashboard.

    The pi/Nemotron-3 runs sat at 88.8% scaffold loss with every trial recorded as `submitted`, so
    nothing in metrics.jsonl, wandb, or the CLI could show that the harness was setting the score.
    """
    env = _setup(tmp_path, monkeypatch)
    env.rollouts.stop_reason = StopReason.NO_TOOL_CALL.value
    tracker = _RecordingTracker()

    _run(env, _cfg(), tracker=tracker)

    rows = [row for row in DistillRunStore(env.run_dir).read_metrics() if "phase" not in row]
    assert rows
    for row in rows:
        assert row["scaffold_loss_rate"] == 1.0
        assert row["stop_reason_counts"] == {"no_tool_call": 4}
        assert row["executed_trials"] == 4
        assert row["infra_failed_trials"] == 0
    # The same numbers reach the tracker, which is what puts them on a chart.
    assert all(metrics.scaffold_loss_rate == 1.0 for _, metrics in tracker.steps)
    assert all(metrics.stop_reason_counts == {"no_tool_call": 4} for _, metrics in tracker.steps)


def test_a_submitting_run_reports_zero_scaffold_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)

    _run(env, _cfg())

    rows = [row for row in DistillRunStore(env.run_dir).read_metrics() if "phase" not in row]
    for row in rows:
        assert row["scaffold_loss_rate"] == 0.0
        assert row["stop_reason_counts"] == {"submitted": 4}


def test_the_cli_progress_line_shows_the_scaffold_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    env.rollouts.stop_reason = StopReason.OUTPUT_TRUNCATED.value

    _run(env, _cfg())

    training = [event.message for event in env.progress if event.phase == "training"]
    assert training
    assert all("scaffold loss 100%" in message for message in training)


# --- an eval where nothing ran is a null measurement, not 0.0% (audit defect 5) ------------------
def test_an_all_infra_failed_eval_refuses_to_become_a_zero_solve_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three Super `baseline-student-before` reports were written as 0.0% from 51/51 rate-limited
    trials, then used as the no-regression leg of a promotion gate. That must stop the run."""
    env = _setup(tmp_path, monkeypatch)
    env.rollouts.infra_fail_all = True

    with pytest.raises(DistillNullEvalError, match="NULL measurement"):
        _run(env, _cfg())

    # Nothing was persisted for the gate to read back.
    evals_dir = DistillRunStore(env.run_dir).evals_dir
    assert not (evals_dir / f"{TEACHER_BASELINE_EVAL}.json").exists()


def test_an_eval_report_records_the_infra_and_scaffold_breakdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    env.rollouts.stop_reason = StopReason.MAX_TURNS.value

    _run(env, _cfg())

    path = DistillRunStore(env.run_dir).evals_dir / f"{STUDENT_AFTER_EVAL}.json"
    report = DistillEvalReport.model_validate_json(path.read_text(encoding="utf-8"))
    assert report.executed_trials == report.trials
    assert report.infra_failed_trials == 0
    assert report.scaffold_loss_rate == 1.0
    assert report.stop_reason_counts == {"max_turns": report.trials}


def test_sampler_refresh_warms_the_new_weights_before_returning() -> None:
    """Every refresh must serve one throwaway token before the batch launches.

    Freshly published weights are cold; launching 64 episodes at once means every first call
    races the same load, and the losers exhaust their retries and report `provider_error` at turn
    0-2 while the survivors run 28 turns cleanly. Measured live on a `baseline-student-before`
    wave: 7 of 51 episodes (~15%) died that way at a mean of 2.0 turns, against 6% for the same
    model sampled from already-warm BASE weights in a 48-episode probe. One serial call absorbs
    the cold start once instead of 64 episodes paying for it in parallel.
    """
    service = _Service()
    sampler = StudentSampler(service, service.create_lora_training_client(_STUDENT, 8), _NAME)

    sampler.refresh(0)

    # `client` is typed as the protocol slice the loop uses; the fake's issued-span ledger is
    # what records the warmup, so narrow to it rather than widening the protocol for a test.
    client = sampler.client
    assert isinstance(client, FakeSamplingClient)
    assert client.issued, "refresh() returned without warming the sampler"
    warmup = client.issued[0]
    assert len(warmup.sampled_ids) == 1, "the warmup must be one throwaway token, not a rollout"


def test_a_sampler_that_cannot_warm_still_yields_a_path(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Warmup is best-effort: aborting a paid run inside it trades a partial batch for none."""

    def refuse(*_args: object, **_kwargs: object) -> NoReturn:
        raise RuntimeError("weights still loading")

    monkeypatch.setattr(FakeSamplingClient, "sample", refuse)
    service = _Service()
    sampler = StudentSampler(service, service.create_lora_training_client(_STUDENT, 8), _NAME)

    with caplog.at_level(logging.WARNING, logger=loop_module.__name__):
        path = sampler.refresh(0)

    assert path.startswith("tinker://")
    assert "did not serve a token" in caplog.text


def test_warmup_exercises_concurrency_not_just_one_token() -> None:
    """The warmup must issue several CONCURRENT calls, not one serial call.

    A single serial token woke the model but a live 64-episode training wave still lost 7 of its
    first 34 episodes (~21%) at a mean of 0.7 turns. The cost is the sampler's ramp to serving
    many simultaneous streams, not one weight load, so a warmup that never fans out cannot
    prevent it.
    """
    service = _Service()
    sampler = StudentSampler(service, service.create_lora_training_client(_STUDENT, 8), _NAME)

    sampler.refresh(0)

    client = sampler.client
    assert isinstance(client, FakeSamplingClient)
    assert len(client.issued) == _WARMUP_STREAMS > 1, (
        "warmup must fan out; one call does not trigger the sampler's ramp"
    )
    assert all(len(span.sampled_ids) == 1 for span in client.issued), (
        "each warmup stream must stay a single throwaway token"
    )
