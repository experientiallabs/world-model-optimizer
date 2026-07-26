"""The on-policy distillation orchestrator: `run_distillation` and its step loop.

One run couples every distill layer end to end: preflight checks before any
spend, teacher-in-harness and student-before baselines on the holdout split,
the optional off-policy phase (`[offpolicy]`: teacher rollouts on the train
split filtered to `offpolicy.keep`, merged into cross_entropy datums, and
trained for `offpolicy.epochs` passes at `offpolicy.minibatch_datums` datums
per optimizer step, resumable at datum granularity through
`offpolicy-cursor.json`; `offpolicy.trajectories_from` loads another run's
recorded collection instead of collecting, charging nothing, and the legacy
`[warmup]` section is the same machinery pinned to full-batch passes with no
cursor), then the training loop
(harbor rollouts from the current student sampler, prefix-merge datums,
teacher scoring, reverse-KL advantages, one optimizer step per training step
under the loss `train.loss` selects: `importance_sampling` or `ppo` over the
same advantage-carrying datums, or `topk_ce`, where the teacher instead top-k
scores each datum in one prefill-only request and the step trains rank-aligned
weighted cross_entropy replicas, the reverse-KL metric still coming from that
same request's realized logprobs), and finally the holdout gate that decides
whether the adapter is promoted into the `AdapterStore`.

Layout: everything lands in the `DistillRunStore` run directory. The rollout
collector writes its per-step `harbor/step-NNNN/` jobs dir under the run dir
for training batches (each trial's sampled token spans live inside its own
harbor trial dir, in `result.json`); eval batches (baselines, interim
evals, student-after) get their own isolated roots under
`eval-rollouts/<eval-name>/` so their harbor job dirs never collide with a
training step's, and a cross_entropy phase's teacher trials likewise land
under `warmup-rollouts/` (the name both phases share, with their assembled
records in `warmup-trials.json`). Every batch (training step, teacher
collection, eval)
additionally renders its first `train.log_sample_rollouts` episodes to
human-readable text under `samples/` and the tracker's samples table
(`wmh.distill.samples`), chat-template framing included.

The Tinker SDK stays an optional extra: the real service client is built
lazily only when no `service_client` is injected, and the thin `Sdk*`
adapters here (mirroring `SdkSampler` and `SdkLogprobScorer`) are the only
code that touches it. Every SDK call is deadline-bounded
(`wmh.distill.deadlines`), so a wedged session raises a typed
`TinkerDeadlineError` instead of hanging; see `SdkTrainingClient` for the
per-call idempotency reasoning that decides what expiry does. Tests drive
the whole loop with the deterministic fakes in `wmh.distill.fake_tinker`,
whose `FakeTrainingClient` asserts the tokens-in-tokens-out invariant on
every `forward_backward` batch.

Resume: cadenced `save_state` checkpoints land in the run store's manifest;
`resume=True` restores the latest checkpoint via `load_state` as the very
first call on a freshly created training client (tinker accepts LoadWeights
only on an uninitialized model, so preflight and the sampler refresh follow
the restore and validate the RESTORED weights), continues the
step count from the checkpoint, restores the prior sessions' USD spend from
the run store's spend ledger (written on every charge, so eval spend between
metrics rows survives), and reuses recorded baseline evals. A finished
cross_entropy phase is recorded as `offpolicy.json` (or the legacy
`warmup.json`) and never re-runs; when no step checkpoint exists yet, that
record's state is what `load_state` restores. An INTERRUPTED off-policy phase
resumes from its `offpolicy-cursor.json`: the cursor names the training state
saved at the last checkpointed optimizer step and how many steps of the
schedule are already applied, so the phase continues at the next minibatch and
reuses the teacher collection recorded in `warmup-trials.json` rather than
paying for it again. An unfinished WARMUP has no cursor and re-runs whole,
with the teacher trials themselves resuming trial-level through harbor (the
teacher's identity is stable across sessions). Steps and student evals whose harbor job dirs
were left by a prior session re-run whole (the rollout collector wipes
stale-policy job dirs). A budget abort (`DistillBudgetError`), an
all-empty-batch abort (`DistillEmptyBatchError`, raised when consecutive
training steps show the student provider producing no completions) and a
degeneration abort (`DistillDegenerationError`, below) all carry the exact
resume command.

Degeneration tripwires: every step also pools the student's own sampled
logprobs and generation lengths into `entropy_per_token` and
`mean_generation_tokens` (`wmh.distill.tripwire`) and compares them against a
baseline THIS run measured at its first training step, which is persisted in
the checkpoint manifest so a resumed session never re-anchors on an already
degenerated policy. Breaches warn, and a kill-level streak aborts the run the
same way the budget path does. The thresholds are fractions of that measured
baseline, never absolute values; `TripwireConfig` records why.
"""

from __future__ import annotations

import hashlib
import logging
import random
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NoReturn, Protocol, cast, runtime_checkable
from uuid import uuid4

from llm_waterfall.types import ChatMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wmh.config.store import validate_name
from wmh.distill.config import DistillConfig, PricingConfig
from wmh.distill.cost import (
    METER_NAMES,
    BudgetExhausted,
    BudgetMeter,
    CostLine,
    MeterName,
    SpanBilling,
    batch_billing,
)
from wmh.distill.data import (
    DatumStats,
    TopkCandidates,
    TrainDatum,
    attach_advantages,
    build_datums,
    build_topk_ce_datums,
    to_tinker_datums,
    to_tinker_sft_datums,
)
from wmh.distill.deadlines import TinkerDeadlineError, call_with_deadline, wait_with_deadline
from wmh.distill.gate import DistillGateRecord, gate_distillation
from wmh.distill.offpolicy import (
    CROSS_ENTROPY_LOSS,
    OffPolicyCursorPosition,
    OffPolicyMetrics,
    OffPolicySchedule,
    OffPolicyStepResult,
    plan_offpolicy_steps,
    run_offpolicy,
)
from wmh.distill.rendering import ChatRendering, RendererTokenizer, build_renderer
from wmh.distill.rollouts import RolloutStats, collect_rollouts, rollout_stats
from wmh.distill.samples import sample_rollouts, samples_markdown
from wmh.distill.store import (
    AdapterStore,
    DistillModelCard,
    DistillRunStore,
    OffPolicyCursor,
    OffPolicyRecord,
    WarmupRecord,
    WarmupTrialsManifest,
    build_handoff_toml,
)
from wmh.distill.teacher import (
    EncodingTokenizer,
    SdkLogprobScorer,
    TeacherClient,
    TinkerTeacher,
    tokenizer_fingerprint_check,
)
from wmh.distill.tokens import TrialRecord
from wmh.distill.tracking import DistillTracker, build_tracker
from wmh.distill.tripwire import (
    PolicyHealth,
    TripwireBaseline,
    TripwireBreach,
    capture_baseline,
    evaluate_breaches,
    health_summary,
    metric_ratio,
    policy_health,
)
from wmh.harness.doc import (
    MAX_OUTPUT_TOKENS_ID,
    MAX_TURNS_ID,
    TEMPERATURE_ID,
    HarnessDoc,
    Surface,
    SurfaceKind,
)
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.providers.tinker import (
    TINKER_API_KEY_ENV,
    SampledSequenceLike,
    SdkSampler,
    evict_shared_sampling_client,
    shared_sampling_client,
    shared_service_client,
)

if TYPE_CHECKING:
    import tinker
    from tinker import types as tinker_types

logger = logging.getLogger(__name__)

_WARMUP_STREAMS = 8
"""Concurrent throwaway calls issued against freshly published sampler weights.

Not 1: a single serial token woke the model but a 64-episode wave still lost ~21% of its
episodes at a mean of 0.7 turns, because the cost is the sampler's ramp to many simultaneous
streams, not one weight load. Not 64: the point is to trigger the ramp, and the wave itself
finishes it, so this trades a few seconds of warmup for the tail of a 45-step run."""

IMPORTANCE_SAMPLING_LOSS = "importance_sampling"
PPO_LOSS = "ppo"
"""The clipped-ratio surrogate over the SAME datums as importance_sampling.

Both are values of the SDK's `types.LossFnType` (tinker 0.23.3) and both take
exactly the `to_tinker_datums` keyset; `ppo` differs only in what the service
does with the advantages: it bounds the update by clipping the policy ratio
(service-side epsilon, no `loss_fn_config` is sent) instead of relying on a
bounded advantage, which is why `train.loss = "ppo"` is the mode that wants
`advantage_clip` unset and `center_advantages = false`. With one
forward/backward per batch off a sampler refreshed every step, that ratio sits
at ~1 and the clip rarely binds (see `TrainConfig.loss`); `advantage_std` and
`grad_norm` in the metrics row are what actually show an outlier-driven step.
"""

ADVANTAGE_LOSS_BY_MODE = {
    "importance_sampling": IMPORTANCE_SAMPLING_LOSS,
    "ppo": PPO_LOSS,
}
"""Wire `loss_fn` per `train.loss` mode that trains advantages (topk_ce is CE)."""

DEFAULT_TITO_MEAN_TOLERANCE = 0.25
"""Max mean |issued - recomputed| logprob gap the preflight recompute accepts.

The fakes agree exactly, but the real service computes sampling and scoring on
different execution paths (kernel and batching differences), so individual
positions legitimately drift by a few hundredths of a nat (observed ~0.08 on
Qwen3.5-4B at temperature 1.0). That noise is zero-mean across the probe
sample, while the failures this check exists to catch (wrong sampler path,
tokenizer or renderer drift, SDK change) shift logprobs systematically by
whole nats. The mean bound separates those regimes robustly.
"""

DEFAULT_TITO_MAX_TOLERANCE = 1.0
"""Max |issued - recomputed| gap at any single position (catastrophic bound)."""

_PREFLIGHT_SAMPLE_TOKENS = 16
"""Length of the preflight sample whose logprobs are recomputed (the TITO proof)."""

EVAL_ROLLOUTS_DIR = "eval-rollouts"
"""Run-dir subdirectory holding each eval batch's isolated rollout root."""

WARMUP_ROLLOUTS_DIR = "warmup-rollouts"
"""Run-dir subdirectory holding a cross_entropy phase's teacher rollout root.

Shared by the off-policy mode and the legacy warmup phase, the same way both
read and write the one `warmup-trials.json` corpus manifest, so either can
resume or load the other's collection.
"""

TEACHER_BASELINE_EVAL = "baseline-teacher"
STUDENT_BEFORE_EVAL = "baseline-student-before"
STUDENT_AFTER_EVAL = "student-after"

DistillPhase = Literal[
    "preflight",
    "baseline",
    "offpolicy",
    "warmup",
    "rollouts",
    "training",
    "eval",
    "finalize",
    "gate",
]


class DistillProgress(BaseModel):
    """One typed progress event emitted to the `on_progress` callback."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    phase: DistillPhase
    message: str
    total_steps: int = Field(ge=0)
    step: int | None = None
    """0-based training step the event belongs to; None for run-level phases."""

    spent_usd: float = Field(ge=0.0)
    """Total priced spend so far, including prior sessions of a resumed run."""


ProgressCallback = Callable[[DistillProgress], None]

LiveTrialPreflight = Callable[[ProviderConfig], None]
"""The documented hook for the live single-trial pi preflight (Phase 6).

Called with the student's rollout provider config (kind TINKER, model = the
current sampler path) after every cheap preflight check has passed and before
any baseline or training spend. A real implementation runs one short pi trial
on a cheap task through the actual harbor stack and raises on failure (spans
missing, prefix property broken, tool calls not round-tripping); it is
deliberately NOT implemented in this module, which stays free of harbor
calls beyond `collect_rollouts`.
"""


class DistillBudgetError(RuntimeError):
    """The hard budget cap was hit; artifacts were persisted so the run can resume.

    Attributes:
        resume_command: The exact CLI command that resumes this run.
        spent_usd: Total priced spend (including prior sessions) at abort.
        max_usd: The configured `budget.max_usd` cap.
    """

    def __init__(
        self, message: str, *, resume_command: str, spent_usd: float, max_usd: float
    ) -> None:
        super().__init__(message)
        self.resume_command = resume_command
        self.spent_usd = spent_usd
        self.max_usd = max_usd


MAX_CONSECUTIVE_EMPTY_STEPS = 2
"""Consecutive all-empty training steps tolerated before the run aborts.

An all-empty step ran trials (`trials > 0`) but every trial produced zero
token spans (`empty_span_trials == trials`) and therefore zero datums: the
student provider is producing no completions at all, so training cannot make
progress and further steps only burn rollout budget. One such step can be a
transient batch-wide outage that self-heals; two in a row means the provider
is down (the live failure mode was silently swallowed worker errors upstream)
and the run aborts with `DistillEmptyBatchError`. The streak is tracked
within one session and a non-empty step resets it.
"""


class DistillEmptyBatchError(RuntimeError):
    """Consecutive training steps produced no completions; the run aborted.

    Mirrors `DistillBudgetError` ergonomics: state was checkpointed and the
    error carries the exact resume command.

    Attributes:
        resume_command: The exact CLI command that resumes this run.
        consecutive_steps: How many all-empty steps ran back to back.
    """

    def __init__(self, message: str, *, resume_command: str, consecutive_steps: int) -> None:
        super().__init__(message)
        self.resume_command = resume_command
        self.consecutive_steps = consecutive_steps


class DistillDegenerationError(RuntimeError):
    """The policy degenerated past a tripwire's kill bound; the run aborted.

    Mirrors `DistillEmptyBatchError` ergonomics: state was checkpointed and the
    error carries the exact resume command (a resumed run restarts from the last
    healthy checkpoint and keeps the SAME persisted baseline).

    Attributes:
        resume_command: The exact CLI command that resumes this run.
        consecutive_steps: How many kill-level steps ran back to back.
        breaches: The final step's kill-level breaches, in report order.
    """

    def __init__(
        self,
        message: str,
        *,
        resume_command: str,
        consecutive_steps: int,
        breaches: Sequence[TripwireBreach],
    ) -> None:
        super().__init__(message)
        self.resume_command = resume_command
        self.consecutive_steps = consecutive_steps
        self.breaches = tuple(breaches)


class DistillNullEvalError(RuntimeError):
    """An eval batch produced no verifier evidence at all, so it has no solve rate.

    Raised instead of writing a `DistillEvalReport` when no trial in the batch was graded, either
    because none ran (the live failure was the E2B concurrent-sandbox cap: 51/51 trials
    rate-limited, harbor jobs finishing in 52 to 199 seconds, and three `baseline-student-before`
    reports written as 0.0%, then used as the no-regression leg of a promotion gate) or because
    the verifier never produced a reward for any of them. A null measurement must stop the run,
    not become a number.
    """


class DistillEvalReport(BaseModel):
    """One eval batch's persisted outcome (baselines, interim evals, student-after)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    provider_model: str = Field(min_length=1)
    """The provider `model` the trials sampled (sampler path or teacher ref)."""

    base_model: str | None = None
    """The base model behind `provider_model` (the provider's `model_type`).

    This is what `eval.student_baseline_from` reuse validates against: a
    student's `provider_model` is a per-run sampler path and never matches
    across runs, while the base model is exactly what a shared pre-training
    baseline must agree on. None only on reports from before this field.
    """

    task_ids: list[str]
    attempts: int = Field(ge=1)
    trials: int = Field(ge=0)
    solve_rate: float = Field(ge=0.0, le=1.0)
    """Passing share of EXECUTED trials (infrastructure failures excluded)."""

    graded_solve_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    """Mean graded test-pass score over `graded_trials`, beside the binary `solve_rate`.

    The power metric for comparing two runs on a small holdout, where binary has no resolution;
    `solve_rate` stays the headline and the gate's number. 0.0 with `graded_trials == 0` is a null
    measurement, not a score. See `RolloutStats.graded_solve_rate`. 0.0 on reports from before this
    field."""

    graded_trials: int = Field(default=0, ge=0)
    """Gradeable trials that carried a readable test report: the `graded_solve_rate`
    denominator."""

    empty_span_trials: int = Field(ge=0)

    executed_trials: int = Field(default=0, ge=0)
    """Trials that produced verifier evidence; 0 only on reports from before this field."""

    infra_failed_trials: int = Field(default=0, ge=0)
    """Trials excluded from `solve_rate` because no verifier reward exists for them.

    Either the agent never ran or its work was never graded; see `RolloutStats.infra_failed_trials`
    for why one count covers both and where the per-trial causes live."""

    scaffold_loss_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    """Share of executed trials that never reached an explicit submit."""

    stop_reason_counts: dict[str, int] = Field(default_factory=dict)
    """Trials per recorded stop reason."""

    source: str | None = None
    """Provenance note when the report was imported from a prior run via
    `eval.teacher_baseline_from` / `eval.student_baseline_from` instead of
    measured here; None for reports this run's own trials produced."""


class StepMetrics(BaseModel):
    """One training step's metrics row (the store adds the `step` key).

    The row reports two accounting stages, and mixing them in a ratio is a
    measurement bug. `fragments`, `fragmentation_rate`, `overflow_drops`, and
    `overlong_drops` describe the BUILD stage: rollouts merged into datums,
    before the teacher scored anything. Every other datum or token count
    (`datums`, `mismatch_drops`, `clipped_tokens`, `loss_tokens`,
    `context_tokens`, `clip_fraction`, the advantage stats, `pg_loss`,
    `grad_norm`) describes the TRAINED batch, after the teacher's misaligned
    rows were dropped, so `clipped_tokens / loss_tokens` equals
    `clip_fraction` and the token counts are what forward_backward consumed.
    The pre-drop volume is not duplicated here: the teacher scored every built
    datum, so `teacher_prefill_tokens` already carries it, and
    `mismatch_drops` says how many datums the gap cost.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tasks: int = Field(ge=0)
    trials: int = Field(ge=0)
    solve_rate: float = Field(ge=0.0, le=1.0)
    """Passing share of EXECUTED trials (infrastructure failures excluded)."""

    graded_solve_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    """Mean graded test-pass score over `graded_trials`, beside the binary `solve_rate`.

    Extra resolution for run-to-run comparison, never a replacement: `solve_rate` is the benchmark's
    own definition and stays the headline. 0.0 with `graded_trials == 0` is a null measurement. See
    `RolloutStats.graded_solve_rate`. 0.0 on rows from before this field."""

    graded_trials: int = Field(default=0, ge=0)
    """Gradeable trials that carried a readable test report: the `graded_solve_rate`
    denominator."""

    raw_solve_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    """Passing share of ALL trials, which is what advantage estimation sees."""

    executed_trials: int = Field(default=0, ge=0)
    infra_failed_trials: int = Field(default=0, ge=0)
    """Trials with no verifier reward (nothing ran, or nothing graded it); not in `solve_rate`."""

    scaffold_loss_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    """Share of executed trials that never reached an explicit submit.

    The audit headline: 88.8% for Super, invisible in every artifact. A rise here means the harness,
    not the model, is deciding the solve rate."""

    stop_reason_counts: dict[str, int] = Field(default_factory=dict)
    """Trials per recorded stop reason (`submitted`, `max_turns`, `budget`, `no_tool_call`,
    `output_truncated`, `unparsed_tool_call`, `provider_error`, `unknown`)."""

    empty_span_trials: int = Field(ge=0)
    truncated_spans: int = Field(default=0, ge=0)
    """Turns that sampled the full `sampling.max_tokens` and were cut off mid-answer.

    Charts as `train/truncated_spans`. Nothing else reports it: harbor's own truncation guard is
    unreachable (see `RolloutStats.truncated_spans`), so without this series a run whose output
    cap is too low reads as a run whose model writes broken actions. 0 on rows from before this
    field."""

    datums: int = Field(ge=0)
    """Datums the optimizer step trained on; under `train.loss = "topk_ce"`
    these are the rank replicas, so k x the surviving source datums."""

    fragments: int = Field(ge=0)
    fragmentation_rate: float = Field(ge=0.0, le=1.0)
    overflow_drops: int = Field(ge=0)
    overlong_drops: int = Field(ge=0)
    mismatch_drops: int = Field(ge=0)
    clipped_tokens: int = Field(ge=0)
    """Trained loss tokens whose raw advantage hit the clip bound (0 under
    `topk_ce`, which clips nothing, and 0 when `train.advantage_clip` is
    unset)."""

    loss_tokens: int = Field(ge=0)
    """Sampled (mask 1.0) tokens in the trained batch, counted once per SOURCE
    datum: under `topk_ce` the k rank replicas share one source's tokens, so
    `student_train_tokens` is k x (`loss_tokens` + `context_tokens`)."""

    context_tokens: int = Field(ge=0)
    """Non-loss tokens in the trained batch, on `loss_tokens`' convention."""

    reverse_kl_per_token: float | None
    """mean(sampled_lp - teacher_lp) over every SCORED loss token; None when
    none. This is the teacher-scoring stage's measurement, not the trained
    batch's: a datum the teacher scored but the alignment check later dropped
    still contributed the positions that did come back."""

    entropy_per_token: float | None
    """`-mean(sampled_logprobs)` over the batch's loss positions, pooled.

    The degeneration tripwire's first signal, and the one a KL curve cannot
    give: reverse KL can fall while the policy collapses into a single mode.
    The tokens came from the policy that scored them, so this is an unbiased
    single-sample estimator of that policy's entropy, pooled over the batch.
    Read it as a LOWER bound on the T=1 entropy, since rollouts sample at
    `sampling.temperature` (0.7 for the headline run) and that concentrates the
    distribution the tokens are drawn from. Pooled over the whole batch, never
    per episode (`wmh.distill.tripwire.policy_health` says why). None when the
    batch recorded no sampled token."""

    mean_generation_tokens: float | None
    """Sampled tokens per episode, pooled over the batch's span-bearing episodes.

    The tripwire's second signal, against the mirror pathology: pure KL gives
    EOS no gradient, so a student either never learns to stop or collapses to
    near-empty answers (a sibling lane fell 2,866 to about 50 tokens). Measured
    from the recorded spans, so whole-episode datum drops (the longest episodes)
    cannot deflate it. None when no episode recorded a sampled token."""

    entropy_baseline: float | None
    """`entropy_per_token` as measured at this run's first training step; None
    until the baseline is captured. Every tripwire bound is a fraction of this,
    never an absolute nats value (see `TripwireConfig`)."""

    entropy_ratio: float | None
    """`entropy_per_token / entropy_baseline`: the number the tripwire bounds.
    1.0 at the baseline step; None until a baseline exists."""

    generation_tokens_baseline: float | None
    """`mean_generation_tokens` at this run's first training step; None until
    the baseline is captured."""

    generation_tokens_ratio: float | None
    """`mean_generation_tokens / generation_tokens_baseline`; None until a
    baseline exists."""

    reward_mean: float | None
    """Mean verifier reward over the step's trials; None when no trial ran.

    Harbor's verifier rewards are binary today, so this equals `solve_rate`
    and only diverges if a verifier ever emits fractional rewards.
    """

    loss: Literal["importance_sampling", "ppo", "topk_ce"]
    """Which objective this step trained (`train.loss`), so a metrics row and
    a dashboard point say what produced them.

    `importance_sampling` and `ppo` submit the same advantage-carrying datums
    and differ only in the service-side loss (`ppo` clips the policy ratio);
    `topk_ce` submits rank-aligned replicas under the `cross_entropy` wire
    loss and carries no advantage metrics at all."""

    advantage_mean: float | None
    """Mean advantage over the trained loss tokens, exactly as trained (after
    any clipping and any centering); None when nothing was trained or the
    mode builds no advantages (`topk_ce`).

    With the default objective (no clip, no centering) this is the mean
    teacher-minus-student gap over the trained tokens, i.e. the negated
    reverse KL: the number that should move toward 0 as the student closes on
    the teacher. Under `train.center_advantages` it is ~0.0 by construction
    and carries no information."""

    advantage_std: float | None
    """Population std over the same loss tokens; None when nothing was trained.

    Read it next to `advantage_mean` under the unclipped objective: it is the
    per-token magnitude the gradient actually sees, and a run whose std grows
    without its mean moving is being driven by outlier tokens."""

    clip_fraction: float = Field(ge=0.0, le=1.0)
    """Trained loss tokens whose raw advantage hit the clip bound, as a
    fraction of the trained loss tokens (0.0 when nothing was trained, and
    0.0 for the whole run when `train.advantage_clip` is unset)."""

    pg_loss: float | None
    """The batch loss the training backend reported for the step's
    forward/backward; None when no optimizer step ran or the backend reported
    no loss metric (tinker 0.23.3 has no typed loss field; see
    `SDK_LOSS_METRIC_NAMES`)."""

    grad_norm: float | None
    """The gradient norm the optim step reported; None when no optimizer step
    ran or the backend reported none (see `SDK_GRAD_NORM_METRIC_NAMES`)."""

    sampler_path: str
    """The tinker:// sampler path the step's rollouts sampled from."""

    student_prefill_tokens: int = Field(ge=0)
    student_cached_prefill_tokens: int = Field(ge=0)
    student_sample_tokens: int = Field(ge=0)
    student_train_tokens: int = Field(ge=0)
    teacher_prefill_tokens: int = Field(ge=0)
    teacher_cached_prefill_tokens: int = Field(ge=0)
    teacher_sample_tokens: int = Field(ge=0)
    usd: float = Field(ge=0.0)
    """Priced spend since the previous metrics row (spend before the first row,
    the baselines and any warmup collection, folds into that first row)."""

    cumulative_usd: float = Field(ge=0.0)
    """Total priced spend through this row, across every session of the run
    (the budget meter's total, matching the spend ledger)."""


class WarmupMetrics(BaseModel):
    """One warmup step's metrics row (the store adds the `step` key).

    Warmup rows share `metrics.jsonl` with training rows and their step
    indices restart at 0 for OPD, so the constant `phase` field is what
    distinguishes them (training rows carry no `phase` key). A run whose
    warmup was skipped (zero kept trials) writes exactly one row with
    `datums = 0` so the degradation to pure OPD is visible in the metrics.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    phase: Literal["warmup"] = "warmup"
    tasks: int = Field(ge=0)
    trials: int = Field(ge=0)
    """Teacher trials collected on the train split (task x rollouts_per_task)."""

    kept_trials: int = Field(ge=0)
    """Trials that survived the `warmup.keep` filter."""

    solve_rate: float = Field(ge=0.0, le=1.0)
    """The teacher's solve rate over the collected warmup trials."""

    datums: int = Field(ge=0)
    loss_tokens: int = Field(ge=0)
    context_tokens: int = Field(ge=0)
    learning_rate: float = Field(gt=0)
    """The effective warmup LR (warmup.learning_rate or train.learning_rate)."""

    student_prefill_tokens: int = Field(ge=0)
    student_cached_prefill_tokens: int = Field(ge=0)
    student_sample_tokens: int = Field(ge=0)
    student_train_tokens: int = Field(ge=0)
    teacher_prefill_tokens: int = Field(ge=0)
    teacher_cached_prefill_tokens: int = Field(ge=0)
    teacher_sample_tokens: int = Field(ge=0)
    usd: float = Field(ge=0.0)
    """Priced spend since the previous metrics row (the teacher collection and
    any earlier baseline spend fold into warmup step 0's row)."""


class SpendSummary(BaseModel):
    """Where the run's money went, in the cost module's line shape."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lines: list[CostLine]
    """This session's actual per-meter tokens and USD."""

    session_usd: float = Field(ge=0.0)
    prior_usd: float = Field(ge=0.0)
    """USD earlier sessions of a resumed run recorded in the spend ledger."""

    total_usd: float = Field(ge=0.0)


class DistillResult(BaseModel):
    """What `run_distillation` returns after the gate has been decided."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    run_dir: str
    steps_completed: int = Field(ge=0)
    final_sampler_path: str
    final_state_path: str
    gate: DistillGateRecord
    adapter_version: int | None
    """The promoted AdapterStore version; None when the gate rejected."""

    spend: SpendSummary


# -- injectable Tinker surface (fakes satisfy these directly) ------------------------------------


SDK_LOSS_METRIC_NAMES = frozenset({"loss", "total_loss"})
"""Metric names accepted as the batch loss in a tinker metrics dict.

The pinned SDK (tinker 0.23.3) exposes no typed loss field anywhere:
`ForwardBackwardOutput` carries exactly `loss_fn_output_type`, the per-datum
`loss_fn_outputs` (field name -> TensorData; the cookbook reads a per-datum
"logprobs" tensor from it), and a server-populated `metrics: dict[str, float]`
whose only documented keys are MoE routing diagnostics. The cookbook's
off-policy distillation reads a `total_loss` key from that dict, so these
spellings (bare, or with the SDK chunk combiner's ":reduction" suffix) are
what `sdk_metric_value` recognizes; anything else stays un-surfaced.
"""

SDK_GRAD_NORM_METRIC_NAMES = frozenset({"grad_norm"})
"""Metric names accepted as the gradient norm in a tinker metrics dict.

`OptimStepResponse` (tinker 0.23.3) carries only an untyped
`metrics: Optional[Dict[str, float]]` with no documented keys, and no grad
norm appears anywhere in the SDK or the cookbook; the value is surfaced only
if the service ever reports one under this name.
"""


def sdk_metric_value(metrics: Mapping[str, float] | None, names: frozenset[str]) -> float | None:
    """Pull one named scalar out of a tinker metrics dict, if it is present.

    Server metric keys carry a ":reduction" suffix (e.g. "total_loss:sum")
    that the SDK's chunk combiner folds over, so the match is on the name
    part before an optional suffix.

    Args:
        metrics: The SDK output's metrics mapping (None on some responses).
        names: The metric names to accept.

    Returns:
        The first matching value in mapping order, or None when the backend
        reported no such metric (never a fabricated stand-in).
    """
    if not metrics:
        return None
    for key, value in metrics.items():
        if key.split(":", 1)[0] in names:
            return float(value)
    return None


class TrainStepOutput(BaseModel):
    """What one `forward_backward` reports back, in loop currency.

    Only real backend-reported values are ever set: `SdkTrainingClient`
    extracts `loss` from `ForwardBackwardOutput.metrics` (see
    `SDK_LOSS_METRIC_NAMES` for exactly what the SDK exposes), and None means
    the backend reported nothing, which the loop logs as an absent metric.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    loss: float | None = None
    """The batch loss the backend reported; None when it reported none."""


class OptimStepOutput(BaseModel):
    """What one `optim_step` reports back, in loop currency.

    `SdkTrainingClient` extracts `grad_norm` from `OptimStepResponse.metrics`
    (see `SDK_GRAD_NORM_METRIC_NAMES`); None means the backend reported
    nothing, which the loop logs as an absent metric.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    grad_norm: float | None = None
    """The gradient norm the backend reported; None when it reported none."""


class DistillSamplingClient(Protocol):
    """The sampling-client slice the loop uses, in token-id terms.

    `wmh.distill.fake_tinker.FakeSamplingClient` satisfies this directly; real
    `tinker.SamplingClient`s are adapted via `SdkSamplingClient`.
    """

    def sample(
        self,
        prompt_token_ids: list[int],
        *,
        max_tokens: int,
        temperature: float,
    ) -> SampledSequenceLike:
        """Sample one sequence conditioned on the prompt token ids."""
        ...

    def compute_logprobs(self, token_ids: list[int]) -> list[float | None]:
        """Per-position logprobs for the sequence; entry 0 is None."""
        ...


@runtime_checkable
class TokenizerSource(Protocol):
    """A sampling client that can also supply its model's tokenizer."""

    def get_tokenizer(self) -> EncodingTokenizer:
        """The tokenizer for the client's model."""
        ...


class DistillTrainingClient(Protocol):
    """The training-client slice the loop drives, in loop currency.

    `forward_backward` takes the loop's own `TrainDatum`s; each backend owns
    the conversion to its wire datum type (`SdkTrainingClient` converts via
    `to_tinker_datums`, test shims convert to `FakeDatum`s), which keeps the
    orchestrator identical for real and fake runs.
    """

    def get_tokenizer(self) -> EncodingTokenizer:
        """The student base model's tokenizer."""
        ...

    def forward_backward(self, datums: Sequence[TrainDatum], loss_fn: str) -> TrainStepOutput:
        """Accumulate gradients for one batch under the named loss.

        Returns:
            The backend-reported step output; `loss` is None when the
            backend reported no loss metric (never fabricated).
        """
        ...

    def optim_step(self, learning_rate: float) -> OptimStepOutput:
        """Apply one optimizer step.

        Returns:
            The backend-reported output; `grad_norm` is None when the
            backend reported no such metric (never fabricated).
        """
        ...

    def save_state(self) -> str:
        """Save resumable training state, returning its tinker:// path."""
        ...

    def load_state(self, path: str) -> None:
        """Restore training state from a previously saved path."""
        ...

    def save_weights_for_sampler(self, name: str) -> str:
        """Save current weights for sampling, returning the sampler path."""
        ...


class DistillServiceClient(Protocol):
    """The service-client slice the loop needs.

    `wmh.distill.fake_tinker.FakeServiceClient` matches this shape (tests wrap
    its training client to convert datums); the real SDK is adapted by
    `SdkServiceClient`.
    """

    def create_lora_training_client(self, base_model: str, rank: int = 32) -> DistillTrainingClient:
        """Create the LoRA training client for the student base model."""
        ...

    def create_sampling_client(self, model_path: str) -> DistillSamplingClient:
        """Create a sampling client for a sampler path or base model name."""
        ...


# -- real-SDK adapters (lazy tinker imports, mirroring SdkSampler/SdkLogprobScorer) --------------


class SdkSamplingClient:
    """Adapts a real `tinker.SamplingClient` to `DistillSamplingClient`.

    Args:
        client: The SDK sampling client, normally fetched from the
            process-wide shared cache in `wmh.providers.tinker`.
        model: The shared-cache key the client was fetched under. When set, a
            `TinkerDeadlineError` from any call evicts that cache entry so
            every future user of the model string (including the harbor trial
            providers sharing the cache) rebuilds through a fresh session;
            None disables eviction for a directly constructed adapter.
    """

    def __init__(self, client: tinker.SamplingClient, *, model: str | None = None) -> None:
        self._client = client
        self._model = model
        self._sampler = SdkSampler(client)
        self._scorer = SdkLogprobScorer(client)

    def _evict_on_deadline(self) -> None:
        """Evict the wedged client from the shared cache (see class docstring)."""
        if self._model is not None:
            evict_shared_sampling_client(self._model, self._client)

    def sample(
        self,
        prompt_token_ids: list[int],
        *,
        max_tokens: int,
        temperature: float,
    ) -> SampledSequenceLike:
        """One synchronous sample through the provider's SDK adapter."""
        try:
            return self._sampler.sample(
                prompt_token_ids, max_tokens=max_tokens, temperature=temperature
            )
        except TinkerDeadlineError:
            self._evict_on_deadline()
            raise

    def compute_logprobs(self, token_ids: list[int]) -> list[float | None]:
        """One synchronous compute_logprobs call on the full sequence."""
        try:
            return self._scorer.compute_logprobs(token_ids)
        except TinkerDeadlineError:
            self._evict_on_deadline()
            raise

    def topk_prompt_logprobs(
        self, token_ids: list[int], k: int
    ) -> tuple[list[float | None], list[TopkCandidates | None]]:
        """One prefill-only sample returning realized and top-k prompt logprobs.

        The teacher's topk_ce path (`TinkerTeacher.score_topk`) narrows its
        injected scorer to `TopkLogprobScorer` at runtime; this delegation
        makes the loop's teacher sampling client satisfy it against the real
        SDK, with the same cache eviction on a deadline expiry as every
        other call.
        """
        try:
            return self._scorer.topk_prompt_logprobs(token_ids, k)
        except TinkerDeadlineError:
            self._evict_on_deadline()
            raise

    def get_tokenizer(self) -> EncodingTokenizer:
        """The HF tokenizer for the client's base model (deadline-bounded fetch)."""
        try:
            return cast(
                "EncodingTokenizer", call_with_deadline("connect", self._client.get_tokenizer)
            )
        except TinkerDeadlineError:
            self._evict_on_deadline()
            raise


class SdkTrainingClient:
    """Adapts a real `tinker.TrainingClient` to `DistillTrainingClient`.

    Save names carry a per-session nonce so a resumed run re-saving the same
    step never collides with an earlier session's artifact names.

    Every SDK call is deadline-bounded (`wmh.distill.deadlines`), and what a
    deadline expiry does depends on the call's idempotency:

    - `forward_backward` and `optim_step` are NOT idempotent mid-batch: the
      request may have executed server-side before the deadline fired, so
      re-submitting could accumulate the batch's gradients or apply the Adam
      step twice. Expiry raises cleanly, aborting the step with state intact
      (the loop's save_state cadence bounds what a resume loses).
    - `save_state` and `save_weights_for_sampler` ARE safe to retry once:
      they read state that does not change between the attempts, and each
      save retry uses a fresh artifact name so a first attempt that completed
      server-side after being abandoned can never collide.
    - `load_state` is NOT retryable on the same client, and must be its FIRST
      call: tinker accepts LoadWeights only on an uninitialized model, and an
      abandoned request keeps running server-side. A live resume proved it,
      dying with "LoadWeights can only be called on uninitialized models"
      when the retry followed a restore the client had given up waiting for.
      Expiry raises, and the retry belongs on a freshly created (still
      uninitialized) client, which is what `_DistillRun._open_training_client`
      does. `get_tokenizer` stays legal before a restore: the SDK answers it
      from a metadata-only GetInfo request that never touches weights.
    """

    def __init__(self, client: tinker.TrainingClient) -> None:
        self._client = client
        self._session = uuid4().hex[:8]
        self._save_counter = 0
        self._initialized = False
        """Whether a call that initializes the model server-side has been issued."""

    def get_tokenizer(self) -> EncodingTokenizer:
        """The student base model's HF tokenizer (deadline-bounded fetch).

        Metadata only (the SDK resolves it through GetInfo), so it never
        initializes the model and stays legal before `load_state`.
        """
        return cast("EncodingTokenizer", call_with_deadline("connect", self._client.get_tokenizer))

    def forward_backward(self, datums: Sequence[TrainDatum], loss_fn: str) -> TrainStepOutput:
        """Convert to real tinker datums and run one bounded forward/backward.

        The loss decides the wire conversion: cross_entropy datums (the
        warmup phase and topk_ce replicas) carry {target_tokens, weights},
        while both advantage losses (importance_sampling and ppo) carry
        {target_tokens, logprobs, advantages}; each loss rejects any other
        keyset server-side. No `loss_fn_config` is sent, so ppo's ratio-clip
        epsilon is the service default.

        Never retried on a deadline expiry: gradients may already have been
        accumulated server-side, so a re-submit could count the batch twice.
        The typed error aborts the step; resume restores the last checkpoint.

        Returns:
            The step output. The SDK's `ForwardBackwardOutput` has no typed
            loss field (only `loss_fn_output_type`, per-datum
            `loss_fn_outputs`, and the untyped `metrics` dict), so `loss` is
            whatever `metrics` reports under `SDK_LOSS_METRIC_NAMES`, or
            None when the service reports no such key.
        """
        converted = (
            to_tinker_sft_datums(datums)
            if loss_fn == CROSS_ENTROPY_LOSS
            else to_tinker_datums(datums)
        )
        self._initialized = True
        future = self._client.forward_backward(converted, cast("tinker_types.LossFnType", loss_fn))
        output = wait_with_deadline("forward_backward", future)
        return TrainStepOutput(loss=sdk_metric_value(output.metrics, SDK_LOSS_METRIC_NAMES))

    def optim_step(self, learning_rate: float) -> OptimStepOutput:
        """One bounded Adam step at the given learning rate.

        Never retried on a deadline expiry: the step may have been applied
        server-side before the deadline fired, and re-applying would
        double-step the optimizer. The typed error aborts the step cleanly.

        Returns:
            The step output. The SDK's `OptimStepResponse` carries only the
            untyped optional `metrics` dict with no documented keys, so
            `grad_norm` is whatever `metrics` reports under
            `SDK_GRAD_NORM_METRIC_NAMES`, or None when absent.
        """
        from tinker import types

        self._initialized = True
        response = wait_with_deadline(
            "optim_step", self._client.optim_step(types.AdamParams(learning_rate=learning_rate))
        )
        return OptimStepOutput(
            grad_norm=sdk_metric_value(response.metrics, SDK_GRAD_NORM_METRIC_NAMES)
        )

    def save_state(self) -> str:
        """Save resumable training state under a session-unique name.

        Retried once on a deadline expiry: saving snapshots state that does
        not change between the attempts, and the retry advances the name
        counter so a first attempt that completed server-side after being
        abandoned can never collide.
        """
        try:
            return self._save_state_once()
        except TinkerDeadlineError as exc:
            logger.warning("retrying save_state once with a fresh request: %s", exc)
            return self._save_state_once()

    def _save_state_once(self) -> str:
        name = f"wmh-distill-{self._session}-state-{self._save_counter:04d}"
        self._save_counter += 1
        self._initialized = True
        return wait_with_deadline("save_state", self._client.save_state(name)).path

    def load_state(self, path: str) -> None:
        """Restore training state from a tinker:// state path, first of all calls.

        Never retried on this client: tinker accepts LoadWeights only on an
        uninitialized model and an abandoned request keeps running
        server-side, so a second attempt here is the live
        "LoadWeights can only be called on uninitialized models" failure. The
        deadline (`WMH_TINKER_DEADLINE_LOAD_STATE`) is the whole budget, and
        the caller retries on a fresh client.

        Args:
            path: The tinker:// state path to restore.

        Raises:
            RuntimeError: If a call that initializes the model already ran on
                this client, so the service could only reject the restore.
            TinkerDeadlineError: If the restore blows its deadline.
        """
        if self._initialized:
            raise RuntimeError(
                "load_state must be the first call on a tinker training client: the "
                "service accepts LoadWeights only on an uninitialized model, and this "
                "client already issued a weights call. Create a fresh training client "
                f"and restore {path} before anything else touches it"
            )
        self._initialized = True
        wait_with_deadline("load_state", self._client.load_state(path))

    def save_weights_for_sampler(self, name: str) -> str:
        """Save sampler weights, returning the tinker:// sampler path.

        Retried once on a deadline expiry: the weights do not change between
        the attempts, and the retry saves under a distinct "-r1" name so a
        first attempt that completed server-side after being abandoned can
        never collide.
        """
        try:
            return self._save_weights_once(f"{name}-{self._session}")
        except TinkerDeadlineError as exc:
            logger.warning("retrying save_weights_for_sampler once under a fresh name: %s", exc)
            return self._save_weights_once(f"{name}-{self._session}-r1")

    def _save_weights_once(self, full_name: str) -> str:
        self._initialized = True
        future = self._client.save_weights_for_sampler(full_name)
        return wait_with_deadline("save_weights_for_sampler", future).path


class SdkServiceClient:
    """Adapts a real `tinker.ServiceClient` to `DistillServiceClient`.

    The wrapped service builds training clients; sampling clients come from
    the process-wide shared cache in `wmh.providers.tinker` (one
    `SamplingClient` per model string, shared with the rollout providers and
    the teacher), so per-refresh construction never adds server-side sessions
    beyond one per distinct sampler path.
    """

    def __init__(self, service: tinker.ServiceClient) -> None:
        self._service = service

    def create_lora_training_client(self, base_model: str, rank: int = 32) -> SdkTrainingClient:
        """Create the real LoRA training client for the student (deadline-bounded)."""

        def build() -> tinker.TrainingClient:
            return self._service.create_lora_training_client(base_model=base_model, rank=rank)

        return SdkTrainingClient(call_with_deadline("connect", build))

    def create_sampling_client(self, model_path: str) -> SdkSamplingClient:
        """A sampling client for a tinker:// path or base model name (shared cache).

        Fetched from (or built into) the process-wide cache, deadline-bounded;
        the returned adapter evicts the cache entry on a `TinkerDeadlineError`
        so every future user rebuilds through a fresh session.
        """
        return SdkSamplingClient(shared_sampling_client(model_path), model=model_path)


def _build_sdk_service_client() -> SdkServiceClient:
    """Build the real service adapter (the `service_client=None` path).

    The SDK's service client pins one live server-side session for the life
    of the process, so the loop adapts `wmh.providers.tinker`'s process-wide
    shared instance instead of constructing a second one.

    Raises:
        ImportError: If the tinker SDK is not installed (distill extra).
        RuntimeError: If TINKER_API_KEY is missing from the environment.
    """
    return SdkServiceClient(shared_service_client())


# -- samplers ------------------------------------------------------------------------------------


def _seed_from_name(name: str) -> int:
    """A stable cross-process seed derived from the run name (hash() is salted)."""
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big")


def tinker_provider_config(model: str, base_model: str) -> ProviderConfig:
    """The rollout provider config for one Tinker model identity.

    The single shape every distillation rollout samples through: the tinker
    provider kind (the only one that records token spans), `model` as the exact
    identity being sampled, and `model_type` as the base model whose renderer
    and tokenizer that identity uses. Both loop callers go through it (the
    student's current sampler weights, the teacher's stable model or
    checkpoint), and so does any out-of-loop caller that needs the same
    provider without standing up a training client (a rollout-only probe).

    Args:
        model: The exact model string to sample: a `tinker://` sampler-weights
            path, a checkpoint ref, or a base model name.
        base_model: The base model name that names the renderer/tokenizer
            identity (equal to `model` when sampling a base model directly).

    Returns:
        The validated provider config.
    """
    return ProviderConfig(kind=ProviderKind.TINKER, model=model, model_type=base_model)


class TaskSampler:
    """A seeded shuffle-cycle over the train task ids.

    Batches are unique within themselves and the cycle visits every task
    before repeating any, so coverage stays even. The sequence is a pure
    function of (task_ids, seed, draw count): a resumed run fast-forwards by
    the completed step count and continues the exact same batch sequence.

    Args:
        task_ids: The train split's task ids; must be non-empty and unique.
        seed: Seed for the shuffle order.

    Raises:
        ValueError: If `task_ids` is empty or contains duplicates.
    """

    def __init__(self, task_ids: Sequence[str], *, seed: int) -> None:
        ids = list(task_ids)
        if not ids:
            raise ValueError("task_ids is empty; a distillation run needs at least one train task")
        if len(set(ids)) != len(ids):
            raise ValueError(
                "task_ids contains duplicates; pass each train task id exactly once "
                "(attempts per task come from train.group_size, not repeated ids)"
            )
        self._ids = ids
        self._rng = random.Random(seed)
        self._cycle: list[str] = []
        self._position = 0
        self._reshuffle()

    def _reshuffle(self) -> None:
        self._cycle = list(self._ids)
        self._rng.shuffle(self._cycle)
        self._position = 0

    def next_batch(self, n: int) -> list[str]:
        """The next batch of at most `n` unique task ids.

        Args:
            n: Requested batch size; clamped to the split size.

        Returns:
            `min(n, len(task_ids))` unique task ids in cycle order.

        Raises:
            ValueError: If `n` is not positive.
        """
        if n < 1:
            raise ValueError(f"batch size must be >= 1, got {n}")
        size = min(n, len(self._ids))
        batch: list[str] = []
        while len(batch) < size:
            if self._position >= len(self._cycle):
                self._reshuffle()
            candidate = self._cycle[self._position]
            self._position += 1
            if candidate not in batch:
                batch.append(candidate)
        return batch


class StudentSampler:
    """Owns the student's sampler-weights refresh cadence.

    `refresh(step)` saves the training client's current weights under a
    step-stamped name and swaps in a sampling client for them; the exposed
    `sampler_path` is the tinker:// path rollout provider configs point at.
    Refreshing the same step twice is a no-op (the finalize path may land on
    a step the cadence already refreshed).

    Args:
        service: The service client sampler clients are created from.
        training: The training client whose weights are saved.
        run_name: Stamped into every sampler-weights save name.
    """

    def __init__(
        self,
        service: DistillServiceClient,
        training: DistillTrainingClient,
        run_name: str,
    ) -> None:
        self._service = service
        self._training = training
        self._run_name = run_name
        self._path: str | None = None
        self._client: DistillSamplingClient | None = None
        self._last_step: int | None = None

    def refresh(self, step: int, *, tag: str | None = None) -> str:
        """Save current weights for sampling and swap in a fresh client.

        Args:
            step: The 0-based training step the weights are current FOR (the
                step about to sample from them); stamps the save name.
            tag: Optional label forcing a save even when `step` was already
                refreshed. The post-warmup refresh needs it: step 0 was
                refreshed before preflight, but warmup then CHANGED the
                weights, so the dedup must not short-circuit. The tag also
                stamps the save name so the two step-0 artifacts cannot
                collide.

        Returns:
            The new (or, when `step` was already refreshed and no tag forces
            a save, current) tinker:// sampler path.

        Raises:
            ValueError: If `step` is negative.
        """
        if step < 0:
            raise ValueError(f"refresh step must be >= 0, got {step}")
        if tag is None and self._path is not None and self._last_step == step:
            return self._path
        label = f"step-{step:04d}" if tag is None else f"{tag}-step-{step:04d}"
        path = self._training.save_weights_for_sampler(f"{self._run_name}-{label}")
        self._client = self._service.create_sampling_client(path)
        self._path = path
        self._last_step = step
        self._warm(step, path)
        logger.debug("student sampler refreshed for step %d: %s", step, path)
        return path

    def _warm(self, step: int, path: str) -> None:
        """Serve throwaway tokens CONCURRENTLY so the batch never races a cold sampler.

        Freshly published sampler weights are not immediately hot. Launching the whole wave at
        once means every episode's FIRST call races the same weight load, and the losers exhaust
        their retries and report `provider_error` at turn 0-2 while the survivors go on to run 24+
        turns cleanly.

        Two live measurements shaped this, and the second is why the warmup is concurrent rather
        than a single call:

        1. A 51-episode eval wave lost 7 episodes (~15%) at a mean of 2.0 turns, against 6% for
           the same model sampled from already-warm BASE weights. A base-weights probe therefore
           cannot detect this at all.
        2. After adding a ONE-token serial warmup, a 64-episode training wave still lost 7 of its
           first 34 episodes (~21%) at a mean of **0.7 turns**. One token wakes the model but does
           not make it scale: the cost is not a single weight load, it is the sampler's ramp to
           serving many streams at once, so the warmup has to exercise concurrency too.

        `_WARMUP_STREAMS` concurrent throwaway calls approximate the wave's own fan-out. Failure
        of any stream is deliberately NOT fatal: a sampler that cannot serve one token will fail
        loudly in the batch anyway, and aborting a paid run inside a warmup trades a partial batch
        for no batch.
        """
        client = self._client
        if client is None:  # pragma: no cover - refresh() just assigned it
            return

        def once() -> None:
            client.sample(prompt_token_ids=[0], max_tokens=1, temperature=0.0)

        failures: list[BaseException] = []
        with ThreadPoolExecutor(max_workers=_WARMUP_STREAMS) as pool:
            for future in [pool.submit(once) for _ in range(_WARMUP_STREAMS)]:
                error = future.exception()
                if error is not None:
                    failures.append(error)
        if failures:
            first = failures[0]
            logger.warning(
                "student sampler warmup for step %d (%s): %d of %d streams did not serve a token "
                "(first: %s: %s); launching the batch anyway, but expect turn-0 provider_error "
                "losses if the weights are still loading",
                step,
                path,
                len(failures),
                _WARMUP_STREAMS,
                type(first).__name__,
                first,
            )

    @property
    def sampler_path(self) -> str:
        """The current tinker:// sampler path."""
        if self._path is None:
            raise RuntimeError(
                "the student sampler has no weights yet; call refresh() before reading sampler_path"
            )
        return self._path

    @property
    def client(self) -> DistillSamplingClient:
        """The sampling client for the current weights."""
        if self._client is None:
            raise RuntimeError(
                "the student sampler has no weights yet; call refresh() before reading client"
            )
        return self._client

    def provider_config(self, base_model: str) -> ProviderConfig:
        """The rollout provider config pointing at the current weights.

        Args:
            base_model: The student base model name (renderer identity).
        """
        return tinker_provider_config(self.sampler_path, base_model)


# -- preflight -----------------------------------------------------------------------------------


def tito_recompute_check(
    client: DistillSamplingClient,
    prompt_token_ids: Sequence[int],
    *,
    sample_tokens: int = _PREFLIGHT_SAMPLE_TOKENS,
    temperature: float = 1.0,
    mean_tolerance: float = DEFAULT_TITO_MEAN_TOLERANCE,
    max_tolerance: float = DEFAULT_TITO_MAX_TOLERANCE,
) -> None:
    """The live tokens-in-tokens-out proof: sample, then recompute agreement.

    Samples a short sequence, then asks the same client to `compute_logprobs`
    on prompt + sampled tokens and requires the recomputed logprobs at the
    sampled positions to agree with the issued ones. Two bounds apply: the
    MEAN absolute gap across the sample must stay within `mean_tolerance`
    (sampler/scorer kernel noise is zero-mean; systematic corruption is not),
    and no single position may exceed `max_tolerance` (a wrong sampler path,
    tokenizer drift, or SDK change shifts logprobs by whole nats). Failing
    either means the sampling and scoring paths do not see the same tokens,
    which would silently corrupt every training datum.

    Args:
        client: The student sampling client under test.
        prompt_token_ids: A non-empty probe prompt.
        sample_tokens: How many tokens to sample.
        temperature: Sampling temperature; 1.0 keeps issued logprobs directly
            comparable to recomputed model logprobs.
        mean_tolerance: Max acceptable mean |issued - recomputed| over the
            sampled positions.
        max_tolerance: Max acceptable |issued - recomputed| at any position.

    Raises:
        RuntimeError: On empty prompt/sample, missing or misaligned logprobs,
            or disagreement beyond either bound; the message names the
            offending statistic and both values.
    """
    prompt = list(prompt_token_ids)
    if not prompt:
        raise RuntimeError(
            "the TITO recompute check needs a non-empty probe prompt; the renderer "
            "produced no tokens for the preflight message, so check the renderer "
            "and tokenizer for the student base model"
        )
    sequence = client.sample(prompt, max_tokens=sample_tokens, temperature=temperature)
    sampled = list(sequence.tokens)
    issued = sequence.logprobs
    if not sampled:
        raise RuntimeError(
            "the TITO recompute check sampled no tokens; the student sampler "
            "returned an empty sequence for a fresh prompt, so check the sampler "
            "weights path and the model's availability"
        )
    if issued is None or len(issued) != len(sampled):
        got = "no logprobs" if issued is None else f"{len(issued)} logprobs"
        raise RuntimeError(
            f"the TITO recompute check got {got} for {len(sampled)} sampled tokens; "
            "per-token logprobs are required for tokens-in-tokens-out training, so "
            "check the tinker SDK version pin"
        )
    full = prompt + sampled
    recomputed = client.compute_logprobs(full)
    if len(recomputed) != len(full):
        raise RuntimeError(
            f"compute_logprobs returned {len(recomputed)} entries for the "
            f"{len(full)}-token recompute sequence; it must return one entry per "
            "position, so check the tinker SDK version pin"
        )
    gaps: list[float] = []
    for offset, issued_lp in enumerate(issued):
        recomputed_lp = recomputed[len(prompt) + offset]
        if recomputed_lp is None:
            raise RuntimeError(
                f"TITO recompute returned no logprob at sampled position {offset} "
                f"(sampler issued {issued_lp:.4f}). The scoring path cannot see "
                "the student's own tokens, so training data would be corrupt; "
                "check that the sampler path and base model match and that the "
                "pinned tinker SDK is unchanged"
            )
        gap = abs(recomputed_lp - issued_lp)
        if gap > max_tolerance:
            raise RuntimeError(
                f"TITO recompute disagreement at sampled position {offset}: the "
                f"sampler issued logprob {issued_lp:.4f} but compute_logprobs "
                f"returned {recomputed_lp:.4f} (gap {gap:.4f} > per-position "
                f"bound {max_tolerance}). A gap this large means the sampling "
                "and scoring paths disagree on the student's own tokens, so "
                "training data would be corrupt; check that the sampler path "
                "and base model match and that the pinned tinker SDK is unchanged"
            )
        gaps.append(gap)
    mean_gap = sum(gaps) / len(gaps)
    if mean_gap > mean_tolerance:
        worst = max(range(len(gaps)), key=gaps.__getitem__)
        raise RuntimeError(
            f"TITO recompute disagreement: mean |issued - recomputed| logprob "
            f"gap {mean_gap:.4f} over {len(gaps)} sampled tokens exceeds "
            f"{mean_tolerance} (worst position {worst}: gap {gaps[worst]:.4f}). "
            "Sampler/scorer kernel noise is zero-mean, so a systematic gap "
            "means the paths disagree on the student's own tokens and training "
            "data would be corrupt; check that the sampler path and base model "
            "match and that the pinned tinker SDK is unchanged"
        )
    logger.info(
        "TITO recompute check passed: mean gap %.4f, max gap %.4f over %d tokens",
        mean_gap,
        max(gaps),
        len(gaps),
    )


# -- internal helpers ----------------------------------------------------------------------------


def resume_command(name: str, run_dir: Path) -> str:
    """The CLI command that resumes a distillation run.

    The CLI layer (`wmh/cli/harness_distill.py`) reuses this helper so the
    command printed on a budget abort stays the command that actually works.
    On resume the run's pinned config is the `config.toml` snapshot inside
    the run dir. `name` must be the agent string as the user typed it (an
    @ref included) or the printed command trips the CLI resume conflict
    check.
    """
    return f"wmh optimize harness {name} harbor --mode distill --run-dir {run_dir} --resume"


def pin_rollout_params(harness: HarnessDoc, cfg: DistillConfig) -> HarnessDoc:
    """Pin the config's rollout knobs onto the harness document the trials run.

    The pi runtimes read their sampling temperature, turn cap, and per-call
    output cap from the document's param surfaces, so `[sampling]` and
    `[rollout] max_turns` only take effect by being written INTO the document:
    `sampling.temperature` -> `param:temperature`, `rollout.max_turns` ->
    `param:max-turns`, `sampling.max_tokens` -> `param:max-output-tokens`.
    The result is a pure function of (seed document, config), so every session
    and step of a run derives the identical document (and harbor job identity).

    A temperature other than 1.0 is allowed but warned about: the sampler's
    issued logprobs are temperature-scaled while the teacher's
    `compute_logprobs` are not, which biases the reverse-KL advantages and the
    importance-sampling correction.

    Args:
        harness: The seed harness document.
        cfg: The validated run config.

    Returns:
        A new validated document with the three param surfaces replaced.
    """
    if cfg.sampling.temperature != 1.0:
        logger.warning(
            "sampling.temperature = %s: sampler-issued logprobs are temperature-scaled "
            "but teacher logprobs are not, so reverse-KL advantages are biased; use "
            "temperature = 1.0 for faithful importance weights",
            cfg.sampling.temperature,
        )
    replacements = {
        TEMPERATURE_ID: str(cfg.sampling.temperature),
        MAX_TURNS_ID: str(cfg.rollout.max_turns),
        MAX_OUTPUT_TOKENS_ID: str(cfg.sampling.max_tokens),
    }
    surfaces = [surface for surface in harness.surfaces if surface.id not in replacements]
    surfaces.extend(
        Surface(id=surface_id, kind=SurfaceKind.PARAM, content=content)
        for surface_id, content in replacements.items()
    )
    # Reconstruct (not model_copy) so the document re-validates as a whole.
    return HarnessDoc(name=harness.name, version=harness.version, surfaces=surfaces)


def _batch_reverse_kl(
    datums: Sequence[TrainDatum], rows: Sequence[Sequence[float | None]]
) -> float | None:
    """The batch's reverse KL per token from realized teacher logprobs.

    `mean(sampled_lp - teacher_lp)` over every scored loss token; None when
    nothing was scored. Both loss modes feed this the teacher's logprobs of
    the REALIZED (student-sampled) tokens (`TeacherClient.score` rows, or
    `TeacherTopkScores.realized`), so the metric means the same thing across
    modes.

    Args:
        datums: The step's datums.
        rows: One per-position realized-logprob row per datum, aligned with
            `datums`; misaligned rows are skipped (the datum path drops and
            counts them separately).
    """
    kl_sum = 0.0
    kl_count = 0
    for datum, row in zip(datums, rows, strict=True):
        if len(row) != len(datum.model_input_tokens):
            continue  # misaligned row: the datum builders drop and count it
        for position, teacher_lp in enumerate(row):
            if teacher_lp is None:
                continue
            kl_sum += datum.sampled_logprobs[position] - teacher_lp
            kl_count += 1
    return kl_sum / kl_count if kl_count else None


def _teacher_rows(
    teacher: TeacherClient, datums: Sequence[TrainDatum]
) -> tuple[list[list[float | None]], float | None]:
    """Score datums with the teacher and compute the batch's reverse KL.

    The teacher consumes the loop's own datums and returns one per-position
    row per datum in the compute_logprobs convention (loss positions carry a
    logprob; everything else, including a position-0 loss token the teacher
    cannot condition on, stays None and is dropped loudly downstream by
    `attach_advantages`).

    Args:
        teacher: The teacher backend.
        datums: The step's datums.

    Returns:
        The per-position teacher rows (aligned one to one with `datums`) and
        the batch's reverse KL per token, `mean(sampled_lp - teacher_lp)` over
        every scored loss token (None when nothing was scored).
    """
    rows = teacher.score(list(datums))
    return rows, _batch_reverse_kl(datums, rows)


def _graded_phrase(graded_solve_rate: float, graded_trials: int) -> str:
    """One progress-line clause for the graded rate, saying so when there is none.

    A batch with no readable test report has no graded score, and its stats carry 0.0 as a
    placeholder; printing that as "graded 0.000" would report a null measurement as a total failure.
    """
    if not graded_trials:
        return "no graded score (no readable test report)"
    return f"graded {graded_solve_rate:.3f} over {graded_trials} graded trial(s)"


def _measured_graded_rate(report: DistillEvalReport) -> float | None:
    """The report's graded solve rate, or None when it measured none.

    None covers both an eval whose trials left no readable test report and a baseline imported from
    a run that predates the field (`graded_trials == 0`), so neither charts a fabricated 0.0.
    """
    return report.graded_solve_rate if report.graded_trials else None


class _RunBudget:
    """This session's `BudgetMeter` plus the USD prior sessions already spent.

    The cap is enforced over the TOTAL (prior + session) so a resumed run
    cannot spend the budget twice; the inner meter never enforces on its own.
    Every charge is pushed through `on_spend` (the run store's spend ledger)
    so every recorded charge survives a crash (metrics rows alone would lose
    everything charged since the last completed step; work still in flight when
    a session dies is charged only when its batch returns).
    """

    def __init__(
        self,
        pricing: PricingConfig,
        max_usd: float | None,
        prior_usd: float,
        *,
        on_spend: Callable[[float], None] | None = None,
    ) -> None:
        self._meter = BudgetMeter(pricing, max_usd=None)
        self._max_usd = max_usd
        self.prior_usd = prior_usd
        self._on_spend = on_spend

    def charge(self, meter: MeterName, tokens: int) -> None:
        """Record actual token usage against one meter and persist the total."""
        self._meter.charge(meter, tokens)
        if self._on_spend is not None:
            self._on_spend(self.total_usd)

    def check(self) -> None:
        """Enforce the cap over the total spend.

        Raises:
            BudgetExhausted: When the cap is set and the total exceeds it.
        """
        if self._max_usd is not None and self.total_usd > self._max_usd:
            raise BudgetExhausted(self.total_usd, self._max_usd)

    @property
    def session_usd(self) -> float:
        """Priced USD spent by this session only."""
        return self._meter.spent_usd

    @property
    def total_usd(self) -> float:
        """Priced USD across prior sessions and this one."""
        return self.prior_usd + self._meter.spent_usd

    def tokens(self, meter: MeterName) -> int:
        """This session's tokens charged to one meter."""
        return self._meter.tokens(meter)

    def lines(self) -> list[CostLine]:
        """This session's actuals in the estimate's line shape."""
        return self._meter.lines()


class _DistillRun:
    """One orchestrated distillation run; `run_distillation` drives it."""

    def __init__(
        self,
        name: str,
        cfg: DistillConfig,
        harness: HarnessDoc,
        train_task_ids: Sequence[str],
        holdout_task_ids: Sequence[str],
        run_dir: Path,
        *,
        service: DistillServiceClient,
        adapter_store: AdapterStore,
        on_progress: ProgressCallback | None,
        live_trial_preflight: LiveTrialPreflight | None,
        tracker: DistillTracker,
        cli_agent: str | None = None,
    ) -> None:
        self._name = name
        # The agent string resume commands print; the run/adapter name strips any
        # @ref, but a resume must be invoked with the string the run started with.
        self._cli_agent = cli_agent if cli_agent is not None else name
        self._cfg = cfg
        self._harness = pin_rollout_params(harness, cfg)
        self._train_ids = list(train_task_ids)
        self._holdout_ids = list(holdout_task_ids)
        self._run_dir = run_dir
        self._service = service
        self._adapters = adapter_store
        self._on_progress = on_progress
        self._live_trial_preflight = live_trial_preflight
        self._tracker = tracker
        self._store = DistillRunStore(run_dir)
        self._teacher_identity = cfg.teacher.checkpoint or cfg.teacher.model
        # Set by _preflight (the renderer needs the training client's tokenizer);
        # kept for sample-rollout logging across every later batch.
        self._rendering: ChatRendering | None = None
        # Set up by execute():
        self._training: DistillTrainingClient
        self._teacher: TinkerTeacher
        self._teacher_client: DistillSamplingClient
        self._sampler: StudentSampler
        self._tasks: TaskSampler
        self._budget: _RunBudget
        self._prev_tokens: dict[MeterName, int] = dict.fromkeys(METER_NAMES, 0)
        self._prev_usd = 0.0
        self._empty_step_streak = 0
        self._degeneration_streak = 0
        # Loaded from the run manifest on resume (never re-measured there);
        # captured at this session's first training step otherwise.
        self._tripwire_baseline: TripwireBaseline | None = None
        self._resumed = False

    # -- plumbing --------------------------------------------------------------------------------

    def _emit(self, phase: DistillPhase, message: str, *, step: int | None = None) -> None:
        logger.info("[%s] %s", phase, message)
        if self._on_progress is not None:
            self._on_progress(
                DistillProgress(
                    phase=phase,
                    message=message,
                    total_steps=self._cfg.train.steps,
                    step=step,
                    spent_usd=max(self._budget.total_usd, 0.0),
                )
            )

    def _abort_for_budget(self, exc: BudgetExhausted, *, completed_step: int | None) -> NoReturn:
        """Persist what can be persisted, then raise the typed budget error.

        Args:
            exc: The meter's exhaustion error.
            completed_step: The last fully completed training step of this
                session, or None when no step completed (nothing new to
                checkpoint; recorded artifacts are reused on resume).
        """
        if completed_step is not None:
            state_path = self._training.save_state()
            self._store.record_checkpoint(completed_step, state_path, self._sampler.sampler_path)
            saved = f"training state was saved (checkpoint at step {completed_step})"
        else:
            saved = (
                "no training step completed in this session, so the run resumes "
                "from its recorded artifacts"
            )
        command = resume_command(self._cli_agent, self._run_dir)
        raise DistillBudgetError(
            f"budget exhausted: ${exc.spent_usd:.2f} spent against the "
            f"${exc.max_usd:.2f} cap (budget.max_usd); {saved}. Raise budget.max_usd "
            f"in {self._store.config_path} and resume with: {command}",
            resume_command=command,
            spent_usd=exc.spent_usd,
            max_usd=exc.max_usd,
        ) from exc

    def _check_empty_batch_streak(self, step: int, trials: int, empty: int, datums: int) -> None:
        """Track all-empty training steps and abort after the tolerated streak.

        Args:
            step: The 0-based training step that just completed its metrics row.
            trials: The step's trial count.
            empty: Trials that recorded no token span.
            datums: Trainable datums the step produced.

        Raises:
            DistillEmptyBatchError: After `MAX_CONSECUTIVE_EMPTY_STEPS` steps
                in a row where every trial produced zero spans and no datums
                (a step with zero trials counts: no trials is at least as dead
                as all-empty trials); the step's state is checkpointed first
                and the message carries the exact resume command.
        """
        if not (empty == trials and datums == 0):
            self._empty_step_streak = 0
            return
        self._empty_step_streak += 1
        logger.warning(
            "step %d is all-empty (%d trial(s), every one without token spans); "
            "%d/%d consecutive empty step(s) before the run aborts",
            step,
            trials,
            self._empty_step_streak,
            MAX_CONSECUTIVE_EMPTY_STEPS,
        )
        if self._empty_step_streak < MAX_CONSECUTIVE_EMPTY_STEPS:
            return
        state_path = self._training.save_state()
        self._store.record_checkpoint(step, state_path, self._sampler.sampler_path)
        command = resume_command(self._cli_agent, self._run_dir)
        raise DistillEmptyBatchError(
            f"aborting after {self._empty_step_streak} consecutive training steps in "
            "which every trial produced zero token spans (0 trainable datums): the "
            "trials are producing no completions, so training cannot make progress. "
            "Two causes account for nearly all of these: provider or session failures "
            "upstream in the rollout trials (see the runner logs for worker completion "
            "warnings), or, with harbor.backend = 'e2b', trials dying at sandbox "
            "creation because the E2B account is at its concurrent-sandbox cap (run "
            "`wmh e2b reap` to see what is holding the slots). Run artifacts were "
            f"persisted (checkpoint at step {step}), so once the cause is fixed resume "
            f"with: {command}",
            resume_command=command,
            consecutive_steps=self._empty_step_streak,
        )

    def _arm_tripwire(self, step: int, health: PolicyHealth) -> TripwireBaseline | None:
        """Return the run's tripwire baseline, capturing it at the first step.

        Capture happens exactly once per RUN, not per session: a baseline read
        back from the manifest on resume is kept as is, because re-measuring
        against a policy that already degenerated is what makes a resumed run's
        tripwire blind. Capture is deliberately NOT gated on
        `tripwire.enabled`: the baseline and the per-step ratios cost nothing
        and stay in `metrics.jsonl` even when the abort is switched off.

        Args:
            step: The 0-based training step being measured.
            health: That step's batch-pooled health.

        Returns:
            The armed baseline, or None when no step has been measurable yet (an
            all-empty batch cannot serve as a reference); the next step retries.
        """
        if self._tripwire_baseline is not None:
            return self._tripwire_baseline
        captured = capture_baseline(step, health)
        if captured is None:
            logger.warning(
                "step %d measured no usable policy baseline (%d episode(s) with spans, "
                "%d sampled token(s)), so the degeneration tripwire stays unarmed; the "
                "next step retries",
                step,
                health.episodes,
                health.sampled_tokens,
            )
            return None
        self._tripwire_baseline = captured
        self._store.write_tripwire_baseline(captured)
        cfg = self._cfg.tripwire
        logger.info(
            "degeneration tripwire armed from step %d: entropy %.4f nats/token, %.0f sampled "
            "tokens/episode (%d episode(s), %d token(s)). Thresholds are FRACTIONS of these, "
            "never absolutes: entropy warn %.2f / kill %.2f, length warn %.2f / kill %.2f, "
            "kill after %d consecutive step(s)%s",
            captured.step,
            captured.entropy_per_token,
            captured.mean_generation_tokens,
            captured.episodes,
            captured.sampled_tokens,
            cfg.entropy_warn_frac,
            cfg.entropy_kill_frac,
            cfg.length_warn_frac,
            cfg.length_kill_frac,
            cfg.kill_consecutive_steps,
            "" if cfg.enabled else " (tripwire.enabled = false: metrics only, no abort)",
        )
        return captured

    def _check_degeneration(
        self, step: int, health: PolicyHealth, baseline: TripwireBaseline | None
    ) -> None:
        """Warn on a degeneration breach, and abort after a kill-level streak.

        Args:
            step: The 0-based training step that just completed its metrics row.
            health: That step's batch-pooled health.
            baseline: The run's armed baseline, or None when none exists yet.

        Raises:
            DistillDegenerationError: After `tripwire.kill_consecutive_steps`
                steps in a row at kill level; the step's state is checkpointed
                first and the message carries the exact resume command. A step
                that measured nothing at all breaks neither the streak nor the
                silence: it is not evidence, and the empty-batch abort owns it.
        """
        cfg = self._cfg.tripwire
        if not cfg.enabled or baseline is None:
            return
        if baseline.step == step:
            # The baseline step is its own reference (ratio 1.0), so it can only
            # breach through a misconfigured fraction above 1.0. It never fires.
            return
        if health.entropy_per_token is None and health.mean_generation_tokens is None:
            # An all-empty batch is no evidence either way, so it neither breaches
            # nor clears a streak (the empty-batch abort is what owns that case).
            return
        breaches = evaluate_breaches(cfg, baseline, health)
        for breach in breaches:
            logger.warning("degeneration tripwire at step %d: %s", step, breach.describe())
        kills = [breach for breach in breaches if breach.level == "kill"]
        if not kills:
            self._degeneration_streak = 0
            return
        self._degeneration_streak += 1
        if self._degeneration_streak < cfg.kill_consecutive_steps:
            logger.warning(
                "step %d is at a degeneration KILL level; %d/%d consecutive step(s) before "
                "the run aborts (a healthy step resets the streak)",
                step,
                self._degeneration_streak,
                cfg.kill_consecutive_steps,
            )
            return
        state_path = self._training.save_state()
        self._store.record_checkpoint(step, state_path, self._sampler.sampler_path)
        command = resume_command(self._cli_agent, self._run_dir)
        detail = "; ".join(breach.describe() for breach in kills)
        raise DistillDegenerationError(
            f"aborting after {self._degeneration_streak} consecutive training steps at a "
            f"degeneration kill level: {detail}. The student's own sampled tokens collapsed "
            "against the baseline this run measured at its first training step, which is the "
            "failure reverse KL alone cannot show (KL falls while the policy degenerates): "
            "entropy falling is mode collapse, length falling is answers collapsing toward "
            "empty. Read the step's solve_rate beside it (a real collapse takes the solve rate "
            "with it, a student that merely got more efficient does not), then lower "
            "train.learning_rate or restart from an earlier checkpoint before spending more. "
            "Run artifacts were "
            f"persisted (checkpoint at step {step}), and the recorded baseline is reused as is "
            f"on resume, so a resumed run is not re-anchored on the collapse: {command}",
            resume_command=command,
            consecutive_steps=self._degeneration_streak,
            breaches=kills,
        )

    def _charge_rollout_billing(self, billing: SpanBilling, *, teacher: bool) -> None:
        """Charge one rollout batch's per-request billing to its model's meters.

        Both models bill the same way (unique tokens at the full prefill
        rate, the repeated per-request volume at the cached rate, sampled
        tokens at the sampling rate); only the meter family differs.
        Teacher-in-harness episodes bill teacher_sample on what they
        generate, which is what estimate_run_cost projects for them.

        Args:
            billing: The batch's measured volumes (`batch_billing`).
            teacher: Charge the teacher meters (teacher-in-harness episodes:
                warmup collection, the gate's teacher baseline) instead of
                the student meters.
        """
        if teacher:
            self._budget.charge("teacher_prefill", billing.unique_tokens)
            self._budget.charge("teacher_cached_prefill", billing.cached_tokens)
            self._budget.charge("teacher_sample", billing.sampled_tokens)
        else:
            self._budget.charge("student_prefill", billing.unique_tokens)
            self._budget.charge("student_cached_prefill", billing.cached_tokens)
            self._budget.charge("student_sample", billing.sampled_tokens)

    def _meter_deltas(self) -> dict[MeterName, int]:
        """Per-meter token deltas since the previous metrics row's cursor."""
        return {
            meter: self._budget.tokens(meter) - self._prev_tokens[meter] for meter in METER_NAMES
        }

    def _student_provider(self) -> ProviderConfig:
        return self._sampler.provider_config(self._cfg.student.base_model)

    def _teacher_provider(self) -> ProviderConfig:
        return tinker_provider_config(self._teacher_identity, self._cfg.teacher.model)

    def _log_sample_rollouts(
        self, *, kind: str, name: str, step: int | None, records: Sequence[TrialRecord]
    ) -> None:
        """Persist and track one batch's first sample rollouts as readable text.

        The first `train.log_sample_rollouts` span-bearing trials render with
        the chat template's special tokens kept (`wmh.distill.samples`) and
        land in `samples/<name>.md` plus the tracker's samples table under
        `kind` ("train" for training batches, "warmup" for the warmup
        collection, "eval-<key>" for eval batches; file stems are
        "step-NNNN", "warmup", and "eval-<key>" respectively). 0 disables,
        and a batch with no span-bearing trial writes nothing. Skipped
        defensively when preflight has not built the renderer yet.
        """
        limit = self._cfg.train.log_sample_rollouts
        if limit == 0 or self._rendering is None:
            return
        samples = sample_rollouts(records, self._rendering, limit)
        if not samples:
            return
        self._store.write_samples(name, samples_markdown(samples))
        self._tracker.log_samples(kind, step, samples)

    # -- preflight -------------------------------------------------------------------------------

    def _preflight(self) -> None:
        """Every check that must pass before the run spends real money.

        Order is cheapest first: renderer resolution and the tokenizer
        fingerprint cost nothing; the pings and the TITO recompute cost a few
        tokens. The live single-trial pi preflight is the injected
        `live_trial_preflight` hook (see `LiveTrialPreflight`).
        """
        cfg = self._cfg
        self._emit("preflight", "running preflight checks before any spend")
        tokenizer = self._training.get_tokenizer()
        # The renderer must exist for the base model before any trial runs; the
        # tokenizer satisfies the renderer's slice at runtime (rendering.py makes
        # the same cast for the cookbook's loosely typed tokenizer parameter).
        rendering: ChatRendering = build_renderer(
            cfg.student.base_model, cast("RendererTokenizer", tokenizer)
        )
        # Kept for sample-rollout logging: every batch renders its first few
        # episodes to readable text via decode_with_specials.
        self._rendering = rendering
        if isinstance(self._teacher_client, TokenizerSource):
            tokenizer_fingerprint_check(
                cfg.student.base_model,
                self._teacher_identity,
                tokenizer,
                self._teacher_client.get_tokenizer(),
            )
        else:
            logger.info(
                "teacher sampling client exposes no tokenizer; skipping the "
                "fingerprint check (injected clients are assumed same-tokenizer)"
            )
        prompt_ids = rendering.build_generation_prompt([ChatMessage(role="user", content="ping")])
        try:
            sequence = self._sampler.client.sample(prompt_ids, max_tokens=1, temperature=0.0)
        except Exception as exc:  # noqa: BLE001 - re-raised with actionable context
            raise RuntimeError(
                f"student preflight ping failed for {self._sampler.sampler_path!r} "
                f"(base model {cfg.student.base_model!r}): {exc}; check that the "
                f"base model is still in Tinker's lineup and that "
                f"{TINKER_API_KEY_ENV} is valid"
            ) from exc
        if not sequence.tokens:
            raise RuntimeError(
                f"student preflight ping for {cfg.student.base_model!r} sampled no "
                "tokens; the sampler returned an empty sequence, so check the model "
                "name and the sampler weights path"
            )
        verify = self._teacher.verify()
        if not verify.ok:
            raise RuntimeError(
                f"teacher preflight ping failed for {verify.model!r}: {verify.detail}; "
                "check that the teacher model/checkpoint is still available on Tinker "
                "and shares the student's tokenizer"
            )
        tito_recompute_check(self._sampler.client, prompt_ids)
        if self._live_trial_preflight is not None:
            self._live_trial_preflight(self._student_provider())
        self._emit("preflight", "preflight checks passed")

    # -- evals -----------------------------------------------------------------------------------

    def _load_eval(self, key: str) -> DistillEvalReport | None:
        path = self._store.evals_dir / f"{key}.json"
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            return DistillEvalReport.model_validate_json(text)
        except ValidationError as exc:
            raise ValueError(
                f"corrupt eval report at {path}: {exc}; delete the file so the "
                "resumed run re-runs this eval"
            ) from exc

    def _run_eval(
        self,
        key: str,
        task_ids: Sequence[str],
        attempts: int,
        provider: ProviderConfig,
        *,
        phase: DistillPhase,
        teacher_metered: bool,
        completed_step: int | None,
    ) -> DistillEvalReport:
        """Run one eval batch as harbor trials and persist its report.

        Args:
            key: The eval's store key; also names its isolated rollout root.
            task_ids: Exact task ids to evaluate on.
            attempts: Attempts per task (the eval's k).
            provider: The worker provider config for the trials.
            phase: The progress phase to emit under.
            teacher_metered: Charge the batch to the teacher meters (sampled
                tokens at teacher_sample plus per-request prefill; see
                `_charge_rollout_billing`) instead of the student meters.
            completed_step: Forwarded to the budget abort for checkpointing.
        """
        cfg = self._cfg
        self._emit(
            phase,
            f"eval {key}: {len(list(task_ids))} task(s) x {attempts} attempt(s) "
            f"of {provider.model}",
            step=completed_step,
        )
        eval_cfg = cfg.model_copy(
            update={"train": cfg.train.model_copy(update={"group_size": attempts})}
        )
        records, stats = collect_rollouts(
            0,
            task_ids,
            eval_cfg,
            self._harness,
            provider,
            self._run_dir / EVAL_ROLLOUTS_DIR / validate_name(key),
        )
        self._charge_rollout_billing(batch_billing(records), teacher=teacher_metered)
        self._log_sample_rollouts(
            kind=f"eval-{key}", name=f"eval-{key}", step=completed_step, records=records
        )
        if stats.trials and not stats.executed_trials:
            raise DistillNullEvalError(
                f"eval {key} is a NULL measurement, not a 0.0: not one of {stats.trials} trial(s) "
                "carries a verifier reward, so no solve rate exists to record. Read the cells' "
                f"`infra-failure:` notes under {self._run_dir / EVAL_ROLLOUTS_DIR / key} for the "
                "exact causes. Trials that never started are almost always the E2B "
                "concurrent-sandbox account cap (a distill trial holds one sandbox: harbor's "
                "task environment), so lower train.trial_concurrency (currently "
                f"{cfg.train.trial_concurrency}), reap orphaned sandboxes, and re-run; trials that "
                "ran but were never graded (VerifierTimeoutError) usually need a longer "
                "verifier.override_timeout_sec in the harbor job template. Writing 0.0% here "
                "would put a null baseline behind gate.require_no_regression"
            )
        report = DistillEvalReport(
            name=key,
            provider_model=provider.model,
            base_model=provider.model_type,
            task_ids=list(task_ids),
            attempts=attempts,
            trials=stats.trials,
            solve_rate=stats.solve_rate,
            graded_solve_rate=stats.graded_solve_rate,
            graded_trials=stats.graded_trials,
            empty_span_trials=stats.empty_span_trials,
            executed_trials=stats.executed_trials,
            infra_failed_trials=stats.infra_failed_trials,
            scaffold_loss_rate=stats.scaffold_loss_rate,
            stop_reason_counts=dict(stats.stop_reason_counts),
        )
        if stats.infra_failed_trials or stats.scaffold_loss_rate:
            self._emit(
                phase,
                f"eval {key}: solve rate {stats.solve_rate:.3f} over "
                f"{stats.executed_trials}/{stats.trials} executed trial(s) "
                f"({stats.infra_failed_trials} infra failure(s) excluded), "
                f"{_graded_phrase(stats.graded_solve_rate, stats.graded_trials)}, scaffold loss "
                f"{stats.scaffold_loss_rate:.0%} {stats.stop_reason_counts}",
                step=completed_step,
            )
        self._store.write_eval(key, report)
        self._tracker.log_eval(
            key,
            report.solve_rate,
            completed_step,
            graded_solve_rate=_measured_graded_rate(report),
        )
        try:
            self._budget.check()
        except BudgetExhausted as exc:
            self._abort_for_budget(exc, completed_step=completed_step)
        return report

    def _eval_or_load(
        self,
        key: str,
        task_ids: Sequence[str],
        attempts: int,
        provider: ProviderConfig,
        *,
        phase: DistillPhase,
        teacher_metered: bool,
        reuse: bool,
        pin_provider: bool = False,
        completed_step: int | None = None,
        baseline_from: str | None = None,
        baseline_from_field: str = "",
    ) -> DistillEvalReport:
        """One baseline eval: this run's recorded copy, a prior run's report, or trials.

        Precedence: a report already recorded under this run's `evals/` wins
        when `reuse` is set (a resume must not re-import or re-run anything,
        and an imported baseline was copied there on the first session), then
        a configured `baseline_from` path imports a prior run's report, and
        only otherwise do trials actually run.

        A reused report must have been measured at the same `attempts` (the
        gate compares solve rates, so mixing estimators of different k would
        silently invalidate the verdict), and with `pin_provider` also under
        the same provider model (the teacher's identity is stable across
        sessions; student sampler paths carry a per-session nonce and are
        intentionally NOT pinned).

        Raises:
            RuntimeError: If a recorded report conflicts with the current
                config; the message says which knob changed and how to
                recover.
        """
        if reuse:
            existing = self._load_eval(key)
            if existing is not None:
                if existing.attempts != attempts:
                    raise RuntimeError(
                        f"recorded eval {key!r} was measured at k={existing.attempts} but the "
                        f"config now asks for k={attempts}; the gate would compare solve "
                        "rates from different estimators. Restore the original gate/eval k "
                        "in the config and resume, or start a fresh run dir"
                    )
                if pin_provider and existing.provider_model != provider.model:
                    raise RuntimeError(
                        f"recorded eval {key!r} was measured against {existing.provider_model!r} "
                        f"but the config now names {provider.model!r}; a mid-run model swap "
                        "would gate against a stale baseline. Restore the original model in "
                        "the config and resume, or start a fresh run dir"
                    )
                logger.info("reusing recorded eval %s (solve rate %.3f)", key, existing.solve_rate)
                return existing
        if baseline_from is not None:
            return self._import_baseline(
                key,
                baseline_from,
                baseline_from_field,
                task_ids,
                attempts,
                provider,
                phase=phase,
                # The teacher's provider model (its stable identity) must match
                # across runs; a student provider model is a per-run sampler
                # path, so student reuse matches on base_model alone.
                match_provider_model=teacher_metered,
            )
        return self._run_eval(
            key,
            task_ids,
            attempts,
            provider,
            phase=phase,
            teacher_metered=teacher_metered,
            completed_step=completed_step,
        )

    def _import_baseline(
        self,
        key: str,
        source: str,
        config_field: str,
        task_ids: Sequence[str],
        attempts: int,
        provider: ProviderConfig,
        *,
        phase: DistillPhase,
        match_provider_model: bool,
    ) -> DistillEvalReport:
        """Reuse a prior run's recorded baseline eval instead of running trials.

        The report is validated against this run before it is trusted: it must
        cover exactly this run's holdout task ids, carry at least `attempts`
        (this run's `gate.k`) attempts per task, and be measured on the same
        model (the provider model for the teacher, the recorded `base_model`
        for the student, whose provider model is a per-run sampler path). A
        valid report is copied into this run's `evals/` with a provenance
        `source` note and logged and tracked like a freshly run eval; nothing
        is charged to the budget.

        Args:
            key: The eval's store key (the baseline's canonical name).
            source: The prior run's `evals/<key>.json` path from the config.
            config_field: The config field naming `source`, for error messages.
            task_ids: This run's holdout task ids.
            attempts: This run's `gate.k`.
            provider: The provider config the baseline WOULD have run with.
            phase: The progress phase to emit under.
            match_provider_model: Also require the report's `provider_model`
                to equal the provider's model (the teacher baseline; a
                teacher's identity is stable across runs).

        Returns:
            The validated report, re-stamped with this run's provenance note.

        Raises:
            ValueError: On a missing or unparsable file or any validation
                mismatch; each message names the config field and the fix.
        """
        path = Path(source)
        hint = f"or unset {config_field} to run the baseline here"
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ValueError(
                f"{config_field} points at {path}, which does not exist; point it "
                f"at a prior run's evals/{key}.json {hint}"
            ) from exc
        try:
            report = DistillEvalReport.model_validate_json(text)
        except ValidationError as exc:
            raise ValueError(
                f"{config_field} points at {path}, which is not a valid eval "
                f"report: {exc}; point it at a prior run's evals/{key}.json {hint}"
            ) from exc
        if set(report.task_ids) != set(task_ids):
            missing = sorted(set(task_ids) - set(report.task_ids))
            extra = sorted(set(report.task_ids) - set(task_ids))
            raise ValueError(
                f"{config_field}: the report at {path} was measured on a different "
                f"task set than this run's holdout split (missing: "
                f"{', '.join(missing) or 'none'}; extra: {', '.join(extra) or 'none'}); "
                f"a baseline only transfers between runs gating on the identical "
                f"holdout split, so reuse one that matches {hint}"
            )
        if report.attempts < attempts:
            raise ValueError(
                f"{config_field}: the report at {path} recorded {report.attempts} "
                f"attempt(s) per task but this run's gate.k is {attempts}; reuse a "
                f"baseline measured with at least gate.k attempts {hint}"
            )
        if match_provider_model and report.provider_model != provider.model:
            raise ValueError(
                f"{config_field}: the report at {path} evaluated "
                f"{report.provider_model!r} but this run's teacher is "
                f"{provider.model!r}; reuse a baseline of the same teacher {hint}"
            )
        expected_base = provider.model_type or provider.model
        if report.base_model is None and not match_provider_model:
            raise ValueError(
                f"{config_field}: the report at {path} records no base_model (it "
                f"predates the field), so it cannot be validated against this run's "
                f"base model {expected_base!r}; re-run that baseline on current code "
                f"{hint}"
            )
        if report.base_model is not None and report.base_model != expected_base:
            raise ValueError(
                f"{config_field}: the report at {path} was measured on base model "
                f"{report.base_model!r} but this run uses {expected_base!r}; a "
                f"baseline only transfers between runs on the same base model, so "
                f"reuse one that matches {hint}"
            )
        stamped = report.model_copy(
            update={"name": key, "source": f"reused from {path} via {config_field}"}
        )
        self._store.write_eval(key, stamped)
        self._tracker.log_eval(
            key, stamped.solve_rate, None, graded_solve_rate=_measured_graded_rate(stamped)
        )
        self._emit(
            phase,
            f"eval {key}: reused {path} (solve rate {stamped.solve_rate:.3f}, "
            f"{len(stamped.task_ids)} task(s) x {stamped.attempts} attempt(s)); "
            "no trials run",
        )
        return stamped

    # -- cross_entropy phases (off-policy distillation, legacy warmup) -----------------------------

    def _append_offpolicy_row(self, offpolicy_step: int, row: OffPolicyMetrics) -> None:
        """Persist and track one off-policy row, advancing the per-row delta cursors."""
        self._store.append_metrics(offpolicy_step, row)
        self._tracker.log_offpolicy_step(offpolicy_step, row)
        self._prev_tokens = {meter: self._budget.tokens(meter) for meter in METER_NAMES}
        self._prev_usd = self._budget.session_usd

    def _offpolicy_row(
        self,
        *,
        epoch: int,
        epochs: int,
        minibatch: int,
        planned_steps: int,
        trials: int,
        kept_trials: int,
        solve_rate: float,
        corpus_datums: int,
        datums: int,
        loss_tokens: int,
        context_tokens: int,
        learning_rate: float,
        loss: float | None,
        grad_norm: float | None,
    ) -> OffPolicyMetrics:
        """One off-policy metrics row carrying the meter deltas since the last row."""
        deltas = self._meter_deltas()
        return OffPolicyMetrics(
            epoch=epoch,
            epochs=epochs,
            minibatch=minibatch,
            planned_steps=planned_steps,
            tasks=len(self._train_ids),
            trials=trials,
            kept_trials=kept_trials,
            solve_rate=solve_rate,
            corpus_datums=corpus_datums,
            datums=datums,
            loss_tokens=loss_tokens,
            context_tokens=context_tokens,
            learning_rate=learning_rate,
            loss=loss,
            grad_norm=grad_norm,
            student_prefill_tokens=deltas["student_prefill"],
            student_cached_prefill_tokens=deltas["student_cached_prefill"],
            student_sample_tokens=deltas["student_sample"],
            student_train_tokens=deltas["student_train"],
            teacher_prefill_tokens=deltas["teacher_prefill"],
            teacher_cached_prefill_tokens=deltas["teacher_cached_prefill"],
            teacher_sample_tokens=deltas["teacher_sample"],
            usd=max(self._budget.session_usd - self._prev_usd, 0.0),
        )

    def _append_warmup_row(self, warmup_step: int, row: WarmupMetrics) -> None:
        """Persist and track one warmup row, advancing the per-row delta cursors."""
        self._store.append_metrics(warmup_step, row)
        self._tracker.log_warmup_step(warmup_step, row)
        self._prev_tokens = {meter: self._budget.tokens(meter) for meter in METER_NAMES}
        self._prev_usd = self._budget.session_usd

    def _warmup_row(
        self,
        *,
        trials: int,
        kept_trials: int,
        solve_rate: float,
        datums: int,
        loss_tokens: int,
        context_tokens: int,
        learning_rate: float,
    ) -> WarmupMetrics:
        """One warmup metrics row carrying the meter deltas since the last row."""
        deltas = self._meter_deltas()
        return WarmupMetrics(
            tasks=len(self._train_ids),
            trials=trials,
            kept_trials=kept_trials,
            solve_rate=solve_rate,
            datums=datums,
            loss_tokens=loss_tokens,
            context_tokens=context_tokens,
            learning_rate=learning_rate,
            student_prefill_tokens=deltas["student_prefill"],
            student_cached_prefill_tokens=deltas["student_cached_prefill"],
            student_sample_tokens=deltas["student_sample"],
            student_train_tokens=deltas["student_train"],
            teacher_prefill_tokens=deltas["teacher_prefill"],
            teacher_cached_prefill_tokens=deltas["teacher_cached_prefill"],
            teacher_sample_tokens=deltas["teacher_sample"],
            usd=max(self._budget.session_usd - self._prev_usd, 0.0),
        )

    def _load_teacher_trials(
        self, source_dir: Path, *, setting: str
    ) -> tuple[list[TrialRecord], RolloutStats]:
        """Load another run's teacher collection instead of collecting rollouts.

        The `trajectories_from` path shared by both cross_entropy phases: the
        source run's collection wrote its assembled (unfiltered) trial records
        to `warmup-trials.json`, so this run reuses them for its own CE passes.
        The source run paid for the teacher rollouts, so loading charges no
        meter; the `keep` filter still applies to the loaded records at the
        call site.

        Args:
            source_dir: The source run's directory.
            setting: The config section that named `source_dir` ("offpolicy" or
                "warmup"), so an error says which key to fix.

        Returns:
            The manifest's trial records plus their recomputed batch stats.

        Raises:
            RuntimeError: If the source run has no warmup trial manifest (its
                teacher collection never completed).
            ValueError: If the manifest's teacher does not match this run's
                teacher identity.
        """
        manifest = DistillRunStore(source_dir).read_warmup_trials()
        if manifest is None:
            raise RuntimeError(
                f"{setting}.trajectories_from points at {source_dir}, but no warmup trial "
                f"manifest exists at {DistillRunStore(source_dir).warmup_trials_path}; the "
                "manifest is written when a run's teacher collection finishes, so point "
                "trajectories_from at a run dir that collected teacher rollouts, "
                "or unset it to collect here"
            )
        if manifest.teacher_model != self._teacher_identity:
            raise ValueError(
                f"the teacher trials in {source_dir} were sampled by teacher "
                f"{manifest.teacher_model!r}, but this run's teacher is "
                f"{self._teacher_identity!r}; cross_entropy training must use THIS teacher's "
                f"trajectories, so point {setting}.trajectories_from at a run with a "
                "matching teacher, or unset it to collect fresh"
            )
        # THIS run's output cap, not the source run's: the truncation count is read
        # beside this run's metrics and must answer "would these turns be cut off here".
        return list(manifest.records), rollout_stats(
            manifest.records, max_tokens=self._cfg.sampling.max_tokens
        )

    def _collect_teacher_trials(
        self, *, rollouts_per_task: int, phase: DistillPhase
    ) -> tuple[list[TrialRecord], RolloutStats]:
        """Collect a cross_entropy phase's teacher rollouts and persist the manifest.

        Runs the teacher through the pi harness on every train task
        (`rollouts_per_task` attempts each, isolated under
        `warmup-rollouts/`), writes the assembled records to
        `warmup-trials.json` (so other runs can load this collection via
        `trajectories_from`, and so a resumed off-policy phase never pays for
        the corpus twice), and charges the teacher meters.

        Args:
            rollouts_per_task: Teacher attempts per train task.
            phase: The progress phase to emit under.
        """
        cfg = self._cfg
        self._emit(
            phase,
            f"collecting teacher trajectories: {len(self._train_ids)} train task(s) x "
            f"{rollouts_per_task} attempt(s) of {self._teacher_identity}",
        )
        # The collector reads attempts from train.group_size, the same override
        # trick _run_eval uses for its k.
        collect_cfg = cfg.model_copy(
            update={"train": cfg.train.model_copy(update={"group_size": rollouts_per_task})}
        )
        records, roll_stats = collect_rollouts(
            0,
            self._train_ids,
            collect_cfg,
            self._harness,
            self._teacher_provider(),
            self._run_dir / WARMUP_ROLLOUTS_DIR,
        )
        # The manifest lands before the keep filter (a loading run may filter
        # differently) and before the budget check (the collection is complete
        # evidence even when this run aborts right after paying for it).
        self._store.write_warmup_trials(
            WarmupTrialsManifest(teacher_model=self._teacher_identity, records=records)
        )
        # Teacher-in-harness billing, same as the gate's teacher baseline:
        # sampled tokens at teacher_sample, per-request prefill split between
        # teacher_prefill (unique) and teacher_cached_prefill (repeats).
        self._charge_rollout_billing(batch_billing(records), teacher=True)
        try:
            self._budget.check()
        except BudgetExhausted as exc:
            self._abort_for_budget(exc, completed_step=None)
        return records, roll_stats

    def _cross_entropy_corpus(
        self,
        *,
        trajectories_from: str | None,
        rollouts_per_task: int,
        setting: str,
        phase: DistillPhase,
        reuse_recorded: bool = False,
    ) -> tuple[list[TrialRecord], RolloutStats, str]:
        """Source one cross_entropy phase's teacher trajectories.

        Three sources, cheapest first: this run's own recorded collection (only
        when `reuse_recorded`, i.e. a resumed off-policy phase, so a resume
        never re-pays for the corpus its cursor indexes into), another run's
        collection named by `trajectories_from`, or a fresh teacher collection.

        Args:
            trajectories_from: Another run dir to load from, or None.
            rollouts_per_task: Teacher attempts per train task when collecting.
            setting: The config section these keys came from, for error text.
            phase: The progress phase to emit under.
            reuse_recorded: Whether this run's own `warmup-trials.json` may be
                reused when it exists and names this run's teacher.

        Returns:
            The trial records, their batch stats, and a past-tense word naming
            the source ("collected", "loaded", or "reused") for the phase's
            messages.
        """
        if reuse_recorded:
            recorded = self._store.read_warmup_trials()
            if recorded is not None and recorded.teacher_model == self._teacher_identity:
                self._emit(
                    phase,
                    f"reusing this run's recorded teacher collection "
                    f"({len(recorded.records)} trial(s)); nothing is charged",
                )
                return (
                    list(recorded.records),
                    rollout_stats(recorded.records, max_tokens=self._cfg.sampling.max_tokens),
                    "reused",
                )
            if recorded is not None:
                logger.warning(
                    "the recorded teacher collection in %s was sampled by %r, not this "
                    "run's teacher %r; collecting fresh trajectories instead",
                    self._store.warmup_trials_path,
                    recorded.teacher_model,
                    self._teacher_identity,
                )
        if trajectories_from is not None:
            source_dir = Path(trajectories_from)
            self._emit(
                phase,
                f"loading teacher trajectories from {source_dir} "
                "(the collection was paid by the source run; nothing is charged)",
            )
            records, roll_stats = self._load_teacher_trials(source_dir, setting=setting)
            return records, roll_stats, "loaded"
        records, roll_stats = self._collect_teacher_trials(
            rollouts_per_task=rollouts_per_task, phase=phase
        )
        return records, roll_stats, "collected"

    def _empty_corpus_reason(
        self,
        *,
        label: str,
        sourced: str,
        keep: str,
        roll_stats: RolloutStats,
        kept: int,
        datum_stats: DatumStats,
    ) -> str:
        """Warn that a cross_entropy phase has nothing to train on, and say why.

        Zero kept datums degrade the run to pure on-policy distillation, never
        an abort: the teacher's trajectories are an input, and a run that can
        still sample its own is still a run.

        Returns:
            The one-line reason recorded in the phase's terminal marker.
        """
        reason = (
            f"{label} {sourced} {roll_stats.trials} teacher trial(s) but kept "
            f"{kept} under keep={keep!r} and built 0 datums"
        )
        logger.warning(
            "%s (teacher solve rate %.2f, %d trial(s) without spans, %d overflow "
            "drop(s), %d overlong drop(s)); skipping %s, the run degrades to "
            "pure on-policy distillation",
            reason,
            roll_stats.solve_rate,
            roll_stats.empty_span_trials,
            datum_stats.overflow_drops,
            datum_stats.overlong_drops,
            label,
        )
        return reason

    def _offpolicy(self, cursor: OffPolicyCursor | None) -> None:
        """The off-policy distillation phase: teacher trials, keep-filter, CE epochs.

        The teacher trajectories come from `_cross_entropy_corpus` (this run's
        recorded collection on a resume, another run's via
        `offpolicy.trajectories_from`, or a fresh collection charged to the
        teacher meters). The kept trials merge into cross_entropy datums
        through the same prefix merge on-policy distillation uses (the teacher
        SAMPLED these tokens, so they are exact sampled ids and need no
        advantages), and `wmh.distill.offpolicy` walks the epoch/minibatch
        schedule over them.

        Resume is at datum granularity: every `offpolicy.checkpoint_every`
        optimizer steps the phase saves training state and rewrites
        `offpolicy-cursor.json`, and a resumed session restores those weights
        (in `execute`) and skips the steps the cursor already counted. Zero
        kept datums degrade the run to pure on-policy distillation (warning,
        one metrics row, the skip recorded), never an abort. Afterwards the
        state is saved and the sampler force-refreshed so any following OPD
        step samples the trained student, and `offpolicy.json` marks the phase
        done (which also drops the cursor).

        Args:
            cursor: The recorded cursor of an interrupted phase, or None to
                start the schedule from its first step.

        Raises:
            RuntimeError: If the cursor was written over a differently sized
                corpus, so its step count no longer names the same minibatches.
        """
        cfg = self._cfg
        offpolicy = cfg.offpolicy
        learning_rate = (
            offpolicy.learning_rate
            if offpolicy.learning_rate is not None
            else cfg.train.learning_rate
        )
        records, roll_stats, sourced = self._cross_entropy_corpus(
            trajectories_from=offpolicy.trajectories_from,
            rollouts_per_task=offpolicy.rollouts_per_task,
            setting="offpolicy",
            phase="offpolicy",
            reuse_recorded=self._resumed,
        )
        kept = [record for record in records if offpolicy.keep == "all" or record.passed]
        # Sampled from the KEPT trials: those are the episodes the phase trains on.
        self._log_sample_rollouts(kind="offpolicy", name="offpolicy", step=None, records=kept)
        datums, datum_stats = build_datums(kept, cfg)
        if not datums:
            reason = self._empty_corpus_reason(
                label="off-policy distillation",
                sourced=sourced,
                keep=offpolicy.keep,
                roll_stats=roll_stats,
                kept=len(kept),
                datum_stats=datum_stats,
            )
            self._append_offpolicy_row(
                0,
                self._offpolicy_row(
                    epoch=0,
                    epochs=0,
                    minibatch=0,
                    planned_steps=0,
                    trials=roll_stats.trials,
                    kept_trials=len(kept),
                    solve_rate=roll_stats.solve_rate,
                    corpus_datums=0,
                    datums=0,
                    loss_tokens=0,
                    context_tokens=0,
                    learning_rate=learning_rate,
                    loss=None,
                    grad_norm=None,
                ),
            )
            self._store.write_offpolicy(
                OffPolicyRecord(
                    epochs=0,
                    steps=0,
                    trials=roll_stats.trials,
                    kept_trials=len(kept),
                    datums=0,
                    skipped_reason=reason,
                )
            )
            self._emit("offpolicy", f"off-policy distillation skipped: {reason}")
            return

        schedule = OffPolicySchedule(
            epochs=offpolicy.epochs,
            minibatch_datums=offpolicy.minibatch_datums,
            learning_rate=learning_rate,
            shuffle_seed=offpolicy.shuffle_seed,
            checkpoint_every=offpolicy.checkpoint_every,
        )
        planned_steps = len(plan_offpolicy_steps(len(datums), schedule))
        start_step = self._offpolicy_start_step(cursor, datum_count=len(datums))

        def record_step(result: OffPolicyStepResult) -> None:
            """Charge, persist, and budget-check one completed optimizer step."""
            self._budget.charge("student_train", result.train_tokens)
            self._append_offpolicy_row(
                result.step,
                self._offpolicy_row(
                    epoch=result.epoch,
                    epochs=schedule.epochs,
                    minibatch=result.minibatch,
                    planned_steps=planned_steps,
                    trials=roll_stats.trials,
                    kept_trials=len(kept),
                    solve_rate=roll_stats.solve_rate,
                    corpus_datums=len(datums),
                    datums=result.datums,
                    loss_tokens=result.loss_tokens,
                    context_tokens=result.context_tokens,
                    learning_rate=result.learning_rate,
                    loss=result.loss,
                    grad_norm=result.grad_norm,
                ),
            )
            try:
                self._budget.check()
            except BudgetExhausted as exc:
                # The cursor (not a step checkpoint) is what an off-policy
                # resume reads, so nothing new is checkpointed here.
                self._abort_for_budget(exc, completed_step=None)

        def checkpoint(position: OffPolicyCursorPosition) -> None:
            """Save training state, then record the cursor that names it."""
            state_path = self._training.save_state()
            self._store.write_offpolicy_cursor(
                OffPolicyCursor(
                    steps_completed=position.steps_completed,
                    epoch=position.epoch,
                    minibatch=position.minibatch,
                    datums=position.datums,
                    state_path=state_path,
                )
            )
            self._emit(
                "offpolicy",
                f"checkpointed at step {position.steps_completed}/{planned_steps} "
                f"(epoch {position.epoch + 1}/{schedule.epochs}); a resume continues here",
            )

        self._emit(
            "offpolicy",
            f"training {schedule.epochs} epoch(s) x {planned_steps // max(schedule.epochs, 1)} "
            f"minibatch(es) over {len(datums)} datum(s) from {len(kept)}/{roll_stats.trials} "
            f"kept teacher trial(s) (lr {learning_rate:g})"
            + (f", resuming at step {start_step}" if start_step else ""),
        )
        run_offpolicy(
            datums,
            schedule,
            trainer=self._training,
            on_step=record_step,
            on_checkpoint=checkpoint,
            start_step=start_step,
        )
        state_path = self._training.save_state()
        # The tag forces the save: step 0 was refreshed before preflight, and
        # this phase CHANGED the weights, so OPD step 0 must sample fresh ones.
        sampler_path = self._sampler.refresh(0, tag="offpolicy")
        self._store.write_offpolicy(
            OffPolicyRecord(
                epochs=schedule.epochs,
                steps=planned_steps,
                trials=roll_stats.trials,
                kept_trials=len(kept),
                datums=len(datums),
                state_path=state_path,
                sampler_path=sampler_path,
            )
        )
        self._emit(
            "offpolicy",
            f"off-policy distillation complete: {planned_steps} step(s) over "
            f"{len(datums)} datum(s); the student now samples from {sampler_path}",
        )

    def _offpolicy_start_step(self, cursor: OffPolicyCursor | None, *, datum_count: int) -> int:
        """How many scheduled steps a resumed off-policy phase skips.

        Args:
            cursor: The recorded cursor, or None for a fresh phase.
            datum_count: The corpus this session built.

        Returns:
            The cursor's completed step count, or 0 when there is no cursor.

        Raises:
            RuntimeError: If the cursor indexes a differently sized corpus. The
                student already carries the cursor's partial training, so the
                phase can neither honor the count nor restart cleanly in place.
        """
        if cursor is None:
            return 0
        if cursor.datums != datum_count:
            raise RuntimeError(
                f"the off-policy cursor at {self._store.offpolicy_cursor_path} was written "
                f"over {cursor.datums} datum(s), but this session built {datum_count}: the "
                "corpus or its filters changed between sessions (offpolicy.keep, "
                "rollout.context_budget_tokens, train.max_datum_tokens), so the recorded "
                "step count no longer names the same minibatches. Restore the previous "
                "settings and resume, or start a fresh --run-dir (the student already "
                f"carries {cursor.steps_completed} step(s) of this phase, so restarting "
                "the schedule in place would train that prefix twice)"
            )
        logger.info(
            "resuming the off-policy phase at step %d over %d datum(s) from %s",
            cursor.steps_completed,
            cursor.datums,
            cursor.state_path,
        )
        return cursor.steps_completed

    def _warmup(self) -> None:
        """The legacy supervised warmup phase, run on the off-policy executor.

        Superseded by `[offpolicy]` and kept for the runs already configured
        against `[warmup]`; the two are mutually exclusive at config load. It
        is exactly the off-policy schedule pinned to its historical shape: one
        FULL-batch forward_backward plus one optim_step per `warmup.steps`
        pass, no per-epoch shuffle, and no cursor (an interrupted warmup
        re-runs whole, teacher trials included).

        The teacher trials come from `_cross_entropy_corpus` (fresh rollouts
        charged to the teacher meters, or another run's recorded collection via
        `warmup.trajectories_from`, which charges nothing). Either way the kept
        trials merge into cross_entropy datums through the same prefix merge
        OPD uses. Zero kept datums degrade the run to pure OPD (warning + one
        metrics row + the skip recorded), never an abort. Afterwards the state
        is saved and the sampler force-refreshed so OPD step 0 samples the
        warmed student, and `warmup.json` marks the phase done for resumes.
        """
        cfg = self._cfg
        warmup = cfg.warmup
        learning_rate = (
            warmup.learning_rate if warmup.learning_rate is not None else cfg.train.learning_rate
        )
        records, roll_stats, sourced = self._cross_entropy_corpus(
            trajectories_from=warmup.trajectories_from,
            rollouts_per_task=warmup.rollouts_per_task,
            setting="warmup",
            phase="warmup",
        )
        kept = [record for record in records if warmup.keep == "all" or record.passed]
        # Sampled from the KEPT trials: the SFT set is what warmup trains on,
        # so those are the episodes worth reading.
        self._log_sample_rollouts(kind="warmup", name="warmup", step=None, records=kept)
        datums, datum_stats = build_datums(kept, cfg)
        if not datums:
            reason = self._empty_corpus_reason(
                label="warmup",
                sourced=sourced,
                keep=warmup.keep,
                roll_stats=roll_stats,
                kept=len(kept),
                datum_stats=datum_stats,
            )
            self._append_warmup_row(
                0,
                self._warmup_row(
                    trials=roll_stats.trials,
                    kept_trials=len(kept),
                    solve_rate=roll_stats.solve_rate,
                    datums=0,
                    loss_tokens=0,
                    context_tokens=0,
                    learning_rate=learning_rate,
                ),
            )
            self._store.write_warmup(
                WarmupRecord(
                    steps=0,
                    trials=roll_stats.trials,
                    kept_trials=len(kept),
                    datums=0,
                    skipped_reason=reason,
                )
            )
            self._emit("warmup", f"warmup skipped: {reason}")
            return

        self._emit(
            "warmup",
            f"training {warmup.steps} cross_entropy pass(es) over {len(datums)} "
            f"datum(s) from {len(kept)}/{roll_stats.trials} kept teacher trial(s) "
            f"(lr {learning_rate:g})",
        )

        def record_step(result: OffPolicyStepResult) -> None:
            """Charge, persist, and budget-check one completed warmup pass."""
            self._budget.charge("student_train", result.train_tokens)
            self._append_warmup_row(
                result.step,
                self._warmup_row(
                    trials=roll_stats.trials,
                    kept_trials=len(kept),
                    solve_rate=roll_stats.solve_rate,
                    datums=len(datums),
                    loss_tokens=datum_stats.loss_tokens,
                    context_tokens=datum_stats.context_tokens,
                    learning_rate=learning_rate,
                ),
            )
            try:
                self._budget.check()
            except BudgetExhausted as exc:
                # No warmup.json yet: an interrupted warmup re-runs whole on
                # resume (the teacher trials resume trial-level via harbor).
                self._abort_for_budget(exc, completed_step=None)

        run_offpolicy(
            datums,
            OffPolicySchedule(
                epochs=warmup.steps,
                minibatch_datums=0,
                learning_rate=learning_rate,
                shuffle_seed=None,
                checkpoint_every=0,
            ),
            trainer=self._training,
            on_step=record_step,
        )
        state_path = self._training.save_state()
        # The tag forces the save: step 0 was refreshed before preflight, and
        # warmup changed the weights, so OPD step 0 must sample fresh ones.
        sampler_path = self._sampler.refresh(0, tag="warmup")
        self._store.write_warmup(
            WarmupRecord(
                steps=warmup.steps,
                trials=roll_stats.trials,
                kept_trials=len(kept),
                datums=len(datums),
                state_path=state_path,
                sampler_path=sampler_path,
            )
        )
        self._emit(
            "warmup",
            f"warmup complete: {warmup.steps} pass(es) over {len(datums)} datum(s); "
            f"OPD starts from {sampler_path}",
        )

    # -- the step loop ---------------------------------------------------------------------------

    def _train_step(self, step: int) -> None:
        cfg = self._cfg
        batch = self._tasks.next_batch(cfg.train.tasks_per_batch)
        self._emit(
            "rollouts",
            f"step {step + 1}/{cfg.train.steps}: {len(batch)} task(s) x "
            f"{cfg.train.group_size} attempt(s) from {self._sampler.sampler_path}",
            step=step,
        )
        sampler_path = self._sampler.sampler_path
        records, roll_stats = collect_rollouts(
            step, batch, cfg, self._harness, self._student_provider(), self._run_dir
        )
        self._charge_rollout_billing(batch_billing(records), teacher=False)
        self._log_sample_rollouts(kind="train", name=f"step-{step:04d}", step=step, records=records)

        # Degeneration signals come from the RAW spans (before any datum drop),
        # and the baseline is armed here so this step's row already carries its
        # own ratios; the breach check itself runs after the row is persisted.
        health = policy_health(records)
        tripwire_baseline = self._arm_tripwire(step, health)

        datums, datum_stats = build_datums(records, cfg)
        teacher_usage_before = self._teacher.usage()
        advantage_mean: float | None = None
        advantage_std: float | None = None
        clipped_tokens = 0
        # Token counts of the batch that actually trains, per source datum.
        # `datum_stats` counts what the teacher was ASKED to score, which is
        # larger whenever a misaligned teacher row drops a datum; reporting
        # that as the step's volume would overstate the batch and break
        # `clipped_tokens / loss_tokens == clip_fraction` (both clip counters
        # are post-drop). See `StepMetrics` for the two-stage contract.
        trained_loss_tokens = 0
        trained_context_tokens = 0
        if cfg.train.loss == "topk_ce":
            # One prefill-only teacher request per datum yields the top-k
            # candidate rows AND the realized logprobs, so the reverse-KL
            # metric below means exactly what it means in the default mode.
            try:
                scores = self._teacher.score_topk(datums, cfg.train.topk)
            finally:
                # Charged even when scoring raises mid-batch: the pool joins
                # every submitted call before propagating, so the whole batch
                # was billed server-side whether or not a row came back.
                self._budget.charge("teacher_prefill", self._teacher.usage() - teacher_usage_before)
            reverse_kl = _batch_reverse_kl(datums, scores.realized)
            trained, topk_stats = build_topk_ce_datums(datums, scores.topk, cfg.train.topk)
            loss_fn = CROSS_ENTROPY_LOSS
            mismatch_drops = topk_stats.mismatch_drops
            trained_loss_tokens = topk_stats.loss_tokens
            trained_context_tokens = topk_stats.context_tokens
        else:
            try:
                rows, reverse_kl = _teacher_rows(self._teacher, datums)
            finally:
                # Same contract as the topk_ce branch above.
                self._budget.charge("teacher_prefill", self._teacher.usage() - teacher_usage_before)
            trained, adv_stats = attach_advantages(datums, rows, cfg)
            # importance_sampling and ppo ride identical datums (see
            # `to_tinker_datums`); only the service-side loss differs.
            loss_fn = ADVANTAGE_LOSS_BY_MODE[cfg.train.loss]
            mismatch_drops = adv_stats.mismatch_drops
            advantage_mean = adv_stats.advantage_mean
            advantage_std = adv_stats.advantage_std
            clipped_tokens = adv_stats.clipped_tokens
            trained_loss_tokens = adv_stats.loss_tokens
            trained_context_tokens = adv_stats.context_tokens

        # In topk_ce mode this is k x the source CE volume (the k rank
        # replicas each carry the full sequence), which is exactly what
        # forward_backward consumes, so the student_train meter stays honest.
        train_tokens = sum(len(datum.model_input_tokens) for datum in trained)
        pg_loss: float | None = None
        grad_norm: float | None = None
        if trained:
            logger.info(
                "step %d: %d datum(s) into forward_backward under loss_fn %r (train.loss = %r)",
                step,
                len(trained),
                loss_fn,
                cfg.train.loss,
            )
            train_output = self._training.forward_backward(trained, loss_fn=loss_fn)
            optim_output = self._training.optim_step(cfg.train.learning_rate)
            pg_loss = train_output.loss
            grad_norm = optim_output.grad_norm
            self._budget.charge("student_train", train_tokens)
        else:
            logger.warning(
                "step %d produced no trainable datums (%d trial(s), %d without spans, "
                "%d overflow drop(s), %d overlong drop(s), %d teacher mismatch "
                "drop(s)); skipping the optimizer step",
                step,
                roll_stats.trials,
                roll_stats.empty_span_trials,
                datum_stats.overflow_drops,
                datum_stats.overlong_drops,
                mismatch_drops,
            )

        deltas = self._meter_deltas()
        metrics = StepMetrics(
            tasks=len(batch),
            trials=roll_stats.trials,
            solve_rate=roll_stats.solve_rate,
            graded_solve_rate=roll_stats.graded_solve_rate,
            graded_trials=roll_stats.graded_trials,
            raw_solve_rate=roll_stats.raw_solve_rate,
            executed_trials=roll_stats.executed_trials,
            infra_failed_trials=roll_stats.infra_failed_trials,
            scaffold_loss_rate=roll_stats.scaffold_loss_rate,
            stop_reason_counts=dict(roll_stats.stop_reason_counts),
            empty_span_trials=roll_stats.empty_span_trials,
            truncated_spans=roll_stats.truncated_spans,
            datums=len(trained),
            fragments=datum_stats.fragments,
            fragmentation_rate=datum_stats.fragmentation_rate,
            overflow_drops=datum_stats.overflow_drops,
            overlong_drops=datum_stats.overlong_drops,
            mismatch_drops=mismatch_drops,
            clipped_tokens=clipped_tokens,
            loss_tokens=trained_loss_tokens,
            context_tokens=trained_context_tokens,
            reverse_kl_per_token=reverse_kl,
            entropy_per_token=health.entropy_per_token,
            mean_generation_tokens=health.mean_generation_tokens,
            entropy_baseline=(
                tripwire_baseline.entropy_per_token if tripwire_baseline is not None else None
            ),
            entropy_ratio=metric_ratio(
                health.entropy_per_token,
                tripwire_baseline.entropy_per_token if tripwire_baseline is not None else None,
            ),
            generation_tokens_baseline=(
                tripwire_baseline.mean_generation_tokens if tripwire_baseline is not None else None
            ),
            generation_tokens_ratio=metric_ratio(
                health.mean_generation_tokens,
                tripwire_baseline.mean_generation_tokens if tripwire_baseline is not None else None,
            ),
            reward_mean=(
                sum(record.reward for record in records) / len(records) if records else None
            ),
            loss=cfg.train.loss,
            advantage_mean=advantage_mean,
            advantage_std=advantage_std,
            clip_fraction=(clipped_tokens / trained_loss_tokens if trained_loss_tokens else 0.0),
            pg_loss=pg_loss,
            grad_norm=grad_norm,
            sampler_path=sampler_path,
            student_prefill_tokens=deltas["student_prefill"],
            student_cached_prefill_tokens=deltas["student_cached_prefill"],
            student_sample_tokens=deltas["student_sample"],
            student_train_tokens=deltas["student_train"],
            teacher_prefill_tokens=deltas["teacher_prefill"],
            teacher_cached_prefill_tokens=deltas["teacher_cached_prefill"],
            teacher_sample_tokens=deltas["teacher_sample"],
            usd=max(self._budget.session_usd - self._prev_usd, 0.0),
            cumulative_usd=max(self._budget.total_usd, 0.0),
        )
        self._store.append_metrics(step, metrics)
        self._tracker.log_step(step, metrics)
        self._prev_tokens = {meter: self._budget.tokens(meter) for meter in METER_NAMES}
        self._prev_usd = self._budget.session_usd
        kl_text = "n/a" if reverse_kl is None else f"{reverse_kl:.4f}"
        self._emit(
            "training",
            f"step {step + 1}/{cfg.train.steps}: solve rate "
            f"{roll_stats.solve_rate:.2f} ({roll_stats.executed_trials}/{roll_stats.trials} "
            f"executed), "
            f"{_graded_phrase(roll_stats.graded_solve_rate, roll_stats.graded_trials)}, "
            f"scaffold loss {roll_stats.scaffold_loss_rate:.0%}, reverse KL/token "
            f"{kl_text}, {health_summary(health, tripwire_baseline)}, "
            f"{len(trained)} datum(s) under {cfg.train.loss}",
            step=step,
        )

        try:
            self._budget.check()
        except BudgetExhausted as exc:
            self._abort_for_budget(exc, completed_step=step)

        self._check_empty_batch_streak(
            step, roll_stats.trials, roll_stats.empty_span_trials, len(trained)
        )
        self._check_degeneration(step, health, tripwire_baseline)

        if (step + 1) % cfg.train.sampler_refresh_every == 0:
            self._sampler.refresh(step + 1)
        if (step + 1) % cfg.train.save_state_every == 0:
            state_path = self._training.save_state()
            self._store.record_checkpoint(step, state_path, self._sampler.sampler_path)
        if cfg.eval.every > 0 and (step + 1) % cfg.eval.every == 0:
            subsample = self._interim_eval_tasks()
            self._run_eval(
                f"step-{step:04d}",
                subsample,
                cfg.eval.k,
                self._student_provider(),
                phase="eval",
                teacher_metered=False,
                completed_step=step,
            )

    def _interim_eval_tasks(self) -> list[str]:
        """A fixed, seeded train subsample so interim evals compare across steps."""
        count = min(self._cfg.eval.tasks, len(self._train_ids))
        rng = random.Random(_seed_from_name(f"{self._name}/interim-eval"))
        return rng.sample(self._train_ids, count)

    # -- execution -------------------------------------------------------------------------------

    def _open_training_client(self, restore_path: str | None) -> DistillTrainingClient:
        """Create the student's training client, restoring state as its first call.

        Tinker accepts LoadWeights only on an uninitialized model, so on
        resume `load_state` must run before ANYTHING else touches the client:
        a sampler-weights save, a forward pass, or a preflight sample would
        make the restore impossible and silently lose every trained step. For
        the same reason a deadline expiry cannot be retried on the same
        client, since the abandoned request may have initialized it
        server-side (a live resume died exactly that way); the retry opens a
        fresh, still-uninitialized client instead.

        Args:
            restore_path: The tinker:// state path to restore, or None for a
                fresh run, which starts from the base model with no restore.

        Returns:
            The training client, with `restore_path` already loaded when one
            was given.

        Raises:
            TinkerDeadlineError: If both restore attempts blow their deadline.
        """
        cfg = self._cfg
        client = self._service.create_lora_training_client(
            cfg.student.base_model, cfg.student.lora_rank
        )
        if restore_path is None:
            return client
        try:
            client.load_state(restore_path)
        except TinkerDeadlineError as exc:
            logger.warning(
                "load_state timed out; retrying on a FRESH training client, because the "
                "abandoned request may have initialized this one (tinker then refuses "
                "LoadWeights): %s",
                exc,
            )
            client = self._service.create_lora_training_client(
                cfg.student.base_model, cfg.student.lora_rank
            )
            client.load_state(restore_path)
        return client

    def execute(self, *, resume: bool) -> DistillResult:
        """Run (or resume) the whole distillation to a gate verdict."""
        cfg = self._cfg
        store = self._store
        self._resumed = resume
        if resume:
            if not store.config_path.exists():
                raise RuntimeError(
                    f"nothing to resume in {self._run_dir}: no config.toml snapshot "
                    "from a prior run; start a fresh run without resume"
                )
            if store.gate_path.exists():
                raise RuntimeError(
                    f"the run in {self._run_dir} already completed: its gate verdict is "
                    f"recorded at {store.gate_path} (see model_card.json for the final "
                    "artifacts). Resuming a finished run would re-spend the holdout eval "
                    "and could promote a duplicate adapter version; start a fresh "
                    "--run-dir to distill again"
                )
            latest = store.latest_checkpoint()
            start_step = latest.step + 1 if latest is not None else 0
            offpolicy_record = store.read_offpolicy()
            offpolicy_cursor = store.read_offpolicy_cursor()
            warmup_record = store.read_warmup()
            # The spend ledger is written on every charge, so it carries spend
            # the metrics rows never see (baselines, interim and finalize
            # evals); summing metrics rows is only the pre-ledger fallback.
            recorded_spend = store.read_spend()
            prior_usd = recorded_spend if recorded_spend is not None else store.budget_spent()
            # Reused exactly as recorded, never re-measured: a fresh baseline
            # taken after a collapse would treat the collapse as normal and the
            # tripwire would never fire again. None only when the prior session
            # never completed a measurable step.
            self._tripwire_baseline = store.read_tripwire_baseline()
            if self._tripwire_baseline is not None:
                logger.info(
                    "reusing the recorded degeneration baseline from step %d (entropy "
                    "%.4f nats/token, %.0f sampled tokens/episode)",
                    self._tripwire_baseline.step,
                    self._tripwire_baseline.entropy_per_token,
                    self._tripwire_baseline.mean_generation_tokens,
                )
        else:
            if store.config_path.exists() or store.last_step() is not None:
                raise ValueError(
                    f"run dir {self._run_dir} already holds a distillation run; pass "
                    "resume=True to continue it, or choose a fresh run dir"
                )
            latest = None
            start_step = 0
            offpolicy_record = None
            offpolicy_cursor = None
            warmup_record = None
            prior_usd = 0.0
        # The snapshot is refreshed on resume too: the caller's config wins (the
        # documented budget-abort recovery is editing budget.max_usd and resuming).
        store.snapshot_config(cfg)

        restore_path: str | None = None
        if latest is not None:
            logger.info(
                "resuming %s from checkpoint step %d (%s)",
                self._name,
                latest.step,
                latest.state_path,
            )
            restore_path = latest.state_path
        elif offpolicy_record is not None and offpolicy_record.state_path is not None:
            # The off-policy phase finished but no OPD step was checkpointed
            # yet: restore what it trained instead of starting OPD cold.
            logger.info(
                "resuming %s from the post-off-policy state (%s)",
                self._name,
                offpolicy_record.state_path,
            )
            restore_path = offpolicy_record.state_path
        elif offpolicy_cursor is not None:
            # An INTERRUPTED off-policy phase: its cursor names the state saved
            # at the last checkpointed step, and the phase skips that prefix.
            logger.info(
                "resuming %s mid-off-policy from %s (%d step(s) already applied)",
                self._name,
                offpolicy_cursor.state_path,
                offpolicy_cursor.steps_completed,
            )
            restore_path = offpolicy_cursor.state_path
        elif warmup_record is not None and warmup_record.state_path is not None:
            # Warmup finished but no OPD step was checkpointed yet: without
            # this restore, a resumed session would start OPD from the COLD
            # student and silently lose the warmup it already paid for.
            logger.info(
                "resuming %s from the post-warmup state (%s)",
                self._name,
                warmup_record.state_path,
            )
            restore_path = warmup_record.state_path
        # The restore is part of opening the client because it must be its
        # FIRST call (see _open_training_client): the teacher client, the
        # student sampler refresh, and preflight all come after, so preflight
        # validates the RESTORED weights rather than the bare base model.
        self._training = self._open_training_client(restore_path)
        self._teacher_client = self._service.create_sampling_client(self._teacher_identity)
        self._teacher = TinkerTeacher(cfg.teacher, sampling_client=self._teacher_client)
        self._sampler = StudentSampler(self._service, self._training, self._name)
        self._sampler.refresh(start_step)
        self._budget = _RunBudget(
            cfg.pricing, cfg.budget.max_usd, prior_usd, on_spend=store.write_spend
        )
        self._tasks = TaskSampler(self._train_ids, seed=_seed_from_name(self._name))
        for _ in range(start_step):
            self._tasks.next_batch(cfg.train.tasks_per_batch)

        self._preflight()
        try:
            self._budget.check()
        except BudgetExhausted as exc:
            self._abort_for_budget(exc, completed_step=None)

        teacher_report = self._eval_or_load(
            TEACHER_BASELINE_EVAL,
            self._holdout_ids,
            cfg.gate.k,
            self._teacher_provider(),
            phase="baseline",
            teacher_metered=True,
            reuse=resume,
            pin_provider=True,
            baseline_from=cfg.eval.teacher_baseline_from,
            baseline_from_field="eval.teacher_baseline_from",
        )
        before_report = self._eval_or_load(
            STUDENT_BEFORE_EVAL,
            self._holdout_ids,
            cfg.gate.k,
            self._student_provider(),
            phase="baseline",
            teacher_metered=False,
            reuse=resume,
            baseline_from=cfg.eval.student_baseline_from,
            baseline_from_field="eval.student_baseline_from",
        )

        if cfg.offpolicy.epochs > 0:
            if offpolicy_record is not None:
                logger.info(
                    "off-policy distillation already recorded in %s; skipping (%s)",
                    store.offpolicy_path,
                    offpolicy_record.skipped_reason
                    or f"{offpolicy_record.steps} step(s) over {offpolicy_record.datums} datum(s)",
                )
            elif start_step > 0:
                logger.info(
                    "resumed past step checkpoints (start step %d) with no off-policy "
                    "record; skipping the phase, the student already trained OPD steps",
                    start_step,
                )
            else:
                self._offpolicy(offpolicy_cursor)

        if cfg.warmup.steps > 0:
            if warmup_record is not None:
                logger.info(
                    "warmup already recorded in %s; skipping (%s)",
                    store.warmup_path,
                    warmup_record.skipped_reason
                    or f"{warmup_record.steps} step(s) over {warmup_record.datums} datum(s)",
                )
            elif start_step > 0:
                logger.info(
                    "resumed past step checkpoints (start step %d) with no warmup "
                    "record; skipping warmup, the student already trained OPD steps",
                    start_step,
                )
            else:
                self._warmup()

        if start_step >= cfg.train.steps:
            logger.info(
                "all %d training step(s) already completed; skipping to finalize",
                cfg.train.steps,
            )
        for step in range(start_step, cfg.train.steps):
            self._train_step(step)

        return self._finalize(teacher_report, before_report)

    def _finalize(
        self, teacher_report: DistillEvalReport, before_report: DistillEvalReport
    ) -> DistillResult:
        cfg = self._cfg
        self._emit("finalize", "saving final training state and sampler weights")
        final_sampler = self._sampler.refresh(cfg.train.steps)
        final_state = self._training.save_state()
        final_step = max(cfg.train.steps - 1, 0)
        self._store.record_checkpoint(final_step, final_state, final_sampler)

        # Reuse a recorded student-after report on resume: a budget abort inside
        # a prior session's finalize already paid for it, and the resumed weights
        # restore the same final checkpoint it measured.
        after_report = self._eval_or_load(
            STUDENT_AFTER_EVAL,
            self._holdout_ids,
            cfg.gate.k,
            self._student_provider(),
            phase="eval",
            teacher_metered=False,
            reuse=self._resumed,
            completed_step=final_step,
        )
        record = gate_distillation(
            teacher_report.solve_rate,
            before_report.solve_rate,
            after_report.solve_rate,
            cfg.gate,
        )
        self._store.write_gate(record)
        card = DistillModelCard(
            base_model=cfg.student.base_model,
            lora_rank=cfg.student.lora_rank,
            teacher_model=self._teacher_identity,
            sampler_path=final_sampler,
            state_path=final_state,
            steps_completed=cfg.train.steps,
            gate=record,
        )
        self._store.write_model_card(card)
        version: int | None = None
        if record.accepted:
            version = self._adapters.save_version(self._name, card)
            self._store.write_handoff(
                build_handoff_toml(final_sampler, base_model=cfg.student.base_model)
            )
            logger.info("adapter %s v%d promoted: %s", self._name, version, record.reason)
        else:
            logger.warning("adapter %s not promoted: %s", self._name, record.reason)
        self._emit("gate", record.reason)
        self._tracker.log_summary(
            gate_accepted=record.accepted,
            gate_reason=record.reason,
            teacher_solve_rate=record.teacher_solve_rate,
            student_before_solve_rate=record.student_before_solve_rate,
            student_after_solve_rate=record.student_after_solve_rate,
            total_usd=self._budget.total_usd,
            steps_completed=cfg.train.steps,
        )
        spend = SpendSummary(
            lines=self._budget.lines(),
            session_usd=self._budget.session_usd,
            prior_usd=self._budget.prior_usd,
            total_usd=self._budget.total_usd,
        )
        return DistillResult(
            name=self._name,
            run_dir=str(self._run_dir),
            steps_completed=cfg.train.steps,
            final_sampler_path=final_sampler,
            final_state_path=final_state,
            gate=record,
            adapter_version=version,
            spend=spend,
        )


def run_distillation(
    name: str,
    cfg: DistillConfig,
    harness: HarnessDoc,
    train_task_ids: Sequence[str],
    holdout_task_ids: Sequence[str],
    run_dir: Path,
    *,
    resume: bool = False,
    on_progress: ProgressCallback | None = None,
    service_client: DistillServiceClient | None = None,
    adapter_store: AdapterStore | None = None,
    live_trial_preflight: LiveTrialPreflight | None = None,
    tracker: DistillTracker | None = None,
    cli_agent: str | None = None,
) -> DistillResult:
    """Run one on-policy distillation end to end (preflight to gate verdict).

    The flow: preflight (renderer resolution, tokenizer fingerprint, one-token
    student and teacher pings, the sample-then-recompute TITO proof, and the
    optional live-trial hook), holdout baselines (teacher-in-harness and
    student-before, each importable from a prior run's report via
    `eval.teacher_baseline_from` / `eval.student_baseline_from` instead of
    running trials), the optional supervised warmup when `warmup.steps > 0`
    (teacher rollouts on the train split, or another run's recorded collection
    when `warmup.trajectories_from` is set; the `warmup.keep`-filtered trials
    merge into cross_entropy datums for `warmup.steps` full-batch passes, then
    a state save and a forced sampler refresh so OPD starts from the warmed
    student; zero kept trials skip the phase with a warning instead of
    aborting), the
    training step loop (harbor rollouts, prefix-merge datums, teacher scoring,
    reverse-KL advantages, importance_sampling forward/backward and one
    optimizer step, metrics row, budget enforcement, then the sampler-refresh
    / save-state / interim-eval cadences), and finally the student-after
    holdout eval feeding `gate_distillation`. An accepted gate saves the
    adapter version (champion alias) and writes the serving handoff into the
    run dir.

    Args:
        name: The adapter/run name (a safe single path segment).
        cfg: The validated run config; snapshotted into the run dir.
        harness: The pinned document whose hash keys every trial's harbor job
            dir; the rollout agent itself is harbor's terminus-2.
        train_task_ids: The train split's task ids (unique, non-empty).
        holdout_task_ids: The holdout split's task ids (unique, non-empty,
            disjoint from the train split); baselines and the gate run here.
        run_dir: The run's artifact directory (`DistillRunStore` layout plus
            the rollout collector's per-step dirs and per-eval
            `eval-rollouts/<name>/` roots).
        resume: Continue an aborted run in `run_dir`: restores the latest
            checkpoint via `load_state` (or the recorded post-warmup state
            when no step checkpoint exists yet), continues the step count,
            restores prior USD spend from the run store's spend ledger,
            reuses recorded baseline evals, and never re-runs a warmup whose
            `warmup.json` record exists. The passed `cfg` wins over the old
            snapshot (the budget-abort recovery is editing `budget.max_usd`
            and resuming).
        on_progress: Optional callback receiving typed `DistillProgress`
            events as phases advance.
        service_client: Injectable Tinker service surface; tests pass fakes
            built on `wmh.distill.fake_tinker`. None builds the real SDK
            client lazily (requires the distill extra and TINKER_API_KEY).
        adapter_store: Where accepted adapters are versioned; defaults to the
            project-local `.wmh` store.
        live_trial_preflight: The Phase 6 live single-trial pi preflight hook
            (see `LiveTrialPreflight`); None skips it.
        tracker: Injectable run tracker; None builds one from the config's
            `[wandb]` section via `build_tracker` (the no-op `NullTracker`
            unless `wandb.enabled` is set). `finish()` is guaranteed on both
            the finalize and budget-abort paths.
        cli_agent: The agent string as the user typed it (it may carry an
            `@ref`, which `name` strips); resume commands print this string
            so the printed command passes the CLI's resume conflict check.
            None falls back to `name`.

    Returns:
        The `DistillResult`: gate record, final sampler/state paths, run dir,
        adapter version (None when rejected), and the spend summary.

    Raises:
        ValueError: On invalid name/splits, a fresh run pointed at a used
            run dir, a `eval.*_baseline_from` report that fails validation
            (wrong task set, too few attempts, or a different model), or
            wandb tracking enabled without credentials.
        RuntimeError: On preflight failures (each message says what to fix),
            or resume with nothing to resume.
        ImportError: If no service client is injected and the tinker SDK is
            not installed, or wandb tracking is enabled without the wandb SDK.
        DistillBudgetError: When `budget.max_usd` is exhausted; state is
            persisted and the error carries the exact resume command.
        DistillEmptyBatchError: When `MAX_CONSECUTIVE_EMPTY_STEPS` training
            steps in a row produce only span-less trials (the student
            provider is producing no completions); state is persisted and
            the error carries the exact resume command.
    """
    validate_name(name)
    train_ids = list(train_task_ids)
    holdout_ids = list(holdout_task_ids)
    if not train_ids or len(set(train_ids)) != len(train_ids):
        raise ValueError(
            "train_task_ids must be non-empty and unique; pass the train split's exact task ids"
        )
    if not holdout_ids or len(set(holdout_ids)) != len(holdout_ids):
        raise ValueError(
            "holdout_task_ids must be non-empty and unique; the baselines and the "
            "promotion gate are measured on the holdout split"
        )
    overlap = sorted(set(train_ids) & set(holdout_ids))
    if overlap:
        raise ValueError(
            f"task id(s) {', '.join(overlap)} appear in BOTH splits; the gate is "
            "only meaningful on tasks the student never trained on, so make the "
            "splits disjoint"
        )
    service = service_client if service_client is not None else _build_sdk_service_client()
    # Built (and so credential-checked) before any spend: a misconfigured
    # tracker must fail fast, not after paid baselines.
    resolved_tracker = tracker if tracker is not None else build_tracker(cfg, run_dir, name)
    run = _DistillRun(
        name,
        cfg,
        harness,
        train_ids,
        holdout_ids,
        run_dir,
        service=service,
        adapter_store=adapter_store if adapter_store is not None else AdapterStore(),
        on_progress=on_progress,
        live_trial_preflight=live_trial_preflight,
        tracker=resolved_tracker,
        cli_agent=cli_agent,
    )
    try:
        return run.execute(resume=resume)
    finally:
        # Both terminal paths (finalize and the budget abort) close the
        # tracking run, so a resumed session starts a fresh wandb run.
        resolved_tracker.finish()
