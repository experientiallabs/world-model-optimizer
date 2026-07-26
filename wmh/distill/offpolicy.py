"""Off-policy distillation: hard-target cross entropy over teacher trajectories.

The student never samples here. The corpus is trajectories the TEACHER
produced through the same pi harness, merged into `TrainDatum`s by the same
prefix merge on-policy distillation uses, and the objective is Tinker's
`cross_entropy` loss over the teacher's realized tokens (they are exact
sampled ids, so no advantages are involved and `to_tinker_sft_datums` is the
wire path).

This module owns only the SCHEDULE and its resume arithmetic, which is what
makes the mode repeatable:

- `plan_offpolicy_steps` expands a schedule into the exact list of optimizer
  steps, each carrying the datum indices its minibatch trains on. It is pure,
  so the plan a resumed session builds is byte-identical to the plan the
  interrupted one was walking, and "resume" is simply "skip the first
  `steps_completed` positions".
- `run_offpolicy` walks that plan against an injected trainer, reporting every
  completed step and every checkpoint back through callbacks. Everything the
  orchestrator owns (metering, metrics rows, budget enforcement, saving
  training state, writing the cursor) lives in those callbacks, so this module
  needs no run store, no budget meter, and no Tinker SDK.

Datums are never split or truncated to make a minibatch even: the last
minibatch of an epoch is short when the corpus does not divide evenly, and
every datum trains in every epoch.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wmh.distill.data import TrainDatum

if TYPE_CHECKING:
    # Imported for typing only: `wmh.distill.loop` imports this module, so a
    # runtime import here would be a real circular dependency (the same reason
    # `wmh.distill.tracking` type-imports its metrics models from the loop).
    from wmh.distill.loop import OptimStepOutput, TrainStepOutput

logger = logging.getLogger(__name__)

CROSS_ENTROPY_LOSS = "cross_entropy"
"""Tinker's supervised loss (see `to_tinker_sft_datums` for its pinned keyset).

Both cross_entropy consumers name it from here: this module's off-policy
schedule, and the loop's `topk_ce` rank replicas.
"""


class OffPolicyTrainer(Protocol):
    """The training-client slice the off-policy schedule drives.

    `wmh.distill.loop.DistillTrainingClient` satisfies it structurally, so the
    loop passes its own client (real SDK or fake) straight through.
    """

    def forward_backward(self, datums: Sequence[TrainDatum], loss_fn: str) -> TrainStepOutput:
        """Accumulate gradients for one minibatch under the named loss."""
        ...

    def optim_step(self, learning_rate: float) -> OptimStepOutput:
        """Apply one optimizer step."""
        ...


class OffPolicySchedule(BaseModel):
    """The resolved epoch/minibatch schedule one off-policy phase runs.

    Resolved means every value is concrete: `learning_rate` has already
    absorbed the `train.learning_rate` fallback, and `minibatch_datums` still
    carries the config's 0-means-full-batch convention because the batch size
    depends on the corpus size, which the schedule does not know.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    epochs: int = Field(ge=0)
    """Passes over the whole datum set; 0 plans no steps at all."""

    minibatch_datums: int = Field(default=0, ge=0)
    """Datums per optimizer step; 0 means one full-batch step per epoch."""

    learning_rate: float = Field(gt=0)

    shuffle_seed: int | None = None
    """Seed for the per-epoch shuffle; None keeps the corpus in build order."""

    checkpoint_every: int = Field(default=0, ge=0)
    """Emit a resume cursor every N optimizer steps; 0 disables checkpointing
    entirely, which is the legacy warmup phase's contract (it re-runs whole)."""


class OffPolicyStepPlan(BaseModel):
    """One scheduled optimizer step: where it sits, and what it trains on."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int = Field(ge=0)
    """0-based optimizer step within the whole phase (across every epoch)."""

    epoch: int = Field(ge=0)
    minibatch: int = Field(ge=0)
    """0-based minibatch index within `epoch`."""

    datum_indices: list[int]
    """Indices into the corpus this minibatch trains on, in training order."""

    @model_validator(mode="after")
    def _check_non_empty(self) -> OffPolicyStepPlan:
        """Reject a step that would train on nothing."""
        if not self.datum_indices:
            raise ValueError("an off-policy step plan must carry at least one datum index")
        return self


class OffPolicyStepResult(BaseModel):
    """One completed off-policy optimizer step, in executor currency.

    The orchestrator turns this into whatever metrics row its phase writes; the
    executor deliberately knows nothing about meters, prices, or run dirs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int = Field(ge=0)
    epoch: int = Field(ge=0)
    minibatch: int = Field(ge=0)
    datums: int = Field(ge=1)
    """Datums in THIS step's minibatch (not the corpus)."""

    loss_tokens: int = Field(ge=0)
    """Loss-masked (teacher-sampled) tokens in this minibatch."""

    context_tokens: int = Field(ge=0)
    """Unmasked context tokens in this minibatch."""

    train_tokens: int = Field(ge=0)
    """Tokens the forward_backward carried, i.e. what `student_train` is charged."""

    learning_rate: float = Field(gt=0)
    loss: float | None
    """The batch loss the backend reported; None when it reported none."""

    grad_norm: float | None
    """The gradient norm the optim step reported; None when it reported none."""


class OffPolicyMetrics(BaseModel):
    """One off-policy optimizer step's metrics row (the store adds `step`).

    Off-policy rows share `metrics.jsonl` with the on-policy training rows and
    their step indices restart at 0 for the OPD loop, so the constant `phase`
    field is what tells them apart (on-policy rows carry no `phase` key). The
    row's `step` key is the phase-global optimizer step, and `epoch` plus
    `minibatch` locate it inside the schedule. A phase that trained nothing
    (zero kept trials) writes exactly one row with `datums = 0` so the skip is
    visible in the metrics rather than only in the log.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    phase: Literal["offpolicy"] = "offpolicy"
    epoch: int = Field(ge=0)
    epochs: int = Field(ge=0)
    """Epochs the schedule plans, so one row is readable on its own."""

    minibatch: int = Field(ge=0)
    planned_steps: int = Field(ge=0)
    """Optimizer steps the whole schedule plans."""

    tasks: int = Field(ge=0)
    trials: int = Field(ge=0)
    """Teacher trials in the corpus collection (tasks x rollouts_per_task)."""

    kept_trials: int = Field(ge=0)
    """Trials that survived the `offpolicy.keep` filter."""

    solve_rate: float = Field(ge=0.0, le=1.0)
    """The teacher's solve rate over the collected trials."""

    corpus_datums: int = Field(ge=0)
    """Datums in the whole corpus (what an epoch covers)."""

    datums: int = Field(ge=0)
    """Datums in THIS step's minibatch; equals `corpus_datums` at full batch."""

    loss_tokens: int = Field(ge=0)
    context_tokens: int = Field(ge=0)
    learning_rate: float = Field(gt=0)
    """The effective LR (offpolicy.learning_rate or train.learning_rate)."""

    loss: float | None
    """The batch loss the backend reported; None when it reported none."""

    grad_norm: float | None
    """The gradient norm the optim step reported; None when it reported none."""

    student_prefill_tokens: int = Field(ge=0)
    student_cached_prefill_tokens: int = Field(ge=0)
    student_sample_tokens: int = Field(ge=0)
    student_train_tokens: int = Field(ge=0)
    teacher_prefill_tokens: int = Field(ge=0)
    teacher_cached_prefill_tokens: int = Field(ge=0)
    teacher_sample_tokens: int = Field(ge=0)
    usd: float = Field(ge=0.0)
    """Priced spend since the previous metrics row (the teacher collection and
    any earlier baseline spend fold into off-policy step 0's row)."""


class OffPolicyCursorPosition(BaseModel):
    """Where a checkpointed off-policy phase would resume.

    Emitted by `run_offpolicy` after a checkpointed step; the orchestrator
    saves training state and persists it as the run store's cursor.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    steps_completed: int = Field(ge=1)
    """Optimizer steps of this phase already applied to the student."""

    epoch: int = Field(ge=0)
    """Epoch the NEXT step belongs to."""

    minibatch: int = Field(ge=0)
    """Minibatch within `epoch` the NEXT step trains."""

    datums: int = Field(ge=1)
    """Corpus size the plan was built over. A resumed session whose corpus has
    a different size is walking a different plan, so it must discard the cursor
    rather than land its step count on the wrong minibatches."""


class OffPolicyOutcome(BaseModel):
    """What one `run_offpolicy` call did, for the phase's terminal record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    planned_steps: int = Field(ge=0)
    """Optimizer steps the whole schedule plans, across every session."""

    steps_run: int = Field(ge=0)
    """Optimizer steps THIS call applied (planned_steps minus the resumed prefix)."""

    epochs: int = Field(ge=0)
    """Epochs the schedule plans."""


OffPolicyStepSink = Callable[[OffPolicyStepResult], None]
"""Called after every optimizer step, before the next one is planned.

The orchestrator charges the step's training tokens, writes its metrics row,
and enforces the budget here; raising aborts the phase with the student's
weights exactly at this step, which is what makes the last cursor honest.
"""

OffPolicyCheckpointSink = Callable[[OffPolicyCursorPosition], None]
"""Called every `checkpoint_every` steps (never after the final step).

The orchestrator saves training state and rewrites the resume cursor here.
"""


def plan_offpolicy_steps(datum_count: int, schedule: OffPolicySchedule) -> list[OffPolicyStepPlan]:
    """Expand a schedule into the exact optimizer steps it runs.

    Pure and deterministic: the same (corpus size, schedule) always produces
    the same plan, including the per-epoch shuffle, which is seeded from
    `(shuffle_seed, epoch)` rather than from a running generator so that
    resuming into the middle of epoch 3 replays epoch 3's exact order.

    Args:
        datum_count: Size of the corpus the phase trains on.
        schedule: The resolved schedule.

    Returns:
        Every optimizer step in training order; empty when the corpus is empty
        or the schedule plans no epochs.
    """
    if datum_count <= 0 or schedule.epochs == 0:
        return []
    size = schedule.minibatch_datums or datum_count
    plans: list[OffPolicyStepPlan] = []
    step = 0
    for epoch in range(schedule.epochs):
        order = list(range(datum_count))
        if schedule.shuffle_seed is not None:
            # A string seed so the (seed, epoch) pair hashes reproducibly across
            # processes; int seeds would have to be combined by hand.
            random.Random(f"{schedule.shuffle_seed}:{epoch}").shuffle(order)
        for minibatch, start in enumerate(range(0, datum_count, size)):
            plans.append(
                OffPolicyStepPlan(
                    step=step,
                    epoch=epoch,
                    minibatch=minibatch,
                    datum_indices=order[start : start + size],
                )
            )
            step += 1
    return plans


def _minibatch_totals(datums: Sequence[TrainDatum]) -> tuple[int, int, int]:
    """One minibatch's (loss tokens, context tokens, total tokens)."""
    train_tokens = sum(len(datum.model_input_tokens) for datum in datums)
    loss_tokens = sum(datum.loss_token_count for datum in datums)
    return loss_tokens, train_tokens - loss_tokens, train_tokens


def run_offpolicy(
    datums: Sequence[TrainDatum],
    schedule: OffPolicySchedule,
    *,
    trainer: OffPolicyTrainer,
    on_step: OffPolicyStepSink,
    on_checkpoint: OffPolicyCheckpointSink | None = None,
    start_step: int = 0,
) -> OffPolicyOutcome:
    """Train the corpus for `schedule.epochs` passes of cross_entropy steps.

    Each planned step is one `forward_backward` over its minibatch plus one
    `optim_step`, reported through `on_step` in that order (the orchestrator's
    charge, metrics row, and budget check all hang off it). Every
    `schedule.checkpoint_every` steps the next position goes to
    `on_checkpoint`; the FINAL step never checkpoints, because the phase's own
    terminal record supersedes any cursor at that point.

    Args:
        datums: The corpus, in build order. `start_step` is only meaningful
            against the identical corpus (the caller validates that through the
            cursor's recorded datum count).
        schedule: The resolved schedule.
        trainer: The training client the steps run against.
        on_step: Called after each completed optimizer step.
        on_checkpoint: Called at the checkpoint cadence; None disables
            checkpointing regardless of `schedule.checkpoint_every`.
        start_step: Optimizer steps already applied by an earlier session,
            skipped without training.

    Returns:
        The plan size, the steps this call actually ran, and the epoch count.

    Raises:
        ValueError: If `start_step` is negative or exceeds the plan.
    """
    plans = plan_offpolicy_steps(len(datums), schedule)
    if start_step < 0 or start_step > len(plans):
        raise ValueError(
            f"off-policy resume asked to skip {start_step} step(s), but the schedule plans "
            f"{len(plans)} over {len(datums)} datum(s); the recorded cursor does not belong "
            "to this corpus, so delete offpolicy-cursor.json to restart the phase"
        )
    if start_step:
        logger.info(
            "resuming off-policy training at step %d/%d (epoch %d, minibatch %d)",
            start_step,
            len(plans),
            plans[start_step].epoch if start_step < len(plans) else schedule.epochs,
            plans[start_step].minibatch if start_step < len(plans) else 0,
        )
    for plan in plans[start_step:]:
        batch = [datums[index] for index in plan.datum_indices]
        train_output = trainer.forward_backward(batch, CROSS_ENTROPY_LOSS)
        optim_output = trainer.optim_step(schedule.learning_rate)
        loss_tokens, context_tokens, train_tokens = _minibatch_totals(batch)
        on_step(
            OffPolicyStepResult(
                step=plan.step,
                epoch=plan.epoch,
                minibatch=plan.minibatch,
                datums=len(batch),
                loss_tokens=loss_tokens,
                context_tokens=context_tokens,
                train_tokens=train_tokens,
                learning_rate=schedule.learning_rate,
                loss=train_output.loss,
                grad_norm=optim_output.grad_norm,
            )
        )
        completed = plan.step + 1
        is_final = completed == len(plans)
        if (
            on_checkpoint is not None
            and schedule.checkpoint_every
            and not is_final
            and completed % schedule.checkpoint_every == 0
        ):
            following = plans[completed]
            on_checkpoint(
                OffPolicyCursorPosition(
                    steps_completed=completed,
                    epoch=following.epoch,
                    minibatch=following.minibatch,
                    datums=len(datums),
                )
            )
    return OffPolicyOutcome(
        planned_steps=len(plans),
        steps_run=max(len(plans) - start_step, 0),
        epochs=schedule.epochs if plans else 0,
    )
