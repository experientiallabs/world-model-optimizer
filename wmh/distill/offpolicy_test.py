"""Tests for the off-policy schedule: its plan, its walk, and its resume arithmetic."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from wmh.distill.data import TrainDatum
from wmh.distill.loop import OptimStepOutput, TrainStepOutput
from wmh.distill.offpolicy import (
    CROSS_ENTROPY_LOSS,
    OffPolicyCursorPosition,
    OffPolicySchedule,
    OffPolicyStepPlan,
    OffPolicyStepResult,
    plan_offpolicy_steps,
    run_offpolicy,
)


class _RecordingTrainer:
    """An `OffPolicyTrainer` logging every batch and learning rate it is handed."""

    def __init__(self, *, loss: float | None = 1.5, grad_norm: float | None = 0.25) -> None:
        self.batches: list[tuple[tuple[str, ...], str]] = []
        self.learning_rates: list[float] = []
        self._loss = loss
        self._grad_norm = grad_norm

    def forward_backward(self, datums: Sequence[TrainDatum], loss_fn: str) -> TrainStepOutput:
        self.batches.append((tuple(datum.trial_name for datum in datums), loss_fn))
        return TrainStepOutput(loss=self._loss)

    def optim_step(self, learning_rate: float) -> OptimStepOutput:
        self.learning_rates.append(learning_rate)
        return OptimStepOutput(grad_norm=self._grad_norm)


def _datum(name: str, *, sampled: int = 2, context: int = 3) -> TrainDatum:
    """One trivially aligned datum with `context` prompt then `sampled` tokens."""
    tokens = list(range(context + sampled))
    return TrainDatum(
        trial_name=name,
        fragment_index=0,
        model_input_tokens=tokens,
        loss_mask=[0.0] * context + [1.0] * sampled,
        sampled_logprobs=[0.0] * context + [-0.5] * sampled,
    )


def _corpus(count: int) -> list[TrainDatum]:
    return [_datum(f"trial-{index}") for index in range(count)]


def _schedule(**overrides: object) -> OffPolicySchedule:
    fields: dict[str, object] = {"epochs": 2, "learning_rate": 1e-4}
    fields.update(overrides)
    return OffPolicySchedule.model_validate(fields)


# -- the plan --------------------------------------------------------------------------------


def test_full_batch_plans_one_step_per_epoch_in_build_order() -> None:
    # minibatch_datums = 0 is the legacy warmup shape: one forward/backward over
    # every datum per epoch, and no reordering at all.
    plans = plan_offpolicy_steps(4, _schedule(epochs=3))

    assert [plan.step for plan in plans] == [0, 1, 2]
    assert [plan.epoch for plan in plans] == [0, 1, 2]
    assert all(plan.minibatch == 0 for plan in plans)
    assert all(plan.datum_indices == [0, 1, 2, 3] for plan in plans)


def test_minibatch_datums_splits_each_epoch_into_optimizer_steps() -> None:
    # 5 datums at 2 per step is 3 minibatches per epoch, and the last one is
    # SHORT rather than padded or dropped: no datum is ever left out of an epoch.
    plans = plan_offpolicy_steps(5, _schedule(epochs=2, minibatch_datums=2))

    assert [(plan.epoch, plan.minibatch) for plan in plans] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    assert [plan.step for plan in plans] == [0, 1, 2, 3, 4, 5]
    assert [plan.datum_indices for plan in plans] == [[0, 1], [2, 3], [4]] * 2


def test_a_minibatch_larger_than_the_corpus_is_one_full_batch_step() -> None:
    plans = plan_offpolicy_steps(3, _schedule(epochs=1, minibatch_datums=64))

    assert len(plans) == 1
    assert plans[0].datum_indices == [0, 1, 2]


def test_no_epochs_or_no_datums_plans_nothing() -> None:
    assert plan_offpolicy_steps(4, _schedule(epochs=0)) == []
    assert plan_offpolicy_steps(0, _schedule(epochs=3)) == []


def test_the_shuffle_is_per_epoch_and_reproducible() -> None:
    # Reproducible is what makes the cursor meaningful: a resumed session
    # rebuilds the identical plan, so skipping N steps lands on the minibatch
    # the interrupted session was about to train.
    schedule = _schedule(epochs=3, minibatch_datums=2, shuffle_seed=7)
    first = plan_offpolicy_steps(6, schedule)
    again = plan_offpolicy_steps(6, schedule)

    assert first == again
    per_epoch = [
        [index for plan in first if plan.epoch == epoch for index in plan.datum_indices]
        for epoch in range(3)
    ]
    # Every epoch is a permutation of the whole corpus, and the epochs differ.
    assert all(sorted(order) == list(range(6)) for order in per_epoch)
    assert len({tuple(order) for order in per_epoch}) > 1
    # A different seed is a different schedule.
    assert plan_offpolicy_steps(6, _schedule(epochs=3, minibatch_datums=2, shuffle_seed=8)) != first


def test_without_a_seed_every_epoch_keeps_build_order() -> None:
    plans = plan_offpolicy_steps(4, _schedule(epochs=2, minibatch_datums=2))

    assert [plan.datum_indices for plan in plans] == [[0, 1], [2, 3], [0, 1], [2, 3]]


# -- the walk --------------------------------------------------------------------------------


def test_each_step_is_one_cross_entropy_pass_plus_one_optimizer_step() -> None:
    datums = _corpus(4)
    trainer = _RecordingTrainer()
    results: list[OffPolicyStepResult] = []

    outcome = run_offpolicy(
        datums,
        _schedule(epochs=2, minibatch_datums=2, learning_rate=3e-5),
        trainer=trainer,
        on_step=results.append,
    )

    assert outcome.planned_steps == 4
    assert outcome.steps_run == 4
    assert outcome.epochs == 2
    assert [loss_fn for _, loss_fn in trainer.batches] == [CROSS_ENTROPY_LOSS] * 4
    assert [names for names, _ in trainer.batches] == [
        ("trial-0", "trial-1"),
        ("trial-2", "trial-3"),
        ("trial-0", "trial-1"),
        ("trial-2", "trial-3"),
    ]
    assert trainer.learning_rates == [3e-5] * 4
    assert [(result.epoch, result.minibatch) for result in results] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]


def test_a_step_result_carries_its_own_minibatch_token_totals() -> None:
    # 2 datums x (3 context + 2 sampled), so the charge is per MINIBATCH, not
    # per corpus: metering a full-corpus volume on every step would triple-count
    # a three-minibatch epoch.
    trainer = _RecordingTrainer()
    results: list[OffPolicyStepResult] = []

    run_offpolicy(
        _corpus(6),
        _schedule(epochs=1, minibatch_datums=2),
        trainer=trainer,
        on_step=results.append,
    )

    assert len(results) == 3
    for result in results:
        assert result.datums == 2
        assert result.loss_tokens == 4
        assert result.context_tokens == 6
        assert result.train_tokens == 10
        assert result.loss == pytest.approx(1.5)
        assert result.grad_norm == pytest.approx(0.25)


def test_a_raising_step_sink_stops_the_schedule_where_it_raised() -> None:
    # The budget abort path: the sink raises after the step it could not afford,
    # and nothing further is trained.
    trainer = _RecordingTrainer()
    seen: list[int] = []

    def on_step(result: OffPolicyStepResult) -> None:
        seen.append(result.step)
        if result.step == 1:
            raise RuntimeError("out of budget")

    with pytest.raises(RuntimeError, match="out of budget"):
        run_offpolicy(
            _corpus(4),
            _schedule(epochs=2, minibatch_datums=2),
            trainer=trainer,
            on_step=on_step,
        )

    assert seen == [0, 1]
    assert len(trainer.batches) == 2


# -- checkpoints and resume ------------------------------------------------------------------


def test_checkpoints_name_the_next_position_and_never_follow_the_last_step() -> None:
    # The final step needs no cursor: the phase's terminal record supersedes it,
    # and a cursor left behind would send the next resume back into a finished
    # schedule.
    positions: list[OffPolicyCursorPosition] = []

    run_offpolicy(
        _corpus(4),
        _schedule(epochs=2, minibatch_datums=2, checkpoint_every=1),
        trainer=_RecordingTrainer(),
        on_step=lambda _: None,
        on_checkpoint=positions.append,
    )

    assert [
        (position.steps_completed, position.epoch, position.minibatch) for position in positions
    ] == [(1, 0, 1), (2, 1, 0), (3, 1, 1)]
    assert all(position.datums == 4 for position in positions)


def test_a_checkpoint_cadence_above_one_emits_fewer_cursors() -> None:
    positions: list[OffPolicyCursorPosition] = []

    run_offpolicy(
        _corpus(6),
        _schedule(epochs=2, minibatch_datums=2, checkpoint_every=2),
        trainer=_RecordingTrainer(),
        on_step=lambda _: None,
        on_checkpoint=positions.append,
    )

    assert [position.steps_completed for position in positions] == [2, 4]


def test_checkpointing_is_off_for_the_legacy_full_batch_shape() -> None:
    # checkpoint_every = 0 is exactly what the warmup phase passes: it re-runs
    # whole rather than resuming, so it must never write a cursor.
    positions: list[OffPolicyCursorPosition] = []

    run_offpolicy(
        _corpus(4),
        _schedule(epochs=3, checkpoint_every=0),
        trainer=_RecordingTrainer(),
        on_step=lambda _: None,
        on_checkpoint=positions.append,
    )

    assert positions == []


def test_a_resumed_walk_skips_the_completed_prefix_and_trains_the_rest() -> None:
    datums = _corpus(4)
    schedule = _schedule(epochs=2, minibatch_datums=2)
    trainer = _RecordingTrainer()
    results: list[OffPolicyStepResult] = []

    outcome = run_offpolicy(datums, schedule, trainer=trainer, on_step=results.append, start_step=3)

    assert outcome.planned_steps == 4
    assert outcome.steps_run == 1
    # Step indices continue the interrupted session's numbering, so the metrics
    # rows a resume appends do not collide with the ones already recorded.
    assert [result.step for result in results] == [3]
    assert [names for names, _ in trainer.batches] == [("trial-2", "trial-3")]


def test_resuming_at_the_end_of_the_schedule_trains_nothing() -> None:
    trainer = _RecordingTrainer()

    outcome = run_offpolicy(
        _corpus(4),
        _schedule(epochs=2, minibatch_datums=2),
        trainer=trainer,
        on_step=lambda _: None,
        start_step=4,
    )

    assert outcome.steps_run == 0
    assert trainer.batches == []


def test_a_cursor_past_the_plan_is_rejected_rather_than_silently_clamped() -> None:
    with pytest.raises(ValueError, match="does not belong to this corpus"):
        run_offpolicy(
            _corpus(4),
            _schedule(epochs=1, minibatch_datums=2),
            trainer=_RecordingTrainer(),
            on_step=lambda _: None,
            start_step=9,
        )


def test_a_step_plan_must_carry_at_least_one_datum() -> None:
    # An empty minibatch would be a forward/backward over nothing, which the
    # service rejects far from here; the plan refuses to describe one.
    with pytest.raises(ValidationError, match="at least one datum index"):
        OffPolicyStepPlan(step=0, epoch=0, minibatch=0, datum_indices=[])
