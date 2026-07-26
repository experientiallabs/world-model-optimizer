"""Cost projection and budget metering for one distillation run.

`estimate_run_cost` turns the run config plus the task split sizes into
per-meter token projections priced from the `[pricing]` section, for the CLI's
cost-confirm prompt. `BudgetMeter` then accumulates the ACTUAL token counts as
the run spends, and `check()` enforces the optional `[budget] max_usd` hard
cap by raising `BudgetExhausted` (the loop saves state and prints the resume
command on that error).

Metering follows Tinker's PER-REQUEST billing, not unique context tokens:
every sampling request bills its whole prompt, so each agent turn re-bills the
episode's full context, with the verbatim repeated prefix billed at the
discounted cached rate (`episode_billing` documents the exact split). Rollout
episodes therefore charge three meters each (full prefill, cached prefill,
sample), and teacher-in-harness episodes bill their sampled tokens at the
teacher's SAMPLING rate. Ignoring the per-request term once under-reported a
console-reconciled run by ~6x (306M billed tokens vs ~50M unique).

The projection is a deliberately simple, documented heuristic: episode counts
come exactly from the config (steps x tasks x group size, plus warmup teacher
episodes, interim evals, and the gate/baseline episodes), and per-episode
tokens come from a turns x tokens-per-turn model capped by the rollout context
budget. Meters mirror the `[pricing]` fields (cached rates fall back to the
documented 20% derivation); a meter without a price surfaces as a None-usd
line so the CLI can warn instead of silently under-reporting.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from wmh.distill.config import DistillConfig, PricingConfig
from wmh.distill.tokens import TrialRecord
from wmh.providers.tinker import TokenSpan

logger = logging.getLogger(__name__)

MeterName = Literal[
    "student_prefill",
    "student_cached_prefill",
    "student_sample",
    "student_train",
    "teacher_prefill",
    "teacher_cached_prefill",
    "teacher_sample",
]

METER_NAMES: tuple[MeterName, ...] = (
    "student_prefill",
    "student_cached_prefill",
    "student_sample",
    "student_train",
    "teacher_prefill",
    "teacher_cached_prefill",
    "teacher_sample",
)

_TOKENS_PER_USD_UNIT = 1_000_000
"""Prices in `[pricing]` are USD per million tokens."""

_AVG_TURN_FRACTION = 0.5
"""Episodes are assumed to use half the configured turn cap on average."""

_SAMPLED_TOKENS_PER_TURN = 512
"""Assumed sampled (assistant/tool-call) tokens per agent turn."""

_OBSERVATION_TOKENS_PER_TURN = 1024
"""Assumed prompt growth per turn (tool results and scaffolding)."""

_BASE_PROMPT_TOKENS = 2048
"""Assumed initial prompt (system prompt, task instruction, tool schemas)."""


class BudgetExhausted(RuntimeError):
    """Raised by `BudgetMeter.check` when actual spend exceeds the hard cap."""

    def __init__(self, spent_usd: float, max_usd: float) -> None:
        self.spent_usd = spent_usd
        self.max_usd = max_usd
        super().__init__(
            f"budget exhausted: ${spent_usd:.2f} spent against the ${max_usd:.2f} "
            "cap (budget.max_usd); the run saves its training state on this error, "
            "so raise budget.max_usd in the run config and resume the run to continue"
        )


class CostLine(BaseModel):
    """One meter's token projection (or actuals) with its optional price."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    meter: MeterName
    tokens: int = Field(ge=0)
    price_per_mtok: float | None
    """USD per million tokens from `[pricing]`; None means unpriced."""

    usd: float | None
    """tokens x price; None when the meter is unpriced (CLI warns on these)."""


class CostEstimate(BaseModel):
    """Per-meter projections for one run, plus the episode counts behind them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lines: list[CostLine]
    train_episodes: int = Field(ge=0)
    eval_episodes: int = Field(ge=0)
    baseline_episodes: int = Field(ge=0)
    """Gate/baseline episodes: student before + student after + teacher-in-harness."""

    warmup_episodes: int = Field(ge=0)
    """Warmup teacher episodes: train tasks x warmup.rollouts_per_task (0 when off)."""

    offpolicy_episodes: int = Field(ge=0)
    """Off-policy teacher episodes: train tasks x offpolicy.rollouts_per_task
    (0 when `offpolicy.epochs` is 0 or the corpus is loaded from another run)."""

    @property
    def priced_usd(self) -> float:
        """Total USD over the priced lines only."""
        return sum(line.usd for line in self.lines if line.usd is not None)

    @property
    def unpriced_meters(self) -> list[MeterName]:
        """Meters with no `[pricing]` entry, for the CLI's warning."""
        return [line.meter for line in self.lines if line.usd is None]

    def is_fully_priced(self) -> bool:
        """Whether every meter carries a price, so `priced_usd` is the whole run."""
        return not self.unpriced_meters


def _meter_price(pricing: PricingConfig, meter: MeterName) -> float | None:
    if meter == "student_prefill":
        return pricing.student_prefill
    if meter == "student_cached_prefill":
        return pricing.effective_student_cached_prefill
    if meter == "student_sample":
        return pricing.student_sample
    if meter == "student_train":
        return pricing.student_train
    if meter == "teacher_prefill":
        return pricing.teacher_prefill
    if meter == "teacher_cached_prefill":
        return pricing.effective_teacher_cached_prefill
    return pricing.teacher_sample


def _line(pricing: PricingConfig, meter: MeterName, tokens: int) -> CostLine:
    price = _meter_price(pricing, meter)
    usd = tokens / _TOKENS_PER_USD_UNIT * price if price is not None else None
    return CostLine(meter=meter, tokens=tokens, price_per_mtok=price, usd=usd)


class SpanBilling(BaseModel):
    """Per-request billing volumes measured from recorded rollout spans.

    The three volumes map onto three meters: `unique_tokens` at the full
    prefill rate, `cached_tokens` at the cached-prefill rate, and
    `sampled_tokens` at the sampling rate.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    unique_tokens: int = Field(ge=0)
    """Distinct episode tokens, billed once at the full prefill rate."""

    cached_tokens: int = Field(ge=0)
    """Repeated per-request prompt volume, billed at the cached-prefill rate."""

    sampled_tokens: int = Field(ge=0)
    """Sampled completion tokens, billed at the sampling rate."""

    def __add__(self, other: SpanBilling) -> SpanBilling:
        return SpanBilling(
            unique_tokens=self.unique_tokens + other.unique_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            sampled_tokens=self.sampled_tokens + other.sampled_tokens,
        )


def episode_billing(spans: Sequence[TokenSpan]) -> SpanBilling:
    """Per-request billing volumes for ONE episode's recorded spans.

    Tinker bills prefill PER REQUEST over each call's full prompt: every
    agent turn re-bills the episode's whole context, with the verbatim
    repeated prefix billed at the cached rate. The model here:

    - per-request volume = sum over spans of `len(prompt_token_ids)`.
    - unique = tokens the episode put through the model for the first time.
      For a prefix-clean episode (every prompt extends the previous prompt
      plus its sampled tokens verbatim, the same test `build_datums` merges
      on) this is exactly the final span's prompt plus sampled length. A
      prefix break restarts the accumulation, so a fragmented episode's
      unique volume is the sum over fragments: re-prefilled context counts
      as unique again, matching what the service re-bills at the full rate.
    - cached = the per-request volume beyond unique, clamped at zero (a
      single-call episode repeats nothing). Under the prefix property every
      repeat is verbatim, so the cached rate applies to all of it.

    Args:
        spans: One episode's recorded spans, in any order (sorted by
            call_index here).

    Returns:
        The episode's billing volumes.
    """
    unique = 0
    per_request = 0
    sampled = 0
    accumulated: list[int] = []
    for span in sorted(spans, key=lambda item: item.call_index):
        prompt = span.prompt_token_ids
        per_request += len(prompt)
        if accumulated and prompt[: len(accumulated)] == accumulated:
            unique += len(prompt) - len(accumulated)
        else:
            unique += len(prompt)
        unique += len(span.sampled_token_ids)
        sampled += len(span.sampled_token_ids)
        accumulated = list(prompt) + list(span.sampled_token_ids)
    return SpanBilling(
        unique_tokens=unique,
        cached_tokens=max(per_request - unique, 0),
        sampled_tokens=sampled,
    )


def batch_billing(records: Sequence[TrialRecord]) -> SpanBilling:
    """Summed `episode_billing` over one rollout batch's trial records.

    Summed per episode (not over a flattened span list) so each episode's
    cached volume clamps independently and one trial's prefix break never
    bleeds into another's accounting.

    Args:
        records: The batch's trial records (span-less trials contribute 0).

    Returns:
        The batch's total billing volumes.
    """
    total = SpanBilling(unique_tokens=0, cached_tokens=0, sampled_tokens=0)
    for record in records:
        total = total + episode_billing(record.spans)
    return total


def estimate_run_cost(cfg: DistillConfig, n_train_tasks: int, n_holdout_tasks: int) -> CostEstimate:
    """Project the run's per-meter token volumes and price them.

    Episode counts (exact, from the config):

    - train: `steps x min(tasks_per_batch, n_train_tasks) x group_size`
    - off-policy: `n_train_tasks x offpolicy.rollouts_per_task` teacher
      episodes when `offpolicy.epochs > 0` and the corpus is collected here
      (`offpolicy.trajectories_from` unset), else 0
    - warmup: `n_train_tasks x warmup.rollouts_per_task` teacher episodes when
      `warmup.steps > 0`, else 0
    - interim evals: `steps // eval.every` evals (0 when eval.every is 0) of
      `min(eval.tasks, n_train_tasks) x eval.k` student episodes each
    - gate/baseline: student-before and student-after at
      `n_holdout x gate.k` each, plus one teacher-in-harness baseline at
      `n_holdout x gate.k`

    Per-episode tokens (heuristic; module constants document the assumptions):

    - `avg_turns = max(1, ceil(rollout.max_turns x 0.5))`
    - `sampled = avg_turns x min(sampling.max_tokens, 512)`
    - `episode_tokens = min(rollout.context_budget_tokens,
      2048 + avg_turns x (1024 + sampled_per_turn))`: the episode's final
      unique sequence length under the prefix property (the estimate assumes
      prefix-clean episodes, so this is also the unique billing volume)
    - per-request prefill: turn k's prompt is
      `min(2048 + k x 1024 + (k - 1) x sampled_per_turn,
      context_budget_tokens)` and every turn re-bills it whole, so the
      per-request volume is the sum over turns; the part beyond
      `episode_tokens` is the verbatim repeat billed at the cached rate
      (see `episode_billing` for the same split on actual spans)

    Meter mapping: every student episode charges `episode_tokens` (its unique
    volume) to student_prefill, the repeated per-request volume to
    student_cached_prefill, and `sampled` to student_sample; every train
    episode additionally charges `episode_tokens` to student_train
    (forward_backward over the full datum; x `train.topk` under the
    `topk_ce` loss, whose k rank replicas each carry the full sequence) and
    `episode_tokens` to teacher_prefill (the teacher scores each episode's
    full sequence once, one full-price request with no repeats to cache;
    the topk_ce prefill-only request bills the same volume). Teacher-in-harness
    episodes (the gate baseline and a cross_entropy phase's collection) charge
    `sampled` to teacher_sample (they bill the teacher's SAMPLING rate on what
    they generate) plus per-request prefill exactly like a student episode, onto
    teacher_prefill and teacher_cached_prefill. The cross_entropy phases'
    training tokens are NOT projected: they depend on how many teacher trials
    pass (unknowable up front) and are bounded by the phase's pass count over
    at most those episodes' tokens.

    Args:
        cfg: The validated run config.
        n_train_tasks: Size of the train task split (must be >= 1).
        n_holdout_tasks: Size of the holdout task split (>= 0; 0 skips the
            gate/baseline episodes entirely).

    Returns:
        The estimate, one line per meter in `METER_NAMES` order.

    Raises:
        ValueError: If the split sizes are out of range.
    """
    if n_train_tasks < 1:
        raise ValueError(
            f"n_train_tasks must be >= 1, got {n_train_tasks}; a distillation run "
            "needs a non-empty train task split"
        )
    if n_holdout_tasks < 0:
        raise ValueError(f"n_holdout_tasks must be >= 0, got {n_holdout_tasks}")

    avg_turns = max(1, math.ceil(cfg.rollout.max_turns * _AVG_TURN_FRACTION))
    sampled_per_turn = min(cfg.sampling.max_tokens, _SAMPLED_TOKENS_PER_TURN)
    context_budget = cfg.rollout.context_budget_tokens
    episode_tokens = min(
        context_budget,
        _BASE_PROMPT_TOKENS + avg_turns * (_OBSERVATION_TOKENS_PER_TURN + sampled_per_turn),
    )
    sampled_tokens = min(avg_turns * sampled_per_turn, episode_tokens)
    # Per-request accounting: every turn re-bills its whole prompt, so the
    # per-request volume sums the per-turn prompts; the episode's distinct
    # tokens bill once at the full rate and the rest is the cached repeat.
    per_request_tokens = sum(
        min(
            _BASE_PROMPT_TOKENS
            + turn * _OBSERVATION_TOKENS_PER_TURN
            + (turn - 1) * sampled_per_turn,
            context_budget,
        )
        for turn in range(1, avg_turns + 1)
    )
    cached_tokens = max(per_request_tokens - episode_tokens, 0)

    tasks_per_step = min(cfg.train.tasks_per_batch, n_train_tasks)
    train_episodes = cfg.train.steps * tasks_per_step * cfg.train.group_size
    warmup_episodes = n_train_tasks * cfg.warmup.rollouts_per_task if cfg.warmup.steps > 0 else 0
    collects_offpolicy = cfg.offpolicy.epochs > 0 and cfg.offpolicy.trajectories_from is None
    offpolicy_episodes = (
        n_train_tasks * cfg.offpolicy.rollouts_per_task if collects_offpolicy else 0
    )
    interim_evals = cfg.train.steps // cfg.eval.every if cfg.eval.every > 0 else 0
    eval_episodes = interim_evals * min(cfg.eval.tasks, n_train_tasks) * cfg.eval.k
    gate_attempts = n_holdout_tasks * cfg.gate.k
    student_baseline_episodes = 2 * gate_attempts  # student-before + student-after
    teacher_baseline_episodes = gate_attempts

    student_episodes = train_episodes + eval_episodes + student_baseline_episodes
    teacher_harness_episodes = teacher_baseline_episodes + warmup_episodes + offpolicy_episodes
    # The topk_ce loss trains k rank-aligned cross_entropy replicas per datum,
    # each carrying the full sequence, so its train volume is k x the default
    # loss's (the loop meters actuals the same way; see build_topk_ce_datums).
    train_replication = cfg.train.topk if cfg.train.loss == "topk_ce" else 1
    projections: dict[MeterName, int] = {
        "student_prefill": student_episodes * episode_tokens,
        "student_cached_prefill": student_episodes * cached_tokens,
        "student_sample": student_episodes * sampled_tokens,
        "student_train": train_episodes * episode_tokens * train_replication,
        "teacher_prefill": (
            train_episodes * episode_tokens + teacher_harness_episodes * episode_tokens
        ),
        "teacher_cached_prefill": teacher_harness_episodes * cached_tokens,
        "teacher_sample": teacher_harness_episodes * sampled_tokens,
    }
    estimate = CostEstimate(
        lines=[_line(cfg.pricing, meter, projections[meter]) for meter in METER_NAMES],
        train_episodes=train_episodes,
        eval_episodes=eval_episodes,
        baseline_episodes=student_baseline_episodes + teacher_baseline_episodes,
        warmup_episodes=warmup_episodes,
        offpolicy_episodes=offpolicy_episodes,
    )
    logger.debug(
        "cost estimate: %d train + %d off-policy + %d warmup + %d eval + %d baseline "
        "episode(s), priced $%.2f%s",
        estimate.train_episodes,
        estimate.offpolicy_episodes,
        estimate.warmup_episodes,
        estimate.eval_episodes,
        estimate.baseline_episodes,
        estimate.priced_usd,
        f" (unpriced meters: {', '.join(estimate.unpriced_meters)})"
        if estimate.unpriced_meters
        else "",
    )
    return estimate


class BudgetMeter:
    """Accumulates actual metered tokens and enforces the hard USD cap.

    Args:
        pricing: The `[pricing]` section; unpriced meters accumulate tokens
            but contribute no USD (mirroring the estimate's None lines).
        max_usd: The `[budget] max_usd` hard cap; None disables enforcement.
    """

    def __init__(self, pricing: PricingConfig, max_usd: float | None = None) -> None:
        self._pricing = pricing
        self._max_usd = max_usd
        self._tokens: dict[MeterName, int] = {meter: 0 for meter in METER_NAMES}
        self._spent_usd = 0.0

    def charge(self, meter: MeterName, tokens: int) -> None:
        """Record actual token usage against one meter.

        Args:
            meter: Which meter the tokens belong to.
            tokens: The token count to add (>= 0).

        Raises:
            ValueError: If `tokens` is negative.
        """
        if tokens < 0:
            raise ValueError(f"cannot charge a negative token count ({tokens}) to {meter}")
        self._tokens[meter] += tokens
        price = _meter_price(self._pricing, meter)
        if price is not None:
            self._spent_usd += tokens / _TOKENS_PER_USD_UNIT * price

    def check(self) -> None:
        """Enforce the hard cap against the priced spend so far.

        Raises:
            BudgetExhausted: When the cap is set and the spend exceeds it.
        """
        if self._max_usd is not None and self._spent_usd > self._max_usd:
            raise BudgetExhausted(self._spent_usd, self._max_usd)

    def tokens(self, meter: MeterName) -> int:
        """Actual tokens charged to one meter so far."""
        return self._tokens[meter]

    @property
    def spent_usd(self) -> float:
        """Priced USD spend so far (unpriced meters contribute nothing)."""
        return self._spent_usd

    def lines(self) -> list[CostLine]:
        """The actuals in the same line shape the estimate uses, for reporting."""
        return [_line(self._pricing, meter, self._tokens[meter]) for meter in METER_NAMES]
