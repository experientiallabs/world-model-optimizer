"""Tests for cost projection and budget metering, with hand-computed numbers."""

import pytest

from wmh.distill.config import (
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
    WarmupConfig,
)
from wmh.distill.cost import (
    METER_NAMES,
    BudgetExhausted,
    BudgetMeter,
    CostEstimate,
    SpanBilling,
    batch_billing,
    episode_billing,
    estimate_run_cost,
)
from wmh.distill.tokens import TrialRecord
from wmh.providers.tinker import TokenSpan

FULL_PRICING = PricingConfig(
    student_prefill=1.0,
    student_sample=2.0,
    student_train=4.0,
    teacher_prefill=10.0,
    teacher_sample=25.0,
)
# Cached rates derive from the 20% default: student 0.2, teacher 2.0 USD/Mtok.


def _config(pricing: PricingConfig | None = None) -> DistillConfig:
    """A small config whose projections are easy to hand-compute.

    Heuristics resolve to: avg_turns = ceil(4 * 0.5) = 2, sampled per turn =
    min(128, 512) = 128, episode tokens = min(65536, 2048 + 2 * (1024 + 128))
    = 4352 (the unique billing volume), sampled per episode = 256. Per-request
    prefill: turn prompts are 2048 + 1024 = 3072 and 2048 + 2048 + 128 = 4224,
    so the per-request volume is 7296 and the cached repeat is 7296 - 4352
    = 2944.
    """
    return DistillConfig(
        student=StudentConfig(base_model="Qwen/Qwen3-4B"),
        teacher=TeacherConfig(model="Qwen/Qwen3-235B-A22B-Instruct-2507"),
        harbor=HarborConfig(job_template="jobs/tb2.yaml"),
        rollout=RolloutConfig(max_turns=4),
        train=TrainConfig(steps=2, tasks_per_batch=3, group_size=2),
        sampling=SamplingConfig(max_tokens=128),
        eval=EvalConfig(every=1, tasks=2, k=1),
        gate=GateConfig(k=2),
        pricing=pricing if pricing is not None else PricingConfig(),
    )


def _tokens(estimate: CostEstimate) -> dict[str, int]:
    return {line.meter: line.tokens for line in estimate.lines}


def _span(call_index: int, prompt: list[int], sampled: list[int]) -> TokenSpan:
    return TokenSpan(
        call_index=call_index,
        prompt_token_ids=prompt,
        sampled_token_ids=sampled,
        sampled_logprobs=[-0.1] * len(sampled),
    )


def _record(trial_name: str, spans: list[TokenSpan]) -> TrialRecord:
    return TrialRecord(
        task_id="task",
        attempt=1,
        trial_name=trial_name,
        reward=1.0,
        passed=True,
        spans=spans,
        artifact_dir=f"/tmp/{trial_name}",
    )


# --- episode_billing / batch_billing ------------------------------------------


def test_episode_billing_prefix_clean_two_turns() -> None:
    # Turn 2's prompt extends turn 1's prompt + sampled verbatim.
    p1, s1 = [1, 2, 3, 4], [5, 6]
    p2, s2 = [1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11]
    billing = episode_billing([_span(0, p1, s1), _span(1, p2, s2)])
    # per-request = 4 + 8 = 12; unique = final prompt + sampled = 8 + 3 = 11;
    # cached repeat = 12 - 11 = 1.
    assert billing == SpanBilling(unique_tokens=11, cached_tokens=1, sampled_tokens=5)


def test_episode_billing_single_call_repeats_nothing() -> None:
    billing = episode_billing([_span(0, [1, 2, 3], [4, 5])])
    # per-request (3) is below unique (5): cached clamps to zero.
    assert billing == SpanBilling(unique_tokens=5, cached_tokens=0, sampled_tokens=2)


def test_episode_billing_fragmented_falls_back_to_summing() -> None:
    # Turn 2 edited its history (not a prefix extension), the same break
    # build_datums fragments on: its whole prompt re-prefills at full price.
    p1, s1 = [1, 2, 3, 4], [5, 6]
    p2, s2 = [1, 2, 99], [7, 8]
    billing = episode_billing([_span(0, p1, s1), _span(1, p2, s2)])
    # unique = (4 + 2) + (3 + 2) = 11; per-request = 4 + 3 = 7 < unique, so
    # nothing is cached.
    assert billing == SpanBilling(unique_tokens=11, cached_tokens=0, sampled_tokens=4)


def test_episode_billing_sorts_spans_by_call_index() -> None:
    p1, s1 = [1, 2, 3, 4], [5, 6]
    p2, s2 = [1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11]
    ordered = episode_billing([_span(0, p1, s1), _span(1, p2, s2)])
    shuffled = episode_billing([_span(1, p2, s2), _span(0, p1, s1)])
    assert shuffled == ordered


def test_episode_billing_empty_spans_bill_nothing() -> None:
    assert episode_billing([]) == SpanBilling(unique_tokens=0, cached_tokens=0, sampled_tokens=0)


def test_batch_billing_sums_and_clamps_per_episode() -> None:
    # Episode A has a positive cached repeat; episode B's single call clamps
    # at zero. Summing per episode keeps A's repeat intact instead of letting
    # B's negative pre-clamp margin eat into it.
    a = _record(
        "a",
        [
            _span(0, [1, 2, 3, 4], [5, 6]),
            _span(1, [1, 2, 3, 4, 5, 6, 7, 8], [9]),
        ],
    )
    b = _record("b", [_span(0, [1, 2], [3, 4, 5])])
    billing = batch_billing([a, b])
    # A: per-request 12, unique 9, cached 3, sampled 3. B: unique 5, cached 0.
    assert billing == SpanBilling(unique_tokens=14, cached_tokens=3, sampled_tokens=6)


def test_batch_billing_spanless_trials_contribute_nothing() -> None:
    assert batch_billing([_record("dead", [])]) == SpanBilling(
        unique_tokens=0, cached_tokens=0, sampled_tokens=0
    )


# --- estimate_run_cost -------------------------------------------------------


def test_estimate_hand_computed_tokens() -> None:
    # Episodes: train = 2 steps x min(3, 5) tasks x 2 group = 12;
    # evals = (2 // 1) x min(2, 5) x 1 = 4; gate attempts = 3 holdout x k=2 = 6;
    # student baselines (before + after) = 12; teacher baseline = 6.
    # Student episodes = 12 + 4 + 12 = 28. Warmup off by default: 0 episodes.
    estimate = estimate_run_cost(_config(), n_train_tasks=5, n_holdout_tasks=3)
    assert estimate.train_episodes == 12
    assert estimate.eval_episodes == 4
    assert estimate.baseline_episodes == 18
    assert estimate.warmup_episodes == 0
    assert _tokens(estimate) == {
        "student_prefill": 28 * 4352,
        "student_cached_prefill": 28 * 2944,
        "student_sample": 28 * 256,
        "student_train": 12 * 4352,
        # Teacher scoring (one full-price request per train episode) plus the
        # teacher baseline's unique volume.
        "teacher_prefill": (12 + 6) * 4352,
        "teacher_cached_prefill": 6 * 2944,
        "teacher_sample": 6 * 256,
    }


def test_estimate_includes_warmup_teacher_episodes() -> None:
    # Warmup on: teacher episodes = 5 train tasks x 3 rollouts_per_task = 15,
    # billed like the teacher baseline (per-request prefill plus teacher_sample
    # on what they generate). Student meters are untouched: warmup samples the
    # TEACHER, and its SFT train tokens are not projected (they depend on the
    # unknown pass rate).
    cfg = _config().model_copy(update={"warmup": WarmupConfig(steps=2, rollouts_per_task=3)})
    estimate = estimate_run_cost(cfg, n_train_tasks=5, n_holdout_tasks=3)
    baseline = estimate_run_cost(_config(), n_train_tasks=5, n_holdout_tasks=3)
    assert estimate.warmup_episodes == 15
    tokens = _tokens(estimate)
    base_tokens = _tokens(baseline)
    assert tokens["teacher_prefill"] == base_tokens["teacher_prefill"] + 15 * 4352
    assert tokens["teacher_cached_prefill"] == base_tokens["teacher_cached_prefill"] + 15 * 2944
    assert tokens["teacher_sample"] == base_tokens["teacher_sample"] + 15 * 256
    for meter in ("student_prefill", "student_cached_prefill", "student_sample", "student_train"):
        assert tokens[meter] == base_tokens[meter]


def test_estimate_includes_offpolicy_teacher_episodes() -> None:
    # The off-policy corpus is collected the same way the warmup one is, so it
    # bills as teacher-in-harness episodes; the CE training tokens stay
    # unprojected (they depend on the unknown pass rate).
    cfg = _config().model_copy(update={"offpolicy": OffPolicyConfig(epochs=2, rollouts_per_task=3)})
    estimate = estimate_run_cost(cfg, n_train_tasks=5, n_holdout_tasks=3)
    baseline = estimate_run_cost(_config(), n_train_tasks=5, n_holdout_tasks=3)
    assert estimate.offpolicy_episodes == 15
    assert estimate.warmup_episodes == 0
    tokens = _tokens(estimate)
    base_tokens = _tokens(baseline)
    assert tokens["teacher_prefill"] == base_tokens["teacher_prefill"] + 15 * 4352
    assert tokens["teacher_cached_prefill"] == base_tokens["teacher_cached_prefill"] + 15 * 2944
    assert tokens["teacher_sample"] == base_tokens["teacher_sample"] + 15 * 256
    for meter in ("student_prefill", "student_cached_prefill", "student_sample", "student_train"):
        assert tokens[meter] == base_tokens[meter]


def test_estimate_projects_no_collection_when_the_corpus_is_loaded() -> None:
    # trajectories_from means another run already paid for the teacher rollouts.
    cfg = _config().model_copy(
        update={
            "offpolicy": OffPolicyConfig(
                epochs=2, rollouts_per_task=3, trajectories_from="runs/prior"
            )
        }
    )
    estimate = estimate_run_cost(cfg, n_train_tasks=5, n_holdout_tasks=3)
    assert estimate.offpolicy_episodes == 0
    assert _tokens(estimate) == _tokens(
        estimate_run_cost(_config(), n_train_tasks=5, n_holdout_tasks=3)
    )


def test_estimate_offpolicy_epochs_zero_means_no_offpolicy_episodes() -> None:
    cfg = _config().model_copy(update={"offpolicy": OffPolicyConfig(epochs=0, rollouts_per_task=3)})
    estimate = estimate_run_cost(cfg, n_train_tasks=5, n_holdout_tasks=3)
    assert estimate.offpolicy_episodes == 0
    assert _tokens(estimate) == _tokens(
        estimate_run_cost(_config(), n_train_tasks=5, n_holdout_tasks=3)
    )


def test_estimate_warmup_steps_zero_means_no_warmup_episodes() -> None:
    # rollouts_per_task alone must not add episodes: steps = 0 disables warmup.
    cfg = _config().model_copy(update={"warmup": WarmupConfig(steps=0, rollouts_per_task=3)})
    estimate = estimate_run_cost(cfg, n_train_tasks=5, n_holdout_tasks=3)
    assert estimate.warmup_episodes == 0
    assert _tokens(estimate) == _tokens(
        estimate_run_cost(_config(), n_train_tasks=5, n_holdout_tasks=3)
    )


def test_estimate_hand_computed_usd() -> None:
    estimate = estimate_run_cost(_config(FULL_PRICING), n_train_tasks=5, n_holdout_tasks=3)
    usd = {line.meter: line.usd for line in estimate.lines}
    assert usd["student_prefill"] == pytest.approx(28 * 4352 / 1e6 * 1.0)
    assert usd["student_cached_prefill"] == pytest.approx(28 * 2944 / 1e6 * 0.2)
    assert usd["student_sample"] == pytest.approx(28 * 256 / 1e6 * 2.0)
    assert usd["student_train"] == pytest.approx(12 * 4352 / 1e6 * 4.0)
    assert usd["teacher_prefill"] == pytest.approx(18 * 4352 / 1e6 * 10.0)
    assert usd["teacher_cached_prefill"] == pytest.approx(6 * 2944 / 1e6 * 2.0)
    assert usd["teacher_sample"] == pytest.approx(6 * 256 / 1e6 * 25.0)
    assert estimate.priced_usd == pytest.approx(1.2186624)
    assert estimate.is_fully_priced()
    assert estimate.unpriced_meters == []


def test_estimate_unpriced_meters_surface_as_none_lines() -> None:
    partial = PricingConfig(student_sample=2.0)
    estimate = estimate_run_cost(_config(partial), n_train_tasks=5, n_holdout_tasks=3)
    by_meter = {line.meter: line for line in estimate.lines}
    assert by_meter["student_sample"].usd == pytest.approx(7168 / 1e6 * 2.0)
    for meter in (
        "student_prefill",
        "student_cached_prefill",  # no student_prefill to derive 20% from
        "student_train",
        "teacher_prefill",
        "teacher_cached_prefill",
        "teacher_sample",
    ):
        assert by_meter[meter].usd is None
        assert by_meter[meter].price_per_mtok is None
    assert not estimate.is_fully_priced()
    assert set(estimate.unpriced_meters) == {
        "student_prefill",
        "student_cached_prefill",
        "student_train",
        "teacher_prefill",
        "teacher_cached_prefill",
        "teacher_sample",
    }
    # priced_usd still totals the priced lines so the CLI can show a floor.
    assert estimate.priced_usd == pytest.approx(0.014336)


def test_estimate_cached_meters_priced_through_the_default_derivation() -> None:
    # Only the full prefill prices are set: the cached meters price at 20%.
    pricing = PricingConfig(student_prefill=1.0, teacher_prefill=10.0)
    estimate = estimate_run_cost(_config(pricing), n_train_tasks=5, n_holdout_tasks=3)
    by_meter = {line.meter: line for line in estimate.lines}
    assert by_meter["student_cached_prefill"].price_per_mtok == pytest.approx(0.2)
    assert by_meter["teacher_cached_prefill"].price_per_mtok == pytest.approx(2.0)
    assert set(estimate.unpriced_meters) == {"student_sample", "student_train", "teacher_sample"}


def test_estimate_lines_cover_every_meter_in_order() -> None:
    estimate = estimate_run_cost(_config(), n_train_tasks=1, n_holdout_tasks=0)
    assert tuple(line.meter for line in estimate.lines) == METER_NAMES


def test_estimate_clamps_batch_to_train_split() -> None:
    # tasks_per_batch = 3 but only 1 train task: train episodes = 2 x 1 x 2 = 4;
    # evals = 2 x min(2, 1) x 1 = 2; no holdout -> student episodes = 6 and no
    # teacher-in-harness episodes at all.
    estimate = estimate_run_cost(_config(), n_train_tasks=1, n_holdout_tasks=0)
    assert estimate.train_episodes == 4
    assert estimate.eval_episodes == 2
    assert estimate.baseline_episodes == 0
    assert _tokens(estimate) == {
        "student_prefill": 6 * 4352,
        "student_cached_prefill": 6 * 2944,
        "student_sample": 6 * 256,
        "student_train": 4 * 4352,
        "teacher_prefill": 4 * 4352,
        "teacher_cached_prefill": 0,
        "teacher_sample": 0,
    }


def test_estimate_eval_every_zero_means_no_interim_evals() -> None:
    cfg = _config().model_copy(deep=True)
    cfg.eval.every = 0
    estimate = estimate_run_cost(cfg, n_train_tasks=5, n_holdout_tasks=3)
    assert estimate.eval_episodes == 0


def test_estimate_context_budget_caps_episode_and_per_turn_prompts() -> None:
    cfg = _config().model_copy(deep=True)
    cfg.rollout.context_budget_tokens = 2048
    # episode tokens = min(2048, 4352) = 2048; sampled = min(256, 2048) = 256.
    # Both per-turn prompts (3072 and 4224) cap at 2048, so the per-request
    # volume is 4096 and the cached repeat is 4096 - 2048 = 2048.
    estimate = estimate_run_cost(cfg, n_train_tasks=5, n_holdout_tasks=3)
    assert _tokens(estimate) == {
        "student_prefill": 28 * 2048,
        "student_cached_prefill": 28 * 2048,
        "student_sample": 28 * 256,
        "student_train": 12 * 2048,
        "teacher_prefill": 18 * 2048,
        "teacher_cached_prefill": 6 * 2048,
        "teacher_sample": 6 * 256,
    }


def test_estimate_rejects_bad_split_sizes() -> None:
    with pytest.raises(ValueError, match="n_train_tasks must be >= 1"):
        estimate_run_cost(_config(), n_train_tasks=0, n_holdout_tasks=3)
    with pytest.raises(ValueError, match="n_holdout_tasks must be >= 0"):
        estimate_run_cost(_config(), n_train_tasks=5, n_holdout_tasks=-1)


# --- BudgetMeter --------------------------------------------------------------


def test_meter_accumulates_tokens_and_usd() -> None:
    meter = BudgetMeter(FULL_PRICING)
    meter.charge("teacher_prefill", 500_000)
    meter.charge("teacher_prefill", 250_000)
    meter.charge("student_sample", 1_000_000)
    assert meter.tokens("teacher_prefill") == 750_000
    assert meter.tokens("student_sample") == 1_000_000
    assert meter.tokens("student_prefill") == 0
    # 0.75M x $10 + 1M x $2 = $9.50, hand-computed.
    assert meter.spent_usd == pytest.approx(9.5)


def test_meter_cached_rates_default_to_20_percent_of_prefill() -> None:
    meter = BudgetMeter(PricingConfig(student_prefill=1.0, teacher_prefill=10.0))
    meter.charge("student_cached_prefill", 1_000_000)
    meter.charge("teacher_cached_prefill", 1_000_000)
    # 1M x $0.20 + 1M x $2.00, both derived from the 20% default.
    assert meter.spent_usd == pytest.approx(2.2)


def test_meter_explicit_cached_rate_overrides_the_default() -> None:
    pricing = PricingConfig(student_prefill=1.0, student_cached_prefill=0.5)
    meter = BudgetMeter(pricing)
    meter.charge("student_cached_prefill", 1_000_000)
    assert meter.spent_usd == pytest.approx(0.5)


def test_meter_teacher_sample_charges_at_the_sampling_rate() -> None:
    meter = BudgetMeter(FULL_PRICING)
    meter.charge("teacher_sample", 1_000_000)
    assert meter.spent_usd == pytest.approx(25.0)


def test_meter_unpriced_meter_counts_tokens_but_no_usd() -> None:
    meter = BudgetMeter(PricingConfig(student_sample=2.0))
    meter.charge("teacher_prefill", 1_000_000)
    assert meter.tokens("teacher_prefill") == 1_000_000
    assert meter.spent_usd == pytest.approx(0.0)
    meter.charge("student_sample", 1_000_000)
    assert meter.spent_usd == pytest.approx(2.0)


def test_meter_check_raises_when_cap_exceeded() -> None:
    meter = BudgetMeter(FULL_PRICING, max_usd=5.0)
    meter.charge("student_sample", 2_000_000)  # $4.00
    meter.check()  # under the cap
    meter.charge("student_sample", 1_000_000)  # $6.00 total
    with pytest.raises(BudgetExhausted) as excinfo:
        meter.check()
    assert excinfo.value.spent_usd == pytest.approx(6.0)
    assert excinfo.value.max_usd == pytest.approx(5.0)
    message = str(excinfo.value)
    assert "$6.00" in message
    assert "$5.00" in message
    assert "resume" in message


def test_meter_exactly_at_cap_is_not_exhausted() -> None:
    meter = BudgetMeter(FULL_PRICING, max_usd=2.0)
    meter.charge("student_sample", 1_000_000)  # exactly $2.00
    meter.check()


def test_meter_without_cap_never_raises() -> None:
    meter = BudgetMeter(FULL_PRICING)
    meter.charge("teacher_prefill", 10_000_000_000)
    meter.check()


def test_meter_rejects_negative_charge() -> None:
    meter = BudgetMeter(FULL_PRICING)
    with pytest.raises(ValueError, match="negative token count"):
        meter.charge("student_train", -1)


def test_meter_lines_mirror_the_estimate_shape() -> None:
    meter = BudgetMeter(PricingConfig(student_train=4.0))
    meter.charge("student_train", 250_000)
    lines = {line.meter: line for line in meter.lines()}
    assert tuple(line.meter for line in meter.lines()) == METER_NAMES
    assert lines["student_train"].tokens == 250_000
    assert lines["student_train"].usd == pytest.approx(1.0)
    assert lines["teacher_prefill"].tokens == 0
    assert lines["teacher_prefill"].usd is None


def test_estimate_topk_ce_multiplies_train_tokens_by_k() -> None:
    """The topk_ce loss trains k full-sequence replicas per datum, so the
    student_train projection scales by train.topk (and nothing else moves)."""
    base = _config()
    topk = base.model_copy(
        update={"train": base.train.model_copy(update={"loss": "topk_ce", "topk": 4})}
    )

    default_tokens = _tokens(estimate_run_cost(base, n_train_tasks=5, n_holdout_tasks=3))
    topk_tokens = _tokens(estimate_run_cost(topk, n_train_tasks=5, n_holdout_tasks=3))

    assert topk_tokens["student_train"] == 4 * default_tokens["student_train"]
    for meter, tokens in default_tokens.items():
        if meter != "student_train":
            assert topk_tokens[meter] == tokens
