"""Tests for the token→cost pricing table."""

from __future__ import annotations

import pytest

from wmo.providers.base import TokenUsage
from wmo.tracking.pricing import _PRICES, cost_usd, price_for


def test_opus_4_8_cost_is_5_in_25_out_per_mtok() -> None:
    # 1M input + 1M output on Opus 4.8 = $5 + $25. (approx: float division, not exact arithmetic)
    cost = cost_usd("claude-opus-4-8", TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000))
    assert cost == pytest.approx(30.0)


def test_bedrock_prefix_normalizes_to_same_price() -> None:
    # The Bedrock-prefixed id prices identically to the direct id.
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=0)
    assert cost_usd("us.anthropic.claude-opus-4-8", usage) == cost_usd("claude-opus-4-8", usage)
    assert cost_usd("us.anthropic.claude-opus-4-8", usage) == pytest.approx(5.0)


def test_bedrock_dated_inference_profile_id_normalizes() -> None:
    # Bedrock inference-profile ids carry a snapshot date + version, e.g.
    # `us.anthropic.claude-haiku-4-5-20251001-v1:0`; they must price like the undated row.
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=0)
    assert price_for("us.anthropic.claude-haiku-4-5-20251001-v1:0") is not None
    assert cost_usd("us.anthropic.claude-haiku-4-5-20251001-v1:0", usage) == pytest.approx(1.0)
    # The `-v1` version suffix alone is also stripped.
    assert cost_usd("anthropic.claude-opus-4-6-v1", usage) == pytest.approx(5.0)


def test_gpt_5_5_output_is_30_per_mtok() -> None:
    # Verified 2026-06-25 against OpenAI's live pricing page: gpt-5.5 is $5 in / $30 out.
    price = price_for("gpt-5.5")
    assert price is not None
    assert (price.input_per_mtok, price.output_per_mtok) == (5.0, 30.0)


def test_gpt_5_5_long_context_prices_each_call_at_published_multipliers() -> None:
    usage = TokenUsage(
        input_tokens=300_000,
        output_tokens=10_000,
        cached_input_tokens=100_000,
    )
    # 200k fresh input at 2x $5/M, 100k cached at 2x $0.50/M,
    # and 10k output at 1.5x $30/M.
    assert cost_usd("gpt-5.5", usage) == pytest.approx(2.0 + 0.1 + 0.45)


def test_openai_snapshot_id_normalizes_and_keeps_long_context_pricing() -> None:
    usage = TokenUsage(input_tokens=300_000, output_tokens=10_000)
    assert cost_usd("gpt-5.5-2026-04-23", usage) == cost_usd("gpt-5.5", usage)


def test_gpt_5_5_threshold_is_strictly_above_272k() -> None:
    usage = TokenUsage(input_tokens=272_000, output_tokens=1_000)
    assert cost_usd("gpt-5.5", usage) == pytest.approx(1.36 + 0.03)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5.6-sol", (5.0, 30.0)),
        ("gpt-5.6-terra", (2.5, 15.0)),
        ("gpt-5.6-luna", (1.0, 6.0)),
        ("claude-opus-5", (5.0, 25.0)),
    ],
)
def test_current_frontier_prices(model: str, expected: tuple[float, float]) -> None:
    price = price_for(model)
    assert price is not None
    assert (price.input_per_mtok, price.output_per_mtok) == expected


def test_fable_5_is_10_in_50_out() -> None:
    price = price_for("claude-fable-5")
    assert price is not None
    assert (price.input_per_mtok, price.output_per_mtok) == (10.0, 50.0)


def test_titan_embedding_keeps_amazon_prefix() -> None:
    # `amazon.` is part of the Titan model id, not a routing prefix — it must still resolve.
    assert price_for("amazon.titan-embed-text-v2:0") is not None


def test_unknown_model_is_zero_and_flagged() -> None:
    assert price_for("totally-made-up-model") is None
    assert cost_usd("totally-made-up-model", TokenUsage(input_tokens=999, output_tokens=999)) == 0.0


def test_partial_usage_is_prorated() -> None:
    # 500k input + 200k output on Opus 4.8 = 0.5*5 + 0.2*25 = 2.5 + 5.0 = 7.5.
    cost = cost_usd("claude-opus-4-8", TokenUsage(input_tokens=500_000, output_tokens=200_000))
    assert cost == pytest.approx(7.5)


def test_no_cache_traffic_costs_are_bit_identical_to_the_two_term_formula() -> None:
    # The D-COMPRESS item-0 regression guarantee: adding cache tiers must not move any cost
    # number when there is no cache traffic. Exact equality (==) on every table row, not approx:
    # the formula must REDUCE to the pre-change arithmetic, not merely land close to it.
    usage = TokenUsage(input_tokens=123_457, output_tokens=76_543)
    for model, price in _PRICES.items():
        expected = (
            usage.input_tokens * price.input_per_mtok + usage.output_tokens * price.output_per_mtok
        ) / 1_000_000
        assert cost_usd(model, usage) == expected, model


def test_claude_cache_reads_bill_at_one_tenth_input_rate() -> None:
    # 1M input, all served from cache, on fable ($10/M input): 0.1x -> $1.00.
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=0, cached_input_tokens=1_000_000)
    assert cost_usd("claude-fable-5", usage) == pytest.approx(1.0)


def test_claude_cache_writes_bill_at_premium() -> None:
    # 1M input, all written to cache, on fable ($10/M input): 1.25x (5m TTL) -> $12.50.
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=0, cache_write_input_tokens=1_000_000)
    assert cost_usd("claude-fable-5", usage) == pytest.approx(12.5)


def test_mixed_cache_tiers_split_the_input() -> None:
    # 1M input on opus-4-8 ($5/M): 500k fresh @ $5 + 300k read @ $0.5 + 200k write @ $6.25
    # = 2.5 + 0.15 + 1.25 = $3.90.
    usage = TokenUsage(
        input_tokens=1_000_000,
        output_tokens=0,
        cached_input_tokens=300_000,
        cache_write_input_tokens=200_000,
    )
    assert cost_usd("claude-opus-4-8", usage) == pytest.approx(3.9)


def test_missing_cache_tier_bills_at_full_input_rate() -> None:
    # Embedding rows carry no cache tiers; cache traffic on them bills at the input rate,
    # never silently free.
    usage = TokenUsage(
        input_tokens=1_000_000, cached_input_tokens=600_000, cache_write_input_tokens=400_000
    )
    assert cost_usd("text-embedding-3-large", usage) == pytest.approx(0.13)


def test_cache_subsets_are_clamped_to_input() -> None:
    # Malformed usage claiming more cache traffic than input must not go negative: the read
    # subset clamps to input and the write subset to what remains.
    usage = TokenUsage(
        input_tokens=100_000,
        output_tokens=0,
        cached_input_tokens=80_000,
        cache_write_input_tokens=80_000,
    )
    # fable: 80k read @ $1/M-effective (0.1*10) + 20k write @ $12.5/M, 0 fresh.
    assert cost_usd("claude-fable-5", usage) == pytest.approx(0.08 + 0.25)
