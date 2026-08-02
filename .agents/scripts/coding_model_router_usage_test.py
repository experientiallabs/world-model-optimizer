"""Tests for request-level pricing used by the coding-router runners."""

from __future__ import annotations

import pytest
from coding_model_router_usage import (
    ESTIMATE_METHOD,
    DetailedUsage,
    estimate_usage_from_trace,
    exact_cost_usd,
    usage_from_trace,
    usage_metering_error,
)

from wmo.providers.base import ProviderKind, TokenUsage
from wmo.providers.pool import PoolEntry


def _entry() -> PoolEntry:
    return PoolEntry(
        name="sol",
        kind=ProviderKind.OPENAI_RESPONSES,
        model="gpt-5.6-sol",
        input_per_mtok=5.0,
        cached_input_per_mtok=0.5,
        output_per_mtok=30.0,
    )


def test_usage_from_trace_preserves_per_call_counters() -> None:
    usage = usage_from_trace(
        {
            "worker_usage": {
                "input_tokens": 301_000,
                "output_tokens": 10_100,
                "cached_input_tokens": 100_000,
                "cache_write_input_tokens": 0,
                "reasoning_tokens": 5_000,
                "calls": 2,
                "call_seconds": [1, 2],
                "call_input_tokens": [1_000, 300_000],
                "call_output_tokens": [100, 10_000],
                "call_cached_input_tokens": [0, 100_000],
                "call_cache_write_input_tokens": [0, 0],
            }
        }
    )
    assert usage.call_input_tokens == [1_000, 300_000]
    assert usage.call_output_tokens == [100, 10_000]
    assert usage.call_seconds == [1.0, 2.0]
    assert usage.calls == 2


def test_exact_cost_applies_long_context_rates_per_call() -> None:
    usage = DetailedUsage(
        total=TokenUsage(
            input_tokens=301_000,
            output_tokens=10_100,
            cached_input_tokens=100_000,
        ),
        calls=2,
        call_seconds=[1.0, 2.0],
        call_input_tokens=[1_000, 300_000],
        call_output_tokens=[100, 10_000],
        call_cached_input_tokens=[0, 100_000],
        call_cache_write_input_tokens=[0, 0],
    )
    # First call: 1k*$5/M + 100*$30/M = $0.008.
    # Long call: 200k*$10/M + 100k*$1/M + 10k*$45/M = $2.55.
    assert exact_cost_usd(_entry(), usage) == pytest.approx(2.558)
    assert usage_metering_error(usage) == ""


def test_legacy_aggregate_above_boundary_is_rejected() -> None:
    usage = DetailedUsage(
        total=TokenUsage(input_tokens=300_000),
        calls=0,
        call_seconds=[],
        call_input_tokens=[],
        call_output_tokens=[],
        call_cached_input_tokens=[],
        call_cache_write_input_tokens=[],
    )
    with pytest.raises(ValueError, match="per-call"):
        exact_cost_usd(_entry(), usage)


def test_usage_metering_rejects_missing_and_partial_request_counters() -> None:
    missing = usage_from_trace({"worker_usage": None})
    partial = usage_from_trace(
        {
            "worker_usage": {
                "calls": 2,
                "call_seconds": [1.0, 2.0],
                "call_input_tokens": [10],
            }
        }
    )

    assert "missing" in usage_metering_error(missing)
    assert "incomplete" in usage_metering_error(partial)


def test_trace_usage_estimate_is_cumulative_and_priceable() -> None:
    usage = estimate_usage_from_trace(
        {
            "instruction": "repair the repository",
            "turns": 2,
            "steps": [
                {
                    "action": {"kind": "tool_call", "name": "bash"},
                    "observation": {"content": "first result"},
                },
                {
                    "action": {"kind": "message", "content": "done"},
                    "observation": {"content": ""},
                },
            ],
        }
    )

    assert ESTIMATE_METHOD == "trace-char-prefix-4k-overhead-v1"
    assert usage.calls == 2
    assert usage.call_input_tokens[1] > usage.call_input_tokens[0]
    assert usage.total.input_tokens == sum(usage.call_input_tokens)
    assert usage.total.output_tokens == sum(usage.call_output_tokens)
    assert exact_cost_usd(_entry(), usage) > 0
