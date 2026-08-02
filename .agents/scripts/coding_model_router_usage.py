"""Exact worker usage extraction and frozen model pricing for the coding-router study."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from wmo.providers.base import TokenUsage
from wmo.providers.pool import PoolEntry

OPENAI_LONG_CONTEXT_THRESHOLD = 272_000
ESTIMATE_METHOD = "trace-char-prefix-4k-overhead-v1"
ESTIMATE_CHARS_PER_TOKEN = 4
ESTIMATE_CALL_OVERHEAD_TOKENS = 4_096
OPENAI_LONG_CONTEXT_MODELS = frozenset(
    {
        "gpt-5.5",
        "gpt-5.5-2026-04-23",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
    }
)


def _nonnegative_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _int_list(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    return [_nonnegative_int(item) for item in value]


def _seconds_list(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    return [
        float(item)
        for item in value
        if isinstance(item, (int, float)) and not isinstance(item, bool) and item >= 0
    ]


@dataclass(frozen=True)
class DetailedUsage:
    """Episode totals plus request-level counters needed for tiered pricing."""

    total: TokenUsage
    calls: int
    call_seconds: list[float]
    call_input_tokens: list[int]
    call_output_tokens: list[int]
    call_cached_input_tokens: list[int]
    call_cache_write_input_tokens: list[int]


def usage_from_trace(trace: dict[str, object]) -> DetailedUsage:
    """Read a Harbor worker trace, accepting legacy traces with aggregate counters only."""
    raw = trace.get("worker_usage")
    if not isinstance(raw, dict):
        return DetailedUsage(TokenUsage(), 0, [], [], [], [], [])
    return DetailedUsage(
        total=TokenUsage(
            input_tokens=_nonnegative_int(raw.get("input_tokens")),
            output_tokens=_nonnegative_int(raw.get("output_tokens")),
            cached_input_tokens=_nonnegative_int(raw.get("cached_input_tokens")),
            cache_write_input_tokens=_nonnegative_int(raw.get("cache_write_input_tokens")),
            reasoning_tokens=_nonnegative_int(raw.get("reasoning_tokens")),
        ),
        calls=_nonnegative_int(raw.get("calls")),
        call_seconds=_seconds_list(raw.get("call_seconds")),
        call_input_tokens=_int_list(raw.get("call_input_tokens")),
        call_output_tokens=_int_list(raw.get("call_output_tokens")),
        call_cached_input_tokens=_int_list(raw.get("call_cached_input_tokens")),
        call_cache_write_input_tokens=_int_list(raw.get("call_cache_write_input_tokens")),
    )


def usage_metering_error(usage: DetailedUsage) -> str:
    """Return why an episode cannot be priced exactly, or an empty string when complete."""
    if usage.calls < 1:
        return "worker usage is missing a completed provider call"
    counters = {
        "call_seconds": len(usage.call_seconds),
        "call_input_tokens": len(usage.call_input_tokens),
        "call_output_tokens": len(usage.call_output_tokens),
        "call_cached_input_tokens": len(usage.call_cached_input_tokens),
        "call_cache_write_input_tokens": len(usage.call_cache_write_input_tokens),
    }
    incomplete = {name: count for name, count in counters.items() if count != usage.calls}
    if incomplete:
        return (
            f"worker usage reports {usage.calls} calls but carries incomplete request counters: "
            f"{incomplete}"
        )
    return ""


def estimate_usage_from_trace(trace: dict[str, object]) -> DetailedUsage:
    """Estimate request tokens from the cumulative trace when counters are absent.

    Each agent step corresponds to one provider response. Input estimates include the task and
    the prior action-observation transcript plus a fixed allowance for the system prompt and tool
    schema. Output estimates use the serialized action. This is intentionally labeled and is used
    for rough cost comparison only.
    """
    instruction = trace.get("instruction")
    base_chars = len(instruction) if isinstance(instruction, str) else 0
    raw_steps = trace.get("steps")
    steps = raw_steps if isinstance(raw_steps, list) else []
    raw_turns = trace.get("turns")
    turns = (
        raw_turns
        if isinstance(raw_turns, int) and not isinstance(raw_turns, bool) and raw_turns > 0
        else len(steps)
    )
    calls = max(turns, len(steps))
    if calls == 0:
        return DetailedUsage(TokenUsage(), 0, [], [], [], [], [])

    input_tokens: list[int] = []
    output_tokens: list[int] = []
    transcript_chars = 0
    for index in range(calls):
        input_tokens.append(
            ESTIMATE_CALL_OVERHEAD_TOKENS
            + math.ceil((base_chars + transcript_chars) / ESTIMATE_CHARS_PER_TOKEN)
        )
        raw_step = steps[index] if index < len(steps) else None
        step: dict[str, object] = (
            {str(key): value for key, value in raw_step.items()}
            if isinstance(raw_step, dict)
            else {}
        )
        action = step.get("action")
        observation = step.get("observation")
        action_chars = len(json.dumps(action, sort_keys=True, default=str))
        observation_chars = len(json.dumps(observation, sort_keys=True, default=str))
        output_tokens.append(max(1, math.ceil(action_chars / ESTIMATE_CHARS_PER_TOKEN)))
        transcript_chars += action_chars + observation_chars

    return DetailedUsage(
        total=TokenUsage(
            input_tokens=sum(input_tokens),
            output_tokens=sum(output_tokens),
        ),
        calls=calls,
        call_seconds=[0.0] * calls,
        call_input_tokens=input_tokens,
        call_output_tokens=output_tokens,
        call_cached_input_tokens=[0] * calls,
        call_cache_write_input_tokens=[0] * calls,
    )


def exact_cost_usd(entry: PoolEntry, usage: DetailedUsage) -> float:
    """Price one episode, applying the frozen OpenAI long-context multiplier per request."""
    if entry.model not in OPENAI_LONG_CONTEXT_MODELS:
        return entry.cost_usd(usage.total)

    lengths = {
        len(usage.call_input_tokens),
        len(usage.call_output_tokens),
        len(usage.call_cached_input_tokens),
        len(usage.call_cache_write_input_tokens),
    }
    if lengths == {0}:
        if usage.total.input_tokens > OPENAI_LONG_CONTEXT_THRESHOLD:
            raise ValueError(
                f"{entry.name} used {usage.total.input_tokens} aggregate input tokens but its "
                "trace has no per-call counters, so the >272k pricing tier cannot be resolved"
            )
        return entry.cost_usd(usage.total)
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError(f"{entry.name} trace carries incomplete per-call token counters")

    price = entry.price()
    read_rate = (
        entry.cached_input_per_mtok
        if entry.cached_input_per_mtok is not None
        else price.cache_read_per_mtok
        if price.cache_read_per_mtok is not None
        else price.input_per_mtok
    )
    write_rate = (
        entry.cache_write_per_mtok
        if entry.cache_write_per_mtok is not None
        else price.cache_write_per_mtok
        if price.cache_write_per_mtok is not None
        else price.input_per_mtok
    )
    total = 0.0
    for input_tokens, output_tokens, cached_tokens, write_tokens in zip(
        usage.call_input_tokens,
        usage.call_output_tokens,
        usage.call_cached_input_tokens,
        usage.call_cache_write_input_tokens,
        strict=True,
    ):
        read = min(cached_tokens, input_tokens)
        write = min(write_tokens, input_tokens - read)
        fresh = input_tokens - read - write
        long = input_tokens > OPENAI_LONG_CONTEXT_THRESHOLD
        input_multiplier = 2.0 if long else 1.0
        output_multiplier = 1.5 if long else 1.0
        total += (
            fresh * price.input_per_mtok * input_multiplier
            + read * read_rate * input_multiplier
            + write * write_rate * input_multiplier
            + output_tokens * price.output_per_mtok * output_multiplier
        ) / 1_000_000
    return total
