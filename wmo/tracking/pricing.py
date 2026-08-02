"""Per-model token pricing → USD cost.

Provider-agnostic: prices are keyed by a normalized model id (provider prefixes like Bedrock's
`us.anthropic.` are stripped before lookup), so the same Opus 4.8 row covers the direct API and
Bedrock. Prices are USD per 1M tokens; an unknown model costs 0.0 and is flagged so callers can
surface "cost unavailable" rather than silently under-reporting.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from wmo.providers.base import TokenUsage

# Bedrock appends a snapshot date and/or version to the model id, e.g.
# `claude-haiku-4-5-20251001-v1:0` or `claude-opus-4-6-v1`. Strip them so the lookup key matches the
# undated table rows (`claude-haiku-4-5`). Only applied to `claude-*` ids.
_BEDROCK_SUFFIX = re.compile(r"(-\d{8})?(-v\d+)?(:\d+)?$")
_OPENAI_SNAPSHOT_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")
_OPENAI_LONG_CONTEXT_THRESHOLD = 272_000
_OPENAI_LONG_CONTEXT_MODELS = frozenset({"gpt-5.5", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"})


class ModelPrice(BaseModel):
    """USD per 1,000,000 tokens, split by input/output plus optional prompt-cache tiers.

    `cache_read_per_mtok` prices prompt tokens served from the provider cache;
    `cache_write_per_mtok` prices tokens written to it (a premium on Anthropic). None means
    "no known cache rate": `cost_usd` then bills that tier at the full input rate, never $0.
    """

    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float | None = None
    cache_write_per_mtok: float | None = None


def _claude_price(input_per_mtok: float, output_per_mtok: float) -> ModelPrice:
    """A Claude row with Anthropic's published cache multipliers applied.

    Cache reads bill at 0.1x the base input rate; cache writes at 1.25x for the default
    5-minute TTL (the 1h TTL bills 2x; nothing in the harness requests it, and a pool entry
    can override per model if that changes).
    """
    return ModelPrice(
        input_per_mtok=input_per_mtok,
        output_per_mtok=output_per_mtok,
        cache_read_per_mtok=input_per_mtok * 0.1,
        cache_write_per_mtok=input_per_mtok * 1.25,
    )


def _openai_price(input_per_mtok: float, output_per_mtok: float) -> ModelPrice:
    """A GPT-5.x row with OpenAI's published cached-input discount applied.

    Cached input bills at 0.1x the base input rate (the gpt-5 family's 90% discount).
    OpenAI charges no cache-write premium and reports no write tokens, so the write tier
    stays None (moot in practice: cache_write_input_tokens is always 0 on OpenAI usage).
    """
    return ModelPrice(
        input_per_mtok=input_per_mtok,
        output_per_mtok=output_per_mtok,
        cache_read_per_mtok=input_per_mtok * 0.1,
    )


# Keyed by normalized model id (see `_normalize`). USD per 1M tokens.
#
# Completion prices verified 2026-06-25 against the live vendor pricing pages:
#   - Claude: platform.claude.com/docs/en/about-claude/models/overview
#   - OpenAI GPT-5.x: developers.openai.com/api/docs/pricing (Standard tier, short context)
# Cache tiers (added 2026-07-25) are the vendors' published multipliers on those verified base
# rates, not independently re-fetched per model: Anthropic reads 0.1x / writes 1.25x (5m TTL),
# OpenAI cached input 0.1x with no write charge. Re-verify per model if cache cost is headline.
# Embedding prices are long-stable list prices NOT re-fetched in that pass (the OpenAI pricing
# page no longer surfaces them); treat as approximate and re-verify if embed cost matters.
_PRICES: dict[str, ModelPrice] = {
    # --- Anthropic / Bedrock (Claude) ---
    "claude-fable-5": _claude_price(input_per_mtok=10.0, output_per_mtok=50.0),
    "claude-mythos-5": _claude_price(input_per_mtok=10.0, output_per_mtok=50.0),
    "claude-opus-5": _claude_price(input_per_mtok=5.0, output_per_mtok=25.0),
    "claude-opus-4-8": _claude_price(input_per_mtok=5.0, output_per_mtok=25.0),
    "claude-opus-4-7": _claude_price(input_per_mtok=5.0, output_per_mtok=25.0),
    "claude-opus-4-6": _claude_price(input_per_mtok=5.0, output_per_mtok=25.0),
    "claude-opus-4-5": _claude_price(input_per_mtok=5.0, output_per_mtok=25.0),
    "claude-opus-4-1": _claude_price(input_per_mtok=15.0, output_per_mtok=75.0),
    "claude-sonnet-5": _claude_price(input_per_mtok=3.0, output_per_mtok=15.0),
    "claude-sonnet-4-6": _claude_price(input_per_mtok=3.0, output_per_mtok=15.0),
    "claude-haiku-4-5": _claude_price(input_per_mtok=1.0, output_per_mtok=5.0),
    # --- OpenAI / Azure OpenAI (GPT-5.x; Azure deployments reuse the base model's price) ---
    "gpt-5.6-sol": _openai_price(input_per_mtok=5.0, output_per_mtok=30.0),
    "gpt-5.6-terra": _openai_price(input_per_mtok=2.5, output_per_mtok=15.0),
    "gpt-5.6-luna": _openai_price(input_per_mtok=1.0, output_per_mtok=6.0),
    "gpt-5.5": _openai_price(input_per_mtok=5.0, output_per_mtok=30.0),
    "gpt-5.5-pro": _openai_price(input_per_mtok=30.0, output_per_mtok=180.0),
    "gpt-5.4": _openai_price(input_per_mtok=2.5, output_per_mtok=15.0),
    "gpt-5.4-mini": _openai_price(input_per_mtok=0.75, output_per_mtok=4.5),
    "gpt-5.4-nano": _openai_price(input_per_mtok=0.2, output_per_mtok=1.25),
    # Self-hosted models (vLLM on our own GPUs) intentionally have NO row: their cost is amortized
    # GPU time, not a per-token API price. `price_for` returns None for them, which the eval grid
    # renders as "no cost"; a 0.0 ModelPrice would instead report a misleading $0.00.
    # --- Embeddings (output tokens are always 0 for embed calls) ---
    "text-embedding-3-small": ModelPrice(input_per_mtok=0.02, output_per_mtok=0.0),
    "text-embedding-3-large": ModelPrice(input_per_mtok=0.13, output_per_mtok=0.0),
    "amazon.titan-embed-text-v2:0": ModelPrice(input_per_mtok=0.02, output_per_mtok=0.0),
}


def _normalize(model: str) -> str:
    """Strip provider/region routing prefixes so one row covers a model across providers.

    Bedrock ids look like `us.anthropic.claude-opus-4-8`; the direct API uses `claude-opus-4-8`.
    We drop a leading region segment (`us.`/`eu.`/...) and an `anthropic.` vendor segment, but keep
    `amazon.titan-...` (its `amazon.` is part of the canonical model id, not a routing prefix).
    """
    normalized = model.strip()
    region_prefixes = ("us.", "eu.", "apac.", "us-gov.", "global.", "jp.", "au.", "ca.")
    for prefix in region_prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    if normalized.startswith("anthropic."):
        normalized = normalized[len("anthropic.") :]
    if normalized.startswith("claude-"):
        # Drop a trailing Bedrock snapshot date / version (`-20251001-v1:0`, `-v1`) so dated
        # inference-profile ids match the undated table rows.
        normalized = _BEDROCK_SUFFIX.sub("", normalized)
    elif normalized.startswith("gpt-"):
        # OpenAI exact snapshots use an ISO date suffix while the frozen table is keyed by
        # stable model family. Keep semantic version digits such as gpt-5.3-codex intact.
        normalized = _OPENAI_SNAPSHOT_SUFFIX.sub("", normalized)
    return normalized


def price_for(model: str) -> ModelPrice | None:
    """Return the price row for `model` (after normalization), or None if unknown."""
    return _PRICES.get(_normalize(model))


def request_price_multipliers(model: str, input_tokens: int) -> tuple[float, float]:
    """Return the input/output price multipliers for one provider request."""
    if (
        _normalize(model) in _OPENAI_LONG_CONTEXT_MODELS
        and input_tokens > _OPENAI_LONG_CONTEXT_THRESHOLD
    ):
        return 2.0, 1.5
    return 1.0, 1.0


def cost_usd(model: str, usage: TokenUsage) -> float:
    """USD cost of `usage` on `model`. Unknown models cost 0.0 (see `price_for` to detect that).

    Cache-adjusted: cache-read and cache-write subsets of `input_tokens` bill at their tier
    rates when the price row carries them, and at the full input rate otherwise (never $0).
    GPT-5.5 and GPT-5.6 calls above 272k input tokens apply their published per-request 2x
    input and 1.5x output multipliers. Callers must invoke this once per provider request,
    which `RunTracker.record` and `MeteredProvider` do. With no cache traffic below that
    threshold this reduces exactly to input*input_rate + output*output_rate.
    """
    price = price_for(model)
    if price is None:
        return 0.0
    read = min(usage.cached_input_tokens, usage.input_tokens)
    write = min(usage.cache_write_input_tokens, usage.input_tokens - read)
    read_rate = (
        price.cache_read_per_mtok if price.cache_read_per_mtok is not None else price.input_per_mtok
    )
    write_rate = (
        price.cache_write_per_mtok
        if price.cache_write_per_mtok is not None
        else price.input_per_mtok
    )
    input_multiplier, output_multiplier = request_price_multipliers(model, usage.input_tokens)
    return (
        (usage.input_tokens - read - write) * price.input_per_mtok * input_multiplier
        + read * read_rate * input_multiplier
        + write * write_rate * input_multiplier
        + usage.output_tokens * price.output_per_mtok * output_multiplier
    ) / 1_000_000
