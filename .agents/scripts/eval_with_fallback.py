"""Run a benchmark fidelity eval through an llm-waterfall chain (same-weights first links).

Chain: bedrock/opus-4-8 -> anthropic-direct/claude-opus-4-8 -> bedrock/opus-4-7, with chain-wide
wraparound retry (RetryPolicy rounds) — llm-waterfall owns ALL failover/backoff, replacing the
earlier hand-rolled FallbackProvider + retry wrapper. The first two links serve identical Opus
4.8 weights via different transports, so Bedrock flaps never change the judge (D12
comparability); 4-7 is the last resort.

Usage: uv run python .agents/scripts/eval_with_fallback.py <suite-root> <benchmark>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from llm_waterfall import Backend, RetryPolicy, Waterfall

from wmh.engine.build import split_traces
from wmh.engine.prompts import BASE_ENV_PROMPT
from wmh.engine.replay import replay
from wmh.ingest import get_adapter
from wmh.optimize.judge import RubricJudge
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind, TokenUsage
from wmh.retrieval.embedders import HashingEmbedder
from wmh.retrieval.retriever import EmbeddingRetriever


class WaterfallProvider:
    """wmh Provider facade over an llm_waterfall.Waterfall (complete() only, for eval runs)."""

    def __init__(self, waterfall: Waterfall) -> None:
        self._waterfall = waterfall
        self.config = ProviderConfig(
            kind=ProviderKind.BEDROCK, model="us.anthropic.claude-opus-4-8", region="us-east-1"
        )

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        # temperature is dropped: the anthropic-direct link rejects it for Opus 4.8
        # ("deprecated for this model"), and fidelity replay doesn't depend on it.
        del temperature
        result = self._waterfall.complete(
            system=system,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            max_tokens=max_tokens,
        )
        usage = TokenUsage(
            input_tokens=result.usage.input_tokens, output_tokens=result.usage.output_tokens
        )
        return Completion(text=result.text, usage=usage)


def main() -> None:
    root, bench = Path(sys.argv[1]), sys.argv[2]
    chain = WaterfallProvider(
        Waterfall(
            [
                Backend("bedrock", "us.anthropic.claude-opus-4-8", region="us-east-1"),
                Backend("anthropic", "claude-opus-4-8"),
                Backend("bedrock", "us.anthropic.claude-opus-4-7", region="us-east-1"),
            ],
            retry=RetryPolicy(rounds=6, backoff_base_s=15.0),
        )
    )
    resilient = chain
    # Mirrors evaluate_files' per-file body, adding replay's concurrency (steps are independent).
    traces = get_adapter("otel-genai").from_file(str(root / bench / "traces.otel.jsonl"))
    train, holdout = split_traces(traces, 0.7)
    if not holdout:
        train, holdout = traces, traces
    from wmh.retrieval.retriever import EmbeddingRetriever

    entry = replay(
        BASE_ENV_PROMPT,
        holdout,
        resilient,
        RubricJudge(resilient),
        retriever=EmbeddingRetriever(HashingEmbedder(dim=512)),
        train=train,
        top_k=5,
        sample_turns="all",
        seed=0,
        concurrency=8,
    )
    flagged = [
        r
        for r in entry.results
        if r.is_error_actual is not None and r.is_error_predicted is not None
    ]
    err_acc = (
        sum(1 for r in flagged if r.is_error_actual == r.is_error_predicted) / len(flagged)
        if flagged
        else None
    )
    scores = [r.score for r in entry.results]
    mean = sum(scores) / len(scores) if scores else 0.0
    var = sum((s - mean) ** 2 for s in scores) / len(scores) if scores else 0.0
    out = {
        "benchmark": bench,
        "fidelity": round(mean, 4),
        "std": round(var**0.5, 4),
        "n_steps": len(scores),
        "error_flag_accuracy": round(err_acc, 4) if err_acc is not None else None,
    }
    print(json.dumps(out))
    Path(f"/tmp/fallback-eval-{bench}.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
