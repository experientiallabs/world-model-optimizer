"""Live closed-loop smoke: 2 pool candidates x 2 bird-sql scenarios against the built WM.

Proves the Task-4 seam end to end on real backends: pool file -> providers, WM env with
score_on_close, evaluate_pool -> OutcomeMatrix persisted for inspection. Cheap by design
(haiku WM, 2 scenarios, max_steps 6).
"""

from __future__ import annotations

import logging
from pathlib import Path

from wmo.engine.world_model import WorldModel
from wmo.env.base import WorldModelEnv
from wmo.env.closed_loop import evaluate_pool
from wmo.env.scenarios import scenarios_from_traces
from wmo.ingest import get_adapter
from wmo.providers.base import ProviderConfig, ProviderKind
from wmo.providers.pool import ModelPool, load_pool
from wmo.providers.registry import get_provider

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("closed_loop_smoke")

WM_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
CANDIDATES = ["haiku-4-5", "gpt-5.4-mini"]


def main() -> None:
    provider = get_provider(
        ProviderConfig(kind=ProviderKind.BEDROCK, model=WM_MODEL, region="us-east-1")
    )
    wm = WorldModel.load(".wmo/models/bird-sql", provider)

    adapter = get_adapter("otel-genai")
    traces = adapter.from_file("packages/environment-capture/bird-sql/traces.otel.jsonl")
    scenarios = scenarios_from_traces(traces)[:2]
    logger.info("scenarios: %s", [s.task[:80] for s in scenarios])

    full = load_pool()
    pool = ModelPool(models=[full.entry(name) for name in CANDIDATES])

    matrix = evaluate_pool(
        lambda: WorldModelEnv(wm, score_on_close=True),
        pool,
        scenarios,
        max_steps=6,
    )
    out = Path(".wmo/evals/closed_loop_smoke.json")
    matrix.save(out)
    logger.info("saved %s", out)
    for outcome in matrix.outcomes:
        latency = max(outcome.call_seconds) if outcome.call_seconds else 0.0
        logger.info(
            "%s on %s: reward=%s success=%s steps=%d cost=$%.5f max_call=%.2fs error=%s",
            outcome.model,
            outcome.scenario_id[:12],
            "-" if outcome.reward is None else f"{outcome.reward:.2f}",
            outcome.success,
            outcome.steps,
            outcome.cost_usd,
            latency,
            outcome.error,
        )


if __name__ == "__main__":
    main()
