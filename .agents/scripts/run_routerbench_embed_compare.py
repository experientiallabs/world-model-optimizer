"""Embedder comparison on RouterBench: hashing vs text-embedding-3-large (same subsample).

The kNN-beats-routers paper's point (2505.12601): cluster routing works iff win-rates are
locally smooth in the embedding space. Hashing trigram is lexical; this run measures how much
a real semantic embedder buys on the identical split. Subsampled to ~12k prompts to respect
the embedding deployment's rate limits; both embedders see exactly the same scenarios.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from wmo.optimize.policy import EmbedderSpec
from wmo.optimize.routing import evaluate_policy, fit_rank_policy
from wmo.research.routerbench import (
    best_single_model,
    load_routerbench,
    oracle,
    split_scenario_ids,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("embed_compare")

PICKLE = Path("/Users/silen/Desktop/Projects/router-refs/routerbench_0shot.pkl")
OUT = Path(".wmo/evals/routerbench/embed_compare.json")

SPECS = {
    "hashing-1024": EmbedderSpec(dim=1024),
    "azure-3large-3072": EmbedderSpec(
        kind="azure",
        dim=3072,
        deployment="text-embedding-3-large",
        endpoint="https://google-sheets.openai.azure.com",
        api_key_env="AZURE_GOOGLE_SHEETS_API_KEY",
        batch=256,
    ),
}


def main() -> None:
    matrix = load_routerbench(PICKLE, sample=12000, seed=7)
    fit_ids, test_ids = split_scenario_ids(matrix, train_fraction=0.7, seed=0)
    logger.info("subsample: fit=%d test=%d", len(fit_ids), len(test_ids))
    name, bs_acc, bs_cost = best_single_model(matrix, fit_ids=fit_ids, eval_ids=test_ids)
    o_acc, o_cost = oracle(matrix, test_ids)
    logger.info("best-single %s: acc=%.4f cost=$%.5f", name, bs_acc, bs_cost)
    logger.info("oracle: acc=%.4f cost=$%.5f", o_acc, o_cost)

    results = {
        "baselines": {
            "best_single": {"model": name, "accuracy": bs_acc, "cost": bs_cost},
            "oracle": {"accuracy": o_acc, "cost": o_cost},
        }
    }
    for label, spec in SPECS.items():
        started = time.monotonic()
        policy = fit_rank_policy(
            matrix,
            fit_ids=fit_ids,
            embedder=spec,
            n_clusters=64,
            seed=42,
            fitted_from=f"routerbench sample12000@7 {label}",
        )
        result = evaluate_policy(policy, matrix, test_ids)
        elapsed = time.monotonic() - started
        results[label] = result.model_dump()
        logger.info(
            "%s: acc=%.4f cost=$%.5f (%.0fs) mix_top=%s",
            label,
            result.accuracy,
            result.cost_per_scenario,
            elapsed,
            sorted(result.model_mix.items(), key=lambda kv: -kv[1])[:3],
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
