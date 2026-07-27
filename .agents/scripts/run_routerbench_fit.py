"""Fit the Avengers-replica rank policy on RouterBench and score it against the baselines.

Stage A of the approved benchmark plan: their frozen 11-model matrix, our fitter. Reference
defaults (k=64, top_k=2, beta=6.0, seed=42); embedder swept hashing-512 vs hashing-1024 (the
provider-embedding comparison lands separately). Prints the benchmark table and a routed-mix
audit; artifacts under .wmo/evals/routerbench/.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

from wmo.optimize.policy import EmbedderSpec
from wmo.optimize.routing import evaluate_policy, fit_rank_policy
from wmo.research.routerbench import (
    best_single_model,
    load_routerbench,
    oracle,
    random_baseline,
    split_scenario_ids,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("routerbench_fit")

PICKLE = Path("/Users/silen/Desktop/Projects/router-refs/routerbench_0shot.pkl")
OUT_DIR = Path(".wmo/evals/routerbench")


def main() -> None:
    matrix = load_routerbench(PICKLE)
    fit_ids, test_ids = split_scenario_ids(matrix, train_fraction=0.7, seed=0)
    logger.info("fit=%d test=%d scenarios", len(fit_ids), len(test_ids))

    name, bs_acc, bs_cost = best_single_model(matrix, fit_ids=fit_ids, eval_ids=test_ids)
    o_acc, o_cost = oracle(matrix, test_ids)
    r_acc, r_cost = random_baseline(matrix, test_ids)
    logger.info("best-single %s: acc=%.4f cost=$%.5f", name, bs_acc, bs_cost)
    logger.info("oracle:            acc=%.4f cost=$%.5f", o_acc, o_cost)
    logger.info("random:            acc=%.4f cost=$%.5f", r_acc, r_cost)

    results = {}
    for dim in (512, 1024):
        started = time.monotonic()
        policy = fit_rank_policy(
            matrix,
            fit_ids=fit_ids,
            embedder=EmbedderSpec(dim=dim),
            n_clusters=64,
            seed=42,
            fitted_from=f"routerbench_0shot.pkl split-seed0 hashing-{dim}",
        )
        fit_seconds = time.monotonic() - started
        result = evaluate_policy(policy, matrix, test_ids)
        results[f"hashing-{dim}"] = result.model_dump()
        logger.info(
            "rank(hashing-%d): acc=%.4f cost=$%.5f (fit %.1fs, unscored=%d)",
            dim,
            result.accuracy,
            result.cost_per_scenario,
            fit_seconds,
            result.unscored_scenarios,
        )
        top_mix = sorted(result.model_mix.items(), key=lambda kv: -kv[1])[:5]
        logger.info("  mix: %s", {m: round(s, 3) for m, s in top_mix})
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        policy.save(OUT_DIR / f"policy_hashing_{dim}.json")

    (OUT_DIR / "results.json").write_text(
        json.dumps(
            {
                "baselines": {
                    "best_single": {"model": name, "accuracy": bs_acc, "cost": bs_cost},
                    "oracle": {"accuracy": o_acc, "cost": o_cost},
                    "random": {"accuracy": r_acc, "cost": r_cost},
                },
                "policies": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("wrote %s", OUT_DIR / "results.json")
    sys.exit(0)


if __name__ == "__main__":
    main()
