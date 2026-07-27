"""Stage B analysis: fit + benchmark the router on OUR 9-model matrix.

Answers the three product questions with our models on certified RouterBench MCQ prompts:
cost, quality, and speed of the routed policy vs the best single model in hindsight, plus
oracle headroom, the knob-swept frontier, and AIQ vs the Zero-Router floor. Latency is the
mix-weighted mean/p50 of the routed choices' measured call seconds.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from statistics import median

from wmo.optimize.outcomes import OutcomeMatrix
from wmo.optimize.policy import EmbedderSpec, RoutingPolicy
from wmo.optimize.routing import evaluate_policy, fit_rank_policy, rerank_policy
from wmo.research.routerbench import (
    aiq,
    best_single_model,
    oracle,
    single_model_points,
    split_scenario_ids,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("ours_analyze")

MATRIX = Path(".wmo/evals/routerbench/ours_matrix.json")
OUT = Path(".wmo/evals/routerbench/ours_results.json")


def _latency(matrix: OutcomeMatrix, ids: list[str], choose) -> tuple[float, float]:
    """(mean_s, p50_s) of the chosen model's measured call seconds per scenario."""
    rows = {}
    for outcome in matrix.outcomes:
        if outcome.reward is None or not outcome.call_seconds:
            continue
        rows[(outcome.scenario_id, outcome.model)] = outcome.call_seconds[0]
    seconds = []
    for sid in ids:
        model = choose(sid)
        value = rows.get((sid, model))
        if value is not None:
            seconds.append(value)
    return (sum(seconds) / len(seconds), median(seconds)) if seconds else (0.0, 0.0)


def main() -> None:
    matrix = OutcomeMatrix.load(MATRIX)
    fit_ids, test_ids = split_scenario_ids(matrix, train_fraction=0.7, seed=0)
    logger.info("ours matrix: %d scenarios (%d fit / %d test), %d models",
                len(matrix.scenario_ids()), len(fit_ids), len(test_ids), len(matrix.pool))

    name, bs_acc, bs_cost = best_single_model(matrix, fit_ids=fit_ids, eval_ids=test_ids)
    o_acc, o_cost = oracle(matrix, test_ids)
    singles = single_model_points(matrix, test_ids)
    logger.info("single models (test): %s",
                {m: (round(c, 5), round(a, 3)) for m, (c, a) in sorted(singles.items(), key=lambda kv: kv[1][0])})
    logger.info("best-single %s: acc=%.4f cost=$%.5f", name, bs_acc, bs_cost)
    logger.info("oracle: acc=%.4f cost=$%.5f", o_acc, o_cost)

    policy = fit_rank_policy(
        matrix, fit_ids=fit_ids, embedder=EmbedderSpec(dim=1024), n_clusters=64, seed=42,
        fitted_from="ours_matrix split0 hashing-1024",
    )
    frontier = []
    for lam in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        swept = rerank_policy(policy, cost_weight=lam) if lam else policy
        result = evaluate_policy(swept, matrix, test_ids)
        decisions = {}
        from wmo.optimize.policy import rank_decision
        import numpy as np
        from sklearn.preprocessing import Normalizer
        tasks = {o.scenario_id: o.task for o in matrix.outcomes}
        emb = np.asarray(swept.embedder.build().embed([tasks[sid] for sid in test_ids]))
        emb = Normalizer(norm="l2").transform(emb)
        for index, sid in enumerate(test_ids):
            decisions[sid] = rank_decision(swept, emb[index]).model
        mean_s, p50_s = _latency(matrix, test_ids, lambda sid: decisions[sid])
        frontier.append({
            "lam": lam, "acc": result.accuracy, "cost": result.cost_per_scenario,
            "mean_s": mean_s, "p50_s": p50_s, "mix": result.model_mix,
        })
        logger.info("lam=%-5g acc=%.4f cost=$%.5f p50=%.2fs mix_top=%s",
                    lam, result.accuracy, result.cost_per_scenario, p50_s,
                    sorted(result.model_mix.items(), key=lambda kv: -kv[1])[:3])

    bs_mean_s, bs_p50_s = _latency(matrix, test_ids, lambda sid: name)
    max_cost = max(max(c for c, a in singles.values()), max(f["cost"] for f in frontier))
    zero = aiq(list(singles.values()), max_cost=max_cost)
    ours = aiq([(f["cost"], f["acc"]) for f in frontier], max_cost=max_cost)
    logger.info("best-single latency: mean=%.2fs p50=%.2fs", bs_mean_s, bs_p50_s)
    logger.info("AIQ: ours=%.4f zero-router=%.4f", ours, zero)

    OUT.write_text(json.dumps({
        "singles": {m: {"cost": c, "acc": a} for m, (c, a) in singles.items()},
        "best_single": {"model": name, "acc": bs_acc, "cost": bs_cost,
                        "mean_s": bs_mean_s, "p50_s": bs_p50_s},
        "oracle": {"acc": o_acc, "cost": o_cost},
        "frontier": frontier,
        "aiq": {"ours": ours, "zero_router": zero},
    }, indent=2), encoding="utf-8")
    logger.info("wrote %s", OUT)
    sys.exit(0)


if __name__ == "__main__":
    main()
