"""Candidate #3 round 1: per-customer PIN selection via hierarchical shrinkage.

The serving question this answers: a new narrow-domain customer arrives with a tiny
evidence bank (17-56 scenarios). Which model should their endpoint PIN as the fallback:
the global prior (what the fleet says), their own noisy local evidence, or an
evidence-weighted blend (empirical Bayes / James-Stein shrinkage)?

Protocol (corpus-as-customer, $0): each of the wm corpora plays the customer. The GLOBAL
prior is the per-model mean over the OTHER corpora (leave-one-corpus-out, so no customer
sees its own data in the prior). LOCAL evidence is the customer's iid fit split (seeds
0-4). The pin is argmax of

    shrunk(m) = w * local_mean(m) + (1 - w) * global_mean(m),   w = n_m / (n_m + k)

with n_m = local scored episodes for model m and k the pseudo-count knob: k=0 is
local-only (the fit-discovered best single), k=inf is global-only (the fleet pin). Scored
on the customer's held-out test split, paired by seed. Kill criterion: no k beats BOTH
endpoints on mean paired accuracy.

Cohorts: the 25-scen wm corpora + the two s80 cohorts (kept separate). Output is a table
in the log + a findings entry; no runs.jsonl rows (pin selection, not a routing variant).

Usage: uv run python .agents/scripts/r1_pin_shrinkage.py
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from wmo.optimize.outcomes import OutcomeMatrix
from wmo.research.routerbench import split_scenario_ids
from wmo.research.routing_corpus import routing_data

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("r1.shrink")

DATA = routing_data()
CORPORA = [
    "bird-sql",
    "continual-learning",
    "crmarena",
    "dabstep",
    "financebench",
    "gaia2",
    "swe-bench",
    "tau-bench",
    "tau-telecom",
    "terminal-tasks",
    "financebench-s80",
    "tau-bench-s80",
]
KS = [0.0, 2.0, 5.0, 10.0, 20.0, 50.0, float("inf")]
SEEDS = range(5)


def model_stats(matrix: OutcomeMatrix, ids: set[str]) -> tuple[dict[str, float], dict[str, int]]:
    """Per-model (mean reward, episode count) over scored outcomes in `ids`."""
    sums: dict[str, list[float]] = defaultdict(list)
    for outcome in matrix.outcomes:
        if outcome.scenario_id in ids and outcome.reward is not None:
            sums[outcome.model].append(outcome.reward)
    return (
        {m: float(np.mean(v)) for m, v in sums.items()},
        {m: len(v) for m, v in sums.items()},
    )


def main() -> None:
    matrices = {c: OutcomeMatrix.load(DATA / "matrices" / f"{c}_matrix.json") for c in CORPORA}
    models = matrices[CORPORA[0]].model_names()

    # Leave-one-corpus-out global priors (episode-weighted mean over the other corpora).
    def global_prior(excluded: str) -> dict[str, float]:
        sums: dict[str, list[float]] = defaultdict(list)
        for corpus, matrix in matrices.items():
            # An s80 cohort and its 25-scen sibling share scenarios; exclude the family.
            if corpus.split("-s80")[0] == excluded.split("-s80")[0]:
                continue
            for outcome in matrix.outcomes:
                if outcome.reward is not None:
                    sums[outcome.model].append(outcome.reward)
        return {m: float(np.mean(v)) for m, v in sums.items() if v}

    # accs[k][(corpus, seed)] = test accuracy of the chosen pin
    accs: dict[float, dict[tuple[str, int], float]] = {k: {} for k in KS}
    for corpus, matrix in matrices.items():
        prior = global_prior(corpus)
        for seed in SEEDS:
            fit_ids, test_ids = split_scenario_ids(matrix, train_fraction=0.7, seed=seed)
            local_mean, local_n = model_stats(matrix, set(fit_ids))
            test_mean, _ = model_stats(matrix, set(test_ids))
            for k in KS:

                def shrunk(
                    model: str,
                    k: float = k,
                    local_mean: dict = local_mean,
                    local_n: dict = local_n,
                    prior: dict = prior,
                ) -> float:
                    local = local_mean.get(model)
                    if local is None or k == float("inf"):
                        return prior.get(model, -1.0)
                    if k == 0.0:
                        return local
                    w = local_n.get(model, 0) / (local_n.get(model, 0) + k)
                    return w * local + (1 - w) * prior.get(model, local)

                pin = max(models, key=shrunk)
                accs[k][(corpus, seed)] = test_mean.get(pin, 0.0)

    cells = sorted(accs[0.0])
    logger.info("pin test accuracy, mean over %d (corpus, seed) cells:", len(cells))
    local_acc = np.array([accs[0.0][c] for c in cells])
    global_acc = np.array([accs[float("inf")][c] for c in cells])
    for k in KS:
        arr = np.array([accs[k][c] for c in cells])
        label = {0.0: "local-only", float("inf"): "global-only"}.get(k, f"k={k:g}")
        wins_l = int(np.sum(arr > local_acc + 1e-12))
        loss_l = int(np.sum(arr < local_acc - 1e-12))
        wins_g = int(np.sum(arr > global_acc + 1e-12))
        loss_g = int(np.sum(arr < global_acc - 1e-12))
        logger.info(
            "%-12s mean=%.4f | vs local %+.4f (%dW/%dL) | vs global %+.4f (%dW/%dL)",
            label,
            float(arr.mean()),
            float((arr - local_acc).mean()),
            wins_l,
            loss_l,
            float((arr - global_acc).mean()),
            wins_g,
            loss_g,
        )
    # Where do the endpoints disagree? The shrinkage value lives only on those cells.
    disagree = [c for c in cells if abs(accs[0.0][c] - accs[float("inf")][c]) > 1e-12]
    logger.info(
        "cells where local and global pins score differently: %d/%d", len(disagree), len(cells)
    )
    for k in (5.0, 20.0):
        arr = np.array([accs[k][c] for c in disagree])
        loc = np.array([accs[0.0][c] for c in disagree])
        glo = np.array([accs[float("inf")][c] for c in disagree])
        logger.info(
            "  on disagreement cells, k=%g: %+0.4f vs local, %+0.4f vs global",
            k,
            float((arr - loc).mean()),
            float((arr - glo).mean()),
        )

    # Nested validation (the JiSi lesson applied to our own knob): choose k for each
    # customer from the OTHER corpus families only, then score it on the customer. This is
    # the number a deployment would actually get.
    nested = {}
    for corpus in matrices:
        family = corpus.split("-s80")[0]
        other = [c for c in cells if c[0].split("-s80")[0] != family]
        k_star = max(KS, key=lambda k: float(np.mean([accs[k][c] for c in other])))
        for seed in SEEDS:
            nested[(corpus, seed)] = accs[k_star][(corpus, seed)]
        logger.info("  nested k* for %s: %s", corpus, k_star)
    arr = np.array([nested[c] for c in cells])
    logger.info(
        "NESTED shrinkage: mean=%.4f | vs local %+.4f (%dW/%dL) | vs global %+.4f (%dW/%dL)",
        float(arr.mean()),
        float((arr - local_acc).mean()),
        int(np.sum(arr > local_acc + 1e-12)),
        int(np.sum(arr < local_acc - 1e-12)),
        float((arr - global_acc).mean()),
        int(np.sum(arr > global_acc + 1e-12)),
        int(np.sum(arr < global_acc - 1e-12)),
    )


if __name__ == "__main__":
    main()
