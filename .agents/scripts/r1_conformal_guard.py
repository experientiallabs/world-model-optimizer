"""Conformal guard (CRC / Learn-then-Test) + champion runs on the headroom matrices.

Directive 2 of the 2026-07-24 master drawing-board entry: replace the hand-tuned z threshold
(a multiple-comparisons hazard: z was picked across 7+ variants on the same seeds) with a
finite-sample calibrated one (Conformal Risk Control, Angelopoulos & Bates et al., ICLR;
Learn-then-Test, Ann. Appl. Stat.).

Construction. The router's guard is a 1-parameter accept rule on pre-guard evidence:
accept the routed pick iff mean_d > lambda * se (mean_d, se = paired per-neighbor reward
delta vs baseline and its standard error). Risk of a decision = clipped regret vs baseline,
loss(lambda) = max(0, r_base - r_pick) * 1{accept}, bounded in [0, 1] and MONOTONE
non-increasing in lambda. CRC therefore applies: hold out a calibration slice of the FIT side
(the retrieval bank shrinks to the rest; test stays untouched), compute R_hat(lambda) on
calibration, and pick

    lambda_hat = min{ lambda : (n/(n+1)) R_hat(lambda) + 1/(n+1) <= alpha }

which guarantees E[test risk] <= alpha for exchangeable calibration/test items. When no
lambda certifies (n too small: 1/(n+1) > alpha), the guard degrades to reject-all =
best-single: the finite-sample answer for tiny corpora, no hand-tuning involved.

Also runs directive 1: the round-2 champion (kNN + relative threshold + oai3l + statz05) on
financebench / tau-bench / continual-learning (the +0.23..0.39 headroom matrices),
paired-by-seed, plus locality-adjusted (rag 8) and unguarded references, and a pooled
3-corpus matrix (headroom3-all) whose larger calibration side gives CRC something to certify.

Usage: uv run python .agents/scripts/r1_conformal_guard.py [--seeds]
All records -> runs/r1.jsonl (variants r1-knn-crc-*, plus champion reruns).
"""

from __future__ import annotations

import importlib.util
import logging
import random
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from wmo.optimize.outcomes import OutcomeMatrix
from wmo.research.routerbench import best_single_model, oracle, split_scenario_ids
from wmo.research.routing_runs import RunRecord, append_run, evaluate_choices
from wmo.research.routing_corpus import routing_data

_spec = importlib.util.spec_from_file_location(
    "r1_retrieval_ablations", Path(__file__).with_name("r1_retrieval_ablations.py")
)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MatrixContext = _mod.MatrixContext
RetrievalParams = _mod.RetrievalParams

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("r1.crc")

DATA = routing_data()
RUNS = DATA / "runs" / "r1.jsonl"

LAMBDA_GRID = [round(v, 2) for v in np.arange(-2.0, 6.01, 0.1)]
CAL_FRAC = 0.4
HEADROOM = ["financebench", "tau-bench", "continual-learning"]


def cell_mean(ctx, sid: str, model: str) -> float | None:  # noqa: ANN001
    cell = ctx.rewards_cell.get((sid, model))
    return sum(cell) / len(cell) if cell else None


def clipped_regret(ctx, sid: str, pick: str, best: str) -> float | None:  # noqa: ANN001
    """max(0, r_best - r_pick) for one item; None when either cell is unscored."""
    r_pick, r_best = cell_mean(ctx, sid, pick), cell_mean(ctx, sid, best)
    if r_pick is None or r_best is None:
        return None
    return max(0.0, r_best - r_pick)


def crc_lambda(losses_by_lambda: dict[float, list[float]], alpha: float) -> float | None:
    """Smallest grid lambda whose CRC-adjusted risk clears alpha; None = reject-all."""
    for lam in LAMBDA_GRID:
        losses = losses_by_lambda[lam]
        n = len(losses)
        if n == 0:
            continue
        if (n / (n + 1)) * float(np.mean(losses)) + 1.0 / (n + 1) <= alpha:
            return lam
    return None


def accepts(ev: dict, lam: float | None, best: str) -> bool:
    """The guard's accept rule at threshold lam (None = reject-all)."""
    if ev["pick"] == best:
        return False  # nothing to accept; pick already the baseline
    if lam is None:
        return False
    return ev["mean_d"] > lam * ev["se"]


def run_crc(
    name: str,
    ctx,  # noqa: ANN001
    split_seed: int,
    alphas: list[float],
    record,  # noqa: ANN001
    fit_ids: list[str],
    test_ids: list[str],
    best_name: str,
) -> None:
    """Calibrate lambda on a held-out fit slice, evaluate the guarded router on test."""
    rng = random.Random(7 * split_seed + 13)
    shuffled = fit_ids[:]
    rng.shuffle(shuffled)
    n_cal = max(1, int(len(shuffled) * CAL_FRAC))
    cal_ids, bank_ids = shuffled[:n_cal], shuffled[n_cal:]

    params = RetrievalParams(second_route=False, guard="none")
    cal_ev: dict[str, dict] = {}
    _mod.route(ctx, params, bank_ids, cal_ids, best_name, evidence=cal_ev)
    test_ev: dict[str, dict] = {}
    test_raw = _mod.route(ctx, params, bank_ids, test_ids, best_name, evidence=test_ev)

    losses_by_lambda: dict[float, list[float]] = {lam: [] for lam in LAMBDA_GRID}
    for sid, ev in cal_ev.items():
        regret = clipped_regret(ctx, sid, ev["pick"], best_name)
        if regret is None:
            continue
        for lam in LAMBDA_GRID:
            losses_by_lambda[lam].append(regret if accepts(ev, lam, best_name) else 0.0)

    for alpha in alphas:
        lam_hat = crc_lambda(losses_by_lambda, alpha)
        picks = {
            sid: (test_raw[sid] if accepts(test_ev[sid], lam_hat, best_name) else best_name)
            for sid in test_ids
        }
        accepted = [s for s in test_ids if picks[s] != best_name]
        realized = [
            r for s in accepted if (r := clipped_regret(ctx, s, picks[s], best_name)) is not None
        ]
        risk = float(np.sum(realized)) / max(1, len(test_ids))
        tag = f"{alpha:.2f}".replace("0.", "")
        record(
            f"r1-knn-crc-a{tag}-oai",
            {
                "alpha": alpha,
                "lambda_hat": lam_hat if lam_hat is not None else "reject-all",
                "cal_n": len(cal_ids),
                "bank_n": len(bank_ids),
                "accept_rate": len(accepted) / len(test_ids),
            },
            evaluate_choices(ctx.matrix, test_ids, lambda s, p=picks: p[s]),
            notes_extra=(
                f"; CRC guard: lambda_hat={lam_hat} alpha={alpha} realized_test_risk="
                f"{risk:.4f} (clipped regret per test item)"
            ),
        )
        logger.info(
            "%s seed%d crc a=%.2f: lambda_hat=%s accept=%.2f realized_risk=%.4f",
            name,
            split_seed,
            alpha,
            lam_hat,
            len(accepted) / len(test_ids),
            risk,
        )


def run_matrix(name: str, ctx, split_seed: int, alphas: list[float], tiny: bool) -> None:  # noqa: ANN001
    matrix = ctx.matrix
    fit_ids, test_ids = split_scenario_ids(matrix, train_fraction=0.7, seed=split_seed)
    best_name, _, _ = best_single_model(matrix, fit_ids=fit_ids, eval_ids=test_ids)
    best_eval = evaluate_choices(matrix, test_ids, lambda _sid: best_name)
    oracle_acc, oracle_cost = oracle(matrix, test_ids)
    ts = datetime.now(tz=UTC).isoformat()

    def record(variant: str, params: dict, result, notes_extra: str = "") -> None:  # noqa: ANN001
        append_run(
            RunRecord(
                run_id=f"{name}-{variant}-{uuid.uuid4().hex[:8]}",
                ts=ts,
                matrix=name,
                variant=variant,
                params=params,
                split_seed=split_seed,
                fit_scenarios=len(fit_ids),
                test_scenarios=len(test_ids),
                result=result,
                baselines={"best_single": best_eval},
                notes=(
                    f"best_single={best_name}; oracle acc={oracle_acc:.4f} "
                    f"cost=${oracle_cost:.5f}; embedder={ctx.embed_kind}{notes_extra}"
                ),
            ),
            RUNS,
        )
        logger.info(
            "%s/%s seed%d: acc=%.4f (best %.4f) cost %+.1f%%",
            name,
            variant,
            split_seed,
            result.accuracy,
            best_eval.accuracy,
            (result.cost_per_call / best_eval.cost_per_call - 1) * 100,
        )

    champion = [
        ("r1-knn-statz05-oai", RetrievalParams(second_route=False, guard="stat", z=0.5)),
        ("r1-knn-noguard-oai", RetrievalParams(second_route=False, guard="none")),
        ("r1-knn-asym-oai", RetrievalParams(second_route=False, guard="stat_asym", z=0.5)),
    ]
    if tiny:
        # Locality-adjusted rag for 17-item fit banks (rag 50 makes every fit item a
        # neighbor, so the profile is global and the guard reverts everything: round 1 pt 6).
        champion += [
            (
                "r1-knn-rag8-statz05-oai",
                RetrievalParams(second_route=False, guard="stat", z=0.5, rag_num=8, min_pairs=4),
            ),
            (
                "r1-knn-rag8-noguard-oai",
                RetrievalParams(second_route=False, guard="none", rag_num=8),
            ),
        ]
    for variant, params in champion:
        picks = _mod.route(ctx, params, fit_ids, test_ids, best_name)
        realized = [
            r
            for s in test_ids
            if picks[s] != best_name
            and (r := clipped_regret(ctx, s, picks[s], best_name)) is not None
        ]
        risk = float(np.sum(realized)) / max(1, len(test_ids))
        record(
            variant,
            params.model_dump(),
            evaluate_choices(matrix, test_ids, lambda s, p=picks: p[s]),
            notes_extra=f"; realized_test_risk={risk:.4f} (clipped regret per test item)",
        )

    run_crc(name, ctx, split_seed, alphas, record, fit_ids, test_ids, best_name)


def main() -> None:
    seeds = _mod.SPLIT_SEEDS if "--seeds" in sys.argv else [0]

    jobs: list[tuple[str, OutcomeMatrix, list[float], bool]] = []
    for corpus in HEADROOM:
        matrix = OutcomeMatrix.load(DATA / "matrices" / f"{corpus}_matrix.json")
        jobs.append((f"wm-{corpus}", matrix, [0.05, 0.15], True))
    pooled = []
    for corpus in HEADROOM:
        matrix = OutcomeMatrix.load(DATA / "matrices" / f"{corpus}_matrix.json")
        for outcome in matrix.outcomes:
            pooled.append(
                outcome.model_copy(update={"scenario_id": f"{corpus}:{outcome.scenario_id}"})
            )
    jobs.append(
        (
            "headroom3-all",
            OutcomeMatrix(pool=matrix.pool, outcomes=pooled),
            [0.05, 0.10],
            False,
        )
    )
    jobs.append(
        (
            "routerbench-ours9",
            OutcomeMatrix.load(DATA / "matrices" / "routerbench-ours9_matrix.json"),
            [0.01, 0.02, 0.05],
            False,
        )
    )

    for name, matrix, alphas, tiny in jobs:
        ctx = MatrixContext(matrix, name, embed="openai", embed_replies=False)
        for seed in seeds:
            run_matrix(name, ctx, seed, alphas, tiny)
    logger.info("runs -> %s", RUNS)


if __name__ == "__main__":
    main()
