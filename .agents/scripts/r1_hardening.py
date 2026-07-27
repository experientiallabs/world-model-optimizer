"""Promotion hardening for the r1 champion: adaptive rag, drift floor curve, CRC under drift.

Master's 2026-07-25 assignment, three items, shared kill bar (the ours9 +1pt paired win must
HOLD while the failure is fixed):

- `adaptive`: replace the hand-set rag8 with a fit-side rule
  rag = min(50, max(4, ceil(bank/2))), min_pairs = min(8, max(3, rag//2)). On banks >= 100
  this is bit-identical to the champion (rag 50 / min_pairs 8), so ours9 holds by
  construction and is verified empirically; on 17-item wm banks it lands at rag 9-ish, the
  region the hand-set rag8 probed. Validated on ours9 (kill bar), every wm corpus
  (candidates: n < 30/seed under the power rule), the pooled 10-corpus wm-all cohort
  (63/seed: a real claim), and the gaia2 / swe-bench monopoly controls (must collapse).
- `floor`: the FULL risk-coverage tradeoff of the quantile drift floor (abstain to
  best-single when a query's nearest bank neighbor is below the q-th percentile of bank
  self-NN similarity), q swept on iid + ood-cluster + ood-task ours9.
- `crc`: (a) a serve-time-observable drift statistic (standardized shift of test max-sim vs
  calibration max-sim; needs NO labels) correlated with certificate violations -> a trust
  rule serving can surface; (b) weighted CRC (covariate-shift conformal: density-ratio
  weights from a 2-feature logistic cal-vs-test discriminator, Kish effective n in the CRC
  correction) vs unweighted, realized risk per split.

Dashboard rules honored: fit outputs (floor values, lambda_hat, n_eff, drift stats) go to
notes, never params. OOD splits use `wmo.research.routing_ood` (r2's splitter) so rows pair
with the existing cohort. Runs -> runs/r1.jsonl.

Usage: uv run python .agents/scripts/r1_hardening.py adaptive|floor|crc [--seeds]
"""

from __future__ import annotations

import importlib.util
import logging
import math
import random
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from wmo.optimize.outcomes import OutcomeMatrix
from wmo.research import routing_ood as _ood
from wmo.research.routerbench import best_single_model, oracle, split_scenario_ids
from wmo.research.routing_runs import RunRecord, append_run, evaluate_choices
from wmo.research.routing_corpus import routing_data

_here = Path(__file__)


def _load(name: str, path: Path):  # noqa: ANN202
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_mod = _load("r1_retrieval_ablations", _here.with_name("r1_retrieval_ablations.py"))
MatrixContext = _mod.MatrixContext
RetrievalParams = _mod.RetrievalParams

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("r1.hard")

DATA = routing_data()
RUNS = DATA / "runs" / "r1.jsonl"
SEEDS = [0, 1, 2, 3, 4]
WM_CORPORA = [
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
]
FLOOR_QS = [None, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]
CRC_ALPHA = 0.02
CAL_FRAC = 0.4
LAMBDA_GRID = [round(v, 2) for v in np.arange(-2.0, 6.01, 0.1)]


def adaptive_rag(bank_size: int) -> tuple[int, int]:
    """The fit-side neighborhood rule: (rag_num, min_pairs) from bank size alone."""
    rag = min(50, max(4, math.ceil(bank_size / 2)))
    return rag, min(8, max(3, rag // 2))


def _record(
    *,
    matrix_name: str,
    ctx,  # noqa: ANN001
    split_kind: str,
    seed: int,
    fit_ids: list[str],
    test_ids: list[str],
    best_name: str,
    variant: str,
    params: dict,
    picks: dict[str, str],
    notes_extra: str = "",
) -> dict:
    matrix = ctx.matrix
    best_eval = evaluate_choices(matrix, test_ids, lambda _s, b=best_name: b)
    oracle_acc, oracle_cost = oracle(matrix, test_ids)
    result = evaluate_choices(matrix, test_ids, lambda s, p=picks: p[s])
    append_run(
        RunRecord(
            run_id=f"{matrix_name}-{variant}-{uuid.uuid4().hex[:8]}",
            ts=datetime.now(tz=UTC).isoformat(),
            matrix=matrix_name,
            variant=variant,
            params={**params, "split": split_kind},
            split_seed=seed,
            fit_scenarios=len(fit_ids),
            test_scenarios=len(test_ids),
            result=result,
            baselines={"best_single": best_eval},
            notes=(
                f"HARDEN split={split_kind}; best_single={best_name}; oracle "
                f"acc={oracle_acc:.4f} cost=${oracle_cost:.5f}; embedder=openai{notes_extra}"
            ),
        ),
        RUNS,
    )
    dacc = result.accuracy - best_eval.accuracy
    logger.info(
        "%s/%s %s seed%d: acc=%.4f dacc=%+.4f cost %+.1f%%%s",
        matrix_name,
        variant,
        split_kind,
        seed,
        result.accuracy,
        dacc,
        (result.cost_per_call / best_eval.cost_per_call - 1) * 100,
        notes_extra,
    )
    return {"dacc": dacc, "acc": result.accuracy}


def _splits(matrix: OutcomeMatrix, kind: str, seed: int) -> tuple[list[str], list[str]]:
    from wmo.optimize.policy import EmbedderSpec

    if kind == "iid":
        return split_scenario_ids(matrix, train_fraction=0.7, seed=seed)
    if kind == "ood-cluster":
        return _ood.split_holdout_clusters(
            matrix, embedder=EmbedderSpec(dim=1024), test_fraction=0.3, seed=seed
        )
    if kind == "ood-task":
        return _ood.split_holdout_tasks(matrix, test_fraction=0.3, seed=seed)
    raise ValueError(kind)


def _wm_matrix(corpus: str) -> OutcomeMatrix:
    return OutcomeMatrix.load(DATA / "matrices" / f"{corpus}_matrix.json")


def _wm_all() -> OutcomeMatrix:
    pooled = []
    pool = None
    for corpus in WM_CORPORA:
        matrix = _wm_matrix(corpus)
        pool = matrix.pool
        for outcome in matrix.outcomes:
            pooled.append(
                outcome.model_copy(update={"scenario_id": f"{corpus}:{outcome.scenario_id}"})
            )
    assert pool is not None
    return OutcomeMatrix(pool=pool, outcomes=pooled)


def cmd_adaptive2(seeds: list[int]) -> None:
    """H1 fix: adaptive rag + binomial SE floor (blocks lucky small-bank acceptances)."""
    jobs: list[tuple[str, OutcomeMatrix]] = [
        (
            "routerbench-ours9",
            OutcomeMatrix.load(DATA / "matrices" / "routerbench-ours9_matrix.json"),
        ),
        ("wm-all", _wm_all()),
    ]
    jobs += [(f"wm-{corpus}", _wm_matrix(corpus)) for corpus in WM_CORPORA]
    for name, matrix in jobs:
        ctx = MatrixContext(matrix, name, embed="openai", embed_replies=False)
        for seed in seeds:
            fit_ids, test_ids = _splits(matrix, "iid", seed)
            best_name, _, _ = best_single_model(matrix, fit_ids=fit_ids, eval_ids=test_ids)
            rag, min_pairs = adaptive_rag(len(fit_ids))
            params = RetrievalParams(
                second_route=False,
                guard="stat",
                z=0.5,
                rag_num=rag,
                min_pairs=min_pairs,
                se_floor=True,
            )
            picks = _mod.route(ctx, params, fit_ids, test_ids, best_name)
            _record(
                matrix_name=name,
                ctx=ctx,
                split_kind="iid",
                seed=seed,
                fit_ids=fit_ids,
                test_ids=test_ids,
                best_name=best_name,
                variant="r1-knn-adapt3-oai" if "--v3" in sys.argv else "r1-knn-adapt2-oai",
                params={
                    "guard": "stat",
                    "z": 0.5,
                    "rule": "rag=min(50,ceil(bank/2))",
                    "se_floor": True,
                },
                picks=picks,
                notes_extra=f"; rag={rag} min_pairs={min_pairs} bank={len(fit_ids)}",
            )


def cmd_crc_floor(seeds: list[int]) -> None:
    """H3 fix candidate: CRC accept AND quantile floor (q=0.05); violations re-measured."""
    matrix = OutcomeMatrix.load(DATA / "matrices" / "routerbench-ours9_matrix.json")
    ctx = MatrixContext(matrix, "routerbench-ours9", embed="openai", embed_replies=False)
    for kind in ("iid", "ood-cluster", "ood-task"):
        for seed in seeds:
            fit_ids, test_ids = _splits(matrix, kind, seed)
            best_name, _, _ = best_single_model(matrix, fit_ids=fit_ids, eval_ids=test_ids)
            rng = random.Random(7 * seed + 13)
            shuffled = fit_ids[:]
            rng.shuffle(shuffled)
            n_cal = max(1, int(len(shuffled) * CAL_FRAC))
            cal_ids, bank_ids = shuffled[:n_cal], shuffled[n_cal:]
            params = RetrievalParams(second_route=False, guard="none")
            cal_ev: dict[str, dict] = {}
            _mod.route(ctx, params, bank_ids, cal_ids, best_name, evidence=cal_ev)
            test_ev: dict[str, dict] = {}
            test_raw = _mod.route(ctx, params, bank_ids, test_ids, best_name, evidence=test_ev)
            bank = np.stack([ctx.task_vecs[sid] for sid in bank_ids])
            self_nn = bank @ bank.T
            np.fill_diagonal(self_nn, -1.0)
            floor = float(np.quantile(self_nn.max(axis=1), 0.05))
            cal_max = {s: float(np.max(bank @ ctx.task_vecs[s])) for s in cal_ids}
            test_max = {s: float(np.max(bank @ ctx.task_vecs[s])) for s in test_ids}

            def regret(sid: str, pick: str, best: str = best_name) -> float | None:
                cp = ctx.rewards_cell.get((sid, pick))
                cb = ctx.rewards_cell.get((sid, best))
                if not cp or not cb:
                    return None
                return max(0.0, sum(cb) / len(cb) - sum(cp) / len(cp))

            def accepted(
                ev: dict,
                lam: float | None,
                sim: float,
                best: str = best_name,
                floor_value: float = floor,
            ) -> bool:
                return (
                    ev["pick"] != best
                    and lam is not None
                    and sim >= floor_value
                    and ev["mean_d"] > lam * ev["se"]
                )

            # The floor is part of the PREDICTOR now, so calibration losses see it too:
            # the certificate is over the floored accept rule, not floored post hoc.
            lam_hat = None
            for lam in LAMBDA_GRID:
                losses = [
                    (regret(s, cal_ev[s]["pick"]) or 0.0)
                    if accepted(cal_ev[s], lam, cal_max[s])
                    else 0.0
                    for s in cal_ids
                ]
                n = len(losses)
                if n and (n / (n + 1)) * float(np.mean(losses)) + 1.0 / (n + 1) <= CRC_ALPHA:
                    lam_hat = lam
                    break
            picks = {
                s: test_raw[s] if accepted(test_ev[s], lam_hat, test_max[s]) else best_name
                for s in test_ids
            }
            realized = [
                r
                for s in test_ids
                if picks[s] != best_name and (r := regret(s, picks[s])) is not None
            ]
            risk = float(np.sum(realized)) / max(1, len(test_ids))
            _record(
                matrix_name="routerbench-ours9",
                ctx=ctx,
                split_kind=kind,
                seed=seed,
                fit_ids=fit_ids,
                test_ids=test_ids,
                best_name=best_name,
                variant="r1-knn-crcfloor-a02-oai",
                params={"alpha": CRC_ALPHA, "floor_q": 0.05},
                picks=picks,
                notes_extra=(
                    f"; lambda_hat={lam_hat} floor={floor:.3f} realized_test_risk={risk:.4f}"
                ),
            )


def cmd_adaptive(seeds: list[int]) -> None:
    """H1: the adaptive neighborhood rule across ours9 + wm corpora + wm-all + monopolies.

    `--matrices a,b` restricts to explicit cohorts by file stem (s80 reruns: each is its own
    cohort, matrix name == file stem, never merged with 25-scenario rows).
    """
    if "--matrices" in sys.argv:
        stems = sys.argv[sys.argv.index("--matrices") + 1].split(",")
        jobs: list[tuple[str, OutcomeMatrix]] = [(stem, _wm_matrix(stem)) for stem in stems]
    else:
        jobs = [
            (
                "routerbench-ours9",
                OutcomeMatrix.load(DATA / "matrices" / "routerbench-ours9_matrix.json"),
            ),
            ("wm-all", _wm_all()),
        ]
        jobs += [(f"wm-{corpus}", _wm_matrix(corpus)) for corpus in WM_CORPORA]
    for name, matrix in jobs:
        ctx = MatrixContext(matrix, name, embed="openai", embed_replies=False)
        for seed in seeds:
            fit_ids, test_ids = _splits(matrix, "iid", seed)
            best_name, _, _ = best_single_model(matrix, fit_ids=fit_ids, eval_ids=test_ids)
            rag, min_pairs = adaptive_rag(len(fit_ids))
            params = RetrievalParams(
                second_route=False, guard="stat", z=0.5, rag_num=rag, min_pairs=min_pairs
            )
            picks = _mod.route(ctx, params, fit_ids, test_ids, best_name)
            _record(
                matrix_name=name,
                ctx=ctx,
                split_kind="iid",
                seed=seed,
                fit_ids=fit_ids,
                test_ids=test_ids,
                best_name=best_name,
                variant="r1-knn-adapt-oai",
                params={"guard": "stat", "z": 0.5, "rule": "rag=min(50,ceil(bank/2))"},
                picks=picks,
                notes_extra=f"; rag={rag} min_pairs={min_pairs} bank={len(fit_ids)}",
            )


def cmd_floor(seeds: list[int]) -> None:
    """H2: quantile-floor risk-coverage curves on ours9, iid + both OOD splits."""
    matrix = OutcomeMatrix.load(DATA / "matrices" / "routerbench-ours9_matrix.json")
    ctx = MatrixContext(matrix, "routerbench-ours9", embed="openai", embed_replies=False)
    for kind in ("iid", "ood-cluster", "ood-task"):
        for seed in seeds:
            fit_ids, test_ids = _splits(matrix, kind, seed)
            best_name, _, _ = best_single_model(matrix, fit_ids=fit_ids, eval_ids=test_ids)
            bank = np.stack([ctx.task_vecs[sid] for sid in fit_ids])
            self_nn = bank @ bank.T
            np.fill_diagonal(self_nn, -1.0)
            self_nn = self_nn.max(axis=1)
            test_max = {sid: float(np.max(bank @ ctx.task_vecs[sid])) for sid in test_ids}
            for q in FLOOR_QS:
                floor = float(np.quantile(self_nn, q)) if q is not None else None
                params = RetrievalParams(
                    second_route=False, guard="stat", z=0.5, distance_floor=floor
                )
                picks = _mod.route(ctx, params, fit_ids, test_ids, best_name)
                coverage = (
                    1.0
                    if floor is None
                    else sum(1 for s in test_ids if test_max[s] >= floor) / len(test_ids)
                )
                _record(
                    matrix_name="routerbench-ours9",
                    ctx=ctx,
                    split_kind=kind,
                    seed=seed,
                    fit_ids=fit_ids,
                    test_ids=test_ids,
                    best_name=best_name,
                    variant=f"r1-knn-adapt-floor-q{q if q is not None else 0}",
                    params={"guard": "stat", "z": 0.5, "floor_q": q or 0},
                    picks=picks,
                    notes_extra=f"; floor={floor if floor is not None else 'none'} "
                    f"coverage={coverage:.3f}",
                )


def _density_weights(cal_vecs: np.ndarray, test_vecs: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """Covariate-shift weights for calibration items: w = p_test(x)/p_cal(x), clipped.

    Features are 2-d and serve-time observable (max sim + mean top-5 sim to the bank), so
    the logistic discriminator cannot overfit raw 3072-dim embeddings.
    """
    from sklearn.linear_model import LogisticRegression

    def feats(vecs: np.ndarray) -> np.ndarray:
        sims = vecs @ bank.T
        top5 = np.sort(sims, axis=1)[:, -5:]
        return np.stack([sims.max(axis=1), top5.mean(axis=1)], axis=1)

    x = np.vstack([feats(cal_vecs), feats(test_vecs)])
    y = np.concatenate([np.zeros(len(cal_vecs)), np.ones(len(test_vecs))])
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(x, y)
    p_test = clf.predict_proba(feats(cal_vecs))[:, 1]
    ratio = (p_test / np.clip(1 - p_test, 1e-6, None)) * (len(cal_vecs) / len(test_vecs))
    return np.clip(ratio, 0.1, 10.0)


def cmd_crc(seeds: list[int]) -> None:
    """H3: drift statistic + weighted CRC vs unweighted, realized risk per split."""
    matrix = OutcomeMatrix.load(DATA / "matrices" / "routerbench-ours9_matrix.json")
    ctx = MatrixContext(matrix, "routerbench-ours9", embed="openai", embed_replies=False)
    for kind in ("iid", "ood-cluster", "ood-task"):
        for seed in seeds:
            fit_ids, test_ids = _splits(matrix, kind, seed)
            best_name, _, _ = best_single_model(matrix, fit_ids=fit_ids, eval_ids=test_ids)
            rng = random.Random(7 * seed + 13)
            shuffled = fit_ids[:]
            rng.shuffle(shuffled)
            n_cal = max(1, int(len(shuffled) * CAL_FRAC))
            cal_ids, bank_ids = shuffled[:n_cal], shuffled[n_cal:]
            params = RetrievalParams(second_route=False, guard="none")
            cal_ev: dict[str, dict] = {}
            _mod.route(ctx, params, bank_ids, cal_ids, best_name, evidence=cal_ev)
            test_ev: dict[str, dict] = {}
            test_raw = _mod.route(ctx, params, bank_ids, test_ids, best_name, evidence=test_ev)

            bank = np.stack([ctx.task_vecs[sid] for sid in bank_ids])
            cal_max = np.asarray([float(np.max(bank @ ctx.task_vecs[s])) for s in cal_ids])
            test_max = np.asarray([float(np.max(bank @ ctx.task_vecs[s])) for s in test_ids])
            # Serve-time drift statistic: standardized downward shift of test similarity.
            drift = float((cal_max.mean() - test_max.mean()) / (cal_max.std() + 1e-9))

            def regret(sid: str, pick: str, best: str = best_name) -> float | None:
                cp = ctx.rewards_cell.get((sid, pick))
                cb = ctx.rewards_cell.get((sid, best))
                if not cp or not cb:
                    return None
                return max(0.0, sum(cb) / len(cb) - sum(cp) / len(cp))

            def accepted(ev: dict, lam: float | None, best: str = best_name) -> bool:
                return ev["pick"] != best and lam is not None and ev["mean_d"] > lam * ev["se"]

            cal_losses = {sid: (regret(sid, ev["pick"]) or 0.0) for sid, ev in cal_ev.items()}
            weights = _density_weights(
                np.stack([ctx.task_vecs[s] for s in cal_ids]),
                np.stack([ctx.task_vecs[s] for s in test_ids]),
                bank,
            )
            w_by_sid = dict(zip(cal_ids, weights, strict=True))

            def crc_lambda(
                weighted: bool,
                *,
                cal_ids: list[str] = cal_ids,
                cal_ev: dict = cal_ev,
                cal_losses: dict = cal_losses,
                w_by_sid: dict = w_by_sid,
                accepted=accepted,  # noqa: ANN001
            ) -> tuple[float | None, float]:
                for lam in LAMBDA_GRID:
                    losses = np.asarray(
                        [cal_losses[sid] if accepted(cal_ev[sid], lam) else 0.0 for sid in cal_ids]
                    )
                    if weighted:
                        w = np.asarray([w_by_sid[sid] for sid in cal_ids])
                        r_hat = float(np.sum(w * losses) / np.sum(w))
                        n_eff = float(np.sum(w) ** 2 / np.sum(w**2))
                    else:
                        r_hat = float(np.mean(losses))
                        n_eff = float(len(losses))
                    if (n_eff / (n_eff + 1)) * r_hat + 1.0 / (n_eff + 1) <= CRC_ALPHA:
                        return lam, n_eff
                return None, 0.0

            for weighted, variant in (
                (False, "r1-knn-crc2-a02-oai"),
                (True, "r1-knn-wcrc-a02-oai"),
            ):
                lam_hat, n_eff = crc_lambda(weighted)
                picks = {
                    s: test_raw[s] if accepted(test_ev[s], lam_hat) else best_name for s in test_ids
                }
                realized = [
                    r
                    for s in test_ids
                    if picks[s] != best_name and (r := regret(s, picks[s])) is not None
                ]
                risk = float(np.sum(realized)) / max(1, len(test_ids))
                _record(
                    matrix_name="routerbench-ours9",
                    ctx=ctx,
                    split_kind=kind,
                    seed=seed,
                    fit_ids=fit_ids,
                    test_ids=test_ids,
                    best_name=best_name,
                    variant=variant,
                    params={"alpha": CRC_ALPHA, "weighted": weighted},
                    picks=picks,
                    notes_extra=(
                        f"; lambda_hat={lam_hat} n_eff={n_eff:.1f} drift_stat={drift:.3f} "
                        f"realized_test_risk={risk:.4f}"
                    ),
                )


def main() -> None:
    seeds = SEEDS if "--seeds" in sys.argv else [0]
    command = next((a for a in sys.argv[1:] if not a.startswith("--")), "adaptive")
    {
        "adaptive": cmd_adaptive,
        "adaptive2": cmd_adaptive2,
        "floor": cmd_floor,
        "crc": cmd_crc,
        "crc-floor": cmd_crc_floor,
    }[command](seeds)
    logger.info("runs -> %s", RUNS)


if __name__ == "__main__":
    main()
