"""Head-to-head completion: r1 champion under OOD + embedder-equalized rank baseline.

Fills the missing cells of the cross-family comparison (is retrieval actually better?):

1. The r1 champion family has NO out-of-distribution rows; its nearest kin (r2-knn-prox-z05)
   goes NEGATIVE on ood-task. Runs r1-knn-statz05-oai (+ quantile distance floor, + asym,
   + noguard, + CRC alpha=0.02) on r2's exact OOD splits (`wmo.research.routing_ood`;
   hashing-1024 spec, test_fraction 0.3, kmeans_seed 1234) so every row pairs with r2's
   by seed.
2. The rank router never got the semantic embedder that lifted r1's family. Runs rank-oai
   (canonical fit_rank_policy + rank_decision on cached text-embedding-3-large vectors) on
   iid AND both OOD splits: if rank-oai matches knn-oai, the ours9 "win" was the embedder.

The quantile floor is fit-side only (no test peeking): abstain to best-single when the
query's nearest bank neighbor is below the 5th percentile of bank self-NN similarity.

Usage: uv run python .agents/scripts/r1_head_to_head.py [--seeds]
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
from pydantic import PrivateAttr

from wmo.optimize.outcomes import OutcomeMatrix
from wmo.optimize.policy import EmbedderSpec, rank_decision
from wmo.optimize.routing import fit_rank_policy
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
logger = logging.getLogger("r1.h2h")

DATA = routing_data()
RUNS = DATA / "runs" / "r1.jsonl"
CRC_ALPHA = 0.02
CAL_FRAC = 0.4
LAMBDA_GRID = [round(v, 2) for v in np.arange(-2.0, 6.01, 0.1)]


class _LookupEmbedder:
    """Embedder protocol over a text -> precomputed-vector map (cached oai3l)."""

    def __init__(self, mapping: dict[str, np.ndarray]) -> None:
        self._mapping = mapping

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, self._mapping[text])) for text in texts]


class LookupSpec(EmbedderSpec):
    """EmbedderSpec whose build() serves cached vectors; for embedder-equalized rank."""

    _lookup: dict = PrivateAttr(default=None)

    def build(self):  # noqa: ANN201
        return _LookupEmbedder(self._lookup)


def _splits(matrix: OutcomeMatrix, kind: str, seed: int) -> tuple[list[str], list[str]]:
    spec = EmbedderSpec(dim=1024)  # r2's exact OOD-split geometry (hashing)
    if kind == "iid":
        return split_scenario_ids(matrix, train_fraction=0.7, seed=seed)
    if kind == "ood-cluster":
        return _ood.split_holdout_clusters(matrix, embedder=spec, test_fraction=0.3, seed=seed)
    if kind == "ood-task":
        return _ood.split_holdout_tasks(matrix, test_fraction=0.3, seed=seed)
    raise ValueError(kind)


def quantile_floor(ctx, fit_ids: list[str], quantile: float = 0.05) -> float:  # noqa: ANN001
    """5th percentile of bank self-NN cosine sim (fit-side geometry only)."""
    bank = np.stack([ctx.task_vecs[sid] for sid in fit_ids])
    sims = bank @ bank.T
    np.fill_diagonal(sims, -1.0)
    return float(np.quantile(sims.max(axis=1), quantile))


def crc_pick(ctx, fit_ids, test_ids, best_name, seed):  # noqa: ANN001, ANN201
    """CRC-guarded picks (alpha=0.02) + realized risk; same construction as round 4."""
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

    def regret(sid: str, pick: str) -> float | None:
        cp, cb = ctx.rewards_cell.get((sid, pick)), ctx.rewards_cell.get((sid, best_name))
        if not cp or not cb:
            return None
        return max(0.0, sum(cb) / len(cb) - sum(cp) / len(cp))

    def accepted(ev: dict, lam: float | None) -> bool:
        return ev["pick"] != best_name and lam is not None and ev["mean_d"] > lam * ev["se"]

    lam_hat = None
    for lam in LAMBDA_GRID:
        losses = [
            (regret(sid, ev["pick"]) or 0.0) if accepted(ev, lam) else 0.0
            for sid, ev in cal_ev.items()
        ]
        n = len(losses)
        if n and (n / (n + 1)) * float(np.mean(losses)) + 1.0 / (n + 1) <= CRC_ALPHA:
            lam_hat = lam
            break
    picks = {
        sid: test_raw[sid] if accepted(test_ev[sid], lam_hat) else best_name for sid in test_ids
    }
    realized = [
        r for s in test_ids if picks[s] != best_name and (r := regret(s, picks[s])) is not None
    ]
    risk = float(np.sum(realized)) / max(1, len(test_ids))
    return picks, lam_hat, risk


def main() -> None:
    seeds = _mod.SPLIT_SEEDS if "--seeds" in sys.argv else [0]
    matrix = OutcomeMatrix.load(DATA / "matrices" / "routerbench-ours9_matrix.json")
    ctx = MatrixContext(matrix, "routerbench-ours9", embed="openai", embed_replies=False)
    text_to_vec = {ctx.tasks[sid]: ctx.task_vecs[sid] for sid in ctx.scenario_ids}
    oai_spec = LookupSpec(kind="hashing", dim=3072)
    oai_spec._lookup = text_to_vec  # noqa: SLF001

    for kind in ("iid", "ood-cluster", "ood-task"):
        for seed in seeds:
            fit_ids, test_ids = _splits(matrix, kind, seed)
            best_name, _, _ = best_single_model(matrix, fit_ids=fit_ids, eval_ids=test_ids)
            best_eval = evaluate_choices(matrix, test_ids, lambda _s, b=best_name: b)
            oracle_acc, oracle_cost = oracle(matrix, test_ids)
            ts = datetime.now(tz=UTC).isoformat()

            def record(  # noqa: PLR0913
                variant: str,
                params: dict,
                result,  # noqa: ANN001
                extra: str = "",
                *,
                kind: str = kind,
                seed: int = seed,
                fit_ids: list[str] = fit_ids,
                test_ids: list[str] = test_ids,
                best_name: str = best_name,
                best_eval=best_eval,  # noqa: ANN001
                oracle_acc: float = oracle_acc,
                oracle_cost: float = oracle_cost,
                ts: str = ts,
            ) -> None:
                append_run(
                    RunRecord(
                        run_id=f"ours9-{variant}-{uuid.uuid4().hex[:8]}",
                        ts=ts,
                        matrix="routerbench-ours9",
                        variant=variant,
                        params={**params, "split": kind},
                        split_seed=seed,
                        fit_scenarios=len(fit_ids),
                        test_scenarios=len(test_ids),
                        result=result,
                        baselines={"best_single": best_eval},
                        notes=(
                            f"H2H split={kind}; best_single={best_name}; oracle "
                            f"acc={oracle_acc:.4f} cost=${oracle_cost:.5f}; embedder=openai"
                            f"{extra}"
                        ),
                    ),
                    RUNS,
                )
                logger.info(
                    "%s/%s seed%d: acc=%.4f (best %.4f) cost %+.1f%%%s",
                    kind,
                    variant,
                    seed,
                    result.accuracy,
                    best_eval.accuracy,
                    (result.cost_per_call / best_eval.cost_per_call - 1) * 100,
                    extra,
                )

            # Embedder-equalized rank (iid + ood): the fairness cell for the incumbent.
            policy = fit_rank_policy(
                matrix,
                fit_ids=fit_ids,
                embedder=oai_spec,
                n_clusters=64,
                seed=42,
                guard_model=best_name,
                min_support=4,
                guard_margin=0.03,
                fitted_from=f"ours9 {kind} seed{seed} oai-equalized",
            )
            decisions = {sid: rank_decision(policy, ctx.task_vecs[sid]).model for sid in test_ids}
            record(
                "r1-rank-oai",
                {"k": 64, "guard": "margin"},
                evaluate_choices(matrix, test_ids, lambda s, p=decisions: p[s]),
            )

            if kind == "iid":
                continue  # r1 champion iid rows already exist; only rank-oai was missing

            floor = quantile_floor(ctx, fit_ids)
            variants = [
                ("r1-knn-statz05-oai", RetrievalParams(second_route=False, guard="stat", z=0.5)),
                (
                    "r1-knn-statz05-oai-qfloor",
                    RetrievalParams(second_route=False, guard="stat", z=0.5, distance_floor=floor),
                ),
                ("r1-knn-asym-oai", RetrievalParams(second_route=False, guard="stat_asym", z=0.5)),
                ("r1-knn-noguard-oai", RetrievalParams(second_route=False, guard="none")),
            ]
            for variant, params in variants:
                picks = _mod.route(ctx, params, fit_ids, test_ids, best_name)
                extra = f"; qfloor={floor:.3f}" if "qfloor" in variant else ""
                record(
                    variant,
                    params.model_dump(),
                    evaluate_choices(matrix, test_ids, lambda s, p=picks: p[s]),
                    extra,
                )
            picks, lam_hat, risk = crc_pick(ctx, fit_ids, test_ids, best_name, seed)
            record(
                "r1-knn-crc-a02-oai",
                {
                    "alpha": CRC_ALPHA,
                    "lambda_hat": lam_hat if lam_hat is not None else "reject-all",
                },
                evaluate_choices(matrix, test_ids, lambda s, p=picks: p[s]),
                f"; lambda_hat={lam_hat} realized_test_risk={risk:.4f} (exchangeability "
                "BROKEN by design under ood: certificate stress test)",
            )
    logger.info("runs -> %s", RUNS)


if __name__ == "__main__":
    main()
