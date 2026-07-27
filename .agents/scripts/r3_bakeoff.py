"""Drift-defense bake-off: which default protects the champion under task drift?

Master assignment (2026-07-25): four defense arms, identical splits
(`wmo.research.routing_ood`, r2's splitter, the same code r1's floor rows used - per-seed
baselines pair exactly), 5 seeds, ours9 iid + ood-cluster + ood-task. Arm (a) champion+floor_q
REUSES r1's
r1-knn-adapt-floor-q* rows from runs/r1.jsonl (verified baseline-equal) instead of
recomputing. This script runs the other three:

  (b) r3b-svmz         - RBF-SVM win-vs-baseline proposer + shared z-guard (the sanctioned
                         drift-safe fallback config from the exploration sweep).
  (c) r3b-hybrid-svm   - champion picks, kept only when the SVM's global P_win(pick) > 0.5
                         (global-agreement gate); r3b-hybrid-global - champion picks, kept
                         only when the pick beats the baseline on a whole-bank paired
                         z-test (z=0.5, doubled when pricier): bank-global agreement.
  (d) r3b-absfloor-f*  - champion + ABSOLUTE max-sim floor (abstain when the query's best
                         bank similarity < f), vs (a)'s bank-quantile floor. Sim probe:
                         iid p5=0.512 < ood-task p50=0.570, so no clean separator exists;
                         the grid {0.50,0.55,0.60,0.65} quantifies the overlap tax.

Champion machinery is r1's route() imported unmodified (repro matched to 4 decimals).
Deviation share (1 - baseline share of picks) is the coverage proxy, read from model_mix.
Kill bar: any arm below best-single on an ood split dies. $0 API (cached 3-large vectors).
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import numpy as np

from wmo.optimize.outcomes import OutcomeMatrix
from wmo.optimize.policy import EmbedderSpec
from wmo.research import routing_ood as OOD
from wmo.research.routerbench import best_single_model, oracle, split_scenario_ids
from wmo.research.routing_runs import RunRecord, append_run, evaluate_choices
from wmo.research.routing_corpus import routing_data

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("r3b")

DATA = routing_data()
TRAIN_FRAC = 0.7
RUNS = DATA / "runs/r3.jsonl"
SPLIT_SEEDS = [0, 1, 2, 3, 4]
ABS_FLOORS = [0.50, 0.55, 0.60, 0.65]


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_HERE = Path(__file__).parent
R1 = load_module("r1_retrieval_ablations", _HERE / "r1_retrieval_ablations.py")
R3X = load_module("r3_explore", _HERE / "r3_explore.py")


def make_split(
    kind: str, matrix: OutcomeMatrix, seed: int
) -> tuple[list[str], list[str]]:
    if kind == "iid":
        return split_scenario_ids(matrix, train_fraction=TRAIN_FRAC, seed=seed)
    if kind == "ood-cluster":
        return OOD.split_holdout_clusters(
            matrix, embedder=EmbedderSpec(dim=1024), test_fraction=0.3, seed=seed
        )
    if kind == "ood-task":
        return OOD.split_holdout_tasks(matrix, test_fraction=0.3, seed=seed)
    raise SystemExit(f"unknown split kind {kind}")


def record_run(
    *,
    matrix_name: str,
    matrix: OutcomeMatrix,
    variant: str,
    params: dict,
    seed: int,
    fit_ids: list[str],
    test_ids: list[str],
    picks: dict[str, str],
    best_name: str,
    notes: str = "",
) -> RunRecord:
    best_eval = evaluate_choices(matrix, test_ids, lambda _sid: best_name)
    result = evaluate_choices(matrix, test_ids, lambda sid: picks[sid])
    oracle_acc, _ = oracle(matrix, test_ids)
    rec = RunRecord(
        run_id=f"{matrix_name}-{variant}-{uuid.uuid4().hex[:8]}",
        ts=datetime.now(tz=UTC).isoformat(),
        matrix=matrix_name,
        variant=variant,
        params=params,
        split_seed=seed,
        fit_scenarios=len(fit_ids),
        test_scenarios=len(test_ids),
        result=result,
        baselines={"best_single": best_eval},
        notes=f"best_single={best_name}; oracle acc={oracle_acc:.4f}; {notes}".strip("; "),
    )
    append_run(rec, RUNS)
    logger.info(
        "%s/%s/%s seed%d: acc=%.4f cost=$%.5f (base %.4f) dev=%.2f",
        matrix_name, variant, params.get("split"), seed, result.accuracy,
        result.cost_per_call, best_eval.accuracy,
        1.0 - result.model_mix.get(best_name, 0.0),
    )
    return rec


def svm_pwin(
    ctx,  # noqa: ANN001 - r1's MatrixContext, a dynamically imported type
    fit_ids: list[str],
    test_ids: list[str],
    best_name: str,
    names: list[str],
) -> np.ndarray:
    """Per-model win-vs-baseline probability on the test side (decisive fit cells only)."""
    cell = ctx.rewards_cell
    fit_x = np.stack([ctx.task_vecs[s] for s in fit_ids])
    test_x = np.stack([ctx.task_vecs[s] for s in test_ids])
    p_win = np.zeros((len(test_ids), len(names)))
    for mi, model in enumerate(names):
        if model == best_name:
            continue
        xs, ys = [], []
        for row, sid in enumerate(fit_ids):
            pv, bv = cell.get((sid, model)), cell.get((sid, best_name))
            if pv and bv:
                d = float(np.mean(pv)) - float(np.mean(bv))
                if abs(d) > 1e-9:
                    xs.append(fit_x[row])
                    ys.append(1 if d > 0 else 0)
        if len(ys) < 12:
            continue
        predict = R3X.clf_family("svm", np.stack(xs), np.asarray(ys))
        p_win[:, mi] = predict(test_x)
    return p_win


def z_certify(
    cell: dict, fit_rows: list[str], pick: str, best_name: str, z_need: float
) -> bool:
    """Paired z-test of pick vs baseline over the given fit scenarios (min 8 pairs)."""
    deltas = [
        float(np.mean(cell[(sid, pick)])) - float(np.mean(cell[(sid, best_name)]))
        for sid in fit_rows
        if (sid, pick) in cell and (sid, best_name) in cell
    ]
    if len(deltas) < 8:
        return False
    arr = np.asarray(deltas)
    se = float(arr.std(ddof=1) / np.sqrt(len(arr)))
    return se > 0 and float(arr.mean()) / se >= z_need


def adaptive_champ_params(bank: int, **extra: object) -> object:
    """r1's promoted adapt3 config: adaptive rag + min_pairs + small-sample SE floor."""
    import math

    rag = min(50, max(4, math.ceil(bank / 2)))
    min_pairs = min(8, max(3, rag // 2))
    return R1.RetrievalParams(
        second_route=False, guard="stat", z=0.5, rag_num=rag, min_pairs=min_pairs,
        se_floor=True, **extra,
    )


def run_bakeoff(args: argparse.Namespace) -> None:
    name = args.matrix
    matrix = OutcomeMatrix.load(DATA / "matrices" / f"{name}_matrix.json")
    ctx = R1.MatrixContext(matrix, name, embed="openai", embed_replies=False)
    names = matrix.model_names()
    for kind in args.splits:
        for seed in args.seeds:
            fit_ids, test_ids = make_split(kind, matrix, seed)
            best_name, _, _ = best_single_model(matrix, fit_ids=fit_ids, eval_ids=test_ids)
            mean_cost = ctx.mean_cost(fit_ids)
            base_cost = mean_cost.get(best_name, 0.0)
            cell = ctx.rewards_cell
            fit_m = np.stack([ctx.task_vecs[s] for s in fit_ids])
            k = min(50, len(fit_ids))

            # ---- (b) svm + z-guard -----------------------------------------
            p_win = svm_pwin(ctx, fit_ids, test_ids, best_name, names)
            arms = set(args.arms)
            picks_svm: dict[str, str] = {}
            for t, sid in enumerate(test_ids):
                mi = int(np.argmax(p_win[t]))
                pick = names[mi]
                if p_win[t, mi] <= 0.5 or pick == best_name:
                    picks_svm[sid] = best_name
                    continue
                sims = fit_m @ ctx.task_vecs[sid]
                kth = np.sort(sims)[-k]
                nbr = [fit_ids[int(j)] for j in np.where(sims > 0.95 * kth)[0]]
                z_need = 1.0 if mean_cost.get(pick, 0.0) > base_cost else 0.5
                picks_svm[sid] = (
                    pick if z_certify(cell, nbr, pick, best_name, z_need) else best_name
                )
            if "svmz" in arms:
                record_run(
                    matrix_name=name, matrix=matrix, variant="r3b-svmz",
                    params={"guard": "statz0.5", "embed": "oai3l", "split": kind},
                    seed=seed, fit_ids=fit_ids, test_ids=test_ids, picks=picks_svm,
                    best_name=best_name,
                )

            # ---- champion picks once (promoted adapt3 config), then the arms ----
            champ = R1.route(
                ctx, adaptive_champ_params(len(fit_ids)), fit_ids, test_ids, best_name
            )

            picks_hsvm = {}
            for t, sid in enumerate(test_ids):
                pick = champ[sid]
                if pick != best_name and p_win[t, names.index(pick)] <= 0.5:
                    pick = best_name
                picks_hsvm[sid] = pick
            if "hybrid" in arms:
                record_run(
                    matrix_name=name, matrix=matrix, variant="r3b-hybrid-svm",
                    params={"gate": "svm_pwin>0.5", "embed": "oai3l", "split": kind},
                    seed=seed, fit_ids=fit_ids, test_ids=test_ids, picks=picks_hsvm,
                    best_name=best_name,
                )

            global_ok = {
                m: z_certify(
                    cell, fit_ids, m, best_name,
                    1.0 if mean_cost.get(m, 0.0) > base_cost else 0.5,
                )
                for m in names
                if m != best_name
            }
            picks_hglob = {
                sid: (pick if pick == best_name or global_ok.get(pick, False) else best_name)
                for sid, pick in champ.items()
            }
            if "hybrid" in arms:
                record_run(
                    matrix_name=name, matrix=matrix, variant="r3b-hybrid-global",
                    params={"gate": "bank_z0.5", "embed": "oai3l", "split": kind},
                    seed=seed, fit_ids=fit_ids, test_ids=test_ids, picks=picks_hglob,
                    best_name=best_name,
                )

            # ---- bank self-NN sims (shared by floorq + layered arms) ---------
            self_nn = []
            for row in range(len(fit_ids)):
                sims = fit_m @ fit_m[row]
                sims[row] = -1.0
                self_nn.append(float(np.max(sims)))

            def floor_picks(
                base_picks: dict[str, str], q: float,
                self_nn: list[float] = self_nn, test_ids: list[str] = test_ids,
                best_name: str = best_name, fit_m: np.ndarray = fit_m,
            ) -> dict[str, str]:
                if q <= 0:
                    return dict(base_picks)
                thresh = float(np.quantile(self_nn, q))
                out = {}
                for sid in test_ids:
                    pick = base_picks[sid]
                    if pick != best_name and float(
                        np.max(fit_m @ ctx.task_vecs[sid])
                    ) < thresh:
                        pick = best_name
                    out[sid] = pick
                return out

            for q in [0.0, 0.05, 0.5] if "floorq" in arms else []:
                record_run(
                    matrix_name=name, matrix=matrix, variant=f"r3b-floorq{q}",
                    params={"floor_q": q, "guard": "stat_adapt3", "z": 0.5,
                            "embed": "oai3l", "split": kind},
                    seed=seed, fit_ids=fit_ids, test_ids=test_ids,
                    picks=floor_picks(champ, q), best_name=best_name,
                )
            for q in [0.05, 0.5] if "layered" in arms else []:
                picks_l = floor_picks(picks_hsvm, q)
                record_run(
                    matrix_name=name, matrix=matrix, variant=f"r3b-hybrid-svm-q{q}",
                    params={"gate": "svm_pwin>0.5", "floor_q": q, "embed": "oai3l",
                            "split": kind},
                    seed=seed, fit_ids=fit_ids, test_ids=test_ids, picks=picks_l,
                    best_name=best_name,
                )

            # ---- (d) champion + absolute max-sim floor -----------------------
            for floor in ABS_FLOORS if "absfloor" in arms else []:
                params = R1.RetrievalParams(
                    second_route=False, guard="stat", z=0.5, distance_floor=floor
                )
                picks_f = R1.route(ctx, params, fit_ids, test_ids, best_name)
                record_run(
                    matrix_name=name, matrix=matrix, variant=f"r3b-absfloor-f{floor}",
                    params={"floor_abs": floor, "guard": "stat", "z": 0.5,
                            "embed": "oai3l", "split": kind},
                    seed=seed, fit_ids=fit_ids, test_ids=test_ids, picks=picks_f,
                    best_name=best_name,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="*", default=["iid", "ood-cluster", "ood-task"])
    parser.add_argument("--seeds", nargs="*", type=int, default=SPLIT_SEEDS)
    parser.add_argument(
        "--arms", nargs="*", default=["svmz", "hybrid", "layered", "absfloor"]
    )
    parser.add_argument("--matrix", default="routerbench-ours9")
    parser.add_argument("--train-frac", type=float, default=0.7)
    args = parser.parse_args()
    global TRAIN_FRAC  # noqa: PLW0603 - one-shot CLI configuration
    TRAIN_FRAC = args.train_frac
    run_bakeoff(args)
    logger.info("runs -> %s", RUNS)


if __name__ == "__main__":
    main()
