"""Sim-to-real drift defense: does the SVM global-agreement gate protect on REAL tau-bench?

r3's slot in the transfer experiment (r1 ran the champion side: r1-real-knn-tau-bench[-floor]
rows, verdict wash-to-hurt, mean -0.0114, floor WORSE). This script reruns the IDENTICAL
protocol (r1_sim_to_real.py: production fit_knn_policy + knn_decision, pinned fable-5,
wm tau-bench banks full + 5 fit-split seeds, real rewards, no judge) and adds the bake-off's
recommended defense: champion picks kept only when an SVM win-vs-fable classifier TRAINED ON
THE SAME WM BANK agrees (P_win > 0.5 on the real query).

Registered prediction (before running): the wm compresses model spread (r1's mechanism), so
wm-trained P_win hovers near 0.5 and the gate blocks most routed-away picks - safety
(dacc ~ 0) at the price of the cost knob. Banks are 18-25 scenarios, so many per-model
classifiers fall below the 12-decisive-cell floor and are inert by construction (the
n-floor behaving as designed).

Rows -> runs/r3.jsonl, matrix="tau-bench-real", variants r3b-real-*; champion/floor
comparison rows are r1's (cited, not recomputed). $0 API (all embeddings cached).
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import numpy as np
from wmo.optimize.knn import fit_knn_policy

from wmo.optimize.outcomes import OutcomeMatrix
from wmo.optimize.policy import EmbedderSpec, knn_decision
from wmo.research.routerbench import split_scenario_ids
from wmo.research.routing_runs import RunRecord, append_run, evaluate_choices
from wmo.research.routing_corpus import routing_data

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("r3.real")

DATA = routing_data()
RUNS = DATA / "runs" / "r3.jsonl"
FALLBACK = "fable-5"


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


class _CachedEmbedder:
    def __init__(self, mapping: dict[str, np.ndarray]) -> None:
        self._mapping = mapping

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, self._mapping[text])) for text in texts]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-cohort", default="tau-bench")
    args = parser.parse_args()
    real = OutcomeMatrix.load(DATA / "matrices" / "tau-bench-real_matrix.json")
    real_ctx = R1.MatrixContext(real, "tau-bench-real", embed="openai", embed_replies=False)
    real_ids = real.scenario_ids()
    names = real.model_names()
    ts = datetime.now(tz=UTC).isoformat()
    fable_eval = evaluate_choices(real, real_ids, lambda _s: FALLBACK)

    cohort = args.bank_cohort
    wm = OutcomeMatrix.load(DATA / "matrices" / f"{cohort}_matrix.json")
    wm_name = f"wm-{cohort}" if cohort == "tau-bench" else cohort
    wm_ctx = R1.MatrixContext(wm, wm_name, embed="openai", embed_replies=False)
    text_to_vec = {wm_ctx.tasks[sid]: wm_ctx.task_vecs[sid] for sid in wm_ctx.scenario_ids}
    embedder = _CachedEmbedder(text_to_vec)

    def record(variant: str, params: dict, picks: dict[str, str], notes_extra: str) -> None:
        result = evaluate_choices(real, real_ids, lambda s, p=picks: p[s])
        append_run(
            RunRecord(
                run_id=f"tau-bench-real-{variant}-{uuid.uuid4().hex[:8]}",
                ts=ts,
                matrix="tau-bench-real",
                variant=variant,
                params=params,
                split_seed=int(params.get("bank_seed", -1)),
                fit_scenarios=0,
                test_scenarios=len(real_ids),
                result=result,
                baselines={"best_single": fable_eval},
                notes=f"SIM-TO-REAL defense; baseline = pinned {FALLBACK}; "
                f"champion comparison rows = r1-real-knn-tau-bench[-floor] in r1.jsonl"
                f"{notes_extra}",
            ),
            RUNS,
        )
        logger.info(
            "%s %s: acc=%.4f (fable %.4f, d %+.4f) cost %+.1f%%%s",
            variant, params, result.accuracy, fable_eval.accuracy,
            result.accuracy - fable_eval.accuracy,
            (result.cost_per_call / fable_eval.cost_per_call - 1) * 100, notes_extra,
        )

    banks: list[tuple[int, list[str]]] = [(-1, list(wm_ctx.scenario_ids))]
    banks += [
        (seed, split_scenario_ids(wm, train_fraction=0.7, seed=seed)[0]) for seed in range(5)
    ]
    for bank_seed, fit_ids in banks:
        # SVM win-vs-fable per model, trained on the wm bank's decisive cells.
        cell = wm_ctx.rewards_cell
        fit_x = np.stack([wm_ctx.task_vecs[s] for s in fit_ids])
        real_x = np.stack([real_ctx.task_vecs[s] for s in real_ids])
        p_win = np.zeros((len(real_ids), len(names)))
        trainable = []
        for mi, model in enumerate(names):
            if model == FALLBACK:
                continue
            xs, ys = [], []
            for row, sid in enumerate(fit_ids):
                pv, bv = cell.get((sid, model)), cell.get((sid, FALLBACK))
                if pv and bv:
                    d = float(np.mean(pv)) - float(np.mean(bv))
                    if abs(d) > 1e-9:
                        xs.append(fit_x[row])
                        ys.append(1 if d > 0 else 0)
            if len(ys) < 12:
                continue
            predict = R3X.clf_family("svm", np.stack(xs), np.asarray(ys))
            p_win[:, mi] = predict(real_x)
            trainable.append(model)

        for floor_q in (0.0, 0.05):
            with tempfile.TemporaryDirectory() as tmp:
                policy = fit_knn_policy(
                    wm,
                    bank_path=Path(tmp) / "bank.npz",
                    fit_ids=fit_ids,
                    embedder=EmbedderSpec(dim=3072),
                    embed_with=embedder,
                    guard_model=FALLBACK,
                    floor_q=floor_q,
                )
                champ = {
                    sid: knn_decision(policy, real_ctx.task_vecs[sid]).model
                    for sid in real_ids
                }
            gated, blocked = {}, 0
            for t, sid in enumerate(real_ids):
                pick = champ[sid]
                if pick != FALLBACK and p_win[t, names.index(pick)] <= 0.5:
                    pick = FALLBACK
                    blocked += 1
                gated[sid] = pick
            routed = sum(1 for p in gated.values() if p != FALLBACK)
            routed_pre = sum(1 for p in champ.values() if p != FALLBACK)
            record(
                "r3b-real-champ" + ("-floor" if floor_q else ""),
                {"bank_cohort": cohort, "floor_q": floor_q, "bank_seed": bank_seed},
                champ,
                f"; PRE-GATE champion (r1 protocol replication, for pairing; r1's own "
                f"rows canonical when present); bank_n={len(fit_ids)} "
                f"routed_away={routed_pre}/{len(real_ids)}",
            )
            record(
                "r3b-real-hybrid-svm" + ("-floor" if floor_q else ""),
                {"bank_cohort": cohort, "floor_q": floor_q, "bank_seed": bank_seed},
                gated,
                f"; bank_n={len(fit_ids)} svm_trainable={len(trainable)} "
                f"blocked={blocked} routed_away={routed}/{len(real_ids)}",
            )
    logger.info("runs -> %s", RUNS)


if __name__ == "__main__":
    main()
