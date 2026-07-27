"""Sim-to-real transfer: route REAL tau-bench tasks from world-model-fitted banks.

The headline experiment of the capture round: the promoted knn policy (production
knn_decision, adaptive rule, SE floor, pinned fable-5 fallback, z=0.5) fits its bank on
WORLD-MODEL tau-bench matrices, routes the REAL tau-bench tasks by embedding, and is scored
against the real benchmark's rewards (no world model, no judge). The question it answers:
do wm-fitted ROUTING DECISIONS help, hurt, or wash on reality?

Protocol:
- Banks: each available wm cohort (tau-bench 25-scen; tau-bench-s80 when captured), fitted
  on the full cohort AND on its 5 iid fit splits (bank-seed variation is the paired axis:
  the real test set is fixed).
- Queries: every tau-bench-real scenario, embedded once to
  cache/tau-bench-real-oai3l-tasks.npy (atomic single process).
- Comparisons per bank: pinned fable-5 (the serving fallback: what production does with
  routing OFF), best-single-on-real (HINDSIGHT anchor: chosen on the test data itself,
  labeled as such, never a deployable row), promoted config floor on (q=0.05) and off.
- Diagnostics: sim(bank, real-task) distribution + top overlap (is the real set the same
  tasks the wm simulated? near-dup retrieval is then the expected and honest mechanism),
  and per-accepted-pick real reward vs fable-5's on the same scenario (read the decisions).

Rows -> runs/r1.jsonl, matrix="tau-bench-real" (its own cohort, never merged with wm rows);
bank provenance in notes, knobs only in params.

Usage: uv run python .agents/scripts/r1_sim_to_real.py
"""

from __future__ import annotations

import importlib.util
import logging
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from wmo.optimize.knn import fit_knn_policy
from wmo.optimize.outcomes import OutcomeMatrix
from wmo.optimize.policy import EmbedderSpec, knn_decision
from wmo.research.routerbench import best_single_model, oracle, split_scenario_ids
from wmo.research.routing_runs import RunRecord, append_run, evaluate_choices
from wmo.research.routing_corpus import routing_data

_spec = importlib.util.spec_from_file_location(
    "r1_retrieval_ablations",
    Path(__file__).with_name("r1_retrieval_ablations.py"),
)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("r1.real")

DATA = routing_data()
RUNS = DATA / "runs" / "r1.jsonl"
REAL = DATA / "matrices" / "tau-bench-real_matrix.json"
BANK_COHORTS = ["tau-bench", "tau-bench-s80"]  # wm cohorts, if present
FALLBACK = "fable-5"


class _CachedEmbedder:
    def __init__(self, mapping: dict[str, np.ndarray]) -> None:
        self._mapping = mapping

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, self._mapping[text])) for text in texts]


def main() -> None:
    if not REAL.exists():
        logger.info("tau-bench-real matrix not captured yet (%s); nothing to do", REAL)
        return
    real = OutcomeMatrix.load(REAL)
    real_ctx = _mod.MatrixContext(real, "tau-bench-real", embed="openai", embed_replies=False)
    real_ids = real.scenario_ids()
    ts = datetime.now(tz=UTC).isoformat()

    fable_eval = evaluate_choices(real, real_ids, lambda _s: FALLBACK)
    # HINDSIGHT anchor: chosen on the evaluation data itself; a ceiling reference, not a row
    # any deployable router could produce.
    hind_name, _, _ = best_single_model(real, fit_ids=real_ids, eval_ids=real_ids)
    hind_eval = evaluate_choices(real, real_ids, lambda _s: hind_name)
    oracle_acc, oracle_cost = oracle(real, real_ids)
    logger.info(
        "real anchors: fable-5 %.4f | hindsight best-single %s %.4f | oracle %.4f",
        fable_eval.accuracy,
        hind_name,
        hind_eval.accuracy,
        oracle_acc,
    )

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
                notes=(
                    f"SIM-TO-REAL; baseline row = pinned {FALLBACK} (serving fallback); "
                    f"hindsight best-single {hind_name} acc={hind_eval.accuracy:.4f}; "
                    f"oracle acc={oracle_acc:.4f} cost=${oracle_cost:.5f}; "
                    f"embedder=openai{notes_extra}"
                ),
            ),
            RUNS,
        )
        logger.info(
            "%s: acc=%.4f (fable %.4f, d %+.4f) cost %+.1f%%%s",
            variant,
            result.accuracy,
            fable_eval.accuracy,
            result.accuracy - fable_eval.accuracy,
            (result.cost_per_call / fable_eval.cost_per_call - 1) * 100,
            notes_extra,
        )

    record(
        "r1-real-fable5",
        {"model": FALLBACK, "bank_seed": -1},
        dict.fromkeys(real_ids, FALLBACK),
        "; the routing-off row",
    )
    record(
        "r1-real-hindsight-best",
        {"model": hind_name, "bank_seed": -1},
        dict.fromkeys(real_ids, hind_name),
        "; HINDSIGHT anchor (chosen on test itself)",
    )

    for cohort in BANK_COHORTS:
        path = DATA / "matrices" / f"{cohort}_matrix.json"
        if not path.exists():
            logger.info("bank cohort %s not present; skipping", cohort)
            continue
        wm = OutcomeMatrix.load(path)
        wm_name = f"wm-{cohort}" if cohort == "tau-bench" else cohort
        wm_ctx = _mod.MatrixContext(wm, wm_name, embed="openai", embed_replies=False)
        text_to_vec = {wm_ctx.tasks[sid]: wm_ctx.task_vecs[sid] for sid in wm_ctx.scenario_ids}
        embedder = _CachedEmbedder(text_to_vec)

        # Diagnostic: how close are real tasks to the wm bank?
        bank_vecs = np.stack([wm_ctx.task_vecs[s] for s in wm_ctx.scenario_ids])
        max_sims = np.asarray([float(np.max(bank_vecs @ real_ctx.task_vecs[s])) for s in real_ids])
        logger.info(
            "%s -> real: sim p50=%.3f p95=%.3f n(>0.95)=%d n(>0.99)=%d",
            cohort,
            float(np.median(max_sims)),
            float(np.percentile(max_sims, 95)),
            int((max_sims > 0.95).sum()),
            int((max_sims > 0.99).sum()),
        )

        banks: list[tuple[int, list[str]]] = [(-1, list(wm_ctx.scenario_ids))]
        banks += [
            (seed, split_scenario_ids(wm, train_fraction=0.7, seed=seed)[0]) for seed in range(5)
        ]
        for bank_seed, fit_ids in banks:
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
                picks = {}
                routed = []
                for sid in real_ids:
                    decision = knn_decision(policy, real_ctx.task_vecs[sid])
                    picks[sid] = decision.model
                    if decision.model != FALLBACK:
                        routed.append((sid, decision.model, decision.reason))
                seed_tag = "full" if bank_seed == -1 else f"s{bank_seed}"
                record(
                    f"r1-real-knn-{cohort}" + ("-floor" if floor_q else ""),
                    {"bank_cohort": cohort, "floor_q": floor_q, "bank_seed": bank_seed},
                    picks,
                    f"; bank={cohort}/{seed_tag} n_bank={len(fit_ids)} "
                    f"routed_away={len(routed)}/{len(real_ids)}",
                )
                for sid, model, reason in routed[:4]:
                    fable_cell = real_ctx.rewards_cell.get((sid, FALLBACK))
                    pick_cell = real_ctx.rewards_cell.get((sid, model))
                    logger.info(
                        "  routed %s -> %s (real reward pick=%s fable=%s) | %s",
                        sid,
                        model,
                        None if not pick_cell else round(sum(pick_cell) / len(pick_cell), 2),
                        None if not fable_cell else round(sum(fable_cell) / len(fable_cell), 2),
                        reason[:110],
                    )
    logger.info("runs -> %s", RUNS)


if __name__ == "__main__":
    main()
