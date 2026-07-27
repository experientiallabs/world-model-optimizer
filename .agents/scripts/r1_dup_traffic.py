"""Q4: the duplicated-traffic experiment: where retrieval SHOULD beat clusters.

Protocol (synthetic, labeled as such in every record): sample fit scenarios, perturb their
task text lightly (word dropout/duplication, case flips: what repeat production traffic looks
like), and route the perturbed query. Ground truth per-model rewards are the ORIGINAL
scenario's cells; the router that recognizes "we have seen this request before" can route to
the model that actually solved it. This is deliberately test-on-fit: duplicate traffic means
the answer IS in the fit data, and the question is which router family can exploit that.
Generalization claims live in the split protocol, not here.

Routers compared on the identical stream: fit-chosen best-single, the Avengers rank router
(k=64, guarded, the master champion), retrieval kNN (statz05 promote candidate), and
retrieval with guard off. Also reports the per-item oracle and the top-1-neighbor hit rate
(did retrieval's nearest neighbor recover the original scenario?).

Usage: uv run python .agents/scripts/r1_dup_traffic.py [--seeds]
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
from wmo.optimize.policy import EmbedderSpec, rank_decision
from wmo.optimize.routing import fit_rank_policy
from wmo.research.routerbench import best_single_model, oracle, split_scenario_ids
from wmo.research.routing_runs import RunRecord, append_run, evaluate_choices
from wmo.retrieval.embedders import HashingEmbedder
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
logger = logging.getLogger("r1.dup")

DATA = routing_data()
RUNS = DATA / "runs" / "r1.jsonl"
DIM = 1024
N_DUPS = 300


def perturb(text: str, rng: random.Random) -> str:
    """Light repeat-traffic noise: word dropout/duplication and case flips."""
    words = text.split(" ")
    out: list[str] = []
    for word in words:
        roll = rng.random()
        if roll < 0.05 and len(words) > 10:
            continue  # drop
        out.append(word)
        if roll > 0.97:
            out.append(word)  # stutter
    mutated = " ".join(out)
    chars = list(mutated)
    for index in rng.sample(range(len(chars)), max(1, len(chars) // 200)):
        chars[index] = chars[index].swapcase()
    return "".join(chars)


def run_seed(name: str, ctx: MatrixContext, split_seed: int) -> None:
    matrix = ctx.matrix
    fit_ids, test_ids = split_scenario_ids(matrix, train_fraction=0.7, seed=split_seed)
    rng = random.Random(1000 + split_seed)
    dup_sids = rng.sample(fit_ids, min(N_DUPS, len(fit_ids)))
    embedder = HashingEmbedder(dim=DIM)
    dup_texts = {sid: perturb(ctx.tasks[sid], rng) for sid in dup_sids}
    dup_vec_list = np.asarray(embedder.embed([dup_texts[sid] for sid in dup_sids]))
    norms = np.linalg.norm(dup_vec_list, axis=1, keepdims=True)
    dup_vec_list = dup_vec_list / np.where(norms > 0, norms, 1.0)
    dup_vecs = {sid: dup_vec_list[i] for i, sid in enumerate(dup_sids)}

    best_name, _, _ = best_single_model(matrix, fit_ids=fit_ids, eval_ids=dup_sids)
    best_eval = evaluate_choices(matrix, dup_sids, lambda _sid: best_name)
    oracle_acc, oracle_cost = oracle(matrix, dup_sids)
    ts = datetime.now(tz=UTC).isoformat()

    fit_matrix = np.stack([ctx.task_vecs[sid] for sid in fit_ids])
    top1_hits = sum(
        1 for sid in dup_sids if fit_ids[int(np.argmax(fit_matrix @ dup_vecs[sid]))] == sid
    )
    sim_to_orig = float(np.mean([float(ctx.task_vecs[sid] @ dup_vecs[sid]) for sid in dup_sids]))
    logger.info(
        "%s seed%d: top1-neighbor hit rate %.3f, mean sim(dup, original) %.3f",
        name,
        split_seed,
        top1_hits / len(dup_sids),
        sim_to_orig,
    )

    def record(variant: str, params: dict, result) -> None:  # noqa: ANN001
        append_run(
            RunRecord(
                run_id=f"{name}-{variant}-{uuid.uuid4().hex[:8]}",
                ts=ts,
                # Distinct matrix name: dup-traffic is a different protocol and must not sit
                # on the split-protocol Pareto (the dashboard's default view filters it out).
                matrix=f"{name}-duptraffic",
                variant=variant,
                params=params,
                split_seed=split_seed,
                fit_scenarios=len(fit_ids),
                test_scenarios=len(dup_sids),
                result=result,
                baselines={"best_single": best_eval},
                notes=(
                    f"DUP-TRAFFIC protocol (synthetic repeats of fit items; NOT the split "
                    f"protocol); best_single={best_name}; oracle acc={oracle_acc:.4f} "
                    f"cost=${oracle_cost:.5f}; top1_hit={top1_hits / len(dup_sids):.3f}; "
                    f"mean_dup_sim={sim_to_orig:.3f}; embedder=hashing-{DIM}"
                ),
            ),
            RUNS,
        )
        logger.info(
            "%s/%s seed%d: acc=%.4f (best %.4f, oracle %.4f) cost=$%.5f (%+.1f%%)",
            name,
            variant,
            split_seed,
            result.accuracy,
            best_eval.accuracy,
            oracle_acc,
            result.cost_per_call,
            (result.cost_per_call / best_eval.cost_per_call - 1) * 100,
        )

    record("r1-dup-best-single", {"model": best_name}, best_eval)

    policy = fit_rank_policy(
        matrix,
        fit_ids=fit_ids,
        embedder=EmbedderSpec(dim=DIM),
        n_clusters=64,
        seed=42,
        guard_model=best_name,
        min_support=4,
        guard_margin=0.03,
        fitted_from=f"{name} split{split_seed} dup-traffic",
    )
    rank_picks = {sid: rank_decision(policy, dup_vecs[sid]).model for sid in dup_sids}
    record(
        "r1-dup-rank",
        {"k": 64, "guard": True},
        evaluate_choices(matrix, dup_sids, lambda s, p=rank_picks: p[s]),
    )

    for variant, params in [
        ("r1-dup-knn-statz05", RetrievalParams(second_route=False, guard="stat", z=0.5)),
        ("r1-dup-knn-noguard", RetrievalParams(second_route=False, guard="none")),
    ]:
        picks = _mod.route(
            ctx,
            params,
            fit_ids,
            dup_sids,
            best_name,
            query_vecs=dup_vecs,
        )
        record(
            variant,
            params.model_dump(),
            evaluate_choices(matrix, dup_sids, lambda s, p=picks: p[s]),
        )


def main() -> None:
    seeds = _mod.SPLIT_SEEDS if "--seeds" in sys.argv else [0]
    matrix = OutcomeMatrix.load(DATA / "matrices" / "routerbench-ours9_matrix.json")
    ctx = MatrixContext(matrix, "routerbench-ours9")
    for seed in seeds:
        run_seed("routerbench-ours9", ctx, seed)
    logger.info("runs -> %s", RUNS)


if __name__ == "__main__":
    main()
