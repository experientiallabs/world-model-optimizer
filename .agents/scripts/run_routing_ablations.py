"""Controlled routing ablations: every variant x every available matrix -> runs.jsonl + reports.

Variants (identical splits, embeddings, and evaluator per matrix): best-single (fit-chosen),
rank router (Avengers replication, cost-knob sweep), IRT head (cost-knob sweep). Each run
persists a RunRecord with the explain block; per-run markdown reports land in
.wmo/evals/reports/. The dashboard (build_dashboard.py) renders runs.jsonl.

`bo2-free` (best-of-2 on one model, picked by the free post-hoc selector) has its OWN entry point
rather than living in `run_matrix`, so it can be appended to a runs.jsonl that already has the
other variants swept without re-emitting duplicate rows for them:

    uv run .agents/scripts/run_routing_ablations.py --only bo2-free --seeds

It is also the only variant scored by `evaluate_call_sequences` (two calls, cost summed) rather
than `evaluate_choices`.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.preprocessing import Normalizer

from wmo.optimize.irt import fit_irt_head
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.policy import EmbedderSpec, rank_decision
from wmo.optimize.routing import fit_rank_policy, rerank_policy
from wmo.research.posthoc_bounds import DEFAULT_SELECTOR, SELECTOR_KEYS, best_of_n_by_model
from wmo.research.routing_runs import (
    ChoiceEval,
    Finish,
    RunRecord,
    append_run,
    evaluate_call_sequences,
    evaluate_choices,
    run_report,
)
from wmo.retrieval.embedders import HashingEmbedder

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("ablations")

RUNS = Path(".wmo/evals/runs.jsonl")
REPORTS = Path(".wmo/evals/reports")
DIM = 1024
K = 64
LAMS = [0.0, 0.02, 0.1]
# Multiple disjoint-ish splits: every scenario reaches test across seeds, so per-matrix
# signal comes from mean +- spread over seeds, not one cherry-pickable 70/30 draw.
SPLIT_SEEDS = [0, 1, 2, 3, 4]


def _matrices() -> dict[str, OutcomeMatrix]:
    out: dict[str, OutcomeMatrix] = {}
    rb = Path("/Users/silen/Desktop/Projects/router-refs/routerbench_0shot.pkl")
    if rb.exists():
        from wmo.research.routerbench import load_routerbench

        out["routerbench"] = load_routerbench(rb)
    lrb = Path("/Users/silen/Desktop/Projects/router-refs/LLMRouterBench/results/bench-release")
    if lrb.is_dir():
        from wmo.research.llmrouterbench import load_llmrouterbench

        out["llmrouterbench-flagship"] = load_llmrouterbench(lrb)
    ours = Path(".wmo/evals/routerbench/ours_matrix.json")
    if ours.exists():
        out["routerbench-ours9"] = OutcomeMatrix.load(ours)
    wm_matrices = []
    for wm in sorted(Path(".wmo/evals/wm").glob("*_matrix.json")):
        corpus = wm.stem.removesuffix("_matrix")
        matrix = OutcomeMatrix.load(wm)
        out[f"wm-{corpus}"] = matrix
        wm_matrices.append((corpus, matrix))
    if len(wm_matrices) >= 2:
        # The pooled cross-corpus aggregate: per-corpus test sides are tiny, so THIS is where
        # the statistically real wm signal lives. Scenario ids get a corpus prefix, which also
        # makes the stratified split per-corpus (each corpus contributes to fit AND test).
        combined = []
        for corpus, matrix in wm_matrices:
            for outcome in matrix.outcomes:
                clone = outcome.model_copy(
                    update={"scenario_id": f"{corpus}:{outcome.scenario_id}"}
                )
                combined.append(clone)
        out["wm-all"] = OutcomeMatrix(pool=wm_matrices[0][1].pool, outcomes=combined)
    return out


def _emit(record: RunRecord) -> None:
    append_run(record, RUNS)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"{record.run_id}.md").write_text(run_report(record), encoding="utf-8")
    result = record.result
    logger.info(
        "%s/%s %s: acc=%.4f cost=$%.5f p50=%s",
        record.matrix,
        record.variant,
        record.params,
        result.accuracy,
        result.cost_per_call,
        f"{result.latency_p50_s:.2f}s" if result.latency_p50_s else "-",
    )


Recorder = Callable[[str, dict, ChoiceEval], None]


def _prepare(
    name: str, matrix: OutcomeMatrix, split_seed: int
) -> tuple[list[str], list[str], str, ChoiceEval, Recorder]:
    """The split, the best-single baseline, and the RunRecord factory every variant shares.

    Shared so `run_matrix` and `run_bo2_free` cannot drift apart on the split, the baseline, or
    the record shape - which is what makes their rows comparable paired-by-seed.
    """
    from wmo.research.routerbench import best_single_model, oracle, split_scenario_ids

    fit_ids, test_ids = split_scenario_ids(matrix, train_fraction=0.7, seed=split_seed)
    ts = datetime.now(tz=UTC).isoformat()
    best_name, _acc, _cost = best_single_model(matrix, fit_ids=fit_ids, eval_ids=test_ids)
    best_eval = evaluate_choices(matrix, test_ids, lambda _sid: best_name)
    oracle_acc, oracle_cost = oracle(matrix, test_ids)

    def record(variant: str, params: dict, result: ChoiceEval) -> None:
        _emit(
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
                notes=f"best_single={best_name}; oracle acc={oracle_acc:.4f} "
                f"cost=${oracle_cost:.5f}; embedder=hashing-{DIM}",
            )
        )

    return fit_ids, test_ids, best_name, best_eval, record


def run_matrix(name: str, matrix: OutcomeMatrix, split_seed: int = 0) -> None:
    fit_ids, test_ids, best_name, best_eval, record = _prepare(name, matrix, split_seed)
    tasks = {o.scenario_id: o.task for o in matrix.outcomes}
    embedder = HashingEmbedder(dim=DIM)
    fit_vecs = np.asarray(embedder.embed([tasks[s] for s in fit_ids]))
    test_vecs = Normalizer(norm="l2").transform(
        np.asarray(embedder.embed([tasks[s] for s in test_ids]))
    )

    record("best-single", {"model": best_name}, best_eval)

    started = time.monotonic()
    policy = fit_rank_policy(
        matrix, fit_ids=fit_ids, embedder=EmbedderSpec(dim=DIM), n_clusters=K, seed=42,
        guard_model=best_name, min_support=4, guard_margin=0.03,
        fitted_from=f"{name} split{split_seed}",
    )
    logger.info("%s: rank fit in %.0fs", name, time.monotonic() - started)
    for lam in LAMS:
        swept = rerank_policy(policy, cost_weight=lam) if lam else policy
        decisions = {
            sid: rank_decision(swept, test_vecs[index]).model
            for index, sid in enumerate(test_ids)
        }
        record(
            "rank", {"k": K, "lam": lam},
            evaluate_choices(matrix, test_ids, lambda sid: decisions[sid]),
        )

    started = time.monotonic()
    head = fit_irt_head(
        matrix, scenario_ids=fit_ids, embeddings=fit_vecs, seed=42, epochs=300,
        hidden=256, dim=64,
    )
    logger.info("%s: irt fit in %.0fs (pairs=%d)", name, time.monotonic() - started, head.pairs_trained)
    costs_by_model: dict[str, list[float]] = {}
    fit_set = set(fit_ids)
    for outcome in matrix.outcomes:
        if outcome.scenario_id in fit_set and outcome.reward is not None:
            costs_by_model.setdefault(outcome.model, []).append(outcome.cost_usd)
    mean_cost = {m: sum(v) / len(v) for m, v in costs_by_model.items() if v}
    cost_scale = sum(mean_cost.values()) / len(mean_cost)
    probs = np.stack([head.predict(vec) for vec in test_vecs])  # [T, M]
    penalties = np.asarray(
        [mean_cost.get(m, cost_scale) / cost_scale for m in head.models]
    )
    baseline_index = head.models.index(best_name)
    for lam in LAMS:
        scores = probs - lam * penalties
        raw = np.argmax(scores, axis=1)
        # Guard: keep the best-single baseline unless the picked model's score actually
        # exceeds the baseline's (worst case == 1x best-single, per product requirement).
        base_cost = mean_cost.get(best_name, cost_scale)
        picks = [
            head.models[int(index)]
            if scores[row, index]
            > scores[row, baseline_index]
            + (0.06 if mean_cost.get(head.models[int(index)], 0.0) > base_cost else 0.03)
            else best_name
            for row, index in enumerate(raw)
        ]
        decisions = dict(zip(test_ids, picks, strict=True))
        record(
            "irt", {"hidden": 256, "dim": 64, "epochs": 300, "lam": lam, "guard": True},
            evaluate_choices(matrix, test_ids, lambda sid: decisions[sid]),
        )

    # ProxRouter-inspired support tilt on the guarded rank policy.
    tilted = policy.model_copy(update={"support_tilt_gamma": 0.5})
    decisions = {
        sid: rank_decision(tilted, test_vecs[index]).model
        for index, sid in enumerate(test_ids)
    }
    record(
        "rank-tilt", {"k": K, "gamma": 0.5, "lam": 0.0},
        evaluate_choices(matrix, test_ids, lambda sid: decisions[sid]),
    )

    # JiSi phase-1, PROXY mode (deployable; the paper's s2s mode peeks at test responses).
    replies = {}
    for outcome in matrix.outcomes:
        if outcome.replies and outcome.reward is not None:
            replies[(outcome.scenario_id, outcome.model)] = outcome.replies[0]
    rewards_cell = {}
    for outcome in matrix.outcomes:
        if outcome.reward is not None:
            rewards_cell.setdefault((outcome.scenario_id, outcome.model), []).append(
                outcome.reward
            )
    fit_norm = Normalizer(norm="l2").transform(fit_vecs)
    sims_all = test_vecs @ fit_norm.T  # [T, F]
    model_names = [entry.name for entry in matrix.pool]
    jisi_picks: dict[str, str] = {}
    for row, sid in enumerate(test_ids):
        sims = sims_all[row]
        k = min(50, len(fit_ids))
        kth = np.sort(sims)[-k]
        neighbor_rows = np.where(sims > 0.95 * kth)[0]
        if not len(neighbor_rows):
            neighbor_rows = np.asarray([int(np.argmax(sims))])
        def profile(rows_idx, weights):
            scores_local = {}
            for m in model_names:
                num = den = 0.0
                for j, weight in zip(rows_idx, weights, strict=True):
                    cell = rewards_cell.get((fit_ids[int(j)], m))
                    if cell:
                        num += weight * (sum(cell) / len(cell))
                        den += weight
                if den:
                    scores_local[m] = num / den
            return scores_local
        first = profile(neighbor_rows, sims[neighbor_rows])
        if not first:
            jisi_picks[sid] = best_name
            continue
        needles = sorted(first, key=lambda m: -first[m])[:3]
        # proxy_s2s: neighbor quality = how consistent each neighbor's needle responses are
        # with the OTHER neighbors' (train-side only; no test response needed).
        refine = []
        for j in neighbor_rows:
            sims_resp = []
            for m in needles:
                mine = replies.get((fit_ids[int(j)], m))
                if not mine:
                    continue
                others = [
                    replies.get((fit_ids[int(o)], m))
                    for o in neighbor_rows
                    if o != j and replies.get((fit_ids[int(o)], m))
                ]
                if not others:
                    continue
                vecs = np.asarray(embedder.embed([mine, *others[:5]]))
                sims_resp.append(float(vecs[0] @ vecs[1:].T.mean(axis=1)))
            resp = float(np.mean(sims_resp)) if sims_resp else 0.0
            refine.append(0.5 * sims[int(j)] + 0.5 * resp)
        refine = np.asarray(refine)
        keep = max(1, int(len(neighbor_rows) * 0.5))
        top_idx = np.argsort(refine)[-keep:]
        second = profile(neighbor_rows[top_idx], refine[top_idx])
        pick = max(second, key=lambda m: (second[m] - LAMS[1] * 0
                    , -model_names.index(m))) if second else best_name
        # Guard: revert to baseline unless the pick's profile beats the baseline's.
        pick_margin = 0.06 if mean_cost.get(pick, 0.0) > mean_cost.get(best_name, 0.0) else 0.03
        if second.get(pick, 0.0) <= second.get(best_name, 0.0) + pick_margin:
            pick = best_name
        jisi_picks[sid] = pick
    record(
        "jisi", {"mode": "proxy", "rag": 50, "thres": 0.95, "guard": True},
        evaluate_choices(matrix, test_ids, lambda sid: jisi_picks[sid]),
    )


def run_bo2_free(name: str, matrix: OutcomeMatrix, split_seed: int = 0) -> None:
    """Best-of-2 on ONE model, choosing between that model's own two rollouts.

    Every other variant picks a model from the QUERY. This one spends 2x on a single model and
    picks between the finished rollouts using the free post-hoc selector, which reads only
    `stop_reason` and `steps` - the DEPLOYABLE side of `evaluate_call_sequences`' information
    boundary, never `reward` or `critique`.

    The model is DISCOVERED on the fit split, never assumed: a candidate must clear fit-split
    best-single by the same margin the other guarded variants use (0.03, doubled to 0.06 when its
    2x cost exceeds best-single's 1x cost, since paying more has to buy more). If none qualifies
    the variant collapses to best-single 1-shot, so it cannot be worse than the 1x baseline by
    construction.
    """
    from wmo.research.routerbench import best_single_model

    fit_ids, test_ids, best_name, best_eval, record = _prepare(name, matrix, split_seed)

    fit_best, fit_acc, fit_cost = best_single_model(matrix, fit_ids=fit_ids, eval_ids=fit_ids)
    fit_bounds = best_of_n_by_model(
        matrix, fit_ids, baseline_accuracy=fit_acc, baseline_cost=fit_cost
    )
    if not fit_bounds:
        logger.info("%s: no cell sampled more than once, skipping bo2-free", name)
        return

    qualified = [
        bound
        for bound in fit_bounds
        if bound.selected_of_n_accuracy
        > fit_acc + (0.06 if bound.oracle_of_n_cost_per_call > fit_cost else 0.03)
    ]
    qualified.sort(key=lambda b: (-b.selected_of_n_accuracy, b.oracle_of_n_cost_per_call, b.model))
    chosen = qualified[0].model if qualified else None
    logger.info(
        "%s seed%d: bo2-free -> %s (fit best-single %s %.4f @ $%.5f); %d/%d qualified",
        name,
        split_seed,
        chosen or f"NONE, falls back to {best_name} 1-shot",
        fit_best,
        fit_acc,
        fit_cost,
        len(qualified),
        len(fit_bounds),
    )

    params = {
        "selector": DEFAULT_SELECTOR,
        "model": chosen or best_name,
        "n": 2 if chosen else 1,
        "guard_margin": 0.03,
        "guard_margin_costly": 0.06,
        "fell_back": chosen is None,
    }
    if chosen is None:
        # Fall back to best-single's OWN evaluation, not a one-call sequence. The two evaluators
        # treat a multi-episode cell differently: evaluate_choices averages the cell's episodes
        # (the expected value of one call), while evaluate_call_sequences consumes the k-th
        # episode, so a 1-call sequence scores episode 0 alone. Scoring the fallback that way made
        # it look 10pt WORSE than best-single on terminal-tasks purely from episode-0 luck, which
        # would both break the "never worse than 1x" guarantee and corrupt the paired delta.
        record("bo2-free", params, best_eval)
        return

    episodes_available: dict[tuple[str, str], int] = {}
    for outcome in matrix.outcomes:
        if outcome.reward is not None:
            cell = (outcome.scenario_id, outcome.model)
            episodes_available[cell] = episodes_available.get(cell, 0) + 1
    selector = SELECTOR_KEYS[DEFAULT_SELECTOR]
    target = chosen

    def policy(sid: str, transcript: list[ScenarioOutcome]) -> str | Finish:
        # `or 1` keeps a scenario with no stored episode at one attempted call, which
        # evaluate_call_sequences then reports as unscored rather than silently skipping it.
        # The selector is order-independent, so the 2-call value is a true expectation (unlike a
        # 1-call sequence, which would be one draw) and stays comparable to best-single's row.
        budget = min(2, episodes_available.get((sid, target), 1)) or 1
        if len(transcript) < budget:
            return target
        return Finish(pick=min(range(len(transcript)), key=lambda i: selector(transcript[i])))

    record("bo2-free", params, evaluate_call_sequences(matrix, test_ids, policy))


def main() -> None:
    argv = sys.argv[1:]
    only = None
    if "--only" in argv:
        index = argv.index("--only")
        only = argv[index + 1] if index + 1 < len(argv) else None
        argv = argv[:index] + argv[index + 2 :]  # drop the flag AND its value
    if only is not None and only != "bo2-free":
        raise SystemExit("--only currently supports just 'bo2-free' (the rest share run_matrix)")
    wanted = [a for a in argv if not a.startswith("--")]
    seeds = SPLIT_SEEDS if "--seeds" in argv else [0]
    for name, matrix in _matrices().items():
        if wanted and name not in wanted:
            continue
        for seed in seeds:
            if only == "bo2-free":
                run_bo2_free(name, matrix, split_seed=seed)
            else:
                run_matrix(name, matrix, split_seed=seed)
    logger.info("runs -> %s, reports -> %s", RUNS, REPORTS)


if __name__ == "__main__":
    main()
