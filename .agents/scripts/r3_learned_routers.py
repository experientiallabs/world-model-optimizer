"""R3 hill-climb driver: learned ability models (IRT family) vs honest simple baselines.

Chat r3 of the routing hill-climb. Every experiment reports through
`wmo.research.routing_runs.evaluate_choices` and appends RunRecords to
`<routing-data>/runs/r3.jsonl` with variants namespaced `r3-<experiment>`.

Predictor families share ONE interface: fit on (matrix, fit_ids, fit_vecs), return a
[test, models] matrix of predicted P(correct). One shared decision rule (cost knob + margin
guard, identical to .agents/scripts/run_routing_ablations.py) turns any P matrix into picks,
so family comparisons are apples-to-apples by construction.

Subcommands: repro (champion irt config, 5 seeds), baselines (logistic-per-model /
kNN-P vs irt), audit (leak checks + shuffled-label control), sweep (capacity/reg with
train+val curves), curves (sample efficiency), calibration (reliability bins).
"""

from __future__ import annotations

import argparse
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from wmo.optimize.irt import IrtHead, _forward_loss_grads, _Params, fit_irt_head
from wmo.optimize.outcomes import OutcomeMatrix
from wmo.research.routerbench import best_single_model, oracle, split_scenario_ids
from wmo.research.routing_runs import ChoiceEval, RunRecord, append_run, evaluate_choices
from wmo.retrieval.embedders import HashingEmbedder
from wmo.research.routing_corpus import routing_data

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("r3")

DATA = routing_data()
RUNS = DATA / "runs/r3.jsonl"
DIM = 1024
SPLIT_SEEDS = [0, 1, 2, 3, 4]
LAMS = [0.0, 0.02, 0.1]
MARGIN = 0.03  # guard margin vs predicted baseline score (doubled when pick is pricier)


# ---------------------------------------------------------------- data plumbing


def load_matrices(names: list[str] | None = None) -> dict[str, OutcomeMatrix]:
    """Matrices from the shared data dir, plus the pooled wm union (wm-all)."""
    out: dict[str, OutcomeMatrix] = {}
    wm: list[tuple[str, OutcomeMatrix]] = []
    for path in sorted((DATA / "matrices").glob("*_matrix.json")):
        corpus = path.stem.removesuffix("_matrix")
        matrix = OutcomeMatrix.load(path)
        if corpus == "routerbench-ours9":
            out[corpus] = matrix
        else:
            out[f"wm-{corpus}"] = matrix
            wm.append((corpus, matrix))
    if len(wm) >= 2:
        combined = [
            o.model_copy(update={"scenario_id": f"{corpus}:{o.scenario_id}"})
            for corpus, matrix in wm
            for o in matrix.outcomes
        ]
        out["wm-all"] = OutcomeMatrix(pool=wm[0][1].pool, outcomes=combined)
    if names:
        missing = [n for n in names if n not in out]
        if missing:
            raise SystemExit(f"unknown matrices {missing}; have {sorted(out)}")
        out = {n: out[n] for n in names}
    return out


def scenario_tasks(matrix: OutcomeMatrix) -> dict[str, str]:
    return {o.scenario_id: o.task for o in matrix.outcomes}


def embed(texts: list[str]) -> np.ndarray:
    return np.asarray(HashingEmbedder(dim=DIM).embed(texts))


def pairs_for(
    matrix: OutcomeMatrix, ids: list[str], vecs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(queries [P,D], model_index [P], labels [P]) over scored outcomes in `ids`."""
    row_of = {sid: i for i, sid in enumerate(ids)}
    model_row = {name: i for i, name in enumerate(matrix.model_names())}
    q_rows, m_idx, labels = [], [], []
    for o in matrix.outcomes:
        if o.reward is None or o.scenario_id not in row_of:
            continue
        q_rows.append(row_of[o.scenario_id])
        m_idx.append(model_row[o.model])
        labels.append(o.reward)
    return vecs[q_rows], np.asarray(m_idx), np.asarray(labels, dtype=np.float64)


def mean_costs(matrix: OutcomeMatrix, fit_ids: list[str]) -> dict[str, float]:
    fit_set = set(fit_ids)
    by_model: dict[str, list[float]] = {}
    for o in matrix.outcomes:
        if o.scenario_id in fit_set and o.reward is not None:
            by_model.setdefault(o.model, []).append(o.cost_usd)
    return {m: sum(v) / len(v) for m, v in by_model.items() if v}


# ---------------------------------------------------------------- shared decision rule


def guarded_picks(
    probs: np.ndarray,
    models: list[str],
    costs: dict[str, float],
    best_name: str,
    lam: float,
    margin: float = MARGIN,
) -> list[str]:
    """The ablations decision rule: argmax(P - lam*penalty), margin guard vs best-single.

    Guard: keep the pick only when its score beats the baseline's predicted score by
    `margin` (doubled when the pick is pricier than the baseline); else fall back.
    """
    cost_scale = sum(costs.values()) / len(costs)
    penalties = np.asarray([costs.get(m, cost_scale) / cost_scale for m in models])
    scores = probs - lam * penalties
    base_idx = models.index(best_name)
    base_cost = costs.get(best_name, cost_scale)
    picks = []
    for row, idx in enumerate(np.argmax(scores, axis=1)):
        pick = models[int(idx)]
        need = 2 * margin if costs.get(pick, 0.0) > base_cost else margin
        picks.append(pick if scores[row, idx] > scores[row, base_idx] + need else best_name)
    return picks


# ---------------------------------------------------------------- predictor families


def bce(probs: np.ndarray, labels: np.ndarray) -> float:
    eps = 1e-12
    return float(-np.mean(labels * np.log(probs + eps) + (1 - labels) * np.log(1 - probs + eps)))


def fit_irt_curves(
    queries: np.ndarray,
    model_index: np.ndarray,
    labels: np.ndarray,
    n_models: int,
    *,
    val: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    seed: int = 42,
    epochs: int = 300,
    hidden: int = 256,
    dim: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    checkpoint: bool = False,
) -> tuple[_Params, list[float], list[float]]:
    """The wmo irt Adam loop, with per-epoch train/val BCE tracking (r3's extra rule).

    With `checkpoint=True` (requires `val`), the returned params are a snapshot from the
    best-val epoch (early stopping without a patience heuristic: run all epochs, keep min).
    """
    rng = np.random.default_rng(seed)
    params = _Params(rng, n_models, queries.shape[1], hidden, dim)
    moments = [(np.zeros_like(t), np.zeros_like(t)) for t in params.tensors()]
    m_bb = v_bb = 0.0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    train_curve, val_curve = [], []
    best: list[np.ndarray] | None = None
    best_bb = 0.0
    for step in range(1, epochs + 1):
        loss, grads, grad_bb = _forward_loss_grads(params, queries, model_index, labels)
        train_curve.append(loss)
        if val is not None:
            vloss, _, _ = _forward_loss_grads(params, *val)
            if checkpoint and (not val_curve or vloss < min(val_curve)):
                best = [t.copy() for t in params.tensors()]
                best_bb = params.bb
            val_curve.append(vloss)
        for i, (tensor, grad) in enumerate(zip(params.tensors(), grads, strict=True)):
            grad = grad + weight_decay * tensor
            m, v = moments[i]
            m[:] = beta1 * m + (1 - beta1) * grad
            v[:] = beta2 * v + (1 - beta2) * grad**2
            tensor -= lr * (m / (1 - beta1**step)) / (np.sqrt(v / (1 - beta2**step)) + eps)
        m_bb = beta1 * m_bb + (1 - beta1) * grad_bb
        v_bb = beta2 * v_bb + (1 - beta2) * grad_bb**2
        params.bb -= lr * (m_bb / (1 - beta1**step)) / (np.sqrt(v_bb / (1 - beta2**step)) + eps)
    if checkpoint and best is not None:
        for tensor, saved in zip(params.tensors(), best, strict=True):
            tensor[:] = saved
        params.bb = best_bb
    return params, train_curve, val_curve


def irt_probs(params_or_head: _Params | IrtHead, test_vecs: np.ndarray) -> np.ndarray:
    """P(correct) [T, M] from either a training-view _Params or a fitted IrtHead."""
    if isinstance(params_or_head, IrtHead):
        return np.stack([params_or_head.predict(v) for v in test_vecs])
    p = params_or_head
    hidden = np.maximum(test_vecs @ p.w1.T + p.b1, 0.0)  # [T, H]
    alpha = hidden @ p.wa.T + p.ba  # [T, dim]
    beta = hidden @ p.wb + p.bb  # [T]
    logits = alpha @ p.theta.T - beta[:, None]
    return 1.0 / (1.0 + np.exp(-logits))


def fit_logistic(
    queries: np.ndarray,
    model_index: np.ndarray,
    labels: np.ndarray,
    n_models: int,
    *,
    val: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    seed: int = 42,
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    checkpoint: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[float], list[float]]:
    """Shnitzer-style per-model logistic regression (soft-label BCE, full-batch Adam).

    Returns (W [M, D], b [M], train_curve, val_curve). Identical optimizer settings to the
    IRT trainer so the comparison isolates the MODEL class, not the training recipe.
    `checkpoint=True` returns the weights from the best-val epoch.
    """
    rng = np.random.default_rng(seed)
    weights = rng.normal(0, 1.0 / np.sqrt(queries.shape[1]), (n_models, queries.shape[1]))
    bias = np.zeros(n_models)
    mw, vw = np.zeros_like(weights), np.zeros_like(weights)
    mb, vb = np.zeros_like(bias), np.zeros_like(bias)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    train_curve, val_curve = [], []
    best: tuple[np.ndarray, np.ndarray] | None = None

    def forward(q: np.ndarray, mi: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(np.sum(q * weights[mi], axis=1) + bias[mi])))

    pairs = len(labels)
    for step in range(1, epochs + 1):
        probs = forward(queries, model_index)
        train_curve.append(bce(probs, labels))
        if val is not None:
            vloss = bce(forward(val[0], val[1]), val[2])
            if checkpoint and (not val_curve or vloss < min(val_curve)):
                best = (weights.copy(), bias.copy())
            val_curve.append(vloss)
        dlogit = (probs - labels) / pairs
        gw = np.zeros_like(weights)
        np.add.at(gw, model_index, dlogit[:, None] * queries)
        gb = np.zeros_like(bias)
        np.add.at(gb, model_index, dlogit)
        gw += weight_decay * weights
        mw = beta1 * mw + (1 - beta1) * gw
        vw = beta2 * vw + (1 - beta2) * gw**2
        weights -= lr * (mw / (1 - beta1**step)) / (np.sqrt(vw / (1 - beta2**step)) + eps)
        mb = beta1 * mb + (1 - beta1) * gb
        vb = beta2 * vb + (1 - beta2) * gb**2
        bias -= lr * (mb / (1 - beta1**step)) / (np.sqrt(vb / (1 - beta2**step)) + eps)
    if checkpoint and best is not None:
        weights, bias = best
    return weights, bias, train_curve, val_curve


def logistic_probs(weights: np.ndarray, bias: np.ndarray, test_vecs: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(test_vecs @ weights.T + bias)))


def knn_probs(
    matrix: OutcomeMatrix,
    fit_ids: list[str],
    fit_vecs: np.ndarray,
    test_vecs: np.ndarray,
    *,
    k: int = 50,
) -> np.ndarray:
    """kNN-P: predicted P = similarity-weighted mean reward over the k nearest fit scenarios.

    Missing (neighbor, model) cells fall back to the model's global fit mean, so every model
    gets a prediction on every query (same coverage the learned heads have).
    """
    model_names = matrix.model_names()
    cell: dict[tuple[str, str], list[float]] = {}
    for o in matrix.outcomes:
        if o.reward is not None:
            cell.setdefault((o.scenario_id, o.model), []).append(o.reward)
    fit_set = set(fit_ids)
    global_mean = np.zeros(len(model_names))
    for mi, m in enumerate(model_names):
        vals = [np.mean(v) for (sid, mm), v in cell.items() if mm == m and sid in fit_set]
        global_mean[mi] = float(np.mean(vals)) if vals else 0.5
    # [F, M] reward table with NaN for missing cells.
    table = np.full((len(fit_ids), len(model_names)), np.nan)
    for fi, sid in enumerate(fit_ids):
        for mi, m in enumerate(model_names):
            vals = cell.get((sid, m))
            if vals:
                table[fi, mi] = float(np.mean(vals))
    sims = test_vecs @ fit_vecs.T  # both L2-normalized (hashing embedder)
    k = min(k, len(fit_ids))
    probs = np.zeros((len(test_vecs), len(model_names)))
    for t in range(len(test_vecs)):
        top = np.argpartition(-sims[t], k - 1)[:k]
        w = np.maximum(sims[t][top], 0.0)
        for mi in range(len(model_names)):
            vals = table[top, mi]
            mask = ~np.isnan(vals)
            probs[t, mi] = (
                float(np.sum(w[mask] * vals[mask]) / np.sum(w[mask]))
                if mask.any() and np.sum(w[mask]) > 0
                else global_mean[mi]
            )
    return probs


# ---------------------------------------------------------------- run recording


def record(
    *,
    matrix_name: str,
    matrix: OutcomeMatrix,
    variant: str,
    params: dict,
    split_seed: int,
    fit_ids: list[str],
    test_ids: list[str],
    picks: dict[str, str],
    best_eval: ChoiceEval,
    best_name: str,
    oracle_acc: float,
    notes: str = "",
) -> RunRecord:
    result = evaluate_choices(matrix, test_ids, lambda sid: picks[sid])
    rec = RunRecord(
        run_id=f"{matrix_name}-{variant}-{uuid.uuid4().hex[:8]}",
        ts=datetime.now(tz=UTC).isoformat(),
        matrix=matrix_name,
        variant=variant,
        params={**params, "embed": EMBED_KIND},
        split_seed=split_seed,
        fit_scenarios=len(fit_ids),
        test_scenarios=len(test_ids),
        result=result,
        baselines={"best_single": best_eval},
        notes=f"best_single={best_name}; oracle acc={oracle_acc:.4f}; {notes}".strip("; "),
    )
    append_run(rec, RUNS)
    logger.info(
        "%s/%s seed%d %s: acc=%.4f cost=$%.5f (best-single %.4f @ $%.5f)",
        matrix_name,
        variant,
        split_seed,
        params,
        result.accuracy,
        result.cost_per_call,
        best_eval.accuracy,
        best_eval.cost_per_call,
    )
    return rec


EMBED_KIND = "hashing"  # set to "oai" by --embed oai (uses r1's 3-large disk cache)


def cached_oai_vecs(name: str, matrix: OutcomeMatrix) -> dict[str, np.ndarray]:
    """R1's cached text-embedding-3-large vectors, keyed by scenario id (L2-normalized).

    Cache rows follow first-seen scenario order in matrix.outcomes (r1's MatrixContext
    convention, same matrix JSONs), verified by shape match against scenario count.
    """
    path = DATA / "cache" / f"{name}-oai3l-tasks.npy"
    if not path.exists():
        raise SystemExit(f"no oai cache for {name} at {path}")
    arr = np.load(path)
    sids = matrix.scenario_ids()
    if arr.shape[0] != len(sids):
        raise SystemExit(f"cache rows {arr.shape[0]} != scenarios {len(sids)} for {name}")
    arr = arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
    return dict(zip(sids, arr, strict=True))


class SplitContext:
    """Everything derived from one (matrix, seed) split, computed once."""

    def __init__(self, name: str, matrix: OutcomeMatrix, seed: int) -> None:
        self.name, self.matrix, self.seed = name, matrix, seed
        self.fit_ids, self.test_ids = split_scenario_ids(matrix, train_fraction=0.7, seed=seed)
        if EMBED_KIND == "oai":
            by_sid = cached_oai_vecs(name, matrix)
            self.fit_vecs = np.stack([by_sid[s] for s in self.fit_ids])
            self.test_vecs = np.stack([by_sid[s] for s in self.test_ids])
        else:
            tasks = scenario_tasks(matrix)
            self.fit_vecs = embed([tasks[s] for s in self.fit_ids])
            self.test_vecs = embed([tasks[s] for s in self.test_ids])
        self.best_name, _, _ = best_single_model(
            matrix, fit_ids=self.fit_ids, eval_ids=self.test_ids
        )
        self.best_eval = evaluate_choices(matrix, self.test_ids, lambda _sid: self.best_name)
        self.oracle_acc, _ = oracle(matrix, self.test_ids)
        self.costs = mean_costs(matrix, self.fit_ids)

    def rec(
        self,
        variant: str,
        params: dict,
        probs: np.ndarray,
        lam: float,
        margin: float = MARGIN,
        notes: str = "",
    ) -> RunRecord:
        picks = guarded_picks(
            probs, self.matrix.model_names(), self.costs, self.best_name, lam, margin=margin
        )
        return record(
            matrix_name=self.name,
            matrix=self.matrix,
            variant=variant,
            params={**params, "lam": lam},
            split_seed=self.seed,
            fit_ids=self.fit_ids,
            test_ids=self.test_ids,
            picks=dict(zip(self.test_ids, picks, strict=True)),
            best_eval=self.best_eval,
            best_name=self.best_name,
            oracle_acc=self.oracle_acc,
            notes=notes,
        )


# ---------------------------------------------------------------- subcommands


def cmd_repro(matrices: dict[str, OutcomeMatrix], seeds: list[int]) -> None:
    """Reproduce the champion irt config through wmo.optimize.irt.fit_irt_head."""
    for name, matrix in matrices.items():
        for seed in seeds:
            ctx = SplitContext(name, matrix, seed)
            head = fit_irt_head(
                matrix,
                scenario_ids=ctx.fit_ids,
                embeddings=ctx.fit_vecs,
                seed=42,
                epochs=300,
                hidden=256,
                dim=64,
            )
            probs = irt_probs(head, ctx.test_vecs)
            for lam in LAMS:
                ctx.rec("r3-irt-repro", {"hidden": 256, "dim": 64, "epochs": 300}, probs, lam)


def cmd_baselines(matrices: dict[str, OutcomeMatrix], seeds: list[int]) -> None:
    """Logistic-per-model + kNN-P vs irt, same embeddings, same decision rule."""
    for name, matrix in matrices.items():
        n_models = len(matrix.model_names())
        for seed in seeds:
            ctx = SplitContext(name, matrix, seed)
            q, mi, y = pairs_for(matrix, ctx.fit_ids, ctx.fit_vecs)
            weights, bias, tc, _ = fit_logistic(q, mi, y, n_models)
            l_probs = logistic_probs(weights, bias, ctx.test_vecs)
            k_probs = knn_probs(matrix, ctx.fit_ids, ctx.fit_vecs, ctx.test_vecs, k=50)
            for lam in LAMS:
                ctx.rec(
                    "r3-logistic",
                    {"epochs": 300},
                    l_probs,
                    lam,
                    notes=f"final train bce={tc[-1]:.4f}",
                )
                ctx.rec("r3-knn", {"k": 50}, k_probs, lam)


def cmd_audit(matrices: dict[str, OutcomeMatrix], seeds: list[int]) -> None:
    """Leak audits: id overlap, duplicate task text, shuffled-label control."""
    for name, matrix in matrices.items():
        tasks = scenario_tasks(matrix)
        for seed in seeds:
            ctx = SplitContext(name, matrix, seed)
            overlap = set(ctx.fit_ids) & set(ctx.test_ids)
            fit_texts = {tasks[s] for s in ctx.fit_ids}
            dup = [s for s in ctx.test_ids if tasks[s] in fit_texts]
            logger.info(
                "%s seed%d AUDIT: id-overlap=%d dup-task-text=%d/%d",
                name,
                seed,
                len(overlap),
                len(dup),
                len(ctx.test_ids),
            )
            # Shuffled-label control: permute rewards within each model across fit
            # scenarios. Any residual "routing skill" is leakage; the guard should
            # collapse picks to ~best-single.
            rng = np.random.default_rng(seed)
            fit_set = set(ctx.fit_ids)
            by_model: dict[str, list[int]] = {}
            for i, o in enumerate(matrix.outcomes):
                if o.scenario_id in fit_set and o.reward is not None:
                    by_model.setdefault(o.model, []).append(i)
            shuffled = [o.model_copy() for o in matrix.outcomes]
            for _m, idxs in by_model.items():
                rewards = [matrix.outcomes[i].reward for i in idxs]
                perm = rng.permutation(len(idxs))
                for j, i in enumerate(idxs):
                    shuffled[i].reward = rewards[perm[j]]
            sh_matrix = OutcomeMatrix(pool=matrix.pool, outcomes=shuffled)
            head = fit_irt_head(
                sh_matrix,
                scenario_ids=ctx.fit_ids,
                embeddings=ctx.fit_vecs,
                seed=42,
                epochs=300,
                hidden=256,
                dim=64,
            )
            probs = irt_probs(head, ctx.test_vecs)
            sh_costs = mean_costs(sh_matrix, ctx.fit_ids)
            sh_best, _, _ = best_single_model(sh_matrix, fit_ids=ctx.fit_ids, eval_ids=ctx.test_ids)
            picks = guarded_picks(probs, matrix.model_names(), sh_costs, sh_best, 0.0)
            base_share = sum(1 for p in picks if p == sh_best) / len(picks)
            # Evaluate on the REAL matrix (test labels untouched by the control).
            record(
                matrix_name=name,
                matrix=matrix,
                variant="r3-irt-shuffled-control",
                params={"hidden": 256, "dim": 64, "epochs": 300, "lam": 0.0},
                split_seed=seed,
                fit_ids=ctx.fit_ids,
                test_ids=ctx.test_ids,
                picks=dict(zip(ctx.test_ids, picks, strict=True)),
                best_eval=ctx.best_eval,
                best_name=ctx.best_name,
                oracle_acc=ctx.oracle_acc,
                notes=f"shuffled-label control; shuffled-best={sh_best} "
                f"baseline-share={base_share:.2f}",
            )


FRONTIER_LAMS = [0.0, 0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2]


def family_predictor(ctx: SplitContext, family: str) -> tuple:
    """Fit one family on the split's FULL fit side; return (predict(vecs)->P, params)."""
    matrix, n_models = ctx.matrix, len(ctx.matrix.model_names())
    if family == "irt":
        head = fit_irt_head(
            matrix,
            scenario_ids=ctx.fit_ids,
            embeddings=ctx.fit_vecs,
            seed=42,
            epochs=300,
            hidden=256,
            dim=64,
        )
        return lambda vecs: irt_probs(head, vecs), {"hidden": 256, "dim": 64, "epochs": 300}
    if family == "logistic":
        q, mi, y = pairs_for(matrix, ctx.fit_ids, ctx.fit_vecs)
        weights, bias, _, _ = fit_logistic(q, mi, y, n_models)
        return lambda vecs: logistic_probs(weights, bias, vecs), {"epochs": 300}
    if family == "knn":
        return (
            lambda vecs: knn_probs(matrix, ctx.fit_ids, ctx.fit_vecs, vecs, k=50),
            {"k": 50},
        )
    raise SystemExit(f"unknown family {family}")


def family_probs(ctx: SplitContext, family: str) -> tuple[np.ndarray, dict]:
    """Fit one predictor family on the split and return (P [T, M] on test, params)."""
    predict, params = family_predictor(ctx, family)
    return predict(ctx.test_vecs), params


def scored_triples(
    matrix: OutcomeMatrix, ids: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(scenario_row, model_row, label) for every scored outcome in `ids`."""
    row_of = {sid: i for i, sid in enumerate(ids)}
    model_row = {m: i for i, m in enumerate(matrix.model_names())}
    rows, mis, ys = [], [], []
    for o in matrix.outcomes:
        if o.reward is not None and o.scenario_id in row_of:
            rows.append(row_of[o.scenario_id])
            mis.append(model_row[o.model])
            ys.append(o.reward)
    return np.asarray(rows), np.asarray(mis), np.asarray(ys, dtype=np.float64)


def cmd_frontier(matrices: dict[str, OutcomeMatrix], seeds: list[int]) -> None:
    """Dense lam sweep per family: the honest frontier comparison (same-lam is not)."""
    for name, matrix in matrices.items():
        for seed in seeds:
            ctx = SplitContext(name, matrix, seed)
            for family in ["irt", "logistic", "knn"]:
                probs, params = family_probs(ctx, family)
                for lam in FRONTIER_LAMS:
                    ctx.rec(f"r3-{family}-frontier", params, probs, lam)


def cmd_calibration(matrices: dict[str, OutcomeMatrix], seeds: list[int]) -> None:
    """Reliability of predicted P per family: 10-bin table + ECE on test cells."""
    for name, matrix in matrices.items():
        model_names = matrix.model_names()
        cell: dict[tuple[str, str], list[float]] = {}
        for o in matrix.outcomes:
            if o.reward is not None:
                cell.setdefault((o.scenario_id, o.model), []).append(o.reward)
        for seed in seeds:
            ctx = SplitContext(name, matrix, seed)
            for family in ["irt", "logistic", "knn"]:
                probs, _ = family_probs(ctx, family)
                preds, actuals = [], []
                for t, sid in enumerate(ctx.test_ids):
                    for mi, m in enumerate(model_names):
                        vals = cell.get((sid, m))
                        if vals:
                            preds.append(probs[t, mi])
                            actuals.append(float(np.mean(vals)))
                preds_a, actuals_a = np.asarray(preds), np.asarray(actuals)
                bins = np.clip((preds_a * 10).astype(int), 0, 9)
                rows, ece = [], 0.0
                for b in range(10):
                    mask = bins == b
                    if not mask.any():
                        continue
                    gap = abs(preds_a[mask].mean() - actuals_a[mask].mean())
                    ece += mask.mean() * gap
                    rows.append(
                        f"[{b / 10:.1f},{(b + 1) / 10:.1f}) n={mask.sum():4d} "
                        f"pred={preds_a[mask].mean():.3f} actual={actuals_a[mask].mean():.3f}"
                    )
                logger.info(
                    "%s seed%d %s: ECE=%.4f\n  %s", name, seed, family, ece, "\n  ".join(rows)
                )


def val_split(ctx: SplitContext, frac: float = 0.15, seed: int = 7) -> tuple[list[int], list[int]]:
    """Deterministic train/val row split WITHIN fit ids (for selection + Platt scaling)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ctx.fit_ids))
    n_val = max(1, int(len(ctx.fit_ids) * frac))
    return sorted(perm[n_val:].tolist()), sorted(perm[:n_val].tolist())


def platt(
    logits: np.ndarray, labels: np.ndarray, *, epochs: int = 500, lr: float = 0.1
) -> tuple[float, float]:
    """Fit sigmoid(a*logit + b) by BCE gradient descent (two scalars, cannot overfit much)."""
    a, b = 1.0, 0.0
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-(a * logits + b)))
        d = (p - labels) / len(labels)
        a -= lr * float(d @ logits)
        b -= lr * float(d.sum())
    return a, b


def sweep_ece(preds: np.ndarray, actuals: np.ndarray) -> float:
    bins = np.clip((preds * 10).astype(int), 0, 9)
    ece = 0.0
    for b in range(10):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(preds[mask].mean() - actuals[mask].mean())
    return float(ece)


SWEEP_GRID = [
    {"hidden": h, "dim": d, "weight_decay": wd}
    for h in [64, 256]
    for d in [16, 64]
    for wd in [1e-4, 1e-3, 1e-2]
]


def cmd_sweep(matrices: dict[str, OutcomeMatrix], seeds: list[int]) -> None:
    """Q1 capacity/reg sweep with train+val curves; select by val BCE at best epoch."""
    for name, matrix in matrices.items():
        n_models = len(matrix.model_names())
        for seed in seeds:
            ctx = SplitContext(name, matrix, seed)
            tr_rows, va_rows = val_split(ctx)
            tr_ids = [ctx.fit_ids[i] for i in tr_rows]
            va_ids = [ctx.fit_ids[i] for i in va_rows]
            tr_vecs, va_vecs = ctx.fit_vecs[tr_rows], ctx.fit_vecs[va_rows]
            q, mi, y = pairs_for(matrix, tr_ids, tr_vecs)
            vq, vmi, vy = pairs_for(matrix, va_ids, va_vecs)
            for cfg in SWEEP_GRID:
                params, tc, vc = fit_irt_cfg(
                    q, mi, y, n_models, (vq, vmi, vy), cfg, checkpoint=False, epochs=300
                )
                best_ep = int(np.argmin(vc)) + 1
                logger.info(
                    "%s seed%d irt %s: train %.4f->%.4f | val %.4f min=%.4f@ep%d final=%.4f",
                    name,
                    seed,
                    cfg,
                    tc[0],
                    tc[-1],
                    vc[0],
                    min(vc),
                    best_ep,
                    vc[-1],
                )


def cmd_tiebreak(matrices: dict[str, OutcomeMatrix], seeds: list[int]) -> None:
    """Eps-cheapest routing: cheapest model with P >= maxP - eps (Platt-scaled P).

    The principled version of 'statistical tie goes to the cheaper model'. Platt scaling is
    fitted per family on a 15%-of-fit val split, so absolute P is trustworthy enough to
    threshold. Sweeps eps; also records an abstention-free coverage variant per eps.
    """
    eps_grid = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08]
    for name, matrix in matrices.items():
        model_names = matrix.model_names()
        for seed in seeds:
            ctx = SplitContext(name, matrix, seed)
            _, va_rows = val_split(ctx)
            va_ids = [ctx.fit_ids[i] for i in va_rows]
            va_vecs = ctx.fit_vecs[va_rows]
            v_rows, v_mis, v_ys = scored_triples(matrix, va_ids)
            order = np.argsort([ctx.costs.get(m, 1.0) for m in model_names])
            for family in ["irt", "logistic", "knn"]:
                predict, params = family_predictor(ctx, family)
                # Platt on the val split. NOTE: the family was fitted on FULL fit (val rows
                # seen in training), which understates miscalibration; a clean version
                # refits on train-minus-val. Recorded in params as platt_leaky.
                va_probs = predict(va_vecs)
                v_pred = np.clip(va_probs[v_rows, v_mis], 1e-6, 1 - 1e-6)
                a, b_ = platt(np.log(v_pred / (1 - v_pred)), v_ys)
                probs = np.clip(predict(ctx.test_vecs), 1e-6, 1 - 1e-6)
                cal = 1.0 / (1.0 + np.exp(-(a * np.log(probs / (1 - probs)) + b_)))
                for eps in eps_grid:
                    picks = []
                    for t in range(len(ctx.test_ids)):
                        cutoff = cal[t].max() - eps
                        picks.append(model_names[int(order[np.argmax(cal[t][order] >= cutoff)])])
                    record(
                        matrix_name=name,
                        matrix=matrix,
                        variant=f"r3-{family}-eps",
                        params={**params, "eps": eps, "platt_leaky": [round(a, 3), round(b_, 3)]},
                        split_seed=seed,
                        fit_ids=ctx.fit_ids,
                        test_ids=ctx.test_ids,
                        picks=dict(zip(ctx.test_ids, picks, strict=True)),
                        best_eval=ctx.best_eval,
                        best_name=ctx.best_name,
                        oracle_acc=ctx.oracle_acc,
                    )


IRT2 = {"hidden": 64, "dim": 16, "weight_decay": 1e-3, "epochs": 300}


def fit_irt_cfg(
    q: np.ndarray,
    mi: np.ndarray,
    y: np.ndarray,
    n_models: int,
    val: tuple[np.ndarray, np.ndarray, np.ndarray],
    cfg: dict,
    checkpoint: bool = True,
    epochs: int | None = None,
) -> tuple[_Params, list[float], list[float]]:
    """fit_irt_curves with a config dict, coerced to concrete types for the type gate."""
    return fit_irt_curves(
        q,
        mi,
        y,
        n_models,
        val=val,
        checkpoint=checkpoint,
        hidden=int(cfg.get("hidden", 256)),
        dim=int(cfg.get("dim", 64)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        epochs=int(epochs if epochs is not None else cfg.get("epochs", 300)),
    )


def _fit_irt2(ctx: SplitContext, matrix: OutcomeMatrix) -> tuple[np.ndarray, float, float, dict]:
    """Early-stopped + Platt-calibrated irt on the split; returns (test P, val bce, ece, meta).

    Protocol: 85/15 train/val inside fit; checkpoint at best-val epoch; Platt scale on the
    val split (2 params, val was held out of training so this is clean); predict on test.
    """
    tr_rows, va_rows = val_split(ctx)
    tr_ids = [ctx.fit_ids[i] for i in tr_rows]
    va_ids = [ctx.fit_ids[i] for i in va_rows]
    tr_vecs, va_vecs = ctx.fit_vecs[tr_rows], ctx.fit_vecs[va_rows]
    q, mi, y = pairs_for(matrix, tr_ids, tr_vecs)
    vq, vmi, vy = pairs_for(matrix, va_ids, va_vecs)
    params, _tc, vc = fit_irt_cfg(q, mi, y, len(matrix.model_names()), (vq, vmi, vy), IRT2)
    v_rows, v_mis, v_ys = scored_triples(matrix, va_ids)
    va_probs = np.clip(irt_probs(params, va_vecs), 1e-6, 1 - 1e-6)
    v_pred = va_probs[v_rows, v_mis]
    a, b_ = platt(np.log(v_pred / (1 - v_pred)), v_ys)
    test_p = np.clip(irt_probs(params, ctx.test_vecs), 1e-6, 1 - 1e-6)
    cal = 1.0 / (1.0 + np.exp(-(a * np.log(test_p / (1 - test_p)) + b_)))
    meta = {**IRT2, "best_epoch": int(np.argmin(vc)) + 1, "platt": [round(a, 3), round(b_, 3)]}
    return cal, float(min(vc)), 0.0, meta


def cmd_irt2(matrices: dict[str, OutcomeMatrix], seeds: list[int]) -> None:
    """The fixed IRT: early stop at best val + Platt. vs matched early-stopped logistic."""
    for name, matrix in matrices.items():
        model_names = matrix.model_names()
        cell: dict[tuple[str, str], list[float]] = {}
        for o in matrix.outcomes:
            if o.reward is not None:
                cell.setdefault((o.scenario_id, o.model), []).append(o.reward)
        for seed in seeds:
            ctx = SplitContext(name, matrix, seed)
            tr_rows, va_rows = val_split(ctx)
            tr_ids = [ctx.fit_ids[i] for i in tr_rows]
            va_ids = [ctx.fit_ids[i] for i in va_rows]
            tr_vecs, va_vecs = ctx.fit_vecs[tr_rows], ctx.fit_vecs[va_rows]
            q, mi, y = pairs_for(matrix, tr_ids, tr_vecs)
            vq, vmi, vy = pairs_for(matrix, va_ids, va_vecs)
            v_rows, v_mis, v_ys = scored_triples(matrix, va_ids)
            order = np.argsort([ctx.costs.get(m, 1.0) for m in model_names])

            def calibrated(
                raw_va: np.ndarray,
                raw_te: np.ndarray,
                v_rows: np.ndarray = v_rows,
                v_mis: np.ndarray = v_mis,
                v_ys: np.ndarray = v_ys,
            ) -> tuple[np.ndarray, list]:
                v_pred = np.clip(raw_va[v_rows, v_mis], 1e-6, 1 - 1e-6)
                a, b_ = platt(np.log(v_pred / (1 - v_pred)), v_ys)
                te = np.clip(raw_te, 1e-6, 1 - 1e-6)
                return 1.0 / (1.0 + np.exp(-(a * np.log(te / (1 - te)) + b_))), [
                    round(a, 3),
                    round(b_, 3),
                ]

            def test_ece(
                probs: np.ndarray,
                ctx: SplitContext = ctx,
                model_names: list[str] = model_names,
                cell: dict = cell,
            ) -> float:
                preds, acts = [], []
                for t, sid in enumerate(ctx.test_ids):
                    for mj, m in enumerate(model_names):
                        vals = cell.get((sid, m))
                        if vals:
                            preds.append(probs[t, mj])
                            acts.append(float(np.mean(vals)))
                return sweep_ece(np.asarray(preds), np.asarray(acts))

            heads = {}
            params, _tc, vc = fit_irt_cfg(q, mi, y, len(model_names), (vq, vmi, vy), IRT2)
            heads["irt2"] = (
                irt_probs(params, va_vecs),
                irt_probs(params, ctx.test_vecs),
                {**IRT2, "best_epoch": int(np.argmin(vc)) + 1, "val_bce": round(min(vc), 4)},
            )
            w, b2, _tc2, vc2 = fit_logistic(
                q, mi, y, len(model_names), val=(vq, vmi, vy), checkpoint=True
            )
            heads["logistic2"] = (
                logistic_probs(w, b2, va_vecs),
                logistic_probs(w, b2, ctx.test_vecs),
                {"best_epoch": int(np.argmin(vc2)) + 1, "val_bce": round(min(vc2), 4)},
            )
            for fam, (raw_va, raw_te, meta) in heads.items():
                cal, ab = calibrated(raw_va, raw_te)
                logger.info(
                    "%s seed%d %s: val_bce=%.4f best_ep=%d ECE raw=%.4f platt=%.4f",
                    name,
                    seed,
                    fam,
                    meta["val_bce"],
                    meta["best_epoch"],
                    test_ece(raw_te),
                    test_ece(cal),
                )
                for lam in [0.0, 0.01, 0.02]:
                    ctx.rec(f"r3-{fam}", {**meta, "platt": ab}, cal, lam)
                for eps in [0.0, 0.01, 0.02, 0.03]:
                    picks = []
                    for t in range(len(ctx.test_ids)):
                        cutoff = cal[t].max() - eps
                        picks.append(model_names[int(order[np.argmax(cal[t][order] >= cutoff)])])
                    record(
                        matrix_name=name,
                        matrix=matrix,
                        variant=f"r3-{fam}-eps",
                        params={**meta, "eps": eps, "platt": ab},
                        split_seed=seed,
                        fit_ids=ctx.fit_ids,
                        test_ids=ctx.test_ids,
                        picks=dict(zip(ctx.test_ids, picks, strict=True)),
                        best_eval=ctx.best_eval,
                        best_name=ctx.best_name,
                        oracle_acc=ctx.oracle_acc,
                    )


def cmd_audit2(matrices: dict[str, OutcomeMatrix], seeds: list[int]) -> None:
    """Shuffled-label control for the FIXED irt (early stop + Platt): must collapse now."""
    for name, matrix in matrices.items():
        for seed in seeds:
            ctx = SplitContext(name, matrix, seed)
            rng = np.random.default_rng(seed)
            fit_set = set(ctx.fit_ids)
            by_model: dict[str, list[int]] = {}
            for i, o in enumerate(matrix.outcomes):
                if o.scenario_id in fit_set and o.reward is not None:
                    by_model.setdefault(o.model, []).append(i)
            shuffled = [o.model_copy() for o in matrix.outcomes]
            for _m, idxs in by_model.items():
                rewards = [matrix.outcomes[i].reward for i in idxs]
                perm = rng.permutation(len(idxs))
                for j, i in enumerate(idxs):
                    shuffled[i].reward = rewards[perm[j]]
            sh_matrix = OutcomeMatrix(pool=matrix.pool, outcomes=shuffled)
            cal, val_bce, _, meta = _fit_irt2(ctx, sh_matrix)
            picks = guarded_picks(cal, matrix.model_names(), ctx.costs, ctx.best_name, 0.0)
            base_share = sum(1 for p in picks if p == ctx.best_name) / len(picks)
            record(
                matrix_name=name,
                matrix=matrix,
                variant="r3-irt2-shuffled-control",
                params={**meta, "lam": 0.0},
                split_seed=seed,
                fit_ids=ctx.fit_ids,
                test_ids=ctx.test_ids,
                picks=dict(zip(ctx.test_ids, picks, strict=True)),
                best_eval=ctx.best_eval,
                best_name=ctx.best_name,
                oracle_acc=ctx.oracle_acc,
                notes=f"shuffled-label control irt2; baseline-share={base_share:.2f} "
                f"val_bce={val_bce:.4f}",
            )


def _platt_apply(a: float, b_: float, raw: np.ndarray) -> np.ndarray:
    clipped = np.clip(raw, 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-(a * np.log(clipped / (1 - clipped)) + b_)))


def cmd_ensemble(matrices: dict[str, OutcomeMatrix], seeds: list[int]) -> None:
    """Round 2a: blend Platt'd irt2 P with Platt'd kNN-P, weight chosen on the val split.

    kNN val predictions use a train-only neighbor store (no self-neighbor leak); kNN test
    predictions use the full fit store. When r1's 3-large cache exists for the matrix, a
    second variant runs the kNN side on those vectors (irt2 stays on hashing: round 1
    showed semantic vectors do nothing for the parametric head but +1pt for retrieval).
    """
    for name, matrix in matrices.items():
        model_names = matrix.model_names()
        n_fit_margin = 0.5  # n-aware guard c
        for seed in seeds:
            ctx = SplitContext(name, matrix, seed)
            tr_rows, va_rows = val_split(ctx)
            tr_ids = [ctx.fit_ids[i] for i in tr_rows]
            va_ids = [ctx.fit_ids[i] for i in va_rows]
            tr_vecs, va_vecs = ctx.fit_vecs[tr_rows], ctx.fit_vecs[va_rows]
            q, mi, y = pairs_for(matrix, tr_ids, tr_vecs)
            vq, vmi, vy = pairs_for(matrix, va_ids, va_vecs)
            v_rows, v_mis, v_ys = scored_triples(matrix, va_ids)
            margin = max(MARGIN, n_fit_margin / np.sqrt(len(ctx.fit_ids)))

            params, _tc, vc = fit_irt_cfg(q, mi, y, len(model_names), (vq, vmi, vy), IRT2)
            irt_va_raw = irt_probs(params, va_vecs)
            irt_te_raw = irt_probs(params, ctx.test_vecs)
            v_pred = np.clip(irt_va_raw[v_rows, v_mis], 1e-6, 1 - 1e-6)
            a_i, b_i = platt(np.log(v_pred / (1 - v_pred)), v_ys)
            irt_va = _platt_apply(a_i, b_i, irt_va_raw)
            irt_te = _platt_apply(a_i, b_i, irt_te_raw)

            knn_sides: list[tuple[str, np.ndarray, np.ndarray]] = []
            knn_va_raw = knn_probs(matrix, tr_ids, tr_vecs, va_vecs, k=50)
            knn_te_raw = knn_probs(matrix, ctx.fit_ids, ctx.fit_vecs, ctx.test_vecs, k=50)
            knn_sides.append(("hashing", knn_va_raw, knn_te_raw))
            oai_cache = DATA / "cache" / f"{name}-oai3l-tasks.npy"
            if oai_cache.exists() and EMBED_KIND != "oai":
                by_sid = cached_oai_vecs(name, matrix)
                fit_o = np.stack([by_sid[s] for s in ctx.fit_ids])
                te_o = np.stack([by_sid[s] for s in ctx.test_ids])
                knn_sides.append(
                    (
                        "oai",
                        knn_probs(matrix, tr_ids, fit_o[tr_rows], fit_o[va_rows], k=50),
                        knn_probs(matrix, ctx.fit_ids, fit_o, te_o, k=50),
                    )
                )
            for knn_embed, kva_raw, kte_raw in knn_sides:
                kv_pred = np.clip(kva_raw[v_rows, v_mis], 1e-6, 1 - 1e-6)
                a_k, b_k = platt(np.log(kv_pred / (1 - kv_pred)), v_ys)
                knn_va = _platt_apply(a_k, b_k, kva_raw)
                knn_te = _platt_apply(a_k, b_k, kte_raw)
                grid = [round(w * 0.1, 1) for w in range(11)]
                val_bces = {
                    w: bce((w * irt_va + (1 - w) * knn_va)[v_rows, v_mis], v_ys) for w in grid
                }
                w_best = min(val_bces, key=lambda w: val_bces[w])
                logger.info(
                    "%s seed%d ens(knn=%s): w*=%.1f val_bce=%.4f (irt-only %.4f, knn-only %.4f)",
                    name,
                    seed,
                    knn_embed,
                    w_best,
                    val_bces[w_best],
                    val_bces[1.0],
                    val_bces[0.0],
                )
                blend_te = w_best * irt_te + (1 - w_best) * knn_te
                for lam in [0.0, 0.01, 0.02]:
                    ctx.rec(
                        "r3-ens",
                        {
                            "w": w_best,
                            "knn_embed": knn_embed,
                            "val_bce": round(val_bces[w_best], 4),
                            "margin": round(float(margin), 4),
                        },
                        blend_te,
                        lam,
                        margin=float(margin),
                    )


def cmd_coverage(matrices: dict[str, OutcomeMatrix], seeds: list[int]) -> None:
    """Round 2b: accuracy-coverage curves; confidence = score gap over the baseline.

    Route the top-q most-confident fraction by raw argmax(P - lam*penalty); abstain the
    rest to best-single. No margin inside the routed set: coverage IS the guard here.
    """
    qs = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    for name, matrix in matrices.items():
        model_names = matrix.model_names()
        for seed in seeds:
            ctx = SplitContext(name, matrix, seed)
            cal, _val_bce, _e, meta = _fit_irt2(ctx, matrix)
            cost_scale = sum(ctx.costs.values()) / len(ctx.costs)
            penalties = np.asarray([ctx.costs.get(m, cost_scale) / cost_scale for m in model_names])
            base_idx = model_names.index(ctx.best_name)
            for lam in [0.01, 0.02]:
                scores = cal - lam * penalties
                pick_idx = np.argmax(scores, axis=1)
                gap = scores[np.arange(len(scores)), pick_idx] - scores[:, base_idx]
                order = np.argsort(-gap)
                for cov in qs:
                    routed = set(order[: int(round(cov * len(order)))].tolist())
                    picks = [
                        model_names[int(pick_idx[t])] if t in routed else ctx.best_name
                        for t in range(len(ctx.test_ids))
                    ]
                    record(
                        matrix_name=name,
                        matrix=matrix,
                        variant="r3-cov",
                        params={**meta, "lam": lam, "coverage": cov},
                        split_seed=seed,
                        fit_ids=ctx.fit_ids,
                        test_ids=ctx.test_ids,
                        picks=dict(zip(ctx.test_ids, picks, strict=True)),
                        best_eval=ctx.best_eval,
                        best_name=ctx.best_name,
                        oracle_acc=ctx.oracle_acc,
                    )


def cmd_nested(matrices: dict[str, OutcomeMatrix], seeds: list[int]) -> None:
    """Round 3a: coverage chosen on the VAL split (nested; kills the post-hoc caveat).

    Selection rule, self-gating by design: for each lam, pick the LARGEST coverage q whose
    paired val delta (routed reward minus baseline reward, per val scenario) has
    mean - z*SE >= 0 (z=1). Tiny val splits (wm corpora) produce huge SEs, so q*=0 and the
    router reverts to best-single by construction; no separate n-floor constant needed.
    """
    qs = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    for name, matrix in matrices.items():
        model_names = matrix.model_names()
        cell: dict[tuple[str, str], list[float]] = {}
        for o in matrix.outcomes:
            if o.reward is not None:
                cell.setdefault((o.scenario_id, o.model), []).append(o.reward)

        def cell_mean(sid: str, model: str, cell: dict = cell) -> float | None:
            vals = cell.get((sid, model))
            return float(np.mean(vals)) if vals else None

        for seed in seeds:
            ctx = SplitContext(name, matrix, seed)
            tr_rows, va_rows = val_split(ctx)
            tr_ids = [ctx.fit_ids[i] for i in tr_rows]
            va_ids = [ctx.fit_ids[i] for i in va_rows]
            tr_vecs, va_vecs = ctx.fit_vecs[tr_rows], ctx.fit_vecs[va_rows]
            q, mi, y = pairs_for(matrix, tr_ids, tr_vecs)
            vq, vmi, vy = pairs_for(matrix, va_ids, va_vecs)
            v_rows, v_mis, v_ys = scored_triples(matrix, va_ids)
            params, _tc, vc = fit_irt_cfg(q, mi, y, len(model_names), (vq, vmi, vy), IRT2)
            v_pred = np.clip(irt_probs(params, va_vecs)[v_rows, v_mis], 1e-6, 1 - 1e-6)
            a, b_ = platt(np.log(v_pred / (1 - v_pred)), v_ys)
            cal_va = _platt_apply(a, b_, irt_probs(params, va_vecs))
            cal_te = _platt_apply(a, b_, irt_probs(params, ctx.test_vecs))
            cost_scale = sum(ctx.costs.values()) / len(ctx.costs)
            pen = np.asarray([ctx.costs.get(m, cost_scale) / cost_scale for m in model_names])
            base_idx = model_names.index(ctx.best_name)
            for lam in [0.01, 0.02]:
                # Val-side confidence ordering and paired deltas.
                s_va = cal_va - lam * pen
                pick_va = np.argmax(s_va, axis=1)
                gap_va = s_va[np.arange(len(s_va)), pick_va] - s_va[:, base_idx]
                order_va = np.argsort(-gap_va)
                deltas = np.zeros(len(va_ids))
                known = np.zeros(len(va_ids), dtype=bool)
                for r, sid in enumerate(va_ids):
                    routed_r = cell_mean(sid, model_names[int(pick_va[r])])
                    base_r = cell_mean(sid, ctx.best_name)
                    if routed_r is not None and base_r is not None:
                        deltas[r] = routed_r - base_r
                        known[r] = True
                q_star = 0.0
                for cov in qs:
                    top = order_va[: int(round(cov * len(order_va)))]
                    d = deltas[top][known[top]]
                    if len(d) < 8:
                        continue  # too few paired val observations to certify
                    lcb = float(np.mean(d)) - float(np.std(d, ddof=1) / np.sqrt(len(d)))
                    if lcb >= 0:
                        q_star = cov
                # Apply q_star on test.
                s_te = cal_te - lam * pen
                pick_te = np.argmax(s_te, axis=1)
                gap_te = s_te[np.arange(len(s_te)), pick_te] - s_te[:, base_idx]
                order_te = np.argsort(-gap_te)
                routed = set(order_te[: int(round(q_star * len(order_te)))].tolist())
                picks = [
                    model_names[int(pick_te[t])] if t in routed else ctx.best_name
                    for t in range(len(ctx.test_ids))
                ]
                record(
                    matrix_name=name,
                    matrix=matrix,
                    variant="r3-cov-nested",
                    params={"lam": lam, "q_star": q_star, "val_n": int(known.sum())},
                    split_seed=seed,
                    fit_ids=ctx.fit_ids,
                    test_ids=ctx.test_ids,
                    picks=dict(zip(ctx.test_ids, picks, strict=True)),
                    best_eval=ctx.best_eval,
                    best_name=ctx.best_name,
                    oracle_acc=ctx.oracle_acc,
                )


def cmd_nested_cv(matrices: dict[str, OutcomeMatrix], seeds: list[int]) -> None:
    """Round 3a': coverage selection on 5-fold OUT-OF-FOLD predictions over the full fit.

    Same LCB rule as cmd_nested, but the selection set is every fit scenario's
    out-of-fold calibrated P (each fold predicted by a model trained on the other four,
    with its own inner 15% val for early stop + Platt), ~6.7x more selection data than
    the single val split. Final routing model = irt2 on the full fit (round-1 protocol).
    """
    qs = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    folds = 5
    for name, matrix in matrices.items():
        model_names = matrix.model_names()
        cell: dict[tuple[str, str], list[float]] = {}
        for o in matrix.outcomes:
            if o.reward is not None:
                cell.setdefault((o.scenario_id, o.model), []).append(o.reward)

        def cell_mean(sid: str, model: str, cell: dict = cell) -> float | None:
            vals = cell.get((sid, model))
            return float(np.mean(vals)) if vals else None

        for seed in seeds:
            ctx = SplitContext(name, matrix, seed)
            n_fit = len(ctx.fit_ids)
            rng = np.random.default_rng(3000 + seed)
            fold_of = rng.permutation(n_fit) % folds
            oof = np.zeros((n_fit, len(model_names)))
            for fold in range(folds):
                tr = np.where(fold_of != fold)[0]
                ho = np.where(fold_of == fold)[0]
                inner_tr, inner_va = val_split_rows(len(tr))
                tr_ids = [ctx.fit_ids[i] for i in tr[inner_tr]]
                va_ids = [ctx.fit_ids[i] for i in tr[inner_va]]
                q, mi, y = pairs_for(matrix, tr_ids, ctx.fit_vecs[tr[inner_tr]])
                vq, vmi, vy = pairs_for(matrix, va_ids, ctx.fit_vecs[tr[inner_va]])
                params, _tc, _vc = fit_irt_cfg(q, mi, y, len(model_names), (vq, vmi, vy), IRT2)
                v_rows, v_mis, v_ys = scored_triples(matrix, va_ids)
                v_pred = np.clip(
                    irt_probs(params, ctx.fit_vecs[tr[inner_va]])[v_rows, v_mis],
                    1e-6,
                    1 - 1e-6,
                )
                a, b_ = platt(np.log(v_pred / (1 - v_pred)), v_ys)
                oof[ho] = _platt_apply(a, b_, irt_probs(params, ctx.fit_vecs[ho]))
            # Final model on the full fit (standard irt2 protocol).
            cal_te, _vb, _e, meta = _fit_irt2(ctx, matrix)
            cost_scale = sum(ctx.costs.values()) / len(ctx.costs)
            pen = np.asarray([ctx.costs.get(m, cost_scale) / cost_scale for m in model_names])
            base_idx = model_names.index(ctx.best_name)
            for lam in [0.01, 0.02]:
                s_of = oof - lam * pen
                pick_of = np.argmax(s_of, axis=1)
                gap_of = s_of[np.arange(n_fit), pick_of] - s_of[:, base_idx]
                order_of = np.argsort(-gap_of)
                deltas = np.zeros(n_fit)
                known = np.zeros(n_fit, dtype=bool)
                for r, sid in enumerate(ctx.fit_ids):
                    routed_r = cell_mean(sid, model_names[int(pick_of[r])])
                    base_r = cell_mean(sid, ctx.best_name)
                    if routed_r is not None and base_r is not None:
                        deltas[r] = routed_r - base_r
                        known[r] = True
                q_star = 0.0
                for cov in qs:
                    top = order_of[: int(round(cov * n_fit))]
                    d = deltas[top][known[top]]
                    if len(d) < 8:
                        continue
                    lcb = float(np.mean(d)) - float(np.std(d, ddof=1) / np.sqrt(len(d)))
                    if lcb >= 0:
                        q_star = cov
                s_te = cal_te - lam * pen
                pick_te = np.argmax(s_te, axis=1)
                gap_te = s_te[np.arange(len(s_te)), pick_te] - s_te[:, base_idx]
                order_te = np.argsort(-gap_te)
                routed = set(order_te[: int(round(q_star * len(order_te)))].tolist())
                picks = [
                    model_names[int(pick_te[t])] if t in routed else ctx.best_name
                    for t in range(len(ctx.test_ids))
                ]
                record(
                    matrix_name=name,
                    matrix=matrix,
                    variant="r3-cov-nested-cv",
                    params={"lam": lam, "q_star": q_star, "sel_n": int(known.sum())},
                    split_seed=seed,
                    fit_ids=ctx.fit_ids,
                    test_ids=ctx.test_ids,
                    picks=dict(zip(ctx.test_ids, picks, strict=True)),
                    best_eval=ctx.best_eval,
                    best_name=ctx.best_name,
                    oracle_acc=ctx.oracle_acc,
                )


def cmd_statz(matrices: dict[str, OutcomeMatrix], seeds: list[int]) -> None:
    """Head-to-head climb: irt2 PROPOSES, neighbor evidence CERTIFIES (r1's stat guard).

    Pick = argmax(calibrated irt2 P - lam*penalty). Certification: over the pick's k=50
    nearest fit neighbors (3-large cosine when cached, else hashing; relative threshold
    0.95 like r1), paired per-neighbor reward deltas (pick minus baseline) must clear
    z standard errors (z=0.5, doubled when the pick is pricier; revert below 8 pairs).
    Hypothesis: if irt2 proposals beat kNN's profile-argmax anywhere, this exceeds r1's
    champion; if picks converge to r1's, the P family is irrelevant behind the stat guard.
    """
    z_base = 0.5
    for name, matrix in matrices.items():
        model_names = matrix.model_names()
        cell: dict[tuple[str, str], list[float]] = {}
        for o in matrix.outcomes:
            if o.reward is not None:
                cell.setdefault((o.scenario_id, o.model), []).append(o.reward)
        oai_cache = DATA / "cache" / f"{name}-oai3l-tasks.npy"
        for seed in seeds:
            ctx = SplitContext(name, matrix, seed)
            cal, _vb, _e, meta = _fit_irt2(ctx, matrix)
            if oai_cache.exists():
                by_sid = cached_oai_vecs(name, matrix)
                fit_n = np.stack([by_sid[s] for s in ctx.fit_ids])
                te_n = np.stack([by_sid[s] for s in ctx.test_ids])
                nbr_embed = "oai"
            else:
                fit_n, te_n, nbr_embed = ctx.fit_vecs, ctx.test_vecs, EMBED_KIND
            sims_all = te_n @ fit_n.T  # both L2-normalized
            cost_scale = sum(ctx.costs.values()) / len(ctx.costs)
            pen = np.asarray([ctx.costs.get(m, cost_scale) / cost_scale for m in model_names])
            base_cost = ctx.costs.get(ctx.best_name, cost_scale)
            k = min(50, len(ctx.fit_ids))
            for lam in [0.0, 0.01, 0.02]:
                scores = cal - lam * pen
                pick_idx = np.argmax(scores, axis=1)
                picks, certified = [], 0
                for t in range(len(ctx.test_ids)):
                    pick = model_names[int(pick_idx[t])]
                    if pick == ctx.best_name:
                        picks.append(pick)
                        continue
                    sims = sims_all[t]
                    kth = np.sort(sims)[-k]
                    nbr = np.where(sims > 0.95 * kth)[0]
                    deltas = []
                    for j in nbr:
                        sid = ctx.fit_ids[int(j)]
                        pv = cell.get((sid, pick))
                        bv = cell.get((sid, ctx.best_name))
                        if pv and bv:
                            deltas.append(float(np.mean(pv)) - float(np.mean(bv)))
                    z_need = 2 * z_base if ctx.costs.get(pick, 0.0) > base_cost else z_base
                    ok = False
                    if len(deltas) >= 8:
                        arr = np.asarray(deltas)
                        se = float(arr.std(ddof=1) / np.sqrt(len(arr)))
                        ok = se > 0 and float(arr.mean()) / se >= z_need
                    picks.append(pick if ok else ctx.best_name)
                    certified += int(ok)
                record(
                    matrix_name=name,
                    matrix=matrix,
                    variant="r3-irt2-statz",
                    params={
                        **meta,
                        "lam": lam,
                        "z": z_base,
                        "nbr_embed": nbr_embed,
                        "certified": certified,
                    },
                    split_seed=seed,
                    fit_ids=ctx.fit_ids,
                    test_ids=ctx.test_ids,
                    picks=dict(zip(ctx.test_ids, picks, strict=True)),
                    best_eval=ctx.best_eval,
                    best_name=ctx.best_name,
                    oracle_acc=ctx.oracle_acc,
                )


def cmd_adaptive(matrices: dict[str, OutcomeMatrix], seeds: list[int]) -> None:
    """Q4: n-aware guard margin max(0.03, c/sqrt(n_fit)) on irt2 (Platt'd P).

    At n_fit=843 (ours9) the floor 0.03 binds (no behavior change); at n_fit=17
    (wm corpora) c=0.5 demands a 0.12 gap, which uncalibratable small-n P cannot
    clear, so the router reverts by construction. The learned-P analog of r1's
    support-aware stat guard.
    """
    for name, matrix in matrices.items():
        for seed in seeds:
            ctx = SplitContext(name, matrix, seed)
            cal, val_bce, _, meta = _fit_irt2(ctx, matrix)
            for c in [0.3, 0.5]:
                margin = max(MARGIN, c / np.sqrt(len(ctx.fit_ids)))
                for lam in [0.0, 0.01]:
                    ctx.rec(
                        "r3-irt2-adm",
                        {**meta, "c": c, "margin": round(float(margin), 4)},
                        cal,
                        lam,
                        margin=float(margin),
                    )


def cmd_curves(matrices: dict[str, OutcomeMatrix], seeds: list[int]) -> None:
    """Q3 sample efficiency: guarded routing accuracy vs #fit scenarios, per family."""
    sizes = [25, 50, 100, 200, 400]
    for name, matrix in matrices.items():
        model_names = matrix.model_names()
        for seed in seeds:
            ctx = SplitContext(name, matrix, seed)
            rng = np.random.default_rng(1000 + seed)
            order_rows = rng.permutation(len(ctx.fit_ids))
            for size in [s for s in sizes if s <= len(ctx.fit_ids)] + [len(ctx.fit_ids)]:
                keep = sorted(order_rows[:size].tolist())
                sub_ids = [ctx.fit_ids[i] for i in keep]
                sub_vecs = ctx.fit_vecs[keep]
                sub_costs = mean_costs(matrix, sub_ids)
                sub_best, _, _ = best_single_model(matrix, fit_ids=sub_ids, eval_ids=ctx.test_ids)
                # irt2 protocol at reduced n (val split shrinks with it).
                sub_ctx = ctx  # reuse embeddings; only ids/vecs differ below
                tr_rows_l, va_rows_l = val_split_rows(len(sub_ids))
                tr_ids = [sub_ids[i] for i in tr_rows_l]
                va_ids = [sub_ids[i] for i in va_rows_l]
                q, mi, y = pairs_for(matrix, tr_ids, sub_vecs[tr_rows_l])
                vq, vmi, vy = pairs_for(matrix, va_ids, sub_vecs[va_rows_l])
                v_rows, v_mis, v_ys = scored_triples(matrix, va_ids)
                heads = {}
                params, _tc, vc = fit_irt_cfg(q, mi, y, len(model_names), (vq, vmi, vy), IRT2)
                heads["irt2"] = (
                    irt_probs(params, sub_vecs[va_rows_l]),
                    irt_probs(params, ctx.test_vecs),
                    round(min(vc), 4),
                )
                w, b2, _tc2, vc2 = fit_logistic(
                    q, mi, y, len(model_names), val=(vq, vmi, vy), checkpoint=True
                )
                heads["logistic2"] = (
                    logistic_probs(w, b2, sub_vecs[va_rows_l]),
                    logistic_probs(w, b2, ctx.test_vecs),
                    round(min(vc2), 4),
                )
                heads["knn"] = (
                    knn_probs(matrix, sub_ids, sub_vecs, sub_vecs[va_rows_l], k=50),
                    knn_probs(matrix, sub_ids, sub_vecs, ctx.test_vecs, k=50),
                    None,
                )
                for fam, (raw_va, raw_te, val_bce) in heads.items():
                    v_pred = np.clip(raw_va[v_rows, v_mis], 1e-6, 1 - 1e-6)
                    a, b_ = platt(np.log(v_pred / (1 - v_pred)), v_ys)
                    te = np.clip(raw_te, 1e-6, 1 - 1e-6)
                    cal = 1.0 / (1.0 + np.exp(-(a * np.log(te / (1 - te)) + b_)))
                    picks = guarded_picks(cal, model_names, sub_costs, sub_best, 0.02)
                    record(
                        matrix_name=name,
                        matrix=matrix,
                        variant=f"r3-{fam}-nfit",
                        params={"n_fit": size, "lam": 0.02, "val_bce": val_bce},
                        split_seed=seed,
                        fit_ids=sub_ids,
                        test_ids=ctx.test_ids,
                        picks=dict(zip(ctx.test_ids, picks, strict=True)),
                        best_eval=sub_ctx.best_eval,
                        best_name=ctx.best_name,
                        oracle_acc=ctx.oracle_acc,
                        notes=f"sub_best={sub_best}",
                    )


def val_split_rows(n: int, frac: float = 0.15, seed: int = 7) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * frac))
    return sorted(perm[n_val:].tolist()), sorted(perm[:n_val].tolist())


def cmd_coldstart(matrices: dict[str, OutcomeMatrix], seeds: list[int]) -> None:
    """Q3 cold start: new pool model with k labeled pairs; theta from a frozen head.

    Protocol per (target, k, seed): fit irt2 with the target's pairs REMOVED from fit;
    theta_target initialized to the mean of the other thetas (population prior); for k>0,
    optimize ONLY theta_target on k labeled fit pairs with the head frozen (classical IRT
    ability estimation: 16 params from k observations). Baseline: logistic with w_target
    trained on the same k pairs (1024 params from k observations). Metric: BCE + rank
    quality of the target's predicted P on test cells, plus guarded routing accuracy.
    """
    targets = ["sonnet-5", "opus-4-8"]
    ks = [0, 5, 10, 20, 40]
    for name, matrix in matrices.items():
        model_names = matrix.model_names()
        cell: dict[tuple[str, str], list[float]] = {}
        for o in matrix.outcomes:
            if o.reward is not None:
                cell.setdefault((o.scenario_id, o.model), []).append(o.reward)
        for seed in seeds:
            ctx = SplitContext(name, matrix, seed)
            tr_rows, va_rows = val_split(ctx)
            tr_ids = [ctx.fit_ids[i] for i in tr_rows]
            va_ids = [ctx.fit_ids[i] for i in va_rows]
            tr_vecs, va_vecs = ctx.fit_vecs[tr_rows], ctx.fit_vecs[va_rows]
            for target in targets:
                if target not in model_names:
                    continue
                t_idx = model_names.index(target)

                # Fit with target's pairs removed (drop from BOTH train and val pairs).
                def drop_target(
                    ids: list[str],
                    vecs: np.ndarray,
                    matrix: OutcomeMatrix = matrix,
                    t_idx: int = t_idx,
                ) -> tuple:
                    q, mi, y = pairs_for(matrix, ids, vecs)
                    keep = mi != t_idx
                    return q[keep], mi[keep], y[keep]

                q, mi, y = drop_target(tr_ids, tr_vecs)
                vq, vmi, vy = drop_target(va_ids, va_vecs)
                params, _tc, vc = fit_irt_cfg(q, mi, y, len(model_names), (vq, vmi, vy), IRT2)
                others = [i for i in range(len(model_names)) if i != t_idx]
                # Scenarios in fit where the target IS scored (the labeled budget pool).
                scored_fit = [
                    (i, sid) for i, sid in enumerate(ctx.fit_ids) if (sid, target) in cell
                ]
                rng = np.random.default_rng(2000 + seed)
                pool = rng.permutation(len(scored_fit))
                for k in ks:
                    theta = params.theta.copy()
                    theta[t_idx] = params.theta[others].mean(axis=0)
                    if k > 0:
                        chosen = [scored_fit[i] for i in pool[:k]]
                        k_vecs = ctx.fit_vecs[[row for row, _sid in chosen]]
                        k_y = np.asarray(
                            [float(np.mean(cell[(sid, target)])) for _row, sid in chosen]
                        )
                        hid = np.maximum(k_vecs @ params.w1.T + params.b1, 0.0)
                        alpha = hid @ params.wa.T + params.ba  # [k, dim]
                        beta = hid @ params.wb + params.bb  # [k]
                        prior = theta[t_idx].copy()
                        th = prior.copy()
                        m_a = v_a = np.zeros_like(th)
                        # MAP ability estimate: SUM data gradient (grows with k) + fixed
                        # Gaussian prior at the population-mean theta (empirical Bayes:
                        # k=0 returns the prior; data dominates as k grows).
                        tau = 2.0
                        for step in range(1, 301):
                            logits = alpha @ th - beta
                            p = 1.0 / (1.0 + np.exp(-logits))
                            g = alpha.T @ (p - k_y) + tau * (th - prior)
                            m_a = 0.9 * m_a + 0.1 * g
                            v_a = 0.999 * v_a + 0.001 * g**2
                            th -= (
                                1e-2
                                * (m_a / (1 - 0.9**step))
                                / (np.sqrt(v_a / (1 - 0.999**step)) + 1e-8)
                            )
                        theta[t_idx] = th
                    saved = params.theta
                    params.theta = theta
                    probs = irt_probs(params, ctx.test_vecs)
                    params.theta = saved
                    # Target-model prediction quality on test cells.
                    preds, acts = [], []
                    for t, sid in enumerate(ctx.test_ids):
                        vals = cell.get((sid, target))
                        if vals:
                            preds.append(probs[t, t_idx])
                            acts.append(float(np.mean(vals)))
                    preds_a = np.clip(np.asarray(preds), 1e-6, 1 - 1e-6)
                    acts_a = np.asarray(acts)
                    t_bce = bce(preds_a, acts_a)
                    hard = acts_a >= 0.5
                    auc = float("nan")
                    if hard.any() and (~hard).any():
                        ranks = preds_a.argsort().argsort().astype(float)
                        auc = float((ranks[hard].mean() - ranks[~hard].mean()) / len(ranks) + 0.5)
                    picks = guarded_picks(probs, model_names, ctx.costs, ctx.best_name, 0.02)
                    result = evaluate_choices(
                        matrix,
                        ctx.test_ids,
                        dict(zip(ctx.test_ids, picks, strict=True)).__getitem__,
                    )
                    logger.info(
                        "%s seed%d coldstart %s k=%d: bce=%.4f auc=%.3f routed_acc=%.4f "
                        "target_share=%.2f",
                        name,
                        seed,
                        target,
                        k,
                        t_bce,
                        auc,
                        result.accuracy,
                        result.model_mix.get(target, 0.0),
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "repro",
            "baselines",
            "audit",
            "frontier",
            "calibration",
            "sweep",
            "tiebreak",
            "irt2",
            "audit2",
            "curves",
            "coldstart",
            "adaptive",
            "ensemble",
            "coverage",
            "nested",
            "nested-cv",
            "statz",
        ],
    )
    parser.add_argument("--matrices", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=SPLIT_SEEDS)
    parser.add_argument("--embed", choices=["hashing", "oai"], default="hashing")
    args = parser.parse_args()
    global EMBED_KIND  # noqa: PLW0603 - one-shot CLI configuration
    EMBED_KIND = args.embed
    matrices = load_matrices(args.matrices)
    logger.info("matrices: %s | seeds: %s", sorted(matrices), args.seeds)
    commands = {
        "repro": cmd_repro,
        "baselines": cmd_baselines,
        "audit": cmd_audit,
        "frontier": cmd_frontier,
        "calibration": cmd_calibration,
        "sweep": cmd_sweep,
        "tiebreak": cmd_tiebreak,
        "irt2": cmd_irt2,
        "audit2": cmd_audit2,
        "curves": cmd_curves,
        "coldstart": cmd_coldstart,
        "adaptive": cmd_adaptive,
        "ensemble": cmd_ensemble,
        "coverage": cmd_coverage,
        "nested": cmd_nested,
        "nested-cv": cmd_nested_cv,
        "statz": cmd_statz,
    }
    commands[args.command](matrices, args.seeds)
    logger.info("runs -> %s", RUNS)


if __name__ == "__main__":
    main()
