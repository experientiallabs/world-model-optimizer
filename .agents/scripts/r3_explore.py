"""R3 exploration sweep: alternative geometries vs the kNN champion, one round each.

Mandate (2026-07-25): r3's IRT line is closed; this driver runs the "did we ever try X"
sweep. Bar for every method: r1's champion (knn-statz05-oai) paired-by-seed on ours9, from
runs/r1.jsonl on shared split seeds. One round per method: beat the champion OR show a
unique capability (ood-task drift, tiny-n) - else record the kill.

Methods:
  repro   - faithful champion rerun through r1's OWN route() (imported, not reimplemented);
            must match r1.jsonl per-seed before anything else counts.
  metric  - method 1: learned linear-ish metric FEEDING the unchanged champion. Transform
            z(x) = normalize([x ; alpha * s(x)]) where s(x) = softmax logits of a fit-only
            multinomial LR predicting the oracle-winner class (cheapest max-reward model).
            Neighbors with the same predicted winner pull together; profile + stat guard
            stay r1's code, only ctx.task_vecs changes. alpha chosen per seed on an inner
            fit-val split by ROUTED reward (never BCE; round-2 lesson).
  clf     - methods 2-4: RBF-SVM / gradient-boosted trees / small MLP win-vs-baseline
            classifiers proposing picks, certified by the shared neighbor z-guard
            (identical harness to round-5's statz, so the comparison to irt2-as-proposer
            is exact). Predicted kill on interaction-information grounds.
  control - shuffled-label controls for every surviving variant (3-large embeddings only).

All offline, $0 API (embeddings come from r1's disk caches). Runs -> runs/r3.jsonl with
variants r3x-*; fit outputs stay OUT of params (master dashboard ruling 2026-07-25).
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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np

from wmo.optimize.outcomes import OutcomeMatrix
from wmo.research.routerbench import best_single_model, oracle, split_scenario_ids
from wmo.research.routing_runs import RunRecord, append_run, evaluate_choices
from wmo.research.routing_corpus import routing_data

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("r3x")

DATA = routing_data()
RUNS = DATA / "runs/r3.jsonl"
R1_SCRIPT = Path(__file__).with_name("r1_retrieval_ablations.py")
SPLIT_SEEDS = [0, 1, 2, 3, 4]


def load_r1() -> ModuleType:
    """Import r1's ablation module (champion code, unmodified)."""
    spec = importlib.util.spec_from_file_location("r1_retrieval_ablations", R1_SCRIPT)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import r1 module from {R1_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["r1_retrieval_ablations"] = module
    spec.loader.exec_module(module)
    return module


CHAMP = {"second_route": False, "guard": "stat", "z": 0.5}


def champion_picks(
    r1: ModuleType,
    ctx: object,
    fit_ids: list[str],
    test_ids: list[str],
    best_name: str,
    rewards_cell: dict | None = None,
) -> dict[str, str]:
    params = r1.RetrievalParams(**CHAMP)
    return r1.route(ctx, params, fit_ids, test_ids, best_name, rewards_cell=rewards_cell)


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
        "%s/%s seed%d %s: acc=%.4f cost=$%.5f (base %.4f)",
        matrix_name,
        variant,
        seed,
        params,
        result.accuracy,
        result.cost_per_call,
        best_eval.accuracy,
    )
    return rec


def oracle_winner_labels(matrix: OutcomeMatrix, fit_ids: list[str]) -> tuple[list[int], list[str]]:
    """Per fit scenario: index of the cheapest max-reward model (the oracle's pick)."""
    cell: dict[tuple[str, str], list[float]] = {}
    cost: dict[str, list[float]] = {}
    for o in matrix.outcomes:
        if o.reward is not None:
            cell.setdefault((o.scenario_id, o.model), []).append(o.reward)
            cost.setdefault(o.model, []).append(o.cost_usd)
    mean_cost = {m: float(np.mean(v)) for m, v in cost.items()}
    names = matrix.model_names()
    labels = []
    for sid in fit_ids:
        rewards = {m: float(np.mean(cell[(sid, m)])) for m in names if (sid, m) in cell}
        if not rewards:
            labels.append(0)
            continue
        top = max(rewards.values())
        winners = [m for m, r in rewards.items() if r >= top - 1e-9]
        pick = min(winners, key=lambda m: mean_cost.get(m, float("inf")))
        labels.append(names.index(pick))
    return labels, names


def fit_winner_lr(
    vecs: np.ndarray, labels: list[int], n_classes: int
) -> Callable[[np.ndarray], np.ndarray]:
    """Multinomial LR on the oracle-winner class; returns logits(x) [*, n_classes]."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(vecs, labels)
    classes = list(clf.classes_)

    def logits(x: np.ndarray) -> np.ndarray:
        raw = clf.decision_function(x)
        if raw.ndim == 1:  # binary edge case
            raw = np.stack([-raw, raw], axis=1)
        full = np.zeros((len(x), n_classes))
        for col, cls in enumerate(classes):
            full[:, int(cls)] = raw[:, col]
        return full

    return logits


def transform_vecs(
    base: dict[str, np.ndarray],
    logits: Callable[[np.ndarray], np.ndarray],
    alpha: float,
    sids: list[str],
) -> dict[str, np.ndarray]:
    """z = normalize([x ; alpha * softmax(logits)]) per sid."""
    xs = np.stack([base[s] for s in sids])
    lg = logits(xs)
    lg = lg - lg.max(axis=1, keepdims=True)
    probs = np.exp(lg)
    probs /= probs.sum(axis=1, keepdims=True)
    z = np.concatenate([xs, alpha * probs], axis=1)
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    return dict(zip(sids, z, strict=True))


def routed_reward(matrix: OutcomeMatrix, ids: list[str], picks: dict[str, str]) -> float:
    return evaluate_choices(matrix, ids, lambda sid: picks[sid]).accuracy


def fisher_weights(vecs: np.ndarray, labels: list[int]) -> np.ndarray:
    """Per-dimension Fisher score: between-class variance / within-class variance."""
    labels_arr = np.asarray(labels)
    grand = vecs.mean(axis=0)
    between = np.zeros(vecs.shape[1])
    within = np.zeros(vecs.shape[1])
    for cls in np.unique(labels_arr):
        rows = vecs[labels_arr == cls]
        if len(rows) < 2:
            continue
        between += len(rows) * (rows.mean(axis=0) - grand) ** 2
        within += ((rows - rows.mean(axis=0)) ** 2).sum(axis=0)
    return between / np.maximum(within, 1e-12)


def fisher_transform(
    base: dict[str, np.ndarray], weights: np.ndarray, beta: float, sids: list[str]
) -> dict[str, np.ndarray]:
    """x' = normalize(x * w^beta): diagonal metric (LMNN-spirit feature weighting)."""
    scale = np.power(np.maximum(weights, 1e-12), beta)
    scale /= scale.mean()
    xs = np.stack([base[s] for s in sids]) * scale
    xs /= np.maximum(np.linalg.norm(xs, axis=1, keepdims=True), 1e-12)
    return dict(zip(sids, xs, strict=True))


BETAS = [0.0, 0.25, 0.5, 1.0]


def cmd_fisher(args: argparse.Namespace) -> None:
    """Method 1b: Fisher diagonal reweighting feeding the unchanged champion."""
    r1 = load_r1()
    for name in args.matrices:
        matrix = OutcomeMatrix.load(DATA / "matrices" / f"{name}_matrix.json")
        ctx = r1.MatrixContext(matrix, name, embed="openai", embed_replies=False)
        base_vecs = dict(ctx.task_vecs)
        all_sids = list(base_vecs)
        for seed in args.seeds:
            fit_ids, test_ids = split_scenario_ids(matrix, train_fraction=0.7, seed=seed)
            best_name, _, _ = best_single_model(matrix, fit_ids=fit_ids, eval_ids=test_ids)
            rng = np.random.default_rng(7)
            perm = rng.permutation(len(fit_ids))
            n_val = max(1, int(0.2 * len(fit_ids)))
            va_ids = sorted(fit_ids[i] for i in perm[:n_val])
            tr_ids = sorted(fit_ids[i] for i in perm[n_val:])
            labels_tr, _ = oracle_winner_labels(matrix, tr_ids)
            w_tr = fisher_weights(np.stack([base_vecs[s] for s in tr_ids]), labels_tr)
            beta_scores = {}
            for beta in BETAS:
                ctx.task_vecs = fisher_transform(base_vecs, w_tr, beta, all_sids)
                picks_va = champion_picks(r1, ctx, tr_ids, va_ids, best_name)
                beta_scores[beta] = routed_reward(matrix, va_ids, picks_va)
            beta_star = max(beta_scores, key=lambda b: beta_scores[b])
            logger.info(
                "%s seed%d beta selection (val routed): %s -> beta*=%s",
                name,
                seed,
                {b: round(v, 4) for b, v in beta_scores.items()},
                beta_star,
            )
            labels_fit, _ = oracle_winner_labels(matrix, fit_ids)
            w_fit = fisher_weights(np.stack([base_vecs[s] for s in fit_ids]), labels_fit)
            ctx.task_vecs = fisher_transform(base_vecs, w_fit, beta_star, all_sids)
            picks = champion_picks(r1, ctx, fit_ids, test_ids, best_name)
            record_run(
                matrix_name=name,
                matrix=matrix,
                variant="r3x-metric-fisher",
                params={"beta_grid": "0-1", "z": 0.5, "embed": "oai3l"},
                seed=seed,
                fit_ids=fit_ids,
                test_ids=test_ids,
                picks=picks,
                best_name=best_name,
                notes=f"beta_star={beta_star}",
            )
            ctx.task_vecs = dict(base_vecs)


def cmd_repro(args: argparse.Namespace) -> None:
    """Champion repro through r1's own code; compare per-seed vs r1.jsonl offline."""
    r1 = load_r1()
    for name in args.matrices:
        matrix = OutcomeMatrix.load(DATA / "matrices" / f"{name}_matrix.json")
        ctx = r1.MatrixContext(matrix, name, embed="openai", embed_replies=False)
        for seed in args.seeds:
            fit_ids, test_ids = split_scenario_ids(matrix, train_fraction=0.7, seed=seed)
            best_name, _, _ = best_single_model(matrix, fit_ids=fit_ids, eval_ids=test_ids)
            picks = champion_picks(r1, ctx, fit_ids, test_ids, best_name)
            record_run(
                matrix_name=name,
                matrix=matrix,
                variant="r3x-champ-repro",
                params={"z": 0.5, "embed": "oai3l"},
                seed=seed,
                fit_ids=fit_ids,
                test_ids=test_ids,
                picks=picks,
                best_name=best_name,
            )


ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0]


def cmd_metric(args: argparse.Namespace) -> None:
    """Method 1: LR-winner-logit metric feeding the unchanged champion."""
    r1 = load_r1()
    for name in args.matrices:
        matrix = OutcomeMatrix.load(DATA / "matrices" / f"{name}_matrix.json")
        ctx = r1.MatrixContext(matrix, name, embed="openai", embed_replies=False)
        base_vecs = dict(ctx.task_vecs)
        all_sids = list(base_vecs)
        n_models = len(matrix.model_names())
        for seed in args.seeds:
            fit_ids, test_ids = split_scenario_ids(matrix, train_fraction=0.7, seed=seed)
            best_name, _, _ = best_single_model(matrix, fit_ids=fit_ids, eval_ids=test_ids)
            # Inner split for alpha selection by ROUTED reward (train-store only).
            rng = np.random.default_rng(7)
            perm = rng.permutation(len(fit_ids))
            n_val = max(1, int(0.2 * len(fit_ids)))
            va_ids = sorted(fit_ids[i] for i in perm[:n_val])
            tr_ids = sorted(fit_ids[i] for i in perm[n_val:])
            labels_tr, _ = oracle_winner_labels(matrix, tr_ids)
            logits_tr = fit_winner_lr(np.stack([base_vecs[s] for s in tr_ids]), labels_tr, n_models)
            alpha_scores = {}
            for alpha in ALPHAS:
                ctx.task_vecs = transform_vecs(base_vecs, logits_tr, alpha, all_sids)
                picks_va = champion_picks(r1, ctx, tr_ids, va_ids, best_name)
                alpha_scores[alpha] = routed_reward(matrix, va_ids, picks_va)
            alpha_star = max(alpha_scores, key=lambda a: alpha_scores[a])
            logger.info(
                "%s seed%d alpha selection (val routed): %s -> alpha*=%s",
                name,
                seed,
                {a: round(v, 4) for a, v in alpha_scores.items()},
                alpha_star,
            )
            # Final: LR refit on FULL fit, champion routed in the transformed space.
            labels_fit, _ = oracle_winner_labels(matrix, fit_ids)
            logits_fit = fit_winner_lr(
                np.stack([base_vecs[s] for s in fit_ids]), labels_fit, n_models
            )
            ctx.task_vecs = transform_vecs(base_vecs, logits_fit, alpha_star, all_sids)
            picks = champion_picks(r1, ctx, fit_ids, test_ids, best_name)
            record_run(
                matrix_name=name,
                matrix=matrix,
                variant="r3x-metric-lrwinner",
                params={"alpha_grid": "0-2", "z": 0.5, "embed": "oai3l"},
                seed=seed,
                fit_ids=fit_ids,
                test_ids=test_ids,
                picks=picks,
                best_name=best_name,
                notes=f"alpha_star={alpha_star}",
            )
            ctx.task_vecs = dict(base_vecs)


def clf_family(
    kind: str, vecs: np.ndarray, wins: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit one win-vs-baseline classifier; returns predict_p(x) in [0,1]."""
    if kind == "svm":
        from sklearn.svm import SVC

        clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
    elif kind == "gbt":
        from sklearn.ensemble import HistGradientBoostingClassifier

        clf = HistGradientBoostingClassifier(
            max_iter=200, early_stopping=True, validation_fraction=0.15, random_state=42
        )
    elif kind == "mlp":
        from sklearn.neural_network import MLPClassifier

        clf = MLPClassifier(
            hidden_layer_sizes=(64,),
            early_stopping=True,
            validation_fraction=0.15,
            max_iter=500,
            random_state=42,
        )
    else:
        raise SystemExit(f"unknown clf kind {kind}")
    if len(set(wins.tolist())) < 2:
        p = float(np.mean(wins))
        return lambda x: np.full(len(x), p)
    clf.fit(vecs, wins)
    one_col = list(clf.classes_).index(1)
    return lambda x: clf.predict_proba(x)[:, one_col]


def cmd_clf(args: argparse.Namespace) -> None:
    """Methods 2-4: classifier proposes, shared neighbor z-guard certifies."""
    r1 = load_r1()
    for name in args.matrices:
        matrix = OutcomeMatrix.load(DATA / "matrices" / f"{name}_matrix.json")
        ctx = r1.MatrixContext(matrix, name, embed="openai", embed_replies=False)
        names = matrix.model_names()
        for seed in args.seeds:
            fit_ids, test_ids = split_scenario_ids(matrix, train_fraction=0.7, seed=seed)
            cell = r1.shuffled_cells(ctx, fit_ids, seed) if args.shuffled else ctx.rewards_cell
            best_name, _, _ = best_single_model(matrix, fit_ids=fit_ids, eval_ids=test_ids)
            mean_cost = ctx.mean_cost(fit_ids)
            base_cost = mean_cost.get(best_name, 0.0)
            cost_scale = sum(mean_cost.values()) / len(mean_cost)
            fit_x = np.stack([ctx.task_vecs[s] for s in fit_ids])
            test_x = np.stack([ctx.task_vecs[s] for s in test_ids])
            fit_m = np.stack([ctx.task_vecs[s] for s in fit_ids])
            for kind in args.kinds:
                # Per-model win-vs-baseline probability.
                p_win = np.zeros((len(test_ids), len(names)))
                for mi, model in enumerate(names):
                    if model == best_name:
                        continue
                    xs, ys = [], []
                    for row, sid in enumerate(fit_ids):
                        pv, bv = cell.get((sid, model)), cell.get((sid, best_name))
                        if pv and bv:
                            d = float(np.mean(pv)) - float(np.mean(bv))
                            if abs(d) > 1e-9:  # decisive cells only
                                xs.append(fit_x[row])
                                ys.append(1 if d > 0 else 0)
                    if len(ys) < 12:
                        continue
                    predict = clf_family(kind, np.stack(xs), np.asarray(ys))
                    p_win[:, mi] = predict(test_x)
                for lam in [0.0, 0.01]:
                    pen = np.asarray([mean_cost.get(m, cost_scale) / cost_scale for m in names])
                    scores = p_win - lam * pen
                    picks = {}
                    k = min(50, len(fit_ids))
                    certified = 0
                    for t, sid in enumerate(test_ids):
                        mi = int(np.argmax(scores[t]))
                        pick = names[mi]
                        if scores[t, mi] <= 0.5 or pick == best_name:
                            picks[sid] = best_name
                            continue
                        sims = fit_m @ ctx.task_vecs[sid]
                        kth = np.sort(sims)[-k]
                        nbr = np.where(sims > 0.95 * kth)[0]
                        deltas = []
                        for j in nbr:
                            pv = cell.get((fit_ids[int(j)], pick))
                            bv = cell.get((fit_ids[int(j)], best_name))
                            if pv and bv:
                                deltas.append(float(np.mean(pv)) - float(np.mean(bv)))
                        z_need = 1.0 if mean_cost.get(pick, 0.0) > base_cost else 0.5
                        ok = False
                        if len(deltas) >= 8:
                            arr = np.asarray(deltas)
                            se = float(arr.std(ddof=1) / np.sqrt(len(arr)))
                            ok = se > 0 and float(arr.mean()) / se >= z_need
                        picks[sid] = pick if ok else best_name
                        certified += int(ok)
                    record_run(
                        matrix_name=name,
                        matrix=matrix,
                        variant=f"r3x-{kind}" + ("-shuffled-control" if args.shuffled else ""),
                        params={"lam": lam, "guard": "statz0.5", "embed": "oai3l"},
                        seed=seed,
                        fit_ids=fit_ids,
                        test_ids=test_ids,
                        picks=picks,
                        best_name=best_name,
                        notes=f"certified={certified}",
                    )


def cmd_control(args: argparse.Namespace) -> None:
    """Shuffled-label control for the metric variant (champion in shuffled-label space)."""
    r1 = load_r1()
    for name in args.matrices:
        matrix = OutcomeMatrix.load(DATA / "matrices" / f"{name}_matrix.json")
        ctx = r1.MatrixContext(matrix, name, embed="openai", embed_replies=False)
        base_vecs = dict(ctx.task_vecs)
        all_sids = list(base_vecs)
        n_models = len(matrix.model_names())
        for seed in args.seeds:
            fit_ids, test_ids = split_scenario_ids(matrix, train_fraction=0.7, seed=seed)
            best_name, _, _ = best_single_model(matrix, fit_ids=fit_ids, eval_ids=test_ids)
            shuffled = r1.shuffled_cells(ctx, fit_ids, seed)
            # Labels AND profile/guard all come from the shuffled cells.
            cell_backup = ctx.rewards_cell
            ctx.rewards_cell = shuffled
            labels_fit, _ = oracle_winner_labels_from_cells(shuffled, matrix, fit_ids)
            logits_fit = fit_winner_lr(
                np.stack([base_vecs[s] for s in fit_ids]), labels_fit, n_models
            )
            ctx.task_vecs = transform_vecs(base_vecs, logits_fit, 1.0, all_sids)
            picks = champion_picks(r1, ctx, fit_ids, test_ids, best_name, rewards_cell=shuffled)
            ctx.rewards_cell = cell_backup
            ctx.task_vecs = dict(base_vecs)
            record_run(
                matrix_name=name,
                matrix=matrix,
                variant="r3x-metric-shuffled-control",
                params={"alpha": 1.0, "z": 0.5, "embed": "oai3l"},
                seed=seed,
                fit_ids=fit_ids,
                test_ids=test_ids,
                picks=picks,
                best_name=best_name,
            )


def oracle_winner_labels_from_cells(
    cells: dict, matrix: OutcomeMatrix, fit_ids: list[str]
) -> tuple[list[int], list[str]]:
    cost: dict[str, list[float]] = {}
    for o in matrix.outcomes:
        if o.reward is not None:
            cost.setdefault(o.model, []).append(o.cost_usd)
    mean_cost = {m: float(np.mean(v)) for m, v in cost.items()}
    names = matrix.model_names()
    labels = []
    for sid in fit_ids:
        rewards = {m: float(np.mean(cells[(sid, m)])) for m in names if (sid, m) in cells}
        if not rewards:
            labels.append(0)
            continue
        top = max(rewards.values())
        winners = [m for m, r in rewards.items() if r >= top - 1e-9]
        labels.append(names.index(min(winners, key=lambda m: mean_cost.get(m, 1e9))))
    return labels, names


def ood_task_split(
    matrix: OutcomeMatrix, *, holdout_fraction: float = 0.3, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Leave-whole-task-out: hold out entire eval-prefix groups (~30% of scenarios)."""
    groups: dict[str, list[str]] = {}
    for sid in matrix.scenario_ids():
        prefix = sid.split(":", 1)[0] if ":" in sid else ""
        groups.setdefault(prefix, []).append(sid)
    rng = np.random.default_rng(seed)
    names = list(groups)
    rng.shuffle(names)
    target = holdout_fraction * sum(len(v) for v in groups.values())
    test: list[str] = []
    for gname in names:
        if len(test) >= target:
            break
        test.extend(groups[gname])
    test_set = set(test)
    fit = [sid for sid in matrix.scenario_ids() if sid not in test_set]
    return sorted(fit), sorted(test)


def cmd_ood(args: argparse.Namespace) -> None:
    """ood-task drift: champion vs RBF-SVM proposer on leave-task-out splits."""
    r1 = load_r1()
    for name in args.matrices:
        matrix = OutcomeMatrix.load(DATA / "matrices" / f"{name}_matrix.json")
        ctx = r1.MatrixContext(matrix, name, embed="openai", embed_replies=False)
        names = matrix.model_names()
        for seed in args.seeds:
            fit_ids, test_ids = ood_task_split(matrix, seed=seed)
            best_name, _, _ = best_single_model(matrix, fit_ids=fit_ids, eval_ids=test_ids)
            shuffle_tag = "-shuffled-control" if args.shuffled else ""
            override = r1.shuffled_cells(ctx, fit_ids, seed) if args.shuffled else None
            picks = champion_picks(r1, ctx, fit_ids, test_ids, best_name, rewards_cell=override)
            record_run(
                matrix_name=name,
                matrix=matrix,
                variant=f"r3x-champ-oodtask{shuffle_tag}",
                params={"z": 0.5, "embed": "oai3l", "split": "ood-task"},
                seed=seed,
                fit_ids=fit_ids,
                test_ids=test_ids,
                picks=picks,
                best_name=best_name,
            )
            # SVM proposer + shared z-guard on the same split.
            cell = override if override is not None else ctx.rewards_cell
            mean_cost = ctx.mean_cost(fit_ids)
            base_cost = mean_cost.get(best_name, 0.0)
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
                predict = clf_family("svm", np.stack(xs), np.asarray(ys))
                p_win[:, mi] = predict(test_x)
            picks_svm = {}
            k = min(50, len(fit_ids))
            for t, sid in enumerate(test_ids):
                mi = int(np.argmax(p_win[t]))
                pick = names[mi]
                if p_win[t, mi] <= 0.5 or pick == best_name:
                    picks_svm[sid] = best_name
                    continue
                sims = fit_x @ ctx.task_vecs[sid]
                kth = np.sort(sims)[-k]
                nbr = np.where(sims > 0.95 * kth)[0]
                deltas = []
                for j in nbr:
                    pv = cell.get((fit_ids[int(j)], pick))
                    bv = cell.get((fit_ids[int(j)], best_name))
                    if pv and bv:
                        deltas.append(float(np.mean(pv)) - float(np.mean(bv)))
                z_need = 1.0 if mean_cost.get(pick, 0.0) > base_cost else 0.5
                ok = False
                if len(deltas) >= 8:
                    arr = np.asarray(deltas)
                    se = float(arr.std(ddof=1) / np.sqrt(len(arr)))
                    ok = se > 0 and float(arr.mean()) / se >= z_need
                picks_svm[sid] = pick if ok else best_name
            record_run(
                matrix_name=name,
                matrix=matrix,
                variant=f"r3x-svm-oodtask{shuffle_tag}",
                params={"lam": 0.0, "guard": "statz0.5", "embed": "oai3l", "split": "ood-task"},
                seed=seed,
                fit_ids=fit_ids,
                test_ids=test_ids,
                picks=picks_svm,
                best_name=best_name,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["repro", "metric", "fisher", "clf", "control", "ood"])
    parser.add_argument("--matrices", nargs="*", default=["routerbench-ours9"])
    parser.add_argument("--seeds", nargs="*", type=int, default=SPLIT_SEEDS)
    parser.add_argument("--kinds", nargs="*", default=["svm", "gbt", "mlp"])
    parser.add_argument("--shuffled", action="store_true")
    args = parser.parse_args()
    {
        "repro": cmd_repro,
        "metric": cmd_metric,
        "fisher": cmd_fisher,
        "clf": cmd_clf,
        "control": cmd_control,
        "ood": cmd_ood,
    }[args.command](args)
    logger.info("runs -> %s", RUNS)


if __name__ == "__main__":
    main()
