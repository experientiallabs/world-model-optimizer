"""Routability triage at ingest: predict the capture verdict from tasks alone.

Items 3-5 of the ingest-verdict assignment. `verdict()` is the pure function destined for
`wmo ingest`; everything else is its validation harness.

- `synthetic`: build ours9 slices of known measured routability (sizes x widths x seeds),
  compute ingest-time features (size, prior-transfer via kNN into the non-ours9 banks) and
  ground-truth labels (measured on the slice's full outcome information: monopoly margin,
  promoted-router lift over 5 split seeds). Emits a TSV for threshold selection.
- `real`: score every real corpus leave-family-out with the PRE-REGISTERED thresholds and
  print the confusion table. Run only AFTER thresholds are written to findings.

Ground-truth label rules (pre-registered in findings/r1.md before any scoring):
  PIN   if M >= 0.10           (top-model mean margin over #2 on the full slice)
  ROUTE if mean L >= +0.005 and >= 3/5 seeds positive  (L = adapt3 router dacc)
  KNOB  otherwise              (TOO-SMALL: n < 40, deterministic size rung)

Ladder (thresholds fitted on synthetic ONLY):
  PIN if M_hat >= theta_M; else TOO-SMALL if n < 40; else ROUTE if D_hat >= theta_D;
  else KNOB.

$0: every embedding comes from the routing corpus cache/.
Usage: uv run python .agents/scripts/r1_routability.py synthetic|real
"""

from __future__ import annotations

import importlib.util
import logging
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from wmo.optimize.outcomes import OutcomeMatrix
from wmo.research.routerbench import best_single_model
from wmo.research.routing_corpus import routing_data

_spec = importlib.util.spec_from_file_location(
    "r1_retrieval_ablations", Path(__file__).with_name("r1_retrieval_ablations.py")
)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("r1.triage")

DATA = routing_data()
OUT = DATA / "findings" / "r1_routability_synthetic.tsv"
BANK_CORPORA = [
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
    "financebench-s80",
    "tau-bench-s80",
    "tau-bench-real",
]
REAL_TARGETS = [*BANK_CORPORA, "routerbench-ours9"]
SIZES = [25, 50, 80, 150]
WIDTHS = [1, 2, 4, 8]
SLICE_SEEDS = range(10)
KNN_K = 8

# PRE-REGISTERED ladder thresholds (fitted on the synthetic set ONLY; written to
# findings/r1.md before any real corpus was scored).
# THETA_M = synthetic KNOB M_hat p95 (0.085) rounded up: <=5% false-PIN by construction.
# ROUTE_MIN_N = the measured size crossover: ours9 slices are ROUTE 0/8 at n<=200,
# sporadic (1-2/8) at 300-800, and 6/8 only at n=1000. Below that, predicted-ROUTE is
# noise, so the ladder abstains from approving capture on it.
THETA_M = 0.09
THETA_D = 0.30
TOO_SMALL_N = 40
ROUTE_MIN_N = 1000


def verdict(n_tasks: int, m_hat: float, d_hat: float) -> str:
    """The pure ingest-time verdict ladder (port target for `wmo ingest`)."""
    if m_hat >= THETA_M:
        return "PIN"
    if n_tasks < TOO_SMALL_N:
        return "TOO-SMALL"
    if n_tasks >= ROUTE_MIN_N and d_hat >= THETA_D:
        return "ROUTE"
    return "KNOB"


def _ctx(stem: str, name: str | None = None):  # noqa: ANN202
    matrix = OutcomeMatrix.load(DATA / "matrices" / f"{stem}_matrix.json")
    ctx_name = name or (
        f"wm-{stem}"
        if f"wm-{stem}-oai3l-tasks.npy" in {p.name for p in (DATA / "cache").glob("*.npy")}
        else stem
    )
    return _mod.MatrixContext(matrix, ctx_name, embed="openai", embed_replies=False)


class SliceCtx:
    """route()-compatible view of a subset of a parent context's scenarios."""

    def __init__(self, parent, ids: list[str]):  # noqa: ANN001
        wanted = set(ids)
        self.model_names = parent.model_names
        self.tasks = {s: parent.tasks[s] for s in ids}
        self.task_vecs = {s: parent.task_vecs[s] for s in ids}
        self.rewards_cell = {
            key: value for key, value in parent.rewards_cell.items() if key[0] in wanted
        }
        self.matrix = parent.matrix  # only mean_cost uses outcomes; filter there
        self._wanted = wanted

    def mean_cost(self, fit_ids: list[str]) -> dict[str, float]:
        fit = set(fit_ids) & self._wanted
        sums: dict[str, tuple[float, int]] = {}
        for outcome in self.matrix.outcomes:
            if outcome.scenario_id in fit and outcome.reward is not None:
                total, count = sums.get(outcome.model, (0.0, 0))
                sums[outcome.model] = (total + outcome.cost_usd, count + 1)
        return {m: t / c for m, (t, c) in sums.items() if c}


def measured_label(ctx, ids: list[str]) -> tuple[str, float, float]:  # noqa: ANN001
    """Ground truth from full information: (label, monopoly margin M, router lift L)."""
    sub = SliceCtx(ctx, ids)
    means: dict[str, list[float]] = defaultdict(list)
    for (sid, model), cell in sub.rewards_cell.items():
        means[model].append(sum(cell) / len(cell))
    model_means = sorted((float(np.mean(v)) for v in means.values()), reverse=True)
    margin = model_means[0] - model_means[1] if len(model_means) > 1 else 1.0
    if margin >= 0.10:
        return "PIN", margin, 0.0

    lifts = []
    for seed in range(5):
        rng = random.Random(seed)
        shuffled = ids[:]
        rng.shuffle(shuffled)
        cut = max(1, int(0.7 * len(shuffled)))
        fit_ids, test_ids = sorted(shuffled[:cut]), sorted(shuffled[cut:])
        best, _, _ = best_single_model(sub.matrix, fit_ids=fit_ids, eval_ids=test_ids)
        rag = min(50, max(4, math.ceil(len(fit_ids) / 2)))
        params = _mod.RetrievalParams(
            second_route=False,
            guard="stat",
            z=0.5,
            rag_num=rag,
            min_pairs=min(8, max(3, rag // 2)),
            se_floor=True,
        )
        picks = _mod.route(sub, params, fit_ids, test_ids, best)

        def acc(choice) -> float:  # noqa: ANN001
            values = []
            for sid in test_ids:
                cell = sub.rewards_cell.get((sid, choice(sid)))
                if cell:
                    values.append(sum(cell) / len(cell))
            return float(np.mean(values)) if values else 0.0

        lifts.append(acc(lambda s, p=picks: p[s]) - acc(lambda _s, b=best: b))
    mean_lift = float(np.mean(lifts))
    positive = sum(1 for lift in lifts if lift > 1e-12)
    if mean_lift >= 0.005 and positive >= 3:
        return "ROUTE", margin, mean_lift
    return "KNOB", margin, mean_lift


def build_bank(exclude_family: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pooled (embeddings, per-model cell means) over banks outside the excluded family."""
    vecs = []
    rows = []
    models: list[str] | None = None
    for stem in [*BANK_CORPORA, "routerbench-ours9"]:
        family = stem.replace("-s80", "").replace("-real", "")
        if family == exclude_family:
            continue
        ctx = _ctx(stem, name="routerbench-ours9" if stem == "routerbench-ours9" else None)
        models = ctx.model_names
        for sid in ctx.scenario_ids:
            vecs.append(ctx.task_vecs[sid])
            rows.append(
                [
                    (sum(c) / len(c)) if (c := ctx.rewards_cell.get((sid, m))) else np.nan
                    for m in ctx.model_names
                ]
            )
    assert models is not None
    return np.stack(vecs), np.asarray(rows), models


def prior_features(
    task_vecs: list[np.ndarray], bank_vecs: np.ndarray, bank_rewards: np.ndarray
) -> tuple[float, float, float]:
    """(M_hat, D_hat, H_hat) from kNN prior transfer; tasks only, no target outcomes."""
    profiles = []
    for vec in task_vecs:
        sims = bank_vecs @ vec
        top = np.argsort(sims)[-KNN_K:]
        weights = np.clip(sims[top], 0.0, None)[:, None]
        rewards = bank_rewards[top]
        scored = ~np.isnan(rewards)
        weight_totals = (scored * weights).sum(axis=0)
        totals = (np.where(scored, np.nan_to_num(rewards), 0.0) * weights).sum(axis=0)
        profile = np.where(weight_totals > 0, totals / np.maximum(weight_totals, 1e-12), np.nan)
        profiles.append(profile)
    matrix = np.asarray(profiles)
    model_means = np.nanmean(matrix, axis=0)
    order = np.argsort(model_means)
    m_hat = float(model_means[order[-1]] - model_means[order[-2]])
    overall_best = int(order[-1])
    winners = np.nanargmax(np.where(np.isnan(matrix), -1.0, matrix), axis=1)
    d_hat = float(np.mean(winners != overall_best))
    h_hat = float(np.nanmean(np.nanmax(matrix, axis=1)) - model_means[order[-1]])
    return m_hat, d_hat, h_hat


def cmd_synthetic() -> None:
    ctx = _ctx("routerbench-ours9", name="routerbench-ours9")
    by_prefix: dict[str, list[str]] = defaultdict(list)
    for sid in ctx.scenario_ids:
        by_prefix[sid.split(":", 1)[0]].append(sid)
    prefixes = [p for p, ids in by_prefix.items() if len(ids) >= 10]
    bank_vecs, bank_rewards, _ = build_bank(exclude_family="routerbench-ours9")

    rows = ["size\twidth\tseed\tlabel\tM\tL\tM_hat\tD_hat\tH_hat"]
    for size in SIZES:
        for width in WIDTHS:
            for seed in SLICE_SEEDS:
                rng = random.Random(1000 * size + 100 * width + seed)
                chosen = rng.sample(prefixes, min(width, len(prefixes)))
                pool = [sid for p in chosen for sid in by_prefix[p]]
                if len(pool) < size:
                    continue
                ids = sorted(rng.sample(pool, size))
                label, margin, lift = measured_label(ctx, ids)
                m_hat, d_hat, h_hat = prior_features(
                    [ctx.task_vecs[s] for s in ids], bank_vecs, bank_rewards
                )
                rows.append(
                    f"{size}\t{width}\t{seed}\t{label}\t{margin:.4f}\t{lift:.4f}"
                    f"\t{m_hat:.4f}\t{d_hat:.4f}\t{h_hat:.4f}"
                )
    OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
    logger.info("%d slices -> %s", len(rows) - 1, OUT)


def cmd_real() -> None:
    logger.info(
        "PRE-REGISTERED thresholds: theta_M=%.3f theta_D=%.3f n<%d", THETA_M, THETA_D, TOO_SMALL_N
    )
    confusion: dict[tuple[str, str], list[str]] = defaultdict(list)
    for stem in REAL_TARGETS:
        name = "routerbench-ours9" if stem == "routerbench-ours9" else None
        ctx = _ctx(stem, name=name)
        ids = list(ctx.scenario_ids)
        family = stem.replace("-s80", "").replace("-real", "")
        if len(ids) < TOO_SMALL_N:
            truth = "TOO-SMALL"
            margin = lift = float("nan")
        else:
            truth, margin, lift = measured_label(ctx, ids)
        bank_vecs, bank_rewards, _ = build_bank(exclude_family=family)
        m_hat, d_hat, _h = prior_features([ctx.task_vecs[s] for s in ids], bank_vecs, bank_rewards)
        predicted = verdict(len(ids), m_hat, d_hat)
        confusion[(truth, predicted)].append(stem)
        logger.info(
            "%-20s n=%4d truth=%-9s pred=%-9s (M=%.3f L=%+.4f | M_hat=%.3f D_hat=%.2f)",
            stem,
            len(ids),
            truth,
            predicted,
            margin,
            lift,
            m_hat,
            d_hat,
        )
    approves = [k for k in confusion if k[1] == "ROUTE"]
    false_approve = sum(len(confusion[k]) for k in approves if k[0] != "ROUTE")
    total_approve = sum(len(confusion[k]) for k in approves)
    logger.info(
        "confusion cells: %s",
        {f"{t}->{p}": v for (t, p), v in sorted(confusion.items())},
    )
    logger.info(
        "FALSE-APPROVE rate: %d/%d predicted-ROUTE were not ROUTE",
        false_approve,
        total_approve,
    )


def main() -> None:
    command = next((a for a in sys.argv[1:] if not a.startswith("-")), "synthetic")
    {"synthetic": cmd_synthetic, "real": cmd_real}[command]()


if __name__ == "__main__":
    main()
