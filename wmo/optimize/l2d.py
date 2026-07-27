"""Learning-to-defer routing: the consistent plug-in rule over an outcome matrix.

Route-vs-fallback is learning-to-defer (Mozannar & Sontag, ICML'20) with the fit-chosen
best-single model as the expert. Our margin and z guards are confidence-threshold deferral
rules of the family M&S proved INCONSISTENT: they can systematically over-defer no matter how
much data arrives. The consistent alternative instantiated here follows the one-vs-all
construction (Verma & Nalisnick, ICML'22; multiple-experts form):

- one head per pool model, fit with a proper loss on that model's per-scenario reward
  (ridge regression on graded rewards: the squared loss is proper, so each head estimates a
  calibrated E[reward | x]; per-head alpha via RidgeCV's fit-side leave-one-out GCV),
- the defer option IS the baseline model's own head (expert correctness = baseline reward),
- decision rule: argmax over cost-adjusted heads, score_m = E[r_m|x] - lam * cost_m / scale,
  with NO margin; exact ties break to the baseline, then cheaper, then pool order.

The deliberate contrast with `ProxScorer.decide`'s guards is the absence of any threshold:
consistency says the plug-in argmax over calibrated heads converges to the Bayes
route-or-defer rule, and any fixed margin biases toward deferral forever.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from sklearn.linear_model import RidgeCV

if TYPE_CHECKING:
    from wmo.optimize.outcomes import OutcomeMatrix

logger = logging.getLogger(__name__)

ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


class L2DDeferral:
    """Fitted per-model reward heads + the consistent deferral decision rule."""

    def __init__(
        self,
        models: list[str],
        baseline: str,
        heads: dict[str, RidgeCV],
        mean_costs: dict[str, float],
        cost_scale: float,
        pool_order: dict[str, int],
    ) -> None:
        self.models = models
        self.baseline = baseline
        self.heads = heads
        self.mean_costs = mean_costs
        self.cost_scale = cost_scale
        self.pool_order = pool_order

    def scores(self, query: np.ndarray, *, lam: float = 0.0) -> dict[str, float]:
        """Cost-adjusted calibrated reward estimates per model (clipped to [0, 1])."""
        query = np.asarray(query, dtype=np.float64).reshape(1, -1)
        out: dict[str, float] = {}
        for model, head in self.heads.items():
            estimate = float(np.clip(head.predict(query)[0], 0.0, 1.0))
            out[model] = estimate - lam * self.mean_costs.get(model, 0.0) / self.cost_scale
        return out

    def decide(self, query: np.ndarray, *, lam: float = 0.0) -> str:
        """The consistent rule: argmax, ties to the baseline then cheaper. No margin."""
        scores = self.scores(query, lam=lam)

        def key(model: str) -> tuple[float, int, float, int]:
            return (
                -round(scores[model], 9),
                0 if model == self.baseline else 1,
                self.mean_costs.get(model, 0.0),
                self.pool_order[model],
            )

        return min(scores, key=key)


def fit_l2d(
    matrix: OutcomeMatrix,
    *,
    fit_ids: list[str],
    embeddings: np.ndarray,
    baseline: str,
) -> L2DDeferral:
    """Fit the per-model heads on `fit_ids` (rows of `embeddings` align to `fit_ids`)."""
    if len(embeddings) != len(fit_ids):
        raise ValueError(f"{len(embeddings)} embedding rows for {len(fit_ids)} fit ids")
    fit_set = set(fit_ids)
    row_of = {sid: row for row, sid in enumerate(fit_ids)}
    cells: dict[tuple[str, str], list[float]] = {}
    costs: dict[str, list[float]] = {}
    total_cost, total_count = 0.0, 0
    for outcome in matrix.outcomes:
        if outcome.scenario_id not in fit_set or outcome.reward is None:
            continue
        cells.setdefault((outcome.scenario_id, outcome.model), []).append(outcome.reward)
        costs.setdefault(outcome.model, []).append(outcome.cost_usd)
        total_cost += outcome.cost_usd
        total_count += 1
    if baseline not in costs:
        raise ValueError(f"baseline '{baseline}' has no scored fit episodes")

    features = np.asarray(embeddings, dtype=np.float64)
    heads: dict[str, RidgeCV] = {}
    models = [entry.name for entry in matrix.pool]
    for model in models:
        rows, targets = [], []
        for sid in fit_ids:
            values = cells.get((sid, model))
            if values:
                rows.append(row_of[sid])
                targets.append(sum(values) / len(values))
        if len(rows) < 3:
            continue  # a head needs a minimum of evidence; absent models never win argmax
        head = RidgeCV(alphas=ALPHAS)
        head.fit(features[rows], np.asarray(targets))
        heads[model] = head
    if baseline not in heads:
        raise ValueError(f"baseline '{baseline}' has too few scored fit scenarios for a head")
    logger.info(
        "l2d: %d heads over %d fit scenarios (baseline %s)", len(heads), len(fit_ids), baseline
    )
    return L2DDeferral(
        models=models,
        baseline=baseline,
        heads=heads,
        mean_costs={m: sum(v) / len(v) for m, v in costs.items()},
        cost_scale=(total_cost / total_count) if total_count else 1.0,
        pool_order={entry.name: index for index, entry in enumerate(matrix.pool)},
    )
