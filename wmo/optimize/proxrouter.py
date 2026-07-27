"""Faithful ProxRouter (arXiv 2510.09852): proximity-weighted nonparametric routing.

ProxRouter estimates each pool model's objective on a query as a weighted average over a
reference set (their eq 6): U_hat^(m)(x) = sum_i w_i(x) * V_i^(m), where V_i^(m) is model m's
mean observed objective at reference element i. Weights come in two stages (their Algorithm 1):

1. Minimum-variance priors: p_i ∝ 1 / Var[V_i]. For the KMeans reference set (KM-Prox) the
   paper estimates Var[V_i] ∝ s_i / n_i (s_i = intra-cluster spread, the mean cosine distance
   of members to their centroid; n_i = cluster size), giving p_i ∝ n_i / s_i. For the kNN
   reference set (kNN-Prox) the prior is uniform 1/k over the k nearest references, 0 elsewhere.
2. Proximity tilt: w_i(x) ∝ p_i(x) * exp(-phi_i(x) / tau), phi_i = cosine distance from the
   query to reference i. The paper runs 1/tau = 20 (KM: K=32; kNN: k=100).

This is a different scoring family from the Avengers rank router (`wmo.optimize.routing`):
it aggregates VALUE estimates over ALL reference elements (no top-k cluster cutoff, no
reciprocal-rank transform). Its published claim is drift robustness: +2.8-8.1pp outlier AUC
on Leave-Task-Out / Few-Shot-Outlier splits with inlier performance unchanged.

Deliberate deltas from the paper, all recorded in findings/r2.md before the first run:
(a) rewards may be graded, not binary; (b) the cost term of their objective (acc - lambda*cost)
is applied at DECISION time in fit-set cost_scale units, matching `rerank_policy`'s
fit-once-slide semantics (identical at lambda=0, where all headline comparisons run);
(c) spreads are floored (s_i + SPREAD_FLOOR) so a singleton or duplicate-text cluster cannot
claim infinite prior weight (the paper is silent on s_i = 0).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer

from wmo.optimize.cluster_labels import label_clusters, tokenize
from wmo.optimize.policy import EmbedderSpec, RoutingDecision
from wmo.providers.pool import PoolEntry

if TYPE_CHECKING:
    from wmo.optimize.outcomes import OutcomeMatrix

logger = logging.getLogger(__name__)

# Fallback floor for intra-cluster spread when every cluster is a singleton (no multi-member
# spread to estimate the noise scale from); see the data-driven floor in `fit_km_prox`.
SPREAD_FLOOR = 1e-3


class ProxReference(BaseModel):
    """One reference element: a cluster summary (KM-Prox) or a single fit query (kNN-Prox)."""

    vector: list[float]  # L2-normalized position in embedding space
    prior: float  # minimum-variance prior p_i, unnormalized
    rewards: dict[str, float]  # V_i^(m): per-model mean reward at this reference
    costs: dict[str, float]  # per-model mean cost (the objective's lambda leg + guard evidence)
    counts: dict[str, int] = Field(default_factory=dict)  # scored episodes behind each mean
    label: str = ""  # human-readable (majority scenario-id prefix), for the request log
    total: int = 0  # fit scenarios summarized by this reference


class ProxPolicy(BaseModel):
    """A fitted ProxRouter policy (either reference-set flavor)."""

    kind: Literal["km-prox", "knn-prox"]
    pool: list[PoolEntry]
    embedder: EmbedderSpec = Field(default_factory=EmbedderSpec)
    references: list[ProxReference]
    tau_inv: float = 20.0  # 1/tau; the paper's experimental value
    knn_k: int = 100  # kNN-Prox: nonzero prior on this many nearest references
    # Neighborhood rule for knn-prox: "fixed" = the paper's k nearest; "relative" = r1's
    # adaptive rule (keep references whose similarity exceeds rag_thres times the knn_k-th
    # best), which lets dense regions widen and sparse regions shrink the evidence set.
    neighbor_rule: str = "fixed"
    rag_thres: float = 0.95
    default_model: str  # fallback for degenerate inputs; also the guard baseline's home
    cost_scale: float = 0.0  # fit-set mean cost per scored episode (the lambda unit)
    # Empirical-Bayes shrinkage (r2 extension, NOT in the paper): each reference's per-model
    # mean is shrunk toward the model's global fit mean by episode support,
    # V_shrunk = (n*V + m*V_global) / (n + m). m=0 is the faithful paper behavior. With m>0 a
    # singleton reference (n=2 noisy episodes) barely moves the estimate off the global mean,
    # so thin-evidence picks fall below the guard margin BY CONSTRUCTION instead of by a hard
    # support cutoff.
    shrink_m: float = 0.0
    global_rewards: dict[str, float] = Field(default_factory=dict)  # per-model fit-set means
    global_costs: dict[str, float] = Field(default_factory=dict)
    fitted_from: str | None = None


def fit_km_prox(
    matrix: OutcomeMatrix,
    *,
    fit_ids: list[str] | None = None,
    embedder: EmbedderSpec | None = None,
    n_clusters: int = 32,
    seed: int = 42,
    tau_inv: float = 20.0,
    fitted_from: str | None = None,
    precomputed: np.ndarray | None = None,
) -> ProxPolicy:
    """Fit the KMeans-reference ProxRouter (paper defaults: K=32, 1/tau=20).

    Clustering mirrors `fit_rank_policy` exactly (k-means++ / elkan / max_iter=1000 / seeded)
    so KM-Prox vs rank comparisons isolate the scoring rule, not the clustering.
    `precomputed` (rows aligned to `fit_ids`) bypasses the spec's embedder: the offline path
    for embedding backends EmbedderSpec cannot rebuild yet (research runs only; the policy
    then cannot serve without the same vectors).
    """
    spec = embedder or EmbedderSpec()
    scenario_ids, embeddings = _fit_embeddings(matrix, fit_ids, spec, precomputed=precomputed)

    k = min(n_clusters, len(scenario_ids))
    kmeans = KMeans(
        n_clusters=k,
        random_state=seed,
        init="k-means++",
        n_init="auto",
        max_iter=1000,
        algorithm="elkan",
    )
    labels = kmeans.fit_predict(embeddings)
    centres = kmeans.cluster_centers_
    # Cosine geometry throughout: normalize centres so spreads and query distances share units.
    centres = Normalizer(norm="l2").transform(centres)

    cluster_of = {sid: int(label) for sid, label in zip(scenario_ids, labels, strict=True)}
    members: dict[int, list[int]] = {c: [] for c in range(k)}
    for row, sid in enumerate(scenario_ids):
        members[cluster_of[sid]].append(row)

    stats = _reference_stats(matrix, {sid: cluster_of[sid] for sid in scenario_ids})
    spreads = {
        cluster: float(np.mean(1.0 - embeddings[rows] @ centres[cluster]))
        for cluster, rows in members.items()
        if rows
    }
    # Data-driven spread floor (delta (c), findings/r2.md): the paper's Var[V_i] ∝ s_i/n_i
    # sends a singleton's variance to 0, which is exactly backwards (one sample carries the
    # FULL per-query noise). Flooring by the mean multi-member spread keeps a singleton's
    # prior at ~1 sample's worth of evidence instead of infinity.
    multi = [spreads[c] for c, rows in members.items() if len(rows) > 1 and c in spreads]
    floor = max(float(np.mean(multi)) if multi else SPREAD_FLOOR, SPREAD_FLOOR)
    tasks: dict[str, str] = {}
    for outcome in matrix.outcomes:
        tasks.setdefault(outcome.scenario_id, outcome.task)
    text_labels = label_clusters(
        [[tasks[scenario_ids[row]] for row in members[c]] for c in range(k)]
    )
    references: list[ProxReference] = []
    for cluster in range(k):
        rows = members[cluster]
        if not rows:
            continue
        n_i = len(rows)
        prior = n_i / (max(spreads[cluster], 0.0) + floor)
        rewards, costs, counts, label = stats.get(cluster, ({}, {}, {}, ""))
        label = label or text_labels[cluster]
        references.append(
            ProxReference(
                vector=[float(v) for v in centres[cluster]],
                prior=prior,
                rewards=rewards,
                costs=costs,
                counts=counts,
                label=label,
                total=n_i,
            )
        )
    logger.info("km-prox: %d references over %d fit scenarios", len(references), len(scenario_ids))
    return _finish_policy(
        matrix, "km-prox", references, spec, tau_inv, 0, set(scenario_ids), fitted_from
    )


def fit_knn_prox(
    matrix: OutcomeMatrix,
    *,
    fit_ids: list[str] | None = None,
    embedder: EmbedderSpec | None = None,
    knn_k: int = 100,
    tau_inv: float = 20.0,
    fitted_from: str | None = None,
    precomputed: np.ndarray | None = None,
) -> ProxPolicy:
    """Fit the kNN-reference ProxRouter (paper defaults: k=100, 1/tau=20).

    Every fit scenario becomes one reference with a uniform prior; the k-nearest cutoff is
    applied at decision time (`ProxScorer`), so the artifact stays split-independent.
    `precomputed` as in `fit_km_prox`.
    """
    spec = embedder or EmbedderSpec()
    scenario_ids, embeddings = _fit_embeddings(matrix, fit_ids, spec, precomputed=precomputed)
    stats = _reference_stats(matrix, {sid: index for index, sid in enumerate(scenario_ids)})
    tasks: dict[str, str] = {}
    for outcome in matrix.outcomes:
        tasks.setdefault(outcome.scenario_id, outcome.task)
    references: list[ProxReference] = []
    for row, sid in enumerate(scenario_ids):
        rewards, costs, counts, label = stats.get(row, ({}, {}, {}, ""))
        if not rewards:
            continue  # an unscored fit scenario carries no evidence; skip it, logged below
        if not label:
            # Per-query reference on prefix-less ids: first distinctive task terms.
            label = " ".join(tokenize(tasks.get(sid, ""))[:3])
        references.append(
            ProxReference(
                vector=[float(v) for v in embeddings[row]],
                prior=1.0,
                rewards=rewards,
                costs=costs,
                counts=counts,
                label=label,
                total=1,
            )
        )
    if len(references) < len(scenario_ids):
        logger.info(
            "knn-prox: dropped %d unscored fit scenarios", len(scenario_ids) - len(references)
        )
    return _finish_policy(
        matrix, "knn-prox", references, spec, tau_inv, knn_k, set(scenario_ids), fitted_from
    )


class ProxScorer:
    """Vectorized decision path for one fitted policy (shared by serve and batch eval).

    Precomputes the reference matrices once; `estimates` runs the paper's Algorithm 1 on an
    already-embedded query. Models missing from some references renormalize their weights over
    the references that DO carry them (dense matrices never hit this; sparse ones stay sane).
    """

    def __init__(self, policy: ProxPolicy) -> None:
        self.policy = policy
        self.models = [entry.name for entry in policy.pool]
        self.vectors = np.asarray([ref.vector for ref in policy.references])  # [R, D]
        self.priors = np.asarray([ref.prior for ref in policy.references], dtype=np.float64)
        index = {name: column for column, name in enumerate(self.models)}
        shape = (len(policy.references), len(self.models))
        self.rewards = np.full(shape, np.nan)
        self.costs = np.full(shape, np.nan)
        counts = np.zeros(shape)
        for row, ref in enumerate(policy.references):
            for name, value in ref.rewards.items():
                if name in index:
                    self.rewards[row, index[name]] = value
                    self.costs[row, index[name]] = ref.costs.get(name, 0.0)
                    counts[row, index[name]] = ref.counts.get(name, 1)
        if policy.shrink_m > 0.0 and policy.global_rewards:
            # Empirical-Bayes shrinkage toward the global fit means (see ProxPolicy.shrink_m).
            # A cell with NO local evidence becomes exactly the global prior, so shrinkage
            # also fills sparse matrices with the honest "we know nothing local" estimate.
            m = policy.shrink_m
            global_r = np.asarray([policy.global_rewards.get(name, np.nan) for name in self.models])
            global_c = np.asarray([policy.global_costs.get(name, np.nan) for name in self.models])
            local_r = np.where(np.isnan(self.rewards), 0.0, self.rewards)
            local_c = np.where(np.isnan(self.costs), 0.0, self.costs)
            with np.errstate(invalid="ignore"):
                self.rewards = (counts * local_r + m * global_r[None, :]) / (counts + m)
                self.costs = (counts * local_c + m * global_c[None, :]) / (counts + m)
        self.known = ~np.isnan(self.rewards)  # [R, M]

    def _weights(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
        """(tilted weights, untilted priors, nearest index, nearest distance) for one query."""
        query = np.asarray(query, dtype=np.float64)
        norm = float(np.linalg.norm(query))
        if norm > 0:
            query = query / norm
        distance = 1.0 - self.vectors @ query  # [R], cosine distance (vectors are unit)
        nearest = int(np.argmin(distance))

        priors = self.priors.copy()
        if self.policy.kind == "knn-prox":
            k = min(self.policy.knn_k, len(priors))
            if self.policy.neighbor_rule == "relative":
                sims = 1.0 - distance
                kth = np.partition(sims, len(sims) - k)[len(sims) - k]
                keep = sims > self.policy.rag_thres * kth
                if not keep.any():
                    keep = sims >= sims.max()
                priors = np.where(keep, priors, 0.0)
            else:
                cutoff = np.partition(distance, k - 1)[k - 1]
                priors = np.where(distance <= cutoff, priors, 0.0)
        # exp(-phi/tau) with the max shifted out for numerical safety; the shift cancels in
        # the per-model normalization.
        logits = -self.policy.tau_inv * distance
        tilt = np.exp(logits - logits.max())
        return priors * tilt, priors, nearest, float(distance[nearest])

    def estimates(self, query: np.ndarray) -> tuple[dict[str, float], dict[str, float], int]:
        """Per-model (U_hat reward, U_hat cost) plus the nearest reference's index."""
        weights, _priors, nearest, _near_dist = self._weights(query)
        masked = np.where(self.known, weights[:, None], 0.0)  # [R, M]
        denom = masked.sum(axis=0)  # [M]
        rewards = np.where(denom > 0, np.nansum(self.rewards * masked, axis=0), np.nan)
        costs = np.where(denom > 0, np.nansum(self.costs * masked, axis=0), np.nan)
        with np.errstate(invalid="ignore"):
            rewards = rewards / np.where(denom > 0, denom, 1.0)
            costs = costs / np.where(denom > 0, denom, 1.0)
        reward_out = {
            model: float(rewards[column])
            for column, model in enumerate(self.models)
            if denom[column] > 0
        }
        cost_out = {
            model: float(costs[column])
            for column, model in enumerate(self.models)
            if denom[column] > 0
        }
        return reward_out, cost_out, nearest

    def decide(
        self,
        query: np.ndarray,
        *,
        lam: float = 0.0,
        guard_model: str | None = None,
        guard_margin: float = 0.0,
        guard_z: float | None = None,
        min_pairs: float = 8.0,
        abstain_distance: float | None = None,
    ) -> RoutingDecision:
        """Argmax of U_hat = reward - lam * cost / cost_scale, with the baseline guard.

        Two guard modes against `guard_model`, applied when the argmax differs from it:

        - Margin (default): the pick's score must beat the baseline's by `guard_margin`
          (doubled when the pick's estimated cost is higher). Flat: blind to evidence volume.
        - Paired z (when `guard_z` is set; overrides the margin): the mean of the
          per-reference paired differences d_i = V_i^pick - V_i^base over the UNTILTED
          neighborhood (the prior support: k nearest for knn, prior-weighted clusters for
          km) must clear `guard_z` weighted standard errors (doubled when pricier), with the
          effective sample size (sum w)^2 / sum w^2 at least `min_pairs`. The tilt proposes
          (bias-reduced estimate), the neighborhood disposes (statistical power): testing
          with the tilted weights would shrink the effective sample to the 2-3 nearest
          references and revert everything. Support-aware: a wide gap on thin evidence
          reverts, a modest gap on massive evidence routes.

        Guarding at decision time (not fit time) keeps the artifact guard-free, so one fit
        serves any baseline.
        """
        weights, test_weights, nearest, near_dist = self._weights(query)
        if (
            abstain_distance is not None
            and guard_model is not None
            and near_dist > abstain_distance
        ):
            # The query sits beyond the fit distribution's support: every reference is an
            # extrapolation, so the honest answer is the baseline, not a certified-looking
            # gap measured on irrelevant evidence.
            ref = self.policy.references[nearest]
            return RoutingDecision(
                model=guard_model,
                cluster_id=nearest,
                cluster_label=ref.label,
                reason=(
                    f"{self.policy.kind}: abstained to baseline "
                    f"(nearest reference {near_dist:.3f} > floor {abstain_distance:.3f})"
                ),
            )
        masked = np.where(self.known, weights[:, None], 0.0)
        denom = masked.sum(axis=0)
        with np.errstate(invalid="ignore"):
            reward_est = np.nansum(self.rewards * masked, axis=0) / np.where(denom > 0, denom, 1)
            cost_est = np.nansum(self.costs * masked, axis=0) / np.where(denom > 0, denom, 1)
        rewards = {m: float(reward_est[c]) for c, m in enumerate(self.models) if denom[c] > 0}
        costs = {m: float(cost_est[c]) for c, m in enumerate(self.models) if denom[c] > 0}
        if not rewards:
            return RoutingDecision(
                model=self.policy.default_model, reason="prox: no reference evidence"
            )
        scale = self.policy.cost_scale or 1.0
        scores = {model: rewards[model] - lam * costs.get(model, 0.0) / scale for model in rewards}
        pool_order = {entry.name: index for index, entry in enumerate(self.policy.pool)}
        pick = max(scores.items(), key=lambda kv: (kv[1], -pool_order[kv[0]]))[0]
        reason = f"{self.policy.kind}: value aggregation (1/tau={self.policy.tau_inv:g})"
        if guard_model is not None and pick != guard_model:
            pricier = costs.get(pick, 0.0) > costs.get(guard_model, float("inf"))
            if guard_z is not None:
                z = guard_z * (2 if pricier else 1)
                if not self._paired_gap_clears(test_weights, pick, guard_model, z, min_pairs):
                    pick = guard_model
                    reason = f"{self.policy.kind}: z-guard reverted to baseline"
            else:
                margin = guard_margin * (2 if pricier else 1)
                if scores.get(guard_model) is None or scores[pick] <= scores[guard_model] + margin:
                    pick = guard_model
                    reason = f"{self.policy.kind}: guard reverted to baseline"
        ref = self.policy.references[nearest]
        return RoutingDecision(
            model=pick,
            cluster_id=nearest,
            cluster_label=ref.label,
            reason=reason,
        )

    def _paired_gap_clears(
        self, weights: np.ndarray, pick: str, base: str, z: float, min_pairs: float
    ) -> bool:
        """Weighted paired test: does pick's gap over base clear z standard errors?"""
        columns = {name: column for column, name in enumerate(self.models)}
        if pick not in columns or base not in columns:
            return False
        pi, bi = columns[pick], columns[base]
        mask = self.known[:, pi] & self.known[:, bi] & (weights > 0)
        if not mask.any():
            return False
        w = weights[mask]
        d = self.rewards[mask, pi] - self.rewards[mask, bi]
        sw = float(w.sum())
        n_eff = sw**2 / float((w**2).sum())
        if n_eff < min_pairs:
            return False
        gap = float((w * d).sum() / sw)
        variance = float((w * (d - gap) ** 2).sum() / sw)
        se = (variance / n_eff) ** 0.5
        return gap > z * se


def support_floor(policy: ProxPolicy, *, quantile: float = 0.95) -> float:
    """Abstention floor from the policy's own reference geometry (no test peeking).

    The `quantile` of each reference's cosine distance to its nearest OTHER reference: a
    query whose nearest reference is farther than this sits outside the fit distribution's
    support. Meaningful for knn-prox (references ARE the fit points); for km-prox centroids
    it under-estimates the support radius, so callers should prefer the knn policy's floor.
    """
    vectors = np.asarray([ref.vector for ref in policy.references])
    if len(vectors) < 2:
        return float("inf")
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -np.inf)
    nn_distance = 1.0 - sims.max(axis=1)
    return float(np.quantile(nn_distance, quantile))


def _fit_embeddings(
    matrix: OutcomeMatrix,
    fit_ids: list[str] | None,
    spec: EmbedderSpec,
    precomputed: np.ndarray | None = None,
) -> tuple[list[str], np.ndarray]:
    """Resolve fit scenario ids and their L2-normalized embeddings."""
    scenario_tasks: dict[str, str] = {}
    for outcome in matrix.outcomes:
        scenario_tasks.setdefault(outcome.scenario_id, outcome.task)
    if fit_ids is not None:
        wanted = set(fit_ids)
        missing = wanted - scenario_tasks.keys()
        if missing:
            raise ValueError(f"fit_ids not in the matrix: {sorted(missing)[:5]}")
        scenario_tasks = {sid: scenario_tasks[sid] for sid in fit_ids}
    if not scenario_tasks:
        raise ValueError("no scenarios to fit on")
    scenario_ids = list(scenario_tasks)
    if precomputed is not None:
        if len(precomputed) != len(scenario_ids):
            raise ValueError(
                f"precomputed embeddings have {len(precomputed)} rows, "
                f"need {len(scenario_ids)} (aligned to fit_ids)"
            )
        embeddings = np.asarray(precomputed, dtype=np.float64)
    else:
        embeddings = np.asarray(spec.build().embed([scenario_tasks[sid] for sid in scenario_ids]))
    return scenario_ids, Normalizer(norm="l2").fit_transform(embeddings)


def _reference_stats(
    matrix: OutcomeMatrix, group_of: dict[str, int]
) -> dict[int, tuple[dict[str, float], dict[str, float], dict[str, int], str]]:
    """Per-group per-model (mean reward, mean cost, count) plus a majority-prefix label."""
    sums: dict[int, dict[str, tuple[float, float, int]]] = {}
    prefixes: dict[int, dict[str, int]] = {}
    for outcome in matrix.outcomes:
        group = group_of.get(outcome.scenario_id)
        if group is None or outcome.reward is None:
            continue
        by_model = sums.setdefault(group, {})
        reward_sum, cost_sum, count = by_model.get(outcome.model, (0.0, 0.0, 0))
        by_model[outcome.model] = (
            reward_sum + outcome.reward,
            cost_sum + outcome.cost_usd,
            count + 1,
        )
        if ":" in outcome.scenario_id:
            counter = prefixes.setdefault(group, {})
            prefix = outcome.scenario_id.split(":", 1)[0]
            counter[prefix] = counter.get(prefix, 0) + 1
    out: dict[int, tuple[dict[str, float], dict[str, float], dict[str, int], str]] = {}
    for group, by_model in sums.items():
        rewards = {model: rs / count for model, (rs, _cs, count) in by_model.items()}
        costs = {model: cs / count for model, (_rs, cs, count) in by_model.items()}
        counts = {model: count for model, (_rs, _cs, count) in by_model.items()}
        label = ""
        if prefixes.get(group):
            label = max(prefixes[group].items(), key=lambda kv: kv[1])[0]
        out[group] = (rewards, costs, counts, label)
    return out


def _finish_policy(
    matrix: OutcomeMatrix,
    kind: Literal["km-prox", "knn-prox"],
    references: list[ProxReference],
    spec: EmbedderSpec,
    tau_inv: float,
    knn_k: int,
    fit_set: set[str],
    fitted_from: str | None,
) -> ProxPolicy:
    if not references:
        raise ValueError("no scored references; cannot fit a prox policy")
    sums: dict[str, tuple[float, float, int]] = {}
    total_cost, total_count = 0.0, 0
    for outcome in matrix.outcomes:
        if outcome.scenario_id not in fit_set or outcome.reward is None:
            continue
        reward_sum, cost_sum, count = sums.get(outcome.model, (0.0, 0.0, 0))
        sums[outcome.model] = (
            reward_sum + outcome.reward,
            cost_sum + outcome.cost_usd,
            count + 1,
        )
        total_cost += outcome.cost_usd
        total_count += 1
    global_rewards = {m: rs / count for m, (rs, _cs, count) in sums.items()}
    global_costs = {m: cs / count for m, (_rs, cs, count) in sums.items()}
    pool_order = {entry.name: index for index, entry in enumerate(matrix.pool)}
    default = min(global_rewards, key=lambda m: (-global_rewards[m], pool_order[m]))
    return ProxPolicy(
        kind=kind,
        pool=matrix.pool,
        embedder=spec,
        references=references,
        tau_inv=tau_inv,
        knn_k=knn_k or 100,
        default_model=default,
        cost_scale=(total_cost / total_count) if total_count else 0.0,
        global_rewards=global_rewards,
        global_costs=global_costs,
        fitted_from=fitted_from,
    )
