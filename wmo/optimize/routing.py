"""The routing fitter: an OutcomeMatrix in, an Avengers-style rank policy out.

Faithful replication of the reference implementation (arXiv 2505.19797,
github.com/ZhangYiqun018/Avengers, core/generate_rank_router.py), stage by stage:

1. Embed the fit scenarios' task texts and L2-normalize (their `Normalizer(norm="l2")`).
2. K-means the normalized embeddings (their exact configuration: k-means++, n_init="auto",
   max_iter=1000, elkan, seeded).
3. Per cluster, rank the pool models by mean reward over the cluster's SCORED episodes
   (their correct/total accuracy, generalized to graded rewards; identical on binary data).
   Models with no scored episode in a cluster are absent from its ranking and fall back to
   `default_rank` at selection time, matching the reference.

Deliberate deltas from the reference, both recorded here so the post-hoc comparison audit has
the list in one place: (a) rewards may be graded, not just 0/1; (b) clusters get a
human-readable label for the request log (the majority scenario-id prefix, or the cluster's
distinctive c-TF-IDF terms when the ids carry no prefix; see `wmo.optimize.cluster_labels`); the
reference has no labels. Cost plays NO part in fitting, exactly like the reference; the cost-aware
variant (Avengers-Pro's alpha) is the first planned variation AFTER replication is validated.

`evaluate_policy` replays a policy of ANY kind (static, rank, knn, linear) over a matrix through
the same decision code serving uses, so benchmark numbers measure the deployed selection path,
not a reimplementation. The kNN family's fit lives in `wmo.optimize.knn`, which explains why it
is a separate module rather than another mode here.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer

from wmo.optimize.cluster_labels import label_clusters
from wmo.optimize.policy import (
    DEFAULT_BETA,
    DEFAULT_RANK,
    DEFAULT_TOP_K_CLUSTERS,
    ClusterRanking,
    EmbedderSpec,
    RoutingDecision,
    RoutingPolicy,
    knn_decision,
    linear_decision,
    rank_decision,
)

if TYPE_CHECKING:
    from wmo.optimize.outcomes import OutcomeMatrix
    from wmo.providers.base import Embedder

logger = logging.getLogger(__name__)


def _guarded_ranking(
    ranking: list[str],
    *,
    guard_model: str,
    scores: dict[str, float],
    costs: dict[str, float],
    support: dict[str, int],
    min_support: int,
    guard_margin: float,
) -> list[str]:
    """Only-replace-if-better, per cluster: the router's worst case must be the guard model.

    A cluster keeps its own winner ONLY when that winner beats the guard's in-cluster evidence by
    `guard_margin` (doubled when the winner is also pricier, which kills the confidently-wrong
    pricier-and-worse mode) on at least `min_support` scored EPISODES. Otherwise the guard leads.

    Absence of guard evidence is not a pass: when the guard model has no scored episode in the
    cluster there is nothing to beat, so the cluster reverts rather than admitting a challenger
    that was never compared. A cluster with no scored evidence at all is left alone (its ranking
    is the fitter's global fallback, not a claim about this cluster).
    """
    if not ranking or not scores:
        return ranking
    top = ranking[0]
    if top == guard_model:
        return ranking
    reverted = [guard_model, *[model for model in ranking if model != guard_model]]
    if guard_model not in scores:
        return reverted
    if support.get(top, 0) < min_support:
        return reverted
    margin = guard_margin
    if costs.get(top, 0.0) > costs.get(guard_model, float("inf")):
        margin = 2 * guard_margin
    if scores[top] <= scores[guard_model] + margin:
        return reverted
    return ranking


def fit_rank_policy(
    matrix: OutcomeMatrix,
    *,
    fit_ids: list[str] | None = None,
    embedder: EmbedderSpec | None = None,
    n_clusters: int = 64,
    seed: int = 42,
    top_k_clusters: int = DEFAULT_TOP_K_CLUSTERS,
    beta: float = DEFAULT_BETA,
    default_rank: int = DEFAULT_RANK,
    default_model: str | None = None,
    guard_model: str | None = None,
    min_support: int = 0,
    guard_margin: float = 0.0,
    fitted_from: str | None = None,
) -> RoutingPolicy:
    """Fit a rank policy on `matrix` (restricted to `fit_ids` when given).

    Defaults mirror the reference (n_clusters=64, seed=42, top_k=2, beta=6.0).
    `default_model` falls back to the best overall mean reward on the fit scenarios
    (ties break by pool order).

    `guard_model` turns on the only-replace-if-better guard (`_guarded_ranking`): a challenger
    must beat the guard's in-cluster mean by `guard_margin` on at least `min_support` scored
    EPISODES (not scenarios) to lead its cluster. The three parameters are recorded on the
    returned policy so `rerank_policy` can re-apply the identical check off the stored evidence.
    """
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

    spec = embedder or EmbedderSpec()
    scenario_ids = list(scenario_tasks)
    embeddings = np.asarray(spec.build().embed([scenario_tasks[sid] for sid in scenario_ids]))
    embeddings = Normalizer(norm="l2").fit_transform(embeddings)

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
    logger.info(
        "k-means: %d clusters over %d scenarios (inertia %.4f)",
        k,
        len(scenario_ids),
        kmeans.inertia_,
    )

    # Per-cluster, per-model reward sums over SCORED episodes only.
    cluster_of = {sid: int(label) for sid, label in zip(scenario_ids, labels, strict=True)}
    sums: dict[int, dict[str, tuple[float, float, int]]] = {c: {} for c in range(k)}
    counts: Counter[int] = Counter(cluster_of.values())
    prefix_counts: dict[int, Counter[str]] = {c: Counter() for c in range(k)}
    member_texts: dict[int, list[str]] = {c: [] for c in range(k)}
    for sid, cluster in cluster_of.items():
        member_texts[cluster].append(scenario_tasks[sid])
        if ":" in sid:
            prefix_counts[cluster][sid.split(":", 1)[0]] += 1
    # Fallback labels for prefix-less ids, which is what a corpus built from real traces has
    # (scenarios are keyed by trace hash): the distinctive c-TF-IDF terms of each cluster's task
    # texts. A majority prefix still wins where one exists. Labels never affect selection.
    text_labels = label_clusters([member_texts[c] for c in range(k)])
    pool_order = {entry.name: index for index, entry in enumerate(matrix.pool)}
    total_cost = 0.0
    total_count = 0
    for outcome in matrix.outcomes:
        cluster = cluster_of.get(outcome.scenario_id)
        if cluster is None or outcome.reward is None:
            continue
        reward_sum, cost_sum, count = sums[cluster].get(outcome.model, (0.0, 0.0, 0))
        sums[cluster][outcome.model] = (
            reward_sum + outcome.reward,
            cost_sum + outcome.cost_usd,
            count + 1,
        )
        total_cost += outcome.cost_usd
        total_count += 1

    clusters: list[ClusterRanking] = []
    for cluster in range(k):
        means = {
            model: reward_sum / count
            for model, (reward_sum, _cost_sum, count) in sums[cluster].items()
        }
        mean_costs = {
            model: cost_sum / count
            for model, (_reward_sum, cost_sum, count) in sums[cluster].items()
        }
        support = {model: count for model, (_reward_sum, _cost_sum, count) in sums[cluster].items()}
        if not means:
            # A cluster with no scored episodes ranks nothing; selection falls through to
            # default_rank scores. Logged, never silent.
            logger.warning("cluster %d has no scored episodes; it ranks no models", cluster)
        ranking = sorted(means, key=lambda m: (-means[m], pool_order[m]))
        if guard_model is not None:
            ranking = _guarded_ranking(
                ranking,
                guard_model=guard_model,
                scores=means,
                costs=mean_costs,
                support=support,
                min_support=min_support,
                guard_margin=guard_margin,
            )
        label = text_labels[cluster]
        if prefix_counts[cluster]:
            label = prefix_counts[cluster].most_common(1)[0][0]
        clusters.append(
            ClusterRanking(
                cluster_id=cluster,
                label=label,
                centroid=[float(v) for v in kmeans.cluster_centers_[cluster]],
                ranking=ranking or [_overall_best(matrix, set(scenario_ids))],
                scores={model: round(mean, 6) for model, mean in means.items()},
                costs={model: round(mean, 8) for model, mean in mean_costs.items()},
                support=support,
                total=counts[cluster],
            )
        )

    chosen_default = default_model or _overall_best(matrix, set(scenario_ids))
    return RoutingPolicy(
        kind="rank",
        default_model=chosen_default,
        pool=matrix.pool,
        embedder=spec,
        clusters=clusters,
        top_k_clusters=top_k_clusters,
        beta=beta,
        default_rank=default_rank,
        cost_scale=(total_cost / total_count) if total_count else 0.0,
        guard_model=guard_model,
        min_support=min_support if guard_model is not None else None,
        guard_margin=guard_margin if guard_model is not None else None,
        fitted_from=fitted_from,
        fit_scenario_ids=list(scenario_ids),
    )


def _overall_best(matrix: OutcomeMatrix, ids: set[str]) -> str:
    sums: dict[str, tuple[float, int]] = {}
    for outcome in matrix.outcomes:
        if outcome.scenario_id not in ids or outcome.reward is None:
            continue
        reward_sum, count = sums.get(outcome.model, (0.0, 0))
        sums[outcome.model] = (reward_sum + outcome.reward, count + 1)
    if not sums:
        raise ValueError("no scored outcomes; cannot pick a default model")
    pool_order = {entry.name: index for index, entry in enumerate(matrix.pool)}
    return min(sums, key=lambda m: (-(sums[m][0] / sums[m][1]), pool_order[m]))


def rerank_policy(policy: RoutingPolicy, *, cost_weight: float) -> RoutingPolicy:
    """Re-rank a fitted policy's clusters under a cost weight, WITHOUT refitting.

    The fit-once, slide-the-knob property (Hybrid LLM, arXiv 2404.14618): every cluster keeps
    its stored reward/cost evidence, and the ranking key becomes
    `mean_reward - cost_weight * mean_cost / cost_scale` (cost in fit-set-average-call units,
    so cost_weight trades one reward point against one average call). `cost_weight=0` returns
    the policy unchanged - the faithful Avengers behavior; the reference has no cost term
    (Avengers-Pro's alpha is the published analogue).

    The fit-time guard travels with the policy and is re-applied here: sliding the knob may not
    undo the only-replace-if-better floor, or every guarded cluster would flip back to its
    unguarded winner at the first non-zero cost weight (the cost term reorders, the guard then
    has to re-veto). Cost pressure can still demote a challenger BELOW the guard; it can never
    promote one the guard rejected.
    """
    if cost_weight == 0.0:
        return policy
    if policy.kind != "rank":
        raise ValueError("only rank policies can be re-ranked")
    if policy.cost_scale <= 0.0:
        raise ValueError("policy has no cost_scale; refit with cost evidence to use the knob")
    pool_order = {entry.name: index for index, entry in enumerate(policy.pool)}
    clusters = []
    for cluster in policy.clusters:
        keyed = {
            model: cluster.scores.get(model, 0.0)
            - cost_weight * cluster.costs.get(model, 0.0) / policy.cost_scale
            for model in cluster.ranking
        }
        ranking = sorted(keyed, key=lambda m: (-keyed[m], pool_order[m]))
        if policy.guard_model is not None:
            ranking = _guarded_ranking(
                ranking,
                guard_model=policy.guard_model,
                scores=cluster.scores,
                costs=cluster.costs,
                support=cluster.support,
                min_support=policy.min_support or 0,
                guard_margin=policy.guard_margin or 0.0,
            )
        clusters.append(cluster.model_copy(update={"ranking": ranking}))
    provenance = f"{policy.fitted_from or 'unknown'} | cost_weight={cost_weight:g}"
    return policy.model_copy(update={"clusters": clusters, "fitted_from": provenance})


class PolicyEval(BaseModel):
    """One policy replayed over a matrix: what the benchmark tables are made of."""

    accuracy: float
    cost_per_scenario: float
    model_mix: dict[str, float]
    scenarios: int
    unscored_scenarios: int  # scenario/model pairs the routed choice had no scored row for


def route_scenarios(
    policy: RoutingPolicy,
    matrix: OutcomeMatrix,
    ids: list[str],
    *,
    embedder: Embedder | None = None,
) -> dict[str, RoutingDecision]:
    """Replay `policy`'s serve-time choice over the scenarios in `ids`, in `ids` order.

    The single owner of offline policy replay: `evaluate_policy` scores through it, and the
    ablation scorecard builds a routed arm's rows from it, so a routed number never depends on
    which caller reimplemented selection.

    Selection runs through the SAME decision code serving uses (`rank_decision` or
    `knn_decision`; queries are batch embedded once for speed, and both normalize internally, so
    the scoring math is shared rather than reimplemented).

    `embedder` overrides the function built from `policy.embedder`, exactly as in `select_model`:
    one client for the whole replay, or cached vectors in research code. It must be the function
    this policy's spec describes, or the fitted centroids (rank) and neighbor bank (knn) are
    meaningless.

    Raises:
        ValueError: when none of `ids` names a scenario the matrix measured, or when `ids`
            repeats a scenario. The return is keyed by scenario, so a repeat cannot be
            represented and would weight that scenario differently for different callers.
    """
    repeated = sorted({sid for sid, count in Counter(ids).items() if count > 1})
    if repeated:
        raise ValueError(
            f"scenario ids repeat in this replay request: {repeated[:5]}; a policy decides once "
            f"per scenario and the result is keyed by scenario id, so pass each id once (to "
            f"weight a scenario more heavily, add episodes to the matrix instead)"
        )
    scenario_tasks: dict[str, str] = {}
    for outcome in matrix.outcomes:
        scenario_tasks.setdefault(outcome.scenario_id, outcome.task)
    wanted = [sid for sid in ids if sid in scenario_tasks]
    if not wanted:
        raise ValueError(
            f"none of the {len(ids)} requested ids are in the matrix; the matrix measured "
            f"{len(scenario_tasks)} scenarios, so check the ids come from this matrix"
        )

    if policy.kind == "static":
        return {
            sid: RoutingDecision(model=policy.default_model, reason="static policy")
            for sid in wanted
        }
    if policy.kind == "knn":
        decide = knn_decision
    elif policy.kind == "linear":
        decide = linear_decision
    else:
        decide = rank_decision
    built = embedder or policy.embedder.build()
    embeddings = np.asarray(built.embed([scenario_tasks[sid] for sid in wanted]))
    return {sid: decide(policy, embeddings[index]) for index, sid in enumerate(wanted)}


def evaluate_policy(
    policy: RoutingPolicy,
    matrix: OutcomeMatrix,
    ids: list[str],
    *,
    embedder: Embedder | None = None,
) -> PolicyEval:
    """Replay `policy` over the scenarios in `ids`, scoring via each routed model's rows.

    Selection is `route_scenarios` (the shared offline replay); this adds the scoring pass over
    the routed rows.

    `embedder` is forwarded to `route_scenarios`: one client for the whole replay, or cached
    vectors in research code. It must be the function `policy.embedder` describes, or the fitted
    centroids (rank) and neighbor bank (knn) are meaningless.
    """
    decisions = route_scenarios(policy, matrix, ids, embedder=embedder)
    wanted = list(decisions)

    by_scenario_model: dict[tuple[str, str], list[float]] = {}
    costs: dict[tuple[str, str], list[float]] = {}
    for outcome in matrix.outcomes:
        key = (outcome.scenario_id, outcome.model)
        if outcome.reward is not None:
            by_scenario_model.setdefault(key, []).append(outcome.reward)
            costs.setdefault(key, []).append(outcome.cost_usd)

    rewards: list[float] = []
    cost_values: list[float] = []
    mix: Counter[str] = Counter()
    unscored = 0
    for sid in wanted:
        model = decisions[sid].model
        mix[model] += 1
        key = (sid, model)
        if key not in by_scenario_model:
            unscored += 1
            continue
        rewards.append(sum(by_scenario_model[key]) / len(by_scenario_model[key]))
        cost_values.append(sum(costs[key]) / len(costs[key]))
    if not rewards:
        raise ValueError("no scored outcomes for any routed choice; nothing to evaluate")
    return PolicyEval(
        accuracy=sum(rewards) / len(rewards),
        cost_per_scenario=sum(cost_values) / len(cost_values),
        model_mix={model: count / len(wanted) for model, count in sorted(mix.items())},
        scenarios=len(wanted),
        unscored_scenarios=unscored,
    )
