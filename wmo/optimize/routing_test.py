"""Tests for the routing fitter (Avengers replication) and its policy evaluation."""

from __future__ import annotations

import pytest

from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.policy import ClusterRanking, EmbedderSpec, RoutingPolicy
from wmo.optimize.routing import (
    evaluate_policy,
    fit_rank_policy,
    rerank_policy,
    route_scenarios,
)
from wmo.providers.base import ProviderKind
from wmo.providers.pool import PoolEntry
from wmo.retrieval.embedders import HashingEmbedder

_SQL_TASKS = [
    "SELECT count(*) FROM superheroes WHERE height > 190",
    "SELECT name FROM users ORDER BY created_at DESC LIMIT 10",
    "SELECT avg(price) FROM orders GROUP BY customer_id",
    "SELECT id FROM events WHERE ts > '2026-01-01' AND kind = 'click'",
    "SELECT t.name, count(*) FROM teams t JOIN players p ON p.team_id = t.id GROUP BY 1",
    "SELECT max(score) FROM matches WHERE season = 2025",
]
_PROSE_TASKS = [
    "write a friendly email to the team about the offsite",
    "draft a short thank-you note for the conference organizers",
    "compose a birthday message for a colleague",
    "write a warm welcome paragraph for new employees",
    "draft an apology note for the delayed shipment",
    "write a cheerful newsletter intro about spring",
]


def _entries() -> list[PoolEntry]:
    return [
        PoolEntry(
            name="sql-model",
            kind=ProviderKind.OPENAI,
            model="custom-sql",
            input_per_mtok=1.0,
            output_per_mtok=1.0,
        ),
        PoolEntry(
            name="prose-model",
            kind=ProviderKind.OPENAI,
            model="custom-prose",
            input_per_mtok=1.0,
            output_per_mtok=1.0,
        ),
    ]


def _matrix() -> OutcomeMatrix:
    """sql-model aces SQL and flunks prose; prose-model is the mirror image."""
    outcomes: list[ScenarioOutcome] = []
    for group, tasks in [("sql", _SQL_TASKS), ("prose", _PROSE_TASKS)]:
        for index, task in enumerate(tasks):
            sid = f"{group}:{index}"
            for model in ["sql-model", "prose-model"]:
                wins = (model == "sql-model") == (group == "sql")
                outcomes.append(
                    ScenarioOutcome(
                        scenario_id=sid,
                        task=task,
                        model=model,
                        reward=1.0 if wins else 0.0,
                        success=wins,
                        cost_usd=0.001,
                    )
                )
    return OutcomeMatrix(pool=_entries(), outcomes=outcomes)


def _fit(**kwargs: object) -> RoutingPolicy:
    matrix = _matrix()
    defaults: dict = {
        "embedder": EmbedderSpec(dim=256),
        "n_clusters": 2,
        "seed": 42,
        "top_k_clusters": 1,
    }
    defaults.update(kwargs)
    return fit_rank_policy(matrix, **defaults)


def test_fit_recovers_the_specialists() -> None:
    policy = _fit()
    assert policy.kind == "rank"
    assert len(policy.clusters) == 2
    result = evaluate_policy(policy, _matrix(), _matrix().scenario_ids())
    # Audited (2026-07-24): hashing-trigram geometry puts one prose text in the sql-majority
    # cluster, so routing is 11/12, not 12/12 - an EMBEDDER locality miss, not a fitter bug
    # (the mixed cluster's ranking still has the sql specialist first at 6/7). The algorithm's
    # guarantee is beating every single model (0.5 here), not perfection.
    assert result.accuracy >= 11 / 12 - 1e-9
    assert result.accuracy > 0.5
    mix = result.model_mix
    assert set(mix) == {"sql-model", "prose-model"}
    assert all(share > 0.3 for share in mix.values())


def test_fit_is_deterministic() -> None:
    assert _fit() == _fit()


def test_cluster_count_clamps_to_scenarios() -> None:
    policy = _fit(n_clusters=500)
    assert len(policy.clusters) == 12  # one per scenario at most


def test_default_model_is_overall_best_when_unset() -> None:
    policy = _fit()
    assert policy.default_model in {"sql-model", "prose-model"}  # tied overall; pool order wins
    assert policy.default_model == "sql-model"


def test_rankings_only_contain_scored_models() -> None:
    matrix = _matrix()
    # prose-model never scored anywhere: with a single cluster it must be absent from the
    # ranking entirely (it falls back to default_rank at selection time, like the reference).
    matrix.outcomes = [o for o in matrix.outcomes if o.model != "prose-model"]
    policy = fit_rank_policy(
        matrix, embedder=EmbedderSpec(dim=256), n_clusters=1, seed=42, top_k_clusters=1
    )
    (cluster,) = policy.clusters
    assert cluster.ranking == ["sql-model"]


def test_evaluate_policy_static_baseline() -> None:
    matrix = _matrix()
    static = RoutingPolicy(kind="static", default_model="sql-model", pool=_entries())
    result = evaluate_policy(static, matrix, matrix.scenario_ids())
    assert result.accuracy == pytest.approx(0.5)
    assert result.model_mix == {"sql-model": 1.0}


def test_fitted_from_provenance_recorded() -> None:
    policy = _fit(fitted_from="routerbench_0shot.pkl@seed0")
    assert policy.fitted_from == "routerbench_0shot.pkl@seed0"


def test_fit_stores_cost_evidence() -> None:
    policy = _fit()
    assert policy.cost_scale == pytest.approx(0.001)
    for cluster in policy.clusters:
        assert set(cluster.costs) == set(cluster.scores)
        assert all(cost == pytest.approx(0.001) for cost in cluster.costs.values())


def test_rerank_zero_weight_is_identity() -> None:
    policy = _fit()
    assert rerank_policy(policy, cost_weight=0.0) == policy


def test_rerank_prefers_cheap_when_quality_is_close() -> None:
    # One cluster: expensive model barely better (0.9 vs 0.8) but 100x the cost.
    embedder = HashingEmbedder(dim=32)
    (centroid,) = embedder.embed(["anything"])
    pool = [
        PoolEntry(
            name="pricey",
            kind=ProviderKind.OPENAI,
            model="p",
            input_per_mtok=1.0,
            output_per_mtok=1.0,
        ),
        PoolEntry(
            name="cheap",
            kind=ProviderKind.OPENAI,
            model="c",
            input_per_mtok=1.0,
            output_per_mtok=1.0,
        ),
    ]
    policy = RoutingPolicy(
        kind="rank",
        default_model="cheap",
        pool=pool,
        embedder=EmbedderSpec(dim=32),
        top_k_clusters=1,
        cost_scale=0.001,
        clusters=[
            ClusterRanking(
                cluster_id=0,
                centroid=centroid,
                ranking=["pricey", "cheap"],
                scores={"pricey": 0.9, "cheap": 0.8},
                costs={"pricey": 0.1, "cheap": 0.001},
            )
        ],
    )
    # cost in cost_scale units: pricey 100, cheap 1. At weight 0.01: 0.9-1.0 < 0.8-0.01.
    reranked = rerank_policy(policy, cost_weight=0.01)
    assert reranked.clusters[0].ranking == ["cheap", "pricey"]
    assert "cost_weight=0.01" in (reranked.fitted_from or "")
    # The original is untouched (rerank returns a new policy).
    assert policy.clusters[0].ranking == ["pricey", "cheap"]


def test_rerank_requires_cost_scale() -> None:
    policy = _fit()
    zeroed = policy.model_copy(update={"cost_scale": 0.0})
    with pytest.raises(ValueError, match="cost_scale"):
        rerank_policy(zeroed, cost_weight=0.5)


def test_baseline_guard_reverts_thin_or_losing_clusters() -> None:
    # Cluster evidence: prose-model barely ahead in a thin cluster -> guard reverts to the
    # global best (sql-model); a cluster where prose wins with support keeps prose first.
    matrix = _matrix()
    policy = fit_rank_policy(
        matrix,
        embedder=EmbedderSpec(dim=256),
        n_clusters=2,
        seed=42,
        top_k_clusters=1,
        guard_model="sql-model",
        min_support=100,  # nothing has this support -> EVERY cluster reverts
    )
    for cluster in policy.clusters:
        assert cluster.ranking[0] == "sql-model"


def test_guard_without_in_cluster_evidence_reverts_the_cluster() -> None:
    # The guard model errored out everywhere, so no cluster has evidence about it. Zero-filling
    # its mean would make the guard PASS exactly where it cannot be checked; the cluster must
    # revert to the guard instead.
    matrix = _matrix()
    matrix.outcomes = [
        o.model_copy(update={"reward": None, "success": False, "error": "provider 500"})
        if o.model == "sql-model"
        else o
        for o in matrix.outcomes
    ]
    policy = fit_rank_policy(
        matrix,
        embedder=EmbedderSpec(dim=256),
        n_clusters=2,
        seed=42,
        top_k_clusters=1,
        guard_model="sql-model",
        min_support=1,
        guard_margin=0.0,
    )
    for cluster in policy.clusters:
        assert "sql-model" not in cluster.scores  # nothing measured about the guard here
        assert cluster.ranking[0] == "sql-model"  # and it still leads


def test_guard_survives_the_cost_knob() -> None:
    # The guard is a property of the fitted policy, not of one call: re-ranking under cost
    # pressure re-sorts the cluster and must then re-apply the same floor.
    matrix = _matrix()
    guarded = fit_rank_policy(
        matrix,
        embedder=EmbedderSpec(dim=256),
        n_clusters=2,
        seed=42,
        top_k_clusters=1,
        guard_model="sql-model",
        min_support=100,  # nothing has this support -> every cluster reverts to the guard
    )
    assert guarded.guard_model == "sql-model"
    assert guarded.min_support == 100
    reranked = rerank_policy(guarded, cost_weight=1e-6)
    for cluster in reranked.clusters:
        assert cluster.ranking[0] == "sql-model"

    # Control: the identical fit WITHOUT a guard does flip under the same knob, so the
    # assertion above is testing the guard and not an inert cost term.
    unguarded = fit_rank_policy(
        matrix, embedder=EmbedderSpec(dim=256), n_clusters=2, seed=42, top_k_clusters=1
    )
    flipped = rerank_policy(unguarded, cost_weight=1e-6)
    assert any(cluster.ranking[0] == "prose-model" for cluster in flipped.clusters)


def test_fit_records_per_model_support() -> None:
    policy = _fit()
    for cluster in policy.clusters:
        assert set(cluster.support) == set(cluster.scores)
        assert sum(cluster.support.values()) == 2 * cluster.total  # both models, one episode each


def test_baseline_guard_keeps_real_winners() -> None:
    matrix = _matrix()
    policy = fit_rank_policy(
        matrix,
        embedder=EmbedderSpec(dim=256),
        n_clusters=2,
        seed=42,
        top_k_clusters=1,
        guard_model="sql-model",
        min_support=2,
    )
    # The prose-majority cluster has strong prose evidence (support >= 2, mean 1.0 vs 0.0):
    # prose-model must survive the guard there.
    assert any(c.ranking[0] == "prose-model" for c in policy.clusters)


def test_evaluate_policy_scores_the_decisions_route_scenarios_returns() -> None:
    """The two consumers of the shared replay must agree on who was routed where.

    `evaluate_policy` and `wmo.optimize.scorecard.rows_for_policy` both build on
    `route_scenarios`. A rebase that reinstated an inline selection block inside
    `evaluate_policy` would silently give the two different mixes; this turns that into a red
    test rather than a discrepancy nobody notices.
    """
    matrix = _matrix()
    policy = _fit()
    ids = matrix.scenario_ids()

    decisions = route_scenarios(policy, matrix, ids)
    result = evaluate_policy(policy, matrix, ids)

    expected: dict[str, float] = {}
    for decision in decisions.values():
        expected[decision.model] = expected.get(decision.model, 0.0) + 1 / len(decisions)
    assert result.model_mix == pytest.approx(expected)
    assert result.scenarios == len(decisions)


def test_route_scenarios_replays_linear_policy_through_shared_decision() -> None:
    matrix = _matrix()
    spec = EmbedderSpec(dim=64)
    query = spec.build().embed([_SQL_TASKS[0]])[0]
    policy = RoutingPolicy(
        kind="linear",
        default_model="prose-model",
        pool=_entries(),
        embedder=spec,
        linear_weak_model="prose-model",
        linear_strong_model="sql-model",
        linear_weak_weights=[0.0] * spec.dim,
        linear_strong_weights=query,
        linear_threshold=0.99,
    )
    scenario_id = "sql:0"
    decision = route_scenarios(policy, matrix, [scenario_id])[scenario_id]
    result = evaluate_policy(policy, matrix, [scenario_id])
    assert decision.model == "sql-model"
    assert result.accuracy == 1.0
    assert result.model_mix == {"sql-model": 1.0}


def test_route_scenarios_rejects_repeated_ids() -> None:
    matrix = _matrix()
    policy = RoutingPolicy(kind="static", default_model="sql-model", pool=_entries())
    ids = matrix.scenario_ids()
    with pytest.raises(ValueError, match="scenario ids repeat"):
        route_scenarios(policy, matrix, [*ids, ids[0]])


def _prefixless_matrix() -> OutcomeMatrix:
    """The same corpus keyed by trace hash, which is what a real-trace build produces."""
    outcomes: list[ScenarioOutcome] = []
    for group, tasks in [("sql", _SQL_TASKS), ("prose", _PROSE_TASKS)]:
        for index, task in enumerate(tasks):
            for model in ["sql-model", "prose-model"]:
                wins = (model == "sql-model") == (group == "sql")
                outcomes.append(
                    ScenarioOutcome(
                        # No `prefix:`, so the majority-prefix label finds nothing.
                        scenario_id=f"{group}{index:04x}",
                        task=task,
                        model=model,
                        reward=1.0 if wins else 0.0,
                        success=wins,
                        cost_usd=0.001,
                    )
                )
    return OutcomeMatrix(pool=_entries(), outcomes=outcomes)


def test_prefix_less_clusters_get_c_tf_idf_labels_instead_of_nothing() -> None:
    # Before the fallback every cluster of a real-trace corpus was labeled "", which made the
    # request log's "why did it go there" column useless on exactly the corpora WMO builds from.
    policy = fit_rank_policy(_prefixless_matrix(), n_clusters=2, embedder=EmbedderSpec(dim=64))
    labels = [cluster.label for cluster in policy.clusters]
    assert all(labels), labels
    # Labels never affect selection, only what the log calls the cluster.
    assert len(set(labels)) == len(labels)


def test_a_majority_prefix_still_wins_over_the_text_fallback() -> None:
    policy = fit_rank_policy(_matrix(), n_clusters=2, embedder=EmbedderSpec(dim=64))
    assert {cluster.label for cluster in policy.clusters} == {"sql", "prose"}
