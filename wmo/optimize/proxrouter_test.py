"""Tests for the faithful ProxRouter implementation."""

from __future__ import annotations

import numpy as np

from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.policy import EmbedderSpec
from wmo.optimize.proxrouter import ProxScorer, fit_km_prox, fit_knn_prox
from wmo.providers.base import ProviderKind
from wmo.providers.pool import PoolEntry


def _pool() -> list[PoolEntry]:
    return [
        PoolEntry(
            name="alpha",
            kind=ProviderKind.OPENAI,
            model="alpha",
            input_per_mtok=2.0,
            output_per_mtok=8.0,
        ),
        PoolEntry(
            name="beta",
            kind=ProviderKind.OPENAI,
            model="beta",
            input_per_mtok=0.2,
            output_per_mtok=0.8,
        ),
    ]


def _matrix() -> OutcomeMatrix:
    """Two lexical islands: alpha wins on math texts, beta wins on legal texts."""
    outcomes = []
    for index in range(8):
        math_id, legal_id = f"math:{index}", f"legal:{index}"
        math_task = f"solve the integral of polynomial number {index} dx calculus"
        legal_task = f"draft the indemnity clause for contract {index} liability law"
        outcomes += [
            ScenarioOutcome(
                scenario_id=math_id, task=math_task, model="alpha", reward=1.0, cost_usd=0.02
            ),
            ScenarioOutcome(
                scenario_id=math_id, task=math_task, model="beta", reward=0.0, cost_usd=0.01
            ),
            ScenarioOutcome(
                scenario_id=legal_id, task=legal_task, model="alpha", reward=0.0, cost_usd=0.02
            ),
            ScenarioOutcome(
                scenario_id=legal_id, task=legal_task, model="beta", reward=1.0, cost_usd=0.01
            ),
        ]
    return OutcomeMatrix(pool=_pool(), outcomes=outcomes)


def _embed(policy_embedder: EmbedderSpec, text: str) -> np.ndarray:
    return np.asarray(policy_embedder.build().embed([text])[0])


def test_km_prox_routes_each_island_to_its_winner() -> None:
    matrix = _matrix()
    policy = fit_km_prox(matrix, embedder=EmbedderSpec(dim=256), n_clusters=2, seed=0)
    scorer = ProxScorer(policy)
    math_pick = scorer.decide(_embed(policy.embedder, "integrate the polynomial dx calculus"))
    legal_pick = scorer.decide(_embed(policy.embedder, "indemnity clause liability contract law"))
    assert math_pick.model == "alpha"
    assert legal_pick.model == "beta"
    assert math_pick.cluster_label in {"math", "legal"}


def test_knn_prox_routes_each_island_to_its_winner() -> None:
    matrix = _matrix()
    policy = fit_knn_prox(matrix, embedder=EmbedderSpec(dim=256), knn_k=4, tau_inv=20.0)
    scorer = ProxScorer(policy)
    assert len(policy.references) == 16
    math_pick = scorer.decide(_embed(policy.embedder, "solve the integral dx calculus"))
    legal_pick = scorer.decide(_embed(policy.embedder, "contract indemnity liability law"))
    assert math_pick.model == "alpha"
    assert legal_pick.model == "beta"


def test_flat_tilt_uniform_prior_recovers_global_mean() -> None:
    """tau_inv=0 with equal priors averages every reference: the paper's high-bias corner."""
    matrix = _matrix()
    policy = fit_knn_prox(matrix, embedder=EmbedderSpec(dim=256), knn_k=16, tau_inv=0.0)
    scorer = ProxScorer(policy)
    rewards, _costs, _nearest = scorer.estimates(_embed(policy.embedder, "anything at all"))
    # Both models are 50/50 globally, so the estimates must collapse to ~0.5 each.
    assert abs(rewards["alpha"] - 0.5) < 1e-9
    assert abs(rewards["beta"] - 0.5) < 1e-9


def test_singleton_cluster_prior_stays_finite() -> None:
    matrix = _matrix()
    outcomes = [
        *matrix.outcomes,
        ScenarioOutcome(
            scenario_id="stray:0",
            task="completely unrelated haiku about mountains",
            model="beta",
            reward=1.0,
            cost_usd=0.01,
        ),
    ]
    grown = OutcomeMatrix(pool=_pool(), outcomes=outcomes)
    policy = fit_km_prox(grown, embedder=EmbedderSpec(dim=256), n_clusters=3, seed=0)
    priors = [ref.prior for ref in policy.references]
    assert all(np.isfinite(p) for p in priors)
    # The singleton (spread ~0, n=1) must not dwarf a real 8-member cluster's prior.
    singleton = min(ref.prior for ref in policy.references if ref.total == 1)
    biggest = max(ref.prior for ref in policy.references if ref.total > 1)
    assert singleton < 2 * biggest


def test_guard_reverts_to_baseline_within_margin() -> None:
    matrix = _matrix()
    policy = fit_km_prox(matrix, embedder=EmbedderSpec(dim=256), n_clusters=2, seed=0)
    scorer = ProxScorer(policy)
    query = _embed(policy.embedder, "integrate the polynomial dx calculus")
    free = scorer.decide(query)
    assert free.model == "alpha"
    # An absurd margin forces the guard: nothing beats the baseline by 5 reward points.
    guarded = scorer.decide(query, guard_model="beta", guard_margin=5.0)
    assert guarded.model == "beta"
    assert "guard" in guarded.reason


def test_missing_model_renormalizes_instead_of_nan() -> None:
    matrix = _matrix()
    # Drop beta's rows from the math island: beta's estimate must come from legal refs only.
    thinned = OutcomeMatrix(
        pool=_pool(),
        outcomes=[
            o
            for o in matrix.outcomes
            if not (o.model == "beta" and o.scenario_id.startswith("math"))
        ],
    )
    policy = fit_knn_prox(thinned, embedder=EmbedderSpec(dim=256), knn_k=16, tau_inv=0.0)
    scorer = ProxScorer(policy)
    rewards, _costs, _nearest = scorer.estimates(_embed(policy.embedder, "any text"))
    assert abs(rewards["beta"] - 1.0) < 1e-9  # beta only has legal evidence, all reward 1.0
    assert np.isfinite(rewards["alpha"])


def test_shrinkage_pulls_thin_evidence_to_global_mean() -> None:
    """With heavy shrinkage a singleton reference cannot outvote the global prior."""
    matrix = _matrix()
    policy = fit_knn_prox(matrix, embedder=EmbedderSpec(dim=256), knn_k=4, tau_inv=20.0)
    scorer_raw = ProxScorer(policy)
    heavy = policy.model_copy(update={"shrink_m": 1000.0})
    scorer_shrunk = ProxScorer(heavy)
    query = _embed(policy.embedder, "solve the integral dx calculus")
    raw_rewards, _c, _n = scorer_raw.estimates(query)
    shrunk_rewards, _c2, _n2 = scorer_shrunk.estimates(query)
    # Raw: local evidence says alpha ~1.0 near math. Shrunk: both collapse to ~0.5 global.
    assert raw_rewards["alpha"] > 0.9
    assert abs(shrunk_rewards["alpha"] - 0.5) < 0.01
    assert abs(shrunk_rewards["beta"] - 0.5) < 0.01
    # So a guarded decision reverts to the baseline: the gap is below any real margin.
    decision = scorer_shrunk.decide(query, guard_model="beta", guard_margin=0.03)
    assert decision.model == "beta"


def test_mild_shrinkage_keeps_strong_local_signal() -> None:
    matrix = _matrix()  # 1 episode per (scenario, model): n=1 vs m=1 halves the gap
    policy = fit_knn_prox(matrix, embedder=EmbedderSpec(dim=256), knn_k=4, tau_inv=20.0)
    mild = policy.model_copy(update={"shrink_m": 1.0})
    rewards, _c, _n = ProxScorer(mild).estimates(
        _embed(policy.embedder, "solve the integral dx calculus")
    )
    assert rewards["alpha"] > rewards["beta"] + 0.2  # signal survives


def test_z_guard_keeps_consistent_evidence_and_reverts_thin() -> None:
    matrix = _matrix()
    policy = fit_knn_prox(matrix, embedder=EmbedderSpec(dim=256), knn_k=8, tau_inv=5.0)
    scorer = ProxScorer(policy)
    query = _embed(policy.embedder, "solve the integral dx calculus")
    # 8 co-scored neighbors, alpha beats beta on every math one: the gap clears any small z.
    kept = scorer.decide(query, guard_model="beta", guard_z=0.5, min_pairs=2)
    assert kept.model == "alpha"
    # Demanding more effective pairs than the neighborhood carries must revert.
    reverted = scorer.decide(query, guard_model="beta", guard_z=0.5, min_pairs=1000)
    assert reverted.model == "beta"
    assert "z-guard" in reverted.reason


def test_z_guard_reverts_when_evidence_is_noise() -> None:
    """Alternating rewards: mean gap ~0 with high variance cannot clear the z-test."""
    outcomes = []
    for index in range(12):
        sid = f"t:{index}"
        task = f"shared generic wording task number {index}"
        alpha_wins = index % 2 == 0
        outcomes += [
            ScenarioOutcome(
                scenario_id=sid,
                task=task,
                model="alpha",
                reward=1.0 if alpha_wins else 0.0,
                cost_usd=0.01,
            ),
            ScenarioOutcome(
                scenario_id=sid,
                task=task,
                model="beta",
                reward=0.0 if alpha_wins else 1.0,
                cost_usd=0.01,
            ),
        ]
    matrix = OutcomeMatrix(pool=_pool(), outcomes=outcomes)
    policy = fit_knn_prox(matrix, embedder=EmbedderSpec(dim=256), knn_k=12, tau_inv=0.0)
    scorer = ProxScorer(policy)
    decision = scorer.decide(
        _embed(policy.embedder, "shared generic wording task"),
        guard_model="beta",
        guard_z=0.5,
        min_pairs=2,
    )
    assert decision.model == "beta"


def test_precomputed_embeddings_match_spec_path() -> None:
    matrix = _matrix()
    spec = EmbedderSpec(dim=256)
    fit_ids = sorted({o.scenario_id for o in matrix.outcomes})
    tasks = {o.scenario_id: o.task for o in matrix.outcomes}
    vecs = np.asarray(spec.build().embed([tasks[sid] for sid in fit_ids]))
    via_spec = fit_knn_prox(matrix, fit_ids=fit_ids, embedder=spec, knn_k=4)
    via_pre = fit_knn_prox(matrix, fit_ids=fit_ids, embedder=spec, knn_k=4, precomputed=vecs)
    assert via_spec.references[0].vector == via_pre.references[0].vector


def test_distance_floor_abstains_out_of_support_only() -> None:
    from wmo.optimize.proxrouter import support_floor

    matrix = _matrix()
    policy = fit_knn_prox(matrix, embedder=EmbedderSpec(dim=256), knn_k=8, tau_inv=5.0)
    scorer = ProxScorer(policy)
    floor = support_floor(policy)
    assert 0.0 < floor < 2.0
    in_support = scorer.decide(
        _embed(policy.embedder, "solve the integral of polynomial number 3 dx calculus"),
        guard_model="beta",
        guard_z=0.5,
        min_pairs=2,
        abstain_distance=floor,
    )
    assert in_support.model == "alpha"  # a fit text verbatim: within support, no abstention
    far = scorer.decide(
        _embed(policy.embedder, "haiku regarding wistful mountain snowfall zzqx"),
        guard_model="beta",
        guard_z=0.5,
        min_pairs=2,
        abstain_distance=floor,
    )
    assert far.model == "beta"
    assert "abstained" in far.reason


def test_cost_lambda_prefers_cheaper_at_parity() -> None:
    """With rewards tied, any positive lambda must route to the cheaper model."""
    outcomes = []
    for index in range(6):
        sid = f"t:{index}"
        task = f"generic task number {index} with shared wording"
        outcomes += [
            ScenarioOutcome(scenario_id=sid, task=task, model="alpha", reward=1.0, cost_usd=0.10),
            ScenarioOutcome(scenario_id=sid, task=task, model="beta", reward=1.0, cost_usd=0.01),
        ]
    matrix = OutcomeMatrix(pool=_pool(), outcomes=outcomes)
    policy = fit_km_prox(matrix, embedder=EmbedderSpec(dim=256), n_clusters=1, seed=0)
    scorer = ProxScorer(policy)
    query = _embed(policy.embedder, "generic task with shared wording")
    assert scorer.decide(query, lam=0.1).model == "beta"
