"""Tests for the learning-to-defer plug-in rule."""

from __future__ import annotations

import numpy as np

from wmo.optimize.l2d import fit_l2d
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.policy import EmbedderSpec
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


def _embed(texts: list[str]) -> np.ndarray:
    return np.asarray(EmbedderSpec(dim=256).build().embed(texts))


def _fit(matrix: OutcomeMatrix, baseline: str) -> tuple:
    fit_ids = sorted({o.scenario_id for o in matrix.outcomes})
    tasks = {o.scenario_id: o.task for o in matrix.outcomes}
    vecs = _embed([tasks[sid] for sid in fit_ids])
    return fit_l2d(matrix, fit_ids=fit_ids, embeddings=vecs, baseline=baseline), tasks


def test_l2d_routes_islands_and_defers_on_ties() -> None:
    outcomes = []
    for index in range(10):
        math_id, legal_id = f"math{index}", f"legal{index}"
        math = f"solve the integral of polynomial number {index} dx calculus"
        legal = f"draft the indemnity clause for contract {index} liability law"
        outcomes += [
            ScenarioOutcome(
                scenario_id=math_id, task=math, model="alpha", reward=1.0, cost_usd=0.02
            ),
            ScenarioOutcome(
                scenario_id=math_id, task=math, model="beta", reward=0.0, cost_usd=0.01
            ),
            ScenarioOutcome(
                scenario_id=legal_id, task=legal, model="alpha", reward=0.0, cost_usd=0.02
            ),
            ScenarioOutcome(
                scenario_id=legal_id, task=legal, model="beta", reward=1.0, cost_usd=0.01
            ),
        ]
    matrix = OutcomeMatrix(pool=_pool(), outcomes=outcomes)
    rule, _tasks = _fit(matrix, baseline="alpha")
    math_query = _embed(["solve the integral of polynomial number 3 dx calculus"])[0]
    legal_query = _embed(["draft the indemnity clause for contract 3 liability law"])[0]
    assert rule.decide(math_query) == "alpha"
    assert rule.decide(legal_query) == "beta"


def test_l2d_uninformative_features_defer_to_global_best() -> None:
    """Identical texts: heads collapse to global means, argmax = the better global model."""
    outcomes = []
    for index in range(10):
        sid = f"t{index}"
        task = "the same words every single time"
        outcomes += [
            ScenarioOutcome(
                scenario_id=sid,
                task=task,
                model="alpha",
                reward=0.8 if index < 8 else 0.0,
                cost_usd=0.02,
            ),
            ScenarioOutcome(
                scenario_id=sid,
                task=task,
                model="beta",
                reward=0.8 if index < 4 else 0.0,
                cost_usd=0.01,
            ),
        ]
    matrix = OutcomeMatrix(pool=_pool(), outcomes=outcomes)
    rule, _tasks = _fit(matrix, baseline="alpha")
    query = _embed(["the same words every single time"])[0]
    assert rule.decide(query) == "alpha"  # global mean 0.64 vs 0.32: defer to baseline


def test_l2d_cost_lambda_flips_exact_ties_to_cheaper() -> None:
    outcomes = []
    for index in range(8):
        sid = f"t{index}"
        task = f"generic shared wording number {index}"
        outcomes += [
            ScenarioOutcome(scenario_id=sid, task=task, model="alpha", reward=1.0, cost_usd=0.10),
            ScenarioOutcome(scenario_id=sid, task=task, model="beta", reward=1.0, cost_usd=0.01),
        ]
    matrix = OutcomeMatrix(pool=_pool(), outcomes=outcomes)
    rule, _tasks = _fit(matrix, baseline="alpha")
    query = _embed(["generic shared wording number 2"])[0]
    assert rule.decide(query, lam=0.1) == "beta"
