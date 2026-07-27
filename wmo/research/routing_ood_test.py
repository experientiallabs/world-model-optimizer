"""Tests for the OOD holdout splits (leakage is the failure mode that matters)."""

from __future__ import annotations

import pytest

from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.policy import EmbedderSpec
from wmo.providers.base import ProviderKind
from wmo.providers.pool import PoolEntry
from wmo.research.routing_ood import split_holdout_clusters, split_holdout_tasks

_TOPICS = {
    "math": "solve the integral of the polynomial dx calculus number",
    "law": "draft the indemnity clause of the liability contract statute",
    "bio": "sequence the ribosome protein genome enzyme pathway cell",
    "code": "refactor the python function loop variable compiler bug",
}


def _matrix(prefixed: bool = True) -> OutcomeMatrix:
    pool = [
        PoolEntry(
            name="alpha",
            kind=ProviderKind.OPENAI,
            model="alpha",
            input_per_mtok=1.0,
            output_per_mtok=1.0,
        )
    ]
    outcomes = []
    for topic, stem in _TOPICS.items():
        for index in range(6):
            sid = f"{topic}:{index}" if prefixed else f"{topic}-{index}"
            outcomes.append(
                ScenarioOutcome(
                    scenario_id=sid,
                    task=f"{stem} variant {index}",
                    model="alpha",
                    reward=1.0,
                    cost_usd=0.01,
                )
            )
    return OutcomeMatrix(pool=pool, outcomes=outcomes)


def test_holdout_tasks_is_disjoint_and_task_pure() -> None:
    matrix = _matrix()
    fit, test = split_holdout_tasks(matrix, test_fraction=0.3, seed=0)
    assert not set(fit) & set(test)
    assert set(fit) | set(test) == set(matrix.scenario_ids())
    fit_tasks = {sid.split(":", 1)[0] for sid in fit}
    test_tasks = {sid.split(":", 1)[0] for sid in test}
    assert not fit_tasks & test_tasks  # no task straddles the split: the whole point


def test_holdout_tasks_rotates_with_seed() -> None:
    matrix = _matrix()
    held = {
        frozenset(sid.split(":", 1)[0] for sid in split_holdout_tasks(matrix, seed=s)[1])
        for s in range(5)
    }
    assert len(held) > 1  # different seeds hold out different tasks


def test_holdout_clusters_is_disjoint_and_covers() -> None:
    matrix = _matrix(prefixed=False)
    fit, test = split_holdout_clusters(
        matrix, embedder=EmbedderSpec(dim=256), test_fraction=0.3, n_clusters=4, seed=0
    )
    assert not set(fit) & set(test)
    assert set(fit) | set(test) == set(matrix.scenario_ids())
    assert fit and test


def test_holdout_clusters_separates_topics() -> None:
    """Lexical islands this clean must land whole-topic on one side or the other."""
    matrix = _matrix(prefixed=False)
    fit, test = split_holdout_clusters(
        matrix, embedder=EmbedderSpec(dim=256), test_fraction=0.3, n_clusters=4, seed=0
    )
    for topic in _TOPICS:
        members = {sid for sid in matrix.scenario_ids() if sid.startswith(topic)}
        assert members <= set(fit) or members <= set(test)


def test_holdout_tasks_requires_prefixes() -> None:
    with pytest.raises(ValueError, match="no task prefix"):
        split_holdout_tasks(_matrix(prefixed=False))
