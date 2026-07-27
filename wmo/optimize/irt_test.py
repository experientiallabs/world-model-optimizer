"""Tests for the IRT routing head (numpy 2PL model: fit, forward, selection)."""

from __future__ import annotations

import numpy as np

from wmo.optimize.irt import IrtHead, fit_irt_head, irt_gradient_check
from wmo.optimize.outcomes import OutcomeMatrix
from wmo.optimize.routing_test import _matrix
from wmo.retrieval.embedders import HashingEmbedder


def _embed(matrix: OutcomeMatrix, dim: int = 128) -> tuple[list[str], np.ndarray, dict]:
    tasks: dict[str, str] = {}
    for o in matrix.outcomes:
        tasks.setdefault(o.scenario_id, o.task)
    ids = list(tasks)
    vectors = np.asarray(HashingEmbedder(dim=dim).embed([tasks[s] for s in ids]))
    return ids, vectors, tasks


def test_gradient_check() -> None:
    # Analytic gradients match finite differences on a tiny random head.
    assert irt_gradient_check(seed=0) < 1e-4


def test_fit_recovers_specialists() -> None:
    matrix = _matrix()  # sql-model aces SQL, prose-model aces prose (from routing_test)
    ids, vectors, tasks = _embed(matrix)
    head = fit_irt_head(
        matrix, scenario_ids=ids, embeddings=vectors, seed=0, epochs=300, hidden=32, dim=16
    )
    assert head.models == ["sql-model", "prose-model"]
    # Route every scenario: the specialist must win on its own turf.
    correct = 0
    for sid, vector in zip(ids, vectors, strict=True):
        probs = head.predict(np.asarray(vector))
        choice = head.models[int(np.argmax(probs))]
        expected = "sql-model" if sid.startswith("sql") else "prose-model"
        correct += choice == expected
    assert correct >= 11  # one miss tolerated (embedding locality, as with the rank fitter)


def test_predict_shapes_and_range() -> None:
    matrix = _matrix()
    ids, vectors, _ = _embed(matrix)
    head = fit_irt_head(
        matrix, scenario_ids=ids, embeddings=vectors, seed=0, epochs=50, hidden=16, dim=8
    )
    probs = head.predict(np.asarray(vectors[0]))
    assert probs.shape == (2,)
    assert np.all((probs > 0) & (probs < 1))


def test_head_round_trips_through_json() -> None:
    matrix = _matrix()
    ids, vectors, _ = _embed(matrix)
    head = fit_irt_head(
        matrix, scenario_ids=ids, embeddings=vectors, seed=0, epochs=20, hidden=16, dim=8
    )
    clone = IrtHead.model_validate_json(head.model_dump_json())
    query = np.asarray(vectors[3])
    assert np.allclose(clone.predict(query), head.predict(query))


def test_fit_is_deterministic() -> None:
    matrix = _matrix()
    ids, vectors, _ = _embed(matrix)
    kwargs: dict = {
        "scenario_ids": ids,
        "embeddings": vectors,
        "seed": 7,
        "epochs": 30,
        "hidden": 16,
        "dim": 8,
    }
    a = fit_irt_head(matrix, **kwargs)
    b = fit_irt_head(matrix, **kwargs)
    assert a == b


def test_unscored_rows_are_excluded() -> None:
    matrix = _matrix()
    matrix.outcomes[0].reward = None  # unscored: must not become a 0-label
    ids, vectors, _ = _embed(matrix)
    head = fit_irt_head(
        matrix, scenario_ids=ids, embeddings=vectors, seed=0, epochs=20, hidden=16, dim=8
    )
    assert head.pairs_trained == len(matrix.outcomes) - 1
