"""Tests for native Codeforces kNN development helpers."""

from __future__ import annotations

import numpy as np
from coding_model_router_codeforces_knn import (
    ARMS,
    Candidate,
    FoldValue,
    _aggregate,
    candidate_grid,
)


def test_candidate_grid_is_dense_and_unique() -> None:
    candidates = candidate_grid()
    assert len(candidates) == 1_280
    assert len({candidate.key for candidate in candidates}) == len(candidates)
    assert {candidate.guard for candidate in candidates} == set(ARMS)
    assert {candidate.z for candidate in candidates} == {0.0, 0.5, 1.0, 2.0}
    assert {candidate.pick_lam for candidate in candidates} == {0.0, 0.01, 0.02, 0.03}


def test_candidate_identity_carries_economic_knobs() -> None:
    candidate = Candidate(512, "luna-low", 16, 0.95, 2.0, 0.03)
    assert candidate.key == "hash512-guard-luna-low-k16-th0.95-z2-lam0.03"


def test_aggregate_preserves_matched_blind_advantage() -> None:
    fold = FoldValue(
        indices=np.asarray([0, 1]),
        choices=np.asarray([0, 1]),
        reward=np.asarray([1.0, 0.8]),
        cost=np.asarray([0.1, 0.2]),
        blind_reward=np.asarray([0.7, 0.7]),
        blind_cost=np.asarray([0.15, 0.15]),
    )
    value = _aggregate([fold])
    assert value["reward"] == 0.9
    assert value["cost_usd"] == 0.30000000000000004
    assert value["advantage"] == 0.20000000000000007
    assert value["counts"] == {
        "luna-low": 1,
        "luna-medium": 1,
        "luna-high": 0,
        "luna-xhigh": 0,
        "luna-max": 0,
    }
