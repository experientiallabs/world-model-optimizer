"""Tests for the focused native Codeforces kNN audit."""

from __future__ import annotations

from coding_model_router_codeforces_knn_audit import FOCUSED


def test_focused_candidates_cover_quality_and_economic_points() -> None:
    assert len(FOCUSED) == 5
    assert len({candidate.key for candidate in FOCUSED}) == len(FOCUSED)
    assert {candidate.dim for candidate in FOCUSED} == {512, 2_048}
    assert {candidate.guard for candidate in FOCUSED} == {"luna-xhigh", "luna-max"}
    assert {candidate.pick_lam for candidate in FOCUSED} == {0.0, 0.01, 0.02}
