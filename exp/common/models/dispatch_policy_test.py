"""Validator and identity-inertness tests for the rung dispatch policy."""

from __future__ import annotations

import pytest

from exp.common.models.dispatch_policy import GatewayRungDispatchPolicy


def test_rung_dispatch_policy_rejects_incoherent_authoring() -> None:
    """Fairness without a bound, and degenerate bounds or weights, fail closed."""
    with pytest.raises(ValueError, match="concurrency_bound"):
        GatewayRungDispatchPolicy(fair_share=True)
    with pytest.raises(ValueError):
        GatewayRungDispatchPolicy(concurrency_bound=0)
    with pytest.raises(ValueError):
        GatewayRungDispatchPolicy(affinity_weight=0.0)
    with pytest.raises(ValueError):
        GatewayRungDispatchPolicy(affinity_weight=float("inf"))


def test_default_policy_contributes_zero_identity_bytes() -> None:
    """An all-default policy dumps empty under exclude-defaults.

    This is what keeps the catalog's pinned identity digest stable when the
    field exists but nothing is authored; the full catalog-level proof lives
    in ``gateway_catalog_test.py``.
    """
    assert (
        GatewayRungDispatchPolicy().model_dump(mode="json", by_alias=True, exclude_defaults=True)
        == {}
    )
