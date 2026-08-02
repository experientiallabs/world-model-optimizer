"""Tests for the graded SWE-rebench execution controller."""

from __future__ import annotations

import coding_model_router_graded_swerebench_execute as execute


def test_launch_identity_accepts_legacy_development_manifest() -> None:
    """A resume accepts the pre-phase launch manifest and operational changes."""
    prior = {
        "protocol": "frozen",
        "concurrency": 100,
        "active_e2b_before": 8,
    }
    resumed = {
        "protocol": "frozen",
        "phase": "development",
        "concurrency": 20,
        "active_e2b_before": 1,
    }

    assert execute._launch_identity(prior) == execute._launch_identity(resumed)


def test_launch_identity_rejects_scientific_drift() -> None:
    """A resume still rejects changes to scientific launch fields."""
    prior = {"protocol": "frozen", "corpus_sha256": "one"}
    resumed = {"protocol": "frozen", "corpus_sha256": "two"}

    assert execute._launch_identity(prior) != execute._launch_identity(resumed)
