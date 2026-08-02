"""Tests for graded SWE-rebench matrix collection."""

import hashlib

import pytest

import coding_model_router_graded_swerebench_collect as collector


def test_patch_provenance_accepts_validated_legacy_captured_patch() -> None:
    report = {
        "official_verifier_reached": True,
        "patch_bytes": 17,
        "patch_sha256": hashlib.sha256(b"captured patch").hexdigest(),
    }

    assert collector._patch_provenance(report, "cell") == (
        "official captured patch",
        "inferred from validated legacy report fields",
    )


def test_patch_provenance_rejects_inconsistent_declared_value() -> None:
    report = {
        "official_verifier_reached": True,
        "patch_bytes": 17,
        "patch_sha256": hashlib.sha256(b"captured patch").hexdigest(),
        "patch_provenance": "official trace reported no source changes",
    }

    with pytest.raises(ValueError, match="inconsistent patch evidence"):
        collector._patch_provenance(report, "cell")


def test_patch_provenance_accepts_declared_empty_captured_patch() -> None:
    report = {
        "official_verifier_reached": True,
        "patch_bytes": 0,
        "patch_sha256": hashlib.sha256(b"").hexdigest(),
        "patch_provenance": "official captured patch",
    }

    assert collector._patch_provenance(report, "cell") == (
        "official captured patch",
        "declared by arm validator report",
    )
