"""Tests for phase-safe SWE-rebench outcome collection."""

from __future__ import annotations

import json
from pathlib import Path

import coding_model_router_swerebench_collect as collect


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_development_launch_context_preserves_prior_spend(tmp_path: Path) -> None:
    root = tmp_path / "matrix"
    root.mkdir()
    launch = root / "launch.json"
    _write_json(
        launch,
        {
            "protocol": collect.DEVELOPMENT_EXECUTION_PROTOCOL,
            "corpus_sha256": collect.CORPUS_SHA256,
            "model": "gpt-5.6-luna",
            "prior_spend_usd": 405.7678502,
            "deep_swe_outcomes_accessed": False,
            "model_persisted": False,
        },
    )

    spend, context = collect._launch_context(root, collect.DEVELOPMENT_PHASE)

    assert spend == 405.7678502
    assert context == {"launch_sha256": collect._sha256(launch)}


def test_confirmation_launch_context_requires_frozen_authorization(
    tmp_path: Path,
) -> None:
    root = tmp_path / "matrix"
    root.mkdir()
    authorization = {
        "confirmation_corpus_sha256": collect.CONFIRMATION_CORPUS_SHA256,
        "selection_lock_sha256": "a" * 64,
    }
    launch = root / "launch.json"
    _write_json(
        launch,
        {
            "protocol": collect.CONFIRMATION_EXECUTION_PROTOCOL,
            "corpus_sha256": collect.CONFIRMATION_CORPUS_SHA256,
            "model": "gpt-5.6-luna",
            "prior_spend_usd": 640.0,
            "deep_swe_outcomes_accessed": False,
            "model_persisted": False,
            "confirmation_outcomes_accessed_before_launch": False,
            "authorization": authorization,
        },
    )

    spend, context = collect._launch_context(root, collect.CONFIRMATION_PHASE)

    assert spend == 640.0
    assert context["launch_sha256"] == collect._sha256(launch)
    assert context["authorization"] == authorization


def test_confirmation_launch_context_rejects_preaccessed_outcomes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "matrix"
    root.mkdir()
    _write_json(
        root / "launch.json",
        {
            "protocol": collect.CONFIRMATION_EXECUTION_PROTOCOL,
            "corpus_sha256": collect.CONFIRMATION_CORPUS_SHA256,
            "model": "gpt-5.6-luna",
            "prior_spend_usd": 640.0,
            "deep_swe_outcomes_accessed": False,
            "model_persisted": False,
            "confirmation_outcomes_accessed_before_launch": True,
            "authorization": {
                "confirmation_corpus_sha256": collect.CONFIRMATION_CORPUS_SHA256,
            },
        },
    )

    try:
        collect._launch_context(root, collect.CONFIRMATION_PHASE)
    except ValueError as error:
        assert "frozen authorization" in str(error)
    else:
        raise AssertionError("pre-accessed confirmation outcomes were accepted")


def test_pooled_confirmation_uses_its_frozen_collection_identity() -> None:
    phase = collect._collection_phase("pooled-confirmation")
    assert phase is collect.POOLED_CONFIRMATION_PHASE
    assert phase.execution_protocol == collect.POOLED_CONFIRMATION_EXECUTION_PROTOCOL
    assert phase.corpus_sha256 == collect.POOLED_CONFIRMATION_CORPUS_SHA256
    assert phase.reuse_smoke is False
