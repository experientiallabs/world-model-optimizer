"""Tests for coding-router paid authorization and smoke replacement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from coding_model_router_authorize import authorize


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(root: Path) -> None:
    _write_json(
        root / "freeze-summary.json",
        {
            "experiment_id": "coding-router-20260728",
            "spend_ceiling_usd": None,
        },
    )
    _write_json(root / "smoke" / "invalidated.json", {"valid": False})
    _write_json(root / "smoke" / "outcomes.json", {"outcomes": []})
    rows = [
        {
            "event_id": "smoke:paid",
            "phase": "smoke",
            "artifact_dir": str(root / "smoke" / "infra-attempts" / "paid"),
            "model_cost_usd": None,
            "model_cost_accounting_status": "missing_provider_usage",
        },
        {
            "event_id": "smoke:infra",
            "phase": "smoke",
            "model_cost_usd": 0.0,
            "model_cost_accounting_status": "exact_from_provider_usage",
        },
    ]
    (root / "spend-ledger.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_authorize_preserves_invalid_smoke_and_debits_unknown_cost(tmp_path: Path) -> None:
    _fixture(tmp_path)

    authorize(
        tmp_path,
        ceiling_usd=20_000.0,
        unknown_cost_budget_debit_usd=300.0,
    )

    freeze = json.loads((tmp_path / "freeze-summary.json").read_text())
    assert freeze["spend_ceiling_usd"] == 20_000.0
    assert freeze["spend_authorization"]["unknown_cost_budget_debit_usd"] == 300.0
    replacement = json.loads((tmp_path / "smoke" / "replacement-authorization.json").read_text())
    archive = Path(replacement["archived_smoke_path"])
    assert archive.is_dir()
    assert (archive / "invalidated.json").is_file()
    rows = [json.loads(line) for line in (tmp_path / "spend-ledger.jsonl").read_text().splitlines()]
    paid = next(row for row in rows if row.get("original_event_id") == "smoke:paid")
    assert paid["model_cost_usd"] is None
    assert paid["budget_debit_usd"] == 300.0
    assert str(paid["event_id"]).startswith("invalid-smoke-attempt-1:")
    assert Path(paid["artifact_dir"]).is_relative_to(archive)
    assert paid["original_artifact_dir"] == str(tmp_path / "smoke" / "infra-attempts" / "paid")


def test_authorize_is_idempotent_for_identical_parameters(tmp_path: Path) -> None:
    _fixture(tmp_path)
    authorize(
        tmp_path,
        ceiling_usd=20_000.0,
        unknown_cost_budget_debit_usd=300.0,
    )

    authorize(
        tmp_path,
        ceiling_usd=20_000.0,
        unknown_cost_budget_debit_usd=300.0,
    )

    assert (tmp_path / "smoke" / "replacement-authorization.json").is_file()


def test_authorize_rejects_a_different_existing_ceiling(tmp_path: Path) -> None:
    _fixture(tmp_path)
    authorize(
        tmp_path,
        ceiling_usd=20_000.0,
        unknown_cost_budget_debit_usd=300.0,
    )

    with pytest.raises(ValueError, match="different spend authorization"):
        authorize(
            tmp_path,
            ceiling_usd=10_000.0,
            unknown_cost_budget_debit_usd=300.0,
        )
