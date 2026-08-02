"""Record paid authorization and prepare one fresh replacement smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

from pydantic import JsonValue

from wmo.core.files import write_text_atomic

EXPERIMENT_ID = "coding-router-20260728"
AUTHORIZATION_PROTOCOL = "coding-router-spend-authorization-v1"
REPLACEMENT_PROTOCOL = "coding-router-replacement-smoke-authorization-v1"
INVALID_SMOKE_PREFIX = "invalid-smoke-attempt-1:"

logger = logging.getLogger(__name__)

JsonObject = dict[str, JsonValue]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _read_object(path: Path) -> JsonObject:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return {str(key): item for key, item in value.items()}


def _read_ledger(path: Path) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} contains a non-object row")
        rows.append({str(key): item for key, item in value.items()})
    return rows


def _write_json(path: Path, value: JsonValue) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_ledger(path: Path, rows: list[JsonObject]) -> None:
    write_text_atomic(
        path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _historical_event_id(event_id: str) -> str:
    return f"{INVALID_SMOKE_PREFIX}{event_id}"


def _rebase_ledger_artifacts(
    rows: list[JsonObject],
    *,
    original_smoke_root: Path,
    archive: Path,
) -> list[JsonObject]:
    """Point active ledger rows at preserved artifacts without mutating the archive."""
    original = original_smoke_root.resolve()
    rebased: list[JsonObject] = []
    for row in rows:
        updated = dict(row)
        raw_artifact = updated.get("artifact_dir")
        if isinstance(raw_artifact, str):
            try:
                relative = Path(raw_artifact).resolve().relative_to(original)
            except ValueError:
                pass
            else:
                updated.setdefault("original_artifact_dir", raw_artifact)
                updated["artifact_dir"] = str((archive / relative).resolve())
        rebased.append(updated)
    return rebased


def _ledger_transition(
    rows: list[JsonObject],
    *,
    unknown_cost_budget_debit_usd: float,
    original_smoke_root: Path,
    archive: Path,
) -> tuple[list[JsonObject], list[str]]:
    """Namespace the invalid smoke and attach a conservative ceiling debit."""
    already_transitioned = [
        row for row in rows if str(row.get("event_id", "")).startswith(INVALID_SMOKE_PREFIX)
    ]
    active_smoke = [
        row
        for row in rows
        if row.get("phase") == "smoke"
        and not str(row.get("event_id", "")).startswith(INVALID_SMOKE_PREFIX)
    ]
    if already_transitioned and active_smoke:
        raise ValueError("ledger contains both transitioned and active invalid-smoke events")
    source = already_transitioned or active_smoke
    unknown = [
        row for row in source if row.get("model_cost_accounting_status") == "missing_provider_usage"
    ]
    if not source or not unknown:
        raise ValueError("the preserved invalid smoke has no unknown-cost paid events")
    debit_each = unknown_cost_budget_debit_usd / len(unknown)
    transitioned: list[JsonObject] = []
    unknown_ids: list[str] = []
    for row in source:
        updated = dict(row)
        event_id = updated.get("event_id")
        if not isinstance(event_id, str):
            raise ValueError("a smoke ledger row has no event_id")
        if not event_id.startswith(INVALID_SMOKE_PREFIX):
            updated["original_event_id"] = event_id
            updated["event_id"] = _historical_event_id(event_id)
        if updated.get("model_cost_accounting_status") == "missing_provider_usage":
            updated["budget_debit_usd"] = debit_each
            updated["budget_debit_basis"] = (
                "user-authorized conservative debit for an invalid pre-metering smoke"
            )
            unknown_ids.append(str(updated["event_id"]))
        transitioned.append(updated)
    unrelated = [row for row in rows if row not in source]
    return (
        _rebase_ledger_artifacts(
            [*unrelated, *transitioned],
            original_smoke_root=original_smoke_root,
            archive=archive,
        ),
        sorted(unknown_ids),
    )


def authorize(
    root: Path,
    *,
    ceiling_usd: float,
    unknown_cost_budget_debit_usd: float,
) -> None:
    """Freeze the hard ceiling and preserve the invalid smoke before replacement."""
    if ceiling_usd <= 0:
        raise ValueError("ceiling_usd must be positive")
    if unknown_cost_budget_debit_usd <= 0:
        raise ValueError("unknown_cost_budget_debit_usd must be positive")
    if unknown_cost_budget_debit_usd >= ceiling_usd:
        raise ValueError("unknown-cost debit must be below the authorized ceiling")

    freeze_path = root / "freeze-summary.json"
    ledger_path = root / "spend-ledger.jsonl"
    smoke_root = root / "smoke"
    if not freeze_path.is_file() or not ledger_path.is_file():
        raise ValueError("frozen experiment summary and spend ledger are required")
    freeze = _read_object(freeze_path)
    if freeze.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("freeze summary belongs to a different experiment")

    existing = freeze.get("spend_authorization")
    if isinstance(existing, dict):
        if (
            existing.get("protocol") == AUTHORIZATION_PROTOCOL
            and existing.get("ceiling_usd") == ceiling_usd
            and existing.get("unknown_cost_budget_debit_usd") == unknown_cost_budget_debit_usd
        ):
            replacement = _read_object(smoke_root / "replacement-authorization.json")
            raw_archive = replacement.get("archived_smoke_path")
            if not isinstance(raw_archive, str) or not Path(raw_archive).is_dir():
                raise ValueError("the matching authorization has no preserved smoke archive")
            rows = _rebase_ledger_artifacts(
                _read_ledger(ledger_path),
                original_smoke_root=smoke_root,
                archive=Path(raw_archive),
            )
            _write_ledger(ledger_path, rows)
            logger.info("matching spend authorization is already frozen")
            return
        raise ValueError("a different spend authorization is already frozen")

    invalidated_path = smoke_root / "invalidated.json"
    if not invalidated_path.is_file():
        raise ValueError("the invalidated first smoke is required before replacement authorization")
    invalidated = _read_object(invalidated_path)
    if invalidated.get("valid") is not False:
        raise ValueError("the existing smoke is not explicitly invalidated")

    smoke_digest = _tree_sha256(smoke_root)
    archive = root / "invalid-smoke-attempts" / f"attempt-1-{smoke_digest[:12]}"
    if archive.exists():
        if _tree_sha256(archive) != smoke_digest:
            raise ValueError(f"invalid smoke archive digest collision at {archive}")
    else:
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(smoke_root), str(archive))

    ledger, unknown_ids = _ledger_transition(
        _read_ledger(ledger_path),
        unknown_cost_budget_debit_usd=unknown_cost_budget_debit_usd,
        original_smoke_root=smoke_root,
        archive=archive,
    )
    _write_ledger(ledger_path, ledger)

    authorized_at = _utc_now()
    authorization: JsonObject = {
        "protocol": AUTHORIZATION_PROTOCOL,
        "authorized_at": authorized_at,
        "currency": "USD",
        "ceiling_usd": ceiling_usd,
        "replacement_smoke_authorized": True,
        "unknown_cost_events": len(unknown_ids),
        "unknown_cost_budget_debit_usd": unknown_cost_budget_debit_usd,
        "accounting_note": (
            "Historic model cost remains unknown. The debit is a ceiling reservation, "
            "not a claim of exact realized spend."
        ),
    }
    freeze["spend_ceiling_usd"] = ceiling_usd
    freeze["spend_authorization"] = authorization
    _write_json(freeze_path, freeze)

    smoke_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        smoke_root / "replacement-authorization.json",
        {
            "protocol": REPLACEMENT_PROTOCOL,
            "authorized_at": authorized_at,
            "replacement_number": 1,
            "archived_smoke_path": str(archive.resolve()),
            "archived_smoke_sha256": smoke_digest,
            "prior_unknown_cost_event_ids": unknown_ids,
            "unknown_cost_budget_debit_usd": unknown_cost_budget_debit_usd,
            "spend_ceiling_usd": ceiling_usd,
        },
    )
    logger.info(
        "authorized $%.2f ceiling, archived invalid smoke, and debited $%.2f "
        "for %d unknown-cost events",
        ceiling_usd,
        unknown_cost_budget_debit_usd,
        len(unknown_ids),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(".wmo") / "experiments" / EXPERIMENT_ID,
    )
    parser.add_argument("--ceiling-usd", type=float, required=True)
    parser.add_argument("--unknown-cost-budget-debit-usd", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    authorize(
        args.root.resolve(),
        ceiling_usd=args.ceiling_usd,
        unknown_cost_budget_debit_usd=args.unknown_cost_budget_debit_usd,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
