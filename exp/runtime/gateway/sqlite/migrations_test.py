"""Tests for guarded SQLite initialization and forward migration."""

from __future__ import annotations

import os
import sqlite3
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from exp.common.models import ConnectionConfig
from exp.runtime.gateway.sqlite import migrations
from exp.runtime.gateway.sqlite.migrations import (
    _MIGRATION_1,
    _MIGRATION_2,
    _MIGRATION_3,
    _MIGRATION_4,
    _MIGRATION_5,
    _MIGRATION_6,
    _MIGRATION_7,
    SCHEMA_VERSION,
    GatewaySchemaError,
    connect_database,
    initialize_database,
    persistent_connection,
)
from exp.runtime.gateway.sqlite.provider_authority import active_provider_connections


def test_persistent_connection_reuses_one_idle_connection_per_thread(tmp_path: Path) -> None:
    """Sequential checkouts on one thread reuse the same cached connection."""
    database = tmp_path / "reuse.db"
    initialize_database(database)
    with persistent_connection(database) as first:
        first_id = id(first)
    with persistent_connection(database) as second:
        assert id(second) == first_id


def test_persistent_connection_overlapping_checkouts_get_distinct_connections(
    tmp_path: Path,
) -> None:
    """Nested checkouts must not share one connection mid-transaction."""
    database = tmp_path / "nested.db"
    initialize_database(database)
    with persistent_connection(database) as outer:
        outer.execute("BEGIN IMMEDIATE")
        try:
            with persistent_connection(database) as inner:
                assert inner is not outer
        finally:
            outer.execute("ROLLBACK")


def test_persistent_connection_discards_a_connection_left_in_transaction(
    tmp_path: Path,
) -> None:
    """A connection exiting mid-transaction is closed instead of cached."""
    database = tmp_path / "dirty.db"
    initialize_database(database)
    with persistent_connection(database) as connection:
        connection.execute("BEGIN IMMEDIATE")
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")
    with persistent_connection(database) as replacement:
        assert replacement.in_transaction is False
        assert replacement.execute("SELECT 1").fetchone()[0] == 1


def test_persistent_connection_closes_and_discards_on_error(tmp_path: Path) -> None:
    """An exception inside the checkout closes the connection and clears the cache."""
    database = tmp_path / "error.db"
    initialize_database(database)
    with pytest.raises(RuntimeError, match="boom"):
        with persistent_connection(database) as connection:
            raise RuntimeError("boom")
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")
    with persistent_connection(database) as replacement:
        assert replacement.execute("SELECT 1").fetchone()[0] == 1


def test_close_idle_connections_releases_only_the_calling_thread_cache(
    tmp_path: Path,
) -> None:
    """Closing idle connections closes the caller's cache and spares other threads."""
    database = tmp_path / "close.db"
    initialize_database(database)
    # Drain connections cached by earlier tests so the counts below are exact.
    migrations.close_idle_connections()
    with persistent_connection(database) as cached:
        pass

    def _cache_and_close_elsewhere() -> tuple[int, bool]:
        """Cache one connection on a worker thread and close that thread's cache."""
        with persistent_connection(database) as connection:
            pass
        closed = migrations.close_idle_connections()
        try:
            connection.execute("SELECT 1")
        except sqlite3.ProgrammingError:
            return closed, True
        return closed, False

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker_closed, worker_connection_closed = pool.submit(_cache_and_close_elsewhere).result()
    assert worker_closed == 1
    assert worker_connection_closed is True
    assert cached.execute("SELECT 1").fetchone()[0] == 1
    assert migrations.close_idle_connections() == 1
    with pytest.raises(sqlite3.ProgrammingError):
        cached.execute("SELECT 1")
    assert migrations.close_idle_connections() == 0
    with persistent_connection(database) as replacement:
        assert replacement is not cached
        assert replacement.execute("SELECT 1").fetchone()[0] == 1


def test_initial_database_is_private_wal_with_foreign_keys(tmp_path: Path) -> None:
    """Fresh state enables WAL, foreign keys, bounded busy waits, and mode 0600."""
    path = tmp_path / "gateway.db"

    assert initialize_database(path) is None
    connection = connect_database(path, busy_timeout_ms=321)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        columns = {
            str(row[1]): row for row in connection.execute("PRAGMA table_info(alias_revisions)")
        }
        assert columns["refusal_failover"][4] == "0"
        attempt_columns = {
            str(row[1]): row for row in connection.execute("PRAGMA table_info(gateway_attempts)")
        }
        assert attempt_columns["attempt_ordinal"][3] == 1
        assert attempt_columns["billing_source"][3] == 1
        assert "customer_managed" in str(attempt_columns["billing_source"][4])
        assert "budget_period_start" in attempt_columns
        assert "budget_reserved_micro_usd" in attempt_columns
        assert "budget_settled_micro_usd" in attempt_columns
        # v16 retains the provider's sanitized rejection sentence.
        assert "failure_message" in attempt_columns
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'gateway_monthly_budgets'"
            ).fetchone()
            is not None
        )
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 321
    finally:
        connection.close()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_forward_migration_creates_consistent_private_backup(tmp_path: Path) -> None:
    """Existing old schemas are backed up before the next forward-only migration."""
    path = tmp_path / "gateway.db"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    connection = connect_database(path)
    try:
        connection.execute("BEGIN EXCLUSIVE")
        for statement in _MIGRATION_1:
            connection.execute(statement)
        connection.execute("PRAGMA user_version = 1")
        connection.execute("COMMIT")
    finally:
        connection.close()

    backup = initialize_database(path)

    assert backup is not None
    assert backup.exists()
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    backup_connection = sqlite3.connect(backup)
    try:
        assert backup_connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert (
            backup_connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'gateway_requests'"
            ).fetchone()
            is None
        )
    finally:
        backup_connection.close()


def test_attempt_billing_migration_is_explicit_and_preserves_v2_backup(tmp_path: Path) -> None:
    """Legacy attempts migrate to customer-managed while the v2 backup stays unchanged."""
    path = tmp_path / "gateway.db"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    connection = connect_database(path)
    try:
        connection.execute("BEGIN EXCLUSIVE")
        for statement in (*_MIGRATION_1, *_MIGRATION_2):
            connection.execute(statement)
        connection.execute(
            "INSERT INTO organizations VALUES (?, ?, ?, 1, ?, ?)",
            ("org-one", "one", "One", "2026-08-18T00:00:00Z", "2026-08-18T00:00:00Z"),
        )
        connection.execute(
            "INSERT INTO identities VALUES (?, ?, ?, NULL, 1, ?, ?)",
            (
                "identity-one",
                "org-one",
                "Identity",
                "2026-08-18T00:00:00Z",
                "2026-08-18T00:00:00Z",
            ),
        )
        connection.execute(
            """
            INSERT INTO virtual_keys (
                key_id, organization_id, identity_id, prefix, fingerprint_version,
                fingerprint_sha256, created_at
            ) VALUES (?, ?, ?, ?, 1, ?, ?)
            """,
            (
                "key-one",
                "org-one",
                "identity-one",
                "exp_test",
                "a" * 64,
                "2026-08-18T00:00:00Z",
            ),
        )
        connection.execute(
            "INSERT INTO catalog_snapshot_refs VALUES (?, ?, ?, ?)",
            ("snapshot-one", "org-one", "b" * 64, "2026-08-18T00:00:00Z"),
        )
        connection.execute(
            """
            INSERT INTO gateway_aliases (
                alias_id, organization_id, alias_name, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                "alias-one",
                "org-one",
                "coding",
                "2026-08-18T00:00:00Z",
                "2026-08-18T00:00:00Z",
            ),
        )
        connection.execute(
            """
            INSERT INTO alias_revisions (
                revision_id, organization_id, alias_id, revision_number, target_kind,
                pool_id, catalog_sha256, snapshot_ref, created_at
            ) VALUES (?, ?, ?, 1, 'direct', ?, ?, ?, ?)
            """,
            (
                "revision-one",
                "org-one",
                "alias-one",
                "pool-one",
                "b" * 64,
                "snapshot-one",
                "2026-08-18T00:00:00Z",
            ),
        )
        connection.execute(
            "UPDATE gateway_aliases SET active_revision_id = ? WHERE alias_id = ?",
            ("revision-one", "alias-one"),
        )
        connection.execute(
            """
            INSERT INTO gateway_requests (
                request_id, organization_id, identity_id, key_id, alias_id,
                alias_revision_id, api_surface, canonical_request_sha256,
                accepted_at, deadline_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'responses', ?, ?, ?)
            """,
            (
                "request-one",
                "org-one",
                "identity-one",
                "key-one",
                "alias-one",
                "revision-one",
                "c" * 64,
                "2026-08-18T00:00:00Z",
                "2026-08-18T00:01:00Z",
            ),
        )
        connection.execute(
            """
            INSERT INTO gateway_attempts (
                attempt_id, request_id, organization_id, route_depth, deployment_id,
                provider, exact_model_id, pool_id, catalog_sha256, state, started_at
            ) VALUES (?, ?, ?, 0, ?, ?, ?, ?, ?, 'completed', ?)
            """,
            (
                "attempt-one",
                "request-one",
                "org-one",
                "deployment-one",
                "openai",
                "exact-one",
                "pool-one",
                "b" * 64,
                "2026-08-18T00:00:01Z",
            ),
        )
        connection.execute("PRAGMA user_version = 2")
        connection.execute("COMMIT")
    finally:
        connection.close()

    backup = initialize_database(path)

    assert backup is not None
    backup_connection = sqlite3.connect(backup)
    current = connect_database(path)
    try:
        backup_columns = {
            str(row[1]) for row in backup_connection.execute("PRAGMA table_info(gateway_attempts)")
        }
        assert "billing_source" not in backup_columns
        assert backup_connection.execute("PRAGMA user_version").fetchone()[0] == 2
        row = current.execute(
            """
            SELECT billing_source, attempt_ordinal, route_depth
            FROM gateway_attempts WHERE attempt_id = 'attempt-one'
            """
        ).fetchone()
        assert row is not None
        assert row["billing_source"] == "customer_managed"
        assert row["attempt_ordinal"] == 0
        assert row["route_depth"] == 0
    finally:
        current.close()
        backup_connection.close()


def test_provider_authority_migration_preserves_v3_backup(tmp_path: Path) -> None:
    """Schema v4 adds serving connection revisions without rewriting prior authority."""
    path = tmp_path / "gateway.db"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    connection = connect_database(path)
    try:
        connection.execute("BEGIN EXCLUSIVE")
        for statement in (*_MIGRATION_1, *_MIGRATION_2):
            connection.execute(statement)
        connection.execute(
            """
            ALTER TABLE gateway_attempts
            ADD COLUMN billing_source TEXT NOT NULL DEFAULT 'customer_managed'
            CHECK (billing_source IN ('customer_managed', 'host_managed'))
            """
        )
        connection.execute("PRAGMA user_version = 3")
        connection.execute("COMMIT")
    finally:
        connection.close()

    backup = initialize_database(path)

    assert backup is not None
    backup_connection = sqlite3.connect(backup)
    current = connect_database(path)
    try:
        assert backup_connection.execute("PRAGMA user_version").fetchone()[0] == 3
        assert (
            backup_connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'provider_connections'"
            ).fetchone()
            is None
        )
        tables = {
            str(row[0])
            for row in current.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'table' AND name LIKE '%provider_connection%'
                """
            )
        }
        assert tables == {
            "alias_revision_provider_connections",
            "provider_connection_revisions",
            "provider_connections",
        }
    finally:
        current.close()
        backup_connection.close()


def test_v6_migration_preserves_billing_and_adds_physical_ordinal(tmp_path: Path) -> None:
    """A v5 host-managed attempt keeps route identity while gaining an ordinal."""
    path = tmp_path / "gateway.db"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    connection = sqlite3.connect(path)
    try:
        connection.execute("BEGIN EXCLUSIVE")
        for migration in (
            _MIGRATION_1,
            _MIGRATION_2,
            _MIGRATION_3,
            _MIGRATION_4,
            _MIGRATION_5,
        ):
            for statement in migration:
                connection.execute(statement)
        connection.execute(
            """
            INSERT INTO gateway_requests (
                request_id, organization_id, identity_id, key_id, alias_id,
                alias_revision_id, api_surface, canonical_request_sha256,
                accepted_at, deadline_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "request-one",
                "org-one",
                "identity-one",
                "key-one",
                "alias-one",
                "revision-one",
                "chat_completions",
                "a" * 64,
                "2026-08-18T00:00:00+00:00",
                "2026-08-18T00:01:00+00:00",
            ),
        )
        connection.execute(
            """
            INSERT INTO gateway_attempts (
                attempt_id, request_id, organization_id, route_depth, deployment_id,
                provider, exact_model_id, pool_id, catalog_sha256, billing_source,
                state, started_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "attempt-one",
                "request-one",
                "org-one",
                3,
                "deployment-one",
                "openai",
                "exact-one",
                "pool-one",
                "b" * 64,
                "host_managed",
                "failed",
                "2026-08-18T00:00:01+00:00",
            ),
        )
        connection.execute("PRAGMA user_version = 5")
        connection.execute("COMMIT")
    finally:
        connection.close()

    backup = initialize_database(path)

    assert backup is not None
    migrated = sqlite3.connect(path)
    try:
        row = migrated.execute(
            """
            SELECT attempt_id, attempt_ordinal, route_depth, billing_source
            FROM gateway_attempts
            """
        ).fetchone()
        assert row == ("attempt-one", 0, 3, "host_managed")
    finally:
        migrated.close()
    prior = sqlite3.connect(backup)
    try:
        assert prior.execute("PRAGMA user_version").fetchone()[0] == 5
        columns = {str(row[1]) for row in prior.execute("PRAGMA table_info(gateway_attempts)")}
        assert "billing_source" in columns
        assert "attempt_ordinal" not in columns
        refusal_columns = {
            str(row[1]) for row in prior.execute("PRAGMA table_info(alias_revisions)")
        }
        assert "refusal_failover" in refusal_columns
        assert (
            prior.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'provider_connections'"
            ).fetchone()
            is not None
        )
    finally:
        prior.close()


def test_v7_migration_assigns_immutable_period_and_preserves_prior_cost(tmp_path: Path) -> None:
    """A v6 attempt enters its original UTC month without a destructive reset."""
    path = tmp_path / "gateway.db"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    connection = sqlite3.connect(path)
    try:
        connection.execute("BEGIN EXCLUSIVE")
        for migration in (
            _MIGRATION_1,
            _MIGRATION_2,
            _MIGRATION_3,
            _MIGRATION_4,
            _MIGRATION_5,
            _MIGRATION_6,
        ):
            for statement in migration:
                connection.execute(statement)
        connection.execute(
            """
            INSERT INTO gateway_requests (
                request_id, organization_id, identity_id, key_id, alias_id,
                alias_revision_id, api_surface, canonical_request_sha256,
                accepted_at, deadline_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "request-one",
                "org-one",
                "identity-one",
                "key-one",
                "alias-one",
                "revision-one",
                "chat_completions",
                "a" * 64,
                "2026-08-31T23:59:00+00:00",
                "2026-09-01T00:01:00+00:00",
            ),
        )
        connection.execute(
            """
            INSERT INTO gateway_attempts (
                attempt_id, request_id, organization_id, attempt_ordinal, route_depth,
                deployment_id, provider, exact_model_id, pool_id, catalog_sha256,
                billing_source, state, started_at, estimated_cost_micro_usd
            ) VALUES (?, ?, ?, 0, 0, ?, ?, ?, ?, ?, ?, 'completed', ?, ?)
            """,
            (
                "attempt-one",
                "request-one",
                "org-one",
                "deployment-one",
                "openai",
                "exact-one",
                "pool-one",
                "b" * 64,
                "host_managed",
                "2026-08-31T23:59:30+00:00",
                17,
            ),
        )
        connection.execute("PRAGMA user_version = 6")
        connection.execute("COMMIT")
    finally:
        connection.close()

    backup = initialize_database(path)

    assert backup is not None
    current = sqlite3.connect(path)
    try:
        row = current.execute(
            """
            SELECT budget_period_start, budget_reserved_micro_usd,
                   budget_settled_micro_usd
            FROM gateway_attempts WHERE attempt_id = 'attempt-one'
            """
        ).fetchone()
        assert row == ("2026-08-01T00:00:00+00:00", None, 17)
        assert current.execute("PRAGMA user_version").fetchone() == (SCHEMA_VERSION,)
    finally:
        current.close()
    prior = sqlite3.connect(backup)
    try:
        columns = {str(row[1]) for row in prior.execute("PRAGMA table_info(gateway_attempts)")}
        assert "budget_period_start" not in columns
        assert prior.execute("PRAGMA user_version").fetchone() == (6,)
    finally:
        prior.close()


def test_v8_migration_adds_default_off_strict_unknown_cost(tmp_path: Path) -> None:
    """A v7 budget migrates with the strict fail-closed mode disabled by default."""
    path = tmp_path / "gateway.db"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    connection = sqlite3.connect(path)
    try:
        connection.execute("BEGIN EXCLUSIVE")
        for migration in (
            _MIGRATION_1,
            _MIGRATION_2,
            _MIGRATION_3,
            _MIGRATION_4,
            _MIGRATION_5,
            _MIGRATION_6,
            _MIGRATION_7,
        ):
            for statement in migration:
                connection.execute(statement)
        connection.execute(
            """
            INSERT INTO gateway_monthly_budgets (
                budget_id, organization_id, period_start, scope_kind, scope_key,
                limit_micro_usd, created_at, updated_at
            ) VALUES (?, ?, ?, 'team', ?, 1000, ?, ?)
            """,
            (
                "budget-one",
                "org-one",
                "2026-08-01T00:00:00+00:00",
                "scope-one",
                "2026-08-18T00:00:00+00:00",
                "2026-08-18T00:00:00+00:00",
            ),
        )
        connection.execute("PRAGMA user_version = 7")
        connection.execute("COMMIT")
    finally:
        connection.close()

    backup = initialize_database(path)

    assert backup is not None
    current = sqlite3.connect(path)
    try:
        row = current.execute(
            "SELECT strict_unknown_cost FROM gateway_monthly_budgets WHERE budget_id = 'budget-one'"
        ).fetchone()
        assert row == (0,)
        assert current.execute("PRAGMA user_version").fetchone() == (SCHEMA_VERSION,)
    finally:
        current.close()
    prior = sqlite3.connect(backup)
    try:
        columns = {
            str(row[1]) for row in prior.execute("PRAGMA table_info(gateway_monthly_budgets)")
        }
        assert "strict_unknown_cost" not in columns
        assert prior.execute("PRAGMA user_version").fetchone() == (7,)
    finally:
        prior.close()


def test_failed_legacy_migration_rolls_back_the_live_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed forward migration leaves the legacy database intact and recoverable."""
    path = tmp_path / "gateway.db"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    connection = sqlite3.connect(path)
    try:
        connection.execute("BEGIN EXCLUSIVE")
        for migration in (_MIGRATION_1, _MIGRATION_2, _MIGRATION_3):
            for statement in migration:
                connection.execute(statement)
        connection.execute("PRAGMA user_version = 3")
        connection.execute("COMMIT")
    finally:
        connection.close()
    monkeypatch.setitem(
        migrations._MIGRATIONS,
        4,
        (
            "ALTER TABLE gateway_attempts RENAME TO gateway_attempts_v3",
            "INVALID MIGRATION STATEMENT",
        ),
    )

    with pytest.raises(GatewaySchemaError, match="migration failed"):
        initialize_database(path)

    connection = sqlite3.connect(path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (3,)
        tables = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    finally:
        connection.close()
    assert "gateway_attempts" in tables
    assert "gateway_attempts_v3" not in tables
    backups = tuple(tmp_path.glob("gateway.db.backup-v3-*"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as backup:
        assert backup.execute("PRAGMA user_version").fetchone() == (3,)


def test_concurrent_initializers_choose_migration_plan_under_exclusive_lock(
    tmp_path: Path,
) -> None:
    """Concurrent initializers re-read schema version after exclusive serialization."""
    path = tmp_path / "gateway.db"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    connection = connect_database(path)
    try:
        connection.execute("BEGIN EXCLUSIVE")
        for statement in _MIGRATION_1:
            connection.execute(statement)
        connection.execute("PRAGMA user_version = 1")
        connection.execute("COMMIT")
    finally:
        connection.close()
    barrier = threading.Barrier(2)

    def initialize() -> Path | None:
        """Start one initializer at the same concurrency boundary."""
        barrier.wait(timeout=5)
        return initialize_database(path, busy_timeout_ms=10_000)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (executor.submit(initialize), executor.submit(initialize))
        results = tuple(future.result(timeout=15) for future in futures)

    assert sum(result is not None for result in results) == 1
    connection = connect_database(path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        connection.close()


def test_newer_and_marker_only_schemas_refuse_without_deleting_state(tmp_path: Path) -> None:
    """Unknown future versions and missing schema objects fail closed."""
    newer = tmp_path / "newer.db"
    descriptor = os.open(newer, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    connection = sqlite3.connect(newer)
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    connection.close()

    with pytest.raises(GatewaySchemaError, match="newer"):
        initialize_database(newer)
    assert newer.exists()
    connection = sqlite3.connect(newer)
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    finally:
        connection.close()

    marker_only = tmp_path / "marker-only.db"
    descriptor = os.open(marker_only, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    connection = sqlite3.connect(marker_only)
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    connection.close()
    with pytest.raises(GatewaySchemaError, match="marker"):
        initialize_database(marker_only)

    corrupt = tmp_path / "corrupt.db"
    corrupt.write_bytes(b"not a sqlite database")
    corrupt.chmod(0o600)
    with pytest.raises(GatewaySchemaError, match="corrupt"):
        initialize_database(corrupt)


def test_v10_migration_widens_api_surface_and_preserves_rows(tmp_path: Path) -> None:
    """The v10 rewrite admits the messages surface without touching v9 data.

    A v9 database with one full request-and-attempt chain migrates in place:
    the existing rows and the child foreign key survive, a ``messages``
    request becomes insertable, and any other surface value stays rejected.
    """
    path = tmp_path / "gateway.db"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    connection = connect_database(path)
    try:
        connection.execute("BEGIN EXCLUSIVE")
        for version in range(1, 10):
            for statement in migrations._MIGRATIONS[version]:
                connection.execute(statement)
        connection.execute("PRAGMA user_version = 9")
        seed_statements = """
            INSERT INTO organizations VALUES ('org', 'org', 'Org', 1, 't', 't');
            INSERT INTO identities VALUES ('id', 'org', 'Identity', NULL, 1, 't', 't');
            INSERT INTO virtual_keys (
                key_id, organization_id, identity_id, prefix,
                fingerprint_version, fingerprint_sha256, created_at
            ) VALUES ('key', 'org', 'id', 'pfx', 1, '{fingerprint}', 't');
            INSERT INTO catalog_snapshot_refs VALUES ('snap', 'org', '{digest}', 't');
            INSERT INTO gateway_aliases (
                alias_id, organization_id, alias_name, active_revision_id,
                created_at, updated_at
            ) VALUES ('alias', 'org', 'alias', NULL, 't', 't');
            INSERT INTO alias_revisions (
                revision_id, organization_id, alias_id, revision_number,
                target_kind, pool_id, catalog_sha256, snapshot_ref, created_at
            ) VALUES ('rev', 'org', 'alias', 1, 'direct', 'pool', '{digest}', 'snap', 't');
            INSERT INTO gateway_requests (
                request_id, organization_id, identity_id, key_id, alias_id,
                alias_revision_id, api_surface, canonical_request_sha256,
                accepted_at, deadline_at
            ) VALUES (
                'req-1', 'org', 'id', 'key', 'alias', 'rev', 'chat_completions',
                '{digest}', 't', 't'
            );
            INSERT INTO gateway_attempts (
                attempt_id, request_id, organization_id, attempt_ordinal,
                route_depth, deployment_id, provider, exact_model_id, pool_id,
                catalog_sha256, state, started_at, budget_period_start
            ) VALUES (
                'att-1', 'req-1', 'org', 0, 0, 'deploy', 'provider', 'exact',
                'pool', '{digest}', 'completed', 't', '2026-08-01T00:00:00+00:00'
            );
            """.format(fingerprint="a" * 64, digest="b" * 64)
        # executescript would commit the exclusive transaction implicitly, so
        # the seed rows are inserted statement by statement instead.
        for statement in seed_statements.split(";"):
            if statement.strip():
                connection.execute(statement)
        connection.execute("COMMIT")
    finally:
        connection.close()

    backup = initialize_database(path)

    assert backup is not None and backup.exists()
    migrated = connect_database(path)
    try:
        assert migrated.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert migrated.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert migrated.execute("PRAGMA foreign_key_check").fetchall() == []
        surviving = migrated.execute(
            "SELECT api_surface, app_referer FROM gateway_requests WHERE request_id = 'req-1'"
        ).fetchone()
        assert (surviving[0], surviving[1]) == ("chat_completions", None)
        request_columns = (
            "request_id, organization_id, identity_id, key_id, alias_id, "
            "alias_revision_id, api_surface, canonical_request_sha256, accepted_at, deadline_at"
        )
        migrated.execute(
            f"INSERT INTO gateway_requests ({request_columns}) "
            "VALUES ('req-2', 'org', 'id', 'key', 'alias', 'rev', 'messages', ?, 't', 't')",
            ("c" * 64,),
        )
        with pytest.raises(sqlite3.IntegrityError):
            migrated.execute(
                f"INSERT INTO gateway_requests ({request_columns}) "
                "VALUES ('req-3', 'org', 'id', 'key', 'alias', 'rev', 'bogus', ?, 't', 't')",
                ("d" * 64,),
            )
        # The child attempt still resolves its rewritten parent's foreign key.
        migrated.execute("DELETE FROM gateway_attempts WHERE attempt_id = 'att-1'")
        migrated.execute("DELETE FROM gateway_requests WHERE request_id = 'req-1'")
    finally:
        migrated.close()


def test_v11_migration_adds_azure_surface_without_rewriting_existing_authority(
    tmp_path: Path,
) -> None:
    """Existing v10 provider revisions migrate with the classic Azure default intact."""
    path = tmp_path / "gateway.db"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    connection = connect_database(path)
    config = ConnectionConfig(
        provider="azure",
        base_url="https://resource.openai.azure.com",
        api_key_env="AZURE_OPENAI_API_KEY",
        api_version="v1",
    )
    try:
        connection.execute("BEGIN EXCLUSIVE")
        for version in range(1, 11):
            for statement in migrations._MIGRATIONS[version]:
                connection.execute(statement)
        connection.execute("PRAGMA user_version = 10")
        connection.execute("INSERT INTO organizations VALUES ('org', 'org', 'Org', 1, 't', 't')")
        connection.execute(
            """
            INSERT INTO provider_connections (
                connection_id, organization_id, active, created_at, updated_at
            ) VALUES ('azure', 'org', 1, 't', 't')
            """
        )
        connection.execute(
            """
            INSERT INTO provider_connection_revisions (
                revision_id, organization_id, connection_id, revision_number,
                provider, base_url, api_key_env, api_version, region,
                connection_sha256, created_at
            ) VALUES ('rev', 'org', 'azure', 1, 'azure', ?, ?, 'v1', NULL, ?, 't')
            """,
            (config.base_url, config.api_key_env, config.identity_sha256()),
        )
        connection.execute(
            "UPDATE provider_connections SET active_revision_id = 'rev' "
            "WHERE connection_id = 'azure'"
        )
        connection.execute("COMMIT")
    finally:
        connection.close()

    backup = initialize_database(path)

    assert backup is not None and backup.exists()
    migrated = connect_database(path)
    try:
        assert migrated.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        row = migrated.execute(
            "SELECT azure_api_surface, connection_sha256 "
            "FROM provider_connection_revisions WHERE revision_id = 'rev'"
        ).fetchone()
        assert row["azure_api_surface"] is None
        assert row["connection_sha256"] == config.identity_sha256()
        with pytest.raises(sqlite3.IntegrityError):
            migrated.execute(
                "UPDATE provider_connection_revisions SET azure_api_surface = 'bogus' "
                "WHERE revision_id = 'rev'"
            )
    finally:
        migrated.close()


def test_v12_adds_bedrock_auth_locators_and_preserves_ambient_authority(
    tmp_path: Path,
) -> None:
    """The current schema gains nullable Bedrock metadata without changing ambient identity."""
    path = tmp_path / "gateway.db"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    ambient = ConnectionConfig(provider="bedrock", region="us-west-2")
    connection = connect_database(path)
    try:
        connection.execute("BEGIN EXCLUSIVE")
        for version in range(1, 12):
            for statement in migrations._MIGRATIONS[version]:
                connection.execute(statement)
        connection.execute("PRAGMA user_version = 11")
        connection.execute("INSERT INTO organizations VALUES ('org', 'org', 'Org', 1, 't', 't')")
        connection.execute(
            """
            INSERT INTO provider_connections (
                connection_id, organization_id, active_revision_id,
                active, created_at, updated_at
            ) VALUES ('bedrock', 'org', NULL, 1, 't', 't')
            """
        )
        connection.execute(
            """
            INSERT INTO provider_connection_revisions (
                revision_id, organization_id, connection_id, revision_number,
                provider, region, azure_api_surface, connection_sha256, created_at
            ) VALUES ('bedrock-revision', 'org', 'bedrock', 1,
                      'bedrock', 'us-west-2', NULL, ?, 't')
            """,
            (ambient.identity_sha256(),),
        )
        connection.execute(
            "UPDATE provider_connections SET active_revision_id = 'bedrock-revision' "
            "WHERE connection_id = 'bedrock'"
        )
        connection.execute("COMMIT")
    finally:
        connection.close()

    backup = initialize_database(path)

    assert backup is not None and backup.exists()
    migrated = connect_database(path)
    try:
        assert migrated.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        columns = {
            str(row[1])
            for row in migrated.execute(
                "PRAGMA table_info(provider_connection_revisions)"
            ).fetchall()
        }
        assert {"aws_access_key_id_env", "bedrock_auth_mode"} <= columns
        row = migrated.execute(
            "SELECT aws_access_key_id_env, bedrock_auth_mode, connection_sha256 "
            "FROM provider_connection_revisions WHERE revision_id = 'bedrock-revision'"
        ).fetchone()
        assert tuple(row) == (None, None, ambient.identity_sha256())
        authorities = active_provider_connections(migrated, organization_id="org")
        assert len(authorities) == 1
        assert authorities[0].config == ambient
        assert migrated.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert migrated.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        migrated.close()


def test_v14_migration_widens_api_surface_to_embeddings_and_preserves_rows(
    tmp_path: Path,
) -> None:
    """The v14 rewrite admits the embeddings surface without touching v13 data.

    A v13 database with one full request-and-attempt chain migrates in place:
    the existing rows and the child foreign key survive, an ``embeddings``
    request becomes insertable, and any other surface value stays rejected.
    """
    path = tmp_path / "gateway.db"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    connection = connect_database(path)
    try:
        connection.execute("BEGIN EXCLUSIVE")
        for version in range(1, 14):
            for statement in migrations._MIGRATIONS[version]:
                connection.execute(statement)
        connection.execute("PRAGMA user_version = 13")
        seed_statements = """
            INSERT INTO organizations VALUES ('org', 'org', 'Org', 1, 't', 't');
            INSERT INTO identities VALUES ('id', 'org', 'Identity', NULL, 1, 't', 't');
            INSERT INTO virtual_keys (
                key_id, organization_id, identity_id, prefix,
                fingerprint_version, fingerprint_sha256, created_at
            ) VALUES ('key', 'org', 'id', 'pfx', 1, '{fingerprint}', 't');
            INSERT INTO catalog_snapshot_refs VALUES ('snap', 'org', '{digest}', 't');
            INSERT INTO gateway_aliases (
                alias_id, organization_id, alias_name, active_revision_id,
                created_at, updated_at
            ) VALUES ('alias', 'org', 'alias', NULL, 't', 't');
            INSERT INTO alias_revisions (
                revision_id, organization_id, alias_id, revision_number,
                target_kind, pool_id, catalog_sha256, snapshot_ref, created_at
            ) VALUES ('rev', 'org', 'alias', 1, 'direct', 'pool', '{digest}', 'snap', 't');
            INSERT INTO gateway_requests (
                request_id, organization_id, identity_id, key_id, alias_id,
                alias_revision_id, api_surface, canonical_request_sha256,
                accepted_at, deadline_at
            ) VALUES (
                'req-1', 'org', 'id', 'key', 'alias', 'rev', 'messages',
                '{digest}', 't', 't'
            );
            INSERT INTO gateway_attempts (
                attempt_id, request_id, organization_id, attempt_ordinal,
                route_depth, deployment_id, provider, exact_model_id, pool_id,
                catalog_sha256, state, started_at, budget_period_start
            ) VALUES (
                'att-1', 'req-1', 'org', 0, 0, 'deploy', 'provider', 'exact',
                'pool', '{digest}', 'completed', 't', '2026-08-01T00:00:00+00:00'
            );
            """.format(fingerprint="a" * 64, digest="b" * 64)
        for statement in seed_statements.split(";"):
            if statement.strip():
                connection.execute(statement)
        connection.execute("COMMIT")
    finally:
        connection.close()

    backup = initialize_database(path)

    assert backup is not None and backup.exists()
    migrated = connect_database(path)
    try:
        assert migrated.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert migrated.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert migrated.execute("PRAGMA foreign_key_check").fetchall() == []
        surviving = migrated.execute(
            "SELECT api_surface FROM gateway_requests WHERE request_id = 'req-1'"
        ).fetchone()
        assert surviving[0] == "messages"
        request_columns = (
            "request_id, organization_id, identity_id, key_id, alias_id, "
            "alias_revision_id, api_surface, canonical_request_sha256, accepted_at, deadline_at"
        )
        migrated.execute(
            f"INSERT INTO gateway_requests ({request_columns}) "
            "VALUES ('req-2', 'org', 'id', 'key', 'alias', 'rev', 'embeddings', ?, 't', 't')",
            ("c" * 64,),
        )
        with pytest.raises(sqlite3.IntegrityError):
            migrated.execute(
                f"INSERT INTO gateway_requests ({request_columns}) "
                "VALUES ('req-3', 'org', 'id', 'key', 'alias', 'rev', 'bogus', ?, 't', 't')",
                ("d" * 64,),
            )
        migrated.execute("DELETE FROM gateway_attempts WHERE attempt_id = 'att-1'")
        migrated.execute("DELETE FROM gateway_requests WHERE request_id = 'req-1'")
    finally:
        migrated.close()


def test_v15_migration_widens_api_surface_to_images_and_preserves_rows(
    tmp_path: Path,
) -> None:
    """The v15 rewrite admits the images surface without touching v14 data.

    A v14 database with one full request-and-attempt chain migrates in place:
    the existing rows and the child foreign key survive, an ``images``
    request becomes insertable, and any other surface value stays rejected.
    """
    path = tmp_path / "gateway.db"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    connection = connect_database(path)
    try:
        connection.execute("BEGIN EXCLUSIVE")
        for version in range(1, 15):
            for statement in migrations._MIGRATIONS[version]:
                connection.execute(statement)
        connection.execute("PRAGMA user_version = 14")
        seed_statements = """
            INSERT INTO organizations VALUES ('org', 'org', 'Org', 1, 't', 't');
            INSERT INTO identities VALUES ('id', 'org', 'Identity', NULL, 1, 't', 't');
            INSERT INTO virtual_keys (
                key_id, organization_id, identity_id, prefix,
                fingerprint_version, fingerprint_sha256, created_at
            ) VALUES ('key', 'org', 'id', 'pfx', 1, '{fingerprint}', 't');
            INSERT INTO catalog_snapshot_refs VALUES ('snap', 'org', '{digest}', 't');
            INSERT INTO gateway_aliases (
                alias_id, organization_id, alias_name, active_revision_id,
                created_at, updated_at
            ) VALUES ('alias', 'org', 'alias', NULL, 't', 't');
            INSERT INTO alias_revisions (
                revision_id, organization_id, alias_id, revision_number,
                target_kind, pool_id, catalog_sha256, snapshot_ref, created_at
            ) VALUES ('rev', 'org', 'alias', 1, 'direct', 'pool', '{digest}', 'snap', 't');
            INSERT INTO gateway_requests (
                request_id, organization_id, identity_id, key_id, alias_id,
                alias_revision_id, api_surface, canonical_request_sha256,
                accepted_at, deadline_at
            ) VALUES (
                'req-1', 'org', 'id', 'key', 'alias', 'rev', 'embeddings',
                '{digest}', 't', 't'
            );
            INSERT INTO gateway_attempts (
                attempt_id, request_id, organization_id, attempt_ordinal,
                route_depth, deployment_id, provider, exact_model_id, pool_id,
                catalog_sha256, state, started_at, budget_period_start
            ) VALUES (
                'att-1', 'req-1', 'org', 0, 0, 'deploy', 'provider', 'exact',
                'pool', '{digest}', 'completed', 't', '2026-08-01T00:00:00+00:00'
            );
            """.format(fingerprint="a" * 64, digest="b" * 64)
        for statement in seed_statements.split(";"):
            if statement.strip():
                connection.execute(statement)
        connection.execute("COMMIT")
    finally:
        connection.close()

    backup = initialize_database(path)

    assert backup is not None and backup.exists()
    migrated = connect_database(path)
    try:
        assert migrated.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert migrated.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert migrated.execute("PRAGMA foreign_key_check").fetchall() == []
        surviving = migrated.execute(
            "SELECT api_surface FROM gateway_requests WHERE request_id = 'req-1'"
        ).fetchone()
        assert surviving[0] == "embeddings"
        request_columns = (
            "request_id, organization_id, identity_id, key_id, alias_id, "
            "alias_revision_id, api_surface, canonical_request_sha256, accepted_at, deadline_at"
        )
        migrated.execute(
            f"INSERT INTO gateway_requests ({request_columns}) "
            "VALUES ('req-2', 'org', 'id', 'key', 'alias', 'rev', 'images', ?, 't', 't')",
            ("c" * 64,),
        )
        with pytest.raises(sqlite3.IntegrityError):
            migrated.execute(
                f"INSERT INTO gateway_requests ({request_columns}) "
                "VALUES ('req-3', 'org', 'id', 'key', 'alias', 'rev', 'bogus', ?, 't', 't')",
                ("d" * 64,),
            )
        migrated.execute("DELETE FROM gateway_attempts WHERE attempt_id = 'att-1'")
        migrated.execute("DELETE FROM gateway_requests WHERE request_id = 'req-1'")
    finally:
        migrated.close()


def test_v17_migration_adds_null_dispatch_disclosures_to_existing_attempts(
    tmp_path: Path,
) -> None:
    """A v16 database migrates with every disclosure column NULL on old rows.

    Deployed ledgers carry settled attempt rows from before dispatch-policy
    disclosures existed; the ALTER-only migration must leave those rows intact
    with all seven new columns NULL (never a default that reads as a
    disclosure) while new writes can populate them.
    """
    path = tmp_path / "gateway.db"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    connection = connect_database(path)
    try:
        connection.execute("BEGIN EXCLUSIVE")
        for version in range(1, 17):
            for statement in migrations._MIGRATIONS[version]:
                connection.execute(statement)
        connection.execute("PRAGMA user_version = 16")
        seed_statements = """
            INSERT INTO organizations VALUES ('org', 'org', 'Org', 1, 't', 't');
            INSERT INTO identities VALUES ('id', 'org', 'Identity', NULL, 1, 't', 't');
            INSERT INTO virtual_keys (
                key_id, organization_id, identity_id, prefix,
                fingerprint_version, fingerprint_sha256, created_at
            ) VALUES ('key', 'org', 'id', 'pfx', 1, '{fingerprint}', 't');
            INSERT INTO catalog_snapshot_refs VALUES ('snap', 'org', '{digest}', 't');
            INSERT INTO gateway_aliases (
                alias_id, organization_id, alias_name, active_revision_id,
                created_at, updated_at
            ) VALUES ('alias', 'org', 'alias', NULL, 't', 't');
            INSERT INTO alias_revisions (
                revision_id, organization_id, alias_id, revision_number,
                target_kind, pool_id, catalog_sha256, snapshot_ref, created_at
            ) VALUES ('rev', 'org', 'alias', 1, 'direct', 'pool', '{digest}', 'snap', 't');
            INSERT INTO gateway_requests (
                request_id, organization_id, identity_id, key_id, alias_id,
                alias_revision_id, api_surface, canonical_request_sha256,
                accepted_at, deadline_at
            ) VALUES (
                'req-1', 'org', 'id', 'key', 'alias', 'rev', 'chat_completions',
                '{digest}', 't', 't'
            );
            INSERT INTO gateway_attempts (
                attempt_id, request_id, organization_id, attempt_ordinal,
                route_depth, deployment_id, provider, exact_model_id, pool_id,
                catalog_sha256, state, started_at, budget_period_start
            ) VALUES (
                'att-1', 'req-1', 'org', 0, 0, 'deploy', 'provider', 'exact',
                'pool', '{digest}', 'completed', 't', '2026-08-01T00:00:00+00:00'
            );
            """.format(fingerprint="a" * 64, digest="b" * 64)
        # executescript would commit the exclusive transaction implicitly, so
        # the seed rows are inserted statement by statement instead.
        for statement in seed_statements.split(";"):
            if statement.strip():
                connection.execute(statement)
        connection.execute("COMMIT")
    finally:
        connection.close()

    backup = initialize_database(path)

    assert backup is not None and backup.exists()
    disclosure_columns = (
        "dispatch_reason",
        "preferred_deployment_id",
        "preferred_input_rate",
        "preferred_cached_input_rate",
        "preferred_output_rate",
        "preferred_reasoning_rate",
        "counterfactual_cost_micro_usd",
    )
    migrated = connect_database(path)
    try:
        assert migrated.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert migrated.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert migrated.execute("PRAGMA foreign_key_check").fetchall() == []
        selected = ", ".join(disclosure_columns)
        row = migrated.execute(
            f"SELECT state, {selected} FROM gateway_attempts WHERE attempt_id = 'att-1'"  # noqa: S608 - fixed column names.
        ).fetchone()
        assert row is not None
        assert row[0] == "completed"
        assert tuple(row)[1:] == (None,) * len(disclosure_columns)
        # New writes can populate the disclosure; old rows never gain one.
        migrated.execute(
            "UPDATE gateway_attempts SET dispatch_reason = 'queue_bound',"
            " preferred_deployment_id = 'deploy-lead', preferred_input_rate = 1"
            " WHERE attempt_id = 'att-1'"
        )
    finally:
        migrated.close()
    prior = sqlite3.connect(backup)
    try:
        columns = {str(entry[1]) for entry in prior.execute("PRAGMA table_info(gateway_attempts)")}
        assert not columns.intersection(disclosure_columns)
        assert prior.execute("PRAGMA user_version").fetchone() == (16,)
    finally:
        prior.close()
