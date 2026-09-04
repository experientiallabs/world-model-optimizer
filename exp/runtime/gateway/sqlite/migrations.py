"""Forward-only SQLite schema initialization and guarded migration."""

from __future__ import annotations

import os
import sqlite3
import stat
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 17


class GatewaySchemaError(RuntimeError):
    """The gateway database schema cannot be opened safely."""


_MIGRATION_1 = (
    """
    CREATE TABLE organizations (
        organization_id TEXT PRIMARY KEY,
        slug TEXT NOT NULL UNIQUE,
        display_name TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE identities (
        identity_id TEXT PRIMARY KEY,
        organization_id TEXT NOT NULL,
        display_name TEXT NOT NULL,
        description TEXT,
        active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (organization_id, identity_id),
        FOREIGN KEY (organization_id) REFERENCES organizations (organization_id)
    ) STRICT
    """,
    """
    CREATE TABLE virtual_keys (
        key_id TEXT PRIMARY KEY,
        organization_id TEXT NOT NULL,
        identity_id TEXT NOT NULL,
        prefix TEXT NOT NULL,
        fingerprint_version INTEGER NOT NULL CHECK (fingerprint_version > 0),
        fingerprint_sha256 TEXT NOT NULL CHECK (length(fingerprint_sha256) = 64),
        expires_at TEXT,
        revoked_at TEXT,
        last_used_at TEXT,
        created_at TEXT NOT NULL,
        UNIQUE (fingerprint_version, fingerprint_sha256),
        UNIQUE (organization_id, prefix),
        UNIQUE (organization_id, key_id),
        FOREIGN KEY (organization_id, identity_id)
            REFERENCES identities (organization_id, identity_id)
    ) STRICT
    """,
    """
    CREATE TABLE catalog_snapshot_refs (
        snapshot_ref TEXT PRIMARY KEY,
        organization_id TEXT NOT NULL,
        catalog_sha256 TEXT NOT NULL CHECK (length(catalog_sha256) = 64),
        created_at TEXT NOT NULL,
        UNIQUE (organization_id, catalog_sha256),
        UNIQUE (organization_id, snapshot_ref),
        FOREIGN KEY (organization_id) REFERENCES organizations (organization_id)
    ) STRICT
    """,
    """
    CREATE TABLE gateway_aliases (
        alias_id TEXT PRIMARY KEY,
        organization_id TEXT NOT NULL,
        alias_name TEXT NOT NULL,
        active_revision_id TEXT,
        active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (organization_id, alias_name),
        UNIQUE (organization_id, alias_id),
        FOREIGN KEY (organization_id) REFERENCES organizations (organization_id),
        FOREIGN KEY (organization_id, alias_id, active_revision_id)
            REFERENCES alias_revisions (organization_id, alias_id, revision_id)
            DEFERRABLE INITIALLY DEFERRED
    ) STRICT
    """,
    """
    CREATE TABLE alias_revisions (
        revision_id TEXT PRIMARY KEY,
        organization_id TEXT NOT NULL,
        alias_id TEXT NOT NULL,
        revision_number INTEGER NOT NULL CHECK (revision_number > 0),
        target_kind TEXT NOT NULL CHECK (target_kind IN ('direct', 'project')),
        pool_id TEXT,
        project_ref TEXT,
        activation_ref TEXT,
        catalog_sha256 TEXT NOT NULL CHECK (length(catalog_sha256) = 64),
        snapshot_ref TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (organization_id, alias_id, revision_number),
        UNIQUE (organization_id, alias_id, revision_id),
        CHECK (
            (target_kind = 'direct' AND pool_id IS NOT NULL
                AND project_ref IS NULL AND activation_ref IS NULL)
            OR
            (target_kind = 'project' AND pool_id IS NULL
                AND project_ref IS NOT NULL AND activation_ref IS NOT NULL)
        ),
        FOREIGN KEY (organization_id, alias_id)
            REFERENCES gateway_aliases (organization_id, alias_id),
        FOREIGN KEY (organization_id, snapshot_ref)
            REFERENCES catalog_snapshot_refs (organization_id, snapshot_ref)
    ) STRICT
    """,
    """
    CREATE TABLE project_activation_bindings (
        organization_id TEXT NOT NULL,
        project_ref TEXT NOT NULL,
        activation_ref TEXT NOT NULL,
        alias_id TEXT NOT NULL,
        revision_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (organization_id, project_ref, activation_ref),
        UNIQUE (organization_id, alias_id),
        UNIQUE (organization_id, revision_id),
        FOREIGN KEY (organization_id, alias_id, revision_id)
            REFERENCES alias_revisions (organization_id, alias_id, revision_id)
            ON DELETE CASCADE
    ) STRICT
    """,
    """
    CREATE TABLE identity_alias_grants (
        organization_id TEXT NOT NULL,
        identity_id TEXT NOT NULL,
        alias_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (organization_id, identity_id, alias_id),
        FOREIGN KEY (organization_id, identity_id)
            REFERENCES identities (organization_id, identity_id),
        FOREIGN KEY (organization_id, alias_id)
            REFERENCES gateway_aliases (organization_id, alias_id)
    ) STRICT
    """,
    """
    CREATE TABLE operation_receipts (
        organization_id TEXT NOT NULL,
        operation_id TEXT NOT NULL,
        operation_kind TEXT NOT NULL,
        request_sha256 TEXT NOT NULL CHECK (length(request_sha256) = 64),
        resource_kind TEXT NOT NULL,
        resource_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (organization_id, operation_id),
        FOREIGN KEY (organization_id) REFERENCES organizations (organization_id)
    ) STRICT
    """,
)

_MIGRATION_2 = (
    """
    CREATE TABLE gateway_requests (
        request_id TEXT PRIMARY KEY,
        organization_id TEXT NOT NULL,
        identity_id TEXT NOT NULL,
        key_id TEXT NOT NULL,
        alias_id TEXT NOT NULL,
        alias_revision_id TEXT NOT NULL,
        api_surface TEXT NOT NULL CHECK (api_surface IN ('chat_completions', 'responses')),
        canonical_request_sha256 TEXT NOT NULL CHECK (length(canonical_request_sha256) = 64),
        caller_operation_sha256 TEXT,
        accepted_at TEXT NOT NULL,
        deadline_at TEXT NOT NULL,
        terminal_state TEXT CHECK (
            terminal_state IS NULL OR terminal_state IN (
                'completed', 'failed', 'cancelled', 'incomplete',
                'expired_before_dispatch', 'unknown_after_crash'
            )
        ),
        terminal_at TEXT,
        content_retained INTEGER NOT NULL DEFAULT 0 CHECK (content_retained = 0),
        UNIQUE (organization_id, request_id),
        FOREIGN KEY (organization_id, identity_id)
            REFERENCES identities (organization_id, identity_id),
        FOREIGN KEY (organization_id, key_id)
            REFERENCES virtual_keys (organization_id, key_id),
        FOREIGN KEY (organization_id, alias_id, alias_revision_id)
            REFERENCES alias_revisions (organization_id, alias_id, revision_id)
    ) STRICT
    """,
    """
    CREATE INDEX gateway_requests_caller_operation
    ON gateway_requests (
        organization_id, identity_id, alias_revision_id, api_surface, caller_operation_sha256
    ) WHERE caller_operation_sha256 IS NOT NULL
    """,
    """
    CREATE TABLE gateway_attempts (
        attempt_id TEXT PRIMARY KEY,
        request_id TEXT NOT NULL,
        organization_id TEXT NOT NULL,
        route_depth INTEGER NOT NULL CHECK (route_depth >= 0),
        deployment_id TEXT NOT NULL,
        provider TEXT NOT NULL,
        exact_model_id TEXT NOT NULL,
        pool_id TEXT NOT NULL,
        catalog_sha256 TEXT NOT NULL CHECK (length(catalog_sha256) = 64),
        pricing_source TEXT,
        pricing_effective_at TEXT,
        route_reason TEXT,
        fallback_reason TEXT,
        input_rate INTEGER CHECK (input_rate IS NULL OR input_rate >= 0),
        cached_input_rate INTEGER CHECK (cached_input_rate IS NULL OR cached_input_rate >= 0),
        output_rate INTEGER CHECK (output_rate IS NULL OR output_rate >= 0),
        reasoning_rate INTEGER CHECK (reasoning_rate IS NULL OR reasoning_rate >= 0),
        state TEXT NOT NULL CHECK (state IN (
            'dispatched', 'completed', 'failed', 'cancelled',
            'incomplete', 'unknown_after_crash'
        )),
        started_at TEXT NOT NULL,
        terminal_at TEXT,
        failure_class TEXT,
        input_tokens INTEGER CHECK (input_tokens IS NULL OR input_tokens >= 0),
        cached_input_tokens INTEGER CHECK (
            cached_input_tokens IS NULL OR cached_input_tokens >= 0
        ),
        output_tokens INTEGER CHECK (output_tokens IS NULL OR output_tokens >= 0),
        reasoning_tokens INTEGER CHECK (reasoning_tokens IS NULL OR reasoning_tokens >= 0),
        usage_source TEXT CHECK (
            usage_source IS NULL OR usage_source IN ('observed', 'estimated', 'unknown')
        ),
        estimated_cost_micro_usd INTEGER CHECK (
            estimated_cost_micro_usd IS NULL OR estimated_cost_micro_usd >= 0
        ),
        content_retained INTEGER NOT NULL DEFAULT 0 CHECK (content_retained = 0),
        UNIQUE (organization_id, attempt_id),
        UNIQUE (request_id, route_depth),
        FOREIGN KEY (organization_id, request_id)
            REFERENCES gateway_requests (organization_id, request_id)
    ) STRICT
    """,
    """
    CREATE INDEX gateway_attempts_usage
    ON gateway_attempts (organization_id, terminal_at, state)
    """,
    """
    CREATE INDEX gateway_requests_identity
    ON gateway_requests (organization_id, identity_id, accepted_at)
    """,
)

_MIGRATION_3 = (
    """
    ALTER TABLE gateway_attempts
    ADD COLUMN billing_source TEXT NOT NULL DEFAULT 'customer_managed'
    CHECK (billing_source IN ('customer_managed', 'host_managed'))
    """,
)

_MIGRATION_4 = (
    """
    CREATE TABLE provider_connections (
        connection_id TEXT PRIMARY KEY,
        organization_id TEXT NOT NULL,
        active_revision_id TEXT,
        active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (organization_id, connection_id),
        FOREIGN KEY (organization_id) REFERENCES organizations (organization_id),
        FOREIGN KEY (organization_id, connection_id, active_revision_id)
            REFERENCES provider_connection_revisions (
                organization_id, connection_id, revision_id
            )
            DEFERRABLE INITIALLY DEFERRED
    ) STRICT
    """,
    """
    CREATE TABLE provider_connection_revisions (
        revision_id TEXT PRIMARY KEY,
        organization_id TEXT NOT NULL,
        connection_id TEXT NOT NULL,
        revision_number INTEGER NOT NULL CHECK (revision_number > 0),
        provider TEXT NOT NULL,
        base_url TEXT,
        api_key_env TEXT,
        api_version TEXT,
        region TEXT,
        connection_sha256 TEXT NOT NULL CHECK (length(connection_sha256) = 64),
        created_at TEXT NOT NULL,
        UNIQUE (organization_id, connection_id, revision_number),
        UNIQUE (organization_id, connection_id, revision_id),
        FOREIGN KEY (organization_id, connection_id)
            REFERENCES provider_connections (organization_id, connection_id)
    ) STRICT
    """,
    """
    CREATE TABLE alias_revision_provider_connections (
        organization_id TEXT NOT NULL,
        alias_id TEXT NOT NULL,
        alias_revision_id TEXT NOT NULL,
        connection_id TEXT NOT NULL,
        connection_revision_id TEXT NOT NULL,
        connection_sha256 TEXT NOT NULL CHECK (length(connection_sha256) = 64),
        created_at TEXT NOT NULL,
        PRIMARY KEY (
            organization_id, alias_id, alias_revision_id, connection_id
        ),
        FOREIGN KEY (organization_id, alias_id, alias_revision_id)
            REFERENCES alias_revisions (organization_id, alias_id, revision_id)
            ON DELETE CASCADE,
        FOREIGN KEY (
            organization_id, connection_id, connection_revision_id
        ) REFERENCES provider_connection_revisions (
            organization_id, connection_id, revision_id
        )
    ) STRICT
    """,
)

_MIGRATION_5 = (
    """
    ALTER TABLE alias_revisions
    ADD COLUMN refusal_failover INTEGER NOT NULL DEFAULT 0 CHECK (refusal_failover IN (0, 1))
    """,
)

_MIGRATION_6 = (
    "ALTER TABLE gateway_attempts RENAME TO gateway_attempts_v5",
    """
    CREATE TABLE gateway_attempts (
        attempt_id TEXT PRIMARY KEY,
        request_id TEXT NOT NULL,
        organization_id TEXT NOT NULL,
        attempt_ordinal INTEGER NOT NULL CHECK (attempt_ordinal >= 0),
        route_depth INTEGER NOT NULL CHECK (route_depth >= 0),
        deployment_id TEXT NOT NULL,
        provider TEXT NOT NULL,
        exact_model_id TEXT NOT NULL,
        pool_id TEXT NOT NULL,
        catalog_sha256 TEXT NOT NULL CHECK (length(catalog_sha256) = 64),
        billing_source TEXT NOT NULL DEFAULT 'customer_managed'
            CHECK (billing_source IN ('customer_managed', 'host_managed')),
        pricing_source TEXT,
        pricing_effective_at TEXT,
        route_reason TEXT,
        fallback_reason TEXT,
        input_rate INTEGER CHECK (input_rate IS NULL OR input_rate >= 0),
        cached_input_rate INTEGER CHECK (cached_input_rate IS NULL OR cached_input_rate >= 0),
        output_rate INTEGER CHECK (output_rate IS NULL OR output_rate >= 0),
        reasoning_rate INTEGER CHECK (reasoning_rate IS NULL OR reasoning_rate >= 0),
        state TEXT NOT NULL CHECK (state IN (
            'dispatched', 'completed', 'failed', 'cancelled',
            'incomplete', 'unknown_after_crash'
        )),
        started_at TEXT NOT NULL,
        terminal_at TEXT,
        failure_class TEXT,
        input_tokens INTEGER CHECK (input_tokens IS NULL OR input_tokens >= 0),
        cached_input_tokens INTEGER CHECK (
            cached_input_tokens IS NULL OR cached_input_tokens >= 0
        ),
        output_tokens INTEGER CHECK (output_tokens IS NULL OR output_tokens >= 0),
        reasoning_tokens INTEGER CHECK (reasoning_tokens IS NULL OR reasoning_tokens >= 0),
        usage_source TEXT CHECK (
            usage_source IS NULL OR usage_source IN ('observed', 'estimated', 'unknown')
        ),
        estimated_cost_micro_usd INTEGER CHECK (
            estimated_cost_micro_usd IS NULL OR estimated_cost_micro_usd >= 0
        ),
        content_retained INTEGER NOT NULL DEFAULT 0 CHECK (content_retained = 0),
        UNIQUE (organization_id, attempt_id),
        UNIQUE (request_id, attempt_ordinal),
        FOREIGN KEY (organization_id, request_id)
            REFERENCES gateway_requests (organization_id, request_id)
    ) STRICT
    """,
    """
    INSERT INTO gateway_attempts (
        attempt_id, request_id, organization_id, attempt_ordinal, route_depth,
        deployment_id, provider, exact_model_id, pool_id, catalog_sha256,
        billing_source, pricing_source, pricing_effective_at, route_reason, fallback_reason,
        input_rate, cached_input_rate, output_rate, reasoning_rate, state, started_at,
        terminal_at, failure_class, input_tokens, cached_input_tokens, output_tokens,
        reasoning_tokens, usage_source, estimated_cost_micro_usd, content_retained
    )
    SELECT
        attempt_id, request_id, organization_id,
        ROW_NUMBER() OVER (
            PARTITION BY request_id ORDER BY started_at, attempt_id
        ) - 1,
        route_depth, deployment_id, provider, exact_model_id, pool_id, catalog_sha256,
        billing_source, pricing_source, pricing_effective_at, route_reason, fallback_reason,
        input_rate, cached_input_rate, output_rate, reasoning_rate, state, started_at,
        terminal_at, failure_class, input_tokens, cached_input_tokens, output_tokens,
        reasoning_tokens, usage_source, estimated_cost_micro_usd, content_retained
    FROM gateway_attempts_v5
    """,
    "DROP TABLE gateway_attempts_v5",
    """
    CREATE INDEX gateway_attempts_usage
    ON gateway_attempts (organization_id, terminal_at, state)
    """,
)

_MIGRATION_7 = (
    """
    ALTER TABLE gateway_attempts
    ADD COLUMN budget_period_start TEXT CHECK (
        budget_period_start IS NULL OR (
            length(budget_period_start) = 25
            AND substr(budget_period_start, 8) = '-01T00:00:00+00:00'
        )
    )
    """,
    """
    ALTER TABLE gateway_attempts
    ADD COLUMN budget_reserved_micro_usd INTEGER CHECK (
        budget_reserved_micro_usd IS NULL OR budget_reserved_micro_usd >= 0
    )
    """,
    """
    ALTER TABLE gateway_attempts
    ADD COLUMN budget_settled_micro_usd INTEGER CHECK (
        budget_settled_micro_usd IS NULL OR budget_settled_micro_usd >= 0
    )
    """,
    """
    UPDATE gateway_attempts
    SET budget_period_start = substr(started_at, 1, 7) || '-01T00:00:00+00:00',
        budget_settled_micro_usd = estimated_cost_micro_usd
    """,
    """
    CREATE TRIGGER gateway_attempts_require_budget_period
    BEFORE INSERT ON gateway_attempts
    WHEN NEW.budget_period_start IS NULL
    BEGIN
        SELECT RAISE(ABORT, 'gateway attempt requires a UTC budget period');
    END
    """,
    """
    CREATE TABLE gateway_monthly_budgets (
        budget_id TEXT PRIMARY KEY,
        organization_id TEXT NOT NULL,
        period_start TEXT NOT NULL CHECK (
            length(period_start) = 25
            AND substr(period_start, 8) = '-01T00:00:00+00:00'
        ),
        scope_kind TEXT NOT NULL CHECK (
            scope_kind IN ('team', 'identity', 'pool', 'deployment')
        ),
        scope_key TEXT NOT NULL,
        identity_id TEXT,
        alias_id TEXT,
        pool_id TEXT,
        deployment_id TEXT,
        limit_micro_usd INTEGER NOT NULL CHECK (limit_micro_usd >= 0),
        reserved_micro_usd INTEGER NOT NULL DEFAULT 0 CHECK (reserved_micro_usd >= 0),
        settled_micro_usd INTEGER NOT NULL DEFAULT 0 CHECK (settled_micro_usd >= 0),
        unknown_cost_attempts INTEGER NOT NULL DEFAULT 0 CHECK (unknown_cost_attempts >= 0),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (organization_id, period_start, scope_key),
        CHECK (
            (scope_kind = 'team' AND identity_id IS NULL AND alias_id IS NULL
                AND pool_id IS NULL AND deployment_id IS NULL)
            OR
            (scope_kind = 'identity' AND identity_id IS NOT NULL AND alias_id IS NULL
                AND pool_id IS NULL AND deployment_id IS NULL)
            OR
            (scope_kind = 'pool' AND identity_id IS NULL AND alias_id IS NOT NULL
                AND pool_id IS NOT NULL AND deployment_id IS NULL)
            OR
            (scope_kind = 'deployment' AND identity_id IS NULL AND alias_id IS NOT NULL
                AND pool_id IS NOT NULL AND deployment_id IS NOT NULL)
        ),
        FOREIGN KEY (organization_id) REFERENCES organizations (organization_id),
        FOREIGN KEY (organization_id, identity_id)
            REFERENCES identities (organization_id, identity_id),
        FOREIGN KEY (organization_id, alias_id)
            REFERENCES gateway_aliases (organization_id, alias_id)
    ) STRICT
    """,
    """
    CREATE TABLE gateway_attempt_budget_charges (
        budget_id TEXT NOT NULL,
        attempt_id TEXT NOT NULL,
        reserved_micro_usd INTEGER CHECK (
            reserved_micro_usd IS NULL OR reserved_micro_usd >= 0
        ),
        settled_micro_usd INTEGER CHECK (
            settled_micro_usd IS NULL OR settled_micro_usd >= 0
        ),
        PRIMARY KEY (budget_id, attempt_id),
        FOREIGN KEY (budget_id) REFERENCES gateway_monthly_budgets (budget_id),
        FOREIGN KEY (attempt_id) REFERENCES gateway_attempts (attempt_id)
    ) STRICT
    """,
    """
    CREATE INDEX gateway_monthly_budgets_period
    ON gateway_monthly_budgets (organization_id, period_start, scope_kind)
    """,
    """
    CREATE INDEX gateway_attempts_budget_period
    ON gateway_attempts (organization_id, budget_period_start, pool_id, deployment_id)
    """,
    """
    CREATE INDEX gateway_attempt_budget_charges_attempt
    ON gateway_attempt_budget_charges (attempt_id)
    """,
)

_MIGRATION_8 = (
    """
    ALTER TABLE gateway_monthly_budgets
    ADD COLUMN strict_unknown_cost INTEGER NOT NULL DEFAULT 0 CHECK (
        strict_unknown_cost IN (0, 1)
    )
    """,
)

_MIGRATION_9 = (
    "ALTER TABLE gateway_attempts ADD COLUMN first_token_at TEXT",
    "ALTER TABLE gateway_requests ADD COLUMN app_referer TEXT",
    "ALTER TABLE gateway_requests ADD COLUMN app_title TEXT",
)

# The v10 gateway_requests definition: identical columns, order, and
# constraints to the v9 table, with the api_surface CHECK widened to admit
# the Anthropic Messages surface. Only the CHECK expression changes, so the
# on-disk record format is untouched.
_GATEWAY_REQUESTS_V10_SQL = """CREATE TABLE gateway_requests (
        request_id TEXT PRIMARY KEY,
        organization_id TEXT NOT NULL,
        identity_id TEXT NOT NULL,
        key_id TEXT NOT NULL,
        alias_id TEXT NOT NULL,
        alias_revision_id TEXT NOT NULL,
        api_surface TEXT NOT NULL CHECK (
            api_surface IN ('chat_completions', 'responses', 'messages')
        ),
        canonical_request_sha256 TEXT NOT NULL CHECK (length(canonical_request_sha256) = 64),
        caller_operation_sha256 TEXT,
        accepted_at TEXT NOT NULL,
        deadline_at TEXT NOT NULL,
        terminal_state TEXT CHECK (
            terminal_state IS NULL OR terminal_state IN (
                'completed', 'failed', 'cancelled', 'incomplete',
                'expired_before_dispatch', 'unknown_after_crash'
            )
        ),
        terminal_at TEXT,
        content_retained INTEGER NOT NULL DEFAULT 0 CHECK (content_retained = 0),
        app_referer TEXT,
        app_title TEXT,
        UNIQUE (organization_id, request_id),
        FOREIGN KEY (organization_id, identity_id)
            REFERENCES identities (organization_id, identity_id),
        FOREIGN KEY (organization_id, key_id)
            REFERENCES virtual_keys (organization_id, key_id),
        FOREIGN KEY (organization_id, alias_id, alias_revision_id)
            REFERENCES alias_revisions (organization_id, alias_id, revision_id)
    ) STRICT"""

_MIGRATION_10 = (
    # SQLite cannot alter a CHECK constraint in place, and gateway_requests is
    # the foreign-key parent of gateway_attempts, so the copy-and-rename
    # rebuild used by migration 6 would trip immediate foreign keys inside
    # this exclusive transaction. A CHECK-only change does not affect the
    # on-disk record format, so the documented lightweight procedure rewrites
    # the stored schema text in place instead.
    "PRAGMA writable_schema = ON",
    (
        "UPDATE sqlite_master SET sql = '"
        + _GATEWAY_REQUESTS_V10_SQL.replace("'", "''")
        + "' WHERE type = 'table' AND name = 'gateway_requests'"
    ),
    "PRAGMA writable_schema = RESET",
    # Real DDL bumps the schema cookie so every connection reparses the
    # rewritten definition instead of trusting a cached schema.
    "CREATE TABLE gateway_schema_refresh_v10 (noop INTEGER) STRICT",
    "DROP TABLE gateway_schema_refresh_v10",
)

_MIGRATION_11 = (
    "ALTER TABLE provider_connection_revisions ADD COLUMN azure_api_surface TEXT "
    "CHECK (azure_api_surface IS NULL OR azure_api_surface IN "
    "('openai_deployments', 'model_inference'))",
)

_MIGRATION_12 = (
    "ALTER TABLE provider_connection_revisions ADD COLUMN aws_access_key_id_env TEXT",
    """
    ALTER TABLE provider_connection_revisions
    ADD COLUMN bedrock_auth_mode TEXT
    CHECK (
        bedrock_auth_mode IS NULL
        OR bedrock_auth_mode IN ('access_key_pair', 'api_key')
    )
    """,
)

# Long-context tier rates freeze on the attempt exactly like the base
# rates, so settlement prices with the schedule that was live at dispatch.
# The threshold column selects the schedule once provider-reported input
# tokens reach it; NULL means the deployment had no tier.
_MIGRATION_13 = (
    """
    ALTER TABLE gateway_attempts
    ADD COLUMN long_context_threshold_tokens INTEGER
    CHECK (long_context_threshold_tokens IS NULL OR long_context_threshold_tokens > 0)
    """,
    "ALTER TABLE gateway_attempts ADD COLUMN long_context_input_rate INTEGER",
    "ALTER TABLE gateway_attempts ADD COLUMN long_context_cached_input_rate INTEGER",
    "ALTER TABLE gateway_attempts ADD COLUMN long_context_output_rate INTEGER",
    "ALTER TABLE gateway_attempts ADD COLUMN long_context_reasoning_rate INTEGER",
)

# The v14 gateway_requests definition: the v10 table with the api_surface
# CHECK widened once more to admit the embeddings surface. Migrations 11-13
# touched other tables, so this is otherwise the live definition verbatim.
_GATEWAY_REQUESTS_V14_SQL = _GATEWAY_REQUESTS_V10_SQL.replace(
    "api_surface IN ('chat_completions', 'responses', 'messages')",
    "api_surface IN ('chat_completions', 'responses', 'messages', 'embeddings')",
)

_MIGRATION_14 = (
    # Same CHECK-only in-place rewrite as migration 10 (see its comment).
    "PRAGMA writable_schema = ON",
    (
        "UPDATE sqlite_master SET sql = '"
        + _GATEWAY_REQUESTS_V14_SQL.replace("'", "''")
        + "' WHERE type = 'table' AND name = 'gateway_requests'"
    ),
    "PRAGMA writable_schema = RESET",
    "CREATE TABLE gateway_schema_refresh_v14 (noop INTEGER) STRICT",
    "DROP TABLE gateway_schema_refresh_v14",
)

# v15: the api_surface CHECK admits the images surface (same in-place rewrite).
_GATEWAY_REQUESTS_V15_SQL = _GATEWAY_REQUESTS_V14_SQL.replace(
    "api_surface IN ('chat_completions', 'responses', 'messages', 'embeddings')",
    "api_surface IN ('chat_completions', 'responses', 'messages', 'embeddings', 'images')",
)

_MIGRATION_15 = (
    "PRAGMA writable_schema = ON",
    (
        "UPDATE sqlite_master SET sql = '"
        + _GATEWAY_REQUESTS_V15_SQL.replace("'", "''")
        + "' WHERE type = 'table' AND name = 'gateway_requests'"
    ),
    "PRAGMA writable_schema = RESET",
    "CREATE TABLE gateway_schema_refresh_v15 (noop INTEGER) STRICT",
    "DROP TABLE gateway_schema_refresh_v15",
)

# v16: retain the provider's own sanitized explanation of a failed attempt.
# The Rust upstream already extracts one bounded, single-line, credential- and
# infrastructure-free sentence from a client-error body (param_attribution);
# this column persists that text on the failed attempt so an operator can see
# WHY a provider rejected the call without re-deriving it from logs. It is the
# same sanitized text the caller already receives, so it does not widen the
# ledger's content-free posture.
_MIGRATION_16 = ("ALTER TABLE gateway_attempts ADD COLUMN failure_message TEXT",)

# v17: cost-optimality disclosure for policy-routed dispatches. When a rung
# dispatch policy or an affinity pool bypasses the route's preferred rung,
# dispatch_reason names why the chosen rung serves (affinity, fair_share_shed,
# queue_bound, rung_dead, saturated_overflow) and the preferred_* columns
# freeze the bypassed rung's identity and base token rates at reservation, so
# settle can price the SAME observed usage counterfactually
# (counterfactual_cost_micro_usd) without any content or re-derivation.
_MIGRATION_17 = (
    "ALTER TABLE gateway_attempts ADD COLUMN dispatch_reason TEXT",
    "ALTER TABLE gateway_attempts ADD COLUMN preferred_deployment_id TEXT",
    "ALTER TABLE gateway_attempts ADD COLUMN preferred_input_rate INTEGER",
    "ALTER TABLE gateway_attempts ADD COLUMN preferred_cached_input_rate INTEGER",
    "ALTER TABLE gateway_attempts ADD COLUMN preferred_output_rate INTEGER",
    "ALTER TABLE gateway_attempts ADD COLUMN preferred_reasoning_rate INTEGER",
    "ALTER TABLE gateway_attempts ADD COLUMN counterfactual_cost_micro_usd INTEGER",
)

_MIGRATIONS = {
    1: _MIGRATION_1,
    2: _MIGRATION_2,
    3: _MIGRATION_3,
    4: _MIGRATION_4,
    5: _MIGRATION_5,
    6: _MIGRATION_6,
    7: _MIGRATION_7,
    8: _MIGRATION_8,
    9: _MIGRATION_9,
    10: _MIGRATION_10,
    11: _MIGRATION_11,
    12: _MIGRATION_12,
    13: _MIGRATION_13,
    14: _MIGRATION_14,
    15: _MIGRATION_15,
    16: _MIGRATION_16,
    17: _MIGRATION_17,
}


def connect_database(
    path: Path, *, busy_timeout_ms: int = 5_000, enable_wal: bool = True
) -> sqlite3.Connection:
    """Open one configured SQLite connection with mandatory safety pragmas.

    Args:
        path: Gateway database path.
        busy_timeout_ms: Bounded lock wait in milliseconds.
        enable_wal: Whether to assert the supported journal mode after version checks.

    Returns:
        Configured connection with row-name access.
    """
    connection = sqlite3.connect(path, timeout=busy_timeout_ms / 1_000, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
    if enable_wal:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
    return connection


class _ThreadConnectionCache(threading.local):
    """Per-thread idle SQLite connections keyed by path and busy timeout."""

    def __init__(self) -> None:
        """Start each thread with an empty idle-connection map."""
        self.idle: dict[tuple[str, int], sqlite3.Connection] = {}


_connection_cache = _ThreadConnectionCache()


@contextmanager
def persistent_connection(
    path: Path, *, busy_timeout_ms: int = 5_000
) -> Iterator[sqlite3.Connection]:
    """Yield one reusable per-thread connection for repeated gateway operations.

    Opening a SQLite connection pays file open, pragma, and WAL setup costs on
    every call, which dominates hot request paths. This checkout keeps one idle
    connection per thread, path, and timeout so sequential operations reuse it,
    while overlapping checkouts on the same thread fall back to a fresh
    connection instead of sharing an in-flight transaction.

    Args:
        path: Gateway database path.
        busy_timeout_ms: Bounded lock wait in milliseconds.

    Yields:
        A configured connection; it returns to the idle cache on clean exit.
    """
    key = (str(path), busy_timeout_ms)
    connection = _connection_cache.idle.pop(key, None)
    if connection is None:
        connection = connect_database(path, busy_timeout_ms=busy_timeout_ms)
    try:
        yield connection
    except BaseException:
        connection.close()
        raise
    if connection.in_transaction:
        connection.close()
        return
    previous = _connection_cache.idle.get(key)
    if previous is not None:
        connection.close()
        return
    _connection_cache.idle[key] = connection


def close_idle_connections() -> int:
    """Close and forget the calling thread's cached idle connections.

    A long-lived worker thread that stops servicing gateway operations calls
    this before it exits, so the database descriptors held by its
    ``persistent_connection`` cache release with the worker instead of
    lingering for the life of the interpreter.

    Returns:
        Number of connections closed.
    """
    idle = _connection_cache.idle
    closed = len(idle)
    for connection in idle.values():
        connection.close()
    idle.clear()
    return closed


def initialize_database(path: Path, *, busy_timeout_ms: int = 5_000) -> Path | None:
    """Create or forward-migrate one database without deleting incompatible state.

    Args:
        path: Gateway database path.
        busy_timeout_ms: Bounded lock wait in milliseconds.

    Returns:
        Backup path when an existing older schema was migrated, otherwise ``None``.

    Raises:
        GatewaySchemaError: State is corrupt, newer, or cannot migrate atomically.
    """
    _create_private_database_file(path)
    try:
        connection = connect_database(path, busy_timeout_ms=busy_timeout_ms, enable_wal=False)
    except sqlite3.DatabaseError as exc:
        raise GatewaySchemaError("gateway database is corrupt or unreadable") from exc
    backup: Path | None = None
    try:
        connection.execute("BEGIN EXCLUSIVE")
        try:
            _supported_schema_version(connection)
            connection.execute("COMMIT")
        except GatewaySchemaError:
            connection.execute("ROLLBACK")
            raise
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("BEGIN EXCLUSIVE")
        try:
            version = _supported_schema_version(connection)
            if 0 < version < SCHEMA_VERSION:
                backup = _backup_database(path, version)
            for next_version in range(version + 1, SCHEMA_VERSION + 1):
                for statement in _MIGRATIONS[next_version]:
                    connection.execute(statement)
                connection.execute(f"PRAGMA user_version = {next_version}")
            _require_schema_objects(connection)
            connection.execute("COMMIT")
        except GatewaySchemaError:
            connection.execute("ROLLBACK")
            raise
        except (sqlite3.DatabaseError, OSError) as exc:
            connection.execute("ROLLBACK")
            raise GatewaySchemaError("gateway database migration failed") from exc
    except sqlite3.DatabaseError as exc:
        raise GatewaySchemaError("gateway database is corrupt or unreadable") from exc
    finally:
        connection.close()
    os.chmod(path, 0o600)
    return backup


def _supported_schema_version(connection: sqlite3.Connection) -> int:
    """Read and validate schema state while the caller holds an exclusive transaction.

    Args:
        connection: Database connection inside an exclusive transaction.

    Returns:
        Supported current schema version.

    Raises:
        GatewaySchemaError: Integrity fails or the schema is newer than this code.
    """
    integrity = connection.execute("PRAGMA integrity_check").fetchone()
    if integrity is None or integrity[0] != "ok":
        raise GatewaySchemaError("gateway database failed integrity check")
    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if version > SCHEMA_VERSION:
        raise GatewaySchemaError(
            f"gateway database schema {version} is newer than supported {SCHEMA_VERSION}"
        )
    return version


def _create_private_database_file(path: Path) -> None:
    """Create a missing database as a user-only regular file."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not path.exists():
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise GatewaySchemaError("gateway database path must be a regular file")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise GatewaySchemaError("gateway database must not be group or world accessible")


def _backup_database(path: Path, version: int) -> Path:
    """Create a consistent mode-0600 SQLite backup before forward migration.

    The caller retains the exclusive migration transaction while this separate read
    connection copies the last committed WAL snapshot.

    Args:
        path: Database protected by the caller's exclusive transaction.
        version: Last committed schema version represented by the backup.

    Returns:
        Private path containing the consistent pre-migration snapshot.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup = path.with_name(f"{path.name}.backup-v{version}-{stamp}")
    descriptor = os.open(backup, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    source = sqlite3.connect(path)
    destination = sqlite3.connect(backup)
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()
    os.chmod(backup, 0o600)
    return backup


def _require_schema_objects(connection: sqlite3.Connection) -> None:
    """Refuse a version marker that does not have every required table."""
    expected = {
        "alias_revisions",
        "catalog_snapshot_refs",
        "gateway_aliases",
        "gateway_attempt_budget_charges",
        "gateway_attempts",
        "gateway_monthly_budgets",
        "gateway_requests",
        "identities",
        "identity_alias_grants",
        "operation_receipts",
        "organizations",
        "provider_connection_revisions",
        "provider_connections",
        "project_activation_bindings",
        "alias_revision_provider_connections",
        "virtual_keys",
    }
    rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    actual = {str(row[0]) for row in rows}
    if not expected.issubset(actual):
        raise GatewaySchemaError("gateway database schema marker does not match required tables")
