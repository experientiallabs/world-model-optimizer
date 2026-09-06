"""Content-free usage aggregation models and queries for the attempt ledger.

Mechanical companion to ``ledger.py``: the read-side usage report models and
the bounded SQL aggregations that fill them, executed inside a caller-owned
SQLite read snapshot so identity and billing-source views stay internally
conserving.
"""

from __future__ import annotations

import sqlite3

from exp.common.core.artifacts import ContractModel
from exp.common.models.catalog import BillingSource


class UsageTerminalCount(ContractModel):
    """Count of attempts ending in one normalized terminal state."""

    state: str
    attempts: int


class IdentityUsage(ContractModel):
    """Content-free usage totals for one identity."""

    organization_id: str
    identity_id: str
    requests: int
    attempts: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    known_estimated_cost_micro_usd: int
    unknown_cost_attempts: int
    total_latency_ms: int
    average_latency_ms: float | None
    terminal_counts: tuple[UsageTerminalCount, ...]


class BillingSourceUsage(ContractModel):
    """Content-free physical-attempt totals for one credential ownership source."""

    billing_source: BillingSource
    attempts: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    known_estimated_cost_micro_usd: int
    unknown_cost_attempts: int
    terminal_counts: tuple[UsageTerminalCount, ...]


class LedgerUsageSnapshot(ContractModel):
    """One SQLite read snapshot containing identity and billing-source aggregates."""

    identities: tuple[IdentityUsage, ...]
    by_billing_source: tuple[BillingSourceUsage, ...]


def identity_usage_rows(
    connection: sqlite3.Connection,
    *,
    organization_id: str,
    predicate: str,
    parameters: tuple[str, ...],
) -> tuple[IdentityUsage, ...]:
    """Read bounded identity aggregates inside the caller's SQLite snapshot."""
    rows = connection.execute(
        f"""
        SELECT i.identity_id,
               COUNT(DISTINCT r.request_id) AS requests,
               COUNT(a.attempt_id) AS attempts,
               COALESCE(SUM(a.input_tokens), 0) AS input_tokens,
               COALESCE(SUM(a.cached_input_tokens), 0) AS cached_input_tokens,
               COALESCE(SUM(a.output_tokens), 0) AS output_tokens,
               COALESCE(SUM(a.reasoning_tokens), 0) AS reasoning_tokens,
               COALESCE(SUM(a.estimated_cost_micro_usd), 0) AS known_cost,
               COALESCE(SUM(CASE
                   WHEN a.attempt_id IS NOT NULL
                    AND a.estimated_cost_micro_usd IS NULL THEN 1 ELSE 0 END), 0
               ) AS unknown_cost_attempts,
               COALESCE(SUM(CASE WHEN a.terminal_at IS NOT NULL THEN
                   ROUND((julianday(a.terminal_at) - julianday(a.started_at)) * 86400000)
                   ELSE 0 END), 0) AS total_latency_ms,
               AVG(CASE WHEN a.terminal_at IS NOT NULL THEN
                   (julianday(a.terminal_at) - julianday(a.started_at)) * 86400000
                   ELSE NULL END) AS average_latency_ms
        FROM identities AS i
        LEFT JOIN gateway_requests AS r
          ON r.organization_id = i.organization_id AND r.identity_id = i.identity_id
        LEFT JOIN gateway_attempts AS a ON a.request_id = r.request_id
        WHERE {predicate}
        GROUP BY i.identity_id ORDER BY i.identity_id
        """,
        parameters,
    ).fetchall()
    terminal_rows = connection.execute(
        f"""
        SELECT i.identity_id, a.state, COUNT(*) AS attempts
        FROM identities AS i
        JOIN gateway_requests AS r
          ON r.organization_id = i.organization_id AND r.identity_id = i.identity_id
        JOIN gateway_attempts AS a ON a.request_id = r.request_id
        WHERE {predicate} AND a.state != 'dispatched'
        GROUP BY i.identity_id, a.state ORDER BY i.identity_id, a.state
        """,
        parameters,
    ).fetchall()
    terminals: dict[str, list[UsageTerminalCount]] = {}
    for row in terminal_rows:
        terminals.setdefault(str(row["identity_id"]), []).append(
            UsageTerminalCount(state=str(row["state"]), attempts=int(row["attempts"]))
        )
    return tuple(
        IdentityUsage(
            organization_id=organization_id,
            identity_id=str(row["identity_id"]),
            requests=int(row["requests"]),
            attempts=int(row["attempts"]),
            input_tokens=int(row["input_tokens"]),
            cached_input_tokens=int(row["cached_input_tokens"]),
            output_tokens=int(row["output_tokens"]),
            reasoning_tokens=int(row["reasoning_tokens"]),
            known_estimated_cost_micro_usd=int(row["known_cost"]),
            unknown_cost_attempts=int(row["unknown_cost_attempts"]),
            total_latency_ms=int(row["total_latency_ms"]),
            average_latency_ms=(
                None if row["average_latency_ms"] is None else float(row["average_latency_ms"])
            ),
            terminal_counts=tuple(terminals.get(str(row["identity_id"]), ())),
        )
        for row in rows
    )


def billing_source_usage_rows(
    connection: sqlite3.Connection,
    *,
    predicate: str,
    parameters: tuple[str, ...],
) -> tuple[BillingSourceUsage, ...]:
    """Read bounded source aggregates inside the caller's SQLite snapshot."""
    rows = connection.execute(
        f"""
        SELECT a.billing_source,
               COUNT(a.attempt_id) AS attempts,
               COALESCE(SUM(a.input_tokens), 0) AS input_tokens,
               COALESCE(SUM(a.cached_input_tokens), 0) AS cached_input_tokens,
               COALESCE(SUM(a.output_tokens), 0) AS output_tokens,
               COALESCE(SUM(a.reasoning_tokens), 0) AS reasoning_tokens,
               COALESCE(SUM(a.estimated_cost_micro_usd), 0) AS known_cost,
               COALESCE(SUM(CASE
                   WHEN a.estimated_cost_micro_usd IS NULL THEN 1 ELSE 0 END), 0
               ) AS unknown_cost_attempts
        FROM gateway_attempts AS a
        JOIN gateway_requests AS r ON r.request_id = a.request_id
        WHERE {predicate}
        GROUP BY a.billing_source ORDER BY a.billing_source
        """,
        parameters,
    ).fetchall()
    terminal_rows = connection.execute(
        f"""
        SELECT a.billing_source, a.state, COUNT(*) AS attempts
        FROM gateway_attempts AS a
        JOIN gateway_requests AS r ON r.request_id = a.request_id
        WHERE {predicate} AND a.state != 'dispatched'
        GROUP BY a.billing_source, a.state ORDER BY a.billing_source, a.state
        """,
        parameters,
    ).fetchall()
    terminals: dict[str, list[UsageTerminalCount]] = {}
    for row in terminal_rows:
        terminals.setdefault(str(row["billing_source"]), []).append(
            UsageTerminalCount(state=str(row["state"]), attempts=int(row["attempts"]))
        )
    return tuple(
        BillingSourceUsage(
            billing_source=BillingSource(str(row["billing_source"])),
            attempts=int(row["attempts"]),
            input_tokens=int(row["input_tokens"]),
            cached_input_tokens=int(row["cached_input_tokens"]),
            output_tokens=int(row["output_tokens"]),
            reasoning_tokens=int(row["reasoning_tokens"]),
            known_estimated_cost_micro_usd=int(row["known_cost"]),
            unknown_cost_attempts=int(row["unknown_cost_attempts"]),
            terminal_counts=tuple(terminals.get(str(row["billing_source"]), ())),
        )
        for row in rows
    )
