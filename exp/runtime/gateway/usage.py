"""Content-free usage reporting for CLI JSON and loopback HTML surfaces."""

from __future__ import annotations

from html import escape
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ContractModel
from exp.runtime.gateway.ledger import SQLiteAttemptLedger
from exp.runtime.gateway.ledger_usage import (
    BillingSourceUsage,
    IdentityUsage,
    UsageTerminalCount,
)


class GatewayUsageTotals(ContractModel):
    """Aggregate usage across the selected identities."""

    requests: int = Field(ge=0)
    attempts: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)
    known_estimated_cost_micro_usd: int = Field(ge=0)
    unknown_cost_attempts: int = Field(ge=0)
    total_latency_ms: int = Field(ge=0)
    terminal_counts: tuple[UsageTerminalCount, ...]


class GatewayUsageReport(ContractModel):
    """Versioned aggregate and per-identity attributed usage report."""

    schema_version: Literal[2] = 2
    organization_id: str
    totals: GatewayUsageTotals
    identities: tuple[IdentityUsage, ...]
    by_billing_source: tuple[BillingSourceUsage, ...]
    cost_description: str = "attributed estimated cost, not provider invoice cost"


def read_usage_report(
    ledger: SQLiteAttemptLedger,
    *,
    organization_id: str,
    identity_id: str | None = None,
) -> GatewayUsageReport:
    """Read one stable content-free usage report.

    Args:
        ledger: Shared content-free attempt ledger.
        organization_id: Tenant whose usage is visible.
        identity_id: Optional exact identity filter.

    Returns:
        Versioned aggregate and per-identity usage.
    """
    snapshot = ledger.usage_snapshot(
        organization_id=organization_id,
        identity_id=identity_id,
    )
    identities = snapshot.identities
    by_billing_source = snapshot.by_billing_source
    terminal: dict[str, int] = {}
    for identity in identities:
        for count in identity.terminal_counts:
            terminal[count.state] = terminal.get(count.state, 0) + count.attempts
    totals = GatewayUsageTotals(
        requests=sum(item.requests for item in identities),
        attempts=sum(item.attempts for item in identities),
        input_tokens=sum(item.input_tokens for item in identities),
        cached_input_tokens=sum(item.cached_input_tokens for item in identities),
        output_tokens=sum(item.output_tokens for item in identities),
        reasoning_tokens=sum(item.reasoning_tokens for item in identities),
        known_estimated_cost_micro_usd=sum(
            item.known_estimated_cost_micro_usd for item in identities
        ),
        unknown_cost_attempts=sum(item.unknown_cost_attempts for item in identities),
        total_latency_ms=sum(item.total_latency_ms for item in identities),
        terminal_counts=tuple(
            UsageTerminalCount(state=state, attempts=attempts)
            for state, attempts in sorted(terminal.items())
        ),
    )
    return GatewayUsageReport(
        organization_id=organization_id,
        totals=totals,
        identities=identities,
        by_billing_source=by_billing_source,
    )


def usage_html(report: GatewayUsageReport) -> str:
    """Render a minimal loopback-only usage page with no content or secret fields.

    Args:
        report: Content-free usage report.

    Returns:
        Complete standalone HTML document.
    """
    rows = "".join(
        "<tr>"
        f"<td>{escape(item.identity_id)}</td>"
        f"<td>{item.requests}</td>"
        f"<td>{item.attempts}</td>"
        f"<td>{item.input_tokens}</td>"
        f"<td>{item.cached_input_tokens}</td>"
        f"<td>{item.output_tokens}</td>"
        f"<td>{item.reasoning_tokens}</td>"
        f"<td>{item.known_estimated_cost_micro_usd}</td>"
        f"<td>{item.unknown_cost_attempts}</td>"
        f"<td>{item.total_latency_ms}</td>"
        f"<td>{escape(_terminal_summary(item))}</td>"
        "</tr>"
        for item in report.identities
    )
    source_rows = "".join(
        "<tr>"
        f"<td>{escape(item.billing_source.value)}</td>"
        f"<td>{item.attempts}</td>"
        f"<td>{item.input_tokens}</td>"
        f"<td>{item.cached_input_tokens}</td>"
        f"<td>{item.output_tokens}</td>"
        f"<td>{item.reasoning_tokens}</td>"
        f"<td>{item.known_estimated_cost_micro_usd}</td>"
        f"<td>{item.unknown_cost_attempts}</td>"
        f"<td>{escape(_source_terminal_summary(item))}</td>"
        "</tr>"
        for item in report.by_billing_source
    )
    totals = report.totals
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>EXP gateway usage</title>"
        "<style>body{font:14px system-ui;margin:48px;color:#0a0a0a}"
        "table{border-collapse:collapse;width:100%;max-width:960px}"
        "th,td{text-align:left;border-bottom:1px solid #ececec;padding:10px 8px}"
        "small{color:#666}</style></head><body>"
        "<h1>Gateway usage</h1>"
        f"<p>{totals.requests} requests, {totals.attempts} attempts, "
        f"{totals.known_estimated_cost_micro_usd} micro-USD attributed estimated cost.</p>"
        "<small>Attributed estimates are not provider invoice cost. Unknown cost remains "
        "unknown.</small>"
        "<table><thead><tr><th>Identity</th><th>Requests</th><th>Attempts</th>"
        "<th>Input tokens</th><th>Cached input</th><th>Output tokens</th>"
        "<th>Reasoning tokens</th><th>Known micro-USD</th>"
        "<th>Unknown-cost attempts</th><th>Total latency ms</th>"
        "<th>Terminal states</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        "<h2>Attempts by billing source</h2>"
        "<table><thead><tr><th>Billing source</th><th>Attempts</th>"
        "<th>Input tokens</th><th>Cached input</th><th>Output tokens</th>"
        "<th>Reasoning tokens</th><th>Known micro-USD</th>"
        "<th>Unknown-cost attempts</th><th>Terminal states</th></tr></thead>"
        f"<tbody>{source_rows}</tbody></table></body></html>"
    )


def _terminal_summary(usage: IdentityUsage) -> str:
    """Return stable terminal-state counts for one HTML table cell."""
    if not usage.terminal_counts:
        return "none"
    return ", ".join(f"{count.state}: {count.attempts}" for count in usage.terminal_counts)


def _source_terminal_summary(usage: BillingSourceUsage) -> str:
    """Return stable terminal-state counts for one billing-source table cell."""
    if not usage.terminal_counts:
        return "none"
    return ", ".join(f"{count.state}: {count.attempts}" for count in usage.terminal_counts)
