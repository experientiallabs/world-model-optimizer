"""Tests for monthly hard limits, reservation concurrency, and UTC rollover."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import canonical_json_bytes
from exp.common.models import ModelCapabilities
from exp.common.models.catalog import (
    GatewayDeploymentMetadata,
    GatewayEquivalenceCertification,
    GatewayTokenPrices,
)
from exp.common.models.gateway_catalog import (
    ExactModelDeployment,
    ExactModelPool,
    NormalizedGatewayCatalog,
)
from exp.runtime.gateway.attempt_tokens import (
    DEFAULT_RESERVATION_OUTPUT_TOKENS,
    worst_case_attempt_tokens,
)
from exp.runtime.gateway.budgets import (
    BudgetReservationRejected,
    BudgetScope,
    BudgetScopeKind,
    SQLiteBudgetStore,
    maximum_attempt_cost_micro_usd,
)
from exp.runtime.gateway.contracts import (
    DirectTarget,
    ExecutionSnapshot,
    GatewayApiSurface,
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayMessage,
    GatewayRequest,
    GatewayUsage,
)
from exp.runtime.gateway.embeddings_contracts import EmbeddingsRequest
from exp.runtime.gateway.images_contracts import ImagesRequest
from exp.runtime.gateway.ledger import SQLiteAttemptLedger
from exp.runtime.gateway.sqlite.store import SQLiteGatewayStore


def _catalog() -> NormalizedGatewayCatalog:
    """Build the normalized catalog pinned by the fixture alias revision."""
    return NormalizedGatewayCatalog(
        deployments=(_deployment(), _deployment(deployment_id="secondary")),
        pools=(
            ExactModelPool(
                pool_id="pool",
                exact_model_id="exact-one",
                deployment_ids=("primary", "secondary"),
                equivalence=GatewayEquivalenceCertification(
                    certification_id="cert-one",
                    provenance="operator replayed both deployments",
                    evidence_sha256="e" * 64,
                    certified_at=datetime(2026, 8, 1, tzinfo=UTC),
                ),
            ),
        ),
    )


class _Clock:
    """Controllable aware wall and monotonic clock."""

    def __init__(self) -> None:
        """Start near the end of one UTC month."""
        self.wall = datetime(2026, 8, 31, 23, 59, tzinfo=UTC)
        self.monotonic_value = 100.0

    def now(self) -> datetime:
        """Return controlled wall time."""
        return self.wall

    def monotonic(self) -> float:
        """Return controlled monotonic time."""
        return self.monotonic_value

    def advance(self, duration: timedelta) -> None:
        """Advance both clocks by one duration."""
        self.wall += duration
        self.monotonic_value += duration.total_seconds()


def _deployment(*, priced: bool = True, deployment_id: str = "primary") -> ExactModelDeployment:
    """Return one exact deployment with optional hard-limit pricing."""
    prices = (
        GatewayTokenPrices(
            input_micro_usd_per_million_tokens=1_000_000,
            output_micro_usd_per_million_tokens=2_000_000,
        )
        if priced
        else GatewayTokenPrices()
    )
    return ExactModelDeployment(
        deployment_id=deployment_id,
        source_alias=deployment_id,
        exact_model_id="exact-one",
        connection=f"connection-{deployment_id}",
        provider="openai-compatible",
        provider_model="provider-model",
        connection_sha256=("b" if deployment_id == "primary" else "c") * 64,
        capabilities_sha256="d" * 64,
        capabilities=ModelCapabilities(maximum_output_tokens=16),
        gateway=GatewayDeploymentMetadata(
            prices=prices,
            pricing_source="test",
        ),
    )


def _request(content: str) -> GatewayRequest:
    """Build one bounded request whose content is never persisted."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content=content),),
        maximum_output_tokens=16,
    )


def _authority(
    tmp_path: Path,
    clock: _Clock,
) -> tuple[SQLiteGatewayStore, SQLiteAttemptLedger, SQLiteBudgetStore, str]:
    """Create real SQLite authority, ledger, budget store, and one granted key."""
    path = tmp_path / "gateway.db"
    store = SQLiteGatewayStore(path, clock=clock)
    ledger = SQLiteAttemptLedger(path, clock=clock)
    budgets = SQLiteBudgetStore(path, clock=clock)
    store.create_organization(organization_id="org", slug="org", display_name="Org")
    store.create_identity(organization_id="org", identity_id="identity", display_name="Identity")
    digest = _catalog().identity_sha256()
    store.register_catalog_snapshot(
        organization_id="org",
        snapshot_ref="snapshot",
        catalog_sha256=digest,
    )
    store.activate_alias_revision(
        organization_id="org",
        alias_id="coding",
        alias_name="coding",
        revision_id="revision",
        target=DirectTarget(pool_id="pool"),
        snapshot_ref="snapshot",
        catalog_sha256=digest,
    )
    store.grant_alias(organization_id="org", identity_id="identity", alias_id="coding")
    key = store.issue_virtual_key(
        organization_id="org",
        identity_id="identity",
        key_id="key",
    ).raw_key
    return store, ledger, budgets, key


def _accepted(
    store: SQLiteGatewayStore,
    ledger: SQLiteAttemptLedger,
    clock: _Clock,
    key: str,
    content: str,
) -> tuple[GatewayRequest, ExecutionSnapshot]:
    """Authorize and durably accept one unique request."""
    request = _request(content)
    authorization = store.authorize_request(
        raw_key=key,
        alias="coding",
        request=request,
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=authorization)
    return request, ExecutionSnapshot(
        authorization=authorization,
        exact_model_id="exact-one",
        pool_id="pool",
        deployment_ids=("primary", "secondary"),
    )


def test_maximum_attempt_cost_is_integer_conservative_and_unknown_prices_fail_closed() -> None:
    """Reservation pricing uses canonical bytes, output ceiling, and no float money."""
    request = _request("four bytes")
    known = maximum_attempt_cost_micro_usd(request, _deployment())

    assert known is not None and isinstance(known, int) and known > 32
    assert maximum_attempt_cost_micro_usd(request, _deployment(priced=False)) is None
    unrepresentable = _deployment().model_copy(
        update={
            "gateway": _deployment().gateway.model_copy(
                update={
                    "prices": GatewayTokenPrices(
                        input_micro_usd_per_million_tokens=10**30,
                        output_micro_usd_per_million_tokens=10**30,
                    )
                }
            )
        }
    )
    assert maximum_attempt_cost_micro_usd(request, unrepresentable) is None
    all_prices = _deployment().model_copy(
        update={
            "gateway": _deployment().gateway.model_copy(
                update={
                    "prices": GatewayTokenPrices(
                        input_micro_usd_per_million_tokens=1_000_000,
                        cached_input_micro_usd_per_million_tokens=3_000_000,
                        output_micro_usd_per_million_tokens=2_000_000,
                        reasoning_micro_usd_per_million_tokens=4_000_000,
                    )
                }
            )
        }
    )
    assert all_prices.gateway.capabilities.reports_cached_input_tokens is False
    assert all_prices.gateway.capabilities.reports_reasoning_tokens is False
    all_dimensions = maximum_attempt_cost_micro_usd(request, all_prices)
    assert all_dimensions is not None and all_dimensions > known


def test_missing_output_ceiling_reserves_against_the_default_instead_of_failing_closed() -> None:
    """A priced route with no output ceiling anywhere stays priceable.

    An output bound is a token count, not a price, so its absence must not
    unprice the route: when the caller omits ``maximum_output_tokens`` and the
    deployment declares none, the reservation uses
    ``DEFAULT_RESERVATION_OUTPUT_TOKENS``, bounded by a smaller declared context
    window. Unknown required prices still fail closed.
    """
    request = _request("four bytes").model_copy(update={"maximum_output_tokens": None})
    no_ceiling = _deployment().model_copy(update={"capabilities": ModelCapabilities()})

    default_bound = maximum_attempt_cost_micro_usd(request, no_ceiling)
    assert default_bound is not None and isinstance(default_bound, int) and default_bound > 0
    explicit_default = maximum_attempt_cost_micro_usd(
        request.model_copy(update={"maximum_output_tokens": DEFAULT_RESERVATION_OUTPUT_TOKENS}),
        no_ceiling,
    )
    # The fallback reserves exactly the default when nothing bounds it lower.
    assert explicit_default is not None
    assert abs(default_bound - explicit_default) < 100

    small_window = _deployment().model_copy(
        update={"capabilities": ModelCapabilities(context_window_tokens=1_024)}
    )
    windowed = maximum_attempt_cost_micro_usd(request, small_window)
    explicit_window = maximum_attempt_cost_micro_usd(
        request.model_copy(update={"maximum_output_tokens": 1_024}),
        small_window,
    )
    # A smaller declared context window bounds the default.
    assert windowed is not None and explicit_window is not None
    assert abs(windowed - explicit_window) < 100

    # A missing required price still fails closed regardless of the default.
    assert (
        maximum_attempt_cost_micro_usd(
            request,
            _deployment(priced=False).model_copy(update={"capabilities": ModelCapabilities()}),
        )
        is None
    )


def test_huge_caller_output_ceiling_clamps_to_deployment_instead_of_failing_closed() -> None:
    """A caller output ceiling above the deployment's own is clamped, not None.

    Without the clamp a very large caller ``maximum_output_tokens`` inflates the
    worst case past ``MAXIMUM_MICRO_USD`` and returns ``None``, which fails a
    perfectly fundable request closed and mis-terminalizes it as a quota refusal.
    The deployment can never emit more than its own ceiling, so the output term
    clamps to it: the huge request stays a small bounded int, differing from the
    at-ceiling request only by the extra JSON bytes of the larger literal.
    """
    deployment = _deployment()  # ModelCapabilities(maximum_output_tokens=16)
    bounded = maximum_attempt_cost_micro_usd(_request("four bytes"), deployment)
    huge = maximum_attempt_cost_micro_usd(
        _request("four bytes").model_copy(update={"maximum_output_tokens": 10**12}),
        deployment,
    )

    assert bounded is not None
    assert huge is not None and isinstance(huge, int)
    # Output is clamped to the ceiling; only the input-byte count of the larger
    # literal differs, so the two stay within a handful of micro-USD of each other.
    assert abs(huge - bounded) < 100


def test_concurrent_identity_reservations_never_exceed_hard_limit(tmp_path: Path) -> None:
    """BEGIN IMMEDIATE serializes competing reservations before attempt insertion."""
    clock = _Clock()
    store, ledger, budgets, key = _authority(tmp_path, clock)
    budgets.set_limit(
        organization_id="org",
        period="2026-08",
        scope=BudgetScope(kind=BudgetScopeKind.IDENTITY, identity_id="identity"),
        limit_micro_usd=500,
    )
    snapshots = [
        _accepted(store, ledger, clock, key, f"concurrent-{index}")[1] for index in range(10)
    ]

    def reserve(index: int) -> str:
        """Attempt one competing fixed-size reservation."""
        return ledger.start_attempt(
            snapshot=snapshots[index],
            deployment=_deployment(),
            attempt_ordinal=0,
            route_depth=0,
            maximum_cost_micro_usd=100,
        )

    accepted: list[str] = []
    rejected = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(reserve, index) for index in range(10)]
        for future in futures:
            try:
                accepted.append(future.result())
            except BudgetReservationRejected:
                rejected += 1

    assert len(accepted) == 5
    assert rejected == 5
    remaining = budgets.remaining(organization_id="org", period="2026-08")[0]
    assert remaining.reserved_micro_usd == 500
    assert remaining.remaining_micro_usd == 0


def test_configured_optional_prices_are_reserved_even_when_reporting_hints_are_false(
    tmp_path: Path,
) -> None:
    """Provider usage cannot settle cached or reasoning cost above its reservation."""
    clock = _Clock()
    store, ledger, budgets, key = _authority(tmp_path, clock)
    deployment = _deployment().model_copy(
        update={
            "gateway": _deployment().gateway.model_copy(
                update={
                    "prices": GatewayTokenPrices(
                        input_micro_usd_per_million_tokens=1_000_000,
                        cached_input_micro_usd_per_million_tokens=3_000_000,
                        output_micro_usd_per_million_tokens=2_000_000,
                        reasoning_micro_usd_per_million_tokens=4_000_000,
                    )
                }
            )
        }
    )
    request, snapshot = _accepted(store, ledger, clock, key, "all-price-dimensions")
    maximum = maximum_attempt_cost_micro_usd(request, deployment)
    assert maximum is not None
    budgets.set_limit(
        organization_id="org",
        period="2026-08",
        scope=BudgetScope(kind=BudgetScopeKind.TEAM),
        limit_micro_usd=maximum,
    )
    attempt = ledger.start_attempt(
        snapshot=snapshot,
        deployment=deployment,
        attempt_ordinal=0,
        route_depth=0,
        maximum_cost_micro_usd=maximum,
    )
    input_ceiling = len(canonical_json_bytes(request))
    ledger.finish_attempt(
        attempt_id=attempt,
        terminal_event=GatewayEvent(
            kind=GatewayEventKind.COMPLETED,
            sequence_number=0,
            usage=GatewayUsage(
                input_tokens=input_ceiling,
                cached_input_tokens=input_ceiling,
                output_tokens=16,
                reasoning_tokens=16,
            ),
        ),
        failure=None,
    )

    remaining = budgets.remaining(organization_id="org", period="2026-08")[0]
    assert remaining.reserved_micro_usd == 0
    assert remaining.settled_micro_usd == maximum
    assert remaining.remaining_micro_usd == 0


def test_settlement_counts_failures_retries_and_rollover_without_reset(tmp_path: Path) -> None:
    """Every physical attempt settles in its start month and old buckets remain unchanged."""
    clock = _Clock()
    store, ledger, budgets, key = _authority(tmp_path, clock)
    budgets.set_limit(
        organization_id="org",
        period="2026-08",
        scope=BudgetScope(kind=BudgetScopeKind.TEAM),
        limit_micro_usd=1_000,
    )
    _request_one, snapshot_one = _accepted(store, ledger, clock, key, "failed-attempt")
    failed = ledger.start_attempt(
        snapshot=snapshot_one,
        deployment=_deployment(),
        attempt_ordinal=0,
        route_depth=0,
        maximum_cost_micro_usd=300,
    )
    ledger.finish_attempt(
        attempt_id=failed,
        terminal_event=None,
        failure=GatewayFailure(
            failure_class=GatewayFailureClass.TRANSPORT,
            safe_message="provider transport failed",
        ),
    )

    # A failure without usage can still be billable, so the maximum reservation remains charged.
    august = budgets.remaining(organization_id="org", period="2026-08")[0]
    assert august.charged_micro_usd == 300
    clock.advance(timedelta(minutes=2))
    budgets.set_limit(
        organization_id="org",
        period="2026-09",
        scope=BudgetScope(kind=BudgetScopeKind.TEAM),
        limit_micro_usd=1_000,
    )
    _request_two, snapshot_two = _accepted(store, ledger, clock, key, "completed-attempt")
    completed = ledger.start_attempt(
        snapshot=snapshot_two,
        deployment=_deployment(),
        attempt_ordinal=0,
        route_depth=0,
        maximum_cost_micro_usd=300,
    )
    ledger.finish_attempt(
        attempt_id=completed,
        terminal_event=GatewayEvent(
            kind=GatewayEventKind.COMPLETED,
            sequence_number=0,
            usage=GatewayUsage(input_tokens=10, output_tokens=5),
        ),
        failure=None,
    )

    assert budgets.remaining(organization_id="org", period="2026-08")[0].charged_micro_usd == 300
    september = budgets.remaining(organization_id="org", period="2026-09")[0]
    assert september.reserved_micro_usd == 0
    assert september.settled_micro_usd == 20
    assert september.remaining_micro_usd == 980


def test_default_limit_reports_unknown_cost_without_blocking_service(tmp_path: Path) -> None:
    """A default limit keeps serving after an unknown-cost attempt and reports its volume."""
    clock = _Clock()
    store, ledger, budgets, key = _authority(tmp_path, clock)
    _first_request, first_snapshot = _accepted(store, ledger, clock, key, "unknown-first")
    attempt = ledger.start_attempt(
        snapshot=first_snapshot,
        deployment=_deployment(priced=False),
        attempt_ordinal=0,
        route_depth=0,
        maximum_cost_micro_usd=None,
    )
    ledger.finish_attempt(
        attempt_id=attempt,
        terminal_event=None,
        failure=GatewayFailure(
            failure_class=GatewayFailureClass.TRANSPORT,
            safe_message="provider transport failed",
        ),
    )
    budgets.set_limit(
        organization_id="org",
        period="2026-08",
        scope=BudgetScope(kind=BudgetScopeKind.TEAM),
        limit_micro_usd=1_000,
    )
    _second_request, second_snapshot = _accepted(store, ledger, clock, key, "unknown-second")

    admitted = ledger.start_attempt(
        snapshot=second_snapshot,
        deployment=_deployment(),
        attempt_ordinal=0,
        route_depth=0,
        maximum_cost_micro_usd=100,
    )

    assert admitted
    remaining = budgets.remaining(organization_id="org", period="2026-08")[0]
    assert remaining.unknown_cost_attempts == 1
    assert remaining.remaining_micro_usd == 1_000 - 100
    assert not remaining.exhausted


def test_default_limit_admits_unpriced_attempts_and_tracks_token_volume(
    tmp_path: Path,
) -> None:
    """An unpriced attempt under a default limit runs and reports observed token volume."""
    clock = _Clock()
    store, ledger, budgets, key = _authority(tmp_path, clock)
    budgets.set_limit(
        organization_id="org",
        period="2026-08",
        scope=BudgetScope(kind=BudgetScopeKind.TEAM),
        limit_micro_usd=1_000,
    )
    _request_value, snapshot = _accepted(store, ledger, clock, key, "unpriced-served")

    attempt = ledger.start_attempt(
        snapshot=snapshot,
        deployment=_deployment(priced=False),
        attempt_ordinal=0,
        route_depth=0,
        maximum_cost_micro_usd=None,
    )
    ledger.finish_attempt(
        attempt_id=attempt,
        terminal_event=GatewayEvent(
            kind=GatewayEventKind.COMPLETED,
            sequence_number=0,
            usage=GatewayUsage(input_tokens=120, output_tokens=16, reasoning_tokens=4),
        ),
        failure=None,
    )

    remaining = budgets.remaining(organization_id="org", period="2026-08")[0]
    assert remaining.unknown_cost_attempts == 1
    assert remaining.unknown_cost_input_tokens == 120
    # Reasoning is a subset of output_tokens, so the volume gauge reports the
    # output total once rather than adding the subset a second time.
    assert remaining.unknown_cost_output_tokens == 16
    assert remaining.remaining_micro_usd == 1_000
    assert not remaining.exhausted


def test_strict_limit_fails_closed_on_unknown_cost(tmp_path: Path) -> None:
    """A strict limit rejects unpriced attempts and blocks while unknown cost remains."""
    clock = _Clock()
    store, ledger, budgets, key = _authority(tmp_path, clock)
    first = _unknown_attempt(store, ledger, clock, key, "strict-unknown")
    assert first
    budgets.set_limit(
        organization_id="org",
        period="2026-08",
        scope=BudgetScope(kind=BudgetScopeKind.TEAM),
        limit_micro_usd=1_000,
        strict_unknown_cost=True,
    )
    _blocked_request, blocked_snapshot = _accepted(store, ledger, clock, key, "strict-blocked")

    with pytest.raises(BudgetReservationRejected, match="prior attempts with unknown cost"):
        ledger.start_attempt(
            snapshot=blocked_snapshot,
            deployment=_deployment(),
            attempt_ordinal=0,
            route_depth=0,
            maximum_cost_micro_usd=100,
        )
    remaining = budgets.remaining(organization_id="org", period="2026-08")[0]
    assert remaining.budget.strict_unknown_cost
    assert remaining.unknown_cost_attempts == 1
    assert remaining.remaining_micro_usd == 0
    assert remaining.exhausted


def test_strict_limit_rejects_new_unpriced_attempts(tmp_path: Path) -> None:
    """A strict limit refuses to admit an attempt whose maximum cost is unknown."""
    clock = _Clock()
    store, ledger, budgets, key = _authority(tmp_path, clock)
    budgets.set_limit(
        organization_id="org",
        period="2026-08",
        scope=BudgetScope(kind=BudgetScopeKind.TEAM),
        limit_micro_usd=1_000,
        strict_unknown_cost=True,
    )
    _request_value, snapshot = _accepted(store, ledger, clock, key, "strict-unpriced")

    with pytest.raises(BudgetReservationRejected, match="known maximum attempt cost"):
        ledger.start_attempt(
            snapshot=snapshot,
            deployment=_deployment(priced=False),
            attempt_ordinal=0,
            route_depth=0,
            maximum_cost_micro_usd=None,
        )


def _unknown_attempt(
    store: SQLiteGatewayStore,
    ledger: SQLiteAttemptLedger,
    clock: _Clock,
    key: str,
    content: str,
) -> str:
    """Run one failed unpriced attempt whose cost is never learned."""
    _request_value, snapshot = _accepted(store, ledger, clock, key, content)
    attempt = ledger.start_attempt(
        snapshot=snapshot,
        deployment=_deployment(priced=False),
        attempt_ordinal=0,
        route_depth=0,
        maximum_cost_micro_usd=None,
    )
    ledger.finish_attempt(
        attempt_id=attempt,
        terminal_event=None,
        failure=GatewayFailure(
            failure_class=GatewayFailureClass.TRANSPORT,
            safe_message="provider transport failed",
        ),
    )
    return attempt


def test_reconcile_unknown_costs_restores_service_with_exact_attribution(
    tmp_path: Path,
) -> None:
    """Reconciliation settles each unknown attempt at the assigned cost and reopens the limit."""
    clock = _Clock()
    store, ledger, budgets, key = _authority(tmp_path, clock)
    scope = BudgetScope(kind=BudgetScopeKind.TEAM)
    first = _unknown_attempt(store, ledger, clock, key, "unknown-one")
    second = _unknown_attempt(store, ledger, clock, key, "unknown-two")
    budgets.set_limit(
        organization_id="org",
        period="2026-08",
        scope=scope,
        limit_micro_usd=1_000,
        strict_unknown_cost=True,
    )
    budgets.set_limit(
        organization_id="org",
        period="2026-08",
        scope=scope,
        limit_micro_usd=10_000,
        replace=True,
        strict_unknown_cost=True,
    )
    _blocked_request, blocked_snapshot = _accepted(store, ledger, clock, key, "still-blocked")
    with pytest.raises(BudgetReservationRejected, match="prior attempts with unknown cost"):
        ledger.start_attempt(
            snapshot=blocked_snapshot,
            deployment=_deployment(),
            attempt_ordinal=0,
            route_depth=0,
            maximum_cost_micro_usd=100,
        )

    reconciled, remaining = budgets.reconcile_unknown_costs(
        organization_id="org",
        period="2026-08",
        scope=scope,
        assigned_cost_micro_usd=40,
    )

    assert reconciled == 2
    assert remaining.unknown_cost_attempts == 0
    assert remaining.settled_micro_usd == 80
    assert remaining.remaining_micro_usd == 10_000 - 80
    assert not remaining.exhausted
    with sqlite3.connect(tmp_path / "gateway.db") as connection:
        connection.row_factory = sqlite3.Row
        charges = {
            str(row["attempt_id"]): int(row["settled_micro_usd"])
            for row in connection.execute(
                """
                SELECT attempt_id, settled_micro_usd FROM gateway_attempt_budget_charges
                WHERE budget_id = ?
                """,
                (remaining.budget.budget_id,),
            )
        }
    assert charges == {first: 40, second: 40}
    _next_request, next_snapshot = _accepted(store, ledger, clock, key, "restored")
    admitted = ledger.start_attempt(
        snapshot=next_snapshot,
        deployment=_deployment(),
        attempt_ordinal=0,
        route_depth=0,
        maximum_cost_micro_usd=100,
    )
    assert admitted
    again, unchanged = budgets.reconcile_unknown_costs(
        organization_id="org",
        period="2026-08",
        scope=scope,
        assigned_cost_micro_usd=40,
    )
    assert again == 0
    assert unchanged.settled_micro_usd == 80


def test_reconcile_requires_an_existing_limit(tmp_path: Path) -> None:
    """Reconciliation refuses to invent a budget for a scope without a stored limit."""
    clock = _Clock()
    _store, _ledger, budgets, _key = _authority(tmp_path, clock)

    with pytest.raises(ValueError, match="does not exist"):
        budgets.reconcile_unknown_costs(
            organization_id="org",
            period="2026-08",
            scope=BudgetScope(kind=BudgetScopeKind.TEAM),
            assigned_cost_micro_usd=1,
        )


def test_limit_created_midflight_adopts_and_settles_existing_reservation(
    tmp_path: Path,
) -> None:
    """A new limit binds an active attempt and later settles it exactly once."""
    clock = _Clock()
    store, ledger, budgets, key = _authority(tmp_path, clock)
    _request_value, snapshot = _accepted(store, ledger, clock, key, "midflight")
    attempt = ledger.start_attempt(
        snapshot=snapshot,
        deployment=_deployment(),
        attempt_ordinal=0,
        route_depth=0,
        maximum_cost_micro_usd=300,
    )

    budgets.set_limit(
        organization_id="org",
        period="2026-08",
        scope=BudgetScope(kind=BudgetScopeKind.TEAM),
        limit_micro_usd=1_000,
    )
    active = budgets.remaining(organization_id="org", period="2026-08")[0]
    assert active.reserved_micro_usd == 300
    assert active.settled_micro_usd == 0

    terminal = GatewayEvent(
        kind=GatewayEventKind.COMPLETED,
        sequence_number=0,
        usage=GatewayUsage(input_tokens=10, output_tokens=5),
    )
    ledger.finish_attempt(attempt_id=attempt, terminal_event=terminal, failure=None)
    ledger.finish_attempt(attempt_id=attempt, terminal_event=terminal, failure=None)

    settled = budgets.remaining(organization_id="org", period="2026-08")[0]
    assert settled.reserved_micro_usd == 0
    assert settled.settled_micro_usd == 20
    assert settled.remaining_micro_usd == 980


def test_crash_recovery_retains_reservation_and_idempotent_settlement_charges_once(
    tmp_path: Path,
) -> None:
    """Unknown dispatches keep their maximum while repeated settlement cannot double-charge."""
    clock = _Clock()
    store, ledger, budgets, key = _authority(tmp_path, clock)
    budgets.set_limit(
        organization_id="org",
        period="2026-08",
        scope=BudgetScope(kind=BudgetScopeKind.TEAM),
        limit_micro_usd=500,
    )
    _request_one, first_snapshot = _accepted(store, ledger, clock, key, "crash")
    ledger.start_attempt(
        snapshot=first_snapshot,
        deployment=_deployment(),
        attempt_ordinal=0,
        route_depth=0,
        maximum_cost_micro_usd=100,
    )
    clock.advance(timedelta(seconds=31))
    assert ledger.reconcile_crashed_requests(cleanup_grace=timedelta(0)) == (0, 1)
    after_crash = budgets.remaining(organization_id="org", period="2026-08")[0]
    assert after_crash.reserved_micro_usd == 100
    assert after_crash.remaining_micro_usd == 400

    _request_two, second_snapshot = _accepted(store, ledger, clock, key, "settle-once")
    settled = ledger.start_attempt(
        snapshot=second_snapshot,
        deployment=_deployment(),
        attempt_ordinal=0,
        route_depth=0,
        maximum_cost_micro_usd=200,
    )
    terminal = GatewayEvent(
        kind=GatewayEventKind.COMPLETED,
        sequence_number=0,
        usage=GatewayUsage(input_tokens=10, output_tokens=5),
    )
    ledger.finish_attempt(attempt_id=settled, terminal_event=terminal, failure=None)
    ledger.finish_attempt(attempt_id=settled, terminal_event=terminal, failure=None)

    remaining = budgets.remaining(organization_id="org", period="2026-08")[0]
    assert remaining.charged_micro_usd == 120
    assert remaining.reserved_micro_usd == 100
    assert remaining.settled_micro_usd == 20
    assert remaining.remaining_micro_usd == 380


def test_migrated_attempts_receive_explicit_period_and_cost_state(tmp_path: Path) -> None:
    """The current schema stores no floating monetary fields or resettable counters."""
    clock = _Clock()
    store, ledger, _budgets, key = _authority(tmp_path, clock)
    _request_value, snapshot = _accepted(store, ledger, clock, key, "schema")
    attempt = ledger.start_attempt(
        snapshot=snapshot,
        deployment=_deployment(),
        attempt_ordinal=0,
        route_depth=0,
        maximum_cost_micro_usd=123,
    )
    connection = sqlite3.connect(ledger.database_path)
    try:
        row = connection.execute(
            """
            SELECT budget_period_start, budget_reserved_micro_usd,
                   budget_settled_micro_usd
            FROM gateway_attempts WHERE attempt_id = ?
            """,
            (attempt,),
        ).fetchone()
    finally:
        connection.close()
    assert row == ("2026-08-01T00:00:00+00:00", 123, None)


def _write_snapshot(tmp_path: Path) -> None:
    """Write the pinned normalized catalog snapshot referenced by the fixture alias."""
    (tmp_path / "snapshot").write_bytes(canonical_json_bytes(_catalog().model_dump(mode="json")))


def test_pool_and_deployment_budget_scopes_require_real_targets(tmp_path: Path) -> None:
    """Pool and deployment limits reject identifiers the ledger can never charge."""
    clock = _Clock()
    _store, _ledger, budgets, _key = _authority(tmp_path, clock)
    _write_snapshot(tmp_path)

    with pytest.raises(ValueError, match="pool is not the active revision target"):
        budgets.set_limit(
            organization_id="org",
            period="2026-08",
            scope=BudgetScope(kind=BudgetScopeKind.POOL, alias_id="coding", pool_id="ghost"),
            limit_micro_usd=100,
        )
    with pytest.raises(ValueError, match="deployment is not in its pool"):
        budgets.set_limit(
            organization_id="org",
            period="2026-08",
            scope=BudgetScope(
                kind=BudgetScopeKind.DEPLOYMENT,
                alias_id="coding",
                pool_id="pool",
                deployment_id="ghost",
            ),
            limit_micro_usd=100,
        )
    changed, pool_limit = budgets.set_limit(
        organization_id="org",
        period="2026-08",
        scope=BudgetScope(kind=BudgetScopeKind.POOL, alias_id="coding", pool_id="pool"),
        limit_micro_usd=100,
    )
    assert changed and pool_limit.scope.pool_id == "pool"
    changed, deployment_limit = budgets.set_limit(
        organization_id="org",
        period="2026-08",
        scope=BudgetScope(
            kind=BudgetScopeKind.DEPLOYMENT,
            alias_id="coding",
            pool_id="pool",
            deployment_id="secondary",
        ),
        limit_micro_usd=100,
    )
    assert changed and deployment_limit.scope.deployment_id == "secondary"


def test_retargeted_alias_rejects_previous_pool_scope(tmp_path: Path) -> None:
    """Only the active revision's pool authorizes new pool limits after a retarget."""
    clock = _Clock()
    store, _ledger, budgets, _key = _authority(tmp_path, clock)
    _write_snapshot(tmp_path)
    store.activate_alias_revision(
        organization_id="org",
        alias_id="coding",
        alias_name="coding",
        revision_id="revision-two",
        target=DirectTarget(pool_id="pool-two"),
        snapshot_ref="snapshot",
        catalog_sha256=_catalog().identity_sha256(),
    )

    with pytest.raises(ValueError, match="pool is not the active revision target"):
        budgets.set_limit(
            organization_id="org",
            period="2026-08",
            scope=BudgetScope(kind=BudgetScopeKind.POOL, alias_id="coding", pool_id="pool"),
            limit_micro_usd=100,
        )


def test_deployment_budget_scope_fails_closed_on_tampered_snapshot(tmp_path: Path) -> None:
    """A snapshot whose content differs from the registered digest is rejected."""
    clock = _Clock()
    _store, _ledger, budgets, _key = _authority(tmp_path, clock)
    tampered = NormalizedGatewayCatalog(
        deployments=(_deployment(),),
        pools=(
            ExactModelPool(
                pool_id="pool",
                exact_model_id="exact-one",
                deployment_ids=("primary",),
            ),
        ),
    )
    (tmp_path / "snapshot").write_bytes(canonical_json_bytes(tampered.model_dump(mode="json")))

    with pytest.raises(ValueError, match="digest does not match"):
        budgets.set_limit(
            organization_id="org",
            period="2026-08",
            scope=BudgetScope(
                kind=BudgetScopeKind.DEPLOYMENT,
                alias_id="coding",
                pool_id="pool",
                deployment_id="primary",
            ),
            limit_micro_usd=100,
        )


def test_deployment_budget_scope_fails_closed_on_unreadable_snapshot(tmp_path: Path) -> None:
    """A missing pinned snapshot rejects deployment scopes instead of trusting them."""
    clock = _Clock()
    _store, _ledger, budgets, _key = _authority(tmp_path, clock)

    with pytest.raises(ValueError, match="snapshot is unreadable"):
        budgets.set_limit(
            organization_id="org",
            period="2026-08",
            scope=BudgetScope(
                kind=BudgetScopeKind.DEPLOYMENT,
                alias_id="coding",
                pool_id="pool",
                deployment_id="primary",
            ),
            limit_micro_usd=100,
        )


def test_reservation_prices_the_long_context_tier_conservatively() -> None:
    """The byte bound decides tier exposure fail-safe.

    Canonical bytes never undercount tokens, so a request whose bytes stay
    below the threshold reserves at base rates, while a byte bound at or
    past the threshold must survive the whole-request premium schedule; a
    tier missing a required rate unprices the route entirely.
    """
    from exp.common.models.catalog import GatewayLongContextTier

    def tiered(tier: GatewayLongContextTier | None) -> ExactModelDeployment:
        base = _deployment()
        return base.model_copy(
            update={
                "gateway": base.gateway.model_copy(
                    update={
                        "prices": GatewayTokenPrices(
                            input_micro_usd_per_million_tokens=1_000_000,
                            output_micro_usd_per_million_tokens=2_000_000,
                            long_context=tier,
                        )
                    }
                )
            }
        )

    tier = GatewayLongContextTier(
        input_threshold_tokens=64,
        input_micro_usd_per_million_tokens=3_000_000,
        output_micro_usd_per_million_tokens=5_000_000,
    )
    request = _request("x" * 400)
    input_bytes = len(canonical_json_bytes(request))
    assert input_bytes >= 64

    flat = maximum_attempt_cost_micro_usd(request, tiered(None))
    premium = maximum_attempt_cost_micro_usd(request, tiered(tier))
    assert flat is not None and premium is not None
    # 16 output tokens from the deployment ceiling; the premium worst case
    # prices both directions at the tier's higher rates.
    assert flat == (input_bytes * 1_000_000 + 16 * 2_000_000 + 999_999) // 1_000_000
    assert premium == (input_bytes * 3_000_000 + 16 * 5_000_000 + 999_999) // 1_000_000

    # Below the threshold the tier cannot trigger, so base rates reserve.
    unreachable = tier.model_copy(update={"input_threshold_tokens": input_bytes + 1})
    assert maximum_attempt_cost_micro_usd(request, tiered(unreachable)) == flat

    # A reachable tier with an unknown required rate fails closed.
    unpriced = GatewayLongContextTier(
        input_threshold_tokens=64,
        input_micro_usd_per_million_tokens=3_000_000,
    )
    assert maximum_attempt_cost_micro_usd(request, tiered(unpriced)) is None


def test_reservation_counts_excluded_provider_carriers_toward_the_tier_bound() -> None:
    """Serialization-excluded carriers cannot dodge the premium reservation.

    Replayed encrypted reasoning is provider-read input excluded from the
    request's plain serialization; without its envelope bytes a
    carrier-heavy request could reserve at base rates and settle at the
    premium schedule, overdrawing a hard budget.
    """
    from exp.common.models.catalog import GatewayLongContextTier
    from exp.runtime.gateway.contracts import EncryptedReasoningBlock, GatewayApiSurface

    tier = GatewayLongContextTier(
        input_threshold_tokens=2_048,
        input_micro_usd_per_million_tokens=3_000_000,
        output_micro_usd_per_million_tokens=5_000_000,
    )
    base = _deployment()
    tiered = base.model_copy(
        update={
            "gateway": base.gateway.model_copy(
                update={
                    "prices": GatewayTokenPrices(
                        input_micro_usd_per_million_tokens=1_000_000,
                        output_micro_usd_per_million_tokens=2_000_000,
                        long_context=tier,
                    )
                }
            )
        }
    )
    carried = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(
            GatewayMessage(
                role="assistant",
                content="tiny",
                provider_reasoning=(
                    EncryptedReasoningBlock(
                        id="rs_carrier",
                        encrypted_content="A" * 4_096,
                        output_index=0,
                    ),
                ),
            ),
        ),
        maximum_output_tokens=16,
    )
    from exp.runtime.gateway.replay_identity import provider_replay_authority

    visible_bytes = len(canonical_json_bytes(carried))
    assert visible_bytes < 2_048
    envelope = provider_replay_authority(carried)
    assert envelope is not None
    bound = visible_bytes + len(canonical_json_bytes(envelope))
    assert bound >= 2_048
    # The premium worst case governs because the carrier bytes cross the
    # threshold even though the visible serialization stays below it.
    expected = (bound * 3_000_000 + 16 * 5_000_000 + 999_999) // 1_000_000
    assert maximum_attempt_cost_micro_usd(carried, tiered) == expected


def test_worst_case_attempt_tokens_matches_over_the_serving_request_union() -> None:
    """The promo token reservation is match-aware: completions reserve worst-case
    input and clamped output; embeddings and image requests reserve their
    byte-bounded input and zero completion output."""
    deployment = _deployment()

    completion_in, completion_out = worst_case_attempt_tokens(_request("four bytes"), deployment)
    assert completion_in > 0
    # 16 is the fixture request's maximum_output_tokens, at/under the deployment ceiling.
    assert completion_out == 16

    embeddings = EmbeddingsRequest(inputs=("hello", "world"))
    emb_in, emb_out = worst_case_attempt_tokens(embeddings, deployment)
    assert emb_in == len(canonical_json_bytes(embeddings)) and emb_in > 0
    assert emb_out == 0

    images = ImagesRequest(prompt="a cat")
    img_in, img_out = worst_case_attempt_tokens(images, deployment)
    assert img_in == len(canonical_json_bytes(images)) and img_in > 0
    assert img_out == 0
