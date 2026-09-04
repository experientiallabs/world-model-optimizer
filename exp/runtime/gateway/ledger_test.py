"""Tests for content-free attempt accounting, recovery, and usage."""

from __future__ import annotations

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.models.catalog import (
    BillingSource,
    GatewayDeploymentMetadata,
    GatewayLongContextTier,
    GatewayTokenPrices,
)
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
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
from exp.runtime.gateway.ledger import (
    GatewayLedgerError,
    IdempotencyConflictError,
    IdempotencyReplayUnavailableError,
    SQLiteAttemptLedger,
)
from exp.runtime.gateway.sqlite.store import SQLiteGatewayStore

_CATALOG_DIGEST = "a" * 64


class FakeLedgerClock:
    """Controllable wall and monotonic clock for ledger tests."""

    def __init__(self) -> None:
        """Initialize fixed wall and monotonic times."""
        self.wall = datetime(2026, 8, 18, 20, 0, tzinfo=UTC)
        self.monotonic_value = 1_000.0

    def now(self) -> datetime:
        """Return the controlled wall time."""
        return self.wall

    def monotonic(self) -> float:
        """Return the controlled monotonic time."""
        return self.monotonic_value

    def advance(self, seconds: float) -> None:
        """Advance wall and monotonic time equally.

        Args:
            seconds: Elapsed seconds.
        """
        self.wall += timedelta(seconds=seconds)
        self.monotonic_value += seconds


def _deployment(
    *,
    priced: bool = True,
    billing_source: BillingSource = BillingSource.CUSTOMER_MANAGED,
) -> ExactModelDeployment:
    """Create one exact singleton deployment with optional known rates."""
    prices = (
        GatewayTokenPrices(
            input_micro_usd_per_million_tokens=2_000_000,
            cached_input_micro_usd_per_million_tokens=1_000_000,
            output_micro_usd_per_million_tokens=4_000_000,
            reasoning_micro_usd_per_million_tokens=5_000_000,
        )
        if priced
        else GatewayTokenPrices()
    )
    return ExactModelDeployment(
        deployment_id="deployment-one",
        source_alias="source-one",
        exact_model_id="exact-one",
        connection="connection-one",
        provider="openai",
        provider_model="provider-model-canary",
        billing_source=billing_source,
        connection_sha256="b" * 64,
        capabilities_sha256="c" * 64,
        gateway=GatewayDeploymentMetadata(
            prices=prices,
            pricing_source="operator-authored",
            pricing_effective_at=datetime(2026, 8, 18, tzinfo=UTC),
        ),
    )


def _request(
    content: str,
    *,
    idempotency_key: str | None = None,
    client_request_id: str | None = None,
) -> GatewayRequest:
    """Create one request whose content must not enter SQLite."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content=content),),
        idempotency_key=idempotency_key,
        client_request_id=client_request_id,
    )


def _authority_fixture(
    tmp_path: Path,
    clock: FakeLedgerClock,
) -> tuple[SQLiteGatewayStore, SQLiteAttemptLedger, str]:
    """Create explicit authority and one granted key for ledger tests."""
    path = tmp_path / "gateway.db"
    store = SQLiteGatewayStore(path, clock=clock)
    ledger = SQLiteAttemptLedger(path, clock=clock)
    store.create_organization(organization_id="org-one", slug="one", display_name="One")
    store.create_identity(
        organization_id="org-one", identity_id="identity-one", display_name="Identity"
    )
    store.register_catalog_snapshot(
        organization_id="org-one",
        snapshot_ref="snapshot-one",
        catalog_sha256=_CATALOG_DIGEST,
    )
    store.activate_alias_revision(
        organization_id="org-one",
        alias_id="alias-one",
        alias_name="coding",
        revision_id="revision-one",
        target=DirectTarget(pool_id="pool-one"),
        snapshot_ref="snapshot-one",
        catalog_sha256=_CATALOG_DIGEST,
    )
    store.grant_alias(organization_id="org-one", identity_id="identity-one", alias_id="alias-one")
    issued = store.issue_virtual_key(
        organization_id="org-one", identity_id="identity-one", key_id="key-one"
    )
    return store, ledger, issued.raw_key


def _execution(authorization: AuthorizationSnapshot) -> ExecutionSnapshot:
    """Bind a typed authorization snapshot to the singleton route."""
    return ExecutionSnapshot(
        authorization=authorization,
        exact_model_id="exact-one",
        pool_id="pool-one",
        deployment_ids=("deployment-one",),
    )


def test_attempt_usage_and_integer_cost_are_content_free(tmp_path: Path) -> None:
    """Attempt settlement preserves normalized usage and attributed integer cost only."""
    clock = FakeLedgerClock()
    store, ledger, raw_key = _authority_fixture(tmp_path, clock)
    request = _request("prompt-content-canary")
    authorization = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=request,
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=authorization)
    attempt_id = ledger.start_attempt(
        snapshot=_execution(authorization),
        deployment=_deployment(),
        attempt_ordinal=0,
        route_depth=0,
        route_reason="direct_alias",
        fallback_reason=None,
    )
    clock.advance(0.125)

    ledger.finish_attempt(
        attempt_id=attempt_id,
        terminal_event=GatewayEvent(
            kind=GatewayEventKind.COMPLETED,
            sequence_number=3,
            usage=GatewayUsage(
                input_tokens=1_000,
                cached_input_tokens=100,
                output_tokens=500,
                reasoning_tokens=50,
            ),
        ),
        failure=None,
    )

    usage = ledger.usage(organization_id="org-one")
    assert len(usage) == 1
    assert usage[0].requests == 1
    assert usage[0].attempts == 1
    assert usage[0].known_estimated_cost_micro_usd == 3_950
    assert usage[0].unknown_cost_attempts == 0
    assert 124 <= usage[0].total_latency_ms <= 126
    assert usage[0].average_latency_ms is not None
    assert usage[0].terminal_counts[0].state == "completed"

    durable = (tmp_path / "gateway.db").read_bytes()
    wal = tmp_path / "gateway.db-wal"
    if wal.exists():
        durable += wal.read_bytes()
    assert b"prompt-content-canary" not in durable
    assert raw_key.encode() not in durable
    assert b"provider-model-canary" not in durable


def test_attempt_retains_dispatch_billing_source_after_catalog_change(tmp_path: Path) -> None:
    """Attempt attribution remains frozen when later catalog ownership changes."""
    clock = FakeLedgerClock()
    store, ledger, raw_key = _authority_fixture(tmp_path, clock)
    authorization = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("billing-freeze"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=authorization)
    dispatched = _deployment(billing_source=BillingSource.HOST_MANAGED)
    attempt_id = ledger.start_attempt(
        snapshot=_execution(authorization),
        deployment=dispatched,
        attempt_ordinal=0,
        route_depth=0,
    )

    authored_after_dispatch = dispatched.model_copy(
        update={"billing_source": BillingSource.CUSTOMER_MANAGED}
    )
    assert authored_after_dispatch.billing_source == BillingSource.CUSTOMER_MANAGED
    ledger.finish_attempt(
        attempt_id=attempt_id,
        terminal_event=GatewayEvent(
            kind=GatewayEventKind.COMPLETED,
            sequence_number=1,
            usage=GatewayUsage(input_tokens=1, output_tokens=1),
        ),
        failure=None,
    )

    connection = sqlite3.connect(tmp_path / "gateway.db")
    try:
        row = connection.execute(
            "SELECT billing_source, state FROM gateway_attempts WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
    finally:
        connection.close()
    assert row == (BillingSource.HOST_MANAGED.value, "completed")


def test_usage_billing_source_buckets_conserve_physical_attempt_totals(tmp_path: Path) -> None:
    """Source buckets conserve attempts, usage, cost, unknowns, and terminal states."""
    clock = FakeLedgerClock()
    store, ledger, raw_key = _authority_fixture(tmp_path, clock)

    host_authorization = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("host-attempt"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=host_authorization)
    host_attempt = ledger.start_attempt(
        snapshot=_execution(host_authorization),
        deployment=_deployment(billing_source=BillingSource.HOST_MANAGED),
        attempt_ordinal=0,
        route_depth=0,
    )
    ledger.finish_attempt(
        attempt_id=host_attempt,
        terminal_event=GatewayEvent(
            kind=GatewayEventKind.COMPLETED,
            sequence_number=1,
            usage=GatewayUsage(input_tokens=3, output_tokens=2),
        ),
        failure=None,
    )

    customer_authorization = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("customer-attempt"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=customer_authorization)
    customer_attempt = ledger.start_attempt(
        snapshot=_execution(customer_authorization),
        deployment=_deployment(
            priced=False,
            billing_source=BillingSource.CUSTOMER_MANAGED,
        ),
        attempt_ordinal=0,
        route_depth=0,
    )
    ledger.finish_attempt(
        attempt_id=customer_attempt,
        terminal_event=None,
        failure=GatewayFailure(
            failure_class=GatewayFailureClass.TRANSPORT,
            safe_message="provider unavailable",
        ),
    )

    buckets = ledger.usage_by_billing_source(organization_id="org-one")

    assert [bucket.billing_source for bucket in buckets] == [
        BillingSource.CUSTOMER_MANAGED,
        BillingSource.HOST_MANAGED,
    ]
    customer, host = buckets
    assert customer.attempts == 1
    assert customer.unknown_cost_attempts == 1
    assert [(item.state, item.attempts) for item in customer.terminal_counts] == [("failed", 1)]
    assert host.attempts == 1
    assert host.input_tokens == 3
    assert host.output_tokens == 2
    assert host.known_estimated_cost_micro_usd == 14
    assert host.unknown_cost_attempts == 0
    assert [(item.state, item.attempts) for item in host.terminal_counts] == [("completed", 1)]


def test_usage_snapshot_conserves_source_totals_during_concurrent_wal_settlement(
    tmp_path: Path,
) -> None:
    """Concurrent settlement cannot split identity and source report snapshots."""
    clock = FakeLedgerClock()
    store, ledger, raw_key = _authority_fixture(tmp_path, clock)
    attempts: list[str] = []
    for index in range(24):
        authorization = store.authorize_request(
            raw_key=raw_key,
            alias="coding",
            request=_request(f"concurrent-usage-{index}"),
            deadline_monotonic=clock.monotonic() + 30,
        )
        ledger.accept_request(authorization=authorization)
        attempts.append(
            ledger.start_attempt(
                snapshot=_execution(authorization),
                deployment=_deployment(
                    billing_source=(
                        BillingSource.HOST_MANAGED
                        if index % 2 == 0
                        else BillingSource.CUSTOMER_MANAGED
                    )
                ),
                attempt_ordinal=0,
                route_depth=0,
            )
        )

    def settle_attempts() -> None:
        """Settle every dispatched attempt while readers hold independent WAL snapshots."""
        for attempt_id in attempts:
            ledger.finish_attempt(
                attempt_id=attempt_id,
                terminal_event=GatewayEvent(
                    kind=GatewayEventKind.COMPLETED,
                    sequence_number=1,
                    usage=GatewayUsage(input_tokens=3, output_tokens=2),
                ),
                failure=None,
            )
            time.sleep(0.001)

    def assert_conservation() -> None:
        """Require every metric and terminal state to reconcile inside one read."""
        snapshot = ledger.usage_snapshot(organization_id="org-one")
        identity = snapshot.identities[0]
        sources = snapshot.by_billing_source
        assert identity.attempts == sum(item.attempts for item in sources)
        for field_name in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "known_estimated_cost_micro_usd",
            "unknown_cost_attempts",
        ):
            assert getattr(identity, field_name) == sum(
                getattr(item, field_name) for item in sources
            )
        identity_terminals = {item.state: item.attempts for item in identity.terminal_counts}
        source_terminals: dict[str, int] = {}
        for source in sources:
            for item in source.terminal_counts:
                source_terminals[item.state] = source_terminals.get(item.state, 0) + item.attempts
        assert identity_terminals == source_terminals

    with ThreadPoolExecutor(max_workers=1) as executor:
        settlement = executor.submit(settle_attempts)
        for _index in range(100):
            assert_conservation()
            if settlement.done():
                break
        settlement.result(timeout=5)

    final = ledger.usage_snapshot(organization_id="org-one")
    assert sum(item.attempts for item in final.identities[0].terminal_counts) == len(attempts)
    assert sum(
        terminal.attempts
        for source in final.by_billing_source
        for terminal in source.terminal_counts
    ) == len(attempts)


def test_unknown_prices_remain_unknown_instead_of_zero(tmp_path: Path) -> None:
    """Observed token usage with absent rates increments unknown-cost accounting."""
    clock = FakeLedgerClock()
    store, ledger, raw_key = _authority_fixture(tmp_path, clock)
    authorization = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("unknown-price"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=authorization)
    attempt_id = ledger.start_attempt(
        snapshot=_execution(authorization),
        deployment=_deployment(priced=False),
        attempt_ordinal=0,
        route_depth=0,
    )
    ledger.finish_attempt(
        attempt_id=attempt_id,
        terminal_event=GatewayEvent(
            kind=GatewayEventKind.COMPLETED,
            sequence_number=1,
            usage=GatewayUsage(input_tokens=1, output_tokens=1),
        ),
        failure=None,
    )

    usage = ledger.usage(organization_id="org-one")[0]
    assert usage.known_estimated_cost_micro_usd == 0
    assert usage.unknown_cost_attempts == 1


def test_cancelled_post_commit_attempt_keeps_observed_billable_usage(tmp_path: Path) -> None:
    """Cancellation does not erase observed provider usage or attributed cost."""
    clock = FakeLedgerClock()
    store, ledger, raw_key = _authority_fixture(tmp_path, clock)
    authorization = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("cancelled-stream"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=authorization)
    attempt_id = ledger.start_attempt(
        snapshot=_execution(authorization),
        deployment=_deployment(),
        attempt_ordinal=0,
        route_depth=0,
    )
    cancelled = GatewayFailure(
        failure_class=GatewayFailureClass.CANCELLED,
        safe_message="client disconnected",
    )
    ledger.finish_attempt(
        attempt_id=attempt_id,
        terminal_event=GatewayEvent(
            kind=GatewayEventKind.FAILED,
            sequence_number=2,
            failure=cancelled,
            usage=GatewayUsage(input_tokens=100, output_tokens=20),
        ),
        failure=cancelled,
    )

    usage = ledger.usage(organization_id="org-one")[0]
    assert usage.known_estimated_cost_micro_usd == 280
    assert usage.terminal_counts[0].attempts == 1
    assert usage.terminal_counts[0].state == "cancelled"


def test_failed_attempt_terminalizes_its_parent_request(tmp_path: Path) -> None:
    """An ordinary terminal provider failure closes both attempt and request state."""
    clock = FakeLedgerClock()
    store, ledger, raw_key = _authority_fixture(tmp_path, clock)
    authorization = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("failed-request"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=authorization)
    attempt_id = ledger.start_attempt(
        snapshot=_execution(authorization),
        deployment=_deployment(),
        attempt_ordinal=0,
        route_depth=0,
    )
    failure = GatewayFailure(
        failure_class=GatewayFailureClass.PROVIDER_INTERNAL,
        safe_message="provider request failed",
    )

    ledger.finish_attempt(attempt_id=attempt_id, terminal_event=None, failure=failure)

    connection = sqlite3.connect(tmp_path / "gateway.db")
    try:
        request_state = connection.execute(
            "SELECT terminal_state FROM gateway_requests WHERE request_id = ?",
            (authorization.request_id,),
        ).fetchone()[0]
        attempt_state = connection.execute(
            "SELECT state FROM gateway_attempts WHERE attempt_id = ?", (attempt_id,)
        ).fetchone()[0]
    finally:
        connection.close()
    assert request_state == "failed"
    assert attempt_state == "failed"


def test_failed_attempt_records_the_sanitized_provider_error_text(tmp_path: Path) -> None:
    """A provider client-error rejection keeps its sanitized sentence on the row."""
    clock = FakeLedgerClock()
    store, ledger, raw_key = _authority_fixture(tmp_path, clock)
    authorization = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("rejected-request"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=authorization)
    attempt_id = ledger.start_attempt(
        snapshot=_execution(authorization),
        deployment=_deployment(),
        attempt_ordinal=0,
        route_depth=0,
    )
    failure = GatewayFailure(
        failure_class=GatewayFailureClass.INVALID_REQUEST,
        safe_message="provider rejected the request; verify the request fields",
        # The sanitized provider sentence: no headers, no request echo, no
        # credentials (the Rust upstream already enforced that shape).
        provider_detail="max_tokens must be greater than thinking budget_tokens.",
    )

    ledger.finish_attempt(attempt_id=attempt_id, terminal_event=None, failure=failure)

    connection = sqlite3.connect(tmp_path / "gateway.db")
    try:
        failure_class, failure_message = connection.execute(
            "SELECT failure_class, failure_message FROM gateway_attempts WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
    finally:
        connection.close()
    assert failure_class == "invalid_request"
    assert failure_message == "max_tokens must be greater than thinking budget_tokens."


def test_failed_attempt_without_provider_detail_leaves_the_message_null(tmp_path: Path) -> None:
    """A failure carrying no provider explanation records a NULL message."""
    clock = FakeLedgerClock()
    store, ledger, raw_key = _authority_fixture(tmp_path, clock)
    authorization = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("bare-failure"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=authorization)
    attempt_id = ledger.start_attempt(
        snapshot=_execution(authorization),
        deployment=_deployment(),
        attempt_ordinal=0,
        route_depth=0,
    )
    ledger.finish_attempt(
        attempt_id=attempt_id,
        terminal_event=None,
        failure=GatewayFailure(
            failure_class=GatewayFailureClass.PROVIDER_INTERNAL,
            safe_message="provider service failed",
        ),
    )

    connection = sqlite3.connect(tmp_path / "gateway.db")
    try:
        failure_message = connection.execute(
            "SELECT failure_message FROM gateway_attempts WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()[0]
    finally:
        connection.close()
    assert failure_message is None


def test_predispatch_failure_terminalizes_real_sqlite_request(tmp_path: Path) -> None:
    """Accepted routing failures cannot remain unterminated without an attempt row."""
    clock = FakeLedgerClock()
    store, ledger, raw_key = _authority_fixture(tmp_path, clock)
    authorization = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("routing-failure"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=authorization)

    ledger.finish_request(
        authorization=authorization,
        failure=GatewayFailure(
            failure_class=GatewayFailureClass.INTERNAL,
            safe_message="route activation failed",
        ),
    )

    connection = sqlite3.connect(tmp_path / "gateway.db")
    try:
        request_state = connection.execute(
            "SELECT terminal_state FROM gateway_requests WHERE request_id = ?",
            (authorization.request_id,),
        ).fetchone()[0]
        attempt_count = connection.execute(
            "SELECT COUNT(*) FROM gateway_attempts WHERE request_id = ?",
            (authorization.request_id,),
        ).fetchone()[0]
    finally:
        connection.close()
    assert request_state == "failed"
    assert attempt_count == 0


def test_intermediate_attempt_can_settle_without_finalizing_parent(tmp_path: Path) -> None:
    """The physical-attempt seam leaves parent finalization to a later route owner."""
    clock = FakeLedgerClock()
    store, ledger, raw_key = _authority_fixture(tmp_path, clock)
    authorization = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("future-waterfall"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=authorization)
    attempt_id = ledger.start_attempt(
        snapshot=_execution(authorization),
        deployment=_deployment(),
        attempt_ordinal=0,
        route_depth=0,
    )

    ledger.finish_attempt(
        attempt_id=attempt_id,
        terminal_event=None,
        failure=GatewayFailure(
            failure_class=GatewayFailureClass.TRANSPORT,
            safe_message="retry on sibling route",
        ),
        finalize_request=False,
    )

    connection = sqlite3.connect(tmp_path / "gateway.db")
    try:
        states = connection.execute(
            """
            SELECT r.terminal_state, a.state
            FROM gateway_requests AS r
            JOIN gateway_attempts AS a ON a.request_id = r.request_id
            WHERE r.request_id = ?
            """,
            (authorization.request_id,),
        ).fetchone()
    finally:
        connection.close()
    assert states == (None, "failed")


def test_concurrent_attempt_ordinal_conflict_rolls_back_without_blocking_retry(
    tmp_path: Path,
) -> None:
    """One physical ordinal wins concurrently and the next ordinal remains writable."""
    clock = FakeLedgerClock()
    store, ledger, raw_key = _authority_fixture(tmp_path, clock)
    authorization = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("concurrent-ordinal"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=authorization)

    def start_first_ordinal() -> str | None:
        """Compete for one physical ordinal and normalize the expected loser."""
        try:
            return ledger.start_attempt(
                snapshot=_execution(authorization),
                deployment=_deployment(),
                attempt_ordinal=0,
                route_depth=0,
            )
        except sqlite3.IntegrityError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(executor.submit(start_first_ordinal) for _index in range(2))
        results = tuple(future.result(timeout=5) for future in futures)
    winners = tuple(result for result in results if result is not None)
    assert len(winners) == 1
    ledger.finish_attempt(
        attempt_id=winners[0],
        terminal_event=None,
        failure=GatewayFailure(
            failure_class=GatewayFailureClass.TRANSPORT,
            safe_message="safe physical retry",
        ),
        finalize_request=False,
    )

    retry_id = ledger.start_attempt(
        snapshot=_execution(authorization),
        deployment=_deployment(),
        attempt_ordinal=1,
        route_depth=0,
    )

    connection = sqlite3.connect(tmp_path / "gateway.db")
    try:
        rows = connection.execute(
            "SELECT attempt_ordinal, route_depth FROM gateway_attempts ORDER BY attempt_ordinal"
        ).fetchall()
    finally:
        connection.close()
    assert rows == [(0, 0), (1, 0)]
    ledger.finish_attempt(
        attempt_id=retry_id,
        terminal_event=GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=0),
        failure=None,
    )


def test_terminal_parent_rejects_late_attempt_dispatch(tmp_path: Path) -> None:
    """No retry path can dispatch after durable request terminalization."""
    clock = FakeLedgerClock()
    store, ledger, raw_key = _authority_fixture(tmp_path, clock)
    authorization = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("late-dispatch"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=authorization)
    ledger.finish_request(
        authorization=authorization,
        failure=GatewayFailure(
            failure_class=GatewayFailureClass.INTERNAL,
            safe_message="route failed before dispatch",
        ),
    )

    with pytest.raises(GatewayLedgerError, match="already terminal"):
        ledger.start_attempt(
            snapshot=_execution(authorization),
            deployment=_deployment(),
            attempt_ordinal=0,
            route_depth=0,
        )


def test_idempotency_is_opt_in_and_restart_replay_fails_closed(tmp_path: Path) -> None:
    """A stored keyed request is never redispatched when response content is unavailable."""
    clock = FakeLedgerClock()
    store, ledger, raw_key = _authority_fixture(tmp_path, clock)
    first = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("same", idempotency_key="caller-operation"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=first)

    restarted = SQLiteAttemptLedger(tmp_path / "gateway.db", clock=clock)
    matching = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("same", idempotency_key="caller-operation"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    with pytest.raises(IdempotencyReplayUnavailableError, match="replay is unavailable"):
        restarted.accept_request(authorization=matching)

    conflicting = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("different", idempotency_key="caller-operation"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    with pytest.raises(IdempotencyConflictError, match="different request"):
        restarted.accept_request(authorization=conflicting)


def test_a_session_correlation_id_never_keys_duplicate_detection(tmp_path: Path) -> None:
    """Sequential distinct requests sharing one X-Client-Request-Id all accept.

    Codex sends its session id in that header on every request of a session
    (captured live 2026-08-29), so treating it as an operation key would
    reject the second request of every real session as a conflict.
    """
    clock = FakeLedgerClock()
    store, ledger, raw_key = _authority_fixture(tmp_path, clock)
    for content in ("first turn", "second turn", "third turn"):
        authorization = store.authorize_request(
            raw_key=raw_key,
            alias="coding",
            request=_request(content, client_request_id="codex-session-one"),
            deadline_monotonic=clock.monotonic() + 30,
        )
        assert authorization.caller_operation_sha256 is None
        ledger.accept_request(authorization=authorization)


def test_a_failed_keyed_request_still_conflicts_a_mutated_retry(tmp_path: Path) -> None:
    """Reusing an Idempotency-Key with a different body fails closed even
    after the prior attempt failed: after a transport-ambiguous failure the
    provider may have executed, so silently running different content under
    the same operation identity is exactly what the key exists to prevent.
    A stuck client must mint a new key rather than mutate the body."""
    clock = FakeLedgerClock()
    store, ledger, raw_key = _authority_fixture(tmp_path, clock)
    failed = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("original", idempotency_key="retry-operation"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=failed)
    ledger.finish_request(
        authorization=failed,
        failure=GatewayFailure(
            failure_class=GatewayFailureClass.TRANSPORT,
            safe_message="provider unavailable",
        ),
    )
    mutated = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("mutated", idempotency_key="retry-operation"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    with pytest.raises(IdempotencyConflictError, match="different request"):
        ledger.accept_request(authorization=mutated)


def test_crash_reconciliation_waits_for_deadline_and_cleanup_bound(tmp_path: Path) -> None:
    """Expired accepted work is free while dispatched work becomes unknown after crash."""
    clock = FakeLedgerClock()
    store, ledger, raw_key = _authority_fixture(tmp_path, clock)
    accepted = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("accepted-only"),
        deadline_monotonic=clock.monotonic() + 10,
    )
    ledger.accept_request(authorization=accepted)
    dispatched = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("dispatched", idempotency_key="crash-operation"),
        deadline_monotonic=clock.monotonic() + 10,
    )
    ledger.accept_request(authorization=dispatched)
    ledger.start_attempt(
        snapshot=_execution(dispatched),
        deployment=_deployment(),
        attempt_ordinal=0,
        route_depth=0,
    )

    clock.advance(12)
    assert ledger.reconcile_crashed_requests(cleanup_grace=timedelta(seconds=5)) == (1, 0)
    clock.advance(4)
    assert ledger.reconcile_crashed_requests(cleanup_grace=timedelta(seconds=5)) == (0, 1)

    superseding = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("dispatched", idempotency_key="crash-operation"),
        deadline_monotonic=clock.monotonic() + 10,
    )
    ledger.accept_request(authorization=superseding)
    ledger.start_attempt(
        snapshot=_execution(superseding),
        deployment=_deployment(),
        attempt_ordinal=0,
        route_depth=0,
    )

    connection = sqlite3.connect(tmp_path / "gateway.db")
    try:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM gateway_attempts WHERE state = 'unknown_after_crash'"
            ).fetchone()[0]
            == 1
        )
        assert connection.execute("SELECT COUNT(*) FROM gateway_attempts").fetchone()[0] == 2
    finally:
        connection.close()


def test_concurrent_multi_identity_wal_preserves_receipts_grants_and_attempts(
    tmp_path: Path,
) -> None:
    """Concurrent writers retain each authority mutation and terminal attempt exactly once."""
    path = tmp_path / "gateway.db"
    store = SQLiteGatewayStore(path, busy_timeout_ms=10_000)
    ledger = SQLiteAttemptLedger(path, busy_timeout_ms=10_000)
    store.create_organization(organization_id="org-one", slug="one", display_name="One")
    store.register_catalog_snapshot(
        organization_id="org-one",
        snapshot_ref="snapshot-one",
        catalog_sha256=_CATALOG_DIGEST,
    )
    store.activate_alias_revision(
        organization_id="org-one",
        alias_id="alias-one",
        alias_name="coding",
        revision_id="revision-one",
        target=DirectTarget(pool_id="pool-one"),
        snapshot_ref="snapshot-one",
        catalog_sha256=_CATALOG_DIGEST,
    )

    def run_identity(index: int) -> None:
        """Create one identity and account one completed request."""
        identity_id = f"identity-{index}"
        key_id = f"key-{index}"
        store.create_identity(
            organization_id="org-one",
            identity_id=identity_id,
            display_name=f"Identity {index}",
            operation_id=f"operation-identity-{index}",
        )
        store.grant_alias(organization_id="org-one", identity_id=identity_id, alias_id="alias-one")
        issued = store.issue_virtual_key(
            organization_id="org-one",
            identity_id=identity_id,
            key_id=key_id,
            operation_id=f"operation-key-{index}",
        )
        authorization = store.authorize_request(
            raw_key=issued.raw_key,
            alias="coding",
            request=_request(f"concurrent-content-{index}"),
            deadline_monotonic=time.monotonic() + 60,
        )
        ledger.accept_request(authorization=authorization)
        attempt_id = ledger.start_attempt(
            snapshot=_execution(authorization),
            deployment=_deployment(),
            attempt_ordinal=0,
            route_depth=0,
        )
        ledger.finish_attempt(
            attempt_id=attempt_id,
            terminal_event=GatewayEvent(
                kind=GatewayEventKind.COMPLETED,
                sequence_number=1,
                usage=GatewayUsage(input_tokens=1, output_tokens=1),
            ),
            failure=None,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        tuple(executor.map(run_identity, range(8)))

    usage = ledger.usage(organization_id="org-one")
    assert len(usage) == 8
    assert sum(item.requests for item in usage) == 8
    assert sum(item.attempts for item in usage) == 8
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM operation_receipts").fetchone()[0] == 16
        assert connection.execute("SELECT COUNT(*) FROM identity_alias_grants").fetchone()[0] == 8
    finally:
        connection.close()


def test_accept_and_finish_persist_app_attribution_and_first_token_at(tmp_path: Path) -> None:
    """Accept freezes the caller app identity and finish records the first-token time."""
    clock = FakeLedgerClock()
    store, ledger, raw_key = _authority_fixture(tmp_path, clock)
    authorization = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("prompt-content-canary"),
        deadline_monotonic=clock.monotonic() + 30,
        app_referer="https://app.example.com",
        app_title="Example App",
    )
    assert authorization.app_referer == "https://app.example.com"
    assert authorization.app_title == "Example App"
    ledger.accept_request(authorization=authorization)
    attempt_id = ledger.start_attempt(
        snapshot=_execution(authorization),
        deployment=_deployment(),
        attempt_ordinal=0,
        route_depth=0,
    )
    clock.advance(0.2)
    first_token_at = clock.now()
    clock.advance(0.3)
    ledger.finish_attempt(
        attempt_id=attempt_id,
        terminal_event=GatewayEvent(
            kind=GatewayEventKind.COMPLETED,
            sequence_number=1,
            usage=GatewayUsage(input_tokens=10, output_tokens=5),
        ),
        failure=None,
        first_token_at=first_token_at,
    )

    connection = sqlite3.connect(tmp_path / "gateway.db")
    connection.row_factory = sqlite3.Row
    try:
        attempt_row = connection.execute(
            "SELECT first_token_at, terminal_at FROM gateway_attempts WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        request_row = connection.execute(
            "SELECT app_referer, app_title FROM gateway_requests WHERE request_id = ?",
            (authorization.request_id,),
        ).fetchone()
    finally:
        connection.close()
    assert datetime.fromisoformat(str(attempt_row["first_token_at"])) == first_token_at
    assert str(attempt_row["terminal_at"]) != str(attempt_row["first_token_at"])
    assert str(request_row["app_referer"]) == "https://app.example.com"
    assert str(request_row["app_title"]) == "Example App"


def test_first_token_and_app_attribution_default_to_null(tmp_path: Path) -> None:
    """An attempt that never streamed a token and a caller without app headers stay null."""
    clock = FakeLedgerClock()
    store, ledger, raw_key = _authority_fixture(tmp_path, clock)
    authorization = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("prompt"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    assert authorization.app_referer is None
    assert authorization.app_title is None
    ledger.accept_request(authorization=authorization)
    attempt_id = ledger.start_attempt(
        snapshot=_execution(authorization),
        deployment=_deployment(),
        attempt_ordinal=0,
        route_depth=0,
    )
    ledger.finish_attempt(
        attempt_id=attempt_id,
        terminal_event=None,
        failure=GatewayFailure(
            failure_class=GatewayFailureClass.TRANSPORT,
            safe_message="upstream unavailable",
        ),
    )
    connection = sqlite3.connect(tmp_path / "gateway.db")
    connection.row_factory = sqlite3.Row
    try:
        attempt_row = connection.execute(
            "SELECT first_token_at FROM gateway_attempts WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        request_row = connection.execute(
            "SELECT app_referer, app_title FROM gateway_requests WHERE request_id = ?",
            (authorization.request_id,),
        ).fetchone()
    finally:
        connection.close()
    assert attempt_row["first_token_at"] is None
    assert request_row["app_referer"] is None
    assert request_row["app_title"] is None


def _tiered_deployment() -> ExactModelDeployment:
    """One deployment priced on the published Gemini-style tier schedule."""
    return _deployment().model_copy(
        update={
            "gateway": GatewayDeploymentMetadata(
                prices=GatewayTokenPrices(
                    input_micro_usd_per_million_tokens=1_250_000,
                    cached_input_micro_usd_per_million_tokens=125_000,
                    output_micro_usd_per_million_tokens=10_000_000,
                    reasoning_micro_usd_per_million_tokens=10_000_000,
                    long_context=GatewayLongContextTier(
                        input_threshold_tokens=200_000,
                        input_micro_usd_per_million_tokens=2_500_000,
                        cached_input_micro_usd_per_million_tokens=250_000,
                        output_micro_usd_per_million_tokens=15_000_000,
                        reasoning_micro_usd_per_million_tokens=15_000_000,
                    ),
                ),
                pricing_source="operator-authored",
                pricing_effective_at=datetime(2026, 8, 18, tzinfo=UTC),
            )
        }
    )


def test_long_context_settlement_reprices_the_whole_request_at_the_threshold(
    tmp_path: Path,
) -> None:
    """Settlement selects the frozen schedule by provider-reported input.

    Both published tier schedules (Gemini's "prompts > 200k" rates and
    Anthropic's legacy 1M-beta premium) reprice the ENTIRE request once
    input reaches the threshold, so exactly the threshold boundary decides:
    199,999 input tokens bill the base schedule, 200,000 and 200,001 bill
    every token at the tier rates.
    """
    cases = (
        # (input_tokens, expected settled micro-USD with 1,000 output tokens)
        (199_999, (199_999 * 1_250_000 + 1_000 * 10_000_000 + 500_000) // 1_000_000),
        (200_000, (200_000 * 2_500_000 + 1_000 * 15_000_000 + 500_000) // 1_000_000),
        (200_001, (200_001 * 2_500_000 + 1_000 * 15_000_000 + 500_000) // 1_000_000),
    )
    for index, (input_tokens, expected) in enumerate(cases):
        clock = FakeLedgerClock()
        store, ledger, raw_key = _authority_fixture(tmp_path / f"case-{index}", clock)
        authorization = store.authorize_request(
            raw_key=raw_key,
            alias="coding",
            request=_request(f"boundary-{input_tokens}"),
            deadline_monotonic=clock.monotonic() + 30,
        )
        ledger.accept_request(authorization=authorization)
        attempt_id = ledger.start_attempt(
            snapshot=_execution(authorization),
            deployment=_tiered_deployment(),
            attempt_ordinal=0,
            route_depth=0,
        )
        ledger.finish_attempt(
            attempt_id=attempt_id,
            terminal_event=GatewayEvent(
                kind=GatewayEventKind.COMPLETED,
                sequence_number=1,
                usage=GatewayUsage(input_tokens=input_tokens, output_tokens=1_000),
            ),
            failure=None,
        )
        usage = ledger.usage(organization_id="org-one")
        assert usage[0].known_estimated_cost_micro_usd == expected, input_tokens


def test_long_context_tier_with_an_unknown_rate_stays_unpriced_above_threshold(
    tmp_path: Path,
) -> None:
    """A tier never inherits base rates: a missing tier rate keeps a
    threshold-crossing attempt honestly unpriced instead of under-billed."""
    clock = FakeLedgerClock()
    store, ledger, raw_key = _authority_fixture(tmp_path, clock)
    incomplete_tier = _deployment().model_copy(
        update={
            "gateway": GatewayDeploymentMetadata(
                prices=GatewayTokenPrices(
                    input_micro_usd_per_million_tokens=1_250_000,
                    output_micro_usd_per_million_tokens=10_000_000,
                    long_context=GatewayLongContextTier(
                        input_threshold_tokens=200_000,
                        input_micro_usd_per_million_tokens=2_500_000,
                    ),
                ),
                pricing_source="operator-authored",
            )
        }
    )
    authorization = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("unpriced-above-threshold"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=authorization)
    attempt_id = ledger.start_attempt(
        snapshot=_execution(authorization),
        deployment=incomplete_tier,
        attempt_ordinal=0,
        route_depth=0,
    )
    ledger.finish_attempt(
        attempt_id=attempt_id,
        terminal_event=GatewayEvent(
            kind=GatewayEventKind.COMPLETED,
            sequence_number=1,
            usage=GatewayUsage(input_tokens=250_000, output_tokens=64),
        ),
        failure=None,
    )
    usage = ledger.usage(organization_id="org-one")
    assert usage[0].known_estimated_cost_micro_usd == 0
    assert usage[0].unknown_cost_attempts == 1


def _spill_fixture(
    tmp_path: Path,
    clock: FakeLedgerClock,
    *,
    preferred_priced: bool = True,
) -> tuple[SQLiteAttemptLedger, ExecutionSnapshot, ExactModelDeployment, ExactModelDeployment]:
    """Build a two-rung snapshot with the chosen rung behind a bypassed lead."""
    store, ledger, raw_key = _authority_fixture(tmp_path, clock)
    authorization = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("spill"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=authorization)
    preferred = _deployment(priced=preferred_priced)
    chosen = preferred.model_copy(
        update={
            "deployment_id": "deployment-two",
            "connection_sha256": "e" * 64,
            "gateway": GatewayDeploymentMetadata(
                prices=GatewayTokenPrices(
                    input_micro_usd_per_million_tokens=100_000,
                    cached_input_micro_usd_per_million_tokens=50_000,
                    output_micro_usd_per_million_tokens=200_000,
                    reasoning_micro_usd_per_million_tokens=200_000,
                ),
                pricing_source="operator-authored",
            ),
        }
    )
    snapshot = ExecutionSnapshot(
        authorization=authorization,
        exact_model_id="exact-one",
        pool_id="pool-one",
        deployment_ids=("deployment-one", "deployment-two"),
    )
    return ledger, snapshot, chosen, preferred


def _attempt_row(tmp_path: Path, attempt_id: str) -> sqlite3.Row:
    """Read one persisted attempt row for disclosure assertions."""
    connection = sqlite3.connect(tmp_path / "gateway.db")
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            "SELECT * FROM gateway_attempts WHERE attempt_id = ?", (attempt_id,)
        ).fetchone()
    finally:
        connection.close()
    assert row is not None
    return row


def test_dispatch_disclosure_persists_and_prices_the_counterfactual(tmp_path: Path) -> None:
    """A spilled attempt keeps the preferred rung's rates and prices the delta."""
    clock = FakeLedgerClock()
    ledger, snapshot, chosen, preferred = _spill_fixture(tmp_path, clock)
    attempt_id = ledger.start_attempt(
        snapshot=snapshot,
        deployment=chosen,
        attempt_ordinal=0,
        route_depth=1,
        dispatch_reason="queue_bound",
        preferred_deployment=preferred,
    )
    ledger.finish_attempt(
        attempt_id=attempt_id,
        terminal_event=GatewayEvent(
            kind=GatewayEventKind.COMPLETED,
            sequence_number=3,
            usage=GatewayUsage(
                input_tokens=1_000,
                cached_input_tokens=100,
                output_tokens=500,
                reasoning_tokens=50,
            ),
        ),
        failure=None,
    )
    row = _attempt_row(tmp_path, attempt_id)
    assert row["dispatch_reason"] == "queue_bound"
    assert row["preferred_deployment_id"] == "deployment-one"
    assert row["preferred_input_rate"] == 2_000_000
    assert row["preferred_cached_input_rate"] == 1_000_000
    assert row["preferred_output_rate"] == 4_000_000
    assert row["preferred_reasoning_rate"] == 5_000_000
    # The SAME settled usage priced at the preferred base rates: 900 fresh
    # input + 100 cached + 450 fresh output + 50 reasoning tokens.
    assert row["counterfactual_cost_micro_usd"] == 3_950
    assert row["estimated_cost_micro_usd"] == 195


def test_counterfactual_stays_null_when_a_preferred_rate_is_unknown(tmp_path: Path) -> None:
    """An unpriced preferred rung never guesses a counterfactual cost."""
    clock = FakeLedgerClock()
    ledger, snapshot, chosen, preferred = _spill_fixture(tmp_path, clock, preferred_priced=False)
    attempt_id = ledger.start_attempt(
        snapshot=snapshot,
        deployment=chosen,
        attempt_ordinal=0,
        route_depth=1,
        dispatch_reason="fair_share_shed",
        preferred_deployment=preferred,
    )
    ledger.finish_attempt(
        attempt_id=attempt_id,
        terminal_event=GatewayEvent(
            kind=GatewayEventKind.COMPLETED,
            sequence_number=1,
            usage=GatewayUsage(input_tokens=10, output_tokens=5),
        ),
        failure=None,
    )
    row = _attempt_row(tmp_path, attempt_id)
    assert row["dispatch_reason"] == "fair_share_shed"
    assert row["counterfactual_cost_micro_usd"] is None


def test_undisclosed_attempts_keep_null_disclosure_columns(tmp_path: Path) -> None:
    """A flag-off attempt writes exactly the rows it wrote before this feature."""
    clock = FakeLedgerClock()
    ledger, snapshot, chosen, _preferred = _spill_fixture(tmp_path, clock)
    attempt_id = ledger.start_attempt(
        snapshot=snapshot,
        deployment=chosen,
        attempt_ordinal=0,
        route_depth=1,
    )
    ledger.finish_attempt(
        attempt_id=attempt_id,
        terminal_event=GatewayEvent(
            kind=GatewayEventKind.COMPLETED,
            sequence_number=1,
            usage=GatewayUsage(input_tokens=10, output_tokens=5),
        ),
        failure=None,
    )
    row = _attempt_row(tmp_path, attempt_id)
    assert row["dispatch_reason"] is None
    assert row["preferred_deployment_id"] is None
    assert row["preferred_input_rate"] is None
    assert row["counterfactual_cost_micro_usd"] is None


def test_preferred_rung_disclosure_requires_divergence(tmp_path: Path) -> None:
    """Passing the chosen rung as its own preferred rung is a contract error."""
    clock = FakeLedgerClock()
    ledger, snapshot, chosen, _preferred = _spill_fixture(tmp_path, clock)
    with pytest.raises(GatewayLedgerError, match="divergent"):
        ledger.start_attempt(
            snapshot=snapshot,
            deployment=chosen,
            attempt_ordinal=0,
            route_depth=1,
            dispatch_reason="queue_bound",
            preferred_deployment=chosen,
        )


def test_dispatch_reason_must_be_display_safe(tmp_path: Path) -> None:
    """Control characters in the disclosure code are rejected like route codes."""
    clock = FakeLedgerClock()
    ledger, snapshot, chosen, preferred = _spill_fixture(tmp_path, clock)
    with pytest.raises(GatewayLedgerError, match="display-safe"):
        ledger.start_attempt(
            snapshot=snapshot,
            deployment=chosen,
            attempt_ordinal=0,
            route_depth=1,
            dispatch_reason="queue\nbound",
            preferred_deployment=preferred,
        )
