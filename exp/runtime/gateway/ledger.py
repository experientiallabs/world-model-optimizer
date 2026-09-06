"""Content-free SQLite request, attempt, recovery, and usage accounting."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.auth import utc_text
from exp.runtime.gateway.budgets import (
    MAXIMUM_MICRO_USD,
    budget_period_start,
    current_budget_period,
    require_attempt_budget,
    settle_attempt_budgets,
)
from exp.runtime.gateway.contracts import (
    AttemptId,
    AuthorizationSnapshot,
    ExecutionSnapshot,
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayUsage,
)
from exp.runtime.gateway.interfaces import GatewayClock
from exp.runtime.gateway.ledger_usage import (
    BillingSourceUsage,
    IdentityUsage,
    LedgerUsageSnapshot,
    billing_source_usage_rows,
    identity_usage_rows,
)
from exp.runtime.gateway.ledger_valuation import estimated_cost_micro_usd, optional_int
from exp.runtime.gateway.sqlite.migrations import initialize_database, persistent_connection
from exp.runtime.gateway.sqlite.store import SystemGatewayClock


class GatewayLedgerError(ValueError):
    """A request or attempt transition violates the content-free ledger contract."""


class AttemptRejectedError(GatewayLedgerError):
    """A typed pre-dispatch rejection raised by ``start_attempt``.

    The rejected reservation wrote nothing durable, so the executor must not
    latch accounting health, must not dispatch a provider, and must not advance
    the fallback waterfall; the exception reaches the protocol boundary
    unchanged so the rejection keeps its own public error shape. ``failure`` is
    the sanitized failure that settles the already-accepted parent request.
    """

    def __init__(self, message: str, *, failure: GatewayFailure) -> None:
        """Retain the internal message and the sanitized settlement failure.

        Args:
            message: Internal diagnostic message, never shown to callers.
            failure: Sanitized failure persisted on the accepted request and,
                absent a more specific boundary mapping, shown to the caller.
        """
        super().__init__(message)
        self.failure = failure


class IdempotencyConflictError(AttemptRejectedError):
    """A caller operation key was reused with different canonical request content."""

    def __init__(self, message: str) -> None:
        """Retain the message with the canonical caller-error settlement shape."""
        super().__init__(
            message,
            failure=GatewayFailure(
                failure_class=GatewayFailureClass.INVALID_REQUEST,
                safe_message="caller operation key was reused with different request content",
            ),
        )


class IdempotencyReplayUnavailableError(AttemptRejectedError):
    """A completed or accepted keyed request exists but its content cannot be replayed."""

    def __init__(self, message: str) -> None:
        """Retain the message with the canonical replay-loss settlement shape."""
        super().__init__(
            message,
            failure=GatewayFailure(
                failure_class=GatewayFailureClass.INTERNAL,
                safe_message="completed keyed result is unavailable for durable replay",
            ),
        )


class SQLiteAttemptLedger:
    """Durable content-free attempt ledger sharing the gateway control database."""

    def __init__(
        self,
        database_path: Path,
        *,
        clock: GatewayClock | None = None,
        busy_timeout_ms: int = 5_000,
    ) -> None:
        """Initialize a ledger on an existing or new gateway database.

        Args:
            database_path: Shared gateway SQLite path.
            clock: Injectable wall and monotonic clock.
            busy_timeout_ms: Maximum SQLite lock wait.
        """
        self.database_path = database_path
        self._clock = SystemGatewayClock() if clock is None else clock
        self._busy_timeout_ms = busy_timeout_ms
        initialize_database(database_path, busy_timeout_ms=busy_timeout_ms)

    @property
    def busy_timeout_ms(self) -> int:
        """Return the configured SQLite lock-wait bound."""
        return self._busy_timeout_ms

    def accept_request(self, *, authorization: AuthorizationSnapshot) -> None:
        """Persist accepted authority before route selection or dispatch.

        Args:
            authorization: Frozen authority and request identity.

        Raises:
            IdempotencyConflictError: The caller operation exists for another request.
            IdempotencyReplayUnavailableError: The matching operation already exists.
        """
        with self._transaction() as connection:
            self.apply_accept_request(connection, authorization=authorization)

    def apply_accept_request(
        self,
        connection: sqlite3.Connection,
        *,
        authorization: AuthorizationSnapshot,
    ) -> None:
        """Run the acceptance write inside the caller's open write transaction.

        Args:
            connection: Open write transaction owned by the caller.
            authorization: Frozen authority and request identity.

        Raises:
            IdempotencyConflictError: The caller operation exists for another request.
            IdempotencyReplayUnavailableError: The matching operation already exists.
        """
        now = self._clock.now()
        remaining = max(0.0, authorization.deadline_monotonic - self._clock.monotonic())
        deadline_at = now + timedelta(seconds=remaining)
        if authorization.caller_operation_sha256 is not None:
            prior = connection.execute(
                """
                SELECT canonical_request_sha256, terminal_state
                FROM gateway_requests
                WHERE organization_id = ? AND identity_id = ?
                  AND alias_revision_id = ? AND api_surface = ?
                  AND caller_operation_sha256 = ?
                ORDER BY accepted_at DESC LIMIT 1
                """,
                (
                    authorization.organization_id,
                    authorization.identity_id,
                    authorization.alias_revision_id,
                    authorization.surface.value,
                    authorization.caller_operation_sha256,
                ),
            ).fetchone()
            if prior is not None:
                if str(prior["canonical_request_sha256"]) != (
                    authorization.canonical_request_sha256
                ):
                    # Deliberately fail closed even when the prior attempt
                    # failed: after an ambiguous failure the provider may
                    # have executed, so different content under one
                    # operation identity is a client bug the key exists to
                    # surface. Retrying different content needs a new key.
                    raise IdempotencyConflictError(
                        "caller operation key was reused with different request content"
                    )
                if str(prior["terminal_state"]) not in {
                    "expired_before_dispatch",
                    "unknown_after_crash",
                }:
                    raise IdempotencyReplayUnavailableError(
                        "matching keyed request exists but durable content replay is unavailable"
                    )
        alias_row = connection.execute(
            """
            SELECT alias_id FROM alias_revisions
            WHERE organization_id = ? AND revision_id = ?
            """,
            (authorization.organization_id, authorization.alias_revision_id),
        ).fetchone()
        if alias_row is None:
            raise GatewayLedgerError("authorized alias revision is not present in the ledger")
        connection.execute(
            """
            INSERT INTO gateway_requests (
                request_id, organization_id, identity_id, key_id, alias_id,
                alias_revision_id, api_surface, canonical_request_sha256,
                caller_operation_sha256, accepted_at, deadline_at,
                app_referer, app_title
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                authorization.request_id,
                authorization.organization_id,
                authorization.identity_id,
                authorization.virtual_key_id,
                str(alias_row["alias_id"]),
                authorization.alias_revision_id,
                authorization.surface.value,
                authorization.canonical_request_sha256,
                authorization.caller_operation_sha256,
                utc_text(now),
                utc_text(deadline_at),
                authorization.app_referer,
                authorization.app_title,
            ),
        )

    def start_attempt(
        self,
        *,
        snapshot: ExecutionSnapshot,
        deployment: ExactModelDeployment,
        attempt_ordinal: int,
        route_depth: int,
        maximum_cost_micro_usd: int | None = None,
        route_reason: str | None = None,
        fallback_reason: str | None = None,
        dispatch_reason: str | None = None,
        preferred_deployment: ExactModelDeployment | None = None,
    ) -> AttemptId:
        """Durably mark a provider dispatch before starting network work.

        Args:
            snapshot: Route-bound immutable request plan.
            deployment: Exact deployment about to receive the request.
            attempt_ordinal: Zero-based physical dispatch position for this request.
            route_depth: Zero-based operational route position.
            maximum_cost_micro_usd: Conservative charge reserved before dispatch.
            route_reason: Optional learned-selection reason code.
            fallback_reason: Optional embedding or router fallback reason code.
            dispatch_reason: Optional policy-dispatch disclosure code.
            preferred_deployment: The route's bypassed preferred rung, given
                only when it differs from ``deployment``; its base rates are
                frozen for the settle-time counterfactual cost.

        Returns:
            Stable new attempt ID.
        """
        with self._transaction() as connection:
            return self.apply_start_attempt(
                connection,
                snapshot=snapshot,
                deployment=deployment,
                attempt_ordinal=attempt_ordinal,
                route_depth=route_depth,
                maximum_cost_micro_usd=maximum_cost_micro_usd,
                route_reason=route_reason,
                fallback_reason=fallback_reason,
                dispatch_reason=dispatch_reason,
                preferred_deployment=preferred_deployment,
            )

    def apply_start_attempt(
        self,
        connection: sqlite3.Connection,
        *,
        snapshot: ExecutionSnapshot,
        deployment: ExactModelDeployment,
        attempt_ordinal: int,
        route_depth: int,
        maximum_cost_micro_usd: int | None = None,
        route_reason: str | None = None,
        fallback_reason: str | None = None,
        dispatch_reason: str | None = None,
        preferred_deployment: ExactModelDeployment | None = None,
    ) -> AttemptId:
        """Run the dispatch reservation inside the caller's open write transaction.

        Args:
            connection: Open write transaction owned by the caller.
            snapshot: Route-bound immutable request plan.
            deployment: Exact deployment about to receive the request.
            attempt_ordinal: Zero-based physical dispatch position for this request.
            route_depth: Zero-based operational route position.
            maximum_cost_micro_usd: Conservative charge reserved before dispatch.
            route_reason: Optional learned-selection reason code.
            fallback_reason: Optional embedding or router fallback reason code.
            dispatch_reason: Optional policy-dispatch disclosure code.
            preferred_deployment: The route's bypassed preferred rung, given
                only when it differs from ``deployment``; its base rates are
                frozen for the settle-time counterfactual cost.

        Returns:
            Stable new attempt ID.
        """
        for value in (route_reason, fallback_reason, dispatch_reason):
            if value is not None and (len(value) > 512 or any(ord(char) < 32 for char in value)):
                raise GatewayLedgerError("route context must be a short display-safe code")
        if deployment.deployment_id not in snapshot.deployment_ids:
            raise GatewayLedgerError("attempt deployment is absent from the execution snapshot")
        if deployment.exact_model_id != snapshot.exact_model_id:
            raise GatewayLedgerError("attempt deployment changes the selected exact model")
        if (
            preferred_deployment is not None
            and preferred_deployment.deployment_id == deployment.deployment_id
        ):
            raise GatewayLedgerError("a preferred rung disclosure requires a divergent rung")
        if maximum_cost_micro_usd is not None and not (
            0 <= maximum_cost_micro_usd <= MAXIMUM_MICRO_USD
        ):
            raise GatewayLedgerError("maximum attempt cost must fit a nonnegative SQLite integer")
        attempt_id = f"attempt-{uuid.uuid4().hex}"
        prices = deployment.gateway.prices
        now = self._clock.now()
        period_start = budget_period_start(current_budget_period(now))
        request = connection.execute(
            """
            SELECT organization_id, identity_id, alias_id, terminal_state
            FROM gateway_requests
            WHERE request_id = ?
            """,
            (snapshot.authorization.request_id,),
        ).fetchone()
        if request is None:
            raise GatewayLedgerError("attempt request was not durably accepted")
        if str(request["organization_id"]) != snapshot.authorization.organization_id:
            raise GatewayLedgerError("attempt authority differs from accepted request")
        if request["terminal_state"] is not None:
            raise GatewayLedgerError("attempt request is already terminal")
        connection.execute(
            """
            INSERT INTO gateway_attempts (
                attempt_id, request_id, organization_id, attempt_ordinal, route_depth,
                deployment_id, provider, exact_model_id, pool_id, catalog_sha256,
                billing_source,
                pricing_source, pricing_effective_at,
                input_rate, cached_input_rate, output_rate, reasoning_rate,
                long_context_threshold_tokens, long_context_input_rate,
                long_context_cached_input_rate, long_context_output_rate,
                long_context_reasoning_rate,
                route_reason, fallback_reason,
                dispatch_reason, preferred_deployment_id,
                preferred_input_rate, preferred_cached_input_rate,
                preferred_output_rate, preferred_reasoning_rate,
                state, started_at, budget_period_start, budget_reserved_micro_usd
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                'dispatched', ?, ?, ?
            )
            """,
            (
                attempt_id,
                snapshot.authorization.request_id,
                snapshot.authorization.organization_id,
                attempt_ordinal,
                route_depth,
                deployment.deployment_id,
                deployment.provider,
                snapshot.exact_model_id,
                snapshot.pool_id,
                snapshot.authorization.catalog_sha256,
                deployment.billing_source.value,
                deployment.gateway.pricing_source,
                (
                    None
                    if deployment.gateway.pricing_effective_at is None
                    else utc_text(deployment.gateway.pricing_effective_at)
                ),
                prices.input_micro_usd_per_million_tokens,
                prices.cached_input_micro_usd_per_million_tokens,
                prices.output_micro_usd_per_million_tokens,
                prices.reasoning_micro_usd_per_million_tokens,
                (
                    None
                    if prices.long_context is None
                    else prices.long_context.input_threshold_tokens
                ),
                (
                    None
                    if prices.long_context is None
                    else prices.long_context.input_micro_usd_per_million_tokens
                ),
                (
                    None
                    if prices.long_context is None
                    else prices.long_context.cached_input_micro_usd_per_million_tokens
                ),
                (
                    None
                    if prices.long_context is None
                    else prices.long_context.output_micro_usd_per_million_tokens
                ),
                (
                    None
                    if prices.long_context is None
                    else prices.long_context.reasoning_micro_usd_per_million_tokens
                ),
                route_reason,
                fallback_reason,
                dispatch_reason,
                None if preferred_deployment is None else preferred_deployment.deployment_id,
                None
                if preferred_deployment is None
                else preferred_deployment.gateway.prices.input_micro_usd_per_million_tokens,
                None
                if preferred_deployment is None
                else preferred_deployment.gateway.prices.cached_input_micro_usd_per_million_tokens,
                None
                if preferred_deployment is None
                else preferred_deployment.gateway.prices.output_micro_usd_per_million_tokens,
                None
                if preferred_deployment is None
                else preferred_deployment.gateway.prices.reasoning_micro_usd_per_million_tokens,
                utc_text(now),
                period_start,
                maximum_cost_micro_usd,
            ),
        )
        require_attempt_budget(
            connection,
            organization_id=snapshot.authorization.organization_id,
            identity_id=str(request["identity_id"]),
            alias_id=str(request["alias_id"]),
            pool_id=snapshot.pool_id,
            deployment_id=deployment.deployment_id,
            attempt_id=attempt_id,
            period_start=period_start,
            maximum_cost_micro_usd=maximum_cost_micro_usd,
        )
        return attempt_id

    def finish_attempt(
        self,
        *,
        attempt_id: AttemptId,
        terminal_event: GatewayEvent | None,
        failure: GatewayFailure | None,
        finalize_request: bool = True,
        first_token_at: datetime | None = None,
    ) -> None:
        """Idempotently settle one attempt with normalized content-free fields.

        Args:
            attempt_id: Stable attempt ID.
            terminal_event: Provider terminal event, possibly carrying usage.
            failure: Sanitized failure when no successful terminal event exists.
            finalize_request: Whether this attempt is the final route for its parent request.
            first_token_at: Wall-clock time the attempt streamed its first token, or ``None``.
        """
        with self._transaction() as connection:
            self.apply_finish_attempt(
                connection,
                attempt_id=attempt_id,
                terminal_event=terminal_event,
                failure=failure,
                finalize_request=finalize_request,
                first_token_at=first_token_at,
            )

    def apply_finish_attempt(
        self,
        connection: sqlite3.Connection,
        *,
        attempt_id: AttemptId,
        terminal_event: GatewayEvent | None,
        failure: GatewayFailure | None,
        finalize_request: bool = True,
        first_token_at: datetime | None = None,
    ) -> None:
        """Run the attempt settlement inside the caller's open write transaction.

        Args:
            connection: Open write transaction owned by the caller.
            attempt_id: Stable attempt ID.
            terminal_event: Provider terminal event, possibly carrying usage.
            failure: Sanitized failure when no successful terminal event exists.
            finalize_request: Whether this attempt is the final route for its parent request.
            first_token_at: Wall-clock time the attempt streamed its first token, or ``None``.
        """
        state, normalized_failure, failure_message, usage = _terminal_values(
            terminal_event, failure
        )
        row = connection.execute(
            """
            SELECT request_id, state, input_rate, cached_input_rate,
                   output_rate, reasoning_rate,
                   long_context_threshold_tokens, long_context_input_rate,
                   long_context_cached_input_rate, long_context_output_rate,
                   long_context_reasoning_rate, budget_reserved_micro_usd,
                   preferred_deployment_id, preferred_input_rate,
                   preferred_cached_input_rate, preferred_output_rate,
                   preferred_reasoning_rate
            FROM gateway_attempts WHERE attempt_id = ?
            """,
            (attempt_id,),
        ).fetchone()
        if row is None:
            raise GatewayLedgerError("attempt does not exist")
        current_state = str(row["state"])
        if current_state != "dispatched":
            if current_state == state:
                return
            raise GatewayLedgerError("attempt is already settled with another terminal state")
        # Both published tier schedules reprice the WHOLE request once
        # provider-reported input tokens reach the frozen threshold, so the
        # tier's rates replace the base rates rather than composing with them.
        threshold = optional_int(row["long_context_threshold_tokens"])
        long_context = (
            threshold is not None
            and usage is not None
            and usage.input_tokens is not None
            and usage.input_tokens >= threshold
        )
        prefix = "long_context_" if long_context else ""
        cost = estimated_cost_micro_usd(
            usage,
            input_rate=optional_int(row[f"{prefix}input_rate"]),
            cached_input_rate=optional_int(row[f"{prefix}cached_input_rate"]),
            output_rate=optional_int(row[f"{prefix}output_rate"]),
            reasoning_rate=optional_int(row[f"{prefix}reasoning_rate"]),
        )
        budget_settlement = (
            cost if cost is not None else optional_int(row["budget_reserved_micro_usd"])
        )
        if budget_settlement is not None and budget_settlement > MAXIMUM_MICRO_USD:
            raise GatewayLedgerError("attempt cost exceeds SQLite integer capacity")
        # Cost-optimality counterfactual: the SAME observed usage priced at the
        # bypassed preferred rung's frozen BASE rates (long-context tiers are
        # deliberately not modeled here; this is telemetry, never billing). A
        # missing preferred rate yields NULL rather than a guess.
        counterfactual_cost = (
            None
            if row["preferred_deployment_id"] is None
            else estimated_cost_micro_usd(
                usage,
                input_rate=optional_int(row["preferred_input_rate"]),
                cached_input_rate=optional_int(row["preferred_cached_input_rate"]),
                output_rate=optional_int(row["preferred_output_rate"]),
                reasoning_rate=optional_int(row["preferred_reasoning_rate"]),
            )
        )
        terminal_at = utc_text(self._clock.now())
        connection.execute(
            """
            UPDATE gateway_attempts
            SET state = ?, terminal_at = ?, first_token_at = ?, failure_class = ?,
                failure_message = ?,
                input_tokens = ?, cached_input_tokens = ?, output_tokens = ?,
                reasoning_tokens = ?, usage_source = ?, estimated_cost_micro_usd = ?,
                counterfactual_cost_micro_usd = ?,
                budget_settled_micro_usd = ?
            WHERE attempt_id = ? AND state = 'dispatched'
            """,
            (
                state,
                terminal_at,
                None if first_token_at is None else utc_text(first_token_at),
                normalized_failure,
                failure_message,
                None if usage is None else usage.input_tokens,
                None if usage is None else usage.cached_input_tokens,
                None if usage is None else usage.output_tokens,
                None if usage is None else usage.reasoning_tokens,
                "unknown" if usage is None else "observed",
                cost,
                counterfactual_cost,
                budget_settlement,
                attempt_id,
            ),
        )
        settle_attempt_budgets(
            connection,
            attempt_id=attempt_id,
            settled_micro_usd=budget_settlement,
        )
        if finalize_request and state in {"completed", "failed", "cancelled", "incomplete"}:
            connection.execute(
                """
                UPDATE gateway_requests SET terminal_state = ?, terminal_at = ?
                WHERE request_id = ? AND terminal_state IS NULL
                """,
                (state, terminal_at, str(row["request_id"])),
            )

    def finish_request(
        self,
        *,
        authorization: AuthorizationSnapshot,
        failure: GatewayFailure,
    ) -> None:
        """Idempotently terminalize accepted work that never reached dispatch.

        Args:
            authorization: Frozen authority identifying the accepted request.
            failure: Sanitized pre-dispatch terminal failure.
        """
        with self._transaction() as connection:
            self.apply_finish_request(connection, authorization=authorization, failure=failure)

    def apply_finish_request(
        self,
        connection: sqlite3.Connection,
        *,
        authorization: AuthorizationSnapshot,
        failure: GatewayFailure,
    ) -> None:
        """Run the pre-dispatch settlement inside the caller's open write transaction.

        Args:
            connection: Open write transaction owned by the caller.
            authorization: Frozen authority identifying the accepted request.
            failure: Sanitized pre-dispatch terminal failure.
        """
        state, normalized_failure, _failure_message, _ = _terminal_values(None, failure)
        del normalized_failure, _failure_message
        row = connection.execute(
            """
            SELECT organization_id, terminal_state FROM gateway_requests
            WHERE request_id = ?
            """,
            (authorization.request_id,),
        ).fetchone()
        if row is None:
            raise GatewayLedgerError("request was not durably accepted")
        if str(row["organization_id"]) != authorization.organization_id:
            raise GatewayLedgerError("request authority differs from accepted request")
        current = row["terminal_state"]
        if current is not None:
            if str(current) == state:
                return
            raise GatewayLedgerError("request is already settled with another terminal state")
        connection.execute(
            """
            UPDATE gateway_requests SET terminal_state = ?, terminal_at = ?
            WHERE request_id = ? AND terminal_state IS NULL
            """,
            (state, utc_text(self._clock.now()), authorization.request_id),
        )

    def reconcile_crashed_requests(self, *, cleanup_grace: timedelta) -> tuple[int, int]:
        """Settle expired pre-dispatch and dispatched work after a crash.

        Args:
            cleanup_grace: Additional bound allowed for upstream cleanup after deadline.

        Returns:
            Counts of expired pre-dispatch requests and unknown dispatched attempts.
        """
        if cleanup_grace < timedelta(0):
            raise ValueError("cleanup grace cannot be negative")
        now = self._clock.now()
        expired_requests = 0
        unknown_attempts = 0
        with self._transaction() as connection:
            request_rows = connection.execute(
                """
                SELECT r.request_id, r.deadline_at,
                       EXISTS(
                           SELECT 1 FROM gateway_attempts AS a WHERE a.request_id = r.request_id
                       ) AS has_attempt
                FROM gateway_requests AS r WHERE r.terminal_state IS NULL
                """
            ).fetchall()
            for request in request_rows:
                deadline = datetime.fromisoformat(str(request["deadline_at"]))
                if int(request["has_attempt"]) == 0 and deadline <= now:
                    connection.execute(
                        """
                        UPDATE gateway_requests
                        SET terminal_state = 'expired_before_dispatch', terminal_at = ?
                        WHERE request_id = ? AND terminal_state IS NULL
                        """,
                        (utc_text(now), str(request["request_id"])),
                    )
                    expired_requests += 1
            attempt_rows = connection.execute(
                """
                SELECT a.attempt_id, a.request_id, r.deadline_at
                FROM gateway_attempts AS a
                JOIN gateway_requests AS r ON r.request_id = a.request_id
                WHERE a.state = 'dispatched'
                """
            ).fetchall()
            for attempt in attempt_rows:
                deadline = datetime.fromisoformat(str(attempt["deadline_at"]))
                if deadline + cleanup_grace > now:
                    continue
                connection.execute(
                    """
                    UPDATE gateway_attempts
                    SET state = 'unknown_after_crash', terminal_at = ?,
                        usage_source = 'unknown'
                    WHERE attempt_id = ? AND state = 'dispatched'
                    """,
                    (utc_text(now), str(attempt["attempt_id"])),
                )
                connection.execute(
                    """
                    UPDATE gateway_requests
                    SET terminal_state = 'unknown_after_crash', terminal_at = ?
                    WHERE request_id = ? AND terminal_state IS NULL
                    """,
                    (utc_text(now), str(attempt["request_id"])),
                )
                unknown_attempts += 1
        return expired_requests, unknown_attempts

    def usage(
        self, *, organization_id: str, identity_id: str | None = None
    ) -> tuple[IdentityUsage, ...]:
        """Aggregate request, usage, cost, and terminal states by identity.

        Args:
            organization_id: Tenant boundary.
            identity_id: Optional identity filter.

        Returns:
            Stable identity usage rows without prompts or outputs.
        """
        return self.usage_snapshot(
            organization_id=organization_id,
            identity_id=identity_id,
        ).identities

    def usage_by_billing_source(
        self,
        *,
        organization_id: str,
        identity_id: str | None = None,
    ) -> tuple[BillingSourceUsage, ...]:
        """Aggregate physical attempts by their frozen credential ownership source.

        Args:
            organization_id: Tenant boundary.
            identity_id: Optional identity filter applied through the parent request.

        Returns:
            Deterministic source buckets without partitioning logical request counts.
        """
        return self.usage_snapshot(
            organization_id=organization_id,
            identity_id=identity_id,
        ).by_billing_source

    def usage_snapshot(
        self,
        *,
        organization_id: str,
        identity_id: str | None = None,
    ) -> LedgerUsageSnapshot:
        """Read identity and source aggregates from one explicit SQLite snapshot.

        Args:
            organization_id: Tenant boundary.
            identity_id: Optional exact identity filter.

        Returns:
            Internally conserving usage aggregates from one WAL read transaction.
        """
        parameters: tuple[str, ...]
        predicate = "i.organization_id = ?"
        source_predicate = "r.organization_id = ?"
        if identity_id is None:
            parameters = (organization_id,)
        else:
            predicate += " AND i.identity_id = ?"
            source_predicate += " AND r.identity_id = ?"
            parameters = (organization_id, identity_id)
        with self._connect() as connection:
            connection.execute("BEGIN")
            try:
                identities = identity_usage_rows(
                    connection,
                    organization_id=organization_id,
                    predicate=predicate,
                    parameters=parameters,
                )
                by_billing_source = billing_source_usage_rows(
                    connection,
                    predicate=source_predicate,
                    parameters=parameters,
                )
            finally:
                connection.rollback()
        return LedgerUsageSnapshot(
            identities=identities,
            by_billing_source=by_billing_source,
        )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Check out one reusable configured connection."""
        with persistent_connection(
            self.database_path, busy_timeout_ms=self._busy_timeout_ms
        ) as connection:
            yield connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Run one explicit immediate transaction with rollback on failure."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            else:
                connection.execute("COMMIT")


def _terminal_values(
    terminal_event: GatewayEvent | None,
    failure: GatewayFailure | None,
) -> tuple[str, str | None, str | None, GatewayUsage | None]:
    """Normalize one finish call to state, failure class, message, and usage.

    The failure message is the provider's own sanitized explanation
    (``provider_detail``); it is present only for a client-error rejection and
    is the same bounded, credential-free sentence the caller already receives.
    """
    event_failure = None if terminal_event is None else terminal_event.failure
    normalized = failure or event_failure
    if terminal_event is None and normalized is None:
        raise GatewayLedgerError("attempt finish needs a terminal event or failure")
    if terminal_event is not None and terminal_event.kind not in {
        GatewayEventKind.COMPLETED,
        GatewayEventKind.INCOMPLETE,
        GatewayEventKind.FAILED,
    }:
        raise GatewayLedgerError("attempt finish event must be terminal")
    if normalized is not None:
        state = (
            "cancelled" if normalized.failure_class == GatewayFailureClass.CANCELLED else "failed"
        )
        return (
            state,
            normalized.failure_class.value,
            normalized.provider_detail,
            (None if terminal_event is None else terminal_event.usage),
        )
    assert terminal_event is not None
    return terminal_event.kind.value, None, None, terminal_event.usage
