"""Durable attempt accounting for the native data plane's waterfall.

One registry owns every admitted request from acceptance to its terminal
settlement: it reserves each physical dispatch (``start_attempt``), lands each
attempt's durable terminal (``settle``), terminalizes requests with no
settleable outcome (``abandon``), and sweeps retained settlements and
abandoned reservations on a timer. Candidate selection enforces the frozen
waterfall policy, including deployment-health circuits and per-deployment
budget skipping. Every method takes and returns one JSON
string, matching the bridge boundary.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections.abc import Callable
from typing import cast

from exp.common.core.artifacts import JsonObject
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.boundary import boundary_protocol_error
from exp.runtime.gateway.budgets import (
    BudgetReservationRejected,
    BudgetScopeKind,
    maximum_attempt_cost_micro_usd,
)
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
)
from exp.runtime.gateway.health import DeploymentHealthRegistry
from exp.runtime.gateway.native_components import SyncWriteLedger
from exp.runtime.gateway.native_execution import (
    DeadRung,
    InflightRequest,
    claim_route_from,
    deployment_health_key,
    next_route_candidate,
)
from exp.runtime.gateway.native_settlement import (
    budget_quota_protocol_error,
    first_token_at_from_settlement,
    ledger_failure,
    terminal_from_settlement,
)
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.gateway.rung_admission import RungLoadRegistry, RungShed
from exp.runtime.openai_protocol.errors import (
    THROTTLED_RETRY_AFTER_SECONDS,
    OpenAIProtocolError,
    public_failure_error,
)

_SWEEP_GRACE_SECONDS = 5.0
_SWEEP_INTERVAL_SECONDS = 5.0
_SWEEP_BATCH = 16
_logger = logging.getLogger(__name__)


class NativeBridgeError(Exception):
    """One sanitized boundary failure delivered to the native data plane."""

    def __init__(self, error: OpenAIProtocolError) -> None:
        """Retain the public error as the JSON payload the data plane returns.

        Args:
            error: Sanitized protocol error carrying its HTTP representation.
        """
        super().__init__(error.detail.message)
        self.public_error_json = json.dumps(
            {
                "status_code": error.status_code,
                "code": error.detail.code,
                "message": error.detail.message,
                "error_type": error.detail.type,
                "param": error.detail.param,
                "retry_after_seconds": error.retry_after_seconds,
            },
            separators=(",", ":"),
        )


def authority_error(exception: Exception) -> NativeBridgeError:
    """Map boundary failures through the shared service-layer mapper.

    Args:
        exception: Store, grant, routing, or execution failure.

    Returns:
        A boundary error carrying the matching public OpenAI error.
    """
    return NativeBridgeError(boundary_protocol_error(exception))


def internal_protocol_error() -> OpenAIProtocolError:
    """Return the public internal error for a broken data-plane wire contract."""
    return OpenAIProtocolError(
        status_code=500,
        code="internal_error",
        message="The gateway request failed.",
        error_type="api_error",
    )


def budget_quota_failure() -> GatewayFailure:
    """Return the sanitized quota failure after no route can reserve its cost."""
    return GatewayFailure(
        failure_class=GatewayFailureClass.QUOTA_EXCEEDED,
        safe_message="monthly gateway allocation is exhausted",
    )


def all_routes_unavailable_failure() -> GatewayFailure:
    """Return the sanitized terminal failure for an exhausted certified pool."""
    return GatewayFailure(
        failure_class=GatewayFailureClass.PROVIDER_INTERNAL,
        safe_message="all exact-model deployments are unavailable",
    )


def all_routes_throttled_failure(remaining_seconds: float) -> GatewayFailure:
    """Return the throttle-window failure for a route the provider backed off.

    Every deployment sitting inside a provider throttle window is caller-facing
    rate limiting (the provider answered 429 and asked for backoff), not
    platform deadness: classing it provider_internal misfiled 429 storms as
    outages and paged operators for caller-driven load (2026-09-04 ledger,
    deepseek-v4-flash-vision-exp). One computed wait (the remaining window,
    floored at the default throttle backoff) rides both the message and
    ``retry_after_seconds`` so the Retry-After header a client honors never
    disagrees with the sentence it reads.

    Args:
        remaining_seconds: Longest remaining throttle window across the route.

    Returns:
        Sanitized throttled failure naming the retry window.
    """
    seconds = max(THROTTLED_RETRY_AFTER_SECONDS, math.ceil(remaining_seconds))
    return GatewayFailure(
        failure_class=GatewayFailureClass.THROTTLED,
        safe_message=(
            "all exact-model deployments are inside a provider throttle window; "
            f"retry in {seconds}s"
        ),
        retry_after_seconds=seconds,
    )


def gateway_updating_failure() -> GatewayFailure:
    """Return the sanitized retryable failure for a transient roll condition.

    A pod that cannot build the authorized catalog revision during a rolling
    deploy (a snapshot authored by another engine version it cannot reconcile)
    surfaces this instead of a closed INTERNAL: the condition clears on its own
    once the roll settles, so the honest answer is a retryable 503, never a bug
    signal that pages or opens a deployment circuit.
    """
    return GatewayFailure(
        failure_class=GatewayFailureClass.UNAVAILABLE,
        safe_message="the gateway is updating; retry the request",
    )


def _failure_from_payload(payload: object) -> GatewayFailure | None:
    """Parse one optional classified failure from a boundary payload."""
    if not isinstance(payload, dict):
        return None
    data = cast("JsonObject", payload)
    rejected_parameter = data.get("rejected_parameter")
    provider_detail = data.get("provider_detail")
    return GatewayFailure(
        failure_class=GatewayFailureClass(str(data["failure_class"])),
        safe_message=str(data["safe_message"]),
        retryable_same_deployment=bool(data.get("retryable_same_deployment", False)),
        failover_eligible=bool(data.get("failover_eligible", False)),
        rejected_parameter=(
            rejected_parameter
            if isinstance(rejected_parameter, str) and rejected_parameter
            else None
        ),
        provider_detail=(
            provider_detail if isinstance(provider_detail, str) and provider_detail else None
        ),
        customer_owned=data.get("customer_owned") is True,
    )


def _deployment_priced_for_service_tier(
    deployment: ExactModelDeployment,
    service_tier: str | None,
    *,
    forwards_tier: bool,
) -> ExactModelDeployment:
    """Reprice one deployment for a requested flex/priority processing tier.

    v1 bills the REQUESTED tier: when the SELECTED candidate actually FORWARDS
    the tier to its provider and carries a pass-through card for it, the card's
    rates replace the base schedule on a copy used only for THIS reservation, so
    the ceiling, the stored per-token rates, and settlement all bill the tier
    transparently. ``forwards_tier`` is the admission-time forwarding decision
    for this exact depth (``GatewayWireProfile.forwards_tier``); gating on it
    keeps FORWARD and BILL consistent even if a card ever sits on a lane whose
    wire would strip the tier (non-tier dialect, tier disabled) — such a depth
    runs the provider's base schedule, so it must bill the base schedule too. No
    tier, no forwarding, or no card returns the deployment unchanged. The copy
    stays Python-side and never crosses the native boundary.
    """
    if not forwards_tier:
        return deployment
    effective = deployment.gateway.prices.for_service_tier(service_tier)
    if effective is deployment.gateway.prices:
        return deployment
    return deployment.model_copy(
        update={"gateway": deployment.gateway.model_copy(update={"prices": effective})}
    )


class NativeAttemptAccounting:
    """Registry of admitted requests and their durable attempt settlements.

    Methods are called from multiple Rust worker threads; the registry is
    guarded by one lock and swept both opportunistically and on a timer so an
    abandoned reservation cannot outlive its request deadline by more than
    the sweep grace. A lost durable terminal write latches the registry
    unhealthy so readiness fails until the next startup reconciliation.
    """

    def __init__(
        self,
        write_ledger: SyncWriteLedger,
        *,
        budget_error_factory: Callable[[str], NativeBridgeError] | None = None,
    ) -> None:
        """Bind the durable ledger and start the settlement sweep.

        Args:
            write_ledger: Blocking durable request and attempt ledger.
            budget_error_factory: Optional hosted mapping for a rejected
                reservation.
        """
        self._write_ledger = write_ledger
        self._budget_error_factory = budget_error_factory
        # The native waterfall's deployment-health circuits, revision-scoped
        # to the traffic this plane serves.
        self._health = DeploymentHealthRegistry()
        # Per-worker in-flight counters for rungs that author a dispatch
        # policy (concurrency bound, weighted fair share); pure in-memory
        # arithmetic, physical-lane scoped so counters survive catalog rolls.
        self._loads = RungLoadRegistry()
        self._inflight: dict[str, InflightRequest] = {}
        self._lock = threading.Lock()
        self._accounting_healthy = True
        self._sweep_retained_replayed = 0
        self._sweep_abandoned_cancelled = 0
        # Admission-time dead-rung skips: a request served off a fallback
        # because a certified rung could not be resolved for dispatch, and the
        # subset of those where the skipped rung was the lead.
        self._admission_dead_rungs_skipped = 0
        self._admission_parameter_coercions = 0
        self._admission_lead_rungs_skipped = 0
        # Dispatch-policy outcomes: rungs bypassed at their bound or their
        # organization's fair share, and dispatches forced past a bound
        # because no other rung could serve.
        self._rung_admission_sheds = 0
        self._rung_saturated_overflows = 0
        # The sweep also runs on a timer so retained settlements and abandoned
        # attempts are recovered even when no further requests arrive.
        self._sweeper = threading.Thread(
            target=self._sweep_loop,
            name="exp-native-settlement-sweep",
            daemon=True,
        )
        self._sweeper.start()

    @property
    def accounting_healthy(self) -> bool:
        """Return whether every durable terminal write has landed."""
        return self._accounting_healthy

    @property
    def health(self) -> DeploymentHealthRegistry:
        """Return the native waterfall's deployment-health circuits."""
        return self._health

    @property
    def loads(self) -> RungLoadRegistry:
        """Return the per-worker rung in-flight registry for bounded admission."""
        return self._loads

    def _reserve_rung_slot(
        self,
        entry: InflightRequest,
        deployment: ExactModelDeployment,
        *,
        force: bool,
    ) -> str | RungShed | None:
        """Reserve one policy-bounded slot on a rung, or report the shed.

        Args:
            entry: The owning in-flight request (organization and weight).
            deployment: The claimed rung about to dispatch.
            force: Admit past the bound because no other rung can serve.

        Returns:
            An opaque reservation ticket, the shed disclosure, or ``None``
            when the rung authors no dispatch policy (the untouched default).
        """
        policy = deployment.gateway.dispatch
        if policy is None or policy.concurrency_bound is None:
            return None
        result = self._loads.reserve(
            (deployment.deployment_id, deployment.connection_sha256),
            organization_id=entry.authorization.organization_id,
            weight=entry.authorization.fair_share_weight,
            bound=policy.concurrency_bound,
            fair_share=policy.fair_share,
            force=force,
        )
        if isinstance(result, RungShed):
            with self._lock:
                self._rung_admission_sheds += 1
        return result

    def rung_admission_counters(self) -> tuple[int, int]:
        """Return ``(sheds, saturated_overflows)`` for the metrics snapshot."""
        with self._lock:
            return (self._rung_admission_sheds, self._rung_saturated_overflows)

    def mark_unhealthy(self) -> None:
        """Latch an unhealthy state after a lost terminal accounting write."""
        self._accounting_healthy = False

    def counters(self) -> tuple[int, int, int]:
        """Return sweep recoveries and registry size for the metrics snapshot."""
        with self._lock:
            return (
                self._sweep_retained_replayed,
                self._sweep_abandoned_cancelled,
                len(self._inflight),
            )

    def record_admission_coercions(self, count: int) -> None:
        """Count disclosed request coercions applied at admission.

        Args:
            count: Number of disclosed substitutions on one admission.
        """
        with self._lock:
            self._admission_parameter_coercions += count

    def admission_parameter_coercions(self) -> int:
        """Return the total disclosed request coercions for metrics."""
        with self._lock:
            return self._admission_parameter_coercions

    def record_admission_rung_skips(self, dead_count: int, *, lead_skipped: bool) -> None:
        """Count admission-time dead-rung skips for the metrics snapshot.

        Args:
            dead_count: Number of certified rungs skipped as dead at admission.
            lead_skipped: Whether the skipped set included the lead rung.
        """
        with self._lock:
            self._admission_dead_rungs_skipped += dead_count
            if lead_skipped:
                self._admission_lead_rungs_skipped += 1

    def admission_rung_skips(self) -> tuple[int, int]:
        """Return ``(lead_rungs_skipped, dead_rungs_skipped)`` for metrics."""
        with self._lock:
            return (self._admission_lead_rungs_skipped, self._admission_dead_rungs_skipped)

    def register(self, entry: InflightRequest) -> None:
        """Track one accepted request until its terminal settlement."""
        with self._lock:
            self._inflight[entry.authorization.request_id] = entry

    def entry(self, request_id: str) -> InflightRequest | None:
        """Return one tracked request, or ``None`` after settlement."""
        with self._lock:
            return self._inflight.get(request_id)

    def start_attempt(self, argument: str) -> str:
        """Reserve one physical dispatch immediately before network work.

        Candidate selection mirrors the executor: the first dispatch claims
        the first healthy route in authored order (with bounded last-resort
        and forced claims through open circuits), a classified failure either
        redials the same deployment or advances to the next claimable one,
        and a deployment whose hard monthly allocation cannot admit this call
        is skipped. Exhaustion finalizes the accepted request here, so the
        data plane only has to answer with the last classified failure.

        Args:
            argument: JSON object with ``request_id``, ``attempt_ordinal``
                (the count of physical dispatches already reserved), optional
                ``current_depth`` (the route position of the failed dispatch,
                absent for the first), and the optional classified
                ``failure`` with its ``retryable_same_deployment`` and
                ``failover_eligible`` flags.

        Returns:
            ``{"attempt_id", "route_depth"}`` for one durably reserved
            dispatch, or ``{"exhausted": true, "failure": {...}}`` after the
            request was finalized with that failure.

        Raises:
            NativeBridgeError: The request is unknown, its deadline passed, a
                non-deployment budget scope rejected the reservation, or the
                reservation write failed; the request is finalized before the
                error is raised.
        """
        data = json.loads(argument)
        request_id = str(data["request_id"])
        with self._lock:
            entry = self._inflight.get(request_id)
        if entry is None or int(data["attempt_ordinal"]) != entry.total_attempts:
            raise NativeBridgeError(internal_protocol_error())
        route = entry.route
        keys = tuple(
            deployment_health_key(entry.authorization, deployment)
            for deployment in route.deployments
        )
        if time.monotonic() >= entry.deadline_monotonic:
            failure = GatewayFailure(
                failure_class=GatewayFailureClass.TIMEOUT,
                safe_message="gateway execution deadline exceeded",
            )
            self.finish_request_quietly(entry.authorization, failure)
            with self._lock:
                self._inflight.pop(request_id, None)
            raise NativeBridgeError(public_failure_error(failure))
        failure = _failure_from_payload(data.get("failure"))
        current_depth = data.get("current_depth")
        if failure is not None and isinstance(current_depth, int):
            candidate = next_route_candidate(
                health=self._health,
                keys=keys,
                failure=failure,
                current_depth=current_depth,
                attempt_counts=entry.attempt_counts,
                total_attempts=entry.total_attempts,
                refusal_failover=entry.authorization.refusal_failover,
                failover_mode=route.snapshot.failover_mode,
            )
            last_failure: GatewayFailure | None = failure
        else:
            candidate = claim_route_from(self._health, keys, 0)
            last_failure = None
        # Rung dispatch policies shed a claimed rung SIDEWAYS to the next
        # claimable one instead of queueing on it (spill in seconds, never a
        # deadline death). Each shed is remembered so the dispatched attempt
        # can disclose the bypassed preferred rung, and so a ladder exhausted
        # ONLY by policy sheds can force-admit past the bound rather than
        # manufacture a failure unbounded admission would not have had.
        policy_sheds: list[tuple[int, str]] = []
        forced_overflow = False
        while True:
            if candidate is None:
                if policy_sheds and last_failure is None and not forced_overflow:
                    forced_overflow = True
                    candidate = policy_sheds[0][0]
                else:
                    break
            deployment = _deployment_priced_for_service_tier(
                route.deployments[candidate],
                getattr(entry.request, "service_tier", None),
                forwards_tier=(
                    candidate < len(entry.tier_forwarded_by_depth)
                    and entry.tier_forwarded_by_depth[candidate]
                ),
            )
            ticket = self._reserve_rung_slot(entry, deployment, force=forced_overflow)
            if isinstance(ticket, RungShed):
                policy_sheds.append((candidate, ticket.reason))
                self._health.release_probe(keys[candidate])
                candidate = claim_route_from(self._health, keys, candidate + 1)
                continue
            dispatch_reason, preferred_deployment = _dispatch_disclosure(
                route,
                candidate,
                policy_sheds=policy_sheds,
                forced_overflow=forced_overflow,
            )
            try:
                attempt_id = self._write_ledger.start_attempt(
                    snapshot=route.snapshot,
                    deployment=deployment,
                    attempt_ordinal=entry.total_attempts,
                    route_depth=candidate,
                    maximum_cost_micro_usd=maximum_attempt_cost_micro_usd(
                        entry.request, deployment
                    ),
                    route_reason=route.route_reason,
                    fallback_reason=route.fallback_reason,
                    dispatch_reason=dispatch_reason,
                    preferred_deployment=preferred_deployment,
                )
            except BudgetReservationRejected as exc:
                if ticket is not None:
                    self._loads.release_ticket(ticket)
                self._health.release_probe(keys[candidate])
                if exc.scope_kind is not BudgetScopeKind.DEPLOYMENT:
                    error = (
                        NativeBridgeError(budget_quota_protocol_error())
                        if self._budget_error_factory is None
                        else self._budget_error_factory(str(data.get("raw_key", "")))
                    )
                    self.finish_request_quietly(entry.authorization, budget_quota_failure())
                    with self._lock:
                        self._inflight.pop(request_id, None)
                    raise error from exc
                # A route whose hard monthly allocation cannot admit this
                # call is skipped; a later certified route may still serve.
                last_failure = (
                    budget_quota_failure()
                    if candidate == len(route.deployments) - 1
                    else all_routes_unavailable_failure()
                )
                candidate = claim_route_from(self._health, keys, candidate + 1)
                continue
            except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
                # A reservation that raised before returning an attempt id
                # wrote nothing durable; the accepted request is terminalized
                # and the sanitized failure answers the caller.
                if ticket is not None:
                    self._loads.release_ticket(ticket)
                self._health.release_probe(keys[candidate])
                error = authority_error(exc)
                self.finish_request_quietly(
                    entry.authorization,
                    GatewayFailure(
                        failure_class=GatewayFailureClass.INTERNAL,
                        safe_message=(
                            "gateway could not reserve attempt accounting before dispatch"
                        ),
                    ),
                )
                with self._lock:
                    self._inflight.pop(request_id, None)
                raise error from exc
            if ticket is not None:
                self._loads.bind(ticket, attempt_id)
            if forced_overflow:
                with self._lock:
                    self._rung_saturated_overflows += 1
            with self._lock:
                entry.attempt_counts[candidate] += 1
                entry.total_attempts += 1
                entry.active_attempt_id = attempt_id
                entry.attempt_depths[attempt_id] = candidate
            return json.dumps(
                {"attempt_id": attempt_id, "route_depth": candidate},
                separators=(",", ":"),
            )
        exhaustion = last_failure
        if exhaustion is None:
            # Nothing dispatched and nothing classified: forced claims admit
            # any non-throttled circuit, so an empty first claim means every
            # deployment sits inside a provider throttle window.
            throttled_remaining = self._health.throttled_remaining_seconds(keys)
            exhaustion = (
                all_routes_throttled_failure(throttled_remaining)
                if throttled_remaining is not None
                else all_routes_unavailable_failure()
            )
        self.finish_request_quietly(entry.authorization, ledger_failure(exhaustion))
        with self._lock:
            self._inflight.pop(request_id, None)
        failure_payload: JsonObject = {
            "failure_class": exhaustion.failure_class.value,
            "safe_message": exhaustion.safe_message,
        }
        if exhaustion.customer_owned:
            # Echoed back so the data plane renders the caller's 400, not the
            # house 502, from the failure that ended the ladder.
            failure_payload["customer_owned"] = True
        if exhaustion.rejected_parameter is not None:
            failure_payload["rejected_parameter"] = exhaustion.rejected_parameter
        if exhaustion.provider_detail is not None:
            failure_payload["provider_detail"] = exhaustion.provider_detail
        if exhaustion.retry_after_seconds is not None:
            failure_payload["retry_after_seconds"] = exhaustion.retry_after_seconds
        return json.dumps(
            {"exhausted": True, "failure": failure_payload},
            separators=(",", ":"),
        )

    def settle(self, argument: str) -> str:
        """Durably settle one previously reserved attempt exactly once.

        A finalizing settlement also terminalizes the request and removes the
        in-flight entry; a non-finalizing one (a failed precommit dispatch
        with a successor still possible) closes only the attempt so the
        waterfall can reserve its next dispatch. Deployment-health circuits
        record every settled outcome, restoring admission first when the
        dispatch had opened.

        Args:
            argument: JSON object with ``request_id``, ``attempt_id``,
                ``outcome``, optional ``usage``, ``tool_names``, ``failure``,
                ``finalize`` (default true), and ``opened`` (default false).

        Returns:
            An empty JSON object; repeated settlement is a no-op.

        Raises:
            NativeBridgeError: The durable terminal write failed; the
                in-flight entry is kept so a retried settlement (from the
                data plane or the deadline sweep) can still reach the ledger.
        """
        data = json.loads(argument)
        request_id = str(data["request_id"])
        with self._lock:
            entry = self._inflight.get(request_id)
        if entry is None:
            return "{}"
        attempt_id = str(data["attempt_id"])
        finalize = bool(data.get("finalize", True))
        opened = bool(data.get("opened", False))
        terminal, failure = terminal_from_settlement(data)
        first_token_at = first_token_at_from_settlement(data)
        try:
            self._write_ledger.finish_attempt(
                attempt_id=attempt_id,
                terminal_event=terminal,
                failure=failure,
                finalize_request=finalize,
                first_token_at=first_token_at,
            )
        except Exception as exc:  # noqa: BLE001 - the data plane retries.
            # The exact settlement is retained so a retry (from the data
            # plane or the timer sweep) lands the ORIGINAL outcome and usage,
            # never a downgraded cancellation.
            with self._lock:
                entry.pending_settlement = data
            raise authority_error(exc) from exc
        self._record_health(entry, attempt_id, opened=opened, failure=failure)
        with self._lock:
            if finalize:
                self._inflight.pop(request_id, None)
            elif entry.active_attempt_id == attempt_id:
                entry.active_attempt_id = None
        return "{}"

    def abandon(self, argument: str) -> str:
        """Terminalize one accepted request with no settleable outcome.

        The data plane calls this when a request ends before any attempt is
        active (a queue-deadline expiry, a drained permit, admission wire
        drift, or a dropped handler between attempts). An active attempt is
        settled with the given failure; otherwise only the accepted request
        is finalized.

        Args:
            argument: JSON object with ``request_id`` and an optional
                ``failure`` (defaulting to a cancellation).

        Returns:
            An empty JSON object; an unknown request is a no-op.

        Raises:
            NativeBridgeError: The durable terminal write failed; the entry
                is kept so the deadline sweep can still close it.
        """
        data = json.loads(argument)
        request_id = str(data["request_id"])
        with self._lock:
            entry = self._inflight.get(request_id)
        if entry is None:
            return "{}"
        failure = _failure_from_payload(data.get("failure")) or GatewayFailure(
            failure_class=GatewayFailureClass.CANCELLED,
            safe_message="gateway request was cancelled",
        )
        try:
            if entry.active_attempt_id is not None:
                self._record_health(entry, entry.active_attempt_id, opened=False, failure=failure)
                self._write_ledger.finish_attempt(
                    attempt_id=entry.active_attempt_id,
                    terminal_event=GatewayEvent(
                        kind=GatewayEventKind.FAILED,
                        sequence_number=0,
                        failure=failure,
                    ),
                    failure=failure,
                    finalize_request=True,
                )
            else:
                self._write_ledger.finish_request(
                    authorization=entry.authorization,
                    failure=failure,
                )
        except Exception as exc:  # noqa: BLE001 - the data plane retries.
            raise authority_error(exc) from exc
        with self._lock:
            self._inflight.pop(request_id, None)
        return "{}"

    def finish_request_quietly(
        self,
        authorization: AuthorizationSnapshot,
        failure: GatewayFailure,
    ) -> None:
        """Finalize accepted pre-dispatch work without masking the primary failure."""
        try:
            self._write_ledger.finish_request(
                authorization=authorization,
                failure=failure,
            )
        except Exception:  # noqa: BLE001 - primary admission failure stays authoritative.
            self._accounting_healthy = False

    def _record_health(
        self,
        entry: InflightRequest,
        attempt_id: str,
        *,
        opened: bool,
        failure: GatewayFailure | None,
    ) -> None:
        """Apply one settled attempt's outcome to the deployment circuits.

        Mirrors the executor's recording order: a successful dispatch opening
        restores admission first, then the terminal outcome either closes the
        circuit or counts against it.

        Args:
            entry: The owning in-flight request.
            attempt_id: The settled attempt.
            opened: Whether the provider dispatch opened successfully.
            failure: The terminal failure, or ``None`` for a success.
        """
        # The rung's bounded-admission slot frees with the health recording:
        # both releases are idempotent, so settle, abandon, and the sweep can
        # each fire without double-counting.
        self._loads.release_attempt(attempt_id)
        depth = entry.attempt_depths.get(attempt_id)
        if depth is None:
            return
        key = deployment_health_key(entry.authorization, entry.route.deployments[depth])
        if opened:
            self._health.dispatch_opened(key)
        if failure is not None:
            self._health.failed(key, failure)
        else:
            self._health.succeeded(key)

    def _sweep_loop(self) -> None:
        """Run the settlement sweep on a timer for the process lifetime."""
        while True:
            time.sleep(_SWEEP_INTERVAL_SECONDS)
            self.sweep_expired()

    def sweep_expired(self) -> None:
        """Recover retained settlements and close abandoned requests.

        A retained settlement (the data plane's terminal write failed) is
        replayed verbatim so the original outcome, usage, and finalize flag
        land. A request with no settlement at all past its deadline plus
        grace is closed as cancelled through its active attempt when one
        exists, otherwise through its request row; that is the backstop for
        wire-contract failures and data-plane crashes short of process death.
        A retained settlement that fails again here latches
        accounting-unhealthy as a durable loss.
        """
        now = time.monotonic()
        with self._lock:
            retained = [
                (request_id, entry)
                for request_id, entry in self._inflight.items()
                if entry.pending_settlement is not None
            ][:_SWEEP_BATCH]
            abandoned = [
                (request_id, entry)
                for request_id, entry in self._inflight.items()
                if entry.pending_settlement is None
                and entry.deadline_monotonic + _SWEEP_GRACE_SECONDS < now
            ][:_SWEEP_BATCH]
        for request_id, entry in retained:
            settlement = entry.pending_settlement
            if settlement is None:
                continue
            terminal, failure = terminal_from_settlement(settlement)
            if self._settle_swept(
                request_id,
                entry,
                attempt_id=str(settlement["attempt_id"]),
                terminal=terminal,
                failure=failure,
                finalize=bool(settlement.get("finalize", True)),
            ):
                with self._lock:
                    entry.pending_settlement = None
                    self._sweep_retained_replayed += 1
        if not abandoned:
            return
        cancelled = GatewayFailure(
            failure_class=GatewayFailureClass.CANCELLED,
            safe_message="gateway request was abandoned before settlement",
        )
        terminal = GatewayEvent(kind=GatewayEventKind.FAILED, sequence_number=0, failure=cancelled)
        for request_id, entry in abandoned:
            if entry.active_attempt_id is None:
                # An accepted request with every attempt already settled (or
                # none reserved) closes through its request row alone.
                try:
                    self._write_ledger.finish_request(
                        authorization=entry.authorization,
                        failure=cancelled,
                    )
                except Exception:  # noqa: BLE001 - keep the entry; the sweep retries.
                    self._accounting_healthy = False
                    continue
                with self._lock:
                    self._inflight.pop(request_id, None)
                    self._sweep_abandoned_cancelled += 1
                continue
            if self._settle_swept(
                request_id,
                entry,
                attempt_id=entry.active_attempt_id,
                terminal=terminal,
                failure=cancelled,
                finalize=True,
            ):
                with self._lock:
                    self._sweep_abandoned_cancelled += 1

    def _settle_swept(
        self,
        request_id: str,
        entry: InflightRequest,
        *,
        attempt_id: str,
        terminal: GatewayEvent,
        failure: GatewayFailure | None,
        finalize: bool,
    ) -> bool:
        """Land one swept settlement; keep the entry for retry on failure.

        Returns:
            Whether the swept terminal write reached the ledger.
        """
        try:
            self._write_ledger.finish_attempt(
                attempt_id=attempt_id,
                terminal_event=terminal,
                failure=failure,
                finalize_request=finalize,
            )
        except Exception:  # noqa: BLE001 - keep the entry; the sweep retries.
            self._accounting_healthy = False
            return False
        self._record_health(entry, attempt_id, opened=False, failure=failure)
        with self._lock:
            if finalize:
                self._inflight.pop(request_id, None)
            elif entry.active_attempt_id == attempt_id:
                entry.active_attempt_id = None
        return True


def _dispatch_disclosure(
    route: GatewayRoute,
    candidate: int,
    *,
    policy_sheds: list[tuple[int, str]],
    forced_overflow: bool,
) -> tuple[str | None, ExactModelDeployment | None]:
    """Name why the chosen rung serves and the bypassed preferred rung, if any.

    Emission is gated so an alias the platform never opted in keeps byte-null
    disclosure columns. On a ``maximize_cache_affinity`` pool every attempt
    discloses against the rendezvous-preferred depth-0 rung: ``affinity`` on
    the happy path, the shed reason when depth 0 was policy-shed in this
    reservation, ``rung_dead`` when it was bypassed by health or an earlier
    failure, ``saturated_overflow`` when the ladder force-admitted past a
    bound. On any other pool a disclosure appears only when a dispatch policy
    actually shed a rung in this reservation, and the preferred rung is the
    shed rung itself (the counterfactual the shed is measured against).

    Args:
        route: Frozen ordered route for this request.
        candidate: The route depth about to dispatch.
        policy_sheds: ``(depth, reason)`` for every policy shed this
            reservation, in ladder order.
        forced_overflow: Whether this dispatch was forced past a bound.

    Returns:
        ``(dispatch_reason, preferred_deployment)``; the deployment is
        ``None`` whenever the chosen rung IS the disclosure's preferred rung.
    """
    if route.snapshot.failover_mode == "maximize_cache_affinity":
        target_depth = 0
        if forced_overflow:
            reason = "saturated_overflow"
        elif candidate == 0:
            reason = "affinity"
        else:
            lead_shed = next((shed for depth, shed in policy_sheds if depth == 0), None)
            reason = lead_shed or "rung_dead"
    elif forced_overflow:
        target_depth = policy_sheds[0][0]
        reason = "saturated_overflow"
    elif policy_sheds:
        target_depth, reason = policy_sheds[0]
    else:
        return None, None
    if target_depth == candidate:
        return reason, None
    return reason, route.deployments[target_depth]


def record_dead_admission_rungs(
    accounting: NativeAttemptAccounting,
    authorization: AuthorizationSnapshot,
    dead: tuple[DeadRung, ...],
    *,
    fallback_available: bool,
) -> None:
    """Record admission-dead rungs and surface a lead masked by fallback."""
    if not dead:
        return
    for rung in dead:
        accounting.health.failed(
            deployment_health_key(authorization, rung.deployment),
            rung.failure,
        )
    lead = next((rung for rung in dead if rung.index == 0), None)
    lead_masked = lead is not None and fallback_available
    accounting.record_admission_rung_skips(len(dead), lead_skipped=lead_masked)
    if lead is not None and fallback_available:
        _logger.warning(
            "gateway admission skipped the lead rung for alias %r: served off a "
            "fallback because deployment %r (provider %r) was dead at admission",
            authorization.alias,
            lead.deployment.deployment_id,
            lead.deployment.provider,
        )
