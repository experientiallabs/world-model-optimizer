"""Tests for the native attempt-accounting registry's waterfall reservations."""

from __future__ import annotations

import json
import time
from datetime import datetime

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models.catalog import (
    GatewayDeploymentCapabilities,
    GatewayDeploymentMetadata,
    GatewayRungDispatchPolicy,
)
from exp.common.models.gateway_catalog import ExactModelDeployment, FailoverMode
from exp.runtime.gateway.budgets import BudgetReservationRejected, BudgetScopeKind
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    ExecutionSnapshot,
    GatewayApiSurface,
    GatewayEvent,
    GatewayFailure,
    GatewayFailureClass,
    GatewayMessage,
    GatewayRequest,
)
from exp.runtime.gateway.native_accounting import (
    NativeAttemptAccounting,
    NativeBridgeError,
    _failure_from_payload,
)
from exp.runtime.gateway.native_execution import InflightRequest, deployment_health_key
from exp.runtime.gateway.native_settlement import ledger_failure
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.openai_protocol.errors import (
    THROTTLED_RETRY_AFTER_SECONDS,
    public_failure_error,
)

_DIGEST = "a" * 64


def _deployment(
    deployment_id: str,
    *,
    connection_sha256: str,
    dispatch: GatewayRungDispatchPolicy | None = None,
) -> ExactModelDeployment:
    """Build one deployment in the shared certified exact-model pool."""
    return ExactModelDeployment(
        deployment_id=deployment_id,
        source_alias=deployment_id,
        exact_model_id="exact-one",
        connection=f"connection-{deployment_id}",
        provider="openai",
        provider_model="provider-model",
        connection_sha256=connection_sha256,
        capabilities_sha256="d" * 64,
        gateway=GatewayDeploymentMetadata(
            capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
            dispatch=dispatch,
        ),
    )


def _authorization(catalog_sha256: str) -> AuthorizationSnapshot:
    """Build one direct authority snapshot pinned to the test catalog."""
    return AuthorizationSnapshot(
        request_id="request-one",
        organization_id="organization-one",
        identity_id="identity-one",
        virtual_key_id="key-one",
        alias="public-model",
        alias_revision_id="revision-one",
        target=DirectTarget(pool_id="pool-one"),
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        catalog_sha256=catalog_sha256,
        canonical_request_sha256=_DIGEST,
        deadline_monotonic=1.0,
    )


def _request() -> GatewayRequest:
    """Build one canonical request for physical execution tests."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hello"),),
    )


def _route(
    deployments: tuple[ExactModelDeployment, ...],
    *,
    refusal_failover: bool = False,
) -> GatewayRoute:
    """Build one frozen certified route with a live request deadline."""
    authorization = _authorization(_DIGEST).model_copy(
        update={
            "deadline_monotonic": time.monotonic() + 30,
            "refusal_failover": refusal_failover,
        }
    )
    return GatewayRoute(
        snapshot=ExecutionSnapshot(
            authorization=authorization,
            exact_model_id="exact-one",
            pool_id="pool-one",
            deployment_ids=tuple(item.deployment_id for item in deployments),
        ),
        deployment=deployments[0],
        fallback_deployments=deployments[1:],
        route_reason="direct",
    )


class _RecordingLedger:
    """Blocking write-ledger fake recording every waterfall write."""

    def __init__(self) -> None:
        """Start with empty write logs and no scripted rejections."""
        self.started: list[JsonObject] = []
        self.finished: list[JsonObject] = []
        self.finished_requests: list[GatewayFailure] = []
        self.budget_rejections: dict[str, BudgetScopeKind] = {}
        self._counter = 0

    def accept_request(self, *, authorization: AuthorizationSnapshot) -> None:
        """Record one accepted request (unused by the registry itself)."""
        del authorization

    def start_attempt(
        self,
        *,
        snapshot: object,
        deployment: ExactModelDeployment,
        attempt_ordinal: int,
        route_depth: int,
        maximum_cost_micro_usd: int | None = None,
        reserved_input_tokens: int | None = None,
        reserved_output_tokens: int | None = None,
        route_reason: str | None = None,
        fallback_reason: str | None = None,
        dispatch_reason: str | None = None,
        preferred_deployment: ExactModelDeployment | None = None,
    ) -> str:
        """Reserve one recorded attempt row, honoring scripted rejections."""
        del snapshot, maximum_cost_micro_usd, route_reason, fallback_reason
        scope = self.budget_rejections.get(deployment.deployment_id)
        if scope is not None:
            raise BudgetReservationRejected(scope_kind=scope, reason="scripted")
        self._counter += 1
        attempt_id = f"attempt-{self._counter}"
        self.started.append(
            {
                "attempt_id": attempt_id,
                "deployment_id": deployment.deployment_id,
                "attempt_ordinal": attempt_ordinal,
                "route_depth": route_depth,
                "reserved_input_tokens": reserved_input_tokens,
                "reserved_output_tokens": reserved_output_tokens,
                "dispatch_reason": dispatch_reason,
                "preferred_deployment_id": (
                    None if preferred_deployment is None else preferred_deployment.deployment_id
                ),
            }
        )
        return attempt_id

    def finish_attempt(
        self,
        *,
        attempt_id: str,
        terminal_event: GatewayEvent | None,
        failure: GatewayFailure | None,
        finalize_request: bool = True,
        first_token_at: datetime | None = None,
    ) -> None:
        """Record one settled attempt."""
        del terminal_event, first_token_at
        self.finished.append(
            {
                "attempt_id": attempt_id,
                "failure_class": None if failure is None else failure.failure_class.value,
                "finalize": finalize_request,
            }
        )

    def finish_request(
        self,
        *,
        authorization: AuthorizationSnapshot,
        failure: GatewayFailure,
    ) -> None:
        """Record one request-only terminalization."""
        del authorization
        self.finished_requests.append(failure)


def _registry() -> tuple[NativeAttemptAccounting, _RecordingLedger, InflightRequest]:
    """Compose one registry over a two-deployment certified route."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger)  # type: ignore[arg-type]
    deployments = (
        _deployment("deployment-a", connection_sha256="b" * 64),
        _deployment("deployment-b", connection_sha256="c" * 64),
    )
    route = _route(deployments)
    entry = InflightRequest(
        authorization=route.snapshot.authorization,
        route=route,
        request=_request(),
        deadline_monotonic=time.monotonic() + 30,
    )
    registry.register(entry)
    return registry, ledger, entry


def _start(
    registry: NativeAttemptAccounting,
    *,
    ordinal: int,
    current_depth: int | None = None,
    failure: JsonObject | None = None,
    request_id: str = "request-one",
) -> JsonObject:
    """Call one start_attempt with the data plane's wire shape."""
    return json.loads(
        registry.start_attempt(
            json.dumps(
                {
                    "request_id": request_id,
                    "attempt_ordinal": ordinal,
                    "current_depth": current_depth,
                    "failure": failure,
                }
            )
        )
    )


def _settle(
    registry: NativeAttemptAccounting,
    *,
    attempt_id: str,
    outcome: str,
    finalize: bool,
    failure: JsonObject | None = None,
    request_id: str = "request-one",
) -> str:
    """Call one settle with the data plane's wire shape."""
    return registry.settle(
        json.dumps(
            {
                "request_id": request_id,
                "attempt_id": attempt_id,
                "outcome": outcome,
                "usage": None,
                "tool_names": [],
                "failure": failure,
                "finalize": finalize,
                "opened": True,
            }
        )
    )


def _retryable_failure() -> JsonObject:
    """One wire failure the executor may redial on the same deployment."""
    return {
        "failure_class": "provider_internal",
        "safe_message": "provider service failed; retry after a short delay",
        "retryable_same_deployment": True,
        "failover_eligible": True,
    }


def test_waterfall_reservations_count_every_physical_dispatch() -> None:
    """Ordinals count all dispatches; depth tracks the deployment position."""
    registry, ledger, _entry = _registry()
    first = _start(registry, ordinal=0)
    assert first == {"attempt_id": "attempt-1", "route_depth": 0}
    assert (
        _settle(
            registry,
            attempt_id="attempt-1",
            outcome="failed",
            finalize=False,
            failure=_retryable_failure(),
        )
        == "{}"
    )
    redial = _start(registry, ordinal=1, current_depth=0, failure=_retryable_failure())
    assert redial == {"attempt_id": "attempt-2", "route_depth": 0}
    assert (
        _settle(
            registry,
            attempt_id="attempt-2",
            outcome="failed",
            finalize=False,
            failure=_retryable_failure(),
        )
        == "{}"
    )
    failover = _start(registry, ordinal=2, current_depth=0, failure=_retryable_failure())
    assert failover == {"attempt_id": "attempt-3", "route_depth": 1}
    assert _settle(registry, attempt_id="attempt-3", outcome="completed", finalize=True) == "{}"
    assert [(row["attempt_ordinal"], row["route_depth"]) for row in ledger.started] == [
        (0, 0),
        (1, 0),
        (2, 1),
    ]
    # Every physical dispatch reserves a positive worst-case token window so the
    # platform's token caps bind on the in-flight burst, not only on settlement.
    for row in ledger.started:
        assert isinstance(row["reserved_input_tokens"], int) and row["reserved_input_tokens"] > 0
        assert isinstance(row["reserved_output_tokens"], int) and row["reserved_output_tokens"] > 0
    assert [row["finalize"] for row in ledger.finished] == [False, False, True]
    assert registry.entry("request-one") is None
    assert ledger.finished_requests == []


def test_deployment_budget_rejection_skips_to_the_next_route() -> None:
    """A deployment-scope budget rejection advances without a caller error."""
    registry, ledger, _entry = _registry()
    ledger.budget_rejections["deployment-a"] = BudgetScopeKind.DEPLOYMENT
    started = _start(registry, ordinal=0)
    assert started["route_depth"] == 1
    assert ledger.started[0]["deployment_id"] == "deployment-b"


def test_non_deployment_budget_rejection_finalizes_with_quota() -> None:
    """A team-scope rejection raises the public quota error and finalizes."""
    registry, ledger, _entry = _registry()
    ledger.budget_rejections["deployment-a"] = BudgetScopeKind.TEAM
    with pytest.raises(NativeBridgeError) as excinfo:
        _start(registry, ordinal=0)
    payload = json.loads(excinfo.value.public_error_json)
    assert payload["status_code"] == 429
    assert payload["code"] == "insufficient_quota"
    assert [failure.failure_class.value for failure in ledger.finished_requests] == [
        "quota_exceeded"
    ]
    assert registry.entry("request-one") is None


def test_exhaustion_finalizes_the_request_with_the_last_failure() -> None:
    """An ineligible failure class exhausts the ladder and finalizes."""
    registry, ledger, _entry = _registry()
    started = _start(registry, ordinal=0)
    assert (
        _settle(
            registry,
            attempt_id=str(started["attempt_id"]),
            outcome="failed",
            finalize=False,
            failure={"failure_class": "invalid_request", "safe_message": "bad request"},
        )
        == "{}"
    )
    exhausted = _start(
        registry,
        ordinal=1,
        current_depth=0,
        failure={
            "failure_class": "invalid_request",
            "safe_message": "bad request",
            "retryable_same_deployment": False,
            "failover_eligible": False,
        },
    )
    assert exhausted["exhausted"] is True
    failure_payload = exhausted["failure"]
    assert isinstance(failure_payload, dict)
    assert failure_payload["failure_class"] == "invalid_request"
    assert [failure.failure_class.value for failure in ledger.finished_requests] == [
        "invalid_request"
    ]
    assert registry.entry("request-one") is None


def test_a_fully_throttled_route_exhausts_as_throttled_not_provider_internal() -> None:
    """Pre-dispatch exhaustion caused only by provider throttle windows is
    caller-facing rate limiting, never platform deadness.

    Production signal (2026-09-04): a single-rung alias whose rung sat inside
    the 30s throttle window after provider 429s reported every shadowed
    request as provider_internal "all exact-model deployments are
    unavailable", misfiling a 429 storm as an outage.
    """
    registry, ledger, entry = _registry()
    throttle = GatewayFailure(
        failure_class=GatewayFailureClass.THROTTLED,
        safe_message="provider throttled the request",
    )
    for deployment in entry.route.deployments:
        registry.health.failed(deployment_health_key(entry.authorization, deployment), throttle)

    exhausted = _start(registry, ordinal=0)

    assert exhausted["exhausted"] is True
    failure_payload = exhausted["failure"]
    assert isinstance(failure_payload, dict)
    assert failure_payload["failure_class"] == "throttled"
    message = str(failure_payload["safe_message"])
    assert "throttle window" in message
    retry_after = failure_payload["retry_after_seconds"]
    assert isinstance(retry_after, int)
    # The advertised Retry-After covers the whole remaining window (floored
    # at the default backoff) and the message names the same wait, so a
    # client honoring the header never retries into the window it was told
    # to sit out.
    assert THROTTLED_RETRY_AFTER_SECONDS <= retry_after <= 30
    assert f"retry in {retry_after}s" in message
    public = public_failure_error(GatewayFailure.model_validate(failure_payload))
    assert public.retry_after_seconds == retry_after
    assert [failure.failure_class.value for failure in ledger.finished_requests] == ["throttled"]
    assert registry.entry("request-one") is None


def test_an_open_circuit_route_still_dispatches_and_never_reports_throttled() -> None:
    """Circuit-open deployments stay dispatchable through forced claims, so
    the throttled exhaustion class is reserved for real throttle windows."""
    registry, ledger, entry = _registry()
    dead = GatewayFailure(
        failure_class=GatewayFailureClass.PROVIDER_AUTHENTICATION,
        safe_message="provider authentication failed",
    )
    for deployment in entry.route.deployments:
        registry.health.failed(deployment_health_key(entry.authorization, deployment), dead)

    started = _start(registry, ordinal=0)

    assert started["route_depth"] == 0
    assert ledger.finished_requests == []


def test_ordinal_mismatch_is_a_wire_contract_failure() -> None:
    """A desynchronized dispatch count fails closed as an internal error."""
    registry, _ledger, _entry = _registry()
    with pytest.raises(NativeBridgeError):
        _start(registry, ordinal=3)


def test_abandon_without_an_active_attempt_finalizes_the_request_row() -> None:
    """Abandoning an accepted request with no reservation closes the request."""
    registry, ledger, _entry = _registry()
    assert registry.abandon(json.dumps({"request_id": "request-one"})) == "{}"
    assert [failure.failure_class.value for failure in ledger.finished_requests] == ["cancelled"]
    assert registry.entry("request-one") is None


def test_sweep_cancels_the_active_attempt_after_the_deadline() -> None:
    """The deadline sweep closes an unsettled reservation as cancelled."""
    registry, ledger, entry = _registry()
    started = _start(registry, ordinal=0)
    entry.deadline_monotonic = time.monotonic() - 60.0
    registry.sweep_expired()
    assert ledger.finished == [
        {
            "attempt_id": started["attempt_id"],
            "failure_class": "cancelled",
            "finalize": True,
        }
    ]
    assert registry.entry("request-one") is None
    assert registry.counters()[1] == 1


def test_rejected_parameter_crosses_the_boundary_only_as_a_string() -> None:
    """The provider-named parameter path survives the failure payload decode."""
    registry, _ledger, _entry = _registry()
    started = _start(registry, ordinal=0)
    assert (
        _settle(
            registry,
            attempt_id=str(started["attempt_id"]),
            outcome="failed",
            finalize=False,
            failure={
                "failure_class": "invalid_request",
                "safe_message": "provider rejected the request",
                "rejected_parameter": "input[1].status",
            },
        )
        == "{}"
    )
    exhausted = _start(
        registry,
        ordinal=1,
        current_depth=0,
        failure={
            "failure_class": "invalid_request",
            "safe_message": "provider rejected the request",
            "rejected_parameter": "input[1].status",
        },
    )
    assert exhausted["exhausted"] is True
    failure_payload = exhausted["failure"]
    assert isinstance(failure_payload, dict)
    assert failure_payload["rejected_parameter"] == "input[1].status"
    # Non-string or empty payload values decode to None, never a coerced str.
    numeric = _failure_from_payload(
        {"failure_class": "invalid_request", "safe_message": "x", "rejected_parameter": 7}
    )
    assert numeric is not None and numeric.rejected_parameter is None
    empty = _failure_from_payload(
        {"failure_class": "invalid_request", "safe_message": "x", "rejected_parameter": ""}
    )
    assert empty is not None and empty.rejected_parameter is None


def _admit(
    registry: NativeAttemptAccounting,
    deployments: tuple[ExactModelDeployment, ...],
    *,
    request_id: str,
    organization_id: str = "organization-one",
    weight: int = 1,
    failover_mode: FailoverMode = "maximize_availability",
) -> InflightRequest:
    """Register one admitted request over the given rung ladder."""
    authorization = _authorization(_DIGEST).model_copy(
        update={
            "request_id": request_id,
            "organization_id": organization_id,
            "fair_share_weight": weight,
            "deadline_monotonic": time.monotonic() + 30,
        }
    )
    route = GatewayRoute(
        snapshot=ExecutionSnapshot(
            authorization=authorization,
            exact_model_id="exact-one",
            pool_id="pool-one",
            deployment_ids=tuple(item.deployment_id for item in deployments),
            failover_mode=failover_mode,
        ),
        deployment=deployments[0],
        fallback_deployments=deployments[1:],
        route_reason="direct",
    )
    entry = InflightRequest(
        authorization=authorization,
        route=route,
        request=_request(),
        deadline_monotonic=time.monotonic() + 30,
    )
    registry.register(entry)
    return entry


def _bounded_pair(
    bound: int,
    *,
    fair_share: bool = False,
) -> tuple[ExactModelDeployment, ExactModelDeployment]:
    """Build a bounded lead rung with an unbounded spill rung behind it."""
    return (
        _deployment(
            "deployment-a",
            connection_sha256="b" * 64,
            dispatch=GatewayRungDispatchPolicy(concurrency_bound=bound, fair_share=fair_share),
        ),
        _deployment("deployment-b", connection_sha256="c" * 64),
    )


class TestRungDispatchPolicy:
    """Bounded-queue spill, fair-share sheds, overflow, and their disclosures."""

    def test_bound_spills_to_the_next_rung_with_disclosure(self) -> None:
        """The dispatch past the bound lands on the spill rung, disclosed."""
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = _bounded_pair(1)
        _admit(registry, deployments, request_id="request-1")
        _admit(registry, deployments, request_id="request-2")
        first = _start(registry, ordinal=0, request_id="request-1")
        assert first["route_depth"] == 0
        assert ledger.started[0]["dispatch_reason"] is None
        assert ledger.started[0]["preferred_deployment_id"] is None
        spilled = _start(registry, ordinal=0, request_id="request-2")
        assert spilled["route_depth"] == 1
        assert ledger.started[1]["dispatch_reason"] == "queue_bound"
        assert ledger.started[1]["preferred_deployment_id"] == "deployment-a"
        assert registry.rung_admission_counters() == (1, 0)

    def test_settle_frees_the_bounded_slot(self) -> None:
        """A settled dispatch returns its slot so the next request is not shed."""
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = _bounded_pair(1)
        _admit(registry, deployments, request_id="request-1")
        _admit(registry, deployments, request_id="request-2")
        started = _start(registry, ordinal=0, request_id="request-1")
        _settle(
            registry,
            attempt_id=str(started["attempt_id"]),
            outcome="completed",
            finalize=True,
            request_id="request-1",
        )
        follow = _start(registry, ordinal=0, request_id="request-2")
        assert follow["route_depth"] == 0
        assert registry.rung_admission_counters() == (0, 0)

    def test_saturated_overflow_never_manufactures_a_failure(self) -> None:
        """A single-rung pool at its bound still dispatches, disclosed as such."""
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        only = (
            _deployment(
                "deployment-a",
                connection_sha256="b" * 64,
                dispatch=GatewayRungDispatchPolicy(concurrency_bound=1),
            ),
        )
        _admit(registry, only, request_id="request-1")
        _admit(registry, only, request_id="request-2")
        assert _start(registry, ordinal=0, request_id="request-1")["route_depth"] == 0
        overflow = _start(registry, ordinal=0, request_id="request-2")
        assert overflow["route_depth"] == 0
        assert ledger.started[1]["dispatch_reason"] == "saturated_overflow"
        assert ledger.started[1]["preferred_deployment_id"] is None
        assert registry.rung_admission_counters() == (1, 1)

    def test_fair_share_shed_discloses_and_spills(self) -> None:
        """An over-share organization spills while the under-share one admits."""
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = _bounded_pair(4, fair_share=True)
        for index in range(1, 4):
            _admit(registry, deployments, request_id=f"a-{index}", organization_id="org-a")
            assert _start(registry, ordinal=0, request_id=f"a-{index}")["route_depth"] == 0
        # org-b admits its first (total 4, at the bound afterwards)...
        _admit(registry, deployments, request_id="b-1", organization_id="org-b")
        assert _start(registry, ordinal=0, request_id="b-1")["route_depth"] == 0
        # ...one org-a request settles, freeing a slot reserved for org-b.
        settled = ledger.started[0]
        _settle(
            registry,
            attempt_id=str(settled["attempt_id"]),
            outcome="completed",
            finalize=True,
            request_id="a-1",
        )
        _admit(registry, deployments, request_id="a-4", organization_id="org-a")
        shed = _start(registry, ordinal=0, request_id="a-4")
        assert shed["route_depth"] == 1
        assert ledger.started[-1]["dispatch_reason"] == "fair_share_shed"
        assert ledger.started[-1]["preferred_deployment_id"] == "deployment-a"
        # The under-share organization still lands on the house rung.
        _admit(registry, deployments, request_id="b-2", organization_id="org-b")
        assert _start(registry, ordinal=0, request_id="b-2")["route_depth"] == 0

    def test_affinity_pool_discloses_every_attempt(self) -> None:
        """Affinity pools stamp the happy path and name a dead preferred rung."""
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = (
            _deployment("deployment-a", connection_sha256="b" * 64),
            _deployment("deployment-b", connection_sha256="c" * 64),
        )
        _admit(
            registry,
            deployments,
            request_id="request-1",
            failover_mode="maximize_cache_affinity",
        )
        first = _start(registry, ordinal=0, request_id="request-1")
        assert first["route_depth"] == 0
        assert ledger.started[0]["dispatch_reason"] == "affinity"
        assert ledger.started[0]["preferred_deployment_id"] is None
        _settle(
            registry,
            attempt_id=str(first["attempt_id"]),
            outcome="failed",
            finalize=False,
            failure={
                "failure_class": "provider_internal",
                "safe_message": "provider failed",
                "failover_eligible": True,
            },
            request_id="request-1",
        )
        failover = _start(
            registry,
            ordinal=1,
            current_depth=0,
            failure={
                "failure_class": "provider_internal",
                "safe_message": "provider failed",
                "failover_eligible": True,
            },
            request_id="request-1",
        )
        assert failover["route_depth"] == 1
        assert ledger.started[1]["dispatch_reason"] == "rung_dead"
        assert ledger.started[1]["preferred_deployment_id"] == "deployment-a"

    def test_affinity_throttle_fails_over_unlike_maximize_cache(self) -> None:
        """A throttle on an affinity pool spills to the deterministic alternate."""
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = (
            _deployment("deployment-a", connection_sha256="b" * 64),
            _deployment("deployment-b", connection_sha256="c" * 64),
        )
        _admit(
            registry,
            deployments,
            request_id="request-1",
            failover_mode="maximize_cache_affinity",
        )
        first = _start(registry, ordinal=0, request_id="request-1")
        throttle: JsonObject = {
            "failure_class": "throttled",
            "safe_message": "provider throttled the request",
            "failover_eligible": True,
        }
        _settle(
            registry,
            attempt_id=str(first["attempt_id"]),
            outcome="failed",
            finalize=False,
            failure=throttle,
            request_id="request-1",
        )
        failover = _start(
            registry, ordinal=1, current_depth=0, failure=throttle, request_id="request-1"
        )
        assert failover["route_depth"] == 1

    def test_flag_off_attempts_carry_no_disclosures_or_load_state(self) -> None:
        """Untouched pools keep null disclosure fields and an empty registry."""
        registry, ledger, _entry = _registry()
        started = _start(registry, ordinal=0)
        _settle(
            registry,
            attempt_id=str(started["attempt_id"]),
            outcome="failed",
            finalize=False,
            failure=_retryable_failure(),
        )
        _start(registry, ordinal=1, current_depth=0, failure=_retryable_failure())
        assert all(row["dispatch_reason"] is None for row in ledger.started)
        assert all(row["preferred_deployment_id"] is None for row in ledger.started)
        assert registry.loads.inflight(("deployment-a", "b" * 64)) == 0
        assert registry.rung_admission_counters() == (0, 0)

    def test_budget_skip_releases_the_reserved_slot(self) -> None:
        """A deployment-budget rejection frees the rung's bounded reservation."""
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = _bounded_pair(1)
        ledger.budget_rejections["deployment-a"] = BudgetScopeKind.DEPLOYMENT
        _admit(registry, deployments, request_id="request-1")
        started = _start(registry, ordinal=0, request_id="request-1")
        assert started["route_depth"] == 1
        assert registry.loads.inflight(("deployment-a", "b" * 64)) == 0

    def test_abandon_releases_the_reserved_slot(self) -> None:
        """An abandoned active attempt frees its rung slot for new arrivals."""
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = _bounded_pair(1)
        _admit(registry, deployments, request_id="request-1")
        _admit(registry, deployments, request_id="request-2")
        assert _start(registry, ordinal=0, request_id="request-1")["route_depth"] == 0
        assert registry.abandon(json.dumps({"request_id": "request-1"})) == "{}"
        assert registry.loads.inflight(("deployment-a", "b" * 64)) == 0
        assert _start(registry, ordinal=0, request_id="request-2")["route_depth"] == 0


def test_provider_detail_crosses_the_boundary_only_as_a_string() -> None:
    """The provider explanation survives the failure payload decode."""
    registry, _ledger, _entry = _registry()
    _start(registry, ordinal=0)
    exhausted = _start(
        registry,
        ordinal=1,
        current_depth=0,
        failure={
            "failure_class": "invalid_request",
            "safe_message": "provider rejected the request",
            "provider_detail": "`top_p` is deprecated for this model.",
        },
    )
    failure_payload = exhausted["failure"]
    assert isinstance(failure_payload, dict)
    assert failure_payload["provider_detail"] == "`top_p` is deprecated for this model."
    numeric = _failure_from_payload(
        {"failure_class": "invalid_request", "safe_message": "x", "provider_detail": 7}
    )
    assert numeric is not None and numeric.provider_detail is None
    empty_detail = _failure_from_payload(
        {"failure_class": "invalid_request", "safe_message": "x", "provider_detail": ""}
    )
    assert empty_detail is not None and empty_detail.provider_detail is None


def test_deployment_priced_for_service_tier_overrides_only_for_a_carried_tier() -> None:
    """A requested tier with a pass-through card reprices the deployment copy;
    no tier, or a tier the deployment lacks, returns the deployment unchanged."""
    from exp.common.models.catalog import (
        GatewayDeploymentMetadata,
        GatewayServiceTierPrices,
        GatewayTokenPrices,
    )
    from exp.common.models.gateway_catalog import ExactModelDeployment
    from exp.runtime.gateway.native_accounting import _deployment_priced_for_service_tier

    deployment = ExactModelDeployment(
        deployment_id="d1",
        source_alias="d1",
        exact_model_id="exact-one",
        connection="connection-d1",
        provider="openai",
        provider_model="provider-model",
        connection_sha256="b" * 64,
        capabilities_sha256="c" * 64,
        gateway=GatewayDeploymentMetadata(
            prices=GatewayTokenPrices(
                input_micro_usd_per_million_tokens=1_000_000,
                output_micro_usd_per_million_tokens=4_000_000,
                flex=GatewayServiceTierPrices(
                    input_micro_usd_per_million_tokens=500_000,
                    output_micro_usd_per_million_tokens=2_000_000,
                ),
            )
        ),
    )

    flex = _deployment_priced_for_service_tier(deployment, "flex", forwards_tier=True)
    assert flex is not deployment
    assert flex.gateway.prices.input_micro_usd_per_million_tokens == 500_000
    assert flex.gateway.prices.output_micro_usd_per_million_tokens == 2_000_000
    # Identity and everything else is preserved on the copy.
    assert flex.deployment_id == "d1" and flex.exact_model_id == "exact-one"

    # No tier, default/auto, and a tier the deployment does not carry: unchanged.
    assert _deployment_priced_for_service_tier(deployment, None, forwards_tier=False) is deployment
    assert (
        _deployment_priced_for_service_tier(deployment, "default", forwards_tier=False)
        is deployment
    )
    assert (
        _deployment_priced_for_service_tier(deployment, "priority", forwards_tier=False)
        is deployment
    )

    # A carded tier that the SELECTED depth does not forward (a card on a lane
    # whose wire would strip the tier) bills the BASE schedule, never the card:
    # forwards_tier=False returns the deployment unchanged even though the flex
    # card exists.
    assert (
        _deployment_priced_for_service_tier(deployment, "flex", forwards_tier=False) is deployment
    )


def test_start_attempt_reprices_only_when_the_selected_depth_forwards_the_tier() -> None:
    """The reservation applies the tier card ONLY on a depth that forwards it.

    Regression for the forward/bill divergence: a flex CARD on a lane the
    selected depth does not forward reserves the BASE schedule, never the card,
    so the gateway can never reserve the tier rate while the provider runs the
    base schedule.
    """
    from exp.common.models.catalog import GatewayServiceTierPrices, GatewayTokenPrices

    class _PriceCapturingLedger(_RecordingLedger):
        """Recording ledger that also captures each reserved input rate."""

        def __init__(self) -> None:
            """Track the per-attempt reserved input rate alongside the base log."""
            super().__init__()
            self.reserved_input_micro: list[int | None] = []

        def start_attempt(
            self,
            *,
            snapshot: object,
            deployment: ExactModelDeployment,
            attempt_ordinal: int,
            route_depth: int,
            maximum_cost_micro_usd: int | None = None,
            reserved_input_tokens: int | None = None,
            reserved_output_tokens: int | None = None,
            route_reason: str | None = None,
            fallback_reason: str | None = None,
            dispatch_reason: str | None = None,
            preferred_deployment: ExactModelDeployment | None = None,
        ) -> str:
            """Record the reserved input rate, then reserve as the base fake does."""
            self.reserved_input_micro.append(
                deployment.gateway.prices.input_micro_usd_per_million_tokens
            )
            return super().start_attempt(
                snapshot=snapshot,
                deployment=deployment,
                attempt_ordinal=attempt_ordinal,
                route_depth=route_depth,
                maximum_cost_micro_usd=maximum_cost_micro_usd,
                reserved_input_tokens=reserved_input_tokens,
                reserved_output_tokens=reserved_output_tokens,
                route_reason=route_reason,
                fallback_reason=fallback_reason,
                dispatch_reason=dispatch_reason,
                preferred_deployment=preferred_deployment,
            )

    carded = _deployment("deployment-a", connection_sha256="b" * 64).model_copy(
        update={
            "gateway": GatewayDeploymentMetadata(
                capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
                prices=GatewayTokenPrices(
                    input_micro_usd_per_million_tokens=1_000_000,
                    output_micro_usd_per_million_tokens=4_000_000,
                    flex=GatewayServiceTierPrices(
                        input_micro_usd_per_million_tokens=500_000,
                        output_micro_usd_per_million_tokens=2_000_000,
                    ),
                ),
            )
        }
    )
    route = _route((carded,))
    flex_request = _request().model_copy(update={"service_tier": "flex"})

    def _reserved_rate(*, forwards: bool) -> int | None:
        ledger = _PriceCapturingLedger()
        registry = NativeAttemptAccounting(ledger)  # type: ignore[arg-type]
        registry.register(
            InflightRequest(
                authorization=route.snapshot.authorization,
                route=route,
                request=flex_request,
                deadline_monotonic=time.monotonic() + 30,
                tier_forwarded_by_depth=(forwards,),
            )
        )
        _start(registry, ordinal=0)
        assert len(ledger.reserved_input_micro) == 1
        return ledger.reserved_input_micro[0]

    # The selected depth forwards flex -> reserve at the flex card rate.
    assert _reserved_rate(forwards=True) == 500_000
    # Same flex card, but the selected depth strips the tier -> reserve at BASE.
    assert _reserved_rate(forwards=False) == 1_000_000


def test_customer_owned_failures_round_trip_and_file_as_the_callers_invalid_request() -> None:
    """A BYOK credential failure keeps its ladder class, echoes its ownership, and
    is recorded as the caller's invalid request."""
    parsed = _failure_from_payload(
        {
            "failure_class": "provider_authentication",
            "safe_message": "your connected openai credential was rejected by the provider",
            "failover_eligible": True,
            "customer_owned": True,
        }
    )
    assert parsed is not None
    assert parsed.customer_owned is True
    assert parsed.failure_class == GatewayFailureClass.PROVIDER_AUTHENTICATION
    assert ledger_failure(parsed).failure_class == GatewayFailureClass.INVALID_REQUEST
    # Only the two customer-configurable provider classes re-file; a
    # house-shaped failure (or one without the flag) is untouched.
    house = _failure_from_payload(
        {
            "failure_class": "provider_authentication",
            "safe_message": "provider authentication failed",
        }
    )
    assert house is not None and ledger_failure(house).failure_class is (
        GatewayFailureClass.PROVIDER_AUTHENTICATION
    )

    registry, _ledger, _entry = _registry()
    _start(registry, ordinal=0)
    exhausted = _start(
        registry,
        ordinal=1,
        current_depth=0,
        failure={
            "failure_class": "provider_quota",
            "safe_message": "your connected openrouter account has exhausted its quota",
            "customer_owned": True,
        },
    )
    failure_payload = exhausted["failure"]
    assert isinstance(failure_payload, dict)
    assert failure_payload["customer_owned"] is True
    assert failure_payload["failure_class"] == "provider_quota"
