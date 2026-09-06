"""Python control plane for the native (Rust) gateway data plane.

The native engine (`exp_gateway_native`) owns sockets, upstream streaming,
normalization, and SSE encoding. Shared Python contracts own decoding,
authorization, payload construction, continuation state, and durable ledger
transactions. Every boundary call takes and returns one JSON string.

Admission returns the full ordered certified route (one wire configuration
per deployment) plus the frozen retry-policy facts, accepting the request
without starting any attempt. The data plane then reserves each physical
dispatch through ``start_attempt`` immediately before network work and lands
each attempt's durable terminal through ``settle`` (finalizing the request
only on the terminal attempt); candidate selection stays here: the frozen
waterfall policy, health circuits, and budget skipping.

Boundary errors raise :class:`NativeBridgeError`, whose ``public_error_json``
attribute carries the sanitized OpenAI-shaped error the data plane returns to
the caller through the shared boundary mapping. Requests the native path
cannot serve (resolved clients exposing no native wire profile) are answered
with an ``{"escalate": reason}`` admission disposition after the accepted
request is finalized content-free; the data plane classifies the reason for
metrics and fails the request closed with the shared internal error.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable

from exp.common.core.artifacts import JsonObject, sha256_bytes
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    GatewayApiSurface,
    GatewayFailure,
    GatewayFailureClass,
    GatewayRequest,
)
from exp.runtime.gateway.group_commit import SyncGroupCommitLedger
from exp.runtime.gateway.guardrails.client import assert_not_internal_classification
from exp.runtime.gateway.guardrails.contracts import GuardrailRejected
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.native import enforce_native_input, enforce_native_output
from exp.runtime.gateway.native_accounting import (
    NativeAttemptAccounting,
    NativeBridgeError,
    gateway_updating_failure,
    record_dead_admission_rungs,
)
from exp.runtime.gateway.native_accounting import (
    authority_error as _authority_error,
)
from exp.runtime.gateway.native_admission import (
    admitted_route_requests,
    fold_parallel_tool_call_disclosures,
    resolve_admission_route,
)
from exp.runtime.gateway.native_batches import NativeBatchRelayMixin
from exp.runtime.gateway.native_bridge_errors import (
    escalation as _escalation,
)
from exp.runtime.gateway.native_bridge_errors import (
    ledger_capability_message,
)
from exp.runtime.gateway.native_bridge_errors import (
    public_capability_error as _public_capability_error,
)
from exp.runtime.gateway.native_components import NativeGatewayComponents, SyncWriteLedger
from exp.runtime.gateway.native_continuation import (
    continuation_binding_error as _continuation_binding_error,
)
from exp.runtime.gateway.native_continuation import (
    remember_continuation,
)
from exp.runtime.gateway.native_continuation import (
    require_bound_wire_authority as _require_bound_wire_authority,
)
from exp.runtime.gateway.native_continuation import (
    select_bound_continuation_route as _select_bound_continuation_route,
)
from exp.runtime.gateway.native_decode import NativeDecodeError, decode_native_body
from exp.runtime.gateway.native_dispatch import dispatch_signature_headers
from exp.runtime.gateway.native_embeddings import NativeEmbeddingsMixin
from exp.runtime.gateway.native_execution import (
    MAXIMUM_SAME_DEPLOYMENT_ATTEMPTS,
    MAXIMUM_TOTAL_ATTEMPTS,
    FrozenDispatchBinding,
    InflightRequest,
    NativeDialectUnavailableError,
    dispatchable_route_profiles,
    resolve_route_profiles,
    select_route_deployments,
)
from exp.runtime.gateway.native_images import NativeImagesMixin
from exp.runtime.gateway.native_observability import NativeObservabilityMixin
from exp.runtime.gateway.native_reasoning import (
    authenticate_reasoning_history,
    has_active_reasoning_content,
    seal_reasoning_carrier_content,
    strip_stale_reasoning_history,
    unseal_reasoning_history,
)
from exp.runtime.gateway.native_responses import (
    ContinuationContext,
    continuation_route_binding,
    continued_request,
    responses_envelope,
)
from exp.runtime.gateway.native_rungs import build_rung_dispatch
from exp.runtime.gateway.native_settlement import (
    optional_text,
)
from exp.runtime.gateway.reasoning_carrier import (
    ReasoningCarrierAuthority,
)
from exp.runtime.gateway.routing import GatewayRoute, GatewayRoutingError
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import (
    ProviderCapabilityError,
    ProviderParameterError,
    normalized_provider_failure,
)
from exp.runtime.models.providers.protocol import GatewayDispatchSigner, NativeWireClient
from exp.runtime.openai_protocol.errors import (
    OpenAIProtocolError,
    invalid_field,
    public_failure_error,
)
from exp.runtime.openai_protocol.requests import DecodedGatewayRequest
from exp.runtime.openai_protocol.state import (
    BoundedContinuationStore,
    ProtocolNamespace,
    replay_key,
)

_logger = logging.getLogger(__name__)


def _log_reasoning_continuation_rejection(
    authorization: AuthorizationSnapshot, stage: str, reason: object
) -> None:
    """Record why a reasoning-carrier continuation failed, for operators only.

    The caller sees one opaque 400 (naming the differing bound claim would be an
    authentic-continuation oracle), but an operator needs the exact reason to tell
    a genuine tamper from a benign authority drift. Nothing here carries a
    credential or the plaintext reasoning; the catalog-generation fields make a
    cross-worker or post-republish drift obvious when diffed against the issuing
    turn's admission log.
    """
    _logger.warning(
        "reasoning carrier continuation rejected",
        extra={
            "operation": "native_reasoning_continuation",
            "stage": stage,
            "reason": str(reason),
            "request_id": authorization.request_id,
            "alias": authorization.alias,
            "alias_revision_id": authorization.alias_revision_id,
            "catalog_sha256": authorization.catalog_sha256,
        },
    )


_REQUEST_TIMEOUT_SECONDS = 120.0


class NativeControlPlane(
    NativeBatchRelayMixin, NativeEmbeddingsMixin, NativeImagesMixin, NativeObservabilityMixin
):
    """Authority and accounting callbacks for the native data plane.

    Rust worker threads share the group-commit writer and the locked in-flight
    registry. Opportunistic sweeps bound abandoned reservations to the request
    deadline plus the sweep grace.
    """

    def __init__(
        self,
        components: NativeGatewayComponents,
        *,
        request_timeout_seconds: float = _REQUEST_TIMEOUT_SECONDS,
        data_plane_metrics: Callable[[], str] | None = None,
        continuation_store: BoundedContinuationStore | None = None,
        readiness_probe: Callable[[], bool] | None = None,
        usage_reporter: Callable[[], JsonObject] | None = None,
        budget_error_factory: Callable[[str], NativeBridgeError] | None = None,
        native_route_eligible: Callable[[GatewayRoute, GatewayRequest], bool] | None = None,
        guardrails: GuardrailEngine | None = None,
    ) -> None:
        """Bind loaded gateway components for serving.

        Args:
            components: Authority, ledger, routes, and runtime catalogs.
            request_timeout_seconds: Total per-request budget from admission.
            data_plane_metrics: Optional provider of the native engine's
                content-free metrics snapshot as one JSON string; the local
                launch injects ``exp_gateway_native.metrics_snapshot_json``.
                Without it the snapshot reports ``data_plane`` as ``None``.
            continuation_store: Optional injected Responses continuation
                state; a host supplies its own bounded namespaced history,
                and the default is one in-process bounded store.
            readiness_probe: Optional hosted lifecycle readiness callback.
            usage_reporter: Optional hosted usage report callback.
            budget_error_factory: Optional hosted mapping for a rejected reservation.
            native_route_eligible: Optional hosted policy for complete native semantics.
            guardrails: Optional identity-scoped engine. ``None`` leaves traffic unguarded.
        """
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        self._components = components
        # The optional batch lane: hosts without it leave every batch route
        # answering the uniform not-enabled error below.
        self._batches = getattr(components, "batches", None)
        # Hosted compositions have no local group-commit writer; they settle
        # directly through their own synchronous ledger.
        group_writer = getattr(components, "write_ledger", None)
        self._write_ledger: SyncWriteLedger = (
            SyncGroupCommitLedger(group_writer) if group_writer is not None else components.ledger
        )
        self._request_timeout_seconds = request_timeout_seconds
        self._data_plane_metrics = data_plane_metrics
        self._continuations = (
            continuation_store if continuation_store is not None else BoundedContinuationStore()
        )
        self._readiness_probe = readiness_probe
        self._usage_reporter = usage_reporter
        self._budget_error_factory = budget_error_factory
        self._native_route_eligible = native_route_eligible
        self._guardrails = guardrails
        # The accounting registry owns in-flight requests, per-dispatch
        # reservations, deployment-health circuits, and the deadline sweep.
        self._accounting = NativeAttemptAccounting(
            self._write_ledger,
            budget_error_factory=budget_error_factory,
        )

    @property
    def request_timeout_seconds(self) -> float:
        """Return the per-request budget shared with the data plane."""
        return self._request_timeout_seconds

    @property
    def reconciled_expired_requests(self) -> int:
        """Return crashed requests reconciled at startup."""
        return self._components.reconciled_expired_requests

    @property
    def reconciled_unknown_attempts(self) -> int:
        """Return crashed attempts reconciled at startup."""
        return self._components.reconciled_unknown_attempts

    def authenticate(self, argument: str) -> str:
        """Authenticate one virtual key before the data plane reads the body.

        Args:
            argument: JSON object with ``raw_key``.

        Returns:
            An empty JSON object on success.

        Raises:
            NativeBridgeError: The key is invalid, expired, or revoked.
        """
        data = json.loads(argument)
        try:
            self._components.store.authenticate_key(raw_key=data["raw_key"])
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            raise _authority_error(exc) from exc
        return "{}"

    def admit(self, argument: str) -> str:
        """Decode, authorize, inspect, route, and durably accept one request.

        The raw body is decoded with the same ``decode_chat`` the python
        engine uses, and every deployment's upstream payload is built with
        the same shared payload builders, so the two engines cannot drift at
        the protocol or provider boundary. No attempt row is written here:
        each physical dispatch is reserved by :meth:`start_attempt`.

        Args:
            argument: JSON object with ``raw_key``, ``body`` (raw request
                body text), optional ``surface`` (``"chat"`` or
                ``"responses"``, defaulting to chat), and optional
                ``app_referer``/``app_title`` caller app identity.

        Returns:
            JSON wire configuration carrying the full ordered certified
            ``route`` (one dialect, endpoint, headers, payload, and
            per-deployment idempotency key entry per deployment) plus the
            frozen retry-policy facts, or an ``{"escalate": reason}``
            disposition (its accepted request already finalized, with no
            attempt row) naming why the native plane cannot serve the
            request.

        Raises:
            NativeBridgeError: Decoding, authorization, routing, or
                capability admission failed.
        """
        assert_not_internal_classification()
        self._accounting.sweep_expired()
        data = json.loads(argument)
        surface = str(data.get("surface", "chat"))
        decoded = self._decode_body(
            data["body"],
            surface=surface,
            idempotency_key=optional_text(data.get("idempotency_key")),
            client_request_id=optional_text(data.get("client_request_id")),
            anthropic_beta=optional_text(data.get("anthropic_beta")),
        )
        request = decoded.request
        deadline = time.monotonic() + self._request_timeout_seconds
        try:
            # ``app_referer``/``app_title`` are forwarded when the native engine includes the
            # caller HTTP-Referer/X-Title in its admit payload; absent them app attribution
            # stays null on the default path until the Rust engine populates them.
            authorization = self._components.store.authorize_request(
                raw_key=data["raw_key"],
                alias=decoded.alias,
                request=request,
                deadline_monotonic=deadline,
                app_referer=optional_text(data.get("app_referer")),
                app_title=optional_text(data.get("app_title")),
            )
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            mapped = _authority_error(exc)
            pointer = self._batch_pointer_error(alias=decoded.alias, mapped=mapped)
            if pointer is not None:
                raise pointer from exc
            raise mapped from exc

        # Responses continuation resolves after authorization and before any
        # ledger write; unavailable, expired, evicted, or cross-namespace
        # state fails closed here.
        continuation_context: ContinuationContext | None = None
        if request.surface == GatewayApiSurface.RESPONSES:
            try:
                request, continuation_context = continued_request(
                    self._continuations,
                    authorization=authorization,
                    request=request,
                )
            except OpenAIProtocolError as exc:
                raise NativeBridgeError(exc) from exc

        pinned_reasoning_route: GatewayRoute | None = None
        try:
            request, pinned_reasoning_route = authenticate_reasoning_history(
                self._components,
                authorization,
                request,
            )
        except Exception as exc:  # noqa: BLE001 - one public shape prevents an oracle.
            _log_reasoning_continuation_rejection(authorization, "authenticate", exc)
            error = invalid_field(
                "messages.reasoning_content",
                "'messages.reasoning_content' must be an authentic continuation for this route.",
            )
            raise NativeBridgeError(error) from exc

        policy = None
        try:
            request, policy = enforce_native_input(
                self._guardrails,
                authorization=authorization,
                request=request,
                deadline_monotonic=deadline,
            )
        except GuardrailRejected as exc:
            raise NativeBridgeError(public_failure_error(exc.failure)) from exc
        retention_request = strip_stale_reasoning_history(request)
        try:
            request, verified_reasoning_route = unseal_reasoning_history(
                self._components,
                authorization,
                request,
            )
        except Exception as exc:  # noqa: BLE001 - one public shape prevents an oracle.
            _log_reasoning_continuation_rejection(authorization, "unseal", exc)
            error = invalid_field(
                "messages.reasoning_content",
                "'messages.reasoning_content' must be an authentic continuation for this route.",
            )
            raise NativeBridgeError(error) from exc
        if (
            pinned_reasoning_route is not None
            and verified_reasoning_route is not None
            and pinned_reasoning_route.deployment != verified_reasoning_route.deployment
        ):
            _log_reasoning_continuation_rejection(
                authorization, "route_pin", "authenticate and unseal resolved different deployments"
            )
            raise NativeBridgeError(
                invalid_field(
                    "messages.reasoning_content",
                    "'messages.reasoning_content' must be an authentic continuation "
                    "for this route.",
                )
            )
        pinned_reasoning_route = verified_reasoning_route
        request = strip_stale_reasoning_history(request)
        if pinned_reasoning_route is not None and not has_active_reasoning_content(request):
            pinned_reasoning_route = None
        if continuation_context is not None:
            # Execution receives authenticated plaintext, but the bounded
            # continuation store keeps the post-guardrail history sealed.
            continuation_context.messages = retention_request.messages

        # The ledger accepts the logical request before route selection, so a
        # keyed operation whose durable terminal already exists (or whose key
        # was reused with different content) fails closed here, before
        # learned selection can run request-time embedding or any other
        # provider-touching work.
        try:
            self._write_ledger.accept_request(authorization=authorization)
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            raise _authority_error(exc) from exc

        # Escalation runs after acceptance; the accepted request is finished
        # quietly before the disposition returns, so an unservable request is
        # accounted content-free and never billed. Routing failures found by
        # the probe are raised against the accepted request below.
        probe_failure: Exception | None = None
        route: GatewayRoute | None = None
        resolved_wires: tuple[tuple[GatewayWireProfile, NativeWireClient], ...] | None = None
        try:
            route = pinned_reasoning_route or self._resolve_route(
                authorization,
                request,
                continuation=continuation_context,
            )
            route = _select_bound_continuation_route(
                route,
                None
                if continuation_context is None
                else continuation_context.required_route_binding,
            )
            # A rung that is dead at admission (a lost credential, a drifted
            # connection) is skipped so a live fallback still serves the
            # request instead of the whole request failing on a dead lead.
            dispatchable = dispatchable_route_profiles(self._components.runtime_catalogs, route)
            record_dead_admission_rungs(
                self._accounting,
                authorization,
                dispatchable.dead,
                fallback_available=bool(dispatchable.indexes),
            )
            if not dispatchable.indexes:
                if (
                    continuation_context is not None
                    and continuation_context.required_route_binding is not None
                ):
                    raise _continuation_binding_error()
                # Every certified rung was operationally dead at admission;
                # there is nothing live to serve, so the accepted request is
                # finished closed.
                return self._escalate_accepted(
                    authorization,
                    "every certified deployment was unavailable at admission",
                )
            route = select_route_deployments(route, dispatchable.indexes)
            resolved_wires = dispatchable.resolved_wires
            _require_bound_wire_authority(
                None
                if continuation_context is None
                else continuation_context.required_route_binding,
                route,
                resolved_wires,
            )
        except NativeDialectUnavailableError as exc:
            return self._escalate_accepted(authorization, str(exc))
        except OpenAIProtocolError as exc:
            # A continuation whose bound provider authority is no longer
            # available is a CLIENT error (400 previous_response_not_found: resend
            # the full conversation), not a gateway-internal fault. Class the
            # durable failure by the public status so usage and health read it
            # as a client failure and it never pages as internal; the caller
            # still receives the exact public error unchanged.
            self._accounting.finish_request_quietly(
                authorization,
                GatewayFailure(
                    failure_class=(
                        GatewayFailureClass.INTERNAL
                        if exc.status_code >= 500
                        else GatewayFailureClass.INVALID_REQUEST
                    ),
                    safe_message=exc.detail.message,
                ),
            )
            raise NativeBridgeError(exc) from exc
        except Exception as exc:  # noqa: BLE001 - raised after route packaging below.
            probe_failure = exc
        if route is not None and self._native_route_eligible is not None:
            try:
                native_route_eligible = self._native_route_eligible(route, request)
            except Exception:  # noqa: BLE001 - hosted policy fails closed.
                native_route_eligible = False
            if not native_route_eligible:
                return self._escalate_accepted(
                    authorization,
                    "host policy does not permit native execution of this route",
                )

        # Admission returns the full ordered route; no attempt row exists
        # until the data plane's first `start_attempt`.
        public_request = request
        provider_request = request.model_copy(update={"stream": True, "include_usage": True})
        try:
            if probe_failure is not None or route is None or resolved_wires is None:
                raise probe_failure or GatewayRoutingError("authorized route did not resolve")
            route, resolved_wires, public_request, provider_request = admitted_route_requests(
                route,
                resolved_wires,
                request,
                accounting=self._accounting,
                authorization=authorization,
                continuation=continuation_context,
            )
            wire_route: list[JsonObject] = []
            parallel_disclosures: set[str] = set()
            signers: list[GatewayDispatchSigner | None] = []
            dispatch_bindings: list[FrozenDispatchBinding | None] = []
            carrier_authorities: list[ReasoningCarrierAuthority | None] = []
            for deployment, (profile, client) in zip(
                route.deployments, resolved_wires, strict=True
            ):
                dispatch = build_rung_dispatch(
                    route,
                    deployment,
                    profile,
                    client,
                    provider_request=provider_request,
                    public_request=public_request,
                    authorization=authorization,
                )
                if dispatch.parallel_disclosure is not None:
                    parallel_disclosures.add(dispatch.parallel_disclosure)
                wire_route.append(dispatch.wire_entry)
                signers.append(dispatch.signer)
                dispatch_bindings.append(dispatch.binding)
                carrier_authorities.append(dispatch.carrier_authority)
            public_request = fold_parallel_tool_call_disclosures(
                public_request,
                parallel_disclosures,
                accounting=self._accounting,
                authorization=authorization,
            )
            if continuation_context is not None:
                continuation_context.route_bindings = tuple(
                    continuation_route_binding(deployment, profile)
                    for deployment, (profile, _client) in zip(
                        route.deployments,
                        resolved_wires,
                        strict=True,
                    )
                )
        except NativeBridgeError:
            # The enriched fail-closed capability rejection above already
            # finished the accepted request; let it cross the boundary as-is.
            raise
        except (ProviderParameterError, ProviderCapabilityError) as exc:
            # One shared normalizer keeps both pre-dispatch rejections
            # field-specific: the parameter path names the parameter and the
            # capability path names the capability, so a triager sees which
            # request feature the route cannot preserve.
            failure = normalized_provider_failure(exc)
            if isinstance(exc, ProviderCapabilityError):
                public_error = _public_capability_error(
                    exc,
                    provider_request.surface,
                    public_stream=public_request.stream,
                    public_tools=bool(public_request.tools),
                    developer_messages_param=decoded.developer_messages_param,
                )
                # The ledger keeps the capability-free generic sentence, but a
                # bare "cannot preserve a requested capability" is untriageable
                # from an alert. Append the PUBLIC field the caller was told
                # about (never the internal literal), so operators read
                # "(field: stop)" without opening the request.
                failure = failure.model_copy(
                    update={
                        "safe_message": ledger_capability_message(
                            failure.safe_message, public_error.detail.param
                        )
                    }
                )
            else:
                public_error = public_failure_error(failure, param=exc.param)
            self._accounting.finish_request_quietly(authorization, failure)
            raise NativeBridgeError(public_error) from exc
        except GatewayRoutingError as exc:
            # A route/catalog that cannot be built during a rolling deploy is a
            # transient control-plane condition, not a bug: record it retryable
            # so it never pages as INTERNAL. The public error is already a 503.
            failure = gateway_updating_failure()
            self._accounting.finish_request_quietly(authorization, failure)
            raise _authority_error(exc) from exc
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            error = _authority_error(exc)
            failure = GatewayFailure(
                failure_class=GatewayFailureClass.INTERNAL,
                safe_message="gateway admission failed before provider dispatch",
            )
            # The public error and the ledger row carry only the sanitized
            # text, so this record is the ONLY place the real exception
            # survives: an unlogged INTERNAL here left a granted alias failing
            # 500 for hours with nothing to diagnose (platform staging,
            # 2026-09-03). The message names the request and alias; the
            # traceback rides exc_info. Nothing here carries a credential.
            _logger.exception(
                "gateway admission failed before provider dispatch",
                extra={
                    "operation": "native_admit",
                    "request_id": authorization.request_id,
                    "alias": authorization.alias,
                    "alias_revision_id": authorization.alias_revision_id,
                    "exception_type": type(exc).__name__,
                },
            )
            self._accounting.finish_request_quietly(authorization, failure)
            raise error from exc

        self._accounting.register(
            InflightRequest(
                authorization=authorization,
                route=route,
                request=provider_request,
                deadline_monotonic=deadline,
                continuation=continuation_context,
                policy=policy,
                signers=tuple(signers),
                dispatch_bindings=tuple(dispatch_bindings),
                reasoning_carrier_authorities=tuple(carrier_authorities),
                tier_forwarded_by_depth=tuple(
                    profile.forwards_tier(provider_request.service_tier)
                    for profile, _client in resolved_wires
                ),
            )
        )
        response: JsonObject = {
            "request_id": authorization.request_id,
            "alias": authorization.alias,
            "alias_revision_id": authorization.alias_revision_id,
            "stream": request.stream,
            "include_usage": request.include_usage,
            "exact_model_id": route.snapshot.exact_model_id,
            "route_reason": route.route_reason,
            "route": wire_route,
            "ignored_parameters": list(public_request.ignored_parameters),
            "maximum_total_attempts": MAXIMUM_TOTAL_ATTEMPTS,
            "maximum_same_deployment_attempts": MAXIMUM_SAME_DEPLOYMENT_ATTEMPTS,
            "refusal_failover": authorization.refusal_failover,
            "output_guardrail": bool(policy is not None and policy.output_checks),
        }
        if request.surface == GatewayApiSurface.RESPONSES:
            response["surface"] = "responses"
            response["envelope"] = responses_envelope(public_request)
        return json.dumps(response, separators=(",", ":"))

    def sign_dispatch(self, argument: str) -> str:
        """Sign one frozen dispatch body immediately before the provider POST.

        The data plane calls this after it acquires its bounded dispatch
        permit and immediately before the open attempt reserved by
        ``start_attempt``, so queue time can never age a signature toward
        AWS's short clock window; a same-deployment redial or a failover
        advance is a fresh physical attempt through ``start_attempt``, so it
        always signs afresh too.

        Args:
            argument: JSON object with ``request_id``, the exact ``url``, and
                the exact frozen ``body`` string the data plane will send.

        Returns:
            JSON object with the ``headers`` to send verbatim.

        Raises:
            NativeBridgeError: The attempt is unknown, its route depth
                carries no signer, or credential resolution failed.
        """
        data = json.loads(argument)
        entry = self._accounting.entry(str(data.get("request_id") or ""))
        signer = None
        binding = None
        if entry is not None and entry.active_attempt_id is not None:
            depth = entry.attempt_depths.get(entry.active_attempt_id)
            if depth is not None and depth < len(entry.signers):
                signer = entry.signers[depth]
            if depth is not None and depth < len(entry.dispatch_bindings):
                binding = entry.dispatch_bindings[depth]
        try:
            url = str(data["url"])
            body = str(data["body"])
            if (
                binding is None
                or url != binding.url
                or sha256_bytes(body.encode("utf-8")) != binding.body_sha256
            ):
                raise public_failure_error(
                    GatewayFailure(
                        failure_class=GatewayFailureClass.INTERNAL,
                        safe_message=(
                            "gateway dispatch differs from the admitted destination or frozen body"
                        ),
                    )
                )
            headers = dispatch_signature_headers(
                signer,
                url=url,
                body=body,
            )
        except OpenAIProtocolError as exc:
            raise NativeBridgeError(exc) from exc
        return json.dumps({"headers": headers}, separators=(",", ":"))

    def start_attempt(self, argument: str) -> str:
        """Reserve one physical dispatch through the accounting registry.

        Args:
            argument: JSON object with ``request_id``, ``attempt_ordinal``,
                optional ``current_depth``, and the optional classified
                ``failure``; see
                :meth:`NativeAttemptAccounting.start_attempt`.

        Returns:
            The registry's reservation or exhaustion disposition.

        Raises:
            NativeBridgeError: The reservation failed; the request is
                finalized before the error is raised.
        """
        return self._accounting.start_attempt(argument)

    def settle(self, argument: str) -> str:
        """Durably settle one reserved attempt through the accounting registry.

        Args:
            argument: JSON settlement payload; see
                :meth:`NativeAttemptAccounting.settle`.

        Returns:
            An empty JSON object; repeated settlement is a no-op.

        Raises:
            NativeBridgeError: The durable terminal write failed; the entry
                is retained so a retried settlement can still land.
        """
        return self._accounting.settle(argument)

    def abandon(self, argument: str) -> str:
        """Terminalize one accepted request through the accounting registry.

        Args:
            argument: JSON object with ``request_id`` and optional
                ``failure``; see :meth:`NativeAttemptAccounting.abandon`.

        Returns:
            An empty JSON object; an unknown request is a no-op.

        Raises:
            NativeBridgeError: The durable terminal write failed; the entry
                is kept so the deadline sweep can still close it.
        """
        return self._accounting.abandon(argument)

    def enforce_output(self, argument: str) -> str:
        """Run one output-chain callback for a native buffered completion."""
        data = json.loads(argument)
        request_id = str(data.get("request_id") or "")
        entry = self._accounting.entry(request_id)
        policy = None if entry is None else entry.policy
        deadline = time.monotonic() if entry is None else entry.deadline_monotonic
        return enforce_native_output(
            self._guardrails,
            policy,
            argument,
            deadline_monotonic=deadline,
        )

    def seal_reasoning_content(self, argument: str) -> str:
        """Seal one winning Fireworks turn before terminal settlement."""
        return seal_reasoning_carrier_content(self._accounting, argument)

    def claim_scope(self, argument: str) -> str:
        """Resolve the replay-store scope for one keyed request.

        The data plane owns the bounded in-process replay store; this call
        performs the decode and authorization once so the store key (tenant
        namespace, hashed caller operation, canonical request digest) is
        computed by exactly one implementation. The surface is part of the
        key, so keyed Chat Completions and keyed Responses operations never
        collide. A direct route whose provider has no native dialect is
        escalated before any replay claim, so an unservable caller operation
        never occupies the replay store; project targets resolve their
        deployment at admission through the same frozen selection, so their
        scope claims natively.

        Args:
            argument: JSON object with ``raw_key``, ``body``, optional
                ``surface`` (``"chat"`` or ``"responses"``, defaulting to
                chat), and optional ``idempotency_key`` and
                ``client_request_id`` header values.

        Returns:
            JSON replay scope with ``organization_id``, ``identity_id``,
            ``alias_revision_id``, ``surface``, ``caller_operation_sha256``,
            and ``canonical_request_sha256``, or an ``{"escalate": reason}``
            disposition naming why the native plane cannot serve the request.

        Raises:
            NativeBridgeError: Decoding or authorization failed.
        """
        data = json.loads(argument)
        decoded = self._decode_body(
            data["body"],
            surface=str(data.get("surface", "chat")),
            idempotency_key=optional_text(data.get("idempotency_key")),
            client_request_id=optional_text(data.get("client_request_id")),
        )
        request = decoded.request
        # Only the standard Idempotency-Key names a retriable operation;
        # client_request_id is a session correlation id real callers reuse
        # across distinct requests, so it never keys replay.
        caller_operation = request.idempotency_key
        if caller_operation is None:
            raise NativeBridgeError(
                OpenAIProtocolError(
                    status_code=400,
                    code="invalid_request",
                    message="A replay scope requires an Idempotency-Key header.",
                    param="Idempotency-Key",
                )
            )
        deadline = time.monotonic() + self._request_timeout_seconds
        try:
            authorization = self._components.store.authorize_request(
                raw_key=data["raw_key"],
                alias=decoded.alias,
                request=request,
                deadline_monotonic=deadline,
                app_referer=optional_text(data.get("app_referer")),
                app_title=optional_text(data.get("app_title")),
            )
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            raise _authority_error(exc) from exc
        if isinstance(authorization.target, DirectTarget):
            try:
                route = self._components.routes.resolve_direct(authorization)
                resolve_route_profiles(self._components.runtime_catalogs, route)
            except NativeDialectUnavailableError as exc:
                return _escalation(str(exc))
            except Exception:  # noqa: BLE001 - the owner's admission records this failure.
                pass
        key = replay_key(
            namespace=ProtocolNamespace(
                organization_id=authorization.organization_id,
                identity_id=authorization.identity_id,
                alias_revision_id=authorization.alias_revision_id,
            ),
            surface=request.surface,
            caller_operation=caller_operation,
            canonical_request_sha256=authorization.canonical_request_sha256,
        )
        if key is None:  # pragma: no cover - caller_operation is checked above.
            raise NativeBridgeError(
                OpenAIProtocolError(
                    status_code=500,
                    code="internal_error",
                    message="The gateway request failed.",
                    error_type="api_error",
                )
            )
        scope: JsonObject = {
            "organization_id": key.namespace.organization_id,
            "identity_id": key.namespace.identity_id,
            "alias_revision_id": key.namespace.alias_revision_id,
            "surface": key.surface.value,
            "caller_operation_sha256": key.caller_operation_sha256,
            "canonical_request_sha256": key.canonical_request_sha256,
        }
        return json.dumps(scope, separators=(",", ":"))

    def remember(self, argument: str) -> str:
        """Retain one finished Responses continuation within strict bounds.

        Args:
            argument: JSON object with ``request_id``, aggregated ``text``,
                ``refusal`` presence, and completed ``tool_calls``; an
                output-less turn carries all of them empty and is retained as
                the conversation so far.

        Returns:
            An empty JSON object; retention that does not apply (a
            ``store: false`` caller, a refusal) is a no-op.

        Raises:
            NativeBridgeError: The continuation exceeds the bounded store or
                a completed tool call carried malformed fields.
        """
        try:
            return remember_continuation(self._accounting, self._continuations, argument)
        except OpenAIProtocolError as exc:
            raise NativeBridgeError(exc) from exc
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            raise _authority_error(exc) from exc

    def _decode_body(
        self,
        body: str,
        *,
        surface: str = "chat",
        idempotency_key: str | None = None,
        client_request_id: str | None = None,
        anthropic_beta: str | None = None,
    ) -> DecodedGatewayRequest:
        """Decode one raw request body with the shared surface decoder."""
        try:
            return decode_native_body(
                body,
                surface=surface,
                idempotency_key=idempotency_key,
                client_request_id=client_request_id,
                anthropic_beta=anthropic_beta,
            )
        except NativeDecodeError as exc:
            raise NativeBridgeError(exc.error) from exc

    def _escalate_accepted(self, authorization: AuthorizationSnapshot, reason: str) -> str:
        """Finish one accepted-but-unservable request and return its disposition.

        The request was durably accepted before route probing, so the plane
        finalizes it content-free (no attempt row ever exists) before the
        escalation disposition tells the data plane to fail the request
        closed.

        Args:
            authorization: Frozen authority for the accepted request.
            reason: Display-safe reason the native path cannot serve it.

        Returns:
            The JSON admission body carrying the escalation disposition.
        """
        self._accounting.finish_request_quietly(
            authorization,
            GatewayFailure(
                failure_class=GatewayFailureClass.INTERNAL,
                safe_message="the native engine cannot serve the authorized route",
            ),
        )
        return _escalation(reason)

    def _resolve_route(
        self,
        authorization: AuthorizationSnapshot,
        request: GatewayRequest,
        *,
        continuation: ContinuationContext | None = None,
    ) -> GatewayRoute:
        """Resolve one direct or project route; see ``resolve_admission_route``."""
        return resolve_admission_route(
            self._components, authorization, request, continuation=continuation
        )
