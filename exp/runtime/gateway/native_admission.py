"""Route admission with disclosed coercion for the native control plane.

Admission prefers rungs that preserve every caller semantic verbatim: the
generation-control narrowing and the per-deployment capability preflight plus
payload build both run here. When no rung preserves the request verbatim,
this module applies the capability-preservation policy's minimal disclosed
coercion exactly once per layer and re-selects; when nothing coercible
remains, the first rung's own field-scoped rejection stays the answer. A
coercion is never
silent: every substitution is disclosed through ``ignored_parameters`` in
``path->effective`` form, warn-logged for operators, and counted in the
admission metrics.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from exp.common.models.catalog import GatewayDeploymentCapabilities
from exp.runtime.gateway.affinity import (
    affinity_fingerprint,
    affinity_seed_material,
    rendezvous_order,
)
from exp.runtime.gateway.contracts import AuthorizationSnapshot, DirectTarget, GatewayRequest
from exp.runtime.gateway.native_accounting import NativeAttemptAccounting
from exp.runtime.gateway.native_components import NativeGatewayComponents
from exp.runtime.gateway.native_execution import (
    reorder_route_deployments,
    request_carries_cache_markers,
    select_route_deployments,
)
from exp.runtime.gateway.native_responses import ContinuationContext
from exp.runtime.gateway.prompt_cache_affinity import provider_prompt_cache_key
from exp.runtime.gateway.prompt_size import require_prompt_fits_context_window
from exp.runtime.gateway.routing import GatewayRoute, GatewayRoutingError
from exp.runtime.models.providers import (
    emulated_gateway_capabilities,
    preflight_gateway_request,
)
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.capability_policy import (
    coerce_generation_parameters,
    coerce_route_rejections,
    coerce_strict_tool_schemas,
    coerce_structured_text_schema,
)
from exp.runtime.models.providers.errors import (
    ProviderCapabilityError,
    ProviderParameterError,
)
from exp.runtime.models.providers.generation_route_compat import (
    compatible_generation_parameter_profile_indexes,
)
from exp.runtime.models.providers.protocol import NativeWireClient
from exp.runtime.models.providers.streaming_requests import (
    dialect_stream_payload,
    route_generation_parameter_requests,
)
from exp.runtime.openai_protocol.state import ProtocolNamespace, episode_namespace

_logger = logging.getLogger(__name__)

_ResolvedWires = tuple[tuple[GatewayWireProfile, NativeWireClient], ...]


def _with_cache_affinity(
    provider_request: GatewayRequest, authorization: AuthorizationSnapshot
) -> GatewayRequest:
    """Attach the tenant's cache-affinity key to the final dispatch request.

    Cache affinity is per tenant and per session, so it is derived once, from
    the request admission settled on and the frozen authority, and read by
    every rung's payload builder that forwards it. It is applied last so no
    admission-time rebuild (route narrowing, capability or schema coercion)
    can drop it; the public request keeps the caller's own ``prompt_cache_key``
    untouched.
    """
    return provider_request.model_copy(
        update={
            "provider_prompt_cache_key": provider_prompt_cache_key(
                provider_request,
                organization_id=str(authorization.organization_id),
                identity_id=str(authorization.identity_id),
            ),
        }
    )


def admitted_route_requests(
    route: GatewayRoute,
    resolved_wires: _ResolvedWires,
    request: GatewayRequest,
    *,
    accounting: NativeAttemptAccounting,
    authorization: AuthorizationSnapshot,
    continuation: ContinuationContext | None = None,
) -> tuple[GatewayRoute, _ResolvedWires, GatewayRequest, GatewayRequest]:
    """Narrow one certified route to rungs that serve the admitted request.

    Args:
        route: Frozen route aligned with ``resolved_wires``.
        resolved_wires: Ordered wire profiles and clients per deployment.
        request: Canonical request produced by the public protocol decoder.
        accounting: Shared accounting owning the coercion counter.
        authorization: Frozen authority for the accepted request.
        continuation: Responses continuation context when this request
            continues a stored response; its original episode key keeps the
            conversation's cache-affinity placement.

    Returns:
        The narrowed route and wires plus the public request (carrying any
        coercion disclosures) and the streaming-forced provider request.

    Raises:
        ProviderParameterError: No rung preserves a generation control and no
            disclosed coercion applies, or rungs declined for different
            field-specific reasons.
        ProviderCapabilityError: The first rung's capability rejection when
            no rung is protocol-compatible; the shared admit handler scopes
            it to the exact public request field.
        GatewayRoutingError: No rung is protocol-compatible and none named a
            rejection.
    """
    # flex/priority are the tiers we price as an OPT-IN pass-through, so they
    # fail CLOSED before any reservation when no rung can BILL the requested one:
    # a BYOK rung forwards any tier (customer pays the provider directly, no
    # platform card needed), while a house rung must carry a per-tier card for
    # THIS tier (`forwards_tier`). A model carded for flex only therefore rejects
    # a priority request instead of forwarding it and silently billing the base
    # rate while the provider charges the priority premium (underbill). Every
    # OTHER tier (auto/default carry no price; scale and any future value) is
    # never rejected here — a non-billable candidate simply strips it at payload
    # build (billing-safe, disclosed), so only the opt-in priced tiers gate.
    # A prompt that cannot fit any rung's context window is refused HERE, before
    # a reservation or a provider call: the provider would only 400 it back
    # (charging nothing but costing a round trip and an opaque message).
    require_prompt_fits_context_window(route, request)

    if request.service_tier in ("flex", "priority"):
        tier = request.service_tier
        if not any(profile.forwards_tier(tier) for profile, _client in resolved_wires):
            raise ProviderCapabilityError(
                capability="service_tier",
                detail=(
                    "This model does not offer a flex or priority processing tier. "
                    "Remove service_tier, or choose a model with tiered pricing enabled."
                ),
            )

    admitted_request = request
    coercion_disclosures: tuple[str, ...] = ()
    full_route = route
    full_wires = resolved_wires

    def candidate_serves(candidate: GatewayRequest) -> bool:
        return _candidate_serves(full_route, full_wires, candidate)

    try:
        compatible_indexes = compatible_generation_parameter_profile_indexes(
            tuple(profile for profile, _client in resolved_wires),
            admitted_request,
        )
    except ProviderParameterError:
        # No rung preserves the request verbatim; retry once with the
        # minimal disclosed coercion when semantics allow, otherwise
        # keep the named rejection.
        coercion = coerce_generation_parameters(
            tuple(profile for profile, _client in resolved_wires),
            admitted_request,
            admits=candidate_serves,
        )
        if coercion is None:
            raise
        compatible_indexes = compatible_generation_parameter_profile_indexes(
            tuple(profile for profile, _client in resolved_wires),
            coercion.request,
        )
        admitted_request = coercion.request
        coercion_disclosures = coercion.disclosures
    route = select_route_deployments(route, compatible_indexes)
    resolved_wires = tuple(resolved_wires[index] for index in compatible_indexes)
    public_request, provider_request = route_generation_parameter_requests(
        tuple(profile for profile, _client in resolved_wires),
        admitted_request,
    )
    provider_request = provider_request.model_copy(update={"stream": True, "include_usage": True})
    protocol_indexes, protocol_errors = protocol_compatible_indexes(
        route,
        resolved_wires,
        provider_request,
        public_stream=public_request.stream,
    )
    if not protocol_indexes:
        # Degrade once with disclosure where the rejection set allows it:
        # a unanimous capability rejection coerces any coercible capability,
        # mixed rejections only the service-tier hint.
        coercion = coerce_route_rejections(
            protocol_errors, len(route.deployments), admitted_request
        )
        if coercion is not None:
            admitted_request = coercion.request
            coercion_disclosures = (*coercion_disclosures, *coercion.disclosures)
            public_request, provider_request = route_generation_parameter_requests(
                tuple(profile for profile, _client in resolved_wires),
                admitted_request,
            )
            provider_request = provider_request.model_copy(
                update={"stream": True, "include_usage": True}
            )
            protocol_indexes, protocol_errors = protocol_compatible_indexes(
                route,
                resolved_wires,
                provider_request,
                public_stream=public_request.stream,
            )
    if not protocol_indexes and provider_request.parallel_tool_calls is not None:
        # LAST resort for parallel_tool_calls: no rung honours the control
        # natively (and no other coercion freed one), so admit the rungs whose
        # only objection is that control. The data plane then drops `true`
        # (the provider's default) or serializes `false` per rung, disclosed
        # (native_bridge's per-rung shaping). A native rung is always
        # preferred, which is why this pass runs after everything else.
        protocol_indexes, protocol_errors = protocol_compatible_indexes(
            route,
            resolved_wires,
            provider_request,
            public_stream=public_request.stream,
            emulate_parallel_tool_calls=True,
        )
    if not protocol_indexes:
        if not protocol_errors:
            raise GatewayRoutingError("authorized route has no compatible deployment")
        # Nothing coercible remains; the shared admit handler scopes
        # capability rejections to their exact public request field.
        raise route_rejection(protocol_errors)
    if len(protocol_indexes) != len(route.deployments):
        selected_indexes = tuple(protocol_indexes)
        route = select_route_deployments(route, selected_indexes)
        resolved_wires = tuple(resolved_wires[index] for index in selected_indexes)
        public_request, provider_request = route_generation_parameter_requests(
            tuple(profile for profile, _client in resolved_wires),
            admitted_request,
        )
        provider_request = provider_request.model_copy(
            update={"stream": True, "include_usage": True}
        )
    # The surviving rungs decide whether the structured-output schema and the
    # strict tool schemas need their objects closed; a route that lost every
    # Anthropic rung above is dispatched with the caller's schemas verbatim.
    surviving_profiles = tuple(profile for profile, _client in resolved_wires)
    for coerce_schema in (coerce_structured_text_schema, coerce_strict_tool_schemas):
        coercion = coerce_schema(surviving_profiles, admitted_request)
        if coercion is None:
            continue
        admitted_request = coercion.request
        coercion_disclosures = (*coercion_disclosures, *coercion.disclosures)
        public_request, provider_request = route_generation_parameter_requests(
            surviving_profiles,
            admitted_request,
        )
        provider_request = provider_request.model_copy(
            update={"stream": True, "include_usage": True}
        )
    if coercion_disclosures:
        record_admission_coercions(accounting, authorization, coercion_disclosures)
        public_request = public_request.model_copy(
            update={
                "ignored_parameters": tuple(
                    dict.fromkeys((*public_request.ignored_parameters, *coercion_disclosures))
                )
            }
        )
    provider_request = _with_cache_affinity(provider_request, authorization)
    route, resolved_wires = _prefer_cache_capable_rungs(route, resolved_wires, provider_request)
    route, resolved_wires = _affinity_ordered_rungs(
        route,
        resolved_wires,
        provider_request,
        authorization=authorization,
        continuation=continuation,
    )
    return route, resolved_wires, public_request, provider_request


def route_rejection(
    errors: Sequence[ProviderParameterError | ProviderCapabilityError],
) -> ProviderParameterError | ProviderCapabilityError:
    """Choose the one rejection the caller can act on when no rung serves.

    Rungs decline for their own reasons, and the first rung's reason is not
    always the caller's remedy. A ladder whose text-only rung refuses any
    image while an inline-only rung refuses just the remote URL can still
    carry the picture: the caller inlines the bytes. Reporting the text-only
    rung's refusal would tell them to drop the image instead. The URL
    rejection therefore wins whenever some rung raised it; otherwise the
    first rung's own rejection stays the answer. A provider-scoped media
    handle is the same story one step further: the rejection that names the
    provider holding the upload beats a rung that merely declares no handle
    support, since only the named provider can ever resolve the handle.

    Args:
        errors: One rejection per declined deployment, in route order.

    Returns:
        The rejection to surface to the caller.
    """
    for preferred in ("media_handle_provider", "image_url_input"):
        for error in errors:
            if isinstance(error, ProviderCapabilityError) and error.capability == preferred:
                return error
    return errors[0]


def _prefer_cache_capable_rungs(
    route: GatewayRoute,
    resolved_wires: _ResolvedWires,
    provider_request: GatewayRequest,
) -> tuple[GatewayRoute, _ResolvedWires]:
    """Dispatch marker-honoring rungs first on cache-preserving pools.

    Under ``maximize_cache`` a cache-marked request must never start on a
    wire that structurally drops its markers while a marker-honoring rung
    stands ready: the pool's whole policy is prefix-cache preservation, and
    a marker-dropping first rung silently bills every turn's full context
    uncached (measured ~10x on a large system prompt). The reorder is
    stable within each group, so certified order still breaks ties, and a
    route that narrowing left with NO marker-honoring rung is unchanged
    here: the dropped markers are already disclosed through the
    ``cache_control`` ``ignored_parameters`` entries. ``maximize_availability``
    pools keep their certified order untouched.
    """
    if route.snapshot.failover_mode != "maximize_cache":
        return route, resolved_wires
    if len(resolved_wires) < 2 or not request_carries_cache_markers(provider_request):
        return route, resolved_wires
    marker_capable = tuple(
        index
        for index, (profile, _client) in enumerate(resolved_wires)
        if profile.dialect == "anthropic_messages"
    )
    if not marker_capable or len(marker_capable) == len(resolved_wires):
        return route, resolved_wires
    order = (
        *marker_capable,
        *(index for index in range(len(resolved_wires)) if index not in marker_capable),
    )
    return (
        reorder_route_deployments(route, order),
        tuple(resolved_wires[index] for index in order),
    )


def _affinity_ordered_rungs(
    route: GatewayRoute,
    resolved_wires: _ResolvedWires,
    provider_request: GatewayRequest,
    *,
    authorization: AuthorizationSnapshot,
    continuation: ContinuationContext | None,
) -> tuple[GatewayRoute, _ResolvedWires]:
    """Dispatch rungs in weighted rendezvous order on affinity pools.

    Under ``maximize_cache_affinity`` the certified order is replaced by the
    request fingerprint's rendezvous permutation over the surviving rungs, so
    every worker sends one conversation to the same rung and, when that rung
    sheds or dies, to the same deterministic alternate. Weights come from each
    deployment's authored ``GatewayRungDispatchPolicy.affinity_weight``
    (default 1.0). The cache-marker guarantee composes: a cache-marked request
    on a route mixing marker-honoring and marker-dropping wires still
    dispatches the marker-honoring group first, rendezvous-ordered within each
    group. The other two failover modes are untouched.
    """
    if route.snapshot.failover_mode != "maximize_cache_affinity":
        return route, resolved_wires
    if len(resolved_wires) < 2:
        return route, resolved_wires
    material = affinity_seed_material(
        provider_request,
        continuation_episode_key=None if continuation is None else continuation.episode_key,
        request_id=authorization.request_id,
    )
    fingerprint = affinity_fingerprint(
        organization_id=authorization.organization_id,
        identity_id=authorization.identity_id,
        material=material,
    )
    weighted_rungs = tuple(
        (
            deployment.deployment_id,
            (
                1.0
                if deployment.gateway.dispatch is None
                or deployment.gateway.dispatch.affinity_weight is None
                else deployment.gateway.dispatch.affinity_weight
            ),
        )
        for deployment in route.deployments
    )
    order = rendezvous_order(fingerprint, weighted_rungs)
    if request_carries_cache_markers(provider_request):
        marker_capable = frozenset(
            index
            for index, (profile, _client) in enumerate(resolved_wires)
            if profile.dialect == "anthropic_messages"
        )
        if marker_capable and len(marker_capable) < len(resolved_wires):
            order = (
                *(index for index in order if index in marker_capable),
                *(index for index in order if index not in marker_capable),
            )
    return (
        reorder_route_deployments(route, order),
        tuple(resolved_wires[index] for index in order),
    )


def _candidate_serves(
    route: GatewayRoute,
    resolved_wires: _ResolvedWires,
    candidate: GatewayRequest,
) -> bool:
    """Probe the full admission pipeline for one coercion candidate.

    The policy layer sees only wire profiles, so without this probe a
    candidate could pass generation narrowing yet land on rungs that all fail
    deployment capability preflight, blocking a farther candidate whose rungs
    serve. The probe mirrors the real pipeline exactly, including the single
    capability coercion admission may run afterwards: clearing one capability
    can merely expose the next, so the coerced candidate must itself pass
    preflight before the snap counts as servable.

    Args:
        route: Frozen full route aligned with ``resolved_wires``.
        resolved_wires: Ordered wire profiles and clients per deployment.
        candidate: One coercion candidate request.

    Returns:
        Whether admission would serve the candidate.
    """
    try:
        candidate_indexes = compatible_generation_parameter_profile_indexes(
            tuple(profile for profile, _client in resolved_wires),
            candidate,
        )
        candidate_route = select_route_deployments(route, candidate_indexes)
        candidate_wires = tuple(resolved_wires[index] for index in candidate_indexes)
        candidate_public, candidate_provider = route_generation_parameter_requests(
            tuple(profile for profile, _client in candidate_wires),
            candidate,
        )
    except (ProviderParameterError, ProviderCapabilityError):
        return False
    candidate_provider = candidate_provider.model_copy(
        update={"stream": True, "include_usage": True}
    )
    indexes, errors = protocol_compatible_indexes(
        candidate_route,
        candidate_wires,
        candidate_provider,
        public_stream=candidate_public.stream,
    )
    if indexes:
        return True
    capability_coercion = coerce_route_rejections(
        errors, len(candidate_route.deployments), candidate
    )
    if capability_coercion is None:
        return False
    try:
        _coerced_public, coerced_provider = route_generation_parameter_requests(
            tuple(profile for profile, _client in candidate_wires),
            capability_coercion.request,
        )
    except (ProviderParameterError, ProviderCapabilityError):
        return False
    coerced_provider = coerced_provider.model_copy(update={"stream": True, "include_usage": True})
    coerced_indexes, _coerced_errors = protocol_compatible_indexes(
        candidate_route,
        candidate_wires,
        coerced_provider,
        public_stream=candidate_public.stream,
    )
    return bool(coerced_indexes)


def protocol_compatible_indexes(
    route: GatewayRoute,
    resolved_wires: _ResolvedWires,
    provider_request: GatewayRequest,
    *,
    public_stream: bool | None,
    emulate_parallel_tool_calls: bool = False,
) -> tuple[tuple[int, ...], tuple[ProviderParameterError | ProviderCapabilityError, ...]]:
    """Select rungs that pass capability preflight and payload build.

    Args:
        route: Frozen route aligned with ``resolved_wires``.
        resolved_wires: Ordered wire profiles and clients per deployment.
        provider_request: Streaming-forced request to validate.
        public_stream: The caller's declared streaming intent.

    Returns:
        Ordered compatible indexes and every rung's rejection in route
        order, so the caller can distinguish a route-wide capability gap
        from rungs declining for different reasons.
    """
    indexes: list[int] = []
    errors: list[ProviderParameterError | ProviderCapabilityError] = []
    for index, (deployment, (profile, _client)) in enumerate(
        zip(route.deployments, resolved_wires, strict=True)
    ):
        try:
            preflight_gateway_request(
                provider_request,
                deployment.gateway.capabilities,
                model_capabilities=deployment.capabilities,
                public_stream=public_stream,
                route_provider=deployment.provider,
                emulated_capabilities=emulated_gateway_capabilities(
                    profile.dialect, emulate_parallel_tool_calls=emulate_parallel_tool_calls
                ),
            )
            dialect_stream_payload(profile, provider_request)
        except (ProviderParameterError, ProviderCapabilityError) as exc:
            errors.append(exc)
            continue
        indexes.append(index)
    return tuple(indexes), tuple(errors)


def shape_parallel_tool_calls(
    request: GatewayRequest,
    capabilities: GatewayDeploymentCapabilities,
) -> tuple[GatewayRequest, str | None]:
    """Shape ``parallel_tool_calls`` for one rung that may lack the control.

    A rung whose wire carries the control forwards it verbatim. One that does
    not gets ``true`` dropped (parallel calls are the provider's own default)
    or ``false`` emulated: the data plane serializes that rung's stream to one
    tool call per turn (``serialize_tool_calls``). Either way the caller reads
    the disclosure in ``ignored_parameters``.

    Args:
        request: The streaming-forced provider request.
        capabilities: The rung's deployment capability declaration.

    Returns:
        The request to build this rung's payload from, and the disclosure to
        publish (``None`` when nothing changed).
    """
    if request.parallel_tool_calls is None or capabilities.supports_parallel_tool_calls:
        return request, None
    if request.parallel_tool_calls:
        return (
            request.model_copy(update={"parallel_tool_calls": None}),
            "parallel_tool_calls->dropped(provider_default)",
        )
    return (
        request.model_copy(update={"parallel_tool_calls": None, "serialize_tool_calls": True}),
        "parallel_tool_calls->emulated(serialized_by_gateway)",
    )


def fold_parallel_tool_call_disclosures(
    public_request: GatewayRequest,
    disclosures: set[str],
    *,
    accounting: NativeAttemptAccounting,
    authorization: AuthorizationSnapshot,
) -> GatewayRequest:
    """Publish per-rung parallel-tool shaping like any other admission coercion.

    Args:
        public_request: The public request the admission answer carries.
        disclosures: Distinct disclosures the per-rung shaping produced.
        accounting: Shared accounting owning the coercion counter.
        authorization: Frozen authority for the accepted request.

    Returns:
        The public request with the disclosures folded into
        ``ignored_parameters`` (unchanged when there are none).
    """
    if not disclosures:
        return public_request
    ordered = tuple(sorted(disclosures))
    record_admission_coercions(accounting, authorization, ordered)
    return public_request.model_copy(
        update={
            "ignored_parameters": tuple(
                dict.fromkeys((*public_request.ignored_parameters, *ordered))
            )
        }
    )


def record_admission_coercions(
    accounting: NativeAttemptAccounting,
    authorization: AuthorizationSnapshot,
    disclosures: tuple[str, ...],
) -> None:
    """Log and count one admission's disclosed request coercions.

    A coercion is never silent: the caller sees it in
    ``ignored_parameters``, the log names it for operators, and the
    metrics snapshot counts it so a persistently coerced alias reaches a
    human instead of quietly serving degraded semantics forever.

    Args:
        accounting: Shared accounting owning the coercion counter.
        authorization: Frozen authority for the accepted request.
        disclosures: Path->effective disclosure strings applied.
    """
    accounting.record_admission_coercions(len(disclosures))
    _logger.warning(
        "gateway admission coerced request semantics for alias %r: %s "
        "(disclosed through ignored_parameters)",
        authorization.alias,
        ", ".join(disclosures),
    )


def resolve_admission_route(
    components: NativeGatewayComponents,
    authorization: AuthorizationSnapshot,
    request: GatewayRequest,
    *,
    continuation: ContinuationContext | None = None,
) -> GatewayRoute:
    """Resolve one direct or project route without an event loop.

    Direct pools resolve entirely inside frozen in-memory catalogs. Project
    targets run frozen learned selection synchronously on this worker thread
    through the shared selection seam and episode identity derivation, so
    there is exactly one policy execution path. A Responses continuation
    carries its original turn's episode key, so a continued request joins the
    same selection episode instead of re-running request-time embedding for a
    fresh one. Request-time embedding failure falls back to the frozen
    conservative baseline inside the shared runtime, and neither path mutates
    policy or evidence.
    """
    if isinstance(authorization.target, DirectTarget):
        return components.routes.resolve_direct(authorization)
    if continuation is not None:
        episode = (
            authorization.organization_id,
            authorization.identity_id,
            authorization.alias_revision_id,
            continuation.episode_key,
        )
    else:
        episode = episode_namespace(
            namespace=ProtocolNamespace(
                organization_id=authorization.organization_id,
                identity_id=authorization.identity_id,
                alias_revision_id=authorization.alias_revision_id,
            ),
            # The session-scoped correlation id is the stronger affinity
            # scope; a per-operation idempotency key only pins retries.
            caller_episode_key=request.client_request_id or request.idempotency_key,
            request_id=authorization.request_id,
        )
    return components.routes.resolve_project_blocking(
        authorization=authorization,
        request=request,
        episode_namespace=episode,
    )
