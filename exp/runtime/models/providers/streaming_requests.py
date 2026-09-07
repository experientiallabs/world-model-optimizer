"""Canonical gateway request translation for launch-provider streaming protocols."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayNamedToolChoice,
    GatewayRequest,
)
from exp.runtime.models.providers.dialect_dispatch import (
    SERVICE_TIER_DIALECTS as SERVICE_TIER_DIALECTS,
)
from exp.runtime.models.providers.dialect_dispatch import (
    TOOL_RESULT_IMAGE_DROP_DISCLOSURE as TOOL_RESULT_IMAGE_DROP_DISCLOSURE,
)
from exp.runtime.models.providers.dialect_dispatch import (
    TOOL_RESULT_IMAGE_PLACEHOLDER as TOOL_RESULT_IMAGE_PLACEHOLDER,
)
from exp.runtime.models.providers.dialect_dispatch import (
    dialect_stream_payload as dialect_stream_payload,
)
from exp.runtime.models.providers.dialect_dispatch import (
    fireworks_continuation_required as fireworks_continuation_required,
)
from exp.runtime.models.providers.dialect_dispatch import (
    strip_tool_result_images as strip_tool_result_images,
)
from exp.runtime.models.providers.errors import (
    ProviderParameterError,
    UnsupportedReasoningEffortError,
)
from exp.runtime.models.providers.fireworks import (
    require_responses_continuation_channel,
)
from exp.runtime.models.providers.generation_parameter_validation import (
    anthropic_reasoning_disengaged,
    mid_conversation_system_present,
    require_assistant_prefill_supported,
    serves_reasoning_summary,
)
from exp.runtime.models.providers.generation_parameter_validation import (
    effective_profile_reasoning_effort as _effective_profile_reasoning_effort,
)
from exp.runtime.models.providers.generation_parameter_validation import (
    profile_reasoning_efforts as _profile_reasoning_efforts,
)
from exp.runtime.models.providers.generation_parameter_validation import (
    require_route_numeric_parameter as _require_route_numeric_parameter,
)
from exp.runtime.models.providers.messages_payloads import (
    anthropic_messages_stream_payload as anthropic_messages_stream_payload,
)
from exp.runtime.models.providers.messages_payloads import (
    bedrock_converse_stream_payload as bedrock_converse_stream_payload,
)
from exp.runtime.models.providers.messages_payloads import (
    gemini_generate_content_stream_payload as gemini_generate_content_stream_payload,
)
from exp.runtime.models.providers.openai_payloads import (
    openai_compatible_stream_payload as openai_compatible_stream_payload,
)
from exp.runtime.models.providers.openai_payloads import (
    openai_responses_stream_payload as openai_responses_stream_payload,
)
from exp.runtime.models.providers.reasoning_compat import (
    REASONING_EFFORTS,
    anthropic_adaptive_only_thinking,
    anthropic_budgeted_enabled_only,
)
from exp.runtime.models.providers.server_tools import (
    anthropic_server_tool_names,
    anthropic_server_tools_message,
    anthropic_server_tools_present,
)

if TYPE_CHECKING:
    from exp.runtime.models.providers.base import GatewayWireProfile

_logger = logging.getLogger(__name__)

_ANTHROPIC_REQUIRED_MAX_TOKENS_DEFAULT = 4096

TOOL_ERROR_FOLD_DISCLOSURE = "messages.content.is_error->content"
"""Disclosure recorded when a tool-result error flag folds into result text.

Only the Anthropic wire has a ``tool_result.is_error`` field; every other
wire receives the flag as a fixed text prefix on the result (see
``folded_tool_error_content``). The flag is baked into the caller's history
on every failed tool call, so a rejection wedges the whole session, and a
silent drop would misstate that the invocation failed.
"""

OPENAI_MINIMUM_OUTPUT_TOKENS = 16
"""Smallest ``max_output_tokens`` the OpenAI wires accept.

The provider rejects lower values by name ("Expected a value >= 16"), while
the Anthropic surface legally carries ``max_tokens`` down to 1, so
Messages-surface requests below the floor are raised to it with disclosure.
"""

GATEWAY_GENERATION_PARAMETER_CONTRACT_VERSION = 2
"""Version of the route admission and provider wire-translation contract."""

_STRICT_STRUCTURED_OUTPUT_DIALECTS = frozenset(
    {"anthropic_messages", "gemini_generate_content", "bedrock_converse_stream"}
)
_NO_PARALLEL_TOOL_CONTROL_DIALECTS = frozenset(
    {"gemini_generate_content", "bedrock_converse_stream"}
)


def route_generation_parameter_requests(
    profiles: Sequence[GatewayWireProfile],
    request: GatewayRequest,
) -> tuple[GatewayRequest, GatewayRequest]:
    """Apply one stable generation-control policy across a provider waterfall.

    A caller-visible semantic parameter is forwarded only when every deployment
    in the certified route supports its exact value. Unsupported or out-of-range
    values fail locally before dispatch. Controls that are provable no-ops, such
    as tool selection without any tool definitions, are removed and disclosed
    through ``ignored_parameters`` on the public request.

    Args:
        profiles: Ordered wire profiles for every deployment in the route.
        request: Decoded public request before provider streaming is forced.

    Returns:
        A pair of ``(public_request, provider_request)``. The public copy keeps
        caller values for response reflection and adds no-op-field disclosure;
        the provider copy removes only controls that cannot change semantics.

    Raises:
        ProviderParameterError: A semantic control is unsupported or invalid
            anywhere in the route.
        ValueError: The route has no wire profiles.
    """
    if not profiles:
        raise ValueError("generation parameter shaping requires at least one wire profile")
    for profile in profiles:
        if fireworks_continuation_required(profile, request):
            require_responses_continuation_channel(request)

    ignored = list(request.ignored_parameters)
    provider_updates: dict[str, object] = {}

    def ignore(field: str, public_path: str | None = None) -> None:
        """Drop one provider field and record the public path reported as ignored."""
        provider_updates[field] = None
        path = public_path or field
        if path not in ignored:
            ignored.append(path)

    if request.maximum_output_tokens is not None:
        route_limits = tuple(
            profile.maximum_output_tokens
            for profile in profiles
            if profile.maximum_output_tokens is not None
        )
        if route_limits and request.maximum_output_tokens > min(route_limits):
            maximum = min(route_limits)
            param = request.maximum_output_tokens_parameter or "max_tokens"
            raise ProviderParameterError(
                message=(
                    f"The value {request.maximum_output_tokens!r} for {param!r} exceeds this "
                    f"model route's maximum of {maximum}."
                ),
                param=param,
                code="invalid_parameter",
            )
        if (
            request.maximum_output_tokens < OPENAI_MINIMUM_OUTPUT_TOKENS
            and (
                request.surface == GatewayApiSurface.MESSAGES
                and any(
                    profile.dialect in {"openai_responses", "openai_compatible"}
                    for profile in profiles
                )
                # A Chat value keeps native semantics on a chat wire (some
                # compatible providers accept an output ceiling of 1); only
                # the TRANSLATED Responses wire imposes OpenAI's minimum, so
                # the floor covers exactly the routes that translate.
                or request.surface == GatewayApiSurface.CHAT_COMPLETIONS
                and any(profile.dialect == "openai_responses" for profile in profiles)
            )
            # The floored value must stay within every rung's declared output
            # ceiling; a route capped below the floor keeps the caller value
            # and the provider's own rejection.
            and (not route_limits or min(route_limits) >= OPENAI_MINIMUM_OUTPUT_TOKENS)
        ):
            # Anthropic and Chat Completions accept output ceilings down to
            # 1 (Claude Code probes with exactly that after a /model switch)
            # while OpenAI rejects max_output_tokens below 16, so a Messages
            # or Chat value translated onto an OpenAI Responses rung rides
            # the provider floor with disclosure instead of surfacing a
            # provider 400 the caller cannot act on (2026-09-05 stragglers).
            # A native Responses caller keeps the named admission rejection
            # below: sub-16 is invalid on its own surface.
            provider_updates["maximum_output_tokens"] = OPENAI_MINIMUM_OUTPUT_TOKENS
            parameter = request.maximum_output_tokens_parameter or "max_tokens"
            path = f"{parameter}->{OPENAI_MINIMUM_OUTPUT_TOKENS}"
            if path not in ignored:
                ignored.append(path)
    elif any(profile.dialect == "anthropic_messages" for profile in profiles):
        # Anthropic requires max_tokens even when the public surface does not.
        # Pin one route-wide default so every waterfall rung sees the same
        # output budget, bounded by the smallest known model ceiling.
        route_limits = tuple(
            profile.maximum_output_tokens
            for profile in profiles
            if profile.maximum_output_tokens is not None
        )
        provider_updates["maximum_output_tokens"] = min(
            (_ANTHROPIC_REQUIRED_MAX_TOKENS_DEFAULT, *route_limits)
        )
    # The rejection names the field the CALLER sent: Claude Code carries its
    # effort as Messages output_config.effort and auto-recovers (drops the
    # field and retries) only when the 400 names that channel, so naming the
    # translated internal field wedges every turn instead (issue #795).
    if request.surface == GatewayApiSurface.RESPONSES:
        effort_path = "reasoning.effort"
    elif request.surface == GatewayApiSurface.MESSAGES:
        effort_path = "output_config.effort"
    else:
        effort_path = "reasoning_effort"

    def profile_reasoning_effort(profile: GatewayWireProfile) -> str | None:
        """Return the caller effort or this wire's required provider default."""
        return _effective_profile_reasoning_effort(profile, request.reasoning_effort)

    def sampling_declared(profile: GatewayWireProfile, *, top_p: bool = False) -> bool:
        """Return whether one rung declares this sampling control at all."""
        return profile.supports_top_p is True if top_p else profile.supports_temperature

    def sampling_supported(profile: GatewayWireProfile, *, top_p: bool = False) -> bool:
        """Return whether one rung accepts this request's sampling mode."""
        return sampling_declared(profile, top_p=top_p) and (
            not profile.sampling_requires_reasoning_none
            or profile_reasoning_effort(profile) == "none"
            # A budgeted-enabled Anthropic rung has no "none" effort on its
            # ladder; its srn hatch opens when the dispatch sends no thinking
            # budget at all, so ordinary thinking-off sampling is honored.
            or (
                profile.reasoning_wire_format == "anthropic_adaptive"
                and anthropic_reasoning_disengaged(request)
            )
        )

    def srn_only_block(*, top_p: bool = False) -> bool:
        """Return whether the ONLY reason the route blocks this control is srn.

        Every rung declares the control, but at least one is a reasoning route
        that accepts sampling only at ``reasoning_effort=none`` and is not at
        none here. Such a control is dropped-and-disclosed rather than rejected:
        the model does accept it, just not at this effort. A rung that does not
        declare the control at all (Anthropic constrained sampling) is a genuine
        unsupported case and is NOT covered here, so it still hard-rejects.
        """
        return all(sampling_declared(profile, top_p=top_p) for profile in profiles) and not all(
            sampling_supported(profile, top_p=top_p) for profile in profiles
        )

    # Sampling a rung cannot carry is DROPPED with disclosure, not refused; the
    # 400 stays only for a value outside a supporting route's declared range
    # (2026-09-06: 1,483 rejections / 289 orgs on one alias OpenAI itself refuses).
    if request.temperature is not None:
        if srn_only_block():
            ignore("temperature", "temperature->dropped(set_reasoning_effort_none)")
        elif not all(sampling_supported(profile) for profile in profiles):
            ignore("temperature", "temperature->dropped(unsupported_by_provider)")
        else:
            _require_route_numeric_parameter(
                profiles,
                param="temperature",
                value=request.temperature,
                supported=sampling_supported,
                minimum=lambda profile: profile.minimum_temperature,
                maximum=lambda profile: profile.maximum_temperature,
            )
    if request.top_p is not None:
        if srn_only_block(top_p=True):
            ignore("top_p", "top_p->dropped(set_reasoning_effort_none)")
        elif not all(sampling_supported(profile, top_p=True) for profile in profiles):
            ignore("top_p", "top_p->dropped(unsupported_by_provider)")
        else:
            _require_route_numeric_parameter(
                profiles,
                param="top_p",
                value=request.top_p,
                supported=lambda profile: sampling_supported(profile, top_p=True),
                minimum=lambda profile: profile.minimum_top_p,
                maximum=lambda profile: profile.maximum_top_p,
            )
    if request.top_k is not None:

        def top_k_supported(profile: GatewayWireProfile) -> bool:
            return profile.supports_top_k and profile.dialect != "openai_responses"

        if not all(top_k_supported(profile) for profile in profiles):
            # top_k is a sampling preference: a rung that does not carry it still
            # returns a valid answer with its own default, so the committed route
            # drops it with disclosure rather than rejecting. Selection prefers a
            # rung that honors it (generation_route_compat), so this drop is the
            # last resort when no rung on the route accepts the field.
            ignore("top_k", "top_k->dropped(unsupported_by_provider)")
        else:
            _require_route_numeric_parameter(
                profiles,
                param="top_k",
                value=request.top_k,
                supported=top_k_supported,
                minimum=lambda profile: profile.minimum_top_k,
                maximum=lambda profile: profile.maximum_top_k,
            )

    # Sampling penalties are soft preferences: a rung that does not carry them
    # still returns a valid answer, so a route with any rung that lacks support
    # drops them with disclosure rather than rejecting. Honoring is gated on the
    # openai_compatible dialect — the ONLY payload that emits penalties — so a
    # capability flag stamped on a non-emitting dialect (e.g. openai_responses)
    # can never claim "honored" and then silently omit the field. Emission stays
    # in one place; if another dialect ever emits penalties, add it here too.
    def penalty_honored(profile: GatewayWireProfile, *, presence: bool) -> bool:
        supported = (
            profile.supports_presence_penalty if presence else profile.supports_frequency_penalty
        )
        return supported and profile.dialect == "openai_compatible"

    if request.frequency_penalty is not None and not all(
        penalty_honored(profile, presence=False) for profile in profiles
    ):
        ignore("frequency_penalty", "frequency_penalty->dropped(unsupported_by_provider)")
    if request.presence_penalty is not None and not all(
        penalty_honored(profile, presence=True) for profile in profiles
    ):
        ignore("presence_penalty", "presence_penalty->dropped(unsupported_by_provider)")
    if request.thinking_default_enable and request.reasoning_effort is None:
        # A level-less "enable thinking" (from a translated thinking:{enabled} or
        # chat_template_kwargs:{enable_thinking:true}) resolves to the model's own
        # default effort here, at the serving route: a route-wide required default
        # when portable, else the LOWEST portable non-none tier (default-not-high
        # avoids surprising cost). A route that supports no reasoning effort cannot
        # enable thinking, so it surfaces rather than silently not thinking.
        portable = set(REASONING_EFFORTS)
        for profile in profiles:
            portable.intersection_update(_profile_reasoning_efforts(profile))
        portable_non_none = tuple(e for e in REASONING_EFFORTS if e in portable and e != "none")
        if not portable_non_none:
            raise ProviderParameterError(
                message=(
                    "This model route cannot enable thinking: it supports no reasoning "
                    "effort. Remove the enable-thinking field or choose a reasoning model."
                ),
                param=effort_path,
                code="unsupported_parameter",
            )
        required_defaults = {
            profile.reasoning_effort
            for profile in profiles
            if profile.reasoning_effort_required and profile.reasoning_effort in portable_non_none
        }
        provider_updates["reasoning_effort"] = (
            next(iter(required_defaults)) if len(required_defaults) == 1 else portable_non_none[0]
        )
    if request.reasoning_effort is not None:
        portable_efforts = set(REASONING_EFFORTS)
        for profile in profiles:
            portable_efforts.intersection_update(_profile_reasoning_efforts(profile))
        if request.reasoning_effort not in portable_efforts:
            raise UnsupportedReasoningEffortError(
                effort=request.reasoning_effort,
                supported_efforts=tuple(
                    effort for effort in REASONING_EFFORTS if effort in portable_efforts
                ),
                param=effort_path,
            )
    else:
        # An omitted caller value remains omitted on the shared request. Each
        # dialect payload injects only its own provider-required default, so a
        # fallback never forces that default onto a wire where it is optional.
        for profile in profiles:
            required_effort = profile_reasoning_effort(profile)
            if required_effort is None:
                continue
            profile_efforts = _profile_reasoning_efforts(profile)
            if required_effort not in profile_efforts:
                raise UnsupportedReasoningEffortError(
                    effort=required_effort,
                    supported_efforts=profile_efforts,
                    param=effort_path,
                )
    # Stop sequences on a native Responses rung: the Responses API has no stop
    # field, so the data plane emulates them (the wire entry carries the exact
    # sequences and the stream is cut at the first match). Nothing to reject.
    if request.reasoning_summary is not None and not all(
        serves_reasoning_summary(profile) for profile in profiles
    ):
        path = next(
            iter(request.reasoning_summary_parameters),
            "reasoning.summary",
        )
        raise ProviderParameterError(
            message=(
                f"The parameter {path!r} is not supported by this model route. "
                "Remove the field or choose a different model."
            ),
            param=path,
            code="unsupported_parameter",
        )
    if request.reasoning_context is not None and not all(
        profile.dialect == "openai_responses" and profile.supports_reasoning for profile in profiles
    ):
        raise ProviderParameterError(
            message=(
                "The parameter 'reasoning.context' is not supported by this model route. "
                "Remove the field or choose a different model."
            ),
            param="reasoning.context",
            code="unsupported_parameter",
        )

    if (
        request.structured_text is not None
        and not request.structured_text.strict
        and any(profile.dialect in _STRICT_STRUCTURED_OUTPUT_DIALECTS for profile in profiles)
    ):
        path = (
            "text.format.strict"
            if request.surface.value == "responses"
            else "response_format.json_schema.strict"
        )
        raise ProviderParameterError(
            message=(
                f"The parameter {path!r} cannot be false on this model route. "
                "Every non-OpenAI structured-output deployment enforces the schema. "
                "Set it to true or choose a different model."
            ),
            param=path,
            code="unsupported_parameter",
        )

    # Two wires define an image carrier inside a tool result: Anthropic
    # (tool_result image blocks) and native Responses (the SDK
    # function_call_output part list). A homogeneous route on either keeps
    # the images; a route with any rung that has no carrier degrades them to
    # positional placeholder text with disclosure instead of rejecting: the
    # block is baked into the caller's history, so a rejection wedges the
    # whole session, and a silent drop at encoding would misstate what the
    # model saw. A non-vision rung on a keeping route still rejects at
    # preflight and the route-wide coercion applies the same disclosed
    # degrade.
    if any(message.role == "tool" and message.images for message in request.messages) and not (
        all(profile.dialect == "anthropic_messages" for profile in profiles)
        or all(profile.dialect == "openai_responses" for profile in profiles)
    ):
        stripped = strip_tool_result_images(request.messages)
        if stripped is not None:
            provider_updates["messages"] = stripped
            if TOOL_RESULT_IMAGE_DROP_DISCLOSURE not in ignored:
                ignored.append(TOOL_RESULT_IMAGE_DROP_DISCLOSURE)

    # Only the Anthropic wire has a tool-result error flag. Every other wire
    # folds the flag into the result text at encoding (a fixed prefix, see
    # ``folded_tool_error_content``) so the model still learns the invocation
    # failed: rejecting wedges the session (the flag is baked into history and
    # Claude Code sets it on every failed tool call), and dropping it silently
    # would misstate what the tool did. Anthropic rungs keep the flag
    # verbatim; the route-level disclosure records the fold.
    if any(message.tool_is_error for message in request.messages) and not all(
        profile.dialect == "anthropic_messages" for profile in profiles
    ):
        if TOOL_ERROR_FOLD_DISCLOSURE not in ignored:
            ignored.append(TOOL_ERROR_FOLD_DISCLOSURE)

    # Context editing is Anthropic-native; any other rung cannot honor it,
    # so the omission is disclosed and the field dropped from dispatch,
    # never a rejection (Claude Code sends it by default).
    if request.context_management is not None and not all(
        profile.dialect == "anthropic_messages" for profile in profiles
    ):
        ignore("context_management")

    # Diagnostics correlation, fast mode, and forwarded beta tokens are
    # equally Anthropic-native; a mixed route drops them with disclosure
    # (Claude Code sends them conditionally), never a rejection.
    if request.diagnostics is not None and not all(
        profile.dialect == "anthropic_messages" for profile in profiles
    ):
        ignore("diagnostics")
    if request.speed is not None and not all(
        profile.dialect == "anthropic_messages" for profile in profiles
    ):
        ignore("speed")
    # Prompt-cache marker: honored on every Anthropic rung, so it is kept as
    # long as ANY rung is Anthropic (only the non-Anthropic rungs silently
    # cannot cache; a cache marker changes cost, not semantics). Dropping it the
    # moment one fallback rung is non-Anthropic used to strip prefix caching
    # from the winning Anthropic rung too, billing every turn's full context
    # uncached (~10x on input for a large system prompt).
    if request.provider_cache_control is not None and not any(
        profile.dialect == "anthropic_messages" for profile in profiles
    ):
        ignore("provider_cache_control", "cache_control")
    if request.inference_geo is not None and not all(
        profile.dialect == "anthropic_messages" for profile in profiles
    ):
        ignore("inference_geo")
    # A provider tier forwards where the caller pays that provider directly
    # (BYOK) OR on a host-funded rung that carries a per-tier pass-through card
    # for THIS SPECIFIC tier (profile.forwards_tier), on a tier-preserving wire.
    # Forwarding is per-tier, not lane-level: the shared request keeps the tier
    # as long as ANY candidate can bill it, and each non-billable candidate
    # strips it at its own payload build (openai_payloads gated on
    # forwards_tier), so forward and bill agree on whichever candidate is
    # selected. A route where NO candidate can bill the tier strips it up front
    # with the same 'service_tier' disclosure the capability_policy drop path
    # uses. Host-funded rungs without a card for this tier never emit it (they
    # bill catalog rates while the tier changes provider pricing: flex
    # discounted, priority premium) but stay in the route as untiered fallbacks.
    if request.service_tier is not None and not any(
        profile.forwards_tier(request.service_tier) for profile in profiles
    ):
        ignore("service_tier")
        provider_updates["service_tier"] = None
    if request.provider_beta_tokens and not all(
        profile.dialect == "anthropic_messages" for profile in profiles
    ):
        for token in request.provider_beta_tokens:
            path = f"anthropic-beta.{token}"
            if path not in ignored:
                ignored.append(path)
        provider_updates["provider_beta_tokens"] = ()

    # A caller output_config whose only content is a canonical effort rides
    # reasoning_effort everywhere; anything more is Anthropic-native and is
    # disclosed-dropped when any rung cannot honor it (Claude Code sends
    # {"effort": ...} by default).
    if request.provider_output_config is not None and not all(
        profile.dialect == "anthropic_messages" for profile in profiles
    ):
        effort_only = set(request.provider_output_config) <= {"effort"}
        if not (effort_only and request.reasoning_effort is not None):
            ignore("provider_output_config", "output_config")

    # Client telemetry and the verbosity hint are native Responses surface;
    # elsewhere they are dropped with disclosure (Codex sends both by
    # default), never a rejection.
    if request.client_metadata is not None and not all(
        profile.dialect == "openai_responses" for profile in profiles
    ):
        ignore("client_metadata")
    if request.text_verbosity is not None and not all(
        profile.dialect == "openai_responses" for profile in profiles
    ):
        ignore("text_verbosity", "text.verbosity")

    # A tool-call cache hint is honored only on the Anthropic wire; any other
    # rung silently cannot cache, so the omission is disclosed, never a
    # rejection (a cache hint changes cost, not semantics).
    if any(
        call.cache_control is not None
        for message in request.messages
        for call in message.tool_calls
    ) and not all(profile.dialect == "anthropic_messages" for profile in profiles):
        if "messages.tool_calls.cache_control" not in ignored:
            ignored.append("messages.tool_calls.cache_control")

    # Block-level cache markers (system and message text runs, tool-result
    # breakpoints) follow the #699 rule: kept while ANY rung is Anthropic
    # (the marker changes cost, not semantics, and only the non-Anthropic
    # rungs silently cannot cache), disclosed only when no rung can honor
    # them. Claude Code marks its system prompt and conversation
    # breakpoints on every request.
    if any(
        message.provider_text_blocks or message.cache_control is not None
        for message in request.messages
    ) and not any(profile.dialect == "anthropic_messages" for profile in profiles):
        if "messages.content.cache_control" not in ignored:
            ignored.append("messages.content.cache_control")

    # LiteLLM stamps ``provider_specific_fields`` on every assistant message it
    # returns, and naive agent loops echo the dump back verbatim. No wire takes
    # the object, so it is dropped on every route with disclosure, never a
    # rejection (the 400 wedged whole Terminus-2 sessions, 2026-09-05).
    if any(message.provider_specific_fields for message in request.messages):
        if "messages.provider_specific_fields" not in ignored:
            ignored.append("messages.provider_specific_fields")

    # Anthropic-native tool-definition annotations exist only on that wire;
    # every other rung drops each one with a per-field disclosure, never a
    # rejection (Claude Code sends eager_input_streaming conditionally).
    if not all(profile.dialect == "anthropic_messages" for profile in profiles):
        tool_annotation_paths = (
            ("tools.cache_control", any(tool.cache_control is not None for tool in request.tools)),
            (
                "tools.eager_input_streaming",
                any(tool.eager_input_streaming is not None for tool in request.tools),
            ),
            ("tools.defer_loading", any(tool.defer_loading is not None for tool in request.tools)),
            (
                "tools.allowed_callers",
                any(tool.allowed_callers is not None for tool in request.tools),
            ),
            (
                "tools.input_examples",
                any(tool.input_examples is not None for tool in request.tools),
            ),
        )
        for path, present in tool_annotation_paths:
            if present and path not in ignored:
                ignored.append(path)

    # Responses tool-call attribution (the nested-tool `namespace` and the
    # SDK 3.0 programmatic `caller`, on calls and on their results) exists
    # only on the native Responses wire; every other rung rebuilds the call
    # or result without it. The call still executes with its exact name and
    # arguments, so the omission is disclosed per field rather than
    # rejected — but only the caller's attribution is dropped, never the
    # call itself.
    if not all(profile.dialect == "openai_responses" for profile in profiles):
        attribution_paths = (
            (
                "messages.tool_calls.namespace->dropped(unsupported_by_provider)",
                any(
                    call.provider_namespace is not None
                    for message in request.messages
                    for call in message.tool_calls
                ),
            ),
            (
                "messages.tool_calls.caller->dropped(unsupported_by_provider)",
                any(
                    call.provider_caller is not None
                    for message in request.messages
                    for call in message.tool_calls
                ),
            ),
            (
                "messages.tool_results.attribution->dropped(unsupported_by_provider)",
                any(
                    message.provider_tool_namespace is not None
                    or message.provider_tool_caller is not None
                    for message in request.messages
                ),
            ),
        )
        for path, present in attribution_paths:
            if present and path not in ignored:
                ignored.append(path)

    # The legacy tool-result name has a slot on BOTH OpenAI wires (Chat tool
    # messages and Responses function_call_output), so it drops with
    # disclosure only when some rung is on neither.
    if any(message.provider_tool_name is not None for message in request.messages) and not all(
        profile.dialect in {"openai_responses", "openai_compatible"} for profile in profiles
    ):
        name_path = "messages.tool_results.name->dropped(unsupported_by_provider)"
        if name_path not in ignored:
            ignored.append(name_path)

    # Opaque provider-reasoning carriers replay only on the one wire that
    # issued them, so a mixed waterfall is rejected instead of dropping them.
    # Plaintext reasoning an exposure-gated rung itself returned (Tencent/
    # DeepSeek) replays only to rungs that expose their reasoning: the
    # provider's wire accepts it verbatim there, and nowhere else was it ever
    # issued. A route with no exposing rung rejects by name; a mixed waterfall
    # keeps it on the exposing rungs and discloses the drop on the others.
    exposed_reasoning_present = any(
        block.kind == "exposed_reasoning_content"
        for message in request.messages
        for block in message.provider_reasoning
    )
    if exposed_reasoning_present and not all(
        profile.reasoning_output_exposed for profile in profiles
    ):
        # Plaintext reasoning is baked into the caller's transcript (an
        # earlier turn on a reasoning-exposed rung, or a client-side AI-SDK
        # re-serialization), so a route that cannot replay it drops the block
        # with disclosure instead of rejecting: "remove the field" is not
        # actionable for a framework-managed history, and a session that ever
        # touched an exposed model would otherwise die the moment it switches
        # models. Exposing rungs — when the route has any — still forward the
        # plaintext verbatim; the others omit it at encoding.
        ignore(
            "messages.reasoning_content",
            "messages.reasoning_content->dropped(unsupported_by_provider)",
        )
    history_thinking_present = any(
        block.kind in {"thinking", "redacted_thinking"}
        for message in request.messages
        for block in message.provider_reasoning
    )
    non_anthropic_route = not all(profile.dialect == "anthropic_messages" for profile in profiles)
    if history_thinking_present and non_anthropic_route:
        raise ProviderParameterError(
            message=(
                "The request replays Anthropic extended-thinking blocks that only a "
                "native Anthropic route can carry. Remove extended-thinking content "
                "or choose a native Anthropic-only route."
            ),
            param="thinking",
            code="unsupported_parameter",
        )
    if request.provider_thinking_config is not None and non_anthropic_route:
        # A thinking CONFIG (unlike replayed thinking blocks) has a serviceable
        # cross-wire reading. The named rejection here is what lets the admit
        # loop offer the disclosed thinking->reasoning_effort translation (or
        # the disclosed drop) in ``coerce_thinking_config``: the substitution
        # is semantic, so it lives in the coercion layer, where it runs only
        # after every rung declined verbatim and never steals narrowing
        # preference from an Anthropic rung that could honor the config.
        raise ProviderParameterError(
            message=(
                "The parameter 'thinking' is not supported by this model route. "
                "Remove the field or choose a native Anthropic-only route."
            ),
            param="thinking",
            code="unsupported_parameter",
        )
    if request.provider_thinking_config is not None and not non_anthropic_route:
        # The adaptive-thinking generation rejects caller enabled/disabled
        # configs outright, so verbatim forwarding is family-gated (a route
        # is one exact-model pool, so the answer is uniform across rungs).
        config_type = str(request.provider_thinking_config.get("type"))
        adaptive_only = all(
            anthropic_adaptive_only_thinking(profile.model_id) for profile in profiles
        )
        # A budgeted-enabled-only model (haiku-4-5) rejects an adaptive config
        # by NAME; the named rejection here is what lets the admit loop offer
        # the disclosed adaptive->enabled(budget) coercion instead of the
        # provider's own opaque 400 (which never fails over).
        budgeted_enabled_only = all(
            profile.dialect == "anthropic_messages"
            and anthropic_budgeted_enabled_only(profile.model_id)
            for profile in profiles
        )
        if budgeted_enabled_only and config_type == "adaptive":
            raise ProviderParameterError(
                message=(
                    "The parameter 'thinking.type' cannot be 'adaptive' on this model: "
                    "it reasons via an explicit token budget. Send thinking "
                    "{type: 'enabled', budget_tokens: N} or remove the field."
                ),
                param="thinking.type",
                code="unsupported_parameter",
            )
        if adaptive_only and config_type == "enabled":
            # Translate to the model's one supported mode, emitted explicitly
            # so the promise holds even on routes with no pinned effort. The
            # token budget has no adaptive equivalent, so it is disclosed as
            # ignored rather than silently mapped onto an effort level.
            provider_updates["provider_thinking_config"] = {"type": "adaptive"}
            if "thinking.budget_tokens" not in ignored:
                ignored.append("thinking.budget_tokens")
            _logger.warning(
                "translated a caller 'enabled' thinking config to adaptive for an "
                "adaptive-only Anthropic route; thinking.budget_tokens was disclosed "
                "as ignored"
            )
        elif adaptive_only and config_type == "disabled":
            raise ProviderParameterError(
                message=(
                    "The parameter 'thinking.type' cannot be 'disabled' on this model: "
                    "it always reasons adaptively. Remove the thinking field or choose "
                    "a model that supports disabling thinking."
                ),
                param="thinking.type",
                code="unsupported_parameter",
            )
    if anthropic_server_tools_present(request) and not all(
        profile.dialect == "anthropic_messages" for profile in profiles
    ):
        server_tool_names = anthropic_server_tool_names(request)
        # Server tools execute inside Anthropic's API; silently dropping a
        # search capability the caller asked for would be a behavior lie, so
        # a route that cannot serve them rejects and NAMES the tool (Claude
        # Code's WebSearch is the common case) so the caller knows which
        # feature needs a Claude model.
        raise ProviderParameterError(
            message=anthropic_server_tools_message(server_tool_names),
            param="tools",
            code="unsupported_parameter",
        )
    if any(message.provider_native_item is not None for message in request.messages) and not all(
        profile.dialect == "openai_responses" for profile in profiles
    ):
        raise ProviderParameterError(
            message=(
                "The request carries native Responses input items (tool namespaces, "
                "custom tool calls, or hosted tool items such as web_search_call and "
                "mcp_call echoes) that only a native OpenAI Responses route can serve. "
                "Choose a different model alias."
            ),
            param="input",
            code="unsupported_parameter",
        )
    outbound_maximum_output_tokens = provider_updates.get(
        "maximum_output_tokens", request.maximum_output_tokens
    )
    if (
        isinstance(outbound_maximum_output_tokens, int)
        and outbound_maximum_output_tokens < OPENAI_MINIMUM_OUTPUT_TOKENS
        and all(profile.dialect == "openai_responses" for profile in profiles)
    ):
        # The provider's own 400 for this is post-dispatch and opaque on some
        # relays; the documented Responses minimum is a request fact, so it is
        # rejected at admission with the bound named. The value judged is the
        # one that would dispatch: a Messages-surface probe already floored
        # with disclosure above never reaches this rejection, while a route
        # whose declared ceiling sits below the floor cannot ride it and gets
        # the named minimum instead of the provider's opaque 400.
        parameter = request.maximum_output_tokens_parameter or "max_output_tokens"
        raise ProviderParameterError(
            message=(
                f"The parameter {parameter!r} must be at least 16 on this model route "
                "(the OpenAI Responses minimum). Raise the value and resend the request."
            ),
            param=parameter,
            code="invalid_parameter",
        )
    if request.provider_native_tools and not all(
        profile.dialect == "openai_responses" for profile in profiles
    ):
        raise ProviderParameterError(
            message=(
                "The request carries native Responses tool declarations (custom, "
                "namespace, web_search, or tool_search entries) that only a native "
                "OpenAI Responses route can serve. Remove those tools or choose a "
                "different model alias."
            ),
            param="tools",
            code="unsupported_parameter",
        )

    if any(profile.dialect == "anthropic_messages" for profile in profiles) and any(
        message.role == "user"
        and not (message.content or "").strip()
        and not message.content_parts
        and message.provider_anthropic_block is None
        and message.provider_native_item is None
        for message in request.messages
    ):
        # The Anthropic wire rejects empty text content blocks post-dispatch
        # ("text content blocks must be non-empty"; 2026-09-05, six orgs on
        # claude-fable routes) and a user turn whose text is all whitespace
        # ("text content blocks must contain non-whitespace text"; a user
        # message must have non-empty content, so unlike an assistant turn
        # it cannot dispatch as an empty array). Empty blocks inside a richer
        # turn drop loss-free at conversion, but a user turn with no readable
        # text has nothing to send and dropping the whole message would
        # change conversation structure, so it is refused by name pre-dispatch.
        raise ProviderParameterError(
            message=(
                "A user message with empty or whitespace-only content cannot be "
                "served by this model route: the provider rejects empty text "
                "content blocks. Add content to the message or remove it."
            ),
            param="messages",
            code="invalid_parameter",
        )

    require_assistant_prefill_supported(profiles, request)

    # A system turn after conversation began has positional semantics that
    # instruction-hoisting wires cannot preserve; those rungs narrow out.
    if mid_conversation_system_present(request) and any(
        profile.dialect in {"gemini_generate_content", "bedrock_converse_stream"}
        for profile in profiles
    ):
        raise ProviderParameterError(
            message=(
                "A system message after conversation start is not supported by this "
                "model route. Move the instruction to the leading system prompt or "
                "choose a different model."
            ),
            param="messages",
            code="unsupported_parameter",
        )

    encrypted_reasoning_present = any(
        block.kind == "encrypted_reasoning"
        for message in request.messages
        for block in message.provider_reasoning
    )
    if encrypted_reasoning_present and not all(
        profile.dialect == "openai_responses" for profile in profiles
    ):
        raise ProviderParameterError(
            message=(
                "The parameter 'reasoning.encrypted_content' is not supported by this "
                "model route. Remove encrypted reasoning or choose a native OpenAI "
                "Responses-only route."
            ),
            param="include",
            code="unsupported_parameter",
        )
    encrypted_reasoning_channel = request.include_encrypted_reasoning and (
        all(profile.dialect == "openai_responses" for profile in profiles)
        or all(profile.fireworks_reasoning_route_sha256 is not None for profile in profiles)
    )
    if request.include_encrypted_reasoning and not encrypted_reasoning_channel:
        raise ProviderParameterError(
            message=(
                "The parameter 'reasoning.encrypted_content' requires one homogeneous "
                "native Responses or Fireworks reasoning-carrier route."
            ),
            param="include",
            code="unsupported_parameter",
        )

    # Tool-selection controls have no semantics without tool definitions and
    # several provider APIs reject the otherwise harmless combination.
    # Verbatim server tools are tool definitions too: a request carrying only
    # web_search keeps its tool_choice on the wire.
    server_tool_names = tuple(
        str(entry["name"]) for entry in request.provider_server_tools if "name" in entry
    )
    if not request.tools and not request.provider_server_tools:
        if request.tool_choice == "required" or isinstance(
            request.tool_choice, GatewayNamedToolChoice
        ):
            raise ProviderParameterError(
                message=(
                    "The parameter 'tool_choice' requires at least one matching tool "
                    "definition. Add the tool or remove the selector."
                ),
                param="tool_choice",
                code="invalid_parameter",
            )
        if request.tool_choice is not None:
            ignore("tool_choice")
        if request.parallel_tool_calls is not None:
            ignore("parallel_tool_calls")
    elif (
        isinstance(request.tool_choice, GatewayNamedToolChoice)
        and not any(tool.name == request.tool_choice.name for tool in request.tools)
        and request.tool_choice.name not in server_tool_names
    ):
        raise ProviderParameterError(
            message=(
                f"The tool named by 'tool_choice' ({request.tool_choice.name!r}) is not "
                "present in this request's tool definitions."
            ),
            param="tool_choice",
            code="invalid_parameter",
        )
    elif request.tool_choice == "none" and request.parallel_tool_calls is not None:
        ignore("parallel_tool_calls")
    elif request.parallel_tool_calls is not None and all(
        profile.dialect in _NO_PARALLEL_TOOL_CONTROL_DIALECTS for profile in profiles
    ):
        # No rung carries a parallel-tool control: `true` is the provider's own
        # default (dropped); `false` is serialized by the data plane. A mixed
        # route keeps the field; toggle-less rungs are shaped per rung at dispatch.
        if request.parallel_tool_calls:
            ignore("parallel_tool_calls", "parallel_tool_calls->dropped(provider_default)")
        else:
            ignore(
                "parallel_tool_calls",
                "parallel_tool_calls->emulated(serialized_by_gateway)",
            )
            provider_updates["serialize_tool_calls"] = True

    # A true logprob request changes the requested result. Until the normalized
    # response can return those arrays, reject it rather than pretending it ran.
    if request.logprobs is True:
        path = (
            "top_logprobs"
            if request.surface.value == "responses" and request.top_logprobs is not None
            else "logprobs"
        )
        raise ProviderParameterError(
            message=(
                f"The parameter {path!r} is not supported by this gateway response contract. "
                "Remove the field and resend the request."
            ),
            param=path,
            code="unsupported_parameter",
        )
    if request.logprobs is False:
        ignore("logprobs")
    if request.top_logprobs is not None:
        raise ProviderParameterError(
            message=(
                "The parameter 'top_logprobs' is not supported by this gateway response "
                "contract. Remove the field and resend the request."
            ),
            param="top_logprobs",
            code="unsupported_parameter",
        )

    ignored_parameters = tuple(ignored)
    public_request = request.model_copy(update={"ignored_parameters": ignored_parameters})
    provider_request = public_request.model_copy(update=provider_updates)
    return public_request, provider_request
