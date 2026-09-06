"""Rung-preference units; admission coercions are exercised e2e in native_bridge_test.py."""

import base64
from typing import Literal, cast

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models.catalog import (
    GatewayDeploymentCapabilities,
    GatewayDeploymentMetadata,
    GatewayRungDispatchPolicy,
    GatewayServiceTierPrices,
    GatewayTokenPrices,
)
from exp.common.models.content import (
    AudioContentPart,
    ImageContentPart,
    MediaHandle,
    TextContentPart,
    VideoContentPart,
)
from exp.common.models.gateway_catalog import ExactModelDeployment, FailoverMode
from exp.common.models.model import ModelCapabilities
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    ExecutionSnapshot,
    GatewayApiSurface,
    GatewayMessage,
    GatewayNamedToolChoice,
    GatewayRequest,
    GatewayToolDefinition,
)
from exp.runtime.gateway.native_accounting import NativeAttemptAccounting
from exp.runtime.gateway.native_admission import (
    _affinity_ordered_rungs,
    _prefer_cache_capable_rungs,
    admitted_route_requests,
    protocol_compatible_indexes,
    route_rejection,
    shape_parallel_tool_calls,
)
from exp.runtime.gateway.native_dispatch import NativeWireClient
from exp.runtime.gateway.prompt_size import MAXIMUM_BYTES_PER_TOKEN
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderCapabilityError, ProviderParameterError


def _deployment(
    deployment_id: str,
    *,
    provider: str = "openai-compatible",
    gateway: GatewayDeploymentMetadata | None = None,
) -> ExactModelDeployment:
    """Build one exact deployment for rung-preference tests."""
    return ExactModelDeployment(
        deployment_id=deployment_id,
        source_alias=deployment_id,
        exact_model_id="exact-one",
        connection=f"connection-{deployment_id}",
        provider=provider,
        provider_model="provider-model",
        connection_sha256="b" * 64,
        capabilities_sha256="c" * 64,
        gateway=gateway or GatewayDeploymentMetadata(),
    )


def _mixed_route(
    failover_mode: str,
    deployments: tuple[ExactModelDeployment, ...] = (),
    surface: GatewayApiSurface = GatewayApiSurface.MESSAGES,
) -> GatewayRoute:
    """Build one two-rung route whose FIRST rung drops cache markers."""
    deployments = deployments or (_deployment("shim"), _deployment("native"))
    authorization = AuthorizationSnapshot(
        request_id="request-one",
        organization_id="organization-one",
        identity_id="identity-one",
        virtual_key_id="key-one",
        alias="public-model",
        alias_revision_id="revision-one",
        target=DirectTarget(pool_id="pool-one"),
        surface=surface,
        catalog_sha256="a" * 64,
        canonical_request_sha256="d" * 64,
        deadline_monotonic=1.0,
    )
    return GatewayRoute(
        snapshot=ExecutionSnapshot(
            authorization=authorization,
            exact_model_id="exact-one",
            pool_id="pool-one",
            deployment_ids=tuple(item.deployment_id for item in deployments),
            failover_mode=cast(FailoverMode, failover_mode),
        ),
        deployment=deployments[0],
        fallback_deployments=deployments[1:],
        route_reason="direct",
    )


def _wires() -> tuple[tuple[GatewayWireProfile, NativeWireClient], ...]:
    """Pair one marker-dropping and one marker-honoring rung, shim first."""
    shim = GatewayWireProfile(dialect="openai_compatible", url="https://shim.test")
    native = GatewayWireProfile(dialect="anthropic_messages", url="https://anthropic.test")
    client = cast(NativeWireClient, object())
    return ((shim, client), (native, client))


def _marked_request() -> GatewayRequest:
    """Build one Messages request carrying a system cache marker."""
    return GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(
            GatewayMessage(
                role="system",
                content="cached prompt",
                provider_text_blocks=(
                    {
                        "type": "text",
                        "text": "cached prompt",
                        "cache_control": {"type": "ephemeral"},
                    },
                ),
            ),
            GatewayMessage(role="user", content="hi"),
        ),
    )


def test_remote_url_refusal_outranks_a_text_only_rung_refusing_every_image() -> None:
    """A text-only rung ahead of an inline-only rung reports the URL, not the image."""
    text_only = ProviderCapabilityError(capability="image_input")
    inline_only = ProviderCapabilityError(capability="image_url_input")
    assert route_rejection((text_only, inline_only, inline_only)) is inline_only
    assert route_rejection((inline_only, text_only)) is inline_only


def test_route_rejection_keeps_the_first_rung_without_a_url_refusal() -> None:
    """Mixed rejections that never name the URL surface the first rung's reason."""
    tools = ProviderCapabilityError(capability="function_tools")
    parameter = ProviderParameterError(message="unsupported", param="top_k", code="unsupported")
    text_only = ProviderCapabilityError(capability="image_input")
    assert route_rejection((tools, text_only)) is tools
    assert route_rejection((parameter, text_only)) is parameter
    assert route_rejection((text_only,)) is text_only


def test_cache_marked_requests_dispatch_marker_honoring_rungs_first() -> None:
    """maximize_cache pools put the marker-carrying wire ahead of the shim.

    The haiku-4.5 incident shape: a certified waterfall paired a native
    Anthropic rung with an aggregator shim, and every marked session that
    dispatched on the shim billed its full context uncached (~10x). The
    pool's whole policy is prefix-cache preservation, so the marker-honoring
    rung dispatches first; certified order still decides everything else.
    """
    route, wires = _prefer_cache_capable_rungs(
        _mixed_route("maximize_cache"), _wires(), _marked_request()
    )
    assert route.deployment.deployment_id == "native"
    assert tuple(item.deployment_id for item in route.fallback_deployments) == ("shim",)
    assert wires[0][0].dialect == "anthropic_messages"

    # A markerless request keeps the certified order.
    plain = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="hi"),),
    )
    route, wires = _prefer_cache_capable_rungs(_mixed_route("maximize_cache"), _wires(), plain)
    assert route.deployment.deployment_id == "shim"

    # maximize_availability pools keep their certified order untouched.
    route, wires = _prefer_cache_capable_rungs(
        _mixed_route("maximize_availability"), _wires(), _marked_request()
    )
    assert route.deployment.deployment_id == "shim"

    # A route with no marker-honoring rung (or only such rungs) is unchanged;
    # the dropped markers are disclosed elsewhere.
    client = cast(NativeWireClient, object())
    shim = (GatewayWireProfile(dialect="openai_compatible", url="https://shim.test"), client)
    route, wires = _prefer_cache_capable_rungs(
        _mixed_route("maximize_cache"),
        (shim, shim),
        _marked_request(),
    )
    assert route.deployment.deployment_id == "shim"


def test_video_requests_skip_rungs_whose_wire_cannot_carry_them() -> None:
    """A waterfall lands a video on the Gemini rung, past Anthropic and inline-only Bedrock."""
    video_route = GatewayDeploymentMetadata(
        capabilities=GatewayDeploymentCapabilities(
            supports_streaming=True,
            supports_video_input=True,
            supports_video_url_input=True,
        )
    )
    inline_only = GatewayDeploymentMetadata(
        capabilities=GatewayDeploymentCapabilities(
            supports_streaming=True, supports_video_input=True
        )
    )
    deployments = (
        _deployment("claude", provider="anthropic"),
        _deployment("nova", provider="bedrock", gateway=inline_only),
        _deployment("gemini", provider="gemini", gateway=video_route),
    )
    route = _mixed_route("maximize_availability", deployments, GatewayApiSurface.CHAT_COMPLETIONS)
    client = cast(NativeWireClient, object())
    wires = (
        (GatewayWireProfile(dialect="anthropic_messages", url="https://anthropic.test"), client),
        (GatewayWireProfile(dialect="bedrock_converse_stream", url="https://bedrock.test"), client),
        (GatewayWireProfile(dialect="gemini_generate_content", url="https://gemini.test"), client),
    )
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(
            GatewayMessage(
                role="user",
                content="describe",
                content_parts=(
                    VideoContentPart(url="https://example.com/clip.mp4"),
                    TextContentPart(text="describe"),
                ),
            ),
        ),
        stream=True,
        include_usage=True,
    )
    indexes, errors = protocol_compatible_indexes(route, wires, request, public_stream=False)
    assert indexes == (2,)
    capabilities = [
        error.capability for error in errors if isinstance(error, ProviderCapabilityError)
    ]
    assert capabilities == ["video_input", "video_url_input"]
    assert len(errors) == 2


def test_oversized_inline_media_skips_the_bedrock_rung() -> None:
    """Inline videos that jointly exceed Converse's 25 MB payload cap fall through to Gemini."""
    video_route = GatewayDeploymentMetadata(
        capabilities=GatewayDeploymentCapabilities(
            supports_streaming=True, supports_video_input=True
        )
    )
    deployments = (
        _deployment("nova", provider="bedrock", gateway=video_route),
        _deployment("gemini", provider="gemini", gateway=video_route),
    )
    route = _mixed_route("maximize_availability", deployments, GatewayApiSurface.CHAT_COMPLETIONS)
    client = cast(NativeWireClient, object())
    wires = (
        (GatewayWireProfile(dialect="bedrock_converse_stream", url="https://bedrock.test"), client),
        (GatewayWireProfile(dialect="gemini_generate_content", url="https://gemini.test"), client),
    )
    chunk = base64.b64encode(b"\0" * (10 * 1024 * 1024)).decode()
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(
            GatewayMessage(
                role="user",
                content="describe",
                content_parts=(
                    VideoContentPart(media_type="video/mp4", data=chunk),
                    VideoContentPart(media_type="video/mp4", data=chunk),
                    TextContentPart(text="describe"),
                ),
            ),
        ),
        stream=True,
        include_usage=True,
    )
    indexes, errors = protocol_compatible_indexes(route, wires, request, public_stream=False)
    assert indexes == (1,)
    assert len(errors) == 1
    assert isinstance(errors[0], ProviderParameterError)
    assert errors[0].param == "messages"


def test_media_handle_requests_land_only_on_the_uploading_providers_rung() -> None:
    """A waterfall skips undeclared and foreign-provider rungs for a handle.

    An OpenAI Files handle passes an Anthropic rung that declares handles
    (wrong provider), an OpenAI rung that never declared them, and lands on
    the declared OpenAI rung. When no rung can serve, the provider mismatch
    is the rejection the caller sees.
    """
    handles = GatewayDeploymentMetadata(
        capabilities=GatewayDeploymentCapabilities(
            supports_streaming=True,
            supports_image_input=True,
            supports_media_handle_input=True,
        )
    )
    inline_only = GatewayDeploymentMetadata(
        capabilities=GatewayDeploymentCapabilities(
            supports_streaming=True, supports_image_input=True
        )
    )
    deployments = (
        _deployment("claude", provider="anthropic", gateway=handles),
        _deployment("gpt-inline", provider="openai", gateway=inline_only),
        _deployment("gpt-files", provider="openai", gateway=handles),
    )
    route = _mixed_route("maximize_availability", deployments, GatewayApiSurface.RESPONSES)
    client = cast(NativeWireClient, object())
    wires = (
        (GatewayWireProfile(dialect="anthropic_messages", url="https://anthropic.test"), client),
        (GatewayWireProfile(dialect="openai_responses", url="https://openai.test"), client),
        (GatewayWireProfile(dialect="openai_responses", url="https://openai.test"), client),
    )
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(
            GatewayMessage(
                role="user",
                content="describe",
                content_parts=(
                    ImageContentPart(handle=MediaHandle(provider="openai", reference="file-abc")),
                    TextContentPart(text="describe"),
                ),
            ),
        ),
        stream=True,
        include_usage=True,
    )
    indexes, errors = protocol_compatible_indexes(route, wires, request, public_stream=False)
    assert indexes == (2,)
    capabilities = [
        error.capability for error in errors if isinstance(error, ProviderCapabilityError)
    ]
    assert capabilities == ["media_handle_provider", "media_handle_input"]

    without_openai = _mixed_route(
        "maximize_availability", deployments[:2], GatewayApiSurface.RESPONSES
    )
    indexes, errors = protocol_compatible_indexes(
        without_openai, wires[:2], request, public_stream=False
    )
    assert indexes == ()
    rejection = route_rejection(errors)
    assert isinstance(rejection, ProviderCapabilityError)
    assert rejection.capability == "media_handle_provider"
    assert rejection.detail is not None and "uploaded to openai" in rejection.detail


def test_audio_requests_skip_rungs_whose_wire_cannot_carry_them() -> None:
    """A clip lands on the declared Chat rung, past Anthropic, Bedrock, and undeclared Gemini."""
    audio_route = GatewayDeploymentMetadata(
        capabilities=GatewayDeploymentCapabilities(
            supports_streaming=True, supports_audio_input=True
        )
    )
    deployments = (
        _deployment("claude", provider="anthropic"),
        _deployment("nova", provider="bedrock", gateway=audio_route),
        _deployment("gemini", provider="gemini"),
        _deployment("router", provider="openrouter", gateway=audio_route),
    )
    route = _mixed_route("maximize_availability", deployments, GatewayApiSurface.CHAT_COMPLETIONS)
    client = cast(NativeWireClient, object())
    wires = (
        (GatewayWireProfile(dialect="anthropic_messages", url="https://anthropic.test"), client),
        (GatewayWireProfile(dialect="bedrock_converse_stream", url="https://bedrock.test"), client),
        (GatewayWireProfile(dialect="gemini_generate_content", url="https://gemini.test"), client),
        (GatewayWireProfile(dialect="openai_compatible", url="https://openrouter.test"), client),
    )
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(
            GatewayMessage(
                role="user",
                content="what is said",
                content_parts=(
                    AudioContentPart(media_type="audio/wav", data="UklGRgAAAABXQVZF"),
                    TextContentPart(text="what is said"),
                ),
            ),
        ),
        stream=True,
        include_usage=True,
    )
    indexes, errors = protocol_compatible_indexes(route, wires, request, public_stream=False)
    assert indexes == (3,)
    capabilities = [
        error.capability for error in errors if isinstance(error, ProviderCapabilityError)
    ]
    assert capabilities == ["audio_input", "audio_input", "audio_input"]
    assert len(errors) == 3


def test_mixed_waterfall_drops_the_tier_to_serve_the_preserving_rung() -> None:
    """Rungs declining for different reasons still serve a tiered request.

    The OpenAI-compatible rung declines parallel tool calls while the
    Anthropic rung declines the service tier, so no unanimous route-wide
    capability exists — yet dropping the disclosed tier lets the Anthropic
    rung serve instead of surfacing a rejection nobody can act on.
    """
    from exp.runtime.gateway.native_accounting import NativeAttemptAccounting

    tools_capable = GatewayDeploymentMetadata(
        capabilities=GatewayDeploymentCapabilities(
            supports_streaming=True,
            supports_parallel_tool_calls=True,
            supports_streaming_tool_arguments=True,
        )
    )
    no_parallel = GatewayDeploymentMetadata(
        capabilities=GatewayDeploymentCapabilities(
            supports_streaming=True,
            supports_streaming_tool_arguments=True,
        )
    )
    deployments = (
        _deployment("shim", gateway=no_parallel),
        _deployment("native", provider="anthropic", gateway=tools_capable),
    )
    route = _mixed_route("maximize_availability", deployments, GatewayApiSurface.CHAT_COMPLETIONS)
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="go"),),
        tools=(GatewayToolDefinition(name="lookup", parameters={"type": "object"}),),
        parallel_tool_calls=True,
        service_tier="flex",
    )

    class _CoercionCounter:
        """Count coercion recordings without a live ledger."""

        recorded = 0

        def record_admission_coercions(self, count: int) -> None:
            self.recorded += count

    accounting = _CoercionCounter()
    # The shim rung is BYOK (tier-eligible), so the tier survives route
    # shaping and the mixed-rejection coercion path is what drops it; a
    # house-funded shim would instead strip the tier during route shaping
    # with the same disclosure and no coercion retry.
    client = cast(NativeWireClient, object())
    wires = (
        (
            GatewayWireProfile(
                dialect="openai_compatible",
                url="https://shim.test",
                billing_customer_managed=True,
            ),
            client,
        ),
        (GatewayWireProfile(dialect="anthropic_messages", url="https://anthropic.test"), client),
    )
    narrowed, _wires_out, public, provider = admitted_route_requests(
        route,
        wires,
        request,
        accounting=cast(NativeAttemptAccounting, accounting),
        authorization=route.snapshot.authorization,
    )

    assert tuple(item.deployment_id for item in narrowed.deployments) == ("native",)
    assert public.ignored_parameters == ("service_tier",)
    assert provider.service_tier is None
    assert accounting.recorded == 1
    # The rebuild after the capability coercion must not lose the affinity
    # key: it is attached to the request admission finally settled on.
    assert provider.provider_prompt_cache_key is not None
    assert public.provider_prompt_cache_key is None


def test_admission_attaches_a_tenant_namespaced_cache_affinity_key() -> None:
    """The provider request carries the derived key; the public request does not.

    The key is derived from the frozen authority plus the caller's
    ``prompt_cache_key`` (or the conversation stem), so it is stable across
    the turns of one session, never the caller's raw value, and never
    part of the public request or its serialized identity.
    """
    from exp.runtime.gateway.native_accounting import NativeAttemptAccounting

    streaming = GatewayDeploymentMetadata(
        capabilities=GatewayDeploymentCapabilities(
            supports_streaming=True,
            supports_streaming_tool_arguments=True,
        )
    )
    deployments = (_deployment("shim", gateway=streaming),)
    route = _mixed_route("maximize_availability", deployments, GatewayApiSurface.CHAT_COMPLETIONS)
    wires = (
        (
            GatewayWireProfile(dialect="openai_compatible", url="https://shim.test"),
            cast(NativeWireClient, object()),
        ),
    )

    class _CoercionCounter:
        """Count coercion recordings without a live ledger."""

        recorded = 0

        def record_admission_coercions(self, count: int) -> None:
            """Accumulate one admission's coercion count."""
            self.recorded += count

    def admit(messages: tuple[GatewayMessage, ...], key: str | None) -> GatewayRequest:
        """Admit one request and return the provider-side request it dispatches."""
        request = GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=messages,
            prompt_cache_key=key,
            stream=True,
            include_usage=True,
        )
        _narrowed, _wires_out, public, provider = admitted_route_requests(
            route,
            wires,
            request,
            accounting=cast(NativeAttemptAccounting, _CoercionCounter()),
            authorization=route.snapshot.authorization,
        )
        assert public.provider_prompt_cache_key is None
        assert public.prompt_cache_key == key
        return provider

    stem = (
        GatewayMessage(role="system", content="You are Terminus."),
        GatewayMessage(role="user", content="Task: list files."),
    )
    turn_1 = admit(stem, None)
    turn_2 = admit(
        (
            *stem,
            GatewayMessage(role="assistant", content='{"command": "ls"}'),
            GatewayMessage(role="user", content="Output: a.txt"),
        ),
        None,
    )
    assert turn_1.provider_prompt_cache_key is not None
    assert turn_1.provider_prompt_cache_key == turn_2.provider_prompt_cache_key
    keyed = admit(stem, "session-7")
    assert keyed.provider_prompt_cache_key is not None
    assert "session-7" not in keyed.provider_prompt_cache_key
    assert keyed.provider_prompt_cache_key != turn_1.provider_prompt_cache_key
    # The affinity key is dispatch state only: it never enters serialization.
    assert "provider_prompt_cache_key" not in keyed.model_dump(mode="json")


def test_disabled_thinking_on_an_adaptive_only_mixed_route_is_dropped_with_disclosure() -> None:
    """A dual-lane opus-5 route serves a disabled-thinking request instead of refusing.

    The aggregator rung cannot carry Anthropic thinking at all and the
    adaptive-only Anthropic rung rejects an explicit ``disabled``, so no rung
    preserves the request verbatim. The disclosed drop lets the route serve,
    the Anthropic rung emitting its sole supported mode.
    """
    from exp.runtime.gateway.native_accounting import NativeAttemptAccounting

    streaming = GatewayDeploymentMetadata(
        capabilities=GatewayDeploymentCapabilities(supports_streaming=True)
    )
    deployments = (
        _deployment("native", provider="anthropic", gateway=streaming),
        _deployment("shim", gateway=streaming),
    )
    route = _mixed_route("maximize_availability", deployments)
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="go"),),
        provider_thinking_config={"type": "disabled"},
        stream=True,
        include_usage=True,
    )

    class _CoercionCounter:
        """Count coercion recordings without a live ledger."""

        recorded = 0

        def record_admission_coercions(self, count: int) -> None:
            self.recorded += count

    accounting = _CoercionCounter()
    client = cast(NativeWireClient, object())
    wires = (
        (
            GatewayWireProfile(
                dialect="anthropic_messages",
                url="https://anthropic.test",
                model_id="claude-opus-5",
                supports_reasoning=True,
                reasoning_wire_format="anthropic_adaptive",
            ),
            client,
        ),
        (
            GatewayWireProfile(
                dialect="openai_compatible",
                url="https://shim.test",
                model_id="anthropic/claude-opus-5",
            ),
            client,
        ),
    )
    narrowed, _wires_out, public, provider = admitted_route_requests(
        route,
        wires,
        request,
        accounting=cast(NativeAttemptAccounting, accounting),
        authorization=route.snapshot.authorization,
    )

    # With the config gone nothing Anthropic-only remains on the request, so
    # the whole certified waterfall stays available, native rung first.
    assert tuple(item.deployment_id for item in narrowed.deployments) == ("native", "shim")
    assert public.ignored_parameters == ("thinking.type->adaptive",)
    assert provider.provider_thinking_config is None
    assert accounting.recorded == 1


_TOOL_IMAGE_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
"""One valid single-pixel PNG, base64 encoded (the owner-reported repro image)."""


def _tool_screenshot_route_request(*, stream: bool) -> GatewayRequest:
    """Decode the owner-reported wedged-session repro through the real surface.

    The exact wire body: a user text turn, an assistant ``tool_use``, and a
    user ``tool_result`` whose content is one base64 PNG image sub-block.
    """
    from exp.runtime.anthropic_protocol.requests import decode_messages

    body: JsonObject = {
        "model": "coding",
        "max_tokens": 128,
        "stream": stream,
        "messages": [
            {"role": "user", "content": "read the screenshot"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call-1", "name": "computer", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call-1",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": _TOOL_IMAGE_PNG,
                                },
                            }
                        ],
                    }
                ],
            },
        ],
    }
    return decode_messages(body).request.model_copy(update={"include_usage": True})


class _AdmissionCoercionCounter:
    """Count coercion recordings without a live ledger."""

    def __init__(self) -> None:
        self.recorded = 0

    def record_admission_coercions(self, count: int) -> None:
        self.recorded += count


@pytest.mark.parametrize("stream", [True, False])
def test_tool_result_image_passes_through_on_a_vision_anthropic_route(stream: bool) -> None:
    """The repro serves verbatim on an image-capable Anthropic route."""
    from exp.runtime.gateway.native_accounting import NativeAttemptAccounting

    vision = GatewayDeploymentMetadata(
        capabilities=GatewayDeploymentCapabilities(
            supports_streaming=True,
            supports_streaming_tool_arguments=True,
            supports_image_input=True,
        )
    )
    deployments = (_deployment("claude", provider="anthropic", gateway=vision),)
    route = _mixed_route("maximize_availability", deployments, GatewayApiSurface.MESSAGES)
    client = cast(NativeWireClient, object())
    wires = (
        (GatewayWireProfile(dialect="anthropic_messages", url="https://anthropic.test"), client),
    )
    accounting = _AdmissionCoercionCounter()

    _narrowed, _wires_out, public, provider = admitted_route_requests(
        route,
        wires,
        _tool_screenshot_route_request(stream=stream),
        accounting=cast(NativeAttemptAccounting, accounting),
        authorization=route.snapshot.authorization,
    )

    tool_message = provider.messages[-1]
    assert tool_message.role == "tool"
    assert [part.kind for part in tool_message.content_parts] == ["image"]
    assert tool_message.images[0].data == _TOOL_IMAGE_PNG
    assert public.ignored_parameters == ()
    assert accounting.recorded == 0


@pytest.mark.parametrize("stream", [True, False])
def test_tool_result_image_degrades_with_disclosure_on_a_non_vision_route(stream: bool) -> None:
    """The repro serves with a disclosed placeholder instead of a 400.

    The image is baked into the caller's history, so a rejection wedges every
    later turn of the session; a fable-5.1-style non-vision route answers the
    degraded request while ``ignored_parameters`` tells the caller what was
    dropped.
    """
    from exp.runtime.gateway.native_accounting import NativeAttemptAccounting
    from exp.runtime.models.providers.streaming_requests import (
        TOOL_RESULT_IMAGE_DROP_DISCLOSURE,
        TOOL_RESULT_IMAGE_PLACEHOLDER,
    )

    text_only = GatewayDeploymentMetadata(
        capabilities=GatewayDeploymentCapabilities(
            supports_streaming=True,
            supports_streaming_tool_arguments=True,
        )
    )
    deployments = (_deployment("claude", provider="anthropic", gateway=text_only),)
    route = _mixed_route("maximize_availability", deployments, GatewayApiSurface.MESSAGES)
    client = cast(NativeWireClient, object())
    wires = (
        (GatewayWireProfile(dialect="anthropic_messages", url="https://anthropic.test"), client),
    )
    accounting = _AdmissionCoercionCounter()

    _narrowed, _wires_out, public, provider = admitted_route_requests(
        route,
        wires,
        _tool_screenshot_route_request(stream=stream),
        accounting=cast(NativeAttemptAccounting, accounting),
        authorization=route.snapshot.authorization,
    )

    tool_message = provider.messages[-1]
    assert tool_message.content_parts == ()
    assert tool_message.content == TOOL_RESULT_IMAGE_PLACEHOLDER
    assert TOOL_RESULT_IMAGE_DROP_DISCLOSURE in public.ignored_parameters
    assert accounting.recorded == 1


def test_a_thinking_config_translates_through_admission_on_an_openai_route() -> None:
    """The full admit loop serves a thinking config on an all-OpenAI route.

    Route shaping rejects the config by name, the coercion translates it to
    the route's effort ladder, and the re-narrowed route serves with the
    translation disclosed and counted.
    """
    from exp.runtime.gateway.native_accounting import NativeAttemptAccounting

    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="go"),),
        provider_thinking_config={"type": "enabled", "budget_tokens": 8192},
        stream=True,
        include_usage=True,
    )
    accounting = _AdmissionCoercionCounter()
    client = cast(NativeWireClient, object())
    reasoning = GatewayDeploymentMetadata(
        capabilities=GatewayDeploymentCapabilities(supports_streaming=True)
    )
    route = _mixed_route(
        "maximize_availability",
        (_deployment("gpt", provider="openai", gateway=reasoning),),
        GatewayApiSurface.MESSAGES,
    )
    wires = (
        (
            GatewayWireProfile(
                dialect="openai_responses",
                url="https://api.openai.test/v1/responses",
                model_id="gpt-5.6-sol",
                supports_reasoning=True,
                reasoning_wire_format="openai_responses",
                supported_reasoning_efforts=("none", "low", "medium", "high"),
            ),
            client,
        ),
    )
    narrowed, _wires_out, public, provider = admitted_route_requests(
        route,
        wires,
        request,
        accounting=cast(NativeAttemptAccounting, accounting),
        authorization=route.snapshot.authorization,
    )

    assert tuple(item.deployment_id for item in narrowed.deployments) == ("gpt",)
    assert "thinking->reasoning_effort:medium" in public.ignored_parameters
    assert provider.provider_thinking_config is None
    assert provider.reasoning_effort == "medium"
    assert accounting.recorded == 1


def test_named_processing_tier_fails_closed_when_no_rung_offers_it() -> None:
    """A flex/priority request rejects fail-closed before reservation when no
    rung can honor it: a host lane without per-tier pass-through pricing and no
    BYOK rung. auto/default never reject."""
    from exp.runtime.gateway.native_accounting import NativeAttemptAccounting

    route = _mixed_route(
        "maximize_availability",
        (
            _deployment(
                "house",
                provider="openai",
                gateway=GatewayDeploymentMetadata(
                    capabilities=GatewayDeploymentCapabilities(supports_streaming=True)
                ),
            ),
        ),
        GatewayApiSurface.CHAT_COMPLETIONS,
    )
    client = cast(NativeWireClient, object())
    # House rung: billing_customer_managed False and no tier pricing, so it does
    # not forward service_tier.
    wires = ((GatewayWireProfile(dialect="openai_compatible", url="https://house.test"), client),)
    accounting = cast(NativeAttemptAccounting, object())

    for tier in ("flex", "priority"):
        request = GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=(GatewayMessage(role="user", content="go"),),
            service_tier=tier,
        )
        with pytest.raises(ProviderCapabilityError) as exc:
            admitted_route_requests(
                route,
                wires,
                request,
                accounting=accounting,
                authorization=route.snapshot.authorization,
            )
        assert exc.value.capability == "service_tier"

    # auto/default carry no price, and `scale` (a valid OpenAI tier we do not
    # price as opt-in) is stripped downstream, not rejected: only flex/priority
    # gate at admission.
    for tier in ("auto", "default", "scale"):
        request = GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=(GatewayMessage(role="user", content="go"),),
            service_tier=tier,
        )
        admitted_route_requests(
            route,
            wires,
            request,
            accounting=accounting,
            authorization=route.snapshot.authorization,
        )


def test_tier_priced_host_lane_admits_the_named_tier() -> None:
    """A host rung whose model carries per-tier pricing forwards the tier, so a
    flex request is admitted (not rejected)."""
    from exp.runtime.gateway.native_accounting import NativeAttemptAccounting

    route = _mixed_route(
        "maximize_availability",
        (
            _deployment(
                "house",
                provider="openai",
                gateway=GatewayDeploymentMetadata(
                    capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
                    prices=GatewayTokenPrices(
                        input_micro_usd_per_million_tokens=1_000_000,
                        output_micro_usd_per_million_tokens=4_000_000,
                        flex=GatewayServiceTierPrices(
                            input_micro_usd_per_million_tokens=500_000,
                            output_micro_usd_per_million_tokens=2_000_000,
                        ),
                    ),
                ),
            ),
        ),
        GatewayApiSurface.CHAT_COMPLETIONS,
    )
    client = cast(NativeWireClient, object())
    wires = (
        (
            GatewayWireProfile(
                dialect="openai_compatible",
                url="https://house.test",
                service_tier_pricing_enabled=True,
                service_tier_cards=frozenset({"flex"}),
            ),
            client,
        ),
    )
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="go"),),
        service_tier="flex",
    )
    _narrowed, _wires_out, _public, provider = admitted_route_requests(
        route,
        wires,
        request,
        accounting=cast(NativeAttemptAccounting, object()),
        authorization=route.snapshot.authorization,
    )
    # The tier survives to the provider request on the tier-priced house lane.
    assert provider.service_tier == "flex"


def test_tier_without_a_card_rejects_while_byok_forwards_any_tier() -> None:
    """The reject keys on the SPECIFIC requested tier's card, not just the lane:
    a house model carded for flex only rejects a priority request (no card ->
    would underbill), while a BYOK rung forwards any tier with no platform card
    (the customer pays the provider directly)."""
    from exp.runtime.gateway.native_accounting import NativeAttemptAccounting

    accounting = cast(NativeAttemptAccounting, object())
    client = cast(NativeWireClient, object())
    streaming = GatewayDeploymentCapabilities(supports_streaming=True)

    # House model carries a FLEX card only.
    flex_only_route = _mixed_route(
        "maximize_availability",
        (
            _deployment(
                "house",
                provider="openai",
                gateway=GatewayDeploymentMetadata(
                    capabilities=streaming,
                    prices=GatewayTokenPrices(
                        input_micro_usd_per_million_tokens=1_000_000,
                        output_micro_usd_per_million_tokens=4_000_000,
                        flex=GatewayServiceTierPrices(
                            input_micro_usd_per_million_tokens=500_000,
                            output_micro_usd_per_million_tokens=2_000_000,
                        ),
                    ),
                ),
            ),
        ),
        GatewayApiSurface.CHAT_COMPLETIONS,
    )
    house_wires = (
        (
            GatewayWireProfile(
                dialect="openai_compatible",
                url="https://house.test",
                service_tier_pricing_enabled=True,
                service_tier_cards=frozenset({"flex"}),
            ),
            client,
        ),
    )
    priority_request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="go"),),
        service_tier="priority",
    )
    with pytest.raises(ProviderCapabilityError) as exc:
        admitted_route_requests(
            flex_only_route,
            house_wires,
            priority_request,
            accounting=accounting,
            authorization=flex_only_route.snapshot.authorization,
        )
    assert exc.value.capability == "service_tier"

    # A BYOK rung (billing_customer_managed) forwards ANY tier with no card.
    byok_route = _mixed_route(
        "maximize_availability",
        (
            _deployment(
                "byok",
                provider="openai",
                gateway=GatewayDeploymentMetadata(capabilities=streaming),
            ),
        ),
        GatewayApiSurface.CHAT_COMPLETIONS,
    )
    byok_wires = (
        (
            GatewayWireProfile(
                dialect="openai_compatible",
                url="https://byok.test",
                billing_customer_managed=True,
            ),
            client,
        ),
    )
    _n, _w, _p, provider = admitted_route_requests(
        byok_route,
        byok_wires,
        priority_request,
        accounting=accounting,
        authorization=byok_route.snapshot.authorization,
    )
    assert provider.service_tier == "priority"


class _CoercionCounter:
    """Count coercion recordings without a live ledger."""

    def __init__(self) -> None:
        """Start at zero recorded coercions."""
        self.recorded = 0

    def record_admission_coercions(self, count: int) -> None:
        """Accumulate one admission's disclosure count."""
        self.recorded += count


_TOOL_CAPABLE = GatewayDeploymentMetadata(
    capabilities=GatewayDeploymentCapabilities(
        supports_streaming=True,
        supports_strict_tools=True,
        supports_streaming_tool_arguments=True,
    )
)
"""One rung declaration that admits every tool control these tests send."""


def _fable_and_shim_wires(
    *, shim_model: str = "anthropic/claude-fable-5-1"
) -> tuple[tuple[GatewayWireProfile, NativeWireClient], ...]:
    """Pair a native fable-5-1 rung with an OpenAI-compatible aggregator rung."""
    client = cast(NativeWireClient, object())
    return (
        (
            GatewayWireProfile(
                dialect="anthropic_messages",
                url="https://anthropic.test",
                model_id="claude-fable-5-1",
            ),
            client,
        ),
        (
            GatewayWireProfile(
                dialect="openai_compatible", url="https://shim.test", model_id=shim_model
            ),
            client,
        ),
    )


def _forced_choice_request(
    surface: GatewayApiSurface, choice: Literal["required"] | GatewayNamedToolChoice
) -> GatewayRequest:
    """Build one streaming request forcing the lookup tool on ``surface``."""
    return GatewayRequest(
        surface=surface,
        messages=(GatewayMessage(role="user", content="weather in Paris"),),
        tools=(GatewayToolDefinition(name="lookup", parameters={"type": "object"}),),
        tool_choice=choice,
        stream=True,
        include_usage=True,
    )


@pytest.mark.parametrize(
    ("surface", "choice"),
    (
        (GatewayApiSurface.CHAT_COMPLETIONS, "required"),
        (GatewayApiSurface.RESPONSES, GatewayNamedToolChoice(name="lookup")),
        (GatewayApiSurface.MESSAGES, "required"),
    ),
)
def test_a_forced_choice_narrows_to_the_rung_that_can_force_tools(
    surface: GatewayApiSurface, choice: Literal["required"] | GatewayNamedToolChoice
) -> None:
    """fable-5-1 declines ``any``/``tool`` by name, so a waterfall with an
    aggregator rung serves the caller's forced choice VERBATIM on that rung
    and discloses nothing."""
    deployments = (
        _deployment("native", provider="anthropic", gateway=_TOOL_CAPABLE),
        _deployment("shim", gateway=_TOOL_CAPABLE),
    )
    route = _mixed_route("maximize_availability", deployments, surface)
    accounting = _CoercionCounter()
    narrowed, _wires_out, public, provider = admitted_route_requests(
        route,
        _fable_and_shim_wires(),
        _forced_choice_request(surface, choice),
        accounting=cast(NativeAttemptAccounting, accounting),
        authorization=route.snapshot.authorization,
    )
    assert tuple(item.deployment_id for item in narrowed.deployments) == ("shim",)
    assert provider.tool_choice == choice
    assert public.ignored_parameters == ()
    assert accounting.recorded == 0


@pytest.mark.parametrize(
    ("surface", "choice"),
    (
        (GatewayApiSurface.CHAT_COMPLETIONS, GatewayNamedToolChoice(name="lookup")),
        (GatewayApiSurface.RESPONSES, "required"),
        (GatewayApiSurface.MESSAGES, GatewayNamedToolChoice(name="lookup")),
    ),
)
def test_a_forced_choice_relaxes_to_auto_with_disclosure_when_no_rung_can_force(
    surface: GatewayApiSurface, choice: Literal["required"] | GatewayNamedToolChoice
) -> None:
    """An all-fable-5-1 route (production shape: ~45 requests in 6h failed
    post-dispatch across all three surfaces) serves under ``auto`` and tells
    the caller through ``ignored_parameters``."""
    deployments = (_deployment("native", provider="anthropic", gateway=_TOOL_CAPABLE),)
    route = _mixed_route("maximize_availability", deployments, surface)
    accounting = _CoercionCounter()
    narrowed, _wires_out, public, provider = admitted_route_requests(
        route,
        _fable_and_shim_wires()[:1],
        _forced_choice_request(surface, choice),
        accounting=cast(NativeAttemptAccounting, accounting),
        authorization=route.snapshot.authorization,
    )
    assert tuple(item.deployment_id for item in narrowed.deployments) == ("native",)
    assert provider.tool_choice == "auto"
    assert public.tool_choice == "auto"
    assert public.ignored_parameters == ("tool_choice->auto",)
    assert accounting.recorded == 1


_MAX_ITEMS_SCHEMA: JsonObject = {
    "type": "object",
    "properties": {"cities": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
    "required": ["cities"],
    "additionalProperties": False,
}


def _strict_tool_request(parameters: JsonObject) -> GatewayRequest:
    """Build one streaming Chat request carrying a single strict tool."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="list cities"),),
        tools=(GatewayToolDefinition(name="list", parameters=parameters, strict=True),),
        stream=True,
        include_usage=True,
    )


def test_a_strict_schema_the_anthropic_validator_rejects_prefers_a_strict_capable_rung() -> None:
    """``maxItems`` under ``strict`` is a known Anthropic 400 (18 requests in
    6h in production), so the waterfall narrows to the OpenAI-compatible rung
    that honors strict verbatim and nothing is disclosed."""
    deployments = (
        _deployment("native", provider="anthropic", gateway=_TOOL_CAPABLE),
        _deployment("shim", gateway=_TOOL_CAPABLE),
    )
    route = _mixed_route("maximize_availability", deployments, GatewayApiSurface.CHAT_COMPLETIONS)
    accounting = _CoercionCounter()
    narrowed, _wires_out, public, provider = admitted_route_requests(
        route,
        _fable_and_shim_wires(),
        _strict_tool_request(_MAX_ITEMS_SCHEMA),
        accounting=cast(NativeAttemptAccounting, accounting),
        authorization=route.snapshot.authorization,
    )
    assert tuple(item.deployment_id for item in narrowed.deployments) == ("shim",)
    assert provider.tools[0].strict is True
    assert provider.tools[0].parameters == _MAX_ITEMS_SCHEMA
    assert public.ignored_parameters == ()
    assert accounting.recorded == 0


def test_a_strict_schema_no_rung_can_honor_drops_strict_and_keeps_the_schema() -> None:
    """On an all-Anthropic route the disclosed degrade drops only ``strict``;
    the schema, ``maxItems`` included, still reaches the model as guidance."""
    deployments = (_deployment("native", provider="anthropic", gateway=_TOOL_CAPABLE),)
    route = _mixed_route("maximize_availability", deployments, GatewayApiSurface.CHAT_COMPLETIONS)
    accounting = _CoercionCounter()
    narrowed, _wires_out, public, provider = admitted_route_requests(
        route,
        _fable_and_shim_wires()[:1],
        _strict_tool_request(_MAX_ITEMS_SCHEMA),
        accounting=cast(NativeAttemptAccounting, accounting),
        authorization=route.snapshot.authorization,
    )
    assert tuple(item.deployment_id for item in narrowed.deployments) == ("native",)
    assert provider.tools[0].strict is False
    assert provider.tools[0].parameters == _MAX_ITEMS_SCHEMA
    assert public.ignored_parameters == ("tools.strict->false",)
    assert accounting.recorded == 1


def test_an_open_strict_schema_is_closed_for_the_anthropic_rung_with_disclosure() -> None:
    """A strict tool whose objects leave ``additionalProperties`` open is a
    400 by name on Anthropic ("must be explicitly set to false"); admission
    closes the objects, keeps ``strict``, and discloses the tightening."""
    deployments = (_deployment("native", provider="anthropic", gateway=_TOOL_CAPABLE),)
    route = _mixed_route("maximize_availability", deployments, GatewayApiSurface.CHAT_COMPLETIONS)
    accounting = _CoercionCounter()
    open_schema: JsonObject = {"type": "object", "properties": {"city": {"type": "string"}}}
    _narrowed, _wires_out, public, provider = admitted_route_requests(
        route,
        _fable_and_shim_wires()[:1],
        _strict_tool_request(open_schema),
        accounting=cast(NativeAttemptAccounting, accounting),
        authorization=route.snapshot.authorization,
    )
    assert provider.tools[0].strict is True
    assert provider.tools[0].parameters == {**open_schema, "additionalProperties": False}
    assert public.ignored_parameters == ("tools.parameters.additionalProperties->false",)
    assert accounting.recorded == 1


def test_a_prompt_certain_to_overflow_the_route_is_refused_before_shaping() -> None:
    """The context-window refusal runs first: no rung shaping, no accounting, exact numbers."""
    deployments = (
        _deployment("shim").model_copy(
            update={"capabilities": ModelCapabilities(context_window_tokens=100)}
        ),
        _deployment("native").model_copy(
            update={"capabilities": ModelCapabilities(context_window_tokens=200)}
        ),
    )
    route = _mixed_route("maximize_availability", deployments)
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="x" * (201 * MAXIMUM_BYTES_PER_TOKEN)),),
    )

    with pytest.raises(ProviderParameterError) as caught:
        admitted_route_requests(
            route,
            _wires(),
            request,
            # Never reached: the refusal precedes every coercion or reservation.
            accounting=cast(NativeAttemptAccounting, object()),
            authorization=route.snapshot.authorization,
        )

    assert caught.value.code == "context_length_exceeded"
    assert caught.value.param == "messages"
    assert "at least 201 tokens" in str(caught.value)
    assert "200 tokens" in str(caught.value)


def test_parallel_tool_calls_shape_per_rung_capability() -> None:
    """A rung with the control forwards it; one without drops `true` or serializes `false`."""
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hi"),),
        parallel_tool_calls=False,
    )
    carried = GatewayDeploymentCapabilities(supports_parallel_tool_calls=True)
    missing = GatewayDeploymentCapabilities(supports_parallel_tool_calls=False)

    shaped, disclosure = shape_parallel_tool_calls(request, carried)
    assert shaped is request and disclosure is None

    shaped, disclosure = shape_parallel_tool_calls(request, missing)
    assert shaped.parallel_tool_calls is None and shaped.serialize_tool_calls is True
    assert disclosure == "parallel_tool_calls->emulated(serialized_by_gateway)"

    shaped, disclosure = shape_parallel_tool_calls(
        request.model_copy(update={"parallel_tool_calls": True}), missing
    )
    assert shaped.parallel_tool_calls is None and shaped.serialize_tool_calls is False
    assert disclosure == "parallel_tool_calls->dropped(provider_default)"

    untouched, disclosure = shape_parallel_tool_calls(
        request.model_copy(update={"parallel_tool_calls": None}), missing
    )
    assert untouched.parallel_tool_calls is None and disclosure is None


def _weighted_deployment(deployment_id: str, weight: float | None) -> ExactModelDeployment:
    """Build one rung carrying an authored affinity weight (or none)."""
    dispatch = None if weight is None else GatewayRungDispatchPolicy(affinity_weight=weight)
    return _deployment(deployment_id, gateway=GatewayDeploymentMetadata(dispatch=dispatch))


def _affinity_fixture(
    failover_mode: str = "maximize_cache_affinity",
) -> tuple[GatewayRoute, tuple[tuple[GatewayWireProfile, NativeWireClient], ...]]:
    """Build a three-rung affinity route over uniform openai-compatible wires."""
    deployments = (
        _weighted_deployment("dep-house", 10.0),
        _weighted_deployment("dep-fireworks", 3.0),
        _weighted_deployment("dep-openrouter", None),
    )
    route = _mixed_route(failover_mode, deployments, GatewayApiSurface.CHAT_COMPLETIONS)
    client = cast(NativeWireClient, object())
    wires = tuple(
        (
            GatewayWireProfile(
                dialect="openai_compatible", url=f"https://{item.deployment_id}.test"
            ),
            client,
        )
        for item in deployments
    )
    return route, wires


def _session_request(client_request_id: str) -> GatewayRequest:
    """Build one chat request carrying a session-scoped correlation id."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hi"),),
        client_request_id=client_request_id,
    )


def _order(route: GatewayRoute) -> tuple[str, ...]:
    """Name the route's dispatch order for readable assertions."""
    return tuple(item.deployment_id for item in route.deployments)


class TestAffinityOrderedRungs:
    """Rendezvous ordering under the affinity flag, and the flag-off gate."""

    def test_legacy_modes_keep_the_certified_order_object(self) -> None:
        """The two shipped modes return the identical route, byte for byte."""
        for mode in ("maximize_availability", "maximize_cache"):
            route, wires = _affinity_fixture(mode)
            ordered, ordered_wires = _affinity_ordered_rungs(
                route,
                wires,
                _session_request("session-1"),
                authorization=route.snapshot.authorization,
                continuation=None,
            )
            assert ordered is route
            assert ordered_wires is wires

    def test_simulated_workers_order_one_session_identically(self) -> None:
        """Independent computations of one session agree on the full ladder."""
        orders = set()
        for _worker in range(6):
            route, wires = _affinity_fixture()
            ordered, _wires_out = _affinity_ordered_rungs(
                route,
                wires,
                _session_request("session-42"),
                authorization=route.snapshot.authorization,
                continuation=None,
            )
            orders.add(_order(ordered))
        assert len(orders) == 1

    def test_different_sessions_reach_different_first_rungs(self) -> None:
        """The rendezvous spreads distinct conversations across rungs."""
        first_rungs = set()
        for index in range(64):
            route, wires = _affinity_fixture()
            ordered, _wires_out = _affinity_ordered_rungs(
                route,
                wires,
                _session_request(f"session-{index}"),
                authorization=route.snapshot.authorization,
                continuation=None,
            )
            first_rungs.add(_order(ordered)[0])
        assert len(first_rungs) > 1

    def test_wires_stay_aligned_with_the_reordered_route(self) -> None:
        """Each reordered rung keeps its own resolved wire."""
        route, wires = _affinity_fixture()
        ordered, ordered_wires = _affinity_ordered_rungs(
            route,
            wires,
            _session_request("session-7"),
            authorization=route.snapshot.authorization,
            continuation=None,
        )
        for deployment, (profile, _client) in zip(ordered.deployments, ordered_wires, strict=True):
            assert profile.url == f"https://{deployment.deployment_id}.test"

    def test_continuation_keeps_the_original_turns_placement(self) -> None:
        """A continued conversation orders exactly like its originating session."""
        from exp.runtime.gateway.native_responses import ContinuationContext
        from exp.runtime.openai_protocol.state import ProtocolNamespace

        route, wires = _affinity_fixture()
        original, _wires_out = _affinity_ordered_rungs(
            route,
            wires,
            _session_request("session-original"),
            authorization=route.snapshot.authorization,
            continuation=None,
        )
        continuation = ContinuationContext(
            namespace=ProtocolNamespace(
                organization_id="organization-one",
                identity_id="identity-one",
                alias_revision_id="revision-one",
            ),
            episode_key="session-original",
            response_id="resp-1",
            messages=(),
        )
        continued_route, continued_wires = _affinity_fixture()
        continued, _wires_out = _affinity_ordered_rungs(
            continued_route,
            continued_wires,
            _session_request("a-fresh-per-turn-id"),
            authorization=continued_route.snapshot.authorization,
            continuation=continuation,
        )
        assert _order(continued) == _order(original)

    def test_marked_requests_keep_marker_honoring_rungs_first(self) -> None:
        """#717 composes: markers partition first, rendezvous orders within."""
        deployments = (
            _weighted_deployment("dep-shim", 10.0),
            _weighted_deployment("dep-native-a", 3.0),
            _weighted_deployment("dep-native-b", 1.0),
        )
        route = _mixed_route("maximize_cache_affinity", deployments, GatewayApiSurface.MESSAGES)
        client = cast(NativeWireClient, object())
        wires = (
            (GatewayWireProfile(dialect="openai_compatible", url="https://shim.test"), client),
            (GatewayWireProfile(dialect="anthropic_messages", url="https://a.test"), client),
            (GatewayWireProfile(dialect="anthropic_messages", url="https://b.test"), client),
        )
        marked = _marked_request().model_copy(update={"client_request_id": "session-1"})
        ordered, _wires_out = _affinity_ordered_rungs(
            route,
            wires,
            marked,
            authorization=route.snapshot.authorization,
            continuation=None,
        )
        assert set(_order(ordered)[:2]) == {"dep-native-a", "dep-native-b"}
        assert _order(ordered)[2] == "dep-shim"
        # The markerless order restricted to the native group matches the
        # within-group order of the marked request: one rendezvous, two views.
        plain_route, plain_wires = (
            _mixed_route("maximize_cache_affinity", deployments, GatewayApiSurface.MESSAGES),
            wires,
        )
        plain_ordered, _wires_out = _affinity_ordered_rungs(
            plain_route,
            plain_wires,
            _session_request("session-1"),
            authorization=plain_route.snapshot.authorization,
            continuation=None,
        )
        plain_native_order = tuple(name for name in _order(plain_ordered) if name != "dep-shim")
        assert _order(ordered)[:2] == plain_native_order
