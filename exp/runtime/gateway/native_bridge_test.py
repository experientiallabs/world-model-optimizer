"""Tests for the native-engine control plane and its committed golden fixtures."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Literal, cast
from unittest import mock

import pytest

import exp.runtime.gateway.native_bridge as native_bridge_module
from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    GatewayDeploymentCapabilities,
    GatewayTokenPrices,
    ModelCapabilities,
)
from exp.runtime.gateway.budgets import BudgetReservationRejected, BudgetScopeKind
from exp.runtime.gateway.catalog_authority import upsert_singleton_deployment
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    GatewayApiSurface,
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayRequest,
    GatewayUsage,
)
from exp.runtime.gateway.ledger import SQLiteAttemptLedger
from exp.runtime.gateway.lifecycle import (
    LocalGatewayComponents,
    _ReadyControlStore,
    load_gateway_components,
)
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.native_bridge import (
    NativeBridgeError,
    NativeControlPlane,
    _public_capability_error,
)
from exp.runtime.gateway.native_bridge_errors import capability_param as _public_capability_param
from exp.runtime.gateway.native_components import NativeGatewayComponents
from exp.runtime.gateway.routing import GatewayRoutingError
from exp.runtime.models.providers.errors import ProviderCapabilityError
from exp.runtime.models.providers.streaming_requests import openai_compatible_stream_payload
from exp.runtime.openai_protocol.errors import OpenAIProtocolError, public_failure_error
from exp.runtime.openai_protocol.requests import decode_chat, decode_responses
from exp.runtime.openai_protocol.state import ProtocolNamespace
from exp.runtime.openai_protocol.streaming import stable_public_id

_PARITY_GOLDENS_PATH = Path(__file__).parent / "testdata" / "parity_goldens.json"
_PublicErrorType = Literal[
    "invalid_request_error",
    "authentication_error",
    "permission_error",
    "insufficient_quota",
    "api_error",
]


@pytest.mark.parametrize(
    ("capability", "chat_param", "responses_param", "messages_param"),
    (
        ("developer_messages", "messages", "instructions", "system"),
        ("function_tools", "tools", "tools", "tools"),
        ("forced_tool_choice", "tool_choice", "tool_choice", "tool_choice"),
        (
            "parallel_tool_calls",
            "parallel_tool_calls",
            "parallel_tool_calls",
            "tool_choice.disable_parallel_tool_use",
        ),
        ("stop_sequences", "stop", None, "stop_sequences"),
        ("streaming", "stream", "stream", "stream"),
        ("streaming_tool_arguments", "tools", "tools", "tools"),
        ("strict_tools", "tools", "tools", "tools"),
        ("structured_output", "response_format", "text.format", None),
        ("structured_text", "response_format", "text.format", None),
    ),
)
def test_public_capability_params_name_real_surface_fields(
    capability: str,
    chat_param: str | None,
    responses_param: str | None,
    messages_param: str | None,
) -> None:
    """Internal admission labels never leak as nonexistent public fields."""
    assert _public_capability_param(capability, GatewayApiSurface.CHAT_COMPLETIONS) == chat_param
    assert _public_capability_param(capability, GatewayApiSurface.RESPONSES) == responses_param
    assert _public_capability_param(capability, GatewayApiSurface.MESSAGES) == messages_param


def test_tool_argument_streaming_failure_names_the_public_tools_field() -> None:
    """Tool argument transport failures identify the tool request that activated them."""
    for surface in GatewayApiSurface:
        assert (
            _public_capability_param(
                "streaming_tool_arguments",
                surface,
                public_stream=False,
                public_tools=True,
            )
            == "tools"
        )


def test_internal_text_streaming_failure_has_no_fake_public_field() -> None:
    """An internal text transport deficit has no caller-removable request field."""
    for surface in GatewayApiSurface:
        assert (
            _public_capability_param(
                "streaming",
                surface,
                public_stream=False,
            )
            is None
        )

        error = _public_capability_error(
            ProviderCapabilityError(capability="streaming"),
            surface,
            public_stream=False,
            public_tools=False,
        )
        assert error.detail.param == "model"
        assert "Choose a different model alias" in error.detail.message


@pytest.mark.parametrize(
    ("surface", "param"),
    [
        (GatewayApiSurface.CHAT_COMPLETIONS, "messages"),
        (GatewayApiSurface.RESPONSES, "input"),
        (GatewayApiSurface.MESSAGES, "messages"),
    ],
)
def test_pdf_capability_errors_explain_the_document_refusal(
    surface: GatewayApiSurface, param: str
) -> None:
    """A refused PDF names the conversation field and says why, on every surface."""
    inline = _public_capability_error(
        ProviderCapabilityError(capability="pdf_input"),
        surface,
        public_stream=False,
        public_tools=False,
    )
    remote = _public_capability_error(
        ProviderCapabilityError(capability="pdf_url_input"),
        surface,
        public_stream=False,
        public_tools=False,
    )
    assert (inline.detail.param, remote.detail.param) == (param, param)
    assert inline.detail.code == remote.detail.code == "unsupported_capability"
    assert "cannot accept PDF document input" in inline.detail.message
    assert "inline PDF data only" in remote.detail.message
    assert "pdf_input" not in inline.detail.message


@pytest.mark.parametrize(
    ("surface", "param"),
    [
        (GatewayApiSurface.CHAT_COMPLETIONS, "messages"),
        (GatewayApiSurface.RESPONSES, "input"),
        (GatewayApiSurface.MESSAGES, "messages"),
    ],
)
def test_media_handle_capability_errors_name_the_provider_holding_the_upload(
    surface: GatewayApiSurface, param: str
) -> None:
    """A refused handle names the conversation field and, when known, the provider."""
    undeclared = _public_capability_error(
        ProviderCapabilityError(capability="media_handle_input"),
        surface,
        public_stream=False,
        public_tools=False,
    )
    detail = (
        "The request references media uploaded to openai, which only an openai route "
        "can resolve, but the selected model alias routes to gemini."
    )
    mismatched = _public_capability_error(
        ProviderCapabilityError(capability="media_handle_provider", detail=detail),
        surface,
        public_stream=False,
        public_tools=False,
    )
    generic = _public_capability_error(
        ProviderCapabilityError(capability="media_handle_provider"),
        surface,
        public_stream=False,
        public_tools=False,
    )
    assert (undeclared.detail.param, mismatched.detail.param) == (param, param)
    assert undeclared.detail.code == mismatched.detail.code == "unsupported_capability"
    assert "cannot reference media uploaded to a provider" in undeclared.detail.message
    assert mismatched.detail.message == detail
    assert "different provider than the selected model route" in generic.detail.message
    assert "media_handle" not in undeclared.detail.message


@pytest.mark.parametrize(
    ("surface", "param"),
    [(GatewayApiSurface.CHAT_COMPLETIONS, "messages"), (GatewayApiSurface.RESPONSES, "input")],
)
def test_audio_capability_error_explains_the_refusal(
    surface: GatewayApiSurface, param: str
) -> None:
    """A refused clip names the conversation field and says why, on every surface."""
    error = _public_capability_error(
        ProviderCapabilityError(capability="audio_input"),
        surface,
        public_stream=False,
        public_tools=False,
    )
    assert error.detail.param == param
    assert error.detail.code == "unsupported_capability"
    assert "cannot accept audio input" in error.detail.message
    assert "audio_input" not in error.detail.message


def test_forced_tool_choice_refusal_tells_the_caller_what_to_send_instead() -> None:
    """The forced-choice refusal names the field AND the way out ('auto'), on
    every surface: the generic "remove the unsupported field" wording left 64
    callers in two hours (2026-09-06) with nothing actionable."""
    for surface in (
        GatewayApiSurface.CHAT_COMPLETIONS,
        GatewayApiSurface.RESPONSES,
        GatewayApiSurface.MESSAGES,
    ):
        error = _public_capability_error(
            ProviderCapabilityError(capability="forced_tool_choice"),
            surface,
            public_stream=True,
            public_tools=True,
        )
        assert error.status_code == 400
        assert error.detail.param == "tool_choice"
        assert error.detail.code == "unsupported_capability"
        assert "tool_choice 'auto'" in error.detail.message
        assert "forced_tool_choice" not in error.detail.message


def test_public_capability_error_never_exposes_internal_labels() -> None:
    """Internal route requirements fail against model without leaking their names."""
    error = _public_capability_error(
        ProviderCapabilityError(capability="tinker_gateway_execution"),
        GatewayApiSurface.CHAT_COMPLETIONS,
        public_stream=False,
        public_tools=False,
    )

    assert error.detail.param == "model"
    assert error.detail.code == "unsupported_capability"
    assert "tinker_gateway_execution" not in error.detail.message


def test_forced_streaming_tool_deficit_names_the_public_tools_field() -> None:
    """An internally streamed tool deficit names the tool request that caused it."""
    error = _public_capability_error(
        ProviderCapabilityError(capability="streaming_tool_arguments"),
        GatewayApiSurface.CHAT_COMPLETIONS,
        public_stream=False,
        public_tools=True,
    )

    assert error.detail.param == "tools"


def _parity_golden(name: str) -> object:
    """Load one committed golden fixture the Rust encoders must reproduce.

    The committed bytes are the durable parity contract: they were generated
    from the retired python data plane's encoders and outlive that code, so
    a Rust encoding change can never weaken parity silently.

    Args:
        name: Top-level key inside ``testdata/parity_goldens.json``.

    Returns:
        The committed golden value (a frame list or a serialized body).
    """
    with _PARITY_GOLDENS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)[name]


def _control_plane(
    root: Path,
    *,
    request_timeout_seconds: float = 120.0,
) -> tuple[NativeControlPlane, str]:
    """Seed one direct alias and load the native control plane over it."""
    _manager, raw_key = _configured_gateway(root)
    components = load_gateway_components(
        root,
        environment={"TEST_PROVIDER_KEY": "provider-secret-canary"},
    )
    control = NativeControlPlane(components, request_timeout_seconds=request_timeout_seconds)
    return control, raw_key


def _chat_body(*, model: str = "coding", stream: bool = False) -> str:
    """Return one raw Chat Completions request body."""
    payload: JsonObject = {"model": model, "messages": [{"role": "user", "content": "hi"}]}
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    return json.dumps(payload)


def _admit(
    control: NativeControlPlane,
    raw_key: str,
    body: str,
    *,
    surface: str | None = None,
    idempotency_key: str | None = None,
    client_request_id: str | None = None,
) -> JsonObject:
    """Run one admission call and decode its JSON response."""
    payload: JsonObject = {
        "raw_key": raw_key,
        "body": body,
        "idempotency_key": idempotency_key,
        "client_request_id": client_request_id,
    }
    if surface is not None:
        payload["surface"] = surface
    return json.loads(control.admit(json.dumps(payload)))


def _claim_scope(
    control: NativeControlPlane,
    raw_key: str,
    body: str,
    *,
    surface: str = "chat",
    idempotency_key: str | None = None,
    client_request_id: str | None = None,
) -> JsonObject:
    """Run one replay-scope call and decode its JSON response."""
    argument = json.dumps(
        {
            "raw_key": raw_key,
            "body": body,
            "surface": surface,
            "idempotency_key": idempotency_key,
            "client_request_id": client_request_id,
        }
    )
    return json.loads(control.claim_scope(argument))


def _start_first(control: NativeControlPlane, admission: JsonObject) -> JsonObject:
    """Reserve the first physical dispatch for one admitted request."""
    return json.loads(
        control.start_attempt(
            json.dumps({"request_id": admission["request_id"], "attempt_ordinal": 0})
        )
    )


def _flatten_started(control: NativeControlPlane, admission: JsonObject) -> JsonObject:
    """Reserve the first dispatch and flatten its wire entry into the admission."""
    started = _start_first(control, admission)
    route = admission["route"]
    assert isinstance(route, list)
    depth = started["route_depth"]
    assert isinstance(depth, int)
    wire = route[depth]
    assert isinstance(wire, dict)
    return {**admission, **wire, **started}


def _admit_started(
    control: NativeControlPlane,
    raw_key: str,
    body: str,
    *,
    idempotency_key: str | None = None,
    client_request_id: str | None = None,
) -> JsonObject:
    """Admit one request, reserve its first dispatch, and flatten the wire view."""
    admission = _admit(
        control,
        raw_key,
        body,
        idempotency_key=idempotency_key,
        client_request_id=client_request_id,
    )
    return _flatten_started(control, admission)


def test_fireworks_carrier_round_trip_rejects_tamper_and_credential_rotation(
    tmp_path: Path,
) -> None:
    """A second replica decrypts the exact turn while tamper and rotation fail closed."""
    _manager, raw_key = _configured_gateway(
        tmp_path,
        base_url="https://api.fireworks.ai/inference/v1",
        capabilities=ModelCapabilities(supports_tools=True),
    )
    first_control = NativeControlPlane(
        load_gateway_components(
            tmp_path,
            environment={"TEST_PROVIDER_KEY": "shared-fireworks-secret"},
        )
    )
    initial = _admit_started(first_control, raw_key, _chat_body())
    hidden = "private provider reasoning that must stay opaque"
    seal_argument = json.dumps(
        {
            "request_id": initial["request_id"],
            "route_depth": initial["route_depth"],
            "route_sha256": initial["fireworks_reasoning_route_sha256"],
            "content": hidden,
            "assistant_content": None,
            "tool_calls": [{"call_id": "call-one", "name": "lookup", "raw_arguments": "{}"}],
        }
    )
    sealed = json.loads(first_control.seal_reasoning_content(seal_argument))["carrier"]
    assert hidden not in sealed
    assert "shared-fireworks-secret" not in sealed
    assert (
        first_control.settle(
            json.dumps(
                {
                    "request_id": initial["request_id"],
                    "attempt_id": initial["attempt_id"],
                    "outcome": "completed",
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                    "tool_names": ["lookup"],
                    "failure": None,
                }
            )
        )
        == "{}"
    )
    with pytest.raises(NativeBridgeError):
        first_control.seal_reasoning_content(seal_argument)
    continuation_body = json.dumps(
        {
            "model": "coding",
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": sealed,
                    "tool_calls": [
                        {
                            "id": "call-one",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-one", "content": "done"},
            ],
        }
    )

    replica = NativeControlPlane(
        load_gateway_components(
            tmp_path,
            environment={"TEST_PROVIDER_KEY": "shared-fireworks-secret"},
        )
    )
    continued = _admit(replica, raw_key, continuation_body)
    route = cast("list[JsonObject]", continued["route"])
    payload = cast("JsonObject", route[0]["upstream_payload"])
    messages = cast("list[JsonObject]", payload["messages"])
    assert continued["route_reason"] == "reasoning_continuation"
    assert messages[1]["reasoning_content"] == hidden

    transplanted = json.loads(continuation_body)
    transplanted["messages"][0]["content"] = "Use this carrier under a different prompt"
    with pytest.raises(NativeBridgeError) as transplanted_error:
        _admit(replica, raw_key, json.dumps(transplanted))
    assert (
        json.loads(transplanted_error.value.public_error_json)["param"]
        == "messages.reasoning_content"
    )

    modified_turn = json.loads(continuation_body)
    modified_turn["messages"][1]["tool_calls"][0]["function"]["arguments"] = '{"tampered":true}'
    with pytest.raises(NativeBridgeError) as modified:
        _admit(replica, raw_key, json.dumps(modified_turn))
    assert json.loads(modified.value.public_error_json)["param"] == "messages.reasoning_content"

    rotated = NativeControlPlane(
        load_gateway_components(
            tmp_path,
            environment={"TEST_PROVIDER_KEY": "rotated-fireworks-secret"},
        )
    )
    with pytest.raises(NativeBridgeError) as rejected:
        _admit(rotated, raw_key, continuation_body)
    public_error = json.loads(rejected.value.public_error_json)
    assert public_error["status_code"] == 400
    assert public_error["param"] == "messages.reasoning_content"

    stale_turn = json.loads(continuation_body)
    stale_turn["messages"].append({"role": "user", "content": "Start a new turn"})
    after_rotation = _admit(rotated, raw_key, json.dumps(stale_turn))
    rotated_route = cast("list[JsonObject]", after_rotation["route"])
    rotated_payload = cast("JsonObject", rotated_route[0]["upstream_payload"])
    rotated_messages = cast("list[JsonObject]", rotated_payload["messages"])
    assert after_rotation["route_reason"] == "direct"
    assert "reasoning_content" not in rotated_messages[1]


def test_hunyuan_tool_turn_reasoning_round_trips_as_a_sealed_carrier(
    tmp_path: Path,
) -> None:
    """A Hunyuan TOOL turn round-trips its reasoning as the sealed carrier.

    The rung is marked as an exposed-plaintext reasoning route on the wire (so
    the data plane returns ``reasoning_content`` to the caller on plain turns,
    which replay as plaintext — see the plain-turn test), while a tool turn's
    round-trip token stays the domain-separated opaque carrier: a
    second replica unseals the exact turn and forwards the plaintext upstream,
    the Fireworks-only ``reasoning_history`` wire flag never appears, and a
    tampered turn fails closed.
    """
    _manager, raw_key = _configured_gateway(
        tmp_path,
        base_url="https://api.hunyuan.cloud.tencent.com/v1",
        capabilities=ModelCapabilities(supports_tools=True, reasoning_output_exposed=True),
    )
    control = NativeControlPlane(
        load_gateway_components(
            tmp_path,
            environment={"TEST_PROVIDER_KEY": "shared-hunyuan-secret"},
        )
    )
    initial = _admit_started(control, raw_key, _chat_body())
    # The rung declares plaintext exposure and a Hunyuan carrier route identity,
    # and is not a Fireworks route.
    assert initial["reasoning_output_exposed"] is True
    assert initial["fireworks_reasoning_route_sha256"] is None
    route_sha256 = initial["hunyuan_reasoning_route_sha256"]
    assert isinstance(route_sha256, str)

    hidden = "let me reason about the tool call privately"
    seal_argument = json.dumps(
        {
            "request_id": initial["request_id"],
            "route_depth": initial["route_depth"],
            "route_sha256": route_sha256,
            "content": hidden,
            "assistant_content": None,
            "tool_calls": [{"call_id": "call-one", "name": "lookup", "raw_arguments": "{}"}],
        }
    )
    sealed = json.loads(control.seal_reasoning_content(seal_argument))["carrier"]
    # The carrier is opaque under the Hunyuan scheme and leaks neither the
    # plaintext reasoning nor the provider credential.
    assert sealed.startswith("x-experiential-hunyuan-reasoning-v1:")
    assert hidden not in sealed
    assert "shared-hunyuan-secret" not in sealed
    assert (
        control.settle(
            json.dumps(
                {
                    "request_id": initial["request_id"],
                    "attempt_id": initial["attempt_id"],
                    "outcome": "completed",
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                    "tool_names": ["lookup"],
                    "failure": None,
                }
            )
        )
        == "{}"
    )

    continuation_body = json.dumps(
        {
            "model": "coding",
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": sealed,
                    "tool_calls": [
                        {
                            "id": "call-one",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-one", "content": "done"},
            ],
        }
    )
    replica = NativeControlPlane(
        load_gateway_components(
            tmp_path,
            environment={"TEST_PROVIDER_KEY": "shared-hunyuan-secret"},
        )
    )
    continued = _admit(replica, raw_key, continuation_body)
    route = cast("list[JsonObject]", continued["route"])
    payload = cast("JsonObject", route[0]["upstream_payload"])
    messages = cast("list[JsonObject]", payload["messages"])
    assert continued["route_reason"] == "reasoning_continuation"
    # The sealed carrier unseals to the exact plaintext forwarded upstream on the
    # assistant turn, and Hunyuan omits the Fireworks-only reasoning_history flag.
    assert messages[1]["reasoning_content"] == hidden
    assert "reasoning_history" not in payload

    # A tampered tool turn fails closed at the carrier authority.
    modified_turn = json.loads(continuation_body)
    modified_turn["messages"][1]["tool_calls"][0]["function"]["arguments"] = '{"tampered":true}'
    with pytest.raises(NativeBridgeError) as modified:
        _admit(replica, raw_key, json.dumps(modified_turn))
    assert json.loads(modified.value.public_error_json)["param"] == "messages.reasoning_content"


@pytest.mark.parametrize("thinking", ["", "The user wants a directory listing; ls is the command."])
def test_hunyuan_plain_turn_plaintext_reasoning_replays_verbatim(
    tmp_path: Path,
    thinking: str,
) -> None:
    """A Terminus-shaped loop round-trips the plaintext the rung itself returned.

    Terminus-2 parses commands out of assistant TEXT and feeds the output back
    as a user message, so every turn is a non-tool turn: the exposing rung
    returns plaintext ``reasoning_content`` and the caller echoes it. The
    provider's wire accepts that text verbatim and validates nothing about it,
    so the gateway forwards it to the exposing rung — no carrier, no route
    pin, no disclosure.
    """
    _manager, raw_key = _configured_gateway(
        tmp_path,
        base_url="https://tokenhub-intl.tencentcloudmaas.com/v1",
        capabilities=ModelCapabilities(supports_tools=True, reasoning_output_exposed=True),
    )
    control = NativeControlPlane(
        load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "k"})
    )
    body = json.dumps(
        {
            "model": "coding",
            "messages": [
                {"role": "user", "content": "List the files."},
                {
                    "role": "assistant",
                    "content": '{"command": "ls"}',
                    "reasoning_content": thinking,
                },
                {"role": "user", "content": "a.txt b.txt"},
            ],
        }
    )
    admitted = _admit(control, raw_key, body)
    route = cast("list[JsonObject]", admitted["route"])
    payload = cast("JsonObject", route[0]["upstream_payload"])
    messages = cast("list[JsonObject]", payload["messages"])
    assert messages[1]["reasoning_content"] == thinking
    assert admitted["route_reason"] != "reasoning_continuation"
    assert "reasoning_history" not in payload
    assert admitted.get("ignored_parameters", []) == []


def test_hunyuan_mixed_carrier_and_plaintext_history_round_trips(tmp_path: Path) -> None:
    """Harbor with interleaved thinking echoes BOTH shapes and both replay.

    A tool turn carries the sealed carrier (unsealed to its plaintext and
    pinned to the issuing rung); a later plain turn carries the plaintext the
    rung returned. One history, both forwarded verbatim.
    """
    _manager, raw_key = _configured_gateway(
        tmp_path,
        base_url="https://tokenhub-intl.tencentcloudmaas.com/v1",
        capabilities=ModelCapabilities(supports_tools=True, reasoning_output_exposed=True),
    )
    control = NativeControlPlane(
        load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "k"})
    )
    initial = _admit_started(control, raw_key, _chat_body())
    route_sha256 = initial["hunyuan_reasoning_route_sha256"]
    hidden = "reason about the lookup privately"
    sealed = json.loads(
        control.seal_reasoning_content(
            json.dumps(
                {
                    "request_id": initial["request_id"],
                    "route_depth": initial["route_depth"],
                    "route_sha256": route_sha256,
                    "content": hidden,
                    "assistant_content": None,
                    "tool_calls": [
                        {"call_id": "call-one", "name": "lookup", "raw_arguments": "{}"}
                    ],
                }
            )
        )
    )["carrier"]
    control.settle(
        json.dumps(
            {
                "request_id": initial["request_id"],
                "attempt_id": initial["attempt_id"],
                "outcome": "completed",
                "usage": {"input_tokens": 3, "output_tokens": 2},
                "tool_names": ["lookup"],
                "failure": None,
            }
        )
    )
    plain = "now summarize what the tool said"
    body = json.dumps(
        {
            "model": "coding",
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": sealed,
                    "tool_calls": [
                        {
                            "id": "call-one",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-one", "content": "done"},
                {"role": "assistant", "content": "It said done.", "reasoning_content": plain},
                {"role": "user", "content": "thanks, and now?"},
            ],
        }
    )
    admitted = _admit(control, raw_key, body)
    route = cast("list[JsonObject]", admitted["route"])
    payload = cast("JsonObject", route[0]["upstream_payload"])
    messages = cast("list[JsonObject]", payload["messages"])
    # The carrier precedes the latest user turn, so it is stale history and
    # is stripped exactly as before; the plaintext plain turn replays.
    assert "reasoning_content" not in messages[1]
    assert messages[3]["reasoning_content"] == plain


@pytest.mark.parametrize("reasoning", ["", "private"])
def test_plaintext_reasoning_degrades_on_a_route_without_exposure(
    tmp_path: Path,
    reasoning: str,
) -> None:
    """A rung that cannot replay plaintext reasoning drops it with disclosure.

    The block is baked into the caller's transcript (an earlier exposed-rung
    turn or a client re-serialization), so admission serves the request and
    discloses the drop — previously a named 400 that killed every session
    the moment it switched from a reasoning-exposed model to any other."""
    _manager, raw_key = _configured_gateway(tmp_path, capabilities=ModelCapabilities())
    control = NativeControlPlane(
        load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "k"})
    )
    body = json.dumps(
        {
            "model": "coding",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "x", "reasoning_content": reasoning},
                {"role": "user", "content": "again"},
            ],
        }
    )
    admitted = _admit(control, raw_key, body)
    ignored = cast("list[str]", admitted["ignored_parameters"])
    assert "messages.reasoning_content->dropped(unsupported_by_provider)" in ignored
    route = cast("list[JsonObject]", admitted["route"])
    payload = cast("JsonObject", route[0]["upstream_payload"])
    sent_messages = cast("list[JsonObject]", payload["messages"])
    assert all("reasoning_content" not in message for message in sent_messages)


def test_hunyuan_endpoint_without_exposure_capability_strips_reasoning(
    tmp_path: Path,
) -> None:
    """A Hunyuan-endpoint rung that does not declare exposure keeps reasoning stripped.

    Exposure is gated on the explicit per-rung capability, not the base URL, so a
    model added to the Tencent endpoint without ``reasoning_output_exposed`` fails
    closed: the data plane still recognizes the carrier route (round-trips stay
    sealed) but never surfaces plaintext ``reasoning_content`` to the caller.
    """
    _manager, raw_key = _configured_gateway(
        tmp_path,
        base_url="https://api.hunyuan.cloud.tencent.com/v1",
        capabilities=ModelCapabilities(supports_tools=True),
    )
    control = NativeControlPlane(
        load_gateway_components(
            tmp_path,
            environment={"TEST_PROVIDER_KEY": "shared-hunyuan-secret"},
        )
    )
    initial = _admit_started(control, raw_key, _chat_body())
    # No exposure capability -> plaintext stays stripped even on the Hunyuan URL,
    # while the carrier route identity still resolves so replay stays sealed.
    assert initial["reasoning_output_exposed"] is False
    assert isinstance(initial["hunyuan_reasoning_route_sha256"], str)
    assert initial["fireworks_reasoning_route_sha256"] is None


def test_fireworks_continuation_pins_the_exact_issuing_fallback_rung(tmp_path: Path) -> None:
    """A fallback-issued carrier replays only to its exact deployment and credential."""
    _manager, raw_key = _configured_pool_gateway(
        tmp_path,
        base_urls=(
            "https://api.fireworks.ai/inference/v1",
            "https://api.fireworks.ai/inference/v1",
        ),
    )
    control = NativeControlPlane(
        load_gateway_components(
            tmp_path,
            environment={"TEST_PROVIDER_KEY": "shared-fireworks-secret"},
        )
    )
    admission = _admit(control, raw_key, _chat_body())
    first = _start_first(control, admission)
    failure = {
        "failure_class": "provider_internal",
        "safe_message": "provider service failed",
        "retryable_same_deployment": False,
        "failover_eligible": True,
    }
    assert (
        control.settle(
            json.dumps(
                {
                    "request_id": admission["request_id"],
                    "attempt_id": first["attempt_id"],
                    "outcome": "failed",
                    "usage": None,
                    "tool_names": [],
                    "failure": failure,
                    "finalize": False,
                }
            )
        )
        == "{}"
    )
    second = json.loads(
        control.start_attempt(
            json.dumps(
                {
                    "request_id": admission["request_id"],
                    "attempt_ordinal": 1,
                    "current_depth": 0,
                    "failure": failure,
                }
            )
        )
    )
    assert second["route_depth"] == 1
    route = cast("list[JsonObject]", admission["route"])
    carrier = json.loads(
        control.seal_reasoning_content(
            json.dumps(
                {
                    "request_id": admission["request_id"],
                    "route_depth": 1,
                    "route_sha256": route[1]["fireworks_reasoning_route_sha256"],
                    "content": "fallback-private-reasoning",
                    "assistant_content": None,
                    "tool_calls": [
                        {"call_id": "call-one", "name": "lookup", "raw_arguments": "{}"}
                    ],
                }
            )
        )
    )["carrier"]
    continuation = _admit(
        control,
        raw_key,
        json.dumps(
            {
                "model": "coding",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": carrier,
                        "tool_calls": [
                            {
                                "id": "call-one",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call-one", "content": "done"},
                ],
            }
        ),
    )

    continued_route = cast("list[JsonObject]", continuation["route"])
    assert [item["deployment_id"] for item in continued_route] == [route[1]["deployment_id"]]
    assert continued_route[0]["model_id"] == "beta-model-exact"


def test_bridge_error_payload_is_openai_shaped() -> None:
    """The boundary error carries the exact public error representation."""
    error = NativeBridgeError(
        OpenAIProtocolError(
            status_code=429,
            code="insufficient_quota",
            message="monthly gateway allocation is exhausted",
            error_type="insufficient_quota",
            retry_after_seconds=60,
        )
    )
    payload = json.loads(error.public_error_json)
    assert payload == {
        "status_code": 429,
        "code": "insufficient_quota",
        "message": "monthly gateway allocation is exhausted",
        "error_type": "insufficient_quota",
        "param": None,
        "retry_after_seconds": 60,
    }


def test_admit_decodes_builds_payload_and_settles(tmp_path: Path) -> None:
    """Admission decodes the raw body, returns the shared upstream payload, and
    settlement lands in the usage report."""
    control, raw_key = _control_plane(tmp_path)
    assert control.authenticate(json.dumps({"raw_key": raw_key})) == "{}"

    admission = _admit_started(control, raw_key, _chat_body())
    assert admission["maximum_total_attempts"] == 8
    assert admission["maximum_same_deployment_attempts"] == 2
    assert admission["refusal_failover"] is False
    assert admission["route_depth"] == 0
    assert admission["dialect"] == "openai_compatible"
    url = admission["url"]
    assert isinstance(url, str) and url.endswith("/chat/completions")
    assert admission["model_id"] == "provider-model-exact"
    headers = admission["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == "Bearer provider-secret-canary"
    assert admission["provider"] == "openai-compatible"
    assert admission["route_reason"] == "direct"
    assert admission["stream"] is False
    assert admission["include_usage"] is False

    decoded = decode_chat(json.loads(_chat_body()))
    provider_request = decoded.request.model_copy(update={"stream": True, "include_usage": True})
    # A generic OpenAI-compatible shim never receives the cache-affinity key
    # (only OpenAI and Tencent endpoints route by it), so the payload is the
    # plain build.
    assert admission["upstream_payload"] == openai_compatible_stream_payload(
        "provider-model-exact", provider_request
    )

    settled = control.settle(
        json.dumps(
            {
                "request_id": admission["request_id"],
                "attempt_id": admission["attempt_id"],
                "outcome": "completed",
                "usage": {"input_tokens": 12, "output_tokens": 5},
                "tool_names": [],
                "failure": None,
            }
        )
    )
    assert settled == "{}"
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 1

    repeat = control.settle(
        json.dumps(
            {
                "request_id": admission["request_id"],
                "attempt_id": admission["attempt_id"],
                "outcome": "completed",
                "usage": None,
                "tool_names": [],
                "failure": None,
            }
        )
    )
    assert repeat == "{}"


def test_local_admit_persists_route_context_in_the_attempt(tmp_path: Path) -> None:
    """The default local composition retains durable route provenance."""
    control, raw_key = _control_plane(tmp_path)

    admission = _admit_started(control, raw_key, _chat_body())

    ledger = cast("SQLiteAttemptLedger", control._components.ledger)  # noqa: SLF001
    with sqlite3.connect(ledger.database_path) as connection:
        row = connection.execute(
            "select route_reason, fallback_reason from gateway_attempts where attempt_id = ?",
            (admission["attempt_id"],),
        ).fetchone()
    assert row == ("direct", None)


def test_admit_rejects_invalid_bodies_with_python_parity(tmp_path: Path) -> None:
    """Invalid JSON, non-objects, and unsupported protocol fields use shared codes."""
    control, raw_key = _control_plane(tmp_path)
    with pytest.raises(NativeBridgeError) as invalid_json:
        control.admit(json.dumps({"raw_key": raw_key, "body": "{not json"}))
    assert json.loads(invalid_json.value.public_error_json)["code"] == "invalid_json"
    with pytest.raises(NativeBridgeError) as not_object:
        control.admit(json.dumps({"raw_key": raw_key, "body": "[1, 2]"}))
    assert json.loads(not_object.value.public_error_json)["code"] == "invalid_request"
    rejected = json.dumps(
        {"model": "coding", "messages": [{"role": "user", "content": "x"}], "logit_bias": {}}
    )
    with pytest.raises(NativeBridgeError) as protocol:
        control.admit(json.dumps({"raw_key": raw_key, "body": rejected}))
    assert json.loads(protocol.value.public_error_json)["status_code"] == 400
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 0


def test_failed_settlement_keeps_the_attempt_retryable(tmp_path: Path) -> None:
    """A lost terminal write latches readiness but stays settleable on retry."""
    control, raw_key = _control_plane(tmp_path)
    admission = _admit_started(control, raw_key, _chat_body())
    settlement = json.dumps(
        {
            "request_id": admission["request_id"],
            "attempt_id": admission["attempt_id"],
            "outcome": "completed",
            "usage": {"input_tokens": 3, "output_tokens": 2},
            "tool_names": [],
            "failure": None,
        }
    )
    ledger = control._components.ledger  # noqa: SLF001 - fault injection for the test.
    with mock.patch.object(
        ledger,
        "apply_finish_attempt",
        side_effect=RuntimeError("simulated terminal write loss"),
    ):
        with pytest.raises(NativeBridgeError):
            control.settle(settlement)
    # A transient failure does not latch readiness; the data plane retries.
    assert control.readiness("{}") == "true"
    assert control.settle(settlement) == "{}"
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 1


def test_bridge_writes_route_through_the_group_commit_writer(tmp_path: Path) -> None:
    """Admission and settlement never use the raw ledger's one-fsync-per-write
    methods; every bridge write reaches SQLite through the shared batching
    writer and still lands in the usage report."""
    control, raw_key = _control_plane(tmp_path)
    ledger = control._components.ledger  # noqa: SLF001 - wiring assertion for the test.
    direct_writes = mock.Mock(side_effect=AssertionError("bridge used the raw per-write path"))
    with (
        mock.patch.object(ledger, "accept_request", direct_writes),
        mock.patch.object(ledger, "start_attempt", direct_writes),
        mock.patch.object(ledger, "finish_attempt", direct_writes),
        mock.patch.object(ledger, "finish_request", direct_writes),
    ):
        admission = _admit_started(control, raw_key, _chat_body())
        assert (
            control.settle(
                json.dumps(
                    {
                        "request_id": admission["request_id"],
                        "attempt_id": admission["attempt_id"],
                        "outcome": "completed",
                        "usage": {"input_tokens": 7, "output_tokens": 2},
                        "tool_names": [],
                        "failure": None,
                    }
                )
            )
            == "{}"
        )
    direct_writes.assert_not_called()
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 1
    assert report["totals"]["input_tokens"] == 7


def test_closed_writer_makes_settle_raise_and_keeps_the_entry_retryable(
    tmp_path: Path,
) -> None:
    """A settle against a closed group-commit writer raises the sanitized
    boundary error and retains the exact settlement for later replay."""
    control, raw_key = _control_plane(tmp_path)
    admission = _admit_started(control, raw_key, _chat_body())
    writer = control._components.write_ledger  # noqa: SLF001 - shutdown-ordering fault injection.
    assert writer is not None
    writer.close()
    settlement = json.dumps(
        {
            "request_id": admission["request_id"],
            "attempt_id": admission["attempt_id"],
            "outcome": "completed",
            "usage": {"input_tokens": 4, "output_tokens": 1},
            "tool_names": [],
            "failure": None,
        }
    )
    with pytest.raises(NativeBridgeError):
        control.settle(settlement)
    entry = control._accounting.entry(str(admission["request_id"]))  # noqa: SLF001
    assert entry is not None
    assert entry.pending_settlement is not None
    assert entry.pending_settlement["outcome"] == "completed"


def test_sweep_replays_the_original_completed_settlement(tmp_path: Path) -> None:
    """A retained settlement lands its completed outcome and usage, never a
    downgraded cancellation."""
    control, raw_key = _control_plane(tmp_path)
    admission = _admit_started(control, raw_key, _chat_body())
    settlement = json.dumps(
        {
            "request_id": admission["request_id"],
            "attempt_id": admission["attempt_id"],
            "outcome": "completed",
            "usage": {"input_tokens": 9, "output_tokens": 4},
            "tool_names": [],
            "failure": None,
        }
    )
    ledger = control._components.ledger  # noqa: SLF001 - fault injection for the test.
    with mock.patch.object(
        ledger,
        "apply_finish_attempt",
        side_effect=RuntimeError("simulated terminal write loss"),
    ):
        with pytest.raises(NativeBridgeError):
            control.settle(settlement)
    control._accounting.sweep_expired()  # noqa: SLF001 - the timer normally drives this.
    assert control._accounting.entry(str(admission["request_id"])) is None  # noqa: SLF001
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 1
    assert report["totals"]["input_tokens"] == 9
    assert report["totals"]["terminal_counts"] == [{"state": "completed", "attempts": 1}]


def test_abandoned_inflight_attempts_are_swept_after_the_deadline(tmp_path: Path) -> None:
    """An admitted request the data plane never settles is closed by the sweep."""
    control, raw_key = _control_plane(tmp_path, request_timeout_seconds=0.01)
    abandoned = _admit_started(control, raw_key, _chat_body())
    time.sleep(0.05)
    with mock.patch("exp.runtime.gateway.native_accounting._SWEEP_GRACE_SECONDS", 0.0):
        second = _admit(control, raw_key, _chat_body())
    assert control._accounting.entry(str(abandoned["request_id"])) is None  # noqa: SLF001
    assert control._accounting.entry(str(second["request_id"])) is not None  # noqa: SLF001
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 2


@pytest.mark.parametrize(
    "auth_mode",
    ("ambient_pair", "ambient_bearer", "explicit_bearer", "explicit_pair"),
)
def test_admit_serves_bedrock_natively_with_a_signed_frozen_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    auth_mode: str,
) -> None:
    """A Bedrock alias admits natively: the wire config carries the exact
    pre-serialized Converse body and SigV4 headers computed over it."""
    from exp.common.models import (
        GatewayDeploymentCapabilities,
        GatewayTokenPrices,
        ModelCapabilities,
    )
    from exp.runtime.gateway.catalog_authority import (
        ConnectionConfig,
        upsert_connection,
        upsert_singleton_deployment,
    )

    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    environment = {"TEST_PROVIDER_KEY": "provider-secret-canary"}
    api_key_env = None
    aws_access_key_id_env = None
    bedrock_auth_mode: Literal["access_key_pair", "api_key"] | None = None
    if auth_mode == "ambient_pair":
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "sigv4-secret-canary")
        monkeypatch.setenv("AWS_SESSION_TOKEN", "session-token-canary")
    elif auth_mode == "ambient_bearer":
        environment["AWS_BEARER_TOKEN_BEDROCK"] = "ambient-bearer-canary"
    elif auth_mode == "explicit_bearer":
        environment.update(
            {
                "AWS_BEARER_TOKEN_BEDROCK": "ambient-bearer-must-not-win",
                "BEDROCK_API_KEY": "explicit-bearer-canary",
            }
        )
        api_key_env = "BEDROCK_API_KEY"
        bedrock_auth_mode = "api_key"
    else:
        environment.update(
            {
                "AWS_BEARER_TOKEN_BEDROCK": "ambient-bearer-must-not-win",
                "BEDROCK_ACCESS_KEY_ID": "AKIAEXPLICITKEY001",
                "BEDROCK_SECRET_ACCESS_KEY": "explicit-secret-canary",
            }
        )
        api_key_env = "BEDROCK_SECRET_ACCESS_KEY"
        aws_access_key_id_env = "BEDROCK_ACCESS_KEY_ID"
    manager, raw_key = _configured_gateway(tmp_path)
    upsert_connection(
        tmp_path,
        name="bedrock-main",
        connection=ConnectionConfig(
            provider="bedrock",
            region="us-east-1",
            api_key_env=api_key_env,
            aws_access_key_id_env=aws_access_key_id_env,
            bedrock_auth_mode=bedrock_auth_mode,
        ),
        replace=False,
    )
    normalized, snapshot, _changed = upsert_singleton_deployment(
        tmp_path,
        deployment_alias="bed",
        connection_name="bedrock-main",
        provider_model="us.anthropic.claude-sonnet-4-5",
        exact_model_id="bedrock-revision-exact",
        revision=None,
        capabilities=ModelCapabilities(
            supports_tools=True,
            supports_structured_output=True,
        ),
        gateway_capabilities=GatewayDeploymentCapabilities(
            supports_streaming=True,
            supports_streaming_tool_arguments=True,
            supports_stop_sequences=True,
            supports_strict_tools=True,
            supports_structured_text=True,
        ),
        prices=GatewayTokenPrices(),
        pricing_source=None,
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id="bed",
        alias_name="bed",
        revision_id="revision-bed",
        pool_id="bed",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.add_grant(identity_id="default", alias_id="bed")
    components = load_gateway_components(
        tmp_path,
        environment=environment,
    )
    control = NativeControlPlane(components)
    body_schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    admission = _admit(
        control,
        raw_key,
        json.dumps(
            {
                "model": "bed",
                "messages": [{"role": "user", "content": "hi"}],
                "stop": ["DONE"],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "Find a record.",
                            "parameters": {"type": "object"},
                            "strict": True,
                        },
                    }
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "description": "Return one answer.",
                        "schema": body_schema,
                        "strict": True,
                    },
                },
            }
        ),
    )
    assert "escalate" not in admission
    # The route entry, not admission itself, carries the per-deployment wire
    # fields; the data plane reserves the physical dispatch through
    # ``start_attempt`` before it dials this deployment.
    started = _flatten_started(control, admission)
    assert started["dialect"] == "bedrock_converse_stream"
    assert started["url"] == (
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/"
        "us.anthropic.claude-sonnet-4-5/converse-stream"
    )
    body = started["upstream_body"]
    assert isinstance(body, str)
    # The frozen body is the only body channel for a signed dispatch; the
    # structured payload is not shipped twice across the boundary.
    assert started["upstream_payload"] is None
    payload = json.loads(body)
    assert body == json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    assert "modelId" not in payload
    assert payload["messages"] == [{"role": "user", "content": [{"text": "hi"}]}]
    assert payload["inferenceConfig"] == {"stopSequences": ["DONE"]}
    assert payload["toolConfig"]["tools"][0]["toolSpec"]["strict"] is True
    assert payload["outputConfig"] == {
        "textFormat": {
            "type": "json_schema",
            "structure": {
                "jsonSchema": {
                    "schema": '{"properties":{"answer":{"type":"string"}},"type":"object"}',
                    "name": "answer",
                    "description": "Return one answer.",
                }
            },
        }
    }
    # The route entry carries no signature: the data plane signs at dispatch
    # time, after its bounded permit, so queue wait cannot age the signature.
    assert started["headers"] == {}
    signed = json.loads(
        control.sign_dispatch(
            json.dumps({"request_id": started["request_id"], "url": started["url"], "body": body})
        )
    )
    headers = signed["headers"]
    authorization = headers.get("Authorization") or headers.get("authorization")
    assert isinstance(authorization, str)
    if auth_mode in {"ambient_bearer", "explicit_bearer"}:
        expected = (
            "ambient-bearer-canary" if auth_mode == "ambient_bearer" else "explicit-bearer-canary"
        )
        assert authorization == f"Bearer {expected}"
    else:
        expected_key = "AKIDEXAMPLE" if auth_mode == "ambient_pair" else "AKIAEXPLICITKEY001"
        assert authorization.startswith(f"AWS4-HMAC-SHA256 Credential={expected_key}/")
        assert "/us-east-1/bedrock/aws4_request" in authorization
        if auth_mode == "ambient_pair":
            assert headers["X-Amz-Security-Token"] == "session-token-canary"
    for secret in (
        "sigv4-secret-canary",
        "ambient-bearer-canary",
        "ambient-bearer-must-not-win",
        "explicit-bearer-canary",
        "explicit-secret-canary",
    ):
        assert secret not in json.dumps(admission)
    signed_json = json.dumps(signed)
    if auth_mode in {"ambient_bearer", "explicit_bearer"}:
        assert "ambient-bearer-must-not-win" not in signed_json
        assert "explicit-secret-canary" not in signed_json
    else:
        for secret in (
            "sigv4-secret-canary",
            "ambient-bearer-must-not-win",
            "explicit-secret-canary",
        ):
            assert secret not in signed_json
    with pytest.raises(NativeBridgeError):
        control.sign_dispatch(
            json.dumps(
                {
                    "request_id": started["request_id"],
                    "url": "https://untrusted.example/collect",
                    "body": body,
                }
            )
        )
    with pytest.raises(NativeBridgeError):
        control.sign_dispatch(
            json.dumps(
                {
                    "request_id": started["request_id"],
                    "url": started["url"],
                    "body": body + " ",
                }
            )
        )
    # Attempts without a retained signer fail closed and sanitized.
    with pytest.raises(NativeBridgeError):
        control.sign_dispatch(
            json.dumps({"request_id": "unknown", "url": started["url"], "body": body})
        )
    # The attempt is durably started, exactly like other native admissions.
    settled = control.settle(
        json.dumps(
            {
                "request_id": started["request_id"],
                "attempt_id": started["attempt_id"],
                "outcome": "completed",
                "usage": {"input_tokens": 3, "output_tokens": 2},
                "tool_names": [],
                "failure": None,
                "finalize": True,
                "opened": True,
            }
        )
    )
    assert settled == "{}"
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 1


def test_admit_serves_gemini_stop_and_schema_on_the_native_wire(tmp_path: Path) -> None:
    """Rust admission receives Gemini's exact stop and structured-output payload."""
    from exp.common.models import GatewayTokenPrices
    from exp.runtime.gateway.catalog_authority import (
        ConnectionConfig,
        upsert_connection,
        upsert_singleton_deployment,
    )

    manager, raw_key = _configured_gateway(tmp_path)
    upsert_connection(
        tmp_path,
        name="gemini-main",
        connection=ConnectionConfig(provider="gemini", api_key_env="GEMINI_TEST_KEY"),
        replace=False,
    )
    normalized, snapshot, _changed = upsert_singleton_deployment(
        tmp_path,
        deployment_alias="gem",
        connection_name="gemini-main",
        provider_model="gemini-2.5-pro",
        exact_model_id="gemini-revision-exact",
        revision=None,
        capabilities=ModelCapabilities(supports_structured_output=True),
        gateway_capabilities=GatewayDeploymentCapabilities(
            supports_streaming=True,
            supports_stop_sequences=True,
            supports_structured_text=True,
        ),
        prices=GatewayTokenPrices(),
        pricing_source=None,
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id="gem",
        alias_name="gem",
        revision_id="revision-gem",
        pool_id="gem",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.add_grant(identity_id="default", alias_id="gem")
    components = load_gateway_components(
        tmp_path,
        environment={
            "TEST_PROVIDER_KEY": "provider-secret-canary",
            "GEMINI_TEST_KEY": "gemini-secret-canary",
        },
    )
    control = NativeControlPlane(components)
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    admission = _admit_started(
        control,
        raw_key,
        json.dumps(
            {
                "model": "gem",
                "messages": [{"role": "user", "content": "hi"}],
                "stop": ["DONE"],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "schema": schema,
                        "strict": True,
                    },
                },
            }
        ),
    )

    assert admission["dialect"] == "gemini_generate_content"
    payload = admission["upstream_payload"]
    assert isinstance(payload, dict)
    generation = payload["generationConfig"]
    assert isinstance(generation, dict)
    assert generation["stopSequences"] == ["DONE"]
    assert generation["responseMimeType"] == "application/json"
    assert generation["responseJsonSchema"] == schema
    settled = control.settle(
        json.dumps(
            {
                "request_id": admission["request_id"],
                "attempt_id": admission["attempt_id"],
                "outcome": "completed",
                "usage": {"input_tokens": 3, "output_tokens": 2},
                "tool_names": [],
                "failure": None,
                "finalize": True,
                "opened": True,
            }
        )
    )
    assert settled == "{}"


def _configured_pool_gateway(
    root: Path,
    *,
    refusal_failover: bool = False,
    base_urls: tuple[str, str] = ("http://127.0.0.1:9/v1", "http://127.0.0.1:10/v1"),
    gateway_capabilities: tuple[GatewayDeploymentCapabilities, GatewayDeploymentCapabilities]
    | None = None,
    api_key_envs: tuple[str, str] = ("TEST_PROVIDER_KEY", "TEST_PROVIDER_KEY"),
    provider: str = "openai-compatible",
    provider_models: tuple[str, str] = ("alpha-model-exact", "beta-model-exact"),
    model_capabilities: tuple[ModelCapabilities, ModelCapabilities] | None = None,
) -> tuple[GatewayManagement, str]:
    """Create one certified two-deployment pool alias, grant, and key.

    Args:
        root: Gateway root to initialize.
        refusal_failover: Whether the alias revision opts into refusal failover.
        base_urls: One provider endpoint per ordered deployment.
        gateway_capabilities: Optional protocol contract for each deployment.
        model_capabilities: Optional semantic model contract for each deployment.
        api_key_envs: One credential environment-variable name per ordered
            deployment, so a test can make one rung's credential resolvable
            while another is absent.
        provider: Provider implementation shared by both pool deployments.
        provider_models: Exact provider model spelling for each deployment.
        model_capabilities: Optional exact model capabilities per deployment.

    Returns:
        The management handle and the issued raw key.
    """
    from datetime import UTC, datetime

    from exp.common.models import GatewayEquivalenceCertification
    from exp.runtime.gateway.catalog_authority import (
        ConnectionConfig,
        upsert_certified_pool,
        upsert_connection,
        upsert_singleton_deployment,
    )

    manager = GatewayManagement(root)
    manager.initialize()
    normalized = None
    snapshot = None
    declared_gateway_capabilities = gateway_capabilities or (
        GatewayDeploymentCapabilities(supports_streaming=True),
        GatewayDeploymentCapabilities(supports_streaming=True),
    )
    declared_model_capabilities = model_capabilities or (
        ModelCapabilities(),
        ModelCapabilities(),
    )
    for alias, base_url, gateway_capability, model_capability, api_key_env, provider_model in zip(
        ("alpha", "beta"),
        base_urls,
        declared_gateway_capabilities,
        declared_model_capabilities,
        api_key_envs,
        provider_models,
        strict=True,
    ):
        upsert_connection(
            root,
            name=f"{alias}-provider",
            connection=ConnectionConfig(
                provider=provider,
                base_url=base_url if provider == "openai-compatible" else None,
                api_key_env=api_key_env,
            ),
            replace=False,
        )
        normalized, snapshot, _changed = upsert_singleton_deployment(
            root,
            deployment_alias=alias,
            connection_name=f"{alias}-provider",
            provider_model=provider_model,
            exact_model_id="model-revision-exact",
            revision=None,
            capabilities=model_capability,
            gateway_capabilities=gateway_capability,
            prices=GatewayTokenPrices(),
            pricing_source=None,
            replace=False,
        )
    assert normalized is not None
    certification = GatewayEquivalenceCertification(
        certification_id="certification-pool",
        provenance="operator-reviewed deployment manifests",
        evidence_sha256="a" * 64,
        certified_at=datetime(2026, 8, 18, tzinfo=UTC),
    )
    normalized, snapshot, _changed = upsert_certified_pool(
        root,
        pool_id="coding",
        exact_model_id="model-revision-exact",
        deployment_aliases=("alpha", "beta"),
        certification=certification,
        expected_catalog_sha256=normalized.identity_sha256(),
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id="coding",
        alias_name="coding",
        revision_id="revision-pool-one",
        pool_id="coding",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
        refusal_failover=refusal_failover,
    )
    manager.create_identity(identity_id="default", display_name="Default")
    manager.add_grant(identity_id="default", alias_id="coding")
    issued = manager.issue_key(identity_id="default", key_id="key-one")
    return manager, issued.raw_key


def _pool_control_plane(
    root: Path,
    *,
    refusal_failover: bool = False,
    gateway_capabilities: tuple[GatewayDeploymentCapabilities, GatewayDeploymentCapabilities]
    | None = None,
    model_capabilities: tuple[ModelCapabilities, ModelCapabilities] | None = None,
) -> tuple[NativeControlPlane, str]:
    """Load the native control plane over one certified two-deployment pool."""
    _manager, raw_key = _configured_pool_gateway(
        root,
        refusal_failover=refusal_failover,
        gateway_capabilities=gateway_capabilities,
        model_capabilities=model_capabilities,
    )
    components = load_gateway_components(
        root,
        environment={"TEST_PROVIDER_KEY": "provider-secret-canary"},
    )
    return NativeControlPlane(components), raw_key


def test_admit_returns_the_full_ordered_route_without_starting_attempts(
    tmp_path: Path,
) -> None:
    """A certified pool admits natively with one wire entry per deployment.

    Admission accepts the request but writes no attempt row; each physical
    dispatch is reserved by ``start_attempt``, and the per-attempt rows carry
    the dispatch ordinal and deployment position.
    """
    control, raw_key = _pool_control_plane(tmp_path, refusal_failover=True)
    admission = _admit(control, raw_key, _chat_body())
    assert "escalate" not in admission
    assert admission["maximum_total_attempts"] == 8
    assert admission["maximum_same_deployment_attempts"] == 2
    assert admission["refusal_failover"] is True
    route = admission["route"]
    assert isinstance(route, list) and len(route) == 2
    first, second = route
    assert isinstance(first, dict) and isinstance(second, dict)
    assert first["model_id"] == "alpha-model-exact"
    assert second["model_id"] == "beta-model-exact"
    assert first["url"] != second["url"]
    assert first["idempotency_key"] != second["idempotency_key"]
    assert first["upstream_payload"] != second["upstream_payload"]

    ledger = cast("SQLiteAttemptLedger", control._components.ledger)  # noqa: SLF001
    with sqlite3.connect(ledger.database_path) as connection:
        assert connection.execute("select count(*) from gateway_attempts").fetchone() == (0,)

    started = _start_first(control, admission)
    assert started["route_depth"] == 0
    with sqlite3.connect(ledger.database_path) as connection:
        rows = connection.execute(
            "select attempt_ordinal, route_depth, state from gateway_attempts"
        ).fetchall()
    assert rows == [(0, 0, "dispatched")]


def test_admit_removes_protocol_incompatible_fallbacks(tmp_path: Path) -> None:
    """A supported stop request keeps the compatible rung instead of failing the pool."""
    control, raw_key = _pool_control_plane(
        tmp_path,
        gateway_capabilities=(
            GatewayDeploymentCapabilities(supports_streaming=True),
            GatewayDeploymentCapabilities(
                supports_streaming=True,
                supports_stop_sequences=True,
            ),
        ),
    )
    body = json.dumps(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "stop": ["DONE"],
        }
    )

    admission = _admit(control, raw_key, body)

    route = cast("list[JsonObject]", admission["route"])
    assert [item["model_id"] for item in route] == ["beta-model-exact"]
    upstream = cast("JsonObject", route[0]["upstream_payload"])
    assert upstream["stop"] == ["DONE"]


@pytest.mark.parametrize(
    ("body_update", "lead", "fallback", "model_capabilities"),
    (
        (
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {"type": "object"},
                            "strict": True,
                        },
                    }
                ]
            },
            GatewayDeploymentCapabilities(supports_streaming=True),
            GatewayDeploymentCapabilities(
                supports_streaming=True,
                supports_streaming_tool_arguments=True,
                supports_strict_tools=True,
            ),
            (ModelCapabilities(supports_tools=True), ModelCapabilities(supports_tools=True)),
        ),
        (
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "schema": {"type": "object"},
                        "strict": True,
                    },
                }
            },
            GatewayDeploymentCapabilities(supports_streaming=True),
            GatewayDeploymentCapabilities(
                supports_streaming=True,
                supports_structured_text=True,
            ),
            (
                ModelCapabilities(supports_structured_output=True),
                ModelCapabilities(supports_structured_output=True),
            ),
        ),
        (
            {"stream": True},
            GatewayDeploymentCapabilities(),
            GatewayDeploymentCapabilities(supports_streaming=True),
            (ModelCapabilities(), ModelCapabilities()),
        ),
        (
            {
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            },
            GatewayDeploymentCapabilities(supports_streaming=True),
            GatewayDeploymentCapabilities(
                supports_streaming=True,
                supports_streaming_tool_arguments=True,
            ),
            (ModelCapabilities(supports_tools=True), ModelCapabilities(supports_tools=True)),
        ),
    ),
)
def test_admit_filters_each_protocol_capability_before_selection(
    tmp_path: Path,
    body_update: JsonObject,
    lead: GatewayDeploymentCapabilities,
    fallback: GatewayDeploymentCapabilities,
    model_capabilities: tuple[ModelCapabilities, ModelCapabilities],
) -> None:
    """Tools, structured output, and streaming retain only a capable rung."""
    control, raw_key = _pool_control_plane(
        tmp_path,
        gateway_capabilities=(lead, fallback),
        model_capabilities=model_capabilities,
    )
    body: JsonObject = {
        "model": "coding",
        "messages": [{"role": "user", "content": "hi"}],
        **body_update,
    }

    admission = _admit(control, raw_key, json.dumps(body))

    route = cast("list[JsonObject]", admission["route"])
    assert [item["model_id"] for item in route] == ["beta-model-exact"]


def test_admit_returns_a_field_specific_400_when_no_rung_supports_tools(
    tmp_path: Path,
) -> None:
    """An all-incompatible tool route names strict_tools and never reports internal."""
    unsupported = GatewayDeploymentCapabilities(supports_streaming=True)
    control, raw_key = _pool_control_plane(
        tmp_path,
        gateway_capabilities=(unsupported, unsupported),
        model_capabilities=(
            ModelCapabilities(supports_tools=True),
            ModelCapabilities(supports_tools=True),
        ),
    )
    body = json.dumps(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object"},
                        "strict": True,
                    },
                }
            ],
        }
    )

    with pytest.raises(NativeBridgeError) as raised:
        _admit(control, raw_key, body)

    error = json.loads(raised.value.public_error_json)
    assert error["status_code"] == 400
    assert error["code"] == "unsupported_capability"
    assert error["param"] == "tools"
    assert "internal" not in error["code"]


def test_non_streaming_tool_transport_failure_names_tools(tmp_path: Path) -> None:
    """A buffered tool request identifies the feature that requires streaming."""
    unsupported = GatewayDeploymentCapabilities(supports_strict_tools=True)
    control, raw_key = _pool_control_plane(
        tmp_path,
        gateway_capabilities=(unsupported, unsupported),
        model_capabilities=(
            ModelCapabilities(supports_tools=True),
            ModelCapabilities(supports_tools=True),
        ),
    )
    body = json.dumps(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        }
    )

    with pytest.raises(NativeBridgeError) as raised:
        _admit(control, raw_key, body)

    error = json.loads(raised.value.public_error_json)
    assert error["code"] == "unsupported_capability"
    assert error["param"] == "tools"
    assert "Remove the field" in error["message"]


def test_admit_preserves_parameter_path_for_an_over_limit_stop_list(tmp_path: Path) -> None:
    """A parameter validator reaches the public 400 with its exact field path intact."""
    capabilities = GatewayDeploymentCapabilities(
        supports_streaming=True,
        supports_stop_sequences=True,
        maximum_stop_sequences=5,
    )
    control, raw_key = _pool_control_plane(
        tmp_path,
        gateway_capabilities=(capabilities, capabilities),
    )
    body = json.dumps(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "stop": ["a", "b", "c", "d", "e", "f"],
        }
    )

    with pytest.raises(NativeBridgeError) as raised:
        _admit(control, raw_key, body)

    error = json.loads(raised.value.public_error_json)
    assert error["status_code"] == 400
    assert error["code"] == "invalid_parameter"
    assert error["param"] == "stop"


@pytest.mark.parametrize(
    ("body_update", "gateway_capabilities", "model_capabilities", "param"),
    (
        (
            {"stop": ["DONE"]},
            GatewayDeploymentCapabilities(supports_streaming=True),
            ModelCapabilities(),
            "stop",
        ),
        (
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "schema": {"type": "object"},
                        "strict": True,
                    },
                }
            },
            GatewayDeploymentCapabilities(
                supports_streaming=True,
                supports_structured_text=True,
            ),
            ModelCapabilities(),
            "response_format",
        ),
    ),
)
def test_admit_scopes_each_public_capability_failure_to_its_request_field(
    tmp_path: Path,
    body_update: JsonObject,
    gateway_capabilities: GatewayDeploymentCapabilities,
    model_capabilities: ModelCapabilities,
    param: str,
) -> None:
    """Public admission failures never degrade to an unscoped internal error."""
    control, raw_key = _pool_control_plane(
        tmp_path,
        gateway_capabilities=(gateway_capabilities, gateway_capabilities),
        model_capabilities=(model_capabilities, model_capabilities),
    )
    body: JsonObject = {
        "model": "coding",
        "messages": [{"role": "user", "content": "hi"}],
        **body_update,
    }

    with pytest.raises(NativeBridgeError) as raised:
        _admit(control, raw_key, json.dumps(body))

    error = json.loads(raised.value.public_error_json)
    assert error["status_code"] == 400
    assert error["code"] == "unsupported_capability"
    assert error["param"] == param


def _partial_pool_control_plane(
    root: Path,
    environment: dict[str, str],
) -> tuple[NativeControlPlane, str]:
    """Load a two-rung pool whose lead reads a separately-toggled credential.

    The lead (``alpha``) reads ``TEST_ALPHA_KEY`` and the fallback (``beta``)
    reads ``TEST_PROVIDER_KEY``. Both must be present in ``environment`` at load
    so readiness admits the alias; a test then mutates the shared ``environment``
    mapping to make the lead dead or healthy at a later admission.
    """
    _manager, raw_key = _configured_pool_gateway(
        root,
        api_key_envs=("TEST_ALPHA_KEY", "TEST_PROVIDER_KEY"),
    )
    components = load_gateway_components(root, environment=environment)
    return NativeControlPlane(components), raw_key


def _openai_responses_pool_control_plane(
    root: Path,
    environment: dict[str, str],
) -> tuple[NativeControlPlane, str]:
    """Load two native Responses rungs with independently mutable credentials."""
    reasoning_capabilities = ModelCapabilities(
        supports_reasoning=True,
        supports_tools=True,
        supports_temperature=False,
    )
    _manager, raw_key = _configured_pool_gateway(
        root,
        api_key_envs=("TEST_ALPHA_KEY", "TEST_BETA_KEY"),
        provider="openai",
        provider_models=("gpt-5.6-sol", "gpt-5.6-sol"),
        model_capabilities=(reasoning_capabilities, reasoning_capabilities),
    )
    return NativeControlPlane(load_gateway_components(root, environment=environment)), raw_key


@pytest.mark.parametrize("rotated_beta", (None, "beta-secret-rotated"))
def test_encrypted_reasoning_pins_winning_fallback_and_rejects_credential_drift(
    tmp_path: Path,
    rotated_beta: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retained ciphertext can replay only on its exact winning authenticated wire."""
    environment = {"TEST_ALPHA_KEY": "alpha-secret", "TEST_BETA_KEY": "beta-secret"}
    control, raw_key = _openai_responses_pool_control_plane(tmp_path, environment)
    first = _admit_responses(control, raw_key, _responses_body())
    initial_route = cast("list[JsonObject]", first["route"])
    assert [wire["deployment_id"] for wire in initial_route] == ["alpha", "beta"]

    lead = _start_first(control, first)
    assert lead["route_depth"] == 0
    control.settle(
        json.dumps(
            {
                "request_id": first["request_id"],
                "attempt_id": lead["attempt_id"],
                "outcome": "failed",
                "usage": None,
                "tool_names": [],
                "failure": {
                    "failure_class": "provider_internal",
                    "safe_message": "provider service failed; retry after a short delay",
                },
                "finalize": False,
                "opened": False,
            }
        )
    )
    fallback = json.loads(
        control.start_attempt(
            json.dumps(
                {
                    "request_id": first["request_id"],
                    "attempt_ordinal": 1,
                    "current_depth": 0,
                    "failure": {
                        "failure_class": "provider_internal",
                        "safe_message": "provider service failed; retry after a short delay",
                        "retryable_same_deployment": False,
                        "failover_eligible": True,
                    },
                }
            )
        )
    )
    assert fallback["route_depth"] == 1
    entry = control._accounting.entry(_admitted_request_id(first))  # noqa: SLF001
    assert entry is not None and entry.continuation is not None
    namespace = entry.continuation.namespace
    control.remember(
        json.dumps(
            {
                "request_id": first["request_id"],
                "text": "",
                "refusal": False,
                "encrypted_reasoning": [
                    {
                        "output_index": 0,
                        "item_id": "rs-beta",
                        "encrypted_content": "beta-bound-ciphertext",
                        "status": "completed",
                    }
                ],
                "tool_calls": [],
            }
        )
    )
    response_id = stable_public_id("resp", _admitted_request_id(first))
    retained = control._continuations.resolve_now(  # noqa: SLF001
        namespace=namespace,
        previous_response_id=response_id,
    )
    assert retained.route_binding is not None
    assert retained.route_binding.deployment_id == "beta"
    assert "beta-secret" not in retained.model_dump_json()
    control.settle(
        json.dumps(
            {
                "request_id": first["request_id"],
                "attempt_id": fallback["attempt_id"],
                "outcome": "completed",
                "usage": {"input_tokens": 5, "output_tokens": 3},
                "tool_names": [],
                "failure": None,
                "finalize": True,
                "opened": True,
            }
        )
    )

    pinned = _admit_responses(
        control,
        raw_key,
        _responses_body(previous_response_id=response_id),
    )
    pinned_route = cast("list[JsonObject]", pinned["route"])
    assert [wire["deployment_id"] for wire in pinned_route] == ["beta"]

    if rotated_beta is None:
        del environment["TEST_BETA_KEY"]
    else:
        environment["TEST_BETA_KEY"] = rotated_beta
    assert environment["TEST_ALPHA_KEY"] == "alpha-secret"
    # Capture the durable failure the accepted request records: a continuation
    # that cannot bind is a client 400, so it must NOT be stamped internal (a
    # 5xx-class ledger row would page the internal-error alert for a caller
    # mistake).
    recorded: list[GatewayFailure] = []
    original_finish = control._accounting.finish_request_quietly  # noqa: SLF001

    def _capture_finish(authorization: AuthorizationSnapshot, failure: GatewayFailure) -> None:
        recorded.append(failure)
        return original_finish(authorization, failure)

    monkeypatch.setattr(control._accounting, "finish_request_quietly", _capture_finish)  # noqa: SLF001
    with pytest.raises(NativeBridgeError) as rejected:
        _admit_responses(
            control,
            raw_key,
            _responses_body(previous_response_id=response_id),
        )
    error = json.loads(rejected.value.public_error_json)
    assert error["status_code"] == 400
    assert error["code"] == "previous_response_not_found"
    assert error["param"] == "previous_response_id"
    assert recorded, "the unavailable continuation must record a durable failure"
    assert recorded[-1].failure_class == GatewayFailureClass.INVALID_REQUEST


def test_admission_maps_a_route_build_failure_to_a_retryable_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A route/catalog that cannot be built during a rolling deploy records a
    retryable UNAVAILABLE ledger failure and answers a retryable 503, never the
    paging INTERNAL that turned the last catalog-schema roll into a fleet-wide
    incident.
    """
    control, raw_key = _control_plane(tmp_path)

    def _raise_routing(*_args: object, **_kwargs: object) -> object:
        raise GatewayRoutingError("authorized catalog snapshot is not active for this revision")

    monkeypatch.setattr(control, "_resolve_route", _raise_routing)  # noqa: SLF001
    recorded: list[GatewayFailure] = []
    original_finish = control._accounting.finish_request_quietly  # noqa: SLF001

    def _capture(authorization: AuthorizationSnapshot, failure: GatewayFailure) -> None:
        recorded.append(failure)
        return original_finish(authorization, failure)

    monkeypatch.setattr(control._accounting, "finish_request_quietly", _capture)  # noqa: SLF001

    with pytest.raises(NativeBridgeError) as rejected:
        _admit(control, raw_key, _chat_body())

    error = json.loads(rejected.value.public_error_json)
    assert error["status_code"] == 503
    assert recorded, "the roll condition must record a durable failure"
    assert recorded[-1].failure_class == GatewayFailureClass.UNAVAILABLE
    assert recorded[-1].safe_message == "the gateway is updating; retry the request"


def test_admit_skips_a_dead_lead_rung_and_serves_the_fallback(tmp_path: Path) -> None:
    """A lead dead at admission fails over to the live fallback and feeds the circuit."""
    environment = {"TEST_ALPHA_KEY": "alpha-secret", "TEST_PROVIDER_KEY": "beta-secret"}
    control, raw_key = _partial_pool_control_plane(tmp_path, environment)

    # The lead credential is lost after load, so its rung is dead at admission
    # while the fallback stays healthy.
    del environment["TEST_ALPHA_KEY"]
    admission = _admit(control, raw_key, _chat_body())

    assert "escalate" not in admission
    route = cast("list[JsonObject]", admission["route"])
    assert [item["model_id"] for item in route] == ["beta-model-exact"]

    snapshot = control.metrics_snapshot()
    control_plane = cast("JsonObject", snapshot["control_plane"])
    assert control_plane["admission_lead_rungs_skipped"] == 1
    assert control_plane["admission_dead_rungs_skipped"] == 1

    # The skip fed the SAME deployment health circuit runtime failures feed:
    # exactly the dead lead's circuit is open (a cooldown, never a blacklist),
    # and admission never claims the fallback so no other circuit was touched.
    states = control._accounting.health._states  # noqa: SLF001 - assert the circuit state.
    assert len(states) == 1
    (dead_state,) = states.values()
    assert dead_state.open_until > time.monotonic()

    # The narrowed route is what serves and anchors accounting.
    started = _start_first(control, admission)
    assert started["route_depth"] == 0


def test_admit_recovers_a_dead_lead_when_its_credential_heals(tmp_path: Path) -> None:
    """A healed lead is served again on a later admission; the skip is never permanent."""
    environment = {"TEST_ALPHA_KEY": "alpha-secret", "TEST_PROVIDER_KEY": "beta-secret"}
    control, raw_key = _partial_pool_control_plane(tmp_path, environment)

    del environment["TEST_ALPHA_KEY"]
    narrowed = cast("list[JsonObject]", _admit(control, raw_key, _chat_body())["route"])
    assert [item["model_id"] for item in narrowed] == ["beta-model-exact"]

    # The credential returns; admission re-resolves the frozen route and the
    # lead is served again, proving the skip is not a permanent blacklist.
    environment["TEST_ALPHA_KEY"] = "alpha-secret"
    recovered = cast("list[JsonObject]", _admit(control, raw_key, _chat_body())["route"])
    assert [item["model_id"] for item in recovered] == ["alpha-model-exact", "beta-model-exact"]


def test_admit_escalates_when_every_rung_is_dead(tmp_path: Path) -> None:
    """A route with no resolvable rung is finished closed, not served."""
    environment = {"TEST_ALPHA_KEY": "alpha-secret", "TEST_PROVIDER_KEY": "beta-secret"}
    control, raw_key = _partial_pool_control_plane(tmp_path, environment)

    environment.clear()
    admission = _admit(control, raw_key, _chat_body())

    assert "escalate" in admission
    control_plane = cast("JsonObject", control.metrics_snapshot()["control_plane"])
    assert control_plane["admission_dead_rungs_skipped"] == 2
    # A total outage escalates on its own path; the lead-skip counter means
    # "a fallback served while the lead was dead" and must stay silent here.
    assert control_plane["admission_lead_rungs_skipped"] == 0


def test_admit_keeps_a_fully_resolvable_route_and_never_skips(tmp_path: Path) -> None:
    """A healthy route admits every rung; only resolve-time deadness narrows."""
    control, raw_key = _pool_control_plane(tmp_path)

    admission = _admit(control, raw_key, _chat_body())

    route = cast("list[JsonObject]", admission["route"])
    assert [item["model_id"] for item in route] == ["alpha-model-exact", "beta-model-exact"]
    control_plane = cast("JsonObject", control.metrics_snapshot()["control_plane"])
    assert control_plane["admission_dead_rungs_skipped"] == 0
    assert control_plane["admission_lead_rungs_skipped"] == 0


def test_pool_failover_records_ordinals_depths_and_one_finalize(tmp_path: Path) -> None:
    """A failed first deployment advances; the ledger shows both dispatches."""
    control, raw_key = _pool_control_plane(tmp_path)
    admission = _admit(control, raw_key, _chat_body())
    first = _start_first(control, admission)
    assert first["route_depth"] == 0
    assert (
        control.settle(
            json.dumps(
                {
                    "request_id": admission["request_id"],
                    "attempt_id": first["attempt_id"],
                    "outcome": "failed",
                    "usage": None,
                    "tool_names": [],
                    "failure": {
                        "failure_class": "provider_internal",
                        "safe_message": "provider service failed; retry after a short delay",
                    },
                    "finalize": False,
                    "opened": False,
                }
            )
        )
        == "{}"
    )
    second = json.loads(
        control.start_attempt(
            json.dumps(
                {
                    "request_id": admission["request_id"],
                    "attempt_ordinal": 1,
                    "current_depth": 0,
                    "failure": {
                        "failure_class": "provider_internal",
                        "safe_message": "provider service failed; retry after a short delay",
                        "retryable_same_deployment": False,
                        "failover_eligible": True,
                    },
                }
            )
        )
    )
    assert second["route_depth"] == 1
    assert (
        control.settle(
            json.dumps(
                {
                    "request_id": admission["request_id"],
                    "attempt_id": second["attempt_id"],
                    "outcome": "completed",
                    "usage": {"input_tokens": 5, "output_tokens": 3},
                    "tool_names": [],
                    "failure": None,
                    "finalize": True,
                    "opened": True,
                }
            )
        )
        == "{}"
    )
    ledger = cast("SQLiteAttemptLedger", control._components.ledger)  # noqa: SLF001
    with sqlite3.connect(ledger.database_path) as connection:
        rows = connection.execute(
            "select attempt_ordinal, route_depth, state"
            " from gateway_attempts order by attempt_ordinal"
        ).fetchall()
    assert rows == [(0, 0, "failed"), (1, 1, "completed")]
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 1
    assert report["totals"]["input_tokens"] == 5


def test_pool_claim_scope_resolves_natively(tmp_path: Path) -> None:
    """A keyed request against a certified pool claims a native replay scope."""
    control, raw_key = _pool_control_plane(tmp_path)
    scope = _claim_scope(control, raw_key, _chat_body(), idempotency_key="pool-op")
    assert "escalate" not in scope
    assert scope["surface"] == "chat_completions"


def test_claim_scope_matches_the_python_replay_key(tmp_path: Path) -> None:
    """The scope carries the same hashed caller operation and canonical digest
    the shared decoder computes for the replay key."""
    control, raw_key = _control_plane(tmp_path)
    scope = _claim_scope(control, raw_key, _chat_body(), idempotency_key="operation-one")
    assert scope["surface"] == "chat_completions"
    assert scope["caller_operation_sha256"] == hashlib.sha256(b"operation-one").hexdigest()
    repeat = _claim_scope(control, raw_key, _chat_body(), idempotency_key="operation-one")
    assert repeat == scope
    # X-Client-Request-Id is session correlation identity, never an
    # operation key: a scope claim without an Idempotency-Key fails closed.
    with pytest.raises(NativeBridgeError) as unkeyed:
        _claim_scope(
            control,
            raw_key,
            _chat_body(),
            client_request_id="operation-one",
        )
    payload = json.loads(unkeyed.value.public_error_json)
    assert payload["status_code"] == 400
    assert payload["param"] == "Idempotency-Key"
    different_body = _claim_scope(
        control,
        raw_key,
        _chat_body(stream=True),
        idempotency_key="operation-two",
    )
    assert different_body["canonical_request_sha256"] != scope["canonical_request_sha256"]
    decoded = decode_chat(json.loads(_chat_body()), idempotency_key="operation-one")
    assert decoded.request.idempotency_key == "operation-one"


def test_admit_escalates_host_ineligible_route_and_finalizes_the_request(tmp_path: Path) -> None:
    """Hosted policy escalation finalizes the accepted request content-free."""
    _manager, raw_key = _configured_gateway(tmp_path)
    components = load_gateway_components(
        tmp_path,
        environment={"TEST_PROVIDER_KEY": "provider-secret-canary"},
    )
    seen_requests: list[GatewayRequest] = []

    def native_route_eligible(_route: object, request: GatewayRequest) -> bool:
        """Reject keyed requests after retaining the decoded request for assertion."""
        seen_requests.append(request)
        return request.idempotency_key is None

    control = NativeControlPlane(components, native_route_eligible=native_route_eligible)

    admission = _admit(control, raw_key, _chat_body(), idempotency_key="shared-replay")

    assert admission == {"escalate": "host policy does not permit native execution of this route"}
    assert [request.idempotency_key for request in seen_requests] == ["shared-replay"]
    # The accepted request is finalized failed with no attempt row, so the
    # escalated admission is accounted content-free and never billed.
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 1
    assert report["totals"]["attempts"] == 0
    assert report["totals"]["known_estimated_cost_micro_usd"] == 0


def test_claim_scope_supports_the_responses_surface(tmp_path: Path) -> None:
    """A keyed Responses request claims a surface-scoped native replay key."""
    control, raw_key = _control_plane(tmp_path)
    body = json.dumps({"model": "coding", "input": "keyed responses"})
    scope = _claim_scope(
        control,
        raw_key,
        body,
        surface="responses",
        idempotency_key="responses-op",
    )
    assert "escalate" not in scope
    assert scope["surface"] == "responses"
    assert scope["caller_operation_sha256"] == hashlib.sha256(b"responses-op").hexdigest()
    # The surface is part of the key, so a chat operation with the same
    # caller operation never collides with the Responses scope.
    chat_scope = _claim_scope(control, raw_key, _chat_body(), idempotency_key="responses-op")
    assert chat_scope["surface"] == "chat_completions"


def test_claim_scope_validates_headers_with_python_parity(tmp_path: Path) -> None:
    """Header validation failures map to the exact shared protocol errors."""
    control, raw_key = _control_plane(tmp_path)
    for bad_value in ("", "x" * 513, "line\nbreak"):
        with pytest.raises(NativeBridgeError) as invalid:
            _claim_scope(control, raw_key, _chat_body(), idempotency_key=bad_value)
        payload = json.loads(invalid.value.public_error_json)
        assert payload["status_code"] == 400
        with pytest.raises(OpenAIProtocolError) as expected:
            decode_chat(json.loads(_chat_body()), idempotency_key=bad_value)
        assert payload["code"] == expected.value.detail.code
        assert payload["message"] == expected.value.detail.message
    # The two headers name independent concepts (operation vs session), so
    # divergent values are legal and the operation key alone scopes replay.
    divergent = _claim_scope(
        control,
        raw_key,
        _chat_body(),
        idempotency_key="one",
        client_request_id="two",
    )
    assert divergent["caller_operation_sha256"] == hashlib.sha256(b"one").hexdigest()


def test_keyed_admissions_enforce_the_durable_ledger_idempotency_rows(
    tmp_path: Path,
) -> None:
    """With the process-local replay store empty (as after a restart), the
    durable ledger fails a repeated caller operation closed exactly as the
    shared ledger enforces: a different body conflicts and an identical body
    reports the replay unavailable, never a second provider dispatch."""
    control, raw_key = _control_plane(tmp_path)
    admission = _admit_started(control, raw_key, _chat_body(), idempotency_key="durable-op")
    control.settle(
        json.dumps(
            {
                "request_id": admission["request_id"],
                "attempt_id": admission["attempt_id"],
                "outcome": "completed",
                "usage": {"input_tokens": 3, "output_tokens": 1},
                "tool_names": [],
                "failure": None,
            }
        )
    )
    with pytest.raises(NativeBridgeError) as conflict:
        _admit(control, raw_key, _chat_body(stream=True), idempotency_key="durable-op")
    conflict_payload = json.loads(conflict.value.public_error_json)
    assert conflict_payload["status_code"] == 409
    assert conflict_payload["code"] == "idempotency_conflict"
    with pytest.raises(NativeBridgeError) as unavailable:
        _admit(control, raw_key, _chat_body(), idempotency_key="durable-op")
    unavailable_payload = json.loads(unavailable.value.public_error_json)
    assert unavailable_payload["status_code"] == 409
    assert unavailable_payload["code"] == "idempotency_replay_unavailable"
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 1


def test_admit_rejects_an_ungranted_alias(tmp_path: Path) -> None:
    """An ungranted alias maps to the shared 403 public error."""
    control, raw_key = _control_plane(tmp_path)
    with pytest.raises(NativeBridgeError) as excinfo:
        control.admit(json.dumps({"raw_key": raw_key, "body": _chat_body(model="ungranted")}))
    payload = json.loads(excinfo.value.public_error_json)
    assert payload["status_code"] == 403
    assert payload["code"] == "model_not_granted"


def test_budget_rejection_uses_host_error_factory(tmp_path: Path) -> None:
    """Hosted policy can refine a quota rejection without moving accounting."""
    control, raw_key = _control_plane(tmp_path)
    customized = NativeBridgeError(
        OpenAIProtocolError(
            status_code=429,
            code="email_unverified",
            message="verify your email",
            error_type="insufficient_quota",
        )
    )
    control._accounting._budget_error_factory = lambda key: customized  # noqa: SLF001
    admission = _admit(control, raw_key, _chat_body())
    with mock.patch.object(
        control._write_ledger,  # noqa: SLF001
        "start_attempt",
        side_effect=BudgetReservationRejected(scope_kind=BudgetScopeKind.TEAM, reason="blocked"),
    ):
        with pytest.raises(NativeBridgeError) as excinfo:
            _start_first(control, admission)
    assert excinfo.value is customized
    # The rejected request is finalized with no attempt row remaining open.
    assert control._accounting.entry(str(admission["request_id"])) is None  # noqa: SLF001


def test_authenticate_rejects_an_invalid_key(tmp_path: Path) -> None:
    """A bad virtual key maps to the shared 401 public error."""
    control, _raw_key = _control_plane(tmp_path)
    with pytest.raises(NativeBridgeError) as excinfo:
        control.authenticate(json.dumps({"raw_key": "exp_vk_invalid"}))
    payload = json.loads(excinfo.value.public_error_json)
    assert payload["status_code"] == 401
    assert payload["code"] == "invalid_key"


def test_models_and_detail_are_exact_openai_discovery_bodies(tmp_path: Path) -> None:
    """Model discovery emits no gateway-specific response fields."""
    control, raw_key = _control_plane(tmp_path)

    models = json.loads(control.models(json.dumps({"raw_key": raw_key})))
    listed = {
        "id": "coding",
        "object": "model",
        "created": 0,
        "owned_by": "exp",
    }
    assert models == {"object": "list", "data": [listed]}
    detail = json.loads(
        control.model_detail(json.dumps({"raw_key": raw_key, "model_id": "coding"}))
    )
    assert detail == listed
    with pytest.raises(NativeBridgeError) as excinfo:
        control.model_detail(json.dumps({"raw_key": raw_key, "model_id": "missing"}))
    assert json.loads(excinfo.value.public_error_json)["status_code"] == 404


def test_readiness_reflects_startup_proof_and_composition_health(tmp_path: Path) -> None:
    """Readiness is true after load and follows the composition health surface.

    The local composition reports its group-commit writer's liveness, so a
    closed (or crashed) writer fails readiness closed before any settlement
    write is even attempted.
    """
    control, _raw_key = _control_plane(tmp_path)
    assert control.readiness("{}") == "true"
    components = cast("LocalGatewayComponents", control._components)  # noqa: SLF001 - fault injection.
    components.write_ledger.close()
    assert not components.accounting_healthy
    assert control.readiness("{}") == "false"


def test_metrics_snapshot_reports_control_plane_state_without_a_data_plane(
    tmp_path: Path,
) -> None:
    """Without an injected provider the snapshot still carries bridge counters."""
    control, raw_key = _control_plane(tmp_path)
    _admit(control, raw_key, _chat_body())

    snapshot = json.loads(control.metrics_json("{}"))

    assert snapshot["data_plane"] is None
    control_plane = snapshot["control_plane"]
    assert control_plane["sweep_retained_settlements_replayed"] == 0
    assert control_plane["sweep_abandoned_attempts_cancelled"] == 0
    assert control_plane["admission_dead_rungs_skipped"] == 0
    assert control_plane["admission_lead_rungs_skipped"] == 0
    assert control_plane["inflight_attempts"] == 1
    assert control_plane["reconciled_expired_requests"] == 0
    assert control_plane["reconciled_unknown_attempts"] == 0
    assert control_plane["accounting_healthy"] is True


def test_metrics_snapshot_merges_the_injected_data_plane_registry(tmp_path: Path) -> None:
    """The injected native snapshot appears verbatim under ``data_plane``."""
    _manager, _raw_key = _configured_gateway(tmp_path)
    components = load_gateway_components(
        tmp_path,
        environment={"TEST_PROVIDER_KEY": "provider-secret-canary"},
    )
    control = NativeControlPlane(
        components,
        data_plane_metrics=lambda: '{"served_requests": 3}',
    )

    snapshot = control.metrics_snapshot()

    assert snapshot["data_plane"] == {"served_requests": 3}


def test_metrics_snapshot_counts_a_replayed_retained_settlement(tmp_path: Path) -> None:
    """A sweep that lands a retained settlement moves the recovery counter."""
    control, raw_key = _control_plane(tmp_path)
    admission = _admit_started(control, raw_key, _chat_body())
    settlement = json.dumps(
        {
            "request_id": admission["request_id"],
            "attempt_id": admission["attempt_id"],
            "outcome": "completed",
            "usage": {"input_tokens": 9, "output_tokens": 4},
            "tool_names": [],
            "failure": None,
        }
    )
    ledger = control._components.ledger  # noqa: SLF001 - fault injection for the test.
    with mock.patch.object(
        ledger,
        "apply_finish_attempt",
        side_effect=RuntimeError("simulated terminal write loss"),
    ):
        with pytest.raises(NativeBridgeError):
            control.settle(settlement)
    control._accounting.sweep_expired()  # noqa: SLF001 - the timer normally drives this.

    snapshot = control.metrics_snapshot()

    control_plane = snapshot["control_plane"]
    assert isinstance(control_plane, dict)
    assert control_plane["sweep_retained_settlements_replayed"] == 1
    assert control_plane["sweep_abandoned_attempts_cancelled"] == 0
    assert control_plane["inflight_attempts"] == 0


def test_metrics_snapshot_counts_a_swept_abandoned_attempt(tmp_path: Path) -> None:
    """An abandoned admission closed by the sweep moves the cancellation counter."""
    control, raw_key = _control_plane(tmp_path, request_timeout_seconds=0.01)
    _admit(control, raw_key, _chat_body())
    time.sleep(0.05)
    with mock.patch("exp.runtime.gateway.native_accounting._SWEEP_GRACE_SECONDS", 0.0):
        control._accounting.sweep_expired()  # noqa: SLF001 - the timer normally drives this.

    snapshot = control.metrics_snapshot()

    control_plane = snapshot["control_plane"]
    assert isinstance(control_plane, dict)
    assert control_plane["sweep_abandoned_attempts_cancelled"] == 1
    assert control_plane["sweep_retained_settlements_replayed"] == 0
    assert control_plane["inflight_attempts"] == 0


def test_readiness_uses_host_lifecycle_probe_and_fails_closed(tmp_path: Path) -> None:
    """A hosted composition can add database, catalog, and drain readiness."""
    _manager, _raw_key = _configured_gateway(tmp_path)
    components = load_gateway_components(
        tmp_path,
        environment={"TEST_PROVIDER_KEY": "provider-secret-canary"},
    )
    ready = True
    control = NativeControlPlane(components, readiness_probe=lambda: ready)
    assert control.readiness("{}") == "true"
    ready = False
    assert control.readiness("{}") == "false"
    failed = NativeControlPlane(
        components,
        readiness_probe=mock.Mock(side_effect=RuntimeError("database unavailable")),
    )
    assert failed.readiness("{}") == "false"


def test_rust_failure_taxonomy_matches_public_failure_error() -> None:
    """The Rust failure-to-public-error table equals `public_failure_error`.

    Quota exhaustion is exempt from message and retry-after comparison: its
    reset boundary is computed control-plane side and never crosses the
    bridge as a Rust failure.
    """
    native = pytest.importorskip("exp_gateway_native")
    for failure_class in GatewayFailureClass:
        failure = GatewayFailure(
            failure_class=failure_class,
            safe_message="parity probe message",
        )
        expected = public_failure_error(failure)
        actual = json.loads(
            native.failure_public_error_fixture(failure_class.value, "parity probe message")
        )
        assert actual["status_code"] == expected.status_code, failure_class
        assert actual["code"] == expected.detail.code, failure_class
        assert actual["error_type"] == expected.detail.type, failure_class
        if failure_class != GatewayFailureClass.QUOTA_EXCEEDED:
            assert actual["message"] == expected.detail.message, failure_class
            assert actual["retry_after_seconds"] == expected.retry_after_seconds, failure_class


def test_rust_chat_sse_frames_match_python_and_the_committed_golden() -> None:
    """Rust Chat SSE frames equal the committed golden and the shared encoder.

    The committed golden frames are the durable contract; the python
    ``ChatSseEncoder`` remains a live secondary reference because the Pi
    bridge still serves through it.
    """
    native = pytest.importorskip("exp_gateway_native")
    from exp.common.models.model import ToolCall
    from exp.runtime.gateway.contracts import (
        GatewayEvent,
        GatewayEventKind,
        GatewayUsage,
    )
    from exp.runtime.openai_protocol.streaming import ChatSseEncoder

    events = [
        GatewayEvent(kind=GatewayEventKind.TEXT_DELTA, sequence_number=0, text_delta="Hel"),
        GatewayEvent(kind=GatewayEventKind.TEXT_DELTA, sequence_number=1, text_delta="lo é"),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_CALL_STARTED,
            sequence_number=2,
            tool_call_index=0,
            tool_call_id="call-1",
            tool_name="search",
        ),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_ARGUMENTS_DELTA,
            sequence_number=3,
            tool_call_index=0,
            raw_arguments_delta='{"q": "x"}',
        ),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_CALL_COMPLETED,
            sequence_number=4,
            tool_call_index=0,
            tool_call=ToolCall(
                call_id="call-1",
                name="search",
                arguments={"q": "x"},
                raw_arguments='{"q": "x"}',
            ),
        ),
        GatewayEvent(
            kind=GatewayEventKind.USAGE,
            sequence_number=5,
            usage=GatewayUsage(input_tokens=10, output_tokens=4, cached_input_tokens=1),
        ),
        GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=6),
    ]
    encoder = ChatSseEncoder(
        request_id="request-abc",
        model="coding",
        created_at=1_700_000_000,
        include_usage=True,
    )
    expected = list(encoder.start())
    for event in events:
        expected.extend(encoder.feed(event))

    fixture = [
        {"kind": "text_delta", "text": "Hel"},
        {"kind": "text_delta", "text": "lo é"},
        {"kind": "tool_call_started", "index": 0, "call_id": "call-1", "name": "search"},
        {"kind": "tool_arguments_delta", "index": 0, "text": '{"q": "x"}'},
        {
            "kind": "tool_call_completed",
            "index": 0,
            "call_id": "call-1",
            "name": "search",
            "raw_arguments": '{"q": "x"}',
        },
        {"kind": "usage", "input_tokens": 10, "output_tokens": 4, "cached_input_tokens": 1},
        {"kind": "completed"},
    ]
    actual = native.encode_chat_fixture(
        "request-abc",
        "coding",
        1_700_000_000,
        True,
        json.dumps(fixture),
    )
    assert list(actual) == _parity_golden("chat_frames")
    assert list(actual) == expected


def test_rust_chat_ignored_parameter_disclosure_matches_python_and_the_committed_golden() -> None:
    """Route-shaped controls are disclosed on the final Chat chunk, byte for byte.

    The committed golden frames freeze the ``x-experiential-ignored-parameters``
    disclosure contract; the python ``ChatSseEncoder`` remains a live secondary
    reference.
    """
    native = pytest.importorskip("exp_gateway_native")
    from exp.runtime.gateway.contracts import GatewayEvent, GatewayEventKind
    from exp.runtime.openai_protocol.streaming import ChatSseEncoder

    events = [
        GatewayEvent(kind=GatewayEventKind.TEXT_DELTA, sequence_number=0, text_delta="Hi"),
        GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=1),
    ]
    encoder = ChatSseEncoder(
        request_id="request-abc",
        model="coding",
        created_at=1_700_000_000,
        include_usage=False,
        ignored_parameters=("logprobs", "reasoning.summary"),
    )
    expected = list(encoder.start())
    for event in events:
        expected.extend(encoder.feed(event))

    fixture = [
        {"kind": "text_delta", "text": "Hi"},
        {"kind": "completed"},
    ]
    actual = native.encode_chat_fixture(
        "request-abc",
        "coding",
        1_700_000_000,
        False,
        json.dumps(fixture),
        ["logprobs", "reasoning.summary"],
    )
    assert list(actual) == _parity_golden("chat_frames_ignored_disclosure")
    assert list(actual) == expected


def _activate_revision_two(root: Path, manager: GatewayManagement) -> str:
    """Repoint the coding alias at a new revision and return its catalog digest."""
    normalized, snapshot, _changed = upsert_singleton_deployment(
        root,
        deployment_alias="coding",
        connection_name="provider-main",
        provider_model="provider-model-next",
        exact_model_id="model-revision-next",
        revision=None,
        capabilities=ModelCapabilities(),
        gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
        prices=GatewayTokenPrices(),
        pricing_source=None,
        replace=True,
    )
    manager.activate_direct_alias(
        alias_id="coding",
        alias_name="coding",
        revision_id="revision-two",
        pool_id="coding",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    return normalized.identity_sha256()


def test_admission_authorized_at_the_swap_instant_stays_pinned_to_its_revision(
    tmp_path: Path,
) -> None:
    """An admission whose authority was minted just before a hot activation
    lands, serves on the retired revision instead of failing at the boundary."""
    manager, raw_key = _configured_gateway(tmp_path)
    components = load_gateway_components(
        tmp_path,
        environment={"TEST_PROVIDER_KEY": "provider-secret-canary"},
    )
    control = NativeControlPlane(components)
    ready = components.store
    assert isinstance(ready, _ReadyControlStore)
    inner = ready.store
    minted = threading.Event()
    swapped = threading.Event()
    original = inner.authorize_request

    def stalled_authorize(
        *,
        raw_key: str,
        alias: str,
        request: GatewayRequest,
        deadline_monotonic: float,
        app_referer: str | None = None,
        app_title: str | None = None,
    ) -> AuthorizationSnapshot:
        """Mint the authorization, then stall until the activation swap lands."""
        authorization = original(
            raw_key=raw_key,
            alias=alias,
            request=request,
            deadline_monotonic=deadline_monotonic,
            app_referer=app_referer,
            app_title=app_title,
        )
        minted.set()
        assert swapped.wait(timeout=10)
        return authorization

    outcomes: list[JsonObject] = []
    errors: list[BaseException] = []

    def admit_old() -> None:
        """Admit one request racing the activation swap."""
        try:
            outcomes.append(_admit_started(control, raw_key, _chat_body()))
        except BaseException as exc:  # noqa: BLE001 - the test asserts no error.
            errors.append(exc)

    racer = threading.Thread(target=admit_old)
    with mock.patch.object(inner, "authorize_request", side_effect=stalled_authorize):
        racer.start()
        assert minted.wait(timeout=10)
        new_digest = _activate_revision_two(tmp_path, manager)
    components.reloader.refresh_if_drifted(("coding", "revision-two", new_digest))
    swapped.set()
    racer.join(timeout=15)
    assert not racer.is_alive()

    assert errors == []
    assert len(outcomes) == 1
    pinned = outcomes[0]
    assert pinned["alias_revision_id"] == "revision-one"
    assert pinned["model_id"] == "provider-model-exact"
    fresh = _admit_started(control, raw_key, _chat_body())
    assert fresh["alias_revision_id"] == "revision-two"
    assert fresh["model_id"] == "provider-model-next"
    for admission in (pinned, fresh):
        assert (
            control.settle(
                json.dumps(
                    {
                        "request_id": admission["request_id"],
                        "attempt_id": admission["attempt_id"],
                        "outcome": "completed",
                        "usage": {"input_tokens": 3, "output_tokens": 2},
                        "tool_names": [],
                        "failure": None,
                    }
                )
            )
            == "{}"
        )
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 2


def _settle_one_completed_chat(control: NativeControlPlane, raw_key: str) -> None:
    """Admit and settle one completed chat request for the presented key."""
    admission = _admit_started(control, raw_key, _chat_body())
    settled = control.settle(
        json.dumps(
            {
                "request_id": admission["request_id"],
                "attempt_id": admission["attempt_id"],
                "outcome": "completed",
                "usage": {"input_tokens": 3, "output_tokens": 2},
                "tool_names": [],
                "failure": None,
            }
        )
    )
    assert settled == "{}"


def test_usage_callbacks_scope_reports_to_the_presented_key(tmp_path: Path) -> None:
    """A key sees only its own identity; anonymous callers see the whole organization."""
    control, raw_key = _control_plane(tmp_path)
    manager = GatewayManagement(tmp_path)
    manager.create_identity(identity_id="neighbor", display_name="Neighbor")
    manager.add_grant(identity_id="neighbor", alias_id="coding")
    neighbor_key = manager.issue_key(identity_id="neighbor", key_id="key-neighbor").raw_key
    _settle_one_completed_chat(control, raw_key)

    organization_wide = json.loads(control.usage_json("{}"))
    assert organization_wide["totals"]["requests"] == 1
    assert [item["identity_id"] for item in organization_wide["identities"]] == [
        "default",
        "neighbor",
    ]

    scoped = json.loads(control.usage_json(json.dumps({"raw_key": raw_key})))
    assert scoped["totals"]["requests"] == 1
    assert [item["identity_id"] for item in scoped["identities"]] == ["default"]

    isolated = json.loads(control.usage_json(json.dumps({"raw_key": neighbor_key})))
    assert isolated["totals"]["requests"] == 0
    assert isolated["totals"]["input_tokens"] == 0
    assert [item["identity_id"] for item in isolated["identities"]] == ["neighbor"]
    assert isolated["by_billing_source"] == []

    page = json.loads(control.usage_page(json.dumps({"raw_key": raw_key})))
    assert "Gateway usage" in page["html"]
    assert "default" in page["html"]
    isolated_page = json.loads(control.usage_page(json.dumps({"raw_key": neighbor_key})))
    assert "default" not in isolated_page["html"]


def test_close_thread_resources_releases_the_calling_thread_connections(
    tmp_path: Path,
) -> None:
    """The worker-exit callback closes this thread's cached SQLite connections.

    Bridge worker threads call this once as their pool shuts down, so the
    per-thread connections cached by ``persistent_connection`` must close on
    the calling thread and later callbacks must reopen cleanly.
    """
    control, raw_key = _control_plane(tmp_path)
    _settle_one_completed_chat(control, raw_key)
    first = json.loads(control.close_thread_resources("{}"))
    assert first["closed_connections"] >= 1
    second = json.loads(control.close_thread_resources("{}"))
    assert second["closed_connections"] == 0
    # The cache repopulates transparently: the next ledger-backed callback
    # reopens a connection and serves normally.
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 1
    assert json.loads(control.close_thread_resources("{}"))["closed_connections"] >= 1


def test_usage_callbacks_reject_an_invalid_key(tmp_path: Path) -> None:
    """A bad key on either usage callback maps to the shared 401 public error."""
    control, _raw_key = _control_plane(tmp_path)
    for callback in (control.usage_json, control.usage_page):
        with pytest.raises(NativeBridgeError) as excinfo:
            callback(json.dumps({"raw_key": "exp_vk_invalid"}))
        payload = json.loads(excinfo.value.public_error_json)
        assert payload["status_code"] == 401
        assert payload["code"] == "invalid_key"


def _responses_body(
    *,
    model: str = "coding",
    stream: bool = False,
    previous_response_id: str | None = None,
    with_tools: bool = False,
    reasoning_summary: str | None = None,
    store: bool | None = None,
) -> str:
    """Return one raw Responses API request body."""
    payload: JsonObject = {"model": model, "input": [{"role": "user", "content": "hi"}]}
    if stream:
        payload["stream"] = True
    if store is not None:
        payload["store"] = store
    if previous_response_id is not None:
        payload["previous_response_id"] = previous_response_id
    if reasoning_summary is not None:
        payload["reasoning"] = {"effort": "high", "summary": reasoning_summary}
    if with_tools:
        payload["tools"] = [
            {
                "type": "function",
                "name": "search",
                "description": "Find things.",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            }
        ]
        payload["tool_choice"] = "auto"
        payload["metadata"] = {"team": "core"}
        payload["max_output_tokens"] = 128
        payload["temperature"] = 0.5
    return json.dumps(payload)


def _admit_responses(control: NativeControlPlane, raw_key: str, body: str) -> JsonObject:
    """Run one Responses-surface admission call and decode its JSON response."""
    return json.loads(
        control.admit(json.dumps({"raw_key": raw_key, "body": body, "surface": "responses"}))
    )


def _admitted_request_id(admission: JsonObject) -> str:
    """Narrow one admission's request identity to a string."""
    request_id = admission["request_id"]
    assert isinstance(request_id, str)
    return request_id


def _payload_messages(admission: JsonObject) -> list[JsonObject]:
    """Narrow one admission's first wire entry to its payload message list."""
    route = admission["route"]
    assert isinstance(route, list)
    wire = route[0]
    assert isinstance(wire, dict)
    payload = wire["upstream_payload"]
    assert isinstance(payload, dict)
    messages = payload["messages"]
    assert isinstance(messages, list)
    return [message for message in messages if isinstance(message, dict)]


def _responses_parity_case() -> tuple[list[GatewayEvent], str]:
    """Return matching python events and the Rust fixture JSON for one stream."""
    from exp.common.models.model import ToolCall

    events = [
        GatewayEvent(
            kind=GatewayEventKind.REASONING_SUMMARY_DELTA,
            sequence_number=0,
            reasoning_summary_output_index=0,
            reasoning_summary_index=0,
            reasoning_item_id="rs_01161eec6982f41bdc4271a8fceb6c60",
            text_delta="Checked the plan.",
        ),
        GatewayEvent(kind=GatewayEventKind.TEXT_DELTA, sequence_number=1, text_delta="Hel"),
        GatewayEvent(kind=GatewayEventKind.TEXT_DELTA, sequence_number=2, text_delta="lo é"),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_CALL_STARTED,
            sequence_number=3,
            tool_call_index=0,
            tool_call_id="call-1",
            tool_name="search",
        ),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_ARGUMENTS_DELTA,
            sequence_number=4,
            tool_call_index=0,
            raw_arguments_delta='{"q": "x"}',
        ),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_CALL_COMPLETED,
            sequence_number=5,
            tool_call_index=0,
            tool_call=ToolCall(
                call_id="call-1",
                name="search",
                arguments={"q": "x"},
                raw_arguments='{"q": "x"}',
            ),
        ),
        GatewayEvent(
            kind=GatewayEventKind.USAGE,
            sequence_number=6,
            usage=GatewayUsage(input_tokens=10, output_tokens=4, cached_input_tokens=1),
        ),
        GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=7),
    ]
    fixture = [
        {
            "kind": "reasoning_summary_delta",
            "output_index": 0,
            "summary_index": 0,
            "item_id": "rs_01161eec6982f41bdc4271a8fceb6c60",
            "text": "Checked the plan.",
        },
        {"kind": "text_delta", "text": "Hel"},
        {"kind": "text_delta", "text": "lo é"},
        {"kind": "tool_call_started", "index": 0, "call_id": "call-1", "name": "search"},
        {"kind": "tool_arguments_delta", "index": 0, "text": '{"q": "x"}'},
        {
            "kind": "tool_call_completed",
            "index": 0,
            "call_id": "call-1",
            "name": "search",
            "raw_arguments": '{"q": "x"}',
        },
        {"kind": "usage", "input_tokens": 10, "output_tokens": 4, "cached_input_tokens": 1},
        {"kind": "completed"},
    ]
    return events, json.dumps(fixture)


def _python_responses_frames(
    body: str,
    events: list[GatewayEvent],
    *,
    created_at: float,
) -> list[str]:
    """Encode one Responses stream through the python encoder."""
    from exp.runtime.openai_protocol.streaming import ResponsesSseEncoder

    decoded = decode_responses(json.loads(body))
    encoder = ResponsesSseEncoder(
        request_id="request-abc",
        model=decoded.alias,
        created_at=created_at,
        request=decoded.request,
    )
    expected = list(encoder.start())
    for event in events:
        expected.extend(encoder.feed(event))
    return expected


def _native_envelope(body: str) -> str:
    """Render the control plane's Responses envelope JSON for one raw body."""
    from exp.runtime.gateway.native_responses import responses_envelope

    return json.dumps(responses_envelope(decode_responses(json.loads(body)).request))


def test_rust_responses_sse_frames_match_python_and_the_committed_golden() -> None:
    """Rust Responses SSE frames equal the committed golden frames.

    The committed golden frames are the durable contract: they were generated
    from the retired python Responses encoder and outlive it.
    """
    native = pytest.importorskip("exp_gateway_native")
    body = _responses_body(stream=True, with_tools=True, reasoning_summary="concise")
    events, fixture = _responses_parity_case()
    expected = _python_responses_frames(body, events, created_at=1_700_000_000.25)
    actual = native.encode_responses_fixture(
        "request-abc",
        "coding",
        1_700_000_000.25,
        _native_envelope(body),
        fixture,
    )
    assert list(actual) == _parity_golden("responses_frames")
    assert list(actual) == expected


def test_rust_responses_refusal_and_incomplete_match_the_committed_golden() -> None:
    """Refusal deltas and the incomplete terminal equal the committed golden."""
    native = pytest.importorskip("exp_gateway_native")

    body = _responses_body(stream=True)
    events = [
        GatewayEvent(kind=GatewayEventKind.REFUSAL_DELTA, sequence_number=0, text_delta="I ca"),
        GatewayEvent(kind=GatewayEventKind.REFUSAL_DELTA, sequence_number=1, text_delta="nnot"),
        GatewayEvent(kind=GatewayEventKind.INCOMPLETE, sequence_number=2),
    ]
    fixture = json.dumps(
        [
            {"kind": "refusal_delta", "text": "I ca"},
            {"kind": "refusal_delta", "text": "nnot"},
            {"kind": "incomplete"},
        ]
    )
    expected = _python_responses_frames(body, events, created_at=1_700_000_000.0)
    actual = native.encode_responses_fixture(
        "request-abc", "coding", 1_700_000_000.0, _native_envelope(body), fixture
    )
    assert list(actual) == _parity_golden("responses_refusal_frames")
    assert list(actual) == expected


def test_rust_responses_failed_terminal_matches_the_committed_golden() -> None:
    """The failed terminal envelope and error body equal the committed golden."""
    native = pytest.importorskip("exp_gateway_native")

    body = _responses_body(stream=True)
    events = [
        GatewayEvent(kind=GatewayEventKind.TEXT_DELTA, sequence_number=0, text_delta="part"),
        GatewayEvent(
            kind=GatewayEventKind.FAILED,
            sequence_number=1,
            failure=GatewayFailure(
                failure_class=GatewayFailureClass.PROVIDER_INTERNAL,
                safe_message="provider exploded",
            ),
        ),
    ]
    fixture = json.dumps(
        [
            {"kind": "text_delta", "text": "part"},
            {"kind": "failed", "text": "provider exploded"},
        ]
    )
    expected = _python_responses_frames(body, events, created_at=1_700_000_000.5)
    actual = native.encode_responses_fixture(
        "request-abc", "coding", 1_700_000_000.5, _native_envelope(body), fixture
    )
    assert list(actual) == _parity_golden("responses_failed_frames")
    assert list(actual) == expected


def test_rust_responses_completed_body_matches_python_and_the_committed_golden() -> None:
    """The Rust non-streaming Responses body equals the committed golden bytes.

    The python ``completed_body`` remains a live secondary reference because
    the Pi bridge still aggregates through it.
    """
    native = pytest.importorskip("exp_gateway_native")
    from exp.runtime.openai_protocol.response import completed_body

    body = _responses_body(with_tools=True, reasoning_summary="concise")
    events, fixture = _responses_parity_case()
    decoded = decode_responses(json.loads(body))
    expected = completed_body(
        request=decoded.request,
        request_id="request-abc",
        model=decoded.alias,
        created_at=1_700_000_000.25,
        events=tuple(events),
    )
    actual = native.completed_responses_fixture(
        "request-abc",
        "coding",
        1_700_000_000.25,
        _native_envelope(body),
        fixture,
    )
    assert actual == _parity_golden("responses_completed_body")
    assert json.loads(actual) == expected
    assert actual == json.dumps(expected, separators=(",", ":"), ensure_ascii=False)


def test_rust_responses_rejects_streams_without_terminals() -> None:
    """A provider stream without a terminal fails closed like the python path."""
    native = pytest.importorskip("exp_gateway_native")
    body = _responses_body()
    fixture = json.dumps([{"kind": "text_delta", "text": "no terminal"}])
    with pytest.raises(ValueError, match="all_routes_failed"):
        native.completed_responses_fixture(
            "request-abc", "coding", 1_700_000_000.0, _native_envelope(body), fixture
        )
    malformed = json.dumps(
        [
            {"kind": "tool_arguments_delta", "index": 3, "text": "{"},
            {"kind": "completed"},
        ]
    )
    with pytest.raises(ValueError, match="invalid_provider_stream"):
        native.encode_responses_fixture(
            "request-abc", "coding", 1_700_000_000.0, _native_envelope(body), malformed
        )


def test_responses_admission_is_native_with_envelope_and_payload(tmp_path: Path) -> None:
    """A Responses request admits natively (no escalation) with the exact
    request-reflecting envelope and a route-safe dialect payload."""
    from exp.runtime.gateway.native_responses import responses_envelope

    control, raw_key = _control_plane(tmp_path)
    payload = json.loads(_responses_body(with_tools=True))
    body = json.dumps(payload)
    admission = _flatten_started(control, _admit_responses(control, raw_key, body))
    assert "escalate" not in admission
    assert admission["surface"] == "responses"
    assert admission["dialect"] == "openai_compatible"
    decoded = decode_responses(json.loads(body))
    public_request = decoded.request
    assert admission["ignored_parameters"] == []
    assert admission["envelope"] == responses_envelope(public_request)
    provider_request = public_request.model_copy(update={"stream": True, "include_usage": True})
    assert admission["upstream_payload"] == openai_compatible_stream_payload(
        "provider-model-exact", provider_request
    )
    upstream_payload = admission["upstream_payload"]
    assert isinstance(upstream_payload, dict)
    assert "reasoning" not in upstream_payload
    assert "reasoning_effort" not in upstream_payload
    settled = control.settle(
        json.dumps(
            {
                "request_id": admission["request_id"],
                "attempt_id": admission["attempt_id"],
                "outcome": "completed",
                "usage": {"input_tokens": 7, "output_tokens": 2},
                "tool_names": [],
                "failure": None,
            }
        )
    )
    assert settled == "{}"
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 1


def test_responses_admission_drops_effort_on_a_reasoning_less_route(tmp_path: Path) -> None:
    """A Responses effort on a zero-reasoning route serves without it, disclosed.

    This surface previously answered the named 400; the owner-approved drop
    policy (2026-09-01) serves the request effortless instead, because a
    zero-reasoning route cannot honor any depth and first-party clients pin
    effort globally.
    """
    control, raw_key = _control_plane(tmp_path)
    payload = json.loads(_responses_body())
    payload["reasoning"] = {"effort": "high"}

    admission = _flatten_started(control, _admit_responses(control, raw_key, json.dumps(payload)))
    assert admission["ignored_parameters"] == ["reasoning_effort"]
    upstream = admission["upstream_payload"]
    assert isinstance(upstream, dict)
    assert "reasoning" not in upstream


def test_responses_continuation_round_trip_and_fail_closed(tmp_path: Path) -> None:
    """Retained history is prepended on continuation; unknown, refused, and
    cross-namespace continuations fail closed with the shared public error."""
    control, raw_key = _control_plane(tmp_path)
    first = _admit_responses(control, raw_key, _responses_body())
    remembered = control.remember(
        json.dumps(
            {
                "request_id": first["request_id"],
                "text": "The answer is 42.",
                "refusal": False,
                "tool_calls": [],
            }
        )
    )
    assert remembered == "{}"
    response_id = stable_public_id("resp", _admitted_request_id(first))
    second = _admit_responses(control, raw_key, _responses_body(previous_response_id=response_id))
    messages = _payload_messages(second)
    assert [message["role"] for message in messages] == ["user", "assistant", "user"]
    assert messages[1]["content"] == "The answer is 42."

    with pytest.raises(NativeBridgeError) as unknown:
        _admit_responses(control, raw_key, _responses_body(previous_response_id="resp_missing"))
    payload = json.loads(unknown.value.public_error_json)
    assert payload["status_code"] == 400
    assert payload["code"] == "previous_response_not_found"
    assert payload["param"] == "previous_response_id"

    refused = _admit_responses(control, raw_key, _responses_body())
    assert (
        control.remember(
            json.dumps(
                {
                    "request_id": refused["request_id"],
                    "text": "partial",
                    "refusal": True,
                    "tool_calls": [],
                }
            )
        )
        == "{}"
    )
    with pytest.raises(NativeBridgeError) as after_refusal:
        _admit_responses(
            control,
            raw_key,
            _responses_body(
                previous_response_id=stable_public_id("resp", _admitted_request_id(refused))
            ),
        )
    assert (
        json.loads(after_refusal.value.public_error_json)["code"] == "previous_response_not_found"
    )

    foreign = ProtocolNamespace(
        organization_id="other-org",
        identity_id="other-identity",
        alias_revision_id="other-revision",
    )
    with pytest.raises(OpenAIProtocolError) as crossed:
        control._continuations.resolve_now(  # noqa: SLF001 - namespace isolation assertion.
            namespace=foreign,
            previous_response_id=response_id,
        )
    assert crossed.value.detail.code == "previous_response_not_found"


def test_responses_output_less_turn_is_continuable(tmp_path: Path) -> None:
    """A turn retained with no text, items, or calls continues as the input so far.

    The data plane remembers an output-less turn (thinking exhausted the
    budget, terminal ``incomplete``) with every retention field empty; the
    control plane must retain the conversation rather than treat empty output
    as nothing to remember, or the response id it already handed out dies
    ``previous_response_not_found`` on the next turn.
    """
    control, raw_key = _control_plane(tmp_path)
    first = _admit_responses(control, raw_key, _responses_body())
    assert (
        control.remember(
            json.dumps(
                {
                    "request_id": first["request_id"],
                    "text": "",
                    "message_outputs": [],
                    "refusal": False,
                    "encrypted_reasoning": [],
                    "tool_calls": [],
                }
            )
        )
        == "{}"
    )
    response_id = stable_public_id("resp", _admitted_request_id(first))
    second = _admit_responses(control, raw_key, _responses_body(previous_response_id=response_id))
    assert [message["role"] for message in _payload_messages(second)] == ["user", "user"]


def test_fallback_served_alias_continuation_degrades_to_resend_not_503(tmp_path: Path) -> None:
    """A continuation on an alias served via its last-good fallback still fails
    with the 400 'resend the full conversation' error when it cannot resolve —
    never a 503. The fallback re-key is upstream of continuation binding, so it
    adds no 5xx path; a fresh request on the same alias serves via the fallback.
    """
    control, raw_key = _control_plane(tmp_path)
    # Dead-pin the active revision so the alias is served on its last-good prior.
    manager = GatewayManagement(tmp_path)
    manager.activate_direct_alias(
        alias_id="coding",
        alias_name="coding",
        revision_id="revision-dead",
        pool_id="coding",
        snapshot_ref="catalog-snapshots/missing.json",
        catalog_sha256="a" * 64,
    )

    # A fresh (non-continuation) Responses request still serves via the fallback.
    served = _admit_responses(control, raw_key, _responses_body())
    assert served["request_id"]

    # A continuation whose previous_response_id cannot resolve returns the shared
    # 400 resend error, not a 503 — confirming the re-key never turns an
    # unresolvable continuation into a server error.
    with pytest.raises(NativeBridgeError) as rejected:
        _admit_responses(control, raw_key, _responses_body(previous_response_id="resp_missing"))
    payload = json.loads(rejected.value.public_error_json)
    assert payload["status_code"] == 400
    assert payload["code"] == "previous_response_not_found"
    assert payload["param"] == "previous_response_id"


def test_responses_tool_call_retention_survives_continuation(tmp_path: Path) -> None:
    """Completed tool calls are retained and replayed into continued history."""
    control, raw_key = _control_plane(tmp_path)
    first = _admit_responses(control, raw_key, _responses_body(with_tools=True))
    assert (
        control.remember(
            json.dumps(
                {
                    "request_id": first["request_id"],
                    "text": "",
                    "refusal": False,
                    "tool_calls": [
                        {"call_id": "call-9", "name": "search", "arguments": '{"q":"x"}'}
                    ],
                }
            )
        )
        == "{}"
    )
    second = _admit_responses(
        control,
        raw_key,
        _responses_body(previous_response_id=stable_public_id("resp", _admitted_request_id(first))),
    )
    assistant = _payload_messages(second)[1]
    assert assistant["role"] == "assistant"
    tool_calls = assistant["tool_calls"]
    assert isinstance(tool_calls, list)
    first_call = tool_calls[0]
    assert isinstance(first_call, dict)
    assert first_call["function"] == {"name": "search", "arguments": '{"q":"x"}'}


def test_fireworks_multihop_responses_retention_stays_sealed(tmp_path: Path) -> None:
    """Executing a retained carrier never stores its decrypted plaintext copy."""
    _manager, raw_key = _configured_gateway(
        tmp_path,
        base_url="https://api.fireworks.ai/inference/v1",
        capabilities=ModelCapabilities(supports_tools=True),
    )
    control = NativeControlPlane(
        load_gateway_components(
            tmp_path,
            environment={"TEST_PROVIDER_KEY": "shared-fireworks-secret"},
        )
    )
    first = _admit_responses(control, raw_key, _responses_body(with_tools=True))
    started = _start_first(control, first)
    route = first["route"]
    assert isinstance(route, list)
    depth = started["route_depth"]
    assert isinstance(depth, int)
    wire = route[depth]
    assert isinstance(wire, dict)
    carrier = json.loads(
        control.seal_reasoning_content(
            json.dumps(
                {
                    "request_id": first["request_id"],
                    "route_depth": depth,
                    "route_sha256": wire["fireworks_reasoning_route_sha256"],
                    "content": "PLAINTEXT-HIDDEN",
                    "assistant_content": None,
                    "tool_calls": [
                        {"call_id": "call-one", "name": "lookup", "raw_arguments": "{}"}
                    ],
                }
            )
        )
    )["carrier"]
    assert (
        control.remember(
            json.dumps(
                {
                    "request_id": first["request_id"],
                    "text": "",
                    "refusal": False,
                    "reasoning_content_carrier": carrier,
                    "tool_calls": [{"call_id": "call-one", "name": "lookup", "arguments": "{}"}],
                }
            )
        )
        == "{}"
    )
    first_response_id = stable_public_id("resp", _admitted_request_id(first))
    second_body = json.dumps(
        {
            "model": "coding",
            "previous_response_id": first_response_id,
            "input": [{"type": "function_call_output", "call_id": "call-one", "output": "done"}],
        }
    )
    second = _admit_responses(control, raw_key, second_body)
    upstream = _payload_messages(second)
    assert upstream[1]["reasoning_content"] == "PLAINTEXT-HIDDEN"
    assert (
        control.remember(
            json.dumps(
                {
                    "request_id": second["request_id"],
                    "text": "finished",
                    "refusal": False,
                    "tool_calls": [],
                }
            )
        )
        == "{}"
    )
    accounting = control._accounting.entry(_admitted_request_id(second))  # noqa: SLF001
    assert accounting is not None and accounting.continuation is not None
    retained = control._continuations.resolve_now(  # noqa: SLF001
        namespace=accounting.continuation.namespace,
        previous_response_id=stable_public_id("resp", _admitted_request_id(second)),
    )
    kinds = [block.kind for message in retained.messages for block in message.provider_reasoning]
    assert kinds == ["sealed_reasoning_content"]
    retained_reasoning = [
        block.model_dump(mode="json")
        for message in retained.messages
        for block in message.provider_reasoning
    ]
    assert "PLAINTEXT-HIDDEN" not in json.dumps(retained_reasoning)


def test_responses_admission_rejects_invalid_bodies_and_bad_keys(tmp_path: Path) -> None:
    """Responses-surface admission fails closed on protocol and key errors."""
    control, raw_key = _control_plane(tmp_path)
    with pytest.raises(NativeBridgeError) as invalid:
        control.admit(json.dumps({"raw_key": raw_key, "body": "{not json", "surface": "responses"}))
    assert json.loads(invalid.value.public_error_json)["code"] == "invalid_json"
    rejected = json.dumps({"model": "coding", "input": "hi", "modalities": ["audio"]})
    with pytest.raises(NativeBridgeError) as protocol:
        control.admit(json.dumps({"raw_key": raw_key, "body": rejected, "surface": "responses"}))
    assert json.loads(protocol.value.public_error_json)["status_code"] == 400
    with pytest.raises(NativeBridgeError) as bad_key:
        control.admit(
            json.dumps(
                {
                    "raw_key": "exp_vk_invalid",
                    "body": _responses_body(),
                    "surface": "responses",
                }
            )
        )
    assert json.loads(bad_key.value.public_error_json)["status_code"] == 401
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 0


def _configured_project_singletons(
    root: Path,
    *,
    base_url: str = "http://127.0.0.1:9/v1",
) -> tuple[GatewayManagement, str]:
    """Create one project alias whose candidates each own a singleton pool."""
    from exp.common.models import (
        BillingSource,
        ConnectionConfig,
        ModelRecord,
        load_model_catalog,
        write_model_catalog,
    )
    from exp.runtime.gateway.catalog_authority import upsert_connection

    manager = GatewayManagement(root)
    manager.initialize()
    for deployment_alias in ("cheap", "baseline"):
        upsert_connection(
            root,
            name=f"{deployment_alias}-provider",
            connection=ConnectionConfig(
                provider="openai-compatible",
                base_url=base_url,
                api_key_env=f"{deployment_alias.upper()}_PROVIDER_KEY",
            ),
            replace=False,
        )
    authored = load_model_catalog(root / "models.toml")
    models = dict(authored.models)
    models["embedder"] = ModelRecord(
        connection="cheap-provider",
        model="embedder-model",
        billing_source=BillingSource.CUSTOMER_MANAGED,
        capabilities=ModelCapabilities(supports_embeddings=True),
    )
    write_model_catalog(root / "models.toml", authored.model_copy(update={"models": models}))
    normalized = None
    snapshot = None
    for deployment_alias in ("cheap", "baseline"):
        normalized, snapshot, _changed = upsert_singleton_deployment(
            root,
            deployment_alias=deployment_alias,
            connection_name=f"{deployment_alias}-provider",
            provider_model=f"{deployment_alias}-model",
            exact_model_id="model-revision-exact",
            revision=None,
            capabilities=ModelCapabilities(),
            gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
            prices=GatewayTokenPrices(),
            pricing_source=None,
            replace=False,
        )
    assert normalized is not None
    assert snapshot is not None
    manager.activate_project_alias(
        alias_id="coding",
        alias_name="coding",
        revision_id="revision-project-one",
        project_ref="project-one",
        activation_ref="activation-one",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.create_identity(identity_id="default", display_name="Default")
    manager.add_grant(identity_id="default", alias_id="coding")
    issued = manager.issue_key(identity_id="default", key_id="key-one")
    return manager, issued.raw_key


def _project_control_plane(
    root: Path,
    *,
    base_url: str = "http://127.0.0.1:9/v1",
) -> tuple[GatewayManagement, NativeControlPlane, str]:
    """Load the native control plane over one real project-backed alias."""
    from exp.runtime.gateway.lifecycle_test import (
        _project_activation,
        _ReadinessProjectRepository,
    )

    environment = {
        "CHEAP_PROVIDER_KEY": "cheap-secret-canary",
        "BASELINE_PROVIDER_KEY": "baseline-secret-canary",
    }
    manager, raw_key = _configured_project_singletons(root, base_url=base_url)
    components = load_gateway_components(
        root,
        environment=environment,
        project_repository=_ReadinessProjectRepository(
            _project_activation(
                root,
                candidate_aliases=("baseline", "cheap"),
                environment=environment,
            )
        ),
    )
    return manager, NativeControlPlane(components), raw_key


def test_project_alias_embedding_failure_serves_the_frozen_baseline_natively(
    tmp_path: Path,
) -> None:
    """A project alias admits natively and a failed embed lands the baseline.

    The unreachable provider endpoint makes the request-time embed fail, so
    the frozen conservative baseline serves the request with content-free
    accounting instead of failing the admission.
    """
    import sqlite3

    manager, control, raw_key = _project_control_plane(tmp_path)
    body = json.dumps(
        {"model": "coding", "messages": [{"role": "user", "content": "project-prompt-canary"}]}
    )
    admission = _admit_started(control, raw_key, body)
    assert "escalate" not in admission
    assert admission["route_reason"] == "learned_router"
    assert admission["model_id"] == "baseline-model"
    assert admission["exact_model_id"] == "model-revision-exact"
    assert admission["alias_revision_id"] == "revision-project-one"
    headers = admission["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == "Bearer baseline-secret-canary"

    assert (
        control.settle(
            json.dumps(
                {
                    "request_id": admission["request_id"],
                    "attempt_id": admission["attempt_id"],
                    "outcome": "completed",
                    "usage": {"input_tokens": 6, "output_tokens": 2},
                    "tool_names": [],
                    "failure": None,
                }
            )
        )
        == "{}"
    )
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 1
    with sqlite3.connect(manager.database_path) as connection:
        rows = connection.execute(
            """
            SELECT route_reason, fallback_reason, exact_model_id, state, content_retained
            FROM gateway_attempts
            """
        ).fetchall()
    assert rows == [("learned_router", "embedding_error", "model-revision-exact", "completed", 0)]
    durable = manager.database_path.read_bytes()
    wal = manager.database_path.with_name("gateway.db-wal")
    if wal.exists():
        durable += wal.read_bytes()
    assert b"project-prompt-canary" not in durable
    assert b"cheap-secret-canary" not in durable
    assert b"baseline-secret-canary" not in durable


def test_project_alias_native_selection_matches_the_python_resolver(tmp_path: Path) -> None:
    """Native admission and the python async resolver pick one deployment.

    A loopback embeddings endpoint gives both engines the same frozen policy
    inputs, so the deployment the native bridge admits equals the deployment
    the shared async route resolution selects.
    """
    import asyncio
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from exp.runtime.openai_protocol.state import ProtocolNamespace, episode_namespace

    class EmbeddingHandler(BaseHTTPRequestHandler):
        """Serve deterministic embeddings for request-time selection."""

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            """Suppress nondeterministic loopback server logs."""
            del format, args

        def do_POST(self) -> None:
            """Return one fixed embedding vector for any embed request."""
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            body = json.dumps(
                {
                    "data": [{"embedding": [1.0, 0.0], "index": 0}],
                    "model": str(payload["model"]),
                    "usage": {"prompt_tokens": 3, "total_tokens": 3},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), EmbeddingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        _manager, control, raw_key = _project_control_plane(tmp_path, base_url=base_url)
        admission = _admit_started(control, raw_key, _chat_body())
        assert "escalate" not in admission
        assert admission["route_reason"] == "learned_router"

        components = control._components  # noqa: SLF001 - parity over one loaded state.
        request = decode_chat(json.loads(_chat_body())).request
        authorization = components.store.authorize_request(
            raw_key=raw_key,
            alias="coding",
            request=request,
            deadline_monotonic=time.monotonic() + 30.0,
        )
        episode = episode_namespace(
            namespace=ProtocolNamespace(
                organization_id=authorization.organization_id,
                identity_id=authorization.identity_id,
                alias_revision_id=authorization.alias_revision_id,
            ),
            caller_episode_key=None,
            request_id=authorization.request_id,
        )
        route = asyncio.run(
            components.routes.resolve(
                authorization=authorization,
                request=request,
                episode_namespace=episode,
            )
        )
        assert route.deployment.deployment_id == admission["deployment_id"]
        assert route.route_reason == admission["route_reason"]
        assert not route.fallback_deployments
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_responses_continuation_reuses_the_original_selection_episode(
    tmp_path: Path,
) -> None:
    """A continued Responses request joins its first turn's selection episode.

    The continuation carries the original episode key, so learned selection
    replays the journaled decision instead of re-running request-time
    embedding for a fresh episode.
    """
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    embed_calls: list[str] = []

    class EmbeddingHandler(BaseHTTPRequestHandler):
        """Serve deterministic embeddings and count every embed request."""

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            """Suppress nondeterministic loopback server logs."""
            del format, args

        def do_POST(self) -> None:
            """Return one fixed embedding vector and record the call."""
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            embed_calls.append(self.path)
            body = json.dumps(
                {
                    "data": [{"embedding": [1.0, 0.0], "index": 0}],
                    "model": str(payload["model"]),
                    "usage": {"prompt_tokens": 3, "total_tokens": 3},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), EmbeddingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        _manager, control, raw_key = _project_control_plane(tmp_path, base_url=base_url)
        first = _admit_responses(control, raw_key, _responses_body())
        assert "escalate" not in first
        first_embeds = len(embed_calls)
        assert first_embeds >= 1
        assert (
            control.remember(
                json.dumps(
                    {
                        "request_id": first["request_id"],
                        "text": "The answer is 42.",
                        "refusal": False,
                        "tool_calls": [],
                    }
                )
            )
            == "{}"
        )
        second = _admit_responses(
            control,
            raw_key,
            _responses_body(
                previous_response_id=stable_public_id("resp", _admitted_request_id(first))
            ),
        )
        assert "escalate" not in second
        assert len(embed_calls) == first_embeds
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_admit_persists_caller_app_identity_for_attribution(tmp_path: Path) -> None:
    """The native admit path forwards caller HTTP-Referer/X-Title to durable attribution."""
    control, raw_key = _control_plane(tmp_path)

    admission = json.loads(
        control.admit(
            json.dumps(
                {
                    "raw_key": raw_key,
                    "body": _chat_body(),
                    "app_referer": "https://app.example.com",
                    "app_title": "Example App",
                }
            )
        )
    )
    admission = _flatten_started(control, admission)

    ledger = cast("SQLiteAttemptLedger", control._components.ledger)  # noqa: SLF001
    with sqlite3.connect(ledger.database_path) as connection:
        row = connection.execute(
            """
            SELECT r.app_referer, r.app_title
            FROM gateway_requests AS r
            JOIN gateway_attempts AS a ON a.request_id = r.request_id
            WHERE a.attempt_id = ?
            """,
            (admission["attempt_id"],),
        ).fetchone()
    assert row == ("https://app.example.com", "Example App")


class _HostedComponents:
    """Hosted-shaped components: no group-commit writer, sync ledger only.

    Mirrors a platform composition over its own store (for example Postgres),
    which cannot provide the local engine's SQLite batching writer. Both the
    protocol-conformant ``write_ledger = None`` and a components object with
    no such attribute at all must compose, because the hosted seam promises
    settlement through the plain synchronous ledger.
    """

    def __init__(self, inner: object, *, omit_write_ledger: bool) -> None:
        self._inner = inner
        if not omit_write_ledger:
            self.write_ledger = None

    def __getattr__(self, name: str) -> object:
        if name == "write_ledger":
            raise AttributeError(name)
        return getattr(self._inner, name)


@pytest.mark.parametrize("omit_write_ledger", [False, True])
def test_hosted_components_without_group_commit_writer_settle(
    tmp_path: Path,
    omit_write_ledger: bool,
) -> None:
    """A composition without the local batching writer settles via its ledger.

    Regression: the control plane once accessed ``components.write_ledger``
    unconditionally, so a hosted composition (platform #620) crashed with
    AttributeError before binding anything.
    """
    _manager, raw_key = _configured_gateway(tmp_path)
    inner = load_gateway_components(
        tmp_path,
        environment={"TEST_PROVIDER_KEY": "provider-secret-canary"},
    )
    hosted = _HostedComponents(inner, omit_write_ledger=omit_write_ledger)
    control = NativeControlPlane(
        cast("NativeGatewayComponents", hosted),
        request_timeout_seconds=120.0,
    )
    admission = _admit_started(control, raw_key, _chat_body())
    settled = control.settle(
        json.dumps(
            {
                "request_id": admission["request_id"],
                "attempt_id": admission["attempt_id"],
                "outcome": "completed",
                "usage": {"input_tokens": 4, "output_tokens": 2},
                "tool_names": [],
                "failure": None,
            }
        )
    )
    assert settled == "{}"
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 1
    assert report["totals"]["terminal_counts"] == [{"state": "completed", "attempts": 1}]


def _messages_fixture_json() -> str:
    """Return the Rust fixture-event JSON for the shared Messages stream."""
    return json.dumps(
        [
            {"kind": "text_delta", "text": "Hel"},
            {"kind": "text_delta", "text": "lo é"},
            {"kind": "tool_call_started", "index": 0, "call_id": "call-1", "name": "search"},
            {"kind": "tool_arguments_delta", "index": 0, "text": '{"q": '},
            {"kind": "tool_arguments_delta", "index": 0, "text": '"x"}'},
            {
                "kind": "tool_call_completed",
                "index": 0,
                "call_id": "call-1",
                "name": "search",
                "raw_arguments": '{"q": "x"}',
            },
            {"kind": "usage", "input_tokens": 10, "output_tokens": 4, "cached_input_tokens": 3},
            {"kind": "completed"},
        ]
    )


def test_rust_messages_sse_frames_match_the_committed_golden() -> None:
    """Rust Messages SSE frames equal the committed golden fixture."""
    native = pytest.importorskip("exp_gateway_native")

    actual = native.encode_messages_fixture("request-abc", "coding", _messages_fixture_json())
    assert list(actual) == _parity_golden("messages_tool_stream_frames")


def test_rust_messages_drop_reasoning_summary_deltas_without_changing_the_golden() -> None:
    """The Messages surface has no reasoning-summary shape, so deltas emit nothing.

    A stream carrying a reasoning summary produces exactly the committed golden
    frames of the same stream without one.
    """
    native = pytest.importorskip("exp_gateway_native")

    events = json.loads(_messages_fixture_json())
    events.insert(
        0,
        {
            "kind": "reasoning_summary_delta",
            "output_index": 0,
            "summary_index": 0,
            "text": "Checked the plan.",
        },
    )
    actual = native.encode_messages_fixture("request-abc", "coding", json.dumps(events))
    assert list(actual) == _parity_golden("messages_tool_stream_frames")


def test_rust_messages_failure_frames_match_the_committed_golden() -> None:
    """A failed Messages terminal equals the committed golden error event."""
    native = pytest.importorskip("exp_gateway_native")

    fixture = json.dumps(
        [
            {"kind": "text_delta", "text": "oops"},
            {"kind": "failed", "text": "provider stream failed"},
        ]
    )
    actual = native.encode_messages_fixture("request-abc", "coding", fixture)
    assert list(actual) == _parity_golden("messages_failure_frames")


def test_rust_messages_completed_body_matches_the_committed_golden() -> None:
    """The Rust non-streaming Anthropic message equals the committed golden body."""
    native = pytest.importorskip("exp_gateway_native")

    actual = native.completed_messages_fixture("request-abc", "coding", _messages_fixture_json())
    assert actual == _parity_golden("messages_tool_stream_body")


def test_rust_anthropic_error_translation_matches_the_committed_goldens() -> None:
    """The Rust Anthropic error envelope equals the committed translations.

    Every failure class is exercised through one committed OpenAI-shaped
    input and its committed Anthropic envelope, plus one param-carrying
    protocol error to prove the param folding. The committed inputs also pin
    the OpenAI-side taxonomy for classes whose live rendering is wall-clock
    dependent (quota reset boundaries).
    """
    native = pytest.importorskip("exp_gateway_native")

    inputs = cast("dict[str, JsonObject]", _parity_golden("anthropic_error_inputs"))
    envelopes = cast("dict[str, JsonObject]", _parity_golden("anthropic_error_envelopes"))
    assert set(inputs) == {failure_class.value for failure_class in GatewayFailureClass}
    assert set(envelopes) == set(inputs)

    for failure_class in GatewayFailureClass:
        payload = inputs[failure_class.value]
        translated = json.loads(native.anthropic_error_fixture(json.dumps(payload)))
        assert translated == envelopes[failure_class.value], failure_class
    with_param = cast("JsonObject", _parity_golden("anthropic_error_with_param_input"))
    translated = json.loads(native.anthropic_error_fixture(json.dumps(with_param)))
    assert translated == _parity_golden("anthropic_error_with_param")


def test_rust_messages_body_preserves_interleaved_block_order() -> None:
    """The native body keeps provider block order in the non-streaming shape."""
    native = pytest.importorskip("exp_gateway_native")

    fixture = json.dumps(
        [
            {"kind": "tool_call_started", "index": 0, "call_id": "call-1", "name": "search"},
            {
                "kind": "tool_call_completed",
                "index": 0,
                "call_id": "call-1",
                "name": "search",
                "raw_arguments": "{}",
            },
            {"kind": "text_delta", "text": "after"},
            {"kind": "completed"},
        ]
    )
    actual = native.completed_messages_fixture("request-abc", "coding", fixture)
    assert actual == _parity_golden("messages_block_order_body")
    assert json.loads(actual)["content"][0]["type"] == "tool_use"
    assert json.loads(actual)["content"][1] == {"type": "text", "text": "after"}


def test_rust_messages_deferred_tool_completion_matches_the_committed_goldens() -> None:
    """Deferred completions (OpenAI-compatible [DONE] ordering) hold parity.

    Text arriving between a tool's arguments and its completion must stream
    and aggregate exactly as the committed goldens recorded, with the tool
    block anchored at its start position.
    """
    native = pytest.importorskip("exp_gateway_native")

    fixture = json.dumps(
        [
            {"kind": "tool_call_started", "index": 0, "call_id": "call-1", "name": "search"},
            {"kind": "tool_arguments_delta", "index": 0, "text": "{}"},
            {"kind": "text_delta", "text": "after"},
            {
                "kind": "tool_call_completed",
                "index": 0,
                "call_id": "call-1",
                "name": "search",
                "raw_arguments": "{}",
            },
            {"kind": "completed"},
        ]
    )
    actual_frames = native.encode_messages_fixture("request-abc", "coding", fixture)
    assert list(actual_frames) == _parity_golden("messages_deferred_frames")
    actual_body = native.completed_messages_fixture("request-abc", "coding", fixture)
    assert actual_body == _parity_golden("messages_deferred_body")


def test_rust_messages_interleaved_parallel_tools_match_the_goldens() -> None:
    """Interleaved parallel tool calls stay in byte parity with the goldens.

    The canonical stream may legally interleave tool A arguments, tool B
    start, and more tool A arguments; the encoder must schedule blocks in
    start order, streaming the open block live and buffering the rest.
    """
    native = pytest.importorskip("exp_gateway_native")

    fixture = json.dumps(
        [
            {"kind": "tool_call_started", "index": 0, "call_id": "call-a", "name": "alpha"},
            {"kind": "tool_arguments_delta", "index": 0, "text": '{"a": '},
            {"kind": "tool_call_started", "index": 1, "call_id": "call-b", "name": "beta"},
            {"kind": "tool_arguments_delta", "index": 1, "text": '{"b": 2}'},
            {"kind": "tool_arguments_delta", "index": 0, "text": "1}"},
            {
                "kind": "tool_call_completed",
                "index": 0,
                "call_id": "call-a",
                "name": "alpha",
                "raw_arguments": '{"a": 1}',
            },
            {
                "kind": "tool_call_completed",
                "index": 1,
                "call_id": "call-b",
                "name": "beta",
                "raw_arguments": '{"b": 2}',
            },
            {"kind": "usage", "input_tokens": 6, "output_tokens": 3},
            {"kind": "completed"},
        ]
    )
    actual_frames = native.encode_messages_fixture("request-abc", "coding", fixture)
    assert list(actual_frames) == _parity_golden("messages_interleaved_frames")
    actual_body = native.completed_messages_fixture("request-abc", "coding", fixture)
    assert actual_body == _parity_golden("messages_interleaved_body")


def test_store_false_skips_continuation_retention(tmp_path: Path) -> None:
    """A store:false response is never remembered, so continuing from it fails
    closed with the shared previous_response_not_found error."""
    control, raw_key = _control_plane(tmp_path)
    first = _admit_responses(control, raw_key, _responses_body(store=False))
    assert (
        control.remember(
            json.dumps(
                {
                    "request_id": first["request_id"],
                    "text": "The answer is 42.",
                    "refusal": False,
                    "tool_calls": [],
                }
            )
        )
        == "{}"
    )
    response_id = stable_public_id("resp", _admitted_request_id(first))
    with pytest.raises(NativeBridgeError) as raised:
        _admit_responses(control, raw_key, _responses_body(previous_response_id=response_id))
    payload = json.loads(raised.value.public_error_json)
    assert payload["code"] == "previous_response_not_found"

    # An explicit store:true keeps the default retention behavior.
    stored = _admit_responses(control, raw_key, _responses_body(store=True))
    control.remember(
        json.dumps(
            {
                "request_id": stored["request_id"],
                "text": "kept",
                "refusal": False,
                "tool_calls": [],
            }
        )
    )
    continued = _admit_responses(
        control,
        raw_key,
        _responses_body(
            previous_response_id=stable_public_id("resp", _admitted_request_id(stored))
        ),
    )
    assert [message["role"] for message in _payload_messages(continued)] == [
        "user",
        "assistant",
        "user",
    ]


def _thinking_fixture_json() -> str:
    """Return the Rust fixture-event JSON for the thinking Messages stream."""
    return json.dumps(
        [
            {"kind": "thinking_delta", "index": 0, "text": "Let me "},
            {"kind": "thinking_delta", "index": 0, "text": "check."},
            {"kind": "thinking_signature", "index": 0, "signature": "c2lnbmF0dXJl"},
            {"kind": "redacted_thinking", "index": 1, "data": "b3BhcXVl"},
            {"kind": "text_delta", "text": "Hello"},
            {"kind": "usage", "input_tokens": 12, "output_tokens": 7, "cached_input_tokens": 2},
            {"kind": "completed"},
        ]
    )


def test_rust_messages_thinking_stream_matches_the_hand_authored_goldens() -> None:
    """Thinking blocks stream and aggregate exactly as the Anthropic spec fixes.

    The golden frames were hand-authored against the public Messages
    streaming contract: the thinking block opens with empty fields, streams
    thinking_delta fragments, closes with one signature_delta, redacted
    thinking travels whole in its start frame, and the non-streaming body
    carries the same blocks in order with the byte-exact signature.
    """
    native = pytest.importorskip("exp_gateway_native")

    frames = native.encode_messages_fixture("request-abc", "coding", _thinking_fixture_json())
    assert list(frames) == _parity_golden("messages_thinking_frames")
    body = native.completed_messages_fixture("request-abc", "coding", _thinking_fixture_json())
    assert body == _parity_golden("messages_thinking_body")


def test_rust_responses_encrypted_reasoning_matches_the_hand_authored_golden() -> None:
    """Requested encrypted reasoning lands verbatim on the reasoning item."""
    native = pytest.importorskip("exp_gateway_native")

    fixture = json.dumps(
        [
            {
                "kind": "reasoning_summary_delta",
                "output_index": 0,
                "summary_index": 0,
                "item_id": "rs_01161eec6982f41bdc4271a8fceb6c60",
                "text": "planned",
            },
            {
                "kind": "encrypted_reasoning",
                "output_index": 0,
                "item_id": "rs_01161eec6982f41bdc4271a8fceb6c60",
                "encrypted_content": "ZW5jcnlwdGVk",
            },
            {"kind": "text_delta", "text": "Hello"},
            {
                "kind": "usage",
                "input_tokens": 12,
                "output_tokens": 9,
                "cached_input_tokens": 0,
                "reasoning_tokens": 4,
            },
            {"kind": "completed"},
        ]
    )
    body = native.completed_responses_fixture(
        "request-abc",
        "coding",
        1_700_000_000.0,
        json.dumps({"include_encrypted_reasoning": True}),
        fixture,
    )
    assert body == _parity_golden("responses_encrypted_reasoning_body")


def test_thinking_bytes_round_trip_the_native_pipeline_exactly() -> None:
    """Non-ASCII thinking text and a multi-kilobyte signature survive the full
    provider-frames-to-public-frames pipeline byte-identically.

    The signature is an opaque cryptographic value the provider verifies on
    replay, so any re-encoding drift (Unicode escaping, truncation, split
    handling) would break every continued Claude Code conversation.
    """
    native = pytest.importorskip("exp_gateway_native")

    thinking_one = "Grüß 事實 مرحبا  "
    thinking_two = "🤔🧠 σκέψη ⇒ done"
    signature = "Eq" + "A0b/+=" * 700  # ~4.2 KB, base64-shaped.
    redacted = "R3" * 1500
    provider_chunks = [
        json.dumps({"type": "message_start", "message": {"usage": {"input_tokens": 3}}}),
        json.dumps(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            }
        ),
        json.dumps(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": thinking_one},
            },
            ensure_ascii=False,
        ),
        json.dumps(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": thinking_two},
            },
            ensure_ascii=False,
        ),
        json.dumps(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": signature},
            }
        ),
        json.dumps({"type": "content_block_stop", "index": 0}),
        json.dumps(
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "redacted_thinking", "data": redacted},
            }
        ),
        json.dumps({"type": "content_block_stop", "index": 1}),
        json.dumps(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 9},
            }
        ),
        json.dumps({"type": "message_stop"}),
    ]
    # The fixture boundary carries raw stream bytes as latin-1 code points.
    frames_json = json.dumps(
        [f"data: {chunk}\n\n".encode().decode("latin-1") for chunk in provider_chunks]
    )
    normalized = json.loads(native.normalize_stream_fixture("anthropic_messages", frames_json))
    assert normalized["failure"] is None
    events = normalized["events"]
    streamed_thinking = "".join(
        event["text"] for event in events if event["kind"] == "thinking_delta"
    )
    assert streamed_thinking.encode() == (thinking_one + thinking_two).encode()
    assert [event["signature"] for event in events if event["kind"] == "thinking_signature"] == [
        signature
    ]

    fixture = json.dumps(events, ensure_ascii=False)
    public_frames = native.encode_messages_fixture("request-abc", "coding", fixture)
    payloads = [json.loads(frame.split("data: ", 1)[1].strip()) for frame in public_frames if frame]
    out_thinking = "".join(
        payload["delta"]["thinking"]
        for payload in payloads
        if payload["type"] == "content_block_delta" and payload["delta"]["type"] == "thinking_delta"
    )
    out_signature = "".join(
        payload["delta"]["signature"]
        for payload in payloads
        if payload["type"] == "content_block_delta"
        and payload["delta"]["type"] == "signature_delta"
    )
    assert out_thinking.encode() == (thinking_one + thinking_two).encode()
    assert out_signature.encode() == signature.encode()

    body = json.loads(native.completed_messages_fixture("request-abc", "coding", fixture))
    assert body["content"][0]["thinking"].encode() == (thinking_one + thinking_two).encode()
    assert body["content"][0]["signature"].encode() == signature.encode()
    assert body["content"][1]["data"].encode() == redacted.encode()


def test_encrypted_content_bytes_survive_the_responses_encoder_exactly() -> None:
    """A large opaque encrypted payload lands byte-identical on the public item."""
    native = pytest.importorskip("exp_gateway_native")

    encrypted = "gAAAA" + "Zm9vYmFy+/=" * 1200  # ~13 KB, base64-shaped.
    fixture = json.dumps(
        [
            {
                "kind": "encrypted_reasoning",
                "output_index": 0,
                "item_id": "rs_01161eec6982f41bdc4271a8fceb6c60",
                "encrypted_content": encrypted,
            },
            {"kind": "text_delta", "text": "done ✓"},
            {"kind": "completed"},
        ],
        ensure_ascii=False,
    )
    body = json.loads(
        native.completed_responses_fixture(
            "request-abc",
            "coding",
            1_700_000_000.0,
            json.dumps({"include_encrypted_reasoning": True}),
            fixture,
        )
    )
    assert body["output"][0]["encrypted_content"].encode() == encrypted.encode()

    frames = native.encode_responses_fixture(
        "request-abc",
        "coding",
        1_700_000_000.0,
        json.dumps({"include_encrypted_reasoning": True}),
        fixture,
    )
    done_items = [
        json.loads(frame.split("data: ", 1)[1].strip())
        for frame in frames
        if "response.output_item.done" in frame
    ]
    reasoning_items = [item["item"] for item in done_items if item["item"]["type"] == "reasoning"]
    assert len(reasoning_items) == 1
    assert reasoning_items[0]["encrypted_content"].encode() == encrypted.encode()


def test_keyed_store_false_never_reaches_the_continuation_store(tmp_path: Path) -> None:
    """An Idempotency-Key on a store:false request opens no side door into
    continuation state: the retention callback stays a no-op, the response ID
    resolves to previous_response_not_found in its own namespace, and keyed
    admission replays the operation without manufacturing stored history."""
    control, raw_key = _control_plane(tmp_path)
    body = _responses_body(store=False)
    admission = json.loads(
        control.admit(
            json.dumps(
                {
                    "raw_key": raw_key,
                    "body": body,
                    "surface": "responses",
                    "idempotency_key": "codex-op",
                }
            )
        )
    )
    assert (
        control.remember(
            json.dumps(
                {
                    "request_id": admission["request_id"],
                    "text": "The answer is 42.",
                    "refusal": False,
                    "tool_calls": [],
                }
            )
        )
        == "{}"
    )
    response_id = stable_public_id("resp", _admitted_request_id(admission))
    # Direct store probe in the exact tenant namespace: nothing was retained.
    entry = control._accounting.entry(  # noqa: SLF001 - namespace extraction for the probe.
        _admitted_request_id(admission)
    )
    assert entry is not None and entry.continuation is not None
    assert entry.continuation.retain is False
    with pytest.raises(OpenAIProtocolError) as direct:
        control._continuations.resolve_now(  # noqa: SLF001 - retention isolation assertion.
            namespace=entry.continuation.namespace,
            previous_response_id=response_id,
        )
    assert direct.value.detail.code == "previous_response_not_found"
    # Continuing from the ID through the public path fails closed too, with
    # or without the original caller operation key.
    for key in (None, "codex-op-next"):
        with pytest.raises(NativeBridgeError) as continued:
            control.admit(
                json.dumps(
                    {
                        "raw_key": raw_key,
                        "body": _responses_body(previous_response_id=response_id),
                        "surface": "responses",
                        "idempotency_key": key,
                    }
                )
            )
        assert (
            json.loads(continued.value.public_error_json)["code"] == "previous_response_not_found"
        )


def test_keyed_reasoning_content_joins_replay_identity(tmp_path: Path) -> None:
    """A caller operation key reused with different replayed reasoning is a
    conflict: opaque carriers are digest-excluded for artifact stability, so
    replay identity must fold them back in through canonical_request_sha256."""
    control, raw_key = _control_plane(tmp_path)

    def reasoning_body(encrypted_content: str) -> str:
        """Return one Responses body replaying encrypted reasoning."""
        return json.dumps(
            {
                "model": "coding",
                "input": [
                    {"type": "message", "role": "user", "content": "go"},
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "summary": [],
                        "encrypted_content": encrypted_content,
                    },
                    {"type": "message", "role": "assistant", "content": "done"},
                    {"type": "message", "role": "user", "content": "continue"},
                ],
            }
        )

    def admit_keyed(body: str) -> JsonObject:
        """Admit one keyed Responses request."""
        return json.loads(
            control.admit(
                json.dumps(
                    {
                        "raw_key": raw_key,
                        "body": body,
                        "surface": "responses",
                        "idempotency_key": "reasoning-op",
                    }
                )
            )
        )

    # The seeded route is OpenAI-compatible, so the carrier is rejected at
    # parameter admission; the accepted request still lands a durable keyed
    # terminal, which is exactly what replay identity is checked against.
    with pytest.raises(NativeBridgeError) as first:
        admit_keyed(reasoning_body("blob-one=="))
    assert json.loads(first.value.public_error_json)["code"] == "unsupported_parameter"

    with pytest.raises(NativeBridgeError) as changed:
        admit_keyed(reasoning_body("blob-two=="))
    assert json.loads(changed.value.public_error_json)["code"] == "idempotency_conflict"

    with pytest.raises(NativeBridgeError) as repeated:
        admit_keyed(reasoning_body("blob-one=="))
    assert json.loads(repeated.value.public_error_json)["code"] != "idempotency_conflict"


def test_capability_rejection_names_the_public_request_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-dispatch capability rejection names the exact public field.

    Both the caller's 400 and the ledger row name it: an alert that only
    read "cannot preserve a requested capability" could not be triaged
    without opening the request.
    """
    control, raw_key = _control_plane(tmp_path)
    recorded: list[GatewayFailure] = []
    original_finish = control._accounting.finish_request_quietly  # noqa: SLF001

    def _capture_finish(authorization: AuthorizationSnapshot, failure: GatewayFailure) -> None:
        recorded.append(failure)
        return original_finish(authorization, failure)

    monkeypatch.setattr(control._accounting, "finish_request_quietly", _capture_finish)  # noqa: SLF001
    body = json.dumps(
        {
            "model": "coding",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": "Follow the sync-lane policy canary-instructions.",
                },
                {"type": "message", "role": "user", "content": "hello canary-input"},
            ],
        }
    )
    with pytest.raises(NativeBridgeError) as raised:
        _admit_responses(control, raw_key, body)
    payload = json.loads(raised.value.public_error_json)
    assert payload["status_code"] == 400
    assert payload["code"] == "unsupported_capability"
    assert payload["error_type"] == "invalid_request_error"
    assert payload["param"] == "input.0.role"
    assert "'input.0.role'" in payload["message"]
    assert "developer_messages" not in payload["message"]
    assert "canary" not in json.dumps(payload)
    assert recorded, "the rejection records a durable failure"
    ledger = recorded[-1]
    assert ledger.failure_class == GatewayFailureClass.UNSUPPORTED_CAPABILITY
    assert ledger.safe_message.endswith("(field: input.0.role)")
    assert "developer_messages" not in ledger.safe_message
    assert "canary" not in ledger.safe_message


def test_reasoning_context_reflects_in_the_envelope_only_when_sent() -> None:
    """The response envelope echoes reasoning.context, and only when present,
    so context-free bodies stay byte-identical to the committed goldens."""
    from exp.runtime.gateway.native_responses import responses_envelope

    with_context = decode_responses(
        {
            "model": "coding",
            "input": "hi",
            "reasoning": {"effort": "high", "context": "all_turns"},
        }
    ).request
    assert responses_envelope(with_context)["reasoning"] == {
        "effort": "high",
        "summary": None,
        "context": "all_turns",
    }
    without = decode_responses({"model": "coding", "input": "hi"}).request
    assert responses_envelope(without)["reasoning"] == {"effort": None, "summary": None}


def _zero_argument_tool_fixture_json() -> str:
    """Return the normalized event sequence a zero-argument tool call produces.

    Mirrors the live captured wire (2026-08-28): one empty streamed fragment,
    then the completion-time `{}` seed, then the verified completed call.
    """
    return json.dumps(
        [
            {"kind": "tool_call_started", "index": 0, "call_id": "call-1", "name": "get_time"},
            {"kind": "tool_arguments_delta", "index": 0, "text": ""},
            {"kind": "tool_arguments_delta", "index": 0, "text": "{}"},
            {
                "kind": "tool_call_completed",
                "index": 0,
                "call_id": "call-1",
                "name": "get_time",
                "raw_arguments": "{}",
            },
            {"kind": "usage", "input_tokens": 5, "output_tokens": 3},
            {"kind": "completed"},
        ]
    )


def test_zero_argument_tool_calls_encode_on_every_public_lane() -> None:
    """The zero-argument completion sequence serves both lanes, both modes.

    Production incident (2026-08-28): every zero-argument tool failed as
    malformed_response. The normalizer fix seeds `{}` at completion; these
    assertions pin that the seeded sequence encodes as a valid Anthropic
    tool_use block and a valid Chat tool call, streaming and non-streaming.
    """
    native = pytest.importorskip("exp_gateway_native")
    fixture = _zero_argument_tool_fixture_json()

    frames = native.encode_messages_fixture("request-abc", "coding", fixture)
    assert any('"type":"tool_use"' in frame for frame in frames)
    assert frames[-1].startswith("event: message_stop")
    streamed_input = "".join(
        payload["delta"]["partial_json"]
        for payload in (
            json.loads(frame.split("data: ", 1)[1].strip()) for frame in frames if frame
        )
        if payload["type"] == "content_block_delta"
        and payload["delta"]["type"] == "input_json_delta"
    )
    assert streamed_input == "{}"
    messages_body = json.loads(native.completed_messages_fixture("request-abc", "coding", fixture))
    assert messages_body["content"][0] == {
        "type": "tool_use",
        "id": "call-1",
        "name": "get_time",
        "input": {},
    }
    assert messages_body["stop_reason"] == "tool_use"

    chat_frames = native.encode_chat_fixture("request-abc", "coding", 1_700_000_000, True, fixture)
    assert chat_frames[-1] == "data: [DONE]\n\n"
    assert any('"finish_reason":"tool_calls"' in frame for frame in chat_frames)
    streamed_arguments = "".join(
        call["function"]["arguments"]
        for payload in (
            json.loads(frame.split("data: ", 1)[1].strip())
            for frame in chat_frames
            if frame.startswith("data: {")
        )
        for choice in payload.get("choices", ())
        for call in choice.get("delta", {}).get("tool_calls", ())
    )
    assert streamed_arguments == "{}"
    responses_body = json.loads(
        native.completed_responses_fixture("request-abc", "coding", 1_700_000_000.0, "{}", fixture)
    )
    call_items = [item for item in responses_body["output"] if item["type"] == "function_call"]
    assert call_items[0]["arguments"] == "{}"
    assert call_items[0]["status"] == "completed"


def test_strict_tools_degrade_with_disclosure_when_no_rung_declares_them(
    tmp_path: Path,
) -> None:
    """A strict tool on a route with no strict-capable rung serves degraded.

    Production shape (org 9dd93c55): synced catalog rows declared
    supports_strict_tools nowhere in the waterfall, so every strict-tool
    request failed closed. Preference for a verbatim rung stays first; the
    disclosed degrade applies only when zero rungs qualify, and the caller
    sees exactly what happened.
    """
    control, raw_key = _control_plane(tmp_path)
    body = json.dumps(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "look it up"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object"},
                        "strict": True,
                    },
                }
            ],
        }
    )
    admission = _flatten_started(control, _admit(control, raw_key, body))
    assert admission["ignored_parameters"] == ["tools.strict->false"]
    upstream = admission["upstream_payload"]
    assert isinstance(upstream, dict)
    tools = cast("list[JsonObject]", upstream["tools"])
    function = cast("JsonObject", tools[0]["function"])
    assert function["strict"] is False
    control_plane = cast("JsonObject", control.metrics_snapshot()["control_plane"])
    assert control_plane["admission_parameter_coercions"] == 1


def test_any_effort_drops_with_disclosure_on_a_reasoning_less_route(
    tmp_path: Path,
) -> None:
    """Every effort level is dropped with disclosure by a non-reasoning route.

    The kimi-k3 shape dropped only an explicit 'none'; the haiku-4.5 shape
    proved a real effort must drop too. First-party clients pin effort
    globally (Claude Code sends its configured effortLevel to every model),
    so a named rejection made whole sessions unusable against non-reasoning
    models the provider itself serves fine without the parameter (owner
    decision, 2026-09-01).
    """
    control, raw_key = _control_plane(tmp_path)

    def chat_body(effort: str) -> str:
        """Return one Chat body carrying the given reasoning effort."""
        return json.dumps(
            {
                "model": "coding",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": effort,
            }
        )

    for attempt, effort in enumerate(("none", "high"), start=1):
        admission = _flatten_started(control, _admit(control, raw_key, chat_body(effort)))
        assert admission["ignored_parameters"] == ["reasoning_effort"], effort
        upstream = admission["upstream_payload"]
        assert isinstance(upstream, dict)
        assert "reasoning_effort" not in upstream
        assert "reasoning" not in upstream
        control_plane = cast("JsonObject", control.metrics_snapshot()["control_plane"])
        assert control_plane["admission_parameter_coercions"] == attempt


def test_effort_carrying_marked_request_serves_native_with_caching_intact(
    tmp_path: Path,
) -> None:
    """The haiku-4.5 regression: effort drops, cache markers reach the wire.

    A Claude Code session pinning effortLevel against a non-reasoning
    Anthropic model must serve on the native rung with its prompt-cache
    markers preserved and the dropped effort disclosed, not 400 and not
    narrow onto a marker-dropping shim.
    """
    root = tmp_path / "anthropic-root"
    root.mkdir()
    _manager, raw_key = _configured_gateway(root, provider="anthropic")
    components = load_gateway_components(
        root,
        environment={"TEST_PROVIDER_KEY": "provider-secret-canary"},
    )
    control = NativeControlPlane(components, request_timeout_seconds=120.0)
    body = json.dumps(
        {
            "model": "coding",
            "max_tokens": 32,
            "system": [
                {"type": "text", "text": "You are terse."},
                {
                    "type": "text",
                    "text": "Big cached block.",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"effort": "high"},
        }
    )
    admission = _flatten_started(control, _admit(control, raw_key, body, surface="messages"))
    assert admission["ignored_parameters"] == ["reasoning_effort"]
    upstream = admission["upstream_payload"]
    assert isinstance(upstream, dict)
    # The dropped effort reaches the provider through NO channel.
    assert "output_config" not in upstream
    assert "reasoning_effort" not in upstream
    # The cache markers survive to the native wire, block structure intact.
    system = cast("list[JsonObject]", upstream["system"])
    assert system[-1]["cache_control"] == {"type": "ephemeral"}


def test_adaptive_thinking_drops_with_the_effort_on_a_reasoning_less_route(
    tmp_path: Path,
) -> None:
    """Claude Code pins thinking adaptive with effortLevel; both drop, disclosed.

    The effort alone dropping left the adaptive object on the wire, and a
    non-reasoning Anthropic rung answers that with a post-dispatch 400 the
    caller cannot act on. A budgeted thinking config carries its own
    semantics and still travels verbatim.
    """
    root = tmp_path / "anthropic-root"
    root.mkdir()
    _manager, raw_key = _configured_gateway(root, provider="anthropic")
    components = load_gateway_components(
        root,
        environment={"TEST_PROVIDER_KEY": "provider-secret-canary"},
    )
    control = NativeControlPlane(components, request_timeout_seconds=120.0)
    body = json.dumps(
        {
            "model": "coding",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
        }
    )
    admission = _flatten_started(control, _admit(control, raw_key, body, surface="messages"))
    assert admission["ignored_parameters"] == ["reasoning_effort", "thinking"]
    upstream = admission["upstream_payload"]
    assert isinstance(upstream, dict)
    assert "thinking" not in upstream
    assert "output_config" not in upstream

    budgeted = json.dumps(
        {
            "model": "coding",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "output_config": {"effort": "high"},
        }
    )
    admission = _flatten_started(control, _admit(control, raw_key, budgeted, surface="messages"))
    assert admission["ignored_parameters"] == ["reasoning_effort"]
    upstream = admission["upstream_payload"]
    assert isinstance(upstream, dict)
    assert upstream["thinking"] == {"type": "enabled", "budget_tokens": 1024}


def test_open_response_format_schema_closes_on_an_anthropic_rung(tmp_path: Path) -> None:
    """A Chat response_format schema without additionalProperties false serves.

    The Anthropic validator rejects open objects that OpenAI accepts, so the
    gateway closes every object on the wire and discloses the tightening
    instead of relaying a post-dispatch 400.
    """
    root = tmp_path / "anthropic-root"
    root.mkdir()
    manager, raw_key = _configured_gateway(root, provider="anthropic")
    normalized, snapshot, _changed = upsert_singleton_deployment(
        root,
        deployment_alias="structured",
        connection_name="provider-main",
        provider_model="provider-model-structured",
        exact_model_id="structured-revision-exact",
        revision=None,
        capabilities=ModelCapabilities(supports_structured_output=True),
        gateway_capabilities=GatewayDeploymentCapabilities(
            supports_streaming=True,
            supports_structured_text=True,
        ),
        prices=GatewayTokenPrices(),
        pricing_source=None,
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id="structured",
        alias_name="structured",
        revision_id="revision-structured",
        pool_id="structured",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.add_grant(identity_id="default", alias_id="structured")
    components = load_gateway_components(
        root,
        environment={"TEST_PROVIDER_KEY": "provider-secret-canary"},
    )
    control = NativeControlPlane(components, request_timeout_seconds=120.0)
    body = json.dumps(
        {
            "model": "structured",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "geo": {
                                "type": "object",
                                "properties": {"lat": {"type": "number"}},
                            },
                        },
                        "required": ["city", "geo"],
                    },
                },
            },
        }
    )
    admission = _flatten_started(control, _admit(control, raw_key, body))
    assert admission["ignored_parameters"] == ["json_schema.additionalProperties->false"]
    upstream = admission["upstream_payload"]
    assert isinstance(upstream, dict)
    output_config = cast("JsonObject", upstream["output_config"])
    schema_format = cast("JsonObject", output_config["format"])
    schema = cast("JsonObject", schema_format["schema"])
    assert schema["additionalProperties"] is False
    properties = cast("JsonObject", schema["properties"])
    geo = cast("JsonObject", properties["geo"])
    assert geo["additionalProperties"] is False
    assert schema["required"] == ["city", "geo"]


def _web_search_fixture_json() -> str:
    """One WebSearch event stream in the fixture-event vocabulary."""
    result_block = (
        '{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1",'
        '"content":[{"type":"web_search_result","encrypted_content":"Et8Q"}],'
        '"caller":{"type":"direct"}}'
    )
    citation = '{"type":"web_search_result_location","cited_text":"3.14.7"}'
    return json.dumps(
        [
            {
                "kind": "server_tool_use_started",
                "index": 0,
                "call_id": "srvtoolu_1",
                "name": "web_search",
            },
            {"kind": "server_tool_arguments_delta", "index": 0, "text": '{"query": "python"}'},
            {
                "kind": "server_tool_use_completed",
                "index": 0,
                "call_id": "srvtoolu_1",
                "name": "web_search",
                "raw_arguments": '{"query": "python"}',
            },
            {"kind": "server_tool_result", "index": 1, "block": result_block},
            {"kind": "text_block_started", "index": 2},
            {"kind": "citation_delta", "index": 2, "citation": citation},
            {"kind": "text_delta", "text": "It is 3.14.7."},
            {"kind": "usage", "input_tokens": 12284, "output_tokens": 103},
            {"kind": "completed"},
        ]
    )


def test_rust_messages_streams_server_tool_blocks_intact() -> None:
    """Server tool events stream back as their native Anthropic blocks."""
    native = pytest.importorskip("exp_gateway_native")

    frames = list(
        native.encode_messages_fixture("request-abc", "coding", _web_search_fixture_json())
    )
    joined = "".join(frames)
    assert '"type":"server_tool_use","id":"srvtoolu_1","name":"web_search"' in joined
    assert '"type":"web_search_tool_result"' in joined
    assert '"caller":{"type":"direct"}' in joined
    assert '"type":"citations_delta"' in joined
    # Provider-executed tool use never becomes the tool_use stop reason.
    assert '"stop_reason":"end_turn"' in joined


def test_rust_messages_completed_body_carries_server_tool_blocks() -> None:
    """The non-streaming aggregation keeps every server-tool block in order."""
    native = pytest.importorskip("exp_gateway_native")

    body = json.loads(
        native.completed_messages_fixture("request-abc", "coding", _web_search_fixture_json())
    )
    kinds = [block["type"] for block in body["content"]]
    assert kinds == ["server_tool_use", "web_search_tool_result", "text"]
    assert body["content"][2]["citations"] == [
        {"type": "web_search_result_location", "cited_text": "3.14.7"}
    ]
    assert body["stop_reason"] == "end_turn"


def test_rust_messages_paused_turn_keeps_its_stop_reason() -> None:
    """A pause_turn terminal survives to the caller instead of end_turn."""
    native = pytest.importorskip("exp_gateway_native")

    fixture = json.dumps(
        [
            {"kind": "text_delta", "text": "searching"},
            {"kind": "paused_turn"},
        ]
    )
    frames = "".join(native.encode_messages_fixture("request-abc", "coding", fixture))
    assert '"stop_reason":"pause_turn"' in frames
    body = json.loads(native.completed_messages_fixture("request-abc", "coding", fixture))
    assert body["stop_reason"] == "pause_turn"


def test_internal_admission_failures_log_the_real_exception(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sanitized 500 hides the cause everywhere else, so the bridge logs it once.

    A granted alias whose admission blew up with an unexpected exception used
    to fail 500 with the traceback recorded nowhere (public error, ledger row,
    and worker log all carry only the sanitized text). The record names the
    request and alias and carries the exception on ``exc_info``.
    """
    control, raw_key = _pool_control_plane(tmp_path)

    def explode(*args: object, **kwargs: object) -> object:
        del args, kwargs
        message = "hydrated snapshot names a deployment this worker never registered"
        raise KeyError(message)

    monkeypatch.setattr(native_bridge_module, "resolve_admission_route", explode)
    body = json.dumps({"model": "coding", "messages": [{"role": "user", "content": "hi"}]})

    with (
        caplog.at_level(logging.ERROR, logger="exp.runtime.gateway.native_bridge"),
        pytest.raises(NativeBridgeError) as raised,
    ):
        _admit(control, raw_key, body)

    error = json.loads(raised.value.public_error_json)
    assert error["status_code"] == 500
    assert error["code"] == "internal_error"
    records = [
        record
        for record in caplog.records
        if record.getMessage() == "gateway admission failed before provider dispatch"
    ]
    assert len(records) == 1
    record = records[0]
    assert record.exc_info is not None
    assert record.exc_info[0] is KeyError
    # ``extra`` fields land on the record's namespace, not on the typed class.
    fields = record.__dict__
    assert fields["alias"] == "coding"
    assert str(fields["request_id"]).startswith("request-")
    assert fields["exception_type"] == "KeyError"
    assert fields["operation"] == "native_admit"


def _affinity_pool_control_plane(root: Path) -> tuple[NativeControlPlane, str, Path]:
    """Load the control plane over a pool opted into cache-affinity routing.

    Seeds the standard certified two-deployment pool, then authors the opt-in
    the way the hosted platform does: the pool's ``failover_mode`` flips to
    ``maximize_cache_affinity`` and each rung carries an affinity weight in
    its dispatch policy, all as catalog data behind a fresh alias revision.
    """
    from exp.common.models.catalog import (
        GatewayRungDispatchPolicy,
        load_model_catalog,
        write_model_catalog,
    )
    from exp.runtime.gateway.catalog_authority import snapshot_current_catalog

    manager, raw_key = _configured_pool_gateway(root)
    catalog_path = root / "models.toml"
    catalog = load_model_catalog(catalog_path)
    weighted_models = dict(catalog.models)
    for alias, weight in (("alpha", 1.0), ("beta", 6.0)):
        record = weighted_models[alias]
        assert record.gateway is not None
        weighted_models[alias] = record.model_copy(
            update={
                "gateway": record.gateway.model_copy(
                    update={"dispatch": GatewayRungDispatchPolicy(affinity_weight=weight)}
                )
            }
        )
    pool = catalog.gateway_pools["coding"].model_copy(
        update={"failover_mode": "maximize_cache_affinity"}
    )
    write_model_catalog(
        catalog_path,
        catalog.model_copy(update={"models": weighted_models, "gateway_pools": {"coding": pool}}),
    )
    _catalog, normalized, snapshot = snapshot_current_catalog(root)
    manager.activate_direct_alias(
        alias_id="coding",
        alias_name="coding",
        revision_id="revision-pool-affinity",
        pool_id="coding",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    components = load_gateway_components(
        root,
        environment={"TEST_PROVIDER_KEY": "provider-secret-canary"},
    )
    return NativeControlPlane(components), raw_key, manager.database_path


def test_affinity_pool_routes_each_session_deterministically(tmp_path: Path) -> None:
    """One session always admits the same rendezvous ladder, disclosed as such.

    Three admissions of one session id produce byte-identical route orders,
    distinct sessions reach distinct first rungs (the whole point of spreading
    by fingerprint), the 6x-weighted rung carries the clear majority, and the
    reserved first dispatch lands durable ``dispatch_reason='affinity'``.
    """
    import sqlite3

    control, raw_key, database_path = _affinity_pool_control_plane(tmp_path)
    body = json.dumps({"model": "coding", "messages": [{"role": "user", "content": "hi"}]})

    def admitted_order(session: str) -> tuple[str, ...]:
        """Admit one request under a session id and name its rung order."""
        admission = _admit(control, raw_key, body, client_request_id=session)
        route = admission["route"]
        assert isinstance(route, list)
        return tuple(str(cast("JsonObject", entry)["deployment_id"]) for entry in route)

    assert len({admitted_order("session-pinned") for _ in range(3)}) == 1
    first_rungs = [admitted_order(f"session-{index}")[0] for index in range(24)]
    assert set(first_rungs) == {"alpha", "beta"}
    assert first_rungs.count("beta") > first_rungs.count("alpha")

    admission = _admit(control, raw_key, body, client_request_id="session-disclosed")
    started = _start_first(control, admission)
    assert started["route_depth"] == 0
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT dispatch_reason, preferred_deployment_id FROM gateway_attempts"
            " WHERE attempt_id = ?",
            (str(started["attempt_id"]),),
        ).fetchone()
    assert row == ("affinity", None)
