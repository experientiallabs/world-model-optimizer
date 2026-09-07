"""Tests for immutable gateway contracts shared by parallel implementation lanes."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import JsonObject
from exp.common.models.model import ToolCall
from exp.runtime.gateway.compatibility import (
    CompatibilityDisposition,
    CompatibilityField,
    CompatibilityManifest,
)
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    ExecutionSnapshot,
    GatewayApiSurface,
    GatewayEvent,
    GatewayEventKind,
    GatewayMessage,
    GatewayNamedToolChoice,
    GatewayProviderNativeTool,
    GatewayRequest,
    GatewayToolDefinition,
    ProjectTarget,
    StructuredTextFormat,
)


def test_gateway_request_preserves_developer_and_raw_tool_history() -> None:
    """Canonical requests retain role identity, strict schemas, and provider-order arguments."""
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(
            GatewayMessage(role="developer", content="Use the support policy."),
            GatewayMessage(role="user", content="Look up account 7."),
            GatewayMessage(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        call_id="call-1",
                        name="lookup",
                        arguments={"account": 7},
                        raw_arguments='{ "account": 7 }',
                    ),
                ),
            ),
            GatewayMessage(role="tool", content="active", tool_call_id="call-1"),
        ),
        tools=(
            GatewayToolDefinition(
                name="lookup",
                parameters={"type": "object"},
                strict=True,
            ),
        ),
        tool_choice=GatewayNamedToolChoice(name="lookup"),
        parallel_tool_calls=True,
        stream=True,
        include_usage=True,
    )

    assert request.messages[0].role == "developer"
    assert request.messages[2].tool_calls[0].arguments_json() == '{ "account": 7 }'
    restored = GatewayRequest.model_validate_json(request.model_dump_json())
    assert restored.messages[2].tool_calls[0].raw_arguments is None
    assert restored.model_dump(mode="json") == request.model_dump(mode="json")


def test_raw_tool_argument_delta_accepts_non_json_fragments_in_order() -> None:
    """Streaming contracts carry arbitrary raw fragments before terminal JSON validation."""
    first = GatewayEvent(
        kind=GatewayEventKind.TOOL_ARGUMENTS_DELTA,
        sequence_number=3,
        tool_call_index=0,
        raw_arguments_delta='{"acc',
    )
    second = GatewayEvent(
        kind=GatewayEventKind.TOOL_ARGUMENTS_DELTA,
        sequence_number=4,
        tool_call_index=0,
        raw_arguments_delta='ount":7}',
    )

    assert first.raw_arguments_delta is not None
    assert second.raw_arguments_delta is not None
    assert first.raw_arguments_delta + second.raw_arguments_delta == '{"account":7}'
    with pytest.raises(ValidationError, match="raw fragment"):
        GatewayEvent(
            kind=GatewayEventKind.TOOL_ARGUMENTS_DELTA,
            sequence_number=5,
            tool_call_index=0,
        )


def test_targets_and_compatibility_manifest_are_closed_and_deterministic() -> None:
    """Target variants discriminate cleanly and public field decisions cannot conflict."""
    target = DirectTarget(pool_id="coding")
    manifest = CompatibilityManifest(
        schema_version=1,
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        fields=(
            CompatibilityField(
                field_path="messages",
                disposition=CompatibilityDisposition.SUPPORTED,
            ),
        ),
    )

    assert target.kind == "direct"
    assert CompatibilityManifest.model_validate_json(manifest.model_dump_json()) == manifest
    with pytest.raises(ValidationError, match="must be unique"):
        CompatibilityManifest(
            schema_version=1,
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            fields=(manifest.fields[0], manifest.fields[0]),
        )


def test_project_authorization_precedes_route_bound_execution() -> None:
    """Authorization can freeze a target before learned selection supplies route identity."""
    authorization = AuthorizationSnapshot(
        request_id="request-1",
        organization_id="organization-1",
        identity_id="identity-1",
        virtual_key_id="key-1",
        alias="coding",
        alias_revision_id="alias-revision-1",
        target=ProjectTarget(
            project_ref="support-agent",
            activation_ref="activation-1",
            catalog_sha256="a" * 64,
        ),
        surface=GatewayApiSurface.RESPONSES,
        catalog_sha256="a" * 64,
        canonical_request_sha256="b" * 64,
        caller_operation_sha256="c" * 64,
        deadline_monotonic=10.0,
    )

    execution = ExecutionSnapshot(
        authorization=authorization,
        exact_model_id="exact-model-1",
        pool_id="pool-1",
        deployment_ids=("deployment-1",),
    )

    assert authorization.target.kind == "project"
    assert authorization.surface is GatewayApiSurface.RESPONSES
    assert authorization.caller_operation_sha256 == "c" * 64
    assert execution.authorization == authorization


def test_tool_error_marker_never_reaches_serialization_or_request_digests() -> None:
    """tool_is_error is authority-visible only, like ToolCall.raw_arguments.

    The canonical request digest anchors replay identity and immutable
    artifacts, so a request carrying the marker must serialize and digest
    byte-identically to the same request without it, and to requests decoded
    before the field existed.
    """
    from exp.common.core.artifacts import sha256_json
    from exp.runtime.openai_protocol.requests import decode_chat

    def request(*, tool_is_error: bool) -> GatewayRequest:
        """Build one tool-continuation request with the marker toggled."""
        return GatewayRequest(
            surface=GatewayApiSurface.MESSAGES,
            messages=(
                GatewayMessage(
                    role="tool",
                    content="boom",
                    tool_call_id="call-1",
                    tool_is_error=tool_is_error,
                ),
            ),
        )

    flagged = request(tool_is_error=True)
    plain = request(tool_is_error=False)
    assert flagged.model_dump(mode="json") == plain.model_dump(mode="json")
    assert "tool_is_error" not in flagged.messages[0].model_dump(mode="json")
    assert sha256_json(flagged) == sha256_json(plain)

    # OpenAI callers cannot express the marker; decode_chat never sets it.
    chat = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "search", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "boom"},
            ],
        }
    )
    assert chat.request.messages[-1].tool_is_error is False

    with pytest.raises(ValidationError, match="tool_is_error is valid only for tool messages"):
        GatewayMessage(role="user", content="hi", tool_is_error=True)


def test_provider_reasoning_carrier_is_ordered_assistant_only_and_digest_free() -> None:
    """Opaque reasoning blocks ride assistant turns without perturbing digests."""
    from exp.common.core.artifacts import sha256_json
    from exp.runtime.gateway.contracts import (
        EncryptedReasoningBlock,
        RedactedThinkingBlock,
        ThinkingBlock,
    )

    blocks = (
        ThinkingBlock(text="step one", signature="sig-1"),
        RedactedThinkingBlock(data="opaque"),
    )
    carried = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="assistant", content="done", provider_reasoning=blocks),),
    )
    bare = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="assistant", content="done"),),
    )
    assert carried.messages[0].provider_reasoning == blocks
    assert carried.model_dump(mode="json") == bare.model_dump(mode="json")
    assert sha256_json(carried) == sha256_json(bare)
    # Plain digests stay byte-identical (immutable artifacts, pre-carrier
    # requests), but replay identity distinguishes reasoning content so a
    # reused caller operation key with different reasoning conflicts.
    from exp.runtime.gateway.replay_identity import canonical_request_sha256

    assert canonical_request_sha256(bare) == sha256_json(bare)
    assert canonical_request_sha256(carried) != canonical_request_sha256(bare)
    recarried = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="assistant", content="done", provider_reasoning=blocks),),
    )
    assert canonical_request_sha256(recarried) == canonical_request_sha256(carried)
    changed = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(
            GatewayMessage(
                role="assistant",
                content="done",
                provider_reasoning=(ThinkingBlock(text="step one", signature="sig-2"),),
            ),
        ),
    )
    assert canonical_request_sha256(changed) != canonical_request_sha256(carried)

    # A thinking-only assistant turn is legal history (e.g. a turn cut off
    # mid-thinking), so the carrier alone satisfies message coherence.
    reasoning_only = GatewayMessage(
        role="assistant",
        provider_reasoning=(EncryptedReasoningBlock(id="rs_1", encrypted_content="blob"),),
    )
    assert reasoning_only.content is None

    with pytest.raises(ValidationError, match="valid only for assistant messages"):
        GatewayMessage(
            role="user",
            content="hi",
            provider_reasoning=(ThinkingBlock(text="x", signature=None),),
        )


def test_reasoning_carrier_request_fields_are_surface_scoped() -> None:
    """store, include, and verbatim thinking config bind to their one surface."""
    messages = (GatewayMessage(role="user", content="hi"),)
    stored = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=messages,
        response_store=False,
        include_encrypted_reasoning=True,
    )
    assert stored.response_store is False
    assert stored.include_encrypted_reasoning is True

    thinking = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=messages,
        provider_thinking_config={"type": "enabled", "budget_tokens": 2048},
    )
    assert thinking.provider_thinking_config == {"type": "enabled", "budget_tokens": 2048}
    # The verbatim config is authority-visible only, never in digests.
    assert "provider_thinking_config" not in thinking.model_dump(mode="json")

    with pytest.raises(ValidationError, match="response_store is valid only"):
        GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=messages,
            response_store=True,
        )
    with pytest.raises(ValidationError, match="include_encrypted_reasoning is valid only"):
        GatewayRequest(
            surface=GatewayApiSurface.MESSAGES,
            messages=messages,
            include_encrypted_reasoning=True,
        )
    with pytest.raises(ValidationError, match="provider_thinking_config is valid only"):
        GatewayRequest(
            surface=GatewayApiSurface.RESPONSES,
            messages=messages,
            provider_thinking_config={"type": "enabled"},
        )


def test_provider_replay_identity_hashes_exact_tool_and_message_state() -> None:
    """Excluded wire identity and raw argument bytes still bind idempotent replay."""
    from exp.runtime.gateway.replay_identity import canonical_request_sha256

    def request(*, item_id: str, raw_arguments: str, output_index: int = 2) -> GatewayRequest:
        return GatewayRequest(
            surface=GatewayApiSurface.RESPONSES,
            messages=(
                GatewayMessage(
                    role="assistant",
                    content="preamble",
                    provider_item_id="msg_1",
                    provider_output_index=1,
                    tool_calls=(
                        ToolCall(
                            call_id="call-1",
                            name="lookup",
                            arguments={"q": "x"},
                            raw_arguments=raw_arguments,
                            provider_item_id=item_id,
                            provider_output_index=output_index,
                        ),
                    ),
                ),
            ),
        )

    original = request(item_id="fc_1", raw_arguments='{"q":"x"}')
    assert canonical_request_sha256(original) == canonical_request_sha256(
        request(item_id="fc_1", raw_arguments='{"q":"x"}')
    )
    assert canonical_request_sha256(original) != canonical_request_sha256(
        request(item_id="fc_2", raw_arguments='{"q":"x"}')
    )
    assert canonical_request_sha256(original) != canonical_request_sha256(
        request(item_id="fc_1", raw_arguments='{ "q" : "x" }')
    )
    assert canonical_request_sha256(original) != canonical_request_sha256(
        request(item_id="fc_1", raw_arguments='{"q":"x"}', output_index=3)
    )


def test_provider_replay_identity_hashes_status_phase_and_idless_call_order() -> None:
    """Excluded OpenAI item fields remain authenticated canonical authority."""
    from exp.runtime.gateway.replay_identity import canonical_request_sha256

    def request(
        *,
        message_status: Literal["in_progress", "completed", "incomplete"] = "incomplete",
        phase: Literal["commentary", "final_answer"] = "commentary",
        call_status: Literal["in_progress", "completed", "incomplete"] = "incomplete",
    ) -> GatewayRequest:
        return GatewayRequest(
            surface=GatewayApiSurface.RESPONSES,
            messages=(
                GatewayMessage(
                    role="assistant",
                    content="checking",
                    provider_item_id="msg-commentary",
                    provider_output_index=1,
                    provider_status=message_status,
                    provider_phase=phase,
                    tool_calls=(
                        ToolCall(
                            call_id="call-required",
                            name="lookup",
                            arguments={},
                            raw_arguments="{}",
                            provider_output_index=2,
                            provider_status=call_status,
                        ),
                    ),
                ),
            ),
        )

    original = request()
    other_phase = request(phase="final_answer")
    other_message_status = request(message_status="completed")
    other_call_status = request(call_status="completed")
    assert original.model_dump(mode="json") == other_phase.model_dump(mode="json")
    assert canonical_request_sha256(original) != canonical_request_sha256(other_phase)
    assert canonical_request_sha256(original) != canonical_request_sha256(other_message_status)
    assert canonical_request_sha256(original) != canonical_request_sha256(other_call_status)


def test_reasoning_stream_events_require_their_payloads() -> None:
    """Each new reasoning event kind carries its block index and payload."""
    thinking = GatewayEvent(
        kind=GatewayEventKind.THINKING_DELTA,
        sequence_number=0,
        reasoning_block_index=0,
        text_delta="because",
    )
    assert thinking.reasoning_block_index == 0
    signature = GatewayEvent(
        kind=GatewayEventKind.THINKING_SIGNATURE,
        sequence_number=1,
        reasoning_block_index=0,
        thinking_signature="sig",
    )
    assert signature.thinking_signature == "sig"
    redacted = GatewayEvent(
        kind=GatewayEventKind.REDACTED_THINKING,
        sequence_number=2,
        reasoning_block_index=1,
        redacted_thinking_data="opaque",
    )
    assert redacted.redacted_thinking_data == "opaque"
    encrypted = GatewayEvent(
        kind=GatewayEventKind.ENCRYPTED_REASONING,
        sequence_number=3,
        reasoning_block_index=0,
        reasoning_item_id="rs_1",
        encrypted_content="blob",
    )
    assert encrypted.encrypted_content == "blob"

    with pytest.raises(ValidationError, match="thinking deltas require"):
        GatewayEvent(kind=GatewayEventKind.THINKING_DELTA, sequence_number=0, text_delta="x")
    with pytest.raises(ValidationError, match="thinking signatures require"):
        GatewayEvent(
            kind=GatewayEventKind.THINKING_SIGNATURE,
            sequence_number=0,
            reasoning_block_index=0,
        )
    with pytest.raises(ValidationError, match="redacted thinking requires"):
        GatewayEvent(
            kind=GatewayEventKind.REDACTED_THINKING,
            sequence_number=0,
            reasoning_block_index=0,
        )
    with pytest.raises(ValidationError, match="encrypted reasoning requires"):
        GatewayEvent(
            kind=GatewayEventKind.ENCRYPTED_REASONING,
            sequence_number=0,
            encrypted_content="blob",
        )


def test_reasoning_context_is_digest_excluded_but_joins_replay_identity() -> None:
    """Context-free requests digest byte-identically to pre-field traffic.

    The field is excluded from model serialization so this release does not
    move canonical digests for context-free traffic; a present value folds
    into replay identity so a reused caller operation key with a different
    context is a conflict, never a silent replay.
    """
    from exp.common.core.artifacts import sha256_json
    from exp.runtime.gateway.replay_identity import canonical_request_sha256

    messages = (GatewayMessage(role="user", content="hi"),)
    bare = GatewayRequest(surface=GatewayApiSurface.RESPONSES, messages=messages)
    carried = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=messages,
        reasoning_context="all_turns",
    )
    assert carried.model_dump(mode="json") == bare.model_dump(mode="json")
    assert sha256_json(carried) == sha256_json(bare)
    assert canonical_request_sha256(bare) == sha256_json(bare)
    assert canonical_request_sha256(carried) != canonical_request_sha256(bare)
    other = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=messages,
        reasoning_context="current_turn",
    )
    assert canonical_request_sha256(other) != canonical_request_sha256(carried)

    with pytest.raises(ValidationError, match="reasoning_context is valid only"):
        GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=messages,
            reasoning_context="all_turns",
        )


def test_context_management_is_digest_excluded_but_joins_replay_identity() -> None:
    """Config-free requests digest byte-identically to pre-field traffic."""
    from exp.common.core.artifacts import sha256_json
    from exp.runtime.gateway.replay_identity import canonical_request_sha256

    messages = (GatewayMessage(role="user", content="hi"),)
    bare = GatewayRequest(surface=GatewayApiSurface.MESSAGES, messages=messages)
    carried = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=messages,
        context_management={"edits": [{"type": "clear_tool_uses_20250919"}]},
    )
    assert carried.model_dump(mode="json") == bare.model_dump(mode="json")
    assert sha256_json(carried) == sha256_json(bare)
    assert canonical_request_sha256(bare) == sha256_json(bare)
    assert canonical_request_sha256(carried) != canonical_request_sha256(bare)

    with pytest.raises(ValidationError, match="context_management is valid only"):
        GatewayRequest(
            surface=GatewayApiSurface.RESPONSES,
            messages=messages,
            context_management={"edits": []},
        )


def test_anthropic_tool_annotations_are_digest_free_but_bind_replay_identity() -> None:
    """Tool carriers never perturb plain digests; present ones bind replay."""
    from exp.common.core.artifacts import sha256_json
    from exp.runtime.gateway.contracts import GatewayToolDefinition
    from exp.runtime.gateway.replay_identity import canonical_request_sha256

    def request(tool: GatewayToolDefinition) -> GatewayRequest:
        return GatewayRequest(
            surface=GatewayApiSurface.MESSAGES,
            messages=(GatewayMessage(role="user", content="hi"),),
            tools=(tool,),
        )

    plain_tool = GatewayToolDefinition(name="bash", parameters={"type": "object"})
    bare = request(plain_tool)
    eager = request(plain_tool.model_copy(update={"eager_input_streaming": True}))
    assert eager.model_dump(mode="json") == bare.model_dump(mode="json")
    assert sha256_json(eager) == sha256_json(bare)
    assert canonical_request_sha256(bare) == sha256_json(bare)
    assert canonical_request_sha256(eager) != canonical_request_sha256(bare)
    assert canonical_request_sha256(eager) == canonical_request_sha256(
        request(plain_tool.model_copy(update={"eager_input_streaming": True}))
    )
    assert canonical_request_sha256(eager) != canonical_request_sha256(
        request(plain_tool.model_copy(update={"eager_input_streaming": False}))
    )
    for annotated in (
        plain_tool.model_copy(update={"defer_loading": False}),
        plain_tool.model_copy(update={"allowed_callers": ("code_execution_20260120",)}),
        plain_tool.model_copy(update={"input_examples": ({"city": "Paris"},)}),
    ):
        assert canonical_request_sha256(request(annotated)) != canonical_request_sha256(bare)

    # A cache hint changes cost, not semantics: neither digest nor replay moves.
    hinted = request(plain_tool.model_copy(update={"cache_control": {"type": "ephemeral"}}))
    assert sha256_json(hinted) == sha256_json(bare)
    assert canonical_request_sha256(hinted) == canonical_request_sha256(bare)


def test_messages_only_carriers_cache_control_and_inference_geo() -> None:
    """The top-level cache marker stays identity-inert; the region binds replay."""
    from exp.common.core.artifacts import sha256_json
    from exp.runtime.gateway.replay_identity import canonical_request_sha256

    messages = (GatewayMessage(role="user", content="hi"),)
    bare = GatewayRequest(surface=GatewayApiSurface.MESSAGES, messages=messages)
    cached = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=messages,
        provider_cache_control={"type": "ephemeral"},
    )
    assert cached.model_dump(mode="json") == bare.model_dump(mode="json")
    assert sha256_json(cached) == sha256_json(bare)
    assert canonical_request_sha256(cached) == canonical_request_sha256(bare)

    regional = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=messages,
        inference_geo="us",
    )
    assert sha256_json(regional) == sha256_json(bare)
    assert canonical_request_sha256(regional) != canonical_request_sha256(bare)

    with pytest.raises(ValidationError, match="provider_cache_control is valid only"):
        GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=messages,
            provider_cache_control={"type": "ephemeral"},
        )
    with pytest.raises(ValidationError, match="inference_geo is valid only"):
        GatewayRequest(
            surface=GatewayApiSurface.RESPONSES,
            messages=messages,
            inference_geo="us",
        )
    from exp.runtime.gateway.contracts import GatewayToolDefinition

    with pytest.raises(ValidationError, match="Anthropic tool carriers are valid only"):
        GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=messages,
            tools=(
                GatewayToolDefinition(
                    name="bash",
                    parameters={"type": "object"},
                    eager_input_streaming=True,
                ),
            ),
        )


def test_server_tool_carriers_are_scoped_verbatim_and_join_replay_identity() -> None:
    """Server tool entries and echoed blocks are Messages-only whole-message carriers."""
    from exp.runtime.gateway.replay_identity import canonical_request_sha256

    server_entry: JsonObject = {"type": "web_search_20250305", "name": "web_search", "max_uses": 8}
    bare = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="search"),),
    )
    carried = bare.model_copy(update={"provider_server_tools": (server_entry,)})
    # Excluded from serialization, distinct in replay identity.
    assert carried.model_dump() == bare.model_dump()
    assert canonical_request_sha256(carried) != canonical_request_sha256(bare)

    echoed = bare.model_copy(
        update={
            "messages": bare.messages
            + (
                GatewayMessage(
                    role="assistant",
                    provider_anthropic_block={
                        "type": "server_tool_use",
                        "id": "srvtoolu_1",
                        "name": "web_search",
                        "input": {},
                    },
                ),
            )
        }
    )
    assert canonical_request_sha256(echoed) != canonical_request_sha256(bare)

    with pytest.raises(ValidationError, match="valid only for Messages"):
        GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=(GatewayMessage(role="user", content="hi"),),
            provider_server_tools=(server_entry,),
        )
    with pytest.raises(ValidationError, match="carries the whole message"):
        GatewayMessage(
            role="assistant",
            content="also text",
            provider_anthropic_block={"type": "server_tool_use"},
        )


def test_tool_choice_may_name_a_server_tool() -> None:
    """Named and required selectors count verbatim server tools as tools."""
    server_entry: JsonObject = {"type": "web_search_20250305", "name": "web_search"}
    for tool_choice in (GatewayNamedToolChoice(name="web_search"), "required"):
        request = GatewayRequest(
            surface=GatewayApiSurface.MESSAGES,
            messages=(GatewayMessage(role="user", content="hi"),),
            provider_server_tools=(server_entry,),
            tool_choice=tool_choice,
        )
        assert request.provider_server_tools == (server_entry,)
    with pytest.raises(ValidationError, match="must name a request tool"):
        GatewayRequest(
            surface=GatewayApiSurface.MESSAGES,
            messages=(GatewayMessage(role="user", content="hi"),),
            provider_server_tools=(server_entry,),
            tool_choice=GatewayNamedToolChoice(name="absent"),
        )


def test_block_cache_markers_are_identity_inert_and_role_scoped() -> None:
    """Cache markers change cost, not semantics: no digest or replay effect."""
    from exp.common.core.artifacts import sha256_json
    from exp.runtime.gateway.replay_identity import canonical_request_sha256

    marked = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(
            GatewayMessage(
                role="system",
                content="a\n\nb",
                provider_text_blocks=(
                    {"type": "text", "text": "a"},
                    {"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}},
                ),
            ),
            GatewayMessage(role="user", content="hi"),
        ),
    )
    bare = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(
            GatewayMessage(role="system", content="a\n\nb"),
            GatewayMessage(role="user", content="hi"),
        ),
    )
    assert marked.model_dump(mode="json") == bare.model_dump(mode="json")
    assert sha256_json(marked) == sha256_json(bare)
    assert canonical_request_sha256(marked) == canonical_request_sha256(bare)

    marked_tool = GatewayMessage(
        role="tool",
        content="ok",
        tool_call_id="call-1",
        cache_control={"type": "ephemeral"},
    )
    bare_tool = GatewayMessage(role="tool", content="ok", tool_call_id="call-1")
    assert sha256_json(marked_tool) == sha256_json(bare_tool)

    with pytest.raises(ValidationError, match="valid only for tool messages"):
        GatewayMessage(role="user", content="hi", cache_control={"type": "ephemeral"})
    with pytest.raises(ValidationError, match="must flatten to the message content"):
        GatewayMessage(
            role="user",
            content="different",
            provider_text_blocks=({"type": "text", "text": "hi"},),
        )
    with pytest.raises(ValidationError, match="not valid for tool messages"):
        GatewayMessage(
            role="tool",
            content="ok",
            tool_call_id="call-1",
            provider_text_blocks=({"type": "text", "text": "ok"},),
        )


def test_native_tool_carriers_are_scoped_verbatim_and_join_replay_identity() -> None:
    """Non-function Responses tool declarations are Responses-only carriers."""
    from exp.runtime.gateway.replay_identity import canonical_request_sha256

    native_entry = GatewayProviderNativeTool(
        index=1,
        tool={"type": "custom", "name": "apply_patch", "format": {"type": "grammar"}},
    )
    bare = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="edit"),),
        tools=(GatewayToolDefinition(name="exec_command", parameters={"type": "object"}),),
    )
    carried = bare.model_copy(update={"provider_native_tools": (native_entry,)})
    # Excluded from serialization, distinct in replay identity.
    assert carried.model_dump() == bare.model_dump()
    assert canonical_request_sha256(carried) != canonical_request_sha256(bare)
    moved = bare.model_copy(
        update={"provider_native_tools": (native_entry.model_copy(update={"index": 0}),)}
    )
    assert canonical_request_sha256(moved) != canonical_request_sha256(carried)

    with pytest.raises(ValidationError, match="valid only for Responses"):
        GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=(GatewayMessage(role="user", content="hi"),),
            provider_native_tools=(native_entry,),
        )


def test_required_tool_choice_counts_native_tool_declarations() -> None:
    """A toolset made only of verbatim native declarations satisfies required."""
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="hi"),),
        provider_native_tools=(GatewayProviderNativeTool(index=0, tool={"type": "web_search"}),),
        tool_choice="required",
    )
    assert request.provider_native_tools[0].tool == {"type": "web_search"}


def test_native_tool_positions_must_tile_the_tools_array() -> None:
    """Duplicate or out-of-range positions are construction errors, keeping
    the native re-emission interleave total by construction."""
    entry = GatewayProviderNativeTool(index=0, tool={"type": "web_search"})
    with pytest.raises(ValidationError, match="distinct indexes"):
        GatewayRequest(
            surface=GatewayApiSurface.RESPONSES,
            messages=(GatewayMessage(role="user", content="hi"),),
            provider_native_tools=(entry, entry),
        )
    with pytest.raises(ValidationError, match="distinct indexes"):
        GatewayRequest(
            surface=GatewayApiSurface.RESPONSES,
            messages=(GatewayMessage(role="user", content="hi"),),
            provider_native_tools=(
                GatewayProviderNativeTool(index=2, tool={"type": "web_search"}),
            ),
        )


def test_videos_join_semantic_identity_while_cache_markers_stay_out() -> None:
    """Video parts change the canonical digest; cache-only markers never do.

    Two requests that differ only in the video they carry must digest apart,
    and a request carrying a provider cache marker on its text must digest
    identically to the same request without it.
    """
    from exp.common.core.artifacts import sha256_json
    from exp.common.models.content import TextContentPart, VideoContentPart

    def request(video_url: str, *, marked: bool) -> GatewayRequest:
        """Build one Chat request with a video, optionally cache-marked."""
        blocks: tuple[JsonObject, ...] = (
            ({"type": "text", "text": "what happens?", "cache_control": {"type": "ephemeral"}},)
            if marked
            else ()
        )
        return GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=(
                GatewayMessage(
                    role="user",
                    content="what happens?",
                    content_parts=(
                        VideoContentPart(url=video_url),
                        TextContentPart(text="what happens?"),
                    ),
                    provider_text_blocks=blocks,
                ),
            ),
        )

    first = request("https://example.com/a.mp4", marked=False)
    second = request("https://example.com/b.mp4", marked=False)
    marked = request("https://example.com/a.mp4", marked=True)
    dumped = first.model_dump(mode="json")
    parts = dumped["messages"][0]["content_parts"]
    assert parts[0]["kind"] == "video"
    assert parts[0]["url"] == "https://example.com/a.mp4"
    assert sha256_json(first) != sha256_json(second)
    assert sha256_json(first) == sha256_json(marked)
    assert first.videos == (first.messages[0].content_parts[0],)


def test_media_handles_join_semantic_identity_and_never_mix_providers() -> None:
    """A handle changes the canonical digest; two providers in one request refuse.

    The digest must separate two handles to different uploads, separate a
    handle from the same media inline, and ignore a cache marker placed on
    a handle-carrying image. No route can serve handles from two providers,
    so the request itself fails validation.
    """
    from exp.common.core.artifacts import sha256_json
    from exp.common.models.content import ImageContentPart, MediaHandle, TextContentPart

    def request(*parts: ImageContentPart) -> GatewayRequest:
        """Build one Chat request carrying the given image parts before a text run."""
        return GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=(
                GatewayMessage(
                    role="user",
                    content="what is this?",
                    content_parts=(*parts, TextContentPart(text="what is this?")),
                ),
            ),
        )

    openai_a = MediaHandle(provider="openai", reference="file-a")
    openai_b = MediaHandle(provider="openai", reference="file-b")
    first = request(ImageContentPart(handle=openai_a))
    second = request(ImageContentPart(handle=openai_b))
    marked = request(ImageContentPart(handle=openai_a, cache_control={"type": "ephemeral"}))
    owned = request(
        ImageContentPart(
            handle=MediaHandle(provider="bedrock", reference="s3://bkt/a.png"),
            media_type="image/png",
        )
    )
    unowned = request(
        ImageContentPart(
            handle=MediaHandle(
                provider="bedrock", reference="s3://bkt/a.png", bucket_owner="123456789012"
            ),
            media_type="image/png",
        )
    )
    dumped = first.model_dump(mode="json")
    part = dumped["messages"][0]["content_parts"][0]
    assert part["handle"] == {"provider": "openai", "reference": "file-a", "bucket_owner": None}
    assert "cache_control" not in part
    assert sha256_json(first) != sha256_json(second)
    assert sha256_json(first) == sha256_json(marked)
    assert sha256_json(owned) != sha256_json(unowned)
    assert first.media_handles == (openai_a,)
    assert request(
        ImageContentPart(handle=openai_a), ImageContentPart(handle=openai_b)
    ).media_handles == (
        openai_a,
        openai_b,
    )

    with pytest.raises(ValidationError, match="same provider"):
        request(
            ImageContentPart(handle=openai_a),
            ImageContentPart(handle=MediaHandle(provider="anthropic", reference="file_a")),
        )


def test_audio_joins_semantic_identity_while_cache_markers_stay_out() -> None:
    """Audio parts change the canonical digest; cache-only markers never do."""
    from exp.common.core.artifacts import sha256_json
    from exp.common.models.content import AudioContentPart, TextContentPart

    def request(data: str, *, marked: bool) -> GatewayRequest:
        """Build one Chat request with a clip between two texts, optionally cache-marked."""
        blocks: tuple[JsonObject, ...] = (
            (
                {"type": "text", "text": "listen: "},
                {"type": "text", "text": "what is said?", "cache_control": {"type": "ephemeral"}},
            )
            if marked
            else ()
        )
        return GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=(
                GatewayMessage(
                    role="user",
                    content="listen: what is said?",
                    content_parts=(
                        TextContentPart(text="listen: "),
                        AudioContentPart(media_type="audio/wav", data=data),
                        TextContentPart(text="what is said?"),
                    ),
                    provider_text_blocks=blocks,
                ),
            ),
        )

    first = request("UklGRgAAAABXQVZF", marked=False)
    second = request("UklGRgQAAABXQVZF", marked=False)
    marked = request("UklGRgAAAABXQVZF", marked=True)
    parts = first.model_dump(mode="json")["messages"][0]["content_parts"]
    assert [part["kind"] for part in parts] == ["text", "audio", "text"]
    assert parts[1] == {"kind": "audio", "media_type": "audio/wav", "data": "UklGRgAAAABXQVZF"}
    assert sha256_json(first) != sha256_json(second)
    assert sha256_json(first) == sha256_json(marked)
    assert first.audios == (first.messages[0].content_parts[1],)


def test_service_tier_is_serialization_inert_but_binds_replay_identity() -> None:
    """The same body at a different provider tier is a different operation."""
    from exp.common.core.artifacts import sha256_json
    from exp.runtime.gateway.replay_identity import canonical_request_sha256

    messages = (GatewayMessage(role="user", content="hi"),)
    bare = GatewayRequest(surface=GatewayApiSurface.CHAT_COMPLETIONS, messages=messages)
    flex = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=messages,
        service_tier="flex",
    )
    assert flex.model_dump(mode="json") == bare.model_dump(mode="json")
    assert sha256_json(flex) == sha256_json(bare)
    assert canonical_request_sha256(bare) == sha256_json(bare)
    assert canonical_request_sha256(flex) != canonical_request_sha256(bare)
    assert canonical_request_sha256(flex) != canonical_request_sha256(
        GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=messages,
            service_tier="priority",
        )
    )
    with pytest.raises(ValidationError, match="not valid for Messages"):
        GatewayRequest(
            surface=GatewayApiSurface.MESSAGES,
            messages=messages,
            service_tier="flex",
        )


def test_json_object_output_is_serialization_inert_but_binds_replay_identity() -> None:
    """Mode-free Chat digests are untouched; an enabled JSON mode is its own operation."""
    from exp.common.core.artifacts import sha256_json
    from exp.runtime.gateway.replay_identity import canonical_request_sha256

    messages = (GatewayMessage(role="user", content="hi"),)
    bare = GatewayRequest(surface=GatewayApiSurface.CHAT_COMPLETIONS, messages=messages)
    json_mode = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=messages,
        json_object_output=True,
    )
    assert "json_object_output" not in bare.model_dump(mode="json")
    assert json_mode.model_dump(mode="json") == bare.model_dump(mode="json")
    assert sha256_json(json_mode) == sha256_json(bare)
    assert canonical_request_sha256(bare) == sha256_json(bare)
    assert canonical_request_sha256(json_mode) != canonical_request_sha256(bare)
    with pytest.raises(ValidationError, match="mutually exclusive"):
        GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=messages,
            json_object_output=True,
            structured_text=StructuredTextFormat(name="out", json_schema={"type": "object"}),
        )
    with pytest.raises(ValidationError, match="valid only for Chat Completions"):
        GatewayRequest(
            surface=GatewayApiSurface.MESSAGES,
            messages=messages,
            json_object_output=True,
        )


def test_tool_messages_carry_text_and_image_parts_only() -> None:
    """A tool screenshot rides the tool message; other media kinds stay out."""
    from exp.common.models.content import (
        DocumentContentPart,
        ImageContentPart,
        TextContentPart,
    )

    message = GatewayMessage(
        role="tool",
        tool_call_id="call-1",
        content="tool said:",
        content_parts=(
            TextContentPart(text="tool said:"),
            ImageContentPart(media_type="image/png", data="aGk="),
        ),
    )
    assert [part.kind for part in message.content_parts] == ["text", "image"]
    assert message.images

    with pytest.raises(ValidationError, match="text and image parts"):
        GatewayMessage(
            role="tool",
            tool_call_id="call-1",
            content="",
            content_parts=(DocumentContentPart(media_type="application/pdf", data="aGk="),),
        )
    with pytest.raises(ValidationError, match="valid only for user and tool"):
        GatewayMessage(
            role="assistant",
            content="hi",
            content_parts=(
                TextContentPart(text="hi"),
                ImageContentPart(media_type="image/png", data="aGk="),
            ),
        )


def test_replay_identity_binds_the_function_call_namespace() -> None:
    """A replayed item differing only by namespace is a different request.

    Namespace-free requests keep their exact pre-existing digest because the
    field joins the replay envelope only when present.
    """
    from exp.runtime.gateway.replay_identity import canonical_request_sha256

    def request(*, namespace: str | None) -> GatewayRequest:
        return GatewayRequest(
            surface=GatewayApiSurface.RESPONSES,
            messages=(
                GatewayMessage(
                    role="assistant",
                    tool_calls=(
                        ToolCall(
                            call_id="call-1",
                            name="spawn_agent",
                            provider_output_index=0,
                            provider_namespace=namespace,
                        ),
                    ),
                ),
            ),
        )

    namespaced = request(namespace="collaboration")
    assert canonical_request_sha256(namespaced) == canonical_request_sha256(
        request(namespace="collaboration")
    )
    assert canonical_request_sha256(namespaced) != canonical_request_sha256(
        request(namespace="agents")
    )
    assert canonical_request_sha256(namespaced) != canonical_request_sha256(request(namespace=None))


def test_replay_identity_binds_tool_output_attribution() -> None:
    """A function_call_output name/namespace pair joins replay identity."""
    from exp.common.core.artifacts import sha256_json
    from exp.runtime.gateway.replay_identity import canonical_request_sha256

    def request(*, namespace: str | None) -> GatewayRequest:
        return GatewayRequest(
            surface=GatewayApiSurface.RESPONSES,
            messages=(
                GatewayMessage(
                    role="tool",
                    content="spawned",
                    tool_call_id="call-1",
                    provider_tool_name="spawn_agent" if namespace is not None else None,
                    provider_tool_namespace=namespace,
                ),
            ),
        )

    plain = request(namespace=None)
    assert canonical_request_sha256(plain) == sha256_json(plain)
    assert canonical_request_sha256(request(namespace="collaboration")) != canonical_request_sha256(
        plain
    )


def test_replay_identity_binds_the_function_call_caller() -> None:
    """A retained SDK 3.0 caller joins replay identity only when present."""
    from exp.runtime.gateway.replay_identity import canonical_request_sha256

    def request(*, caller: JsonObject | None) -> GatewayRequest:
        return GatewayRequest(
            surface=GatewayApiSurface.RESPONSES,
            messages=(
                GatewayMessage(
                    role="assistant",
                    tool_calls=(
                        ToolCall(
                            call_id="call-1",
                            name="lookup",
                            provider_output_index=0,
                            provider_caller=caller,
                        ),
                    ),
                ),
            ),
        )

    program: JsonObject = {"type": "program", "caller_id": "call_prog"}
    direct: JsonObject = {"type": "direct"}
    attributed = request(caller=program)
    assert canonical_request_sha256(attributed) == canonical_request_sha256(request(caller=program))
    assert canonical_request_sha256(attributed) != canonical_request_sha256(request(caller=direct))
    assert canonical_request_sha256(attributed) != canonical_request_sha256(request(caller=None))


def test_a_tool_result_caller_is_valid_only_on_tool_messages() -> None:
    """The output-side attribution carrier keeps the tool-role contract."""
    import pytest as _pytest

    message = GatewayMessage(
        role="tool",
        content="ok",
        tool_call_id="call-1",
        provider_tool_caller={"type": "direct"},
    )
    assert message.provider_tool_caller == {"type": "direct"}
    with _pytest.raises(ValueError, match="attribution"):
        GatewayMessage(
            role="assistant",
            content="ok",
            provider_tool_caller={"type": "direct"},
        )
