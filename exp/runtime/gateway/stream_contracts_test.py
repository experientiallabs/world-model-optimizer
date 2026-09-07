"""Tests for stream outcome contracts."""

import pytest

from exp.runtime.gateway.stream_contracts import (
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayRefusalReason,
)


def test_gateway_failure_carries_an_optional_bounded_refusal_reason() -> None:
    """The refusal reason is an optional typed field that round-trips through
    the contract's JSON serialization and defaults to absent."""
    bare = GatewayFailure(
        failure_class=GatewayFailureClass.PROVIDER_INTERNAL,
        safe_message="provider stream failed",
    )
    assert bare.refusal_reason is None

    refused = GatewayFailure(
        failure_class=GatewayFailureClass.REFUSAL,
        safe_message="provider refused the request: cybersecurity policy",
        refusal_reason=GatewayRefusalReason.CYBER_POLICY,
    )
    assert refused.refusal_reason is GatewayRefusalReason.CYBER_POLICY
    restored = GatewayFailure.model_validate(refused.model_dump(mode="json"))
    assert restored.refusal_reason is GatewayRefusalReason.CYBER_POLICY
    # The enum members are exactly the closed vocabulary shared with the engine.
    assert {reason.value for reason in GatewayRefusalReason} == {
        "cyber_policy",
        "cbrn",
        "content_policy",
        "recitation",
        "data_inspection",
        "unspecified",
    }


@pytest.mark.parametrize("length", [257, 65_536])
def test_stream_started_event_preserves_long_tool_id(length: int) -> None:
    """Provider tool IDs fit the same bound on output as on replay."""
    event = GatewayEvent(
        kind=GatewayEventKind.TOOL_CALL_STARTED,
        sequence_number=0,
        tool_call_index=0,
        tool_call_id="x" * length,
        tool_name="terminal",
    )
    assert event.tool_call_id == "x" * length
