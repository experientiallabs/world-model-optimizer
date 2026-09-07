"""Normalize native data-plane settlement payloads for durable accounting."""

from __future__ import annotations

from datetime import datetime

from exp.common.core.artifacts import JsonObject, stable_id
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import (
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayRefusalReason,
    GatewayUsage,
)
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.openai_protocol.errors import OpenAIProtocolError, public_failure_error

_TERMINAL_KINDS = {
    "completed": GatewayEventKind.COMPLETED,
    "incomplete": GatewayEventKind.INCOMPLETE,
    "failed": GatewayEventKind.FAILED,
}


def refusal_reason_from_payload(value: object) -> GatewayRefusalReason | None:
    """Parse one optional bounded refusal reason from a boundary payload.

    An unknown token fails closed to ``None`` rather than raising, so a future
    native reason a stale worker does not know never breaks settlement.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        return GatewayRefusalReason(value)
    except ValueError:
        return None


def ledger_failure(failure: GatewayFailure) -> GatewayFailure:
    """The failure as the ledger records it.

    A customer-owned provider failure (their BYOK credential or account) keeps
    its provider class for ladder decisions, but the durable row files it as
    the caller's invalid request: it is their configuration, never operator
    deadness that pages or opens a house circuit.
    """
    if failure.customer_owned and failure.failure_class in {
        GatewayFailureClass.PROVIDER_AUTHENTICATION,
        GatewayFailureClass.PROVIDER_QUOTA,
    }:
        return failure.model_copy(update={"failure_class": GatewayFailureClass.INVALID_REQUEST})
    return failure


def terminal_from_settlement(
    data: JsonObject,
) -> tuple[GatewayEvent, GatewayFailure | None]:
    """Build a durable terminal event from one native settlement payload.

    Args:
        data: Parsed outcome, usage, tool names, and optional failure.

    Returns:
        The normalized terminal event and optional failure.
    """
    raw_usage = data.get("usage")
    raw_tool_names = data.get("tool_names")
    usage = _usage_from_payload(
        raw_usage if isinstance(raw_usage, dict) else None,
        [str(name) for name in raw_tool_names] if isinstance(raw_tool_names, list) else [],
    )
    failure_payload = data.get("failure")
    failure = None
    if isinstance(failure_payload, dict):
        provider_detail = failure_payload.get("provider_detail")
        failure = GatewayFailure(
            failure_class=GatewayFailureClass(str(failure_payload["failure_class"])),
            safe_message=str(failure_payload["safe_message"]),
            provider_detail=(
                provider_detail if isinstance(provider_detail, str) and provider_detail else None
            ),
            customer_owned=failure_payload.get("customer_owned") is True,
            # The bounded refusal category rides the settlement argument so the
            # control plane counts refusals by reason without parsing detail.
            refusal_reason=refusal_reason_from_payload(failure_payload.get("refusal_reason")),
        )
        # A rejected credential or exhausted account on the customer's own
        # BYOK rung kept its ladder class in the data plane (so another
        # customer-managed rung could still serve), but the ledger files it
        # where it belongs: the caller's configuration, never operator
        # deadness that pages.
        failure = ledger_failure(failure)
    kind = _TERMINAL_KINDS[str(data["outcome"])]
    terminal = GatewayEvent(
        kind=kind,
        sequence_number=0,
        usage=usage,
        failure=failure if kind == GatewayEventKind.FAILED else None,
    )
    return terminal, failure


def first_token_at_from_settlement(data: JsonObject) -> datetime | None:
    """Return the winning attempt's first-token wall-clock time from a settlement payload.

    The native data plane includes ``first_token_at`` as an ISO-8601 timestamp only when it
    observed a first streamed token. A missing, non-string, or unparseable value yields
    ``None`` so accounting stays backward-compatible with engines that omit the field.

    Args:
        data: Parsed native settlement payload.

    Returns:
        The timezone-aware first-token time, or ``None`` when it is absent or malformed.
    """
    raw = data.get("first_token_at")
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _usage_from_payload(
    payload: JsonObject | None,
    tool_names: list[str],
) -> GatewayUsage | None:
    """Build normalized usage from settlement scalars and tool names."""
    names = tuple(str(name) for name in tool_names)
    if payload is None or payload.get("input_tokens") is None:
        return GatewayUsage(tool_names=names) if names else None
    return GatewayUsage(
        input_tokens=_optional_count(payload.get("input_tokens")),
        output_tokens=_optional_count(payload.get("output_tokens")),
        cached_input_tokens=_optional_count(payload.get("cached_input_tokens")),
        reasoning_tokens=_optional_count(payload.get("reasoning_tokens")),
        tool_names=names,
    )


def _optional_count(value: object) -> int | None:
    """Return one integer settlement token count or ``None``."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def deployment_operation_key(route: GatewayRoute, deployment: ExactModelDeployment) -> str:
    """Derive the stable per-deployment idempotency key used by dispatch.

    Mirrors the executor's provider-operation identity so retried physical
    dispatches of the same deployment reuse one caller operation while every
    later route position derives its own.

    Args:
        route: Resolved ordered route.
        deployment: The certified deployment being dispatched.

    Returns:
        Stable content-addressed operation identity.
    """
    authorization = route.snapshot.authorization
    return stable_id(
        "gateway-provider-operation",
        {
            "request_id": authorization.request_id,
            "catalog_sha256": authorization.catalog_sha256,
            "deployment_id": deployment.deployment_id,
            "connection_sha256": deployment.connection_sha256,
        },
    )


def optional_text(value: object) -> str | None:
    """Return one optional boundary string value or ``None``."""
    return value if isinstance(value, str) else None


def budget_quota_protocol_error() -> OpenAIProtocolError:
    """Return the public quota error for an exhausted monthly allocation."""
    failure = GatewayFailure(
        failure_class=GatewayFailureClass.QUOTA_EXCEEDED,
        safe_message="monthly gateway allocation is exhausted",
    )
    return public_failure_error(failure)
