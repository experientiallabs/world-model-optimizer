"""Stable OpenAI-shaped public errors for the shared serving boundary."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ContractModel, JsonObject
from exp.runtime.gateway.contracts import (
    GatewayFailure,
    GatewayFailureClass,
    GatewayRefusalReason,
)

THROTTLED_RETRY_AFTER_SECONDS = 5
UNAVAILABLE_RETRY_AFTER_SECONDS = 2


class OpenAIErrorDetail(ContractModel):
    """One public OpenAI error without provider or credential details."""

    message: str = Field(min_length=1, max_length=2_048)
    type: Literal[
        "invalid_request_error",
        "authentication_error",
        "permission_error",
        "insufficient_quota",
        "api_error",
    ]
    param: str | None = Field(default=None, max_length=512)
    code: str = Field(min_length=1, max_length=128)
    refusal_reason: GatewayRefusalReason | None = None
    """The bounded refusal category, present only on a ``refusal`` error, so a
    caller reads which policy declined without any provider prose."""


class OpenAIErrorEnvelope(ContractModel):
    """Top-level error shape parsed by official OpenAI clients."""

    error: OpenAIErrorDetail


class OpenAIProtocolError(ValueError):
    """Field-specific public protocol error carrying its HTTP representation."""

    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        error_type: Literal[
            "invalid_request_error",
            "authentication_error",
            "permission_error",
            "insufficient_quota",
            "api_error",
        ] = "invalid_request_error",
        param: str | None = None,
        retry_after_seconds: int | None = None,
        refusal_reason: GatewayRefusalReason | None = None,
    ) -> None:
        """Create one sanitized public protocol failure.

        Args:
            status_code: HTTP status associated with the error.
            code: Stable machine-readable gateway code.
            message: Display-safe explanation and remediation.
            error_type: OpenAI error category.
            param: Exact public request field responsible for the error.
            retry_after_seconds: Optional positive wait advertised as ``Retry-After``.
            refusal_reason: Bounded refusal category, set only on a ``refusal``.
        """
        if retry_after_seconds is not None and retry_after_seconds <= 0:
            raise ValueError("retry_after_seconds must be positive")
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        self.detail = OpenAIErrorDetail(
            message=message,
            type=error_type,
            param=param,
            code=code,
            refusal_reason=refusal_reason,
        )

    def envelope(self) -> OpenAIErrorEnvelope:
        """Return the immutable OpenAI error envelope."""
        return OpenAIErrorEnvelope(error=self.detail)

    def json_body(self) -> JsonObject:
        """Return a JSON-compatible body suitable for an HTTP response.

        ``refusal_reason`` is additive: it appears only on a refusal, so every
        other error keeps its exact pre-existing envelope shape.
        """
        body = self.envelope().model_dump(mode="json")
        if self.detail.refusal_reason is None:
            error = body.get("error")
            if isinstance(error, dict):
                error.pop("refusal_reason", None)
        return body

    def headers(self) -> dict[str, str]:
        """Return transport headers implied by this error, such as ``Retry-After``."""
        if self.retry_after_seconds is None:
            return {}
        return {"Retry-After": str(self.retry_after_seconds)}


def invalid_field(param: str, message: str | None = None) -> OpenAIProtocolError:
    """Build one field-specific invalid-request error.

    Args:
        param: Public request field path.
        message: Optional safe explanation.

    Returns:
        Stable invalid-parameter error.
    """
    return OpenAIProtocolError(
        status_code=400,
        code="invalid_parameter",
        message=message or f"Invalid value for '{param}'.",
        param=param,
    )


def unsupported_field(
    param: str, *, capability: bool = False, message: str | None = None
) -> OpenAIProtocolError:
    """Build one explicit unsupported field or capability error.

    Args:
        param: Public request field path.
        capability: Whether the field is conditionally supported by deployments.
        message: Optional safe explanation replacing the generic one.

    Returns:
        Stable pre-dispatch rejection.
    """
    code = "unsupported_capability" if capability else "unsupported_parameter"
    noun = "capability" if capability else "parameter"
    return OpenAIProtocolError(
        status_code=400,
        code=code,
        message=message
        or (
            f"The {noun} '{param}' is not supported by this gateway profile. "
            "Remove the field and resend the request."
        ),
        param=param,
    )


def public_failure_error(
    failure: GatewayFailure,
    *,
    param: str | None = None,
    now: datetime | None = None,
) -> OpenAIProtocolError:
    """Map one sanitized gateway failure to a stable public error.

    Quota exhaustion and provider throttling advertise a ``Retry-After`` wait. Gateway
    budgets are hard UTC-calendar-month allocations, so an exhausted quota reports the
    exact next month boundary as its reset time.

    Args:
        failure: Provider-neutral failure already stripped of sensitive details.
        param: Optional request field responsible for the failure.
        now: Injectable current UTC time used only to compute the quota reset boundary.

    Returns:
        OpenAI-shaped protocol error with no raw provider data.
    """
    mappings: dict[
        GatewayFailureClass,
        tuple[
            int,
            str,
            Literal[
                "invalid_request_error",
                "authentication_error",
                "permission_error",
                "insufficient_quota",
                "api_error",
            ],
        ],
    ] = {
        GatewayFailureClass.INVALID_REQUEST: (400, "invalid_request", "invalid_request_error"),
        GatewayFailureClass.UNSUPPORTED_CAPABILITY: (
            400,
            "unsupported_capability",
            "invalid_request_error",
        ),
        GatewayFailureClass.AUTHENTICATION: (401, "invalid_key", "authentication_error"),
        GatewayFailureClass.AUTHORIZATION: (403, "model_not_granted", "permission_error"),
        GatewayFailureClass.QUOTA_EXCEEDED: (429, "insufficient_quota", "insufficient_quota"),
        GatewayFailureClass.THROTTLED: (429, "unavailable_route", "api_error"),
        GatewayFailureClass.TIMEOUT: (504, "deadline_exceeded", "api_error"),
        GatewayFailureClass.CANCELLED: (499, "request_cancelled", "api_error"),
        GatewayFailureClass.GUARDRAIL: (400, "content_filter", "invalid_request_error"),
        # A refusal with no visible refusal text is the model's answer to the
        # request content, not a routing failure. OpenAI rejects such prompts
        # as a 400 ``invalid_request_error`` ("rejected as a result of our
        # safety system"); the provider billed the processed input, so a 502
        # would misdescribe a charged call as an infrastructure fault.
        GatewayFailureClass.REFUSAL: (400, "refusal", "invalid_request_error"),
        GatewayFailureClass.UNAVAILABLE: (503, "gateway_unavailable", "api_error"),
    }
    status, code, error_type = mappings.get(
        failure.failure_class,
        (502, "all_routes_failed", "api_error"),
    )
    detail_code = failure.safe_details.get("code")
    if (
        failure.failure_class is GatewayFailureClass.INVALID_REQUEST
        and isinstance(detail_code, str)
        and detail_code in {"invalid_parameter", "unsupported_parameter"}
    ):
        code = detail_code
        detail_param = failure.safe_details.get("param")
        if param is None and isinstance(detail_param, str):
            param = detail_param
    message = failure.safe_message
    if (
        failure.failure_class is GatewayFailureClass.INVALID_REQUEST
        and failure.provider_detail is not None
    ):
        # The provider's own sentence names the field and the reason, so it
        # replaces the generic "verify the request fields" advice.
        message = f"{message.split(';')[0].strip()}: {failure.provider_detail}"
        if param is None and failure.rejected_parameter is not None:
            param = failure.rejected_parameter
    if failure.failure_class is GatewayFailureClass.UNSUPPORTED_CAPABILITY and isinstance(
        failure.safe_details.get("capability"), str
    ):
        if param is None:
            message = (
                "The requested capability is not supported by this model route. "
                "Remove the unsupported field or choose a different model."
            )
        else:
            message = (
                f"The capability '{param}' is not supported by this model route. "
                "Remove the field or choose a different model."
            )
    retry_after_seconds: int | None = None
    if failure.failure_class is GatewayFailureClass.THROTTLED:
        # A failure carrying its known throttle window advertises that wait
        # (floored at the default) so the header never contradicts the
        # message; without one the fixed default backoff applies.
        retry_after_seconds = max(
            THROTTLED_RETRY_AFTER_SECONDS,
            failure.retry_after_seconds or THROTTLED_RETRY_AFTER_SECONDS,
        )
    elif failure.failure_class is GatewayFailureClass.UNAVAILABLE:
        retry_after_seconds = UNAVAILABLE_RETRY_AFTER_SECONDS
    elif failure.failure_class is GatewayFailureClass.QUOTA_EXCEEDED:
        moment = now if now is not None else datetime.now(UTC)
        reset = _next_utc_month_start(moment)
        retry_after_seconds = max(1, math.ceil((reset - moment).total_seconds()))
        boundary = reset.isoformat().replace("+00:00", "Z")
        message = (
            f"{message}. The allocation resets at {boundary}; retry after that time "
            "or ask the gateway operator to raise the monthly budget."
        )
    # A refusal names its bounded category to the caller; an unnamed one is
    # explicitly ``unspecified`` so a client can always branch on the field.
    refusal_reason = (
        (failure.refusal_reason or GatewayRefusalReason.UNSPECIFIED)
        if failure.failure_class is GatewayFailureClass.REFUSAL
        else None
    )
    return OpenAIProtocolError(
        status_code=status,
        code=code,
        message=message,
        error_type=error_type,
        param=param,
        retry_after_seconds=retry_after_seconds,
        refusal_reason=refusal_reason,
    )


def _next_utc_month_start(moment: datetime) -> datetime:
    """Return the start of the UTC calendar month strictly after ``moment``.

    Args:
        moment: Timezone-aware current time.

    Returns:
        Midnight UTC on the first day of the following month.
    """
    anchored = moment.astimezone(UTC)
    if anchored.month == 12:
        return datetime(anchored.year + 1, 1, 1, tzinfo=UTC)
    return datetime(anchored.year, anchored.month + 1, 1, tzinfo=UTC)
