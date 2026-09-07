"""Public error translation helpers for the native gateway bridge."""

from __future__ import annotations

import json

from exp.runtime.gateway.contracts import GatewayApiSurface
from exp.runtime.models.providers.errors import ProviderCapabilityError
from exp.runtime.openai_protocol.errors import OpenAIProtocolError, unsupported_field

_PUBLIC_REQUEST_CAPABILITY_PARAMS = {
    GatewayApiSurface.CHAT_COMPLETIONS: {
        "developer_messages": "messages",
        "function_tools": "tools",
        "forced_tool_choice": "tool_choice",
        "image_input": "messages",
        "image_url_input": "messages",
        "video_input": "messages",
        "video_url_input": "messages",
        "audio_input": "messages",
        "pdf_input": "messages",
        "pdf_url_input": "messages",
        "media_handle_input": "messages",
        "media_handle_provider": "messages",
        "parallel_tool_calls": "parallel_tool_calls",
        "service_tier": "service_tier",
        "stop_sequences": "stop",
        "streaming": "stream",
        "streaming_tool_arguments": "stream",
        "strict_tools": "tools",
        "structured_output": "response_format",
        "structured_text": "response_format",
    },
    GatewayApiSurface.RESPONSES: {
        "developer_messages": "instructions",
        "function_tools": "tools",
        "forced_tool_choice": "tool_choice",
        "image_input": "input",
        "image_url_input": "input",
        "video_input": "input",
        "video_url_input": "input",
        "audio_input": "input",
        "pdf_input": "input",
        "pdf_url_input": "input",
        "media_handle_input": "input",
        "media_handle_provider": "input",
        "parallel_tool_calls": "parallel_tool_calls",
        "service_tier": "service_tier",
        "streaming": "stream",
        "streaming_tool_arguments": "stream",
        "strict_tools": "tools",
        "structured_output": "text.format",
        "structured_text": "text.format",
    },
    GatewayApiSurface.MESSAGES: {
        "developer_messages": "system",
        "function_tools": "tools",
        "forced_tool_choice": "tool_choice",
        "image_input": "messages",
        "image_url_input": "messages",
        "video_input": "messages",
        "video_url_input": "messages",
        "audio_input": "messages",
        "pdf_input": "messages",
        "pdf_url_input": "messages",
        "media_handle_input": "messages",
        "media_handle_provider": "messages",
        "parallel_tool_calls": "tool_choice.disable_parallel_tool_use",
        "stop_sequences": "stop_sequences",
        "streaming": "stream",
        "streaming_tool_arguments": "stream",
        "strict_tools": "tools",
    },
}


_ATTACHMENT_CAPABILITY_MESSAGES = {
    "service_tier": (
        "This model does not offer a flex or priority processing tier. "
        "Remove service_tier, or choose a model with tiered pricing enabled."
    ),
    "image_input": (
        "The selected model route cannot accept image input. "
        "Send text only or choose an image-capable model alias."
    ),
    "image_url_input": (
        "The selected model route accepts inline image data only. "
        "Send the image as a base64 data URL or choose a different model alias."
    ),
    "video_input": (
        "The selected model route cannot accept video input. "
        "Send text only or choose a video-capable model alias."
    ),
    "video_url_input": (
        "The selected model route accepts inline video data only. "
        "Send the video as a base64 data URL or choose a different model alias."
    ),
    "audio_input": (
        "The selected model route cannot accept audio input. "
        "Send text only or choose an audio-capable model alias."
    ),
    "pdf_input": (
        "The selected model route cannot accept PDF document input. "
        "Send text only or choose a document-capable model alias."
    ),
    "pdf_url_input": (
        "The selected model route accepts inline PDF data only. "
        "Send the document as base64 file data or choose a different model alias."
    ),
    "media_handle_input": (
        "The selected model route cannot reference media uploaded to a provider. "
        "Send the media inline or choose a model alias that accepts provider file handles."
    ),
    "media_handle_provider": (
        "The request references media uploaded to a different provider than the "
        "selected model route. Send the media inline or choose a model alias "
        "served by the provider that holds the upload."
    ),
    "forced_tool_choice": (
        "The selected model rejects a forced tool_choice ('required' or a named tool), "
        "and the gateway will not silently weaken it to 'auto'. Send tool_choice 'auto' "
        "(the model may still call the tool) or choose a different model alias."
    ),
}
"""Why an attachment was refused, since the field itself is the caller's message.

The shared unsupported-field wording asks the caller to remove the named
field, which no image or document request can do: the field is the
conversation. A capability error carrying its own caller-safe ``detail`` (a
media handle naming which provider holds the upload) wins over the generic
wording for its capability."""


def capability_param(
    capability: str,
    surface: GatewayApiSurface,
    *,
    public_stream: bool = True,
    public_tools: bool = False,
) -> str | None:
    """Translate an internal capability label to the caller's request field."""
    del public_tools
    if capability == "streaming_tool_arguments":
        return "tools"
    if capability == "streaming" and not public_stream:
        return None
    return _PUBLIC_REQUEST_CAPABILITY_PARAMS[surface].get(capability)


def public_capability_error(
    error: ProviderCapabilityError,
    surface: GatewayApiSurface,
    *,
    public_stream: bool,
    public_tools: bool,
    developer_messages_param: str | None = None,
) -> OpenAIProtocolError:
    """Translate one internal admission label into a stable public 400."""
    param = (
        developer_messages_param
        if error.capability == "developer_messages" and developer_messages_param is not None
        else capability_param(
            error.capability,
            surface,
            public_stream=public_stream,
            public_tools=public_tools,
        )
    )
    attachment_reason = _ATTACHMENT_CAPABILITY_MESSAGES.get(error.capability)
    if param is not None and attachment_reason is not None:
        return OpenAIProtocolError(
            status_code=400,
            code="unsupported_capability",
            message=error.detail or attachment_reason,
            param=param,
        )
    if param is not None:
        return unsupported_field(param, capability=True)
    return OpenAIProtocolError(
        status_code=400,
        code="unsupported_capability",
        message=(
            "The selected model route cannot serve this request. "
            "Choose a different model alias and resend the request."
        ),
        param="model",
    )


def escalation(reason: str) -> str:
    """Return a content-free native admission escalation disposition."""
    return json.dumps({"escalate": reason}, separators=(",", ":"))


def ledger_capability_message(safe_message: str, public_param: str | None) -> str:
    """Suffix the public request field onto the ledger's generic capability sentence.

    Args:
        safe_message: The provider-neutral capability rejection text.
        public_param: The caller-facing field the public 400 named, if any.

    Returns:
        The ledger message, with ``(field: <param>)`` appended when known.
    """
    if not public_param:
        return safe_message
    return f"{safe_message} (field: {public_param})"
