"""Decode Chat/Responses structured-output formats into canonical structured text."""

from __future__ import annotations

from exp.runtime.gateway.contracts import StructuredTextFormat
from exp.runtime.openai_protocol.errors import invalid_field
from exp.runtime.openai_protocol.wire_models import _ChatResponseFormat, _ResponseText


def chat_json_object_output(value: _ChatResponseFormat | None) -> bool:
    """Return whether the Chat response format selects schema-free JSON mode."""
    return value is not None and value.type == "json_object"


def chat_structured_text(value: _ChatResponseFormat | None) -> StructuredTextFormat | None:
    """Convert the Chat response format to the internal structured-text shape.

    ``json_object`` is not a schema: it rides ``json_object_output`` on the
    canonical request (see :func:`chat_json_object_output`) and yields no
    structured text here.
    """
    if value is None or value.type in {"text", "json_object"}:
        return None
    schema = value.json_schema
    if schema is None:
        raise invalid_field("response_format.json_schema")
    return StructuredTextFormat(
        name=schema.name,
        description=schema.description,
        json_schema=schema.schema_,
        strict=schema.strict,
    )


def responses_structured_text(value: _ResponseText | None) -> StructuredTextFormat | None:
    """Convert the Responses JSON Schema text format when requested."""
    if value is None or value.format is None or value.format.type == "text":
        return None
    schema = value.format.schema_
    name = value.format.name
    if schema is None or name is None:
        raise invalid_field("text.format")
    return StructuredTextFormat(
        name=name,
        description=value.format.description,
        json_schema=schema,
        strict=value.format.strict,
    )
