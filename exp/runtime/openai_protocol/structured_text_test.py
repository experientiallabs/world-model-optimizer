"""Tests for Chat/Responses structured-output decoding into canonical structured text."""

from __future__ import annotations

from exp.runtime.openai_protocol.structured_text import (
    chat_json_object_output,
    chat_structured_text,
    responses_structured_text,
)
from exp.runtime.openai_protocol.wire_models import (
    _ChatResponseFormat,
    _ResponseFormat,
    _ResponseText,
    _StructuredSchema,
)


def test_chat_json_object_is_a_schema_free_mode_not_a_schema() -> None:
    """json_object selects the canonical JSON-object mode and yields no structured text."""
    fmt = _ChatResponseFormat(type="json_object")
    assert chat_structured_text(fmt) is None
    assert chat_json_object_output(fmt) is True


def test_chat_json_object_output_is_false_for_other_formats() -> None:
    """Only json_object selects the schema-free mode."""
    schema = _StructuredSchema(name="answer", schema={"type": "object"}, strict=True)
    assert chat_json_object_output(None) is False
    assert chat_json_object_output(_ChatResponseFormat(type="text")) is False
    assert (
        chat_json_object_output(_ChatResponseFormat(type="json_schema", json_schema=schema))
        is False
    )


def test_chat_json_schema_is_carried_verbatim() -> None:
    """A caller json_schema is preserved (name, schema, strict) unchanged."""
    schema = _StructuredSchema(name="answer", schema={"type": "object"}, strict=True)
    result = chat_structured_text(_ChatResponseFormat(type="json_schema", json_schema=schema))
    assert result is not None
    assert result.name == "answer"
    assert result.json_schema == {"type": "object"}
    assert result.strict is True


def test_chat_text_and_none_yield_no_structured_text() -> None:
    """`text` and an absent response_format carry no structured output."""
    assert chat_structured_text(None) is None
    assert chat_structured_text(_ChatResponseFormat(type="text")) is None


def test_responses_json_schema_is_carried_verbatim() -> None:
    """The Responses text.format json_schema decodes into structured text."""
    fmt = _ResponseFormat(type="json_schema", name="answer", schema={"type": "object"})
    result = responses_structured_text(_ResponseText(format=fmt))
    assert result is not None
    assert result.name == "answer"
    assert result.json_schema == {"type": "object"}


def test_responses_text_and_none_yield_no_structured_text() -> None:
    """`text` and an absent Responses format carry no structured output."""
    assert responses_structured_text(None) is None
    assert responses_structured_text(_ResponseText(format=_ResponseFormat(type="text"))) is None
