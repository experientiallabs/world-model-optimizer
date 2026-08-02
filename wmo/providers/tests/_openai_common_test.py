"""Unit tests for the shared OpenAI-shaped request mapping."""

from __future__ import annotations

from typing import cast

import httpx
import pytest
from openai import BadRequestError

from wmo.providers import _openai_common
from wmo.providers.base import Message


def test_to_messages_prepends_system_when_present() -> None:
    out = _openai_common.to_messages(
        "sys", [Message(role="user", content="a"), Message(role="assistant", content="b")]
    )
    assert out == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]


def test_to_messages_omits_empty_system() -> None:
    out = _openai_common.to_messages("", [Message(role="user", content="a")])
    assert out == [{"role": "user", "content": "a"}]


def test_complete_handles_missing_usage() -> None:
    class _Choice:
        def __init__(self) -> None:
            self.message = type("M", (), {"content": "hi"})()

    class _Resp:
        choices = [_Choice()]
        usage = None

    class _Chat:
        def create(self, **kwargs: object) -> _Resp:
            return _Resp()

    chat = cast("_openai_common._ChatCompletions", _Chat())
    completion = _openai_common.complete(chat, "m", "", [Message(role="user", content="x")], 8)
    assert completion.text == "hi"
    assert completion.usage.input_tokens == 0
    assert completion.usage.output_tokens == 0


def test_complete_raises_clearly_on_empty_choices() -> None:
    class _Resp:
        choices: list[object] = []
        usage = None

    class _Chat:
        def create(self, **kwargs: object) -> _Resp:
            return _Resp()

    chat = cast("_openai_common._ChatCompletions", _Chat())
    # Content filtering can return zero choices; we want a clear ValueError, not IndexError.
    with pytest.raises(ValueError, match="no choices"):
        _openai_common.complete(chat, "m", "", [Message(role="user", content="x")], 8)


def _bad_request(message: str) -> BadRequestError:
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return BadRequestError(message, response=response, body={"error": {"message": message}})


def test_complete_retries_without_temperature_when_model_rejects_it() -> None:
    class _Choice:
        def __init__(self) -> None:
            self.message = type("M", (), {"content": "ok"})()

    class _Resp:
        choices = [_Choice()]
        usage = None

    class _Chat:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> _Resp:
            self.calls.append(dict(kwargs))
            if "temperature" in kwargs:
                raise _bad_request(
                    "Unsupported value: 'temperature' does not support 0.0 with this model."
                )
            return _Resp()

    fake = _Chat()
    chat = cast("_openai_common._ChatCompletions", fake)
    completion = _openai_common.complete(
        chat, "m", "", [Message(role="user", content="x")], 8, temperature=0.0
    )
    assert completion.text == "ok"
    assert len(fake.calls) == 2
    assert "temperature" not in fake.calls[1]


def test_complete_reraises_unrelated_bad_requests() -> None:
    class _Chat:
        def create(self, **kwargs: object) -> object:
            raise _bad_request("context length exceeded")

    chat = cast("_openai_common._ChatCompletions", _Chat())
    with pytest.raises(BadRequestError, match="context length"):
        _openai_common.complete(
            chat, "m", "", [Message(role="user", content="x")], 8, temperature=0.7
        )


def test_embed_with_usage_preserves_provider_prompt_tokens() -> None:
    class _Item:
        embedding = [0.1, 0.2]

    class _Usage:
        prompt_tokens = 37

    class _Response:
        data = [_Item()]
        usage = _Usage()

    class _Embeddings:
        def create(self, **kwargs: object) -> _Response:
            return _Response()

    result = _openai_common.embed_with_usage(
        cast("_openai_common._Embeddings", _Embeddings()),
        "text-embedding-3-large",
        ["route me"],
        2,
    )
    assert result.vectors == [[0.1, 0.2]]
    assert result.usage.input_tokens == 37
    assert result.model == "text-embedding-3-large"
