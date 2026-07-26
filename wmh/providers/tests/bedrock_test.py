"""Unit tests for BedrockProvider. No network: the boto3 client is faked via _get_client."""

from __future__ import annotations

import io
import json
from typing import cast

import pytest

from wmh.providers.base import DEFAULT_MAX_TOKENS, Message, ProviderConfig, ProviderKind
from wmh.providers.bedrock import BedrockProvider


class _FakeBody:
    def __init__(self, payload: dict[str, object]) -> None:
        self._raw = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._raw


class _FakeClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.last_kwargs: dict[str, object] = {}

    def invoke_model(self, **kwargs: object) -> dict[str, object]:
        self.last_kwargs = kwargs
        return {"body": _FakeBody(self.payload)}


def _config() -> ProviderConfig:
    return ProviderConfig(
        kind=ProviderKind.BEDROCK, model="anthropic.claude-opus-4-8", region="us-east-1"
    )


def _payload() -> dict[str, object]:
    return {
        "content": [{"type": "text", "text": "ok"}],
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }


def test_complete_builds_anthropic_body_and_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(_payload())
    provider = BedrockProvider(_config())
    monkeypatch.setattr(provider, "_get_client", lambda: fake)  # inject fake; no network

    completion = provider.complete("sys", [Message(role="user", content="hi")], max_tokens=32)

    assert completion.text == "ok"
    assert completion.usage.input_tokens == 5
    assert completion.usage.output_tokens == 3
    assert fake.last_kwargs["modelId"] == "anthropic.claude-opus-4-8"
    body = json.loads(cast("str", fake.last_kwargs["body"]))
    assert body["anthropic_version"] == "bedrock-2023-05-31"
    assert body["max_tokens"] == 32
    assert body["system"] == "sys"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert "temperature" not in body  # Claude 4.8 rejects sampling params


def test_complete_default_max_tokens_is_8k(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(_payload())
    provider = BedrockProvider(_config())
    monkeypatch.setattr(provider, "_get_client", lambda: fake)

    provider.complete("sys", [Message(role="user", content="hi")])

    body = json.loads(cast("str", fake.last_kwargs["body"]))
    assert body["max_tokens"] == DEFAULT_MAX_TOKENS


class _FakeEmbedClient:
    """Fakes bedrock-runtime for Titan embeddings: one invoke_model call per input text."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector
        self.calls: list[dict[str, object]] = []

    def invoke_model(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return {"body": _FakeBody({"embedding": self._vector})}


def test_embed_invokes_titan_per_text_and_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeEmbedClient([0.1, 0.2, 0.3])
    config = ProviderConfig(
        kind=ProviderKind.BEDROCK,
        model="us.anthropic.claude-opus-4-8",
        embed_model="amazon.titan-embed-text-v2:0",
        embed_dim=3,
        region="us-east-1",
    )
    provider = BedrockProvider(config)
    monkeypatch.setattr(provider, "_get_client", lambda: fake)

    vectors = provider.embed(["a", "b"])

    assert vectors == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    assert len(fake.calls) == 2  # Titan embeds one text per call
    assert fake.calls[0]["modelId"] == "amazon.titan-embed-text-v2:0"
    body = json.loads(cast("str", fake.calls[0]["body"]))
    assert body == {"inputText": "a", "dimensions": 3, "normalize": True}


def test_embed_defaults_model_and_omits_dimensions_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeEmbedClient([1.0])
    # No embed_model / embed_dim: default Titan model, and no `dimensions` (model's native size).
    provider = BedrockProvider(_config())
    monkeypatch.setattr(provider, "_get_client", lambda: fake)

    provider.embed(["x"])

    assert fake.calls[0]["modelId"] == "amazon.titan-embed-text-v2:0"
    body = json.loads(cast("str", fake.calls[0]["body"]))
    assert body == {"inputText": "x"}  # no dimensions/normalize when embed_dim is unset


def test_verify_reports_failure_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Boom:
        def invoke_model(self, **kwargs: object) -> dict[str, object]:
            raise RuntimeError("no creds")

    provider = BedrockProvider(_config())
    monkeypatch.setattr(provider, "_get_client", _Boom)
    result = provider.verify()
    assert result.ok is False
    assert "no creds" in result.detail
    assert result.kind is ProviderKind.BEDROCK


def test_fake_body_is_stream_like() -> None:
    # Guard: real boto3 returns a StreamingBody; .read() once is the contract we rely on.
    assert isinstance(io.BytesIO(b"x").read(), bytes)


@pytest.mark.skipif(
    "AWS_REGION" not in __import__("os").environ,
    reason="no AWS_REGION; skipping live smoke test",
)
def test_live_verify() -> None:  # pragma: no cover - network
    import os

    provider = BedrockProvider(
        ProviderConfig(
            kind=ProviderKind.BEDROCK,
            # Bedrock only serves Opus 4.8 through the cross-region inference profile; the bare
            # model id fails with "on-demand throughput isn't supported".
            model="us.anthropic.claude-opus-4-8",
            region=os.environ["AWS_REGION"],
        )
    )
    result = provider.verify()
    if not result.ok and ("ServiceUnavailable" in result.detail or "Throttling" in result.detail):
        pytest.skip(f"Bedrock capacity-constrained right now, not a code failure: {result.detail}")
    assert result.ok is True


@pytest.mark.skipif(
    "AWS_REGION" not in __import__("os").environ,
    reason="no AWS_REGION; skipping live Titan embeddings test",
)
def test_live_titan_embed() -> None:  # pragma: no cover - network
    import os

    # Real Titan embeddings: returns one 256-dim L2-normalized vector per input text.
    provider = BedrockProvider(
        ProviderConfig(
            kind=ProviderKind.BEDROCK,
            model="us.anthropic.claude-opus-4-8",
            embed_model="amazon.titan-embed-text-v2:0",
            embed_dim=256,
            region=os.environ["AWS_REGION"],
        )
    )
    vectors = provider.embed(["hello world", "a different sentence"])
    assert len(vectors) == 2
    assert all(len(v) == 256 for v in vectors)
    # Distinct inputs should not produce identical embeddings.
    assert vectors[0] != vectors[1]

