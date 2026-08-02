"""Tests for the offline HashingEmbedder and the get_embedder factory."""

from __future__ import annotations

import math

import pytest

from wmo.config import HarnessConfig
from wmo.providers.base import (
    EmbedderKind,
    EmbeddingResult,
    ProviderConfig,
    ProviderKind,
    TokenUsage,
)
from wmo.retrieval.embedders import HashingEmbedder, get_embedder


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def test_embedding_is_deterministic_and_normalized() -> None:
    emb = HashingEmbedder(dim=128)
    v1 = emb.embed(["get_user u_kath"])[0]
    v2 = emb.embed(["get_user u_kath"])[0]
    assert v1 == v2
    assert len(v1) == 128
    assert _cosine(v1, v1) == pytest.approx(1.0)


def test_similar_text_is_closer_than_dissimilar() -> None:
    emb = HashingEmbedder(dim=512)
    base = emb.embed(["get_reservation r_042"])[0]
    similar = emb.embed(["get_reservation r_043"])[0]
    different = emb.embed(["issue_refund o_999 amount=500"])[0]
    assert _cosine(base, similar) > _cosine(base, different)


def test_batch_matches_individual() -> None:
    emb = HashingEmbedder(dim=64)
    batch = emb.embed(["a", "b"])
    assert batch[0] == emb.embed(["a"])[0]
    assert batch[1] == emb.embed(["b"])[0]


def test_rejects_nonpositive_dim() -> None:
    with pytest.raises(ValueError, match="positive"):
        HashingEmbedder(dim=0)


# --- get_embedder factory ------------------------------------------------------------------------


def test_get_embedder_defaults_to_hashing_sized_to_embed_dim() -> None:
    embedder = get_embedder(HarnessConfig(embed_dim=77))  # default embed_provider is HASHING
    assert isinstance(embedder, HashingEmbedder)
    assert len(embedder.embed(["x"])[0]) == 77


def test_get_embedder_builds_provider_with_embed_dim_threaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, ProviderConfig] = {}

    def fake_get_provider(config: ProviderConfig) -> object:
        captured["config"] = config
        return object()  # the factory returns whatever the registry builds

    # The factory imports get_provider from wmo.providers lazily; patch it there.
    import wmo.providers as providers_pkg

    monkeypatch.setattr(providers_pkg, "get_provider", fake_get_provider)

    config = HarnessConfig(
        providers=[
            ProviderConfig(
                kind=ProviderKind.BEDROCK,
                model="us.anthropic.claude-opus-4-8",
                embed_model="amazon.titan-embed-text-v2:0",
            )
        ],
        embed_provider=EmbedderKind.BEDROCK,
        embed_dim=256,
    )
    get_embedder(config)

    built = captured["config"]
    assert built.kind is ProviderKind.BEDROCK
    assert built.embed_model == "amazon.titan-embed-text-v2:0"
    assert built.embed_dim == 256  # embed_dim stamped onto the provider config


def test_get_embedder_missing_provider_config_raises() -> None:
    # embed_provider points at OPENAI but no OpenAI ProviderConfig is registered.
    config = HarnessConfig(embed_provider=EmbedderKind.OPENAI)
    with pytest.raises(ValueError, match="no provider config for openai"):
        get_embedder(config)


def test_batched_embedder_chunks_requests() -> None:
    from wmo.retrieval.embedders import BatchedEmbedder

    class _Recorder:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls.append(len(texts))
            return [[float(len(t))] for t in texts]

    recorder = _Recorder()
    batched = BatchedEmbedder(recorder, batch=2)
    vectors = batched.embed(["a", "bb", "ccc", "dddd", "eeeee"])
    assert recorder.calls == [2, 2, 1]
    assert vectors == [[1.0], [2.0], [3.0], [4.0], [5.0]]


def test_batched_embedder_preserves_usage_across_chunks() -> None:
    from wmo.retrieval.embedders import BatchedEmbedder

    class _Metered:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return self.embed_with_usage(texts).vectors

        def embed_with_usage(self, texts: list[str]) -> EmbeddingResult:
            return EmbeddingResult(
                vectors=[[float(len(text))] for text in texts],
                usage=TokenUsage(input_tokens=10 * len(texts)),
                model="text-embedding-3-large",
            )

    result = BatchedEmbedder(_Metered(), batch=2).embed_with_usage(
        ["a", "bb", "ccc", "dddd", "eeeee"]
    )
    assert result.vectors == [[1.0], [2.0], [3.0], [4.0], [5.0]]
    assert result.usage.input_tokens == 50
    assert result.model == "text-embedding-3-large"
