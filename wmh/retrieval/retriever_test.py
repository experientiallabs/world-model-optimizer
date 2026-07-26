"""Tests for the EmbeddingRetriever (DreamGym top-k replay buffer).

A deterministic fake Provider returns canned embeddings keyed off the encoded (state, action) text,
so we can assert exact top-k ordering by cosine similarity without any network.
"""

from __future__ import annotations

import pytest

from wmh.core.types import Action, ActionKind, EnvState, Observation, Step, Trace
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.retrieval.retriever import EmbeddingRetriever, Retriever


class FakeEmbedProvider:
    """Maps known encoded texts to fixed vectors; unknown texts get an orthogonal default."""

    def __init__(self, table: dict[str, list[float]], default: list[float]) -> None:
        self.config = ProviderConfig(kind=ProviderKind.ANTHROPIC, model="m")
        self._table = table
        self._default = default
        self.embed_calls = 0

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        return Completion(text="")

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        out: list[list[float]] = []
        for t in texts:
            vec = next((v for key, v in self._table.items() if key in t), self._default)
            out.append(list(vec))
        return out

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


def _step(tool: str, arg: int, obs: str) -> Step:
    return Step(
        action=Action(kind=ActionKind.TOOL_CALL, name=tool, arguments={"x": arg}),
        observation=Observation(content=obs),
        state_before=EnvState(structured={"loc": tool}),
    )


def test_embedding_retriever_satisfies_protocol() -> None:
    provider = FakeEmbedProvider({}, default=[1.0, 0.0])
    assert isinstance(EmbeddingRetriever(provider), Retriever)


def test_topk_orders_by_cosine_similarity() -> None:
    # Three steps with distinct embeddings; query aligns most with "alpha", then "beta".
    table = {
        "alpha": [1.0, 0.0, 0.0],
        "beta": [0.8, 0.6, 0.0],
        "gamma": [0.0, 0.0, 1.0],
    }
    provider = FakeEmbedProvider(table, default=[1.0, 0.0, 0.0])  # query encodes to "alpha" tool
    retriever = EmbeddingRetriever(provider)
    steps = [_step("gamma", 1, "g"), _step("alpha", 2, "a"), _step("beta", 3, "b")]
    retriever.index([Trace(trace_id="t", steps=steps)])

    top = retriever.topk(EnvState(structured={"loc": "alpha"}), steps[1].action, k=2)
    assert [s.action.name for s in top] == ["alpha", "beta"]


def test_topk_respects_k_and_returns_at_most_buffer_size() -> None:
    provider = FakeEmbedProvider({"alpha": [1.0, 0.0]}, default=[1.0, 0.0])
    retriever = EmbeddingRetriever(provider)
    retriever.index([Trace(trace_id="t", steps=[_step("alpha", 1, "a")])])

    assert retriever.topk(EnvState(), _step("alpha", 1, "a").action, k=5) != []
    assert len(retriever.topk(EnvState(), _step("alpha", 1, "a").action, k=5)) == 1
    assert retriever.topk(EnvState(), _step("alpha", 1, "a").action, k=0) == []


def test_topk_on_empty_buffer_returns_empty() -> None:
    provider = FakeEmbedProvider({}, default=[1.0, 0.0])
    retriever = EmbeddingRetriever(provider)
    retriever.index([])
    assert retriever.topk(EnvState(), _step("alpha", 1, "a").action, k=3) == []


def test_add_enriches_buffer_online() -> None:
    table = {"alpha": [1.0, 0.0], "beta": [0.0, 1.0]}
    provider = FakeEmbedProvider(table, default=[1.0, 0.0])
    retriever = EmbeddingRetriever(provider)
    retriever.index([Trace(trace_id="t", steps=[_step("beta", 1, "b")])])

    added = _step("alpha", 9, "freshly added")
    retriever.add(added)
    top = retriever.topk(EnvState(structured={"loc": "alpha"}), added.action, k=1)
    assert top[0].observation.content == "freshly added"


def test_add_into_empty_buffer_then_retrieves() -> None:
    provider = FakeEmbedProvider({"alpha": [1.0, 0.0]}, default=[1.0, 0.0])
    retriever = EmbeddingRetriever(provider)
    retriever.index([])  # buffer starts empty
    step = _step("alpha", 1, "first")
    retriever.add(step)
    assert retriever.topk(EnvState(), step.action, k=1)[0].observation.content == "first"


# phi text rendering (encode_state_action) is owned and tested by wmh/core/render_test.py.


def test_save_load_roundtrip_preserves_topk(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
    from wmh.retrieval.embedders import HashingEmbedder

    steps = [_step("alpha", 1, "a"), _step("beta", 2, "b"), _step("gamma", 3, "c")]
    src = EmbeddingRetriever(HashingEmbedder(dim=64))
    src.index([Trace(trace_id="t", steps=steps)])
    query_state = EnvState(structured={"loc": "beta"})
    before = [s.observation.content for s in src.topk(query_state, steps[1].action, k=2)]

    src.save(tmp_path / "index")
    # Fresh retriever reloads without re-embedding and returns identical ranking.
    dst = EmbeddingRetriever(HashingEmbedder(dim=64))
    dst.load(tmp_path / "index")
    after = [s.observation.content for s in dst.topk(query_state, steps[1].action, k=2)]
    assert before == after
    assert len(dst._steps) == 3


def test_save_load_survives_unicode_line_separators(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
    """Observations from scraped web/AX content can contain \u2028/\u2029/\x85.

    ``str.splitlines()`` splits on those, cutting a persisted JSON line mid-string
    (hit by the gui-tasks corpus — the first web-content corpus). The loader must
    split on newlines only.
    """
    from wmh.retrieval.embedders import HashingEmbedder

    weird = "before\u2028middle\x85after\u2029end"
    steps = [_step("alpha", 1, weird), _step("beta", 2, "plain")]
    src = EmbeddingRetriever(HashingEmbedder(dim=64))
    src.index([Trace(trace_id="t", steps=steps)])
    src.save(tmp_path / "index")

    dst = EmbeddingRetriever(HashingEmbedder(dim=64))
    dst.load(tmp_path / "index")
    assert len(dst._steps) == 2
    assert dst._steps[0].observation.content == weird


def test_save_load_empty_buffer(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
    from wmh.retrieval.embedders import HashingEmbedder

    src = EmbeddingRetriever(HashingEmbedder(dim=16))
    src.index([])
    src.save(tmp_path / "index")
    dst = EmbeddingRetriever(HashingEmbedder(dim=16))
    dst.load(tmp_path / "index")
    assert dst.topk(EnvState(), _step("alpha", 1, "a").action, k=3) == []


class _RecordingEmbedder:
    """Records the exact texts it was asked to embed; returns a fixed vector for each."""

    def __init__(self) -> None:
        self.config = ProviderConfig(kind=ProviderKind.ANTHROPIC, model="m")
        self.seen: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.seen.extend(texts)
        return [[1.0, 0.0] for _ in texts]


def test_key_mode_action_embeds_command_only_not_state_scaffolding() -> None:
    step = Step(
        action=Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": "ls -la"}),
        observation=Observation(content="out"),
    )
    rec = _RecordingEmbedder()
    EmbeddingRetriever(rec, key_mode="action").index([Trace(trace_id="t", steps=[step])])
    key = rec.seen[0]
    assert "ls -la" in key and "bash" in key
    assert "STATE:" not in key and "ACTION kind=" not in key  # no scaffolding

    rec2 = _RecordingEmbedder()
    EmbeddingRetriever(rec2, key_mode="state_action").index([Trace(trace_id="t", steps=[step])])
    assert "STATE:" in rec2.seen[0]  # default keeps the full summary


def test_key_mode_rejects_unknown() -> None:
    emb = _RecordingEmbedder()
    # The Literal type rules this out statically; the runtime guard still defends untyped callers
    # (e.g. a raw CLI string), which is what this asserts.
    with pytest.raises(ValueError, match="key_mode"):
        EmbeddingRetriever(emb, key_mode="bogus")  # ty: ignore[invalid-argument-type]
