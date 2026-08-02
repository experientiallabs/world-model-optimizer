"""Tests for the OpenAI-compatible chat endpoint (routing, streaming, affinity, request log)."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from llm_waterfall.types import (
    ChatChoice,
    ChatFunctionCall,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatToolCall,
    ChatUsage,
)

from wmo.core.types import JsonObject
from wmo.optimize.compression import (
    CompressionConfig,
    CompressionResult,
    Compressor,
    get_compressor,
    register_compressor,
)
from wmo.optimize.policy import (
    KNN_BANK_FILENAME,
    POLICY_FILENAME,
    ClusterRanking,
    EmbedderSpec,
    KnnBank,
    RoutingPolicy,
)
from wmo.providers.base import (
    Completion,
    EmbeddingResult,
    Message,
    ProviderKind,
    StreamChunk,
    TokenUsage,
    VerifyResult,
)
from wmo.providers.openrouter_pricing import CATALOG_PATH_ENV, PriceCatalog
from wmo.providers.pool import PoolEntry, load_pool
from wmo.retrieval.embedders import HashingEmbedder
from wmo.serving import chat as chat_module
from wmo.serving.chat import ChatMessage as EndpointMessage
from wmo.serving.chat import (
    EndpointRuntime,
    RequestLog,
    RequestLogRecord,
    create_chat_router,
    install_openai_error_shapes,
)
from wmo.serving.endpoint_config import ENDPOINT_CONFIG_FILENAME, EndpointConfig
from wmo.serving.query_embeddings import QUERY_EMBEDDING_FILENAME, QueryEmbeddingStore
from wmo.serving.savings import EndpointSavings, SavingsWindow
from wmo.tracking.pricing import ModelPrice

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

    from wmo.providers.base import ProviderConfig


class _EchoProvider:
    """Fake provider: replies with its own pool name so tests see who served."""

    def __init__(self, entry: PoolEntry) -> None:
        self.config: ProviderConfig = entry.provider_config()
        self.name = entry.name

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        return Completion(
            text=f"served by {self.name}",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )

    def stream(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Iterator[StreamChunk]:
        yield StreamChunk(delta="served by ")
        yield StreamChunk(delta=self.name)
        yield StreamChunk(done=True, usage=TokenUsage(input_tokens=10, output_tokens=5))

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def verify(self) -> VerifyResult:
        raise NotImplementedError


def _pool() -> list[PoolEntry]:
    return [
        PoolEntry(
            name="fable-5",
            kind=ProviderKind.ANTHROPIC,
            model="claude-fable-5",
        ),
        PoolEntry(
            name="haiku-4-5",
            kind=ProviderKind.ANTHROPIC,
            model="claude-haiku-4-5",
        ),
    ]


def _cluster_policy() -> RoutingPolicy:
    embedder = HashingEmbedder(dim=64)
    sql, prose = embedder.embed(["SELECT count(*) FROM superheroes", "write a friendly email"])
    return RoutingPolicy(
        kind="rank",
        default_model="haiku-4-5",
        pool=_pool(),
        embedder=EmbedderSpec(dim=64),
        top_k_clusters=1,
        clusters=[
            ClusterRanking(
                cluster_id=0, label="sql", centroid=sql, ranking=["fable-5", "haiku-4-5"]
            ),
            ClusterRanking(
                cluster_id=1, label="prose", centroid=prose, ranking=["haiku-4-5", "fable-5"]
            ),
        ],
    )


def _linear_policy() -> RoutingPolicy:
    spec = EmbedderSpec(dim=32)
    query = spec.build().embed(["route this coding task"])[0]
    return RoutingPolicy(
        kind="linear",
        default_model="fable-5",
        pool=_pool(),
        embedder=spec,
        linear_weak_model="haiku-4-5",
        linear_strong_model="fable-5",
        linear_weak_weights=[0.0] * spec.dim,
        linear_strong_weights=query,
        linear_threshold=0.5,
    )


def _client(tmp_path: Path, policy: RoutingPolicy | None = None) -> tuple[TestClient, Path]:
    log_path = tmp_path / "requests.jsonl"
    runtime = EndpointRuntime(
        name="tau-bench",
        policy=policy or RoutingPolicy(kind="static", default_model="haiku-4-5", pool=_pool()),
        provider_factory=_EchoProvider,
        log=RequestLog(log_path),
    )
    app = FastAPI()
    app.include_router(create_chat_router({"tau-bench": runtime}))
    return TestClient(app), log_path


def test_completion_matches_openai_shape(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "tau-bench"  # the endpoint, not the mechanism
    assert body["choices"][0]["message"]["content"] == "served by haiku-4-5"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    assert response.headers["x-wmo-routed-model"] == "haiku-4-5"


def test_linear_policy_routes_through_the_serving_endpoint(tmp_path: Path) -> None:
    client, log_path = _client(tmp_path, policy=_linear_policy())
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "route this coding task"}],
        },
    )
    assert response.status_code == 200
    assert response.headers["x-wmo-routed-model"] == "fable-5"
    assert response.json()["choices"][0]["message"]["content"] == "served by fable-5"
    row = _rows(log_path)[0]
    assert row["model"] == "fable-5"
    assert "linear router: predicted uplift" in str(row["routing_reason"])


def test_streaming_emits_openai_chunks(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "hi"}],
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        lines = [line for line in response.iter_lines() if line.startswith("data: ")]
    payloads = [line.removeprefix("data: ") for line in lines]
    assert payloads[-1] == "[DONE]"
    chunks = [json.loads(p) for p in payloads[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"])
    assert text == "served by haiku-4-5"
    # OpenAI include_usage framing: finish_reason chunk, THEN a choices-less usage chunk.
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["total_tokens"] == 15


def test_cluster_routing_and_affinity(tmp_path: Path) -> None:
    client, _ = _client(tmp_path, policy=_cluster_policy())
    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "SELECT count(*) FROM superheroes"}],
        },
    )
    assert first.headers["x-wmo-routed-model"] == "fable-5"
    reply = first.json()["choices"][0]["message"]["content"]
    # Turn 2 would route to the prose cluster if fresh, but the conversation prefix
    # (turn-1 user + assistant reply) pins the incumbent: affinity wins.
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [
                {"role": "user", "content": "SELECT count(*) FROM superheroes"},
                {"role": "assistant", "content": reply},
                {"role": "user", "content": "write a friendly email about it"},
            ],
        },
    )
    assert second.headers["x-wmo-routed-model"] == "fable-5"


def test_unknown_endpoint_404s_with_available(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404
    body = response.json()
    # OpenAI error shape: clients read body["error"]["message"], never FastAPI's "detail".
    assert body["error"]["code"] == "model_not_found"
    assert "tau-bench" in body["error"]["message"]


def test_models_endpoint_lists_endpoints(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    body = client.get("/v1/models").json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == ["tau-bench"]


def test_request_log_rows(tmp_path: Path) -> None:
    client, log_path = _client(tmp_path, policy=_cluster_policy())
    client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "SELECT 1"}],
        },
    )
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["endpoint"] == "tau-bench"
    assert row["model"] == "fable-5"
    assert row["cluster_id"] == 0
    assert row["cluster_label"] == "sql"
    assert row["routing_reason"]
    assert row["input_tokens"] == 10
    assert row["output_tokens"] == 5
    assert row["cost_usd"] == pytest.approx((10 * 10.0 + 5 * 50.0) / 1_000_000)
    assert row["latency_ms"] >= 0
    assert row["status"] == "ok"
    assert row["ts"]
    assert row["id"]
    assert row["leg"] == "serving"
    assert row["cached_tokens"] == 0  # carried for the metering contract, not yet captured
    assert row["router_cost_usd"] == 0.0  # hashing policy routes for free; passed through


def test_semantic_router_cost_uses_provider_reported_embedding_tokens(
    tmp_path: Path,
) -> None:
    class _MeteredEmbedder:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return self.embed_with_usage(texts).vectors

        def embed_with_usage(self, texts: list[str]) -> EmbeddingResult:
            return EmbeddingResult(
                vectors=HashingEmbedder(dim=64).embed(texts),
                usage=TokenUsage(input_tokens=1_000),
                model="text-embedding-3-large",
            )

    log_path = tmp_path / "requests.jsonl"
    runtime = EndpointRuntime(
        name="tau-bench",
        policy=_knn_policy(tmp_path),
        provider_factory=_EchoProvider,
        log=RequestLog(log_path),
    )
    runtime._policy_embedder = _MeteredEmbedder()
    app = FastAPI()
    app.include_router(create_chat_router({"tau-bench": runtime}))

    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "SELECT 1"}],
        },
    )

    assert response.status_code == 200
    assert _rows(log_path)[0]["router_cost_usd"] == pytest.approx(0.00013)


def test_create_app_mounts_endpoints_from_policies(tmp_path: Path) -> None:
    from wmo.serving.server import create_app

    app = create_app(
        artifact_dirs=(str(tmp_path),),
        world_models={},
        policies={
            "tau-bench": RoutingPolicy(kind="static", default_model="haiku-4-5", pool=_pool())
        },
    )
    client = TestClient(app)
    body = client.get("/v1/models").json()
    assert [m["id"] for m in body["data"]] == ["tau-bench"]


def test_create_app_without_policies_serves_empty_model_list(tmp_path: Path) -> None:
    # A client wired up before any policy is fitted gets an empty list and an OpenAI-shaped
    # "no endpoint" error, never a bare 404 on the whole /v1 surface.
    from wmo.serving.server import create_app

    app = create_app(artifact_dirs=(str(tmp_path),), world_models={})
    client = TestClient(app)
    models = client.get("/v1/models")
    assert models.status_code == 200
    assert models.json()["data"] == []
    chat = client.post(
        "/v1/chat/completions",
        json={"model": "anything", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert chat.status_code == 404
    assert chat.json()["error"]["code"] == "model_not_found"


def test_provider_failure_logs_error_and_502s(tmp_path: Path) -> None:
    class _BoomProvider(_EchoProvider):
        def complete(
            self,
            system: str,
            messages: list[Message],
            *,
            temperature: float = 0.7,
            max_tokens: int = 8192,
        ) -> Completion:
            raise RuntimeError("upstream on fire")

    log_path = tmp_path / "requests.jsonl"
    runtime = EndpointRuntime(
        name="tau-bench",
        policy=RoutingPolicy(kind="static", default_model="haiku-4-5", pool=_pool()),
        provider_factory=_BoomProvider,
        log=RequestLog(log_path),
    )
    app = FastAPI()
    app.include_router(create_chat_router({"tau-bench": runtime}))
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 502
    row = json.loads(log_path.read_text().splitlines()[0])
    assert row["status"] == "error"
    assert "upstream on fire" in row["error_message"]


def test_abandoned_stream_still_records_metering(tmp_path: Path) -> None:
    # A client that disconnects mid-stream closes the generator; the upstream call still
    # consumed tokens, so a request-log row must land anyway (D-METERING: no silent loss).
    class _EndlessProvider(_EchoProvider):
        def stream(
            self,
            system: str,
            messages: list[Message],
            *,
            temperature: float = 0.7,
            max_tokens: int = 8192,
        ) -> Iterator[StreamChunk]:
            for _ in range(1_000_000):  # far more than any client buffer; never finishes
                yield StreamChunk(delta="xxxxxxxx")

    log_path = tmp_path / "requests.jsonl"
    runtime = EndpointRuntime(
        name="tau-bench",
        policy=RoutingPolicy(kind="static", default_model="haiku-4-5", pool=_pool()),
        provider_factory=_EndlessProvider,
        log=RequestLog(log_path),
    )
    app = FastAPI()
    app.include_router(create_chat_router({"tau-bench": runtime}))

    # Drive the ASGI app directly: after the request body, `receive` reports
    # http.disconnect, which is what starlette listens for to cancel a StreamingResponse
    # and close its body iterator (GeneratorExit in the generator).
    body = json.dumps(
        {
            "model": "tau-bench",
            "stream": True,
            # Long enough that the disconnect path's chars/4 input estimate is nonzero.
            "messages": [{"role": "user", "content": "summarize this for me " * 8}],
        }
    ).encode()
    state = {"request_sent": False}

    async def receive() -> dict[str, object]:
        if not state["request_sent"]:
            state["request_sent"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
        return None

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("test", 0),
        "server": ("test", 80),
    }
    asyncio.run(app(cast("Any", scope), cast("Any", receive), cast("Any", send)))

    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert "disconnected" in rows[0]["error_message"]
    # The provider's exact usage rides only the terminal chunk the client never took, so
    # the row carries a chars/4 estimate of the partial generation and says so. Zero here
    # was the old behavior: the whole abandoned generation billed as free.
    assert rows[0]["input_tokens"] > 0
    assert rows[0]["output_tokens"] > 0
    assert "estimate" in rows[0]["error_message"]


def test_create_app_with_injected_policies_and_no_artifact_dirs(tmp_path: Path) -> None:
    # The injected-policies test pattern must not require an artifact root for the request log.
    from wmo.serving.server import create_app

    app = create_app(
        artifact_dirs=(),
        world_models={},
        policies={"ep": RoutingPolicy(kind="static", default_model="haiku-4-5", pool=_pool())},
    )
    client = TestClient(app)
    assert [m["id"] for m in client.get("/v1/models").json()["data"]] == ["ep"]


def _knn_policy(tmp_path: Path) -> RoutingPolicy:
    """A knn policy written to disk exactly as the fitter emits it: policy.json + npz sidecar.

    Six SQL scenarios where fable-5 wins outright and six prose ones where haiku-4-5 does, with
    haiku-4-5 as the pinned fallback. Loading it back proves the endpoint serves the artifact
    pair with no serving-side knowledge of the bank format.
    """
    sql = ["SELECT count(*) FROM superheroes", "SELECT name FROM users LIMIT 10"] * 3
    prose = ["write a friendly email to the team", "draft a thank-you note"] * 3
    rewards = [[1.0, 0.0]] * len(sql) + [[0.0, 1.0]] * len(prose)
    bank = KnnBank(
        embeddings=np.asarray(HashingEmbedder(dim=64).embed(sql + prose), dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        costs=np.asarray([[0.01, 0.001]] * len(rewards), dtype=np.float32),
        models=["fable-5", "haiku-4-5"],
        scenario_ids=[f"s{index}" for index in range(len(rewards))],
    )
    bank.save(tmp_path / KNN_BANK_FILENAME)
    RoutingPolicy(
        kind="knn",
        default_model="haiku-4-5",
        guard_model="haiku-4-5",
        pool=_pool(),
        embedder=EmbedderSpec(dim=64),
        rag_num=6,
        knn_min_pairs=4,
        # What the fitter persists for this bank: the mean of the per-model mean cell costs.
        cost_scale=0.0055,
    ).save(tmp_path / POLICY_FILENAME)
    return RoutingPolicy.load(tmp_path / POLICY_FILENAME)


def test_knn_policy_routes_a_request_end_to_end(tmp_path: Path) -> None:
    client, log_path = _client(tmp_path, policy=_knn_policy(tmp_path))
    routed = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "SELECT count(*) FROM superheroes"}],
        },
    )
    # The SQL neighborhood is unanimous, so the guard lets the request leave the fallback.
    assert routed.headers["x-wmo-routed-model"] == "fable-5"
    assert routed.json()["choices"][0]["message"]["content"] == "served by fable-5"

    fallback = client.post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": [{"role": "user", "content": "draft a thank-you"}]},
    )
    assert fallback.headers["x-wmo-routed-model"] == "haiku-4-5"

    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [row["model"] for row in rows] == ["fable-5", "haiku-4-5"]
    assert rows[0]["routing_reason"].startswith("knn: ")
    assert rows[0]["cluster_id"] is None  # a knn decision cites neighbors, not clusters


def test_cache_aware_knn_logs_the_incumbent_credit_end_to_end(
    tmp_path: Path,
) -> None:
    fitted = _knn_policy(tmp_path)
    priced_pool = [
        entry.model_copy(
            update={
                "input_per_mtok": 1.0,
                "cached_input_per_mtok": 0.1,
                "output_per_mtok": 1.0,
            }
        )
        for entry in fitted.pool
    ]
    policy = fitted.model_copy(
        update={
            "cache_aware": True,
            "pool": priced_pool,
        }
    )
    policy.attach_bank(fitted.knn_bank())
    client, log_path = _client(tmp_path, policy=policy)
    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "SELECT count(*) FROM superheroes"}],
        },
    )
    reply = first.json()["choices"][0]["message"]["content"]
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [
                {"role": "user", "content": "SELECT count(*) FROM superheroes"},
                {"role": "assistant", "content": reply},
                {"role": "user", "content": "write a friendly email"},
            ],
        },
    )

    assert first.status_code == second.status_code == 200
    rows = _rows(log_path)
    assert rows[0]["cache_credit_usd"] is None
    assert isinstance(rows[1]["cache_credit_usd"], float)
    assert rows[1]["cache_credit_usd"] > 0


def test_runtime_builds_the_policy_embedder_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An azure embedder spec would otherwise construct a fresh SDK client per request.
    from wmo.optimize.policy import EmbedderSpec

    builds = {"n": 0}
    original = EmbedderSpec.build

    def counting_build(self: EmbedderSpec) -> object:
        builds["n"] += 1
        return original(self)

    monkeypatch.setattr(EmbedderSpec, "build", counting_build)
    client, _ = _client(tmp_path, policy=_cluster_policy())
    for _ in range(3):
        client.post(
            "/v1/chat/completions",
            json={"model": "tau-bench", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert builds["n"] == 1


def test_static_policy_never_builds_its_embedder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A static route must keep serving even when the embedder spec cannot initialize here.
    from wmo.optimize.policy import EmbedderSpec

    def exploding_build(self: EmbedderSpec) -> object:
        raise RuntimeError("no credentials in this environment")

    monkeypatch.setattr(EmbedderSpec, "build", exploding_build)
    runtime = EndpointRuntime(
        name="tau-bench",
        policy=RoutingPolicy(
            kind="static",
            default_model="haiku-4-5",
            pool=_pool(),
            embedder=EmbedderSpec(dim=64),
        ),
        provider_factory=_EchoProvider,
        log=RequestLog(tmp_path / "requests.jsonl"),
    )
    app = FastAPI()
    app.include_router(create_chat_router({"tau-bench": runtime}))
    response = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200


# --- the cost/quality dial: GET/PUT /v1/endpoints/{name}/config -----------------------------


def _dial_client(
    tmp_path: Path, policy: RoutingPolicy, *, config_path: Path | None = None
) -> tuple[TestClient, EndpointRuntime]:
    """A client with OpenAI error shapes installed, so 400s look like the real server's."""
    runtime = EndpointRuntime(
        name="tau-bench",
        policy=policy,
        provider_factory=_EchoProvider,
        log=RequestLog(tmp_path / "requests.jsonl"),
        config_path=config_path,
    )
    app = FastAPI()
    app.include_router(create_chat_router({"tau-bench": runtime}))
    install_openai_error_shapes(app)
    return TestClient(app), runtime


def test_config_reports_an_as_fitted_endpoint_with_the_measured_anchors(tmp_path: Path) -> None:
    client, _ = _dial_client(tmp_path, _knn_policy(tmp_path))
    body = client.get("/v1/endpoints/tau-bench/config").json()
    assert body["endpoint"] == "tau-bench"
    assert body["dialable"] is True
    # Nobody has set the dial, and mounting did not set one for them.
    assert body["cost_quality"] is None
    assert body["named_point"] == "as-fitted"
    # The anchors are what a slider labels itself with: measured quality and cost per position,
    # sorted, and carrying nothing else. A client interpolates between them itself, so the
    # response must never hand it a delta for a position nobody measured.
    assert [anchor["s"] for anchor in body["anchors"]] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert [anchor["label"] for anchor in body["anchors"]] == [
        "Quality max",
        "Balanced (default)",
        "Cost saver",
        "Deep saver",
        "Max savings",
    ]
    balanced = next(anchor for anchor in body["anchors"] if anchor["s"] == 0.25)
    assert balanced == {
        "s": 0.25,
        "label": "Balanced (default)",
        "quality_delta_pt": 0.99,
        "cost_delta_pct": -24.7,
    }


def test_a_dial_between_anchors_is_labelled_custom(tmp_path: Path) -> None:
    # Never borrow the nearer anchor's name: its label sits next to its measured numbers.
    client, _ = _dial_client(tmp_path, _knn_policy(tmp_path))
    body = client.put("/v1/endpoints/tau-bench/config", json={"cost_quality": 0.26}).json()
    assert body["named_point"] == "Custom"
    assert body["cost_quality"] == 0.26


def test_put_moves_the_dial_on_the_live_endpoint(tmp_path: Path) -> None:
    client, runtime = _dial_client(tmp_path, _knn_policy(tmp_path))
    put = client.put("/v1/endpoints/tau-bench/config", json={"cost_quality": 1.0})
    assert put.status_code == 200
    body = put.json()
    assert body["cost_quality"] == 1.0
    assert body["named_point"] == "Max savings"
    assert body["knobs"] == {
        "knn_z": 0.5,
        "floor_q": 0.05,
        "pick_lam": 0.03,
        "guard_mode": "asymmetric",
    }
    # The runtime is serving the new policy, not just reporting it.
    assert runtime.policy.pick_lam == 0.03
    assert runtime.policy.guard_mode == "asymmetric"
    assert client.get("/v1/endpoints/tau-bench/config").json() == body
    # And the routed request says the knob was in play.
    routed = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "SELECT count(*) FROM superheroes"}],
        },
    )
    assert routed.status_code == 200
    rows = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert "cost knob lam=0.03" in rows[-1]["routing_reason"]


def test_put_persists_the_dial_next_to_the_policy(tmp_path: Path) -> None:
    config_path = tmp_path / ENDPOINT_CONFIG_FILENAME
    client, _ = _dial_client(tmp_path, _knn_policy(tmp_path), config_path=config_path)
    client.put("/v1/endpoints/tau-bench/config", json={"cost_quality": 0.6})
    # A restart has to come back on the same dial, so the file is the record.
    assert EndpointConfig.load(config_path).cost_quality == 0.6
    restarted = EndpointRuntime(
        name="tau-bench",
        policy=_knn_policy(tmp_path),
        log=RequestLog(tmp_path / "requests.jsonl"),
        cost_quality=EndpointConfig.load(config_path).cost_quality,
    )
    assert restarted.cost_quality == 0.6
    assert restarted.policy.guard_mode == "asymmetric"


def test_mounting_a_dial_setting_applies_it_before_the_first_request(tmp_path: Path) -> None:
    runtime = EndpointRuntime(
        name="tau-bench",
        policy=_knn_policy(tmp_path),
        provider_factory=_EchoProvider,
        log=RequestLog(tmp_path / "requests.jsonl"),
        cost_quality=0.75,
    )
    assert runtime.cost_quality == 0.75
    assert runtime.policy.pick_lam == pytest.approx(0.02)


def test_put_rejects_a_dial_outside_the_range(tmp_path: Path) -> None:
    client, runtime = _dial_client(tmp_path, _knn_policy(tmp_path))
    response = client.put("/v1/endpoints/tau-bench/config", json={"cost_quality": 1.5})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
    assert "cost_quality" in response.json()["error"]["message"]
    assert runtime.cost_quality is None  # a rejected change changes nothing


def test_config_on_a_policy_kind_without_a_dial(tmp_path: Path) -> None:
    static = RoutingPolicy(kind="static", default_model="haiku-4-5", pool=_pool())
    client, _ = _dial_client(tmp_path, static)
    body = client.get("/v1/endpoints/tau-bench/config").json()
    assert body["dialable"] is False
    assert body["cost_quality"] is None and body["knobs"] is None
    conflict = client.put("/v1/endpoints/tau-bench/config", json={"cost_quality": 0.5})
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "dial_unavailable"
    assert "kind='static'" in conflict.json()["error"]["message"]


def test_put_409s_when_the_policy_carries_no_cost_evidence(tmp_path: Path) -> None:
    # The coverage leg needs no prices; the price leg cannot be honored without them, and
    # saying so beats serving a dial position that silently does nothing.
    priceless = _knn_policy(tmp_path).model_copy(update={"cost_scale": 0.0})
    client, _ = _dial_client(tmp_path, priceless)
    assert (
        client.put("/v1/endpoints/tau-bench/config", json={"cost_quality": 0.2}).status_code == 200
    )
    conflict = client.put("/v1/endpoints/tau-bench/config", json={"cost_quality": 0.8})
    assert conflict.status_code == 409
    assert "cost_scale" in conflict.json()["error"]["message"]


def test_config_404s_for_an_unknown_endpoint(tmp_path: Path) -> None:
    client, _ = _dial_client(tmp_path, _knn_policy(tmp_path))
    missing = client.get("/v1/endpoints/nope/config")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "model_not_found"
    assert "tau-bench" in missing.json()["error"]["message"]
    assert client.put("/v1/endpoints/nope/config", json={"cost_quality": 0.5}).status_code == 404


def test_the_openai_surface_is_untouched_by_the_dial_routes(tmp_path: Path) -> None:
    # The customer-facing contract must not grow: /v1/models still lists endpoints only, and a
    # chat completion still names the endpoint.
    client, _ = _dial_client(tmp_path, _knn_policy(tmp_path))
    client.put("/v1/endpoints/tau-bench/config", json={"cost_quality": 0.9})
    assert [m["id"] for m in client.get("/v1/models").json()["data"]] == ["tau-bench"]
    completion = client.post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert completion.status_code == 200
    assert completion.json()["model"] == "tau-bench"
    assert "cost_quality" not in completion.text


def test_put_rejects_non_finite_dial_values(tmp_path: Path) -> None:
    # A slider bug that sends NaN must be a readable 400: NaN fails every comparison the guard
    # makes, so accepting it would quietly stop the endpoint from routing at all.
    client, runtime = _dial_client(tmp_path, _knn_policy(tmp_path))
    for payload in ('{"cost_quality": NaN}', '{"cost_quality": Infinity}'):
        response = client.put(
            "/v1/endpoints/tau-bench/config",
            content=payload,
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 400, payload
        assert response.json()["error"]["code"] == "invalid_request"
    assert runtime.cost_quality is None


# --- the savings card: GET /v1/endpoints/{name}/savings --------------------------------------


def _served_request(client: TestClient, content: str) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": [{"role": "user", "content": content}]},
    )
    assert response.status_code == 200


def test_savings_start_at_the_empty_state(tmp_path: Path) -> None:
    client, _ = _dial_client(tmp_path, _knn_policy(tmp_path))
    body = client.get("/v1/endpoints/tau-bench/savings").json()
    assert body["requests_served"] == 0
    assert body["cost_saved_usd"] == 0.0
    assert body["cost_saved_pct"] == 0.0
    assert body["time_saved_s_estimate"] == 0.0
    assert body["expected_quality_delta_pt"] == 0.0
    assert body["window"] == "all_time"
    assert body["estimate_basis"]  # never empty: the card explains the zero


def test_savings_accrue_as_the_endpoint_serves(tmp_path: Path) -> None:
    client, runtime = _dial_client(tmp_path, _knn_policy(tmp_path))
    _served_request(client, "SELECT count(*) FROM superheroes")  # routes to fable-5
    _served_request(client, "draft a thank-you")  # stays on the haiku-4-5 fallback
    body = client.get("/v1/endpoints/tau-bench/savings").json()
    assert body["requests_served"] == 2
    # fable-5 is the pricier model here, so routing away from the cheap fallback COSTS money and
    # the card says so with a negative saving rather than hiding it.
    assert body["actual_cost_usd"] > body["baseline_cost_estimate_usd"]
    assert body["cost_saved_usd"] < 0.0
    assert any("haiku-4-5" in basis for basis in body["estimate_basis"])


def test_savings_window_parameter_selects_the_period(tmp_path: Path) -> None:
    client, _ = _dial_client(tmp_path, _knn_policy(tmp_path))
    assert client.get("/v1/endpoints/tau-bench/savings?window=7d").json()["window"] == "7d"
    bad = client.get("/v1/endpoints/tau-bench/savings?window=forever")
    assert bad.status_code == 400
    assert bad.json()["error"]["code"] == "invalid_request"


def test_savings_survive_a_restart_by_reading_the_log(tmp_path: Path) -> None:
    # The persisted JSONL is the source, so a customer's savings are not reset by a deploy.
    client, _ = _dial_client(tmp_path, _knn_policy(tmp_path))
    _served_request(client, "SELECT count(*) FROM superheroes")
    restarted, _ = _dial_client(tmp_path, _knn_policy(tmp_path))
    body = restarted.get("/v1/endpoints/tau-bench/savings").json()
    assert body["requests_served"] == 1


def test_savings_ignore_other_endpoints_rows(tmp_path: Path) -> None:
    log = RequestLog(tmp_path / "requests.jsonl")
    log.append(
        RequestLogRecord(
            id="other",
            ts=datetime.now(UTC).isoformat(),
            endpoint="somebody-else",
            model="haiku-4-5",
            provider_model="claude-haiku-4-5",
            routing_reason="test",
            input_tokens=1000,
            output_tokens=1000,
            cost_usd=1.0,
        )
    )
    runtime = EndpointRuntime(
        name="tau-bench",
        policy=_knn_policy(tmp_path),
        provider_factory=_EchoProvider,
        log=log,
    )
    assert runtime.savings().requests_served == 0


def test_savings_skip_an_unreadable_log_row(tmp_path: Path) -> None:
    # A line truncated by a hard kill must not take the whole card down.
    client, _ = _dial_client(tmp_path, _knn_policy(tmp_path))
    _served_request(client, "SELECT count(*) FROM superheroes")
    with (tmp_path / "requests.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"id": "truncated"\n')
    body = client.get("/v1/endpoints/tau-bench/savings").json()
    assert body["requests_served"] == 1


def test_savings_are_available_for_a_static_endpoint(tmp_path: Path) -> None:
    static = RoutingPolicy(kind="static", default_model="haiku-4-5", pool=_pool())
    client, _ = _dial_client(tmp_path, static)
    _served_request(client, "hi")
    body = client.get("/v1/endpoints/tau-bench/savings").json()
    assert body["requests_served"] == 1
    assert body["cost_saved_usd"] == 0.0  # nothing to save against itself
    assert body["expected_quality_delta_pt"] == 0.0


def test_savings_404_for_an_unknown_endpoint(tmp_path: Path) -> None:
    client, _ = _dial_client(tmp_path, _knn_policy(tmp_path))
    missing = client.get("/v1/endpoints/nope/savings")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "model_not_found"


def test_moving_the_dial_refreshes_the_quality_expectation(tmp_path: Path) -> None:
    client, _ = _dial_client(tmp_path, _knn_policy(tmp_path))
    _served_request(client, "draft a thank-you")
    client.put("/v1/endpoints/tau-bench/config", json={"cost_quality": 1.0})
    body = client.get("/v1/endpoints/tau-bench/savings").json()
    assert body["expected_quality_delta_pt"] == -0.54  # the Max savings anchor, not a stale 0.0


def test_savings_are_recomputed_when_the_log_grows_not_on_a_timer(tmp_path: Path) -> None:
    # Cached between requests (a polling dashboard must not re-read the JSONL every paint), and
    # invalidated by the request that changes the total, so the card is never stale.
    client, runtime = _dial_client(tmp_path, _knn_policy(tmp_path))
    reads = {"n": 0}
    original = RequestLog.replay

    def counting_replay(self: RequestLog, endpoint: str) -> list[RequestLogRecord]:
        reads["n"] += 1
        return original(self, endpoint)

    RequestLog.replay = counting_replay
    try:
        client.get("/v1/endpoints/tau-bench/savings")
        client.get("/v1/endpoints/tau-bench/savings")
        assert reads["n"] == 1  # the second read came from the cache
        _served_request(client, "draft a thank-you")
        assert client.get("/v1/endpoints/tau-bench/savings").json()["requests_served"] == 1
        assert reads["n"] == 2
    finally:
        RequestLog.replay = original
    assert runtime.savings().requests_served == 1


def test_failed_dial_persist_leaves_the_live_endpoint_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Persist-then-install: a dial whose file write fails must not serve until restart un-sets
    # it, and must not move the live endpoint at all.
    from wmo.serving.endpoint_config import EndpointConfig

    _, runtime = _dial_client(
        tmp_path, _knn_policy(tmp_path), config_path=tmp_path / "endpoint.toml"
    )
    runtime.set_cost_quality(0.25)
    before = runtime.policy

    def exploding_save(self: EndpointConfig, path: Path) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(EndpointConfig, "save", exploding_save)
    with pytest.raises(OSError, match="disk full"):
        runtime.set_cost_quality(1.0)
    assert runtime.policy is before  # live dial unmoved
    assert runtime.cost_quality == 0.25


def test_slow_savings_computation_cannot_resurrect_the_old_dial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A savings computation that captures the old policy, races a dial move, and finishes late
    # must NOT store its stale result under a revision the new dial answers to.
    import wmo.serving.chat as chat_module

    client, runtime = _dial_client(tmp_path, _knn_policy(tmp_path))
    runtime.set_cost_quality(0.25)
    # The empty log zeroes every savings field; serve one request so the quality expectation
    # actually reflects the dial.
    client.post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": [{"role": "user", "content": "hi"}]},
    )
    original = chat_module.compute_savings

    moved = {"done": False}

    def racing_compute(
        rows: list[RequestLogRecord],
        policy: RoutingPolicy,
        *,
        window: SavingsWindow = "all_time",
    ) -> EndpointSavings:
        if not moved["done"]:
            moved["done"] = True
            runtime.set_cost_quality(1.0)  # the dial moves mid-computation
        return original(rows, policy, window=window)

    monkeypatch.setattr(chat_module, "compute_savings", racing_compute)
    stale = runtime.savings()  # computed against the 0.25 policy, returned to ITS caller
    fresh = runtime.savings()  # must recompute against the 1.0 policy, not read a stale cache
    assert fresh.expected_quality_delta_pt != stale.expected_quality_delta_pt
    assert fresh.expected_quality_delta_pt == pytest.approx(-0.54, abs=0.01)


def test_config_reports_the_coverage_setting_the_policy_was_fitted_with(tmp_path: Path) -> None:
    # An as-fitted endpoint's knobs must describe THAT fit, not the dial's default: a policy
    # fitted at the quality-max coverage setting reports 0.5, and one whose fit never recorded a
    # coverage setting reports null rather than a 0.0 that reads as "no floor".
    fitted_wide = _knn_policy(tmp_path).model_copy(update={"floor_q": 0.5})
    client, _ = _dial_client(tmp_path, fitted_wide)
    assert client.get("/v1/endpoints/tau-bench/config").json()["knobs"]["floor_q"] == 0.5

    unrecorded = _knn_policy(tmp_path).model_copy(update={"floor_q": None})
    older, _ = _dial_client(tmp_path, unrecorded)
    body = older.get("/v1/endpoints/tau-bench/config").json()
    assert body["knobs"]["floor_q"] is None
    assert body["knobs"]["knn_z"] == 0.5  # the rest of the knobs still report
    # Dialing it fixes the gap: the mapping records the quantile it applied.
    dialed = older.put("/v1/endpoints/tau-bench/config", json={"cost_quality": 0.0}).json()
    assert dialed["knobs"]["floor_q"] == 0.5


def test_the_seven_day_savings_window_is_never_served_from_cache(tmp_path: Path) -> None:
    # A bounded window ages with the clock, so an idle endpoint must not keep serving the answer
    # it computed an hour ago; the all-time card is still cached between requests.
    client, _ = _dial_client(tmp_path, _knn_policy(tmp_path))
    _served_request(client, "draft a thank-you")
    reads = {"n": 0}
    original = RequestLog.replay

    def counting_replay(self: RequestLog, endpoint: str) -> list[RequestLogRecord]:
        reads["n"] += 1
        return original(self, endpoint)

    RequestLog.replay = counting_replay
    try:
        client.get("/v1/endpoints/tau-bench/savings?window=7d")
        client.get("/v1/endpoints/tau-bench/savings?window=7d")
        assert reads["n"] == 2  # recomputed every read
        client.get("/v1/endpoints/tau-bench/savings")
        client.get("/v1/endpoints/tau-bench/savings")
        assert reads["n"] == 3  # all_time cached after the first
    finally:
        RequestLog.replay = original


# --- tool calling: tools/tool_choice in, tool_calls out, tool results replayed ----------------


_TOOL_ARGUMENTS = '{"table": "superheroes", "limit": 10}'

# The one tool name `_TOOLS` declares, named so a tool_choice can reference it without indexing
# into loosely typed wire JSON.
_TOOL_NAME = "lookup"

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": _TOOL_NAME,
            "description": "read rows from a table",
            "parameters": {
                "type": "object",
                "properties": {"table": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["table"],
            },
        },
    }
]


class _ToolProvider(_EchoProvider):
    """Fake structured provider: calls `lookup` once, then answers from the tool result.

    Every request it receives lands in `seen`, so a test can assert what actually reached the
    provider: the tool path exists precisely so nothing is flattened away on the way there.
    """

    def __init__(
        self,
        entry: PoolEntry,
        seen: list[ChatRequest],
        *,
        finish_reason: str | None = "tool_calls",
        n_tool_calls: int = 1,
    ) -> None:
        super().__init__(entry)
        self._seen = seen
        self._finish_reason = finish_reason
        self._n_tool_calls = n_tool_calls

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        self._seen.append(request)
        results = [m for m in request.messages if m.role == "tool"]
        if results:
            answer = f"{self.name} read {results[-1].content}"
            return ChatResponse(
                choices=[
                    ChatChoice(
                        message=ChatMessage(role="assistant", content=answer),
                        finish_reason="stop",
                    )
                ],
                usage=ChatUsage(prompt_tokens=13, completion_tokens=9),
            )
        return ChatResponse(
            choices=[
                ChatChoice(
                    message=ChatMessage(
                        role="assistant",
                        content=None,  # the shape a tool-only turn really has on the wire
                        tool_calls=[
                            ChatToolCall(
                                id=f"call_{index}",
                                function=ChatFunctionCall(name="lookup", arguments=_TOOL_ARGUMENTS),
                            )
                            for index in range(1, self._n_tool_calls + 1)
                        ],
                    ),
                    finish_reason=self._finish_reason,
                )
            ],
            usage=ChatUsage.model_validate(
                {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "prompt_tokens_details": {"cached_tokens": 4},
                }
            ),
        )


def _tool_client(
    tmp_path: Path,
    policy: RoutingPolicy | None = None,
    *,
    finish_reason: str | None = "tool_calls",
    n_tool_calls: int = 1,
) -> tuple[TestClient, Path, list[ChatRequest]]:
    """A client whose pool models all speak the structured tool-calling contract."""
    seen: list[ChatRequest] = []
    log_path = tmp_path / "requests.jsonl"
    runtime = EndpointRuntime(
        name="tau-bench",
        policy=policy or RoutingPolicy(kind="static", default_model="haiku-4-5", pool=_pool()),
        provider_factory=lambda entry: _ToolProvider(
            entry, seen, finish_reason=finish_reason, n_tool_calls=n_tool_calls
        ),
        log=RequestLog(log_path),
    )
    app = FastAPI()
    app.include_router(create_chat_router({"tau-bench": runtime}))
    install_openai_error_shapes(app)
    return TestClient(app), log_path, seen


def _rows(log_path: Path) -> list[JsonObject]:
    """The request log's rows as raw JSON, so a test reads the on-disk field names verbatim."""
    return [json.loads(line) for line in log_path.read_text().splitlines()]


def _error_message(row: JsonObject) -> str:
    """One row's `error_message`, narrowed: an error row always carries one, as a string."""
    message = row["error_message"]
    assert isinstance(message, str)
    return message


def test_a_tools_request_is_served_not_rejected(tmp_path: Path) -> None:
    # The whole point: a tool-calling agent could not use this endpoint at all before.
    client, log_path, seen = _tool_client(tmp_path)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "how many superheroes are there?"}],
            "tools": _TOOLS,
            "tool_choice": "auto",
        },
    )
    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["content"] is None  # tool-only turn: null, not an empty string
    assert message["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": _TOOL_ARGUMENTS},
        }
    ]
    assert response.json()["choices"][0]["finish_reason"] == "tool_calls"
    assert response.json()["model"] == "tau-bench"  # still the endpoint, never the routed model
    # The tools reached the provider as tools, and tool_choice rode along with them.
    assert seen[0].tools is not None
    assert [tool.function.name for tool in seen[0].tools] == ["lookup"]
    assert seen[0].tool_choice == "auto"
    assert seen[0].max_completion_tokens == 8192
    # One row per request, with the provider's real usage including its cached-prompt split.
    rows = _rows(log_path)
    assert len(rows) == 1
    assert (rows[0]["input_tokens"], rows[0]["output_tokens"]) == (11, 7)
    assert rows[0]["cached_tokens"] == 4
    assert rows[0]["status"] == "ok"


def test_a_tool_result_message_reaches_the_provider(tmp_path: Path) -> None:
    client, log_path, seen = _tool_client(tmp_path)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "how many superheroes are there?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": _TOOL_ARGUMENTS},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "42 rows"},
            ],
            "tools": _TOOLS,
        },
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "haiku-4-5 read 42 rows"
    assert response.json()["choices"][0]["finish_reason"] == "stop"
    # What the provider saw: the tool result kept its role and its tool_call_id, the assistant
    # turn kept its call, and the system turn stayed inline and in order.
    sent = seen[0].messages
    assert [m.role for m in sent] == ["system", "user", "assistant", "tool"]
    assert sent[2].content is None  # an empty content string is not sent in its place
    assert sent[2].tool_calls is not None
    assert sent[2].tool_calls[0].id == "call_1"
    assert (sent[3].tool_call_id, sent[3].content) == ("call_1", "42 rows")
    assert len(_rows(log_path)) == 1


def test_streaming_a_tool_call_emits_reassemblable_deltas(tmp_path: Path) -> None:
    client, log_path, _ = _tool_client(tmp_path)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "how many superheroes are there?"}],
            "tools": _TOOLS,
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        payloads = [
            line.removeprefix("data: ") for line in response.iter_lines() if line.startswith("data")
        ]
    assert payloads[-1] == "[DONE]"
    chunks = [json.loads(payload) for payload in payloads[:-1]]
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
    # Reassemble exactly as an OpenAI client does: concatenate function.arguments per index.
    arguments: dict[int, str] = {}
    names: dict[int, str] = {}
    ids: dict[int, str] = {}
    for chunk in chunks:
        for choice in chunk["choices"]:
            for call in choice["delta"].get("tool_calls") or []:
                index = call["index"]
                ids.setdefault(index, call["id"])
                names.setdefault(index, call["function"]["name"])
                arguments[index] = arguments.get(index, "") + call["function"]["arguments"]
    assert ids == {0: "call_1"} and names == {0: "lookup"}
    assert json.loads(arguments[0]) == {"table": "superheroes", "limit": 10}
    assert chunks[-2]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["total_tokens"] == 18
    rows = _rows(log_path)
    assert len(rows) == 1
    assert rows[0]["ttfb_ms"] is not None  # a re-emitted stream still reports when it could start
    assert rows[0]["output_tokens"] == 7


def test_affinity_survives_a_tool_round_trip(tmp_path: Path) -> None:
    # remember() has to store the assistant turn's tool_calls, not just its (empty) text, or the
    # next request's prefix fingerprints to a transcript that was never sent and the conversation
    # re-routes at exactly the point its prompt cache is warmest.
    client, log_path, _ = _tool_client(tmp_path, policy=_cluster_policy())
    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "SELECT count(*) FROM superheroes"}],
            "tools": _TOOLS,
        },
    )
    assert first.headers["x-wmo-routed-model"] == "fable-5"
    call = first.json()["choices"][0]["message"]["tool_calls"][0]
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [
                {"role": "user", "content": "SELECT count(*) FROM superheroes"},
                {"role": "assistant", "content": None, "tool_calls": [call]},
                {"role": "tool", "tool_call_id": call["id"], "content": "42 rows"},
            ],
            "tools": _TOOLS,
        },
    )
    assert second.headers["x-wmo-routed-model"] == "fable-5"
    rows = _rows(log_path)
    assert [row["routing_reason"] for row in rows] == [
        "rank router: nearest cluster 0 (sql)",
        "sticky: conversation affinity",
    ]


def test_routing_reads_the_user_turn_not_the_tool_result(tmp_path: Path) -> None:
    # A tool result is machine output the model asked for, not the customer's request: routing on
    # it would send one conversation to a different model on every turn. This transcript's user
    # turn is SQL (fable-5's cluster) while its tool result is prose (haiku-4-5's), and the
    # prefix was never remembered here, so only _routable_text decides.
    client, log_path, _ = _tool_client(tmp_path, policy=_cluster_policy())
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [
                {"role": "user", "content": "SELECT count(*) FROM superheroes"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": _TOOL_ARGUMENTS},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "write a friendly email about it",
                },
            ],
        },
    )
    assert response.status_code == 200
    assert response.headers["x-wmo-routed-model"] == "fable-5"
    assert _rows(log_path)[0]["cluster_label"] == "sql"  # not a sticky decision: a fresh one


def test_tools_on_a_pool_model_without_a_structured_backend(tmp_path: Path) -> None:
    # _EchoProvider has complete/stream but no complete_chat: the text-only provider
    # class this fallback path exists for.
    client, log_path = _client(tmp_path)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": _TOOLS,
        },
    )
    assert response.status_code == 501
    error = response.json()["error"]
    assert error["code"] == "tool_calling_unsupported"
    assert "haiku-4-5" in error["message"]  # names the pool entry an operator has to change
    assert "anthropic" in error["message"]
    rows = _rows(log_path)
    assert len(rows) == 1  # a capability gap is still one metered request
    assert rows[0]["status"] == "error"
    assert rows[0]["model"] == "haiku-4-5"
    assert "cannot serve tool calls" in _error_message(rows[0])


def test_a_tool_transcript_without_tools_still_takes_the_structured_path(tmp_path: Path) -> None:
    # Some agents stop re-sending `tools` once the loop is under way; the transcript alone must
    # be enough, because the text path cannot represent a tool result at all.
    client, _, seen = _tool_client(tmp_path)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [
                {"role": "user", "content": "how many superheroes are there?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": _TOOL_ARGUMENTS},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "42 rows"},
            ],
        },
    )
    assert response.status_code == 200
    assert seen[0].tools is None
    assert [m.role for m in seen[0].messages] == ["user", "assistant", "tool"]


def test_finish_reason_length_outranks_tool_calls(tmp_path: Path) -> None:
    # Arguments truncated at the output cap are invalid JSON; a client that reads "tool_calls"
    # would parse them as complete, so the truncation signal wins.
    client, _, _ = _tool_client(tmp_path, finish_reason="length")
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": _TOOLS,
        },
    )
    assert response.json()["choices"][0]["finish_reason"] == "length"
    assert response.json()["choices"][0]["message"]["tool_calls"]  # still reported


def test_finish_reason_is_derived_when_the_provider_omits_it(tmp_path: Path) -> None:
    client, _, _ = _tool_client(tmp_path, finish_reason=None)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": _TOOLS,
        },
    )
    assert response.json()["choices"][0]["finish_reason"] == "tool_calls"


def test_a_provider_returning_no_choices_is_a_502_with_a_log_row(tmp_path: Path) -> None:
    class _EmptyProvider(_ToolProvider):
        def complete_chat(self, request: ChatRequest) -> ChatResponse:
            return ChatResponse(choices=[])

    log_path = tmp_path / "requests.jsonl"
    runtime = EndpointRuntime(
        name="tau-bench",
        policy=RoutingPolicy(kind="static", default_model="haiku-4-5", pool=_pool()),
        provider_factory=lambda entry: _EmptyProvider(entry, []),
        log=RequestLog(log_path),
    )
    app = FastAPI()
    app.include_router(create_chat_router({"tau-bench": runtime}))
    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": _TOOLS,
        },
    )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_error"
    rows = _rows(log_path)
    assert len(rows) == 1
    assert "no choices" in _error_message(rows[0])


def test_a_tool_message_without_a_tool_call_id_is_a_400(tmp_path: Path) -> None:
    client, _, _ = _tool_client(tmp_path)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "tool", "content": "42 rows"},
            ],
        },
    )
    assert response.status_code == 400
    message = response.json()["error"]["message"]
    assert "tool_call_id" in message and "messages.1" in message


def test_a_tool_result_sent_as_an_assistant_turn_is_a_400(tmp_path: Path) -> None:
    client, _, _ = _tool_client(tmp_path)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "tool_call_id": "call_1", "content": "42 rows"},
            ],
        },
    )
    assert response.status_code == 400
    assert "role='tool'" in response.json()["error"]["message"]


def test_tool_results_alone_are_not_a_conversation(tmp_path: Path) -> None:
    client, _, _ = _tool_client(tmp_path)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "tool", "tool_call_id": "call_1", "content": "42 rows"}],
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_messages"


def test_the_remaining_unsupported_parameters_are_still_rejected(tmp_path: Path) -> None:
    client, _, _ = _tool_client(tmp_path)
    for payload, expected in (
        ({"n": 2}, "n != 1"),
        ({"logprobs": True}, "logprobs"),
        ({"response_format": {"type": "json_object"}}, "response_format"),
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "tau-bench",
                "messages": [{"role": "user", "content": "hi"}],
                **payload,
            },
        )
        assert response.status_code == 400, payload
        assert response.json()["error"]["code"] == "unsupported_parameter"
        assert expected in response.json()["error"]["message"]


def test_fingerprints_separate_tool_branches(tmp_path: Path) -> None:
    # Affinity is keyed on the transcript, so two different calls out of the same prefix (or two
    # results for different calls) must not share a key: they are different conversations.
    from wmo.serving.chat import _fingerprint

    def call(name: str) -> ChatToolCall:
        return ChatToolCall(id=f"call_{name}", function=ChatFunctionCall(name=name, arguments="{}"))

    prefix = [EndpointMessage(role="user", content="hi")]
    lookup = [*prefix, EndpointMessage(role="assistant", tool_calls=[call("lookup")])]
    write = [*prefix, EndpointMessage(role="assistant", tool_calls=[call("write")])]
    assert _fingerprint(lookup) != _fingerprint(write)
    assert _fingerprint(lookup) == _fingerprint(
        [*prefix, EndpointMessage(role="assistant", tool_calls=[call("lookup")])]
    )
    # Same result text, different call answered: still different conversations.
    first = [*lookup, EndpointMessage(role="tool", tool_call_id="call_lookup", content="42")]
    second = [*lookup, EndpointMessage(role="tool", tool_call_id="call_other", content="42")]
    assert _fingerprint(first) != _fingerprint(second)
    # And a text-only transcript keeps hashing to one stable key.
    text = [*prefix, EndpointMessage(role="assistant", content="hello")]
    assert _fingerprint(text) == _fingerprint(
        [*prefix, EndpointMessage(role="assistant", content="hello")]
    )


def test_an_empty_tool_list_is_served_as_plain_chat(tmp_path: Path) -> None:
    # A client whose tool registry is empty sends `tools: []`. That advertises nothing, so it must
    # not demand a tool-calling backend (this pool model has none) nor reach one as an empty array.
    client, log_path = _client(tmp_path)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [],
        },
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "served by haiku-4-5"
    assert _rows(log_path)[0]["status"] == "ok"


def test_tool_choice_without_tools_is_a_400(tmp_path: Path) -> None:
    client, _, _ = _tool_client(tmp_path)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": "required",
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_tool_choice"
    assert "declare the tool definitions" in response.json()["error"]["message"]


@pytest.mark.parametrize(
    "tool_choice",
    [
        {"type": "function", "function": {}},
        {"type": "function", "function": {"name": ""}},
        {"type": "function"},
        {"type": "function", "function": {"name": None}},
    ],
)
def test_a_function_tool_choice_with_no_usable_name_is_a_400(
    tmp_path: Path, tool_choice: dict[str, object]
) -> None:
    """A function-shaped choice that names nothing demands a call it cannot identify.

    Forwarded, it reaches an OpenAI-compatible backend as a malformed request that this endpoint
    reports as a 502 blaming the model, while Bedrock discards the requirement and can answer in
    prose to a client that requires a call. Refuse it at the door instead.
    """
    client, _, _ = _tool_client(tmp_path)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": _TOOLS,
            "tool_choice": tool_choice,
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_tool_choice"
    assert "no usable `function.name`" in response.json()["error"]["message"]


def test_a_function_tool_choice_naming_a_declared_tool_still_serves(tmp_path: Path) -> None:
    """The guard must not start refusing the well-formed named choice it sits next to."""
    client, _, provider = _tool_client(tmp_path)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": _TOOLS,
            "tool_choice": {
                "type": "function",
                "function": {"name": _TOOL_NAME},
            },
        },
    )
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("tool_choice", ["bogus", "", False, 42, 0, [], {"type": "retrieval"}])
def test_an_unroutable_tool_choice_is_a_400(tmp_path: Path, tool_choice: object) -> None:
    """REVERSED from an earlier revision of this branch, which forwarded unrecognized values.

    That was justified as forward-compatibility with OpenAI's growing vocabulary (`required`
    postdates `auto`), but no backend this endpoint routes to can honor a word it does not know:
    OpenAI-compatible servers reject it, which this endpoint reports as a 502 blaming the model for
    a client mistake, and Bedrock drops the field and can answer in prose to a client that required
    a call. Forwarding only converted a clear 400 into one of those two, so the accepted set is now
    closed and a new OpenAI word is a one-line addition to `_TOOL_CHOICE_WORDS`.

    `False`/`0` are in here on purpose: `True == 1` compares equal to neither string, so a
    membership-only test would let booleans and numbers through.
    """
    client, _, _ = _tool_client(tmp_path)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": _TOOLS,
            "tool_choice": tool_choice,
        },
    )
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "invalid_tool_choice"
    assert "this endpoint can route" in response.json()["error"]["message"]


@pytest.mark.parametrize("tool_choice", ["none", "auto", "required"])
def test_every_tool_choice_word_openai_defines_still_serves(
    tmp_path: Path, tool_choice: str
) -> None:
    """Closing the accepted set must not refuse any value OpenAI actually defines."""
    client, _, _ = _tool_client(tmp_path)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": _TOOLS,
            "tool_choice": tool_choice,
        },
    )
    assert response.status_code == 200, response.text


def test_omitting_tool_choice_entirely_still_serves(tmp_path: Path) -> None:
    """Absent is not unrecognized: the overwhelmingly common case must not start failing."""
    client, _, _ = _tool_client(tmp_path)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": _TOOLS,
        },
    )
    assert response.status_code == 200, response.text


def test_a_tool_call_streams_from_a_provider_with_no_streaming_backend(tmp_path: Path) -> None:
    # The re-emitted stream needs no native streaming surface at all (the module docstring's
    # tradeoff), so a structured-only provider can serve a streamed tool call; a streamed request
    # with no tools on that same provider still gets the honest 501.
    class _NoStreamProvider(_ToolProvider):
        stream = None  # type: ignore[assignment]  # not a StreamingProvider at all

    log_path = tmp_path / "requests.jsonl"
    runtime = EndpointRuntime(
        name="tau-bench",
        policy=RoutingPolicy(kind="static", default_model="haiku-4-5", pool=_pool()),
        provider_factory=lambda entry: _NoStreamProvider(entry, []),
        log=RequestLog(log_path),
    )
    app = FastAPI()
    app.include_router(create_chat_router({"tau-bench": runtime}))
    client = TestClient(app)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": _TOOLS,
        },
    ) as response:
        assert response.status_code == 200
        chunks = [
            json.loads(line.removeprefix("data: "))
            for line in response.iter_lines()
            if line.startswith("data: ") and not line.endswith("[DONE]")
        ]
    assert chunks[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "lookup"
    plain = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert plain.status_code == 501
    assert plain.json()["error"]["code"] == "streaming_unsupported"


def test_content_is_still_required_except_on_a_tool_call_turn(tmp_path: Path) -> None:
    # Only the assistant turn whose whole output is tool_calls may omit content; a user turn that
    # forgot it is the client bug the old required field caught, and must keep 400ing.
    client, _, _ = _tool_client(tmp_path)
    missing = client.post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": [{"role": "user"}]},
    )
    assert missing.status_code == 400
    assert "needs `content`" in missing.json()["error"]["message"]
    allowed = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [
                {"role": "user", "content": "how many superheroes are there?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": _TOOL_ARGUMENTS},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "42 rows"},
            ],
        },
    )
    assert allowed.status_code == 200


def test_parallel_tool_calls_reaches_the_provider(tmp_path: Path) -> None:
    # An agent whose executor answers one call per turn sends parallel_tool_calls=false. The
    # endpoint used to declare no such field, so pydantic dropped it and the client got a
    # multi-call turn back: a parameter the stack already carries, lost at the endpoint.
    client, _, seen = _tool_client(tmp_path)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "how many superheroes are there?"}],
            "tools": _TOOLS,
            "parallel_tool_calls": False,
        },
    )
    assert response.status_code == 200
    # On the wire, not just on the model: provider_payload is what a backend actually receives.
    assert (seen[0].model_extra or {}).get("parallel_tool_calls") is False
    assert seen[0].provider_payload("claude-haiku-4-5")["parallel_tool_calls"] is False


def test_parallel_tool_calls_is_only_sent_when_the_client_sent_it(tmp_path: Path) -> None:
    # A backend that rejects the field must never see it unasked, so an absent one stays absent.
    client, _, seen = _tool_client(tmp_path)
    client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": _TOOLS,
        },
    )
    assert seen[0].model_extra == {}
    assert "parallel_tool_calls" not in seen[0].provider_payload("claude-haiku-4-5")


def test_an_upstream_tool_calls_reason_with_no_calls_reports_stop(tmp_path: Path) -> None:
    # A self-hosted backend whose tool parser failed to extract the call reports finish_reason
    # tool_calls on a plain text turn. Passing that through emits a response no typed client can
    # handle: the agent loop idiom iterates message.tool_calls, which is null here.
    class _UnparsedProvider(_ToolProvider):
        def complete_chat(self, request: ChatRequest) -> ChatResponse:
            return ChatResponse(
                choices=[
                    ChatChoice(
                        message=ChatMessage(role="assistant", content="I will call a tool"),
                        finish_reason="tool_calls",
                    )
                ],
                usage=ChatUsage(prompt_tokens=11, completion_tokens=7),
            )

    log_path = tmp_path / "requests.jsonl"
    runtime = EndpointRuntime(
        name="tau-bench",
        policy=RoutingPolicy(kind="static", default_model="haiku-4-5", pool=_pool()),
        provider_factory=lambda entry: _UnparsedProvider(entry, []),
        log=RequestLog(log_path),
    )
    app = FastAPI()
    app.include_router(create_chat_router({"tau-bench": runtime}))
    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": _TOOLS,
        },
    )
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"  # never a reason the message cannot back up
    assert "tool_calls" not in choice["message"]


def test_affinity_survives_a_parallel_tool_round_trip(tmp_path: Path) -> None:
    # Parallel calls are OpenAI's default, and answering N of them appends N+1 messages (the
    # assistant turn plus one result each). A lookup that strips a fixed one message matches only
    # the single-call case, so a two-call turn re-routed mid-loop: the prompt cache is forfeited
    # exactly when the transcript is longest, and the new model is handed tool_call ids the old
    # one produced.
    client, log_path, _ = _tool_client(tmp_path, policy=_cluster_policy(), n_tool_calls=2)
    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "SELECT count(*) FROM superheroes"}],
            "tools": _TOOLS,
        },
    )
    calls = first.json()["choices"][0]["message"]["tool_calls"]
    assert [call["id"] for call in calls] == ["call_1", "call_2"]
    assert first.headers["x-wmo-routed-model"] == "fable-5"
    messages: list[JsonObject] = [
        {"role": "user", "content": "SELECT count(*) FROM superheroes"},
        {"role": "assistant", "content": None, "tool_calls": calls},
    ]
    messages += [
        {"role": "tool", "tool_call_id": call["id"], "content": "42 rows"} for call in calls
    ]
    second = client.post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": messages, "tools": _TOOLS},
    )
    assert second.headers["x-wmo-routed-model"] == "fable-5"
    assert [row["routing_reason"] for row in _rows(log_path)] == [
        "rank router: nearest cluster 0 (sql)",
        "sticky: conversation affinity",
    ]


def test_affinity_survives_a_parallel_round_trip_that_would_re_route(tmp_path: Path) -> None:
    # The sharp version: here the routable user turn is PROSE, so a missed incumbent does not
    # merely re-decide, it hands haiku-4-5 an assistant turn whose tool_call ids fable-5 made.
    client, log_path, _ = _tool_client(tmp_path, policy=_cluster_policy(), n_tool_calls=2)
    sql = {"role": "user", "content": "SELECT count(*) FROM superheroes"}
    first = client.post("/v1/chat/completions", json={"model": "tau-bench", "messages": [sql]})
    assert first.headers["x-wmo-routed-model"] == "fable-5"
    turn_two: list[JsonObject] = [
        sql,
        {"role": "assistant", "content": first.json()["choices"][0]["message"]["content"]},
        {"role": "user", "content": "write a friendly email about it"},
    ]
    second = client.post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": turn_two, "tools": _TOOLS},
    )
    assert second.headers["x-wmo-routed-model"] == "fable-5"  # affinity, not the prose cluster
    calls = second.json()["choices"][0]["message"]["tool_calls"]
    turn_three: list[JsonObject] = [
        *turn_two,
        {"role": "assistant", "content": None, "tool_calls": calls},
    ]
    turn_three += [
        {"role": "tool", "tool_call_id": call["id"], "content": "42 rows"} for call in calls
    ]
    third = client.post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": turn_three, "tools": _TOOLS},
    )
    assert third.headers["x-wmo-routed-model"] == "fable-5"
    assert [row["routing_reason"] for row in _rows(log_path)][1:] == [
        "sticky: conversation affinity",
        "sticky: conversation affinity",
    ]


def test_an_unusable_structured_response_still_meters_what_it_billed(tmp_path: Path) -> None:
    # Zero choices is the normal shape of an upstream content filter, and a filtered response
    # still bills the prompt it read. Recording TokenUsage() on that row reported real spend as
    # free, the same silent usage loss the abandoned-stream path exists to prevent.
    class _FilteredProvider(_ToolProvider):
        def complete_chat(self, request: ChatRequest) -> ChatResponse:
            return ChatResponse(choices=[], usage=ChatUsage(prompt_tokens=900, completion_tokens=0))

    log_path = tmp_path / "requests.jsonl"
    runtime = EndpointRuntime(
        name="tau-bench",
        policy=RoutingPolicy(kind="static", default_model="haiku-4-5", pool=_pool()),
        provider_factory=lambda entry: _FilteredProvider(entry, []),
        log=RequestLog(log_path),
    )
    app = FastAPI()
    app.include_router(create_chat_router({"tau-bench": runtime}))
    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": _TOOLS,
        },
    )
    assert response.status_code == 502
    row = _rows(log_path)[0]
    assert row["status"] == "error"
    assert (row["input_tokens"], row["output_tokens"]) == (900, 0)
    assert row["cost_usd"] == pytest.approx(900 * 1.0 / 1_000_000)


def test_a_failure_before_the_call_still_meters_zero(tmp_path: Path) -> None:
    # The other half of the split: nothing reached the provider, so there is nothing to bill.
    class _RefusingProvider(_ToolProvider):
        def complete_chat(self, request: ChatRequest) -> ChatResponse:
            raise RuntimeError("connection reset")

    log_path = tmp_path / "requests.jsonl"
    runtime = EndpointRuntime(
        name="tau-bench",
        policy=RoutingPolicy(kind="static", default_model="haiku-4-5", pool=_pool()),
        provider_factory=lambda entry: _RefusingProvider(entry, []),
        log=RequestLog(log_path),
    )
    app = FastAPI()
    app.include_router(create_chat_router({"tau-bench": runtime}))
    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": _TOOLS,
        },
    )
    assert response.status_code == 502
    row = _rows(log_path)[0]
    assert (row["input_tokens"], row["output_tokens"], row["cost_usd"]) == (0, 0, 0.0)
    assert "connection reset" in _error_message(row)


def test_an_empty_tool_calls_list_does_not_excuse_missing_content(tmp_path: Path) -> None:
    # `tool_calls: []` advertises no call, so the turn has neither text nor calls. Accepting it
    # sent an empty assistant turn down the text path, and most backends reject one: a client bug
    # would arrive as a 502 naming the pool model instead of a 400 naming the message.
    client, _, seen = _tool_client(tmp_path)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "tool_calls": []},
            ],
        },
    )
    assert response.status_code == 400
    assert "needs `content`" in response.json()["error"]["message"]
    assert seen == []


def test_an_explicit_null_content_is_rejected_like_an_omitted_one(tmp_path: Path) -> None:
    # `content: null` is what an SDK puts on the wire for an unset value, so it has to answer the
    # same way as an omitted key; normalizing null to "" first made it a silent empty billed turn.
    client, log_path, _ = _tool_client(tmp_path)
    for role in ("user", "system", "assistant"):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "tau-bench",
                "messages": [{"role": "user", "content": "hi"}, {"role": role, "content": None}],
            },
        )
        assert response.status_code == 400, role
        assert "needs `content`" in response.json()["error"]["message"]
    assert not log_path.exists()  # nothing was billed
    # The one turn that legitimately has none still gets through with content null.
    allowed = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [
                {"role": "user", "content": "how many superheroes are there?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": _TOOL_ARGUMENTS},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "42 rows"},
            ],
        },
    )
    assert allowed.status_code == 200


def test_a_provider_reply_with_neither_text_nor_calls_is_not_a_502(tmp_path: Path) -> None:
    # The mandatory-content rule catches a CLIENT sending an empty turn. An upstream reply with
    # no text is still renderable (it goes back out as content null), so refusing it here would
    # invent a 502 for a response the client can read.
    class _SilentProvider(_ToolProvider):
        def complete_chat(self, request: ChatRequest) -> ChatResponse:
            return ChatResponse(
                choices=[
                    ChatChoice(
                        message=ChatMessage(role="assistant", content=None), finish_reason="stop"
                    )
                ],
                usage=ChatUsage(prompt_tokens=11, completion_tokens=0),
            )

    log_path = tmp_path / "requests.jsonl"
    runtime = EndpointRuntime(
        name="tau-bench",
        policy=RoutingPolicy(kind="static", default_model="haiku-4-5", pool=_pool()),
        provider_factory=lambda entry: _SilentProvider(entry, []),
        log=RequestLog(log_path),
    )
    app = FastAPI()
    app.include_router(create_chat_router({"tau-bench": runtime}))
    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": _TOOLS,
        },
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] is None
    assert _rows(log_path)[0]["status"] == "ok"


def test_an_empty_tool_list_with_a_satisfiable_tool_choice_is_plain_chat(tmp_path: Path) -> None:
    # `tools: []` is normalized away FOR the client whose registry is empty, and that client
    # often sets tool_choice from config anyway. "auto" and "none" are both satisfied by not
    # calling a tool, so refusing them defeated the normalization; "none" literally means "do not
    # call tools", so refusing it for having none is backwards.
    client, log_path = _client(tmp_path)
    for tool_choice in ("auto", "none"):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "tau-bench",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [],
                "tool_choice": tool_choice,
            },
        )
        assert response.status_code == 200, tool_choice
        assert response.json()["choices"][0]["message"]["content"] == "served by haiku-4-5"
    assert [row["status"] for row in _rows(log_path)] == ["ok", "ok"]


def test_a_tool_choice_that_demands_a_call_without_tools_is_still_a_400(tmp_path: Path) -> None:
    # The half that cannot be honored: a client that MUST get a call back would otherwise read a
    # prose answer as the model's own choice.
    client, _, _ = _tool_client(tmp_path)
    for tool_choice in (
        "required",
        {"type": "function", "function": {"name": "lookup"}},
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "tau-bench",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [],
                "tool_choice": tool_choice,
            },
        )
        assert response.status_code == 400, tool_choice
        assert response.json()["error"]["code"] == "invalid_tool_choice"


# The transcript a mid-loop agent replays: the assistant turn it got back plus the result of the
# one call on it. It is what makes `needs_tool_calling()` true with no `tools` array in sight.
_REPLAYED_TOOL_TRANSCRIPT: list[JsonObject] = [
    {"role": "user", "content": "how many superheroes are there?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": _TOOL_ARGUMENTS},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "42 rows"},
]


def test_a_replayed_tool_transcript_does_not_bypass_tool_choice_validation(tmp_path: Path) -> None:
    # The turn that used to slip through: an agent that stops re-sending `tools` once the loop is
    # under way but keeps a demanding tool_choice from its config. The transcript makes this a
    # tool-calling request, yet there is still nothing for the choice to select, so forwarding it
    # would reach the provider as `tool_choice` with `tools: null`: an OpenAI-compatible backend
    # rejects that pair (a 502 here, blaming the model) and Bedrock drops the required choice and
    # answers in prose to a client that cannot use one.
    client, log_path, seen = _tool_client(tmp_path)
    for tool_choice in (
        "required",
        {"type": "function", "function": {"name": "lookup"}},
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "tau-bench",
                "messages": _REPLAYED_TOOL_TRANSCRIPT,
                "tool_choice": tool_choice,
            },
        )
        assert response.status_code == 400, tool_choice
        assert response.json()["error"]["code"] == "invalid_tool_choice"
        assert "declare the tool definitions" in response.json()["error"]["message"]
    assert seen == []  # refused before the upstream call, so nothing was billed for it
    assert not log_path.exists()


def test_a_replayed_transcript_that_re_sends_tools_serves_a_demanding_choice(
    tmp_path: Path,
) -> None:
    # The same mid-loop turn done right: `tools` re-sent, so "required" IS satisfiable and must
    # keep being served, with the choice riding through beside the tools it selects from.
    client, log_path, seen = _tool_client(tmp_path)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": _REPLAYED_TOOL_TRANSCRIPT,
            "tools": _TOOLS,
            "tool_choice": "required",
        },
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "haiku-4-5 read 42 rows"
    assert seen[0].tool_choice == "required"
    assert seen[0].tools is not None
    assert [tool.function.name for tool in seen[0].tools] == ["lookup"]
    assert _rows(log_path)[0]["status"] == "ok"


def test_a_named_tool_choice_must_name_a_declared_tool(tmp_path: Path) -> None:
    # No provider can call a function it was never given, so naming one that is absent from `tools`
    # is the same unhonorable request as naming one with no tools at all, one 400 earlier.
    client, log_path, seen = _tool_client(tmp_path)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "drop the superheroes table"}],
            "tools": _TOOLS,
            "tool_choice": {"type": "function", "function": {"name": "delete_rows"}},
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_tool_choice"
    message = response.json()["error"]["message"]
    assert "'delete_rows'" in message  # what went wrong
    assert "lookup" in message  # and what could be named instead
    assert seen == []
    assert not log_path.exists()


def test_a_non_demanding_tool_choice_is_not_forwarded_without_tools(tmp_path: Path) -> None:
    # "auto" and "none" are satisfied by not calling a tool, so a replayed transcript carrying one
    # must keep serving. It still must not reach the provider as `tool_choice` with `tools: null`,
    # the same malformed pair a demanding choice is refused for, so the choice is dropped instead:
    # with no tools advertised there is no call for it to permit or forbid.
    client, log_path, seen = _tool_client(tmp_path)
    for tool_choice in ("auto", "none"):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "tau-bench",
                "messages": _REPLAYED_TOOL_TRANSCRIPT,
                "tool_choice": tool_choice,
            },
        )
        assert response.status_code == 200, tool_choice
        assert response.json()["choices"][0]["message"]["content"] == "haiku-4-5 read 42 rows"
    assert [request.tool_choice for request in seen] == [None, None]
    assert [request.tools for request in seen] == [None, None]
    assert [row["status"] for row in _rows(log_path)] == ["ok", "ok"]


def test_an_openrouter_candidate_is_served_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `kind = "openrouter"` pool entry survives the whole serving path.

    Catalog-resolved price -> valid PoolEntry -> policy artifact on disk -> mounted endpoint ->
    an OpenAI-compatible completion attributed to that candidate. The provider factory is the
    usual echo fake (no network); `pool_test.py` covers the real `pool_provider` resolution.
    """
    catalog = tmp_path / "openrouter-prices.json"
    catalog.write_text(
        PriceCatalog(
            fetched_at=time.time(),
            source="test fixture",
            prices={"z-ai/glm-4.6": ModelPrice(input_per_mtok=0.4, output_per_mtok=1.75)},
        ).model_dump_json(),
        encoding="utf-8",
    )
    monkeypatch.setenv(CATALOG_PATH_ENV, str(catalog))
    pool_file = tmp_path / "pool.toml"
    pool_file.write_text(
        '[[model]]\nname = "or-glm"\nkind = "openrouter"\nmodel = "z-ai/glm-4.6"\ntier = "open"\n',
        encoding="utf-8",
    )
    policy_path = tmp_path / POLICY_FILENAME
    RoutingPolicy(kind="static", default_model="or-glm", pool=load_pool(pool_file).models).save(
        policy_path
    )

    runtime = EndpointRuntime(
        name="tau-bench",
        policy=RoutingPolicy.load(policy_path),
        provider_factory=_EchoProvider,
        log=RequestLog(tmp_path / "requests.jsonl"),
    )
    app = FastAPI()
    app.include_router(create_chat_router({"tau-bench": runtime}))
    response = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    assert response.headers["x-wmo-routed-model"] == "or-glm"
    assert response.json()["choices"][0]["message"]["content"] == "served by or-glm"


def _evidence_client(tmp_path: Path) -> tuple[TestClient, Path, QueryEmbeddingStore]:
    """A knn endpoint whose query vectors are recorded, as `create_app` wires them by default."""
    store = QueryEmbeddingStore(tmp_path / QUERY_EMBEDDING_FILENAME)
    runtime = EndpointRuntime(
        name="tau-bench",
        policy=_knn_policy(tmp_path),
        provider_factory=_EchoProvider,
        log=RequestLog(tmp_path / "requests.jsonl"),
        embeddings=store,
    )
    app = FastAPI()
    app.include_router(create_chat_router({"tau-bench": runtime}))
    return TestClient(app), tmp_path / "requests.jsonl", store


def test_a_served_request_logs_its_evidence_and_a_resolvable_embedding_ref(
    tmp_path: Path,
) -> None:
    """End to end: one real request, and the row it leaves carries the decision's numbers."""
    client, log_path, store = _evidence_client(tmp_path)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "SELECT count(*) FROM superheroes"}],
        },
    )
    assert response.status_code == 200
    row = _rows(log_path)[0]

    # The routed pick cleared the guard, so the row says so in fields, not only in prose.
    assert row["gate"] == "passed"
    assert row["propensity"] == "greedy"
    assert isinstance(row["n_pairs"], int) and row["n_pairs"] > 0
    assert isinstance(row["mean_diff"], float)
    assert isinstance(row["se"], float)

    # And the vector it was routed on resolves from the ref the same row carries.
    ref = row["query_embedding_ref"]
    assert isinstance(ref, str)
    assert ref.endswith(f"#{row['id']}")
    vector = store.get(ref)
    assert vector is not None
    assert vector.shape == (64,)
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-3)


def test_a_static_endpoint_leaves_the_evidence_columns_null(tmp_path: Path) -> None:
    # Nothing was embedded and no guard ran, so every added column is null rather than zero.
    client, log_path = _client(tmp_path)
    client.post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": [{"role": "user", "content": "hi"}]},
    )
    row = _rows(log_path)[0]
    for column in (
        "mean_diff",
        "se",
        "n_pairs",
        "gate",
        "propensity",
        "cache_credit_usd",
        "query_embedding_ref",
    ):
        assert row[column] is None, column


def test_query_embedding_logging_can_be_switched_off(tmp_path: Path) -> None:
    store = QueryEmbeddingStore(tmp_path / QUERY_EMBEDDING_FILENAME)
    runtime = EndpointRuntime(
        name="tau-bench",
        policy=_knn_policy(tmp_path),
        provider_factory=_EchoProvider,
        log=RequestLog(tmp_path / "requests.jsonl"),
        embeddings=store,
        log_query_embeddings=False,
    )
    app = FastAPI()
    app.include_router(create_chat_router({"tau-bench": runtime}))
    TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": [{"role": "user", "content": "hello"}]},
    )
    row = _rows(tmp_path / "requests.jsonl")[0]
    # The decision's evidence is still logged; only the vector is withheld.
    assert row["query_embedding_ref"] is None
    assert row["propensity"] is not None
    assert not (tmp_path / QUERY_EMBEDDING_FILENAME).exists()


# --- D-COMPRESS: the serving compression stage ---


class _CapturingProvider(_EchoProvider):
    """Echo provider that also records every (system, messages) it was asked to serve."""

    def __init__(self, entry: PoolEntry) -> None:
        super().__init__(entry)
        self.seen: list[tuple[str, list[Message]]] = []

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        self.seen.append((system, list(messages)))
        return super().complete(system, messages, temperature=temperature, max_tokens=max_tokens)

    def stream(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Iterator[StreamChunk]:
        self.seen.append((system, list(messages)))
        yield from super().stream(system, messages, temperature=temperature, max_tokens=max_tokens)


def _app_for(runtime: EndpointRuntime) -> FastAPI:
    app = FastAPI()
    app.include_router(create_chat_router({"tau-bench": runtime}))
    return app


def _compressed_runtime(
    tmp_path: Path, compression: CompressionConfig | None
) -> tuple[TestClient, Path, EndpointRuntime, dict[str, _CapturingProvider]]:
    providers: dict[str, _CapturingProvider] = {}

    def factory(entry: PoolEntry) -> _CapturingProvider:
        provider = _CapturingProvider(entry)
        providers[entry.name] = provider
        return provider

    log_path = tmp_path / "requests.jsonl"
    policy = RoutingPolicy(
        kind="static", default_model="haiku-4-5", pool=_pool(), compression=compression
    )
    runtime = EndpointRuntime(
        name="tau-bench", policy=policy, provider_factory=factory, log=RequestLog(log_path)
    )
    return TestClient(_app_for(runtime)), log_path, runtime, providers


def test_identity_compression_serves_bit_for_bit(tmp_path: Path) -> None:
    # The seam's do-no-harm proof: identity compression and no compression hand the provider
    # byte-identical (system, turns), and the log accounts raw == compressed.
    body = {
        "model": "tau-bench",
        "messages": [
            {"role": "system", "content": "be  terse\twith   spacing"},
            {"role": "user", "content": "  what is 2+2?  keep my  spacing "},
        ],
    }
    client_off, _, _, providers_off = _compressed_runtime(tmp_path / "off", None)
    client_off.post("/v1/chat/completions", json=body)
    identity = CompressionConfig(compressor_id="identity", aggressiveness=1.0)
    client_on, log_path, _, providers_on = _compressed_runtime(tmp_path / "on", identity)
    response = client_on.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert providers_on["haiku-4-5"].seen == providers_off["haiku-4-5"].seen
    row = _rows(log_path)[-1]
    assert row["compressor_id"] == "identity"
    assert row["tokens_in_raw"] == row["tokens_in_compressed"]
    assert cast("int", row["tokens_in_raw"]) > 0


def test_truncate_compresses_what_the_provider_sees(tmp_path: Path) -> None:
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    client, log_path, _, providers = _compressed_runtime(tmp_path, config)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [
                {"role": "system", "content": "system prompts are never compressed"},
                {"role": "user", "content": "one two three four five six seven eight"},
            ],
        },
    )

    assert response.status_code == 200
    system, turns = providers["haiku-4-5"].seen[0]
    assert system == "system prompts are never compressed"  # verbatim
    assert turns[0].content == "one two three four"  # trailing half dropped
    row = _rows(log_path)[-1]
    assert row["compressor_id"] == "truncate"
    assert row["compressor_version"] == "1"
    assert row["aggressiveness"] == 0.5
    assert cast("int", row["tokens_in_compressed"]) < cast("int", row["tokens_in_raw"])
    # OPAQUE: compression never surfaces in the response body or headers.
    assert "compress" not in response.text
    assert not any("compress" in key.lower() for key in response.headers)


class _SpyCompressor:
    """Delegates to the real compressor, recording which segments it was handed."""

    def __init__(self, inner: Compressor) -> None:
        self.inner = inner
        self.id = inner.id
        self.version = inner.version
        self.append_stable = inner.append_stable
        self.calls: list[list[str]] = []

    def compress(self, segments: list[str], config: CompressionConfig) -> CompressionResult:
        self.calls.append(list(segments))
        return self.inner.compress(segments, config)


def test_incumbent_prefix_is_reused_not_recompressed(tmp_path: Path) -> None:
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    client, _, runtime, providers = _compressed_runtime(tmp_path, config)
    spy = _SpyCompressor(cast("Compressor", runtime._compressor))
    runtime._compressor = spy

    first_user = "alpha beta gamma delta epsilon zeta"
    first = client.post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": [{"role": "user", "content": first_user}]},
    )
    reply = first.json()["choices"][0]["message"]["content"]
    turn_one = list(providers["haiku-4-5"].seen[0][1])

    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [
                {"role": "user", "content": first_user},
                {"role": "assistant", "content": reply},
                {"role": "user", "content": "eta theta iota kappa"},
            ],
        },
    )

    assert second.status_code == 200
    # The compressor only ever saw the turn-local segment on turn two: the cached prefix was
    # RETRIEVED from the affinity state, not recompressed.
    assert spy.calls == [[first_user], ["eta theta iota kappa"]]
    # And the provider-visible prefix is byte-identical across turns (prompt cache survives).
    second_turns = providers["haiku-4-5"].seen[1][1]
    assert [(m.role, m.content) for m in second_turns[: len(turn_one)]] == [
        (m.role, m.content) for m in turn_one
    ]
    assert second_turns[0].content == "alpha beta gamma"  # compressed once, reused verbatim
    assert second_turns[1].content == reply  # the model's own reply is never compressed
    assert second_turns[2].content == "eta theta"


def test_lost_affinity_recompression_is_byte_identical(tmp_path: Path) -> None:
    # Affinity eviction must not break the provider-visible prefix: per-segment determinism
    # reproduces the same bytes when the whole transcript is recompressed from scratch.
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    client, _, runtime, providers = _compressed_runtime(tmp_path, config)
    first_user = "alpha beta gamma delta epsilon zeta"
    first = client.post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": [{"role": "user", "content": first_user}]},
    )
    reply = first.json()["choices"][0]["message"]["content"]
    turn_one = providers["haiku-4-5"].seen[0][1]

    with runtime._lock:
        runtime._affinity.clear()
        runtime._clear_compressed()

    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [
                {"role": "user", "content": first_user},
                {"role": "assistant", "content": reply},
                {"role": "user", "content": "eta theta iota kappa"},
            ],
        },
    )
    assert second.status_code == 200
    second_turns = providers["haiku-4-5"].seen[1][1]
    assert [(m.role, m.content) for m in second_turns[: len(turn_one) + 1]] == [
        *[(m.role, m.content) for m in turn_one],
        ("assistant", reply),
    ]


def test_compression_fields_populate_on_stream_path(tmp_path: Path) -> None:
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    client, log_path, _, providers = _compressed_runtime(tmp_path, config)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "stream": True,
            "messages": [{"role": "user", "content": "one two three four five six"}],
        },
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
    assert "[DONE]" in body
    assert "compress" not in body  # opaque on the stream too
    _, turns = providers["haiku-4-5"].seen[0]
    assert turns[0].content == "one two three"
    rows = _rows(log_path)
    assert len(rows) == 1  # one record per request, stream included
    assert rows[0]["compressor_id"] == "truncate"
    assert cast("int", rows[0]["tokens_in_compressed"]) < cast("int", rows[0]["tokens_in_raw"])


def test_compression_fields_populate_on_error_path(tmp_path: Path) -> None:
    class _FailingProvider(_EchoProvider):
        def complete(
            self,
            system: str,
            messages: list[Message],
            *,
            temperature: float = 0.7,
            max_tokens: int = 8192,
        ) -> Completion:
            raise RuntimeError("upstream on fire")

    log_path = tmp_path / "requests.jsonl"
    policy = RoutingPolicy(
        kind="static",
        default_model="haiku-4-5",
        pool=_pool(),
        compression=CompressionConfig(compressor_id="truncate", aggressiveness=0.5),
    )
    runtime = EndpointRuntime(
        name="tau-bench", policy=policy, provider_factory=_FailingProvider, log=RequestLog(log_path)
    )
    response = TestClient(_app_for(runtime)).post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "one two three four five six"}],
        },
    )
    assert response.status_code == 502
    rows = _rows(log_path)
    assert len(rows) == 1  # one record per request, error included
    assert rows[0]["status"] == "error"
    assert rows[0]["compressor_id"] == "truncate"
    assert cast("int", rows[0]["tokens_in_compressed"]) < cast("int", rows[0]["tokens_in_raw"])


def test_uncompressed_rows_keep_default_compression_fields(tmp_path: Path) -> None:
    client, log_path = _client(tmp_path)
    client.post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": [{"role": "user", "content": "hi"}]},
    )
    row = _rows(log_path)[-1]
    assert row["compressor_id"] == ""
    assert row["compressor_version"] == ""
    assert row["tokens_in_raw"] == 0
    assert row["tokens_in_compressed"] == 0
    assert row["aggressiveness"] == 0.0


def test_dial_swap_keeps_the_compressor_matched_to_the_live_policy(tmp_path: Path) -> None:
    # #270's dial replaces the whole policy object at runtime (_install_policy); the resolved
    # compressor must follow it, not stay pinned to the mount-time object.
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    # Both stamps, as a real `route fit --compressor` writes them: a knn bank is only servable
    # under the representation it was fitted on, so a policy carrying one without the other
    # would not mount at all (see the requirement-A tests in policy_test.py).
    policy = _knn_policy(tmp_path).model_copy(
        update={"compression": config, "fit_compression": config}
    )
    client, runtime = _dial_client(tmp_path, policy)

    response = client.put("/v1/endpoints/tau-bench/config", json={"cost_quality": 1.0})
    assert response.status_code == 200

    served = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "one two three four five six"}],
        },
    )
    assert served.status_code == 200
    assert runtime.policy.compression is not None  # the dial carried the config through
    row = _rows(tmp_path / "requests.jsonl")[-1]
    assert row["compressor_id"] == "truncate"
    assert cast("int", row["tokens_in_compressed"]) < cast("int", row["tokens_in_raw"])


def test_compression_leaves_tool_calls_and_tool_results_verbatim(tmp_path: Path) -> None:
    # v1 scope (#278 x D-COMPRESS): the compressor shortens user prose only. A tool call's
    # arguments and a tool result are a structured contract the model reads back exactly, so
    # truncating them would change what the transcript MEANS, not just how long it is.
    policy = RoutingPolicy(
        kind="static",
        default_model="haiku-4-5",
        pool=_pool(),
        compression=CompressionConfig(compressor_id="truncate", aggressiveness=0.5),
    )
    client, log_path, seen = _tool_client(tmp_path, policy)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": _REPLAYED_TOOL_TRANSCRIPT},
    )

    assert response.status_code == 200
    served = seen[0].messages
    assert served[0].content == "how many superheroes"  # the user turn, compressed
    assert served[1].tool_calls is not None
    assert served[1].tool_calls[0].function.arguments == _TOOL_ARGUMENTS  # verbatim
    assert served[2].content == "42 rows"  # the tool result, verbatim
    assert _rows(log_path)[-1]["compressor_id"] == "truncate"


def test_mounting_a_mismatched_compression_artifact_fails_loudly(tmp_path: Path) -> None:
    # D-COMPRESS requirement A at the serving boundary: an endpoint whose bank was fitted on raw
    # text must not come up compressing. It would still answer every request, just with routing
    # quietly collapsed to the expensive fallback, which is the failure C2 measured and the one
    # nothing downstream would notice.
    policy = _knn_policy(tmp_path).model_copy(
        update={"compression": CompressionConfig(compressor_id="truncate", aggressiveness=0.5)}
    )
    with pytest.raises(ValueError, match="fitted on raw text"):
        EndpointRuntime(
            name="tau-bench",
            policy=policy,
            provider_factory=_EchoProvider,
            log=RequestLog(tmp_path / "requests.jsonl"),
        )


def test_the_stage_makes_one_compressor_call_per_request(tmp_path: Path) -> None:
    # Round trip dominates for an endpoint-backed compressor, so the stage must hand over a
    # request's whole segment set in ONE call rather than one call per message.
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    client, _, runtime, _ = _compressed_runtime(tmp_path, config)
    spy = _SpyCompressor(cast("Compressor", runtime._compressor))
    runtime._compressor = spy

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [
                {"role": "system", "content": "never compressed"},
                {"role": "user", "content": "first user turn here"},
                {"role": "assistant", "content": "an earlier reply"},
                {"role": "user", "content": "second user turn here"},
            ],
        },
    )

    assert response.status_code == 200
    # One call (a cold transcript: no affinity to reuse), carrying BOTH user turns.
    assert spy.calls == [["first user turn here", "second user turn here"]]


class _PricedCompressor:
    """A compressor with a real bill and a real wall clock, so the log fields have something
    non-zero to carry."""

    id = "priced-for-tests"
    version = "1"
    append_stable = True

    def compress(self, segments: list[str], config: CompressionConfig) -> CompressionResult:
        time.sleep(0.02)  # a stand-in for the endpoint round trip
        inner = get_compressor("truncate").compress(segments, config)
        return inner.model_copy(update={"cost_usd": 0.00054})


_PRICED = _PricedCompressor()


def test_the_compressors_bill_and_wall_clock_reach_the_log(tmp_path: Path) -> None:
    # They were computed and then dropped at the log boundary, which left savings crediting the
    # token reduction without ever subtracting what the reduction cost.
    register_compressor(_PRICED)
    config = CompressionConfig(compressor_id=_PRICED.id, aggressiveness=0.5)
    client, log_path, _, _ = _compressed_runtime(tmp_path, config)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "one two three four five six"}],
        },
    )

    assert response.status_code == 200
    row = _rows(log_path)[-1]
    assert row["compressor_cost_usd"] == pytest.approx(0.00054)
    assert cast("float", row["compressor_latency_s"]) >= 0.02
    # And the request's own clock SPANS the compression stage rather than starting after it.
    assert cast("float", row["latency_ms"]) >= 20.0


def test_an_uncompressed_row_carries_no_compressor_bill(tmp_path: Path) -> None:
    client, log_path = _client(tmp_path)
    client.post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": [{"role": "user", "content": "hi"}]},
    )
    row = _rows(log_path)[-1]
    assert row["compressor_cost_usd"] == 0.0
    assert row["compressor_latency_s"] == 0.0


def test_the_compressor_bill_reaches_the_log_on_the_error_path_too(tmp_path: Path) -> None:
    # The compressor was paid even though the provider call failed, so the row must say so.
    register_compressor(_PRICED)

    class _FailingProvider(_EchoProvider):
        def complete(
            self,
            system: str,
            messages: list[Message],
            *,
            temperature: float = 0.7,
            max_tokens: int = 8192,
        ) -> Completion:
            raise RuntimeError("upstream on fire")

    log_path = tmp_path / "requests.jsonl"
    policy = RoutingPolicy(
        kind="static",
        default_model="haiku-4-5",
        pool=_pool(),
        compression=CompressionConfig(compressor_id=_PRICED.id, aggressiveness=0.5),
    )
    runtime = EndpointRuntime(
        name="tau-bench", policy=policy, provider_factory=_FailingProvider, log=RequestLog(log_path)
    )
    response = TestClient(_app_for(runtime)).post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "one two three four five six"}],
        },
    )
    assert response.status_code == 502
    row = _rows(log_path)[-1]
    assert row["status"] == "error"
    assert row["compressor_cost_usd"] == pytest.approx(0.00054)


def _one_exchange(client: TestClient, content: str) -> None:
    """Drive one full request so the runtime remembers a compressed transcript for it."""
    response = client.post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": [{"role": "user", "content": content}]},
    )
    assert response.status_code == 200


def test_the_compressed_cache_is_bounded_by_bytes_not_by_entry_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A compressed transcript is the whole conversation, not a fingerprint: at the affinity
    # map's 4096-entry cap, measured 99KB conversations would be 0.41GB and tool-heavy ones
    # multiple GB. The cap is therefore in bytes, and the running total tracks the map.
    monkeypatch.setattr(chat_module, "_COMPRESSED_CAPACITY_BYTES", 2_000)
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.0)
    client, _, runtime, _ = _compressed_runtime(tmp_path, config)

    for index in range(12):
        _one_exchange(client, f"conversation {index} " + "padding word " * 40)

    with runtime._lock:
        stored = len(runtime._compressed)
        total = runtime._compressed_bytes
        measured = sum(size for _, size in runtime._compressed.values())
    assert 0 < stored < 12  # older conversations were evicted, newer ones kept
    assert total <= 2_000  # the bound actually binds
    assert total == measured  # the running total never drifts from what is held


def test_the_byte_total_is_repaired_when_a_conversation_is_re_remembered(tmp_path: Path) -> None:
    # remember() overwrites a key on each turn of the same conversation. Without subtracting the
    # previous size first, the total would climb forever and start evicting live entries.
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.0)
    client, _, runtime, _ = _compressed_runtime(tmp_path, config)
    _one_exchange(client, "alpha beta gamma delta")
    with runtime._lock:
        after_first = runtime._compressed_bytes
        measured_first = sum(size for _, size in runtime._compressed.values())
    _one_exchange(client, "alpha beta gamma delta")  # the identical request again
    with runtime._lock:
        assert runtime._compressed_bytes == after_first == measured_first
        assert runtime._compressed_bytes == sum(size for _, size in runtime._compressed.values())


def test_installing_a_different_compression_config_drops_stale_prefixes(tmp_path: Path) -> None:
    # A stored prefix carries the OUTGOING config's bytes. Reusing one after the config changed
    # would hand the provider a transcript that is half one compression config and half another.
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    client, _, runtime, _ = _compressed_runtime(tmp_path, config)
    _one_exchange(client, "alpha beta gamma delta epsilon zeta")
    with runtime._lock:
        assert runtime._compressed  # there is something to go stale

    runtime._install_policy(
        runtime.policy.model_copy(
            update={"compression": CompressionConfig(compressor_id="identity")}
        )
    )

    with runtime._lock:
        assert not runtime._compressed
        assert runtime._compressed_bytes == 0


def test_installing_the_same_compression_config_keeps_the_prefixes(tmp_path: Path) -> None:
    # The dial's actual behavior today: it carries `compression` through unchanged, and a dial
    # move must not throw away every warm prefix (and with it the provider's prompt cache).
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    client, _, runtime, _ = _compressed_runtime(tmp_path, config)
    _one_exchange(client, "alpha beta gamma delta epsilon zeta")
    with runtime._lock:
        before = dict(runtime._compressed)

    runtime._install_policy(runtime.policy.model_copy(update={"guard_margin": 0.01}))

    with runtime._lock:
        assert dict(runtime._compressed) == before


def test_a_pre_routing_failure_still_bills_the_compression_that_ran(tmp_path: Path) -> None:
    # Finding 13: the compressor ran on real hardware and was paid before routing blew up, so a
    # row encoding compressor_id="" with zero cost would not be an omission, it would be an
    # assertion that no compression happened. The savings math reads these rows.
    register_compressor(_PRICED)
    log_path = tmp_path / "requests.jsonl"
    policy = RoutingPolicy(
        kind="static",
        default_model="haiku-4-5",
        pool=_pool(),
        compression=CompressionConfig(compressor_id=_PRICED.id, aggressiveness=0.5),
    )

    def _explode(entry: PoolEntry) -> _EchoProvider:
        raise RuntimeError("api_key_env is unset")

    runtime = EndpointRuntime(
        name="tau-bench", policy=policy, provider_factory=_explode, log=RequestLog(log_path)
    )
    response = TestClient(_app_for(runtime)).post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "one two three four five six"}],
        },
    )

    assert response.status_code == 502
    row = _rows(log_path)[-1]
    assert row["routing_reason"] == "error-before-routing"
    assert row["compressor_id"] == _PRICED.id
    assert row["compressor_cost_usd"] == pytest.approx(0.00054)
    assert cast("float", row["compressor_latency_s"]) >= 0.02
    assert cast("int", row["tokens_in_compressed"]) < cast("int", row["tokens_in_raw"])


def test_a_failure_inside_the_compression_stage_bills_nothing(tmp_path: Path) -> None:
    # The other half of "the zero encoding must never lie": when the stage itself is what
    # failed, no compression was delivered and the row must say exactly that.
    class _Exploding:
        id = "exploding-for-tests"
        version = "1"
        append_stable = True

        def compress(self, segments: list[str], config: CompressionConfig) -> CompressionResult:
            raise RuntimeError("compressor endpoint unreachable")

    register_compressor(cast("Compressor", _Exploding()))
    log_path = tmp_path / "requests.jsonl"
    policy = RoutingPolicy(
        kind="static",
        default_model="haiku-4-5",
        pool=_pool(),
        compression=CompressionConfig(compressor_id="exploding-for-tests"),
    )
    runtime = EndpointRuntime(
        name="tau-bench", policy=policy, provider_factory=_EchoProvider, log=RequestLog(log_path)
    )
    response = TestClient(_app_for(runtime)).post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": [{"role": "user", "content": "hello there"}]},
    )

    assert response.status_code == 502
    row = _rows(log_path)[-1]
    assert row["compressor_id"] == ""
    assert row["compressor_cost_usd"] == 0.0
    assert row["tokens_in_raw"] == 0


def test_a_compressor_that_returns_the_wrong_segment_count_is_named(tmp_path: Path) -> None:
    # Finding 12 at the serving boundary: one segment too few used to surface as an anonymous
    # 502 from a StopIteration, one too many was served with the extra silently discarded.
    class _Splitter:
        id = "splitter-for-tests"
        version = "1"
        append_stable = True

        def compress(self, segments: list[str], config: CompressionConfig) -> CompressionResult:
            out = [part for segment in segments for part in segment.split(" ", 1)]
            return CompressionResult(
                segments=out, tokens_in_raw=1, tokens_in_compressed=1, latency_s=0.0
            )

    register_compressor(cast("Compressor", _Splitter()))
    log_path = tmp_path / "requests.jsonl"
    policy = RoutingPolicy(
        kind="static",
        default_model="haiku-4-5",
        pool=_pool(),
        compression=CompressionConfig(compressor_id="splitter-for-tests"),
    )
    runtime = EndpointRuntime(
        name="tau-bench", policy=policy, provider_factory=_EchoProvider, log=RequestLog(log_path)
    )
    response = TestClient(_app_for(runtime)).post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": [{"role": "user", "content": "two words"}]},
    )

    assert response.status_code == 502  # refused, not served with a corrupted transcript
    assert "splitter-for-tests" in _error_message(_rows(log_path)[-1])


def test_config_carries_the_pareto_curve_when_the_runtime_has_one(tmp_path: Path) -> None:
    from wmo.optimize.pareto import ParetoCurve, ParetoPoint

    curve = ParetoCurve(
        points=[
            ParetoPoint(
                id="cheap",
                kind="model",
                label="cheap",
                cost_per_completed_task_usd=0.01,
                mean_reward=0.5,
                task_success_rate=0.5,
                latency_p50_s=1.0,
                n_scenarios=2,
                n_scored=4,
                n_excluded=0,
                on_frontier=True,
            )
        ],
        recommended="cheap",
        complete=True,
        n_scenarios=2,
        provenance="wm_simulated",
        judge="test-judge",
    )
    policy = _knn_policy(tmp_path)
    runtime = EndpointRuntime(
        name="tau-bench",
        policy=policy,
        provider_factory=_EchoProvider,
        log=RequestLog(tmp_path / "requests.jsonl"),
        pareto=curve,
    )
    app = FastAPI()
    app.include_router(create_chat_router({"tau-bench": runtime}))
    body = TestClient(app).get("/v1/endpoints/tau-bench/config").json()

    assert body["pareto"]["recommended"] == "cheap"
    assert body["pareto"]["complete"] is True
    assert body["pareto"]["points"][0]["on_frontier"] is True
    # The per-workload curve and the global ours9 anchors travel side by side, distinct.
    assert [anchor["s"] for anchor in body["anchors"]] == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_config_omits_pareto_for_artifacts_that_predate_it(tmp_path: Path) -> None:
    client, _ = _dial_client(tmp_path, _knn_policy(tmp_path))
    body = client.get("/v1/endpoints/tau-bench/config").json()

    assert body["pareto"] is None


def test_structured_usage_prices_the_anthropic_translators_emission() -> None:
    """The Anthropic/Bedrock translators emit the OpenAI details shape plus the
    Anthropic write field; serving must price both cache legs from it (missing
    them measured a 5.8x overcharge on a 9k-cached agent turn)."""
    from wmo.serving.chat import _structured_usage

    response = ChatResponse.model_validate(
        {
            "model": "claude-sonnet-5",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 9600,
                "completion_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 9000},
                "cache_read_input_tokens": 9000,
                "cache_creation_input_tokens": 500,
            },
        }
    )

    usage = _structured_usage(response)

    assert usage.cached_input_tokens == 9000
    assert usage.cache_write_input_tokens == 500
    assert usage.input_tokens == 9600  # totals unchanged: reads/writes are a SUBSET


def test_structured_usage_still_prefers_the_openai_shape() -> None:
    from wmo.serving.chat import _structured_usage

    response = ChatResponse.model_validate(
        {
            "model": "gpt-5.5",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 40},
            },
        }
    )

    assert _structured_usage(response).cached_input_tokens == 40
