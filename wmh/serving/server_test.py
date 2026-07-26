"""Tests for the FastAPI serving layer, with injected in-process WorldModels (no network)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from wmh.config.card import CardCorpus, ModelCard, TracesSource
from wmh.core.types import Action, ActionKind, Observation, Step, Trace
from wmh.engine.knowledge import KnowledgeBase
from wmh.engine.world_model import WorldModel
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder
from wmh.serving.builds import BuildManager
from wmh.serving.server import _load_card_or_none, create_app, resolve_model_dirs


class FakeProvider:
    def __init__(self) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        return Completion(text='{"output": "user found", "is_error": false}')

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


def _world_model() -> WorldModel:
    retriever = EmbeddingRetriever(HashingEmbedder(dim=32))
    retriever.index(
        [
            Trace(
                trace_id="t",
                steps=[
                    Step(
                        action=Action(
                            kind=ActionKind.TOOL_CALL, name="get_user", arguments={"id": "u1"}
                        ),
                        observation=Observation(content="found u1"),
                    )
                ],
            )
        ]
    )
    return WorldModel(FakeProvider(), retriever, top_k=3)


def _client(world_models: dict[str, WorldModel] | None = None) -> TestClient:
    models = world_models or {"airline": _world_model()}
    return TestClient(create_app(world_models=models))


def test_healthz() -> None:
    assert _client().get("/healthz").json() == {"status": "ok"}


def test_lists_world_models_by_name() -> None:
    client = _client({"airline": _world_model(), "retail": _world_model()})
    body = client.get("/world_models").json()
    assert body["world_models"] == ["airline", "retail"]


def test_lists_cards_alongside_names() -> None:
    card = ModelCard(
        name="airline",
        title="Airline",
        corpus=CardCorpus(traces=1, steps=2),
        provider="bedrock",
        model_id="m",
    )
    app = create_app(world_models={"airline": _world_model()}, cards={"airline": card})
    body = TestClient(app).get("/world_models").json()
    assert body["models"] == [{"name": "airline", "card": card.model_dump()}]


def test_models_without_cards_list_null_cards() -> None:
    body = _client().get("/world_models").json()
    assert body["models"] == [{"name": "airline", "card": None}]


def _fake_artifact(root: Path, name: str) -> None:
    model_dir = root / "models" / name
    model_dir.mkdir(parents=True)
    (model_dir / "config.toml").write_text("", encoding="utf-8")


def test_resolve_model_dirs_merges_roots(tmp_path: Path) -> None:
    _fake_artifact(tmp_path / "a", "m1")
    _fake_artifact(tmp_path / "b", "m2")
    resolved = resolve_model_dirs([str(tmp_path / "a"), str(tmp_path / "b")], None)
    assert sorted(resolved) == ["m1", "m2"]


def test_resolve_model_dirs_rejects_name_collision(tmp_path: Path) -> None:
    _fake_artifact(tmp_path / "a", "m1")
    _fake_artifact(tmp_path / "b", "m1")
    with pytest.raises(ValueError, match="m1"):
        resolve_model_dirs([str(tmp_path / "a"), str(tmp_path / "b")], None)


def test_resolve_model_dirs_ignores_collision_outside_requested_names(tmp_path: Path) -> None:
    _fake_artifact(tmp_path / "a", "dup")
    _fake_artifact(tmp_path / "b", "dup")
    _fake_artifact(tmp_path / "a", "wanted")
    resolved = resolve_model_dirs([str(tmp_path / "a"), str(tmp_path / "b")], ["wanted"])
    assert sorted(resolved) == ["wanted"]


def test_load_card_or_none_degrades_malformed_card(tmp_path: Path) -> None:
    (tmp_path / "card.json").write_text("{ broken", encoding="utf-8")
    assert _load_card_or_none(tmp_path) is None


def _ok_build_fn(config, *, file: str, root: str, reporter) -> None:  # noqa: ANN001 - test stub
    reporter.ingest_done(1, 1)
    Path(root).mkdir(parents=True, exist_ok=True)
    (Path(root) / "config.toml").write_text("", encoding="utf-8")


def _build_client(tmp_path: Path) -> TestClient:
    manager = BuildManager(
        store_root=tmp_path / ".wmh",
        build_fn=_ok_build_fn,
        verify_fn=lambda config: None,
        register=lambda name, model_dir: None,
    )
    return TestClient(create_app(world_models={"airline": _world_model()}, build_manager=manager))


def test_build_routes_run_a_build_and_stream_events(tmp_path: Path) -> None:
    client = _build_client(tmp_path)
    upload = client.post(
        "/world_models/builds/uploads",
        files={"file": ("t.jsonl", b"{}\n", "application/jsonl")},
    )
    started = client.post(
        "/world_models/builds", json={"name": "fresh", "file": upload.json()["file"]}
    )
    assert started.status_code == 202
    build_id = started.json()["build_id"]
    # Poll until the background build reaches a terminal state.
    import time

    for _ in range(50):
        if client.get(f"/world_models/builds/{build_id}").json()["status"] != "running":
            break
        time.sleep(0.05)
    assert client.get(f"/world_models/builds/{build_id}").json()["status"] == "succeeded"
    with client.stream("GET", f"/world_models/builds/{build_id}/events") as stream:
        assert stream.headers["content-type"].startswith("text/event-stream")
        text = "".join(stream.iter_text())
    assert '"type": "done"' in text


def test_build_routes_unavailable_without_manager() -> None:
    client = _client()
    resp = client.post("/world_models/builds", json={"name": "x", "file": "/nope"})
    assert resp.status_code == 503


def test_build_route_rejects_server_local_path(tmp_path: Path) -> None:
    client = _build_client(tmp_path)
    secret = tmp_path / "secret.jsonl"
    secret.write_text("{}\n", encoding="utf-8")

    response = client.post("/world_models/builds", json={"name": "fresh", "file": str(secret)})

    assert response.status_code == 422
    assert "upload" in response.json()["detail"]


def test_upload_rejects_foreign_origin(tmp_path: Path) -> None:
    client = _build_client(tmp_path)
    resp = client.post(
        "/world_models/builds/uploads",
        files={"file": ("t.jsonl", b"{}\n", "application/jsonl")},
        headers={"Origin": "https://evil.example.com"},
    )
    assert resp.status_code == 403
    ok = client.post(
        "/world_models/builds/uploads",
        files={"file": ("t.jsonl", b"{}\n", "application/jsonl")},
        headers={"Origin": "http://localhost:6001"},
    )
    assert ok.status_code == 200


def test_upload_returns_opaque_filename(tmp_path: Path) -> None:
    client = _build_client(tmp_path)
    response = client.post(
        "/world_models/builds/uploads",
        files={"file": ("traces.jsonl", b"{}\n", "application/jsonl")},
    )

    assert response.status_code == 200
    upload = response.json()["file"]
    assert Path(upload).name == upload


def test_session_lifecycle_and_step_are_namespaced() -> None:
    client = _client()
    resp = client.post("/world_models/airline/sessions", json={"task": "look up a user"})
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]

    step = client.post(
        f"/world_models/airline/sessions/{session_id}/step",
        json={"action": {"kind": "tool_call", "name": "get_user", "arguments": {"id": "u2"}}},
    )
    assert step.status_code == 200
    assert step.json()["observation"]["content"] == "user found"

    got = client.get(f"/world_models/airline/sessions/{session_id}")
    assert got.status_code == 200
    assert len(got.json()["history"]) == 1


def test_unknown_world_model_is_404() -> None:
    client = _client()
    resp = client.post("/world_models/nope/sessions", json={"task": "x"})
    assert resp.status_code == 404


def test_step_on_missing_session_is_404() -> None:
    client = _client()
    resp = client.post(
        "/world_models/airline/sessions/nope/step",
        json={"action": {"kind": "message", "content": "hi"}},
    )
    assert resp.status_code == 404


def test_sessions_are_isolated_between_named_models() -> None:
    client = _client({"airline": _world_model(), "retail": _world_model()})
    created = client.post("/world_models/airline/sessions", json={"task": "x"})
    session_id = created.json()["session_id"]
    # A session created on `airline` is not visible under `retail`.
    miss = client.get(f"/world_models/retail/sessions/{session_id}")
    assert miss.status_code == 404


class RewardJudgeProvider(FakeProvider):
    """Replies like the reward judge (JSON episode score) instead of an env observation."""

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        return Completion(
            text='{"success": true, "reward": 0.8, "step_rewards": [0.6], "critique": "solid"}'
        )


def _rewarded_world_model() -> WorldModel:
    retriever = EmbeddingRetriever(HashingEmbedder(dim=32))
    return WorldModel(FakeProvider(), retriever, top_k=3, reward_provider=RewardJudgeProvider())


def test_score_session_returns_episode_score_and_stamps_final_reward() -> None:
    wm = _rewarded_world_model()
    client = TestClient(create_app(world_models={"airline": wm}))
    session_id = client.post(
        "/world_models/airline/sessions", json={"task": "find user u1"}
    ).json()["session_id"]
    client.post(
        f"/world_models/airline/sessions/{session_id}/step",
        json={"action": {"kind": "tool_call", "name": "get_user", "arguments": {"id": "u1"}}},
    )
    response = client.post(f"/world_models/airline/sessions/{session_id}/score")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["reward"] == 0.8
    assert body["step_rewards"] == [0.6]
    assert body["critique"] == "solid"
    # the scalar also lands on the final step's observation (replay-buffer visibility)
    session = client.get(f"/world_models/airline/sessions/{session_id}").json()
    assert session["history"][-1]["observation"]["reward"] == 0.8


def test_score_unknown_session_is_404() -> None:
    client = _client()
    assert client.post("/world_models/airline/sessions/nope/score").status_code == 404


def test_end_session_returns_usage_and_frees_the_session() -> None:
    client = _client()
    session_id = client.post("/world_models/airline/sessions", json={}).json()["session_id"]
    response = client.delete(f"/world_models/airline/sessions/{session_id}")
    assert response.status_code == 200
    assert "events" in response.json() or "run_id" in response.json()
    assert client.get(f"/world_models/airline/sessions/{session_id}").status_code == 404


def _knowledge_world_model(tmp_path) -> WorldModel:  # noqa: ANN001 - pytest fixture
    kb = KnowledgeBase(tmp_path / "knowledge")
    kb.write_file("rules.md", "- gate: auth required")
    retriever = EmbeddingRetriever(HashingEmbedder(dim=32))
    return WorldModel(FakeProvider(), retriever, top_k=1, knowledge=kb)


def test_knowledge_read_lists_files(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
    client = _client({"airline": _knowledge_world_model(tmp_path)})
    response = client.get("/world_models/airline/knowledge")
    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is True
    assert body["files"]["rules.md"] == "- gate: auth required"


def test_knowledge_read_on_model_without_kb_reports_disabled() -> None:
    client = _client()
    response = client.get("/world_models/airline/knowledge")
    assert response.status_code == 200
    assert response.json() == {"enabled": False, "files": {}}


def test_knowledge_put_replaces_one_file(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
    client = _client({"airline": _knowledge_world_model(tmp_path)})
    put = client.put(
        "/world_models/airline/knowledge/rules.md",
        json={"content": "- gate: auth AND ownership required"},
    )
    assert put.status_code == 200
    body = client.get("/world_models/airline/knowledge").json()
    assert body["files"]["rules.md"] == "- gate: auth AND ownership required"


def test_knowledge_put_without_kb_is_a_clear_conflict() -> None:
    client = _client()
    response = client.put("/world_models/airline/knowledge/rules.md", json={"content": "x"})
    assert response.status_code == 409
    assert "knowledge" in response.json()["detail"]


def test_knowledge_put_rejects_non_markdown_names(tmp_path) -> None:  # noqa: ANN001
    client = _client({"airline": _knowledge_world_model(tmp_path)})
    response = client.put("/world_models/airline/knowledge/evil.txt", json={"content": "x"})
    assert response.status_code == 400


def test_traces_none_for_plain_injected_model() -> None:
    body = _client().get("/world_models/airline/traces").json()
    assert body["source"] == "none"
    assert body["downloadable"] is False
    assert body["scenarios"] == []


def test_traces_downloadable_when_card_declares_hub_source() -> None:
    card = ModelCard(
        name="airline",
        title="Airline",
        corpus=CardCorpus(traces=1, steps=2),
        provider="bedrock",
        model_id="m",
        traces_hf=TracesSource(repo="org/wmh-airline", path="traces.otel.jsonl"),
    )
    app = create_app(world_models={"airline": _world_model()}, cards={"airline": card})
    body = TestClient(app).get("/world_models/airline/traces").json()
    assert body["downloadable"] is True
    assert body["source"] == "hub"


def test_download_traces_400_without_source() -> None:
    resp = _client().post("/world_models/airline/traces/download")
    assert resp.status_code == 400
def test_score_accepts_a_rubric_body_and_stays_backward_compatible() -> None:
    wm = _rewarded_world_model()
    client = TestClient(create_app(world_models={"airline": wm}))
    session_id = client.post("/world_models/airline/sessions", json={"task": "t"}).json()[
        "session_id"
    ]
    client.post(
        f"/world_models/airline/sessions/{session_id}/step",
        json={"action": {"kind": "tool_call", "name": "get_user", "arguments": {}}},
    )
    # with a rubric body
    assert (
        client.post(
            f"/world_models/airline/sessions/{session_id}/score",
            json={"rubric": "complete iff X"},
        ).status_code
        == 200
    )
    # and without one (existing consumers unchanged)
    assert client.post(f"/world_models/airline/sessions/{session_id}/score").status_code == 200
