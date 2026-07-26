"""Tests for the platform HTTP client (httpx mock transport, no network)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from llm_waterfall.types import ChatMessage, ChatRequest

from wmh.core.types import Action, ActionKind
from wmh.platform.client import (
    PlatformClient,
    PlatformError,
    RemoteAgentSession,
    fetch_cli_config,
)

API_URL = "https://api.test"

_WHOAMI = {
    "actor": {"kind": "api_key", "id": "api-key:org-1"},
    "orgs": [{"id": "org-1", "slug": "acme", "name": "Acme"}],
}


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> PlatformClient:
    return PlatformClient(API_URL, "xpl_secret", transport=httpx.MockTransport(handler))


def test_whoami_parses_and_sends_bearer_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer xpl_secret"
        assert request.url.path == "/api/whoami"
        return httpx.Response(200, json=_WHOAMI)

    with _client(handler) as client:
        identity = client.whoami()

    assert identity.actor.kind == "api_key"
    assert identity.orgs[0].slug == "acme"


def test_error_payloads_become_platform_errors_with_status() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "Organization not found: o"})

    with (
        _client(handler) as client,
        pytest.raises(PlatformError, match="Organization not found") as info,
    ):
        client.whoami()
    assert info.value.status_code == 404


def test_401_error_suggests_logging_in() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "Unauthorized"})

    with _client(handler) as client, pytest.raises(PlatformError, match="wmh login"):
        client.whoami()


def test_unified_run_target_and_world_model_session_payloads() -> None:
    """The run client resolves once, then uses the hosted world-model session API."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/api/run-targets/wm-1":
            return httpx.Response(
                200,
                json={
                    "id": "wm-1",
                    "kind": "world_model",
                    "org_id": "org-1",
                    "name": "tau",
                    "status": "ready",
                },
            )
        if request.url.path == "/api/world-models/wm-1/sessions":
            assert json.loads(request.read()) == {"task": "book a flight"}
            return httpx.Response(
                201,
                json={"id": "sess-1", "world_model_id": "wm-1", "status": "active"},
            )
        assert request.url.path == "/api/sessions/sess-1/step"
        body = json.loads(request.read())
        assert body["action"] == {
            "kind": "tool_call",
            "name": "search",
            "arguments": {"q": "SFO"},
            "content": None,
        }
        return httpx.Response(200, json={"observation": {"content": "three flights"}})

    with _client(handler) as client:
        target = client.resolve_run_target("wm-1")
        session = client.create_world_model_session(target.id, task="book a flight")
        observation = client.step_world_model_session(
            session.id,
            Action(kind=ActionKind.TOOL_CALL, name="search", arguments={"q": "SFO"}),
        )

    assert target.kind == "world_model"
    assert observation.content == "three flights"
    assert seen == [
        "/api/run-targets/wm-1",
        "/api/world-models/wm-1/sessions",
        "/api/sessions/sess-1/step",
    ]


def test_hosted_agent_session_uses_regular_create_and_workspace_patch_routes() -> None:
    """Agent runs use regular sessions plus the live workspace patch protocol."""
    seen: list[str] = []
    session = {
        "id": "sess-1",
        "agent_id": "agent-1",
        "status": "starting",
        "workspace_sync": True,
        "launched_from": "cli",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        path = request.url.path
        if path.endswith("/workspace-uploads"):
            body = request.read()
            assert b"archive-bytes" in body
            return httpx.Response(201, json={"id": "upload-1"})
        if path.endswith("/sessions") and request.method == "POST":
            assert json.loads(request.read()) == {
                "instruction": "fix the tests",
                "workspace_upload_id": "upload-1",
            }
            return httpx.Response(202, json=session)
        if path.endswith("/events"):
            assert request.url.params["after"] == "0"
            return httpx.Response(
                200,
                json={
                    "events": [
                        {"seq": 1, "kind": "assistant_message", "payload": {"text": "done"}}
                    ],
                    "last_seq": 1,
                    "status": "running",
                },
            )
        if path.endswith("/commands"):
            assert json.loads(request.read()) == {"kind": "user_message", "text": "continue"}
            return httpx.Response(202, json={"command_id": 1})
        if path.endswith("/workspace/patches/patch-7/ack"):
            return httpx.Response(204)
        if path.endswith("/workspace/patches/patch-7"):
            return httpx.Response(200, content=b"remote-patch")
        if path.endswith("/workspace/patches"):
            body = request.read()
            assert b"local-patch" in body
            return httpx.Response(200, json={"applied": ["local.txt"], "conflicts": []})
        if path.endswith("/workspace/ack"):
            return httpx.Response(204)
        if path.endswith("/workspace"):
            return httpx.Response(200, content=b"final-archive")
        return httpx.Response(200, json={**session, "status": "ended"})

    with _client(handler) as client:
        created = client.create_agent_session(
            "agent-1", workspace=b"archive-bytes", instruction="fix the tests"
        )
        page = client.list_agent_session_events("agent-1", created.id, after=0)
        client.post_agent_session_command("agent-1", created.id, "user_message", text="continue")
        current = client.get_agent_session("agent-1", created.id)
        patch_result = client.upload_agent_workspace_patch("agent-1", created.id, b"local-patch")
        patch = client.download_agent_workspace_patch("agent-1", created.id, "patch-7")
        client.acknowledge_agent_workspace_patch("agent-1", created.id, "patch-7")
        final = client.download_agent_workspace("agent-1", created.id)
        client.acknowledge_agent_workspace("agent-1", created.id)

    assert created.workspace_sync
    assert page.events[0].payload["text"] == "done"
    assert current.status == "ended"
    assert patch_result.applied == ["local.txt"]
    assert patch == b"remote-patch"
    assert final == b"final-archive"
    assert seen == [
        "POST /api/agents/agent-1/workspace-uploads",
        "POST /api/agents/agent-1/sessions",
        "GET /api/agents/agent-1/sessions/sess-1/events",
        "POST /api/agents/agent-1/sessions/sess-1/commands",
        "GET /api/agents/agent-1/sessions/sess-1",
        "POST /api/agents/agent-1/sessions/sess-1/workspace/patches",
        "GET /api/agents/agent-1/sessions/sess-1/workspace/patches/patch-7",
        "POST /api/agents/agent-1/sessions/sess-1/workspace/patches/patch-7/ack",
        "GET /api/agents/agent-1/sessions/sess-1/workspace",
        "POST /api/agents/agent-1/sessions/sess-1/workspace/ack",
    ]


def test_hosted_agent_session_without_workspace_posts_create_directly() -> None:
    """The default agent run does not stage any local workspace upload."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        assert request.url.path == "/api/agents/agent-1/sessions"
        assert json.loads(request.read()) == {"instruction": "remote task"}
        return httpx.Response(
            202,
            json={
                "id": "sess-1",
                "agent_id": "agent-1",
                "status": "starting",
                "workspace_sync": False,
                "launched_from": "cli",
            },
        )

    with _client(handler) as client:
        created = client.create_agent_session("agent-1", workspace=None, instruction="remote task")

    assert not created.workspace_sync
    assert seen == ["POST /api/agents/agent-1/sessions"]


def test_detached_session_resolution_and_end_routes() -> None:
    """A bare session id resolves to its agent, and end uses the reconciling route."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        if request.url.path == "/api/agent-sessions/sess-9":
            return httpx.Response(
                200,
                json={
                    "id": "sess-9",
                    "agent_id": "agent-2",
                    "status": "running",
                    "workspace_sync": False,
                    "launched_from": "cli",
                },
            )
        assert request.url.path == "/api/agents/agent-2/sessions/sess-9/end"
        return httpx.Response(
            202,
            json={
                "id": "sess-9",
                "agent_id": "agent-2",
                "status": "failed",
                "workspace_sync": False,
                "launched_from": "cli",
                "ended_reason": "stalled",
                "error": "driver heartbeat went stale",
            },
        )

    with _client(handler) as client:
        resolved = client.resolve_agent_session("sess-9")
        ended = client.end_agent_session(resolved.agent_id, resolved.id)

    assert resolved.agent_id == "agent-2"
    assert resolved.status == "running"
    assert ended.status == "failed"
    assert ended.ended_reason == "stalled"
    assert seen == [
        "GET /api/agent-sessions/sess-9",
        "POST /api/agents/agent-2/sessions/sess-9/end",
    ]


def test_hosted_agent_session_accepts_future_launch_origins() -> None:
    """A compatible platform origin extension does not break session decoding."""
    session = RemoteAgentSession.model_validate(
        {
            "id": "sess-1",
            "agent_id": "agent-1",
            "status": "running",
            "workspace_sync": False,
            "launched_from": "automation",
        }
    )

    assert session.launched_from == "automation"


def test_builtin_local_pi_run_payloads() -> None:
    """The built-in harness has an org-scoped, metered platform worker path."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/api/orgs/org-1/local-pi-runs":
            return httpx.Response(
                201,
                json={
                    "id": "run-1",
                    "org_id": "org-1",
                    "status": "running",
                    "worker_provider": "bedrock",
                    "worker_model": "claude-haiku-4-5",
                },
            )
        if request.url.path.endswith("/worker-completion"):
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )
        assert request.url.path.endswith("/finish")
        assert json.loads(request.read()) == {
            "status": "ended",
            "ended_reason": "user_ended",
            "error": None,
        }
        return httpx.Response(202, json={})

    with _client(handler) as client:
        run = client.create_local_pi_run("org-1")
        response = client.complete_local_pi_worker(
            "org-1",
            run.id,
            ChatRequest(messages=[ChatMessage(role="user", content="hi")]),
        )
        client.finish_local_pi_run("org-1", run.id, status="ended", ended_reason="user_ended")

    assert response.choices[0].message.content == "ok"
    assert seen == [
        "/api/orgs/org-1/local-pi-runs",
        "/api/orgs/org-1/local-pi-runs/run-1/worker-completion",
        "/api/orgs/org-1/local-pi-runs/run-1/finish",
    ]


def test_push_model_bundle_runs_ticket_put_finalize(tmp_path: Path) -> None:
    bundle_path = tmp_path / "tau-bench.tar.gz"
    bundle_path.write_bytes(b"bundle-bytes")
    digest = hashlib.sha256(b"bundle-bytes").hexdigest()
    seen: dict[str, object] = {}
    finalize_body: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/orgs/org-1/world-models/tau-bench/bundle/uploads":
            return httpx.Response(
                201,
                json={
                    "upload_url": "https://storage.test/upload/staging/cli/abc.tar.gz?token=t",
                    "token": "t",
                    "staging_path": "staging/cli/abc.tar.gz",
                },
            )
        if request.url.host == "storage.test":
            seen["put_body"] = request.read()
            seen["put_method"] = request.method
            return httpx.Response(200, json={"Key": "abc"})
        assert request.url.path == "/api/orgs/org-1/world-models/tau-bench/bundle"
        finalize_body.update(json.loads(request.read()))
        return httpx.Response(201, json={"id": "wm-1", "name": "tau-bench", "status": "ready"})

    with _client(handler) as client:
        pushed = client.push_model_bundle(
            "org-1",
            "tau-bench",
            bundle_path,
            digest,
            len(b"bundle-bytes"),
            {"serve_provider": "anthropic"},
        )

    assert pushed.status == "ready"
    assert seen["put_method"] == "PUT"
    assert seen["put_body"] == b"bundle-bytes"
    assert finalize_body["staging_path"] == "staging/cli/abc.tar.gz"
    assert finalize_body["sha256"] == digest
    assert finalize_body["meta"] == {"serve_provider": "anthropic"}


def test_push_model_bundle_surfaces_storage_upload_failure(tmp_path: Path) -> None:
    bundle_path = tmp_path / "tau-bench.tar.gz"
    bundle_path.write_bytes(b"bundle-bytes")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/bundle/uploads"):
            return httpx.Response(
                201,
                json={
                    "upload_url": "https://storage.test/upload/x?token=t",
                    "token": "t",
                    "staging_path": "staging/cli/x.tar.gz",
                },
            )
        return httpx.Response(413, text="Payload too large")

    with (
        _client(handler) as client,
        pytest.raises(PlatformError, match="upload to storage failed"),
    ):
        client.push_model_bundle("org-1", "tau-bench", bundle_path, "0" * 64, 12, {})


def _download_handler(
    content: bytes, declared_sha256: str
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "storage.test":
            return httpx.Response(200, content=content)
        assert request.url.path == "/api/orgs/org-1/world-models/tau-bench/bundle"
        return httpx.Response(
            200,
            json={
                "url": "https://storage.test/signed/bundle.tar.gz?token=t",
                "sha256": declared_sha256,
                "byte_size": len(content),
                "artifact_id": "artifact-1",
                "expires_in": 600,
            },
        )

    return handler


def test_download_model_bundle_streams_and_verifies_digest(tmp_path: Path) -> None:
    content = b"bundle-bytes"
    dest = tmp_path / "tau-bench.tar.gz"

    handler = _download_handler(content, hashlib.sha256(content).hexdigest())
    with _client(handler) as client:
        digest = client.download_model_bundle("org-1", "tau-bench", dest)

    assert dest.read_bytes() == content
    assert digest == hashlib.sha256(content).hexdigest()


def test_download_model_bundle_rejects_digest_mismatch(tmp_path: Path) -> None:
    dest = tmp_path / "tau-bench.tar.gz"
    handler = _download_handler(b"bundle-bytes", "0" * 64)
    with _client(handler) as client, pytest.raises(PlatformError, match="digest mismatch"):
        client.download_model_bundle("org-1", "tau-bench", dest)
    assert not dest.exists()


def test_harness_round_trip_payloads() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            assert request.url.path == "/api/orgs/org-1/harnesses/agent/versions"
            payload = json.loads(request.read())
            assert payload["doc_hash"] == "a" * 32
            return httpx.Response(
                201,
                json={"name": "agent", "version": 3, "doc_hash": "a" * 32, "created": True},
            )
        if request.url.path == "/api/orgs/org-1/harnesses/agent/versions/3":
            return httpx.Response(
                200, json={"version": 3, "doc": {"name": "agent"}, "doc_hash": "a" * 32}
            )
        assert request.url.path == "/api/orgs/org-1/harnesses/agent"
        return httpx.Response(
            200,
            json={
                "harness": {"id": "h-1", "name": "agent", "latest_version": 3},
                "versions": [{"version": 3, "doc_hash": "a" * 32}],
            },
        )

    with _client(handler) as client:
        pushed = client.push_harness_version("org-1", "agent", {"name": "agent"}, "a" * 32)
        assert pushed.version == 3
        assert pushed.created

        harness, versions = client.get_harness("org-1", "agent")
        assert harness.latest_version == 3
        assert versions[0].doc_hash == "a" * 32

        doc = client.get_harness_version("org-1", "agent", 3)
        assert doc.doc == {"name": "agent"}


def test_fetch_cli_config_reads_the_api_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://platform.test/api/cli/config"
        return httpx.Response(200, json={"apiUrl": "https://api.test/"})

    api_url = fetch_cli_config("https://platform.test/", transport=httpx.MockTransport(handler))
    assert api_url == "https://api.test"


def test_fetch_cli_config_maps_failures() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    with pytest.raises(PlatformError, match="discovery failed"):
        fetch_cli_config("https://platform.test", transport=httpx.MockTransport(handler))


def test_get_world_model_reads_one_record_by_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/world-models/wm-9"
        return httpx.Response(
            200,
            json={"id": "wm-9", "name": "postgres", "status": "ready", "config": {"depth": 2}},
        )

    with _client(handler) as client:
        model = client.get_world_model("wm-9")

    assert (model.id, model.name, model.status) == ("wm-9", "postgres", "ready")
