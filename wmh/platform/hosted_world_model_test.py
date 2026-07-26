"""Tests for the hosted world-model environment (httpx mock transport, no network)."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx

from wmh.core.types import Action, ActionKind
from wmh.platform.client import PlatformClient
from wmh.platform.hosted_world_model import HostedWorldModelSource

API_URL = "https://api.test"


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> PlatformClient:
    return PlatformClient(API_URL, "xpl_secret", transport=httpx.MockTransport(handler))


def _session_response(session_id: str) -> httpx.Response:
    return httpx.Response(
        201, json={"id": session_id, "world_model_id": "wm-1", "status": "active"}
    )


def test_each_rollout_opens_its_own_hosted_session_and_steps_it() -> None:
    requests: list[tuple[str, object]] = []
    counter = iter(("sess-1", "sess-2"))

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        requests.append((request.url.path, body))
        if request.url.path.endswith("/sessions"):
            return _session_response(next(counter))
        return httpx.Response(200, json={"observation": {"content": "ok", "exit_code": 0}})

    with _client(handler) as client:
        source = HostedWorldModelSource(client, "wm-1", label="postgres")
        first = source.open("migrate the db")
        observation = first.execute(
            Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": "ls"})
        )
        first.close()
        second = source.open("migrate the db")
        second.execute(Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": "ls"}))
        second.close()

    assert observation.content == "ok"
    assert source.label == "postgres"
    assert [path for path, _ in requests] == [
        "/api/world-models/wm-1/sessions",
        "/api/sessions/sess-1/step",
        "/api/world-models/wm-1/sessions",
        "/api/sessions/sess-2/step",
    ]
    # Every rollout carries the task and steps its OWN session: attempt k must not inherit
    # attempt k-1's server-side state.
    assert requests[0][1] == {"task": "migrate the db"}
    assert requests[1][1] == {
        "action": {
            "kind": "tool_call",
            "name": "bash",
            "arguments": {"command": "ls"},
            "content": None,
        }
    }


def test_freezing_a_hosted_source_is_a_no_op() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover - never called
        raise AssertionError("freezing must not talk to the platform")

    with _client(handler) as client, HostedWorldModelSource(client, "wm-1", label="x").frozen():
        pass
