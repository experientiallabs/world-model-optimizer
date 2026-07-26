"""Typed HTTP client for the platform's CLI registry surface.

Every call carries the org API key as a bearer credential; the platform scopes
reads and writes to that key's organization at member strength. Error payloads
are the platform's uniform ``{"error": message}`` shape, surfaced as
:class:`PlatformError` with the HTTP status attached.
"""

from __future__ import annotations

import hashlib
import json
from importlib import metadata
from pathlib import Path
from typing import Literal

import httpx
from llm_waterfall import ChatRequest, ChatResponse
from pydantic import BaseModel

from wmh.core.types import Action, JsonValue, Observation

_TIMEOUT_SECONDS = 120.0
_WORKSPACE_TIMEOUT_SECONDS = 300.0


class PlatformError(RuntimeError):
    """A platform request failed; carries the HTTP status when one exists."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ActorInfo(BaseModel):
    """Who the platform resolved the credential to."""

    kind: str  # "api_key" | "user"
    id: str


class OrgInfo(BaseModel):
    """One organization visible to the credential."""

    id: str
    slug: str
    name: str


class WhoAmI(BaseModel):
    """Response of ``GET /api/whoami``."""

    actor: ActorInfo
    orgs: list[OrgInfo]


class RemoteWorldModel(BaseModel):
    """The slice of a world-model row the CLI presents."""

    id: str
    name: str
    display_name: str | None = None
    status: str
    updated_at: str | None = None


class RemoteHarness(BaseModel):
    """The slice of a registry harness row the CLI presents."""

    id: str
    name: str
    latest_version: int
    updated_at: str | None = None


class RemoteHarnessVersion(BaseModel):
    """One doc-less entry of a harness's version lineage."""

    version: int
    doc_hash: str
    created_at: str | None = None


class HarnessVersionDoc(BaseModel):
    """One full harness version, doc included."""

    version: int
    doc: dict[str, JsonValue]
    doc_hash: str


class PushedHarnessVersion(BaseModel):
    """Response of a harness push: the version the doc landed as."""

    name: str
    version: int
    doc_hash: str
    created: bool  # False when the push was an idempotent repeat of the tip


class RunTarget(BaseModel):
    """A platform id resolved to one executable resource kind."""

    id: str
    kind: Literal["world_model", "agent"]
    org_id: str
    name: str
    display_name: str | None = None
    status: str


class RemoteWorldModelSession(BaseModel):
    """The slice of a hosted world-model session needed by ``wmh run``."""

    id: str
    world_model_id: str
    status: str


class RemoteAgentSession(BaseModel):
    """Hosted E2B agent session state needed by the CLI driver."""

    id: str
    agent_id: str
    status: str
    workspace_sync: bool
    launched_from: str
    starting_detail: str | None = None
    ended_reason: str | None = None
    error: str | None = None


class RemoteAgentSessionEvent(BaseModel):
    """One durable event from a hosted agent session transcript."""

    seq: int
    kind: Literal[
        "user_message",
        "assistant_message",
        "tool_call",
        "tool_output",
        "tool_result",
        "submit",
        "state",
        "status",
        "error",
        "workspace_patch",
    ]
    payload: dict[str, JsonValue]


class RemoteAgentEventPage(BaseModel):
    """One poll page of hosted transcript events and current session status."""

    events: list[RemoteAgentSessionEvent]
    last_seq: int
    status: str


class WorkspacePatchResult(BaseModel):
    """Paths accepted or rejected while applying a live workspace patch."""

    applied: list[str]
    conflicts: list[str]


class LocalPiRunInfo(BaseModel):
    """An org-scoped platform usage record for the built-in local pi harness."""

    id: str
    org_id: str
    status: str
    worker_provider: str
    worker_model: str


def fetch_cli_config(web_url: str, *, transport: httpx.BaseTransport | None = None) -> str | None:
    """Ask the web app which backend host the CLI should call.

    ``GET {web_url}/api/cli/config`` is public: the backend URL is not a
    secret (every Endpoints page shows it) and everything behind it is
    bearer-gated.
    """
    with httpx.Client(timeout=30.0, transport=transport) as client:
        response = client.get(f"{web_url.rstrip('/')}/api/cli/config")
        if response.status_code != 200:
            msg = f"platform discovery failed with HTTP {response.status_code} at {web_url}"
            raise PlatformError(msg, status_code=response.status_code)
        api_url = response.json().get("apiUrl")
        return str(api_url).rstrip("/") if api_url else None


class PlatformClient:
    """Requests against the platform registry, authenticated with an org API key."""

    def __init__(
        self,
        api_url: str,
        token: str,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        try:
            version = metadata.version("world-model-harness")
        except metadata.PackageNotFoundError:
            version = "dev"
        self._client = httpx.Client(
            base_url=api_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": f"wmh/{version}",
            },
            timeout=_TIMEOUT_SECONDS,
            transport=transport,
        )
        # Bundle bytes move directly against storage's signed URLs; that
        # client carries no platform credential.
        self._transfer = httpx.Client(
            headers={"User-Agent": f"wmh/{version}"},
            timeout=_TIMEOUT_SECONDS,
            transport=transport,
        )

    def __enter__(self) -> PlatformClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()
        self._transfer.close()

    # -- identity ------------------------------------------------------------------------------

    def whoami(self) -> WhoAmI:
        response = self._client.get("/api/whoami")
        self._raise_for_error(response)
        return WhoAmI.model_validate(response.json())

    # -- unified runs --------------------------------------------------------------------------

    def resolve_run_target(self, target_id: str) -> RunTarget:
        """Resolve an opaque platform id without guessing from failed requests."""
        response = self._client.get(f"/api/run-targets/{target_id}")
        self._raise_for_error(response)
        return RunTarget.model_validate(response.json())

    def create_world_model_session(
        self, world_model_id: str, *, task: str | None = None
    ) -> RemoteWorldModelSession:
        """Open a hosted session for a platform world model."""
        response = self._client.post(
            f"/api/world-models/{world_model_id}/sessions", json={"task": task}
        )
        self._raise_for_error(response)
        return RemoteWorldModelSession.model_validate(response.json())

    def step_world_model_session(self, session_id: str, action: Action) -> Observation:
        """Advance a hosted world-model session by one action."""
        response = self._client.post(
            f"/api/sessions/{session_id}/step",
            json={"action": action.model_dump(mode="json")},
        )
        self._raise_for_error(response)
        return Observation.model_validate(response.json()["observation"])

    def create_agent_session(
        self,
        agent_id: str,
        *,
        workspace: bytes | None,
        instruction: str | None = None,
    ) -> RemoteAgentSession:
        """Create a hosted agent session, optionally staging a local snapshot."""
        payload: dict[str, JsonValue] = {"instruction": instruction}
        if workspace is not None:
            upload = self._client.post(
                f"/api/agents/{agent_id}/workspace-uploads",
                files={"workspace": ("workspace.tar.gz", workspace, "application/gzip")},
                timeout=_WORKSPACE_TIMEOUT_SECONDS,
            )
            self._raise_for_error(upload)
            payload["workspace_upload_id"] = str(upload.json()["id"])
        response = self._client.post(
            f"/api/agents/{agent_id}/sessions",
            json=payload,
        )
        self._raise_for_error(response)
        return RemoteAgentSession.model_validate(response.json())

    def get_agent_session(self, agent_id: str, session_id: str) -> RemoteAgentSession:
        """Read current hosted agent session state."""
        response = self._client.get(f"/api/agents/{agent_id}/sessions/{session_id}")
        self._raise_for_error(response)
        return RemoteAgentSession.model_validate(response.json())

    def resolve_agent_session(self, session_id: str) -> RemoteAgentSession:
        """Resolve a bare session id to its owning agent and current state."""
        response = self._client.get(f"/api/agent-sessions/{session_id}")
        self._raise_for_error(response)
        return RemoteAgentSession.model_validate(response.json())

    def end_agent_session(self, agent_id: str, session_id: str) -> RemoteAgentSession:
        """Request an end, reconciling directly when the hosted driver is gone."""
        response = self._client.post(f"/api/agents/{agent_id}/sessions/{session_id}/end")
        self._raise_for_error(response)
        return RemoteAgentSession.model_validate(response.json())

    def list_agent_session_events(
        self, agent_id: str, session_id: str, *, after: int
    ) -> RemoteAgentEventPage:
        """Poll hosted transcript events after one durable sequence cursor."""
        response = self._client.get(
            f"/api/agents/{agent_id}/sessions/{session_id}/events",
            params={"after": after},
        )
        self._raise_for_error(response)
        return RemoteAgentEventPage.model_validate(response.json())

    def post_agent_session_command(
        self, agent_id: str, session_id: str, kind: str, *, text: str | None = None
    ) -> None:
        """Steer, interrupt, or end one hosted agent session."""
        response = self._client.post(
            f"/api/agents/{agent_id}/sessions/{session_id}/commands",
            json={"kind": kind, "text": text},
        )
        self._raise_for_error(response)

    def upload_agent_workspace_patch(
        self, agent_id: str, session_id: str, content: bytes
    ) -> WorkspacePatchResult:
        """Apply local changes conditionally to a running hosted workspace."""
        response = self._client.post(
            f"/api/agents/{agent_id}/sessions/{session_id}/workspace/patches",
            files={"patch": ("workspace-patch.tar.gz", content, "application/gzip")},
            timeout=_WORKSPACE_TIMEOUT_SECONDS,
        )
        self._raise_for_error(response)
        return WorkspacePatchResult.model_validate(response.json())

    def download_agent_workspace_patch(
        self, agent_id: str, session_id: str, revision: str
    ) -> bytes:
        """Download one remote-to-local live workspace patch."""
        response = self._client.get(
            f"/api/agents/{agent_id}/sessions/{session_id}/workspace/patches/{revision}",
            timeout=_WORKSPACE_TIMEOUT_SECONDS,
        )
        self._raise_for_error(response)
        return response.content

    def acknowledge_agent_workspace_patch(
        self, agent_id: str, session_id: str, revision: str
    ) -> None:
        """Remove a remote patch after it is safely reflected or reported locally."""
        response = self._client.post(
            f"/api/agents/{agent_id}/sessions/{session_id}/workspace/patches/{revision}/ack"
        )
        self._raise_for_error(response)

    def download_agent_workspace(self, agent_id: str, session_id: str) -> bytes:
        """Download a terminal hosted session's final E2B workspace snapshot."""
        response = self._client.get(
            f"/api/agents/{agent_id}/sessions/{session_id}/workspace",
            timeout=_WORKSPACE_TIMEOUT_SECONDS,
        )
        self._raise_for_error(response)
        return response.content

    def acknowledge_agent_workspace(self, agent_id: str, session_id: str) -> None:
        """Confirm the final archive is safe locally so platform objects can be removed."""
        response = self._client.post(f"/api/agents/{agent_id}/sessions/{session_id}/workspace/ack")
        self._raise_for_error(response)

    # -- world models --------------------------------------------------------------------------

    def get_world_model(self, world_model_id: str) -> RemoteWorldModel:
        """Read one world model by platform id."""
        response = self._client.get(f"/api/world-models/{world_model_id}")
        self._raise_for_error(response)
        return RemoteWorldModel.model_validate(response.json())

    def list_world_models(self, org_id: str) -> list[RemoteWorldModel]:
        response = self._client.get(f"/api/orgs/{org_id}/world-models")
        self._raise_for_error(response)
        rows = response.json().get("world_models", [])
        return [RemoteWorldModel.model_validate(row) for row in rows]

    def push_model_bundle(
        self,
        org_id: str,
        name: str,
        bundle_path: Path,
        sha256: str,
        byte_size: int,
        meta: dict[str, JsonValue],
    ) -> RemoteWorldModel:
        """Push a packed bundle file: ticket, direct PUT to storage, finalize.

        The bundle bytes stream from disk straight to the signed staging URL;
        only the finalize declaration (digest + size + serve metadata) goes
        through the API.
        """
        ticket_response = self._client.post(
            f"/api/orgs/{org_id}/world-models/{name}/bundle/uploads"
        )
        self._raise_for_error(ticket_response)
        ticket = ticket_response.json()
        upload_url = str(ticket["upload_url"])
        with bundle_path.open("rb") as fh:
            upload_response = self._transfer.put(
                upload_url,
                content=fh,
                headers={
                    "Content-Type": "application/gzip",
                    "x-upsert": "false",
                    "Authorization": f"Bearer {ticket.get('token', '')}",
                },
            )
        if not upload_response.is_success:
            msg = (
                f"bundle upload to storage failed with HTTP {upload_response.status_code}: "
                f"{upload_response.text[:200]}"
            )
            raise PlatformError(msg, status_code=upload_response.status_code)

        finalize = self._client.post(
            f"/api/orgs/{org_id}/world-models/{name}/bundle",
            json={
                "staging_path": ticket["staging_path"],
                "sha256": sha256,
                "byte_size": byte_size,
                "meta": meta,
            },
        )
        self._raise_for_error(finalize)
        return RemoteWorldModel.model_validate(finalize.json())

    def download_model_bundle(self, org_id: str, name: str, dest: Path) -> str:
        """Stream a model's bundle from storage to ``dest``, verifying its digest.

        The API hands back an expiring signed URL plus the recorded sha256;
        the bytes come straight from storage's CDN and are hashed as they
        stream to disk.

        Returns:
            The verified sha256 hex digest.
        """
        response = self._client.get(f"/api/orgs/{org_id}/world-models/{name}/bundle")
        self._raise_for_error(response)
        payload = response.json()
        declared = str(payload["sha256"])

        digest = hashlib.sha256()
        part_path = dest.with_name(f"{dest.name}.part")
        with self._transfer.stream("GET", str(payload["url"])) as stream:
            if not stream.is_success:
                msg = f"bundle download failed with HTTP {stream.status_code}"
                raise PlatformError(msg, status_code=stream.status_code)
            with part_path.open("wb") as fh:
                for chunk in stream.iter_bytes():
                    digest.update(chunk)
                    fh.write(chunk)
        actual = digest.hexdigest()
        if actual != declared:
            part_path.unlink(missing_ok=True)
            msg = f"bundle digest mismatch for {name}: expected {declared}, got {actual}"
            raise PlatformError(msg)
        part_path.replace(dest)
        return actual

    # -- harnesses -----------------------------------------------------------------------------

    def list_harnesses(self, org_id: str) -> list[RemoteHarness]:
        response = self._client.get(f"/api/orgs/{org_id}/harnesses")
        self._raise_for_error(response)
        rows = response.json().get("harnesses", [])
        return [RemoteHarness.model_validate(row) for row in rows]

    def get_harness(
        self, org_id: str, name: str
    ) -> tuple[RemoteHarness, list[RemoteHarnessVersion]]:
        response = self._client.get(f"/api/orgs/{org_id}/harnesses/{name}")
        self._raise_for_error(response)
        payload = response.json()
        harness = RemoteHarness.model_validate(payload["harness"])
        versions = [RemoteHarnessVersion.model_validate(row) for row in payload["versions"]]
        return harness, versions

    def get_harness_version(self, org_id: str, name: str, version: int) -> HarnessVersionDoc:
        response = self._client.get(f"/api/orgs/{org_id}/harnesses/{name}/versions/{version}")
        self._raise_for_error(response)
        return HarnessVersionDoc.model_validate(response.json())

    def push_harness_version(
        self,
        org_id: str,
        name: str,
        doc: dict[str, JsonValue],
        doc_hash: str,
    ) -> PushedHarnessVersion:
        response = self._client.post(
            f"/api/orgs/{org_id}/harnesses/{name}/versions",
            json={"doc": doc, "doc_hash": doc_hash},
        )
        self._raise_for_error(response)
        return PushedHarnessVersion.model_validate(response.json())

    # -- built-in local pi runs ---------------------------------------------------------------

    def create_local_pi_run(self, org_id: str) -> LocalPiRunInfo:
        """Open a metered platform run for WMH's built-in local pi harness."""
        response = self._client.post(f"/api/orgs/{org_id}/local-pi-runs")
        self._raise_for_error(response)
        return LocalPiRunInfo.model_validate(response.json())

    def complete_local_pi_worker(
        self, org_id: str, run_id: str, request: ChatRequest
    ) -> ChatResponse:
        """Answer one built-in pi worker turn through the platform."""
        response = self._client.post(
            f"/api/orgs/{org_id}/local-pi-runs/{run_id}/worker-completion",
            json=request.model_dump(mode="json", exclude_none=True),
        )
        self._raise_for_error(response)
        return ChatResponse.model_validate(response.json())

    def finish_local_pi_run(
        self,
        org_id: str,
        run_id: str,
        *,
        status: str,
        ended_reason: str,
        error: str | None = None,
    ) -> None:
        """Report the terminal transition of a built-in local pi run."""
        response = self._client.post(
            f"/api/orgs/{org_id}/local-pi-runs/{run_id}/finish",
            json={"status": status, "ended_reason": ended_reason, "error": error},
        )
        self._raise_for_error(response)

    # -- internals -----------------------------------------------------------------------------

    def _raise_for_error(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        try:
            message = response.json().get("error", response.text)
        except (json.JSONDecodeError, ValueError):
            message = response.text or f"HTTP {response.status_code}"
        if response.status_code == 401:
            message = f"{message} — run `wmh login` (or check WMH_PLATFORM_TOKEN)"
        raise PlatformError(str(message), status_code=response.status_code)
