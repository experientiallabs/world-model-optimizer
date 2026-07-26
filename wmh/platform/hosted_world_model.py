"""The platform's world model as a closed-loop environment.

`wmh optimize` and closed-loop eval hold an `EnvironmentSource`, not a world model, so the
environment under test can live on the platform: every rollout opens a hosted session
(`POST /api/world-models/{id}/sessions`) and each agent action is answered by
`POST /api/sessions/{id}/step`. Prediction, retrieval, knowledge, and the serving provider all
stay host-side, which is what lets a logged-in user optimize against a world model they never
built or downloaded locally.
"""

from __future__ import annotations

import contextlib
from contextlib import AbstractContextManager

from wmh.core.types import Action, Observation
from wmh.harness.environment import AgentEnvironment
from wmh.platform.client import PlatformClient


class HostedWorldModelEnvironment:
    """One hosted world-model session, driven action by action over HTTP."""

    def __init__(self, client: PlatformClient, world_model_id: str, task: str) -> None:
        """Open a hosted session for `task` against the platform world model."""
        self._client = client
        self._session = client.create_world_model_session(world_model_id, task=task)

    @property
    def session_id(self) -> str:
        return self._session.id

    def execute(self, action: Action) -> Observation:
        return self._client.step_world_model_session(self._session.id, action)

    def close(self) -> None:
        """Hosted sessions are retired by the platform; nothing to release locally."""


class HostedWorldModelSource:
    """The `EnvironmentSource` backed by a platform world model.

    The client is shared across rollouts (httpx pools connections and is thread-safe), so
    concurrent (task, attempt) cells each get their own hosted session over one connection pool.
    Closing is the caller's job: one source outlives many evaluation waves.
    """

    def __init__(self, client: PlatformClient, world_model_id: str, *, label: str) -> None:
        """Serve environments from `world_model_id`; `label` names it in reports."""
        self._client = client
        self._world_model_id = world_model_id
        self._label = label

    @property
    def label(self) -> str:
        return self._label

    @property
    def world_model_id(self) -> str:
        return self._world_model_id

    def open(self, task: str) -> AgentEnvironment:
        return HostedWorldModelEnvironment(self._client, self._world_model_id, task)

    def frozen(self) -> AbstractContextManager[None]:
        """A no-op: the hosted model owns its own state, so there is nothing local to freeze."""
        return contextlib.nullcontext()
