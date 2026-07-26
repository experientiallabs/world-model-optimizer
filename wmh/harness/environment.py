"""The environment seam: the agent loop talks to an interface, not to any backend directly.

The `AgentEnvironment` protocol is the substitution point: closed-loop eval binds it to the world
model (`wmh.evals.closed_loop.WorldModelEnvironment` — every tool call answered by
`WorldModel.step`), and a real execution backend (a managed sandbox) implements the same two
methods, so the *same* agent loop and scoring can run against reality when one is available. That
symmetry is what makes a simulated report comparable to a real one (`wmh.evals.agreement`).
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable

from wmh.core.types import Action, ActionKind, Observation

# The tool names the environment answers (everything except the runtime-handled `submit`).
ENV_TOOLS = frozenset({"bash", "read_file", "write_file"})


@runtime_checkable
class AgentEnvironment(Protocol):
    """Executes an agent Action and returns what the environment observed."""

    def execute(self, action: Action) -> Observation:
        """Run one action; return the resulting observation."""
        ...

    def close(self) -> None:
        """Release any underlying resources (end the session)."""
        ...


@runtime_checkable
class EnvironmentSource(Protocol):
    """Opens one `AgentEnvironment` per task and brackets a wave of rollouts.

    Closed-loop eval and the harness search hold this, not a world model: the environment under
    test may be a locally loaded world model (`wmh.evals.closed_loop.WorldModelSource`) or the
    platform's hosted one (`wmh.platform.hosted_world_model.HostedWorldModelSource`), and neither
    the scoring core nor the optimizer can tell the difference.
    """

    @property
    def label(self) -> str:
        """The environment's name, as reports and CLI output should identify it."""
        ...

    def open(self, task: str) -> AgentEnvironment:
        """Open a fresh environment for one rollout of `task`."""
        ...

    def frozen(self) -> AbstractContextManager[None]:
        """A window in which concurrent rollouts may share this source safely."""
        ...


def is_env_action(action: Action) -> bool:
    """True when the action is one the environment answers (a tool call to an env tool)."""
    return action.kind == ActionKind.TOOL_CALL and action.name in ENV_TOOLS
