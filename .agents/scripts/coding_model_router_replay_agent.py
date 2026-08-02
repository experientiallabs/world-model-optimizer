"""Harbor agent that deterministically replays one preserved WMO tool trace."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import override

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig

logger = logging.getLogger(__name__)


class ReplayWmoTraceAgent(BaseAgent):
    """Replay recorded bash and file-write actions without making model calls."""

    SUPPORTS_WINDOWS = False

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        logger: logging.Logger | None = None,
        mcp_servers: list[MCPServerConfig] | None = None,
        skills_dir: str | None = None,
        *args: object,
        trace_path: str,
        extra_env: dict[str, str] | None = None,
        **kwargs: object,
    ) -> None:
        del args, kwargs
        super().__init__(
            logs_dir=logs_dir,
            model_name=model_name,
            logger=logger,
            mcp_servers=mcp_servers,
            skills_dir=skills_dir,
            extra_env=extra_env,
        )
        self._trace_path = Path(trace_path)

    @staticmethod
    @override
    def name() -> str:
        return "wmo-trace-replay"

    @override
    def version(self) -> str:
        return "1.0.0"

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        del environment

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        del instruction, context
        trace = json.loads(self._trace_path.read_text(encoding="utf-8"))
        actions = [
            step.get("action")
            for step in trace.get("steps", [])
            if isinstance(step, dict) and isinstance(step.get("action"), dict)
        ]
        replayed = 0
        for index, action in enumerate(actions, start=1):
            if action.get("kind") != "tool_call":
                continue
            name = action.get("name")
            arguments = action.get("arguments")
            if not isinstance(arguments, dict):
                raise ValueError(f"trace action {index} has invalid arguments")
            if name == "bash":
                command = arguments.get("command")
                if not isinstance(command, str):
                    raise ValueError(f"trace bash action {index} has no command")
                result = await environment.exec(command=command, timeout_sec=240)
                logger.info(
                    "replayed bash action %d with exit %d",
                    index,
                    result.return_code,
                )
                replayed += 1
            elif name == "write_file":
                target = arguments.get("path")
                content = arguments.get("content")
                if not isinstance(target, str) or not isinstance(content, str):
                    raise ValueError(f"trace write action {index} is invalid")
                local_copy = self.logs_dir / f"replay-write-{index}"
                local_copy.write_text(content, encoding="utf-8")
                await environment.upload_file(
                    source_path=local_copy,
                    target_path=target,
                )
                logger.info("replayed write action %d to %s", index, target)
                replayed += 1
            elif name in {"read_file", "submit"}:
                logger.info("skipped non-mutating action %d (%s)", index, name)
            else:
                raise ValueError(f"unsupported trace tool {name!r} at action {index}")
        if replayed < 1:
            raise ValueError("trace contains no replayable actions")
