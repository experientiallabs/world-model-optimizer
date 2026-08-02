"""Offline tests for deterministic Harbor trace replay."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from coding_model_router_replay_agent import ReplayWmoTraceAgent
from harbor.models.agent.context import AgentContext


@dataclass
class _ExecResult:
    return_code: int


class _Environment:
    def __init__(self) -> None:
        self.commands: list[tuple[str, int | None]] = []
        self.uploads: dict[str, str] = {}

    async def exec(self, command: str, timeout_sec: int | None = None) -> _ExecResult:
        self.commands.append((command, timeout_sec))
        return _ExecResult(return_code=1 if command == "expected failure" else 0)

    async def upload_file(self, source_path: Path, target_path: str) -> None:
        self.uploads[target_path] = source_path.read_text(encoding="utf-8")


def test_replay_runs_recorded_mutations_in_order_and_tolerates_command_failure(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "wmo-run.json"
    trace_path.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "action": {
                            "kind": "tool_call",
                            "name": "write_file",
                            "arguments": {"path": "/app/fix.py", "content": "value = 1\n"},
                        }
                    },
                    {
                        "action": {
                            "kind": "tool_call",
                            "name": "bash",
                            "arguments": {"command": "expected failure"},
                        }
                    },
                    {
                        "action": {
                            "kind": "tool_call",
                            "name": "bash",
                            "arguments": {"command": "python3 -m py_compile /app/fix.py"},
                        }
                    },
                    {"action": {"kind": "message", "content": "done"}},
                ]
            }
        ),
        encoding="utf-8",
    )
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    environment = _Environment()
    agent = ReplayWmoTraceAgent(logs_dir=logs_dir, trace_path=str(trace_path))

    asyncio.run(
        agent.run(
            "repair the repository",
            environment,  # type: ignore[arg-type]
            AgentContext(),
        )
    )

    assert environment.uploads == {"/app/fix.py": "value = 1\n"}
    assert environment.commands == [
        ("expected failure", 240),
        ("python3 -m py_compile /app/fix.py", 240),
    ]
