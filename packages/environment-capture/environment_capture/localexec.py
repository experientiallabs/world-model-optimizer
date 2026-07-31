"""A real local environment: bash commands executed in a workspace directory.

Each command runs in a fresh subshell with the workspace as cwd (no state leaks between
commands beyond the filesystem itself — the same discipline the shared-sandbox benchmarks use),
with stdout and stderr folded into one observation string, which is what the agent would see in
a terminal.

The environment is DEFINED as workspace-scoped: by default a command that targets host locations
(absolute host roots, `~`, `$HOME`, `cd ..`) is refused WITHOUT executing, and the refusal text
is the observation. This is a containment guard, not a sandbox — a determined command can still
reach the host through an interpreter — so corpora are additionally audited with
`environment_capture.hygiene` before they are committed.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

from environment_capture.adapter import ExecResult
from environment_capture.hygiene import command_targets_host

_TIMEOUT_RETURNCODE = 124  # matches coreutils `timeout`
_BLOCKED_MESSAGE = (
    "blocked: this environment is scoped to the task workspace; use relative paths "
    "(the task's files are inside the current directory)"
)

# The capture process holds live provider credentials (AWS/Anthropic/HF keys and tokens). The
# workspace subprocess must never inherit them, so it is given an explicit allowlist of
# operational variables only — an agent that runs `env`, `printenv`, or `echo $VAR` finds no
# secret to read. This is the actual boundary; `command_targets_host` is defense-in-depth. The
# Python/venv/conda variables keep in-tree imports and non-system interpreters working (they are
# not credentials); provider secret variables are conventionally *_KEY/*_TOKEN/*_SECRET and are
# absent here by construction.
_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "PWD",
        "TMPDIR",
        "TERM",
        "TZ",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "LC_COLLATE",
        "LC_MESSAGES",
        "LC_NUMERIC",
        "LC_TIME",
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
    }
)


def _scrubbed_env() -> dict[str, str]:
    """The parent environment reduced to the non-secret operational allowlist."""
    return {key: value for key, value in os.environ.items() if key in _ENV_ALLOWLIST}


class LocalBashEnv:
    """CommandEnv backed by real subprocess execution in a workspace directory."""

    def __init__(
        self,
        workspace: Path | None = None,
        *,
        timeout_s: int = 60,
        cleanup: bool = False,
        contain: bool = True,
    ) -> None:
        """Execute in `workspace` (a fresh temp dir if None, then cleaned up on close)."""
        self._own_workspace = workspace is None
        self.workspace = workspace or Path(tempfile.mkdtemp(prefix="envcap-"))
        self.timeout_s = timeout_s
        self._cleanup = cleanup or self._own_workspace
        self.contain = contain

    def execute(self, command: str) -> ExecResult:
        if self.contain and command_targets_host(command):
            return ExecResult(output=_BLOCKED_MESSAGE, returncode=1)
        try:
            kwargs = {}
            if os.name == "posix":
                kwargs["start_new_session"] = True
                
            process = subprocess.Popen(
                ["bash", "-c", command],
                cwd=self.workspace,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",  # binary output becomes a real observation, not a crash
                env=_scrubbed_env(),  # never expose the capture process's provider credentials
                **kwargs
            )
            stdout, stderr = process.communicate(timeout=self.timeout_s)
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True)
            try:
                process.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            return ExecResult(
                output=f"command timed out after {self.timeout_s}s",
                returncode=_TIMEOUT_RETURNCODE,
            )
        output = stdout
        if stderr:
            output = f"{output}{stderr}" if output else stderr
        return ExecResult(output=output.rstrip("\n"), returncode=returncode)

    def close(self) -> None:
        if self._cleanup:
            shutil.rmtree(self.workspace, ignore_errors=True)
