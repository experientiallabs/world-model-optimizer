"""Tests for the local bash workspace environment."""

from __future__ import annotations

from pathlib import Path

from environment_capture.localexec import LocalBashEnv


def test_executes_in_workspace_and_captures_output(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.txt").write_text("hello capex")
    env = LocalBashEnv(workspace=tmp_path)
    try:
        result = env.execute("ls docs && grep -c capex docs/a.txt")
        assert result.returncode == 0
        assert "a.txt" in result.output
        assert "1" in result.output
    finally:
        env.close()


def test_nonzero_returncode_and_stderr_are_captured(tmp_path: Path) -> None:
    env = LocalBashEnv(workspace=tmp_path)
    try:
        result = env.execute("cat missing.txt")
        assert result.returncode != 0
        assert "missing.txt" in result.output  # stderr folded into the observation
    finally:
        env.close()


def test_timeout_returns_error_result(tmp_path: Path) -> None:
    env = LocalBashEnv(workspace=tmp_path, timeout_s=1)
    try:
        result = env.execute("sleep 5")
        assert result.returncode != 0
        assert "timed out" in result.output
    finally:
        env.close()


def test_timeout_kills_orphaned_child_processes(tmp_path: Path) -> None:
    """A command that times out must not leave orphaned child processes running in the background."""
    env = LocalBashEnv(workspace=tmp_path, timeout_s=1)
    script = tmp_path / "spin.py"
    script.write_text(
        "import time\n"
        "while True:\n"
        "    with open('ping.txt', 'a') as f: f.write('x\\n')\n"
        "    time.sleep(0.1)\n"
    )
    try:
        result = env.execute("python spin.py")
        assert result.returncode != 0
        assert "timed out" in result.output
        
        # Verify the background process is dead by checking if ping.txt stops growing.
        import time
        ping_file = tmp_path / "ping.txt"
        size_before = ping_file.stat().st_size if ping_file.exists() else 0
        time.sleep(0.4)
        size_after = ping_file.stat().st_size if ping_file.exists() else 0
        assert size_before == size_after, "Background process is still running and writing to file!"
    finally:
        env.close()


def test_state_does_not_leak_between_commands(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    env = LocalBashEnv(workspace=tmp_path)
    try:
        env.execute("export SECRET=42; cd sub >/dev/null")
        result = env.execute("pwd && echo ${SECRET:-unset}")
        assert str(tmp_path) in result.output  # fresh subshell, cwd reset
        assert "unset" in result.output
    finally:
        env.close()


def test_containment_guard_blocks_host_targeting_commands(tmp_path: Path) -> None:
    """The env is DEFINED as workspace-scoped: a host-targeting command is refused without
    executing, and the refusal is the real observation the agent (and the corpus) sees."""
    env = LocalBashEnv(workspace=tmp_path)
    try:
        for command in ("ls ~", "find / -name x", "cat /Users/someone/.ssh/config", "cd .."):
            result = env.execute(command)
            assert result.returncode != 0, command
            assert "workspace" in result.output, command
        # And nothing was actually executed against the host.
        ok = env.execute("echo safe && ls")
        assert ok.returncode == 0
    finally:
        env.close()


def test_containment_guard_can_be_disabled(tmp_path: Path) -> None:
    env = LocalBashEnv(workspace=tmp_path, contain=False)
    try:
        result = env.execute("ls /tmp >/dev/null; echo ran")
        assert "ran" in result.output
    finally:
        env.close()


def test_subprocess_environment_is_scrubbed_of_credentials(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """The workspace subprocess inherits an operational allowlist only — provider credentials in
    the capture process's environment are never readable via `env`/`printenv`/`echo $VAR`."""
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "sentinel-should-not-leak")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-leak")
    monkeypatch.setenv("HF_TOKEN", "hf_shouldnotleak")
    # contain=False isolates the env-scrub boundary from the command guard.
    env = LocalBashEnv(workspace=tmp_path, contain=False)
    try:
        dump = env.execute("env; printenv AWS_SECRET_ACCESS_KEY; echo ref:$ANTHROPIC_API_KEY")
        assert "should-not-leak" not in dump.output
        assert "sk-ant-should-not-leak" not in dump.output
        # Operational variables the shell/tools need survive the scrub.
        path = env.execute("echo $PATH")
        assert path.output.strip()
    finally:
        env.close()


def test_binary_output_is_replaced_not_fatal(tmp_path: Path) -> None:
    """An agent cat-ing a staged binary (sqlite db, csv with stray bytes) must yield a real
    observation with replacement chars, not a UnicodeDecodeError that burns the whole task."""
    (tmp_path / "blob.bin").write_bytes(b"\xca\xfe\xba\xbe binary")
    env = LocalBashEnv(workspace=tmp_path)
    try:
        result = env.execute("cat blob.bin")
        assert result.returncode == 0
        assert "binary" in result.output
    finally:
        env.close()
