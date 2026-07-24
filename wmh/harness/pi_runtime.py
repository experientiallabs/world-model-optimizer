"""`PiRuntime`: run the vendored pi agent (a real multi-file TypeScript harness) as an episode.

The harness under search is the pi agent's own source: each file is a `code:` surface carrying a
`path`. To run one task the runtime materializes those files into a checkout on a runner box,
starts a local shim, and drives pi headless through it (`wmh/harness/pi_entry/entry.ts`):

- pi's LLM calls hit the shim's OpenAI-compatible `/v1/chat/completions`; the shim validates the
  structured request and delegates it to the caller's tool-calling provider. Provider-owned auth,
  routing, translation, retries, and waterfall failover stay on the control host.
- pi's task tools POST `/tool`, which the runtime answers from the `AgentEnvironment` (the world
  model in simulation, the real backend in the transfer check). These calls are the recorded
  transcript the judge grades.
- `submit` POSTs `/done`; the runtime returns a `RunResult` shaped exactly like the other runtimes.

The runner is remote (node lives on a separate box, never the control host), reached over SSH with
a reverse tunnel so the runner's node process can call back to the shim. The environment budget is
enforced kit-style: past the cap, `/tool` returns an error observation and the episode ends.

Concurrency note: episodes are serialized on one runner directory + port. Parallel rollouts must
pass distinct `port`/`workdir` (a per-episode caller responsibility); the default is a single
sequential lane, which is what the current search driver uses.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from llm_waterfall import ChatRequest
from pydantic import JsonValue

from wmh.core.types import Action, ActionKind, EnvState, JsonObject, Observation, Step
from wmh.harness.environment import AgentEnvironment
from wmh.harness.runner_link import params_schema, resolve_env_tool_call
from wmh.harness.runtime import DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_MAX_TURNS, RunResult, StopReason
from wmh.harness.skills import SkillLibrary
from wmh.harness.tools import READ_SKILL, ToolSpec
from wmh.providers.base import Provider, ToolCallingProvider

# The runner: node runs here, reached over SSH. The checkout keeps pi's node_modules; per-episode
# source is overwritten from the harness surfaces.
PI_RUNNER_HOST = os.environ.get("PI_RUNNER_HOST", "kion@nucbox.local")
PI_RUNNER_DIR = os.environ.get("PI_RUNNER_DIR", "~/pi-run")
DEFAULT_MAX_ENV_ACTIONS = 40
_ENTRY_TS = os.path.join(os.path.dirname(__file__), "pi_entry", "entry.ts")
# Runner paths are interpolated into remote shell commands, so restrict them to characters that
# cannot break out of the command (allows `~` expansion; rejects spaces, quotes, `;`, `$`, etc.).
_SAFE_REMOTE_PATH = re.compile(r"^[A-Za-z0-9_./~-]+$")


class _MaterializeError(RuntimeError):
    """Remote source materialization failed; the episode must not run stale files."""


class _Episode:
    """Mutable per-run state the shim handlers share."""

    def __init__(
        self,
        *,
        instruction: str,
        system_prompt: str,
        tools: list[ToolSpec],
        provider: ToolCallingProvider,
        environment: AgentEnvironment,
        temperature: float,
        skills: SkillLibrary,
        max_env_actions: int,
        max_turns: int,
        max_output_tokens: int,
    ) -> None:
        self.instruction = instruction
        self.system_prompt = system_prompt
        self.tools = tools
        self.provider = provider
        self.environment = environment
        self.temperature = temperature
        self.skills = skills
        self.max_env_actions = max_env_actions
        self.max_turns = max_turns
        self.max_output_tokens = max_output_tokens
        self.steps: list[Step] = []
        self.answer: str = ""
        self.proxy_error: str = ""
        self.done = threading.Event()
        self._env_calls = 0

    def task_json(self) -> JsonObject:
        return {
            "instruction": self.instruction,
            "system": self.system_prompt,
            "max_turns": self.max_turns,
            "max_output_tokens": self.max_output_tokens,
            "tools": [
                {"name": t.name, "description": t.description, "parameters": _params_schema(t)}
                for t in self.tools
            ],
        }

    def run_tool(self, name: str, arguments: JsonObject) -> JsonObject:
        action = Action(kind=ActionKind.TOOL_CALL, name=name, arguments=arguments)
        obs, consumed = resolve_env_tool_call(
            action,
            tools=self.tools,
            skills=self.skills,
            environment=self.environment,
            env_calls=self._env_calls,
            max_env_actions=self.max_env_actions,
        )
        if consumed:
            self._env_calls += 1
        self.steps.append(
            Step(action=action, observation=obs, state_before=EnvState(), task=self.instruction)
        )
        return {"content": obs.content, "is_error": obs.is_error}

    def worker_request(self, body: JsonObject) -> ChatRequest:
        """Apply the document sampling policy to one runner-authored structured request."""
        request_body = dict(body)
        request_body["temperature"] = self.temperature
        return ChatRequest.model_validate(request_body)


# The tool `parameters` schema builder lives in runner_link (shared with the frame transport);
# re-exported here under its old private name so existing callers and tests keep working.
_params_schema = params_schema


class _ShimServer(ThreadingHTTPServer):
    """A threading HTTP server that carries the current episode for its handlers."""

    # Environment calls mutate evaluator-owned state, so server_close() must join every active
    # handler: no environment write may land after the episode is declared finished (against a
    # real execution environment a late write would mutate state the evaluator is already
    # verifying; with the world model it was merely cosmetic). The join is bounded by whatever
    # the slowest handler is blocked on, worst case a completion handler waiting out the provider
    # SDK's timeout and retries (minutes during an outage), not just a tool command's budget. A
    # slow close is the accepted price of a trustworthy verdict.
    daemon_threads = False
    episode: _Episode


class _ShimHandler(BaseHTTPRequestHandler):
    # HTTP/1.1 so the OpenAI SDK's keep-alive works; the SSE handler forces a fresh socket per
    # turn (see _serve_completion) to avoid mis-framing the pipelined next request.
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - base API name
        return  # silence per-request stderr spam

    @property
    def _ep(self) -> _Episode:
        assert isinstance(self.server, _ShimServer)
        return self.server.episode

    def _read_body(self) -> JsonObject:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw or b"{}")

    def _send_json(self, obj: JsonObject, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.rstrip("/") == "/task":
            self._send_json(self._ep.task_json())
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = self.path.rstrip("/")
        if path == "/v1/chat/completions":
            self._serve_completion(self._read_body())
        elif path == "/tool":
            body = self._read_body()
            name = body.get("name")
            args = body.get("arguments")
            self._send_json(
                self._ep.run_tool(
                    name if isinstance(name, str) else "",
                    args if isinstance(args, dict) else {},
                )
            )
        elif path == "/done":
            answer = self._read_body().get("answer")
            self._ep.answer = answer if isinstance(answer, str) else ""
            self._send_json({})
            self._ep.done.set()
        else:
            self._send_json({"error": "not found"}, status=404)

    def _serve_completion(self, body: JsonObject) -> None:
        """Delegate pi's structured request to the provider and synthesize OpenAI SSE."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            completion = self._ep.provider.complete_chat(self._ep.worker_request(body))
            choice = completion.choices[0]
            message = choice.message
            content = message.content if isinstance(message.content, str) else ""
            delta: dict[str, JsonValue] = {"role": "assistant", "content": content}
            if message.tool_calls:
                delta["tool_calls"] = [
                    {
                        "index": i,
                        **tool_call.model_dump(mode="json"),
                    }
                    for i, tool_call in enumerate(message.tool_calls)
                ]
            first = {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
            last = {
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": choice.finish_reason or "stop"}
                ]
            }
            self.wfile.write(f"data: {json.dumps(first)}\n\n".encode())
            self.wfile.write(f"data: {json.dumps(last)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
        except Exception as exc:  # noqa: BLE001 - never crash the shim
            self._ep.proxy_error = str(exc)
            err = json.dumps({"error": {"message": f"agent provider failed: {exc}"}})
            self.wfile.write(f"data: {err}\n\ndata: [DONE]\n\n".encode())


class PiRuntime:
    """Runs one episode of the vendored pi harness against an `AgentEnvironment`."""

    def __init__(
        self,
        provider: Provider,
        *,
        files: dict[str, str],
        tools: list[ToolSpec],
        temperature: float = 0.7,
        skills: SkillLibrary | None = None,
        system_prompt: str = "",
        port: int = 8891,
        workdir: str | None = None,
        max_env_actions: int = DEFAULT_MAX_ENV_ACTIONS,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> None:
        if not isinstance(provider, ToolCallingProvider):
            raise TypeError("PiRuntime needs a ToolCallingProvider")
        self._provider = provider
        self._files = files
        self._skills = skills if skills is not None else SkillLibrary()
        self._tools = list(tools)
        if len(self._skills) and READ_SKILL.name not in {tool.name for tool in self._tools}:
            self._tools.append(READ_SKILL)
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        self._temperature = temperature
        self._system_prompt = system_prompt
        self._port = port
        self._workdir = workdir or f"{PI_RUNNER_DIR}/ep-{port}"
        self._max_env_actions = max_env_actions
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be >= 1")
        self._max_turns = max_turns
        self._max_output_tokens = max_output_tokens
        for label, path in (("PI_RUNNER_DIR", PI_RUNNER_DIR), ("workdir", self._workdir)):
            if not _SAFE_REMOTE_PATH.match(path):
                raise ValueError(
                    f"unsafe remote {label} {path!r}: only [A-Za-z0-9_./~-] allowed "
                    "(it is interpolated into a remote shell command)"
                )

    def run(self, task_id: str, instruction: str, environment: AgentEnvironment) -> RunResult:
        episode = _Episode(
            instruction=instruction,
            system_prompt=self._system_prompt,
            tools=self._tools,
            provider=self._provider,
            environment=environment,
            temperature=self._temperature,
            skills=self._skills,
            max_env_actions=self._max_env_actions,
            max_turns=self._max_turns,
            max_output_tokens=self._max_output_tokens,
        )
        server = _ShimServer(("127.0.0.1", self._port), _ShimHandler)
        server.episode = episode
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            try:
                self._materialize()
            except _MaterializeError as exc:
                # Remote write failed; do not run node against stale files from a prior episode.
                return self._error_result(task_id, episode, instruction, str(exc), StopReason.ERROR)
            code, note = self._run_node()
        finally:
            server.shutdown()
            server.server_close()
        if not episode.done.is_set():
            stop = StopReason.ERROR if code != 0 else StopReason.MAX_TURNS
            return self._error_result(
                task_id, episode, instruction, note or "episode ended without submit", stop
            )
        if episode.proxy_error:
            # The worker LLM proxy failed (auth/outage/HTTP error); entry.ts still POSTs /done, but
            # this is infrastructure failure, not an agent submission — never count it as SUBMITTED.
            return self._error_result(
                task_id,
                episode,
                instruction,
                f"worker LLM proxy error: {episode.proxy_error}",
                StopReason.ERROR,
            )
        return RunResult(
            task_id=task_id,
            steps=episode.steps,
            stop_reason=StopReason.SUBMITTED,
            answer=episode.answer,
            turns=len(episode.steps),
        )

    @staticmethod
    def _error_result(
        task_id: str, episode: _Episode, instruction: str, note: str, stop: StopReason
    ) -> RunResult:
        episode.steps.append(
            Step(
                action=Action(kind=ActionKind.MESSAGE, content="(pi runtime)"),
                observation=Observation(content=note, is_error=True),
                state_before=EnvState(),
                task=instruction,
            )
        )
        return RunResult(
            task_id=task_id,
            steps=episode.steps,
            stop_reason=stop,
            answer="",
            turns=len(episode.steps),
        )

    def _materialize(self) -> None:
        """Write the harness's code surfaces + entry.ts into the runner checkout via SSH.

        The files stream as one JSON blob into a python materializer on the runner (one SSH round
        trip, no per-file scp), with node_modules symlinked from the persistent checkout.
        """
        blob = json.dumps({"entry.ts": _read(_ENTRY_TS), **self._files})
        writer = (
            "import json,sys,os\n"
            "d=json.load(sys.stdin)\n"
            "for p,c in d.items():\n"
            "    os.makedirs(os.path.dirname(p) or '.',exist_ok=True)\n"
            "    open(p,'w').write(c)\n"
        )
        remote = (
            f"mkdir -p {self._workdir}"
            f" && ln -sfn {PI_RUNNER_DIR}/node_modules {self._workdir}/node_modules"
            f" && cd {self._workdir} && python3 -c {_shq(writer)}"
        )
        result = _ssh(remote, input_bytes=blob.encode("utf-8"))
        if result.returncode != 0:
            detail = (result.stderr or b"").decode("utf-8", "replace").strip()[-300:]
            raise _MaterializeError(f"remote materialize failed (rc={result.returncode}): {detail}")

    def _run_node(self) -> tuple[int, str]:
        """Run entry.ts on the runner with a reverse tunnel back to the local shim."""
        url = f"http://127.0.0.1:{self._port}"
        remote_cmd = (
            f"cd {self._workdir} && PI_SHIM_URL={url} "
            f"timeout 300 node --experimental-strip-types entry.ts"
        )
        proc = subprocess.run(
            [
                "ssh",
                "-o",
                "ConnectTimeout=10",
                "-o",
                "BatchMode=yes",
                "-R",
                f"{self._port}:127.0.0.1:{self._port}",
                PI_RUNNER_HOST,
                remote_cmd,
            ],
            capture_output=True,
            text=True,
            timeout=360,
        )
        return proc.returncode, (proc.stderr or "").strip()[-500:]


def _ssh(remote_cmd: str, input_bytes: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", PI_RUNNER_HOST, remote_cmd],
        input=input_bytes,
        capture_output=True,
        timeout=120,
    )


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _shq(text: str) -> str:
    """Single-quote a string for a remote shell (the python -c body)."""
    return "'" + text.replace("'", "'\\''") + "'"
