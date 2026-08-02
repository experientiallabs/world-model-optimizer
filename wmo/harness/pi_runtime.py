"""`PiRuntime`: run the vendored pi agent (a real multi-file TypeScript harness) as an episode.

The harness under search is the pi agent's own source: each file is a `code:` surface carrying a
`path`. To run one task the runtime materializes those files into a checkout on a runner box,
starts a local shim, and drives pi headless through it (`wmo/harness/pi_entry/entry.ts`):

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

Concurrency note: the default binds an OS-assigned local shim port and derives a matching remote
runner directory for each episode. Callers may still pin `port` and `workdir` for a deliberate
single sequential lane.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from llm_waterfall import ChatRequest, ChatResponse
from pydantic import JsonValue

from wmo.core.types import Action, ActionKind, EnvState, JsonObject, Observation, Step
from wmo.harness.environment import AgentEnvironment
from wmo.harness.runner_link import (
    HostEpisode,
    params_schema,
    provider_context_window,
    stop_reason_for_done,
)
from wmo.harness.runtime import (
    DEFAULT_EVAL_EPISODE_TIMEOUT_S,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_TURNS,
    RunResult,
    StopReason,
    TokenUsage,
    validate_episode_timeout_s,
)
from wmo.harness.skills import SkillLibrary
from wmo.harness.tools import READ_SKILL, ToolSpec
from wmo.providers.base import (
    UNPARSED_TOOL_CALLS_KEY,
    Provider,
    ToolCallingProvider,
    structured_token_usage,
)

# The runner: node runs here, reached over SSH. The checkout keeps pi's node_modules; per-episode
# source is overwritten from the harness surfaces.
PI_RUNNER_HOST = os.environ.get("PI_RUNNER_HOST", "kion@nucbox.local")
PI_RUNNER_DIR = os.environ.get("PI_RUNNER_DIR", "~/pi-run")
DEFAULT_MAX_ENV_ACTIONS = 40
_PI_ENTRY_DIR = os.path.join(os.path.dirname(__file__), "pi_entry")
_ENTRY_TS = os.path.join(_PI_ENTRY_DIR, "entry.ts")
# entry.ts imports the shared classify + nudge policy, so it must be materialized alongside it.
_TERMINATION_TS = os.path.join(_PI_ENTRY_DIR, "runner_termination.ts")
# Cleanup headroom past the node wall budget: one SSH round trip plus process teardown.
_NODE_TEARDOWN_GRACE_S = 60.0
_NODE_HARD_KILL_AFTER_S = 10
# A newly provisioned control host often has no known_hosts entry for the operator-configured
# runner. `accept-new` permits that first connection but still rejects a changed key, unlike
# `StrictHostKeyChecking=no`. Keep the same policy on materialization and the tunneled node run.
_SSH_OPTIONS = (
    "-o",
    "ConnectTimeout=10",
    "-o",
    "BatchMode=yes",
    "-o",
    "StrictHostKeyChecking=accept-new",
)
# Runner paths are interpolated into remote shell commands, so restrict them to characters that
# cannot break out of the command (allows `~` expansion; rejects spaces, quotes, `;`, `$`, etc.).
_SAFE_REMOTE_PATH = re.compile(r"^[A-Za-z0-9_./~-]+$")


class _MaterializeError(RuntimeError):
    """Remote source materialization failed; the episode must not run stale files."""


class _Episode(HostEpisode):
    """Mutable per-run state the shim handlers share.

    Inherits the host-side episode contract (environment tool routing under a budget, transcript
    recording) from `HostEpisode` and adds what only the HTTP shim needs: the worker prompts and
    sampling policy, plus the completion signal entry.ts reads back.
    """

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
        context_window: int | None = None,
    ) -> None:
        super().__init__(
            instruction=instruction,
            tools=tools,
            environment=environment,
            skills=skills,
            max_env_actions=max_env_actions,
        )
        self.system_prompt = system_prompt
        self.provider = provider
        self.temperature = temperature
        self.max_turns = max_turns
        self.max_output_tokens = max_output_tokens
        self.context_window = context_window
        self.proxy_error: str = ""
        self.done_reason: str = ""
        # The host's view of the most recent worker completion, which entry.ts reads back through
        # GET /signal so its termination classifier sees the same evidence the frame runners get.
        self.finish_reason: str = ""
        self.unparsed_tool_calls: list[str] = []
        self.tool_call_turns: int = 0
        self.worker_usage = TokenUsage()
        self.done = threading.Event()

    def task_json(self) -> JsonObject:
        return {
            "instruction": self.instruction,
            "system": self.system_prompt,
            "max_turns": self.max_turns,
            "max_output_tokens": self.max_output_tokens,
            "context_window": self.context_window,
            "tools": self.tool_specs(),
        }

    def signal_json(self) -> JsonObject:
        """The host's view of the last worker completion, for entry.ts's classifier."""
        return {
            "finish_reason": self.finish_reason,
            "unparsed_tool_calls": list(self.unparsed_tool_calls),
            "provider_error": self.proxy_error,
            "tool_call_turns": self.tool_call_turns,
        }

    def worker_request(self, body: JsonObject) -> ChatRequest:
        """Apply the document sampling policy to one runner-authored structured request."""
        request_body = dict(body)
        request_body["temperature"] = self.temperature
        return ChatRequest.model_validate(request_body)

    def record_worker_completion(self, completion: ChatResponse, elapsed_s: float) -> None:
        """Meter one successful provider call at its request-level pricing boundary."""
        reported = structured_token_usage(completion)
        usage = self.worker_usage
        usage.calls += 1
        usage.input_tokens += reported.input_tokens
        usage.output_tokens += reported.output_tokens
        usage.cached_input_tokens += reported.cached_input_tokens
        usage.cache_write_input_tokens += reported.cache_write_input_tokens
        usage.reasoning_tokens += reported.reasoning_tokens
        usage.call_seconds.append(elapsed_s)
        usage.call_input_tokens.append(reported.input_tokens)
        usage.call_output_tokens.append(reported.output_tokens)
        usage.call_cached_input_tokens.append(reported.cached_input_tokens)
        usage.call_cache_write_input_tokens.append(reported.cache_write_input_tokens)

    def record_worker_failure(self, elapsed_s: float) -> None:
        """Record a failed provider attempt without inventing token counters."""
        self.worker_usage.calls += 1
        self.worker_usage.call_seconds.append(elapsed_s)


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
        path = self.path.rstrip("/")
        if path == "/task":
            self._send_json(self._ep.task_json())
        elif path == "/signal":
            self._send_json(self._ep.signal_json())
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
            body = self._read_body()
            answer = body.get("answer")
            reason = body.get("reason")
            self._ep.answer = answer if isinstance(answer, str) else ""
            self._ep.done_reason = reason if isinstance(reason, str) else ""
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
        started = time.perf_counter()
        usage_recorded = False
        try:
            completion = self._ep.provider.complete_chat(self._ep.worker_request(body))
            self._ep.record_worker_completion(completion, time.perf_counter() - started)
            usage_recorded = True
            choice = completion.choices[0]
            message = choice.message
            # Record the host's view of this turn BEFORE streaming it: entry.ts reads it back to
            # classify why the episode ended (truncated at the cap vs unparsed vs prose only).
            self._ep.proxy_error = ""
            self._ep.finish_reason = choice.finish_reason or "stop"
            unparsed = (choice.model_extra or {}).get(UNPARSED_TOOL_CALLS_KEY)
            self._ep.unparsed_tool_calls = (
                [str(item) for item in unparsed] if isinstance(unparsed, list) else []
            )
            if message.tool_calls:
                self._ep.tool_call_turns += 1
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
            if not usage_recorded:
                self._ep.record_worker_failure(time.perf_counter() - started)
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
        port: int = 0,
        workdir: str | None = None,
        max_env_actions: int = DEFAULT_MAX_ENV_ACTIONS,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        episode_timeout_s: float = DEFAULT_EVAL_EPISODE_TIMEOUT_S,
        context_window: int | None = None,
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
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65_535:
            raise ValueError("port must be an integer in [0, 65535]")
        self._port = port
        self._workdir = workdir
        self._max_env_actions = max_env_actions
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be >= 1")
        self._max_turns = max_turns
        self._max_output_tokens = max_output_tokens
        # The SSH path used to hardcode `timeout 300 node`, so every configured wall budget was
        # silently 300s and 30% of long TerminalBench-2 trials died on it.
        self._episode_timeout_s = validate_episode_timeout_s(episode_timeout_s)
        self._context_window = (
            context_window if context_window is not None else provider_context_window(provider)
        )
        paths = [("PI_RUNNER_DIR", PI_RUNNER_DIR)]
        if self._workdir is not None:
            paths.append(("workdir", self._workdir))
        for label, path in paths:
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
            context_window=self._context_window,
        )
        server = _ShimServer(("127.0.0.1", self._port), _ShimHandler)
        port = server.server_port
        workdir = self._workdir or f"{PI_RUNNER_DIR}/ep-{port}"
        server.episode = episode
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            try:
                self._materialize(workdir)
            except _MaterializeError as exc:
                # Remote write failed; do not run node against stale files from a prior episode.
                return self._error_result(task_id, episode, instruction, str(exc), StopReason.ERROR)
            code, note = self._run_node(port, workdir)
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
            # this is infrastructure failure, not an agent submission, so never count it as
            # SUBMITTED.
            return self._error_result(
                task_id,
                episode,
                instruction,
                f"worker LLM proxy error: {episode.proxy_error}",
                StopReason.PROVIDER_ERROR,
            )
        # entry.ts reports WHY it finished; only an explicit submit is a completion.
        stop_reason = stop_reason_for_done(episode.done_reason)
        return RunResult(
            task_id=task_id,
            steps=episode.steps,
            stop_reason=stop_reason,
            answer=episode.answer,
            turns=len(episode.steps),
            worker_usage=episode.worker_usage if episode.worker_usage.calls else None,
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
            worker_usage=episode.worker_usage if episode.worker_usage.calls else None,
        )

    def _materialize(self, workdir: str) -> None:
        """Write the harness's code surfaces + entry.ts into the runner checkout via SSH.

        The files stream as one JSON blob into a python materializer on the runner (one SSH round
        trip, no per-file scp), with node_modules symlinked from the persistent checkout.
        """
        blob = json.dumps(
            {
                "entry.ts": _read(_ENTRY_TS),
                "runner_termination.ts": _read(_TERMINATION_TS),
                **self._files,
            }
        )
        writer = (
            "import json,sys,os\n"
            "d=json.load(sys.stdin)\n"
            "for p,c in d.items():\n"
            "    os.makedirs(os.path.dirname(p) or '.',exist_ok=True)\n"
            "    open(p,'w').write(c)\n"
        )
        remote = (
            f"mkdir -p {workdir}"
            f" && ln -sfn {PI_RUNNER_DIR}/node_modules {workdir}/node_modules"
            f" && cd {workdir} && python3 -c {_shq(writer)}"
        )
        result = _ssh(remote, input_bytes=blob.encode("utf-8"))
        if result.returncode != 0:
            detail = (result.stderr or b"").decode("utf-8", "replace").strip()[-300:]
            raise _MaterializeError(f"remote materialize failed (rc={result.returncode}): {detail}")

    def _run_node(self, port: int, workdir: str) -> tuple[int, str]:
        """Run entry.ts on the runner with a reverse tunnel back to the local shim.

        The node wall budget is the configured episode timeout, not a fixed 300s: TerminalBench-2
        tasks compile toolchains and boot VMs, and the old constant killed 30% of them mid-turn.

        `timeout` takes whole seconds here, and the budget is rounded UP to one: truncating would
        silently shorten every fractional budget, and any budget under a second would truncate to
        `timeout 0`, which GNU coreutils reads as "no timeout at all" and would leave the node
        unbounded until the outer SSH subprocess deadline fires.
        """
        url = f"http://127.0.0.1:{port}"
        node_timeout_s = math.ceil(self._episode_timeout_s)
        remote_cmd = (
            f"cd {workdir} && PI_SHIM_URL={url} "
            f"timeout --kill-after={_NODE_HARD_KILL_AFTER_S} {node_timeout_s} "
            "node --experimental-strip-types entry.ts"
        )
        proc = subprocess.run(
            [
                "ssh",
                *_SSH_OPTIONS,
                "-R",
                f"{port}:127.0.0.1:{port}",
                PI_RUNNER_HOST,
                remote_cmd,
            ],
            capture_output=True,
            text=True,
            timeout=self._episode_timeout_s + _NODE_TEARDOWN_GRACE_S,
        )
        return proc.returncode, (proc.stderr or "").strip()[-500:]


def _ssh(remote_cmd: str, input_bytes: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["ssh", *_SSH_OPTIONS, PI_RUNNER_HOST, remote_cmd],
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
