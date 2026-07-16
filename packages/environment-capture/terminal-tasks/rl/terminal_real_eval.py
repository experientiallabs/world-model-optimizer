#!/usr/bin/env python3
"""Evaluate a policy checkpoint on the REAL terminal-tasks gym over the pinned eval scenarios.

This is the real-environment counterpart of the D30 WM eval (BENCH-B, D67): the SAME 28 pinned
held-out scenarios, but each episode runs against a REAL Debian bash shell in a throwaway docker
container instead of the world model. The policy is driven over an OpenAI-compatible
chat-completions endpoint (a vLLM serve of a spliced checkpoint) using structured tool calls, and
the finished transcript is scored by wmh's `EpisodeRewardJudge` — the SAME judge machinery that
scores WM-side rows, so the two are directly comparable. Records are keyed by each scenario's source
trace id (``provenance[0]``) so they pair 1:1 with the WM-eval rows.

The terminal counterpart of the tau-bench rl real-env harness. Unlike the tau harness (which
shells out to the ``tau2`` CLI and imports nothing but stdlib), this one imports wmh for the judge
and drives the policy loop itself over httpx; docker is the only external process it shells out to.

Usage (a box with docker + a vLLM serving the checkpoint on :8004, Bedrock creds for the judge):
    python real_eval.py \
        --agent-model rpp_n4_0192 \
        --agent-api-base http://127.0.0.1:8004/v1 \
        --out /data/output/real_terminal_rpp_n4_0192.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from pydantic import BaseModel, Field, JsonValue, ValidationError

from wmh.core.types import Action, ActionKind, JsonObject, Observation, Step
from wmh.optimize.reward import EpisodeRewardJudge, EpisodeScore
from wmh.providers import get_provider
from wmh.providers.base import Provider, ProviderConfig, ProviderKind
from wmh.providers.retry import RetryingProvider

_HERE = Path(__file__).resolve().parent
DEFAULT_SCENARIOS = _HERE / "scenarios_eval.jsonl"
DEFAULT_TOOLS = _HERE / "tools.json"

# Judge PINNED to Bedrock Opus 4.8 (us-east-1) so every checkpoint row is scored by the same judge
# that scores the WM-side rows (D67 comparability).
JUDGE_MODEL = "us.anthropic.claude-opus-4-8"
JUDGE_REGION = "us-east-1"

IMAGE = "debian:bookworm-slim"
# Mirrors ../capture_terminal.py so the eval shell matches the captured one.
SETUP = (
    "apt-get update -qq && "
    "apt-get install -y -qq curl python3 jq git ca-certificates >/dev/null 2>&1"
)

MAX_TOKENS = 3000
EXEC_TIMEOUT = 60  # per bash command, seconds
_HTTP_TIMEOUT = 180.0
_HTTP_ATTEMPTS = 3
_HTTP_BACKOFF = 2.0

# One nudge when a turn has no tool call, mirroring the tau scaffold: give the policy a chance to
# either act or cleanly stop, then end the episode on a second toolless turn.
NUDGE = "Use the bash tool, or stop if the task is complete."
# Per-command output cap fed back to the policy (and the judge): see the truncation
# note in the episode loop.
MAX_OUTPUT_CHARS = 6000

SYSTEM_PROMPT = """You are an expert operating a Debian bash shell to complete a task.

Work exclusively through the `bash` tool: every action you take must be a bash command issued via \
that tool. Available: curl, python3, jq, git, and standard coreutils; the network is up. Work step \
by step, inspecting each command's output before deciding the next one. When the task is fully \
complete, stop calling tools and reply with a short confirmation instead of another command.

TASK:
{task}"""


class Scenario(BaseModel):
    """One pinned eval scenario (a line of ``scenarios_eval.jsonl``)."""

    task: str
    provenance: list[str]
    category: str = ""
    rubric: JsonValue = None


class EpisodeRecord(BaseModel):
    """One (scenario, trial) outcome. Keyed by `scenario_id` to pair 1:1 with the WM-eval rows."""

    scenario_id: str  # the scenario's source trace id (provenance[0])
    rollout_index: int
    reward: float
    success: bool
    critique: str
    steps: int  # number of bash commands the policy executed
    errors: list[str]  # non-empty iff the episode crashed (excluded from the summary rates)
    env: str = "real-terminal"


class _ToolCallFunction(BaseModel):
    name: str = ""
    arguments: str = ""  # OpenAI sends tool-call arguments as a JSON *string*


class _ToolCall(BaseModel):
    id: str = ""
    function: _ToolCallFunction = Field(default_factory=_ToolCallFunction)


class _AssistantMessage(BaseModel):
    """The assistant message from one chat-completions response (OpenAI shape)."""

    content: str | None = None
    tool_calls: list[_ToolCall] = Field(default_factory=list)


class _Choice(BaseModel):
    message: _AssistantMessage


class _ChatResponse(BaseModel):
    choices: list[_Choice]


def load_scenarios(path: Path) -> list[Scenario]:
    """Parse the pinned scenarios JSONL into `Scenario`s (one per non-blank line)."""
    return [
        Scenario.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()
    ]


def load_tool_specs(path: Path) -> list[JsonObject]:
    """Build the chat-completions `tools` list from ``tools.json`` ({name: [required params]}).

    Deriving the arg shape from the pinned tools file keeps the eval-time tool identical to the
    tool the corpus was captured with, rather than hardcoding it in two places.
    """
    spec: dict[str, list[str]] = json.loads(path.read_text())
    tools: list[JsonObject] = []
    for name, params in spec.items():
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": "Run a bash command in the task shell.",
                    "parameters": {
                        "type": "object",
                        "properties": {p: {"type": "string"} for p in params},
                        "required": list(params),
                    },
                },
            }
        )
    return tools


def _chat(
    client: httpx.Client,
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: list[JsonObject],
    tools: list[JsonObject],
    temperature: float,
) -> _AssistantMessage:
    """One chat-completions call -> the assistant message. Retries transient HTTP/parse failures."""
    url = api_base.rstrip("/") + "/chat/completions"
    payload: JsonObject = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    last_exc: Exception | None = None
    for attempt in range(_HTTP_ATTEMPTS):
        try:
            resp = client.post(url, json=payload, headers=headers, timeout=_HTTP_TIMEOUT)
            resp.raise_for_status()
            return _ChatResponse.model_validate(resp.json()).choices[0].message
        except (httpx.HTTPError, ValidationError, json.JSONDecodeError, IndexError) as exc:
            last_exc = exc
            if attempt + 1 < _HTTP_ATTEMPTS:
                time.sleep(_HTTP_BACKOFF * (attempt + 1))
    raise RuntimeError(f"chat/completions failed after {_HTTP_ATTEMPTS} attempts: {last_exc}")


def _assistant_dict(
    message: _AssistantMessage, parsed_calls: list[tuple[str | None, str]]
) -> JsonObject:
    """Re-serialize an assistant message for appending back into the running messages list.

    Arguments are NORMALIZED to clean JSON of the (already-)parsed command rather than
    echoed verbatim: WM-trained checkpoints sometimes emit malformed argument JSON, and
    vLLM's chat template json-decodes replayed tool_calls — echoing the raw string 400s
    every subsequent turn of the episode (observed: 18/56 episodes on the first trained
    ckpt). ``parsed_calls`` aligns 1:1 with ``message.tool_calls``.
    """
    out: JsonObject = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        out["tool_calls"] = [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": json.dumps({"command": command}),
                },
            }
            for tc, (call_id, command) in zip(message.tool_calls, parsed_calls, strict=True)
        ]
    return out


def run_policy_loop(
    task: str,
    *,
    client: httpx.Client,
    api_base: str,
    api_key: str,
    model: str,
    tools: list[JsonObject],
    temperature: float,
    max_steps: int,
    execute: Callable[[str], tuple[str, int]],
) -> list[Step]:
    """Drive the policy against a bash shell until it stops, max_steps, or two toolless turns.

    Args:
        task: The instruction shown to the policy (also embedded in the system prompt).
        client: An httpx client pointed at the OpenAI-compatible endpoint (injected for testing).
        api_base: Endpoint base URL, e.g. ``http://127.0.0.1:8004/v1``.
        api_key: Bearer token for the endpoint (vLLM ignores it; sent for compatibility).
        model: Model name the endpoint serves.
        tools: The chat-completions `tools` list (see `load_tool_specs`).
        temperature: Sampling temperature for the policy.
        max_steps: Hard cap on assistant turns.
        execute: Runs one bash command, returning (combined stdout+stderr, exit code).

    Returns:
        The executed steps in order — one `Step` per bash command the policy actually ran. A
        toolless turn triggers a single nudge; a second consecutive toolless turn ends the episode.
    """
    messages: list[JsonObject] = [
        {"role": "system", "content": SYSTEM_PROMPT.format(task=task)},
        {"role": "user", "content": "Begin."},
    ]
    steps: list[Step] = []
    nudged = False
    for _ in range(max_steps):
        message = _chat(
            client,
            api_base=api_base,
            api_key=api_key,
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
        )
        parsed_calls = [
            (tc.id, _parse_command(tc.function.arguments)) for tc in message.tool_calls or []
        ]
        messages.append(_assistant_dict(message, parsed_calls))
        if not message.tool_calls:
            if nudged:
                break  # second toolless turn in a row -> the policy is done (or stuck)
            messages.append({"role": "user", "content": NUDGE})
            nudged = True
            continue
        nudged = False
        for call_id, command in parsed_calls:
            output, code = execute(command)
            # Cap per-command output: unbounded dumps (cat of large files, verbose
            # installs) overflow the policy's context and 400 the endpoint (observed:
            # 10/56 base-row episodes died on it). Head+tail keeps both the banner and
            # the part that usually matters (the end).
            if len(output) > MAX_OUTPUT_CHARS:
                keep = MAX_OUTPUT_CHARS // 2
                output = (
                    output[:keep]
                    + f"\n... [{len(output) - MAX_OUTPUT_CHARS} chars truncated] ...\n"
                    + output[-keep:]
                )
            content = f"[exit {code}] {output}" if code != 0 else output
            steps.append(
                Step(
                    action=Action(
                        kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": command}
                    ),
                    observation=Observation(content=content, is_error=code != 0),
                )
            )
            messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
    return steps


def _parse_command(arguments: str) -> str:
    """Extract the bash command from a tool call's JSON argument string (empty on garbage).

    WM-trained checkpoints sometimes append stray tokens after valid argument JSON;
    ``raw_decode`` salvages the leading object instead of dropping the whole call.
    """
    try:
        parsed, _end = json.JSONDecoder().raw_decode((arguments or "{}").strip())
    except json.JSONDecodeError:
        return ""
    if not isinstance(parsed, dict):
        return ""
    command = parsed.get("command", "")
    return command if isinstance(command, str) else ""


def _docker(args: list[str], *, timeout: int) -> tuple[str, int]:
    """Run a docker CLI command -> (combined output, returncode). Timeout maps to code 124."""
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, errors="replace"
        )
        return (proc.stdout + proc.stderr), proc.returncode
    except subprocess.TimeoutExpired:
        return "(timed out)", 124


PROVISIONED_IMAGE = "wmh-realterm-base:latest"


def _ensure_provisioned_image() -> None:
    """Build the tools image ONCE per host: base image + SETUP, committed locally.

    Per-episode ``apt-get install`` runs put 30-60s of mirror traffic on every episode's
    critical path (56 installs per row); provisioning once and committing removes it.
    The provisioning container is named uniquely and exec'd BY NAME — on a cold host the
    first ``docker run`` mixes pull progress into the combined output, so a parsed id
    is unreliable (observed as 'No such container' on fresh boxes).
    """
    _out, rc = _docker(["docker", "image", "inspect", PROVISIONED_IMAGE], timeout=60)
    if rc == 0:
        return
    name = f"wmh-realterm-provision-{uuid.uuid4().hex[:12]}"
    out, rc = _docker(["docker", "run", "-d", "--name", name, IMAGE, "sleep", "3600"], timeout=600)
    if rc != 0:
        raise RuntimeError(f"docker run (provision) failed: {out}")
    try:
        out, rc = _docker(["docker", "exec", name, "sh", "-c", SETUP], timeout=600)
        if rc != 0:
            raise RuntimeError(f"tool install failed: {out}")
        out, rc = _docker(["docker", "commit", name, PROVISIONED_IMAGE], timeout=120)
        if rc != 0:
            raise RuntimeError(f"docker commit failed: {out}")
    finally:
        _docker(["docker", "rm", "-f", name], timeout=60)


def _start_container() -> str:
    """Start a fresh pre-provisioned container; returns its NAME (stable exec handle).

    Caller must ``docker rm -f`` it.
    """
    name = f"wmh-realterm-{uuid.uuid4().hex[:12]}"
    out, rc = _docker(
        ["docker", "run", "-d", "--name", name, PROVISIONED_IMAGE, "sleep", "3600"], timeout=120
    )
    if rc != 0:
        raise RuntimeError(f"docker run failed: {out}")
    return name


def _exec_in(cid: str, command: str) -> tuple[str, int]:
    """Run one bash command in the container under a per-command timeout."""
    return _docker(
        ["docker", "exec", "-w", "/root", cid, "sh", "-c", command], timeout=EXEC_TIMEOUT
    )


def judge_provider() -> Provider:
    """Bedrock Opus 4.8 behind retries, pinned to one model (D67 comparability).

    RetryingProvider forwards the judge's explicit ``temperature=0.0`` (the repo's chain
    provider deliberately drops sampling params, which would silently change the judge);
    per-episode error records absorb anything retries can't ride out.
    """
    config = ProviderConfig(kind=ProviderKind.BEDROCK, model=JUDGE_MODEL, region=JUDGE_REGION)
    return RetryingProvider(get_provider(config))


def run_real_episode(
    task: str,
    *,
    client: httpx.Client,
    judge: EpisodeRewardJudge,
    model: str,
    api_base: str,
    api_key: str,
    temperature: float,
    max_steps: int,
    tools: list[JsonObject],
) -> tuple[EpisodeScore, list[Step]]:
    """Run one policy episode against a fresh container, then judge the transcript.

    The container is always removed, even if the policy loop or judge raises.
    """
    cid = _start_container()
    try:

        def execute(command: str) -> tuple[str, int]:
            return _exec_in(cid, command)

        steps = run_policy_loop(
            task,
            client=client,
            api_base=api_base,
            api_key=api_key,
            model=model,
            tools=tools,
            temperature=temperature,
            max_steps=max_steps,
            execute=execute,
        )
        return judge.score(task, steps), steps
    finally:
        _docker(["docker", "rm", "-f", cid], timeout=60)


def _record(
    scenario_id: str,
    rollout_index: int,
    *,
    score: EpisodeScore | None = None,
    n_steps: int = 0,
    error: Exception | None = None,
) -> EpisodeRecord:
    """One paired-analysis record; crash defaults apply when ``score`` is None."""
    return EpisodeRecord(
        scenario_id=scenario_id,
        rollout_index=rollout_index,
        reward=score.reward if score else 0.0,
        success=bool(score.success) if score else False,
        critique=score.critique if score else "",
        steps=n_steps,
        errors=[] if error is None else [f"{type(error).__name__}: {error}"],
        env="real-terminal",
    )


def evaluate(
    scenarios: list[Scenario],
    run_one: Callable[[str], tuple[EpisodeScore, list[Step]]],
    *,
    trials: int,
    concurrency: int,
) -> list[EpisodeRecord]:
    """Run every (scenario, trial) episode over a thread pool and collect paired-analysis records.

    Args:
        scenarios: The pinned eval scenarios.
        run_one: Runs one episode for a task, returning (score, steps) or raising on a crash.
        trials: Episodes per scenario (rollout_index 0..trials-1).
        concurrency: Max concurrent episodes.

    Returns:
        One record per (scenario, trial), sorted by (scenario_id, rollout_index). Crashed episodes
        become error records rather than aborting the run.
    """
    jobs = [(s.provenance[0], s.task, t) for s in scenarios for t in range(trials)]
    records: list[EpisodeRecord] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(run_one, task): (sid, ridx) for sid, task, ridx in jobs}
        for future in as_completed(futures):
            scenario_id, rollout_index = futures[future]
            try:
                score, steps = future.result()
                records.append(_record(scenario_id, rollout_index, score=score, n_steps=len(steps)))
            except Exception as exc:  # noqa: BLE001 - one crashed episode is a record, not a stop
                records.append(_record(scenario_id, rollout_index, error=exc))
    records.sort(key=lambda r: (r.scenario_id, r.rollout_index))
    return records


def _summary(records: list[EpisodeRecord], model: str) -> str:
    """The final one-line results banner; episodes with errors are excluded from the rates."""
    ok = [r for r in records if not r.errors]
    successes = sum(1 for r in ok if r.success)
    mean_reward = sum(r.reward for r in ok) / max(len(ok), 1)
    return (
        f"REAL TERMINAL RESULTS ({model}): {len(records)} records, "
        f"{len(records) - len(ok)} errored, "
        f"success {successes}/{len(ok)} ({successes / max(len(ok), 1) * 100:.1f}%), "
        f"mean reward {mean_reward:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default=str(DEFAULT_SCENARIOS))
    parser.add_argument("--agent-model", required=True, help="model name served by the endpoint")
    parser.add_argument("--agent-api-base", required=True, help="OpenAI-compatible base URL (/v1)")
    parser.add_argument("--agent-api-key", default="dummy")
    parser.add_argument("--agent-temperature", type=float, default=1.0)
    parser.add_argument("--trials", type=int, default=2, help="episodes per scenario")
    parser.add_argument("--concurrency", type=int, default=2, help="max concurrent episodes")
    parser.add_argument("--max-steps", type=int, default=20, help="max assistant turns per episode")
    parser.add_argument("--out", required=True, help="records jsonl for paired analysis")
    args = parser.parse_args()

    scenarios = load_scenarios(Path(args.scenarios))
    tools = load_tool_specs(DEFAULT_TOOLS)
    _ensure_provisioned_image()
    judge = EpisodeRewardJudge(judge_provider())
    client = httpx.Client()

    def run_one(task: str) -> tuple[EpisodeScore, list[Step]]:
        return run_real_episode(
            task,
            client=client,
            judge=judge,
            model=args.agent_model,
            api_base=args.agent_api_base,
            api_key=args.agent_api_key,
            temperature=args.agent_temperature,
            max_steps=args.max_steps,
            tools=tools,
        )

    print(
        f"[real_eval] {len(scenarios)} scenarios x {args.trials} trials "
        f"(concurrency {args.concurrency}) vs {args.agent_model}",
        flush=True,
    )
    try:
        records = evaluate(scenarios, run_one, trials=args.trials, concurrency=args.concurrency)
    finally:
        client.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")
    print(_summary(records, args.agent_model), flush=True)
    if records and all(r.errors for r in records):
        print("[real_eval] every episode errored", file=sys.stderr, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
