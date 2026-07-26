"""The ICL arm of BENCH-B: in-context learning against the tau-bench world model.

The no-gradient control every trained arm (SFT/PPO/REINFORCE++/GRPO/SDPO) is compared against.
A policy LLM proposes tau tool calls; the world model answers; the episode-end reward judge
scores the rollout and its `critique` is what gets learned — injected back into the policy's
context instead of into its weights.

Modes (`--mode`):
- `base`     no memory, one attempt per scenario. With the Qwen policy this IS the base-model row.
- `single`   k attempts per scenario; each retry sees the judge critiques of ITS OWN prior
             attempts (within-scenario self-correction). Reported row = final attempt.
- `collect`  run the TRAIN scenarios once each and write the cross-task memory JSONL
             (task outcome + critique per scenario) that `multi` consumes. Train scenarios
             only — collecting on eval scenarios would leak judge feedback about the eval
             tasks into the eval row, so the script refuses.
- `multi`    one attempt per scenario with the frozen cross-task memory from `collect`
             injected (the CLaaS paper's ICL arm shape).

Everything an arm consumes is PINNED (see pin_scenarios.py): the scenario sets AND the
per-domain tool inventory (tools.json) — nothing is re-derived from the corpus at run time.
Rows key on scenario provenance and are appended to the results file as each scenario
completes, so a mid-run failure keeps every finished row. Policy backends:
`bedrock:<model-id>` (dev stand-in; NOTE Bedrock drops sampling params, so --temperature has
no effect there) or `vllm:<model>@<base-url>` (Qwen3.5-9B on the wake/sleep server's vLLM,
which does honor --temperature).

Examples (from the repo root, script at packages/environment-capture/tau-bench/rl/icl.py):
    uv run python <script> --mode base --scenarios eval --limit 2
    uv run python <script> --mode collect --scenarios train --wm haiku
    uv run python <script> --mode multi --scenarios eval --wm gpt-5.5 \
        --policy vllm:Qwen/Qwen3.5-9B@http://localhost:8001/v1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from wmh.core.parsing import extract_json_object
from wmh.core.render import render_action
from wmh.core.types import Action, ActionKind, EnvState, JsonObject, Step
from wmh.engine.world_model import WorldModel
from wmh.env import DONE_SIGNAL, Scenario, WorldModelEnv, run_episode
from wmh.optimize.reward import EpisodeScore
from wmh.providers.base import Message, Provider, ProviderConfig, ProviderKind
from wmh.providers.registry import get_provider
from wmh.providers.waterfall import WaterfallProvider

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent / "models" / "tau-bench"
_SCENARIO_FILES = {"train": _HERE / "scenarios_train.jsonl", "eval": _HERE / "scenarios_eval.jsonl"}
_TOOLS_PATH = _HERE / "tools.json"

HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"  # dated profile id (undated is rejected)
OPUS = "us.anthropic.claude-opus-4-8"  # eval reward judge: third family vs both WM backends
REGION = "us-east-1"
# The judge fails over with the SAME Opus id in every link (identical judgement). Lessons from
# a saturated afternoon: (1) `us.` profile capacity (ServiceUnavailable) is shared across an
# account's regions, but per-region API quotas differ, so same-account region hops still help
# on bursts; (2) the endflow account has NO Opus access — AccessDenied is a non-capacity error
# that propagates immediately and kills the cascade, so only Opus-capable pools belong in the
# chain; (3) saturation is bursty — --retry-errors sweeps converge in 1-2 passes.
JUDGE_POOLS = ((None, "us-east-1"), (None, "us-west-2"), (None, "us-east-2"))
MAX_STEPS = 20  # shared eval protocol (DECISIONS.md D30)
POLICY_MAX_TOKENS = 6000  # room for Qwen3.5 reasoning + tool call (D30/D31)
OBS_CHARS = 800  # observation excerpt per history line in the policy prompt
MEMORY_RECORDS = 8  # most-recent cross-task records injected in `multi`

AGENT_SYSTEM = """You are a customer-service agent operating {domain} tools.
Work the task step by step: one tool call at a time, read the result, then decide the next call.

Available tools (name: argument keys):
{tools}

Reply with ONLY one JSON object per turn:
  {{"tool": "<name>", "arguments": {{...}}}}         to call a tool
  {{"done": true, "summary": "<what you did>"}}      when the task is complete or impossible
{memory}"""

_REPARSE_NUDGE = (
    "Your previous reply was not one valid JSON object of the two allowed shapes. "
    "Reply again with ONLY the JSON."
)


class PinnedScenario(Scenario):
    """One line of a pinned scenario file: a `Scenario` plus its tau domain."""

    domain: str = "unknown"


class MemoryRecord(BaseModel):
    """What one finished episode teaches the next ones."""

    scenario_id: str  # first provenance trace_id
    domain: str
    success: bool
    reward: float
    critique: str


class RowResult(BaseModel):
    """One eval row entry: a scored episode on one scenario."""

    scenario_id: str
    domain: str
    attempt: int
    steps: int
    stop_reason: str
    reward: float
    success: bool
    critique: str
    parse_failures: int = 0  # policy replies that were not valid action JSON
    error: str | None = None  # infra failure (env error / judge failure); row is NOT a score
    wm_cost_usd: float | None = None
    seconds: float = 0.0


class _ToolCall(BaseModel):
    tool: str
    arguments: JsonObject = Field(default_factory=dict)


class _Done(BaseModel):
    done: bool
    summary: str = ""


class PolicyAgent:
    """wmh `Agent`: an LLM proposing tau tool calls from task + episode history (+ ICL memory).

    A malformed policy reply gets ONE re-ask with a terse nudge; a second failure ends the
    episode (never feed garbage to the world model). `parse_failures` counts both, so rows
    where the parser — not the policy's decisions — shaped the outcome are identifiable.
    """

    def __init__(
        self,
        provider: Provider,
        tools_block: str,
        domain: str,
        memory_block: str = "",
        temperature: float = 0.7,
    ) -> None:
        self._provider = provider
        self._system = AGENT_SYSTEM.format(
            domain=domain,
            tools=tools_block,
            memory=f"\n{memory_block}" if memory_block else "",
        )
        self._temperature = temperature
        self.parse_failures = 0

    def act(self, task: str | None, state: EnvState, history: list[Step]) -> Action:
        user = self._render_turn(task, history)
        action = self._propose(user)
        if action is None:
            self.parse_failures += 1
            action = self._propose(f"{user}\n\n{_REPARSE_NUDGE}")
        if action is None:
            self.parse_failures += 1
            return Action(kind=ActionKind.MESSAGE, content=DONE_SIGNAL)
        return action

    def _render_turn(self, task: str | None, history: list[Step]) -> str:
        lines = [f"TASK:\n{task or '(none)'}", "", "EPISODE SO FAR:"]
        if not history:
            lines.append("(no steps yet)")
        for i, step in enumerate(history, start=1):
            observation = step.observation.content[:OBS_CHARS]
            flag = " [ERROR]" if step.observation.is_error else ""
            lines.append(f"{i}. {render_action(step.action)}\n   -> {flag}{observation}")
        lines.append("\nYour next JSON:")
        return "\n".join(lines)

    def _propose(self, user: str) -> Action | None:
        completion = self._provider.complete(
            self._system,
            [Message(role="user", content=user)],
            temperature=self._temperature,
            max_tokens=POLICY_MAX_TOKENS,
        )
        return _parse_action(completion.text)


_QWEN_FUNCTION_RE = re.compile(r"<function=([\w.-]+)>(.*?)</function>", re.DOTALL)
_QWEN_PARAMETER_RE = re.compile(r"<parameter=([\w.-]+)>(.*?)</parameter>", re.DOTALL)


def _parse_action(text: str) -> Action | None:
    """One of the allowed reply shapes, or None if the reply matches none.

    Accepts the prompted JSON shapes first, then falls back to Qwen3.5's native XML tool-call
    format (`<function=...><parameter=...>` — see DECISIONS.md D31): the wake/sleep server
    applies no tool parser, and the ICL rows must measure the policy, not the parser.
    """
    raw = extract_json_object(text)
    if raw is not None:
        try:
            call = _ToolCall.model_validate_json(raw)
            return Action(kind=ActionKind.TOOL_CALL, name=call.tool, arguments=call.arguments)
        except ValidationError:
            pass
        try:
            done = _Done.model_validate_json(raw)
        except ValidationError:
            done = None
        if done is not None:
            return Action(kind=ActionKind.MESSAGE, content=DONE_SIGNAL) if done.done else None
    return _parse_qwen_xml(text)


def _parse_qwen_xml(text: str) -> Action | None:
    """Qwen3.5's XML tool-call format. Parameter values parse as JSON where possible."""
    match = _QWEN_FUNCTION_RE.search(text)
    if match is None:
        return None
    name, body = match.group(1), match.group(2)
    arguments: JsonObject = {}
    for key, value in _QWEN_PARAMETER_RE.findall(body):
        value = value.strip()
        try:
            arguments[key] = json.loads(value)
        except ValueError:
            arguments[key] = value
    if name.lower() == "done":
        return Action(kind=ActionKind.MESSAGE, content=DONE_SIGNAL)
    return Action(kind=ActionKind.TOOL_CALL, name=name, arguments=arguments)


def _load_tools() -> dict[str, str]:
    """The pinned per-domain tool inventory, rendered as prompt blocks."""
    inventory: dict[str, dict[str, list[str]]] = json.loads(_TOOLS_PATH.read_text(encoding="utf-8"))
    return {
        domain: "\n".join(
            f"- {name}: {', '.join(args) or '(no args)'}" for name, args in by_name.items()
        )
        for domain, by_name in inventory.items()
    }


def _memory_block(records: list[MemoryRecord], domain: str) -> str:
    """Cross-task learnings: same-domain first, most recent first, capped at MEMORY_RECORDS."""
    same = [r for r in reversed(records) if r.domain == domain]
    other = [r for r in reversed(records) if r.domain != domain]
    picked = (same + other)[:MEMORY_RECORDS]
    if not picked:
        return ""
    lines = ["== Learnings from prior tasks (use them; do not repeat mistakes) =="]
    for r in picked:
        outcome = "SUCCEEDED" if r.success else "FAILED"
        lines.append(f"[{outcome} r={r.reward:.2f} {r.domain}] {r.critique}")
    return "\n".join(lines)


def _self_critique_block(prior: list[RowResult]) -> str:
    if not prior:
        return ""
    lines = ["== Feedback on YOUR previous attempts at THIS task =="]
    for r in prior:
        lines.append(f"[attempt {r.attempt}: reward={r.reward:.2f}] {r.critique}")
    return "\n".join(lines)


def _build_wm(kind: str) -> WorldModel:
    """The environment: haiku (training WM) or gpt-5.5 (pinned eval WM). Judge rides along."""
    if kind == "haiku":
        env_provider = get_provider(
            ProviderConfig(kind=ProviderKind.BEDROCK, model=HAIKU, region=REGION)
        )
        judge = env_provider  # cheap critiques while collecting memory / dev
    elif kind == "gpt-5.5":
        env_provider = get_provider(ProviderConfig(kind=ProviderKind.OPENAI, model="gpt-5.5"))
        judge = WaterfallProvider(
            [
                ProviderConfig(kind=ProviderKind.BEDROCK, model=OPUS, region=region)
                for _profile, region in JUDGE_POOLS
            ]
        )
    else:
        raise SystemExit(f"unknown --wm {kind!r}; use haiku or gpt-5.5")
    return WorldModel.load(str(_MODEL_DIR), env_provider, reward_provider=judge)


def _build_policy(spec: str) -> Provider:
    """--policy bedrock:<model-id> | vllm:<model>@<base-url> | openai:<model>."""
    scheme, _, rest = spec.partition(":")
    if scheme == "bedrock":
        return get_provider(ProviderConfig(kind=ProviderKind.BEDROCK, model=rest, region=REGION))
    if scheme == "vllm":
        model, _, url = rest.partition("@")
        if not url:
            raise SystemExit("vllm policy needs vllm:<model>@<base-url>")
        return get_provider(ProviderConfig(kind=ProviderKind.OPENAI, model=model, endpoint=url))
    if scheme == "openai":
        return get_provider(ProviderConfig(kind=ProviderKind.OPENAI, model=rest))
    raise SystemExit(f"unknown policy spec {spec!r}")


def _episode(
    wm: WorldModel,
    policy: Provider,
    scenario: PinnedScenario,
    tools: dict[str, str],
    memory_block: str,
    temperature: float,
    attempt: int,
) -> RowResult:
    """One scored episode. Infra failures come back as an `error` row, never an exception."""
    agent = PolicyAgent(
        policy,
        tools.get(scenario.domain, "(tools unknown)"),
        scenario.domain,
        memory_block=memory_block,
        temperature=temperature,
    )
    env = WorldModelEnv(wm, score_on_close=True)
    started = time.monotonic()
    result = run_episode(env, agent, task=scenario.task, max_steps=MAX_STEPS)
    usage = env.usage
    try:
        score: EpisodeScore = env.last_score
    except RuntimeError as exc:
        # Judge failed during the scoring close (throttle/network); the session was still
        # freed. Record an error row so the batch continues and the row is excluded upstream.
        score = EpisodeScore(reward=0.0, success=False, critique="")
        error = f"scoring failed: {exc.__cause__ or exc}"
    else:
        error = result.error and f"{result.stop_reason}: {result.error}"
    return RowResult(
        scenario_id=scenario.provenance[0],
        domain=scenario.domain,
        attempt=attempt,
        steps=len(result.steps),
        stop_reason=str(result.stop_reason),
        reward=score.reward,
        success=score.success,
        critique=score.critique,
        parse_failures=agent.parse_failures,
        error=error,
        wm_cost_usd=usage.total.cost_usd if usage else None,
        seconds=round(time.monotonic() - started, 1),
    )


def _summarize(rows: list[RowResult]) -> str:
    if not rows:
        return "no rows"
    scored = [r for r in rows if r.error is None]
    errored = len(rows) - len(scored)
    if not scored:
        return f"no scored rows ({errored} infra errors)"
    by_domain: dict[str, list[RowResult]] = defaultdict(list)
    for r in scored:
        by_domain[r.domain].append(r)
    success = sum(1 for r in scored if r.success) / len(scored)
    reward = sum(r.reward for r in scored) / len(scored)
    cost = sum(r.wm_cost_usd or 0.0 for r in rows)
    parse_failures = sum(r.parse_failures for r in rows)
    parts = [
        f"n={len(scored)} success={success:.2%} mean_reward={reward:.3f} wm_cost=${cost:.2f}"
        f" infra_errors={errored} parse_failures={parse_failures}"
    ]
    for domain, group in sorted(by_domain.items()):
        ds = sum(1 for r in group if r.success) / len(group)
        parts.append(f"  {domain}: n={len(group)} success={ds:.2%}")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["base", "single", "collect", "multi"], required=True)
    parser.add_argument("--scenarios", choices=["train", "eval"], required=True)
    parser.add_argument("--wm", default="haiku", help="haiku (training WM) | gpt-5.5 (eval WM)")
    parser.add_argument(
        "--policy", default=f"bedrock:{HAIKU}", help="bedrock:<id> | vllm:<m>@<url>"
    )
    parser.add_argument("--limit", type=int, default=0, help="cap scenarios (0 = all)")
    parser.add_argument("--attempts", type=int, default=2, help="attempts per scenario (single)")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,  # shared eval protocol (D30); vllm/openai only — bedrock drops it
        help="policy sampling (vllm/openai only)",
    )
    parser.add_argument(
        "--episodes-per-scenario",
        type=int,
        default=1,
        help="episodes per eval scenario (shared protocol D30 uses 2 for official rows)",
    )
    parser.add_argument("--memory", type=Path, default=_HERE / "icl_memory.jsonl")
    parser.add_argument("--out", type=Path, default=None, help="results JSONL (default: derived)")
    parser.add_argument("--wandb", action="store_true", help="log rows to wandb wmh-rl-transfer")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="re-run ONLY the scenarios whose rows in --out carry an infra error, keep good rows",
    )
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error("--attempts must be >= 1")
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    if args.mode == "collect" and args.scenarios != "train":
        parser.error(
            "--mode collect only runs on --scenarios train: memory collected from eval "
            "scenarios would leak judge feedback about the eval tasks into the multi row"
        )

    run = None
    if args.wandb:
        # Fail fast (before any WM/provider setup) if wandb isn't installed:
        # examples are gate-excluded, so this is an opt-in extra (uv run --with wandb ...).
        import wandb

        run = wandb.init(
            project="wmh-rl-transfer",
            name=f"icl-{args.mode}-{args.scenarios}-wm_{args.wm}",
            config={k: str(v) for k, v in vars(args).items()},
        )

    scenario_path = _SCENARIO_FILES[args.scenarios]
    scenarios = [
        PinnedScenario.model_validate_json(line)
        for line in scenario_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        scenarios = scenarios[: args.limit]
    if args.episodes_per_scenario < 1:
        parser.error("--episodes-per-scenario must be >= 1")
    scenarios = [s for s in scenarios for _ in range(args.episodes_per_scenario)]

    tools = _load_tools()
    wm = _build_wm(args.wm)
    policy = _build_policy(args.policy)

    memory: list[MemoryRecord] = []
    if args.mode == "multi":
        if not args.memory.exists():
            raise SystemExit(f"--mode multi needs {args.memory} (run --mode collect first)")
        memory = [
            MemoryRecord.model_validate_json(line)
            for line in args.memory.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    out = args.out or _HERE / f"icl_{args.mode}_{args.scenarios}_wm-{args.wm}.results.jsonl"
    kept_rows: list[RowResult] = []
    write_target = out  # retry passes write to a tmp file and atomically replace at the end,
    # so an interrupted sweep never destroys the prior rows (fresh runs keep per-row appends)
    if args.retry_errors:
        # Failed rows only ever lost their JUDGE call (or env error) — the fix is to redo those
        # scenarios and splice, not re-pay for the rows that scored.
        if not out.exists():
            raise SystemExit(f"--retry-errors needs an existing results file at {out}")
        prior_rows = [
            RowResult.model_validate_json(line)
            for line in out.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        kept_rows = [r for r in prior_rows if r.error is None]
        retry_counts: dict[str, int] = {}
        for r in prior_rows:
            if r.error is not None:
                retry_counts[r.scenario_id] = retry_counts.get(r.scenario_id, 0) + 1
        retry_list: list[PinnedScenario] = []
        for scenario in scenarios:
            key = scenario.provenance[0]
            if retry_counts.get(key, 0) > 0:
                retry_counts[key] -= 1
                retry_list.append(scenario)
        scenarios = retry_list
        unmatched = {k: v for k, v in retry_counts.items() if v > 0}
        if unmatched:
            raise SystemExit(
                "retry-errors could not match these errored rows against the current scenario "
                "selection (check --episodes-per-scenario/--limit match the original run): "
                f"{unmatched}"
            )
        print(f"retry-errors: {len(kept_rows)} rows kept, {len(scenarios)} scenarios to re-run")
    attempts = args.attempts if args.mode == "single" else 1
    if args.retry_errors:
        write_target = out.with_suffix(out.suffix + ".retry-tmp")
    rows: list[RowResult] = list(kept_rows)
    with write_target.open("w", encoding="utf-8") as sink:  # rows land as they finish
        for row in kept_rows:
            sink.write(row.model_dump_json() + "\n")
        sink.flush()
        for i, scenario in enumerate(scenarios, start=1):
            prior: list[RowResult] = []
            for attempt in range(1, attempts + 1):
                if args.mode == "multi":
                    block = _memory_block(memory, scenario.domain)
                elif args.mode == "single":
                    block = _self_critique_block(prior)
                else:
                    block = ""
                row = _episode(wm, policy, scenario, tools, block, args.temperature, attempt)
                prior.append(row)
                note = f" ERROR={row.error}" if row.error else ""
                print(
                    f"[{i}/{len(scenarios)} a{attempt}] {scenario.domain} "
                    f"reward={row.reward:.2f} success={row.success} steps={row.steps} "
                    f"({row.seconds}s){note}"
                )
            final = prior[-1]
            rows.append(final)
            sink.write(final.model_dump_json() + "\n")
            sink.flush()
            if run is not None:
                run.log(
                    {
                        "reward": final.reward,
                        "success": int(final.success),
                        "steps": final.steps,
                        "parse_failures": final.parse_failures,
                        "infra_error": int(final.error is not None),
                        "wm_cost_usd": final.wm_cost_usd or 0.0,
                        "scenario_index": i,
                    }
                )


    if args.retry_errors:
        os.replace(write_target, out)
    if args.mode == "collect":
        # Rebuild memory from EVERY scored row of the final results (kept + new): a retry pass
        # must never truncate memory to just the retried scenarios. Judge-parse failures are
        # excluded — an "Unparseable reward-judge reply" is not a learning.
        records = [
            MemoryRecord(
                scenario_id=r.scenario_id,
                domain=r.domain,
                success=r.success,
                reward=r.reward,
                critique=r.critique,
            )
            for r in rows
            if r.error is None and not r.critique.startswith("Unparseable reward-judge reply")
        ]
        args.memory.write_text(
            "\n".join(r.model_dump_json() for r in records) + "\n", encoding="utf-8"
        )
        print(f"memory -> {args.memory} ({len(records)} records)")
    print(f"\n== icl --mode {args.mode} --scenarios {args.scenarios} --wm {args.wm} ==")
    print(_summarize(rows))
    print(f"rows -> {out}")
    if run is not None:
        scored = [r for r in rows if r.error is None]
        if scored:
            run.summary["success_rate"] = sum(1 for r in scored if r.success) / len(scored)
            run.summary["mean_reward"] = sum(r.reward for r in scored) / len(scored)
        run.summary["infra_errors"] = len(rows) - len(scored)
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
