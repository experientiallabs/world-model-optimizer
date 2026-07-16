"""Evaluate a Qwen student endpoint against the eval-pool scenarios in the frozen world model.

The student is any OpenAI-compatible chat endpoint (the box's pilot vLLM for student-before, the
CLaaS-served LoRA for student-after), reached via SSH tunnel. Episodes use the same Qwen
convention as the SFT data (<think> + <tool_call>/<tool_response>); the judge is Gemini Flash
grading each episode against the scenario checklist. Every scenario runs `--passes` times (house
rule: 3) and results break down per source domain (D35: read non-telecom columns for signal).

Usage (from the repo root, with e.g. `ssh -N -L 8000:localhost:8000 azureuser@4.154.170.26 &`):
    uv run python .agents/scripts/eval_student.py --endpoint http://localhost:8000/v1 \
        --model Qwen/Qwen3.5-9B --label student-before [--passes 3] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / ".agents" / "scripts"))

from collect_teacher import (  # noqa: E402
    _load_gemini_key,
    domain_tool_hint,
    foundry,
    gemini,
    openai_direct,
    opus_judge,
)
from run_scenario_e2e import NOVA_LITE, TRACES, WM_DIR, bedrock  # noqa: E402

from wmh.core.types import Action, ActionKind, EnvState, Step  # noqa: E402
from wmh.engine.world_model import WorldModel  # noqa: E402
from wmh.env.base import WorldModelEnv  # noqa: E402
from wmh.env.episode import DONE_SIGNAL, run_episode  # noqa: E402
from wmh.ingest import get_adapter  # noqa: E402
from wmh.scenarios import ChecklistJudge, ScenarioSet  # noqa: E402

DISTILL = REPO / ".agents" / "docs" / "research" / "distill"
JUDGE_MODEL = (
    "gemini-2.5-flash"  # default; --judge-model overrides (bedrock: prefix for Converse models)
)
MAX_STEPS = 10
WORKERS = 4

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def student_system(tool_hint: str) -> str:
    return (
        "You are a customer-service agent operating in a tool environment.\n"
        f"Available tools (from prior episodes in this domain): {tool_hint}\n"
        "Think before every action inside <think></think>, then either emit exactly one\n"
        '<tool_call>{"name": ..., "arguments": {...}}</tool_call>\n'
        "or reply to the user in plain text when the task is complete. Always check policy "
        "constraints before irreversible actions."
    )


class QwenStudentAgent:
    """Agent protocol adapter over an OpenAI-compatible chat endpoint, Qwen tool convention."""

    def __init__(self, client, model: str, tool_hint: str, temperature: float) -> None:  # noqa: ANN001
        self._client = client
        self._model = model
        self._tool_hint = tool_hint
        self._temperature = temperature
        self.final_text: str | None = None  # last plain-text reply; the judge must see it

    def act(self, task: str | None, state: EnvState, history: list[Step]) -> Action:
        messages = [{"role": "system", "content": student_system(self._tool_hint)}]
        user = task or ""
        if state.scratchpad and not history:
            user += f"\n\nEnvironment notes: {state.scratchpad}"
        messages.append({"role": "user", "content": user})
        for step in history:
            call = json.dumps(
                {"name": step.action.name, "arguments": step.action.arguments}, default=str
            )
            messages.append({"role": "assistant", "content": f"<tool_call>{call}</tool_call>"})
            messages.append(
                {
                    "role": "user",
                    "content": f"<tool_response>{step.observation.content}</tool_response>",
                }
            )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=10240,  # Qwen3.5 reasoning tokens count against the budget
            temperature=self._temperature,
        )
        text = response.choices[0].message.content or ""
        stripped = THINK_RE.sub("", text)
        match = TOOL_CALL_RE.search(stripped)
        if match is None:
            self.final_text = stripped.strip()
            return Action(kind=ActionKind.MESSAGE, content=DONE_SIGNAL)
        try:
            data = json.loads(match.group(1).strip())
            arguments = data.get("arguments") or {}
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            return Action(
                kind=ActionKind.TOOL_CALL, name=str(data.get("name", "")), arguments=arguments
            )
        except (json.JSONDecodeError, TypeError):
            return Action(kind=ActionKind.MESSAGE, content=stripped.strip()[:400])


def _resolve(spec: str):  # noqa: ANN202 - Provider; routes bedrock:/foundry:/openai:/gemini names
    if spec == "opus-judge":
        return opus_judge()
    if spec.startswith("bedrock:"):
        return bedrock(spec.removeprefix("bedrock:"))
    if spec.startswith("foundry:"):
        return foundry(spec.removeprefix("foundry:"))
    if spec.startswith("openai:"):
        return openai_direct(spec.removeprefix("openai:"))
    if spec.startswith("gemini"):
        return gemini(spec)
    return bedrock(spec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--label", required=True)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)  # BENCH-B2 eval protocol
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval-pool", default="eval_pool.json", help="pool file under distill/")
    parser.add_argument("--wm-model", default=NOVA_LITE, help="Bedrock model backing the WM")
    parser.add_argument(
        "--judge-model", default=JUDGE_MODEL, help="judge: gemini model name, or bedrock:<model-id>"
    )
    args = parser.parse_args()

    _load_gemini_key()
    from openai import OpenAI

    client = OpenAI(base_url=args.endpoint, api_key="not-needed")
    judge = ChecklistJudge(_resolve(args.judge_model))
    world_model = WorldModel.load(
        str(WM_DIR), _resolve(args.wm_model), telemetry_root=str(REPO / ".wmh")
    )

    pool = ScenarioSet.load(DISTILL / args.eval_pool)
    scenarios = pool.scenarios[: args.limit] if args.limit else pool.scenarios
    traces = get_adapter("otel-genai").from_file(str(TRACES))
    traces_by_id = {t.trace_id: t for t in traces}
    t0 = time.time()

    def run_cell(cell: tuple) -> dict:  # noqa: ANN001
        scenario, pass_index = cell
        tool_hint = domain_tool_hint(traces, scenario.provenance[0])
        agent = QwenStudentAgent(client, args.model, tool_hint, args.temperature)
        episode = run_episode(
            WorldModelEnv(world_model),
            agent,
            scenario.task,
            seed_state=scenario.seed_state,
            max_steps=MAX_STEPS,
        )
        judged_steps = list(episode.steps)
        if agent.final_text:
            # "inform/refuse/communicate" checklist items live in the final reply; without this
            # the judge auto-fails them (the bug behind the first 11% baseline run).
            from wmh.core.types import Observation

            judged_steps.append(
                Step(
                    action=Action(kind=ActionKind.MESSAGE, content=agent.final_text),
                    observation=Observation(content=""),
                )
            )
        verdict = judge.score(scenario.task, scenario.checklist, judged_steps)
        source = traces_by_id.get(scenario.provenance[0])
        domain = source.metadata.get("domain") if source else "unknown"
        return {
            "scenario_id": scenario.scenario_id,
            "pass_index": pass_index,
            "domain": domain,
            "success": verdict.success,
            "pass_rate": verdict.pass_rate,
            "steps": len(episode.steps),
            "stop_reason": episode.stop_reason.value,
            "error": episode.error,
        }

    cells = [(s, i) for s in scenarios for i in range(args.passes)]
    results: list[dict] = []
    with world_model.frozen(), ThreadPoolExecutor(max_workers=WORKERS) as pool_executor:
        for result in pool_executor.map(run_cell, cells):
            results.append(result)
            done = len(results)
            if done % 10 == 0:
                print(f"  {done}/{len(cells)} cells ({time.time() - t0:.0f}s)", flush=True)

    by_domain: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_domain[r["domain"]].append(r)
    summary = {
        "label": args.label,
        "model": args.model,
        "endpoint": args.endpoint,
        "passes": args.passes,
        "temperature": args.temperature,
        "scenarios": len(scenarios),
        "success_rate": sum(r["success"] for r in results) / len(results),
        "mean_pass_rate": sum(r["pass_rate"] for r in results) / len(results),
        "per_domain": {
            d: {
                "episodes": len(rs),
                "success_rate": sum(r["success"] for r in rs) / len(rs),
                "mean_pass_rate": sum(r["pass_rate"] for r in rs) / len(rs),
            }
            for d, rs in sorted(by_domain.items())
        },
        "wall_clock_seconds": round(time.time() - t0, 1),
        "results": results,
    }
    out = DISTILL / f"eval_{args.label}.json"
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(
        f"{args.label}: success {summary['success_rate']:.1%}, "
        f"pass-rate {summary['mean_pass_rate']:.3f} over {len(results)} episodes -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
