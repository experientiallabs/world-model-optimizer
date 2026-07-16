"""D90.2 audit probe: reproduce the swe substrate run's first policy turn, raw.

477/600 training episodes ended at steps=2 with zero parsed tool calls (stop=done: the
scaffold treats a reply with no parseable tool call as a final answer). Telemetry cannot
distinguish capability (the model wrote prose) from format (the model emitted tool syntax
that both text parsers missed) — rule 12 says read the actual completions.

This replays the EXACT first turn of the training episodes: same system prompt template
({task}/{done_tool}), same "Begin." opener, same tools payload, same temp/max_tokens,
against a raw-text vLLM (no server-side tool parsing — matching the training rl_server).
Each completion is then run through the scaffold's own fallback chain
(THINK_BLOCK_RE -> parse_tool_calls -> parse_qwen_xml_tool_calls) and dumped verbatim
for human reading, with the parse verdict attached.

Run on box-6 from /mnt/b2/claas-verl-wm-tau with the policy served raw on :8004.
"""

import asyncio
import json
import sys
from pathlib import Path

import httpx
from transformers import AutoTokenizer

from claas.benchmarks.common import THINK_BLOCK_RE, parse_tool_calls
from claas.benchmarks.wm_tau.scaffold import parse_qwen_xml_tool_calls
from claas.benchmarks.wm_tau.scenarios import DONE_TOOL, build_tool_schemas, load_scenarios, load_tools

WMH = Path("/mnt/b2/world-model-harness/packages/environment-capture/swe-bench/rl")
OUT = Path("/mnt/b2/output/swe_notool_probe.jsonl")
BASE = "/mnt/b2/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
N_SCEN, N_SAMPLES = 12, 2


async def main() -> None:
    scenarios = load_scenarios(str(WMH / "scenarios_train_v2.jsonl"))[:N_SCEN]
    tools = build_tool_schemas(load_tools(str(WMH / "tools.json")))
    system_prompt = (WMH / "system_prompt.txt").read_text()
    # Render the chat template ourselves (tools included), exactly like the training
    # rl_server, and complete raw text — the audit needs the unparsed emission.
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    rows = []
    async with httpx.AsyncClient(timeout=600.0) as client:
        for s in scenarios:
            messages = [
                {"role": "system", "content": system_prompt.format(task=s.task, done_tool=DONE_TOOL)},
                {"role": "user", "content": "Begin."},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tools=tools, add_generation_prompt=True, tokenize=False
            )
            for i in range(N_SAMPLES):
                resp = await client.post(
                    "http://127.0.0.1:8004/v1/completions",
                    json={
                        "model": "policy",
                        "prompt": prompt,
                        "temperature": 1.0,
                        "max_tokens": 6000,
                    },
                )
                resp.raise_for_status()
                text = resp.json()["choices"][0]["text"] or ""
                structured = []
                clean = THINK_BLOCK_RE.sub("", text).strip()
                parsed_json = parse_tool_calls(clean, name_key="name", args_key="arguments")
                parsed_xml = [] if parsed_json else parse_qwen_xml_tool_calls(clean)
                verdict = (
                    "structured" if structured
                    else "parsed_json" if parsed_json
                    else "parsed_xml" if parsed_xml
                    else "NO_TOOL"
                )
                rows.append({
                    "scenario_id": s.scenario_id,
                    "sample": i,
                    "verdict": verdict,
                    "text": text,
                })
                print(f"{s.scenario_id[:8]} s{i}: {verdict} ({len(text)} chars)", flush=True)
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print("VERDICTS:", counts)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
