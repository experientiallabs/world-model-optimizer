"""Experiment A: Qwen-AgentWorld's own model fed the FULL interaction history.

DOCUMENTED DEVIATION from their shipped `eval.py infer` (@354f733), which sends only
`system_str` + `current_prompt`. Here the history turns 1..turn_idx-1 are injected as
alternating user/assistant chat turns (action -> observation) before the current action —
the same information their `build_judge_messages` shows the judge, in the model's native
chat form. Everything else matches their pipeline: their data rows, temperature 0,
max_tokens 32768, `gen` output field consumed by their judge stage unmodified.

Runs on the serving box against a local vLLM. Stdlib + openai only (no wmh imports).

Usage:
    python awb_infer_history.py --data-dir <shard_dir> \
        --base-url http://127.0.0.1:8899/v1 --model Qwen/Qwen-AgentWorld-35B-A3B \
        --output-dir <out_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from openai import OpenAI

MAX_TOKENS = 32768
RETRIES = 3


def build_messages(job: dict) -> list[dict]:
    prompts, responses = job.get("prompt", []), job.get("response", [])
    turn_idx = int(job.get("turn_idx", 1))
    n_history = min(turn_idx - 1, len(prompts), len(responses))
    messages = []
    if job.get("system_str"):
        messages.append({"role": "system", "content": job["system_str"]})
    for i in range(n_history):
        messages.append({"role": "user", "content": prompts[i]})
        messages.append({"role": "assistant", "content": responses[i]})
    # current_prompt fallback mirrors their run_inference exactly.
    current = job.get("current_prompt", "")
    if not current and turn_idx - 1 < len(prompts):
        current = prompts[turn_idx - 1]
    messages.append({"role": "user", "content": current})
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    jobs = []
    for f in sorted(Path(args.data_dir).glob("*_test.jsonl")):
        with f.open(encoding="utf-8") as fh:
            jobs.extend(json.loads(line) for line in fh if line.strip())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.jsonl"
    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    if r.get("gen"):
                        done.add((str(r["id"]), int(r["turn_idx"])))
    todo = [j for j in jobs if (str(j["id"]), int(j["turn_idx"])) not in done]
    print(f"{len(jobs)} jobs, {len(done)} done, {len(todo)} to run", flush=True)

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    with out_path.open("a", encoding="utf-8") as out:
        for i, job in enumerate(todo):
            gen = ""
            for attempt in range(RETRIES):
                try:
                    response = client.chat.completions.create(
                        model=args.model,
                        messages=build_messages(job),
                        max_tokens=MAX_TOKENS,
                        temperature=0,
                    )
                    gen = response.choices[0].message.content or ""
                    break
                except Exception as e:
                    print(f"[{i + 1}] attempt {attempt + 1} failed: {e}", file=sys.stderr, flush=True)
                    time.sleep(5 * (attempt + 1))
            job["gen"] = gen
            job["wmh_protocol"] = "full_history_chat_turns"
            out.write(json.dumps(job, ensure_ascii=False) + "\n")
            out.flush()
            if (i + 1) % 10 == 0 or (i + 1) == len(todo):
                print(f"[{i + 1}/{len(todo)}]", flush=True)


if __name__ == "__main__":
    main()
