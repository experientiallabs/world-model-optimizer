"""AgentWorldBench `infer` stage backed by a wmh world model.

Replaces `eval.py infer` from github.com/QwenLM/Qwen-AgentWorld. Reads the benchmark's
per-domain `*_test.jsonl` rows, feeds each row's FULL interaction history to a wmh world
model, and writes `predictions.jsonl` with the `gen` field their `judge`/`score` stages
consume unchanged.

Protocol note (verified against their repo @354f733): their shipped `infer` sends only
`system_str` + `current_prompt` — NO prior turns — while `build_judge_messages` scores the
prediction against the full history context. We follow the paper protocol (the world model
receives the interaction history): turns `1..turn_idx-1` are seeded as teacher-forced
history, exactly what the judge sees.

Two modes:
- `wm`: a built wmh model (optimized prompt + RAG index), via the serving session path.
- `base`: BASE_ENV_PROMPT + no retrieval, any provider — for domains without a wmh corpus
  and for the RAG-vs-base ablation.

Built for long runs: rows are predicted on a thread pool (`--concurrency`), each row retries
with backoff on provider errors, completed rows are appended to the output immediately, and
`--resume` skips rows already predicted in an existing output file.

Usage (from the repo root):
    uv run python .agents/scripts/agentworldbench/awb_infer.py \
        --data .wmh/agentworldbench/data_full/terminal_test.jsonl \
        --mode wm --model-dir packages/environment-capture/terminal-tasks/models/terminal-tasks \
        --concurrency 3 --resume \
        --output .wmh/agentworldbench/results54/terminal_wm/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from wmh.core.types import Action, ActionKind, EnvState, JsonObject, Observation, Step
from wmh.engine.loader import load_world_model
from wmh.engine.prompts import BASE_ENV_PROMPT
from wmh.optimize.gepa import predict_observation, verify_observation
from wmh.providers import get_provider
from wmh.providers.base import Provider, ProviderConfig, ProviderKind
from wmh.tracking.metered import MeteredProvider
from wmh.tracking.tracker import Phase, RunTracker

RETRY_ATTEMPTS = 4


def history_steps(row: JsonObject) -> list[Step]:
    """Turns 1..turn_idx-1 as teacher-forced Steps (prompt/response are parallel lists)."""
    prompts, responses = row["prompt"], row["response"]
    n_history = min(int(row["turn_idx"]) - 1, len(prompts), len(responses))
    return [
        Step(
            action=Action(kind=ActionKind.MESSAGE, content=prompts[i]),
            observation=Observation(content=responses[i]),
        )
        for i in range(n_history)
    ]


def current_action(row: JsonObject) -> Action:
    # Falsy check mirrors upstream eval.py exactly (their judge uses the same fallback).
    content = row.get("current_prompt") or row["prompt"][int(row["turn_idx"]) - 1]
    return Action(kind=ActionKind.MESSAGE, content=content)


def wrap_gen(observation: Observation) -> str:
    """Their output_parser extracts the LAST <predicted_observation> block."""
    return f"<predicted_observation>\n{observation.content}\n</predicted_observation>"


def with_retry(fn: Callable[[], Observation]) -> Observation:
    """Retry provider errors with capped exponential backoff + jitter (Bedrock throttling)."""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return fn()
        except Exception:
            if attempt == RETRY_ATTEMPTS - 1:
                raise
            delay = min(60.0, 5.0 * 2**attempt) + random.uniform(0.0, 3.0)
            time.sleep(delay)
    raise AssertionError("unreachable")


def predict_row_wm(
    world_model, serve_model: str, model_dir: str, row: JsonObject, *, max_fidelity: bool = False
) -> None:
    """One row through the serving session path (retrieval included), per-session metering."""

    def attempt() -> Observation:
        session = world_model.new_session(task=None)
        try:
            world_model.seed_session(session.id, history_steps(row))
            return world_model.step(session.id, current_action(row))
        finally:
            usage = world_model.end_session(session.id)
            row["wmh_infer"] = {
                "mode": "wm",
                "model_dir": model_dir,
                "max_fidelity": max_fidelity,
                "serve_model": serve_model,
                "input_tokens": usage.total.input_tokens,
                "output_tokens": usage.total.output_tokens,
                "cost_usd": usage.total.cost_usd,
            }

    row["gen"] = wrap_gen(with_retry(attempt))


def predict_row_base(
    provider: Provider,
    serve_model: str,
    row: JsonObject,
    *,
    reasoning: bool = False,
    verify: bool = False,
) -> None:
    """One row with BASE_ENV_PROMPT and no retrieval; per-row tracker for clean attribution.

    `reasoning`/`verify` are the corpus-free max levers (verify = one extra completion that
    audits the draft — doubles per-row provider cost).
    """
    tracker = RunTracker(run_id=f"awb-base-{row['id']}-{row['turn_idx']}", kind="eval")
    tracker.start()
    metered = MeteredProvider(provider, tracker, base_phase=Phase.SERVE)

    def attempt() -> Observation:
        history = history_steps(row)
        draft = predict_observation(
            metered,
            BASE_ENV_PROMPT,
            None,
            EnvState(),
            current_action(row),
            demos=[],
            history=history,
            reasoning=reasoning,
        )
        if not verify:
            return draft
        return verify_observation(
            metered,
            BASE_ENV_PROMPT,
            None,
            EnvState(),
            current_action(row),
            draft,
            demos=[],
            history=history,
            reasoning=reasoning,
        )

    try:
        row["gen"] = wrap_gen(with_retry(attempt))
    finally:
        total = tracker.record_summary().total
        row["wmh_infer"] = {
            "mode": "base",
            "serve_model": serve_model,
            "reasoning": reasoning,
            "verify": verify,
            "input_tokens": total.input_tokens,
            "output_tokens": total.output_tokens,
            "cost_usd": total.cost_usd,
        }


def row_key(row: JsonObject) -> tuple[str, int]:
    return (str(row["id"]), int(row["turn_idx"]))


def load_done(out: Path) -> dict[tuple[str, int], JsonObject]:
    """Rows already predicted (non-empty gen) in an existing output file."""
    done: dict[tuple[str, int], JsonObject] = {}
    if out.exists():
        # Iterate the file (splits on \n only): rows can embed  -class separators that
        # str.splitlines() would split mid-JSON-string (bit the search domain).
        with out.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("gen"):
                    done[row_key(row)] = row
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="one AgentWorldBench {domain}_test.jsonl")
    parser.add_argument("--mode", choices=["wm", "base"], required=True)
    parser.add_argument("--model-dir", help="built wmh model dir (required for --mode wm)")
    parser.add_argument("--provider-model", default="us.anthropic.claude-opus-4-8")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--limit", type=int, default=None, help="first N rows only")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="skip rows already in --output")
    parser.add_argument("--output", required=True, help="predictions.jsonl path")
    parser.add_argument(
        "--max-fidelity",
        action="store_true",
        help="wm mode: apply the model's auto_fidelity.json winner (plain load = pure RAG)",
    )
    parser.add_argument("--reasoning", action="store_true", help="base mode: reasoning lever")
    parser.add_argument(
        "--verify", action="store_true", help="base mode: verify second pass (2x provider cost)"
    )
    args = parser.parse_args()
    if args.mode == "wm" and not args.model_dir:
        parser.error("--mode wm requires --model-dir")

    with Path(args.data).open(encoding="utf-8") as f:  # \n-only splitting, see load_done
        rows: list[JsonObject] = [json.loads(line) for line in f if line.strip()]
    if args.limit is not None:
        rows = rows[: args.limit]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out) if args.resume else {}
    todo = [r for r in rows if row_key(r) not in done]
    # Compact the file to exactly the completed rows, then append as new rows finish — a
    # crash/restart with --resume never duplicates a row or loses a finished one.
    with out.open("w", encoding="utf-8") as f:
        for row in done.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"{len(rows)} rows from {args.data} (mode={args.mode}); {len(done)} done, {len(todo)} to run"
    )

    if args.mode == "wm":
        world_model, provider = load_world_model(args.model_dir, max_fidelity=args.max_fidelity)
        serve_model = provider.config.model
        winner_path = Path(args.model_dir) / "auto_fidelity.json"
        if args.max_fidelity and not winner_path.exists():
            parser.error(f"--max-fidelity: no auto_fidelity.json in {args.model_dir}")
        winner = json.loads(winner_path.read_text()) if winner_path.exists() else None
        if args.max_fidelity:
            print(f"max-fidelity winner: {winner}")
        predict = lambda row: predict_row_wm(  # noqa: E731
            world_model, serve_model, args.model_dir, row, max_fidelity=args.max_fidelity
        )
    else:
        provider = get_provider(
            ProviderConfig(kind=ProviderKind.BEDROCK, model=args.provider_model, region=args.region)
        )
        predict = lambda row: predict_row_base(  # noqa: E731
            provider, args.provider_model, row, reasoning=args.reasoning, verify=args.verify
        )

    write_lock = threading.Lock()
    counter = {"done": 0, "failed": 0}

    def work(row: JsonObject) -> None:
        try:
            predict(row)
        except Exception:
            row["gen"] = ""  # their judge marks gen == "" as failed
            row.setdefault("wmh_infer", {"mode": args.mode, "error": True})
            failed = True
            print(f"FAILED id={row.get('id')} turn={row.get('turn_idx')}", file=sys.stderr)
            traceback.print_exc()
        else:
            failed = False
        with write_lock:
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            counter["done"] += 1
            counter["failed"] += 1 if failed else 0
            n = counter["done"]
        if n % 10 == 0 or n == len(todo):
            print(f"[{n}/{len(todo)}] done ({counter['failed']} failed)", flush=True)

    if args.concurrency > 1 and len(todo) > 1:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(work, todo))
    else:
        for row in todo:
            work(row)

    finished = load_done(out)
    total_cost = sum((r["wmh_infer"].get("cost_usd") or 0.0) for r in finished.values())
    print(
        f"wrote {out} — {len(finished)}/{len(rows)} predictions "
        f"({counter['failed']} failed this pass), infer cost ${total_cost:.2f}"
    )


if __name__ == "__main__":
    main()
