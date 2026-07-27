"""Stage B capture: OUR 9-model pool over certified RouterBench MCQ prompts.

Runs every pool model on a stratified sample of gold-certified prompts (exact-match grading,
judge-free), recording per-call usage, latency, and entry-priced cost. Resumable: rows append
to a JSONL keyed (scenario_id, model); rerunning skips what exists. Output: an OutcomeMatrix
JSON ready for `wmo optimize route fit`.
"""

from __future__ import annotations

import json
import logging
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from routerbench_gold import parse_letter  # noqa: E402

from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome  # noqa: E402
from wmo.providers.base import Message, TokenUsage  # noqa: E402
from wmo.providers.pool import load_pool, pool_provider  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
for noisy in ("httpx", "urllib3", "botocore", "anthropic", "openai"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("routerbench_ours")

PICKLE = Path("/Users/silen/Desktop/Projects/router-refs/routerbench_0shot.pkl")
GOLD = Path(".wmo/evals/routerbench/gold.json")
ROWS = Path(".wmo/evals/routerbench/ours_rows.jsonl")
MATRIX_OUT = Path(".wmo/evals/routerbench/ours_matrix.json")

SAMPLE = 1200
SEED = 11
CONCURRENCY = 12
MAX_TOKENS = 1024


def _prompt_text(raw: object) -> str:
    if isinstance(raw, list):
        return str(raw[0])
    return str(raw)


def main() -> None:
    gold: dict[str, str] = json.loads(GOLD.read_text())
    frame = pd.read_pickle(PICKLE)
    prompts: dict[str, str] = {}
    for row in frame[["eval_name", "sample_id", "prompt"]].to_dict("records"):
        sid = f"{row['eval_name']}:{row['sample_id']}"
        if sid in gold:
            prompts[sid] = _prompt_text(row["prompt"])

    # Stratified proportional sample over eval prefixes.
    by_eval: dict[str, list[str]] = {}
    for sid in sorted(prompts):
        by_eval.setdefault(sid.split(":", 1)[0], []).append(sid)
    rng = random.Random(SEED)
    total = sum(len(v) for v in by_eval.values())
    chosen: list[str] = []
    for name, ids in sorted(by_eval.items()):
        quota = max(1, round(SAMPLE * len(ids) / total))
        chosen.extend(rng.sample(ids, min(quota, len(ids))))
    chosen = sorted(chosen)[:SAMPLE]
    logger.info("sample: %d prompts over %d evals", len(chosen), len(by_eval))

    pool = load_pool()
    providers = {entry.name: pool_provider(entry) for entry in pool.models}

    done: set[tuple[str, str]] = set()
    if ROWS.exists():
        for line in ROWS.read_text().splitlines():
            row = json.loads(line)
            done.add((row["scenario_id"], row["model"]))
        logger.info("resuming: %d rows already captured", len(done))
    lock = threading.Lock()
    handle = ROWS.open("a", encoding="utf-8")

    def run_cell(entry_name: str, sid: str) -> None:
        entry = pool.entry(entry_name)
        provider = providers[entry_name]
        started = time.monotonic()
        try:
            completion = provider.complete(
                "", [Message(role="user", content=prompts[sid])], max_tokens=MAX_TOKENS
            )
            seconds = time.monotonic() - started
            letter = parse_letter(completion.text)
            record = {
                "scenario_id": sid,
                "model": entry_name,
                "letter": letter,
                "gold": gold[sid],
                "reward": 1.0 if letter == gold[sid] else 0.0,
                "input_tokens": completion.usage.input_tokens,
                "output_tokens": completion.usage.output_tokens,
                "cost_usd": entry.cost_usd(completion.usage),
                "seconds": seconds,
                "reply": completion.text[:400],
            }
        except Exception as exc:  # noqa: BLE001 - capture must survive one bad cell
            record = {
                "scenario_id": sid,
                "model": entry_name,
                "error": f"{type(exc).__name__}: {exc}"[:300],
                "seconds": time.monotonic() - started,
            }
        with lock:
            handle.write(json.dumps(record) + "\n")
            handle.flush()

    cells = [
        (entry.name, sid)
        for entry in pool.models
        for sid in chosen
        if (sid, entry.name) not in done
    ]
    logger.info("cells to run: %d", len(cells))
    completed = 0
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = [executor.submit(run_cell, name, sid) for name, sid in cells]
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 500 == 0:
                logger.info("progress: %d/%d", completed, len(cells))
    handle.close()

    # Assemble the OutcomeMatrix from ALL captured rows for the chosen sample.
    outcomes: list[ScenarioOutcome] = []
    total_cost = 0.0
    for line in ROWS.read_text().splitlines():
        row = json.loads(line)
        if row["scenario_id"] not in prompts or row["scenario_id"] not in set(chosen):
            continue
        if "error" in row:
            outcomes.append(
                ScenarioOutcome(
                    scenario_id=row["scenario_id"],
                    task=prompts[row["scenario_id"]],
                    model=row["model"],
                    error=row["error"],
                )
            )
            continue
        total_cost += row["cost_usd"]
        outcomes.append(
            ScenarioOutcome(
                scenario_id=row["scenario_id"],
                task=prompts[row["scenario_id"]],
                model=row["model"],
                reward=row["reward"],
                success=row["reward"] >= 1.0,
                steps=1,
                stop_reason="routerbench-ours",
                usage=TokenUsage(
                    input_tokens=row["input_tokens"], output_tokens=row["output_tokens"]
                ),
                cost_usd=row["cost_usd"],
                call_seconds=[row["seconds"]],
                replies=[row["reply"]],
            )
        )
    OutcomeMatrix(pool=pool.models, outcomes=outcomes).save(MATRIX_OUT)
    logger.info(
        "matrix: %d outcomes, capture cost $%.2f -> %s", len(outcomes), total_cost, MATRIX_OUT
    )


if __name__ == "__main__":
    main()
