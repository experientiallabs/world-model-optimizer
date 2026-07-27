"""Real tau-bench matrix: our 9 pool models against Sierra's tau2-bench, scored by ITS reward.

The sim-to-real arm of the capture round. Every other matrix in this project scores candidates
inside a world model with an LLM judge; this one runs the actual benchmark and takes tau2's own
deterministic reward (DB state check plus action/communication checks). Fit on the wm matrices,
test here.

What is held fixed so the comparison is about the candidate and nothing else:

- The USER SIMULATOR is part of the environment, so it is pinned to one cheap model for every run
  (`USER_LLM`). Letting it vary by candidate would change the environment per candidate and make
  the rewards incomparable.
- Empty LLM args (`{}`) for both streams, which drops tau2's default temperature. Several pool
  models reject sampling params, and dropping it everywhere removes a source of cross-model
  variation rather than only working around the strict ones.
- Same task ids for every model, same `--max-concurrency`, one trial per (task, model) per pass.

Two deviations from the brief, both deliberate:

1. `scenario_id` is `"<domain>:<task_id>"`, not the bare tau2 task id. Airline and retail both
   number their tasks from "0", and all 50 airline ids collide with retail ids, so bare ids would
   silently merge two different tasks into one matrix cell.
2. `cost_usd` is computed from the recorded token usage at OUR pool prices, not taken from tau2's
   `agent_cost`. tau2 gets its number from litellm's price table, which does not know our Azure
   MaaS deployments: on the smoke run it priced Kimi-K2.6 at $0.0197 where the published eastus2
   meters give $0.0351. The sidecar JSONL keeps tau2's figure alongside ours for audit.

Budget: `BUDGET_STOP` is checked after every batch, so a stop leaves a usable partial grid. Models
run cheapest-first for the same reason.

Usage (from the repo root, with the tau2 venv and keys available):
    uv run .agents/scripts/run_tau_bench_real.py --tasks-per-domain 20 --trial 0
    uv run .agents/scripts/run_tau_bench_real.py --write-matrix-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.providers.base import TokenUsage
from wmo.providers.pool import load_pool

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from wmo.providers.pool import PoolEntry

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("tau-real")

CAPTURE = Path("/Users/silen/Desktop/Projects/wmh-bench-infra/tools/tau2-capture")
POOL = Path("/Users/silen/Desktop/Projects/wmh-optimizer-switch/.wmh/pool.toml")
OUT_DIR = Path(".wmo/evals/tau-bench-real")
ROWS = OUT_DIR / "rows.jsonl"
MATRIX = Path(".wmo/evals/tau-bench-real_matrix.json")
DOMAINS = ("airline", "retail")

# The user simulator is environment, not candidate: one cheap model, identical for every run.
USER_LLM = "azure/gpt-5.4-mini"
BUDGET_STOP = 140.0
MAX_CONCURRENCY = 4
BATCH_TIMEOUT_S = 5400

# litellm route per pool entry name. Anthropic reads ANTHROPIC_API_KEY; the two Azure families use
# DIFFERENT env prefixes (AZURE_* vs AZURE_AI_*), which is what lets a google-sheets user simulator
# run against a silen-resource candidate without either clobbering the other's credentials.
ROUTES = {
    "gpt-5.4-mini": "azure/gpt-5.4-mini",
    "gpt-5.5": "azure/gpt-5.5",
    "deepseek-v4-pro": "azure_ai/DeepSeek-V4-Pro",
    "kimi-k2.6": "azure_ai/Kimi-K2.6",
    "glm-5.2": "azure_ai/FW-GLM-5.2",
    "haiku-4-5": "anthropic/claude-haiku-4-5",
    "sonnet-5": "anthropic/claude-sonnet-5",
    "opus-4-8": "anthropic/claude-opus-4-8",
    "fable-5": "anthropic/claude-fable-5",
}


def price_order(pool: Sequence[PoolEntry]) -> list[PoolEntry]:
    """Cheapest first, using the token mix the smoke run actually measured.

    Ordering by a realistic mix rather than by input price alone: agent-side episodes are roughly
    30k input to 1k output, so a model with a cheap input rate and an expensive output rate does
    not get to look cheaper than it is.
    """

    def episode_cost(entry: PoolEntry) -> float:
        return entry.cost_usd(TokenUsage(input_tokens=30_000, output_tokens=1_000))

    return sorted(pool, key=episode_cost)


def task_ids(domain: str, limit: int) -> list[str]:
    """The first `limit` tau2 task ids for `domain`, in file order (deterministic and resumable)."""
    path = CAPTURE / "tau2-bench/data/tau2/domains" / domain / "tasks.json"
    tasks = json.loads(path.read_text(encoding="utf-8"))
    return [str(task["id"]) for task in tasks[:limit]]


def load_rows() -> list[dict[str, Any]]:
    if not ROWS.exists():
        return []
    return [json.loads(line) for line in ROWS.read_text(encoding="utf-8").splitlines() if line]


def append_rows(rows: Iterable[dict[str, Any]]) -> None:
    ROWS.parent.mkdir(parents=True, exist_ok=True)
    with ROWS.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _stream_tokens(messages: list[dict[str, Any]], role: str) -> tuple[int, int]:
    prompt = completion = 0
    for message in messages:
        if message.get("role") != role:
            continue
        usage = message.get("usage") or {}
        prompt += int(usage.get("prompt_tokens") or 0)
        completion += int(usage.get("completion_tokens") or 0)
    return prompt, completion


def run_batch(
    entry: PoolEntry, domain: str, ids: list[str], trial: int, env: dict[str, str]
) -> list[dict[str, Any]]:
    """Run one (model, domain) batch through the tau2 CLI and return sidecar rows."""
    save_to = f"real_{entry.name.replace('.', '_')}_{domain}_t{trial}"
    command = [
        str(CAPTURE / ".venv/bin/tau2"),
        "run",
        "--domain",
        domain,
        "--agent-llm",
        ROUTES[entry.name],
        "--agent-llm-args",
        "{}",
        "--user-llm",
        USER_LLM,
        "--user-llm-args",
        "{}",
        "--num-trials",
        "1",
        "--task-ids",
        *ids,
        "--max-concurrency",
        str(MAX_CONCURRENCY),
        "--save-to",
        save_to,
    ]
    logger.info("  %s / %s: %d tasks -> %s", entry.name, domain, len(ids), save_to)
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell, ids come from tasks.json
        command, cwd=CAPTURE, env=env, capture_output=True, text=True, timeout=BATCH_TIMEOUT_S
    )
    results = CAPTURE / "tau2-bench/data/simulations" / save_to / "results.json"
    if not results.is_file():
        logger.error("  no results.json (exit %d); stderr tail:", completed.returncode)
        logger.error("  %s", (completed.stderr or "")[-600:])
        return []

    payload = json.loads(results.read_text(encoding="utf-8"))
    text_by_id = {
        str(task["id"]): json.dumps(task.get("user_scenario", {}).get("instructions", {}))
        for task in payload.get("tasks", [])
    }
    rows: list[dict[str, Any]] = []
    for sim in payload.get("simulations", []):
        messages = sim.get("messages", [])
        agent_in, agent_out = _stream_tokens(messages, "assistant")
        user_in, user_out = _stream_tokens(messages, "user")
        task_id = str(sim.get("task_id"))
        rows.append(
            {
                "scenario_id": f"{domain}:{task_id}",
                "domain": domain,
                "task_id": task_id,
                "task": text_by_id.get(task_id, ""),
                "model": entry.name,
                "route": ROUTES[entry.name],
                "episode": trial,
                "reward": (sim.get("reward_info") or {}).get("reward"),
                "termination_reason": sim.get("termination_reason") or "",
                "duration_s": sim.get("duration"),
                "agent_input_tokens": agent_in,
                "agent_output_tokens": agent_out,
                "user_input_tokens": user_in,
                "user_output_tokens": user_out,
                # Ours (pool prices) is authoritative; tau2's is litellm's and wrong for MaaS.
                "cost_usd_pool": entry.cost_usd(
                    TokenUsage(input_tokens=agent_in, output_tokens=agent_out)
                ),
                "cost_usd_tau2_agent": sim.get("agent_cost"),
                "cost_usd_tau2_user": sim.get("user_cost"),
                "cost_estimated": True,
                "steps": sum(1 for m in messages if m.get("tool_calls")),
                "call_seconds": [
                    float(m["generation_time_seconds"])
                    for m in messages
                    if m.get("role") == "assistant" and m.get("generation_time_seconds")
                ],
                "replies": [
                    m.get("content") or ""
                    for m in messages
                    if m.get("role") == "assistant" and m.get("content")
                ],
                "user_sim": USER_LLM,
            }
        )
    return rows


def to_matrix(rows: list[dict[str, Any]], pool: Sequence[PoolEntry]) -> OutcomeMatrix:
    """Sidecar rows -> OutcomeMatrix. Rows with no reward stay unscored, never zeroed."""
    outcomes = [
        ScenarioOutcome(
            scenario_id=row["scenario_id"],
            task=row["task"],
            model=row["model"],
            episode=int(row["episode"]),
            reward=row["reward"],
            success=bool(row["reward"] is not None and row["reward"] >= 1.0),
            steps=int(row["steps"]),
            stop_reason=row["termination_reason"],
            usage=TokenUsage(
                input_tokens=int(row["agent_input_tokens"]),
                output_tokens=int(row["agent_output_tokens"]),
            ),
            cost_usd=float(row["cost_usd_pool"]),
            call_seconds=[float(value) for value in row["call_seconds"]],
            replies=[str(value) for value in row["replies"]],
        )
        for row in rows
    ]
    return OutcomeMatrix(pool=list(pool), outcomes=outcomes)


def build_env() -> dict[str, str]:
    """tau2 needs its data dir plus both Azure credential families and the Anthropic key."""
    env = dict(os.environ)
    env["TAU2_DATA_DIR"] = str(CAPTURE / "tau2-bench/data")
    missing = [
        name
        for name in (
            "ANTHROPIC_API_KEY",
            "AZURE_GOOGLE_SHEETS_API_KEY",
            "AZURE_SILEN_RESOURCE_API_KEY",
        )
        if not env.get(name)
    ]
    if missing:
        raise SystemExit(f"missing credentials in the environment: {missing}")
    env["AZURE_API_KEY"] = env["AZURE_GOOGLE_SHEETS_API_KEY"]
    env["AZURE_API_BASE"] = "https://google-sheets.openai.azure.com"
    env["AZURE_API_VERSION"] = "2024-10-21"
    env["AZURE_AI_API_KEY"] = env["AZURE_SILEN_RESOURCE_API_KEY"]
    env["AZURE_AI_API_BASE"] = "https://silen-resource.services.ai.azure.com/models"
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-per-domain", type=int, default=20)
    parser.add_argument(
        "--trial", type=int, default=0, help="episode index; rerun with 1 for a 2nd"
    )
    parser.add_argument("--only", nargs="*", default=None, help="restrict to these pool names")
    parser.add_argument("--write-matrix-only", action="store_true")
    args = parser.parse_args()

    pool = load_pool(POOL).models
    rows = load_rows()
    if args.write_matrix_only:
        MATRIX.parent.mkdir(parents=True, exist_ok=True)
        to_matrix(rows, pool).save(MATRIX)
        logger.info("wrote %s from %d rows", MATRIX, len(rows))
        return

    wanted = {domain: task_ids(domain, args.tasks_per_domain) for domain in DOMAINS}
    done = {(row["scenario_id"], row["model"], int(row["episode"])) for row in rows}
    spent = sum(
        float(row["cost_usd_pool"]) + float(row.get("cost_usd_tau2_user") or 0.0) for row in rows
    )
    logger.info(
        "resume: %d rows on disk, $%.2f already spent, budget stop $%.0f",
        len(rows),
        spent,
        BUDGET_STOP,
    )

    env = build_env()
    for entry in price_order(pool):
        if entry.name not in ROUTES:
            logger.warning("no litellm route for %s, skipping", entry.name)
            continue
        if args.only and entry.name not in args.only:
            continue
        for domain in DOMAINS:
            missing = [
                task_id
                for task_id in wanted[domain]
                if (f"{domain}:{task_id}", entry.name, args.trial) not in done
            ]
            if not missing:
                continue
            if spent >= BUDGET_STOP:
                logger.warning("BUDGET STOP at $%.2f; leaving the grid partial", spent)
                to_matrix(load_rows(), pool).save(MATRIX)
                return
            batch = run_batch(entry, domain, missing, args.trial, env)
            append_rows(batch)
            batch_cost = sum(
                float(row["cost_usd_pool"]) + float(row.get("cost_usd_tau2_user") or 0.0)
                for row in batch
            )
            spent += batch_cost
            scored = [row["reward"] for row in batch if row["reward"] is not None]
            logger.info(
                "  -> %d sims, mean reward %.3f, batch $%.3f, running total $%.2f",
                len(batch),
                sum(scored) / len(scored) if scored else float("nan"),
                batch_cost,
                spent,
            )

    MATRIX.parent.mkdir(parents=True, exist_ok=True)
    to_matrix(load_rows(), pool).save(MATRIX)
    logger.info("wrote %s; total spend $%.2f", MATRIX, spent)


if __name__ == "__main__":
    main()
