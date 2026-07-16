"""Summarize wm_tau episode-result JSONL files: row stats + recent rollouts to eyeball.

Usage: python3 summarize_wm_tau_results.py results.jsonl [--last N] [--full]
Episodes with errors are excluded from the stats (broken runs append to the same file);
they are counted separately so silent loss is visible.
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--last", type=int, default=4, help="rollouts to print")
    parser.add_argument("--full", action="store_true", help="print full critiques/actions")
    args = parser.parse_args()

    with open(args.path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    clean = [r for r in rows if not r["errors"]]
    errored = len(rows) - len(clean)
    if not clean:
        print(f"{args.path}: no clean episodes ({errored} errored)")
        return

    n = len(clean)
    success = sum(r["success"] for r in clean)
    mean_reward = sum(r["reward"] for r in clean) / n
    mean_steps = sum(r["steps"] for r in clean) / n
    serve_cost = sum(r["wm_serve_cost_usd"] for r in clean)
    judge_cost = sum(r["wm_judge_cost_usd"] for r in clean)
    print(
        f"{args.path}\n"
        f"  clean={n} errored={errored} | success_rate={success / n:.3f} ({success}/{n}) "
        f"| mean_reward={mean_reward:.3f} | mean_steps={mean_steps:.1f}\n"
        f"  wm serve cost=${serve_cost:.2f} judge cost=${judge_cost:.2f}"
    )
    per_scenario: dict[str, list[float]] = {}
    for r in clean:
        per_scenario.setdefault(r["scenario_id"][:8], []).append(r["reward"])
    line = " ".join(f"{k}:{'/'.join(f'{v:.2f}' for v in vs)}" for k, vs in per_scenario.items())
    print(f"  per-scenario rewards: {line}")

    cap = None if args.full else 240
    tail = clean[-args.last :] if args.last > 0 else []
    for r in tail:
        print("=" * 72)
        actions = [a for turn in r["actions"] for a in turn]
        print(
            f"{r['scenario_id'][:8]} r{r['rollout_index']} reward={r['reward']:.2f} "
            f"success={r['success']} steps={r['steps']} stop={r['stop_reason']}"
        )
        print(f"  actions: {actions if args.full else actions[:10]}")
        print(f"  step_rewards: {r['step_rewards']}")
        print(f"  critique: {r['critique'][:cap]}")


if __name__ == "__main__":
    main()
