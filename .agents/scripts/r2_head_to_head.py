"""Head-to-head: r2-auto vs every r2 variant and r1's recorded champion, paired by seed.

r1's runs use the same `split_scenario_ids` seeds on the same matrices (their wm matrices
are named `wm-<corpus>`), so (matrix, seed) rows pair exactly on iid splits. Reports, per
matrix/split: paired dAcc vs the IN-CELL best-single for every variant, plus the direct
paired difference r2-auto minus r1-knn-statz05-oai where r1 rows exist.

Usage: uv run python .agents/scripts/r2_head_to_head.py
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from wmo.research.routing_corpus import routing_data

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("h2h")

DATA = routing_data()
R1_ALIAS = {
    "tau-bench": "wm-tau-bench",
    "bird-sql": "wm-bird-sql",
    "continual-learning": "wm-continual-learning",
    "terminal-tasks": "wm-terminal-tasks",
    "crmarena": "wm-crmarena",
    "dabstep": "wm-dabstep",
    "financebench": "wm-financebench",
    "routerbench-ours9": "routerbench-ours9",
}


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    r2 = _load(DATA / "runs" / "r2.jsonl")
    r1 = _load(DATA / "runs" / "r1.jsonl")
    cells: dict[tuple, dict] = {}
    for rec in r2:
        cells[(rec["matrix"], rec["params"].get("split"), rec["variant"], rec["split_seed"])] = rec
    r1_cells: dict[tuple, dict] = {}
    for rec in r1:
        if rec["params"].get("split") in (None, "iid"):  # r1 later batches add OOD rows
            r1_cells[(rec["matrix"], rec["variant"], rec["split_seed"])] = rec

    matrices = [
        "routerbench-ours9", "tau-bench", "financebench", "continual-learning",
        "bird-sql", "terminal-tasks", "wm-all",
    ]
    variants = [
        "r2-auto", "r2-costaware-single", "r2-knn-prox-z05-oai", "r2-knn-prox-oai",
        "r2-l2d-lam01", "r2-prox-val",
    ]
    for matrix in matrices:
        for split in ["iid", "ood-cluster", "ood-task"]:
            rows = []
            for variant in variants:
                deltas, dcost, wins = [], [], 0
                for seed in range(5):
                    rec = cells.get((matrix, split, variant, seed))
                    base = cells.get((matrix, split, "r2-best-single", seed))
                    if not rec or not base:
                        continue
                    delta = rec["result"]["accuracy"] - base["result"]["accuracy"]
                    deltas.append(delta)
                    wins += delta > 0
                    if base["result"]["cost_per_call"] > 0:
                        dcost.append(
                            (rec["result"]["cost_per_call"] / base["result"]["cost_per_call"] - 1)
                            * 100
                        )
                if len(deltas) >= 4:
                    rows.append((variant, statistics.mean(deltas), min(deltas), wins,
                                 statistics.mean(dcost), len(deltas)))
            # r1 champion on iid, paired vs THEIR recorded best-single baseline.
            if split == "iid":
                alias = R1_ALIAS.get(matrix)
                deltas, dcost = [], []
                for seed in range(5):
                    rec = r1_cells.get((alias, "r1-knn-statz05-oai", seed)) or r1_cells.get(
                        (alias, "r1-knn-statz05", seed)
                    )
                    if rec and rec.get("baselines", {}).get("best_single"):
                        base = rec["baselines"]["best_single"]
                        deltas.append(rec["result"]["accuracy"] - base["accuracy"])
                        if base["cost_per_call"] > 0:
                            dcost.append(
                                (rec["result"]["cost_per_call"] / base["cost_per_call"] - 1) * 100
                            )
                if len(deltas) >= 4:
                    rows.append(("r1-knn-statz05(-oai)", statistics.mean(deltas), min(deltas),
                                 sum(d > 0 for d in deltas), statistics.mean(dcost), len(deltas)))
            if rows:
                logger.info("\n== %s / %s ==", matrix, split)
                for variant, mean, worst, wins, cost, n in rows:
                    logger.info(
                        "  %-24s dAcc %+.4f (worst %+.4f, %d/%d) dCost %+6.1f%%",
                        variant, mean, worst, wins, n, cost,
                    )
    # Direct paired difference: r2-auto minus r1 champion on shared (matrix, seed).
    logger.info("\n== direct paired r2-auto minus r1-knn-statz05-oai (iid) ==")
    for matrix in matrices:
        alias = R1_ALIAS.get(matrix)
        diffs = []
        for seed in range(5):
            mine = cells.get((matrix, "iid", "r2-auto", seed))
            theirs = r1_cells.get((alias, "r1-knn-statz05-oai", seed)) or r1_cells.get(
                (alias, "r1-knn-statz05", seed)
            )
            if mine and theirs:
                diffs.append(mine["result"]["accuracy"] - theirs["result"]["accuracy"])
        if len(diffs) >= 4:
            logger.info(
                "  %-20s %+.4f (worst %+.4f, %d/%d wins)",
                matrix, statistics.mean(diffs), min(diffs),
                sum(d > 0 for d in diffs), len(diffs),
            )


if __name__ == "__main__":
    main()
