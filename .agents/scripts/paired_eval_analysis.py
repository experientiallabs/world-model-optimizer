"""Paired per-scenario comparison of wm_tau eval rows (same pinned eval scenarios).

Usage: python3 paired_eval_analysis.py base.jsonl [label=path.jsonl ...]
Prints per-scenario mean rewards side-by-side plus paired deltas vs the first file.
"""

from __future__ import annotations

import json
import statistics
import sys


def load(path: str) -> dict[str, float]:
    by: dict[str, list[float]] = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if not r["errors"]:
                by.setdefault(r["scenario_id"][:8], []).append(r["reward"])
    return {k: sum(v) / len(v) for k, v in by.items()}


def main() -> None:
    base_path = sys.argv[1]
    bad = [arg for arg in sys.argv[2:] if "=" not in arg]
    if bad:
        raise SystemExit(f"comparison args must be label=path.jsonl, got: {bad}")
    others = [arg.split("=", 1) for arg in sys.argv[2:]]
    labels = [label for label, _ in others]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"duplicate labels: {labels}")
    base = load(base_path)
    rows = {label: load(path) for label, path in others}
    common = sorted(set(base).intersection(*[set(r) for r in rows.values()]))

    header = f"{'scenario':10} {'base':>6}" + "".join(f" {label:>8}" for label in rows)
    print(header)
    for s in common:
        line = f"{s:10} {base[s]:6.2f}" + "".join(f" {rows[label][s]:8.2f}" for label in rows)
        print(line)
    if not common:
        raise SystemExit("no clean scenarios shared across all files - nothing to compare")
    for label, vals in rows.items():
        deltas = [vals[s] - base[s] for s in common]
        wins = sum(1 for d in deltas if d > 0.05)
        losses = sum(1 for d in deltas if d < -0.05)
        print(
            f"paired {label}-base: mean {statistics.mean(deltas):+.3f} "
            f"median {statistics.median(deltas):+.3f} wins {wins} losses {losses} "
            f"(n={len(deltas)})"
        )


if __name__ == "__main__":
    main()
