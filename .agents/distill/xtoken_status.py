"""One 80-column status line for a running cross-tokenizer distillation run.

Reads the run's `metrics.jsonl` (written per completed step) and the launcher
log, so it never touches the network and cannot perturb the run. Emits exactly
one line: step progress, resource use, and an ETA extrapolated from completed
step durations.

Usage:
    uv run python .agents/distill/xtoken_status.py .wmh/xtoken-runs/run3 --steps 6
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

FIREWORKS_INPUT_USD_PER_MTOK = 1.40
"""Teacher scoring is prefill-only, and `echo` bills uncached."""

BAR_WIDTH = 8


def _bar(done: int, total: int) -> str:
    """A fixed-width progress bar, filled proportionally."""
    if total <= 0:
        return "?" * BAR_WIDTH
    filled = min(BAR_WIDTH, round(BAR_WIDTH * done / total))
    return "#" * filled + "." * (BAR_WIDTH - filled)


def _human_eta(seconds: float) -> str:
    """A compact ETA: seconds under a minute, then minutes, then hours."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--steps", type=int, required=True, help="total planned steps")
    parser.add_argument("--log", default=None, help="launcher log, for liveness")
    parser.add_argument("--url", default="", help="wandb run URL, appended to the line")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    rows: list[dict[str, float]] = []
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))

    done = len(rows)
    if not rows:
        age = ""
        if args.log and Path(args.log).exists():
            age = f" up {_human_eta(time.time() - Path(args.log).stat().st_mtime)}"
        tail = f" {args.url}" if args.url else ""
        print(f"s0/{args.steps} 0% | sampling{age} | starting{tail}")
        return

    last = rows[-1]
    mean_secs = sum(float(r["seconds"]) for r in rows) / done
    eta = _human_eta(mean_secs * max(0, args.steps - done))
    teacher_tokens = float(last.get("teacher_tokens", 0))
    spend = teacher_tokens * FIREWORKS_INPUT_USD_PER_MTOK / 1e6
    tok = float(last.get("mean_sampled_tokens", 0))
    kept = int(last.get("datums", 0))
    dropped = int(last.get("dropped_rollouts", 0))
    kl = last.get("chunk_reverse_kl")
    kl_text = "n/a" if kl is None else f"{float(kl):+.3f}"
    cov = float(last.get("coverage_rate", 0)) * 100

    # Plain words, no jargon: this line is read on its own with no context.
    done_pct = 100 * done / args.steps
    tail = f"  {args.url}" if args.url else ""
    # Spelled out: this line is read cold, with no legend and no context.
    total_rollouts = kept + dropped
    finished = (
        f"all {kept} answers complete"
        if dropped == 0
        else f"{kept} of {total_rollouts} complete, {dropped} CUT OFF"
    )
    gap = "n/a" if kl is None else f"{float(kl):.3f}"
    scored = "" if cov >= 99.5 else f" | only {cov:.0f}% of tokens scored"
    line = (
        f"training step {done} of {args.steps} | {finished} | "
        f"{tok / 1000:.0f}k tokens of reasoning each | "
        f"student-vs-teacher gap {gap} (want it falling){scored} | "
        f"spent ${spend:.2f} | ~{eta} to go{tail}"
    )
    print(line)


if __name__ == "__main__":
    main()
