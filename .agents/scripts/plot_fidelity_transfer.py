#!/usr/bin/env python3
"""Plot the D67 fidelity->transfer curve: training-WM fidelity (X) vs real-env transfer (Y).

Reads the committed artifacts on this branch — the four fidelity cells
(.agents/docs/research/tau_fidelity_cells/) and the real-env row records
(.agents/docs/research/real_tau_eval_results/) — computes each curve point's paired
delta vs the base row, and draws the chart in the repo's chart style (see
plot_trace_scaling.py). The collapsed protocol-faithful sonnet checkpoint is drawn as a
separate marker to show the stability cliff without letting it distort the healthy curve.

Usage (repo root):
    uv run python .agents/scripts/plot_fidelity_transfer.py \
        --out docs/research/fidelity_transfer_curve.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[2]
_CELLS = _REPO / ".agents/docs/research/tau_fidelity_cells"
_ROWS = _REPO / ".agents/docs/research/real_tau_eval_results"

_INK = "#0a0a0a"
_GRID = "#ececec"
_BLUE = "#0070f3"
_RED = "#e00"

# (label, fidelity-cell json, real-env row jsonl, checkpoint provenance)
POINTS = [
    ("haiku no-RAG", "haiku_norag", "haiku_norag"),
    ("haiku+RAG", "haiku_rag", "haiku_rag"),
    ("sonnet-5+RAG", "sonnet_rag", "sonnet2"),  # pre-collapse drained rerun
    ("opus-4.8+RAG", "opus_rag", "opus_rag"),  # pre-collapse drain
]
COLLAPSED = ("sonnet-5+RAG (collapsed ckpt)", "sonnet_rag", "sonnet_rag")


def _rows(path: Path) -> dict[tuple[str, int], dict]:
    """Clean records keyed by (scenario_id, rollout_index)."""
    out: dict[tuple[str, int], dict] = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if not r["errors"]:
            out[(r["scenario_id"], r["rollout_index"])] = r
    return out


def paired_delta(arm: Path, base: dict[tuple[str, int], dict]) -> tuple[float, int]:
    """Mean per-episode reward delta vs base over the paired intersection."""
    a = _rows(arm)
    common = sorted(set(base) & set(a))
    if not common:
        raise SystemExit(
            f"no paired episodes between {arm.name} and the base row — wrong save tag?"
        )
    return sum(a[k]["reward"] - base[k]["reward"] for k in common) / len(common), len(common)


def fidelity(cell: Path) -> float:
    return float(json.loads(cell.read_text())["report"]["overall_fidelity"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(_REPO / "docs/research/fidelity_transfer_curve.png"))
    args = parser.parse_args()

    base = _rows(_ROWS / "real_tau2_base.jsonl")
    xs, ys, labels = [], [], []
    for label, cell, row in POINTS:
        xs.append(fidelity(_CELLS / f"fidelity_{cell}.json"))
        d, _n = paired_delta(_ROWS / f"real_tau2_curve_{row}.jsonl", base)
        ys.append(d)
        labels.append(label)
    cx = fidelity(_CELLS / f"fidelity_{COLLAPSED[1]}.json")
    cy, _ = paired_delta(_ROWS / f"real_tau2_curve_{COLLAPSED[2]}.jsonl", base)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    ax.axhline(0.0, color=_GRID, linewidth=1.5, zorder=1)
    ax.plot(
        xs,
        ys,
        color=_BLUE,
        linewidth=2,
        marker="o",
        markersize=7,
        zorder=3,
        label="healthy checkpoint (uniform rule: last pre-collapse drain)",
    )
    ax.scatter(
        [cx],
        [cy],
        color=_RED,
        marker="x",
        s=90,
        zorder=3,
        label="collapsed checkpoint (protocol-faithful)",
    )
    for x, y, label in zip(xs, ys, labels, strict=True):
        ax.annotate(
            label, (x, y), textcoords="offset points", xytext=(8, 8), fontsize=9, color=_INK
        )
    ax.annotate(
        COLLAPSED[0], (cx, cy), textcoords="offset points", xytext=(8, -12), fontsize=9, color=_RED
    )

    ax.set_xlabel("training-WM fidelity (D12 rubric, pinned Opus judge)", fontsize=11, color=_INK)
    ax.set_ylabel("real-env paired Δ reward vs base (n=40)", fontsize=11, color=_INK)
    ax.set_title(
        "Training-WM fidelity vs real-environment transfer (tau)",
        fontsize=15,
        color=_INK,
        fontweight="bold",
        loc="left",
        pad=14,
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.grid(axis="y", color=_GRID, linewidth=1)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")  # noqa: T201 - CLI output
    for x, y, label in zip(xs, ys, labels, strict=True):
        print(f"  {label:24} fidelity={x:.3f} pairedΔ={y:+.3f}")  # noqa: T201
    print(f"  {COLLAPSED[0]:24} fidelity={cx:.3f} pairedΔ={cy:+.3f}")  # noqa: T201


if __name__ == "__main__":
    main()
