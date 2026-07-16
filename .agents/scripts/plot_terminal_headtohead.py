#!/usr/bin/env python3
"""Plot the B2 headline figure: on-policy WM training vs offline SFT on terminal-tasks.

Reads the committed row records on this branch — the seeded sonnet-era WM rows
(.agents/docs/research/sonnet_era_wm_rows/) and the real-environment rows
(.agents/docs/research/real_terminal_eval_results/) — and draws paired per-episode
reward deltas vs the SAME base row for each channel, side by side. Both arms are the
same Qwen3.5-9B; only the training data differs (WM rollouts vs recorded
demonstrations). Chart style follows the repo brand system (AGENTS.md §visuals).

Usage (repo root):
    uv run python .agents/scripts/plot_terminal_headtohead.py \
        --out docs/research/terminal_headtohead.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[2]
_WM = _REPO / ".agents/docs/research/sonnet_era_wm_rows"
_REAL = _REPO / ".agents/docs/research/real_terminal_eval_results"

_INK = "#0a0a0a"
_GRID = "#ececec"
_BLUE = "#0070f3"
_AMBER = "#f5a623"

# channel -> (base row, [(arm label, arm row, color)])
PANELS = [
    (
        "In-WM eval (sonnet-era, seeded)",
        _WM / "wm_sonnet_term_base_seeded.jsonl",
        [
            ("WM-trained\n(R++ n=4, substrate)", _WM / "wm_sonnet_term_sub_0075.jsonl", _BLUE),
            ("offline SFT\n(228 demonstrations)", _WM / "wm_sonnet_term_sft.jsonl", _AMBER),
        ],
    ),
    (
        "Real environment (docker, Opus judge)",
        _REAL / "real_terminal2_base.jsonl",
        [
            ("WM-trained\n(R++ n=4, substrate)", _REAL / "real_terminal_termsub.jsonl", _BLUE),
            ("offline SFT\n(228 demonstrations)", _REAL / "real_terminal_termsft.jsonl", _AMBER),
        ],
    ),
]


def _rows(path: Path) -> dict[tuple[str, int], float]:
    """Clean records (no transport errors) keyed by (scenario_id, rollout_index)."""
    out: dict[tuple[str, int], float] = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if not r.get("errors"):
            out[(r["scenario_id"], r["rollout_index"])] = r["reward"]
    return out


def paired_delta(arm: Path, base: dict[tuple[str, int], float]) -> tuple[float, int]:
    """Mean per-episode reward delta vs base over the paired intersection."""
    a = _rows(arm)
    common = sorted(set(base) & set(a))
    if not common:
        raise SystemExit(f"no paired episodes between {arm.name} and its base row")
    return sum(a[k] - base[k] for k in common) / len(common), len(common)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/research/terminal_headtohead.png")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), sharey=True)
    fig.patch.set_facecolor("white")

    for ax, (title, base_path, arms) in zip(axes, PANELS, strict=True):
        base = _rows(base_path)
        base_mean = sum(base.values()) / len(base)
        ax.set_facecolor("white")
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(_GRID)
        ax.tick_params(colors=_INK, length=0)
        ax.grid(axis="y", color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.axhline(0.0, color=_INK, linewidth=1.0)

        labels, deltas, colors, ns = [], [], [], []
        for label, path, color in arms:
            delta, n = paired_delta(path, base)
            labels.append(label)
            deltas.append(delta)
            colors.append(color)
            ns.append(n)
        bars = ax.bar(labels, deltas, width=0.55, color=colors)
        for bar, delta, n in zip(bars, deltas, ns, strict=True):
            va = "bottom" if delta >= 0 else "top"
            offset = 0.006 if delta >= 0 else -0.006
            ax.annotate(
                f"{delta:+.3f}\n(n={n})",
                (bar.get_x() + bar.get_width() / 2, delta + offset),
                ha="center",
                va=va,
                fontsize=10,
                color=_INK,
            )
        ax.set_title(
            f"{title}\nbase mean reward {base_mean:.3f}",
            loc="left",
            fontsize=10,
            color=_INK,
        )

    axes[0].set_ylabel("paired per-episode reward Δ vs base", color=_INK, fontsize=10)
    axes[0].set_ylim(-0.21, 0.06)
    fig.suptitle(
        "Terminal-tasks: on-policy WM training vs offline SFT (same base Qwen3.5-9B)",
        x=0.02,
        ha="left",
        fontsize=12,
        color=_INK,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"wrote {out}")  # noqa: T201 - CLI output


if __name__ == "__main__":
    main()
