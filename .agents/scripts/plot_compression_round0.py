"""Round-0 figures for the compression track PR: the append-stability split + its cost.

Two figures from cache/round0-results.json (the audit's 60 method x corpus rows), brand
system per AGENTS.md rule 14 (ink #0a0a0a, hairline #ececec, white surface) with the
dataviz-validated categorical steps (#0070f3 blue / #b8770a amber / #7928ca purple /
#0d9488 teal; palette validated with the dataviz skill's validator, all checks pass).

Fig 1 (the mechanism finding): mean append churn per method, colored by SELECTION-RULE
class - absolute/fixed rules are byte-stable, percentile/windowed rules churn - which is
the track's first published result.
Fig 2 (the cost of stability): token ratio vs compressor latency, one point per method,
family-colored, filled = append-only survivor, hollow = churns.

Usage: uv run python .agents/scripts/plot_compression_round0.py
Writes .agents/docs/research/compression_figures/*.png
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path.home() / "Desktop/Projects/wmh-compression-data/cache/round0-results.json"
OUT = Path(".agents/docs/research/compression_figures")

INK = "#0a0a0a"
GRID = "#ececec"
MUTED = "#6b6b6b"
BLUE, AMBER, PURPLE, TEAL = "#0070f3", "#b8770a", "#7928ca", "#0d9488"
GRAY = "#9a9a9a"

# method -> (selection-rule class, family)
META: dict[str, tuple[str, str]] = {
    "head-truncate-absolute": ("absolute", "heuristic"),
    "head-truncate-ratio": ("absolute", "heuristic"),
    "dedup-keep-first": ("absolute", "heuristic"),
    "per-turn-truncate-at-append": ("absolute", "symbolic"),
    "json-minify": ("absolute", "symbolic"),
    "selective-context-absolute": ("absolute", "heuristic"),
    "llmlingua2-fixed-threshold": ("absolute", "learned"),
    "selective-context-percentile": ("percentile", "heuristic"),
    "llmlingua2-percentile": ("percentile", "learned"),
    "rolling-observation-mask": ("percentile", "symbolic"),
    "tail-recency-window": ("percentile", "heuristic"),
    "random-removal": ("control", "control"),
}
RULE_COLOR = {"absolute": BLUE, "percentile": AMBER, "control": GRAY}
FAMILY_COLOR = {"heuristic": BLUE, "symbolic": TEAL, "learned": PURPLE, "control": GRAY}


def style(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    rows = json.loads(DATA.read_text())
    per_method: dict[str, dict[str, float]] = {}
    for name in META:
        mine = [r for r in rows if r["method"] == name]
        per_method[name] = {
            "churn": st.mean(r["churn_mean"] for r in mine),
            "ratio": st.mean(r["token_ratio"] for r in mine),
            "latency": st.mean(r["latency_s_per_10k_tok_p50"] for r in mine),
            "append_only": all(r["append_only"] for r in mine),
        }
    OUT.mkdir(parents=True, exist_ok=True)

    # Fig 1: churn per method, colored by selection-rule class.
    order = sorted(per_method, key=lambda m: per_method[m]["churn"])
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=200)
    ys = range(len(order))
    for y, name in zip(ys, order, strict=True):
        rule = META[name][0]
        churn = per_method[name]["churn"]
        ax.barh(y, max(churn, 0.004), height=0.62, color=RULE_COLOR[rule], zorder=3)
        label = f"{churn:.0%}" if churn > 0 else "0 (byte-stable)"
        ax.text(
            max(churn, 0.004) + 0.012, y, label, va="center", fontsize=8.5, color=MUTED
        )
    ax.set_yticks(list(ys), order, fontsize=9, color=INK)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel(
        "compressed-prefix churn per appended turn (share of emitted prefix that changes)",
        fontsize=9,
        color=MUTED,
    )
    style(ax)
    ax.set_title(
        "The selection rule decides cache safety, the scorer does not",
        loc="left",
        fontsize=12,
        color=INK,
        pad=14,
    )
    ax.text(
        0,
        1.015,
        "absolute/fixed rules (blue) are append-only on every corpus; per-input percentile "
        "rules (amber) churn 26-61%; random control (gray)",
        transform=ax.transAxes,
        fontsize=8.5,
        color=MUTED,
    )
    fig.tight_layout()
    fig.savefig(OUT / "round0_append_churn.png", facecolor="white")

    # Fig 2: ratio vs latency, family-colored, filled = survivor.
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=200)
    for name, stats in per_method.items():
        family = META[name][1]
        color = FAMILY_COLOR[family]
        filled = stats["append_only"]
        ax.scatter(
            stats["latency"],
            stats["ratio"],
            s=90,
            facecolors=color if filled else "white",
            edgecolors=color,
            linewidths=1.6,
            zorder=3,
        )
        short = name.replace("selective-context", "sel-ctx")
        # Hand-placed offsets where the default collides (dense right cluster + the
        # random/head-truncate pair) - step 7 of the dataviz procedure.
        offsets = {
            "llmlingua2-fixed-threshold": (-8, 12),
            "llmlingua2-percentile": (-8, -16),
            "sel-ctx-percentile": (10, 2),
            "sel-ctx-absolute": (-105, 6),
            "random-removal": (10, 6),
            "head-truncate-ratio": (10, 2),
            "tail-recency-window": (10, -16),
        }
        dx, dy = offsets.get(short, (7, 5))
        ha = "right" if short == "llmlingua2-fixed-threshold" else "left"
        ax.annotate(
            short,
            (stats["latency"], stats["ratio"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=7.8,
            color=INK,
            ha=ha,
        )
    ax.set_xscale("log")
    ax.set_xlim(right=40)
    ax.set_xlabel("compressor latency, seconds per 10k input tokens (CPU, log scale)", fontsize=9, color=MUTED)
    ax.set_ylabel("token ratio (kept / raw; lower = smaller prompt)", fontsize=9, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    style(ax)
    ax.set_title(
        "What stability costs: reduction vs compressor latency",
        loc="left",
        fontsize=12,
        color=INK,
        pad=14,
    )
    ax.text(
        0,
        1.015,
        "filled = append-only survivor, hollow = churns the cached prefix; "
        "heuristic blue, symbolic teal, learned purple, control gray",
        transform=ax.transAxes,
        fontsize=8.5,
        color=MUTED,
    )
    fig.tight_layout()
    fig.savefig(OUT / "round0_ratio_vs_latency.png", facecolor="white")
    print(f"wrote {OUT}/round0_append_churn.png and round0_ratio_vs_latency.png")


if __name__ == "__main__":
    main()
