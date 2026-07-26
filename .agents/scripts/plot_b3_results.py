#!/usr/bin/env python
"""Render the two BENCH-B3 headline figures for PR #78 (brand system per AGENTS.md rule 15).

(a) kimi three-bar: base / GRPO / offline-SFT success on the haiku-kimi-era WM eval,
    with the evaluation-collusion caveat printed on the figure itself.
(b) tau GRPO in-WM vs real-env: the +10.8-pt in-WM gain that measures 0.0 on real tau2,
    with the serving-config correction trail in the caption.

Numbers are hard-coded from the committed rows (results_grpo_sdpo.md + raw JSONLs on this
branch). matplotlib is not a project dependency; run ephemerally:

    uv run --with matplotlib python .agents/scripts/plot_b3_results.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

_INK = "#0a0a0a"
_MUTED = "#8a8a8a"
_GRID = "#ececec"
_BLUE = "#0070f3"
_PURPLE = "#7928ca"

_OUT = Path(__file__).resolve().parents[2] / "packages/environment-capture/tau-bench/rl/figures"


def _clean_axes(ax) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_GRID)
    ax.tick_params(colors=_MUTED, length=0)
    ax.yaxis.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def kimi_three_bar() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    labels = ["base\nQwen3.5-9B", "GRPO\n(on-policy in the WM)", "offline SFT\n(Kimi-K2.6 demos)"]
    values = [0.330, 0.240, 0.830]
    colors = [_INK, _BLUE, _PURPLE]
    bars = ax.bar(labels, values, width=0.52, color=colors)
    for bar, v in zip(bars, values):
        ax.annotate(
            f"{v:.3f}".rstrip("0"),
            (bar.get_x() + bar.get_width() / 2, v),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=11,
            fontweight="bold",
            color=_INK,
        )
    ax.annotate(
        "paired −0.079\n(33W-43L)", (1, 0.240), textcoords="offset points", xytext=(30, 18),
        ha="left", fontsize=8.5, color=_BLUE,
    )
    ax.annotate(
        "paired +0.518\n(74W-7L)", (2, 0.830), textcoords="offset points", xytext=(-100, -6),
        ha="right", fontsize=8.5, color=_PURPLE,
    )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("success rate (WM eval)", fontsize=9, color=_MUTED)
    _clean_axes(ax)
    fig.text(
        0.055, 0.945,
        "kimi (gui-tasks): offline SFT dominates; GRPO is negative",
        fontsize=12.5, fontweight="bold", color=_INK,
    )
    fig.text(
        0.055, 0.895,
        "haiku-kimi-era WM eval · pinned D73 protocol (100 scenarios × 1 ep) — D90.6 rows of record",
        fontsize=8.5, color=_MUTED,
    )
    fig.text(
        0.055, 0.022,
        "Caveat — evaluation collusion: the eval WM is built from the same Kimi-K2.6 demonstrations the SFT\n"
        "policy imitates; in-WM eval likely flatters imitation. No real-env harness exists for kimi to resolve it.",
        fontsize=8, color=_MUTED, style="italic",
    )
    fig.subplots_adjust(left=0.095, right=0.97, top=0.84, bottom=0.22)
    fig.savefig(_OUT / "b3_kimi_head_to_head.png", facecolor="white")
    plt.close(fig)


def tau_wm_vs_real() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    x = [0, 1, 2.4, 3.4]
    values = [0.550, 0.658, 0.900, 0.900]
    colors = [_INK, _BLUE, _INK, _BLUE]
    bars = ax.bar(x, values, width=0.72, color=colors)
    for bar, v in zip(bars, values):
        ax.annotate(
            f"{v:.3f}".rstrip("0"),
            (bar.get_x() + bar.get_width() / 2, v),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=11,
            fontweight="bold",
            color=_INK,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["base", "GRPO", "base", "GRPO"], fontsize=10, color=_INK)
    ax.annotate("+10.8 pts", (0.5, 0.73), ha="center", fontsize=10, fontweight="bold", color=_BLUE)
    ax.annotate("+0.0 pts (paired, 2W-2L)", (2.9, 1.0), ha="center", fontsize=10, fontweight="bold", color=_BLUE)
    ax.text(0.5, -0.165, "in-WM eval\n(GPT-5.5-era WM, n=79)", ha="center", fontsize=9, color=_MUTED)
    ax.text(2.9, -0.165, "real tau2 gym\n(pinned scenarios, n=40)", ha="center", fontsize=9, color=_MUTED)
    ax.set_ylim(0, 1.09)
    ax.set_ylabel("success rate", fontsize=9, color=_MUTED)
    _clean_axes(ax)
    ax.tick_params(axis="x", pad=2)
    fig.text(
        0.055, 0.945,
        "tau-bench: the in-WM GRPO gain is real-env-neutral",
        fontsize=12.5, fontweight="bold", color=_INK,
    )
    fig.text(
        0.055, 0.895,
        "same checkpoint, same 20 pinned scenarios — no transfer, no harm",
        fontsize=8.5, color=_MUTED,
    )
    fig.text(
        0.055, 0.022,
        "Correction trail: a first-pass real row read 0.600 — entirely a serving-config artifact (--reasoning-parser\n"
        "violates the pinned think-in-content harness config, D70; one flag = 30 pts). Row of record: pinned config, 0 errors.",
        fontsize=8, color=_MUTED, style="italic",
    )
    fig.subplots_adjust(left=0.095, right=0.97, top=0.84, bottom=0.26)
    fig.savefig(_OUT / "b3_tau_wm_vs_real.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    _OUT.mkdir(parents=True, exist_ok=True)
    kimi_three_bar()
    tau_wm_vs_real()
    print(f"wrote {_OUT}/b3_kimi_head_to_head.png and b3_tau_wm_vs_real.png")  # noqa: T201
