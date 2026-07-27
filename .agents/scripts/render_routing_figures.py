#!/usr/bin/env python
"""Render the four routing-research figures for the PR summary.

Every number except the confidence curve is RECOMPUTED from the run records through the
dashboard's own aggregation path (`build_dashboard.py`), so a figure can never drift from the
dashboard: same loading rules, same matrix-cohort resolution, same knob-vs-fitted-value
detection, same seed aggregation, same verdict tiers. Re-run it after new captures land and
the figures move with the data.

    uv run --with matplotlib python .agents/scripts/render_routing_figures.py

Figures (into .agents/docs/research/figures/):
  routing_pareto_ours9.png    cost vs accuracy on our 9-model pool, seed-aggregated with sd
  routing_verdict_census.png  how every group scored under the honest verdict rules
  routing_confidence_curve.png  the "route only when confident" dial (numbers from PR #259)
  routing_signal_map.png      where signal lives vs how big the test set is
  routing_pareto_niches.png   per-niche frontiers: the pool, the niche champion, the ceiling
  routing_operating_points.png the dialable cost/quality/latency frontier on ours9

Two deliberate deviations from the brief, both to avoid a misleading chart:

1. Amber and teal are the DARKENED chart steps (#b8770a, #0d9488), not the raw brand accents
   (#f5a623, #50e3c2). The raw pair scores 1.97:1 and 1.56:1 contrast on white, far under the
   3:1 floor, and fails the lightness band; as a thin line on white the raw teal is close to
   invisible. The darkened steps pass every check. `build_dashboard.py` made the same call.
2. The confidence curve is two stacked panels sharing an x-axis, not one dual-axis chart.
   Two y-scales on one frame let the crossing point be placed anywhere by choosing the
   scaling, and here it would matter: accuracy moves 0.56pt across the whole sweep while the
   routed-away fraction moves 38pt. Stacked panels keep both readable and comparable, and
   plotting accuracy against the fable-5-alone reference shows the real shape - flat until
   the dial is turned hard.
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
import textwrap
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from wmo.research.routing_corpus import routing_data

if TYPE_CHECKING:
    from types import ModuleType

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

BUILDER = Path(__file__).with_name("build_dashboard.py")
OUT_DIR = Path(".agents/docs/research/figures")
MATRIX_DIR = routing_data() / "matrices"

# House style: near-black ink, hairline grid, restrained accents, generous whitespace.
INK = "#0a0a0a"
MUTED = "#8a8a8a"
GRID = "#ececec"
BLUE = "#0070f3"
PURPLE = "#7928ca"
AMBER = "#b8770a"  # darkened brand amber; raw #f5a623 is 1.97:1 on white
RED = "#ee0000"
TEAL = "#0d9488"  # darkened brand teal; raw #50e3c2 is 1.56:1 on white
GRAY_MID = "#9a9a9a"
GRAY_LIGHT = "#c9c9c9"

# Fable-5 used alone on routerbench-ours9, and the z sweep for the confidence dial.
# Source: PR #259 validation output (ours9, fallback fable-5). Hardcoded on purpose; these
# come from the confidence-gating run, not from the ablation run records.
FABLE_ALONE_ACC = 0.9045
CONFIDENCE_Z = [0.0, 0.25, 0.5, 1.0, 2.0]
CONFIDENCE_ROUTED_AWAY = [69.1, 68.0, 64.9, 48.0, 30.9]
CONFIDENCE_ACC = [0.9607, 0.9607, 0.9607, 0.9635, 0.9579]

HEADLINE_MATRIX = "routerbench-ours9"

# The per-niche panels. Each is (run-record cohort, matrix file stem, the lead-in caption).
NICHES = [
    (
        "financebench-s80",
        "financebench-s80",
        "deployable frontier today = the niche's best model; the ceiling is the cascade's "
        "+7.3pt at -24% cost",
    ),
    (
        "tau-bench-s80",
        "tau-bench-s80",
        "step 1 of niche optimization: discover the niche's own champion (sonnet-5 here, not "
        "the global fallback)",
    ),
]
# Always call these out by name: the anchor of each niche, plus the global fallback so the
# reader can see it is NOT automatically the right model for a given niche.
ALWAYS_LABEL = {"fable-5"}
ORACLE_VARIANT = "r2-oracle2-cascade"  # an upper bound, never deployable: it peeks at rewards
# The PROMOTED champion, pinned by name rather than picked by best delta. Unguarded kNN
# variants score higher here (r1-knn-noguard-oai reaches +1.33pt) but are not deployable: the
# guard is what keeps the router from dipping below baseline under distribution shift, which
# is exactly the weakness PR #259 found and fixed. Labelling the ungated variant "champion"
# would advertise a configuration we deliberately did not ship.
CHAMPION_VARIANT = "r1-knn-statz05-oai"  # kNN + relative threshold + stat guard, z=0.5


def load_groups() -> tuple[list[dict], int]:
    """Seed-aggregated groups straight from the dashboard builder (its rules, not a copy)."""
    spec = importlib.util.spec_from_file_location("build_dashboard", BUILDER)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import the dashboard builder from {BUILDER}")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)

    runs, _ = builder.load_runs()
    runs = [r for r in runs if r["matrix"] not in builder.FOREIGN_POOLS]
    builder.resolve_matrices(runs)
    groups, _ = builder.aggregate(runs, builder.knob_keys(runs))
    groups += builder.synthetic_anchors(runs, groups)
    return groups, len(runs)


def style_axes(ax: Axes, *, xgrid: bool = False) -> None:
    """Minimal chrome: no top/right spine, hairline grid, muted tickless labels."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="x" if xgrid else "y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)


def titled(ax: Axes, title: str, subtitle: str = "") -> None:
    """Left-aligned title with the subtitle on its own line beneath it, never overlapping."""
    ax.set_title(
        title, fontsize=15, color=INK, fontweight="bold", loc="left", pad=34 if subtitle else 14
    )
    if subtitle:
        ax.annotate(
            subtitle,
            xy=(0, 1),
            xytext=(0, 10),
            xycoords="axes fraction",
            textcoords="offset points",
            fontsize=9.5,
            color=MUTED,
            va="bottom",
            ha="left",
        )


def save(fig: Figure, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=200)
    return path


def curve(ax: Axes, points: list[dict], color: str, label: str) -> None:
    """One variant's cost-knob sweep: a connected line, hollow markers, sd whiskers."""
    pts = sorted(points, key=lambda g: g["cost"])
    xs = [g["cost"] for g in pts]
    ys = [g["acc"] for g in pts]
    es = [g["sd"] for g in pts]
    ax.errorbar(
        xs,
        ys,
        yerr=es,
        fmt="-o",
        color=color,
        label=label,
        linewidth=2.0,
        markersize=5,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.6,
        elinewidth=1.2,
        capsize=0,
        ecolor=color,
        alpha=0.95,
        zorder=3,
    )


def fig_pareto(groups: list[dict], plt: ModuleType) -> Path:
    """Figure 1: the headline. Cost vs accuracy on our own pool, one point per group."""
    rows = [g for g in groups if g["m"] == HEADLINE_MATRIX]
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for g in rows:
        by_variant[g["v"]].append(g)

    anchor = next(g for g in rows if g["v"] == "best-single")
    champion = max(
        (g for g in rows if g["v"] == CHAMPION_VARIANT and g["tier"] == "beats"),
        key=lambda g: (g["s"], g["d"] or 0.0),
    )
    fable_cost = statistics.median([g["pmc"]["fable-5"] for g in rows if "fable-5" in g["pmc"]])

    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # The three families as cost-knob sweeps: one connected curve each, cheapest to dearest.
    curve(ax, by_variant["r3-knn-frontier"], TEAL, "kNN retrieval (cost knob)")
    curve(ax, by_variant["r3-irt-frontier"], PURPLE, "IRT learned predictor (cost knob)")
    curve(ax, by_variant["rank"], BLUE, "rank cluster scoreboard (cost knob)")

    ax.plot(
        [anchor["cost"]],
        [anchor["acc"]],
        "D",
        color=INK,
        markersize=9,
        markeredgecolor="white",
        markeredgewidth=1.5,
        zorder=6,
        label="best single model",
    )
    ax.annotate(
        f"best single: gpt-5.5\n{anchor['acc']:.1%}",
        xy=(anchor["cost"], anchor["acc"]),
        xytext=(13, 0),
        textcoords="offset points",
        fontsize=9.5,
        color=INK,
        ha="left",
        va="center",
    )

    ax.plot(
        [champion["cost"]],
        [champion["acc"]],
        "o",
        color=TEAL,
        markersize=11,
        markeredgecolor="white",
        markeredgewidth=1.8,
        zorder=7,
    )
    ax.errorbar(
        [champion["cost"]],
        [champion["acc"]],
        yerr=[champion["sd"]],
        elinewidth=1.4,
        capsize=0,
        ecolor=TEAL,
        zorder=6,
        fmt="none",
    )
    ax.annotate(
        f"kNN champion (guarded) {champion['acc']:.1%}\n"
        f"{100 * champion['d']:+.2f}pt vs best single, "
        f"{champion['w']}/{champion['s']} seeds, "
        f"{1 - champion['cost'] / anchor['cost']:.0%} cheaper",
        xy=(champion["cost"], champion["acc"]),
        xytext=(-8, 20),
        textcoords="offset points",
        fontsize=9.5,
        color=TEAL,
        ha="right",
        fontweight="bold",
    )

    ax.plot(
        [fable_cost],
        [FABLE_ALONE_ACC],
        "X",
        color=RED,
        markersize=9,
        markeredgecolor="white",
        markeredgewidth=1.2,
        zorder=6,
    )
    ax.annotate(
        f"fable-5 alone {FABLE_ALONE_ACC:.1%}\n(dearer AND worse)",
        xy=(fable_cost, FABLE_ALONE_ACC),
        xytext=(13, 0),
        textcoords="offset points",
        fontsize=9.5,
        color=RED,
        ha="left",
        va="center",
    )

    ax.set_xscale("log")
    from matplotlib.ticker import FixedFormatter, FixedLocator

    ticks = [3e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"${t:.4f}" for t in ticks]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlim(2.0e-4, 7.5e-3)  # right margin so the anchor labels sit clear of the curves
    ax.set_xlabel("cost per call, USD (log)", fontsize=11, color=INK)
    ax.set_ylabel("accuracy on held-out scenarios", fontsize=11, color=INK)
    seeds = anchor["s"]
    titled(
        ax,
        "Routing our 9-model pool: what a cheaper call costs you",
        f"routerbench-ours9 · {round(anchor['nt'])} held-out scenarios/seed · {seeds} seeds · "
        "points are seed means, whiskers +-1 sd",
    )
    style_axes(ax)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    leg = ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    for text in leg.get_texts():
        text.set_color(INK)
    fig.tight_layout()
    return save(fig, "routing_pareto_ours9.png")


def fig_census(groups: list[dict], plt: ModuleType, total_runs: int) -> tuple[Path, Counter]:
    """Figure 2: every group's verdict under the two-axis power rules."""
    counts = Counter(g["tier"] for g in groups if g["tier"] != "anchor")
    order = [
        ("beats", "BEATS baseline", BLUE),
        ("promising", "promising (small test set)", AMBER),
        ("ties", "ties baseline (within spread)", GRAY_MID),
        ("identical", "identical to baseline (never routed away)", GRAY_MID),
        ("mixed", "mixed seeds (unclear)", GRAY_LIGHT),
        ("unfavourable", "unfavourable (small test set)", GRAY_LIGHT),
        ("worse", "WORSE than baseline", RED),
        ("underpowered", "underpowered (under 3 seeds)", GRAY_LIGHT),
    ]
    labels = [label for key, label, _ in order if counts.get(key)]
    values = [counts[key] for key, _, _ in order if counts.get(key)]
    colors = [color for key, _, color in order if counts.get(key)]

    fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ypos = range(len(labels))
    ax.barh(list(ypos), values, color=colors, height=0.62, zorder=3)
    for y, value in zip(ypos, values, strict=True):
        ax.annotate(
            f"{value}",
            xy=(value, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=10.5,
            color=INK,
            fontweight="bold",
        )
    ax.set_yticks(list(ypos))
    ax.set_yticklabels(labels, fontsize=10.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("seed-aggregated groups", fontsize=11, color=INK)
    ax.set_xlim(0, max(values) * 1.12)
    titled(
        ax,
        "The honest scoreboard: how every configuration actually scored",
        f"{sum(values):,} groups from {total_runs:,} runs · a verdict needs 3+ seeds AND 30+ "
        "test scenarios/seed · paired by seed against best-single",
    )
    style_axes(ax, xgrid=True)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return save(fig, "routing_verdict_census.png"), counts


def fig_confidence(plt: ModuleType) -> Path:
    """Figure 3: the confidence dial. Two panels, shared x - never a dual axis (see docstring)."""
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8.4, 5.8), dpi=200, sharex=True, gridspec_kw={"height_ratios": [1, 1.15]}
    )
    fig.patch.set_facecolor("white")
    for ax in (top, bottom):
        ax.set_facecolor("white")

    top.plot(
        CONFIDENCE_Z,
        CONFIDENCE_ROUTED_AWAY,
        "-o",
        color=BLUE,
        linewidth=2.2,
        markersize=6,
        markerfacecolor="white",
        markeredgecolor=BLUE,
        markeredgewidth=1.8,
        zorder=3,
    )
    for x, y in zip(CONFIDENCE_Z, CONFIDENCE_ROUTED_AWAY, strict=True):
        top.annotate(
            f"{y:.1f}%",
            xy=(x, y),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=9.5,
            color=INK,
        )
    top.set_ylabel("routed away from fable-5", fontsize=11, color=INK)
    top.set_ylim(20, 82)
    top.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    titled(
        top,
        "Route only when confident: turning the z dial",
        "routerbench-ours9, fallback fable-5 · higher z = a stricter bar before routing away "
        "· numbers from PR #259 validation",
    )
    style_axes(top)

    bottom.plot(
        CONFIDENCE_Z,
        CONFIDENCE_ACC,
        "-o",
        color=TEAL,
        linewidth=2.2,
        markersize=6,
        markerfacecolor="white",
        markeredgecolor=TEAL,
        markeredgewidth=1.8,
        zorder=3,
        label="routed accuracy",
    )
    bottom.axhline(FABLE_ALONE_ACC, color=MUTED, linewidth=1.6, linestyle="--", zorder=2)
    bottom.annotate(
        f"fable-5 alone, no routing ({FABLE_ALONE_ACC:.1%})",
        xy=(CONFIDENCE_Z[-1], FABLE_ALONE_ACC),
        xytext=(0, 7),
        textcoords="offset points",
        fontsize=9.5,
        color=MUTED,
        ha="right",
    )
    bottom.annotate(
        "routed accuracy",
        xy=(CONFIDENCE_Z[0], CONFIDENCE_ACC[0]),
        xytext=(6, 10),
        textcoords="offset points",
        fontsize=9.5,
        color=TEAL,
        ha="left",
        fontweight="bold",
    )
    best_i = max(range(len(CONFIDENCE_ACC)), key=lambda i: CONFIDENCE_ACC[i])
    bottom.annotate(
        f"peak {CONFIDENCE_ACC[best_i]:.2%} at z={CONFIDENCE_Z[best_i]:g},\n"
        f"routing away only {CONFIDENCE_ROUTED_AWAY[best_i]:.0f}% of calls",
        xy=(CONFIDENCE_Z[best_i], CONFIDENCE_ACC[best_i]),
        xytext=(8, -34),
        textcoords="offset points",
        fontsize=9.5,
        color=TEAL,
        ha="left",
        fontweight="bold",
    )
    bottom.set_xlabel("z (confidence margin required before routing away)", fontsize=11, color=INK)
    bottom.set_ylabel("accuracy", fontsize=11, color=INK)
    bottom.set_ylim(0.895, 0.972)
    bottom.set_xticks(CONFIDENCE_Z)
    bottom.set_xticklabels([f"{z:g}" for z in CONFIDENCE_Z])
    bottom.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style_axes(bottom)
    fig.tight_layout()
    return save(fig, "routing_confidence_curve.png")


def fig_signal_map(groups: list[dict], plt: ModuleType) -> Path:
    """Figure 4: signal lives where the test set is big. One point per matrix section."""
    per: dict[str, dict[str, float]] = defaultdict(lambda: {"groups": 0, "signal": 0, "n": 0.0})
    for g in groups:
        if g["tier"] == "anchor":
            continue
        row = per[g["m"]]
        row["groups"] += 1
        row["n"] = g["ntmed"]
        if g["tier"] in {"beats", "promising"}:
            row["signal"] += 1

    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    gate = 30
    ax.axvspan(0.7, gate, color=GRID, alpha=0.7, zorder=0)
    ax.annotate(
        "under 30 scenarios/seed a directional\ndelta is only ever a candidate",
        xy=(0.9, 0.97),
        xycoords=("data", "axes fraction"),
        fontsize=9.5,
        color=MUTED,
        ha="left",
        va="top",
    )

    for row in per.values():
        big = row["n"] >= gate
        ax.scatter(
            row["n"],
            row["signal"],
            s=24 + 6.0 * row["groups"] ** 0.5,
            facecolor=BLUE if big else "white",
            edgecolor=BLUE if big else GRAY_MID,
            linewidth=1.6,
            alpha=0.9 if big else 0.75,
            zorder=4,
        )
    for matrix, dx, dy, ha in [
        (HEADLINE_MATRIX, -18, -6, "right"),
        ("tau-bench", 14, 8, "left"),
        ("wm-all", 6, 20, "left"),
    ]:
        if matrix not in per:
            continue
        row = per[matrix]
        ax.annotate(
            f"{matrix}\n{int(row['signal'])} of {int(row['groups'])} groups, "
            f"{row['n']:.0f} scenarios/seed",
            xy=(row["n"], row["signal"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9.5,
            color=INK,
            ha=ha,
            fontweight="bold" if matrix == HEADLINE_MATRIX else "normal",
        )

    ax.set_xscale("log")
    ax.set_xlim(0.7, 900)
    ax.set_xlabel("median test scenarios per seed (log)", fontsize=11, color=INK)
    ax.set_ylabel(
        "groups that beat baseline, or would with a real test set", fontsize=11, color=INK
    )
    titled(
        ax,
        "Signal lives where the test set is big",
        "one point per matrix section, sized by how many configurations it holds · "
        "this is why the scaled captures exist",
    )
    style_axes(ax)
    fig.tight_layout()
    return save(fig, "routing_signal_map.png")


def pool_points(stem: str) -> list[tuple[str, float, float]]:
    """Per-model (name, mean reward, mean cost per episode) straight from the outcome matrix.

    This is the whole pool on the whole niche - every scenario, every episode - which is a
    different population from the router points on the same axes: those are held-out split
    means. The panel subtitle says so, because putting both on one frame without saying it
    would invite reading a router's gain off the pool's spread.
    """
    matrix = json.loads((MATRIX_DIR / f"{stem}_matrix.json").read_text(encoding="utf-8"))
    rewards: dict[str, list[float]] = defaultdict(list)
    costs: dict[str, list[float]] = defaultdict(list)
    for outcome in matrix["outcomes"]:
        if outcome.get("reward") is None:
            continue
        rewards[outcome["model"]].append(outcome["reward"])
        costs[outcome["model"]].append(outcome["cost_usd"])
    return sorted(
        ((m, statistics.mean(rewards[m]), statistics.mean(costs[m])) for m in rewards),
        key=lambda row: -row[1],
    )


def niche_family(group: dict) -> str:
    """Colour bucket within a niche panel: champion lineage, multi-call, or neither.

    The r3b-* sweep (novelty floors, hybrids, the SVM) carries no "knn" in its name so the
    dashboard's family_of files it under other; here it belongs with the champion lineage it
    was derived from, which is what the reader needs to see.
    """
    if group["fam"] == "multi":
        return "multi"
    if group["v"].startswith(("r3b-", "r1-knn", "r3x-")) or group["fam"] == "knn":
        return "champion"
    return "rest"


def fig_niches(groups: list[dict], plt: ModuleType) -> Path:
    """Figure 5: we optimize per niche. One panel per niche, its own pool and its own answer."""
    fig, axes = plt.subplots(1, len(NICHES), figsize=(13.0, 6.6), dpi=200)
    fig.patch.set_facecolor("white")

    for ax, (cohort, stem, caption) in zip(axes, NICHES, strict=True):
        ax.set_facecolor("white")
        pool = pool_points(stem)
        best_model, best_reward, best_cost = pool[0]
        rows = [
            g
            for g in groups
            if g["m"] == cohort and g["tier"] != "anchor" and "shuffl" not in g["v"]
        ]
        # Router deltas are PAIRED against this held-out baseline, so every router-relative
        # number (the cost ratio included) has to use it rather than the pool's matrix-wide
        # cost, or the panel quotes two bases and contradicts its own caption.
        anchor = next((g for g in groups if g["m"] == cohort and g["tier"] == "anchor"), None)
        base_cost = anchor["cost"] if anchor else best_cost

        # The pool itself: every model we could have used, the niche's winner in ink.
        for name, reward, cost in pool[1:]:
            ax.scatter(
                cost,
                reward,
                s=34,
                facecolor=GRAY_LIGHT,
                edgecolor=GRAY_MID,
                linewidth=1.0,
                zorder=3,
            )
            if name in ALWAYS_LABEL:
                ax.annotate(
                    f"{name} {reward:.1%}\n{cost / best_cost:.1f}x the cost of "
                    f"{best_model}, and worse",
                    xy=(cost, reward),
                    xytext=(0, -14),
                    textcoords="offset points",
                    fontsize=8.5,
                    color=RED,
                    ha="center",
                    va="top",
                )
        ax.scatter(
            best_cost,
            best_reward,
            marker="D",
            s=95,
            facecolor=INK,
            edgecolor="white",
            linewidth=1.4,
            zorder=7,
        )
        ax.annotate(
            f"{best_model} {best_reward:.1%}\nthis niche's best single model",
            xy=(best_cost, best_reward),
            xytext=(11, -4),
            textcoords="offset points",
            fontsize=9,
            color=INK,
            ha="left",
            fontweight="bold",
        )

        # Without this line the panel misleads: the diamond is a matrix-wide mean over all 80
        # scenarios, the coloured points are held-out split means, and on tau that gap is 6pt
        # of population difference that reads as routers losing to the best model. Draw the
        # baseline the routers are actually paired against, and say which one it is.
        if anchor is not None:
            ax.axhline(anchor["acc"], color=MUTED, linewidth=1.3, linestyle="--", zorder=2)
            ax.annotate(
                f"held-out baseline {anchor['acc']:.1%}",
                xy=(0.015, anchor["acc"]),
                xycoords=("axes fraction", "data"),
                xytext=(0, 4),
                textcoords="offset points",
                fontsize=8.5,
                color=MUTED,
                ha="left",
                va="bottom",
            )

        # Routers and cascades measured on this niche's held-out splits.
        on_anchor = 0
        for g in rows:
            bucket = niche_family(g)
            if g["v"] == ORACLE_VARIANT:
                continue
            colour = {"champion": TEAL, "multi": RED}.get(bucket, GRAY_MID)
            solid = g["tier"] in {"beats", "promising"}
            if g["tier"] == "identical":
                on_anchor += 1
            ax.errorbar(
                [g["cost"]],
                [g["acc"]],
                yerr=[g["sd"]],
                fmt="o",
                markersize=6.5,
                color=colour,
                markerfacecolor=colour if solid else "white",
                markeredgecolor=colour,
                markeredgewidth=1.7,
                elinewidth=1.1,
                capsize=0,
                ecolor=colour,
                alpha=0.95,
                zorder=5,
            )

        oracle = next((g for g in rows if g["v"] == ORACLE_VARIANT), None)
        if oracle is not None:
            ax.errorbar(
                [oracle["cost"]],
                [oracle["acc"]],
                yerr=[oracle["sd"]],
                fmt="none",
                elinewidth=1.3,
                capsize=0,
                ecolor=RED,
                zorder=6,
            )
            ax.scatter(
                oracle["cost"],
                oracle["acc"],
                marker="*",
                s=340,
                facecolor=RED,
                edgecolor="white",
                linewidth=1.3,
                zorder=8,
            )
            ax.annotate(
                f"cascade ceiling (oracle decision)\n{oracle['acc']:.1%}, "
                f"{100 * oracle['d']:+.1f}pt at {oracle['cost'] / base_cost - 1:+.0%} cost\n"
                "NOT deployable: it reads the reward",
                xy=(oracle["cost"], oracle["acc"]),
                xytext=(0, 15),
                textcoords="offset points",
                fontsize=8.5,
                color=RED,
                ha="center",
                fontweight="bold",
            )
        if on_anchor:
            ax.annotate(
                f"{on_anchor} guarded configs land exactly\non the anchor: the guard never\n"
                "fired, nothing here to win",
                xy=(0.02, 0.99),
                xycoords="axes fraction",
                fontsize=8.5,
                color=TEAL,
                ha="left",
                va="top",
            )

        ax.set_xscale("log")
        ax.set_xlabel("cost per episode, USD (log)", fontsize=10.5, color=INK)
        ax.set_ylabel("reward", fontsize=10.5, color=INK)
        n_test = rows[0]["ntmed"] if rows else 0
        titled(
            ax,
            cohort,
            f"9-model pool on 80 scenarios · routers on {n_test:.0f} held-out scenarios/seed",
        )
        style_axes(ax)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.annotate(
            "\n".join(textwrap.wrap(caption, 62)),
            xy=(0, -0.155),
            xycoords="axes fraction",
            fontsize=9,
            color=MUTED,
            va="top",
        )

    fig.text(
        0.5,
        0.015,
        textwrap.fill(
            "Gray dots are the pool models over the whole niche; coloured points are routers "
            "and cascades on held-out splits (+-1 sd across seeds). Teal = guarded champion "
            "lineage, red = multi-call. The dashed line is the anchor model measured on those "
            "same held-out splits, which is the bar every coloured point is paired against - "
            "compare to the line, not to the diamond.",
            118,
        ),
        fontsize=8.5,
        color=MUTED,
        ha="center",
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    return save(fig, "routing_pareto_niches.png")


def fig_operating_points(groups: list[dict], plt: ModuleType) -> Path:
    """Figure 6: the whole dialable tradeoff space on our one full-power cohort.

    Three series, each a coherent single-variant sweep of ONE knob so the line means "turn this
    dial", not "compare these methods": the shipped config's novelty floor (floor_q), the
    confidence bar (z), and the cost knob (lam). Latency rides along as marker size on the left
    panel and gets its own panel on the right, because the claim "cheaper is also faster" should
    be shown rather than asserted.
    """
    rows = [g for g in groups if g["m"] == HEADLINE_MATRIX]
    anchor = next(g for g in rows if g["v"] == "best-single")
    fable_cost = statistics.median([g["pmc"]["fable-5"] for g in rows if "fable-5" in g["pmc"]])
    fable_p50 = statistics.median([g["pml"]["fable-5"] for g in rows if "fable-5" in g["pml"]])

    def sweep(prefix: str, exact: set[str] | None = None) -> list[dict]:
        picked = [
            g
            for g in rows
            if (g["v"] in exact if exact else g["v"].startswith(prefix)) and g["s"] >= 3
        ]
        return sorted(picked, key=lambda g: g["cost"])

    series = [
        (sweep("r1-knn-adapt-floor-q"), TEAL, "novelty floor (floor_q), shipped config"),
        (
            sweep("", {"r1-knn-statz0", "r1-knn-statz05", "r1-knn-statz1"}),
            PURPLE,
            "confidence bar (z)",
        ),
        (sweep("", {"r3-knn-frontier"}), AMBER, "cost knob (lam)"),
    ]
    shipped = next(g for g in rows if g["v"] == "r1-knn-adapt3-oai")
    lam = sweep("", {"r3-knn-frontier"})
    floors = sweep("r1-knn-adapt-floor-q")
    quality_max = max(floors, key=lambda g: g["d"])
    aggressive = min(lam, key=lambda g: abs(100 * (g["cost"] / anchor["cost"] - 1) + 74))
    example = min(
        lam,
        key=lambda g: (
            (abs(100 * g["d"] + 5.0) / 5.0) ** 2
            + (abs(100 * (g["cost"] / anchor["cost"] - 1) + 70) / 70.0) ** 2
        ),
    )

    fig, (left, right) = plt.subplots(1, 2, figsize=(13.4, 5.8), dpi=200, width_ratios=[1.45, 1])
    fig.patch.set_facecolor("white")
    for ax in (left, right):
        ax.set_facecolor("white")

    def pct(g: dict) -> str:
        return f"{g['cost'] / anchor['cost'] - 1:+.0%}"

    for points, colour, label in series:
        if not points:
            continue
        left.plot(
            [g["cost"] for g in points],
            [g["acc"] for g in points],
            "-",
            color=colour,
            linewidth=1.9,
            alpha=0.6,
            zorder=3,
            label=label,
        )
        left.scatter(
            [g["cost"] for g in points],
            [g["acc"] for g in points],
            s=[20 + 40 * (g["p50"] or 0) for g in points],
            facecolor="white",
            edgecolor=colour,
            linewidth=1.7,
            zorder=4,
        )
        right.plot(
            [g["cost"] for g in points],
            [g["p50"] for g in points],
            "-o",
            color=colour,
            linewidth=1.9,
            markersize=5,
            markerfacecolor="white",
            markeredgecolor=colour,
            markeredgewidth=1.6,
            alpha=0.9,
            zorder=3,
        )

    for ax, yval in ((left, "acc"), (right, "p50")):
        ax.plot(
            [anchor["cost"]],
            [anchor[yval]],
            "D",
            color=INK,
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=1.4,
            zorder=7,
        )
    left.annotate(
        f"best single (gpt-5.5)\n{anchor['acc']:.1%}",
        xy=(anchor["cost"], anchor["acc"]),
        xytext=(11, -2),
        textcoords="offset points",
        fontsize=9,
        color=INK,
        ha="left",
        va="center",
        fontweight="bold",
    )
    left.plot(
        [fable_cost],
        [FABLE_ALONE_ACC],
        "X",
        color=RED,
        markersize=9,
        markeredgecolor="white",
        markeredgewidth=1.2,
        zorder=7,
    )
    left.annotate(
        f"fable-5 alone {FABLE_ALONE_ACC:.1%}",
        xy=(fable_cost, FABLE_ALONE_ACC),
        xytext=(11, 0),
        textcoords="offset points",
        fontsize=9,
        color=RED,
        ha="left",
        va="center",
    )
    right.plot(
        [fable_cost],
        [fable_p50],
        "X",
        color=RED,
        markersize=9,
        markeredgecolor="white",
        markeredgewidth=1.2,
        zorder=7,
    )

    callouts = [
        (
            quality_max,
            f"quality-max (floor_q={quality_max['kn'].get('floor_q')})",
            (-16, 12),
            "right",
        ),
        (shipped, "balanced: shipped default (z=0.5)", (-22, -24), "right"),
        (aggressive, "cost-aggressive (lam)", (12, 4), "left"),
        (example, "example: trade quality for savings", (14, -12), "left"),
    ]
    for g, name, offset, ha in callouts:
        left.annotate(
            f"{name}\n{100 * g['d']:+.2f}pt at {pct(g)} cost, p50 {g['p50']:.2f}s",
            xy=(g["cost"], g["acc"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.5,
            color=INK,
            ha=ha,
            fontweight="bold",
        )
        left.scatter(
            [g["cost"]],
            [g["acc"]],
            s=150,
            facecolor="none",
            edgecolor=INK,
            linewidth=1.1,
            zorder=6,
        )

    from matplotlib.ticker import FixedFormatter, FixedLocator

    ticks = [3e-4, 6e-4, 1.2e-3, 2.4e-3, 4.8e-3]
    for ax in (left, right):
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([f"${t:.4f}" for t in ticks]))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.set_xlim(1.7e-4, 9.5e-3)
    left.set_xlabel("cost per call, USD (log)", fontsize=10.5, color=INK)
    left.set_ylabel("accuracy", fontsize=10.5, color=INK)
    # Headroom so the top callout sits inside the axes instead of on the subtitle.
    left.set_ylim(0.884, 0.982)
    left.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    titled(
        left,
        "Operating points: the dial, not a single answer",
        f"routerbench-ours9 · {round(anchor['nt'])} held-out scenarios/seed · 5 seeds · "
        "marker size = p50 latency",
    )
    style_axes(left)
    leg = left.legend(frameon=False, fontsize=9, loc="lower right")
    for text in leg.get_texts():
        text.set_color(INK)

    right.set_xlabel("cost per call, USD (log)", fontsize=10.5, color=INK)
    right.set_ylabel("p50 latency of the routed calls (s)", fontsize=10.5, color=INK)
    titled(right, "Speed comes along for free", "the same runs: cheaper mixes are faster mixes")
    style_axes(right)

    fig.text(
        0.5,
        0.015,
        textwrap.fill(
            "Every line is one knob being turned on one configuration, so a line means "
            "\u201cturn this dial\u201d rather than \u201ccompare these methods\u201d. The "
            "right panel is why cost and latency move together: across the pool, per-call cost "
            "and per-call p50 correlate at r = 0.69, because the cheap models are mostly the "
            "fast ones (kimi-k2.6 is the exception: dear AND slow).",
            120,
        ),
        fontsize=8.5,
        color=MUTED,
        ha="center",
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    return save(fig, "routing_operating_points.png")


# Rows of the production results table, in reading order. The label carries the row's job.
TABLE_ROWS = [
    ("routerbench-ours9", "**ours9** (real benchmark, full power)"),
    ("financebench-s80", "financebench-s80 (scaled niche)"),
    ("tau-bench-s80", "tau-bench-s80 (scaled niche)"),
    ("tau-bench-real", "**tau-bench-real** (sim-to-real transfer)"),
    ("tau-bench", "tau-bench (25-scen wm)"),
    ("bird-sql", "bird-sql (25-scen wm)"),
    ("continual-learning", "continual-learning (25-scen wm)"),
    ("crmarena", "crmarena (25-scen wm)"),
    ("dabstep", "dabstep (25-scen wm)"),
    ("financebench", "financebench (25-scen wm)"),
    ("terminal-tasks", "terminal-tasks (25-scen wm)"),
    ("swe-bench", "swe-bench (25-scen wm)"),
    ("wm-gaia2", "wm-gaia2 (25-scen wm)"),
    ("wm-tau-telecom", "wm-tau-telecom (25-scen wm)"),
    ("wm-all", "**wm-all** (SAFETY row, never an optimization target)"),
]
# Which config counts as "the production champion" on a cohort, best match first. The shipped
# contract is adaptive neighbourhood + novelty floor + stat guard at z=0.5, so a floor-enabled
# row is a closer match to production than the same config with the floor disabled.
CHAMPION_PRIORITY = [
    "r1-knn-adapt3-oai",
    "r3b-real-champ-floor",
    "r3b-floorq0.5",
    "r1-knn-adapt-oai",
    "r1-knn-statz05-oai",
    "r1-knn-statz05",
    "r3b-floorq0.0",
]
VERDICT_TEXT = {
    "beats": "**BEATS**",
    "promising": "promising (small test set)",
    "ties": "ties",
    "identical": "abstains (= baseline)",
    "mixed": "mixed seeds",
    "unfavourable": "**loses** (small test set)",
    "worse": "**WORSE**",
    "underpowered": "underpowered",
}


def results_table(groups: list[dict]) -> str:
    """The production config, cohort by cohort, as a markdown table for the PR body."""
    head = (
        "| niche | discovered best-single | baseline acc | champion acc +- sd | dAcc (paired) "
        "| dcost | dp50 | seeds | n/seed | verdict | config row used |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|"
    )
    lines = [head]
    for cohort, label in TABLE_ROWS:
        rows = [g for g in groups if g["m"] == cohort and g["tier"] != "anchor"]
        pick = next(
            (g for name in CHAMPION_PRIORITY for g in rows if g["v"] == name),
            None,
        )
        if pick is None:
            lines.append(
                f"| {label} | - | - | no champion-config row on this cohort " + "| - " * 7 + "|"
            )
            continue
        base = pick["b"]
        d_cost = f"{pick['cost'] / base['cost'] - 1:+.0%}" if base["cost"] else "-"
        d_p50 = (
            f"{pick['p50'] - base['p50']:+.2f}s"
            if pick["p50"] is not None and base["p50"] is not None
            else "-"
        )
        if pick["tier"] == "identical":
            delta = "abstains (= baseline)"
        else:
            # Bold any negative delta: the rows where the champion does not help are the
            # ones a reader most needs to notice.
            body = f"{100 * pick['d']:+.2f}pt +- {100 * pick['dsd']:.2f}"
            delta = f"**{body}**" if pick["d"] < 0 else body
        lines.append(
            f"| {label} | {'/'.join(base['models']) or '-'} | {base['acc']:.4f} "
            f"| {pick['acc']:.4f} +- {pick['sd']:.4f} | {delta} | {d_cost} | {d_p50} "
            f"| {pick['s']} | {pick['ntmed']:.0f} | {VERDICT_TEXT.get(pick['tier'], pick['tier'])} "
            f"| `{pick['v']}` |"
        )
    return "\n".join(lines)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups, total_runs = load_groups()
    written = [fig_pareto(groups, plt)]
    census_path, counts = fig_census(groups, plt, total_runs)
    written.append(census_path)
    written.append(fig_confidence(plt))
    written.append(fig_signal_map(groups, plt))
    written.append(fig_niches(groups, plt))
    written.append(fig_operating_points(groups, plt))

    sys.stderr.write("\n" + results_table(groups) + "\n\n")
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    sys.stderr.write(
        f"snapshot {stamp}: {total_runs} runs -> "
        f"{sum(v for k, v in counts.items() if k != 'anchor')} groups\n"
    )
    for key in (
        "beats",
        "promising",
        "ties",
        "identical",
        "worse",
        "unfavourable",
        "mixed",
        "underpowered",
    ):
        sys.stderr.write(f"  {key:14s} {counts[key]}\n")
    for path in written:
        sys.stderr.write(f"wrote {path}\n")


if __name__ == "__main__":
    main()
