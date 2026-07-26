"""Render the status figures for the Notion page as one self-contained HTML file.

Two panels, in the order a reader needs them to follow the lane's central claim:

1. The teacher does not dominate the student on math. On AIME the base
   Qwen3.6-27B scores ABOVE the GLM-5.2 teacher on every available
   measurement; on MATH-500 the teacher leads by ~6 points. A distillation
   gain cannot be demonstrated where the teacher has no headroom, which is
   why the lane moved to agentic tool use (TerminalBench-2).
2. Why several earlier numbers were wrong: measured pass@1 is a FLOOR
   whenever rollouts truncate. The same model and grader move 78.3% -> 93.3%
   on AIME purely by lifting the token budget from 32,768 to 65,536.

Every value is computed from the eval JSONs under `.wmh/xtoken-runs/evals/`
rather than hardcoded. An earlier revision of this script hardcoded figures
that were later withdrawn, and the figure kept asserting them after the page
had been corrected.

Palette per AGENTS.md rule 14 (ink, hairline grid, white ground, brand accents).

Usage:
    uv run python .agents/distill/figures.py --out /tmp/xtoken_figures.html
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# Emit text as <text> rather than vector outlines: the outline form inflates
# the SVG roughly 6x for no visual gain at these sizes.
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["svg.hashsalt"] = "xtoken"
matplotlib.rcParams["path.simplify"] = True
matplotlib.rcParams["path.simplify_threshold"] = 1.0
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

INK = "#0a0a0a"
GRID = "#ececec"
BLUE = "#0070f3"
AMBER = "#f5a623"
RED = "#ee0000"

EVALS = Path(".wmh/xtoken-runs/evals")


class Eval:
    """One scored eval file, normalized across the two row schemas in use.

    Rows carry either a `truncated` bool (the Tinker-side harness) or a
    `finish_reason` string (the hosted-teacher harness), and the teacher's
    Azure rows can carry neither. A row whose completion state is unknown is
    counted as neither stopped nor truncated, so `stopped_rate` and
    `truncation_rate` need not sum to 1 -- the shortfall IS the unknown
    fraction and the caller must surface it rather than assume.
    """

    def __init__(self, name: str) -> None:
        rows = json.loads((EVALS / f"{name}.json").read_text())
        self.name = name
        self.rows = rows if isinstance(rows, list) else rows["rows"]
        self.n = len(self.rows)

    def _state(self, row: dict) -> str:
        if row.get("truncated") is True:
            return "truncated"
        reason = row.get("finish_reason")
        if reason == "length":
            return "truncated"
        if reason == "stop" or row.get("truncated") is False:
            return "stopped"
        return "unknown"

    @property
    def pass_at_1(self) -> float:
        return 100 * sum(1 for r in self.rows if r.get("correct")) / self.n

    @property
    def truncation_rate(self) -> float:
        return 100 * sum(1 for r in self.rows if self._state(r) == "truncated") / self.n

    @property
    def unknown_rate(self) -> float:
        return 100 * sum(1 for r in self.rows if self._state(r) == "unknown") / self.n

    @property
    def pass_at_1_stopped(self) -> tuple[float, int]:
        """Accuracy over rows that verifiably stopped, and how many those are.

        This is NOT an unbiased estimate of the model's accuracy: excluding
        truncated rows conditions on finishing, which correlates with problem
        difficulty. It is reported only as an upper bound on the floor.
        """
        stopped = [r for r in self.rows if self._state(r) == "stopped"]
        if not stopped:
            return float("nan"), 0
        return 100 * sum(1 for r in stopped if r.get("correct")) / len(stopped), len(stopped)


def _style(axis: plt.Axes) -> None:
    axis.set_facecolor("white")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=INK, labelsize=9, length=0)
    axis.yaxis.set_major_formatter(PercentFormatter())
    axis.yaxis.grid(True, color=GRID, linewidth=1)
    axis.set_axisbelow(True)


def headroom_panel(axis: plt.Axes, evals: dict[str, Eval]) -> None:
    """Teacher vs untrained student, at the budgets where nothing truncated."""
    student = [evals["q27b_aime_65k"].pass_at_1, evals["q27b_math500_65k"].pass_at_1]
    teacher_stopped, teacher_n = evals["glm_aime_azure"].pass_at_1_stopped
    teacher = [teacher_stopped, evals["glm_math500_v2"].pass_at_1]

    offsets = [-0.2, 0.2]
    positions = [0, 1]
    axis.bar([p + offsets[0] for p in positions], student, width=0.36,
             color=AMBER, edgecolor="none", label="Qwen3.6-27B (untrained student)")
    axis.bar([p + offsets[1] for p in positions], teacher, width=0.36,
             color=BLUE, edgecolor="none", label="GLM-5.2 (teacher)")

    for position, offset, value in ((0, -0.2, student[0]), (1, -0.2, student[1]),
                                    (0, 0.2, teacher[0]), (1, 0.2, teacher[1])):
        axis.text(position + offset, value + 1.8, f"{value:.1f}", ha="center",
                  color=INK, fontsize=10.5, fontweight="bold")

    axis.text(0.0, 112, "student ABOVE teacher", ha="center", color=RED,
              fontsize=9.5, fontweight="bold")
    axis.text(1.0, 112, f"teacher +{teacher[1] - student[1]:.1f} pt", ha="center",
              color=INK, fontsize=9.5, fontweight="bold")
    axis.plot([-0.2, 0.2], [107, 107], color=RED, linewidth=1)
    axis.plot([0.8, 1.2], [107, 107], color=INK, linewidth=1)

    axis.set_xticks(positions)
    axis.set_xticklabels([f"AIME 24+25\n(n={evals['q27b_aime_65k'].n})",
                          f"MATH-500\n(n={evals['q27b_math500_65k'].n})"], fontsize=9.5)
    axis.set_ylim(0, 126)
    axis.set_yticks([0, 20, 40, 60, 80, 100])
    axis.set_ylabel("pass@1", color=INK, fontsize=9.5)
    axis.legend(frameon=False, fontsize=9, loc="upper center", ncol=2,
                bbox_to_anchor=(0.5, -0.09), handlelength=1.4, columnspacing=1.6)
    axis.set_title("Math cannot show a distillation gain:\nthe teacher has no headroom to transfer",
                   loc="left", color=INK, fontsize=13, fontweight="bold", pad=14)
    _style(axis)


def floor_panel(axis: plt.Axes, evals: dict[str, Eval]) -> None:
    """The same model and grader, two budgets: truncation suppresses accuracy."""
    small, large = evals["q27b_aime"], evals["q27b_aime_65k"]
    ninebee = evals["q9b_aime"]

    labels = ["Qwen3.6-27B\n32,768", "Qwen3.6-27B\n65,536", "Qwen3.5-9B\n32,000"]
    scores = [small.pass_at_1, large.pass_at_1, ninebee.pass_at_1]
    truncs = [small.truncation_rate, large.truncation_rate, ninebee.truncation_rate]
    colors = [AMBER, AMBER, RED]

    axis.bar(labels, scores, width=0.5, color=colors, edgecolor="none")
    for index, (score, trunc) in enumerate(zip(scores, truncs, strict=True)):
        axis.text(index, score + 1.8, f"{score:.1f}", ha="center", color=INK,
                  fontsize=10.5, fontweight="bold")
        axis.text(index, 4.0, f"{trunc:.0f}% cut", ha="center", va="bottom",
                  color="white" if colors[index] == RED else INK, fontsize=9,
                  fontweight="bold")

    axis.plot([0, 0, 1, 1], [small.pass_at_1 + 7, 112, 112, large.pass_at_1 + 7],
              color=INK, linewidth=1)
    axis.text(0.5, 114, f"+{large.pass_at_1 - small.pass_at_1:.1f} pt from budget alone",
              ha="center", color=INK, fontsize=9.5, fontweight="bold")

    axis.set_ylim(0, 126)
    axis.set_yticks([0, 20, 40, 60, 80, 100])
    axis.set_ylabel("AIME 24+25 pass@1", color=INK, fontsize=9.5)
    axis.set_title("Any score with truncation is a FLOOR:\nsame model, same grader, budget swept",
                   loc="left", color=INK, fontsize=13, fontweight="bold", pad=14)
    _style(axis)


def render(evals: dict[str, Eval]) -> str:
    figure, axes = plt.subplots(1, 2, figsize=(13.6, 5.6))
    figure.patch.set_facecolor("white")
    headroom_panel(axes[0], evals)
    floor_panel(axes[1], evals)
    figure.tight_layout(pad=2.4, rect=(0, 0.13, 1, 1))

    teacher = evals["glm_aime_azure"]
    stopped, count = teacher.pass_at_1_stopped
    figure.text(
        0.008, 0.075,
        f"LEFT: Student bars: 0% truncation at a 65,536 budget. The teacher's AIME bar "
        f"({stopped:.1f}%) covers only the {count}/{teacher.n} rows that verifiably stopped "
        f"({teacher.truncation_rate:.0f}% truncated, {teacher.unknown_rate:.0f}% no "
        f"finish_reason), "
        f"so it conditions on finishing and flatters the teacher; raw is {teacher.pass_at_1:.1f}%. "
        f"The student still leads on AIME.",
        fontsize=7.8, color="#666666", ha="left", va="top", wrap=True)
    figure.text(
        0.008, 0.028,
        "RIGHT: A truncated rollout emits no boxed answer and scores as wrong, so truncation "
        "suppresses measured accuracy. Three conclusions in this lane were withdrawn after being "
        "traced to this; treat any sub-65k number as a lower bound, never as an estimate.",
        fontsize=7.8, color="#666666", ha="left", va="top", wrap=True)
    buffer = io.StringIO()
    figure.savefig(buffer, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/xtoken_figures.html")
    args = parser.parse_args()

    names = ["q27b_aime", "q27b_aime_65k", "q27b_math500_65k", "q9b_aime",
             "glm_aime_azure", "glm_math500_v2"]
    evals = {name: Eval(name) for name in names}

    for name, ev in evals.items():
        stopped, count = ev.pass_at_1_stopped
        print(f"{name:20s} n={ev.n:3d} pass@1={ev.pass_at_1:5.1f}% "
              f"trunc={ev.truncation_rate:5.1f}% unknown={ev.unknown_rate:5.1f}% "
              f"stopped-only={stopped:5.1f}% (n={count})")

    svg = render(evals)
    Path(args.out).write_text(
        "<!doctype html><meta charset=utf-8>"
        "<title>X-Token GLM-5.2 -> Qwen3.6-27B: status</title>"
        f"<body style='margin:0;background:white'>{svg}</body>"
    )
    print(f"wrote {args.out} ({len(svg) / 1024:.1f} KiB svg)")


if __name__ == "__main__":
    main()
