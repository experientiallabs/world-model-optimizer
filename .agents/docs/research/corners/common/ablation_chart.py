"""The canonical training-stage-vs-quality ablation chart, shared by all three corner chats.

Charter deliverable 1: training stage (student base -> cycle-1 -> future gated cycles) on x,
quality on y, three ablation lines (distill-only / +routing / +compaction), teacher and
fable-5 anchor as reference lines, a CI on every point. Each corner chat renders THIS
implementation through its lens into its own figures/ directory; the numbers are built once
here so three renderings can never disagree.

Provenance discipline (binding, common/README.md): the figure is split into one panel per
provenance (wm_simulated vs real_episode). A series line never crosses panels, a reference
line lives only on the panel whose provenance it was measured under, and every panel names
its judge. Series that have not landed yet (grid arms still merging, policy fits pending) are
NAMED on the figure rather than silently absent.

Every render writes a sidecar `<out>.json` with the exact numbers behind the figure, so a
reader can audit any point without re-running the build.

Run (matplotlib comes from the viz extra):

    uv run --extra viz python .agents/docs/research/corners/common/ablation_chart.py \
        --lens quality --out .agents/docs/research/corners/quality/figures/training-stage
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from statistics import median
from typing import Literal

from pydantic import BaseModel, Field

import data
import stats

logger = logging.getLogger(__name__)

Lens = Literal["quality", "cost", "latency"]
Series = Literal["distill-only", "+routing", "+compaction"]
Provenance = Literal["wm_simulated", "real_episode"]

# Stage order is meaningful: the x axis reads as the training timeline.
STAGES: tuple[str, ...] = ("student base", "cycle 1")

CYCLE1_NOTE = "not promoted (gate refused: no teacher headroom)"

PANEL_TITLES: dict[str, str] = {
    "wm_simulated": "WM-simulated (leak-free test band)",
    "real_episode": "Real tau2 episodes (canonical pins)",
}


class StagePoint(BaseModel):
    """One measured point: a series at a training stage, with its CI and its two other axes."""

    stage: str
    series: Series
    mean_reward: float
    ci_low: float
    ci_high: float
    n_scenarios: int
    n_episodes: int
    provenance: Provenance
    judge: str
    note: str = ""  # e.g. the unpromoted-student label
    # The other two objectives, honest units named in the string (charter: every chart reports
    # all three). Built by the builder from what the source actually recorded.
    cost_note: str
    latency_note: str


class ReferenceLine(BaseModel):
    """A horizontal anchor (teacher, fable-5) drawn only on its own provenance panel."""

    name: str
    mean_reward: float
    provenance: Provenance
    judge: str


class ChartData(BaseModel):
    """Everything one rendering needs, and the auditable record of where it came from."""

    points: list[StagePoint]
    references: list[ReferenceLine]
    stages: list[str]
    # Series that exist in the design but have not landed, named per panel ("no silent caps").
    pending: dict[str, list[str]]
    sources: list[str] = Field(min_length=1)


def build_shared_chart_data() -> ChartData:
    """Assemble the chart from what has landed on disk; name what is still pending.

    Sources (charter, read-only): cycle-1 episode rows (exist since 2026-07-27) for the
    real-episode panel; the grid's per-arm matrices for the WM panel (landing as arms merge;
    the +routing and +compaction series additionally wait on the master's per-arm policy fits
    and the student sweep cells, coordinated in the plan ledger).
    """
    points: list[StagePoint] = []
    references: list[ReferenceLine] = []
    pending: dict[str, list[str]] = {"wm_simulated": [], "real_episode": []}
    sources: list[str] = []

    rows = data.load_cycle1_rows()
    sources.append(f"cycle-1 episode-rows.jsonl ({len(rows)} rows) in {data.cycle1_run_dir()}")
    for stage, arm, note in (
        ("student base", "student-before", ""),
        ("cycle 1", "student-after", CYCLE1_NOTE),
    ):
        arm_rows = [r for r in rows if r.arm == arm and not r.infra_failed]
        by_task = data.cycle1_rewards_by_task(rows, arm=arm)
        ci = stats.mean_with_ci(by_task)
        points.append(
            StagePoint(
                stage=stage,
                series="distill-only",
                mean_reward=ci.mean,
                ci_low=ci.ci_low,
                ci_high=ci.ci_high,
                n_scenarios=ci.n_scenarios,
                n_episodes=ci.n_episodes,
                provenance="real_episode",
                judge=data.CYCLE1_JUDGE,
                note=note,
                cost_note="per-arm $ not recorded in cycle-1 rows",
                latency_note=(
                    f"p50 episode {median(r.duration_s for r in arm_rows):.0f}s wall "
                    f"(not model-time)"
                ),
            )
        )
    teacher = stats.mean_with_ci(data.cycle1_rewards_by_task(rows, arm="teacher"))
    references.append(
        ReferenceLine(
            name=f"teacher Qwen3.6-27B {teacher.mean:.3f}",
            mean_reward=teacher.mean,
            provenance="real_episode",
            judge=data.CYCLE1_JUDGE,
        )
    )
    pending["real_episode"].append(
        "+routing / +compaction real legs: run after the master's canonical-pins runner fix"
    )

    identity = data.load_arm_matrix(data.IDENTITY_ARM)
    if identity is None:
        pending["wm_simulated"].append(
            "all WM series + fable-5 anchor: identity arm matrix still merging"
        )
    else:
        sources.append(f"grid identity arm matrix ({len(identity.outcomes)} cells)")
        anchor_rewards = data.rewards_by_scenario(identity.outcomes, model="fable-5")
        if anchor_rewards:
            anchor = stats.mean_with_ci(anchor_rewards)
            references.append(
                ReferenceLine(
                    name=f"fable-5 anchor {anchor.mean:.3f}",
                    mean_reward=anchor.mean,
                    provenance="wm_simulated",
                    judge="WM verifier (see grid matrix.meta.json cohort pins)",
                )
            )
        pending["wm_simulated"].extend(
            [
                "distill-only (WM): student per-arm sweep cells pending",
                "+routing (WM): per-arm policy fits pending (master owns fit timing)",
                "+compaction (WM): fits over the llmlingua2-endpoint arm pending; "
                "measured tradeoff, not recommendation",
            ]
        )

    return ChartData(
        points=points,
        references=references,
        stages=list(STAGES),
        pending=pending,
        sources=sources,
    )


def render_training_stage_chart(chart: ChartData, out: Path, *, lens: Lens = "quality") -> Path:
    """Render the shared chart through one corner's lens; write the PNG and its numbers JSON.

    The lens changes EMPHASIS only (which secondary objective is annotated beside each point);
    the numbers are identical across lenses by construction. Panels are one per provenance,
    sharing the y axis so quality reads across them, with the honesty rule that no line or
    comparison crosses the panel boundary.
    """
    import palette  # matplotlib import lives behind the viz extra
    from matplotlib import pyplot as plt

    palette.apply_style()
    provenances: list[Provenance] = ["wm_simulated", "real_episode"]
    fig, axes = plt.subplots(1, len(provenances), figsize=(11, 5), sharey=True)
    stage_x = {stage: i for i, stage in enumerate(chart.stages)}

    for ax, provenance in zip(axes, provenances, strict=True):
        panel_points = [p for p in chart.points if p.provenance == provenance]
        panel_refs = [r for r in chart.references if r.provenance == provenance]
        ax.set_title(PANEL_TITLES[provenance], fontsize=11)
        ax.set_xticks(list(stage_x.values()), chart.stages)
        ax.set_xlim(-0.5, len(chart.stages) - 0.5)
        ax.grid(axis="x", visible=False)

        for series, color in palette.SERIES_COLORS.items():
            series_points = sorted(
                (p for p in panel_points if p.series == series),
                key=lambda p: stage_x[p.stage],
            )
            if not series_points:
                continue
            xs = [stage_x[p.stage] for p in series_points]
            ys = [p.mean_reward for p in series_points]
            ax.errorbar(
                xs,
                ys,
                yerr=[
                    [p.mean_reward - p.ci_low for p in series_points],
                    [p.ci_high - p.mean_reward for p in series_points],
                ],
                color=color,
                linewidth=2,
                marker="o",
                markersize=6,
                capsize=3,
                elinewidth=1,
            )
            # Direct label at the line's end: identity is never color-alone.
            ax.annotate(
                series,
                (xs[-1], ys[-1]),
                xytext=(8, 0),
                textcoords="offset points",
                color=color,
                fontsize=9,
                va="center",
            )
            for point in series_points:
                secondary = {
                    "quality": f"{point.mean_reward:.3f}",
                    "cost": point.cost_note,
                    "latency": point.latency_note,
                }[lens]
                label = f"{point.note}\n{secondary}" if point.note else secondary
                ax.annotate(
                    label,
                    (stage_x[point.stage], point.ci_high),
                    xytext=(0, 8),
                    textcoords="offset points",
                    color=palette.MUTED,
                    fontsize=7.5,
                    ha="center",
                    va="bottom",
                )

        for ref in panel_refs:
            ax.axhline(ref.mean_reward, color=palette.MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
            ax.annotate(
                ref.name,
                (ax.get_xlim()[0] + 0.05, ref.mean_reward),
                xytext=(0, 4),
                textcoords="offset points",
                color=palette.MUTED,
                fontsize=8.5,
            )

        if pending := chart.pending.get(provenance):
            ax.annotate(
                "pending:\n" + "\n".join(f"- {item}" for item in pending),
                (0.03, 0.04),
                xycoords="axes fraction",
                color=palette.MUTED,
                fontsize=7.5,
                va="bottom",
            )

    axes[0].set_ylabel("task reward (0-1)")
    fig.suptitle(
        f"Training stage vs quality: the joint tau-bench ladder ({lens} lens)",
        x=0.02,
        ha="left",
        fontsize=13,
    )
    judges = sorted({p.judge for p in chart.points} | {r.judge for r in chart.references})
    caption = (
        "CIs: 95% cluster bootstrap over scenarios (paired-stats conventions, corners/common). "
        + " Judges: "
        + "; ".join(judges)
        + ". Panels never share a comparison: provenance is a hard boundary."
    )
    fig.text(0.02, -0.03, caption, color="#8a8a8a", fontsize=7.5, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = out.with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    out.with_suffix(".json").write_text(chart.model_dump_json(indent=2), encoding="utf-8")
    logger.info("wrote %s and its numbers sidecar", out)
    return out


def main() -> None:
    """CLI face: build from what has landed, render through one lens."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lens", choices=("quality", "cost", "latency"), default="quality")
    parser.add_argument("--out", type=Path, required=True, help="output path (suffix ignored)")
    args = parser.parse_args()
    chart = build_shared_chart_data()
    rendered = render_training_stage_chart(chart, args.out, lens=args.lens)
    logger.info("rendered %s", rendered)


if __name__ == "__main__":
    main()
