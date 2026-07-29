"""Quality-corner topline figures: quality vs anchor, and the quality COST of each lever.

Charter deliverable 2 through the quality lens. Two figures, each with a numbers sidecar:

1. `lever-quality-cost`: one row per optimizer lever (distill / +routing / truncate control /
   +compaction), each a PAIRED per-scenario delta against its own baseline with a 95% CI, the
   per-scenario deltas shown as dots, and the +-0.02 noise floor drawn as a band. A lever's
   row renders only when its data has landed; missing rows are named on the figure.
2. `quality-vs-anchor`: every measured (candidate model x compression arm) config's mean
   reward with its CI against the fable-5 anchor, WM test band. Grid-dependent, so this
   script exits with a named pending note until the identity arm matrix lands.

Both figures annotate the other two objectives per the charter (honest units, or the honest
absence: cycle-1 rows record no per-arm cost).

Run:

    uv run --extra viz python .agents/docs/research/corners/quality/render_quality.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from statistics import median

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import data  # noqa: E402
import palette  # noqa: E402
import stats  # noqa: E402
import teacher_view  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

FIGURES = Path(__file__).resolve().parent / "figures"


class LeverRow(BaseModel):
    """One lever's paired quality delta, with its provenance and its other two objectives."""

    lever: str
    baseline: str
    delta: stats.PairedDelta
    provenance: str
    judge: str
    cost_note: str
    latency_note: str


class LeverChart(BaseModel):
    """The lever figure's numbers: rendered rows plus the levers still pending."""

    rows: list[LeverRow]
    pending: list[str]


def build_lever_rows() -> LeverChart:
    """Every lever whose paired data exists on disk, plus the named list of those that do not."""
    rows: list[LeverRow] = []
    pending: list[str] = []

    cycle_rows = data.load_cycle1_rows()
    after = [r for r in cycle_rows if r.arm == "student-after" and not r.infra_failed]
    before = [r for r in cycle_rows if r.arm == "student-before" and not r.infra_failed]
    rows.append(
        LeverRow(
            lever="distill (cycle 1, unpromoted)",
            baseline="student base (Qwen3.5-9B)",
            delta=stats.paired_delta(
                data.cycle1_rewards_by_task(cycle_rows, arm="student-after"),
                data.cycle1_rewards_by_task(cycle_rows, arm="student-before"),
            ),
            provenance="real_episode",
            judge=data.CYCLE1_JUDGE,
            cost_note="per-arm $ not recorded in cycle-1 rows",
            latency_note=(
                f"p50 episode wall {median(r.duration_s for r in before):.0f}s -> "
                f"{median(r.duration_s for r in after):.0f}s"
            ),
        )
    )

    identity = data.load_arm_matrix(data.IDENTITY_ARM)
    if identity is None:
        pending.append("truncate and +compaction levers: identity arm matrix still merging")
        pending.append("+routing lever: identity matrix + the master's policy fits")
        return LeverChart(rows=rows, pending=pending)

    for arm_name, lever, note in (
        ("truncate", "truncate control (ratio-matched)", "control, not a product lever"),
        ("llmlingua2-endpoint", "+compaction (llmlingua2)", "measured tradeoff, not recommendation"),
    ):
        arm_matrix = data.load_arm_matrix(arm_name)
        if arm_matrix is None:
            pending.append(f"{lever}: {arm_name} arm matrix still merging")
            continue
        # Paired per candidate model, pooled per scenario across the shared pool: each model's
        # scenario rewards pair against ITS OWN identity rows, keyed scenario|model so pairs
        # never cross models.
        arm_map: dict[str, list[float]] = {}
        base_map: dict[str, list[float]] = {}
        for model in sorted({o.model for o in identity.outcomes}):
            for sid, rewards in data.rewards_by_scenario(arm_matrix.outcomes, model=model).items():
                arm_map[f"{sid}|{model}"] = rewards
            for sid, rewards in data.rewards_by_scenario(identity.outcomes, model=model).items():
                base_map[f"{sid}|{model}"] = rewards
        if not arm_map:
            pending.append(f"{lever}: no scored rows yet")
            continue
        delta = stats.paired_delta(arm_map, base_map)
        compressor_cost = sum(o.compressor_cost_usd for o in arm_matrix.outcomes if o.scored)
        rows.append(
            LeverRow(
                lever=lever,
                baseline="identity arm, same models",
                delta=delta,
                provenance="wm_simulated",
                judge="WM verifier (grid cohort pins)",
                cost_note=f"compressor overhead ${compressor_cost:.2f} over scored cells; {note}",
                latency_note="see the latency corner's clocks",
            )
        )
    pending.append("+routing lever: waits on the master's per-arm policy fits")
    return LeverChart(rows=rows, pending=pending)


def render_lever_chart(chart: LeverChart) -> Path:
    """One row per lever: per-pair delta dots, the paired mean with CI, the noise-floor band."""
    palette.apply_style()
    height = 1.6 + 1.1 * len(chart.rows)
    fig, ax = plt.subplots(figsize=(9, height))

    span = max(
        [0.1] + [abs(v) for row in chart.rows for v in row.delta.scenario_deltas.values()]
    )
    ax.axvspan(
        -stats.NOISE_FLOOR_REWARD,
        stats.NOISE_FLOOR_REWARD,
        color=palette.NOISE_BAND_COLOR,
        alpha=palette.NOISE_BAND_ALPHA,
        zorder=0,
    )
    ax.annotate(
        "noise floor (+-0.02)",
        (0, 0.02),
        xycoords=("data", "axes fraction"),
        ha="center",
        va="bottom",
        fontsize=8,
        color=palette.MUTED,
    )
    ax.axvline(0, color=palette.MUTED, linewidth=0.8)

    verdict_prose = {
        "no_effect": "no measurable effect (CI spans zero)",
        "within_noise_floor": "within the noise floor",
        "measurable": "measurable effect",
    }
    labels: list[str] = []
    for i, row in enumerate(chart.rows):
        y = len(chart.rows) - 1 - i
        color = palette.SERIES_COLORS.get("distill-only" if "distill" in row.lever else "+compaction")
        # One marker per DISTINCT delta, sized by its count with the count labeled: k attempts
        # per task quantize deltas, so exact ties are the norm and plain overplotting would
        # hide most of the n (cycle-1 has 13 tasks sitting exactly at zero).
        counts: dict[float, int] = {}
        for value in row.delta.scenario_deltas.values():
            counts[value] = counts.get(value, 0) + 1
        for value, count in sorted(counts.items()):
            ax.plot(
                [value],
                [y - 0.22],
                linestyle="none",
                marker="o",
                markersize=4 + 2.5 * (count**0.5),
                color=color,
                alpha=0.45,
            )
            if count > 1:
                ax.annotate(
                    f"x{count}",
                    (value, y - 0.22),
                    xytext=(0, -11),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7.5,
                    color=palette.MUTED,
                )
        ax.errorbar(
            [row.delta.mean_delta],
            [y],
            xerr=[
                [row.delta.mean_delta - row.delta.ci_low],
                [row.delta.ci_high - row.delta.mean_delta],
            ],
            color=color,
            marker="D",
            markersize=7,
            capsize=3,
            linewidth=2,
        )
        p = row.delta.sign_test_p
        sign_note = f"; sign test p={p:.2f}" if p is not None else "; no scenario moved"
        ax.annotate(
            f"{row.delta.mean_delta:+.3f} [{row.delta.ci_low:+.3f}, {row.delta.ci_high:+.3f}], "
            f"{verdict_prose[row.delta.verdict]}{sign_note}  "
            f"({row.delta.n_up} up / {row.delta.n_down} down / {row.delta.n_tied} tied, "
            f"n={row.delta.n_pairs})",
            (0.01, y + 0.32),
            xycoords=("axes fraction", "data"),
            fontsize=8,
            color=palette.INK,
        )
        ax.annotate(
            f"{row.provenance} | {row.cost_note} | {row.latency_note}",
            (0.01, y - 0.55),
            xycoords=("axes fraction", "data"),
            fontsize=7.5,
            color=palette.MUTED,
        )
        labels.append(f"{row.lever}\nvs {row.baseline}")

    ax.set_yticks(range(len(chart.rows) - 1, -1, -1), labels, fontsize=9)
    ax.set_ylim(-0.85, len(chart.rows) - 0.3 + 0.5)
    ax.set_xlim(-span * 1.15, span * 1.15)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("paired per-scenario reward delta (lever minus its baseline)")
    ax.set_title("The quality cost of each lever (paired, noise floor drawn)")
    if chart.pending:
        palette.footnote(
            fig,
            "Pending: " + "; ".join(chart.pending) + ". Judges: "
            + "; ".join(sorted({row.judge for row in chart.rows})) + ".",
            y=-0.12,
        )

    out = FIGURES / "lever-quality-cost.png"
    palette.save_fig(fig, out)
    out.with_suffix(".json").write_text(chart.model_dump_json(indent=2), encoding="utf-8")
    return out


def render_quality_vs_anchor() -> Path | None:
    """Every measured config's quality vs the fable-5 anchor (WM test band); None until data."""
    arms = [(name, data.load_arm_matrix(name)) for name in data.GRID_ARMS]
    landed = [(name, m) for name, m in arms if m is not None]
    if not landed:
        logger.info("quality-vs-anchor pending: no grid arm matrix has landed yet")
        return None

    palette.apply_style()
    arm_colors = {"identity": palette.BLUE, "truncate": palette.MUTED,
                  "llmlingua2-endpoint": palette.RED}
    configs: list[tuple[str, str, stats.MeanCI]] = []
    anchor: stats.MeanCI | None = None
    for name, matrix in landed:
        for model in sorted(matrix.model_names()):
            rewards = data.rewards_by_scenario(matrix.outcomes, model=model)
            if not rewards:
                continue
            ci = stats.mean_with_ci(rewards)
            configs.append((name, model, ci))
            if name == data.IDENTITY_ARM and model == "fable-5":
                anchor = ci
    configs.sort(key=lambda c: c[2].mean)

    fig, ax = plt.subplots(figsize=(9, 1.5 + 0.32 * len(configs)))
    for y, (arm_name, model, ci) in enumerate(configs):
        ax.errorbar(
            [ci.mean], [y],
            xerr=[[ci.mean - ci.ci_low], [ci.ci_high - ci.mean]],
            color=arm_colors[arm_name], marker="o", markersize=5, capsize=2, linewidth=1.5,
        )
    ax.set_yticks(
        range(len(configs)),
        [f"{model} / {arm_name}" for arm_name, model, _ in configs],
        fontsize=8,
    )
    if anchor is not None:
        ax.axvline(anchor.mean, color=palette.MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.annotate(f"fable-5 anchor {anchor.mean:.3f}", (anchor.mean, len(configs) - 0.5),
                    fontsize=8.5, color=palette.MUTED, ha="center", va="bottom")
    ax.set_xlabel("mean task reward (WM test band, 95% cluster-bootstrap CI)")
    ax.set_title("Quality vs anchor, every measured config")
    ax.grid(axis="y", visible=False)
    pending_arms = [name for name, m in arms if m is None]
    palette.footnote(
        fig,
        "wm_simulated; WM verifier per grid cohort pins; per-config cost and latency in the "
        "cost and latency corners' figures over the same cells."
        + (f" Pending arms: {', '.join(pending_arms)}." if pending_arms else ""),
    )
    out = FIGURES / "quality-vs-anchor.png"
    palette.save_fig(fig, out)
    return out


def render_teacher_verdict() -> Path | None:
    """The teacher-search verdict table as a figure: the distill go/no-go made visible.

    Per-model paired gain over the cheapest candidate on the identity arm, CI-guarded with
    the shared verdict rule. Renders from pre-retry chunks when only those exist, with
    completeness labeled. Computation handoff contract: common/teacher_view.py docstring
    (wmo.optimize.teacher on jt/teacher-gate replaces it when it lands).
    """
    snapshot = data.load_arm_snapshot(data.IDENTITY_ARM)
    if snapshot is None:
        logger.info("teacher-verdict pending: identity arm has no matrix or chunks yet")
        return None
    rows = teacher_view.teacher_verdict_rows(snapshot.matrix)

    palette.apply_style()
    fig, ax = plt.subplots(figsize=(9, 1.6 + 0.42 * len(rows)))
    ax.axvspan(
        -stats.NOISE_FLOOR_REWARD,
        stats.NOISE_FLOOR_REWARD,
        color=palette.NOISE_BAND_COLOR,
        alpha=palette.NOISE_BAND_ALPHA,
        zorder=0,
    )
    ax.axvline(0, color=palette.MUTED, linewidth=0.8)
    for y, row in enumerate(reversed(rows)):
        color = palette.BLUE if row.verdict == "headroom" else palette.MUTED
        ax.errorbar(
            [row.delta.mean_delta],
            [y],
            xerr=[
                [row.delta.mean_delta - row.delta.ci_low],
                [row.delta.ci_high - row.delta.mean_delta],
            ],
            color=color,
            marker="D",
            markersize=6,
            capsize=3,
            linewidth=1.8,
        )
        cost = (
            f"${row.cost_per_completed_task_usd:.2f}/task"
            if row.cost_per_completed_task_usd is not None
            else "cost undefined (nothing completed)"
        )
        ax.annotate(
            f"{row.verdict} | {cost}",
            (1.01, y),
            xycoords=("axes fraction", "data"),
            fontsize=8,
            color=palette.INK if row.verdict == "headroom" else palette.MUTED,
            va="center",
        )
    ax.set_yticks(
        range(len(rows) - 1, -1, -1),
        [f"{row.model}  ({row.mean_reward:.3f})" for row in rows],
        fontsize=9,
    )
    ax.set_xlabel(
        f"paired per-scenario reward gain over the cheapest candidate ({rows[0].baseline})"
    )
    ax.set_title("Teacher search: who has headroom worth distilling (CI-guarded)")
    ax.grid(axis="y", visible=False)
    palette.footnote(
        fig,
        f"wm_simulated, identity arm ({snapshot.status}); WM verifier per grid cohort pins; "
        f"noise floor +-0.02 shaded; 'headroom' needs a CI excluding zero AND a mean past the "
        f"floor. Baseline = cheapest by effective cost per completed task (proxies the student "
        f"tier until student cells merge; cycle-1's lesson: a 1.6pt teacher gap distills "
        f"nothing). Verdict via corners/common until wmo.optimize.teacher (jt/teacher-gate) "
        f"lands, then this figure renders the repo function's table.",
        y=-0.1,
    )
    out = FIGURES / "teacher-verdict.png"
    palette.save_fig(fig, out)
    out.with_suffix(".json").write_text(
        "[" + ",".join(row.model_dump_json() for row in rows) + "]", encoding="utf-8"
    )
    return out


def main() -> None:
    """Render every quality topline figure whose data has landed; log what is pending."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    chart = build_lever_rows()
    logger.info("rendered %s", render_lever_chart(chart))
    anchor_fig = render_quality_vs_anchor()
    if anchor_fig is not None:
        logger.info("rendered %s", anchor_fig)
    teacher_fig = render_teacher_verdict()
    if teacher_fig is not None:
        logger.info("rendered %s", teacher_fig)


if __name__ == "__main__":
    main()
