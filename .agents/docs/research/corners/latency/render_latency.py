"""Render the LATENCY corner's phase-1 figures from the canonical tau grid + cycle artifacts.

Deliverables (charter `../README.md`, conventions binding per `../common/README.md`):

1. `latency_per_config`: per-task model-seconds p50 (dot) and p95 (whisker) for every measured
   (candidate x compression arm) config, each row annotated with the other two objectives.
2. `latency_quality_frontier`: quality (with cluster-bootstrap CI) vs per-task p50 across all
   measured configs, marker area encoding effective cost per completed task, the
   latency-quality Pareto front drawn, the noise-floor band under the best config shaded.
3. `training-stage`: the SHARED training-stage-vs-quality chart rendered through the
   latency lens (each stage point annotated with its p50 episode wall seconds) - DELEGATED
   to the canonical `common/ablation_chart.py` so the three chats' stage charts cannot
   disagree on numbers; this script only picks the lens and the output path.
4. `cold_start_first_vs_warm`: each config's first-call vs warm-call seconds, kimi-k3's
   documented 51 s serverless cold start called out explicitly.
5. `latency_max_corner.json` + stdout table: the LATENCY-MAX named corner - the mountable
   config minimizing per-task p50 at acceptable quality - with all three objectives, the
   PAIRED delta vs the best config (stats.paired_delta, the binding convention), and a
   sensitivity sweep over the quality floor.

Offline computation only (zero LLM spend). The grid may still be filling: partial chunk
reads are labeled on every figure, and the script is safe to re-run whenever a chunk or
merged matrix lands.

    # from the corners worktree root
    uv run --extra viz python .agents/docs/research/corners/latency/render_latency.py
    # schema smoke against a synthetic matrix, no grid data needed, writes to --out/synthetic
    uv run --extra viz python .agents/docs/research/corners/latency/render_latency.py --synthetic
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
from pathlib import Path

CORNERS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CORNERS_DIR / "common"))

from data import ArmSnapshot, all_arm_snapshots, rewards_by_scenario
from latency import (
    ARM_COLORS,
    ConfigPoint,
    config_points,
    dump_points,
    first_vs_warm,
    p50_p95,
)
from palette import BLUE, INK, MUTED, NOISE_BAND_ALPHA, NOISE_BAND_COLOR, RED, apply_style
from stats import NOISE_FLOOR_REWARD, paired_delta

import matplotlib.pyplot as plt  # noqa: E402  (palette already selected the Agg backend)

from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.providers.pool import PoolEntry

# The documented cold start this corner must call out: run_tau_grid.py measured kimi-k3's
# first call at 51 seconds (serverless scale-up), recorded as wall time, never killed.
K3_COLD_START_S = 51.0
K3_NAME = "kimi-k3"

# Grid provenance note shared by every grid-derived figure (labeling rules, common/README.md).
GRID_BASIS = (
    "wm_simulated; WM verifier rewards (model dir .wmo/models/tau-bench, cohort tip 11188624); "
    "per-task s = sum of model call seconds + compressor wall (env/judge time excluded; blank-"
    "retry calls INCLUDED per the scorecard contract); per-call stats drop blank-answered "
    "calls (#295)."
)


def _footnote(fig, text: str) -> None:
    """Wrapped provenance line. Interim local helper: replace with palette.footnote once the
    palette module's in-flight helper edits land (they are uncommitted sibling work)."""
    wrapped = "\n".join(textwrap.wrap(text, width=150))
    fig.text(0.01, -0.02, wrapped, color=MUTED, fontsize=7.5, ha="left", va="top")


def _save(fig, out_base: Path) -> None:
    """PNG + SVG beside each other; savefig params come from palette.apply_style."""
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(out_base.with_suffix(suffix))
    plt.close(fig)


def _label(point: ConfigPoint) -> str:
    return f"{point.model} @ {point.arm}"


def _cost_text(point: ConfigPoint) -> str:
    if point.cost_per_completed_usd is None:
        return "cost/task undefined (0 completed)"
    return f"${point.cost_per_completed_usd:.3f}/task"


def fig_latency_per_config(points: list[ConfigPoint], status: str, out: Path) -> None:
    """Deliverable 1: per-config per-task p50/p95, annotated with quality and cost."""
    ordered = sorted(points, key=lambda p: p.p50_task_s)
    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.42 * len(ordered) + 1.5)))
    for y, point in enumerate(ordered):
        color = ARM_COLORS.get(point.arm, INK)
        ax.plot(
            [point.p50_task_s, point.p95_task_s], [y, y], color=color, linewidth=1.2, alpha=0.55
        )
        ax.plot(point.p95_task_s, y, marker="|", color=color, markersize=9)
        ax.plot(point.p50_task_s, y, marker="o", color=color, markersize=6)
        ax.annotate(
            f"  reward {point.mean_reward:.2f} · {_cost_text(point)} · n={point.n_scored}",
            (point.p95_task_s, y),
            textcoords="offset points",
            xytext=(6, -3),
            color=MUTED,
            fontsize=7.5,
        )
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels([_label(p) for p in ordered], fontsize=8.5, color=INK)
    ax.set_title("tau grid: per-task model seconds by config (p50 dot, p95 whisker)")
    ax.set_xlabel("per-task model seconds (p50 / p95)")
    handles = [
        plt.Line2D([], [], color=c, marker="o", linestyle="-", label=a)
        for a, c in ARM_COLORS.items()
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    _footnote(fig, f"{GRID_BASIS} Data: {status}.")
    _save(fig, out / "latency_per_config")


def pareto_front(points: list[ConfigPoint]) -> list[ConfigPoint]:
    """Latency-quality non-dominated set: fastest-first, keep strict quality improvements."""
    front: list[ConfigPoint] = []
    best = float("-inf")
    for point in sorted(points, key=lambda p: (p.p50_task_s, -p.mean_reward)):
        if point.mean_reward > best:
            front.append(point)
            best = point.mean_reward
    return front


def fig_frontier(
    points: list[ConfigPoint], snapshots: list[ArmSnapshot], status: str, out: Path
) -> None:
    """Deliverable 2: the latency-quality frontier, cost as marker area, CI on quality.

    The CI is the cluster bootstrap over scenarios (stats.mean_with_ci); the shaded band
    under the best config's mean is the binding noise floor - configs whose mean sits inside
    it are ties with the best on quality, whatever the dots suggest.
    """
    from stats import mean_with_ci

    reward_maps = {
        (s.name, m): rewards_by_scenario(s.matrix.outcomes, model=m)
        for s in snapshots
        for m in s.matrix.model_names()
    }
    fig, ax = plt.subplots(figsize=(9, 6))
    costs = [p.cost_per_completed_usd for p in points if p.cost_per_completed_usd]
    scale = max(costs) if costs else 1.0
    best = max(p.mean_reward for p in points)
    ax.axhspan(
        best - NOISE_FLOOR_REWARD, best, color=NOISE_BAND_COLOR, alpha=NOISE_BAND_ALPHA, zorder=1
    )
    ax.annotate(
        f"noise floor: ties with best (±{NOISE_FLOOR_REWARD:g} reward)",
        (0.99, best - NOISE_FLOOR_REWARD / 2),
        xycoords=("axes fraction", "data"),
        ha="right", va="center", fontsize=7.5, color=MUTED,
    )
    for point in points:
        color = ARM_COLORS.get(point.arm, INK)
        area = 30 + 220 * ((point.cost_per_completed_usd or 0.0) / scale)
        reward_map = reward_maps.get((point.arm, point.model))
        if reward_map:
            ci = mean_with_ci(reward_map)
            ax.plot(
                [point.p50_task_s, point.p50_task_s], [ci.ci_low, ci.ci_high],
                color=color, linewidth=1.0, alpha=0.5, zorder=2,
            )
        ax.scatter(
            point.p50_task_s, point.mean_reward, s=area, color=color, alpha=0.75,
            edgecolors="white", linewidths=0.8, zorder=3,
        )
    front = pareto_front(points)
    ax.step(
        [p.p50_task_s for p in front], [p.mean_reward for p in front],
        where="post", color=INK, linewidth=1.1, alpha=0.8, zorder=2,
    )
    for point in front:
        ax.annotate(
            f"{_label(point)}\n{_cost_text(point)}",
            (point.p50_task_s, point.mean_reward),
            textcoords="offset points", xytext=(8, 6), fontsize=7.5, color=INK,
        )
    handles = [
        plt.Line2D([], [], color=c, marker="o", linestyle="", label=a)
        for a, c in ARM_COLORS.items()
    ] + [plt.Line2D([], [], color=INK, linewidth=1.1, label="Pareto front")]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    ax.set_title("tau grid: latency-quality frontier (marker area = $/completed task)")
    ax.set_xlabel("per-task model seconds, p50")
    ax.set_ylabel("mean reward (per-scenario averaged, 95% cluster-bootstrap CI)")
    _footnote(fig, f"{GRID_BASIS} Data: {status}.")
    _save(fig, out / "latency_quality_frontier")


def fig_cold_start(snapshots: list[ArmSnapshot], status: str, out: Path) -> None:
    """Deliverable 4: K3's cold start made visible - first call vs warm calls per candidate."""
    per_model: dict[str, tuple[list[float], list[float]]] = {}
    for snapshot in snapshots:
        for model in snapshot.matrix.model_names():
            rows = [o for o in snapshot.matrix.outcomes if o.model == model]
            first, warm = first_vs_warm(rows)
            have = per_model.setdefault(model, ([], []))
            have[0].extend(first)
            have[1].extend(warm)
    measured = {m: fw for m, fw in per_model.items() if fw[0]}
    if not measured:
        return
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.5 * len(measured) + 1.5)))
    names = sorted(measured, key=lambda m: max(measured[m][0]), reverse=True)
    for y, model in enumerate(names):
        first, warm = measured[model]
        ax.scatter(warm, [y] * len(warm), s=12, color=MUTED, alpha=0.35, zorder=2)
        ax.scatter(
            first, [y] * len(first), s=26, color=RED if model == K3_NAME else BLUE,
            alpha=0.8, zorder=3,
        )
        worst = max(first)
        if model == K3_NAME and worst > 2 * (p50_p95(warm)[0] if warm else 1.0):
            ax.annotate(
                f"cold start: {worst:.0f}s first call (runner-documented ~{K3_COLD_START_S:.0f}s)",
                (worst, y), textcoords="offset points", xytext=(8, -3),
                fontsize=8, color=RED,
            )
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_title("episode first call (color) vs warm calls (grey), per candidate")
    ax.set_xlabel("call seconds (log scale; blank-answered calls dropped per #295)")
    _footnote(
        fig,
        f"{GRID_BASIS} First-call points are cold-start WITNESSES only (idle gaps between "
        f"chunks are unrecorded). Data: {status}.",
    )
    _save(fig, out / "cold_start_first_vs_warm")


def latency_max_corner(
    points: list[ConfigPoint],
    snapshots: list[ArmSnapshot],
    floors_pts: tuple[float, ...],
    out: Path,
) -> dict:
    """Deliverable 5: the LATENCY-MAX named corner with a quality-floor sensitivity sweep.

    "Acceptable quality" is a floor on the PAIRED per-scenario delta against the best
    measured config (stats.paired_delta, the binding convention: unpaired mean comparisons
    are banned). All three objectives ride on the pick, along with the paired delta and its
    noise-floor flag. This is an OFFLINE mount choice among measured configs, not an online
    routing rule - the recorded limitation, DECISIONS 2026-07-27 three-corners entry.
    """
    reward_maps = {
        (s.name, m): rewards_by_scenario(s.matrix.outcomes, model=m)
        for s in snapshots
        for m in s.matrix.model_names()
    }
    best = max(points, key=lambda p: p.mean_reward)
    best_map = reward_maps[(best.arm, best.model)]

    deltas: dict[tuple[str, str], object] = {}
    for point in points:
        key = (point.arm, point.model)
        if key == (best.arm, best.model):
            continue
        deltas[key] = paired_delta(reward_maps[key], best_map)

    sweep = []
    for floor in floors_pts:
        eligible = [
            p
            for p in points
            if (p.arm, p.model) == (best.arm, best.model)
            or deltas[(p.arm, p.model)].mean_delta >= -floor / 100.0
        ]
        pick = min(eligible, key=lambda p: p.p50_task_s)
        delta = deltas.get((pick.arm, pick.model))
        sweep.append(
            {
                "quality_floor_pts_below_best": floor,
                "best_config": _label(best),
                "eligible_configs": len(eligible),
                "pick": pick.model_dump(),
                "paired_delta_vs_best": delta.model_dump() if delta is not None else None,
            }
        )
    result = {
        "corner": "latency-max",
        "definition": (
            "mountable config (candidate model x compression arm; routed configs join once "
            "the master's per-arm fits land) minimizing per-task p50 model seconds subject "
            "to a paired per-scenario reward delta vs the best measured config no worse "
            "than -floor"
        ),
        "limitation": (
            "offline mount choice, not per-query latency routing (no online latency term in "
            "knn_decision); DECISIONS.md 2026-07-27 three-corners entry"
        ),
        "sweep": sweep,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "latency_max_corner.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def synthetic_snapshots() -> list[ArmSnapshot]:
    """A tiny deterministic fake grid that exercises every code path, zero spend.

    Covers: two arms (one compressed), a slow-but-good model, a fast-but-weak model, a
    serverless model with a 51 s cold first call, blank-retry calls (fast + empty reply),
    unscored rows, and a compressed arm's compressor wall/cost fields.
    """
    rng = random.Random(7)
    pool = [
        PoolEntry(
            name=name, kind="openai", model=name, endpoint="https://synthetic.invalid",
            input_per_mtok=1.0, output_per_mtok=2.0,
        )
        for name in ("slow-good", "fast-weak", K3_NAME)
    ]
    quality = {"slow-good": 0.9, "fast-weak": 0.55, K3_NAME: 0.85}
    pace = {"slow-good": 6.0, "fast-weak": 1.2, K3_NAME: 2.5}

    def episode(model: str, sid: str, ep: int, arm: str) -> ScenarioOutcome:
        steps = rng.randint(4, 9)
        calls = [max(0.2, rng.gauss(pace[model], 0.4)) for _ in range(steps)]
        replies = ["ok"] * steps
        if model == K3_NAME and sid.endswith("0") and ep == 0:
            calls[0] = K3_COLD_START_S
        if model == "fast-weak" and ep == 1:
            calls.insert(1, 0.3)  # a blank retry: fast call, empty reply
            replies.insert(1, "")
        reward = min(1.0, max(0.0, rng.gauss(quality[model], 0.15)))
        unscored = model == "fast-weak" and sid.endswith("3") and ep == 0
        compressed = arm == "truncate"
        return ScenarioOutcome(
            scenario_id=sid, task=f"task {sid}", model=model, episode=ep,
            reward=None if unscored else reward,
            success=(not unscored) and reward >= 0.7,
            error="synthetic transient fault" if unscored else None,
            steps=steps, call_seconds=calls, replies=replies,
            cost_usd=0.01 * steps,
            compressor_id="truncate" if compressed else "",
            compressor_version="v1" if compressed else "",
            aggressiveness=0.33 if compressed else 0.0,
            compressor_latency_s=0.07 * steps if compressed else 0.0,
            compressor_cost_usd=0.0002 * steps if compressed else 0.0,
        )

    return [
        ArmSnapshot(
            name=arm_name,
            matrix=OutcomeMatrix(
                pool=pool,
                outcomes=[
                    episode(entry.name, f"scen-{i}", ep, arm_name)
                    for entry in pool
                    for i in range(6)
                    for ep in range(2)
                ],
            ),
            status="synthetic smoke",
        )
        for arm_name in ("identity", "truncate")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "figures")
    parser.add_argument(
        "--quality-floors", type=float, nargs="+", default=[2.0, 5.0, 10.0],
        help="corner sensitivity: paired reward points below the best measured config",
    )
    parser.add_argument("--synthetic", action="store_true", help="schema smoke, no real data")
    args = parser.parse_args()

    apply_style()
    out = args.out / "synthetic" if args.synthetic else args.out
    snapshots = synthetic_snapshots() if args.synthetic else all_arm_snapshots()
    status = (
        "; ".join(f"{s.name}: {s.status}" for s in snapshots)
        if snapshots
        else "no grid data on disk yet"
    )
    print(f"grid arms -> {status}")

    points = config_points(snapshots) if snapshots else []
    if points:
        dump_points(points, out / "config_points.json")
        fig_latency_per_config(points, status, out)
        fig_frontier(points, snapshots, status, out)
        fig_cold_start(snapshots, status, out)
        corner = latency_max_corner(points, snapshots, tuple(args.quality_floors), out)
        print("\nLATENCY-MAX corner (per quality floor):")
        for entry in corner["sweep"]:
            pick = entry["pick"]
            cost = pick["cost_per_completed_usd"]
            delta = entry["paired_delta_vs_best"]
            noise = " (within noise floor of best)" if delta and delta["within_noise_floor"] else ""
            print(
                f"  floor -{entry['quality_floor_pts_below_best']:.0f}pts: "
                f"{pick['model']} @ {pick['arm']}  "
                f"p50 {pick['p50_task_s']:.1f}s/task, reward {pick['mean_reward']:.3f}{noise}, "
                f"{'$' + format(cost, '.3f') + '/task' if cost is not None else 'cost undefined'}"
                f"  ({entry['eligible_configs']} eligible)"
            )
    else:
        print("no config points yet; skipping grid figures (re-run when chunks land)")

    if not args.synthetic:
        # The shared stage chart is the canonical implementation, rendered through this
        # corner's lens; a second hand-rolled stage chart could disagree on the numbers.
        from ablation_chart import build_shared_chart_data, render_training_stage_chart

        rendered = render_training_stage_chart(
            build_shared_chart_data(), out / "training-stage", lens="latency"
        )
        print(f"\nshared stage chart (latency lens) -> {rendered}")
    else:
        print("synthetic mode: stage chart skipped (cycle artifacts are real data)")

    print(f"\nwrote figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
