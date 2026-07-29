"""Latency metrics for the corner analyses: the two clocks and the #295 rule, in code form.

Owned by the latency corner chat (the charter's "first chat to need a helper writes it");
the cost and quality chats consume these to annotate their own figures and extend them here,
never fork them. Pure stdlib plus the wmo package; no viz dependency.

Two clocks appear in the corner data sources and are never pooled or compared:

- GRID rows (`ScenarioOutcome.call_seconds`): wall seconds of the candidate's MODEL calls,
  environment and judge time excluded (`wmo/optimize/outcomes.py`). Per-TASK seconds sum a
  scored row's calls plus its compressor wall time, matching the scorecard's `LatencyBlock`
  contract (per-task model seconds, optimizer overhead included).
- CYCLE rows (`duration_s`): whole-episode wall clock of a REAL tau2 episode, user simulator
  and environment time included. An episode-wall p50 says nothing about a grid per-task p50;
  every chart labels which clock it carries.

The #295 blank-retry rule (`wmo/optimize/report.py::_productive_call_seconds`, kept private
there so the contract is restated here): a call the provider answered with blank text
returns fast and carries no work, so PER-CALL statistics drop it (a model that blanks often
must not look fast; kimi-k2.6 blanked on 24% of calls in the original capture). PER-TASK
statistics keep every call: a blank retry's wall time is real time the task took, and the
scorecard sums it. Cost keeps blank calls everywhere, because they were really paid for.
"""

from __future__ import annotations

from statistics import median, quantiles
from typing import TYPE_CHECKING

from pydantic import BaseModel

from wmo.optimize.scorecard import RowOverhead, effective_cost_per_completed_task

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from data import ArmSnapshot
    from wmo.optimize.outcomes import ScenarioOutcome

# Fixed arm-identity colors (palette.py's series-identity principle applied to the grid's
# compression arms; line-safe accents only). llmlingua2-endpoint IS the compaction lever, so
# it wears +compaction's red; truncate is its ratio-matched dumb control; identity is the
# uncompressed baseline. Hex literals equal palette.py's BLUE/PURPLE/RED rather than imports,
# because importing palette pulls matplotlib and this module stays viz-free (stats.py's
# policy). Candidate to move INTO palette.py once its in-flight helper edits land.
ARM_COLORS: dict[str, str] = {
    "identity": "#0070f3",
    "truncate": "#7928ca",
    "llmlingua2-endpoint": "#ee0000",
}


def per_task_model_seconds(row: ScenarioOutcome) -> float:
    """One scored row's per-task model seconds: every call plus its compressor wall time.

    Matches the scorecard's `LatencyBlock` contract exactly, including keeping blank-retry
    calls (their wall time is real time the task took). Environment and judge time are
    excluded by `call_seconds`' own contract, so this understates operator wall clock.
    """
    return sum(row.call_seconds) + row.compressor_latency_s


def productive_call_seconds(rows: Sequence[ScenarioOutcome]) -> list[float]:
    """Per-CALL seconds with blank-answered calls dropped, the #295 rule (module docstring).

    `call_seconds` and `replies` are appended in lockstep by the closed-loop timed provider,
    so a blank reply is identifiable by index; producers that do not promise the pairing show
    up as unequal lengths and keep every call. Compressor wall time is NOT added here: it is
    per-episode, not per-call, and belongs to per-task statistics only.
    """
    seconds: list[float] = []
    for row in rows:
        if len(row.replies) != len(row.call_seconds):
            seconds.extend(row.call_seconds)
            continue
        seconds.extend(
            value
            for value, reply in zip(row.call_seconds, row.replies, strict=True)
            if reply.strip()
        )
    return seconds


def p50_p95(values: Sequence[float]) -> tuple[float, float]:
    """Median and inclusive-method p95, the scorecard's quantile convention.

    Inclusive because it is bounded by the observed data: at ablation sample sizes the
    default exclusive method extrapolates a tail longer than anything measured (scorecard
    docstring). At this grid's n (~40 tasks/config), p95 is roughly the second-slowest task:
    report it, never let it carry a headline alone.
    """
    if not values:
        raise ValueError("no values to take quantiles of")
    if len(values) == 1:
        return values[0], values[0]
    return median(values), quantiles(values, n=20, method="inclusive")[-1]


def first_vs_warm(rows: Sequence[ScenarioOutcome]) -> tuple[list[float], list[float]]:
    """Each scored episode's FIRST call versus its remaining calls, for cold-start reading.

    A serverless candidate's cold start (kimi-k3 was measured at 51 s on its first call,
    run_tau_grid.py docstring) lands in some episode's first call; warm behaviour is the
    rest. This is a witness, not a clean measurement: only the first call after provider
    idle is truly cold, and idle gaps between grid chunks are not recorded.
    """
    first: list[float] = []
    warm: list[float] = []
    for row in rows:
        if row.reward is None or not row.call_seconds:
            continue
        first.append(row.call_seconds[0])
        warm.extend(row.call_seconds[1:])
    return first, warm


class ConfigPoint(BaseModel):
    """One measured (candidate model x compression arm) config on all three objectives.

    The unit the corners are read off: a single-model config plus its arm's compression is
    mountable today (a fixed policy plus a `CompressionConfig`); routed configs join once the
    joint-tau master fits per-arm policies (never fitted by a corner chat). Quality is
    averaged per scenario then across (the scorecard convention); cost is cache-adjusted
    effective cost per completed task with compressor overhead folded in (public scorecard
    API); latency carries the grid clock only, both statistics, never the cycle clock.
    """

    arm: str
    model: str
    n_scenarios: int
    n_scored: int
    n_unscored: int
    n_completed: int
    mean_reward: float
    success_rate: float
    cost_per_completed_usd: float | None
    provider_cost_usd: float
    overhead_cost_usd: float
    p50_task_s: float
    p95_task_s: float
    p50_call_s: float
    p95_call_s: float
    first_call_max_s: float
    mean_steps: float
    provenance: str = "wm_simulated"


def config_points(snapshots: Sequence[ArmSnapshot]) -> list[ConfigPoint]:
    """Every (model x arm) with at least one scored row, aggregated on all three objectives."""
    points: list[ConfigPoint] = []
    for snapshot in snapshots:
        for model in snapshot.matrix.model_names():
            rows = [o for o in snapshot.matrix.outcomes if o.model == model]
            scored = [o for o in rows if o.reward is not None]
            if not scored:
                continue

            by_scenario: dict[str, list[ScenarioOutcome]] = {}
            for row in scored:
                by_scenario.setdefault(row.scenario_id, []).append(row)
            scenario_rewards = [
                sum(r.reward for r in group if r.reward is not None) / len(group)
                for group in by_scenario.values()
            ]
            scenario_success = [
                sum(1.0 for r in group if r.success) / len(group)
                for group in by_scenario.values()
            ]

            overheads = [
                RowOverhead(
                    scenario_id=row.scenario_id,
                    model=row.model,
                    episode=row.episode,
                    component="compressor",
                    cost_usd=row.compressor_cost_usd,
                    latency_s=row.compressor_latency_s,
                )
                for row in rows
                if row.compressor_id
            ]
            cost = effective_cost_per_completed_task(rows, overheads=overheads)

            p50_task, p95_task = p50_p95([per_task_model_seconds(row) for row in scored])
            calls = productive_call_seconds(scored)
            p50_call, p95_call = p50_p95(calls) if calls else (0.0, 0.0)
            first, _warm = first_vs_warm(scored)

            points.append(
                ConfigPoint(
                    arm=snapshot.name,
                    model=model,
                    n_scenarios=len(by_scenario),
                    n_scored=len(scored),
                    n_unscored=len(rows) - len(scored),
                    n_completed=cost.n_completed,
                    mean_reward=sum(scenario_rewards) / len(scenario_rewards),
                    success_rate=sum(scenario_success) / len(scenario_success),
                    cost_per_completed_usd=cost.cost_per_completed_task_usd,
                    provider_cost_usd=cost.provider_cost_usd,
                    overhead_cost_usd=cost.overhead_cost_usd,
                    p50_task_s=p50_task,
                    p95_task_s=p95_task,
                    p50_call_s=p50_call,
                    p95_call_s=p95_call,
                    first_call_max_s=max(first) if first else 0.0,
                    mean_steps=sum(r.steps for r in scored) / len(scored),
                )
            )
    return points


def dump_points(points: Sequence[ConfigPoint], path: Path) -> None:
    """The computed table as JSON, so siblings and reviewers can audit numbers off-figure."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([p.model_dump() for p in points], indent=2) + "\n", encoding="utf-8"
    )
