"""Per-model teacher headroom over the cheapest candidate: the distill go/no-go made visible.

Cycle 1's lesson is the reason this table exists: warmup distillation from a teacher with 1.6
points of headroom measured nothing (gate refused, correctly). Teacher search asks the same
question ahead of spend, over matrices the grid already bought: which pool models sit
measurably ABOVE the cheap end of the pool, with how much gain, at what cost?

HANDOFF CONTRACT: the product's teacher-search verdict is becoming a repo function
(`wmo.optimize.teacher`, branch jt/teacher-gate) computed over these same matrices. When it
lands on this branch, this module's computation is REPLACED by consuming that function's
verdict artifact and the quality corner's figure becomes rendering only. Until then this
implements the same reading (per-model PAIRED gain over the cheapest candidate, CI-guarded)
through the corners' binding stats so the chart exists the moment the grid merges. A
disagreement between the two implementations is a bug in one of them, never a second opinion.

Baseline discipline: "cheapest" is the lowest cache-adjusted effective cost per completed
task among candidates that completed anything (the scorecard's own accounting), which proxies
the student tier until the student's own cells merge into the matrices; pass `baseline`
explicitly at that point. Run this on the IDENTITY arm: a compressed arm's rewards answer a
different question.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

import data
import stats
from wmo.optimize.scorecard import effective_cost_per_completed_task, rows_for_model

if TYPE_CHECKING:
    from wmo.optimize.outcomes import OutcomeMatrix

Verdict = Literal[
    "baseline", "headroom", "below baseline", "within noise floor", "no measurable effect"
]


class TeacherVerdictRow(BaseModel):
    """One candidate's gain over the baseline, CI-guarded, with its own quality and cost."""

    model: str
    baseline: str
    mean_reward: float
    n_scenarios: int
    delta: stats.PairedDelta
    cost_per_completed_task_usd: float | None
    verdict: Verdict


def cheapest_scored_candidate(matrix: OutcomeMatrix) -> str:
    """The candidate with the lowest effective cost per completed task; name ties break it.

    Candidates that completed nothing have no cost per completed task and cannot anchor the
    table (an undefined baseline would make every gain undefined with it).

    Raises:
        ValueError: when no candidate completed any task; there is no baseline to state gains
            against, so the table should not render at all.
    """
    priced: list[tuple[float, str]] = []
    for model in matrix.model_names():
        cost = effective_cost_per_completed_task(rows_for_model(matrix, model))
        if cost.cost_per_completed_task_usd is not None:
            priced.append((cost.cost_per_completed_task_usd, model))
    if not priced:
        raise ValueError(
            "no candidate in this matrix completed a single task, so there is no cheapest "
            "candidate to anchor teacher gains against; check the matrix for unscored rows"
        )
    return min(priced)[1]


def teacher_verdict_rows(
    matrix: OutcomeMatrix, *, baseline: str | None = None
) -> list[TeacherVerdictRow]:
    """Every candidate's paired gain over the baseline, sorted best gain first.

    The verdict applies the corners' shared rule (stats.PairedDelta.verdict): "headroom"
    requires a CI excluding zero AND a mean past the noise floor, in the positive direction.
    Everything else is stated as what it is, never rounded up to a go signal.
    """
    anchor_model = baseline or cheapest_scored_candidate(matrix)
    anchor_rewards = data.rewards_by_scenario(matrix.outcomes, model=anchor_model)
    rows: list[TeacherVerdictRow] = []
    for model in matrix.model_names():
        rewards = data.rewards_by_scenario(matrix.outcomes, model=model)
        if not rewards:
            continue  # nothing scored: no quality statement to make about this candidate
        quality = stats.mean_with_ci(rewards)
        delta = stats.paired_delta(rewards, anchor_rewards)
        cost = effective_cost_per_completed_task(rows_for_model(matrix, model))
        rows.append(
            TeacherVerdictRow(
                model=model,
                baseline=anchor_model,
                mean_reward=quality.mean,
                n_scenarios=quality.n_scenarios,
                delta=delta,
                cost_per_completed_task_usd=cost.cost_per_completed_task_usd,
                verdict=_verdict(model, anchor_model, delta),
            )
        )
    rows.sort(key=lambda row: row.delta.mean_delta, reverse=True)
    return rows


def _verdict(model: str, baseline: str, delta: stats.PairedDelta) -> Verdict:
    """Map the shared delta verdict onto the teacher question's vocabulary."""
    if model == baseline:
        return "baseline"
    if delta.verdict == "measurable":
        return "headroom" if delta.mean_delta > 0 else "below baseline"
    if delta.verdict == "within_noise_floor":
        return "within noise floor"
    return "no measurable effect"
