"""The cost/quality Pareto curve over an outcome matrix: the product's central artifact.

The pipeline's promise is not one number, it is a CURVE: every measured way to serve the
workload, placed on (effective cost per completed task, quality), with the non-dominated
points marked so an operator can pick any point and mount it. This module computes that
curve ONE way for every surface that shows it: the endpoint report, the admin frontend's
graph, the live estimate a run emits while a sweep is still landing cells (D-RUNS stage
artifacts), and the research analyses. Two surfaces disagreeing about the same curve is the
two-truths bug this module exists to prevent.

Costs come from `wmo.optimize.scorecard.effective_cost_per_completed_task` (the D-COMPRESS
accounting rule: cache-adjusted, per COMPLETED task, unscored spend excluded and counted)
and nowhere else. Quality is the scenario-mean reward, matching the scorecard's averaging.
A point computed from a partial matrix says so: `n_scenarios`/`n_scored` travel on every
point and `ParetoCurve.complete` is False whenever any candidate has unscored cells, so a
mid-sweep estimate can never be mistaken for the final curve.

The routed points are POLICY REPLAYS (`rows_for_policy`), measured on the scenarios the
caller passes (a report passes the fit's held-out band; a live estimate passes everything
scored so far). `recommended` is the point the product would mount today: the routed point
at the balanced dial when a policy is given, else the guarded fit's own answer (the best
single model on the matrix). It is a default, not a verdict; the curve exists so an
operator can choose differently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from wmo.optimize.outcomes import OutcomeMatrix
from wmo.optimize.scorecard import (
    DEFAULT_COMPLETION,
    CompletionRule,
    effective_cost_per_completed_task,
    rows_for_model,
    rows_for_policy,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wmo.optimize.policy import RoutingPolicy
    from wmo.providers.base import Embedder

PointKind = Literal["model", "routed"]

# The curve's artifact name, written beside `report.json` by the report writers and loaded
# from the model directory at serving mount (GET /config carries it to the platform).
PARETO_FILENAME = "pareto.json"

# The two ways a curve's rewards can have been produced. Consumers refuse to blend them, which
# is exactly why they are named constants rather than literals repeated at each call site: a
# curve mislabelled here presents real measurements as a simulation, or the reverse.
WM_SIMULATED = "wm_simulated"
REAL_EPISODE = "real_episode"

# What scores a world-model sweep: the model's own verifier. A real-benchmark matrix was scored
# by the benchmark instead and has to say which one.
DEFAULT_WM_JUDGE = "world-model verifier"

# Frontier eligibility: a point must have scored rows on at least this fraction of the
# band's best scenario coverage. Without it, an arm that loses most episodes to its own
# failures is judged only on the survivors - on a real tau2 grid, qwen3.5-9b scored 5 of 12
# episodes, aced them, and "dominated" the anchor measured on all 12. Survivorship is not
# dominance, so under-covered points stay plotted and labeled but are never marked frontier
# nor eligible for `recommended`. On a matrix with full coverage nothing changes.
FRONTIER_COVERAGE_FRACTION = 0.9
FRONTIER_RULE = (
    "frontier eligibility: scored-scenario coverage >= 90% of the band's best; "
    "under-covered points are plotted but never frontier nor recommended"
)


class ParetoPoint(BaseModel):
    """One measured way to serve the workload, on all three objectives."""

    model_config = ConfigDict(frozen=True)

    id: str  # pool model name, or "routed@<dial>"
    kind: PointKind
    label: str
    cost_per_completed_task_usd: float | None  # None: nothing completed, unplottable
    mean_reward: float
    task_success_rate: float
    latency_p50_s: float
    n_scenarios: int
    n_scored: int
    n_excluded: int  # unscored episodes behind this point (infrastructure, not zeros)
    on_frontier: bool = False
    # False when coverage is too thin for this point's axes to be compared against the
    # band's (see FRONTIER_RULE); such a point can neither hold nor take the frontier.
    frontier_eligible: bool = True
    dial: float | None = None  # routed points: the cost_quality position replayed
    mix: dict[str, int] = Field(default_factory=dict)  # routed points: scenarios per model


class ParetoCurve(BaseModel):
    """The measured curve plus the honesty fields no rendering may drop.

    `complete` is False while any candidate still has unscored cells: early cells are not a
    random sample of the workload, so an incomplete curve is an ESTIMATE and every consumer
    must label it as one. `recommended` names a point id from `points`.
    """

    points: list[ParetoPoint]
    recommended: str | None
    complete: bool
    n_scenarios: int
    provenance: str  # e.g. "wm_simulated"; consumers print it next to every rendering
    judge: str
    # The eligibility rule the frontier flags were computed under, stated on the artifact
    # so a renderer can show WHY an under-covered point is unmarked. Defaulted so curves
    # written before the rule existed still parse.
    frontier_rule: str = FRONTIER_RULE


def pareto_curve(
    matrix: OutcomeMatrix,
    *,
    judge: str,
    provenance: str = WM_SIMULATED,
    policy: RoutingPolicy | None = None,
    dials: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    scenario_ids: Sequence[str] | None = None,
    embedder: Embedder | None = None,
    completion: CompletionRule = DEFAULT_COMPLETION,
) -> ParetoCurve:
    """Compute the cost/quality curve over `matrix`, optionally with a policy's dial points.

    Args:
        matrix: the measured pool x scenario grid, complete or mid-sweep.
        judge: the verifier that produced the rewards; printed on every rendering.
        provenance: wm_simulated or real_episode; never blended by consumers.
        policy: a fitted routing policy whose dial detents become routed points. The caller
            owns split discipline: pass `scenario_ids` the fit never saw (a report's
            held-out band) or accept in-sample routed points labeled by its own context.
        dials: the cost_quality detents to replay when `policy` is given.
        scenario_ids: restrict the curve to these scenarios (default: every scenario with
            any scored row). Model points and routed points share the restriction so the
            curve is one comparison.
        embedder: one embedder for the whole replay (see `rows_for_policy`).
        completion: what counts as a completed task.

    Raises:
        ValueError: when no scenario has a scored row, or `scenario_ids` names scenarios
            the matrix never measured.
    """
    from wmo.optimize.knn import apply_cost_quality

    ids = list(scenario_ids) if scenario_ids is not None else matrix.scenario_ids()
    known = set(matrix.scenario_ids())
    ghosts = [sid for sid in ids if sid not in known]
    if ghosts:
        raise ValueError(
            f"scenario_ids name {len(ghosts)} scenario(s) this matrix never measured "
            f"(first: {ghosts[0]!r}); a curve over unmeasured scenarios would be invented"
        )
    wanted = set(ids)

    points: list[ParetoPoint] = []
    any_unscored = False
    for model in matrix.model_names():
        rows = [o for o in rows_for_model(matrix, model) if o.scenario_id in wanted]
        point = _point(model, "model", model, rows, completion)
        if point is None:
            any_unscored = True  # a candidate with no scored row yet is missing, not absent
            continue
        any_unscored = any_unscored or point.n_excluded > 0
        points.append(point)
    if not points:
        raise ValueError(
            "no pool model has a scored row on the requested scenarios; there is no curve "
            "to compute yet"
        )

    if policy is not None:
        shared = embedder or _replay_embedder(policy)
        variants: list[tuple[float | None, RoutingPolicy]]
        if policy.kind == "linear":
            variants = [(None, policy)]
        else:
            variants = [(dial, apply_cost_quality(policy, dial)) for dial in dials]
        for dial, variant in variants:
            rows = rows_for_policy(
                matrix, variant, ids=ids, embedder=shared
            )
            mix: dict[str, int] = {}
            for sid in ids:
                for chosen in {row.model for row in rows if row.scenario_id == sid}:
                    mix[chosen] = mix.get(chosen, 0) + 1
            point_id = "routed" if dial is None else f"routed@{dial:g}"
            label = "routed" if dial is None else f"routed (dial {dial:g})"
            point = _point(
                point_id,
                "routed",
                label,
                rows,
                completion,
            )
            if point is not None:
                points.append(point.model_copy(update={"dial": dial, "mix": mix}))

    flagged = _mark_frontier(points)
    return ParetoCurve(
        points=flagged,
        recommended=_recommended(flagged, policy),
        complete=not any_unscored,
        n_scenarios=len(ids),
        provenance=provenance,
        judge=judge,
    )


def held_out_curve(
    matrix: OutcomeMatrix,
    policy: RoutingPolicy,
    *,
    judge: str,
    provenance: str = WM_SIMULATED,
    embedder: Embedder | None = None,
) -> ParetoCurve:
    """The curve a report ships: routed points on the fit's held-out band only.

    Mirrors `wmo.optimize.report.build_report`'s split discipline: `fit_scenario_ids` are
    excluded so no routed point is graded on the fit's own training data. A policy fitted
    on EVERY scenario has no held-out band; the curve then carries the model points alone
    (over all scenarios) rather than in-sample routed points dressed as measurements.

    A STATIC policy (a pin) has no dial to replay, but the WORKLOAD's frontier
    exists regardless of what serves it - and a pinned endpoint is exactly the case where
    an operator most wants to see what else was measured. The curve then carries the model
    points over the same held-out band a report describes (the full matrix when the policy
    records no fit split), with `recommended` naming what the product mounts today - the
    policy's own default model - but only while that point honors the curve's own frontier
    rule: a pin whose coverage the rule disqualifies must not be recommended by the very
    artifact that states the rule (bench-defaults/tau finding 11, 2026-07-29). A rank
    policy is deliberately NOT short-circuited: it routes at serve time, so a curve with
    no routed points and its degenerate-input fallback as `recommended` would misdescribe
    it - the dial replay refuses rank policies and the report writer says the curve was
    skipped, the honest status quo.
    """
    held_out = [sid for sid in matrix.scenario_ids() if sid not in set(policy.fit_scenario_ids)]
    if policy.kind == "static":
        curve = pareto_curve(
            matrix,
            judge=judge,
            provenance=provenance,
            scenario_ids=held_out or None,
        )
        pinned = next((p for p in curve.points if p.id == policy.default_model), None)
        if pinned is not None and pinned.frontier_eligible:
            return curve.model_copy(update={"recommended": policy.default_model})
        return curve
    if not held_out:
        return pareto_curve(matrix, judge=judge, provenance=provenance)
    return pareto_curve(
        matrix,
        judge=judge,
        provenance=provenance,
        policy=policy,
        scenario_ids=held_out,
        embedder=embedder,
    )


def _replay_embedder(policy: RoutingPolicy) -> Embedder:
    """The function the policy's bank geometry was fitted with.

    A compressed-fit policy embeds COMPRESSED text (C2's representation-consistency rule,
    exactly as `build_report` replays); querying its bank with raw vectors trips the
    novelty floor and mismeasures the routed points.
    """
    from wmo.optimize.compression import CompressingEmbedder

    built = policy.embedder.build()
    if policy.fit_compression is not None:
        return CompressingEmbedder(built, policy.fit_compression)
    return built


def _point(
    point_id: str,
    kind: PointKind,
    label: str,
    rows: list,
    completion: CompletionRule,
) -> ParetoPoint | None:
    """Aggregate one config's rows into a point; None when nothing is scored yet."""
    scored = [row for row in rows if row.reward is not None]
    if not scored:
        return None
    cost = effective_cost_per_completed_task(rows, completion=completion)
    by_scenario: dict[str, list[float]] = {}
    success: dict[str, list[float]] = {}
    per_task_seconds: list[float] = []
    for row in scored:
        by_scenario.setdefault(row.scenario_id, []).append(row.reward)
        success.setdefault(row.scenario_id, []).append(1.0 if completion.completed(row) else 0.0)
        per_task_seconds.append(sum(row.call_seconds))
    scenario_means = [sum(v) / len(v) for v in by_scenario.values()]
    success_means = [sum(v) / len(v) for v in success.values()]
    per_task_seconds.sort()
    return ParetoPoint(
        id=point_id,
        kind=kind,
        label=label,
        cost_per_completed_task_usd=cost.cost_per_completed_task_usd,
        mean_reward=sum(scenario_means) / len(scenario_means),
        task_success_rate=sum(success_means) / len(success_means),
        latency_p50_s=per_task_seconds[len(per_task_seconds) // 2],
        n_scenarios=len(by_scenario),
        n_scored=len(scored),
        n_excluded=cost.n_excluded,
    )


def _mark_frontier(points: list[ParetoPoint]) -> list[ParetoPoint]:
    """Flag the non-dominated points on (cost down, reward up); ties stay on the frontier.

    A point with no defined cost cannot be placed on the cost axis and is never on the
    frontier (it stays in `points` so a renderer can show it as unplaced rather than
    dropping it silently); its `frontier_eligible` still reports coverage honestly, so a
    renderer never explains an unplaced point with the coverage rule's copy. A point whose
    scored-scenario coverage falls below FRONTIER_COVERAGE_FRACTION of the best MODEL
    point's is excluded from the dominance comparison entirely - both as a candidate and
    as a dominator - because its axes describe the episodes it survived, not the band. The
    floor deliberately ignores routed points: a routed point's coverage is the union over
    the models it picked, and letting the union raise the floor would disqualify every
    model point on the router's coverage advantage rather than on measurement.
    """
    model_coverage = max((p.n_scenarios for p in points if p.kind == "model"), default=0)
    best_coverage = model_coverage or max((p.n_scenarios for p in points), default=0)
    floor = FRONTIER_COVERAGE_FRACTION * best_coverage
    eligible_ids = {p.id for p in points if p.n_scenarios >= floor}
    contenders = [
        (p, p.cost_per_completed_task_usd)
        for p in points
        if p.cost_per_completed_task_usd is not None and p.id in eligible_ids
    ]

    def dominated(p: ParetoPoint, cost: float) -> bool:
        return any(
            other_cost <= cost
            and o.mean_reward >= p.mean_reward
            and (other_cost < cost or o.mean_reward > p.mean_reward)
            for o, other_cost in contenders
            if o is not p
        )

    flagged = {p.id: not dominated(p, cost) for p, cost in contenders}
    return [
        p.model_copy(
            update={
                "on_frontier": flagged.get(p.id, False),
                "frontier_eligible": p.id in eligible_ids,
            }
        )
        for p in points
    ]


def _recommended(points: list[ParetoPoint], policy: RoutingPolicy | None) -> str | None:
    """The point the product would mount today.

    With a policy: its balanced-dial routed point (the shipped default detent), unless
    the curve's own frontier rule disqualified it - the artifact must never recommend a
    point it marks ineligible. Without one: the frontier's best-quality point, which is
    what the guarded fit would discover and fall back to. None only when nothing is
    placeable.
    """
    if policy is not None:
        balanced_dial = None if policy.kind == "linear" else 0.25
        balanced = next(
            (p for p in points if p.kind == "routed" and p.dial == balanced_dial),
            None,
        )
        if (
            balanced is not None
            and balanced.cost_per_completed_task_usd is not None
            and balanced.frontier_eligible
        ):
            return balanced.id
    frontier = [p for p in points if p.on_frontier]
    if not frontier:
        return None
    return max(frontier, key=lambda p: p.mean_reward).id
