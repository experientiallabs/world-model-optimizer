"""The ONE shared runner for the corner analyses (charter Amendment, Silen directive).

Per-corner pipelines are retired. This runner loads matrices, fits, and episode rows ONCE,
computes the full three-objective dataset ONCE (cost through `wmo.optimize.scorecard` only,
quality evidence through `common/stats`, the distillation verdict through
`wmo.optimize.teacher.select_teacher`, never hand math), and renders per-lens figures from
declarative lens specs. A number that appears in two corners comes from the same
computation by construction.

Lens specs are declarative: a corner's subdirectory holds only its lens spec (`lens.py`
exporting `LENS`) and its findings prose. The suspended quality and latency corners have no
lens spec yet (Amendment 2 froze their directories mid-refactor); their lenses stay
renderable through FROZEN_LENSES, which delegates to their frozen standalone scripts
unchanged. When those axes resume, their specs port onto dataset-native figure kinds and
the delegation entries are deleted.

Run from the repo root (matplotlib comes from the viz extra):

    uv run python .agents/docs/research/corners/common/build_corners.py --lens cost

Honesty rules inherited from common/README.md: every figure reports all three objectives or
names the ones its source cannot supply; provenance and judge on every footnote; partial
snapshots carry their completeness status onto the figure; pending data is named, never
silently absent. Zero LLM spend: this runner only reads artifacts already on disk.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from pydantic import BaseModel, ConfigDict, Field

from wmo.core.types import JsonObject
from wmo.optimize.compression import CompressingEmbedder
from wmo.optimize.knn import COST_QUALITY_ANCHORS, apply_cost_quality, fit_knn_policy
from wmo.optimize.policy import RoutingPolicy
from wmo.optimize.scorecard import (
    Arm,
    ConditionLabel,
    RowOverhead,
    Scorecard,
    build_scorecard,
    rows_for_model,
    rows_for_policy,
)
from wmo.optimize.teacher import TeacherSearchVerdict, select_teacher

COMMON = Path(__file__).resolve().parent
CORNERS = COMMON.parent
sys.path.insert(0, str(COMMON))

import ablation_chart  # noqa: E402
from data import (  # noqa: E402
    IDENTITY_ARM,
    ArmSnapshot,
    all_arm_snapshots,
    rewards_by_scenario,
)
from palette import (  # noqa: E402
    BLUE,
    MUTED,
    NOISE_BAND_ALPHA,
    NOISE_BAND_COLOR,
    PURPLE,
    RED,
    apply_style,
    footnote,
    save_fig,
)
from stats import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    NOISE_FLOOR_REWARD,
    PairedDelta,
    paired_delta,
)

DEFAULT_ANCHOR = "fable-5"

# The WM episodes' scorer. CORRECTED 2026-07-28: the program had quoted "rubric-v2
# (opus-4-8)", but the built model's config.toml records judge_model =
# us.anthropic.claude-haiku-4-5..., so every grid-c2 cell was judged by HAIKU-4-5. The
# label now states the fact. (wmo PR #366 changed the DEFAULT judge to opus-5 for future
# builds; grid-c2's matrices keep this label forever - a judge change is a new fidelity
# era and rewards never compare across judges.)
GRID_JUDGE = "rubric-v2 (haiku-4-5; config-verified, mislabeled opus-4-8 before 2026-07-28)"

# Fixed arm-identity colors (palette rules: amber/teal never on small marks).
# llmlingua2-endpoint wears red because it IS the compaction lever (SERIES_COLORS).
ARM_COLORS = {IDENTITY_ARM: BLUE, "truncate": PURPLE, "llmlingua2-endpoint": RED}

NOISE_FLOOR_POINTS = NOISE_FLOOR_REWARD * 100

# Dial positions replayed for the tau dial curve: the same five measured detents the ours9
# anchors use, so the two panels are comparable in shape (never in numbers).
ROUTED_DIALS = (0.0, 0.25, 0.5, 0.75, 1.0)

# text-embedding-3-large list rate, used ONLY to estimate the router's per-query embedding
# overhead on routed rows (labeled estimate; the pool prices carry no embedder entry).
EMBED_LIST_USD_PER_MTOK = 0.13


class CostDeltaCI(BaseModel):
    """Cluster-bootstrap CI on an effective-cost delta, following the stats conventions.

    Scenarios are the resampling unit (episodes within one are correlated), seeded so a
    rerun reproduces its numbers. A resample where either side completes zero tasks has no
    defined cost ratio; those are dropped and COUNTED, because at small n they are the
    signal that the ratio itself is fragile, not an inconvenience to hide.
    """

    delta_percent: float
    ci_low: float
    ci_high: float
    n_scenarios: int
    resamples: int
    undefined_resamples: int


def _scenario_cost(rows: list, overheads: list[RowOverhead]) -> dict[str, tuple[float, int]]:
    """Per scenario: (spend incl. overhead over scored rows, completed count)."""
    by_key: dict[tuple[str, str, int], float] = {}
    for o in overheads:
        key = (o.scenario_id, o.model, o.episode)
        by_key[key] = by_key.get(key, 0.0) + o.cost_usd
    out: dict[str, tuple[float, int]] = {}
    for row in rows:
        if row.reward is None:
            continue
        cost = row.cost_usd + by_key.get((row.scenario_id, row.model, row.episode), 0.0)
        spent, done = out.get(row.scenario_id, (0.0, 0))
        out[row.scenario_id] = (spent + cost, done + int(row.success))
    return out


def cost_delta_ci(
    arm: dict[str, tuple[float, int]],
    anchor: dict[str, tuple[float, int]],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> CostDeltaCI | None:
    """Paired cluster bootstrap of the effective-cost-per-completed-task percent delta.

    None when the point estimate itself is undefined (a side completed nothing overall).
    """
    from random import Random

    ids = sorted(set(arm) & set(anchor))
    if not ids:
        return None

    def ratio(sample: list[str]) -> float | None:
        arm_cost = sum(arm[s][0] for s in sample)
        arm_done = sum(arm[s][1] for s in sample)
        ref_cost = sum(anchor[s][0] for s in sample)
        ref_done = sum(anchor[s][1] for s in sample)
        if not arm_done or not ref_done or ref_cost == 0.0:
            return None
        return ((arm_cost / arm_done) - (ref_cost / ref_done)) / (ref_cost / ref_done) * 100.0

    point = ratio(ids)
    if point is None:
        return None
    rng = Random(seed)
    draws: list[float] = []
    undefined = 0
    for _ in range(resamples):
        value = ratio(rng.choices(ids, k=len(ids)))
        if value is None:
            undefined += 1
        else:
            draws.append(value)
    draws.sort()
    lo = draws[int(0.025 * len(draws))]
    hi = draws[min(len(draws) - 1, int(0.975 * len(draws)))]
    return CostDeltaCI(
        delta_percent=point,
        ci_low=lo,
        ci_high=hi,
        n_scenarios=len(ids),
        resamples=resamples,
        undefined_resamples=undefined,
    )


class FigureSpec(BaseModel):
    """One figure a lens wants: a registered kind plus its parameters."""

    model_config = ConfigDict(frozen=True)

    kind: str
    filename: str
    params: JsonObject = Field(default_factory=dict)


class LensSpec(BaseModel):
    """A corner's declarative rendering request: which figures, into which subdirectory.

    A lens may point at a different arm-matrix root (`dataset_root`, grid dir layout:
    `<root>/<arm>/matrix.json`) with its own provenance labels; the DEFAULT is the live tau
    grid cohort, so existing lenses are unchanged. This is the tb2-cost sibling's seam
    (DECISIONS 2026-07-28 lens-needs entry): same computation, second dataset, corpora
    never blended because every record carries its dataset label.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    corner_dir: str  # subdirectory of corners/ that owns the output
    figures: tuple[FigureSpec, ...]
    dataset_root: str | None = None  # default: the live tau grid cohort (data.grid_dir)
    dataset_label: str = "tau-bench"
    judge_label: str | None = None  # default: GRID_JUDGE (the tau cohort's pin)


class ConfigRecord(BaseModel):
    """One measured configuration (arm x model) on all three objectives, both anchors."""

    key: str  # "<arm>/<model>"
    arm: str
    model: str
    vs_anchor: Scorecard
    vs_anchor_evidence: PairedDelta
    vs_anchor_cost_ci: CostDeltaCI | None = None
    vs_best: Scorecard | None = None
    vs_best_evidence: PairedDelta | None = None
    vs_best_cost_ci: CostDeltaCI | None = None


class RoutedRecord(BaseModel):
    """One routed policy replay at one dial position, on the fit's held-out eval band only.

    The replay never touches `fit_scenario_ids` (routing on the band the policy was fitted
    on would grade the fit on its own training data). Router embedding cost is attached as
    RowOverhead per episode, priced as an ESTIMATE from query length at the 3-large list
    rate (the replay's own embedding calls are analysis spend, logged separately; a served
    endpoint pays the same shape per query).
    """

    arm: str
    dial: float
    vs_anchor: Scorecard
    vs_anchor_evidence: PairedDelta
    vs_anchor_cost_ci: CostDeltaCI | None = None
    vs_best: Scorecard | None = None
    vs_best_evidence: PairedDelta | None = None
    vs_best_cost_ci: CostDeltaCI | None = None
    n_eval_scenarios: int
    routed_mix: dict[str, int]  # model -> scenarios routed to it


class CornersDataset(BaseModel):
    """Everything phase-1 figures draw from, computed once per invocation.

    `records` cost/latency numbers come from `wmo.optimize.scorecard` (the D-COMPRESS
    accounting rule) and nowhere else; `teacher` comes from
    `wmo.optimize.teacher.select_teacher` (the repo's distillation decision) and is cited,
    never recomputed by hand. `status` is the completeness line every figure footnote
    carries while the cohort is still landing.
    """

    anchor_model: str
    dataset_label: str = "tau-bench"
    judge_label: str = GRID_JUDGE
    best_single: str | None = None
    records: list[ConfigRecord] = []
    routed: list[RoutedRecord] = []
    embedding_replay_calls: int = 0
    embedding_replay_est_usd: float = 0.0
    teacher: TeacherSearchVerdict | None = None
    teacher_unavailable_reason: str | None = None
    status: str = "no grid data on disk yet"
    pending: list[str] = []
    notes: list[str] = []


def _condition(
    model: str, optimizer: str, *, dataset: str = "tau-bench", judge: str = GRID_JUDGE
) -> ConditionLabel:
    return ConditionLabel(
        base_model=model,
        optimizer=optimizer,
        dataset=dataset,
        split="wm-test-band",
        judge=judge,
        provenance="wm_simulated",
    )


def _compressor_overheads(rows: list) -> list[RowOverhead]:
    """The compressor's own bill, folded in as the D-COMPRESS accounting rule requires."""
    return [
        RowOverhead(
            scenario_id=row.scenario_id,
            model=row.model,
            episode=row.episode,
            component="compressor",
            cost_usd=row.compressor_cost_usd,
            latency_s=row.compressor_latency_s,
        )
        for row in rows
        if row.compressor_cost_usd > 0.0 or row.compressor_latency_s > 0.0
    ]


def _config_arm(
    snapshot: ArmSnapshot, model: str, *, dataset: str = "tau-bench", judge: str = GRID_JUDGE
) -> Arm:
    rows = rows_for_model(snapshot.matrix, model)
    optimizer = "none" if snapshot.name == IDENTITY_ARM else f"compaction({snapshot.name})"
    return Arm(
        name=f"{model} [{snapshot.name}]",
        condition=_condition(model, optimizer, dataset=dataset, judge=judge),
        rows=rows,
        overheads=_compressor_overheads(rows),
    )


def _best_single(snapshot: ArmSnapshot) -> str | None:
    """The strongest pool model by mean reward over its scored identity rows."""
    means: dict[str, float] = {}
    for model in snapshot.matrix.model_names():
        rewards = [
            o.reward for o in rows_for_model(snapshot.matrix, model) if o.reward is not None
        ]
        if rewards:
            means[model] = sum(rewards) / len(rewards)
    return max(means, key=lambda m: means[m]) if means else None


def build_dataset(
    anchor_model: str = DEFAULT_ANCHOR,
    *,
    root: Path | None = None,
    dataset_label: str = "tau-bench",
    judge: str = GRID_JUDGE,
    loo: bool = False,
) -> CornersDataset:
    """Load once, aggregate once. Every figure below renders from this object.

    `root`/`dataset_label`/`judge` come from the lens spec (default: the live tau cohort),
    so a second corpus renders through the SAME computation with its own provenance labels.
    """
    snapshots = all_arm_snapshots(root=root)
    present = {s.name for s in snapshots}
    from data import GRID_ARMS  # local: keeps the module-level import list honest

    dataset = CornersDataset(
        anchor_model=anchor_model,
        dataset_label=dataset_label,
        judge_label=judge,
        pending=[arm for arm in GRID_ARMS if arm not in present],
        status="; ".join(f"{s.name}: {s.status}" for s in snapshots)
        or "no grid data on disk yet",
    )
    identity = next((s for s in snapshots if s.name == IDENTITY_ARM), None)
    if identity is None or anchor_model not in identity.matrix.model_names():
        dataset.notes.append("identity arm (or its anchor rows) not landed: records empty")
        return dataset

    anchor_rows = rows_for_model(identity.matrix, anchor_model)
    if not any(row.reward is not None for row in anchor_rows):
        dataset.notes.append(f"{anchor_model} has no scored identity rows yet: records empty")
        return dataset
    anchor = Arm(
        name=f"{anchor_model} [anchor]",
        condition=_condition(anchor_model, "none", dataset=dataset_label, judge=judge),
        rows=anchor_rows,
    )
    dataset.best_single = _best_single(identity)
    best_anchor = None
    if dataset.best_single is not None:
        best_anchor = Arm(
            name=f"{dataset.best_single} [best-single anchor]",
            condition=_condition(
                dataset.best_single, "none", dataset=dataset_label, judge=judge
            ).replace(notes="best-single anchor"),
            rows=rows_for_model(identity.matrix, dataset.best_single),
        )

    for snapshot in snapshots:
        for model in snapshot.matrix.model_names():
            key = f"{snapshot.name}/{model}"
            if not any(
                row.reward is not None for row in rows_for_model(snapshot.matrix, model)
            ):
                dataset.notes.append(f"{key}: no scored rows yet, skipped")
                continue
            try:
                config = _config_arm(snapshot, model, dataset=dataset_label, judge=judge)
            except ValueError as exc:
                # A mid-repair snapshot can hold a cell twice (original + retried row).
                dataset.notes.append(f"{key}: rows not usable yet ({exc})")
                continue
            config_rewards = rewards_by_scenario(config.rows, model=model)
            if config.condition.key() == anchor.condition.key():
                continue
            try:
                card = build_scorecard(arm=config, anchor=anchor)
            except ValueError as exc:
                dataset.notes.append(f"{key}: not comparable yet ({exc})")
                continue
            config_cost = _scenario_cost(config.rows, config.overheads)
            record = ConfigRecord(
                key=key,
                arm=snapshot.name,
                model=model,
                vs_anchor=card,
                vs_anchor_evidence=paired_delta(
                    config_rewards, rewards_by_scenario(anchor.rows, model=anchor_model)
                ),
                vs_anchor_cost_ci=cost_delta_ci(
                    config_cost, _scenario_cost(anchor.rows, [])
                ),
            )
            if best_anchor is not None and config.condition.key() != best_anchor.condition.key():
                try:
                    record.vs_best = build_scorecard(arm=config, anchor=best_anchor)
                    record.vs_best_evidence = paired_delta(
                        config_rewards,
                        rewards_by_scenario(
                            best_anchor.rows, model=best_anchor.condition.base_model
                        ),
                    )
                    record.vs_best_cost_ci = cost_delta_ci(
                        config_cost, _scenario_cost(best_anchor.rows, [])
                    )
                except ValueError as exc:
                    dataset.notes.append(f"{key} vs best-single: not comparable yet ({exc})")
            dataset.records.append(record)

    for snapshot in snapshots:
        _routed_records(
            snapshot, anchor, best_anchor, dataset, root=root, dataset_label=dataset_label,
            judge=judge,
        )

    if loo:
        from data import grid_dir

        reference_path = (root or grid_dir()) / IDENTITY_ARM / "policy.json"
        if reference_path.exists():
            loo_routed_records(
                dataset, anchor, best_anchor, identity, RoutingPolicy.load(reference_path)
            )
        else:
            dataset.notes.append("loo: no identity policy.json to take the recipe from")

    try:
        dataset.teacher = select_teacher(identity.matrix)
    except ValueError as exc:
        dataset.teacher_unavailable_reason = str(exc)
    return dataset


def _by_scenario_rewards(rows: list) -> dict[str, list[float]]:
    """Scored rewards per scenario across ALL models (a routed arm spans the pool)."""
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row.reward is not None:
            grouped.setdefault(row.scenario_id, []).append(row.reward)
    return grouped


def _routed_records(
    snapshot: ArmSnapshot,
    anchor: Arm,
    best_anchor: Arm | None,
    dataset: CornersDataset,
    *,
    root: Path | None = None,
    dataset_label: str = "tau-bench",
    judge: str = GRID_JUDGE,
) -> None:
    """Replay this arm's fitted policy (when one exists) at the five dial detents.

    Eval band only: `fit_scenario_ids` are excluded, so the rung is scored on scenarios the
    fit never saw. Embedding calls for the replay are real spend (allowed by Silen's
    2026-07-28 ruling, ~cents); their count and list-price estimate are recorded on the
    dataset. Any failure (missing env key, mid-repair matrix) becomes a named note, never a
    silent absence.
    """
    from data import grid_dir  # matches all_arm_snapshots' default root

    policy_path = (root or grid_dir()) / snapshot.name / "policy.json"
    if not policy_path.exists():
        dataset.pending.append(f"routed rung [{snapshot.name}]: no policy.json yet")
        return
    try:
        policy = RoutingPolicy.load(policy_path)
        embedder = policy.embedder.build()
        if policy.fit_compression is not None:
            # Compressed-arm banks are fitted through CompressingEmbedder (queries embedded
            # in the arm's SERVED geometry, C2's representation-consistency rule). Replaying
            # with the raw embedder would query a compressed-geometry bank with raw vectors,
            # which is the measured floor-tripping failure, so the replay compresses too.
            embedder = CompressingEmbedder(embedder, policy.fit_compression)
        eval_ids = [
            sid
            for sid in snapshot.matrix.scenario_ids()
            if sid not in set(policy.fit_scenario_ids)
        ]
        if not eval_ids:
            dataset.notes.append(f"routed rung [{snapshot.name}]: no held-out scenarios")
            return
        for dial in ROUTED_DIALS:
            dialed = apply_cost_quality(policy, dial)
            rows = rows_for_policy(
                snapshot.matrix, dialed, ids=eval_ids, embedder=embedder
            )
            dataset.embedding_replay_calls += len(eval_ids)
            dataset.embedding_replay_est_usd += sum(
                len(snapshot.matrix.for_scenario(sid)[0].task) // 4 for sid in eval_ids
            ) / 1e6 * EMBED_LIST_USD_PER_MTOK
            overheads = [
                RowOverhead(
                    scenario_id=row.scenario_id,
                    model=row.model,
                    episode=row.episode,
                    component="router-embedding(list-price estimate)",
                    cost_usd=len(row.task) // 4 / 1e6 * EMBED_LIST_USD_PER_MTOK,
                )
                for row in rows
            ] + _compressor_overheads(rows)
            arm = Arm(
                name=f"routed@{dial:g} [{snapshot.name}]",
                condition=_condition(
                    "pool(routed)",
                    f"routing(knn dial={dial:g})"
                    + ("" if snapshot.name == IDENTITY_ARM else f"+compaction({snapshot.name})"),
                    dataset=dataset_label,
                    judge=judge,
                ),
                rows=rows,
                overheads=overheads,
            )
            routed_rewards = _by_scenario_rewards(rows)
            mix: dict[str, int] = {}
            for sid in eval_ids:
                chosen = {row.model for row in rows if row.scenario_id == sid}
                for model in chosen:
                    mix[model] = mix.get(model, 0) + 1
            routed_cost = _scenario_cost(rows, overheads)
            record = RoutedRecord(
                arm=snapshot.name,
                dial=dial,
                vs_anchor=build_scorecard(arm=arm, anchor=anchor),
                vs_anchor_evidence=paired_delta(
                    routed_rewards,
                    rewards_by_scenario(anchor.rows, model=anchor.condition.base_model),
                ),
                vs_anchor_cost_ci=cost_delta_ci(routed_cost, _scenario_cost(anchor.rows, [])),
                n_eval_scenarios=len(eval_ids),
                routed_mix=mix,
            )
            if best_anchor is not None:
                record.vs_best = build_scorecard(arm=arm, anchor=best_anchor)
                record.vs_best_evidence = paired_delta(
                    routed_rewards,
                    rewards_by_scenario(
                        best_anchor.rows, model=best_anchor.condition.base_model
                    ),
                )
                record.vs_best_cost_ci = cost_delta_ci(
                    routed_cost, _scenario_cost(best_anchor.rows, [])
                )
            dataset.routed.append(record)
    except Exception as exc:  # noqa: BLE001 - the runner renders what it can and names the rest
        dataset.notes.append(f"routed rung [{snapshot.name}]: failed ({exc})")


def loo_routed_records(
    dataset: CornersDataset,
    anchor: Arm,
    best_anchor: Arm | None,
    snapshot: ArmSnapshot,
    reference_policy: RoutingPolicy,
) -> None:
    """Leave-one-out routed rungs: every scenario measured held-out (the power fix).

    For each of the matrix's scenarios, fit the SAME recipe as the reference policy on the
    other n-1 (fit_knn_policy's own small-bank scaling reproduces the recipe at any bank
    size; the guard/fallback is re-discovered per fold exactly as the real fit does), dial
    it, and route only the left-out scenario. Pooled across folds this yields a routed
    measurement on EVERY scenario, none of it seen by the fit that routed it. Banks go to a
    temp dir, never the cohort's. Appended to `dataset.routed` as arm
    "<arm>+loo" so downstream summaries and the n=6 single-split records stay distinct.
    """
    import tempfile

    embedder = reference_policy.embedder.build()
    if reference_policy.fit_compression is not None:
        embedder = CompressingEmbedder(embedder, reference_policy.fit_compression)
    all_ids = snapshot.matrix.scenario_ids()
    per_dial_rows: dict[float, list] = {d: [] for d in ROUTED_DIALS}
    with tempfile.TemporaryDirectory() as tmp:
        for index, held_out in enumerate(all_ids):
            fold_policy = fit_knn_policy(
                snapshot.matrix,
                bank_path=Path(tmp) / f"fold-{index}.bank.npz",
                fit_ids=[sid for sid in all_ids if sid != held_out],
                embedder=reference_policy.embedder,
                embed_with=embedder,
                floor_q=reference_policy.floor_q,
            )
            for dial in ROUTED_DIALS:
                rows = rows_for_policy(
                    snapshot.matrix,
                    apply_cost_quality(fold_policy, dial),
                    ids=[held_out],
                    embedder=embedder,
                )
                per_dial_rows[dial].extend(rows)
            dataset.embedding_replay_calls += len(all_ids) + len(ROUTED_DIALS)

    for dial, rows in per_dial_rows.items():
        overheads = [
            RowOverhead(
                scenario_id=row.scenario_id,
                model=row.model,
                episode=row.episode,
                component="router-embedding(list-price estimate)",
                cost_usd=len(row.task) // 4 / 1e6 * EMBED_LIST_USD_PER_MTOK,
            )
            for row in rows
        ] + _compressor_overheads(rows)
        arm = Arm(
            name=f"routed-loo@{dial:g} [{snapshot.name}]",
            condition=_condition(
                "pool(routed)",
                f"routing(knn loo dial={dial:g})",
                dataset=dataset.dataset_label,
                judge=dataset.judge_label,
            ),
            rows=rows,
            overheads=overheads,
        )
        routed_rewards = _by_scenario_rewards(rows)
        routed_cost = _scenario_cost(rows, overheads)
        mix: dict[str, int] = {}
        for sid in all_ids:
            for model in {row.model for row in rows if row.scenario_id == sid}:
                mix[model] = mix.get(model, 0) + 1
        record = RoutedRecord(
            arm=f"{snapshot.name}+loo",
            dial=dial,
            vs_anchor=build_scorecard(arm=arm, anchor=anchor),
            vs_anchor_evidence=paired_delta(
                routed_rewards,
                rewards_by_scenario(anchor.rows, model=anchor.condition.base_model),
            ),
            vs_anchor_cost_ci=cost_delta_ci(routed_cost, _scenario_cost(anchor.rows, [])),
            n_eval_scenarios=len(all_ids),
            routed_mix=mix,
        )
        if best_anchor is not None:
            record.vs_best = build_scorecard(arm=arm, anchor=best_anchor)
            record.vs_best_evidence = paired_delta(
                routed_rewards,
                rewards_by_scenario(best_anchor.rows, model=best_anchor.condition.base_model),
            )
            record.vs_best_cost_ci = cost_delta_ci(
                routed_cost, _scenario_cost(best_anchor.rows, [])
            )
        dataset.routed.append(record)


REAL_PROBE_DIR = Path.home() / "Desktop/Projects/world-model-harness/.wmo/jt/real-probe"
REAL_JUDGE = "tau2 reward (7/20 pinned tasks include tau2's NL-assertion judge)"


def _real_rows(model: str) -> list:
    """The probe's real-episode rows for one arm, as ScenarioOutcome (scorecard input).

    tau2's reward is binary-ish in [0,1]; success = full reward. Cost is candidate-side
    at OUR pool prices (cost_usd_pool); the user simulator is environment, identical for
    both arms, and excluded from the candidate comparison.
    """
    from wmo.optimize.outcomes import ScenarioOutcome as Row

    path = REAL_PROBE_DIR / "rows.jsonl"
    if not path.exists():
        return []
    out: list[Row] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if raw.get("model") != model:
            continue
        out.append(
            Row(
                scenario_id=str(raw["scenario_id"]),
                task=str(raw.get("scenario_id")),
                model=model,
                episode=int(raw.get("episode", 0)),
                reward=raw.get("reward"),
                success=raw.get("reward") == 1.0,
                cost_usd=float(raw.get("cost_usd_pool") or 0.0),
                call_seconds=[float(raw.get("duration_s") or 0.0)],
                error=str(raw.get("error")) if raw.get("error") else None,
            )
        )
    return out


def real_probe_section(dataset: CornersDataset) -> dict[str, object] | None:
    """The WM-vs-real probe verdict (pre-registered in the real-leg prompt).

    Real side: routed-decision-direct (the balanced policy's decision executed direct,
    endpoint tool-call gap filed) vs fable-5, real tau2 episodes, tau2's own reward.
    Sim side: the identity matrix's opus-5 vs fable-5 rows. The verdict is per-scenario
    PAIRED SIGN AGREEMENT (the binding sim-to-real convention) plus the two headline
    checks: routed cost saving vs fable-5, and quality at-least-parity. Never pools the
    provenances; the comparison is deltas-vs-deltas.
    """
    routed_rows = _real_rows("routed-decision-direct")
    anchor_rows = _real_rows("fable-5")
    if not routed_rows or not anchor_rows:
        return None
    routed = Arm(
        name="routed-decision-direct [real]",
        condition=ConditionLabel(
            base_model="opus-5(routed decision)", optimizer="routing(balanced, direct-exec)",
            dataset="tau2-real", split="pinned-eval-20", judge=REAL_JUDGE,
            provenance="real_episode",
        ),
        rows=routed_rows,
    )
    anchor = Arm(
        name="fable-5 [real]",
        condition=ConditionLabel(
            base_model="fable-5", optimizer="none", dataset="tau2-real",
            split="pinned-eval-20", judge=REAL_JUDGE, provenance="real_episode",
        ),
        rows=anchor_rows,
    )
    card = build_scorecard(arm=routed, anchor=anchor)
    real_delta = paired_delta(
        _by_scenario_rewards(routed_rows), _by_scenario_rewards(anchor_rows)
    )
    cost_ci = cost_delta_ci(_scenario_cost(routed_rows, []), _scenario_cost(anchor_rows, []))

    # Sign agreement: real per-scenario deltas vs the WM's (opus-5 minus fable-5), matched
    # by position in the pinned split (the WM's scenario ids are trace hashes, the real
    # ids are domain:task ids; both derive from the same pinned 20 in the same order, but
    # a positional map would be fragile, so agreement is computed on the DELTA VECTORS'
    # signs per shared count with the mapping stated).
    wm_identity = next((s for s in all_arm_snapshots() if s.name == IDENTITY_ARM), None)
    agreement = None
    if wm_identity is not None and dataset.anchor_model in wm_identity.matrix.model_names():
        wm_delta = paired_delta(
            rewards_by_scenario(wm_identity.matrix.outcomes, model="opus-5"),
            rewards_by_scenario(wm_identity.matrix.outcomes, model=dataset.anchor_model),
        )
        agreement = {
            "note": (
                "sign agreement requires a scenario-id bridge (wm ids are trace hashes, "
                "real ids are domain:task); computed by the probe script via the pinned "
                "split order and reported in findings"
            ),
            "wm_mean_delta_pt": round(wm_delta.mean_delta * 100, 2),
            "real_mean_delta_pt": round(real_delta.mean_delta * 100, 2),
            "wm_n_up_down_tied": [wm_delta.n_up, wm_delta.n_down, wm_delta.n_tied],
            "real_n_up_down_tied": [real_delta.n_up, real_delta.n_down, real_delta.n_tied],
        }
    return {
        "label": "measured",
        "provenance": "real_episode",
        "judge": REAL_JUDGE,
        "deviation": (
            "routed arm executed direct (served endpoint 501s tool calls on "
            "anthropic-kind entries; finding filed); balanced policy routes 100% opus-5, "
            "verified live + 20/20 LOO folds"
        ),
        "vs_anchor": _summary(card, real_delta, cost_ci),
        "sim_vs_real": agreement,
    }


def fig_dial_curve(dataset: CornersDataset, spec: FigureSpec, out: Path) -> None:
    """The dial's measured cost curve: ours9 anchors AS MEASURED; tau panel pending fits.

    The corpora are never blended (D-DIAL v2 re-anchors jointly later).
    """
    # Separate y axes on purpose: the panels state deltas against DIFFERENT anchors
    # (ours9 vs its best single; tau vs fable-5 per the anchor ruling), so a shared scale
    # would visually blend two quantities the corpora rule keeps apart (and it clipped the
    # tau points clean off the canvas on the first render).
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.5))
    dials = [a.cost_quality for a in COST_QUALITY_ANCHORS]
    costs = [a.cost_delta_percent for a in COST_QUALITY_ANCHORS]
    left.plot(dials, costs, color=BLUE, marker="o")
    for a in COST_QUALITY_ANCHORS:
        left.annotate(
            f"{a.named_point}\n{a.quality_delta_points:+.2f} pt",
            xy=(a.cost_quality, a.cost_delta_percent),
            xytext=(0, -22),
            textcoords="offset points",
            fontsize=7.5,
            ha="center",
        )
    left.set_title("Dial cost curve: routerbench-ours9 (measured)")
    left.set_xlabel("cost_quality dial position")
    left.set_ylabel("cost % vs best single pool model")
    left.set_ylim(min(costs) - 12, 6)
    right.set_xlabel("cost_quality dial position")
    tau_points = [
        r
        for r in dataset.routed
        if r.arm == IDENTITY_ARM and r.vs_anchor.cost_delta_percent is not None
    ]
    if tau_points:
        dials = [r.dial for r in tau_points]
        tau_costs = [r.vs_anchor.cost_delta_percent for r in tau_points]
        right.plot(dials, tau_costs, color=PURPLE, marker="o")
        right.set_ylabel(f"cost % vs {dataset.anchor_model}")
        right.set_ylim(min(tau_costs) - 12, max(0.0, max(tau_costs)) + 6)
        seen: set[tuple[float, float]] = set()
        for r in tau_points:
            point = (r.vs_anchor.cost_delta_percent, r.vs_anchor.quality_delta_points)
            if point in seen:  # consecutive dials that map to the same routed mix
                continue
            seen.add(point)
            right.annotate(
                f"{r.vs_anchor.quality_delta_points:+.1f} pt · "
                f"p50 {r.vs_anchor.latency.p50_model_s:.0f}s",
                xy=(r.dial, r.vs_anchor.cost_delta_percent),
                xytext=(0, -22),
                textcoords="offset points",
                fontsize=7.5,
                ha="center",
            )
        n = tau_points[0].n_eval_scenarios
        right.set_title(
            f"Dial cost curve: tau grid, routed replay vs {dataset.anchor_model} "
            f"(n={n} held-out)"
        )
    else:
        right.set_title("Dial cost curve: tau grid (pending per-arm fits)")
        right.text(
            0.5,
            0.5,
            "awaiting the master's bank-refit-per-arm fits",
            transform=right.transAxes,
            ha="center",
            color=MUTED,
            fontsize=9,
        )
    footnote(
        fig,
        "left: the five COST_QUALITY_ANCHORS (wmo/optimize/knn.py), measured on "
        "routerbench-ours9 (1199 scenarios, 9 models, 5 seeds) vs ITS best single model · "
        f"right: identity-arm policy replayed at the same detents on ITS held-out eval band, "
        f"deltas vs {dataset.anchor_model} per the anchor ruling, quality + p50 annotated · "
        "corpora and anchors NEVER blended across panels (D-DIAL v2 re-anchors jointly) · "
        "ours9 anchors carry no latency (pre-v2 measurement, unknown renders nothing)",
    )
    save_fig(fig, out / spec.filename)


def fig_savings_frontier(dataset: CornersDataset, spec: FigureSpec, out: Path) -> None:
    """Savings vs the anchor per config and rung: cost x, quality y (noise band), latency
    labeled per point."""
    if not dataset.records:
        return
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.axhspan(
        -NOISE_FLOOR_POINTS,
        NOISE_FLOOR_POINTS,
        color=NOISE_BAND_COLOR,
        alpha=NOISE_BAND_ALPHA,
        zorder=0,
    )
    ax.text(
        0.99,
        NOISE_FLOOR_POINTS,
        "noise floor ",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=MUTED,
    )
    plotted = [r for r in dataset.records if r.vs_anchor.cost_delta_percent is not None]
    # Selective direct labels (never a label on every point): the cost/quality Pareto
    # frontier, the cost-inversion outliers (compression made it dearer than the anchor),
    # and the within-noise-floor triangles. Everything else is in numbers.json.
    def dominated(r: ConfigRecord) -> bool:
        c, q = r.vs_anchor.cost_delta_percent, r.vs_anchor.quality_delta_points
        return any(
            o.vs_anchor.cost_delta_percent <= c
            and o.vs_anchor.quality_delta_points >= q
            and (
                o.vs_anchor.cost_delta_percent < c or o.vs_anchor.quality_delta_points > q
            )
            for o in plotted
            if o is not r and o.vs_anchor.cost_delta_percent is not None
        )

    labeled = {
        r.key
        for r in plotted
        if not dominated(r)
        or r.vs_anchor.cost_delta_percent > 0.0
        or r.vs_anchor_evidence.within_noise_floor
    }
    for index, record in enumerate(plotted):
        card = record.vs_anchor
        marker = "^" if record.vs_anchor_evidence.within_noise_floor else "o"
        ax.scatter(
            card.cost_delta_percent,
            card.quality_delta_points,
            color=ARM_COLORS[record.arm],
            s=42,
            marker=marker,
            zorder=3,
        )
        if record.key in labeled:
            ax.annotate(
                f"{record.model} · p50 {card.latency.p50_model_s:.0f}s"
                f" · n{card.scenarios_compared}",
                xy=(card.cost_delta_percent, card.quality_delta_points),
                xytext=(6, 4 if index % 2 == 0 else -12),
                textcoords="offset points",
                fontsize=8,
            )
    for routed in dataset.routed:
        card = routed.vs_anchor
        if card.cost_delta_percent is None:
            continue
        is_loo = routed.arm.endswith("+loo")
        ax.scatter(
            card.cost_delta_percent,
            card.quality_delta_points,
            color=ARM_COLORS[routed.arm.removesuffix("+loo")],
            s=110,
            marker="*",
            zorder=4,
        )
        if routed.dial in (0.25, 1.0):
            ax.annotate(
                f"routed{'-loo' if is_loo else ''}@{routed.dial:g} · n{card.scenarios_compared}",
                xy=(card.cost_delta_percent, card.quality_delta_points),
                xytext=(6, -12 if not is_loo else 6),
                textcoords="offset points",
                fontsize=8,
            )
    ax.axhline(0.0, color=MUTED, linewidth=0.8, linestyle=":")
    ax.axvline(0.0, color=MUTED, linewidth=0.8, linestyle=":")
    handles = [
        Line2D([], [], marker="o", linestyle="", color=color, label=arm)
        for arm, color in ARM_COLORS.items()
    ]
    handles.append(
        Line2D([], [], marker="^", linestyle="", color=MUTED, label="Δ within noise floor")
    )
    if dataset.routed:
        handles.append(
            Line2D([], [], marker="*", linestyle="", color=MUTED, label="routed policy (dial)")
        )
    ax.legend(handles=handles, loc="lower left")
    ax.set_title(f"Savings vs {dataset.anchor_model}, per config and per compression rung")
    ax.set_xlabel(
        f"effective cost per completed task, % vs {dataset.anchor_model} (negative = cheaper)"
    )
    ax.set_ylabel(f"quality, points vs {dataset.anchor_model} (mean reward x 100)")
    footnote(
        fig,
        f"measured · wm_simulated · {dataset.dataset_label} · judge {dataset.judge_label} "
        f"· anchor {dataset.anchor_model} "
        f"(identity arm) · effective cost = cache-adjusted provider spend + compressor "
        f"inference, per COMPLETED task, unscored spend excluded and reported "
        f"(wmo.optimize.scorecard) · compaction rungs are a measured tradeoff, not a "
        f"recommendation (accuracy verdict pending) · snapshot: {dataset.status}"
        + (f" · PENDING arms: {', '.join(dataset.pending)}" if dataset.pending else ""),
    )
    save_fig(fig, out / spec.filename)


def fig_cost_per_task(dataset: CornersDataset, spec: FigureSpec, out: Path) -> None:
    """Absolute effective cost per completed task, dot plot per model per arm."""
    by_model: dict[str, dict[str, float]] = {}
    anchor_cost: float | None = None
    for record in dataset.records:
        cost = record.vs_anchor.cost.cost_per_completed_task_usd
        if cost is not None:
            by_model.setdefault(record.model, {})[record.arm] = cost
        anchor_cost = record.vs_anchor.anchor_cost.cost_per_completed_task_usd or anchor_cost
    if not by_model:
        return
    order = sorted(by_model, key=lambda m: min(by_model[m].values()))
    fig, ax = plt.subplots(figsize=(9.0, 0.45 * len(order) + 1.8))
    for y, model in enumerate(order):
        for arm_name, cost in by_model[model].items():
            ax.scatter(cost, y, color=ARM_COLORS[arm_name], s=40, zorder=3)
    if anchor_cost is not None:
        ax.axvline(anchor_cost, color=MUTED, linewidth=1.2, linestyle="--")
        ax.text(
            anchor_cost,
            len(order) - 0.3,
            f"  {dataset.anchor_model} anchor",
            color=MUTED,
            fontsize=9,
        )
    ax.set_yticks(range(len(order)), order)
    ax.set_xscale("log")
    ax.set_title("Effective cost per completed task (log $), by model and compression rung")
    ax.set_xlabel("cache-adjusted effective $ per completed task")
    footnote(
        fig,
        f"measured · wm_simulated · {dataset.dataset_label} · judge {dataset.judge_label} "
        f"· dots colored by arm: identity "
        f"blue, truncate purple, llmlingua2-endpoint red · scorecard accounting (unscored "
        f"spend excluded, reported in numbers.json) · quality and latency per config are in "
        f"numbers.json and the frontier chart · snapshot: {dataset.status}",
    )
    save_fig(fig, out / spec.filename)


def fig_training_stage(dataset: CornersDataset, spec: FigureSpec, out: Path) -> None:
    """The shared training-stage chart via the canonical ablation_chart (one implementation,
    charter deliverable 1), through this lens."""
    lens = str(spec.params.get("lens", "cost"))
    chart = ablation_chart.build_shared_chart_data()
    ablation_chart.render_training_stage_chart(chart, out / spec.filename, lens=lens)


def _bar(
    ax: plt.Axes,
    x: int,
    cost: float | None,
    color: str,
    label: str,
    detail: str,
    dy: int = 4,
) -> None:
    """One three-stage bar: effective $/completed task, detail text above, name below.
    `dy` staggers the annotation when neighboring bars are near-equal height."""
    if cost is None:
        ax.text(x, 0.02, "undefined\n(0 completed)", ha="center", fontsize=7.5, color=MUTED)
        return
    ax.bar(x, cost, width=0.62, color=color, zorder=3)
    ax.annotate(
        detail, xy=(x, cost), xytext=(0, dy), textcoords="offset points",
        ha="center", fontsize=7.2,
    )


def fig_three_stage(dataset: CornersDataset, spec: FigureSpec, out: Path) -> None:
    """The per-benchmark three-stage chart (Silen directive, DECISIONS 2026-07-28).

    Stage 1 routing: anchor vs best-single vs routed, model mix annotated on routed bars,
    both measured dials. Stage 2 +compaction: the compression arms added, negatives kept
    and labeled. Stage 3 distillation: the gate verdicts rendered as the mechanism working
    (text panel; there is no promoted student anywhere by decision). Every bar states all
    three objectives (cost = bar height, quality delta + p50 in the detail text); held-out
    basis for routed comparisons; provenance in the footnote.

    Params: `constant_routed` annotates a fit that routes 0% away when no routed replay
    exists (the terminal case); `distill_note` is the benchmark's stage-3 story.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.0, 5.2), width_ratios=(3, 3, 2))
    anchor = dataset.anchor_model
    best = dataset.best_single or "?"

    def detail(cost: float, quality: float, p50: float, n: int, mix: str = "") -> str:
        core = f"${cost:.2f}\n{quality:+.1f} pt · p50 {p50:.0f}s · n{n}"
        return f"{core}\n{mix}" if mix else core

    def _bar_ylim_headroom(ax: plt.Axes) -> None:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 1.22)  # room for the three-line detail annotations

    def mix_text(mix: dict[str, int]) -> str:
        total = sum(mix.values())
        ordered = sorted(mix.items(), key=lambda kv: -kv[1])
        return " / ".join(f"{100 * v // total}% {m}" for m, v in ordered)

    # Stage 1: routing. Common basis = the routed records' held-out band when replays
    # exist; otherwise the per-model records' common scenario set with the fit's constant
    # choice annotated.
    routed_by_dial = {r.dial: r for r in dataset.routed if r.arm == IDENTITY_ARM}
    x = 0
    if routed_by_dial:
        rec = routed_by_dial[0.25]
        _bar(ax1, 0, rec.vs_anchor.anchor_cost.cost_per_completed_task_usd, MUTED, anchor,
             detail(rec.vs_anchor.anchor_cost.cost_per_completed_task_usd or 0.0, 0.0,
                    rec.vs_anchor.anchor_latency.p50_model_s, rec.vs_anchor.scenarios_compared))
        best_cost = (
            None if rec.vs_best is None else rec.vs_best.anchor_cost.cost_per_completed_task_usd
        )
        if rec.vs_best is not None and best_cost is not None:
            _bar(ax1, 1, best_cost, BLUE, best,
                 detail(best_cost, -rec.vs_best.quality_delta_points,
                        rec.vs_best.anchor_latency.p50_model_s, rec.vs_best.scenarios_compared))
        for i, dial in enumerate((0.25, 1.0)):
            r = routed_by_dial.get(dial)
            if r is None or r.vs_anchor.cost.cost_per_completed_task_usd is None:
                continue
            _bar(ax1, 2 + i, r.vs_anchor.cost.cost_per_completed_task_usd, PURPLE,
                 f"routed@{dial:g}",
                 detail(r.vs_anchor.cost.cost_per_completed_task_usd,
                        r.vs_anchor.quality_delta_points, r.vs_anchor.latency.p50_model_s,
                        r.vs_anchor.scenarios_compared, mix_text(r.routed_mix)),
                 dy=4 if i == 0 else 44)
        labels = [anchor, best, "routed@0.25", "routed@1.0"]
        ax1.set_xticks(range(len(labels)), labels, fontsize=8)
        _bar_ylim_headroom(ax1)
    else:
        by_key = {r.key: r for r in dataset.records}
        best_rec = by_key.get(f"{IDENTITY_ARM}/{best}")
        if best_rec is not None:
            card = best_rec.vs_anchor
            _bar(ax1, 0, card.anchor_cost.cost_per_completed_task_usd, MUTED, anchor,
                 detail(card.anchor_cost.cost_per_completed_task_usd or 0.0, 0.0,
                        card.anchor_latency.p50_model_s, card.scenarios_compared))
            _bar(ax1, 1, card.cost.cost_per_completed_task_usd, BLUE, best,
                 detail(card.cost.cost_per_completed_task_usd or 0.0,
                        card.quality_delta_points, card.latency.p50_model_s,
                        card.scenarios_compared))
            constant = str(spec.params.get("constant_routed", ""))
            if constant and card.cost.cost_per_completed_task_usd is not None:
                _bar(ax1, 2, card.cost.cost_per_completed_task_usd, PURPLE, "routed",
                     detail(card.cost.cost_per_completed_task_usd,
                            card.quality_delta_points, card.latency.p50_model_s,
                            card.scenarios_compared, constant))
            ax1.set_xticks(range(3), [anchor, best, "routed"], fontsize=8)
    ax1.set_title("Stage 1: routing")
    ax1.set_ylabel("effective $ per completed task")

    # Stage 2: +compaction, negatives kept. Anchor + best-single across the three arms.
    arm_order = [IDENTITY_ARM, "truncate", "llmlingua2-endpoint"]
    arm_short = {IDENTITY_ARM: "raw", "truncate": "trunc", "llmlingua2-endpoint": "llml2"}
    positions: list[int] = []
    ticks: list[str] = []
    x = 0
    by_key = {r.key: r for r in dataset.records}
    for model in (anchor, best):
        for arm_name in arm_order:
            rec = by_key.get(f"{arm_name}/{model}")
            if rec is None:
                if model == anchor and arm_name == IDENTITY_ARM and routed_by_dial:
                    cost = routed_by_dial[0.25].vs_anchor.anchor_cost.cost_per_completed_task_usd
                    _bar(ax2, x, cost, ARM_COLORS[arm_name], "", f"${cost:.2f}\nanchor")
                    positions.append(x)
                    ticks.append(f"{model}\nraw")
                    x += 1
                continue
            card = rec.vs_anchor
            _bar(ax2, x, card.cost.cost_per_completed_task_usd, ARM_COLORS[arm_name], "",
                 detail(card.cost.cost_per_completed_task_usd or 0.0,
                        card.quality_delta_points, card.latency.p50_model_s,
                        card.scenarios_compared))
            positions.append(x)
            ticks.append(f"{model}\n{arm_short[arm_name]}")
            x += 1
        x += 1  # gap between model groups
    ax2.set_xticks(positions, ticks, fontsize=7)
    _bar_ylim_headroom(ax2)
    ax2.set_title("Stage 2: +compaction (taller = the finding)")

    # Stage 3: distillation, the verdicts as text.
    ax3.axis("off")
    ax3.set_title("Stage 3: distillation")
    verdict = (
        dataset.teacher.reason
        if dataset.teacher is not None
        else f"teacher gate: {dataset.teacher_unavailable_reason}"
    )
    note = str(spec.params.get("distill_note", ""))
    ax3.text(
        0.0, 0.95,
        (note + "\n\n" if note else "")
        + "Teacher gate (descriptive only, per the\nno-distillation ruling):\n"
        + "\n".join(textwrap.wrap(verdict, width=46)),
        va="top", fontsize=8,
    )

    footnote(
        fig,
        f"measured · wm_simulated · dataset {dataset.dataset_label} · judge "
        f"{dataset.judge_label} · anchor {anchor}, best-single {best} · effective cost per "
        f"COMPLETED task (wmo.optimize.scorecard; compressor + router overhead folded in) · "
        f"stage-1 routed bars on the fit's held-out band · compaction rungs are a measured "
        f"tradeoff, not a recommendation · CIs per record in numbers.json · {dataset.status}",
    )
    save_fig(fig, out / spec.filename)


def fig_three_stage_ours9(dataset: CornersDataset, spec: FigureSpec, out: Path) -> None:
    """The routing-only benchmark's stage chart: routerbench-ours9 from the D-DIAL anchors.

    ours9 has no compaction or distillation stages and no cost-per-task unit here (its
    anchors are deltas vs ITS best single model), so this renders stage 1 alone in delta
    space, with the frontier-pinned model mix the routing lane measured (DECISIONS
    2026-07-28 three-stage directive). Never blended with the tau/terminal charts.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    detents = {a.cost_quality: a for a in COST_QUALITY_ANCHORS}
    for i, dial in enumerate((0.0, 0.25, 0.5, 1.0)):
        a = detents[dial]
        ax.bar(i, a.cost_delta_percent, width=0.62, color=PURPLE, zorder=3)
        ax.annotate(
            f"{a.cost_delta_percent:+.1f}%\n{a.quality_delta_points:+.2f} pt",
            xy=(i, a.cost_delta_percent), xytext=(0, -16), textcoords="offset points",
            ha="center", fontsize=7.5,
        )
    ax.set_xticks(range(4), [detents[d].named_point for d in (0.0, 0.25, 0.5, 1.0)], fontsize=8)
    ax.axhline(0.0, color=MUTED, linewidth=0.8)
    ax.set_ylabel("cost % vs ours9's best single model")
    ax.set_title("Stage 1 (routing-only benchmark): routerbench-ours9 dial anchors")
    mix = str(spec.params.get("mix", ""))
    if mix:
        ax.text(0.02, 0.04, "\n".join(textwrap.wrap(f"frontier-pinned routed mix: {mix}", 60)),
                transform=ax.transAxes, fontsize=7.5, color=MUTED)
    footnote(
        fig,
        "measured on routerbench-ours9 (1199 scenarios, 9 models, 5 seeds) vs ITS best "
        "single pool model (wmo/optimize/knn.py COST_QUALITY_ANCHORS; mix per the routing "
        "lane, DECISIONS 2026-07-28) · routing-only benchmark: no compaction or "
        "distillation stage exists here · never blended with tau/terminal charts",
    )
    save_fig(fig, out / spec.filename)


FIGURE_KINDS = {
    "dial_curve": fig_dial_curve,
    "savings_frontier": fig_savings_frontier,
    "cost_per_task": fig_cost_per_task,
    "training_stage": fig_training_stage,
    "three_stage": fig_three_stage,
    "three_stage_ours9": fig_three_stage_ours9,
}

# The suspended corners' lenses, renderable via their FROZEN standalone scripts (Amendment
# 2: do not delete, do not rewrite). Delegation keeps them runnable through the one
# entrypoint; when an axis resumes, its chat writes a lens.py on dataset-native kinds and
# its entry here is deleted.
FROZEN_LENSES: dict[str, str] = {
    "quality": "quality/render_quality.py",
    "latency": "latency/render_latency.py",
}


def load_lens(name: str) -> LensSpec:
    """A corner's lens spec, from `corners/<name>/lens.py` exporting LENS."""
    lens_file = CORNERS / name / "lens.py"
    if not lens_file.exists():
        raise FileNotFoundError(
            f"no lens spec at {lens_file}; a corner declares its figures in lens.py "
            f"(frozen lenses: {sorted(FROZEN_LENSES)})"
        )
    namespace: dict[str, object] = {}
    exec(  # noqa: S102 - our own checked-in spec file, the declarative-config tradeoff
        compile(lens_file.read_text(encoding="utf-8"), str(lens_file), "exec"), namespace
    )
    lens = namespace.get("LENS")
    # Revalidated rather than isinstance-checked: when this file runs as __main__, the spec
    # file's `from build_corners import LensSpec` imports a SECOND copy of this module, and
    # the two class objects fail isinstance despite being the same model.
    if lens is None or not hasattr(lens, "model_dump"):
        raise TypeError(f"{lens_file} must export LENS: LensSpec (via common/build_corners)")
    return LensSpec.model_validate(lens.model_dump())


def render_lens(lens: LensSpec, dataset: CornersDataset, out_dir: Path | None = None) -> Path:
    """Render every figure a lens declares, from the one shared dataset."""
    out = out_dir or (CORNERS / lens.corner_dir / "figures")
    out.mkdir(parents=True, exist_ok=True)
    for spec in lens.figures:
        builder = FIGURE_KINDS.get(spec.kind)
        if builder is None:
            raise KeyError(
                f"lens '{lens.name}' asks for unknown figure kind '{spec.kind}'; "
                f"registered kinds: {sorted(FIGURE_KINDS)}. Extend build_corners.py, never "
                f"fork a parallel aggregation."
            )
        builder(dataset, spec, out)
    return out


def dump_numbers(dataset: CornersDataset, lens: LensSpec) -> Path:
    """Write the lens's numbers.json: the audit trail behind every figure."""
    out = CORNERS / lens.corner_dir / "numbers.json"
    payload = {
        "status": dataset.status,
        "dataset": dataset.dataset_label,
        "judge": dataset.judge_label,
        "pending_arms": dataset.pending,
        "anchor": dataset.anchor_model,
        "best_single": dataset.best_single,
        "teacher_verdict": None
        if dataset.teacher is None
        else json.loads(dataset.teacher.model_dump_json()),
        "teacher_unavailable_reason": dataset.teacher_unavailable_reason,
        "records": {
            r.key: {
                "vs_anchor": _summary(r.vs_anchor, r.vs_anchor_evidence, r.vs_anchor_cost_ci),
                "vs_best_single": None
                if r.vs_best is None or r.vs_best_evidence is None
                else _summary(r.vs_best, r.vs_best_evidence, r.vs_best_cost_ci),
            }
            for r in dataset.records
        },
        "routed": [
            {
                "arm": r.arm,
                "dial": r.dial,
                "n_eval_scenarios": r.n_eval_scenarios,
                "routed_mix": r.routed_mix,
                "vs_anchor": _summary(r.vs_anchor, r.vs_anchor_evidence, r.vs_anchor_cost_ci),
                "vs_best_single": None
                if r.vs_best is None or r.vs_best_evidence is None
                else _summary(r.vs_best, r.vs_best_evidence, r.vs_best_cost_ci),
            }
            for r in dataset.routed
        ],
        "embedding_replay": {
            "calls": dataset.embedding_replay_calls,
            "est_usd_at_3large_list": round(dataset.embedding_replay_est_usd, 6),
            "label": "estimate; analysis spend allowed by Silen 2026-07-28 ruling",
        },
        "real_probe": real_probe_section(dataset),
        "notes": dataset.notes,
    }
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return out


def _summary(
    card: Scorecard, evidence: PairedDelta, cost_ci: CostDeltaCI | None = None
) -> dict[str, object]:
    """The numbers a reader needs, plus what they must never be read without."""
    return {
        "label": "measured",
        "provenance": card.provenance,
        "judge": card.judge,
        "cost_delta_percent_ci95": None
        if cost_ci is None
        else {
            "low": round(cost_ci.ci_low, 1),
            "high": round(cost_ci.ci_high, 1),
            "n_scenarios": cost_ci.n_scenarios,
            "undefined_resamples": cost_ci.undefined_resamples,
        },
        "quality_delta_points": round(card.quality_delta_points, 2),
        "quality_delta_paired": {
            "mean": round(evidence.mean_delta, 4),
            "ci95": [round(evidence.ci_low, 4), round(evidence.ci_high, 4)],
            "sign_test_p": evidence.sign_test_p,
            "n_pairs": evidence.n_pairs,
            "within_noise_floor": evidence.within_noise_floor,
        },
        "cost_delta_percent": None
        if card.cost_delta_percent is None
        else round(card.cost_delta_percent, 1),
        "latency_p50_delta_percent": None
        if card.latency_p50_delta_percent is None
        else round(card.latency_p50_delta_percent, 1),
        "cost_per_completed_task_usd": card.cost.cost_per_completed_task_usd,
        "anchor_cost_per_completed_task_usd": card.anchor_cost.cost_per_completed_task_usd,
        "task_success_rate": round(card.quality.task_success_rate, 4),
        "mean_reward": round(card.quality.mean_reward, 4),
        "latency_p50_model_s": round(card.latency.p50_model_s, 2),
        "latency_p95_model_s": round(card.latency.p95_model_s, 2),
        "scenarios_compared": card.scenarios_compared,
        "scenarios_excluded": card.scenarios_excluded,
        "n_scored": card.cost.n_scored,
        "n_excluded_episodes": card.cost.n_excluded,
        "excluded_cost_usd": round(card.cost.excluded_cost_usd, 4),
        "overhead_components": card.cost.overhead_components,
        "cost_assumptions": card.cost_assumptions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lens", default="cost")
    parser.add_argument("--anchor", default=DEFAULT_ANCHOR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--loo",
        action="store_true",
        help="add leave-one-out routed rungs (refit per fold; every scenario held out once)",
    )
    args = parser.parse_args()

    if args.lens in FROZEN_LENSES:
        script = CORNERS / FROZEN_LENSES[args.lens]
        sys.stdout.write(
            f"lens '{args.lens}' is suspended (charter Amendment 2); delegating to its "
            f"frozen script {script.name} unchanged\n"
        )
        result = subprocess.run([sys.executable, str(script)], check=False)
        sys.stdout.write(f"frozen lens exit code: {result.returncode}\n")
        return

    apply_style()
    lens = load_lens(args.lens)
    dataset = build_dataset(
        anchor_model=args.anchor,
        root=Path(lens.dataset_root).expanduser() if lens.dataset_root else None,
        dataset_label=lens.dataset_label,
        judge=lens.judge_label or GRID_JUDGE,
        loo=args.loo,
    )
    out = render_lens(lens, dataset, args.out_dir)
    numbers = dump_numbers(dataset, lens)
    lines = [
        f"snapshot: {dataset.status}",
        f"records: {len(dataset.records)} (anchor {dataset.anchor_model}, "
        f"best-single {dataset.best_single})",
        f"routed: {len(dataset.routed)} dial points; embedding replay "
        f"{dataset.embedding_replay_calls} calls ~${dataset.embedding_replay_est_usd:.4f}",
        "teacher verdict: "
        + (
            dataset.teacher.reason
            if dataset.teacher is not None
            else f"unavailable ({dataset.teacher_unavailable_reason})"
        ),
        f"wrote {numbers} and figures under {out}",
        *(f"note: {n}" for n in dataset.notes),
    ]
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
