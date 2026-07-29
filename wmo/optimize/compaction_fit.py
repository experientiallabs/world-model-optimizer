"""Per-cluster compaction fitter: populate `ClusterRanking.compression` from measured arms.

#265 merged the artifact field ("this cluster's compression choice, fitted jointly with the
model ranking") with nothing fitting it. This module is the fitter. Its inputs are the grid's
per-arm outcome matrices: one OFF arm (uncompressed) plus one matrix per measured compression
config, all on the SAME scenario cohort, each self-describing via
`OutcomeMatrix.measured_compression()`. Its output is a per-cluster
`CompressionConfig | None` map plus the paired evidence behind every decision.

The choice rule is conservative by design: UNCOMPRESSED IS THE FALLBACK, and a cluster
receives a compressed config only on statistical evidence it does not lose quality AND a
measured effective-cost win. Concretely, an arm passes in a cluster iff

- quality (NON-INFERIORITY, the 2026-07-28 gate ruling): the paired per-cell reward delta
  (arm minus off, cells = (scenario, model) means over scored episodes) satisfies
  mean - z * SE >= -margin on at least `min_pairs` paired cells, with margin = 0.02 absolute
  reward (the fleet's noise-floor bar) and the knn guard's small-sample SE floor
  (`SE_FLOOR_MAX_PAIRS`). An arm that PRESERVES quality at lower cost passes, which is the
  product case; the earlier zero-margin superiority form structurally refused it, and that
  miscalibration was most of round 2's "0/8" headline. Note the floor's interaction: below
  30 pairs the floored SE (sqrt(0.25/n)) exceeds the margin at any practical z, so thin
  clusters still cannot deviate, by construction rather than by accident.
- cost: the arm's effective cost per COMPLETED task on those cells, compressor bill folded
  as a `RowOverhead`, beats the off arm's on the same cells. Input-token accounting alone is
  banned here: the grid measured dumb deletion RAISING effective cost by lengthening
  episodes, so only the completed-task metric can gate.
- control dominance (the 2026-07-28 HIGH-1 ruling): no control arm in the cluster may pass
  its own quality gate with a delta at or above the winner's, REGARDLESS of cost rank. A
  matched-ratio control matching the winner's quality signal means the signal is not
  compression-specific; the cluster is refused and flagged, never silently stamped.

Among passing arms the tie-break is lowest effective cost, then lower aggressiveness. Arms
are MEASURED points only: the fitter never interpolates an aggressiveness the grid did not
run, matching the D-DIAL v2 ruling that aggressiveness is a step function over mounted
artifacts. Every arm's config is verified against its matrix's own
`measured_compression()`, so a caller cannot stamp an arm that was never measured.

Two mandatory validations ride along. The A/A bar (`aa_report`) splits the off arm by episode
parity and runs the same gates: a guard loose enough to compress on noise fails there before
it can ship. Control arms (`control=True`) are evaluated and reported but never chosen.

Evidence provenance (the 2026-07-28 MEDIUM-3 ruling): every fit records where its matrices
came from and which scorer era produced their rewards; a fit whose era is unlabeled or
known-broken carries a pending-rescore quality label that consumers must surface.

Representation rule (proposed in DECISIONS 2026-07-27, co-signed 2026-07-28 with
conditions): a policy carrying per-cluster configs routes on RAW text (policy-level
`compression` and `fit_compression` both None) and compresses after cluster assignment.
`apply_compaction` enforces that exclusivity; the serving-side delta lands separately
behind the routing-lane co-sign.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

from wmo.optimize.compression import (
    CompressionConfig,
    compression_signature,
    same_compression,
    servable_compressor,
)
from wmo.optimize.policy import (
    SE_FLOOR_MAX_PAIRS,
    ClusterRanking,
    EmbedderSpec,
    RoutingPolicy,
    write_artifact_atomically,
)
from wmo.optimize.scorecard import (
    DEFAULT_COMPLETION,
    CompletionRule,
    RowOverhead,
    effective_cost_per_completed_task,
)

if TYPE_CHECKING:
    from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
    from wmo.providers.base import Embedder

logger = logging.getLogger(__name__)

DEFAULT_COMPACTION_Z = 0.5  # the routing guard's confidence bar, reused deliberately
DEFAULT_COMPACTION_MIN_PAIRS = 8  # below this a cluster never deviates from uncompressed
# The ruled non-inferiority margin: 0.02 absolute reward, the fleet's established noise-floor
# bar (ladder verdict rule), adopted by the 2026-07-28 gate ruling. 0.0 recovers the
# superiority form for comparison sweeps; serving fits use the ruled default.
DEFAULT_NONINFERIORITY_MARGIN = 0.02

# The A/A pseudo-arm's placeholder identity. It exists only inside `aa_report` evidence rows;
# it is marked control, is never eligible, and never reaches an artifact.
AA_SIGNATURE = "aa-episode-split"


class EvidenceProvenance(BaseModel):
    """Where a fit's evidence came from, carried on every output (MEDIUM-3 ruling).

    `scorer_era` names the judge/scorer state the matrices' rewards were produced under
    (e.g. "post-rescore-2026-07-28"); empty or containing "broken" means the quality axis is
    not currently trustworthy and `quality_label` says so. Consumers surface the label; they
    do not get to average a pending-rescore number into a headline silently.
    """

    off_source: str = ""
    arm_sources: dict[str, str] = Field(default_factory=dict)  # signature -> matrix source
    scorer_era: str = ""
    quality_label: str = ""  # "" = trusted; "pending-rescore" = quality side not yet believable

    @classmethod
    def build(
        cls, *, off_source: str, arm_sources: dict[str, str], scorer_era: str
    ) -> EvidenceProvenance:
        broken = not scorer_era or "broken" in scorer_era.lower()
        return cls(
            off_source=off_source,
            arm_sources=arm_sources,
            scorer_era=scorer_era,
            quality_label="pending-rescore" if broken else "",
        )


class ClusterArmEvidence(BaseModel):
    """The paired statistics behind one (cluster, arm) decision, kept auditable.

    `would_win` is true when the arm passed both gates and led the tie-break regardless of
    eligibility, so a control or unservable arm that would have taken the cluster is visible
    instead of silently skipped. `chosen` additionally requires eligible, not control, and
    not control-dominated. `control_dominated` marks a winner refused because a control
    matched or beat its quality delta (HIGH-1).
    """

    cluster_id: int
    signature: str  # compression_signature of the arm
    config: CompressionConfig | None = None  # None only for the A/A pseudo-arm
    control: bool = False
    eligible: bool = True  # servable_compressor accepted the config
    n_pairs: int = 0
    mean_diff: float = 0.0
    se: float = 0.0
    quality_pass: bool = False
    off_cost_per_completed: float | None = None
    arm_cost_per_completed: float | None = None
    cost_pass: bool = False
    would_win: bool = False
    control_dominated: bool = False
    chosen: bool = False


class CompactionFit(BaseModel):
    """The fitter's output: the per-cluster map plus everything needed to audit it."""

    per_cluster: dict[int, CompressionConfig | None]
    evidence: list[ClusterArmEvidence]
    z: float
    min_pairs: int
    margin: float = DEFAULT_NONINFERIORITY_MARGIN
    # Which selection rule chose among passing arms: "cost" (default) or "aggressiveness"
    # (calibrator-v1). Part of the fit's identity; consumers must not mix maps across rules.
    selection: str = "cost"
    coverage: list[str] = Field(default_factory=list)  # human-readable coverage notes
    # Investigation flags, one string per event: a control led the tie-break, or a control
    # quality-dominated the winner. Any entry means the fit must not ship without a human read.
    control_flags: list[str] = Field(default_factory=list)
    provenance: EvidenceProvenance = Field(default_factory=EvidenceProvenance)

    def compressed_clusters(self) -> int:
        return sum(1 for config in self.per_cluster.values() if config is not None)


class ArmMatrices(BaseModel):
    """One measured arm: its matrix and how the fitter should treat it.

    `pseudo` is reserved for `aa_report`'s episode-split arm, whose placeholder config
    deliberately does not match its matrix; every real arm is verified against
    `measured_compression()`.
    """

    model_config = {"arbitrary_types_allowed": True}

    matrix: object  # OutcomeMatrix; typed loosely to avoid a runtime import cycle
    config: CompressionConfig
    control: bool = False
    pseudo: bool = False


def _cells(matrix: OutcomeMatrix) -> dict[tuple[str, str], list[ScenarioOutcome]]:
    """Scored rows grouped by (scenario_id, model): the unit quality is paired on."""
    cells: dict[tuple[str, str], list[ScenarioOutcome]] = defaultdict(list)
    for row in matrix.outcomes:
        if row.scored:
            cells[(row.scenario_id, row.model)].append(row)
    return cells


def _cell_reward(rows: list[ScenarioOutcome]) -> float:
    return float(np.mean([row.reward for row in rows]))


def _compressor_overheads(rows: list[ScenarioOutcome]) -> list[RowOverhead]:
    """The rows' own compressor bill as scorecard overheads (the #294 convention)."""
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


def _effective_cost(rows: list[ScenarioOutcome], completion: CompletionRule) -> float | None:
    if not rows:
        return None
    result = effective_cost_per_completed_task(
        rows, overheads=_compressor_overheads(rows), completion=completion
    )
    return result.cost_per_completed_task_usd


def check_cohort(
    off: OutcomeMatrix, arms: list[ArmMatrices], *, allow_uneven: bool = False
) -> list[str]:
    """Enforce that every arm was measured on the off arm's cohort, both directions.

    Strict mode (the default) requires identical scored (scenario, model) cell sets: a grid arm
    measured on a subset ranks its clusters on DIFFERENT evidence, which is the bias the route
    sweep's coverage gate exists to block. `allow_uneven` downgrades the mismatch to printed
    notes; pairing then happens on the intersection per cluster, and the caller labels the fit
    accordingly (the grid mid-flight is the intended use).
    """
    notes: list[str] = []
    off_cells = set(_cells(off))
    for arm in arms:
        arm_cells = set(_cells(arm.matrix))
        missing = len(off_cells - arm_cells)
        extra = len(arm_cells - off_cells)
        if missing or extra:
            notes.append(
                f"{compression_signature(arm.config)}: {missing} off cells unmeasured on this "
                f"arm, {extra} arm cells absent from off ({len(arm_cells & off_cells)} shared)"
            )
    if notes and not allow_uneven:
        raise ValueError(
            "arm matrices do not cover the off arm's cohort; pairing on a subset biases the "
            "per-cluster ranking toward whichever cells each arm happened to keep:\n  "
            + "\n  ".join(notes)
            + "\nRe-run the missing cells, or pass allow_uneven for an interim, "
            "candidate-labeled fit."
        )
    return notes


def overlay_clusters(
    matrix: OutcomeMatrix,
    *,
    embed_with: Embedder,
    n_clusters: int = 8,
    seed: int = 42,
    default_model: str,
) -> tuple[list[ClusterRanking], dict[str, int]]:
    """K-means compression-overlay clusters on the off arm's raw task embeddings.

    For `kind="rank"` policies the fitted clusters already exist and the caller uses
    `assign_to_clusters` against them instead. This overlay is for `kind="knn"` policies, which
    route without clusters: the overlay rides on the artifact solely so the compress stage has
    an assignment surface, fitted on the SAME raw geometry the router embeds queries in (the
    proposed serving delta reuses the decision's own query embedding, so overlay centroids in
    any other space would be meaningless).

    The `ranking` field is stamped `[default_model]`: it satisfies the model's shape and makes
    plain that routing never reads these clusters. Conventions mirror `fit_rank_policy`
    (L2-normalized embeddings, k-means++ with the reference seed).
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import Normalizer

    tasks: dict[str, str] = {}
    for outcome in matrix.outcomes:
        tasks.setdefault(outcome.scenario_id, outcome.task)
    scenario_ids = list(tasks)
    if not scenario_ids:
        raise ValueError("no scenarios to cluster")
    embeddings = np.asarray(embed_with.embed([tasks[sid] for sid in scenario_ids]))
    embeddings = Normalizer(norm="l2").fit_transform(embeddings)
    k = min(n_clusters, len(scenario_ids))
    kmeans = KMeans(
        n_clusters=k,
        random_state=seed,
        init="k-means++",
        n_init="auto",
        max_iter=1000,
        algorithm="elkan",
    )
    labels = kmeans.fit_predict(embeddings)
    assignment = {sid: int(label) for sid, label in zip(scenario_ids, labels, strict=True)}
    counts = defaultdict(int)
    for cluster in assignment.values():
        counts[cluster] += 1
    clusters = [
        ClusterRanking(
            cluster_id=cluster_id,
            label=f"compaction-{cluster_id}",
            centroid=kmeans.cluster_centers_[cluster_id].tolist(),
            ranking=[default_model],
            total=counts[cluster_id],
        )
        for cluster_id in range(k)
    ]
    return clusters, assignment


def assign_to_clusters(
    clusters: list[ClusterRanking], matrix: OutcomeMatrix, *, embed_with: Embedder
) -> dict[str, int]:
    """Nearest-centroid assignment of the matrix's scenarios to existing policy clusters.

    Used on `kind="rank"` policies so the compaction gates run on THE clusters the policy
    routes with, not a parallel clustering that would drift from it.
    """
    tasks: dict[str, str] = {}
    for outcome in matrix.outcomes:
        tasks.setdefault(outcome.scenario_id, outcome.task)
    scenario_ids = list(tasks)
    embeddings = np.asarray(embed_with.embed([tasks[sid] for sid in scenario_ids]))
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms == 0.0, 1.0, norms)
    centroids = np.asarray([c.centroid for c in clusters])
    cluster_ids = [c.cluster_id for c in clusters]
    sims = embeddings @ centroids.T
    return {sid: cluster_ids[int(np.argmax(sims[index]))] for index, sid in enumerate(scenario_ids)}


# Selection rules among gate-passing arms. "cost" is the conservative default (cheapest,
# then least aggressive). "aggressiveness" is calibrator-v1 (the learned-calibration-ratio
# directive): the HIGHEST aggressiveness whose arm passes BOTH gates, so the learned
# operating point inherits the corrected gate's guarantee instead of a point estimate.
# Either way the choice is made at cluster/artifact grain from MEASURED arms only, and the
# rule is part of the fit's identity (recorded on the output; a different rule = a
# different artifact, exactly like a compressor version).
SELECTION_RULES = ("cost", "aggressiveness")
CALIBRATOR_VERSION = "calibrator-v1"


def fit_compaction(
    assignment: dict[str, int],
    off: OutcomeMatrix,
    arms: list[ArmMatrices],
    *,
    z: float = DEFAULT_COMPACTION_Z,
    min_pairs: int = DEFAULT_COMPACTION_MIN_PAIRS,
    margin: float = DEFAULT_NONINFERIORITY_MARGIN,
    completion: CompletionRule = DEFAULT_COMPLETION,
    allow_uneven: bool = False,
    provenance: EvidenceProvenance | None = None,
    selection: str = "cost",
) -> CompactionFit:
    """The gates: per cluster, choose among measured arms or stay uncompressed.

    `assignment` maps scenario_id to cluster_id (from the policy's own clusters via
    `assign_to_clusters`, or from `overlay_clusters`). See the module docstring for the rule;
    every decision's paired statistics are returned as evidence rows.
    """
    if selection not in SELECTION_RULES:
        raise ValueError(f"unknown selection rule '{selection}'; use one of {SELECTION_RULES}")
    if off.measured_compression() is not None:
        raise ValueError("the off arm must be uncompressed; got a matrix with a compression arm")
    signatures = [compression_signature(arm.config) for arm in arms]
    if len(set(signatures)) != len(signatures):
        raise ValueError(f"duplicate arm signatures: {sorted(signatures)}")
    for arm in arms:
        # Measured arms only, verified: a config that disagrees with what the matrix's episodes
        # actually ran under would stamp an arm that was never measured. The CLI reads configs
        # FROM the matrices so it cannot hit this; named future consumers calling the module
        # directly can, which is why it is asserted here and not only at the boundary.
        if not arm.pseudo and not same_compression(arm.config, arm.matrix.measured_compression()):
            raise ValueError(
                f"arm config {compression_signature(arm.config)} does not match its matrix's "
                f"measured arm {compression_signature(arm.matrix.measured_compression())}; "
                "the fitter only stamps configs whose episodes were actually measured"
            )
    coverage = check_cohort(off, arms, allow_uneven=allow_uneven)

    off_cells = _cells(off)
    cluster_ids = sorted(set(assignment.values()))
    evidence: list[ClusterArmEvidence] = []
    per_cluster: dict[int, CompressionConfig | None] = {}
    control_flags: list[str] = []

    for cluster_id in cluster_ids:
        member_cells = [key for key in off_cells if assignment.get(key[0]) == cluster_id]
        candidates: list[ClusterArmEvidence] = []
        for arm in arms:
            signature = compression_signature(arm.config)
            arm_cells = _cells(arm.matrix)
            paired = [key for key in member_cells if key in arm_cells]
            diffs = np.asarray(
                [_cell_reward(arm_cells[key]) - _cell_reward(off_cells[key]) for key in paired]
            )
            n_pairs = int(diffs.size)
            mean_diff = float(diffs.mean()) if n_pairs else 0.0
            se = float(diffs.std(ddof=1)) / n_pairs**0.5 if n_pairs > 1 else 0.0
            if 0 < n_pairs < SE_FLOOR_MAX_PAIRS:
                se = max(se, (0.25 / n_pairs) ** 0.5)
            # NON-INFERIORITY (the ruled gate): the lower confidence bound at z must clear
            # -margin, not 0. Quality preservation at lower cost passes; quality loss beyond
            # the fleet's noise-floor bar cannot.
            quality_pass = n_pairs >= min_pairs and mean_diff - z * se >= -margin

            arm_rows = [row for key in paired for row in arm_cells[key]]
            off_rows = [row for key in paired for row in off_cells[key]]
            arm_cost = _effective_cost(arm_rows, completion)
            off_cost = _effective_cost(off_rows, completion)
            cost_pass = arm_cost is not None and off_cost is not None and arm_cost < off_cost

            try:
                servable_compressor(arm.config)
                eligible = True
            except ValueError:
                eligible = False
            candidates.append(
                ClusterArmEvidence(
                    cluster_id=cluster_id,
                    signature=signature,
                    config=arm.config,
                    control=arm.control,
                    eligible=eligible,
                    n_pairs=n_pairs,
                    mean_diff=mean_diff,
                    se=se,
                    quality_pass=quality_pass,
                    off_cost_per_completed=off_cost,
                    arm_cost_per_completed=arm_cost,
                    cost_pass=cost_pass,
                )
            )

        passing = [c for c in candidates if c.quality_pass and c.cost_pass]
        per_cluster[cluster_id] = None
        if passing:
            if selection == "aggressiveness":
                # calibrator-v1: most aggressive passing arm, cheaper first on ties.
                ranked = sorted(
                    passing,
                    key=lambda c: (-c.config.aggressiveness, c.arm_cost_per_completed),
                )
            else:
                # Conservative default: lowest effective cost, then lower aggressiveness.
                ranked = sorted(
                    passing,
                    key=lambda c: (c.arm_cost_per_completed, c.config.aggressiveness),
                )
            ranked[0].would_win = True
            if ranked[0].control:
                control_flags.append(
                    f"cluster={cluster_id} {ranked[0].signature} led the tie-break"
                )
            # CONTROL DOMINANCE (HIGH-1): a control passing its own quality gate with a
            # delta at or above the leading real arm's means the cluster's signal is not
            # compression-specific, REGARDLESS of which arm is cheaper. The flag fires even
            # when the leading arm is ineligible (a surface sweep must see the dominance,
            # not have eligibility hide it); the refusal applies to whatever would be chosen.
            leader = next((c for c in ranked if not c.control), None)
            if leader is not None:
                dominating = [
                    c
                    for c in candidates
                    if c.control and c.quality_pass and c.mean_diff >= leader.mean_diff
                ]
                if dominating:
                    leader.control_dominated = True
                    for control in dominating:
                        control_flags.append(
                            f"cluster={cluster_id} control {control.signature} quality-dominates "
                            f"{leader.signature} ({control.mean_diff:+.4f} >= "
                            f"{leader.mean_diff:+.4f})"
                        )
                elif leader.eligible:
                    leader.chosen = True
                    per_cluster[cluster_id] = leader.config
        evidence.extend(candidates)

    fit = CompactionFit(
        per_cluster=per_cluster,
        evidence=evidence,
        z=z,
        min_pairs=min_pairs,
        margin=margin,
        selection=selection,
        coverage=coverage,
        control_flags=control_flags,
        provenance=provenance or EvidenceProvenance(),
    )
    logger.info(
        "compaction fit: %d/%d clusters compressed (z=%g, margin=%g, min_pairs=%d)%s%s",
        fit.compressed_clusters(),
        len(cluster_ids),
        z,
        margin,
        min_pairs,
        f"; CONTROL FLAGS: {control_flags}" if control_flags else "",
        f" [{fit.provenance.quality_label}]" if fit.provenance.quality_label else "",
    )
    return fit


def aa_pseudo_arms(off: OutcomeMatrix) -> tuple[OutcomeMatrix, OutcomeMatrix]:
    """Split the off arm by episode parity into two pseudo-arms measuring NOTHING.

    Any per-cluster difference between the two sides is noise by construction, which is what
    makes them the fitter's A/A control: gates loose enough to deviate here would compress on
    noise in production. Episode indices are renumbered per side so each pseudo-matrix is
    well-formed on its own.
    """
    from wmo.optimize.outcomes import OutcomeMatrix

    even: list[ScenarioOutcome] = []
    odd: list[ScenarioOutcome] = []
    counters: dict[tuple[str, str, int], int] = defaultdict(int)
    for row in off.outcomes:
        side = even if row.episode % 2 == 0 else odd
        key = (row.scenario_id, row.model, row.episode % 2)
        side.append(row.model_copy(update={"episode": counters[key]}))
        counters[key] += 1
    return (
        OutcomeMatrix(pool=off.pool, outcomes=even),
        OutcomeMatrix(pool=off.pool, outcomes=odd),
    )


def aa_report(
    assignment: dict[str, int],
    off: OutcomeMatrix,
    *,
    z: float = DEFAULT_COMPACTION_Z,
    min_pairs: int = DEFAULT_COMPACTION_MIN_PAIRS,
    margin: float = DEFAULT_NONINFERIORITY_MARGIN,
    completion: CompletionRule = DEFAULT_COMPLETION,
) -> list[ClusterArmEvidence]:
    """The A/A kill bar: run the gates on the episode-parity pseudo-arms.

    Returns the evidence rows whose gates BOTH passed (the deviations); an empty list is the
    bar passing. The pseudo-arm is marked control and carries a placeholder config, so nothing
    from this report can reach an artifact even if a caller mishandles it. Runs at the SAME
    margin as the real fit: the margin loosens the quality gate, so an A/A bar run without it
    would certify a z the real gate then betrays.
    """
    pseudo_off, pseudo_arm = aa_pseudo_arms(off)
    placeholder = CompressionConfig(compressor_id=AA_SIGNATURE, compressor_version="0")
    rows: list[ClusterArmEvidence] = []
    fit = fit_compaction(
        assignment,
        pseudo_off,
        [ArmMatrices(matrix=pseudo_arm, config=placeholder, control=True, pseudo=True)],
        z=z,
        min_pairs=min_pairs,
        margin=margin,
        completion=completion,
        allow_uneven=True,  # single-episode cells exist on one side only; pairing drops them
    )
    for row in fit.evidence:
        row.signature = AA_SIGNATURE
        row.config = None
        if row.quality_pass and row.cost_pass:
            rows.append(row)
    return rows


# `models/policy.json` -> `models/policy.json.compaction.json`: same derivation shape as
# `knn_bank_path_for`, same reason (distinct policy filenames always give distinct sidecar
# filenames, and one owner for the derivation means nothing can disagree about which map
# belongs to which policy).
COMPACTION_SUFFIX = ".compaction.json"


def compaction_path_for(policy_path: Path) -> Path:
    """The compaction sidecar that belongs to one policy file (mirrors `knn_bank_path_for`)."""
    return policy_path.with_name(f"{policy_path.name}{COMPACTION_SUFFIX}")


class CompactionArtifact(BaseModel):
    """The per-cluster compaction map as a policy SIDECAR (knn policies, v1).

    The merged `RoutingPolicy` validator forbids clusters on a knn policy, so the overlay
    cannot ride on the policy model without the co-signed contract delta. Until that lands,
    the map ships beside the policy the way the knn bank itself does: the clusters here carry
    centroids in the SAME raw embedding geometry the policy routes in, plus their fitted
    compression configs.

    Identity binding (MEDIUM-4 ruling): `policy_digest` is the sha256 of the policy file the
    map was fitted beside, and `embedder` is that policy's embedding spec. A consumer loads
    the sidecar through `load_compaction_sidecar`, which refuses both mismatches: mounting a
    foreign policy's centroids is the C2 floor-failure class (geometry that looks valid and
    silently routes wrong), so it must fail closed, not load. Nothing serves this yet; the
    serving delta is gated on the routing-lane co-sign.
    """

    clusters: list[ClusterRanking]
    z: float
    min_pairs: int
    margin: float = DEFAULT_NONINFERIORITY_MARGIN
    policy_digest: str = ""
    embedder: EmbedderSpec = Field(default_factory=EmbedderSpec)
    provenance: EvidenceProvenance = Field(default_factory=EvidenceProvenance)
    fitted_from: str | None = None


def _policy_digest(policy_path: Path) -> str:
    return hashlib.sha256(policy_path.read_bytes()).hexdigest()


def save_compaction_sidecar(
    policy_path: Path,
    policy: RoutingPolicy,
    clusters: list[ClusterRanking],
    fit: CompactionFit,
    *,
    fitted_from: str | None = None,
) -> CompactionArtifact:
    """Write the stamped overlay beside `policy_path` (see `CompactionArtifact`).

    The policy must already be SAVED at `policy_path` (the digest binds to the bytes on
    disk), must be raw-routed (the exclusivity rule applies to the sidecar exactly as it
    applies to `apply_compaction`: a per-cluster map beside an endpoint-level compression
    config is two representations in one artifact directory), and every stamped config must
    be servable. The write is atomic, so a failure leaves any previous sidecar intact.
    """
    if policy.compression is not None or policy.fit_compression is not None:
        raise ValueError(
            "per-cluster compaction requires a raw-routed policy: policy-level compression "
            f"is {compression_signature(policy.compression)} and fit_compression is "
            f"{compression_signature(policy.fit_compression)}; per-cluster mode and "
            "endpoint-level mode never mix, on the sidecar exactly as on the policy"
        )
    if not policy_path.exists():
        raise ValueError(
            f"policy file {policy_path} does not exist; save the policy first, the sidecar's "
            "digest binds to its bytes"
        )
    stamped = [
        cluster.model_copy(update={"compression": fit.per_cluster.get(cluster.cluster_id)})
        for cluster in clusters
    ]
    for cluster in stamped:
        if cluster.compression is not None:
            servable_compressor(cluster.compression)
    artifact = CompactionArtifact(
        clusters=stamped,
        z=fit.z,
        min_pairs=fit.min_pairs,
        margin=fit.margin,
        policy_digest=_policy_digest(policy_path),
        embedder=policy.embedder,
        provenance=fit.provenance,
        fitted_from=fitted_from,
    )
    write_artifact_atomically(
        compaction_path_for(policy_path), artifact.model_dump_json(indent=2).encode()
    )
    return artifact


def load_compaction_sidecar(policy_path: Path, policy: RoutingPolicy) -> CompactionArtifact:
    """Load and VERIFY the compaction sidecar belonging to `policy_path`.

    Refuses a digest mismatch (the policy file changed since the map was fitted beside it)
    and an embedder mismatch (the centroids live in a different geometry than the one the
    policy routes in). Either would mount foreign-geometry centroids, which is the measured
    C2 floor-failure class; the consumer contract is fail-closed.
    """
    path = compaction_path_for(policy_path)
    artifact = CompactionArtifact.model_validate_json(path.read_text())
    digest = _policy_digest(policy_path)
    if artifact.policy_digest != digest:
        raise ValueError(
            f"compaction sidecar {path.name} was fitted beside a different policy file "
            f"(digest {artifact.policy_digest[:12]}.. != {digest[:12]}..); refit the map for "
            "this policy rather than mounting another policy's centroids"
        )
    if artifact.embedder != policy.embedder:
        raise ValueError(
            f"compaction sidecar {path.name} carries centroids in a different embedding "
            "geometry than this policy routes in; a nearest-centroid assignment across "
            "geometries is the measured floor-failure class, so it does not load"
        )
    return artifact


def apply_compaction(policy: RoutingPolicy, fit: CompactionFit) -> RoutingPolicy:
    """Stamp the fitted per-cluster map onto `policy` under the exclusivity rule.

    The co-signed representation rule (DECISIONS 2026-07-27/28): a policy carrying
    per-cluster configs routes on RAW text, so policy-level `compression` and
    `fit_compression` must both be None; mixing modes in one artifact is refused here, not
    discovered at mount. Every stamped config is checked servable, so a control or churny
    arm that leaked into the map fails closed at stamp time.

    Cluster-carrying kinds only (`rank`): a knn policy cannot carry clusters under the merged
    validator, so its map goes through `save_compaction_sidecar` instead.
    """
    if policy.compression is not None or policy.fit_compression is not None:
        raise ValueError(
            "per-cluster compaction requires a raw-routed policy: policy-level compression "
            f"is {compression_signature(policy.compression)} and fit_compression is "
            f"{compression_signature(policy.fit_compression)}. Fit the base policy without "
            "--compressor; per-cluster mode and endpoint-level mode never mix."
        )
    if not policy.clusters:
        raise ValueError(
            "policy has no clusters to stamp; fit a rank policy or attach overlay_clusters first"
        )
    known = {cluster.cluster_id for cluster in policy.clusters}
    unknown = sorted(set(fit.per_cluster) - known)
    if unknown:
        raise ValueError(f"fit names cluster ids absent from the policy: {unknown[:5]}")
    for cluster_id, config in fit.per_cluster.items():
        if config is not None:
            servable_compressor(config)
        del cluster_id
    clusters = [
        cluster.model_copy(update={"compression": fit.per_cluster.get(cluster.cluster_id)})
        for cluster in policy.clusters
    ]
    return policy.model_copy(update={"clusters": clusters})
