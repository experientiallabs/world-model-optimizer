"""Ground-truth robustness harness for the compaction gates (C2 overnight, menu C).

Four constructed cluster types with KNOWN correct verdicts, in one synthetic grid. The
corrected non-inferiority gate must:

- ACCEPT the genuinely-compressible cluster (quality preserved exactly, cost halved):
  this is the product case the zero-margin superiority gate structurally refused, and the
  reason HIGH-2 exists.
- REFUSE the verbatim-critical cluster (quality drops far beyond the margin).
- REFUSE the lottery cluster (mean delta ~0 with huge cell variance: the SE term, not the
  margin, must do the refusing).
- REFUSE AND FLAG the control-dominant cluster (the real arm passes both gates, but a
  matched-ratio control passes quality with a larger delta at a WORSE cost rank: HIGH-1's
  exact blind spot).

Every cluster carries 40 paired cells (20 scenarios x 2 models), deliberately above
SE_FLOOR_MAX_PAIRS so the small-sample floor is off and the verdicts test the gate's own
arithmetic rather than the floor's backstop. This file is the permanent regression suite
that retires the "gate calibration vs corpus fact" ambiguity: any future gate change must
keep all four verdicts.
"""

from __future__ import annotations

from wmo.optimize.compaction_fit import (
    ArmMatrices,
    aa_report,
    fit_compaction,
    overlay_clusters,
)
from wmo.optimize.compression import CompressionConfig
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.providers.base import ProviderKind, TokenUsage
from wmo.providers.pool import PoolEntry
from wmo.retrieval.embedders import HashingEmbedder

MODELS = ["worker-a", "worker-b"]
SCENARIOS_PER_GROUP = 20  # 40 paired cells per cluster: the SE floor is OFF (>= 30)

# Lexically distant vocabularies so the hashing embedder separates the four ground-truth
# groups into four clusters.
GROUPS = {
    "compressible": "quarterly revenue ledger fiscal audit dividend margin balance capital",
    "verbatim": "order identifier exact serial number code reference token digits precise",
    "lottery": "random draw shuffle coin flip chance dice spin gamble outcome luck",
    "controlled": "python traceback compile refactor merge repository branch commit test",
}

REAL = CompressionConfig(compressor_id="truncate", compressor_version="1", aggressiveness=0.2)
CONTROL = CompressionConfig(compressor_id="identity", compressor_version="1")


def _off_reward(group: str, index: int) -> float:
    if group == "lottery":
        return float(index % 2)  # half succeed: room to swing both ways
    return 1.0 if index % 10 else 0.0  # 90% completion: headroom without degeneracy


def _real_arm_reward(group: str, index: int) -> float:
    off = _off_reward(group, index)
    if group == "compressible":
        return off  # EXACT quality preservation: the product case
    if group == "verbatim":
        return max(0.0, off - 1.0)  # completed tasks now fail: far beyond the margin
    if group == "lottery":
        return 1.0 - off  # every cell flips: mean delta ~0, cell variance ~1
    return min(1.0, off + 0.1) if index % 10 else 0.0  # controlled: a modest real lift


def _control_arm_reward(group: str, index: int) -> float:
    off = _off_reward(group, index)
    if group == "controlled":
        # Dominates the real arm's quality delta (+0.2 vs +0.1 per lifted cell).
        return min(1.0, off + 0.2) if index % 10 else 0.0
    return max(0.0, off - 1.0)  # elsewhere: clearly fails its own quality gate


def _matrix(reward_fn, cost: float, config: CompressionConfig | None) -> OutcomeMatrix:  # noqa: ANN001
    pool = [
        PoolEntry(name=name, kind=ProviderKind.ANTHROPIC, model="claude-haiku-4-5")
        for name in MODELS
    ]
    rows = []
    for group, text in GROUPS.items():
        for index in range(SCENARIOS_PER_GROUP):
            for model in MODELS:
                for episode in range(2):
                    reward = reward_fn(group, index)
                    rows.append(
                        ScenarioOutcome(
                            scenario_id=f"{group}-{index}",
                            task=f"{text} case {index}",
                            model=model,
                            episode=episode,
                            reward=reward,
                            success=reward >= 1.0,
                            usage=TokenUsage(input_tokens=100, output_tokens=10),
                            cost_usd=cost,
                            compressor_id=config.compressor_id if config else "",
                            compressor_version=config.compressor_version if config else "",
                            aggressiveness=config.aggressiveness if config else 0.0,
                        )
                    )
    return OutcomeMatrix(pool=pool, outcomes=rows)


def _ground_truth_fit():  # noqa: ANN202 - shared across assertions below
    off = _matrix(_off_reward, 0.010, None)
    real = _matrix(_real_arm_reward, 0.005, REAL)
    # The control sits BETWEEN the real arm and off on cost, so it never leads the
    # tie-break: the dominance check has to catch it on quality alone (HIGH-1's blind spot).
    control = _matrix(_control_arm_reward, 0.008, CONTROL)
    clusters, assignment = overlay_clusters(
        off, embed_with=HashingEmbedder(dim=512), n_clusters=4, default_model="worker-a"
    )
    groups_seen = {sid.rsplit("-", 1)[0] for sid in assignment}
    by_group = {
        group: {assignment[sid] for sid in assignment if sid.startswith(f"{group}-")}
        for group in groups_seen
    }
    # Each ground-truth group must land in exactly one cluster, all distinct, or the
    # verdicts below would test the clustering rather than the gates.
    assert all(len(ids) == 1 for ids in by_group.values()), by_group
    cluster_of = {group: next(iter(ids)) for group, ids in by_group.items()}
    assert len(set(cluster_of.values())) == 4, cluster_of
    fit = fit_compaction(
        assignment,
        off,
        [
            ArmMatrices(matrix=real, config=REAL),
            ArmMatrices(matrix=control, config=CONTROL, control=True),
        ],
    )
    return off, assignment, cluster_of, fit


def test_compressible_cluster_is_accepted() -> None:
    _off, _assignment, cluster_of, fit = _ground_truth_fit()
    assert fit.per_cluster[cluster_of["compressible"]] == REAL


def test_verbatim_critical_cluster_is_refused() -> None:
    _off, _assignment, cluster_of, fit = _ground_truth_fit()
    assert fit.per_cluster[cluster_of["verbatim"]] is None
    row = next(
        r
        for r in fit.evidence
        if r.cluster_id == cluster_of["verbatim"] and r.signature.startswith("compressor 'trunc")
    )
    assert not row.quality_pass  # refused on quality, not on cost or eligibility


def test_lottery_cluster_is_refused_by_the_se_term() -> None:
    _off, _assignment, cluster_of, fit = _ground_truth_fit()
    assert fit.per_cluster[cluster_of["lottery"]] is None
    row = next(
        r
        for r in fit.evidence
        if r.cluster_id == cluster_of["lottery"] and r.signature.startswith("compressor 'trunc")
    )
    # Mean inside the margin but the confidence bound far outside it: the SE term refuses.
    assert abs(row.mean_diff) <= 0.02
    assert row.se > 0.1
    assert not row.quality_pass


def test_control_dominant_cluster_is_refused_and_flagged() -> None:
    _off, _assignment, cluster_of, fit = _ground_truth_fit()
    cluster_id = cluster_of["controlled"]
    assert fit.per_cluster[cluster_id] is None
    assert any(
        "quality-dominates" in flag and f"cluster={cluster_id}" in flag
        for flag in fit.control_flags
    )
    winner = next(
        r
        for r in fit.evidence
        if r.cluster_id == cluster_id and r.signature.startswith("compressor 'trunc")
    )
    # The real arm PASSED both gates and was cheaper; only the dominance check stops it.
    assert winner.quality_pass and winner.cost_pass and winner.control_dominated


def test_only_the_compressible_cluster_is_compressed() -> None:
    _off, _assignment, _cluster_of, fit = _ground_truth_fit()
    assert fit.compressed_clusters() == 1


def test_aa_bar_is_clean_on_the_synthetic_off_arm() -> None:
    off, assignment, _cluster_of, _fit = _ground_truth_fit()
    assert aa_report(assignment, off) == []
