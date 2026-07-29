"""Tests for the per-cluster compaction fitter.

Pure offline stats over synthetic outcome matrices: no provider, no judge, no spend. The
synthetic grid has two well-separated scenario groups (financey vs codey task texts, so the
hashing embedder clusters them apart) and arms constructed so exactly one cluster carries a
genuine compression win, which is what the gates must find and the only thing they may find.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from wmo.optimize.compaction_fit import (
    ArmMatrices,
    aa_report,
    apply_compaction,
    assign_to_clusters,
    fit_compaction,
    overlay_clusters,
)
from wmo.optimize.compression import CompressionConfig
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.policy import EmbedderSpec, RoutingPolicy
from wmo.providers.base import ProviderKind, TokenUsage
from wmo.providers.pool import PoolEntry
from wmo.retrieval.embedders import HashingEmbedder

MODELS = ["worker-a", "worker-b"]
# Two lexically distant scenario groups so the hashing embedder separates them cleanly.
GROUP_TEXTS = {
    "fin": "quarterly revenue balance sheet capital expenditure fiscal audit dividend",
    "code": "python traceback refactor unit test compile stack trace repository merge",
}


def _pool() -> list[PoolEntry]:
    return [
        PoolEntry(name=name, kind=ProviderKind.ANTHROPIC, model="claude-haiku-4-5")
        for name in MODELS
    ]


def _row(
    sid: str,
    task: str,
    model: str,
    episode: int,
    reward: float,
    *,
    cost: float = 0.01,
    config: CompressionConfig | None = None,
) -> ScenarioOutcome:
    return ScenarioOutcome(
        scenario_id=sid,
        task=task,
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


def _scenarios() -> list[tuple[str, str, str, int]]:
    """(sid, task, group, index) for 12 scenarios, 6 per group."""
    out = []
    for group, text in GROUP_TEXTS.items():
        for index in range(6):
            out.append((f"{group}-{index}", f"{text} variant {index}", group, index))
    return out


def _matrix(
    reward_fn: Callable[[str, int], float],
    cost_fn: Callable[[str, int], float],
    config: CompressionConfig | None = None,
    episodes: int = 2,
) -> OutcomeMatrix:
    """reward_fn/cost_fn take (group, scenario_index)."""
    rows = []
    for sid, task, group, index in _scenarios():
        for model in MODELS:
            for episode in range(episodes):
                rows.append(
                    _row(
                        sid,
                        task,
                        model,
                        episode,
                        reward_fn(group, index),
                        cost=cost_fn(group, index),
                        config=config,
                    )
                )
    return OutcomeMatrix(pool=_pool(), outcomes=rows)


def _off_rewards(group: str, index: int) -> float:
    """The off arm leaves headroom on fin (half its scenarios fail) and is perfect on code.

    The registered quality gate demands POSITIVE paired evidence (mean - z*SE >= 0 under the
    small-sample SE floor), so a winning arm must genuinely lift fin, not merely match it.
    """
    if group == "fin":
        return 1.0 if index < 3 else 0.0
    return 1.0


def _winning_arm(*, cost: float = 0.005, config: CompressionConfig) -> OutcomeMatrix:
    """Lifts fin to perfect at lower cost; wrecks code (quality gate must block it there)."""
    return _matrix(
        lambda group, index: 1.0 if group == "fin" else 0.0,
        lambda group, index: cost,
        config=config,
    )


TRUNCATE = CompressionConfig(compressor_id="truncate", compressor_version="1", aggressiveness=0.2)
UNKNOWN = CompressionConfig(compressor_id="not-registered", compressor_version="1")


@pytest.fixture
def off() -> OutcomeMatrix:
    return _matrix(_off_rewards, lambda group, index: 0.01)


@pytest.fixture
def assignment(off: OutcomeMatrix) -> dict[str, int]:
    clusters, assignment = overlay_clusters(
        off, embed_with=HashingEmbedder(dim=256), n_clusters=2, default_model="worker-a"
    )
    # The two lexical groups must land in different clusters or every test below is vacuous.
    fin = {assignment[sid] for sid, _, group, _i in _scenarios() if group == "fin"}
    code = {assignment[sid] for sid, _, group, _i in _scenarios() if group == "code"}
    assert len(fin) == 1 and len(code) == 1 and fin != code
    return assignment


def test_win_cluster_gets_the_config_and_loss_cluster_stays_none(
    off: OutcomeMatrix, assignment: dict[str, int]
) -> None:
    # The arm lifts fin (positive paired evidence) at lower cost, and wrecks code.
    arm = _winning_arm(config=TRUNCATE)
    fit = fit_compaction(assignment, off, [ArmMatrices(matrix=arm, config=TRUNCATE)])
    fin_cluster = assignment["fin-0"]
    code_cluster = assignment["code-0"]
    assert fit.per_cluster[fin_cluster] == TRUNCATE
    assert fit.per_cluster[code_cluster] is None


def test_matching_quality_is_not_positive_evidence(
    off: OutcomeMatrix, assignment: dict[str, int]
) -> None:
    # An arm identical to off on quality: mean paired delta is 0, and the floored SE makes
    # the registered gate (mean - z*SE >= 0) fail. Preserving quality is not evidence; this
    # is the strictness the A/A bar depends on.
    arm = _matrix(_off_rewards, lambda group, index: 0.001, config=TRUNCATE)
    fit = fit_compaction(assignment, off, [ArmMatrices(matrix=arm, config=TRUNCATE)])
    assert fit.compressed_clusters() == 0
    assert all(not row.quality_pass for row in fit.evidence)


def test_quality_pass_alone_is_not_enough_without_a_cost_win(
    off: OutcomeMatrix, assignment: dict[str, int]
) -> None:
    # The arm lifts fin but costs MORE per completed task (the dumb-deletion failure the
    # grid measured): the cost gate must hold every cluster at None.
    arm = _winning_arm(cost=0.05, config=TRUNCATE)
    fit = fit_compaction(assignment, off, [ArmMatrices(matrix=arm, config=TRUNCATE)])
    assert fit.compressed_clusters() == 0
    fin_rows = [r for r in fit.evidence if r.cluster_id == assignment["fin-0"]]
    assert all(row.quality_pass and not row.cost_pass for row in fin_rows)


def test_thin_clusters_never_deviate(off: OutcomeMatrix, assignment: dict[str, int]) -> None:
    # min_pairs above the cluster size: no deviation regardless of how good the arm looks.
    arm = _winning_arm(config=TRUNCATE)
    fit = fit_compaction(assignment, off, [ArmMatrices(matrix=arm, config=TRUNCATE)], min_pairs=99)
    assert fit.compressed_clusters() == 0


def test_controls_are_reported_but_never_chosen(
    off: OutcomeMatrix, assignment: dict[str, int]
) -> None:
    arm = _winning_arm(config=TRUNCATE)
    fit = fit_compaction(assignment, off, [ArmMatrices(matrix=arm, config=TRUNCATE, control=True)])
    assert fit.compressed_clusters() == 0
    assert fit.control_flags  # it would have won, and that is loud, not silent


def test_unservable_arm_is_evaluated_but_ineligible(
    off: OutcomeMatrix, assignment: dict[str, int]
) -> None:
    arm = _winning_arm(config=UNKNOWN)
    fit = fit_compaction(assignment, off, [ArmMatrices(matrix=arm, config=UNKNOWN)])
    assert fit.compressed_clusters() == 0
    assert any(row.would_win and not row.eligible for row in fit.evidence)


def test_uneven_coverage_is_refused_unless_opted_in(
    off: OutcomeMatrix, assignment: dict[str, int]
) -> None:
    arm = _winning_arm(config=TRUNCATE)
    arm = OutcomeMatrix(pool=arm.pool, outcomes=arm.outcomes[: len(arm.outcomes) // 2])
    with pytest.raises(ValueError, match="cohort"):
        fit_compaction(assignment, off, [ArmMatrices(matrix=arm, config=TRUNCATE)])
    fit = fit_compaction(
        assignment, off, [ArmMatrices(matrix=arm, config=TRUNCATE)], allow_uneven=True
    )
    assert fit.coverage  # the mismatch is printed, not swallowed


def test_off_arm_must_be_uncompressed(assignment: dict[str, int]) -> None:
    compressed = _matrix(_off_rewards, lambda group, index: 0.01, config=TRUNCATE)
    with pytest.raises(ValueError, match="off arm"):
        fit_compaction(assignment, compressed, [])


def test_aa_bar_passes_on_pure_noise(assignment: dict[str, int]) -> None:
    # Episode 1 rewards differ from episode 0 by symmetric noise; the A/A gates must not
    # deviate anywhere (the SE floor keeps even zero-variance clusters honest).
    import random

    rng = random.Random(7)
    rows = []
    for sid, task, _group, _index in _scenarios():
        for model in MODELS:
            for episode in range(2):
                rows.append(_row(sid, task, model, episode, rng.choice([0.0, 1.0]), cost=0.01))
    off = OutcomeMatrix(pool=_pool(), outcomes=rows)
    assert aa_report(assignment, off) == []


def test_apply_compaction_stamps_and_enforces_exclusivity(
    off: OutcomeMatrix, assignment: dict[str, int]
) -> None:
    clusters, _ = overlay_clusters(
        off, embed_with=HashingEmbedder(dim=256), n_clusters=2, default_model="worker-a"
    )
    policy = RoutingPolicy(
        kind="rank",
        default_model="worker-a",
        pool=_pool(),
        embedder=EmbedderSpec(dim=256),
        clusters=clusters,
    )
    arm = _winning_arm(config=TRUNCATE)
    fit = fit_compaction(assignment, off, [ArmMatrices(matrix=arm, config=TRUNCATE)])
    stamped = apply_compaction(policy, fit)
    stamped_configs = {c.cluster_id: c.compression for c in stamped.clusters}
    assert stamped_configs == fit.per_cluster

    endpoint_level = policy.model_copy(
        update={"compression": TRUNCATE, "fit_compression": TRUNCATE}
    )
    with pytest.raises(ValueError, match="never mix"):
        apply_compaction(endpoint_level, fit)


def _knn_policy_on_disk(tmp_path: Path, clusters: list) -> tuple[Path, RoutingPolicy]:
    policy = RoutingPolicy(
        kind="rank",  # any savable kind works for sidecar identity tests
        default_model="worker-a",
        pool=_pool(),
        embedder=EmbedderSpec(dim=256),
        clusters=clusters,
    )
    path = tmp_path / "policy.json"
    path.write_text(policy.model_dump_json())
    return path, policy


def test_knn_map_ships_as_an_identity_bound_sidecar(
    off: OutcomeMatrix, assignment: dict[str, int], tmp_path: Path
) -> None:
    # The merged RoutingPolicy validator forbids clusters on a knn policy, so the overlay
    # goes beside the policy the way the knn bank does, BOUND to it: policy digest and
    # embedder spec ride in the artifact and load_compaction_sidecar refuses mismatches
    # (mounting foreign-geometry centroids is the measured floor-failure class).
    from wmo.optimize.compaction_fit import (
        compaction_path_for,
        load_compaction_sidecar,
        save_compaction_sidecar,
    )

    clusters, _ = overlay_clusters(
        off, embed_with=HashingEmbedder(dim=256), n_clusters=2, default_model="worker-a"
    )
    policy_path, policy = _knn_policy_on_disk(tmp_path, clusters)
    arm = _winning_arm(config=TRUNCATE)
    fit = fit_compaction(assignment, off, [ArmMatrices(matrix=arm, config=TRUNCATE)])
    save_compaction_sidecar(policy_path, policy, clusters, fit, fitted_from="test cohort")

    loaded = load_compaction_sidecar(policy_path, policy)
    assert {c.cluster_id: c.compression for c in loaded.clusters} == fit.per_cluster
    assert loaded.z == fit.z
    assert compaction_path_for(policy_path).name == "policy.json.compaction.json"

    # Digest mismatch: the policy file changed after the map was fitted beside it.
    policy_path.write_text(
        policy.model_copy(update={"default_model": "worker-b"}).model_dump_json()
    )
    with pytest.raises(ValueError, match="different policy file"):
        load_compaction_sidecar(policy_path, policy)

    # Embedder mismatch: same bytes, different geometry claimed by the caller.
    policy_path.write_text(policy.model_dump_json())
    foreign = policy.model_copy(update={"embedder": EmbedderSpec(dim=512)})
    with pytest.raises(ValueError, match="different embedding geometry"):
        load_compaction_sidecar(policy_path, foreign)


def test_sidecar_refuses_an_endpoint_level_policy(
    off: OutcomeMatrix, assignment: dict[str, int], tmp_path: Path
) -> None:
    from wmo.optimize.compaction_fit import save_compaction_sidecar

    clusters, _ = overlay_clusters(
        off, embed_with=HashingEmbedder(dim=256), n_clusters=2, default_model="worker-a"
    )
    policy_path, policy = _knn_policy_on_disk(tmp_path, clusters)
    mixed = policy.model_copy(update={"compression": TRUNCATE, "fit_compression": TRUNCATE})
    arm = _winning_arm(config=TRUNCATE)
    fit = fit_compaction(assignment, off, [ArmMatrices(matrix=arm, config=TRUNCATE)])
    with pytest.raises(ValueError, match="never mix"):
        save_compaction_sidecar(policy_path, mixed, clusters, fit)


def test_module_fitter_refuses_an_unmeasured_arm_config(
    off: OutcomeMatrix, assignment: dict[str, int]
) -> None:
    # Named future consumers bypass the CLI (which reads configs FROM matrices); the module
    # itself must refuse a config the matrix's episodes never ran under.
    arm = _winning_arm(config=TRUNCATE)
    lied = CompressionConfig(compressor_id="truncate", compressor_version="1", aggressiveness=0.9)
    with pytest.raises(ValueError, match="never .* measured|actually measured"):
        fit_compaction(assignment, off, [ArmMatrices(matrix=arm, config=lied)])


def test_control_dominance_refuses_a_winner_regardless_of_cost_rank(
    off: OutcomeMatrix, assignment: dict[str, int]
) -> None:
    # HIGH-1: the real arm is CHEAPER (so it leads the tie-break) but a control passes its
    # own quality gate with a LARGER delta. The cluster must be refused and flagged; the
    # old gate stayed silent here because the control was not ranked[0].
    real = _winning_arm(cost=0.001, config=TRUNCATE)
    control_cfg = CompressionConfig(
        compressor_id="identity", compressor_version="1", aggressiveness=0.0
    )
    control = _matrix(
        lambda group, index: 1.0,  # beats the real arm's fin delta (larger mean_diff)
        lambda group, index: 0.009,  # cheaper than off, but pricier than the real arm
        config=control_cfg,
    )
    fit = fit_compaction(
        assignment,
        off,
        [
            ArmMatrices(matrix=real, config=TRUNCATE),
            ArmMatrices(matrix=control, config=control_cfg, control=True),
        ],
    )
    assert fit.compressed_clusters() == 0
    assert any("quality-dominates" in flag for flag in fit.control_flags)
    assert any(row.control_dominated for row in fit.evidence)


def test_assign_to_clusters_matches_overlay_assignment(off: OutcomeMatrix) -> None:
    embedder = HashingEmbedder(dim=256)
    clusters, assignment = overlay_clusters(
        off, embed_with=embedder, n_clusters=2, default_model="worker-a"
    )
    reassigned = assign_to_clusters(clusters, off, embed_with=embedder)
    assert reassigned == assignment


def test_calibrator_selection_picks_the_most_aggressive_passing_arm(
    off: OutcomeMatrix, assignment: dict[str, int]
) -> None:
    # calibrator-v1 (the learned-calibration-ratio directive): among arms passing BOTH
    # gates, take the highest aggressiveness, so the learned operating point inherits the
    # corrected gate's guarantee. The rule is recorded on the fit (identity-grained).
    gentle = _winning_arm(cost=0.004, config=TRUNCATE)  # aggressiveness 0.2, cheaper
    bold_cfg = CompressionConfig(
        compressor_id="truncate", compressor_version="1", aggressiveness=0.6
    )
    bold = _winning_arm(cost=0.005, config=bold_cfg)  # more aggressive, still passing
    arms = [
        ArmMatrices(matrix=gentle, config=TRUNCATE),
        ArmMatrices(matrix=bold, config=bold_cfg),
    ]
    conservative = fit_compaction(assignment, off, arms)
    calibrated = fit_compaction(assignment, off, arms, selection="aggressiveness")
    fin = assignment["fin-0"]
    assert conservative.per_cluster[fin].aggressiveness == 0.2  # cheapest wins by default
    assert calibrated.per_cluster[fin].aggressiveness == 0.6  # calibrator takes the ceiling
    assert calibrated.selection == "aggressiveness" and conservative.selection == "cost"

    import pytest as _pytest

    with _pytest.raises(ValueError, match="selection rule"):
        fit_compaction(assignment, off, arms, selection="vibes")
