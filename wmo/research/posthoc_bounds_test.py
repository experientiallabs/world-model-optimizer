"""Tests for the post-hoc routing bounds (best-of-n ceiling, free selectors, corr decomposition).

The synthetic fixture is hand-computable, so every assertion below is an exact value rather than a
tolerance. The final test pins the headline numbers from the 2026-07-24 audit against the shared
matrices and skips when that data dir is absent (it lives outside the repo).
"""

from __future__ import annotations

import pytest

from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.providers.base import ProviderKind, TokenUsage
from wmo.providers.pool import PoolEntry
from wmo.research.posthoc_bounds import (
    DEFAULT_SELECTOR,
    SELECTOR_KEYS,
    all_finished,
    best_of_n_by_model,
    corpus_bounds,
    feature_correlations,
    one_finished,
    pooled_correct_z,
    selector_bound,
)
from wmo.research.routing_corpus import routing_data_root

MATRICES = routing_data_root() / "matrices"


def _pool() -> list[PoolEntry]:
    return [
        PoolEntry(
            name="cheap",
            kind=ProviderKind.OPENAI,
            model="c",
            input_per_mtok=1.0,
            output_per_mtok=1.0,
        ),
        PoolEntry(
            name="pricey",
            kind=ProviderKind.OPENAI,
            model="p",
            input_per_mtok=1.0,
            output_per_mtok=1.0,
        ),
    ]


def _episode(
    sid: str,
    model: str,
    episode: int,
    reward: float,
    *,
    steps: int,
    cost: float,
    stop_reason: str = "agent_done",
) -> ScenarioOutcome:
    return ScenarioOutcome(
        scenario_id=sid,
        task=f"task-{sid}",
        model=model,
        episode=episode,
        reward=reward,
        success=reward > 0.5,
        steps=steps,
        stop_reason=stop_reason,
        usage=TokenUsage(input_tokens=100, output_tokens=10 * steps),
        cost_usd=cost,
        replies=["r"] * steps,
    )


def _matrix() -> OutcomeMatrix:
    """Two scenarios x two models x two episodes, with the better episode always taking more steps.

    cheap:  s0 rewards (0.0, 1.0) -> cell mean 0.50   s1 rewards (0.0, 0.0) -> 0.00
    pricey: s0 rewards (1.0, 1.0) -> cell mean 1.00   s1 rewards (0.0, 1.0) -> 0.50
    So best-single = pricey at 0.75, and the cross-model oracle is (1.00 + 0.50) / 2 = 0.75.
    """
    outcomes = [
        _episode("s0", "cheap", 0, 0.0, steps=1, cost=0.001),
        _episode("s0", "cheap", 1, 1.0, steps=5, cost=0.001),
        _episode("s1", "cheap", 0, 0.0, steps=1, cost=0.001),
        _episode("s1", "cheap", 1, 0.0, steps=5, cost=0.001),
        _episode("s0", "pricey", 0, 1.0, steps=1, cost=0.02),
        _episode("s0", "pricey", 1, 1.0, steps=5, cost=0.02),
        _episode("s1", "pricey", 0, 0.0, steps=1, cost=0.02),
        _episode("s1", "pricey", 1, 1.0, steps=5, cost=0.02),
    ]
    return OutcomeMatrix(pool=_pool(), outcomes=outcomes)


def test_corpus_bounds_anchors_and_ceiling() -> None:
    bounds = corpus_bounds(_matrix(), "synthetic")
    assert bounds.scenarios == 2
    assert bounds.models == 2
    assert bounds.episodes_per_cell == [2]
    assert bounds.best_single == "pricey"
    assert bounds.best_single_accuracy == pytest.approx(0.75)
    assert bounds.best_single_cost_per_call == pytest.approx(0.02)
    assert bounds.oracle_accuracy == pytest.approx(0.75)
    # Two of the four cells' episodes disagree (cheap/s0 and pricey/s1), each by a full point.
    assert bounds.episode_disagreement_mean == pytest.approx(0.5)
    assert bounds.episode_disagreement_fraction == pytest.approx(0.5)

    by_model = {bound.model: bound for bound in bounds.best_of_n}
    cheap = by_model["cheap"]
    assert cheap.episodes == 2
    assert cheap.cells == 2
    assert cheap.one_shot_accuracy == pytest.approx(0.25)
    assert cheap.oracle_of_n_accuracy == pytest.approx(0.5)
    # Best-of-n costs the SUM of its episodes, not their mean.
    assert cheap.oracle_of_n_cost_per_call == pytest.approx(0.002)
    pricey = by_model["pricey"]
    assert pricey.oracle_of_n_accuracy == pytest.approx(1.0)
    assert pricey.beats_best_single_accuracy is True
    assert pricey.beats_best_single_cost is False


def test_best_of_n_by_model_restricts_to_the_given_scenarios() -> None:
    """The fit-split discovery path: only `ids` may inform the choice, never held-out scenarios."""
    matrix = _matrix()
    fit_only = best_of_n_by_model(matrix, ["s0"], baseline_accuracy=0.0, baseline_cost=1.0)
    by_model = {bound.model: bound for bound in fit_only}
    assert by_model["cheap"].cells == 1
    # s0 alone: cheap's episodes are (0.0, 1.0), so oracle-of-2 is 1.0 where the full matrix gave
    # 0.5. A discovery step that leaked s1 in would not see this number.
    assert by_model["cheap"].oracle_of_n_accuracy == pytest.approx(1.0)
    assert by_model["pricey"].oracle_of_n_accuracy == pytest.approx(1.0)
    # Baselines are passed in, so the flags reflect the caller's reference, not the subset's own.
    assert all(bound.beats_best_single_accuracy for bound in fit_only)


def test_best_of_n_by_model_is_empty_without_repeated_episodes() -> None:
    outcomes = [
        _episode("s0", "cheap", 0, 1.0, steps=1, cost=0.001),
        _episode("s1", "cheap", 0, 0.0, steps=1, cost=0.001),
    ]
    matrix = OutcomeMatrix(pool=_pool(), outcomes=outcomes)
    assert best_of_n_by_model(matrix, baseline_accuracy=0.5, baseline_cost=0.001) == []
    # An explicit depth of 1 is still best-of-nothing, so it is rejected rather than silently
    # reporting 1-shot numbers as a best-of-n row.
    assert best_of_n_by_model(_matrix(), baseline_accuracy=0.5, baseline_cost=0.01, depth=1) == []


def test_default_selector_reaches_the_ceiling_when_effort_tracks_reward() -> None:
    """The fixture gives the better episode more steps, so the free selector should be perfect."""
    bounds = corpus_bounds(_matrix(), "synthetic")
    for bound in bounds.best_of_n:
        assert bound.selected_of_n_accuracy == pytest.approx(bound.oracle_of_n_accuracy)


def test_selector_bound_counts_only_decisive_cells_and_averages_ties() -> None:
    # One cell where steps rank the episodes and rewards differ (decisive, and the pick is right),
    # one where steps tie so the feature cannot distinguish them (not decisive, contributes 0.5).
    outcomes = [
        _episode("s0", "cheap", 0, 0.0, steps=1, cost=0.001),
        _episode("s0", "cheap", 1, 1.0, steps=5, cost=0.001),
        _episode("s1", "cheap", 0, 0.0, steps=3, cost=0.001),
        _episode("s1", "cheap", 1, 1.0, steps=3, cost=0.001),
    ]
    matrix = OutcomeMatrix(pool=_pool(), outcomes=outcomes)
    bound = selector_bound(matrix, "synthetic", DEFAULT_SELECTOR, SELECTOR_KEYS[DEFAULT_SELECTOR])
    assert bound.cells == 2
    assert bound.decisive_cells == 1
    assert bound.correct_fraction == pytest.approx(1.0)
    assert bound.random_of_n == pytest.approx(0.5)
    assert bound.oracle_of_n == pytest.approx(1.0)
    # Decisive cell yields 1.0, tied cell yields its mean 0.5 -> 0.75, i.e. half the headroom.
    assert bound.selector_accuracy == pytest.approx(0.75)
    assert bound.harvested_fraction == pytest.approx(0.5)


def test_pooled_sign_selector_is_worse_than_chance_on_the_same_data() -> None:
    """The negative control: ordering by the pooled sign inverts a selector that works."""
    matrix = _matrix()
    good = selector_bound(matrix, "s", DEFAULT_SELECTOR, SELECTOR_KEYS[DEFAULT_SELECTOR])
    key = "fewer-steps (pooled-sign control)"
    bad = selector_bound(matrix, "s", key, SELECTOR_KEYS[key])
    assert good.harvested_fraction > 0.0
    assert bad.harvested_fraction < 0.0
    assert bad.correct_fraction < good.correct_fraction


def test_cell_filters_isolate_the_step_cap() -> None:
    outcomes = [
        # Both finished: the cap plays no part in this cell.
        _episode("s0", "cheap", 0, 0.0, steps=1, cost=0.001),
        _episode("s0", "cheap", 1, 1.0, steps=5, cost=0.001),
        # Exactly one finished: the pure cap contrast.
        _episode("s1", "cheap", 0, 1.0, steps=4, cost=0.001),
        _episode("s1", "cheap", 1, 0.0, steps=8, cost=0.001, stop_reason="max_steps"),
    ]
    matrix = OutcomeMatrix(pool=_pool(), outcomes=outcomes)
    key = SELECTOR_KEYS[DEFAULT_SELECTOR]
    both = selector_bound(matrix, "s", DEFAULT_SELECTOR, key, cell_filter=all_finished)
    assert both.cells == 1
    assert both.decisive_cells == 1
    assert both.correct_fraction == pytest.approx(1.0)
    single = selector_bound(matrix, "s", DEFAULT_SELECTOR, key, cell_filter=one_finished)
    assert single.cells == 1
    # The finisher scored 1.0 despite taking FEWER steps, so only the finish term can find it.
    assert single.selector_accuracy == pytest.approx(1.0)


def test_correlation_decomposition_separates_difficulty_from_rollout_quality() -> None:
    """The phenomenon that produced the backwards selector: pooled and within-cell signs differ.

    Cell A is easy (few steps, rewards 0/1); cell B is hard (many steps, both rewards 0). Between
    cells, more steps means less reward. Within cell A, more steps means MORE reward.
    """
    outcomes = [
        _episode("easy", "cheap", 0, 0.0, steps=2, cost=0.001),
        _episode("easy", "cheap", 1, 1.0, steps=4, cost=0.001),
        _episode("hard", "cheap", 0, 0.0, steps=20, cost=0.001),
        _episode("hard", "cheap", 1, 0.0, steps=22, cost=0.001),
    ]
    matrix = OutcomeMatrix(pool=_pool(), outcomes=outcomes)
    steps = next(row for row in feature_correlations(matrix, "s") if row.feature == "steps")
    assert steps.pooled < 0.0
    assert steps.between_cell < 0.0
    assert steps.within_cell > 0.0


def test_pooled_correct_z_pools_decisive_picks() -> None:
    matrix = _matrix()
    bounds = [
        selector_bound(matrix, "a", DEFAULT_SELECTOR, SELECTOR_KEYS[DEFAULT_SELECTOR]),
        selector_bound(matrix, "b", DEFAULT_SELECTOR, SELECTOR_KEYS[DEFAULT_SELECTOR]),
    ]
    correct, decisive, z = pooled_correct_z(bounds)
    assert (correct, decisive) == (4, 4)  # two decisive cells per copy, all picked correctly
    assert z == pytest.approx(2.0)  # (1.0 - 0.5) / sqrt(0.25 / 4) = 0.5 / 0.25
    assert pooled_correct_z([]) == (0, 0, 0.0)


def test_single_episode_matrix_has_no_best_of_n() -> None:
    outcomes = [
        _episode("s0", "cheap", 0, 1.0, steps=1, cost=0.001),
        _episode("s1", "cheap", 0, 0.0, steps=1, cost=0.001),
    ]
    bounds = corpus_bounds(OutcomeMatrix(pool=_pool(), outcomes=outcomes), "synthetic")
    assert bounds.episodes_per_cell == [1]
    assert bounds.best_of_n == []
    assert bounds.episode_disagreement_mean is None
    with pytest.raises(ValueError, match="more than one scored episode"):
        selector_bound(
            OutcomeMatrix(pool=_pool(), outcomes=outcomes),
            "synthetic",
            DEFAULT_SELECTOR,
            SELECTOR_KEYS[DEFAULT_SELECTOR],
        )


@pytest.mark.skipif(not MATRICES.is_dir(), reason="shared routing matrices are not present")
def test_audit_headline_numbers_do_not_drift() -> None:
    """Pin the 2026-07-24 audit headlines on the real matrices.

    Companion to `.agents/scripts/audit_posthoc_bounds.py`, which prints the full tables.
    Tolerances are tight (+-0.002 on rewards) because the matrices are frozen data: a change here
    means either the matrices were recaptured or the estimator moved.
    """
    tau = corpus_bounds(OutcomeMatrix.load(MATRICES / "tau-bench_matrix.json"), "tau-bench")
    assert tau.best_single == "fable-5"
    assert tau.best_single_accuracy == pytest.approx(0.537, abs=0.002)
    assert tau.oracle_accuracy == pytest.approx(0.742, abs=0.002)
    kimi = next(bound for bound in tau.best_of_n if bound.model == "kimi-k2.6")
    # The ceiling: +7.1pt over best-single at 3.6x cheaper.
    assert kimi.oracle_of_n_accuracy == pytest.approx(0.608, abs=0.002)
    assert kimi.oracle_of_n_cost_per_call == pytest.approx(0.04871, abs=1e-4)
    # The achievable point with the free selector: parity on accuracy, still 3.6x cheaper.
    assert kimi.selected_of_n_accuracy == pytest.approx(0.544, abs=0.002)
    assert kimi.beats_best_single_accuracy is True
    assert kimi.beats_best_single_cost is True

    fin = corpus_bounds(OutcomeMatrix.load(MATRICES / "financebench_matrix.json"), "financebench")
    opus = next(bound for bound in fin.best_of_n if bound.model == "opus-4-8")
    # The strongest achievable win in the audit: +4.9pt over best-single at 1.4x cheaper.
    assert fin.best_single_accuracy == pytest.approx(0.478, abs=0.002)
    assert opus.selected_of_n_accuracy == pytest.approx(0.527, abs=0.002)
    assert opus.oracle_of_n_cost_per_call == pytest.approx(0.04559, abs=1e-4)

    term = corpus_bounds(OutcomeMatrix.load(MATRICES / "terminal-tasks_matrix.json"), "terminal")
    mini = next(bound for bound in term.best_of_n if bound.model == "gpt-5.4-mini")
    assert mini.oracle_of_n_accuracy == pytest.approx(0.974, abs=0.002)
    assert mini.oracle_of_n_cost_per_call == pytest.approx(0.00283, abs=1e-5)

    # (b) The sign matters: the pooled-sign selector loses, the corrected one wins, and the
    # both-finished control shows the effort term is not just max_steps truncation.
    corpora = ["tau-bench", "financebench", "continual-learning", "dabstep", "terminal-tasks"]
    loaded = [(name, OutcomeMatrix.load(MATRICES / f"{name}_matrix.json")) for name in corpora]
    backwards = "fewer-steps (pooled-sign control)"
    _, _, z_backwards = pooled_correct_z(
        selector_bound(matrix, name, backwards, SELECTOR_KEYS[backwards]) for name, matrix in loaded
    )
    assert z_backwards < 0.0
    _, decisive, z_forward = pooled_correct_z(
        selector_bound(matrix, name, DEFAULT_SELECTOR, SELECTOR_KEYS[DEFAULT_SELECTOR])
        for name, matrix in loaded
    )
    assert decisive == 327
    # 231/327 = 70.6% on these five corpora, vs 67.5% pooled over all eight (the two weakest
    # corpora, crmarena and tau-telecom, are excluded here). Null SE, per `pooled_correct_z`.
    assert z_forward == pytest.approx(7.47, abs=0.05)
    _, control_decisive, z_control = pooled_correct_z(
        selector_bound(
            matrix, name, "more-replies", SELECTOR_KEYS["more-replies"], cell_filter=all_finished
        )
        for name, matrix in loaded
    )
    assert control_decisive == 205
    # 145/205 = 70.7% with the step cap held constant, the control that matters most.
    assert z_control == pytest.approx(5.94, abs=0.05)

    # (c) The within-cell sign flip on tau-bench, which is why the naive selector was backwards.
    rows = {
        row.feature: row
        for row in feature_correlations(OutcomeMatrix.load(MATRICES / "tau-bench_matrix.json"), "t")
    }
    assert rows["steps"].pooled < 0.0
    assert rows["steps"].between_cell < 0.0
    assert rows["steps"].within_cell == pytest.approx(0.310, abs=0.005)
    assert rows["n_replies"].within_cell == pytest.approx(0.363, abs=0.005)
