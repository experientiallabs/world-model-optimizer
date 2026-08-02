"""Tests for the promoted external BigCodeBench router fitter."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from scipy import sparse


def _module() -> ModuleType:
    path = Path(__file__).with_name("coding_model_router_bigcodebench_fit.py")
    spec = importlib.util.spec_from_file_location("bigcodebench_fit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _module()


def _write_object(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_loader_refuses_scores_when_oracle_did_not_pass(tmp_path: Path) -> None:
    _write_object(
        tmp_path / "oracle-report.json",
        {
            "protocol": {"target_outcomes_used": False},
            "passed": False,
        },
    )
    with pytest.raises(ValueError, match="router fitting is forbidden"):
        module.load_fit_data(tmp_path)


def test_grouped_folds_have_zero_family_overlap() -> None:
    groups = [f"family-{index // 2}" for index in range(20)]
    folds = module.grouped_folds(groups)
    assert len(folds) == 5
    assert sorted(np.concatenate([test for _, test in folds]).tolist()) == list(range(20))
    for train, test in folds:
        assert {groups[index] for index in train}.isdisjoint({groups[index] for index in test})


def _locked_candidate() -> object:
    config, digest = module.canonical_candidate_config(
        {"alpha": 1.0, "dim": 512, "estimator": "ridge"}
    )
    return module.LockedCandidate(
        family="ordinal",
        name="ordinal-ridge",
        config_json=config,
        config_sha256=digest,
        fit_reward=0.95,
        fit_cost_usd=0.01,
        matched_blind_reward=0.90,
        latency_p95_ms=1.0,
        artifact_bytes=100,
    )


def _deployment_consensus() -> object:
    config, digest = module.canonical_candidate_config(
        {"alpha": 1.0, "dim": 512, "estimator": "ridge"}
    )
    return module.DeploymentConsensus(
        family="ordinal",
        name="ordinal-ridge",
        order=0,
        config_json=config,
        config_sha256=digest,
        mean_fit_reward=0.95,
        mean_fit_cost_usd=0.01,
        mean_matched_blind_reward=0.90,
        mean_baseline_reward=0.96,
        minimum_seed_retention=0.95 / 0.96,
        fit_quality_feasible=True,
    )


def _selection_lock(tmp_path: Path) -> tuple[Path, object]:
    root = tmp_path / "root"
    root.mkdir()
    (root / "tasks.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "scores.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "outcomes.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "oracle-report.json").write_text(
        json.dumps({"passed": True, "protocol": {"target_outcomes_used": False}}),
        encoding="utf-8",
    )
    digests = {
        name: module._sha256(root / filename)
        for name, filename in {
            "tasks_sha256": "tasks.jsonl",
            "scores_sha256": "scores.jsonl",
            "outcomes_sha256": "outcomes.jsonl",
            "oracle_report_sha256": "oracle-report.json",
        }.items()
    }
    lock = module.SelectionLock(
        protocol="bigcodebench-fit-only-selection-v1",
        **digests,
        code_commit="a" * 40,
        seeds=[
            module.SeedSelection(
                seed=seed,
                fit_tasks=240,
                heldout_tasks=60,
                fit_ids_sha256=str(seed) * 64,
                heldout_ids_sha256=str(seed + 1) * 64,
                baseline_arm="luna-max",
                baseline_fit_reward=0.96,
                baseline_fit_cost_usd=0.02,
                selected=_locked_candidate(),
            )
            for seed in module.OUTER_SEEDS
        ],
        deployment_consensus=_deployment_consensus(),
    )
    return root, lock


def test_selection_lock_round_trip_and_matrix_fingerprint(tmp_path: Path) -> None:
    root, lock = _selection_lock(tmp_path)
    path = tmp_path / "selection-lock.json"
    module.write_selection_lock(path, lock)
    loaded = module.require_selection_lock(root, path)
    assert loaded == lock
    with pytest.raises(FileExistsError):
        module.write_selection_lock(path, lock)

    (root / "scores.jsonl").write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="scores_sha256"):
        module.require_selection_lock(root, path)


def test_seeded_outer_splits_are_distinct_and_grouped() -> None:
    groups = [f"family-{index // 2}" for index in range(40)]
    splits = module.outer_splits(groups)
    assert [split.seed for split in splits] == list(module.OUTER_SEEDS)
    assert len({tuple(split.test_indices.tolist()) for split in splits}) == 5
    for split in splits:
        assert sorted([*split.train_indices, *split.test_indices]) == list(range(40))
        assert 6 <= len(split.test_indices) <= 10
        assert {groups[index] for index in split.train_indices}.isdisjoint(
            {groups[index] for index in split.test_indices}
        )


def _knn_data() -> object:
    texts = [
        *(f"sql query select join table {index}" for index in range(10)),
        *(f"async python coroutine await task {index}" for index in range(10)),
    ]
    rewards = np.zeros((20, len(module.ARMS), module.ATTEMPTS), dtype=np.float64)
    rewards[:10, 0, :] = 1.0
    rewards[10:, 4, :] = 1.0
    costs = np.broadcast_to(
        np.asarray([0.001, 0.002, 0.003, 0.004, 0.005])[None, :, None],
        rewards.shape,
    ).copy()
    return module.FitData(
        task_ids=[f"task-{index}" for index in range(20)],
        groups=[f"family-{index // 2}" for index in range(20)],
        texts=texts,
        is_hard=np.zeros(20, dtype=np.bool_),
        rewards=rewards,
        costs=costs,
    )


def test_outcome_matrix_preserves_every_attempt() -> None:
    data = _knn_data()
    matrix = module.outcome_matrix(data)
    assert [entry.name for entry in matrix.pool] == list(module.ARMS)
    assert len(matrix.outcomes) == 20 * len(module.ARMS) * module.ATTEMPTS
    assert {outcome.model for outcome in matrix.outcomes} == set(module.ARMS)


def test_feature_scaling_uses_only_declared_fit_rows() -> None:
    data = _knn_data()
    baseline = module.feature_matrix(data, dim=512, scale_indices=np.arange(10))
    changed = _knn_data()
    changed.texts[19] = changed.texts[19] * 1_000
    shifted = module.feature_matrix(changed, dim=512, scale_indices=np.arange(10))
    assert np.array_equal(baseline[:10].toarray(), shifted[:10].toarray())


def test_native_knn_replay_matches_tensor_value(tmp_path: Path) -> None:
    data = _knn_data()
    replay = module.fit_native_knn_replay(
        data,
        np.asarray([*range(8), *range(10, 18)]),
        np.asarray([8, 9, 18, 19]),
        bank_path=tmp_path / "native-knn.bank.npz",
        dim=512,
        guard_arm="luna-max",
        rag_num=8,
        rag_thres=0.9,
        z=0.0,
        min_pairs=3,
        se_floor=False,
        floor_q=0.0,
        pick_lam=0.0,
        guard_mode="symmetric",
    )
    assert replay.bank_path.exists()
    assert replay.policy.kind == "knn"
    assert replay.policy.guard_model == "luna-max"
    assert replay.choices.shape == (4,)
    assert replay.value.reward == 1.0


def test_native_knn_refuses_overlapping_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="overlap"):
        module.fit_native_knn_replay(
            _knn_data(),
            np.asarray([0, 1]),
            np.asarray([1, 2]),
            bank_path=tmp_path / "unused.npz",
            dim=512,
            guard_arm="luna-max",
            rag_num=4,
            rag_thres=0.9,
            z=0.0,
            min_pairs=3,
            se_floor=False,
            floor_q=0.0,
            pick_lam=0.0,
            guard_mode="symmetric",
        )


def test_fit_selected_static_uses_fit_only_quality_then_cost() -> None:
    data = _knn_data()
    data.rewards[:, 3:, :] = 1.0
    selected = module.fit_selected_static(data, np.arange(20))
    assert selected.name == "luna-xhigh"
    assert selected.reward == 1.0
    assert selected.cost_usd == pytest.approx(0.004)


def test_fit_candidate_selects_cheapest_quality_feasible_point() -> None:
    candidates = [
        module.CandidateMetric("best", 1.0, 1.0, 0.2, 100, 0),
        module.CandidateMetric("cheap", 0.95, 0.4, 0.3, 80, 1),
        module.CandidateMetric("too-low", 0.94, 0.1, 0.1, 20, 2),
    ]
    assert module.select_fit_candidate(candidates, baseline_reward=1.0).name == "cheap"


def test_fit_candidate_fallback_maximizes_quality() -> None:
    candidates = [
        module.CandidateMetric("less", 0.7, 0.1, 0.1, 10, 0),
        module.CandidateMetric("more", 0.8, 0.2, 0.2, 20, 1),
    ]
    assert module.select_fit_candidate(candidates, baseline_reward=1.0).name == "more"


def test_ordinal_predictions_are_bounded_and_monotone() -> None:
    train_features = sparse.csr_matrix(np.asarray([[0.0], [0.2], [0.8], [1.0]], dtype=np.float64))
    test_features = sparse.csr_matrix(np.asarray([[0.1], [0.9]], dtype=np.float64))
    rewards = np.asarray(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.3, 0.4, 0.5, 0.6],
            [0.4, 0.5, 0.6, 0.7, 0.8],
            [0.5, 0.6, 0.7, 0.8, 0.9],
        ],
        dtype=np.float64,
    )
    predicted = module.ordinal_ridge_predictions(
        train_features,
        test_features,
        rewards,
        alpha=1.0,
    )
    assert predicted.shape == (2, len(module.ARMS))
    assert np.all((0.0 <= predicted) & (predicted <= 1.0))
    assert np.all(np.diff(predicted, axis=1) >= 0.0)


def test_ordinal_extra_trees_are_deterministic_and_monotone() -> None:
    train_features = sparse.csr_matrix(np.arange(120, dtype=np.float64).reshape(30, 4))
    test_features = train_features[:3]
    rewards = np.column_stack(
        [np.linspace(0.1 + 0.1 * arm, 0.5 + 0.1 * arm, 30) for arm in range(len(module.ARMS))]
    )
    first = module.ordinal_extra_trees_predictions(
        train_features,
        test_features,
        rewards,
        n_estimators=200,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=11,
    )
    second = module.ordinal_extra_trees_predictions(
        train_features,
        test_features,
        rewards,
        n_estimators=200,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=11,
    )
    assert np.array_equal(first, second)
    assert np.all((0.0 <= first) & (first <= 1.0))
    assert np.all(np.diff(first, axis=1) >= 0.0)


def test_matched_task_blind_control_preserves_arm_mix() -> None:
    rewards = np.zeros((4, len(module.ARMS)), dtype=np.float64)
    rewards[:2, 0] = 1.0
    rewards[2:, 1] = 1.0
    costs = np.tile(np.arange(1.0, 6.0), (4, 1))
    choices = np.asarray([0, 0, 1, 1], dtype=np.int64)
    value = module.evaluate_choices(rewards, costs, choices)
    assert value.reward == 1.0
    assert value.matched_blind_reward == 0.5
    assert value.cost_usd == 1.5
    assert value.matched_blind_cost_usd == 1.5
    assert value.arm_counts == {
        "luna-low": 2,
        "luna-medium": 2,
        "luna-high": 0,
        "luna-xhigh": 0,
        "luna-max": 0,
    }


def test_doubly_robust_dense_targets_equal_observed_arm_means() -> None:
    rewards = np.zeros((3, len(module.ARMS), module.ATTEMPTS), dtype=np.float64)
    rewards[0, 0, :] = 1.0
    rewards[1, 2, :3] = 1.0
    rewards[2, 4, 0] = 1.0
    direct = np.full((3, len(module.ARMS)), 0.37, dtype=np.float64)
    pseudo = module.doubly_robust_pseudo_values(rewards, direct)
    assert np.allclose(pseudo, rewards.mean(axis=2))


def test_multi_action_learners_are_bounded() -> None:
    train_features = sparse.csr_matrix(np.arange(160, dtype=np.float64).reshape(40, 4))
    test_features = train_features[:4]
    pseudo = np.column_stack(
        [np.linspace(0.1 * arm, 0.8 + 0.02 * arm, 40) for arm in range(len(module.ARMS))]
    )
    ridge = module.multi_action_ridge_predictions(
        train_features,
        test_features,
        pseudo,
        alpha=1.0,
    )
    hist = module.multi_action_hist_predictions(
        train_features,
        test_features,
        pseudo,
        max_leaf_nodes=7,
        learning_rate=0.03,
        min_samples_leaf=10,
        random_state=7,
    )
    assert ridge.shape == hist.shape == (4, len(module.ARMS))
    assert np.all((0.0 <= ridge) & (ridge <= 1.0))
    assert np.all((0.0 <= hist) & (hist <= 1.0))


def test_shadow_price_can_move_choice_to_cheaper_effort() -> None:
    predicted = np.asarray([[0.80, 0.81, 0.82, 0.83, 0.84]])
    costs = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    assert module.shadow_price_choices(predicted, costs, lam=0.0).tolist() == [4]
    assert module.shadow_price_choices(predicted, costs, lam=0.04).tolist() == [0]


def test_empirical_bayes_uses_loo_and_unseen_global_fallback() -> None:
    rewards = np.zeros((3, len(module.ARMS), module.ATTEMPTS), dtype=np.float64)
    rewards[0, 0, :] = 1.0
    rewards[1, 0, :] = 1.0
    rewards[2, 1, :] = 1.0
    train_base, test_base = module.empirical_bayes_family_predictions(
        ["shared", "shared", "solo"],
        ["shared", "unseen"],
        rewards,
        prior_strength=5.0,
    )
    global_mean = (rewards.sum(axis=(0, 2)) + 0.5) / (rewards.shape[0] * module.ATTEMPTS + 1.0)
    expected_shared = (np.asarray([10.0, 0.0, 0.0, 0.0, 0.0]) + 5.0 * global_mean) / 15.0
    assert np.allclose(train_base[0], (rewards[1].sum(axis=1) + 5.0 * global_mean) / 10.0)
    assert np.allclose(train_base[2], global_mean)
    assert np.allclose(test_base[0], expected_shared)
    assert np.allclose(test_base[1], global_mean)


def test_empirical_bayes_moments_are_finite_for_extreme_arms() -> None:
    rewards = np.zeros((2, len(module.ARMS), module.ATTEMPTS), dtype=np.float64)
    rewards[:, -1, :] = 1.0
    _, train_se, _, test_se = module.empirical_bayes_family_moments(
        ["a", "b"],
        ["unseen"],
        rewards,
        prior_strength=2.0,
    )
    assert np.isfinite(train_se).all()
    assert np.isfinite(test_se).all()
    assert np.all(train_se > 0.0)
    assert np.all(test_se > 0.0)


def test_empirical_bayes_residual_predictions_are_monotone() -> None:
    train_features = sparse.csr_matrix(np.eye(4, dtype=np.float64))
    test_features = sparse.csr_matrix(np.eye(4, dtype=np.float64)[:2])
    rewards = np.zeros((4, len(module.ARMS), module.ATTEMPTS), dtype=np.float64)
    rewards[0, 0:2, :] = 1.0
    rewards[1, 0:3, :] = 1.0
    rewards[2, 0:4, :] = 1.0
    rewards[3, :, :] = 1.0
    predicted = module.empirical_bayes_ridge_predictions(
        train_features,
        test_features,
        ["a", "a", "b", "b"],
        ["a", "new"],
        rewards,
        prior_strength=5.0,
        alpha=1.0,
    )
    assert predicted.shape == (2, len(module.ARMS))
    assert np.all((0.0 <= predicted) & (predicted <= 1.0))
    assert np.all(np.diff(predicted, axis=1) >= 0.0)


def test_lower_bound_choice_uses_cheapest_feasible_or_fallback() -> None:
    predicted = np.asarray(
        [
            [0.96, 0.97, 0.98, 0.99, 1.0],
            [0.70, 0.75, 0.80, 0.85, 0.90],
        ]
    )
    standard_errors = np.full_like(predicted, 0.01)
    costs = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    choices = module.lower_bound_choices(
        predicted,
        standard_errors,
        costs,
        quality_floor=0.95,
        fallback_arm=4,
        z=0.5,
    )
    assert choices.tolist() == [0, 4]


def test_negative_controls_are_deterministic_and_preserve_profiles() -> None:
    rewards = np.arange(4 * len(module.ARMS) * module.ATTEMPTS, dtype=np.float64).reshape(
        4,
        len(module.ARMS),
        module.ATTEMPTS,
    )
    first = module.random_choices(20, seed=9)
    second = module.random_choices(20, seed=9)
    shuffled = module.shuffled_task_rewards(rewards, seed=4)
    assert np.array_equal(first, second)
    assert sorted(shuffled[:, 0, 0].tolist()) == sorted(rewards[:, 0, 0].tolist())


def test_cost_only_choice_uses_fit_costs() -> None:
    costs = np.asarray([[3.0, 2.0, 1.0, 4.0, 5.0], [3.0, 2.0, 1.0, 4.0, 5.0]])
    assert module.cost_only_choices(costs).tolist() == [2, 2]


def test_artifact_size_and_single_route_latency(tmp_path: Path) -> None:
    first = tmp_path / "policy.json"
    second = tmp_path / "bank.npz"
    first.write_bytes(b"abc")
    second.write_bytes(b"12345")
    assert module.artifact_size([first, second]) == 8

    latency = module.measure_route_latency(
        lambda text: len(text) % len(module.ARMS), ["abc"], decisions=20
    )
    assert latency.decisions == 20
    assert latency.p50_ms >= 0.0
    assert latency.p95_ms >= latency.p50_ms
    assert latency.passed
