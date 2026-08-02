"""Offline tests for the cross-dataset reasoning-effort transfer experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from coding_model_router_cross_dataset import (
    ExternalTask,
    TransferPolicy,
    _collect_external_tasks,
    _effort_calibration,
    _effort_for_ease,
    _external_oof_predictions,
    _family_efforts,
    _normalized,
    _select_binary_threshold,
    _spearman,
    _weighted_knn,
)


def test_weighted_knn_uses_nearest_reward_profile() -> None:
    bank = _normalized(np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]))
    rewards = np.asarray([1.0, 0.8, 0.0])
    queries = _normalized(np.asarray([[1.0, 0.0], [0.0, 1.0]]))
    prediction, nearest = _weighted_knn(bank, rewards, queries, k=2)
    assert prediction[0] > 0.85
    assert prediction[1] < 0.5
    assert np.allclose(nearest, 1.0)


def test_effort_mapping_is_monotone_and_collapses_absent_levels_upward() -> None:
    thresholds = (0.1, 0.2, 0.4, 0.7)
    available = {"low", "medium", "high", "xhigh", "max"}
    assert _effort_for_ease(0.8, thresholds, available) == "low"
    assert _effort_for_ease(0.5, thresholds, available) == "medium"
    assert _effort_for_ease(0.3, thresholds, available) == "high"
    assert _effort_for_ease(0.15, thresholds, available) == "xhigh"
    assert _effort_for_ease(0.05, thresholds, available) == "max"
    assert _effort_for_ease(0.05, thresholds, {"low", "medium", "high", "xhigh"}) == "xhigh"


def test_rank_normalized_calibration_uses_only_feature_quantiles() -> None:
    ease = np.arange(100, dtype=np.float64)
    nearest = np.arange(100, dtype=np.float64) / 100.0
    policy = TransferPolicy(
        k=6,
        floor_sim=0.5,
        ease_thresholds=(1.0, 2.0, 3.0, 4.0),
        external_embeddings=np.ones((2, 2)),
        external_strong_rewards=np.ones(2),
    )
    calibration = _effort_calibration(
        ease,
        nearest,
        policy,
        mode="rank_normalized",
    )
    assert not calibration.confirmatory
    assert np.allclose(calibration.ease_thresholds, (9.9, 24.75, 44.55, 69.3))
    assert np.isclose(calibration.novelty_floor_similarity, 0.099)


def test_binary_threshold_minimizes_strong_traffic_at_retention_floor() -> None:
    predicted = np.asarray([0.9, 0.8, 0.2, 0.1])
    cheap = np.asarray([0.0, 0.0, 1.0, 1.0])
    strong = np.ones(4)
    threshold, strong_share, retention = _select_binary_threshold(
        predicted,
        cheap,
        strong,
        retention_floor=0.95,
    )
    assert np.isclose(threshold, 0.8)
    assert np.isclose(strong_share, 0.5)
    assert np.isclose(retention, 1.0)


def test_family_efforts_keeps_only_real_effort_ladders() -> None:
    arms = [
        "family_a_low",
        "family_a_medium",
        "family_a_high",
        "family_b_low",
        "family_b_high",
        "family_c_low",
        "family_c_medium",
        "family_c_high",
        "family_c_max",
        "not_an_effort_arm",
    ]
    families = _family_efforts(arms)
    assert set(families) == {"family_a", "family_c"}
    assert families["family_c"]["max"] == 8


def test_external_oof_predictions_never_retrieve_same_repo() -> None:
    embeddings = _normalized(
        np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.7, 0.7],
                [0.7, 0.7],
                [-1.0, 0.0],
                [-1.0, 0.0],
                [0.0, -1.0],
                [0.0, -1.0],
            ]
        )
    )
    rewards = np.asarray([1.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.2, 0.2, 0.8, 0.8])
    groups = [f"repo-{index // 2}" for index in range(10)]
    predictions, nearest = _external_oof_predictions(
        embeddings,
        rewards,
        groups,
        k=2,
        seed=37,
    )
    assert predictions.shape == rewards.shape
    assert nearest.shape == rewards.shape
    assert np.isfinite(predictions).all()


def test_spearman_handles_ties_and_perfect_order() -> None:
    left = np.asarray([1.0, 2.0, 2.0, 4.0])
    assert np.isclose(_spearman(left, left), 1.0)
    assert _spearman(left, left[::-1]) < 0.0


def test_external_task_cache_round_trips(tmp_path: Path) -> None:
    cache = tmp_path / "tasks.json"
    cache.write_text(
        '[{"instance_id":"task-1","repo":"org/repo","text":"fix it",'
        '"cheap_reward":0.2,"strong_reward":0.5,"cheap_attempts":5,"strong_attempts":5}]',
        encoding="utf-8",
    )
    assert _collect_external_tasks(cache) == [
        ExternalTask(
            instance_id="task-1",
            repo="org/repo",
            text="fix it",
            cheap_reward=0.2,
            strong_reward=0.5,
            cheap_attempts=5,
            strong_attempts=5,
        )
    ]
