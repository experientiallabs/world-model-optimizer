"""Tests for the external-only DeepSWE autoresearch runner."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path
from types import ModuleType

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from wmo.optimize.policy import RoutingPolicy, select_model
from wmo.retrieval.embedders import HashingEmbedder


def _module() -> ModuleType:
    path = Path(__file__).with_name("coding_model_router_autoresearch.py")
    spec = importlib.util.spec_from_file_location("coding_model_router_autoresearch", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _r2e_bundle(text: str, repo: str) -> bytes:
    target = io.BytesIO()
    with tarfile.open(fileobj=target, mode="w:gz") as archive:
        for name, payload in (
            ("instruction.md", text.encode()),
            (
                "environment/workspace/metadata.json",
                json.dumps({"repo_name": repo}).encode(),
            ),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    return target.getvalue()


def _write_r2e_fixture(root: Path) -> None:
    tasks = [f"r2egym-{index:04d}" for index in range(4)]
    pq.write_table(
        pa.table(
            {
                "path": tasks,
                "task_binary": [
                    _r2e_bundle(f"task {index}", f"repo-{index % 2}") for index in range(4)
                ],
            }
        ),
        root / "dcagent_tasks.parquet",
    )
    pq.write_table(
        pa.table({"task": tasks, "episode": ["episode"] * 4, "date": ["date"] * 4}),
        root / "gpt5_codex_attempted.parquet",
    )
    pq.write_table(
        pa.table({"path": [tasks[0]]}),
        root / "gpt5_codex_solved.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "task": tasks,
                "result": ["1.0", "1.0", "0.0", "AgentTimeoutError"],
            }
        ),
        root / "kimi25_outcomes.parquet",
    )


def test_r2e_loader_reads_paired_gradeable_outcomes_from_artifacts(tmp_path: Path) -> None:
    module = _module()
    _write_r2e_fixture(tmp_path)
    source = module._load_r2e(tmp_path)
    assert source.task_ids == ["r2egym-0000", "r2egym-0001", "r2egym-0002"]
    assert source.texts == ["task 0", "task 1", "task 2"]
    assert source.groups == ["repo-0", "repo-1", "repo-0"]
    assert source.weak.tolist() == [1.0, 0.0, 0.0]
    assert source.strong.tolist() == [1.0, 1.0, 0.0]


def test_operating_point_uses_strong_only_where_score_is_high() -> None:
    module = _module()
    scores = np.asarray([0.9, 0.8, 0.2, 0.1])
    weak = np.asarray([0.0, 0.0, 1.0, 1.0])
    strong = np.asarray([1.0, 1.0, 1.0, 1.0])
    point = module._operating_point(
        scores,
        weak,
        strong,
        ["a", "a", "b", "b"],
        0.95,
    )
    assert point["strong_traffic"] == 0.5
    assert point["mean_retention"] == 1.0


def test_native_candidate_family_uses_servable_hashing_features() -> None:
    module = _module()
    candidates = module._candidate_space("native-linear")
    observed = [candidate for candidate in candidates if candidate.label_mode == "observed"]
    assert len(observed) == 9
    assert {candidate.analyzer for candidate in observed} == {"hashing"}
    assert {candidate.components for candidate in observed} == {512, 2048, 8192}

    transformer = module._features(observed[0])
    first = transformer.fit_transform(["same task", "different task"])
    second = transformer.transform(["same task", "different task"])
    assert np.array_equal(first, second)
    assert first.shape == (2, 512)
    assert first.tolist() == HashingEmbedder(dim=512).embed(["same task", "different task"])


def test_structural_irt_family_uses_deterministic_pre_inference_features() -> None:
    module = _module()
    candidates = module._candidate_space("structural-irt")
    observed = [candidate for candidate in candidates if candidate.label_mode == "observed"]
    assert len(observed) == 5
    assert {candidate.analyzer for candidate in observed} == {"structural"}
    assert {candidate.estimator for candidate in observed} == {
        "irt-difficulty",
        "ridge-heads",
    }

    transformer = module._features(observed[0])
    texts = [
        "Fix timeout in `client.py`.\n\n```python\nraise TimeoutError()\n```",
        "Add a color option.",
    ]
    first = transformer.fit_transform(texts)
    second = transformer.transform(texts)
    assert np.array_equal(first, second)
    assert first.shape == (2, 27)


def test_task_profile_family_uses_disjoint_static_taxonomy_classifier() -> None:
    module = _module()
    candidates = module._candidate_space("task-profile")
    observed = [candidate for candidate in candidates if candidate.label_mode == "observed"]
    assert len(observed) == 8
    assert {candidate.analyzer for candidate in observed} == {"hashing"}
    assert {candidate.estimator for candidate in observed} == {"profile-uplift"}

    teacher = module.ProfileTeacherData(
        task_ids=["e1", "e2", "h1", "h2"],
        groups=["r1", "r2", "r3", "r4"],
        texts=["small typo", "simple rename", "complex failure", "distributed deadlock"],
        difficulty=["easy", "easy", "hard", "hard"],
        intent_completeness=["complete"] * 4,
        pr_categories=[["minor_bug"], ["minor_bug"], ["core_feat"], ["core_feat"]],
    )
    spec = module.CandidateSpec(
        "profile-test",
        "hashing",
        64,
        "profile-uplift",
        profile_mode="difficulty",
        profile_temperature=0.1,
        profile_prior_strength=0.0,
        min_profile_tasks=1,
    )
    features = module._features(spec).fit_transform(teacher.texts)
    classifier = module._fit_profile_classifier(spec, teacher, features)
    model = module._fit_profile_uplift(
        spec,
        classifier,
        features,
        np.zeros(4),
        np.asarray([0.1, 0.1, 0.9, 0.9]),
        np.ones(4),
    )
    scores = model.predict(features)
    assert float(scores[2:].mean()) > float(scores[:2].mean())


def test_profile_teacher_filter_removes_outcome_identity_and_text_overlap() -> None:
    module = _module()
    teacher = module.ProfileTeacherData(
        task_ids=["same-id", "other-id", "kept"],
        groups=["r1", "r2", "r3"],
        texts=["first", "same normalized text", "unique teacher"],
        difficulty=["easy", "medium", "hard"],
        intent_completeness=["complete"] * 3,
        pr_categories=[[], [], []],
    )
    combined = module.CombinedData(
        source_names=["source", "source"],
        task_ids=["same-id", "outcome-2"],
        groups=["g1", "g2"],
        texts=["different", "same   normalized text"],
        weak=np.zeros(2),
        strong=np.ones(2),
        sample_weight=np.ones(2),
    )
    filtered = module._disjoint_profile_teacher(teacher, combined, minimum_rows=1)
    assert filtered.task_ids == ["kept"]


def test_irt_uplift_is_largest_for_middle_difficulty() -> None:
    module = _module()
    easiness = np.asarray([-4.0, 0.0, 4.0])
    weights = np.ones(3)
    weak_offset = module._weighted_irt_offset(easiness, 0.3, weights)
    strong_offset = module._weighted_irt_offset(easiness, 0.7, weights)
    uplift = module._sigmoid(easiness + strong_offset) - module._sigmoid(easiness + weak_offset)
    assert uplift[1] > uplift[0]
    assert uplift[1] > uplift[2]


def test_native_linear_heads_are_plain_numeric_artifact() -> None:
    module = _module()
    spec = module.CandidateSpec(
        "hash8-ridge-heads-a1",
        "hashing",
        8,
        "ridge-heads",
    )
    features = module._features(spec).fit_transform(["first task", "second task"])
    estimators = module._fit_estimators(
        spec,
        features,
        np.asarray([0.0, 1.0]),
        np.asarray([1.0, 1.0]),
        np.ones(2),
    )
    artifact = module._native_linear_heads(
        spec,
        estimators,
        {"0.95": {"threshold": 0.1}},
    )
    assert artifact["schema"] == "wmo-linear-heads-v1"
    assert len(artifact["weak_weights"]) == 8
    assert len(artifact["strong_weights"]) == 8
    assert artifact["target_outcomes_used"] is False


def test_export_linear_writes_valid_two_arm_policy(tmp_path: Path) -> None:
    module = _module()
    heads = tmp_path / "heads.json"
    matrix = tmp_path / "matrix.json"
    output = tmp_path / "policy.json"
    heads.write_text(
        json.dumps(
            {
                "schema": "wmo-linear-heads-v1",
                "embedder": {"kind": "hashing", "dim": 8},
                "weak_weights": [0.0] * 8,
                "strong_weights": [0.0] * 8,
                "weak_bias": 0.2,
                "strong_bias": 0.8,
                "operating_points": {"0.97": {"threshold": 0.1}},
                "target_outcomes_used": False,
                "target_embeddings_used": False,
            }
        ),
        encoding="utf-8",
    )
    matrix.write_text(
        json.dumps(
            {
                "pool": [
                    {
                        "name": name,
                        "kind": "openai",
                        "model": name,
                        "input_per_mtok": 1.0,
                        "output_per_mtok": 2.0,
                    }
                    for name in ("weak", "strong")
                ]
            }
        ),
        encoding="utf-8",
    )
    args = module._parser().parse_args(
        [
            "export-linear",
            "--heads",
            str(heads),
            "--matrix",
            str(matrix),
            "--weak-model",
            "weak",
            "--strong-model",
            "strong",
            "--quality-floor",
            "0.97",
            "--output",
            str(output),
        ]
    )
    module._export_linear(args)
    policy = RoutingPolicy.load(output)
    assert [entry.name for entry in policy.pool] == ["weak", "strong"]
    assert policy.linear_threshold == 0.1
    assert select_model(policy, "coding task").model == "strong"


def test_three_level_route_is_monotone_in_predicted_uplift() -> None:
    module = _module()
    decisions = module._route_indices(
        np.asarray([-0.1, 0.3, 0.8]),
        [0.2, 0.6],
        3,
    )
    assert decisions.tolist() == [0, 1, 2]


def test_task_blind_control_matches_router_traffic_and_analytic_mean() -> None:
    module = _module()
    target = module.TargetData(
        task_ids=["a", "b", "c", "d"],
        texts=["a", "b", "c", "d"],
        groups=["g1", "g2", "g3", "g4"],
        arms=["weak", "strong"],
        rewards=np.asarray(
            [
                [0.0, 0.2, 0.4, 0.6],
                [1.0, 0.8, 0.6, 0.4],
            ]
        ),
        costs=np.asarray(
            [
                [1.0, 1.0, 1.0, 1.0],
                [3.0, 3.0, 3.0, 3.0],
            ]
        ),
    )
    decisions = np.asarray([0, 1, 1, 0])
    columns = np.arange(4)
    routed_reward = target.rewards[decisions, columns]
    routed_cost = target.costs[decisions, columns]

    control = module._matched_task_blind_control(
        target,
        np.asarray([0, 1]),
        decisions,
        routed_reward,
        routed_cost,
        seed=7,
        samples=1_000,
    )

    assert control is not None
    assert control["traffic_matched"] == {"weak": 2, "strong": 2}
    assert np.isclose(control["expected_reward"], 0.5)
    assert np.isclose(control["expected_cost_usd"], 8.0)
    assert np.isclose(control["router_cost_delta_vs_random_mean_usd"], 0.0)


def test_two_level_threshold_uses_the_requested_frozen_quality_floor() -> None:
    module = _module()
    thresholds = {
        "0.95": {"threshold": 0.1},
        "0.97": {"threshold": 0.2},
        "0.99": {"threshold": 0.3},
    }
    assert module._thresholds(thresholds, 2, "0.95") == [0.1]
    assert module._thresholds(thresholds, 2, "0.97") == [0.2]
    assert module._thresholds(thresholds, 2, "0.99") == [0.3]


def test_combine_deduplicates_exact_normalized_text() -> None:
    module = _module()
    source = module.SourceData(
        name="first",
        task_ids=["a", "b"],
        groups=["r1", "r2"],
        texts=["same task", "unique"],
        weak=np.asarray([0.0, 0.0]),
        strong=np.asarray([1.0, 1.0]),
        weak_attempts=np.ones(2),
        strong_attempts=np.ones(2),
    )
    duplicate = module.SourceData(
        name="second",
        task_ids=["c"],
        groups=["r3"],
        texts=["same   task"],
        weak=np.asarray([0.0]),
        strong=np.asarray([1.0]),
        weak_attempts=np.ones(1),
        strong_attempts=np.ones(1),
    )
    combined = module._combine([source, duplicate])
    assert combined.task_ids == ["a", "b"]


def test_combine_keeps_distinct_variants_with_shared_id_and_balances_sources() -> None:
    module = _module()
    variants = module.SourceData(
        name="variants",
        task_ids=["shared", "shared"],
        groups=["repo", "repo"],
        texts=["original prompt", "perturbed prompt"],
        weak=np.asarray([0.0, 0.0]),
        strong=np.asarray([1.0, 1.0]),
        weak_attempts=np.ones(2),
        strong_attempts=np.ones(2),
    )
    singleton = module.SourceData(
        name="singleton",
        task_ids=["other"],
        groups=["repo-2"],
        texts=["another task"],
        weak=np.asarray([0.0]),
        strong=np.asarray([1.0]),
        weak_attempts=np.ones(1),
        strong_attempts=np.ones(1),
    )
    combined = module._combine([variants, singleton])
    assert len(combined.task_ids) == 3
    assert len(set(combined.task_ids)) == 3
    totals = {
        source: float(
            combined.sample_weight[np.asarray(combined.source_names, dtype=object) == source].sum()
        )
        for source in set(combined.source_names)
    }
    assert totals["variants"] == totals["singleton"]


def test_empirical_bayes_rate_preserves_source_mean_and_arm_order() -> None:
    module = _module()
    weak_mean = 0.075
    strong_mean = 0.112
    attempts = 2
    weak_rates = np.asarray(
        [
            module._empirical_bayes_rate(0.0, attempts, weak_mean),
            module._empirical_bayes_rate(1.0, attempts, weak_mean),
        ]
    )
    strong_rates = np.asarray(
        [
            module._empirical_bayes_rate(0.0, attempts, strong_mean),
            module._empirical_bayes_rate(1.0, attempts, strong_mean),
        ]
    )
    assert np.isclose(weak_rates.mean(), (0.5 + 4.0 * weak_mean) / 6.0)
    assert np.isclose(strong_rates.mean(), (0.5 + 4.0 * strong_mean) / 6.0)
    assert strong_rates.mean() > weak_rates.mean()


def test_calibrated_irt_probability_matches_arm_mean_and_peaks_uplift_midrange() -> None:
    module = _module()
    easiness = np.linspace(-4.0, 4.0, 401)
    weak = module._calibrated_irt_probability(easiness, 0.60)
    strong = module._calibrated_irt_probability(easiness, 0.72)
    uplift = strong - weak
    assert np.isclose(weak.mean(), 0.60)
    assert np.isclose(strong.mean(), 0.72)
    assert int(np.argmax(uplift)) not in (0, len(uplift) - 1)
    assert uplift[len(uplift) // 2] > uplift[0]
    assert uplift[len(uplift) // 2] > uplift[-1]


def test_shuffled_control_preserves_source_outcomes_but_breaks_pairing() -> None:
    module = _module()
    spec = module.CandidateSpec(
        "shuffle",
        "word",
        8,
        "ridge-uplift",
        label_mode="shuffled",
    )
    weak = np.asarray([0.0, 0.2, 0.4, 0.6, 0.1, 0.3, 0.5, 0.7])
    strong = np.asarray([0.9, 0.7, 0.5, 0.3, 1.0, 0.8, 0.6, 0.4])
    sources = ["a"] * 4 + ["b"] * 4
    shuffled_weak, shuffled_strong = module._training_outcomes(
        spec,
        weak,
        strong,
        sources,
        seed=43,
    )
    for indices in (slice(0, 4), slice(4, 8)):
        assert sorted(shuffled_weak[indices]) == sorted(weak[indices])
        assert sorted(shuffled_strong[indices]) == sorted(strong[indices])
    assert not np.array_equal(shuffled_weak, weak)
    assert not np.array_equal(shuffled_strong, strong)


def test_promotion_decision_applies_relative_quality_and_paired_allowance() -> None:
    module = _module()
    best = {
        "arm": "best",
        "reward": 0.95,
        "cost_usd": 680.0,
    }
    passed = module._promotion_decision(
        0.933,
        268.0,
        best,
        [-0.035, -0.005],
    )
    assert passed["quality_retention"] > 0.95
    assert passed["cost_savings"] > 0.60
    assert passed["paired_quality_passed"] is True
    assert passed["passed"] is True

    failed_interval = module._promotion_decision(
        0.933,
        268.0,
        best,
        [-0.050, -0.005],
    )
    assert failed_interval["point_estimate_passed"] is True
    assert failed_interval["paired_quality_passed"] is False
    assert failed_interval["passed"] is False
