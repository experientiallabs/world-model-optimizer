"""Tests for broad SWE-smith artifact auditing and consensus selection."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import joblib
import numpy as np


def _module() -> ModuleType:
    scripts = Path(__file__).parent
    sys.path.insert(0, str(scripts))
    path = scripts / "coding_model_router_swe_smith_postfit.py"
    spec = importlib.util.spec_from_file_location("coding_model_router_swe_smith_postfit", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_numeric_artifact_round_trip_has_stable_routes(tmp_path: Path) -> None:
    module = _module()
    candidate = next(
        spec
        for spec in module.selection._candidate_specs()
        if spec.name == "hash512-ridge-heads-a1"
    )
    winner = {
        "name": candidate.name,
        "family": "numeric",
        "config": dataclasses.asdict(candidate),
        "primary": {"threshold": 0.0},
    }
    texts = [f"fix function {index} with regression test" for index in range(12)]
    weak = np.asarray([float(index % 3 == 0) for index in range(12)])
    strong = np.asarray([float(index % 2 == 0) for index in range(12)])
    first_path = tmp_path / "first.joblib"
    second_path = tmp_path / "second.joblib"
    module.selection._fit_artifact(winner, texts, weak, strong, first_path)
    module.selection._fit_artifact(winner, texts, weak, strong, second_path)
    first = module._artifact_routes(joblib.load(first_path), texts)
    second = module._artifact_routes(joblib.load(second_path), texts)
    assert first.tolist() == second.tolist()
    assert module._route_digest(first) == module._route_digest(second)


def test_knn_artifact_routes_unseen_prompts_without_network(tmp_path: Path) -> None:
    module = _module()
    config = module.selection.KnnConfig(512, 8, 0.9, 0.0, 8)
    winner = {
        "name": config.name,
        "family": "knn",
        "config": dataclasses.asdict(config),
        "primary": {"threshold": 0.0},
    }
    texts = [f"repair parser edge case {index}" for index in range(16)]
    weak = np.zeros(16)
    strong = np.ones(16)
    artifact = tmp_path / "knn.joblib"
    module.selection._fit_artifact(winner, texts, weak, strong, artifact)
    routes = module._artifact_routes(joblib.load(artifact), ["repair parser edge case 99"])
    assert routes.tolist() == [True]


def test_candidate_summary_requires_every_seed_to_retain_quality() -> None:
    module = _module()
    rows = [
        {
            "family": "numeric",
            "config": {"value": seed},
            "primary": {
                "retention": retention,
                "strong_traffic": 0.5,
                "router_reward": 0.4,
            },
        }
        for seed, retention in enumerate((0.96, 0.95, 0.97, 0.94, 0.98))
    ]
    summary = module._candidate_summary("candidate", rows)
    assert summary["minimum_retention"] == 0.94
    assert summary["fit_quality_feasible"] is False
    assert len(module._canonical_order()) == 455


def test_outcome_blind_controls_have_frozen_route_counts() -> None:
    module = _module()
    task_ids = [f"task-{index}" for index in range(21)]
    first = module._hashed_exact_count(task_ids, 7, "matched")
    second = module._hashed_exact_count(task_ids, 7, "matched")
    assert first.tolist() == second.tolist()
    assert int(np.sum(first)) == 7
    uniform = module._hashed_uniform(task_ids, "uniform")
    assert uniform.tolist() == module._hashed_uniform(task_ids, "uniform").tolist()


def test_repository_bootstrap_is_seed_balanced_and_deterministic() -> None:
    module = _module()
    rows = [
        {
            "seed": seed,
            "repo": f"repo-{repo}",
            "router_reward": 1.0,
            "strong_reward": 1.0,
            "weak_reward": 0.0,
            "task_blind_reward": 0.0,
            "shuffled_reward": 0.0,
            "random_reward": 0.0,
        }
        for seed in range(5)
        for repo in range(3)
    ]
    first = module._bootstrap_sample(rows, np.random.default_rng(20260731))
    second = module._bootstrap_sample(rows, np.random.default_rng(20260731))
    assert first == second
    assert len(first) == len(rows)
    assert {row["seed"] for row in first} == set(range(5))


def test_external_replay_and_promotion_integrate_on_synthetic_rows(tmp_path: Path) -> None:
    module = _module()
    sources = tmp_path / "sources"
    reports = tmp_path / "reports"
    consensus = tmp_path / "consensus"
    audits = tmp_path / "audits"
    controls = tmp_path / "controls"
    for directory in (sources, reports, consensus, audits):
        directory.mkdir()
    candidate = next(
        spec
        for spec in module.selection._candidate_specs()
        if spec.name == "hash512-ridge-heads-a1"
    )
    shuffled = next(
        spec
        for spec in module.selection._control_specs()
        if spec.name == "hash2048-ridge-heads-shuffled-a1"
    )
    winner = {
        "name": candidate.name,
        "family": "numeric",
        "is_control": False,
        "config": dataclasses.asdict(candidate),
        "primary": {
            "threshold": 0.0,
            "retention": 1.0,
            "strong_traffic": 1.0,
            "router_reward": 1.0,
        },
    }
    shuffled_row = {
        "name": shuffled.name,
        "family": "numeric-control",
        "is_control": True,
        "config": dataclasses.asdict(shuffled),
        "primary": {
            "threshold": 0.0,
            "retention": 1.0,
            "strong_traffic": 1.0,
            "router_reward": 1.0,
        },
    }
    manifest_rows = []
    for seed in range(5):
        fit_rows = [
            {
                "instance_id": f"fit-{seed}-{index}",
                "repo": f"fit-repo-{index // 2}",
                "text": f"fix fit issue {index}",
                "cheap_reward": 0.0,
                "strong_reward": float(index % 2 == 0),
            }
            for index in range(10)
        ]
        heldout_rows = [
            {
                "instance_id": f"heldout-{seed}-{index}",
                "repo": f"heldout-repo-{index // 2}",
                "text": f"fix heldout issue {index}",
                "cheap_reward": 0.0,
                "strong_reward": float(index % 2 == 0),
            }
            for index in range(10)
        ]
        fit_path = sources / f"seed-{seed}-fit.json"
        heldout_path = sources / f"seed-{seed}-heldout.json"
        module._write_json(fit_path, fit_rows)
        module._write_json(heldout_path, heldout_rows)
        manifest_rows.append(
            {
                "seed": seed,
                "fit_sha256": module._sha256_file(fit_path),
                "heldout_sha256": module._sha256_file(heldout_path),
            }
        )
        fit_texts = [str(row["text"]) for row in fit_rows]
        weak = np.asarray([float(row["cheap_reward"]) for row in fit_rows])
        strong = np.asarray([float(row["strong_reward"]) for row in fit_rows])
        artifact_path = consensus / f"seed-{seed}.joblib"
        module.selection._fit_artifact(winner, fit_texts, weak, strong, artifact_path)
        consensus_report_path = consensus / f"seed-{seed}.json"
        module._write_json(
            consensus_report_path,
            {"seed": seed, "consensus_name": candidate.name, "winner": winner},
        )
        module._write_json(
            audits / f"seed-{seed}.json",
            {
                "passed": True,
                "report_sha256": module._sha256_file(consensus_report_path),
                "artifact_sha256": module._sha256_file(artifact_path),
            },
        )
        module._write_json(
            reports / f"seed-{seed}.json",
            {"seed": seed, "leaderboard": [winner, shuffled_row]},
        )
    split_manifest = tmp_path / "split-manifest.json"
    module._write_json(split_manifest, {"seeds": manifest_rows})
    lock = tmp_path / "selection-lock.json"
    module._write_json(
        lock,
        {
            "consensus_feasible": True,
            "consensus_name": candidate.name,
            "outer_heldout_evaluated": False,
        },
    )
    evaluation = tmp_path / "evaluation.json"
    module._evaluate(
        argparse.Namespace(
            output=evaluation,
            lock=lock,
            split_manifest=split_manifest,
            sources_dir=sources,
            reports_dir=reports,
            consensus_dir=consensus,
            audits_dir=audits,
            control_artifact_dir=controls,
        )
    )
    evaluated = module._read_object(evaluation)
    assert evaluated["outer_heldout_replay_count"] == 1
    assert len(evaluated["rows"]) == 50
    module.BOOTSTRAP_SAMPLES = 50
    promotion = tmp_path / "promotion.json"
    module._promote(
        argparse.Namespace(
            evaluation=evaluation,
            lock=lock,
            audits_dir=audits,
            output=promotion,
        )
    )
    promoted = module._read_object(promotion)
    assert promoted["bootstrap_samples"] == 50
    assert promoted["target_outcomes_used"] is False
