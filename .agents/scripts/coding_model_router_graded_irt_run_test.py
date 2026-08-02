"""Tests for remote graded IRT seed-report orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import coding_model_router_graded_irt_run as run
import numpy as np
import pytest
from coding_model_router_graded_irt_nested import (
    IrtStructure,
    OperatingPoint,
    PolicyMetric,
    frozen_operating_points,
    frozen_structures,
)
from coding_model_router_graded_irt_run import (
    IrtData,
    PolicyMetricRow,
    RouteLatencyReport,
    RouteLatencyRow,
    SeedMetricReport,
    load_irt_data,
    measure_latency_report,
    select_reports,
)
from coding_model_router_graded_swerebench_fit import ARMS, Data
from pydantic import ValidationError


def _metric(
    seed: int,
    structure: IrtStructure,
    point: OperatingPoint,
    *,
    eligible: bool,
) -> PolicyMetric:
    return PolicyMetric(
        seed=seed,
        structure_key=structure.key,
        operating_key=point.key,
        structure_order=structure.order,
        operating_order=point.order,
        coefficient_count=structure.coefficient_count(512),
        latent_dimension=structure.latent_dimension,
        cost_penalty=point.cost_penalty,
        kl_radius=point.kl_radius,
        reward=0.96 if eligible else 0.90,
        cost_per_task=4.0 if eligible else 9.0,
        quality_retention=0.96 if eligible else 0.90,
        cost_savings=0.60 if eligible else 0.10,
        matched_blind_advantage=0.02 if eligible else 0.0,
        shuffled_label_advantage=0.01 if eligible else 0.0,
        robust_quality_margin=0.01 if eligible else -0.01,
        robust_cost_margin=0.01 if eligible else -0.01,
        worst_large_repository_loss=0.01,
        dominated_by_static=False,
        eligible=eligible,
    )


def _seed_report(seed: int) -> SeedMetricReport:
    structures = frozen_structures()
    points = frozen_operating_points()
    winner = (structures[0].key, points[1].key)
    metrics = [
        PolicyMetricRow.from_metric(
            _metric(
                seed,
                structure,
                point,
                eligible=(structure.key, point.key) == winner,
            )
        )
        for structure in structures
        for point in points
    ]
    return SeedMetricReport(
        seed=seed,
        tasks=652,
        metrics=metrics,
        corpus_sha256="a" * 64,
        outcomes_sha256="b" * 64,
        audit_sha256="c" * 64,
        task_ids_sha256="d" * 64,
        source_sha256={name: "e" * 64 for name in run.SOURCE_NAMES},
    )


def _latency_report(rows: list[RouteLatencyRow]) -> RouteLatencyReport:
    return RouteLatencyReport(
        rows=rows,
        source_sha256={name: "e" * 64 for name in run.SOURCE_NAMES},
        development_corpus_sha256="a" * 64,
        development_outcomes_sha256="b" * 64,
        development_audit_sha256="c" * 64,
        latency_corpus_sha256="f" * 64,
        latency_task_ids_sha256="0" * 64,
        single_core=True,
        decisions_per_policy=10_000,
        measurement_method="synthetic test timing",
    )


def test_policy_metric_row_round_trips_without_arrays() -> None:
    metric = _metric(11, frozen_structures()[0], frozen_operating_points()[1], eligible=True)
    row = PolicyMetricRow.from_metric(metric)
    assert row.to_metric() == metric
    assert all(not isinstance(value, np.ndarray) for value in row.model_dump().values())


def test_seed_report_rejects_an_incomplete_grid() -> None:
    metric = _metric(11, frozen_structures()[0], frozen_operating_points()[1], eligible=True)
    with pytest.raises(ValidationError, match="incomplete"):
        SeedMetricReport(
            seed=11,
            tasks=652,
            metrics=[PolicyMetricRow.from_metric(metric)],
            corpus_sha256="a" * 64,
            outcomes_sha256="b" * 64,
            audit_sha256="c" * 64,
            task_ids_sha256="d" * 64,
            source_sha256={name: "e" * 64 for name in run.SOURCE_NAMES},
        )


def test_selection_requires_exact_latency_coverage_for_eligible_policies() -> None:
    reports = [_seed_report(seed) for seed in (11, 23, 37, 41, 59)]
    key = f"{frozen_structures()[0].key}__{frozen_operating_points()[1].key}"
    latency = _latency_report(
        [
            RouteLatencyRow(
                policy_key=key,
                p50_ms=0.5,
                p95_ms=1.0,
                decisions=10_000,
                network_calls=0,
            )
        ]
    )
    result, scientific = select_reports(reports, latency)
    assert scientific == {key}
    assert result.selected_key == key

    with pytest.raises(ValueError, match="exactly cover"):
        select_reports(
            reports,
            _latency_report([]),
        )


def test_count_loader_preserves_exact_denominators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_ids = ["repo__one-1", "repo__two-2"]
    corpus = tmp_path / "corpus.json"
    outcomes = tmp_path / "outcomes.jsonl"
    audit = tmp_path / "audit.json"
    corpus.write_text(
        json.dumps(
            {
                "tasks": [
                    {"task_id": task_ids[0], "prompt": "Fix parser one."},
                    {"task_id": task_ids[1], "prompt": "Fix parser two."},
                ]
            }
        ),
        encoding="utf-8",
    )
    outcomes.write_text("placeholder\n", encoding="utf-8")
    audit.write_text("{}\n", encoding="utf-8")
    rewards = np.repeat(np.asarray([[0.5] * len(ARMS)]), len(task_ids), axis=0)
    costs = np.repeat(np.asarray([[1.0] * len(ARMS)]), len(task_ids), axis=0)
    monkeypatch.setattr(
        run,
        "load_data",
        lambda *args: Data(
            task_ids=task_ids,
            repositories=["repo/one", "repo/two"],
            texts=["unused one", "unused two"],
            rewards=rewards,
            costs=costs,
            rough_cumulative_spend_usd=1.0,
        ),
    )
    rows = [
        {
            "task_id": task_id,
            "arm": arm,
            "f2p_passed": 1,
            "f2p_total": 2,
        }
        for task_id in task_ids
        for arm in ARMS
    ]
    monkeypatch.setattr(run, "_rows", lambda path: rows)
    data = load_irt_data(corpus, outcomes, audit)
    assert data.prompts == ["Fix parser one.", "Fix parser two."]
    assert np.array_equal(data.passed, np.ones((2, len(ARMS))))
    assert np.array_equal(data.total, np.full((2, len(ARMS)), 2.0))


def test_latency_audit_times_online_prompt_routes_without_persisting_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reports = [_seed_report(seed) for seed in (11, 23, 37, 41, 59)]
    monkeypatch.setattr(run, "LATENCY_DECISIONS", 10)
    monkeypatch.setattr(
        run,
        "_source_sha256",
        lambda: {name: "e" * 64 for name in run.SOURCE_NAMES},
    )
    monkeypatch.setattr(run, "fit_projected_binomial_irt", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        run,
        "frozen_feature_view",
        lambda texts, name: np.ones((len(texts), 2), dtype=np.float64),
    )
    monkeypatch.setattr(
        run,
        "predict_projected_probabilities",
        lambda fit, features: np.repeat(
            np.asarray([[0.95, 0.8, 0.7, 0.6, 0.5, 0.9]], dtype=np.float64),
            len(features),
            axis=0,
        ),
    )
    data = IrtData(
        task_ids=["repo__one-1", "repo__two-2"],
        repositories=np.asarray(["repo/one", "repo/two"], dtype=object),
        prompts=["Fix parser one.", "Fix parser two."],
        passed=np.asarray([[9, 8, 7, 6, 5, 9], [9, 8, 7, 6, 5, 9]], dtype=np.float64),
        total=np.full((2, len(ARMS)), 10.0),
        costs=np.asarray([[1, 2, 3, 4, 5, 10], [1, 2, 3, 4, 5, 10]], dtype=np.float64),
        corpus_sha256="a" * 64,
        outcomes_sha256="b" * 64,
        audit_sha256="c" * 64,
    )
    report = measure_latency_report(
        data,
        [{"task_id": "latency__one-1", "prompt": "Fix an unseen parser."}],
        reports,
        latency_corpus_sha256="f" * 64,
        single_core=True,
    )
    assert len(report.rows) == 1
    assert report.rows[0].decisions == 10
    assert report.rows[0].network_calls == 0
    assert report.fitted_coefficients_persisted is False
