"""Tests for audited BigCodeBench selection-lock assembly."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load(name: str) -> ModuleType:
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fit = _load("coding_model_router_bigcodebench_fit")
select = _load("coding_model_router_bigcodebench_select")
run = _load("coding_model_router_bigcodebench_select_run")
module = _load("coding_model_router_bigcodebench_lock")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _root(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name in ("tasks.jsonl", "scores.jsonl", "outcomes.jsonl"):
        (tmp_path / name).write_text("", encoding="utf-8")
    (tmp_path / "oracle-report.json").write_text(
        json.dumps({"passed": True, "protocol": {"target_outcomes_used": False}}),
        encoding="utf-8",
    )
    return tmp_path


def _candidate(index: int) -> object:
    config_json, config_sha256 = fit.canonical_candidate_config(
        {"family": "ordinal", "index": index}
    )
    return run.CandidateRecord(
        family="ordinal",
        name=f"candidate-{index}",
        order=index,
        config_json=config_json,
        config_sha256=config_sha256,
        fit_reward=0.8,
        fit_cost_usd=0.4,
        matched_blind_reward=0.75,
        matched_blind_cost_usd=0.4,
        baseline_reward=0.82,
        baseline_cost_usd=1.0,
    )


def _controls(fit_tasks: int = 240) -> list[object]:
    kinds_and_names = [
        *(("static", f"static-{arm}") for arm in fit.ARMS),
        ("matched-task-blind", "selected-matched-task-blind"),
        ("random", "seeded-uniform-random"),
        ("cost-only", "fit-cost-only"),
        ("shuffled-label", "selected-shuffled-labels"),
    ]
    return [
        run.ControlRecord(
            kind=kind,
            name=name,
            reward=0.7,
            cost_usd=0.3,
            arm_counts={arm: fit_tasks if arm == fit.ARMS[0] else 0 for arm in fit.ARMS},
        )
        for kind, name in kinds_and_names
    ]


def _evidence(root: Path, evidence: Path) -> tuple[list[Path], list[Path]]:
    candidates = [_candidate(index) for index in range(1_028)]
    reports: list[Path] = []
    audits: list[Path] = []
    for seed in range(5):
        report = run.SeedFitReport(
            seed=seed,
            code_commit="a" * 40,
            tasks_sha256=_sha256(root / "tasks.jsonl"),
            scores_sha256=_sha256(root / "scores.jsonl"),
            outcomes_sha256=_sha256(root / "outcomes.jsonl"),
            oracle_report_sha256=_sha256(root / "oracle-report.json"),
            fit_tasks=240,
            heldout_tasks=60,
            fit_ids_sha256=f"{seed}" * 64,
            heldout_ids_sha256=f"{seed + 1}" * 64,
            baseline_arm=fit.ARMS[-1],
            baseline_fit_reward=0.82,
            baseline_fit_cost_usd=1.0,
            candidates=candidates,
            controls=_controls(),
            selected_name=candidates[seed].name,
        )
        report_path = evidence / f"seed-{seed}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report.model_dump_json(), encoding="utf-8")
        reports.append(report_path)
        audit = module.SeedWinnerAudit(
            seed=seed,
            seed_report_sha256=_sha256(report_path),
            candidate_name=candidates[seed].name,
            config_sha256=candidates[seed].config_sha256,
            artifact_kind="numeric-router",
            artifact_sha256="b" * 64,
            artifact_bytes=1_024,
            decisions=10_000,
            latency_p50_ms=1.0,
            latency_p95_ms=2.0,
            latency_passed=True,
        )
        audit_path = evidence / f"seed-{seed}-audit.json"
        audit_path.write_text(audit.model_dump_json(), encoding="utf-8")
        audits.append(audit_path)
    return reports, audits


def test_assembly_requires_audited_exact_winners(tmp_path: Path) -> None:
    root = _root(tmp_path / "root")
    reports, audits = _evidence(root, tmp_path / "evidence")
    lock = module.assemble_selection_lock(
        root,
        report_paths=reports,
        audit_paths=audits,
        output=tmp_path / "selection-lock.json",
    )
    assert len(lock.seeds) == 5
    assert all(seed.selected.latency_p95_ms == 2.0 for seed in lock.seeds)
    assert lock.deployment_consensus.name == "candidate-0"
    assert lock.deployment_consensus.fit_quality_feasible is True


def test_consensus_rejects_different_candidate_inventories(tmp_path: Path) -> None:
    root = _root(tmp_path / "root")
    reports, _ = _evidence(root, tmp_path / "evidence")
    report = run.SeedFitReport.model_validate_json(reports[0].read_text(encoding="utf-8"))
    candidates = list(report.candidates)
    candidates[-1] = candidates[-1].model_copy(update={"name": "different-candidate"})
    reports[0].write_text(
        report.model_copy(update={"candidates": candidates}).model_dump_json(),
        encoding="utf-8",
    )
    values = [
        run.SeedFitReport.model_validate_json(path.read_text(encoding="utf-8")) for path in reports
    ]
    with pytest.raises(ValueError, match="different candidate inventories"):
        module.fit_only_deployment_consensus(values)


def test_assembly_rejects_an_audit_for_a_different_report(tmp_path: Path) -> None:
    root = _root(tmp_path / "root")
    reports, audits = _evidence(root, tmp_path / "evidence")
    value = module.SeedWinnerAudit.model_validate_json(audits[0].read_text(encoding="utf-8"))
    audits[0].write_text(
        value.model_copy(update={"seed_report_sha256": "f" * 64}).model_dump_json(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different fit report"):
        module.assemble_selection_lock(
            root,
            report_paths=reports,
            audit_paths=audits,
            output=tmp_path / "selection-lock.json",
        )
