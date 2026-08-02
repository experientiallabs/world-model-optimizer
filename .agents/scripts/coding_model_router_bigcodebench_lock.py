"""Assemble the immutable BigCodeBench selection lock from audited seed winners."""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Literal

from coding_model_router_bigcodebench_fit import (
    DeploymentConsensus,
    LockedCandidate,
    SeedSelection,
    SelectionLock,
    require_selection_lock,
    write_selection_lock,
)
from coding_model_router_bigcodebench_select_run import CandidateRecord, SeedFitReport
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)


class SeedWinnerAudit(BaseModel):
    """One outer seed's content-addressed, latency-verified winner artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol: Literal["bigcodebench-winner-artifact-audit-v1"] = (
        "bigcodebench-winner-artifact-audit-v1"
    )
    seed: int = Field(ge=0, le=4)
    seed_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_name: str = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_kind: Literal["wmo-knn", "numeric-router"]
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sidecar_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    artifact_bytes: int = Field(gt=0)
    decisions: int = Field(ge=10_000)
    latency_p50_ms: float = Field(ge=0.0, lt=5.0)
    latency_p95_ms: float = Field(ge=0.0, lt=20.0)
    latency_passed: Literal[True]
    network_calls_per_route: Literal[0] = 0
    foundation_model_weights_persisted: Literal[False] = False
    target_outcomes_used: Literal[False] = False
    outer_heldout_evaluated: Literal[False] = False


class LockInputs(BaseModel):
    """Validated five-seed evidence used to assemble one selection lock."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    reports: list[SeedFitReport] = Field(min_length=5, max_length=5)
    audits: list[SeedWinnerAudit] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def _five_matched_seeds(self) -> LockInputs:
        expected = list(range(5))
        if sorted(report.seed for report in self.reports) != expected:
            raise ValueError("fit reports must contain seeds 0 through 4 exactly once")
        if sorted(audit.seed for audit in self.audits) != expected:
            raise ValueError("winner audits must contain seeds 0 through 4 exactly once")
        return self


def _sha256(path: Path) -> str:
    """Return one evidence file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected(report: SeedFitReport) -> CandidateRecord:
    """Return the exact candidate named by one validated seed report."""
    return next(
        candidate for candidate in report.candidates if candidate.name == report.selected_name
    )


def fit_only_deployment_consensus(reports: list[SeedFitReport]) -> DeploymentConsensus:
    """Select one configuration from five complete fit-only candidate inventories."""
    ordered = sorted(reports, key=lambda report: report.seed)
    if [report.seed for report in ordered] != list(range(5)):
        raise ValueError("deployment consensus requires seeds 0 through 4")
    reference = {candidate.name: candidate for candidate in ordered[0].candidates}
    if len(reference) != 1_028:
        raise ValueError("deployment consensus requires the frozen candidate inventory")
    candidates_by_seed: list[dict[str, CandidateRecord]] = []
    for report in ordered:
        candidates = {candidate.name: candidate for candidate in report.candidates}
        if set(candidates) != set(reference):
            raise ValueError("fit reports contain different candidate inventories")
        for name, candidate in candidates.items():
            expected = reference[name]
            if (
                candidate.family != expected.family
                or candidate.order != expected.order
                or candidate.config_json != expected.config_json
                or candidate.config_sha256 != expected.config_sha256
            ):
                raise ValueError(f"fit reports disagree on candidate identity {name}")
        candidates_by_seed.append(candidates)
    summaries: list[tuple[CandidateRecord, float, float, float, float, bool]] = []
    for name, candidate in reference.items():
        records = [candidates[name] for candidates in candidates_by_seed]
        retentions = [
            record.fit_reward / report.baseline_fit_reward
            for record, report in zip(records, ordered, strict=True)
        ]
        summaries.append(
            (
                candidate,
                sum(record.fit_reward for record in records) / 5.0,
                sum(record.fit_cost_usd for record in records) / 5.0,
                sum(record.matched_blind_reward for record in records) / 5.0,
                min(retentions),
                all(retention >= 0.95 for retention in retentions),
            )
        )
    feasible = [summary for summary in summaries if summary[-1]]
    if feasible:
        selected = min(feasible, key=lambda summary: (summary[2], summary[0].order))
    else:
        selected = min(
            summaries,
            key=lambda summary: (-summary[4], -summary[1], summary[2], summary[0].order),
        )
    candidate, mean_reward, mean_cost, mean_blind_reward, min_retention, is_feasible = selected
    return DeploymentConsensus(
        family=candidate.family,
        name=candidate.name,
        order=candidate.order,
        config_json=candidate.config_json,
        config_sha256=candidate.config_sha256,
        mean_fit_reward=mean_reward,
        mean_fit_cost_usd=mean_cost,
        mean_matched_blind_reward=mean_blind_reward,
        mean_baseline_reward=sum(report.baseline_fit_reward for report in ordered) / 5.0,
        minimum_seed_retention=min_retention,
        fit_quality_feasible=is_feasible,
    )


def load_lock_inputs(
    report_paths: list[Path],
    audit_paths: list[Path],
) -> LockInputs:
    """Load five fit reports and their five independently written winner audits."""
    reports = [
        SeedFitReport.model_validate_json(path.read_text(encoding="utf-8")) for path in report_paths
    ]
    audits = [
        SeedWinnerAudit.model_validate_json(path.read_text(encoding="utf-8"))
        for path in audit_paths
    ]
    inputs = LockInputs(reports=reports, audits=audits)
    report_path_by_seed = {
        report.seed: path for report, path in zip(reports, report_paths, strict=True)
    }
    audit_by_seed = {audit.seed: audit for audit in audits}
    for report in reports:
        selected = _selected(report)
        audit = audit_by_seed[report.seed]
        if audit.seed_report_sha256 != _sha256(report_path_by_seed[report.seed]):
            raise ValueError(f"seed {report.seed} audit names a different fit report")
        if audit.candidate_name != selected.name or audit.config_sha256 != selected.config_sha256:
            raise ValueError(f"seed {report.seed} audit names a different winner")
    return inputs


def assemble_selection_lock(
    root: Path,
    *,
    report_paths: list[Path],
    audit_paths: list[Path],
    output: Path,
) -> SelectionLock:
    """Verify all evidence and atomically publish the heldout-evaluation boundary."""
    inputs = load_lock_inputs(report_paths, audit_paths)
    reports = sorted(inputs.reports, key=lambda report: report.seed)
    audits = {audit.seed: audit for audit in inputs.audits}
    current = {
        "tasks_sha256": _sha256(root / "tasks.jsonl"),
        "scores_sha256": _sha256(root / "scores.jsonl"),
        "outcomes_sha256": _sha256(root / "outcomes.jsonl"),
        "oracle_report_sha256": _sha256(root / "oracle-report.json"),
    }
    commits = {report.code_commit for report in reports}
    if len(commits) != 1:
        raise ValueError("seed fit reports use different source commits")
    for report in reports:
        for field, digest in current.items():
            if getattr(report, field) != digest:
                raise ValueError(f"seed {report.seed} {field} differs from the current matrix")
    seeds: list[SeedSelection] = []
    for report in reports:
        candidate = _selected(report)
        audit = audits[report.seed]
        seeds.append(
            SeedSelection(
                seed=report.seed,
                fit_tasks=report.fit_tasks,
                heldout_tasks=report.heldout_tasks,
                fit_ids_sha256=report.fit_ids_sha256,
                heldout_ids_sha256=report.heldout_ids_sha256,
                baseline_arm=report.baseline_arm,
                baseline_fit_reward=report.baseline_fit_reward,
                baseline_fit_cost_usd=report.baseline_fit_cost_usd,
                selected=LockedCandidate(
                    family=candidate.family,
                    name=candidate.name,
                    config_json=candidate.config_json,
                    config_sha256=candidate.config_sha256,
                    fit_reward=candidate.fit_reward,
                    fit_cost_usd=candidate.fit_cost_usd,
                    matched_blind_reward=candidate.matched_blind_reward,
                    latency_p95_ms=audit.latency_p95_ms,
                    artifact_bytes=audit.artifact_bytes,
                ),
            )
        )
    lock = SelectionLock(
        protocol="bigcodebench-fit-only-selection-v1",
        **current,
        code_commit=next(iter(commits)),
        seeds=seeds,
        deployment_consensus=fit_only_deployment_consensus(reports),
    )
    write_selection_lock(output, lock)
    return require_selection_lock(root, output)


def parse_args() -> argparse.Namespace:
    """Parse the lock-assembly command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--audits-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Assemble one immutable selection lock from five audited seed winners."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    lock = assemble_selection_lock(
        args.root.resolve(),
        report_paths=[args.reports_dir.resolve() / f"seed-{seed}.json" for seed in range(5)],
        audit_paths=[args.audits_dir.resolve() / f"seed-{seed}-audit.json" for seed in range(5)],
        output=args.output.resolve(),
    )
    logger.info("selection lock assembled seeds=%d output=%s", len(lock.seeds), args.output)


if __name__ == "__main__":
    main()
