"""Refit one externally promoted BigCodeBench consensus on all source rows."""

from __future__ import annotations

import argparse
import hashlib
import logging
from collections import Counter
from pathlib import Path
from typing import Literal, cast

import joblib
import numpy as np
from coding_model_router_bigcodebench_fit import (
    ARMS,
    FitData,
    SelectionLock,
    artifact_size,
    load_fit_data,
    measure_route_latency,
    require_selection_lock,
)
from coding_model_router_bigcodebench_knn_audit import fit_knn_winner, latency_audit
from coding_model_router_bigcodebench_lock import load_lock_inputs
from coding_model_router_bigcodebench_numeric_audit import (
    NumericPayload,
    _features,
    _record_spec,
    fit_numeric_payload,
    numeric_choices,
)
from coding_model_router_bigcodebench_promote import ExternalPromotionReport
from coding_model_router_bigcodebench_select import _candidate_choices
from coding_model_router_bigcodebench_select_run import CandidateRecord, SeedFitReport
from pydantic import BaseModel, ConfigDict, Field

from wmo.core.files import write_text_atomic
from wmo.optimize.policy import RoutingPolicy

logger = logging.getLogger(__name__)
DEPLOYMENT_SEED = 20_260_731
DECISIONS = 10_000


class DeploymentArtifactReport(BaseModel):
    """Content-addressed full-source artifact and latency evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol: Literal["bigcodebench-full-source-deployment-v1"] = (
        "bigcodebench-full-source-deployment-v1"
    )
    code_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    selection_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    external_promotion_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    family: Literal["knn", "ordinal", "doubly-robust", "empirical-bayes"]
    candidate_name: str = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_arm: str
    deployment_seed: int
    fit_tasks: int = Field(gt=0)
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
    source_outer_heldout_evaluated: Literal[True] = True
    target_outcomes_used: Literal[False] = False
    target_evaluated: Literal[False] = False


def _sha256(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_seed_reports(reports_dir: Path, audits_dir: Path) -> list[SeedFitReport]:
    """Load five seed inventories whose content digests match winner audits."""
    inputs = load_lock_inputs(
        [reports_dir / f"seed-{seed}.json" for seed in range(5)],
        [audits_dir / f"seed-{seed}-audit.json" for seed in range(5)],
    )
    reports = sorted(inputs.reports, key=lambda report: report.seed)
    if [report.seed for report in reports] != list(range(5)):
        raise ValueError("deployment reports must contain seeds 0 through 4")
    return reports


def deployment_baseline(reports: list[SeedFitReport]) -> str:
    """Choose the protocol's majority fit-selected deployment guard arm."""
    if sorted(report.seed for report in reports) != list(range(5)):
        raise ValueError("deployment baseline needs seeds 0 through 4")
    counts = Counter(report.baseline_arm for report in reports)
    if set(counts) - set(ARMS):
        raise ValueError("deployment reports name an unknown baseline arm")
    maximum = max(counts.values())
    tied = [arm for arm in ARMS if counts[arm] == maximum]
    static_values: dict[str, list[tuple[float, float]]] = {arm: [] for arm in tied}
    for report in reports:
        controls = {control.name: control for control in report.controls}
        for arm in tied:
            control = controls.get(f"static-{arm}")
            if control is None:
                raise ValueError(f"seed {report.seed} lacks static control {arm}")
            static_values[arm].append((control.reward, control.cost_usd))
    return min(
        tied,
        key=lambda arm: (
            -float(np.mean([value[0] for value in static_values[arm]])),
            float(np.mean([value[1] for value in static_values[arm]])),
            ARMS.index(arm),
        ),
    )


def consensus_record(lock: SelectionLock, reports: list[SeedFitReport]) -> CandidateRecord:
    """Recover the exact consensus candidate from every complete fit inventory."""
    records: list[CandidateRecord] = []
    for report in reports:
        matches = [
            candidate
            for candidate in report.candidates
            if candidate.name == lock.deployment_consensus.name
        ]
        if len(matches) != 1:
            raise ValueError(f"seed {report.seed} lacks one exact consensus candidate")
        records.append(matches[0])
    reference = records[0]
    if any(
        (
            record.family,
            record.name,
            record.order,
            record.config_json,
            record.config_sha256,
        )
        != (
            reference.family,
            reference.name,
            reference.order,
            reference.config_json,
            reference.config_sha256,
        )
        for record in records[1:]
    ):
        raise ValueError("seed reports disagree on the consensus candidate identity")
    consensus = lock.deployment_consensus
    if (
        reference.family != consensus.family
        or reference.name != consensus.name
        or reference.order != consensus.order
        or reference.config_json != consensus.config_json
        or reference.config_sha256 != consensus.config_sha256
    ):
        raise ValueError("selection lock consensus differs from the fit inventories")
    return reference


def require_reports_match_lock(lock: SelectionLock, reports: list[SeedFitReport]) -> None:
    """Prove the audited reports reproduce every seed field frozen in the lock."""
    by_seed = {report.seed: report for report in reports}
    if set(by_seed) != set(range(5)):
        raise ValueError("deployment reports must contain five unique seeds")
    for selection in lock.seeds:
        report = by_seed[selection.seed]
        selected = next(
            candidate for candidate in report.candidates if candidate.name == report.selected_name
        )
        if (
            report.code_commit != lock.code_commit
            or report.tasks_sha256 != lock.tasks_sha256
            or report.scores_sha256 != lock.scores_sha256
            or report.outcomes_sha256 != lock.outcomes_sha256
            or report.oracle_report_sha256 != lock.oracle_report_sha256
            or report.fit_tasks != selection.fit_tasks
            or report.heldout_tasks != selection.heldout_tasks
            or report.fit_ids_sha256 != selection.fit_ids_sha256
            or report.heldout_ids_sha256 != selection.heldout_ids_sha256
            or report.baseline_arm != selection.baseline_arm
            or report.baseline_fit_reward != selection.baseline_fit_reward
            or report.baseline_fit_cost_usd != selection.baseline_fit_cost_usd
            or selected.family != selection.selected.family
            or selected.name != selection.selected.name
            or selected.config_json != selection.selected.config_json
            or selected.config_sha256 != selection.selected.config_sha256
        ):
            raise ValueError(f"seed {selection.seed} report differs from the selection lock")


def require_external_promotion(lock_path: Path, promotion_path: Path) -> ExternalPromotionReport:
    """Require one positive promotion verdict bound to the exact selection lock."""
    report = ExternalPromotionReport.model_validate_json(promotion_path.read_text(encoding="utf-8"))
    if not report.passed:
        raise ValueError("full-source deployment requires external promotion")
    if report.selection_lock_sha256 != _sha256(lock_path):
        raise ValueError("external promotion names a different selection lock")
    return report


def _fit_numeric(
    data: FitData,
    record: CandidateRecord,
    artifact_dir: Path,
) -> tuple[Path, float, float]:
    """Fit, persist, reload, and latency-audit one full-source numeric router."""
    spec = _record_spec(record)
    indices = np.arange(len(data.task_ids), dtype=np.int64)
    payload = fit_numeric_payload(
        data,
        indices,
        spec,
        config_sha256=record.config_sha256,
        seed=DEPLOYMENT_SEED,
    )
    texts = list(data.texts)
    groups = list(data.groups)
    hard = [bool(value) for value in data.is_hard]
    choices = numeric_choices(payload, spec, texts, groups, hard)
    features = _features(
        texts,
        hard,
        dim=spec.dim,
        structural_scale=payload["structural_scale"],
    )
    expected = _candidate_choices(
        spec,
        data,
        indices,
        indices,
        features,
        features,
        seed=DEPLOYMENT_SEED,
    )
    if not np.array_equal(choices, expected):
        raise AssertionError("full-source numeric routes differ from the locked candidate")
    artifact_dir.mkdir(parents=True, exist_ok=False)
    artifact_path = artifact_dir / "numeric-router.joblib"
    joblib.dump(payload, artifact_path, compress=3, protocol=5)
    restored = cast(NumericPayload, joblib.load(artifact_path))
    if not np.array_equal(choices, numeric_choices(restored, spec, texts, groups, hard)):
        raise AssertionError("persisted full-source numeric router changed its routes")
    if len(set(texts)) != len(texts):
        raise ValueError("numeric latency audit requires unique source prompts")
    metadata = {
        text: (group, hard_value)
        for text, group, hard_value in zip(texts, groups, hard, strict=True)
    }

    def route_one(text: str) -> int:
        group, hard_value = metadata[text]
        return int(numeric_choices(restored, spec, [text], [group], [hard_value])[0])

    latency = measure_route_latency(route_one, texts, decisions=DECISIONS)
    if not latency.passed:
        raise ValueError("full-source numeric router failed the frozen latency gate")
    return artifact_path, latency.p50_ms, latency.p95_ms


def build_deployment(
    root: Path,
    *,
    lock_path: Path,
    promotion_path: Path,
    reports_dir: Path,
    audits_dir: Path,
    artifact_dir: Path,
    output: Path,
) -> DeploymentArtifactReport:
    """Build one immutable full-source artifact after source-only promotion."""
    if output.exists() or artifact_dir.exists():
        raise FileExistsError("deployment artifact or report already exists")
    lock = require_selection_lock(root, lock_path)
    promotion = require_external_promotion(lock_path, promotion_path)
    if promotion.code_commit != lock.code_commit:
        raise ValueError("promotion and selection lock use different source commits")
    if not lock.deployment_consensus.fit_quality_feasible:
        raise ValueError("deployment consensus failed its fit-only quality gate")
    reports = load_seed_reports(reports_dir, audits_dir)
    require_reports_match_lock(lock, reports)
    record = consensus_record(lock, reports)
    baseline_arm = deployment_baseline(reports)
    data = load_fit_data(root)
    indices = np.arange(len(data.task_ids), dtype=np.int64)
    if record.family == "knn":
        _, policy_path, bank_path = fit_knn_winner(
            data,
            indices,
            record,
            baseline_arm=baseline_arm,
            artifact_dir=artifact_dir,
        )
        restored = RoutingPolicy.load(policy_path)
        latency = latency_audit(restored, data.texts, decisions=DECISIONS)
        if not latency.passed:
            raise ValueError("full-source kNN router failed the frozen latency gate")
        artifact_kind: Literal["wmo-knn", "numeric-router"] = "wmo-knn"
        artifact_path = policy_path
        sidecar_sha256 = _sha256(bank_path)
        bytes_used = artifact_size([policy_path, bank_path])
        p50_ms = latency.p50_ms
        p95_ms = latency.p95_ms
    else:
        artifact_kind = "numeric-router"
        artifact_path, p50_ms, p95_ms = _fit_numeric(data, record, artifact_dir)
        sidecar_sha256 = None
        bytes_used = artifact_size([artifact_path])
    report = DeploymentArtifactReport(
        code_commit=lock.code_commit,
        selection_lock_sha256=_sha256(lock_path),
        external_promotion_sha256=_sha256(promotion_path),
        family=record.family,
        candidate_name=record.name,
        config_sha256=record.config_sha256,
        baseline_arm=baseline_arm,
        deployment_seed=DEPLOYMENT_SEED,
        fit_tasks=len(data.task_ids),
        artifact_kind=artifact_kind,
        artifact_sha256=_sha256(artifact_path),
        sidecar_sha256=sidecar_sha256,
        artifact_bytes=bytes_used,
        decisions=DECISIONS,
        latency_p50_ms=p50_ms,
        latency_p95_ms=p95_ms,
        latency_passed=True,
    )
    write_text_atomic(output, report.model_dump_json(indent=2) + "\n")
    logger.info(
        "full-source deployment built family=%s candidate=%s p95_ms=%.6f bytes=%d",
        report.family,
        report.candidate_name,
        report.latency_p95_ms,
        report.artifact_bytes,
    )
    return report


def parse_args() -> argparse.Namespace:
    """Parse the full-source deployment command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--promotion", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--audits-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Build one promoted full-source deployment artifact."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    build_deployment(
        args.root.resolve(),
        lock_path=args.lock.resolve(),
        promotion_path=args.promotion.resolve(),
        reports_dir=args.reports_dir.resolve(),
        audits_dir=args.audits_dir.resolve(),
        artifact_dir=args.artifact_dir.resolve(),
        output=args.output.resolve(),
    )


if __name__ == "__main__":
    main()
