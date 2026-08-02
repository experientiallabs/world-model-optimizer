"""Build and latency-audit a fit-selected BigCodeBench WMO kNN winner."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import cast

import numpy as np
from coding_model_router_bigcodebench_fit import (
    FitData,
    LatencyMetric,
    artifact_size,
    load_fit_data,
    measure_route_latency,
    outcome_matrix,
    outer_splits,
)
from coding_model_router_bigcodebench_lock import SeedWinnerAudit
from coding_model_router_bigcodebench_select_run import CandidateRecord, SeedFitReport
from pydantic import BaseModel, ConfigDict, Field

from wmo.core.files import write_text_atomic
from wmo.optimize.knn import fit_knn_policy
from wmo.optimize.policy import EmbedderSpec, RoutingPolicy, knn_decision

logger = logging.getLogger(__name__)


class KnnArtifactAudit(BaseModel):
    """Content and one-core latency evidence for one selected WMO kNN artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol: str = "bigcodebench-knn-artifact-audit-v1"
    candidate_name: str = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bank_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_bytes: int = Field(gt=0)
    decisions: int = Field(gt=0)
    latency_p50_ms: float = Field(ge=0.0)
    latency_p95_ms: float = Field(ge=0.0)
    latency_passed: bool
    network_calls_per_route: int = Field(default=0, ge=0, le=0)
    foundation_model_weights_persisted: bool = False
    target_outcomes_used: bool = False
    outer_heldout_evaluated: bool = False


def _sha256(path: Path) -> str:
    """Return one artifact's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(record: CandidateRecord) -> dict[str, str | int | float | bool | None]:
    """Load and type-check one canonical candidate configuration."""
    value = json.loads(record.config_json)
    if not isinstance(value, dict):
        raise ValueError("candidate config must be one JSON object")
    return {str(key): cast(str | int | float | bool | None, item) for key, item in value.items()}


def _number(config: dict[str, str | int | float | bool | None], key: str) -> float:
    """Read one numeric config field without accepting booleans."""
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"kNN config field {key} is not numeric")
    return float(value)


def fit_knn_winner(
    data: FitData,
    fit_indices: np.ndarray,
    record: CandidateRecord,
    *,
    baseline_arm: str,
    artifact_dir: Path,
) -> tuple[RoutingPolicy, Path, Path]:
    """Refit one selected kNN config on the complete outer-fit partition."""
    if record.family != "knn":
        raise ValueError("native kNN audit received a non-kNN winner")
    config = _config(record)
    guard = config.get("guard_model")
    guard_model = baseline_arm if guard == "fit-best" else guard
    if not isinstance(guard_model, str):
        raise ValueError("kNN config has no valid guard model")
    guard_mode = config.get("guard_mode")
    if guard_mode not in {"symmetric", "asymmetric"}:
        raise ValueError("kNN config has no valid guard mode")
    dim = int(_number(config, "dim"))
    artifact_dir.mkdir(parents=True, exist_ok=False)
    bank_path = artifact_dir / "knn-bank.npz"
    policy_path = artifact_dir / "policy.json"
    policy = fit_knn_policy(
        outcome_matrix(data),
        bank_path=bank_path,
        fit_ids=[data.task_ids[int(index)] for index in fit_indices],
        embedder=EmbedderSpec(kind="hashing", dim=dim),
        guard_model=guard_model,
        rag_num=int(_number(config, "rag_num")),
        rag_thres=_number(config, "rag_thres"),
        z=_number(config, "z"),
        min_pairs=int(_number(config, "min_pairs")),
        se_floor=True,
        floor_q=0.0,
        pick_lam=_number(config, "pick_lam"),
        fitted_from="bigcodebench-v0.2.4 selected outer fit only",
    ).model_copy(update={"guard_mode": guard_mode})
    policy.save(policy_path)
    return policy, policy_path, bank_path


def latency_audit(
    policy: RoutingPolicy,
    texts: list[str],
    *,
    decisions: int = 10_000,
) -> LatencyMetric:
    """Measure the actual single-request WMO kNN route path without network calls."""
    embedder = policy.embedder.build()

    def route_one(text: str) -> int:
        vector = np.asarray(embedder.embed([text])[0], dtype=np.float64)
        model = knn_decision(policy, vector).model
        return next(index for index, entry in enumerate(policy.pool) if entry.name == model)

    return measure_route_latency(route_one, texts, decisions=decisions)


def audit_knn_winner(
    data: FitData,
    fit_indices: np.ndarray,
    record: CandidateRecord,
    *,
    baseline_arm: str,
    artifact_dir: Path,
    decisions: int = 10_000,
) -> KnnArtifactAudit:
    """Build, measure, and fingerprint one selected outer-fit kNN artifact."""
    policy, policy_path, bank_path = fit_knn_winner(
        data,
        fit_indices,
        record,
        baseline_arm=baseline_arm,
        artifact_dir=artifact_dir,
    )
    latency = latency_audit(
        policy,
        [data.texts[int(index)] for index in fit_indices],
        decisions=decisions,
    )
    return KnnArtifactAudit(
        candidate_name=record.name,
        config_sha256=record.config_sha256,
        policy_sha256=_sha256(policy_path),
        bank_sha256=_sha256(bank_path),
        artifact_bytes=artifact_size([policy_path, bank_path]),
        decisions=latency.decisions,
        latency_p50_ms=latency.p50_ms,
        latency_p95_ms=latency.p95_ms,
        latency_passed=latency.passed,
    )


def audit_seed_knn_winner(
    root: Path,
    *,
    report_path: Path,
    artifact_dir: Path,
    output: Path,
    decisions: int = 10_000,
) -> SeedWinnerAudit:
    """Audit one seed's selected kNN winner and write lock-compatible evidence."""
    if output.exists():
        raise FileExistsError(f"winner audit already exists: {output}")
    report = SeedFitReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    record = next(
        candidate for candidate in report.candidates if candidate.name == report.selected_name
    )
    if record.family != "knn":
        raise ValueError(f"seed {report.seed} selected non-kNN family {record.family}")
    data = load_fit_data(root)
    split = next(split for split in outer_splits(data.groups) if split.seed == report.seed)
    audit = audit_knn_winner(
        data,
        split.train_indices,
        record,
        baseline_arm=report.baseline_arm,
        artifact_dir=artifact_dir,
        decisions=decisions,
    )
    if not audit.latency_passed:
        raise ValueError(f"seed {report.seed} kNN winner failed the frozen one-core latency gate")
    winner = SeedWinnerAudit(
        seed=report.seed,
        seed_report_sha256=_sha256(report_path),
        candidate_name=record.name,
        config_sha256=record.config_sha256,
        artifact_kind="wmo-knn",
        artifact_sha256=audit.policy_sha256,
        sidecar_sha256=audit.bank_sha256,
        artifact_bytes=audit.artifact_bytes,
        decisions=audit.decisions,
        latency_p50_ms=audit.latency_p50_ms,
        latency_p95_ms=audit.latency_p95_ms,
        latency_passed=True,
    )
    write_text_atomic(output, winner.model_dump_json(indent=2) + "\n")
    logger.info(
        "seed=%d audited kNN winner=%s p50_ms=%.6f p95_ms=%.6f bytes=%d",
        report.seed,
        record.name,
        winner.latency_p50_ms,
        winner.latency_p95_ms,
        winner.artifact_bytes,
    )
    return winner


def parse_args() -> argparse.Namespace:
    """Parse the remote kNN winner-audit command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Build and audit one fit-selected kNN winner on the remote CPU."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    audit_seed_knn_winner(
        args.root.resolve(),
        report_path=args.report.resolve(),
        artifact_dir=args.artifact_dir.resolve(),
        output=args.output.resolve(),
    )


if __name__ == "__main__":
    main()
