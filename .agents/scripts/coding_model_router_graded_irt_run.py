"""Run and aggregate conditional graded IRT seed shards on remote compute.

The seed command persists aggregate scalar policy metrics only. The select command validates all
five seed shards plus a separate route-latency audit and emits a compact decision report. Fitted
coefficients, cross-fit probabilities, task embeddings, and outcome matrices are never serialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from coding_model_router_graded_irt_core import (
    fit_projected_binomial_irt,
    predict_projected_probabilities,
)
from coding_model_router_graded_irt_features import frozen_feature_view, frozen_feature_views
from coding_model_router_graded_irt_nested import (
    OUTER_SEEDS,
    IrtStructure,
    NestedSelectionResult,
    OperatingPoint,
    PolicyMetric,
    RouteLatencyMetric,
    _best_static_arm,
    evaluate_nested_seed,
    frozen_operating_points,
    frozen_structures,
    select_nested_metrics,
)
from coding_model_router_graded_irt_protocol import cosine_knn_laplacian
from coding_model_router_graded_irt_selection import quality_guarded_choices
from coding_model_router_graded_swerebench_fit import (
    ARMS,
    Data,
    _read_object,
    _rows,
    _sha256,
    load_confirmation,
    load_data,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator

from wmo.core.files import write_text_atomic

logger = logging.getLogger(__name__)

SEED_PROTOCOL = "coding-router-graded-irt-seed-metrics-v1"
LATENCY_PROTOCOL = "coding-router-graded-irt-route-latency-v1"
SELECTION_PROTOCOL = "coding-router-graded-irt-selection-v1"
EXPECTED_METRICS_PER_SEED = 80 * 25
LATENCY_DECISIONS = 10_000
SOURCE_NAMES = (
    "coding_model_router_graded_irt_features.py",
    "coding_model_router_graded_irt_core.py",
    "coding_model_router_graded_irt_protocol.py",
    "coding_model_router_graded_irt_selection.py",
    "coding_model_router_graded_irt_nested.py",
    "coding_model_router_graded_irt_run.py",
    "coding_model_router_graded_swerebench_fit.py",
)


class PolicyMetricRow(BaseModel):
    """JSON-safe aggregate representation of one seed and policy metric."""

    model_config = ConfigDict(extra="forbid")

    seed: int
    structure_key: str
    operating_key: str
    structure_order: int
    operating_order: int
    coefficient_count: int
    latent_dimension: int
    cost_penalty: float
    kl_radius: float
    reward: float
    cost_per_task: float
    quality_retention: float
    cost_savings: float
    matched_blind_advantage: float
    shuffled_label_advantage: float
    robust_quality_margin: float
    robust_cost_margin: float
    worst_large_repository_loss: float
    dominated_by_static: bool
    eligible: bool

    @classmethod
    def from_metric(cls, metric: PolicyMetric) -> PolicyMetricRow:
        """Build one persisted row without retaining fitted numeric state."""
        return cls(**metric.__dict__)

    def to_metric(self) -> PolicyMetric:
        """Restore the pure aggregate dataclass used by the selector."""
        return PolicyMetric(**self.model_dump())


class SeedMetricReport(BaseModel):
    """One independently computed seed shard with aggregate metrics only."""

    model_config = ConfigDict(extra="forbid")

    protocol: str = SEED_PROTOCOL
    seed: int
    tasks: int = Field(gt=0)
    metrics: list[PolicyMetricRow]
    corpus_sha256: str
    outcomes_sha256: str
    audit_sha256: str
    task_ids_sha256: str
    source_sha256: dict[str, str]
    target_outcomes_used: bool = False
    deep_swe_outcomes_accessed: bool = False
    confirmation_outcomes_accessed: bool = False
    fitted_coefficients_persisted: bool = False
    crossfit_probabilities_persisted: bool = False
    task_embeddings_persisted: bool = False

    @model_validator(mode="after")
    def validate_invariants(self) -> SeedMetricReport:
        """Reject incomplete grids, forbidden evidence, and mixed seeds."""
        if self.protocol != SEED_PROTOCOL:
            raise ValueError("unexpected graded IRT seed protocol")
        if len(self.metrics) != EXPECTED_METRICS_PER_SEED:
            raise ValueError("graded IRT seed metric grid is incomplete")
        if {metric.seed for metric in self.metrics} != {self.seed}:
            raise ValueError("graded IRT seed report mixes seeds")
        keys = {
            metric.structure_key + "__" + metric.operating_key
            for metric in self.metrics
        }
        if len(keys) != len(self.metrics):
            raise ValueError("graded IRT seed report duplicates policies")
        if (
            self.target_outcomes_used
            or self.deep_swe_outcomes_accessed
            or self.confirmation_outcomes_accessed
            or self.fitted_coefficients_persisted
            or self.crossfit_probabilities_persisted
            or self.task_embeddings_persisted
        ):
            raise ValueError("graded IRT seed report violates the evidence boundary")
        if set(self.source_sha256) != set(SOURCE_NAMES):
            raise ValueError("graded IRT seed report has an incomplete source manifest")
        return self


class RouteLatencyRow(BaseModel):
    """One policy's measured single-core online route latency."""

    model_config = ConfigDict(extra="forbid")

    policy_key: str
    p50_ms: float = Field(ge=0.0)
    p95_ms: float = Field(ge=0.0)
    decisions: int = Field(ge=0)
    network_calls: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_quantiles(self) -> RouteLatencyRow:
        """Reject non-finite or reversed latency quantiles."""
        if not math.isfinite(self.p50_ms) or not math.isfinite(self.p95_ms):
            raise ValueError("graded IRT route latency must be finite")
        if self.p95_ms < self.p50_ms:
            raise ValueError("graded IRT route latency quantiles are reversed")
        return self

    def to_metric(self) -> RouteLatencyMetric:
        """Convert to the selector's immutable latency contract."""
        return RouteLatencyMetric(
            p50_ms=self.p50_ms,
            p95_ms=self.p95_ms,
            decisions=self.decisions,
            network_calls=self.network_calls,
        )


class RouteLatencyReport(BaseModel):
    """Complete latency evidence for every scientifically eligible policy."""

    model_config = ConfigDict(extra="forbid")

    protocol: str = LATENCY_PROTOCOL
    rows: list[RouteLatencyRow]
    source_sha256: dict[str, str]
    development_corpus_sha256: str = ""
    development_outcomes_sha256: str = ""
    development_audit_sha256: str = ""
    latency_corpus_sha256: str = ""
    latency_task_ids_sha256: str = ""
    single_core: bool = False
    decisions_per_policy: int = 0
    measurement_method: str = ""
    target_outcomes_used: bool = False
    deep_swe_outcomes_accessed: bool = False
    confirmation_outcomes_accessed: bool = False
    fitted_coefficients_persisted: bool = False
    task_embeddings_persisted: bool = False
    network_calls: int = 0

    @model_validator(mode="after")
    def validate_invariants(self) -> RouteLatencyReport:
        """Reject duplicated policies or forbidden persisted state."""
        if self.protocol != LATENCY_PROTOCOL:
            raise ValueError("unexpected graded IRT latency protocol")
        if len({row.policy_key for row in self.rows}) != len(self.rows):
            raise ValueError("graded IRT latency report duplicates policies")
        if self.fitted_coefficients_persisted or self.task_embeddings_persisted:
            raise ValueError("graded IRT latency report persisted fitted state")
        if self.network_calls != 0 or any(row.network_calls != 0 for row in self.rows):
            raise ValueError("graded IRT latency report made an online network call")
        if set(self.source_sha256) != set(SOURCE_NAMES):
            raise ValueError("graded IRT latency report has an incomplete source manifest")
        if (
            not self.single_core
            or self.decisions_per_policy < LATENCY_DECISIONS
            or not self.measurement_method
        ):
            raise ValueError("graded IRT latency report lacks the frozen measurement contract")
        if (
            self.target_outcomes_used
            or self.deep_swe_outcomes_accessed
            or self.confirmation_outcomes_accessed
        ):
            raise ValueError("graded IRT latency report crossed an outcome boundary")
        if any(row.decisions != self.decisions_per_policy for row in self.rows):
            raise ValueError("graded IRT latency rows use inconsistent decision counts")
        return self


class IrtData(BaseModel):
    """Dense external count matrix retained only in one remote worker process."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_ids: list[str]
    repositories: np.ndarray
    prompts: list[str]
    passed: np.ndarray
    total: np.ndarray
    costs: np.ndarray
    corpus_sha256: str
    outcomes_sha256: str
    audit_sha256: str


def _task_ids_sha256(task_ids: Sequence[str]) -> str:
    """Return a stable digest of the ordered retained task identities."""
    payload = "".join(task_id + "\n" for task_id in task_ids).encode()
    return hashlib.sha256(payload).hexdigest()


def _source_sha256() -> dict[str, str]:
    """Hash every source file needed to reproduce a seed shard."""
    root = Path(__file__).resolve().parent
    return {name: _sha256(root / name) for name in SOURCE_NAMES}


def load_irt_data(corpus_path: Path, outcomes_path: Path, audit_path: Path) -> IrtData:
    """Load the audited dense external matrix as exact fail-to-pass counts."""
    base: Data = load_data(corpus_path, outcomes_path, audit_path)
    raw_tasks = _read_object(corpus_path).get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("graded IRT corpus lacks task rows")
    task_by_id = {
        str(task.get("task_id")): task
        for task in raw_tasks
        if isinstance(task, dict) and isinstance(task.get("task_id"), str)
    }
    prompts = []
    for task_id in base.task_ids:
        task = task_by_id.get(task_id)
        prompt = task.get("prompt") if isinstance(task, dict) else None
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"graded IRT task lacks a prompt: {task_id}")
        prompts.append(prompt)

    task_index = {task_id: index for index, task_id in enumerate(base.task_ids)}
    arm_index = {arm: index for index, arm in enumerate(ARMS)}
    passed = np.full(base.rewards.shape, np.nan, dtype=np.float64)
    total = np.full_like(passed, np.nan)
    observed: set[tuple[str, str]] = set()
    for row in _rows(outcomes_path):
        task_id = row.get("task_id")
        arm = row.get("arm")
        identity = (str(task_id), str(arm))
        f2p_passed = row.get("f2p_passed")
        f2p_total = row.get("f2p_total")
        if (
            not isinstance(task_id, str)
            or task_id not in task_index
            or not isinstance(arm, str)
            or arm not in arm_index
            or identity in observed
            or isinstance(f2p_passed, bool)
            or not isinstance(f2p_passed, int)
            or isinstance(f2p_total, bool)
            or not isinstance(f2p_total, int)
            or f2p_total <= 0
            or not 0 <= f2p_passed <= f2p_total
        ):
            raise ValueError(f"invalid graded IRT count row: {identity}")
        index = task_index[task_id]
        arm_value = arm_index[arm]
        if abs(base.rewards[index, arm_value] - f2p_passed / f2p_total) > 1e-9:
            raise ValueError(f"graded IRT count and reward disagree: {identity}")
        passed[index, arm_value] = f2p_passed
        total[index, arm_value] = f2p_total
        observed.add(identity)
    if (
        len(observed) != passed.size
        or not np.isfinite(passed).all()
        or not np.isfinite(total).all()
    ):
        raise ValueError("graded IRT count matrix is not dense")
    if not np.all(total == total[:, :1]):
        raise ValueError("graded IRT fail-to-pass denominator differs across arms")
    return IrtData(
        task_ids=base.task_ids,
        repositories=np.asarray(base.repositories, dtype=object),
        prompts=prompts,
        passed=passed,
        total=total,
        costs=base.costs,
        corpus_sha256=_sha256(corpus_path),
        outcomes_sha256=_sha256(outcomes_path),
        audit_sha256=_sha256(audit_path),
    )


def evaluate_seed_report(data: IrtData, *, seed: int) -> SeedMetricReport:
    """Run one complete seed shard and retain aggregate scalar metrics only."""
    if seed not in OUTER_SEEDS:
        raise ValueError(f"unsupported frozen graded IRT seed: {seed}")
    metrics = evaluate_nested_seed(
        frozen_feature_views(data.prompts),
        data.passed,
        data.total,
        data.costs,
        data.repositories,
        seed=seed,
    )
    return SeedMetricReport(
        seed=seed,
        tasks=len(data.task_ids),
        metrics=[PolicyMetricRow.from_metric(metric) for metric in metrics],
        corpus_sha256=data.corpus_sha256,
        outcomes_sha256=data.outcomes_sha256,
        audit_sha256=data.audit_sha256,
        task_ids_sha256=_task_ids_sha256(data.task_ids),
        source_sha256=_source_sha256(),
    )


def _scientifically_eligible_keys(
    metrics: Sequence[PolicyMetric],
    *,
    seeds: Sequence[int],
) -> set[str]:
    """Return policies that pass every non-latency gate in every seed."""
    expected_seeds = set(seeds)
    by_key: dict[str, list[PolicyMetric]] = {}
    for metric in metrics:
        by_key.setdefault(metric.key, []).append(metric)
    return {
        key
        for key, values in by_key.items()
        if len(values) == len(expected_seeds)
        and {value.seed for value in values} == expected_seeds
        and all(value.eligible for value in values)
    }


def measure_latency_report(
    data: IrtData,
    latency_tasks: Sequence[dict[str, object]],
    seed_reports: Sequence[SeedMetricReport],
    *,
    latency_corpus_sha256: str,
    single_core: bool,
) -> RouteLatencyReport:
    """Fit eligible structures ephemerally and time exact online route decisions."""
    if not single_core:
        raise ValueError("graded IRT latency must run with one CPU in its affinity mask")
    prompts: list[str] = []
    task_ids: list[str] = []
    for task in latency_tasks:
        prompt = task.get("prompt")
        task_id = task.get("task_id")
        if not isinstance(prompt, str) or not prompt or not isinstance(task_id, str) or not task_id:
            raise ValueError("graded IRT latency corpus has an invalid label-free task")
        prompts.append(prompt)
        task_ids.append(task_id)
    if not prompts:
        raise ValueError("graded IRT latency corpus is empty")
    first = seed_reports[0]
    if (
        any(report.source_sha256 != first.source_sha256 for report in seed_reports)
        or first.source_sha256 != _source_sha256()
        or first.corpus_sha256 != data.corpus_sha256
        or first.outcomes_sha256 != data.outcomes_sha256
        or first.audit_sha256 != data.audit_sha256
    ):
        raise ValueError("graded IRT latency inputs differ from the seed workers")
    metrics = tuple(metric.to_metric() for report in seed_reports for metric in report.metrics)
    scientific_keys = _scientifically_eligible_keys(metrics, seeds=OUTER_SEEDS)
    structure_by_key = {structure.key: structure for structure in frozen_structures()}
    point_by_key = {point.key: point for point in frozen_operating_points()}
    policies: dict[str, list[tuple[str, OperatingPoint]]] = {}
    for policy_key in sorted(scientific_keys):
        structure_key, operating_key = policy_key.split("__", 1)
        if structure_key not in structure_by_key or operating_key not in point_by_key:
            raise ValueError(f"graded IRT eligible policy is unknown: {policy_key}")
        policies.setdefault(structure_key, []).append((policy_key, point_by_key[operating_key]))

    rewards = data.passed / data.total
    guard_arm = _best_static_arm(rewards, data.costs)
    mean_costs = np.mean(data.costs, axis=0)
    decision_costs = np.repeat(mean_costs[None, :], LATENCY_DECISIONS, axis=0)
    rows: list[RouteLatencyRow] = []
    route_prompts = [prompts[index % len(prompts)] for index in range(LATENCY_DECISIONS)]
    for structure_key in sorted(policies):
        structure = structure_by_key[structure_key]
        development_features = frozen_feature_view(
            data.prompts,
            name=structure.feature_name,
        )
        graph_laplacian = None
        if structure.graph_l2 > 0.0:
            graph_laplacian = cosine_knn_laplacian(development_features, neighbors=8)
        fit = fit_projected_binomial_irt(
            development_features,
            data.passed,
            data.total,
            structure.latent_dimension,
            projection_l2=structure.regularization,
            monotone_luna=structure.monotone_luna,
            graph_laplacian=graph_laplacian,
            graph_l2=structure.graph_l2,
        )
        probabilities = np.empty((LATENCY_DECISIONS, len(ARMS)), dtype=np.float64)
        base_nanoseconds = np.empty(LATENCY_DECISIONS, dtype=np.float64)
        for index, prompt in enumerate(route_prompts):
            started = time.perf_counter_ns()
            target_features = frozen_feature_view((prompt,), name=structure.feature_name)
            probabilities[index] = predict_projected_probabilities(fit, target_features)[0]
            base_nanoseconds[index] = time.perf_counter_ns() - started
        for policy_key, point in policies[structure_key]:
            choice_nanoseconds = np.empty(LATENCY_DECISIONS, dtype=np.float64)
            for index in range(LATENCY_DECISIONS):
                started = time.perf_counter_ns()
                quality_guarded_choices(
                    probabilities[index : index + 1],
                    decision_costs[index : index + 1],
                    guard_arm=guard_arm,
                    cost_penalty=point.cost_penalty,
                )
                choice_nanoseconds[index] = time.perf_counter_ns() - started
            milliseconds = (base_nanoseconds + choice_nanoseconds) / 1_000_000.0
            rows.append(
                RouteLatencyRow(
                    policy_key=policy_key,
                    p50_ms=float(np.percentile(milliseconds, 50)),
                    p95_ms=float(np.percentile(milliseconds, 95)),
                    decisions=LATENCY_DECISIONS,
                    network_calls=0,
                )
            )
        del fit, probabilities, base_nanoseconds
    return RouteLatencyReport(
        rows=rows,
        source_sha256=_source_sha256(),
        development_corpus_sha256=data.corpus_sha256,
        development_outcomes_sha256=data.outcomes_sha256,
        development_audit_sha256=data.audit_sha256,
        latency_corpus_sha256=latency_corpus_sha256,
        latency_task_ids_sha256=_task_ids_sha256(task_ids),
        single_core=True,
        decisions_per_policy=LATENCY_DECISIONS,
        measurement_method=(
            "shared per-structure prompt transform and probability samples plus exact "
            "per-policy guarded-choice samples on one CPU"
        ),
    )


def select_reports(
    seed_reports: Sequence[SeedMetricReport],
    latency_report: RouteLatencyReport,
    *,
    structures: Sequence[IrtStructure] | None = None,
    operating_points: Sequence[OperatingPoint] | None = None,
    seeds: Sequence[int] = OUTER_SEEDS,
) -> tuple[NestedSelectionResult, set[str]]:
    """Validate complete aggregate shards and apply the latency-gated selector."""
    selected_seeds = tuple(int(seed) for seed in seeds)
    if len(seed_reports) != len(selected_seeds) or {row.seed for row in seed_reports} != set(
        selected_seeds
    ):
        raise ValueError("graded IRT seed report set is incomplete or duplicated")
    first = seed_reports[0]
    shared_fields = (
        "tasks",
        "corpus_sha256",
        "outcomes_sha256",
        "audit_sha256",
        "task_ids_sha256",
        "source_sha256",
    )
    if any(
        any(getattr(report, field) != getattr(first, field) for field in shared_fields)
        for report in seed_reports[1:]
    ):
        raise ValueError("graded IRT seed reports do not share identical inputs and sources")
    if latency_report.source_sha256 != first.source_sha256:
        raise ValueError("graded IRT latency and seed reports use different sources")
    metrics = tuple(metric.to_metric() for report in seed_reports for metric in report.metrics)
    scientific_keys = _scientifically_eligible_keys(metrics, seeds=selected_seeds)
    latency = {row.policy_key: row.to_metric() for row in latency_report.rows}
    if set(latency) != scientific_keys:
        raise ValueError("graded IRT latency audit does not exactly cover eligible policies")
    selected_structures = tuple(structures) if structures is not None else frozen_structures()
    selected_points = (
        tuple(operating_points) if operating_points is not None else frozen_operating_points()
    )
    result = select_nested_metrics(
        metrics,
        structures=selected_structures,
        operating_points=selected_points,
        seeds=selected_seeds,
        route_latency=latency,
    )
    return result, scientific_keys


def _write_model(path: Path, model: BaseModel) -> None:
    """Atomically write one validated compact JSON artifact."""
    if path.exists():
        raise FileExistsError(path)
    write_text_atomic(path, model.model_dump_json(indent=2) + "\n")


def _run_seed(args: argparse.Namespace) -> int:
    data = load_irt_data(args.corpus, args.outcomes, args.audit)
    report = evaluate_seed_report(data, seed=args.seed)
    _write_model(args.report_out, report)
    logger.info("wrote graded IRT seed=%d metrics=%d", report.seed, len(report.metrics))
    return 0


def _run_select(args: argparse.Namespace) -> int:
    seed_reports = [
        SeedMetricReport.model_validate_json(path.read_text(encoding="utf-8"))
        for path in args.seed_reports
    ]
    latency = RouteLatencyReport.model_validate_json(
        args.latency_report.read_text(encoding="utf-8")
    )
    if seed_reports[0].source_sha256 != _source_sha256():
        raise ValueError("graded IRT selector source differs from the seed workers")
    result, scientific_keys = select_reports(seed_reports, latency)
    selected_metrics = [
        PolicyMetricRow.from_metric(metric)
        for metric in result.metrics
        if metric.key == result.selected_key
    ]
    payload = {
        "protocol": SELECTION_PROTOCOL,
        "valid": True,
        "development_passed": result.selected_key is not None,
        "selected_key": result.selected_key,
        "scientifically_eligible_policies": len(scientific_keys),
        "latency_audited_policies": len(latency.rows),
        "selected_seed_metrics": [row.model_dump(mode="json") for row in selected_metrics],
        "seed_report_sha256": {
            str(path): _sha256(path) for path in args.seed_reports
        },
        "latency_report_sha256": _sha256(args.latency_report),
        "source_sha256": _source_sha256(),
        "target_outcomes_used": False,
        "deep_swe_outcomes_accessed": False,
        "confirmation_outcomes_accessed": False,
        "fitted_coefficients_persisted": False,
        "crossfit_probabilities_persisted": False,
        "task_embeddings_persisted": False,
    }
    if args.report_out.exists():
        raise FileExistsError(args.report_out)
    write_text_atomic(args.report_out, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    logger.info(
        "graded IRT selection eligible=%d selected=%s",
        len(scientific_keys),
        result.selected_key,
    )
    return 0


def _run_latency(args: argparse.Namespace) -> int:
    data = load_irt_data(args.corpus, args.outcomes, args.audit)
    seed_reports = [
        SeedMetricReport.model_validate_json(path.read_text(encoding="utf-8"))
        for path in args.seed_reports
    ]
    latency_tasks = load_confirmation(args.latency_corpus)
    affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else set()
    report = measure_latency_report(
        data,
        latency_tasks,
        seed_reports,
        latency_corpus_sha256=_sha256(args.latency_corpus),
        single_core=len(affinity) == 1,
    )
    _write_model(args.report_out, report)
    logger.info("wrote graded IRT latency policies=%d", len(report.rows))
    return 0


def parse_args() -> argparse.Namespace:
    """Parse the remote seed and aggregate-selection commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    seed = subparsers.add_parser("seed")
    seed.add_argument("--corpus", type=Path, required=True)
    seed.add_argument("--outcomes", type=Path, required=True)
    seed.add_argument("--audit", type=Path, required=True)
    seed.add_argument("--seed", type=int, choices=OUTER_SEEDS, required=True)
    seed.add_argument("--report-out", type=Path, required=True)
    seed.set_defaults(run=_run_seed)

    latency = subparsers.add_parser("latency")
    latency.add_argument("--corpus", type=Path, required=True)
    latency.add_argument("--outcomes", type=Path, required=True)
    latency.add_argument("--audit", type=Path, required=True)
    latency.add_argument("--latency-corpus", type=Path, required=True)
    latency.add_argument("--seed-reports", type=Path, nargs=5, required=True)
    latency.add_argument("--report-out", type=Path, required=True)
    latency.set_defaults(run=_run_latency)

    select = subparsers.add_parser("select")
    select.add_argument("--seed-reports", type=Path, nargs=5, required=True)
    select.add_argument("--latency-report", type=Path, required=True)
    select.add_argument("--report-out", type=Path, required=True)
    select.set_defaults(run=_run_select)
    return parser.parse_args()


def main() -> int:
    """Run one remote-only graded IRT orchestration phase."""
    args = parse_args()
    return int(args.run(args))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    raise SystemExit(main())
