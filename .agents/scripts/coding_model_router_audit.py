"""Audit coding-router completion against the frozen project brief."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, JsonValue

from wmo.core.files import write_text_atomic

EXPERIMENT_ID = "coding-router-20260728"
SEEDS = tuple(range(5))
BENCHMARKS = ("terminal-bench-2", "swe-bench-verified")
SMOKE_TASKS = ("break-filter-js-from-html", "log-summary-date-ranges")
SMOKE_ARMS = ("oai-luna-high", "ant-haiku45")
FINAL_REPORT_ROWS = (
    "best single",
    "cheapest single",
    "unguarded",
    "guarded",
    "selected pareto",
    "oracle",
)

logger = logging.getLogger(__name__)

JsonObject = dict[str, JsonValue]
AuditStatus = Literal["passed", "failed", "blocked"]


class RequirementResult(BaseModel):
    """One completion requirement and its authoritative evidence."""

    requirement: str
    status: AuditStatus
    summary: str
    evidence: list[str]
    details: JsonObject = Field(default_factory=dict)


class CompletionAudit(BaseModel):
    """Machine-readable completion verdict for the coding-router experiment."""

    protocol: Literal["coding-router-completion-audit-v1"]
    experiment_id: str
    audited_at: str
    completion_status: Literal["complete", "incomplete"]
    ready_for_material_paid_execution: bool
    target_achieved: bool | None
    known_model_spend_usd: float
    estimated_model_spend_usd: float
    conservative_budget_debit_usd: float
    unknown_cost_events: int
    requirements: list[RequirementResult]
    blocking_requirements: list[str]


def _relative(path: Path, root: Path) -> str:
    """Return a stable root-relative evidence path."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a local artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> JsonObject | None:
    """Read a JSON object, returning None for missing or malformed artifacts."""
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    return {str(key): item for key, item in value.items()}


def _read_rows(path: Path) -> list[JsonObject] | None:
    """Read a JSONL ledger without accepting malformed or non-object rows."""
    if not path.is_file():
        return None
    rows: list[JsonObject] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                return None
            rows.append({str(key): item for key, item in value.items()})
    except (OSError, json.JSONDecodeError):
        return None
    return rows


def _number(value: JsonValue | None) -> float | None:
    """Return a finite JSON number without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if result == result and abs(result) != float("inf") else None


def _smoke_usage_complete(row: JsonObject) -> bool:
    """Check that one smoke row has aligned nonempty call-level usage."""
    seconds = row.get("call_seconds")
    input_tokens = row.get("call_input_tokens")
    output_tokens = row.get("call_output_tokens")
    return (
        isinstance(seconds, list)
        and bool(seconds)
        and isinstance(input_tokens, list)
        and len(seconds) == len(input_tokens)
        and isinstance(output_tokens, list)
        and len(seconds) == len(output_tokens)
        and _number(row.get("cost_usd")) is not None
    )


def _result(
    requirement: str,
    status: AuditStatus,
    summary: str,
    root: Path,
    paths: list[Path],
    **details: JsonValue,
) -> RequirementResult:
    """Build one normalized requirement result."""
    return RequirementResult(
        requirement=requirement,
        status=status,
        summary=summary,
        evidence=[_relative(path, root) for path in paths],
        details=details,
    )


def _frozen_protocol(root: Path) -> RequirementResult:
    summary_path = root / "freeze-summary.json"
    pool_path = root / "pool.toml"
    task_paths = [root / "tasks" / f"{benchmark}.json" for benchmark in BENCHMARKS]
    split_paths = [root / "splits" / f"seed-{seed}.json" for seed in SEEDS]
    paths = [summary_path, pool_path, *task_paths, *split_paths]
    summary = _read_object(summary_path)
    if summary is None or any(not path.is_file() for path in paths):
        return _result(
            "frozen_protocol",
            "failed",
            "Frozen source, model, task, or split artifacts are missing or malformed.",
            root,
            paths,
        )
    digest_fields = {
        pool_path: summary.get("pool_sha256"),
        task_paths[0]: summary.get("terminal_manifest_sha256"),
        task_paths[1]: summary.get("swe_manifest_sha256"),
    }
    split_digests = summary.get("split_sha256")
    if not isinstance(split_digests, dict):
        split_digests = {}
    digest_fields.update({path: split_digests.get(path.name) for path in split_paths})
    digests_match = all(
        isinstance(expected, str) and _sha256(path) == expected
        for path, expected in digest_fields.items()
    )
    valid = (
        summary.get("experiment_id") == EXPERIMENT_ID
        and isinstance(summary.get("source_commit"), str)
        and bool(summary.get("source_commit"))
        and (_number(summary.get("model_arms")) or 0) > 0
        and digests_match
    )
    return _result(
        "frozen_protocol",
        "passed" if valid else "failed",
        (
            "Source commit, roster, task manifests, and five splits are frozen "
            "with matching digests."
            if valid
            else "Frozen artifacts do not match the recorded experiment identity or digests."
        ),
        root,
        paths,
        digests_match=digests_match,
        source_commit=summary.get("source_commit"),
    )


def _valid_smoke(root: Path) -> RequirementResult:
    smoke = root / "smoke"
    paths = [
        smoke / "outcomes.json",
        smoke / "smoke-report.json",
        smoke / "resume-proof.json",
        smoke / "policy" / "policy.json",
    ]
    invalidated = _read_object(smoke / "invalidated.json")
    outcomes = _read_object(paths[0])
    report = _read_object(paths[1])
    resume = _read_object(paths[2])
    raw_rows = outcomes.get("outcomes") if outcomes is not None else None
    rows = raw_rows if isinstance(raw_rows, list) else []
    gradeable: list[JsonObject] = [
        {str(key): item for key, item in row.items()}
        for row in rows
        if isinstance(row, dict) and _number(row.get("reward")) is not None
    ]
    expected = {(f"terminal-bench-2:{task}", arm) for task in SMOKE_TASKS for arm in SMOKE_ARMS}
    observed = {(row.get("scenario_id"), row.get("model")) for row in gradeable}
    per_call_complete = all(_smoke_usage_complete(row) for row in gradeable)
    resumed_cells = resume.get("resumed_cells") if resume is not None else None
    resumed = (
        resume is not None
        and resume.get("unchanged") is True
        and isinstance(resumed_cells, int)
        and not isinstance(resumed_cells, bool)
        and resumed_cells >= 1
    )
    valid = (
        invalidated is None
        and all(path.is_file() for path in paths)
        and len(gradeable) == 4
        and observed == expected
        and per_call_complete
        and report is not None
        and report.get("gradeable_cells") == 4
        and resumed
    )
    evidence = [*paths]
    if invalidated is not None:
        evidence.append(smoke / "invalidated.json")
    reason = (
        "The integrated four-cell smoke, fit/replay, and interrupted resume passed."
        if valid
        else (
            str(invalidated.get("reason"))
            if invalidated is not None and isinstance(invalidated.get("reason"), str)
            else "The required four-cell smoke evidence is absent, incomplete, or invalid."
        )
    )
    return _result(
        "valid_integrated_smoke",
        "passed" if valid else "blocked",
        reason,
        root,
        evidence,
        gradeable_cells=len(gradeable),
        invalidated=invalidated is not None,
        resume_proved=resumed,
    )


def _authorized_ceiling(root: Path) -> RequirementResult:
    path = root / "freeze-summary.json"
    summary = _read_object(path)
    ceiling = _number(summary.get("spend_ceiling_usd")) if summary is not None else None
    valid = ceiling is not None and ceiling > 0
    return _result(
        "authorized_material_spend_ceiling",
        "passed" if valid else "blocked",
        (
            f"A positive material spend ceiling of ${ceiling:.2f} is frozen."
            if valid
            else "No positive material paid sweep ceiling is authorized."
        ),
        root,
        [path],
        spend_ceiling_usd=ceiling,
    )


def _dense_matrix(root: Path) -> RequirementResult:
    paths = [
        root / "full" / "outcomes.json",
        root / "full" / "summary.json",
        root / "analysis" / "matrix-validation.json",
    ]
    outcomes = _read_object(paths[0])
    summary = _read_object(paths[1])
    validation = _read_object(paths[2])
    raw_rows = outcomes.get("outcomes") if outcomes is not None else None
    rows = raw_rows if isinstance(raw_rows, list) else []
    gradeable = [
        row for row in rows if isinstance(row, dict) and _number(row.get("reward")) is not None
    ]
    expected = _number(summary.get("cells_expected")) if summary is not None else None
    valid = (
        all(path.is_file() for path in paths)
        and summary is not None
        and summary.get("stage") == "full"
        and expected is not None
        and expected > 0
        and len(gradeable) == int(expected)
        and _number(summary.get("gradeable_cells")) == expected
        and validation is not None
        and validation.get("status") == "complete"
        and _number(validation.get("gradeable_cells")) == expected
    )
    return _result(
        "real_execution_scored_matrix",
        "passed" if valid else "blocked",
        (
            f"The dense real matrix is complete with {int(expected)} gradeable cells."
            if valid and expected is not None
            else "The full execution-scored matrix has not been completed and validated."
        ),
        root,
        paths,
        expected_cells=expected,
        observed_gradeable_cells=len(gradeable),
    )


def _ledger(root: Path) -> tuple[RequirementResult, float, float, float, int]:
    path = root / "spend-ledger.jsonl"
    rows = _read_rows(path)
    if rows is None:
        return (
            _result(
                "spend_ledger",
                "failed",
                "The spend ledger is missing or malformed.",
                root,
                [path],
            ),
            0.0,
            0.0,
            0.0,
            0,
        )
    known = 0.0
    estimated = 0.0
    debit = 0.0
    unknown = 0
    unreconciled = 0
    reserved = 0
    for row in rows:
        if row.get("status") == "reserved":
            reserved += 1
            continue
        cost = _number(row.get("model_cost_usd"))
        if cost is None:
            unknown += 1
            budget_debit = _number(row.get("budget_debit_usd"))
            if budget_debit is None or budget_debit <= 0:
                unreconciled += 1
            else:
                debit += budget_debit
        else:
            if row.get("model_cost_accounting_status") == "estimated_from_trace":
                estimated += cost
            else:
                known += cost
    valid = unreconciled == 0 and reserved == 0
    return (
        _result(
            "spend_ledger",
            "passed" if valid else "blocked",
            (
                (
                    f"All {len(rows)} ledger events are exact, trace-estimated, or carry "
                    "an explicit conservative ceiling debit, with no open reservation."
                )
                if valid
                else (
                    f"The ledger has {unreconciled} unreconciled unknown-cost events and "
                    f"{reserved} open reservations."
                )
            ),
            root,
            [path],
            events=len(rows),
            known_model_spend_usd=known,
            estimated_model_spend_usd=estimated,
            conservative_budget_debit_usd=debit,
            unknown_cost_events=unknown,
            unreconciled_unknown_cost_events=unreconciled,
            open_reservations=reserved,
        ),
        known,
        estimated,
        debit,
        unknown,
    )


def _five_seed_evaluation(root: Path) -> RequirementResult:
    paths = [
        root / "analysis" / "selection-lock.json",
        root / "analysis" / "outer-results.json",
        root / "analysis" / "evaluation-complete.json",
    ]
    lock = _read_object(paths[0])
    outer = _read_object(paths[1])
    complete = _read_object(paths[2])
    lock_seeds = lock.get("seeds") if lock is not None else None
    outer_seeds = outer.get("seeds") if outer is not None else None
    observed_lock = (
        {row.get("seed") for row in lock_seeds if isinstance(row, dict)}
        if isinstance(lock_seeds, list)
        else set()
    )
    observed_outer = (
        {row.get("seed") for row in outer_seeds if isinstance(row, dict)}
        if isinstance(outer_seeds, list)
        else set()
    )
    digests_match = (
        (
            complete is not None
            and complete.get("heldout_evaluated") is True
            and complete.get("selection_lock_sha256") == _sha256(paths[0])
            and complete.get("outer_results_sha256") == _sha256(paths[1])
        )
        if paths[0].is_file() and paths[1].is_file()
        else False
    )
    valid = observed_lock == set(SEEDS) and observed_outer == set(SEEDS) and digests_match
    return _result(
        "five_seed_fit_locked_evaluation",
        "passed" if valid else "blocked",
        (
            "Fit-only selection and one-time heldout evaluation are complete across five seeds."
            if valid
            else "Five-seed fit-only selection and heldout evaluation are incomplete."
        ),
        root,
        paths,
        selection_seeds=sorted(seed for seed in observed_lock if isinstance(seed, int)),
        evaluation_seeds=sorted(seed for seed in observed_outer if isinstance(seed, int)),
        completion_digests_match=digests_match,
    )


def _pareto_conclusion(root: Path) -> tuple[RequirementResult, bool | None]:
    path = root / "analysis" / "outer-results.json"
    outer = _read_object(path)
    pareto = outer.get("pareto") if outer is not None else None
    promoted = outer.get("promoted") if outer is not None else None
    gates = outer.get("all_seed_promotion_gates") if outer is not None else None
    bootstrap = outer.get("paired_cluster_bootstrap") if outer is not None else None
    capability_slices = outer.get("capability_slices") if outer is not None else None
    ablations = outer.get("one_at_a_time_ablations") if outer is not None else None
    target = promoted if isinstance(promoted, bool) else None
    valid = (
        isinstance(pareto, list)
        and bool(pareto)
        and isinstance(gates, list)
        and len(gates) == len(SEEDS)
        and all(isinstance(value, bool) for value in gates)
        and isinstance(bootstrap, dict)
        and _number(bootstrap.get("retention_lower_95")) is not None
        and isinstance(capability_slices, dict)
        and bool(capability_slices)
        and isinstance(ablations, dict)
        and ablations.get("benchmark_stratified") == "ablation:benchmark_stratified"
        and ablations.get("missing_fit_coverage_0.8") == "ablation:missing_fit_coverage_0.8"
        and ablations.get("latency_only_static") == "latency_only"
        and target is not None
    )
    return (
        _result(
            "pareto_and_scientific_conclusion",
            "passed" if valid else "blocked",
            (
                "The measured frontier establishes that the 95 percent quality and 40 percent "
                f"savings target was {'achieved' if target else 'not achieved'}."
                if valid
                else "The Pareto frontier and promotion verdict are not complete."
            ),
            root,
            [path],
            target_achieved=target,
            pareto_points=len(pareto) if isinstance(pareto, list) else 0,
            capability_slices=(
                len(capability_slices) if isinstance(capability_slices, dict) else 0
            ),
        ),
        target,
    )


def _deployable_policy(root: Path) -> RequirementResult:
    policy_path = root / "analysis" / "deployable" / "policy.json"
    policy = _read_object(policy_path)
    bank_name = policy.get("knn_bank_path") if policy is not None else None
    bank_path = (
        policy_path.parent / bank_name
        if isinstance(bank_name, str) and not Path(bank_name).is_absolute()
        else Path(bank_name)
        if isinstance(bank_name, str)
        else policy_path.parent / "missing"
    )
    valid = (
        policy is not None
        and policy.get("kind") == "knn"
        and isinstance(bank_name, str)
        and bank_path.is_file()
    )
    return _result(
        "deployable_wmo_policy",
        "passed" if valid else "blocked",
        (
            "A native WMO guarded kNN policy and portable evidence bank are present."
            if valid
            else "The selected native WMO policy or its evidence bank is missing."
        ),
        root,
        [policy_path, bank_path],
        policy_kind=policy.get("kind") if policy is not None else None,
    )


def _serving(root: Path) -> RequirementResult:
    directory = root / "serving"
    results = sorted(directory.glob("result-*.json"))
    passed = [
        path
        for path in results
        if (
            (result := _read_object(path)) is not None
            and result.get("completion_status") == "passed"
            and _number(result.get("requests")) == 8
            and result.get("fallback_gate") == "novelty-abstain"
            and result.get("affinity_reason") == "sticky: conversation affinity"
            and (_number(result.get("cache_aware_credit_usd")) or 0) > 0
        )
    ]
    prepare = directory / "prepare.json"
    valid = prepare.is_file() and bool(passed)
    evidence = [prepare, *(passed[-1:] if passed else results[-1:])]
    return _result(
        "real_wmo_serving_verification",
        "passed" if valid else "blocked",
        (
            "The selected policy passed the bounded real WMO serving check."
            if valid
            else "No passed real WMO serving verification is present."
        ),
        root,
        evidence,
        passed_attempts=len(passed),
        total_attempts=len(results),
    )


def _world_model(root: Path) -> RequirementResult:
    paths = [
        root / "world-model" / "prepare.json",
        root / "world-model" / "build-usage.json",
        root / "world-model" / "simulated" / "completion.json",
        root / "world-model" / "simulated" / "outcomes.json",
        root / "world-model" / "comparison.json",
    ]
    completion = _read_object(paths[2])
    comparison = _read_object(paths[4])
    required_comparison = (
        "binary_agreement",
        "false_positive_rate",
        "false_negative_rate",
        "calibration",
        "candidate_rank_spearman",
        "best_single_agreement",
        "selected_model_agreement",
        "guard_decision_agreement",
        "promotion_agreement",
    )
    valid = (
        all(path.is_file() for path in paths)
        and completion is not None
        and completion.get("complete") is True
        and comparison is not None
        and comparison.get("protocol") == "coding-world-model-compare-v1"
        and all(key in comparison for key in required_comparison)
    )
    return _result(
        "world_model_real_comparison",
        "passed" if valid else "blocked",
        (
            "The separate simulated matrix and real-versus-simulated deployment "
            "comparison are complete."
            if valid
            else "The world-model simulation and deployment-decision comparison are incomplete."
        ),
        root,
        paths,
    )


def _final_report(root: Path) -> RequirementResult:
    path = root / "final-report.md"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    normalized = text.casefold()
    missing_rows = [row for row in FINAL_REPORT_ROWS if row not in normalized]
    has_table = all(
        heading in normalized
        for heading in (
            "| policy |",
            "quality retained",
            "cost savings",
            "latency p50/p95",
            "verdict",
        )
    )
    has_target = (
        "target was achieved" in normalized
        or "target was not achieved" in normalized
        or "target was not reached" in normalized
    )
    valid = has_table and not missing_rows and has_target
    return _result(
        "concise_traceable_final_report",
        "passed" if valid else "blocked",
        (
            "The required policy table and plain-language scientific conclusion are present."
            if valid
            else (
                "The required final report is missing or does not contain every "
                "mandated policy row."
            )
        ),
        root,
        [path],
        missing_policy_rows=missing_rows,
        has_target_statement=has_target,
    )


def audit(root: Path) -> CompletionAudit:
    """Evaluate every terminal condition without making network or provider calls."""
    ledger, known_spend, estimated_spend, budget_debit, unknown_costs = _ledger(root)
    pareto, target = _pareto_conclusion(root)
    requirements = [
        _frozen_protocol(root),
        _valid_smoke(root),
        _authorized_ceiling(root),
        _dense_matrix(root),
        ledger,
        _five_seed_evaluation(root),
        pareto,
        _deployable_policy(root),
        _serving(root),
        _world_model(root),
        _final_report(root),
    ]
    blocking = [
        requirement.requirement for requirement in requirements if requirement.status != "passed"
    ]
    smoke_passed = next(
        item.status == "passed"
        for item in requirements
        if item.requirement == "valid_integrated_smoke"
    )
    ceiling_passed = next(
        item.status == "passed"
        for item in requirements
        if item.requirement == "authorized_material_spend_ceiling"
    )
    return CompletionAudit(
        protocol="coding-router-completion-audit-v1",
        experiment_id=EXPERIMENT_ID,
        audited_at=datetime.now(UTC).isoformat(),
        completion_status="complete" if not blocking else "incomplete",
        ready_for_material_paid_execution=smoke_passed
        and ceiling_passed
        and ledger.status == "passed",
        target_achieved=target,
        known_model_spend_usd=known_spend,
        estimated_model_spend_usd=estimated_spend,
        conservative_budget_debit_usd=budget_debit,
        unknown_cost_events=unknown_costs,
        requirements=requirements,
        blocking_requirements=blocking,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(".wmo") / "experiments" / EXPERIMENT_ID,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Write the current completion audit and optionally enforce completeness."""
    args = _parse_args()
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root / "completion-audit.json"
    result = audit(root)
    write_text_atomic(
        output,
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )
    logger.info(
        "completion audit: %s, %d blockers, $%.6f exact, $%.6f estimated, $%.2f conservative debit",
        result.completion_status,
        len(result.blocking_requirements),
        result.known_model_spend_usd,
        result.estimated_model_spend_usd,
        result.conservative_budget_debit_usd,
    )
    if args.require_complete and result.completion_status != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
