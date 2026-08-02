"""Render the coding-router final report from immutable experiment evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import statistics
from pathlib import Path
from typing import cast

from pydantic import JsonValue

from wmo.core.files import write_text_atomic

EXPERIMENT_ID = "coding-router-20260728"
SEEDS = tuple(range(5))
BENCHMARKS = ("terminal-bench-2", "swe-bench-verified")
QUALITY_RETENTION_GATE = 0.95
SAVINGS_GATE = 0.40

JsonObject = dict[str, JsonValue]
logger = logging.getLogger(__name__)


def _read_object(path: Path) -> JsonObject:
    """Read one required JSON object."""
    if not path.is_file():
        raise ValueError(f"required evidence is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"required evidence is malformed: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"required evidence is not a JSON object: {path}")
    return {str(key): item for key, item in value.items()}


def _object(value: JsonValue | None, *, label: str) -> JsonObject:
    """Require one nested JSON object."""
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return {str(key): item for key, item in value.items()}


def _object_rows(value: JsonValue | None, *, label: str) -> list[JsonObject]:
    """Require a nonempty list of JSON objects."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list")
    rows: list[JsonObject] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{label}[{index}] must be an object")
        rows.append({str(key): child for key, child in item.items()})
    return rows


def _number(value: JsonValue | None, *, label: str) -> float:
    """Require one finite JSON number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of an evidence artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _point(seed: JsonObject, point_id: str) -> JsonObject:
    """Return one named policy metric from a seed result."""
    points = _object(seed.get("points"), label="seed.points")
    return _object(points.get(point_id), label=f"seed.points.{point_id}")


def _mean_mapping(
    metrics: list[JsonObject],
    field: str,
) -> JsonObject:
    """Average a numeric mapping across seed metrics."""
    mappings = [_object(metric.get(field), label=field) for metric in metrics]
    keys = sorted(set().union(*(mapping.keys() for mapping in mappings)))
    return {
        key: statistics.fmean(
            _number(mapping.get(key, 0.0), label=f"{field}.{key}") for mapping in mappings
        )
        for key in keys
    }


def _mean_benchmark_mapping(metrics: list[JsonObject]) -> JsonObject:
    """Average preregistered per-benchmark metrics across seeds."""
    result: JsonObject = {}
    for benchmark in BENCHMARKS:
        rows = [
            _object(
                _object(metric.get("per_benchmark"), label="per_benchmark").get(benchmark),
                label=f"per_benchmark.{benchmark}",
            )
            for metric in metrics
        ]
        result[benchmark] = {
            field: statistics.fmean(
                _number(row.get(field), label=f"{benchmark}.{field}") for row in rows
            )
            for field in ("quality", "cost_per_task", "success_rate", "scenarios")
        }
    return result


def _aggregate_point(seeds: list[JsonObject], point_id: str) -> JsonObject:
    """Average one policy point over the five paired outer splits."""
    metrics = [_point(seed, point_id) for seed in seeds]
    numeric_fields = (
        "quality",
        "cost_per_task",
        "total_cost",
        "success_rate",
        "latency_p50_s",
        "latency_p95_s",
        "route_away_rate",
        "guard_reversion_rate",
        "novelty_abstention_rate",
        "scenarios",
    )
    aggregate: JsonObject = {
        field: statistics.fmean(
            _number(metric.get(field), label=f"{point_id}.{field}") for metric in metrics
        )
        for field in numeric_fields
    }
    effective = [metric.get("effective_cost_per_success") for metric in metrics]
    aggregate["effective_cost_per_success"] = (
        statistics.fmean(
            _number(value, label=f"{point_id}.effective_cost_per_success") for value in effective
        )
        if all(value is not None for value in effective)
        else None
    )
    aggregate["model_mix"] = _mean_mapping(metrics, "model_mix")
    aggregate["per_benchmark"] = _mean_benchmark_mapping(metrics)
    return aggregate


def _percent(value: float) -> str:
    """Format one ratio as a percentage."""
    return f"{value * 100:.1f}%"


def _cost(value: float) -> str:
    """Format one per-task model cost."""
    return f"${value:.4f}"


def _mix_text(mix: JsonObject) -> str:
    """Render a descending model mix."""
    rows = sorted(
        ((model, _number(share, label=f"model_mix.{model}")) for model, share in mix.items()),
        key=lambda row: (-row[1], row[0]),
    )
    return ", ".join(f"`{model}` {_percent(share)}" for model, share in rows if share > 0)


def _metric_with_comparison(metric: JsonObject, baseline: JsonObject) -> JsonObject:
    """Add baseline-relative retention and savings to one aggregate metric."""
    quality = _number(metric.get("quality"), label="metric.quality")
    cost = _number(metric.get("cost_per_task"), label="metric.cost_per_task")
    baseline_quality = _number(baseline.get("quality"), label="baseline.quality")
    baseline_cost = _number(baseline.get("cost_per_task"), label="baseline.cost_per_task")
    if baseline_quality <= 0:
        retention = float(quality >= baseline_quality)
    else:
        retention = quality / baseline_quality
    if baseline_cost <= 0:
        if cost == baseline_cost:
            savings = 0.0
        else:
            raise ValueError("baseline cost must be positive for savings reporting")
    else:
        savings = 1.0 - cost / baseline_cost
    return {
        **metric,
        "quality_retained": retention,
        "absolute_quality_delta": quality - baseline_quality,
        "cost_savings": savings,
    }


def _validate_evidence(root: Path) -> tuple[JsonObject, JsonObject, JsonObject, Path]:
    """Validate immutable evaluation, world-model, and serving evidence."""
    outcomes_path = root / "full" / "outcomes.json"
    lock_path = root / "analysis" / "selection-lock.json"
    outer_path = root / "analysis" / "outer-results.json"
    complete_path = root / "analysis" / "evaluation-complete.json"
    policy_path = root / "analysis" / "deployable" / "policy.json"
    comparison_path = root / "world-model" / "comparison.json"

    outcomes = _read_object(outcomes_path)
    lock = _read_object(lock_path)
    outer = _read_object(outer_path)
    complete = _read_object(complete_path)
    policy = _read_object(policy_path)
    comparison = _read_object(comparison_path)

    _object_rows(outcomes.get("outcomes"), label="full outcomes")
    seeds = _object_rows(outer.get("seeds"), label="outer seeds")
    observed_seeds = {row.get("seed") for row in seeds}
    if observed_seeds != set(SEEDS):
        raise ValueError("outer results must contain exactly the five frozen split seeds")
    if not isinstance(outer.get("promoted"), bool):
        raise ValueError("outer results have no scientific promotion verdict")
    _object_rows(outer.get("pareto"), label="outer pareto")
    _object(outer.get("paired_cluster_bootstrap"), label="paired_cluster_bootstrap")
    capability_slices = _object(
        outer.get("capability_slices"),
        label="outer capability_slices",
    )
    if not capability_slices:
        raise ValueError("outer results have no declared capability slices")
    ablations = _object(
        outer.get("one_at_a_time_ablations"),
        label="outer one_at_a_time_ablations",
    )
    if (
        ablations.get("benchmark_stratified") != "ablation:benchmark_stratified"
        or ablations.get("missing_fit_coverage_0.8") != "ablation:missing_fit_coverage_0.8"
        or ablations.get("latency_only_static") != "latency_only"
    ):
        raise ValueError("outer results have no frozen one-at-a-time ablations")
    if lock.get("matrix_sha256") != _sha256(outcomes_path):
        raise ValueError("selection lock does not match the real outcome matrix")
    if outer.get("matrix_sha256") != _sha256(outcomes_path):
        raise ValueError("outer results do not match the real outcome matrix")
    if outer.get("selection_lock_sha256") != _sha256(lock_path):
        raise ValueError("outer results do not match the selection lock")
    if (
        complete.get("heldout_evaluated") is not True
        or complete.get("selection_lock_sha256") != _sha256(lock_path)
        or complete.get("outer_results_sha256") != _sha256(outer_path)
    ):
        raise ValueError("heldout completion evidence does not match evaluation artifacts")

    bank_name = policy.get("knn_bank_path")
    if policy.get("kind") != "knn" or not isinstance(bank_name, str):
        raise ValueError("deployable policy is not a native WMO kNN artifact")
    bank_path = Path(bank_name) if Path(bank_name).is_absolute() else policy_path.parent / bank_name
    if not bank_path.is_file():
        raise ValueError("deployable policy evidence bank is missing")

    if (
        comparison.get("protocol") != "coding-world-model-compare-v1"
        or not isinstance(comparison.get("promotion_agreement"), bool)
        or not isinstance(comparison.get("deployment_consensus_config_agreement"), bool)
        or not isinstance(comparison.get("deployment_consensus_baseline_agreement"), bool)
    ):
        raise ValueError("world-model deployment comparison is incomplete")

    serving_results = sorted((root / "serving").glob("result-*.json"))
    passed_results = []
    for path in serving_results:
        result = _read_object(path)
        if (
            result.get("completion_status") == "passed"
            and _number(result.get("requests"), label="serving.requests") == 8
            and result.get("fallback_gate") == "novelty-abstain"
            and result.get("affinity_reason") == "sticky: conversation affinity"
            and _number(
                result.get("cache_aware_credit_usd"),
                label="serving.cache_aware_credit_usd",
            )
            > 0
        ):
            passed_results.append(path)
    if not (root / "serving" / "prepare.json").is_file() or not passed_results:
        raise ValueError("no passed real WMO serving verification is present")
    return outer, outcomes, comparison, passed_results[-1]


def _cost_accounting(outcomes: JsonObject) -> JsonObject:
    """Summarize exact and trace-estimated matrix costs without invoice claims."""
    rows = _object_rows(outcomes.get("outcomes"), label="full outcomes")
    exact_cost = 0.0
    estimated_cost = 0.0
    exact_cells = 0
    estimated_cells = 0
    for index, row in enumerate(rows):
        cost = _number(row.get("cost_usd"), label=f"outcomes[{index}].cost_usd")
        if row.get("usage_accounting") == "estimated":
            estimated_cost += cost
            estimated_cells += 1
        else:
            exact_cost += cost
            exact_cells += 1
    return {
        "exact_cells": exact_cells,
        "estimated_cells": estimated_cells,
        "exact_model_cost_usd": exact_cost,
        "estimated_model_cost_usd": estimated_cost,
        "operational_cost_comparison_is_approximate": estimated_cells > 0,
    }


def _unsafe_benchmarks(guarded: JsonObject, baseline: JsonObject) -> list[str]:
    """Find benchmark cohorts that fail the preregistered catastrophic-loss gates."""
    guarded_rows = _object(guarded.get("per_benchmark"), label="guarded.per_benchmark")
    baseline_rows = _object(baseline.get("per_benchmark"), label="baseline.per_benchmark")
    unsafe: list[str] = []
    for benchmark in BENCHMARKS:
        guarded_quality = _number(
            _object(guarded_rows.get(benchmark), label=benchmark).get("quality"),
            label=f"guarded.{benchmark}.quality",
        )
        baseline_quality = _number(
            _object(baseline_rows.get(benchmark), label=benchmark).get("quality"),
            label=f"baseline.{benchmark}.quality",
        )
        retention = (
            guarded_quality / baseline_quality
            if baseline_quality > 0
            else float(guarded_quality >= baseline_quality)
        )
        if retention < 0.90 or guarded_quality - baseline_quality < -0.10:
            unsafe.append(benchmark)
    return unsafe


def _capability_rows(outer: JsonObject) -> tuple[list[JsonObject], list[str]]:
    """Normalize guarded capability results and identify unsafe cohorts."""
    raw = _object(outer.get("capability_slices"), label="capability_slices")
    rows: list[JsonObject] = []
    unsafe: list[str] = []
    for capability in sorted(raw):
        value = _object(raw[capability], label=f"capability_slices.{capability}")
        seeds_observed = _number(
            value.get("seeds_observed"),
            label=f"{capability}.seeds_observed",
        )
        points = _object(value.get("points"), label=f"{capability}.points")
        guarded = _object(
            points.get("guarded_knn"),
            label=f"{capability}.guarded_knn",
        )
        retention = _number(
            guarded.get("quality_retained"),
            label=f"{capability}.quality_retained",
        )
        absolute_delta = _number(
            guarded.get("absolute_quality_delta"),
            label=f"{capability}.absolute_quality_delta",
        )
        if retention < 0.90 or absolute_delta < -0.10:
            unsafe.append(capability)
        rows.append(
            {
                "capability": capability,
                "seeds_observed": seeds_observed,
                **guarded,
            }
        )
    return rows, unsafe


def _table_row(
    label: str,
    metric: JsonObject,
    *,
    verdict: str,
) -> str:
    """Render one required Markdown policy row."""
    return (
        f"| {label} "
        f"| {_number(metric.get('quality'), label='quality'):.4f} "
        f"| {_percent(_number(metric.get('quality_retained'), label='retention'))} "
        f"| {_cost(_number(metric.get('cost_per_task'), label='cost_per_task'))} "
        f"| {_percent(_number(metric.get('cost_savings'), label='cost_savings'))} "
        f"| {_percent(_number(metric.get('success_rate'), label='success_rate'))} "
        f"| {_number(metric.get('latency_p50_s'), label='latency_p50_s'):.1f}s/"
        f"{_number(metric.get('latency_p95_s'), label='latency_p95_s'):.1f}s "
        f"| {verdict} |"
    )


def build_report(root: Path) -> tuple[str, JsonObject]:
    """Build a complete final report without making network or provider calls."""
    outer, outcomes, comparison, serving_result_path = _validate_evidence(root)
    seeds = _object_rows(outer.get("seeds"), label="outer seeds")
    baseline = _aggregate_point(seeds, "best_single")
    raw_points = {
        "best_single": baseline,
        "cheapest_single": _aggregate_point(seeds, "cheapest_single"),
        "unguarded_knn": _aggregate_point(seeds, "unguarded_knn"),
        "guarded_knn": _aggregate_point(seeds, "guarded_knn"),
        "selected_pareto": _aggregate_point(seeds, "guarded_knn"),
        "oracle": _aggregate_point(seeds, "oracle"),
    }
    points = {
        name: _metric_with_comparison(metric, baseline) for name, metric in raw_points.items()
    }
    promoted = cast("bool", outer["promoted"])
    pareto = _object_rows(outer.get("pareto"), label="outer pareto")
    selected_frontier_rows = [row for row in pareto if row.get("id") == "guarded_knn"]
    if len(selected_frontier_rows) != 1 or not isinstance(
        selected_frontier_rows[0].get("on_frontier"), bool
    ):
        raise ValueError("Pareto results do not identify the fit-locked guarded point")
    selected_on_frontier = cast("bool", selected_frontier_rows[0]["on_frontier"])
    bootstrap = _object(outer.get("paired_cluster_bootstrap"), label="paired bootstrap")
    retention_lower = _number(
        bootstrap.get("retention_lower_95"),
        label="paired bootstrap retention_lower_95",
    )
    guarded = points["guarded_knn"]
    unsafe = _unsafe_benchmarks(guarded, points["best_single"])
    capability_rows, unsafe_capabilities = _capability_rows(outer)
    accounting = _cost_accounting(outcomes)
    selected_mix = _object(guarded.get("model_mix"), label="guarded.model_mix")
    baseline_name = outer.get("deployment_consensus_baseline")
    if not isinstance(baseline_name, str):
        raise ValueError("outer results have no deployment consensus baseline")
    route_away = _number(guarded.get("route_away_rate"), label="guarded.route_away_rate")
    guard_reversion = _number(
        guarded.get("guard_reversion_rate"),
        label="guarded.guard_reversion_rate",
    )
    novelty = _number(
        guarded.get("novelty_abstention_rate"),
        label="guarded.novelty_abstention_rate",
    )
    same_world_decision = (
        comparison.get("promotion_agreement") is True
        and comparison.get("deployment_consensus_config_agreement") is True
        and comparison.get("deployment_consensus_baseline_agreement") is True
    )
    approximate = cast("bool", accounting["operational_cost_comparison_is_approximate"])

    baseline_verdict = "Fit-selected reference"
    cheapest_verdict = (
        "Passes quality gate"
        if _number(points["cheapest_single"].get("quality_retained"), label="retention")
        >= QUALITY_RETENTION_GATE
        else "Below quality gate"
    )
    unguarded_verdict = (
        "Passes both gates"
        if (
            _number(points["unguarded_knn"].get("quality_retained"), label="retention")
            >= QUALITY_RETENTION_GATE
            and _number(points["unguarded_knn"].get("cost_savings"), label="savings")
            >= SAVINGS_GATE
        )
        else "Does not pass both gates"
    )
    guarded_verdict = (
        "Promote"
        if promoted and selected_on_frontier
        else "Passes gates, not on measured frontier"
        if promoted
        else "Do not promote"
    )
    oracle_verdict = "Unattainable upper bound"

    target_statement = "The target was achieved." if promoted else "The target was not achieved."
    accounting_statement = (
        "Cost comparisons are approximate because at least one full-matrix cell uses "
        "a labeled trace estimate."
        if approximate
        else "Full-matrix model costs use provider-reported counters."
    )
    unsafe_statement = (
        ", ".join(f"`{capability}`" for capability in unsafe_capabilities)
        if unsafe_capabilities
        else (
            "No declared capability cohort failed the catastrophic-loss gate"
            if not unsafe
            else ", ".join(f"`{benchmark}`" for benchmark in unsafe)
        )
    )
    world_statement = (
        "Yes. Simulation matched the real promotion decision, deployment configuration, "
        "and fit-selected baseline."
        if same_world_decision
        else "No. Simulation differed from the real deployment decision, configuration, "
        "or fit-selected baseline."
    )
    rollout = (
        "Start a limited production rollout with routed-model audit evidence enabled, "
        f"`{baseline_name}` as the mandatory fallback, and automatic rollback if live "
        "quality or savings breach the frozen gates."
        if promoted
        else f"Keep `{baseline_name}` as the production default. Do not roll out the router "
        "until a newly fit policy passes the same five-seed and serving gates."
    )

    report = "\n".join(
        [
            "# Coding Model Router Final Report",
            "",
            "| Policy | Quality | Quality retained | Cost/task | Cost savings | Completion | "
            "Latency p50/p95 | Verdict |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
            _table_row("Best Single", points["best_single"], verdict=baseline_verdict),
            _table_row("Cheapest Single", points["cheapest_single"], verdict=cheapest_verdict),
            _table_row("Unguarded Router", points["unguarded_knn"], verdict=unguarded_verdict),
            _table_row("Guarded Router", guarded, verdict=guarded_verdict),
            _table_row(
                "Selected Pareto Point (fit-locked guarded kNN)",
                points["selected_pareto"],
                verdict=guarded_verdict,
            ),
            _table_row("Oracle Upper Bound", points["oracle"], verdict=oracle_verdict),
            "",
            "## Conclusion",
            "",
            target_statement,
            "",
            (
                f"The selected policy is the fit-locked guarded WMO kNN router with "
                f"`{baseline_name}` as its frontier fallback. Its model mix is "
                f"{_mix_text(selected_mix)}."
            ),
            "",
            (
                f"It routes away from the frontier model on {_percent(route_away)} of heldout "
                "requests. That routing mix produces the measured cost reduction shown above."
            ),
            "",
            (
                f"It reverts through the statistical guard on {_percent(guard_reversion)} and "
                f"abstains for novelty on {_percent(novelty)} of requests."
            ),
            "",
            f"Unsafe task classes: {unsafe_statement}.",
            "",
            f"World-model evaluation selected the same policy: {world_statement}",
            "",
            (
                f"The paired clustered-bootstrap lower 95 percent retention bound is "
                f"{_percent(retention_lower)}. {accounting_statement}"
            ),
            "",
            "## Capability Slices",
            "",
            "| Capability | Seeds | Quality | Quality retained | Absolute delta | "
            "Cost/task | Cost savings | Completion |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *[
                (
                    f"| {row['capability']} "
                    f"| {int(_number(row.get('seeds_observed'), label='seeds'))}/5 "
                    f"| {_number(row.get('quality'), label='quality'):.4f} "
                    f"| {_percent(_number(row.get('quality_retained'), label='retention'))} "
                    f"| {_number(row.get('absolute_quality_delta'), label='delta'):+.4f} "
                    f"| {_cost(_number(row.get('cost_per_task'), label='cost'))} "
                    f"| {_percent(_number(row.get('cost_savings'), label='savings'))} "
                    f"| {_percent(_number(row.get('success_rate'), label='completion'))} |"
                )
                for row in capability_rows
            ],
            "",
            "## Limitations",
            "",
            (
                "- The real result covers only the frozen Terminal-Bench 2 and SWE-bench "
                "Verified cohorts, model roster, harness, attempt count, and provider versions."
            ),
            (
                "- Estimated usage is operational evidence, not provider invoice evidence."
                if approximate
                else "- Model cost excludes any environment charge not exposed by E2B."
            ),
            (
                "- The world model was built from one reward-free baseline trajectory per task, "
                "so its comparison does not establish unseen-task generalization."
            ),
            (
                "- The simulated matrix uses WMO's native agent scaffold while the real matrix "
                "uses Harbor and Pi, which is a simulation-to-real confound."
            ),
            (
                "- Prompt-cache-aware switching and conversation affinity are serving-only "
                "operational checks. The one-shot benchmark matrix does not identify their "
                "quality effect."
            ),
            *(
                [
                    (
                        "- The fit-locked guarded point passed the frozen promotion gates but was "
                        "not on the measured heldout frontier. Heldout results were not used to "
                        "replace the preregistered deployment choice."
                    )
                ]
                if promoted and not selected_on_frontier
                else []
            ),
            "",
            "## Rollout Recommendation",
            "",
            rollout,
            "",
            "## Traceability",
            "",
            f"- Real matrix SHA-256: `{outer['matrix_sha256']}`",
            f"- Selection lock SHA-256: `{outer['selection_lock_sha256']}`",
            f"- Serving evidence: `{serving_result_path.relative_to(root)}`",
            "- World-model comparison: `world-model/comparison.json`",
            "- Machine-readable summary: `final-summary.json`",
            "",
        ]
    )
    summary: JsonObject = {
        "protocol": "coding-router-final-report-v1",
        "experiment_id": EXPERIMENT_ID,
        "target_achieved": promoted,
        "quality_retention_gate": QUALITY_RETENTION_GATE,
        "cost_savings_gate": SAVINGS_GATE,
        "paired_retention_lower_95": retention_lower,
        "selected_policy": {
            "id": "guarded_knn",
            "baseline": baseline_name,
            "deployment_consensus_config": outer.get("deployment_consensus_config"),
            "model_mix": selected_mix,
            "on_measured_heldout_frontier": selected_on_frontier,
        },
        "points": points,
        "unsafe_benchmarks": unsafe,
        "unsafe_capabilities": unsafe_capabilities,
        "capability_slices": {cast("str", row["capability"]): row for row in capability_rows},
        "world_model_same_deployment_decision": same_world_decision,
        "cost_accounting": accounting,
        "serving_result": str(serving_result_path.relative_to(root)),
        "evidence": {
            "real_matrix_sha256": outer["matrix_sha256"],
            "selection_lock_sha256": outer["selection_lock_sha256"],
        },
    }
    return report, summary


def write_report(root: Path) -> tuple[Path, Path]:
    """Atomically write the final Markdown and machine-readable summary."""
    report, summary = build_report(root)
    report_path = root / "final-report.md"
    summary_path = root / "final-summary.json"
    write_text_atomic(
        summary_path,
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    write_text_atomic(report_path, report)
    return report_path, summary_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(".wmo") / "experiments" / EXPERIMENT_ID,
    )
    return parser.parse_args()


def main() -> None:
    """Render the final report from complete persisted evidence."""
    args = _parse_args()
    report_path, summary_path = write_report(args.root.resolve())
    logger.info("wrote %s and %s", report_path, summary_path)


if __name__ == "__main__":
    main()
