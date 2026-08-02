"""Run the frozen graded workload-budget development study on remote compute."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from coding_model_router_graded_swerebench_fit import (
    ARMS,
    SEEDS,
    _frontiers,
    _sha256,
    load_confirmation,
    load_data,
)
from coding_model_router_graded_wiserouter import (
    NULL_COUNT,
    NULL_SEED_START,
    PROTOCOL,
    Candidate,
    ContextPlan,
    build_context_plan,
    candidate_grid,
    evaluate_candidate_seed,
    fit_full_policy,
    freeze_choices,
    measure_latency_ms,
    null_gate,
    passes_primary_gates,
    permute_repository_blocks,
)

logger = logging.getLogger("coding-router-graded-wiserouter")


def _task_text(task: dict[str, Any]) -> str:
    """Build the exact pre-call task text used by development."""
    return (
        f"repository={task['repository']}\n"
        f"language={task['language']}\n"
        f"{task['prompt']}"
    )


def _candidate_summary(
    candidate: Candidate,
    seed_metrics: list[dict[str, Any]],
    *,
    primary_eligible: bool,
    control: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return one aggregate-only candidate report row."""
    return {
        "key": candidate.key,
        "order": candidate.order,
        "hash_dim": candidate.hash_dim,
        "contexts": candidate.contexts,
        "shrinkage": candidate.shrinkage,
        "target_savings": candidate.savings,
        "seed_metrics": seed_metrics,
        "primary_eligible": primary_eligible,
        "null_gate": control,
        "scientifically_eligible": bool(control and control["passed"] is True),
    }


def _selection_key(row: dict[str, Any]) -> tuple[float, float, float, int, int, float, int]:
    """Apply the frozen deterministic winner ordering."""
    candidate = row["candidate"]
    metrics = row["metrics"]
    control = row["control"]
    if (
        not isinstance(candidate, Candidate)
        or not isinstance(metrics, list)
        or not isinstance(control, dict)
    ):
        raise TypeError("selection row is malformed")
    return (
        float(np.mean([float(value["cost_usd_per_task"]) for value in metrics])),
        -min(float(value["quality_retention"]) for value in metrics),
        -float(control["real_minus_null95"]),
        candidate.contexts,
        candidate.hash_dim,
        -candidate.shrinkage,
        candidate.order,
    )


def run_study(
    corpus: Path,
    outcomes: Path,
    audit: Path,
    confirmation_corpus: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Run development selection, null controls, latency, and conditional route freeze."""
    data = load_data(corpus, outcomes, audit)
    confirmation = load_confirmation(confirmation_corpus)
    plans: dict[tuple[int, int, int], ContextPlan | None] = {}
    grid = candidate_grid()
    plan_count = len(SEEDS) * len({value.hash_dim for value in grid}) * len(
        {value.contexts for value in grid}
    )
    plans_completed = 0
    for seed in SEEDS:
        for hash_dim in sorted({candidate.hash_dim for candidate in grid}):
            for contexts in sorted({candidate.contexts for candidate in grid}):
                plans[(seed, hash_dim, contexts)] = build_context_plan(
                    data.texts,
                    data.repositories,
                    hash_dim=hash_dim,
                    contexts=contexts,
                    seed=seed,
                )
                plans_completed += 1
                logger.info(
                    "context_plans_completed=%d context_plans_total=%d",
                    plans_completed,
                    plan_count,
                )

    primary: list[dict[str, Any]] = []
    metrics_by_key: dict[str, list[dict[str, Any]]] = {}
    for candidate_index, candidate in enumerate(grid, start=1):
        metrics: list[dict[str, Any]] = []
        for seed in SEEDS:
            plan = plans[(seed, candidate.hash_dim, candidate.contexts)]
            if plan is None:
                break
            metrics.append(evaluate_candidate_seed(data, candidate, plan))
        metrics_by_key[candidate.key] = metrics
        if passes_primary_gates(metrics):
            primary.append({"candidate": candidate, "metrics": metrics})
        logger.info(
            "primary_candidates_evaluated=%d primary_candidates_total=%d primary_eligible=%d",
            candidate_index,
            len(grid),
            len(primary),
        )

    null_inputs = [
        (
            permute_repository_blocks(
                data.rewards,
                data,
                seed=NULL_SEED_START + offset,
            ),
            permute_repository_blocks(
                data.costs,
                data,
                seed=NULL_SEED_START + offset,
            ),
        )
        for offset in range(NULL_COUNT)
    ]
    eligible: list[dict[str, Any]] = []
    controls: dict[str, dict[str, Any]] = {}
    for primary_index, value in enumerate(primary, start=1):
        candidate = value["candidate"]
        metrics = value["metrics"]
        if not isinstance(candidate, Candidate) or not isinstance(metrics, list):
            raise TypeError("primary candidate row is malformed")
        null_metrics: list[list[dict[str, Any]]] = []
        for fit_rewards, fit_costs in null_inputs:
            rows = []
            for seed in SEEDS:
                plan = plans[(seed, candidate.hash_dim, candidate.contexts)]
                if plan is None:
                    raise RuntimeError("primary candidate lost a fitted context plan")
                rows.append(
                    evaluate_candidate_seed(
                        data,
                        candidate,
                        plan,
                        fit_rewards=fit_rewards,
                        fit_costs=fit_costs,
                    )
                )
            null_metrics.append(rows)
        control = null_gate(metrics, null_metrics)
        controls[candidate.key] = control
        if control["passed"] is True:
            eligible.append({"candidate": candidate, "metrics": metrics, "control": control})
        logger.info(
            "null_candidates_evaluated=%d null_candidates_total=%d scientifically_eligible=%d",
            primary_index,
            len(primary),
            len(eligible),
        )

    selected_row = min(eligible, key=_selection_key) if eligible else None
    latency = None
    selected_key = None
    routes = None
    if selected_row is not None:
        selected = selected_row["candidate"]
        if not isinstance(selected, Candidate):
            raise TypeError("selected candidate is malformed")
        policy = fit_full_policy(data, selected, seed=20_260_801)
        confirmation_ids = [str(task["task_id"]) for task in confirmation]
        confirmation_texts = [_task_text(task) for task in confirmation]
        latency = measure_latency_ms(policy, confirmation_ids, confirmation_texts)
        if latency["eligible"] is True:
            selected_key = selected.key
            choices = freeze_choices(policy, confirmation_ids, confirmation_texts)
            routes = {
                "protocol": "coding-router-graded-wiserouter-confirmation-routes-v1",
                "selected_key": selected.key,
                "development_outcomes_sha256": _sha256(outcomes),
                "development_audit_sha256": _sha256(audit),
                "confirmation_corpus_sha256": _sha256(confirmation_corpus),
                "target_outcomes_used": False,
                "deep_swe_outcomes_accessed": False,
                "confirmation_outcomes_accessed": False,
                "fitted_numeric_state_persisted": False,
                "task_embeddings_persisted": False,
                "routes": [
                    {
                        "task_id": task_id,
                        "arm": ARMS[int(choices[index])],
                        "route_order": index,
                    }
                    for index, task_id in enumerate(confirmation_ids)
                ],
            }

    candidates = []
    for candidate in grid:
        metrics = metrics_by_key[candidate.key]
        control = controls.get(candidate.key)
        candidates.append(
            _candidate_summary(
                candidate,
                metrics,
                primary_eligible=passes_primary_gates(metrics),
                control=control,
            )
        )
    report = {
        "protocol": PROTOCOL,
        "valid": True,
        "tasks": len(data.task_ids),
        "arms": list(ARMS),
        "candidate_count": len(candidates),
        "primary_eligible_count": len(primary),
        "scientifically_eligible_count": len(eligible),
        "selected_key": selected_key,
        "development_passed": selected_key is not None,
        "latency": latency,
        "candidates": candidates,
        "frontiers": _frontiers(data),
        "inputs": {
            str(corpus): _sha256(corpus),
            str(outcomes): _sha256(outcomes),
            str(audit): _sha256(audit),
            str(confirmation_corpus): _sha256(confirmation_corpus),
        },
        "rough_cumulative_spend_usd": data.rough_cumulative_spend_usd,
        "provider_calls": 0,
        "target_outcomes_used": False,
        "deep_swe_outcomes_accessed": False,
        "confirmation_outcomes_accessed": False,
        "confirmation_routes_frozen": routes is not None,
        "fitted_numeric_state_persisted": False,
        "task_embeddings_persisted": False,
        "outcome_matrix_persisted": False,
    }
    return report, routes


def main() -> None:
    """Run the frozen study and write aggregate-only results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--confirmation-corpus", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--routes-out", type=Path, required=True)
    args = parser.parse_args()
    report, routes = run_study(
        args.corpus,
        args.outcomes,
        args.audit,
        args.confirmation_corpus,
    )
    args.report_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if routes is not None:
        args.routes_out.write_text(
            json.dumps(routes, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    logger.info(
        "graded workload-router development completed: passed=%s selected=%s",
        report["development_passed"],
        report["selected_key"],
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
