"""Run the sealed graded SWE-rebench confirmation matrix after development passes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import coding_model_router_graded_swerebench_execute as runner

PROTOCOL = "coding-router-graded-swerebench-confirmation-execution-v1"
CORPUS_SHA256 = "c9443c9956e496123f396ee793efbb3368312092c4dcbd4e5e10bb77bd814f0a"
TASKS = 320
FIT_PROTOCOL = "coding-router-graded-swerebench-wmo-knn-v1"
COLLECTION_PROTOCOL = "coding-router-graded-swerebench-development-collection-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return {str(key): item for key, item in value.items()}


def authorization(
    corpus: Path,
    development_audit: Path,
    fit_report: Path,
    routes_path: Path,
) -> tuple[float, dict[str, Any]]:
    """Validate the development pass without reading confirmation outcomes."""
    if _sha256(corpus) != CORPUS_SHA256:
        raise ValueError("confirmation corpus changed")
    audit = _object(development_audit)
    report = _object(fit_report)
    routes = _object(routes_path)
    selected = report.get("development", {}).get("selected")
    selected_candidate = selected.get("candidate") if isinstance(selected, dict) else None
    route_rows = routes.get("routes")
    if (
        audit.get("protocol") != COLLECTION_PROTOCOL
        or audit.get("valid") is not True
        or audit.get("target_outcomes_used") is not False
        or audit.get("deep_swe_outcomes_accessed") is not False
        or audit.get("confirmation_outcomes_accessed") is not False
        or report.get("protocol") != FIT_PROTOCOL
        or report.get("valid") is not True
        or report.get("development_passed") is not True
        or not isinstance(selected_candidate, str)
        or report.get("confirmation_routes_frozen") is not True
        or report.get("target_outcomes_used") is not False
        or report.get("deep_swe_outcomes_accessed") is not False
        or report.get("confirmation_outcomes_accessed") is not False
        or routes.get("protocol") != FIT_PROTOCOL
        or routes.get("selected_candidate") != selected_candidate
        or not isinstance(route_rows, list)
        or len(route_rows) != TASKS
        or len({str(row.get("task_id")) for row in route_rows if isinstance(row, dict)})
        != TASKS
        or routes.get("deep_swe_outcomes_accessed") is not False
        or routes.get("confirmation_outcomes_accessed") is not False
        or routes.get("fitted_numeric_state_persisted") is not False
    ):
        raise ValueError("development fit did not safely authorize confirmation")
    spend = audit.get("rough_cumulative_experiment_spend_usd")
    if isinstance(spend, bool) or not isinstance(spend, (int, float)):
        raise ValueError("development audit lacks cumulative spend")
    return float(spend), {
        "development_audit_sha256": _sha256(development_audit),
        "development_outcomes_sha256": audit["outcomes_sha256"],
        "fit_report_sha256": _sha256(fit_report),
        "confirmation_routes_sha256": _sha256(routes_path),
        "confirmation_corpus_sha256": CORPUS_SHA256,
        "selected_candidate": selected_candidate,
        "confirmation_outcomes_accessed_before_launch": False,
        "deep_swe_outcomes_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--verifier-tasks", type=Path, required=True)
    parser.add_argument("--patched-taskset", type=Path, required=True)
    parser.add_argument("--patch-report", type=Path, required=True)
    parser.add_argument("--development-audit", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=100)
    args = parser.parse_args()
    prior_spend, frozen = authorization(
        args.corpus,
        args.development_audit,
        args.fit_report,
        args.routes,
    )
    runner.PROTOCOL = PROTOCOL
    runner.PHASE_NAME = "confirmation"
    runner.EXPECTED_TASKS = TASKS
    runner.REMOTE_SEGMENT = "confirmation"
    runner.METADATA_PHASE = "graded-swerebench-confirmation"
    runner.CORPUS_SHA256 = CORPUS_SHA256
    runner.PRIOR_SPEND_USD = prior_spend
    runner.EXTERNAL_AUTHORIZATION = frozen
    runner.execute(
        args.root,
        args.corpus,
        args.verifier_tasks,
        args.patched_taskset,
        args.patch_report,
        concurrency=args.concurrency,
    )


if __name__ == "__main__":
    main()
