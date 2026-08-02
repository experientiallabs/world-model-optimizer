"""Run one selected model-effort arm on the untouched external confirmation cohort."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import coding_model_router_swerebench_execute as runner

MODELS = {
    "gpt-5.6-luna": ("luna", (1.0, 0.1, 6.0)),
    "gpt-5.6-terra": ("terra", (2.5, 0.25, 15.0)),
    "gpt-5.6-sol": ("sol", (5.0, 0.5, 30.0)),
}
CONFIRMATION_CORPUS_SHA256 = (
    "9798dd1e58be0d13331d097307670dc3fc3760ad211da20e6367666523f080a7"
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return {str(key): item for key, item in value.items()}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} contains a non-object row")
        rows.append({str(key): item for key, item in value.items()})
    return rows


def authorize(
    model: str,
    effort: str,
    fit_output: Path,
    development_audit_path: Path,
    corpus_path: Path,
) -> tuple[float, dict[str, object]]:
    """Verify frozen pair selection and route controls before provider execution."""
    if model not in MODELS or effort not in runner.EFFORTS:
        raise ValueError("requested arm is outside the frozen roster")
    if runner._sha256(corpus_path) != CONFIRMATION_CORPUS_SHA256:
        raise ValueError("confirmation corpus changed")
    report_path = fit_output / "selection-report.json"
    lock_path = fit_output / "selection-lock.json"
    routes_path = fit_output / "confirmation-routes.jsonl"
    blind_path = fit_output / "confirmation-blind-routes.jsonl"
    null_path = fit_output / "confirmation-null-routes.jsonl"
    required = (report_path, lock_path, routes_path, blind_path, null_path)
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("fit output lacks frozen confirmation routes")
    report = _read_object(report_path)
    lock = _read_object(lock_path)
    audit = _read_object(development_audit_path)
    prefix = MODELS[model][0]
    arm = f"{prefix}-{effort}"
    selected_pair = report.get("selected_pair")
    baseline = report.get("development_static_baseline")
    if (
        report.get("valid") is not True
        or report.get("development_passed") is not True
        or report.get("confirmation_authorized") is not True
        or not isinstance(selected_pair, list)
        or len(selected_pair) != 2
        or baseline not in selected_pair
        or arm not in selected_pair
        or report.get("target_outcomes_used") is not False
        or report.get("deep_swe_outcomes_accessed") is not False
        or report.get("fitted_numeric_router_state_persisted") is not False
    ):
        raise ValueError("fit report does not authorize the requested arm")
    if (
        lock.get("valid") is not True
        or lock.get("selected_pair") != selected_pair
        or lock.get("development_static_baseline") != baseline
        or lock.get("selected_candidate") != report.get("selected_candidate")
        or lock.get("null_count") != 128
        or lock.get("null_unique_route_hashes") != 128
        or lock.get("target_outcomes_used") is not False
        or lock.get("deep_swe_outcomes_accessed") is not False
        or lock.get("fitted_numeric_router_state_persisted") is not False
        or report.get("selection_lock_sha256") != runner._sha256(lock_path)
        or lock.get("confirmation_routes_sha256") != runner._sha256(routes_path)
        or lock.get("confirmation_blind_routes_sha256") != runner._sha256(blind_path)
        or lock.get("confirmation_null_routes_sha256") != runner._sha256(null_path)
    ):
        raise ValueError("selection lock or route hashes changed")
    if (
        audit.get("valid") is not True
        or audit.get("target_outcomes_used") is not False
        or audit.get("deep_swe_outcomes_accessed") is not False
        or report.get("inputs", {}).get("completion_audit_sha256")
        != runner._sha256(development_audit_path)
    ):
        raise ValueError("development audit is incomplete or not content-addressed")
    spend = audit.get("rough_cumulative_experiment_spend_usd")
    if (
        isinstance(spend, bool)
        or not isinstance(spend, (int, float))
        or not 0.0 <= float(spend) < 20_000.0
    ):
        raise ValueError("development audit has invalid cumulative spend")
    tasks = _read_object(corpus_path).get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 200:
        raise ValueError("confirmation corpus must contain 200 tasks")
    task_ids = [str(task.get("task_id")) for task in tasks if isinstance(task, dict)]
    routes = _read_rows(routes_path)
    blind = _read_rows(blind_path)
    nulls = _read_rows(null_path)
    if (
        len(task_ids) != 200
        or len(set(task_ids)) != 200
        or len(routes) != 200
        or len(blind) != 200
        or len(nulls) != 128 * 200
        or [str(row.get("task_id")) for row in routes] != task_ids
        or [str(row.get("task_id")) for row in blind] != task_ids
    ):
        raise ValueError("frozen routes do not exactly cover confirmation tasks")
    allowed = set(selected_pair)
    if any(row.get("arm") not in allowed for row in routes + blind + nulls):
        raise ValueError("frozen route contains an arm outside the selected pair")
    return float(spend), {
        "selected_candidate": report["selected_candidate"],
        "selected_pair": selected_pair,
        "development_static_baseline": baseline,
        "selection_report_sha256": runner._sha256(report_path),
        "selection_lock_sha256": runner._sha256(lock_path),
        "confirmation_routes_sha256": runner._sha256(routes_path),
        "confirmation_blind_routes_sha256": runner._sha256(blind_path),
        "confirmation_null_routes_sha256": runner._sha256(null_path),
        "development_audit_sha256": runner._sha256(development_audit_path),
        "confirmation_corpus_sha256": CONFIRMATION_CORPUS_SHA256,
        "requested_arm": arm,
        "target_outcomes_used": False,
    }


def configure(model: str, effort: str, prior_spend: float) -> tuple[str, str]:
    """Bind the generic resumable runner to one selected arm."""
    if model not in MODELS or effort not in runner.EFFORTS:
        raise ValueError("unsupported selected arm")
    slug, _ = MODELS[model]
    arm = f"{slug}-{effort}"
    runner.MODEL = model
    runner.EFFORTS = (effort,)
    runner.DEFAULT_PRIOR_SPEND_USD = prior_spend
    runner.REMOTE_VALIDATOR = runner.REMOTE_VALIDATOR.replace(
        '"gpt-5.6-luna"', f'"{model}"'
    )
    runner.REUSED_TASKS = set()
    runner.SMOKE_ARCHIVE_SHA256 = {}
    runner.DEVELOPMENT_PHASE = runner.ExecutionPhase(
        name="development",
        protocol=f"coding-router-model-effort-confirmation-{arm}-v1",
        corpus_sha256=CONFIRMATION_CORPUS_SHA256,
        remote_segment=f"model-effort-v43-confirmation-{arm}",
        metadata_phase=f"model-effort-v43-confirmation-{arm}",
        reuse_smoke=False,
        metadata_owner="coding-router-v43",
    )
    return slug, arm


def main() -> None:
    """Authorize and run or resume one selected confirmation arm."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--effort", choices=runner.EFFORTS, required=True)
    parser.add_argument("--fit-output", type=Path, required=True)
    parser.add_argument("--development-audit", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=100)
    args = parser.parse_args()
    prior_spend, authorization = authorize(
        args.model,
        args.effort,
        args.fit_output,
        args.development_audit,
        args.corpus,
    )
    _, arm = configure(args.model, args.effort, prior_spend)
    runner.EXTERNAL_AUTHORIZATION = authorization
    args.root.mkdir(parents=True, exist_ok=True)
    authorization_path = args.root / "authorization.json"
    if authorization_path.is_file():
        if _read_object(authorization_path) != authorization:
            raise ValueError("resume authorization changed")
    else:
        runner._write_json(authorization_path, authorization)
    runner.execute(
        args.root,
        args.corpus,
        concurrency=args.concurrency,
        limit_tasks=None,
        phase_name="development",
    )
    logging.getLogger("coding-router-model-effort-confirm").info(
        "completed selected confirmation arm=%s", arm
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
