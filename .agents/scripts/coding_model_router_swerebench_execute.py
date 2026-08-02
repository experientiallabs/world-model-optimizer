"""Run the frozen SWE-rebench development matrix on resumable E2B workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, NamedTuple

from e2b import CommandResult, Sandbox, Template

logger = logging.getLogger("coding-router-swerebench-execute")

PROTOCOL = "coding-router-swerebench-development-execution-v1"
CONFIRMATION_PROTOCOL = "coding-router-swerebench-confirmation-execution-v1"
POOLED_CONFIRMATION_PROTOCOL = "coding-router-pooled-uplift-confirmation-execution-v1"
TEMPLATE_NAME = "deepswe-router-responses-v2"
TEMPLATE_ID = "j1a2bxbpllu3rp84b4qj"
TEMPLATE_BUILD_ID = "e971c040-95bd-45c1-89ee-fb597bf75671"
MODEL: str = "gpt-5.6-luna"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
MODEL_PRICES_PER_MTOK = {
    "gpt-5.6-luna": (1.0, 0.1, 6.0),
    "gpt-5.6-terra": (2.5, 0.25, 15.0),
    "gpt-5.6-sol": (5.0, 0.5, 30.0),
}
CORPUS_SHA256 = "7d846b5576d15e68fd18ac21bfe0610cc1614b3b35ec0ae0cb8cfae0b82962c1"
CONFIRMATION_CORPUS_SHA256 = (
    "9798dd1e58be0d13331d097307670dc3fc3760ad211da20e6367666523f080a7"
)
POOLED_CONFIRMATION_CORPUS_SHA256 = (
    "6edd8ed4777d6bc48cf29f76a9fb4b9d60e3324908aa79d4d03df8617f6be825"
)
POOLED_CONFIRMATION_MANIFEST_SHA256 = (
    "7bd743a794c5054e053a9d163c088d0f9f72fbd911043c44f90b792801eade60"
)
POOLED_CONFIRMATION_ROUTES_SHA256 = (
    "aac7523746ee9aac0f9789ba9ee4d4e260fad8d2447730102d7aaa44816224c8"
)
POOLED_CONFIRMATION_NULL_ROUTES_SHA256 = (
    "4e1570b285eac8da96c13069479f3f7ea9e49b7bedaae8c90d636777a3212a59"
)
POOLED_CONFIRMATION_ROUTE_AUDIT_SHA256 = (
    "4f31fe2245cbad1123beada405d18d12b6e323963bde14c3ac69a90993c4db6b"
)
POOLED_CONFIRMATION_FREEZE_LOCK_SHA256 = (
    "c8deac37d91912e268108a94a227abae34ab858e3bbbd2c637623697da092751"
)
POOLED_SELECTED_CANDIDATE = "direct_ridge-hash8192-a10"
POOLED_PRIOR_SPEND_USD = 887.541861
DEFAULT_PRIOR_SPEND_USD = 405.7678502
SMOKE_REPORT_SHA256 = "ee76a57040cbe7aaef692d2fc3f3df66d7a556cbf6dda74119e0802cb4230e13"
SMOKE_ARCHIVE_SHA256 = {
    "xhigh": "bf1d576d25f1b56ae3a9484db5d5599576519a218aec3073db29272345f4015b",
    "max": "c449dc999a4d604546c358affcf5e1cba1865aba8ca312789b92b5eb27bb4e6a",
}
REUSED_TASKS = {
    "0xs34n__starknet.js-538",
    "acloudguru__serverless-plugin-aws-alerts-13",
}
DOCKER_ADAPTER_REPORT_SHA256 = (
    "08499c87fa93b9ec58c76fabbf16c388df169506bd5160e2ca604f3a7b62938a"
)
RESPONSES_ADAPTER_REPORT_SHA256 = (
    "476f4a5e0a67fc4880fc80ed27e52d333620461da63775689e8f5be38e66179c"
)
E2B_ACCOUNT_CAP = 1_000
EXTERNAL_AUTHORIZATION: dict[str, object] | None = None
TASK_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+$")
IMAGE_PATTERN = re.compile(r"^docker\.io/swerebenchv2/[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+$")
STATE_LOCK = threading.Lock()


class ExecutionPhase(NamedTuple):
    """Frozen execution differences between development and confirmation."""

    name: str
    protocol: str
    corpus_sha256: str
    remote_segment: str
    metadata_phase: str
    reuse_smoke: bool
    metadata_owner: str


DEVELOPMENT_PHASE = ExecutionPhase(
    name="development",
    protocol=PROTOCOL,
    corpus_sha256=CORPUS_SHA256,
    remote_segment="development",
    metadata_phase="swerebench-development-matrix",
    reuse_smoke=True,
    metadata_owner="coding-router-v40",
)
CONFIRMATION_PHASE = ExecutionPhase(
    name="confirmation",
    protocol=CONFIRMATION_PROTOCOL,
    corpus_sha256=CONFIRMATION_CORPUS_SHA256,
    remote_segment="confirmation",
    metadata_phase="swerebench-confirmation-matrix",
    reuse_smoke=False,
    metadata_owner="coding-router-v40",
)
POOLED_CONFIRMATION_PHASE = ExecutionPhase(
    name="pooled-confirmation",
    protocol=POOLED_CONFIRMATION_PROTOCOL,
    corpus_sha256=POOLED_CONFIRMATION_CORPUS_SHA256,
    remote_segment="pooled-confirmation-v42",
    metadata_phase="pooled-uplift-confirmation-matrix",
    reuse_smoke=False,
    metadata_owner="coding-router-v42",
)

REMOTE_VALIDATOR: str = r'''"""Audit new matrix traces and write a compact report."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--traces", type=Path, required=True)
parser.add_argument("--task", required=True)
parser.add_argument("--effort", required=True)
parser.add_argument("--expected", type=int, required=True)
parser.add_argument("--attempt-offset", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

records = [
    json.loads(line)
    for line in args.traces.read_bytes().split(b"\n")
    if line.strip()
]
if len(records) != args.expected:
    raise SystemExit(f"expected {args.expected} outer rows, found {len(records)}")
cells = []
totals = {field: 0 for field in (
    "prompt_tokens", "cached_input_tokens", "completion_tokens", "reasoning_tokens"
)}
for index, outer in enumerate(records):
    traces = outer.get("traces")
    if not isinstance(traces, list) or len(traces) != 1:
        raise SystemExit(f"row {index} lacks exactly one official trace")
    trace = traces[0]
    if trace.get("task", {}).get("data", {}).get("name") != args.task:
        raise SystemExit(f"row {index} has a different task")
    if trace.get("verifiers", {}).get("commit") != "f6e420b9908ae14d625f079881f13c15011ee1c9":
        raise SystemExit(f"row {index} has a different verifier commit")
    calls = trace.get("calls")
    if not isinstance(calls, list) or not calls or len(calls) > 40:
        raise SystemExit(f"row {index} has invalid provider calls")
    reward = trace.get("rewards", {}).get("solved", {}).get("score")
    trace_errors = trace.get("errors", [])
    timeout_error = isinstance(trace_errors, list) and any(
        isinstance(error, dict)
        and error.get("type") == "HarnessError"
        and isinstance(error.get("message"), str)
        and error["message"].startswith("agent timeout:")
        for error in trace_errors
    )
    mini_swe_agent_exit_137 = isinstance(trace_errors, list) and any(
        isinstance(error, dict)
        and error.get("type") == "HarnessError"
        and isinstance(error.get("message"), str)
        and error["message"].startswith("harness 'mini_swe_agent' exited 137:")
        for error in trace_errors
    )
    post_execution_agent_failure = (
        reward is None
        and trace.get("ok") is False
        and outer.get("ok") is False
        and trace.get("stop_condition") in {"error", "max_turns"}
        and trace.get("info", {}).get("patch") is None
        and isinstance(trace_errors, list)
        and (timeout_error or mini_swe_agent_exit_137)
    )
    if post_execution_agent_failure:
        reward = 0.0
        reward_provenance = (
            "gradeable post-execution agent timeout"
            if timeout_error
            else "gradeable post-execution mini-swe-agent exit 137"
        )
    elif (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or float(reward) not in {0.0, 1.0}
    ):
        raise SystemExit(f"row {index} lacks an official binary reward")
    else:
        reward_provenance = "official verifier"
    scoring = trace.get("timing", {}).get("scoring", {})
    start, end = scoring.get("start"), scoring.get("end")
    if post_execution_agent_failure:
        scoring_seconds = None
    elif not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or end <= start:
        raise SystemExit(f"row {index} lacks official scoring timing")
    else:
        scoring_seconds = end - start
    usage = {field: 0 for field in totals}
    provider_errors = []
    for call_index, call in enumerate(calls):
        if call.get("model") != "gpt-5.6-luna":
            raise SystemExit(f"row {index} call {call_index} has a different model")
        if call.get("endpoint") != "/responses":
            raise SystemExit(f"row {index} call {call_index} has a different endpoint")
        sampling = call.get("sampling", {})
        if sampling.get("reasoning_effort") != args.effort or sampling.get("max_tokens") != 32768:
            raise SystemExit(f"row {index} call {call_index} has different sampling")
        call_usage = call.get("usage")
        if call_usage is None:
            error = call.get("error")
            if not isinstance(error, dict):
                raise SystemExit(f"row {index} call {call_index} lacks usage and error")
            status = error.get("status_code")
            if not isinstance(status, int) or not 429 <= status <= 599:
                raise SystemExit(f"row {index} call {call_index} has ungradeable missing usage")
            provider_errors.append({
                "call_index": call_index,
                "type": error.get("type"),
                "status_code": status,
                "usage_charge": "zero; provider returned no inference usage",
            })
            continue
        if not isinstance(call_usage, dict):
            raise SystemExit(f"row {index} call {call_index} has invalid usage")
        for field in totals:
            value = call_usage.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SystemExit(f"row {index} call {call_index} has invalid {field}")
            usage[field] += value
            totals[field] += value
    if usage["reasoning_tokens"] > usage["completion_tokens"]:
        raise SystemExit(f"row {index} reasoning exceeds output tokens")
    inference_calls = len(calls) - len(provider_errors)
    if not 1 <= inference_calls <= 20:
        raise SystemExit(f"row {index} has invalid provider inference calls")
    patch = trace.get("info", {}).get("patch")
    if post_execution_agent_failure:
        patch_bytes = 0
        patch_sha256 = None
    elif not isinstance(patch, str):
        raise SystemExit(f"row {index} lacks a patch string")
    else:
        patch_bytes = len(patch.encode())
        patch_sha256 = hashlib.sha256(patch.encode()).hexdigest()
    cells.append({
        "attempt_number": args.attempt_offset + index,
        "reward": float(reward),
        "reward_provenance": reward_provenance,
        "official_verifier_reached": not post_execution_agent_failure,
        "provider_calls": len(calls),
        "provider_inference_calls": inference_calls,
        "provider_errors": provider_errors,
        "stop_condition": trace.get("stop_condition"),
        "trace_ok": trace.get("ok"),
        "outer_ok": outer.get("ok"),
        "trace_errors": trace_errors,
        "outer_errors": outer.get("errors", []),
        "patch_bytes": patch_bytes,
        "patch_sha256": patch_sha256,
        "scoring_seconds": scoring_seconds,
        "usage": usage,
    })
report = {
    "protocol": "coding-router-swerebench-effort-artifact-v1",
    "valid": True,
    "task_id": args.task,
    "model": "gpt-5.6-luna",
    "effort": args.effort,
    "new_cells": args.expected,
    "attempt_offset": args.attempt_offset,
    "cells": cells,
    "usage": totals,
    "usage_provenance": "exact token counts from pinned verifier Responses traces",
}
args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
'''


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def _execution_phase(name: str) -> ExecutionPhase:
    if name == DEVELOPMENT_PHASE.name:
        return DEVELOPMENT_PHASE
    if name == CONFIRMATION_PHASE.name:
        return CONFIRMATION_PHASE
    if name == POOLED_CONFIRMATION_PHASE.name:
        return POOLED_CONFIRMATION_PHASE
    raise ValueError(f"unknown execution phase: {name!r}")


def _confirmation_authorization(
    fit_output: Path,
    development_audit_path: Path,
    confirmation_corpus_path: Path,
) -> tuple[float, dict[str, object]]:
    """Validate the frozen label-free route before any confirmation outcome exists."""
    required = {
        "development_report": fit_output / "development-report.json",
        "selection_lock": fit_output / "selection-lock.json",
        "route_audit": fit_output / "route-audit.json",
        "routes": fit_output / "confirmation-routes.jsonl",
        "shuffled_routes": fit_output / "confirmation-shuffled-routes.jsonl",
    }
    for label, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"confirmation authorization lacks {label}: {path}")
    report = _read_object(required["development_report"])
    lock = _read_object(required["selection_lock"])
    route_audit = _read_object(required["route_audit"])
    development_audit = _read_object(development_audit_path)
    confirmation_corpus = _read_object(confirmation_corpus_path)
    raw_tasks = confirmation_corpus.get("tasks")
    if not isinstance(raw_tasks, list) or len(raw_tasks) != 200:
        raise ValueError("confirmation corpus does not contain exactly 200 tasks")
    task_ids = []
    for index, raw_task in enumerate(raw_tasks):
        if not isinstance(raw_task, dict) or not isinstance(raw_task.get("task_id"), str):
            raise ValueError(f"confirmation corpus task {index} is invalid")
        task_ids.append(raw_task["task_id"])
    if len(set(task_ids)) != 200:
        raise ValueError("confirmation corpus task identities are not unique")

    false_flags = (
        "target_outcomes_used",
        "deep_swe_outcomes_accessed",
        "confirmation_outcomes_accessed",
    )
    for label, payload in {
        "development report": report,
        "selection lock": lock,
        "route audit": route_audit,
    }.items():
        for flag in false_flags:
            if flag in payload and payload.get(flag) is not False:
                raise ValueError(f"{label} has unsafe {flag}")
    if (
        report.get("development_passed") is not True
        or report.get("confirmation_authorized") is not True
        or report.get("confirmation_routes_written") is not True
    ):
        raise ValueError("external development did not authorize confirmation")
    if (
        development_audit.get("valid") is not True
        or development_audit.get("deep_swe_outcomes_accessed") is not False
        or development_audit.get("target_outcomes_used") is not False
        or float(development_audit.get("retained_task_coverage", 0.0)) < 0.95
    ):
        raise ValueError("development collection audit does not pass isolation gates")
    if _sha256(confirmation_corpus_path) != CONFIRMATION_CORPUS_SHA256:
        raise ValueError("confirmation corpus hash mismatch")
    inputs = report.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("development report lacks content-addressed inputs")
    if (
        inputs.get("collection_audit_sha256") != _sha256(development_audit_path)
        or inputs.get("confirmation_corpus_sha256") != CONFIRMATION_CORPUS_SHA256
        or lock.get("collection_audit_sha256") != _sha256(development_audit_path)
        or lock.get("confirmation_corpus_sha256") != CONFIRMATION_CORPUS_SHA256
    ):
        raise ValueError("fit authorization inputs drifted")
    if route_audit.get("selection_lock_sha256") != _sha256(required["selection_lock"]):
        raise ValueError("selection lock hash mismatch")
    if (
        route_audit.get("confirmation_routes_sha256") != _sha256(required["routes"])
        or route_audit.get("shuffled_routes_sha256")
        != _sha256(required["shuffled_routes"])
        or route_audit.get("fitted_numeric_state_persisted") is not False
    ):
        raise ValueError("sealed confirmation route audit is invalid")
    latency = route_audit.get("latency")
    if not isinstance(latency, dict) or latency.get("passed") is not True:
        raise ValueError("frozen route failed the latency gate")
    for label in ("routes", "shuffled_routes"):
        rows = _read_rows(required[label])
        if len(rows) != 200 or [row.get("task_id") for row in rows] != task_ids:
            raise ValueError(f"{label} do not exactly cover the frozen confirmation corpus")
        for row in rows:
            if row.get("reasoning_effort") not in EFFORTS:
                raise ValueError(f"{label} contain an invalid reasoning effort")
    prior_spend = development_audit.get("rough_cumulative_experiment_spend_usd")
    if (
        isinstance(prior_spend, bool)
        or not isinstance(prior_spend, (int, float))
        or not 0.0 <= float(prior_spend) < 20_000.0
    ):
        raise ValueError("development audit has invalid cumulative spend")
    hashes = {f"{label}_sha256": _sha256(path) for label, path in required.items()}
    hashes["development_audit_sha256"] = _sha256(development_audit_path)
    hashes["confirmation_corpus_sha256"] = CONFIRMATION_CORPUS_SHA256
    return float(prior_spend), hashes


def _pooled_confirmation_authorization(
    fit_output: Path,
    confirmation_manifest_path: Path,
    confirmation_corpus_path: Path,
) -> tuple[float, dict[str, object]]:
    """Validate the pooled route and all null routes before provider execution."""
    required = {
        "route_audit": fit_output / "route-audit.json",
        "freeze_lock": fit_output / "freeze-lock.json",
        "routes": fit_output / "confirmation-routes.jsonl",
        "null_routes": fit_output / "confirmation-null-routes.jsonl",
    }
    for label, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"pooled confirmation lacks {label}: {path}")
    expected_hashes = {
        "route_audit": POOLED_CONFIRMATION_ROUTE_AUDIT_SHA256,
        "freeze_lock": POOLED_CONFIRMATION_FREEZE_LOCK_SHA256,
        "routes": POOLED_CONFIRMATION_ROUTES_SHA256,
        "null_routes": POOLED_CONFIRMATION_NULL_ROUTES_SHA256,
    }
    for label, expected in expected_hashes.items():
        if _sha256(required[label]) != expected:
            raise ValueError(f"frozen pooled {label} hash mismatch")
    if _sha256(confirmation_corpus_path) != POOLED_CONFIRMATION_CORPUS_SHA256:
        raise ValueError("pooled confirmation corpus hash mismatch")
    if _sha256(confirmation_manifest_path) != POOLED_CONFIRMATION_MANIFEST_SHA256:
        raise ValueError("pooled confirmation manifest hash mismatch")

    corpus = _read_object(confirmation_corpus_path)
    manifest = _read_object(confirmation_manifest_path)
    audit = _read_object(required["route_audit"])
    freeze = _read_object(required["freeze_lock"])
    raw_tasks = corpus.get("tasks")
    if not isinstance(raw_tasks, list) or len(raw_tasks) != 200:
        raise ValueError("pooled confirmation corpus must contain exactly 200 tasks")
    task_ids = [
        task.get("task_id") if isinstance(task, dict) else None for task in raw_tasks
    ]
    if not all(isinstance(task_id, str) for task_id in task_ids):
        raise ValueError("pooled confirmation corpus contains an invalid task identity")
    if len(set(task_ids)) != 200:
        raise ValueError("pooled confirmation task identities are not unique")

    if (
        manifest.get("valid") is not True
        or manifest.get("confirmation_tasks") != 200
        or manifest.get("confirmation_tasks_sha256")
        != POOLED_CONFIRMATION_CORPUS_SHA256
        or manifest.get("target_repository_overlap") != 0
        or manifest.get("target_normalized_prompt_overlap") != 0
        or manifest.get("development_repository_overlap") != 0
        or manifest.get("development_task_id_overlap") != 0
        or manifest.get("development_normalized_prompt_overlap") != 0
        or manifest.get("target_reward_fields_accessed") is not False
        or manifest.get("target_cost_fields_accessed") is not False
    ):
        raise ValueError("pooled confirmation cohort failed isolation gates")
    if (
        audit.get("valid") is not True
        or audit.get("selected_candidate") != POOLED_SELECTED_CANDIDATE
        or audit.get("confirmation_tasks") != 200
        or audit.get("null_count") != 128
        or audit.get("null_unique_route_hashes") != 128
        or audit.get("route_latency_p95_ms", 5.0) >= 5.0
        or audit.get("target_outcomes_used") is not False
        or audit.get("deep_swe_outcomes_accessed") is not False
        or audit.get("internet_access") is not False
        or audit.get("fitted_numeric_router_state_persisted") is not False
    ):
        raise ValueError("pooled confirmation route audit is unsafe")
    if (
        audit.get("confirmation_tasks_sha256")
        != POOLED_CONFIRMATION_CORPUS_SHA256
        or audit.get("confirmation_manifest_sha256")
        != POOLED_CONFIRMATION_MANIFEST_SHA256
        or audit.get("confirmation_routes_sha256")
        != POOLED_CONFIRMATION_ROUTES_SHA256
        or audit.get("confirmation_null_routes_sha256")
        != POOLED_CONFIRMATION_NULL_ROUTES_SHA256
    ):
        raise ValueError("pooled confirmation route audit inputs drifted")
    if (
        freeze.get("valid") is not True
        or freeze.get("selected_candidate") != POOLED_SELECTED_CANDIDATE
        or freeze.get("provider_calls_before_freeze") != 0
        or freeze.get("target_outcomes_used") is not False
        or freeze.get("fitted_numeric_router_state_persisted") is not False
        or freeze.get("confirmation_tasks_sha256")
        != POOLED_CONFIRMATION_CORPUS_SHA256
        or freeze.get("confirmation_routes_sha256")
        != POOLED_CONFIRMATION_ROUTES_SHA256
        or freeze.get("confirmation_null_routes_sha256")
        != POOLED_CONFIRMATION_NULL_ROUTES_SHA256
        or freeze.get("route_audit_sha256")
        != POOLED_CONFIRMATION_ROUTE_AUDIT_SHA256
    ):
        raise ValueError("pooled confirmation freeze lock is unsafe")

    allowed_arms = {"luna-high", "luna-max"}
    routes = _read_rows(required["routes"])
    if len(routes) != 200 or [row.get("task_id") for row in routes] != task_ids:
        raise ValueError("pooled real routes do not exactly cover the corpus")
    if any(
        row.get("arm") not in allowed_arms
        or row.get("target_outcomes_used") is not False
        for row in routes
    ):
        raise ValueError("pooled real routes contain an unsafe decision")
    null_routes = _read_rows(required["null_routes"])
    if len(null_routes) != 128 * 200:
        raise ValueError("pooled null routes do not contain 128 complete routes")
    for null_index in range(128):
        rows = null_routes[null_index * 200 : (null_index + 1) * 200]
        if [row.get("task_id") for row in rows] != task_ids:
            raise ValueError(f"pooled null route {null_index} does not cover the corpus")
        if any(
            row.get("null_index") != null_index
            or row.get("arm") not in allowed_arms
            or row.get("target_outcomes_used") is not False
            for row in rows
        ):
            raise ValueError(f"pooled null route {null_index} is unsafe")

    hashes = {f"{label}_sha256": _sha256(path) for label, path in required.items()}
    hashes["confirmation_manifest_sha256"] = POOLED_CONFIRMATION_MANIFEST_SHA256
    hashes["confirmation_corpus_sha256"] = POOLED_CONFIRMATION_CORPUS_SHA256
    hashes["null_route_count"] = 128
    return POOLED_PRIOR_SPEND_USD, hashes


def _max_concurrency(phase: ExecutionPhase) -> int:
    return 200 if phase is POOLED_CONFIRMATION_PHASE else 100


def _capacity() -> int:
    paginator = Sandbox.list(limit=100)
    count = 0
    while True:
        count += len(paginator.next_items())
        if not paginator.has_next:
            return count


def _docker_image(alias: str) -> str:
    prefix = "prime/primeintellect/"
    if not alias.startswith(prefix):
        raise ValueError(f"unexpected frozen image alias: {alias!r}")
    image = "docker.io/swerebenchv2/" + alias.removeprefix(prefix)
    if not IMAGE_PATTERN.fullmatch(image):
        raise ValueError(f"unsafe frozen image: {image!r}")
    return image


def _effort_order(task_index: int) -> tuple[str, ...]:
    offset = task_index % len(EFFORTS)
    return EFFORTS[offset:] + EFFORTS[:offset]


def _new_rollouts(
    task_id: str,
    effort: str,
    *,
    reuse_smoke: bool = True,
) -> tuple[int, int]:
    if reuse_smoke and task_id in REUSED_TASKS and effort in SMOKE_ARCHIVE_SHA256:
        return 1, 1
    return 2, 0


def _config(task_id: str, effort: str, rollouts: int, output_dir: str) -> str:
    if not TASK_PATTERN.fullmatch(task_id):
        raise ValueError(f"unsafe frozen task id: {task_id!r}")
    return f'''model = "{MODEL}"
num_tasks = 1
num_rollouts = {rollouts}
shuffle = false
max_concurrent = 2
verbose = false
rich = false
server = false
push = false
output_dir = "{output_dir}"

[client]
type = "eval"
base_url = "https://api.openai.com/v1"
api_key_var = "OPENAI_API_KEY"

[sampling]
temperature = 1.0
reasoning_effort = "{effort}"
max_tokens = 32768

[env]
max_concurrent_agents = 1

[env.taskset]
id = "swerebench-v2-v1"
dataset_name = "PrimeIntellect/SWE-rebench-V2-Filtered-Verified"
split = "train"
filter_fn = "lambda row: row['instance_id'] == '{task_id}'"

[env.agent]
max_turns = 20
max_output_tokens = 131072

[env.agent.harness]
id = "mini_swe_agent"
version = "2.4.5"

[env.agent.runtime]
type = "docker"

[env.agent.timeout]
setup = 900
rollout = 900
finalize = 300
scoring = 900

[env.retries]
max_retries = 0
'''


def _run(
    sandbox: Sandbox,
    command: str,
    *,
    timeout: float,
    check: bool = True,
) -> CommandResult:
    result = sandbox.commands.run(command, timeout=timeout)
    if check and result.exit_code:
        raise RuntimeError(
            f"remote command failed exit={result.exit_code} stderr={result.stderr[-1000:]!r}"
        )
    return result


def _run_durable_eval(
    sandbox: Sandbox,
    command: str,
    *,
    effort: str,
    exit_status_path: str,
    state: dict[str, Any],
    state_path: Path,
    attempt: dict[str, Any],
    timeout: float,
    poll_interval: float = 10.0,
) -> tuple[CommandResult, Sandbox]:
    """Run one scientific command once and poll its persisted exit status.

    Long-lived E2B output streams have intermittently failed at the HTTP/2
    control plane while the remote command kept running. A short launcher starts
    a detached wrapper exactly once, while remote PID and atomic exit markers
    make completion recoverable without ever issuing the scientific command a
    second time.
    """
    temporary_status_path = f"{exit_status_path}.tmp"
    wrapper_path = f"{exit_status_path}.wrapper.sh"
    pid_path = f"{exit_status_path}.pid"
    temporary_pid_path = f"{pid_path}.tmp"
    lock_path = f"{exit_status_path}.launch-lock"
    log_path = f"{exit_status_path}.log"
    wrapped = (
        "set +e\n"
        f"{command}\n"
        "router_eval_status=$?\n"
        f"printf '%s\\n' \"$router_eval_status\" > {temporary_status_path}\n"
        f"mv {temporary_status_path} {exit_status_path}\n"
        'exit "$router_eval_status"'
    )
    sandbox.files.write(wrapper_path, wrapped)
    launcher = (
        "set -eu\n"
        f"if mkdir {lock_path} 2>/dev/null; then\n"
        f"  nohup bash {wrapper_path} > {log_path} 2>&1 </dev/null &\n"
        "  router_eval_pid=$!\n"
        f"  printf '%s\\n' \"$router_eval_pid\" > {temporary_pid_path}\n"
        f"  mv {temporary_pid_path} {pid_path}\n"
        "else\n"
        f"  test -s {pid_path}\n"
        "fi\n"
        f"cat {pid_path}"
    )
    launch_result = sandbox.commands.run(launcher, timeout=120)
    pid = int(launch_result.stdout.strip())
    if pid <= 1:
        raise ValueError(f"invalid durable eval pid: {pid}")
    process = {
        "pid": pid,
        "exit_status_path": exit_status_path,
        "pid_path": pid_path,
        "wrapper_path": wrapper_path,
        "scientific_command_starts": 1,
    }
    processes = attempt.setdefault("effort_processes", {})
    if not isinstance(processes, dict):
        raise ValueError("invalid durable effort process state")
    processes[effort] = process
    _write_json(state_path, state)

    deadline = time.monotonic() + timeout
    active = sandbox
    missing_pid_polls = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"durable eval timed out effort={effort} pid={pid}; "
                "the scientific command was not rerun"
            )
        try:
            if active.files.exists(exit_status_path, request_timeout=60):
                raw_status = active.files.read(exit_status_path).strip()
                exit_code = int(raw_status)
                if exit_code < 0 or exit_code > 255:
                    raise ValueError(f"invalid durable eval exit status: {raw_status!r}")
                process["exit_code"] = exit_code
                process["completed"] = True
                _write_json(state_path, state)
                return (
                    CommandResult(
                        stdout="",
                        stderr="",
                        exit_code=exit_code,
                        error=None,
                    ),
                    active,
                )
            process_result = active.commands.run(
                (
                    f"if kill -0 {pid} 2>/dev/null; then "
                    "printf running; else printf stopped; fi"
                ),
                timeout=60,
            )
            running = process_result.stdout.strip() == "running"
        except Exception as error:  # noqa: BLE001 - reconnect exact remote PID state
            logger.warning(
                "durable eval poll reconnect effort=%s pid=%d error=%r",
                effort,
                pid,
                error,
            )
            poll_errors = process.setdefault("poll_errors", [])
            if isinstance(poll_errors, list):
                poll_errors.append(repr(error))
            _write_json(state_path, state)
            active = Sandbox.connect(sandbox.sandbox_id, request_timeout=60)
        else:
            if not running:
                if active.files.exists(exit_status_path, request_timeout=60):
                    continue
                missing_pid_polls += 1
                process["missing_pid_polls"] = missing_pid_polls
                _write_json(state_path, state)
                if missing_pid_polls <= 5:
                    time.sleep(min(1.0, remaining))
                    continue
                raise RuntimeError(
                    f"durable eval pid={pid} ended without exit marker; "
                    "the scientific command was not rerun"
                )
            missing_pid_polls = 0
        time.sleep(min(poll_interval, remaining))


def _sync(sandbox: Sandbox, remote: str, local: Path) -> None:
    stream = sandbox.files.read(
        remote,
        format="stream",
        request_timeout=180,
        stream_idle_timeout=180,
        gzip=True,
    )
    temporary = local.with_suffix(local.suffix + ".tmp")
    with stream, temporary.open("wb") as handle:
        for chunk in stream:
            handle.write(chunk)
    temporary.replace(local)


def _verify_completed_effort(task_dir: Path, effort: str, payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    archive = task_dir / f"{effort}.tar.gz"
    report = task_dir / f"{effort}.report.json"
    if not archive.is_file() or not report.is_file():
        return False
    if _sha256(archive) != payload.get("archive_sha256"):
        return False
    if _sha256(report) != payload.get("report_sha256"):
        return False
    report_data = _read_object(report)
    return bool(report_data.get("valid")) and report_data.get("effort") == effort


def _task_complete(task_dir: Path, state: dict[str, Any]) -> bool:
    efforts = state.get("efforts", {})
    return isinstance(efforts, dict) and all(
        _verify_completed_effort(task_dir, effort, efforts.get(effort))
        for effort in EFFORTS
    )


def _task_excluded(state: dict[str, Any]) -> bool:
    """Return whether a task has an audited infrastructure-cell exclusion."""
    exclusion = state.get("exclusion")
    return (
        state.get("stage") == "excluded-infrastructure"
        and isinstance(exclusion, dict)
        and exclusion.get("scope") == "whole-task"
        and isinstance(exclusion.get("effort"), str)
        and isinstance(exclusion.get("reason"), str)
        and isinstance(exclusion.get("evidence_sha256"), str)
        and isinstance(exclusion.get("usage"), dict)
        and isinstance(exclusion.get("provider_calls"), int)
        and isinstance(exclusion.get("observed_scientific_cells"), int)
        and exclusion.get("scientific_cells_rerun") == 0
    )


def _update_summary(
    root: Path,
    total_tasks: int,
    *,
    protocol: str = PROTOCOL,
    prior_spend_usd: float = DEFAULT_PRIOR_SPEND_USD,
) -> None:
    with STATE_LOCK:
        states = list((root / "tasks").glob("*/state.json"))
        completed_efforts = 0
        completed_new_cells = 0
        reused_cells = 0
        complete_tasks = 0
        excluded_tasks = 0
        failed_tasks = 0
        provider_calls = 0
        usage = {
            "prompt_tokens": 0,
            "cached_input_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
        }
        for state_path in states:
            state = _read_object(state_path)
            efforts = state.get("efforts", {})
            if isinstance(efforts, dict):
                for payload in efforts.values():
                    if not isinstance(payload, dict):
                        continue
                    completed_efforts += 1
                    completed_new_cells += int(payload.get("new_cells", 0))
                    reused_cells += int(payload.get("reused_cells", 0))
                    provider_calls += int(payload.get("provider_calls", 0))
                    payload_usage = payload.get("usage", {})
                    if isinstance(payload_usage, dict):
                        for field in usage:
                            usage[field] += int(payload_usage.get(field, 0))
            if _task_excluded(state):
                exclusion = state["exclusion"]
                excluded_usage = exclusion["usage"]
                completed_new_cells += int(exclusion["observed_scientific_cells"])
                provider_calls += int(exclusion["provider_calls"])
                for field in usage:
                    usage[field] += int(excluded_usage.get(field, 0))
            if state.get("stage") == "complete":
                complete_tasks += 1
            if _task_excluded(state):
                excluded_tasks += 1
            if state.get("stage") == "failed":
                failed_tasks += 1
        input_rate, cached_input_rate, output_rate = MODEL_PRICES_PER_MTOK[MODEL]
        cost = (
            usage["prompt_tokens"] * input_rate / 1_000_000
            + usage["cached_input_tokens"] * cached_input_rate / 1_000_000
            + usage["completion_tokens"] * output_rate / 1_000_000
        )
        _write_json(
            root / "progress.json",
            {
                "protocol": protocol,
                "total_tasks": total_tasks,
                "expected_cells": total_tasks * len(EFFORTS) * 2,
                "complete_tasks": complete_tasks,
                "excluded_tasks": excluded_tasks,
                "retained_task_coverage": (total_tasks - excluded_tasks) / total_tasks,
                "failed_tasks": failed_tasks,
                "completed_efforts": completed_efforts,
                "completed_new_cells": completed_new_cells,
                "reused_smoke_cells": reused_cells,
                "completed_scientific_cells": completed_new_cells + reused_cells,
                "provider_calls": provider_calls,
                "usage": usage,
                "matrix_cost_usd": cost,
                "cost_provenance": "trace-derived frozen list-price estimate",
                "rough_cumulative_experiment_spend_usd": prior_spend_usd + cost,
            },
        )


def _run_task(
    root: Path,
    task_index: int,
    row: dict[str, Any],
    api_key: str,
    total_tasks: int,
    phase: ExecutionPhase = DEVELOPMENT_PHASE,
    prior_spend_usd: float = DEFAULT_PRIOR_SPEND_USD,
) -> None:
    task_id = str(row["task_id"])
    image = _docker_image(str(row["image_name"]))
    task_dir = root / "tasks" / f"{task_index:04d}"
    task_dir.mkdir(parents=True, exist_ok=True)
    state_path = task_dir / "state.json"
    if state_path.is_file():
        state = _read_object(state_path)
        if (
            state.get("protocol") != phase.protocol
            or state.get("task_id") != task_id
            or state.get("image") != image
        ):
            raise ValueError(f"task state identity drift at {task_dir}")
        if _task_excluded(state):
            return
        if _task_complete(task_dir, state):
            return
    else:
        state = {
            "protocol": phase.protocol,
            "task_index": task_index,
            "task_id": task_id,
            "image": image,
            "effort_order": list(_effort_order(task_index)),
            "efforts": {},
            "sandbox_attempts": [],
            "stage": "pending",
        }
        _write_json(state_path, state)

    effort_state = state["efforts"]
    if not isinstance(effort_state, dict):
        raise ValueError(f"invalid effort state for {task_id}")
    missing = [
        effort
        for effort in _effort_order(task_index)
        if not _verify_completed_effort(task_dir, effort, effort_state.get(effort))
    ]
    if not missing:
        state["stage"] = "complete"
        _write_json(state_path, state)
        return

    sandbox = Sandbox.create(
        TEMPLATE_NAME,
        timeout=6 * 3_600,
        secure=True,
        allow_internet_access=True,
        envs={"OPENAI_API_KEY": api_key},
        metadata={
            "owner": phase.metadata_owner,
            "phase": phase.metadata_phase,
            "task_index": str(task_index),
            "task_id": task_id,
        },
    )
    attempts = state["sandbox_attempts"]
    if not isinstance(attempts, list):
        raise ValueError(f"invalid sandbox attempt state for {task_id}")
    attempt: dict[str, Any] = {
        "sandbox_id": sandbox.sandbox_id,
        "missing_efforts": missing,
        "terminated": False,
    }
    attempts.append(attempt)
    state["stage"] = "running"
    _write_json(state_path, state)
    verified = False
    remote_root = f"/home/user/router-v40-{phase.remote_segment}/{task_index:04d}"
    try:
        _run(sandbox, f"mkdir -p {remote_root}/runtime", timeout=120)
        _run(
            sandbox,
            (
                "test \"$(sha256sum /opt/coding-router/"
                "swerebench-docker-adapter-report.json | cut -d' ' -f1)\" = "
                f"{DOCKER_ADAPTER_REPORT_SHA256}"
            ),
            timeout=120,
        )
        _run(
            sandbox,
            (
                "test \"$(sha256sum /opt/coding-router/"
                "verifiers-responses-adapter-report.json | cut -d' ' -f1)\" = "
                f"{RESPONSES_ADAPTER_REPORT_SHA256}"
            ),
            timeout=120,
        )
        _run(
            sandbox,
            (
                f"cp /opt/coding-router/*-adapter-report.json {remote_root}/runtime/ "
                f"&& sha256sum {remote_root}/runtime/* > {remote_root}/runtime/sha256sums"
            ),
            timeout=120,
        )
        _run(sandbox, f"sudo docker pull {image}", timeout=1_800)
        image_id = _run(
            sandbox,
            f"sudo docker image inspect {image} --format '{{{{.Id}}}}'",
            timeout=120,
        ).stdout.strip()
        state["docker_image_id"] = image_id
        sandbox.files.write(f"{remote_root}/validate.py", REMOTE_VALIDATOR)

        for effort in missing:
            rollouts, attempt_offset = _new_rollouts(
                task_id,
                effort,
                reuse_smoke=phase.reuse_smoke,
            )
            output_dir = f"{remote_root}/{effort}"
            config_path = f"{remote_root}/{effort}.toml"
            report_path = f"{remote_root}/{effort}.report.json"
            archive_path = f"{remote_root}/{effort}.tar.gz"
            sandbox.files.write(
                config_path,
                _config(task_id, effort, rollouts, output_dir),
            )
            state["stage"] = f"running-{effort}"
            _write_json(state_path, state)
            eval_result, sandbox = _run_durable_eval(
                sandbox,
                f"cd /opt/verifiers && sudo -E .venv/bin/eval @ {config_path}",
                effort=effort,
                exit_status_path=f"{remote_root}/{effort}.eval-exit-status",
                state=state,
                state_path=state_path,
                attempt=attempt,
                timeout=3 * 3_600,
            )
            _run(
                sandbox,
                (
                    f"sudo /opt/verifiers/.venv/bin/python {remote_root}/validate.py "
                    f"--traces {output_dir}/traces.jsonl --task {task_id} "
                    f"--effort {effort} --expected {rollouts} "
                    f"--attempt-offset {attempt_offset} --output {report_path}"
                ),
                timeout=120,
            )
            _run(
                sandbox,
                (
                    f"sudo tar -C {remote_root} -czf {archive_path} "
                    f"runtime {effort}.toml {effort}.report.json {effort}"
                ),
                timeout=300,
            )
            local_archive = task_dir / f"{effort}.tar.gz"
            local_report = task_dir / f"{effort}.report.json"
            _sync(sandbox, archive_path, local_archive)
            local_report.write_text(
                sandbox.files.read(report_path), encoding="utf-8"
            )
            report = _read_object(local_report)
            if report.get("valid") is not True or report.get("task_id") != task_id:
                raise ValueError(f"downloaded report failed validation for {task_id}/{effort}")
            cells = report.get("cells", [])
            provider_calls = sum(
                int(cell.get("provider_calls", 0))
                for cell in cells
                if isinstance(cell, dict)
            )
            reused = 1 if attempt_offset else 0
            effort_state[effort] = {
                "archive_sha256": _sha256(local_archive),
                "report_sha256": _sha256(local_report),
                "new_cells": rollouts,
                "reused_cells": reused,
                "reused_smoke_archive_sha256": SMOKE_ARCHIVE_SHA256.get(effort)
                if reused
                else None,
                "provider_calls": provider_calls,
                "usage": report.get("usage"),
                "eval_exit_code": eval_result.exit_code,
                "sandbox_id": sandbox.sandbox_id,
            }
            state["stage"] = f"completed-{effort}"
            _write_json(state_path, state)
            _update_summary(
                root,
                total_tasks,
                protocol=phase.protocol,
                prior_spend_usd=prior_spend_usd,
            )
            logger.info(
                "task=%d/%d id=%s effort=%s new_cells=%d reused=%d",
                task_index + 1,
                total_tasks,
                task_id,
                effort,
                rollouts,
                reused,
            )
        state["stage"] = "complete"
        verified = True
    except Exception as error:
        state["stage"] = "failed"
        state["error"] = repr(error)
        attempt["error"] = repr(error)
        logger.exception("task failed index=%d id=%s", task_index, task_id)
        raise
    finally:
        if verified:
            sandbox.kill()
            attempt["terminated"] = True
            state["sandbox_terminated"] = True
        _write_json(state_path, state)
        _update_summary(
            root,
            total_tasks,
            protocol=phase.protocol,
            prior_spend_usd=prior_spend_usd,
        )


def execute(
    root: Path,
    corpus_path: Path,
    *,
    concurrency: int,
    limit_tasks: int | None,
    phase_name: str = "development",
    fit_output: Path | None = None,
    development_audit: Path | None = None,
) -> None:
    """Validate the frozen launch and execute or resume every missing task."""
    phase = _execution_phase(phase_name)
    if _sha256(corpus_path) != phase.corpus_sha256:
        raise ValueError(f"{phase.name} corpus hash mismatch")
    authorization: dict[str, object] | None = None
    prior_spend_usd = DEFAULT_PRIOR_SPEND_USD
    if phase is CONFIRMATION_PHASE:
        if limit_tasks is not None:
            raise ValueError("confirmation does not allow a task limit")
        if fit_output is None or development_audit is None:
            raise ValueError("confirmation requires fit output and development audit")
        prior_spend_usd, authorization = _confirmation_authorization(
            fit_output,
            development_audit,
            corpus_path,
        )
    elif phase is POOLED_CONFIRMATION_PHASE:
        if limit_tasks is not None:
            raise ValueError("pooled confirmation does not allow a task limit")
        if fit_output is None or development_audit is None:
            raise ValueError(
                "pooled confirmation requires frozen routes and cohort manifest"
            )
        prior_spend_usd, authorization = _pooled_confirmation_authorization(
            fit_output,
            development_audit,
            corpus_path,
        )
    elif fit_output is not None or development_audit is not None:
        raise ValueError("development does not accept confirmation authorization inputs")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is unavailable")
    max_concurrency = _max_concurrency(phase)
    if concurrency < 1 or concurrency > max_concurrency:
        raise ValueError(f"concurrency must be between 1 and {max_concurrency}")
    if not Template.exists(TEMPLATE_NAME):
        raise RuntimeError(f"required E2B template is absent: {TEMPLATE_NAME}")
    active = _capacity()
    if active + concurrency > E2B_ACCOUNT_CAP:
        raise RuntimeError(
            f"E2B capacity is insufficient: active={active} launch={concurrency} "
            f"cap={E2B_ACCOUNT_CAP}"
        )
    corpus = _read_object(corpus_path)
    rows = corpus.get("tasks")
    if not isinstance(rows, list) or len(rows) != 200:
        raise ValueError("development corpus does not contain exactly 200 tasks")
    selected = rows[:limit_tasks] if limit_tasks is not None else rows
    root.mkdir(parents=True, exist_ok=True)
    (root / "tasks").mkdir(exist_ok=True)
    launch_path = root / "launch.json"
    launch: dict[str, object] = {
        "protocol": phase.protocol,
        "corpus_path": str(corpus_path.resolve()),
        "corpus_sha256": phase.corpus_sha256,
        "template": TEMPLATE_NAME,
        "template_id": TEMPLATE_ID,
        "template_build_id": TEMPLATE_BUILD_ID,
        "model": MODEL,
        "efforts": list(EFFORTS),
        "attempts_per_effort": 2,
        "tasks": len(selected),
        "expected_cells": len(selected) * len(EFFORTS) * 2,
        "reused_smoke_cells": (4 if len(selected) >= 2 else 2)
        if phase.reuse_smoke
        else 0,
        "concurrency": concurrency,
        "active_e2b_before": active,
        "e2b_account_cap": E2B_ACCOUNT_CAP,
        "cost_ceiling_usd": 20_000.0,
        "prior_spend_usd": prior_spend_usd,
        "deep_swe_outcomes_accessed": False,
        "model_persisted": False,
    }
    if phase.reuse_smoke:
        launch["smoke_report_sha256"] = SMOKE_REPORT_SHA256
        launch["smoke_archive_sha256"] = SMOKE_ARCHIVE_SHA256
    elif authorization is not None:
        launch["phase"] = phase.name
        launch["authorization"] = authorization
        launch["confirmation_outcomes_accessed_before_launch"] = False
    elif EXTERNAL_AUTHORIZATION is not None:
        launch["phase"] = "externally-authorized-confirmation"
        launch["authorization"] = EXTERNAL_AUTHORIZATION
        launch["confirmation_outcomes_accessed_before_launch"] = False
    if launch_path.is_file():
        prior_launch = _read_object(launch_path)
        operational = {"active_e2b_before", "concurrency"}
        frozen_prior = {
            key: value for key, value in prior_launch.items() if key not in operational
        }
        frozen_resume = {
            key: value for key, value in launch.items() if key not in operational
        }
        if frozen_prior != frozen_resume:
            raise ValueError("resume launch manifest differs from the frozen experiment")
    else:
        _write_json(launch_path, launch)
    _update_summary(
        root,
        len(selected),
        protocol=phase.protocol,
        prior_spend_usd=prior_spend_usd,
    )
    errors: list[Exception] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                _run_task,
                root,
                index,
                row,
                api_key,
                len(selected),
                phase,
                prior_spend_usd,
            ): index
            for index, raw_row in enumerate(selected)
            for row in [_read_row(raw_row, index)]
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:  # noqa: BLE001 - isolate task workers
                errors.append(error)
    _update_summary(
        root,
        len(selected),
        protocol=phase.protocol,
        prior_spend_usd=prior_spend_usd,
    )
    if errors:
        raise RuntimeError(f"{len(errors)} task workers failed; inspect task states")


def _read_row(value: object, index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"corpus task {index} is not an object")
    return value


def main() -> None:
    """Parse command line arguments and run the external development matrix."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=25)
    parser.add_argument("--limit-tasks", type=int)
    parser.add_argument(
        "--phase",
        choices=(
            DEVELOPMENT_PHASE.name,
            CONFIRMATION_PHASE.name,
            POOLED_CONFIRMATION_PHASE.name,
        ),
        default=DEVELOPMENT_PHASE.name,
    )
    parser.add_argument("--fit-output", type=Path)
    parser.add_argument("--development-audit", type=Path)
    args = parser.parse_args()
    execute(
        args.root,
        args.corpus,
        concurrency=args.concurrency,
        limit_tasks=args.limit_tasks,
        phase_name=args.phase,
        fit_output=args.fit_output,
        development_audit=args.development_audit,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
