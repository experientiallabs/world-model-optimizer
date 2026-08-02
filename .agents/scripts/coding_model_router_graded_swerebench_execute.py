"""Run the frozen graded SWE-rebench development matrix on E2B."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, NamedTuple

import coding_model_router_swerebench_execute as runner
from e2b import Sandbox, Template

logger = logging.getLogger("coding-router-graded-swerebench-execute")

PROTOCOL = "coding-router-graded-swerebench-development-execution-v1"
PHASE_NAME = "development"
EXPECTED_TASKS = 673
REMOTE_SEGMENT = "development"
METADATA_PHASE = "graded-swerebench-development"
EXTERNAL_AUTHORIZATION: dict[str, Any] | None = None
TEMPLATE_NAME = "deepswe-router-responses-v2"
TEMPLATE_ID = "j1a2bxbpllu3rp84b4qj"
TEMPLATE_BUILD_ID = "e971c040-95bd-45c1-89ee-fb597bf75671"
VERIFIERS_COMMIT = "f6e420b9908ae14d625f079881f13c15011ee1c9"
CORPUS_SHA256 = "48d88436a083b66972c25cd7d9439fd149c95bcf9caded2bab7f3b6453aea3d5"
VERIFIER_TASKS_SHA256 = (
    "bebfbf48f3d0b6f0fca6715c39dffb17c2bec44b52780ddfac7d812f0f3673f8"
)
TASKSET_SOURCE_SHA256 = (
    "a2790c3f296a28f40eb8732d68c091cc7b9899e08916aedec6b2b53a644f7b3e"
)
TASKSET_PATCHED_SHA256 = (
    "cf920fb55d6704da9ba0b6fe7cf676fdac8ec1aeb719c3640f713fdbf7ad0cce"
)
TASKSET_REMOTE = (
    "/opt/verifiers/.venv/lib/python3.12/site-packages/swerebench_v2_v1/taskset.py"
)
DOCKER_ADAPTER_REPORT_SHA256 = runner.DOCKER_ADAPTER_REPORT_SHA256
RESPONSES_ADAPTER_REPORT_SHA256 = runner.RESPONSES_ADAPTER_REPORT_SHA256
PRIOR_SPEND_USD = 3_025.10805955
COST_CEILING_USD = 20_000.0
E2B_ACCOUNT_CAP = 1_000
MAX_CONCURRENCY = 100
STATE_LOCK = threading.Lock()
TASK_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+$")
IMAGE_PATTERN = re.compile(r"^docker\.io/swerebenchv2/[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+$")


class Arm(NamedTuple):
    """One frozen model by reasoning-effort action."""

    name: str
    model: str
    effort: str


ARMS = (
    Arm("luna-low", "gpt-5.6-luna", "low"),
    Arm("luna-medium", "gpt-5.6-luna", "medium"),
    Arm("luna-high", "gpt-5.6-luna", "high"),
    Arm("luna-xhigh", "gpt-5.6-luna", "xhigh"),
    Arm("luna-max", "gpt-5.6-luna", "max"),
    Arm("sol-max", "gpt-5.6-sol", "max"),
)
ARM_BY_NAME = {arm.name: arm for arm in ARMS}
MODEL_PRICES_PER_MTOK = {
    "gpt-5.6-luna": (1.0, 0.1, 6.0),
    "gpt-5.6-sol": (5.0, 0.5, 30.0),
}

REMOTE_VALIDATOR = r'''"""Audit one graded SWE-rebench trace and write a compact report."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--traces", type=Path, required=True)
parser.add_argument("--task", required=True)
parser.add_argument("--arm", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--effort", required=True)
parser.add_argument("--f2p-total", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

records = [
    json.loads(line)
    for line in args.traces.read_bytes().split(b"\n")
    if line.strip()
]
if len(records) != 1:
    raise SystemExit(f"expected one outer row, found {len(records)}")
outer = records[0]
traces = outer.get("traces")
if not isinstance(traces, list) or len(traces) != 1:
    raise SystemExit("outer row lacks exactly one official trace")
trace = traces[0]
task_data = trace.get("task", {}).get("data", {})
if task_data.get("name") != args.task:
    raise SystemExit("trace has a different task")
if len(task_data.get("fail_to_pass") or []) != args.f2p_total:
    raise SystemExit("trace has a different fail-to-pass denominator")
if trace.get("verifiers", {}).get("commit") != "f6e420b9908ae14d625f079881f13c15011ee1c9":
    raise SystemExit("trace has a different verifier commit")
calls = trace.get("calls")
if not isinstance(calls, list) or not calls or len(calls) > 40:
    raise SystemExit("trace has invalid provider calls")

reward = trace.get("rewards", {}).get("solved", {}).get("score")
trace_errors = trace.get("errors", [])
timeout_error = isinstance(trace_errors, list) and any(
    isinstance(error, dict)
    and error.get("type") == "HarnessError"
    and isinstance(error.get("message"), str)
    and error["message"].startswith("agent timeout:")
    for error in trace_errors
)
exit_137 = isinstance(trace_errors, list) and any(
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
    and (timeout_error or exit_137)
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
    or not 0.0 <= float(reward) <= 1.0
):
    raise SystemExit("trace lacks an official graded reward")
else:
    reward_provenance = "official graded F2P verifier"

f2p_passed_float = float(reward) * args.f2p_total
f2p_passed = round(f2p_passed_float)
if abs(f2p_passed_float - f2p_passed) > 1e-6:
    raise SystemExit("graded reward is inconsistent with the F2P denominator")

scoring = trace.get("timing", {}).get("scoring", {})
start, end = scoring.get("start"), scoring.get("end")
if post_execution_agent_failure:
    scoring_seconds = None
elif not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or end <= start:
    raise SystemExit("trace lacks official scoring timing")
else:
    scoring_seconds = end - start

usage = {field: 0 for field in (
    "prompt_tokens", "cached_input_tokens", "completion_tokens", "reasoning_tokens"
)}
provider_errors = []
estimated_usage_calls = []
for call_index, call in enumerate(calls):
    if call.get("model") != args.model:
        raise SystemExit(f"call {call_index} has a different model")
    if call.get("endpoint") != "/responses":
        raise SystemExit(f"call {call_index} has a different endpoint")
    sampling = call.get("sampling", {})
    if sampling.get("reasoning_effort") != args.effort or sampling.get("max_tokens") != 32768:
        raise SystemExit(f"call {call_index} has different sampling")
    call_usage = call.get("usage")
    if call_usage is None:
        error = call.get("error")
        status = error.get("status_code") if isinstance(error, dict) else None
        if isinstance(status, int) and 400 <= status <= 599:
            provider_errors.append({
                "call_index": call_index,
                "type": error.get("type"),
                "status_code": status,
                "usage_charge": "zero; provider rejected the request before inference",
            })
            continue
        serialized = json.dumps(call, sort_keys=True, separators=(",", ":"))
        estimated_input = 4096 + max(1, (len(serialized.encode()) + 3) // 4)
        usage["prompt_tokens"] += estimated_input
        estimated_usage_calls.append({
            "call_index": call_index,
            "input_tokens": estimated_input,
            "output_tokens": 0,
            "method": "4096 token allowance plus serialized trace bytes divided by four",
        })
        continue
    if not isinstance(call_usage, dict):
        raise SystemExit(f"call {call_index} has invalid usage")
    for field in usage:
        value = call_usage.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SystemExit(f"call {call_index} has invalid {field}")
        usage[field] += value
if usage["reasoning_tokens"] > usage["completion_tokens"]:
    raise SystemExit("reasoning exceeds output tokens")
inference_calls = len(calls) - len(provider_errors)
if not 1 <= inference_calls <= 20:
    raise SystemExit("trace has invalid provider inference calls")

patch = trace.get("info", {}).get("patch")
if post_execution_agent_failure:
    patch_bytes = 0
    patch_sha256 = None
    patch_provenance = "post-execution agent failure"
elif patch is None:
    patch_bytes = 0
    patch_sha256 = hashlib.sha256(b"").hexdigest()
    patch_provenance = "official trace reported no source changes"
elif not isinstance(patch, str):
    raise SystemExit("trace lacks a patch string")
else:
    patch_bytes = len(patch.encode())
    patch_sha256 = hashlib.sha256(patch.encode()).hexdigest()
    patch_provenance = "official captured patch"

report = {
    "protocol": "coding-router-graded-swerebench-arm-artifact-v1",
    "valid": True,
    "task_id": args.task,
    "arm": args.arm,
    "model": args.model,
    "effort": args.effort,
    "reward": float(reward),
    "f2p_passed": f2p_passed,
    "f2p_total": args.f2p_total,
    "reward_provenance": reward_provenance,
    "official_verifier_reached": not post_execution_agent_failure,
    "provider_calls": len(calls),
    "provider_inference_calls": inference_calls,
    "provider_errors": provider_errors,
    "estimated_usage_calls": estimated_usage_calls,
    "stop_condition": trace.get("stop_condition"),
    "trace_ok": trace.get("ok"),
    "outer_ok": outer.get("ok"),
    "trace_errors": trace_errors,
    "outer_errors": outer.get("errors", []),
    "patch_bytes": patch_bytes,
    "patch_sha256": patch_sha256,
    "patch_provenance": patch_provenance,
    "scoring_seconds": scoring_seconds,
    "usage": usage,
    "usage_provenance": (
        "mixed exact and conservative trace-derived token estimate"
        if estimated_usage_calls
        else "exact token counts from pinned verifier Responses trace"
    ),
}
args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
'''


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return {str(key): item for key, item in value.items()}


def _read_verifier_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"private verifier row {line_number} is not an object")
            task_id = str(value.get("task_id") or "")
            if task_id in rows:
                raise ValueError(f"private verifier task repeats: {task_id}")
            rows[task_id] = {str(key): item for key, item in value.items()}
    return rows


def _arm_order(task_index: int) -> tuple[Arm, ...]:
    offset = task_index % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def _config(arm: Arm, task_json: str, output_dir: str) -> str:
    return f'''model = "{arm.model}"
num_tasks = 1
num_rollouts = 1
shuffle = false
max_concurrent = 1
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
reasoning_effort = "{arm.effort}"
max_tokens = 32768

[env]
max_concurrent_agents = 1

[env.taskset]
id = "swerebench-v2-v1"
task_json = "{task_json}"

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


def _completed(task_dir: Path, arm: Arm, value: object) -> bool:
    if not isinstance(value, dict):
        return False
    archive = task_dir / f"{arm.name}.tar.gz"
    report = task_dir / f"{arm.name}.report.json"
    if not archive.is_file() or not report.is_file():
        return False
    if _sha256(archive) != value.get("archive_sha256"):
        return False
    if _sha256(report) != value.get("report_sha256"):
        return False
    payload = _read_object(report)
    return (
        payload.get("valid") is True
        and payload.get("arm") == arm.name
        and payload.get("model") == arm.model
        and payload.get("effort") == arm.effort
    )


def _excluded(state: dict[str, Any]) -> bool:
    """Return whether one frozen cell made the whole task unusable without a rerun."""
    exclusion = state.get("exclusion")
    return (
        isinstance(exclusion, dict)
        and (state.get("stage"), exclusion.get("reason"))
        in {
            (
                "excluded-audit-artifact-loss",
                "validator rejected official no-change trace",
            ),
            (
                "excluded-ungradeable-scientific-cell",
                "official trace lacked a graded reward after one frozen attempt",
            ),
            (
                "excluded-ungradeable-scientific-cell",
                "scientific artifact became irrecoverable after E2B transport loss",
            ),
            (
                "excluded-ungradeable-scientific-cell",
                "official graded trace became irrecoverable after missing usage audit failure",
            ),
        }
        and exclusion.get("scope") == "whole-task"
        and isinstance(exclusion.get("arm"), str)
        and exclusion.get("observed_scientific_cells") == 1
        and exclusion.get("scientific_cells_rerun") == 0
        and exclusion.get("provider_usage_recoverable") is False
    )


def _update_summary(root: Path, total_tasks: int) -> None:
    with STATE_LOCK:
        completed_arms = 0
        complete_tasks = 0
        excluded_tasks = 0
        failed_tasks = 0
        provider_calls = 0
        usage = {
            arm.name: {
                "prompt_tokens": 0,
                "cached_input_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
            }
            for arm in ARMS
        }
        for state_path in (root / "tasks").glob("*/state.json"):
            state = _read_object(state_path)
            arm_states = state.get("arms", {})
            if isinstance(arm_states, dict):
                for name, value in arm_states.items():
                    if name not in ARM_BY_NAME or not isinstance(value, dict):
                        continue
                    completed_arms += 1
                    provider_calls += int(value.get("provider_calls", 0))
                    arm_usage = value.get("usage", {})
                    if isinstance(arm_usage, dict):
                        for field in usage[name]:
                            usage[name][field] += int(arm_usage.get(field, 0))
            if state.get("stage") == "complete":
                complete_tasks += 1
            if _excluded(state):
                excluded_tasks += 1
            if state.get("stage") == "failed":
                failed_tasks += 1
        costs: dict[str, float] = {}
        for arm in ARMS:
            input_rate, cached_rate, output_rate = MODEL_PRICES_PER_MTOK[arm.model]
            arm_usage = usage[arm.name]
            costs[arm.name] = (
                arm_usage["prompt_tokens"] * input_rate
                + arm_usage["cached_input_tokens"] * cached_rate
                + arm_usage["completion_tokens"] * output_rate
            ) / 1_000_000
        matrix_cost = sum(costs.values())
        _write_json(
            root / "progress.json",
            {
                "protocol": PROTOCOL,
                "total_tasks": total_tasks,
                "expected_cells": total_tasks * len(ARMS),
                "complete_tasks": complete_tasks,
                "excluded_tasks": excluded_tasks,
                "retained_task_coverage": (total_tasks - excluded_tasks) / total_tasks,
                "failed_tasks": failed_tasks,
                "completed_arms": completed_arms,
                "completed_scientific_cells": completed_arms,
                "provider_calls": provider_calls,
                "usage_by_arm": usage,
                "cost_usd_by_arm": costs,
                "matrix_cost_usd": matrix_cost,
                "cost_provenance": "trace-derived frozen list-price estimate",
                "rough_cumulative_experiment_spend_usd": PRIOR_SPEND_USD + matrix_cost,
            },
        )


def _run_task(
    root: Path,
    task_index: int,
    public: dict[str, Any],
    private: dict[str, Any],
    api_key: str,
    patched_taskset: Path,
    patch_report: Path,
    total_tasks: int,
) -> None:
    task_id = str(public["task_id"])
    image = str(public["image_name"])
    if not TASK_PATTERN.fullmatch(task_id) or not IMAGE_PATTERN.fullmatch(image):
        raise ValueError(f"unsafe frozen task: {task_id}")
    if (
        private.get("task_id") != task_id
        or private.get("split") != PHASE_NAME
        or int(private.get("f2p_total", -1)) != int(public["f2p_total"])
    ):
        raise ValueError(f"public/private task drift: {task_id}")
    task_dir = root / "tasks" / f"{task_index:04d}"
    task_dir.mkdir(parents=True, exist_ok=True)
    state_path = task_dir / "state.json"
    if state_path.is_file():
        state = _read_object(state_path)
        if (
            state.get("protocol") != PROTOCOL
            or state.get("task_id") != task_id
            or state.get("image") != image
        ):
            raise ValueError(f"task state identity drift: {task_id}")
        if _excluded(state):
            return
    else:
        state = {
            "protocol": PROTOCOL,
            "task_index": task_index,
            "task_id": task_id,
            "image": image,
            "f2p_total": int(public["f2p_total"]),
            "arm_order": [arm.name for arm in _arm_order(task_index)],
            "arms": {},
            "sandbox_attempts": [],
            "stage": "pending",
        }
        _write_json(state_path, state)
    arm_states = state.get("arms")
    if not isinstance(arm_states, dict):
        raise ValueError(f"invalid arm state: {task_id}")
    missing = [
        arm
        for arm in _arm_order(task_index)
        if not _completed(task_dir, arm, arm_states.get(arm.name))
    ]
    if not missing:
        state["stage"] = "complete"
        _write_json(state_path, state)
        return
    attempts = state.get("sandbox_attempts")
    if not isinstance(attempts, list):
        raise ValueError(f"invalid sandbox attempt state: {task_id}")

    for arm in missing:
        arm_attempts = [
            value
            for value in attempts
            if isinstance(value, dict) and value.get("arm") == arm.name
        ]
        if len(arm_attempts) >= 2:
            raise RuntimeError(f"frozen infrastructure retry exhausted: {task_id}/{arm.name}")
        sandbox = Sandbox.create(
            TEMPLATE_NAME,
            timeout=3_600,
            secure=True,
            allow_internet_access=True,
            envs={"OPENAI_API_KEY": api_key},
            metadata={
                "owner": "coding-router-v47",
                "phase": METADATA_PHASE,
                "task_index": str(task_index),
                "task_id": task_id,
                "arm": arm.name,
            },
        )
        attempt: dict[str, Any] = {
            "sandbox_id": sandbox.sandbox_id,
            "arm": arm.name,
            "terminated": False,
        }
        attempts.append(attempt)
        state["stage"] = f"running-{arm.name}"
        _write_json(state_path, state)
        remote_root = (
            f"/home/user/router-v47-{REMOTE_SEGMENT}/{task_index:04d}/{arm.name}"
        )
        remote_task = f"{remote_root}/task.json"
        try:
            runner._run(sandbox, f"mkdir -p {remote_root}/runtime", timeout=120)
            sandbox.files.write(remote_task, json.dumps(private, sort_keys=True))
            sandbox.files.write(f"{remote_root}/taskset.py", patched_taskset.read_bytes())
            sandbox.files.write(
                f"{remote_root}/taskset-patch-report.json",
                patch_report.read_bytes(),
            )
            sandbox.files.write(f"{remote_root}/validate.py", REMOTE_VALIDATOR)
            runner._run(
                sandbox,
                (
                    f"test \"$(sudo sha256sum {TASKSET_REMOTE} | cut -d' ' -f1)\" = "
                    f"{TASKSET_SOURCE_SHA256} && "
                    f"sudo cp {remote_root}/taskset.py {TASKSET_REMOTE} && "
                    f"test \"$(sudo sha256sum {TASKSET_REMOTE} | cut -d' ' -f1)\" = "
                    f"{TASKSET_PATCHED_SHA256} && "
                    f"cp {remote_root}/taskset-patch-report.json "
                    f"{remote_root}/runtime/ && "
                    "test \"$(sha256sum /opt/coding-router/"
                    "swerebench-docker-adapter-report.json | cut -d' ' -f1)\" = "
                    f"{DOCKER_ADAPTER_REPORT_SHA256} && "
                    "test \"$(sha256sum /opt/coding-router/"
                    "verifiers-responses-adapter-report.json | cut -d' ' -f1)\" = "
                    f"{RESPONSES_ADAPTER_REPORT_SHA256} && "
                    f"cp /opt/coding-router/*-adapter-report.json "
                    f"{remote_root}/runtime/ && "
                    f"sha256sum {remote_root}/runtime/* > "
                    f"{remote_root}/runtime/sha256sums"
                ),
                timeout=120,
            )
            runner._run(sandbox, f"sudo docker pull {image}", timeout=1_800)
            attempt["docker_image_id"] = runner._run(
                sandbox,
                f"sudo docker image inspect {image} --format '{{{{.Id}}}}'",
                timeout=120,
            ).stdout.strip()
            output_dir = f"{remote_root}/{arm.name}"
            config_path = f"{remote_root}/{arm.name}.toml"
            report_path = f"{remote_root}/{arm.name}.report.json"
            archive_path = f"{remote_root}/{arm.name}.tar.gz"
            sandbox.files.write(config_path, _config(arm, remote_task, output_dir))
            state["stage"] = f"running-{arm.name}"
            _write_json(state_path, state)
            result, sandbox = runner._run_durable_eval(
                sandbox,
                f"cd /opt/verifiers && sudo -E .venv/bin/eval @ {config_path}",
                effort=arm.name,
                exit_status_path=f"{remote_root}/{arm.name}.eval-exit-status",
                state=state,
                state_path=state_path,
                attempt=attempt,
                timeout=3_500,
            )
            runner._run(
                sandbox,
                (
                    f"sudo /opt/verifiers/.venv/bin/python {remote_root}/validate.py "
                    f"--traces {output_dir}/traces.jsonl --task {task_id} "
                    f"--arm {arm.name} --model {arm.model} --effort {arm.effort} "
                    f"--f2p-total {public['f2p_total']} --output {report_path}"
                ),
                timeout=120,
            )
            runner._run(
                sandbox,
                (
                    f"sudo tar -C {remote_root} -czf {archive_path} runtime "
                    f"{arm.name}.toml {arm.name}.report.json {arm.name}"
                ),
                timeout=300,
            )
            local_archive = task_dir / f"{arm.name}.tar.gz"
            local_report = task_dir / f"{arm.name}.report.json"
            runner._sync(sandbox, archive_path, local_archive)
            runner._sync(sandbox, report_path, local_report)
            report = _read_object(local_report)
            if (
                report.get("valid") is not True
                or report.get("task_id") != task_id
                or report.get("arm") != arm.name
            ):
                raise ValueError(f"downloaded report failed validation: {task_id}/{arm.name}")
            arm_states[arm.name] = {
                "archive_sha256": _sha256(local_archive),
                "report_sha256": _sha256(local_report),
                "model": arm.model,
                "effort": arm.effort,
                "provider_calls": report["provider_calls"],
                "usage": report["usage"],
                "reward": report["reward"],
                "f2p_passed": report["f2p_passed"],
                "f2p_total": report["f2p_total"],
                "eval_exit_code": result.exit_code,
                "sandbox_id": sandbox.sandbox_id,
            }
            state["stage"] = f"completed-{arm.name}"
            _write_json(state_path, state)
            _update_summary(root, total_tasks)
            logger.info(
                "task=%d/%d id=%s arm=%s reward=%.6f",
                task_index + 1,
                total_tasks,
                task_id,
                arm.name,
                float(report["reward"]),
            )
        except Exception as error:
            state["stage"] = "failed"
            state["error"] = repr(error)
            attempt["error"] = repr(error)
            logger.exception(
                "task failed index=%d id=%s arm=%s",
                task_index,
                task_id,
                arm.name,
            )
            raise
        finally:
            sandbox.kill()
            attempt["terminated"] = True
            _write_json(state_path, state)
            _update_summary(root, total_tasks)
    state["stage"] = "complete"
    state["all_sandboxes_terminated"] = True
    _write_json(state_path, state)


def execute(
    root: Path,
    corpus_path: Path,
    verifier_tasks_path: Path,
    patched_taskset: Path,
    patch_report: Path,
    *,
    concurrency: int,
) -> None:
    """Validate the frozen launch and execute or resume missing cells."""
    expected_hashes = {
        corpus_path: CORPUS_SHA256,
        verifier_tasks_path: VERIFIER_TASKS_SHA256,
        patched_taskset: TASKSET_PATCHED_SHA256,
    }
    for path, expected in expected_hashes.items():
        if _sha256(path) != expected:
            raise ValueError(f"frozen launch input changed: {path}")
    patch = _read_object(patch_report)
    if (
        patch.get("source_sha256") != TASKSET_SOURCE_SHA256
        or patch.get("patched_sha256") != TASKSET_PATCHED_SHA256
        or patch.get("changes", {}).get("reward_is_f2p_passed_over_total") is not True
    ):
        raise ValueError("graded taskset patch report is invalid")
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise ValueError(f"concurrency must be between 1 and {MAX_CONCURRENCY}")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is unavailable")
    if not Template.exists(TEMPLATE_NAME):
        raise RuntimeError(f"required E2B template is absent: {TEMPLATE_NAME}")
    active = runner._capacity()
    if active + concurrency > E2B_ACCOUNT_CAP:
        raise RuntimeError(
            f"E2B capacity is insufficient: active={active} launch={concurrency} "
            f"cap={E2B_ACCOUNT_CAP}"
        )
    corpus = _read_object(corpus_path)
    public_rows = corpus.get("tasks")
    if not isinstance(public_rows, list) or len(public_rows) != EXPECTED_TASKS:
        raise ValueError(f"{PHASE_NAME} corpus must contain exactly {EXPECTED_TASKS} tasks")
    private_by_id = _read_verifier_rows(verifier_tasks_path)
    task_ids = [str(row.get("task_id")) for row in public_rows if isinstance(row, dict)]
    if len(task_ids) != EXPECTED_TASKS or len(set(task_ids)) != EXPECTED_TASKS:
        raise ValueError(f"{PHASE_NAME} corpus identities are invalid")
    if any(task_id not in private_by_id for task_id in task_ids):
        raise ValueError("development corpus lacks a private verifier row")

    root.mkdir(parents=True, exist_ok=True)
    (root / "tasks").mkdir(exist_ok=True)
    launch = {
        "protocol": PROTOCOL,
        "corpus_path": str(corpus_path.resolve()),
        "corpus_sha256": CORPUS_SHA256,
        "verifier_tasks_path": str(verifier_tasks_path.resolve()),
        "verifier_tasks_sha256": VERIFIER_TASKS_SHA256,
        "taskset_source_sha256": TASKSET_SOURCE_SHA256,
        "taskset_patched_sha256": TASKSET_PATCHED_SHA256,
        "template": TEMPLATE_NAME,
        "template_id": TEMPLATE_ID,
        "template_build_id": TEMPLATE_BUILD_ID,
        "verifiers_commit": VERIFIERS_COMMIT,
        "arms": [arm._asdict() for arm in ARMS],
        "attempts_per_arm": 1,
        "tasks": len(public_rows),
        "expected_cells": len(public_rows) * len(ARMS),
        "concurrency": concurrency,
        "active_e2b_before": active,
        "e2b_account_cap": E2B_ACCOUNT_CAP,
        "cost_ceiling_usd": COST_CEILING_USD,
        "prior_spend_usd": PRIOR_SPEND_USD,
        "phase": PHASE_NAME,
        "deep_swe_outcomes_accessed": False,
        "confirmation_outcomes_accessed": False,
        "model_persisted": False,
    }
    if PHASE_NAME == "confirmation":
        if EXTERNAL_AUTHORIZATION is None:
            raise ValueError("confirmation execution lacks frozen development authorization")
        launch["authorization"] = EXTERNAL_AUTHORIZATION
        launch["confirmation_outcomes_accessed_before_launch"] = False
    launch_path = root / "launch.json"
    if launch_path.is_file():
        prior = _read_object(launch_path)
        if _launch_identity(prior) != _launch_identity(launch):
            raise ValueError("resume launch manifest differs from the frozen experiment")
    else:
        _write_json(launch_path, launch)
    _update_summary(root, len(public_rows))

    errors: list[Exception] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                _run_task,
                root,
                index,
                public,
                private_by_id[str(public["task_id"])],
                api_key,
                patched_taskset,
                patch_report,
                len(public_rows),
            ): index
            for index, value in enumerate(public_rows)
            for public in [value if isinstance(value, dict) else {}]
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:  # noqa: BLE001 - isolate task workers
                errors.append(error)
    _update_summary(root, len(public_rows))
    if errors:
        raise RuntimeError(f"{len(errors)} task workers failed; inspect task states")


def _launch_identity(launch: dict[str, Any]) -> dict[str, Any]:
    """Return the scientific launch identity for resume comparison."""
    operational = {"active_e2b_before", "concurrency"}
    identity = {key: value for key, value in launch.items() if key not in operational}
    identity.setdefault("phase", PHASE_NAME)
    return identity


def main() -> None:
    """Parse arguments and execute the frozen development matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--verifier-tasks", type=Path, required=True)
    parser.add_argument("--patched-taskset", type=Path, required=True)
    parser.add_argument("--patch-report", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=100)
    args = parser.parse_args()
    execute(
        args.root,
        args.corpus,
        args.verifier_tasks,
        args.patched_taskset,
        args.patch_report,
        concurrency=args.concurrency,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
