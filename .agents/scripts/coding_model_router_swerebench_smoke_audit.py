"""Audit the frozen four-cell SWE-rebench Responses smoke without extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import tarfile
import tomllib
from pathlib import Path
from typing import Any

logger = logging.getLogger("coding-router-swerebench-smoke-audit")

PROTOCOL = "coding-router-swerebench-smoke-audit-v1"
MODEL = "gpt-5.6-luna"
EFFORTS = ("xhigh", "max")
TASKS = (
    "0xs34n__starknet.js-538",
    "acloudguru__serverless-plugin-aws-alerts-13",
)
VERIFIERS_COMMIT = "f6e420b9908ae14d625f079881f13c15011ee1c9"
MAX_TOKENS = 32_768
MAX_TURNS = 20
PRICES_PER_MTOK = {
    "prompt_tokens": 1.0,
    "cached_input_tokens": 0.1,
    "completion_tokens": 6.0,
}


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    return value


def _read_member(archive: tarfile.TarFile, name: str) -> bytes:
    try:
        member = archive.getmember(name)
    except KeyError as error:
        raise ValueError(f"archive is missing {name}") from error
    if not member.isfile() or member.issym() or member.islnk():
        raise ValueError(f"archive member {name} is not a regular file")
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError(f"archive member {name} cannot be read")
    return stream.read()


def _validate_members(archive: tarfile.TarFile, effort: str) -> None:
    expected = {
        f"{effort}",
        f"{effort}/eval.log",
        f"{effort}/config.toml",
        f"{effort}/traces.jsonl",
    }
    names = {member.name.rstrip("/") for member in archive.getmembers()}
    if names != expected:
        raise ValueError(
            f"{effort} archive members differ from the frozen shape: {sorted(names)}"
        )
    for member in archive.getmembers():
        if member.name.rstrip("/") == effort:
            if not member.isdir():
                raise ValueError(f"{effort} archive root is not a directory")
        elif not member.isfile() or member.issym() or member.islnk():
            raise ValueError(f"unsafe archive member: {member.name}")


def _validate_config(content: bytes, effort: str) -> None:
    config = _object(tomllib.loads(content.decode("utf-8")), "config")
    sampling = _object(config.get("sampling"), "config.sampling")
    env = _object(config.get("env"), "config.env")
    agent = _object(env.get("agent"), "config.env.agent")
    harness = _object(agent.get("harness"), "config.env.agent.harness")
    taskset = _object(env.get("taskset"), "config.env.taskset")
    if config.get("model") != MODEL:
        raise ValueError(f"{effort} config model differs from {MODEL}")
    if sampling != {
        "temperature": 1.0,
        "reasoning_effort": effort,
        "max_tokens": MAX_TOKENS,
    }:
        raise ValueError(f"{effort} sampling config differs from the frozen values")
    if config.get("num_tasks") != 2 or config.get("num_rollouts") != 1:
        raise ValueError(f"{effort} config does not describe exactly two cells")
    if agent.get("max_turns") != MAX_TURNS or agent.get("max_output_tokens") != 131_072:
        raise ValueError(f"{effort} agent limits differ from the frozen values")
    if harness.get("id") != "mini_swe_agent" or harness.get("version") != "2.4.5":
        raise ValueError(f"{effort} harness differs from the frozen version")
    if taskset.get("id") != "swerebench-v2-v1":
        raise ValueError(f"{effort} taskset differs from the official frozen taskset")


def _usage(call: dict[str, Any], effort: str) -> dict[str, int]:
    if call.get("model") != MODEL:
        raise ValueError(f"{effort} call attests a different model")
    if call.get("endpoint") != "/responses":
        raise ValueError(f"{effort} call did not use the Responses endpoint")
    sampling = _object(call.get("sampling"), f"{effort} call sampling")
    if sampling.get("reasoning_effort") != effort:
        raise ValueError(f"{effort} call attests a different reasoning effort")
    if sampling.get("max_tokens") != MAX_TOKENS:
        raise ValueError(f"{effort} call attests a different output limit")
    usage = _object(call.get("usage"), f"{effort} call usage")
    result: dict[str, int] = {}
    for field in (*PRICES_PER_MTOK, "reasoning_tokens"):
        value = usage.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{effort} call has invalid {field}")
        result[field] = value
    if result["reasoning_tokens"] > result["completion_tokens"]:
        raise ValueError(f"{effort} call reasoning exceeds completion tokens")
    return result


def _validate_trace(
    row: object,
    effort: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    outer = _object(row, f"{effort} trace row")
    if outer.get("ok") is not True or outer.get("errors") != []:
        raise ValueError(f"{effort} outer trace row is not successful")
    traces = outer.get("traces")
    if not isinstance(traces, list) or len(traces) != 1:
        raise ValueError(f"{effort} outer row does not contain exactly one trace")
    trace = _object(traces[0], f"{effort} inner trace")
    if trace.get("ok") is not True or trace.get("errors") != []:
        raise ValueError(f"{effort} inner trace is not successful")
    if trace.get("is_completed") is not True:
        raise ValueError(f"{effort} trace is incomplete")
    stop = trace.get("stop_condition")
    if stop not in {"agent_completed", "max_turns"}:
        raise ValueError(f"{effort} trace has ungradeable stop condition {stop!r}")

    task = _object(trace.get("task"), f"{effort} task")
    task_data = _object(task.get("data"), f"{effort} task data")
    task_id = task_data.get("name")
    if task_id not in TASKS:
        raise ValueError(f"{effort} trace has unexpected task {task_id!r}")
    verifiers = _object(trace.get("verifiers"), f"{effort} verifiers")
    if verifiers.get("commit") != VERIFIERS_COMMIT:
        raise ValueError(f"{effort} trace has a different verifier commit")

    agent = _object(trace.get("agent"), f"{effort} agent")
    config = _object(agent.get("config"), f"{effort} agent config")
    sampling = _object(config.get("sampling"), f"{effort} agent sampling")
    if config.get("model") != MODEL or config.get("max_turns") != MAX_TURNS:
        raise ValueError(f"{effort} trace has a different model or turn limit")
    if sampling.get("reasoning_effort") != effort or sampling.get("max_tokens") != MAX_TOKENS:
        raise ValueError(f"{effort} trace has different sampling values")

    rewards = _object(trace.get("rewards"), f"{effort} rewards")
    solved = _object(rewards.get("solved"), f"{effort} solved reward")
    reward = solved.get("score")
    if (
        not isinstance(reward, (int, float))
        or isinstance(reward, bool)
        or float(reward) not in {0.0, 1.0}
    ):
        raise ValueError(f"{effort} trace has invalid official reward {reward!r}")
    timing = _object(trace.get("timing"), f"{effort} timing")
    scoring = _object(timing.get("scoring"), f"{effort} scoring timing")
    start, end = scoring.get("start"), scoring.get("end")
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or end <= start:
        raise ValueError(f"{effort} trace lacks positive official scoring timing")

    calls = trace.get("calls")
    if not isinstance(calls, list) or not calls or len(calls) > MAX_TURNS:
        raise ValueError(f"{effort} trace has invalid provider call count")
    totals = {field: 0 for field in (*PRICES_PER_MTOK, "reasoning_tokens")}
    for raw_call in calls:
        call_usage = _usage(_object(raw_call, f"{effort} call"), effort)
        for field, value in call_usage.items():
            totals[field] += value

    info = _object(trace.get("info"), f"{effort} info")
    patch = info.get("patch")
    if not isinstance(patch, str):
        raise ValueError(f"{effort} trace patch is not a string")
    return (
        {
            "task_id": task_id,
            "reward": float(reward),
            "stop_condition": stop,
            "provider_calls": len(calls),
            "patch_bytes": len(patch.encode("utf-8")),
            "patch_sha256": _sha256_bytes(patch.encode("utf-8")),
            "scoring_seconds": end - start,
            "usage": totals,
        },
        totals,
    )


def _audit_archive(path: Path, effort: str) -> tuple[dict[str, Any], dict[str, int]]:
    with tarfile.open(path, "r:gz") as archive:
        _validate_members(archive, effort)
        _validate_config(_read_member(archive, f"{effort}/config.toml"), effort)
        trace_content = _read_member(archive, f"{effort}/traces.jsonl")
    rows = [json.loads(line) for line in trace_content.splitlines() if line.strip()]
    if len(rows) != 2:
        raise ValueError(f"{effort} archive does not contain exactly two trace rows")
    cells: list[dict[str, Any]] = []
    totals = {field: 0 for field in (*PRICES_PER_MTOK, "reasoning_tokens")}
    for row in rows:
        cell, usage = _validate_trace(row, effort)
        cells.append(cell)
        for field, value in usage.items():
            totals[field] += value
    if {cell["task_id"] for cell in cells} != set(TASKS):
        raise ValueError(f"{effort} archive does not contain both frozen tasks")
    cells.sort(key=lambda cell: TASKS.index(str(cell["task_id"])))
    return (
        {
            "path": str(path.resolve()),
            "sha256": _sha256_path(path),
            "effort": effort,
            "cells": cells,
            "usage": totals,
        },
        totals,
    )


def audit(
    archives: dict[str, Path],
    state_path: Path,
    prior_spend_usd: float,
) -> dict[str, Any]:
    """Return a strict report for the exact frozen smoke artifacts."""
    state = _object(json.loads(state_path.read_text(encoding="utf-8")), "state")
    if state.get("account_cap") != 1000:
        raise ValueError("state has a different E2B account cap")
    if state.get("task_ids") != list(TASKS) or state.get("efforts") != list(EFFORTS):
        raise ValueError("state has different frozen cells")
    if state.get("expected_cells") != 4 or state.get("scientific_attempt") != 0:
        raise ValueError("state has a different cell count or scientific attempt")
    if (
        state.get("sandbox_terminated") is not True
        or state.get("owned_eval_terminated") is not True
    ):
        raise ValueError("state does not attest termination of the owned smoke runtime")
    sandbox_id = state.get("sandbox_id")
    if not isinstance(sandbox_id, str) or not sandbox_id:
        raise ValueError("state lacks the exact owned sandbox id")

    audited_archives: list[dict[str, Any]] = []
    usage_totals = {field: 0 for field in (*PRICES_PER_MTOK, "reasoning_tokens")}
    for effort in EFFORTS:
        archive, usage = _audit_archive(archives[effort], effort)
        expected_sha = state.get(f"{effort}_archive_sha256")
        if archive["sha256"] != expected_sha:
            raise ValueError(f"{effort} archive hash differs from state")
        audited_archives.append(archive)
        for field, value in usage.items():
            usage_totals[field] += value

    provider_calls = sum(
        cell["provider_calls"]
        for archive in audited_archives
        for cell in archive["cells"]
    )
    cost_usd = sum(
        usage_totals[field] * price / 1_000_000
        for field, price in PRICES_PER_MTOK.items()
    )
    rewards = {
        effort: {
            cell["task_id"]: cell["reward"]
            for cell in archive["cells"]
        }
        for effort, archive in zip(EFFORTS, audited_archives, strict=True)
    }
    return {
        "protocol": PROTOCOL,
        "valid": True,
        "scientific_attempt": 0,
        "model": MODEL,
        "cells": 4,
        "tasks": list(TASKS),
        "efforts": list(EFFORTS),
        "official_rewards": rewards,
        "provider_calls": provider_calls,
        "usage": usage_totals,
        "usage_provenance": "exact token counts from pinned verifier Responses traces",
        "cost_usd": cost_usd,
        "cost_provenance": "trace-derived frozen list-price estimate",
        "prices_per_mtok_usd": PRICES_PER_MTOK,
        "reasoning_tokens_charge": "included in completion_tokens; not charged twice",
        "prior_experiment_spend_usd": prior_spend_usd,
        "rough_cumulative_spend_usd": prior_spend_usd + cost_usd,
        "remaining_authorized_budget_usd": 20_000.0 - prior_spend_usd - cost_usd,
        "sandbox": {"id": sandbox_id, "terminated": True},
        "resume": {
            "infrastructure_resumes": state.get("infrastructure_resume"),
            "pre_inference_provider_calls": 0,
            "gradeable_cells_rerun": 0,
            "same_scientific_attempt": True,
        },
        "state_correction": {
            "field": "provider_inference_calls",
            "stale_value": state.get("provider_inference_calls"),
            "audited_value": provider_calls,
            "reason": "state was written before final raw Responses traces were audited",
        },
        "state": {"path": str(state_path.resolve()), "sha256": _sha256_path(state_path)},
        "archives": audited_archives,
    }


def main() -> None:
    """Parse arguments, audit the smoke, and write one canonical report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--xhigh-archive", type=Path, required=True)
    parser.add_argument("--max-archive", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-spend-usd", type=float, default=405.33)
    args = parser.parse_args()
    report = audit(
        {"xhigh": args.xhigh_archive, "max": args.max_archive},
        args.state,
        args.prior_spend_usd,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info(
        "audited cells=%d calls=%d estimated_cost_usd=%.6f output=%s",
        report["cells"],
        report["provider_calls"],
        report["cost_usd"],
        args.output,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
