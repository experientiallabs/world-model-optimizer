"""Stage and run the bounded real-provider WMO serving verification."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

import httpx
from coding_model_router_analyze import (
    EXPERIMENT_ID,
    _canonical_matrix,
    _read_object,
    _sha256,
    _write_json,
)
from coding_model_router_world_model import (
    AZURE_ENV_SOURCE,
    PROVIDER_ENV_SOURCE,
    _complete_event,
    _ledger_rows,
    _reserve_event,
)

from wmo.config import load_env_file
from wmo.optimize.policy import RoutingPolicy

ENDPOINT = "coding-router"
OPENAI_PROBE = "coding-router-openai-probe"
ANTHROPIC_PROBE = "coding-router-anthropic-probe"
FALLBACK_PROBE = "coding-router-fallback-probe"
CACHE_AWARE_PROBE = "coding-router-cache-aware-probe"
OPENAI_ARM = "oai-luna-high"
ANTHROPIC_ARM = "ant-haiku45"
SERVING_RESERVATION_USD = 100.0
MAX_ATTEMPTS = 3
REQUEST_TIMEOUT_S = 300.0

TOOL_PROMPT = (
    "You are reviewing a new Python retry helper. Call lookup_contract for retry.py before "
    "suggesting one concise invariant test."
)
ANTHROPIC_PROMPT = (
    "In three concise bullets, explain why atomic file replacement prevents torn JSON state "
    "during a process crash."
)
MAIN_PROMPT = (
    "Design one property-based test for a bounded LRU cache that stores multi-turn tool "
    "transcripts. Return only the test idea and its invariant."
)
FALLBACK_PROMPT = (
    "zzqv jxkp synthetic unseen coding dialect: prove the safe router abstention path."
)
DIAL_PROMPT = (
    "Name one race condition to test when a live routing dial is changed while requests are "
    "already in flight."
)
CACHE_PROMPT_ONE = (
    "Design one invariant for a cache-aware coding router that already has a conversation "
    "incumbent."
)
CACHE_PROMPT_TWO = (
    "Now name one safe condition under which that router may switch models mid-conversation."
)

TOOLS: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_contract",
            "description": "Read a named source contract before reviewing it.",
            "parameters": {
                "type": "object",
                "properties": {"file": {"type": "string"}},
                "required": ["file"],
            },
        },
    }
]


def _store(root: Path) -> Path:
    return root / "serving" / "store"


def _copy_world_artifact(source: Path, target: Path) -> None:
    if target.exists():
        raise ValueError(f"{target} already exists; staged serving artifacts are immutable")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)


def _copy_policy(policy: RoutingPolicy, source: Path, target: Path) -> None:
    policy_path = target / "policy.json"
    if policy.kind == "knn":
        bank = Path(policy.knn_bank_path)
        if bank.is_absolute() or bank.name != policy.knn_bank_path:
            raise ValueError("deployable kNN policy must use a portable sidecar filename")
        source_bank = source / bank
        if not source_bank.is_file():
            raise ValueError(f"deployable evidence bank is missing at {source_bank}")
        shutil.copy2(source_bank, target / bank.name)
    policy.save(policy_path)
    # Force the lazy sidecar validation now, before a paid request can reach the endpoint.
    if policy.kind == "knn":
        RoutingPolicy.load(policy_path).knn_bank()


def _prepare(root: Path) -> None:
    if not (root / "analysis" / "evaluation-complete.json").is_file():
        raise ValueError("real heldout evaluation must finish before serving verification")
    world_artifact = root / "world-model" / "artifact"
    if not (world_artifact / "config.toml").is_file():
        raise ValueError("the low-fidelity world-model artifact is required for wmo serve")
    deploy = root / "analysis" / "deployable"
    source_policy = deploy / "policy.json"
    if not source_policy.is_file():
        raise ValueError("the deployable real routing policy is missing")
    policy = RoutingPolicy.load(source_policy)
    policy.knn_bank()
    names = {entry.name for entry in policy.pool}
    for required in (OPENAI_ARM, ANTHROPIC_ARM):
        if required not in names:
            raise ValueError(f"serving probe arm {required} is absent from the frozen pool")

    existing_tasks = {outcome.task for outcome in _canonical_matrix(root)[0].outcomes}
    for prompt in (
        TOOL_PROMPT,
        ANTHROPIC_PROMPT,
        MAIN_PROMPT,
        FALLBACK_PROMPT,
        DIAL_PROMPT,
        CACHE_PROMPT_ONE,
        CACHE_PROMPT_TWO,
    ):
        if prompt in existing_tasks:
            raise ValueError("a serving prompt duplicates a frozen benchmark task")

    models = _store(root) / "models"
    main_dir = models / ENDPOINT
    _copy_world_artifact(world_artifact, main_dir)
    _copy_policy(policy, deploy, main_dir)

    openai_dir = models / OPENAI_PROBE
    _copy_world_artifact(world_artifact, openai_dir)
    RoutingPolicy(
        kind="static",
        default_model=OPENAI_ARM,
        pool=policy.pool,
    ).save(openai_dir / "policy.json")

    anthropic_dir = models / ANTHROPIC_PROBE
    _copy_world_artifact(world_artifact, anthropic_dir)
    RoutingPolicy(
        kind="static",
        default_model=ANTHROPIC_ARM,
        pool=policy.pool,
    ).save(anthropic_dir / "policy.json")

    fallback_dir = models / FALLBACK_PROBE
    _copy_world_artifact(world_artifact, fallback_dir)
    forced_fallback = policy.model_copy(
        update={
            "floor_sim": 2.0,
            "floor_q": 1.0,
        }
    )
    forced_fallback.attach_bank(policy.knn_bank())
    _copy_policy(forced_fallback, deploy, fallback_dir)

    cache_dir = models / CACHE_AWARE_PROBE
    _copy_world_artifact(world_artifact, cache_dir)
    cache_aware = policy.model_copy(update={"cache_aware": True})
    cache_aware.attach_bank(policy.knn_bank())
    _copy_policy(cache_aware, deploy, cache_dir)

    marker = root / "serving" / "prepare.json"
    _write_json(
        marker,
        {
            "protocol": "coding-router-serving-v1",
            "world_model_config_sha256": _sha256(world_artifact / "config.toml"),
            "deployable_policy_sha256": _sha256(source_policy),
            "deployable_bank_sha256": _sha256(policy.bank_path()),
            "endpoints": [
                ENDPOINT,
                OPENAI_PROBE,
                ANTHROPIC_PROBE,
                FALLBACK_PROBE,
                CACHE_AWARE_PROBE,
            ],
            "synthetic_probe_disclosure": (
                f"{OPENAI_PROBE} and {ANTHROPIC_PROBE} are static provider-path probes; "
                f"{FALLBACK_PROBE} is the deployable kNN policy with a forced novelty floor. "
                f"{CACHE_AWARE_PROBE} changes only the cache-aware flag for an operational "
                f"ablation. Only {ENDPOINT} is the selected deployable policy."
            ),
            "unseen_prompts": 7,
            "paid_calls": 0,
            "launch_command": (
                "python .agents/scripts/coding_model_router_serve_verify.py run "
                "--root .wmo/experiments/coding-router-20260728 --port <free-local-port>"
            ),
        },
    )


def _server_command(store: Path, port: int) -> list[str]:
    candidate = Path(sys.executable).with_name("wmo")
    executable = candidate if candidate.is_file() else Path(shutil.which("wmo") or "")
    if not executable.is_file():
        raise ValueError("wmo executable is unavailable beside the active Python interpreter")
    return [
        str(executable),
        "serve",
        "--root",
        str(store),
        "--port",
        str(port),
    ]


def _assert_port_free(port: int) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))


def _rows_after(path: Path, offset: int) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines()[offset:]:
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} contains a non-object row")
        rows.append({str(key): item for key, item in value.items()})
    return rows


def _number(row: dict[str, object], key: str) -> float:
    value = row.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _next_event_id(root: Path) -> str:
    prefix = "serving-verification:"
    rows = _ledger_rows(root / "spend-ledger.jsonl")
    if any(
        row.get("phase") == "serving_verification" and row.get("completion_status") == "passed"
        for row in rows
    ):
        raise ValueError("the bounded serving verification already passed")
    attempts = sum(
        isinstance(row.get("event_id"), str) and cast("str", row["event_id"]).startswith(prefix)
        for row in rows
    )
    if attempts >= MAX_ATTEMPTS:
        raise ValueError("the serving verification exhausted its three infrastructure attempts")
    return f"{prefix}{attempts + 1}"


def _post(
    client: httpx.Client,
    endpoint: str,
    *,
    messages: list[dict[str, object]],
    tools: list[dict[str, object]] | None = None,
    tool_choice: object | None = None,
) -> httpx.Response:
    body: dict[str, object] = {
        "model": endpoint,
        "messages": messages,
        "max_completion_tokens": 4096,
    }
    if tools is not None:
        body["tools"] = tools
        body["parallel_tool_calls"] = False
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    response = client.post("/v1/chat/completions", json=body)
    if response.status_code != 200:
        raise ValueError(f"{endpoint} returned HTTP {response.status_code}: {response.text[:500]}")
    payload = response.json()
    if (
        payload.get("object") != "chat.completion"
        or payload.get("model") != endpoint
        or not isinstance(payload.get("choices"), list)
        or not isinstance(payload.get("usage"), dict)
    ):
        raise ValueError(f"{endpoint} returned a non-OpenAI-compatible completion shape")
    return response


def _wait_ready(
    client: httpx.Client,
    process: subprocess.Popen[bytes],
    expected: set[str],
) -> None:
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ValueError(f"wmo serve exited before readiness with code {process.returncode}")
        try:
            response = client.get("/v1/models")
            if response.status_code == 200:
                models = {row["id"] for row in response.json()["data"]}
                if models == expected:
                    return
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    raise TimeoutError("wmo serve did not expose the frozen endpoint set within 45 seconds")


def _exercise(client: httpx.Client) -> dict[str, object]:
    tool_first = _post(
        client,
        OPENAI_PROBE,
        messages=[{"role": "user", "content": TOOL_PROMPT}],
        tools=TOOLS,
        tool_choice="required",
    )
    if tool_first.headers.get("x-wmo-routed-model") != OPENAI_ARM:
        raise ValueError("the OpenAI provider-path probe routed to the wrong arm")
    first_message = tool_first.json()["choices"][0]["message"]
    calls = first_message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        raise ValueError("the real OpenAI route did not return the required tool call")
    call = calls[0]
    tool_second = _post(
        client,
        OPENAI_PROBE,
        messages=[
            {"role": "user", "content": TOOL_PROMPT},
            {
                "role": "assistant",
                "content": first_message.get("content"),
                "tool_calls": calls,
            },
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": (
                    "retry.py parses Retry-After as seconds or an HTTP date and clamps "
                    "negative delays to zero."
                ),
            },
        ],
        tools=TOOLS,
    )
    if tool_second.headers.get("x-wmo-routed-model") != OPENAI_ARM:
        raise ValueError("conversation affinity did not retain the OpenAI arm")

    anthropic = _post(
        client,
        ANTHROPIC_PROBE,
        messages=[{"role": "user", "content": ANTHROPIC_PROMPT}],
    )
    if anthropic.headers.get("x-wmo-routed-model") != ANTHROPIC_ARM:
        raise ValueError("the Anthropic provider-path probe routed to the wrong arm")

    main = _post(
        client,
        ENDPOINT,
        messages=[{"role": "user", "content": MAIN_PROMPT}],
    )
    if not main.headers.get("x-wmo-routed-model"):
        raise ValueError("the selected endpoint returned no routed-model audit header")
    fallback = _post(
        client,
        FALLBACK_PROBE,
        messages=[{"role": "user", "content": FALLBACK_PROMPT}],
    )
    cache_first = _post(
        client,
        CACHE_AWARE_PROBE,
        messages=[{"role": "user", "content": CACHE_PROMPT_ONE}],
    )
    cache_first_message = cache_first.json()["choices"][0]["message"]
    cache_second = _post(
        client,
        CACHE_AWARE_PROBE,
        messages=[
            {"role": "user", "content": CACHE_PROMPT_ONE},
            {
                "role": "assistant",
                "content": cache_first_message.get("content"),
            },
            {"role": "user", "content": CACHE_PROMPT_TWO},
        ],
    )

    config = client.get(f"/v1/endpoints/{ENDPOINT}/config")
    if config.status_code != 200 or not config.json().get("dialable"):
        raise ValueError("the selected endpoint did not expose its live cost-quality dial")
    original_dial = config.json().get("cost_quality")
    if not isinstance(original_dial, (int, float)) or isinstance(original_dial, bool):
        raise ValueError("the selected endpoint carries no restorable numeric dial")
    dial_request: httpx.Response | None = None
    try:
        for value in (0.0, 1.0):
            moved = client.put(
                f"/v1/endpoints/{ENDPOINT}/config",
                json={"cost_quality": value},
            )
            if moved.status_code != 200 or moved.json().get("cost_quality") != value:
                raise ValueError(f"live dial did not move to {value}")
        dial_request = _post(
            client,
            ENDPOINT,
            messages=[{"role": "user", "content": DIAL_PROMPT}],
        )
    finally:
        restored = client.put(
            f"/v1/endpoints/{ENDPOINT}/config",
            json={"cost_quality": float(original_dial)},
        )
        if restored.status_code != 200:
            raise ValueError("the selected serving dial could not be restored")
    if dial_request is None:
        raise ValueError("the selected serving dial request did not complete")

    return {
        "openai_route": tool_first.headers["x-wmo-routed-model"],
        "anthropic_route": anthropic.headers["x-wmo-routed-model"],
        "selected_route": main.headers["x-wmo-routed-model"],
        "fallback_route": fallback.headers["x-wmo-routed-model"],
        "cache_aware_first_route": cache_first.headers["x-wmo-routed-model"],
        "cache_aware_second_route": cache_second.headers["x-wmo-routed-model"],
        "dial_route": dial_request.headers["x-wmo-routed-model"],
        "tool_call_name": call["function"]["name"],
        "tool_round_trip_finish_reason": tool_second.json()["choices"][0]["finish_reason"],
        "original_dial": float(original_dial),
    }


def _validate_rows(
    rows: list[dict[str, object]],
    result: dict[str, object],
    default_model: str,
    pool_names: set[str],
) -> dict[str, object]:
    if len(rows) != 8:
        raise ValueError(f"expected exactly eight serving log rows, found {len(rows)}")
    if any(row.get("status") != "ok" for row in rows):
        raise ValueError("one or more serving requests did not finish successfully")
    if any(
        result.get(field) not in pool_names
        for field in (
            "selected_route",
            "dial_route",
            "cache_aware_first_route",
            "cache_aware_second_route",
        )
    ):
        raise ValueError("the selected endpoint emitted an unknown routed-model audit value")
    by_endpoint: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        endpoint = row.get("endpoint")
        if isinstance(endpoint, str):
            by_endpoint.setdefault(endpoint, []).append(row)
        for field in (
            "cost_usd",
            "router_cost_usd",
            "input_tokens",
            "output_tokens",
            "cached_tokens",
            "cache_credit_usd",
        ):
            if field == "cache_credit_usd" and row.get(field) is None:
                continue
            if not isinstance(row.get(field), (int, float)) or isinstance(row.get(field), bool):
                raise ValueError(f"serving log row has no numeric {field}")
    openai_rows = by_endpoint.get(OPENAI_PROBE, [])
    if len(openai_rows) != 2 or openai_rows[1].get("routing_reason") != (
        "sticky: conversation affinity"
    ):
        raise ValueError("tool round trip did not leave conversation-affinity evidence")
    fallback_rows = by_endpoint.get(FALLBACK_PROBE, [])
    if (
        len(fallback_rows) != 1
        or fallback_rows[0].get("gate") != "novelty-abstain"
        or fallback_rows[0].get("model") != default_model
    ):
        raise ValueError("forced novelty probe did not exercise the safe fallback")
    cache_rows = by_endpoint.get(CACHE_AWARE_PROBE, [])
    if (
        len(cache_rows) != 2
        or not isinstance(cache_rows[1].get("cache_credit_usd"), (int, float))
        or isinstance(cache_rows[1].get("cache_credit_usd"), bool)
        or _number(cache_rows[1], "cache_credit_usd") <= 0
    ):
        raise ValueError("cache-aware probe did not log a positive incumbent cache credit")
    result["affinity_reason"] = openai_rows[1]["routing_reason"]
    result["fallback_gate"] = fallback_rows[0]["gate"]
    result["cache_aware_credit_usd"] = _number(
        cache_rows[1],
        "cache_credit_usd",
    )
    result["provider_cost_usd"] = sum(_number(row, "cost_usd") for row in rows)
    result["router_cost_usd"] = sum(_number(row, "router_cost_usd") for row in rows)
    result["cached_input_tokens"] = sum(_number(row, "cached_tokens") for row in rows)
    result["input_tokens"] = sum(_number(row, "input_tokens") for row in rows)
    result["output_tokens"] = sum(_number(row, "output_tokens") for row in rows)
    return result


def _run(root: Path, port: int) -> None:
    marker = _read_object(root / "serving" / "prepare.json")
    deploy = root / "analysis" / "deployable"
    if marker.get("deployable_policy_sha256") != _sha256(deploy / "policy.json"):
        raise ValueError("deployable policy changed after serving preparation")
    policy = RoutingPolicy.load(deploy / "policy.json")
    if marker.get("deployable_bank_sha256") != _sha256(policy.bank_path()):
        raise ValueError("deployable evidence bank changed after serving preparation")
    for variable in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        if not os.environ.get(variable):
            raise ValueError(f"{variable} is required for real serving verification")

    command = _server_command(_store(root), port)
    _assert_port_free(port)
    event_id = _next_event_id(root)
    _reserve_event(
        root,
        event_id,
        SERVING_RESERVATION_USD,
        "serving_verification",
        provider="mixed",
        model="wmo-routing-endpoint",
        benchmark="synthetic-serving",
    )
    log_path = _store(root) / "serving" / "requests.jsonl"
    start_rows = len(log_path.read_text(encoding="utf-8").splitlines()) if log_path.is_file() else 0
    server_log = root / "serving" / "wmo-serve.log"
    server_log.parent.mkdir(parents=True, exist_ok=True)
    process: subprocess.Popen[bytes] | None = None
    failure: Exception | None = None
    result: dict[str, object] = {}
    with server_log.open("ab") as handle:
        try:
            process = subprocess.Popen(
                command,
                cwd=Path.cwd(),
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}",
                timeout=REQUEST_TIMEOUT_S,
            ) as client:
                _wait_ready(
                    client,
                    process,
                    {
                        ENDPOINT,
                        OPENAI_PROBE,
                        ANTHROPIC_PROBE,
                        FALLBACK_PROBE,
                        CACHE_AWARE_PROBE,
                    },
                )
                result = _exercise(client)
        except Exception as exc:  # noqa: BLE001 - persist paid failed attempts
            failure = exc
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)

    rows = _rows_after(log_path, start_rows)
    try:
        result = _validate_rows(
            rows,
            result,
            policy.default_model,
            {entry.name for entry in policy.pool},
        )
    except Exception as validation_error:  # noqa: BLE001 - preserve the first failure
        if failure is None:
            failure = validation_error
    total_cost = sum(
        _number(row, "cost_usd")
        + _number(row, "router_cost_usd")
        + _number(row, "compressor_cost_usd")
        for row in rows
    )
    result.update(
        {
            "protocol": "coding-router-serving-v1",
            "completion_status": "failed" if failure is not None else "passed",
            "requests": len(rows),
            "total_cost_usd": total_cost,
            "server_log": str(server_log),
            "request_log": str(log_path),
            "error": (f"{type(failure).__name__}: {failure}" if failure is not None else None),
        }
    )
    _write_json(root / "serving" / f"result-{event_id.rsplit(':', 1)[1]}.json", result)
    _complete_event(
        root,
        event_id,
        {
            "phase": "serving_verification",
            "provider": "mixed",
            "model": "wmo-routing-endpoint",
            "benchmark": "synthetic-serving",
            "model_cost_usd": total_cost,
            "requests": len(rows),
            "completion_status": result["completion_status"],
            "error": result["error"],
        },
    )
    if total_cost > SERVING_RESERVATION_USD:
        raise ValueError("serving verification exceeded its conservative reservation")
    if failure is not None:
        raise failure


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(".wmo") / "experiments" / EXPERIMENT_ID,
    )
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    load_env_file(PROVIDER_ENV_SOURCE)
    load_env_file(AZURE_ENV_SOURCE)
    args = _parse_args()
    root = cast("Path", args.root).resolve()
    port = cast("int", args.port)
    if not 1024 <= port <= 65535:
        raise ValueError("--port must be between 1024 and 65535")
    if args.phase == "prepare":
        _prepare(root)
    else:
        _run(root, port)


if __name__ == "__main__":
    main()
