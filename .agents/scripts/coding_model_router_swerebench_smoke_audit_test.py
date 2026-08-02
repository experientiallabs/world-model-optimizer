"""Tests for the frozen SWE-rebench smoke artifact audit."""

from __future__ import annotations

import importlib.util
import io
import json
import tarfile
from pathlib import Path
from types import ModuleType

import pytest


def _load_module() -> ModuleType:
    path = Path(__file__).with_name("coding_model_router_swerebench_smoke_audit.py")
    spec = importlib.util.spec_from_file_location("swerebench_smoke_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _add(archive: tarfile.TarFile, name: str, content: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(content)
    archive.addfile(member, io.BytesIO(content))


def _archive(path: Path, module: ModuleType, effort: str) -> None:
    config = f'''model = "{module.MODEL}"
num_tasks = 2
num_rollouts = 1

[env]
[env.taskset]
id = "swerebench-v2-v1"
[env.agent]
max_turns = 20
max_output_tokens = 131072
[env.agent.harness]
id = "mini_swe_agent"
version = "2.4.5"

[sampling]
temperature = 1.0
reasoning_effort = "{effort}"
max_tokens = 32768
'''.encode()
    rows = []
    for index, task in enumerate(module.TASKS):
        call = {
            "model": module.MODEL,
            "endpoint": "/responses",
            "sampling": {"reasoning_effort": effort, "max_tokens": 32768},
            "usage": {
                "prompt_tokens": 10 + index,
                "cached_input_tokens": 20,
                "completion_tokens": 30,
                "reasoning_tokens": 5,
            },
        }
        rows.append(
            {
                "ok": True,
                "errors": [],
                "traces": [
                    {
                        "ok": True,
                        "errors": [],
                        "is_completed": True,
                        "stop_condition": "agent_completed",
                        "task": {"data": {"name": task}},
                        "verifiers": {"commit": module.VERIFIERS_COMMIT},
                        "agent": {
                            "config": {
                                "model": module.MODEL,
                                "max_turns": 20,
                                "sampling": {
                                    "reasoning_effort": effort,
                                    "max_tokens": 32768,
                                },
                            }
                        },
                        "rewards": {"solved": {"score": float(index)}},
                        "timing": {"scoring": {"start": 1.0, "end": 2.0}},
                        "calls": [call],
                        "info": {"patch": f"patch-{task}"},
                    }
                ],
            }
        )
    traces = b"\n".join(json.dumps(row).encode() for row in rows) + b"\n"
    with tarfile.open(path, "w:gz") as archive:
        root = tarfile.TarInfo(effort)
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        _add(archive, f"{effort}/eval.log", b"complete\n")
        _add(archive, f"{effort}/config.toml", config)
        _add(archive, f"{effort}/traces.jsonl", traces)


def _state(path: Path, module: ModuleType, archives: dict[str, Path]) -> None:
    state = {
        "account_cap": 1000,
        "task_ids": list(module.TASKS),
        "efforts": list(module.EFFORTS),
        "expected_cells": 4,
        "scientific_attempt": 0,
        "sandbox_terminated": True,
        "owned_eval_terminated": True,
        "sandbox_id": "owned-sandbox",
        "infrastructure_resume": 3,
        "provider_inference_calls": 0,
    }
    for effort, archive in archives.items():
        state[f"{effort}_archive_sha256"] = module._sha256_path(archive)
    path.write_text(json.dumps(state), encoding="utf-8")


def test_audit_validates_calls_cost_and_stale_state(tmp_path: Path) -> None:
    module = _load_module()
    archives = {effort: tmp_path / f"{effort}.tar.gz" for effort in module.EFFORTS}
    for effort, path in archives.items():
        _archive(path, module, effort)
    state = tmp_path / "state.json"
    _state(state, module, archives)
    report = module.audit(archives, state, 405.33)
    assert report["valid"] is True
    assert report["cells"] == 4
    assert report["provider_calls"] == 4
    assert report["usage"] == {
        "prompt_tokens": 42,
        "cached_input_tokens": 80,
        "completion_tokens": 120,
        "reasoning_tokens": 20,
    }
    assert report["cost_usd"] == pytest.approx(0.00077)
    assert report["state_correction"]["stale_value"] == 0
    assert report["state_correction"]["audited_value"] == 4


def test_audit_rejects_archive_hash_drift(tmp_path: Path) -> None:
    module = _load_module()
    archives = {effort: tmp_path / f"{effort}.tar.gz" for effort in module.EFFORTS}
    for effort, path in archives.items():
        _archive(path, module, effort)
    state = tmp_path / "state.json"
    _state(state, module, archives)
    payload = json.loads(state.read_text(encoding="utf-8"))
    payload["max_archive_sha256"] = "0" * 64
    state.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="archive hash differs"):
        module.audit(archives, state, 405.33)
