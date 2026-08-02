"""Pure unit tests for the paired Moonshiner effort-matrix worker."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_module() -> ModuleType:
    path = Path(__file__).with_name("coding_model_router_moonshiner_matrix.py")
    spec = importlib.util.spec_from_file_location("moonshiner_matrix", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_module()


def test_schedule_has_paired_arms_and_is_deterministic() -> None:
    first = module._schedule(["b", "a", "c"], seed=17)
    second = module._schedule(["c", "a", "b"], seed=17)
    assert [(cell.task_id, cell.arm) for cell in first] == [
        (cell.task_id, cell.arm) for cell in second
    ]
    assert len(first) == 6
    assert all(cell.attempt == 0 for cell in first)
    assert all(cell.cell_id.endswith(":attempt-0") for cell in first)
    for task_id in {"a", "b", "c"}:
        assert {cell.arm for cell in first if cell.task_id == task_id} == {
            "luna-xhigh",
            "luna-max",
        }


def test_cost_uses_provider_report_when_present() -> None:
    cost, accounting = module._cost({"cost": {"total_cost_usd": 1.25}})
    assert cost == 1.25
    assert accounting == "provider_reported"


def test_cost_estimates_tokens_without_double_charging_cache() -> None:
    cost, accounting = module._cost(
        {
            "input_tokens": 1_000_000,
            "cached_input_tokens": 400_000,
            "output_tokens": 100_000,
        }
    )
    assert cost == 1.24
    assert accounting == "trace_token_estimate"


def test_extract_upstream_model_from_responses_sse() -> None:
    body = (
        b'event: response.created\n'
        b'data: {"type":"response.created","response":{"model":"gpt-5.6-luna"}}\n\n'
    )
    assert module._extract_upstream_model(body) == "gpt-5.6-luna"


def test_pi_usage_aggregates_final_assistant_messages() -> None:
    events = "\n".join(
        [
            '{"type":"message_start","message":{"role":"assistant",'
            '"usage":{"input":0,"output":0}}}',
            '{"type":"message_end","message":{"role":"assistant",'
            '"usage":{"input":3,"output":10,"cacheRead":20,"cacheWrite":30,'
            '"reasoning":4}}}',
            '{"type":"turn_end","message":{"role":"assistant",'
            '"usage":{"input":3,"output":10,"cacheRead":20,"cacheWrite":30}}}',
            '{"type":"message_end","message":{"role":"assistant",'
            '"usage":{"input":2,"output":7,"cacheRead":40,"cacheWrite":5,'
            '"reasoning":1}}}',
        ]
    )
    assert module._pi_usage_from_stream(events) == {
        "input_tokens": 100,
        "cached_input_tokens": 60,
        "output_tokens": 17,
        "reasoning_tokens": 5,
        "turns": 2,
    }


def test_bounded_pi_command_caps_output_file_size() -> None:
    assert module._bounded_pi_command(["bwrap", "--", "pi"]) == [
        "prlimit",
        "--fsize=67108864:67108864",
        "--",
        "bwrap",
        "--",
        "pi",
    ]


def test_reservation_releases_when_a_cell_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingTeacher:
        def __init__(self, config: object, role: object) -> None:
            del config, role

        def preflight(self, *, require_auth: bool) -> None:
            assert require_auth is True

    generate = ModuleType("generate_traces")

    def fail_trace(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise LookupError("synthetic failure")

    setattr(generate, "trace_task", fail_trace)
    runtime = ModuleType("runtimes.pi")
    setattr(runtime, "PiRuntime", FailingTeacher)
    common = ModuleType("common")
    setattr(common, "remove_workspace", lambda path: None)
    modules = {
        "common": common,
        "generate_traces": generate,
        "runtimes.pi": runtime,
    }
    monkeypatch.setattr(module.importlib, "import_module", modules.__getitem__)
    monkeypatch.setattr(
        module,
        "_moonshiner_config",
        lambda root, effort: {"teacher": {"reasoning": effort}, "root": root},
    )
    monkeypatch.setattr(module, "_seed", lambda corpus, root, task_id: corpus[task_id])
    state = module.MatrixState(tmp_path / "output", ceiling_usd=200.0)
    with pytest.raises(LookupError, match="synthetic failure"):
        module._run_cell(
            module.Cell("task", "luna-xhigh", "xhigh"),
            root=tmp_path,
            corpus_by_id={"task": {"task_id": "task", "seed_relpath": "seed"}},
            output=tmp_path / "output",
            state=state,
        )
    assert state.reserved == 0.0
