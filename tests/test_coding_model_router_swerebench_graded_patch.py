"""Focused tests for the graded SWE-rebench taskset patch."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).parents[1] / ".agents" / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "coding_model_router_swerebench_graded_patch",
    SCRIPTS / "coding_model_router_swerebench_graded_patch.py",
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _fixture() -> str:
    return "\n# anchor\n".join(old for old, _ in module.REPLACEMENTS)


def test_patch_applies_each_frozen_change_once() -> None:
    source = _fixture()
    patched = module._patch_text(source)

    assert all(new in patched for _, new in module.REPLACEMENTS)
    assert patched != source
    assert "task_json: str | None = None" in patched
    assert "return graded_f2p_reward(status_map, self.data.fail_to_pass)" in patched


def test_graded_reward_uses_only_fail_to_pass_fraction() -> None:
    patched = module._patch_text(_fixture())
    function = patched.split("def graded_f2p_reward", 1)[1].split(
        "return passed / len(expected)", 1
    )[0]

    assert "return passed / len(expected)" in patched
    assert "for t in fail_to_pass" in function
    assert "pass_to_pass" not in function


def test_patch_fails_closed_when_an_anchor_changes() -> None:
    source = _fixture().replace(module.REWARD_OLD, "        return 0.0\n")
    try:
        module._patch_text(source)
    except ValueError as error:
        assert str(error) == "pinned taskset patch anchor changed"
    else:
        raise AssertionError("changed taskset anchor was accepted")
