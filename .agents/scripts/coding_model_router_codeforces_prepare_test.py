"""Unit tests for the Codeforces trace-corpus freezer."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    path = Path(__file__).with_name("coding_model_router_codeforces_prepare.py")
    spec = importlib.util.spec_from_file_location("codeforces_prepare", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_module()


def test_bucket_maps_hard_indices() -> None:
    assert module._bucket("C2") == "C"
    assert module._bucket("D") == "D"
    assert module._bucket("E1") == "E"
    assert module._bucket("F") == "F+"
    assert module._bucket("H2") == "F+"
    assert module._bucket("B") is None


def test_candidate_ignores_generation_and_freezes_tests() -> None:
    row = {
        "id": "1/C",
        "contest_id": "1",
        "index": "C",
        "title": "Hard task",
        "prompt": "Solve this problem",
        "generation": "must not be used",
        "problem_type": "diff",
        "interaction_format": "",
        "public_tests": {
            "input": [f"{index}\n" for index in range(8)],
            "output": [f"{index}\n" for index in range(8)],
        },
        "private_tests": {"input": [], "output": []},
        "generated_tests": {"input": [], "output": []},
        "accepted_solutions": [
            {
                "programmingLanguage": "Python 3",
                "code": "print(input())",
            }
        ],
    }
    candidate = module._candidate(row, seed=17)
    assert candidate is not None
    assert candidate.task_id == "1/C"
    assert len(candidate.tests) == 8
    assert "must not be used" not in candidate.prompt


def test_output_comparison_uses_tokens() -> None:
    assert module._output_matches("1  2\n3\n", "1 2 3")
    assert not module._output_matches("1 2", "1 3")


def test_excluded_task_ids_require_unique_valid_rows(tmp_path: Path) -> None:
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"tasks": [{"task_id": "1/C"}, {"task_id": "2/D"}]}))
    assert module._excluded_task_ids(path) == {"1/C", "2/D"}
    path.write_text(json.dumps({"tasks": [{"task_id": "1/C"}, {"task_id": "1/C"}]}))
    try:
        module._excluded_task_ids(path)
    except ValueError as error:
        assert "invalid or duplicate" in str(error)
    else:
        raise AssertionError("duplicate task IDs should be rejected")
