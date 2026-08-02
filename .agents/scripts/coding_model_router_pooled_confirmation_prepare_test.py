"""Unit tests for the fresh pooled-uplift confirmation selector."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_module() -> ModuleType:
    directory = Path(__file__).parent
    sys.path.insert(0, str(directory))
    path = directory / "coding_model_router_pooled_confirmation_prepare.py"
    spec = importlib.util.spec_from_file_location("pooled_confirmation_prepare", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_module()


def test_metadata_normalizes_repositories_and_prompts() -> None:
    repositories, prompts, task_ids = module._metadata(
        [
            {
                "task_id": "task-1",
                "repository": "Owner/Repo.git",
                "prompt": " Fix   this\nparser ",
            }
        ]
    )
    assert repositories == {"owner/repo"}
    assert prompts == {module.base._digest("fix this parser")}
    assert task_ids == {"task-1"}


def test_load_exclusions_requires_complete_frozen_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pooled = tmp_path / "pooled.json"
    pooled.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": f"pooled-{index}",
                        "repository": f"pooled/repo-{index}",
                        "prompt": f"Pooled prompt {index}",
                    }
                    for index in range(400)
                ]
            }
        ),
        encoding="utf-8",
    )
    swesmith = tmp_path / "swesmith.jsonl"
    swesmith.write_text(
        "".join(
            json.dumps(
                {
                    "task_id": f"swesmith-{index}",
                    "repository": f"swesmith/repo-{index}",
                    "prompt": f"SWE-smith prompt {index}",
                }
            )
            + "\n"
            for index in range(1_551)
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "POOLED_EXCLUSIONS_SHA256", module._sha256(pooled))
    monkeypatch.setattr(module, "SWESMITH_TASKS_SHA256", module._sha256(swesmith))
    repositories, prompts, task_ids, audit = module._load_exclusions(pooled, swesmith)
    assert len(repositories) == 1_951
    assert len(prompts) == 1_951
    assert len(task_ids) == 1_951
    assert audit["pooled_tasks"] == 400
    assert audit["swesmith_tasks"] == 1_551
