"""Focused tests for the graded SWE-rebench cohort freeze."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).parents[1] / ".agents" / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "coding_model_router_graded_swerebench_prepare",
    SCRIPTS / "coding_model_router_graded_swerebench_prepare.py",
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_repository_split_is_deterministic_and_disjoint() -> None:
    rows = [
        {"task_id": f"owner__repo-{repo}.{task}", "repository": f"owner/repo-{repo}"}
        for repo in range(100)
        for task in range(3)
    ]
    first = module._split(rows)
    second = module._split(list(reversed(rows)))
    first_dev, first_confirmation = first
    second_dev, second_confirmation = second

    assert {row["task_id"] for row in first_dev} == {
        row["task_id"] for row in second_dev
    }
    assert {row["task_id"] for row in first_confirmation} == {
        row["task_id"] for row in second_confirmation
    }
    assert {row["repository"] for row in first_dev}.isdisjoint(
        {row["repository"] for row in first_confirmation}
    )
    assert 0.65 <= len(first_dev) / len(rows) <= 0.75


def test_target_exclusions_use_only_label_free_fields(tmp_path: Path) -> None:
    index = tmp_path / "index.json"
    prompts = tmp_path / "prompts.json"
    index.write_text(
        json.dumps(
            {
                "rows": [
                    {"repository": f"Owner/Repo-{row}", "task_id": f"task-{row}"}
                    for row in range(113)
                ]
            }
        ),
        encoding="utf-8",
    )
    prompts.write_text(
        json.dumps(
            {
                "rows": [
                    {"task_id": f"task-{row}", "text": f" Repair   BUG {row} "}
                    for row in range(113)
                ],
                "target_reward_fields_accessed": False,
                "target_cost_fields_accessed": False,
            }
        ),
        encoding="utf-8",
    )

    repositories, prompt_hashes = module._target_exclusions(index, prompts)

    assert "owner/repo-0" in repositories
    assert module._digest("repair bug 0") in prompt_hashes
    assert len(repositories) == len(prompt_hashes) == 113


def test_target_exclusions_reject_accessed_outcomes(tmp_path: Path) -> None:
    index = tmp_path / "index.json"
    prompts = tmp_path / "prompts.json"
    index.write_text(
        json.dumps({"rows": [{"repository": f"repo-{row}"} for row in range(113)]}),
        encoding="utf-8",
    )
    prompts.write_text(
        json.dumps(
            {
                "rows": [{"text": f"task-{row}"} for row in range(113)],
                "target_reward_fields_accessed": True,
                "target_cost_fields_accessed": False,
            }
        ),
        encoding="utf-8",
    )

    try:
        module._target_exclusions(index, prompts)
    except ValueError as error:
        assert str(error) == "target prompt view accessed rewards"
    else:
        raise AssertionError("accessed target rewards were accepted")
