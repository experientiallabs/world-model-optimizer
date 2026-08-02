"""Unit tests for the SWE-smith difficulty corpus preparer."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    path = Path(__file__).with_name("coding_model_router_swesmith_prepare.py")
    spec = importlib.util.spec_from_file_location("swesmith_prepare", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_module()


def test_decode_messages_accepts_one_or_two_json_layers() -> None:
    messages = [{"role": "user", "content": "Repair it"}]
    once = json.dumps(messages)
    twice = json.dumps(once)
    assert module._decode_messages(once) == messages
    assert module._decode_messages(twice) == messages


def test_user_text_reads_only_initial_user_message() -> None:
    messages = [
        {"role": "system", "content": "Ignore me"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "First"},
                {"type": "image", "url": "ignored"},
                {"type": "text", "text": "task"},
            ],
        },
        {"role": "assistant", "content": "Reasoning"},
        {"role": "user", "content": "Verifier feedback"},
    ]
    assert module._user_text(messages) == "First\ntask"


def test_problem_statement_removes_fixed_wrapper() -> None:
    wrapped = "Header\n<pr_description>\nFix the parser\n</pr_description>\nFooter"
    assert module._problem_statement(wrapped) == "Fix the parser"
    assert module._problem_statement("  Bare task  ") == "Bare task"


def test_normalized_text_collapses_formatting_variants() -> None:
    assert module._normalized_text("  Fix   the\nparser ") == "fix the parser"


def test_repository_is_derived_from_instance_prefix() -> None:
    assert module._repository("Owner__Project.abc123") == "owner/project"


def test_exclusions_use_label_free_target_and_development_metadata(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.json"
    target.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "repository": f"target/repo-{index}",
                        "text": f"Target prompt {index}",
                    }
                    for index in range(113)
                ],
                "target_reward_fields_accessed": False,
                "target_cost_fields_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    development = tmp_path / "development.json"
    development.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "repository": f"development/repo-{index}",
                        "prompt": f"Development prompt {index}",
                    }
                    for index in range(190)
                ]
            }
        ),
        encoding="utf-8",
    )
    repositories, prompts, hashes = module._exclusions(target, development)
    assert len(repositories) == 303
    assert len(prompts) == 303
    assert hashes["target_view_sha256"] == module._sha256(target)
    assert hashes["development_tasks_sha256"] == module._sha256(development)
