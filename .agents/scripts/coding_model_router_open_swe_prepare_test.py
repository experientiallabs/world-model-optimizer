"""Tests for compact Open-SWE task and profile preparation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

duckdb = pytest.importorskip("duckdb")


def _module() -> ModuleType:
    path = Path(__file__).with_name("coding_model_router_open_swe_prepare.py")
    spec = importlib.util.spec_from_file_location("coding_model_router_open_swe_prepare", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_profile_teacher_excludes_selected_paired_outcome_tasks(tmp_path: Path) -> None:
    module = _module()
    connection = duckdb.connect()
    connection.execute(
        """
        CREATE TABLE tasks (
            instance_id VARCHAR,
            repo VARCHAR,
            language VARCHAR,
            problem_statement VARCHAR,
            difficulty VARCHAR,
            intent_completeness VARCHAR,
            pr_categories VARCHAR[]
        )
        """
    )
    connection.execute(
        """
        INSERT INTO tasks VALUES
        ('paired', 'repo-a', 'python', 'paired prompt', 'hard', 'complete', ['minor_bug']),
        ('teacher', 'repo-b', 'rust', 'teacher prompt', 'easy', 'partial', ['core_feat'])
        """
    )
    connection.execute(
        """
        CREATE TABLE outcomes (
            instance_id VARCHAR,
            scaffold VARCHAR,
            model_mode VARCHAR
        )
        """
    )
    connection.execute(
        """
        INSERT INTO outcomes VALUES
        ('paired', 'openhands', 'weak'),
        ('paired', 'openhands', 'strong')
        """
    )
    output = tmp_path / "teacher.json"
    stats = module._write_profile_teacher_source(
        connection,
        {
            "scaffold": "openhands",
            "weak_model_mode": "weak",
            "strong_model_mode": "strong",
        },
        output,
    )
    rows = json.loads(output.read_text(encoding="utf-8"))
    assert [row["instance_id"] for row in rows] == ["teacher"]
    assert rows[0]["pr_categories"] == ["core_feat"]
    assert stats["tasks"] == 1
