"""Tests for exact repository-tree source projection and coverage."""

from __future__ import annotations

import pytest
from coding_model_router_graded_repository_tree_acquire import (
    DatasetTask,
    _coverage_report,
    validate_projection,
)


def _manifest() -> list[dict[str, object]]:
    return [
        {
            "task_id": "owner__repo-1",
            "repository": "owner/repo",
            "language": "python",
            "prompt": "Fix parser",
            "image_name": "docker.io/example:1",
        }
    ]


def _dataset() -> list[dict[str, object]]:
    return [
        {
            "instance_id": "owner__repo-1",
            "repo": "owner/repo",
            "language": "Python",
            "problem_statement": "Fix parser",
            "base_commit": "abc123",
            "image_name": "docker.io/example:1",
        },
        {
            "instance_id": "other__repo-2",
            "repo": "other/repo",
            "language": "Python",
            "problem_statement": "Ignored",
            "base_commit": "def456",
            "image_name": "docker.io/example:2",
        },
    ]


def test_projection_uses_only_exact_retained_identities() -> None:
    result = validate_projection(_manifest(), _dataset())
    assert result.tasks == (
        DatasetTask(
            task_id="owner__repo-1",
            repository="owner/repo",
            language="Python",
            prompt="Fix parser",
            base_commit="abc123",
            image_name="docker.io/example:1",
        ),
    )
    assert result.failures == ()


@pytest.mark.parametrize("field", ["repo", "language", "problem_statement", "image_name"])
def test_projection_rejects_manifest_mismatch(field: str) -> None:
    rows = _dataset()
    rows[0][field] = "changed"
    result = validate_projection(_manifest(), rows)
    assert result.tasks == ()
    assert result.failures[0]["reason_type"].startswith("source-identity-mismatch:")


def test_projection_rejects_missing_and_duplicate_tasks() -> None:
    missing = validate_projection(_manifest(), [])
    assert missing.tasks == ()
    assert missing.failures[0]["reason_type"] == "source-row-missing"
    with pytest.raises(ValueError, match="repeats"):
        validate_projection(_manifest(), [_dataset()[0], _dataset()[0]])


def test_coverage_is_label_free_and_stratified() -> None:
    tasks = [
        DatasetTask("one", "repo/a", "Python", "p", "a", "i"),
        DatasetTask("two", "repo/a", "Python", "p", "b", "i"),
        DatasetTask("three", "repo/b", "Go", "p", "c", "i"),
    ]
    report = _coverage_report(tasks, {"one", "three"}, [{"task_id": "two"}])
    assert report["coverage"] == pytest.approx(2 / 3)
    assert report["outcomes_joined"] is False
    assert report["provider_calls"] == 0
    assert report["by_language"]["python"]["coverage"] == 0.5
