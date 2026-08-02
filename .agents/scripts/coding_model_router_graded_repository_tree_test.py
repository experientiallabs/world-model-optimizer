"""Tests for the frozen repository-tree feature contract."""

from __future__ import annotations

import numpy as np
import pytest
from coding_model_router_graded_wiserouter import _structural
from coding_model_router_graded_repository_tree import (
    RawTreeEntry,
    TreeFile,
    feature_blocks,
    feature_views,
    validate_tree,
)


def _tree() -> tuple[TreeFile, ...]:
    return validate_tree(
        [
            RawTreeEntry("src/HTTPServer.py", "blob", "100755", 400),
            RawTreeEntry("tests/test_http_server.py", "blob", "100644", 200),
            RawTreeEntry("docs/server.md", "blob", "100644", 100),
            RawTreeEntry("pyproject.toml", "blob", "100644", 80),
            RawTreeEntry("vendor/ignored.py", "blob", "100644", 10_000),
            RawTreeEntry("src", "tree", "040000", None),
        ],
        truncated=False,
    )


def test_tree_validation_normalizes_sorts_and_excludes_vendor() -> None:
    files = _tree()
    assert [file.path for file in files] == [
        "docs/server.md",
        "pyproject.toml",
        "src/httpserver.py",
        "tests/test_http_server.py",
    ]
    assert files[2].executable is True


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        (RawTreeEntry("../secret", "blob", "100644", 1), "traversal"),
        (RawTreeEntry("module", "commit", "160000", None), "submodule"),
        (RawTreeEntry("file.py", "blob", "100644", -1), "nonnegative"),
    ],
)
def test_tree_validation_fails_closed(entry: RawTreeEntry, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_tree([entry], truncated=False)


def test_tree_validation_rejects_truncation_and_normalized_duplicates() -> None:
    with pytest.raises(ValueError, match="truncated"):
        validate_tree([], truncated=True)
    entries = [
        RawTreeEntry("Src/File.py", "blob", "100644", 1),
        RawTreeEntry("src/file.py", "blob", "100644", 1),
    ]
    with pytest.raises(ValueError, match="unique"):
        validate_tree(entries, truncated=False)


def test_feature_blocks_are_finite_deterministic_and_nested() -> None:
    issue = "Fix `src/HTTPServer.py` parser in HTTPServer and add test_http_server.py"
    first = feature_blocks(_tree(), issue=issue, language="Python")
    second = feature_blocks(_tree(), issue=issue, language="Python")
    first_views = feature_views(first)
    second_views = feature_views(second)
    assert [view.shape for view in first_views] == [(61,), (100,), (115,)]
    for left, right in zip(first_views, second_views, strict=True):
        assert np.array_equal(left, right)
        assert np.isfinite(left).all()
    assert np.array_equal(first_views[1][:61], first_views[0])
    assert np.array_equal(first_views[2][:100], first_views[1])


def test_localization_changes_with_issue_but_structure_does_not() -> None:
    matching = feature_blocks(_tree(), issue="src/HTTPServer.py parser", language="Python")
    unrelated = feature_blocks(_tree(), issue="database migration queue", language="Python")
    assert np.array_equal(matching.structure, unrelated.structure)
    assert not np.array_equal(matching.localization, unrelated.localization)


def test_empty_issue_and_missing_language_are_explicit_and_finite() -> None:
    blocks = feature_blocks(_tree(), issue="", language="Unknown")
    assert np.isfinite(blocks.structure).all()
    assert np.isfinite(blocks.localization).all()
    assert np.isfinite(blocks.prompt_shape).all()


def test_prompt_shape_exactly_matches_the_frozen_existing_block() -> None:
    issue = "Fix `src/server.py` after traceback in a Rust package test"
    blocks = feature_blocks(_tree(), issue=issue, language="Python")
    assert np.array_equal(blocks.prompt_shape, np.asarray(_structural(issue)))
