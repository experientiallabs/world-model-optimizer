"""Tests for the pinned SWE-rebench Docker image adapter."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    path = Path(__file__).with_name("coding_model_router_swerebench_docker_adapter.py")
    spec = importlib.util.spec_from_file_location("swerebench_docker_adapter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adapt_source_changes_only_image_resolution() -> None:
    module = _load_module()
    source = '''

def repo_workdir(repo: str) -> str:
    return repo

value = [
                    image=row["image_name"],
]
'''
    adapted = module.adapt_source(source)
    assert 'image=_coding_router_docker_image(row["image_name"]),' in adapted
    assert 'return f"docker.io/swerebenchv2/{suffix}"' in adapted
    assert "def repo_workdir(repo: str) -> str:" in adapted
    assert adapted.count("def repo_workdir") == 1


def test_adapt_source_rejects_unpinned_shape() -> None:
    module = _load_module()
    source = "\ndef repo_workdir(repo: str) -> str:\n    return repo\n"
    try:
        module.adapt_source(source)
    except ValueError as error:
        assert "image construction line" in str(error)
    else:
        raise AssertionError("adapter accepted a source without the pinned image line")
