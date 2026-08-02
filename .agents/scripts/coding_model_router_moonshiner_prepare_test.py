"""Unit tests for the Moonshiner external-corpus freezer."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    path = Path(__file__).with_name("coding_model_router_moonshiner_prepare.py")
    spec = importlib.util.spec_from_file_location("moonshiner_prepare", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_module()


def _seed(root: Path, task_id: str, *, language: str, category: str) -> Path:
    directory = root / task_id
    (directory / "files").mkdir(parents=True)
    (directory / "files/test.sh").write_text("exit 1\n", encoding="utf-8")
    (directory / "task.json").write_text(
        json.dumps(
            {
                "id": task_id,
                "prompt": f"Repair {task_id}",
                "lang": language,
                "category": category,
                "verify_cmd": "bash test.sh",
                "verify_timeout": 60,
            }
        ),
        encoding="utf-8",
    )
    (directory / "reference_fix.patch").write_text("patch\n", encoding="utf-8")
    return directory


def test_eligible_seed_rejects_network_and_tool_behavior(tmp_path: Path) -> None:
    accepted = _seed(tmp_path, "accepted", language="python", category="debug")
    row = module._eligible_seed(
        accepted,
        languages=module.DEFAULT_LANGUAGES,
        max_files=10,
        max_bytes=1_000,
        max_verify_timeout_s=120,
    )
    assert row is not None

    network = _seed(tmp_path, "network", language="python", category="debug")
    payload = json.loads((network / "task.json").read_text(encoding="utf-8"))
    payload["network"] = "loopback-only"
    (network / "task.json").write_text(json.dumps(payload), encoding="utf-8")
    assert (
        module._eligible_seed(
            network,
            languages=module.DEFAULT_LANGUAGES,
            max_files=10,
            max_bytes=1_000,
            max_verify_timeout_s=120,
        )
        is None
    )

    behavior = _seed(tmp_path, "behavior", language="bash", category="tool")
    payload = json.loads((behavior / "task.json").read_text(encoding="utf-8"))
    payload["kind"] = "tool_behavior"
    (behavior / "task.json").write_text(json.dumps(payload), encoding="utf-8")
    assert (
        module._eligible_seed(
            behavior,
            languages=module.DEFAULT_LANGUAGES,
            max_files=10,
            max_bytes=1_000,
            max_verify_timeout_s=120,
        )
        is None
    )


def test_round_robin_is_deterministic_and_spreads_buckets(tmp_path: Path) -> None:
    rows = []
    for index, (language, category) in enumerate(
        [
            ("python", "debug"),
            ("python", "debug"),
            ("python", "build"),
            ("bash", "debug"),
            ("bash", "build"),
        ]
    ):
        directory = _seed(
            tmp_path,
            f"task-{index}",
            language=language,
            category=category,
        )
        row = module._eligible_seed(
            directory,
            languages=module.DEFAULT_LANGUAGES,
            max_files=10,
            max_bytes=1_000,
            max_verify_timeout_s=120,
        )
        assert row is not None
        rows.append(row)
    first = module._round_robin_candidates(rows, count=4, seed=17)
    second = module._round_robin_candidates(rows, count=4, seed=17)
    assert [row.task_id for row in first] == [row.task_id for row in second]
    assert len({(row.language, row.category) for row in first}) == 4


def test_target_metadata_normalizes_prompt_whitespace(tmp_path: Path) -> None:
    path = tmp_path / "target.json"
    path.write_text(
        json.dumps({"rows": [{"id": "target-1", "text": " Repair   This\nBug "}]}),
        encoding="utf-8",
    )
    ids, prompts = module._target_metadata(path)
    assert ids == {"target-1"}
    assert prompts == {"repair this bug"}


def test_source_task_ids_reads_shard_inventory(tmp_path: Path) -> None:
    path = tmp_path / "dataset-manifest.json"
    path.write_text(
        json.dumps(
            {
                "shards": [
                    {"path": "data/one.parquet", "tasks": ["task-a", "task-b"]},
                    {"path": "data/two.parquet", "tasks": ["task-b", "task-c"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert module._source_task_ids(path) == {"task-a", "task-b", "task-c"}
