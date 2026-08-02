"""Unit tests for audited Moonshiner attempt-zero reuse."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    path = Path(__file__).with_name("coding_model_router_moonshiner_reuse.py")
    spec = importlib.util.spec_from_file_location("moonshiner_reuse", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_module()


def test_assemble_reuses_only_complete_tasks(tmp_path: Path) -> None:
    fingerprint = "seed-fingerprint"
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        json.dumps(
            {
                "tasks": [
                    {"task_id": "complete", "seed_fingerprint": fingerprint},
                    {"task_id": "partial", "seed_fingerprint": fingerprint},
                ]
            }
        ),
        encoding="utf-8",
    )
    source_root = tmp_path / "source"
    traces = source_root / "traces"
    rows = []
    for task_id, arms in (
        ("complete", sorted(module.ARMS)),
        ("partial", ["luna-max"]),
    ):
        for arm in arms:
            relative = f"{arm}/raw/{task_id}.jsonl"
            trace = traces / relative
            trace.parent.mkdir(parents=True, exist_ok=True)
            trace.write_text(f"{task_id}:{arm}\n", encoding="utf-8")
            rows.append(
                {
                    "task_id": task_id,
                    "arm": arm,
                    "cell_id": f"{task_id}:{arm}",
                    "model": "gpt-5.6-luna",
                    "observed_model": "gpt-5.6-luna",
                    "model_attested": True,
                    "protected_intact": True,
                    "workspace_removed": True,
                    "target_outcomes_used": False,
                    "seed_fingerprint": fingerprint,
                    "trace_path": relative,
                    "raw_sha256": module._sha256(trace),
                }
            )
    source = source_root / "outcomes.jsonl"
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output = tmp_path / "output"
    module.assemble(corpus, corpus, [source], output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["reusable_tasks"] == 1
    assert manifest["reused_cells"] == 5
    assert manifest["missing_task_ids"] == ["partial"]
    migrated = [json.loads(line) for line in (output / "outcomes.jsonl").read_text().splitlines()]
    assert all(row["attempt"] == 0 for row in migrated)
    assert all(row["reused_without_provider_call"] is True for row in migrated)
