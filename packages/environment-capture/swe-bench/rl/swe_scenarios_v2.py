#!/usr/bin/env python3
"""Upgrade the pinned swe scenario files to ScenarioSuite v2: {task, seed_state (+ rubrics already pinned)}.

The D62/D65/D66 reward-substrate fix in file form. For each pinned v1 scenario (identity
= its provenance trace id, never re-derived), join the corpus trace and attach:

- ``seed_state`` — the recorded environment's case facts (what its tools actually
  returned), rendered exactly like the D65 probe's ``_seed_state_from``: seeding pins the
  imagined world so identical actions get identical verdicts (unseeded worlds scored the
  same rollout 0.95→0.15 across sessions, D62).
- ``rubric`` stays exactly as pinned — swe's rubrics are real SWE-bench FAIL_TO_PASS
  tests (D74), already joined by the coordinator; v2 only adds the seeded world.

Emits ``scenarios_train_v2.jsonl`` / ``scenarios_eval_v2.jsonl`` next to the v1 pins.
Provenance and task text are byte-identical to v1 — v2 adds fields, never re-splits.

Run from the repo root (corpus must be downloaded, see the package README):
    uv run python packages/environment-capture/swe-bench/rl/swe_scenarios_v2.py
"""

from __future__ import annotations

import json
from pathlib import Path

from wmh.core.types import Trace
from wmh.ingest import get_adapter

_DIR = Path(__file__).resolve().parent
_TRACES_PATH = _DIR.parent / "traces.otel.jsonl"
FACT_CHARS = 600  # per-fact cap, matches the D65 probe


def seed_scratchpad(trace: Trace) -> str:
    """The recorded world's case facts, in the probe's exact rendering."""
    facts = [
        f"- {step.action.name or 'message'} -> {step.observation.content[:FACT_CHARS]}"
        for step in trace.steps
    ]
    return (
        "FIXED CASE FACTS for this task's world (the environment's database is exactly this; "
        "stay consistent with it):\n" + "\n".join(facts)
    )


def upgrade(rows: list[dict], traces_by_id: dict[str, Trace]) -> list[dict]:
    """v1 scenario rows -> v2 rows with seed_state + rubric joined from the corpus."""
    out = []
    for row in rows:
        trace = traces_by_id.get(row["provenance"][0])
        if trace is None:
            raise SystemExit(f"pinned trace id missing from corpus: {row['provenance'][0]}")
        out.append(row | {"seed_state": {"structured": {}, "scratchpad": seed_scratchpad(trace)}})
    return out


def main() -> None:
    traces = get_adapter("otel-genai").from_file(str(_TRACES_PATH))
    by_id = {t.trace_id: t for t in traces}
    for name in ("scenarios_train", "scenarios_eval"):
        rows = [
            json.loads(line)
            for line in (_DIR / f"{name}.jsonl").read_text().splitlines()
            if line.strip()
        ]
        upgraded = upgrade(rows, by_id)
        out_path = _DIR / f"{name}_v2.jsonl"
        with out_path.open("w") as f:
            for row in upgraded:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(  # noqa: T201 - CLI output
            f"wrote {out_path.name}: {len(upgraded)} scenarios (seed_state attached)"
        )


if __name__ == "__main__":
    main()
