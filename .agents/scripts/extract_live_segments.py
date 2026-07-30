"""Extract live-structure segments for the round 2 threshold calibration (track C1).

Live structure = what _CompressingProvider actually compresses per episode: the task plus
every tool observation (user-role contents), taken from the bundle traces. The round 1
replies-only proxy under-predicted live compression (keep 0.799 proxy vs 0.652 live on
financebench) because tool observations dominate live prompts; this extractor exists so
every round 2 calibration runs on the real segment mix.

Usage: uv run python .agents/scripts/extract_live_segments.py
Writes ~/Desktop/Projects/wmh-compression-data/cache/live-segments-<corpus>.jsonl
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from wmo.ingest import get_adapter

log = logging.getLogger("extract_live_segments")

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path.home() / "Desktop/Projects/wmh-compression-data/cache"
CORPORA = ["financebench", "tau-bench", "terminal-tasks", "swe-bench"]
MAX_EPISODES = 60
MAX_SEGMENTS = 24
SEED = 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for corpus in CORPORA:
        traces = get_adapter("otel-genai").from_file(
            str(REPO / f"packages/environment-capture/{corpus}/traces.otel.jsonl")
        )
        rng = random.Random(SEED)
        picked = traces if len(traces) <= MAX_EPISODES else rng.sample(traces, MAX_EPISODES)
        rows = []
        for trace in picked:
            task = next((s.task for s in trace.steps if s.task), "") or ""
            observations = [
                s.observation.content
                for s in trace.steps
                if s.observation is not None and s.observation.content
            ]
            segments = ([task] if task else []) + observations
            segments = [str(s) for s in segments[:MAX_SEGMENTS] if str(s).strip()]
            if len(segments) >= 2:
                rows.append({"corpus": corpus, "trace_id": trace.trace_id, "segments": segments})
        out = OUT_DIR / f"live-segments-{corpus}.jsonl"
        with out.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_seg = sum(len(r["segments"]) for r in rows)
        n_chars = sum(len(s) for r in rows for s in r["segments"])
        log.info(
            "%s: %d episodes -> %d segments, %.1f MB -> %s",
            corpus,
            len(rows),
            n_seg,
            n_chars / 1e6,
            out.name,
        )


if __name__ == "__main__":
    main()
