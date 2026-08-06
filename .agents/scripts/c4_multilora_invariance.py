"""Run the C4 multi-adapter invariance suite (kill bars KB1-KB4) on real segments.

Payload: live user-segments captured by C1 (task + tool observations, the text the
production compressor actually sees), a sample per corpus, plus the server's own
self-test segments. Adapters: C1's four per-corpus LoRA r16 checkpoints.

Writes cache/c4-invariance-<device>.json in the compression data root and appends a
RunRecord-shaped row to runs/c4.jsonl.

Usage:
    python c4_multilora_invariance.py --device cpu --segments-per-corpus 3 --max-chars 2000
    python c4_multilora_invariance.py --device cuda:1
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import tempfile
from pathlib import Path

from c4_multilora import BASE_ADAPTER, SERVER, MultiAdapterCompressor, run_invariance_suite

log = logging.getLogger("c4_invariance")

DATA_ROOT = Path(
    os.environ.get("C4_DATA_ROOT", "~/Desktop/Projects/wmh-compression-data")
).expanduser()
ADAPTER_ROOT = DATA_ROOT / "cache/c4-adapters"
CORPORA = ("financebench", "swe-bench", "tau-bench", "terminal-tasks")


def load_segments(per_corpus: int, max_chars: int) -> list[str]:
    """A deterministic sample of live segments across the four corpora."""
    segments: list[str] = []
    for corpus in CORPORA:
        path = DATA_ROOT / f"cache/live-segments-{corpus}.jsonl"
        taken = 0
        for line in path.open():
            if taken >= per_corpus:
                break
            record = json.loads(line)
            for segment in record.get("segments", []):
                if taken >= per_corpus:
                    break
                if not segment.strip():
                    continue
                segments.append(segment[:max_chars])
                taken += 1
    return segments


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--segments-per-corpus", type=int, default=6)
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--skip-merged", action="store_true")
    parser.add_argument(
        "--merged-thresholds",
        default=None,
        help="comma-separated threshold sweep for the merged-vs-unmerged check "
        "(default: 0.05..0.95 step 0.05)",
    )
    args = parser.parse_args()
    merged_thresholds = (
        [float(part) for part in args.merged_thresholds.split(",")]
        if args.merged_thresholds
        else None
    )

    adapter_dirs = {
        corpus: ADAPTER_ROOT / f"adapted-{corpus}-lora"
        for corpus in CORPORA
        if (ADAPTER_ROOT / f"adapted-{corpus}-lora").is_dir()
    }
    log.info("adapters: %s", sorted(adapter_dirs))
    segments = load_segments(args.segments_per_corpus, args.max_chars)
    log.info(
        "payload: %d segments (server sha %s)", len(segments), SERVER.LOADED_SERVER_SHA256[:12]
    )

    compressor = MultiAdapterCompressor(SERVER.DEFAULT_MODEL_ID, args.device, adapter_dirs)
    log.info("base %s adapters %s", compressor.base_fingerprint, compressor.adapter_fingerprints)
    with tempfile.TemporaryDirectory(prefix="c4-merged-") as workdir:
        verdicts = run_invariance_suite(
            compressor,
            segments,
            merged_thresholds=merged_thresholds,
            merged_workdir=None if args.skip_merged else Path(workdir),
        )

    verdicts["owner"] = "c4"
    verdicts["segments_per_corpus"] = args.segments_per_corpus
    verdicts["max_chars"] = args.max_chars
    device_slug = args.device.replace(":", "")
    out_path = DATA_ROOT / f"cache/c4-invariance-{device_slug}.json"
    out_path.write_text(json.dumps(verdicts, indent=2))
    log.info("verdicts -> %s", out_path)

    booleans = {
        key: value
        for key, value in verdicts.items()
        if key.startswith("kb") and isinstance(value, (bool, str))
    }
    log.info("verdict summary: %s", booleans)

    run_row = {
        "run_id": f"c4-invariance-{device_slug}",
        "owner": "c4",
        "ts": datetime.datetime.now(tz=datetime.UTC).isoformat(),
        "matrix": "c4-multilora-invariance",
        "variant": args.device,
        "params": {
            "segments_per_corpus": args.segments_per_corpus,
            "max_chars": args.max_chars,
            "adapters": sorted(adapter_dirs),
            "base_adapter_name": BASE_ADAPTER,
            "server_sha256": SERVER.LOADED_SERVER_SHA256,
        },
        "result": booleans,
        "notes": f"full verdicts in {out_path.name}",
    }
    runs_path = DATA_ROOT / "runs/c4.jsonl"
    with runs_path.open("a") as handle:
        handle.write(json.dumps(run_row) + "\n")
    log.info("run row -> %s", runs_path)


if __name__ == "__main__":
    main()
