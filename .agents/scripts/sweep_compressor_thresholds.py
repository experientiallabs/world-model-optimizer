"""Round 2 rung A: per-corpus absolute-threshold sweep through the CANONICAL scorer.

Imports the endpoint's own LLMLingua2FixedThreshold (deploy/compressor-endpoint/server.py,
the implementation ruled canonical after round 1's truncation bug) and calls its compress()
at every threshold, so there is no reconstruction step that could diverge from serving.
Runs on box-6 GPU1; GPU0 belongs to the production wmo-compressor service and is never
touched.

Per (corpus, threshold): achieved keep ratio (canonical subword counts), proxy retention
on kept words (numeric-bearing, identifier-like, entity-like), split fit/holdout by
episode (seed 0). Plus, at candidate thresholds: determinism (same call twice), and
append/batch stability (full-batch vs per-episode vs per-segment calls byte-identical,
which is exactly why fp32 is the serving rule).

Usage (box):
    ./venv/bin/python sweep_compressor_thresholds.py --device cuda:1 \
        --segments 'live-segments-*.jsonl' --out sweep-results.json
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import logging
import random
import re
import sys
import time
from pathlib import Path

log = logging.getLogger("sweep")

HERE = Path(__file__).resolve().parent

NUMERIC_RE = re.compile(r"\d")
IDENTIFIER_RE = re.compile(r"[/_\-\.=]|^--|^-[a-zA-Z]$|[a-z][A-Z]")
ENTITY_RE = re.compile(r"^[A-Z][a-zA-Z]+")
THRESHOLDS = [round(0.20 + 0.02 * i, 2) for i in range(31)]
SPLIT_SEED = 0


SERVER_CANDIDATES = (
    HERE.parents[1] / "deploy/compressor-endpoint/server.py",  # in-repo canonical copy
    HERE / "server.py",  # box-side manual copy (recorded via sha256 either way)
)


def _load_server_module():  # noqa: ANN202
    """Load the canonical scorer, preferring the in-repo path, and record its sha256.

    Gate finding M5: the canonicality guarantee must not rest on an unrecorded manual
    copy. Whichever file loads, its digest lands in the results JSON next to the model
    fingerprint, so a reviewer can diff it against deploy/compressor-endpoint/server.py
    at the pinned commit.
    """
    import hashlib

    path = next((p for p in SERVER_CANDIDATES if p.is_file()), None)
    if path is None:
        raise SystemExit(f"canonical server.py not found at any of: {SERVER_CANDIDATES}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("compressor_server", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compressor_server"] = mod
    spec.loader.exec_module(mod)
    mod.LOADED_SERVER_PATH = str(path)
    mod.LOADED_SERVER_SHA256 = digest
    return mod


def word_classes(words: list[str]) -> dict[str, int]:
    counts = {"numeric": 0, "identifier": 0, "entity": 0, "all": len(words)}
    for w in words:
        w = w.strip()
        if NUMERIC_RE.search(w):
            counts["numeric"] += 1
        if IDENTIFIER_RE.search(w):
            counts["identifier"] += 1
        if ENTITY_RE.match(w):
            counts["entity"] += 1
    return counts


def retention(server, raw: list[str], compressed: list[str]) -> dict[str, float]:  # noqa: ANN001
    raw_counts = {"numeric": 0, "identifier": 0, "entity": 0, "all": 0}
    kept_counts = {"numeric": 0, "identifier": 0, "entity": 0, "all": 0}
    for r, c in zip(raw, compressed):
        for target, text in ((raw_counts, r), (kept_counts, c)):
            for k, v in word_classes(server.WORD_RE.findall(text)).items():
                target[k] += v
    return {
        k: (kept_counts[k] / raw_counts[k]) if raw_counts[k] else 1.0
        for k in raw_counts
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--segments", default=str(HERE / "live-segments-*.jsonl"))
    ap.add_argument("--out", default=str(HERE / "sweep-results.json"))
    ap.add_argument("--stability-thresholds", default="0.5")
    args = ap.parse_args()

    server = _load_server_module()
    comp = server.LLMLingua2FixedThreshold(server.DEFAULT_MODEL_ID, args.device)
    log.info("model fingerprint: %s  version: %s", comp.model_fingerprint, comp.version)

    results: dict = {
        "fingerprint": comp.model_fingerprint,
        "version": comp.version,
        "server_path": server.LOADED_SERVER_PATH,
        "server_sha256": server.LOADED_SERVER_SHA256,
        "rows": [],
    }
    for path in sorted(glob.glob(args.segments)):
        corpus = Path(path).stem.replace("live-segments-", "")
        episodes = [json.loads(ln) for ln in open(path)]
        rng = random.Random(SPLIT_SEED)
        rng.shuffle(episodes)
        half = len(episodes) // 2
        parts = {"fit": episodes[:half], "holdout": episodes[half:]}
        log.info("%s: %d fit / %d holdout episodes", corpus, half, len(episodes) - half)
        for part_name, part in parts.items():
            segments = [s for ep in part for s in ep["segments"]]
            for threshold in THRESHOLDS:
                t0 = time.perf_counter()
                outcome = comp.compress(segments, threshold)
                keep = outcome.tokens_out / outcome.tokens_in if outcome.tokens_in else 1.0
                ret = retention(server, segments, outcome.segments)
                results["rows"].append(
                    {
                        "corpus": corpus,
                        "part": part_name,
                        "threshold": threshold,
                        "keep_subword": round(keep, 4),
                        "retention": {k: round(v, 4) for k, v in ret.items()},
                        "n_segments": len(segments),
                        "compute_s": round(time.perf_counter() - t0, 2),
                    }
                )
            log.info("%s/%s: swept %d thresholds", corpus, part_name, len(THRESHOLDS))

        # Stability checks at candidate thresholds, on 12 holdout episodes.
        sample = parts["holdout"][:12]
        for threshold in [float(x) for x in args.stability_thresholds.split(",")]:
            flat = [s for ep in sample for s in ep["segments"]]
            full1 = comp.compress(flat, threshold).segments
            full2 = comp.compress(flat, threshold).segments
            per_episode: list[str] = []
            for ep in sample:
                per_episode.extend(comp.compress(ep["segments"], threshold).segments)
            per_segment: list[str] = []
            for s in flat:
                per_segment.extend(comp.compress([s], threshold).segments)
            results["rows"].append(
                {
                    "corpus": corpus,
                    "part": "stability",
                    "threshold": threshold,
                    "deterministic": full1 == full2,
                    "batch_invariant_episode": full1 == per_episode,
                    "batch_invariant_segment": full1 == per_segment,
                    "n_segments": len(flat),
                }
            )
            log.info(
                "%s stability @%.2f: det=%s ep-invariant=%s seg-invariant=%s",
                corpus,
                threshold,
                full1 == full2,
                full1 == per_episode,
                full1 == per_segment,
            )

    Path(args.out).write_text(json.dumps(results, indent=2))
    log.info("wrote %s (%d rows)", args.out, len(results["rows"]))


if __name__ == "__main__":
    main()
