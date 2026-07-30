"""R3 matching protocol: bisect every scorer's threshold on CAPTURED LIVE segments.

Closes the overnight 2.8pt keep caveat: thresholds are matched on the exact text the
serving path sees (raw user segments captured from the scaled off arm), in canonical
subword coordinates, to stock@0.5's live keep. Also runs the stability audit per
student at its bisected threshold.

    ./venv/bin/python box_bisect_live.py --device cuda:0
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import random
import sys
from pathlib import Path

log = logging.getLogger("bisect_live")
HERE = Path(__file__).resolve().parent

STUDENTS = [
    "adapted-financebench-full-aggressive",
    "adapted-financebench-full-aggressive-keep40",
    "adapted-financebench-full-aggressive-keep55",
    "adapted-financebench-full-aggressive-keep70",
    "adapted-financebench-full-composition",
]


def load_server():  # noqa: ANN201
    path = HERE / "server.py"
    spec = importlib.util.spec_from_file_location("compressor_server", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compressor_server"] = mod
    spec.loader.exec_module(mod)
    mod.LOADED_SERVER_SHA256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return mod


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--segments", default=str(HERE / "truth-s80-off_segments.jsonl"))
    ap.add_argument("--sample-episodes", type=int, default=120)
    args = ap.parse_args()
    server = load_server()

    episodes = [json.loads(ln) for ln in open(args.segments)]
    rng = random.Random(0)
    rng.shuffle(episodes)
    segments = [s for ep in episodes[: args.sample_episodes] for s in ep["segments"]]
    log.info("live segments: %d from %d episodes", len(segments), min(len(episodes), args.sample_episodes))

    stock = server.LLMLingua2FixedThreshold(server.DEFAULT_MODEL_ID, args.device)
    o = stock.compress(segments, 0.5)
    target = o.tokens_out / o.tokens_in
    log.info("stock@0.5 LIVE keep = %.4f (the match target)", target)
    results = {
        "target_keep_live": round(target, 4),
        "server_sha256": server.LOADED_SERVER_SHA256,
        "n_segments": len(segments),
        "scorers": {},
    }
    del stock
    import torch

    torch.cuda.empty_cache()
    for name in STUDENTS:
        comp = server.LLMLingua2FixedThreshold(str(HERE / name), args.device)
        lo, hi, mid, keep = 0.0, 1.0, 0.5, 1.0
        for _ in range(22):
            mid = (lo + hi) / 2
            oo = comp.compress(segments, mid)
            keep = oo.tokens_out / oo.tokens_in
            if abs(keep - target) <= 0.003:
                break
            if keep > target:
                lo = mid
            else:
                hi = mid
        sample = segments[:400]
        a = comp.compress(sample, mid).segments
        b = comp.compress(sample, mid).segments
        per_one: list[str] = []
        for s in sample[:120]:
            per_one.extend(comp.compress([s], mid).segments)
        results["scorers"][name] = {
            "matched_threshold": round(mid, 4),
            "achieved_live_keep": round(keep, 4),
            "deterministic": a == b,
            "batch_invariant": a[:120] == per_one,
            "model_fingerprint": comp.model_fingerprint,
        }
        log.info("%s: %s", name, results["scorers"][name])
        del comp
        torch.cuda.empty_cache()
    (HERE / "live-bisect.json").write_text(json.dumps(results, indent=2))
    log.info("BISECT-DONE")


if __name__ == "__main__":
    main()
