"""Box-side gate-fix batch: fin match target, canonical evals, stability at calibrated.

Runs on box-6 GPU1 (or any idle GPU): (1) stock@0.5 keep on the financebench holdout
windows + adapted threshold bisected to match (the mini-leg's ratio-matching input);
(2) canonical matched-keep evals per corpus (gate HIGH); (3) stability at the four
calibrated thresholds via the sweep script is launched separately.

    ./venv/bin/python box_fix_batch.py --device cuda:1
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import random
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("box_fix_batch")
HERE = Path(__file__).resolve().parent


def load_server():  # noqa: ANN201
    path = HERE / "server.py"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("compressor_server", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compressor_server"] = mod
    spec.loader.exec_module(mod)
    mod.LOADED_SERVER_SHA256 = digest
    return mod


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    server = load_server()

    # (1) fin match target for the mini-leg.
    examples = [json.loads(ln) for ln in (HERE / "labels-financebench-aggressive.jsonl").open()]
    traces = sorted({e["trace_id"] for e in examples})
    random.Random(0).shuffle(traces)
    hold = set(traces[: max(1, len(traces) // 5)])
    windows = [e["segment"] for e in examples if e["trace_id"] in hold]
    stock = server.LLMLingua2FixedThreshold(server.DEFAULT_MODEL_ID, args.device)
    outcome = stock.compress(windows, 0.5)
    stock_keep = outcome.tokens_out / outcome.tokens_in
    adapted = server.LLMLingua2FixedThreshold(str(HERE / "adapted-financebench-full"), args.device)
    lo, hi, mid, kk = 0.0, 1.0, 0.5, 1.0
    for _ in range(20):
        mid = (lo + hi) / 2
        oo = adapted.compress(windows, mid)
        kk = oo.tokens_out / oo.tokens_in
        if abs(kk - stock_keep) <= 0.004:
            break
        if kk > stock_keep:
            lo = mid
        else:
            hi = mid
    result = {
        "stock_keep_at_0.5": round(stock_keep, 4),
        "adapted_matched_threshold": round(mid, 4),
        "adapted_keep": round(kk, 4),
        "server_sha256": server.LOADED_SERVER_SHA256,
    }
    (HERE / "fin-match.json").write_text(json.dumps(result, indent=2))
    log.info("fin-match: %s", result)
    del stock, adapted
    import torch

    torch.cuda.empty_cache()

    # (2) canonical evals per corpus (gate HIGH).
    for corpus in ("financebench", "swe-bench", "tau-bench", "terminal-tasks"):
        ckpt = HERE / f"adapted-{corpus}-full"
        if not ckpt.exists():
            log.warning("%s: no checkpoint yet, skipping", corpus)
            continue
        subprocess.run(
            [
                sys.executable,
                str(HERE / "eval_adapted_canonical.py"),
                "--corpus",
                corpus,
                "--device",
                args.device,
                "--checkpoints",
                f"full={ckpt}",
            ],
            check=True,
        )
    log.info("BOX-FIX-BATCH-DONE")


if __name__ == "__main__":
    main()
