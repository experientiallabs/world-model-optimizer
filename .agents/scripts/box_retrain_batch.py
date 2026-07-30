"""OVN-C box batch: retrain on opus-5 labels, canonical 3-way eval, per-checkpoint stability.

Per corpus: (1) train full-FT on the opus-5 label set; (2) canonical matched-keep eval
of stock vs gpt5.5-full vs opus5-full, each against ITS OWN teacher's holdout labels
AND the other teacher's (cross-eval: a scorer that only agrees with its own teacher has
learned an opinion, not the domain); (3) determinism + batch-invariance stability on
every adapted checkpoint at its matched threshold (the extreme-threshold risk found in
the canonical re-eval).

    ./venv/bin/python box_retrain_batch.py --device cuda:1
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

log = logging.getLogger("box_retrain")
HERE = Path(__file__).resolve().parent
CORPORA = ("financebench", "swe-bench", "tau-bench", "terminal-tasks")


def load_server():  # noqa: ANN201
    path = HERE / "server.py"
    spec = importlib.util.spec_from_file_location("compressor_server", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compressor_server"] = mod
    spec.loader.exec_module(mod)
    mod.LOADED_SERVER_SHA256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return mod


def holdout_windows(corpus: str, suffix: str) -> list[str]:
    examples = [json.loads(ln) for ln in (HERE / f"labels-{corpus}-{suffix}.jsonl").open()]
    traces = sorted({e["trace_id"] for e in examples})
    random.Random(0).shuffle(traces)
    hold = set(traces[: max(1, len(traces) // 5)])
    return [e["segment"] for e in examples if e["trace_id"] in hold]


def match_threshold(comp, windows: list[str], target: float = 0.65) -> float:  # noqa: ANN001
    lo, hi, mid = 0.0, 1.0, 0.5
    for _ in range(20):
        mid = (lo + hi) / 2
        o = comp.compress(windows, mid)
        keep = o.tokens_out / o.tokens_in
        if abs(keep - target) <= 0.005:
            break
        if keep > target:
            lo = mid
        else:
            hi = mid
    return mid


def stability(comp, windows: list[str], threshold: float) -> dict:  # noqa: ANN001
    a = comp.compress(windows, threshold).segments
    b = comp.compress(windows, threshold).segments
    per_one: list[str] = []
    for w in windows:
        per_one.extend(comp.compress([w], threshold).segments)
    return {
        "deterministic": a == b,
        "batch_invariant": a == per_one,
        "threshold": round(threshold, 4),
        "n_windows": len(windows),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    server = load_server()
    report: dict = {"server_sha256": server.LOADED_SERVER_SHA256, "stability": {}, "cross_eval": {}}

    for corpus in CORPORA:
        # (1) train on opus labels.
        subprocess.run(
            [
                sys.executable,
                str(HERE / "train_adapted_scorer.py"),
                "--corpus",
                corpus,
                "--arm",
                "full",
                "--device",
                args.device,
                "--labels-suffix",
                "aggressive-opus-5",
            ],
            check=True,
        )
        # (2) canonical eval of all three scorers against EACH teacher's holdout.
        for label_suffix in ("aggressive", "aggressive-opus-5"):
            subprocess.run(
                [
                    sys.executable,
                    str(HERE / "eval_adapted_canonical.py"),
                    "--corpus",
                    corpus,
                    "--device",
                    args.device,
                    "--labels-suffix",
                    label_suffix,
                    "--checkpoints",
                    (
                        f"gpt55-full={HERE / f'adapted-{corpus}-full'},"
                        f"opus5-full={HERE / f'adapted-{corpus}-full-aggressive-opus-5'}"
                    ),
                    "--out",
                    str(HERE / f"canonical-eval3-{corpus}-{label_suffix}.json"),
                ],
                check=True,
            )
        # (3) stability per adapted checkpoint at its matched threshold.
        windows = holdout_windows(corpus, "aggressive")[:60]
        for name, path in (
            ("gpt55-full", HERE / f"adapted-{corpus}-full"),
            ("opus5-full", HERE / f"adapted-{corpus}-full-aggressive-opus-5"),
        ):
            comp = server.LLMLingua2FixedThreshold(str(path), args.device)
            thr = match_threshold(comp, windows)
            report["stability"][f"{corpus}/{name}"] = stability(comp, windows, thr)
            log.info("%s/%s stability: %s", corpus, name, report["stability"][f"{corpus}/{name}"])
            del comp
            import torch

            torch.cuda.empty_cache()
    (HERE / "retrain-batch-report.json").write_text(json.dumps(report, indent=2))
    log.info("RETRAIN-BATCH-DONE")


if __name__ == "__main__":
    main()
