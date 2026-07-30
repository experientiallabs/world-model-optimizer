"""Gate-HIGH fix: adapted-vs-stock eval THROUGH the canonical scorer, serving coordinates.

The round 2 evaluator reimplemented word scoring with the round-1 bug's chunking (fixed
400-word windows, 512 truncation, unscored words force-kept), so its matched-keep was
matched in harness coordinates. This evaluator has no scoring path of its own: every
number comes out of the canonical LLMLingua2FixedThreshold.compress() (loaded from
deploy/compressor-endpoint/server.py, sha256 recorded), with keep measured in the
canonical subword counts (tokens_out / tokens_in) and predicted keep/drop labels
recovered by subsequence alignment of the canonical output against the raw window.

Matched keep: per scorer, bisect the threshold until holdout keep hits the target
within tolerance; compare scorers at their own matched thresholds, never at a shared
one. LoRA checkpoints must be pre-merged (merge_lora.py) so from_pretrained loads them.

Usage (GPU box):
    ./venv/bin/python eval_adapted_canonical.py --corpus swe-bench --device cuda:1
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import random
import re
import sys
from pathlib import Path

log = logging.getLogger("eval_canonical")

HERE = Path(__file__).resolve().parent
STOCK_MODEL_ID = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
WORD_RE = re.compile(r"\S+\s*")
NUMERIC_RE = re.compile(r"\d")
IDENTIFIER_RE = re.compile(r"[/_\-\.=]|^--|^-[a-zA-Z]$|[a-z][A-Z]")
ENTITY_RE = re.compile(r"^[A-Z][a-zA-Z]+")
SEED = 0

SERVER_CANDIDATES = (
    HERE.parents[1] / "deploy/compressor-endpoint/server.py",
    HERE / "server.py",
)


def load_server():  # noqa: ANN201
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


def split_by_trace(examples: list[dict]) -> tuple[list[dict], list[dict]]:
    traces = sorted({e["trace_id"] for e in examples})
    rng = random.Random(SEED)
    rng.shuffle(traces)
    holdout = set(traces[: max(1, len(traces) // 5)])
    return (
        [e for e in examples if e["trace_id"] not in holdout],
        [e for e in examples if e["trace_id"] in holdout],
    )


def align_pred(raw: str, compressed: str) -> list[int] | None:
    """Greedy subsequence alignment of the canonical output back to raw words."""
    raw_words = [w.strip() for w in WORD_RE.findall(raw)]
    out_words = [w.strip() for w in WORD_RE.findall(compressed)]
    labels = [0] * len(raw_words)
    i = 0
    for w in out_words:
        while i < len(raw_words) and raw_words[i] != w:
            i += 1
        if i == len(raw_words):
            return None
        labels[i] = 1
        i += 1
    return labels


def holdout_keep(comp, windows: list[str], threshold: float) -> float:  # noqa: ANN001
    outcome = comp.compress(windows, threshold)
    return outcome.tokens_out / outcome.tokens_in if outcome.tokens_in else 1.0


def match_threshold(comp, windows: list[str], target: float, tol: float = 0.005) -> float:  # noqa: ANN001
    lo, hi = 0.0, 1.0
    for _ in range(20):
        mid = (lo + hi) / 2
        keep = holdout_keep(comp, windows, mid)
        if abs(keep - target) <= tol:
            return mid
        if keep > target:
            lo = mid  # keeping too much -> raise the bar
        else:
            hi = mid
    return (lo + hi) / 2


def metrics(comp, windows: list[str], gold: list[list[int]], threshold: float) -> dict:  # noqa: ANN001
    outcome = comp.compress(windows, threshold)
    kept_gold_kept = gold_kept = dropped = dropped_gold_dropped = kept = total = 0
    cls_raw = {"numeric": 0, "identifier": 0, "entity": 0}
    cls_kept = {"numeric": 0, "identifier": 0, "entity": 0}
    n_align_failures = 0
    for raw, out, labels in zip(windows, outcome.segments, gold):
        pred = align_pred(raw, out)
        if pred is None:
            n_align_failures += 1
            continue
        words = [w.strip() for w in WORD_RE.findall(raw)]
        for w, p, g in zip(words, pred, labels):
            total += 1
            if p:
                kept += 1
            else:
                dropped += 1
                if g == 0:
                    dropped_gold_dropped += 1
            if g == 1:
                gold_kept += 1
                if p:
                    kept_gold_kept += 1
            for cls, rx in (
                ("numeric", NUMERIC_RE),
                ("identifier", IDENTIFIER_RE),
                ("entity", ENTITY_RE),
            ):
                if rx.match(w) if cls == "entity" else rx.search(w):
                    cls_raw[cls] += 1
                    if p:
                        cls_kept[cls] += 1
    return {
        "achieved_keep_subword": round(
            outcome.tokens_out / outcome.tokens_in if outcome.tokens_in else 1.0, 4
        ),
        "achieved_keep_word": round(kept / max(1, total), 4),
        "teacher_keep_recall": round(kept_gold_kept / max(1, gold_kept), 4),
        "teacher_drop_agreement": round(dropped_gold_dropped / max(1, dropped), 4),
        "retention": {k: round(cls_kept[k] / max(1, cls_raw[k]), 4) for k in cls_raw},
        "align_failures": n_align_failures,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--target-keep", type=float, default=0.65)
    ap.add_argument("--labels-suffix", default="aggressive")
    ap.add_argument("--checkpoints", default=None, help="comma name=path list; default stock+full+lora-merged")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    server = load_server()
    examples = [
        json.loads(ln)
        for ln in (HERE / f"labels-{args.corpus}-{args.labels_suffix}.jsonl").open()
    ]
    _, holdout = split_by_trace(examples)
    windows = [e["segment"] for e in holdout]
    gold = [e["labels"] for e in holdout]
    log.info("%s: %d holdout windows", args.corpus, len(windows))

    sources = {"stock": STOCK_MODEL_ID}
    if args.checkpoints:
        for pair in args.checkpoints.split(","):
            name, path = pair.split("=", 1)
            sources[name] = path
    else:
        for arm in ("full", "lora-merged"):
            ckpt = HERE / f"adapted-{args.corpus}-{arm}"
            if ckpt.exists():
                sources[arm] = str(ckpt)

    results = {
        "corpus": args.corpus,
        "target_keep": args.target_keep,
        "coordinates": "canonical-subword (server.compress)",
        "server_path": server.LOADED_SERVER_PATH,
        "server_sha256": server.LOADED_SERVER_SHA256,
        "labels_suffix": args.labels_suffix,
        "n_holdout_windows": len(windows),
        "scorers": {},
    }
    for name, source in sources.items():
        comp = server.LLMLingua2FixedThreshold(source, args.device)
        threshold = match_threshold(comp, windows, args.target_keep)
        m = metrics(comp, windows, gold, threshold)
        m["matched_threshold"] = round(threshold, 4)
        m["model_fingerprint"] = comp.model_fingerprint
        results["scorers"][name] = m
        log.info("%s: %s", name, m)
        del comp
        import torch

        torch.cuda.empty_cache()
    out = args.out or str(HERE / f"canonical-eval-{args.corpus}-{args.labels_suffix}.json")
    Path(out).write_text(json.dumps(results, indent=2))
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
