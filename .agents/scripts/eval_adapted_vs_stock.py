"""Rung B eval: adapted vs stock scorer at MATCHED ACHIEVED KEEP on held-out windows.

The registered deliverable: never compare at matched threshold (thresholds mean
different things to different scorers); find each scorer's threshold that achieves the
same holdout keep ratio, then compare what they keep. Metrics at matched keep:
teacher-agreement (recall of teacher-kept words, drop-agreement on teacher-dropped
words) and the class-retention proxies (numeric / identifier / entity words).

Runs on the box next to the trained checkpoints:
    ./venv/bin/python eval_adapted_vs_stock.py --corpus terminal-tasks --device cuda:1
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path

log = logging.getLogger("eval_adapted")

HERE = Path(__file__).resolve().parent
MODEL_ID = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
WORD_RE = re.compile(r"\S+\s*")
NUMERIC_RE = re.compile(r"\d")
IDENTIFIER_RE = re.compile(r"[/_\-\.=]|^--|^-[a-zA-Z]$|[a-z][A-Z]")
ENTITY_RE = re.compile(r"^[A-Z][a-zA-Z]+")
SEED = 0
BATCH = 16


def split_by_trace(examples: list[dict]) -> tuple[list[dict], list[dict]]:
    traces = sorted({e["trace_id"] for e in examples})
    rng = random.Random(SEED)
    rng.shuffle(traces)
    holdout = set(traces[: max(1, len(traces) // 5)])
    return (
        [e for e in examples if e["trace_id"] not in holdout],
        [e for e in examples if e["trace_id"] in holdout],
    )


def word_probs(model, tokenizer, torch, device: str, windows: list[list[str]]) -> list[list[float]]:  # noqa: ANN001
    """Per-word keep probabilities, unscored words default to keep=1.0 (canonical rule)."""
    out: list[list[float]] = []
    for start in range(0, len(windows), BATCH):
        batch = windows[start : start + BATCH]
        enc = tokenizer(
            batch,
            is_split_into_words=True,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        keep = torch.softmax(logits.float(), dim=-1)[:, :, 1].cpu()
        for row, words in enumerate(batch):
            sums = [0.0] * len(words)
            counts = [0] * len(words)
            for pos, wid in enumerate(enc.word_ids(row)):
                if wid is None or wid >= len(words):
                    continue
                sums[wid] += float(keep[row, pos])
                counts[wid] += 1
            out.append([s / c if c else 1.0 for s, c in zip(sums, counts)])
    return out


def threshold_for_keep(probs: list[list[float]], target_keep: float) -> float:
    flat = sorted(p for ps in probs for p in ps)
    cut = int(len(flat) * (1 - target_keep))
    return flat[min(cut, len(flat) - 1)]


def metrics_at(probs: list[list[float]], threshold: float, windows: list[list[str]], labels: list[list[int]]) -> dict:
    kept_gold_kept = gold_kept = kept = total = 0
    dropped_gold_dropped = dropped = 0
    cls_raw = {"numeric": 0, "identifier": 0, "entity": 0}
    cls_kept = {"numeric": 0, "identifier": 0, "entity": 0}
    for ps, ws, ls in zip(probs, windows, labels):
        for p, w, gold in zip(ps, ws, ls):
            total += 1
            keep = p >= threshold
            if keep:
                kept += 1
            else:
                dropped += 1
                if gold == 0:
                    dropped_gold_dropped += 1
            if gold == 1:
                gold_kept += 1
                if keep:
                    kept_gold_kept += 1
            for cls, rx in (("numeric", NUMERIC_RE), ("identifier", IDENTIFIER_RE), ("entity", ENTITY_RE)):
                if (rx.search(w) if cls != "entity" else rx.match(w)):
                    cls_raw[cls] += 1
                    if keep:
                        cls_kept[cls] += 1
    return {
        "achieved_keep": round(kept / max(1, total), 4),
        "teacher_keep_recall": round(kept_gold_kept / max(1, gold_kept), 4),
        "teacher_drop_agreement": round(dropped_gold_dropped / max(1, dropped), 4),
        "retention": {k: round(cls_kept[k] / max(1, cls_raw[k]), 4) for k in cls_raw},
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--target-keep", type=float, default=0.65)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    path = HERE / f"labels-{args.corpus}-aggressive.jsonl"
    examples = [json.loads(ln) for ln in path.open()]
    _, holdout = split_by_trace(examples)
    windows = [[w.strip() for w in WORD_RE.findall(e["segment"])][:400] for e in holdout]
    labels = [e["labels"][:400] for e in holdout]
    log.info("%s: %d holdout windows", args.corpus, len(windows))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    results = {"corpus": args.corpus, "target_keep": args.target_keep, "n_holdout": len(windows), "scorers": {}}
    arms = {"stock": MODEL_ID}
    for arm in ("lora", "full"):
        ckpt = HERE / f"adapted-{args.corpus}-{arm}"
        if ckpt.exists():
            arms[arm] = str(ckpt)
    for name, source in arms.items():
        if name == "lora":
            from peft import PeftModel

            base = AutoModelForTokenClassification.from_pretrained(MODEL_ID, dtype=torch.float32)
            model = PeftModel.from_pretrained(base, source).merge_and_unload()
        else:
            model = AutoModelForTokenClassification.from_pretrained(source, dtype=torch.float32)
        model = model.to(device=args.device, dtype=torch.float32).eval()
        probs = word_probs(model, tokenizer, torch, args.device, windows)
        threshold = threshold_for_keep(probs, args.target_keep)
        m = metrics_at(probs, threshold, windows, labels)
        m["threshold_at_matched_keep"] = round(threshold, 4)
        results["scorers"][name] = m
        log.info("%s @keep=%.2f: %s", name, args.target_keep, m)
        del model
        torch.cuda.empty_cache()
    out = args.out or str(HERE / f"adapted-eval-{args.corpus}.json")
    Path(out).write_text(json.dumps(results, indent=2))
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
