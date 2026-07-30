"""Rung B training: adapt the 177M LLMLingua-2 token classifier to our corpora.

Trains on the self-distilled keep/drop labels (build_distill_labels.py), two arms per
corpus family: LoRA (r=16 on attention projections) and full fine-tune (177M is small
enough that both are cheap). fp32 end to end, per the batch-invariance serving rule.
Runs on box-6 GPU1 next to tenants: batch sizes are kept small and memory is capped.

Eval on held-out segments (episode-level split, seed 0): label agreement F1 vs the
teacher, plus the achieved-ratio curve so the adapted-vs-stock comparison happens at
MATCHED ACHIEVED RATIO (the registered deliverable), never at matched threshold.

Usage (box):
    ./venv/bin/python train_adapted_scorer.py --corpus financebench --arm lora
    ./venv/bin/python train_adapted_scorer.py --corpus financebench --arm full
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path

log = logging.getLogger("train_adapted")

HERE = Path(__file__).resolve().parent
MODEL_ID = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
WORD_RE = re.compile(r"\S+\s*")
SEED = 0
EPOCHS = 3
LR_LORA = 1e-4
LR_FULL = 2e-5
BATCH = 8
MAX_WORDS = 400  # window matching the server's chunk budget


def load_examples(corpus: str, suffix: str = "aggressive") -> list[dict]:
    path = HERE / f"labels-{corpus}-{suffix}.jsonl"
    return [json.loads(ln) for ln in path.open()]


def split_by_trace(examples: list[dict]) -> tuple[list[dict], list[dict]]:
    traces = sorted({e["trace_id"] for e in examples})
    rng = random.Random(SEED)
    rng.shuffle(traces)
    holdout = set(traces[: max(1, len(traces) // 5)])
    return (
        [e for e in examples if e["trace_id"] not in holdout],
        [e for e in examples if e["trace_id"] in holdout],
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--arm", choices=["lora", "full"], required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--labels-suffix", default="aggressive")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_ID, dtype=torch.float32)
    if args.arm == "lora":
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["query", "value"],
                task_type="TOKEN_CLS",
            ),
        )
    model = model.to(device=args.device, dtype=torch.float32).train()

    examples = load_examples(args.corpus, args.labels_suffix)
    fit, holdout = split_by_trace(examples)
    log.info("%s/%s: %d fit / %d holdout segments", args.corpus, args.arm, len(fit), len(holdout))

    def encode(batch: list[dict]):  # noqa: ANN202
        words = [[w.strip() for w in WORD_RE.findall(e["segment"])][:MAX_WORDS] for e in batch]
        labels = [e["labels"][:MAX_WORDS] for e in batch]
        enc = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        )
        target = torch.full(enc["input_ids"].shape, -100, dtype=torch.long)
        for row in range(len(batch)):
            for pos, wid in enumerate(enc.word_ids(row)):
                if wid is not None and wid < len(labels[row]):
                    target[row, pos] = labels[row][wid]
        return enc, target

    def eval_holdout(m) -> tuple[float, float, float, list[float]]:  # noqa: ANN001
        m.eval()
        tp = fp = fn = 0
        probs: list[float] = []
        keep_idx = 1  # id2label {0: LABEL_0, 1: LABEL_1}; 1 = preserve for this model
        with torch.no_grad():
            for i in range(0, len(holdout), BATCH):
                enc, target = encode(holdout[i : i + BATCH])
                enc = enc.to(args.device)
                logits = m(**enc).logits
                keep_p = torch.softmax(logits.float(), dim=-1)[:, :, keep_idx].cpu()
                pred = keep_p >= 0.5
                mask = target != -100
                gold = target == 1
                tp += int((pred & gold & mask).sum())
                fp += int((pred & ~gold & mask).sum())
                fn += int((~pred & gold & mask).sum())
                probs.extend(keep_p[mask].tolist())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        m.train()
        return precision, recall, f1, probs

    p0, r0, f0, _ = eval_holdout(model)
    log.info("STOCK holdout label agreement (pre-training): P=%.3f R=%.3f F1=%.3f", p0, r0, f0)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR_LORA if args.arm == "lora" else LR_FULL,
    )
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    rng = random.Random(SEED)
    for epoch in range(EPOCHS):
        rng.shuffle(fit)
        total = 0.0
        for i in range(0, len(fit), BATCH):
            enc, target = encode(fit[i : i + BATCH])
            enc = enc.to(args.device)
            target = target.to(args.device)
            logits = model(**enc).logits
            loss = loss_fn(logits.view(-1, logits.shape[-1]), target.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += float(loss)
        log.info("epoch %d: mean loss %.4f", epoch, total / max(1, len(fit) // BATCH))


    precision, recall, f1, probs_all = eval_holdout(model)
    model.eval()
    log.info("ADAPTED holdout label agreement: P=%.3f R=%.3f F1=%.3f", precision, recall, f1)

    out = args.out or str(HERE / f"adapted-{args.corpus}-{args.arm}-{args.labels_suffix}")
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    Path(out, "eval.json").write_text(
        json.dumps(
            {
                "corpus": args.corpus,
                "arm": args.arm,
                "stock_f1": f0,
                "stock_precision": p0,
                "stock_recall": r0,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "n_fit": len(fit),
                "n_holdout": len(holdout),
                "prob_quantiles": {
                    q: sorted(probs_all)[int(len(probs_all) * q / 100)]
                    for q in (10, 25, 50, 75, 90)
                },
            },
            indent=2,
        )
    )
    log.info("saved %s", out)


if __name__ == "__main__":
    main()
