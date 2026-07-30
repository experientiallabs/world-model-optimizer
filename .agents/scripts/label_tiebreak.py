"""OVN-B: fable-5 tie-breaker labels on the windows where gpt-5.5 and opus-5 disagree most.

Cross-teacher agreement measured drop-set Jaccard 0.06-0.38: the teachers disagree at
the core, and the disagreement mass is exactly where labels are informative. This runner
takes the N windows per corpus with the LOWEST per-window drop-set agreement and asks a
third teacher (fable-5) under the same aggressive instruction + task context + retry
rule. Output rows carry all three teachers' keep fractions for the morning analysis.

    uv run python .agents/scripts/label_tiebreak.py --cap-usd 8
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path

from wmo.providers.base import Message
from wmo.providers.pool import load_pool, pool_provider

log = logging.getLogger("tiebreak")

DATA_ROOT = Path.home() / "Desktop/Projects/wmh-compression-data"
LABELS = DATA_ROOT / "cache/distill-labels"
METERING_PATH = DATA_ROOT / "cache/metering-c1.jsonl"
CORPORA = ["financebench", "swe-bench", "tau-bench", "terminal-tasks"]
N_PER_CORPUS = 30
WORD_RE = re.compile(r"\S+\s*")

INSTRUCTION = (
    "Compress the text by deleting as many words as possible while keeping every "
    "number, identifier, file path, code token, error message, entity name, and any "
    "word needed to preserve the facts. Grammar and fluency do not matter; telegraphic "
    "output is good. HARD RULE: the output must be an exact subsequence of the input's "
    "whitespace-delimited words (delete whole words only; never rewrite, reorder, "
    "merge, or add words). Output ONLY the compressed text."
)


def align_labels(raw: str, compressed: str) -> list[int] | None:
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "urllib3", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cap-usd", type=float, required=True)
    ap.add_argument("--model", default="fable-5")
    args = ap.parse_args()

    pool = load_pool()
    entry = pool.entry(args.model)
    price = entry.price()
    provider = pool_provider(entry)
    spent = 0.0
    n_discarded = 0
    for corpus in CORPORA:
        g = {
            (r["trace_id"], r["segment_index"], r.get("window_index", 0)): r
            for r in map(json.loads, (LABELS / f"labels-{corpus}-aggressive.jsonl").open())
        }
        o = {
            k: r
            for k, r in (
                (
                    (r["trace_id"], r["segment_index"], r.get("window_index", 0)),
                    r,
                )
                for r in map(
                    json.loads, (LABELS / f"labels-{corpus}-aggressive-opus-5.jsonl").open()
                )
            )
        }
        scored = []
        for k in g:
            if k not in o or len(g[k]["labels"]) != len(o[k]["labels"]):
                continue
            ga, oa = g[k]["labels"], o[k]["labels"]
            union = sum(1 for a, b in zip(ga, oa) if a == 0 or b == 0)
            both = sum(1 for a, b in zip(ga, oa) if a == 0 and b == 0)
            jacc = both / union if union else 1.0
            scored.append((jacc, k))
        scored.sort()
        picked = [k for _, k in scored[:N_PER_CORPUS]]
        out_path = LABELS / f"labels-{corpus}-tiebreak-{args.model}.jsonl"
        done = set()
        if out_path.exists():
            done = {
                (r["trace_id"], r["segment_index"], r.get("window_index", 0))
                for r in map(json.loads, out_path.open())
            }
        with out_path.open("a") as f:
            for k in picked:
                if k in done:
                    continue
                row = g[k]
                task_ctx = row["segment"][:1200]
                user = (
                    f"TASK CONTEXT (for judging relevance only; never copy it into the "
                    f"output): {task_ctx}\n\nTEXT TO COMPRESS:\n{row['segment']}"
                )
                completion = provider.complete(
                    INSTRUCTION, [Message(role="user", content=user)], temperature=0.0, max_tokens=2048
                )
                spent += (
                    completion.usage.input_tokens * price.input_per_mtok
                    + completion.usage.output_tokens * price.output_per_mtok
                ) / 1e6
                labels = align_labels(row["segment"], completion.text)
                if labels is None:
                    retry = provider.complete(
                        INSTRUCTION,
                        [
                            Message(role="user", content=user),
                            Message(role="assistant", content=completion.text),
                            Message(
                                role="user",
                                content=(
                                    "Your output was NOT an exact subsequence of the input's "
                                    "whitespace-delimited words. Redo it: only delete whole words. "
                                    "Output ONLY the compressed text."
                                ),
                            ),
                        ],
                        temperature=0.0,
                        max_tokens=2048,
                    )
                    spent += (
                        retry.usage.input_tokens * price.input_per_mtok
                        + retry.usage.output_tokens * price.output_per_mtok
                    ) / 1e6
                    labels = align_labels(row["segment"], retry.text)
                if labels is None:
                    n_discarded += 1
                    continue
                f.write(
                    json.dumps(
                        {
                            "corpus": corpus,
                            "trace_id": k[0],
                            "segment_index": k[1],
                            "window_index": k[2],
                            "segment": row["segment"],
                            "labels": labels,
                            "teacher": args.model,
                            "keep_fraction": sum(labels) / max(1, len(labels)),
                            "gpt55_keep": sum(g[k]["labels"]) / max(1, len(g[k]["labels"])),
                            "opus5_keep": sum(o[k]["labels"]) / max(1, len(o[k]["labels"])),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                f.flush()
                if spent > args.cap_usd:
                    log.warning("cap reached at $%.2f", spent)
                    break
        log.info("%s: tiebreak labels written, spend $%.2f", corpus, spent)
        if spent > args.cap_usd:
            break
    with METERING_PATH.open("a") as f:
        f.write(
            json.dumps(
                {
                    "ts": datetime.now(UTC).isoformat(),
                    "provider_model": args.model,
                    "purpose": "tiebreak-labels",
                    "owner": "c1",
                    "spend_usd": round(spent, 4),
                    "discarded": n_discarded,
                }
            )
            + "\n"
        )
    log.info("done: spend $%.2f, discarded %d", spent, n_discarded)


if __name__ == "__main__":
    main()
