"""Round 0 compression audit runner (track C1): append-stability + offline scorecard.

Builds multi-turn audit transcripts from the routing matrices' stored replies (real
corpora, per the C1 brief), wires the model-backed scorers (GPT-2 self-information for
the Selective-Context-style filter, LLMLingua-2 BERT keep-probabilities for the learned
filter) into the pure method slate in wmh/research/compression.py, and measures per
method x corpus: same-input determinism, append churn, token-reduction ratio, and
compressor latency.

Run inside the scratch venv (this script needs torch + transformers, which are NOT wmh
dependencies):

    uv venv ~/Desktop/Projects/wmh-compression-data/cache/venv-c1 --python 3.12
    uv pip install --python ~/Desktop/Projects/wmh-compression-data/cache/venv-c1 \
        torch transformers
    ~/Desktop/Projects/wmh-compression-data/cache/venv-c1/bin/python \
        .agents/scripts/run_compression_audit.py --build-corpus
    ~/Desktop/Projects/wmh-compression-data/cache/venv-c1/bin/python \
        .agents/scripts/run_compression_audit.py

wmh/research/compression.py is loaded straight from its file path so the scratch venv
does not need the full wmh dependency tree (.agents workspace pragma).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import statistics
import time
import uuid
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

log = logging.getLogger("compression_audit")

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path.home() / "Desktop/Projects/wmh-compression-data"
ROUTING_MATRICES = Path.home() / "Desktop/Projects/wmh-routing-data/matrices"
CORPUS_PATH = DATA_ROOT / "cache/audit-transcripts.jsonl"
RESULTS_PATH = DATA_ROOT / "cache/round0-results.json"
TABLE_PATH = DATA_ROOT / "cache/round0-table.md"
RUNS_PATH = DATA_ROOT / "runs/c1.jsonl"

CORPORA = ["tau-bench", "terminal-tasks", "swe-bench", "financebench", "tau-bench-real"]
MAX_TRANSCRIPTS_PER_CORPUS = 24
MIN_TURNS = 4
MAX_TURNS = 12
TARGET_KEEP_RATIO = 0.5  # every ratio-style method is matched to this
LLMLINGUA2_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"


def _load_compression_module():  # noqa: ANN202 - .agents pragma
    import sys

    spec = importlib.util.spec_from_file_location(
        "compression", REPO / "wmh/research/compression.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compression"] = mod  # dataclasses resolves cls.__module__ via sys.modules
    spec.loader.exec_module(mod)
    return mod


C = _load_compression_module()


# ---------------------------------------------------------------------------
# Corpus building.
# ---------------------------------------------------------------------------


def _pretty_task(task: str) -> str:
    try:
        parsed = json.loads(task)
        if isinstance(parsed, dict):
            return "\n".join(f"{k}: {v}" for k, v in parsed.items())
    except (json.JSONDecodeError, ValueError):
        pass
    return task


def build_corpus() -> None:
    """Sample audit transcripts from the routing matrices, deterministically."""
    import random

    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for corpus in CORPORA:
        path = ROUTING_MATRICES / f"{corpus}_matrix.json"
        outcomes = json.loads(path.read_text())["outcomes"]
        candidates = [
            e
            for e in outcomes
            if isinstance(e.get("replies"), list)
            and len(e["replies"]) >= MIN_TURNS
            and not e.get("error")
        ]
        # Spread across scenarios: at most one episode per (scenario, model).
        seen: set[tuple[str, str]] = set()
        unique = []
        for e in candidates:
            key = (e["scenario_id"], e["model"])
            if key in seen:
                continue
            seen.add(key)
            unique.append(e)
        rng = random.Random(0)
        rng.shuffle(unique)
        picked = unique[:MAX_TRANSCRIPTS_PER_CORPUS]
        for e in picked:
            turns = [_pretty_task(e["task"])] + [str(r) for r in e["replies"][: MAX_TURNS - 1]]
            rows.append(
                {
                    "corpus": corpus,
                    "scenario_id": e["scenario_id"],
                    "model": e["model"],
                    "episode": e["episode"],
                    "turns": turns,
                }
            )
        log.info("%s: %d transcripts (from %d candidates)", corpus, len(picked), len(unique))
    with CORPUS_PATH.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("wrote %d transcripts -> %s", len(rows), CORPUS_PATH)


def load_corpus() -> dict[str, list[list[str]]]:
    by_corpus: dict[str, list[list[str]]] = {}
    with CORPUS_PATH.open() as f:
        for line in f:
            row = json.loads(line)
            by_corpus.setdefault(row["corpus"], []).append(row["turns"])
    return by_corpus


# ---------------------------------------------------------------------------
# Model-backed scorers (cached per unit/turn; locality is what the audit tests).
# ---------------------------------------------------------------------------


class Gpt2SelfInfo:
    """Bits-per-token self-information of a unit, scored in isolation (fp32 CPU)."""

    def __init__(self) -> None:
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        self.torch = torch
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").eval()

    @lru_cache(maxsize=200_000)
    def bits_per_token(self, unit: str) -> float:
        text = unit.strip()
        if not text:
            return 0.0
        ids = self.tokenizer(text, truncation=True, max_length=512, return_tensors="pt")[
            "input_ids"
        ]
        if ids.shape[1] < 2:
            return 20.0  # single-token units carry maximal per-token surprise; keep them
        with self.torch.no_grad():
            logits = self.model(ids).logits
        logprobs = self.torch.log_softmax(logits[0, :-1], dim=-1)
        token_lp = logprobs[range(ids.shape[1] - 1), ids[0, 1:]]
        return float(-token_lp.mean().item() / 0.6931471805599453)  # nats -> bits


class LLMLingua2Scorer:
    """Per-turn word keep-probabilities from the LLMLingua-2 177M token classifier.

    The turn is scored in isolation (per-turn chunking at 510-token windows relative to
    the turn start), so the scorer itself is append-local; the selection rule is the
    variable under audit. Deviation from the stock pipeline, recorded in findings: stock
    chunks the WHOLE prompt at global 512 boundaries, which adds churn on top of the
    percentile rule, so our percentile-mode churn is a lower bound on stock.
    """

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(LLMLINGUA2_MODEL)
        self.model = AutoModelForTokenClassification.from_pretrained(LLMLINGUA2_MODEL).eval()
        id2label = self.model.config.id2label
        keep_ids = [i for i, lab in id2label.items() if str(lab).lower() in ("1", "preserve")]
        self.keep_idx = keep_ids[0] if keep_ids else 1
        log.info("llmlingua-2 id2label=%s keep_idx=%d", id2label, self.keep_idx)

    @lru_cache(maxsize=100_000)
    def word_probs(self, turn: str) -> tuple[tuple[str, float], ...]:
        words_ws = C.split_words(turn)
        words = [w.strip() for w in words_ws]
        if not words:
            return ()
        probs: list[float] = []
        window = 400  # words per window; conservative vs the 512-subtoken limit
        for start in range(0, len(words), window):
            chunk = words[start : start + window]
            enc = self.tokenizer(
                chunk,
                is_split_into_words=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with self.torch.no_grad():
                logits = self.model(**enc).logits
            p_keep = self.torch.softmax(logits[0], dim=-1)[:, self.keep_idx]
            word_ids = enc.word_ids(0)
            sums = [0.0] * len(chunk)
            counts = [0] * len(chunk)
            for pos, wid in enumerate(word_ids):
                if wid is None:
                    continue
                sums[wid] += float(p_keep[pos])
                counts[wid] += 1
            probs.extend(s / c if c else 0.0 for s, c in zip(sums, counts))
        return tuple(zip(words_ws, probs))


# ---------------------------------------------------------------------------
# The slate.
# ---------------------------------------------------------------------------


def calibrate_absolute_threshold(score_fn, units: list[str], keep_ratio: float) -> float:  # noqa: ANN001
    """Freeze an absolute threshold that keeps ~keep_ratio of corpus units on average."""
    scores = sorted(score_fn(u) for u in units)
    idx = max(0, min(len(scores) - 1, round(len(scores) * (1 - keep_ratio))))
    return scores[idx]


def build_slate(corpus_turns: list[list[str]], gpt2: Gpt2SelfInfo, ll2: LLMLingua2Scorer) -> list:
    """Instantiate every slate method for one corpus (fixed, recorded configs)."""
    all_text = C.join_turns([t for turns in corpus_turns for t in turns])
    median_words = int(statistics.median(len(C.split_words(C.join_turns(t))) for t in corpus_turns))
    all_units = [u for u in C.split_units(all_text) if u.strip()]
    si_threshold = calibrate_absolute_threshold(
        gpt2.bits_per_token, all_units, TARGET_KEEP_RATIO
    )
    log.info(
        "corpus calibration: median_words=%d selfinfo_abs_threshold=%.3f bits/token",
        median_words,
        si_threshold,
    )

    def si_score(unit: str) -> float:
        return gpt2.bits_per_token(unit)

    def ll2_score(turn: str) -> list[tuple[str, float]]:
        return list(ll2.word_probs(turn))

    budget = max(1, round(median_words * TARGET_KEEP_RATIO))
    return [
        C.HeadTruncateAbsolute(budget_words=budget),
        C.HeadTruncateRatio(keep_ratio=TARGET_KEEP_RATIO),
        C.TailRecencyWindow(budget_words=budget),
        C.RandomRemoval(remove_ratio=1 - TARGET_KEEP_RATIO),
        C.DedupKeepFirst(jaccard=0.9),
        C.PerTurnTruncateAtAppend(per_turn_budget_words=max(20, budget // MAX_TURNS)),
        C.RollingObservationMask(window=3),
        C.JsonMinify(),
        C.ScoredUnitFilter(
            score_fn=si_score,
            mode="percentile",
            keep_ratio=TARGET_KEEP_RATIO,
            name="selective-context-percentile",
        ),
        C.ScoredUnitFilter(
            score_fn=si_score,
            mode="absolute",
            threshold=si_threshold,
            name="selective-context-absolute",
        ),
        C.ScoredTokenFilter(
            token_score_fn=ll2_score,
            mode="percentile",
            keep_ratio=TARGET_KEEP_RATIO,
            name="llmlingua2-percentile",
        ),
        C.ScoredTokenFilter(
            token_score_fn=ll2_score,
            mode="absolute",
            threshold=0.5,
            name="llmlingua2-fixed-threshold",
        ),
    ]


# ---------------------------------------------------------------------------
# Audit loop.
# ---------------------------------------------------------------------------


def count_tokens(tokenizer, text: str) -> int:  # noqa: ANN001
    return len(tokenizer(text)["input_ids"])


def audit_corpus(  # noqa: ANN001
    corpus: str, transcripts: list[list[str]], methods: list, gpt2_tok, clear_caches
) -> list[dict]:
    results = []
    for method in methods:
        audit = C.MethodAudit(method=method.name, corpus=corpus)
        clear_caches()  # latency below is scorer-inclusive, cold per method
        cold_latencies: list[tuple[int, float]] = []  # (raw_tokens, seconds), cold cache
        for i, turns in enumerate(transcripts):
            raw = C.join_turns(turns)
            t0 = time.perf_counter()
            compressed = method(turns)
            cold_latencies.append((count_tokens(gpt2_tok, raw), time.perf_counter() - t0))
            audit.raw_chars += len(raw)
            audit.compressed_chars += len(compressed)
            audit.churns.extend(C.append_churn(method, turns))
            if i < 3 and not C.is_deterministic(method, turns):
                audit.deterministic = False
        raw_tok_total = sum(t for t, _ in cold_latencies)
        compressed_tok_total = sum(
            count_tokens(gpt2_tok, method(turns)) for turns in transcripts
        )
        per_10k = sorted(s / t * 10_000 for t, s in cold_latencies if t)
        results.append(
            {
                "method": method.name,
                "corpus": corpus,
                "deterministic": audit.deterministic,
                "append_only": audit.append_only,
                "churn_mean": round(audit.churn_mean, 4),
                "churn_max": round(audit.churn_max, 4),
                "frac_append_stable": round(audit.frac_append_stable, 4),
                "token_ratio": round(compressed_tok_total / raw_tok_total, 4),
                "latency_s_per_10k_tok_p50": round(per_10k[len(per_10k) // 2], 4),
                "n_transcripts": len(transcripts),
                "n_append_events": len(audit.churns),
            }
        )
        log.info(
            "%s / %s: append_only=%s churn_mean=%.3f ratio=%.3f",
            corpus,
            method.name,
            audit.append_only,
            audit.churn_mean,
            compressed_tok_total / raw_tok_total,
        )
    return results


def write_outputs(all_results: list[dict], device: str) -> None:
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2))
    ts = datetime.now(UTC).isoformat()
    with RUNS_PATH.open("a") as f:
        for r in all_results:
            rid = hashlib.sha1(
                f"{r['corpus']}-{r['method']}-round0-{uuid.uuid4()}".encode()
            ).hexdigest()[:8]
            f.write(
                json.dumps(
                    {
                        "run_id": f"{r['corpus']}-round0-{r['method']}-{rid}",
                        "ts": ts,
                        "matrix": r["corpus"],
                        "variant": f"round0-audit-{r['method']}",
                        "params": {
                            "target_keep_ratio": TARGET_KEEP_RATIO,
                            "device": device,
                        },
                        "split_seed": 0,
                        "fit_scenarios": 0,
                        "test_scenarios": r["n_transcripts"],
                        "result": {k: v for k, v in r.items() if k not in ("method", "corpus")},
                        "notes": "round0 offline audit; no accuracy; latency on local CPU fp32",
                    }
                )
                + "\n"
            )
    methods = sorted({r["method"] for r in all_results})
    corpora = sorted({r["corpus"] for r in all_results})
    lines = [
        "| method | det | append-only (corpora) | churn mean | token ratio | s/10k tok p50 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for m in methods:
        rows = [r for r in all_results if r["method"] == m]
        stable = sum(1 for r in rows if r["append_only"])
        det = all(r["deterministic"] for r in rows)
        lines.append(
            f"| {m} | {'yes' if det else 'NO'} | {stable}/{len(corpora)} | "
            f"{statistics.fmean(r['churn_mean'] for r in rows):.3f} | "
            f"{statistics.fmean(r['token_ratio'] for r in rows):.3f} | "
            f"{statistics.median(r['latency_s_per_10k_tok_p50'] for r in rows):.3f} |"
        )
    TABLE_PATH.write_text("\n".join(lines) + "\n")
    log.info("wrote %s, %s, appended %d rows to %s", RESULTS_PATH, TABLE_PATH, len(all_results), RUNS_PATH)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-corpus", action="store_true")
    parser.add_argument("--corpora", nargs="*", default=CORPORA)
    args = parser.parse_args()
    if args.build_corpus:
        build_corpus()
        return
    import torch

    torch.set_num_threads(8)
    from transformers import GPT2TokenizerFast

    gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2")
    gpt2 = Gpt2SelfInfo()
    ll2 = LLMLingua2Scorer()
    by_corpus = load_corpus()
    all_results: list[dict] = []
    for corpus in args.corpora:
        transcripts = by_corpus.get(corpus, [])
        if not transcripts:
            log.warning("no transcripts for %s", corpus)
            continue
        methods = build_slate(transcripts, gpt2, ll2)

        def clear_caches() -> None:
            gpt2.bits_per_token.cache_clear()
            ll2.word_probs.cache_clear()

        all_results.extend(audit_corpus(corpus, transcripts, methods, gpt2_tok, clear_caches))
    write_outputs(all_results, device="cpu-fp32")


if __name__ == "__main__":
    main()
