"""C2 Q2 stage 1: compressed variants of every routable task text (torch venv).

The served router (#259) embeds the LATEST USER MESSAGE (`_routable_text` in
wmh/serving/chat.py) and conversation affinity pins the incumbent after turn 1, so
routing-decision churn under compression is a property of the FIRST request of a
conversation: the task text, which is also what the knn fit bank embeds. This stage
produces, per corpus, every (method, aggressiveness) compressed variant of every task
text, plus achieved token ratios. Stage 2 (c2_churn_measure.py, wmh env) embeds them
and measures decision churn.

Methods: C1's bar-1 survivors with an aggressiveness knob get 3 levels; single-config
survivors get 1; random removal is the control, run at 3 levels so churn-vs-ratio
curves can be compared at matched achieved ratio at analysis time.
per-turn-truncate-at-append is skipped: on a single-turn request it IS head truncation
at a different budget, so it would duplicate the head-truncate rows.

Run inside C1's scratch venv (torch + transformers, not wmh deps):

    ~/Desktop/Projects/wmh-compression-data/cache/venv-c1/bin/python \
        .agents/scripts/c2_churn_variants.py

Scorer configs match C1's audit exactly (CPU fp32, per-turn locality, calibration
frozen per corpus) so achieved ratios stay comparable with round 0.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import statistics
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("c2_churn_variants")

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path.home() / "Desktop/Projects/wmh-compression-data"
ROUTING_MATRICES = Path.home() / "Desktop/Projects/wmh-routing-data/matrices"
OUT_DIR = DATA_ROOT / "cache/churn-variants"
CORPORA = ["routerbench-ours9", "financebench-s80", "tau-bench-s80"]

# Aggressiveness levels: nominal keep targets for ratio-style knobs, thresholds for the
# learned filter (higher threshold = more aggressive; 0.5 is C1's audited config).
KEEP_LEVELS = {"low": 0.7, "mid": 0.5, "high": 0.3}
LL2_THRESHOLDS = {"low": 0.35, "mid": 0.5, "high": 0.65}


def _load(name: str, path: Path):  # noqa: ANN202 - .agents pragma
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("compression", REPO / "wmh/research/compression.py")
AUDIT = _load("run_compression_audit", REPO / ".agents/scripts/run_compression_audit.py")


def task_texts(corpus: str) -> tuple[list[str], list[str]]:
    """(scenario_ids, tasks) in first-appearance order, the .npy cache row order."""
    outcomes = json.loads((ROUTING_MATRICES / f"{corpus}_matrix.json").read_text())["outcomes"]
    order: list[str] = []
    tasks: dict[str, str] = {}
    for entry in outcomes:
        if entry["scenario_id"] not in tasks:
            tasks[entry["scenario_id"]] = entry["task"]
            order.append(entry["scenario_id"])
    return order, [tasks[sid] for sid in order]


def build_variants(corpus: str, tasks: list[str], gpt2, ll2) -> dict[str, object]:  # noqa: ANN001
    median_words = int(statistics.median(len(C.split_words(t)) for t in tasks))
    all_units = [u for u in C.split_units("\n".join(tasks)) if u.strip()]
    methods: dict[str, object] = {}
    for level, keep in KEEP_LEVELS.items():
        si_threshold = AUDIT.calibrate_absolute_threshold(gpt2.bits_per_token, all_units, keep)
        methods[f"selective-context-absolute:{level}"] = C.ScoredUnitFilter(
            score_fn=gpt2.bits_per_token,
            mode="absolute",
            threshold=si_threshold,
            name=f"sc-abs-{level}",
        )
        methods[f"head-truncate-absolute:{level}"] = C.HeadTruncateAbsolute(
            budget_words=max(1, round(median_words * keep))
        )
        methods[f"random-removal:{level}"] = C.RandomRemoval(remove_ratio=1 - keep)
    for level, threshold in LL2_THRESHOLDS.items():
        methods[f"llmlingua2-fixed-threshold:{level}"] = C.ScoredTokenFilter(
            token_score_fn=lambda turn: list(ll2.word_probs(turn)),
            mode="absolute",
            threshold=threshold,
            name=f"ll2-fixed-{level}",
        )
    methods["dedup-keep-first:only"] = C.DedupKeepFirst(jaccard=0.9)
    methods["json-minify:only"] = C.JsonMinify()
    return methods


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gpt2 = AUDIT.Gpt2SelfInfo()
    ll2 = AUDIT.LLMLingua2Scorer()
    for corpus in CORPORA:
        out_path = OUT_DIR / f"{corpus}.jsonl"
        if out_path.exists():
            log.info("%s: exists, skipping", out_path)
            continue
        ids, tasks = task_texts(corpus)
        methods = build_variants(corpus, tasks, gpt2, ll2)
        ratios: dict[str, float] = {}
        with out_path.open("w") as handle:
            for variant, method in methods.items():
                raw_tok = comp_tok = 0
                for sid, task in zip(ids, tasks):
                    compressed = method([task])
                    raw_tok += AUDIT.count_tokens(gpt2.tokenizer, task)
                    comp_tok += AUDIT.count_tokens(gpt2.tokenizer, compressed)
                    handle.write(
                        json.dumps(
                            {"scenario_id": sid, "variant": variant, "text": compressed},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                ratios[variant] = comp_tok / raw_tok if raw_tok else 1.0
                log.info("%s / %s: achieved token ratio %.3f", corpus, variant, ratios[variant])
        (OUT_DIR / f"{corpus}-ratios.json").write_text(json.dumps(ratios, indent=2))
        log.info("wrote %s (%d scenarios x %d variants)", out_path, len(ids), len(methods))


if __name__ == "__main__":
    main()
