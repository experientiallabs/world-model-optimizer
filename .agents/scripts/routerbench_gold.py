"""Recover certified gold answers for RouterBench's MCQ evals from the matrix itself.

RouterBench ships no gold column, but its MCQ prompts embed the choices and its responses are
bare letters, so gold is recoverable by consensus: parse the letter from every model with
score 1.0; certify the prompt when (a) at least two score-1.0 models agree on one letter and
(b) no score-0.0 model's parsed letter equals it. Certification rate is printed per eval, and
uncertified prompts are dropped, never guessed.

Output: .wmo/evals/routerbench/gold.json {scenario_id: letter} + stats.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("routerbench_gold")

PICKLE = Path("/Users/silen/Desktop/Projects/router-refs/routerbench_0shot.pkl")
OUT = Path(".wmo/evals/routerbench/gold.json")

MCQ_PREFIXES = ("mmlu-", "arc-challenge", "hellaswag", "winogrande")

MODELS = [
    "WizardLM/WizardLM-13B-V1.2",
    "claude-instant-v1",
    "claude-v1",
    "claude-v2",
    "gpt-3.5-turbo-1106",
    "gpt-4-1106-preview",
    "meta/code-llama-instruct-34b-chat",
    "meta/llama-2-70b-chat",
    "mistralai/mistral-7b-chat",
    "mistralai/mixtral-8x7b-chat",
    "zero-one-ai/Yi-34B-Chat",
]

_LETTER = re.compile(r"\b([A-E])\b[).:]?")


def parse_letter(response: str) -> str | None:
    """First standalone choice letter in a response ('C', 'A)', 'Answer: B', ...)."""
    text = response.strip().strip("'[]\"")
    match = _LETTER.search(text[:80])
    return match.group(1) if match else None


def main() -> None:
    frame = pd.read_pickle(PICKLE)
    frame = frame[frame["eval_name"].str.startswith(MCQ_PREFIXES)]
    logger.info("MCQ rows: %d prompts", len(frame))

    gold: dict[str, str] = {}
    stats: Counter[str] = Counter()
    for row in frame.to_dict("records"):
        sid = f"{row['eval_name']}:{row['sample_id']}"
        winners: Counter[str] = Counter()
        losers: set[str] = set()
        for model in MODELS:
            letter = parse_letter(str(row[f"{model}|model_response"]))
            if letter is None:
                continue
            score = float(row[model])
            if score >= 1.0:
                winners[letter] += 1
            elif score <= 0.0:
                losers.add(letter)
        if not winners:
            stats["no_winner"] += 1
            continue
        if len(winners) > 1:
            stats["winner_conflict"] += 1
            continue
        (letter, count), *_ = winners.most_common(1)
        if count < 2:
            stats["single_witness"] += 1
            continue
        if letter in losers:
            stats["loser_contradiction"] += 1
            continue
        gold[sid] = letter
        stats["certified"] += 1

    total = len(frame)
    logger.info("certified %d/%d (%.1f%%); drops: %s", len(gold), total, 100 * len(gold) / total, dict(stats))
    by_eval: Counter[str] = Counter(sid.split(":", 1)[0] for sid in gold)
    logger.info("certified by eval (top): %s", dict(by_eval.most_common(8)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(gold, indent=0), encoding="utf-8")
    logger.info("wrote %s", OUT)
    sys.exit(0)


if __name__ == "__main__":
    main()
