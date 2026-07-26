"""Measure a Tinker student's MATH-500 pass@1 baseline, to size the headroom.

Single-turn, no harbor and no E2B: render each MATH-500 problem with the base
model's cookbook renderer, sample greedily, extract the final `\\boxed{}` and
compare against the dataset answer. The point is to decide whether a student
has room for a distillation gain to be visible, so it also reports the
truncation rate and the no-answer rate, both of which depress accuracy for
reasons that have nothing to do with math ability.

Usage:
    uv run python .agents/distill/math500_baseline.py --model Qwen/Qwen3.5-9B --n 100
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger("math500")

SYSTEM_PROMPT = (
    "You are a careful mathematician. Solve the problem and put your final answer "
    "in \\boxed{}."
)

_BOXED = re.compile(r"\\boxed\s*\{")


def extract_boxed(text: str) -> str | None:
    """The content of the LAST `\\boxed{...}`, brace-matched.

    A regex alone cannot do this: answers contain nested braces
    (`\\boxed{\\frac{1}{2}}`), so the closing brace is found by counting depth.
    """
    best: str | None = None
    for match in _BOXED.finditer(text):
        depth = 1
        start = match.end()
        for index in range(start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    best = text[start:index]
                    break
    return best


def normalize(answer: str) -> str:
    """Canonicalize a LaTeX answer enough for exact comparison.

    Deliberately conservative: it strips presentation (spaces, `\\left`,
    trailing periods, `\\!`), unifies `dfrac`/`tfrac` to `frac`, drops `\\text{}`
    wrappers and units-ish trailing text, and removes thousands separators. It
    does NOT try to evaluate expressions, so `0.5` and `\\frac{1}{2}` stay
    different; that under-counts correctness slightly and is reported as such.
    """
    text = answer.strip()
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = text.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    text = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mbox\s*\{([^}]*)\}", r"\1", text)
    text = text.replace("^{\\circ}", "").replace("^\\circ", "")
    text = text.replace("\\%", "").replace("%", "")
    text = text.replace("\\$", "").replace("$", "")
    text = re.sub(r"(\d),(\d\d\d)", r"\1\2", text)
    text = text.rstrip(".")
    text = re.sub(r"\s+", "", text)
    # A bare trailing "\\)" or wrapping parens around a single value.
    if text.startswith("(") and text.endswith(")") and text.count("(") == 1:
        text = text[1:-1]
    return text


def answers_match(gold: str, predicted: str | None) -> bool:
    """Whether a prediction matches the gold answer after normalization.

    Beyond exact normalized equality, one rewrite is applied because MATH gold
    answers are inconsistent about it: an answer may carry its variable
    (`x=5`) while the model reports just the value (`5`), or vice versa. Both
    sides are therefore also compared on the text after the last `=`. Observed
    live: GLM-5.2 answered `5` to a problem whose gold was `x=5` and was scored
    wrong, which would have understated the teacher and thus the gate.

    No arithmetic is evaluated, so `0.5` and `\\frac{1}{2}` still differ. That
    is deliberate: a grader that starts evaluating expressions can silently
    accept a wrong answer that happens to simplify, and under-counting is the
    safer bias for a gate denominator.
    """
    if predicted is None:
        return False
    left = normalize(gold)
    right = normalize(predicted)
    if left == right:
        return True
    # AIME gold answers are zero-padded to three digits ('025'), while models
    # answer '25'. Observed live: 4 of GLM-5.2's 8 apparent AIME errors were
    # only this. Integers are compared by VALUE, which is unambiguous.
    if _as_int(left) is not None and _as_int(left) == _as_int(right):
        return True
    tail_gold = normalize(gold.rsplit("=", 1)[-1])
    tail_pred = normalize(predicted.rsplit("=", 1)[-1])
    if tail_gold == tail_pred:
        return True
    return _as_int(tail_gold) is not None and _as_int(tail_gold) == _as_int(tail_pred)


def _as_int(text: str) -> int | None:
    """The integer a normalized answer denotes, or None when it is not one."""
    stripped = text.lstrip("+")
    candidate = stripped[1:] if stripped.startswith("-") else stripped
    if not candidate or not candidate.isdigit():
        return None
    try:
        return int(stripped)
    except ValueError:  # pragma: no cover - isdigit already guarantees this parses
        return None


def _jsonl_rows(repo: str, filename: str) -> list[dict[str, str]]:
    """Rows of a JSONL dataset file, normalized to (problem, answer)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, filename, repo_type="dataset")
    out: list[dict[str, str]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        problem = row.get("problem") or row.get("question") or row.get("prompt")
        answer = row.get("answer") or row.get("solution") or row.get("expected_answer")
        if problem is None or answer is None:
            continue
        out.append({"problem": str(problem), "answer": str(answer)})
    return out


def _parquet_rows(repo: str, filename: str) -> list[dict[str, str]]:
    """Rows of a parquet dataset file, normalized to (problem, answer)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, filename, repo_type="dataset")
    table = pq.read_table(path).to_pylist()
    out: list[dict[str, str]] = []
    for row in table:
        problem = row.get("problem") or row.get("question") or row.get("Problem")
        answer = row.get("answer") or row.get("Answer") or row.get("expected_answer")
        if problem is None or answer is None:
            continue
        out.append({"problem": str(problem), "answer": str(answer)})
    return out


DATASETS = ("math500", "aime24", "aime25", "aime")
"""Eval sets this harness knows. `aime` is 2024 and 2025 concatenated."""


def load_problems(limit: int, dataset: str = "math500") -> list[dict[str, str]]:
    """Problems for one eval set, normalized to (problem, answer).

    AIME answers are integers 0-999, so grading them is far more reliable than
    MATH-500's LaTeX: the normalization false negatives that cost the teacher a
    point on MATH-500 cannot occur.

    Args:
        limit: Maximum problems to return, in file order (0 or negative means all).
        dataset: One of `DATASETS`.

    Returns:
        The problems, truncated to `limit`.

    Raises:
        ValueError: If `dataset` is unknown.
    """
    if dataset == "math500":
        rows = _jsonl_rows("HuggingFaceH4/MATH-500", "test.jsonl")
    elif dataset == "aime24":
        rows = _parquet_rows("HuggingFaceH4/aime_2024", "data/train-00000-of-00001.parquet")
    elif dataset == "aime25":
        rows = _jsonl_rows("math-ai/aime25", "test.jsonl")
    elif dataset == "aime":
        rows = load_problems(0, "aime24") + load_problems(0, "aime25")
    else:
        raise ValueError(f"unknown dataset {dataset!r}; choose one of {', '.join(DATASETS)}")
    return rows[:limit] if limit and limit > 0 else rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--dataset", default="math500", choices=DATASETS)
    parser.add_argument("--n", type=int, default=100, help="0 means the whole set")
    parser.add_argument("--k", type=int, default=1, help="samples per problem (avg@k)")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--out", default=None, help="Optional JSON path for per-problem rows")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import tinker
    from llm_waterfall.types import ChatMessage

    from wmh.distill.rendering import build_renderer

    problems = load_problems(args.n, args.dataset)
    logger.info("loaded %d %s problems (k=%d)", len(problems), args.dataset, args.k)
    if args.k > 1 and args.temperature == 0.0:
        logger.warning(
            "k=%d with temperature 0.0 draws identical samples; pass --temperature 0.6 or 1.0",
            args.k,
        )

    service = tinker.ServiceClient()
    sampler = service.create_sampling_client(base_model=args.model)
    rendering = build_renderer(args.model, sampler.get_tokenizer())
    stop = rendering.stop_sequences
    params = tinker.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stop=stop if stop and isinstance(stop[0], str) else None,
    )

    def run(item: tuple[int, int, dict[str, str]]) -> dict[str, object]:
        index, attempt, row = item
        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=row["problem"]),
        ]
        prompt_ids = rendering.build_generation_prompt(messages)
        future = sampler.sample(
            prompt=tinker.ModelInput.from_ints(prompt_ids),
            num_samples=1,
            sampling_params=params,
        )
        response = future.result()
        sequence = response.sequences[0]
        text = rendering.decode(list(sequence.tokens))
        predicted = extract_boxed(text)
        gold = row["answer"]
        return {
            "index": index,
            "attempt": attempt,
            "gold": gold,
            "predicted": predicted,
            "correct": answers_match(gold, predicted),
            "sampled_tokens": len(sequence.tokens),
            "truncated": len(sequence.tokens) >= args.max_tokens,
            "text_tail": text[-300:],
        }

    work = [
        (index, attempt, row)
        for index, row in enumerate(problems)
        for attempt in range(args.k)
    ]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(run, work))

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    truncated = sum(1 for r in results if r["truncated"])
    no_answer = sum(1 for r in results if r["predicted"] is None)
    mean_tokens = sum(int(r["sampled_tokens"]) for r in results) / total
    standard_error = (correct / total * (1 - correct / total) / total) ** 0.5

    logger.info("")
    logger.info("model:            %s", args.model)
    logger.info("problems:         %d (temperature %.1f, max_tokens %d)", total, args.temperature, args.max_tokens)
    logger.info("pass@1:           %.1f%%  (SE %.1fpp)", 100 * correct / total, 100 * standard_error)
    logger.info("truncated:        %d (%.0f%%)  <- depresses accuracy, raise max_tokens", truncated, 100 * truncated / total)
    logger.info("no boxed answer:  %d (%.0f%%)", no_answer, 100 * no_answer / total)
    logger.info("mean sampled tok: %.0f", mean_tokens)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
        logger.info("per-problem rows: %s", args.out)


if __name__ == "__main__":
    main()
