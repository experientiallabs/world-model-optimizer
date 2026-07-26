"""Score a TRAINED Tinker sampler on AIME (or MATH-500) with the baselines' grader.

Distillation is only worth anything if the student got better, and that question
is unanswerable without a post-training number that is comparable to the
pre-training one. Comparable means the SAME grader: `extract_boxed` and
`answers_match` are imported from `math500_baseline`, never restated here, so a
grader tweak can never move the trained number without moving the baseline too.

The measured quantity is a floor, not a point estimate, whenever a rollout is
truncated: a truncated sample has no `\\boxed{}` and scores wrong for a reason
that has nothing to do with math. So the truncation rate is reported NEXT TO the
accuracy, and accuracy over the finished samples is reported beside it, which is
the only pair that lets a reader tell "the student got worse" from "the student
got more verbose".

No reference numbers are embedded here on purpose. Prior runs' summaries live in
`.wmh/xtoken-runs/evals/`, and a difference between two of them is only a result
when dataset, temperature AND token budget all match -- differencing runs at
different budgets measures the budget. Every constant a previous revision of this
docstring carried was later withdrawn for exactly that reason.

Usage:
    # a trained checkpoint (path from training.save_weights_for_sampler(...).result().path)
    uv run python .agents/distill/aime_eval.py \
        --sampler-path tinker://my-run/weights/step-0007 \
        --dataset aime --out .wmh/xtoken-evals/step0007-aime.json

    # the untrained student, to re-derive the baseline under identical settings
    uv run python .agents/distill/aime_eval.py --base-model Qwen/Qwen3.5-9B --dataset aime
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tinker
from llm_waterfall.types import ChatMessage
from math500_baseline import (
    DATASETS,
    SYSTEM_PROMPT,
    answers_match,
    extract_boxed,
    load_problems,
)
from pydantic import BaseModel, Field

from wmh.distill.rendering import build_renderer

logger = logging.getLogger("aime-eval")

_CONTEXT_MARGIN = 16
"""Tokens held back from the context so the sampler cannot overrun it."""

EVAL_ARCHIVE = Path(".wmh/xtoken-runs/evals")
"""Where prior runs' summaries live, for a like-for-like comparison by hand.

Deliberately NOT a table of baseline constants. An earlier revision hardcoded
`aime: teacher 75.0, student 53.3` and `math500: teacher 87.5, student 75.0` and
printed a delta against them; every one of those four numbers was later
withdrawn, and because the delta controlled for neither temperature nor token
budget it produced a reported -48.3pp "regression" that was really a comparison
between a 45%-truncated run and a 0%-truncated one at different budgets. pass@1
is only meaningful alongside its temperature, budget and truncation rate, so this
script now reports its own conditions and leaves the comparison to a reader who
can match them."""


class EvalRow(BaseModel):
    """One sampled attempt at one problem."""

    index: int
    attempt: int
    gold: str
    predicted: str | None
    correct: bool
    sampled_tokens: int
    truncated: bool
    """True when the sample used its whole budget, so its answer was cut off."""

    text_tail: str
    """Last 300 characters, enough to see whether the model was mid-sentence."""


class EvalSummary(BaseModel):
    """Everything needed to compare this run against another one."""

    weights: str
    """The sampler path or base model actually evaluated."""

    base_model: str
    """Base model the renderer was built for, resolved from the sampler when implicit."""

    dataset: str
    problems: int
    samples: int
    k: int
    temperature: float
    max_tokens: int

    pass_at_1: float
    """Fraction correct over all samples (avg@k when k > 1)."""

    standard_error: float
    """Problem-clustered SE: attempts at one problem are not independent draws."""

    truncation_rate: float
    """Fraction of samples that used the whole budget. Nonzero makes pass@1 a FLOOR."""

    no_answer_rate: float
    """Fraction of samples with no parseable `\\boxed{}`, truncated or not."""

    pass_at_1_finished: float | None
    """Accuracy over non-truncated samples only, for reading the floor's size.

    None when EVERY sample truncated: there is no accuracy to report then, and a
    0.0 there would read as "the model got everything wrong".
    """

    mean_sampled_tokens: float
    errors: int
    """Samples that failed at the API and are excluded from every rate above."""


class EvalReport(BaseModel):
    """The written artifact: summary plus per-problem rows."""

    summary: EvalSummary
    rows: list[EvalRow] = Field(default_factory=list)


def clustered_standard_error(rows: list[EvalRow]) -> float:
    """Standard error of the mean accuracy, clustering attempts by problem.

    With `k > 1` the samples are not independent: several attempts share one
    problem, and a binomial SE over all samples would understate the true
    uncertainty (it would keep shrinking by drawing the same 30 problems again).
    The estimator therefore averages within each problem first and takes the SE
    across problems. At `k = 1` it reduces to the usual binomial SE up to the
    `n - 1` denominator, so it stays comparable with the baseline scripts.

    Args:
        rows: Successful samples, each carrying the problem index it came from.

    Returns:
        The standard error as a fraction (0..1); 0.0 when fewer than two
        problems produced samples, where no spread can be estimated.
    """
    per_problem: dict[int, list[float]] = {}
    for row in rows:
        per_problem.setdefault(row.index, []).append(1.0 if row.correct else 0.0)
    means = [sum(values) / len(values) for values in per_problem.values()]
    if len(means) < 2:
        return 0.0
    return statistics.stdev(means) / len(means) ** 0.5


def resolve_sampler(
    service: tinker.ServiceClient, sampler_path: str | None, base_model: str | None
) -> tuple[tinker.SamplingClient, str, str]:
    """The sampling client to evaluate, plus what it is.

    A trained checkpoint is a `tinker://` weights path with no model name in it,
    but the renderer needs the base model, so it is read back off the sampler
    (`get_base_model`) instead of being guessed or duplicated in a flag. An
    explicit `--base-model` alongside `--sampler-path` overrides that, which is
    only correct if the checkpoint really was trained from it.

    Args:
        service: Connected Tinker service client.
        sampler_path: A `tinker://` weights path, or None.
        base_model: A catalog base model name, or None.

    Returns:
        The sampling client, a human label for what was evaluated, and the base
        model name to render with.

    Raises:
        SystemExit: If neither argument was given, since there is nothing to score.
    """
    if sampler_path:
        sampler = service.create_sampling_client(model_path=sampler_path)
        own = sampler.get_base_model()
        if base_model and base_model != own:
            logger.warning(
                "--base-model %s overrides the sampler's own base model %s; the renderer will "
                "not match the weights unless the checkpoint really was trained from %s",
                base_model,
                own,
                base_model,
            )
        return sampler, sampler_path, base_model or own
    if base_model:
        return service.create_sampling_client(base_model=base_model), base_model, base_model
    raise SystemExit(
        "nothing to evaluate: pass --sampler-path tinker://<run>/weights/<name> for a trained "
        "checkpoint, or --base-model Qwen/Qwen3.5-9B to re-measure the untrained baseline"
    )


def context_limit_for(service: tinker.ServiceClient, base_model: str) -> int | None:
    """The catalog context length for a base model, or None when it is not listed."""
    return next(
        (
            model.max_context_length
            for model in service.get_server_capabilities().supported_models
            if model.model_name == base_model and model.max_context_length
        ),
        None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sampler-path",
        default=None,
        help="tinker:// weights path from save_weights_for_sampler (the trained student)",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="catalog base model; alone it scores the UNTRAINED student, and with "
        "--sampler-path it overrides the renderer's base model",
    )
    parser.add_argument("--dataset", default="aime", choices=DATASETS)
    parser.add_argument("--n", type=int, default=0, help="problems to score (0 means all)")
    parser.add_argument("--k", type=int, default=1, help="samples per problem (avg@k)")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="rollout budget; the training budget by default, so eval truncation matches "
        "what training tolerated",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0.0 matches the published baselines; raise it only together with --k",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--out", default=None, help="JSON path for the summary plus per-problem rows"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.k > 1 and args.temperature == 0.0:
        logger.warning(
            "--k %d at temperature 0.0 draws %d identical samples and reports a fake SE; "
            "pass --temperature 0.6 or 1.0",
            args.k,
            args.k,
        )

    problems = load_problems(args.n, args.dataset)
    if not problems:
        raise SystemExit(
            f"{args.dataset} loaded 0 problems; check the --dataset name against "
            f"{', '.join(DATASETS)}"
        )

    service = tinker.ServiceClient()
    sampler, weights, base_model = resolve_sampler(service, args.sampler_path, args.base_model)
    rendering = build_renderer(base_model, sampler.get_tokenizer())
    stop = rendering.stop_sequences
    stop_strings = stop if stop and isinstance(stop[0], str) else None
    context_limit = context_limit_for(service, base_model)
    if context_limit is None:
        logger.warning(
            "%s is not in the Tinker catalog, so the %d-token budget cannot be checked against "
            "its context; a too-large budget will fail at the API",
            base_model,
            args.max_tokens,
        )
    logger.info(
        "scoring %s (base %s) on %d %s problems, k=%d, max_tokens=%d",
        weights,
        base_model,
        len(problems),
        args.dataset,
        args.k,
        args.max_tokens,
    )

    errors = 0

    def run(item: tuple[int, int, dict[str, str]]) -> EvalRow | None:
        """Sample one attempt at one problem and grade it."""
        nonlocal errors
        index, attempt, row = item
        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=row["problem"]),
        ]
        prompt_ids = rendering.build_generation_prompt(messages)
        budget = args.max_tokens
        if context_limit is not None:
            room = context_limit - len(prompt_ids) - _CONTEXT_MARGIN
            if room < 1:
                raise SystemExit(
                    f"problem {index} has a {len(prompt_ids)}-token prompt, which leaves no "
                    f"room in a {context_limit}-token context; score a longer-context model"
                )
            budget = min(budget, room)
            if budget < args.max_tokens:
                logger.warning(
                    "problem %d gets %d of the requested %d tokens (prompt is %d tokens)",
                    index,
                    budget,
                    args.max_tokens,
                    len(prompt_ids),
                )
        try:
            response = sampler.sample(
                prompt=tinker.ModelInput.from_ints(prompt_ids),
                num_samples=1,
                sampling_params=tinker.SamplingParams(
                    max_tokens=budget,
                    temperature=args.temperature,
                    stop=stop_strings,
                ),
            ).result()
        except Exception as exc:  # noqa: BLE001 - one failed sample must not void the eval
            errors += 1
            logger.warning(
                "problem %d attempt %d failed to sample (%r); it is EXCLUDED from every rate, "
                "so the reported n shrinks rather than the accuracy dropping",
                index,
                attempt,
                exc,
            )
            return None
        sequence = response.sequences[0]
        text = rendering.decode(list(sequence.tokens))
        predicted = extract_boxed(text)
        return EvalRow(
            index=index,
            attempt=attempt,
            gold=row["answer"],
            predicted=predicted,
            correct=answers_match(row["answer"], predicted),
            sampled_tokens=len(sequence.tokens),
            truncated=len(sequence.tokens) >= budget,
            text_tail=text[-300:],
        )

    work = [
        (index, attempt, row)
        for index, row in enumerate(problems)
        for attempt in range(args.k)
    ]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = [row for row in pool.map(run, work) if row is not None]

    if not rows:
        raise SystemExit(
            f"every one of the {len(work)} samples failed ({errors} error(s)); check "
            "TINKER_API_KEY and the --sampler-path, then rerun"
        )

    samples = len(rows)
    correct = sum(1 for row in rows if row.correct)
    truncated = [row for row in rows if row.truncated]
    finished = [row for row in rows if not row.truncated]
    no_answer = sum(1 for row in rows if row.predicted is None)
    scored_problems = len({row.index for row in rows})
    summary = EvalSummary(
        weights=weights,
        base_model=base_model,
        dataset=args.dataset,
        problems=scored_problems,
        samples=samples,
        k=args.k,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        pass_at_1=correct / samples,
        standard_error=clustered_standard_error(rows),
        truncation_rate=len(truncated) / samples,
        no_answer_rate=no_answer / samples,
        pass_at_1_finished=(
            sum(1 for row in finished if row.correct) / len(finished) if finished else None
        ),
        mean_sampled_tokens=sum(row.sampled_tokens for row in rows) / samples,
        errors=errors,
    )

    logger.info("")
    logger.info("weights:          %s", summary.weights)
    logger.info("base model:       %s", summary.base_model)
    logger.info(
        "dataset:          %s (%d problems x k=%d = %d samples, temperature %.1f)",
        summary.dataset,
        summary.problems,
        summary.k,
        summary.samples,
        summary.temperature,
    )
    # Accuracy and truncation on ONE line: a nonzero truncation rate turns the
    # accuracy into a floor, so the two numbers must never be quoted apart.
    logger.info(
        "pass@1:           %.1f%%  (SE %.1fpp)   |   truncated: %.1f%% (%d/%d)",
        100 * summary.pass_at_1,
        100 * summary.standard_error,
        100 * summary.truncation_rate,
        len(truncated),
        summary.samples,
    )
    if summary.pass_at_1_finished is not None and truncated:
        logger.info(
            "                  ^ a FLOOR, not a measurement: %.1f%% over the %d finished "
            "samples alone. Raise --max-tokens (ceiling is the model context).",
            100 * summary.pass_at_1_finished,
            len(finished),
        )
    elif truncated:
        logger.info(
            "                  ^ EVERY sample truncated, so this 0.0% measures the budget and "
            "nothing about the model. Raise --max-tokens (ceiling is the model context)."
        )
    else:
        logger.info("                  ^ no truncation, so this is a measurement, not a floor")
    logger.info("no boxed answer:  %.1f%% (%d)", 100 * summary.no_answer_rate, no_answer)
    logger.info(
        "mean sampled tok: %.0f (budget %d)", summary.mean_sampled_tokens, summary.max_tokens
    )
    if summary.errors:
        logger.warning("api errors:       %d sample(s) excluded from every rate", summary.errors)
    # No baseline delta is printed. Comparing pass@1 across runs is only valid at
    # matched dataset, temperature and token budget, and this script cannot know
    # whether an archived run matches -- several archived files record no
    # temperature at all. Name the conditions and point at the archive instead.
    logger.info(
        "conditions:       %s, n=%d, k=%d, temperature %.1f, budget %d, truncation %.1f%%",
        summary.dataset,
        summary.problems,
        summary.k,
        summary.temperature,
        summary.max_tokens,
        100 * summary.truncation_rate,
    )
    if summary.truncation_rate > 0:
        logger.info(
            "                  ^ any comparison against a run with a DIFFERENT truncation "
            "rate measures the budget, not the model",
        )
    if EVAL_ARCHIVE.is_dir():
        logger.info(
            "prior runs for a like-for-like comparison: %s (match temperature AND budget "
            "before differencing)",
            EVAL_ARCHIVE,
        )

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        report = EvalReport(summary=summary, rows=sorted(rows, key=lambda r: (r.index, r.attempt)))
        path.write_text(json.dumps(report.model_dump(), indent=1), encoding="utf-8")
        logger.info("per-problem rows: %s", path)


if __name__ == "__main__":
    main()
