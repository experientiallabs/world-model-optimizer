"""Screen DeepScaleR problems against a fixed rollout budget before training on them.

The training budget is DECIDED at 32,768 tokens, and a rollout that does not
finish inside it is never trained on (truncated reasoning is the wrong target
and it silently rewrites the batch composition). So the problems the student
cannot finish are removed UP FRONT instead of being discovered and dropped mid
step: this pass samples the student ONCE per problem at the training
temperature, records whether it finished and how many tokens it used, and
writes a manifest of the finishers plus a summary.

The number that decides whether 32k is the right budget is the FINISH RATE. Up
to 33% of problems may be filtered; above that the budget is too small (or the
student too verbose) and the decision has to be revisited, so the rate is
logged prominently against that tolerance, next to the token distribution that
says how much slack the budget actually has.

One sample per problem is a noisy per-problem verdict (a problem near the
budget can finish on one draw and not on the next) but an unbiased estimate of
the population rate, which is what the budget decision needs. Sampling is at
the TRAINING temperature, not greedily, for the same reason: the screen has to
see the distribution training will see.

Resumable by construction: a 32k generation takes minutes and a full pass takes
hours, so every verdict is appended to a progress ledger beside the manifest as
soon as it lands, and a rerun skips problems already screened.

Usage:
    uv run python .agents/distill/screen_problems.py --limit 200 \
        --out .wmh/xtoken-screen/deepscaler-32k.json
    # after an interruption, the exact same command resumes
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from typing import TextIO

import tinker
from llm_waterfall.types import ChatMessage
from pydantic import BaseModel, Field

from wmh.distill.rendering import build_renderer

logger = logging.getLogger("screen-problems")

SYSTEM_PROMPT = (
    "You are a careful mathematician. Solve the problem and put your final answer in \\boxed{}."
)
"""The training prompt. Screening under a different prompt would measure a different length."""

FILTER_TOLERANCE = 0.33
"""Fraction of problems Kion accepted losing to the budget filter."""

_CONTEXT_MARGIN = 16
"""Tokens held back from the context so the sampler cannot overrun it."""

_MANIFEST_EVERY = 20
"""Rewrite the manifest every N verdicts, so an interrupted pass still leaves one."""


class ScreenRow(BaseModel):
    """One problem's screening verdict."""

    problem_id: str
    """sha256 prefix of the problem text: the resume key, stable across --limit changes."""

    index: int
    """Position in the loaded DeepScaleR order, for cross-referencing a run."""

    problem: str
    answer: str

    sampled_tokens: int
    """Tokens the student generated. Equals the budget exactly when it did not finish."""

    finished: bool
    """True when generation stopped on its own before exhausting the budget."""

    budget: int
    """The budget this problem actually got, which the context can force below --max-tokens."""

    requested_max_tokens: int
    """The --max-tokens this verdict was screened against.

    Separate from `budget` so a resume can tell "screened at a different budget"
    (a mixing hazard) from "screened at this budget, shrunk by a long prompt"
    (legitimate). Comparing `budget` alone cannot distinguish the two.
    """


class TokenStats(BaseModel):
    """Distribution of sampled tokens over the problems that FINISHED.

    Unfinished problems are censored at the budget, so including them would pull
    every statistic toward the budget and understate the real tail.
    """

    count: int
    mean: float
    median: float
    p90: int
    p99: int
    max: int


class ScreenSummary(BaseModel):
    """The budget decision this pass exists to make."""

    student: str
    max_tokens: int
    temperature: float
    screened: int
    finished: int
    filtered: int
    finish_rate: float
    filtered_rate: float
    filter_tolerance: float = FILTER_TOLERANCE
    within_tolerance: bool
    """filtered_rate <= filter_tolerance: whether 32k survives its own screen."""

    finished_token_stats: TokenStats


class ScreenManifest(BaseModel):
    """The manifest: what to train on, plus why it is that set."""

    summary: ScreenSummary
    problems: list[ScreenRow] = Field(default_factory=list)
    """Only the finishers, in loaded order. This is the training set."""


def load_train_problems(limit: int, *, integers_only: bool = True) -> list[dict[str, str]]:
    """DeepScaleR problems, optionally restricted to integer answers.

    A local copy of `xtoken_run.load_train_problems` on purpose: the screen must
    enumerate the SAME problems in the SAME order as the trainer, and a shared
    import across two disposable scripts would let one drift under the other
    without either failing.

    Args:
        limit: Maximum problems to return, in file order (0 or negative means all).
        integers_only: Keep only integer answers, which is what the grader scores
            unambiguously and what the trainer trains on.

    Returns:
        The problems as (problem, answer) dicts, truncated to `limit`.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "agentica-org/DeepScaleR-Preview-Dataset", "deepscaler.json", repo_type="dataset"
    )
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[dict[str, str]] = []
    for row in rows:
        answer = str(row.get("answer", "")).strip()
        if integers_only and not answer.lstrip("-").isdigit():
            continue
        out.append({"problem": str(row["problem"]), "answer": answer})
        if limit and limit > 0 and len(out) >= limit:
            break
    return out


def problem_id(problem: str) -> str:
    """Stable resume key for one problem: a sha256 prefix of its text."""
    return sha256(problem.encode("utf-8")).hexdigest()[:16]


def percentile(values: list[int], quantile: float) -> int:
    """Nearest-rank percentile of `values` (`quantile` in 0..100).

    Nearest-rank rather than interpolated: the answer is always a token count
    that was actually observed, which is the honest thing to quote when sizing a
    budget against a tail.

    Args:
        values: Observations, non-empty.
        quantile: Percentile to take, 0 to 100.

    Returns:
        The observation at that rank.

    Raises:
        ValueError: If `values` is empty, since there is no percentile to report.
    """
    if not values:
        raise ValueError("percentile of an empty sample; screen at least one problem first")
    ordered = sorted(values)
    rank = math.ceil(quantile / 100 * len(ordered))
    return ordered[min(len(ordered) - 1, max(0, rank - 1))]


def token_stats(token_counts: list[int]) -> TokenStats:
    """Distribution summary of finished-rollout token counts (empty means all zeros)."""
    if not token_counts:
        return TokenStats(count=0, mean=0.0, median=0.0, p90=0, p99=0, max=0)
    return TokenStats(
        count=len(token_counts),
        mean=sum(token_counts) / len(token_counts),
        median=statistics.median(token_counts),
        p90=percentile(token_counts, 90),
        p99=percentile(token_counts, 99),
        max=max(token_counts),
    )


def build_manifest(
    rows: list[ScreenRow], *, student: str, max_tokens: int, temperature: float
) -> ScreenManifest:
    """The manifest for a set of verdicts: finishers plus the budget decision.

    Args:
        rows: Every screened problem, finished or not. The denominator of the
            finish rate is this list, so filtered problems must stay in it.
        student: Model that was screened, recorded so a manifest cannot be
            silently reused for a different student.
        max_tokens: Requested rollout budget.
        temperature: Sampling temperature used.

    Returns:
        The manifest, whose `problems` are only the finishers.
    """
    finishers = [row for row in rows if row.finished]
    screened = len(rows)
    finish_rate = len(finishers) / screened if screened else 0.0
    filtered_rate = 1.0 - finish_rate if screened else 0.0
    summary = ScreenSummary(
        student=student,
        max_tokens=max_tokens,
        temperature=temperature,
        screened=screened,
        finished=len(finishers),
        filtered=screened - len(finishers),
        finish_rate=finish_rate,
        filtered_rate=filtered_rate,
        within_tolerance=filtered_rate <= FILTER_TOLERANCE,
        finished_token_stats=token_stats([row.sampled_tokens for row in finishers]),
    )
    return ScreenManifest(summary=summary, problems=sorted(finishers, key=lambda row: row.index))


def write_manifest(path: Path, manifest: ScreenManifest) -> None:
    """Write the manifest atomically, so an interrupted rewrite cannot truncate it."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest.model_dump(), indent=1), encoding="utf-8")
    temporary.replace(path)


def ledger_path_for(manifest_path: Path) -> Path:
    """Where every verdict is appended, including the filtered ones.

    The manifest holds only finishers, so it cannot answer "was this problem
    already screened?" for a problem that failed: resuming off the manifest
    alone would re-run exactly the most expensive generations (the ones that
    burned the whole budget). The ledger is the resume record; the manifest is
    the derived artifact.
    """
    return manifest_path.with_suffix(manifest_path.suffix + ".progress.jsonl")


def check_resume_compatible(
    manifest_path: Path, prior: dict[str, ScreenRow], *, student: str, max_tokens: int
) -> None:
    """Refuse to resume a screen that was run under different settings.

    A verdict only means something relative to the budget and the model that
    produced it, so appending 65k verdicts to a 32k ledger (or a different
    student's) would quietly produce a finish rate that describes neither.
    Reruns after an interruption are the normal case, so this fails closed and
    names the fix rather than warning into a scrollback.

    Args:
        manifest_path: The `--out` path, quoted in the error.
        prior: Verdicts already on disk.
        student: Student this invocation would screen.
        max_tokens: Budget this invocation would screen against.

    Raises:
        SystemExit: If the prior verdicts were produced under a different budget
            or a different student.
    """
    if not prior:
        return
    # The REQUESTED budget, not the effective one: a long prompt legitimately
    # shrinks the effective budget for a single problem, and that must not read
    # as a settings change. Both directions of a real change are hazards. A
    # smaller earlier budget carries over false filters (a problem unfinished at
    # 8k may finish at 32k); a larger one carries over verdicts the new budget
    # would not have earned.
    budgets = sorted({row.requested_max_tokens for row in prior.values()} - {max_tokens})
    if budgets:
        raise SystemExit(
            f"{ledger_path_for(manifest_path)} already holds verdicts screened at "
            f"--max-tokens {budgets} but this run asks for {max_tokens}; mixing them would "
            "report a finish rate that belongs to no single budget. Rerun with the original "
            "--max-tokens, or point --out at a new file."
        )
    if not manifest_path.exists():
        return
    prior_student = ScreenManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    ).summary.student
    if prior_student and prior_student != student:
        raise SystemExit(
            f"{manifest_path} was screened with student {prior_student} but --student is "
            f"{student}; verdicts are model-specific. Point --out at a new file."
        )


def load_prior_rows(manifest_path: Path) -> dict[str, ScreenRow]:
    """Verdicts already on disk, keyed by problem id.

    Reads the ledger first. If only a manifest exists (a ledger deleted by hand,
    or a manifest produced elsewhere) its finishers are still honored and the
    caller is warned that the filtered problems will be screened again, because
    that information does not exist in the manifest.

    Args:
        manifest_path: The `--out` path; the ledger is derived from it.

    Returns:
        Problem id to verdict, last write winning.
    """
    ledger = ledger_path_for(manifest_path)
    rows: dict[str, ScreenRow] = {}
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = ScreenRow.model_validate_json(line)
            rows[row.problem_id] = row
        return rows
    if manifest_path.exists():
        manifest = ScreenManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        for row in manifest.problems:
            rows[row.problem_id] = row
        logger.warning(
            "%s exists but its ledger %s does not; only the %d finishers can be skipped and "
            "every previously FILTERED problem will be sampled again (the manifest does not "
            "record them). Keep the .progress.jsonl next to the manifest to avoid this.",
            manifest_path,
            ledger,
            len(rows),
        )
    return rows


def log_decision(summary: ScreenSummary) -> None:
    """Log the finish rate against the 33% filter tolerance, prominently.

    This block is the output of the pass: everything else is bookkeeping.
    """
    stats = summary.finished_token_stats
    tolerance = 100 * summary.filter_tolerance
    verdict = (
        f"WITHIN the {tolerance:.0f}% tolerance: the {summary.max_tokens}-token budget stands"
        if summary.within_tolerance
        else f"OVER the {tolerance:.0f}% tolerance: {summary.max_tokens} tokens is too small, "
        "revisit the budget"
    )
    logger.info("")
    logger.info("=" * 72)
    logger.info(
        "FINISH RATE: %.1f%%  (%d of %d problems)",
        100 * summary.finish_rate,
        summary.finished,
        summary.screened,
    )
    logger.info(
        "FILTERED:    %.1f%%  (%d problems) -> %s",
        100 * summary.filtered_rate,
        summary.filtered,
        verdict,
    )
    logger.info("=" * 72)
    logger.info("student:     %s (temperature %.1f)", summary.student, summary.temperature)
    logger.info("budget:      %d tokens", summary.max_tokens)
    logger.info("finished rollout tokens (censored problems excluded):")
    logger.info(
        "  mean %.0f | median %.0f | p90 %d | p99 %d | max %d",
        stats.mean,
        stats.median,
        stats.p90,
        stats.p99,
        stats.max,
    )
    if stats.count:
        logger.info(
            "  p99 uses %.0f%% of the budget: %s",
            100 * stats.p99 / summary.max_tokens,
            "comfortable slack"
            if stats.p99 < 0.9 * summary.max_tokens
            else "the tail is pressed against the budget, expect the finish rate to move",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", default="Qwen/Qwen3.5-9B")
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="problems to screen, in DeepScaleR order (0 means every integer-answer "
        "problem, which is thousands of multi-minute generations)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="the rollout budget under test; a problem that needs more is FILTERED",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="must match the training temperature, or the screen measures the wrong "
        "length distribution",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--out", default=".wmh/xtoken-screen/deepscaler-32k.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    manifest_path = Path(args.out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = ledger_path_for(manifest_path)

    prior = load_prior_rows(manifest_path)
    check_resume_compatible(
        manifest_path, prior, student=args.student, max_tokens=args.max_tokens
    )
    if prior:
        logger.info("resuming: %d problem(s) already screened in %s", len(prior), ledger)

    problems = load_train_problems(args.limit)
    todo = [
        (index, row)
        for index, row in enumerate(problems)
        if problem_id(row["problem"]) not in prior
    ]
    logger.info(
        "loaded %d DeepScaleR problems (integer answers); %d to screen at a %d-token budget",
        len(problems),
        len(todo),
        args.max_tokens,
    )

    if not todo:
        # Nothing left to sample, so no Tinker session is opened at all: a rerun
        # of a finished screen just rebuilds the manifest and reports.
        if not prior:
            raise SystemExit(
                f"--limit {args.limit} selected 0 problems and nothing was screened before; "
                "raise --limit (0 means the whole integer-answer set)"
            )
        manifest = build_manifest(
            list(prior.values()),
            student=args.student,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        write_manifest(manifest_path, manifest)
        log_decision(manifest.summary)
        logger.info("manifest: %s (%d trainable problems)", manifest_path, len(manifest.problems))
        return

    service = tinker.ServiceClient()
    sampler = service.create_sampling_client(base_model=args.student)
    rendering = build_renderer(args.student, sampler.get_tokenizer())
    stop = rendering.stop_sequences
    stop_strings = stop if stop and isinstance(stop[0], str) else None
    context_limit = next(
        (
            model.max_context_length
            for model in service.get_server_capabilities().supported_models
            if model.model_name == args.student and model.max_context_length
        ),
        None,
    )
    if context_limit is None:
        raise SystemExit(
            f"could not read max_context_length for {args.student} from the Tinker catalog; "
            "check the model name against `service.get_server_capabilities()`"
        )
    logger.info("model context limit: %d tokens", context_limit)

    rows: dict[str, ScreenRow] = dict(prior)
    lock = threading.Lock()
    errors = 0

    def screen_one(item: tuple[int, dict[str, str]], sink: TextIO) -> ScreenRow | None:
        """Sample one problem once and record whether it finished inside the budget."""
        nonlocal errors
        index, row = item
        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=row["problem"]),
        ]
        prompt_ids = rendering.build_generation_prompt(messages)
        room = context_limit - len(prompt_ids) - _CONTEXT_MARGIN
        if room < 1:
            raise SystemExit(
                f"problem {index} has a {len(prompt_ids)}-token prompt, which leaves no room "
                f"in a {context_limit}-token context; drop it from the dataset slice"
            )
        budget = min(args.max_tokens, room)
        if budget < args.max_tokens:
            logger.warning(
                "problem %d gets only %d of the requested %d tokens (prompt is %d); its "
                "verdict is against the SMALLER budget",
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
        except Exception as exc:  # noqa: BLE001 - one bad sample must not end a multi-hour pass
            with lock:
                errors += 1
            logger.warning(
                "problem %d failed to sample (%r); it is NOT recorded, so a rerun retries it",
                index,
                exc,
            )
            return None
        sampled = len(response.sequences[0].tokens)
        screened = ScreenRow(
            problem_id=problem_id(row["problem"]),
            index=index,
            problem=row["problem"],
            answer=row["answer"],
            sampled_tokens=sampled,
            finished=sampled < budget,
            budget=budget,
            requested_max_tokens=args.max_tokens,
        )
        with lock:
            # The ledger is written BEFORE anything is derived from the verdict:
            # a crash may lose the manifest, never the expensive generation.
            sink.write(screened.model_dump_json() + "\n")
            sink.flush()
            rows[screened.problem_id] = screened
            done = len(rows)
            snapshot = list(rows.values()) if done % _MANIFEST_EVERY == 0 else None
        if snapshot is not None:
            write_manifest(
                manifest_path,
                build_manifest(
                    snapshot,
                    student=args.student,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                ),
            )
        logger.info(
            "problem %d: %s at %d tokens (%d/%d screened)",
            index,
            "finished" if screened.finished else "UNFINISHED, filtered",
            sampled,
            len(rows),
            len(problems),
        )
        return screened

    with ledger.open("a", encoding="utf-8") as sink:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(lambda item: screen_one(item, sink), todo))

    if not rows:
        raise SystemExit(
            f"no problem was screened ({errors} sampling error(s)); check TINKER_API_KEY and "
            "the --student name, then rerun (nothing was recorded, so nothing is lost)"
        )

    manifest = build_manifest(
        list(rows.values()),
        student=args.student,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    write_manifest(manifest_path, manifest)
    log_decision(manifest.summary)
    if errors:
        logger.warning(
            "%d problem(s) errored and are absent from both numerator and denominator; "
            "rerun the same command to screen them",
            errors,
        )
    logger.info("manifest: %s (%d trainable problems)", manifest_path, len(manifest.problems))
    logger.info("ledger:   %s (every verdict, including the filtered ones)", ledger)


if __name__ == "__main__":
    main()
