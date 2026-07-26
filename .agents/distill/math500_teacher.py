"""Measure the GLM-5.2 teacher's MATH-500 / AIME pass@1, for the same grader as the student.

The distillation gate is a fraction of the teacher's solve rate, so the teacher
number has to come from the SAME extraction and normalization as the student's
or the ratio is meaningless. This imports both from `math500_baseline` rather
than restating them.

Generation goes through a chat-completions endpoint. The teacher only needs to
GENERATE here, so evals default to the Azure AI Foundry deployment, which
proxies the same weights (its responses report
`accounts/fireworks/models/glm-5p2`) and is a resource we can burn freely.
Fireworks is reserved for TRAINING tokens, where its unique `echo` prompt-logprob
surface is required (see `wmh.distill.xtoken.prompt_logprobs`); spending eval
tokens there would eat the training budget for no benefit.

Three failure modes made the first teacher measurement a FLOOR rather than a
score, and all three are handled here because each one silently scores a
correct answer as wrong:

1. Answer in the reasoning channel. Some GLM deployments split the response
   into `reasoning_content` (the chain of thought) and `content` (the final
   answer). When the reasoning runs long the `content` field comes back empty
   even though the `\\boxed{}` was emitted, so `_extract_answer` falls back to
   the reasoning channel and records which channel the answer came from.
2. Truncation. A response cut off at `max_tokens` has no answer, which grades
   as wrong; it is not a wrong answer. Truncation is counted separately and a
   truncation-excluded pass@1 is reported alongside the raw one.
3. Client read timeouts. A 600s client timeout on a request the server was
   still happily answering was previously recorded as an incorrect answer.
   The timeout now defaults to `DEFAULT_TIMEOUT_S` and retryable failures are
   retried, so slowness costs wall clock instead of accuracy.

Usage:
    uv run python .agents/distill/math500_teacher.py --dataset aime --n 0
    uv run python .agents/distill/math500_teacher.py --provider fireworks --n 10
"""

from __future__ import annotations

import argparse
import http.client
import json
import logging
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from math500_baseline import (
    DATASETS,
    SYSTEM_PROMPT,
    answers_match,
    extract_boxed,
    load_problems,
)

logger = logging.getLogger("math500-teacher")

FIREWORKS_CHAT_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
FIREWORKS_MODEL = "accounts/fireworks/models/glm-5p2"

PROVIDERS = ("azure", "fireworks")
"""Where eval generation goes. `azure` is the default and is free to burn."""

# Fireworks output price, used only to report what a fireworks-provider run cost
# against the training budget.
FIREWORKS_OUTPUT_USD_PER_MTOK = 4.40

DEFAULT_TIMEOUT_S = 5400
"""Client read timeout, in seconds.

Deliberately far longer than any plausible generation. A reasoning model at a
100k-token budget behind a shared endpoint can take well over an hour of wall
clock for one problem, and the previous 600s cap turned 11 of 60 AIME requests
into recorded WRONG ANSWERS. A timeout is a measurement failure, never
evidence about the model, so the cap is set where it can only fire on a truly
dead connection.
"""

REASONING_FIELDS = ("reasoning_content", "reasoning")
"""Message fields, in order, that may carry the answer when `content` does not."""


def resolve_provider(provider: str, model: str | None) -> tuple[str, str, dict[str, str]]:
    """The (url, model, headers) for one provider, read from the environment.

    Args:
        provider: One of `PROVIDERS`.
        model: Explicit model or deployment override; None uses the provider default.

    Returns:
        The chat-completions URL, the model id to send, and the auth headers.

    Raises:
        SystemExit: If the provider's credentials are absent, naming the env var.
    """
    if provider == "fireworks":
        key = os.environ.get("FIREWORKS_API_KEY")
        if not key:
            raise SystemExit(
                "FIREWORKS_API_KEY is not set; source platform/.env.local, or use the "
                "default --provider azure so training budget is not spent on evals"
            )
        return (
            FIREWORKS_CHAT_URL,
            model or FIREWORKS_MODEL,
            {"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        )
    endpoint = os.environ.get("AZURE_FOUNDRY_ENDPOINT")
    key = os.environ.get("AZURE_FOUNDRY_API_KEY")
    deployment = model or os.environ.get("AZURE_FOUNDRY_GLM52_DEPLOYMENT")
    missing = [
        name
        for name, value in (
            ("AZURE_FOUNDRY_ENDPOINT", endpoint),
            ("AZURE_FOUNDRY_API_KEY", key),
            ("AZURE_FOUNDRY_GLM52_DEPLOYMENT", deployment),
        )
        if not value
    ]
    if missing:
        raise SystemExit(
            f"{' and '.join(missing)} not set; source platform/.env.local before running"
        )
    assert endpoint is not None and key is not None and deployment is not None  # noqa: S101
    return (
        endpoint.rstrip("/") + "/chat/completions",
        deployment,
        {"Content-Type": "application/json", "api-key": key, "Authorization": f"Bearer {key}"},
    )


def _channel_text(value: object) -> str:
    """A reasoning field flattened to text.

    Providers disagree about the shape: a plain string, or a list of content
    blocks each carrying `text`/`content`/`thinking`. Anything else yields "".
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for block in value:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                for key in ("text", "content", "thinking", "reasoning"):
                    inner = block.get(key)
                    if isinstance(inner, str):
                        parts.append(inner)
                        break
        return "".join(parts)
    return ""


def _extract_answer(message: dict[str, object]) -> tuple[str | None, str, str]:
    """The predicted answer, the channel it came from, and that channel's text.

    `content` wins when it contains a `\\boxed{}`. Otherwise the reasoning
    channels are searched, because a long chain of thought can consume the whole
    token budget and leave `content` empty while the answer sits, boxed, at the
    end of the reasoning. Recovering it is a measurement fix, not a grader
    loosening: the same `extract_boxed` runs on both channels.
    """
    content = _channel_text(message.get("content"))
    predicted = extract_boxed(content)
    if predicted is not None:
        return predicted, "content", content
    for field in REASONING_FIELDS:
        text = _channel_text(message.get(field))
        if not text:
            continue
        predicted = extract_boxed(text)
        if predicted is not None:
            return predicted, field, text
    # Nothing boxed anywhere; report the longest channel so the row is inspectable.
    fallback = content
    for field in REASONING_FIELDS:
        text = _channel_text(message.get(field))
        if len(text) > len(fallback):
            fallback = text
    return None, "none", fallback


def _first_message(payload: dict[str, object]) -> tuple[dict[str, object], object]:
    """The first choice's `message` and its `finish_reason`, defensively.

    A missing or malformed choice yields an empty message rather than raising,
    so one odd response cannot abort a whole eval sweep.
    """
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return {}, None
    choice: dict[str, object] = {str(key): value for key, value in choices[0].items()}
    message = choice.get("message")
    if not isinstance(message, dict):
        return {}, choice.get("finish_reason")
    return {str(key): value for key, value in message.items()}, choice.get("finish_reason")


def _completion_tokens(payload: dict[str, object]) -> int:
    """The response's completion-token count, or 0 when usage is absent."""
    usage = payload.get("usage")
    value = usage.get("completion_tokens") if isinstance(usage, dict) else None
    return value if isinstance(value, int) else 0


def _is_retryable(exc: BaseException) -> bool:
    """Whether a request failure is worth another attempt.

    Read timeouts, dropped connections and 429/5xx are endpoint weather. A 4xx
    other than 429 is a bug in the request and retrying only wastes time.
    """
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or exc.code >= 500
    return isinstance(
        exc,
        (
            urllib.error.URLError,
            TimeoutError,
            socket.timeout,
            http.client.HTTPException,
            ConnectionError,
            json.JSONDecodeError,
        ),
    )


def _post(
    url: str, headers: dict[str, str], body: dict[str, object], timeout: float, attempts: int
) -> tuple[dict[str, object] | None, str | None]:
    """POST a chat-completions body, retrying retryable failures.

    Returns:
        The parsed payload and None, or None and the last error's repr.
    """
    last = "no attempts"
    for attempt in range(attempts):
        request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response), None
        except Exception as exc:  # noqa: BLE001 - classified by _is_retryable
            last = repr(exc)
            if not _is_retryable(exc) or attempt == attempts - 1:
                return None, last
            time.sleep(min(60.0, 5.0 * 2**attempt))
    return None, last


def classify(row: dict[str, object]) -> str:
    """The reason one row is not a plain correct answer.

    Distinguishes the three measurement artifacts from a real mistake, so an
    accuracy can be reported with its floor removed:

    - `error`: the request never returned an answer (timeout, 5xx).
    - `truncated`: the response hit `max_tokens` with no boxed answer.
    - `no_answer`: it stopped normally but never boxed anything.
    - `wrong`: an answer was produced and it disagrees with the gold.
    """
    if row.get("error"):
        return "error"
    if row.get("correct"):
        return "correct"
    if row.get("predicted") is None:
        return "truncated" if row.get("finish_reason") == "length" else "no_answer"
    return "wrong"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="azure", choices=PROVIDERS)
    parser.add_argument("--model", default=None, help="model/deployment override")
    parser.add_argument("--dataset", default="math500", choices=DATASETS)
    parser.add_argument("--n", type=int, default=100, help="0 means the whole set")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="client read timeout in seconds; see DEFAULT_TIMEOUT_S",
    )
    parser.add_argument("--attempts", type=int, default=3, help="tries per problem")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--tail-chars",
        type=int,
        default=1500,
        help="completion tail kept per row, for classifying failures by hand",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    url, model_id, headers = resolve_provider(args.provider, args.model)
    logger.info("provider %s -> %s (model %s)", args.provider, url, model_id)
    problems = load_problems(args.n, args.dataset)
    logger.info(
        "loaded %d %s problems (max_tokens %d, timeout %.0fs, attempts %d)",
        len(problems),
        args.dataset,
        args.max_tokens,
        args.timeout,
        args.attempts,
    )

    # Rows land in a sidecar JSONL as they finish, so a run that dies late still
    # leaves every completed problem on disk.
    stream_path = Path(f"{args.out}.jsonl") if args.out else None
    if stream_path is not None:
        stream_path.write_text("", encoding="utf-8")
    stream_lock = threading.Lock()
    done = [0]

    def run(item: tuple[int, dict[str, str]]) -> dict[str, object]:
        index, row = item
        body: dict[str, object] = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["problem"]},
            ],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
        started = time.monotonic()
        payload, error = _post(url, headers, body, args.timeout, args.attempts)
        elapsed = time.monotonic() - started
        if payload is None:
            result: dict[str, object] = {
                "index": index,
                "gold": row["answer"],
                "predicted": None,
                "correct": False,
                "error": error,
                "completion_tokens": 0,
                "finish_reason": None,
                "answer_source": "none",
                "seconds": round(elapsed, 1),
            }
        else:
            message, finish_reason = _first_message(payload)
            predicted, source, text = _extract_answer(message)
            content = _channel_text(message.get("content"))
            reasoning = "".join(_channel_text(message.get(f)) for f in REASONING_FIELDS)
            result = {
                "index": index,
                "gold": row["answer"],
                "predicted": predicted,
                "correct": answers_match(row["answer"], predicted),
                "completion_tokens": _completion_tokens(payload),
                "finish_reason": finish_reason,
                "answer_source": source,
                "content_chars": len(content),
                "reasoning_chars": len(reasoning),
                "seconds": round(elapsed, 1),
                "text_tail": text[-args.tail_chars :],
                # Both channels are kept, not just whichever one won: a row with
                # no boxed answer can only be classified (model never boxed, vs
                # cap landed mid-`\boxed{`) by reading the answer channel that
                # LOST, which `text_tail` by definition does not show.
                "content_tail": content[-args.tail_chars :],
                "reasoning_tail": reasoning[-args.tail_chars :],
            }
        result["status"] = classify(result)
        with stream_lock:
            done[0] += 1
            logger.info(
                "[%d/%d] problem %d %s (%s tok, %.0fs, src %s)",
                done[0],
                len(problems),
                index,
                result["status"],
                result["completion_tokens"],
                result["seconds"],
                result["answer_source"],
            )
            if stream_path is not None:
                with stream_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result) + "\n")
        return result

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(run, enumerate(problems)))

    total = len(results)
    counts = {name: 0 for name in ("correct", "wrong", "truncated", "no_answer", "error")}
    for row in results:
        counts[str(row["status"])] += 1
    correct = counts["correct"]
    timeouts = sum(1 for r in results if "timed out" in str(r.get("error") or "").lower())
    truncated_any = sum(1 for r in results if r.get("finish_reason") == "length")
    recovered = sum(1 for r in results if r.get("answer_source") in REASONING_FIELDS)
    recovered_correct = sum(
        1 for r in results if r.get("answer_source") in REASONING_FIELDS and r.get("correct")
    )
    out_tokens = sum(int(r["completion_tokens"]) for r in results)
    standard_error = (correct / total * (1 - correct / total) / total) ** 0.5
    # The denominator with the two measurement artifacts removed: everything the
    # model was actually given a fair chance to answer.
    scorable = total - counts["truncated"] - counts["error"]

    logger.info("")
    logger.info("provider/model:   %s / %s", args.provider, model_id)
    logger.info(
        "problems:         %d %s (temperature %.1f, max_tokens %d)",
        total,
        args.dataset,
        args.temperature,
        args.max_tokens,
    )
    logger.info(
        "pass@1 (raw):     %.1f%%  (SE %.1fpp)  %d/%d",
        100 * correct / total,
        100 * standard_error,
        correct,
        total,
    )
    if scorable and scorable != total:
        scorable_se = (correct / scorable * (1 - correct / scorable) / scorable) ** 0.5
        logger.info(
            "pass@1 (scorable):%.1f%%  (SE %.1fpp)  %d/%d, truncation+errors excluded",
            100 * correct / scorable,
            100 * scorable_se,
            correct,
            scorable,
        )
    logger.info(
        "truncated:        %d (%.0f%%) hit max_tokens; %d of those lost the answer",
        truncated_any,
        100 * truncated_any / total,
        counts["truncated"],
    )
    logger.info("request errors:   %d (of which read timeouts: %d)", counts["error"], timeouts)
    logger.info("no answer (stop): %d", counts["no_answer"])
    logger.info("genuine wrong:    %d", counts["wrong"])
    logger.info("recovered from reasoning channel: %d (%d correct)", recovered, recovered_correct)
    if args.provider == "fireworks":
        logger.info(
            "output tokens:    %d  (approx $%.2f against the TRAINING budget)",
            out_tokens,
            out_tokens * FIREWORKS_OUTPUT_USD_PER_MTOK / 1e6,
        )
    else:
        logger.info("output tokens:    %d  (azure, not billed to the training budget)", out_tokens)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
        logger.info("per-problem rows: %s", args.out)


if __name__ == "__main__":
    main()
