"""Teacher-forced scoring against a self-hosted vLLM `/v1/completions` endpoint.

This is the cross-tokenizer teacher's ONLY network surface. `PromptLogprobClient.score`
submits an exact token sequence we already own and returns one logprob per position, so
the returned row indexes one for one into the teacher token ids the chunk aligner
produced. Nothing here tokenizes, renders, or samples.

Five wire facts are load-bearing (each was a real bug or a live probe finding):

1. The prompt goes on the wire as `list[int]`, NEVER as text. vLLM re-tokenizes a text
   prompt server-side with `add_special_tokens` defaulting to True, which prepends
   GLM's prefix/BOS and shifts EVERY position by one against our local offsets. There
   is no error: the response looks perfectly well formed and every span sum is wrong.
2. `/v1/chat/completions` supports neither `echo` nor `prompt_logprobs`, so it cannot
   score a prompt at all. The chat template is applied client-side (see
   `wmh.distill.rendering`) and the rendered ids come here.
3. Position convention: `prompt_logprobs[p]` is the distribution FOR token p (entry 0
   is null because token 0 has no context). That is exactly the Tinker
   `compute_logprobs` convention `wmh.distill.teacher` already uses, so `score`
   returns `len(token_ids)` entries with entry 0 = None and no shifting anywhere.
   A short row is rejected rather than returned: it would silently corrupt every
   downstream chunk sum.
4. The response shape varies by vLLM version: `prompt_logprobs` sits either at the top
   level or under `choices[0]`. Both are read. Each position is a dict keyed by token
   id (a JSON string) whose value carries a `logprob`, and the REALIZED token's entry
   is the one we want, never the argmax.
5. Auth is a Bearer token when `api_key` is set. The repo convention for self-hosted
   endpoints is the `WMH_ENDPOINT_API_KEY` env var (see `wmh.providers.openai`), but
   this module never reads the environment: the caller passes the key in.

Deadlines are owned here rather than through `wmh.distill.deadlines`. That module
bounds Tinker SDK calls (futures and blocking calls with no timeout parameter) and its
knobs are Tinker-named env vars; httpx already bounds an HTTP request natively, so
wrapping it in a watchdog thread would add a second, weaker timer and a misleading
`TinkerDeadlineError`. The default `timeout_s` is 1200s (20 minutes), sized for the
real workload rather than for latency headroom: merged datum length is median ~14.5k
and up to 65.5k teacher tokens, and a distillation step fires many of these
concurrently, so a request waits in the server's queue behind other prefills before
its own runs. Every request is bounded on connect, read, write, and pool, so a wedged
connection raises `PromptLogprobTimeoutError` instead of hanging a run forever.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wmh.providers.base import ProviderKind, VerifyResult

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 1200.0
"""Per-request wall-clock bound; see the module docstring for the sizing."""

DEFAULT_MAX_ATTEMPTS = 30
"""Attempts per `score` call, counting the first; only transient failures retry.

Sized against the MEASURED placeholder-corruption rate, not against ordinary
transient-failure intuition. A sibling lane re-measured 26-45% of echo requests
returning all-zero logprobs, and at 45% bad the retry depth is load-bearing: 6
attempts leaves ~0.8% failure per sequence, which across 32 sequences in a step
is a ~23% chance of losing the ENTIRE step (they hit exactly that, failing at
7178/7178 positions after 6 attempts). 30 attempts puts it near 1e-11. Retries
are cheap here because a placeholder response is refused before any span is
summed, and each retry reconnects so it can reach a different replica.
"""

_RETRY_BASE_DELAY_S = 2.0
_RETRY_MAX_DELAY_S = 30.0

_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
"""Statuses worth another attempt: server-side capacity, restarts, cold boots."""

_COMPLETIONS_PATH = "/v1/completions"

_VERIFY_PROBE_TOKEN_IDS: tuple[int, ...] = (1, 2, 3, 4)
"""A tiny fixed sequence for verify(); small ids are valid in any real vocab."""

_MAX_CANDIDATES_IN_ERROR = 8
"""How many returned candidate ids an error message quotes before eliding."""

_MAX_BODY_CHARS_IN_ERROR = 300


class PromptLogprobError(RuntimeError):
    """A teacher scoring request failed, or its response was unusable.

    Raised for HTTP failures, unparsable bodies, and shape violations (a row of
    the wrong length, a missing realized token). The message names the endpoint
    and the remedy, because a wrong shape here is silent data corruption
    downstream rather than a crash.
    """


PLACEHOLDER_ZERO_FRACTION = 0.9
"""Fraction of exactly-0.0 scored positions above which a row is placeholder junk.

Fireworks' serverless echo has a measured silent-corruption mode: 26 to 45% of
requests return all-zero `token_logprobs` (with a junk one-entry `top_logprobs`
of `{"!": 0.0}`) and HTTP 200, and responses also come back partly real and
partly placeholder, so see `MAX_PLACEHOLDER_RUN` for the mixed case. Nothing in
the response marks it, so an unchecked
consumer trains on zeros: every chunk's teacher sum becomes 0.0 and its
advantage becomes `-student_logprob`, which is not a KL signal at all.

Real logprobs are never all exactly 0.0 over a long span, though INDIVIDUAL
positions legitimately are (a near-certain continuation returns -0.0, observed
on digits inside an equation), so the test is a high fraction rather than any
occurrence. Pinning `x-session-affinity` is NOT the fix: it locked onto a bad
replica 16 times out of 16. Detect and retry instead.
"""

_MIN_POSITIONS_FOR_PLACEHOLDER_CHECK = 8
"""Below this many scored positions, an all-zero row is not distinguishable
from a genuinely deterministic short span, so it is allowed through."""


class PromptLogprobPlaceholderError(PromptLogprobError):
    """The endpoint returned a placeholder all-zero row instead of real logprobs.

    Retryable on purpose: the corruption is per-replica, so another attempt
    usually lands on a healthy one.
    """


MAX_PLACEHOLDER_RUN = 24
"""Longest run of consecutive exactly-0.0 scored positions treated as real.

A whole-response fraction test is not enough: responses come back MIXED, part
real and part placeholder (measured by a sibling lane on a 12k-token response),
and a half-corrupt row passes any 90%-zeros threshold while silently poisoning
half the span. Placeholder corruption appears as a contiguous BLOCK of exact
zeros, whereas genuine near-certain tokens are isolated (observed: -0.0 on
individual digits inside an equation). So a long consecutive run is the
signature to reject on, independently of the overall fraction.
"""


def _reject_placeholder_row(row: list[float | None], url: str) -> None:
    """Raise when a scored row looks like placeholder output, whole or partial.

    Two independent tests, because the corruption has two shapes:
    a high overall fraction of exact zeros (whole-response placeholder), and a
    long contiguous run of them (a mixed response whose fraction stays low).

    Args:
        row: The per-position row, position 0 already None.
        url: Endpoint, for the error message.

    Raises:
        PromptLogprobPlaceholderError: If the row looks placeholder-corrupted.
    """
    scored = [value for value in row[1:] if value is not None]
    if len(scored) < _MIN_POSITIONS_FOR_PLACEHOLDER_CHECK:
        return
    zeros = sum(1 for value in scored if value == 0.0)
    fraction = zeros / len(scored)
    longest_run = 0
    current_run = 0
    for value in scored:
        current_run = current_run + 1 if value == 0.0 else 0
        longest_run = max(longest_run, current_run)
    whole = fraction >= PLACEHOLDER_ZERO_FRACTION
    partial = longest_run >= MAX_PLACEHOLDER_RUN
    if not whole and not partial:
        return
    reason = (
        f"{zeros} of {len(scored)} scored positions are exactly 0.0 ({fraction:.0%})"
        if whole
        else f"a run of {longest_run} consecutive exactly-0.0 positions (a MIXED response, "
        f"only {fraction:.0%} zeros overall, which a fraction test alone would pass)"
    )
    raise PromptLogprobPlaceholderError(
        f"teacher scoring from {url} returned {reason}, which is the known placeholder "
        "response rather than real logprobs. This is a per-replica fault that HTTP 200 "
        "hides, so it is retried on a fresh connection; if it persists the endpoint is "
        "serving corrupt echo responses and the run must stop rather than train on zeros "
        "(do NOT pin session affinity, which locks onto the bad replica)"
    )


class PromptLogprobTimeoutError(PromptLogprobError, TimeoutError):
    """A teacher scoring request blew its wall-clock deadline.

    Subclasses `TimeoutError` and keeps "timed out" in the message so retry
    layers classify it as transient capacity, matching
    `wmh.distill.deadlines.TinkerDeadlineError`'s contract.
    """


class _LogprobEntry(BaseModel):
    """One candidate in a position's `prompt_logprobs` dict (rank/decoded ignored)."""

    model_config = ConfigDict(extra="ignore")

    logprob: float


class _LegacyLogprobs(BaseModel):
    """The legacy `logprobs` object returned by the `echo` dialect.

    Flat arrays, one entry per echoed prompt position plus the throwaway
    sampled token, so `token_logprobs[p]` scores `token_ids[p]`. Verified
    against Fireworks serving GLM-5.2: `token_ids` comes back identical to the
    ids sent, which is what makes span alignment safe.
    """

    model_config = ConfigDict(extra="ignore")

    token_logprobs: list[float | None] = Field(default_factory=list)
    token_ids: list[int] = Field(default_factory=list)


class _Choice(BaseModel):
    """The one completion choice, carrying whichever logprob shape the server returns."""

    model_config = ConfigDict(extra="ignore")

    prompt_logprobs: list[dict[str, _LogprobEntry] | None] | None = None
    logprobs: _LegacyLogprobs | None = None


class _CompletionsResponse(BaseModel):
    """A `/v1/completions` response, tolerant of where the logprobs land."""

    model_config = ConfigDict(extra="ignore")

    prompt_logprobs: list[dict[str, _LogprobEntry] | None] | None = None
    choices: list[_Choice] = Field(default_factory=list)


ScoringDialect = Literal["echo", "prompt_logprobs"]
"""How the server exposes logprobs over caller-supplied prompt tokens.

`echo` is the OpenAI legacy form (`echo: true` plus an INTEGER `logprobs`),
which is what hosted providers expose; Fireworks' GLM-5.2 needs it, and it
additionally echoes `token_ids` so the round trip can be verified.
`prompt_logprobs` is the vLLM-native form, available when we run the server
ourselves. The two differ in both request and response shape, so the dialect is
explicit rather than sniffed: a silent guess would send a body the server
accepts while returning nothing useful.
"""


class _EchoRequest(BaseModel):
    """Scoring body for the `echo` dialect.

    `logprobs` MUST be an integer. Sending `true` selects the newer
    OpenAI-style `content` array, which carries no prompt positions at all, and
    the server returns 200 either way.
    """

    model_config = ConfigDict(extra="forbid")

    model: str
    prompt: list[int]
    max_tokens: int = 1
    echo: bool = True
    logprobs: int = 1
    temperature: float = 0.0


class _CompletionsRequest(BaseModel):
    """Scoring body for the vLLM-native `prompt_logprobs` dialect."""

    model_config = ConfigDict(extra="forbid")

    model: str
    prompt: list[int]
    max_tokens: int = 1
    prompt_logprobs: int = 0
    temperature: float = 0.0


def _completions_url(endpoint: str) -> str:
    """The `/v1/completions` URL for a configured endpoint.

    Accepts either a bare server root (`https://host`) or a root that already
    carries the OpenAI `/v1` prefix (the form `wmh providers` stores for
    OpenAI-compatible servers), so a caller cannot accidentally produce
    `/v1/v1/completions`.

    Args:
        endpoint: The teacher server base URL.

    Returns:
        The absolute URL to POST scoring requests to.

    Raises:
        ValueError: If `endpoint` is blank.
    """
    base = endpoint.strip().rstrip("/")
    if not base:
        raise ValueError(
            "PromptLogprobClient needs a teacher endpoint URL, for example "
            "'https://my-vllm-host' or 'https://my-vllm-host/v1'; got an empty string"
        )
    if base.endswith("/v1"):
        return base + "/completions"
    return base + _COMPLETIONS_PATH


class PromptLogprobClient:
    """Scores exact teacher token ids on a vLLM `/v1/completions` endpoint.

    One client is safe to share across threads: httpx connection pooling and the
    usage counter are both synchronized, so a scoring pool can fan out over
    datums against a single client (that is how the teacher is driven in a step).

    Example:
        >>> client = PromptLogprobClient("https://vllm-host", "zai-org/GLM-5.2")
        >>> row = client.score(teacher_token_ids)  # doctest: +SKIP
        >>> row[0] is None  # position 0 has no context  # doctest: +SKIP
        True
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        api_key: str | None = None,
        dialect: ScoringDialect = "echo",
        timeout_s: float = DEFAULT_TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Build a client bound to one endpoint and one served model.

        Args:
            endpoint: Teacher server base URL, with or without a `/v1` suffix.
            model: The model id the server serves, sent as the request's `model`.
            api_key: Bearer token for the endpoint. The repo convention is to pass
                `WMH_ENDPOINT_API_KEY`; this module never reads the environment
                itself, so the real provider keys cannot leak to an arbitrary host.
                None (the norm for a private vLLM host) sends no auth header.
            dialect: Which logprob surface the server exposes (see
                `ScoringDialect`). Defaults to `echo`, what hosted providers
                offer; use `prompt_logprobs` against a vLLM server we run.
            timeout_s: Per-request wall-clock bound in seconds, applied to connect,
                read, write, and pool waits. Defaults to `DEFAULT_TIMEOUT_S`
                (1200s), sized for a 65k-token prefill queued behind other
                requests. Retries multiply the worst case by `max_attempts`.
            transport: httpx transport override. Tests pass an
                `httpx.MockTransport` so no request ever leaves the process.
            max_attempts: Total attempts per `score` call. Only transient failures
                (timeouts, transport errors, retryable statuses) consume attempts.
            sleep: Backoff sleeper, injectable so tests do not wait.

        Raises:
            ValueError: If the endpoint is blank, `timeout_s` is not a positive
                finite number, or `max_attempts` is below 1.
        """
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError(
                f"timeout_s must be a positive finite number of seconds, got {timeout_s!r}; "
                f"use the default ({DEFAULT_TIMEOUT_S:g}) unless the endpoint is known to be fast"
            )
        if max_attempts < 1:
            raise ValueError(
                f"max_attempts must be at least 1, got {max_attempts}; pass 1 to disable retries"
            )
        self._url = _completions_url(endpoint)
        self._model = model
        self._dialect: ScoringDialect = dialect
        self._timeout_s = timeout_s
        self._max_attempts = max_attempts
        self._sleep = sleep
        self._usage_tokens = 0
        self._placeholder_responses = 0
        self._usage_lock = threading.Lock()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._headers = headers
        self._transport = transport
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_s),
            transport=transport,
            headers=headers,
        )

    @property
    def url(self) -> str:
        """The absolute scoring URL this client posts to."""
        return self._url

    @property
    def model(self) -> str:
        """The served model id sent with every request."""
        return self._model

    def score(self, token_ids: list[int]) -> list[float | None]:
        """Teacher logprobs for an exact token sequence, one entry per position.

        Args:
            token_ids: The teacher's own token ids, in the teacher's vocabulary.
                They are sent verbatim as integers, so the returned row aligns
                index for index with this list.

        Returns:
            A list of `len(token_ids)` entries. Entry 0 is always None (token 0
            has no context); entry p is the teacher's logprob of `token_ids[p]`
            given `token_ids[:p]`.

        Raises:
            ValueError: If `token_ids` is empty.
            PromptLogprobTimeoutError: If every attempt blew the deadline.
            PromptLogprobError: On a non-retryable HTTP status, an exhausted
                retry budget, an unparsable body, a row length that does not
                match the prompt, or a position whose dict lacks the realized
                token.
        """
        if not token_ids:
            raise ValueError(
                "score() needs at least one token id; an empty prompt has nothing to score "
                "(filter empty spans out before scoring)"
            )
        return self._score(token_ids, count_usage=True)

    def verify(self) -> VerifyResult:
        """One tiny scoring probe, reporting failure as `ok=False` instead of raising.

        Mirrors `wmh.providers.base.verify_via_ping` and `TinkerTeacher.verify`, so
        preflight can report every misconfigured backend at once. The probe's tokens
        are excluded from `usage()`.

        Returns:
            `ok=True` when the endpoint answered with a well-formed row, otherwise
            `ok=False` with the failure text in `detail`.
        """
        try:
            self._score(list(_VERIFY_PROBE_TOKEN_IDS), count_usage=False)
        except Exception as exc:  # noqa: BLE001 - verify reports failure, never raises
            return VerifyResult(
                ok=False, kind=ProviderKind.OPENAI, model=self._model, detail=str(exc)
            )
        return VerifyResult(ok=True, kind=ProviderKind.OPENAI, model=self._model, detail=self._url)

    def usage(self) -> int:
        """Cumulative teacher tokens submitted for scoring (verify probes excluded).

        Counts every dispatched attempt, not only successful ones: a request that
        timed out or died mid-response has usually already run its prefill on the
        server, so the work is real. This matches `TinkerTeacher.usage`'s
        "submitted, not billed-on-success" contract and feeds the same
        teacher_prefill meter.
        """
        with self._usage_lock:
            return self._usage_tokens

    def close(self) -> None:
        """Close the underlying connection pool."""
        self._client.close()

    def __enter__(self) -> PromptLogprobClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _score(self, token_ids: list[int], *, count_usage: bool) -> list[float | None]:
        """Score once, retrying the placeholder-response fault.

        The placeholder check can only run after the body is parsed, so it gets
        its own attempt loop rather than riding `_post`'s: the request succeeded
        at the HTTP level and only its CONTENT is unusable.
        """
        if self._dialect == "echo":
            body = _EchoRequest(model=self._model, prompt=token_ids).model_dump()
        else:
            body = _CompletionsRequest(model=self._model, prompt=token_ids).model_dump()
        last: PromptLogprobPlaceholderError | None = None
        for attempt in range(1, self._max_attempts + 1):
            response = self._post(body, token_count=len(token_ids) if count_usage else 0)
            try:
                if self._dialect == "echo":
                    return self._echo_row(response, token_ids)
                rows = self._prompt_logprob_rows(response, expected=len(token_ids))
                return self._realized_row(rows, token_ids)
            except PromptLogprobPlaceholderError as exc:
                last = exc
                self._placeholder_responses += 1
                if attempt < self._max_attempts:
                    # A pooled keep-alive connection is implicit replica
                    # affinity: every retry down the same socket reaches the
                    # same (bad) replica, which is why 4 of 4 attempts came
                    # back identically corrupt in the first live run. Drop the
                    # pool so the retry reconnects and can land elsewhere.
                    self._reconnect()
                    logger.warning(
                        "teacher returned a placeholder all-zero row (attempt %d/%d); "
                        "reconnecting so the retry can reach a different replica",
                        attempt,
                        self._max_attempts,
                    )
        assert last is not None  # noqa: S101 - only reachable after a placeholder raise
        raise last

    def _reconnect(self) -> None:
        """Rebuild the connection pool so the next request opens a fresh socket.

        The transport is preserved when one was injected (tests pass a
        `MockTransport`), so this is a no-op there beyond the pool swap.
        """
        with self._usage_lock:
            old = self._client
            self._client = httpx.Client(
                timeout=httpx.Timeout(self._timeout_s),
                transport=self._transport,
                headers=self._headers,
            )
        old.close()

    def placeholder_responses(self) -> int:
        """How many placeholder all-zero rows this client has seen and retried.

        Worth logging per step: a rising rate means the endpoint is degrading,
        and the measured baseline is 26 to 38% of requests.
        """
        return self._placeholder_responses

    def _echo_row(self, response: _CompletionsResponse, token_ids: list[int]) -> list[float | None]:
        """Per-position logprobs from the legacy `echo` response shape.

        The response echoes the prompt plus the one throwaway sampled token, so
        it is one entry LONGER than the prompt and is truncated back. Two checks
        are not optional:

        - `token_ids` must equal the ids we sent. The server is free to
          re-tokenize or prepend a BOS; if it did, every span offset the chunk
          aligner computed would be shifted and every span sum silently wrong.
        - position 0 is forced to None. Fireworks returns `0.0` there rather
          than null, and a logprob of 0.0 means probability 1, so summing it
          would inflate any span that started at position 0.
        """
        if not response.choices or response.choices[0].logprobs is None:
            raise PromptLogprobError(
                f"teacher scoring response from {self._url} carried no logprobs object; the "
                "echo dialect needs `echo: true` with an INTEGER `logprobs` (sending `true` "
                "returns a content array with no prompt positions). Confirm the served model "
                f"{self._model!r} supports /v1/completions with echo"
            )
        legacy = response.choices[0].logprobs
        expected = len(token_ids)
        returned_ids = legacy.token_ids[:expected]
        if returned_ids != token_ids:
            first = next(
                (
                    index
                    for index, (sent, got) in enumerate(zip(token_ids, returned_ids, strict=False))
                    if sent != got
                ),
                min(len(token_ids), len(returned_ids)),
            )
            raise PromptLogprobError(
                f"teacher scoring echoed {len(returned_ids)} token id(s) that do not match the "
                f"{expected} sent (first difference at position {first}); the server "
                "re-tokenized the prompt instead of scoring the exact ids, so every chunk span "
                "offset would be wrong. Send the prompt as a token-id array (not text) and "
                "confirm the endpoint's tokenizer matches the one the chunk plan was built with"
            )
        values = legacy.token_logprobs[:expected]
        if len(values) != expected:
            raise PromptLogprobError(
                f"teacher scoring returned {len(values)} logprob(s) for a {expected}-token "
                "prompt; the echo response must carry one entry per prompt position"
            )
        row: list[float | None] = list(values)
        # Position 0 has no context; Fireworks reports 0.0 rather than null.
        row[0] = None
        _reject_placeholder_row(row, self._url)
        return row

    def _post(self, body: dict[str, object], *, token_count: int) -> _CompletionsResponse:
        """POST one scoring request, retrying only transient failures."""
        last_error: PromptLogprobError | None = None
        for attempt in range(1, self._max_attempts + 1):
            if token_count:
                with self._usage_lock:
                    self._usage_tokens += token_count
            try:
                response = self._client.post(self._url, json=body)
            except httpx.TimeoutException as exc:
                last_error = PromptLogprobTimeoutError(
                    f"teacher scoring timed out after {self._timeout_s:g}s against "
                    f"{self._url} (attempt {attempt}/{self._max_attempts}): {exc}. Raise "
                    "timeout_s, lower the number of concurrent scoring calls, or check the "
                    "endpoint is up and not stuck in a cold boot"
                )
            except httpx.TransportError as exc:
                last_error = PromptLogprobError(
                    f"teacher scoring could not reach {self._url} "
                    f"(attempt {attempt}/{self._max_attempts}): {exc!r}. Check the endpoint URL "
                    "and that the vLLM server is running and reachable from this host"
                )
            else:
                if response.status_code < 400:
                    return self._parse(response)
                error = self._status_error(response, attempt)
                if response.status_code not in _RETRYABLE_STATUS:
                    raise error
                last_error = error
            if attempt < self._max_attempts:
                delay = min(_RETRY_BASE_DELAY_S * 2 ** (attempt - 1), _RETRY_MAX_DELAY_S)
                logger.warning(
                    "teacher scoring attempt %d/%d failed (%s); retrying in %.0fs",
                    attempt,
                    self._max_attempts,
                    last_error,
                    delay,
                )
                self._sleep(delay)
        assert last_error is not None  # noqa: S101 - the loop runs at least once
        raise last_error

    def _status_error(self, response: httpx.Response, attempt: int) -> PromptLogprobError:
        """A typed error for a failing HTTP status, with a status-specific remedy."""
        body = response.text[:_MAX_BODY_CHARS_IN_ERROR]
        if response.status_code in (401, 403):
            remedy = (
                "the endpoint rejected the credentials: pass the api_key this server expects "
                "(the repo convention is the WMH_ENDPOINT_API_KEY env var, read by the caller)"
            )
        elif response.status_code == 404:
            remedy = (
                f"no such route or model: check the base URL and that the server serves model "
                f"{self._model!r} (GET /v1/models lists it)"
            )
        elif response.status_code in _RETRYABLE_STATUS:
            remedy = (
                "the server failed transiently (capacity, restart, or cold boot); retries are "
                "exhausted, so lower scoring concurrency or check the server logs"
            )
        else:
            remedy = (
                "the server rejected the request: confirm this vLLM build supports "
                "prompt_logprobs on /v1/completions and that no token id exceeds its vocab"
            )
        return PromptLogprobError(
            f"teacher scoring got HTTP {response.status_code} from {self._url} "
            f"(attempt {attempt}/{self._max_attempts}): {body!r}. {remedy}"
        )

    def _parse(self, response: httpx.Response) -> _CompletionsResponse:
        """Parse a 2xx body, turning malformed JSON into a typed error."""
        try:
            payload = response.json()
        except ValueError as exc:
            raise PromptLogprobError(
                f"teacher scoring got a non-JSON response from {self._url}: "
                f"{response.text[:_MAX_BODY_CHARS_IN_ERROR]!r}. Check the URL points at a vLLM "
                "OpenAI server and not at a proxy or web page"
            ) from exc
        try:
            return _CompletionsResponse.model_validate(payload)
        except ValidationError as exc:
            raise PromptLogprobError(
                f"teacher scoring could not read the response from {self._url}: {exc}. Each "
                "prompt_logprobs position must be null or a dict of token id to an object with "
                "a 'logprob'; check the vLLM version's response format"
            ) from exc

    def _prompt_logprob_rows(
        self, response: _CompletionsResponse, *, expected: int
    ) -> list[dict[str, _LogprobEntry] | None]:
        """The per-position candidate dicts, from either shape, length-validated."""
        rows = response.prompt_logprobs
        if rows is None and response.choices:
            rows = response.choices[0].prompt_logprobs
        if rows is None:
            raise PromptLogprobError(
                f"teacher scoring response from {self._url} carried no prompt_logprobs (neither "
                "at the top level nor under choices[0]). The request must go to /v1/completions "
                "with prompt_logprobs set: /v1/chat/completions supports neither prompt_logprobs "
                "nor echo, and older servers may not support it at all"
            )
        if len(rows) != expected:
            raise PromptLogprobError(
                f"teacher scoring returned {len(rows)} prompt_logprobs entries for a "
                f"{expected}-token prompt at {self._url} (model {self._model}). Every position "
                "must be scored, since a short or long row silently corrupts every downstream "
                "span sum. Send the prompt as a list[int] (a text prompt is re-tokenized "
                "server-side, which shifts every position), and confirm the server's "
                f"max_model_len covers {expected} tokens"
            )
        return rows

    def _realized_row(
        self,
        rows: list[dict[str, _LogprobEntry] | None],
        token_ids: list[int],
    ) -> list[float | None]:
        """Pull each position's REALIZED token logprob, never the argmax."""
        row: list[float | None] = [None]
        for index in range(1, len(token_ids)):
            candidates = rows[index]
            token_id = token_ids[index]
            entry = candidates.get(str(token_id)) if candidates else None
            if entry is None:
                raise PromptLogprobError(
                    f"teacher scoring returned no logprob for the realized token {token_id} at "
                    f"position {index} of {len(token_ids)} from {self._url} "
                    f"(candidates: {_describe_candidates(candidates)}). The realized token is "
                    "always included when the prompt is sent as token ids with prompt_logprobs "
                    f"set, so this means the ids are not in {self._model}'s vocabulary or the "
                    "prompt was re-tokenized server-side"
                )
            row.append(entry.logprob)
        logger.debug(
            "teacher scored %d position(s) at %s, %d tokens submitted so far",
            len(token_ids),
            self._url,
            self.usage(),
        )
        return row


def _describe_candidates(candidates: dict[str, _LogprobEntry] | None) -> str:
    """A short, deterministic rendering of a position's returned candidate ids."""
    if not candidates:
        return "none (the position was null or empty)"
    keys = sorted(candidates)
    shown = ", ".join(keys[:_MAX_CANDIDATES_IN_ERROR])
    if len(keys) > _MAX_CANDIDATES_IN_ERROR:
        return f"{len(keys)} returned, first ids {shown}, ..."
    return shown
