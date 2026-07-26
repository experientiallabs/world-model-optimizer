"""Tests for teacher-forced scoring over a `/v1/completions` endpoint.

Every test runs through `httpx.MockTransport`, so no request leaves the
process. The recorded shapes come from live probes: the `echo` bodies match
Fireworks serving GLM-5.2 (including its `0.0` at position 0), and the
`prompt_logprobs` bodies match vLLM's native dict-per-position form.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from wmh.distill.xtoken.prompt_logprobs import (
    DEFAULT_MAX_ATTEMPTS,
    PLACEHOLDER_ZERO_FRACTION,
    PromptLogprobClient,
    PromptLogprobError,
    PromptLogprobTimeoutError,
    ScoringDialect,
)

ENDPOINT = "https://teacher.invalid/v1"
MODEL = "accounts/fireworks/models/glm-5p2"


def _echo_body(token_ids: list[int], logprobs: list[float | None]) -> dict[str, object]:
    """A legacy echo response: prompt positions plus one throwaway sampled token."""
    return {
        "choices": [
            {
                "index": 0,
                "text": "",
                "logprobs": {
                    "tokens": [f"t{i}" for i in token_ids] + ["extra"],
                    "token_logprobs": [*logprobs, -1.25],
                    "token_ids": [*token_ids, 999],
                },
            }
        ],
        "usage": {"prompt_tokens": len(token_ids)},
    }


Handler = Callable[[httpx.Request], httpx.Response]
"""The MockTransport handler signature every test supplies."""


def _client(
    handler: Handler,
    *,
    dialect: ScoringDialect = "echo",
    api_key: str | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> PromptLogprobClient:
    """A client wired to a mock transport, with sleeping disabled."""
    return PromptLogprobClient(
        ENDPOINT,
        MODEL,
        dialect=dialect,
        api_key=api_key,
        max_attempts=max_attempts,
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )


def test_echo_dialect_returns_one_entry_per_position() -> None:
    ids = [785, 6722, 315, 9621]
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_echo_body(ids, [0.0, -12.7, -4.2, -5.4]))

    with _client(handler) as client:
        row = client.score(ids)

    assert len(row) == len(ids)
    # Position 0 is discarded: Fireworks reports 0.0 there, which would read as
    # probability 1 and inflate any span starting at 0.
    assert row[0] is None
    assert row[1:] == pytest.approx([-12.7, -4.2, -5.4])
    # The request must carry the ids verbatim, an INTEGER logprobs, and echo.
    body = seen[0]
    assert body["prompt"] == ids
    assert body["echo"] is True
    assert body["logprobs"] == 1
    assert not isinstance(body["logprobs"], bool)
    assert body["model"] == MODEL


def test_echo_dialect_rejects_retokenized_prompt() -> None:
    # The server echoing different ids means it re-tokenized our prompt, so every
    # chunk span offset would be shifted. That must be an error, never a silent row.
    ids = [785, 6722, 315]

    def handler(request: httpx.Request) -> httpx.Response:
        body = _echo_body([1, 785, 6722], [None, -1.0, -2.0])
        return httpx.Response(200, json=body)

    with _client(handler) as client, pytest.raises(PromptLogprobError, match="re-tokenized"):
        client.score(ids)


def test_echo_dialect_rejects_short_row() -> None:
    ids = [785, 6722, 315, 9621]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "logprobs": {
                            "token_logprobs": [0.0, -1.0],
                            "token_ids": ids,
                        }
                    }
                ]
            },
        )

    with _client(handler) as client, pytest.raises(PromptLogprobError, match="one entry per"):
        client.score(ids)


def test_echo_dialect_rejects_missing_logprobs_object() -> None:
    # This is what a boolean `logprobs: true` produces: 200 with no prompt positions.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"index": 0, "text": ""}]})

    with _client(handler) as client, pytest.raises(PromptLogprobError, match="INTEGER"):
        client.score([785, 6722])


def test_prompt_logprobs_dialect_extracts_the_realized_token() -> None:
    # vLLM native shape: one dict per position keyed by token id. The realized
    # token's entry is wanted, never the highest-scoring candidate.
    ids = [11, 22, 33]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["prompt_logprobs"] == 0
        assert "echo" not in body
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "prompt_logprobs": [
                            None,
                            {"22": {"logprob": -0.5}, "99": {"logprob": -0.1}},
                            {"33": {"logprob": -2.5}, "98": {"logprob": -0.2}},
                        ]
                    }
                ]
            },
        )

    with _client(handler, dialect="prompt_logprobs") as client:
        row = client.score(ids)

    assert row[0] is None
    assert row[1] == pytest.approx(-0.5)
    assert row[2] == pytest.approx(-2.5)


def test_auth_header_present_only_when_key_given() -> None:
    captured: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.headers.get("authorization"))
        return httpx.Response(200, json=_echo_body([1, 2], [0.0, -1.0]))

    with _client(handler, api_key="secret-key") as client:
        client.score([1, 2])
    with _client(handler) as client:
        client.score([1, 2])

    assert captured[0] == "Bearer secret-key"
    assert captured[1] is None


def test_usage_accumulates_submitted_teacher_tokens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_echo_body([1, 2, 3], [0.0, -1.0, -2.0]))

    with _client(handler) as client:
        assert client.usage() == 0
        client.score([1, 2, 3])
        client.score([1, 2, 3])
        assert client.usage() == 6


def test_verify_probe_is_excluded_from_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        ids = json.loads(request.content)["prompt"]
        return httpx.Response(200, json=_echo_body(ids, [0.0] + [-1.0] * (len(ids) - 1)))

    with _client(handler) as client:
        result = client.verify()
        assert result.ok is True
        assert client.usage() == 0


def test_verify_reports_failure_instead_of_raising() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with _client(handler, max_attempts=1) as client:
        result = client.verify()
    assert result.ok is False
    assert "500" in (result.detail or "")


def test_retryable_status_is_retried_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="cold boot")
        return httpx.Response(200, json=_echo_body([1, 2], [0.0, -1.0]))

    with _client(handler, max_attempts=3) as client:
        row = client.score([1, 2])
    assert calls["n"] == 2
    assert row[1] == pytest.approx(-1.0)


def test_non_retryable_status_fails_immediately() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404, text="no such model")

    with _client(handler, max_attempts=3) as client, pytest.raises(PromptLogprobError, match="404"):
        client.score([1, 2])
    assert calls["n"] == 1


def test_timeout_raises_a_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    with (
        _client(handler, max_attempts=2) as client,
        pytest.raises(PromptLogprobTimeoutError, match="timed out"),
    ):
        client.score([1, 2])


def test_empty_prompt_is_a_caller_bug() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never called
        return httpx.Response(200, json=_echo_body([1], [0.0]))

    with _client(handler) as client, pytest.raises(ValueError, match="at least one token id"):
        client.score([])


def test_endpoint_without_v1_suffix_gets_one() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=_echo_body([1, 2], [0.0, -1.0]))

    client = PromptLogprobClient(
        "https://teacher.invalid",
        MODEL,
        transport=httpx.MockTransport(handler),
        sleep=lambda _s: None,
    )
    with client:
        client.score([1, 2])
    assert seen[0] == "https://teacher.invalid/v1/completions"
    assert "/v1/v1/" not in seen[0]


def test_blank_endpoint_is_rejected() -> None:
    with pytest.raises(ValueError, match="endpoint URL"):
        PromptLogprobClient("   ", MODEL)


def test_placeholder_all_zero_row_is_retried_then_succeeds() -> None:
    # Fireworks' measured silent-corruption mode: HTTP 200 with every scored
    # position exactly 0.0. Training on that would replace the KL signal with
    # -student_logprob, so it must never reach the caller.
    ids = list(range(100, 116))
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json=_echo_body(ids, [0.0] * len(ids)))
        return httpx.Response(200, json=_echo_body(ids, [0.0] + [-1.5] * (len(ids) - 1)))

    with _client(handler, max_attempts=3) as client:
        row = client.score(ids)
    assert calls["n"] == 2
    assert row[1] == pytest.approx(-1.5)
    assert client.placeholder_responses() == 1


def test_persistent_placeholder_rows_raise_rather_than_return_zeros() -> None:
    ids = list(range(100, 116))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_echo_body(ids, [0.0] * len(ids)))

    with (
        _client(handler, max_attempts=2) as client,
        pytest.raises(PromptLogprobError, match="placeholder"),
    ):
        client.score(ids)


def test_genuinely_deterministic_positions_are_not_flagged() -> None:
    # A near-certain continuation legitimately returns -0.0 (observed on digits
    # inside an equation), so isolated zeros must pass.
    ids = list(range(100, 116))
    values: list[float | None] = [0.0, -0.0, -0.0, -3.2] + [-1.1] * (len(ids) - 4)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_echo_body(ids, values))

    with _client(handler) as client:
        row = client.score(ids)
    assert row[3] == pytest.approx(-3.2)
    assert client.placeholder_responses() == 0


def test_short_spans_are_exempt_from_the_placeholder_check() -> None:
    # Too few positions to distinguish placeholder from determinism.
    ids = [1, 2, 3]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_echo_body(ids, [0.0, 0.0, 0.0]))

    with _client(handler) as client:
        row = client.score(ids)
    assert row[1] == 0.0


def test_mixed_placeholder_response_is_rejected() -> None:
    """A response that is part real and part placeholder must not pass.

    Measured by a sibling lane: responses come back MIXED, so a whole-response
    fraction test alone lets a half-corrupt row through and poisons half the
    span. Here only 40% of positions are zero, well under the 90% fraction
    threshold, but they form one contiguous block.
    """
    ids = list(range(100, 160))
    real = [-1.2, -0.7, -3.1, -2.2]
    values: list[float | None] = [0.0]
    while len(values) < 36:
        values.append(real[len(values) % len(real)])
    values.extend([0.0] * (len(ids) - len(values)))
    zero_fraction = sum(1 for v in values[1:] if v == 0.0) / (len(ids) - 1)
    assert zero_fraction < PLACEHOLDER_ZERO_FRACTION

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json=_echo_body(ids, values))
        return httpx.Response(200, json=_echo_body(ids, [0.0] + [-1.1] * (len(ids) - 1)))

    with _client(handler, max_attempts=3) as client:
        row = client.score(ids)
    assert calls["n"] == 2
    assert client.placeholder_responses() == 1
    assert row[-1] == pytest.approx(-1.1)


def test_scattered_legitimate_zeros_still_pass() -> None:
    """Isolated exact zeros are real: near-certain tokens return -0.0."""
    ids = list(range(100, 160))
    values: list[float | None] = [0.0]
    for index in range(1, len(ids)):
        values.append(0.0 if index % 7 == 0 else -1.4)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_echo_body(ids, values))

    with _client(handler) as client:
        row = client.score(ids)
    assert client.placeholder_responses() == 0
    assert row[1] == pytest.approx(-1.4)
