"""OpenAI direct provider (GPT 5.5). Reads OPENAI_API_KEY from the environment.

With `ProviderConfig.endpoint` set, the same provider speaks to any OpenAI-compatible server
(vLLM, llama.cpp, a proxy) instead: the endpoint becomes the client's base_url, auth comes from
`WMO_ENDPOINT_API_KEY` (never `OPENAI_API_KEY` — the real key must not leak to arbitrary hosts),
and `temperature` IS forwarded (self-hosted servers accept sampling params; GPT 5.5 rejects them).

Every call path resolves the output-budget parameter name once, from
`ProviderConfig.resolved_chat_max_tokens_field`, and passes it down. GPT-5.x wants
`max_completion_tokens`, but a server outside the built-in catalog can want the classic
`max_tokens` (Tinker's OpenAI-compatible endpoint does) and answers the wrong name with a 400,
so `complete` and `stream` must honor the config exactly as `complete_chat` does.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from wmo.providers import _openai_common
from wmo.providers.base import (
    DEFAULT_MAX_TOKENS,
    ChatRequest,
    ChatResponse,
    Completion,
    EmbeddingResult,
    Message,
    ProviderConfig,
    StreamChunk,
    VerifyResult,
    normalize_chat_temperature,
    verify_via_ping,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from openai import OpenAI


class OpenAIProvider:
    """GPT 5.5 via the OpenAI API."""

    def __init__(self, config: ProviderConfig, *, api_key: str | None = None) -> None:
        self.config = config
        # Trusted explicit credential from get_provider (pool entries with api_key_env). When
        # set, it authenticates even against a config endpoint: the pool file is operator
        # config, so its endpoint+key pairing is trusted, unlike a model bundle's config.toml.
        self._api_key = api_key
        self._client: OpenAI | None = None
        self._forward_temperature = config.resolved_chat_forward_temperature()
        self._max_tokens_field = config.resolved_chat_max_tokens_field()

    def _get_client(self) -> OpenAI:
        # Lazy: don't import the SDK or read the key env vars until first use.
        if self._client is None:
            from openai import OpenAI

            # Bound each request: a reasoning model (GPT-5.5) can leave a connection open with no
            # output, hanging an eval/build indefinitely (Bedrock already caps this via botocore
            # timeouts). `timeout=240` turns a stall into a bounded failure instead of a silent
            # multi-hour hang. Retry ownership is split by CONCERN, not stacked: the SDK's
            # `max_retries=1` owns a single same-endpoint transient retry (one blip on THIS server),
            # while the llm-waterfall chain owns cross-endpoint failover on capacity errors (move to
            # the NEXT backend). They don't compound the way Bedrock's botocore retries did (3 same
            # -model attempts before failover) because one is bounded at 1; and unlike a Bedrock
            # target, a grid's OpenAI/self-hosted target is a SINGLE provider with no chain behind
            # it, so removing this retry would turn any transient 429/5xx into a permanent 0.0 step
            # and bias the comparison against exactly those models. Key + OPENAI_BASE_URL from env.
            if self._api_key is not None:
                # Trusted explicit credential (see __init__): the operator paired this key with
                # this endpoint, so it is sent as-is (base_url=None keeps the SDK default).
                self._client = OpenAI(
                    base_url=self.config.endpoint,
                    api_key=self._api_key,
                    timeout=240.0,
                    max_retries=1,
                )
            elif self.config.endpoint:
                # OpenAI-compatible server. Auth comes from WMO_ENDPOINT_API_KEY; NEVER send
                # the real OPENAI_API_KEY to an arbitrary base_url. Most self-hosted servers
                # ignore auth, but the SDK insists on *a* key, hence the placeholder.
                self._client = OpenAI(
                    base_url=self.config.endpoint,
                    api_key=os.environ.get("WMO_ENDPOINT_API_KEY") or "not-needed",
                    timeout=240.0,
                    max_retries=1,
                )
            else:
                self._client = OpenAI(timeout=240.0, max_retries=1)
        return self._client

    def prepare(self) -> None:
        """Import the SDK and build the client, which resolves the key. No request is sent.

        Satisfies `wmo.providers.base.PreparableProvider`. Building the client is the whole check
        here: `OpenAI()` refuses to construct when no key resolves ("Missing credentials. Please
        pass an `api_key` ... or set the `OPENAI_API_KEY` ... environment variables"), and it opens
        no connection, so the answer costs nothing. The cached client is the one later calls use.

        Raises:
            openai.OpenAIError: No key resolved for this configuration.
        """
        self._get_client()

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion:
        return _openai_common.complete(
            self._get_client().chat.completions,
            self.config.model,
            system,
            messages,
            max_tokens,
            # Self-hosted OpenAI-compatible servers honor sampling params (a policy being
            # trained NEEDS temperature diversity); real OpenAI GPT-5.5 rejects them.
            temperature=temperature if self.config.endpoint else None,
            max_tokens_field=self._max_tokens_field,
        )

    def stream(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Iterator[StreamChunk]:
        """Stream a completion natively (same temperature rule as complete)."""
        return _openai_common.stream(
            self._get_client().chat.completions,
            self.config.model,
            system,
            messages,
            max_tokens,
            temperature=temperature if self.config.endpoint else None,
            max_tokens_field=self._max_tokens_field,
        )

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        """Run a full structured request on the configured OpenAI-compatible backend."""
        request = normalize_chat_temperature(
            request,
            forward_temperature=self._forward_temperature,
        )
        return _openai_common.complete_chat(
            self._get_client().chat.completions,
            self.config.model,
            request,
            max_tokens_field=self._max_tokens_field,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embed_with_usage(texts).vectors

    def embed_with_usage(self, texts: list[str]) -> EmbeddingResult:
        if self.config.embed_model is None:
            raise ValueError("OpenAIProvider.embed requires config.embed_model to be set.")
        return _openai_common.embed_with_usage(
            self._get_client().embeddings, self.config.embed_model, texts, self.config.embed_dim
        )

    def verify(self) -> VerifyResult:
        return verify_via_ping(self)
