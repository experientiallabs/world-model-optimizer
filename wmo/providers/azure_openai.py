"""Azure OpenAI provider (GPT 5.5).

The real AZURE_OPENAI_API_KEY is only ever sent to the trusted, operator-supplied
AZURE_OPENAI_ENDPOINT. A config-controlled endpoint (ProviderConfig.endpoint, which can arrive
in an untrusted model bundle's config.toml) is treated as an untrusted host: auth for it comes
from WMO_ENDPOINT_API_KEY, never the real key, mirroring OpenAIProvider. Deployment name and
api_version come from ProviderConfig.deployment / ProviderConfig.api_version.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

from wmo.providers import _openai_common, _responses_common
from wmo.providers.base import (
    DEFAULT_MAX_TOKENS,
    PING_MAX_TOKENS,
    ChatRequest,
    ChatResponse,
    Completion,
    EmbeddingResult,
    Message,
    ProviderConfig,
    StreamChunk,
    TokenUsage,
    VerifyResult,
    normalize_chat_temperature,
    verify_via_ping,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from openai import AzureOpenAI, OpenAI


# One read-only verify probe for reasoning configs, sent through the same structured Responses
# route real calls use. PING_MAX_TOKENS leaves room for a reasoning prelude; base's
# reachable-error forgiveness covers models that still exhaust it.
_REASONING_PING = ChatRequest.model_validate(
    {
        "messages": [{"role": "user", "content": "ping"}],
        "max_completion_tokens": PING_MAX_TOKENS,
    }
)


class AzureOpenAIProvider:
    """GPT 5.5 via an Azure OpenAI deployment."""

    def __init__(self, config: ProviderConfig, *, api_key: str | None = None) -> None:
        self.config = config
        # Trusted explicit credential from get_provider (pool entries with api_key_env). When
        # set, it authenticates the config endpoint directly: the pool file is operator config,
        # so its endpoint+key pairing is trusted and the untrusted-endpoint downgrade below
        # does not apply. Multi-account pools (several Azure resources, one key each) need this.
        self._api_key = api_key
        self._client: AzureOpenAI | None = None
        self._responses_client: OpenAI | None = None
        self._forward_temperature = config.resolved_chat_forward_temperature()

    def _resolved_endpoint(self) -> tuple[str, bool]:
        """Return the endpoint and whether it came from untrusted model configuration."""
        env_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        endpoint = self.config.endpoint or env_endpoint
        if not endpoint:
            raise ValueError(
                "AzureOpenAIProvider needs an endpoint: set config.endpoint or "
                "AZURE_OPENAI_ENDPOINT."
            )
        # Compare canonically so a trailing slash or host-casing difference between the config
        # value and the trusted env endpoint doesn't misclassify the same Azure resource as an
        # untrusted host (which would strip the real key and break the call).
        is_config_endpoint = self.config.endpoint is not None and not _same_endpoint(
            self.config.endpoint, env_endpoint
        )
        return endpoint, is_config_endpoint

    def _get_client(self) -> AzureOpenAI:
        # Lazy: construct on first use. api_version must be supplied by config; the endpoint and
        # api_key are resolved with a trust check (see below), never blindly from the environment.
        if self._client is None:
            # Validate config before reaching for the SDK, so a config error doesn't depend on the
            # optional `openai` extra being installed.
            if self.config.api_version is None:
                raise ValueError("AzureOpenAIProvider requires config.api_version to be set.")

            endpoint, is_config_endpoint = self._resolved_endpoint()

            from openai import AzureOpenAI

            if self._api_key is not None:
                # Trusted explicit credential (see __init__): authenticate this endpoint with
                # exactly the operator-paired key, bypassing the untrusted-endpoint downgrade.
                self._client = AzureOpenAI(
                    api_version=self.config.api_version,
                    azure_endpoint=endpoint,
                    api_key=self._api_key,
                )
            elif is_config_endpoint:
                # A config-controlled endpoint (config.toml can come from an untrusted model
                # bundle) is an untrusted host. NEVER let the SDK fall back to the real
                # AZURE_OPENAI_API_KEY for it: auth comes from WMO_ENDPOINT_API_KEY, mirroring
                # OpenAIProvider. The SDK insists on *a* key, hence the placeholder.
                self._client = AzureOpenAI(
                    api_version=self.config.api_version,
                    azure_endpoint=endpoint,
                    api_key=os.environ.get("WMO_ENDPOINT_API_KEY") or "not-needed",
                )
            else:
                # Trusted endpoint (operator-supplied AZURE_OPENAI_ENDPOINT): the SDK reads the
                # real AZURE_OPENAI_API_KEY from the environment.
                self._client = AzureOpenAI(
                    api_version=self.config.api_version,
                    azure_endpoint=endpoint,
                )
        return self._client

    def _get_responses_client(self) -> OpenAI:
        """Create the Azure v1 client used by configured structured reasoning calls."""
        if self._responses_client is None:
            endpoint, is_config_endpoint = self._resolved_endpoint()
            if self._api_key is not None:
                # Same trusted explicit credential as _get_client.
                api_key: str | None = self._api_key
            elif is_config_endpoint:
                api_key = os.environ.get("WMO_ENDPOINT_API_KEY") or "not-needed"
            else:
                api_key = os.environ.get("AZURE_OPENAI_API_KEY")
                if not api_key:
                    raise ValueError(
                        "AzureOpenAIProvider needs AZURE_OPENAI_API_KEY for the v1 Responses API."
                    )

            from openai import OpenAI

            self._responses_client = OpenAI(
                api_key=api_key,
                base_url=_responses_base_url(endpoint),
                timeout=240.0,
                # A lost SDK response may already be billed. Keep one WMO proposer turn equal to
                # one provider dispatch so usage and the operator's hard budget stay observable.
                max_retries=0,
            )
        return self._responses_client

    def prepare(self) -> None:
        """Resolve deployment, api_version, endpoint, and key by building this config's client.

        Satisfies `wmo.providers.base.PreparableProvider`. Branches exactly like `verify`: a
        `reasoning_effort` config dispatches through the Azure v1 Responses client, everything else
        through the api-versioned chat client, so the thing that gets prepared is the thing the
        request will use. All four failures are local, which is what makes this free: `_deployment`
        rejects a config with no deployment name, `_get_client` rejects a missing api_version and a
        missing endpoint before it reaches for the SDK, and `AzureOpenAI(...)` itself refuses to
        construct without a resolvable key. No connection is opened.

        Raises:
            ValueError: The config is missing a deployment, an api_version, or an endpoint.
            openai.OpenAIError: No key resolved for this endpoint.
        """
        self._deployment()
        if self.config.reasoning_effort is not None:
            self._get_responses_client()
            return
        self._get_client()

    def _deployment(self) -> str:
        # On Azure, the `model` arg to the API is the deployment name, not the base model id.
        if self.config.deployment is None:
            raise ValueError("AzureOpenAIProvider requires config.deployment to be set.")
        return self.config.deployment

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion:
        if self.config.reasoning_effort is not None:
            # Text completion consumers (proposer, judges, the baseline loop) must use the same
            # verified v1 Responses route as complete_chat; the api-versioned chat client would
            # silently drop reasoning_effort. Reasoning models reject non-default sampling, so
            # `temperature` is not forwarded (matching OpenAIResponsesProvider.complete).
            response = self.complete_chat(
                ChatRequest.model_validate(
                    {
                        "messages": [
                            *([{"role": "system", "content": system}] if system else []),
                            *[{"role": m.role, "content": m.content} for m in messages],
                        ],
                        "max_completion_tokens": max_tokens,
                    }
                )
            )
            message = response.choices[0].message if response.choices else None
            text = (
                message.content if message is not None and isinstance(message.content, str) else ""
            )
            usage = response.token_usage()
            return Completion(
                text=text,
                usage=TokenUsage(
                    input_tokens=usage.input_tokens, output_tokens=usage.output_tokens
                ),
            )
        return _openai_common.complete(
            self._get_client().chat.completions,
            self._deployment(),
            system,
            messages,
            max_tokens,
            # Open-tier deployments (DeepSeek, Kimi) genuinely sample and take the classic
            # max_tokens param; GPT-5.x rejects sampling params and wants max_completion_tokens.
            temperature=temperature if self._forward_temperature else None,
            max_tokens_field=self.config.resolved_chat_max_tokens_field(),
        )

    def stream(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Iterator[StreamChunk]:
        """Stream a completion natively from the deployment.

        Temperature is forwarded only for deployments that sample (open-tier models like
        DeepSeek/Kimi); GPT-5.x deployments reject sampling params, mirroring `complete`.
        """
        if self.config.reasoning_effort is not None:
            # The api-versioned chat-completions stream would silently drop reasoning_effort;
            # honest failure until a native Responses-route stream lands with the serving PR.
            raise NotImplementedError(
                "streaming is not yet implemented for reasoning_effort Azure configs; "
                "unset reasoning_effort or use complete()"
            )
        return _openai_common.stream(
            self._get_client().chat.completions,
            self._deployment(),
            system,
            messages,
            max_tokens,
            temperature=temperature if self._forward_temperature else None,
            max_tokens_field=self.config.resolved_chat_max_tokens_field(),
        )

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        """Run a full structured request on the configured Azure deployment."""
        request = normalize_chat_temperature(
            request,
            forward_temperature=self._forward_temperature,
        )
        if self.config.reasoning_effort is not None:
            # Structured reasoning needs Azure's native v1 Responses route: encrypted-reasoning
            # replay and reasoning.effort are not part of the api-versioned chat-completions API.
            return _responses_common.complete_chat(
                self._get_responses_client().responses,
                self._deployment(),
                request,
                reasoning_effort=self.config.reasoning_effort,
                allow_sampling=False,
            )
        return _openai_common.complete_chat(
            self._get_client().chat.completions,
            self._deployment(),
            request,
            max_tokens_field=self.config.resolved_chat_max_tokens_field(),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embed_with_usage(texts).vectors

    def embed_with_usage(self, texts: list[str]) -> EmbeddingResult:
        # As with `model` in complete(), `embed_model` must be the Azure *deployment* name of an
        # embedding model, not a base OpenAI model id, or the call 404s.
        if self.config.embed_model is None:
            raise ValueError("AzureOpenAIProvider.embed requires config.embed_model (deployment).")
        return _openai_common.embed_with_usage(
            self._get_client().embeddings, self.config.embed_model, texts, self.config.embed_dim
        )

    def verify(self) -> VerifyResult:
        if self.config.reasoning_effort is not None:
            # A reasoning config dispatches through the Azure v1 Responses route; a plain
            # chat-completions ping would prove the wrong endpoint and api surface.
            return verify_via_ping(self, ping=lambda: self.complete_chat(_REASONING_PING))
        return verify_via_ping(self)


def _same_endpoint(a: str, b: str | None) -> bool:
    """True when two endpoint strings name the same host, path, and query.

    Scheme and host are compared case-insensitively (per URL semantics); path and query are
    compared case-sensitively (both are case-sensitive), with only a trailing slash ignored. This
    tolerates a trailing slash or host-casing difference for the *same* Azure resource without
    treating a URL that differs in a case-sensitive path or query component as equal.
    """
    if b is None:
        return False
    pa, pb = urlsplit(a), urlsplit(b)
    return (pa.scheme.lower(), pa.netloc.lower(), pa.path.rstrip("/"), pa.query) == (
        pb.scheme.lower(),
        pb.netloc.lower(),
        pb.path.rstrip("/"),
        pb.query,
    )


def _responses_base_url(endpoint: str) -> str:
    """Normalize an Azure resource endpoint onto its native v1 route."""
    parsed = urlsplit(endpoint)
    path = parsed.path.rstrip("/")
    if not path.lower().endswith("/openai/v1"):
        path = f"{path}/openai/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, f"{path}/", parsed.query, ""))
