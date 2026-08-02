"""OpenAI direct provider using the Responses API. Reads OPENAI_API_KEY from the environment."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from wmo.providers import _openai_common, _responses_common
from wmo.providers.base import (
    DEFAULT_MAX_TOKENS,
    ChatRequest,
    ChatResponse,
    Completion,
    EmbeddingResult,
    Message,
    ProviderConfig,
    StreamChunk,
    TokenUsage,
    VerifyResult,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from openai import OpenAI
    from openai.types.responses.response_input_param import ResponseInputParam
    from openai.types.shared_params.reasoning import Reasoning


class OpenAIResponsesProvider:
    """GPT 5.x via OpenAI's Responses API."""

    def __init__(self, config: ProviderConfig, *, api_key: str | None = None) -> None:
        self.config = config
        # Trusted explicit credential from get_provider (pool entries with api_key_env);
        # None means the SDK reads OPENAI_API_KEY from the environment.
        self._api_key = api_key
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        """Create and cache the OpenAI SDK client on first use."""
        # Lazy: don't import the SDK or read OPENAI_API_KEY until first use.
        if self._client is None:
            from openai import OpenAI

            if self._api_key is not None:
                self._client = OpenAI(api_key=self._api_key)
            else:
                self._client = OpenAI()  # picks up OPENAI_API_KEY from the environment
        return self._client

    def prepare(self) -> None:
        """Import the SDK and build the client, which resolves the key. No request is sent.

        Satisfies `wmo.providers.base.PreparableProvider`, exactly as in `wmo.providers.openai`:
        `OpenAI()` refuses to construct without a resolvable key and opens no connection, so this
        turns a missing credential into a local failure instead of a first-call one.

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
        """Generate a completion through the Responses API.

        Args:
            system: System prompt to send as the first Responses input item.
            messages: Conversation messages following the system prompt.
            temperature: Accepted for provider interface compatibility; not forwarded because the
                benchmark path keeps Responses-model sampling at provider defaults.
            max_tokens: Maximum number of output tokens to request.

        Returns:
            Completion text and token usage parsed from the Responses API response.
        """
        # GPT-5.x Responses models reject non-default sampling in this benchmark path.
        del temperature
        response_input = cast("ResponseInputParam", _responses_input(system, messages))
        responses = self._get_client().responses
        if self.config.reasoning_effort:
            response = responses.create(
                model=self.config.model,
                input=response_input,
                max_output_tokens=max_tokens,
                store=False,
                reasoning=cast("Reasoning", {"effort": self.config.reasoning_effort}),
            )
        else:
            response = responses.create(
                model=self.config.model,
                input=response_input,
                max_output_tokens=max_tokens,
                store=False,
            )
        return Completion(text=_response_text(response), usage=_usage_from_response(response))

    def stream(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Iterator[StreamChunk]:
        """Stream a completion through the Responses API (temperature not forwarded)."""
        del temperature  # mirror complete(): Responses models keep provider-default sampling
        response_input = cast("ResponseInputParam", _responses_input(system, messages))
        kwargs: dict[str, object] = {
            "model": self.config.model,
            "input": response_input,
            "max_output_tokens": max_tokens,
            "store": False,
            "stream": True,
        }
        if self.config.reasoning_effort:
            kwargs["reasoning"] = cast("Reasoning", {"effort": self.config.reasoning_effort})
        usage = TokenUsage()
        # The evolving Responses stream-event union stays behind this one SDK-boundary cast
        # (same pattern as _openai_common.complete_chat).
        events = cast("Any", self._get_client().responses).create(**kwargs)
        try:
            for event in events:
                kind = getattr(event, "type", "")
                if kind == "response.output_text.delta":
                    if event.delta:
                        yield StreamChunk(delta=event.delta)
                elif kind == "response.completed":
                    # Same extractor as complete(): keeps the cached-token split, which OpenAI
                    # populates automatically for prompts >= 1024 tokens.
                    usage = _usage_from_response(event.response)
        finally:
            close = getattr(events, "close", None)
            if callable(close):
                close()
        yield StreamChunk(done=True, usage=usage)

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        """Run a full structured request through the native Responses API."""
        return _responses_common.complete_chat(
            self._get_client().responses,
            self.config.model,
            request,
            reasoning_effort=self.config.reasoning_effort,
            allow_sampling=False,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed text through OpenAI's embeddings API.

        Args:
            texts: Text strings to embed.

        Returns:
            One embedding vector per input text.

        Raises:
            ValueError: If `ProviderConfig.embed_model` is unset.
        """
        return self.embed_with_usage(texts).vectors

    def embed_with_usage(self, texts: list[str]) -> EmbeddingResult:
        """Embed text and retain provider-reported prompt tokens for routing cost."""
        if self.config.embed_model is None:
            raise ValueError("OpenAIResponsesProvider.embed requires config.embed_model to be set.")
        return _openai_common.embed_with_usage(
            self._get_client().embeddings, self.config.embed_model, texts, self.config.embed_dim
        )

    def verify(self) -> VerifyResult:
        """Run a cheap completion request and report provider availability."""
        try:
            self.complete(
                "",
                [Message(role="user", content="Reply with exactly: ok")],
                max_tokens=256,
            )
        except Exception as exc:  # noqa: BLE001 - verify reports failure, never raises
            return VerifyResult(
                ok=False,
                kind=self.config.kind,
                model=self.config.model,
                detail=str(exc),
            )
        return VerifyResult(ok=True, kind=self.config.kind, model=self.config.model)


def _responses_input(system: str, messages: list[Message]) -> list[dict[str, str]]:
    """Convert the provider message shape into Responses API input items."""
    out: list[dict[str, str]] = []
    if system:
        out.append({"role": "system", "content": system})
    out.extend({"role": message.role, "content": message.content} for message in messages)
    return out


def _get(value: object, key: str) -> object:
    """Read a field from either an SDK object or mapping."""
    if isinstance(value, Mapping):
        return cast("Mapping[str, object]", value).get(key)
    return getattr(value, key, None)


def _as_int(value: object) -> int:
    """Coerce numeric SDK usage fields into integers, defaulting invalid values to zero."""
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _response_text(response: object) -> str:
    """Extract generated text from direct or nested Responses SDK shapes."""
    direct = _get(response, "output_text")
    if isinstance(direct, str) and direct:
        return direct

    chunks: list[str] = []
    output = _get(response, "output")
    if isinstance(output, list):
        for item in output:
            content = _get(item, "content")
            if not isinstance(content, list):
                continue
            for block in content:
                text = _get(block, "text")
                if isinstance(text, str):
                    chunks.append(text)
    return "".join(chunks)


def _usage_from_response(response: object) -> TokenUsage:
    """Extract token usage from a Responses SDK object or mapping."""
    usage = _get(response, "usage")
    if usage is None:
        return TokenUsage()
    details = _get(usage, "input_tokens_details")
    return TokenUsage(
        input_tokens=_as_int(_get(usage, "input_tokens")),
        output_tokens=_as_int(_get(usage, "output_tokens")),
        cached_input_tokens=_as_int(_get(details, "cached_tokens")) if details is not None else 0,
    )
