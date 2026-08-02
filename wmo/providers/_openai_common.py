"""Shared request mapping / response parsing for the two OpenAI-shaped backends.

`OpenAIProvider` and `AzureOpenAIProvider` differ only in how their client is constructed; the
chat-completion and embedding wire formats are identical, so that logic lives here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

from llm_waterfall import ChatMaxTokensField, ChatRequest, ChatResponse
from openai import BadRequestError

from wmo.providers.base import (
    Completion,
    EmbeddingResult,
    Message,
    StreamChunk,
    TokenUsage,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from openai.types import CreateEmbeddingResponse
    from openai.types.chat import ChatCompletion, ChatCompletionMessageParam


class _ChatCompletions(Protocol):
    def create(
        self,
        *,
        model: str,
        messages: list[ChatCompletionMessageParam],
        max_completion_tokens: int,
        temperature: float = ...,
    ) -> ChatCompletion: ...


class _Embeddings(Protocol):
    def create(
        self, *, model: str, input: list[str], dimensions: int = ...
    ) -> CreateEmbeddingResponse: ...


def to_messages(system: str, messages: list[Message]) -> list[ChatCompletionMessageParam]:
    """Fold the system prompt into the message list as OpenAI's leading `system` turn."""
    out: list[dict[str, str]] = []
    if system:
        out.append({"role": "system", "content": system})
    out.extend({"role": m.role, "content": m.content} for m in messages)
    return cast("list[ChatCompletionMessageParam]", out)


def complete(
    chat_completions: _ChatCompletions,
    model: str,
    system: str,
    messages: list[Message],
    max_tokens: int,
    temperature: float | None = None,
    max_tokens_field: str = "max_completion_tokens",
) -> Completion:
    """Run one chat completion and map it onto our `Completion`.

    `max_tokens_field` names the output-budget parameter the deployment accepts: GPT-5.x wants
    `max_completion_tokens`, while Azure MaaS open models (DeepSeek, Kimi) still take the
    classic `max_tokens` (see `ProviderConfig.resolved_chat_max_tokens_field`). `temperature`
    is sent ONLY when given: GPT 5.5's reasoning models reject non-default sampling params
    (callers pass None), while OpenAI-compatible servers (vLLM policies) need it.
    """
    # The output-budget param name is dynamic, so this call crosses the SDK boundary through
    # the same one-line cast `stream` uses below.
    resource = cast("Any", chat_completions)
    base_kwargs: dict[str, Any] = {
        "model": model,
        "messages": to_messages(system, messages),
        max_tokens_field: max_tokens,
    }
    if temperature is None:
        response: ChatCompletion = resource.create(**base_kwargs)
    else:
        try:
            response = resource.create(**base_kwargs, temperature=temperature)
        except BadRequestError as exc:
            # Reasoning-model deployments (GPT-5.x behind Azure/custom endpoints) reject any
            # non-default temperature with a 400 unsupported_value. The caller can't know which
            # models sample; degrade to the model's default rather than failing the request.
            if "temperature" not in str(exc):
                raise
            response = resource.create(**base_kwargs)
    if not response.choices:
        # Content filtering (and some error modes) can return zero choices; surface it clearly
        # rather than letting choices[0] raise a bare IndexError.
        raise ValueError(f"{model} returned no choices")
    text = response.choices[0].message.content or ""
    usage = response.usage
    token_usage = _chat_usage(usage) if usage is not None else TokenUsage()
    return Completion(text=text, usage=token_usage)


def stream(
    chat_completions: object,
    model: str,
    system: str,
    messages: list[Message],
    max_tokens: int,
    temperature: float | None = None,
    max_tokens_field: str = "max_completion_tokens",
) -> Iterator[StreamChunk]:
    """Stream one chat completion as `StreamChunk`s (deltas, then a terminal chunk with usage).

    `stream_options.include_usage` makes the wire stream end with a usage-bearing chunk, so the
    terminal `StreamChunk` carries real token counts instead of estimates. `temperature` and
    `max_tokens_field` follow the same rules as `complete`.
    """
    resource = cast("Any", chat_completions)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": to_messages(system, messages),
        max_tokens_field: max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    usage = TokenUsage()
    upstream = resource.create(**kwargs)
    try:
        for chunk in upstream:
            choices = getattr(chunk, "choices", None) or []
            if choices:
                delta = getattr(choices[0], "delta", None)
                text = getattr(delta, "content", None) if delta is not None else None
                if text:
                    yield StreamChunk(delta=text)
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = _chat_usage(chunk_usage)
    finally:
        # The SDK Stream holds an httpx response; without an explicit close an abandoned
        # stream releases the connection only when the object is garbage collected.
        close = getattr(upstream, "close", None)
        if callable(close):
            close()
    yield StreamChunk(done=True, usage=usage)


def _chat_usage(usage: object) -> TokenUsage:
    """Chat-completions usage -> TokenUsage, including the cached-prompt split when reported."""
    details = getattr(usage, "prompt_tokens_details", None)
    return TokenUsage(
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        cached_input_tokens=(getattr(details, "cached_tokens", None) or 0) if details else 0,
    )


def complete_chat(
    chat_completions: object,
    model: str,
    request: ChatRequest,
    *,
    max_tokens_field: ChatMaxTokensField,
) -> ChatResponse:
    """Run a validated structured request against an OpenAI-compatible SDK resource."""
    # ChatRequest validates the stable tool-calling core before this SDK boundary. The OpenAI
    # package models its evolving request surface as a large TypedDict union, so the narrow cast
    # preserves forward-compatible extra fields without leaking Any into the public contract.
    resource = cast("Any", chat_completions)
    response = resource.create(**request.provider_payload(model, max_tokens_field=max_tokens_field))
    return ChatResponse.model_validate(response.model_dump(mode="json"))


def embed(
    embeddings: _Embeddings, model: str, texts: list[str], dim: int | None = None
) -> list[list[float]]:
    """Embed `texts` against `model` (an OpenAI model id, or an Azure embedding deployment).

    `dim`, when set, requests a specific output dimension via the `dimensions` param (supported by
    text-embedding-3-* and their Azure deployments) so the index and query vectors match.
    """
    return embed_with_usage(embeddings, model, texts, dim).vectors


def embed_with_usage(
    embeddings: _Embeddings,
    model: str,
    texts: list[str],
    dim: int | None = None,
) -> EmbeddingResult:
    """Embed text and retain the provider's billable prompt-token count."""
    response = (
        embeddings.create(model=model, input=texts, dimensions=dim)
        if dim is not None
        else embeddings.create(model=model, input=texts)
    )
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage is not None else 0
    return EmbeddingResult(
        vectors=[item.embedding for item in response.data],
        usage=TokenUsage(input_tokens=prompt_tokens or 0),
        model=model,
    )
