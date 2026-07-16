"""AWS Bedrock provider (Claude 4.8 / Amazon Nova). Reads AWS credentials from the environment."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, TypedDict, cast

from wmh.core.types import JsonValue
from wmh.providers._bedrock_chat import converse_request, converse_response
from wmh.providers.base import (
    DEFAULT_MAX_TOKENS,
    ChatRequest,
    ChatResponse,
    Completion,
    Message,
    ProviderConfig,
    TokenUsage,
    VerifyResult,
    verify_via_ping,
)

if TYPE_CHECKING:
    from botocore.client import BaseClient

# Bedrock speaks the same Anthropic Messages schema as the direct API, pinned by this version tag.
_ANTHROPIC_BEDROCK_VERSION = "bedrock-2023-05-31"

# Default Titan text-embeddings model (confirmed reachable; v2 supports `dimensions` 256/512/1024).
_DEFAULT_EMBED_MODEL = "amazon.titan-embed-text-v2:0"


class _ContentBlock(TypedDict):
    type: str
    text: str


class _Usage(TypedDict):
    input_tokens: int
    output_tokens: int


class _BedrockResponse(TypedDict):
    content: list[_ContentBlock]
    usage: _Usage


class _TitanEmbedResponse(TypedDict):
    embedding: list[float]


class _NovaContentBlock(TypedDict):
    text: str


class _NovaMessage(TypedDict):
    content: list[_NovaContentBlock]


class _NovaOutput(TypedDict):
    message: _NovaMessage


class _NovaUsage(TypedDict):
    inputTokens: int
    outputTokens: int


class _NovaResponse(TypedDict):
    output: _NovaOutput
    usage: _NovaUsage


def _is_nova(model_id: str) -> bool:
    """Whether `model_id` is an Amazon Nova model (e.g. `us.amazon.nova-lite-v1:0`)."""
    return ".nova-" in model_id or model_id.startswith("amazon.nova-")


class BedrockProvider:
    """Claude 4.8 via the Bedrock Runtime (InvokeModel with the Anthropic Messages body)."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._client: BaseClient | None = None

    def _get_client(self) -> BaseClient:
        # Lazy: import boto3 and open the client only on first use. region falls back to
        # AWS_REGION / the default boto3 chain when config.region is unset.
        if self._client is None:
            import boto3
            from botocore.config import Config

            # Bound each request so a stalled connection RAISES instead of blocking forever. Without
            # this, a single hung InvokeModel wedges the whole run (long GEPA/eval jobs never
            # finish) and a failover chain can't fail over — it only reacts to raised errors.
            # `read_timeout` is generous because reasoning models can generate for a while at up to
            # `max_tokens` (a mid-generation cutoff wastes the whole call and, under a fallback
            # chain, silently substitutes a different model into an eval).
            #
            # `total_max_attempts=1` disables botocore's OWN retries on purpose (it counts the
            # initial request; botocore's `max_attempts` counts retries AFTER it): throttling/5xx/
            # timeouts should surface IMMEDIATELY to the caller, where the failover chain owns retry
            # policy (fail over to the next model). Leaving botocore's adaptive retries on would
            # stack 3 internal attempts per model UNDER our 4-model failover — up to 12 backend
            # calls with back-off for one throttled request — turning graceful degradation into a
            # slow crawl.
            # "standard" mode makes max_attempts mean TOTAL attempts (legacy mode still
            # sneaks in one internal retry, which showed up as a long silent stall before the
            # CLI's own narrated backoff could react).
            # `tcp_keepalive` guards the other stall mode: a keep-alive connection the LB
            # silently dropped during an idle gap. Without it, the next call on that socket
            # hangs until read_timeout (observed as ~10-minute stalls on the FIRST call after
            # idle: turn-1 WM steps, sparse judge calls); with it, the OS detects the dead
            # peer and the call fails fast into the failover chain.
            client_config = Config(
                connect_timeout=15,
                read_timeout=600,
                retries={"max_attempts": 1, "mode": "standard"},
                tcp_keepalive=True,
            )
            self._client = boto3.client(
                "bedrock-runtime", region_name=self.config.region, config=client_config
            )
        return self._client

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion:
        if _is_nova(self.config.model):
            return self._complete_nova(
                system, messages, temperature=temperature, max_tokens=max_tokens
            )
        if "anthropic" not in self.config.model:
            # Kimi, DeepSeek, and other third-party models: the model-agnostic Converse API.
            return self._complete_converse(
                system, messages, temperature=temperature, max_tokens=max_tokens
            )
        # Claude 4.8 rejects sampling params, so temperature is intentionally not forwarded.
        body = {
            "anthropic_version": _ANTHROPIC_BEDROCK_VERSION,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        raw = self._get_client().invoke_model(modelId=self.config.model, body=json.dumps(body))
        data = cast("_BedrockResponse", json.loads(raw["body"].read()))
        text = "".join(block["text"] for block in data["content"] if block["type"] == "text")
        usage = TokenUsage(
            input_tokens=data["usage"]["input_tokens"],
            output_tokens=data["usage"]["output_tokens"],
        )
        return Completion(text=text, usage=usage)

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        """Run a full structured agent request through Bedrock Converse."""
        raw = self._get_client().converse(**converse_request(request, self.config.model))
        return converse_response(raw, self.config.model)

    def _complete_converse(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
    ) -> Completion:
        """Complete via the Converse API (model-agnostic: Kimi, DeepSeek, ...).

        Converse normalizes request/response shapes across vendors; thinking models may emit
        `reasoningContent` blocks, which are skipped — callers get the visible text only.
        """
        kwargs: dict[str, JsonValue] = {
            "modelId": self.config.model,
            "messages": [{"role": m.role, "content": [{"text": m.content}]} for m in messages],
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
        }
        if system:
            kwargs["system"] = [{"text": system}]
        response = self._get_client().converse(**kwargs)
        blocks = response["output"]["message"]["content"]
        text = "".join(block["text"] for block in blocks if "text" in block)
        usage = TokenUsage(
            input_tokens=int(response["usage"]["inputTokens"]),
            output_tokens=int(response["usage"]["outputTokens"]),
        )
        return Completion(text=text, usage=usage)

    def _complete_nova(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
    ) -> Completion:
        """Complete via an Amazon Nova model (different request/response schema than Anthropic).

        Nova wraps message content in `[{"text": ...}]` blocks, takes sampling params under
        `inferenceConfig`, and returns the reply under `output.message`. Unlike the Claude path,
        `temperature` IS forwarded — Nova accepts it.
        """
        body: dict[str, JsonValue] = {
            "messages": [{"role": m.role, "content": [{"text": m.content}]} for m in messages],
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
        }
        if system:
            body["system"] = [{"text": system}]
        raw = self._get_client().invoke_model(modelId=self.config.model, body=json.dumps(body))
        data = cast("_NovaResponse", json.loads(raw["body"].read()))
        text = "".join(block["text"] for block in data["output"]["message"]["content"])
        usage = TokenUsage(
            input_tokens=data["usage"]["inputTokens"],
            output_tokens=data["usage"]["outputTokens"],
        )
        return Completion(text=text, usage=usage)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed via Amazon Titan text embeddings on Bedrock (phi for retrieval).

        Titan's InvokeModel embeds one text per call (no batch input), so we loop. `embed_model`
        selects the Titan model (defaults to titan-embed-text-v2); `embed_dim`, when set, requests a
        specific output dimension (v2 supports 256/512/1024) so the index and query vectors match.
        """
        model = self.config.embed_model or _DEFAULT_EMBED_MODEL
        client = self._get_client()
        vectors: list[list[float]] = []
        for text in texts:
            body: dict[str, JsonValue] = {"inputText": text}
            if self.config.embed_dim is not None:
                body["dimensions"] = self.config.embed_dim
                body["normalize"] = True
            raw = client.invoke_model(modelId=model, body=json.dumps(body))
            data = cast("_TitanEmbedResponse", json.loads(raw["body"].read()))
            vectors.append(data["embedding"])
        return vectors

    def verify(self) -> VerifyResult:
        return verify_via_ping(self)
