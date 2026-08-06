"""Minimal OpenAI-compatible /v1/chat/completions shim for Qwen-AgentWorld's judge stage.

Lets `eval.py judge` run unmodified against a model its hardcoded request shape can't reach
directly, via `--judge-base-url http://127.0.0.1:8765/v1 --judge-api-key EMPTY`. Two backends:

- `--backend bedrock` (Anthropic via AWS): plumbing-proof stand-in.
- `--backend azure` (Azure OpenAI v1-compat endpoint): translates their hardcoded `max_tokens`
  to `max_completion_tokens` (gpt-5.x rejects `max_tokens`) and forwards `temperature`.
  Key from AZURE_OPENAI_API_KEY.

Judge-pinning (D12): whatever runs behind this shim is the recorded judge for those rows —
AgentWorldBench's paper numbers used OpenAI `gpt-5.2-2025-12-11`, so any other judge here is
non-comparable to their table and must be labeled.

Usage:
    uv run python .agents/scripts/agentworldbench/judge_shim.py \
        --backend azure --endpoint https://google-sheets.openai.azure.com --model gpt-5.4-mini
    curl http://127.0.0.1:8765/usage   # metered judge tokens (+cost where priced) so far
"""

from __future__ import annotations

import argparse
import os
import time
import uuid

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from wmh.providers import get_provider
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind, TokenUsage
from wmh.tracking.metered import MeteredProvider
from wmh.tracking.tracker import Phase, RunTracker

MAX_OUTPUT_TOKENS = 16384  # clamp their 32768 default; reasoning models spend output on thinking


class AzureV1Backend:
    """Direct Azure OpenAI v1-compat caller: Bearer auth, max_completion_tokens, temperature kept.

    Bypasses wmh's AzureOpenAIProvider because that provider intentionally never forwards
    temperature and requires deployment/api_version config; the judge protocol wants an exact
    temperature pin and the v1-compat endpoint takes the plain model/deployment name.
    """

    def __init__(self, endpoint: str, model: str, tracker: RunTracker) -> None:
        from openai import OpenAI

        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("AZURE_OPENAI_API_KEY is not set (put it in .env or export it).")
        self._client = OpenAI(base_url=f"{endpoint.rstrip('/')}/openai/v1", api_key=api_key)
        self._model = model
        self._tracker = tracker

    def complete(
        self, system: str, messages: list[Message], *, temperature: float, max_tokens: int
    ) -> Completion:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages],
            max_completion_tokens=max_tokens,
            temperature=temperature,
        )
        usage = TokenUsage(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )
        self._tracker.record(Phase.JUDGE, self._model, usage)
        return Completion(text=response.choices[0].message.content or "", usage=usage)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = MAX_OUTPUT_TOKENS
    temperature: float = 0.0


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage


class UsageSummary(BaseModel):
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    judge_model: str = Field(description="the actual backing model, not the requested alias")


def create_app(backend: str, model: str, region: str, endpoint: str | None) -> FastAPI:
    tracker = RunTracker(run_id="awb-judge-shim", kind="eval")
    tracker.start()
    provider: AzureV1Backend | MeteredProvider
    if backend == "azure":
        if not endpoint:
            raise SystemExit("--backend azure requires --endpoint")
        provider = AzureV1Backend(endpoint, model, tracker)
    else:
        provider = MeteredProvider(
            get_provider(ProviderConfig(kind=ProviderKind.BEDROCK, model=model, region=region)),
            tracker,
            base_phase=Phase.JUDGE,
        )
    app = FastAPI(title="wmh AgentWorldBench judge shim")
    calls = {"n": 0}

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatRequest) -> ChatResponse:
        system = "\n\n".join(m.content for m in request.messages if m.role == "system")
        turns = [
            Message(role="assistant" if m.role == "assistant" else "user", content=m.content)
            for m in request.messages
            if m.role != "system"
        ]
        completion = provider.complete(
            system,
            turns,
            temperature=request.temperature,
            max_tokens=min(request.max_tokens, MAX_OUTPUT_TOKENS),
        )
        calls["n"] += 1
        return ChatResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=model,
            choices=[ChatChoice(message=ChatMessage(role="assistant", content=completion.text))],
            usage=ChatUsage(
                prompt_tokens=completion.usage.input_tokens,
                completion_tokens=completion.usage.output_tokens,
                total_tokens=completion.usage.input_tokens + completion.usage.output_tokens,
            ),
        )

    @app.get("/usage")
    def usage() -> UsageSummary:
        total = tracker.record_summary().total
        return UsageSummary(
            calls=calls["n"],
            input_tokens=total.input_tokens,
            output_tokens=total.output_tokens,
            cost_usd=total.cost_usd,
            judge_model=model,
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["bedrock", "azure"], default="bedrock")
    parser.add_argument("--model", default="us.anthropic.claude-opus-4-8")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--endpoint", default=None, help="Azure resource URL (azure backend)")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run(
        create_app(args.backend, args.model, args.region, args.endpoint),
        host="127.0.0.1",
        port=args.port,
    )


if __name__ == "__main__":
    main()
