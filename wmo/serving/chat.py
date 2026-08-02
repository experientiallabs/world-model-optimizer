"""OpenAI-compatible `/v1/chat/completions` serving over a routing policy.

The endpoint face of the pivot: a customer points their OpenAI client at wmo, `model` in the
request names an ENDPOINT (world model + policy), and the learned inference policy picks which
pool model actually serves each call. Responses stay OpenAI-pure and name only the endpoint;
the mechanism (routed model, cluster, reason) goes to the request log and the
`x-wmo-routed-model` debug header, never customer-facing copy.

Conversation affinity: provider prompt caches are per-model, so switching mid-conversation
forfeits warm cache reads. The runtime fingerprints each finished exchange (full transcript
including the assistant reply, tool calls and all) and, when the next request arrives with that
transcript as its prefix, `select_model` sees the incumbent and sticks to it by default. A tool
round trip is such a prefix: the client appends the assistant's `tool_calls` turn plus one
`role="tool"` result PER CALL, so the reply is remembered with its tool calls and the prefix is
looked up at the last assistant turn rather than one message back, or a parallel round trip would
route fresh and forfeit the cache exactly when the transcript is longest.

Tool calling: `tools`, `tool_choice`, `parallel_tool_calls`, assistant `tool_calls`, and
`role="tool"` results all round-trip. A request carrying any of them is served through the
provider's structured `complete_chat`, the only seam that preserves tool calls, instead of the
text `complete`; a routed pool entry whose provider has no structured backend fails loudly (501,
naming that entry) because silently dropping tools turns a compatibility gap into an apparent
model-quality problem. A `tool_choice` that DEMANDS a call (`"required"`, or a named function) is
checked against the tools that request declares, replayed transcript or not, and refused with a
400 when nothing there can satisfy it: `tool_choice` with `tools: null` is malformed upstream, so
the alternative is a 502 that reads as a model failure (or, on Bedrock, prose returned to a client
that requires a call). Streamed tool calls are RE-EMITTED, not forwarded: no provider exposes a
streaming tool-call surface (`StreamingProvider.stream` yields text deltas only), so a
tool-bearing streamed request makes one non-streamed `complete_chat` call and re-frames the
result as OpenAI SSE (one `choices[0].delta.tool_calls` entry per call, carrying `index`, `id`,
`function.name` and the whole `function.arguments` string, then the `finish_reason` chunk). A
client reassembles the arguments exactly as it would from real fragments; the tradeoff is
time-to-first-byte, which waits for the whole upstream response instead of the first token.

Compression stage (D-COMPRESS): when the policy carries a compression config, the pipeline is
request -> [compress] -> [route] -> provider call. Only user-message content is compressed;
system prompts, the model's own prior replies, tool calls, and tool results pass through
verbatim (v1 scope: a tool payload is a structured contract, not prose to shorten). The affinity
state decides segment boundaries: an incumbent conversation's compressed prefix is stored
alongside its fingerprint and REUSED, never recompressed, so the provider-visible prefix stays
byte-identical across turns (the prompt cache survives by construction). Routing embeds the
compressed text (the router sees what the model sees) while stickiness keys on the raw
transcript the client resends. Compression fields go to the request log only, never response
bodies or headers.

Request log: one JSONL row per call with the D-SERVING-LOG fields (id, ts, endpoint, routed
model, cluster, tokens incl. cached, cache-adjusted cost, latency, ttfb, status, reason).
Provider cache CONTROLS (breakpoint placement, TTL) are not exposed yet; they land with the
cache-aware routing model.

The endpoint's operator surface lives here too, deliberately outside the OpenAI routes so a
customer's OpenAI client sees exactly what it saw before: `GET`/`PUT /v1/endpoints/{name}/config`
reads and moves the cost/quality dial (`wmo.optimize.knn.apply_cost_quality`) on the live runtime
with no restart and no refit, and `GET /v1/endpoints/{name}/savings` totals what the endpoint has
saved so far out of the request log (`wmo.serving.savings`).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import threading
import time
import uuid
from collections import OrderedDict, deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from llm_waterfall.types import ChatChoice, ChatRequest, ChatResponse, ChatTool, ChatToolCall
from llm_waterfall.types import ChatMessage as ProviderChatMessage
from pydantic import BaseModel, Field, JsonValue, field_validator, model_validator
from starlette.background import BackgroundTask

from wmo.optimize.compression import (
    CompressionConfig,
    CompressionStats,
    Compressor,
    compress_segments,
    estimate_tokens,
    same_compression,
)
from wmo.optimize.knn import (
    COST_QUALITY_ANCHORS,
    CostQualityAnchor,
    apply_cost_quality,
    cost_quality_named_point,
)
from wmo.optimize.pareto import ParetoCurve
from wmo.optimize.policy import (
    Embedder,
    GateOutcome,
    Propensity,
    RoutingDecision,
    RoutingPolicy,
    select_model,
)
from wmo.providers.base import (
    DEFAULT_MAX_TOKENS,
    Message,
    Provider,
    StreamingProvider,
    TokenUsage,
    ToolCallingProvider,
)
from wmo.providers.pool import PoolEntry, pool_provider
from wmo.serving.endpoint_config import EndpointConfig
from wmo.serving.query_embeddings import QueryEmbeddingStore
from wmo.serving.savings import EndpointSavings, SavingsWindow, compute_savings

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping
    from pathlib import Path

logger = logging.getLogger(__name__)

# Finished-exchange fingerprints remembered per endpoint for conversation affinity. Bounded so
# a long-running server cannot grow without limit; least-recently-used conversations re-route.
_AFFINITY_CAPACITY = 4096

# The compressed-transcript cache is bounded by BYTES, not by entry count. An affinity entry is
# a fingerprint and a model name (tens of bytes, so 4096 of them is nothing), but a compressed
# transcript is the conversation itself: a measured 40-turn conversation runs about 99KB, and
# tool results push that much higher, so reusing the count cap would allow 0.41GB of ordinary
# traffic and multiple GB of tool-heavy traffic. 64MB holds roughly 650 conversations of that
# measured size, which is far more than the affinity map keeps anyway, and costs a fixed,
# statable amount of memory per endpoint instead of an unbounded one.
_COMPRESSED_CAPACITY_BYTES = 64 * 1024 * 1024


ChatRole = Literal["system", "user", "assistant", "tool"]


class ChatMessage(BaseModel):
    """One chat turn, normalized from the OpenAI request shapes clients actually send.

    `developer` (OpenAI's system replacement on gpt-5-class models) maps to `system`, and
    multi-part text content (`[{"type": "text", "text": ...}]`, emitted by LangChain and the
    Vercel AI SDK) is joined; a non-text part is a hard error, not a silent drop. `content` is
    also the empty string when a client sends null, which is what an assistant turn that only
    calls tools looks like on the wire.

    The tool-calling half is the shape an agent replays: `tool_calls` on the assistant turn it
    got back, then one `role="tool"` message per result carrying the `tool_call_id` it answers.
    Both reuse llm-waterfall's models (`ChatToolCall`), the same types `complete_chat` takes, so
    a call survives the hop to the provider without a second parallel definition of the format.
    """

    role: ChatRole
    content: str = ""
    tool_calls: list[ChatToolCall] | None = None
    tool_call_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _content_is_required_off_a_tool_call_turn(cls, data: object) -> object:
        """Reject a turn with no text unless it is the one turn that legitimately has none.

        Checked on the RAW payload so both spellings of "no content" answer the same way: an
        omitted key and an explicit `null` (what an SDK puts on the wire for an unset value) are
        the same request, and `_normalize_content` turns the second into "" before any
        field-level check could tell them apart. Either way an empty billed turn is a client bug
        this endpoint has to name, not forward.

        The one exception is an assistant turn whose whole output is `tool_calls`, and it is the
        calls that earn it: `tool_calls: []` advertises no call, so a turn carrying one still
        needs its text.
        """
        if not isinstance(data, dict) or data.get("content") is not None:
            return data
        if data.get("role") == "assistant" and data.get("tool_calls"):
            return data
        raise ValueError(
            "a message needs `content` (only an assistant turn carrying tool_calls may omit it)"
        )

    @field_validator("role", mode="before")
    @classmethod
    def _developer_is_system(cls, value: object) -> object:
        return "system" if value == "developer" else value

    @field_validator("content", mode="before")
    @classmethod
    def _normalize_content(cls, value: object) -> object:
        if value is None:
            return ""
        if not isinstance(value, list):
            return value
        parts: list[str] = []
        for part in value:
            if not (isinstance(part, dict) and part.get("type") == "text"):
                raise ValueError(
                    "only text content parts are supported; images and other modalities "
                    "are not available on this endpoint yet"
                )
            parts.append(str(part.get("text", "")))
        return "".join(parts)

    @model_validator(mode="after")
    def _tool_fields_match_the_role(self) -> ChatMessage:
        """Reject the tool-field placements that would silently misrepresent a transcript.

        A tool result carried on an assistant turn (or a result with no `tool_call_id`) still
        reaches the model, as prose in the wrong voice with no link to the call it answers, so
        accepting it would degrade the model's own output rather than fail. `content` is kept
        mandatory by `_content_is_required_off_a_tool_call_turn`, which has to read the raw
        payload.
        """
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError(
                "a role='tool' message needs the tool_call_id of the call it answers; "
                "copy it from the assistant turn's tool_calls[].id"
            )
        if self.tool_call_id is not None and self.role != "tool":
            raise ValueError(
                f"tool_call_id is only valid on a role='tool' message, not on role="
                f"'{self.role}'; send the tool's output as a role='tool' message"
            )
        if self.tool_calls is not None and self.role != "assistant":
            raise ValueError(
                f"tool_calls are only valid on a role='assistant' message, not on role="
                f"'{self.role}'; replay the assistant turn the model returned them on"
            )
        return self

    def for_provider(self) -> ProviderChatMessage:
        """This turn as the provider-neutral structured message `complete_chat` takes."""
        return ProviderChatMessage(
            role=self.role,
            # An assistant turn that only calls tools has no text: send null rather than "", so
            # `ChatRequest.provider_payload` (exclude_none) drops the field entirely instead of
            # asking a backend to accept an empty content block.
            content=self.content if self.content or not self.tool_calls else None,
            tool_calls=self.tool_calls,
            tool_call_id=self.tool_call_id,
        )


class StreamOptions(BaseModel):
    """OpenAI `stream_options`: opt-in for the trailing usage chunk on streamed responses."""

    include_usage: bool = False


# Every `tool_choice` string OpenAI defines. A value outside this set, and outside the
# `{"type": "function", ...}` object shape, is refused rather than forwarded: no backend this
# endpoint routes to can honor one. OpenAI-compatible servers reject it, which surfaces as a 502
# that blames the model for a client mistake, and Bedrock drops the field and can answer in prose
# to a client that required a call. Forwarding an unrecognized word to stay
# forward-compatible sounds generous and in practice only converts a clear 400 into one of those
# two. When OpenAI adds a word, it is added here.
_TOOL_CHOICE_WORDS = ("none", "auto", "required")


def _tool_choice_is_recognized(tool_choice: JsonValue) -> bool:
    """Whether `tool_choice` is a value this endpoint can actually route (see `_TOOL_CHOICE_WORDS`).

    Note `tool_choice in _TOOL_CHOICE_WORDS` is deliberately not the whole test: `True` compares
    equal to `1` but not to any of these strings, so booleans and numbers fall through to False as
    they should.
    """
    return tool_choice in _TOOL_CHOICE_WORDS or _is_function_choice_shape(tool_choice)


def _tool_choice_demands_a_call(tool_choice: JsonValue) -> bool:
    """Whether a `tool_choice` can only be honored by actually calling a tool.

    `"none"` and `"auto"` are both satisfied by not calling one, so neither demands anything: an
    empty tool registry (the `tools: []` this endpoint normalizes away) is served as plain chat
    rather than refused, and `"none"` (literally "do not call tools") is never refused for having
    none. `"required"` and a `{"type": "function", ...}` object cannot be honored with nothing to
    choose from. Unrecognized values never reach here: they are refused up front by
    `_tool_choice_is_recognized`.

    Args:
        tool_choice: The request's `tool_choice` exactly as the client sent it, or None when it
            sent none.

    Returns:
        True when honoring the choice requires a tool call.
    """
    return tool_choice is not None and tool_choice not in ("auto", "none")


def _is_function_choice_shape(tool_choice: JsonValue) -> bool:
    """Whether `tool_choice` uses OpenAI's `{"type": "function", ...}` object shape.

    Separate from `_named_tool_choice` because the two answer different questions: this one says
    the client MEANT to name a single tool, that one says whether a name can actually be read out.
    A choice that is this shape but yields no name is malformed rather than permissive, and the
    difference decides whether it is refused or forwarded.
    """
    return isinstance(tool_choice, dict) and tool_choice.get("type") == "function"


def _named_tool_choice(tool_choice: JsonValue) -> str | None:
    """The function name an OpenAI `{"type": "function", ...}` tool_choice names.

    Args:
        tool_choice: The request's `tool_choice` exactly as the client sent it. Arbitrary JSON,
            because OpenAI's own vocabulary has grown (`"required"` postdates `"auto"`) and an
            unrecognized value is still forwarded rather than rewritten.

    Returns:
        The demanded function name, or None when the choice names no single function: the
        `"auto"`/`"none"`/`"required"` strings, and any shape no name can be read out of.
    """
    if not isinstance(tool_choice, dict) or tool_choice.get("type") != "function":
        return None
    function = tool_choice.get("function")
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    return name if isinstance(name, str) and name else None


class ChatCompletionRequest(BaseModel):
    """The OpenAI request subset the endpoint serves: text chat plus function tool calling.

    Unsupported FUNCTIONAL parameters are declared here so they can be rejected explicitly
    (see `unsupported_features`); genuinely unknown extra fields are ignored like OpenAI does.

    What is still rejected, and why each one is a 400 rather than a silent drop:

    * `n != 1`: the endpoint meters one request-log row and remembers one incumbent reply per
      call, so extra choices would be billed and cached against a transcript nobody sent.
    * `logprobs`: no provider seam returns per-token logprobs, and an absent `logprobs` field
      reads to a client as "this model has none" rather than "this endpoint cannot".
    * `response_format`: support would have to hold for every entry in the pool (the routed
      model is not the client's choice) and the text `complete` path cannot carry it at all, so
      a schema honored only on some routes is worse than a refusal a client can act on.

    `parallel_tool_calls` is the other kind of functional parameter: it is honored, forwarded to
    the provider by `_provider_request`. Declaring it is what makes that possible, since pydantic
    would otherwise ignore it as an unknown extra and the endpoint would hand a client that asked
    for one call at a time a multi-call turn. With no tools in play it needs no forwarding at all:
    no call can happen, so "one at a time" is already true.
    """

    model: str  # the ENDPOINT name
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    tools: list[ChatTool] | None = None
    tool_choice: JsonValue | None = None
    # Forwarded to the provider (see `_provider_request`), not dropped: an agent whose executor
    # answers one call per turn sets it to false and would otherwise get a multi-call turn back.
    parallel_tool_calls: bool | None = None
    # Declared only to be REJECTED with a clear 400 (see the class docstring).
    response_format: JsonValue | None = None
    n: int = 1
    logprobs: bool | None = None

    def output_budget(self) -> int:
        return self.max_completion_tokens or self.max_tokens or DEFAULT_MAX_TOKENS

    def wants_stream_usage(self) -> bool:
        return self.stream_options is not None and self.stream_options.include_usage

    @field_validator("tools", mode="after")
    @classmethod
    def _an_empty_tool_list_is_no_tools(cls, value: list[ChatTool] | None) -> list[ChatTool] | None:
        """`tools: []` advertises nothing, so it is not a tool request.

        Clients whose tool registry happens to be empty send the empty list, and forwarding it
        would send an empty `tools` array to a backend that rejects one: an upstream error on a
        request that is really plain chat.
        """
        return value or None

    def needs_tool_calling(self) -> bool:
        """Whether this request must be served through `complete_chat`.

        True for advertised tools AND for a transcript that merely REPLAYS them: the text
        `complete` seam takes role+content messages only, so an assistant `tool_calls` turn or a
        `role="tool"` result would be dropped or mangled on the way to the model even when the
        client stopped sending `tools` (some agents omit them once the loop is under way).
        `tool_choice` alone does not qualify: with nothing to choose from there is no call to
        make, so it is either satisfied by plain chat or a client mistake (see
        `unsatisfiable_tool_choice`), never a reason to demand a tool-calling backend.
        """
        return self.tools is not None or any(
            m.tool_calls or m.role == "tool" for m in self.messages
        )

    def unsatisfiable_tool_choice(self) -> str:
        """Why this request's `tool_choice` cannot be honored, '' when it can.

        Judged against the tools THIS request declares, never against the transcript: a mid-loop
        turn that replays an assistant `tool_calls` turn plus its results but stops re-sending
        `tools` (some agents do) still has nothing for a demanding choice to select, so keying the
        check on the presence of a tool transcript let exactly that request through to the provider
        as `tool_choice` with `tools: null`. An OpenAI-compatible backend rejects that pair as a
        malformed request, which surfaces here as a 502 that reads as a model failure, and Bedrock
        Converse discards the required choice and answers in prose to a client that cannot use one.

        A named choice is checked by NAME for the same reason: no provider can call a function it
        was never given, so `{"type": "function", "function": {"name": "delete_rows"}}` alongside a
        `tools` array that declares only `lookup` is the same unhonorable request, one 400 earlier.

        Returns:
            A message naming what went wrong and what to send instead, or '' when the choice is
            serveable (including every non-demanding choice, see `_tool_choice_demands_a_call`).
        """
        if self.tool_choice is not None and not _tool_choice_is_recognized(self.tool_choice):
            words = ", ".join(f'"{word}"' for word in _TOOL_CHOICE_WORDS)
            return (
                f"tool_choice {self.tool_choice!r} is not a value this endpoint can route; use "
                f'{words}, or {{"type": "function", "function": {{"name": "<tool>"}}}}'
            )
        if not _tool_choice_demands_a_call(self.tool_choice):
            return ""
        if self.tools is None:
            return (
                "tool_choice needs tools: declare the tool definitions this call may choose "
                "from, or drop tool_choice"
            )
        named = _named_tool_choice(self.tool_choice)
        if named is None:
            if _is_function_choice_shape(self.tool_choice):
                # Function-SHAPED but nameless: `{"type": "function", "function": {}}` or an empty
                # name. It demands a call and identifies nothing to call, so no provider can honor
                # it. Forwarding it reaches an OpenAI-compatible backend as a malformed request,
                # which this endpoint surfaces as a 502 blaming the model, while Bedrock discards
                # the requirement and can answer in prose to a client that requires a call.
                return (
                    "tool_choice is a function choice with no usable `function.name`; give the "
                    'name of the tool it must call, or use "auto" or "required"'
                )
            # Not this shape at all: an unrecognized future vocabulary word. Forward it rather
            # than rewrite it, which is the same call `_tool_choice_demands_a_call` documents.
            return ""
        declared = [tool.function.name for tool in self.tools]
        if named in declared:
            return ""
        return (
            f"tool_choice names {named!r}, which this call does not declare; add its definition "
            f"to `tools` or name one of the tools you did declare: {', '.join(declared)}"
        )

    def unsupported_features(self) -> str:
        """Name the requested-but-unsupported params, '' when the request is serveable."""
        used = [
            name
            for name, value in (
                ("response_format", self.response_format),
                ("logprobs", self.logprobs),
            )
            if value
        ]
        if self.n != 1:
            used.append("n != 1")
        return ", ".join(used)


def _error_response(status_code: int, message: str, *, err_type: str, code: str) -> Response:
    """An OpenAI-shaped error body: real OpenAI clients read `body["error"]["message"]`.

    FastAPI's default `{"detail": ...}` shape parses as a generic APIStatusError in the openai
    SDK but loses the message; this shape surfaces it exactly like the upstream API does.
    """
    return Response(
        content=json.dumps(
            {"error": {"message": message, "type": err_type, "param": None, "code": code}},
            ensure_ascii=False,
        ),
        status_code=status_code,
        media_type="application/json",
    )


def install_openai_error_shapes(app: FastAPI) -> None:
    """Convert request-validation failures to OpenAI's 400 + error-body shape.

    Without this a malformed request gets FastAPI's 422 `{"detail": [...]}`, which OpenAI
    clients surface as an empty error. App-level because exception handlers cannot attach to
    a router; every app that mounts `create_chat_router` should call this.
    """

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, exc: RequestValidationError) -> Response:
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(part) for part in first.get("loc", []) if part != "body")
        message = f"invalid request: {location}: {first.get('msg', 'validation failed')}"
        return _error_response(
            400, message, err_type="invalid_request_error", code="invalid_request"
        )


class RequestLogRecord(BaseModel):
    """One metered call, as the request log persists it (D-METERING / D-SERVING-LOG shape).

    This is the wmo half of the metering contract: the platform wrap adds tenancy
    (org_id, api_key_id) when it persists these rows. `cached_tokens` mirrors
    `TokenUsage.cached_input_tokens` (cache-read prompt tokens, a subset of `input_tokens`);
    `cost_usd` is cache-adjusted via `PoolEntry.cost_usd`. `router_cost_usd` is the policy's
    OWN inference cost per call, 0 for the free hashing policy and real once a trained router
    serves.
    """

    id: str
    ts: str
    endpoint: str
    leg: Literal["serving", "optimization", "eval", "overhead"] = "serving"
    model: str  # routed pool entry name
    provider_model: str  # the provider runtime id behind it
    cluster_id: int | None = None
    cluster_label: str = ""
    routing_reason: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0  # cache-read prompt tokens (subset of input_tokens)
    # Cache-WRITE prompt tokens (subset of input_tokens, disjoint from
    # cached_tokens). Priced at the write premium by the entry's cost model;
    # persisted so a post-hoc reconstruction (savings counterfactuals)
    # decomposes the same tokens the price did.
    cache_write_tokens: int = 0
    cost_usd: float = 0.0  # effective cost: cached tokens billed at the cache-read rate
    router_cost_usd: float = 0.0  # the routing decision's own inference cost, passed through
    # D-COMPRESS fields: stored and OPAQUE like the routing fields above (log only, never in
    # response bodies or headers). 0/"" defaults = the request served uncompressed. Token
    # counts are the compressor's deterministic proxy totals (see wmo.optimize.compression);
    # billable truth stays in input_tokens/cost_usd from the provider-reported usage.
    tokens_in_raw: int = 0
    tokens_in_compressed: int = 0
    compressor_id: str = ""
    compressor_version: str = ""
    aggressiveness: float = 0.0
    # The compressor's OWN bill and wall clock, named as on `ScenarioOutcome` so eval and
    # serving rows read the same. Both are real money and real time the customer paid: the
    # track's rule is that every savings number is cache-adjusted effective cost per completed
    # task, compressor cost and latency INCLUDED, so a row that logged the token reduction
    # without these would overstate the saving (see `wmo.serving.savings.compute_savings`).
    compressor_cost_usd: float = 0.0
    compressor_latency_s: float = 0.0
    # Wall clock for the whole served request, compression stage included: the client waits for
    # the compressor's round trip too, so excluding it would make compression read as
    # latency-neutral no matter what it cost.
    latency_ms: float = 0.0
    ttfb_ms: float | None = None
    status: Literal["ok", "error"] = "ok"
    error_message: str | None = None
    # The routing decision's evidence, flattened (see `wmo.optimize.policy.RoutingEvidence`).
    # Flat rather than nested because this row is a JSONL contract the platform reads column-wise:
    # counting fallback-forced requests or histogramming n_pairs should not need a JSON path. All
    # nullable, and all null for the kinds that compute no paired evidence (static, rank) and for
    # requests the guard never reached.
    mean_diff: float | None = None
    se: float | None = None
    n_pairs: int | None = None
    gate: GateOutcome | None = None
    propensity: Propensity | None = None
    # Expected prompt-cache saving applied to the incumbent's effective input price by a
    # cache-aware kNN decision. Null when no cache credit entered routing arithmetic.
    cache_credit_usd: float | None = None
    # Resolves to the vector this request was routed on
    # (`wmo.serving.query_embeddings.QueryEmbeddingStore.get`). Null when the store is off, when
    # the policy embeds nothing (static), or when the write failed.
    query_embedding_ref: str | None = None


class RequestLog:
    """Append-only JSONL request log plus a bounded in-memory tail."""

    def __init__(self, path: Path | None, *, keep: int = 200) -> None:
        self._path = path
        self._recent: deque[RequestLogRecord] = deque(maxlen=keep)
        self._revision = 0
        self._lock = threading.Lock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: RequestLogRecord) -> None:
        with self._lock:
            self._recent.append(record)
            self._revision += 1
            if self._path is not None:
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(record.model_dump_json() + "\n")

    @property
    def revision(self) -> int:
        """Rows appended so far: a cheap "has anything changed" stamp for derived summaries."""
        with self._lock:
            return self._revision

    def recent(self) -> list[RequestLogRecord]:
        with self._lock:
            return list(self._recent)

    def replay(self, endpoint: str) -> list[RequestLogRecord]:
        """Every persisted row for `endpoint`, oldest first (the in-memory tail when no file).

        Read from disk rather than from the tail so a total computed over it survives a restart
        and covers more than the last few hundred calls. A row this build cannot parse (an older
        schema, a line truncated by a hard kill) is skipped: a savings figure that refuses to
        render because of one bad line is worse than one computed over the rest.
        """
        if self._path is None or not self._path.is_file():
            return [record for record in self.recent() if record.endpoint == endpoint]
        rows: list[RequestLogRecord] = []
        with self._path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = RequestLogRecord.model_validate_json(line)
                except ValueError:
                    logger.warning("skipping an unreadable request log row in %s", self._path)
                    continue
                if record.endpoint == endpoint:
                    rows.append(record)
        return rows


class EndpointRuntime:
    """One served endpoint: its policy, its providers, its affinity memory, its log.

    `cost_quality` sets the endpoint's cost/quality dial at mount time (see
    `wmo.optimize.knn.apply_cost_quality`); None serves the policy exactly as fitted, so
    mounting never silently re-tunes an artifact. `config_path` is the `endpoint.toml` a live
    dial change is persisted to, so the setting survives a restart; None keeps changes in memory
    (injected-policy tests, and any caller that owns persistence itself).

    `embeddings` is the query-vector sidecar (`wmo.serving.query_embeddings`); requests are
    recorded into it by default, and `log_query_embeddings=False` switches that off per endpoint
    without disturbing the request log. A store constructed on no path is already inert, which is
    what an in-memory serving setup gets.
    """

    def __init__(
        self,
        name: str,
        policy: RoutingPolicy,
        *,
        provider_factory: Callable[[PoolEntry], Provider] = pool_provider,
        log: RequestLog,
        cost_quality: float | None = None,
        config_path: Path | None = None,
        embeddings: QueryEmbeddingStore | None = None,
        log_query_embeddings: bool = True,
        pareto: ParetoCurve | None = None,
    ) -> None:
        self.name = name
        self.policy = policy
        # The measured curve written at optimize time (pareto.json); served verbatim on
        # GET /config. None for artifacts that predate it: absence, never an empty curve.
        self.pareto = pareto
        self.log = log
        self._embeddings = embeddings if log_query_embeddings else None
        self._base_policy = policy
        self._config_path = config_path
        self._provider_factory = provider_factory
        self._providers: dict[str, Provider] = {}
        self._policy_embedder: Embedder | None = None
        self._affinity: OrderedDict[str, str] = OrderedDict()
        # window -> (log revision the summary was computed at, the summary)
        self._savings: dict[SavingsWindow, tuple[int, EndpointSavings]] = {}
        # Compressed provider-visible transcripts, keyed by the SAME remembered-prefix
        # fingerprint as _affinity: the affinity state decides compression segment boundaries.
        # Only populated when the policy carries a compression config. Bounded by BYTES
        # (_COMPRESSED_CAPACITY_BYTES), so the value is (transcript, its measured size) and
        # `_compressed_bytes` is their running sum.
        self._compressed: OrderedDict[str, tuple[list[ChatMessage], int]] = OrderedDict()
        self._compressed_bytes = 0
        # Resolved at mount and re-resolved on every dial-driven policy install, mirroring the
        # embedder-once pattern: no per-request registry lookups. Resolving through the policy
        # re-runs the D-COMPRESS mount gates, so an artifact that was assembled in memory
        # (`model_copy`, which skips validators) cannot serve an unservable compressor either.
        self._compressor: Compressor | None = policy.serving_compressor()
        self._lock = threading.Lock()
        # Serializes dial changes end to end (persist + install); _lock alone only protects
        # the in-memory swap and would let two PUTs interleave file writes and installs.
        self._dial_lock = threading.Lock()
        if cost_quality is not None:
            self._install_policy(apply_cost_quality(self._base_policy, cost_quality))

    @property
    def cost_quality(self) -> float | None:
        """The dial the served policy is currently on (None: served as fitted)."""
        return self.policy.cost_quality

    def set_cost_quality(self, cost_quality: float) -> None:
        """Move the dial on the live endpoint, and persist it when there is a file to persist to.

        PERSIST FIRST, then swap, both under one dial lock: a failed write must leave the live
        endpoint exactly where it was (never a dial that evaporates on restart), and two
        overlapping PUTs must resolve to ONE (position, file) pair rather than interleaving a
        swap from one with the write from the other.

        In-flight requests keep the policy they started on: the swap replaces the whole policy
        object, so no request can ever read half of one dial position and half of another. The
        pool, baseline, and evidence bank are identical across positions, so a conversation that
        spans a swap is still served by a model this endpoint knows.

        The write is a read-modify-write, not a fresh config. `endpoint.toml` holds settings this
        endpoint never touches (today `[representation]` and the query-embedding switch), and
        writing a config built from the dial alone would DELETE them: the endpoint would keep
        serving, because the swap happens in memory, and the loss would only surface at the next
        mount, as a representation mismatch that fails the whole app. A dial change must not be
        able to do that.
        """
        adjusted = apply_cost_quality(self._base_policy, cost_quality)
        with self._dial_lock:
            if self._config_path is not None:
                settings = EndpointConfig.load(self._config_path)
                settings.model_copy(update={"cost_quality": cost_quality}).save(self._config_path)
            self._install_policy(adjusted)

    def _install_policy(self, adjusted: RoutingPolicy) -> None:
        # Resolved OUTSIDE the lock and before the swap: it re-runs the D-COMPRESS mount gates,
        # and a policy that fails them must leave the live endpoint exactly as it was rather
        # than half-installed.
        compressor = adjusted.serving_compressor()
        with self._lock:
            stale = not same_compression(self.policy.compression, adjusted.compression)
            self.policy = adjusted
            # Keep the resolved compressor matched to the policy object requests will read;
            # apply_cost_quality carries `compression` through, but the invariant should not
            # depend on that staying true.
            self._compressor = compressor
            if stale:
                # Every stored prefix was produced by the OUTGOING config, so reusing one would
                # hand the provider a transcript that is half one compression config and half
                # another. Today's dial carries `compression` through unchanged and this never
                # fires; it exists so that a dial which ever does vary compression cannot serve
                # a spliced transcript.
                self._clear_compressed()
            self._savings.clear()  # the dial changed, so the quality expectation did too

    def _clear_compressed(self) -> None:
        """Drop every stored compressed transcript. Caller holds `_lock`.

        The map and its running byte total are one piece of state; clearing them apart is how a
        byte-bounded cache starts evicting for memory it is no longer holding.
        """
        self._compressed.clear()
        self._compressed_bytes = 0

    def savings(self, window: SavingsWindow = "all_time") -> EndpointSavings:
        """What this endpoint has saved so far (see `wmo.serving.savings`).

        The all-time total is cached until the log grows or the dial moves, so a dashboard
        polling it does not re-read the whole JSONL on every paint, and the request that changes
        the total is the thing that invalidates it.

        A BOUNDED window is recomputed on every read and never cached, because its answer moves
        with the clock and not only with the log: rows age out of a 7-day window while nothing
        appends, so an idle endpoint would otherwise keep serving week-old traffic as this
        week's. Replay is a single sequential read of an append-only file, which is the cheap
        part of this call; the common case (the all-time card) still pays it once per new request.
        """
        revision = self.log.revision
        cacheable = window == "all_time"
        with self._lock:
            cached = self._savings.get(window) if cacheable else None
            if cached is not None and cached[0] == revision:
                return cached[1]
            policy = self.policy
        computed = compute_savings(self.log.replay(self.name), policy, window=window)
        if cacheable:
            with self._lock:
                if self.policy is policy:
                    # Store only if the dial has not moved since we captured the policy: a slow
                    # computation racing a dial swap must not resurrect the OLD dial's quality
                    # expectation under a revision the new dial also answers to.
                    self._savings[window] = (revision, computed)
        return computed

    def decide(
        self, messages: list[ChatMessage], *, route_text: str | None = None
    ) -> RoutingDecision:
        """Route the request.

        Stickiness keys on the RAW transcript the client resends; `route_text` (the compressed
        routable text when compression is on) is what gets embedded, so the router scores
        exactly what the model will see.
        """
        incumbent = None
        conversation_chars = 0
        remembered = _remembered_prefix(messages)
        if remembered is not None:
            with self._lock:
                incumbent = self._affinity.get(_fingerprint(remembered))
            if incumbent is not None:
                # The transcript the fingerprint matched IS the shared prefix a warm cache
                # would cover; its length feeds the cache-aware credit (chars/4 tokens,
                # documented in `cache_credit_usd`). Content only: tool payloads are part of
                # the prefix too, so this errs conservative, which is the designed direction.
                conversation_chars = sum(len(m.content or "") for m in remembered)
        text = route_text if route_text is not None else _routable_text(messages)
        return select_model(
            self.policy,
            text,
            incumbent=incumbent,
            embedder=self._embedder(),
            conversation_chars=conversation_chars,
        )

    def record_query_embedding(self, record_id: str, decision: RoutingDecision) -> str | None:
        """Persist the vector `decision` was routed on, returning the log row's ref (or None).

        None whenever there is nothing to record OR nothing to record it to: a static policy
        embeds no query, a sticky decision never consulted the policy, and the store is inert
        when logging is off. The caller writes the ref straight onto the log row, so those cases
        simply leave the column null.
        """
        vector = decision.query_embedding()
        if self._embeddings is None or vector is None:
            return None
        return self._embeddings.append(record_id, vector)

    def compress(
        self, messages: list[ChatMessage]
    ) -> tuple[list[ChatMessage], CompressionStats | None]:
        """The [compress] stage: raw request messages -> provider-visible messages + stats.

        Cache safety by construction: when the conversation's previous exchange is known
        (affinity hit on the remembered raw prefix), the stored compressed prefix is returned
        verbatim and only the turns appended since it pass through the compressor. On a miss
        (new conversation, or affinity evicted) every user message is compressed fresh;
        per-segment determinism makes that reproduce the same bytes, so the provider-visible
        prefix stays append-only either way. Returns the input list untouched when compression
        is off.

        At most ONE compressor call per request, carrying every segment that needs compressing:
        an endpoint-backed compressor pays one round trip per request, not one per message.
        """
        # One snapshot, one lock: `_install_policy` swaps the policy and its compressor together
        # under this lock, so reading them in two unsynchronized statements could pair a new
        # config with the previous implementation. Harmless while both dial positions share a
        # compressor, silently wrong the moment they do not.
        with self._lock:
            policy = self.policy
            compressor = self._compressor
        config = policy.compression
        if config is None or compressor is None:
            return messages, None
        started = time.monotonic()
        prefix: list[ChatMessage] | None = None
        remembered = _remembered_prefix(messages)
        if remembered is not None:
            key = _fingerprint(remembered)
            with self._lock:
                cached = self._compressed.get(key)
                if cached is not None:
                    self._compressed.move_to_end(key)  # LRU is by USE, not just by write
                    prefix = cached[0]
        if prefix is not None:
            # The stored prefix has one entry per remembered message, so the tail is everything
            # the client appended after the turn we already compressed.
            tail, cost_usd = _compress_user_turns(messages[len(prefix) :], compressor, config)
            compressed = [*prefix, *tail]
        else:
            compressed, cost_usd = _compress_user_turns(messages, compressor, config)
        return compressed, CompressionStats(
            compressor_id=compressor.id,
            compressor_version=compressor.version,
            aggressiveness=config.aggressiveness,
            tokens_in_raw=sum(estimate_tokens(m.content) for m in messages),
            tokens_in_compressed=sum(estimate_tokens(m.content) for m in compressed),
            latency_s=time.monotonic() - started,
            cost_usd=cost_usd,
        )

    def _embedder(self) -> Embedder | None:
        """Build the policy's embedder once per runtime, not once per request.

        Static policies never embed, so their (possibly unbuildable-in-this-environment)
        embedder spec must not be constructed at all: a static route has to keep serving
        even when an azure spec's credentials are absent here.

        An azure spec otherwise constructs a fresh SDK client (TLS handshake and all) inside
        every request's latency budget. Double-checked locking mirrors provider_for.
        """
        if self.policy.kind == "static" or self.policy.embedder is None:
            return None
        with self._lock:
            embedder = self._policy_embedder
        if embedder is None:
            embedder = self.policy.embedder.build()
            with self._lock:
                if self._policy_embedder is None:
                    self._policy_embedder = embedder
                embedder = self._policy_embedder
        return embedder

    def remember(
        self,
        messages: list[ChatMessage],
        reply: ChatMessage,
        model: str,
        *,
        compressed: list[ChatMessage] | None = None,
    ) -> None:
        """Record the finished exchange so the conversation's next request finds its incumbent.

        `reply` is the whole assistant turn, not just its text: a tool-calling turn carries
        empty content plus `tool_calls`, and the client replays it verbatim, so remembering the
        text alone would fingerprint a transcript that is never sent again and drop affinity at
        the first tool call.

        `messages` must be the RAW request messages (the client resends that transcript, so the
        fingerprint must match it). `compressed` is the provider-visible transcript when
        compression ran; stored under the same key so the next turn reuses the exact bytes the
        provider's prompt cache was written with.

        The two caches are bounded independently (entries for affinity, bytes for transcripts),
        so they can disagree about which conversations they remember. Both directions of
        disagreement are safe: a compressed prefix evicted while its affinity entry survives
        falls back to full recompression, which per-segment determinism makes byte-identical,
        and an affinity entry evicted while its prefix survives simply re-routes while still
        reusing the exact bytes that fingerprint was stored with.
        """
        transcript = [*messages, reply]
        key = _fingerprint(transcript)
        with self._lock:
            self._affinity[key] = model
            self._affinity.move_to_end(key)
            while len(self._affinity) > _AFFINITY_CAPACITY:
                self._affinity.popitem(last=False)
            if compressed is not None:
                transcript = [*compressed, reply]
                previous = self._compressed.pop(key, None)
                if previous is not None:
                    self._compressed_bytes -= previous[1]
                size = _transcript_bytes(transcript)
                self._compressed[key] = (transcript, size)
                self._compressed_bytes += size
                while self._compressed and self._compressed_bytes > _COMPRESSED_CAPACITY_BYTES:
                    _, (_, evicted) = self._compressed.popitem(last=False)
                    self._compressed_bytes -= evicted

    def provider_for(self, pool_name: str) -> tuple[PoolEntry, Provider]:
        entry = next(e for e in self.policy.pool if e.name == pool_name)
        with self._lock:
            provider = self._providers.get(pool_name)
        if provider is None:
            # Construct OUTSIDE the lock: a slow client build (TLS handshake, credential
            # resolution) must not head-of-line-block every other request's affinity lookup.
            # A racing duplicate build is harmless; first insert wins.
            provider = self._provider_factory(entry)
            with self._lock:
                provider = self._providers.setdefault(pool_name, provider)
        return entry, provider


def _transcript_bytes(messages: list[ChatMessage]) -> int:
    """Roughly what one stored transcript costs in memory, for the byte-bounded cache.

    Sums the UTF-8 length of everything that varies with conversation size: message content,
    tool-call ids/names/arguments, and tool_call_id. Tool arguments and results are counted
    because they are where transcripts actually get large. It is an estimate, not an allocator
    figure (it ignores per-object overhead, which is roughly constant per message), and it is
    used only to decide when to evict.
    """
    total = 0
    for message in messages:
        total += len(message.content.encode("utf-8")) + len(message.role)
        if message.tool_call_id is not None:
            total += len(message.tool_call_id)
        for call in message.tool_calls or ():
            total += len(call.id) + len(call.function.name)
            total += len(call.function.arguments.encode("utf-8"))
    return total


def _compress_user_turns(
    messages: list[ChatMessage], compressor: Compressor, config: CompressionConfig
) -> tuple[list[ChatMessage], float]:
    """Rewrite user-turn content through the compressor; every other turn passes through.

    v1 scope: system prompts (the most cacheable segment), the model's own replies, tool calls,
    and tool results are never touched. A tool payload is a structured contract the model has to
    read back exactly, so shortening it would change what the transcript MEANS, not just how
    long it is. Returns the rewritten messages plus the compressor's own cost.

    Goes through `compress_segments`, which chunks to the compressor's declared cap and enforces
    the return-shape contract. One request rarely carries enough user turns to need chunking,
    but a replayed transcript has no bound on its length and a wrong-length return here would
    otherwise desynchronize the whole conversation.
    """
    segments = [m.content for m in messages if m.role == "user"]
    if not segments:
        return list(messages), 0.0
    result = compress_segments(compressor, segments, config)
    replacements = iter(result.segments)
    rewritten = [
        m.model_copy(update={"content": next(replacements)}) if m.role == "user" else m
        for m in messages
    ]
    return rewritten, result.cost_usd


def _remembered_prefix(messages: list[ChatMessage]) -> list[ChatMessage] | None:
    """The prefix of `messages` that a previous reply would have been remembered under.

    `remember` stores `[*request, assistant_reply]`, so every remembered transcript ENDS on an
    assistant turn and the prefix to look up is the one that ends at the last assistant message,
    whatever the client appended after it. Stripping a fixed one message instead would only ever
    match a transcript that grew by exactly one, which a PARALLEL tool round trip never is: the
    client appends the assistant turn plus one `role="tool"` result per call, so a two-call turn
    would miss its incumbent and re-route mid-loop, forfeiting the prompt cache exactly when the
    transcript is longest and handing the new model tool_call ids the old one produced.

    None when there is nothing that could have been remembered: no assistant turn, or one at index
    0 (`remember` never stores a single-message transcript, since a request carries >= 1 message).
    """
    for index in range(len(messages) - 1, 0, -1):
        if messages[index].role == "assistant":
            return messages[: index + 1]
    return None


def _fingerprint(messages: list[ChatMessage]) -> str:
    """Hash a transcript for conversation affinity.

    Tool calls are part of a turn's identity. An assistant turn that only calls tools has empty
    content, so hashing (role, content) alone would give every branch out of one prefix the same
    key: two different tool calls, or two results for different `tool_call_id`s, would collide
    and the map would hand back an incumbent it never chose for that branch.
    """
    canonical = json.dumps(
        [
            (
                m.role,
                m.content,
                [call.model_dump(mode="json") for call in m.tool_calls] if m.tool_calls else None,
                m.tool_call_id,
            )
            for m in messages
        ],
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _routable_text(messages: list[ChatMessage]) -> str:
    """The text the router reads: this conversation's own request, never a tool result.

    A tool result is machine output the model asked for (a file dump, a stack trace), not what
    the customer asked for, so routing on it would send successive turns of ONE conversation to
    different models as the tool output varies and would feed the policy features it was never
    fitted on. The last user turn still describes the request, so a tool round trip keeps
    routing on it (and affinity pins the incumbent anyway).
    """
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    # No user turn at all (a replayed or synthetic transcript): fall back to the last turn that
    # carries the request's own words, still skipping tool results for the reason above.
    for message in reversed(messages):
        if message.role != "tool":
            return message.content
    return messages[-1].content


def _split_for_provider(messages: list[ChatMessage]) -> tuple[str, list[Message]]:
    """Fold system turns into the provider's system string; keep user/assistant order.

    Text path only: `Message` has no tool role, so a `needs_tool_calling` request never reaches
    here (it is served through `_provider_request` and `complete_chat` instead). A tool result
    that arrives anyway raises rather than being dropped, because dropping it would hand the
    model a transcript with the answer to its own call missing.
    """
    system_parts = [m.content for m in messages if m.role == "system"]
    turns: list[Message] = []
    for message in messages:
        match message.role:
            case "system":
                continue
            case "user" | "assistant":
                turns.append(Message(role=message.role, content=message.content))
            case "tool":
                raise ValueError(
                    "a tool result reached the text completion path; tool-bearing requests must "
                    "route through complete_chat (ChatCompletionRequest.needs_tool_calling)"
                )
    return "\n\n".join(system_parts), turns


def _provider_request(request: ChatCompletionRequest, messages: list[ChatMessage]) -> ChatRequest:
    """The structured request for a tool-bearing call, for `ToolCallingProvider.complete_chat`.

    `messages` is the provider-visible transcript, which is `request.messages` unless the
    compression stage rewrote it (tool calls and tool results ride through either way; only
    user-turn text is ever compressed).

    System turns stay INLINE here, unlike `_split_for_provider`: `complete_chat` takes the whole
    OpenAI-shaped transcript and each backend re-splits it for its own wire format (Bedrock
    Converse lifts system turns into `system`, tool results into `toolResult` blocks), so
    flattening them first would lose the order the client sent. `temperature` rides through as
    sent, None included: no field on the wire is both OpenAI's own default and what a reasoning
    model that rejects the parameter needs.

    `tool_choice` goes out only ALONGSIDE the tools it selects from. A replayed tool transcript
    reaches here with no `tools` at all, and `tool_choice` with `tools: null` is a malformed
    request to an OpenAI-compatible backend; a DEMANDING choice never gets this far (the route
    400s it), so the only one dropped here is an `"auto"`/`"none"` that no absent tool can
    contradict.

    `parallel_tool_calls` rides through as one of `ChatRequest`'s extras, which is where the
    providers already read it from (`wmo.providers._responses_common`) and where
    `ChatRequest.provider_payload` puts it on a chat-completions wire. Set only when the client
    sent it, so a pool entry whose backend rejects the field never sees it unasked.
    """
    provider_request = ChatRequest(
        messages=[m.for_provider() for m in messages],
        tools=request.tools,
        tool_choice=request.tool_choice if request.tools else None,
        temperature=request.temperature,
        max_completion_tokens=request.output_budget(),
    )
    extras = provider_request.model_extra
    if extras is not None and request.parallel_tool_calls is not None:
        # Written into `model_extra` rather than passed to the constructor: it is not a declared
        # field, so the constructor keeps type-checking the ones that are.
        extras["parallel_tool_calls"] = request.parallel_tool_calls
    return provider_request


def _structured_usage(response: ChatResponse) -> TokenUsage:
    """Real token usage from a structured response, cached-prompt split included.

    `ChatResponse.token_usage()` parses the provider's own `prompt_tokens_details` (cache
    reads bill at a discount and the request log prices each row with the split via
    `PoolEntry.cost_usd`); this projects it onto wmo's counters. A shape the parse cannot
    read costs the row its cached split (priced at the full input rate, never silently
    free), not the request.
    """
    counts = response.token_usage()
    return TokenUsage(
        input_tokens=counts.input_tokens,
        output_tokens=counts.output_tokens,
        cached_input_tokens=counts.cached_input_tokens,
        cache_write_input_tokens=counts.cache_write_input_tokens,
    )


def _reply_message(choice: ChatChoice) -> ChatMessage:
    """The assistant turn a client will replay, as this module's own message type.

    Validated rather than constructed so the provider's content (a string or text parts) goes
    through the same normalizer a client's messages do, and so a shape this endpoint cannot
    represent fails inside the upstream-call guard (a 502 plus its log row) instead of escaping
    as a bare 500.

    A null content becomes "" here instead of being refused: the mandatory-content rule catches a
    CLIENT sending an empty billed turn, while an upstream reply with neither text nor calls is
    still something this endpoint can render (it goes back out as content null), so rejecting it
    would invent a 502 for a response the client can read.
    """
    return ChatMessage.model_validate(
        {
            "role": "assistant",
            "content": choice.message.content if choice.message.content is not None else "",
            "tool_calls": choice.message.tool_calls,
        }
    )


# The finish reasons OpenAI's own schema declares; the SDK types the field as this exact set,
# so a value outside it is a value a typed client cannot represent.
_OPENAI_FINISH_REASONS = frozenset(
    {"stop", "length", "tool_calls", "content_filter", "function_call"}
)


def _finish_reason(upstream: str | None, *, has_tool_calls: bool) -> str:
    """Map a provider's finish reason onto OpenAI's vocabulary.

    `length` outranks `tool_calls`: a call cut off at the output cap has truncated (invalid)
    JSON arguments, and a client that reads "tool_calls" would parse them as complete. Anything
    outside OpenAI's set (Bedrock's `end_turn`, Anthropic's `stop_sequence`) reports `stop`
    rather than a value a typed client cannot represent.

    The reason is DERIVED from the reply in both directions: an upstream `tool_calls` on a turn
    that carries none reports `stop`. Reporting it verbatim is what a self-hosted backend whose
    tool parser failed to extract the call sends, and the standard agent loop
    (`if finish_reason == "tool_calls": for call in message.tool_calls`) raises TypeError on the
    null `tool_calls` that reply really has.
    """
    if upstream == "length":
        return "length"
    if has_tool_calls:
        return "tool_calls"
    if upstream == "tool_calls":
        return "stop"
    return upstream if upstream in _OPENAI_FINISH_REASONS else "stop"


def _message_payload(reply: ChatMessage) -> dict[str, JsonValue]:
    """One OpenAI response message: content null (not "") when the turn is tool calls only."""
    payload: dict[str, JsonValue] = {"role": "assistant", "content": reply.content or None}
    if reply.tool_calls:
        payload["tool_calls"] = [call.model_dump(mode="json") for call in reply.tool_calls]
    return payload


def _tool_call_deltas(tool_calls: list[ChatToolCall]) -> list[JsonValue]:
    """Streaming tool-call entries: `index`, `id`, and the whole `function.arguments` string.

    OpenAI splits `arguments` across many chunks and a client concatenates them per `index`;
    one complete fragment is that same contract with one fragment, which is the honest framing
    for a call the endpoint could not stream (see the module docstring).
    """
    return [
        {
            "index": index,
            "id": call.id,
            "type": call.type,
            "function": {"name": call.function.name, "arguments": call.function.arguments},
        }
        for index, call in enumerate(tool_calls)
    ]


def _usage_dict(usage: TokenUsage) -> dict[str, object]:
    return {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.input_tokens + usage.output_tokens,
        # OpenAI's cached-prompt reporting shape; 0 when the upstream provider reported none.
        "prompt_tokens_details": {"cached_tokens": usage.cached_input_tokens},
        # Anthropic's cache-write shape, only when one happened: a chained wmo
        # endpoint keeps pricing writes at the premium, while plain responses
        # stay free of nonstandard keys.
        **(
            {"cache_creation_input_tokens": usage.cache_write_input_tokens}
            if usage.cache_write_input_tokens > 0
            else {}
        ),
    }


def _sse(payload: object) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _chunk_payload(
    completion_id: str,
    created: int,
    endpoint: str,
    delta: dict[str, JsonValue],
    *,
    finish_reason: str | None = None,
) -> dict[str, object]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": endpoint,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _completion_body(
    completion_id: str,
    created: int,
    endpoint: str,
    message: dict[str, JsonValue],
    *,
    finish_reason: str,
    usage: TokenUsage,
) -> str:
    """One non-streamed `chat.completion` body, naming the ENDPOINT and never the routed model."""
    return json.dumps(
        {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": endpoint,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": _usage_dict(usage),
        },
        ensure_ascii=False,
    )


class ServedKnobs(BaseModel):
    """The knobs an endpoint is ACTUALLY serving, read off its policy.

    Same field names as the mapping's `CostQualityKnobs`, but `floor_q` is nullable: a policy can
    carry a novelty threshold whose quantile was never recorded (fitted before the field existed,
    or set by hand), and the honest answer there is null rather than a 0.0 that reads as "no
    floor". The threshold itself is not reported: it is a similarity number that means different
    things on different evidence banks, so it would tell a reader nothing they could act on.
    """

    knn_z: float
    floor_q: float | None
    pick_lam: float
    guard_mode: Literal["symmetric", "asymmetric"]


class EndpointConfigResponse(BaseModel):
    """The endpoint's cost/quality dial, everything needed to render it, and where it stands.

    `cost_quality` is null when the endpoint serves its policy exactly as fitted (nobody has set
    the dial), and `named_point` is "as-fitted" then: there is no dial position to label, and
    calling that "Custom" would imply someone chose it. Any position that IS set gets its
    anchor's label only when it sits exactly on that anchor, else "Custom".

    `knobs` is what the served policy is actually running, so a client can see the effect of the
    dial and not just its label. `anchors` are the MEASURED points behind the mapping
    (routerbench-ours9, 5 held-out splits, quality and cost both against the best single pool
    model), sorted by position: they are the ONLY deltas this response carries, because a delta
    quoted for an arbitrary position would read as a measurement of that position. A client that
    wants a curve interpolates between them itself.

    `dialable` is false for policy kinds with no dial (static and rank endpoints), and then the
    dial fields are null and PUT returns 409.

    `pareto` is the MEASURED cost/quality curve for THIS endpoint's workload
    (`wmo.optimize.pareto`, written as `pareto.json` beside the report at optimize time):
    every candidate and the routed dial detents on (effective cost per completed task,
    reward), frontier-flagged, with a recommended point. None for endpoints optimized before
    the artifact existed; a renderer shows nothing then, never an empty chart. Unlike
    `anchors` (a global table measured on routerbench-ours9), the curve is per-workload; the
    two must never be blended into one figure.
    """

    endpoint: str
    dialable: bool
    cost_quality: float | None
    named_point: str
    knobs: ServedKnobs | None
    anchors: list[CostQualityAnchor]
    pareto: ParetoCurve | None = None


class EndpointConfigUpdate(BaseModel):
    """A live dial change: the one field the platform's slider sends.

    `allow_inf_nan=False` because a slider bug that sends Infinity or NaN must be a 400 with a
    readable message, not a policy carrying an unusable knob (NaN fails every comparison the
    guard makes, so it would silently disable routing).
    """

    cost_quality: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


def _config_response(runtime: EndpointRuntime) -> EndpointConfigResponse:
    dialable = runtime.policy.kind == "knn"
    dial = runtime.cost_quality if dialable else None
    return EndpointConfigResponse(
        endpoint=runtime.name,
        dialable=dialable,
        cost_quality=dial,
        named_point=cost_quality_named_point(dial) if dial is not None else "as-fitted",
        knobs=(
            ServedKnobs(
                knn_z=runtime.policy.knn_z,
                # Straight off the policy, which records the quantile its threshold came from, so
                # an as-fitted endpoint reports the coverage setting it was FITTED with instead of
                # the dial's default. Null when that policy never recorded one.
                floor_q=runtime.policy.floor_q,
                pick_lam=runtime.policy.pick_lam,
                guard_mode=runtime.policy.guard_mode,
            )
            if dialable
            else None
        ),
        anchors=list(COST_QUALITY_ANCHORS),
        pareto=runtime.pareto,
    )


def create_chat_router(endpoints: Mapping[str, EndpointRuntime]) -> APIRouter:
    """Mount `/v1/models` and `/v1/chat/completions` plus the dial and savings routes."""
    router = APIRouter()

    def _endpoint_or_none(name: str) -> EndpointRuntime | None:
        return endpoints.get(name)

    def _endpoint_or_error(name: str) -> EndpointRuntime | Response:
        runtime = endpoints.get(name)
        if runtime is not None:
            return runtime
        available = ", ".join(sorted(endpoints)) or "(none)"
        return _error_response(
            404,
            f"no endpoint {name!r}; have: {available}",
            err_type="invalid_request_error",
            code="model_not_found",
        )

    @router.get("/v1/endpoints/{name}/config", response_model=EndpointConfigResponse)
    def get_endpoint_config(name: str) -> EndpointConfigResponse | Response:
        found = _endpoint_or_error(name)
        if isinstance(found, Response):
            return found
        return _config_response(found)

    @router.put("/v1/endpoints/{name}/config", response_model=EndpointConfigResponse)
    def put_endpoint_config(
        name: str, update: EndpointConfigUpdate
    ) -> EndpointConfigResponse | Response:
        found = _endpoint_or_error(name)
        if isinstance(found, Response):
            return found
        try:
            found.set_cost_quality(update.cost_quality)
        except ValueError as exc:
            # A dial the policy cannot honor: a non-knn kind, or a savings position on a policy
            # fitted without cost evidence. Both are configuration, not transport, so say which.
            return _error_response(
                409, str(exc), err_type="invalid_request_error", code="dial_unavailable"
            )
        return _config_response(found)

    @router.get("/v1/endpoints/{name}/savings", response_model=EndpointSavings)
    def get_endpoint_savings(
        name: str, window: SavingsWindow = "all_time"
    ) -> EndpointSavings | Response:
        """What this endpoint has saved so far, from its own request log (`?window=7d` for a week).

        Available for every policy kind, including static: a static endpoint has simply saved
        nothing yet, which is a truthful answer and the honest "before" state the improvement
        story is told against.
        """
        found = _endpoint_or_error(name)
        if isinstance(found, Response):
            return found
        return found.savings(window)

    @router.get("/v1/models")
    def list_models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "created": 0, "owned_by": "wmo"}
                for name in sorted(endpoints)
            ],
        }

    @router.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> Response:
        runtime = _endpoint_or_none(request.model)
        if runtime is None:
            available = ", ".join(sorted(endpoints)) or "(none)"
            return _error_response(
                404,
                f"no endpoint {request.model!r}; have: {available}",
                err_type="invalid_request_error",
                code="model_not_found",
            )
        unsupported = request.unsupported_features()
        if unsupported:
            # Silently dropping n/response_format/logprobs would make a client read a shape it
            # did not ask for: a compatibility gap disguised as a model-quality problem. Reject
            # loudly instead (see ChatCompletionRequest for why each one is still out).
            return _error_response(
                400,
                f"this endpoint does not support {unsupported} yet",
                err_type="invalid_request_error",
                code="unsupported_parameter",
            )
        unsatisfiable = request.unsatisfiable_tool_choice()
        if unsatisfiable:
            # Honoring it is impossible (the tool it demands is not on this request) and dropping
            # it would be a functional parameter silently ignored, so name the missing half here
            # rather than let the provider answer with a 502 or with prose (see
            # `unsatisfiable_tool_choice`).
            return _error_response(
                400,
                unsatisfiable,
                err_type="invalid_request_error",
                code="invalid_tool_choice",
            )
        if not any(m.role in ("user", "assistant") for m in request.messages):
            # Tool results do not count: a result with no assistant call to answer is a client
            # bug, and there would be nothing for the router to read (see `_routable_text`).
            return _error_response(
                400,
                "at least one user or assistant message is required",
                err_type="invalid_request_error",
                code="invalid_messages",
            )
        # Started BEFORE the compression stage, not after: the compressor's round trip is time
        # the client spends waiting, so a clock that skipped it would report compression as
        # latency-neutral however slow it was. `compressor_latency_s` breaks the stage out of
        # this total for anyone who needs the split.
        started = time.monotonic()
        # Bound before the try so the failure path can report a compression stage that already
        # RAN and already cost money before whatever failed next.
        compression: CompressionStats | None = None
        try:
            # request -> [compress] -> [route]: the router embeds the compressed text below.
            provider_messages, compression = runtime.compress(request.messages)
            decision = runtime.decide(
                request.messages, route_text=_routable_text(provider_messages)
            )
            entry, provider = runtime.provider_for(decision.model)
        except Exception as exc:  # noqa: BLE001 - reported as an OpenAI-shaped 502 + log row
            # The likeliest production failure: an unset api_key_env or a failing embed call.
            # Without this guard it surfaces as a bare text/plain 500 with no log row.
            logger.error("routing/provider setup for %s failed: %s", runtime.name, exc)
            runtime.log.append(
                RequestLogRecord(
                    id=f"chatcmpl-{uuid.uuid4().hex}",
                    ts=datetime.now(tz=UTC).isoformat(),
                    endpoint=runtime.name,
                    model="",
                    provider_model="",
                    routing_reason="error-before-routing",
                    # A compressor that ran before the failure was PAID for, on real hardware.
                    # Leaving these at zero would not be an omission, it would be a row
                    # asserting that no compression happened, and the savings math reads these
                    # rows. Still zero when the compression stage itself is what failed.
                    tokens_in_raw=compression.tokens_in_raw if compression else 0,
                    tokens_in_compressed=compression.tokens_in_compressed if compression else 0,
                    compressor_id=compression.compressor_id if compression else "",
                    compressor_version=compression.compressor_version if compression else "",
                    aggressiveness=compression.aggressiveness if compression else 0.0,
                    compressor_cost_usd=compression.cost_usd if compression else 0.0,
                    compressor_latency_s=compression.latency_s if compression else 0.0,
                    latency_ms=(time.monotonic() - started) * 1000,
                    status="error",
                    error_message=str(exc),
                )
            )
            return _error_response(
                502,
                f"endpoint setup failed ({type(exc).__name__})",
                err_type="api_error",
                code="routing_error",
            )
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        # Written once per request, keyed by the id every log row for this call carries, so the
        # vector is stored exactly once no matter which path below reports the outcome.
        embedding_ref = runtime.record_query_embedding(completion_id, decision)
        evidence = decision.evidence

        def _record(
            usage: TokenUsage,
            *,
            ttfb_ms: float | None,
            status: Literal["ok", "error"] = "ok",
            error_message: str | None = None,
        ) -> None:
            runtime.log.append(
                RequestLogRecord(
                    id=completion_id,
                    ts=datetime.now(tz=UTC).isoformat(),
                    endpoint=runtime.name,
                    model=entry.name,
                    provider_model=entry.model,
                    cluster_id=decision.cluster_id,
                    cluster_label=decision.cluster_label,
                    routing_reason=decision.reason,
                    mean_diff=evidence.mean_diff if evidence else None,
                    se=evidence.se if evidence else None,
                    n_pairs=evidence.n_pairs if evidence else None,
                    gate=evidence.gate if evidence else None,
                    propensity=evidence.propensity if evidence else None,
                    cache_credit_usd=evidence.cache_credit_usd if evidence else None,
                    query_embedding_ref=embedding_ref,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cached_tokens=usage.cached_input_tokens,
                    cache_write_tokens=usage.cache_write_input_tokens,
                    cost_usd=entry.call_cost_usd(usage),
                    router_cost_usd=decision.router_cost_usd(),
                    tokens_in_raw=compression.tokens_in_raw if compression else 0,
                    tokens_in_compressed=compression.tokens_in_compressed if compression else 0,
                    compressor_id=compression.compressor_id if compression else "",
                    compressor_version=compression.compressor_version if compression else "",
                    aggressiveness=compression.aggressiveness if compression else 0.0,
                    compressor_cost_usd=compression.cost_usd if compression else 0.0,
                    compressor_latency_s=compression.latency_s if compression else 0.0,
                    latency_ms=(time.monotonic() - started) * 1000,
                    ttfb_ms=ttfb_ms,
                    status=status,
                    error_message=error_message,
                )
            )

        headers = {"x-wmo-routed-model": decision.model}

        if request.needs_tool_calling():
            if not isinstance(provider, ToolCallingProvider):
                # The routed pool entry cannot carry tool calls at all. Dropping the tools and
                # answering in prose would read as a bad model; 501 + a log row says which entry
                # and what to change (mirrors the streaming capability gap below).
                detail = (
                    f"pool model '{entry.name}' (kind '{entry.kind.value}') cannot serve tool "
                    "calls: its provider has no structured chat backend. Retry without `tools`, "
                    "or give this endpoint a pool whose entries all support tool calling"
                )
                _record(TokenUsage(), ttfb_ms=None, status="error", error_message=detail)
                return _error_response(
                    501, detail, err_type="api_error", code="tool_calling_unsupported"
                )
            # Zero until the call comes back, then the provider's own counts, so a response that
            # arrived and cannot be USED still meters what it billed. Zero choices is the normal
            # shape of an upstream content filter, which charges for the prompt it read; recording
            # TokenUsage() there would under-report the row and its cost as free, the same silent
            # usage loss the abandoned-stream path exists to prevent (D-METERING).
            usage = TokenUsage()
            try:
                structured = provider.complete_chat(_provider_request(request, provider_messages))
                usage = _structured_usage(structured)
                if not structured.choices:
                    # Content filtering (and some provider error modes) return zero choices.
                    raise ValueError(f"{entry.model} returned no choices")
                choice = structured.choices[0]
                reply = _reply_message(choice)
                finish_reason = _finish_reason(
                    choice.finish_reason, has_tool_calls=bool(reply.tool_calls)
                )
            except Exception as exc:  # noqa: BLE001 - reported as an OpenAI-shaped 502
                # Same split as the text path: detail to the log, class name to the client.
                _record(usage, ttfb_ms=None, status="error", error_message=str(exc))
                logger.error("tool-calling call for %s failed: %s", entry.name, exc)
                return _error_response(
                    502,
                    f"upstream model call failed ({type(exc).__name__})",
                    err_type="api_error",
                    code="upstream_error",
                )
            runtime.remember(
                request.messages,
                reply,
                decision.model,
                compressed=provider_messages if compression else None,
            )
            if not request.stream:
                _record(usage, ttfb_ms=None)
                return Response(
                    content=_completion_body(
                        completion_id,
                        created,
                        runtime.name,
                        _message_payload(reply),
                        finish_reason=finish_reason,
                        usage=usage,
                    ),
                    media_type="application/json",
                    headers=headers,
                )

            # Re-emitted, not forwarded (module docstring): the upstream call is already
            # finished, so the row lands here and cannot be lost to a client disconnect, and
            # ttfb is the whole upstream latency because that is when the first byte can go out.
            _record(usage, ttfb_ms=(time.monotonic() - started) * 1000)

            def _tool_events() -> Iterator[str]:
                yield _sse(
                    _chunk_payload(
                        completion_id, created, runtime.name, {"role": "assistant", "content": ""}
                    )
                )
                if reply.content:
                    yield _sse(
                        _chunk_payload(
                            completion_id, created, runtime.name, {"content": reply.content}
                        )
                    )
                if reply.tool_calls:
                    yield _sse(
                        _chunk_payload(
                            completion_id,
                            created,
                            runtime.name,
                            {"tool_calls": _tool_call_deltas(reply.tool_calls)},
                        )
                    )
                yield _sse(
                    _chunk_payload(
                        completion_id, created, runtime.name, {}, finish_reason=finish_reason
                    )
                )
                if request.wants_stream_usage():
                    yield _sse(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": runtime.name,
                            "choices": [],
                            "usage": _usage_dict(usage),
                        }
                    )
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                _tool_events(),
                media_type="text/event-stream",
                headers={**headers, "Cache-Control": "no-cache"},
            )

        system, turns = _split_for_provider(provider_messages)
        if not request.stream:
            try:
                completion = provider.complete(
                    system,
                    turns,
                    temperature=request.temperature if request.temperature is not None else 1.0,
                    max_tokens=request.output_budget(),
                )
            except Exception as exc:  # noqa: BLE001 - reported as an OpenAI-shaped 502
                # Full detail goes to the request log and server log only: upstream exception
                # text can carry internal endpoints/stack info (CodeQL: information exposure).
                _record(TokenUsage(), ttfb_ms=None, status="error", error_message=str(exc))
                logger.error("upstream call for %s failed: %s", entry.name, exc)
                return _error_response(
                    502,
                    f"upstream model call failed ({type(exc).__name__})",
                    err_type="api_error",
                    code="upstream_error",
                )
            runtime.remember(
                request.messages,
                ChatMessage(role="assistant", content=completion.text),
                decision.model,
                compressed=provider_messages if compression else None,
            )
            _record(completion.usage, ttfb_ms=None)
            return Response(
                content=_completion_body(
                    completion_id,
                    created,
                    runtime.name,
                    {"role": "assistant", "content": completion.text},
                    finish_reason="stop",
                    usage=completion.usage,
                ),
                media_type="application/json",
                headers=headers,
            )

        if not isinstance(provider, StreamingProvider):
            # A real endpoint-level failure: keep one-record-per-request intact.
            _record(
                TokenUsage(),
                ttfb_ms=None,
                status="error",
                error_message=f"pool model '{entry.name}' has no native streaming backend",
            )
            return _error_response(
                501,
                f"pool model '{entry.name}' has no native streaming backend",
                err_type="api_error",
                code="streaming_unsupported",
            )
        try:
            upstream = provider.stream(
                system,
                turns,
                temperature=request.temperature if request.temperature is not None else 1.0,
                max_tokens=request.output_budget(),
            )
            first = next(upstream, None)
        except Exception as exc:  # noqa: BLE001 - reported as an OpenAI-shaped 502
            _record(TokenUsage(), ttfb_ms=None, status="error", error_message=str(exc))
            logger.error("stream start for %s failed: %s", entry.name, exc)
            return _error_response(
                502,
                f"upstream model call failed ({type(exc).__name__})",
                err_type="api_error",
                code="upstream_error",
            )
        ttfb_ms = (time.monotonic() - started) * 1000

        # Shared with _finalize: a disconnecting client makes starlette CANCEL the stream
        # without ever closing the sync generator, so cleanup inside the generator (finally,
        # GeneratorExit) never runs on that path. The BackgroundTask below is the only hook
        # that fires on both normal completion and disconnect. The generator appends every
        # delta it emitted to `streamed_parts` so the disconnect path can estimate the
        # partial generation it never got a usage chunk for.
        stream_state: dict[str, object] = {"recorded": False, "usage": None}
        streamed_parts: list[str] = []
        # The probe chunk was already pulled off the upstream before the response body
        # exists, so it is generated (and billed) even when the client disconnects before
        # the generator ever runs; seed it so the estimate counts it exactly once.
        if first is not None and first.delta:
            streamed_parts.append(first.delta)

        def _events() -> Iterator[str]:
            yield _sse(
                _chunk_payload(
                    completion_id, created, runtime.name, {"role": "assistant", "content": ""}
                )
            )
            parts = streamed_parts
            usage = TokenUsage()
            try:
                chunk = first
                is_probe = True  # the probe chunk's delta is pre-seeded above
                while chunk is not None:
                    if chunk.done:
                        if chunk.usage is not None:
                            usage = chunk.usage
                            # Stash for _finalize: a client that closes on
                            # [DONE] without reading EOF must still get its
                            # EXACT usage recorded, never an estimate.
                            stream_state["usage"] = usage
                    elif chunk.delta:
                        if not is_probe:
                            parts.append(chunk.delta)
                        yield _sse(
                            _chunk_payload(
                                completion_id,
                                created,
                                runtime.name,
                                {"content": chunk.delta},
                            )
                        )
                    is_probe = False
                    chunk = next(upstream, None)
            except Exception as exc:  # noqa: BLE001 - response already started; log and end
                stream_state["recorded"] = True
                _record(usage, ttfb_ms=ttfb_ms, status="error", error_message=str(exc))
                logger.error("stream from %s failed mid-response: %s", entry.name, exc)
                yield "data: [DONE]\n\n"
                return
            # Record BEFORE the closing yields: openai-python clients stop
            # reading at [DONE], so starlette can cancel this generator
            # anywhere in the tail — metering must already be done by then.
            stream_state["recorded"] = True
            _record(usage, ttfb_ms=ttfb_ms)
            try:
                runtime.remember(
                    request.messages,
                    ChatMessage(role="assistant", content="".join(parts)),
                    decision.model,
                    compressed=provider_messages if compression else None,
                )
            except Exception:  # noqa: BLE001 - affinity is best-effort; the response is not
                logger.exception("conversation affinity update failed for %s", runtime.name)
            yield _sse(
                _chunk_payload(completion_id, created, runtime.name, {}, finish_reason="stop")
            )
            if request.wants_stream_usage():
                # OpenAI's include_usage framing: one extra chunk with NO choices and the
                # final usage, after the finish_reason chunk and before [DONE].
                yield _sse(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": runtime.name,
                        "choices": [],
                        "usage": _usage_dict(usage),
                    }
                )
            yield "data: [DONE]\n\n"

        def _finalize() -> None:
            """Runs after the response ends, HOWEVER it ends (starlette BackgroundTask).

            An abandoned stream still consumed upstream tokens: leaving it unrecorded would
            be silent usage loss (D-METERING). The provider's usage arrives only in the
            terminal chunk the client never waited for, so the row carries a chars/4
            estimate of what actually went out and came back — the same documented,
            conservative proxy the cache-credit estimate uses — and names itself as one.
            An estimate that errs low still beats the zero-token row an earlier version
            wrote here, which billed the whole partial generation at $0. Also closes the
            upstream iterator, which the cancelled threadpool iteration otherwise leaks.
            """
            close = getattr(upstream, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()
            if not stream_state["recorded"]:
                stream_state["recorded"] = True
                exact = stream_state["usage"]
                if isinstance(exact, TokenUsage):
                    # The terminal chunk DID arrive before the disconnect; the
                    # estimate is for when it did not.
                    _record(
                        exact,
                        ttfb_ms=ttfb_ms,
                        status="error",
                        error_message="client disconnected before [DONE]; exact usage recorded",
                    )
                    return
                sent_chars = sum(
                    len(message.content) for message in provider_messages if message.content
                )
                streamed_chars = sum(len(part) for part in streamed_parts)
                _record(
                    TokenUsage(
                        input_tokens=sent_chars // 4,
                        output_tokens=streamed_chars // 4,
                    ),
                    ttfb_ms=ttfb_ms,
                    status="error",
                    error_message=(
                        "client disconnected mid-stream; usage is a chars/4 estimate of the "
                        "partial generation (the provider reports exact counts only in the "
                        "terminal chunk, which never arrived)"
                    ),
                )

        return StreamingResponse(
            _events(),
            media_type="text/event-stream",
            headers={**headers, "Cache-Control": "no-cache"},
            background=BackgroundTask(_finalize),
        )

    return router
