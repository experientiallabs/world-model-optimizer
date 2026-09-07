# Local trace input

`exp build PROJECT --traces PATH --source SOURCE` reads one explicit local corpus through one canonical
loader. Each source is declared, never guessed:

| `--source` | Local input |
|---|---|
| `otlp` | OpenTelemetry JSON or JSONL using the supported GenAI span mapping. |
| `otel-genai` | Exported flat GenAI span records, re-encoded into OTLP and read by the same mapping. |
| `posthog` | PostHog LLM-observability export. |
| `braintrust` | Braintrust log rows or an `events`, `rows`, `data`, `results`, or `items` envelope. |
| `langfuse` | Langfuse traces with observations, or bare observations carrying `traceId`. |
| `langsmith` | LangSmith runs, or a `runs` envelope. |
| `mastra` | Mastra spans, or a `spans` envelope. |
| `phoenix` | Phoenix and OpenInference spans, native nested, flat dotted, or OTLP JSON. |
| `chat-json` | OpenAI-style chat conversations, one object, an array, or bare message arrays. |
| `datadog` | Datadog LLM Observability spans, a v0.4 array of trace arrays, or a `spans` envelope. |

Every file source accepts JSON or JSONL. A malformed JSONL line is never skipped silently: it is
retained as an explicit normalization issue. Every normalized trace keeps the immutable source
identity and the exact source-byte digest, the original source trace and span identifiers in
`exp.source.trace.id` and `exp.source.span.id`, declared parent relationships when they are
unambiguous, and declared model evidence. Opaque vendor identifiers map to deterministic
W3C-shaped identifiers, so the canonical identity is stable while the source identity stays
readable. Provider and model identity resolves only when the export declares both; a model named
without a provider stays as `gen_ai.request.model` evidence and is never completed by inference.
Tool results pair with tool calls by explicit call identifier, falling back to tool name and source
order only when the export declares no identifier.

A Python caller reaches the same seam by name instead of importing one loader per vendor:

```python
from exp.simulation.ingest import CANONICAL_TRACE_SOURCES, load_trace_source

result = load_trace_source("langfuse", Path("export.jsonl"))
```

The source table is explicit, so an undeclared name fails closed rather than being detected.

## Stored evidence

Raw exports remain at the customer path. Experiential stores a normalized immutable snapshot and explicit
normalization issues under the selected project. Public build accepts 100 through 1000 valid
normalized traces after validation and source deduplication. The limit applies to normalized traces,
not to the smaller representative task count that semantic deduplication and mining produce.

No generic vendor-adapter registry and no format detection are part of the command. Build does not
pull a remote source, propose a rubric, run a judge, or call the selected world model. Representative
task selection uses the deterministic local hashing embedder unless a Python caller supplies another
explicit descriptor embedder.

After local trace and split construction, build shows the selected world model and embedder, the
conservative embedding-cost ceiling, and the configured maximum. An estimate over that maximum
fails before credentials or provider construction. An estimate within the maximum runs without a
confirmation prompt, builds the serving and fit-only RAG indexes, and binds the grounded world model.
`--dry-run` shows the same preflight with no provider call and no completed-build selection. Exact
replay verifies and reuses completed indexes with no provider call. Anonymous aggregate PostHog
product telemetry may send after successful persistence unless the user runs
`exp config telemetry disable`. Telemetry does not include prompts, traces, paths, model names, or
customer content.

## Authorized PostHog HogQL pull

The CLI deliberately reads only local exports. An authorized Python caller may instead import
`PostHogPullRequest` and `pull_posthog_traces` from `exp.simulation.ingest`. The request defaults to
a bounded 1,000-row query and may set another positive limit through 10,000. The caller may inject a
deterministic HTTP client; otherwise the function owns one bounded `httpx.Client`. The HTTPS host
and credential are explicit request values or resolved from the focused PostHog environment
settings. The query orders by `timestamp, uuid`, applies the same canonical converter as local
export ingestion, and returns normalized traces plus retained issues. This does not weaken the
`exp build` local-file boundary and requires separate authorization for the customer PostHog
project.
