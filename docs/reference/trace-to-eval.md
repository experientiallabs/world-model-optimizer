# Trace → Eval (OpenTelemetry JSONL → fidelity CSV/JSON)

This document explains the `wmh.engine.trace_to_eval` feature: a small integration layer that
turns **OpenTelemetry GenAI trace JSONL** files into **fidelity-eval runs**.

It reuses the harness’ existing evaluation pipeline so results are consistent with other
`wmh eval` workflows.

## Problem it solves

The harness already supports:
- ingesting OpenTelemetry GenAI traces (`wmh.ingest.otel_genai`)
- replaying held-out steps open-loop and scoring reconstruction fidelity
  (`wmh.engine.eval` + `wmh.engine.replay`)

However, converting a trace corpus into artifacts researchers can inspect is still
custom code for each dataset.

`TraceToEvalConverter` provides a small, reusable layer that:
- ingests trace files
- runs the canonical fidelity eval orchestration
- writes:
  - a structured JSON report (per-file replay report + overall summary)
  - a flat CSV (one row per scored step)

## Where it lives

- Implementation: `wmh/engine/trace_to_eval.py`
- Example dataset: `examples/demo-trace-eval/traces.otel.jsonl`
- Unit tests: `wmh/engine/trace_to_eval_test.py`

## Python API

### Minimal example

```python
from pathlib import Path

from wmh.engine.trace_to_eval import TraceToEvalConverter
from wmh.providers import get_provider
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.optimize.judge import RubricJudge

traces = [Path("examples/demo-trace-eval/traces.otel.jsonl").resolve()]

provider = get_provider(
    ProviderConfig(
        kind=ProviderKind.BEDROCK,
        model="us.anthropic.claude-opus-4-8",
        region="us-east-1",
    )
)

judge = RubricJudge(provider)

converter = TraceToEvalConverter(rag_enabled=False, results_root=".wmh/evals")
run = converter.run(
    trace_files=traces,
    prompt="BASE",
    provider=provider,
    judge=judge,
    run_id="demo-trace-eval",
    suite_id=None,
)

print(run.artifacts.json_path)
print(run.artifacts.csv_path)
```

### What happens under the hood

`TraceToEvalConverter.run(...)` routes to the canonical fidelity eval orchestration:
- ingest + normalization: `wmh.ingest.get_adapter(adapter_name).from_file(...)`
- split into train/holdout: `wmh.engine.build.split_traces`
- replay + score: `wmh.engine.eval.evaluate_files(...)` → `wmh.engine.replay.replay(...)`

The converter itself focuses only on:
- validating inputs
- selecting artifact output locations
- flattening per-step results into CSV

## CLI usage

This PR currently exposes the pipeline through the Python API (via the converter). If you want
CLI support in a follow-up, keep the command surface small and route through the same converter.

## Input format: OTel GenAI semantic-convention JSONL

Input is **OpenTelemetry JSONL**, where each line is a span-like JSON object.

This example uses the minimal structure supported by
`wmh.ingest.otel_genai.OtelGenAIAdapter`.

### Required semconv keys

For **LLM / agent spans** (e.g. `name = "chat"`):
- `gen_ai.operation.name` : one of `chat`, `text_completion`, `invoke_agent`, `generate_content`
- either:
  - `gen_ai.tool.name` and `gen_ai.tool.call.arguments` (tool-call style), or
  - `gen_ai.prompt` / `gen_ai.completion` (message style)

For **tool execution spans** (e.g. `name = "execute_tool"`):
- `gen_ai.operation.name` : `execute_tool`
- a tool output attribute such as `gen_ai.tool.message` (used as the observation text)

### Optional enrichments

- `wmh.state.structured` and/or `wmh.state.scratchpad` on an action (LLM) span:
  - stored as `Step.state_before`
- `wmh.trace.metadata` on any span:
  - stored as `Trace.metadata`

### How traces are converted into steps

The adapter orders spans by `startTimeUnixNano` and pairs:
- an LLM/action span with the *next* tool execution span into one `Step`.

This means a single trace is typically represented by:
- 1+ LLM spans (action)
- 1+ tool spans (observation)

If a trace’s final LLM span has no following tool span, the adapter still emits a `Step` but
with an empty observation.

## Output format

`TraceToEvalConverter` writes outputs into:

- `results_root/<suite_id or "ad-hoc">/<run_id>.json`
- `results_root/<suite_id or "ad-hoc">/<run_id>.csv`

### JSON

The JSON report contains:
- run metadata (`run_id`, `started_at`, config)
- `report`: the structured `EvalReport` produced by `wmh.engine.eval`

### CSV columns

CSV is flattened to one row per scored `StepResult` and includes:
- `trace_id`
- `task`
- `action` (rendered action string)
- `actual` (recorded observation content)
- `predicted` (model predicted observation content)
- `score` (judge fidelity score)
- `is_error_actual`
- `is_error_predicted`
- `critique`
- `dimensions` (JSON-encoded rubric dimensions)

## Extending to custom trace sources

To support new trace sources:
1. implement a new `TraceAdapter` under `wmh/` (or add a new adapter in `wmh/ingest/`)
2. register it via `wmh.ingest.adapter.register_adapter(...)`
3. call `TraceToEvalConverter(adapter_name="<your-adapter>", ...)`

The converter intentionally does not interpret trace structure beyond selecting the adapter and
reusing the existing eval orchestration.

## Notes on train/holdout split and retrieval

The eval pipeline splits traces deterministically using `wmh.engine.build.split_traces`.

If RAG is enabled (`rag_enabled=True`), replay uses a leak-free demo retriever:
- demos come from the TRAIN split only
- the held-out trace’s own prior steps are excluded from demos (by design)

This keeps the fidelity score focused on next-step prediction quality.

