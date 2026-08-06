# Demo: Trace → Eval (OpenTelemetry JSONL → fidelity eval + CSV/JSON logs)

This example ships a small synthetic `traces.otel.jsonl` corpus and demonstrates the minimal
"trace → eval" pipeline implemented in:

- `wmh/engine/trace_to_eval.py`

The pipeline reuses the harness' existing eval orchestration:

- ingest OpenTelemetry GenAI traces → normalized `Trace`/`Step` types (`wmh.ingest.otel_genai`)
- run open-loop replay + judge scoring (`wmh.engine.eval` → `wmh.engine.replay`)
- persist a structured JSON report and a flat CSV with one row per scored step

## Trace format

The file is **OpenTelemetry JSONL**, where each line is a single span-like JSON object.

Required semconv keys (as used by `wmh.ingest.otel_genai.OtelGenAIAdapter`):

- For LLM/agent spans (e.g. `chat`):
  - `gen_ai.operation.name`: one of `chat`, `text_completion`, `invoke_agent`, `generate_content`
  - `gen_ai.tool.name` (tool call case) and `gen_ai.tool.call.arguments` (stringified JSON)
  - or `gen_ai.prompt` / `gen_ai.completion` (message case)
- For tool execution spans (e.g. `execute_tool`):
  - `gen_ai.operation.name`: `execute_tool`
  - tool output attribute such as `gen_ai.tool.message` (used as the observation text)

Optional enrichments:
- `wmh.state.structured`, `wmh.state.scratchpad` on the LLM span
- `wmh.trace.metadata` on any span in the trace

See `wmh/ingest/otel_genai.py` for the complete mapping.

## Run the pipeline (library usage)

The pipeline is designed to be driven from Python, keeping PR scope small.

```bash
uv run python -c "\
from pathlib import Path\
\
from wmh.engine.trace_to_eval import TraceToEvalConverter\
from wmh.optimize.judge import RubricJudge\
from wmh.providers import get_provider\
from wmh.providers.base import ProviderConfig, ProviderKind\
\
# Synthetic traces shipped with this example\
traces = [Path('examples/demo-trace-eval/traces.otel.jsonl').resolve()]\
\
# Replace the provider with the credentials you have configured.\
provider = get_provider(ProviderConfig(kind=ProviderKind.BEDROCK, model='us.anthropic.claude-opus-4-8', region='us-east-1'))\
\
judge = RubricJudge(provider)\
\
conv = TraceToEvalConverter(rag_enabled=False, results_root='.wmh/evals')\
run = conv.run(\
    trace_files=traces,\
    prompt='BASE',\
    provider=provider,\
    judge=judge,\
    run_id='demo-trace-eval',\
    suite_id=None,\
)\
\
print(run.artifacts.json_path)\
print(run.artifacts.csv_path)\
"
```

Notes:
- The example uses `prompt='BASE'` because the converter only routes the prompt string into
  replay; in a real run you likely want `wmh.engine.prompts.BASE_ENV_PROMPT` or an optimized prompt.
- For CI/unit tests, the converter is tested with fake providers (no network).

## What to look at

After running:

- JSON: `.wmh/evals/<suite>/<run_id>.json`
- CSV:  `.wmh/evals/<suite>/<run_id>.csv`

Each CSV row corresponds to a scored `StepResult` (predicted observation vs. recorded
observation).

