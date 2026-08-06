"""Tests for the trace-to-eval converter.

The converter is intentionally thin: it calls `wmh.engine.eval.evaluate_files` and then
serializes `EvalReport` into JSON + CSV.

So these tests validate:
- output files are created
- CSV has one row per StepResult
- the CSV fields come from the underlying `ReplayReport` step results

We use fakes for provider + judge so the tests do not call any real LLM.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from pydantic import BaseModel

from wmh.core.types import Observation, Step
from wmh.engine.trace_to_eval import TraceToEvalConverter
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.optimize.judge import JudgeResult


class FakeProvider:
    def __init__(self, reply: str) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")
        self._reply = reply

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        return Completion(text=self._reply)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


class FakeJudge:
    def __init__(self, score: float) -> None:
        self._score = score

    def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
        return JudgeResult(score=self._score, critique="ok")


def _write_otel_jsonl(path: Path, *, n_traces: int) -> None:
    # Minimal OTLP-JSON-ish payload per line containing enough semconv for the adapter.
    # We purposely mirror the structure used in `wmh/engine/eval_test.py`.
    lines: list[str] = []
    for t in range(n_traces):
        tid = f"{t:032x}"
        lines.append(
            json.dumps(
                {
                    "traceId": tid,
                    "spanId": f"{tid[:8]}0000",
                    "name": "chat",
                    "startTimeUnixNano": 1,
                    "attributes": [
                        {
                            "key": "gen_ai.operation.name",
                            "value": {"stringValue": "chat"},
                        },
                        {
                            "key": "gen_ai.tool.name",
                            "value": {"stringValue": "get_user"},
                        },
                        {
                            "key": "gen_ai.tool.call.arguments",
                            "value": {"stringValue": json.dumps({"i": t})},
                        },
                        {
                            "key": "gen_ai.prompt",
                            "value": {"stringValue": f"look up u{t}"},
                        },
                    ],
                }
            )
        )
        lines.append(
            json.dumps(
                {
                    "traceId": tid,
                    "spanId": f"{tid[:8]}0001",
                    "name": "execute_tool",
                    "startTimeUnixNano": 2,
                    "attributes": [
                        {
                            "key": "gen_ai.operation.name",
                            "value": {"stringValue": "execute_tool"},
                        },
                        {
                            "key": "gen_ai.tool.message",
                            "value": {"stringValue": f"found u{t}"},
                        },
                    ],
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_converter_writes_json_and_csv_and_has_one_row_per_step(tmp_path: Path) -> None:
    corpus = tmp_path / "trace.otel.jsonl"
    _write_otell_jsonl(corpus, n_traces=3)

    out_dir = tmp_path / "out"
    converter = TraceToEvalConverter(results_root=out_dir)

    run = converter.run(
        trace_files=[corpus],
        prompt="BASE",
        provider=FakeProvider('{"output": "x", "is_error": false}'),
        judge=FakeJudge(score=0.25),
        run_id="r1",
        suite_id="demo",
    )

    assert run.run_id == "r1"
    assert run.artifacts.json_path.exists()
    assert run.artifacts.csv_path.exists()

    payload = json.loads(run.artifacts.json_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "r1"
    assert "report" in payload
    assert payload["report"]["total_steps"] > 0

    rows: list[dict[str, str]] = []
    with run.artifacts.csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)

    # Each trace produces exactly one Step under this synthetic structure.
    assert len(rows) == payload["report"]["total_steps"]

    row0 = rows[0]
    assert row0["trace_id"]
    assert row0["action"]
    assert row0["actual"]
    assert row0["predicted"]
    assert row0["score"] == "0.25"


def test_converter_disables_rag_when_requested(tmp_path: Path) -> None:
    corpus = tmp_path / "trace.otel.jsonl"
    _write_otell_jsonl(corpus, n_traces=2)

    converter = TraceToEvalConverter(results_root=tmp_path, rag_enabled=False)
    run = converter.run(
        trace_files=[corpus],
        prompt="BASE",
        provider=FakeProvider('{"output": "x", "is_error": false}'),
        judge=FakeJudge(score=0.1),
        run_id="r2",
        suite_id=None,
    )
    assert run.artifacts.csv_path.exists()
    payload = json.loads(run.artifacts.json_path.read_text(encoding="utf-8"))
    assert payload["report"]["overall_fidelity"] == 0.1

