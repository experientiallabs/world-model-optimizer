"""Trace-to-eval pipeline.

This module provides a minimal integration point between:

- **Trace ingestion**: OpenTelemetry JSONL → normalized `Trace`/`Step` objects
  (`wmh.ingest.otel_genai` via the trace adapter registry).
- **Eval orchestration**: open-loop replay + judge scoring of reconstruction fidelity
  (`wmh.engine.eval` → `wmh.engine.replay`).

Why it exists

Dataset authors often already have agent traces recorded in OpenTelemetry, but they still need a
consistent way to:

1. convert those traces into a fidelity-eval run,
2. and persist results into inspectable artifacts (CSV/JSON).

`TraceToEvalConverter` is a thin integration layer that reuses the repo’s canonical eval code
paths, while adding:

- validated, typed inputs
- output directory conventions under `.wmh/evals/`
- CSV flattening (one row per scored step)

This is intentionally small: it does not invent new evaluation logic. The “proper way” is always
routed through the existing harness eval stack.
"""


from __future__ import annotations

import csv
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel

from wmh.engine.eval import EvalReport, evaluate_files
from wmh.providers.base import Embedder, Provider

from wmh.optimize.judge import Judge


class TraceEvalRunConfig(BaseModel):
    """Configuration persisted alongside eval artifacts."""

    adapter_name: str = "otel-genai"
    train_split: float = 0.7
    top_k: int = 5
    sample_turns: str = "all"  # "all" | "sampled"
    seed: int = 0
    rag_enabled: bool = True
    embed_dim: int | None = None


class EvalArtifacts(BaseModel):
    """Paths for persisted artifacts (JSON + CSV)."""

    json_path: Path
    csv_path: Path


class TraceEvalRun(BaseModel):
    """A complete persisted trace-eval run."""

    run_id: str
    started_at: str
    config: TraceEvalRunConfig
    report: EvalReport
    artifacts: EvalArtifacts


@dataclass(frozen=True)
class _CsvRow:
    trace_id: str
    task: str | None
    action: str
    actual: str
    predicted: str
    score: float
    is_error_actual: bool
    is_error_predicted: bool
    critique: str
    dimensions: str

    def to_csv_dict(self) -> dict[str, str | float | bool]:
        return {
            "trace_id": self.trace_id,
            "task": self.task or "",
            "action": self.action,
            "actual": self.actual,
            "predicted": self.predicted,
            "score": self.score,
            "is_error_actual": self.is_error_actual,
            "is_error_predicted": self.is_error_predicted,
            "critique": self.critique,
            "dimensions": self.dimensions,
        }


class TraceToEvalConverter:
    """Convert OpenTelemetry JSONL traces into fidelity-eval artifacts.


    This converter is a thin layer over `wmh.engine.eval.evaluate_files`. It exists so dataset
    authors can consistently produce CSV/JSON logs without reimplementing eval orchestration.

    Typical usage:

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
    print(run.artifacts.csv_path)
    ```

    The converter itself focuses on:

    - input validation and artifact directory conventions
    - invoking the canonical eval orchestration (ingest → split → replay → judge)
    - flattening per-step results into a CSV
    """


    def __init__(
        self,
        *,
        adapter_name: str = "otel-genai",
        train_split: float = 0.7,
        top_k: int = 5,
        sample_turns: str = "all",
        seed: int = 0,
        rag_enabled: bool = True,
        embedder: Embedder | None = None,
        results_root: str | Path = ".wmh/evals",
    ) -> None:
        if not 0.0 < train_split < 1.0:
            raise ValueError("train_split must be between 0 and 1")
        if sample_turns not in {"all", "sampled"}:
            raise ValueError("sample_turns must be one of: all, sampled")
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if seed < 0:
            raise ValueError("seed must be >= 0")

        self._adapter_name = adapter_name
        self._train_split = train_split
        self._top_k = top_k
        self._sample_turns = sample_turns
        self._seed = seed
        self._rag_enabled = rag_enabled
        self._embedder = embedder
        self._results_root = Path(results_root)

    def run(
        self,
        *,
        trace_files: list[Path],
        prompt: str,
        provider: Provider,
        judge: Judge,
        run_id: str | None = None,
        suite_id: str | None = None,
    ) -> TraceEvalRun:
        """Run replay-based fidelity eval on the given trace files.

        Args:
            trace_files: OTLP-JSON / JSONL files accepted by the configured trace adapter.
            prompt: The environment prompt for the world-model replay.
            provider: The model provider used to generate predicted observations.
            judge: The judge that scores predicted vs actual observations.
            run_id: Optional caller-provided id. When omitted, a new UUID is generated.
            suite_id: Optional grouping label; when provided, artifacts are written under
                `<results_root>/<suite_id>/`.

        Returns:
            A `TraceEvalRun` with the structured eval report and artifact paths.
        """
        if not trace_files:
            raise ValueError("trace_files must be non-empty")
        for path in trace_files:
            if not path.exists():
                raise FileNotFoundError(str(path))

        run_id = run_id or uuid.uuid4().hex
        started_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

        out_dir = self._results_root / (suite_id or "ad-hoc")
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{run_id}.json"
        csv_path = out_dir / f"{run_id}.csv"

        embedder = self._embedder if self._rag_enabled else None
        embed_dim = None
        if embedder is not None and hasattr(embedder, "dim"):
            # HashingEmbedder exposes dim; other embedders in this repo follow the same contract.
            embed_dim = int(getattr(embedder, "dim"))

        report = evaluate_files(
            trace_files,
            prompt,
            provider,
            judge,
            embedder=embedder,
            train_split=self._train_split,
            top_k=self._top_k,
            sample_turns=self._sample_turns,
            seed=self._seed,
            adapter_name=self._adapter_name,
        )

        run_config = TraceEvalRunConfig(
            adapter_name=self._adapter_name,
            train_split=self._train_split,
            top_k=self._top_k,
            sample_turns=self._sample_turns,
            seed=self._seed,
            rag_enabled=self._rag_enabled,
            embed_dim=embed_dim,
        )

        artifacts = EvalArtifacts(json_path=json_path, csv_path=csv_path)
        self._write_json(json_path, run_id, started_at, run_config, report)
        self._write_csv(csv_path, report)

        return TraceEvalRun(
            run_id=run_id,
            started_at=started_at,
            config=run_config,
            report=report,
            artifacts=artifacts,
        )

    def _write_json(
        self,
        path: Path,
        run_id: str,
        started_at: str,
        config: TraceEvalRunConfig,
        report: EvalReport,
    ) -> None:
        payload = {
            "run_id": run_id,
            "started_at": started_at,
            "config": config.model_dump(mode="json"),
            "report": report.model_dump(mode="json"),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_csv(self, path: Path, report: EvalReport) -> None:
        rows = list(_iter_csv_rows(report))
        # Deterministic ordering: trace name order is preserved by the caller's file list in
        # evaluate_files, but we still sort within each trace by step result sequence.
        # Our `_iter_csv_rows` already yields in that order.
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "trace_id",
                    "task",
                    "action",
                    "actual",
                    "predicted",
                    "score",
                    "is_error_actual",
                    "is_error_predicted",
                    "critique",
                    "dimensions",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row.to_csv_dict())


def _iter_csv_rows(report: EvalReport) -> Iterable[_CsvRow]:
    for _, per_file in report.per_file.items():
        for step in per_file.results:
            dims_json = json.dumps(step.dimensions, sort_keys=True)
            yield _CsvRow(
                trace_id=step.trace_id,
                task=step.task,
                action=step.action,
                actual=step.actual,
                predicted=step.predicted,
                score=step.score,
                is_error_actual=step.is_error_actual,
                is_error_predicted=step.is_error_predicted,
                critique=step.critique,
                dimensions=dims_json,
            )

