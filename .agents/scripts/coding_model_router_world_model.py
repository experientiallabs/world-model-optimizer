"""Build, run, and compare the separate WMO simulation matrix for the coding router."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
from coding_model_router_analyze import (
    BENCHMARKS,
    EXPERIMENT_ID,
    SEEDS,
    _canonical_matrix,
    _evaluate,
    _read_object,
    _scenario_tasks,
    _select,
    _sha256,
    _write_json,
)
from coding_model_router_usage import DetailedUsage, exact_cost_usd
from scipy.stats import spearmanr

from wmo.config import HarnessConfig, load_env_file
from wmo.core.files import write_text_atomic
from wmo.core.types import Step, Trace
from wmo.engine.build import build as build_world_model
from wmo.engine.world_model import WorldModel
from wmo.env.base import WorldModelEnv
from wmo.env.closed_loop import PoolCell, pool_cells, run_cell
from wmo.env.scenarios import Scenario, tools_hint_from_traces
from wmo.ingest.otel_genai import OtelGenAIAdapter
from wmo.ingest.otel_writer import write_traces_jsonl
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.providers.base import EmbedderKind, Provider, ProviderConfig, ProviderKind
from wmo.providers.pool import ModelPool, load_pool
from wmo.providers.registry import get_provider
from wmo.retrieval.embedders import HashingEmbedder
from wmo.tracking.metered import MeteredProvider, classify_build_call
from wmo.tracking.tracker import RunRecord, RunTracker

AZURE_MODEL_TYPE = "gpt-5.5"
AZURE_API_VERSION = "2024-10-21"
AZURE_DEPLOYMENT_ENV = "AZURE_FOUNDRY_GPT55_DEPLOYMENT"
AZURE_ENDPOINT_ENV = "AZURE_FOUNDRY_ENDPOINT"
AZURE_KEY_ENV = "AZURE_FOUNDRY_API_KEY"
AZURE_ENV_SOURCE = "/Users/admin/Documents/experientiallabs/platform/.env.local"
PROVIDER_ENV_SOURCE = "/Users/admin/Documents/experientiallabs/world-model-optimizer/.env.local"
SIM_CONCURRENCY = 4
SIM_CELL_RESERVATION_USD = 500.0
BUILD_RESERVATION_USD = 50.0
MAX_ATTEMPTS = 3


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _trace_path(artifact_dir: str) -> Path:
    root = Path(artifact_dir)
    for candidate in (root / "agent" / "wmo-run.json", root / "wmo-run.json"):
        if candidate.is_file():
            return candidate
    matches = sorted(root.rglob("wmo-run.json"))
    if matches:
        return matches[0]
    raise ValueError(f"no wmo-run.json under {root}")


def _prepare_traces(root: Path) -> list[Trace]:
    matrix, _ = _canonical_matrix(root)
    lock = _read_object(root / "analysis" / "selection-lock.json")
    baseline = lock.get("deployment_consensus_baseline")
    if not isinstance(baseline, str):
        raise ValueError("real selection lock has no deployment consensus baseline")
    traces: list[Trace] = []
    for outcome in matrix.outcomes:
        if outcome.model != baseline:
            continue
        raw = _read_object(_trace_path(outcome.artifact_dir))
        raw_steps = raw.get("steps")
        if not isinstance(raw_steps, list):
            raise ValueError(f"{outcome.artifact_dir} has no step list")
        steps = [Step.model_validate(item) for item in raw_steps]
        for step in steps:
            step.task = outcome.task
        trace_id = hashlib.sha256(
            f"coding-wm-v1:{outcome.scenario_id}:{baseline}".encode()
        ).hexdigest()
        traces.append(
            Trace(
                trace_id=trace_id,
                source="coding-router-real-baseline",
                steps=steps,
                metadata={
                    "benchmark": outcome.benchmark,
                    "scenario_id": outcome.scenario_id,
                    "model": baseline,
                },
            )
        )
    if len(traces) != len(matrix.scenario_ids()):
        raise ValueError(
            f"expected one baseline trace per scenario, found {len(traces)} for "
            f"{len(matrix.scenario_ids())} scenarios"
        )
    return traces


def _prepare(root: Path) -> None:
    world_root = root / "world-model"
    corpus = world_root / "corpus.otel.jsonl"
    if corpus.exists():
        raise ValueError(f"{corpus} already exists; world-model corpus is immutable")
    traces = _prepare_traces(root)
    spans = write_traces_jsonl(traces, corpus)
    _write_json(
        world_root / "prepare.json",
        {
            "protocol": "coding-world-model-v1",
            "corpus_sha256": _sha256(corpus),
            "traces": len(traces),
            "spans": spans,
            "source_policy": "fit-only deployment consensus baseline",
            "reward_labels_in_corpus": False,
            "azure_model_type": AZURE_MODEL_TYPE,
            "azure_config_source": AZURE_ENV_SOURCE,
            "required_environment_available": {
                AZURE_DEPLOYMENT_ENV: bool(os.environ.get(AZURE_DEPLOYMENT_ENV)),
                AZURE_ENDPOINT_ENV: bool(os.environ.get(AZURE_ENDPOINT_ENV)),
                AZURE_KEY_ENV: bool(os.environ.get(AZURE_KEY_ENV)),
            },
        },
    )


def _azure_provider() -> Provider:
    deployment = os.environ.get(AZURE_DEPLOYMENT_ENV)
    endpoint = os.environ.get(AZURE_ENDPOINT_ENV)
    key = os.environ.get(AZURE_KEY_ENV)
    missing = [
        name
        for name, value in (
            (AZURE_DEPLOYMENT_ENV, deployment),
            (AZURE_ENDPOINT_ENV, endpoint),
            (AZURE_KEY_ENV, key),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"missing Azure configuration variables: {', '.join(missing)}")
    return get_provider(
        ProviderConfig(
            kind=ProviderKind.AZURE_OPENAI,
            model_type=AZURE_MODEL_TYPE,
            model=AZURE_MODEL_TYPE,
            endpoint=endpoint,
            deployment=deployment,
            api_version=AZURE_API_VERSION,
            reasoning_effort="high",
        ),
        api_key=key,
    )


def _persisted_config() -> HarnessConfig:
    return HarnessConfig(
        providers=[
            ProviderConfig(
                kind=ProviderKind.AZURE_OPENAI,
                model_type=AZURE_MODEL_TYPE,
                model=AZURE_MODEL_TYPE,
                deployment=f"env:{AZURE_DEPLOYMENT_ENV}",
                api_version=AZURE_API_VERSION,
                reasoning_effort="high",
            )
        ],
        serve_provider=ProviderKind.AZURE_OPENAI,
        embed_provider=EmbedderKind.HASHING,
        embed_dim=512,
        top_k=5,
        train_split=0.8,
        gepa_budget=0,
        judge_model=AZURE_MODEL_TYPE,
        trace_adapter="otel-genai",
        reasoning=True,
        verify=True,
    )


def _ledger_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} contains a non-object row")
        rows.append({str(key): item for key, item in value.items()})
    return rows


def _number(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return 0.0
    return float(value)


def _spent_reserved(rows: list[dict[str, object]]) -> tuple[float, float]:
    spent = 0.0
    reserved = 0.0
    for row in rows:
        if row.get("status") == "reserved":
            reserved += _number(row.get("reserved_usd"))
        elif row.get("status") == "completed" or row.get("status") is None:
            spent += _number(row.get("model_cost_usd"))
            spent += _number(row.get("world_model_cost_usd"))
    return spent, reserved


def _write_ledger(path: Path, rows: list[dict[str, object]]) -> None:
    write_text_atomic(
        path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )


def _ceiling(root: Path) -> float:
    freeze = root / "freeze-summary.json"
    value = _read_object(freeze).get("spend_ceiling_usd") if freeze.is_file() else None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError("freeze-summary.json has no authorized positive spend ceiling")
    return float(value)


def _reserve_event(
    root: Path,
    event_id: str,
    amount: float,
    phase: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    benchmark: str | None = None,
) -> None:
    path = root / "spend-ledger.jsonl"
    rows = _ledger_rows(path)
    existing = next((row for row in rows if row.get("event_id") == event_id), None)
    if existing is not None:
        raise ValueError(f"{event_id} already exists in the spend ledger")
    spent, reserved = _spent_reserved(rows)
    if spent + reserved + amount > _ceiling(root):
        raise ValueError(
            f"{event_id} reservation would reach ${spent + reserved + amount:.2f}, "
            f"above the frozen ${_ceiling(root):.2f} ceiling"
        )
    rows.append(
        {
            "event_id": event_id,
            "recorded_at": _utc_now(),
            "phase": phase,
            "provider": provider,
            "model": model,
            "benchmark": benchmark,
            "status": "reserved",
            "reserved_usd": amount,
        }
    )
    _write_ledger(path, rows)


def _complete_event(root: Path, event_id: str, row: dict[str, object]) -> None:
    path = root / "spend-ledger.jsonl"
    rows = _ledger_rows(path)
    reserved = next(
        (existing for existing in rows if existing.get("event_id") == event_id),
        None,
    )
    if reserved is None or reserved.get("status") != "reserved":
        raise ValueError(f"{event_id} has no active spend reservation")
    rows = [existing for existing in rows if existing.get("event_id") != event_id]
    rows.append(
        {
            "event_id": event_id,
            "recorded_at": _utc_now(),
            "status": "completed",
            **row,
        }
    )
    _write_ledger(path, rows)


def _build(root: Path) -> None:
    world_root = root / "world-model"
    corpus = world_root / "corpus.otel.jsonl"
    prepare = _read_object(world_root / "prepare.json")
    if prepare.get("corpus_sha256") != _sha256(corpus):
        raise ValueError("world-model corpus changed after prepare")
    artifact = world_root / "artifact"
    if (artifact / "config.toml").is_file():
        raise ValueError(f"{artifact} is already built")
    event_id = "world-model:build"
    _reserve_event(
        root,
        event_id,
        BUILD_RESERVATION_USD,
        "world_model_build",
        provider="azure",
        model=AZURE_MODEL_TYPE,
        benchmark="all",
    )
    provider = _azure_provider()
    tracker = RunTracker(run_id=event_id, kind="build")
    metered = MeteredProvider(provider, tracker, classify=classify_build_call)
    failure: Exception | None = None
    try:
        with tracker.timed():
            build_world_model(
                _persisted_config(),
                file=str(corpus),
                root=str(artifact),
                serve_provider=metered,
                judge_provider=metered,
                embedder=HashingEmbedder(dim=512),
            )
    except Exception as exc:  # noqa: BLE001 - account for every failed provider build
        failure = exc
    usage = tracker.record_summary()
    _write_json(world_root / "build-usage.json", usage.model_dump(mode="json"))
    # This low-fidelity RAG build has no optimizer or provider calls. The metered
    # record makes that a checked assertion rather than an accounting assumption.
    _complete_event(
        root,
        event_id,
        {
            "phase": "world_model_build",
            "provider": "azure",
            "model": AZURE_MODEL_TYPE,
            "model_cost_usd": usage.total.cost_usd,
            "usage": usage.model_dump(mode="json"),
            "completion_status": "failed" if failure is not None else "built",
            "error": (f"{type(failure).__name__}: {failure}" if failure is not None else None),
            "cost_note": (
                "low-fidelity RAG build made no inference call"
                if not usage.total.calls
                else "unexpected provider activity was metered and the build was rejected"
            ),
        },
    )
    if failure is not None:
        raise failure
    if usage.total.calls:
        raise ValueError("low-fidelity world-model build unexpectedly made provider calls")


class _UsageEnv(WorldModelEnv):
    """WorldModelEnv that returns its closed session meter to one cell owner."""

    def __init__(self, world_model: WorldModel, sink: list[RunRecord]) -> None:
        super().__init__(world_model, score_on_close=True)
        self._sink = sink
        self._reported = False

    def close(self) -> None:
        active = self._session_id is not None
        super().close()
        if active and not self._reported and self.usage is not None:
            self._sink.append(self.usage)
            self._reported = True


class _SimulationState:
    """Atomic simulated matrix and spend-ledger owner."""

    def __init__(self, root: Path, pool: ModelPool) -> None:
        self.root = root
        self.pool = pool
        self.path = root / "world-model" / "simulated" / "outcomes.json"
        self.lock = threading.Lock()
        self.matrix = (
            OutcomeMatrix.load(self.path)
            if self.path.is_file()
            else OutcomeMatrix(pool=pool.models, outcomes=[])
        )
        if self.matrix.pool != pool.models:
            raise ValueError("existing simulated matrix carries a different pool")

    def completed(self, scenario_id: str, model: str) -> bool:
        return any(
            outcome.scenario_id == scenario_id
            and outcome.model == model
            and outcome.reward is not None
            for outcome in self.matrix.outcomes
        )

    def attempts(self, scenario_id: str, model: str) -> int:
        return sum(
            outcome.scenario_id == scenario_id and outcome.model == model
            for outcome in self.matrix.outcomes
        )

    def reserve(self, scenario_id: str, model: str, attempt: int) -> str:
        event_id = f"world-model:simulate:{scenario_id}:{model}:{attempt}"
        with self.lock:
            _reserve_event(
                self.root,
                event_id,
                SIM_CELL_RESERVATION_USD,
                "world_model_simulation",
                provider="mixed",
                model=model,
                benchmark=scenario_id.split(":", 1)[0],
            )
        return event_id

    def persist(
        self,
        outcome: ScenarioOutcome,
        *,
        event_id: str,
        world_usage: RunRecord | None,
    ) -> None:
        with self.lock:
            key = (outcome.scenario_id, outcome.model, outcome.attempt_number)
            self.matrix.outcomes = [
                row
                for row in self.matrix.outcomes
                if (row.scenario_id, row.model, row.attempt_number) != key
            ] + [outcome]
            self.matrix.save(self.path)
            _complete_event(
                self.root,
                event_id,
                {
                    "phase": "world_model_simulation",
                    "provider": "mixed",
                    "model": outcome.model,
                    "benchmark": outcome.benchmark,
                    "scenario_id": outcome.scenario_id,
                    "attempt_number": outcome.attempt_number,
                    "model_cost_usd": outcome.cost_usd,
                    "world_model_provider": "azure",
                    "world_model": AZURE_MODEL_TYPE,
                    "world_model_cost_usd": (
                        world_usage.total.cost_usd if world_usage is not None else 0.0
                    ),
                    "world_model_usage": (
                        world_usage.model_dump(mode="json") if world_usage is not None else None
                    ),
                    "completion_status": ("scored" if outcome.reward is not None else "unscored"),
                },
            )


def _simulated_outcome(
    cell: PoolCell,
    *,
    world_model: WorldModel,
    tools_hint: str,
    attempt: int,
) -> tuple[ScenarioOutcome, RunRecord | None]:
    records: list[RunRecord] = []
    try:
        outcome = run_cell(
            cell,
            lambda: _UsageEnv(world_model, records),
            max_steps=20,
            agent_temperature=0.0,
            tools_hint=tools_hint,
        )
    except Exception as exc:  # noqa: BLE001 - persist the paid failed attempt for resume
        outcome = ScenarioOutcome(
            scenario_id=cell.key.scenario_id,
            task=cell.scenario.task,
            model=cell.entry.name,
            episode=attempt - 1,
            attempt_number=attempt,
            reward=None,
            success=False,
            completion_status="infrastructure_failure",
            failure_class="simulation_infrastructure",
            error=f"{type(exc).__name__}: {exc}",
            remeasured=attempt > 1,
        )
    outcome.benchmark = outcome.scenario_id.split(":", 1)[0]
    outcome.episode = attempt - 1
    outcome.attempt_number = attempt
    outcome.remeasured = attempt > 1
    outcome.completion_status = (
        "simulated_scored" if outcome.reward is not None else "simulation_unscored"
    )
    outcome.failure_class = "" if outcome.reward is not None else "simulation_infrastructure"
    outcome.cost_usd = exact_cost_usd(
        cell.entry,
        DetailedUsage(
            total=outcome.usage,
            calls=len(outcome.call_seconds),
            call_seconds=outcome.call_seconds,
            call_input_tokens=outcome.call_input_tokens,
            call_output_tokens=outcome.call_output_tokens,
            call_cached_input_tokens=outcome.call_cached_input_tokens,
            call_cache_write_input_tokens=outcome.call_cache_write_input_tokens,
        ),
    )
    return outcome, records[-1] if records else None


def _simulate(root: Path) -> None:
    artifact = root / "world-model" / "artifact"
    if not (artifact / "config.toml").is_file():
        raise ValueError("world-model artifact is not built")
    real, _ = _canonical_matrix(root)
    pool = load_pool(root / "pool.toml")
    provider = _azure_provider()
    world_model = WorldModel.load(
        str(artifact),
        provider,
        embedder=HashingEmbedder(dim=512),
        telemetry_root=root / "world-model" / "telemetry",
        reward_provider=provider,
    )
    corpus = root / "world-model" / "corpus.otel.jsonl"
    traces = OtelGenAIAdapter().from_file(str(corpus))
    tools_hint = tools_hint_from_traces(traces)
    scenario_ids, tasks = _scenario_tasks(real)
    scenarios = [
        Scenario(task=tasks[scenario_id], provenance=[scenario_id]) for scenario_id in scenario_ids
    ]
    state = _SimulationState(root, pool)
    pending: list[tuple[PoolCell, int]] = []
    for cell in pool_cells(pool, scenarios):
        scenario_id = cell.key.scenario_id
        if state.completed(scenario_id, cell.entry.name):
            continue
        attempt = state.attempts(scenario_id, cell.entry.name) + 1
        if attempt <= MAX_ATTEMPTS:
            pending.append((cell.model_copy(update={"remeasured": attempt > 1}), attempt))

    with (
        world_model.frozen(),
        ThreadPoolExecutor(
            max_workers=SIM_CONCURRENCY,
            thread_name_prefix="coding-wm",
        ) as executor,
    ):
        active: dict[
            Future[tuple[ScenarioOutcome, RunRecord | None]],
            tuple[str, PoolCell, int],
        ] = {}
        cursor = 0
        while cursor < len(pending) or active:
            while cursor < len(pending) and len(active) < SIM_CONCURRENCY:
                cell, attempt = pending[cursor]
                event_id = state.reserve(cell.key.scenario_id, cell.entry.name, attempt)
                future = executor.submit(
                    _simulated_outcome,
                    cell,
                    world_model=world_model,
                    tools_hint=tools_hint,
                    attempt=attempt,
                )
                active[future] = (event_id, cell, attempt)
                cursor += 1
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                event_id, cell, attempt = active.pop(future)
                outcome, usage = future.result()
                state.persist(outcome, event_id=event_id, world_usage=usage)
                if outcome.reward is None and attempt < MAX_ATTEMPTS:
                    pending.append(
                        (
                            cell.model_copy(update={"episode": attempt, "remeasured": True}),
                            attempt + 1,
                        )
                    )

    incomplete = [
        (scenario_id, model)
        for scenario_id in scenario_ids
        for model in [entry.name for entry in pool.models]
        if not state.completed(scenario_id, model)
    ]
    _write_json(
        root / "world-model" / "simulated" / "completion.json",
        {
            "expected_cells": len(scenario_ids) * len(pool.models),
            "gradeable_cells": sum(outcome.reward is not None for outcome in state.matrix.outcomes),
            "incomplete_cells": incomplete,
            "complete": not incomplete,
        },
    )
    if incomplete:
        raise ValueError(
            f"simulated matrix has {len(incomplete)} ungradeable cells after "
            f"{MAX_ATTEMPTS} attempts"
        )


def _analysis_root(root: Path) -> Path:
    return root / "world-model" / "analysis-input"


def _prepare_simulated_analysis(root: Path) -> Path:
    target = _analysis_root(root)
    target.mkdir(parents=True, exist_ok=True)
    (target / "tasks").mkdir(exist_ok=True)
    (target / "splits").mkdir(exist_ok=True)
    (target / "full").mkdir(exist_ok=True)
    for benchmark in BENCHMARKS:
        shutil.copyfile(
            root / "tasks" / f"{benchmark}.json",
            target / "tasks" / f"{benchmark}.json",
        )
    for seed in SEEDS:
        shutil.copyfile(
            root / "splits" / f"seed-{seed}.json",
            target / "splits" / f"seed-{seed}.json",
        )
    shutil.copyfile(root / "pool.toml", target / "pool.toml")
    shutil.copyfile(
        root / "world-model" / "simulated" / "outcomes.json",
        target / "full" / "outcomes.json",
    )
    embedding = root / "embeddings" / "text-embedding-3-large-3072.npy"
    if not embedding.is_file():
        raise ValueError("real task embedding cache is required for simulated policy selection")
    (target / "embeddings").mkdir(exist_ok=True)
    shutil.copyfile(
        embedding,
        target / "embeddings" / embedding.name,
    )
    return target


def _analyze_simulated(root: Path) -> None:
    target = _prepare_simulated_analysis(root)
    lock = target / "analysis" / "selection-lock.json"
    results = target / "analysis" / "outer-results.json"
    if not lock.exists():
        _select(target)
    if not results.exists():
        _evaluate(target)


def _metric_number(metric: dict[str, object], key: str) -> float:
    value = metric.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"metric {key} is not numeric")
    return float(value)


def _model_scores(matrix: OutcomeMatrix) -> dict[str, float]:
    by_model_benchmark: dict[str, dict[str, list[float]]] = {
        model: {benchmark: [] for benchmark in BENCHMARKS} for model in matrix.model_names()
    }
    for outcome in matrix.outcomes:
        if outcome.reward is None:
            continue
        by_model_benchmark[outcome.model][outcome.benchmark].append(outcome.reward)
    return {
        model: sum(
            0.5 * float(np.mean(by_model_benchmark[model][benchmark])) for benchmark in BENCHMARKS
        )
        for model in matrix.model_names()
    }


def _seed_map(value: object, *, label: str) -> dict[int, dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError(f"{label} is not a list")
    rows: dict[int, dict[str, object]] = {}
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"{label} has a non-object row")
        row = {str(key): entry for key, entry in item.items()}
        seed = row.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError(f"{label} row has no integer seed")
        rows[seed] = row
    return rows


def _paired_map(row: dict[str, object]) -> dict[str, dict[str, object]]:
    paired = row.get("paired")
    if not isinstance(paired, list):
        raise ValueError("outer seed row has no paired rows")
    result: dict[str, dict[str, object]] = {}
    for item in paired:
        if not isinstance(item, dict):
            raise ValueError("outer paired row is invalid")
        paired_row = {str(key): value for key, value in item.items()}
        scenario_id = paired_row.get("scenario_id")
        if not isinstance(scenario_id, str):
            raise ValueError("outer paired row is invalid")
        result[scenario_id] = paired_row
    return result


def _point(row: dict[str, object], name: str) -> dict[str, object]:
    points = row.get("points")
    if not isinstance(points, dict):
        raise ValueError("outer seed row has no points")
    value = points.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"outer seed row has no {name} point")
    return {str(key): item for key, item in value.items()}


def _compare(root: Path) -> None:
    sim_root = _analysis_root(root)
    real_matrix, _ = _canonical_matrix(root)
    simulated_matrix, _ = _canonical_matrix(sim_root)
    real_cells = {
        (row.scenario_id, row.model): row for row in real_matrix.outcomes if row.reward is not None
    }
    simulated_cells = {
        (row.scenario_id, row.model): row
        for row in simulated_matrix.outcomes
        if row.reward is not None
    }
    if real_cells.keys() != simulated_cells.keys():
        raise ValueError("real and simulated matrices do not carry the same gradeable cells")

    actual: list[float] = []
    predicted: list[float] = []
    for key in sorted(real_cells):
        actual.append(cast("float", real_cells[key].reward))
        predicted.append(cast("float", simulated_cells[key].reward))
    actual_binary = [value >= 0.5 for value in actual]
    predicted_binary = [value >= 0.5 for value in predicted]
    true_positive = sum(a and p for a, p in zip(actual_binary, predicted_binary, strict=True))
    true_negative = sum(
        not a and not p for a, p in zip(actual_binary, predicted_binary, strict=True)
    )
    false_positive = sum(not a and p for a, p in zip(actual_binary, predicted_binary, strict=True))
    false_negative = sum(a and not p for a, p in zip(actual_binary, predicted_binary, strict=True))
    calibration: list[dict[str, object]] = []
    for lower in np.arange(0.0, 1.0, 0.1):
        upper = float(lower + 0.1)
        indices = [
            index
            for index, value in enumerate(predicted)
            if float(lower) <= value < upper or (upper >= 1.0 and value == 1.0)
        ]
        if indices:
            calibration.append(
                {
                    "lower": float(lower),
                    "upper": upper,
                    "count": len(indices),
                    "predicted_mean": float(np.mean([predicted[index] for index in indices])),
                    "actual_rate": float(np.mean([actual[index] for index in indices])),
                }
            )

    real_scores = _model_scores(real_matrix)
    simulated_scores = _model_scores(simulated_matrix)
    correlation = spearmanr(
        [real_scores[model] for model in real_matrix.model_names()],
        [simulated_scores[model] for model in real_matrix.model_names()],
    ).statistic

    real_lock = _read_object(root / "analysis" / "selection-lock.json")
    simulated_lock = _read_object(sim_root / "analysis" / "selection-lock.json")
    real_results = _read_object(root / "analysis" / "outer-results.json")
    simulated_results = _read_object(sim_root / "analysis" / "outer-results.json")
    real_lock_seeds = _seed_map(real_lock.get("seeds"), label="real lock seeds")
    simulated_lock_seeds = _seed_map(simulated_lock.get("seeds"), label="simulated lock seeds")
    real_outer = _seed_map(real_results.get("seeds"), label="real outer seeds")
    simulated_outer = _seed_map(simulated_results.get("seeds"), label="simulated outer seeds")

    selected_model_equal = 0
    selected_model_total = 0
    guard_equal = 0
    guard_total = 0
    seed_comparisons: list[dict[str, object]] = []
    for seed in SEEDS:
        real_paired = _paired_map(real_outer[seed])
        simulated_paired = _paired_map(simulated_outer[seed])
        if real_paired.keys() != simulated_paired.keys():
            raise ValueError(f"seed {seed} paired heldout cohorts differ")
        for scenario_id in real_paired:
            real_row = real_paired[scenario_id]
            simulated_row = simulated_paired[scenario_id]
            selected_model_equal += int(
                real_row.get("router_model") == simulated_row.get("router_model")
            )
            selected_model_total += 1
            guard_equal += int(real_row.get("guard_gate") == simulated_row.get("guard_gate"))
            guard_total += 1
        real_guarded = _point(real_outer[seed], "guarded_knn")
        real_base = _point(real_outer[seed], "best_single")
        sim_guarded = _point(simulated_outer[seed], "guarded_knn")
        sim_base = _point(simulated_outer[seed], "best_single")
        seed_comparisons.append(
            {
                "seed": seed,
                "best_single_agreement": (
                    real_lock_seeds[seed].get("baseline")
                    == simulated_lock_seeds[seed].get("baseline")
                ),
                "config_agreement": (
                    real_lock_seeds[seed].get("config") == simulated_lock_seeds[seed].get("config")
                ),
                "real_quality_delta": (
                    _metric_number(real_guarded, "quality") - _metric_number(real_base, "quality")
                ),
                "simulated_quality_delta": (
                    _metric_number(sim_guarded, "quality") - _metric_number(sim_base, "quality")
                ),
                "real_cost_delta": (
                    _metric_number(real_guarded, "cost_per_task")
                    - _metric_number(real_base, "cost_per_task")
                ),
                "simulated_cost_delta": (
                    _metric_number(sim_guarded, "cost_per_task")
                    - _metric_number(sim_base, "cost_per_task")
                ),
            }
        )

    comparison = {
        "protocol": "coding-world-model-compare-v1",
        "real_matrix_sha256": _sha256(root / "full" / "outcomes.json"),
        "simulated_matrix_sha256": _sha256(root / "world-model" / "simulated" / "outcomes.json"),
        "cells": len(actual),
        "binary_agreement": (true_positive + true_negative) / len(actual),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "false_positive_rate": (
            false_positive / (false_positive + true_negative)
            if false_positive + true_negative
            else None
        ),
        "false_negative_rate": (
            false_negative / (false_negative + true_positive)
            if false_negative + true_positive
            else None
        ),
        "brier_score": float(
            np.mean(
                [
                    (prediction - truth) ** 2
                    for prediction, truth in zip(predicted, actual, strict=True)
                ]
            )
        ),
        "calibration": calibration,
        "candidate_scores_real": real_scores,
        "candidate_scores_simulated": simulated_scores,
        "candidate_rank_spearman": (float(correlation) if not np.isnan(correlation) else None),
        "best_single_agreement": sum(bool(row["best_single_agreement"]) for row in seed_comparisons)
        / len(SEEDS),
        "selected_config_agreement": sum(bool(row["config_agreement"]) for row in seed_comparisons)
        / len(SEEDS),
        "selected_model_agreement": selected_model_equal / selected_model_total,
        "guard_decision_agreement": guard_equal / guard_total,
        "seed_comparisons": seed_comparisons,
        "real_promoted": real_results.get("promoted"),
        "simulated_promoted": simulated_results.get("promoted"),
        "promotion_agreement": (real_results.get("promoted") == simulated_results.get("promoted")),
        "deployment_consensus_config_agreement": (
            real_lock.get("deployment_consensus_config")
            == simulated_lock.get("deployment_consensus_config")
        ),
        "deployment_consensus_baseline_agreement": (
            real_lock.get("deployment_consensus_baseline")
            == simulated_lock.get("deployment_consensus_baseline")
        ),
        "limitations": [
            (
                "The world model was trained on one reward-free real baseline trajectory per "
                "task; task-specific environment observations are therefore available to "
                "retrieval."
            ),
            (
                "The simulated matrix uses WMO's native LLMAgent scaffold, while the real "
                "matrix uses the benchmark's Harbor and Pi scaffold. Their difference is a "
                "simulation-to-real confound."
            ),
        ],
    }
    _write_json(root / "world-model" / "comparison.json", comparison)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=("prepare", "build", "simulate", "analyze", "compare"),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(".wmo") / "experiments" / EXPERIMENT_ID,
    )
    return parser.parse_args()


def main() -> None:
    load_env_file(PROVIDER_ENV_SOURCE)
    load_env_file(AZURE_ENV_SOURCE)
    args = _parse_args()
    root = cast("Path", args.root).resolve()
    if args.phase == "prepare":
        _prepare(root)
    elif args.phase == "build":
        _build(root)
    elif args.phase == "simulate":
        _simulate(root)
    elif args.phase == "analyze":
        _analyze_simulated(root)
    else:
        _compare(root)


if __name__ == "__main__":
    main()
