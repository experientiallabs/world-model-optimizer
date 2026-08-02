"""Nested fit-only selection and one-shot heldout evaluation for the coding router.

The command deliberately has separate ``select`` and ``evaluate`` phases. ``select`` may read
only each outer seed's fit rows and atomically freezes all five choices. ``evaluate`` refuses to
run without that lock and writes its final result only once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import statistics
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np
from coding_model_router_matrix import (
    FAST_DEV_ARMS,
    FAST_DEV_BENCHMARK,
    FAST_DEV_TASK_COUNT,
    _fast_dev_task_ids,
)

from wmo.core.files import write_text_atomic
from wmo.optimize.knn import cost_quality_knobs, fit_knn_artifact, fit_knn_policy
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.policy import (
    KNN_BANK_FILENAME,
    EmbedderSpec,
    RoutingDecision,
    RoutingEvidence,
    RoutingPolicy,
)
from wmo.optimize.routing import evaluate_policy, fit_rank_policy, route_scenarios
from wmo.providers.base import Embedder

EXPERIMENT_ID = "coding-router-20260728"
BENCHMARKS = ("terminal-bench-2", "swe-bench-verified")
BENCHMARK_WEIGHT = 0.5
SEEDS = tuple(range(5))
INNER_FOLDS = 5
BOOTSTRAP_SAMPLES = 10_000
QUALITY_RETENTION_GATE = 0.95
SAVINGS_GATE = 0.40
BENCHMARK_RETENTION_FLOOR = 0.90
BENCHMARK_ABSOLUTE_LOSS_LIMIT = 0.10
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
OPENAI_EMBEDDING_DIM = 3072
FAST_DEV_FIT_TASKS = 8
MISSING_CELL_COVERAGE = 0.8

CAPABILITY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "build-and-dependency": (
        "build",
        "compile",
        "configure",
        "cython",
        "gcov",
        "install",
        "pypi",
        "qemu",
        "webserver",
    ),
    "code-generation-and-translation": (
        "cobol",
        "code-from",
        "codegolf",
        "compiler",
        "interpreter",
        "metacircular",
        "polyglot",
        "prove",
        "write-",
    ),
    "data-ml-and-scientific": (
        "circuit",
        "data",
        "dna",
        "eigen",
        "fasttext",
        "fitting",
        "inference",
        "mcmc",
        "model",
        "mteb",
        "optimization",
        "protein",
        "pytorch",
        "sampler",
        "sampling",
        "scientific",
        "stan",
        "tensor",
        "torch",
    ),
    "debugging-and-test-repair": (
        "break-",
        "cancel-",
        "crash",
        "fix-",
        "leak",
        "recovery",
        "repair",
        "sanitize",
        "truncate",
        "vulnerable",
    ),
    "security-and-recovery": (
        "cert",
        "crack",
        "cryptanalysis",
        "leak",
        "password",
        "recovery",
        "sanitize",
        "secret",
        "vulnerab",
    ),
}

logger = logging.getLogger("coding-model-router-analyze")


@dataclass(frozen=True)
class RouterConfig:
    """One fully specified pre-inference guarded kNN operating point."""

    embedder: str = "hashing-1024"
    neighbors: int = 50
    similarity_threshold: float = 0.95
    novelty_quantile: float = 0.05
    guard_z: float = 0.5
    min_pairs: int = 8
    se_floor: bool = True
    asymmetric_guard: bool = False
    cost_quality_dial: float = 0.25


SEARCH_SPACE: tuple[tuple[str, tuple[object, ...]], ...] = (
    ("embedder", ("hashing-1024", "openai-text-embedding-3-large")),
    ("neighbors", (8, 16, 32, 50)),
    ("similarity_threshold", (0.90, 0.95, 0.98)),
    ("novelty_quantile", (0.0, 0.05, 0.20, 0.50)),
    ("guard_z", (0.0, 0.5, 1.0, 1.645)),
    ("min_pairs", (3, 5, 8, 12)),
    ("se_floor", (False, True)),
    ("asymmetric_guard", (False, True)),
    ("cost_quality_dial", (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)),
)


@dataclass(frozen=True)
class Metric:
    quality: float
    cost_per_task: float
    total_cost: float
    effective_cost_per_success: float | None
    success_rate: float
    latency_p50_s: float
    latency_p95_s: float
    model_mix: dict[str, float]
    route_away_rate: float
    guard_reversion_rate: float
    novelty_abstention_rate: float
    per_benchmark: dict[str, dict[str, float]]
    scenarios: int

    def json(self) -> dict[str, object]:
        return asdict(self)


class CachedEmbedder:
    """Serve a scenario-aligned embedding cache by pre-call task text."""

    def __init__(self, matrix: OutcomeMatrix, vectors: np.ndarray) -> None:
        ids, tasks = _scenario_tasks(matrix)
        if vectors.ndim != 2 or vectors.shape[0] != len(ids):
            raise ValueError(
                f"embedding cache has shape {vectors.shape}, expected ({len(ids)}, dimension)"
            )
        self.dim = int(vectors.shape[1])
        self._by_text = {
            tasks[scenario_id]: vectors[index] for index, scenario_id in enumerate(ids)
        }

    def embed(self, texts: list[str]) -> list[list[float]]:
        missing = [text for text in texts if text not in self._by_text]
        if missing:
            raise KeyError(f"{len(missing)} task texts are absent from the embedding cache")
        return [self._by_text[text].tolist() for text in texts]


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return {str(key): item for key, item in value.items()}


def _object_mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return {str(key): item for key, item in value.items()}


def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _integer(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return value


def _number(value: object, *, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    return float(value)


def _boolean(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _config_from_mapping(value: object) -> RouterConfig:
    row = _object_mapping(value, label="router config")
    return RouterConfig(
        embedder=_string(row.get("embedder"), label="config.embedder"),
        neighbors=_integer(row.get("neighbors"), label="config.neighbors"),
        similarity_threshold=_number(
            row.get("similarity_threshold"), label="config.similarity_threshold"
        ),
        novelty_quantile=_number(row.get("novelty_quantile"), label="config.novelty_quantile"),
        guard_z=_number(row.get("guard_z"), label="config.guard_z"),
        min_pairs=_integer(row.get("min_pairs"), label="config.min_pairs"),
        se_floor=_boolean(row.get("se_floor"), label="config.se_floor"),
        asymmetric_guard=_boolean(row.get("asymmetric_guard"), label="config.asymmetric_guard"),
        cost_quality_dial=_number(row.get("cost_quality_dial"), label="config.cost_quality_dial"),
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _scenario_tasks(matrix: OutcomeMatrix) -> tuple[list[str], dict[str, str]]:
    ids: list[str] = []
    tasks: dict[str, str] = {}
    for outcome in matrix.outcomes:
        if outcome.scenario_id not in tasks:
            ids.append(outcome.scenario_id)
            tasks[outcome.scenario_id] = outcome.task
        elif tasks[outcome.scenario_id] != outcome.task:
            raise ValueError(f"{outcome.scenario_id} carries inconsistent pre-call task text")
    return ids, tasks


def _capability_slice_ids(matrix: OutcomeMatrix) -> dict[str, list[str]]:
    """Declare overlapping capability cohorts from pre-call task information only."""
    scenario_ids, tasks = _scenario_tasks(matrix)
    prompt_lengths = np.asarray([len(tasks[scenario_id]) for scenario_id in scenario_ids])
    long_context_floor = float(np.quantile(prompt_lengths, 0.75))
    slices: dict[str, list[str]] = {
        "repository-level-bug-fixing": [],
        "terminal-operation-and-tool-use": [],
        "long-context": [],
        **{name: [] for name in CAPABILITY_KEYWORDS},
    }
    for scenario_id in scenario_ids:
        benchmark, task_id = scenario_id.split(":", 1)
        normalized = f"{task_id} {tasks[scenario_id]}".casefold()
        if benchmark == "swe-bench-verified":
            slices["repository-level-bug-fixing"].append(scenario_id)
            slices["debugging-and-test-repair"].append(scenario_id)
        if benchmark == "terminal-bench-2":
            slices["terminal-operation-and-tool-use"].append(scenario_id)
        if len(tasks[scenario_id]) >= long_context_floor:
            slices["long-context"].append(scenario_id)
        for name, keywords in CAPABILITY_KEYWORDS.items():
            if name == "debugging-and-test-repair" and benchmark == "swe-bench-verified":
                continue
            if any(keyword in normalized for keyword in keywords):
                slices[name].append(scenario_id)
    empty = [name for name, ids in slices.items() if not ids]
    if empty:
        raise ValueError(f"declared capability slices are empty: {empty}")
    return slices


def _manifest_rows(root: Path) -> dict[str, dict[str, dict[str, object]]]:
    rows: dict[str, dict[str, dict[str, object]]] = {}
    for benchmark in BENCHMARKS:
        raw = _read_object(root / "tasks" / f"{benchmark}.json").get("tasks")
        if not isinstance(raw, list):
            raise ValueError(f"{benchmark} task manifest has no task list")
        by_id: dict[str, dict[str, object]] = {}
        for item in raw:
            if not isinstance(item, dict) or not isinstance(item.get("task_id"), str):
                raise ValueError(f"{benchmark} task manifest has an invalid row")
            row = _object_mapping(item, label=f"{benchmark} task row")
            task_id = _string(row.get("task_id"), label=f"{benchmark} task_id")
            by_id[task_id] = row
        rows[benchmark] = by_id
    return rows


def _expected_scenarios(manifests: dict[str, dict[str, dict[str, object]]]) -> list[str]:
    return [
        f"{benchmark}:{task_id}" for benchmark in BENCHMARKS for task_id in manifests[benchmark]
    ]


def _canonical_matrix(root: Path) -> tuple[OutcomeMatrix, dict[str, dict[str, dict[str, object]]]]:
    """Collapse infrastructure attempts to one gradeable row and require the exact dense grid."""
    source = OutcomeMatrix.load(root / "full" / "outcomes.json")
    manifests = _manifest_rows(root)
    expected_ids = _expected_scenarios(manifests)
    expected = {
        (scenario_id, model) for scenario_id in expected_ids for model in source.model_names()
    }
    gradeable: dict[tuple[str, str], list[ScenarioOutcome]] = defaultdict(list)
    for outcome in source.outcomes:
        if outcome.reward is not None:
            gradeable[(outcome.scenario_id, outcome.model)].append(outcome)
    gradeable_keys = set(gradeable)
    missing = sorted(expected - gradeable_keys)
    extra = sorted(gradeable_keys - expected)
    duplicates = sorted(key for key, rows in gradeable.items() if len(rows) != 1)
    if missing or extra or duplicates:
        report = {
            "expected_cells": len(expected),
            "gradeable_cells": len(gradeable_keys),
            "missing": missing,
            "extra": extra,
            "duplicate_gradeable": duplicates,
            "all_attempt_rows": len(source.outcomes),
        }
        _write_json(root / "analysis" / "matrix-validation.json", report)
        raise ValueError(
            "matrix is not the exact dense gradeable grid: "
            f"{len(missing)} missing, {len(extra)} extra, {len(duplicates)} duplicate"
        )
    canonical = OutcomeMatrix(
        pool=source.pool,
        outcomes=[gradeable[(sid, model)][0] for sid, model in sorted(expected)],
    )
    _write_json(
        root / "analysis" / "matrix-validation.json",
        {
            "expected_cells": len(expected),
            "gradeable_cells": len(gradeable_keys),
            "scenarios": len(expected_ids),
            "models": len(source.pool),
            "all_attempt_rows": len(source.outcomes),
            "status": "complete",
        },
    )
    return canonical, manifests


def _fast_dev_matrix(root: Path) -> tuple[OutcomeMatrix, list[str]]:
    """Load the exact gradeable fast tranche without reading any outer-heldout row."""
    source_path = root / "full" / "outcomes.json"
    source = OutcomeMatrix.load(source_path)
    task_ids = _fast_dev_task_ids(root)
    scenario_ids = [f"{FAST_DEV_BENCHMARK}:{task_id}" for task_id in task_ids]
    expected = {(scenario_id, model) for scenario_id in scenario_ids for model in FAST_DEV_ARMS}
    gradeable: dict[tuple[str, str], list[ScenarioOutcome]] = defaultdict(list)
    for outcome in source.outcomes:
        key = (outcome.scenario_id, outcome.model)
        if key in expected and outcome.reward is not None:
            gradeable[key].append(outcome)
    missing = sorted(expected - gradeable.keys())
    duplicates = sorted(key for key, rows in gradeable.items() if len(rows) != 1)
    if missing or duplicates:
        _write_json(
            root / "analysis" / "fast-dev-validation.json",
            {
                "expected_cells": len(expected),
                "gradeable_cells": len(gradeable),
                "missing": missing,
                "duplicate_gradeable": duplicates,
                "status": "incomplete",
            },
        )
        raise ValueError(
            "fast development tranche is incomplete: "
            f"{len(missing)} missing, {len(duplicates)} duplicate"
        )
    pool = [entry for entry in source.pool if entry.name in FAST_DEV_ARMS]
    if {entry.name for entry in pool} != set(FAST_DEV_ARMS):
        raise ValueError("full matrix pool does not contain every fast development arm")
    matrix = OutcomeMatrix(
        pool=pool,
        outcomes=[gradeable[key][0] for key in sorted(expected)],
    )
    _write_json(
        root / "analysis" / "fast-dev-validation.json",
        {
            "expected_cells": FAST_DEV_TASK_COUNT * len(FAST_DEV_ARMS),
            "gradeable_cells": len(matrix.outcomes),
            "tasks": task_ids,
            "arms": list(FAST_DEV_ARMS),
            "status": "complete",
        },
    )
    return matrix, scenario_ids


def _fast_static_metric(
    matrix: OutcomeMatrix,
    ids: list[str],
    model: str,
) -> dict[str, float]:
    cells = _cell_map(matrix)
    rows = [cells[(scenario_id, model)] for scenario_id in ids]
    return {
        "quality": statistics.fmean(cast("list[float]", [row.reward for row in rows])),
        "cost_per_task": statistics.fmean(row.cost_usd for row in rows),
    }


def _develop(root: Path) -> None:
    """Fit a mutable diagnostic router on the fast tranche, never for promotion."""
    matrix, scenario_ids = _fast_dev_matrix(root)
    fit_ids = scenario_ids[:FAST_DEV_FIT_TASKS]
    replay_ids = scenario_ids[FAST_DEV_FIT_TASKS:]
    order = {entry.name: index for index, entry in enumerate(matrix.pool)}
    fit_static = {
        model: _fast_static_metric(matrix, fit_ids, model) for model in matrix.model_names()
    }
    baseline = min(
        matrix.model_names(),
        key=lambda model: (
            -fit_static[model]["quality"],
            fit_static[model]["cost_per_task"],
            order[model],
        ),
    )
    policy_path = root / "analysis" / "fast-dev-policy" / "policy.json"
    fitted = fit_knn_artifact(
        matrix,
        out_path=policy_path,
        matrix_source=str(root / "full" / "outcomes.json"),
        embedder=EmbedderSpec(kind="hashing", dim=1024),
        fit_ids=fit_ids,
        fallback=baseline,
        z=0.5,
        rag_num=8,
        rag_thres=0.95,
        min_pairs=3,
        se_floor=True,
        floor_q=0.05,
    )
    replay = evaluate_policy(fitted.policy, matrix, replay_ids)
    replay_baseline = _fast_static_metric(matrix, replay_ids, baseline)
    retention = (
        replay.accuracy / replay_baseline["quality"]
        if replay_baseline["quality"] > 0
        else float(replay.accuracy >= replay_baseline["quality"])
    )
    savings = (
        1.0 - replay.cost_per_scenario / replay_baseline["cost_per_task"]
        if replay_baseline["cost_per_task"] > 0
        else 0.0
    )
    _write_json(
        root / "analysis" / "fast-dev-report.json",
        {
            "protocol": "coding-router-fast-dev-v1",
            "diagnostic_only": True,
            "promotion_evidence": False,
            "matrix_sha256": _sha256(root / "full" / "outcomes.json"),
            "code_commit": _git_commit(),
            "benchmark": FAST_DEV_BENCHMARK,
            "arms": list(FAST_DEV_ARMS),
            "fit_ids": fit_ids,
            "replay_ids": replay_ids,
            "fit_selected_baseline": baseline,
            "fit_static": fit_static,
            "fit_router": fitted.model_dump(mode="json"),
            "replay_baseline": replay_baseline,
            "replay_router": replay.model_dump(mode="json"),
            "replay_quality_retention": retention,
            "replay_cost_savings": savings,
            "policy_path": str(policy_path.resolve()),
            "outer_heldout_rows_read": 0,
        },
    )


def _subset(matrix: OutcomeMatrix, ids: list[str]) -> OutcomeMatrix:
    wanted = set(ids)
    return OutcomeMatrix(
        pool=matrix.pool,
        outcomes=[outcome for outcome in matrix.outcomes if outcome.scenario_id in wanted],
    )


def _missing_fit_matrix(
    matrix: OutcomeMatrix,
    fit_ids: list[str],
    *,
    baseline: str,
    seed: int,
    coverage: float,
) -> OutcomeMatrix:
    """Mask nonbaseline fit cells deterministically for the missing-cell ablation."""
    if not 0 < coverage <= 1:
        raise ValueError("missing-cell coverage must be in (0, 1]")
    fit = set(fit_ids)
    kept: set[tuple[str, str]] = set()
    for model in matrix.model_names():
        model_ids = sorted(
            {
                outcome.scenario_id
                for outcome in matrix.outcomes
                if outcome.model == model and outcome.scenario_id in fit
            },
            key=lambda scenario_id: (
                hashlib.sha256(f"missing-v1:{seed}:{model}:{scenario_id}".encode()).digest(),
                scenario_id,
            ),
        )
        keep_count = len(model_ids) if model == baseline else max(1, int(len(model_ids) * coverage))
        kept.update((scenario_id, model) for scenario_id in model_ids[:keep_count])
    return OutcomeMatrix(
        pool=matrix.pool,
        outcomes=[
            outcome
            for outcome in matrix.outcomes
            if outcome.scenario_id not in fit or (outcome.scenario_id, outcome.model) in kept
        ],
    )


def _cell_map(matrix: OutcomeMatrix) -> dict[tuple[str, str], ScenarioOutcome]:
    return {(outcome.scenario_id, outcome.model): outcome for outcome in matrix.outcomes}


def _benchmark(scenario_id: str) -> str:
    benchmark = scenario_id.split(":", 1)[0]
    if benchmark not in BENCHMARKS:
        raise ValueError(f"unexpected benchmark prefix in {scenario_id}")
    return benchmark


def _percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def _metrics(
    matrix: OutcomeMatrix,
    ids: list[str],
    assignments: dict[str, str],
    *,
    baseline: str,
    decisions: dict[str, RoutingDecision] | None = None,
) -> Metric:
    cells = _cell_map(matrix)
    by_benchmark: dict[str, list[ScenarioOutcome]] = {name: [] for name in BENCHMARKS}
    selected: list[ScenarioOutcome] = []
    mix: Counter[str] = Counter()
    reverts = 0
    novelty = 0
    for scenario_id in ids:
        model = assignments[scenario_id]
        row = cells[(scenario_id, model)]
        selected.append(row)
        by_benchmark[_benchmark(scenario_id)].append(row)
        mix[model] += 1
        if decisions is not None:
            evidence = decisions[scenario_id].evidence
            reverts += int(evidence is not None and evidence.gate == "reverted")
            novelty += int(evidence is not None and evidence.gate == "novelty-abstain")

    per_benchmark: dict[str, dict[str, float]] = {}
    for benchmark, rows in by_benchmark.items():
        if not rows:
            raise ValueError(f"no {benchmark} rows in metric cohort")
        per_benchmark[benchmark] = {
            "quality": statistics.fmean(cast("list[float]", [row.reward for row in rows])),
            "cost_per_task": statistics.fmean(row.cost_usd for row in rows),
            "success_rate": statistics.fmean(float(row.success) for row in rows),
            "scenarios": float(len(rows)),
        }
    quality = sum(BENCHMARK_WEIGHT * per_benchmark[name]["quality"] for name in BENCHMARKS)
    cost_per_task = sum(
        BENCHMARK_WEIGHT * per_benchmark[name]["cost_per_task"] for name in BENCHMARKS
    )
    success_rate = sum(
        BENCHMARK_WEIGHT * per_benchmark[name]["success_rate"] for name in BENCHMARKS
    )
    latencies = [sum(row.call_seconds) for row in selected]
    total_cost = sum(row.cost_usd for row in selected)
    successes = sum(row.success for row in selected)
    n = len(ids)
    return Metric(
        quality=quality,
        cost_per_task=cost_per_task,
        total_cost=total_cost,
        effective_cost_per_success=(total_cost / successes if successes else None),
        success_rate=success_rate,
        latency_p50_s=_percentile(latencies, 0.50),
        latency_p95_s=_percentile(latencies, 0.95),
        model_mix={model: count / n for model, count in sorted(mix.items())},
        route_away_rate=1.0 - mix.get(baseline, 0) / n,
        guard_reversion_rate=reverts / n,
        novelty_abstention_rate=novelty / n,
        per_benchmark=per_benchmark,
        scenarios=n,
    )


def _capability_metric(
    matrix: OutcomeMatrix,
    ids: list[str],
    assignments: dict[str, str],
    *,
    baseline_assignments: dict[str, str],
) -> dict[str, object]:
    """Score one overlapping capability cohort against aligned baseline rows."""
    cells = _cell_map(matrix)
    selected = [cells[(scenario_id, assignments[scenario_id])] for scenario_id in ids]
    baseline = [cells[(scenario_id, baseline_assignments[scenario_id])] for scenario_id in ids]
    quality = statistics.fmean(cast("list[float]", [row.reward for row in selected]))
    baseline_quality = statistics.fmean(cast("list[float]", [row.reward for row in baseline]))
    cost = statistics.fmean(row.cost_usd for row in selected)
    baseline_cost = statistics.fmean(row.cost_usd for row in baseline)
    retention = (
        quality / baseline_quality if baseline_quality > 0 else float(quality >= baseline_quality)
    )
    return {
        "scenarios": len(ids),
        "quality": quality,
        "baseline_quality": baseline_quality,
        "quality_retained": retention,
        "absolute_quality_delta": quality - baseline_quality,
        "cost_per_task": cost,
        "baseline_cost_per_task": baseline_cost,
        "cost_savings": 1.0 - cost / baseline_cost if baseline_cost > 0 else 0.0,
        "success_rate": statistics.fmean(float(row.success) for row in selected),
        "latency_p50_s": _percentile(
            [sum(row.call_seconds) for row in selected],
            0.50,
        ),
        "latency_p95_s": _percentile(
            [sum(row.call_seconds) for row in selected],
            0.95,
        ),
        "model_mix": {
            model: count / len(ids)
            for model, count in sorted(
                Counter(assignments[scenario_id] for scenario_id in ids).items()
            )
        },
    }


def _capability_metrics(
    matrix: OutcomeMatrix,
    ids: list[str],
    assignments: dict[str, dict[str, str]],
    *,
    baseline: str,
) -> dict[str, dict[str, dict[str, object]]]:
    """Score named policy points on every declared pre-call capability cohort."""
    wanted = set(ids)
    baseline_assignments = _static_assignments(ids, baseline)
    result: dict[str, dict[str, dict[str, object]]] = {}
    for capability, slice_ids in _capability_slice_ids(matrix).items():
        cohort = [scenario_id for scenario_id in slice_ids if scenario_id in wanted]
        if not cohort:
            continue
        result[capability] = {
            point: _capability_metric(
                matrix,
                cohort,
                point_assignments,
                baseline_assignments=baseline_assignments,
            )
            for point, point_assignments in assignments.items()
        }
    return result


def _static_assignments(ids: list[str], model: str) -> dict[str, str]:
    return dict.fromkeys(ids, model)


def _guard_gate(decision: RoutingDecision) -> str | None:
    evidence = decision.evidence
    return evidence.gate if evidence is not None else None


def _best_single(matrix: OutcomeMatrix, ids: list[str]) -> str:
    """Best 0.5/0.5 quality aggregate, then lower cost, then frozen pool order."""
    order = {entry.name: index for index, entry in enumerate(matrix.pool)}
    scored: list[tuple[float, float, int, str]] = []
    for model in matrix.model_names():
        metric = _metrics(
            matrix,
            ids,
            _static_assignments(ids, model),
            baseline=model,
        )
        scored.append((-metric.quality, metric.cost_per_task, order[model], model))
    return min(scored)[-1]


def _cheapest_single(matrix: OutcomeMatrix, ids: list[str]) -> str:
    order = {entry.name: index for index, entry in enumerate(matrix.pool)}
    return min(
        matrix.model_names(),
        key=lambda model: (
            _metrics(matrix, ids, _static_assignments(ids, model), baseline=model).cost_per_task,
            order[model],
        ),
    )


def _fastest_single(matrix: OutcomeMatrix, ids: list[str]) -> str:
    """Select a latency-only static ablation on outer-fit rows."""
    order = {entry.name: index for index, entry in enumerate(matrix.pool)}
    return min(
        matrix.model_names(),
        key=lambda model: (
            _metrics(
                matrix,
                ids,
                _static_assignments(ids, model),
                baseline=model,
            ).latency_p50_s,
            order[model],
        ),
    )


def _split(root: Path, seed: int) -> tuple[list[str], list[str]]:
    raw = _read_object(root / "splits" / f"seed-{seed}.json")
    fit: list[str] = []
    heldout: list[str] = []
    for benchmark in BENCHMARKS:
        row = raw.get(benchmark)
        if not isinstance(row, dict):
            raise ValueError(f"seed {seed} has no {benchmark} split")
        for side, target in (("fit", fit), ("heldout", heldout)):
            ids = row.get(side)
            if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
                raise ValueError(f"seed {seed} {benchmark} {side} is invalid")
            target.extend(f"{benchmark}:{item}" for item in ids)
    if set(fit) & set(heldout):
        raise ValueError(f"seed {seed} fit and heldout overlap")
    return fit, heldout


def _group(scenario_id: str, manifests: dict[str, dict[str, dict[str, object]]]) -> tuple[str, str]:
    benchmark, task_id = scenario_id.split(":", 1)
    value = manifests[benchmark][task_id].get("group")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{scenario_id} has no frozen group")
    return benchmark, value


def _inner_fold(
    scenario_id: str,
    *,
    seed: int,
    manifests: dict[str, dict[str, dict[str, object]]],
) -> int:
    benchmark, group = _group(scenario_id, manifests)
    digest = hashlib.sha256(f"inner-v1:{seed}:{benchmark}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % INNER_FOLDS


def _embedder(
    matrix: OutcomeMatrix,
    config: RouterConfig,
    root: Path,
) -> tuple[EmbedderSpec, Embedder]:
    if config.embedder == "hashing-1024":
        spec = EmbedderSpec(kind="hashing", dim=1024)
        return spec, spec.build()
    path = root / "embeddings" / "text-embedding-3-large-3072.npy"
    if not path.is_file():
        raise ValueError(
            f"{path} is missing; build the paid label-free embedding cache only after the "
            "experiment spend ceiling is authorized"
        )
    cached = CachedEmbedder(matrix, np.load(path))
    return (
        EmbedderSpec(
            kind="openai",
            dim=OPENAI_EMBEDDING_DIM,
            deployment=OPENAI_EMBEDDING_MODEL,
        ),
        cached,
    )


def _fit_policy(
    matrix: OutcomeMatrix,
    train_ids: list[str],
    baseline: str,
    config: RouterConfig,
    root: Path,
    bank_path: Path,
) -> tuple[RoutingPolicy, Embedder]:
    spec, embedder = _embedder(matrix, config, root)
    fitted = fit_knn_policy(
        matrix,
        bank_path=bank_path,
        fit_ids=train_ids,
        embedder=spec,
        embed_with=embedder,
        guard_model=baseline,
        rag_num=config.neighbors,
        rag_thres=config.similarity_threshold,
        z=config.guard_z,
        min_pairs=config.min_pairs,
        se_floor=config.se_floor,
        floor_q=config.novelty_quantile,
        fitted_from=f"{EXPERIMENT_ID} nested-fit",
    )
    knobs = cost_quality_knobs(config.cost_quality_dial)
    policy = fitted.model_copy(
        update={
            "guard_mode": "asymmetric" if config.asymmetric_guard else "symmetric",
            "pick_lam": knobs.pick_lam,
            "cost_quality": config.cost_quality_dial,
        }
    )
    policy.attach_bank(fitted.knn_bank())
    return policy, embedder


def _retention(router: Metric, baseline: Metric) -> float:
    if baseline.quality <= 0:
        return 1.0 if router.quality >= baseline.quality else 0.0
    return router.quality / baseline.quality


def _savings(router: Metric, baseline: Metric) -> float:
    if baseline.cost_per_task <= 0:
        return 0.0
    return 1.0 - router.cost_per_task / baseline.cost_per_task


def _passes_quality(router: Metric, baseline: Metric) -> bool:
    if _retention(router, baseline) < QUALITY_RETENTION_GATE:
        return False
    for benchmark in BENCHMARKS:
        quality = router.per_benchmark[benchmark]["quality"]
        base = baseline.per_benchmark[benchmark]["quality"]
        retention = quality / base if base > 0 else float(quality >= base)
        if retention < BENCHMARK_RETENTION_FLOOR or base - quality > BENCHMARK_ABSOLUTE_LOSS_LIMIT:
            return False
    return True


def _inner_evaluate(
    matrix: OutcomeMatrix,
    manifests: dict[str, dict[str, dict[str, object]]],
    *,
    root: Path,
    seed: int,
    fit_ids: list[str],
    baseline: str,
    config: RouterConfig,
) -> tuple[Metric, Metric]:
    """Five-fold group CV on one outer fit partition, with no outer-heldout rows present."""
    fit_matrix = _subset(matrix, fit_ids)
    baseline_metric = _metrics(
        fit_matrix,
        fit_ids,
        _static_assignments(fit_ids, baseline),
        baseline=baseline,
    )
    assignments: dict[str, str] = {}
    decisions: dict[str, RoutingDecision] = {}
    scratch = root / "analysis" / "search-scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        directory_path = Path(directory)
        for fold in range(INNER_FOLDS):
            validation = [
                sid for sid in fit_ids if _inner_fold(sid, seed=seed, manifests=manifests) == fold
            ]
            validation_set = set(validation)
            train = [sid for sid in fit_ids if sid not in validation_set]
            if not validation:
                raise ValueError(f"seed {seed} inner fold {fold} is empty")
            if not train:
                raise ValueError(f"seed {seed} inner fold {fold} has no training rows")
            policy, embedder = _fit_policy(
                fit_matrix,
                train,
                baseline,
                config,
                root,
                directory_path / f"fold-{fold}-{KNN_BANK_FILENAME}",
            )
            fold_decisions = route_scenarios(
                policy,
                fit_matrix,
                validation,
                embedder=embedder,
            )
            decisions.update(fold_decisions)
            assignments.update({sid: decision.model for sid, decision in fold_decisions.items()})
    if set(assignments) != set(fit_ids):
        missing = sorted(set(fit_ids) - assignments.keys())
        raise ValueError(f"seed {seed} inner CV omitted {len(missing)} fit scenarios")
    return (
        _metrics(
            fit_matrix,
            fit_ids,
            assignments,
            baseline=baseline,
            decisions=decisions,
        ),
        baseline_metric,
    )


def _config_key(config: RouterConfig) -> str:
    return json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))


def _search_seed(
    matrix: OutcomeMatrix,
    manifests: dict[str, dict[str, dict[str, object]]],
    *,
    root: Path,
    seed: int,
    fit_ids: list[str],
    baseline: str,
) -> tuple[RouterConfig, list[dict[str, object]], Metric, Metric]:
    cache: dict[str, tuple[Metric, Metric]] = {}

    def evaluate(config: RouterConfig) -> tuple[Metric, Metric]:
        key = _config_key(config)
        if key not in cache:
            cache[key] = _inner_evaluate(
                matrix,
                manifests,
                root=root,
                seed=seed,
                fit_ids=fit_ids,
                baseline=baseline,
                config=config,
            )
        return cache[key]

    current = RouterConfig()
    history: list[dict[str, object]] = []
    for search_pass in range(2):
        for coordinate, values in SEARCH_SPACE:
            candidates: list[tuple[RouterConfig, Metric, Metric, bool, int]] = []
            for value_order, value in enumerate(values):
                candidate = replace(current, **{coordinate: value})
                router_metric, baseline_metric = evaluate(candidate)
                candidates.append(
                    (
                        candidate,
                        router_metric,
                        baseline_metric,
                        _passes_quality(router_metric, baseline_metric),
                        value_order,
                    )
                )
            feasible = [row for row in candidates if row[3]]
            if feasible:
                selected = min(
                    feasible,
                    key=lambda row: (
                        row[1].cost_per_task,
                        -row[1].quality,
                        row[4],
                    ),
                )
            else:
                selected = min(
                    candidates,
                    key=lambda row: (
                        -_retention(row[1], row[2]),
                        -row[1].quality,
                        row[1].cost_per_task,
                        row[4],
                    ),
                )
            current = selected[0]
            history.append(
                {
                    "pass": search_pass + 1,
                    "coordinate": coordinate,
                    "selected_value": getattr(current, coordinate),
                    "selected_config": asdict(current),
                    "feasible_values": [
                        getattr(row[0], coordinate) for row in candidates if row[3]
                    ],
                    "candidates": [
                        {
                            "value": getattr(row[0], coordinate),
                            "quality": row[1].quality,
                            "cost_per_task": row[1].cost_per_task,
                            "retention": _retention(row[1], row[2]),
                            "feasible": row[3],
                            "per_benchmark": row[1].per_benchmark,
                        }
                        for row in candidates
                    ],
                }
            )
    router_metric, baseline_metric = evaluate(current)
    return current, history, router_metric, baseline_metric


def _consensus_config(configs: list[RouterConfig]) -> RouterConfig:
    values: dict[str, object] = {}
    for coordinate, order in SEARCH_SPACE:
        selected = [getattr(config, coordinate) for config in configs]
        counts = Counter(selected)
        maximum = max(counts.values())
        tied = {value for value, count in counts.items() if count == maximum}
        values[coordinate] = next(value for value in order if value in tied)
    return _config_from_mapping(values)


def _consensus_baseline(
    matrix: OutcomeMatrix,
    root: Path,
    baselines: list[str],
) -> str:
    counts = Counter(baselines)
    maximum = max(counts.values())
    tied = {model for model, count in counts.items() if count == maximum}
    if len(tied) == 1:
        return next(iter(tied))
    order = {entry.name: index for index, entry in enumerate(matrix.pool)}
    quality: dict[str, list[float]] = defaultdict(list)
    costs: dict[str, list[float]] = defaultdict(list)
    for seed in SEEDS:
        fit_ids, _ = _split(root, seed)
        for model in tied:
            metric = _metrics(
                matrix,
                fit_ids,
                _static_assignments(fit_ids, model),
                baseline=model,
            )
            quality[model].append(metric.quality)
            costs[model].append(metric.cost_per_task)
    return min(
        tied,
        key=lambda model: (
            -statistics.fmean(quality[model]),
            statistics.fmean(costs[model]),
            order[model],
        ),
    )


def _select(root: Path) -> None:
    lock_path = root / "analysis" / "selection-lock.json"
    if lock_path.exists():
        raise ValueError(f"{lock_path} already exists; selection is immutable")
    matrix, manifests = _canonical_matrix(root)
    seed_rows: list[dict[str, object]] = []
    configs: list[RouterConfig] = []
    baselines: list[str] = []
    for seed in SEEDS:
        fit_ids, _ = _split(root, seed)
        baseline = _best_single(matrix, fit_ids)
        config, history, router_metric, baseline_metric = _search_seed(
            matrix,
            manifests,
            root=root,
            seed=seed,
            fit_ids=fit_ids,
            baseline=baseline,
        )
        configs.append(config)
        baselines.append(baseline)
        seed_rows.append(
            {
                "seed": seed,
                "baseline": baseline,
                "config": asdict(config),
                "inner_router": router_metric.json(),
                "inner_baseline": baseline_metric.json(),
                "inner_retention": _retention(router_metric, baseline_metric),
                "inner_savings": _savings(router_metric, baseline_metric),
                "search_history": history,
            }
        )
    consensus = _consensus_config(configs)
    lock = {
        "experiment_id": EXPERIMENT_ID,
        "selection_protocol": "nested-group-cv-v2",
        "matrix_sha256": _sha256(root / "full" / "outcomes.json"),
        "pool_sha256": _sha256(root / "pool.toml"),
        "split_sha256": {
            str(seed): _sha256(root / "splits" / f"seed-{seed}.json") for seed in SEEDS
        },
        "code_commit": _git_commit(),
        "benchmark_weights": dict.fromkeys(BENCHMARKS, BENCHMARK_WEIGHT),
        "quality_retention_gate": QUALITY_RETENTION_GATE,
        "savings_gate": SAVINGS_GATE,
        "benchmark_retention_floor": BENCHMARK_RETENTION_FLOOR,
        "benchmark_absolute_loss_limit": BENCHMARK_ABSOLUTE_LOSS_LIMIT,
        "seeds": seed_rows,
        "deployment_consensus_config": asdict(consensus),
        "deployment_consensus_baseline": _consensus_baseline(matrix, root, baselines),
        "heldout_evaluated": False,
    }
    _write_json(lock_path, lock)


def _raw_knn_decision(policy: RoutingPolicy, query: np.ndarray) -> RoutingDecision:
    """Guard ablation: native bank/profile/cost tilt with no novelty or baseline veto."""
    bank = policy.knn_bank()
    vector = np.asarray(query, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm:
        vector /= norm
    similarities = bank.embeddings @ vector
    budget = min(policy.rag_num, similarities.shape[0])
    kth = float(np.sort(similarities)[-budget])
    rows = np.flatnonzero(similarities > policy.rag_thres * kth)
    if rows.size == 0:
        rows = np.asarray([int(np.argmax(similarities))])
    weights = np.clip(similarities[rows], 0.0, None).astype(np.float64)[:, None]
    rewards = bank.rewards[rows].astype(np.float64)
    scored = ~np.isnan(rewards)
    totals = (scored * weights).sum(axis=0)
    profile = np.full(totals.shape, np.nan)
    np.divide(
        (np.where(scored, np.nan_to_num(rewards), 0.0) * weights).sum(axis=0),
        totals,
        out=profile,
        where=totals > 0,
    )
    costs = bank.mean_costs()
    priced = np.where(np.isnan(costs), policy.cost_scale, costs)
    tilt = (
        policy.pick_lam * priced / policy.cost_scale
        if policy.pick_lam > 0 and policy.cost_scale > 0
        else np.zeros(profile.shape)
    )
    order = {entry.name: index for index, entry in enumerate(policy.pool)}
    candidates = [
        (index, model) for index, model in enumerate(bank.models) if not np.isnan(profile[index])
    ]
    if not candidates:
        return RoutingDecision(
            model=policy.default_model,
            reason="unguarded kNN: no scored neighbors",
            evidence=RoutingEvidence(propensity="fallback-forced"),
        )
    index, model = max(
        candidates,
        key=lambda item: (profile[item[0]] - tilt[item[0]], -order[item[1]]),
    )
    return RoutingDecision(
        model=model,
        reason=f"unguarded kNN: {rows.size} neighbors, profile={profile[index]:.3f}",
        evidence=RoutingEvidence(propensity="greedy"),
    )


def _unguarded_decisions(
    policy: RoutingPolicy,
    matrix: OutcomeMatrix,
    ids: list[str],
    embedder: Embedder,
) -> dict[str, RoutingDecision]:
    _, tasks = _scenario_tasks(matrix)
    vectors = np.asarray(embedder.embed([tasks[scenario_id] for scenario_id in ids]))
    return {
        scenario_id: _raw_knn_decision(policy, vectors[index])
        for index, scenario_id in enumerate(ids)
    }


def _oracle_assignments(matrix: OutcomeMatrix, ids: list[str]) -> dict[str, str]:
    cells = _cell_map(matrix)
    order = {entry.name: index for index, entry in enumerate(matrix.pool)}
    return {
        scenario_id: min(
            matrix.model_names(),
            key=lambda model: (
                -cast("float", cells[(scenario_id, model)].reward),
                cells[(scenario_id, model)].cost_usd,
                order[model],
            ),
        )
        for scenario_id in ids
    }


def _random_assignments(matrix: OutcomeMatrix, ids: list[str], seed: int) -> dict[str, str]:
    models = matrix.model_names()
    return {
        scenario_id: models[
            int.from_bytes(
                hashlib.sha256(f"random-v1:{seed}:{scenario_id}".encode()).digest()[:8],
                "big",
            )
            % len(models)
        ]
        for scenario_id in ids
    }


def _policy_with_dial(policy: RoutingPolicy, dial: float) -> RoutingPolicy:
    updated = policy.model_copy(
        update={
            "pick_lam": cost_quality_knobs(dial).pick_lam,
            "cost_quality": dial,
        }
    )
    updated.attach_bank(policy.knn_bank())
    return updated


def _seed_evaluation(
    matrix: OutcomeMatrix,
    root: Path,
    *,
    seed: int,
    lock_row: dict[str, object],
) -> dict[str, object]:
    fit_ids, heldout_ids = _split(root, seed)
    baseline = cast("str", lock_row["baseline"])
    config = _config_from_mapping(lock_row.get("config"))
    seed_dir = root / "analysis" / "policies" / f"seed-{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    bank_path = seed_dir / KNN_BANK_FILENAME
    policy, embedder = _fit_policy(
        matrix,
        fit_ids,
        baseline,
        config,
        root,
        bank_path,
    )
    policy.save(seed_dir / "policy.json")

    decisions = route_scenarios(
        policy,
        matrix,
        heldout_ids,
        embedder=embedder,
    )
    guarded = _metrics(
        matrix,
        heldout_ids,
        {sid: decision.model for sid, decision in decisions.items()},
        baseline=baseline,
        decisions=decisions,
    )
    baseline_metric = _metrics(
        matrix,
        heldout_ids,
        _static_assignments(heldout_ids, baseline),
        baseline=baseline,
    )
    unguarded_decisions = _unguarded_decisions(policy, matrix, heldout_ids, embedder)
    unguarded = _metrics(
        matrix,
        heldout_ids,
        {sid: decision.model for sid, decision in unguarded_decisions.items()},
        baseline=baseline,
        decisions=unguarded_decisions,
    )
    stratified_decisions: dict[str, RoutingDecision] = {}
    for benchmark in BENCHMARKS:
        benchmark_fit = [sid for sid in fit_ids if _benchmark(sid) == benchmark]
        benchmark_heldout = [sid for sid in heldout_ids if _benchmark(sid) == benchmark]
        benchmark_policy, benchmark_embedder = _fit_policy(
            matrix,
            benchmark_fit,
            baseline,
            config,
            root,
            seed_dir / f"benchmark-{benchmark}-{KNN_BANK_FILENAME}",
        )
        stratified_decisions.update(
            route_scenarios(
                benchmark_policy,
                matrix,
                benchmark_heldout,
                embedder=benchmark_embedder,
            )
        )
    missing_matrix = _missing_fit_matrix(
        matrix,
        fit_ids,
        baseline=baseline,
        seed=seed,
        coverage=MISSING_CELL_COVERAGE,
    )
    missing_policy, missing_embedder = _fit_policy(
        missing_matrix,
        fit_ids,
        baseline,
        config,
        root,
        seed_dir / f"missing-coverage-0.8-{KNN_BANK_FILENAME}",
    )
    missing_decisions = route_scenarios(
        missing_policy,
        missing_matrix,
        heldout_ids,
        embedder=missing_embedder,
    )
    cheapest = _cheapest_single(matrix, fit_ids)
    fastest = _fastest_single(matrix, fit_ids)
    random_assignments = _random_assignments(matrix, heldout_ids, seed)
    oracle_assignments = _oracle_assignments(matrix, heldout_ids)

    rank = fit_rank_policy(
        matrix,
        fit_ids=fit_ids,
        embedder=EmbedderSpec(kind="hashing", dim=1024),
        n_clusters=64,
        seed=seed,
        default_model=baseline,
        fitted_from=f"{EXPERIMENT_ID} seed={seed} rank-ablation",
    )
    rank_decisions = route_scenarios(rank, matrix, heldout_ids)
    baseline_assignments = _static_assignments(heldout_ids, baseline)
    cheapest_assignments = _static_assignments(heldout_ids, cheapest)
    fastest_assignments = _static_assignments(heldout_ids, fastest)
    guarded_assignments = {sid: decision.model for sid, decision in decisions.items()}
    unguarded_assignments = {sid: decision.model for sid, decision in unguarded_decisions.items()}
    rank_assignments = {sid: decision.model for sid, decision in rank_decisions.items()}
    stratified_assignments = {sid: decision.model for sid, decision in stratified_decisions.items()}
    missing_assignments = {sid: decision.model for sid, decision in missing_decisions.items()}
    capability_assignments: dict[str, dict[str, str]] = {
        "best_single": baseline_assignments,
        "cheapest_single": cheapest_assignments,
        "cost_only": cheapest_assignments,
        "latency_only": fastest_assignments,
        "random": random_assignments,
        "unguarded_knn": unguarded_assignments,
        "guarded_knn": guarded_assignments,
        "rank": rank_assignments,
        "oracle": oracle_assignments,
        "ablation:benchmark_stratified": stratified_assignments,
        "ablation:missing_fit_coverage_0.8": missing_assignments,
    }
    points: dict[str, dict[str, object]] = {
        "best_single": baseline_metric.json(),
        "cheapest_single": _metrics(
            matrix,
            heldout_ids,
            cheapest_assignments,
            baseline=baseline,
        ).json(),
        "cost_only": _metrics(
            matrix,
            heldout_ids,
            cheapest_assignments,
            baseline=baseline,
        ).json(),
        "latency_only": _metrics(
            matrix,
            heldout_ids,
            fastest_assignments,
            baseline=baseline,
        ).json(),
        "random": _metrics(
            matrix,
            heldout_ids,
            random_assignments,
            baseline=baseline,
        ).json(),
        "unguarded_knn": unguarded.json(),
        "guarded_knn": guarded.json(),
        "rank": _metrics(
            matrix,
            heldout_ids,
            rank_assignments,
            baseline=baseline,
            decisions=rank_decisions,
        ).json(),
        "oracle": _metrics(
            matrix,
            heldout_ids,
            oracle_assignments,
            baseline=baseline,
        ).json(),
        "ablation:benchmark_stratified": _metrics(
            matrix,
            heldout_ids,
            stratified_assignments,
            baseline=baseline,
            decisions=stratified_decisions,
        ).json(),
        "ablation:missing_fit_coverage_0.8": _metrics(
            matrix,
            heldout_ids,
            missing_assignments,
            baseline=baseline,
            decisions=missing_decisions,
        ).json(),
    }
    for model in matrix.model_names():
        point_id = f"static:{model}"
        assignments = _static_assignments(heldout_ids, model)
        capability_assignments[point_id] = assignments
        points[point_id] = _metrics(
            matrix,
            heldout_ids,
            assignments,
            baseline=baseline,
        ).json()
    for dial in cast("tuple[float, ...]", SEARCH_SPACE[-1][1]):
        dial_policy = _policy_with_dial(policy, dial)
        dial_decisions = route_scenarios(
            dial_policy,
            matrix,
            heldout_ids,
            embedder=embedder,
        )
        point_id = f"guarded_dial:{dial:g}"
        assignments = {sid: decision.model for sid, decision in dial_decisions.items()}
        capability_assignments[point_id] = assignments
        points[point_id] = _metrics(
            matrix,
            heldout_ids,
            assignments,
            baseline=baseline,
            decisions=dial_decisions,
        ).json()

    cells = _cell_map(matrix)
    paired = [
        {
            "scenario_id": sid,
            "benchmark": _benchmark(sid),
            "router_model": decisions[sid].model,
            "router_reward": cells[(sid, decisions[sid].model)].reward,
            "baseline_reward": cells[(sid, baseline)].reward,
            "router_cost": cells[(sid, decisions[sid].model)].cost_usd,
            "baseline_cost": cells[(sid, baseline)].cost_usd,
            "guard_gate": _guard_gate(decisions[sid]),
        }
        for sid in heldout_ids
    ]
    return {
        "seed": seed,
        "fit_scenarios": len(fit_ids),
        "heldout_scenarios": len(heldout_ids),
        "baseline": baseline,
        "cheapest_fit_single": cheapest,
        "fastest_fit_single": fastest,
        "config": asdict(config),
        "points": points,
        "capability_slices": _capability_metrics(
            matrix,
            heldout_ids,
            capability_assignments,
            baseline=baseline,
        ),
        "paired": paired,
        "guarded_retention": _retention(guarded, baseline_metric),
        "guarded_absolute_quality_delta": guarded.quality - baseline_metric.quality,
        "guarded_savings": _savings(guarded, baseline_metric),
        "guarded_quality_gate": _passes_quality(guarded, baseline_metric),
    }


def _bootstrap(
    seed_rows: list[dict[str, object]],
    manifests: dict[str, dict[str, dict[str, object]]],
) -> dict[str, float]:
    by_seed_benchmark_group: dict[int, dict[str, dict[str, list[tuple[float, float]]]]] = {
        seed: {benchmark: defaultdict(list) for benchmark in BENCHMARKS} for seed in SEEDS
    }
    for seed_row in seed_rows:
        seed = cast("int", seed_row["seed"])
        paired = seed_row.get("paired")
        if not isinstance(paired, list):
            raise ValueError(f"seed {seed} has no paired heldout rows")
        for item in paired:
            if not isinstance(item, dict):
                raise ValueError(f"seed {seed} has an invalid paired row")
            pair_row = _object_mapping(item, label=f"seed {seed} paired row")
            scenario_id = _string(pair_row.get("scenario_id"), label=f"seed {seed} scenario_id")
            benchmark, group = _group(scenario_id, manifests)
            by_seed_benchmark_group[seed][benchmark][group].append(
                (
                    _number(pair_row.get("router_reward"), label="router_reward"),
                    _number(pair_row.get("baseline_reward"), label="baseline_reward"),
                )
            )

    rng = random.Random(20260728)
    retention: list[float] = []
    deltas: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        seed_router: list[float] = []
        seed_baseline: list[float] = []
        for seed in SEEDS:
            benchmark_router: list[float] = []
            benchmark_baseline: list[float] = []
            for benchmark in BENCHMARKS:
                grouped = by_seed_benchmark_group[seed][benchmark]
                groups = sorted(grouped)
                sampled = [groups[rng.randrange(len(groups))] for _ in groups]
                pairs = [pair for group in sampled for pair in grouped[group]]
                benchmark_router.append(statistics.fmean(pair[0] for pair in pairs))
                benchmark_baseline.append(statistics.fmean(pair[1] for pair in pairs))
            seed_router.append(statistics.fmean(benchmark_router))
            seed_baseline.append(statistics.fmean(benchmark_baseline))
        router_quality = statistics.fmean(seed_router)
        baseline_quality = statistics.fmean(seed_baseline)
        retention.append(
            router_quality / baseline_quality
            if baseline_quality > 0
            else float(router_quality >= baseline_quality)
        )
        deltas.append(router_quality - baseline_quality)
    return {
        "samples": float(BOOTSTRAP_SAMPLES),
        "retention_lower_95": _percentile(retention, 0.025),
        "retention_median": _percentile(retention, 0.50),
        "retention_upper_95": _percentile(retention, 0.975),
        "absolute_delta_lower_95": _percentile(deltas, 0.025),
        "absolute_delta_median": _percentile(deltas, 0.50),
        "absolute_delta_upper_95": _percentile(deltas, 0.975),
    }


def _aggregate_points(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    names = sorted(
        set.intersection(*[set(cast("dict[str, object]", row["points"])) for row in seed_rows])
    )
    points: list[dict[str, object]] = []
    for name in names:
        metrics = [
            cast("dict[str, object]", cast("dict[str, object]", row["points"])[name])
            for row in seed_rows
        ]
        quality = statistics.fmean(
            _number(metric.get("quality"), label=f"{name}.quality") for metric in metrics
        )
        cost = statistics.fmean(
            _number(metric.get("cost_per_task"), label=f"{name}.cost") for metric in metrics
        )
        points.append(
            {
                "id": name,
                "quality": quality,
                "cost_per_task": cost,
                "success_rate": statistics.fmean(
                    _number(metric.get("success_rate"), label=f"{name}.success")
                    for metric in metrics
                ),
                "latency_p50_s": statistics.fmean(
                    _number(metric.get("latency_p50_s"), label=f"{name}.latency_p50")
                    for metric in metrics
                ),
                "latency_p95_s": statistics.fmean(
                    _number(metric.get("latency_p95_s"), label=f"{name}.latency_p95")
                    for metric in metrics
                ),
                "on_frontier": False,
            }
        )
    for point in points:
        point["on_frontier"] = not any(
            other["id"] != point["id"]
            and _number(other.get("cost_per_task"), label="point.cost")
            <= _number(point.get("cost_per_task"), label="point.cost")
            and _number(other.get("quality"), label="point.quality")
            >= _number(point.get("quality"), label="point.quality")
            and (
                _number(other.get("cost_per_task"), label="point.cost")
                < _number(point.get("cost_per_task"), label="point.cost")
                or _number(other.get("quality"), label="point.quality")
                > _number(point.get("quality"), label="point.quality")
            )
            for other in points
        )
    return sorted(
        points,
        key=lambda point: (
            _number(point.get("cost_per_task"), label="point.cost"),
            -_number(point.get("quality"), label="point.quality"),
        ),
    )


def _aggregate_capability_slices(
    seed_rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Aggregate overlapping capability results without changing headline weights."""
    by_seed: list[dict[str, object]] = []
    for row in seed_rows:
        slices = row.get("capability_slices")
        if not isinstance(slices, dict):
            raise ValueError("seed result has no capability slices")
        by_seed.append({str(key): value for key, value in slices.items()})
    capabilities = sorted(set().union(*(rows.keys() for rows in by_seed)))
    result: dict[str, dict[str, object]] = {}
    numeric_fields = (
        "quality",
        "baseline_quality",
        "quality_retained",
        "absolute_quality_delta",
        "cost_per_task",
        "baseline_cost_per_task",
        "cost_savings",
        "success_rate",
        "latency_p50_s",
        "latency_p95_s",
        "scenarios",
    )
    for capability in capabilities:
        observed = [
            _object_mapping(rows[capability], label=f"capability {capability}")
            for rows in by_seed
            if capability in rows
        ]
        point_names = sorted(
            set.intersection(*[set(_object_mapping(row, label=capability)) for row in observed])
        )
        points: dict[str, object] = {}
        for point_name in point_names:
            metrics = [
                _object_mapping(row[point_name], label=f"{capability}.{point_name}")
                for row in observed
            ]
            model_mixes = [
                _object_mapping(metric.get("model_mix"), label="model_mix") for metric in metrics
            ]
            models = sorted(set().union(*(mix.keys() for mix in model_mixes)))
            points[point_name] = {
                **{
                    field: statistics.fmean(
                        _number(
                            metric.get(field),
                            label=f"{capability}.{point_name}.{field}",
                        )
                        for metric in metrics
                    )
                    for field in numeric_fields
                },
                "model_mix": {
                    model: statistics.fmean(
                        _number(
                            mix.get(model, 0.0),
                            label=f"{capability}.{point_name}.model_mix.{model}",
                        )
                        for mix in model_mixes
                    )
                    for model in models
                },
            }
        result[capability] = {
            "seeds_observed": len(observed),
            "points": points,
        }
    return result


def _deploy_refit(
    matrix: OutcomeMatrix,
    root: Path,
    *,
    config: RouterConfig,
    baseline: str,
) -> None:
    deploy = root / "analysis" / "deployable"
    deploy.mkdir(parents=True, exist_ok=True)
    policy, _ = _fit_policy(
        matrix,
        matrix.scenario_ids(),
        baseline,
        config,
        root,
        deploy / KNN_BANK_FILENAME,
    )
    policy.save(deploy / "policy.json")


def _evaluate(root: Path) -> None:
    lock_path = root / "analysis" / "selection-lock.json"
    if not lock_path.is_file():
        raise ValueError("selection-lock.json is required before heldout evaluation")
    final_path = root / "analysis" / "outer-results.json"
    if final_path.exists():
        raise ValueError(f"{final_path} already exists; heldout evaluation is immutable")
    lock = _read_object(lock_path)
    if lock.get("heldout_evaluated") is not False:
        raise ValueError("selection lock does not authorize a first heldout evaluation")
    if lock.get("matrix_sha256") != _sha256(root / "full" / "outcomes.json"):
        raise ValueError("outcome matrix changed after selection lock")
    if lock.get("code_commit") != _git_commit():
        raise ValueError("code commit changed after selection lock")
    split_digests = lock.get("split_sha256")
    if not isinstance(split_digests, dict):
        raise ValueError("selection lock has no split digests")
    for seed in SEEDS:
        if split_digests.get(str(seed)) != _sha256(root / "splits" / f"seed-{seed}.json"):
            raise ValueError(f"seed {seed} split changed after selection lock")

    matrix, manifests = _canonical_matrix(root)
    lock_seeds = lock.get("seeds")
    if not isinstance(lock_seeds, list) or len(lock_seeds) != len(SEEDS):
        raise ValueError("selection lock does not carry five seed selections")
    partial_path = root / "analysis" / "outer-results.partial.json"
    partial = _read_object(partial_path) if partial_path.is_file() else {"seeds": []}
    completed_raw = partial.get("seeds")
    if not isinstance(completed_raw, list):
        raise ValueError("partial outer result has an invalid seed list")
    completed: dict[int, dict[str, object]] = {}
    for item in completed_raw:
        if not isinstance(item, dict):
            continue
        row = _object_mapping(item, label="partial seed row")
        raw_seed = row.get("seed")
        if isinstance(raw_seed, int) and not isinstance(raw_seed, bool):
            completed[raw_seed] = row
    for seed in SEEDS:
        if seed not in completed:
            lock_row = next(
                row for row in lock_seeds if isinstance(row, dict) and row.get("seed") == seed
            )
            completed[seed] = _seed_evaluation(
                matrix,
                root,
                seed=seed,
                lock_row=cast("dict[str, object]", lock_row),
            )
            _write_json(partial_path, {"seeds": [completed[key] for key in sorted(completed)]})

    seed_rows: list[dict[str, object]] = [completed[seed] for seed in SEEDS]
    bootstrap = _bootstrap(seed_rows, manifests)
    seed_promotions = [
        _boolean(row.get("guarded_quality_gate"), label="guarded_quality_gate")
        and _number(row.get("guarded_retention"), label="guarded_retention")
        >= QUALITY_RETENTION_GATE
        and _number(row.get("guarded_savings"), label="guarded_savings") >= SAVINGS_GATE
        for row in seed_rows
    ]
    promoted = all(seed_promotions) and bootstrap["retention_lower_95"] >= QUALITY_RETENTION_GATE
    result = {
        "experiment_id": EXPERIMENT_ID,
        "matrix_sha256": lock["matrix_sha256"],
        "selection_lock_sha256": _sha256(lock_path),
        "seeds": seed_rows,
        "paired_cluster_bootstrap": bootstrap,
        "all_seed_promotion_gates": seed_promotions,
        "promoted": promoted,
        "pareto": _aggregate_points(seed_rows),
        "capability_slices": _aggregate_capability_slices(seed_rows),
        "one_at_a_time_ablations": {
            "benchmark_stratified": "ablation:benchmark_stratified",
            "missing_fit_coverage_0.8": "ablation:missing_fit_coverage_0.8",
            "missing_fit_coverage_1.0_control": "guarded_knn",
            "latency_only_static": "latency_only",
            "production_eligible": False,
        },
        "deployment_consensus_config": lock["deployment_consensus_config"],
        "deployment_consensus_baseline": lock["deployment_consensus_baseline"],
        "headline_scope": "five-seed nested selection; deployable all-row refit is not heldout",
    }
    _deploy_refit(
        matrix,
        root,
        config=_config_from_mapping(lock.get("deployment_consensus_config")),
        baseline=cast("str", lock["deployment_consensus_baseline"]),
    )
    _write_json(final_path, result)
    _write_json(
        root / "analysis" / "evaluation-complete.json",
        {
            "selection_lock_sha256": _sha256(lock_path),
            "outer_results_sha256": _sha256(final_path),
            "heldout_evaluated": True,
        },
    )


def _validate(root: Path) -> None:
    matrix, manifests = _canonical_matrix(root)
    expected = set(_expected_scenarios(manifests))
    for seed in SEEDS:
        fit, heldout = _split(root, seed)
        if set(fit) | set(heldout) != expected:
            raise ValueError(f"seed {seed} does not partition the exact manifest")
        folds = Counter(_inner_fold(sid, seed=seed, manifests=manifests) for sid in fit)
        if set(folds) != set(range(INNER_FOLDS)):
            raise ValueError(f"seed {seed} has empty inner folds: {dict(folds)}")
    logger.info(
        f"validated {len(matrix.scenario_ids())} scenarios x {len(matrix.pool)} models, "
        "five outer splits and five group-preserving inner folds"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("develop", "validate", "select", "evaluate"))
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(".wmo") / "experiments" / EXPERIMENT_ID,
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    root = cast("Path", args.root).resolve()
    if args.phase == "develop":
        _develop(root)
    elif args.phase == "validate":
        _validate(root)
    elif args.phase == "select":
        _select(root)
    else:
        _evaluate(root)


if __name__ == "__main__":
    main()
