"""Optimize a model and reasoning-effort policy on published DeepSWE v1.1 outcomes.

The benchmark is already a dense execution-scored counterfactual table, so this runner spends
nothing on model calls. It converts the published trials to WMO's ``OutcomeMatrix``, keeps
reasoning effort in the arm identity, drops tasks rather than arms when a cell is missing, and
uses graded fail-to-pass reward.

Policy selection is nested and repository grouped. Each outer split is touched once. Inside the
outer fit set, the selector compares:

* a static cost-quality frontier rule; and
* WMO's guarded kNN with the small set of DeepSWE-sensitive cost knobs.

If task-text routing does not beat the static frontier, the production artifact is deliberately
static. That is an optimization result, not a router failure disguised as a dynamic policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import statistics
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from wmo.core.files import write_text_atomic
from wmo.optimize.knn import best_single_on_fit, fit_knn_policy
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.policy import (
    KNN_BANK_FILENAME,
    POLICY_FILENAME,
    EmbedderSpec,
    RoutingPolicy,
)
from wmo.optimize.routing import route_scenarios
from wmo.providers.base import (
    Completion,
    Message,
    ProviderConfig,
    ProviderKind,
    TokenUsage,
    VerifyResult,
)
from wmo.providers.pool import PoolEntry
from wmo.serving.chat import EndpointRuntime, RequestLog, create_chat_router

logger = logging.getLogger("coding-model-router-deepswe")

BENCHMARK = "deepswe-1.1"
OUTER_SEEDS = (11, 23, 37, 41, 59)
STATIC_QUALITY_FLOORS = (0.95, 0.96, 0.97, 0.98, 0.99, 1.0)
KNN_Z_VALUES = (0.0, 0.5, 1.0, 2.0)
KNN_LAM_VALUES = (0.0, 0.01, 0.02, 0.03)
QUALITY_RETENTION_GATE = 0.95
COST_SAVINGS_GATE = 0.40
OUTER_FIT_FRACTION = 0.70
INNER_FOLDS = 5
BOOTSTRAP_SAMPLES = 10_000
PROMPT_BOILERPLATE = (
    "\nIMPORTANT: Please work on this in a new branch from main and "
    "commit everything when you are done.\n"
)
# Published list prices, USD per million tokens, fetched for the source experiment on
# 2026-07-28. DeepSWE's measured ``cost_usd`` remains the optimization cost. These rates keep
# the pool snapshot honest and usable by later request-cost tooling.
MODEL_PRICES: dict[str, tuple[float, float, float, float]] = {
    # model: (uncached input, cached input, output, cache write)
    "gpt-5.4": (2.50, 0.25, 15.00, 2.50),
    "gpt-5.5": (5.00, 0.50, 30.00, 5.00),
    "gpt-5.6-luna": (1.00, 0.10, 6.00, 1.25),
    "gpt-5.6-terra": (2.50, 0.25, 15.00, 3.125),
    "gpt-5.6-sol": (5.00, 0.50, 30.00, 6.25),
    "claude-fable-5": (10.00, 1.00, 50.00, 12.50),
    "claude-opus-4-8": (5.00, 0.50, 25.00, 6.25),
    "claude-opus-5": (5.00, 0.50, 25.00, 6.25),
    "claude-sonnet-4-6": (3.00, 0.30, 15.00, 3.75),
    "claude-sonnet-5": (3.00, 0.30, 15.00, 3.75),
}


@dataclass(frozen=True)
class Candidate:
    """One mechanically searchable policy configuration."""

    kind: Literal["static", "knn"]
    quality_floor: float
    knn_z: float = 0.0
    pick_lam: float = 0.0

    @property
    def key(self) -> str:
        if self.kind == "static":
            return f"static-q{self.quality_floor:.2f}"
        return f"knn-q{self.quality_floor:.2f}-z{self.knn_z:g}-lam{self.pick_lam:g}"


@dataclass(frozen=True)
class Pair:
    """One paired candidate and fit-selected-baseline observation."""

    scenario_id: str
    group: str
    candidate_reward: float
    candidate_cost: float
    baseline_reward: float
    baseline_cost: float


@dataclass(frozen=True)
class LoadedDeepSwe:
    """Complete-case DeepSWE matrix and pre-inference metadata."""

    matrix: OutcomeMatrix
    groups: dict[str, str]
    dropped_tasks: tuple[str, ...]
    source_hashes: dict[str, str]


class CachedTaskEmbedder:
    """Serve a task-id keyed embedding cache by the corresponding prompt text."""

    def __init__(
        self,
        matrix: OutcomeMatrix,
        vectors: dict[str, list[float]],
    ) -> None:
        tasks: dict[str, str] = {}
        for outcome in matrix.outcomes:
            tasks.setdefault(outcome.scenario_id, outcome.task)
        missing = sorted(set(tasks) - set(vectors))
        if missing:
            raise ValueError(f"embedding cache is missing {len(missing)} DeepSWE tasks")
        widths = {len(vectors[scenario_id]) for scenario_id in tasks}
        if len(widths) != 1 or not widths or next(iter(widths)) < 1:
            raise ValueError("DeepSWE embedding cache has inconsistent or empty vectors")
        self.dim = next(iter(widths))
        self._by_text = {task: vectors[scenario_id] for scenario_id, task in tasks.items()}

    def embed(self, texts: list[str]) -> list[list[float]]:
        missing = [text for text in texts if text not in self._by_text]
        if missing:
            raise KeyError(f"{len(missing)} prompts are absent from the DeepSWE cache")
        return [self._by_text[text] for text in texts]


class _ServingProbeProvider:
    """Credential-free provider used to prove the selected arm reaches WMO serving."""

    def __init__(self, entry: PoolEntry) -> None:
        self.entry = entry
        self.config: ProviderConfig = entry.provider_config()

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        del system, messages, temperature, max_tokens
        return Completion(
            text=f"served by {self.entry.name}",
            usage=TokenUsage(input_tokens=3, output_tokens=2),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError(f"static serving must not embed {len(texts)} prompts")

    def verify(self) -> VerifyResult:
        raise AssertionError("offline serving verification must not call provider verification")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Root containing data/deepswe and results/deepswe_embeddings.json.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(".wmo/experiments/coding-router-deepswe-20260729"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    return parser


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return {str(key): item for key, item in value.items()}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_model(model: str) -> str:
    """Normalize DeepSWE's display spelling to the provider runtime spelling."""
    return re.sub(r"^gpt-(\d+)-(\d+)", r"gpt-\1.\2", model)


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty value list")
    return statistics.fmean(values)


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _load_deepswe(source_root: Path) -> tuple[LoadedDeepSwe, CachedTaskEmbedder]:
    data_root = source_root / "data" / "deepswe"
    trials_path = data_root / "trials.json"
    tasks_path = data_root / "tasks.json"
    task_root = data_root / "deep-swe-main" / "tasks"
    embedding_path = source_root / "results" / "deepswe_embeddings.json"
    paths = (trials_path, tasks_path, embedding_path)
    missing_paths = [str(path) for path in paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"DeepSWE source files are missing: {missing_paths}")

    trials = _read_object(trials_path)
    tasks = _read_object(tasks_path)
    raw_trial_rows = trials.get("rows")
    raw_task_rows = tasks.get("rows")
    if not isinstance(raw_trial_rows, list) or not isinstance(raw_task_rows, list):
        raise ValueError("DeepSWE trials/tasks artifacts have no row lists")
    if trials.get("n_trials") != len(raw_trial_rows) or tasks.get("n_tasks") != len(raw_task_rows):
        raise ValueError("DeepSWE artifact row count disagrees with its header")

    trial_rows = [cast(dict[str, object], row) for row in raw_trial_rows if isinstance(row, dict)]
    task_rows = [cast(dict[str, object], row) for row in raw_task_rows if isinstance(row, dict)]
    task_meta = {str(row["id"]): row for row in task_rows if isinstance(row.get("id"), str)}
    scored = [row for row in trial_rows if row.get("included_in_score") is True]
    all_arms = sorted({str(row["config"]) for row in scored if isinstance(row.get("config"), str)})
    arms = [arm for arm in all_arms if "_gpt_" in arm or "_claude_" in arm]
    if not arms:
        raise ValueError("DeepSWE has no OpenAI or Anthropic model-effort arms")

    representative: dict[str, dict[str, object]] = {}
    cells: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in scored:
        arm = row.get("config")
        task_id = row.get("task_name")
        if not isinstance(arm, str) or arm not in arms or not isinstance(task_id, str):
            continue
        representative.setdefault(arm, row)
        cells[(arm, task_id)].append(row)

    task_ids = sorted(task_meta)
    reward_cells: dict[tuple[str, str], float] = {}
    cost_cells: dict[tuple[str, str], float] = {}
    complete_tasks: list[str] = []
    dropped_tasks: list[str] = []
    for task_id in task_ids:
        complete = True
        for arm in arms:
            rows = cells.get((arm, task_id), [])
            rewards: list[float] = []
            costs: list[float] = []
            for row in rows:
                passed = _number(row.get("f2p_passed"))
                total = _number(row.get("f2p_total"))
                cost = _number(row.get("cost_usd"))
                if passed is not None and total is not None and total > 0:
                    rewards.append(passed / total)
                if cost is not None:
                    costs.append(cost)
            if not rewards or not costs:
                complete = False
                continue
            reward_cells[(task_id, arm)] = _mean(rewards)
            cost_cells[(task_id, arm)] = _mean(costs)
        (complete_tasks if complete else dropped_tasks).append(task_id)
    if len(complete_tasks) < 2:
        raise ValueError("DeepSWE complete-case cohort has fewer than two tasks")

    pool: list[PoolEntry] = []
    for arm in arms:
        row = representative[arm]
        model = row.get("model")
        if not isinstance(model, str):
            raise ValueError(f"DeepSWE arm {arm} has no provider model id")
        effort = row.get("reasoning_effort")
        if effort is not None and not isinstance(effort, str):
            raise ValueError(f"DeepSWE arm {arm} has an invalid reasoning effort")
        runtime_model = _runtime_model(model)
        prices = MODEL_PRICES.get(runtime_model)
        if prices is None:
            raise ValueError(f"DeepSWE arm {arm} has no frozen provider price")
        pool.append(
            PoolEntry(
                name=arm,
                kind=(ProviderKind.OPENAI_RESPONSES if "_gpt_" in arm else ProviderKind.ANTHROPIC),
                model=runtime_model,
                reasoning_effort=effort,
                input_per_mtok=prices[0],
                cached_input_per_mtok=prices[1],
                output_per_mtok=prices[2],
                cache_write_per_mtok=prices[3],
            )
        )

    groups: dict[str, str] = {}
    outcomes: list[ScenarioOutcome] = []
    for task_id in complete_tasks:
        meta = task_meta[task_id]
        repository = meta.get("repository")
        if not isinstance(repository, str) or not repository:
            raise ValueError(f"DeepSWE task {task_id} has no repository group")
        instruction_path = task_root / task_id / "instruction.md"
        if not instruction_path.is_file():
            raise FileNotFoundError(f"DeepSWE task prompt is absent: {instruction_path}")
        prompt = instruction_path.read_text(encoding="utf-8")
        if prompt.endswith(PROMPT_BOILERPLATE):
            prompt = prompt[: -len(PROMPT_BOILERPLATE)]
        groups[task_id] = repository
        for arm in arms:
            reward = reward_cells[(task_id, arm)]
            outcomes.append(
                ScenarioOutcome(
                    scenario_id=task_id,
                    task=prompt,
                    model=arm,
                    benchmark=BENCHMARK,
                    reward=reward,
                    success=reward >= 1.0,
                    cost_usd=cost_cells[(task_id, arm)],
                    completion_status="published_execution_scored",
                    artifact_dir=str(trials_path),
                )
            )

    matrix = OutcomeMatrix(pool=pool, outcomes=outcomes)
    raw_vectors = _read_object(embedding_path)
    vectors: dict[str, list[float]] = {}
    for key, vector in raw_vectors.items():
        if isinstance(vector, list):
            parsed = [_number(value) for value in cast(list[object], vector)]
            if any(value is None for value in parsed):
                raise ValueError(f"DeepSWE embedding {key} contains a non-numeric value")
            vectors[str(key)] = [cast(float, value) for value in parsed]
    embedder = CachedTaskEmbedder(matrix, vectors)
    return (
        LoadedDeepSwe(
            matrix=matrix,
            groups=groups,
            dropped_tasks=tuple(dropped_tasks),
            source_hashes={str(path): _sha256(path) for path in paths},
        ),
        embedder,
    )


def _repository_split(
    scenario_ids: list[str],
    groups: dict[str, str],
    *,
    seed: int,
    fit_fraction: float = OUTER_FIT_FRACTION,
) -> tuple[list[str], list[str]]:
    repositories = sorted({groups[scenario_id] for scenario_id in scenario_ids})
    random.Random(seed).shuffle(repositories)
    fit_count = min(
        len(repositories) - 1,
        max(1, round(len(repositories) * fit_fraction)),
    )
    fit_groups = set(repositories[:fit_count])
    fit = [scenario_id for scenario_id in scenario_ids if groups[scenario_id] in fit_groups]
    heldout = [scenario_id for scenario_id in scenario_ids if groups[scenario_id] not in fit_groups]
    if not fit or not heldout:
        raise ValueError(f"repository split {seed} produced an empty side")
    if {groups[scenario_id] for scenario_id in fit} & {
        groups[scenario_id] for scenario_id in heldout
    }:
        raise AssertionError("repository split leaked a group")
    return fit, heldout


def _repository_folds(
    scenario_ids: list[str],
    groups: dict[str, str],
    *,
    seed: int,
) -> list[tuple[list[str], list[str]]]:
    repositories = sorted({groups[scenario_id] for scenario_id in scenario_ids})
    random.Random(seed).shuffle(repositories)
    fold_count = min(INNER_FOLDS, len(repositories))
    buckets = [set(repositories[index::fold_count]) for index in range(fold_count)]
    folds: list[tuple[list[str], list[str]]] = []
    for bucket in buckets:
        fit = [scenario_id for scenario_id in scenario_ids if groups[scenario_id] not in bucket]
        heldout = [scenario_id for scenario_id in scenario_ids if groups[scenario_id] in bucket]
        if fit and heldout:
            folds.append((fit, heldout))
    if len(folds) < 2:
        raise ValueError("nested selection needs at least two nonempty repository folds")
    return folds


def _cell_lookup(matrix: OutcomeMatrix) -> dict[tuple[str, str], tuple[float, float]]:
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for outcome in matrix.outcomes:
        if outcome.reward is not None:
            grouped[(outcome.scenario_id, outcome.model)].append((outcome.reward, outcome.cost_usd))
    return {
        key: (
            _mean([value[0] for value in values]),
            _mean([value[1] for value in values]),
        )
        for key, values in grouped.items()
    }


def _static_frontier_arm(
    matrix: OutcomeMatrix,
    fit_ids: list[str],
    *,
    quality_floor: float,
    cells: dict[tuple[str, str], tuple[float, float]],
) -> str:
    means: dict[str, tuple[float, float]] = {}
    for model in matrix.model_names():
        values = [cells[(scenario_id, model)] for scenario_id in fit_ids]
        means[model] = (
            _mean([value[0] for value in values]),
            _mean([value[1] for value in values]),
        )
    strongest = best_single_on_fit(matrix, fit_ids)
    required = quality_floor * means[strongest][0]
    eligible = [model for model, (quality, _cost) in means.items() if quality >= required]
    return min(
        eligible,
        key=lambda model: (means[model][1], -means[model][0], matrix.model_names().index(model)),
    )


def _static_pairs(
    model: str,
    baseline: str,
    scenario_ids: list[str],
    groups: dict[str, str],
    cells: dict[tuple[str, str], tuple[float, float]],
) -> list[Pair]:
    return [
        Pair(
            scenario_id=scenario_id,
            group=groups[scenario_id],
            candidate_reward=cells[(scenario_id, model)][0],
            candidate_cost=cells[(scenario_id, model)][1],
            baseline_reward=cells[(scenario_id, baseline)][0],
            baseline_cost=cells[(scenario_id, baseline)][1],
        )
        for scenario_id in scenario_ids
    ]


def _policy_pairs(
    policy: RoutingPolicy,
    baseline: str,
    matrix: OutcomeMatrix,
    scenario_ids: list[str],
    groups: dict[str, str],
    cells: dict[tuple[str, str], tuple[float, float]],
    embedder: CachedTaskEmbedder,
) -> list[Pair]:
    decisions = route_scenarios(policy, matrix, scenario_ids, embedder=embedder)
    if set(decisions) != set(scenario_ids):
        raise ValueError("WMO policy replay did not decide every requested DeepSWE task")
    pairs: list[Pair] = []
    for scenario_id in scenario_ids:
        model = decisions[scenario_id].model
        if (scenario_id, model) not in cells:
            raise ValueError(f"WMO policy routed {scenario_id} to an unscored arm {model}")
        pairs.extend(_static_pairs(model, baseline, [scenario_id], groups, cells))
    return pairs


def _metrics(pairs: list[Pair]) -> dict[str, float | int]:
    candidate_quality = _mean([pair.candidate_reward for pair in pairs])
    baseline_quality = _mean([pair.baseline_reward for pair in pairs])
    candidate_cost = _mean([pair.candidate_cost for pair in pairs])
    baseline_cost = _mean([pair.baseline_cost for pair in pairs])
    return {
        "scenarios": len(pairs),
        "candidate_quality": candidate_quality,
        "baseline_quality": baseline_quality,
        "quality_retention": (candidate_quality / baseline_quality if baseline_quality else 1.0),
        "quality_delta": candidate_quality - baseline_quality,
        "candidate_cost_per_task": candidate_cost,
        "baseline_cost_per_task": baseline_cost,
        "cost_savings": (1.0 - candidate_cost / baseline_cost if baseline_cost else 0.0),
    }


def _candidate_space() -> list[Candidate]:
    candidates = [
        Candidate(kind="static", quality_floor=quality_floor)
        for quality_floor in STATIC_QUALITY_FLOORS
    ]
    candidates.extend(
        Candidate(
            kind="knn",
            quality_floor=quality_floor,
            knn_z=knn_z,
            pick_lam=pick_lam,
        )
        for quality_floor in STATIC_QUALITY_FLOORS
        for knn_z in KNN_Z_VALUES
        for pick_lam in KNN_LAM_VALUES
    )
    return candidates


def _knn_policy(
    matrix: OutcomeMatrix,
    fit_ids: list[str],
    *,
    guard: str,
    embedder: CachedTaskEmbedder,
    embedder_spec: EmbedderSpec,
    bank_path: Path,
) -> RoutingPolicy:
    return fit_knn_policy(
        matrix,
        bank_path=bank_path,
        fit_ids=fit_ids,
        embedder=embedder_spec,
        embed_with=embedder,
        guard_model=guard,
        rag_num=50,
        rag_thres=0.95,
        z=0.0,
        min_pairs=8,
        se_floor=True,
        floor_q=0.0,
        pick_lam=0.0,
        fitted_from=f"{BENCHMARK} repository-grouped fit",
    )


def _select_candidate(
    matrix: OutcomeMatrix,
    fit_ids: list[str],
    groups: dict[str, str],
    *,
    seed: int,
    embedder: CachedTaskEmbedder,
    embedder_spec: EmbedderSpec,
) -> tuple[Candidate, dict[str, dict[str, object]]]:
    cells = _cell_lookup(matrix)
    candidates = _candidate_space()
    by_candidate: dict[str, list[Pair]] = {candidate.key: [] for candidate in candidates}
    fold_metrics: dict[str, list[dict[str, float | int]]] = {
        candidate.key: [] for candidate in candidates
    }
    folds = _repository_folds(fit_ids, groups, seed=1_000 + seed)
    with tempfile.TemporaryDirectory(prefix="wmo-deepswe-inner-") as directory:
        root = Path(directory)
        for fold_index, (inner_fit, inner_heldout) in enumerate(folds):
            baseline = best_single_on_fit(matrix, inner_fit)
            for quality_floor in STATIC_QUALITY_FLOORS:
                guard = _static_frontier_arm(
                    matrix,
                    inner_fit,
                    quality_floor=quality_floor,
                    cells=cells,
                )
                static = Candidate(kind="static", quality_floor=quality_floor)
                static_rows = _static_pairs(
                    guard,
                    baseline,
                    inner_heldout,
                    groups,
                    cells,
                )
                by_candidate[static.key].extend(static_rows)
                fold_metrics[static.key].append(_metrics(static_rows))

                base_policy = _knn_policy(
                    matrix,
                    inner_fit,
                    guard=guard,
                    embedder=embedder,
                    embedder_spec=embedder_spec,
                    bank_path=root / f"fold-{fold_index}-q{quality_floor:.2f}.npz",
                )
                for knn_z in KNN_Z_VALUES:
                    for pick_lam in KNN_LAM_VALUES:
                        candidate = Candidate(
                            kind="knn",
                            quality_floor=quality_floor,
                            knn_z=knn_z,
                            pick_lam=pick_lam,
                        )
                        policy = base_policy.model_copy(
                            update={
                                "knn_z": knn_z,
                                "pick_lam": pick_lam,
                                "guard_mode": "asymmetric",
                            }
                        )
                        rows = _policy_pairs(
                            policy,
                            baseline,
                            matrix,
                            inner_heldout,
                            groups,
                            cells,
                            embedder,
                        )
                        by_candidate[candidate.key].extend(rows)
                        fold_metrics[candidate.key].append(_metrics(rows))

    reports: dict[str, dict[str, object]] = {}
    eligible: list[Candidate] = []
    for candidate in candidates:
        aggregate = _metrics(by_candidate[candidate.key])
        folds_for_candidate = fold_metrics[candidate.key]
        minimum_fold_retention = min(float(row["quality_retention"]) for row in folds_for_candidate)
        report: dict[str, object] = {
            "candidate": asdict(candidate),
            "aggregate": aggregate,
            "minimum_fold_quality_retention": minimum_fold_retention,
            "minimum_fold_cost_savings": min(
                float(row["cost_savings"]) for row in folds_for_candidate
            ),
            "folds": folds_for_candidate,
        }
        reports[candidate.key] = report
        if (
            float(aggregate["quality_retention"]) >= QUALITY_RETENTION_GATE
            and minimum_fold_retention >= QUALITY_RETENTION_GATE
        ):
            eligible.append(candidate)
    if not eligible:
        raise ValueError("no nested DeepSWE candidate retained 95% quality in every inner fold")

    def selection_score(candidate: Candidate) -> tuple[float, float, bool, str]:
        aggregate = cast(dict[str, float | int], reports[candidate.key]["aggregate"])
        return (
            float(aggregate["cost_savings"]),
            float(aggregate["quality_retention"]),
            candidate.kind == "static",
            candidate.key,
        )

    selected = max(
        eligible,
        key=selection_score,
    )
    return selected, reports


def _fit_selected_policy(
    candidate: Candidate,
    matrix: OutcomeMatrix,
    fit_ids: list[str],
    *,
    cells: dict[tuple[str, str], tuple[float, float]],
    embedder: CachedTaskEmbedder,
    embedder_spec: EmbedderSpec,
    bank_path: Path,
) -> tuple[RoutingPolicy, str]:
    guard = _static_frontier_arm(
        matrix,
        fit_ids,
        quality_floor=candidate.quality_floor,
        cells=cells,
    )
    if candidate.kind == "static":
        return (
            RoutingPolicy(
                kind="static",
                default_model=guard,
                pool=matrix.pool,
                fitted_from=f"{BENCHMARK} fit-only static cost-quality frontier",
                fit_scenario_ids=fit_ids,
            ),
            guard,
        )
    policy = _knn_policy(
        matrix,
        fit_ids,
        guard=guard,
        embedder=embedder,
        embedder_spec=embedder_spec,
        bank_path=bank_path,
    ).model_copy(
        update={
            "knn_z": candidate.knn_z,
            "pick_lam": candidate.pick_lam,
            "guard_mode": "asymmetric",
        }
    )
    return policy, guard


def _bootstrap(
    pairs: list[Pair],
    *,
    seed: int,
    samples: int,
) -> dict[str, list[float]]:
    if samples < 1:
        raise ValueError("--bootstrap-samples must be positive")
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, pair in enumerate(pairs):
        by_group[pair.group].append(index)
    group_names = sorted(by_group)
    rng = np.random.default_rng(seed)
    retention = np.empty(samples, dtype=np.float64)
    savings = np.empty(samples, dtype=np.float64)
    margin = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        chosen = rng.choice(group_names, size=len(group_names), replace=True)
        indices = np.concatenate(
            [np.asarray(by_group[str(group)], dtype=np.int64) for group in chosen]
        )
        candidate_reward = _mean([pairs[index].candidate_reward for index in indices])
        baseline_reward = _mean([pairs[index].baseline_reward for index in indices])
        candidate_cost = sum(pairs[index].candidate_cost for index in indices)
        baseline_cost = sum(pairs[index].baseline_cost for index in indices)
        retention[sample] = candidate_reward / baseline_reward if baseline_reward else 1.0
        savings[sample] = 1.0 - candidate_cost / baseline_cost if baseline_cost else 0.0
        margin[sample] = candidate_reward - QUALITY_RETENTION_GATE * baseline_reward
    return {
        "quality_retention_95ci": [
            float(value) for value in np.quantile(retention, [0.025, 0.975])
        ],
        "cost_savings_95ci": [float(value) for value in np.quantile(savings, [0.025, 0.975])],
        "quality_margin_over_95pct_95ci": [
            float(value) for value in np.quantile(margin, [0.025, 0.975])
        ],
    }


def _verify_serving(policy_path: Path, out_root: Path) -> dict[str, object]:
    policy = RoutingPolicy.load(policy_path)
    selected = next(entry for entry in policy.pool if entry.name == policy.default_model)
    constructed: list[_ServingProbeProvider] = []

    def provider_factory(entry: PoolEntry) -> _ServingProbeProvider:
        provider = _ServingProbeProvider(entry)
        constructed.append(provider)
        return provider

    runtime = EndpointRuntime(
        name="deepswe-router",
        policy=policy,
        provider_factory=provider_factory,
        log=RequestLog(out_root / "serving-requests.jsonl"),
    )
    app = FastAPI()
    app.include_router(create_chat_router({"deepswe-router": runtime}))
    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "deepswe-router",
            "messages": [{"role": "user", "content": "repair this repository"}],
        },
    )
    if response.status_code != 200:
        raise ValueError(f"WMO serving verification returned HTTP {response.status_code}")
    routed_model = response.headers.get("x-wmo-routed-model")
    if routed_model != policy.default_model:
        raise ValueError(f"WMO served {routed_model!r}, expected {policy.default_model!r}")
    if len(constructed) != 1:
        raise ValueError("WMO serving verification did not construct exactly one provider")
    provider = constructed[0]
    if provider.entry != selected:
        raise ValueError("WMO serving constructed a pool arm other than the selected arm")
    if provider.config.reasoning_effort != selected.reasoning_effort:
        raise ValueError("WMO serving dropped the selected reasoning effort")

    result: dict[str, object] = {
        "status": "passed",
        "transport": "WMO OpenAI-compatible HTTP route with credential-free probe provider",
        "policy_kind": policy.kind,
        "routed_arm": routed_model,
        "provider_kind": selected.kind.value,
        "provider_model": provider.config.model,
        "reasoning_effort": provider.config.reasoning_effort,
        "response_model": response.json().get("model"),
        "paid_calls": 0,
    }
    write_text_atomic(
        out_root / "serving-verification.json",
        json.dumps(result, indent=2, sort_keys=True) + "\n",
    )
    return result


def _run(
    source_root: Path,
    out_root: Path,
    *,
    bootstrap_samples: int,
) -> dict[str, object]:
    loaded, embedder = _load_deepswe(source_root)
    matrix = loaded.matrix
    scenario_ids = matrix.scenario_ids()
    cells = _cell_lookup(matrix)
    embedder_spec = EmbedderSpec(
        kind="openai",
        deployment="text-embedding-3-large",
        dim=embedder.dim,
    )
    out_root.mkdir(parents=True, exist_ok=True)
    matrix.save(out_root / "matrix.json")

    split_rows: list[dict[str, object]] = []
    for seed in OUTER_SEEDS:
        fit_ids, heldout_ids = _repository_split(
            scenario_ids,
            loaded.groups,
            seed=seed,
        )
        selected, search = _select_candidate(
            matrix,
            fit_ids,
            loaded.groups,
            seed=seed,
            embedder=embedder,
            embedder_spec=embedder_spec,
        )
        policy, guard = _fit_selected_policy(
            selected,
            matrix,
            fit_ids,
            cells=cells,
            embedder=embedder,
            embedder_spec=embedder_spec,
            bank_path=out_root / f"seed-{seed}.knn.npz",
        )
        baseline = best_single_on_fit(matrix, fit_ids)
        pairs = (
            _static_pairs(
                policy.default_model,
                baseline,
                heldout_ids,
                loaded.groups,
                cells,
            )
            if policy.kind == "static"
            else _policy_pairs(
                policy,
                baseline,
                matrix,
                heldout_ids,
                loaded.groups,
                cells,
                embedder,
            )
        )
        metrics = _metrics(pairs)
        intervals = _bootstrap(pairs, seed=seed, samples=bootstrap_samples)
        point_pass = (
            float(metrics["quality_retention"]) >= QUALITY_RETENTION_GATE
            and float(metrics["cost_savings"]) >= COST_SAVINGS_GATE
        )
        split_rows.append(
            {
                "seed": seed,
                "fit_tasks": len(fit_ids),
                "heldout_tasks": len(heldout_ids),
                "fit_repositories": len({loaded.groups[item] for item in fit_ids}),
                "heldout_repositories": len({loaded.groups[item] for item in heldout_ids}),
                "repository_overlap": sorted(
                    {loaded.groups[item] for item in fit_ids}
                    & {loaded.groups[item] for item in heldout_ids}
                ),
                "baseline": baseline,
                "selected": asdict(selected),
                "selected_policy_kind": policy.kind,
                "selected_model": policy.default_model,
                "guard_model": guard,
                "metrics": metrics,
                "intervals": intervals,
                "point_gate_pass": point_pass,
                "quality_ci_supports_gate": (intervals["quality_margin_over_95pct_95ci"][0] >= 0.0),
                "savings_ci_supports_gate": (
                    intervals["cost_savings_95ci"][0] >= COST_SAVINGS_GATE
                ),
                "search": search,
            }
        )
        logger.info(
            "DeepSWE seed %d selected %s %s: retention %.3f, savings %.1f%%",
            seed,
            policy.kind,
            policy.default_model,
            float(metrics["quality_retention"]),
            100 * float(metrics["cost_savings"]),
        )

    production_candidate, production_search = _select_candidate(
        matrix,
        scenario_ids,
        loaded.groups,
        seed=20_260_729,
        embedder=embedder,
        embedder_spec=embedder_spec,
    )
    production_policy, production_guard = _fit_selected_policy(
        production_candidate,
        matrix,
        scenario_ids,
        cells=cells,
        embedder=embedder,
        embedder_spec=embedder_spec,
        bank_path=out_root / KNN_BANK_FILENAME,
    )
    production_policy.save(out_root / POLICY_FILENAME)
    serving_verification = _verify_serving(
        out_root / POLICY_FILENAME,
        out_root,
    )

    point_passes = [bool(row["point_gate_pass"]) for row in split_rows]
    quality_ci_passes = [bool(row["quality_ci_supports_gate"]) for row in split_rows]
    savings_ci_passes = [bool(row["savings_ci_supports_gate"]) for row in split_rows]
    report: dict[str, object] = {
        "status": "complete",
        "benchmark": BENCHMARK,
        "scientific_status": "post_hoc_repeated_grouped_validation",
        "target_labels_used_for_training": True,
        "reward": "mean graded f2p_passed/f2p_total over included published trials",
        "arms": len(matrix.pool),
        "tasks_raw": len(matrix.scenario_ids()) + len(loaded.dropped_tasks),
        "tasks_complete_case": len(matrix.scenario_ids()),
        "repositories": len(set(loaded.groups.values())),
        "dropped_tasks": list(loaded.dropped_tasks),
        "drop_rule": "drop a task across all arms when any reward or cost cell is missing",
        "arm_identity": "provider model plus reasoning effort",
        "source_hashes": loaded.source_hashes,
        "outer_seeds": list(OUTER_SEEDS),
        "splits": split_rows,
        "gate": {
            "quality_retention": QUALITY_RETENTION_GATE,
            "cost_savings": COST_SAVINGS_GATE,
            "point_gate_passes": sum(point_passes),
            "point_gate_total": len(point_passes),
            "point_gate_all_seeds": all(point_passes),
            "quality_ci_supports_gate_seeds": sum(quality_ci_passes),
            "savings_ci_supports_gate_seeds": sum(savings_ci_passes),
            "promotion_ready": (
                all(point_passes) and all(quality_ci_passes) and all(savings_ci_passes)
            ),
        },
        "production": {
            "candidate": asdict(production_candidate),
            "policy_kind": production_policy.kind,
            "model": production_policy.default_model,
            "guard_model": production_guard,
            "policy_path": str((out_root / POLICY_FILENAME).resolve()),
            "knn_bank_path": (
                str((out_root / KNN_BANK_FILENAME).resolve())
                if production_policy.kind == "knn"
                else None
            ),
            "selection": production_search,
        },
        "serving_verification": serving_verification,
        "limitations": [
            (
                "DeepSWE outcomes were inspected before this repeated grouped analysis, "
                "so the result is post hoc."
            ),
            (
                "The benchmark measures long-horizon repository tasks, "
                "not short interactive coding requests."
            ),
            (
                "A static model-effort arm is preferred when nested grouped validation "
                "finds no dynamic routing gain."
            ),
        ],
    }
    write_text_atomic(
        out_root / "report.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    return report


def main() -> None:
    args = _parser().parse_args()
    report = _run(
        args.source_root.resolve(),
        args.out.resolve(),
        bootstrap_samples=args.bootstrap_samples,
    )
    gate = cast(dict[str, object], report["gate"])
    production = cast(dict[str, object], report["production"])
    logger.info(
        "DeepSWE optimization complete: %s %s, point gates %s/%s, promotion_ready=%s",
        production["policy_kind"],
        production["model"],
        gate["point_gate_passes"],
        gate["point_gate_total"],
        gate["promotion_ready"],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
