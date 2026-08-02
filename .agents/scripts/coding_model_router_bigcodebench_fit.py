"""Fit latency-neutral routers on a promoted external BigCodeBench matrix.

The module enforces the held-out-oracle promotion boundary before it reads any
score row. It contains the shared data and evaluation primitives for the frozen
router families. DeepSWE artifacts are outside this script's contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import sparse
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from wmo.core.files import write_text_atomic
from wmo.optimize.knn import fit_knn_policy
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.policy import EmbedderSpec, RoutingPolicy
from wmo.optimize.routing import evaluate_policy, route_scenarios
from wmo.providers.base import ProviderKind
from wmo.providers.pool import PoolEntry

ARMS = ("luna-low", "luna-medium", "luna-high", "luna-xhigh", "luna-max")
ATTEMPTS = 5
EXPECTED_TASKS = 300
EXPECTED_CELLS = EXPECTED_TASKS * len(ARMS) * ATTEMPTS
OUTER_SEEDS = (0, 1, 2, 3, 4)
OUTER_TEST_FRACTION = 0.2


@dataclass(frozen=True)
class FitData:
    """Dense external task, reward, and cost tensors for router fitting."""

    task_ids: list[str]
    groups: list[str]
    texts: list[str]
    is_hard: np.ndarray
    rewards: np.ndarray
    costs: np.ndarray


@dataclass(frozen=True)
class PolicyValue:
    """Observed policy value and its matched task-blind control."""

    reward: float
    cost_usd: float
    matched_blind_reward: float
    matched_blind_cost_usd: float
    arm_counts: dict[str, int]


@dataclass(frozen=True)
class NativeKnnReplay:
    """WMO-native kNN policy and its heldout routes."""

    policy: RoutingPolicy
    choices: np.ndarray
    value: PolicyValue
    bank_path: Path


@dataclass(frozen=True)
class TaskSplit:
    """One deterministic library-grouped outer split."""

    seed: int
    train_indices: np.ndarray
    test_indices: np.ndarray


@dataclass(frozen=True)
class CandidateMetric:
    """Fit-only quality, cost, and serving-footprint summary."""

    name: str
    reward: float
    cost_usd: float
    latency_p95_ms: float
    artifact_bytes: int
    order: int


@dataclass(frozen=True)
class LatencyMetric:
    """Single-request route latency and frozen deployment-gate result."""

    decisions: int
    p50_ms: float
    p95_ms: float
    passed: bool


class RouteOne(Protocol):
    """One latency-neutral route decision callable."""

    def __call__(self, text: str) -> int:
        """Select one frozen effort index."""
        ...


class LockedCandidate(BaseModel):
    """One fit-only selected policy point with canonical configuration provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    family: Literal["knn", "ordinal", "doubly-robust", "empirical-bayes"]
    name: str = Field(min_length=1)
    config_json: str = Field(min_length=2)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fit_reward: float = Field(ge=0.0, le=1.0)
    fit_cost_usd: float = Field(ge=0.0)
    matched_blind_reward: float = Field(ge=0.0, le=1.0)
    latency_p95_ms: float = Field(ge=0.0)
    artifact_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def _canonical_config_matches_digest(self) -> LockedCandidate:
        value = json.loads(self.config_json)
        if not isinstance(value, dict):
            raise ValueError("locked candidate config must be one JSON object")
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
        if canonical != self.config_json:
            raise ValueError("locked candidate config is not canonical JSON")
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        if digest != self.config_sha256:
            raise ValueError("locked candidate config digest differs")
        return self


class SeedSelection(BaseModel):
    """Fit-only decision and split provenance for one outer seed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int
    fit_tasks: int = Field(gt=0)
    heldout_tasks: int = Field(gt=0)
    fit_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    heldout_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_arm: str
    baseline_fit_reward: float = Field(ge=0.0, le=1.0)
    baseline_fit_cost_usd: float = Field(ge=0.0)
    selected: LockedCandidate


class DeploymentConsensus(BaseModel):
    """Single fit-only configuration frozen for later full-source refitting."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol: Literal["bigcodebench-five-seed-fit-consensus-v1"] = (
        "bigcodebench-five-seed-fit-consensus-v1"
    )
    family: Literal["knn", "ordinal", "doubly-robust", "empirical-bayes"]
    name: str = Field(min_length=1)
    order: int = Field(ge=0)
    config_json: str = Field(min_length=2)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mean_fit_reward: float = Field(ge=0.0, le=1.0)
    mean_fit_cost_usd: float = Field(ge=0.0)
    mean_matched_blind_reward: float = Field(ge=0.0, le=1.0)
    mean_baseline_reward: float = Field(gt=0.0, le=1.0)
    minimum_seed_retention: float = Field(ge=0.0)
    fit_quality_feasible: bool
    target_outcomes_used: Literal[False] = False
    outer_heldout_evaluated: Literal[False] = False

    @model_validator(mode="after")
    def _canonical_config_matches_digest(self) -> DeploymentConsensus:
        value = json.loads(self.config_json)
        if not isinstance(value, dict):
            raise ValueError("deployment consensus config must be one JSON object")
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
        if canonical != self.config_json:
            raise ValueError("deployment consensus config is not canonical JSON")
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        if digest != self.config_sha256:
            raise ValueError("deployment consensus config digest differs")
        return self


class SelectionLock(BaseModel):
    """Immutable fit-only boundary required before outer-heldout replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol: Literal["bigcodebench-fit-only-selection-v1"]
    tasks_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scores_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    outcomes_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    oracle_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    code_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    target_outcomes_used: Literal[False] = False
    outer_heldout_evaluated: Literal[False] = False
    seeds: list[SeedSelection] = Field(min_length=5, max_length=5)
    deployment_consensus: DeploymentConsensus

    @model_validator(mode="after")
    def _five_exact_outer_seeds(self) -> SelectionLock:
        if sorted(selection.seed for selection in self.seeds) != list(OUTER_SEEDS):
            raise ValueError("selection lock must contain outer seeds 0 through 4 exactly once")
        return self


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return {str(key): item for key, item in value.items()}


def _read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} contains a non-object row")
        rows.append({str(key): item for key, item in value.items()})
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_candidate_config(value: dict[str, str | int | float | bool]) -> tuple[str, str]:
    """Return canonical JSON and SHA-256 for one frozen candidate configuration."""
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return canonical, hashlib.sha256(canonical.encode()).hexdigest()


def _ordered_id_digest(task_ids: list[str], indices: np.ndarray) -> str:
    payload = "".join(f"{task_ids[int(index)]}\n" for index in indices)
    return hashlib.sha256(payload.encode()).hexdigest()


def seed_split_provenance(data: FitData, split: TaskSplit) -> tuple[str, str]:
    """Return ordered fit and heldout task-id digests for one outer split."""
    return (
        _ordered_id_digest(data.task_ids, split.train_indices),
        _ordered_id_digest(data.task_ids, split.test_indices),
    )


def write_selection_lock(path: Path, lock: SelectionLock) -> None:
    """Atomically publish a fit-only lock before any outer-heldout evaluation."""
    if path.exists():
        raise FileExistsError(f"selection lock already exists: {path}")
    write_text_atomic(path, lock.model_dump_json(indent=2) + "\n")


def require_selection_lock(root: Path, path: Path) -> SelectionLock:
    """Load a selection lock and prove it still names the exact scored matrix."""
    lock = SelectionLock.model_validate_json(path.read_text(encoding="utf-8"))
    current = {
        "tasks_sha256": _sha256(root / "tasks.jsonl"),
        "scores_sha256": _sha256(root / "scores.jsonl"),
        "outcomes_sha256": _sha256(root / "outcomes.jsonl"),
        "oracle_report_sha256": _sha256(root / "oracle-report.json"),
    }
    for field, digest in current.items():
        if getattr(lock, field) != digest:
            raise ValueError(f"selection lock {field} does not match the current artifact")
    oracle = _read_object(root / "oracle-report.json")
    if oracle.get("passed") is not True:
        raise ValueError("selection lock cannot authorize evaluation after a failed oracle")
    protocol = oracle.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("oracle report has no protocol")
    _require_target_safe(
        {str(key): item for key, item in protocol.items()},
        label="oracle report",
    )
    return lock


def _task_id(row: dict[str, object]) -> str:
    value = row.get("task_id")
    if not isinstance(value, str) or not value:
        raise ValueError("row has no task_id")
    return value


def _require_target_safe(value: dict[str, object], *, label: str) -> None:
    if value.get("target_outcomes_used") is not False:
        raise ValueError(f"{label} crossed the target outcome boundary")


def _expected_cell_ids(task_ids: list[str]) -> set[str]:
    return {
        f"{task_id}:{arm}:attempt-{attempt}"
        for task_id in task_ids
        for arm in ARMS
        for attempt in range(ATTEMPTS)
    }


def load_fit_data(root: Path) -> FitData:
    """Load a promoted external matrix after validating every frozen boundary."""
    oracle = _read_object(root / "oracle-report.json")
    protocol = oracle.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("oracle report has no protocol")
    _require_target_safe(
        {str(key): item for key, item in protocol.items()},
        label="oracle report",
    )
    if oracle.get("passed") is not True:
        raise ValueError("external oracle did not pass; router fitting is forbidden")

    matrix = _read_object(root / "matrix-manifest.json")
    score_manifest = _read_object(root / "score-manifest.json")
    _require_target_safe(matrix, label="matrix manifest")
    _require_target_safe(score_manifest, label="score manifest")
    tasks = _read_rows(root / "tasks.jsonl")
    if len(tasks) != EXPECTED_TASKS:
        raise ValueError(f"expected {EXPECTED_TASKS} tasks, found {len(tasks)}")
    if (
        matrix.get("cells") != EXPECTED_CELLS
        or score_manifest.get("cells") != EXPECTED_CELLS
        or score_manifest.get("scores_sha256") != _sha256(root / "scores.jsonl")
    ):
        raise ValueError("matrix or score manifest is incomplete or changed")

    task_ids = [_task_id(task) for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task manifest contains duplicate ids")
    task_index = {task_id: index for index, task_id in enumerate(task_ids)}
    arm_index = {arm: index for index, arm in enumerate(ARMS)}
    expected = _expected_cell_ids(task_ids)

    score_rows = _read_rows(root / "scores.jsonl")
    outcome_rows = _read_rows(root / "outcomes.jsonl")
    score_by_cell: dict[str, dict[str, object]] = {}
    outcome_by_cell: dict[str, dict[str, object]] = {}
    for label, rows, destination in (
        ("score", score_rows, score_by_cell),
        ("outcome", outcome_rows, outcome_by_cell),
    ):
        for row in rows:
            _require_target_safe(row, label=f"{label} row")
            cell_id = row.get("cell_id")
            if not isinstance(cell_id, str) or cell_id in destination:
                raise ValueError(f"{label} rows contain a missing or duplicate cell id")
            destination[cell_id] = row
        if set(destination) != expected:
            raise ValueError(f"{label} rows do not match the frozen dense matrix")

    rewards = np.full((len(tasks), len(ARMS), ATTEMPTS), np.nan)
    costs = np.full_like(rewards, np.nan)
    for cell_id in sorted(expected):
        score = score_by_cell[cell_id]
        outcome = outcome_by_cell[cell_id]
        task_id = _task_id(score)
        arm = score.get("arm")
        attempt = score.get("attempt")
        if (
            task_id != _task_id(outcome)
            or arm != outcome.get("arm")
            or attempt != outcome.get("attempt")
            or not isinstance(arm, str)
            or arm not in arm_index
            or not isinstance(attempt, int)
            or not 0 <= attempt < ATTEMPTS
        ):
            raise ValueError(f"score and outcome identity differ for {cell_id}")
        index = (task_index[task_id], arm_index[arm], attempt)
        rewards[index] = float(cast(float, score["reward"]))
        costs[index] = float(cast(float, outcome["cost_usd"]))
    if not np.isfinite(rewards).all() or not np.isfinite(costs).all():
        raise ValueError("reward or cost tensor is not finite and dense")

    return FitData(
        task_ids=task_ids,
        groups=[str(task["library_group"]) for task in tasks],
        texts=[str(task["instruct_prompt"]) for task in tasks],
        is_hard=np.asarray([bool(task.get("is_hard")) for task in tasks]),
        rewards=rewards,
        costs=costs,
    )


def _structural_row(text: str, *, is_hard: bool) -> list[float]:
    lower = text.casefold()
    lines = text.splitlines()
    words = text.split()
    return [
        math.log1p(len(text)),
        math.log1p(len(words)),
        math.log1p(len(lines)),
        float(text.count("`")),
        float(text.count("\n")),
        float(lower.count("import ")),
        float(lower.count("raise ")),
        float(lower.count("assert ")),
        float(lower.count("example")),
        float(lower.count("test")),
        float(lower.count("exception")),
        float(lower.count("recursive")),
        float("async" in lower),
        float("class " in lower),
        float(is_hard),
    ]


def feature_matrix(
    data: FitData,
    *,
    dim: int,
    scale_indices: np.ndarray | None = None,
) -> sparse.csr_matrix:
    """Build prompt features with structural scaling learned on fit tasks only."""
    if dim not in {512, 2_048, 8_192}:
        raise ValueError("hash dimension is outside the frozen search space")
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        n_features=dim,
        alternate_sign=True,
        norm="l2",
    )
    text = cast(sparse.csr_matrix, vectorizer.transform(data.texts))
    structural = np.asarray(
        [
            _structural_row(prompt, is_hard=bool(data.is_hard[index]))
            for index, prompt in enumerate(data.texts)
        ],
        dtype=np.float64,
    )
    if scale_indices is None:
        scale_rows = np.arange(len(data.task_ids))
    else:
        scale_rows = np.asarray(
            _indices(data, scale_indices, label="feature scale"),
            dtype=np.int64,
        )
    scale = np.maximum(np.std(structural[scale_rows], axis=0), 1.0)
    structural /= scale
    return sparse.hstack([text, sparse.csr_matrix(structural)], format="csr")


def grouped_folds(groups: list[str], *, splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic family-grouped folds and prove zero overlap."""
    if len(set(groups)) < splits:
        raise ValueError("too few task-family groups for the frozen fold count")
    indices = np.arange(len(groups))
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for train, test in GroupKFold(n_splits=splits).split(indices, groups=groups):
        train_groups = {groups[index] for index in train}
        test_groups = {groups[index] for index in test}
        if train_groups & test_groups:
            raise AssertionError("task-family group crossed a fit boundary")
        result.append((train, test))
    return result


def seeded_outer_split(groups: list[str], *, seed: int) -> TaskSplit:
    """Assign a near-20-percent heldout prefix of deterministically hashed groups."""
    if seed not in OUTER_SEEDS:
        raise ValueError("outer seed is outside the frozen five-seed set")
    if len(groups) < 2 or len(set(groups)) < 2:
        raise ValueError("an outer split needs at least two task-family groups")
    members: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        members.setdefault(group, []).append(index)
    ranked = sorted(
        members,
        key=lambda group: (
            hashlib.sha256(f"bigcodebench-outer-v1:{seed}:{group}".encode()).digest(),
            group,
        ),
    )
    target = round(len(groups) * OUTER_TEST_FRACTION)
    selected: list[str] = []
    selected_count = 0
    for group in ranked:
        next_count = selected_count + len(members[group])
        if selected and abs(selected_count - target) < abs(next_count - target):
            break
        selected.append(group)
        selected_count = next_count
        if selected_count >= target:
            break
    test_groups = set(selected)
    test = np.asarray(
        [index for index, group in enumerate(groups) if group in test_groups],
        dtype=np.int64,
    )
    train = np.asarray(
        [index for index, group in enumerate(groups) if group not in test_groups],
        dtype=np.int64,
    )
    if train.size == 0 or test.size == 0:
        raise ValueError("outer split produced an empty partition")
    if {groups[index] for index in train} & {groups[index] for index in test}:
        raise AssertionError("task-family group crossed an outer boundary")
    return TaskSplit(seed=seed, train_indices=train, test_indices=test)


def outer_splits(groups: list[str]) -> list[TaskSplit]:
    """Return all five preregistered deterministic outer splits."""
    splits = [seeded_outer_split(groups, seed=seed) for seed in OUTER_SEEDS]
    heldout_sets = {tuple(split.test_indices.tolist()) for split in splits}
    if len(heldout_sets) != len(OUTER_SEEDS):
        raise ValueError("outer seeds did not produce five distinct heldout sets")
    return splits


def _indices(data: FitData, values: np.ndarray, *, label: str) -> list[int]:
    indices = [int(value) for value in np.asarray(values).tolist()]
    if len(indices) != len(set(indices)):
        raise ValueError(f"{label} indices contain duplicates")
    if any(index < 0 or index >= len(data.task_ids) for index in indices):
        raise ValueError(f"{label} indices are outside the task matrix")
    if not indices:
        raise ValueError(f"{label} indices are empty")
    return indices


def static_metrics(data: FitData, indices: np.ndarray) -> list[CandidateMetric]:
    """Measure every static effort using only the supplied fit tasks."""
    selected = _indices(data, indices, label="static fit")
    rewards = data.rewards[selected].mean(axis=(0, 2))
    costs = data.costs[selected].mean(axis=(0, 2))
    return [
        CandidateMetric(
            name=arm,
            reward=float(rewards[index]),
            cost_usd=float(costs[index]),
            latency_p95_ms=0.0,
            artifact_bytes=0,
            order=index,
        )
        for index, arm in enumerate(ARMS)
    ]


def fit_selected_static(data: FitData, indices: np.ndarray) -> CandidateMetric:
    """Select the strongest static effort on fit, breaking ties toward lower cost."""
    return min(
        static_metrics(data, indices),
        key=lambda metric: (-metric.reward, metric.cost_usd, metric.order),
    )


def select_fit_candidate(
    candidates: list[CandidateMetric],
    *,
    baseline_reward: float,
    quality_retention: float = 0.95,
) -> CandidateMetric:
    """Pick the least-cost fit-only point that clears the frozen quality floor."""
    if not candidates:
        raise ValueError("fit-only selection received no candidates")
    if baseline_reward <= 0.0:
        raise ValueError("fit-only baseline reward must be positive")
    if not 0.0 < quality_retention <= 1.0:
        raise ValueError("quality retention must be in (0, 1]")
    feasible = [
        candidate
        for candidate in candidates
        if candidate.reward >= quality_retention * baseline_reward
    ]
    if feasible:
        return min(
            feasible,
            key=lambda metric: (
                metric.cost_usd,
                metric.latency_p95_ms,
                metric.artifact_bytes,
                metric.order,
            ),
        )
    return min(
        candidates,
        key=lambda metric: (
            -metric.reward,
            metric.cost_usd,
            metric.latency_p95_ms,
            metric.artifact_bytes,
            metric.order,
        ),
    )


def effort_pool() -> list[PoolEntry]:
    """Return five priced WMO arms for one Luna runtime model."""
    efforts = ("low", "medium", "high", "xhigh", "max")
    return [
        PoolEntry(
            name=arm,
            kind=ProviderKind.OPENAI_RESPONSES,
            model="gpt-5.6-luna",
            reasoning_effort=effort,
            input_per_mtok=1.0,
            cached_input_per_mtok=0.1,
            output_per_mtok=6.0,
        )
        for arm, effort in zip(ARMS, efforts, strict=True)
    ]


def outcome_matrix(data: FitData) -> OutcomeMatrix:
    """Project the dense external tensor into WMO's native outcome contract."""
    expected = (len(data.task_ids), len(ARMS), ATTEMPTS)
    if data.rewards.shape != expected or data.costs.shape != expected:
        raise ValueError("fit tensors do not match the task, arm, and attempt contract")
    if not np.isfinite(data.rewards).all() or not np.isfinite(data.costs).all():
        raise ValueError("fit tensors must be finite and dense")
    outcomes = [
        ScenarioOutcome(
            scenario_id=data.task_ids[task_index],
            task=data.texts[task_index],
            model=arm,
            benchmark="bigcodebench-v0.2.4",
            episode=attempt,
            attempt_number=attempt + 1,
            reward=float(data.rewards[task_index, arm_index, attempt]),
            success=bool(data.rewards[task_index, arm_index, attempt] >= 1.0),
            cost_usd=float(data.costs[task_index, arm_index, attempt]),
            completion_status="scored",
            usage_accounting="exact-or-trace-estimated",
        )
        for task_index in range(len(data.task_ids))
        for arm_index, arm in enumerate(ARMS)
        for attempt in range(ATTEMPTS)
    ]
    return OutcomeMatrix(pool=effort_pool(), outcomes=outcomes)


def fit_native_knn_replay(
    data: FitData,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    bank_path: Path,
    dim: int,
    guard_arm: str,
    rag_num: int,
    rag_thres: float,
    z: float,
    min_pairs: int,
    se_floor: bool,
    floor_q: float,
    pick_lam: float,
    guard_mode: Literal["symmetric", "asymmetric"],
) -> NativeKnnReplay:
    """Fit and replay WMO guarded kNN without any network or target data."""
    train = _indices(data, train_indices, label="train")
    test = _indices(data, test_indices, label="test")
    if set(train) & set(test):
        raise ValueError("train and test tasks overlap")
    if dim not in {512, 2_048, 8_192}:
        raise ValueError("hash dimension is outside the frozen search space")
    if guard_arm not in ARMS:
        raise ValueError("guard arm is outside the frozen effort roster")
    if guard_mode not in {"symmetric", "asymmetric"}:
        raise ValueError("guard mode must be symmetric or asymmetric")

    matrix = outcome_matrix(data)
    spec = EmbedderSpec(kind="hashing", dim=dim)
    train_ids = [data.task_ids[index] for index in train]
    test_ids = [data.task_ids[index] for index in test]
    policy = fit_knn_policy(
        matrix,
        bank_path=bank_path,
        fit_ids=train_ids,
        embedder=spec,
        guard_model=guard_arm,
        rag_num=rag_num,
        rag_thres=rag_thres,
        z=z,
        min_pairs=min_pairs,
        se_floor=se_floor,
        floor_q=floor_q,
        pick_lam=pick_lam,
        fitted_from="bigcodebench-v0.2.4 external fit only",
    ).model_copy(update={"guard_mode": guard_mode})
    decisions = route_scenarios(policy, matrix, test_ids, embedder=spec.build())
    arm_index = {arm: index for index, arm in enumerate(ARMS)}
    choices = np.asarray(
        [arm_index[decisions[task_id].model] for task_id in test_ids],
        dtype=np.int64,
    )
    rewards = data.rewards[test].mean(axis=2)
    costs = data.costs[test].mean(axis=2)
    value = evaluate_choices(rewards, costs, choices)
    native = evaluate_policy(policy, matrix, test_ids, embedder=spec.build())
    if native.unscored_scenarios != 0:
        raise AssertionError("WMO replay selected an unscored effort cell")
    if not math.isclose(native.accuracy, value.reward, rel_tol=1e-6, abs_tol=1e-8):
        raise AssertionError("WMO replay reward differs from tensor evaluation")
    if not math.isclose(native.cost_per_scenario, value.cost_usd, rel_tol=1e-6, abs_tol=1e-8):
        raise AssertionError("WMO replay cost differs from tensor evaluation")
    return NativeKnnReplay(
        policy=policy,
        choices=choices,
        value=value,
        bank_path=bank_path,
    )


def ordinal_ridge_predictions(
    train_features: sparse.csr_matrix,
    test_features: sparse.csr_matrix,
    train_rewards: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """Predict monotone rewards from one low-effort and four adjacent-uplift heads."""
    if train_rewards.ndim != 2 or train_rewards.shape[1] != len(ARMS):
        raise ValueError("ordinal training rewards have the wrong shape")
    targets = np.column_stack([train_rewards[:, 0], np.diff(train_rewards, axis=1)])
    predicted_parts: list[np.ndarray] = []
    for column in range(targets.shape[1]):
        model = Ridge(alpha=alpha)
        model.fit(train_features, targets[:, column])
        predicted_parts.append(np.asarray(model.predict(test_features), dtype=np.float64))
    parts = np.column_stack(predicted_parts)
    absolute = np.column_stack([parts[:, 0], parts[:, 0, None] + np.cumsum(parts[:, 1:], axis=1)])
    return np.maximum.accumulate(np.clip(absolute, 0.0, 1.0), axis=1)


def ordinal_extra_trees_predictions(
    train_features: sparse.csr_matrix,
    test_features: sparse.csr_matrix,
    train_rewards: np.ndarray,
    *,
    n_estimators: int,
    min_samples_leaf: int,
    max_features: Literal["sqrt", "third"],
    random_state: int,
) -> np.ndarray:
    """Predict monotone adjacent-effort rewards with deterministic ExtraTrees heads."""
    if train_rewards.ndim != 2 or train_rewards.shape[1] != len(ARMS):
        raise ValueError("ordinal training rewards have the wrong shape")
    if n_estimators not in {200, 500}:
        raise ValueError("tree count is outside the frozen search space")
    if min_samples_leaf not in {5, 10, 20}:
        raise ValueError("tree leaf size is outside the frozen search space")
    if max_features not in {"sqrt", "third"}:
        raise ValueError("tree feature fraction is outside the frozen search space")
    targets = np.column_stack([train_rewards[:, 0], np.diff(train_rewards, axis=1)])
    predicted_parts: list[np.ndarray] = []
    feature_rule: str | float = "sqrt" if max_features == "sqrt" else 1.0 / 3.0
    for column in range(targets.shape[1]):
        model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=feature_rule,
            random_state=random_state + column,
            n_jobs=1,
        )
        model.fit(train_features, targets[:, column])
        predicted_parts.append(np.asarray(model.predict(test_features), dtype=np.float64))
    parts = np.column_stack(predicted_parts)
    absolute = np.column_stack([parts[:, 0], parts[:, 0, None] + np.cumsum(parts[:, 1:], axis=1)])
    return np.maximum.accumulate(np.clip(absolute, 0.0, 1.0), axis=1)


def doubly_robust_pseudo_values(
    rewards: np.ndarray,
    direct_predictions: np.ndarray,
    *,
    propensity: float = 1.0 / len(ARMS),
) -> np.ndarray:
    """Build multi-action AIPW targets from the dense randomized effort matrix."""
    if rewards.ndim != 3 or rewards.shape[1:] != (len(ARMS), ATTEMPTS):
        raise ValueError("doubly robust rewards have the wrong shape")
    if direct_predictions.shape != rewards.shape[:2]:
        raise ValueError("direct predictions do not match doubly robust rewards")
    if not 0.0 < propensity <= 1.0:
        raise ValueError("propensity must be in (0, 1]")
    tasks = rewards.shape[0]
    pseudo = np.empty((tasks, len(ARMS)), dtype=np.float64)
    for target_arm in range(len(ARMS)):
        values = np.broadcast_to(
            direct_predictions[:, target_arm, None, None],
            (tasks, len(ARMS), ATTEMPTS),
        ).copy()
        residual = (
            rewards[:, target_arm, :] - direct_predictions[:, target_arm, None]
        ) / propensity
        values[:, target_arm, :] += residual
        pseudo[:, target_arm] = values.mean(axis=(1, 2))
    return pseudo


def multi_action_ridge_predictions(
    train_features: sparse.csr_matrix,
    test_features: sparse.csr_matrix,
    pseudo_values: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """Fit one Ridge policy head per doubly robust effort pseudo-value."""
    if pseudo_values.ndim != 2 or pseudo_values.shape[1] != len(ARMS):
        raise ValueError("multi-action pseudo-values have the wrong shape")
    predictions: list[np.ndarray] = []
    for arm_index in range(len(ARMS)):
        model = Ridge(alpha=alpha)
        model.fit(train_features, pseudo_values[:, arm_index])
        predictions.append(np.asarray(model.predict(test_features), dtype=np.float64))
    return np.clip(np.column_stack(predictions), 0.0, 1.0)


def multi_action_hist_predictions(
    train_features: sparse.csr_matrix,
    test_features: sparse.csr_matrix,
    pseudo_values: np.ndarray,
    *,
    max_leaf_nodes: int,
    learning_rate: float,
    min_samples_leaf: int,
    random_state: int,
) -> np.ndarray:
    """Fit deterministic histogram-boosted doubly robust policy heads."""
    if pseudo_values.ndim != 2 or pseudo_values.shape[1] != len(ARMS):
        raise ValueError("multi-action pseudo-values have the wrong shape")
    if max_leaf_nodes not in {7, 15, 31}:
        raise ValueError("histogram leaf count is outside the frozen search space")
    if learning_rate not in {0.03, 0.1}:
        raise ValueError("histogram learning rate is outside the frozen search space")
    if min_samples_leaf not in {10, 20}:
        raise ValueError("histogram minimum leaf size is outside the frozen search space")
    dense_train = np.asarray(train_features.toarray(), dtype=np.float64)
    dense_test = np.asarray(test_features.toarray(), dtype=np.float64)
    predictions: list[np.ndarray] = []
    for arm_index in range(len(ARMS)):
        model = HistGradientBoostingRegressor(
            max_leaf_nodes=max_leaf_nodes,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state + arm_index,
        )
        model.fit(dense_train, pseudo_values[:, arm_index])
        predictions.append(np.asarray(model.predict(dense_test), dtype=np.float64))
    return np.clip(np.column_stack(predictions), 0.0, 1.0)


def shadow_price_choices(
    predicted_rewards: np.ndarray,
    arm_costs: np.ndarray,
    *,
    lam: float,
) -> np.ndarray:
    """Choose reward minus normalized fit-only cost at one frozen shadow price."""
    if predicted_rewards.ndim != 2 or predicted_rewards.shape[1] != len(ARMS):
        raise ValueError("predicted rewards have the wrong effort shape")
    if arm_costs.shape != (len(ARMS),) or not np.isfinite(arm_costs).all():
        raise ValueError("arm costs have the wrong shape or contain non-finite values")
    if np.any(arm_costs < 0.0) or float(np.mean(arm_costs)) <= 0.0:
        raise ValueError("arm costs must be nonnegative with a positive mean")
    if lam not in {0.0, 0.0025, 0.005, 0.01, 0.02, 0.04}:
        raise ValueError("shadow price is outside the frozen search space")
    normalized_cost = arm_costs / float(np.mean(arm_costs))
    return np.argmax(predicted_rewards - lam * normalized_cost[None, :], axis=1).astype(np.int64)


def _family_posterior_moments(
    successes: np.ndarray,
    trials: float,
    global_mean: np.ndarray,
    *,
    prior_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = successes + prior_strength * global_mean
    beta = trials - successes + prior_strength * (1.0 - global_mean)
    total = alpha + beta
    mean = alpha / total
    variance = alpha * beta / (total**2 * (total + 1.0))
    return mean, np.sqrt(np.maximum(variance, 0.0))


def empirical_bayes_family_moments(
    train_groups: list[str],
    test_groups: list[str],
    train_rewards: np.ndarray,
    *,
    prior_strength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return LOO train and heldout beta-binomial means and standard errors."""
    if train_rewards.ndim != 3 or train_rewards.shape[1:] != (len(ARMS), ATTEMPTS):
        raise ValueError("empirical-Bayes rewards have the wrong shape")
    if len(train_groups) != train_rewards.shape[0]:
        raise ValueError("empirical-Bayes groups do not match rewards")
    if prior_strength <= 0.0:
        raise ValueError("prior strength must be positive")
    global_successes = train_rewards.sum(axis=(0, 2))
    global_trials = float(train_rewards.shape[0] * ATTEMPTS)
    global_mean = (global_successes + 0.5) / (global_trials + 1.0)
    family_successes: dict[str, np.ndarray] = {}
    family_trials: dict[str, float] = {}
    for index, group in enumerate(train_groups):
        family_successes.setdefault(group, np.zeros(len(ARMS), dtype=np.float64))
        family_successes[group] += train_rewards[index].sum(axis=1)
        family_trials[group] = family_trials.get(group, 0.0) + ATTEMPTS

    train_mean = np.empty((len(train_groups), len(ARMS)), dtype=np.float64)
    train_se = np.empty_like(train_mean)
    for index, group in enumerate(train_groups):
        successes = family_successes[group] - train_rewards[index].sum(axis=1)
        trials = family_trials[group] - ATTEMPTS
        train_mean[index], train_se[index] = _family_posterior_moments(
            successes,
            trials,
            global_mean,
            prior_strength=prior_strength,
        )
    test_mean = np.empty((len(test_groups), len(ARMS)), dtype=np.float64)
    test_se = np.empty_like(test_mean)
    for index, group in enumerate(test_groups):
        successes = family_successes.get(group, np.zeros(len(ARMS), dtype=np.float64))
        trials = family_trials.get(group, 0.0)
        test_mean[index], test_se[index] = _family_posterior_moments(
            successes,
            trials,
            global_mean,
            prior_strength=prior_strength,
        )
    return train_mean, train_se, test_mean, test_se


def empirical_bayes_family_predictions(
    train_groups: list[str],
    test_groups: list[str],
    train_rewards: np.ndarray,
    *,
    prior_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return leave-one-task-out train and heldout family posterior means."""
    train_mean, _, test_mean, _ = empirical_bayes_family_moments(
        train_groups,
        test_groups,
        train_rewards,
        prior_strength=prior_strength,
    )
    return train_mean, test_mean


def empirical_bayes_ridge_predictions(
    train_features: sparse.csr_matrix,
    test_features: sparse.csr_matrix,
    train_groups: list[str],
    test_groups: list[str],
    train_rewards: np.ndarray,
    *,
    prior_strength: float,
    alpha: float,
) -> np.ndarray:
    """Add local adjacent-effort residual heads to family-shrunk posteriors."""
    train_base, test_base = empirical_bayes_family_predictions(
        train_groups,
        test_groups,
        train_rewards,
        prior_strength=prior_strength,
    )
    observed = train_rewards.mean(axis=2)
    observed_parts = np.column_stack([observed[:, 0], np.diff(observed, axis=1)])
    base_parts = np.column_stack([train_base[:, 0], np.diff(train_base, axis=1)])
    residual_parts: list[np.ndarray] = []
    for column in range(observed_parts.shape[1]):
        model = Ridge(alpha=alpha)
        model.fit(train_features, observed_parts[:, column] - base_parts[:, column])
        residual_parts.append(np.asarray(model.predict(test_features), dtype=np.float64))
    test_parts = np.column_stack([test_base[:, 0], np.diff(test_base, axis=1)]) + np.column_stack(
        residual_parts
    )
    absolute = np.column_stack(
        [
            test_parts[:, 0],
            test_parts[:, 0, None] + np.cumsum(test_parts[:, 1:], axis=1),
        ]
    )
    return np.maximum.accumulate(np.clip(absolute, 0.0, 1.0), axis=1)


def lower_bound_choices(
    predicted_rewards: np.ndarray,
    standard_errors: np.ndarray,
    arm_costs: np.ndarray,
    *,
    quality_floor: float,
    fallback_arm: int,
    z: float,
) -> np.ndarray:
    """Choose the cheapest arm whose lower bound clears a fit-only quality floor."""
    if predicted_rewards.shape != standard_errors.shape or predicted_rewards.ndim != 2:
        raise ValueError("lower-bound means and standard errors differ")
    if predicted_rewards.shape[1] != len(ARMS) or arm_costs.shape != (len(ARMS),):
        raise ValueError("lower-bound inputs have the wrong effort shape")
    if not np.isfinite(arm_costs).all() or np.any(arm_costs < 0.0):
        raise ValueError("lower-bound arm costs must be finite and nonnegative")
    if not 0.0 <= quality_floor <= 1.0:
        raise ValueError("quality floor must be in [0, 1]")
    if not 0 <= fallback_arm < len(ARMS):
        raise ValueError("fallback arm is outside the frozen effort roster")
    if z not in {0.0, 0.5, 1.0, 1.645}:
        raise ValueError("lower-bound z is outside the frozen search space")
    if np.any(standard_errors < 0.0) or not np.isfinite(standard_errors).all():
        raise ValueError("standard errors must be finite and nonnegative")
    order = np.argsort(arm_costs, kind="stable")
    lower = predicted_rewards - z * standard_errors
    choices = np.full(predicted_rewards.shape[0], fallback_arm, dtype=np.int64)
    for row_index in range(predicted_rewards.shape[0]):
        feasible = [
            arm_index for arm_index in order if lower[row_index, arm_index] >= quality_floor
        ]
        if feasible:
            choices[row_index] = feasible[0]
    return choices


def cost_only_choices(costs: np.ndarray) -> np.ndarray:
    """Route every task to the lowest observed fit-cost arm."""
    if costs.ndim != 2 or costs.shape[1] != len(ARMS):
        raise ValueError("cost-only matrix has the wrong effort shape")
    if not np.isfinite(costs).all() or np.any(costs < 0.0):
        raise ValueError("cost-only matrix must be finite and nonnegative")
    arm = int(np.argmin(costs.mean(axis=0)))
    return np.full(costs.shape[0], arm, dtype=np.int64)


def random_choices(tasks: int, *, seed: int) -> np.ndarray:
    """Return deterministic uniform random effort assignments."""
    if tasks <= 0:
        raise ValueError("random control needs at least one task")
    return np.random.default_rng(seed).integers(0, len(ARMS), size=tasks, dtype=np.int64)


def shuffled_task_rewards(rewards: np.ndarray, *, seed: int) -> np.ndarray:
    """Permute complete task reward profiles for a label-destruction control."""
    if rewards.ndim != 3 or rewards.shape[1:] != (len(ARMS), ATTEMPTS):
        raise ValueError("shuffled rewards have the wrong dense effort shape")
    permutation = np.random.default_rng(seed).permutation(rewards.shape[0])
    return rewards[permutation].copy()


def artifact_size(paths: list[Path]) -> int:
    """Return the exact bytes in a fitted policy's declared files."""
    if not paths:
        raise ValueError("artifact size needs at least one file")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"artifact files are missing: {missing[:3]}")
    return sum(path.stat().st_size for path in paths)


def measure_route_latency(
    route_one: RouteOne,
    texts: list[str],
    *,
    decisions: int = 10_000,
    warmup: int = 100,
) -> LatencyMetric:
    """Measure single-request routing against the frozen one-core latency gate."""
    if not texts:
        raise ValueError("latency measurement needs at least one prompt")
    if decisions <= 0 or warmup < 0:
        raise ValueError("latency decision and warmup counts are invalid")
    for index in range(warmup):
        route_one(texts[index % len(texts)])
    elapsed = np.empty(decisions, dtype=np.float64)
    for index in range(decisions):
        started = time.perf_counter_ns()
        route_one(texts[index % len(texts)])
        elapsed[index] = (time.perf_counter_ns() - started) / 1_000_000.0
    p50, p95 = np.quantile(elapsed, np.asarray([0.5, 0.95]))
    return LatencyMetric(
        decisions=decisions,
        p50_ms=float(p50),
        p95_ms=float(p95),
        passed=bool(p50 < 5.0 and p95 < 20.0),
    )


def evaluate_choices(
    rewards: np.ndarray,
    costs: np.ndarray,
    choices: np.ndarray,
) -> PolicyValue:
    """Evaluate routes and the task-blind mixture with identical arm traffic."""
    if rewards.shape != costs.shape or rewards.ndim != 2:
        raise ValueError("policy evaluation matrices differ or are not two-dimensional")
    if rewards.shape[1] != len(ARMS) or choices.shape != (rewards.shape[0],):
        raise ValueError("policy choices do not match the effort matrix")
    if np.any(choices < 0) or np.any(choices >= len(ARMS)):
        raise ValueError("policy selected an unknown effort")
    rows = np.arange(len(choices))
    counts = np.bincount(choices, minlength=len(ARMS))
    traffic = counts / len(choices)
    return PolicyValue(
        reward=float(np.mean(rewards[rows, choices])),
        cost_usd=float(np.mean(costs[rows, choices])),
        matched_blind_reward=float(np.sum(traffic * rewards.mean(axis=0))),
        matched_blind_cost_usd=float(np.sum(traffic * costs.mean(axis=0))),
        arm_counts={arm: int(counts[index]) for index, arm in enumerate(ARMS)},
    )
