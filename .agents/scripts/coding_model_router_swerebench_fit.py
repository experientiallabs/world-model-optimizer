"""Fit and seal latency-neutral SWE-rebench effort routes on E2B.

The fitter consumes only the frozen external development matrix and label-free
confirmation prompts. It writes decisions and audit metadata, never fitted
numeric model state. DeepSWE artifacts are outside this module's contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from coding_model_router_codeforces_fit import (
    ARMS,
    HIGH_INDEX,
    Data,
    _features,
    _fit_delta_models,
    _score_delta_models,
)
from coding_model_router_codeforces_irt import _fit_irt
from scipy import sparse
from scipy.special import expit
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from wmo.optimize.knn import fit_knn_policy
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.policy import EmbedderSpec, RoutingPolicy, knn_decision
from wmo.providers.base import ProviderKind
from wmo.providers.pool import PoolEntry

logger = logging.getLogger("coding-model-router-swerebench-fit")

PROTOCOL = "coding-router-swerebench-fit-only-selection-v1"
DEVELOPMENT_CORPUS_SHA256 = "7d846b5576d15e68fd18ac21bfe0610cc1614b3b35ec0ae0cb8cfae0b82962c1"
CONFIRMATION_CORPUS_SHA256 = "9798dd1e58be0d13331d097307670dc3fc3760ad211da20e6367666523f080a7"
TASKS = 200
MIN_RETAINED_TASKS = 190
ATTEMPTS = 2
EFFORTS = ("low", "medium", "high", "xhigh", "max")
HASH_DIMS = (512, 2_048, 8_192)
RIDGE_ALPHAS = (1.0, 10.0, 100.0)
QUALITY_THRESHOLDS = (0.0, 0.02, 0.05, 0.10, 0.20)
IRT_WEIGHTS = (0.70, 0.80, 0.90, 0.95, 0.98, 0.99)
KNN_COUNTS = (8, 16, 32, 64)
KNN_RELATIVE_THRESHOLD = 0.95
KNN_Z = (0.0, 0.5, 1.0, 1.645, 2.0)
PICK_LAMS = (0.0, 0.01, 0.02, 0.03)
QUALITY_TOLERANCE = 0.005
QUALITY_RETENTION = 0.95
SHUFFLE_SEED = 20_260_731

Family = Literal["direct", "ordinal", "pairwise", "irt", "knn"]


@dataclass(frozen=True)
class SourceData:
    """Dense attempts plus the task-level view used by every candidate."""

    data: Data
    raw_rewards: np.ndarray
    raw_costs: np.ndarray
    languages: list[str]
    repositories: list[str]


@dataclass(frozen=True)
class Candidate:
    """One completely enumerated external-only candidate configuration."""

    family: Family
    order: int
    dim: int = 0
    alpha: float = 0.0
    threshold: float = 0.0
    irt_weight: float = 0.0
    guard: str = ""
    rag_num: int = 0
    z: float = 0.0
    pick_lam: float = 0.0

    @property
    def key(self) -> str:
        """Return the stable candidate identity."""
        if self.family in {"direct", "ordinal", "pairwise"}:
            return (
                f"{self.family}-hash{self.dim}-a{self.alpha:g}"
                f"-t{self.threshold:g}"
            )
        if self.family == "irt":
            return f"irt2pl-hash{self.dim}-a{self.alpha:g}-w{self.irt_weight:g}"
        return (
            f"knn-hash{self.dim}-guard-{self.guard}-k{self.rag_num}"
            f"-z{self.z:g}-lam{self.pick_lam:g}"
        )

    def config(self) -> dict[str, object]:
        """Return a canonicalizable configuration with no implicit defaults."""
        return {
            "family": self.family,
            "dim": self.dim,
            "alpha": self.alpha,
            "threshold": self.threshold,
            "irt_weight": self.irt_weight,
            "guard": self.guard,
            "rag_num": self.rag_num,
            "rag_thres": KNN_RELATIVE_THRESHOLD,
            "floor_q": 0.0,
            "floor_sim": None,
            "z": self.z,
            "min_pairs": 8,
            "se_floor": True,
            "guard_mode": "asymmetric" if self.family == "knn" else "",
            "pick_lam": self.pick_lam,
        }


@dataclass(frozen=True)
class CandidateResult:
    """Grouped out-of-fold value and promotion eligibility."""

    candidate: Candidate
    choices: np.ndarray
    reward: float
    cost_usd: float
    matched_blind_reward: float
    matched_blind_cost_usd: float
    advantage: float
    arm_counts: dict[str, int]
    fold_rows: list[dict[str, object]]
    retention_passed: bool
    dominated_by_static: list[str]

    @property
    def eligible(self) -> bool:
        """Return whether this point may be frozen for confirmation."""
        return (
            self.advantage > 0.0
            and self.retention_passed
            and not self.dominated_by_static
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return {str(key): item for key, item in value.items()}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} contains a non-object row")
        rows.append({str(key): item for key, item in value.items()})
    return rows


def _task_text(task: dict[str, Any]) -> str:
    return (
        f"repository={task['repository']}\n"
        f"language={task['language']}\n"
        f"{task['prompt']}"
    )


def _structural(task: dict[str, Any]) -> list[float]:
    prompt = str(task["prompt"])
    lower = prompt.casefold()
    language = str(task["language"]).casefold()
    return [
        math.log1p(len(prompt)),
        math.log1p(len(prompt.split())),
        math.log1p(len(prompt.splitlines())),
        float(prompt.count("`")),
        float(prompt.count("\n")),
        float(lower.count("test")),
        float(lower.count("error")),
        float(lower.count("exception")),
        float(lower.count("expected")),
        float(lower.count("import ")),
        float(lower.count("class ")),
        float("python" in language),
        float("javascript" in language or "typescript" in language),
        float("rust" in language),
        float("go" == language),
    ]


def load_source(corpus_path: Path, outcomes_path: Path, audit_path: Path) -> SourceData:
    """Load the dense retained-task matrix after whole-task exclusions."""
    if _sha256(corpus_path) != DEVELOPMENT_CORPUS_SHA256:
        raise ValueError("development corpus hash changed")
    audit = _read_object(audit_path)
    retained_tasks = audit.get("tasks")
    exclusions = audit.get("excluded_tasks")
    if (
        audit.get("valid") is not True
        or audit.get("source_tasks") != TASKS
        or not isinstance(retained_tasks, int)
        or not MIN_RETAINED_TASKS <= retained_tasks <= TASKS
        or not isinstance(exclusions, list)
        or len(exclusions) != TASKS - retained_tasks
        or audit.get("retained_task_coverage") != retained_tasks / TASKS
        or audit.get("cells") != retained_tasks * len(ARMS) * ATTEMPTS
        or audit.get("target_outcomes_used") is not False
        or audit.get("deep_swe_outcomes_accessed") is not False
        or audit.get("outcomes_sha256") != _sha256(outcomes_path)
    ):
        raise ValueError("development collection audit is incomplete or unsafe")
    corpus = _read_object(corpus_path)
    raw_tasks = corpus.get("tasks")
    if not isinstance(raw_tasks, list) or len(raw_tasks) != TASKS:
        raise ValueError("development corpus must contain exactly 200 tasks")
    tasks = [
        {str(key): item for key, item in task.items()}
        for task in raw_tasks
        if isinstance(task, dict)
    ]
    if len(tasks) != TASKS:
        raise ValueError("development corpus contains a non-object task")
    source_task_ids = [str(task["task_id"]) for task in tasks]
    if len(set(source_task_ids)) != TASKS:
        raise ValueError("development corpus contains duplicate task IDs")
    excluded_ids: set[str] = set()
    for raw_exclusion in exclusions:
        if not isinstance(raw_exclusion, dict):
            raise ValueError("development exclusion is not an object")
        task_id = raw_exclusion.get("task_id")
        if (
            not isinstance(task_id, str)
            or task_id not in source_task_ids
            or task_id in excluded_ids
            or raw_exclusion.get("scope") != "whole-task"
            or raw_exclusion.get("scientific_cells_rerun") != 0
        ):
            raise ValueError("development exclusion is invalid")
        excluded_ids.add(task_id)
    tasks = [task for task in tasks if str(task["task_id"]) not in excluded_ids]
    if len(tasks) != retained_tasks:
        raise ValueError("retained development task count does not match the audit")
    task_ids = [str(task["task_id"]) for task in tasks]
    task_index = {task_id: index for index, task_id in enumerate(task_ids)}
    arm_index = {arm: index for index, arm in enumerate(ARMS)}
    rewards = np.full((retained_tasks, len(ARMS), ATTEMPTS), np.nan)
    costs = np.full_like(rewards, np.nan)
    observed: set[tuple[str, str, int]] = set()
    for row in _read_rows(outcomes_path):
        task_id = row.get("task_id")
        arm = row.get("arm")
        attempt = row.get("attempt_number")
        identity = (str(task_id), str(arm), cast(int, attempt))
        reward = row.get("reward")
        cost = row.get("cost_usd")
        if (
            not isinstance(task_id, str)
            or task_id not in task_index
            or not isinstance(arm, str)
            or arm not in arm_index
            or not isinstance(attempt, int)
            or isinstance(attempt, bool)
            or not 0 <= attempt < ATTEMPTS
            or identity in observed
            or row.get("model") != "gpt-5.6-luna"
            or row.get("target_outcomes_used") is not False
            or isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or float(reward) not in {0.0, 1.0}
            or isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(float(cost))
            or float(cost) < 0.0
        ):
            raise ValueError(f"invalid development outcome {identity!r}")
        index = (task_index[task_id], arm_index[arm], attempt)
        rewards[index] = float(reward)
        costs[index] = float(cost)
        observed.add(identity)
    if len(observed) != retained_tasks * len(ARMS) * ATTEMPTS:
        raise ValueError("development outcome matrix is incomplete")
    if not np.isfinite(rewards).all() or not np.isfinite(costs).all():
        raise ValueError("development outcome matrix is not finite and dense")
    repositories = [str(task["repository"]) for task in tasks]
    data = Data(
        task_ids=task_ids,
        groups=repositories,
        texts=[_task_text(task) for task in tasks],
        structural=np.asarray([_structural(task) for task in tasks], dtype=np.float64),
        rewards=rewards.mean(axis=2),
        costs=costs.mean(axis=2),
    )
    return SourceData(
        data=data,
        raw_rewards=rewards,
        raw_costs=costs,
        languages=[str(task["language"]) for task in tasks],
        repositories=repositories,
    )


def load_route_tasks(path: Path) -> list[dict[str, Any]]:
    """Load the frozen label-free confirmation prompts."""
    if _sha256(path) != CONFIRMATION_CORPUS_SHA256:
        raise ValueError("confirmation corpus hash changed")
    corpus = _read_object(path)
    raw = corpus.get("tasks")
    if not isinstance(raw, list) or len(raw) != TASKS:
        raise ValueError("confirmation corpus must contain exactly 200 tasks")
    tasks = [
        {str(key): item for key, item in task.items()}
        for task in raw
        if isinstance(task, dict)
    ]
    if len(tasks) != TASKS or len({str(task["task_id"]) for task in tasks}) != TASKS:
        raise ValueError("confirmation tasks are invalid or duplicated")
    return tasks


def candidate_grid() -> tuple[Candidate, ...]:
    """Enumerate the complete preregistered development search."""
    values: list[Candidate] = []

    def add(family: Family, **fields: object) -> None:
        values.append(Candidate(family=family, order=len(values), **fields))

    for family in ("direct", "ordinal", "pairwise"):
        for dim in HASH_DIMS:
            for alpha in RIDGE_ALPHAS:
                for threshold in QUALITY_THRESHOLDS:
                    add(family, dim=dim, alpha=alpha, threshold=threshold)
    for dim in HASH_DIMS:
        for alpha in RIDGE_ALPHAS:
            for weight in IRT_WEIGHTS:
                add("irt", dim=dim, alpha=alpha, irt_weight=weight)
    for dim in HASH_DIMS:
        for guard in ARMS:
            for rag_num in KNN_COUNTS:
                for z in KNN_Z:
                    for pick_lam in PICK_LAMS:
                        add(
                            "knn",
                            dim=dim,
                            guard=guard,
                            rag_num=rag_num,
                            z=z,
                            pick_lam=pick_lam,
                        )
    result = tuple(values)
    if len(result) != 1_389 or len({candidate.key for candidate in result}) != len(result):
        raise AssertionError("candidate grid is incomplete or has duplicate identities")
    return result


def _cheapest_near_best(
    predictions: np.ndarray,
    arm_costs: np.ndarray,
    threshold: float,
) -> np.ndarray:
    if predictions.ndim != 2 or predictions.shape[1] != len(ARMS):
        raise ValueError("predictions have the wrong effort shape")
    if arm_costs.shape != (len(ARMS),) or not np.isfinite(arm_costs).all():
        raise ValueError("arm costs have the wrong shape")
    order = np.argsort(arm_costs, kind="stable")
    best = np.max(predictions, axis=1)
    choices = np.empty(len(predictions), dtype=np.int64)
    for row in range(len(predictions)):
        feasible = [arm for arm in order if predictions[row, arm] >= best[row] - threshold]
        choices[row] = int(feasible[0])
    return choices


def _ordinal_predictions(
    train_features: sparse.csr_matrix,
    test_features: sparse.csr_matrix,
    rewards: np.ndarray,
    *,
    alpha: float,
    monotone: bool,
    shrink: np.ndarray | None = None,
) -> np.ndarray:
    targets = np.column_stack([rewards[:, 0], np.diff(rewards, axis=1)])
    parts: list[np.ndarray] = []
    for column in range(targets.shape[1]):
        model = Ridge(alpha=alpha)
        model.fit(train_features, targets[:, column])
        prediction = np.asarray(model.predict(test_features), dtype=np.float64)
        if shrink is not None and column > 0:
            prediction *= float(shrink[column - 1])
        parts.append(prediction)
    predicted = np.column_stack(
        [parts[0], parts[0][:, None] + np.cumsum(np.column_stack(parts[1:]), axis=1)]
    )
    predicted = np.clip(predicted, 0.0, 1.0)
    return np.maximum.accumulate(predicted, axis=1) if monotone else predicted


def _pairwise_shrink(raw_rewards: np.ndarray) -> np.ndarray:
    paired = np.diff(raw_rewards, axis=1)
    task_means = paired.mean(axis=2)
    observed_variance = np.var(task_means, axis=0)
    noise_variance = np.mean(np.var(paired, axis=2, ddof=1), axis=0) / ATTEMPTS
    signal = np.maximum(observed_variance - noise_variance, 0.0)
    denominator = np.maximum(observed_variance, 1e-12)
    return np.clip(signal / denominator, 0.0, 1.0)


def _irt_probabilities(
    data: Data,
    train: np.ndarray,
    test: np.ndarray,
    *,
    dim: int,
    alpha: float,
    rewards: np.ndarray,
) -> np.ndarray:
    latent = _fit_irt(rewards[train])
    features = _features(data, dim, train)
    predictor = Ridge(alpha=alpha, solver="lsqr", max_iter=500, tol=1e-5)
    predictor.fit(
        features[train],
        np.column_stack([latent.difficulties, latent.log_discriminations]),
    )
    predicted = np.asarray(predictor.predict(features[test]), dtype=np.float64)
    difficulty = np.clip(predicted[:, 0], -8.0, 8.0)
    discrimination = np.exp(np.clip(predicted[:, 1], -3.0, 3.0))
    logits = discrimination[:, None] * (latent.abilities[None, :] - difficulty[:, None])
    return expit(logits)


def _irt_choices(probabilities: np.ndarray, costs: np.ndarray, weight: float) -> np.ndarray:
    low = float(np.min(costs))
    span = float(np.max(costs)) - low
    normalized = np.zeros_like(costs) if span <= 0.0 else (costs - low) / span
    objective = weight * probabilities - (1.0 - weight) * normalized[None, :]
    return np.argmax(objective, axis=1).astype(np.int64)


def _pool() -> list[PoolEntry]:
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
        for arm, effort in zip(ARMS, EFFORTS, strict=True)
    ]


def _matrix(data: Data, rewards: np.ndarray | None = None) -> OutcomeMatrix:
    values = data.rewards if rewards is None else rewards
    outcomes = [
        ScenarioOutcome(
            scenario_id=task_id,
            task=data.texts[task_index],
            model=arm,
            benchmark="swerebench-v2-external-development",
            episode=0,
            attempt_number=1,
            reward=float(values[task_index, arm_index]),
            success=bool(values[task_index, arm_index] >= 1.0),
            cost_usd=float(data.costs[task_index, arm_index]),
            completion_status="scored",
            usage_accounting="trace-derived",
        )
        for task_index, task_id in enumerate(data.task_ids)
        for arm_index, arm in enumerate(ARMS)
    ]
    return OutcomeMatrix(pool=_pool(), outcomes=outcomes)


def _tune_knn(base: RoutingPolicy, candidate: Candidate) -> RoutingPolicy:
    return base.model_copy(
        update={
            "default_model": candidate.guard,
            "guard_model": candidate.guard,
            "rag_num": candidate.rag_num,
            "rag_thres": KNN_RELATIVE_THRESHOLD,
            "floor_q": 0.0,
            "floor_sim": None,
            "knn_z": candidate.z,
            "knn_min_pairs": 8,
            "guard_mode": "asymmetric",
            "pick_lam": candidate.pick_lam,
        }
    )


def _best_static_arm(data: Data, train: np.ndarray) -> int:
    reward = data.rewards[train].mean(axis=0)
    best = float(np.max(reward))
    candidates = np.flatnonzero(np.isclose(reward, best, atol=1e-12))
    costs = data.costs[train].mean(axis=0)
    return int(candidates[int(np.argmin(costs[candidates]))])


def _folds(data: Data) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(len(data.task_ids), dtype=np.int64)
    folds = list(GroupKFold(n_splits=5).split(indices, groups=np.asarray(data.groups)))
    for train, test in folds:
        if set(np.asarray(data.groups)[train]) & set(np.asarray(data.groups)[test]):
            raise AssertionError("repository crossed a development fold")
    return folds


def _candidate_value(
    data: Data,
    candidate: Candidate,
    choices: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> CandidateResult:
    rows = np.arange(len(data.task_ids))
    routed_reward = data.rewards[rows, choices]
    routed_cost = data.costs[rows, choices]
    counts = np.bincount(choices, minlength=len(ARMS))
    traffic = counts / len(choices)
    blind_reward = data.rewards @ traffic
    blind_cost = data.costs @ traffic
    fold_rows: list[dict[str, object]] = []
    for fold, (train, test) in enumerate(folds):
        static_arm = _best_static_arm(data, train)
        route_reward = float(np.mean(routed_reward[test]))
        static_reward = float(np.mean(data.rewards[test, static_arm]))
        fold_rows.append(
            {
                "fold": fold,
                "train_tasks": len(train),
                "test_tasks": len(test),
                "train_repositories": len(set(np.asarray(data.groups)[train])),
                "test_repositories": len(set(np.asarray(data.groups)[test])),
                "repository_overlap": 0,
                "route_reward": route_reward,
                "fit_selected_static_arm": ARMS[static_arm],
                "fit_selected_static_reward": static_reward,
                "retention": route_reward / static_reward if static_reward > 0.0 else 1.0,
            }
        )
    reward = float(np.mean(routed_reward))
    cost = float(np.mean(routed_cost))
    static = [
        (arm, float(data.rewards[:, index].mean()), float(data.costs[:, index].mean()))
        for index, arm in enumerate(ARMS)
    ]
    dominated = [
        arm
        for arm, static_reward, static_cost in static
        if static_reward >= reward
        and static_cost <= cost
        and (static_reward > reward or static_cost < cost)
    ]
    return CandidateResult(
        candidate=candidate,
        choices=choices,
        reward=reward,
        cost_usd=cost,
        matched_blind_reward=float(np.mean(blind_reward)),
        matched_blind_cost_usd=float(np.mean(blind_cost)),
        advantage=float(np.mean(routed_reward - blind_reward)),
        arm_counts={arm: int(counts[index]) for index, arm in enumerate(ARMS)},
        fold_rows=fold_rows,
        retention_passed=all(
            float(row["retention"]) >= QUALITY_RETENTION for row in fold_rows
        ),
        dominated_by_static=dominated,
    )


def evaluate_grid(source: SourceData, work_dir: Path) -> list[CandidateResult]:
    """Run grouped OOF evaluation for every frozen candidate."""
    data = source.data
    folds = _folds(data)
    grid = candidate_grid()
    by_key = {candidate.key: candidate for candidate in grid}
    choices = {
        candidate.key: np.full(len(data.task_ids), -1, dtype=np.int64)
        for candidate in grid
    }
    non_knn = [candidate for candidate in grid if candidate.family != "knn"]
    knn = [candidate for candidate in grid if candidate.family == "knn"]
    matrix = _matrix(data)
    for fold, (train, test) in enumerate(folds):
        costs = data.costs[train].mean(axis=0)
        for dim in HASH_DIMS:
            features = _features(data, dim, train)
            train_features = features[train]
            test_features = features[test]
            for alpha in RIDGE_ALPHAS:
                direct_models = _fit_delta_models(
                    features,
                    data.rewards - data.rewards[:, [HIGH_INDEX]],
                    train,
                    alpha=alpha,
                )
                direct = _score_delta_models(features, test, direct_models)
                ordinal = _ordinal_predictions(
                    train_features,
                    test_features,
                    data.rewards[train],
                    alpha=alpha,
                    monotone=True,
                )
                pairwise = _ordinal_predictions(
                    train_features,
                    test_features,
                    data.rewards[train],
                    alpha=alpha,
                    monotone=False,
                    shrink=_pairwise_shrink(source.raw_rewards[train]),
                )
                irt = _irt_probabilities(
                    data,
                    train,
                    test,
                    dim=dim,
                    alpha=alpha,
                    rewards=data.rewards,
                )
                for candidate in non_knn:
                    if candidate.dim != dim or candidate.alpha != alpha:
                        continue
                    if candidate.family == "irt":
                        value = _irt_choices(irt, costs, candidate.irt_weight)
                    else:
                        predictions = {
                            "direct": direct,
                            "ordinal": ordinal,
                            "pairwise": pairwise,
                        }[candidate.family]
                        value = _cheapest_near_best(
                            predictions,
                            costs,
                            candidate.threshold,
                        )
                    choices[candidate.key][test] = value
            spec = EmbedderSpec(kind="hashing", dim=dim)
            embedder = spec.build()
            base = fit_knn_policy(
                matrix,
                bank_path=work_dir / f"fold-{fold}-hash{dim}.bank.npz",
                fit_ids=[data.task_ids[index] for index in train],
                embedder=spec,
                embed_with=embedder,
                guard_model=ARMS[0],
                rag_num=64,
                rag_thres=KNN_RELATIVE_THRESHOLD,
                z=0.0,
                min_pairs=8,
                se_floor=True,
                floor_q=0.0,
                pick_lam=0.0,
                fitted_from=f"SWE-rebench external development fold {fold}",
            )
            vectors = np.asarray(
                embedder.embed([data.texts[index] for index in test]),
                dtype=np.float64,
            )
            for candidate in knn:
                if candidate.dim != dim:
                    continue
                policy = _tune_knn(base, candidate)
                choices[candidate.key][test] = np.asarray(
                    [ARMS.index(knn_decision(policy, vector).model) for vector in vectors],
                    dtype=np.int64,
                )
        logger.info("completed grouped development fold %d/%d", fold + 1, len(folds))
    results: list[CandidateResult] = []
    for key, value in choices.items():
        if np.any(value < 0):
            raise RuntimeError(f"candidate {key} has incomplete OOF choices")
        results.append(_candidate_value(data, by_key[key], value, folds))
    return results


def selection_finalists(results: list[CandidateResult]) -> list[CandidateResult]:
    """Return points tied through cost and quality before the latency tie break."""
    eligible = [result for result in results if result.eligible]
    if not eligible:
        return []
    strongest = max(result.reward for result in eligible)
    near = [
        result for result in eligible if result.reward >= strongest - QUALITY_TOLERANCE
    ]
    minimum_cost = min(result.cost_usd for result in near)
    cheapest = [
        result
        for result in near
        if math.isclose(result.cost_usd, minimum_cost, rel_tol=0.0, abs_tol=1e-12)
    ]
    highest_reward = max(result.reward for result in cheapest)
    return [
        result
        for result in cheapest
        if math.isclose(result.reward, highest_reward, rel_tol=0.0, abs_tol=1e-12)
    ]


def select_candidate(
    results: list[CandidateResult],
    *,
    latency_by_key: dict[str, float] | None = None,
) -> CandidateResult | None:
    """Apply gates and the latency then family-order tie breaks."""
    finalists = selection_finalists(results)
    if not finalists:
        return None
    latency = latency_by_key or {}
    return min(
        finalists,
        key=lambda result: (
            latency.get(result.candidate.key, math.inf),
            result.candidate.order,
            result.candidate.key,
        ),
    )


def _result_row(result: CandidateResult) -> dict[str, object]:
    return {
        "key": result.candidate.key,
        "candidate": result.candidate.config(),
        "reward": result.reward,
        "cost_usd_per_task": result.cost_usd,
        "matched_blind_reward": result.matched_blind_reward,
        "matched_blind_cost_usd_per_task": result.matched_blind_cost_usd,
        "matched_blind_advantage": result.advantage,
        "arm_counts": result.arm_counts,
        "folds": result.fold_rows,
        "retention_passed": result.retention_passed,
        "dominated_by_static": result.dominated_by_static,
        "eligible": result.eligible,
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _route_feature(
    task: dict[str, Any],
    vectorizer: HashingVectorizer,
    structural_scale: np.ndarray,
) -> sparse.csr_matrix:
    text = cast(sparse.csr_matrix, vectorizer.transform([_task_text(task)]))
    structural = np.asarray([_structural(task)], dtype=np.float64) / structural_scale
    return sparse.hstack([text, sparse.csr_matrix(structural)], format="csr")


def _fit_text_router(
    source: SourceData,
    candidate: Candidate,
    *,
    label_rewards: np.ndarray,
) -> Callable[[dict[str, Any]], int]:
    data = source.data
    train = np.arange(len(data.task_ids), dtype=np.int64)
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        n_features=candidate.dim,
        alternate_sign=True,
        norm="l2",
    )
    text = cast(sparse.csr_matrix, vectorizer.transform(data.texts))
    structural_scale = np.maximum(np.std(data.structural, axis=0), 1.0)
    structural = data.structural / structural_scale
    features = sparse.hstack([text, sparse.csr_matrix(structural)], format="csr")
    costs = data.costs.mean(axis=0)
    if candidate.family == "direct":
        models = _fit_delta_models(
            features,
            label_rewards - label_rewards[:, [HIGH_INDEX]],
            train,
            alpha=candidate.alpha,
        )

        def route_direct(task: dict[str, Any]) -> int:
            row = _route_feature(task, vectorizer, structural_scale)
            predicted = np.zeros((1, len(ARMS)), dtype=np.float64)
            for arm_index, model in enumerate(models):
                if model is not None:
                    predicted[0, arm_index] = float(model.predict(row)[0])
            return int(
                _cheapest_near_best(predicted, costs, candidate.threshold)[0]
            )

        return route_direct

    if candidate.family in {"ordinal", "pairwise"}:
        targets = np.column_stack([label_rewards[:, 0], np.diff(label_rewards, axis=1)])
        models = []
        for column in range(targets.shape[1]):
            model = Ridge(alpha=candidate.alpha)
            model.fit(features, targets[:, column])
            models.append(model)
        shrink = (
            _pairwise_shrink(source.raw_rewards)
            if candidate.family == "pairwise"
            else np.ones(len(ARMS) - 1)
        )

        def route_ordinal(task: dict[str, Any]) -> int:
            row = _route_feature(task, vectorizer, structural_scale)
            parts = np.asarray(
                [float(model.predict(row)[0]) for model in models],
                dtype=np.float64,
            )
            parts[1:] *= shrink
            predicted = np.concatenate(
                [parts[:1], parts[:1] + np.cumsum(parts[1:])]
            )[None, :]
            predicted = np.clip(predicted, 0.0, 1.0)
            if candidate.family == "ordinal":
                predicted = np.maximum.accumulate(predicted, axis=1)
            return int(
                _cheapest_near_best(predicted, costs, candidate.threshold)[0]
            )

        return route_ordinal

    if candidate.family != "irt":
        raise ValueError("text router received a non-text candidate")
    latent = _fit_irt(label_rewards)
    predictor = Ridge(alpha=candidate.alpha, solver="lsqr", max_iter=500, tol=1e-5)
    predictor.fit(
        features,
        np.column_stack([latent.difficulties, latent.log_discriminations]),
    )

    def route_irt(task: dict[str, Any]) -> int:
        row = _route_feature(task, vectorizer, structural_scale)
        predicted = np.asarray(predictor.predict(row), dtype=np.float64)[0]
        difficulty = float(np.clip(predicted[0], -8.0, 8.0))
        discrimination = float(np.exp(np.clip(predicted[1], -3.0, 3.0)))
        probability = expit(discrimination * (latent.abilities - difficulty))[None, :]
        return int(_irt_choices(probability, costs, candidate.irt_weight)[0])

    return route_irt


def _fit_knn_router(
    source: SourceData,
    candidate: Candidate,
    bank_path: Path,
    *,
    label_rewards: np.ndarray,
) -> Callable[[dict[str, Any]], int]:
    data = source.data
    matrix = _matrix(data, label_rewards)
    spec = EmbedderSpec(kind="hashing", dim=candidate.dim)
    embedder = spec.build()
    base = fit_knn_policy(
        matrix,
        bank_path=bank_path,
        fit_ids=data.task_ids,
        embedder=spec,
        embed_with=embedder,
        guard_model=candidate.guard,
        rag_num=candidate.rag_num,
        rag_thres=KNN_RELATIVE_THRESHOLD,
        z=candidate.z,
        min_pairs=8,
        se_floor=True,
        floor_q=0.0,
        pick_lam=candidate.pick_lam,
        fitted_from="SWE-rebench external development only",
    )
    policy = _tune_knn(base, candidate)

    def route_knn(task: dict[str, Any]) -> int:
        vector = np.asarray(embedder.embed([_task_text(task)])[0], dtype=np.float64)
        return ARMS.index(knn_decision(policy, vector).model)

    return route_knn


def _fit_router(
    source: SourceData,
    candidate: Candidate,
    bank_path: Path,
    *,
    label_rewards: np.ndarray,
) -> Callable[[dict[str, Any]], int]:
    if candidate.family == "knn":
        return _fit_knn_router(
            source,
            candidate,
            bank_path,
            label_rewards=label_rewards,
        )
    return _fit_text_router(source, candidate, label_rewards=label_rewards)


def _permuted_labels(source: SourceData) -> np.ndarray:
    labels = source.data.rewards.copy()
    rng = np.random.default_rng(SHUFFLE_SEED)
    groups = np.asarray(source.data.groups, dtype=object)
    for group in sorted(set(source.data.groups)):
        members = np.flatnonzero(groups == group)
        labels[members] = source.data.rewards[rng.permutation(members)]
    return labels


def _route_rows(
    tasks: list[dict[str, Any]],
    route: Callable[[dict[str, Any]], int],
    candidate: Candidate,
    *,
    provenance: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task in tasks:
        choice = route(task)
        if not 0 <= choice < len(ARMS):
            raise ValueError("fitted router selected an unknown effort")
        rows.append(
            {
                "task_id": str(task["task_id"]),
                "repository": str(task["repository"]),
                "prompt_sha256": str(task["prompt_sha256"]),
                "arm": ARMS[choice],
                "model": "gpt-5.6-luna",
                "reasoning_effort": EFFORTS[choice],
                "candidate_key": candidate.key,
                "provenance": provenance,
                "target_outcomes_used": False,
                "deep_swe_outcomes_accessed": False,
            }
        )
    return rows


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _latency(
    route: Callable[[dict[str, Any]], int],
    tasks: list[dict[str, Any]],
) -> dict[str, object]:
    warmup = min(50, len(tasks))
    for index in range(warmup):
        route(tasks[index])
    durations = np.empty(1_000, dtype=np.float64)
    for index in range(len(durations)):
        start = time.perf_counter_ns()
        route(tasks[index % len(tasks)])
        durations[index] = (time.perf_counter_ns() - start) / 1_000_000
    p50, p95 = np.quantile(durations, [0.5, 0.95])
    return {
        "decisions": len(durations),
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "passed": float(p95) < 5.0,
    }


def fit(
    corpus: Path,
    outcomes: Path,
    audit: Path,
    confirmation_corpus: Path,
    output: Path,
) -> None:
    """Evaluate the frozen grid and write a fit-only selection report."""
    if output.exists():
        raise FileExistsError(f"fit output already exists: {output}")
    source = load_source(corpus, outcomes, audit)
    confirmation = load_route_tasks(confirmation_corpus)
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="swerebench-fit-") as temporary:
        results = evaluate_grid(source, Path(temporary))
    finalists = selection_finalists(results)
    tie_latency: dict[str, float] = {}
    if finalists:
        with tempfile.TemporaryDirectory(prefix="swerebench-tie-latency-") as temporary:
            temporary_root = Path(temporary)
            for finalist in finalists:
                route = _fit_router(
                    source,
                    finalist.candidate,
                    temporary_root / f"{finalist.candidate.order}.bank.npz",
                    label_rewards=source.data.rewards,
                )
                measured = _latency(route, confirmation)
                tie_latency[finalist.candidate.key] = float(measured["p95_ms"])
    selected = select_candidate(results, latency_by_key=tie_latency)
    static = [
        {
            "arm": arm,
            "reward": float(source.data.rewards[:, index].mean()),
            "cost_usd_per_task": float(source.data.costs[:, index].mean()),
        }
        for index, arm in enumerate(ARMS)
    ]
    report: dict[str, object] = {
        "protocol": PROTOCOL,
        "source_tasks": TASKS,
        "tasks": len(source.data.task_ids),
        "retained_task_coverage": len(source.data.task_ids) / TASKS,
        "excluded_tasks": TASKS - len(source.data.task_ids),
        "repositories": len(set(source.data.groups)),
        "candidate_count": len(results),
        "selection_rule": (
            "among eligible points within 0.005 reward of strongest, choose least cost, "
            "then higher reward and simpler preregistered family order"
        ),
        "static_efforts": static,
        "selected": _result_row(selected) if selected is not None else None,
        "selection_tie_latency_p95_ms": tie_latency,
        "development_passed": False,
        "confirmation_tasks_loaded_label_free": len(confirmation),
        "confirmation_routes_written": False,
        "fitted_numeric_state_persisted": False,
        "target_outcomes_used": False,
        "deep_swe_outcomes_accessed": False,
        "inputs": {
            "development_corpus_sha256": _sha256(corpus),
            "outcomes_sha256": _sha256(outcomes),
            "collection_audit_sha256": _sha256(audit),
            "confirmation_corpus_sha256": _sha256(confirmation_corpus),
        },
        "candidates": [_result_row(result) for result in results],
    }
    if selected is None:
        _write_json(output / "development-report.json", report)
        logger.info("no external development candidate passed every frozen gate")
        return
    canonical_config = json.dumps(
        selected.candidate.config(),
        sort_keys=True,
        separators=(",", ":"),
    )
    lock = {
        "protocol": PROTOCOL,
        "selected_key": selected.candidate.key,
        "selected_config": selected.candidate.config(),
        "selected_config_sha256": hashlib.sha256(canonical_config.encode()).hexdigest(),
        "development_corpus_sha256": _sha256(corpus),
        "development_outcomes_sha256": _sha256(outcomes),
        "collection_audit_sha256": _sha256(audit),
        "confirmation_corpus_sha256": _sha256(confirmation_corpus),
        "target_outcomes_used": False,
        "deep_swe_outcomes_accessed": False,
        "fitted_numeric_state_persisted": False,
        "confirmation_outcomes_accessed": False,
        "confirmation_routes_written": False,
    }
    _write_json(output / "selection-lock.json", lock)
    with tempfile.TemporaryDirectory(prefix="swerebench-route-") as temporary:
        temporary_root = Path(temporary)
        route = _fit_router(
            source,
            selected.candidate,
            temporary_root / "selected.bank.npz",
            label_rewards=source.data.rewards,
        )
        latency = _latency(route, confirmation)
        route_rows = _route_rows(
            confirmation,
            route,
            selected.candidate,
            provenance="external-development-fit-only",
        )
        shuffled_route = _fit_router(
            source,
            selected.candidate,
            temporary_root / "shuffled.bank.npz",
            label_rewards=_permuted_labels(source),
        )
        shuffled_rows = _route_rows(
            confirmation,
            shuffled_route,
            selected.candidate,
            provenance="within-repository-permuted-development-control",
        )
    route_path = output / "confirmation-routes.jsonl"
    shuffled_path = output / "confirmation-shuffled-routes.jsonl"
    _write_rows(route_path, route_rows)
    _write_rows(shuffled_path, shuffled_rows)
    route_audit = {
        "protocol": "coding-router-swerebench-sealed-confirmation-routes-v1",
        "tasks": len(route_rows),
        "unique_task_ids": len({str(row["task_id"]) for row in route_rows}),
        "selected_config_sha256": lock["selected_config_sha256"],
        "selection_lock_sha256": _sha256(output / "selection-lock.json"),
        "confirmation_routes_sha256": _sha256(route_path),
        "shuffled_routes_sha256": _sha256(shuffled_path),
        "latency": latency,
        "fitted_numeric_state_persisted": False,
        "target_outcomes_used": False,
        "deep_swe_outcomes_accessed": False,
        "confirmation_outcomes_accessed": False,
    }
    _write_json(output / "route-audit.json", route_audit)
    report["development_passed"] = bool(latency["passed"])
    report["confirmation_routes_written"] = True
    report["confirmation_authorized"] = bool(latency["passed"])
    report["route_audit_sha256"] = _sha256(output / "route-audit.json")
    report["latency"] = latency
    _write_json(output / "development-report.json", report)
    logger.info(
        "selected external candidate=%s reward=%.4f cost=%.6f advantage=%.6f p95_ms=%.4f",
        selected.candidate.key,
        selected.reward,
        selected.cost_usd,
        selected.advantage,
        float(latency["p95_ms"]),
    )


def main() -> None:
    """Parse the E2B-only fit command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--confirmation-corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fit(
        args.corpus,
        args.outcomes,
        args.audit,
        args.confirmation_corpus,
        args.output,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
