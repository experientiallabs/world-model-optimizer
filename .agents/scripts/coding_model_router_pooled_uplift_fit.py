"""Fit the frozen pooled cross-attempt reasoning-effort uplift router on E2B."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from coding_model_router_swerebench_fit import ARMS
from coding_model_router_swesmith_null_fit import (
    ExternalTask,
    FittedScorer,
    ScorerCandidate,
    _structural,
)
from coding_model_router_swesmith_null_fit import (
    _design as _prior_design,
)
from coding_model_router_swesmith_null_fit import (
    _fit_scorer as _fit_prior,
)
from scipy import sparse
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

logger = logging.getLogger("coding-router-pooled-uplift-fit")

PROTOCOL = "coding-router-pooled-cross-attempt-uplift-v1"
TASK_HASHES = (
    "7d846b5576d15e68fd18ac21bfe0610cc1614b3b35ec0ae0cb8cfae0b82962c1",
    "9798dd1e58be0d13331d097307670dc3fc3760ad211da20e6367666523f080a7",
)
OUTCOME_HASHES = (
    "5c2097116b03291f20bc33d6a376cb01d9a2e9fb182f46c482df5508b7140ee2",
    "0c03bcbd935c0983c9e6355413222fb1545206d6ae5a91329b505f77f35300d6",
)
AUDIT_HASHES = (
    "ca20ebdc85bda0482e9c726a95c4216bc1b4acec63fe86db88ef4fc4431ab316",
    "cced3491bfb5e4cb5eeaebde6473ad399c3ed4ebf8cbee916fb598625a0f4744",
)
EXTERNAL_TASKS_SHA256 = "9a4b3b749fb2123933335f9c4db41057247f49b37c53a7c075143b44e800aa7c"
EXTERNAL_MANIFEST_SHA256 = "ca253f77f7991d4b3ade6d299c6b41c42b7291b077b5d319167d95a73d526939"
POOLED_EXCLUSIONS_SHA256 = "4f05067a44fc69f1e53bb26f79da615a51119740fe9753a3e690434aa9e2b01f"
ATTEMPTS = 2
HIGH_INDEX = 2
MAX_INDEX = 4
NULL_COUNT = 128
SEED = 20_260_801
BOOTSTRAPS = 10_000
QUALITY_RETENTION = 0.95
MAX_ROUTE_P95_MS = 5.0
THRESHOLD_PERCENTILES = tuple(range(10, 91, 5))

Family = Literal["direct_ridge", "two_head_ridge", "hist", "extra_trees"]


@dataclass(frozen=True)
class PooledData:
    """Dense pooled task, outcome, cost, and pre-inference metadata."""

    task_ids: list[str]
    repositories: list[str]
    languages: list[str]
    texts: list[str]
    rewards: np.ndarray
    costs: np.ndarray


@dataclass(frozen=True)
class Candidate:
    """One preregistered uplift learner."""

    order: int
    family: Family
    dim: int = 0
    alpha: float = 0.0
    leaves: int = 0
    min_leaf: int = 0

    @property
    def key(self) -> str:
        """Return the frozen candidate identity."""
        if self.family in {"direct_ridge", "two_head_ridge"}:
            return f"{self.family}-hash{self.dim}-a{self.alpha:g}"
        if self.family == "hist":
            return f"hist-leaves{self.leaves}-l2-{self.alpha:g}"
        return f"extra-trees-256-leaf{self.min_leaf}"


@dataclass(frozen=True)
class FeatureBank:
    """Precomputed deterministic feature matrices for one task cohort."""

    texts: list[str]
    dense: np.ndarray
    hashed: dict[int, sparse.csr_matrix]


@dataclass(frozen=True)
class FittedUplift:
    """One ephemeral fitted uplift estimator."""

    candidate: Candidate
    estimator: Any


@dataclass(frozen=True)
class RouteMetrics:
    """Aggregate route value and its traffic-matched comparator."""

    reward: float
    cost_usd: float
    blind_reward: float
    blind_cost_usd: float
    advantage: float
    retention: float
    dominated_by_static: tuple[str, ...]
    arm_counts: dict[str, int]
    task_advantages: np.ndarray
    task_repositories: list[str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} contains a non-object row")
        rows.append({str(key): item for key, item in value.items()})
    return rows


def _task_text(repository: str, prompt: str) -> str:
    return f"repository={repository}\n{prompt}"


def _load_cohort(
    tasks_path: Path,
    outcomes_path: Path,
    audit_path: Path,
    *,
    expected_tasks_hash: str,
    expected_outcomes_hash: str,
    expected_audit_hash: str,
) -> PooledData:
    """Load one frozen dense cohort with its whole-task exclusions."""
    if _sha256(tasks_path) != expected_tasks_hash:
        raise ValueError("pooled task manifest hash changed")
    if _sha256(outcomes_path) != expected_outcomes_hash:
        raise ValueError("pooled outcome hash changed")
    if _sha256(audit_path) != expected_audit_hash:
        raise ValueError("pooled completion audit hash changed")
    audit = _read_object(audit_path)
    retained = audit.get("tasks")
    excluded = audit.get("excluded_tasks")
    if (
        audit.get("valid") is not True
        or audit.get("source_tasks") != 200
        or not isinstance(retained, int)
        or not 190 <= retained <= 200
        or not isinstance(excluded, list)
        or len(excluded) != 200 - retained
        or audit.get("cells") != retained * len(ARMS) * ATTEMPTS
        or audit.get("target_outcomes_used") is not False
        or audit.get("deep_swe_outcomes_accessed") is not False
        or audit.get("outcomes_sha256") != expected_outcomes_hash
    ):
        raise ValueError("pooled completion audit is incomplete or unsafe")
    raw_tasks = _read_object(tasks_path).get("tasks")
    if not isinstance(raw_tasks, list) or len(raw_tasks) != 200:
        raise ValueError("pooled task manifest must contain 200 tasks")
    excluded_ids = {
        str(row["task_id"])
        for row in excluded
        if isinstance(row, dict) and row.get("scope") == "whole-task"
    }
    if len(excluded_ids) != len(excluded):
        raise ValueError("pooled whole-task exclusion is invalid")
    tasks = [
        row
        for row in raw_tasks
        if isinstance(row, dict) and str(row.get("task_id")) not in excluded_ids
    ]
    if len(tasks) != retained:
        raise ValueError("pooled retained task count differs from its audit")
    task_ids = [str(row["task_id"]) for row in tasks]
    task_index = {task_id: index for index, task_id in enumerate(task_ids)}
    arm_index = {arm: index for index, arm in enumerate(ARMS)}
    rewards = np.full((retained, len(ARMS), ATTEMPTS), np.nan)
    costs = np.full_like(rewards, np.nan)
    observed: set[tuple[str, str, int]] = set()
    for row in _read_rows(outcomes_path):
        task_id = row.get("task_id")
        arm = row.get("arm")
        attempt = row.get("attempt_number")
        reward = row.get("reward")
        cost = row.get("cost_usd")
        identity = (str(task_id), str(arm), int(attempt) if isinstance(attempt, int) else -1)
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
            raise ValueError(f"invalid pooled outcome {identity!r}")
        index = (task_index[task_id], arm_index[arm], attempt)
        rewards[index] = float(reward)
        costs[index] = float(cost)
        observed.add(identity)
    if len(observed) != retained * len(ARMS) * ATTEMPTS:
        raise ValueError("pooled outcome matrix is incomplete")
    if not np.isfinite(rewards).all() or not np.isfinite(costs).all():
        raise ValueError("pooled outcome matrix is not dense and finite")
    return PooledData(
        task_ids=task_ids,
        repositories=[str(row["repository"]) for row in tasks],
        languages=[str(row["language"]) for row in tasks],
        texts=[_task_text(str(row["repository"]), str(row["prompt"])) for row in tasks],
        rewards=rewards,
        costs=costs,
    )


def _pool(cohorts: list[PooledData]) -> PooledData:
    """Join disjoint cohorts without changing task or attempt order."""
    task_ids = [task_id for cohort in cohorts for task_id in cohort.task_ids]
    repositories = [repo for cohort in cohorts for repo in cohort.repositories]
    repository_sets = [set(cohort.repositories) for cohort in cohorts]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("pooled task IDs overlap")
    if repository_sets[0] & repository_sets[1]:
        raise ValueError("pooled repositories overlap")
    return PooledData(
        task_ids=task_ids,
        repositories=repositories,
        languages=[value for cohort in cohorts for value in cohort.languages],
        texts=[value for cohort in cohorts for value in cohort.texts],
        rewards=np.concatenate([cohort.rewards for cohort in cohorts], axis=0),
        costs=np.concatenate([cohort.costs for cohort in cohorts], axis=0),
    )


def _load_external(tasks_path: Path, manifest_path: Path) -> list[ExternalTask]:
    """Load the pooled-exclusion SWE-smith prior source."""
    if _sha256(tasks_path) != EXTERNAL_TASKS_SHA256:
        raise ValueError("pooled external prior task hash changed")
    if _sha256(manifest_path) != EXTERNAL_MANIFEST_SHA256:
        raise ValueError("pooled external prior manifest hash changed")
    manifest = _read_object(manifest_path)
    if (
        manifest.get("valid") is not True
        or manifest.get("tasks_sha256") != EXTERNAL_TASKS_SHA256
        or (manifest.get("input_sha256") or {}).get("development_tasks_sha256")
        != POOLED_EXCLUSIONS_SHA256
        or manifest.get("target_reward_fields_accessed") is not False
        or manifest.get("target_cost_fields_accessed") is not False
        or manifest.get("later_trajectory_turns_used") is not False
        or manifest.get("patch_field_read") is not False
    ):
        raise ValueError("pooled external prior manifest is incomplete or unsafe")
    tasks: list[ExternalTask] = []
    for row in _read_rows(tasks_path):
        tasks.append(
            ExternalTask(
                task_id=str(row["task_id"]),
                repository=str(row["repository"]),
                prompt=str(row["prompt"]),
                target=float(row["difficulty_target"]),
            )
        )
    if len(tasks) != 1_551:
        raise ValueError("pooled external prior task count changed")
    return tasks


def _candidate_grid() -> tuple[Candidate, ...]:
    values: list[Candidate] = []
    for family in ("direct_ridge", "two_head_ridge"):
        for dim in (512, 2_048, 8_192):
            for alpha in (1.0, 10.0, 100.0):
                values.append(
                    Candidate(
                        order=len(values),
                        family=family,
                        dim=dim,
                        alpha=alpha,
                    )
                )
    for leaves in (7, 15):
        for alpha in (1.0, 10.0):
            values.append(
                Candidate(
                    order=len(values),
                    family="hist",
                    alpha=alpha,
                    leaves=leaves,
                )
            )
    for min_leaf in (5, 10, 20):
        values.append(
            Candidate(
                order=len(values),
                family="extra_trees",
                min_leaf=min_leaf,
            )
        )
    return tuple(values)


def _features(data: PooledData, prior: FittedScorer) -> FeatureBank:
    """Build frozen task-text, trace-prior, and interaction features."""
    structural = np.asarray([_structural(text) for text in data.texts], dtype=np.float64)
    prior_scores = np.asarray(
        prior.model.predict(_prior_design(data.texts, prior.vectorizer)),
        dtype=np.float64,
    )
    dense = np.column_stack(
        [structural, prior_scores, structural * prior_scores[:, None]]
    )
    hashed = {
        dim: HashingVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            n_features=dim,
            alternate_sign=True,
            norm="l2",
            lowercase=True,
        ).transform(data.texts)
        for dim in (512, 2_048, 8_192)
    }
    return FeatureBank(texts=data.texts, dense=dense, hashed=hashed)


def _matrix(bank: FeatureBank, candidate: Candidate) -> np.ndarray | sparse.csr_matrix:
    if candidate.family in {"direct_ridge", "two_head_ridge"}:
        return sparse.hstack(
            [bank.hashed[candidate.dim], sparse.csr_matrix(bank.dense)],
            format="csr",
        )
    return bank.dense


def _fit_model(
    candidate: Candidate,
    matrix: np.ndarray | sparse.csr_matrix,
    train: np.ndarray,
    high: np.ndarray,
    maximum: np.ndarray,
) -> FittedUplift:
    """Fit one candidate without persisting estimator state."""
    if candidate.family == "direct_ridge":
        estimator: Any = Ridge(alpha=candidate.alpha, solver="lsqr")
        estimator.fit(matrix[train], maximum[train] - high[train])
    elif candidate.family == "two_head_ridge":
        estimator = Ridge(alpha=candidate.alpha, solver="lsqr")
        estimator.fit(matrix[train], np.column_stack([high[train], maximum[train]]))
    elif candidate.family == "hist":
        estimator = HistGradientBoostingRegressor(
            max_leaf_nodes=candidate.leaves,
            l2_regularization=candidate.alpha,
            random_state=SEED,
        )
        estimator.fit(matrix[train], maximum[train] - high[train])
    else:
        estimator = ExtraTreesRegressor(
            n_estimators=256,
            min_samples_leaf=candidate.min_leaf,
            max_features="sqrt",
            random_state=SEED,
            n_jobs=1,
        )
        estimator.fit(matrix[train], maximum[train] - high[train])
    return FittedUplift(candidate=candidate, estimator=estimator)


def _predict(
    fitted: FittedUplift,
    matrix: np.ndarray | sparse.csr_matrix,
    indices: np.ndarray,
) -> np.ndarray:
    values = np.asarray(fitted.estimator.predict(matrix[indices]), dtype=np.float64)
    if fitted.candidate.family == "two_head_ridge":
        return values[:, 1] - values[:, 0]
    return values.reshape(-1)


def _route(scores: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(scores >= threshold, MAX_INDEX, HIGH_INDEX).astype(np.int64)


def _attempt_metrics(
    data: PooledData,
    choices: np.ndarray,
    indices: np.ndarray,
    attempt: int,
) -> RouteMetrics:
    rewards = data.rewards[indices, :, attempt]
    costs = data.costs[indices, :, attempt]
    selected_rewards = rewards[np.arange(len(indices)), choices]
    selected_costs = costs[np.arange(len(indices)), choices]
    counts = np.bincount(choices, minlength=len(ARMS))
    traffic = counts / len(indices)
    blind_rewards = rewards @ traffic
    blind_costs = costs @ traffic
    reward = float(np.mean(selected_rewards))
    cost = float(np.mean(selected_costs))
    static_rewards = rewards.mean(axis=0)
    static_costs = costs.mean(axis=0)
    strongest = float(np.max(static_rewards))
    dominated = tuple(
        ARMS[index]
        for index in range(len(ARMS))
        if static_rewards[index] >= reward - 1e-12 and static_costs[index] <= cost + 1e-12
    )
    return RouteMetrics(
        reward=reward,
        cost_usd=cost,
        blind_reward=float(np.mean(blind_rewards)),
        blind_cost_usd=float(np.mean(blind_costs)),
        advantage=reward - float(np.mean(blind_rewards)),
        retention=1.0 if strongest <= 0.0 else reward / strongest,
        dominated_by_static=dominated,
        arm_counts={ARMS[index]: int(value) for index, value in enumerate(counts) if value},
        task_advantages=selected_rewards - blind_rewards,
        task_repositories=[data.repositories[index] for index in indices],
    )


def _aggregate_metrics(
    data: PooledData,
    choices: np.ndarray,
    evaluations: list[tuple[np.ndarray, int]],
) -> RouteMetrics:
    """Aggregate held-out evaluations while matching blind traffic within each."""
    routed_rewards: list[np.ndarray] = []
    routed_costs: list[np.ndarray] = []
    blind_rewards: list[np.ndarray] = []
    blind_costs: list[np.ndarray] = []
    advantages: list[np.ndarray] = []
    repositories: list[str] = []
    counts = np.zeros(len(ARMS), dtype=np.int64)
    for indices, attempt in evaluations:
        selected = choices[indices, attempt]
        rewards = data.rewards[indices, :, attempt]
        costs = data.costs[indices, :, attempt]
        local_counts = np.bincount(selected, minlength=len(ARMS))
        traffic = local_counts / len(indices)
        route_reward = rewards[np.arange(len(indices)), selected]
        route_cost = costs[np.arange(len(indices)), selected]
        blind_reward = rewards @ traffic
        blind_cost = costs @ traffic
        routed_rewards.append(route_reward)
        routed_costs.append(route_cost)
        blind_rewards.append(blind_reward)
        blind_costs.append(blind_cost)
        advantages.append(route_reward - blind_reward)
        repositories.extend(data.repositories[index] for index in indices)
        counts += local_counts
    route_reward_values = np.concatenate(routed_rewards)
    route_cost_values = np.concatenate(routed_costs)
    blind_reward_values = np.concatenate(blind_rewards)
    blind_cost_values = np.concatenate(blind_costs)
    task_advantages = np.concatenate(advantages)
    static_rewards = np.concatenate(
        [data.rewards[indices, :, attempt] for indices, attempt in evaluations],
        axis=0,
    ).mean(axis=0)
    static_costs = np.concatenate(
        [data.costs[indices, :, attempt] for indices, attempt in evaluations],
        axis=0,
    ).mean(axis=0)
    reward = float(np.mean(route_reward_values))
    cost = float(np.mean(route_cost_values))
    strongest = float(np.max(static_rewards))
    dominated = tuple(
        ARMS[index]
        for index in range(len(ARMS))
        if static_rewards[index] >= reward - 1e-12 and static_costs[index] <= cost + 1e-12
    )
    return RouteMetrics(
        reward=reward,
        cost_usd=cost,
        blind_reward=float(np.mean(blind_reward_values)),
        blind_cost_usd=float(np.mean(blind_cost_values)),
        advantage=float(np.mean(task_advantages)),
        retention=1.0 if strongest <= 0.0 else reward / strongest,
        dominated_by_static=dominated,
        arm_counts={ARMS[index]: int(value) for index, value in enumerate(counts) if value},
        task_advantages=task_advantages,
        task_repositories=repositories,
    )


def _select_threshold(
    data: PooledData,
    predictions: np.ndarray,
    indices: np.ndarray,
    attempt: int,
) -> tuple[float | None, dict[str, object] | None]:
    """Choose the lowest-cost eligible training-repository operating point."""
    rows: list[tuple[tuple[float, float, float, int], float, dict[str, object]]] = []
    for order, percentile in enumerate(THRESHOLD_PERCENTILES):
        threshold = float(np.percentile(predictions, percentile))
        choices = _route(predictions, threshold)
        metrics = _attempt_metrics(data, choices, indices, attempt)
        row = {
            "percentile": percentile,
            "threshold": threshold,
            **_metrics_dict(metrics),
        }
        if (
            metrics.retention >= QUALITY_RETENTION
            and metrics.advantage > 0.0
            and not metrics.dominated_by_static
        ):
            rows.append(
                (
                    (
                        metrics.cost_usd,
                        -metrics.advantage,
                        -metrics.reward,
                        order,
                    ),
                    threshold,
                    row,
                )
            )
    if not rows:
        return None, None
    _, threshold, row = min(rows, key=lambda value: value[0])
    return threshold, row


def _block_permutations(
    data: PooledData,
    indices: np.ndarray,
    scores: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Permute complete equal-length repository score blocks within language."""
    blocks: dict[tuple[str, int], list[np.ndarray]] = {}
    local_repositories = np.asarray([data.repositories[index] for index in indices])
    local_languages = np.asarray([data.languages[index] for index in indices])
    for repository in sorted(set(local_repositories)):
        local = np.flatnonzero(local_repositories == repository)
        languages = set(local_languages[local])
        if len(languages) != 1:
            raise ValueError(f"held-out repository spans languages: {repository}")
        blocks.setdefault((str(local_languages[local[0]]), len(local)), []).append(local)
    rng = np.random.default_rng(seed)
    values = np.empty((NULL_COUNT, len(indices)), dtype=np.float64)
    for null_index in range(NULL_COUNT):
        shuffled = np.empty_like(scores)
        for key in sorted(blocks):
            recipients = blocks[key]
            order = rng.permutation(len(recipients))
            for recipient, donor_index in zip(recipients, order, strict=True):
                donor = recipients[int(donor_index)]
                shuffled[recipient] = scores[donor]
        values[null_index] = shuffled
    return values


def _bootstrap_lower(metrics: RouteMetrics) -> float:
    """Return the repository-bootstrap 95 percent lower advantage bound."""
    repositories = np.asarray(metrics.task_repositories)
    unique = np.asarray(sorted(set(metrics.task_repositories)))
    by_repository = {
        repository: np.flatnonzero(repositories == repository) for repository in unique
    }
    rng = np.random.default_rng(SEED)
    samples = np.empty(BOOTSTRAPS, dtype=np.float64)
    for index in range(BOOTSTRAPS):
        selected = rng.choice(unique, size=len(unique), replace=True)
        values = np.concatenate([by_repository[repository] for repository in selected])
        samples[index] = float(np.mean(metrics.task_advantages[values]))
    return float(np.quantile(samples, 0.025, method="lower"))


def _metrics_dict(metrics: RouteMetrics) -> dict[str, object]:
    return {
        "reward": metrics.reward,
        "cost_usd_per_task": metrics.cost_usd,
        "matched_blind_reward": metrics.blind_reward,
        "matched_blind_cost_usd_per_task": metrics.blind_cost_usd,
        "matched_blind_advantage": metrics.advantage,
        "quality_retention": metrics.retention,
        "dominated_by_static": list(metrics.dominated_by_static),
        "arm_counts": metrics.arm_counts,
    }


def _higher_quantile(values: np.ndarray) -> float:
    return float(np.quantile(values, 0.95, method="higher"))


def _evaluate_candidate(
    data: PooledData,
    bank: FeatureBank,
    candidate: Candidate,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, object], np.ndarray, list[np.ndarray], list[tuple[np.ndarray, int]]]:
    """Produce nested cross-attempt held-out routes and their frozen nulls."""
    matrix = _matrix(bank, candidate)
    choices = np.full((len(data.task_ids), ATTEMPTS), -1, dtype=np.int64)
    null_choices = [np.full_like(choices, -1) for _ in range(NULL_COUNT)]
    evaluations: list[tuple[np.ndarray, int]] = []
    evaluation_rows: list[dict[str, object]] = []
    for fold, (train, test) in enumerate(folds):
        if set(np.asarray(data.repositories)[train]) & set(np.asarray(data.repositories)[test]):
            raise AssertionError("pooled repository crossed an outer fold")
        for direction, (fit_attempt, score_attempt) in enumerate(((0, 1), (1, 0))):
            fitted = _fit_model(
                candidate,
                matrix,
                train,
                data.rewards[:, HIGH_INDEX, fit_attempt],
                data.rewards[:, MAX_INDEX, fit_attempt],
            )
            train_predictions = _predict(fitted, matrix, train)
            threshold, calibration = _select_threshold(
                data,
                train_predictions,
                train,
                score_attempt,
            )
            test_predictions = _predict(fitted, matrix, test)
            if threshold is None:
                selected = np.full(len(test), MAX_INDEX, dtype=np.int64)
                permutations = np.tile(selected, (NULL_COUNT, 1))
            else:
                selected = _route(test_predictions, threshold)
                null_scores = _block_permutations(
                    data,
                    test,
                    test_predictions,
                    seed=SEED + fold * 2 + direction,
                )
                permutations = np.asarray(
                    [_route(row, threshold) for row in null_scores],
                    dtype=np.int64,
                )
            choices[test, score_attempt] = selected
            for null_index in range(NULL_COUNT):
                null_choices[null_index][test, score_attempt] = permutations[null_index]
            metrics = _attempt_metrics(data, selected, test, score_attempt)
            evaluation_rows.append(
                {
                    "fold": fold,
                    "fit_attempt": fit_attempt,
                    "score_attempt": score_attempt,
                    "threshold_found": threshold is not None,
                    "threshold": threshold,
                    "calibration": calibration,
                    **_metrics_dict(metrics),
                }
            )
            evaluations.append((test, score_attempt))
    if np.any(choices < 0) or any(np.any(value < 0) for value in null_choices):
        raise AssertionError("nested pooled route matrix is incomplete")
    metrics = _aggregate_metrics(data, choices, evaluations)
    null_advantages = np.asarray(
        [_aggregate_metrics(data, value, evaluations).advantage for value in null_choices]
    )
    row = {
        "key": candidate.key,
        "order": candidate.order,
        "family": candidate.family,
        "dim": candidate.dim,
        "alpha": candidate.alpha,
        "leaves": candidate.leaves,
        "min_leaf": candidate.min_leaf,
        **_metrics_dict(metrics),
        "bootstrap_ci95_lower": _bootstrap_lower(metrics),
        "positive_evaluations": sum(
            float(value["matched_blind_advantage"]) > 0.0 for value in evaluation_rows
        ),
        "evaluations": evaluation_rows,
        "null_advantages": null_advantages.tolist(),
    }
    return row, choices, null_choices, evaluations


def _select_full_threshold(
    data: PooledData,
    predictions: np.ndarray,
) -> tuple[float | None, dict[str, object] | None, np.ndarray | None]:
    """Freeze one full-development threshold after nested candidate selection."""
    indices = np.arange(len(data.task_ids), dtype=np.int64)
    evaluations = [(indices, 0), (indices, 1)]
    values: list[
        tuple[tuple[float, float, float, int], float, dict[str, object], np.ndarray]
    ] = []
    for order, percentile in enumerate(THRESHOLD_PERCENTILES):
        threshold = float(np.percentile(predictions, percentile))
        task_choices = _route(predictions, threshold)
        choices = np.column_stack([task_choices, task_choices])
        metrics = _aggregate_metrics(data, choices, evaluations)
        row = {
            "percentile": percentile,
            "threshold": threshold,
            **_metrics_dict(metrics),
        }
        if (
            metrics.retention >= QUALITY_RETENTION
            and metrics.advantage > 0.0
            and not metrics.dominated_by_static
        ):
            values.append(
                (
                    (
                        metrics.cost_usd,
                        -metrics.advantage,
                        -metrics.reward,
                        order,
                    ),
                    threshold,
                    row,
                    task_choices,
                )
            )
    if not values:
        return None, None, None
    _, threshold, row, task_choices = min(values, key=lambda value: value[0])
    return threshold, row, task_choices


def _single_prediction(
    text: str,
    prior: FittedScorer,
    fitted: FittedUplift,
    vectorizer: HashingVectorizer | None,
) -> float:
    """Execute the complete shared-feature pre-inference scoring path once."""
    structural = np.asarray([_structural(text)], dtype=np.float64)
    if prior.vectorizer is None:
        raise AssertionError("the frozen prior must use character hashing")
    prior_hash = prior.vectorizer.transform([text])
    prior_matrix = sparse.hstack(
        [prior_hash, sparse.csr_matrix(structural)],
        format="csr",
    )
    prior_score = float(prior.model.predict(prior_matrix)[0])
    dense = np.column_stack(
        [structural, np.asarray([prior_score]), structural * prior_score]
    )
    candidate = fitted.candidate
    if candidate.family in {"direct_ridge", "two_head_ridge"}:
        if candidate.dim == 8_192:
            candidate_hash = prior_hash
        else:
            if vectorizer is None:
                raise AssertionError("a hashed uplift candidate needs its vectorizer")
            candidate_hash = vectorizer.transform([text])
        matrix: np.ndarray | sparse.csr_matrix = sparse.hstack(
            [candidate_hash, sparse.csr_matrix(dense)],
            format="csr",
        )
    else:
        matrix = dense
    values = np.asarray(fitted.estimator.predict(matrix), dtype=np.float64)
    if candidate.family == "two_head_ridge":
        return float(values[0, 1] - values[0, 0])
    return float(values.reshape(-1)[0])


def _latency_p95_ms(
    texts: list[str],
    prior: FittedScorer,
    fitted: FittedUplift,
) -> tuple[float, int]:
    candidate = fitted.candidate
    vectorizer = (
        HashingVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            n_features=candidate.dim,
            alternate_sign=True,
            norm="l2",
            lowercase=True,
        )
        if candidate.family in {"direct_ridge", "two_head_ridge"}
        and candidate.dim != 8_192
        else None
    )
    values: list[float] = []
    for text in texts:
        for _ in range(5):
            started = time.perf_counter_ns()
            _single_prediction(text, prior, fitted, vectorizer)
            values.append((time.perf_counter_ns() - started) / 1_000_000)
    return float(np.percentile(np.asarray(values), 95)), len(values)


def fit(
    task_paths: tuple[Path, Path],
    outcome_paths: tuple[Path, Path],
    audit_paths: tuple[Path, Path],
    external_tasks_path: Path,
    external_manifest_path: Path,
    output: Path,
) -> None:
    """Run the complete nested pooled development experiment."""
    if output.exists():
        raise FileExistsError(f"pooled uplift output already exists: {output}")
    cohorts = [
        _load_cohort(
            task_paths[index],
            outcome_paths[index],
            audit_paths[index],
            expected_tasks_hash=TASK_HASHES[index],
            expected_outcomes_hash=OUTCOME_HASHES[index],
            expected_audit_hash=AUDIT_HASHES[index],
        )
        for index in range(2)
    ]
    data = _pool(cohorts)
    if len(data.task_ids) != 393:
        raise ValueError("pooled retained development count changed")
    external = _load_external(external_tasks_path, external_manifest_path)
    prior_candidate = ScorerCandidate(order=8, dim=8_192, alpha=10.0)
    prior = _fit_prior(external, prior_candidate)
    bank = _features(data, prior)
    indices = np.arange(len(data.task_ids), dtype=np.int64)
    folds = list(
        GroupKFold(n_splits=5).split(indices, groups=np.asarray(data.repositories))
    )
    candidate_rows: list[dict[str, object]] = []
    nested_choices: dict[str, np.ndarray] = {}
    null_advantages_by_candidate: list[np.ndarray] = []
    evaluations: list[tuple[np.ndarray, int]] | None = None
    for candidate in _candidate_grid():
        row, choices, null_choices, candidate_evaluations = _evaluate_candidate(
            data,
            bank,
            candidate,
            folds,
        )
        if evaluations is None:
            evaluations = candidate_evaluations
        elif any(
            attempt != candidate_evaluations[index][1]
            or not np.array_equal(test, candidate_evaluations[index][0])
            for index, (test, attempt) in enumerate(evaluations)
        ):
            raise AssertionError("candidate outer evaluations changed")
        null_values = np.asarray(row.pop("null_advantages"), dtype=np.float64)
        null_advantages_by_candidate.append(null_values)
        candidate_rows.append(row)
        nested_choices[candidate.key] = choices
        logger.info(
            "nested candidate=%s reward=%.6f advantage=%.6f bootstrap_lower=%.6f",
            candidate.key,
            float(row["reward"]),
            float(row["matched_blind_advantage"]),
            float(row["bootstrap_ci95_lower"]),
        )
    family_null_max = np.max(np.vstack(null_advantages_by_candidate), axis=0)
    family_null95 = _higher_quantile(family_null_max)
    full_indices = np.arange(len(data.task_ids), dtype=np.int64)
    full_routes: dict[str, np.ndarray] = {}
    statistically_eligible: list[tuple[Candidate, dict[str, object]]] = []
    by_key = {candidate.key: candidate for candidate in _candidate_grid()}
    for row in candidate_rows:
        row["family_wise_null_p95"] = family_null95
        row["real_minus_family_null_p95"] = (
            float(row["matched_blind_advantage"]) - family_null95
        )
        row["statistically_eligible"] = bool(
            float(row["quality_retention"]) >= QUALITY_RETENTION
            and float(row["matched_blind_advantage"]) > 0.0
            and not row["dominated_by_static"]
            and float(row["matched_blind_advantage"]) > family_null95
            and float(row["bootstrap_ci95_lower"]) > 0.0
            and int(row["positive_evaluations"]) >= 7
        )
        if row["statistically_eligible"]:
            statistically_eligible.append((by_key[str(row["key"])], row))
    eligible: list[dict[str, object]] = []
    for candidate, row in statistically_eligible:
        matrix = _matrix(bank, candidate)
        high = data.rewards[:, HIGH_INDEX, :].mean(axis=1)
        maximum = data.rewards[:, MAX_INDEX, :].mean(axis=1)
        fitted = _fit_model(candidate, matrix, full_indices, high, maximum)
        predictions = _predict(fitted, matrix, full_indices)
        threshold, calibration, task_choices = _select_full_threshold(data, predictions)
        latency, samples = _latency_p95_ms(data.texts, prior, fitted)
        row["full_threshold"] = threshold
        row["full_calibration"] = calibration
        row["route_latency_p95_ms"] = latency
        row["route_latency_samples"] = samples
        row["eligible"] = bool(
            threshold is not None
            and task_choices is not None
            and latency < MAX_ROUTE_P95_MS
        )
        if row["eligible"]:
            if task_choices is None:
                raise AssertionError("eligible candidate has no full route")
            full_routes[candidate.key] = task_choices
            eligible.append(row)
    for row in candidate_rows:
        row.setdefault("full_threshold", None)
        row.setdefault("full_calibration", None)
        row.setdefault("route_latency_p95_ms", None)
        row.setdefault("route_latency_samples", 0)
        row.setdefault("eligible", False)
    selected = (
        min(
            eligible,
            key=lambda row: (
                float(row["cost_usd_per_task"]),
                -float(row["real_minus_family_null_p95"]),
                -float(row["reward"]),
                int(row["order"]),
            ),
        )
        if eligible
        else None
    )
    output.mkdir(parents=True)
    report = {
        "protocol": PROTOCOL,
        "development_passed": selected is not None,
        "pooled_tasks": len(data.task_ids),
        "pooled_repositories": len(set(data.repositories)),
        "attempts": ATTEMPTS,
        "arms": ARMS,
        "trained_arms": [ARMS[HIGH_INDEX], ARMS[MAX_INDEX]],
        "task_sha256": list(TASK_HASHES),
        "outcome_sha256": list(OUTCOME_HASHES),
        "audit_sha256": list(AUDIT_HASHES),
        "external_tasks_sha256": _sha256(external_tasks_path),
        "external_manifest_sha256": _sha256(external_manifest_path),
        "external_prior": "charhash8192-a10",
        "outer_folds": 5,
        "outer_evaluations": 10,
        "null_count": NULL_COUNT,
        "null_seed": SEED,
        "family_wise_null_p95": family_null95,
        "family_wise_null_maxima": family_null_max.tolist(),
        "bootstrap_count": BOOTSTRAPS,
        "candidates": candidate_rows,
        "selected": selected,
        "target_outcomes_used": False,
        "deep_swe_outcomes_accessed": False,
        "old_confirmation_used_as": "development-only",
        "internet_access": False,
        "fitted_numeric_router_state_persisted": False,
    }
    report_path = output / "development-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if selected is not None:
        key = str(selected["key"])
        task_choices = full_routes[key]
        routes_path = output / "development-routes.jsonl"
        routes_path.write_text(
            "".join(
                json.dumps(
                    {
                        "task_id": task_id,
                        "repository": data.repositories[index],
                        "arm": ARMS[int(task_choices[index])],
                        "target_outcomes_used": False,
                    },
                    sort_keys=True,
                )
                + "\n"
                for index, task_id in enumerate(data.task_ids)
            ),
            encoding="utf-8",
        )
        lock = {
            "protocol": PROTOCOL,
            "eligible": True,
            "candidate": key,
            "candidate_order": selected["order"],
            "threshold": selected["full_threshold"],
            "arms": {"default": ARMS[HIGH_INDEX], "uplift": ARMS[MAX_INDEX]},
            "family_wise_null_p95": family_null95,
            "development_report_sha256": _sha256(report_path),
            "development_routes_sha256": _sha256(routes_path),
            "target_outcomes_used": False,
            "deep_swe_outcomes_accessed": False,
            "fitted_numeric_router_state_persisted": False,
        }
        (output / "selection-lock.json").write_text(
            json.dumps(lock, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    logger.info(
        "pooled uplift passed=%s statistically_eligible=%d eligible=%d null95=%.6f",
        selected is not None,
        len(statistically_eligible),
        len(eligible),
        family_null95,
    )


def main() -> None:
    """Run the pooled uplift CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-tasks", type=Path, required=True)
    parser.add_argument("--development-outcomes", type=Path, required=True)
    parser.add_argument("--development-audit", type=Path, required=True)
    parser.add_argument("--confirmation-tasks", type=Path, required=True)
    parser.add_argument("--confirmation-outcomes", type=Path, required=True)
    parser.add_argument("--confirmation-audit", type=Path, required=True)
    parser.add_argument("--external-tasks", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fit(
        (args.development_tasks, args.confirmation_tasks),
        (args.development_outcomes, args.confirmation_outcomes),
        (args.development_audit, args.confirmation_audit),
        args.external_tasks,
        args.external_manifest,
        args.output,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
