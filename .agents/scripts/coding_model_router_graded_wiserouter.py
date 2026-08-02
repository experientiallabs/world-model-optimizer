"""Fit a workload-budget router on the graded external SWE-rebench matrix.

This module keeps all fitted state in memory. It exposes pure, deterministic helpers for the
frozen WISERouter-inspired development study and never reads confirmation or target outcomes.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from coding_model_router_graded_irt_protocol import repository_grouped_folds
from coding_model_router_graded_swerebench_fit import (
    ARMS,
    MIN_SAVINGS,
    QUALITY_RETENTION,
    SEEDS,
    Data,
    _metrics,
)
from scipy.optimize import linprog
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import HashingVectorizer

PROTOCOL = "coding-router-graded-wiserouter-v1"
FOLDS = 5
HASH_DIMS = (512, 2_048)
CONTEXT_COUNTS = (8, 16, 32)
SHRINKAGE_VALUES = (0.0, 4.0, 16.0)
SAVINGS_VALUES = (0.40, 0.45, 0.50, 0.55, 0.60)
MIN_CONTEXT_SUPPORT = 8
NULL_COUNT = 128
NULL_SEED_START = 20_260_801
MAX_ROUTE_P50_MS = 5.0
MAX_ROUTE_P95_MS = 20.0


@dataclass(frozen=True)
class Candidate:
    """One frozen workload-budget operating point."""

    order: int
    hash_dim: int
    contexts: int
    shrinkage: float
    savings: float

    @property
    def key(self) -> str:
        """Return a stable candidate identity."""
        return (
            f"hash{self.hash_dim}-j{self.contexts}-shrink{self.shrinkage:g}"
            f"-save{self.savings:g}"
        )


@dataclass(frozen=True)
class FoldPlan:
    """One fitted, repository-disjoint context assignment."""

    fold: int
    train: np.ndarray
    test: np.ndarray
    train_contexts: np.ndarray
    test_contexts: np.ndarray


@dataclass(frozen=True)
class ContextPlan:
    """All five folds for one seed and context representation."""

    seed: int
    hash_dim: int
    contexts: int
    folds: tuple[FoldPlan, ...]


@dataclass(frozen=True)
class ContextStatistics:
    """Fit-only reward, cost, and context statistics used by the LP."""

    context_probability: np.ndarray
    rewards: np.ndarray
    costs: np.ndarray
    baseline_cost: float


@dataclass(frozen=True)
class FullPolicy:
    """Ephemeral all-development state for latency and route freezing."""

    candidate: Candidate
    shape_mean: np.ndarray
    shape_scale: np.ndarray
    centroids: np.ndarray
    statistics: ContextStatistics
    action_probabilities: np.ndarray


def candidate_grid() -> tuple[Candidate, ...]:
    """Return the exact 90-point preregistered grid."""
    result = tuple(
        Candidate(index, hash_dim, contexts, shrinkage, savings)
        for index, (hash_dim, contexts, shrinkage, savings) in enumerate(
            (hash_dim, contexts, shrinkage, savings)
            for hash_dim in HASH_DIMS
            for contexts in CONTEXT_COUNTS
            for shrinkage in SHRINKAGE_VALUES
            for savings in SAVINGS_VALUES
        )
    )
    if len(result) != 90 or len({candidate.key for candidate in result}) != 90:
        raise AssertionError("WISERouter candidate grid is incomplete or duplicated")
    return result


def _structural(text: str) -> list[float]:
    """Return the frozen 15-value pre-call prompt-shape block."""
    lower = text.casefold()
    lines = text.splitlines()
    path_tokens = sum(token.count("/") for token in text.split())
    stack_markers = sum(
        lower.count(marker)
        for marker in ("traceback", "stack trace", " at ", "exception:")
    )
    return [
        math.log1p(len(text)),
        math.log1p(len(text.split())),
        math.log1p(len(lines)),
        math.log1p(text.count("```")),
        math.log1p(stack_markers),
        math.log1p(path_tokens),
        math.log1p(text.count("`") + text.count('"') + text.count("'")),
        math.log1p(sum(lower.count(word) for word in ("fix", "bug", "repair"))),
        math.log1p(lower.count("test")),
        math.log1p(
            sum(lower.count(word) for word in ("dependency", "package", "build"))
        ),
        float("python" in lower or ".py" in lower),
        float("javascript" in lower or ".js" in lower or "node" in lower),
        float("typescript" in lower or ".ts" in lower),
        float("rust" in lower or ".rs" in lower or "cargo" in lower),
        float("golang" in lower or ".go" in lower),
    ]


def _hash_matrix(texts: Sequence[str], dimension: int) -> np.ndarray:
    """Return stateless signed character-hash features."""
    if dimension not in HASH_DIMS and dimension < 2:
        raise ValueError("hash dimension must be at least two")
    vectorizer = HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        n_features=dimension,
        alternate_sign=True,
        norm="l2",
        lowercase=True,
        dtype=np.float64,
    )
    matrix = np.asarray(vectorizer.transform(list(texts)).toarray(), dtype=np.float64)
    if matrix.shape != (len(texts), dimension) or not np.isfinite(matrix).all():
        raise RuntimeError("stateless task hashing produced invalid features")
    return matrix


def _shape_matrix(texts: Sequence[str]) -> np.ndarray:
    """Return unstandardized prompt-shape rows."""
    matrix = np.asarray([_structural(text) for text in texts], dtype=np.float64)
    if matrix.shape != (len(texts), 15) or not np.isfinite(matrix).all():
        raise RuntimeError("prompt-shape transform produced invalid features")
    return matrix


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    """L2-normalize nonzero feature rows."""
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0.0) or not np.isfinite(norms).all():
        raise ValueError("feature rows must have finite positive norms")
    return values / norms[:, None]


def _combined_features(
    hashed: np.ndarray,
    shape: np.ndarray,
    train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize fit-only shape values and return one normalized feature view."""
    mean = np.mean(shape[train], axis=0)
    scale = np.std(shape[train], axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    combined = np.concatenate([hashed, (shape - mean) / scale], axis=1)
    return _normalize_rows(combined), mean, scale


def _assign_contexts(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign rows to normalized centroids by descending cosine similarity."""
    similarities = features @ centroids.T
    if not np.isfinite(similarities).all():
        raise RuntimeError("context similarities contain non-finite values")
    return np.argmax(similarities, axis=1).astype(np.int64)


def build_context_plan(
    texts: Sequence[str],
    repositories: Sequence[str],
    *,
    hash_dim: int,
    contexts: int,
    seed: int,
) -> ContextPlan | None:
    """Fit five repository-disjoint spherical K-means context partitions."""
    if len(texts) != len(repositories) or not texts:
        raise ValueError("texts and repositories must align and be nonempty")
    hashed = _hash_matrix(texts, hash_dim)
    shape = _shape_matrix(texts)
    grouped = repository_grouped_folds(
        np.asarray(repositories, dtype=object),
        n_splits=FOLDS,
        seed=seed,
    )
    plans: list[FoldPlan] = []
    for fold, split in enumerate(grouped):
        features, _, _ = _combined_features(hashed, shape, split.train)
        fitted = KMeans(
            n_clusters=contexts,
            init="k-means++",
            n_init=20,
            max_iter=300,
            random_state=seed * 101 + fold,
            algorithm="lloyd",
        ).fit(features[split.train])
        centroids = _normalize_rows(np.asarray(fitted.cluster_centers_, dtype=np.float64))
        train_contexts = _assign_contexts(features[split.train], centroids)
        if np.min(np.bincount(train_contexts, minlength=contexts)) < MIN_CONTEXT_SUPPORT:
            return None
        plans.append(
            FoldPlan(
                fold=fold,
                train=split.train,
                test=split.test,
                train_contexts=train_contexts,
                test_contexts=_assign_contexts(features[split.test], centroids),
            )
        )
    return ContextPlan(
        seed=seed,
        hash_dim=hash_dim,
        contexts=contexts,
        folds=tuple(plans),
    )


def _context_statistics(
    assignments: np.ndarray,
    rewards: np.ndarray,
    costs: np.ndarray,
    *,
    contexts: int,
    shrinkage: float,
) -> ContextStatistics:
    """Estimate shrunken context-arm reward and cost means."""
    if (
        assignments.shape != (len(rewards),)
        or rewards.ndim != 2
        or costs.shape != rewards.shape
        or np.any((assignments < 0) | (assignments >= contexts))
        or shrinkage < 0.0
        or not np.isfinite(rewards).all()
        or not np.isfinite(costs).all()
    ):
        raise ValueError("context statistics received invalid arrays")
    counts = np.bincount(assignments, minlength=contexts).astype(np.float64)
    if np.min(counts) < MIN_CONTEXT_SUPPORT:
        raise ValueError("context support fell below the frozen minimum")
    global_rewards = np.mean(rewards, axis=0)
    global_costs = np.mean(costs, axis=0)
    context_rewards = np.empty((contexts, rewards.shape[1]), dtype=np.float64)
    context_costs = np.empty_like(context_rewards)
    for context in range(contexts):
        selected = assignments == context
        denominator = counts[context] + shrinkage
        context_rewards[context] = (
            np.sum(rewards[selected], axis=0) + shrinkage * global_rewards
        ) / denominator
        context_costs[context] = (
            np.sum(costs[selected], axis=0) + shrinkage * global_costs
        ) / denominator
    baseline = min(
        range(rewards.shape[1]),
        key=lambda arm: (-global_rewards[arm], global_costs[arm], arm),
    )
    return ContextStatistics(
        context_probability=counts / np.sum(counts),
        rewards=context_rewards,
        costs=context_costs,
        baseline_cost=float(global_costs[baseline]),
    )


def solve_workload_policy(
    statistics: ContextStatistics,
    *,
    savings: float,
) -> np.ndarray:
    """Solve the frozen offline workload-level reward maximization LP."""
    contexts, arms = statistics.rewards.shape
    if (
        statistics.costs.shape != (contexts, arms)
        or statistics.context_probability.shape != (contexts,)
        or not 0.0 <= savings < 1.0
    ):
        raise ValueError("workload statistics or savings target are invalid")
    weights = statistics.context_probability[:, None]
    objective = -(weights * statistics.rewards).reshape(-1)
    cost_row = (weights * statistics.costs).reshape(1, -1)
    budget = (1.0 - savings) * statistics.baseline_cost
    equalities = np.zeros((contexts, contexts * arms), dtype=np.float64)
    for context in range(contexts):
        equalities[context, context * arms : (context + 1) * arms] = 1.0
    result = linprog(
        objective,
        A_ub=cost_row,
        b_ub=np.asarray([budget], dtype=np.float64),
        A_eq=equalities,
        b_eq=np.ones(contexts, dtype=np.float64),
        bounds=(0.0, 1.0),
        method="highs",
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"workload LP failed: {result.message}")
    policy = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0).reshape(
        contexts, arms
    )
    policy /= np.sum(policy, axis=1, keepdims=True)
    expected_cost = float(np.sum(weights * policy * statistics.costs))
    if expected_cost > budget + 1e-8:
        raise RuntimeError("workload LP exceeded its expected budget")
    return policy


def _digest_int(*parts: str) -> int:
    """Return a deterministic integer digest."""
    return int.from_bytes(hashlib.sha256(":".join(parts).encode()).digest()[:8], "big")


def deterministic_choices(
    task_ids: Sequence[str],
    indices: np.ndarray,
    assignments: np.ndarray,
    policy: np.ndarray,
    statistics: ContextStatistics,
    *,
    seed: int,
    fold: int,
) -> np.ndarray:
    """Round context probabilities to complete label-free task choices."""
    if (
        indices.shape != assignments.shape
        or policy.shape != statistics.rewards.shape
        or len(task_ids) <= int(np.max(indices, initial=-1))
    ):
        raise ValueError("deterministic rounding inputs do not align")
    result = np.full(len(indices), -1, dtype=np.int64)
    arms = policy.shape[1]
    for context in range(policy.shape[0]):
        local = np.flatnonzero(assignments == context)
        if not len(local):
            continue
        ordered = sorted(
            (int(value) for value in local),
            key=lambda value: hashlib.sha256(
                (
                    f"wiserouter-v1:{seed}:{fold}:{context}:"
                    f"{task_ids[int(indices[value])]}"
                ).encode()
            ).digest(),
        )
        offset = _digest_int("wiserouter-v1", str(seed), str(fold), str(context)) % len(
            ordered
        )
        ordered = ordered[offset:] + ordered[:offset]
        targets = policy[context] * len(ordered)
        counts = np.floor(targets + 1e-12).astype(np.int64)
        remaining = len(ordered) - int(np.sum(counts))
        if remaining < 0:
            raise RuntimeError("probability rounding over-allocated a context")
        fractional = targets - counts
        allocation_order = sorted(
            range(arms),
            key=lambda arm: (
                -float(fractional[arm]),
                float(statistics.costs[context, arm]),
                -float(statistics.rewards[context, arm]),
                arm,
            ),
        )
        for arm in allocation_order[:remaining]:
            counts[arm] += 1
        cursor = 0
        arm_order = sorted(
            range(arms),
            key=lambda arm: (
                float(statistics.costs[context, arm]),
                -float(statistics.rewards[context, arm]),
                arm,
            ),
        )
        for arm in arm_order:
            next_cursor = cursor + int(counts[arm])
            result[np.asarray(ordered[cursor:next_cursor], dtype=np.int64)] = arm
            cursor = next_cursor
        if cursor != len(ordered):
            raise RuntimeError("probability rounding did not cover a context")
    if np.any(result < 0):
        raise RuntimeError("deterministic routing did not cover every task")
    return result


def evaluate_candidate_seed(
    data: Data,
    candidate: Candidate,
    plan: ContextPlan,
    *,
    fit_rewards: np.ndarray | None = None,
    fit_costs: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate one candidate with fit-only labels and actual held-out outcomes."""
    if (
        plan.seed not in SEEDS
        or plan.hash_dim != candidate.hash_dim
        or plan.contexts != candidate.contexts
    ):
        raise ValueError("candidate and context plan do not match")
    train_rewards = data.rewards if fit_rewards is None else fit_rewards
    train_costs = data.costs if fit_costs is None else fit_costs
    if train_rewards.shape != data.rewards.shape or train_costs.shape != data.costs.shape:
        raise ValueError("fit-only outcome matrices do not align")
    choices = np.full(len(data.task_ids), -1, dtype=np.int64)
    for fold in plan.folds:
        statistics = _context_statistics(
            fold.train_contexts,
            train_rewards[fold.train],
            train_costs[fold.train],
            contexts=candidate.contexts,
            shrinkage=candidate.shrinkage,
        )
        policy = solve_workload_policy(statistics, savings=candidate.savings)
        choices[fold.test] = deterministic_choices(
            data.task_ids,
            fold.test,
            fold.test_contexts,
            policy,
            statistics,
            seed=plan.seed,
            fold=fold.fold,
        )
    if np.any(choices < 0):
        raise RuntimeError("cross-fitted workload route is incomplete")
    return {"seed": plan.seed, **_metrics(data, choices)}


def passes_primary_gates(seed_metrics: Sequence[dict[str, Any]]) -> bool:
    """Apply the frozen point gates independently to every seed."""
    return (
        len(seed_metrics) == len(SEEDS)
        and all(float(row["quality_retention"]) >= QUALITY_RETENTION for row in seed_metrics)
        and all(float(row["cost_savings"]) >= MIN_SAVINGS for row in seed_metrics)
        and all(float(row["matched_blind_advantage"]) > 0.0 for row in seed_metrics)
        and all(not row["dominated_by_static"] for row in seed_metrics)
    )


def _language(text: str) -> str:
    """Extract the frozen language line from one exact pre-call task text."""
    match = re.search(r"(?:^|\n)language=([^\n]+)", text)
    return match.group(1).strip().casefold() if match else "unknown"


def permute_repository_blocks(values: np.ndarray, data: Data, *, seed: int) -> np.ndarray:
    """Permute complete repository outcome blocks inside exact frozen strata."""
    if values.shape[0] != len(data.task_ids):
        raise ValueError("repository-block values do not align with tasks")
    by_repository: dict[str, list[int]] = {}
    for index, repository in enumerate(data.repositories):
        by_repository.setdefault(repository, []).append(index)
    strata: dict[tuple[int, tuple[str, ...]], list[str]] = {}
    languages = [_language(text) for text in data.texts]
    ordered_indices: dict[str, list[int]] = {}
    for repository, indices in by_repository.items():
        ordered = sorted(indices, key=lambda index: (languages[index], data.task_ids[index]))
        ordered_indices[repository] = ordered
        signature = (len(ordered), tuple(languages[index] for index in ordered))
        strata.setdefault(signature, []).append(repository)
    result = values.copy()
    rng = np.random.default_rng(seed)
    for repositories in strata.values():
        ordered_repositories = sorted(repositories)
        if len(ordered_repositories) < 2:
            continue
        shuffled = list(
            np.asarray(ordered_repositories, dtype=object)[
                rng.permutation(len(ordered_repositories))
            ]
        )
        sources = shuffled[1:] + shuffled[:1]
        for target, source in zip(shuffled, sources, strict=True):
            result[ordered_indices[str(target)]] = values[ordered_indices[str(source)]]
    return result


def higher_quantile(values: Sequence[float], probability: float) -> float:
    """Return the conservative higher empirical quantile."""
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("quantile inputs are invalid")
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability, method="higher"))


def null_gate(
    real_metrics: Sequence[dict[str, Any]],
    null_metrics: Sequence[Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    """Evaluate the frozen 128 repository-block null controls."""
    if len(real_metrics) != len(SEEDS) or len(null_metrics) != NULL_COUNT:
        raise ValueError("null gate requires five real seeds and 128 complete nulls")
    real_by_seed = {
        int(row["seed"]): float(row["matched_blind_advantage"]) for row in real_metrics
    }
    if set(real_by_seed) != set(SEEDS):
        raise ValueError("real seed metrics are incomplete")
    null_means: list[float] = []
    null_by_seed = {seed: [] for seed in SEEDS}
    for rows in null_metrics:
        values = {int(row["seed"]): float(row["matched_blind_advantage"]) for row in rows}
        if set(values) != set(SEEDS):
            raise ValueError("one null route lacks complete seed metrics")
        null_means.append(float(np.mean(list(values.values()))))
        for seed in SEEDS:
            null_by_seed[seed].append(values[seed])
    null95 = higher_quantile(null_means, 0.95)
    seed_margins = {
        str(seed): real_by_seed[seed] - higher_quantile(null_by_seed[seed], 0.95)
        for seed in SEEDS
    }
    real_mean = float(np.mean(list(real_by_seed.values())))
    passed = real_mean > null95 and sum(value > 0.0 for value in seed_margins.values()) >= 4
    return {
        "passed": passed,
        "real_mean_matched_blind_advantage": real_mean,
        "null95_mean_matched_blind_advantage": null95,
        "real_minus_null95": real_mean - null95,
        "seed_real_minus_null95": seed_margins,
    }


def fit_full_policy(data: Data, candidate: Candidate, *, seed: int) -> FullPolicy:
    """Fit one ephemeral all-development policy for latency or route freezing."""
    hashed = _hash_matrix(data.texts, candidate.hash_dim)
    shape = _shape_matrix(data.texts)
    all_indices = np.arange(len(data.task_ids), dtype=np.int64)
    features, mean, scale = _combined_features(hashed, shape, all_indices)
    fitted = KMeans(
        n_clusters=candidate.contexts,
        init="k-means++",
        n_init=20,
        max_iter=300,
        random_state=seed,
        algorithm="lloyd",
    ).fit(features)
    centroids = _normalize_rows(np.asarray(fitted.cluster_centers_, dtype=np.float64))
    assignments = _assign_contexts(features, centroids)
    statistics = _context_statistics(
        assignments,
        data.rewards,
        data.costs,
        contexts=candidate.contexts,
        shrinkage=candidate.shrinkage,
    )
    return FullPolicy(
        candidate=candidate,
        shape_mean=mean,
        shape_scale=scale,
        centroids=centroids,
        statistics=statistics,
        action_probabilities=solve_workload_policy(
            statistics,
            savings=candidate.savings,
        ),
    )


def _one_feature(text: str, policy: FullPolicy) -> np.ndarray:
    """Build one exact online feature row from stateless task text."""
    hashed = _hash_matrix([text], policy.candidate.hash_dim)
    shape = (_shape_matrix([text]) - policy.shape_mean) / policy.shape_scale
    return _normalize_rows(np.concatenate([hashed, shape], axis=1))


def online_choice(text: str, task_id: str, policy: FullPolicy, *, step: int) -> int:
    """Execute one conservative full-path workload decision."""
    context = int(_assign_contexts(_one_feature(text, policy), policy.centroids)[0])
    probabilities = policy.action_probabilities[context]
    unit = _digest_int("wiserouter-online-v1", task_id, str(step)) / float(2**64)
    choice = int(np.searchsorted(np.cumsum(probabilities), unit, side="right"))
    return min(choice, len(ARMS) - 1)


def freeze_choices(
    policy: FullPolicy,
    task_ids: Sequence[str],
    texts: Sequence[str],
    *,
    seed: int = 20_260_801,
) -> np.ndarray:
    """Freeze one complete label-free workload route for a known batch."""
    if len(task_ids) != len(texts) or not task_ids:
        raise ValueError("route-freeze task identities and texts must align")
    hashed = _hash_matrix(texts, policy.candidate.hash_dim)
    shape = (_shape_matrix(texts) - policy.shape_mean) / policy.shape_scale
    features = _normalize_rows(np.concatenate([hashed, shape], axis=1))
    assignments = _assign_contexts(features, policy.centroids)
    return deterministic_choices(
        task_ids,
        np.arange(len(task_ids), dtype=np.int64),
        assignments,
        policy.action_probabilities,
        policy.statistics,
        seed=seed,
        fold=0,
    )


def measure_latency_ms(
    policy: FullPolicy,
    task_ids: Sequence[str],
    texts: Sequence[str],
    *,
    decisions: int = 10_000,
) -> dict[str, Any]:
    """Measure the full no-network decision path on one CPU."""
    if len(task_ids) != len(texts) or not task_ids or decisions < 10_000:
        raise ValueError("latency inputs must contain at least 10,000 decisions")
    samples = np.empty(decisions, dtype=np.float64)
    for step in range(decisions):
        index = step % len(task_ids)
        started = time.perf_counter_ns()
        online_choice(texts[index], task_ids[index], policy, step=step)
        samples[step] = (time.perf_counter_ns() - started) / 1_000_000.0
    p50 = float(np.percentile(samples, 50))
    p95 = float(np.percentile(samples, 95))
    return {
        "decisions": decisions,
        "p50_ms": p50,
        "p95_ms": p95,
        "eligible": p50 < MAX_ROUTE_P50_MS and p95 < MAX_ROUTE_P95_MS,
        "network_calls": 0,
        "single_core": True,
    }
