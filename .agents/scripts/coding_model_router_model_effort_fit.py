"""Select and seal an external model-by-effort router without target outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge

logger = logging.getLogger("coding-router-model-effort-fit")

PROTOCOL = "coding-router-model-effort-pair-selection-v1"
CORPUS_SHA256 = "7d846b5576d15e68fd18ac21bfe0610cc1614b3b35ec0ae0cb8cfae0b82962c1"
CONFIRMATION_CORPUS_SHA256 = "9798dd1e58be0d13331d097307670dc3fc3760ad211da20e6367666523f080a7"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
MODELS = ("luna", "terra", "sol")
MODEL_IDS = {
    "luna": "gpt-5.6-luna",
    "terra": "gpt-5.6-terra",
    "sol": "gpt-5.6-sol",
}
ARMS = tuple(f"{model}-{effort}" for model in MODELS for effort in EFFORTS)
ATTEMPTS = 2
SEEDS = (11, 23, 37, 41, 59)
FOLDS = 5
HASH_DIMS = (512, 2_048, 8_192)
RIDGE_ALPHAS = (1.0, 10.0, 100.0)
RIDGE_THRESHOLDS = (-0.10, 0.0, 0.02, 0.05, 0.10, 0.20)
KNN_COUNTS = (8, 16, 32, 64)
KNN_Z = (0.0, 0.5, 1.0, 1.645, 2.0)
QUALITY_RETENTION = 0.95
MIN_SAVINGS = 0.40
NULL_COUNT = 128
MAX_ROUTE_P95_MS = 5.0

Family = Literal["ridge", "knn"]


@dataclass(frozen=True)
class Data:
    """Dense task-level model-by-effort outcomes."""

    task_ids: list[str]
    repositories: list[str]
    texts: list[str]
    rewards: np.ndarray
    costs: np.ndarray


@dataclass(frozen=True)
class Candidate:
    """One frozen pair policy configuration."""

    order: int
    family: Family
    cheap: int
    expensive: int
    dim: int
    alpha: float = 0.0
    threshold: float = 0.0
    k: int = 0
    z: float = 0.0

    @property
    def key(self) -> str:
        """Return the stable candidate identity."""
        pair = f"{ARMS[self.cheap]}__{ARMS[self.expensive]}"
        if self.family == "ridge":
            return (
                f"ridge-{pair}-hash{self.dim}-a{self.alpha:g}"
                f"-threshold{self.threshold:g}"
            )
        return f"knn-{pair}-hash{self.dim}-k{self.k}-z{self.z:g}"


@dataclass(frozen=True)
class Metrics:
    """One route's cost-quality and matched-blind measurements."""

    reward: float
    cost_usd: float
    retention: float
    savings: float
    blind_reward: float
    blind_cost_usd: float
    advantage: float
    expensive_traffic: int
    dominated_by_static: tuple[str, ...]


@dataclass(frozen=True)
class FittedRoute:
    """Ephemeral in-memory state for latency-faithful route decisions."""

    candidate: Candidate
    vectorizer: HashingVectorizer
    train: sparse.csr_matrix
    train_y: np.ndarray
    ridge: Ridge | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
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


def load_data(corpus: Path, outcomes: Path, audit_path: Path) -> Data:
    """Load a complete whole-task-intersected 15-arm matrix."""
    if _sha256(corpus) != CORPUS_SHA256:
        raise ValueError("development corpus changed")
    audit = _read_object(audit_path)
    retained = audit.get("tasks")
    exclusions = audit.get("excluded_tasks")
    outcomes_sha = _sha256(outcomes)
    if (
        audit.get("valid") is not True
        or not isinstance(retained, int)
        or not 190 <= retained <= 200
        or not isinstance(exclusions, list)
        or len(exclusions) != 200 - retained
        or audit.get("retained_task_coverage") != retained / 200
        or audit.get("arms") != list(ARMS)
        or audit.get("cells") != retained * len(ARMS) * ATTEMPTS
        or audit.get("outcomes_sha256") != outcomes_sha
        or audit.get("target_outcomes_used") is not False
        or audit.get("deep_swe_outcomes_accessed") is not False
    ):
        raise ValueError("merged development audit is incomplete or unsafe")
    raw_tasks = _read_object(corpus).get("tasks")
    if not isinstance(raw_tasks, list) or len(raw_tasks) != 200:
        raise ValueError("development corpus must contain 200 tasks")
    excluded_ids = {
        str(row["task_id"])
        for row in exclusions
        if isinstance(row, dict) and row.get("scope") == "whole-task"
    }
    tasks = [
        {str(key): item for key, item in task.items()}
        for task in raw_tasks
        if isinstance(task, dict) and str(task.get("task_id")) not in excluded_ids
    ]
    if len(tasks) != retained:
        raise ValueError("retained tasks do not match whole-task exclusions")
    task_ids = [str(task["task_id"]) for task in tasks]
    task_index = {task_id: index for index, task_id in enumerate(task_ids)}
    arm_index = {arm: index for index, arm in enumerate(ARMS)}
    rewards = np.full((retained, len(ARMS), ATTEMPTS), np.nan)
    costs = np.full_like(rewards, np.nan)
    observed: set[tuple[str, str, int]] = set()
    for row in _read_rows(outcomes):
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
            or row.get("model") != MODEL_IDS[arm.split("-", 1)[0]]
            or row.get("target_outcomes_used") is not False
            or isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or float(reward) not in {0.0, 1.0}
            or isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(float(cost))
            or float(cost) < 0.0
        ):
            raise ValueError(f"invalid merged outcome {identity!r}")
        index = (task_index[task_id], arm_index[arm], attempt)
        rewards[index] = float(reward)
        costs[index] = float(cost)
        observed.add(identity)
    if len(observed) != retained * len(ARMS) * ATTEMPTS:
        raise ValueError("merged matrix is not dense")
    return Data(
        task_ids=task_ids,
        repositories=[str(task["repository"]) for task in tasks],
        texts=[_task_text(task) for task in tasks],
        rewards=rewards.mean(axis=2),
        costs=costs.mean(axis=2),
    )


def load_confirmation(path: Path) -> list[dict[str, Any]]:
    """Load the untouched label-free external confirmation cohort."""
    if _sha256(path) != CONFIRMATION_CORPUS_SHA256:
        raise ValueError("confirmation corpus changed")
    raw = _read_object(path).get("tasks")
    if not isinstance(raw, list) or len(raw) != 200:
        raise ValueError("confirmation corpus must contain 200 tasks")
    tasks = [
        {str(key): item for key, item in task.items()}
        for task in raw
        if isinstance(task, dict)
    ]
    if len(tasks) != 200 or len({str(task["task_id"]) for task in tasks}) != 200:
        raise ValueError("confirmation task identities are invalid")
    return tasks


def _vectorizer(dim: int) -> HashingVectorizer:
    return HashingVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        n_features=dim,
        alternate_sign=True,
        norm="l2",
        lowercase=True,
    )


def grouped_folds(groups: list[str], seed: int) -> np.ndarray:
    """Assign complete repositories to balanced deterministic folds."""
    unique = sorted(
        set(groups),
        key=lambda group: hashlib.sha256(f"{seed}|{group}".encode()).hexdigest(),
    )
    mapping = {group: index % FOLDS for index, group in enumerate(unique)}
    result = np.asarray([mapping[group] for group in groups], dtype=np.int64)
    for fold in range(FOLDS):
        train_groups = {groups[index] for index in np.flatnonzero(result != fold)}
        test_groups = {groups[index] for index in np.flatnonzero(result == fold)}
        if not test_groups or train_groups & test_groups:
            raise ValueError("repository-grouped fold assignment is invalid")
    return result


def candidate_grid(rewards: np.ndarray, costs: np.ndarray) -> tuple[Candidate, ...]:
    """Enumerate every baseline-guard pair and frozen parameter setting."""
    mean_rewards = rewards.mean(axis=0)
    mean_costs = costs.mean(axis=0)
    baseline = min(
        range(len(ARMS)),
        key=lambda arm: (-mean_rewards[arm], mean_costs[arm], arm),
    )
    values: list[Candidate] = []
    for left, right in combinations(range(len(ARMS)), 2):
        if baseline not in {left, right}:
            continue
        cheap, expensive = sorted((left, right), key=lambda arm: (mean_costs[arm], arm))
        for dim in HASH_DIMS:
            for alpha in RIDGE_ALPHAS:
                for threshold in RIDGE_THRESHOLDS:
                    values.append(
                        Candidate(
                            order=len(values),
                            family="ridge",
                            cheap=cheap,
                            expensive=expensive,
                            dim=dim,
                            alpha=alpha,
                            threshold=threshold,
                        )
                    )
            for k in KNN_COUNTS:
                for z in KNN_Z:
                    values.append(
                        Candidate(
                            order=len(values),
                            family="knn",
                            cheap=cheap,
                            expensive=expensive,
                            dim=dim,
                            k=k,
                            z=z,
                        )
                    )
    result = tuple(values)
    if len(result) != 1_596 or len({candidate.key for candidate in result}) != len(result):
        raise AssertionError("candidate grid is incomplete or duplicated")
    return result


def _static_baseline(data: Data) -> int:
    rewards = data.rewards.mean(axis=0)
    costs = data.costs.mean(axis=0)
    return min(
        range(len(ARMS)),
        key=lambda arm: (-rewards[arm], costs[arm], arm),
    )


def _neighbors(
    features: sparse.csr_matrix,
    folds: np.ndarray,
    k: int,
) -> np.ndarray:
    result = np.full((features.shape[0], k), -1, dtype=np.int64)
    for fold in range(FOLDS):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        similarity = (features[test] @ features[train].T).toarray()
        take = min(k, len(train))
        order = np.argsort(-similarity, axis=1, kind="stable")[:, :take]
        result[test, :take] = train[order]
        if take < k:
            result[test, take:] = result[test, take - 1 : take]
    if (result < 0).any():
        raise ValueError("kNN cross-fit left an unfilled neighbor")
    return result


def _metrics(data: Data, choices: np.ndarray, cheap: int, expensive: int) -> Metrics:
    rows = np.arange(len(choices))
    reward = float(np.mean(data.rewards[rows, choices]))
    cost = float(np.mean(data.costs[rows, choices]))
    static_rewards = data.rewards.mean(axis=0)
    static_costs = data.costs.mean(axis=0)
    baseline = min(
        range(len(ARMS)),
        key=lambda arm: (-static_rewards[arm], static_costs[arm], arm),
    )
    baseline_reward = float(static_rewards[baseline])
    baseline_cost = float(static_costs[baseline])
    expensive_count = int(np.sum(choices == expensive))
    expensive_share = expensive_count / len(choices)
    blind_reward = float(
        (1.0 - expensive_share) * static_rewards[cheap]
        + expensive_share * static_rewards[expensive]
    )
    blind_cost = float(
        (1.0 - expensive_share) * static_costs[cheap]
        + expensive_share * static_costs[expensive]
    )
    dominated = tuple(
        ARMS[arm]
        for arm in range(len(ARMS))
        if static_rewards[arm] >= reward
        and static_costs[arm] <= cost
        and (static_rewards[arm] > reward or static_costs[arm] < cost)
    )
    return Metrics(
        reward=reward,
        cost_usd=cost,
        retention=reward / baseline_reward if baseline_reward else 0.0,
        savings=1.0 - cost / baseline_cost if baseline_cost else 0.0,
        blind_reward=blind_reward,
        blind_cost_usd=blind_cost,
        advantage=reward - blind_reward,
        expensive_traffic=expensive_count,
        dominated_by_static=dominated,
    )


def _metric_json(metrics: Metrics) -> dict[str, object]:
    return {
        "reward": metrics.reward,
        "cost_usd_per_task": metrics.cost_usd,
        "quality_retention": metrics.retention,
        "cost_savings": metrics.savings,
        "matched_blind_reward": metrics.blind_reward,
        "matched_blind_cost_usd_per_task": metrics.blind_cost_usd,
        "matched_blind_advantage": metrics.advantage,
        "expensive_traffic": metrics.expensive_traffic,
        "dominated_by_static": list(metrics.dominated_by_static),
    }


def _crossfit(data: Data) -> tuple[Candidate, dict[str, object]] | None:
    features = {dim: _vectorizer(dim).transform(data.texts).tocsr() for dim in HASH_DIMS}
    candidates = candidate_grid(data.rewards, data.costs)
    choices: dict[tuple[int, str], np.ndarray] = {}
    for seed in SEEDS:
        folds = grouped_folds(data.repositories, seed)
        ridge_predictions: dict[tuple[int, float], np.ndarray] = {}
        neighbor_indices: dict[tuple[int, int], np.ndarray] = {}
        for dim in HASH_DIMS:
            matrix = features[dim]
            for alpha in RIDGE_ALPHAS:
                predicted = np.empty_like(data.rewards)
                for fold in range(FOLDS):
                    train = np.flatnonzero(folds != fold)
                    test = np.flatnonzero(folds == fold)
                    model = Ridge(alpha=alpha)
                    model.fit(matrix[train], data.rewards[train])
                    predicted[test] = model.predict(matrix[test])
                ridge_predictions[(dim, alpha)] = predicted
            for k in KNN_COUNTS:
                neighbor_indices[(dim, k)] = _neighbors(matrix, folds, k)
        for candidate in candidates:
            if candidate.family == "ridge":
                prediction = ridge_predictions[(candidate.dim, candidate.alpha)]
                delta = prediction[:, candidate.expensive] - prediction[:, candidate.cheap]
                route = np.where(
                    delta > candidate.threshold,
                    candidate.expensive,
                    candidate.cheap,
                )
            else:
                neighbors = neighbor_indices[(candidate.dim, candidate.k)]
                paired = (
                    data.rewards[neighbors, candidate.expensive]
                    - data.rewards[neighbors, candidate.cheap]
                )
                mean = paired.mean(axis=1)
                se = paired.std(axis=1, ddof=1) / math.sqrt(candidate.k)
                route = np.where(
                    mean - candidate.z * se > 0.0,
                    candidate.expensive,
                    candidate.cheap,
                )
            choices[(seed, candidate.key)] = route.astype(np.int64)

    eligible: list[tuple[float, float, int, Candidate, list[Metrics]]] = []
    evaluated: list[dict[str, object]] = []
    for candidate in candidates:
        seed_metrics = [
            _metrics(
                data,
                choices[(seed, candidate.key)],
                candidate.cheap,
                candidate.expensive,
            )
            for seed in SEEDS
        ]
        passed = (
            all(metric.retention >= QUALITY_RETENTION for metric in seed_metrics)
            and all(metric.savings >= MIN_SAVINGS for metric in seed_metrics)
            and float(np.mean([metric.advantage for metric in seed_metrics])) > 0.0
            and all(not metric.dominated_by_static for metric in seed_metrics)
        )
        mean_cost = float(np.mean([metric.cost_usd for metric in seed_metrics]))
        mean_reward = float(np.mean([metric.reward for metric in seed_metrics]))
        row = {
            "candidate": candidate.key,
            "family": candidate.family,
            "cheap_arm": ARMS[candidate.cheap],
            "expensive_arm": ARMS[candidate.expensive],
            "eligible": passed,
            "mean_reward": mean_reward,
            "mean_cost_usd_per_task": mean_cost,
            "mean_matched_blind_advantage": float(
                np.mean([metric.advantage for metric in seed_metrics])
            ),
            "seeds": [
                {"seed": seed, **_metric_json(metric)}
                for seed, metric in zip(SEEDS, seed_metrics, strict=True)
            ],
        }
        evaluated.append(row)
        if passed:
            eligible.append((mean_cost, -mean_reward, candidate.order, candidate, seed_metrics))
    if not eligible:
        return None
    _, _, _, selected, selected_metrics = min(eligible)
    report = {
        "candidate_count": len(candidates),
        "eligible_count": len(eligible),
        "selected_candidate": selected.key,
        "selected_order": selected.order,
        "selected_family": selected.family,
        "selected_pair": [ARMS[selected.cheap], ARMS[selected.expensive]],
        "development_static_baseline": ARMS[_static_baseline(data)],
        "selected_seed_metrics": [
            {"seed": seed, **_metric_json(metric)}
            for seed, metric in zip(SEEDS, selected_metrics, strict=True)
        ],
        "top_eligible": [
            next(row for row in evaluated if row["candidate"] == item[3].key)
            for item in sorted(eligible)[:100]
        ],
    }
    return selected, report


def _fit_router(
    candidate: Candidate,
    train_texts: list[str],
    train_y: np.ndarray,
) -> FittedRoute:
    vectorizer = _vectorizer(candidate.dim)
    train = vectorizer.transform(train_texts).tocsr()
    model: Ridge | None = None
    if candidate.family == "ridge":
        model = Ridge(alpha=candidate.alpha)
        model.fit(train, train_y)
    return FittedRoute(candidate, vectorizer, train, train_y, model)


def _predict_route(fitted: FittedRoute, route_texts: list[str]) -> np.ndarray:
    candidate = fitted.candidate
    route = fitted.vectorizer.transform(route_texts).tocsr()
    if candidate.family == "ridge":
        if fitted.ridge is None:
            raise ValueError("fitted Ridge route lacks its ephemeral estimator")
        delta = np.asarray(fitted.ridge.predict(route), dtype=np.float64)
        return np.where(delta > candidate.threshold, candidate.expensive, candidate.cheap)
    similarity = (route @ fitted.train.T).toarray()
    order = np.argsort(-similarity, axis=1, kind="stable")[:, : candidate.k]
    paired = fitted.train_y[order]
    mean = paired.mean(axis=1)
    se = paired.std(axis=1, ddof=1) / math.sqrt(candidate.k)
    return np.where(mean - candidate.z * se > 0.0, candidate.expensive, candidate.cheap)


def _fit_route(
    candidate: Candidate,
    train_texts: list[str],
    train_y: np.ndarray,
    route_texts: list[str],
) -> np.ndarray:
    return _predict_route(_fit_router(candidate, train_texts, train_y), route_texts)


def _group_permutation(values: np.ndarray, groups: list[str], seed: int) -> np.ndarray:
    unique = sorted(set(groups))
    means = np.asarray([values[np.asarray(groups) == group].mean() for group in unique])
    shuffled = np.random.default_rng(seed).permutation(means)
    mapping = dict(zip(unique, shuffled, strict=True))
    return np.asarray([mapping[group] for group in groups], dtype=np.float64)


def _route_rows(
    tasks: list[dict[str, Any]],
    choices: np.ndarray,
    candidate: Candidate,
    *,
    null_index: int | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task, choice in zip(tasks, choices, strict=True):
        arm = ARMS[int(choice)]
        model_prefix, effort = arm.split("-", 1)
        row: dict[str, object] = {
            "task_id": str(task["task_id"]),
            "arm": arm,
            "model": MODEL_IDS[model_prefix],
            "reasoning_effort": effort,
            "selected_candidate": candidate.key,
            "target_outcomes_used": False,
            "deep_swe_outcomes_accessed": False,
        }
        if null_index is not None:
            row["null_index"] = null_index
        rows.append(row)
    return rows


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def fit(
    corpus: Path,
    outcomes: Path,
    audit_path: Path,
    confirmation_path: Path,
    output: Path,
) -> None:
    """Cross-fit the frozen grid and seal one pair route plus null controls."""
    data = load_data(corpus, outcomes, audit_path)
    confirmation = load_confirmation(confirmation_path)
    result = _crossfit(data)
    output.mkdir(parents=True, exist_ok=False)
    inputs = {
        "corpus_sha256": _sha256(corpus),
        "outcomes_sha256": _sha256(outcomes),
        "completion_audit_sha256": _sha256(audit_path),
        "confirmation_corpus_sha256": _sha256(confirmation_path),
    }
    if result is None:
        report = {
            "protocol": PROTOCOL,
            "valid": True,
            "development_passed": False,
            "confirmation_authorized": False,
            "reason": "no candidate passed frozen development gates",
            "inputs": inputs,
            "target_outcomes_used": False,
            "deep_swe_outcomes_accessed": False,
            "fitted_numeric_router_state_persisted": False,
        }
        (output / "selection-report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return
    selected, selection = result
    train_y = data.rewards[:, selected.expensive] - data.rewards[:, selected.cheap]
    route_texts = [_task_text(task) for task in confirmation]
    fitted = _fit_router(selected, data.texts, train_y)
    start = time.perf_counter()
    choices = _predict_route(fitted, route_texts)
    batch_elapsed_ms = (time.perf_counter() - start) * 1000.0
    latencies: list[float] = []
    for text in route_texts:
        start = time.perf_counter()
        _predict_route(fitted, [text])
        latencies.append((time.perf_counter() - start) * 1000.0)
    latency_p95 = float(np.percentile(latencies, 95))
    if latency_p95 >= MAX_ROUTE_P95_MS:
        raise ValueError(f"selected route exceeds latency gate: {latency_p95:.6f} ms")

    real_rows = _route_rows(confirmation, choices, selected)
    expensive_count = int(np.sum(choices == selected.expensive))
    blind_order = sorted(
        range(len(confirmation)),
        key=lambda index: hashlib.sha256(
            f"matched-blind|{confirmation[index]['task_id']}".encode()
        ).hexdigest(),
    )
    blind = np.full(len(confirmation), selected.cheap, dtype=np.int64)
    blind[blind_order[:expensive_count]] = selected.expensive
    blind_rows = _route_rows(confirmation, blind, selected)

    null_rows: list[dict[str, object]] = []
    null_hashes: set[str] = set()
    null_seeds: list[int] = []
    seed = 20_260_801
    while len(null_seeds) < NULL_COUNT and seed < 20_270_801:
        null_y = _group_permutation(train_y, data.repositories, seed)
        null_choices = _fit_route(selected, data.texts, null_y, route_texts)
        route_hash = hashlib.sha256(null_choices.tobytes()).hexdigest()
        if route_hash not in null_hashes:
            null_index = len(null_seeds)
            null_rows.extend(
                _route_rows(
                    confirmation,
                    null_choices,
                    selected,
                    null_index=null_index,
                )
            )
            null_hashes.add(route_hash)
            null_seeds.append(seed)
        seed += 1
    if len(null_seeds) != NULL_COUNT:
        raise ValueError("could not freeze 128 unique family-null routes")

    routes_path = output / "confirmation-routes.jsonl"
    blind_path = output / "confirmation-blind-routes.jsonl"
    null_path = output / "confirmation-null-routes.jsonl"
    _write_rows(routes_path, real_rows)
    _write_rows(blind_path, blind_rows)
    _write_rows(null_path, null_rows)
    selection_lock = {
        "protocol": PROTOCOL,
        "valid": True,
        "selected_candidate": selected.key,
        "selected_pair": [ARMS[selected.cheap], ARMS[selected.expensive]],
        "development_static_baseline": selection["development_static_baseline"],
        "selected_config": {
            "family": selected.family,
            "dim": selected.dim,
            "alpha": selected.alpha,
            "threshold": selected.threshold,
            "k": selected.k,
            "z": selected.z,
        },
        "inputs": inputs,
        "confirmation_routes_sha256": _sha256(routes_path),
        "confirmation_blind_routes_sha256": _sha256(blind_path),
        "confirmation_null_routes_sha256": _sha256(null_path),
        "null_count": NULL_COUNT,
        "null_unique_route_hashes": len(null_hashes),
        "null_seeds": null_seeds,
        "target_outcomes_used": False,
        "deep_swe_outcomes_accessed": False,
        "fitted_numeric_router_state_persisted": False,
    }
    lock_path = output / "selection-lock.json"
    lock_path.write_text(
        json.dumps(selection_lock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "protocol": PROTOCOL,
        "valid": True,
        "development_passed": True,
        "confirmation_authorized": True,
        **selection,
        "confirmation_traffic": {
            ARMS[selected.cheap]: len(choices) - expensive_count,
            ARMS[selected.expensive]: expensive_count,
        },
        "matched_blind_traffic_identical": bool(
            np.array_equal(
                np.bincount(choices, minlength=len(ARMS)),
                np.bincount(blind, minlength=len(ARMS)),
            )
        ),
        "route_latency_p95_ms": latency_p95,
        "route_batch_elapsed_ms": batch_elapsed_ms,
        "inputs": inputs,
        "selection_lock_sha256": _sha256(lock_path),
        "confirmation_routes_sha256": _sha256(routes_path),
        "confirmation_blind_routes_sha256": _sha256(blind_path),
        "confirmation_null_routes_sha256": _sha256(null_path),
        "target_outcomes_used": False,
        "deep_swe_outcomes_accessed": False,
        "fitted_numeric_router_state_persisted": False,
    }
    (output / "selection-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "selected=%s traffic=%d/%d latency_p95_ms=%.6f",
        selected.key,
        len(choices) - expensive_count,
        expensive_count,
        latency_p95,
    )


def main() -> None:
    """Parse frozen external inputs and fit on the current compute host."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fit(args.corpus, args.outcomes, args.audit, args.confirmation, args.output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
