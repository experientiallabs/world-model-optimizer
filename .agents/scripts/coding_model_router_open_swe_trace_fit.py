"""Evaluate trajectory-distilled effort routing on external Open-SWE outcomes.

The fitter joins compact text-free trajectory summaries to pinned SWE-rebench
task text. Repository-held-out nested validation compares direct lexical heads
with burden-only and concatenated trajectory-distilled features. DeepSWE data
is outside this module's contract, and no fitted model is persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

logger = logging.getLogger("coding-router-open-swe-trace-fit")

REBENCH_SHA256 = "0e0bf9355f892ad74ae98d4e1c404f39fd6654a8e351ee3e6ab162e4a64cd3ad"
WEAK_MODE = "qwen35"
STRONG_MODE = "minimax_m25"
SCAFFOLD = "openhands"
SEED = 20_260_731
RETENTION_TARGETS = (0.95, 0.97, 0.99)
BOOTSTRAP_SAMPLES = 2_000
BURDEN_COLUMNS = (
    "messages",
    "assistant_turns",
    "reasoning_characters",
    "content_characters",
    "tool_calls",
    "distinct_tools",
    "repeated_calls",
    "max_repeated_call_run",
    "shell_calls",
    "search_calls",
    "read_calls",
    "edit_calls",
    "test_calls",
    "model_patch_files",
    "model_patch_lines",
)


@dataclass(frozen=True)
class Candidate:
    """One frozen text-to-route family and regularization point."""

    family: Literal["direct", "distilled", "concat"]
    dimension: int
    burden_alpha: float
    reward_alpha: float

    @property
    def name(self) -> str:
        return (
            f"{self.family}-hash{self.dimension}-burden{self.burden_alpha:g}-"
            f"reward{self.reward_alpha:g}"
        )


CANDIDATES = tuple(
    [
        Candidate("direct", dimension, 1.0, reward_alpha)
        for dimension in (2_048, 8_192)
        for reward_alpha in (1.0, 10.0)
    ]
    + [
        Candidate(family, dimension, burden_alpha, reward_alpha)
        for family in ("distilled", "concat")
        for dimension in (2_048, 8_192)
        for burden_alpha in (1.0, 10.0)
        for reward_alpha in (1.0, 10.0)
    ]
)


@dataclass(frozen=True)
class Data:
    """Dense task labels plus sparse pre-inference text features."""

    task_ids: list[str]
    repos: np.ndarray
    texts: list[str]
    cheap: np.ndarray
    strong: np.ndarray
    cheap_attempts: np.ndarray
    strong_attempts: np.ndarray
    burden: np.ndarray
    attempt_rows: dict[str, dict[str, list[tuple[str, float]]]]
    overlap_audit: dict[str, object]

    @property
    def uplift(self) -> np.ndarray:
        return self.strong - self.cheap


@dataclass(frozen=True)
class RoutePoint:
    """One quality-constrained score threshold represented by traffic rank."""

    target: float
    traffic: float
    reward: float
    retention: float
    strong_count: int


@dataclass(frozen=True)
class BurdenFold:
    """Cross-fitted train features and held-out features for one inner fold."""

    train: np.ndarray
    test: np.ndarray
    train_burden: np.ndarray
    test_burden: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo(value: str) -> str:
    normalized = value.casefold().strip()
    normalized = re.sub(r"^https?://github\.com/", "", normalized)
    return normalized.removesuffix(".git").strip("/") or "unknown"


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = _rank(left)
    right_rank = _rank(right)
    if float(np.std(left_rank)) == 0.0 or float(np.std(right_rank)) == 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _quantiles(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.median(array)), float(np.quantile(array, 0.75) - np.quantile(array, 0.25))


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected a numeric report value")
    return float(value)


def _burden_signature(partitions: dict[tuple[str, str], list[list[float]]]) -> list[float]:
    """Form an agent-neutral task signature from partition medians and IQRs."""
    if len(partitions) < 2:
        raise ValueError("burden signature needs at least two agent partitions")
    medians: list[list[float]] = []
    iqrs: list[list[float]] = []
    for rows in partitions.values():
        if not rows:
            continue
        transformed = np.log1p(np.asarray(rows, dtype=np.float64))
        medians.append(np.median(transformed, axis=0).tolist())
        iqrs.append(
            (
                np.quantile(transformed, 0.75, axis=0) - np.quantile(transformed, 0.25, axis=0)
            ).tolist()
        )
    if len(medians) < 2:
        raise ValueError("burden signature has fewer than two populated partitions")
    median_array = np.asarray(medians, dtype=np.float64)
    iqr_array = np.asarray(iqrs, dtype=np.float64)
    return np.concatenate(
        [
            np.median(median_array, axis=0),
            np.quantile(median_array, 0.75, axis=0) - np.quantile(median_array, 0.25, axis=0),
            np.median(iqr_array, axis=0),
        ]
    ).tolist()


def _load_text(rebench: Path) -> dict[str, tuple[str, str, str]]:
    if _sha256(rebench) != REBENCH_SHA256:
        raise ValueError("SWE-rebench source hash mismatch")
    table = pq.read_table(
        rebench,
        columns=["instance_id", "repo", "language", "problem_statement"],
    )
    result: dict[str, tuple[str, str, str]] = {}
    for row in table.to_pylist():
        task_id = str(row["instance_id"])
        text = str(row["problem_statement"] or "").strip()
        if not task_id or not text:
            continue
        value = (str(row["repo"]), str(row["language"]), text)
        prior = result.setdefault(task_id, value)
        if prior != value:
            raise ValueError(f"SWE-rebench task identity collision: {task_id}")
    return result


def _load_target_metadata(path: Path) -> tuple[set[str], set[str], dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("target metadata must be an object")
    if payload.get("target_cost_fields_accessed") is not False:
        raise ValueError("target metadata accessed cost fields")
    if payload.get("target_reward_fields_accessed") is not False:
        raise ValueError("target metadata accessed reward fields")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("target metadata has no rows")
    ids: set[str] = set()
    texts: set[str] = set()
    for row_value in rows:
        if not isinstance(row_value, dict) or set(row_value) != {"id", "repository", "text"}:
            raise ValueError("target metadata row is not label-free")
        task_id = row_value.get("id")
        text = row_value.get("text")
        repository = row_value.get("repository")
        if not all(isinstance(value, str) and value for value in (task_id, text, repository)):
            raise ValueError("target metadata row has invalid identity")
        ids.add(str(task_id).casefold())
        texts.add(_normalize_text(str(text)))
    return (
        ids,
        texts,
        {
            "target_rows": len(rows),
            "target_task_ids": len(ids),
            "target_normalized_texts": len(texts),
            "target_metadata_sha256": _sha256(path),
            "target_cost_fields_accessed": False,
            "target_reward_fields_accessed": False,
        },
    )


def _load_data(summary_dir: Path, rebench: Path, target_metadata: Path) -> Data:
    text = _load_text(rebench)
    target_ids, target_texts, overlap_audit = _load_target_metadata(target_metadata)
    columns = [
        "instance_id",
        "repo",
        "language",
        "scaffold",
        "model_mode",
        "trajectory_id",
        "resolved",
        *BURDEN_COLUMNS,
    ]
    table = ds.dataset(summary_dir, format="parquet").to_table(columns=columns)
    partitions: dict[str, dict[tuple[str, str], list[list[float]]]] = {}
    attempts: dict[str, dict[str, list[tuple[str, float]]]] = {}
    identities: dict[str, tuple[str, str]] = {}
    seen_trajectories: set[str] = set()
    for row in table.to_pylist():
        task_id = str(row["instance_id"])
        repo = _repo(str(row["repo"]))
        language = str(row["language"])
        trajectory_id = str(row["trajectory_id"])
        if trajectory_id in seen_trajectories:
            raise ValueError(f"duplicate trajectory id: {trajectory_id}")
        seen_trajectories.add(trajectory_id)
        identity = (repo, language)
        prior = identities.setdefault(task_id, identity)
        if prior != identity:
            raise ValueError(f"trajectory task identity collision: {task_id}")
        scaffold = str(row["scaffold"])
        mode = str(row["model_mode"])
        values = [float(row[column]) for column in BURDEN_COLUMNS]
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError(f"invalid burden metric for task {task_id}")
        partitions.setdefault(task_id, {}).setdefault((scaffold, mode), []).append(values)
        if scaffold == SCAFFOLD and mode in {WEAK_MODE, STRONG_MODE}:
            reward = float(row["resolved"])
            if reward not in {0.0, 1.0}:
                raise ValueError(f"nonbinary source reward for task {task_id}")
            attempts.setdefault(task_id, {}).setdefault(mode, []).append((trajectory_id, reward))
    paired = sorted(
        task_id
        for task_id, arm_rows in attempts.items()
        if task_id in text
        and arm_rows.get(WEAK_MODE)
        and arm_rows.get(STRONG_MODE)
        and len(partitions.get(task_id, {})) >= 2
    )
    id_overlap = [task_id for task_id in paired if task_id.casefold() in target_ids]
    text_overlap = [
        task_id for task_id in paired if _normalize_text(text[task_id][2]) in target_texts
    ]
    removed = set(id_overlap) | set(text_overlap)
    retained = [task_id for task_id in paired if task_id not in removed]
    overlap_audit.update(
        {
            "source_paired_before_decontamination": len(paired),
            "exact_task_id_overlap": len(id_overlap),
            "normalized_text_overlap": len(text_overlap),
            "removed_tasks": len(removed),
            "source_tasks_after_decontamination": len(retained),
        }
    )
    if len(retained) < 10_000:
        raise ValueError(f"paired trajectory cohort is unexpectedly small: {len(retained)}")
    cheap = np.asarray(
        [np.mean([reward for _, reward in attempts[task_id][WEAK_MODE]]) for task_id in retained],
        dtype=np.float64,
    )
    strong = np.asarray(
        [np.mean([reward for _, reward in attempts[task_id][STRONG_MODE]]) for task_id in retained],
        dtype=np.float64,
    )
    burden = np.asarray(
        [_burden_signature(partitions[task_id]) for task_id in retained],
        dtype=np.float64,
    )
    return Data(
        task_ids=retained,
        repos=np.asarray([_repo(text[task_id][0]) for task_id in retained], dtype=object),
        texts=[text[task_id][2] for task_id in retained],
        cheap=cheap,
        strong=strong,
        cheap_attempts=np.asarray(
            [len(attempts[task_id][WEAK_MODE]) for task_id in retained],
            dtype=np.float64,
        ),
        strong_attempts=np.asarray(
            [len(attempts[task_id][STRONG_MODE]) for task_id in retained],
            dtype=np.float64,
        ),
        burden=burden,
        attempt_rows={task_id: attempts[task_id] for task_id in retained},
        overlap_audit=overlap_audit,
    )


def _hash_features(texts: list[str], dimension: int) -> sparse.csr_matrix:
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        n_features=dimension,
        alternate_sign=True,
        norm="l2",
    )
    return cast(sparse.csr_matrix, vectorizer.transform(texts))


def _group_splits(
    indices: np.ndarray, groups: np.ndarray, folds: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = len(set(cast(list[str], groups[indices].tolist())))
    if unique < 2:
        raise ValueError("grouped fit needs at least two repositories")
    splitter = GroupKFold(n_splits=min(folds, unique))
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for local_train, local_test in splitter.split(indices, groups=groups[indices]):
        train = indices[local_train]
        test = indices[local_test]
        if set(groups[train]) & set(groups[test]):
            raise ValueError("repository overlap in grouped split")
        result.append((train, test))
    return result


def _fit_burden(
    features: sparse.csr_matrix,
    burden: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    model = Ridge(alpha=alpha, solver="lsqr", max_iter=300, tol=1e-3)
    model.fit(features[train], burden[train])
    return np.asarray(model.predict(features[test]), dtype=np.float64)


def _crossfit_burden(
    features: sparse.csr_matrix,
    burden: np.ndarray,
    indices: np.ndarray,
    groups: np.ndarray,
    *,
    alpha: float,
    folds: int,
) -> np.ndarray:
    predictions = np.zeros((len(indices), burden.shape[1]), dtype=np.float64)
    positions = {int(index): position for position, index in enumerate(indices)}
    for train, test in _group_splits(indices, groups, folds):
        predicted = _fit_burden(features, burden, train, test, alpha=alpha)
        local = [positions[int(index)] for index in test]
        predictions[local] = predicted
    return predictions


def _burden_folds(
    data: Data,
    direct: sparse.csr_matrix,
    indices: np.ndarray,
    *,
    alpha: float,
) -> list[BurdenFold]:
    result: list[BurdenFold] = []
    for train, test in _group_splits(indices, data.repos, 5):
        result.append(
            BurdenFold(
                train=train,
                test=test,
                train_burden=_crossfit_burden(
                    direct,
                    data.burden,
                    train,
                    data.repos,
                    alpha=alpha,
                    folds=4,
                ),
                test_burden=_fit_burden(
                    direct,
                    data.burden,
                    train,
                    test,
                    alpha=alpha,
                ),
            )
        )
    return result


def _combine_features(
    direct: sparse.csr_matrix,
    burden: np.ndarray,
    family: Literal["direct", "distilled", "concat"],
    *,
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    if family == "direct":
        return direct, np.zeros(burden.shape[1]), np.ones(burden.shape[1])
    if mean is None:
        mean = np.mean(burden, axis=0)
    if scale is None:
        scale = np.std(burden, axis=0)
    safe_scale = np.where(scale > 1e-8, scale, 1.0)
    dense = sparse.csr_matrix((burden - mean) / safe_scale)
    if family == "distilled":
        return dense, mean, safe_scale
    return sparse.hstack([direct, dense], format="csr"), mean, safe_scale


def _fit_reward_heads(
    features: sparse.csr_matrix,
    cheap: np.ndarray,
    strong: np.ndarray,
    cheap_attempts: np.ndarray,
    strong_attempts: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    cheap_head = Ridge(alpha=alpha, solver="lsqr", max_iter=300, tol=1e-3)
    strong_head = Ridge(alpha=alpha, solver="lsqr", max_iter=300, tol=1e-3)
    cheap_head.fit(features[train], cheap[train], sample_weight=cheap_attempts[train])
    strong_head.fit(features[train], strong[train], sample_weight=strong_attempts[train])
    return np.asarray(
        strong_head.predict(features[test]) - cheap_head.predict(features[test]),
        dtype=np.float64,
    )


def _candidate_oof(
    data: Data,
    direct: sparse.csr_matrix,
    indices: np.ndarray,
    candidate: Candidate,
    *,
    cheap: np.ndarray | None = None,
    strong: np.ndarray | None = None,
    burden_folds: list[BurdenFold] | None = None,
) -> np.ndarray:
    cheap_values = data.cheap if cheap is None else cheap
    strong_values = data.strong if strong is None else strong
    predictions = np.zeros(len(indices), dtype=np.float64)
    positions = {int(index): position for position, index in enumerate(indices)}
    grouped = _group_splits(indices, data.repos, 5)
    if candidate.family != "direct":
        if burden_folds is None:
            burden_folds = _burden_folds(
                data,
                direct,
                indices,
                alpha=candidate.burden_alpha,
            )
        if len(burden_folds) != len(grouped):
            raise ValueError("burden fold cache does not match grouped splits")
    for fold_index, (train, test) in enumerate(grouped):
        if candidate.family == "direct":
            train_features = direct
            test_features = direct
        else:
            assert burden_folds is not None
            cached = burden_folds[fold_index]
            if not np.array_equal(cached.train, train) or not np.array_equal(cached.test, test):
                raise ValueError("burden fold cache identity mismatch")
            local_direct_train = direct[train]
            local_direct_test = direct[test]
            train_features, mean, scale = _combine_features(
                local_direct_train,
                cached.train_burden,
                candidate.family,
            )
            test_features, _, _ = _combine_features(
                local_direct_test,
                cached.test_burden,
                candidate.family,
                mean=mean,
                scale=scale,
            )
        if candidate.family == "direct":
            fold_scores = _fit_reward_heads(
                train_features,
                cheap_values,
                strong_values,
                data.cheap_attempts,
                data.strong_attempts,
                train,
                test,
                alpha=candidate.reward_alpha,
            )
        else:
            local_train = np.arange(len(train), dtype=np.int64)
            local_test = np.arange(len(test), dtype=np.int64)
            fold_scores = _fit_reward_heads(
                sparse.vstack([train_features, test_features], format="csr"),
                np.concatenate([cheap_values[train], cheap_values[test]]),
                np.concatenate([strong_values[train], strong_values[test]]),
                np.concatenate([data.cheap_attempts[train], data.cheap_attempts[test]]),
                np.concatenate([data.strong_attempts[train], data.strong_attempts[test]]),
                local_train,
                len(train) + local_test,
                alpha=candidate.reward_alpha,
            )
        predictions[[positions[int(index)] for index in test]] = fold_scores
    return predictions


def _fit_candidate(
    data: Data,
    direct: sparse.csr_matrix,
    train: np.ndarray,
    test: np.ndarray,
    candidate: Candidate,
    *,
    cheap: np.ndarray | None = None,
    strong: np.ndarray | None = None,
) -> np.ndarray:
    cheap_values = data.cheap if cheap is None else cheap
    strong_values = data.strong if strong is None else strong
    if candidate.family == "direct":
        return _fit_reward_heads(
            direct,
            cheap_values,
            strong_values,
            data.cheap_attempts,
            data.strong_attempts,
            train,
            test,
            alpha=candidate.reward_alpha,
        )
    train_burden = _crossfit_burden(
        direct,
        data.burden,
        train,
        data.repos,
        alpha=candidate.burden_alpha,
        folds=5,
    )
    test_burden = _fit_burden(
        direct,
        data.burden,
        train,
        test,
        alpha=candidate.burden_alpha,
    )
    train_features, mean, scale = _combine_features(
        direct[train],
        train_burden,
        candidate.family,
    )
    test_features, _, _ = _combine_features(
        direct[test],
        test_burden,
        candidate.family,
        mean=mean,
        scale=scale,
    )
    return _fit_reward_heads(
        sparse.vstack([train_features, test_features], format="csr"),
        np.concatenate([cheap_values[train], cheap_values[test]]),
        np.concatenate([strong_values[train], strong_values[test]]),
        np.concatenate([data.cheap_attempts[train], data.cheap_attempts[test]]),
        np.concatenate([data.strong_attempts[train], data.strong_attempts[test]]),
        np.arange(len(train), dtype=np.int64),
        len(train) + np.arange(len(test), dtype=np.int64),
        alpha=candidate.reward_alpha,
    )


def _select_point(
    scores: np.ndarray, cheap: np.ndarray, strong: np.ndarray, target: float
) -> RoutePoint:
    if not 0.0 < target <= 1.0 or len(scores) == 0:
        raise ValueError("invalid route point inputs")
    order = np.argsort(-scores, kind="mergesort")
    baseline = float(np.sum(cheap))
    required = target * float(np.mean(strong))
    cumulative = np.concatenate([[0.0], np.cumsum((strong - cheap)[order])])
    rewards = (baseline + cumulative) / len(scores)
    feasible = np.flatnonzero(rewards >= required - 1e-12)
    count = int(feasible[0]) if len(feasible) else len(scores)
    reward = float(rewards[count])
    strong_mean = float(np.mean(strong))
    retention = reward / strong_mean if strong_mean > 0.0 else 1.0
    return RoutePoint(target, count / len(scores), reward, retention, count)


def _route(
    scores: np.ndarray, cheap: np.ndarray, strong: np.ndarray, traffic: float
) -> tuple[np.ndarray, np.ndarray]:
    count = min(len(scores), max(0, int(round(traffic * len(scores)))))
    selected = np.zeros(len(scores), dtype=bool)
    selected[np.argsort(-scores, kind="mergesort")[:count]] = True
    return np.where(selected, strong, cheap), selected


def _shuffle_pairs(data: Data, indices: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    cheap = data.cheap.copy()
    strong = data.strong.copy()
    rng = np.random.default_rng(seed)
    for repo in sorted(set(cast(list[str], data.repos[indices].tolist()))):
        members = indices[data.repos[indices] == repo]
        permutation = rng.permutation(members)
        cheap[members] = data.cheap[permutation]
        strong[members] = data.strong[permutation]
    return cheap, strong


def _bootstrap_difference(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
) -> list[float]:
    repositories = sorted(set(cast(list[str], groups.tolist())))
    by_repo = {repo: np.flatnonzero(groups == repo) for repo in repositories}
    rng = np.random.default_rng(seed)
    samples = np.zeros(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for sample in range(BOOTSTRAP_SAMPLES):
        selected = rng.choice(repositories, size=len(repositories), replace=True)
        indices = np.concatenate([by_repo[str(repo)] for repo in selected])
        samples[sample] = float(np.mean(values[indices]))
    return np.quantile(samples, [0.025, 0.5, 0.975]).astype(float).tolist()


def _heldout_attempt_oracle(data: Data) -> dict[str, object]:
    task_ids: list[str] = []
    repos: list[str] = []
    weak_fit: list[float] = []
    weak_test: list[float] = []
    strong_fit: list[float] = []
    strong_test: list[float] = []
    repo_by_task = dict(zip(data.task_ids, cast(list[str], data.repos.tolist()), strict=True))
    for task_id in data.task_ids:
        weak = sorted(data.attempt_rows[task_id][WEAK_MODE])
        strong = sorted(data.attempt_rows[task_id][STRONG_MODE])
        if len(weak) < 2 or len(strong) < 2:
            continue
        weak_fit.append(float(np.mean([reward for _, reward in weak[::2]])))
        weak_test.append(float(np.mean([reward for _, reward in weak[1::2]])))
        strong_fit.append(float(np.mean([reward for _, reward in strong[::2]])))
        strong_test.append(float(np.mean([reward for _, reward in strong[1::2]])))
        task_ids.append(task_id)
        repos.append(repo_by_task[task_id])
    if not task_ids:
        return {"tasks": 0, "available": False}
    weak_fit_array = np.asarray(weak_fit)
    weak_test_array = np.asarray(weak_test)
    strong_fit_array = np.asarray(strong_fit)
    strong_test_array = np.asarray(strong_test)
    choose_strong = strong_fit_array >= weak_fit_array
    oracle = np.where(choose_strong, strong_test_array, weak_test_array)
    static_strong = float(np.mean(strong_fit_array)) >= float(np.mean(weak_fit_array))
    static = strong_test_array if static_strong else weak_test_array
    difference = oracle - static
    return {
        "tasks": len(task_ids),
        "repositories": len(set(repos)),
        "available": True,
        "fit_selected_static": STRONG_MODE if static_strong else WEAK_MODE,
        "heldout_oracle_reward": float(np.mean(oracle)),
        "heldout_static_reward": float(np.mean(static)),
        "headroom": float(np.mean(difference)),
        "repository_bootstrap_headroom_95ci": _bootstrap_difference(
            difference,
            np.asarray(repos, dtype=object),
            seed=SEED + 91,
        ),
    }


def _candidate_points(
    data: Data,
    indices: np.ndarray,
    scores: np.ndarray,
    candidate: Candidate,
) -> list[dict[str, object]]:
    return [
        {
            "candidate": candidate.name,
            "family": candidate.family,
            "dimension": candidate.dimension,
            "burden_alpha": candidate.burden_alpha,
            "reward_alpha": candidate.reward_alpha,
            **_select_point(scores, data.cheap[indices], data.strong[indices], target).__dict__,
            "uplift_spearman": _spearman(scores, data.uplift[indices]),
        }
        for target in RETENTION_TARGETS
    ]


def _select_candidate(rows: list[dict[str, object]]) -> dict[str, object]:
    feasible = [row for row in rows if _number(row["retention"]) >= _number(row["target"])]
    pool = feasible or rows
    family_order = {"direct": 0, "distilled": 1, "concat": 2}
    return min(
        pool,
        key=lambda row: (
            float(row["traffic"]),
            -float(row["reward"]),
            family_order[str(row["family"])],
            str(row["candidate"]),
            -float(row["target"]),
        ),
    )


def fit(
    summary_dir: Path,
    rebench: Path,
    target_metadata: Path,
    output: Path,
) -> dict[str, object]:
    data = _load_data(summary_dir, rebench, target_metadata)
    all_indices = np.arange(len(data.task_ids), dtype=np.int64)
    features = {dimension: _hash_features(data.texts, dimension) for dimension in (2_048, 8_192)}
    router_scores = np.zeros(len(data.task_ids), dtype=np.float64)
    router_reward = np.zeros(len(data.task_ids), dtype=np.float64)
    router_strong = np.zeros(len(data.task_ids), dtype=bool)
    blind_reward = np.zeros(len(data.task_ids), dtype=np.float64)
    direct_reward = np.zeros(len(data.task_ids), dtype=np.float64)
    direct_strong = np.zeros(len(data.task_ids), dtype=bool)
    shuffled_scores = np.zeros(len(data.task_ids), dtype=np.float64)
    shuffled_reward = np.zeros(len(data.task_ids), dtype=np.float64)
    outer_rows: list[dict[str, object]] = []
    for outer_fold, (train, test) in enumerate(_group_splits(all_indices, data.repos, 5)):
        candidate_rows: list[dict[str, object]] = []
        burden_cache = {
            (dimension, burden_alpha): _burden_folds(
                data,
                features[dimension],
                train,
                alpha=burden_alpha,
            )
            for dimension in (2_048, 8_192)
            for burden_alpha in (1.0, 10.0)
        }
        for candidate in CANDIDATES:
            scores = _candidate_oof(
                data,
                features[candidate.dimension],
                train,
                candidate,
                burden_folds=(
                    None
                    if candidate.family == "direct"
                    else burden_cache[(candidate.dimension, candidate.burden_alpha)]
                ),
            )
            candidate_rows.extend(_candidate_points(data, train, scores, candidate))
        selected = _select_candidate(candidate_rows)
        direct_selected = _select_candidate(
            [row for row in candidate_rows if row["family"] == "direct"]
        )
        candidate = next(item for item in CANDIDATES if item.name == selected["candidate"])
        direct_candidate = next(
            item for item in CANDIDATES if item.name == direct_selected["candidate"]
        )
        test_scores = _fit_candidate(
            data,
            features[candidate.dimension],
            train,
            test,
            candidate,
        )
        test_reward, test_strong = _route(
            test_scores,
            data.cheap[test],
            data.strong[test],
            _number(selected["traffic"]),
        )
        router_scores[test] = test_scores
        router_reward[test] = test_reward
        router_strong[test] = test_strong
        blind_reward[test] = data.cheap[test] + float(np.mean(test_strong)) * data.uplift[test]
        direct_scores = _fit_candidate(
            data,
            features[direct_candidate.dimension],
            train,
            test,
            direct_candidate,
        )
        direct_fold_reward, direct_fold_strong = _route(
            direct_scores,
            data.cheap[test],
            data.strong[test],
            _number(direct_selected["traffic"]),
        )
        direct_reward[test] = direct_fold_reward
        direct_strong[test] = direct_fold_strong
        shuffled_cheap, shuffled_strong = _shuffle_pairs(data, train, seed=SEED + outer_fold)
        shuffled_fold_scores = _fit_candidate(
            data,
            features[candidate.dimension],
            train,
            test,
            candidate,
            cheap=shuffled_cheap,
            strong=shuffled_strong,
        )
        shuffled_fold_reward, _ = _route(
            shuffled_fold_scores,
            data.cheap[test],
            data.strong[test],
            _number(selected["traffic"]),
        )
        shuffled_scores[test] = shuffled_fold_scores
        shuffled_reward[test] = shuffled_fold_reward
        strong_mean = float(np.mean(data.strong[test]))
        retention = float(np.mean(test_reward)) / strong_mean if strong_mean > 0.0 else 1.0
        outer_rows.append(
            {
                "fold": outer_fold,
                "train_tasks": len(train),
                "test_tasks": len(test),
                "train_repositories": len(set(data.repos[train])),
                "test_repositories": len(set(data.repos[test])),
                "repository_overlap": 0,
                "selected": selected,
                "selected_direct": direct_selected,
                "test_reward": float(np.mean(test_reward)),
                "test_strong_reward": strong_mean,
                "test_retention": retention,
                "test_traffic": float(np.mean(test_strong)),
                "candidate_points": candidate_rows,
            }
        )
    advantage = router_reward - blind_reward
    shuffled_advantage = shuffled_reward - blind_reward
    router_reward_mean = float(np.mean(router_reward))
    direct_reward_mean = float(np.mean(direct_reward))
    router_traffic = float(np.mean(router_strong))
    direct_traffic = float(np.mean(direct_strong))
    advantage_interval = _bootstrap_difference(advantage, data.repos, seed=SEED)
    shuffled_interval = _bootstrap_difference(
        shuffled_advantage,
        data.repos,
        seed=SEED + 1,
    )
    gate = {
        "matched_blind_bootstrap_lower_positive": advantage_interval[0] > 0.0,
        "uplift_spearman_positive": _spearman(router_scores, data.uplift) > 0.0,
        "every_outer_fold_retention_at_least_0_95": all(
            _number(row["test_retention"]) >= 0.95 for row in outer_rows
        ),
        "strong_traffic_at_least_20_percent_below_static": router_traffic <= 0.80,
        "shuffled_control_fails": not (
            shuffled_interval[0] > 0.0 and _spearman(shuffled_scores, data.uplift) > 0.0
        ),
        "trajectory_family_selected_every_outer_fold": all(
            str(cast(dict[str, object], row["selected"])["family"]) != "direct"
            for row in outer_rows
        ),
        "not_dominated_by_direct_hash": not (
            router_reward_mean < direct_reward_mean and router_traffic >= direct_traffic
        ),
        "complete_and_target_sealed": bool(np.all(np.isfinite(router_reward))),
    }
    gate["passed"] = all(bool(value) for value in gate.values())
    report = {
        "schema": "open-swe-trajectory-distillation-nested-v1",
        "tasks": len(data.task_ids),
        "repositories": len(set(cast(list[str], data.repos.tolist()))),
        "burden_features": data.burden.shape[1],
        "candidate_count": len(CANDIDATES),
        "weak_mode": WEAK_MODE,
        "strong_mode": STRONG_MODE,
        "weak_reward": float(np.mean(data.cheap)),
        "strong_reward": float(np.mean(data.strong)),
        "router_reward": router_reward_mean,
        "router_traffic": router_traffic,
        "matched_blind_reward": float(np.mean(blind_reward)),
        "matched_blind_advantage": float(np.mean(advantage)),
        "matched_blind_advantage_95ci": advantage_interval,
        "uplift_spearman": _spearman(router_scores, data.uplift),
        "direct_hash_reward": direct_reward_mean,
        "direct_hash_traffic": direct_traffic,
        "shuffled_reward": float(np.mean(shuffled_reward)),
        "shuffled_advantage_95ci": shuffled_interval,
        "shuffled_uplift_spearman": _spearman(shuffled_scores, data.uplift),
        "heldout_attempt_oracle": _heldout_attempt_oracle(data),
        "target_overlap_audit": data.overlap_audit,
        "outer_folds": outer_rows,
        "external_gate": gate,
        "target_outcomes_used": False,
        "target_embeddings_used": False,
        "no_persisted_fitted_model": True,
        "inputs": {
            "rebench_sha256": _sha256(rebench),
            "target_metadata_sha256": _sha256(target_metadata),
            "summary_files": len(list(summary_dir.glob("*.parquet"))),
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / "outer-scores.jsonl").open("w", encoding="utf-8") as handle:
        for index, task_id in enumerate(data.task_ids):
            handle.write(
                json.dumps(
                    {
                        "task_id": task_id,
                        "repo": str(data.repos[index]),
                        "score": float(router_scores[index]),
                        "uplift": float(data.uplift[index]),
                        "strong": bool(router_strong[index]),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    logger.info(
        "trajectory distillation complete tasks=%d reward=%.6f traffic=%.4f "
        "advantage=%.6f ci_low=%.6f gate=%s",
        len(data.task_ids),
        router_reward_mean,
        router_traffic,
        float(np.mean(advantage)),
        advantage_interval[0],
        gate["passed"],
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", type=Path, required=True)
    parser.add_argument("--rebench", type=Path, required=True)
    parser.add_argument("--target-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    fit(
        args.summaries.resolve(),
        args.rebench.resolve(),
        args.target_metadata.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
