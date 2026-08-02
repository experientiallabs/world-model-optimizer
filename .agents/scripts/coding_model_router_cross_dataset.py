"""Train a task-hardness router on Nebius traces and evaluate it once on DeepSWE.

The transfer contract is intentionally strict:

* Training labels come only from ``nebius/SWE-agent-trajectories``.
* DeepSWE rewards and costs are loaded only after the hardness model, neighborhood size,
  novelty floor, and effort-bin thresholds are frozen.
* The primary action space is reasoning effort within one model family. Model-family
  switching is reported only as a secondary sensitivity analysis.

This is a one-off research runner, not a public WMO command. It writes durable artifacts under
the requested experiment root and never changes either source dataset.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import logging
import os
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict, cast

import fsspec
import numpy as np
import openai
import pyarrow.parquet as pq

from wmo.core.files import write_text_atomic

logger = logging.getLogger(__name__)

EMBED_MODEL = "text-embedding-3-large"
NEBIUS_DATASET = "nebius/SWE-agent-trajectories"
NEBIUS_COMMIT_API = "https://huggingface.co/api/datasets/nebius/SWE-agent-trajectories"
NEBIUS_TRACE_PARQUETS = tuple(
    "https://huggingface.co/datasets/nebius/SWE-agent-trajectories/"
    f"resolve/refs%2Fconvert%2Fparquet/default/train/{index:04d}.parquet"
    for index in range(12)
)
NEBIUS_TASK_PARQUETS = (
    "https://huggingface.co/datasets/nebius/SWE-bench-extra/"
    "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
    "https://huggingface.co/datasets/princeton-nlp/SWE-bench/"
    "resolve/refs%2Fconvert%2Fparquet/default/dev/0000.parquet",
)
CHEAP_EXTERNAL_MODEL = "swe-agent-llama-8b"
STRONG_EXTERNAL_MODEL = "swe-agent-llama-70b"
K_CANDIDATES = (6, 12, 20, 40)
EFFORTS = ("low", "medium", "high", "xhigh", "max")
# Easiest 30% gets low effort, then 25%, 20%, 15%, and the hardest 10% gets max.
# These masses are frozen before DeepSWE is opened.
EFFORT_EASE_QUANTILES = (0.10, 0.25, 0.45, 0.70)
NOVELTY_QUANTILE = 0.10
BOOTSTRAP_SAMPLES = 10_000
PRIMARY_FAMILY = "mini_swe_agent_claude_opus_5"


class RawTrialRow(TypedDict):
    """Columns read from one Nebius trajectory row."""

    instance_id: str
    model_name: str
    target: bool


class TaskRow(TypedDict):
    """Columns read from one SWE task-definition row."""

    instance_id: str
    repo: str
    problem_statement: str


@dataclasses.dataclass(frozen=True)
class ExternalTask:
    """One paired Nebius task used to fit the hardness signal."""

    instance_id: str
    repo: str
    text: str
    cheap_reward: float
    strong_reward: float
    cheap_attempts: int
    strong_attempts: int


@dataclasses.dataclass(frozen=True)
class DeepSweData:
    """Complete DeepSWE task, arm, reward, cost, and embedding matrix."""

    task_ids: list[str]
    groups: list[str]
    arms: list[str]
    rewards: np.ndarray
    costs: np.ndarray
    embeddings: np.ndarray


@dataclasses.dataclass(frozen=True)
class TransferPolicy:
    """Frozen external hardness model and effort mapping."""

    k: int
    floor_sim: float
    ease_thresholds: tuple[float, float, float, float]
    external_embeddings: np.ndarray
    external_strong_rewards: np.ndarray


@dataclasses.dataclass(frozen=True)
class EffortCalibration:
    """Thresholds and novelty guard used for one target evaluation."""

    mode: str
    ease_thresholds: tuple[float, float, float, float]
    novelty_floor_similarity: float
    confirmatory: bool


@dataclasses.dataclass(frozen=True)
class BinaryScalePolicy:
    """External scale-benefit policy transferred to low versus high effort."""

    k: int
    threshold: float
    external_embeddings: np.ndarray
    external_advantages: np.ndarray


def _json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return cast(dict[str, object], value)


def _as_float(value: object) -> float:
    if not isinstance(value, int | float | str):
        raise TypeError(f"expected a numeric JSON value, got {type(value).__name__}")
    return float(value)


def _as_int(value: object) -> int:
    if not isinstance(value, int | str):
        raise TypeError(f"expected an integer JSON value, got {type(value).__name__}")
    return int(value)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _load_env_file(path: Path) -> None:
    """Load one existing env file without logging or returning secret values."""
    if not path.is_file():
        raise FileNotFoundError(f"environment file does not exist: {path}")
    loaded = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)
            loaded += 1
    logger.info("loaded %d environment variable names from %s", loaded, path)


def _remote_rows(url: str, columns: list[str]) -> list[dict[str, object]]:
    """Read selected Parquet columns through HTTP range requests."""
    with fsspec.open(url, "rb").open() as handle:
        table = pq.read_table(handle, columns=columns)
    return cast(list[dict[str, object]], table.to_pylist())


def _dataset_source_manifest() -> dict[str, object]:
    """Resolve source revision and schema metadata without downloading payloads."""
    with fsspec.open(NEBIUS_COMMIT_API, "rt", encoding="utf-8").open() as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("Hugging Face dataset API did not return an object")
    card = metadata.get("cardData")
    card_data = card if isinstance(card, dict) else {}
    return {
        "dataset": NEBIUS_DATASET,
        "source_revision": metadata.get("sha"),
        "last_modified": metadata.get("lastModified"),
        "license": card_data.get("license"),
        "trace_parquet_urls": list(NEBIUS_TRACE_PARQUETS),
        "task_parquet_urls": list(NEBIUS_TASK_PARQUETS),
    }


def _repo_from_instance(instance_id: str) -> str:
    prefix = instance_id.rsplit("-", 1)[0]
    return prefix.replace("__", "/", 1)


def _collect_external_tasks(cache_path: Path) -> list[ExternalTask]:
    """Aggregate repeated paired 8B and 70B outcomes and join issue text."""
    if cache_path.is_file():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        return [ExternalTask(**row) for row in raw]

    aggregates: dict[str, dict[str, list[int]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for index, url in enumerate(NEBIUS_TRACE_PARQUETS, start=1):
        rows = _remote_rows(url, ["instance_id", "model_name", "target"])
        for untyped in rows:
            row = cast(RawTrialRow, untyped)
            if row["model_name"] in (CHEAP_EXTERNAL_MODEL, STRONG_EXTERNAL_MODEL):
                aggregates[row["instance_id"]][row["model_name"]].append(int(row["target"]))
        logger.info("read Nebius label shard %d/%d", index, len(NEBIUS_TRACE_PARQUETS))

    paired = {
        instance_id: by_model
        for instance_id, by_model in aggregates.items()
        if CHEAP_EXTERNAL_MODEL in by_model and STRONG_EXTERNAL_MODEL in by_model
    }
    texts: dict[str, TaskRow] = {}
    for url in NEBIUS_TASK_PARQUETS:
        for untyped in _remote_rows(url, ["instance_id", "repo", "problem_statement"]):
            row = cast(TaskRow, untyped)
            if row["instance_id"] in paired:
                texts[row["instance_id"]] = row
    missing = sorted(set(paired) - set(texts))
    if missing:
        _write_json(
            cache_path.with_name("nebius-dropped-missing-task-text.json"),
            {
                "reason": "paired trajectory labels exist, but neither public task corpus "
                "contains the issue definition",
                "task_ids": missing,
            },
        )
        logger.warning(
            "dropping %d paired Nebius tasks without public task text: %s",
            len(missing),
            missing[:5],
        )

    tasks: list[ExternalTask] = []
    for instance_id in sorted(set(paired) - set(missing)):
        by_model = paired[instance_id]
        cheap = by_model[CHEAP_EXTERNAL_MODEL]
        strong = by_model[STRONG_EXTERNAL_MODEL]
        task_row = texts[instance_id]
        tasks.append(
            ExternalTask(
                instance_id=instance_id,
                repo=task_row["repo"] or _repo_from_instance(instance_id),
                text=task_row["problem_statement"],
                cheap_reward=statistics.fmean(cheap),
                strong_reward=statistics.fmean(strong),
                cheap_attempts=len(cheap),
                strong_attempts=len(strong),
            )
        )
    _write_json(cache_path, [dataclasses.asdict(task) for task in tasks])
    return tasks


def _embedding_cache(path: Path) -> dict[str, list[float]]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain an embedding object")
    return {
        str(key): [_as_float(number) for number in cast(list[object], vector)]
        for key, vector in value.items()
    }


def _normalized(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > 0.0, norms, 1.0)


def _embed_external(
    tasks: list[ExternalTask],
    *,
    cache_path: Path,
) -> tuple[np.ndarray, int]:
    """Embed missing Nebius task texts, preserving exact embedding-token usage."""
    cache = _embedding_cache(cache_path)
    missing = [task for task in tasks if task.instance_id not in cache]
    prompt_tokens = 0
    if missing:
        client = openai.OpenAI()
        for start in range(0, len(missing), 64):
            batch = missing[start : start + 64]
            response = client.embeddings.create(
                model=EMBED_MODEL,
                input=[task.text[:8000] for task in batch],
            )
            if response.usage is not None:
                prompt_tokens += int(response.usage.prompt_tokens)
            for task, item in zip(batch, response.data, strict=True):
                cache[task.instance_id] = [float(value) for value in item.embedding]
            _write_json(cache_path, cache)
            logger.info(
                "embedded %d/%d missing Nebius tasks",
                min(start + len(batch), len(missing)),
                len(missing),
            )
    matrix = np.asarray([cache[task.instance_id] for task in tasks], dtype=np.float32)
    return _normalized(matrix), prompt_tokens


def _weighted_knn(
    bank: np.ndarray,
    rewards: np.ndarray,
    queries: np.ndarray,
    *,
    k: int,
    exclude_rows: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict rewards and return each query's nearest similarity."""
    predictions = np.empty(queries.shape[0], dtype=np.float64)
    nearest = np.empty(queries.shape[0], dtype=np.float64)
    for query_index, query in enumerate(queries):
        similarities = bank @ query
        if exclude_rows is not None:
            similarities[exclude_rows[query_index]] = -np.inf
        count = min(k, int(np.isfinite(similarities).sum()))
        if count < 1:
            raise ValueError("a kNN query has no eligible bank rows")
        rows = np.argpartition(-similarities, count - 1)[:count]
        rows = rows[np.argsort(-similarities[rows])]
        weights = np.clip(similarities[rows], 0.0, None) + 1e-6
        predictions[query_index] = float(np.sum(rewards[rows] * weights) / np.sum(weights))
        nearest[query_index] = float(similarities[rows[0]])
    return predictions, nearest


def _rank(values: np.ndarray) -> np.ndarray:
    """Return stable average ranks for Spearman correlation."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2:
        return 0.0
    left_rank = _rank(left)
    right_rank = _rank(right)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    return float(left_centered @ right_centered / denominator) if denominator else 0.0


def _fold_assignments(groups: list[str], seed: int, folds: int = 5) -> np.ndarray:
    """Assign whole repositories to deterministic, approximately balanced folds."""
    unique = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    group_fold = {group: index % folds for index, group in enumerate(unique)}
    return np.asarray([group_fold[group] for group in groups], dtype=np.int64)


def _external_oof_predictions(
    embeddings: np.ndarray,
    rewards: np.ndarray,
    groups: list[str],
    *,
    k: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    folds = _fold_assignments(groups, seed)
    predictions = np.empty(rewards.size, dtype=np.float64)
    nearest = np.empty(rewards.size, dtype=np.float64)
    for fold in range(5):
        test = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        pred, near = _weighted_knn(
            embeddings[train],
            rewards[train],
            embeddings[test],
            k=k,
        )
        predictions[test] = pred
        nearest[test] = near
    return predictions, nearest


def _fit_transfer_policy(
    tasks: list[ExternalTask],
    embeddings: np.ndarray,
) -> tuple[TransferPolicy, dict[str, object]]:
    rewards = np.asarray([task.strong_reward for task in tasks], dtype=np.float64)
    groups = [task.repo for task in tasks]
    candidates: list[dict[str, object]] = []
    for k in K_CANDIDATES:
        correlations = []
        for seed in (11, 23, 37, 41, 59):
            predicted, _ = _external_oof_predictions(
                embeddings,
                rewards,
                groups,
                k=k,
                seed=seed,
            )
            correlations.append(_spearman(predicted, rewards))
        candidates.append(
            {
                "k": k,
                "mean_repo_grouped_spearman": statistics.fmean(correlations),
                "seed_spearman": correlations,
            }
        )
    chosen = max(candidates, key=lambda row: float(row["mean_repo_grouped_spearman"]))
    k = _as_int(chosen["k"])
    oof, nearest = _external_oof_predictions(
        embeddings,
        rewards,
        groups,
        k=k,
        seed=37,
    )
    thresholds = tuple(float(np.quantile(oof, quantile)) for quantile in EFFORT_EASE_QUANTILES)

    # Self-nearest distribution, with every row excluded from its own query.
    excluded = np.arange(embeddings.shape[0], dtype=np.int64)[:, None]
    _, self_nearest = _weighted_knn(
        embeddings,
        rewards,
        embeddings,
        k=1,
        exclude_rows=excluded,
    )
    floor_sim = float(np.quantile(self_nearest, NOVELTY_QUANTILE))
    policy = TransferPolicy(
        k=k,
        floor_sim=floor_sim,
        ease_thresholds=cast(tuple[float, float, float, float], thresholds),
        external_embeddings=embeddings,
        external_strong_rewards=rewards,
    )
    report: dict[str, object] = {
        "training_dataset": NEBIUS_DATASET,
        "paired_tasks": len(tasks),
        "repositories": len(set(groups)),
        "cheap_external_model": CHEAP_EXTERNAL_MODEL,
        "strong_external_model": STRONG_EXTERNAL_MODEL,
        "cheap_mean_reward": statistics.fmean(task.cheap_reward for task in tasks),
        "strong_mean_reward": statistics.fmean(task.strong_reward for task in tasks),
        "k_search": candidates,
        "selected_k": k,
        "selected_seed_oof_spearman": _spearman(oof, rewards),
        "novelty_quantile": NOVELTY_QUANTILE,
        "novelty_floor_similarity": floor_sim,
        "effort_order": list(EFFORTS),
        "effort_ease_quantiles": list(EFFORT_EASE_QUANTILES),
        "effort_ease_thresholds": list(thresholds),
        "target_labels_consulted": False,
    }
    return policy, report


def _select_binary_threshold(
    predicted_advantage: np.ndarray,
    cheap_reward: np.ndarray,
    strong_reward: np.ndarray,
    *,
    retention_floor: float,
) -> tuple[float, float, float]:
    """Minimize strong traffic while retaining external strong-model quality."""
    strong_mean = float(strong_reward.mean())
    if strong_mean <= 0.0:
        raise ValueError("strong external reward must be positive")
    candidates: list[tuple[float, float, float]] = []
    for threshold in sorted(set(float(value) for value in predicted_advantage)):
        use_strong = predicted_advantage >= threshold
        reward = float(np.where(use_strong, strong_reward, cheap_reward).mean())
        retention = reward / strong_mean
        if retention >= retention_floor:
            candidates.append((float(use_strong.mean()), threshold, retention))
    if not candidates:
        raise ValueError("no binary external threshold meets the retention floor")
    strong_share, threshold, retention = min(candidates)
    return threshold, strong_share, retention


def _fit_binary_scale_policy(
    tasks: list[ExternalTask],
    embeddings: np.ndarray,
) -> tuple[BinaryScalePolicy, dict[str, object]]:
    """Fit an external model-scale decision as a proxy for reasoning effort."""
    cheap = np.asarray([task.cheap_reward for task in tasks], dtype=np.float64)
    strong = np.asarray([task.strong_reward for task in tasks], dtype=np.float64)
    advantages = strong - cheap
    groups = [task.repo for task in tasks]
    candidates: list[dict[str, object]] = []
    for k in K_CANDIDATES:
        correlations = []
        for seed in (11, 23, 37, 41, 59):
            predicted, _ = _external_oof_predictions(
                embeddings,
                advantages,
                groups,
                k=k,
                seed=seed,
            )
            correlations.append(_spearman(predicted, advantages))
        candidates.append(
            {
                "k": k,
                "mean_repo_grouped_spearman": statistics.fmean(correlations),
                "seed_spearman": correlations,
            }
        )
    chosen = max(candidates, key=lambda row: float(row["mean_repo_grouped_spearman"]))
    k = _as_int(chosen["k"])
    oof, _ = _external_oof_predictions(
        embeddings,
        advantages,
        groups,
        k=k,
        seed=37,
    )
    retention_floor = 0.95
    threshold, strong_share, retention = _select_binary_threshold(
        oof,
        cheap,
        strong,
        retention_floor=retention_floor,
    )
    policy = BinaryScalePolicy(
        k=k,
        threshold=threshold,
        external_embeddings=embeddings,
        external_advantages=advantages,
    )
    report: dict[str, object] = {
        "proxy_hypothesis": (
            "task-level benefit from 70B over 8B transfers to benefit from high over low "
            "reasoning effort"
        ),
        "k_search": candidates,
        "selected_k": k,
        "selected_seed_oof_spearman": _spearman(oof, advantages),
        "external_retention_floor": retention_floor,
        "selected_advantage_threshold": threshold,
        "selected_strong_traffic_share": strong_share,
        "selected_external_quality_retained": retention,
        "target_labels_consulted": False,
    }
    return policy, report


def _deep_swe_matrix(
    *,
    source_root: Path,
    task_root: Path,
    embedding_cache_path: Path,
) -> DeepSweData:
    """Load the complete 110-task DeepSWE matrix after the policy is frozen."""
    trials = _json_object(source_root / "trials.json")
    tasks = _json_object(source_root / "tasks.json")
    trial_rows = cast(list[dict[str, object]], trials["rows"])
    task_rows = cast(list[dict[str, object]], tasks["rows"])
    task_meta = {str(row["id"]): row for row in task_rows}

    cells: dict[tuple[str, str], list[dict[str, object]]] = collections.defaultdict(list)
    for row in trial_rows:
        if bool(row.get("included_in_score")):
            cells[(str(row["config"]), str(row["task_name"]))].append(row)
    arms = sorted({arm for arm, _ in cells})
    task_ids = sorted(task_meta)

    rewards = np.full((len(arms), len(task_ids)), np.nan, dtype=np.float64)
    costs = np.full((len(arms), len(task_ids)), np.nan, dtype=np.float64)
    for arm_index, arm in enumerate(arms):
        for task_index, task_id in enumerate(task_ids):
            rows = cells.get((arm, task_id), [])
            reward_values = [_as_float(row["f2p"]) for row in rows if row.get("f2p") is not None]
            cost_values = [
                _as_float(row["cost_usd"]) for row in rows if row.get("cost_usd") is not None
            ]
            if reward_values:
                rewards[arm_index, task_index] = statistics.fmean(reward_values)
            if cost_values:
                costs[arm_index, task_index] = statistics.fmean(cost_values)
    complete = ~(np.isnan(rewards).any(axis=0) | np.isnan(costs).any(axis=0))
    task_ids = [task_id for task_id, keep in zip(task_ids, complete, strict=True) if keep]
    rewards = rewards[:, complete]
    costs = costs[:, complete]

    cache = _embedding_cache(embedding_cache_path)
    missing = [task_id for task_id in task_ids if task_id not in cache]
    if missing:
        raise ValueError(
            f"DeepSWE embedding cache is missing {len(missing)} tasks; "
            f"the strict zero-spend transfer runner will not embed target data: {missing[:5]}"
        )
    embeddings = _normalized(np.asarray([cache[task_id] for task_id in task_ids], dtype=np.float32))
    groups = [str(task_meta[task_id]["repository"]) for task_id in task_ids]
    for task_id in task_ids:
        if not (task_root / task_id / "instruction.md").is_file():
            raise FileNotFoundError(f"DeepSWE task prompt is absent: {task_id}")
    return DeepSweData(
        task_ids=task_ids,
        groups=groups,
        arms=arms,
        rewards=rewards,
        costs=costs,
        embeddings=embeddings,
    )


def _family_efforts(arms: list[str]) -> dict[str, dict[str, int]]:
    families: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for index, arm in enumerate(arms):
        family, effort = arm.rsplit("_", 1)
        if effort in EFFORTS:
            families[family][effort] = index
    return {
        family: by_effort
        for family, by_effort in families.items()
        if len(by_effort) >= 3 and "low" in by_effort and "high" in by_effort
    }


def _effort_for_ease(
    ease: float,
    thresholds: tuple[float, float, float, float],
    available: set[str],
) -> str:
    """Map external ease to a reasoning effort, collapsing absent levels upward."""
    if ease >= thresholds[3]:
        desired = "low"
    elif ease >= thresholds[2]:
        desired = "medium"
    elif ease >= thresholds[1]:
        desired = "high"
    elif ease >= thresholds[0]:
        desired = "xhigh"
    else:
        desired = "max"
    start = EFFORTS.index(desired)
    for effort in EFFORTS[start:]:
        if effort in available:
            return effort
    for effort in reversed(EFFORTS[:start]):
        if effort in available:
            return effort
    raise ValueError("a model family has no recognized reasoning effort")


def _effort_calibration(
    ease: np.ndarray,
    nearest: np.ndarray,
    policy: TransferPolicy,
    *,
    mode: str,
) -> EffortCalibration:
    """Create either the frozen strict policy or a label-free target rank repair."""
    if mode == "strict_absolute":
        return EffortCalibration(
            mode=mode,
            ease_thresholds=policy.ease_thresholds,
            novelty_floor_similarity=policy.floor_sim,
            confirmatory=True,
        )
    if mode == "rank_normalized":
        thresholds = tuple(float(np.quantile(ease, quantile)) for quantile in EFFORT_EASE_QUANTILES)
        return EffortCalibration(
            mode=mode,
            ease_thresholds=cast(tuple[float, float, float, float], thresholds),
            novelty_floor_similarity=float(np.quantile(nearest, NOVELTY_QUANTILE)),
            confirmatory=False,
        )
    raise ValueError(f"unrecognized transfer mode: {mode}")


def _bootstrap(
    policy_reward: np.ndarray,
    policy_cost: np.ndarray,
    baseline_reward: np.ndarray,
    baseline_cost: np.ndarray,
    groups: list[str],
    *,
    seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    unique = sorted(set(groups))
    by_group = {
        group: np.flatnonzero(np.asarray(groups, dtype=object) == group) for group in unique
    }
    deltas = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    ratios = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for sample in range(BOOTSTRAP_SAMPLES):
        selected_groups = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([by_group[str(group)] for group in selected_groups])
        deltas[sample] = float(policy_reward[selected].mean() - baseline_reward[selected].mean())
        denominator = float(policy_cost[selected].sum())
        ratios[sample] = (
            float(baseline_cost[selected].sum() / denominator) if denominator else float("inf")
        )
    return {
        "graded_delta_95ci": [float(value) for value in np.quantile(deltas, [0.025, 0.975])],
        "cost_ratio_95ci": [float(value) for value in np.quantile(ratios, [0.025, 0.975])],
    }


def _evaluate_effort_transfer(
    deep: DeepSweData,
    policy: TransferPolicy,
    *,
    mode: str,
) -> dict[str, object]:
    ease, nearest = _weighted_knn(
        policy.external_embeddings,
        policy.external_strong_rewards,
        deep.embeddings,
        k=policy.k,
    )
    calibration = _effort_calibration(ease, nearest, policy, mode=mode)
    families = _family_efforts(deep.arms)
    family_rows: list[dict[str, object]] = []
    for family, by_effort in sorted(families.items()):
        available = set(by_effort)
        selected_efforts: list[str] = []
        selected_indices: list[int] = []
        for task_index, score in enumerate(ease):
            if nearest[task_index] < calibration.novelty_floor_similarity:
                effort = next(effort for effort in reversed(EFFORTS) if effort in available)
            else:
                effort = _effort_for_ease(
                    float(score),
                    calibration.ease_thresholds,
                    available,
                )
            selected_efforts.append(effort)
            selected_indices.append(by_effort[effort])

        columns = np.arange(len(deep.task_ids), dtype=np.int64)
        routed_reward = deep.rewards[np.asarray(selected_indices), columns]
        routed_cost = deep.costs[np.asarray(selected_indices), columns]
        static = []
        for effort, arm_index in sorted(by_effort.items(), key=lambda item: EFFORTS.index(item[0])):
            static.append(
                {
                    "effort": effort,
                    "arm": deep.arms[arm_index],
                    "graded": float(deep.rewards[arm_index].mean()),
                    "cost_usd": float(deep.costs[arm_index].sum()),
                }
            )
        best = min(
            static,
            key=lambda row: (-float(row["graded"]), float(row["cost_usd"])),
        )
        baseline_index = by_effort[str(best["effort"])]
        baseline_reward = deep.rewards[baseline_index]
        baseline_cost = deep.costs[baseline_index]
        bootstrap = _bootstrap(
            routed_reward,
            routed_cost,
            baseline_reward,
            baseline_cost,
            deep.groups,
            seed=37,
        )
        mix = collections.Counter(selected_efforts)
        routed_graded = float(routed_reward.mean())
        baseline_graded = float(baseline_reward.mean())
        routed_cost_total = float(routed_cost.sum())
        baseline_cost_total = float(baseline_cost.sum())
        family_rows.append(
            {
                "family": family,
                "primary": family == PRIMARY_FAMILY,
                "tasks": len(deep.task_ids),
                "static_efforts": static,
                "best_static_effort": best["effort"],
                "best_static_graded": baseline_graded,
                "best_static_cost_usd": baseline_cost_total,
                "router_graded": routed_graded,
                "quality_retained": (routed_graded / baseline_graded if baseline_graded else 0.0),
                "graded_delta": routed_graded - baseline_graded,
                "router_cost_usd": routed_cost_total,
                "cost_ratio": (
                    baseline_cost_total / routed_cost_total if routed_cost_total else float("inf")
                ),
                "cost_savings": (
                    1.0 - routed_cost_total / baseline_cost_total if baseline_cost_total else 0.0
                ),
                "effort_mix": dict(sorted(mix.items())),
                "novelty_abstentions": int(np.sum(nearest < calibration.novelty_floor_similarity)),
                **bootstrap,
            }
        )
    primary = next((row for row in family_rows if row["primary"]), None)
    return {
        "evaluation_dataset": "DeepSWE v1.1 published trials",
        "target_tasks": len(deep.task_ids),
        "target_repositories": len(set(deep.groups)),
        "target_labels_used_for_training": False,
        "target_labels_used_for_threshold_fit": False,
        "transfer_mode": mode,
        "confirmatory": calibration.confirmatory,
        "calibration": {
            "effort_ease_quantiles": list(EFFORT_EASE_QUANTILES),
            "effort_ease_thresholds": list(calibration.ease_thresholds),
            "novelty_quantile": NOVELTY_QUANTILE,
            "novelty_floor_similarity": calibration.novelty_floor_similarity,
            "uses_target_features": mode == "rank_normalized",
            "uses_target_rewards_or_costs": False,
            "post_hoc_after_strict_target_result_observed": mode == "rank_normalized",
        },
        "primary_family": PRIMARY_FAMILY,
        "primary_result": primary,
        "family_sensitivity": family_rows,
        "target_ease_summary": {
            "minimum": float(ease.min()),
            "median": float(np.median(ease)),
            "maximum": float(ease.max()),
            "nearest_similarity_minimum": float(nearest.min()),
            "nearest_similarity_median": float(np.median(nearest)),
            "nearest_similarity_maximum": float(nearest.max()),
        },
    }


def _evaluate_binary_scale_transfer(
    deep: DeepSweData,
    policy: BinaryScalePolicy,
) -> dict[str, object]:
    """Transfer the external scale-benefit decision to low versus high effort."""
    advantage, nearest = _weighted_knn(
        policy.external_embeddings,
        policy.external_advantages,
        deep.embeddings,
        k=policy.k,
    )
    use_high = advantage >= policy.threshold
    families = _family_efforts(deep.arms)
    family_rows: list[dict[str, object]] = []
    for family, by_effort in sorted(families.items()):
        columns = np.arange(len(deep.task_ids), dtype=np.int64)
        selected_indices = np.where(use_high, by_effort["high"], by_effort["low"])
        routed_reward = deep.rewards[selected_indices, columns]
        routed_cost = deep.costs[selected_indices, columns]
        static = []
        for effort, arm_index in sorted(by_effort.items(), key=lambda item: EFFORTS.index(item[0])):
            static.append(
                {
                    "effort": effort,
                    "arm": deep.arms[arm_index],
                    "graded": float(deep.rewards[arm_index].mean()),
                    "cost_usd": float(deep.costs[arm_index].sum()),
                }
            )
        best = min(
            static,
            key=lambda row: (-float(row["graded"]), float(row["cost_usd"])),
        )
        baseline_index = by_effort[str(best["effort"])]
        baseline_reward = deep.rewards[baseline_index]
        baseline_cost = deep.costs[baseline_index]
        bootstrap = _bootstrap(
            routed_reward,
            routed_cost,
            baseline_reward,
            baseline_cost,
            deep.groups,
            seed=37,
        )
        routed_graded = float(routed_reward.mean())
        baseline_graded = float(baseline_reward.mean())
        routed_cost_total = float(routed_cost.sum())
        baseline_cost_total = float(baseline_cost.sum())
        family_rows.append(
            {
                "family": family,
                "primary": family == PRIMARY_FAMILY,
                "tasks": len(deep.task_ids),
                "static_efforts": static,
                "best_static_effort": best["effort"],
                "best_static_graded": baseline_graded,
                "best_static_cost_usd": baseline_cost_total,
                "router_graded": routed_graded,
                "quality_retained": (routed_graded / baseline_graded if baseline_graded else 0.0),
                "graded_delta": routed_graded - baseline_graded,
                "router_cost_usd": routed_cost_total,
                "cost_ratio": (
                    baseline_cost_total / routed_cost_total if routed_cost_total else float("inf")
                ),
                "cost_savings": (
                    1.0 - routed_cost_total / baseline_cost_total if baseline_cost_total else 0.0
                ),
                "effort_mix": {
                    "high": int(np.sum(use_high)),
                    "low": int(np.sum(~use_high)),
                },
                **bootstrap,
            }
        )
    primary = next((row for row in family_rows if row["primary"]), None)
    return {
        "evaluation_dataset": "DeepSWE v1.1 published trials",
        "target_tasks": len(deep.task_ids),
        "target_repositories": len(set(deep.groups)),
        "target_labels_used_for_training": False,
        "target_labels_used_for_threshold_fit": False,
        "transfer_mode": "binary_scale_benefit",
        "confirmatory": False,
        "post_hoc_after_strict_target_result_observed": True,
        "uses_target_wide_feature_calibration": False,
        "uses_target_rewards_or_costs_for_policy": False,
        "primary_family": PRIMARY_FAMILY,
        "primary_result": primary,
        "family_sensitivity": family_rows,
        "target_advantage_summary": {
            "minimum": float(advantage.min()),
            "median": float(np.median(advantage)),
            "maximum": float(advantage.max()),
            "nearest_similarity_minimum": float(nearest.min()),
            "nearest_similarity_median": float(np.median(nearest)),
            "nearest_similarity_maximum": float(nearest.max()),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--deep-swe-source-root", type=Path, required=True)
    parser.add_argument("--deep-swe-task-root", type=Path, required=True)
    parser.add_argument("--deep-swe-embedding-cache", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    artifact_root = args.artifact_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    _load_env_file(args.env_file.resolve())

    tasks = _collect_external_tasks(artifact_root / "nebius-paired-tasks.json")
    embeddings, prompt_tokens = _embed_external(
        tasks,
        cache_path=artifact_root / "nebius-embeddings.json",
    )
    usage_path = artifact_root / "usage-provenance.json"
    previous_prompt_tokens = 0
    if usage_path.is_file():
        previous_usage = _json_object(usage_path)
        previous_prompt_tokens = _as_int(previous_usage.get("exact_new_embedding_prompt_tokens", 0))
    cumulative_prompt_tokens = previous_prompt_tokens + prompt_tokens
    policy, training_report = _fit_transfer_policy(tasks, embeddings)
    binary_policy, binary_training_report = _fit_binary_scale_policy(tasks, embeddings)
    _write_json(
        artifact_root / "frozen-transfer-policy.json",
        {
            **training_report,
            "binary_scale_policy": binary_training_report,
            "source_manifest": _dataset_source_manifest(),
            "frozen_at": datetime.now(UTC).isoformat(),
            "deep_swe_opened_after_freeze": True,
        },
    )

    deep = _deep_swe_matrix(
        source_root=args.deep_swe_source_root.resolve(),
        task_root=args.deep_swe_task_root.resolve(),
        embedding_cache_path=args.deep_swe_embedding_cache.resolve(),
    )
    evaluation = _evaluate_effort_transfer(deep, policy, mode="strict_absolute")
    exploratory = _evaluate_effort_transfer(deep, policy, mode="rank_normalized")
    binary_exploratory = _evaluate_binary_scale_transfer(deep, binary_policy)
    _write_json(artifact_root / "deep-swe-evaluation.json", evaluation)
    _write_json(
        artifact_root / "deep-swe-evaluation-rank-normalized.json",
        exploratory,
    )
    _write_json(
        artifact_root / "deep-swe-evaluation-binary-scale.json",
        binary_exploratory,
    )
    _write_json(
        usage_path,
        {
            "embedding_model": EMBED_MODEL,
            "exact_new_embedding_prompt_tokens": cumulative_prompt_tokens,
            "new_embedding_prompt_tokens_this_run": prompt_tokens,
            "target_embedding_calls": 0,
            "target_embeddings_reused_from_existing_independent_cache": True,
            "provider_cost_usd": None,
            "provider_cost_note": (
                "Exact embedding token count is retained. Provider billing was not returned, "
                "and the user requested rough estimation rather than a launch gate."
            ),
        },
    )
    primary = cast(dict[str, object] | None, evaluation["primary_result"])
    if primary is None:
        raise ValueError(f"primary effort family is absent: {PRIMARY_FAMILY}")
    logger.info(
        "cross-dataset effort router: quality retained %.3f, cost ratio %.2fx, savings %.1f%%",
        _as_float(primary["quality_retained"]),
        _as_float(primary["cost_ratio"]),
        100.0 * _as_float(primary["cost_savings"]),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
