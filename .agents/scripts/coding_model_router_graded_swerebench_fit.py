"""Fit WMO kNN on graded external SWE-rebench outcomes using remote compute."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from wmo.optimize.knn import fit_knn_policy
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.policy import EmbedderSpec, RoutingPolicy, knn_decision
from wmo.providers.base import ProviderKind
from wmo.providers.pool import PoolEntry

PROTOCOL = "coding-router-graded-swerebench-wmo-knn-v1"
COLLECTION_PROTOCOL = "coding-router-graded-swerebench-development-collection-v1"
DEVELOPMENT_CORPUS_SHA256 = (
    "48d88436a083b66972c25cd7d9439fd149c95bcf9caded2bab7f3b6453aea3d5"
)
CONFIRMATION_CORPUS_SHA256 = (
    "c9443c9956e496123f396ee793efbb3368312092c4dcbd4e5e10bb77bd814f0a"
)
ARMS = (
    "luna-low",
    "luna-medium",
    "luna-high",
    "luna-xhigh",
    "luna-max",
    "sol-max",
)
MODEL_IDS = {
    "luna-low": "gpt-5.6-luna",
    "luna-medium": "gpt-5.6-luna",
    "luna-high": "gpt-5.6-luna",
    "luna-xhigh": "gpt-5.6-luna",
    "luna-max": "gpt-5.6-luna",
    "sol-max": "gpt-5.6-sol",
}
PRICES = {
    "gpt-5.6-luna": (1.0, 0.1, 6.0),
    "gpt-5.6-sol": (5.0, 0.5, 30.0),
}
SEEDS = (11, 23, 37, 41, 59)
FOLDS = 5
K_VALUES = (8, 16, 32, 64)
Z_VALUES = (0.0, 0.5, 1.0, 1.645, 2.0)
LAM_VALUES = (0.0, 0.01, 0.02, 0.03)
QUALITY_RETENTION = 0.95
MIN_SAVINGS = 0.40
MAX_ROUTE_P95_MS = 5.0

EMBEDDING_REPOSITORY = "jinaai/jina-embeddings-v2-base-code"
EMBEDDING_REVISION = "516f4baf13dec4ddddda8631e019b5737c8bc250"
EMBEDDING_MODEL_RELATIVE_PATH = "onnx/model_quantized.onnx"
EMBEDDING_MODEL_EXPECTED_BYTES = 161_895_621
MAX_TOKENS = 1_024
BATCH_SIZE = 4


@dataclass(frozen=True)
class Data:
    """Dense one-attempt graded development matrix."""

    task_ids: list[str]
    repositories: list[str]
    texts: list[str]
    rewards: np.ndarray
    costs: np.ndarray
    rough_cumulative_spend_usd: float


@dataclass(frozen=True)
class Candidate:
    """One preregistered WMO kNN operating point."""

    order: int
    guard: str
    k: int
    z: float
    pick_lam: float

    @property
    def key(self) -> str:
        return f"guard-{self.guard}-k{self.k}-z{self.z:g}-lam{self.pick_lam:g}"


class CachedEmbedder:
    """Serve ephemeral precomputed code vectors by exact task text."""

    def __init__(self, texts: list[str], vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError("embedding matrix shape does not match texts")
        if len(set(texts)) != len(texts):
            raise ValueError("task texts must be unique")
        if not np.isfinite(vectors).all():
            raise ValueError("embedding matrix contains non-finite values")
        self.dim = int(vectors.shape[1])
        self._vectors = {
            text: [float(value) for value in vectors[index]]
            for index, text in enumerate(texts)
        }

    def embed(self, texts: list[str]) -> list[list[float]]:
        if any(text not in self._vectors for text in texts):
            raise KeyError("embedding cache missed an exact task text")
        return [self._vectors[text] for text in texts]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    return {str(key): item for key, item in value.items()}


def _read_object(path: Path) -> dict[str, Any]:
    return _object(json.loads(path.read_text(encoding="utf-8")), str(path))


def _rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            rows.append(_object(json.loads(line), f"outcome row {line_number}"))
    return rows


def _task_text(task: dict[str, Any]) -> str:
    """Use only pre-call task information."""
    return (
        f"repository={task['repository']}\n"
        f"language={task['language']}\n"
        f"{task['prompt']}"
    )


def grouped_folds(groups: list[str], seed: int) -> np.ndarray:
    """Assign whole repositories to deterministic balanced folds."""
    unique = sorted(
        set(groups),
        key=lambda group: hashlib.sha256(f"{seed}|{group}".encode()).hexdigest(),
    )
    mapping = {group: index % FOLDS for index, group in enumerate(unique)}
    folds = np.asarray([mapping[group] for group in groups], dtype=np.int64)
    for fold in range(FOLDS):
        train = {groups[index] for index in np.flatnonzero(folds != fold)}
        heldout = {groups[index] for index in np.flatnonzero(folds == fold)}
        if not heldout or train & heldout:
            raise ValueError("repository-grouped folds overlap or are empty")
    return folds


def candidate_grid() -> tuple[Candidate, ...]:
    values = tuple(
        Candidate(index, guard, k, z, pick_lam)
        for index, (guard, k, z, pick_lam) in enumerate(
            (guard, k, z, pick_lam)
            for guard in ARMS
            for k in K_VALUES
            for z in Z_VALUES
            for pick_lam in LAM_VALUES
        )
    )
    if len(values) != 480 or len({candidate.key for candidate in values}) != 480:
        raise AssertionError("candidate grid is incomplete or duplicated")
    return values


def load_data(corpus_path: Path, outcomes_path: Path, audit_path: Path) -> Data:
    """Load and validate the whole-task-intersected dense matrix."""
    if _sha256(corpus_path) != DEVELOPMENT_CORPUS_SHA256:
        raise ValueError("development corpus changed")
    audit = _read_object(audit_path)
    if (
        audit.get("protocol") != COLLECTION_PROTOCOL
        or audit.get("valid") is not True
        or not isinstance(audit.get("tasks"), int)
        or int(audit["tasks"]) < 640
        or audit.get("arms") != list(ARMS)
        or audit.get("attempts_per_arm") != 1
        or audit.get("cells") != int(audit["tasks"]) * len(ARMS)
        or audit.get("outcomes_sha256") != _sha256(outcomes_path)
        or audit.get("target_outcomes_used") is not False
        or audit.get("deep_swe_outcomes_accessed") is not False
        or audit.get("confirmation_outcomes_accessed") is not False
    ):
        raise ValueError("development collection audit is unsafe")
    raw_tasks = _read_object(corpus_path).get("tasks")
    exclusions = audit.get("excluded_tasks")
    if not isinstance(raw_tasks, list) or len(raw_tasks) != 673 or not isinstance(exclusions, list):
        raise ValueError("development corpus or exclusions are invalid")
    excluded = {
        str(row["task_id"])
        for row in exclusions
        if isinstance(row, dict) and row.get("scope") == "whole-task"
    }
    tasks = [
        _object(task, "development task")
        for task in raw_tasks
        if isinstance(task, dict) and str(task.get("task_id")) not in excluded
    ]
    if len(tasks) != audit["tasks"]:
        raise ValueError("retained tasks do not match exclusions")
    task_ids = [str(task["task_id"]) for task in tasks]
    task_index = {task_id: index for index, task_id in enumerate(task_ids)}
    arm_index = {arm: index for index, arm in enumerate(ARMS)}
    rewards = np.full((len(tasks), len(ARMS)), np.nan)
    costs = np.full_like(rewards, np.nan)
    observed: set[tuple[str, str]] = set()
    for row in _rows(outcomes_path):
        task_id = row.get("task_id")
        arm = row.get("arm")
        reward = row.get("reward")
        cost = row.get("cost_usd")
        identity = (str(task_id), str(arm))
        if (
            not isinstance(task_id, str)
            or task_id not in task_index
            or not isinstance(arm, str)
            or arm not in arm_index
            or identity in observed
            or row.get("attempt_number") != 0
            or row.get("model") != MODEL_IDS[arm]
            or row.get("target_outcomes_used") is not False
            or isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(float(reward))
            or not 0.0 <= float(reward) <= 1.0
            or isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(float(cost))
            or float(cost) < 0.0
        ):
            raise ValueError(f"invalid outcome identity: {identity}")
        rewards[task_index[task_id], arm_index[arm]] = float(reward)
        costs[task_index[task_id], arm_index[arm]] = float(cost)
        observed.add(identity)
    if len(observed) != rewards.size or np.isnan(rewards).any() or np.isnan(costs).any():
        raise ValueError("development matrix is not dense")
    spend = audit.get("rough_cumulative_experiment_spend_usd")
    if isinstance(spend, bool) or not isinstance(spend, (int, float)):
        raise ValueError("development audit lacks cumulative spend")
    return Data(
        task_ids=task_ids,
        repositories=[str(task["repository"]) for task in tasks],
        texts=[_task_text(task) for task in tasks],
        rewards=rewards,
        costs=costs,
        rough_cumulative_spend_usd=float(spend),
    )


def load_confirmation(path: Path) -> list[dict[str, Any]]:
    if _sha256(path) != CONFIRMATION_CORPUS_SHA256:
        raise ValueError("confirmation corpus changed")
    rows = _read_object(path).get("tasks")
    if not isinstance(rows, list) or len(rows) != 320:
        raise ValueError("confirmation corpus must contain 320 tasks")
    tasks = [_object(task, "confirmation task") for task in rows]
    if len({str(task["task_id"]) for task in tasks}) != 320:
        raise ValueError("confirmation task identities repeat")
    return tasks


def _pool() -> list[PoolEntry]:
    result: list[PoolEntry] = []
    for arm in ARMS:
        model = MODEL_IDS[arm]
        result.append(
            PoolEntry(
                name=arm,
                kind=ProviderKind.OPENAI_RESPONSES,
                model=model,
                reasoning_effort=arm.split("-", 1)[1],
                input_per_mtok=PRICES[model][0],
                cached_input_per_mtok=PRICES[model][1],
                output_per_mtok=PRICES[model][2],
            )
        )
    return result


def _matrix(data: Data) -> OutcomeMatrix:
    return OutcomeMatrix(
        pool=_pool(),
        outcomes=[
            ScenarioOutcome(
                scenario_id=task_id,
                task=data.texts[task_index],
                model=arm,
                benchmark="graded-swerebench-v2-external-development",
                episode=0,
                attempt_number=1,
                reward=float(data.rewards[task_index, arm_index]),
                success=bool(data.rewards[task_index, arm_index] >= 1.0),
                cost_usd=float(data.costs[task_index, arm_index]),
                completion_status="scored",
                usage_accounting="trace-derived",
            )
            for task_index, task_id in enumerate(data.task_ids)
            for arm_index, arm in enumerate(ARMS)
        ],
    )


def _static_baseline(data: Data) -> int:
    rewards = data.rewards.mean(axis=0)
    costs = data.costs.mean(axis=0)
    return min(range(len(ARMS)), key=lambda arm: (-rewards[arm], costs[arm], arm))


def _metrics(data: Data, choices: np.ndarray) -> dict[str, Any]:
    if choices.shape != (len(data.task_ids),) or np.any(
        (choices < 0) | (choices >= len(ARMS))
    ):
        raise ValueError("routing choices must cover every task with a known arm")
    rows = np.arange(len(choices))
    reward = float(np.mean(data.rewards[rows, choices]))
    cost = float(np.mean(data.costs[rows, choices]))
    static_rewards = data.rewards.mean(axis=0)
    static_costs = data.costs.mean(axis=0)
    baseline = _static_baseline(data)
    traffic = np.bincount(choices, minlength=len(ARMS)).astype(float) / len(choices)
    blind_reward = float(traffic @ static_rewards)
    dominated = [
        ARMS[index]
        for index in range(len(ARMS))
        if static_rewards[index] >= reward
        and static_costs[index] <= cost
        and (static_rewards[index] > reward or static_costs[index] < cost)
    ]
    return {
        "reward": reward,
        "cost_usd_per_task": cost,
        "quality_retention": reward / float(static_rewards[baseline]),
        "absolute_quality_delta": reward - float(static_rewards[baseline]),
        "cost_savings": 1.0 - cost / float(static_costs[baseline]),
        "matched_blind_reward": blind_reward,
        "matched_blind_advantage": reward - blind_reward,
        "model_mix": {
            ARMS[index]: float(share)
            for index, share in enumerate(traffic)
            if share > 0.0
        },
        "dominated_by_static": dominated,
    }


def _passes_development_gates(seed_metrics: list[dict[str, Any]]) -> bool:
    """Apply every frozen promotion gate independently to every split seed."""
    return (
        len(seed_metrics) == len(SEEDS)
        and all(float(row["quality_retention"]) >= QUALITY_RETENTION for row in seed_metrics)
        and all(float(row["cost_savings"]) >= MIN_SAVINGS for row in seed_metrics)
        and all(float(row["matched_blind_advantage"]) > 0.0 for row in seed_metrics)
        and all(not row["dominated_by_static"] for row in seed_metrics)
    )


def _frontiers(data: Data) -> dict[str, Any]:
    static_rewards = data.rewards.mean(axis=0)
    static_costs = data.costs.mean(axis=0)
    static = [
        {
            "arm": arm,
            "reward": float(static_rewards[index]),
            "cost_usd_per_task": float(static_costs[index]),
        }
        for index, arm in enumerate(ARMS)
    ]
    rows = np.arange(len(data.task_ids))
    best_reward = data.rewards.max(axis=1)
    full_choices = np.asarray(
        [
            min(
                np.flatnonzero(data.rewards[index] == best_reward[index]),
                key=lambda arm: (data.costs[index, arm], int(arm)),
            )
            for index in rows
        ]
    )
    pair_oracles = []
    for left, right in combinations(range(len(ARMS)), 2):
        choices = np.where(
            data.rewards[:, right] > data.rewards[:, left],
            right,
            np.where(
                data.rewards[:, left] > data.rewards[:, right],
                left,
                np.where(data.costs[:, right] < data.costs[:, left], right, left),
            ),
        )
        pair_oracles.append({"pair": [ARMS[left], ARMS[right]], **_metrics(data, choices)})
    return {
        "fit_selected_static": ARMS[_static_baseline(data)],
        "static": static,
        "full_oracle": _metrics(data, full_choices),
        "pair_oracles": pair_oracles,
    }


def _embed(
    texts: list[str],
    model_path: Path,
    tokenizer_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    if model_path.stat().st_size != EMBEDDING_MODEL_EXPECTED_BYTES:
        raise ValueError("embedding ONNX content length changed")
    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError as error:  # pragma: no cover - remote-only dependencies
        raise RuntimeError("remote ONNX dependencies are unavailable") from error
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=MAX_TOKENS)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inputs = {value.name for value in session.get_inputs()}
    if not {"input_ids", "attention_mask"}.issubset(inputs):
        raise ValueError(f"unsupported ONNX inputs: {sorted(inputs)}")
    parts: list[np.ndarray] = []
    started = time.perf_counter()
    for start in range(0, len(texts), BATCH_SIZE):
        encoded = tokenizer.encode_batch(texts[start : start + BATCH_SIZE])
        width = max(len(value.ids) for value in encoded)
        input_ids = np.zeros((len(encoded), width), dtype=np.int64)
        attention = np.zeros_like(input_ids)
        token_types = np.zeros_like(input_ids)
        for row, value in enumerate(encoded):
            size = len(value.ids)
            input_ids[row, :size] = value.ids
            attention[row, :size] = 1
            if value.type_ids:
                token_types[row, :size] = value.type_ids
        feed = {"input_ids": input_ids, "attention_mask": attention}
        if "token_type_ids" in inputs:
            feed["token_type_ids"] = token_types
        output = np.asarray(session.run(None, feed)[0], dtype=np.float32)
        if output.ndim == 3:
            mask = attention[..., None].astype(np.float32)
            output = (output * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1.0)
        norms = np.linalg.norm(output, axis=1, keepdims=True)
        parts.append(output / np.maximum(norms, np.finfo(np.float32).eps))
    vectors = np.concatenate(parts, axis=0)
    if vectors.shape[0] != len(texts) or not np.isfinite(vectors).all():
        raise ValueError("embedding inference produced invalid vectors")
    return vectors, {
        "tasks": len(texts),
        "dimension": int(vectors.shape[1]),
        "seconds": time.perf_counter() - started,
        "model_sha256": _sha256(model_path),
        "tokenizer_sha256": _sha256(tokenizer_path),
    }


def _tune(policy: RoutingPolicy, candidate: Candidate) -> RoutingPolicy:
    return policy.model_copy(
        update={
            "default_model": candidate.guard,
            "guard_model": candidate.guard,
            "rag_num": candidate.k,
            "rag_thres": 0.95,
            "floor_q": 0.0,
            "floor_sim": None,
            "knn_z": candidate.z,
            "knn_min_pairs": 8,
            "se_floor": True,
            "guard_mode": "asymmetric",
            "pick_lam": candidate.pick_lam,
        }
    )


def _crossfit(data: Data, vectors: np.ndarray) -> tuple[Candidate | None, dict[str, Any]]:
    matrix = _matrix(data)
    embedder = CachedEmbedder(data.texts, vectors)
    spec = EmbedderSpec(kind="hashing", dim=embedder.dim)
    candidates = candidate_grid()
    by_base: dict[tuple[str, int], list[Candidate]] = {}
    for candidate in candidates:
        by_base.setdefault((candidate.guard, candidate.k), []).append(candidate)
    routes = {
        (seed, candidate.key): np.full(len(data.task_ids), -1, dtype=np.int16)
        for seed in SEEDS
        for candidate in candidates
    }
    timings: list[float] = []
    with tempfile.TemporaryDirectory(prefix="graded-wmo-knn-") as directory:
        root = Path(directory)
        for seed in SEEDS:
            folds = grouped_folds(data.repositories, seed)
            for fold in range(FOLDS):
                train = np.flatnonzero(folds != fold)
                heldout = np.flatnonzero(folds == fold)
                fit_ids = [data.task_ids[index] for index in train]
                for (guard, k), variants in by_base.items():
                    policy = fit_knn_policy(
                        matrix,
                        bank_path=root / f"seed-{seed}-fold-{fold}-{guard}-k{k}.npz",
                        fit_ids=fit_ids,
                        embedder=spec,
                        embed_with=embedder,
                        guard_model=guard,
                        rag_num=k,
                        rag_thres=0.95,
                        z=0.0,
                        min_pairs=8,
                        se_floor=True,
                        floor_q=0.0,
                        pick_lam=0.0,
                        fitted_from=f"{PROTOCOL} repository-grouped development",
                    )
                    for candidate in variants:
                        tuned = _tune(policy, candidate)
                        selected = routes[(seed, candidate.key)]
                        for index in heldout:
                            started = time.perf_counter_ns()
                            decision = knn_decision(tuned, vectors[index])
                            timings.append((time.perf_counter_ns() - started) / 1_000_000)
                            selected[index] = ARMS.index(decision.model)
    eligible: list[tuple[float, float, int, Candidate, list[dict[str, Any]]]] = []
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        seed_metrics = [_metrics(data, routes[(seed, candidate.key)]) for seed in SEEDS]
        passed = _passes_development_gates(seed_metrics)
        item = {
            "candidate": candidate.key,
            "configuration": asdict(candidate),
            "eligible": passed,
            "mean_reward": float(np.mean([float(row["reward"]) for row in seed_metrics])),
            "mean_cost_usd_per_task": float(
                np.mean([float(row["cost_usd_per_task"]) for row in seed_metrics])
            ),
            "mean_matched_blind_advantage": float(
                np.mean([float(row["matched_blind_advantage"]) for row in seed_metrics])
            ),
            "seeds": [
                {"seed": seed, **metrics}
                for seed, metrics in zip(SEEDS, seed_metrics, strict=True)
            ],
        }
        rows.append(item)
        if passed:
            eligible.append(
                (
                    float(item["mean_cost_usd_per_task"]),
                    -float(item["mean_reward"]),
                    candidate.order,
                    candidate,
                    seed_metrics,
                )
            )
    selected = min(eligible)[3] if eligible else None
    selected_row = (
        next(row for row in rows if row["candidate"] == selected.key)
        if selected is not None
        else None
    )
    closest = sorted(
        rows,
        key=lambda row: (
            max(
                0.0,
                QUALITY_RETENTION
                - min(float(seed["quality_retention"]) for seed in row["seeds"]),
            ),
            max(
                0.0,
                MIN_SAVINGS - min(float(seed["cost_savings"]) for seed in row["seeds"]),
            ),
            -float(row["mean_matched_blind_advantage"]),
            float(row["mean_cost_usd_per_task"]),
        ),
    )[:20]
    return selected, {
        "candidate_count": len(candidates),
        "eligible_count": len(eligible),
        "selected": selected_row,
        "closest_candidates": closest,
        "route_decision_latency_ms": {
            "samples": len(timings),
            "p50": float(np.quantile(timings, 0.50)),
            "p95": float(np.quantile(timings, 0.95)),
            "maximum": float(np.max(timings)),
            "gate_ms": MAX_ROUTE_P95_MS,
        },
    }


def _freeze_routes(
    selected: Candidate,
    data: Data,
    confirmation: list[dict[str, Any]],
    development_vectors: np.ndarray,
    confirmation_vectors: np.ndarray,
) -> dict[str, Any]:
    all_texts = data.texts + [_task_text(task) for task in confirmation]
    embedder = CachedEmbedder(
        all_texts,
        np.concatenate([development_vectors, confirmation_vectors], axis=0),
    )
    spec = EmbedderSpec(kind="hashing", dim=embedder.dim)
    with tempfile.TemporaryDirectory(prefix="graded-wmo-final-") as directory:
        policy = fit_knn_policy(
            _matrix(data),
            bank_path=Path(directory) / "bank.npz",
            fit_ids=data.task_ids,
            embedder=spec,
            embed_with=embedder,
            guard_model=selected.guard,
            rag_num=selected.k,
            rag_thres=0.95,
            z=0.0,
            min_pairs=8,
            se_floor=True,
            floor_q=0.0,
            pick_lam=0.0,
            fitted_from=f"{PROTOCOL} full development",
        )
        policy = _tune(policy, selected)
        rows = []
        timings = []
        for index, task in enumerate(confirmation):
            started = time.perf_counter_ns()
            decision = knn_decision(policy, confirmation_vectors[index])
            timings.append((time.perf_counter_ns() - started) / 1_000_000)
            rows.append(
                {
                    "task_id": task["task_id"],
                    "repository": task["repository"],
                    "arm": decision.model,
                    "reason": decision.reason,
                }
            )
    return {
        "protocol": PROTOCOL,
        "selected_candidate": selected.key,
        "routes": rows,
        "route_decision_latency_ms": {
            "p50": float(np.quantile(timings, 0.50)),
            "p95": float(np.quantile(timings, 0.95)),
            "maximum": float(np.max(timings)),
        },
        "deep_swe_outcomes_accessed": False,
        "confirmation_outcomes_accessed": False,
        "fitted_numeric_state_persisted": False,
    }


def run(args: argparse.Namespace) -> int:
    data = load_data(args.corpus, args.outcomes, args.audit)
    confirmation = load_confirmation(args.confirmation_corpus)
    texts = data.texts + [_task_text(task) for task in confirmation]
    vectors, embedding = _embed(texts, args.embedding_model, args.tokenizer)
    dev_vectors = vectors[: len(data.texts)]
    confirmation_vectors = vectors[len(data.texts) :]
    selected, development = _crossfit(data, dev_vectors)
    routes = None
    if selected is not None:
        routes = _freeze_routes(
            selected,
            data,
            confirmation,
            dev_vectors,
            confirmation_vectors,
        )
        args.routes_out.write_text(json.dumps(routes, indent=2, sort_keys=True) + "\n")
    report = {
        "protocol": PROTOCOL,
        "valid": True,
        "development_passed": selected is not None,
        "deep_swe_outcomes_accessed": False,
        "target_outcomes_used": False,
        "confirmation_outcomes_accessed": False,
        "fitted_numeric_state_persisted": False,
        "embedding_model_persisted": False,
        "task_embeddings_persisted": False,
        "knn_bank_persisted": False,
        "embedding": {
            "repository": EMBEDDING_REPOSITORY,
            "revision": EMBEDDING_REVISION,
            "relative_path": EMBEDDING_MODEL_RELATIVE_PATH,
            **embedding,
        },
        "frontiers": _frontiers(data),
        "development": development,
        "confirmation_routes_frozen": routes is not None,
        "rough_cumulative_spend_usd": data.rough_cumulative_spend_usd,
    }
    args.report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--confirmation-corpus", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--routes-out", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
