"""Fit and gate latency-neutral effort routers on frozen Codeforces outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

logger = logging.getLogger("coding-model-router-codeforces-fit")

ARMS = ("luna-low", "luna-medium", "luna-high", "luna-xhigh", "luna-max")
HIGH_INDEX = 2
ATTEMPTS = 2
EXPECTED_TASKS = 160
BOOTSTRAPS = 2_000
QUALITY_TOLERANCE = 0.005
DIMS = (512, 2_048)
ALPHAS = (1.0, 10.0, 100.0)
THRESHOLDS = (0.0, 0.02, 0.05, 0.10)


@dataclass(frozen=True)
class Data:
    """Dense source tasks, rewards, costs, and label-free features."""

    task_ids: list[str]
    groups: list[str]
    texts: list[str]
    structural: np.ndarray
    rewards: np.ndarray
    costs: np.ndarray


@dataclass(frozen=True)
class Candidate:
    """One frozen direct-uplift scorer configuration."""

    dim: int
    alpha: float
    threshold: float

    @property
    def name(self) -> str:
        """Return a stable configuration label."""
        return f"direct-hash{self.dim}-a{self.alpha:g}-t{self.threshold:g}"


CANDIDATES = tuple(
    Candidate(dim, alpha, threshold) for dim in DIMS for alpha in ALPHAS for threshold in THRESHOLDS
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return {str(key): item for key, item in value.items()}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} contains a non-object row")
            rows.append({str(key): item for key, item in value.items()})
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _structural(task: dict[str, Any]) -> list[float]:
    prompt = str(task["prompt"])
    lower = prompt.casefold()
    bucket = str(task["bucket"])
    return [
        math.log1p(len(prompt)),
        math.log1p(len(prompt.split())),
        math.log1p(len(prompt.splitlines())),
        math.log1p(len(cast(list[object], task["tests"]))),
        math.log1p(float(task["time_limit_s"])),
        math.log1p(float(task["memory_limit_mb"])),
        float(prompt.count("`")),
        float(prompt.count("\n")),
        float(lower.count("input")),
        float(lower.count("output")),
        float(lower.count("example")),
        float(lower.count("constraint")),
        float("graph" in lower),
        float("tree" in lower),
        float("string" in lower),
        float("array" in lower),
        float("dynamic programming" in lower),
        float(bucket == "C"),
        float(bucket == "D"),
        float(bucket == "E"),
        float(bucket == "F+"),
    ]


def load_data(
    corpus_path: Path,
    outcomes_path: Path,
    *,
    expected_tasks: int = EXPECTED_TASKS,
) -> Data:
    """Load and prove the exact dense source matrix without target labels."""
    corpus = _read_object(corpus_path)
    if (
        corpus.get("target_outcomes_used") is not False
        or corpus.get("published_generations_loaded") is not False
    ):
        raise ValueError("corpus violated a frozen information boundary")
    raw_tasks = corpus.get("tasks")
    if not isinstance(raw_tasks, list) or len(raw_tasks) != expected_tasks:
        raise ValueError(f"corpus does not contain the expected {expected_tasks} tasks")
    tasks = [
        {str(key): item for key, item in task.items()}
        for task in raw_tasks
        if isinstance(task, dict)
    ]
    if len(tasks) != len(raw_tasks):
        raise ValueError("corpus contains a non-object task")
    task_ids = [str(task["task_id"]) for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("corpus contains duplicate task IDs")
    task_index = {task_id: index for index, task_id in enumerate(task_ids)}
    arm_index = {arm: index for index, arm in enumerate(ARMS)}
    rewards = np.full((len(tasks), len(ARMS), ATTEMPTS), np.nan)
    costs = np.full_like(rewards, np.nan)
    observed: set[str] = set()
    for row in _read_rows(outcomes_path):
        cell_id = str(row.get("cell_id") or "")
        task_id = str(row.get("task_id") or "")
        arm = str(row.get("arm") or "")
        attempt = row.get("attempt")
        if (
            not cell_id
            or cell_id in observed
            or task_id not in task_index
            or arm not in arm_index
            or not isinstance(attempt, int)
            or isinstance(attempt, bool)
            or not 0 <= attempt < ATTEMPTS
        ):
            raise ValueError(f"invalid or duplicate source cell {cell_id!r}")
        if (
            row.get("observed_model") != "gpt-5.6-luna"
            or row.get("target_outcomes_used") is not False
            or int(row.get("tests_total") or 0) < 10
        ):
            raise ValueError(f"ungradeable source cell {cell_id}")
        index = (task_index[task_id], arm_index[arm], attempt)
        rewards[index] = float(row["reward"])
        costs[index] = float(row["cost_usd"])
        observed.add(cell_id)
    if len(observed) != expected_tasks * len(ARMS) * ATTEMPTS:
        raise ValueError("source matrix is incomplete")
    if not np.isfinite(rewards).all() or not np.isfinite(costs).all():
        raise ValueError("source matrix is not finite and dense")
    return Data(
        task_ids=task_ids,
        groups=[str(task["contest_id"]) for task in tasks],
        texts=[str(task["prompt"]) for task in tasks],
        structural=np.asarray([_structural(task) for task in tasks], dtype=np.float64),
        rewards=rewards.mean(axis=2),
        costs=costs.mean(axis=2),
    )


def _features(data: Data, dim: int, train: np.ndarray) -> sparse.csr_matrix:
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        n_features=dim,
        alternate_sign=True,
        norm="l2",
    )
    text = cast(sparse.csr_matrix, vectorizer.transform(data.texts))
    structural = data.structural.copy()
    scale = np.maximum(np.std(structural[train], axis=0), 1.0)
    structural /= scale
    return sparse.hstack([text, sparse.csr_matrix(structural)], format="csr")


def _predict_deltas(
    data: Data,
    train: np.ndarray,
    test: np.ndarray,
    candidate: Candidate,
    *,
    label_rewards: np.ndarray | None = None,
) -> np.ndarray:
    features = _features(data, candidate.dim, train)
    labels = data.rewards if label_rewards is None else label_rewards
    deltas = labels - labels[:, [HIGH_INDEX]]
    models = _fit_delta_models(features, deltas, train, alpha=candidate.alpha)
    return _score_delta_models(features, test, models)


def _fit_delta_models(
    features: sparse.csr_matrix,
    deltas: np.ndarray,
    train: np.ndarray,
    *,
    alpha: float,
) -> tuple[Ridge | None, ...]:
    """Fit the ephemeral per-arm delta heads used by one route policy."""
    models: list[Ridge | None] = []
    for arm_index in range(len(ARMS)):
        if arm_index == HIGH_INDEX:
            models.append(None)
            continue
        model = Ridge(alpha=alpha)
        model.fit(features[train], deltas[train, arm_index])
        models.append(model)
    return tuple(models)


def _score_delta_models(
    features: sparse.csr_matrix,
    test: np.ndarray,
    models: tuple[Ridge | None, ...],
) -> np.ndarray:
    """Score fitted delta heads without fitting or persisting model state."""
    if len(models) != len(ARMS):
        raise ValueError("delta head count does not match the arm count")
    predictions = np.zeros((len(test), len(ARMS)), dtype=np.float64)
    for arm_index, model in enumerate(models):
        if model is None:
            continue
        predictions[:, arm_index] = model.predict(features[test])
    return predictions


def _choose(
    predictions: np.ndarray,
    mean_costs: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    adjusted = predictions.copy()
    for arm_index in range(len(ARMS)):
        if arm_index != HIGH_INDEX:
            adjusted[:, arm_index] -= threshold
    choices = np.empty(len(adjusted), dtype=np.int64)
    for index, row in enumerate(adjusted):
        best = float(np.max(row))
        candidates = np.flatnonzero(np.isclose(row, best, atol=1e-12))
        choices[index] = int(candidates[int(np.argmin(mean_costs[candidates]))])
    return choices


def _value(data: Data, indices: np.ndarray, choices: np.ndarray) -> dict[str, Any]:
    rewards = data.rewards[indices]
    costs = data.costs[indices]
    counts = np.bincount(choices, minlength=len(ARMS))
    fractions = counts / len(choices)
    routed_reward = rewards[np.arange(len(indices)), choices]
    routed_cost = costs[np.arange(len(indices)), choices]
    blind_reward = rewards @ fractions
    blind_cost = costs @ fractions
    return {
        "reward": float(np.mean(routed_reward)),
        "cost_usd": float(np.sum(routed_cost)),
        "matched_blind_reward": float(np.mean(blind_reward)),
        "matched_blind_cost_usd": float(np.sum(blind_cost)),
        "advantage": float(np.mean(routed_reward - blind_reward)),
        "counts": {ARMS[i]: int(counts[i]) for i in range(len(ARMS))},
        "routed_reward_by_task": routed_reward,
        "routed_cost_by_task": routed_cost,
        "blind_reward_by_task": blind_reward,
        "blind_cost_by_task": blind_cost,
    }


def _inner_oof(data: Data, outer_train: np.ndarray, candidate: Candidate) -> dict[str, Any]:
    groups = np.asarray(data.groups, dtype=object)[outer_train]
    predictions = np.zeros((len(outer_train), len(ARMS)), dtype=np.float64)
    splitter = GroupKFold(n_splits=4)
    for fit_local, test_local in splitter.split(outer_train, groups=groups):
        fit = outer_train[fit_local]
        test = outer_train[test_local]
        predictions[test_local] = _predict_deltas(data, fit, test, candidate)
    choices = _choose(
        predictions,
        data.costs[outer_train].mean(axis=0),
        threshold=candidate.threshold,
    )
    value = _value(data, outer_train, choices)
    value["high_reward"] = float(data.rewards[outer_train, HIGH_INDEX].mean())
    value["high_cost_usd"] = float(data.costs[outer_train, HIGH_INDEX].sum())
    return value


def _select_candidate(
    data: Data, outer_train: np.ndarray
) -> tuple[Candidate, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        value = _inner_oof(data, outer_train, candidate)
        feasible = (
            float(value["reward"]) >= float(value["high_reward"]) - QUALITY_TOLERANCE
            and float(value["cost_usd"]) < float(value["high_cost_usd"])
            and float(value["advantage"]) > 0.0
        )
        rows.append(
            {
                "name": candidate.name,
                "dim": candidate.dim,
                "alpha": candidate.alpha,
                "threshold": candidate.threshold,
                "reward": value["reward"],
                "cost_usd": value["cost_usd"],
                "high_reward": value["high_reward"],
                "high_cost_usd": value["high_cost_usd"],
                "advantage": value["advantage"],
                "feasible": feasible,
            }
        )
    feasible_rows = [row for row in rows if row["feasible"]]
    if feasible_rows:
        selected = min(
            feasible_rows,
            key=lambda row: (
                float(row["cost_usd"]),
                -float(row["reward"]),
                str(row["name"]),
            ),
        )
    else:
        selected = max(
            rows,
            key=lambda row: (
                float(row["advantage"]),
                float(row["reward"]) - float(row["high_reward"]),
                -float(row["cost_usd"]),
                str(row["name"]),
            ),
        )
    candidate = next(row for row in CANDIDATES if row.name == selected["name"])
    return candidate, rows


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
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _bootstrap(
    groups: list[str],
    router: np.ndarray,
    blind: np.ndarray,
    *,
    seed: int,
) -> list[float]:
    unique = sorted(set(groups))
    group_array = np.asarray(groups, dtype=object)
    members = {group: np.flatnonzero(group_array == group) for group in unique}
    rng = np.random.default_rng(seed)
    values = np.empty(BOOTSTRAPS, dtype=np.float64)
    for sample in range(BOOTSTRAPS):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([members[str(group)] for group in sampled])
        values[sample] = np.mean(router[indices] - blind[indices])
    return [float(value) for value in np.quantile(values, [0.025, 0.5, 0.975])]


def fit(
    corpus: Path,
    outcomes: Path,
    output: Path,
    *,
    seed: int,
    expected_tasks: int = EXPECTED_TASKS,
) -> None:
    """Run nested grouped selection and write the external promotion report."""
    data = load_data(corpus, outcomes, expected_tasks=expected_tasks)
    splitter = GroupKFold(n_splits=5)
    all_indices = np.arange(len(data.task_ids))
    choices = np.full(len(data.task_ids), -1, dtype=np.int64)
    shuffled_choices = np.full_like(choices, -1)
    predicted = np.zeros(len(data.task_ids), dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    for fold, (train, test) in enumerate(
        splitter.split(all_indices, groups=np.asarray(data.groups))
    ):
        if set(np.asarray(data.groups)[train]) & set(np.asarray(data.groups)[test]):
            raise AssertionError("contest group crossed an outer fold")
        candidate, inner = _select_candidate(data, train)
        predictions = _predict_deltas(data, train, test, candidate)
        choices[test] = _choose(
            predictions,
            data.costs[train].mean(axis=0),
            threshold=candidate.threshold,
        )
        predicted[test] = predictions[np.arange(len(test)), choices[test]]
        shuffled_rewards = data.rewards.copy()
        permutation = np.random.default_rng(seed + fold).permutation(train)
        shuffled_rewards[train] = data.rewards[permutation]
        shuffled_predictions = _predict_deltas(
            data,
            train,
            test,
            candidate,
            label_rewards=shuffled_rewards,
        )
        shuffled_choices[test] = _choose(
            shuffled_predictions,
            data.costs[train].mean(axis=0),
            threshold=candidate.threshold,
        )
        static_fit = data.rewards[train].mean(axis=0)
        best_reward = float(np.max(static_fit))
        static_candidates = np.flatnonzero(np.isclose(static_fit, best_reward))
        static_arm = int(
            static_candidates[int(np.argmin(data.costs[train].mean(axis=0)[static_candidates]))]
        )
        fold_rows.append(
            {
                "fold": fold,
                "train_tasks": len(train),
                "test_tasks": len(test),
                "train_contests": len({data.groups[index] for index in train}),
                "test_contests": len({data.groups[index] for index in test}),
                "contest_overlap": 0,
                "selected_candidate": candidate.name,
                "selected_static_arm": ARMS[static_arm],
                "inner_candidates": inner,
            }
        )
    if np.any(choices < 0) or np.any(shuffled_choices < 0):
        raise RuntimeError("outer predictions are incomplete")
    value = _value(data, all_indices, choices)
    shuffled = _value(data, all_indices, shuffled_choices)
    router_reward = cast(np.ndarray, value["routed_reward_by_task"])
    blind_reward = cast(np.ndarray, value["blind_reward_by_task"])
    shuffled_router = cast(np.ndarray, shuffled["routed_reward_by_task"])
    shuffled_blind = cast(np.ndarray, shuffled["blind_reward_by_task"])
    observed_uplift = router_reward - data.rewards[:, HIGH_INDEX]
    interval = _bootstrap(data.groups, router_reward, blind_reward, seed=seed)
    shuffled_interval = _bootstrap(
        data.groups,
        shuffled_router,
        shuffled_blind,
        seed=seed + 1,
    )
    static = [
        {
            "arm": arm,
            "reward": float(data.rewards[:, index].mean()),
            "cost_usd": float(data.costs[:, index].sum()),
        }
        for index, arm in enumerate(ARMS)
    ]
    dominated_by = [
        row["arm"]
        for row in static
        if float(row["reward"]) >= float(value["reward"])
        and float(row["cost_usd"]) <= float(value["cost_usd"])
        and (
            float(row["reward"]) > float(value["reward"])
            or float(row["cost_usd"]) < float(value["cost_usd"])
        )
    ]
    shuffled_gate = float(shuffled["advantage"]) > 0.0 and shuffled_interval[0] > 0.0
    gate = {
        "positive_oof_uplift_spearman": _spearman(predicted, observed_uplift) > 0.0,
        "positive_matched_blind_advantage": float(value["advantage"]) > 0.0,
        "positive_contest_bootstrap_lower_bound": interval[0] > 0.0,
        "not_static_dominated": not dominated_by,
        "shuffled_control_failed": not shuffled_gate,
    }
    gate["passed"] = all(gate.values())
    consensus = max(
        {row["selected_candidate"] for row in fold_rows},
        key=lambda name: (
            sum(row["selected_candidate"] == name for row in fold_rows),
            str(name),
        ),
    )
    consensus_candidate = next(row for row in CANDIDATES if row.name == consensus)
    full_predictions = _predict_deltas(
        data,
        all_indices,
        all_indices,
        consensus_candidate,
    )
    started = time.perf_counter_ns()
    for _ in range(100):
        _choose(
            full_predictions,
            data.costs.mean(axis=0),
            threshold=consensus_candidate.threshold,
        )
    decision_batch_ms = (time.perf_counter_ns() - started) / 1_000_000 / 100
    full_features = _features(data, consensus_candidate.dim, all_indices)
    full_deltas = data.rewards - data.rewards[:, [HIGH_INDEX]]
    full_models = _fit_delta_models(
        full_features,
        full_deltas,
        all_indices,
        alpha=consensus_candidate.alpha,
    )
    started = time.perf_counter_ns()
    for _ in range(100):
        inference_features = _features(data, consensus_candidate.dim, all_indices)
        inference_predictions = _score_delta_models(
            inference_features,
            all_indices,
            full_models,
        )
        _choose(
            inference_predictions,
            data.costs.mean(axis=0),
            threshold=consensus_candidate.threshold,
        )
    inference_batch_ms = (time.perf_counter_ns() - started) / 1_000_000 / 100
    report = {
        "protocol": "codeforces-nested-contest-grouped-fit-v1",
        "tasks": len(data.task_ids),
        "contest_groups": len(set(data.groups)),
        "arms": list(ARMS),
        "attempts": ATTEMPTS,
        "static_efforts": static,
        "nested_outer_folds": fold_rows,
        "router": {
            "reward": value["reward"],
            "cost_usd": value["cost_usd"],
            "arm_counts": value["counts"],
            "matched_blind_reward": value["matched_blind_reward"],
            "matched_blind_cost_usd": value["matched_blind_cost_usd"],
            "advantage_vs_matched_blind": value["advantage"],
            "predicted_uplift_spearman": _spearman(predicted, observed_uplift),
            "contest_bootstrap_advantage_95ci": interval,
            "dominated_by_static_arms": dominated_by,
        },
        "shuffled_control": {
            "reward": shuffled["reward"],
            "cost_usd": shuffled["cost_usd"],
            "advantage_vs_matched_blind": shuffled["advantage"],
            "contest_bootstrap_advantage_95ci": shuffled_interval,
            "passed_primary_advantage_gate": shuffled_gate,
        },
        "deployment_consensus_candidate": consensus,
        "route_decision_batch_160_mean_ms": decision_batch_ms,
        "pre_inference_batch_160_mean_ms": inference_batch_ms,
        "external_gate": gate,
        "confirmation_authorized": bool(gate["passed"]),
        "deep_swe_evaluation_authorized": False,
        "no_persisted_fitted_model": True,
        "target_outcomes_used": False,
        "target_embeddings_used": False,
        "inputs": {
            "corpus_sha256": _sha256(corpus),
            "outcomes_sha256": _sha256(outcomes),
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "outer-predictions.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "task_id": task_id,
                    "contest_id": data.groups[index],
                    "selected_arm": ARMS[int(choices[index])],
                    "predicted_uplift": float(predicted[index]),
                    "observed_uplift_vs_high": float(observed_uplift[index]),
                    "reward": float(router_reward[index]),
                    "matched_blind_reward": float(blind_reward[index]),
                    "target_outcomes_used": False,
                },
                sort_keys=True,
            )
            + "\n"
            for index, task_id in enumerate(data.task_ids)
        ),
        encoding="utf-8",
    )
    logger.info(
        "fit complete reward=%.4f cost=%.4f advantage=%.6f low=%.6f gate=%s",
        value["reward"],
        value["cost_usd"],
        value["advantage"],
        interval[0],
        gate["passed"],
    )


def main() -> None:
    """Parse the remote fitting command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--expected-tasks", type=int, default=EXPECTED_TASKS)
    args = parser.parse_args()
    fit(
        args.corpus,
        args.outcomes,
        args.output,
        seed=args.seed,
        expected_tasks=args.expected_tasks,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
