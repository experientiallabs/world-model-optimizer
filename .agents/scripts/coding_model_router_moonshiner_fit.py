"""Fit and gate external-only effort routers on Moonshiner outcomes.

The script joins the five Luna reasoning efforts for the frozen Moonshiner
corpus, evaluates candidate task scorers with grouped out-of-fold predictions,
and compares every routed operating point with a matched task-blind mixture.
It never reads DeepSWE artifacts.
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
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

logger = logging.getLogger("coding-model-router-moonshiner-fit")

EFFORTS = ("low", "medium", "high", "xhigh", "max")
ARMS = tuple(f"luna-{effort}" for effort in EFFORTS)
TRAFFIC_FRACTIONS = (0.05, 0.10, 0.20, 0.40)
BOOTSTRAP_SAMPLES = 2_000


@dataclass(frozen=True)
class Candidate:
    name: str
    kind: Literal["direct", "latent", "monotone-heads", "control"]
    dim: int
    alpha: float
    shuffled: bool = False


@dataclass(frozen=True)
class Data:
    task_ids: list[str]
    groups: list[str]
    texts: list[str]
    rewards: np.ndarray
    costs: np.ndarray
    structural: np.ndarray


CANDIDATES = (
    Candidate("task-blind-control", "control", 1, 1.0),
    Candidate("shuffled-direct-hash512-a10", "direct", 512, 10.0, shuffled=True),
    Candidate("direct-hash512-a10", "direct", 512, 10.0),
    Candidate("direct-hash2048-a10", "direct", 2_048, 10.0),
    Candidate("latent-hash512-a10", "latent", 512, 10.0),
    Candidate("latent-hash2048-a10", "latent", 2_048, 10.0),
    Candidate("monotone-heads-hash512-a10", "monotone-heads", 512, 10.0),
    Candidate("monotone-heads-hash2048-a10", "monotone-heads", 2_048, 10.0),
)


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


def _family(task_id: str, category: str) -> str:
    if task_id.startswith("bash-it-"):
        return "bash-it"
    match = re.match(
        r"^(behavior-(?:dependency-planning|error-recovery|format-sensitivity))",
        task_id,
    )
    if match:
        return match.group(1)
    if task_id.startswith("vcf91-"):
        return "vcf91"
    match = re.match(r"^(w18-sc\d+)", task_id)
    if match:
        return match.group(1)
    return f"category:{category}"


def _structural_row(task: dict[str, object]) -> list[float]:
    prompt = str(task["prompt"])
    words = prompt.split()
    lines = prompt.splitlines()
    verify = str(task.get("verify_cmd") or "")
    language = str(task.get("language") or "")
    return [
        math.log1p(len(prompt)),
        math.log1p(len(words)),
        math.log1p(len(lines)),
        math.log1p(int(cast(int, task.get("fixture_files") or 0))),
        math.log1p(int(cast(int, task.get("fixture_bytes") or 0))),
        math.log1p(int(cast(int, task.get("verify_timeout_s") or 0))),
        float(prompt.count("`")),
        float(prompt.count("\n")),
        float("test" in prompt.lower()),
        float("incident" in prompt.lower()),
        float("debug" in prompt.lower()),
        float("performance" in prompt.lower()),
        float("security" in prompt.lower()),
        float("unittest" in verify),
        float("pytest" in verify),
        float("bash" in verify),
        float(language in {"bash", "zsh"}),
        float(language in {"py", "python"}),
        float(language in {"c", "cpp"}),
        float(language in {"ts", "typescript", "js"}),
    ]


def _load_data(corpus_path: Path, outcome_paths: list[Path]) -> Data:
    corpus = _read_object(corpus_path)
    raw_tasks = corpus.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError(f"{corpus_path} has no tasks")
    tasks = [
        {str(key): item for key, item in task.items()}
        for task in raw_tasks
        if isinstance(task, dict)
    ]
    if len(tasks) != len(raw_tasks):
        raise ValueError(f"{corpus_path} contains an invalid task")
    cells: dict[tuple[str, str], dict[int, dict[str, object]]] = {}
    for path in outcome_paths:
        for row in _read_rows(path):
            task_id = row.get("task_id")
            arm = row.get("arm")
            attempt = row.get("attempt")
            if (
                not isinstance(task_id, str)
                or not isinstance(arm, str)
                or not isinstance(attempt, int)
                or isinstance(attempt, bool)
            ):
                raise ValueError(f"{path} contains an invalid cell")
            key = (task_id, arm)
            attempts = cells.setdefault(key, {})
            if attempt in attempts:
                raise ValueError(f"duplicate outcome cell: {key} attempt={attempt}")
            if (
                row.get("model_attested") is not True
                or row.get("protected_intact") is not True
                or row.get("target_outcomes_used") is not False
            ):
                raise ValueError(f"ungradeable outcome cell: {key}")
            attempts[attempt] = row
    task_ids = [str(task["task_id"]) for task in tasks]
    rewards = np.zeros((len(tasks), len(ARMS)), dtype=np.float64)
    costs = np.zeros_like(rewards)
    for task_index, task_id in enumerate(task_ids):
        for arm_index, arm in enumerate(ARMS):
            attempts = cells.get((task_id, arm))
            if attempts is None:
                raise ValueError(f"missing outcome cell: {(task_id, arm)}")
            if set(attempts) != {0, 1, 2}:
                raise ValueError(
                    f"incomplete attempts for {(task_id, arm)}: {sorted(attempts)}"
                )
            rewards[task_index, arm_index] = float(
                np.mean([float(cast(float, row["reward"])) for row in attempts.values()])
            )
            costs[task_index, arm_index] = float(
                np.mean(
                    [float(cast(float, row["cost_usd"])) for row in attempts.values()]
                )
            )
    return Data(
        task_ids=task_ids,
        groups=[
            _family(str(task["task_id"]), str(task.get("category") or "unknown"))
            for task in tasks
        ],
        texts=[str(task["prompt"]) for task in tasks],
        rewards=rewards,
        costs=costs,
        structural=np.asarray([_structural_row(task) for task in tasks], dtype=np.float64),
    )


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


def _features(data: Data, dim: int) -> sparse.csr_matrix:
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        n_features=dim,
        alternate_sign=True,
        norm="l2",
    )
    text = cast(sparse.csr_matrix, vectorizer.transform(data.texts))
    structural = data.structural.copy()
    scale = np.maximum(np.std(structural, axis=0), 1.0)
    structural /= scale
    return sparse.hstack([text, sparse.csr_matrix(structural)], format="csr")


def _latent_scores(
    train_features: sparse.csr_matrix,
    test_features: sparse.csr_matrix,
    train_rewards: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    difficulty = 1.0 - train_rewards.mean(axis=1)
    model = Ridge(alpha=alpha)
    model.fit(train_features, difficulty)
    train_difficulty = np.asarray(model.predict(train_features), dtype=np.float64)
    test_difficulty = np.asarray(model.predict(test_features), dtype=np.float64)
    calibrations: list[np.ndarray] = []
    for arm_index in (3, 4):
        calibration = IsotonicRegression(increasing=False, out_of_bounds="clip")
        calibration.fit(train_difficulty, train_rewards[:, arm_index])
        calibrations.append(
            np.asarray(calibration.predict(test_difficulty), dtype=np.float64)
        )
    xhigh, maximum = calibrations
    return np.maximum(maximum, xhigh) - xhigh


def _monotone_head_scores(
    train_features: sparse.csr_matrix,
    test_features: sparse.csr_matrix,
    train_rewards: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    for arm_index in range(len(ARMS)):
        model = Ridge(alpha=alpha)
        model.fit(train_features, train_rewards[:, arm_index])
        predictions.append(
            np.asarray(model.predict(test_features), dtype=np.float64)
        )
    matrix = np.clip(np.column_stack(predictions), 0.0, 1.0)
    monotone = np.maximum.accumulate(matrix, axis=1)
    return monotone[:, 4] - monotone[:, 3]


def _oof_scores(data: Data, candidate: Candidate, *, seed: int) -> np.ndarray:
    direct = data.rewards[:, 4] - data.rewards[:, 3]
    labels = direct.copy()
    if candidate.shuffled:
        labels = labels[np.random.default_rng(seed).permutation(len(labels))]
    if candidate.kind == "control":
        return np.zeros(len(data.task_ids), dtype=np.float64)
    features = _features(data, candidate.dim)
    predictions = np.zeros(len(data.task_ids), dtype=np.float64)
    splitter = GroupKFold(n_splits=5)
    for train, test in splitter.split(features, groups=np.asarray(data.groups)):
        train_features = features[train]
        test_features = features[test]
        if candidate.kind == "direct":
            model = Ridge(alpha=candidate.alpha)
            model.fit(train_features, labels[train])
            predictions[test] = model.predict(test_features)
        elif candidate.kind == "latent":
            predictions[test] = _latent_scores(
                train_features,
                test_features,
                data.rewards[train],
                alpha=candidate.alpha,
            )
        else:
            predictions[test] = _monotone_head_scores(
                train_features,
                test_features,
                data.rewards[train],
                alpha=candidate.alpha,
            )
    return predictions


def _operating_points(
    data: Data,
    scores: np.ndarray,
) -> list[dict[str, float | int]]:
    xhigh = data.rewards[:, 3]
    maximum = data.rewards[:, 4]
    delta = maximum - xhigh
    order = np.argsort(-scores, kind="mergesort")
    rows: list[dict[str, float | int]] = []
    for fraction in TRAFFIC_FRACTIONS:
        count = max(1, int(round(fraction * len(scores))))
        routed = xhigh.copy()
        routed[order[:count]] = maximum[order[:count]]
        blind = xhigh + fraction * delta
        rows.append(
            {
                "max_traffic_fraction": fraction,
                "max_traffic_tasks": count,
                "router_reward": float(routed.mean()),
                "matched_blind_reward": float(blind.mean()),
                "advantage_vs_matched_blind": float(routed.mean() - blind.mean()),
            }
        )
    return rows


def _bootstrap_advantage(
    data: Data,
    scores: np.ndarray,
    *,
    fraction: float,
    seed: int,
) -> list[float]:
    groups = sorted(set(data.groups))
    group_indices = {
        group: np.flatnonzero(np.asarray(data.groups) == group) for group in groups
    }
    rng = np.random.default_rng(seed)
    xhigh = data.rewards[:, 3]
    maximum = data.rewards[:, 4]
    values = np.zeros(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for sample_index in range(BOOTSTRAP_SAMPLES):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([group_indices[str(group)] for group in sampled_groups])
        count = max(1, int(round(fraction * len(indices))))
        local_order = np.argsort(-scores[indices], kind="mergesort")
        routed = xhigh[indices].copy()
        routed[local_order[:count]] = maximum[indices[local_order[:count]]]
        blind = xhigh[indices] + fraction * (maximum[indices] - xhigh[indices])
        values[sample_index] = routed.mean() - blind.mean()
    return [float(value) for value in np.quantile(values, [0.025, 0.5, 0.975])]


def _monotonicity(data: Data) -> dict[str, object]:
    violations: list[str] = []
    patterns: dict[str, int] = {}
    for task_id, row in zip(data.task_ids, data.rewards, strict=True):
        pattern = "".join(str(int(value)) for value in row)
        patterns[pattern] = patterns.get(pattern, 0) + 1
        if any(row[left] > row[right] for left in range(5) for right in range(left + 1, 5)):
            violations.append(task_id)
    return {
        "tasks_with_monotonicity_violation": len(violations),
        "violation_task_ids": violations,
        "patterns": dict(sorted(patterns.items())),
    }


def fit(
    corpus: Path,
    outcomes: list[Path],
    output: Path,
    *,
    seed: int,
) -> None:
    data = _load_data(corpus, outcomes)
    direct_uplift = data.rewards[:, 4] - data.rewards[:, 3]
    leaderboard: list[dict[str, object]] = []
    score_bank: dict[str, np.ndarray] = {}
    for candidate in CANDIDATES:
        scores = _oof_scores(data, candidate, seed=seed)
        score_bank[candidate.name] = scores
        operating_points = _operating_points(data, scores)
        best_point = max(
            operating_points,
            key=lambda row: float(row["advantage_vs_matched_blind"]),
        )
        leaderboard.append(
            {
                "candidate": candidate.name,
                "kind": candidate.kind,
                "dim": candidate.dim,
                "alpha": candidate.alpha,
                "shuffled": candidate.shuffled,
                "oof_uplift_spearman": _spearman(scores, direct_uplift),
                "score_std": float(np.std(scores)),
                "operating_points": operating_points,
                "best_advantage_vs_matched_blind": best_point[
                    "advantage_vs_matched_blind"
                ],
                "best_max_traffic_fraction": best_point["max_traffic_fraction"],
            }
        )
    eligible = [
        row
        for row in leaderboard
        if row["kind"] != "control" and row["shuffled"] is not True
    ]
    eligible.sort(
        key=lambda row: (
            -float(cast(float, row["best_advantage_vs_matched_blind"])),
            -float(cast(float, row["oof_uplift_spearman"])),
            str(row["candidate"]),
        )
    )
    selected = eligible[0]
    selected_name = str(selected["candidate"])
    selected_fraction = float(cast(float, selected["best_max_traffic_fraction"]))
    interval = _bootstrap_advantage(
        data,
        score_bank[selected_name],
        fraction=selected_fraction,
        seed=seed,
    )
    gate = {
        "positive_oof_uplift_spearman": (
            float(cast(float, selected["oof_uplift_spearman"])) > 0.0
        ),
        "positive_matched_blind_advantage": (
            float(cast(float, selected["best_advantage_vs_matched_blind"])) > 0.0
        ),
        "positive_group_bootstrap_lower_bound": interval[0] > 0.0,
    }
    gate["passed"] = all(bool(value) for value in gate.values())
    effort_rows = [
        {
            "arm": arm,
            "passes": int(data.rewards[:, index].sum()),
            "pass_rate": float(data.rewards[:, index].mean()),
            "spend_usd": float(data.costs[:, index].sum()),
        }
        for index, arm in enumerate(ARMS)
    ]
    report = {
        "protocol": "moonshiner-five-effort-external-fit-v1",
        "tasks": len(data.task_ids),
        "groups": len(set(data.groups)),
        "efforts": effort_rows,
        "direct_xhigh_to_max_uplift_mean": float(direct_uplift.mean()),
        "direct_uplift_counts": {
            "max_better": int(np.sum(direct_uplift > 0)),
            "equal": int(np.sum(direct_uplift == 0)),
            "xhigh_better": int(np.sum(direct_uplift < 0)),
        },
        "monotonicity": _monotonicity(data),
        "candidate_space": [
            {
                "name": candidate.name,
                "kind": candidate.kind,
                "dim": candidate.dim,
                "alpha": candidate.alpha,
                "shuffled": candidate.shuffled,
            }
            for candidate in CANDIDATES
        ],
        "leaderboard": leaderboard,
        "selected_candidate": selected,
        "selected_group_bootstrap_advantage_95ci": interval,
        "external_gate": gate,
        "target_outcomes_used": False,
        "target_embeddings_used": False,
        "deep_swe_evaluation_authorized": bool(gate["passed"]),
        "inputs": {
            "corpus_sha256": _sha256(corpus),
            "outcomes": {
                str(path): _sha256(path)
                for path in outcomes
            },
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    score_path = output / "oof-scores.jsonl"
    score_path.write_text(
        "".join(
            json.dumps(
                {
                    "task_id": task_id,
                    "group": data.groups[index],
                    "direct_uplift": float(direct_uplift[index]),
                    "scores": {
                        candidate.name: float(score_bank[candidate.name][index])
                        for candidate in CANDIDATES
                    },
                },
                sort_keys=True,
            )
            + "\n"
            for index, task_id in enumerate(data.task_ids)
        ),
        encoding="utf-8",
    )
    logger.info(
        "fit complete tasks=%d groups=%d selected=%s rho=%.4f advantage=%.6f "
        "bootstrap_low=%.6f gate=%s",
        len(data.task_ids),
        len(set(data.groups)),
        selected_name,
        float(cast(float, selected["oof_uplift_spearman"])),
        float(cast(float, selected["best_advantage_vs_matched_blind"])),
        interval[0],
        gate["passed"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    fit(args.corpus, args.outcomes, args.output, seed=args.seed)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
