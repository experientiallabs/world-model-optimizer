"""Develop native guarded kNN effort routes on external Codeforces outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from coding_model_router_codeforces_fit import ARMS, Data, load_data
from sklearn.model_selection import GroupKFold

from wmo.optimize.knn import fit_knn_policy
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.policy import EmbedderSpec, RoutingPolicy
from wmo.optimize.routing import route_scenarios
from wmo.providers.base import Embedder, ProviderKind
from wmo.providers.pool import PoolEntry

logger = logging.getLogger("coding-model-router-codeforces-knn")

DIMS = (512, 2_048)
RAG_NUMS = (8, 16, 32, 50)
RAG_THRESHOLDS = (0.9, 0.95)
Z_VALUES = (0.0, 0.5, 1.0, 2.0)
PICK_LAMS = (0.0, 0.01, 0.02, 0.03)
BOOTSTRAPS = 2_000


@dataclass(frozen=True)
class Candidate:
    """One native kNN policy point in the external development grid."""

    dim: int
    guard: str
    rag_num: int
    rag_threshold: float
    z: float
    pick_lam: float

    @property
    def key(self) -> str:
        """Return the stable configuration identity."""
        return (
            f"hash{self.dim}-guard-{self.guard}-k{self.rag_num}"
            f"-th{self.rag_threshold:g}-z{self.z:g}-lam{self.pick_lam:g}"
        )


@dataclass(frozen=True)
class FoldValue:
    """Held-out routed and matched-mixture observations for one fold."""

    indices: np.ndarray
    choices: np.ndarray
    reward: np.ndarray
    cost: np.ndarray
    blind_reward: np.ndarray
    blind_cost: np.ndarray


def candidate_grid() -> tuple[Candidate, ...]:
    """Return the bounded native kNN development grid."""
    return tuple(
        Candidate(dim, guard, rag_num, rag_threshold, z, pick_lam)
        for dim in DIMS
        for guard in ARMS
        for rag_num in RAG_NUMS
        for rag_threshold in RAG_THRESHOLDS
        for z in Z_VALUES
        for pick_lam in PICK_LAMS
    )


def _pool() -> list[PoolEntry]:
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


def _matrix(data: Data, rewards: np.ndarray | None = None) -> OutcomeMatrix:
    values = data.rewards if rewards is None else rewards
    outcomes = [
        ScenarioOutcome(
            scenario_id=task_id,
            task=data.texts[task_index],
            model=arm,
            benchmark="codeforces-cots-effort",
            episode=0,
            attempt_number=1,
            reward=float(values[task_index, arm_index]),
            success=bool(values[task_index, arm_index] >= 1.0),
            cost_usd=float(data.costs[task_index, arm_index]),
            completion_status="scored",
            usage_accounting="trace-estimated",
        )
        for task_index, task_id in enumerate(data.task_ids)
        for arm_index, arm in enumerate(ARMS)
    ]
    return OutcomeMatrix(pool=_pool(), outcomes=outcomes)


def _policy(
    base: RoutingPolicy,
    candidate: Candidate,
) -> RoutingPolicy:
    return base.model_copy(
        update={
            "default_model": candidate.guard,
            "guard_model": candidate.guard,
            "rag_num": candidate.rag_num,
            "rag_thres": candidate.rag_threshold,
            "knn_z": candidate.z,
            "pick_lam": candidate.pick_lam,
            "guard_mode": "asymmetric",
        }
    )


def _value(data: Data, indices: np.ndarray, choices: np.ndarray) -> FoldValue:
    rewards = data.rewards[indices]
    costs = data.costs[indices]
    fractions = np.bincount(choices, minlength=len(ARMS)) / len(choices)
    rows = np.arange(len(indices))
    return FoldValue(
        indices=indices,
        choices=choices,
        reward=rewards[rows, choices],
        cost=costs[rows, choices],
        blind_reward=rewards @ fractions,
        blind_cost=costs @ fractions,
    )


def _routes(
    data: Data,
    matrix: OutcomeMatrix,
    policy: RoutingPolicy,
    test: np.ndarray,
    embedder: Embedder,
) -> FoldValue:
    ids = [data.task_ids[index] for index in test]
    decisions = route_scenarios(policy, matrix, ids, embedder=embedder)
    arm_index = {arm: index for index, arm in enumerate(ARMS)}
    choices = np.asarray([arm_index[decisions[task_id].model] for task_id in ids])
    return _value(data, test, choices)


def _aggregate(folds: list[FoldValue]) -> dict[str, Any]:
    choices = np.concatenate([fold.choices for fold in folds])
    reward = np.concatenate([fold.reward for fold in folds])
    cost = np.concatenate([fold.cost for fold in folds])
    blind_reward = np.concatenate([fold.blind_reward for fold in folds])
    blind_cost = np.concatenate([fold.blind_cost for fold in folds])
    counts = np.bincount(choices, minlength=len(ARMS))
    return {
        "reward": float(reward.mean()),
        "cost_usd": float(cost.sum()),
        "matched_blind_reward": float(blind_reward.mean()),
        "matched_blind_cost_usd": float(blind_cost.sum()),
        "advantage": float((reward - blind_reward).mean()),
        "counts": {arm: int(counts[index]) for index, arm in enumerate(ARMS)},
        "reward_by_task": reward,
        "blind_reward_by_task": blind_reward,
    }


def _bootstrap(
    data: Data,
    folds: list[FoldValue],
    *,
    seed: int,
) -> list[float]:
    index = np.concatenate([fold.indices for fold in folds])
    routed = np.concatenate([fold.reward for fold in folds])
    blind = np.concatenate([fold.blind_reward for fold in folds])
    groups = np.asarray(data.groups, dtype=object)[index]
    unique = sorted(set(str(group) for group in groups))
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    values = np.empty(BOOTSTRAPS, dtype=np.float64)
    for sample in range(BOOTSTRAPS):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([members[str(group)] for group in selected])
        values[sample] = float(np.mean(routed[rows] - blind[rows]))
    return [float(value) for value in np.quantile(values, [0.025, 0.5, 0.975])]


def _static(data: Data) -> list[dict[str, Any]]:
    return [
        {
            "arm": arm,
            "reward": float(data.rewards[:, index].mean()),
            "cost_usd": float(data.costs[:, index].sum()),
        }
        for index, arm in enumerate(ARMS)
    ]


def _dominated(row: dict[str, Any], static: list[dict[str, Any]]) -> list[str]:
    return [
        str(point["arm"])
        for point in static
        if float(point["reward"]) >= float(row["reward"])
        and float(point["cost_usd"]) <= float(row["cost_usd"])
        and (
            float(point["reward"]) > float(row["reward"])
            or float(point["cost_usd"]) < float(row["cost_usd"])
        )
    ]


def _shuffled_control(
    data: Data,
    candidate: Candidate,
    folds: list[tuple[np.ndarray, np.ndarray]],
    root: Path,
    *,
    seed: int,
) -> list[FoldValue]:
    results: list[FoldValue] = []
    spec = EmbedderSpec(kind="hashing", dim=candidate.dim)
    embedder = spec.build()
    for fold_index, (train, test) in enumerate(folds):
        shuffled = data.rewards.copy()
        permutation = np.random.default_rng(seed + fold_index).permutation(train)
        shuffled[train] = data.rewards[permutation]
        matrix = _matrix(data, shuffled)
        base = fit_knn_policy(
            matrix,
            bank_path=root / f"shuffle-{fold_index}.npz",
            fit_ids=[data.task_ids[index] for index in train],
            embedder=spec,
            embed_with=embedder,
            guard_model=candidate.guard,
            rag_num=candidate.rag_num,
            rag_thres=candidate.rag_threshold,
            z=candidate.z,
            min_pairs=8,
            se_floor=True,
            floor_q=0.0,
            pick_lam=candidate.pick_lam,
            fitted_from="Codeforces shuffled-label development control",
        )
        results.append(_routes(data, matrix, _policy(base, candidate), test, embedder))
    return results


def develop(
    corpus: Path,
    outcomes: Path,
    output: Path,
    *,
    expected_tasks: int,
    seed: int,
) -> None:
    """Evaluate the bounded native grid with contest-grouped out-of-fold replay."""
    data = load_data(corpus, outcomes, expected_tasks=expected_tasks)
    matrix = _matrix(data)
    indices = np.arange(len(data.task_ids))
    folds = list(GroupKFold(n_splits=5).split(indices, groups=np.asarray(data.groups)))
    candidates = candidate_grid()
    by_candidate: dict[str, list[FoldValue]] = defaultdict(list)
    with tempfile.TemporaryDirectory(prefix="codeforces-knn-") as directory:
        root = Path(directory)
        for fold_index, (train, test) in enumerate(folds):
            if set(np.asarray(data.groups)[train]) & set(np.asarray(data.groups)[test]):
                raise AssertionError("contest group crossed a development fold")
            for dim in DIMS:
                spec = EmbedderSpec(kind="hashing", dim=dim)
                embedder = spec.build()
                base = fit_knn_policy(
                    matrix,
                    bank_path=root / f"fold-{fold_index}-hash{dim}.npz",
                    fit_ids=[data.task_ids[index] for index in train],
                    embedder=spec,
                    embed_with=embedder,
                    guard_model=ARMS[0],
                    rag_num=50,
                    rag_thres=0.95,
                    z=0.0,
                    min_pairs=8,
                    se_floor=True,
                    floor_q=0.0,
                    pick_lam=0.0,
                    fitted_from="Codeforces external development only",
                )
                for candidate in candidates:
                    if candidate.dim != dim:
                        continue
                    by_candidate[candidate.key].append(
                        _routes(data, matrix, _policy(base, candidate), test, embedder)
                    )
            logger.info("completed grouped fold %d/%d", fold_index + 1, len(folds))
        static = _static(data)
        rows: list[dict[str, Any]] = []
        for candidate in candidates:
            value = _aggregate(by_candidate[candidate.key])
            row = {
                "candidate": candidate.__dict__,
                "key": candidate.key,
                "reward": value["reward"],
                "cost_usd": value["cost_usd"],
                "matched_blind_reward": value["matched_blind_reward"],
                "matched_blind_cost_usd": value["matched_blind_cost_usd"],
                "advantage": value["advantage"],
                "counts": value["counts"],
            }
            row["dominated_by_static_arms"] = _dominated(row, static)
            rows.append(row)
        nondominated = [
            row
            for row in rows
            if not row["dominated_by_static_arms"] and float(row["advantage"]) > 0.0
        ]
        selected_row = max(
            nondominated or rows,
            key=lambda row: (
                float(row["reward"]) - 0.002 * math.log1p(float(row["cost_usd"])),
                float(row["advantage"]),
                -float(row["cost_usd"]),
                str(row["key"]),
            ),
        )
        selected = next(
            candidate for candidate in candidates if candidate.key == selected_row["key"]
        )
        interval = _bootstrap(data, by_candidate[selected.key], seed=seed)
        shuffled_folds = _shuffled_control(data, selected, folds, root, seed=seed + 10_000)
        shuffled = _aggregate(shuffled_folds)
        shuffled_interval = _bootstrap(data, shuffled_folds, seed=seed + 20_000)

    selected_row["contest_bootstrap_advantage_95ci"] = interval
    selected_row["dominated_by_static_arms"] = _dominated(selected_row, static)
    gate = {
        "positive_matched_blind_advantage": float(selected_row["advantage"]) > 0.0,
        "positive_contest_bootstrap_lower_bound": interval[0] > 0.0,
        "not_static_dominated": not selected_row["dominated_by_static_arms"],
        "shuffled_control_failed": not (
            float(shuffled["advantage"]) > 0.0 and shuffled_interval[0] > 0.0
        ),
    }
    gate["passed"] = all(gate.values())
    report = {
        "protocol": "codeforces-native-knn-development-v1",
        "tasks": len(data.task_ids),
        "contest_groups": len(set(data.groups)),
        "candidate_count": len(candidates),
        "selection_rule": (
            "maximize reward minus 0.002 times log1p(total cost), then advantage and cost"
        ),
        "static_efforts": static,
        "selected": selected_row,
        "shuffled_control": {
            "reward": shuffled["reward"],
            "cost_usd": shuffled["cost_usd"],
            "advantage": shuffled["advantage"],
            "contest_bootstrap_advantage_95ci": shuffled_interval,
        },
        "development_gate": gate,
        "confirmation_authorized": False,
        "target_outcomes_used": False,
        "inputs": {
            "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
            "outcomes_sha256": hashlib.sha256(outcomes.read_bytes()).hexdigest(),
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output / "candidate-grid.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    logger.info(
        "development complete candidate=%s reward=%.4f cost=%.4f advantage=%.5f low=%.5f",
        selected.key,
        selected_row["reward"],
        selected_row["cost_usd"],
        selected_row["advantage"],
        interval[0],
    )


def main() -> None:
    """Parse the external development command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()
    develop(
        args.corpus,
        args.outcomes,
        args.output,
        expected_tasks=args.expected_tasks,
        seed=args.seed,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
