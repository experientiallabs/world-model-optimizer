"""Evaluate a RADAR-inspired reasoning-effort router on Codeforces outcomes.

The router fits a regularized two-parameter item-response model to graded
source rewards, predicts task difficulty and discrimination from pre-call text,
and selects a reasoning-effort arm through a frozen performance-cost
scalarization. All selection uses nested contest-grouped validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
from coding_model_router_codeforces_fit import (
    ARMS,
    BOOTSTRAPS,
    Data,
    _bootstrap,
    _features,
    _spearman,
    _value,
    load_data,
)
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

logger = logging.getLogger("coding-model-router-codeforces-irt")

SEED = 20_260_731
QUALITY_TOLERANCE = 0.005
DIMENSIONS = (512, 2_048)
RIDGE_ALPHAS = (1.0, 10.0)
SCALARIZATION_WEIGHTS = (0.70, 0.80, 0.90, 0.95, 0.98, 0.99)
SCALARIZATIONS: tuple[Literal["linear", "chebyshev"], ...] = (
    "linear",
    "chebyshev",
)
IRT_L2 = 0.01
DISCRIMINATION_L2 = 0.05


@dataclass(frozen=True)
class IrtFit:
    """Latent arm abilities and task response characteristics."""

    abilities: np.ndarray
    difficulties: np.ndarray
    log_discriminations: np.ndarray
    loss: float
    iterations: int


@dataclass(frozen=True)
class Candidate:
    """One frozen representation and performance-cost scalarization."""

    dimension: int
    alpha: float
    scalarization: Literal["linear", "chebyshev"]
    weight: float

    @property
    def name(self) -> str:
        """Return a stable candidate label."""
        return f"irt2pl-hash{self.dimension}-a{self.alpha:g}-{self.scalarization}-w{self.weight:g}"


CANDIDATES = tuple(
    Candidate(dimension, alpha, scalarization, weight)
    for dimension in DIMENSIONS
    for alpha in RIDGE_ALPHAS
    for scalarization in SCALARIZATIONS
    for weight in SCALARIZATION_WEIGHTS
)


@dataclass(frozen=True)
class InnerFold:
    """Cached held-out IRT probabilities and fit-only costs for one fold."""

    test_local: np.ndarray
    probabilities: dict[tuple[int, float], np.ndarray]
    mean_costs: np.ndarray
    static_arm: int


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected a numeric report value")
    return float(value)


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-4, 1.0 - 1e-4)
    return np.log(clipped / (1.0 - clipped))


def _irt_loss_and_gradient(
    parameters: np.ndarray,
    rewards: np.ndarray,
) -> tuple[float, np.ndarray]:
    task_count, arm_count = rewards.shape
    abilities = parameters[:arm_count]
    difficulties = parameters[arm_count : arm_count + task_count]
    log_discriminations = parameters[arm_count + task_count :]
    discriminations = np.exp(log_discriminations)
    logits = discriminations[:, None] * (abilities[None, :] - difficulties[:, None])
    loss = float(np.mean(np.logaddexp(0.0, logits) - rewards * logits))
    loss += IRT_L2 * float(np.mean(abilities**2) + np.mean(difficulties**2))
    loss += DISCRIMINATION_L2 * float(np.mean(log_discriminations**2))

    error = (expit(logits) - rewards) / rewards.size
    ability_gradient = np.sum(error * discriminations[:, None], axis=0)
    ability_gradient += 2.0 * IRT_L2 * abilities / arm_count
    difficulty_gradient = -np.sum(error * discriminations[:, None], axis=1)
    difficulty_gradient += 2.0 * IRT_L2 * difficulties / task_count
    discrimination_gradient = np.sum(error * logits, axis=1)
    discrimination_gradient += 2.0 * DISCRIMINATION_L2 * log_discriminations / task_count
    gradient = np.concatenate([ability_gradient, difficulty_gradient, discrimination_gradient])
    return loss, gradient


def _fit_irt(rewards: np.ndarray) -> IrtFit:
    """Fit a regularized graded-response 2PL model without task-text leakage."""
    if rewards.ndim != 2 or rewards.shape[1] != len(ARMS):
        raise ValueError("IRT rewards must be a dense task-by-arm matrix")
    if not np.isfinite(rewards).all() or np.any((rewards < 0.0) | (rewards > 1.0)):
        raise ValueError("IRT rewards must be finite fractions in [0, 1]")
    task_count, arm_count = rewards.shape
    initial_abilities = _logit(np.mean(rewards, axis=0))
    initial_abilities -= np.mean(initial_abilities)
    initial_difficulties = -_logit(np.mean(rewards, axis=1))
    initial_difficulties -= np.mean(initial_difficulties)
    initial = np.concatenate([initial_abilities, initial_difficulties, np.zeros(task_count)])
    bounds = [(-8.0, 8.0)] * (arm_count + task_count) + [(-3.0, 3.0)] * task_count
    result = minimize(
        _irt_loss_and_gradient,
        initial,
        args=(rewards,),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 1_000, "ftol": 1e-10, "gtol": 1e-7},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"IRT optimization failed: {result.message}")
    parameters = np.asarray(result.x, dtype=np.float64)
    return IrtFit(
        abilities=parameters[:arm_count],
        difficulties=parameters[arm_count : arm_count + task_count],
        log_discriminations=parameters[arm_count + task_count :],
        loss=float(result.fun),
        iterations=int(result.nit),
    )


def _predict_probabilities(
    data: Data,
    train: np.ndarray,
    test: np.ndarray,
    *,
    dimension: int,
    alpha: float,
    label_rewards: np.ndarray | None = None,
) -> tuple[np.ndarray, IrtFit]:
    latent, predictor = _fit_text_policy(
        data,
        train,
        dimension=dimension,
        alpha=alpha,
        label_rewards=label_rewards,
    )
    return _score_text_policy(data, train, test, dimension, latent, predictor), latent


def _fit_text_policy(
    data: Data,
    train: np.ndarray,
    *,
    dimension: int,
    alpha: float,
    label_rewards: np.ndarray | None = None,
) -> tuple[IrtFit, Ridge]:
    labels = data.rewards if label_rewards is None else label_rewards
    latent = _fit_irt(labels[train])
    features = _features(data, dimension, train)
    targets = np.column_stack([latent.difficulties, latent.log_discriminations])
    predictor = Ridge(alpha=alpha, solver="lsqr", max_iter=500, tol=1e-5)
    predictor.fit(features[train], targets)
    return latent, predictor


def _score_text_policy(
    data: Data,
    train: np.ndarray,
    test: np.ndarray,
    dimension: int,
    latent: IrtFit,
    predictor: Ridge,
) -> np.ndarray:
    features = _features(data, dimension, train)
    predicted = np.asarray(predictor.predict(features[test]), dtype=np.float64)
    difficulties = np.clip(predicted[:, 0], -8.0, 8.0)
    discriminations = np.exp(np.clip(predicted[:, 1], -3.0, 3.0))
    logits = discriminations[:, None] * (latent.abilities[None, :] - difficulties[:, None])
    return expit(logits)


def _normalized_costs(mean_costs: np.ndarray) -> np.ndarray:
    low = float(np.min(mean_costs))
    span = float(np.max(mean_costs)) - low
    if span <= 0.0:
        return np.zeros_like(mean_costs)
    return (mean_costs - low) / span


def _choose(
    probabilities: np.ndarray,
    mean_costs: np.ndarray,
    candidate: Candidate,
) -> np.ndarray:
    """Select arms with the paper's linear or Chebyshev scalarization."""
    costs = _normalized_costs(mean_costs)
    if candidate.scalarization == "linear":
        objective = candidate.weight * probabilities - (1.0 - candidate.weight) * costs
        target = np.max(objective, axis=1)
        maximize = True
    else:
        objective = np.maximum(
            candidate.weight * (1.0 - probabilities),
            (1.0 - candidate.weight) * costs,
        )
        target = np.min(objective, axis=1)
        maximize = False
    choices = np.empty(len(probabilities), dtype=np.int64)
    for index, row in enumerate(objective):
        candidates = np.flatnonzero(np.isclose(row, target[index], atol=1e-12))
        if not len(candidates):
            candidates = np.asarray(
                [int(np.argmax(row) if maximize else np.argmin(row))],
                dtype=np.int64,
            )
        choices[index] = int(candidates[int(np.argmin(mean_costs[candidates]))])
    return choices


def _best_static(data: Data, indices: np.ndarray) -> int:
    means = np.mean(data.rewards[indices], axis=0)
    best = float(np.max(means))
    candidates = np.flatnonzero(np.isclose(means, best, atol=1e-12))
    mean_costs = np.mean(data.costs[indices], axis=0)
    return int(candidates[int(np.argmin(mean_costs[candidates]))])


def _inner_folds(data: Data, outer_train: np.ndarray) -> list[InnerFold]:
    groups = np.asarray(data.groups, dtype=object)[outer_train]
    result: list[InnerFold] = []
    for fit_local, test_local in GroupKFold(n_splits=4).split(
        outer_train,
        groups=groups,
    ):
        fit = outer_train[fit_local]
        test = outer_train[test_local]
        cache: dict[tuple[int, float], np.ndarray] = {}
        for dimension in DIMENSIONS:
            for alpha in RIDGE_ALPHAS:
                cache[(dimension, alpha)] = _predict_probabilities(
                    data,
                    fit,
                    test,
                    dimension=dimension,
                    alpha=alpha,
                )[0]
        result.append(
            InnerFold(
                test_local=test_local,
                probabilities=cache,
                mean_costs=np.mean(data.costs[fit], axis=0),
                static_arm=_best_static(data, fit),
            )
        )
    return result


def _candidate_value(
    data: Data,
    outer_train: np.ndarray,
    folds: list[InnerFold],
    candidate: Candidate,
) -> tuple[dict[str, object], float, float]:
    choices = np.full(len(outer_train), -1, dtype=np.int64)
    static_choices = np.full(len(outer_train), -1, dtype=np.int64)
    for fold in folds:
        choices[fold.test_local] = _choose(
            fold.probabilities[(candidate.dimension, candidate.alpha)],
            fold.mean_costs,
            candidate,
        )
        static_choices[fold.test_local] = fold.static_arm
    if np.any(choices < 0) or np.any(static_choices < 0):
        raise RuntimeError("inner grouped predictions are incomplete")
    value = cast(dict[str, object], _value(data, outer_train, choices))
    row = np.arange(len(outer_train))
    static_reward = float(np.mean(data.rewards[outer_train][row, static_choices]))
    static_cost = float(np.sum(data.costs[outer_train][row, static_choices]))
    return value, static_reward, static_cost


def _select_candidate(
    data: Data,
    outer_train: np.ndarray,
) -> tuple[Candidate, list[dict[str, object]]]:
    folds = _inner_folds(data, outer_train)
    rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        value, static_reward, static_cost = _candidate_value(
            data,
            outer_train,
            folds,
            candidate,
        )
        reward = _number(value["reward"])
        cost = _number(value["cost_usd"])
        advantage = _number(value["advantage"])
        feasible = (
            reward >= static_reward - QUALITY_TOLERANCE and cost < static_cost and advantage > 0.0
        )
        rows.append(
            {
                "name": candidate.name,
                "dimension": candidate.dimension,
                "alpha": candidate.alpha,
                "scalarization": candidate.scalarization,
                "weight": candidate.weight,
                "reward": reward,
                "cost_usd": cost,
                "matched_blind_advantage": advantage,
                "fit_selected_static_reward": static_reward,
                "fit_selected_static_cost_usd": static_cost,
                "feasible": feasible,
            }
        )
    feasible = [row for row in rows if bool(row["feasible"])]
    if feasible:
        selected = min(
            feasible,
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
                float(row["matched_blind_advantage"]),
                float(row["reward"]) - float(row["fit_selected_static_reward"]),
                -float(row["cost_usd"]),
                str(row["name"]),
            ),
        )
    return next(candidate for candidate in CANDIDATES if candidate.name == selected["name"]), rows


def _shuffle_training_rewards(
    data: Data,
    train: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    result = data.rewards.copy()
    permutation = np.random.default_rng(seed).permutation(train)
    result[train] = data.rewards[permutation]
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(
    corpus: Path,
    outcomes: Path,
    output: Path,
    *,
    expected_tasks: int,
    seed: int = SEED,
) -> dict[str, object]:
    """Run nested contest-grouped IRT development and write a sealed report."""
    data = load_data(corpus, outcomes, expected_tasks=expected_tasks)
    all_indices = np.arange(len(data.task_ids), dtype=np.int64)
    groups = np.asarray(data.groups, dtype=object)
    choices = np.full(len(data.task_ids), -1, dtype=np.int64)
    static_choices = np.full(len(data.task_ids), -1, dtype=np.int64)
    shuffled_choices = np.full(len(data.task_ids), -1, dtype=np.int64)
    predicted_uplift = np.zeros(len(data.task_ids), dtype=np.float64)
    ability_rows: list[dict[str, float]] = []
    fold_rows: list[dict[str, object]] = []
    for fold, (train, test) in enumerate(GroupKFold(n_splits=5).split(all_indices, groups=groups)):
        if set(groups[train]) & set(groups[test]):
            raise AssertionError("contest group crossed an outer fold")
        candidate, inner_rows = _select_candidate(data, train)
        probabilities, latent = _predict_probabilities(
            data,
            train,
            test,
            dimension=candidate.dimension,
            alpha=candidate.alpha,
        )
        mean_costs = np.mean(data.costs[train], axis=0)
        choices[test] = _choose(probabilities, mean_costs, candidate)
        static_arm = _best_static(data, train)
        static_choices[test] = static_arm
        predicted_uplift[test] = (
            probabilities[np.arange(len(test)), choices[test]] - probabilities[:, static_arm]
        )
        shuffled_probabilities, _ = _predict_probabilities(
            data,
            train,
            test,
            dimension=candidate.dimension,
            alpha=candidate.alpha,
            label_rewards=_shuffle_training_rewards(data, train, seed=seed + fold),
        )
        shuffled_choices[test] = _choose(shuffled_probabilities, mean_costs, candidate)
        ability_rows.append({arm: float(latent.abilities[index]) for index, arm in enumerate(ARMS)})
        fold_rows.append(
            {
                "fold": fold,
                "train_tasks": len(train),
                "test_tasks": len(test),
                "train_contests": len(set(groups[train])),
                "test_contests": len(set(groups[test])),
                "contest_overlap": 0,
                "selected_candidate": candidate.name,
                "selected_static_arm": ARMS[static_arm],
                "irt_loss": latent.loss,
                "irt_iterations": latent.iterations,
                "inner_candidates": inner_rows,
            }
        )
    if np.any(choices < 0) or np.any(static_choices < 0) or np.any(shuffled_choices < 0):
        raise RuntimeError("outer grouped predictions are incomplete")

    value = cast(dict[str, object], _value(data, all_indices, choices))
    shuffled = cast(dict[str, object], _value(data, all_indices, shuffled_choices))
    row = np.arange(len(data.task_ids))
    routed_rewards = cast(np.ndarray, value["routed_reward_by_task"])
    blind_rewards = cast(np.ndarray, value["blind_reward_by_task"])
    shuffled_rewards = cast(np.ndarray, shuffled["routed_reward_by_task"])
    shuffled_blind = cast(np.ndarray, shuffled["blind_reward_by_task"])
    static_rewards = data.rewards[row, static_choices]
    observed_uplift = routed_rewards - static_rewards
    interval = _bootstrap(data.groups, routed_rewards, blind_rewards, seed=seed)
    shuffled_interval = _bootstrap(
        data.groups,
        shuffled_rewards,
        shuffled_blind,
        seed=seed + 1,
    )
    static = [
        {
            "arm": arm,
            "reward": float(np.mean(data.rewards[:, index])),
            "cost_usd": float(np.sum(data.costs[:, index])),
        }
        for index, arm in enumerate(ARMS)
    ]
    dominated_by = [
        str(row_value["arm"])
        for row_value in static
        if _number(row_value["reward"]) >= _number(value["reward"])
        and _number(row_value["cost_usd"]) <= _number(value["cost_usd"])
        and (
            _number(row_value["reward"]) > _number(value["reward"])
            or _number(row_value["cost_usd"]) < _number(value["cost_usd"])
        )
    ]
    shuffled_gate = _number(shuffled["advantage"]) > 0.0 and shuffled_interval[0] > 0.0
    gate: dict[str, object] = {
        "positive_oof_uplift_spearman": _spearman(
            predicted_uplift,
            observed_uplift,
        )
        > 0.0,
        "positive_matched_blind_advantage": _number(value["advantage"]) > 0.0,
        "positive_contest_bootstrap_lower_bound": interval[0] > 0.0,
        "not_static_dominated": not dominated_by,
        "shuffled_control_failed": not shuffled_gate,
        "complete_and_target_sealed": bool(np.all(np.isfinite(routed_rewards))),
    }
    gate["passed"] = all(bool(item) for item in gate.values())
    consensus_name = max(
        {str(row_value["selected_candidate"]) for row_value in fold_rows},
        key=lambda name: (
            sum(row_value["selected_candidate"] == name for row_value in fold_rows),
            name,
        ),
    )
    consensus = next(candidate for candidate in CANDIDATES if candidate.name == consensus_name)
    full_latent, full_predictor = _fit_text_policy(
        data,
        all_indices,
        dimension=consensus.dimension,
        alpha=consensus.alpha,
    )
    started = time.perf_counter_ns()
    for _ in range(100):
        probabilities = _score_text_policy(
            data,
            all_indices,
            all_indices,
            consensus.dimension,
            full_latent,
            full_predictor,
        )
        _choose(probabilities, np.mean(data.costs, axis=0), consensus)
    inference_batch_ms = (time.perf_counter_ns() - started) / 1_000_000 / 100
    report: dict[str, object] = {
        "protocol": "codeforces-radar-irt-nested-v1",
        "paper": "https://arxiv.org/abs/2509.25426",
        "adaptation": "graded-response 2PL with sparse pre-call text heads",
        "tasks": len(data.task_ids),
        "contest_groups": len(set(data.groups)),
        "arms": list(ARMS),
        "candidate_count": len(CANDIDATES),
        "static_efforts": static,
        "nested_outer_folds": fold_rows,
        "outer_fold_abilities": ability_rows,
        "router": {
            "reward": value["reward"],
            "cost_usd": value["cost_usd"],
            "arm_counts": value["counts"],
            "matched_blind_reward": value["matched_blind_reward"],
            "matched_blind_cost_usd": value["matched_blind_cost_usd"],
            "advantage_vs_matched_blind": value["advantage"],
            "predicted_uplift_spearman": _spearman(
                predicted_uplift,
                observed_uplift,
            ),
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
        "deployment_consensus_candidate": consensus_name,
        "deployment_consensus_abilities": {
            arm: float(full_latent.abilities[index]) for index, arm in enumerate(ARMS)
        },
        "pre_inference_batch_160_mean_ms": inference_batch_ms,
        "development_gate": gate,
        "confirmation_authorized": bool(gate["passed"]),
        "deep_swe_evaluation_authorized": False,
        "target_outcomes_used": False,
        "target_embeddings_used": False,
        "no_persisted_fitted_model": True,
        "bootstrap_samples": BOOTSTRAPS,
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
    with (output / "outer-predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, task_id in enumerate(data.task_ids):
            handle.write(
                json.dumps(
                    {
                        "task_id": task_id,
                        "contest_id": data.groups[index],
                        "selected_arm": ARMS[int(choices[index])],
                        "fit_selected_static_arm": ARMS[int(static_choices[index])],
                        "predicted_uplift": float(predicted_uplift[index]),
                        "observed_uplift": float(observed_uplift[index]),
                        "reward": float(routed_rewards[index]),
                        "matched_blind_reward": float(blind_rewards[index]),
                        "target_outcomes_used": False,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    logger.info(
        "IRT development complete reward=%.4f cost=%.4f advantage=%.6f low=%.6f gate=%s",
        _number(value["reward"]),
        _number(value["cost_usd"]),
        _number(value["advantage"]),
        interval[0],
        gate["passed"],
    )
    return report


def main() -> None:
    """Parse the remote IRT development command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, default=160)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    fit(
        args.corpus.resolve(),
        args.outcomes.resolve(),
        args.output.resolve(),
        expected_tasks=args.expected_tasks,
        seed=args.seed,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
