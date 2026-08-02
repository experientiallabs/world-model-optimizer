"""Pure robust-selection helpers for the conditional graded IRT router study.

This module turns out-of-fold, pre-call arm predictions into deterministic guarded routes and
evaluates those routes under the frozen repository-level forward-KL uncertainty set. It performs
no fitting, network access, filesystem access, or serialization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from coding_model_router_graded_irt_core import kl_robust_lower_bound


@dataclass(frozen=True)
class RepositoryRobustMetrics:
    """Worst-case reward and cost for one deterministic routing policy."""

    reward_lower_bound: float
    cost_upper_bound: float
    repository_rewards: np.ndarray
    repository_costs: np.ndarray


@dataclass(frozen=True)
class RepositoryRobustMargin:
    """Worst-case lower bound for one paired task-level margin."""

    lower_bound: float
    repository_margins: np.ndarray


def _validate_arm_matrices(probabilities: np.ndarray, costs: np.ndarray) -> None:
    """Validate aligned task-by-arm prediction and pre-call cost matrices."""
    if (
        probabilities.ndim != 2
        or probabilities.shape[0] < 1
        or probabilities.shape[1] < 2
        or costs.shape != probabilities.shape
    ):
        raise ValueError("probabilities and costs must be aligned task-by-arm matrices")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("arm probabilities must be finite values in [0, 1]")
    if not np.isfinite(costs).all() or np.any(costs < 0.0):
        raise ValueError("arm costs must be finite and nonnegative")


def quality_guarded_choices(
    probabilities: np.ndarray,
    costs: np.ndarray,
    *,
    guard_arm: int,
    cost_penalty: float,
    quality_floor: float = 0.95,
) -> np.ndarray:
    """Choose a cost-pressured arm and revert unsafe predictions to the static guard.

    Costs are normalized independently per task, preserving task-specific scale differences.
    The penalty creates the frozen cost-quality candidate path. A non-guard choice is accepted
    only when its predicted reward is at least ``quality_floor`` times the guard prediction.
    """
    _validate_arm_matrices(probabilities, costs)
    arm_count = probabilities.shape[1]
    if not 0 <= guard_arm < arm_count:
        raise ValueError("guard arm index is outside the arm matrix")
    if not np.isfinite(cost_penalty) or cost_penalty < 0.0:
        raise ValueError("cost penalty must be finite and nonnegative")
    if not np.isfinite(quality_floor) or not 0.0 < quality_floor <= 1.0:
        raise ValueError("quality floor must be finite and in (0, 1]")

    minimum = np.min(costs, axis=1, keepdims=True)
    span = np.max(costs, axis=1, keepdims=True) - minimum
    normalized_costs = np.divide(
        costs - minimum,
        span,
        out=np.zeros_like(costs, dtype=np.float64),
        where=span > 0.0,
    )
    objective = probabilities - cost_penalty * normalized_costs
    choices = np.empty(len(probabilities), dtype=np.int64)
    for task_index, row in enumerate(objective):
        best = float(np.max(row))
        candidates = np.flatnonzero(np.isclose(row, best, atol=1e-12, rtol=0.0))
        choices[task_index] = min(
            (int(candidate) for candidate in candidates),
            key=lambda candidate: (costs[task_index, candidate], candidate),
        )
    rows = np.arange(len(probabilities), dtype=np.int64)
    safe = probabilities[rows, choices] >= (
        quality_floor * probabilities[:, guard_arm] - 1e-12
    )
    choices[~safe] = guard_arm
    return choices


def _repository_means(values: np.ndarray, repositories: np.ndarray) -> np.ndarray:
    """Aggregate a finite task vector into deterministic equal-repository means."""
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("repository values must be a nonempty finite task vector")
    if repositories.shape != values.shape:
        raise ValueError("repository identities must align with task rows")
    repository_values = tuple(str(value) for value in repositories)
    if any(not value for value in repository_values):
        raise ValueError("repository identities must be nonempty strings")
    repository_array = np.asarray(repository_values, dtype=object)
    return np.asarray(
        [
            float(np.mean(values[repository_array == repository]))
            for repository in sorted(set(repository_values))
        ],
        dtype=np.float64,
    )


def repository_robust_margin(
    values: np.ndarray,
    repositories: np.ndarray,
    *,
    radius: float,
) -> RepositoryRobustMargin:
    """Return the forward-KL lower bound for a paired task-level margin."""
    repository_margins = _repository_means(values, repositories)
    return RepositoryRobustMargin(
        lower_bound=kl_robust_lower_bound(repository_margins, radius),
        repository_margins=repository_margins,
    )


def repository_robust_metrics(
    rewards: np.ndarray,
    costs: np.ndarray,
    choices: np.ndarray,
    repositories: np.ndarray,
    *,
    radius: float,
) -> RepositoryRobustMetrics:
    """Evaluate one routed policy under an equal-repository forward-KL ball."""
    _validate_arm_matrices(rewards, costs)
    if choices.shape != (len(rewards),) or not np.issubdtype(choices.dtype, np.integer):
        raise ValueError("choices must be an integer arm index per task")
    if np.any((choices < 0) | (choices >= rewards.shape[1])):
        raise ValueError("route choices contain an invalid arm index")
    rows = np.arange(len(rewards), dtype=np.int64)
    routed_rewards = rewards[rows, choices]
    routed_costs = costs[rows, choices]
    reward_array = _repository_means(routed_rewards, repositories)
    cost_array = _repository_means(routed_costs, repositories)
    return RepositoryRobustMetrics(
        reward_lower_bound=kl_robust_lower_bound(reward_array, radius),
        cost_upper_bound=-kl_robust_lower_bound(-cost_array, radius),
        repository_rewards=reward_array,
        repository_costs=cost_array,
    )
