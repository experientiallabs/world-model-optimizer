"""Run fit-only grouped selection for latency-neutral BigCodeBench effort routers.

This module consumes only outer-fit rows. It deliberately has no heldout evaluation
entry point; the immutable selection lock in the sibling fitter is the boundary that
the later evaluation stage must cross.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from coding_model_router_bigcodebench_fit import (
    ARMS,
    CandidateMetric,
    FitData,
    PolicyValue,
    doubly_robust_pseudo_values,
    empirical_bayes_family_moments,
    empirical_bayes_ridge_predictions,
    evaluate_choices,
    feature_matrix,
    fit_selected_static,
    grouped_folds,
    lower_bound_choices,
    multi_action_hist_predictions,
    multi_action_ridge_predictions,
    ordinal_extra_trees_predictions,
    ordinal_ridge_predictions,
    outcome_matrix,
    select_fit_candidate,
    shadow_price_choices,
)
from scipy import sparse

from wmo.optimize.knn import fit_knn_policy
from wmo.optimize.policy import EmbedderSpec, knn_decision

Family = Literal["ordinal", "doubly-robust", "empirical-bayes"]
Estimator = Literal["ridge", "extra-trees", "histogram"]
HASH_DIMS = (512, 2_048, 8_192)
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
SHADOW_PRICES = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.04)


@dataclass(frozen=True)
class CandidateSpec:
    """One preregistered non-kNN candidate configuration."""

    family: Family
    estimator: Estimator
    dim: int
    order: int
    alpha: float = 0.0
    n_estimators: int = 0
    min_samples_leaf: int = 0
    max_features: Literal["", "sqrt", "third"] = ""
    max_leaf_nodes: int = 0
    learning_rate: float = 0.0
    lam: float = 0.0
    prior_strength: float = 0.0
    z: float = 0.0

    @property
    def name(self) -> str:
        """Return a deterministic readable identity for logs and the lock."""
        fields = [self.family, self.estimator, f"d{self.dim}"]
        for label, value in (
            ("a", self.alpha),
            ("trees", self.n_estimators),
            ("leaf", self.min_samples_leaf),
            ("feat", self.max_features),
            ("nodes", self.max_leaf_nodes),
            ("lr", self.learning_rate),
            ("lam", self.lam),
            ("prior", self.prior_strength),
            ("z", self.z),
        ):
            if value not in {0, ""}:
                fields.append(
                    f"{label}{value:g}" if isinstance(value, float) else f"{label}{value}"
                )
        return "-".join(fields)

    def config(self) -> dict[str, str | int | float | bool]:
        """Return the complete canonicalizable selection configuration."""
        return {
            "family": self.family,
            "estimator": self.estimator,
            "dim": self.dim,
            "alpha": self.alpha,
            "n_estimators": self.n_estimators,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "max_leaf_nodes": self.max_leaf_nodes,
            "learning_rate": self.learning_rate,
            "lam": self.lam,
            "prior_strength": self.prior_strength,
            "z": self.z,
        }


@dataclass(frozen=True)
class CandidateValidation:
    """Grouped out-of-fold value for one fit-only candidate."""

    spec: CandidateSpec | KnnCandidateSpec
    value: PolicyValue
    baseline: PolicyValue
    metric: CandidateMetric


@dataclass(frozen=True)
class KnnCandidateSpec:
    """One preregistered WMO kNN retrieval and statistical-guard point."""

    dim: int
    rag_num: int
    rag_thres: float
    z: float
    min_pairs: int
    order: int
    guard_model: str | None = None
    guard_mode: Literal["symmetric", "asymmetric"] = "symmetric"
    pick_lam: float = 0.0

    @property
    def name(self) -> str:
        """Return the stable grid identity."""
        identity = (
            f"knn-d{self.dim}-k{self.rag_num}-thres{self.rag_thres:g}"
            f"-z{self.z:g}-pairs{self.min_pairs}"
        )
        if self.guard_model is not None:
            identity += f"-guard-{self.guard_model}-{self.guard_mode}-lam{self.pick_lam:g}"
        return identity

    def config(self) -> dict[str, str | int | float | bool]:
        """Return the complete canonicalizable kNN configuration."""
        return {
            "family": "knn",
            "dim": self.dim,
            "rag_num": self.rag_num,
            "rag_thres": self.rag_thres,
            "z": self.z,
            "min_pairs": self.min_pairs,
            "guard_strategy": "fixed-arm" if self.guard_model is not None else "fit-best",
            "guard_model": self.guard_model or "fit-best",
            "guard_mode": self.guard_mode,
            "pick_lam": self.pick_lam,
        }


def candidate_grid() -> list[CandidateSpec]:
    """Enumerate the exact 576 preregistered non-kNN points."""
    candidates: list[CandidateSpec] = []

    def add(**values: str | int | float) -> None:
        candidates.append(CandidateSpec(order=len(candidates), **values))

    for dim in HASH_DIMS:
        for alpha in RIDGE_ALPHAS:
            add(family="ordinal", estimator="ridge", dim=dim, alpha=alpha)
        for n_estimators in (200, 500):
            for min_samples_leaf in (5, 10, 20):
                for max_features in ("sqrt", "third"):
                    add(
                        family="ordinal",
                        estimator="extra-trees",
                        dim=dim,
                        n_estimators=n_estimators,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                    )
        for alpha in RIDGE_ALPHAS:
            for lam in SHADOW_PRICES:
                add(
                    family="doubly-robust",
                    estimator="ridge",
                    dim=dim,
                    alpha=alpha,
                    lam=lam,
                )
        for max_leaf_nodes in (7, 15, 31):
            for learning_rate in (0.03, 0.1):
                for min_samples_leaf in (10, 20):
                    for lam in SHADOW_PRICES:
                        add(
                            family="doubly-robust",
                            estimator="histogram",
                            dim=dim,
                            max_leaf_nodes=max_leaf_nodes,
                            learning_rate=learning_rate,
                            min_samples_leaf=min_samples_leaf,
                            lam=lam,
                        )
        for prior_strength in (2.0, 5.0, 10.0, 20.0, 50.0):
            for alpha in RIDGE_ALPHAS:
                for z in (0.0, 0.5, 1.0, 1.645):
                    add(
                        family="empirical-bayes",
                        estimator="ridge",
                        dim=dim,
                        alpha=alpha,
                        prior_strength=prior_strength,
                        z=z,
                    )
    if len(candidates) != 576 or len({candidate.name for candidate in candidates}) != 576:
        raise AssertionError("non-kNN candidate grid is incomplete or has duplicate identities")
    return candidates


def knn_candidate_grid() -> list[KnnCandidateSpec]:
    """Enumerate the exact 432 preregistered WMO kNN base points."""
    candidates = [
        KnnCandidateSpec(
            dim=dim,
            rag_num=rag_num,
            rag_thres=rag_thres,
            z=z,
            min_pairs=min_pairs,
            order=576 + index,
        )
        for index, (dim, rag_num, rag_thres, z, min_pairs) in enumerate(
            (dim, rag_num, rag_thres, z, min_pairs)
            for dim in HASH_DIMS
            for rag_num in (8, 16, 32, 64)
            for rag_thres in (0.90, 0.95, 0.98)
            for z in (0.0, 0.5, 1.0, 1.645)
            for min_pairs in (8, 16, 32)
        )
    ]
    if len(candidates) != 432 or len({candidate.name for candidate in candidates}) != 432:
        raise AssertionError("kNN candidate grid is incomplete or has duplicate identities")
    return candidates


def knn_economic_grid(base: KnnCandidateSpec) -> list[KnnCandidateSpec]:
    """Enumerate 20 economic refinements around one selected kNN base point."""
    candidates = [
        KnnCandidateSpec(
            dim=base.dim,
            rag_num=base.rag_num,
            rag_thres=base.rag_thres,
            z=base.z,
            min_pairs=base.min_pairs,
            order=1_008 + index,
            guard_model=guard_model,
            guard_mode="asymmetric",
            pick_lam=pick_lam,
        )
        for index, (guard_model, pick_lam) in enumerate(
            (guard_model, pick_lam) for guard_model in ARMS for pick_lam in (0.0, 0.01, 0.02, 0.03)
        )
    ]
    if len(candidates) != 20 or len({candidate.name for candidate in candidates}) != 20:
        raise AssertionError("kNN economic grid is incomplete or has duplicate identities")
    return candidates


def _residual_standard_errors(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Return per-arm residual uncertainty with a small-sample binomial floor."""
    if observed.shape != predicted.shape or observed.ndim != 2:
        raise ValueError("residual uncertainty inputs differ")
    rmse = np.sqrt(np.mean((observed - predicted) ** 2, axis=0))
    floor = math.sqrt(0.25 / observed.shape[0])
    return np.maximum(rmse, floor)


def _candidate_choices(
    spec: CandidateSpec,
    data: FitData,
    train: np.ndarray,
    test: np.ndarray,
    train_features: sparse.csr_matrix,
    test_features: sparse.csr_matrix,
    *,
    seed: int,
) -> np.ndarray:
    train_rewards = data.rewards[train]
    observed = train_rewards.mean(axis=2)
    arm_costs = data.costs[train].mean(axis=(0, 2))
    baseline = fit_selected_static(data, train)
    fallback = ARMS.index(baseline.name)
    quality_floor = 0.95 * baseline.reward

    if spec.family == "ordinal":
        if spec.estimator == "ridge":
            train_predicted = ordinal_ridge_predictions(
                train_features,
                train_features,
                observed,
                alpha=spec.alpha,
            )
            predicted = ordinal_ridge_predictions(
                train_features,
                test_features,
                observed,
                alpha=spec.alpha,
            )
        else:
            train_predicted = ordinal_extra_trees_predictions(
                train_features,
                train_features,
                observed,
                n_estimators=spec.n_estimators,
                min_samples_leaf=spec.min_samples_leaf,
                max_features=spec.max_features,
                random_state=seed,
            )
            predicted = ordinal_extra_trees_predictions(
                train_features,
                test_features,
                observed,
                n_estimators=spec.n_estimators,
                min_samples_leaf=spec.min_samples_leaf,
                max_features=spec.max_features,
                random_state=seed,
            )
        uncertainty = np.broadcast_to(
            _residual_standard_errors(observed, train_predicted),
            predicted.shape,
        )
        return lower_bound_choices(
            predicted,
            uncertainty,
            arm_costs,
            quality_floor=quality_floor,
            fallback_arm=fallback,
            z=1.0,
        )

    if spec.family == "doubly-robust":
        direct = np.zeros_like(observed)
        pseudo = doubly_robust_pseudo_values(train_rewards, direct)
        if spec.estimator == "ridge":
            predicted = multi_action_ridge_predictions(
                train_features,
                test_features,
                pseudo,
                alpha=spec.alpha,
            )
        else:
            predicted = multi_action_hist_predictions(
                train_features,
                test_features,
                pseudo,
                max_leaf_nodes=spec.max_leaf_nodes,
                learning_rate=spec.learning_rate,
                min_samples_leaf=spec.min_samples_leaf,
                random_state=seed,
            )
        return shadow_price_choices(predicted, arm_costs, lam=spec.lam)

    _, _, _, posterior_se = empirical_bayes_family_moments(
        [data.groups[index] for index in train],
        [data.groups[index] for index in test],
        train_rewards,
        prior_strength=spec.prior_strength,
    )
    predicted = empirical_bayes_ridge_predictions(
        train_features,
        test_features,
        [data.groups[index] for index in train],
        [data.groups[index] for index in test],
        train_rewards,
        prior_strength=spec.prior_strength,
        alpha=spec.alpha,
    )
    return lower_bound_choices(
        predicted,
        posterior_se,
        arm_costs,
        quality_floor=quality_floor,
        fallback_arm=fallback,
        z=spec.z,
    )


def evaluate_candidate_oof(
    data: FitData,
    outer_fit: np.ndarray,
    spec: CandidateSpec,
    *,
    seed: int,
    feature_cache: dict[tuple[int, tuple[int, ...]], sparse.csr_matrix] | None = None,
    evaluation_data: FitData | None = None,
) -> CandidateValidation:
    """Fit one candidate in grouped folds and evaluate its routes on declared outcomes."""
    indices = np.asarray(outer_fit, dtype=np.int64)
    if indices.size == 0 or len(set(indices.tolist())) != len(indices):
        raise ValueError("outer fit indices are empty or duplicated")
    groups = [data.groups[index] for index in indices]
    choices = np.empty(len(indices), dtype=np.int64)
    baseline_choices = np.empty(len(indices), dtype=np.int64)
    cache = feature_cache if feature_cache is not None else {}
    for fold, (train_relative, test_relative) in enumerate(grouped_folds(groups)):
        train = indices[train_relative]
        test = indices[test_relative]
        key = (spec.dim, tuple(int(index) for index in train))
        features = cache.get(key)
        if features is None:
            features = feature_matrix(data, dim=spec.dim, scale_indices=train)
            cache[key] = features
        choices[test_relative] = _candidate_choices(
            spec,
            data,
            train,
            test,
            features[train],
            features[test],
            seed=seed * 100 + fold,
        )
        baseline_choices[test_relative] = ARMS.index(fit_selected_static(data, train).name)
    observed = evaluation_data or data
    if observed.task_ids != data.task_ids:
        raise ValueError("candidate evaluation data has different task identities")
    rewards = observed.rewards[indices].mean(axis=2)
    costs = observed.costs[indices].mean(axis=2)
    value = evaluate_choices(rewards, costs, choices)
    baseline = evaluate_choices(rewards, costs, baseline_choices)
    metric = CandidateMetric(
        name=spec.name,
        reward=value.reward,
        cost_usd=value.cost_usd,
        latency_p95_ms=0.0,
        artifact_bytes=0,
        order=spec.order,
    )
    return CandidateValidation(spec=spec, value=value, baseline=baseline, metric=metric)


def select_non_knn_candidate(
    data: FitData,
    outer_fit: np.ndarray,
    candidates: list[CandidateSpec],
    *,
    seed: int,
) -> tuple[CandidateValidation, list[CandidateValidation]]:
    """Select the least-cost fit-quality-feasible non-kNN point for one outer seed."""
    if not candidates:
        raise ValueError("non-kNN selection received no candidates")
    feature_cache: dict[tuple[int, tuple[int, ...]], sparse.csr_matrix] = {}
    results = [
        evaluate_candidate_oof(
            data,
            outer_fit,
            candidate,
            seed=seed,
            feature_cache=feature_cache,
        )
        for candidate in candidates
    ]
    baseline_reward = results[0].baseline.reward
    if any(not math.isclose(result.baseline.reward, baseline_reward) for result in results[1:]):
        raise AssertionError("candidate evaluations used different fit-only baselines")
    selected_metric = select_fit_candidate(
        [result.metric for result in results],
        baseline_reward=baseline_reward,
    )
    selected = next(result for result in results if result.metric.name == selected_metric.name)
    return selected, results


def _evaluate_knn_candidates(
    data: FitData,
    outer_fit: np.ndarray,
    candidates: list[KnnCandidateSpec],
    *,
    seed: int,
    work_dir: Path,
    evaluation_data: FitData | None = None,
) -> list[CandidateValidation]:
    """Evaluate WMO kNN points using five grouped outer-fit folds and shared banks."""
    if not candidates:
        raise ValueError("kNN selection received no candidates")
    indices = np.asarray(outer_fit, dtype=np.int64)
    if indices.size == 0 or len(set(indices.tolist())) != len(indices):
        raise ValueError("outer fit indices are empty or duplicated")
    unknown_dims = {candidate.dim for candidate in candidates} - set(HASH_DIMS)
    if unknown_dims:
        raise ValueError(f"kNN candidates use unfrozen dimensions: {sorted(unknown_dims)}")
    groups = [data.groups[index] for index in indices]
    choices = {candidate.name: np.empty(len(indices), dtype=np.int64) for candidate in candidates}
    baseline_choices = np.empty(len(indices), dtype=np.int64)
    matrix = outcome_matrix(data)
    by_dim = {
        dim: [candidate for candidate in candidates if candidate.dim == dim]
        for dim in sorted({candidate.dim for candidate in candidates})
    }
    for fold, (train_relative, test_relative) in enumerate(grouped_folds(groups)):
        train = indices[train_relative]
        test = indices[test_relative]
        baseline = fit_selected_static(data, train)
        baseline_choices[test_relative] = ARMS.index(baseline.name)
        train_ids = [data.task_ids[index] for index in train]
        test_texts = [data.texts[index] for index in test]
        for dim, dimensional_candidates in by_dim.items():
            embedder = EmbedderSpec(kind="hashing", dim=dim)
            policy = fit_knn_policy(
                matrix,
                bank_path=work_dir / f"seed-{seed}" / f"fold-{fold}-d{dim}.bank.npz",
                fit_ids=train_ids,
                embedder=embedder,
                guard_model=baseline.name,
                rag_num=64,
                rag_thres=0.90,
                z=0.0,
                min_pairs=8,
                se_floor=True,
                floor_q=0.0,
                pick_lam=0.0,
                fitted_from=f"bigcodebench inner seed={seed} fold={fold}",
            )
            vectors = np.asarray(embedder.build().embed(test_texts), dtype=np.float64)
            for candidate in dimensional_candidates:
                guard_model = candidate.guard_model or baseline.name
                tuned = policy.model_copy(
                    update={
                        "default_model": guard_model,
                        "guard_model": guard_model,
                        "rag_num": candidate.rag_num,
                        "rag_thres": candidate.rag_thres,
                        "knn_z": candidate.z,
                        "knn_min_pairs": candidate.min_pairs,
                        "guard_mode": candidate.guard_mode,
                        "pick_lam": candidate.pick_lam,
                    }
                )
                decisions = [knn_decision(tuned, vector).model for vector in vectors]
                choices[candidate.name][test_relative] = np.asarray(
                    [ARMS.index(model) for model in decisions],
                    dtype=np.int64,
                )
    observed = evaluation_data or data
    if observed.task_ids != data.task_ids:
        raise ValueError("kNN evaluation data has different task identities")
    rewards = observed.rewards[indices].mean(axis=2)
    costs = observed.costs[indices].mean(axis=2)
    baseline_value = evaluate_choices(rewards, costs, baseline_choices)
    results: list[CandidateValidation] = []
    for candidate in candidates:
        value = evaluate_choices(rewards, costs, choices[candidate.name])
        results.append(
            CandidateValidation(
                spec=candidate,
                value=value,
                baseline=baseline_value,
                metric=CandidateMetric(
                    name=candidate.name,
                    reward=value.reward,
                    cost_usd=value.cost_usd,
                    latency_p95_ms=0.0,
                    artifact_bytes=0,
                    order=candidate.order,
                ),
            )
        )
    return results


def select_knn_candidate(
    data: FitData,
    outer_fit: np.ndarray,
    candidates: list[KnnCandidateSpec],
    *,
    seed: int,
    work_dir: Path,
    evaluation_data: FitData | None = None,
) -> tuple[CandidateValidation, list[CandidateValidation]]:
    """Select one WMO kNN point using five grouped outer-fit folds and shared banks."""
    results = _evaluate_knn_candidates(
        data,
        outer_fit,
        candidates,
        seed=seed,
        work_dir=work_dir,
        evaluation_data=evaluation_data,
    )
    selected_metric = select_fit_candidate(
        [result.metric for result in results],
        baseline_reward=results[0].baseline.reward,
    )
    selected = next(result for result in results if result.metric.name == selected_metric.name)
    return selected, results


def select_knn_economic_refinement(
    data: FitData,
    outer_fit: np.ndarray,
    base: CandidateValidation,
    *,
    seed: int,
    work_dir: Path,
) -> tuple[CandidateValidation, list[CandidateValidation]]:
    """Select the base or one of its 20 fit-only kNN economic refinements."""
    if not isinstance(base.spec, KnnCandidateSpec):
        raise TypeError("kNN economic refinement requires a selected kNN base")
    refinements = _evaluate_knn_candidates(
        data,
        outer_fit,
        knn_economic_grid(base.spec),
        seed=seed,
        work_dir=work_dir,
    )
    baseline_reward = base.baseline.reward
    if any(not math.isclose(result.baseline.reward, baseline_reward) for result in refinements):
        raise AssertionError("economic refinements used a different fit-only baseline")
    options = [base, *refinements]
    selected_metric = select_fit_candidate(
        [result.metric for result in options],
        baseline_reward=baseline_reward,
    )
    selected = next(result for result in options if result.metric.name == selected_metric.name)
    return selected, refinements
