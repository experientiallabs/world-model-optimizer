"""Select a trace-burden-aware Open-SWE effort router with nested CV.

The base uplift model is the externally selected hybrid Ridge scorer. This
experiment subtracts a preregistered multiple of the BeyondSWE trace-burden
score and chooses the cheapest proxy route that retains at least 90 percent of
the best positive inner-fold reward advantage. DeepSWE is never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import coding_model_router_open_swe_hurdle as hurdle
import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

logger = logging.getLogger("coding-router-open-swe-cost-aware")

TRAFFIC_FRACTION = 0.20
PENALTIES = (0.0, 0.025, 0.05, 0.10, 0.20)
REWARD_RETENTION = 0.90
BOOTSTRAP_SAMPLES = 2_000


@dataclass(frozen=True)
class Metrics:
    penalty: float
    reward_advantage: float
    proxy_cost_delta_per_task: float
    selected_proxy_mean: float
    uplift_spearman: float


def _base_candidate() -> hurdle.Candidate:
    return hurdle.Candidate(
        "hybrid-ridge-direct-a10",
        "hybrid",
        "direct-ridge",
        alpha=10.0,
    )


def _fit_base(
    data: hurdle.Data,
    features: sparse.csr_matrix,
    train: np.ndarray,
    test: np.ndarray,
) -> np.ndarray:
    model = Ridge(alpha=10.0)
    model.fit(
        features[train],
        data.uplift[train],
        sample_weight=data.precision[train],
    )
    return np.asarray(model.predict(features[test]), dtype=np.float64)


def _scores(base: np.ndarray, auxiliary: np.ndarray, penalty: float) -> np.ndarray:
    return base - penalty * auxiliary


def _metrics(
    data: hurdle.Data,
    indices: np.ndarray,
    base: np.ndarray,
    penalty: float,
) -> Metrics:
    scores = _scores(base, data.auxiliary[indices], penalty)
    count = max(1, int(round(TRAFFIC_FRACTION * len(indices))))
    order = np.argsort(-scores, kind="mergesort")
    chosen = order[:count]
    routed = data.cheap[indices].copy()
    routed[chosen] = data.strong[indices[chosen]]
    blind = data.cheap[indices] + TRAFFIC_FRACTION * data.uplift[indices]
    proxy = np.exp(np.clip(data.auxiliary[indices], -3.0, 3.0))
    proxy_delta = float(
        (np.sum(proxy[chosen]) - TRAFFIC_FRACTION * np.sum(proxy)) / len(indices)
    )
    return Metrics(
        penalty=penalty,
        reward_advantage=float(np.mean(routed) - np.mean(blind)),
        proxy_cost_delta_per_task=proxy_delta,
        selected_proxy_mean=float(np.mean(proxy[chosen])),
        uplift_spearman=hurdle._spearman(scores, data.uplift[indices]),
    )


def _select(rows: list[Metrics]) -> Metrics:
    best_reward = max(row.reward_advantage for row in rows)
    eligible = [
        row
        for row in rows
        if row.reward_advantage > 0.0
        and row.reward_advantage >= REWARD_RETENTION * best_reward
    ]
    if not eligible:
        return max(
            rows,
            key=lambda row: (
                row.reward_advantage,
                -row.proxy_cost_delta_per_task,
                -row.penalty,
            ),
        )
    return min(
        eligible,
        key=lambda row: (
            row.proxy_cost_delta_per_task,
            -row.reward_advantage,
            row.penalty,
        ),
    )


def _oof_base(
    data: hurdle.Data,
    features: sparse.csr_matrix,
    indices: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    groups = data.groups[indices]
    splitter = GroupKFold(n_splits=5)
    predictions = np.zeros(len(indices), dtype=np.float64)
    audits: list[dict[str, int]] = []
    for fold, (local_train, local_test) in enumerate(
        splitter.split(indices, groups=groups)
    ):
        train = indices[local_train]
        test = indices[local_test]
        overlap = set(data.groups[train]) & set(data.groups[test])
        if overlap:
            raise ValueError(f"fold {fold} has repository overlap")
        predictions[local_test] = _fit_base(data, features, train, test)
        audits.append(
            {
                "fold": fold,
                "train_tasks": len(train),
                "test_tasks": len(test),
                "group_overlap": len(overlap),
            }
        )
    return predictions, audits


def _bootstrap(
    data: hurdle.Data,
    scores: np.ndarray,
    *,
    seed: int,
) -> dict[str, list[float]]:
    groups = sorted(set(cast(list[str], data.groups.tolist())))
    by_group = {
        group: np.flatnonzero(data.groups == group)
        for group in groups
    }
    rng = np.random.default_rng(seed)
    reward = np.zeros(BOOTSTRAP_SAMPLES, dtype=np.float64)
    proxy = np.zeros(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for sample in range(BOOTSTRAP_SAMPLES):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([by_group[str(group)] for group in sampled])
        count = max(1, int(round(TRAFFIC_FRACTION * len(indices))))
        order = np.argsort(-scores[indices], kind="mergesort")
        chosen = order[:count]
        routed = data.cheap[indices].copy()
        routed[chosen] = data.strong[indices[chosen]]
        blind = data.cheap[indices] + TRAFFIC_FRACTION * data.uplift[indices]
        reward[sample] = float(np.mean(routed) - np.mean(blind))
        burden = np.exp(np.clip(data.auxiliary[indices], -3.0, 3.0))
        proxy[sample] = float(
            (
                np.sum(burden[chosen])
                - TRAFFIC_FRACTION * np.sum(burden)
            )
            / len(indices)
        )
    return {
        "reward_advantage_95ci": [
            float(value) for value in np.quantile(reward, [0.025, 0.5, 0.975])
        ],
        "proxy_cost_delta_per_task_95ci": [
            float(value) for value in np.quantile(proxy, [0.025, 0.5, 0.975])
        ],
    }


def fit(
    paired_path: Path,
    auxiliary_path: Path,
    external_report_path: Path,
    output: Path,
    *,
    seed: int,
) -> None:
    external = json.loads(external_report_path.read_text(encoding="utf-8"))
    if (
        not isinstance(external, dict)
        or not isinstance(external.get("external_gate"), dict)
        or cast(dict[str, object], external["external_gate"]).get("passed") is not True
        or not isinstance(external.get("selected_final_candidate"), dict)
        or cast(dict[str, object], external["selected_final_candidate"]).get("candidate")
        != "hybrid-ridge-direct-a10"
    ):
        raise ValueError("base external gate or frozen candidate is invalid")
    data = hurdle._load_data(paired_path, auxiliary_path)
    candidate = _base_candidate()
    features = hurdle._features(data, candidate)
    indices = np.arange(len(data.task_ids), dtype=np.int64)
    nested_base = np.zeros(len(indices), dtype=np.float64)
    nested_penalty = np.zeros(len(indices), dtype=np.float64)
    outer_rows: list[dict[str, object]] = []
    outer = GroupKFold(n_splits=5)
    for fold, (train, test) in enumerate(outer.split(indices, groups=data.groups)):
        overlap = set(data.groups[train]) & set(data.groups[test])
        if overlap:
            raise ValueError(f"outer fold {fold} has repository overlap")
        inner_base, _ = _oof_base(data, features, train)
        candidates = [
            _metrics(data, train, inner_base, penalty)
            for penalty in PENALTIES
        ]
        selected = _select(candidates)
        base = _fit_base(data, features, train, test)
        nested_base[test] = base
        nested_penalty[test] = selected.penalty
        outer_rows.append(
            {
                "fold": fold,
                "train_tasks": len(train),
                "test_tasks": len(test),
                "group_overlap": len(overlap),
                "selected_penalty": selected.__dict__,
                "inner_candidates": [row.__dict__ for row in candidates],
            }
        )
    nested_scores = nested_base - nested_penalty * data.auxiliary
    nested_count = max(1, int(round(TRAFFIC_FRACTION * len(indices))))
    nested_order = np.argsort(-nested_scores, kind="mergesort")
    nested_chosen = nested_order[:nested_count]
    nested_router = data.cheap.copy()
    nested_router[nested_chosen] = data.strong[nested_chosen]
    nested_blind = data.cheap + TRAFFIC_FRACTION * data.uplift
    nested_proxy = np.exp(np.clip(data.auxiliary, -3.0, 3.0))
    nested_metrics = {
        "reward_advantage": float(np.mean(nested_router) - np.mean(nested_blind)),
        "proxy_cost_delta_per_task": float(
            (
                np.sum(nested_proxy[nested_chosen])
                - TRAFFIC_FRACTION * np.sum(nested_proxy)
            )
            / len(indices)
        ),
        "uplift_spearman": hurdle._spearman(nested_scores, data.uplift),
        "outer_penalty_counts": dict(
            sorted(
                {
                    str(value): int(np.sum(nested_penalty == value))
                    for value in sorted(set(nested_penalty.tolist()))
                }.items()
            )
        ),
    }
    full_base, audits = _oof_base(data, features, indices)
    full_candidates = [
        _metrics(data, indices, full_base, penalty)
        for penalty in PENALTIES
    ]
    selected_final = _select(full_candidates)
    interval = _bootstrap(data, nested_scores, seed=seed)
    reward_interval = interval["reward_advantage_95ci"]
    proxy_interval = interval["proxy_cost_delta_per_task_95ci"]
    gate = {
        "selected_penalty_positive": selected_final.penalty > 0.0,
        "nested_reward_advantage_positive": (
            float(nested_metrics["reward_advantage"]) > 0.0
        ),
        "nested_proxy_cost_delta_negative": (
            float(nested_metrics["proxy_cost_delta_per_task"]) < 0.0
        ),
        "nested_reward_bootstrap_lower_bound_positive": reward_interval[0] > 0.0,
        "nested_proxy_bootstrap_upper_bound_negative": proxy_interval[2] < 0.0,
    }
    gate["passed"] = all(bool(value) for value in gate.values())
    report = {
        "protocol": "open-swe-cost-aware-uplift-nested-v1",
        "adaptive_after_full_prompt_target_diagnostic": True,
        "target_data_used": False,
        "tasks": len(data.task_ids),
        "repositories": len(set(cast(list[str], data.groups.tolist()))),
        "traffic_fraction": TRAFFIC_FRACTION,
        "penalty_candidates": list(PENALTIES),
        "selection_rule": {
            "minimum_best_reward_retention": REWARD_RETENTION,
            "then_minimize_trace_burden_proxy": True,
        },
        "outer_folds": outer_rows,
        "nested_external_test": nested_metrics,
        "nested_group_bootstrap": interval,
        "full_source_candidates": [row.__dict__ for row in full_candidates],
        "selected_final": selected_final.__dict__,
        "selected_final_fold_audits": audits,
        "external_gate": gate,
        "deep_swe_evaluation_authorized": bool(gate["passed"]),
        "no_persisted_fitted_model": True,
        "inputs": {
            "paired_sha256": hashlib.sha256(paired_path.read_bytes()).hexdigest(),
            "auxiliary_scores_sha256": hashlib.sha256(
                auxiliary_path.read_bytes()
            ).hexdigest(),
            "base_external_report_sha256": hashlib.sha256(
                external_report_path.read_bytes()
            ).hexdigest(),
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "cost-aware external fit penalty=%.3f reward=%.6f proxy_delta=%.6f "
        "reward_low=%.6f proxy_high=%.6f gate=%s",
        selected_final.penalty,
        float(nested_metrics["reward_advantage"]),
        float(nested_metrics["proxy_cost_delta_per_task"]),
        reward_interval[0],
        proxy_interval[2],
        gate["passed"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--auxiliary-scores", type=Path, required=True)
    parser.add_argument("--external-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    fit(
        args.paired,
        args.auxiliary_scores,
        args.external_report,
        args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
