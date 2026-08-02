"""Build and latency-audit a fit-selected numeric BigCodeBench router."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Literal, TypedDict, cast

import joblib
import numpy as np
from coding_model_router_bigcodebench_fit import (
    ARMS,
    FitData,
    LatencyMetric,
    _structural_row,
    artifact_size,
    doubly_robust_pseudo_values,
    empirical_bayes_family_moments,
    empirical_bayes_family_predictions,
    fit_selected_static,
    load_fit_data,
    lower_bound_choices,
    measure_route_latency,
    outer_splits,
    shadow_price_choices,
)
from coding_model_router_bigcodebench_lock import SeedWinnerAudit
from coding_model_router_bigcodebench_select import (
    CandidateSpec,
    _candidate_choices,
    _residual_standard_errors,
)
from coding_model_router_bigcodebench_select_run import CandidateRecord, SeedFitReport
from scipy import sparse
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge

from wmo.core.files import write_text_atomic

logger = logging.getLogger(__name__)
NumericFamily = Literal["ordinal", "doubly-robust", "empirical-bayes"]
EstimatorModel = Ridge | ExtraTreesRegressor | HistGradientBoostingRegressor


class NumericPayload(TypedDict):
    """Portable fitted state for one prompt-only numeric effort router."""

    protocol: str
    family: NumericFamily
    spec_json: str
    config_sha256: str
    dim: int
    structural_scale: np.ndarray
    arm_costs: np.ndarray
    fallback_arm: int
    quality_floor: float
    uncertainty: np.ndarray | None
    estimators: list[EstimatorModel]
    train_groups: list[str] | None
    train_rewards: np.ndarray | None


def _sha256(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _structural_scale(data: FitData, fit_indices: np.ndarray) -> np.ndarray:
    """Fit the frozen structural-feature scale on outer-fit rows only."""
    rows = np.asarray(
        [
            _structural_row(data.texts[int(index)], is_hard=bool(data.is_hard[int(index)]))
            for index in fit_indices
        ],
        dtype=np.float64,
    )
    return np.maximum(np.std(rows, axis=0), 1.0)


def _features(
    texts: list[str],
    is_hard: list[bool],
    *,
    dim: int,
    structural_scale: np.ndarray,
) -> sparse.csr_matrix:
    """Transform prompts with the exact frozen hashing and structural features."""
    if len(texts) != len(is_hard):
        raise ValueError("numeric router prompts and hard flags differ")
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        n_features=dim,
        alternate_sign=True,
        norm="l2",
    )
    hashed = cast(sparse.csr_matrix, vectorizer.transform(texts))
    structural = np.asarray(
        [_structural_row(text, is_hard=hard) for text, hard in zip(texts, is_hard, strict=True)],
        dtype=np.float64,
    )
    if structural_scale.shape != (structural.shape[1],):
        raise ValueError("numeric router structural scale has the wrong shape")
    structural /= structural_scale
    return sparse.hstack([hashed, sparse.csr_matrix(structural)], format="csr")


def _fit_estimators(
    spec: CandidateSpec,
    features: sparse.csr_matrix,
    train_rewards: np.ndarray,
    train_groups: list[str],
    *,
    seed: int,
) -> tuple[list[EstimatorModel], np.ndarray | None]:
    """Fit the selected family's exact deterministic prediction heads."""
    observed = train_rewards.mean(axis=2)
    if spec.family == "ordinal":
        targets = np.column_stack([observed[:, 0], np.diff(observed, axis=1)])
        estimators: list[EstimatorModel] = []
        predicted_parts: list[np.ndarray] = []
        for column in range(targets.shape[1]):
            if spec.estimator == "ridge":
                model: EstimatorModel = Ridge(alpha=spec.alpha)
            else:
                feature_rule: str | float = "sqrt" if spec.max_features == "sqrt" else 1.0 / 3.0
                model = ExtraTreesRegressor(
                    n_estimators=spec.n_estimators,
                    min_samples_leaf=spec.min_samples_leaf,
                    max_features=feature_rule,
                    random_state=seed + column,
                    n_jobs=1,
                )
            model.fit(features, targets[:, column])
            estimators.append(model)
            predicted_parts.append(np.asarray(model.predict(features), dtype=np.float64))
        parts = np.column_stack(predicted_parts)
        absolute = np.column_stack(
            [parts[:, 0], parts[:, 0, None] + np.cumsum(parts[:, 1:], axis=1)]
        )
        predicted = np.maximum.accumulate(np.clip(absolute, 0.0, 1.0), axis=1)
        return estimators, _residual_standard_errors(observed, predicted)

    if spec.family == "doubly-robust":
        pseudo = doubly_robust_pseudo_values(train_rewards, np.zeros_like(observed))
        estimators = []
        dense = np.asarray(features.toarray(), dtype=np.float64)
        for arm_index in range(len(ARMS)):
            if spec.estimator == "ridge":
                model = Ridge(alpha=spec.alpha)
                model.fit(features, pseudo[:, arm_index])
            else:
                model = HistGradientBoostingRegressor(
                    max_leaf_nodes=spec.max_leaf_nodes,
                    learning_rate=spec.learning_rate,
                    min_samples_leaf=spec.min_samples_leaf,
                    random_state=seed + arm_index,
                )
                model.fit(dense, pseudo[:, arm_index])
            estimators.append(model)
        return estimators, None

    train_base, _ = empirical_bayes_family_predictions(
        train_groups,
        train_groups,
        train_rewards,
        prior_strength=spec.prior_strength,
    )
    observed_parts = np.column_stack([observed[:, 0], np.diff(observed, axis=1)])
    base_parts = np.column_stack([train_base[:, 0], np.diff(train_base, axis=1)])
    estimators = []
    for column in range(observed_parts.shape[1]):
        model = Ridge(alpha=spec.alpha)
        model.fit(features, observed_parts[:, column] - base_parts[:, column])
        estimators.append(model)
    return estimators, None


def fit_numeric_payload(
    data: FitData,
    fit_indices: np.ndarray,
    spec: CandidateSpec,
    *,
    config_sha256: str,
    seed: int,
) -> NumericPayload:
    """Fit one selected non-kNN router on the complete outer-fit partition."""
    indices = np.asarray(fit_indices, dtype=np.int64)
    structural_scale = _structural_scale(data, indices)
    features = _features(
        [data.texts[int(index)] for index in indices],
        [bool(data.is_hard[int(index)]) for index in indices],
        dim=spec.dim,
        structural_scale=structural_scale,
    )
    train_rewards = data.rewards[indices]
    train_groups = [data.groups[int(index)] for index in indices]
    estimators, uncertainty = _fit_estimators(
        spec,
        features,
        train_rewards,
        train_groups,
        seed=seed,
    )
    baseline = fit_selected_static(data, indices)
    return NumericPayload(
        protocol="bigcodebench-numeric-router-v1",
        family=spec.family,
        spec_json=json.dumps(spec.config(), sort_keys=True, separators=(",", ":")),
        config_sha256=config_sha256,
        dim=spec.dim,
        structural_scale=structural_scale,
        arm_costs=data.costs[indices].mean(axis=(0, 2)),
        fallback_arm=ARMS.index(baseline.name),
        quality_floor=0.95 * baseline.reward,
        uncertainty=uncertainty,
        estimators=estimators,
        train_groups=train_groups if spec.family == "empirical-bayes" else None,
        train_rewards=train_rewards if spec.family == "empirical-bayes" else None,
    )


def numeric_choices(
    payload: NumericPayload,
    spec: CandidateSpec,
    texts: list[str],
    groups: list[str],
    is_hard: list[bool],
) -> np.ndarray:
    """Route a batch through one already-fitted numeric artifact."""
    if len(texts) != len(groups) or len(texts) != len(is_hard):
        raise ValueError("numeric route inputs differ")
    if payload["spec_json"] != json.dumps(spec.config(), sort_keys=True, separators=(",", ":")):
        raise ValueError("numeric payload and candidate specification differ")
    features = _features(
        texts,
        is_hard,
        dim=payload["dim"],
        structural_scale=payload["structural_scale"],
    )
    estimators = payload["estimators"]
    if spec.family == "ordinal":
        parts = np.column_stack(
            [np.asarray(model.predict(features), dtype=np.float64) for model in estimators]
        )
        absolute = np.column_stack(
            [parts[:, 0], parts[:, 0, None] + np.cumsum(parts[:, 1:], axis=1)]
        )
        predicted = np.maximum.accumulate(np.clip(absolute, 0.0, 1.0), axis=1)
        uncertainty = payload["uncertainty"]
        if uncertainty is None:
            raise ValueError("ordinal numeric payload has no uncertainty")
        return lower_bound_choices(
            predicted,
            np.broadcast_to(uncertainty, predicted.shape),
            payload["arm_costs"],
            quality_floor=payload["quality_floor"],
            fallback_arm=payload["fallback_arm"],
            z=1.0,
        )
    if spec.family == "doubly-robust":
        if spec.estimator == "histogram":
            values: sparse.csr_matrix | np.ndarray = np.asarray(
                features.toarray(), dtype=np.float64
            )
        else:
            values = features
        predicted = np.clip(
            np.column_stack(
                [np.asarray(model.predict(values), dtype=np.float64) for model in estimators]
            ),
            0.0,
            1.0,
        )
        return shadow_price_choices(predicted, payload["arm_costs"], lam=spec.lam)

    train_groups = payload["train_groups"]
    train_rewards = payload["train_rewards"]
    if train_groups is None or train_rewards is None:
        raise ValueError("empirical-Bayes payload has no fit evidence")
    _, _, test_base, test_se = empirical_bayes_family_moments(
        train_groups,
        groups,
        train_rewards,
        prior_strength=spec.prior_strength,
    )
    residual_parts = np.column_stack(
        [np.asarray(model.predict(features), dtype=np.float64) for model in estimators]
    )
    test_parts = np.column_stack([test_base[:, 0], np.diff(test_base, axis=1)]) + residual_parts
    absolute = np.column_stack(
        [test_parts[:, 0], test_parts[:, 0, None] + np.cumsum(test_parts[:, 1:], axis=1)]
    )
    predicted = np.maximum.accumulate(np.clip(absolute, 0.0, 1.0), axis=1)
    return lower_bound_choices(
        predicted,
        test_se,
        payload["arm_costs"],
        quality_floor=payload["quality_floor"],
        fallback_arm=payload["fallback_arm"],
        z=spec.z,
    )


def _record_spec(record: CandidateRecord) -> CandidateSpec:
    """Rebuild an exact selected non-kNN candidate from its canonical record."""
    from coding_model_router_bigcodebench_evaluate import candidate_spec_from_lock

    spec = candidate_spec_from_lock(
        record.family,
        record.config_json,
        name=record.name,
        order=record.order,
    )
    if not isinstance(spec, CandidateSpec):
        raise ValueError("numeric audit received a kNN candidate")
    return spec


def audit_seed_numeric_winner(
    root: Path,
    *,
    report_path: Path,
    artifact_dir: Path,
    output: Path,
    decisions: int = 10_000,
) -> SeedWinnerAudit:
    """Fit, cross-check, persist, and latency-audit one numeric seed winner."""
    if output.exists() or artifact_dir.exists():
        raise FileExistsError("numeric winner artifact or audit already exists")
    report = SeedFitReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    record = next(
        candidate for candidate in report.candidates if candidate.name == report.selected_name
    )
    spec = _record_spec(record)
    data = load_fit_data(root)
    split = next(split for split in outer_splits(data.groups) if split.seed == report.seed)
    payload = fit_numeric_payload(
        data,
        split.train_indices,
        spec,
        config_sha256=record.config_sha256,
        seed=report.seed,
    )
    fit_texts = [data.texts[int(index)] for index in split.train_indices]
    fit_groups = [data.groups[int(index)] for index in split.train_indices]
    fit_hard = [bool(data.is_hard[int(index)]) for index in split.train_indices]
    choices = numeric_choices(payload, spec, fit_texts, fit_groups, fit_hard)
    if choices.shape != (len(split.train_indices),):
        raise AssertionError("numeric artifact did not route every outer-fit task")
    reference_features = _features(
        data.texts,
        [bool(value) for value in data.is_hard],
        dim=spec.dim,
        structural_scale=payload["structural_scale"],
    )
    expected = _candidate_choices(
        spec,
        data,
        split.train_indices,
        split.train_indices,
        reference_features[split.train_indices],
        reference_features[split.train_indices],
        seed=report.seed,
    )
    if not np.array_equal(choices, expected):
        raise AssertionError("numeric artifact routes differ from the frozen selected candidate")
    artifact_dir.mkdir(parents=True, exist_ok=False)
    artifact_path = artifact_dir / "numeric-router.joblib"
    joblib.dump(payload, artifact_path, compress=3, protocol=5)
    restored = cast(NumericPayload, joblib.load(artifact_path))
    if not np.array_equal(
        choices,
        numeric_choices(restored, spec, fit_texts, fit_groups, fit_hard),
    ):
        raise AssertionError("persisted numeric artifact changed its routes")
    metadata = {
        text: (group, hard)
        for text, group, hard in zip(fit_texts, fit_groups, fit_hard, strict=True)
    }

    def route_one(text: str) -> int:
        group, hard = metadata[text]
        return int(numeric_choices(restored, spec, [text], [group], [hard])[0])

    latency: LatencyMetric = measure_route_latency(
        route_one,
        fit_texts,
        decisions=decisions,
    )
    if not latency.passed:
        raise ValueError(f"seed {report.seed} numeric winner failed the frozen latency gate")
    audit = SeedWinnerAudit(
        seed=report.seed,
        seed_report_sha256=_sha256(report_path),
        candidate_name=record.name,
        config_sha256=record.config_sha256,
        artifact_kind="numeric-router",
        artifact_sha256=_sha256(artifact_path),
        artifact_bytes=artifact_size([artifact_path]),
        decisions=latency.decisions,
        latency_p50_ms=latency.p50_ms,
        latency_p95_ms=latency.p95_ms,
        latency_passed=True,
    )
    write_text_atomic(output, audit.model_dump_json(indent=2) + "\n")
    logger.info(
        "seed=%d audited numeric winner=%s p50_ms=%.6f p95_ms=%.6f bytes=%d",
        report.seed,
        record.name,
        audit.latency_p50_ms,
        audit.latency_p95_ms,
        audit.artifact_bytes,
    )
    return audit


def parse_args() -> argparse.Namespace:
    """Parse the remote numeric winner-audit command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Build and audit one fit-selected numeric router on the remote CPU."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    audit_seed_numeric_winner(
        args.root.resolve(),
        report_path=args.report.resolve(),
        artifact_dir=args.artifact_dir.resolve(),
        output=args.output.resolve(),
    )


if __name__ == "__main__":
    main()
