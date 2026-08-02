"""Evaluate the frozen direct effort router on untouched Codeforces tasks.

The route is fit only on the complete development matrix. Confirmation labels
are used once for evaluation and cannot alter the representation, Ridge alpha,
threshold, arm roster, or tie break. DeepSWE stays sealed until every frozen
confirmation condition passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import cast

import numpy as np
from coding_model_router_codeforces_fit import (
    ARMS,
    HIGH_INDEX,
    Data,
    _bootstrap,
    _choose,
    _fit_delta_models,
    _score_delta_models,
    _spearman,
    _value,
    load_data,
)
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

DIMENSION = 512
ALPHA = 10.0
THRESHOLD = 0.0
FROZEN_CANDIDATE = "direct-hash512-a10-t0"
SEED = 20_260_731


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scale(data: Data) -> np.ndarray:
    return np.maximum(np.std(data.structural, axis=0), 1.0)


def _score_features(data: Data, scale: np.ndarray) -> sparse.csr_matrix:
    """Build the complete pre-inference feature matrix with frozen scaling."""
    if scale.shape != (data.structural.shape[1],):
        raise ValueError("structural scale shape does not match the task features")
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        n_features=DIMENSION,
        alternate_sign=True,
        norm="l2",
    )
    text = cast(sparse.csr_matrix, vectorizer.transform(data.texts))
    structural = sparse.csr_matrix(data.structural / scale)
    return sparse.hstack([text, structural], format="csr")


def _route(
    development: Data,
    confirmation: Data,
    *,
    label_rewards: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[object, ...], np.ndarray]:
    scale = _scale(development)
    development_features = _score_features(development, scale)
    confirmation_features = _score_features(confirmation, scale)
    labels = development.rewards if label_rewards is None else label_rewards
    deltas = labels - labels[:, [HIGH_INDEX]]
    models = _fit_delta_models(
        development_features,
        deltas,
        np.arange(len(development.task_ids)),
        alpha=ALPHA,
    )
    predictions = _score_delta_models(
        confirmation_features,
        np.arange(len(confirmation.task_ids)),
        models,
    )
    choices = _choose(
        predictions,
        np.mean(development.costs, axis=0),
        threshold=THRESHOLD,
    )
    return choices, predictions, cast(tuple[object, ...], models), scale


def evaluate(
    development_corpus: Path,
    development_outcomes: Path,
    confirmation_corpus: Path,
    confirmation_outcomes: Path,
    output: Path,
) -> dict[str, object]:
    """Run the one-shot frozen confirmation and write a sealed report."""
    development = load_data(development_corpus, development_outcomes, expected_tasks=160)
    confirmation = load_data(confirmation_corpus, confirmation_outcomes, expected_tasks=160)
    if set(development.task_ids) & set(confirmation.task_ids):
        raise ValueError("development and confirmation tasks overlap")
    choices, predictions, models, scale = _route(development, confirmation)
    indices = np.arange(len(confirmation.task_ids))
    value = _value(confirmation, indices, choices)
    routed_rewards = cast(np.ndarray, value["routed_reward_by_task"])
    blind_rewards = cast(np.ndarray, value["blind_reward_by_task"])
    interval = _bootstrap(
        confirmation.groups,
        routed_rewards,
        blind_rewards,
        seed=SEED,
    )
    static = [
        {
            "arm": arm,
            "reward": float(np.mean(confirmation.rewards[:, arm_index])),
            "cost_usd": float(np.sum(confirmation.costs[:, arm_index])),
        }
        for arm_index, arm in enumerate(ARMS)
    ]
    dominated_by = [
        str(row["arm"])
        for row in static
        if float(row["reward"]) >= float(value["reward"])
        and float(row["cost_usd"]) <= float(value["cost_usd"])
        and (
            float(row["reward"]) > float(value["reward"])
            or float(row["cost_usd"]) < float(value["cost_usd"])
        )
    ]
    shuffled_labels = development.rewards[
        np.random.default_rng(SEED).permutation(len(development.task_ids))
    ]
    shuffled_choices, _, _, _ = _route(
        development,
        confirmation,
        label_rewards=shuffled_labels,
    )
    shuffled = _value(confirmation, indices, shuffled_choices)
    shuffled_router = cast(np.ndarray, shuffled["routed_reward_by_task"])
    shuffled_blind = cast(np.ndarray, shuffled["blind_reward_by_task"])
    shuffled_interval = _bootstrap(
        confirmation.groups,
        shuffled_router,
        shuffled_blind,
        seed=SEED + 1,
    )
    shuffled_gate = float(shuffled["advantage"]) > 0.0 and shuffled_interval[0] > 0.0
    started = time.perf_counter_ns()
    for _ in range(100):
        inference_features = _score_features(confirmation, scale)
        inference_predictions = _score_delta_models(
            inference_features,
            indices,
            cast(tuple, models),
        )
        _choose(
            inference_predictions,
            np.mean(development.costs, axis=0),
            threshold=THRESHOLD,
        )
    inference_batch_ms = (time.perf_counter_ns() - started) / 1_000_000 / 100
    observed_uplift = routed_rewards - confirmation.rewards[:, HIGH_INDEX]
    predicted = predictions[indices, choices]
    gate: dict[str, bool] = {
        "complete_and_target_sealed": bool(np.all(np.isfinite(routed_rewards))),
        "positive_matched_blind_advantage": float(value["advantage"]) > 0.0,
        "positive_contest_bootstrap_lower_bound": interval[0] > 0.0,
        "not_static_dominated": not dominated_by,
        "shuffled_control_failed": not shuffled_gate,
        "pre_inference_latency_below_5ms_per_task": inference_batch_ms / 160 < 5.0,
    }
    gate["passed"] = all(gate.values())
    report: dict[str, object] = {
        "protocol": "codeforces-direct-effort-confirmation-v1",
        "frozen_candidate": FROZEN_CANDIDATE,
        "tasks": len(confirmation.task_ids),
        "contest_groups": len(set(confirmation.groups)),
        "router": {
            "reward": value["reward"],
            "cost_usd": value["cost_usd"],
            "arm_counts": value["counts"],
            "matched_blind_reward": value["matched_blind_reward"],
            "matched_blind_cost_usd": value["matched_blind_cost_usd"],
            "advantage_vs_matched_blind": value["advantage"],
            "contest_bootstrap_advantage_95ci": interval,
            "predicted_uplift_spearman": _spearman(predicted, observed_uplift),
            "dominated_by_static_arms": dominated_by,
        },
        "static_efforts": static,
        "shuffled_control": {
            "reward": shuffled["reward"],
            "cost_usd": shuffled["cost_usd"],
            "advantage_vs_matched_blind": shuffled["advantage"],
            "contest_bootstrap_advantage_95ci": shuffled_interval,
            "passed_primary_advantage_gate": shuffled_gate,
        },
        "pre_inference_batch_160_mean_ms": inference_batch_ms,
        "confirmation_gate": gate,
        "deep_swe_evaluation_authorized": bool(gate["passed"]),
        "target_outcomes_used": False,
        "target_embeddings_used": False,
        "no_persisted_fitted_model": True,
        "inputs": {
            "development_corpus_sha256": _sha256(development_corpus),
            "development_outcomes_sha256": _sha256(development_outcomes),
            "confirmation_corpus_sha256": _sha256(confirmation_corpus),
            "confirmation_outcomes_sha256": _sha256(confirmation_outcomes),
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, task_id in enumerate(confirmation.task_ids):
            handle.write(
                json.dumps(
                    {
                        "task_id": task_id,
                        "contest_id": confirmation.groups[index],
                        "selected_arm": ARMS[int(choices[index])],
                        "predicted_uplift": float(predicted[index]),
                        "observed_uplift_vs_high": float(observed_uplift[index]),
                        "reward": float(routed_rewards[index]),
                        "matched_blind_reward": float(blind_rewards[index]),
                        "target_outcomes_used": False,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return report


def main() -> None:
    """Parse the frozen confirmation command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-corpus", type=Path, required=True)
    parser.add_argument("--development-outcomes", type=Path, required=True)
    parser.add_argument("--confirmation-corpus", type=Path, required=True)
    parser.add_argument("--confirmation-outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(
        args.development_corpus,
        args.development_outcomes,
        args.confirmation_corpus,
        args.confirmation_outcomes,
        args.output,
    )


if __name__ == "__main__":
    main()
