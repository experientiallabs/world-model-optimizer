"""Fit a task-only difficulty scorer on BeyondSWE and gate it on Open-SWE.

Candidate selection uses only grouped out-of-fold prediction of released
BeyondSWE Codex trace burden. The selected scorer is then refit on all BeyondSWE
rows and evaluated once against disjoint Open-SWE paired weak/strong outcomes.
DeepSWE data is never read by this script.
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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

logger = logging.getLogger("coding-router-beyondswe-fit")

TRAFFIC_FRACTIONS = (0.05, 0.10, 0.20, 0.40)
GATE_TRAFFIC_FRACTION = 0.20
BOOTSTRAP_SAMPLES = 2_000


@dataclass(frozen=True)
class Candidate:
    name: str
    feature_kind: Literal["structural", "char", "word", "hybrid"]
    dim: int
    alpha: float
    estimator: Literal["ridge", "extra-trees"] = "ridge"
    shuffled: bool = False


CANDIDATES = (
    Candidate("structural-ridge-a10", "structural", 1, 10.0),
    Candidate("structural-extra-trees", "structural", 1, 0.0, "extra-trees"),
    Candidate("char-hash2048-a10", "char", 2_048, 10.0),
    Candidate("char-hash8192-a10", "char", 8_192, 10.0),
    Candidate("char-hash8192-a100", "char", 8_192, 100.0),
    Candidate("word-hash2048-a10", "word", 2_048, 10.0),
    Candidate("word-hash8192-a10", "word", 8_192, 10.0),
    Candidate("hybrid-hash4096-a10", "hybrid", 4_096, 10.0),
    Candidate("hybrid-hash8192-a100", "hybrid", 8_192, 100.0),
    Candidate("shuffled-hybrid-hash4096-a10", "hybrid", 4_096, 10.0, shuffled=True),
)


@dataclass(frozen=True)
class SourceData:
    task_ids: list[str]
    groups: list[str]
    texts: list[str]
    structural: np.ndarray
    burden: np.ndarray
    raw_labels: dict[str, np.ndarray]


@dataclass(frozen=True)
class ValidationData:
    task_ids: list[str]
    groups: list[str]
    texts: list[str]
    structural: np.ndarray
    cheap: np.ndarray
    strong: np.ndarray


def _read_object(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return {str(key): value for key, value in raw.items()}


def _read_list(path: Path) -> list[dict[str, object]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain one JSON list")
    rows = [
        {str(key): value for key, value in row.items()}
        for row in raw
        if isinstance(row, dict)
    ]
    if len(rows) != len(raw):
        raise ValueError(f"{path} contains a non-object row")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _repo_group(repo: str) -> str:
    normalized = repo.casefold().strip()
    normalized = re.sub(r"^https?://github\.com/", "", normalized)
    normalized = normalized.removesuffix(".git").strip("/")
    return normalized or "unknown"


def _structural(text: str) -> list[float]:
    lowered = text.casefold()
    lines = text.splitlines()
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)
    return [
        math.log1p(len(text)),
        math.log1p(len(words)),
        math.log1p(len(lines)),
        float(text.count("\n")),
        float(text.count("```")),
        float(text.count("`")),
        float(text.count("/")),
        float(text.count(".")),
        float(text.count(":")),
        float(text.count("{") + text.count("}")),
        float(text.count("[") + text.count("]")),
        float(text.count("(") + text.count(")")),
        float(len(re.findall(r"(?m)^\s*(?:[-*+] |\d+[.)] )", text))),
        float(len(re.findall(r"(?m)^#{1,6}\s", text))),
        float(len(re.findall(r"\btest\w*\b", lowered))),
        float(len(re.findall(r"\b(?:bug|fix|error|fail)\w*\b", lowered))),
        float(len(re.findall(r"\b(?:implement|add|build|create)\w*\b", lowered))),
        float(len(re.findall(r"\b(?:repo|package|module|dependency)\w*\b", lowered))),
    ]


def _zscore(values: np.ndarray) -> np.ndarray:
    scale = float(np.std(values))
    if scale == 0.0:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / scale


def _burden(rows: list[dict[str, object]]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    labels = {
        "reward_deficit": np.asarray(
            [1.0 - _number(row["reward"], name="reward") for row in rows],
            dtype=np.float64,
        ),
        "log_steps": np.log1p(
            np.asarray(
                [
                    _number(row["trajectory_steps"], name="trajectory_steps")
                    for row in rows
                ],
                dtype=np.float64,
            )
        ),
        "log_prompt_tokens": np.log1p(
            np.asarray(
                [
                    _number(row["total_prompt_tokens"], name="total_prompt_tokens")
                    for row in rows
                ],
                dtype=np.float64,
            )
        ),
        "log_completion_tokens": np.log1p(
            np.asarray(
                [
                    _number(
                        row["total_completion_tokens"],
                        name="total_completion_tokens",
                    )
                    for row in rows
                ],
                dtype=np.float64,
            )
        ),
    }
    composite = np.mean(
        np.column_stack([_zscore(labels[name]) for name in sorted(labels)]),
        axis=1,
    )
    return composite, labels


def _source(path: Path) -> SourceData:
    raw = _read_object(path)
    values = raw.get("rows")
    if not isinstance(values, list):
        raise ValueError(f"{path} has no rows")
    rows = [
        {str(key): value for key, value in row.items()}
        for row in values
        if isinstance(row, dict)
    ]
    if len(rows) != len(values):
        raise ValueError(f"{path} contains a non-object row")
    burden, raw_labels = _burden(rows)
    texts = [str(row["text"]) for row in rows]
    return SourceData(
        task_ids=[str(row["task_id"]) for row in rows],
        groups=[_repo_group(str(row["repo"])) for row in rows],
        texts=texts,
        structural=np.asarray([_structural(text) for text in texts], dtype=np.float64),
        burden=burden,
        raw_labels=raw_labels,
    )


def _validation(path: Path) -> ValidationData:
    rows = _read_list(path)
    texts = [str(row["text"]) for row in rows]
    return ValidationData(
        task_ids=[str(row["instance_id"]) for row in rows],
        groups=[_repo_group(str(row["repo"])) for row in rows],
        texts=texts,
        structural=np.asarray([_structural(text) for text in texts], dtype=np.float64),
        cheap=np.asarray(
            [_number(row["cheap_reward"], name="cheap_reward") for row in rows],
            dtype=np.float64,
        ),
        strong=np.asarray(
            [_number(row["strong_reward"], name="strong_reward") for row in rows],
            dtype=np.float64,
        ),
    )


def _features(
    texts: list[str],
    structural: np.ndarray,
    candidate: Candidate,
) -> sparse.csr_matrix:
    scale = np.maximum(np.std(structural, axis=0), 1.0)
    structural_matrix = sparse.csr_matrix(structural / scale)
    if candidate.feature_kind == "structural":
        return structural_matrix
    matrices: list[sparse.csr_matrix] = []
    if candidate.feature_kind in {"char", "hybrid"}:
        char = HashingVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            n_features=candidate.dim,
            alternate_sign=True,
            norm="l2",
        )
        matrices.append(cast(sparse.csr_matrix, char.transform(texts)))
    if candidate.feature_kind in {"word", "hybrid"}:
        word = HashingVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            n_features=candidate.dim,
            alternate_sign=True,
            norm="l2",
        )
        matrices.append(cast(sparse.csr_matrix, word.transform(texts)))
    matrices.append(structural_matrix)
    return sparse.hstack(matrices, format="csr")


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
    if float(np.std(left_rank)) == 0.0 or float(np.std(right_rank)) == 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _fit_predict(
    train_features: sparse.csr_matrix,
    test_features: sparse.csr_matrix,
    train_labels: np.ndarray,
    candidate: Candidate,
    *,
    seed: int,
) -> np.ndarray:
    if candidate.estimator == "extra-trees":
        model = ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=5,
            max_features=0.8,
            n_jobs=1,
            random_state=seed,
        )
        model.fit(train_features.toarray(), train_labels)
        return np.asarray(model.predict(test_features.toarray()), dtype=np.float64)
    model = Ridge(alpha=candidate.alpha)
    model.fit(train_features, train_labels)
    return np.asarray(model.predict(test_features), dtype=np.float64)


def _oof_source(
    data: SourceData,
    candidate: Candidate,
    *,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    features = _features(data.texts, data.structural, candidate)
    labels = data.burden.copy()
    if candidate.shuffled:
        labels = labels[np.random.default_rng(seed).permutation(len(labels))]
    scores = np.zeros(len(labels), dtype=np.float64)
    audits: list[dict[str, int]] = []
    splitter = GroupKFold(n_splits=5)
    groups = np.asarray(data.groups)
    for fold, (train, test) in enumerate(splitter.split(features, groups=groups)):
        overlap = set(groups[train]) & set(groups[test])
        if overlap:
            raise ValueError(f"source fold {fold} has repository overlap")
        scores[test] = _fit_predict(
            features[train],
            features[test],
            labels[train],
            candidate,
            seed=seed + fold,
        )
        audits.append(
            {
                "fold": fold,
                "train_tasks": len(train),
                "test_tasks": len(test),
                "group_overlap": len(overlap),
            }
        )
    return scores, audits


def _transfer_scores(
    source: SourceData,
    validation: ValidationData,
    candidate: Candidate,
    *,
    seed: int,
) -> np.ndarray:
    combined_texts = source.texts + validation.texts
    combined_structural = np.vstack([source.structural, validation.structural])
    features = _features(combined_texts, combined_structural, candidate)
    split = len(source.texts)
    return _fit_predict(
        features[:split],
        features[split:],
        source.burden,
        candidate,
        seed=seed,
    )


def _operating_points(
    validation: ValidationData,
    scores: np.ndarray,
) -> list[dict[str, float | int]]:
    delta = validation.strong - validation.cheap
    order = np.argsort(-scores, kind="mergesort")
    rows: list[dict[str, float | int]] = []
    for fraction in TRAFFIC_FRACTIONS:
        count = max(1, int(round(fraction * len(scores))))
        routed = validation.cheap.copy()
        routed[order[:count]] = validation.strong[order[:count]]
        blind = validation.cheap + fraction * delta
        rows.append(
            {
                "strong_traffic_fraction": fraction,
                "strong_traffic_tasks": count,
                "router_reward": float(np.mean(routed)),
                "matched_blind_reward": float(np.mean(blind)),
                "advantage_vs_matched_blind": float(np.mean(routed) - np.mean(blind)),
            }
        )
    return rows


def _bootstrap(
    validation: ValidationData,
    scores: np.ndarray,
    *,
    seed: int,
) -> list[float]:
    groups = sorted(set(validation.groups))
    group_indices = {
        group: np.flatnonzero(np.asarray(validation.groups) == group) for group in groups
    }
    rng = np.random.default_rng(seed)
    values = np.zeros(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for sample_index in range(BOOTSTRAP_SAMPLES):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([group_indices[str(group)] for group in sampled])
        count = max(1, int(round(GATE_TRAFFIC_FRACTION * len(indices))))
        order = np.argsort(-scores[indices], kind="mergesort")
        routed = validation.cheap[indices].copy()
        routed[order[:count]] = validation.strong[indices[order[:count]]]
        blind = validation.cheap[indices] + GATE_TRAFFIC_FRACTION * (
            validation.strong[indices] - validation.cheap[indices]
        )
        values[sample_index] = float(np.mean(routed) - np.mean(blind))
    return [
        float(value)
        for value in np.quantile(values, np.asarray([0.025, 0.5, 0.975]))
    ]


def fit(
    source_path: Path,
    validation_path: Path,
    output: Path,
    *,
    seed: int,
) -> None:
    source = _source(source_path)
    validation = _validation(validation_path)
    source_results: list[dict[str, object]] = []
    source_scores: dict[str, np.ndarray] = {}
    fold_audits: dict[str, list[dict[str, int]]] = {}
    for candidate in CANDIDATES:
        scores, audits = _oof_source(source, candidate, seed=seed)
        source_scores[candidate.name] = scores
        fold_audits[candidate.name] = audits
        source_results.append(
            {
                "candidate": candidate.name,
                "feature_kind": candidate.feature_kind,
                "dim": candidate.dim,
                "alpha": candidate.alpha,
                "estimator": candidate.estimator,
                "shuffled": candidate.shuffled,
                "oof_burden_spearman": _spearman(scores, source.burden),
                "score_std": float(np.std(scores)),
            }
        )
    eligible = [row for row in source_results if row["shuffled"] is not True]
    selected = max(
        eligible,
        key=lambda row: (
            float(cast(float, row["oof_burden_spearman"])),
            str(row["candidate"]),
        ),
    )
    selected_name = str(selected["candidate"])
    candidate = next(item for item in CANDIDATES if item.name == selected_name)
    transfer = _transfer_scores(source, validation, candidate, seed=seed)
    uplift = validation.strong - validation.cheap
    operating_points = _operating_points(validation, transfer)
    gate_point = next(
        row
        for row in operating_points
        if row["strong_traffic_fraction"] == GATE_TRAFFIC_FRACTION
    )
    interval = _bootstrap(validation, transfer, seed=seed)
    validation_spearman = _spearman(transfer, uplift)
    shuffled = next(row for row in source_results if row["shuffled"] is True)
    gate = {
        "source_oof_burden_spearman_above_0_10": (
            float(cast(float, selected["oof_burden_spearman"])) > 0.10
        ),
        "source_beats_shuffled_control": (
            float(cast(float, selected["oof_burden_spearman"]))
            > float(cast(float, shuffled["oof_burden_spearman"]))
        ),
        "validation_uplift_spearman_positive": validation_spearman > 0.0,
        "validation_matched_blind_advantage_positive": (
            float(gate_point["advantage_vs_matched_blind"]) > 0.0
        ),
        "validation_group_bootstrap_lower_bound_positive": interval[0] > 0.0,
    }
    gate["passed"] = all(bool(value) for value in gate.values())
    report = {
        "protocol": "beyondswe-trace-burden-open-swe-gate-v1",
        "source": {
            "tasks": len(source.task_ids),
            "repositories": len(set(source.groups)),
            "candidate_selection_uses_only_source_oof": True,
            "candidate_results": source_results,
            "selected_candidate": selected,
            "fold_audits": fold_audits[selected_name],
            "label_components": {
                name: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
                for name, values in source.raw_labels.items()
            },
        },
        "validation": {
            "tasks": len(validation.task_ids),
            "repositories": len(set(validation.groups)),
            "cheap_reward": float(np.mean(validation.cheap)),
            "strong_reward": float(np.mean(validation.strong)),
            "mean_uplift": float(np.mean(uplift)),
            "uplift_spearman": validation_spearman,
            "operating_points": operating_points,
            "gate_traffic_fraction": GATE_TRAFFIC_FRACTION,
            "gate_group_bootstrap_advantage_95ci": interval,
        },
        "external_gate": gate,
        "target_outcomes_used": False,
        "target_embeddings_used": False,
        "deep_swe_evaluation_authorized": bool(gate["passed"]),
        "no_persisted_fitted_model": True,
        "inputs": {
            "source_sha256": _sha256(source_path),
            "validation_sha256": _sha256(validation_path),
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "validation-scores.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "task_id": task_id,
                    "repo": validation.groups[index],
                    "score": float(transfer[index]),
                    "cheap_reward": float(validation.cheap[index]),
                    "strong_reward": float(validation.strong[index]),
                    "uplift": float(uplift[index]),
                },
                sort_keys=True,
            )
            + "\n"
            for index, task_id in enumerate(validation.task_ids)
        ),
        encoding="utf-8",
    )
    logger.info(
        "BeyondSWE fit complete selected=%s source_rho=%.4f validation_rho=%.4f "
        "advantage=%.6f bootstrap_low=%.6f gate=%s",
        selected_name,
        float(cast(float, selected["oof_burden_spearman"])),
        validation_spearman,
        float(gate_point["advantage_vs_matched_blind"]),
        interval[0],
        gate["passed"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    fit(args.source, args.validation, args.output, seed=args.seed)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
