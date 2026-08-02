"""Nested external validation for a zero-inflated effort-uplift router.

Open-SWE paired outcomes are split by repository in every outer and inner fold.
The BeyondSWE trace-burden score is a fixed auxiliary feature produced before
Open-SWE labels are inspected. Candidate selection happens only inside each
outer training fold. DeepSWE data is never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
from scipy import sparse
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.model_selection import GroupKFold

logger = logging.getLogger("coding-router-open-swe-hurdle")

TRAFFIC_FRACTION = 0.20
BOOTSTRAP_SAMPLES = 2_000


@dataclass(frozen=True)
class Candidate:
    name: str
    features: Literal["structural", "char", "word", "hybrid"]
    estimator: Literal[
        "direct-ridge",
        "two-head-ridge",
        "extra-direct",
        "hist-direct",
        "sign-ranker",
        "hurdle",
    ]
    dim: int = 4_096
    alpha: float = 10.0
    shuffled: bool = False


CANDIDATES = (
    Candidate("structural-extra-direct-l20", "structural", "extra-direct"),
    Candidate("structural-hist-direct-l30", "structural", "hist-direct"),
    Candidate("hybrid-ridge-direct-a10", "hybrid", "direct-ridge"),
    Candidate("hybrid-ridge-heads-a10", "hybrid", "two-head-ridge"),
    Candidate("char-sign-ranker", "char", "sign-ranker"),
    Candidate("word-sign-ranker", "word", "sign-ranker"),
    Candidate("hybrid-sign-ranker", "hybrid", "sign-ranker"),
    Candidate("hybrid-hurdle", "hybrid", "hurdle"),
    Candidate(
        "shuffled-hybrid-ridge-direct-a10",
        "hybrid",
        "direct-ridge",
        shuffled=True,
    ),
)


@dataclass(frozen=True)
class Data:
    task_ids: list[str]
    groups: np.ndarray
    texts: list[str]
    structural: np.ndarray
    auxiliary: np.ndarray
    cheap: np.ndarray
    strong: np.ndarray
    cheap_attempts: np.ndarray
    strong_attempts: np.ndarray

    @property
    def uplift(self) -> np.ndarray:
        return self.strong - self.cheap

    @property
    def precision(self) -> np.ndarray:
        denominator = (1.0 / self.cheap_attempts) + (1.0 / self.strong_attempts)
        weights = 2.0 / denominator
        return weights / float(np.mean(weights))


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


def _read_auxiliary(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            task_id = raw.get("task_id")
            score = raw.get("score")
            if not isinstance(task_id, str) or not isinstance(score, (int, float)):
                raise ValueError(f"{path}:{line_number} has no task_id or score")
            if task_id in scores:
                raise ValueError(f"{path} has duplicate task_id={task_id}")
            scores[task_id] = float(score)
    return scores


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _repo(repo: str) -> str:
    value = repo.casefold().strip()
    value = re.sub(r"^https?://github\.com/", "", value)
    return value.removesuffix(".git").strip("/") or "unknown"


def _structural(text: str) -> list[float]:
    normalized = text.replace("\r\n", "\n")
    lowered = normalized.casefold()
    length = max(len(normalized), 1)
    lines = normalized.splitlines() or [normalized]
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", normalized)
    unique_words = len({word.casefold() for word in words})
    per_thousand = 1_000.0 / length
    counts = [
        normalized.count("\n"),
        normalized.count("```"),
        normalized.count("`"),
        len(re.findall(r"https?://", lowered)),
        len(re.findall(r"""(?:^|[\s`'"])[\w.-]+/[\w./-]+""", normalized)),
        len(re.findall(r"\b(?:error|exception|traceback|panic|failed|failure)\b", lowered)),
        len(re.findall(r"\b(?:test|tests|testing|assert|expected|actual)\b", lowered)),
        len(re.findall(r"\b(?:bug|fix|regression|broken|incorrect)\b", lowered)),
        len(re.findall(r"\b(?:feature|support|implement|add|enhancement)\b", lowered)),
        len(re.findall(r"\b(?:performance|slow|latency|memory|timeout)\b", lowered)),
        len(re.findall(r"\b(?:api|interface|method|function|class)\b", lowered)),
        len(re.findall(r"(?m)^\s*(?:[-*+] |\d+[.)] )", normalized)),
        len(re.findall(r"(?m)^#{1,6}\s", normalized)),
        len(re.findall(r"(?m)^\s*>", normalized)),
        normalized.count("{") + normalized.count("}"),
        normalized.count("[") + normalized.count("]"),
        normalized.count("(") + normalized.count(")"),
        normalized.count("="),
    ]
    uppercase = sum(character.isupper() for character in normalized)
    digits = sum(character.isdigit() for character in normalized)
    punctuation = sum(
        not character.isalnum() and not character.isspace()
        for character in normalized
    )
    line_lengths = [len(line) for line in lines]
    return [
        math.log1p(length) / 12.0,
        math.log1p(len(lines)) / 8.0,
        math.log1p(len(words)) / 12.0,
        min(sum(line_lengths) / len(line_lengths) / 200.0, 2.0),
        min(max(line_lengths) / 2_000.0, 2.0),
        unique_words / max(len(words), 1),
        uppercase / length,
        digits / length,
        punctuation / length,
        *[math.log1p(count * per_thousand) / 8.0 for count in counts],
    ]


def _load_data(paired_path: Path, auxiliary_path: Path) -> Data:
    rows = _read_list(paired_path)
    auxiliary = _read_auxiliary(auxiliary_path)
    ids = [str(row["instance_id"]) for row in rows]
    missing = sorted(set(ids) - auxiliary.keys())
    extra = sorted(auxiliary.keys() - set(ids))
    if missing or extra:
        raise ValueError(
            f"auxiliary score identity mismatch missing={len(missing)} extra={len(extra)}"
        )
    texts = [str(row["text"]) for row in rows]
    data = Data(
        task_ids=ids,
        groups=np.asarray([_repo(str(row["repo"])) for row in rows], dtype=object),
        texts=texts,
        structural=np.asarray([_structural(text) for text in texts], dtype=np.float64),
        auxiliary=np.asarray([auxiliary[task_id] for task_id in ids], dtype=np.float64),
        cheap=np.asarray(
            [_number(row["cheap_reward"], name="cheap_reward") for row in rows],
            dtype=np.float64,
        ),
        strong=np.asarray(
            [_number(row["strong_reward"], name="strong_reward") for row in rows],
            dtype=np.float64,
        ),
        cheap_attempts=np.asarray(
            [_number(row["cheap_attempts"], name="cheap_attempts") for row in rows],
            dtype=np.float64,
        ),
        strong_attempts=np.asarray(
            [_number(row["strong_attempts"], name="strong_attempts") for row in rows],
            dtype=np.float64,
        ),
    )
    if np.any(data.cheap_attempts <= 0) or np.any(data.strong_attempts <= 0):
        raise ValueError("attempt counts must be positive")
    return data


def _features(data: Data, candidate: Candidate) -> sparse.csr_matrix:
    static = sparse.csr_matrix(
        np.column_stack([data.structural, data.auxiliary]),
        dtype=np.float64,
    )
    if candidate.features == "structural":
        return static
    matrices: list[sparse.csr_matrix] = []
    if candidate.features in {"char", "hybrid"}:
        vectorizer = HashingVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            n_features=candidate.dim,
            alternate_sign=True,
            norm="l2",
        )
        matrices.append(cast(sparse.csr_matrix, vectorizer.transform(data.texts)))
    if candidate.features in {"word", "hybrid"}:
        vectorizer = HashingVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            n_features=candidate.dim,
            alternate_sign=True,
            norm="l2",
        )
        matrices.append(cast(sparse.csr_matrix, vectorizer.transform(data.texts)))
    matrices.append(static)
    return sparse.hstack(matrices, format="csr")


def _constant_or_classifier(
    train_features: sparse.csr_matrix,
    train_labels: np.ndarray,
    test_features: sparse.csr_matrix,
    *,
    sample_weight: np.ndarray,
    seed: int,
) -> np.ndarray:
    classes = np.unique(train_labels)
    if len(classes) == 1:
        return np.full(test_features.shape[0], float(classes[0]))
    model = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        max_iter=2_000,
        tol=1e-4,
        class_weight="balanced",
        random_state=seed,
        average=True,
    )
    model.fit(train_features, train_labels, sample_weight=sample_weight)
    probabilities = model.predict_proba(test_features)
    return np.asarray(probabilities[:, -1], dtype=np.float64)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.sum(values * weights) / np.sum(weights))


def _fit_predict(
    data: Data,
    features: sparse.csr_matrix,
    train: np.ndarray,
    test: np.ndarray,
    candidate: Candidate,
    *,
    seed: int,
) -> np.ndarray:
    uplift = data.uplift.copy()
    if candidate.shuffled:
        uplift[train] = uplift[
            np.random.default_rng(seed).permutation(train)
        ]
    precision = data.precision
    train_features = features[train]
    test_features = features[test]
    if candidate.estimator == "direct-ridge":
        model = Ridge(alpha=candidate.alpha)
        model.fit(train_features, uplift[train], sample_weight=precision[train])
        return np.asarray(model.predict(test_features), dtype=np.float64)
    if candidate.estimator == "two-head-ridge":
        cheap_model = Ridge(alpha=candidate.alpha)
        strong_model = Ridge(alpha=candidate.alpha)
        cheap_model.fit(
            train_features,
            data.cheap[train],
            sample_weight=data.cheap_attempts[train],
        )
        strong_model.fit(
            train_features,
            data.strong[train],
            sample_weight=data.strong_attempts[train],
        )
        return np.asarray(
            strong_model.predict(test_features) - cheap_model.predict(test_features),
            dtype=np.float64,
        )
    if candidate.estimator == "extra-direct":
        model = ExtraTreesRegressor(
            n_estimators=240,
            min_samples_leaf=20,
            max_features=0.8,
            n_jobs=1,
            random_state=seed,
        )
        model.fit(
            train_features.toarray(),
            uplift[train],
            sample_weight=precision[train],
        )
        return np.asarray(model.predict(test_features.toarray()), dtype=np.float64)
    if candidate.estimator == "hist-direct":
        model = HistGradientBoostingRegressor(
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            learning_rate=0.05,
            l2_regularization=10.0,
            random_state=seed,
        )
        model.fit(
            train_features.toarray(),
            uplift[train],
            sample_weight=precision[train],
        )
        return np.asarray(model.predict(test_features.toarray()), dtype=np.float64)
    discordant = train[np.abs(uplift[train]) > 1e-12]
    if len(discordant) < 2:
        return np.zeros(len(test), dtype=np.float64)
    sign = (uplift[discordant] > 0.0).astype(np.int64)
    sign_weight = precision[discordant] * np.abs(uplift[discordant])
    positive_probability = _constant_or_classifier(
        features[discordant],
        sign,
        test_features,
        sample_weight=sign_weight,
        seed=seed,
    )
    positive = uplift[discordant] > 0.0
    positive_mean = _weighted_mean(
        uplift[discordant][positive],
        sign_weight[positive],
    )
    negative_mean = _weighted_mean(
        uplift[discordant][~positive],
        sign_weight[~positive],
    )
    signed_value = (
        positive_probability * positive_mean
        + (1.0 - positive_probability) * negative_mean
    )
    if candidate.estimator == "sign-ranker":
        return signed_value
    discordance_probability = _constant_or_classifier(
        train_features,
        (np.abs(uplift[train]) > 1e-12).astype(np.int64),
        test_features,
        sample_weight=precision[train],
        seed=seed + 10_000,
    )
    return discordance_probability * signed_value


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


def _advantage(data: Data, indices: np.ndarray, scores: np.ndarray) -> float:
    count = max(1, int(round(TRAFFIC_FRACTION * len(indices))))
    order = np.argsort(-scores, kind="mergesort")
    routed = data.cheap[indices].copy()
    routed[order[:count]] = data.strong[indices[order[:count]]]
    blind = data.cheap[indices] + TRAFFIC_FRACTION * data.uplift[indices]
    return float(np.mean(routed) - np.mean(blind))


def _oof(
    data: Data,
    features: sparse.csr_matrix,
    indices: np.ndarray,
    candidate: Candidate,
    *,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    local_groups = data.groups[indices]
    splitter = GroupKFold(n_splits=5)
    predictions = np.zeros(len(indices), dtype=np.float64)
    audits: list[dict[str, int]] = []
    for fold, (local_train, local_test) in enumerate(
        splitter.split(indices, groups=local_groups)
    ):
        train = indices[local_train]
        test = indices[local_test]
        overlap = set(data.groups[train]) & set(data.groups[test])
        if overlap:
            raise ValueError(f"fold {fold} has repository overlap")
        predictions[local_test] = _fit_predict(
            data,
            features,
            train,
            test,
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
    return predictions, audits


def _candidate_metrics(
    data: Data,
    indices: np.ndarray,
    scores: np.ndarray,
    candidate: Candidate,
) -> dict[str, object]:
    return {
        "candidate": candidate.name,
        "features": candidate.features,
        "estimator": candidate.estimator,
        "shuffled": candidate.shuffled,
        "uplift_spearman": _spearman(scores, data.uplift[indices]),
        "matched_blind_advantage": _advantage(data, indices, scores),
        "score_std": float(np.std(scores)),
    }


def _bootstrap(data: Data, scores: np.ndarray, *, seed: int) -> list[float]:
    groups = sorted(set(cast(list[str], data.groups.tolist())))
    by_group = {
        group: np.flatnonzero(data.groups == group)
        for group in groups
    }
    rng = np.random.default_rng(seed)
    values = np.zeros(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for sample_index in range(BOOTSTRAP_SAMPLES):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([by_group[str(group)] for group in sampled])
        values[sample_index] = _advantage(data, indices, scores[indices])
    return [
        float(value)
        for value in np.quantile(values, np.asarray([0.025, 0.5, 0.975]))
    ]


def fit(
    paired_path: Path,
    auxiliary_path: Path,
    auxiliary_report_path: Path,
    output: Path,
    *,
    seed: int,
) -> None:
    auxiliary_report = json.loads(auxiliary_report_path.read_text(encoding="utf-8"))
    if not isinstance(auxiliary_report, dict):
        raise ValueError("auxiliary report is not an object")
    source = auxiliary_report.get("source")
    if not isinstance(source, dict) or source.get(
        "candidate_selection_uses_only_source_oof"
    ) is not True:
        raise ValueError("auxiliary scorer was not selected only on BeyondSWE OOF")
    data = _load_data(paired_path, auxiliary_path)
    feature_bank = {
        candidate.name: _features(data, candidate)
        for candidate in CANDIDATES
    }
    all_indices = np.arange(len(data.task_ids), dtype=np.int64)
    nested_scores = np.zeros(len(data.task_ids), dtype=np.float64)
    outer_rows: list[dict[str, object]] = []
    selections: Counter[str] = Counter()
    outer = GroupKFold(n_splits=5)
    for outer_fold, (train, test) in enumerate(
        outer.split(all_indices, groups=data.groups)
    ):
        overlap = set(data.groups[train]) & set(data.groups[test])
        if overlap:
            raise ValueError(f"outer fold {outer_fold} has repository overlap")
        inner_rows: list[dict[str, object]] = []
        for candidate in CANDIDATES:
            scores, _ = _oof(
                data,
                feature_bank[candidate.name],
                train,
                candidate,
                seed=seed + outer_fold * 100,
            )
            inner_rows.append(_candidate_metrics(data, train, scores, candidate))
        eligible = [row for row in inner_rows if row["shuffled"] is not True]
        selected = max(
            eligible,
            key=lambda row: (
                float(cast(float, row["matched_blind_advantage"])),
                float(cast(float, row["uplift_spearman"])),
                str(row["candidate"]),
            ),
        )
        selected_name = str(selected["candidate"])
        selected_candidate = next(
            candidate for candidate in CANDIDATES if candidate.name == selected_name
        )
        nested_scores[test] = _fit_predict(
            data,
            feature_bank[selected_name],
            train,
            test,
            selected_candidate,
            seed=seed + outer_fold,
        )
        selections[selected_name] += 1
        outer_rows.append(
            {
                "fold": outer_fold,
                "train_tasks": len(train),
                "test_tasks": len(test),
                "group_overlap": len(overlap),
                "selected_candidate": selected,
                "inner_candidates": inner_rows,
            }
        )
    nested_metrics = {
        "uplift_spearman": _spearman(nested_scores, data.uplift),
        "matched_blind_advantage": _advantage(data, all_indices, nested_scores),
        "group_bootstrap_advantage_95ci": _bootstrap(data, nested_scores, seed=seed),
        "selection_counts": dict(sorted(selections.items())),
    }
    final_candidates: list[dict[str, object]] = []
    final_oof: dict[str, np.ndarray] = {}
    final_audits: dict[str, list[dict[str, int]]] = {}
    for candidate in CANDIDATES:
        scores, audits = _oof(
            data,
            feature_bank[candidate.name],
            all_indices,
            candidate,
            seed=seed,
        )
        final_oof[candidate.name] = scores
        final_audits[candidate.name] = audits
        final_candidates.append(
            _candidate_metrics(data, all_indices, scores, candidate)
        )
    deployable = [row for row in final_candidates if row["shuffled"] is not True]
    selected_final = max(
        deployable,
        key=lambda row: (
            float(cast(float, row["matched_blind_advantage"])),
            float(cast(float, row["uplift_spearman"])),
            str(row["candidate"]),
        ),
    )
    nested_interval = nested_metrics["group_bootstrap_advantage_95ci"]
    shuffled = next(row for row in final_candidates if row["shuffled"] is True)
    gate = {
        "nested_uplift_spearman_positive": (
            float(nested_metrics["uplift_spearman"]) > 0.0
        ),
        "nested_matched_blind_advantage_positive": (
            float(nested_metrics["matched_blind_advantage"]) > 0.0
        ),
        "nested_group_bootstrap_lower_bound_positive": nested_interval[0] > 0.0,
        "selected_final_beats_shuffled_control": (
            float(cast(float, selected_final["matched_blind_advantage"]))
            > float(cast(float, shuffled["matched_blind_advantage"]))
        ),
    }
    gate["passed"] = all(bool(value) for value in gate.values())
    report = {
        "protocol": "open-swe-zero-inflated-uplift-nested-v1",
        "tasks": len(data.task_ids),
        "repositories": len(set(cast(list[str], data.groups.tolist()))),
        "response_patterns": {
            "positive_uplift": int(np.sum(data.uplift > 0.0)),
            "zero_uplift": int(np.sum(data.uplift == 0.0)),
            "negative_uplift": int(np.sum(data.uplift < 0.0)),
        },
        "traffic_fraction": TRAFFIC_FRACTION,
        "nested_external_test": nested_metrics,
        "outer_folds": outer_rows,
        "full_source_candidate_results": final_candidates,
        "selected_final_candidate": selected_final,
        "selected_final_fold_audits": final_audits[str(selected_final["candidate"])],
        "external_gate": gate,
        "auxiliary": {
            "source": "BeyondSWE GPT-5.4 XHigh Codex trace burden",
            "selected_before_open_swe_labels": True,
            "report_sha256": hashlib.sha256(
                auxiliary_report_path.read_bytes()
            ).hexdigest(),
        },
        "target_outcomes_used": False,
        "target_embeddings_used": False,
        "deep_swe_evaluation_authorized": bool(gate["passed"]),
        "no_persisted_fitted_model": True,
        "inputs": {
            "paired_sha256": hashlib.sha256(paired_path.read_bytes()).hexdigest(),
            "auxiliary_scores_sha256": hashlib.sha256(
                auxiliary_path.read_bytes()
            ).hexdigest(),
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "nested-scores.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "task_id": task_id,
                    "repo": str(data.groups[index]),
                    "score": float(nested_scores[index]),
                    "uplift": float(data.uplift[index]),
                },
                sort_keys=True,
            )
            + "\n"
            for index, task_id in enumerate(data.task_ids)
        ),
        encoding="utf-8",
    )
    logger.info(
        "nested Open-SWE complete selected=%s rho=%.4f advantage=%.6f "
        "bootstrap_low=%.6f gate=%s",
        selected_final["candidate"],
        float(nested_metrics["uplift_spearman"]),
        float(nested_metrics["matched_blind_advantage"]),
        nested_interval[0],
        gate["passed"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--auxiliary-scores", type=Path, required=True)
    parser.add_argument("--auxiliary-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    fit(
        args.paired,
        args.auxiliary_scores,
        args.auxiliary_report,
        args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
