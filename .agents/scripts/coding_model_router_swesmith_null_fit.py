"""Fit the frozen SWE-smith null-penalized effort router on E2B."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from coding_model_router_swerebench_fit import ARMS, SourceData, load_source
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

logger = logging.getLogger("coding-router-swesmith-null-fit")

PROTOCOL = "coding-router-swesmith-null-stability-fit-v1"
EXTERNAL_TASKS_SHA256 = "9a4b3b749fb2123933335f9c4db41057247f49b37c53a7c075143b44e800aa7c"
EXTERNAL_MANIFEST_SHA256 = "79fe94cade619cbaa7ded6bb2391418fac30b04fbea50c36cf9c92dac2dd3b02"
DEVELOPMENT_CORPUS_SHA256 = "7d846b5576d15e68fd18ac21bfe0610cc1614b3b35ec0ae0cb8cfae0b82962c1"
THRESHOLD_PERCENTILES = tuple(range(10, 91, 5))
NULL_COUNT = 128
NULL_SEED = 20_260_801
QUALITY_RETENTION = 0.95
MAX_ROUTE_P95_MS = 5.0
HIGH_INDEX = 2
MAX_INDEX = 4


@dataclass(frozen=True)
class ExternalTask:
    """One compact external task and its smoothed success target."""

    task_id: str
    repository: str
    prompt: str
    target: float


@dataclass(frozen=True)
class ScorerCandidate:
    """One preregistered external difficulty scorer."""

    order: int
    dim: int
    alpha: float

    @property
    def key(self) -> str:
        """Return the frozen scorer identity."""
        if self.dim == 0:
            return f"structural-only-a{self.alpha:g}"
        return f"charhash{self.dim}-a{self.alpha:g}"


@dataclass(frozen=True)
class FittedScorer:
    """Ephemeral fitted scorer used only within one isolated process."""

    candidate: ScorerCandidate
    vectorizer: HashingVectorizer | None
    model: Ridge


@dataclass(frozen=True)
class RouteMetrics:
    """Reward, cost, blind comparator, and static-frontier metrics."""

    reward: float
    cost_usd: float
    blind_reward: float
    blind_cost_usd: float
    advantage: float
    retention: float
    dominated_by_static: tuple[str, ...]
    arm_counts: dict[str, int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return {str(key): item for key, item in value.items()}


def _read_external(path: Path, manifest_path: Path) -> list[ExternalTask]:
    """Read the compact external corpus after verifying its isolation manifest."""
    if _sha256(path) != EXTERNAL_TASKS_SHA256:
        raise ValueError("external SWE-smith task corpus hash changed")
    if _sha256(manifest_path) != EXTERNAL_MANIFEST_SHA256:
        raise ValueError("external SWE-smith manifest hash changed")
    manifest = _read_object(manifest_path)
    if (
        manifest.get("valid") is not True
        or manifest.get("tasks_sha256") != EXTERNAL_TASKS_SHA256
        or manifest.get("retained_tasks") != 1_551
        or manifest.get("target_reward_fields_accessed") is not False
        or manifest.get("target_cost_fields_accessed") is not False
        or manifest.get("later_trajectory_turns_used") is not False
        or manifest.get("patch_field_read") is not False
        or manifest.get("fitted_model_persisted") is not False
    ):
        raise ValueError("external SWE-smith manifest is incomplete or unsafe")
    tasks: list[ExternalTask] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError("external corpus contains a non-object row")
        target = raw.get("difficulty_target")
        if (
            not isinstance(raw.get("task_id"), str)
            or not isinstance(raw.get("repository"), str)
            or not isinstance(raw.get("prompt"), str)
            or isinstance(target, bool)
            or not isinstance(target, (int, float))
            or not 0.0 < float(target) < 1.0
        ):
            raise ValueError("external corpus contains an invalid task row")
        tasks.append(
            ExternalTask(
                task_id=str(raw["task_id"]),
                repository=str(raw["repository"]),
                prompt=str(raw["prompt"]),
                target=float(target),
            )
        )
    if len(tasks) != 1_551 or len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("external SWE-smith task identities are incomplete")
    return tasks


def _scorer_grid() -> tuple[ScorerCandidate, ...]:
    values = [ScorerCandidate(order=0, dim=0, alpha=100.0)]
    for dim in (512, 2_048, 8_192):
        for alpha in (1.0, 10.0, 100.0):
            values.append(ScorerCandidate(order=len(values), dim=dim, alpha=alpha))
    return tuple(values)


def _task_text(repository: str, prompt: str) -> str:
    return f"repository={repository}\n{prompt}"


def _structural(text: str) -> list[float]:
    """Return the fixed pre-inference structural feature block."""
    lower = text.casefold()
    lines = text.splitlines()
    path_tokens = sum(token.count("/") for token in text.split())
    stack_markers = sum(
        lower.count(marker)
        for marker in ("traceback", "stack trace", " at ", "exception:")
    )
    return [
        math.log1p(len(text)),
        math.log1p(len(text.split())),
        math.log1p(len(lines)),
        math.log1p(text.count("```")),
        math.log1p(stack_markers),
        math.log1p(path_tokens),
        math.log1p(text.count("`") + text.count('"') + text.count("'")),
        math.log1p(sum(lower.count(word) for word in ("fix", "bug", "repair"))),
        math.log1p(lower.count("test")),
        math.log1p(
            sum(lower.count(word) for word in ("dependency", "package", "build"))
        ),
        float("python" in lower or ".py" in lower),
        float("javascript" in lower or ".js" in lower or "node" in lower),
        float("typescript" in lower or ".ts" in lower),
        float("rust" in lower or ".rs" in lower or "cargo" in lower),
        float("golang" in lower or ".go" in lower),
    ]


def _vectorizer(dim: int) -> HashingVectorizer | None:
    if dim == 0:
        return None
    return HashingVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        n_features=dim,
        alternate_sign=True,
        norm="l2",
        lowercase=True,
    )


def _design(
    texts: list[str],
    vectorizer: HashingVectorizer | None,
) -> sparse.csr_matrix:
    structural = sparse.csr_matrix(
        np.asarray([_structural(text) for text in texts], dtype=np.float64)
    )
    if vectorizer is None:
        return structural
    hashed = vectorizer.transform(texts)
    return sparse.hstack([hashed, structural], format="csr")


def _rho(expected: np.ndarray, observed: np.ndarray) -> float:
    value = spearmanr(expected, observed).statistic
    return -1.0 if not math.isfinite(float(value)) else float(value)


def _select_scorer(
    tasks: list[ExternalTask],
) -> tuple[ScorerCandidate, list[dict[str, object]]]:
    """Select the frozen scorer with repository-grouped external folds."""
    texts = [_task_text(task.repository, task.prompt) for task in tasks]
    targets = np.asarray([task.target for task in tasks], dtype=np.float64)
    groups = np.asarray([task.repository for task in tasks])
    indices = np.arange(len(tasks), dtype=np.int64)
    folds = list(GroupKFold(n_splits=5).split(indices, groups=groups))
    results: list[dict[str, object]] = []
    for candidate in _scorer_grid():
        vectorizer = _vectorizer(candidate.dim)
        design = _design(texts, vectorizer)
        predictions = np.empty(len(tasks), dtype=np.float64)
        fold_rows: list[dict[str, object]] = []
        for fold, (train, test) in enumerate(folds):
            if set(groups[train]) & set(groups[test]):
                raise AssertionError("external repository crossed a scorer fold")
            model = Ridge(alpha=candidate.alpha, solver="lsqr")
            model.fit(design[train], targets[train])
            predictions[test] = model.predict(design[test])
            fold_rows.append(
                {
                    "fold": fold,
                    "tasks": len(test),
                    "repositories": len(set(groups[test])),
                    "mse": float(np.mean(np.square(predictions[test] - targets[test]))),
                    "spearman": _rho(predictions[test], targets[test]),
                }
            )
        results.append(
            {
                "key": candidate.key,
                "order": candidate.order,
                "dim": candidate.dim,
                "alpha": candidate.alpha,
                "grouped_mse": float(np.mean(np.square(predictions - targets))),
                "grouped_spearman": _rho(predictions, targets),
                "folds": fold_rows,
            }
        )
    winner = min(
        results,
        key=lambda row: (
            float(row["grouped_mse"]),
            -float(row["grouped_spearman"]),
            int(row["dim"]),
            -float(row["alpha"]),
            int(row["order"]),
        ),
    )
    selected = next(item for item in _scorer_grid() if item.key == winner["key"])
    return selected, results


def _fit_scorer(tasks: list[ExternalTask], candidate: ScorerCandidate) -> FittedScorer:
    texts = [_task_text(task.repository, task.prompt) for task in tasks]
    targets = np.asarray([task.target for task in tasks], dtype=np.float64)
    vectorizer = _vectorizer(candidate.dim)
    model = Ridge(alpha=candidate.alpha, solver="lsqr")
    model.fit(_design(texts, vectorizer), targets)
    return FittedScorer(candidate=candidate, vectorizer=vectorizer, model=model)


def _development_texts(path: Path, source: SourceData) -> list[str]:
    """Return calibration texts with the exact external scorer input contract."""
    raw_tasks = _read_object(path).get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("development corpus has no task rows")
    by_id = {
        str(row["task_id"]): _task_text(str(row["repository"]), str(row["prompt"]))
        for row in raw_tasks
        if isinstance(row, dict)
    }
    if any(task_id not in by_id for task_id in source.data.task_ids):
        raise ValueError("retained development task text is absent")
    return [by_id[task_id] for task_id in source.data.task_ids]


def _score_texts(texts: list[str], scorer: FittedScorer) -> np.ndarray:
    return np.asarray(
        scorer.model.predict(_design(texts, scorer.vectorizer)),
        dtype=np.float64,
    )


def _route(scores: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(scores <= threshold, MAX_INDEX, HIGH_INDEX).astype(np.int64)


def _metrics(source: SourceData, choices: np.ndarray, indices: np.ndarray) -> RouteMetrics:
    rewards = source.data.rewards[indices]
    costs = source.data.costs[indices]
    selected_rewards = rewards[np.arange(len(indices)), choices[indices]]
    selected_costs = costs[np.arange(len(indices)), choices[indices]]
    counts = np.bincount(choices[indices], minlength=len(ARMS))
    traffic = counts / len(indices)
    blind_rewards = rewards @ traffic
    blind_costs = costs @ traffic
    reward = float(np.mean(selected_rewards))
    cost = float(np.mean(selected_costs))
    static_rewards = rewards.mean(axis=0)
    static_costs = costs.mean(axis=0)
    strongest = float(np.max(static_rewards))
    dominated = tuple(
        ARMS[index]
        for index in range(len(ARMS))
        if static_rewards[index] >= reward - 1e-12 and static_costs[index] <= cost + 1e-12
    )
    return RouteMetrics(
        reward=reward,
        cost_usd=cost,
        blind_reward=float(np.mean(blind_rewards)),
        blind_cost_usd=float(np.mean(blind_costs)),
        advantage=reward - float(np.mean(blind_rewards)),
        retention=1.0 if strongest <= 0.0 else reward / strongest,
        dominated_by_static=dominated,
        arm_counts={ARMS[index]: int(value) for index, value in enumerate(counts) if value},
    )


def _null_scores(source: SourceData, scores: np.ndarray) -> np.ndarray:
    """Build frozen equal-length repository-block permutations within language."""
    blocks: dict[tuple[str, int], list[np.ndarray]] = {}
    repositories = np.asarray(source.repositories)
    languages = np.asarray(source.languages)
    for repository in sorted(set(source.repositories)):
        indices = np.flatnonzero(repositories == repository)
        block_languages = set(languages[indices])
        if len(block_languages) != 1:
            raise ValueError(f"repository spans languages: {repository}")
        key = (str(languages[indices[0]]), len(indices))
        blocks.setdefault(key, []).append(indices)
    rng = np.random.default_rng(NULL_SEED)
    values = np.empty((NULL_COUNT, len(scores)), dtype=np.float64)
    for null_index in range(NULL_COUNT):
        shuffled = np.empty_like(scores)
        for key in sorted(blocks):
            recipients = blocks[key]
            order = rng.permutation(len(recipients))
            for recipient, donor_index in zip(recipients, order, strict=True):
                donor = recipients[int(donor_index)]
                shuffled[recipient] = scores[donor]
        values[null_index] = shuffled
    return values


def _higher_quantile(values: np.ndarray) -> float:
    return float(np.quantile(values, 0.95, method="higher"))


def _latencies_ms(texts: list[str], scorer: FittedScorer) -> list[float]:
    values: list[float] = []
    for text in texts:
        for _ in range(5):
            started = time.perf_counter_ns()
            score = float(scorer.model.predict(_design([text], scorer.vectorizer))[0])
            _ = MAX_INDEX if score <= 0.5 else HIGH_INDEX
            values.append((time.perf_counter_ns() - started) / 1_000_000)
    return values


def _as_dict(metrics: RouteMetrics) -> dict[str, object]:
    return {
        "reward": metrics.reward,
        "cost_usd_per_task": metrics.cost_usd,
        "matched_blind_reward": metrics.blind_reward,
        "matched_blind_cost_usd_per_task": metrics.blind_cost_usd,
        "matched_blind_advantage": metrics.advantage,
        "quality_retention": metrics.retention,
        "dominated_by_static": list(metrics.dominated_by_static),
        "arm_counts": metrics.arm_counts,
    }


def fit(
    external_tasks_path: Path,
    external_manifest_path: Path,
    development_corpus_path: Path,
    outcomes_path: Path,
    audit_path: Path,
    output: Path,
) -> None:
    """Run external scorer selection and null-penalized development calibration."""
    if output.exists():
        raise FileExistsError(f"fit output already exists: {output}")
    if _sha256(development_corpus_path) != DEVELOPMENT_CORPUS_SHA256:
        raise ValueError("development corpus hash changed")
    external = _read_external(external_tasks_path, external_manifest_path)
    source = load_source(development_corpus_path, outcomes_path, audit_path)
    selected_scorer, scorer_results = _select_scorer(external)
    fitted = _fit_scorer(external, selected_scorer)
    development_texts = _development_texts(development_corpus_path, source)
    scores = _score_texts(development_texts, fitted)
    null_scores = _null_scores(source, scores)
    indices = np.arange(len(source.data.task_ids), dtype=np.int64)
    fold_pairs = list(
        GroupKFold(n_splits=5).split(
            indices,
            groups=np.asarray(source.repositories),
        )
    )
    fold_tests = [test for _, test in fold_pairs]
    thresholds = [float(np.percentile(scores, value)) for value in THRESHOLD_PERCENTILES]
    candidate_rows: list[dict[str, object]] = []
    route_choices: list[np.ndarray] = []
    null_choices: list[list[np.ndarray]] = []
    for order, (percentile, threshold) in enumerate(
        zip(THRESHOLD_PERCENTILES, thresholds, strict=True)
    ):
        choices = _route(scores, threshold)
        permutations = [_route(row, threshold) for row in null_scores]
        route_choices.append(choices)
        null_choices.append(permutations)
        metrics = _metrics(source, choices, indices)
        null_advantages = np.asarray(
            [_metrics(source, row, indices).advantage for row in permutations]
        )
        fold_rows: list[dict[str, object]] = []
        positive_fold_margins = 0
        for fold, test in enumerate(fold_tests):
            fold_metrics = _metrics(source, choices, test)
            fold_null = np.asarray(
                [_metrics(source, row, test).advantage for row in permutations]
            )
            null95 = _higher_quantile(fold_null)
            margin = fold_metrics.advantage - null95
            positive_fold_margins += int(margin > 0.0)
            fold_rows.append(
                {
                    "fold": fold,
                    **_as_dict(fold_metrics),
                    "null_advantage_p95": null95,
                    "real_minus_null_p95": margin,
                }
            )
        null95 = _higher_quantile(null_advantages)
        candidate_rows.append(
            {
                "order": order,
                "percentile": percentile,
                "threshold": threshold,
                **_as_dict(metrics),
                "null_advantage_p95": null95,
                "real_minus_null_p95": metrics.advantage - null95,
                "positive_fold_null_margins": positive_fold_margins,
                "all_fold_retention_passed": all(
                    float(row["quality_retention"]) >= QUALITY_RETENTION
                    for row in fold_rows
                ),
                "folds": fold_rows,
            }
        )
    stability_winners: list[int | None] = []
    for heldout_fold, (train, _) in enumerate(fold_pairs):
        eligible: list[tuple[tuple[float, float, float, int], int]] = []
        for order, row in enumerate(candidate_rows):
            metrics = _metrics(source, route_choices[order], train)
            advantages = np.asarray(
                [
                    _metrics(source, choices, train).advantage
                    for choices in null_choices[order]
                ]
            )
            null95 = _higher_quantile(advantages)
            if (
                metrics.retention >= QUALITY_RETENTION
                and metrics.advantage > 0.0
                and not metrics.dominated_by_static
                and metrics.advantage > null95
            ):
                eligible.append(
                    (
                        (
                            metrics.cost_usd,
                            -(metrics.advantage - null95),
                            -metrics.reward,
                            int(row["order"]),
                        ),
                        order,
                    )
                )
        winner = min(eligible)[1] if eligible else None
        stability_winners.append(winner)
        logger.info("stability heldout_fold=%d winner=%s", heldout_fold, winner)
    latency_values = _latencies_ms(development_texts, fitted)
    latency_p95 = float(np.percentile(np.asarray(latency_values), 95))
    eligible_rows: list[dict[str, object]] = []
    for order, row in enumerate(candidate_rows):
        stability_count = sum(winner == order for winner in stability_winners)
        row["stability_selection_count"] = stability_count
        row["route_latency_p95_ms"] = latency_p95
        row["eligible"] = bool(
            row["all_fold_retention_passed"]
            and float(row["matched_blind_advantage"]) > 0.0
            and not row["dominated_by_static"]
            and float(row["real_minus_null_p95"]) > 0.0
            and int(row["positive_fold_null_margins"]) >= 4
            and stability_count >= 4
            and latency_p95 < MAX_ROUTE_P95_MS
        )
        if row["eligible"]:
            eligible_rows.append(row)
    selected = (
        min(
            eligible_rows,
            key=lambda row: (
                float(row["cost_usd_per_task"]),
                -float(row["real_minus_null_p95"]),
                -float(row["reward"]),
                int(row["order"]),
            ),
        )
        if eligible_rows
        else None
    )
    output.mkdir(parents=True)
    report = {
        "protocol": PROTOCOL,
        "development_passed": selected is not None,
        "external_tasks_sha256": _sha256(external_tasks_path),
        "external_manifest_sha256": _sha256(external_manifest_path),
        "development_corpus_sha256": _sha256(development_corpus_path),
        "development_outcomes_sha256": _sha256(outcomes_path),
        "development_audit_sha256": _sha256(audit_path),
        "external_tasks": len(external),
        "external_repositories": len({task.repository for task in external}),
        "development_tasks": len(source.data.task_ids),
        "development_repositories": len(set(source.repositories)),
        "selected_scorer": {
            "key": selected_scorer.key,
            "dim": selected_scorer.dim,
            "alpha": selected_scorer.alpha,
        },
        "scorer_candidates": scorer_results,
        "null_count": NULL_COUNT,
        "null_seed": NULL_SEED,
        "null_method": "repository-block-within-language-and-task-count",
        "stability_winners": stability_winners,
        "route_latency_p95_ms": latency_p95,
        "route_latency_samples": len(latency_values),
        "candidates": candidate_rows,
        "selected": selected,
        "target_outcomes_used": False,
        "deep_swe_outcomes_accessed": False,
        "internet_access": False,
        "fitted_numeric_router_state_persisted": False,
    }
    report_path = output / "development-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if selected is not None:
        order = int(selected["order"])
        route_path = output / "development-routes.jsonl"
        route_path.write_text(
            "".join(
                json.dumps(
                    {
                        "task_id": task_id,
                        "repository": source.repositories[index],
                        "arm": ARMS[int(route_choices[order][index])],
                        "target_outcomes_used": False,
                    },
                    sort_keys=True,
                )
                + "\n"
                for index, task_id in enumerate(source.data.task_ids)
            )
        )
        lock = {
            "protocol": PROTOCOL,
            "eligible": True,
            "scorer": report["selected_scorer"],
            "threshold_percentile": selected["percentile"],
            "threshold_value": selected["threshold"],
            "harder_direction": "lower-score",
            "arms": {"default": ARMS[HIGH_INDEX], "hard": ARMS[MAX_INDEX]},
            "external_tasks_sha256": report["external_tasks_sha256"],
            "development_outcomes_sha256": report["development_outcomes_sha256"],
            "development_report_sha256": _sha256(report_path),
            "development_routes_sha256": _sha256(route_path),
            "null_count": NULL_COUNT,
            "null_seed": NULL_SEED,
            "target_outcomes_used": False,
            "deep_swe_outcomes_accessed": False,
            "fitted_numeric_router_state_persisted": False,
        }
        (output / "selection-lock.json").write_text(
            json.dumps(lock, indent=2, sort_keys=True) + "\n"
        )
    logger.info(
        "null fit passed=%s scorer=%s eligible=%d latency_p95_ms=%.3f",
        selected is not None,
        selected_scorer.key,
        len(eligible_rows),
        latency_p95,
    )


def main() -> None:
    """Run the null-penalized fit CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-tasks", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--development-corpus", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fit(
        args.external_tasks,
        args.external_manifest,
        args.development_corpus,
        args.outcomes,
        args.audit,
        args.output,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
