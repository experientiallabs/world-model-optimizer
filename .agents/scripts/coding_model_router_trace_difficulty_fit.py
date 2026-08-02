"""Fit an external trace-trained difficulty router without target outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import coding_model_router_model_effort_fit as base
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge

PROTOCOL = "coding-router-trace-difficulty-selection-v1"
TRACE_DATASET = "chilomax/SWE-smith-trajectories"
TRACE_CONFIG = "default"
TRACE_SPLIT = "tool"
TRACE_COMMIT = "3bcb3dd101c2930e3386e0103dd9fde084587a1c"
TRACE_ROWS = 24_100
TRACE_SHARDS = tuple(f"{index:04d}.parquet" for index in range(8))
HASH_DIMS = (2_048, 8_192, 32_768)
RIDGE_ALPHAS = (1.0, 10.0, 100.0)
ROUTE_PERCENTILES = tuple(range(5, 100, 5))
GUARD_ARM = "sol-max"
QUALITY_RETENTION = 0.95
MIN_SAVINGS = 0.40
NULL_COUNT = 128
MAX_ROUTE_P95_MS = 5.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return {str(key): item for key, item in value.items()}


def _initial_user_text(raw: str) -> str:
    messages = json.loads(raw)
    if not isinstance(messages, list):
        raise ValueError("trace messages are not a list")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts = [
                str(item["text"])
                for item in content
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            ]
            if not parts:
                continue
            text = "\n".join(parts)
        else:
            continue
        opening = "<pr_description>"
        closing = "</pr_description>"
        start = text.find(opening)
        end = text.find(closing, start + len(opening))
        if start >= 0 and end > start:
            return text[start + len(opening) : end].strip()
        return text
    raise ValueError("trace lacks an initial user task description")


def _repository(instance_id: str) -> str:
    value = instance_id.split(".", 1)[0]
    if "__" not in value:
        raise ValueError(f"cannot derive repository from {instance_id!r}")
    return value


def load_traces(trace_dir: Path) -> tuple[list[str], list[str], np.ndarray, dict[str, Any]]:
    """Load and aggregate the frozen trace split by task instance."""
    prompts: dict[tuple[str, str], str] = {}
    labels: dict[tuple[str, str], list[float]] = defaultdict(list)
    instance_ids: set[str] = set()
    shard_rows: dict[str, int] = {}
    shard_sha256: dict[str, str] = {}
    seen_trajectories: set[str] = set()
    for name in TRACE_SHARDS:
        path = trace_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        table = pq.read_table(
            path,
            columns=["messages", "instance_id", "resolved", "traj_id"],
        )
        shard_rows[name] = table.num_rows
        shard_sha256[name] = _sha256(path)
        for row in table.to_pylist():
            instance_id = row.get("instance_id")
            resolved = row.get("resolved")
            trajectory_id = row.get("traj_id")
            raw_messages = row.get("messages")
            if (
                not isinstance(instance_id, str)
                or not isinstance(resolved, bool)
                or not isinstance(trajectory_id, str)
                or trajectory_id in seen_trajectories
                or not isinstance(raw_messages, str)
            ):
                raise ValueError("trace row identity or label is invalid")
            seen_trajectories.add(trajectory_id)
            prompt = _initial_user_text(raw_messages)
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            key = (instance_id, prompt_hash)
            prompts[key] = prompt
            labels[key].append(float(resolved))
            instance_ids.add(instance_id)
    if sum(shard_rows.values()) != TRACE_ROWS or len(seen_trajectories) != TRACE_ROWS:
        raise ValueError("trace split row count or trajectory identities changed")
    prompt_keys = sorted(prompts)
    texts = [prompts[key] for key in prompt_keys]
    groups = [_repository(key[0]) for key in prompt_keys]
    targets = np.asarray(
        [float(np.mean(labels[key])) for key in prompt_keys],
        dtype=np.float64,
    )
    provenance = {
        "dataset": TRACE_DATASET,
        "config": TRACE_CONFIG,
        "split": TRACE_SPLIT,
        "commit": TRACE_COMMIT,
        "rows": TRACE_ROWS,
        "instances": len(instance_ids),
        "instance_prompt_variants": len(prompt_keys),
        "repositories": len(set(groups)),
        "resolved_rate": float(targets.mean()),
        "shard_rows": shard_rows,
        "shard_sha256": shard_sha256,
        "assistant_or_tool_text_used": False,
        "patches_used": False,
    }
    return texts, groups, targets, provenance


def _vectorizer(dim: int) -> HashingVectorizer:
    return HashingVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        n_features=dim,
        alternate_sign=True,
        norm="l2",
        lowercase=True,
    )


def _correlation(observed: np.ndarray, predicted: np.ndarray) -> float:
    value = float(spearmanr(observed, predicted).statistic)
    return value if math.isfinite(value) else 0.0


def select_trace_model(
    texts: list[str],
    groups: list[str],
    targets: np.ndarray,
) -> tuple[int | None, float | None, list[dict[str, Any]]]:
    """Select one trace-only estimator using repository-grouped CV."""
    features = {dim: _vectorizer(dim).transform(texts).tocsr() for dim in HASH_DIMS}
    rows: list[dict[str, Any]] = []
    for dim in HASH_DIMS:
        for alpha in RIDGE_ALPHAS:
            seed_scores = []
            for seed in base.SEEDS:
                folds = base.grouped_folds(groups, seed)
                predicted = np.empty(len(texts), dtype=np.float64)
                for fold in range(base.FOLDS):
                    train = np.flatnonzero(folds != fold)
                    test = np.flatnonzero(folds == fold)
                    model = Ridge(alpha=alpha)
                    model.fit(features[dim][train], targets[train])
                    predicted[test] = model.predict(features[dim][test])
                seed_scores.append(_correlation(targets, predicted))
            rows.append(
                {
                    "dim": dim,
                    "alpha": alpha,
                    "seed_spearman": dict(zip(base.SEEDS, seed_scores, strict=True)),
                    "mean_spearman": float(np.mean(seed_scores)),
                    "eligible": all(score > 0.0 for score in seed_scores)
                    and float(np.mean(seed_scores)) > 0.0,
                }
            )
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        return None, None, rows
    selected = min(
        eligible,
        key=lambda row: (-float(row["mean_spearman"]), int(row["dim"]), float(row["alpha"])),
    )
    return int(selected["dim"]), float(selected["alpha"]), rows


def _fit_scores(
    train_texts: list[str],
    targets: np.ndarray,
    route_texts: list[str],
    dim: int,
    alpha: float,
) -> np.ndarray:
    vectorizer = _vectorizer(dim)
    train = vectorizer.transform(train_texts).tocsr()
    route = vectorizer.transform(route_texts).tocsr()
    model = Ridge(alpha=alpha)
    model.fit(train, targets)
    return np.asarray(model.predict(route), dtype=np.float64)


def _choices(scores: np.ndarray, alternate: int, percentile: int) -> np.ndarray:
    guard = base.ARMS.index(GUARD_ARM)
    count = max(1, min(len(scores) - 1, round(len(scores) * percentile / 100)))
    order = np.argsort(-scores, kind="stable")
    choices = np.full(len(scores), guard, dtype=np.int64)
    choices[order[:count]] = alternate
    return choices


def _metric_json(metrics: base.Metrics) -> dict[str, Any]:
    return {
        "reward": metrics.reward,
        "cost_usd_per_task": metrics.cost_usd,
        "quality_retention": metrics.retention,
        "cost_savings": metrics.savings,
        "matched_blind_reward": metrics.blind_reward,
        "matched_blind_cost_usd_per_task": metrics.blind_cost_usd,
        "matched_blind_advantage": metrics.advantage,
        "expensive_traffic": metrics.expensive_traffic,
        "dominated_by_static": list(metrics.dominated_by_static),
    }


def select_route(
    data: base.Data, scores: np.ndarray
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Select the cheapest eligible guard pair and ease-score rank."""
    guard = base.ARMS.index(GUARD_ARM)
    evaluated: list[dict[str, Any]] = []
    eligible: list[tuple[float, float, int, int, dict[str, Any]]] = []
    for alternate in range(len(base.ARMS)):
        if alternate == guard:
            continue
        for percentile in ROUTE_PERCENTILES:
            choices = _choices(scores, alternate, percentile)
            metrics = base._metrics(data, choices, alternate, guard)
            passed = (
                metrics.retention >= QUALITY_RETENTION
                and metrics.savings >= MIN_SAVINGS
                and metrics.advantage > 0.0
                and not metrics.dominated_by_static
            )
            row = {
                "alternate_arm": base.ARMS[alternate],
                "easy_percentile": percentile,
                "eligible": passed,
                **_metric_json(metrics),
            }
            evaluated.append(row)
            if passed:
                eligible.append(
                    (metrics.cost_usd, -metrics.reward, alternate, percentile, row)
                )
    if not eligible:
        return None, evaluated
    _, _, alternate, percentile, row = min(eligible)
    return (
        {
            "alternate": alternate,
            "alternate_arm": base.ARMS[alternate],
            "percentile": percentile,
            "metrics": row,
            "eligible_count": len(eligible),
        },
        evaluated,
    )


def _route_rows(
    tasks: list[dict[str, Any]],
    choices: np.ndarray,
    selected_candidate: str,
    *,
    null_index: int | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for task, choice in zip(tasks, choices, strict=True):
        arm = base.ARMS[int(choice)]
        prefix, effort = arm.split("-", 1)
        row = {
            "task_id": str(task["task_id"]),
            "arm": arm,
            "model": base.MODEL_IDS[prefix],
            "reasoning_effort": effort,
            "selected_candidate": selected_candidate,
            "target_outcomes_used": False,
            "deep_swe_outcomes_accessed": False,
        }
        if null_index is not None:
            row["null_index"] = null_index
        rows.append(row)
    return rows


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _null_choices(
    choices: np.ndarray,
    repositories: list[str],
    seed: int,
) -> np.ndarray:
    by_size: dict[int, list[str]] = defaultdict(list)
    indices: dict[str, list[int]] = defaultdict(list)
    for index, repository in enumerate(repositories):
        indices[repository].append(index)
    for repository, values in indices.items():
        by_size[len(values)].append(repository)
    result = np.empty_like(choices)
    rng = np.random.default_rng(seed)
    for repositories_of_size in by_size.values():
        targets = sorted(repositories_of_size)
        sources = list(rng.permutation(targets))
        for target, source in zip(targets, sources, strict=True):
            result[indices[target]] = choices[indices[source]]
    return result


def fit(
    trace_dir: Path,
    development_corpus: Path,
    confirmation_corpus: Path,
    outcomes: Path,
    audit_path: Path,
    output: Path,
) -> None:
    """Fit trace difficulty ephemerally and freeze routes only when gates pass."""
    trace_texts, trace_groups, targets, provenance = load_traces(trace_dir)
    dim, alpha, trace_grid = select_trace_model(trace_texts, trace_groups, targets)
    output.mkdir(parents=True, exist_ok=False)
    inputs = {
        "development_corpus_sha256": _sha256(development_corpus),
        "confirmation_corpus_sha256": _sha256(confirmation_corpus),
        "outcomes_sha256": _sha256(outcomes),
        "completion_audit_sha256": _sha256(audit_path),
    }
    if dim is None or alpha is None:
        report = {
            "protocol": PROTOCOL,
            "valid": True,
            "development_passed": False,
            "confirmation_authorized": False,
            "reason": "trace-only difficulty model failed frozen external CV gate",
            "trace_source": provenance,
            "trace_model": {"grid": trace_grid},
            "inputs": inputs,
            "target_outcomes_used": False,
            "deep_swe_outcomes_accessed": False,
            "fitted_numeric_router_state_persisted": False,
        }
        (output / "selection-report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return
    data = base.load_data(development_corpus, outcomes, audit_path)
    confirmation = base.load_confirmation(confirmation_corpus)
    development_scores = _fit_scores(
        trace_texts, targets, data.texts, dim, alpha
    )
    confirmation_texts = [base._task_text(task) for task in confirmation]
    confirmation_scores = _fit_scores(
        trace_texts, targets, confirmation_texts, dim, alpha
    )
    selected_route, route_grid = select_route(data, development_scores)
    trace_summary = {
        "selected_dim": dim,
        "selected_alpha": alpha,
        "grid": trace_grid,
    }
    if selected_route is None:
        near_misses = sorted(
            route_grid,
            key=lambda row: (
                -float(row["quality_retention"]),
                -float(row["cost_savings"]),
                -float(row["matched_blind_advantage"]),
                str(row["alternate_arm"]),
                int(row["easy_percentile"]),
            ),
        )[:50]
        report = {
            "protocol": PROTOCOL,
            "valid": True,
            "development_passed": False,
            "confirmation_authorized": False,
            "reason": "no trace-difficulty route passed frozen development gates",
            "candidate_count": 266,
            "trace_source": provenance,
            "trace_model": trace_summary,
            "top_near_misses": near_misses,
            "inputs": inputs,
            "target_outcomes_used": False,
            "deep_swe_outcomes_accessed": False,
            "fitted_numeric_router_state_persisted": False,
        }
        (output / "selection-report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return

    alternate = int(selected_route["alternate"])
    percentile = int(selected_route["percentile"])
    selected_candidate = (
        f"trace-difficulty-{base.ARMS[alternate]}__{GUARD_ARM}"
        f"-hash{dim}-a{alpha:g}-easy{percentile}"
    )
    confirmation_choices = _choices(confirmation_scores, alternate, percentile)
    real_rows = _route_rows(confirmation, confirmation_choices, selected_candidate)
    alternate_count = int(np.sum(confirmation_choices == alternate))
    blind_order = sorted(
        range(len(confirmation)),
        key=lambda index: hashlib.sha256(
            f"trace-difficulty-blind|{confirmation[index]['task_id']}".encode()
        ).hexdigest(),
    )
    guard = base.ARMS.index(GUARD_ARM)
    blind = np.full(len(confirmation), guard, dtype=np.int64)
    blind[blind_order[:alternate_count]] = alternate
    blind_rows = _route_rows(confirmation, blind, selected_candidate)

    repositories = [str(task["repository"]) for task in confirmation]
    null_rows: list[dict[str, Any]] = []
    route_hashes: set[str] = set()
    null_seeds: list[int] = []
    seed = 20_260_801
    while len(null_seeds) < NULL_COUNT and seed < 20_270_801:
        null = _null_choices(confirmation_choices, repositories, seed)
        route_hash = hashlib.sha256(null.tobytes()).hexdigest()
        if route_hash not in route_hashes and not np.array_equal(null, confirmation_choices):
            null_rows.extend(
                _route_rows(
                    confirmation,
                    null,
                    selected_candidate,
                    null_index=len(null_seeds),
                )
            )
            route_hashes.add(route_hash)
            null_seeds.append(seed)
        seed += 1
    if len(null_seeds) != NULL_COUNT:
        raise ValueError("could not freeze 128 unique repository-group null routes")

    routes_path = output / "confirmation-routes.jsonl"
    blind_path = output / "confirmation-blind-routes.jsonl"
    null_path = output / "confirmation-null-routes.jsonl"
    _write_rows(routes_path, real_rows)
    _write_rows(blind_path, blind_rows)
    _write_rows(null_path, null_rows)

    lookup = {row["task_id"]: row["arm"] for row in real_rows}
    latencies = []
    for task in confirmation:
        start = time.perf_counter()
        lookup[str(task["task_id"])]
        latencies.append((time.perf_counter() - start) * 1000.0)
    latency_p95 = float(np.percentile(latencies, 95))
    if latency_p95 >= MAX_ROUTE_P95_MS:
        raise ValueError(f"route lookup exceeds latency gate: {latency_p95:.6f} ms")

    selected_pair = [base.ARMS[alternate], GUARD_ARM]
    lock = {
        "protocol": PROTOCOL,
        "valid": True,
        "selected_candidate": selected_candidate,
        "selected_pair": selected_pair,
        "development_static_baseline": GUARD_ARM,
        "selected_config": {
            "family": "trace-difficulty-ridge-rank",
            "dim": dim,
            "alpha": alpha,
            "easy_percentile": percentile,
        },
        "inputs": inputs,
        "trace_source": provenance,
        "confirmation_routes_sha256": _sha256(routes_path),
        "confirmation_blind_routes_sha256": _sha256(blind_path),
        "confirmation_null_routes_sha256": _sha256(null_path),
        "null_count": NULL_COUNT,
        "null_unique_route_hashes": len(route_hashes),
        "null_seeds": null_seeds,
        "target_outcomes_used": False,
        "deep_swe_outcomes_accessed": False,
        "fitted_numeric_router_state_persisted": False,
    }
    lock_path = output / "selection-lock.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "protocol": PROTOCOL,
        "valid": True,
        "development_passed": True,
        "confirmation_authorized": True,
        "candidate_count": 266,
        "eligible_count": selected_route["eligible_count"],
        "selected_candidate": selected_candidate,
        "selected_pair": selected_pair,
        "development_static_baseline": GUARD_ARM,
        "selected_development_metrics": selected_route["metrics"],
        "trace_source": provenance,
        "trace_model": trace_summary,
        "confirmation_traffic": {
            base.ARMS[alternate]: alternate_count,
            GUARD_ARM: len(confirmation) - alternate_count,
        },
        "matched_blind_traffic_identical": True,
        "route_latency_p95_ms": latency_p95,
        "inputs": inputs,
        "selection_lock_sha256": _sha256(lock_path),
        "confirmation_routes_sha256": _sha256(routes_path),
        "confirmation_blind_routes_sha256": _sha256(blind_path),
        "confirmation_null_routes_sha256": _sha256(null_path),
        "target_outcomes_used": False,
        "deep_swe_outcomes_accessed": False,
        "fitted_numeric_router_state_persisted": False,
    }
    (output / "selection-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--development-corpus", type=Path, required=True)
    parser.add_argument("--confirmation-corpus", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fit(
        args.trace_dir,
        args.development_corpus,
        args.confirmation_corpus,
        args.outcomes,
        args.audit,
        args.output,
    )


if __name__ == "__main__":
    main()
