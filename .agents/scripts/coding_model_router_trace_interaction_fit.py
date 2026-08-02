"""Learn a task by model interaction metric from public coding traces for WMO kNN."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import coding_model_router_model_effort_fit as base
import coding_model_router_semantic_knn_fit as semantic
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

PROTOCOL = "coding-router-trace-interaction-v1"
TRACE_DATASET = "chilomax/SWE-smith-trajectories"
TRACE_CONFIG = "default"
TRACE_SPLIT = "tool"
TRACE_COMMIT = "3bcb3dd101c2930e3386e0103dd9fde084587a1c"
TRACE_ROWS = 24_100
TRACE_SHARDS = tuple(f"{index:04d}.parquet" for index in range(8))
SOURCE_MODELS = (
    "claude-3-5-sonnet-20241022",
    "claude-3-7-sonnet-20250219",
    "gpt-4o-2024-08-06",
)
SOURCE_PAIRS = tuple(combinations(SOURCE_MODELS, 2))
MIN_PAIR_VARIANTS = 100
RIDGE_ALPHAS = (1.0, 10.0, 100.0)


@dataclass(frozen=True)
class TraceData:
    """Exact public trace prompt variants with source-model resolved rates."""

    texts: list[str]
    groups: list[str]
    model_rewards: list[dict[str, float]]
    keys: list[tuple[str, str]]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class Target:
    """One source outcome coordinate and the trace rows on which it is observed."""

    name: str
    indices: np.ndarray
    values: np.ndarray
    pair_specific: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_traces(trace_dir: Path) -> TraceData:
    """Load exact prompt variants that have outcomes from at least two source models."""
    prompts: dict[tuple[str, str], str] = {}
    labels: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    seen: set[str] = set()
    shard_rows: dict[str, int] = {}
    shard_sha256: dict[str, str] = {}
    instances: set[str] = set()
    for name in TRACE_SHARDS:
        path = trace_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        table = pq.read_table(
            path,
            columns=["messages", "instance_id", "resolved", "model", "traj_id"],
        )
        shard_rows[name] = table.num_rows
        shard_sha256[name] = _sha256(path)
        for row in table.to_pylist():
            raw = row.get("messages")
            instance_id = row.get("instance_id")
            resolved = row.get("resolved")
            model = row.get("model")
            trajectory_id = row.get("traj_id")
            if (
                not isinstance(raw, str)
                or not isinstance(instance_id, str)
                or not isinstance(resolved, bool)
                or model not in SOURCE_MODELS
                or not isinstance(trajectory_id, str)
                or trajectory_id in seen
            ):
                raise ValueError("trace row identity, model, or outcome is invalid")
            seen.add(trajectory_id)
            instances.add(instance_id)
            prompt = _initial_user_text(raw)
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            key = (instance_id, prompt_hash)
            prompts[key] = prompt
            labels[(instance_id, prompt_hash, str(model))].append(float(resolved))
    if len(seen) != TRACE_ROWS or sum(shard_rows.values()) != TRACE_ROWS:
        raise ValueError("trace row count or trajectory identities changed")

    rewards: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for (instance_id, prompt_hash, model), values in labels.items():
        rewards[(instance_id, prompt_hash)][model] = float(np.mean(values))
    keys = sorted(key for key, values in rewards.items() if len(values) >= 2)
    pair_counts = {
        f"{left}__{right}": sum(
            left in rewards[key] and right in rewards[key] for key in keys
        )
        for left, right in SOURCE_PAIRS
    }
    return TraceData(
        texts=[prompts[key] for key in keys],
        groups=[_repository(key[0]) for key in keys],
        model_rewards=[rewards[key] for key in keys],
        keys=keys,
        provenance={
            "dataset": TRACE_DATASET,
            "config": TRACE_CONFIG,
            "split": TRACE_SPLIT,
            "commit": TRACE_COMMIT,
            "rows": TRACE_ROWS,
            "instances": len(instances),
            "exact_multi_model_prompt_variants": len(keys),
            "repositories": len({_repository(key[0]) for key in keys}),
            "pair_variant_counts": pair_counts,
            "shard_rows": shard_rows,
            "shard_sha256": shard_sha256,
            "assistant_or_tool_text_used": False,
            "patches_used": False,
        },
    )


def build_targets(data: TraceData) -> list[Target]:
    """Build the generic source success coordinate and eligible pair residual coordinates."""
    targets = [
        Target(
            name="generic-resolution",
            indices=np.arange(len(data.texts), dtype=np.int64),
            values=np.asarray(
                [float(np.mean(list(rewards.values()))) for rewards in data.model_rewards],
                dtype=np.float64,
            ),
            pair_specific=False,
        )
    ]
    for left, right in SOURCE_PAIRS:
        indices = np.asarray(
            [
                index
                for index, rewards in enumerate(data.model_rewards)
                if left in rewards and right in rewards
            ],
            dtype=np.int64,
        )
        if len(indices) < MIN_PAIR_VARIANTS:
            continue
        targets.append(
            Target(
                name=f"{left}__minus__{right}",
                indices=indices,
                values=np.asarray(
                    [
                        data.model_rewards[index][left] - data.model_rewards[index][right]
                        for index in indices
                    ],
                    dtype=np.float64,
                ),
                pair_specific=True,
            )
        )
    return targets


def _correlation(observed: np.ndarray, predicted: np.ndarray) -> float:
    value = float(spearmanr(observed, predicted).statistic)
    return value if math.isfinite(value) else 0.0


def select_alpha(
    data: TraceData,
    targets: list[Target],
    vectors: np.ndarray,
) -> tuple[float | None, list[dict[str, Any]]]:
    """Select a single Ridge strength using only repository-grouped public trace CV."""
    if sum(target.pair_specific for target in targets) < 2:
        return None, []
    rows: list[dict[str, Any]] = []
    for alpha in RIDGE_ALPHAS:
        seed_reports: list[dict[str, Any]] = []
        for seed in base.SEEDS:
            target_scores: dict[str, float] = {}
            for target in targets:
                groups = [data.groups[index] for index in target.indices]
                folds = base.grouped_folds(groups, seed)
                predicted = np.empty(len(target.indices), dtype=np.float64)
                source = vectors[target.indices]
                for fold in range(base.FOLDS):
                    train = np.flatnonzero(folds != fold)
                    test = np.flatnonzero(folds == fold)
                    model = Ridge(alpha=alpha)
                    model.fit(source[train], target.values[train])
                    predicted[test] = model.predict(source[test])
                target_scores[target.name] = _correlation(target.values, predicted)
            seed_reports.append(
                {
                    "seed": seed,
                    "target_spearman": target_scores,
                    "mean_spearman": float(np.mean(list(target_scores.values()))),
                }
            )
        target_means = {
            target.name: float(
                np.mean(
                    [report["target_spearman"][target.name] for report in seed_reports]
                )
            )
            for target in targets
        }
        pair_positive = sum(
            target.pair_specific and target_means[target.name] > 0.0 for target in targets
        )
        mean_score = float(np.mean(list(target_means.values())))
        rows.append(
            {
                "alpha": alpha,
                "mean_spearman": mean_score,
                "target_mean_spearman": target_means,
                "positive_pair_targets": pair_positive,
                "seeds": seed_reports,
                "eligible": mean_score > 0.0
                and all(float(report["mean_spearman"]) > 0.0 for report in seed_reports)
                and pair_positive >= 2,
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        return None, rows
    selected = min(
        eligible,
        key=lambda row: (-float(row["mean_spearman"]), float(row["alpha"])),
    )
    return float(selected["alpha"]), rows


def fit_interaction_vectors(
    data: TraceData,
    targets: list[Target],
    source_vectors: np.ndarray,
    route_vectors: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit trace-only coordinates and predict normalized representations for route tasks."""
    source_coordinates: list[np.ndarray] = []
    route_coordinates: list[np.ndarray] = []
    coordinates: list[dict[str, Any]] = []
    for target in targets:
        model = Ridge(alpha=alpha)
        model.fit(source_vectors[target.indices], target.values)
        source_prediction = np.asarray(model.predict(source_vectors), dtype=np.float64)
        route_prediction = np.asarray(model.predict(route_vectors), dtype=np.float64)
        mean = float(source_prediction.mean())
        std = float(source_prediction.std())
        if std <= np.finfo(np.float64).eps:
            raise ValueError(f"trace interaction coordinate {target.name} has zero variance")
        source_coordinates.append((source_prediction - mean) / std)
        route_coordinates.append((route_prediction - mean) / std)
        coordinates.append(
            {
                "name": target.name,
                "training_rows": len(target.indices),
                "training_target_mean": float(target.values.mean()),
                "prediction_mean": mean,
                "prediction_std": std,
            }
        )
    source_matrix = np.column_stack(source_coordinates)
    route_matrix = np.column_stack(route_coordinates)
    norms = np.linalg.norm(route_matrix, axis=1, keepdims=True)
    route_matrix = route_matrix / np.maximum(norms, np.finfo(np.float64).eps)
    return route_matrix.astype(np.float32), {
        "dimensions": int(route_matrix.shape[1]),
        "coordinates": coordinates,
        "source_representation_sha256": hashlib.sha256(
            source_matrix.astype(np.float32).tobytes()
        ).hexdigest(),
    }


def run(args: argparse.Namespace) -> int:
    """Run source pretraining and conditionally evaluate the frozen external WMO grid."""
    trace_data = load_traces(args.trace_dir)
    targets = build_targets(trace_data)
    source_vectors, source_embedding = semantic._embed(
        trace_data.texts,
        model_path=args.embedding_model,
        tokenizer_path=args.tokenizer,
    )
    alpha, alpha_grid = select_alpha(trace_data, targets, source_vectors)
    report: dict[str, Any] = {
        "protocol": PROTOCOL,
        "valid": True,
        "source_pretraining_passed": alpha is not None,
        "development_passed": False,
        "deep_swe_outcomes_accessed": False,
        "target_outcomes_used": False,
        "trace_source": trace_data.provenance,
        "source_embedding": source_embedding,
        "source_targets": [
            {
                "name": target.name,
                "rows": len(target.indices),
                "pair_specific": target.pair_specific,
            }
            for target in targets
        ],
        "source_alpha_grid": alpha_grid,
        "selected_alpha": alpha,
        "embedding_model_persisted": False,
        "task_embeddings_persisted": False,
        "ridge_weights_persisted": False,
        "knn_bank_persisted": False,
        "fitted_numeric_state_persisted": False,
        "rough_cumulative_spend_usd": 3_025.10805955,
    }
    if alpha is not None:
        data = base.load_data(args.corpus, args.outcomes, args.audit)
        confirmation = base.load_confirmation(args.confirmation_corpus)
        route_texts = data.texts + [base._task_text(task) for task in confirmation]
        route_semantic, route_embedding = semantic._embed(
            route_texts,
            model_path=args.embedding_model,
            tokenizer_path=args.tokenizer,
        )
        interaction, interaction_report = fit_interaction_vectors(
            trace_data,
            targets,
            source_vectors,
            route_semantic,
            alpha=alpha,
        )
        development_vectors = interaction[: len(data.texts)]
        confirmation_vectors = interaction[len(data.texts) :]
        selected, development = semantic._crossfit(data, development_vectors)
        report.update(
            {
                "development_passed": selected is not None,
                "route_embedding": route_embedding,
                "interaction_representation": interaction_report,
                "development": development,
            }
        )
        if selected is not None:
            routes = semantic._freeze_confirmation_routes(
                selected,
                data,
                confirmation,
                development_vectors,
                confirmation_vectors,
            )
            args.routes_out.write_text(
                json.dumps(routes, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    args.report_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--confirmation-corpus", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--routes-out", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
