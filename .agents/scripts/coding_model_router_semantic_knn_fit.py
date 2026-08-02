"""Fit a frozen semantic WMO kNN router on external model by effort outcomes.

Embedding inference and all numeric fitting are intended to run on ephemeral remote compute. The
only durable outputs are an aggregate report and, if development passes, label-free confirmation
route decisions. DeepSWE data is neither accepted nor read by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import coding_model_router_model_effort_fit as base
import numpy as np

from wmo.optimize.knn import fit_knn_policy
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.policy import EmbedderSpec, RoutingPolicy, knn_decision
from wmo.providers.base import ProviderKind
from wmo.providers.pool import PoolEntry

PROTOCOL = "coding-router-semantic-knn-v1"
EMBEDDING_REPOSITORY = "jinaai/jina-embeddings-v2-base-code"
EMBEDDING_REVISION = "516f4baf13dec4ddddda8631e019b5737c8bc250"
EMBEDDING_MODEL_RELATIVE_PATH = "onnx/model_quantized.onnx"
EMBEDDING_MODEL_EXPECTED_BYTES = 161_895_621
EMBEDDING_MODEL_EXPECTED_ETAG = "cdf0fdec74ef1aa8b68d360e29d9a9eee569fea6123cf494604c7e530af27c3f"
MAX_TOKENS = 1_024
BATCH_SIZE = 4
GUARDS = (
    "luna-low",
    "luna-medium",
    "luna-high",
    "luna-xhigh",
    "luna-max",
    "terra-xhigh",
    "terra-max",
    "sol-max",
)
K_VALUES = (8, 16, 32, 64)
Z_VALUES = (0.0, 0.5, 1.0, 1.645, 2.0)
LAM_VALUES = (0.0, 0.01, 0.02, 0.03)
QUALITY_RETENTION = 0.95
MIN_SAVINGS = 0.40
MAX_ROUTE_P95_MS = 5.0

PRICES = {
    "luna": (1.0, 0.1, 6.0),
    "terra": (2.5, 0.25, 15.0),
    "sol": (5.0, 0.5, 30.0),
}


@dataclass(frozen=True)
class Candidate:
    """One frozen semantic kNN policy configuration."""

    order: int
    guard: str
    k: int
    z: float
    pick_lam: float

    @property
    def key(self) -> str:
        return f"guard-{self.guard}-k{self.k}-z{self.z:g}-lam{self.pick_lam:g}"


class CachedEmbedder:
    """Serve precomputed vectors by exact pre-call task text."""

    def __init__(self, texts: list[str], vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[0] != len(texts) or vectors.shape[1] < 1:
            raise ValueError("embedding matrix shape does not match task texts")
        if len(set(texts)) != len(texts):
            raise ValueError("task texts must be unique for exact cached embedding lookup")
        if not np.isfinite(vectors).all():
            raise ValueError("embedding matrix contains non-finite values")
        self.dim = int(vectors.shape[1])
        self._vectors = {
            text: [float(value) for value in vectors[index]]
            for index, text in enumerate(texts)
        }

    def embed(self, texts: list[str]) -> list[list[float]]:
        missing = [text for text in texts if text not in self._vectors]
        if missing:
            raise KeyError(f"{len(missing)} texts are absent from the embedding cache")
        return [self._vectors[text] for text in texts]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_grid() -> tuple[Candidate, ...]:
    """Return the complete preregistered 640-policy grid."""
    values: list[Candidate] = []
    for guard in GUARDS:
        for k in K_VALUES:
            for z in Z_VALUES:
                for pick_lam in LAM_VALUES:
                    values.append(Candidate(len(values), guard, k, z, pick_lam))
    result = tuple(values)
    if len(result) != 640 or len({candidate.key for candidate in result}) != 640:
        raise AssertionError("semantic kNN candidate grid is incomplete or duplicated")
    return result


def _pool() -> list[PoolEntry]:
    result: list[PoolEntry] = []
    for arm in base.ARMS:
        model, effort = arm.split("-", 1)
        prices = PRICES[model]
        result.append(
            PoolEntry(
                name=arm,
                kind=ProviderKind.OPENAI_RESPONSES,
                model=base.MODEL_IDS[model],
                reasoning_effort=effort,
                input_per_mtok=prices[0],
                cached_input_per_mtok=prices[1],
                output_per_mtok=prices[2],
            )
        )
    return result


def _matrix(data: base.Data) -> OutcomeMatrix:
    outcomes = [
        ScenarioOutcome(
            scenario_id=task_id,
            task=data.texts[task_index],
            model=arm,
            benchmark="swerebench-v2-external-development",
            episode=0,
            attempt_number=1,
            reward=float(data.rewards[task_index, arm_index]),
            success=bool(data.rewards[task_index, arm_index] >= 1.0),
            cost_usd=float(data.costs[task_index, arm_index]),
            completion_status="scored",
            usage_accounting="trace-derived",
        )
        for task_index, task_id in enumerate(data.task_ids)
        for arm_index, arm in enumerate(base.ARMS)
    ]
    return OutcomeMatrix(pool=_pool(), outcomes=outcomes)


def _embed(
    texts: list[str],
    *,
    model_path: Path,
    tokenizer_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute normalized code embeddings with the frozen quantized ONNX model."""
    if model_path.stat().st_size != EMBEDDING_MODEL_EXPECTED_BYTES:
        raise ValueError("embedding ONNX content length differs from preregistration")
    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError as error:  # pragma: no cover, installed only in the remote runner
        raise RuntimeError("onnxruntime and tokenizers are required on remote compute") from error

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=MAX_TOKENS)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inputs = {item.name: item for item in session.get_inputs()}
    if "input_ids" not in inputs or "attention_mask" not in inputs:
        raise ValueError(f"unsupported ONNX inputs: {sorted(inputs)}")

    parts: list[np.ndarray] = []
    started = time.perf_counter()
    for start in range(0, len(texts), BATCH_SIZE):
        encoded = tokenizer.encode_batch(texts[start : start + BATCH_SIZE])
        width = max(len(item.ids) for item in encoded)
        input_ids = np.zeros((len(encoded), width), dtype=np.int64)
        attention_mask = np.zeros_like(input_ids)
        token_type_ids = np.zeros_like(input_ids)
        for row, item in enumerate(encoded):
            size = len(item.ids)
            input_ids[row, :size] = np.asarray(item.ids, dtype=np.int64)
            attention_mask[row, :size] = 1
            if item.type_ids:
                token_type_ids[row, :size] = np.asarray(item.type_ids, dtype=np.int64)
        feed: dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in inputs:
            feed["token_type_ids"] = token_type_ids
        output = np.asarray(session.run(None, feed)[0], dtype=np.float32)
        if output.ndim == 3:
            mask = attention_mask[..., None].astype(np.float32)
            output = (output * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1.0)
        if output.ndim != 2 or output.shape[0] != len(encoded):
            raise ValueError(f"unexpected ONNX output shape {output.shape}")
        norms = np.linalg.norm(output, axis=1, keepdims=True)
        parts.append(output / np.maximum(norms, np.finfo(np.float32).eps))
    vectors = np.concatenate(parts, axis=0)
    if vectors.shape[0] != len(texts) or not np.isfinite(vectors).all():
        raise ValueError("embedding inference produced an invalid matrix")
    return vectors, {
        "tasks": len(texts),
        "dimension": int(vectors.shape[1]),
        "max_tokens": MAX_TOKENS,
        "batch_size": BATCH_SIZE,
        "seconds": time.perf_counter() - started,
        "onnx_inputs": sorted(inputs),
        "onnx_output_shape": list(vectors.shape),
        "model_sha256": _sha256(model_path),
        "tokenizer_sha256": _sha256(tokenizer_path),
    }


def _metrics(data: base.Data, choices: np.ndarray) -> dict[str, Any]:
    rows = np.arange(len(choices))
    reward = float(np.mean(data.rewards[rows, choices]))
    cost = float(np.mean(data.costs[rows, choices]))
    static_rewards = data.rewards.mean(axis=0)
    static_costs = data.costs.mean(axis=0)
    baseline = base.ARMS.index("sol-max")
    traffic = np.bincount(choices, minlength=len(base.ARMS)).astype(np.float64) / len(choices)
    blind_reward = float(traffic @ static_rewards)
    blind_cost = float(traffic @ static_costs)
    dominated = [
        base.ARMS[arm]
        for arm in range(len(base.ARMS))
        if static_rewards[arm] >= reward
        and static_costs[arm] <= cost
        and (static_rewards[arm] > reward or static_costs[arm] < cost)
    ]
    return {
        "reward": reward,
        "cost_usd_per_task": cost,
        "quality_retention": reward / float(static_rewards[baseline]),
        "cost_savings": 1.0 - cost / float(static_costs[baseline]),
        "matched_blind_reward": blind_reward,
        "matched_blind_cost_usd_per_task": blind_cost,
        "matched_blind_advantage": reward - blind_reward,
        "model_mix": {
            base.ARMS[index]: float(share)
            for index, share in enumerate(traffic)
            if share > 0.0
        },
        "dominated_by_static": dominated,
    }


def _tune(policy: RoutingPolicy, candidate: Candidate) -> RoutingPolicy:
    return policy.model_copy(
        update={
            "default_model": candidate.guard,
            "guard_model": candidate.guard,
            "rag_num": candidate.k,
            "rag_thres": 0.95,
            "floor_q": 0.0,
            "floor_sim": None,
            "knn_z": candidate.z,
            "knn_min_pairs": 8,
            "se_floor": True,
            "guard_mode": "asymmetric",
            "pick_lam": candidate.pick_lam,
        }
    )


def _crossfit(
    data: base.Data,
    vectors: np.ndarray,
) -> tuple[Candidate | None, dict[str, Any]]:
    matrix = _matrix(data)
    embedder = CachedEmbedder(data.texts, vectors)
    spec = EmbedderSpec(kind="hashing", dim=embedder.dim)
    candidates = candidate_grid()
    by_base: dict[tuple[str, int], list[Candidate]] = {}
    for candidate in candidates:
        by_base.setdefault((candidate.guard, candidate.k), []).append(candidate)
    routes = {
        (seed, candidate.key): np.full(len(data.task_ids), -1, dtype=np.int16)
        for seed in base.SEEDS
        for candidate in candidates
    }
    timings_ms: list[float] = []

    with tempfile.TemporaryDirectory(prefix="wmo-semantic-knn-") as directory:
        root = Path(directory)
        for seed in base.SEEDS:
            folds = base.grouped_folds(data.repositories, seed)
            for fold in range(base.FOLDS):
                train = np.flatnonzero(folds != fold)
                heldout = np.flatnonzero(folds == fold)
                train_ids = [data.task_ids[index] for index in train]
                for (guard, k), variants in by_base.items():
                    policy = fit_knn_policy(
                        matrix,
                        bank_path=root / f"seed-{seed}-fold-{fold}-{guard}-k{k}.npz",
                        fit_ids=train_ids,
                        embedder=spec,
                        embed_with=embedder,
                        guard_model=guard,
                        rag_num=k,
                        rag_thres=0.95,
                        z=0.0,
                        min_pairs=8,
                        se_floor=True,
                        floor_q=0.0,
                        pick_lam=0.0,
                        fitted_from=f"{PROTOCOL} repo-grouped external development",
                    )
                    for candidate in variants:
                        tuned = _tune(policy, candidate)
                        selected = routes[(seed, candidate.key)]
                        for index in heldout:
                            started = time.perf_counter_ns()
                            decision = knn_decision(tuned, vectors[index])
                            timings_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
                            selected[index] = base.ARMS.index(decision.model)

    eligible: list[tuple[float, float, int, Candidate, list[dict[str, Any]]]] = []
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        seed_metrics: list[dict[str, Any]] = []
        for seed in base.SEEDS:
            choices = routes[(seed, candidate.key)]
            if np.any(choices < 0):
                raise ValueError(f"crossfit left unfilled routes for {candidate.key}, seed {seed}")
            seed_metrics.append(_metrics(data, choices))
        passed = (
            all(float(item["quality_retention"]) >= QUALITY_RETENTION for item in seed_metrics)
            and all(float(item["cost_savings"]) >= MIN_SAVINGS for item in seed_metrics)
            and float(np.mean([float(item["matched_blind_advantage"]) for item in seed_metrics]))
            > 0.0
            and all(not item["dominated_by_static"] for item in seed_metrics)
        )
        mean_cost = float(np.mean([float(item["cost_usd_per_task"]) for item in seed_metrics]))
        mean_reward = float(np.mean([float(item["reward"]) for item in seed_metrics]))
        row = {
            "candidate": candidate.key,
            "configuration": asdict(candidate),
            "eligible": passed,
            "mean_reward": mean_reward,
            "mean_cost_usd_per_task": mean_cost,
            "mean_matched_blind_advantage": float(
                np.mean([float(item["matched_blind_advantage"]) for item in seed_metrics])
            ),
            "seeds": [
                {"seed": seed, **metrics}
                for seed, metrics in zip(base.SEEDS, seed_metrics, strict=True)
            ],
        }
        rows.append(row)
        if passed:
            eligible.append((mean_cost, -mean_reward, candidate.order, candidate, seed_metrics))

    selected: Candidate | None = None
    selected_row: dict[str, Any] | None = None
    if eligible:
        selected = min(eligible)[3]
        selected_row = next(row for row in rows if row["candidate"] == selected.key)
    near = sorted(
        rows,
        key=lambda row: (
            max(0.0, QUALITY_RETENTION - min(float(x["quality_retention"]) for x in row["seeds"])),
            max(0.0, MIN_SAVINGS - min(float(x["cost_savings"]) for x in row["seeds"])),
            -float(row["mean_matched_blind_advantage"]),
            float(row["mean_cost_usd_per_task"]),
        ),
    )[:20]
    return selected, {
        "candidate_count": len(candidates),
        "eligible_count": len(eligible),
        "selected": selected_row,
        "closest_candidates": near,
        "route_decision_latency_ms": {
            "samples": len(timings_ms),
            "p50": float(np.quantile(timings_ms, 0.50)),
            "p95": float(np.quantile(timings_ms, 0.95)),
            "maximum": float(np.max(timings_ms)),
            "gate_ms": MAX_ROUTE_P95_MS,
        },
    }


def _freeze_confirmation_routes(
    selected: Candidate,
    data: base.Data,
    confirmation: list[dict[str, Any]],
    development_vectors: np.ndarray,
    confirmation_vectors: np.ndarray,
) -> dict[str, Any]:
    matrix = _matrix(data)
    all_texts = data.texts + [base._task_text(task) for task in confirmation]
    embedder = CachedEmbedder(
        all_texts,
        np.concatenate([development_vectors, confirmation_vectors], axis=0),
    )
    spec = EmbedderSpec(kind="hashing", dim=embedder.dim)
    with tempfile.TemporaryDirectory(prefix="wmo-semantic-knn-final-") as directory:
        policy = fit_knn_policy(
            matrix,
            bank_path=Path(directory) / "bank.npz",
            fit_ids=data.task_ids,
            embedder=spec,
            embed_with=embedder,
            guard_model=selected.guard,
            rag_num=selected.k,
            rag_thres=0.95,
            z=0.0,
            min_pairs=8,
            se_floor=True,
            floor_q=0.0,
            pick_lam=0.0,
            fitted_from=f"{PROTOCOL} full external development",
        )
        policy = _tune(policy, selected)
        rows: list[dict[str, Any]] = []
        timings: list[float] = []
        for index, task in enumerate(confirmation):
            started = time.perf_counter_ns()
            decision = knn_decision(policy, confirmation_vectors[index])
            timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
            rows.append(
                {
                    "task_id": str(task["task_id"]),
                    "repository": str(task["repository"]),
                    "arm": decision.model,
                    "reason": decision.reason,
                }
            )
    return {
        "protocol": PROTOCOL,
        "selected_candidate": selected.key,
        "routes": rows,
        "route_decision_latency_ms": {
            "p50": float(np.quantile(timings, 0.50)),
            "p95": float(np.quantile(timings, 0.95)),
            "maximum": float(np.max(timings)),
        },
        "deep_swe_outcomes_accessed": False,
        "fitted_numeric_state_persisted": False,
    }


def run(args: argparse.Namespace) -> int:
    """Run the frozen external development test and conditionally freeze confirmation routes."""
    data = base.load_data(args.corpus, args.outcomes, args.audit)
    confirmation = base.load_confirmation(args.confirmation_corpus)
    texts = data.texts + [base._task_text(task) for task in confirmation]
    vectors, embedding_report = _embed(
        texts,
        model_path=args.embedding_model,
        tokenizer_path=args.tokenizer,
    )
    development_vectors = vectors[: len(data.texts)]
    confirmation_vectors = vectors[len(data.texts) :]
    selected, development = _crossfit(data, development_vectors)
    confirmation_routes: dict[str, Any] | None = None
    if selected is not None:
        confirmation_routes = _freeze_confirmation_routes(
            selected,
            data,
            confirmation,
            development_vectors,
            confirmation_vectors,
        )
        args.routes_out.write_text(
            json.dumps(confirmation_routes, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    report = {
        "protocol": PROTOCOL,
        "valid": True,
        "development_passed": selected is not None,
        "deep_swe_outcomes_accessed": False,
        "target_outcomes_used": False,
        "fitted_numeric_state_persisted": False,
        "embedding_model_persisted": False,
        "task_embeddings_persisted": False,
        "knn_bank_persisted": False,
        "embedding": {
            "repository": EMBEDDING_REPOSITORY,
            "revision": EMBEDDING_REVISION,
            "relative_path": EMBEDDING_MODEL_RELATIVE_PATH,
            "expected_bytes": EMBEDDING_MODEL_EXPECTED_BYTES,
            "expected_etag": EMBEDDING_MODEL_EXPECTED_ETAG,
            **embedding_report,
        },
        "development": development,
        "confirmation_routes_frozen": confirmation_routes is not None,
        "rough_cumulative_spend_usd": 3_025.10805955,
    }
    args.report_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
