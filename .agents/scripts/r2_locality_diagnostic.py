"""The 2505.12601 locality diagnostic: are model win-rates locally smooth in embedding space?

"Simple kNN Beats Complex Learned Routers" (arXiv 2505.12601) argues neighborhood routers work
exactly when per-model correctness varies smoothly over the query embedding space. This script
measures that premise directly, per (matrix x embedder), with three leave-one-out numbers:

- r2 (LOO kNN R^2): 1 - MSE(loo-knn prediction, actual) / Var(actual), averaged over models.
  Positive = neighbors predict a model's reward better than the global mean does.
- top1: fraction of scenarios where the LOO-kNN-predicted best model IS the actual best
  (ties toward cheaper), i.e. smoothness where routing needs it.
- regret: mean(actual reward of oracle pick - actual reward of predicted pick): the routing
  cost of trusting the neighborhood.

Each metric is reported next to its shuffled-label floor (labels permuted within model,
seed 0): smoothness is only real if it clears the floor. Embedders: hashing-1024 (the
serving default) and text-embedding-3-large (r1's cache for ours9; new matrices embed once,
cached to the routing corpus cache/, ~$0.02 total, budget line in findings/r2.md).

Usage: uv run python .agents/scripts/r2_locality_diagnostic.py [matrix ...] [--k=10]
Writes findings/r2_locality.json and logs a table.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from wmo.optimize.outcomes import OutcomeMatrix
from wmo.retrieval.embedders import HashingEmbedder
from wmo.research.routing_corpus import routing_data

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("r2loc")

DATA = routing_data()
DIM = 1024
K = 10


def _openai_embed(texts: list[str], cache_path: Path) -> np.ndarray:
    """text-embedding-3-large with a disk cache, same convention as r1's scripts."""
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(texts):
            return cached
        logger.warning(
            "cache %s rows=%d, need %d; re-embedding", cache_path, cached.shape[0], len(texts)
        )
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("text-embedding-3-large needs OPENAI_API_KEY in the environment")
    vectors: list[list[float]] = []
    for start in range(0, len(texts), 128):
        chunk = [t[:18000] if t.strip() else " " for t in texts[start : start + 128]]
        payload = json.dumps({"model": "text-embedding-3-large", "input": chunk}).encode()
        request = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        data = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    data = json.loads(response.read())
                break
            except Exception:  # noqa: BLE001 (retry transients, then raise)
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
        assert data is not None
        vectors.extend(item["embedding"] for item in data["data"])
    out = np.asarray(vectors, dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, out)
    return out


def _matrices() -> dict[str, OutcomeMatrix]:
    out = {}
    for path in sorted((DATA / "matrices").glob("*_matrix.json")):
        name = path.stem.removesuffix("_matrix")
        matrix = OutcomeMatrix.load(path)
        if len(matrix.scenario_ids()) >= 20:
            out[name] = matrix
    return out


def _norm(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms > 0, norms, 1.0)


def diagnose(
    matrix: OutcomeMatrix,
    vecs: np.ndarray,
    sids: list[str],
    k: int,
    *,
    shuffle_seed: int | None = None,
) -> dict:
    """LOO-kNN smoothness metrics over one embedded matrix (see module docstring)."""
    models = [entry.name for entry in matrix.pool]
    col = {name: c for c, name in enumerate(models)}
    row = {sid: r for r, sid in enumerate(sids)}
    rewards = np.full((len(sids), len(models)), np.nan)
    costs = np.full((len(sids), len(models)), np.nan)
    cells: dict[tuple[int, int], list[float]] = {}
    cell_costs: dict[tuple[int, int], list[float]] = {}
    for o in matrix.outcomes:
        if o.reward is None or o.scenario_id not in row:
            continue
        key = (row[o.scenario_id], col[o.model])
        cells.setdefault(key, []).append(o.reward)
        cell_costs.setdefault(key, []).append(o.cost_usd)
    for (r, c), vals in cells.items():
        rewards[r, c] = sum(vals) / len(vals)
        costs[r, c] = sum(cell_costs[(r, c)]) / len(vals)

    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        for c in range(len(models)):
            scored = [r for r in range(len(sids)) if not np.isnan(rewards[r, c])]
            perm = scored[:]
            rng.shuffle(perm)
            rewards[scored, c] = rewards[perm, c]

    sims = vecs @ vecs.T
    np.fill_diagonal(sims, -np.inf)  # leave-one-out
    neighbor_idx = np.argsort(-sims, axis=1)[:, :k]  # [N, k]

    known = ~np.isnan(rewards)
    filled = np.where(known, rewards, 0.0)
    pred = np.full_like(rewards, np.nan)
    for r in range(len(sids)):
        nbrs = neighbor_idx[r]
        counts = known[nbrs].sum(axis=0)
        sums = filled[nbrs].sum(axis=0)
        with np.errstate(invalid="ignore"):
            pred[r] = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    r2_per_model: dict[str, float] = {}
    for c in range(len(models)):
        mask = known[:, c] & ~np.isnan(pred[:, c])
        if mask.sum() < 5:
            continue
        actual, guess = rewards[mask, c], pred[mask, c]
        var = float(np.var(actual))
        if var < 1e-9:
            continue  # a constant column carries no smoothness signal either way
        r2_per_model[models[c]] = 1.0 - float(np.mean((actual - guess) ** 2)) / var

    top1_hits, regrets = [], []
    for r in range(len(sids)):
        mask = known[r] & ~np.isnan(pred[r])
        if mask.sum() < 2:
            continue
        cand = np.where(mask)[0]
        pred_best = cand[np.lexsort((costs[r, cand], -pred[r, cand]))][0]
        actual_best = cand[np.lexsort((costs[r, cand], -rewards[r, cand]))][0]
        top1_hits.append(1.0 if rewards[r, pred_best] >= rewards[r, actual_best] - 1e-9 else 0.0)
        regrets.append(float(rewards[r, actual_best] - rewards[r, pred_best]))
    return {
        "r2_mean": float(np.mean(list(r2_per_model.values()))) if r2_per_model else float("nan"),
        "r2_per_model": {name: round(v, 4) for name, v in r2_per_model.items()},
        "top1": float(np.mean(top1_hits)) if top1_hits else float("nan"),
        "regret": float(np.mean(regrets)) if regrets else float("nan"),
        "n": len(sids),
    }


def main() -> None:
    args = sys.argv[1:]
    wanted = [a for a in args if not a.startswith("--")]
    k = K
    for arg in args:
        if arg.startswith("--k="):
            k = int(arg.split("=", 1)[1])
    results: dict[str, dict] = {}
    for name, matrix in _matrices().items():
        if wanted and name not in wanted:
            continue
        tasks: dict[str, str] = {}
        for o in matrix.outcomes:
            tasks.setdefault(o.scenario_id, o.task)
        sids = list(tasks)
        texts = [tasks[sid] for sid in sids]
        embedded = {
            "hashing": _norm(np.asarray(HashingEmbedder(dim=DIM).embed(texts))),
            "oai3l": _norm(
                _openai_embed(texts, DATA / "cache" / f"{name}-oai3l-tasks.npy").astype(np.float64)
            ),
        }
        results[name] = {}
        for embed_name, vecs in embedded.items():
            real = diagnose(matrix, vecs, sids, k)
            floor = diagnose(matrix, vecs, sids, k, shuffle_seed=0)
            results[name][embed_name] = {"real": real, "floor": floor}
            logger.info(
                "%-22s %-8s r2 %+0.3f (floor %+0.3f)  top1 %.3f (floor %.3f)  "
                "regret %.4f (floor %.4f)  n=%d",
                name,
                embed_name,
                real["r2_mean"],
                floor["r2_mean"],
                real["top1"],
                floor["top1"],
                real["regret"],
                floor["regret"],
                real["n"],
            )
    out = DATA / "findings" / "r2_locality.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("results -> %s", out)


if __name__ == "__main__":
    main()
