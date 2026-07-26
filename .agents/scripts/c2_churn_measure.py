"""C2 Q2 stage 2: routing-decision churn when the served router sees compressed text.

Consumes stage 1's variants (c2_churn_variants.py), embeds them with the same
text-embedding-3-large call the routing caches were built with (same [:18000]
truncation, order-stable npy caches), fits the served champion per split seed on RAW
fit-side embeddings (guard fable-5, z=0.5, floor_q=0.05, the #259 operating point),
and measures per (corpus, variant, seed) over the TEST split:

- churn: fraction of requests whose routed model changes vs the raw-text decision
- direction: of churned requests, how many move to a cheaper vs pricier model
  (fit-split mean cost per model), and how many enter/leave the fallback
- floor trips: novelty-floor abstain rate raw vs compressed (the floor quantile was
  calibrated on raw self-similarities, so compressed queries may trip it more)
- route-away rate raw vs compressed
- bank-refit variant: the bank itself refit on compressed fit texts, compressed
  queries against it, churn measured vs the raw-raw decisions

Run from the MAIN checkout (needs wmh at #259+; feat/compression-track predates the
knn policy):

    cd ~/Desktop/Projects/world-model-harness && \
        uv run python ~/Desktop/Projects/wmh-compress-c2/.agents/scripts/c2_churn_measure.py

Reads OPENAI_API_KEY from the environment or the repo .env.
"""

from __future__ import annotations

import json
import logging
import os
import random
import statistics
import tempfile
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

from wmh.optimize.knn import fit_knn_policy
from wmh.optimize.outcomes import OutcomeMatrix
from wmh.optimize.policy import KNN_BANK_FILENAME, EmbedderSpec, knn_decision

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("c2_churn_measure")

ROUTING_DATA = Path.home() / "Desktop/Projects/wmh-routing-data"
DATA_ROOT = Path.home() / "Desktop/Projects/wmh-compression-data"
VARIANTS_DIR = DATA_ROOT / "cache/churn-variants"
EMB_DIR = DATA_ROOT / "cache/churn-emb"
RESULTS_PATH = DATA_ROOT / "cache/churn-results.json"
CORPORA = ["routerbench-ours9", "financebench-s80", "tau-bench-s80"]
SEEDS = [0, 1, 2, 3, 4]
FALLBACK = "fable-5"
FLOOR_Q = 0.05


def api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        env = Path.home() / "Desktop/Projects/world-model-harness/.env"
        for line in env.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        raise ValueError("OPENAI_API_KEY not found")
    return key


def openai_embed(texts: list[str], cache_path: Path) -> np.ndarray:
    """R1's embedding call, verbatim conventions: batches of 128, [:18000] truncation."""
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(texts):
            return cached
    key = api_key()
    vectors: list[list[float]] = []
    total_tokens = 0
    for start in range(0, len(texts), 128):
        chunk = [t[:18000] if t.strip() else " " for t in texts[start : start + 128]]
        payload = json.dumps({"model": "text-embedding-3-large", "input": chunk}).encode()
        request = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=payload,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    data = json.loads(response.read())
                break
            except Exception as error:  # noqa: BLE001 - retry transients, then raise
                if attempt == 3:
                    raise
                log.warning("embed batch %d retry %d: %s", start, attempt + 1, error)
                time.sleep(5 * (attempt + 1))
        vectors.extend(d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"]))
        total_tokens += data.get("usage", {}).get("total_tokens", 0)
    array = np.asarray(vectors, dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, array)
    log.info("embedded %d texts, %d tokens (~$%.3f) -> %s",
             len(texts), total_tokens, total_tokens / 1e6 * 0.13, cache_path.name)
    return array


def stratified_split(matrix: OutcomeMatrix, *, train_fraction: float = 0.7, seed: int = 0):
    """The routing protocol's split (validate_knn_promotion.py's reimplementation)."""
    by_eval: dict[str, list[str]] = {}
    for scenario_id in matrix.scenario_ids():
        prefix = scenario_id.split(":", 1)[0] if ":" in scenario_id else ""
        by_eval.setdefault(prefix, []).append(scenario_id)
    rng = random.Random(seed)
    fit: list[str] = []
    test: list[str] = []
    for _name, ids in sorted(by_eval.items()):
        shuffled = ids[:]
        rng.shuffle(shuffled)
        if len(shuffled) > 1:
            cut = min(max(1, round(len(shuffled) * train_fraction)), len(shuffled) - 1)
        else:
            cut = 1
        fit.extend(shuffled[:cut])
        test.extend(shuffled[cut:])
    return sorted(fit), sorted(test)


class VectorEmbedder:
    """Serves precomputed vectors keyed by the RAW task text (fit-time interface)."""

    def __init__(self, raw_task_by_sid: dict[str, str], vec_by_sid: dict[str, np.ndarray]):
        self._by_text = {raw_task_by_sid[sid]: vec for sid, vec in vec_by_sid.items()}

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._by_text[t].tolist() for t in texts]


def scenario_order(matrix: OutcomeMatrix) -> tuple[list[str], dict[str, str]]:
    order: list[str] = []
    tasks: dict[str, str] = {}
    for outcome in matrix.outcomes:
        if outcome.scenario_id not in tasks:
            tasks[outcome.scenario_id] = outcome.task
            order.append(outcome.scenario_id)
    return order, tasks


def model_costs_on_fit(matrix: OutcomeMatrix, fit_ids: list[str]) -> dict[str, float]:
    sums: dict[str, tuple[float, int]] = {}
    wanted = set(fit_ids)
    for outcome in matrix.outcomes:
        if outcome.scenario_id in wanted and outcome.reward is not None:
            total, count = sums.get(outcome.model, (0.0, 0))
            sums[outcome.model] = (total + outcome.cost_usd, count + 1)
    return {m: t / c for m, (t, c) in sums.items()}


def decide_all(policy, vectors: dict[str, np.ndarray], ids: list[str]):  # noqa: ANN001
    out: dict[str, tuple[str, bool]] = {}
    for sid in ids:
        decision = knn_decision(policy, vectors[sid])
        out[sid] = (decision.model, decision.reason.startswith("knn novelty abstain"))
    return out


def cell_tables(matrix: OutcomeMatrix):
    """Per-(scenario, model) mean reward and cost: the outcome-matrix lookup tables."""
    sums: dict[tuple[str, str], tuple[float, float, int]] = {}
    for outcome in matrix.outcomes:
        if outcome.reward is None:
            continue
        key = (outcome.scenario_id, outcome.model)
        reward, cost, count = sums.get(key, (0.0, 0.0, 0))
        sums[key] = (reward + outcome.reward, cost + outcome.cost_usd, count + 1)
    return {k: (r / c_n, c / c_n) for k, (r, c, c_n) in sums.items()}


def decision_value(cells, decisions, ids):  # noqa: ANN001
    """ROUTING-CHANNEL value of a decision map: reward/cost of the picked model per
    scenario, by matrix lookup. Execution stays UNCOMPRESSED (the matrix episodes ran
    on raw text), so this isolates what compression does to the routing decision; the
    execution-channel delta needs the live grid."""
    rewards = [cells[(sid, decisions[sid][0])][0] for sid in ids if (sid, decisions[sid][0]) in cells]
    costs = [cells[(sid, decisions[sid][0])][1] for sid in ids if (sid, decisions[sid][0]) in cells]
    return statistics.mean(rewards), statistics.mean(costs)


def main() -> None:
    results: list[dict] = []
    spec = EmbedderSpec(
        kind="azure", dim=3072, deployment="text-embedding-3-large", endpoint="https://x"
    )
    for corpus in CORPORA:
        matrix = OutcomeMatrix.load(ROUTING_DATA / "matrices" / f"{corpus}_matrix.json")
        order, raw_tasks = scenario_order(matrix)
        raw_vectors = np.load(ROUTING_DATA / "cache" / f"{corpus}-oai3l-tasks.npy").astype(
            np.float32
        )
        assert raw_vectors.shape[0] == len(order), f"{corpus}: cache misaligned"
        raw_by_sid = {sid: raw_vectors[i] for i, sid in enumerate(order)}

        variants: dict[str, dict[str, str]] = {}
        with (VARIANTS_DIR / f"{corpus}.jsonl").open() as handle:
            for line in handle:
                row = json.loads(line)
                variants.setdefault(row["variant"], {})[row["scenario_id"]] = row["text"]
        ratios = json.loads((VARIANTS_DIR / f"{corpus}-ratios.json").read_text())

        variant_vecs: dict[str, dict[str, np.ndarray]] = {}
        for variant, texts_by_sid in sorted(variants.items()):
            safe = variant.replace(":", "_")
            arr = openai_embed(
                [texts_by_sid[sid] for sid in order], EMB_DIR / f"{corpus}-{safe}.npy"
            )
            variant_vecs[variant] = {sid: arr[i] for i, sid in enumerate(order)}

        cells = cell_tables(matrix)
        for seed in SEEDS:
            fit_ids, test_ids = stratified_split(matrix, seed=seed)
            costs = model_costs_on_fit(matrix, fit_ids)
            with tempfile.TemporaryDirectory() as directory:
                policy = fit_knn_policy(
                    matrix,
                    bank_path=Path(directory) / KNN_BANK_FILENAME,
                    fit_ids=fit_ids,
                    embedder=spec,
                    embed_with=VectorEmbedder(raw_tasks, raw_by_sid),
                    guard_model=FALLBACK,
                    floor_q=FLOOR_Q,
                    fitted_from=f"{corpus} seed={seed} raw",
                )
                raw_decisions = decide_all(policy, raw_by_sid, test_ids)
                raw_reward, raw_cost = decision_value(cells, raw_decisions, test_ids)

                for variant in sorted(variants):
                    comp_decisions = decide_all(policy, variant_vecs[variant], test_ids)
                    comp_reward, comp_cost = decision_value(cells, comp_decisions, test_ids)
                    churned = [
                        sid for sid in test_ids
                        if comp_decisions[sid][0] != raw_decisions[sid][0]
                    ]
                    to_cheaper = sum(
                        1 for sid in churned
                        if costs.get(comp_decisions[sid][0], 0.0)
                        < costs.get(raw_decisions[sid][0], 0.0)
                    )
                    results.append(
                        {
                            "corpus": corpus,
                            "variant": variant,
                            "seed": seed,
                            "mode": "query-only",
                            "n_test": len(test_ids),
                            "achieved_ratio": ratios[variant],
                            "churn": len(churned) / len(test_ids),
                            "to_cheaper": to_cheaper,
                            "to_pricier": len(churned) - to_cheaper,
                            "raw_route_away": statistics.mean(
                                raw_decisions[sid][0] != FALLBACK for sid in test_ids
                            ),
                            "comp_route_away": statistics.mean(
                                comp_decisions[sid][0] != FALLBACK for sid in test_ids
                            ),
                            "raw_floor_trips": sum(
                                raw_decisions[sid][1] for sid in test_ids
                            ),
                            "comp_floor_trips": sum(
                                comp_decisions[sid][1] for sid in test_ids
                            ),
                            "comp_pick_counts": dict(
                                Counter(comp_decisions[sid][0] for sid in test_ids)
                            ),
                            "raw_reward": raw_reward,
                            "comp_reward": comp_reward,
                            "raw_cost": raw_cost,
                            "comp_cost": comp_cost,
                        }
                    )

                # Bank-refit leg: the bank itself fit on compressed fit texts.
                for variant in sorted(variants):
                    with tempfile.TemporaryDirectory() as refit_dir:
                        refit_policy = fit_knn_policy(
                            matrix,
                            bank_path=Path(refit_dir) / KNN_BANK_FILENAME,
                            fit_ids=fit_ids,
                            embedder=spec,
                            embed_with=VectorEmbedder(raw_tasks, variant_vecs[variant]),
                            guard_model=FALLBACK,
                            floor_q=FLOOR_Q,
                            fitted_from=f"{corpus} seed={seed} bank-refit {variant}",
                        )
                        refit_decisions = decide_all(
                            refit_policy, variant_vecs[variant], test_ids
                        )
                    refit_reward, refit_cost = decision_value(cells, refit_decisions, test_ids)
                    churned = [
                        sid for sid in test_ids
                        if refit_decisions[sid][0] != raw_decisions[sid][0]
                    ]
                    results.append(
                        {
                            "corpus": corpus,
                            "variant": variant,
                            "seed": seed,
                            "mode": "bank-refit",
                            "n_test": len(test_ids),
                            "achieved_ratio": ratios[variant],
                            "churn": len(churned) / len(test_ids),
                            "comp_route_away": statistics.mean(
                                refit_decisions[sid][0] != FALLBACK for sid in test_ids
                            ),
                            "comp_floor_trips": sum(
                                refit_decisions[sid][1] for sid in test_ids
                            ),
                            "raw_reward": raw_reward,
                            "comp_reward": refit_reward,
                            "raw_cost": raw_cost,
                            "comp_cost": refit_cost,
                        }
                    )
            log.info("%s seed %d done", corpus, seed)

    RESULTS_PATH.write_text(json.dumps(results, indent=1))
    log.info("wrote %d rows -> %s", len(results), RESULTS_PATH)

    # Seed-aggregated summary table. Value columns are the ROUTING-CHANNEL deltas
    # (matrix lookup, execution uncompressed): paired per-seed comp minus raw.
    keys = sorted({(r["corpus"], r["variant"], r["mode"]) for r in results})
    print(f"\n{'corpus':<18} {'variant':<34} {'mode':<11} {'ratio':>5} "
          f"{'churn mean+-sd':>15} {'away r->c':>11} {'floor r->c':>10} "
          f"{'d_reward (wins)':>16} {'d_cost%':>8}")
    for corpus, variant, mode in keys:
        rows = [
            r for r in results
            if (r["corpus"], r["variant"], r["mode"]) == (corpus, variant, mode)
        ]
        churns = [r["churn"] for r in rows]
        away = ""
        if mode == "query-only":
            away = (f"{statistics.mean(r['raw_route_away'] for r in rows):.2f}->"
                    f"{statistics.mean(r['comp_route_away'] for r in rows):.2f}")
        floors = (f"{sum(r.get('raw_floor_trips', 0) for r in rows)}->"
                  f"{sum(r['comp_floor_trips'] for r in rows)}")
        d_reward = [r["comp_reward"] - r["raw_reward"] for r in rows]
        wins = sum(1 for d in d_reward if d >= 0)
        d_cost = statistics.mean(r["comp_cost"] / r["raw_cost"] - 1.0 for r in rows)
        print(f"{corpus:<18} {variant:<34} {mode:<11} {rows[0]['achieved_ratio']:>5.2f} "
              f"{statistics.mean(churns):>7.3f}+-{statistics.stdev(churns):<6.3f} "
              f"{away:>11} {floors:>10} "
              f"{statistics.mean(d_reward):>+8.4f} ({wins}/5) {d_cost:>+7.1%}")


if __name__ == "__main__":
    main()
