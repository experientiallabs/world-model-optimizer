"""Validation gate for the kNN policy promotion: reproduce the champion through wmo code.

Chat R1 measured `knn-statz05-oai` with a research script
(`.agents/scripts/r1_retrieval_ablations.py`): +1.04 accuracy points
over the best single model at -27% cost on routerbench-ours9, 5 of 5 split seeds. This script
re-runs that measurement through the PRODUCTION path only: `wmo.optimize.knn.fit_knn_policy`
writes a real .npz sidecar, and `wmo.optimize.routing.evaluate_policy` replays the policy through
the same `knn_decision` serving calls. If the numbers move, the promotion changed the algorithm.

Split identity is proven, not assumed: the stratified 70/30 split is reimplemented here rather
than imported from `wmo.research.routerbench`, so a bug in the research helper cannot hide
itself, and every seed's fit/test
sizes and best-single baseline accuracy are checked against the recorded runs in
`<routing-data>/runs/r1.jsonl`. Query embeddings come from R1's cached text-embedding-3-large
vectors, so no text is re-embedded.

The routing data (`runs/`, `matrices/`, `cache/`) is a multi-GB research corpus that is not in
git. It defaults to the gitignored `.wmo/routing-data/` under the repo root; point
`$WMO_ROUTING_DATA` at it if you keep it elsewhere.

Usage:
    uv run python .agents/scripts/validate_knn_promotion.py            # ours9 gate + curves
    uv run python .agents/scripts/validate_knn_promotion.py --curve-only
"""

from __future__ import annotations

import json
import logging
import random
import statistics
import sys
import tempfile
from pathlib import Path

import numpy as np

from wmo.optimize.knn import best_single_on_fit, fit_knn_policy
from wmo.optimize.outcomes import OutcomeMatrix
from wmo.optimize.policy import KNN_BANK_FILENAME, EmbedderSpec, select_model
from wmo.optimize.routing import evaluate_policy
from wmo.research.routing_corpus import routing_data

logging.basicConfig(level=logging.INFO, format="%(message)s")
FAILURES: list[str] = []

logger = logging.getLogger("validate-knn")

SEEDS = [0, 1, 2, 3, 4]
# The champion's measured result, as recorded in runs/r1.jsonl (variant r1-knn-statz05-oai).
CHAMPION_DELTA = 0.0104
DELTA_TOLERANCE = 0.003


def stratified_split(
    matrix: OutcomeMatrix, *, train_fraction: float = 0.7, seed: int = 0
) -> tuple[list[str], list[str]]:
    """The routing protocol's split, reimplemented (see module docstring on identity checks).

    Stratified by the scenario id's dataset prefix, so no small eval vanishes from either side;
    ids without a prefix share one stratum.
    """
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


class CachedEmbedder:
    """Serves R1's cached text-embedding-3-large vectors by task text (no API calls).

    The cache is row-aligned to the matrix's scenarios in first-appearance order, which is how
    the research script wrote it.
    """

    def __init__(self, matrix: OutcomeMatrix, cache_path: Path) -> None:
        order: list[str] = []
        tasks: dict[str, str] = {}
        for outcome in matrix.outcomes:
            if outcome.scenario_id not in tasks:
                tasks[outcome.scenario_id] = outcome.task
                order.append(outcome.scenario_id)
        vectors = np.load(cache_path)
        if vectors.shape[0] != len(order):
            raise ValueError(
                f"cache {cache_path} has {vectors.shape[0]} rows but the matrix has "
                f"{len(order)} scenarios; it was built for a different matrix"
            )
        self.dim = int(vectors.shape[1])
        self._by_text = {tasks[sid]: vectors[index] for index, sid in enumerate(order)}

    def embed(self, texts: list[str]) -> list[list[float]]:
        missing = [text for text in texts if text not in self._by_text]
        if missing:
            raise KeyError(f"{len(missing)} task texts are not in the embedding cache")
        return [self._by_text[text].tolist() for text in texts]


def baseline_on_test(matrix: OutcomeMatrix, model: str, test_ids: list[str]) -> tuple[float, float]:
    """(mean reward, mean cost) of one model over the test scenarios it was scored on."""
    rewards: list[float] = []
    costs: list[float] = []
    by_cell: dict[str, list[tuple[float, float]]] = {}
    wanted = set(test_ids)
    for outcome in matrix.outcomes:
        if outcome.scenario_id in wanted and outcome.model == model and outcome.reward is not None:
            by_cell.setdefault(outcome.scenario_id, []).append((outcome.reward, outcome.cost_usd))
    for cells in by_cell.values():
        rewards.append(sum(reward for reward, _ in cells) / len(cells))
        costs.append(sum(cost for _, cost in cells) / len(cells))
    return sum(rewards) / len(rewards), sum(costs) / len(costs)


def recorded_runs(variant: str, matrix_name: str) -> dict[int, dict[str, float]]:
    """Per-seed headline numbers R1 recorded for `variant`, keyed by split seed."""
    out: dict[int, dict[str, float]] = {}
    with (routing_data() / "runs" / "r1.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if (
                record["variant"] != variant
                or record["matrix"] != matrix_name
                or "H2H" in record["notes"]
            ):
                continue
            out[record["split_seed"]] = {
                "accuracy": record["result"]["accuracy"],
                "cost": record["result"]["cost_per_call"],
                "baseline_accuracy": record["baselines"]["best_single"]["accuracy"],
                "baseline_cost": record["baselines"]["best_single"]["cost_per_call"],
                "fit": record["fit_scenarios"],
                "test": record["test_scenarios"],
            }
    return out


def ours9_gate(*, se_floor: bool) -> None:
    """The gate: 5 seeds of routerbench-ours9 through fit_knn_policy + evaluate_policy."""
    data = routing_data()
    matrix = OutcomeMatrix.load(data / "matrices" / "routerbench-ours9_matrix.json")
    embedder = CachedEmbedder(matrix, data / "cache" / "routerbench-ours9-oai3l-tasks.npy")
    spec = EmbedderSpec(
        kind="azure", dim=embedder.dim, deployment="text-embedding-3-large", endpoint="https://x"
    )
    reference = recorded_runs("r1-knn-statz05-oai", "routerbench-ours9")
    deltas: list[float] = []
    cost_ratios: list[float] = []

    for seed in SEEDS:
        fit_ids, test_ids = stratified_split(matrix, seed=seed)
        expected = reference[seed]
        if (len(fit_ids), len(test_ids)) != (expected["fit"], expected["test"]):
            raise AssertionError(
                f"seed {seed} split is {len(fit_ids)}/{len(test_ids)}, R1 recorded "
                f"{expected['fit']}/{expected['test']}: not the same split"
            )
        baseline = best_single_on_fit(matrix, fit_ids)
        base_accuracy, base_cost = baseline_on_test(matrix, baseline, test_ids)
        if abs(base_accuracy - expected["baseline_accuracy"]) > 1e-6:
            raise AssertionError(
                f"seed {seed} best-single ({baseline}) scores {base_accuracy:.6f} on test, R1 "
                f"recorded {expected['baseline_accuracy']:.6f}: not the same split or baseline"
            )

        with tempfile.TemporaryDirectory() as directory:
            policy = fit_knn_policy(
                matrix,
                bank_path=Path(directory) / KNN_BANK_FILENAME,
                fit_ids=fit_ids,
                embedder=spec,
                embed_with=embedder,
                se_floor=se_floor,
                fitted_from=f"routerbench-ours9 seed={seed}",
            )
            result = evaluate_policy(policy, matrix, test_ids, embedder=embedder)

        delta = result.accuracy - base_accuracy
        deltas.append(delta)
        cost_ratios.append(result.cost_per_scenario / base_cost - 1.0)
        logger.info(
            "seed %d: acc %.4f vs best-single %s %.4f (delta %+.4f), cost $%.6f (%+0.1f%%) "
            "[R1 recorded acc %.4f delta %+.4f]",
            seed,
            result.accuracy,
            baseline,
            base_accuracy,
            delta,
            result.cost_per_scenario,
            cost_ratios[-1] * 100,
            expected["accuracy"],
            expected["accuracy"] - expected["baseline_accuracy"],
        )

    mean_delta = statistics.mean(deltas)
    wins = sum(1 for delta in deltas if delta > 0)
    logger.info(
        "GATE se_floor=%s: mean delta %+.4f (stdev %.4f, sem %.4f), %d/5 seed wins, "
        "mean cost %+0.1f%% vs champion %+.4f",
        se_floor,
        mean_delta,
        statistics.stdev(deltas),
        statistics.stdev(deltas) / len(deltas) ** 0.5,
        wins,
        statistics.mean(cost_ratios) * 100,
        CHAMPION_DELTA,
    )
    passed = abs(mean_delta - CHAMPION_DELTA) <= DELTA_TOLERANCE and wins == 5
    logger.info(
        "GATE se_floor=%s: %s (needs |delta - %.4f| <= %.3f and 5/5 wins)",
        se_floor,
        "PASS" if passed else "FAIL",
        CHAMPION_DELTA,
        DELTA_TOLERANCE,
    )
    if not passed:
        FAILURES.append(f"ours9 gate se_floor={se_floor}")


def confidence_curve(matrix_name: str, cache_name: str, *, rag_num: int, fallback: str) -> None:
    """Routed-away fraction and quality/cost as the z knob tightens, with `fallback` pinned."""
    data = routing_data()
    matrix = OutcomeMatrix.load(data / "matrices" / f"{matrix_name}_matrix.json")
    embedder = CachedEmbedder(matrix, data / "cache" / cache_name)
    spec = EmbedderSpec(
        kind="azure", dim=embedder.dim, deployment="text-embedding-3-large", endpoint="https://x"
    )
    fit_ids, test_ids = stratified_split(matrix, seed=0)
    base_accuracy, base_cost = baseline_on_test(matrix, fallback, test_ids)
    tasks = {outcome.scenario_id: outcome.task for outcome in matrix.outcomes}
    logger.info(
        "%s (fit %d, test %d, rag_num %d, fallback %s: acc %.4f, cost $%.6f)",
        matrix_name,
        len(fit_ids),
        len(test_ids),
        rag_num,
        fallback,
        base_accuracy,
        base_cost,
    )
    for z in (0.0, 0.25, 0.5, 1.0, 2.0):
        with tempfile.TemporaryDirectory() as directory:
            policy = fit_knn_policy(
                matrix,
                bank_path=Path(directory) / KNN_BANK_FILENAME,
                fit_ids=fit_ids,
                embedder=spec,
                embed_with=embedder,
                guard_model=fallback,
                rag_num=rag_num,
                z=z,
            )
            result = evaluate_policy(policy, matrix, test_ids, embedder=embedder)
            decisions = [
                select_model(policy, tasks[sid], embedder=embedder) for sid in sorted(test_ids)
            ]
        reverts = [d for d in decisions if "reverted to" in d.reason]
        stray = [d.model for d in reverts if d.model != fallback]
        if stray:
            raise AssertionError(f"z={z}: guard reverted to {sorted(set(stray))}, not {fallback}")
        routed = 1.0 - result.model_mix.get(fallback, 0.0)
        logger.info(
            "  z=%-4g routed away %5.1f%% | acc %.4f (%+.4f) | cost $%.6f (%+0.1f%%) | "
            "%d guard reverts, all to %s",
            z,
            routed * 100,
            result.accuracy,
            result.accuracy - base_accuracy,
            result.cost_per_scenario,
            (result.cost_per_scenario / base_cost - 1.0) * 100,
            len(reverts),
            fallback,
        )


def main() -> None:
    if "--curve-only" not in sys.argv:
        logger.info("=== ours9 champion gate (se_floor off: R1's exact configuration) ===")
        ours9_gate(se_floor=False)
        logger.info("=== ours9 champion gate (se_floor on: the production default) ===")
        ours9_gate(se_floor=True)
    logger.info("=== confidence curve: fable-5 pinned as the fallback ===")
    confidence_curve(
        "routerbench-ours9", "routerbench-ours9-oai3l-tasks.npy", rag_num=50, fallback="fable-5"
    )
    # tau-bench is 25 scenarios; the adaptive rule caps the default budget at ceil(17/2) = 9
    # there, so the two curves below contrast the capped default against an explicit rag=5.
    confidence_curve("tau-bench", "wm-tau-bench-oai3l-tasks.npy", rag_num=50, fallback="fable-5")
    confidence_curve("tau-bench", "wm-tau-bench-oai3l-tasks.npy", rag_num=5, fallback="fable-5")


if __name__ == "__main__":
    main()
    if FAILURES:
        logger.error("VALIDATION FAILED: %s", ", ".join(FAILURES))
        sys.exit(1)
